//! The bind-group cache: a dispatch reuses the bind group of an earlier dispatch of the same kernel on the same
//! buffers instead of creating one (3–4 µs of host time each, a third of what a batch-1 dispatch costs).
//!
//! The parameter block is bound with a dynamic offset (its slot in the uniform chunk), so a bind group depends only
//! on `(kernel, uniform chunk, storage buffers)`. An inference forward meets the same combinations on every call and
//! creates no bind group after its first calls, and so does a training step at a fixed shape, since the pool hands
//! out the same buffer for the same request every step (`pool.rs`). Shapes that change (a last partial batch,
//! validation) still miss, so a miss is kept cheap (one hash and a set insert besides the bind group itself), and a
//! combination is cached only on its second sighting ([`BindCache::get`]); one seen once is just remembered.
//!
//! A cached bind group holds its buffers alive, so only buffers the pool owns are cached (and the context's dummy
//! buffer): the pool knows when it lets one of those go (evicted, or dropped by [`crate::GpuContext::trim_pool`])
//! and records its id, and the next dispatch or submission drops every entry that binds it, found through an exact
//! index from buffer to entries, so the driver can free the memory. A buffer outside the pool (a raw buffer passed
//! to a kernel, a trace replaying buffers the pool let go of) is never cached, since nothing would release it.
//! [`crate::GpuContext::release_memory`] empties the cache at once. The cache holds at most
//! [`2 × GENERATION`](GENERATION) bind groups: when the current generation is full it becomes the old one, and the
//! previous old one is dropped; a hit in the old generation moves back to the current one, so what every pass uses
//! stays.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};

/// Bind groups per generation (the cache holds at most twice this many). A folo-n forward uses about 120 and a
/// training step about 1500.
pub(crate) const GENERATION: usize = 4096;

/// An FxHash-style hasher: one rotate, xor and multiply per word, much cheaper than SipHash on the handful of
/// pointers a key hashes. Not DoS-resistant, which does not matter for keys made of our own handles.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl Fx {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.add(u64::from_le_bytes(*w));
        }
        if !rest.is_empty() {
            let mut w = [0; 8];
            w[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(w));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }

    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    fn finish(&self) -> u64 {
        // The multiply mixes upwards only, and handles hash as aligned addresses (low bits zero): rotate the
        // well-mixed high bits down, where hash maps take the bucket index from.
        self.0.rotate_left(26)
    }
}

/// A hash map or set using [`Fx`].
pub(crate) type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
pub(crate) type FxMap<V> = FxHashMap<u64, V>;
pub(crate) type FxSet = HashSet<u64, BuildHasherDefault<Fx>>;

/// A stable id of a wgpu handle (its hash; handles hash by identity). Two live handles collide only by chance,
/// which costs a needless miss or an early drop of an entry, never a wrong bind group: hits compare the handles.
pub(crate) fn id_of(h: &impl Hash) -> u64 {
    let mut s = Fx::default();
    h.hash(&mut s);
    s.finish()
}

struct Entry {
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    buffers: Vec<wgpu::Buffer>,
    /// [`id_of`] of each of `buffers`, for [`BindCache::retire`].
    ids: Vec<u64>,
    bind: wgpu::BindGroup,
}

impl Entry {
    fn matches(
        &self,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        buffers: &[&wgpu::Buffer],
    ) -> bool {
        self.layout == *layout
            && self.uniform == *uniform
            && self.buffers.len() == buffers.len()
            && self.buffers.iter().zip(buffers).all(|(a, b)| a == *b)
    }
}

/// Bind groups by `(layout, uniform chunk, storage buffers)`; see the module docs.
#[derive(Default)]
pub(crate) struct BindCache {
    cur: FxMap<Entry>,
    old: FxMap<Entry>,
    /// The keys of the entries that bind each buffer (by [`id_of`]), so retiring a buffer touches only its own
    /// entries. Exact: a key leaves its buffers' sets whenever its entry leaves the cache, so it holds no more than
    /// the entries do.
    users: FxMap<FxSet>,
    /// Keys missed once and not cached (yet): a second miss caches the key. Cleared when it outgrows
    /// `2 × GENERATION`.
    seen: FxSet,
    /// Set by [`crate::GpuContext::set_bind_cache`]`(false)`: every dispatch creates its bind group (A/B runs).
    pub(crate) off: bool,
}

impl BindCache {
    /// The bind group of `layout` with `uniform` (one [`crate::pipeline::UNIFORM_SLOT`] at a dynamic offset) at
    /// `@binding(0)` and `buffers` at `@binding(1..)`, from the cache or created with `device` (`created` is set
    /// then). A created one is cached if its key missed before (since the last clear of the recent keys) and
    /// `cacheable` accepts `buffers` with their [`id_of`] ids.
    pub(crate) fn get(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        buffers: &[&wgpu::Buffer],
        cacheable: impl FnOnce(&[&wgpu::Buffer], &[u64]) -> bool,
        created: &mut bool,
    ) -> wgpu::BindGroup {
        if self.off {
            *created = true;
            return create(device, layout, uniform, buffers);
        }
        let key = key(layout, uniform, buffers);
        if let Some(e) = self.cur.get(&key)
            && e.matches(layout, uniform, buffers)
        {
            return e.bind.clone();
        }
        if self.old.get(&key).is_some_and(|e| e.matches(layout, uniform, buffers)) {
            let e = self.old.remove(&key).expect("checked");
            let bind = e.bind.clone();
            // The key keeps its index entries: it is still cached.
            self.roll();
            self.cur.insert(key, e);
            return bind;
        }
        *created = true;
        let bind = create(device, layout, uniform, buffers);
        if self.seen.len() >= 2 * GENERATION {
            self.seen.clear();
        }
        if self.seen.insert(key) {
            return bind;
        }
        // Forgotten either way, so a refused key pays the check below on every other miss only.
        self.seen.remove(&key);
        let ids: Vec<u64> = buffers.iter().map(|b| id_of(*b)).collect();
        if !cacheable(buffers, &ids) {
            return bind;
        }
        // A collision: the other entry under this key leaves first (a key is in at most one generation).
        if let Some(e) = self.cur.remove(&key).or_else(|| self.old.remove(&key)) {
            self.unindex(key, &e.ids);
        }
        self.roll();
        for &id in &ids {
            self.users.entry(id).or_default().insert(key);
        }
        self.cur.insert(
            key,
            Entry {
                layout: layout.clone(),
                uniform: uniform.clone(),
                buffers: buffers.iter().map(|&b| b.clone()).collect(),
                ids,
                bind: bind.clone(),
            },
        );
        bind
    }

    /// Makes room for one entry in the current generation: when it is full it becomes the old one, and the
    /// previous old one is dropped.
    fn roll(&mut self) {
        if self.cur.len() < GENERATION {
            return;
        }
        let dropped = std::mem::replace(&mut self.old, std::mem::take(&mut self.cur));
        for (k, e) in dropped {
            self.unindex(k, &e.ids);
        }
    }

    /// Removes `key` from the index sets of the buffers `ids` (its entry left the cache).
    fn unindex(&mut self, key: u64, ids: &[u64]) {
        for id in ids {
            if let Some(keys) = self.users.get_mut(id) {
                keys.remove(&key);
                if keys.is_empty() {
                    self.users.remove(id);
                }
            }
        }
    }

    /// Drops every entry that binds one of the buffers with these [`id_of`] ids.
    pub(crate) fn retire(&mut self, ids: &FxSet) {
        for id in ids {
            let Some(keys) = self.users.remove(id) else { continue };
            for k in keys {
                if let Some(e) = self.cur.remove(&k).or_else(|| self.old.remove(&k)) {
                    self.unindex(k, &e.ids);
                }
            }
        }
    }

    /// Drops every entry.
    pub(crate) fn clear(&mut self) {
        self.cur.clear();
        self.old.clear();
        self.users.clear();
        self.seen.clear();
    }

    /// Bind groups held.
    pub(crate) fn len(&self) -> usize {
        self.cur.len() + self.old.len()
    }

    /// Buffers in the reverse index and the keys it lists in all.
    #[cfg(test)]
    pub(crate) fn index_size(&self) -> (usize, usize) {
        (self.users.len(), self.users.values().map(HashSet::len).sum())
    }

    /// Where the bind group of these handles is: `Some(true)` in the current generation, `Some(false)` in the old
    /// one, `None` not cached.
    #[cfg(test)]
    pub(crate) fn holds(
        &self,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        buffers: &[&wgpu::Buffer],
    ) -> Option<bool> {
        let key = key(layout, uniform, buffers);
        let found = |m: &FxMap<Entry>| m.get(&key).is_some_and(|e| e.matches(layout, uniform, buffers));
        if found(&self.cur) {
            Some(true)
        } else if found(&self.old) {
            Some(false)
        } else {
            None
        }
    }
}

/// The cache key of a bind group (see [`BindCache::get`]).
fn key(layout: &wgpu::BindGroupLayout, uniform: &wgpu::Buffer, buffers: &[&wgpu::Buffer]) -> u64 {
    let mut s = Fx::default();
    layout.hash(&mut s);
    uniform.hash(&mut s);
    buffers.hash(&mut s);
    s.finish()
}

/// A new bind group of `layout` (see [`BindCache::get`]).
pub(super) fn create(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let mut entries = Vec::with_capacity(buffers.len() + 1);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: uniform,
            offset: 0,
            size: wgpu::BufferSize::new(crate::pipeline::UNIFORM_SLOT),
        }),
    });
    for (i, buf) in buffers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: buf.as_entire_binding() });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}
