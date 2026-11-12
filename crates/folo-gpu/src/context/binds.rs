//! The bind-group cache: a dispatch reuses the bind group of an earlier dispatch of the same kernel on the same
//! buffers instead of creating one (3–4 µs of host time each, a third of what a batch-1 dispatch costs).
//!
//! The buffer pool hands out the same buffers in the same order on every forward pass or training step, so the
//! same `(kernel, uniform chunk, storage buffers)` combinations come back each time. The parameter block is bound
//! with a dynamic offset (its slot in the uniform chunk), so it is not part of the key.
//!
//! A cached bind group holds its buffers alive. When the pool lets a buffer go (evicted, or dropped by
//! [`crate::GpuContext::trim_pool`]) it records the buffer's id, and the next dispatch drops every entry that
//! binds it, so the driver can free the memory; [`crate::GpuContext::release_memory`] empties the cache at once.
//! The cache holds at most [`2 × GENERATION`](GENERATION) bind groups: when the current generation is full it
//! becomes the old one, and the previous old one is dropped; a hit in the old generation moves back to the
//! current one, so what every pass uses stays.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Bind groups per generation (the cache holds at most twice this many). A folo-n forward uses about 120 and a
/// training step about 1500.
pub(crate) const GENERATION: usize = 4096;

/// A stable id of a wgpu handle (its hash; handles hash by identity). Two live handles collide only by chance,
/// which costs a needless miss or an early drop of an entry, never a wrong bind group: hits compare the handles.
pub(crate) fn id_of(h: &impl Hash) -> u64 {
    let mut s = DefaultHasher::new();
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
    cur: HashMap<u64, Entry>,
    old: HashMap<u64, Entry>,
}

impl BindCache {
    /// The bind group of `layout` with `uniform` (one [`crate::pipeline::UNIFORM_SLOT`] at a dynamic offset) at
    /// `@binding(0)` and `buffers` at `@binding(1..)`, from the cache or created with `device` (`created` is set
    /// then).
    pub(crate) fn get(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        buffers: &[&wgpu::Buffer],
        created: &mut bool,
    ) -> wgpu::BindGroup {
        let mut s = DefaultHasher::new();
        layout.hash(&mut s);
        uniform.hash(&mut s);
        buffers.hash(&mut s);
        let key = s.finish();
        if let Some(e) = self.cur.get(&key)
            && e.matches(layout, uniform, buffers)
        {
            return e.bind.clone();
        }
        if let Some(e) = self.old.remove(&key)
            && e.matches(layout, uniform, buffers)
        {
            let bind = e.bind.clone();
            self.insert(key, e);
            return bind;
        }
        *created = true;
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
        let bind =
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries });
        self.insert(
            key,
            Entry {
                layout: layout.clone(),
                uniform: uniform.clone(),
                buffers: buffers.iter().map(|&b| b.clone()).collect(),
                ids: buffers.iter().map(|b| id_of(*b)).collect(),
                bind: bind.clone(),
            },
        );
        bind
    }

    fn insert(&mut self, key: u64, e: Entry) {
        if self.cur.len() >= GENERATION {
            self.old = std::mem::take(&mut self.cur);
        }
        self.cur.insert(key, e);
    }

    /// Drops every entry that binds one of the buffers with these [`id_of`] ids.
    pub(crate) fn retire(&mut self, ids: &HashSet<u64>) {
        if ids.is_empty() {
            return;
        }
        for map in [&mut self.cur, &mut self.old] {
            map.retain(|_, e| !e.ids.iter().any(|i| ids.contains(i)));
        }
    }

    /// Drops every entry.
    pub(crate) fn clear(&mut self) {
        self.cur.clear();
        self.old.clear();
    }

    /// Bind groups held.
    pub(crate) fn len(&self) -> usize {
        self.cur.len() + self.old.len()
    }
}
