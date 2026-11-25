//! Byte-bounded caches that drop what recent frames did not use: the renderer's own (shadows, paths,
//! resampled images, glyph and gradient tables) and `MaskCache`, the window's glyph / icon masks.

use super::hash::FastBuild;
use crate::{AlphaMask, Bounds, ColorMask, DevicePixels};
use anyhow::Result;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

/// How often long-TTL caches (`max_idle >= IDLE_SWEEP_INTERVAL`) scan for idle entries when under
/// their byte budget. Sweeping every 64 frames (~1 s at 60 Hz) avoids scanning 7 hash maps on every
/// steady frame while still dropping stale entries within `max_idle + 64` frames.
const IDLE_SWEEP_INTERVAL: u64 = 64;

struct Entry<V> {
    value: V,
    bytes: usize,
    used: u64,
}

/// A map whose entries cost bytes. `end_frame` evicts the least recently used entries that the
/// current frame did not use while the total is over the budget, and drops entries unused for more
/// than `max_idle` frames (checked every frame for short limits `< 64`, every 64 frames or when
/// over budget for longer limits).
pub(crate) struct ByteCache<K, V> {
    map: HashMap<K, Entry<V>, FastBuild>,
    bytes: usize,
    budget: usize,
    max_idle: u64,
    frame: u64,
}

impl<K: Hash + Eq + Clone, V: Clone> ByteCache<K, V> {
    pub fn new(budget: usize, max_idle: u64) -> Self {
        ByteCache { map: HashMap::default(), bytes: 0, budget, max_idle, frame: 0 }
    }

    /// The value for `key`, marking it used this frame.
    pub fn get(&mut self, key: &K) -> Option<V> {
        self.get_by(key)
    }

    /// The value for a borrowed `key`, marking it used this frame.
    pub fn get_by<Q: Hash + Eq + ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        let frame = self.frame;
        self.map.get_mut(key).map(|e| {
            e.used = frame;
            e.value.clone()
        })
    }

    pub fn insert(&mut self, key: K, value: V, bytes: usize) {
        let e = Entry { value, bytes, used: self.frame };
        if let Some(old) = self.map.insert(key, e) {
            self.bytes -= old.bytes;
        }
        self.bytes += bytes;
    }

    /// The value for `key`, made by `make` (which returns it and its size) when absent.
    pub fn get_or_insert_with(&mut self, key: &K, make: impl FnOnce() -> (V, usize)) -> V {
        if let Some(v) = self.get(key) {
            return v;
        }
        let (v, bytes) = make();
        self.insert(key.clone(), v.clone(), bytes);
        v
    }

    /// Looks up `key` in a single hash probe, marking it used this frame and returning a reference
    /// without cloning `V`. On a miss, `make` receives `&K` and produces `Some((value, bytes))` to
    /// insert and borrow.
    pub fn get_or_try_insert_ref<E>(
        &mut self,
        key: K,
        make: impl FnOnce(&K) -> std::result::Result<Option<(V, usize)>, E>,
    ) -> std::result::Result<Option<&V>, E> {
        let frame = self.frame;
        match self.map.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let entry = entry.into_mut();
                entry.used = frame;
                Ok(Some(&entry.value))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let Some((value, bytes)) = make(entry.key())? else {
                    return Ok(None);
                };
                self.bytes += bytes;
                Ok(Some(&entry.insert(Entry { value, bytes, used: frame }).value))
            }
        }
    }

    /// Ends a frame: evicts as described on the type, then starts the next frame.
    pub fn end_frame(&mut self) {
        let frame = self.frame;
        let over_budget = self.bytes > self.budget;
        let sweep_idle = self.max_idle != u64::MAX
            && !self.map.is_empty()
            && frame > self.max_idle
            && (over_budget
                || self.max_idle < IDLE_SWEEP_INTERVAL
                || (frame & (IDLE_SWEEP_INTERVAL - 1)) == 0);
        if sweep_idle {
            let max_idle = self.max_idle;
            let mut freed = 0;
            self.map.retain(|_, e| {
                let keep = frame - e.used <= max_idle;
                if !keep {
                    freed += e.bytes;
                }
                keep
            });
            self.bytes -= freed;
        }
        if self.bytes > self.budget {
            let mut old: Vec<(u64, K)> =
                self.map.iter().filter(|(_, e)| e.used < frame).map(|(k, e)| (e.used, k.clone())).collect();
            old.sort_unstable_by_key(|(used, _)| *used);
            for (_, k) in old {
                if self.bytes <= self.budget {
                    break;
                }
                if let Some(e) = self.map.remove(&k) {
                    self.bytes -= e.bytes;
                }
            }
        }
        self.frame += 1;
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.bytes = 0;
    }
}

/// A cached glyph or icon mask.
#[derive(Clone, Debug)]
pub enum Mask {
    /// A monochrome mask (glyphs, SVG icons).
    Alpha(Arc<AlphaMask>),
    /// A colour mask (emoji).
    Color(Arc<ColorMask>),
}

impl Mask {
    /// The bytes the mask holds.
    pub fn byte_size(&self) -> usize {
        match self {
            Mask::Alpha(m) => m.data.len(),
            Mask::Color(m) => m.data.len(),
        }
    }

    /// The alpha mask, if this is one.
    pub fn alpha(&self) -> Option<&Arc<AlphaMask>> {
        match self {
            Mask::Alpha(m) => Some(m),
            Mask::Color(_) => None,
        }
    }

    /// The colour mask, if this is one.
    pub fn color(&self) -> Option<&Arc<ColorMask>> {
        match self {
            Mask::Color(m) => Some(m),
            Mask::Alpha(_) => None,
        }
    }

    /// Consumes this mask and returns the alpha mask, if it is one (avoiding an extra `Arc::clone`).
    pub fn into_alpha(self) -> Option<Arc<AlphaMask>> {
        match self {
            Mask::Alpha(m) => Some(m),
            Mask::Color(_) => None,
        }
    }

    /// Consumes this mask and returns the colour mask, if it is one (avoiding an extra `Arc::clone`).
    pub fn into_color(self) -> Option<Arc<ColorMask>> {
        match self {
            Mask::Color(m) => Some(m),
            Mask::Alpha(_) => None,
        }
    }
}

/// A value stored in a [`MaskCache`] whose pixel memory can be accounted against the byte budget.
pub trait CachedMask: Clone {
    /// Bytes held by this value's pixel data (excluding map entry bookkeeping).
    fn byte_size(&self) -> usize;
}

impl CachedMask for Mask {
    fn byte_size(&self) -> usize {
        Mask::byte_size(self)
    }
}

/// A cached glyph entry in `Window::glyph_masks`: its device-pixel raster bounds paired with its
/// mask, or `Empty` for zero-area glyphs (such as spaces) so steady frames never re-query
/// `TextSystem::raster_bounds` (and its `RwLock`).
#[derive(Clone, Debug)]
pub enum GlyphMask {
    /// A glyph whose raster bounds are empty (nothing to draw).
    Empty,
    /// A rasterized glyph with its raster bounds relative to the floored baseline origin.
    Present {
        /// Device-pixel raster bounds relative to the glyph's floored baseline origin.
        raster_bounds: Bounds<DevicePixels>,
        /// The rasterized alpha or colour mask.
        mask: Mask,
    },
}

impl CachedMask for GlyphMask {
    fn byte_size(&self) -> usize {
        match self {
            GlyphMask::Empty => 0,
            GlyphMask::Present { mask, .. } => mask.byte_size(),
        }
    }
}

impl From<Arc<AlphaMask>> for Mask {
    fn from(m: Arc<AlphaMask>) -> Self {
        Mask::Alpha(m)
    }
}

impl From<Arc<ColorMask>> for Mask {
    fn from(m: Arc<ColorMask>) -> Self {
        Mask::Color(m)
    }
}

/// The window's cache of glyph and SVG icon masks (it stands where GPUI's sprite atlas stood).
///
/// Call `begin_frame` before a frame's paint and `end_frame` after it. While the cache holds more
/// than its byte budget, `end_frame` drops the least recently used masks the frame did not use;
/// masks unused for more than `max_idle_frames` frames are dropped regardless (checked every frame
/// when `max_idle_frames < 64`, and every 64 frames or when over budget for longer limits).
pub struct MaskCache<K, V = Mask> {
    inner: ByteCache<K, V>,
}

impl<K: Hash + Eq + Clone, V: CachedMask> MaskCache<K, V> {
    /// The default budget: 32 MB of masks.
    pub const DEFAULT_BUDGET: usize = 32 << 20;
    /// The default idle limit: about a minute at 60 Hz.
    pub const DEFAULT_MAX_IDLE_FRAMES: u64 = 3600;

    /// A cache with the default budget and idle limit.
    pub fn new() -> Self {
        Self::with_budget(Self::DEFAULT_BUDGET, Self::DEFAULT_MAX_IDLE_FRAMES)
    }

    /// A cache holding about `budget_bytes` of masks, dropping masks idle for `max_idle_frames`.
    pub fn with_budget(budget_bytes: usize, max_idle_frames: u64) -> Self {
        MaskCache { inner: ByteCache::new(budget_bytes, max_idle_frames) }
    }

    /// The mask for `key`; on a miss `make` renders it. `Ok(None)` (nothing to draw) is not cached,
    /// as GPUI's atlas does not cache it.
    pub fn get_or_insert_with(
        &mut self,
        key: &K,
        make: impl FnOnce() -> Result<Option<V>>,
    ) -> Result<Option<V>> {
        self.get_or_insert_with_by(key, || key.clone(), make)
    }

    /// Like [`Self::get_or_insert_with`], looking up by a borrowed key `Q` so cache hits do not
    /// have to construct or clone an owned `K`. On a miss, `make_key` builds the owned key only if
    /// `make` produces `Some(mask)`.
    pub fn get_or_insert_with_by<Q: Hash + Eq + ?Sized>(
        &mut self,
        key: &Q,
        make_key: impl FnOnce() -> K,
        make: impl FnOnce() -> Result<Option<V>>,
    ) -> Result<Option<V>>
    where
        K: Borrow<Q>,
    {
        if let Some(m) = self.inner.get_by(key) {
            return Ok(Some(m));
        }
        let Some(mask) = make()? else {
            return Ok(None);
        };
        let bytes = mask.byte_size() + std::mem::size_of::<K>() + 64;
        self.inner.insert(make_key(), mask.clone(), bytes);
        Ok(Some(mask))
    }

    /// Like [`Self::get_or_insert_with`], returning a reference to the cached value in a single hash
    /// probe so callers can inspect metadata (or cull against a clip rect) before cloning an `Arc`.
    pub(crate) fn get_or_insert_with_ref(
        &mut self,
        key: K,
        make: impl FnOnce(&K) -> Result<Option<V>>,
    ) -> Result<Option<&V>> {
        self.inner.get_or_try_insert_ref(key, |key| {
            let Some(mask) = make(key)? else {
                return Ok(None);
            };
            let bytes = mask.byte_size() + std::mem::size_of::<K>() + 64;
            Ok(Some((mask, bytes)))
        })
    }

    /// Starts a frame (a no-op kept for symmetry with `end_frame`).
    pub fn begin_frame(&mut self) {}

    /// Ends a frame and evicts.
    pub fn end_frame(&mut self) {
        self.inner.end_frame();
    }

    /// Bytes held (masks plus bookkeeping).
    pub fn bytes(&self) -> usize {
        self.inner.bytes()
    }

    /// How many masks are cached.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Drops every mask.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<K: Hash + Eq + Clone, V: CachedMask> Default for MaskCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
