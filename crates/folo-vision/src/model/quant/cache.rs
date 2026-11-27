//! Boundary grid metadata, fast FNV-1a hashing, and packed conv/stem/bias cache keys for [`super::exec`].

use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

use folo_core::Tensor;
use folo_core::cpu::q8::{Grid, QConv, QStem};
use folo_nn::{Activation, ConvBnAct, Result};

const BIAS_CACHE_KEY: u64 = 0x666f_6c6f_6269_6173; // "folobias"

/// Fast 64-bit FNV-1a hasher for internal boundary-name lookups and `QConv` cache keys (avoids SipHash overhead
/// on ~130 lookups per forward pass).
pub(crate) struct FnvHasher(u64);

impl Default for FnvHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl FnvHasher {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    #[inline]
    pub(crate) fn new() -> Self {
        Self(Self::OFFSET)
    }
}

impl Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(Self::PRIME);
        }
        self.0 = h;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        let mut h = self.0 ^ i;
        h = h.wrapping_mul(Self::PRIME);
        self.0 = h;
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

/// [`BuildHasher`] producing [`FnvHasher`].
#[derive(Default, Clone, Copy)]
pub(crate) struct FnvBuildHasher;

impl BuildHasher for FnvBuildHasher {
    type Hasher = FnvHasher;

    #[inline]
    fn build_hasher(&self) -> FnvHasher {
        FnvHasher::new()
    }
}

/// Deterministic FNV-1a hash of a slice of per-channel [`Grid`]s.
pub(crate) fn hash_grids(grids: &[Grid]) -> u64 {
    let mut h = FnvHasher::new();
    grids.len().hash(&mut h);
    for g in grids {
        g.scale.to_bits().hash(&mut h);
        g.zero.hash(&mut h);
    }
    h.finish()
}

/// Maximum `Grid::scale` across `grids` (matches [`folo_core::cpu::q8::should_decouple_parts`]'s per-part max).
pub(crate) fn max_scale_of(grids: &[Grid]) -> f32 {
    grids.iter().map(|g| g.scale).fold(0.0f32, f32::max)
}

/// O(parts) equivalent of [`folo_core::cpu::q8::should_decouple_parts`] using precomputed per-part `max_scale`.
#[inline]
pub(crate) fn should_decouple_max_scales(max_scales: &[f32]) -> bool {
    if max_scales.len() <= 1 {
        return false;
    }
    let mut lo = f32::INFINITY;
    let mut hi = 0.0f32;
    for &m in max_scales {
        if m > 0.0 {
            lo = lo.min(m);
            hi = hi.max(m);
        }
    }
    lo.is_finite() && hi > 1.5 * lo
}

/// Cached snapshot + fingerprint of a bias tensor, stored on the bias tensor's own [`Tensor::derived_f32`] slot
/// so cache hits avoid locking, copying, and re-hashing every channel on every conv, while any in-place write to
/// the bias tensor automatically invalidates it. Acquired and released BEFORE locking `weight` to prevent
/// self-deadlock when weight and bias alias the same storage.
pub(crate) struct BiasCache {
    pub(crate) hash: u64,
    pub(crate) bits: Arc<[u32]>,
    pub(crate) vals: Arc<[f32]>,
}

impl BiasCache {
    pub(crate) fn load(bias: Option<&Tensor>) -> Result<Option<Arc<Self>>> {
        let Some(b) = bias else {
            return Ok(None);
        };
        Ok(Some(b.derived_f32(BIAS_CACHE_KEY, |s| {
            let mut h = FnvHasher::new();
            s.len().hash(&mut h);
            let mut bits = Vec::with_capacity(s.len());
            for &v in s {
                let b = v.to_bits();
                b.hash(&mut h);
                bits.push(b);
            }
            BiasCache { hash: h.finish(), bits: bits.into(), vals: s.into() }
        })?))
    }
}

/// Precomputed per-boundary grids + zero points + hashes + max scales.
///
/// Built once in [`super::Int8Plan::new`] (or lazily cached on first use for `version: 1` per-tensor ranges).
/// Precomputing `half_hashes` and `half_max_scales` makes `C3k2` / `C2Psa` half-channel `narrow` O(1) without
/// re-hashing or scanning every channel on every forward pass.
pub(crate) struct BoundaryGrids {
    pub(crate) grids: Arc<[Grid]>,
    pub(crate) zeros: Arc<[u8]>,
    pub(crate) hash: u64,
    pub(crate) max_scale: f32,
    pub(crate) half_hashes: Option<(u64, u64)>,
    pub(crate) half_max_scales: Option<(f32, f32)>,
}

impl BoundaryGrids {
    pub(crate) fn new(grids: Vec<Grid>) -> Self {
        let hash = hash_grids(&grids);
        let max_scale = max_scale_of(&grids);
        let (half_hashes, half_max_scales) = if grids.len() >= 2 && grids.len().is_multiple_of(2) {
            let mid = grids.len() / 2;
            let (lo, hi) = grids.split_at(mid);
            (Some((hash_grids(lo), hash_grids(hi))), Some((max_scale_of(lo), max_scale_of(hi))))
        } else {
            (None, None)
        };
        let zeros: Vec<u8> = grids.iter().map(|g| g.zero).collect();
        BoundaryGrids {
            grids: grids.into(),
            zeros: zeros.into(),
            hash,
            max_scale,
            half_hashes,
            half_max_scales,
        }
    }

    /// Hash and max scale for channel slice `offset..offset + len`, hitting precomputed values in O(1) for the
    /// full tensor and both half-channel `C3k2`/`C2Psa` splits.
    #[inline]
    pub(crate) fn slice_meta(&self, offset: usize, len: usize) -> (u64, f32) {
        let total = self.grids.len();
        if offset == 0 && len == total {
            return (self.hash, self.max_scale);
        }
        if let (Some((h0, h1)), Some((m0, m1))) = (self.half_hashes, self.half_max_scales) {
            let mid = total / 2;
            if len == mid {
                if offset == 0 {
                    return (h0, m0);
                }
                if offset == mid {
                    return (h1, m1);
                }
            }
        }
        let s = &self.grids[offset..offset + len];
        (hash_grids(s), max_scale_of(s))
    }
}

/// Everything a conv's packed weights are made from besides the weights' elements, kept with the packing so a
/// hash collision cannot pass for a hit.
pub(crate) struct Key {
    pub(crate) layout: (Vec<usize>, usize, Vec<usize>),
    pub(crate) geometry: [usize; 5],
    pub(crate) silu: bool,
    pub(crate) decouple: bool,
    pub(crate) part_refs: Vec<(Arc<[Grid]>, usize, usize)>,
    pub(crate) in_grids: Vec<(u32, u8)>,
    pub(crate) bias_cache: Option<Arc<BiasCache>>,
    pub(crate) bias_bits: Vec<u32>,
}

/// Minimal part metadata needed to hash, match, and pack a [`QConv`] without exposing `exec` internals.
#[derive(Clone, Copy)]
pub(crate) struct PartKeyRef<'a> {
    pub(crate) grids_arc: &'a Arc<[Grid]>,
    pub(crate) grid_offset: usize,
    pub(crate) channels: usize,
    pub(crate) grids_hash: u64,
    pub(crate) max_scale: f32,
}

impl<'a> PartKeyRef<'a> {
    #[inline]
    pub(crate) fn grids(&self) -> &'a [Grid] {
        &self.grids_arc[self.grid_offset..self.grid_offset + self.channels]
    }
}

impl Key {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matches(
        &self,
        dims: &[usize],
        offset: usize,
        strides: &[usize],
        geometry: [usize; 5],
        silu: bool,
        decouple: bool,
        parts: &[PartKeyRef<'_>],
        bias: Option<&Arc<BiasCache>>,
    ) -> bool {
        if self.geometry != geometry
            || self.silu != silu
            || self.decouple != decouple
            || self.layout.1 != offset
            || self.layout.0.as_slice() != dims
            || self.layout.2.as_slice() != strides
            || self.part_refs.len() != parts.len()
        {
            return false;
        }
        let fast_parts = self.part_refs.iter().zip(parts).all(|((g_arc, g_off, len), p)| {
            Arc::ptr_eq(g_arc, p.grids_arc) && *g_off == p.grid_offset && *len == p.channels
        });
        if !fast_parts {
            let mut idx = 0;
            for (p, &(_, _, want_len)) in parts.iter().zip(&self.part_refs) {
                if p.channels != want_len {
                    return false;
                }
                for g in &p.grids_arc[p.grid_offset..p.grid_offset + p.channels] {
                    if self.in_grids.get(idx).copied() != Some((g.scale.to_bits(), g.zero)) {
                        return false;
                    }
                    idx += 1;
                }
            }
            if idx != self.in_grids.len() {
                return false;
            }
        }
        match (&self.bias_cache, bias) {
            (Some(a), Some(b)) if Arc::ptr_eq(a, b) => true,
            (None, None) => true,
            (_, Some(b)) => self.bias_bits.as_slice() == b.bits.as_ref(),
            (_, None) => self.bias_bits.len() == geometry[1] && self.bias_bits.iter().all(|&wb| wb == 0),
        }
    }
}

pub(crate) struct Packed {
    pub(crate) key: Key,
    pub(crate) conv: Arc<QConv>,
}

pub(crate) struct StemKey {
    pub(crate) layout: (Vec<usize>, usize, Vec<usize>),
    pub(crate) geometry: [usize; 4],
    pub(crate) silu: bool,
    pub(crate) bias_cache: Option<Arc<BiasCache>>,
    pub(crate) bias_bits: Vec<u32>,
}

impl StemKey {
    pub(crate) fn matches(
        &self,
        dims: &[usize],
        offset: usize,
        strides: &[usize],
        geometry: [usize; 4],
        silu: bool,
        bias: Option<&Arc<BiasCache>>,
    ) -> bool {
        if self.geometry != geometry
            || self.silu != silu
            || self.layout.1 != offset
            || self.layout.0.as_slice() != dims
            || self.layout.2.as_slice() != strides
        {
            return false;
        }
        match (&self.bias_cache, bias) {
            (Some(a), Some(b)) if Arc::ptr_eq(a, b) => true,
            (None, None) => true,
            (_, Some(b)) => self.bias_bits.as_slice() == b.bits.as_ref(),
            (_, None) => self.bias_bits.len() == geometry[0] && self.bias_bits.iter().all(|&wb| wb == 0),
        }
    }
}

pub(crate) struct PackedStem {
    pub(crate) key: StemKey,
    pub(crate) stem: Arc<QStem>,
}

/// Load or pack `conv`'s int8 weights for input `parts`, caching the result on `conv.conv.weight` via
/// [`Tensor::derived_f32`]. Zero heap allocations and O(parts) FNV-1a hashing on cache hit.
pub(crate) fn load_qconv(
    conv: &ConvBnAct,
    parts: &[PartKeyRef<'_>],
    allow_decouple: bool,
) -> Result<Arc<QConv>> {
    let cfg = conv.conv.config();
    let geometry = [cfg.in_channels, cfg.out_channels, cfg.kernel, cfg.stride, cfg.padding];
    let silu = conv.act == Activation::Silu;

    let decouple = if allow_decouple && parts.len() > 1 {
        if parts.len() <= 8 {
            let mut ms = [0.0f32; 8];
            for (i, a) in parts.iter().enumerate() {
                ms[i] = a.max_scale;
            }
            should_decouple_max_scales(&ms[..parts.len()])
        } else {
            let ms: Vec<f32> = parts.iter().map(|a| a.max_scale).collect();
            should_decouple_max_scales(&ms)
        }
    } else {
        false
    };

    let l = conv.conv.weight.layout();
    let (dims, offset, strides) = (l.shape().dims(), l.offset(), l.strides());
    let bias_cache = BiasCache::load(conv.conv.bias.as_ref())?;
    let mut h = FnvHasher::new();
    0x7163_6f6e_7676_3500u64.hash(&mut h); // "qconvv5\0"
    dims.hash(&mut h);
    offset.hash(&mut h);
    strides.hash(&mut h);
    geometry.hash(&mut h);
    silu.hash(&mut h);
    decouple.hash(&mut h);
    parts.len().hash(&mut h);
    for a in parts {
        a.channels.hash(&mut h);
        a.grids_hash.hash(&mut h);
    }
    match &bias_cache {
        Some(b) => {
            1u8.hash(&mut h);
            b.hash.hash(&mut h);
        }
        None => {
            0u8.hash(&mut h);
            cfg.out_channels.hash(&mut h);
        }
    }
    let make_key = || Key {
        layout: (dims.to_vec(), offset, strides.to_vec()),
        geometry,
        silu,
        decouple,
        part_refs: parts.iter().map(|a| (a.grids_arc.clone(), a.grid_offset, a.channels)).collect(),
        in_grids: parts.iter().flat_map(|a| a.grids().iter().map(|g| (g.scale.to_bits(), g.zero))).collect(),
        bias_cache: bias_cache.clone(),
        bias_bits: match &bias_cache {
            Some(b) => b.bits.to_vec(),
            None => vec![0u32; cfg.out_channels],
        },
    };
    let pack = |w: &[f32]| {
        let part_grids: Vec<&[Grid]> = parts.iter().map(PartKeyRef::grids).collect();
        let shape = (cfg.out_channels, cfg.in_channels, cfg.kernel);
        match &bias_cache {
            Some(b) => Arc::new(QConv::new_parts(
                w,
                &b.vals,
                shape,
                cfg.stride,
                cfg.padding,
                silu,
                &part_grids,
                decouple,
            )),
            None => {
                let zeros = vec![0.0; cfg.out_channels];
                Arc::new(QConv::new_parts(
                    w,
                    &zeros,
                    shape,
                    cfg.stride,
                    cfg.padding,
                    silu,
                    &part_grids,
                    decouple,
                ))
            }
        }
    };
    let kept = conv.conv.weight.derived_f32(h.finish(), |w| Packed { key: make_key(), conv: pack(w) })?;
    if kept.key.matches(dims, offset, strides, geometry, silu, decouple, parts, bias_cache.as_ref()) {
        return Ok(kept.conv.clone());
    }
    Ok(pack(&conv.conv.weight.to_vec::<f32>()?))
}

/// Load or pack the 3-channel stem's `u8` weights, cached on `conv.conv.weight` (zero heap allocations on hit).
pub(crate) fn load_qstem(conv: &ConvBnAct) -> Result<Arc<QStem>> {
    let cfg = conv.conv.config();
    let geometry = [cfg.out_channels, cfg.kernel, cfg.stride, cfg.padding];
    let silu = conv.act == Activation::Silu;
    let l = conv.conv.weight.layout();
    let (dims, offset, strides) = (l.shape().dims(), l.offset(), l.strides());
    let bias_cache = BiasCache::load(conv.conv.bias.as_ref())?;
    let mut h = FnvHasher::new();
    0x7173_7465_6d76_3500u64.hash(&mut h); // "qstemv5\0"
    dims.hash(&mut h);
    offset.hash(&mut h);
    strides.hash(&mut h);
    geometry.hash(&mut h);
    silu.hash(&mut h);
    match &bias_cache {
        Some(b) => {
            1u8.hash(&mut h);
            b.hash.hash(&mut h);
        }
        None => {
            0u8.hash(&mut h);
            cfg.out_channels.hash(&mut h);
        }
    }
    let make_key = || StemKey {
        layout: (dims.to_vec(), offset, strides.to_vec()),
        geometry,
        silu,
        bias_cache: bias_cache.clone(),
        bias_bits: match &bias_cache {
            Some(b) => b.bits.to_vec(),
            None => vec![0u32; cfg.out_channels],
        },
    };
    let pack = |w: &[f32]| match &bias_cache {
        Some(b) => {
            Arc::new(QStem::new(w, &b.vals, (cfg.out_channels, cfg.kernel), cfg.stride, cfg.padding, silu))
        }
        None => {
            let zeros = vec![0.0; cfg.out_channels];
            Arc::new(QStem::new(w, &zeros, (cfg.out_channels, cfg.kernel), cfg.stride, cfg.padding, silu))
        }
    };
    let kept = conv.conv.weight.derived_f32(h.finish(), |w| PackedStem { key: make_key(), stem: pack(w) })?;
    if kept.key.matches(dims, offset, strides, geometry, silu, bias_cache.as_ref()) {
        return Ok(kept.stem.clone());
    }
    Ok(pack(&conv.conv.weight.to_vec::<f32>()?))
}
