//! The int8 executor (docs/MODEL.md §9.6): the quantised graph ([`walk`]) on folo-core's int8 kernels
//! ([`folo_core::cpu::q8`]). Activations are u8 C4 buffers from the CPU buffer cache with per-channel grids
//! (`version: 2`, or uniform grids for `version: 1`); each conv's packed weights (and direct `u8` stem) are built
//! on its first call and kept with its weight tensor ([`Tensor::derived_f32`]) until that is written.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock, RwLock};

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid, QConv, QOut, QStem};
use folo_nn::{Activation, ConvBnAct, Error, Result};

use super::calib::conv_f32;
use super::walk::{self, Exec};
use super::{Calibration, VERSION_V2, int8_able, stem_int8_able};
use crate::model::FoloModel;

/// Shared per-channel grids, zero points, and their precomputed hash for an activation boundary.
type GridTriple = (Arc<[Grid]>, Arc<[u8]>, u64);

const BIAS_CACHE_KEY: u64 = 0x666f_6c6f_6269_6173; // "folobias"

fn hash_grids(grids: &[Grid]) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    grids.len().hash(&mut h);
    for g in grids {
        g.scale.to_bits().hash(&mut h);
        g.zero.hash(&mut h);
    }
    h.finish()
}

/// Cached snapshot + fingerprint of a bias tensor, stored on the bias tensor's own [`Tensor::derived_f32`] slot
/// so cache hits avoid locking, copying, and re-hashing every channel on every conv, while any in-place write to
/// the bias tensor automatically invalidates it. Acquired and released BEFORE locking `weight` to prevent
/// self-deadlock when weight and bias alias the same storage.
struct BiasCache {
    hash: u64,
    bits: Arc<[u32]>,
    vals: Arc<[f32]>,
}

impl BiasCache {
    fn load(bias: Option<&Tensor>) -> Result<Option<Arc<Self>>> {
        let Some(b) = bias else {
            return Ok(None);
        };
        Ok(Some(b.derived_f32(BIAS_CACHE_KEY, |s| {
            let mut h = std::hash::DefaultHasher::new();
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

/// Precomputed per-boundary grids + zero points + hash (built once in [`Int8Plan::new`] or lazily cached on
/// first use for `version: 1` per-tensor ranges so every forward skips `BTreeMap` lookups, `Grid::from_range`
/// divisions, and per-layer `Vec`/`Arc` allocations).
struct BoundaryGrids {
    grids: Arc<[Grid]>,
    zeros: Arc<[u8]>,
    hash: u64,
}

impl BoundaryGrids {
    fn new(grids: Vec<Grid>) -> Self {
        let hash = hash_grids(&grids);
        let zeros: Vec<u8> = grids.iter().map(|g| g.zero).collect();
        BoundaryGrids { grids: grids.into(), zeros: zeros.into(), hash }
    }

    fn triple(&self) -> GridTriple {
        (self.grids.clone(), self.zeros.clone(), self.hash)
    }
}

/// A calibrated model's int8 plan. A conv's packed weights are kept with its weight tensor
/// ([`Tensor::derived_f32`]) under a key of everything else they are made from: a write to the weights drops
/// them, even one that overlaps a forward, and the next forward packs them again.
pub struct Int8Plan {
    cal: Calibration,
    /// Precomputed per-channel grids for boundaries with `channel_ranges` entries.
    ch_grids: HashMap<String, BoundaryGrids>,
    /// Precomputed per-tensor fallback grid for boundaries without `channel_ranges`.
    tensor_grids: HashMap<String, Grid>,
    /// Lazily expanded per-boundary grids for `version: 1` per-tensor calibrations (avoids per-conv heap
    /// allocations on every forward pass).
    uniform_grids: RwLock<HashMap<String, BoundaryGrids>>,
    /// O(1) lookup set of `cal.f32_convs`.
    f32_set: HashSet<String>,
}

impl Int8Plan {
    /// The plan of `cal`.
    pub fn new(cal: Calibration) -> Self {
        let mut ch_grids = HashMap::with_capacity(cal.channel_ranges.len());
        let mut tensor_grids = HashMap::with_capacity(cal.ranges.len());
        if cal.check_version().is_ok() {
            for name in cal.ranges.keys() {
                if let Ok(g) = cal.param(name) {
                    tensor_grids.insert(name.clone(), g);
                }
                if let Some(crs) = cal.channel_ranges.get(name)
                    && let Ok(gs) = cal.channel_params(name, crs.len())
                {
                    ch_grids.insert(name.clone(), BoundaryGrids::new(gs));
                }
            }
        }
        let f32_set = cal.f32_convs.iter().cloned().collect();
        Int8Plan { cal, ch_grids, tensor_grids, uniform_grids: RwLock::new(HashMap::new()), f32_set }
    }

    /// The calibration.
    pub fn calibration(&self) -> &Calibration {
        &self.cal
    }

    #[inline]
    fn is_f32(&self, name: &str) -> bool {
        self.f32_set.contains(name)
    }

    fn grids_for(&self, name: &str, c: usize) -> Result<GridTriple> {
        if let Some(bg) = self.ch_grids.get(name)
            && bg.grids.len() == c
        {
            return Ok(bg.triple());
        }
        if !self.cal.channel_ranges.contains_key(name)
            && let Some(&g) = self.tensor_grids.get(name)
        {
            if let Ok(guard) = self.uniform_grids.read()
                && let Some(bg) = guard.get(name)
                && bg.grids.len() == c
            {
                return Ok(bg.triple());
            }
            if let Ok(mut guard) = self.uniform_grids.write() {
                if let Some(bg) = guard.get(name)
                    && bg.grids.len() == c
                {
                    return Ok(bg.triple());
                }
                let bg = BoundaryGrids::new(vec![g; c]);
                let out = bg.triple();
                guard.insert(name.to_string(), bg);
                return Ok(out);
            }
            return Ok(BoundaryGrids::new(vec![g; c]).triple());
        }
        let gs = self.cal.channel_params(name, c)?;
        Ok(BoundaryGrids::new(gs).triple())
    }
}

/// A C4 buffer that goes back to the CPU buffer cache when its last activation is dropped.
struct Buf(Vec<u32>);

impl Drop for Buf {
    fn drop(&mut self) {
        q8::recycle(std::mem::take(&mut self.0));
    }
}

/// A materialised u8 C4 activation (or a channel slice of one) with per-channel affine grids.
#[derive(Clone)]
struct QActU8 {
    buf: Arc<Buf>,
    offset: usize,
    img_stride: usize,
    dims: [usize; 4],
    grids: Arc<[Grid]>,
    zeros: Arc<[u8]>,
    grid_offset: usize,
    grids_hash: u64,
}

impl QActU8 {
    fn grids(&self) -> &[Grid] {
        &self.grids[self.grid_offset..self.grid_offset + self.dims[1]]
    }

    fn zeros(&self) -> &[u8] {
        &self.zeros[self.grid_offset..self.grid_offset + self.dims[1]]
    }

    fn view(&self) -> C4<'_> {
        C4 {
            data: &self.buf.0,
            offset: self.offset,
            img_stride: self.img_stride,
            c: self.dims[1],
            zero: self.zeros().first().copied().unwrap_or(0),
        }
    }

    fn new(buf: Vec<u32>, dims: [usize; 4], grids: Arc<[Grid]>, zeros: Arc<[u8]>, grids_hash: u64) -> QActU8 {
        let img_stride = dims[1] / 4 * dims[2] * dims[3];
        QActU8 {
            buf: Arc::new(Buf(buf)),
            offset: 0,
            img_stride,
            dims,
            grids,
            zeros,
            grid_offset: 0,
            grids_hash,
        }
    }

    fn from_f32(
        x: &Tensor,
        grids: Arc<[Grid]>,
        zeros: Arc<[u8]>,
        grid_offset: usize,
        grids_hash: u64,
    ) -> Result<QActU8> {
        let (n, c, h, w) = x.dims4()?;
        if !c.is_multiple_of(4) {
            return Err(Error::Invalid(format!("int8: {c} channels is not a multiple of 4")));
        }
        let slice = &grids[grid_offset..grid_offset + c];
        let mut out = q8::alloc(n, c, h * w);
        x.with_f32_slice(|v| q8::quantize_per_channel(v, n, c, h * w, slice, &mut out))?;
        let img_stride = c / 4 * h * w;
        Ok(QActU8 {
            buf: Arc::new(Buf(out)),
            offset: 0,
            img_stride,
            dims: [n, c, h, w],
            grids,
            zeros,
            grid_offset,
            grids_hash,
        })
    }

    fn to_f32(&self) -> Result<Tensor> {
        let [n, c, h, w] = self.dims;
        let mut out = folo_core::cpu::take_buffer(n * c * h * w);
        q8::dequantize_per_channel(self.view(), n, h * w, self.grids(), &mut out);
        Ok(Tensor::from_vec(out, [n, c, h, w])?)
    }
}

/// An unquantised `f32` activation with deferred `u8` quantisation (materialised at most once if an int8 op
/// consumes it, and passed through directly in `f32` to `f32` convs and the `f32` head).
#[derive(Clone)]
struct QActF32 {
    tensor: Tensor,
    dims: [usize; 4],
    grids: Arc<[Grid]>,
    zeros: Arc<[u8]>,
    grid_offset: usize,
    grids_hash: u64,
    u8_cache: Arc<OnceLock<QActU8>>,
}

#[derive(Clone)]
enum QActRepr {
    U8(QActU8),
    F32(QActF32),
}

/// An activation in the int8 executor: either a materialised u8 C4 buffer or a deferred `f32` tensor carrying
/// its boundary's per-channel affine grids.
#[derive(Clone)]
pub struct QAct {
    repr: QActRepr,
}

impl QAct {
    fn new(buf: Vec<u32>, dims: [usize; 4], grids: Arc<[Grid]>, zeros: Arc<[u8]>, grids_hash: u64) -> QAct {
        QAct { repr: QActRepr::U8(QActU8::new(buf, dims, grids, zeros, grids_hash)) }
    }

    fn from_f32_lazy(tensor: Tensor, (grids, zeros, grids_hash): GridTriple) -> Result<QAct> {
        let (n, c, h, w) = tensor.dims4()?;
        Ok(QAct {
            repr: QActRepr::F32(QActF32 {
                tensor,
                dims: [n, c, h, w],
                grids,
                zeros,
                grid_offset: 0,
                grids_hash,
                u8_cache: Arc::new(OnceLock::new()),
            }),
        })
    }

    #[inline]
    fn dims(&self) -> [usize; 4] {
        match &self.repr {
            QActRepr::U8(u) => u.dims,
            QActRepr::F32(f) => f.dims,
        }
    }

    fn to_f32(&self) -> Result<Tensor> {
        match &self.repr {
            QActRepr::U8(u) => u.to_f32(),
            QActRepr::F32(f) => Ok(f.tensor.clone()),
        }
    }

    fn as_u8(&self) -> Result<&QActU8> {
        match &self.repr {
            QActRepr::U8(u) => Ok(u),
            QActRepr::F32(f) => {
                if let Some(u) = f.u8_cache.get() {
                    return Ok(u);
                }
                let u = QActU8::from_f32(
                    &f.tensor,
                    f.grids.clone(),
                    f.zeros.clone(),
                    f.grid_offset,
                    f.grids_hash,
                )?;
                Ok(f.u8_cache.get_or_init(|| u))
            }
        }
    }
}

/// Everything a conv's packed weights are made from besides the weights' elements, kept with the packing so a
/// hash collision cannot pass for a hit.
struct Key {
    /// The weight view's dims, offset and strides (the elements it reads).
    layout: (Vec<usize>, usize, Vec<usize>),
    geometry: [usize; 5],
    silu: bool,
    decouple: bool,
    /// Per-part `(grids Arc, offset, len)` for O(1) pointer fast-path matching.
    part_refs: Vec<(Arc<[Grid]>, usize, usize)>,
    /// Per-input-channel `(scale_bits, zero)` fallback when `Arc` pointers differ.
    in_grids: Vec<(u32, u8)>,
    /// Bias cache pointer for O(1) fast-path matching.
    bias_cache: Option<Arc<BiasCache>>,
    /// Fallback bias bits when `Arc` pointers differ.
    bias_bits: Vec<u32>,
}

impl Key {
    #[allow(clippy::too_many_arguments)]
    fn matches(
        &self,
        dims: &[usize],
        offset: usize,
        strides: &[usize],
        geometry: [usize; 5],
        silu: bool,
        decouple: bool,
        parts: &[&QActU8],
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
            Arc::ptr_eq(g_arc, &p.grids) && *g_off == p.grid_offset && *len == p.dims[1]
        });
        if !fast_parts {
            let mut idx = 0;
            for (p, &(_, _, want_len)) in parts.iter().zip(&self.part_refs) {
                if p.dims[1] != want_len {
                    return false;
                }
                for g in p.grids() {
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

/// A packed conv and its [`Key`].
struct Packed {
    key: Key,
    conv: Arc<QConv>,
}

/// Cache key for a packed [`QStem`].
struct StemKey {
    layout: (Vec<usize>, usize, Vec<usize>),
    geometry: [usize; 4],
    silu: bool,
    bias_cache: Option<Arc<BiasCache>>,
    bias_bits: Vec<u32>,
}

impl StemKey {
    fn matches(
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

struct PackedStem {
    key: StemKey,
    stem: Arc<QStem>,
}

/// The int8 executor of one forward.
struct Int8<'a> {
    plan: &'a Int8Plan,
    /// Optional pre-packed `[n, h, w]` `[R, G, B, 0]` input from `Letterbox::apply_rgb0_u8`.
    rgb0_input: Option<(&'a [u32], [usize; 3])>,
}

impl Int8<'_> {
    /// `conv`'s packed weights for inputs `parts`: kept with the weight tensor, keyed by the rest.
    /// Zero heap allocations and O(parts) hashing on cache hit.
    fn qconv(&self, conv: &ConvBnAct, parts: &[&QActU8]) -> Result<Arc<QConv>> {
        let cfg = conv.conv.config();
        let geometry = [cfg.in_channels, cfg.out_channels, cfg.kernel, cfg.stride, cfg.padding];
        let silu = conv.act == Activation::Silu;
        let mut pg_arr: [&[Grid]; 8] = [&[]; 8];
        let part_grids_vec;
        let part_grids: &[&[Grid]] = if parts.len() <= 8 {
            for (i, a) in parts.iter().enumerate() {
                pg_arr[i] = a.grids();
            }
            &pg_arr[..parts.len()]
        } else {
            part_grids_vec = parts.iter().map(|a| a.grids()).collect::<Vec<_>>();
            &part_grids_vec
        };
        let decouple = self.plan.cal.version >= VERSION_V2 && q8::should_decouple_parts(part_grids);
        let l = conv.conv.weight.layout();
        let (dims, offset, strides) = (l.shape().dims(), l.offset(), l.strides());
        let bias_cache = BiasCache::load(conv.conv.bias.as_ref())?;
        let mut h = std::hash::DefaultHasher::new();
        "folo-vision int8 QConv v4".hash(&mut h);
        dims.hash(&mut h);
        offset.hash(&mut h);
        strides.hash(&mut h);
        geometry.hash(&mut h);
        silu.hash(&mut h);
        decouple.hash(&mut h);
        parts.len().hash(&mut h);
        for a in parts {
            a.dims[1].hash(&mut h);
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
            part_refs: parts.iter().map(|a| (a.grids.clone(), a.grid_offset, a.dims[1])).collect(),
            in_grids: part_grids
                .iter()
                .flat_map(|gs| gs.iter().map(|g| (g.scale.to_bits(), g.zero)))
                .collect(),
            bias_cache: bias_cache.clone(),
            bias_bits: match &bias_cache {
                Some(b) => b.bits.to_vec(),
                None => vec![0u32; cfg.out_channels],
            },
        };
        let pack = |w: &[f32]| {
            let shape = (cfg.out_channels, cfg.in_channels, cfg.kernel);
            match &bias_cache {
                Some(b) => Arc::new(QConv::new_parts(
                    w,
                    &b.vals,
                    shape,
                    cfg.stride,
                    cfg.padding,
                    silu,
                    part_grids,
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
                        part_grids,
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

    /// The stem's packed `u8` weights, cached on the weight tensor (zero heap allocations on cache hit).
    fn qstem(&self, conv: &ConvBnAct) -> Result<Arc<QStem>> {
        let cfg = conv.conv.config();
        let geometry = [cfg.out_channels, cfg.kernel, cfg.stride, cfg.padding];
        let silu = conv.act == Activation::Silu;
        let l = conv.conv.weight.layout();
        let (dims, offset, strides) = (l.shape().dims(), l.offset(), l.strides());
        let bias_cache = BiasCache::load(conv.conv.bias.as_ref())?;
        let mut h = std::hash::DefaultHasher::new();
        "folo-vision int8 QStem v4".hash(&mut h);
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
            Some(b) => Arc::new(QStem::new(
                w,
                &b.vals,
                (cfg.out_channels, cfg.kernel),
                cfg.stride,
                cfg.padding,
                silu,
            )),
            None => {
                let zeros = vec![0.0; cfg.out_channels];
                Arc::new(QStem::new(w, &zeros, (cfg.out_channels, cfg.kernel), cfg.stride, cfg.padding, silu))
            }
        };
        let kept =
            conv.conv.weight.derived_f32(h.finish(), |w| PackedStem { key: make_key(), stem: pack(w) })?;
        if kept.key.matches(dims, offset, strides, geometry, silu, bias_cache.as_ref()) {
            return Ok(kept.stem.clone());
        }
        Ok(pack(&conv.conv.weight.to_vec::<f32>()?))
    }

    /// The int8 conv into `out`.
    fn run(
        &self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&QActU8],
        res: Option<&QActU8>,
        out: QOut<'_>,
    ) -> Result<[usize; 4]> {
        let [n, _, h, w] = parts[0].dims;
        if parts.iter().any(|a| a.dims[0] != n || a.dims[2] != h || a.dims[3] != w) {
            return Err(Error::Invalid(format!("int8 {name}: parts of different shapes")));
        }
        let qc = self.qconv(conv, parts)?;
        let (oh, ow) = qc.out_hw(h, w);
        let res = res.map(|r| (r.view(), r.grids()));
        if parts.len() <= 8 {
            let mut v_arr = [parts[0].view(); 8];
            let mut z_arr: [&[u8]; 8] = [&[]; 8];
            for (i, a) in parts.iter().enumerate() {
                v_arr[i] = a.view();
                z_arr[i] = a.zeros();
            }
            q8::conv_per_channel(&v_arr[..parts.len()], &z_arr[..parts.len()], n, h, w, &qc, res, out);
        } else {
            let views: Vec<C4> = parts.iter().map(|a| a.view()).collect();
            let zeros: Vec<&[u8]> = parts.iter().map(|a| a.zeros()).collect();
            q8::conv_per_channel(&views, &zeros, n, h, w, &qc, res, out);
        }
        Ok([n, qc.out_channels(), oh, ow])
    }
}

impl Exec for Int8<'_> {
    type A = QAct;

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<QAct> {
        use folo_nn::Forward;
        let o = conv.conv.config().out_channels;
        let (grids, zeros, grids_hash) = self.plan.grids_for(name, o)?;
        let (n, c, h, w) = match self.rgb0_input {
            Some((_, [n, h, w])) => (n, 3, h, w),
            None => x.dims4()?,
        };
        if self.plan.cal.u8_stem
            && !self.plan.f32_set.contains(name)
            && let Some((oh, ow)) = stem_int8_able(conv, [n, c, h, w])
        {
            let qs = self.qstem(conv)?;
            let mut out = q8::alloc(n, o, oh * ow);
            if let Some((rgb0, _)) = self.rgb0_input {
                q8::stem_u8(rgb0, n, h, w, &qs, QOut::U8PerChannel { data: &mut out, grids: &grids });
            } else {
                let mut rgb0 = q8::alloc(n, 4, h * w);
                x.with_f32_slice(|v| q8::pack_f32_to_rgb0_u8(v, n, h, w, &mut rgb0))?;
                q8::stem_u8(&rgb0, n, h, w, &qs, QOut::U8PerChannel { data: &mut out, grids: &grids });
                q8::recycle(rgb0);
            }
            return Ok(QAct::new(out, [n, o, oh, ow], grids, zeros, grids_hash));
        }
        let y = if let Some((rgb0, [n, h, w])) = self.rgb0_input {
            let hw = h * w;
            let mut planar = folo_core::cpu::take_buffer(n * 3 * hw);
            const PIXEL_SCALE: f32 = 1.0 / 255.0;
            for img in 0..n {
                let src = &rgb0[img * hw..(img + 1) * hw];
                let base = img * 3 * hw;
                for (p, &word) in src.iter().enumerate() {
                    let [r, g, b, _] = word.to_le_bytes();
                    planar[base + p] = f32::from(r) * PIXEL_SCALE;
                    planar[base + hw + p] = f32::from(g) * PIXEL_SCALE;
                    planar[base + 2 * hw + p] = f32::from(b) * PIXEL_SCALE;
                }
            }
            let x_f32 = Tensor::from_vec(planar, vec![n, 3, h, w])?;
            conv.forward(&x_f32)?
        } else {
            conv.forward(x)?
        };
        QAct::from_f32_lazy(y, (grids, zeros, grids_hash))
    }

    fn conv(&mut self, name: &str, conv: &ConvBnAct, parts: &[&QAct], res: Option<&QAct>) -> Result<QAct> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let (grids, zeros, grids_hash) = self.plan.grids_for(name, o)?;
        let able = parts
            .iter()
            .all(|a| a.dims()[1].is_multiple_of(4))
            .then(|| int8_able(conv, [n, c, h, w], res.map(QAct::dims)))
            .flatten();
        let (Some((oh, ow)), false) = (able, self.plan.is_f32(name)) else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            let r = res.map(QAct::to_f32).transpose()?;
            let y = conv_f32(conv, &ts.iter().collect::<Vec<_>>(), r.as_ref())?;
            return QAct::from_f32_lazy(y, (grids, zeros, grids_hash));
        };
        let u8_res = res.map(QAct::as_u8).transpose()?;
        let mut out = q8::alloc(n, o, oh * ow);
        let dims = if parts.len() <= 8 {
            let first = parts[0].as_u8()?;
            let mut arr = [first; 8];
            for (i, a) in parts.iter().enumerate().skip(1) {
                arr[i] = a.as_u8()?;
            }
            self.run(
                name,
                conv,
                &arr[..parts.len()],
                u8_res,
                QOut::U8PerChannel { data: &mut out, grids: &grids },
            )?
        } else {
            let u8_parts: Vec<&QActU8> = parts.iter().map(|a| a.as_u8()).collect::<Result<_>>()?;
            self.run(name, conv, &u8_parts, u8_res, QOut::U8PerChannel { data: &mut out, grids: &grids })?
        };
        Ok(QAct::new(out, dims, grids, zeros, grids_hash))
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &QAct) -> Result<Tensor> {
        let xd = x.dims();
        let (Some((oh, ow)), false) = (int8_able(conv, xd, None), self.plan.is_f32(name)) else {
            return conv_f32(conv, &[&x.to_f32()?], None);
        };
        let u8_x = x.as_u8()?;
        let (n, o) = (xd[0], conv.conv.config().out_channels);
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims = self.run(name, conv, &[u8_x], None, QOut::F32 { data: &mut out })?;
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn conv_out_f32(&mut self, name: &str, conv: &ConvBnAct, parts: &[&QAct]) -> Result<Tensor> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let _ = self.plan.grids_for(name, o)?;
        let able = parts
            .iter()
            .all(|a| a.dims()[1].is_multiple_of(4))
            .then(|| int8_able(conv, [n, c, h, w], None))
            .flatten();
        let (Some((oh, ow)), false) = (able, self.plan.is_f32(name)) else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            return conv_f32(conv, &ts.iter().collect::<Vec<_>>(), None);
        };
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims = if parts.len() <= 8 {
            let first = parts[0].as_u8()?;
            let mut arr = [first; 8];
            for (i, a) in parts.iter().enumerate().skip(1) {
                arr[i] = a.as_u8()?;
            }
            self.run(name, conv, &arr[..parts.len()], None, QOut::F32 { data: &mut out })?
        } else {
            let u8_parts: Vec<&QActU8> = parts.iter().map(|a| a.as_u8()).collect::<Result<_>>()?;
            self.run(name, conv, &u8_parts, None, QOut::F32 { data: &mut out })?
        };
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<QAct> {
        let c = x.dims4()?.1;
        let triple = self.plan.grids_for(name, c)?;
        QAct::from_f32_lazy(x.clone(), triple)
    }

    fn dequantize(&mut self, x: &QAct) -> Result<Tensor> {
        x.to_f32()
    }

    fn narrow(&mut self, x: &QAct, start: usize, len: usize) -> Result<QAct> {
        match &x.repr {
            QActRepr::U8(u) => {
                let [n, c, h, w] = u.dims;
                if !start.is_multiple_of(4) || !len.is_multiple_of(4) || start + len > c {
                    return Err(Error::Invalid(format!(
                        "int8: channels {start}..{} of {c} are not whole C4 groups",
                        start + len
                    )));
                }
                let mut y = u.clone();
                y.offset += start / 4 * h * w;
                y.grid_offset += start;
                y.dims = [n, len, h, w];
                y.grids_hash = hash_grids(y.grids());
                Ok(QAct { repr: QActRepr::U8(y) })
            }
            QActRepr::F32(f) => {
                let [n, c, h, w] = f.dims;
                if start + len > c {
                    return Err(Error::Invalid(format!(
                        "int8: channels {start}..{} out of bounds for {c}",
                        start + len
                    )));
                }
                let grid_offset = f.grid_offset + start;
                let grids_hash = hash_grids(&f.grids[grid_offset..grid_offset + len]);
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor: f.tensor.narrow(1, start, len)?,
                        dims: [n, len, h, w],
                        grids: f.grids.clone(),
                        zeros: f.zeros.clone(),
                        grid_offset,
                        grids_hash,
                        u8_cache: Arc::new(OnceLock::new()),
                    }),
                })
            }
        }
    }

    fn max_pool(&mut self, x: &QAct, k: usize) -> Result<QAct> {
        match &x.repr {
            QActRepr::U8(u) => {
                let [n, c, h, w] = u.dims;
                let mut out = q8::alloc(n, c, h * w);
                q8::max_pool(u.view(), n, h, w, k, &mut out);
                Ok(QAct {
                    repr: QActRepr::U8(QActU8 {
                        buf: Arc::new(Buf(out)),
                        offset: 0,
                        img_stride: c / 4 * h * w,
                        dims: u.dims,
                        grids: u.grids.clone(),
                        zeros: u.zeros.clone(),
                        grid_offset: u.grid_offset,
                        grids_hash: u.grids_hash,
                    }),
                })
            }
            QActRepr::F32(f) => {
                let tensor = f.tensor.max_pool2d(k, 1, k / 2)?;
                let (n, c, oh, ow) = tensor.dims4()?;
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor,
                        dims: [n, c, oh, ow],
                        grids: f.grids.clone(),
                        zeros: f.zeros.clone(),
                        grid_offset: f.grid_offset,
                        grids_hash: f.grids_hash,
                        u8_cache: Arc::new(OnceLock::new()),
                    }),
                })
            }
        }
    }

    fn upsample2(&mut self, x: &QAct) -> Result<QAct> {
        match &x.repr {
            QActRepr::U8(u) => {
                let [n, c, h, w] = u.dims;
                let mut out = q8::alloc(n, c, 4 * h * w);
                q8::upsample2(u.view(), n, h, w, &mut out);
                Ok(QAct {
                    repr: QActRepr::U8(QActU8 {
                        buf: Arc::new(Buf(out)),
                        offset: 0,
                        img_stride: c / 4 * 4 * h * w,
                        dims: [n, c, 2 * h, 2 * w],
                        grids: u.grids.clone(),
                        zeros: u.zeros.clone(),
                        grid_offset: u.grid_offset,
                        grids_hash: u.grids_hash,
                    }),
                })
            }
            QActRepr::F32(f) => {
                let [n, c, h, w] = f.dims;
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor: f.tensor.upsample_nearest2d(2)?,
                        dims: [n, c, 2 * h, 2 * w],
                        grids: f.grids.clone(),
                        zeros: f.zeros.clone(),
                        grid_offset: f.grid_offset,
                        grids_hash: f.grids_hash,
                        u8_cache: Arc::new(OnceLock::new()),
                    }),
                })
            }
        }
    }

    fn dims(&self, x: &QAct) -> [usize; 4] {
        x.dims()
    }
}

/// Layers 0–22 of `model` on `x` with the int8 plan: the head's three inputs (`f32`). Call it on the CPU, inside
/// folo-core's pool, under `no_grad`, on a fused eval-mode `f32` model.
pub fn features(model: &FoloModel, plan: &Int8Plan, x: &Tensor) -> Result<[Tensor; 3]> {
    walk::features(model, &mut Int8 { plan, rgb0_input: None }, x)
}

/// [`features`] reading a pre-packed `[n, h, w]` `u32` `[R, G, B, 0]` canvas (from `Letterbox::apply_rgb0_u8`)
/// straight into the direct `u8` AVX-VNNI stem with zero `f32` image materialisation.
pub fn features_rgb0_u8(
    model: &FoloModel,
    plan: &Int8Plan,
    rgb0: &[u32],
    n: usize,
    h: usize,
    w: usize,
) -> Result<[Tensor; 3]> {
    let dummy = Tensor::zeros([0]);
    walk::features(model, &mut Int8 { plan, rgb0_input: Some((rgb0, [n, h, w])) }, &dummy)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::model::blocks::Init;
    use crate::model::quant::Method;
    use folo_core::rng::Rng;

    /// Weight and bias views sharing the same underlying storage `RwLock` must never deadlock during `qconv` or
    /// `qstem` packing (cache miss) or lookup (cache hit).
    #[test]
    fn aliased_weight_and_bias_storage_does_not_deadlock() {
        let mut rng = Rng::seed(99);
        let cal = Calibration {
            version: VERSION_V2,
            method: Method::MinMax,
            images: 1,
            ranges: BTreeMap::new(),
            channel_ranges: BTreeMap::new(),
            u8_stem: true,
            f32_convs: Vec::new(),
            weights: None,
        };
        let plan = Int8Plan::new(cal);
        let exec = Int8 { plan: &plan, rgb0_input: None };

        // Aliased conv (`bias` is a view into `weight`'s own storage).
        let mut conv = Init::new(&mut rng).conv(8, 8, 3, 1);
        conv.fuse_().unwrap();
        conv.conv.bias = Some(conv.conv.weight.reshape([8 * 8 * 9]).unwrap().narrow(0, 0, 8).unwrap());
        let g = Grid::from_range(-1.0, 1.0);
        let grids = vec![g; 8];
        let h = hash_grids(&grids);
        let act = QActU8::new(q8::alloc(1, 8, 16), [1, 8, 4, 4], grids.into(), vec![g.zero; 8].into(), h);
        let first = exec.qconv(&conv, &[&act]).unwrap();
        let second = exec.qconv(&conv, &[&act]).unwrap();
        assert!(Arc::ptr_eq(&first, &second), "second qconv call must hit cache");

        // Aliased stem (`bias` is a view into `weight`'s own storage).
        let mut stem = Init::new(&mut rng).conv(3, 16, 3, 2);
        stem.fuse_().unwrap();
        stem.conv.bias = Some(stem.conv.weight.reshape([16 * 3 * 9]).unwrap().narrow(0, 0, 16).unwrap());
        let s1 = exec.qstem(&stem).unwrap();
        let s2 = exec.qstem(&stem).unwrap();
        assert!(Arc::ptr_eq(&s1, &s2), "second qstem call must hit cache");
    }
}
