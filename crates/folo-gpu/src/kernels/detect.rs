//! Detection kernels: the task-aligned assigner (docs/MODEL.md §7.3) and the fused detection loss (§7.4) on raw
//! buffers. [`crate::detect`] wraps them for folo-core tensors (with the loss's backward as a custom op).
//!
//! Buffers (all dense from byte 0, `B` images, `A` anchors, `NC` classes, `G` gt slots per image, `K = topk`,
//! `R = reg_max`):
//! - `box_raw [B, 4R, A]`, `cls_raw [B, NC, A]` f32: the head outputs.
//! - `meta` (u32 words, f32 by bit pattern): the anchors `(x, y, stride)` (grid units) at word 0, then the gt
//!   boxes `[B, G, 4]` (xyxy input pixels) at [`DetectDesc::gt_box_offset`], labels `[B, G]`, counts `[B]`.
//! - `pd [B, A, 4]` f32: decoded boxes (pixels). `pick [B, G, K, 3]`: per gt its top-k `(anchor or NONE, m, u)`.
//!   `own [2, B, A]`: the step-4 owner (max `u` bits, then the complemented gt index). `slot [B, G, K, 2]`:
//!   `(t or 0, positive)`. `dense [2, B, A]`: `(label + 1 or 0, t)` per anchor. `misc [4]`: bit 0 of word 0 is
//!   set when a head value is not finite.
//! - Loss: `part` (BCE partial sums), `terms [B·G·K, 2]`, `stats [12]` f32 and `total [1]` (see
//!   `shaders/det_combine.wgsl`).

use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

const COMMON: &str = include_str!("../shaders/detect_common.wgsl");

macro_rules! detect_kernel {
    ($id:ident, $name:literal, $file:literal, [$($acc:ident),*]) => {
        static $id: KernelDef = KernelDef {
            name: $name,
            storage: &[$(Access::$acc),*],
            source: Source::Generated(|_| {
                // The shortest decimal that round-trips, so WGSL parses back the same f32.
                let vk = v_k();
                format!("const V_K: f32 = {vk:?}f;\n{COMMON}{}", include_str!(concat!("../shaders/", $file)))
            }),
        };
    };
}

detect_kernel!(DECODE, "tal_decode", "tal_decode.wgsl", [Read, Read, ReadWrite, ReadWrite]);
detect_kernel!(TOPK, "tal_topk", "tal_topk.wgsl", [Read, Read, Read, ReadWrite]);
detect_kernel!(OWNER, "tal_owner", "tal_owner.wgsl", [Read, ReadWrite]);
detect_kernel!(TARGETS, "tal_targets", "tal_targets.wgsl", [Read, Read, Read, ReadWrite, ReadWrite]);
detect_kernel!(BCE_FWD, "det_bce_fwd", "det_bce_fwd.wgsl", [Read, Read, ReadWrite, ReadWrite]);
detect_kernel!(BCE_BWD, "det_bce_bwd", "det_bce_bwd.wgsl", [Read, Read, Read, Read, ReadWrite]);
detect_kernel!(BOX, "det_box", "det_box.wgsl", [Read, Read, Read, Read, ReadWrite, Read, Read]);
detect_kernel!(COMBINE, "det_combine", "det_combine.wgsl", [Read, Read, Read, Read, ReadWrite, ReadWrite]);

/// Largest `reg_max` (DFL bins per side) the loss kernels take.
pub const MAX_REG_MAX: usize = 32;

/// Largest `topk` the assigner kernels take.
pub const MAX_TOPK: usize = 64;

/// Anchors per BCE workgroup (4 per thread).
const BCE_CHUNK: usize = 1024;

/// `4 / π²` as the CPU CIoU computes it in f32.
pub(crate) fn v_k() -> f32 {
    4.0 / (std::f32::consts::PI * std::f32::consts::PI)
}

/// Problem sizes and settings of one assigner + loss call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectDesc {
    /// Images.
    pub b: usize,
    /// Anchors per image.
    pub a: usize,
    /// Classes.
    pub nc: usize,
    /// Gt slots per image (the largest gt count of the batch).
    pub g: usize,
    /// Candidates kept per gt (`topk`, at most [`MAX_TOPK`]).
    pub k: usize,
    /// Positives each gt keeps after conflict resolution, its highest-metric ones (`topk2`, the one-to-one
    /// assignment of docs/MODEL.md §6.6); `0` keeps them all.
    pub k2: usize,
    /// DFL bins per side.
    pub reg_max: usize,
    /// Exponent of the score in the alignment metric.
    pub alpha: f32,
    /// Exponent of the IoU in the alignment metric.
    pub beta: f32,
    /// Inside-box margin and normaliser epsilon.
    pub eps: f32,
    /// Candidate boxes widen sides shorter than this (pixels).
    pub min_box_size: f32,
}

impl DetectDesc {
    /// Word offset of the gt boxes in `meta`.
    pub fn gt_box_offset(&self) -> usize {
        3 * self.a
    }

    /// Word offset of the gt labels in `meta`.
    pub fn gt_label_offset(&self) -> usize {
        self.gt_box_offset() + 4 * self.b * self.g
    }

    /// Word offset of the per-image gt counts in `meta`.
    pub fn gt_count_offset(&self) -> usize {
        self.gt_label_offset() + self.b * self.g
    }

    /// Words of `meta`.
    pub fn meta_words(&self) -> usize {
        self.gt_count_offset() + self.b
    }

    /// Pick slots, `B · G · K`.
    pub fn slots(&self) -> usize {
        self.b * self.g * self.k
    }

    /// BCE partial sums written by [`bce_forward`].
    pub fn bce_partials(&self) -> usize {
        self.b * self.nc * self.a.div_ceil(BCE_CHUNK)
    }

    fn validate(&self) -> Result<()> {
        if self.k == 0 || self.k > MAX_TOPK {
            return Err(shape_err(format!("detect: topk {} must be in 1..={MAX_TOPK}", self.k)));
        }
        if !(2..=MAX_REG_MAX).contains(&self.reg_max) {
            return Err(shape_err(format!("detect: reg_max {} must be in 2..={MAX_REG_MAX}", self.reg_max)));
        }
        if self.g > 65535 || self.b > 65535 || self.b * self.nc > 65535 {
            return Err(shape_err(format!(
                "detect: {} images x {} gts x {} classes exceed one dispatch",
                self.b, self.g, self.nc
            )));
        }
        let words = self.meta_words().max(self.b * 4 * self.reg_max * self.a).max(self.b * self.nc * self.a);
        if words >= u32::MAX as usize / 4 {
            return Err(shape_err("detect: problem too large for 32-bit indexing"));
        }
        Ok(())
    }

    /// `(workgroup size, alpha is 0.5, beta as an integer or 0)` of the top-k kernel: the largest power-of-two
    /// workgroup up to 128 whose lists (`WG · K` entries of 12 bytes) and three per-thread words fit in
    /// `max_storage` bytes of workgroup memory.
    fn topk_spec(&self, max_storage: u32) -> Result<Spec> {
        let bytes = |wg: usize| wg * self.k * 12 + wg * 12;
        let wg = [128usize, 64, 32, 16, 8, 4, 2, 1]
            .into_iter()
            .find(|&wg| bytes(wg) <= max_storage as usize)
            .ok_or_else(|| {
                shape_err(format!(
                    "detect: topk {} does not fit {max_storage} bytes of workgroup memory",
                    self.k
                ))
            })?;
        let beta_int =
            if self.beta.fract() == 0.0 && (1.0..=16.0).contains(&self.beta) { self.beta as u32 } else { 0 };
        Ok(Spec::new()
            .u32("K", self.k as u32)
            .u32("WG", wg as u32)
            .bool("ALPHA_SQRT", self.alpha == 0.5)
            .u32("BETA_INT", beta_int))
    }
}

/// The assigner: decode, top-k per gt, conflict resolution and soft targets. `own`, `dense` and `misc` must be
/// zeroed by the caller (they are encoded in order after the clears).
#[allow(clippy::too_many_arguments)]
pub fn assign(
    ctx: &GpuContext,
    d: &DetectDesc,
    box_raw: &wgpu::Buffer,
    cls_raw: &wgpu::Buffer,
    meta: &wgpu::Buffer,
    pd: &wgpu::Buffer,
    pick: &wgpu::Buffer,
    own: &wgpu::Buffer,
    slot: &wgpu::Buffer,
    dense: &wgpu::Buffer,
    misc: &wgpu::Buffer,
) -> Result<()> {
    d.validate()?;
    let ba = d.b * d.a;
    if ba == 0 {
        return Ok(());
    }
    let r = Spec::new().u32("R", d.reg_max as u32);
    let k = ctx.kernel(&DECODE, r)?;
    let p = Params::new().uz(ba)?.uz(d.a)?;
    ctx.dispatch(&k, &[box_raw, meta, pd, misc], &p, ctx.grid_1d(ba as u64, 256)?)?;
    if d.slots() == 0 {
        return Ok(());
    }
    let k = ctx.kernel(&TOPK, d.topk_spec(ctx.caps().max_workgroup_storage)?)?;
    let p = Params::new()
        .uz(d.a)?
        .uz(d.nc)?
        .uz(d.g)?
        .uz(d.gt_box_offset())?
        .uz(d.gt_label_offset())?
        .uz(d.gt_count_offset())?
        .f(d.eps)
        .f(d.min_box_size)
        .f(d.alpha)
        .f(d.beta);
    ctx.dispatch(&k, &[pd, cls_raw, meta, pick], &p, [d.g as u32, d.b as u32, 1])?;
    let p = Params::new().uz(d.slots())?.uz(d.a)?.uz(d.g)?.uz(ba)?;
    for mode in 0..2 {
        let k = ctx.kernel(&OWNER, Spec::new().u32("MODE", mode).u32("K", d.k as u32))?;
        ctx.dispatch(&k, &[pick, own], &p, ctx.grid_1d(d.slots() as u64, 256)?)?;
    }
    let k = ctx.kernel(&TARGETS, Spec::new().u32("K", d.k as u32).u32("K2", d.k2 as u32))?;
    let p = Params::new().uz(d.a)?.uz(d.g)?.uz(ba)?.uz(d.gt_label_offset())?.f(d.eps);
    ctx.dispatch(&k, &[pick, own, meta, slot, dense], &p, [d.g as u32, d.b as u32, 1])
}

/// Σ BCE partial sums of the classification term into `part` ([`DetectDesc::bce_partials`] floats).
pub fn bce_forward(
    ctx: &GpuContext,
    d: &DetectDesc,
    cls_raw: &wgpu::Buffer,
    dense: &wgpu::Buffer,
    part: &wgpu::Buffer,
    misc: &wgpu::Buffer,
) -> Result<()> {
    let chunks = d.a.div_ceil(BCE_CHUNK);
    let k = ctx.kernel(&BCE_FWD, Spec::new())?;
    let p = Params::new().uz(d.a)?.uz(d.nc)?.uz(chunks)?.uz(d.b * d.a)?;
    ctx.dispatch(&k, &[cls_raw, dense, part, misc], &p, [chunks as u32, (d.b * d.nc) as u32, 1])
}

/// The gradient of the total with respect to `cls_raw` into `dcls` (`[B, NC, A]`).
#[allow(clippy::too_many_arguments)]
pub fn bce_backward(
    ctx: &GpuContext,
    d: &DetectDesc,
    cls_raw: &wgpu::Buffer,
    dense: &wgpu::Buffer,
    stats: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    dcls: &wgpu::Buffer,
) -> Result<()> {
    let n = d.b * d.nc * d.a;
    let k = ctx.kernel(&BCE_BWD, Spec::new())?;
    let p = Params::new().uz(d.a)?.uz(d.nc)?.uz(n)?.uz(d.b * d.a)?;
    ctx.dispatch(&k, &[cls_raw, dense, stats, grad, dcls], &p, ctx.grid_1d(n as u64, 256)?)
}

/// The box terms: `backward = false` writes the per-slot `(box, dfl)` terms into `out`; `backward = true` writes
/// the gradient with respect to `box_raw` into `out` (`[B, 4R, A]`, zeroed by the caller), reading the scales in
/// `stats` and the upstream gradient `grad`.
#[allow(clippy::too_many_arguments)]
pub fn box_terms(
    ctx: &GpuContext,
    d: &DetectDesc,
    backward: bool,
    box_raw: &wgpu::Buffer,
    meta: &wgpu::Buffer,
    pick: &wgpu::Buffer,
    slot: &wgpu::Buffer,
    out: &wgpu::Buffer,
    stats: &wgpu::Buffer,
    grad: &wgpu::Buffer,
) -> Result<()> {
    if d.slots() == 0 {
        return Ok(());
    }
    let spec = Spec::new().u32("R", d.reg_max as u32).u32("K", d.k as u32).u32("MODE", backward as u32);
    let k = ctx.kernel(&BOX, spec)?;
    let hi = d.reg_max as f32 - 1.0 - 0.01;
    let p = Params::new().uz(d.slots())?.uz(d.a)?.uz(d.g)?.uz(d.gt_box_offset())?.f(hi);
    ctx.dispatch(&k, &[box_raw, meta, pick, slot, out, stats, grad], &p, ctx.grid_1d(d.slots() as u64, 64)?)
}

/// The final reduction into `stats` and `total` (see the module docs).
#[allow(clippy::too_many_arguments)]
pub fn combine(
    ctx: &GpuContext,
    d: &DetectDesc,
    gains: [f32; 3],
    part: &wgpu::Buffer,
    terms: &wgpu::Buffer,
    slot: &wgpu::Buffer,
    misc: &wgpu::Buffer,
    stats: &wgpu::Buffer,
    total: &wgpu::Buffer,
) -> Result<()> {
    let k = ctx.kernel(&COMBINE, Spec::new())?;
    let p =
        Params::new().uz(d.bce_partials())?.uz(d.slots())?.f(d.b as f32).f(gains[0]).f(gains[1]).f(gains[2]);
    ctx.dispatch(&k, &[part, terms, slot, misc, stats, total], &p, [1, 1, 1])
}

static CANDIDATES: KernelDef = KernelDef {
    name: "det_candidates",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/det_candidates.wgsl")),
};

/// Words of one image's region in the output of [`multi_label_candidates`]: a count, then `cap` records.
pub fn candidate_region_words(cap: usize) -> usize {
    1 + CANDIDATE_RECORD_WORDS * cap
}

/// Words per record of [`multi_label_candidates`]: `(anchor, class, score bits)`.
pub const CANDIDATE_RECORD_WORDS: usize = 3;

/// The multi-label candidates of decoded detections: `decoded` is f32 `[B, 4 + nc, A]` (boxes, then class scores,
/// channel-major), `out` u32 `[B, 1 + 3 · cap]`. Per image: the number of `(anchor, class)` pairs whose score is
/// `> conf` (all of them, also past `cap`), then the first `cap` of them as `(anchor, class, score bits)` in
/// arrival order (any order: the host sorts). The scores are copied, not recomputed, so they are the decoded
/// values bit for bit. `out` is cleared first.
#[allow(clippy::too_many_arguments)]
pub fn multi_label_candidates(
    ctx: &GpuContext,
    decoded: &wgpu::Buffer,
    out: &wgpu::Buffer,
    batch: usize,
    anchors: usize,
    nc: usize,
    cap: usize,
    conf: f32,
) -> Result<()> {
    if anchors == 0 || nc == 0 || cap == 0 {
        return Err(shape_err("multi_label_candidates: empty head or capacity"));
    }
    let overflow = || shape_err("multi_label_candidates: too large");
    let ba = batch.checked_mul(anchors).ok_or_else(overflow)?;
    let dec = ba.checked_mul(4 + nc).ok_or_else(overflow)?;
    let words = batch.checked_mul(candidate_region_words(cap)).ok_or_else(overflow)?;
    // Every index the shader forms must fit in u32.
    if dec.max(words) > u32::MAX as usize {
        return Err(overflow());
    }
    super::check_buf(decoded, dec, crate::DType::F32, "candidates decoded")?;
    super::check_buf(out, words, crate::DType::F32, "candidates output")?;
    if batch == 0 {
        return Ok(());
    }
    ctx.clear(out, 4 * words as u64);
    let k = ctx.kernel(&CANDIDATES, Spec::new())?;
    let p = Params::new().uz(ba)?.uz(anchors)?.uz(nc)?.uz(cap)?.f(conf);
    ctx.dispatch(&k, &[decoded, out], &p, ctx.grid_1d(ba as u64, 256)?)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] =
    &[&DECODE, &TOPK, &OWNER, &TARGETS, &BCE_FWD, &BCE_BWD, &BOX, &COMBINE, &CANDIDATES];

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(k: usize) -> DetectDesc {
        DetectDesc {
            b: 1,
            a: 100,
            nc: 2,
            g: 1,
            k,
            k2: 0,
            reg_max: 16,
            alpha: 0.5,
            beta: 6.0,
            eps: 1e-9,
            min_box_size: 16.0,
        }
    }

    /// The top-k workgroup shrinks until its lists fit the adapter's workgroup memory (16 KiB is the minimum FOLO
    /// accepts), and every topk up to MAX_TOPK fits it.
    #[test]
    fn topk_workgroup_fits_workgroup_memory() {
        for (k, storage, want) in
            [(10, 48 * 1024, 128), (10, 16 * 1024, 64), (64, 16 * 1024, 16), (17, 49152, 128)]
        {
            let spec = desc(k).topk_spec(storage).unwrap();
            let wg = spec.get_u32("WG", 0) as usize;
            assert_eq!(wg, want, "k {k}, {storage} bytes");
            assert!(wg * k * 12 + wg * 12 <= storage as usize);
        }
        for k in 1..=MAX_TOPK {
            assert!(desc(k).topk_spec(16 * 1024).is_ok(), "k {k}");
        }
    }
}
