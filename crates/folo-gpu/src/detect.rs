//! The task-aligned assigner (docs/MODEL.md §7.3) and the detection loss (§7.4) on folo-core GPU tensors, so a
//! training step never downloads the head outputs.
//!
//! [`assign`] runs the assigner kernels of [`crate::kernels::detect`] on the raw head outputs; [`detection_loss`]
//! adds the fused classification (BCE) and box (CIoU + DFL) terms and returns the total as a folo-core custom op
//! whose backward is two more kernels, plus a small `stats` tensor (the reported items, the positive count and
//! the target score sum). Nothing here waits for the GPU: the caller reads `stats` once (the loss's only
//! readback).
//!
//! The math follows folo-vision's host implementation step by step (`folo_vision::assign` and
//! `folo_vision::loss`), with the transcendental functions written out in plain arithmetic (see
//! `shaders/detect_common.wgsl`), so decisions agree with the CPU and values agree to float rounding.
//! Differences: the positives are listed per gt slot (`[B, G, K]`), not per anchor, and the sums run in a
//! different (but fixed) order.

use std::sync::Arc;

use folo_core::{DType, Device, Error, Result, Tensor, no_grad};

use crate::backend::{BACKEND_NAME, context_for, dense_buffer, tensor_from_buffer};
use crate::context::GpuContext;
use crate::kernels::detect::{self as k, DetectDesc};
use crate::pool::GpuBuffer;

pub use crate::kernels::detect::{CANDIDATE_RECORD_WORDS, MAX_REG_MAX, MAX_TOPK, candidate_region_words};

/// Assigner settings (folo-vision's `TalConfig`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TalSettings {
    /// Candidates kept per gt (at most [`MAX_TOPK`]).
    pub topk: usize,
    /// Positives each gt keeps after conflict resolution (its highest-metric ones; folo-vision's
    /// `TalConfig::topk2`); `None` keeps them all.
    pub topk2: Option<usize>,
    /// Exponent of the score in the alignment metric.
    pub alpha: f32,
    /// Exponent of the IoU in the alignment metric.
    pub beta: f32,
    /// Inside-box margin and normaliser epsilon.
    pub eps: f32,
    /// Candidate boxes widen sides shorter than this (pixels).
    pub min_box_size: f32,
    /// Epsilon-free TAL target normalization.
    pub eps_free: bool,
    /// Scale-adaptive NWD blending weight in TAL (`0.0` = legacy CIoU).
    pub nwd_weight: f32,
}

/// Which box regression geometry loss [`detection_loss`] uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoxLossKind {
    /// Complete IoU (default).
    #[default]
    Ciou,
    /// Rational-Focal Edge-Decoupled IoU (`R-PIoU`).
    Rpiou,
}

/// Loss settings (folo-vision's `LossConfig`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossSettings {
    /// DFL bins per side.
    pub reg_max: usize,
    /// Weights of the box, classification and DFL terms.
    pub gains: [f32; 3],
    /// Assigner settings.
    pub tal: TalSettings,
    /// Matchability-Aware Loss exponent on background (`0.0` = plain BCE).
    pub mal_gamma: f32,
    /// Scale-adaptive NWD blending weight in box loss (`0.0` = pure CIoU).
    pub nwd_weight: f32,
    /// Box regression geometry loss (`Ciou` or `Rpiou`).
    pub box_loss: BoxLossKind,
    /// High-IoU focusing slope `β` in `[0, 1]` for [`BoxLossKind::Rpiou`].
    pub rpiou_beta: f32,
    /// Rational edge-penalty sharpness `κ > 0` for [`BoxLossKind::Rpiou`].
    pub rpiou_kappa: f32,
    /// Grid-unit GT-size floor `δ₀ > 0` for [`BoxLossKind::Rpiou`].
    pub rpiou_delta: f32,
    /// Weight `λ_ot >= 0` of the 1D optimal-transport cumulative tail-barrier DFL term.
    pub dfl_ot_weight: f32,
}

/// The anchors: points in grid units and the stride of each.
#[derive(Clone, Copy, Debug)]
pub struct AnchorTable<'a> {
    /// `(x, y)` in grid units.
    pub points: &'a [[f32; 2]],
    /// Stride of each anchor (pixels).
    pub strides: &'a [f32],
}

/// The ground truth of one image: `xyxy` boxes in input pixels and their classes.
#[derive(Clone, Copy, Debug)]
pub struct ImageGt<'a> {
    /// Boxes.
    pub boxes: &'a [[f32; 4]],
    /// Class of each box.
    pub labels: &'a [u32],
}

/// The assigner's result on the device.
pub struct Assignment {
    /// Sizes.
    pub desc: DetectDesc,
    /// `[B, G, K, 3]` u32: per gt slot its top-k `(anchor or u32::MAX, m bits, u bits)`.
    pub picks: Tensor,
    /// `[B, G, K, 2]` u32: `(t bits or 0, 1 if the pick is a final positive)`.
    pub slots: Tensor,
    /// `[2, B, A]` u32: per anchor `label + 1` (0 = background), then `t` bits.
    pub dense: Tensor,
    pub(crate) meta: Tensor,
    misc: Tensor,
}

/// What [`detection_loss`] returns.
pub struct DeviceLoss {
    /// `B · (box + cls + dfl)`, rank 0, differentiable with respect to both head outputs. NaN if a head value
    /// was not finite.
    pub total: Tensor,
    /// `[12]` f32: total, box, cls, dfl (the reported items), positives, Σ t, non-finite flag, then the backward
    /// scales (see `shaders/det_combine.wgsl`). A copy: the backward reads its own, private buffers, which the
    /// caller cannot reach (use [`assign`] to inspect an assignment).
    pub stats: Tensor,
}

/// Whether `device` is served by a folo-gpu context (so [`assign`] and [`detection_loss`] can run there).
pub fn supported(device: Device) -> bool {
    context_for(device).is_some()
}

fn err(msg: impl Into<String>) -> Error {
    Error::InvalidArgument { op: "folo_gpu::detect", msg: msg.into() }
}

pub(crate) fn gpu(e: crate::GpuError) -> Error {
    Error::Backend { backend: BACKEND_NAME, msg: e.to_string() }
}

/// A tensor with its elements dense from byte 0 and its buffer.
pub(crate) type DenseInput = (Tensor, wgpu::Buffer);

/// A tensor with its elements dense from byte 0 (a copy when needed) and its buffer.
pub(crate) fn dense(t: &Tensor) -> Result<DenseInput> {
    let t = t.detach();
    let t = if dense_buffer(&t).is_some() {
        t
    } else {
        let _g = no_grad();
        let c = Tensor::zeros_with(t.shape(), t.dtype(), t.device())?;
        c.copy_from(&t)?;
        c
    };
    let b = dense_buffer(&t).ok_or_else(|| err("tensor is not on a folo-gpu device"))?;
    Ok((t, b))
}

pub(crate) struct Buffers {
    pub(crate) ctx: GpuContext,
    pub(crate) device: Device,
}

impl Buffers {
    pub(crate) fn alloc(&self, dtype: DType, shape: &[usize], zero: bool) -> Result<(Tensor, wgpu::Buffer)> {
        let n: usize = shape.iter().product();
        let bytes = (n * dtype.size_in_bytes()).max(4) as u64;
        let buf: GpuBuffer = self.ctx.alloc(bytes).map_err(gpu)?;
        if zero {
            self.ctx.clear(buf.raw(), bytes);
        }
        let raw = buf.raw().clone();
        Ok((tensor_from_buffer(self.device, dtype, shape, buf)?, raw))
    }
}

fn check(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &AnchorTable<'_>,
    gts: &[ImageGt<'_>],
    reg_max: usize,
    tal: &TalSettings,
) -> Result<DetectDesc> {
    for t in [box_raw, cls_raw] {
        if t.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch {
                op: "folo_gpu::detect",
                expected: DType::F32,
                got: t.dtype(),
            });
        }
    }
    if box_raw.device() != cls_raw.device() {
        return Err(Error::DeviceMismatch {
            op: "folo_gpu::detect",
            lhs: box_raw.device(),
            rhs: cls_raw.device(),
        });
    }
    let ((b, ch, a), (b2, nc, a2)) = match (box_raw.dims3(), cls_raw.dims3()) {
        (Ok(x), Ok(y)) => (x, y),
        _ => return Err(err("box_raw and cls_raw must be rank 3")),
    };
    if ch != 4 * reg_max || b2 != b || a2 != a || nc == 0 || a == 0 {
        return Err(err(format!(
            "box_raw {:?} / cls_raw {:?} must be [B, {}, A] and [B, nc >= 1, A]",
            box_raw.dims(),
            cls_raw.dims(),
            4 * reg_max
        )));
    }
    if anchors.points.len() != a || anchors.strides.len() != a {
        return Err(err(format!(
            "{} anchor points / {} strides for A = {a}",
            anchors.points.len(),
            anchors.strides.len()
        )));
    }
    if gts.len() != b {
        return Err(err(format!("{} gt images for a batch of {b}", gts.len())));
    }
    for (i, g) in gts.iter().enumerate() {
        if g.boxes.len() != g.labels.len() {
            return Err(err(format!("image {i}: {} boxes but {} labels", g.boxes.len(), g.labels.len())));
        }
        if let Some(&l) = g.labels.iter().find(|&&l| l as usize >= nc) {
            return Err(err(format!("image {i}: class {l} >= nc {nc}")));
        }
    }
    if tal.topk == 0 || tal.topk > MAX_TOPK {
        return Err(err(format!("topk {} must be in 1..={MAX_TOPK}", tal.topk)));
    }
    if tal.topk2 == Some(0) {
        return Err(err("topk2 must be at least 1"));
    }
    if !(tal.nwd_weight.is_finite() && (0.0..=1.0).contains(&tal.nwd_weight)) {
        return Err(err(format!("tal.nwd_weight {} must be finite and in [0, 1]", tal.nwd_weight)));
    }
    Ok(DetectDesc {
        b,
        a,
        nc,
        g: gts.iter().map(|g| g.boxes.len()).max().unwrap_or(0),
        k: tal.topk,
        // A second filter at least as wide as the first keeps everything.
        k2: tal.topk2.filter(|&k2| k2 < tal.topk).unwrap_or(0),
        reg_max,
        alpha: tal.alpha,
        beta: tal.beta,
        eps: tal.eps,
        min_box_size: tal.min_box_size,
        eps_free: tal.eps_free,
        tal_nwd_weight: tal.nwd_weight,
        mal_gamma: 0.0,
        box_nwd_weight: 0.0,
        box_loss_kind: 0,
        rpiou_beta: 0.5,
        rpiou_kappa: 2.0,
        rpiou_delta: 0.5,
        dfl_ot_weight: 0.0,
        box_gain: 0.0,
        dfl_gain: 0.0,
    })
}

/// The per-step GT tail of `meta` (after the cached `3 * A` anchor words): boxes, labels and counts.
fn gt_tail_words(d: &DetectDesc, gts: &[ImageGt<'_>]) -> Vec<u32> {
    let base = d.gt_box_offset();
    let mut tail = vec![0u32; d.meta_words() - base];
    let label_off = d.gt_label_offset() - base;
    let count_off = d.gt_count_offset() - base;
    for (i, g) in gts.iter().enumerate() {
        for (j, (bx, &l)) in g.boxes.iter().zip(g.labels).enumerate() {
            let slot = i * d.g + j;
            for (c, v) in bx.iter().enumerate() {
                tail[4 * slot + c] = v.to_bits();
            }
            tail[label_off + slot] = l;
        }
        tail[count_off + i] = g.boxes.len() as u32;
    }
    tail
}

fn assign_impl(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &AnchorTable<'_>,
    gts: &[ImageGt<'_>],
    reg_max: usize,
    tal: &TalSettings,
) -> Result<(Buffers, Assignment, DenseInput, DenseInput)> {
    let d = check(box_raw, cls_raw, anchors, gts, reg_max, tal)?;
    d.validate().map_err(gpu)?;
    let device = box_raw.device();
    let ctx = context_for(device).ok_or_else(|| err(format!("{device} is not a folo-gpu device")))?;
    let bufs = Buffers { ctx, device };
    let (bt, box_buf) = dense(box_raw)?;
    let (ct, cls_buf) = dense(cls_raw)?;
    let anchor_bytes = (d.gt_box_offset() * 4) as u64;
    let total_words = d.meta_words();
    let anchor_buf = bufs.ctx.cached_anchors(anchors.points, anchors.strides).map_err(gpu)?;
    let gt_tail = gt_tail_words(&d, gts);
    let meta_buf = bufs.ctx.alloc((total_words * 4).max(4) as u64).map_err(gpu)?;
    if !gt_tail.is_empty() {
        bufs.ctx.write(&meta_buf, anchor_bytes, bytemuck::cast_slice(&gt_tail)).map_err(gpu)?;
    }
    bufs.ctx.copy(&anchor_buf, 0, meta_buf.raw(), 0, anchor_bytes);
    let meta_raw = meta_buf.raw().clone();
    let meta = tensor_from_buffer(device, DType::U32, [total_words], meta_buf)?;
    let (_pd, pd) = bufs.alloc(DType::F32, &[d.b, d.a, 4], false)?;
    let (picks, pick) = bufs.alloc(DType::U32, &[d.b, d.g, d.k, 3], false)?;
    let (_own, own) = bufs.alloc(DType::U32, &[2, d.b, d.a], true)?;
    let (slots, slot) = bufs.alloc(DType::U32, &[d.b, d.g, d.k, 2], false)?;
    let (dense_t, dense_b) = bufs.alloc(DType::U32, &[2, d.b, d.a], true)?;
    let (misc, misc_b) = bufs.alloc(DType::U32, &[4], true)?;
    k::assign(&bufs.ctx, &d, &box_buf, &cls_buf, &meta_raw, &pd, &pick, &own, &slot, &dense_b, &misc_b)
        .map_err(gpu)?;
    let asg = Assignment { desc: d, picks, slots, dense: dense_t, meta, misc };
    Ok((bufs, asg, (bt, box_buf), (ct, cls_buf)))
}

/// Run the task-aligned assigner on the raw head outputs (`box_raw [B, 4·reg_max, A]`, `cls_raw [B, nc, A]`,
/// both on a folo-gpu device), the anchors and each image's ground truth. Errors on shape mismatches, a label
/// `>= nc`, or `topk` outside `1..=MAX_TOPK`.
pub fn assign(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &AnchorTable<'_>,
    gts: &[ImageGt<'_>],
    reg_max: usize,
    tal: &TalSettings,
) -> Result<Assignment> {
    Ok(assign_impl(box_raw, cls_raw, anchors, gts, reg_max, tal)?.1)
}

/// Everything the backward reads, kept alive by the custom op.
struct Saved {
    ctx: GpuContext,
    device: Device,
    desc: DetectDesc,
    box_raw: Tensor,
    cls_raw: Tensor,
    meta: Tensor,
    picks: Tensor,
    slots: Tensor,
    dense: Tensor,
    stats: Tensor,
}

impl Saved {
    fn backward(&self, g: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let d = &self.desc;
        let (_g, gbuf) = dense(g)?;
        let buf = |t: &Tensor| dense_buffer(t).ok_or_else(|| err("saved tensor is not dense"));
        let bufs = Buffers { ctx: self.ctx.clone(), device: self.device };
        let (dbox, dbox_b) = bufs.alloc(DType::F32, &[d.b, 4 * d.reg_max, d.a], true)?;
        k::box_terms(
            &self.ctx,
            d,
            true,
            &buf(&self.box_raw)?,
            &buf(&self.meta)?,
            &buf(&self.picks)?,
            &buf(&self.slots)?,
            &dbox_b,
            &buf(&self.stats)?,
            &gbuf,
        )
        .map_err(gpu)?;
        let (dcls, dcls_b) = bufs.alloc(DType::F32, &[d.b, d.nc, d.a], false)?;
        k::bce_backward(
            &self.ctx,
            d,
            &buf(&self.cls_raw)?,
            &buf(&self.dense)?,
            &buf(&self.stats)?,
            &gbuf,
            &dcls_b,
        )
        .map_err(gpu)?;
        // No gradient for the saved assignment buffers (inputs only so autograd checks their storage versions).
        Ok(vec![Some(dbox), Some(dcls), None, None, None, None])
    }
}

/// The detection loss of docs/MODEL.md §7 on the device: [`assign`], then the classification, box and DFL terms
/// and their weighted, batch-scaled total. See [`DeviceLoss`]; read `stats` for the values.
pub fn detection_loss(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &AnchorTable<'_>,
    gts: &[ImageGt<'_>],
    s: &LossSettings,
) -> Result<DeviceLoss> {
    Ok(detection_loss_with_assignment(box_raw, cls_raw, anchors, gts, s)?.0)
}

/// [`detection_loss`], also returning the device [`Assignment`] so distillation (`bckd_gamma`, `gt_verify`)
/// can reuse it without running the assigner twice.
pub fn detection_loss_with_assignment(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &AnchorTable<'_>,
    gts: &[ImageGt<'_>],
    s: &LossSettings,
) -> Result<(DeviceLoss, Assignment)> {
    if !(s.mal_gamma.is_finite() && s.mal_gamma >= 0.0) {
        return Err(err(format!("mal_gamma {} must be finite and >= 0", s.mal_gamma)));
    }
    if !(s.nwd_weight.is_finite() && (0.0..=1.0).contains(&s.nwd_weight)) {
        return Err(err(format!("nwd_weight {} must be finite and in [0, 1]", s.nwd_weight)));
    }
    if !(s.rpiou_beta.is_finite() && (0.0..=1.0).contains(&s.rpiou_beta)) {
        return Err(err(format!("rpiou_beta {} must be finite and in [0, 1]", s.rpiou_beta)));
    }
    if !(s.rpiou_kappa.is_finite() && s.rpiou_kappa > 0.0) {
        return Err(err(format!("rpiou_kappa {} must be finite and > 0", s.rpiou_kappa)));
    }
    if !(s.rpiou_delta.is_finite() && s.rpiou_delta > 0.0) {
        return Err(err(format!("rpiou_delta {} must be finite and > 0", s.rpiou_delta)));
    }
    if !(s.dfl_ot_weight.is_finite() && s.dfl_ot_weight >= 0.0) {
        return Err(err(format!("dfl_ot_weight {} must be finite and >= 0", s.dfl_ot_weight)));
    }
    let (bufs, asg, (box_t, box_buf), (cls_t, cls_buf)) =
        assign_impl(box_raw, cls_raw, anchors, gts, s.reg_max, &s.tal)?;
    let mut d = asg.desc;
    d.mal_gamma = s.mal_gamma;
    d.box_nwd_weight = s.nwd_weight;
    d.box_loss_kind = match s.box_loss {
        BoxLossKind::Ciou => 0,
        BoxLossKind::Rpiou => 1,
    };
    d.rpiou_beta = s.rpiou_beta;
    d.rpiou_kappa = s.rpiou_kappa;
    d.rpiou_delta = s.rpiou_delta;
    d.dfl_ot_weight = s.dfl_ot_weight;
    d.box_gain = s.gains[0];
    d.dfl_gain = s.gains[2];
    let ctx = &bufs.ctx;
    let raw = |t: &Tensor| dense_buffer(t).ok_or_else(|| err("assignment tensor is not dense"));
    let (dense_b, misc_b, slot_b, pick_b, meta_b) =
        (raw(&asg.dense)?, raw(&asg.misc)?, raw(&asg.slots)?, raw(&asg.picks)?, raw(&asg.meta)?);
    let (_part, part) = bufs.alloc(DType::F32, &[d.bce_partials()], false)?;
    k::bce_forward(ctx, &d, &cls_buf, &dense_b, &part, &misc_b).map_err(gpu)?;
    let (_terms, terms) = bufs.alloc(DType::F32, &[d.slots(), 4], false)?;
    let dummy = ctx.dummy().clone();
    k::box_terms(ctx, &d, false, &box_buf, &meta_b, &pick_b, &slot_b, &terms, &dummy, &dummy).map_err(gpu)?;
    let (stats, stats_b) = bufs.alloc(DType::F32, &[12], false)?;
    let (total, total_b) = bufs.alloc(DType::F32, &[], false)?;
    k::combine(ctx, &d, s.gains, &part, &terms, &slot_b, &misc_b, &stats_b, &total_b).map_err(gpu)?;
    let saved = Arc::new(Saved {
        ctx: ctx.clone(),
        device: bufs.device,
        desc: d,
        box_raw: box_t,
        cls_raw: cls_t,
        meta: asg.meta.clone(),
        picks: asg.picks.clone(),
        slots: asg.slots.clone(),
        dense: asg.dense.clone(),
        stats: stats.clone(),
    });
    // Pass the publicly returned assignment tensors as no-grad inputs so autograd checks their storage versions
    // and refuses backward if a caller mutates them in place after forward.
    let total = Tensor::custom_op(
        "detection_loss_gpu",
        &[box_raw, cls_raw, &asg.meta, &asg.picks, &asg.slots, &asg.dense],
        total,
        move |g| saved.backward(g),
    )?;
    // The caller gets a copy of the stats, so nothing it does to them can change the backward.
    let (public, public_b) = bufs.alloc(DType::F32, &[12], false)?;
    ctx.copy(&stats_b, 0, &public_b, 0, 48);
    Ok((DeviceLoss { total, stats: public }, asg))
}

/// The multi-label candidates of a decoded head (validation's NMS input) on the device, so only they are
/// downloaded: `decoded` is f32 `[B, 4 + nc, A]` (`x1, y1, x2, y2`, then the class scores, channel-major) on a
/// folo-gpu device; the result is u32 `[B, 1 + 3 · cap]` ([`crate::kernels::detect::multi_label_candidates`]:
/// per image the count of `(anchor, class)` pairs with a score `> conf`, then up to `cap` records `(anchor, class,
/// score bits)` in any order). Nothing waits for the GPU.
pub fn multi_label_candidates(decoded: &Tensor, conf: f32, cap: usize) -> Result<Tensor> {
    let [b, rows, a] = *decoded.dims() else {
        return Err(err(format!("candidates: decoded must be [B, 4 + nc, A], got {:?}", decoded.dims())));
    };
    if rows <= 4 || decoded.dtype() != DType::F32 {
        return Err(err(format!("candidates: decoded must be f32 [B, 4 + nc, A], got {:?}", decoded.dims())));
    }
    let device = decoded.device();
    let ctx = context_for(device).ok_or_else(|| err(format!("{device} is not a folo-gpu device")))?;
    let bufs = Buffers { ctx, device };
    let (_keep, dec) = dense(decoded)?;
    let (out, out_b) = bufs.alloc(DType::U32, &[b, candidate_region_words(cap)], false)?;
    k::multi_label_candidates(&bufs.ctx, &dec, &out_b, b, a, rows - 4, cap, conf).map_err(gpu)?;
    Ok(out)
}
