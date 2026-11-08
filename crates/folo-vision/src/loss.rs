//! The detection loss of `docs/MODEL.md` §7: task-aligned assignment, then BCE, CIoU and DFL on folo-core tensors.
//!
//! # Inputs
//!
//! - `box_raw: [B, 4 * reg_max, A]` and `cls_raw: [B, nc, A]`, the head's raw training outputs (§6.3: channel
//!   `side * reg_max + bin`, side order `l t r b`; anchors level-major, row-major within a level).
//! - The [`Anchors`] of the input size (grid-unit points and strides, §6.2).
//! - The ground truth of each image as [`ImageTargets`] in input pixels; [`targets_from_normalized`] converts
//!   folo-data's `[batch_idx, class, cx, cy, w, h]` rows.
//!
//! # Steps (map to the spec)
//!
//! 1. [`assign_batch`] (§7.1, §7.3), **detached, on the host**: the raw outputs are read once as `&[f32]`
//!    (borrowed on the CPU, downloaded otherwise: the loss's only full-size device-to-host transfer; on a device the
//!    CIoU `atan` also round-trips `N_fg` values and the reported items read back four scalars), each image's boxes
//!    are DFL-decoded to pixels and its class scores of the classes present are passed through a sigmoid, and
//!    [`assign`] runs per image (images in parallel). The result, [`BatchAssignment`], lists the foreground
//!    anchors of the whole batch in a fixed order (image, then anchor).
//! 2. [`loss_with_assignment`] (§7.4) builds the differentiable terms. The foreground entries are gathered once by
//!    flat index (no per-anchor tensor work):
//!    - `L_cls = (Σ_bg softplus(x) + Σ_fg (softplus(x) − x · t)) / tss` over `x = cls_raw`, which equals
//!      `Σ BCE(cls_raw, T) / tss` because `T` is zero except at the assigned class of foreground anchors. Each
//!      foreground term is formed before summing, so confident positives do not cancel.
//!    - `pd_box` (grid units) = anchor ∓ `Σ_i i · softmax(logits)_i`, and `L_box = Σ_fg t · (1 − CIoU) / tss` with
//!      [`ciou`], the tensor form of [`crate::boxes::ciou_eps`] (α detached).
//!    - `L_dfl = Σ_fg t · DFL / tss` from `log_softmax` gathered at the two target bins.
//!    - `L = B · (7.5 · L_box + 0.5 · L_cls + 1.5 · L_dfl)`; [`LossItems`] reports the weighted terms without `B`.
//!
//!    On the CPU the two full-size steps (the BCE sum over `B · nc · A` logits and the gather of the foreground box
//!    logits) are fused custom ops with hand-written backwards; on other devices they are composites of tensor ops.
//!    Everything after the gather works on `[N_fg, ..]` tensors with ordinary differentiable ops.
//! 3. [`detection_loss`] runs both. If the head output holds a NaN or an infinity it skips the assigner and returns
//!    a non-finite [`LossOutput`] (see [`LossOutput::is_finite`]) so the trainer drops the batch (§8.3).
//!
//! Every sum over many elements is a fixed tree of per-row sums, so the loss and its gradients do not depend on
//! thread scheduling: repeated runs are bitwise identical.

use crate::assign::{TalConfig, assign, bbox2dist_clamped, dfl_bins};
use crate::boxes::Xyxy;
use crate::decode::{Anchors, dist2bbox, sigmoid};
use folo_core::cpu::fast_exp;
use folo_core::{DType, Error, Result, Tensor, no_grad};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const OP: &str = "detection_loss";

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument { op: OP, msg: msg.into() }
}

/// Loss hyper-parameters. Defaults are `docs/MODEL.md` §7.4's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LossConfig {
    /// Weight of the CIoU term.
    pub box_gain: f32,
    /// Weight of the classification BCE.
    pub cls_gain: f32,
    /// Weight of the distribution focal loss.
    pub dfl_gain: f32,
    /// DFL bins per side (the head's `reg_max`).
    pub reg_max: usize,
    /// Assigner settings.
    pub tal: TalConfig,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self { box_gain: 7.5, cls_gain: 0.5, dfl_gain: 1.5, reg_max: 16, tal: TalConfig::default() }
    }
}

/// The ground truth of one image: boxes in input pixels and their class ids.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageTargets {
    /// `xyxy` boxes in input pixels.
    pub boxes: Vec<Xyxy>,
    /// Class id of each box.
    pub labels: Vec<u32>,
}

impl ImageTargets {
    /// Number of boxes.
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    /// Whether the image has no boxes.
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
}

/// Group folo-data's target rows `[batch_idx, class, cx, cy, w, h]` (coordinates normalised to the network input)
/// by image and convert them to `xyxy` input pixels: `x` scales by `width`, `y` by `height` (§7).
///
/// Errors if a row's `batch_idx` or `class` is not a non-negative integer, `batch_idx >= batch`, a coordinate is
/// non-finite, or the box has `w <= 0` or `h <= 0` (folo-data drops degenerate boxes, so that is a bug upstream).
pub fn targets_from_normalized(
    rows: &[[f32; 6]],
    batch: usize,
    height: u32,
    width: u32,
) -> Result<Vec<ImageTargets>> {
    let (fw, fh) = (width as f32, height as f32);
    let mut out = vec![ImageTargets::default(); batch];
    for (r, row) in rows.iter().enumerate() {
        let &[bi, cls, cx, cy, w, h] = row;
        let is_index = |v: f32| v >= 0.0 && v.fract() == 0.0 && v < u32::MAX as f32;
        if !is_index(bi) || bi as usize >= batch {
            return Err(invalid(format!(
                "target row {r}: batch_idx {bi} is not an image index below {batch}"
            )));
        }
        if !is_index(cls) {
            return Err(invalid(format!("target row {r}: class {cls} is not a class id")));
        }
        if ![cx, cy, w, h].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
            return Err(invalid(format!("target row {r}: degenerate box cx {cx} cy {cy} w {w} h {h}")));
        }
        let (cx, cy, hw, hh) = (cx * fw, cy * fh, w * fw * 0.5, h * fh * 0.5);
        let img = &mut out[bi as usize];
        img.boxes.push(Xyxy::new(cx - hw, cy - hh, cx + hw, cy + hh));
        img.labels.push(cls as u32);
    }
    Ok(out)
}

/// The foreground anchors of a batch with their targets, in order of image, then anchor. Produced by
/// [`assign_batch`]; everything here is a constant for autograd.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchAssignment {
    /// Images in the batch (`B`).
    pub batch: usize,
    /// Anchors per image (`A`).
    pub num_anchors: usize,
    /// Classes (`nc`).
    pub num_classes: usize,
    /// Image of each foreground entry.
    pub image: Vec<u32>,
    /// Anchor index of each foreground entry.
    pub anchor: Vec<u32>,
    /// Assigned class.
    pub label: Vec<u32>,
    /// Soft target `t[a]` (§7.3 step 5).
    pub score: Vec<f32>,
    /// Assigned gt box in input pixels (§7.3 step 6, before the division by the stride).
    pub target: Vec<Xyxy>,
    /// Whether every value of `box_raw` and `cls_raw` was finite. When false the assigner did not run and there
    /// are no foreground entries.
    pub inputs_finite: bool,
}

impl BatchAssignment {
    /// Number of foreground anchors in the batch.
    pub fn num_fg(&self) -> usize {
        self.anchor.len()
    }

    /// The unfloored target-score sum `Σ T` over the batch (accumulated in `f64`, image-then-anchor order).
    pub fn target_score_sum(&self) -> f64 {
        self.score.iter().map(|&s| s as f64).sum()
    }
}

/// The reported loss items of §7.4: the weighted terms, **not** multiplied by the batch size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LossItems {
    /// `box_gain · L_box`.
    #[serde(rename = "box")]
    pub box_: f32,
    /// `cls_gain · L_cls`.
    pub cls: f32,
    /// `dfl_gain · L_dfl`.
    pub dfl: f32,
}

impl LossItems {
    /// `box + cls + dfl`.
    pub fn total(&self) -> f32 {
        self.box_ + self.cls + self.dfl
    }

    /// Whether all three items are finite.
    pub fn is_finite(&self) -> bool {
        self.box_.is_finite() && self.cls.is_finite() && self.dfl.is_finite()
    }
}

/// The result of [`detection_loss`] / [`loss_with_assignment`].
#[derive(Clone, Debug)]
pub struct LossOutput {
    /// `L = B · (box + cls + dfl)` as a rank-0 tensor on the head's device, differentiable with respect to
    /// `box_raw` and `cls_raw`. When the head output was non-finite it is a detached NaN: check
    /// [`LossOutput::is_finite`] before calling `backward()`.
    pub total: Tensor,
    /// The value of `total` (it can overflow to infinity even when every item is finite).
    pub total_value: f32,
    /// The detached, reported items (§7.4).
    pub items: LossItems,
    /// Foreground anchors in the batch.
    pub num_fg: usize,
    /// The unfloored target-score sum `Σ T` (the normaliser is `max(Σ T, 1)`).
    pub target_score_sum: f32,
}

impl LossOutput {
    /// Whether the loss is finite: the reported items and the batch-scaled `total`. The trainer skips backward
    /// otherwise (§8.3).
    pub fn is_finite(&self) -> bool {
        self.items.is_finite() && self.total_value.is_finite()
    }
}

/// Check the head outputs against the anchors and config; returns `(B, nc, A)`.
fn check_inputs(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    cfg: &LossConfig,
) -> Result<(usize, usize, usize)> {
    for t in [box_raw, cls_raw] {
        if t.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch { op: OP, expected: DType::F32, got: t.dtype() });
        }
    }
    if box_raw.device() != cls_raw.device() {
        return Err(Error::DeviceMismatch { op: OP, lhs: box_raw.device(), rhs: cls_raw.device() });
    }
    if cfg.reg_max < 2 {
        return Err(invalid(format!("reg_max must be >= 2, got {}", cfg.reg_max)));
    }
    let shapes = || format!("box_raw {}, cls_raw {}", box_raw.shape(), cls_raw.shape());
    let ((b, ch, a), (b2, nc, a2)) = match (box_raw.dims3(), cls_raw.dims3()) {
        (Ok(x), Ok(y)) => (x, y),
        _ => return Err(invalid(format!("{}: both must be rank 3", shapes()))),
    };
    if ch != 4 * cfg.reg_max || b2 != b || a2 != a || a != anchors.len() || nc == 0 || a == 0 {
        return Err(invalid(format!(
            "{} must be [B, {}, A] and [B, nc >= 1, A] with A = {} anchors (> 0)",
            shapes(),
            4 * cfg.reg_max,
            anchors.len()
        )));
    }
    Ok((b, nc, a))
}

/// Run `f` on the elements of a `F32` tensor as one contiguous host slice. This is the loss's device-to-host
/// boundary: CPU tensors are borrowed (copied only if strided), device tensors are downloaded.
fn host_f32<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> R) -> Result<R> {
    if t.device().is_cpu() {
        t.with_f32_slice(f)
    } else {
        let v = t.to_vec::<f32>()?;
        Ok(f(&v))
    }
}

/// DFL decode of one image's channel-major box logits `[4 · reg_max, A]` (§6.4) to boxes in input pixels. Loops
/// run over contiguous anchor rows so they vectorise; the same max-subtracted softmax as
/// [`crate::decode::softmax_expectation`].
fn decode_channel_major(reg: &[f32], anchors: &Anchors, reg_max: usize) -> Vec<Xyxy> {
    let n_a = anchors.len();
    let mut ltrb = vec![[0f32; 4]; n_a];
    let (mut m, mut z, mut e) = (vec![0f32; n_a], vec![0f32; n_a], vec![0f32; n_a]);
    for side in 0..4 {
        let bins = &reg[side * reg_max * n_a..(side + 1) * reg_max * n_a];
        m.fill(f32::NEG_INFINITY);
        for row in bins.chunks_exact(n_a) {
            m.iter_mut().zip(row).for_each(|(m, &x)| *m = m.max(x));
        }
        z.fill(0.0);
        e.fill(0.0);
        for (i, row) in bins.chunks_exact(n_a).enumerate() {
            let fi = i as f32;
            for (((z, e), &m), &x) in z.iter_mut().zip(e.iter_mut()).zip(&m).zip(row) {
                let p = fast_exp(x - m);
                *z += p;
                *e += p * fi;
            }
        }
        for ((d, &z), &e) in ltrb.iter_mut().zip(&z).zip(&e) {
            d[side] = e / z;
        }
    }
    ltrb.iter()
        .enumerate()
        .map(|(a, &d)| {
            let s = anchors.strides[a];
            dist2bbox(d, anchors.points[a]).scale(s, s)
        })
        .collect()
}

/// One image's foreground entries.
#[derive(Default)]
struct ImageFg {
    anchor: Vec<u32>,
    label: Vec<u32>,
    score: Vec<f32>,
    target: Vec<Xyxy>,
}

/// The detached part of the loss (§7.1, §7.3): decode the predictions on the host and run the task-aligned
/// assigner on every image.
///
/// Only the score columns of the classes present in an image are computed (the assigner reads no others), so the
/// host work is `O(B · A · (4 · reg_max + classes present))`, independent of `nc`. Errors on mismatched shapes,
/// `targets.len() != B`, or a gt label `>= nc`.
pub fn assign_batch(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<BatchAssignment> {
    let (b, nc, n_a) = check_inputs(box_raw, cls_raw, anchors, cfg)?;
    if targets.len() != b {
        return Err(invalid(format!("{} target images for a batch of {b}", targets.len())));
    }
    for (i, t) in targets.iter().enumerate() {
        if t.labels.len() != t.boxes.len() {
            return Err(invalid(format!("image {i}: {} boxes but {} labels", t.boxes.len(), t.labels.len())));
        }
        if let Some(&l) = t.labels.iter().find(|&&l| l as usize >= nc) {
            return Err(invalid(format!("image {i}: class {l} >= nc {nc}")));
        }
    }
    let reg_max = cfg.reg_max;
    let points_px = anchors.points_px();
    let (box_len, cls_len) = (4 * reg_max * n_a, nc * n_a);

    let per_image: Vec<(bool, ImageFg)> = host_f32(box_raw, |boxes| {
        host_f32(cls_raw, |cls| {
            (0..b)
                .into_par_iter()
                .map(|i| {
                    let reg = &boxes[i * box_len..(i + 1) * box_len];
                    let logits = &cls[i * cls_len..(i + 1) * cls_len];
                    let finite = reg.iter().chain(logits).all(|v| v.is_finite());
                    let gt = &targets[i];
                    if !finite || gt.is_empty() {
                        return (finite, ImageFg::default());
                    }
                    let pd_px = decode_channel_major(reg, anchors, reg_max);
                    // The assigner sees only the classes present, renumbered 0..k.
                    let mut classes = gt.labels.clone();
                    classes.sort_unstable();
                    classes.dedup();
                    let k = classes.len();
                    let local: Vec<u32> = gt
                        .labels
                        .iter()
                        .map(|l| classes.binary_search(l).expect("class present") as u32)
                        .collect();
                    let mut scores = vec![0f32; n_a * k];
                    for (j, &c) in classes.iter().enumerate() {
                        let row = &logits[c as usize * n_a..(c as usize + 1) * n_a];
                        for (s, &x) in scores.iter_mut().skip(j).step_by(k).zip(row) {
                            *s = sigmoid(x);
                        }
                    }
                    let asg = assign(&scores, k, &pd_px, &points_px, &gt.boxes, &local, &cfg.tal);
                    let mut fg = ImageFg::default();
                    for a in (0..n_a).filter(|&a| asg.fg_mask[a]) {
                        fg.anchor.push(a as u32);
                        fg.label.push(gt.labels[asg.target_gt[a]]);
                        fg.score.push(asg.target_scores[a]);
                        fg.target.push(asg.target_bboxes[a]);
                    }
                    (true, fg)
                })
                .collect()
        })
    })??;

    let mut out = BatchAssignment {
        batch: b,
        num_anchors: n_a,
        num_classes: nc,
        inputs_finite: per_image.iter().all(|(f, _)| *f),
        ..Default::default()
    };
    if out.inputs_finite {
        for (i, (_, fg)) in per_image.into_iter().enumerate() {
            out.image.extend(std::iter::repeat_n(i as u32, fg.anchor.len()));
            out.anchor.extend(fg.anchor);
            out.label.extend(fg.label);
            out.score.extend(fg.score);
            out.target.extend(fg.target);
        }
    }
    Ok(out)
}

/// Sum of every element as a rank-0 tensor, in an order that does not depend on thread scheduling: long inputs are
/// summed in rows of 4096 (each row serially), then the row sums the same way.
fn sum_fixed_order(x: &Tensor) -> Result<Tensor> {
    const ROW: usize = 4096;
    let n = x.numel();
    let flat = x.flatten_all()?;
    if n <= ROW {
        return flat.sum_all();
    }
    let rows = n.div_ceil(ROW);
    let padded = if rows * ROW > n { flat.pad(&[(0, rows * ROW - n)], 0.0)? } else { flat };
    sum_fixed_order(&padded.reshape([rows, ROW])?.sum(1, false)?)
}

/// `atan(x)` elementwise. folo-core has no `atan` kernel, so the forward runs on the host (the CIoU inputs are one
/// value per foreground anchor) and the backward `g / (1 + x²)` uses tensor ops on `x`'s device.
fn atan(x: &Tensor) -> Result<Tensor> {
    let v: Vec<f32> = x.to_vec::<f32>()?.into_iter().map(f32::atan).collect();
    let out = Tensor::from_vec(v, x.shape())?.to_device(x.device())?;
    let xs = x.detach();
    Tensor::custom_op("atan", &[x], out, move |g| Ok(vec![Some(g.div(&xs.square()?.add_scalar(1.0)?)?)]))
}

/// Rows per parallel task of [`bce_sum_cpu`].
const PAR_ROWS: usize = 8;

/// `softplus(x) = max(x, 0) + ln(1 + e^-|x|)`, as folo-core computes it.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + fast_exp(-x.abs()).ln_1p()
}

/// `Σ softplus(x)` with 8 independent accumulators (vectorises).
fn softplus_sum(r: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (chunks, rest) = r.as_chunks::<8>();
    for ch in chunks {
        for (a, &x) in acc.iter_mut().zip(ch) {
            *a += softplus(x);
        }
    }
    acc.iter().sum::<f32>() + rest.iter().map(|&x| softplus(x)).sum::<f32>()
}

/// Stable sigmoid, as folo-core computes it.
#[inline(always)]
fn sigmoid_fast(x: f32) -> f32 {
    let e = fast_exp(-x.abs());
    let r = 1.0 / (1.0 + e);
    if x >= 0.0 { r } else { e * r }
}

/// `Σ_i softplus(x_i) − Σ_k x[idx_k] · t_k` over a CPU tensor, i.e. `Σ BCE(x, T)` for a target `T` that is `t_k` at
/// the flat positions `idx_k` and 0 elsewhere, with the backward `g · (sigmoid(x) − T)` in one pass.
///
/// This fuses what the composite form does in several full-size passes (softplus, sums, and an `index_select`
/// whose backward materialises a dense zero gradient). `x` is viewed as rows of its last dimension; rows are
/// summed in parallel, each serially, and combined in row order in `f64`, so the result is reproducible.
fn bce_sum_cpu(x: &Tensor, idx: Vec<usize>, t: Vec<f32>) -> Result<Tensor> {
    let row = x.dims().last().copied().unwrap_or(1).max(1);
    // Foreground positions sorted, so each row finds its own with a binary search. They are distinct (checked by
    // the caller). The dense sum skips them and each gets its own `BCE(x, t)`: summing `softplus(x)` and `x · t`
    // separately would cancel catastrophically for confident positives (large `x`, `t ≈ 1`).
    let mut order: Vec<usize> = (0..idx.len()).collect();
    order.sort_unstable_by_key(|&k| idx[k]);
    let pos: Vec<usize> = order.iter().map(|&k| idx[k]).collect();
    let value = x.with_f32_slice(|v| {
        let rows: Vec<f32> = v
            .par_chunks(row * PAR_ROWS)
            .enumerate()
            .flat_map_iter(|(ci, c)| {
                let pos = &pos;
                c.chunks(row).enumerate().map(move |(ri, r)| {
                    let start = (ci * PAR_ROWS + ri) * row;
                    let lo = pos.partition_point(|&p| p < start);
                    let hi = pos.partition_point(|&p| p < start + r.len());
                    let mut s = 0f32;
                    let mut from = 0;
                    for &p in &pos[lo..hi] {
                        s += softplus_sum(&r[from..p - start]);
                        from = p - start + 1;
                    }
                    s + softplus_sum(&r[from..])
                })
            })
            .collect();
        let dense: f64 = rows.iter().map(|&r| r as f64).sum();
        // softplus(x) >= max(x, 0) and t in [0, 1], so every foreground term is >= 0.
        let fg: f64 =
            order.iter().map(|&k| softplus(v[idx[k]]) as f64 - v[idx[k]] as f64 * t[k] as f64).sum();
        (dense + fg) as f32
    })?;
    let xs = x.detach();
    Tensor::custom_op("bce_with_logits_sum", &[x], Tensor::scalar(value), move |g| {
        let g = g.item::<f32>()?;
        let mut grad = xs.with_f32_slice(|v| {
            let mut out = vec![0f32; v.len()];
            out.par_chunks_mut(row * PAR_ROWS).zip(v.par_chunks(row * PAR_ROWS)).for_each(|(o, x)| {
                for (o, &x) in o.iter_mut().zip(x) {
                    *o = g * sigmoid_fast(x);
                }
            });
            out
        })?;
        for (&i, &t) in idx.iter().zip(&t) {
            grad[i] -= g * t;
        }
        Ok(vec![Some(Tensor::from_vec(grad, xs.shape())?)])
    })
}

/// `x.flatten_all().index_select(0, idx)` for a CPU tensor, with a backward that scatters into a fresh zeroed
/// buffer (lazily zeroed pages) instead of copying a dense zero tensor.
fn gather_flat_cpu(x: &Tensor, idx: Vec<usize>) -> Result<Tensor> {
    let picked = x.with_f32_slice(|v| idx.iter().map(|&i| v[i]).collect::<Vec<f32>>())?;
    let out = Tensor::from_vec(picked, [idx.len()])?;
    let shape = x.shape().clone();
    Tensor::custom_op("gather_flat", &[x], out, move |g| {
        let mut grad = vec![0f32; shape.numel()];
        g.with_f32_slice(|g| {
            for (&i, &g) in idx.iter().zip(g) {
                grad[i] += g;
            }
        })?;
        Ok(vec![Some(Tensor::from_vec(grad, &shape)?)])
    })
}

/// Differentiable CIoU of `docs/MODEL.md` §7.2 between rows of `b1` and `b2` (`[N, 4]`, `xyxy`), as `[N]`.
///
/// The same formula and `eps` placements as [`crate::boxes::ciou_eps`] (which the assigner uses); α is computed
/// under `no_grad`, so no gradient flows through it. Gradients flow into both inputs.
pub fn ciou(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    const EPS: f32 = 1e-7;
    let col = |b: &Tensor, k: usize| b.narrow(1, k, 1)?.squeeze(1);
    let (ax1, ay1, ax2, ay2) = (col(b1, 0)?, col(b1, 1)?, col(b1, 2)?, col(b1, 3)?);
    let (bx1, by1, bx2, by2) = (col(b2, 0)?, col(b2, 1)?, col(b2, 2)?, col(b2, 3)?);
    let (w1, h1) = (ax2.sub(&ax1)?, ay2.sub(&ay1)?.add_scalar(EPS)?);
    let (w2, h2) = (bx2.sub(&bx1)?, by2.sub(&by1)?.add_scalar(EPS)?);
    let iw = ax2.minimum(&bx2)?.sub(&ax1.maximum(&bx1)?)?.relu()?;
    let ih = ay2.minimum(&by2)?.sub(&ay1.maximum(&by1)?)?.relu()?;
    let inter = iw.mul(&ih)?;
    let union = w1.mul(&h1)?.add(&w2.mul(&h2)?)?.sub(&inter)?.add_scalar(EPS)?;
    let iou = inter.div(&union)?;
    let cw = ax2.maximum(&bx2)?.sub(&ax1.minimum(&bx1)?)?;
    let ch = ay2.maximum(&by2)?.sub(&ay1.minimum(&by1)?)?;
    let c2 = cw.square()?.add(&ch.square()?)?.add_scalar(EPS)?;
    let dx = bx1.add(&bx2)?.sub(&ax1)?.sub(&ax2)?;
    let dy = by1.add(&by2)?.sub(&ay1)?.sub(&ay2)?;
    let rho2 = dx.square()?.add(&dy.square()?)?.mul_scalar(0.25)?;
    let d = atan(&w2.div(&h2)?)?.sub(&atan(&w1.div(&h1)?)?)?;
    let v = d.square()?.mul_scalar(4.0 / (std::f32::consts::PI * std::f32::consts::PI))?;
    let alpha = {
        let _g = no_grad();
        v.div(&v.sub(&iou)?.add_scalar(1.0 + EPS)?)?
    };
    iou.sub(&rho2.div(&c2)?.add(&v.mul(&alpha)?)?)
}

/// The differentiable terms of §7.4 for a fixed assignment (from [`assign_batch`] on the same outputs).
///
/// The assignment is a constant, so this is the function whose gradient training uses; [`detection_loss`] is
/// `assign_batch` followed by this. Errors if the assignment does not match the outputs' shapes.
pub fn loss_with_assignment(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    cfg: &LossConfig,
) -> Result<LossOutput> {
    loss_impl(box_raw, cls_raw, anchors, asg, cfg, box_raw.device().is_cpu())
}

/// [`loss_with_assignment`] with the kernel choice explicit: `fused` uses the CPU custom ops ([`bce_sum_cpu`],
/// [`gather_flat_cpu`]), otherwise the device-generic composite of tensor ops. Both give the same values and
/// gradients (the tests compare them).
fn loss_impl(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    cfg: &LossConfig,
    fused: bool,
) -> Result<LossOutput> {
    let (b, nc, n_a) = check_inputs(box_raw, cls_raw, anchors, cfg)?;
    if (asg.batch, asg.num_classes, asg.num_anchors) != (b, nc, n_a) {
        return Err(invalid(format!(
            "assignment is for B {} nc {} A {}, outputs are B {b} nc {nc} A {n_a}",
            asg.batch, asg.num_classes, asg.num_anchors
        )));
    }
    let n = asg.num_fg();
    if [asg.image.len(), asg.label.len(), asg.score.len(), asg.target.len()] != [n; 4] {
        return Err(invalid("assignment vectors differ in length"));
    }
    let dev = cls_raw.device();
    let r = cfg.reg_max;
    let tss_raw = asg.target_score_sum();
    let tss = tss_raw.max(1.0) as f32;

    // Per foreground entry: the flat positions of its class logit and of its 4·reg_max box logits, its anchor
    // point and grid-unit target box, and its DFL bins with weights t[a] / 4 (the mean over sides), so the DFL term
    // is one weighted sum.
    let mut idx_cls = Vec::with_capacity(n);
    let mut idx_box = Vec::with_capacity(n * 4 * r);
    let mut anchor4 = Vec::with_capacity(n * 4);
    let mut tbox = Vec::with_capacity(n * 4);
    let mut bins = Vec::with_capacity(n * 8);
    let mut bin_w = Vec::with_capacity(n * 8);
    for k in 0..n {
        let (i, a, c) = (asg.image[k] as usize, asg.anchor[k] as usize, asg.label[k] as usize);
        if i >= b || a >= n_a || c >= nc {
            return Err(invalid(format!(
                "assignment entry {k} (image {i}, anchor {a}, class {c}) is out of range"
            )));
        }
        idx_cls.push((i * nc + c) * n_a + a);
        let base = i * 4 * r * n_a + a;
        idx_box.extend((0..4 * r).map(|ch| base + ch * n_a));
        let p = anchors.points[a];
        anchor4.extend([p[0], p[1], p[0], p[1]]);
        let inv = 1.0 / anchors.strides[a];
        let g = asg.target[k].scale(inv, inv);
        tbox.extend([g.x1, g.y1, g.x2, g.y2]);
        let wt = asg.score[k] * 0.25;
        for d in bbox2dist_clamped(p, &g, r) {
            let (j, wl, wr) = dfl_bins(d);
            bins.extend([j as i64, j as i64 + 1]);
            bin_w.extend([wl * wt, wr * wt]);
        }
    }
    // An anchor has at most one gt (§7.3 step 4); the BCE kernels rely on distinct foreground positions.
    let mut keys: Vec<usize> = (0..n).map(|k| asg.image[k] as usize * n_a + asg.anchor[k] as usize).collect();
    keys.sort_unstable();
    if keys.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid("assignment lists an anchor of an image more than once"));
    }
    let on_dev = |t: Tensor| t.to_device(dev);
    let index = |v: Vec<usize>| -> Result<Tensor> {
        let len = v.len();
        on_dev(Tensor::from_vec(v.into_iter().map(|i| i as i64).collect::<Vec<_>>(), [len])?)
    };
    let t = on_dev(Tensor::from_slice(&asg.score, [n])?)?;

    // Classification: Σ BCE(x, T) = Σ_bg softplus(x) + Σ_fg (softplus(x) − x · t), since T is t at the assigned
    // class of each foreground anchor and 0 elsewhere. Foreground terms are formed per element before any sum, so a
    // confident positive (large x, t ≈ 1) does not cancel against a large dense sum.
    let cls_sum = if fused {
        bce_sum_cpu(cls_raw, idx_cls, asg.score.clone())?
    } else if n == 0 {
        sum_fixed_order(&cls_raw.softplus()?.sum(2, false)?)?
    } else {
        // Move the foreground logits to −100 (softplus ≈ 4e-44, gradient ≈ 0) for the dense sum; the shift is
        // detached and `index_add` passes the gradient through to `cls_raw` unchanged. Per-(image, class) row sums
        // keep a fixed order.
        let idx = index(idx_cls)?;
        let flat = cls_raw.flatten_all()?;
        let x_fg = flat.index_select(0, &idx)?;
        let shift = x_fg.detach().neg()?.sub_scalar(100.0)?;
        let bg = flat.index_add(0, &idx, &shift)?.reshape(cls_raw.dims())?;
        let dense = sum_fixed_order(&bg.softplus()?.sum(2, false)?)?;
        dense.add(&sum_fixed_order(&x_fg.softplus()?.sub(&x_fg.mul(&t)?)?)?)?
    };

    let (l_box, l_dfl) = if n == 0 {
        (Tensor::zeros_with([], DType::F32, dev)?, Tensor::zeros_with([], DType::F32, dev)?)
    } else {
        let logits = if fused {
            gather_flat_cpu(box_raw, idx_box)?
        } else {
            box_raw.flatten_all()?.index_select(0, &index(idx_box)?)?
        };
        let logits = logits.reshape([n, 4, r])?;

        // Box: DFL expectation → grid-unit xyxy → CIoU against the target.
        let proj = on_dev(Tensor::arange(0.0, r as f32, 1.0)?)?;
        let dist = logits.softmax(2)?.mul(&proj)?.sum(2, false)?;
        let sign = on_dev(Tensor::from_slice(&[-1.0f32, -1.0, 1.0, 1.0], [4])?)?;
        let pd = on_dev(Tensor::from_vec(anchor4, [n, 4])?)?.add(&dist.mul(&sign)?)?;
        let tb = on_dev(Tensor::from_vec(tbox, [n, 4])?)?;
        let l_box = sum_fixed_order(&ciou(&pd, &tb)?.rsub_scalar(1.0)?.mul(&t)?)?.div_scalar(tss)?;

        // DFL: cross-entropy against the two neighbouring bins of each side's target distance.
        let bins = on_dev(Tensor::from_vec(bins, [n, 4, 2])?)?;
        let bin_w = on_dev(Tensor::from_vec(bin_w, [n, 4, 2])?)?;
        let picked = logits.log_softmax(2)?.gather(2, &bins)?;
        let l_dfl = sum_fixed_order(&picked.mul(&bin_w)?)?.neg()?.div_scalar(tss)?;
        (l_box, l_dfl)
    };
    let l_cls = cls_sum.div_scalar(tss)?;

    let (wb, wc, wd) =
        (l_box.mul_scalar(cfg.box_gain)?, l_cls.mul_scalar(cfg.cls_gain)?, l_dfl.mul_scalar(cfg.dfl_gain)?);
    let items = LossItems { box_: wb.item::<f32>()?, cls: wc.item::<f32>()?, dfl: wd.item::<f32>()? };
    let total = wb.add(&wc)?.add(&wd)?.mul_scalar(b as f32)?;
    let total_value = total.item::<f32>()?;
    Ok(LossOutput { total, total_value, items, num_fg: n, target_score_sum: tss_raw as f32 })
}

/// The full detection loss of `docs/MODEL.md` §7: [`assign_batch`] then [`loss_with_assignment`].
///
/// `box_raw: [B, 4 · reg_max, A]`, `cls_raw: [B, nc, A]` (§6.3), `anchors` for the input size, `targets` one entry
/// per image in input pixels. If any head output is non-finite the result is a detached NaN loss
/// ([`LossOutput::is_finite`] is false) and the assigner is skipped.
pub fn detection_loss(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<LossOutput> {
    let asg = assign_batch(box_raw, cls_raw, anchors, targets, cfg)?;
    if !asg.inputs_finite {
        let nan = f32::NAN;
        return Ok(LossOutput {
            total: Tensor::scalar(nan).to_device(cls_raw.device())?,
            total_value: nan,
            items: LossItems { box_: nan, cls: nan, dfl: nan },
            num_fg: 0,
            target_score_sum: 0.0,
        });
    }
    loss_with_assignment(box_raw, cls_raw, anchors, &asg, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::make_anchors;

    /// The fused CPU kernels and the device-generic composite give the same loss and gradients.
    #[test]
    fn fused_matches_composite() {
        let anchors = make_anchors(96, 96, &[8, 16, 32], 0.5);
        let (b, nc, a, r) = (3, 5, anchors.len(), 16);
        let mut rng = folo_core::rng::Rng::seed(3);
        let box_raw = Tensor::rand_uniform([b, 4 * r, a], 0.0, 4.0, &mut rng).requires_grad_();
        let cls_raw = Tensor::rand_uniform([b, nc, a], -6.0, 3.0, &mut rng).requires_grad_();
        let gt = |boxes: Vec<Xyxy>, labels: Vec<u32>| ImageTargets { boxes, labels };
        let targets = vec![
            gt(vec![Xyxy::new(10.0, 12.0, 60.0, 70.0), Xyxy::new(40.0, 5.0, 90.0, 40.0)], vec![1, 4]),
            gt(vec![], vec![]),
            gt(vec![Xyxy::new(20.0, 30.0, 34.0, 41.0)], vec![0]),
        ];
        let cfg = LossConfig::default();
        let asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
        assert!(asg.num_fg() > 0);
        let fused = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, true).unwrap();
        let composite = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, false).unwrap();
        let close = |x: f32, y: f32| (x - y).abs() <= 1e-5 * (1.0 + x.abs().max(y.abs()));
        let (fi, ci) = (fused.items, composite.items);
        assert!(close(fi.box_, ci.box_) && close(fi.cls, ci.cls) && close(fi.dfl, ci.dfl), "{fi:?} {ci:?}");
        let (gf, gc) = (fused.total.backward().unwrap(), composite.total.backward().unwrap());
        for t in [&box_raw, &cls_raw] {
            let (x, y) =
                (gf.get(t).unwrap().to_vec::<f32>().unwrap(), gc.get(t).unwrap().to_vec::<f32>().unwrap());
            for (i, (x, y)) in x.iter().zip(&y).enumerate() {
                assert!((x - y).abs() <= 1e-6 + 1e-5 * y.abs(), "element {i}: {x} vs {y}");
            }
        }
    }

    /// Confident positives (logit ≈ 80, t = 1) among background logits: BCE is non-negative and tiny, on both
    /// kernel paths. Summing softplus and `x · t` separately would leave an error of about ulp(80) ≈ 8e-6, of
    /// either sign.
    #[test]
    fn confident_positives_do_not_cancel() {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let mut cls = vec![-30.0f32; 2 * a];
        let fg = [3usize, 17, 40, 63];
        for (k, &an) in fg.iter().enumerate() {
            cls[a + an] = 80.4 + 0.1 * k as f32; // class 1
        }
        let cls_raw = Tensor::from_vec(cls, [1, 2, a]).unwrap().requires_grad_();
        let box_raw = Tensor::zeros([1, 64, a]);
        let asg = BatchAssignment {
            batch: 1,
            num_anchors: a,
            num_classes: 2,
            image: vec![0; 4],
            anchor: fg.iter().map(|&an| an as u32).collect(),
            label: vec![1; 4],
            score: vec![1.0; 4],
            target: vec![Xyxy::new(0.0, 0.0, 64.0, 64.0); 4],
            inputs_finite: true,
        };
        let cfg = LossConfig::default();
        // Background: 2A − 4 logits at −30; foreground terms ≈ e^-80 ≈ 0. tss = 4.
        let want = 0.5 * (2 * a - 4) as f64 * (-30f64).exp().ln_1p() / 4.0;
        for fused in [true, false] {
            let out = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, fused).unwrap();
            let got = out.items.cls as f64;
            assert!(got >= 0.0 && (got - want).abs() <= 1e-6 * want, "fused {fused}: {got} vs {want}");
            // d/dx = B · 0.5 · (sigmoid(x) − T) / tss ≈ 0 at the positives.
            let g = out.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
            assert!(fg.iter().all(|&an| g[a + an].abs() < 1e-9), "fused {fused}");
        }
    }

    /// Duplicate foreground anchors are rejected (an anchor has at most one gt).
    #[test]
    fn rejects_duplicate_anchors() {
        let anchors = make_anchors(32, 32, &[8], 0.5);
        let a = anchors.len();
        let asg = BatchAssignment {
            batch: 1,
            num_anchors: a,
            num_classes: 2,
            image: vec![0, 0],
            anchor: vec![5, 5],
            label: vec![0, 1],
            score: vec![0.5, 0.5],
            target: vec![Xyxy::new(0.0, 0.0, 32.0, 32.0); 2],
            inputs_finite: true,
        };
        let (box_raw, cls_raw) = (Tensor::zeros([1, 64, a]), Tensor::zeros([1, 2, a]));
        assert!(loss_with_assignment(&box_raw, &cls_raw, &anchors, &asg, &LossConfig::default()).is_err());
    }

    #[test]
    fn sum_fixed_order_matches_sum() {
        for n in [1, 4096, 4097, 3 * 4096 + 5, 40_000_000 / 1000] {
            let x = Tensor::arange(0.0, n as f32, 1.0).unwrap().mul_scalar(1e-3).unwrap();
            let want = (n as f64 - 1.0) * n as f64 / 2.0 * 1e-3;
            let got = sum_fixed_order(&x).unwrap().item::<f32>().unwrap() as f64;
            assert!((got - want).abs() <= 1e-5 * want.max(1.0), "{n}: {got} vs {want}");
        }
    }
}
