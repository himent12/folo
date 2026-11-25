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
//!
//! Files: `assignment.rs` ([`assign_batch`], the detached host part), `terms/` ([`loss_with_assignment`] and
//! [`ciou`], the differentiable terms), `device.rs` ([`assign_batch_device`], the folo-gpu kernels), `distill/`
//! (the opt-in distillation term of §8.9, [`distill_loss`]) and `tests.rs`.
//! This file: [`LossConfig`], [`ImageTargets`], [`BatchAssignment`], [`detection_loss`], [`dual_loss`] (§6.6) and
//! [`step_loss`] (one training step's loss, with the distillation term).
//!
//! [`assign`]: crate::assign::assign

mod assignment;
mod device;
pub mod distill;
mod terms;
#[cfg(test)]
mod tests;

pub use assignment::assign_batch;
pub use device::assign_batch_device;
pub use distill::{
    DistillConfig, DistillLoss, DistillTargets, TeacherView, distill_loss, distill_loss_with_context,
    gt_verify_weights_cpu,
};
pub use terms::{ciou, inner_ciou_loss, loss_with_assignment, nwd_loss, rpiou_loss, wiou_v3_loss};

use crate::assign::TalConfig;
use crate::boxes::Xyxy;
use crate::decode::Anchors;
use folo_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

use device::detection_loss_device;

const OP: &str = "detection_loss";

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument { op: OP, msg: msg.into() }
}

/// Which classification loss [`detection_loss`] uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClsLossKind {
    /// Standard BCE-with-logits (or MAL when `mal_gamma > 0`, default).
    #[default]
    Bce,
    /// Varifocal Loss (Zhang et al. 2021): positives (`t > 0`) use quality-weighted `t · BCE(x, t)` and
    /// negatives (`t == 0`) use `vfl_alpha · p^vfl_gamma · softplus(x)`.
    Vfl,
    /// Quality Focal Loss (Li et al. 2020): `|t − p|^qfl_beta · BCE(x, t)` across both positives and negatives.
    Qfl,
}

/// Which box regression geometry loss [`detection_loss`] uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoxLossKind {
    /// Complete IoU (default, `docs/MODEL.md` §7.2).
    #[default]
    Ciou,
    /// Rational-Focal Edge-Decoupled IoU (`R-PIoU`): `(1 - u)(1 + β·u) + κ·E² / (1 + κ·E²)`, purely algebraic.
    Rpiou,
    /// Inner-CIoU (Zhang et al. 2023): CIoU with its IoU term computed on auxiliary boxes scaled by `inner_ratio`.
    InnerCiou,
    /// Wise-IoU v3 (Tong et al. 2023): distance-attended IoU loss with non-monotonic dynamic focusing (`wiou_alpha`, `wiou_delta`, `wiou_momentum`).
    WiouV3,
}

/// Loss hyper-parameters. Defaults are `docs/MODEL.md` §7.4's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LossConfig {
    /// Weight of the box regression term.
    pub box_gain: f32,
    /// Weight of the classification BCE.
    pub cls_gain: f32,
    /// Weight of the distribution focal loss.
    pub dfl_gain: f32,
    /// DFL bins per side (the head's `reg_max`).
    pub reg_max: usize,
    /// Assigner settings.
    pub tal: TalConfig,
    /// Matchability-Aware Loss (MAL) background focal exponent (`0.0` = plain BCE). When `> 0`, negatives use
    /// `p^γ · softplus(x)` while positives keep 100% unweighted BCE.
    pub mal_gamma: f32,
    /// Classification loss family ([`ClsLossKind::Bce`] default; opt-in `Vfl` or `Qfl`).
    pub cls_loss: ClsLossKind,
    /// Negative weight `α > 0` for [`ClsLossKind::Vfl`] (`0.75` default).
    pub vfl_alpha: f32,
    /// Negative focal exponent `γ >= 0` for [`ClsLossKind::Vfl`] (`2.0` default).
    pub vfl_gamma: f32,
    /// Focal exponent `β` (`0` or `>= 1`) for [`ClsLossKind::Qfl`] (`2.0` default).
    pub qfl_beta: f32,
    /// Scale-adaptive NWD blending weight `λ₀` in `[0, 1]` for the box loss (`0.0` = no NWD blend).
    pub nwd_weight: f32,
    /// Box regression geometry loss (`Ciou` default, `Rpiou`, `InnerCiou`, or `WiouV3`).
    pub box_loss: BoxLossKind,
    /// Auxiliary box scale ratio `r` in `[0.5, 1.5]` for [`BoxLossKind::InnerCiou`] (`0.7` default).
    pub inner_ratio: f32,
    /// Non-monotonic focusing base `α > 1` for [`BoxLossKind::WiouV3`] (`1.9` default).
    pub wiou_alpha: f32,
    /// Non-monotonic focusing pivot `δ > 0` for [`BoxLossKind::WiouV3`] (`3.0` default).
    pub wiou_delta: f32,
    /// EMA momentum `m` in `[0, 1]` for the [`BoxLossKind::WiouV3`] running `L_IoU` mean (`0.01` default).
    pub wiou_momentum: f32,
    /// High-IoU focusing slope `β` in `[0, 1]` for [`BoxLossKind::Rpiou`] (`0.5` default).
    pub rpiou_beta: f32,
    /// Rational edge-penalty sharpness `κ > 0` for [`BoxLossKind::Rpiou`] (`2.0` default).
    pub rpiou_kappa: f32,
    /// Grid-unit GT-size floor `δ₀ > 0` for [`BoxLossKind::Rpiou`] (`0.5` default).
    pub rpiou_delta: f32,
    /// Weight `λ_ot >= 0` of the 1D optimal-transport cumulative tail-barrier term added to DFL (`0.0` = standard DFL).
    pub dfl_ot_weight: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            box_gain: 7.5,
            cls_gain: 0.5,
            dfl_gain: 1.5,
            reg_max: 16,
            tal: TalConfig::default(),
            mal_gamma: 0.0,
            cls_loss: ClsLossKind::Bce,
            vfl_alpha: 0.75,
            vfl_gamma: 2.0,
            qfl_beta: 2.0,
            nwd_weight: 0.0,
            box_loss: BoxLossKind::Ciou,
            inner_ratio: 0.7,
            wiou_alpha: 1.9,
            wiou_delta: 3.0,
            wiou_momentum: 0.01,
            rpiou_beta: 0.5,
            rpiou_kappa: 2.0,
            rpiou_delta: 0.5,
            dfl_ot_weight: 0.0,
        }
    }
}

impl LossConfig {
    /// The one-to-one branch's loss (MODEL.md §6.6): §7 with the assignment of [`TalConfig::o2o`]
    /// (`topk = 7`, then `topk2 = 1`).
    pub fn o2o() -> Self {
        Self { tal: TalConfig::o2o(), ..Self::default() }
    }

    /// The one-to-one branch's loss of a run with this config: the same gains and `reg_max`, and this assigner with
    /// the one-to-one selection ([`TalConfig::one_to_one`]). With `keeps_tal` off (`TrainConfig.o2o_keeps_tal` of a
    /// run started before the key) the assigner is [`TalConfig::o2o`], as that run trained.
    pub fn o2o_branch(&self, keeps_tal: bool) -> Self {
        let tal = if keeps_tal { self.tal.one_to_one() } else { TalConfig::o2o() };
        Self { tal, ..*self }
    }

    /// Validate all loss hyper-parameters (shared by [`detection_loss`] and `TrainConfig::validate`).
    pub fn validate_params(&self) -> std::result::Result<(), String> {
        if !(self.mal_gamma.is_finite() && (0.0..=5.0).contains(&self.mal_gamma)) {
            return Err(format!("loss.mal_gamma must be finite and in 0..=5, got {}", self.mal_gamma));
        }
        if !(self.rpiou_beta.is_finite() && (0.0..=1.0).contains(&self.rpiou_beta)) {
            return Err(format!("loss.rpiou_beta must be finite and in 0..=1, got {}", self.rpiou_beta));
        }
        if !(self.rpiou_kappa.is_finite() && self.rpiou_kappa > 0.0) {
            return Err(format!("loss.rpiou_kappa must be finite and > 0, got {}", self.rpiou_kappa));
        }
        if !(self.rpiou_delta.is_finite() && self.rpiou_delta > 0.0) {
            return Err(format!("loss.rpiou_delta must be finite and > 0, got {}", self.rpiou_delta));
        }
        if !(self.dfl_ot_weight.is_finite() && self.dfl_ot_weight >= 0.0) {
            return Err(format!("loss.dfl_ot_weight must be finite and >= 0, got {}", self.dfl_ot_weight));
        }
        if self.cls_loss != ClsLossKind::Bce && self.mal_gamma != 0.0 {
            return Err(format!(
                "loss.mal_gamma ({}) applies only to ClsLossKind::Bce, got {:?}",
                self.mal_gamma, self.cls_loss
            ));
        }
        if !(self.vfl_alpha.is_finite() && self.vfl_alpha > 0.0) {
            return Err(format!("loss.vfl_alpha must be finite and > 0, got {}", self.vfl_alpha));
        }
        if !(self.vfl_gamma.is_finite() && self.vfl_gamma >= 0.0) {
            return Err(format!("loss.vfl_gamma must be finite and >= 0, got {}", self.vfl_gamma));
        }
        if !(self.qfl_beta.is_finite() && (self.qfl_beta == 0.0 || self.qfl_beta >= 1.0)) {
            return Err(format!("loss.qfl_beta must be finite and 0 or >= 1, got {}", self.qfl_beta));
        }
        if !(self.inner_ratio.is_finite() && (0.5..=1.5).contains(&self.inner_ratio)) {
            return Err(format!(
                "loss.inner_ratio must be finite and in 0.5..=1.5, got {}",
                self.inner_ratio
            ));
        }
        if !(self.wiou_alpha.is_finite() && self.wiou_alpha > 1.0) {
            return Err(format!("loss.wiou_alpha must be finite and > 1, got {}", self.wiou_alpha));
        }
        if !(self.wiou_delta.is_finite() && self.wiou_delta > 0.0) {
            return Err(format!("loss.wiou_delta must be finite and > 0, got {}", self.wiou_delta));
        }
        if !(self.wiou_momentum.is_finite() && (0.0..=1.0).contains(&self.wiou_momentum)) {
            return Err(format!(
                "loss.wiou_momentum must be finite and in 0..=1, got {}",
                self.wiou_momentum
            ));
        }
        Ok(())
    }
}

/// The weight `w_m` of the one-to-many loss after `completed` epochs of a run of `epochs` (MODEL.md §6.6):
/// `max(1 − e / max(E − 1, 1), 0) · (0.8 − 0.1) + 0.1`, from 0.8 in the first epoch to 0.1 in the last. The
/// one-to-one loss gets `1 − w_m`.
pub fn o2m_weight(completed: u32, epochs: u32) -> f32 {
    let e = completed as f64;
    let d = epochs.saturating_sub(1).max(1) as f64;
    ((1.0 - e / d).max(0.0) * (0.8 - 0.1) + 0.1) as f32
}

/// The dual-assignment loss of a `head = "o2o"` model (MODEL.md §6.6): `L = w_m · L_o2m + (1 − w_m) · L_o2o`.
/// The reported items are the same weighted sums of the two branches' items; `num_fg` and `target_score_sum`
/// are the one-to-many branch's. Non-finite when either part is.
pub fn dual_loss(o2m: &LossOutput, o2o: &LossOutput, w_m: f32) -> Result<LossOutput> {
    let w_o = 1.0 - w_m;
    let mix = |a: f32, b: f32| w_m * a + w_o * b;
    let (m, o) = (&o2m.items, &o2o.items);
    Ok(LossOutput {
        total: o2m.total.mul_scalar(w_m)?.add(&o2o.total.mul_scalar(w_o)?)?,
        total_value: mix(o2m.total_value, o2o.total_value),
        items: LossItems { box_: mix(m.box_, o.box_), cls: mix(m.cls, o.cls), dfl: mix(m.dfl, o.dfl) },
        num_fg: o2m.num_fg,
        target_score_sum: o2m.target_score_sum,
        mean_iou_loss: o2m.mean_iou_loss,
    })
}

/// The loss of one training step (§8.3) of `batch` images: the one-to-many loss `o2m` plus the distillation term
/// `kd` of §8.9, if any, and for a `head = "o2o"` model (`o2o = Some((L_o2o, w_m))`) the [`dual_loss`] of that sum
/// and the one-to-one loss: `L = w_m · (L_o2m + L_distill) + (1 − w_m) · L_o2o`, so the teacher is weighted with the
/// branch it teaches. `total_value` gains `batch · (kd.cls + kd.dfl)` before the weighting; the items stay the
/// supervised ones. Without `o2o` it is `o2m` plus `kd`.
pub fn step_loss(
    mut o2m: LossOutput,
    kd: Option<&DistillLoss>,
    o2o: Option<(&LossOutput, f32)>,
    batch: usize,
) -> Result<LossOutput> {
    if let Some(kd) = kd {
        o2m.total = o2m.total.add(&kd.total)?;
        o2m.total_value += batch as f32 * (kd.cls + kd.dfl);
    }
    match o2o {
        Some((o2o, w_m)) => dual_loss(&o2m, o2o, w_m),
        None => Ok(o2m),
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
    /// Mean unweighted `1 - IoU` over foreground anchors when [`BoxLossKind::WiouV3`] runs with `num_fg > 0`,
    /// used by the trainer to update the Tong et al. §III-B exponential moving average `L_IoU_bar`.
    pub mean_iou_loss: Option<f32>,
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
    cfg.validate_params().map_err(invalid)?;
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

/// The full detection loss of `docs/MODEL.md` §7: [`assign_batch`] then [`loss_with_assignment`] on the host,
/// or, when the head outputs are on a folo-gpu device, the same computation as GPU kernels
/// ([`folo_gpu::detect::detection_loss`]: assigner, fused BCE, fused CIoU + DFL, with a fused backward) whose
/// only host round trip is reading back the reported items.
///
/// `box_raw: [B, 4 · reg_max, A]`, `cls_raw: [B, nc, A]` (§6.3), `anchors` for the input size, `targets` one entry
/// per image in input pixels. If any head output is non-finite the result is a NaN loss
/// ([`LossOutput::is_finite`] is false; on the host the assigner is skipped and `total` is detached).
pub fn detection_loss(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<LossOutput> {
    detection_loss_with_wiou_mean(box_raw, cls_raw, anchors, targets, cfg, 1.0)
}

/// [`detection_loss`] with an explicit WIoU v3 running-EMA denominator `wiou_iou_mean` (Tong et al. §III-B).
pub(crate) fn detection_loss_with_wiou_mean(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
    wiou_iou_mean: f32,
) -> Result<LossOutput> {
    if device_loss_supported(box_raw, cfg) {
        return detection_loss_device(box_raw, cls_raw, anchors, targets, cfg);
    }
    detection_loss_host_with_wiou_mean(box_raw, cls_raw, anchors, targets, cfg, wiou_iou_mean)
}

/// Supervised assignment context kept alongside [`LossOutput`] when distillation needs GT verification or
/// conflict-free BCKD clamping (zero duplicate assigner pass).
pub enum DistillAssignment {
    /// Host assignment.
    Host(BatchAssignment),
    /// Device assignment buffers on `folo-gpu`.
    Device(folo_gpu::detect::Assignment),
}

/// [`detection_loss`], also returning the supervised [`DistillAssignment`] when `keep_assignment` is true.
pub fn detection_loss_for_distill(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
    keep_assignment: bool,
) -> Result<(LossOutput, Option<DistillAssignment>)> {
    detection_loss_for_distill_with_wiou_mean(box_raw, cls_raw, anchors, targets, cfg, keep_assignment, 1.0)
}

/// [`detection_loss_for_distill`] with an explicit WIoU v3 running-EMA denominator `wiou_iou_mean`.
pub(crate) fn detection_loss_for_distill_with_wiou_mean(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
    keep_assignment: bool,
    wiou_iou_mean: f32,
) -> Result<(LossOutput, Option<DistillAssignment>)> {
    if !keep_assignment {
        return Ok((
            detection_loss_with_wiou_mean(box_raw, cls_raw, anchors, targets, cfg, wiou_iou_mean)?,
            None,
        ));
    }
    if device_loss_supported(box_raw, cfg) {
        let (out, asg) =
            device::detection_loss_device_with_assignment(box_raw, cls_raw, anchors, targets, cfg)?;
        return Ok((out, asg.map(DistillAssignment::Device)));
    }
    let asg = assign_batch(box_raw, cls_raw, anchors, targets, cfg)?;
    if !asg.inputs_finite {
        return Ok((nan_loss(cls_raw.device())?, None));
    }
    let out = terms::loss_with_assignment_wiou(box_raw, cls_raw, anchors, &asg, cfg, wiou_iou_mean)?;
    Ok((out, Some(DistillAssignment::Host(asg))))
}

/// Whether [`detection_loss`] runs on the device for these outputs: they live on a folo-gpu device and `topk`
/// and `reg_max` fit its kernels (`1..=64`, `2..=32`); other settings take the host path.
pub fn device_loss_supported(box_raw: &Tensor, cfg: &LossConfig) -> bool {
    folo_gpu::detect::supported(box_raw.device())
        && cfg.cls_loss == ClsLossKind::Bce
        && matches!(cfg.box_loss, BoxLossKind::Ciou | BoxLossKind::Rpiou)
        && (1..=folo_gpu::detect::MAX_TOPK).contains(&cfg.tal.topk)
        && cfg.tal.topk2 != Some(0)
        && (2..=folo_gpu::detect::MAX_REG_MAX).contains(&cfg.reg_max)
}

/// [`detection_loss`] with the assigner on the host, whatever the device (the reference the device path is
/// tested against).
pub fn detection_loss_host(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<LossOutput> {
    detection_loss_host_with_wiou_mean(box_raw, cls_raw, anchors, targets, cfg, 1.0)
}

fn detection_loss_host_with_wiou_mean(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
    wiou_iou_mean: f32,
) -> Result<LossOutput> {
    let asg = assign_batch(box_raw, cls_raw, anchors, targets, cfg)?;
    if !asg.inputs_finite {
        return nan_loss(cls_raw.device());
    }
    terms::loss_with_assignment_wiou(box_raw, cls_raw, anchors, &asg, cfg, wiou_iou_mean)
}

fn nan_loss(device: folo_core::Device) -> Result<LossOutput> {
    let nan = f32::NAN;
    Ok(LossOutput {
        total: Tensor::scalar(nan).to_device(device)?,
        total_value: nan,
        items: LossItems { box_: nan, cls: nan, dfl: nan },
        num_fg: 0,
        target_score_sum: 0.0,
        mean_iou_loss: None,
    })
}
