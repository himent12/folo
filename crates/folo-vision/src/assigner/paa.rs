//! PAA (Probabilistic Anchor Assignment, Kim & Lee ECCV 2020) standalone sample assigner and 2-component
//! 1D Gaussian Mixture EM solver on plain slices.
//!
//! For one image with `A` anchors across pyramid levels (`level_sizes` summing to `A`), `C` classes, and `G`
//! ground-truth boxes:
//!
//! 1. **Per-level candidate pre-selection (FOLO IoU-first variant).** For each valid (positive-area) GT box
//!    `g` and each pyramid level `l`, pick the top `min(topk, level_sizes[l])` anchors by descending
//!    `IoU(gt[g], anchor[a])` (breaking ties by smaller squared center distance to `gt[g]`, then smaller
//!    anchor index), optionally filtering by `anchor_iou >= min_anchor_iou`. (Note: Kim & Lee's Procedure 1
//!    first assigns anchors to their highest-IoU GT and ranks eligible anchors by joint prediction score;
//!    FOLO's variant selects spatial candidates by anchor-to-GT IoU first so each GT's pool stays spatially
//!    anchored before scoring.)
//! 2. **Joint classification + regression anchor loss.** For each candidate `a in C_g`:
//!    ```text
//!    cls_loss(g, a) = softplus(-x[a, y_g]) + sum_{c != y_g} softplus(x[a, c])
//!    reg_loss(g, a) = -ln(IoU(gt[g], pd_bboxes[a]) + iou_eps)
//!    loss(g, a)     = cls_weight * cls_loss(g, a) + reg_weight * reg_loss(g, a)
//!    ```
//! 3. **2-component 1D Gaussian Mixture EM separation** ([`fit_gmm_1d_two_component`]). Fit a deterministic
//!    2-component 1D GMM initialized at `(min_loss, max_loss)` with equal mixing weights `0.5` and variance
//!    floor `gmm_var_floor`. Ordering components so `pos_mean <= neg_mean`, candidates with
//!    `loss <= pos_mean` are positive, candidates with `loss >= neg_mean` (when `pos_mean < neg_mean`) are
//!    negative, and intermediate candidates are positive iff `P(pos | loss) >= 0.5`. This prevents unequal
//!    component variances from flipping tail assignments while guaranteeing the lowest-loss candidate stays
//!    positive.
//! 4. **Conflict resolution.** An anchor selected as positive by multiple GTs is assigned to the GT with the
//!    strictly lowest joint `loss(g, a)`, ties broken deterministically by smaller GT index.

use serde::{Deserialize, Serialize};

use super::atss::{box_iou_f64, box_is_finite, check_level_sizes};
use crate::assign::Assignment;
use crate::boxes::Xyxy;
use crate::error::{Error, Result};

/// Hyper-parameters for [`assign_paa`] (Kim & Lee ECCV 2020).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PaaConfig {
    /// Maximum number of candidate anchors selected per FPN pyramid level (paper default `9`).
    pub topk: usize,
    /// Weight on the multi-label BCE-with-logits classification loss (default `1.0`).
    pub cls_weight: f32,
    /// Weight on the `-ln(IoU + iou_eps)` localization loss (default `1.0`).
    pub reg_weight: f32,
    /// Optional minimum anchor-to-GT IoU in `[0, 1]` required to enter a GT's candidate pool (default `0.0`,
    /// which allows fallback to nearest anchors when small GTs do not overlap any anchor box).
    pub min_anchor_iou: f32,
    /// Positive floor inside `-ln(IoU + iou_eps)` so zero-IoU predictions remain finite (default `1e-8`).
    pub iou_eps: f32,
    /// Maximum EM iterations for the 2-component 1D GMM (default `50`).
    pub gmm_max_iters: usize,
    /// Absolute log-likelihood convergence tolerance for EM (default `1e-6`).
    pub gmm_tol: f64,
    /// Strict positive variance floor for GMM components so singular clusters stay finite (default `1e-6`).
    pub gmm_var_floor: f64,
}

impl Default for PaaConfig {
    fn default() -> Self {
        Self {
            topk: 9,
            cls_weight: 1.0,
            reg_weight: 1.0,
            min_anchor_iou: 0.0,
            iou_eps: 1e-8,
            gmm_max_iters: 50,
            gmm_tol: 1e-6,
            gmm_var_floor: 1e-6,
        }
    }
}

impl PaaConfig {
    /// Validate all PAA hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if self.topk == 0 {
            return Err(Error::Config("paa.topk must be >= 1, got 0".into()));
        }
        if !(self.cls_weight.is_finite() && self.cls_weight >= 0.0) {
            return Err(Error::Config(format!(
                "paa.cls_weight must be finite and >= 0, got {}",
                self.cls_weight
            )));
        }
        if !(self.reg_weight.is_finite() && self.reg_weight >= 0.0) {
            return Err(Error::Config(format!(
                "paa.reg_weight must be finite and >= 0, got {}",
                self.reg_weight
            )));
        }
        if !(self.min_anchor_iou.is_finite() && (0.0..=1.0).contains(&self.min_anchor_iou)) {
            return Err(Error::Config(format!(
                "paa.min_anchor_iou must be finite and in [0, 1], got {}",
                self.min_anchor_iou
            )));
        }
        if !(self.iou_eps.is_finite() && self.iou_eps > 0.0) {
            return Err(Error::Config(format!("paa.iou_eps must be finite and > 0, got {}", self.iou_eps)));
        }
        if self.gmm_max_iters == 0 {
            return Err(Error::Config("paa.gmm_max_iters must be >= 1, got 0".into()));
        }
        if !(self.gmm_tol.is_finite() && self.gmm_tol >= 0.0) {
            return Err(Error::Config(format!("paa.gmm_tol must be finite and >= 0, got {}", self.gmm_tol)));
        }
        if !(self.gmm_var_floor.is_finite() && self.gmm_var_floor > 0.0) {
            return Err(Error::Config(format!(
                "paa.gmm_var_floor must be finite and > 0, got {}",
                self.gmm_var_floor
            )));
        }
        Ok(())
    }
}

/// Fitted 2-component 1D Gaussian Mixture Model produced by [`fit_gmm_1d_two_component`].
///
/// Component `pos` always represents the lower-loss (positive) mode (`pos_mean <= neg_mean`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gmm1dResult {
    /// Mean of the lower-loss (positive) Gaussian component.
    pub pos_mean: f64,
    /// Mean of the higher-loss (negative) Gaussian component (`>= pos_mean`).
    pub neg_mean: f64,
    /// Variance of the positive component (`>= var_floor`).
    pub pos_var: f64,
    /// Variance of the negative component (`>= var_floor`).
    pub neg_var: f64,
    /// Mixing weight of the positive component in `[0, 1]`.
    pub pos_weight: f64,
    /// Mixing weight of the negative component (`1 - pos_weight`).
    pub neg_weight: f64,
    /// Final total log-likelihood `sum_i ln(w_pos N(x_i; pos) + w_neg N(x_i; neg))`.
    pub log_likelihood: f64,
    /// Number of EM iterations executed.
    pub iterations: usize,
    /// Whether `|ll_t - ll_{t-1}| <= tol` was reached within `max_iters`.
    pub converged: bool,
    /// Per-sample positive classification (`[M]`).
    pub positive_mask: Vec<bool>,
    /// Per-sample posterior probability `P(pos | x_i)` (`[M]`).
    pub pos_posteriors: Vec<f64>,
}

/// Exact log-PDF of a 1D Gaussian `N(mu, var)` evaluated at `x`.
#[inline]
fn log_gaussian_1d(x: f64, mu: f64, var: f64) -> f64 {
    let diff = x - mu;
    -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + (diff * diff) / var)
}

/// Numerically stable `ln(exp(a) + exp(b))`.
#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if !hi.is_finite() {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// Exact responsibility `P(comp 0 | x)` with deterministic Mahalanobis fallback if both component
/// log-densities underflow to `-inf`.
#[inline]
fn responsibility_0(x: f64, mu0: f64, var0: f64, lp0: f64, mu1: f64, var1: f64, lp1: f64) -> (f64, f64) {
    let lse = log_add_exp(lp0, lp1);
    if lse.is_finite() {
        return (lse, (lp0 - lse).exp().clamp(0.0, 1.0));
    }
    let z0 = (x - mu0).abs() / var0.sqrt();
    let z1 = (x - mu1).abs() / var1.sqrt();
    let r0 = match z0.total_cmp(&z1) {
        std::cmp::Ordering::Less => 1.0,
        std::cmp::Ordering::Greater => 0.0,
        std::cmp::Ordering::Equal => 0.5,
    };
    (lse, r0)
}

/// Fit a deterministic 2-component 1D Gaussian Mixture Model via EM on candidate anchor losses `scores`
/// (lower score = more likely positive).
///
/// Initialization follows PAA (Kim & Lee ECCV 2020 §3.2): component means start at `(min(scores), max(scores))`,
/// mixing weights start at `(0.5, 0.5)`, and both variances start at `max(sample_var, var_floor)`.
/// Moment accumulation is performed in centered/scaled coordinates `[0, 1]` to prevent overflow on large
/// finite offsets; if the fitted variance or log-likelihood cannot be represented in `f64`, returns
/// [`Error::Config`].
///
/// # Edge cases
/// - Empty `scores`: returns an empty, converged [`Gmm1dResult`].
/// - Single score or constant scores (`max - min <= 1e-12`): marks all candidates positive without division by
///   zero.
/// - Non-finite scores, unrepresentable variance scale, or invalid solver parameters return [`Error::Config`].
pub fn fit_gmm_1d_two_component(
    scores: &[f64],
    max_iters: usize,
    tol: f64,
    var_floor: f64,
) -> Result<Gmm1dResult> {
    if max_iters == 0 {
        return Err(Error::Config("fit_gmm_1d_two_component: max_iters must be >= 1".into()));
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(Error::Config(format!(
            "fit_gmm_1d_two_component: tol must be finite and >= 0, got {tol}"
        )));
    }
    if !(var_floor.is_finite() && var_floor > 0.0) {
        return Err(Error::Config(format!(
            "fit_gmm_1d_two_component: var_floor must be finite and > 0, got {var_floor}"
        )));
    }
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_finite() {
            return Err(Error::Config(format!("fit_gmm_1d_two_component: scores[{i}] is non-finite ({s})")));
        }
    }

    let n = scores.len();
    if n == 0 {
        return Ok(Gmm1dResult {
            pos_mean: 0.0,
            neg_mean: 0.0,
            pos_var: var_floor,
            neg_var: var_floor,
            pos_weight: 0.5,
            neg_weight: 0.5,
            log_likelihood: 0.0,
            iterations: 0,
            converged: true,
            positive_mask: Vec::new(),
            pos_posteriors: Vec::new(),
        });
    }
    if n == 1 {
        let x0 = scores[0];
        let ll = log_gaussian_1d(x0, x0, var_floor);
        return Ok(Gmm1dResult {
            pos_mean: x0,
            neg_mean: x0,
            pos_var: var_floor,
            neg_var: var_floor,
            pos_weight: 1.0,
            neg_weight: 0.0,
            log_likelihood: ll,
            iterations: 0,
            converged: true,
            positive_mask: vec![true],
            pos_posteriors: vec![1.0],
        });
    }

    let mut s_min = f64::INFINITY;
    let mut s_max = f64::NEG_INFINITY;
    for &x in scores {
        if x < s_min {
            s_min = x;
        }
        if x > s_max {
            s_max = x;
        }
    }
    let spread = s_max - s_min;
    if !spread.is_finite() {
        return Err(Error::Config("fit_gmm_1d_two_component: score range overflows f64".into()));
    }
    if spread == 0.0 {
        let ll = (n as f64) * log_gaussian_1d(s_min, s_min, var_floor);
        if !ll.is_finite() {
            return Err(Error::Config("fit_gmm_1d_two_component: log-likelihood overflows f64".into()));
        }
        return Ok(Gmm1dResult {
            pos_mean: s_min,
            neg_mean: s_min,
            pos_var: var_floor,
            neg_var: var_floor,
            pos_weight: 0.5,
            neg_weight: 0.5,
            log_likelihood: ll,
            iterations: 0,
            converged: true,
            positive_mask: vec![true; n],
            pos_posteriors: vec![0.5; n],
        });
    }
    let mean_offset = scores.iter().map(|&x| x - s_min).sum::<f64>() / n as f64;
    let mean_all = s_min + mean_offset;
    if spread <= 1e-12 {
        let ll: f64 = scores.iter().map(|&x| log_gaussian_1d(x, mean_all, var_floor)).sum();
        if !ll.is_finite() {
            return Err(Error::Config("fit_gmm_1d_two_component: log-likelihood overflows f64".into()));
        }
        return Ok(Gmm1dResult {
            pos_mean: mean_all,
            neg_mean: mean_all,
            pos_var: var_floor,
            neg_var: var_floor,
            pos_weight: 0.5,
            neg_weight: 0.5,
            log_likelihood: ll,
            iterations: 0,
            converged: true,
            positive_mask: vec![true; n],
            pos_posteriors: vec![0.5; n],
        });
    }

    let spread_sq = spread * spread;
    if !spread_sq.is_finite() || spread_sq == 0.0 {
        return Err(Error::Config(
            "fit_gmm_1d_two_component: score variance scale cannot be represented in f64".into(),
        ));
    }
    // Work in normalized coordinates `u_i = (x_i - s_min) / spread in [0, 1]` so all intermediate moments
    // stay strictly bounded and immune to large-offset cancellation or quadratic overflow.
    let inv_spread = 1.0 / spread;
    let u_scores: Vec<f64> = scores.iter().map(|&x| (x - s_min) * inv_spread).collect();
    let var_floor_u = (var_floor / spread_sq).max(1e-15);
    let u_mean = u_scores.iter().sum::<f64>() / n as f64;
    let init_var_u = (u_scores
        .iter()
        .map(|&u| {
            let d = u - u_mean;
            d * d
        })
        .sum::<f64>()
        / n as f64)
        .max(var_floor_u);

    let mut mu0 = 0.0f64;
    let mut mu1 = 1.0f64;
    let mut var0 = init_var_u;
    let mut var1 = init_var_u;
    let mut w0 = 0.5f64;
    let mut w1 = 0.5f64;

    let mut resp0 = vec![0.5f64; n];
    let mut prev_ll = f64::NEG_INFINITY;
    let mut iters = 0usize;
    let mut converged = false;

    const MIN_WEIGHT: f64 = 1e-6;
    let ll_shift = (n as f64) * spread.ln();

    for iter in 1..=max_iters {
        iters = iter;
        let log_w0 = w0.ln();
        let log_w1 = w1.ln();
        let mut cur_ll = 0.0f64;
        for (i, &u) in u_scores.iter().enumerate() {
            let lp0 = log_w0 + log_gaussian_1d(u, mu0, var0);
            let lp1 = log_w1 + log_gaussian_1d(u, mu1, var1);
            let (lse, r0) = responsibility_0(u, mu0, var0, lp0, mu1, var1, lp1);
            cur_ll += lse;
            resp0[i] = r0;
        }
        let ll = cur_ll - ll_shift;

        // M-step in normalized coordinates.
        let mut n0 = 0.0f64;
        let mut n1 = 0.0f64;
        let mut su0 = 0.0f64;
        let mut su1 = 0.0f64;
        for (&u, &r0) in u_scores.iter().zip(&resp0) {
            let r1 = 1.0 - r0;
            n0 += r0;
            n1 += r1;
            su0 += r0 * u;
            su1 += r1 * u;
        }
        if n0 > 1e-12 {
            mu0 = su0 / n0;
            let mut suu0 = 0.0f64;
            for (&u, &r0) in u_scores.iter().zip(&resp0) {
                let d = u - mu0;
                suu0 += r0 * d * d;
            }
            var0 = (suu0 / n0).max(var_floor_u);
        } else {
            var0 = var_floor_u;
        }
        if n1 > 1e-12 {
            mu1 = su1 / n1;
            let mut suu1 = 0.0f64;
            for (&u, &r0) in u_scores.iter().zip(&resp0) {
                let r1 = 1.0 - r0;
                let d = u - mu1;
                suu1 += r1 * d * d;
            }
            var1 = (suu1 / n1).max(var_floor_u);
        } else {
            var1 = var_floor_u;
        }
        w0 = (n0 / n as f64).clamp(MIN_WEIGHT, 1.0 - MIN_WEIGHT);
        w1 = 1.0 - w0;

        if iter > 1 && ll.is_finite() && prev_ll.is_finite() && (ll - prev_ll).abs() <= tol {
            converged = true;
            break;
        }
        prev_ll = ll;
    }

    // Unscale back to original coordinates and canonicalize component 0 as lower-loss (`pos_mean <= neg_mean`).
    let (mu0_raw, mu1_raw) = (s_min + spread * mu0, s_min + spread * mu1);
    let (var0_raw, var1_raw) = ((var0 * spread_sq).max(var_floor), (var1 * spread_sq).max(var_floor));
    let (
        pos_mean,
        neg_mean,
        pos_var,
        neg_var,
        pos_weight,
        neg_weight,
        pos_mu_u,
        neg_mu_u,
        pos_var_u,
        neg_var_u,
    ) = if mu0_raw <= mu1_raw {
        (mu0_raw, mu1_raw, var0_raw, var1_raw, w0, w1, mu0, mu1, var0, var1)
    } else {
        (mu1_raw, mu0_raw, var1_raw, var0_raw, w1, w0, mu1, mu0, var1, var0)
    };
    if !(pos_mean.is_finite() && neg_mean.is_finite() && pos_var.is_finite() && neg_var.is_finite()) {
        return Err(Error::Config(
            "fit_gmm_1d_two_component: fitted parameters cannot be represented in f64".into(),
        ));
    }

    let log_wp = pos_weight.ln();
    let log_wn = neg_weight.ln();
    let mut final_ll = 0.0f64;
    let mut pos_posteriors = Vec::with_capacity(n);
    let mut positive_mask = Vec::with_capacity(n);
    for (&x, &u) in scores.iter().zip(&u_scores) {
        let lp = log_wp + log_gaussian_1d(u, pos_mu_u, pos_var_u);
        let ln_ = log_wn + log_gaussian_1d(u, neg_mu_u, neg_var_u);
        let (lse, post) = responsibility_0(u, pos_mu_u, pos_var_u, lp, neg_mu_u, neg_var_u, ln_);
        final_ll += lse;
        pos_posteriors.push(post);
        let is_pos = if x <= pos_mean {
            true
        } else if pos_mean < neg_mean && x >= neg_mean {
            false
        } else {
            post >= 0.5
        };
        positive_mask.push(is_pos);
    }
    final_ll -= ll_shift;
    if !final_ll.is_finite() {
        return Err(Error::Config(
            "fit_gmm_1d_two_component: log-likelihood cannot be represented in f64".into(),
        ));
    }

    Ok(Gmm1dResult {
        pos_mean,
        neg_mean,
        pos_var,
        neg_var,
        pos_weight,
        neg_weight,
        log_likelihood: final_ll,
        iterations: iters,
        converged,
        positive_mask,
        pos_posteriors,
    })
}

/// Per-anchor assignment produced by [`assign_paa`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PaaAssignment {
    /// Whether each anchor is foreground (`[A]`).
    pub fg_mask: Vec<bool>,
    /// Assigned GT index per anchor (`[A]`).
    pub target_gt: Vec<usize>,
    /// Assigned class id per anchor (`[A]`, `num_classes` for background).
    pub target_labels: Vec<u32>,
    /// Assigned GT box per anchor (`[A]`).
    pub target_bboxes: Vec<Xyxy>,
    /// Predicted-to-GT IoU for each foreground anchor (`[A]`, in `[0, 1]`).
    pub target_ious: Vec<f32>,
    /// Joint PAA matching loss in `f64` for each foreground anchor (`[A]`; `0.0` for background).
    pub matched_losses: Vec<f64>,
    /// Fitted 2-component 1D GMM diagnostics per GT (`[G]`; `None` for degenerate or empty-pool GTs).
    pub gmm_stats: Vec<Option<Gmm1dResult>>,
    /// Total foreground anchors after conflict resolution.
    pub num_fg: usize,
    /// `true` when all prediction, anchor, and GT inputs were finite; `false` returns an empty assignment.
    pub inputs_finite: bool,
}

impl PaaAssignment {
    fn empty(num_anchors: usize, num_gts: usize, num_classes: usize, inputs_finite: bool) -> Self {
        Self {
            fg_mask: vec![false; num_anchors],
            target_gt: vec![0; num_anchors],
            target_labels: vec![num_classes as u32; num_anchors],
            target_bboxes: vec![Xyxy::default(); num_anchors],
            target_ious: vec![0.0; num_anchors],
            matched_losses: vec![0.0; num_anchors],
            gmm_stats: vec![None; num_gts],
            num_fg: 0,
            inputs_finite,
        }
    }

    /// Convert to [`Assignment`], using [`Self::target_ious`] as `target_scores`.
    pub fn to_assignment(&self) -> Assignment {
        Assignment {
            fg_mask: self.fg_mask.clone(),
            target_gt: self.target_gt.clone(),
            target_labels: self.target_labels.clone(),
            target_bboxes: self.target_bboxes.clone(),
            target_scores: self.target_ious.clone(),
            num_fg: self.num_fg,
        }
    }

    /// Build dense `[A, num_classes]` row-major continuous quality targets (`target_ious[a]` at
    /// `(a, target_labels[a])`, `0.0` elsewhere).
    pub fn dense_quality_targets(&self, num_classes: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.fg_mask.len() * num_classes];
        for (a, &fg) in self.fg_mask.iter().enumerate() {
            if fg {
                out[a * num_classes + self.target_labels[a] as usize] = self.target_ious[a];
            }
        }
        out
    }
}

#[inline]
fn softplus_f64(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[inline]
fn one_hot_bce_cost_f64(logits: &[f32], gt_class: usize) -> f64 {
    let mut sum = 0.0f64;
    for (c, &x) in logits.iter().enumerate() {
        if c == gt_class {
            sum += softplus_f64(-(x as f64));
        } else {
            sum += softplus_f64(x as f64);
        }
    }
    sum
}

/// Run PAA (Probabilistic Anchor Assignment) for one image.
///
/// - `pd_cls_logits`: `[A * num_classes]` row-major raw classification logits (detached).
/// - `num_classes`: number of foreground classes `C`.
/// - `pd_bboxes`: `[A]` predicted boxes (`Xyxy`).
/// - `anchor_bboxes`: `[A]` anchor boxes ordered contiguously by pyramid level.
/// - `level_sizes`: number of anchors in each pyramid level (summing to `A`).
/// - `gt_bboxes`: `[G]` ground-truth boxes.
/// - `gt_labels`: `[G]` class ids (`< num_classes`).
/// - `cfg`: [`PaaConfig`] hyper-parameters.
///
/// # Errors
/// Returns [`Error::Config`] on invalid config parameters, slice length mismatches, out-of-range
/// `gt_labels`, or `num_classes == 0` with non-empty GTs.
#[allow(clippy::too_many_arguments)]
pub fn assign_paa(
    pd_cls_logits: &[f32],
    num_classes: usize,
    pd_bboxes: &[Xyxy],
    anchor_bboxes: &[Xyxy],
    level_sizes: &[usize],
    gt_bboxes: &[Xyxy],
    gt_labels: &[u32],
    cfg: &PaaConfig,
) -> Result<PaaAssignment> {
    cfg.validate()?;

    let n_a = anchor_bboxes.len();
    let n_g = gt_bboxes.len();
    check_level_sizes(level_sizes, n_a, "paa")?;
    if pd_bboxes.len() != n_a {
        return Err(Error::Config(format!("paa: pd_bboxes length {} != A ({n_a})", pd_bboxes.len())));
    }
    let expected_cls_len = n_a
        .checked_mul(num_classes)
        .ok_or_else(|| Error::Config(format!("paa: A ({n_a}) * num_classes ({num_classes}) overflows")))?;
    if pd_cls_logits.len() != expected_cls_len {
        return Err(Error::Config(format!(
            "paa: pd_cls_logits length {} != A * num_classes ({n_a} * {num_classes})",
            pd_cls_logits.len()
        )));
    }
    if gt_labels.len() != n_g {
        return Err(Error::Config(format!("paa: gt_labels length {} != G ({n_g})", gt_labels.len())));
    }
    if n_g > 0 && num_classes == 0 {
        return Err(Error::Config("paa: num_classes must be >= 1 when gt_bboxes is non-empty".into()));
    }
    for (g, &l) in gt_labels.iter().enumerate() {
        if (l as usize) >= num_classes {
            return Err(Error::Config(format!("paa: gt_labels[{g}] = {l} >= num_classes ({num_classes})")));
        }
    }

    let all_finite = pd_cls_logits.iter().all(|v| v.is_finite())
        && pd_bboxes.iter().all(box_is_finite)
        && anchor_bboxes.iter().all(box_is_finite)
        && gt_bboxes.iter().all(box_is_finite);
    if !all_finite {
        return Ok(PaaAssignment::empty(n_a, n_g, num_classes, false));
    }
    if n_g == 0 || n_a == 0 {
        return Ok(PaaAssignment::empty(n_a, n_g, num_classes, true));
    }

    let anchor_centers: Vec<[f64; 2]> = anchor_bboxes
        .iter()
        .map(|b| [0.5 * (b.x1 as f64 + b.x2 as f64), 0.5 * (b.y1 as f64 + b.y2 as f64)])
        .collect();

    let cls_w = cfg.cls_weight as f64;
    let reg_w = cfg.reg_weight as f64;
    let iou_eps = cfg.iou_eps as f64;
    let min_aiou = cfg.min_anchor_iou as f64;

    let mut out = PaaAssignment::empty(n_a, n_g, num_classes, true);
    // Winning `(gt_idx, loss, pred_iou)` per anchor across GTs.
    let mut owner: Vec<Option<(usize, f64, f64)>> = vec![None; n_a];
    // `(anchor_iou, dist_sq, anchor_idx)` within one level.
    let mut level_cands: Vec<(f64, f64, usize)> = Vec::new();
    let mut cand_anchors: Vec<usize> = Vec::new();
    let mut cand_losses: Vec<f64> = Vec::new();
    let mut cand_pred_ious: Vec<f64> = Vec::new();

    for g in 0..n_g {
        let gt = &gt_bboxes[g];
        let gw = gt.x2 as f64 - gt.x1 as f64;
        let gh = gt.y2 as f64 - gt.y1 as f64;
        if gw <= 0.0 || gh <= 0.0 {
            continue;
        }
        let gt_cx = 0.5 * (gt.x1 as f64 + gt.x2 as f64);
        let gt_cy = 0.5 * (gt.y1 as f64 + gt.y2 as f64);
        let gt_class = gt_labels[g] as usize;

        cand_anchors.clear();
        let mut offset = 0usize;
        for &lvl_len in level_sizes {
            let end = offset + lvl_len;
            if lvl_len > 0 {
                level_cands.clear();
                level_cands.reserve(lvl_len);
                for (a, center) in anchor_centers.iter().enumerate().take(end).skip(offset) {
                    let aiou = box_iou_f64(gt, &anchor_bboxes[a]);
                    if min_aiou > 0.0 && aiou < min_aiou {
                        continue;
                    }
                    let dx = center[0] - gt_cx;
                    let dy = center[1] - gt_cy;
                    level_cands.push((aiou, dx * dx + dy * dy, a));
                }
                let m_l = level_cands.len();
                let k_l = cfg.topk.min(m_l);
                if m_l > k_l && k_l > 0 {
                    level_cands.select_nth_unstable_by(k_l - 1, |x, y| {
                        y.0.total_cmp(&x.0).then_with(|| x.1.total_cmp(&y.1)).then_with(|| x.2.cmp(&y.2))
                    });
                    level_cands.truncate(k_l);
                }
                level_cands.sort_unstable_by_key(|&(_, _, a)| a);
                for &(_, _, a) in &level_cands {
                    cand_anchors.push(a);
                }
            }
            offset = end;
        }

        if cand_anchors.is_empty() {
            continue;
        }

        cand_losses.clear();
        cand_pred_ious.clear();
        cand_losses.reserve(cand_anchors.len());
        cand_pred_ious.reserve(cand_anchors.len());
        for &a in &cand_anchors {
            let pred_iou = box_iou_f64(gt, &pd_bboxes[a]);
            let reg_loss = -(pred_iou + iou_eps).ln();
            let cls_loss = if cls_w == 0.0 {
                0.0
            } else {
                let row = &pd_cls_logits[a * num_classes..(a + 1) * num_classes];
                one_hot_bce_cost_f64(row, gt_class)
            };
            cand_losses.push(cls_w * cls_loss + reg_w * reg_loss);
            cand_pred_ious.push(pred_iou);
        }

        let gmm = fit_gmm_1d_two_component(&cand_losses, cfg.gmm_max_iters, cfg.gmm_tol, cfg.gmm_var_floor)?;
        for (idx, &a) in cand_anchors.iter().enumerate() {
            if !gmm.positive_mask[idx] {
                continue;
            }
            let loss = cand_losses[idx];
            let pred_iou = cand_pred_ious[idx];
            let slot = &mut owner[a];
            if slot.is_none_or(|(_, best_loss, _)| loss < best_loss) {
                *slot = Some((g, loss, pred_iou));
            }
        }
        out.gmm_stats[g] = Some(gmm);
    }

    for (a, slot) in owner.into_iter().enumerate() {
        if let Some((g, loss, iou)) = slot {
            out.fg_mask[a] = true;
            out.target_gt[a] = g;
            out.target_labels[a] = gt_labels[g];
            out.target_bboxes[a] = gt_bboxes[g];
            out.target_ious[a] = iou as f32;
            out.matched_losses[a] = loss;
            out.num_fg += 1;
        }
    }
    Ok(out)
}
