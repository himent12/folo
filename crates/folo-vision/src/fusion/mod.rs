//! Cluster box fusion after NMS (docs/MODEL.md §9.4), opt-in through [`crate::nms::NmsConfig::fusion`].
//!
//! NMS keeps one candidate per object and drops the other members of its cluster: several more trained estimates
//! of the same four edges. Fusion moves each side of a kept box to a weighted mean of that side over its cluster:
//!
//! - [`BoxFusion::Vote`]: score-weighted box voting (each member weighs its class score);
//! - [`BoxFusion::Dfl`]: DFL evidence fusion, the score times the member's **per-side precision**, read from its
//!   DFL distribution ([`crate::decode::side_precision`]): an anchor near an object's left edge knows that edge
//!   sharply and the far edge vaguely, so its left edge counts more than its right.
//!
//! Scores, classes, the kept set and its order are NMS's; only the boxes move. The cluster of a kept candidate is
//! every candidate that entered NMS (after the confidence filter and the candidate cap) with its class (any class
//! under `agnostic`) whose IoU with the kept box is `>= iou`, the kept candidate included. Lane `fusion` of
//! hill-climb 3 chose the rule and the weights on 5000 train2017 images (docs/notes/hill/r3-fusion.md).
//!
//! [`ScoreRule::Consensus`], opt-in through [`crate::nms::NmsConfig::score`], uses such a cluster for the other half
//! of a detection, always within the box's own class: it scales each kept box's score by the score mass of its
//! cluster and re-sorts ([`consensus_scores`], docs/MODEL.md §9.4, docs/notes/hill/r6-consensus.md). Boxes, classes
//! and the kept set do not change.

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;

/// `ε` of [`crate::decode::side_precision`], in bins²: the floor of a side's excess variance.
pub const DFL_FUSION_EPS: f32 = 0.1;

/// Default cluster IoU of [`crate::nms::NmsConfig::fusion_iou`].
pub const FUSION_IOU: f32 = 0.75;

/// The cluster IoU `τ` of [`ScoreRule::Consensus`]. Fixed: it does not follow `fusion_iou`.
pub const CONSENSUS_IOU: f32 = 0.75;

/// The cap `R` of [`ScoreRule::Consensus`]: a cluster carrying this many times the kept box's score keeps the score.
pub const CONSENSUS_CAP: f64 = 10.0;

/// The cluster IoU `τ` of [`ScoreRule::Elc`], [`ScoreRule::DflConsensus`] and [`ScoreRule::DflRepulse`].
pub const ELC_IOU: f32 = 0.75;

/// The cluster support cap `R` of [`ScoreRule::Elc`], [`ScoreRule::DflConsensus`] and [`ScoreRule::DflRepulse`].
pub const ELC_CAP: f64 = 10.0;

/// The log-support slope `α` of [`ScoreRule::Elc`] and [`ScoreRule::DflConsensus`].
pub const ELC_ALPHA: f64 = 0.65;

/// The scale-preserving support reference `R_ref` of [`ScoreRule::Elc`] and [`ScoreRule::DflConsensus`].
pub const ELC_R_REF: f64 = 4.0;

/// The DFL excess-std slope `β` of [`ScoreRule::DflConsensus`].
pub const DFL_CONSENSUS_BETA: f64 = 0.60;

/// The scale-preserving DFL excess-std reference `σ_ref` (bins) of [`ScoreRule::DflConsensus`].
pub const DFL_CONSENSUS_SIG_REF: f64 = 0.50;

/// Lower clamp on the 4-side mean fused DFL excess std (bins) in [`ScoreRule::DflConsensus`].
const DFL_CONSENSUS_SIG_MIN: f64 = 0.1;

/// Power exponent `q` on cluster scores in [`ScoreRule::DflRepulse`] (`s^1.5 = s · √s`).
pub const DFL_REPULSE_Q: f64 = 1.5;

/// Log-support slope `α` of [`ScoreRule::DflRepulse`].
pub const DFL_REPULSE_ALPHA: f64 = 0.70;

/// Scale-preserving power-weighted support reference `R_ref` of [`ScoreRule::DflRepulse`].
pub const DFL_REPULSE_R_REF: f64 = 3.0;

/// DFL Shannon-entropy slope `β` (nats⁻¹) of [`ScoreRule::DflRepulse`].
pub const DFL_REPULSE_BETA: f64 = 0.55;

/// Scale-preserving mean 4-side DFL entropy reference `H_ref` (nats) of [`ScoreRule::DflRepulse`].
pub const DFL_REPULSE_ENT_REF: f64 = 1.5;

/// Cross-stride support weight `λ` of [`ScoreRule::DflRepulse`].
pub const DFL_REPULSE_STRIDE_W: f64 = 0.30;

/// Log-survival soft-repulsion strength `γ` of [`ScoreRule::DflRepulse`].
pub const DFL_REPULSE_GAMMA: f64 = 2.5;

/// Log-survival soft-repulsion strength `γ` of [`ScoreRule::DflCgr`] (Confidence-Gated Repulsion).
pub const DFL_CGR_GAMMA: f64 = 3.0;

/// Suppressor confidence-gating exponent `ψ` of [`ScoreRule::DflCgr`] (`p_m^0.25`).
pub const DFL_CGR_PSI: f64 = 0.25;

/// IoU-affinity exponent `ρ` of [`BoxFusion::Sebf`] (`IoU^4`).
pub const SEBF_RHO: f64 = 4.0;

/// Student-t degrees of freedom `ν` of [`BoxFusion::Sebf`] 1-step IRLS.
pub const SEBF_NU: f64 = 3.0;

/// Relative box-span scale floor `c_rel` of [`BoxFusion::Sebf`] Student-t variance (`(0.03 · L)²`).
pub const SEBF_C_REL: f64 = 0.03;

/// Overlap power `p` in `ln(max(1 − IoU^p, ε_r))` of [`ScoreRule::DflRepulse`] and [`ScoreRule::DflCgr`].
pub const DFL_REPULSE_POW: f64 = 4.0;

/// Survival floor `ε_r` inside `ln(max(1 − IoU^4, ε_r))` of [`ScoreRule::DflRepulse`].
const DFL_REPULSE_FLOOR: f64 = 1e-4;

/// Probability clamp before taking log-odds in [`elc_scores`], [`dfl_consensus_scores`] and [`dfl_repulse_scores`].
const ELC_PROB_EPS: f64 = 1e-6;

/// How the boxes NMS keeps are scored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreRule {
    /// The class score (the default; every published number).
    #[default]
    None,
    /// The class score scaled by the score mass of the box's cluster ([`consensus_scores`]). The score scale
    /// changes: a box that fired alone is divided by up to `√10`, so a threshold on it means something else.
    Consensus,
    /// Exclusive Logit-Space Consensus ([`elc_scores`]): each candidate votes only for the kept box that
    /// suppressed it in NMS, updating log-odds so high-confidence scores are protected and the `[0, 1]` scale
    /// is preserved.
    Elc,
    /// Scale-Free Exclusive Logit-Consensus with fused DFL precision ([`dfl_consensus_scores`]): adds a
    /// scale-free fused DFL excess-std term from the DFL moments already computed for [`BoxFusion::Dfl`].
    /// Falls back to [`ScoreRule::Elc`] when DFL bin precisions are unavailable.
    #[serde(rename = "dfl-consensus")]
    DflConsensus,
    /// Power-weighted exclusive logit consensus with DFL Shannon entropy, cross-stride agreement, and greedy
    /// sequential log-survival soft repulsion on output boxes ([`dfl_repulse_scores`], docs/MODEL.md §9.4).
    /// Preserves the `[0, 1]` score scale and gracefully omits entropy/stride terms when unavailable.
    #[serde(rename = "dfl-repulse")]
    DflRepulse,
    /// Confidence-Gated DFL Repulsion ([`dfl_cgr_scores`], docs/MODEL.md §9.4): [`ScoreRule::DflRepulse`] with
    /// suppressor confidence gating `p_m^{0.25}` inside the log-survival kernel (`γ = 3.0`), preventing weak
    /// detections from suppressing adjacent true objects. Best paired with [`BoxFusion::Sebf`] and `--iou 0.78`.
    #[serde(rename = "dfl-cgr")]
    DflCgr,
}

impl ScoreRule {
    /// The name used by the CLI and in JSON (`none`, `consensus`, `elc`, `dfl-consensus`, `dfl-repulse`, `dfl-cgr`).
    pub fn as_str(self) -> &'static str {
        match self {
            ScoreRule::None => "none",
            ScoreRule::Consensus => "consensus",
            ScoreRule::Elc => "elc",
            ScoreRule::DflConsensus => "dfl-consensus",
            ScoreRule::DflRepulse => "dfl-repulse",
            ScoreRule::DflCgr => "dfl-cgr",
        }
    }

    /// Whether this rule uses exclusive NMS suppressor ownership and centered logit-space calibration.
    #[inline]
    pub fn is_calibrated(self) -> bool {
        matches!(self, ScoreRule::Elc | ScoreRule::DflConsensus | ScoreRule::DflRepulse | ScoreRule::DflCgr)
    }

    /// Whether this rule uses per-anchor scale-free DFL bin precisions ([`crate::decode::bin_precision`]).
    #[inline]
    pub fn wants_bin_precisions(self) -> bool {
        self == ScoreRule::DflConsensus
    }

    /// Whether this rule uses per-anchor mean 4-side DFL Shannon entropy ([`crate::decode::dfl_entropy`]).
    #[inline]
    pub fn wants_dfl_entropy(self) -> bool {
        matches!(self, ScoreRule::DflRepulse | ScoreRule::DflCgr)
    }

    /// Whether this rule uses per-anchor stride for cross-scale consensus.
    #[inline]
    pub fn wants_strides(self) -> bool {
        matches!(self, ScoreRule::DflRepulse | ScoreRule::DflCgr)
    }
}

/// The `n` candidates grouped by class (stable: candidate order inside a class), so a cluster scans one bucket; with
/// `agnostic`, one bucket of every candidate.
struct Buckets<'a> {
    order: Vec<u32>,
    classes: &'a [u32],
    agnostic: bool,
}

impl<'a> Buckets<'a> {
    fn new(n: usize, classes: &'a [u32], agnostic: bool) -> Self {
        let mut order: Vec<u32> = (0..n as u32).collect();
        if !agnostic {
            order.sort_by_key(|&i| classes[i as usize]);
        }
        Buckets { order, classes, agnostic }
    }

    fn of(&self, c: u32) -> &[u32] {
        if self.agnostic {
            return &self.order;
        }
        let lo = self.order.partition_point(|&i| self.classes[i as usize] < c);
        let hi = self.order.partition_point(|&i| self.classes[i as usize] <= c);
        &self.order[lo..hi]
    }
}

/// IoU of `b` (of area `ba`) and `c`, in the arithmetic of `nms::nms_sorted` (and `boxes::box_iou`).
#[inline]
fn overlap(b: &Xyxy, ba: f32, c: &Xyxy) -> f32 {
    let w = (b.x2.min(c.x2) - b.x1.max(c.x1)).max(0.0);
    let h = (b.y2.min(c.y2) - b.y1.max(c.y1)).max(0.0);
    let inter = w * h;
    let union = ba + c.area() - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// The consensus scores of the `kept` candidates (indices into `boxes` / `scores` / `classes`, the candidates that
/// entered NMS), in `kept`'s order:
///
/// ```text
/// M_k  = Σ s_j over the candidates j of k's class with IoU(b_k, b_j) >= τ      (j = k always included)
/// s'_k = s_k · min(M_k / s_k, R)^a / R^a                                       τ = 0.75, a = 0.5, R = 10
/// ```
///
/// The sum runs in `f64` in candidate order, so the result does not depend on threads or on other classes'
/// candidates: the cluster stays within `k`'s class even when NMS is `agnostic`. A member whose score is not `> 0` or
/// whose box is not finite adds nothing, a kept box that is not finite has no members but itself (`M = s`), and a
/// kept score that is not `> 0` stays as it is.
pub fn consensus_scores(kept: &[usize], boxes: &[Xyxy], scores: &[f32], classes: &[u32]) -> Vec<f32> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let buckets = Buckets::new(boxes.len(), classes, false);
    kept.iter()
        .map(|&k| {
            if scores[k].is_nan() || scores[k] <= 0.0 {
                return scores[k];
            }
            let b = boxes[k];
            let ba = b.area();
            let s = scores[k] as f64;
            let mut mass = 0f64;
            // A NaN box can reach any IoU from f32 min / max, so both boxes of a pair must be finite.
            if finite(&b) {
                for &j in buckets.of(classes[k]) {
                    let j = j as usize;
                    let c = &boxes[j];
                    let member = j == k || (overlap(&b, ba, c) >= CONSENSUS_IOU && finite(c));
                    if member && scores[j] > 0.0 {
                        mass += scores[j] as f64;
                    }
                }
            } else {
                mass = s;
            }
            // a = 0.5: a square root, which IEEE 754 rounds exactly on every platform.
            (s * (mass / s).min(CONSENSUS_CAP).sqrt() / CONSENSUS_CAP.sqrt()) as f32
        })
        .collect()
}

/// Overflow-safe logistic sigmoid in `f64`.
#[inline]
fn branch_safe_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Class-local suppressor ownership of every candidate `j` relative to `kept` (assumed in descending score order):
/// each kept box owns itself, and every other candidate of class `c` is owned by the first kept box of class `c`
/// whose IoU with it is `> nms_iou` (`u32::MAX` if none). Used when NMS ran `agnostic` or when a caller invokes
/// [`elc_scores`] / [`dfl_consensus_scores`] / [`dfl_repulse_scores`] without precomputed owners.
fn class_local_owners(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    buckets: &Buckets<'_>,
    nms_iou: f32,
) -> Vec<u32> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let mut own = vec![u32::MAX; boxes.len()];
    for (kp, &k) in kept.iter().enumerate() {
        own[k] = kp as u32;
    }
    for (kp, &k) in kept.iter().enumerate() {
        let b = boxes[k];
        if !finite(&b) {
            continue;
        }
        let sk = scores[k];
        let ba = b.area();
        for &j in buckets.of(classes[k]) {
            let j = j as usize;
            // Require `j` to be lower-ranked than `k` so an agnostic cross-class suppression cannot let a weaker
            // survivor steal a stronger suppressed same-class candidate.
            let lower_ranked = scores[j] < sk || (scores[j] == sk && j > k);
            if own[j] == u32::MAX && lower_ranked && finite(&boxes[j]) && overlap(&b, ba, &boxes[j]) > nms_iou
            {
                own[j] = kp as u32;
            }
        }
    }
    own
}

/// Exclusive Logit-Space Consensus scores of `kept` (docs/MODEL.md §9.4), in `kept`'s order:
///
/// ```text
/// M_excl,k = Σ s_j over candidates j of k's class owned by k with IoU(b_k, b_j) >= 0.75   (j = k included)
/// logit(s'_k) = logit(s_k) + 0.65 · ln(min(M_excl,k / s_k, 10) / 4.0)
/// ```
///
/// `owner` maps each candidate index to the position in `kept` of the kept box of its class that suppressed it
/// (or itself, or `u32::MAX` if unowned); when `None` (e.g. under `agnostic` NMS), class-local ownership is derived
/// from `kept` and `nms_iou`. Sums run in `f64` in candidate order for same-platform bit determinism (`f64::ln` /
/// `exp` may differ by 1 ULP across OS libms).
pub fn elc_scores(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    owner: Option<&[u32]>,
    nms_iou: f32,
) -> Vec<f32> {
    dfl_consensus_scores(kept, boxes, scores, classes, owner, nms_iou, None)
}

/// Scale-Free Exclusive Logit-Consensus with fused DFL precision (docs/MODEL.md §9.4), in `kept`'s order:
///
/// ```text
/// σ̄_bin,k = (1/4) Σ_{e=0..4} ( (Σ_{j ∈ C_excl(k)} s_j · bin_prec(j)[e]) / M_excl,k )^{-1/2}
/// logit(s'_k) = logit(s_k) + 0.65 · ln(min(M_excl,k / s_k, 10) / 4.0) − 0.60 · ln(max(σ̄_bin,k, 0.1) / 0.50)
/// ```
///
/// When `bin_precision` is `None`, the DFL term is omitted and this reduces identically to [`elc_scores`].
pub fn dfl_consensus_scores(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    owner: Option<&[u32]>,
    nms_iou: f32,
    bin_precision: Option<&dyn Fn(usize) -> [f32; 4]>,
) -> Vec<f32> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let buckets = Buckets::new(boxes.len(), classes, false);
    let computed_owners;
    let owners: &[u32] = match owner {
        Some(o) => {
            assert_eq!(o.len(), boxes.len(), "dfl_consensus_scores: owner length mismatch");
            o
        }
        None => {
            computed_owners = class_local_owners(kept, boxes, scores, classes, &buckets, nms_iou);
            &computed_owners
        }
    };
    kept.iter()
        .enumerate()
        .map(|(kp, &k)| {
            if !scores[k].is_finite() || scores[k] <= 0.0 {
                return scores[k];
            }
            let b = boxes[k];
            let ba = b.area();
            let s = scores[k] as f64;
            let kp_u32 = kp as u32;
            let mut mass = 0f64;
            let mut prec_sum = [0f64; 4];
            let mut prec_mass = [0f64; 4];
            let mut add_member = |j: usize, sj: f64| {
                mass += sj;
                if let Some(bp_fn) = bin_precision {
                    let bp = bp_fn(j);
                    for e in 0..4 {
                        let p = bp[e] as f64;
                        if p.is_finite() && p > 0.0 {
                            prec_sum[e] += sj * p;
                            prec_mass[e] += sj;
                        }
                    }
                }
            };
            if finite(&b) {
                for &j in buckets.of(classes[k]) {
                    let j = j as usize;
                    let c = &boxes[j];
                    let member =
                        j == k || (owners[j] == kp_u32 && finite(c) && overlap(&b, ba, c) >= ELC_IOU);
                    if member && scores[j].is_finite() && scores[j] > 0.0 {
                        add_member(j, scores[j] as f64);
                    }
                }
            } else {
                add_member(k, s);
            }
            let sc = s.clamp(ELC_PROB_EPS, 1.0 - ELC_PROB_EPS);
            let mut z = (sc / (1.0 - sc)).ln();
            let ratio = (mass / s).clamp(1.0, ELC_CAP);
            z += ELC_ALPHA * (ratio / ELC_R_REF).ln();
            if bin_precision.is_some() && prec_mass.iter().all(|&m| m > 0.0) {
                let mut sum_std = 0f64;
                for e in 0..4 {
                    let mp = (prec_sum[e] / prec_mass[e]).max(1e-12);
                    sum_std += mp.sqrt().recip();
                }
                let sig_bin = (0.25 * sum_std).max(DFL_CONSENSUS_SIG_MIN);
                z -= DFL_CONSENSUS_BETA * (sig_bin / DFL_CONSENSUS_SIG_REF).ln();
            }
            branch_safe_sigmoid(z) as f32
        })
        .collect()
}

/// Power-weighted Exclusive Logit-Consensus with DFL Shannon entropy, cross-stride agreement and greedy
/// sequential log-survival soft repulsion (docs/MODEL.md §9.4), in `kept`'s order:
///
/// ```text
/// M_{q,k}  = Σ_{j ∈ C_excl(k)} s_j^{1.5}                                          (j = k included)
/// H̄_k      = (Σ_{j ∈ C_excl(k)} s_j · H_j) / (Σ s_j)                              (omitted if entropy is None)
/// f_cross  = (Σ_{j ∈ C_excl(k), stride_j ≠ stride_k} s_j) / (Σ s_j)               (omitted if stride is None)
/// z_k^(0)  = logit(s_k) + 0.70 · ln(min(M_{q,k} / s_k^{1.5}, 10) / 3.0)
///            − 0.55 · (H̄_k − 1.5) + 0.30 · f_cross
/// ```
///
/// Then, within each class `c`, greedy sequential log-survival soft repulsion runs on `repulse_boxes` (the output
/// boxes corresponding to `kept`, i.e. fused boxes when box fusion is enabled): repeatedly freeze the active box
/// `m` with the largest current logit (ties to earlier `kept` position) and update every remaining active box `u`
/// of class `c` with `o = IoU(b̂_m, b̂_u) > 0` by `z_u += 2.5 · ln(max(1 − o^4, 1e-4))`, before returning
/// `sigmoid(z)` in `f32`.
///
/// Sums run in `f64` in candidate order; cluster ownership and repulsion stay class-local even under `agnostic`
/// NMS. Non-finite/non-positive kept scores stay unchanged, and non-finite `repulse_boxes` do not participate in
/// repulsion.
#[allow(clippy::too_many_arguments)]
pub fn dfl_repulse_scores(
    kept: &[usize],
    boxes: &[Xyxy],
    repulse_boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    owner: Option<&[u32]>,
    nms_iou: f32,
    entropy: Option<&dyn Fn(usize) -> f32>,
    stride: Option<&dyn Fn(usize) -> f32>,
) -> Vec<f32> {
    dfl_repulse_scores_impl(
        kept,
        boxes,
        repulse_boxes,
        scores,
        classes,
        owner,
        nms_iou,
        entropy,
        stride,
        DFL_REPULSE_GAMMA,
        0.0,
    )
}

/// Confidence-Gated DFL Repulsion (`ScoreRule::DflCgr`, docs/MODEL.md §9.4): [`dfl_repulse_scores`] with
/// suppressor confidence gating `p_m^{0.25}` inside `γ · ln(max(1 − p_m^{0.25} · o⁴, 1e-4))` (`γ = 3.0`).
#[allow(clippy::too_many_arguments)]
pub fn dfl_cgr_scores(
    kept: &[usize],
    boxes: &[Xyxy],
    repulse_boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    owner: Option<&[u32]>,
    nms_iou: f32,
    entropy: Option<&dyn Fn(usize) -> f32>,
    stride: Option<&dyn Fn(usize) -> f32>,
) -> Vec<f32> {
    dfl_repulse_scores_impl(
        kept,
        boxes,
        repulse_boxes,
        scores,
        classes,
        owner,
        nms_iou,
        entropy,
        stride,
        DFL_CGR_GAMMA,
        DFL_CGR_PSI,
    )
}

#[allow(clippy::too_many_arguments)]
fn dfl_repulse_scores_impl(
    kept: &[usize],
    boxes: &[Xyxy],
    repulse_boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    owner: Option<&[u32]>,
    nms_iou: f32,
    entropy: Option<&dyn Fn(usize) -> f32>,
    stride: Option<&dyn Fn(usize) -> f32>,
    gamma: f64,
    psi: f64,
) -> Vec<f32> {
    assert_eq!(repulse_boxes.len(), kept.len(), "dfl_repulse_scores: repulse_boxes length mismatch");
    assert_eq!(scores.len(), boxes.len(), "dfl_repulse_scores: boxes/scores length mismatch");
    assert_eq!(classes.len(), boxes.len(), "dfl_repulse_scores: boxes/classes length mismatch");
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let buckets = Buckets::new(boxes.len(), classes, false);
    let computed_owners;
    let owners: &[u32] = match owner {
        Some(o) => {
            assert_eq!(o.len(), boxes.len(), "dfl_repulse_scores: owner length mismatch");
            o
        }
        None => {
            computed_owners = class_local_owners(kept, boxes, scores, classes, &buckets, nms_iou);
            &computed_owners
        }
    };
    let n_kept = kept.len();
    let mut z = vec![0.0f64; n_kept];
    let mut valid = vec![false; n_kept];
    for (kp, &k) in kept.iter().enumerate() {
        if !scores[k].is_finite() || scores[k] <= 0.0 {
            continue;
        }
        valid[kp] = true;
        let b = boxes[k];
        let ba = b.area();
        let s = scores[k] as f64;
        let sq = s * s.sqrt();
        let kp_u32 = kp as u32;
        let st_k = stride.map_or(0.0, |f| f(k));
        let st_k_valid = st_k.is_finite() && st_k > 0.0;
        let mut mass_q = 0f64;
        let mut ent_sum = 0f64;
        let mut ent_mass = 0f64;
        let mut cross_mass = 0f64;
        let mut stride_mass = 0f64;
        let mut add_member = |j: usize, sj: f64| {
            mass_q += sj * sj.sqrt();
            if let Some(ent_fn) = entropy {
                let h = ent_fn(j) as f64;
                if h.is_finite() && h >= 0.0 {
                    ent_sum += sj * h;
                    ent_mass += sj;
                }
            }
            if let Some(st_fn) = stride {
                let st_j = st_fn(j);
                if st_j.is_finite() && st_j > 0.0 {
                    if st_k_valid && st_j != st_k {
                        cross_mass += sj;
                    }
                    stride_mass += sj;
                }
            }
        };
        if finite(&b) {
            for &j in buckets.of(classes[k]) {
                let j = j as usize;
                let c = &boxes[j];
                let member = j == k || (owners[j] == kp_u32 && finite(c) && overlap(&b, ba, c) >= ELC_IOU);
                if member && scores[j].is_finite() && scores[j] > 0.0 {
                    add_member(j, scores[j] as f64);
                }
            }
        } else {
            add_member(k, s);
        }
        let sc = s.clamp(ELC_PROB_EPS, 1.0 - ELC_PROB_EPS);
        let mut zk = (sc / (1.0 - sc)).ln();
        let ratio = if sq > 0.0 { (mass_q / sq).clamp(1.0, ELC_CAP) } else { 1.0 };
        zk += DFL_REPULSE_ALPHA * (ratio / DFL_REPULSE_R_REF).ln();
        if ent_mass > 0.0 {
            zk -= DFL_REPULSE_BETA * (ent_sum / ent_mass - DFL_REPULSE_ENT_REF);
        }
        if st_k_valid && stride_mass > 0.0 {
            zk += DFL_REPULSE_STRIDE_W * (cross_mass / stride_mass);
        }
        z[kp] = zk;
    }

    // Greedy sequential class-local log-survival soft repulsion on `repulse_boxes`.
    if n_kept > 1 {
        let mut order: Vec<usize> =
            (0..n_kept).filter(|&kp| valid[kp] && finite(&repulse_boxes[kp])).collect();
        // Stable sort preserves ascending `kp` within each class for deterministic tie-breaking.
        order.sort_by_key(|&kp| classes[kept[kp]]);
        let mut lo = 0;
        while lo < order.len() {
            let cls = classes[kept[order[lo]]];
            let mut hi = lo + 1;
            while hi < order.len() && classes[kept[order[hi]]] == cls {
                hi += 1;
            }
            let grp = &order[lo..hi];
            let m = grp.len();
            if m > 1 {
                let areas: Vec<f32> = grp.iter().map(|&kp| repulse_boxes[kp].area()).collect();
                let mut active = vec![true; m];
                for _ in 0..m - 1 {
                    let mut best_u = usize::MAX;
                    let mut best_z = f64::NEG_INFINITY;
                    for u in 0..m {
                        if active[u] && (best_u == usize::MAX || z[grp[u]] > best_z) {
                            best_u = u;
                            best_z = z[grp[u]];
                        }
                    }
                    if best_u == usize::MAX {
                        break;
                    }
                    active[best_u] = false;
                    let gate = if psi > 0.0 { branch_safe_sigmoid(best_z).powf(psi) } else { 1.0 };
                    let bb = repulse_boxes[grp[best_u]];
                    let ba = areas[best_u];
                    for u in 0..m {
                        if !active[u] {
                            continue;
                        }
                        let cb = &repulse_boxes[grp[u]];
                        if bb.x2 <= cb.x1 || cb.x2 <= bb.x1 || bb.y2 <= cb.y1 || cb.y2 <= bb.y1 {
                            continue;
                        }
                        let o = overlap(&bb, ba, cb) as f64;
                        if o > 0.0 {
                            let o2 = o * o;
                            let o4 = o2 * o2;
                            z[grp[u]] += gamma * (1.0 - gate * o4).max(DFL_REPULSE_FLOOR).ln();
                        }
                    }
                }
            }
            lo = hi;
        }
    }

    (0..n_kept)
        .map(|kp| if valid[kp] { branch_safe_sigmoid(z[kp]) as f32 } else { scores[kept[kp]] })
        .collect()
}

/// How kept boxes are refined from their NMS cluster.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoxFusion {
    /// The kept boxes as NMS keeps them (the default; every published number).
    #[default]
    None,
    /// Score-weighted box voting.
    Vote,
    /// DFL evidence fusion: score times per-side precision. A caller without the DFL logits (boxes-only inputs)
    /// gives every side the same precision, which is [`BoxFusion::Vote`].
    Dfl,
    /// Student-t Exclusive Box Fusion ([`fuse_boxes_sebf`], docs/MODEL.md §9.4): exclusive NMS suppressor
    /// ownership + `IoU⁴` affinity + 1-step Student-t (`ν = 3`) IRLS outlier rejection per side, using the same
    /// per-side DFL precisions as [`BoxFusion::Dfl`].
    Sebf,
}

impl BoxFusion {
    /// The name used by the CLI and in JSON (`none`, `vote`, `dfl`, `sebf`).
    pub fn as_str(self) -> &'static str {
        match self {
            BoxFusion::None => "none",
            BoxFusion::Vote => "vote",
            BoxFusion::Dfl => "dfl",
            BoxFusion::Sebf => "sebf",
        }
    }

    /// Whether this fusion rule uses per-side DFL precisions ([`crate::decode::side_precision`]).
    #[inline]
    pub fn wants_side_precision(self) -> bool {
        matches!(self, BoxFusion::Dfl | BoxFusion::Sebf)
    }

    /// Whether this fusion rule uses exclusive NMS suppressor ownership.
    #[inline]
    pub fn wants_owner(self) -> bool {
        self == BoxFusion::Sebf
    }
}

/// Student-t Exclusive Box Fusion (`BoxFusion::Sebf`, docs/MODEL.md §9.4): each kept box `k` fuses only candidates
/// exclusively owned by `k` in NMS (`owner[j] == kp`) with `IoU(b_k, b_j) >= iou` (`j == k` always included),
/// weighting each side by `s_j · IoU⁴ · prec(j)[e]`, followed by one Student-t (`ν = 3`) IRLS refinement step
/// with variance floor `max(1 / prec(j)[e], (0.03 · L_e)²)` to reject cross-object outliers.
#[allow(clippy::too_many_arguments)]
pub fn fuse_boxes_sebf(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    agnostic: bool,
    iou: f32,
    owner: Option<&[u32]>,
    nms_iou: f32,
    precision: Option<&dyn Fn(usize) -> [f32; 4]>,
) -> Vec<Xyxy> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let buckets = Buckets::new(boxes.len(), classes, agnostic);
    let computed_owners;
    let owners: &[u32] = match owner {
        Some(o) => {
            assert_eq!(o.len(), boxes.len(), "fuse_boxes_sebf: owner length mismatch");
            o
        }
        None => {
            computed_owners = class_local_owners(kept, boxes, scores, classes, &buckets, nms_iou);
            &computed_owners
        }
    };
    let mut members: Vec<(usize, f32, [f32; 4])> = Vec::new();
    kept.iter()
        .enumerate()
        .map(|(kp, &k)| {
            let b = boxes[k];
            if !finite(&b) {
                return b;
            }
            let ba = b.area();
            let kp_u32 = kp as u32;
            members.clear();
            for &j in buckets.of(classes[k]) {
                let j = j as usize;
                let c = &boxes[j];
                if j == k {
                    if scores[j].is_finite() && scores[j] > 0.0 {
                        members.push((j, 1.0, precision.map_or([1.0; 4], |f| f(j))));
                    }
                } else if owners[j] == kp_u32 && finite(c) && scores[j].is_finite() && scores[j] > 0.0 {
                    let o = overlap(&b, ba, c);
                    if !o.is_nan() && o >= iou {
                        members.push((j, o, precision.map_or([1.0; 4], |f| f(j))));
                    }
                }
            }
            if members.len() <= 1 {
                return b;
            }
            let old = b.to_array();
            let (mut acc1, mut sum1) = ([0f64; 4], [0f64; 4]);
            for &(j, o0, p) in &members {
                let s = scores[j] as f64;
                let o2 = (o0 as f64) * (o0 as f64);
                let aff = o2 * o2;
                let x = boxes[j].to_array();
                for e in 0..4 {
                    let pe = p[e] as f64;
                    if pe.is_finite() && pe > 0.0 {
                        let wt = s * aff * pe;
                        acc1[e] += wt * x[e] as f64;
                        sum1[e] += wt;
                    }
                }
            }
            let mut b1 = [0f64; 4];
            for e in 0..4 {
                b1[e] = if sum1[e] > 0.0 { acc1[e] / sum1[e] } else { old[e] as f64 };
            }
            let b1_xyxy = Xyxy::new(b1[0] as f32, b1[1] as f32, b1[2] as f32, b1[3] as f32);
            let b1_area = b1_xyxy.area();
            let bw = (b1[2] - b1[0]).max(1.0);
            let bh = (b1[3] - b1[1]).max(1.0);
            let c2 = [
                (SEBF_C_REL * bw) * (SEBF_C_REL * bw),
                (SEBF_C_REL * bh) * (SEBF_C_REL * bh),
                (SEBF_C_REL * bw) * (SEBF_C_REL * bw),
                (SEBF_C_REL * bh) * (SEBF_C_REL * bh),
            ];
            let (mut acc2, mut sum2) = ([0f64; 4], [0f64; 4]);
            for &(j, _, p) in &members {
                let s = scores[j] as f64;
                let o1_raw = overlap(&b1_xyxy, b1_area, &boxes[j]) as f64;
                let o1 = if j == k { o1_raw.max(1.0) } else { o1_raw };
                let o1_2 = o1 * o1;
                let aff2 = o1_2 * o1_2;
                let x = boxes[j].to_array();
                for e in 0..4 {
                    let pe = p[e] as f64;
                    if pe.is_finite() && pe > 0.0 {
                        let v_raw = if precision.is_some() { 1.0 / pe } else { 0.0 };
                        let v_tilde = v_raw.max(c2[e]);
                        let diff = x[e] as f64 - b1[e];
                        let lam_t = (SEBF_NU + 1.0) / (SEBF_NU + (diff * diff) / v_tilde);
                        let wt = s * aff2 * pe * lam_t;
                        acc2[e] += wt * x[e] as f64;
                        sum2[e] += wt;
                    }
                }
            }
            let mut b2 = [0f64; 4];
            for e in 0..4 {
                b2[e] = if sum2[e] > 0.0 { acc2[e] / sum2[e] } else { b1[e] };
            }
            if b2[2] >= b2[0] && b2[3] >= b2[1] {
                Xyxy::new(b2[0] as f32, b2[1] as f32, b2[2] as f32, b2[3] as f32)
            } else {
                b1_xyxy
            }
        })
        .collect()
}

/// The fused boxes of the `kept` candidates (indices into `boxes` / `scores` / `classes`, the candidates that
/// entered NMS), in `kept`'s order. Member `j` weighs `scores[j] · precision(j)[side]` on each side, with
/// `precision` `None` meaning 1 (voting); the sums run in `f64` over the cluster in candidate order. A member side
/// whose weight is not `> 0` (a zero score, a NaN) or whose edge is not finite is skipped, and a side with no weight
/// keeps the kept box's edge.
pub fn fuse_boxes(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    agnostic: bool,
    iou: f32,
    precision: Option<&dyn Fn(usize) -> [f32; 4]>,
) -> Vec<Xyxy> {
    let buckets = Buckets::new(boxes.len(), classes, agnostic);
    kept.iter()
        .map(|&k| {
            let b = boxes[k];
            let ba = b.area();
            let mut acc = [0f64; 4];
            let mut sum = [0f64; 4];
            for &j in buckets.of(classes[k]) {
                let j = j as usize;
                let c = &boxes[j];
                let o = overlap(&b, ba, c);
                if o.is_nan() || o < iou {
                    continue;
                }
                let s = scores[j] as f64;
                let p = precision.map_or([1.0; 4], |f| f(j));
                let x = c.to_array();
                for e in 0..4 {
                    let wt = s * p[e] as f64;
                    // A non-finite edge (a NaN box can join a cluster: its IoU comes out as 1) is left out.
                    if wt > 0.0 && x[e].is_finite() {
                        acc[e] += wt * x[e] as f64;
                        sum[e] += wt;
                    }
                }
            }
            let old = b.to_array();
            let e = |i: usize| if sum[i] > 0.0 { (acc[i] / sum[i]) as f32 } else { old[i] };
            Xyxy::new(e(0), e(1), e(2), e(3))
        })
        .collect()
}

#[cfg(test)]
mod tests;
