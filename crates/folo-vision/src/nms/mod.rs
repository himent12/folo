//! Non-maximum suppression.
//!
//! - [`nms_sorted`]: greedy single-set NMS over boxes already sorted by descending score (the kernel).
//! - [`batched_nms`]: class-aware NMS over arbitrary boxes, scores and classes. Boxes of different classes never
//!   suppress each other (unless `agnostic`).
//! - [`non_max_suppression`]: the full detector post-process from decoded head outputs (boxes `[A]`, class scores
//!   `[A, C]`), mirroring Ultralytics' `non_max_suppression` (confidence filter, optional multi-label, class filter,
//!   candidate cap, batched NMS, `max_det`).
//! - [`nms_detections`]: the same filters applied to an existing list of [`Detection`]s.
//! - [`topk_detections`] / [`topk_candidates`]: the NMS-free selection of the one-to-one head (docs/MODEL.md
//!   §6.6), and [`postprocess`], which picks it or [`non_max_suppression`] by [`HeadMode`].
//!
//! **Semantics** (identical to torchvision / Ultralytics): candidates are those with `score > conf` (strict); they
//! are visited in descending score order (ties: lower input index first); a box is suppressed when its IoU with an
//! already kept box of the same class is `> iou` (strict). Output is sorted by descending score.
//!
//! **Fast path.** The candidates are sorted once; per-class buckets are then filled by a stable counting sort, so
//! each bucket is already in score order and no per-class sort is needed. Buckets run in parallel (rayon) when there
//! is enough work. Inside a bucket the boxes are laid out structure-of-arrays and every kept box suppresses all later
//! boxes in one branch-free pass the compiler vectorises (split across threads for very large buckets); a bucket stops
//! as soon as it has `max_det` survivors, since nothing after that can reach the output. With `max_det = 300` the
//! cost is therefore `O(n log n + max_det * n)` instead of the `O(n²)` worst case of textbook greedy NMS.

use crate::boxes::Xyxy;
use crate::fusion::{
    BoxFusion, FUSION_IOU, ScoreRule, consensus_scores, dfl_cgr_scores, dfl_consensus_scores,
    dfl_repulse_scores, elc_scores, fuse_boxes, fuse_boxes_sebf,
};
use crate::model::HeadMode;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod kernel;
mod matrix;
mod topk;
pub use kernel::{
    batched_nms, batched_nms_with_kind, diou_nms_sorted, diou_nms_sorted_with_owner, nms_sorted,
    nms_sorted_with_owner, soft_nms_sorted,
};
pub(crate) use matrix::exact_iou_f64;
pub use matrix::matrix_nms_sorted;
pub use topk::{topk_candidates, topk_detections};

/// Default Gaussian bandwidth `sigma` for [`NmsKind::SoftGaussian`] (Bodla et al. 2017).
pub const DEFAULT_SOFT_NMS_SIGMA: f32 = 0.5;

fn default_soft_nms_sigma() -> f32 {
    DEFAULT_SOFT_NMS_SIGMA
}

/// Suppression rule used by NMS (default: [`NmsKind::Standard`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NmsKind {
    /// Standard greedy IoU NMS: suppress when `IoU > iou`.
    #[default]
    Standard,
    /// DIoU-NMS (Zheng et al. 2020): suppress when `IoU − rho² / c² > iou`, keeping overlapping boxes whose
    /// centres are distinct.
    Diou,
    /// Linear Soft-NMS (Bodla et al. 2017): when `IoU > iou`, decay `score *= (1 − IoU)` instead of
    /// hard-dropping; drop once `score <= conf`.
    SoftLinear,
    /// Gaussian Soft-NMS (Bodla et al. 2017): when `IoU > 0`, decay `score *= exp(−IoU² / sigma)`; drop once
    /// `score <= conf`.
    SoftGaussian {
        /// Gaussian decay bandwidth `sigma > 0` (default [`DEFAULT_SOFT_NMS_SIGMA`]).
        #[serde(default = "default_soft_nms_sigma")]
        sigma: f32,
    },
    /// Linear Matrix-NMS (Wang et al. 2020, SOLOv2): non-sequential parallel score decay
    /// `decay_j = min_{i < j} (1 - IoU_{i,j}) / (1 - max_{k < i} IoU_{k,i})`, dropping once `score <= conf`.
    MatrixLinear,
    /// Gaussian Matrix-NMS (Wang et al. 2020, SOLOv2): non-sequential parallel score decay
    /// `decay_j = min_{i < j} exp(-(IoU_{i,j}² - max_{k < i} IoU_{k,i}²) / sigma)`, dropping once `score <= conf`.
    MatrixGaussian {
        /// Gaussian decay bandwidth `sigma > 0` (default [`DEFAULT_SOFT_NMS_SIGMA`]).
        #[serde(default = "default_soft_nms_sigma")]
        sigma: f32,
    },
}

impl NmsKind {
    fn is_standard(&self) -> bool {
        matches!(self, NmsKind::Standard)
    }

    /// True for score-decaying NMS variants (`SoftLinear` / `SoftGaussian` / `MatrixLinear` / `MatrixGaussian`),
    /// which decay scores instead of hard-dropping and therefore have no hard suppression clusters.
    #[inline]
    pub fn is_soft(&self) -> bool {
        matches!(
            self,
            NmsKind::SoftLinear
                | NmsKind::SoftGaussian { .. }
                | NmsKind::MatrixLinear
                | NmsKind::MatrixGaussian { .. }
        )
    }

    /// True for Matrix-NMS variants (`MatrixLinear` / `MatrixGaussian`).
    #[inline]
    pub fn is_matrix(&self) -> bool {
        matches!(self, NmsKind::MatrixLinear | NmsKind::MatrixGaussian { .. })
    }
}

/// Pre-NMS extraction threshold for single-label calibrated scoring (`Elc` / `DflConsensus` / `DflRepulse` /
/// `DflCgr`) when `conf > 0.10`: nearby supporting anchors down to `0.10` join the cluster before final scores are
/// filtered by `score > conf`. When `conf <= 0.10` (e.g. evaluation at `0.001`), pre-NMS extraction uses `conf` and
/// calibrated scores below `conf` are kept without post-filtering.
pub const CALIBRATED_PRE_CONF: f32 = 0.10;

/// Anchor-row count above which candidate extraction runs in parallel.
const PAR_ROWS: usize = 4096;

/// One detection: a box, its confidence and its class index.
///
/// Serialises flat as `{ "x1", "y1", "x2", "y2", "score", "class" }` (the `ScoredBox` shape of `docs/API.md` minus
/// `name`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// The box, `xyxy`.
    #[serde(flatten)]
    pub bbox: Xyxy,
    /// Confidence in `[0, 1]`.
    pub score: f32,
    /// Class index.
    pub class: u32,
}

/// Post-processing settings. Defaults are Ultralytics' predict defaults; [`NmsConfig::val`] gives its validation
/// settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NmsConfig {
    /// Keep candidates with `score > conf`.
    pub conf: f32,
    /// Suppress a box when its IoU with a kept box is `> iou` (used by `Standard`, `Diou`, and `SoftLinear`;
    /// ignored by `SoftGaussian`, `MatrixLinear`, and `MatrixGaussian`).
    pub iou: f32,
    /// Maximum detections returned per image.
    pub max_det: usize,
    /// Only the `max_candidates` highest-scoring candidates enter NMS (Ultralytics' `max_nms`).
    pub max_candidates: usize,
    /// Keep only these classes (applied after the per-anchor best class is chosen, as Ultralytics does).
    pub classes: Option<Vec<u32>>,
    /// Class-agnostic NMS: boxes of different classes suppress each other.
    pub agnostic: bool,
    /// One candidate per (anchor, class) pair above `conf` instead of one per anchor (its best class).
    pub multi_label: bool,
    /// Refine each kept box from its NMS cluster ([`crate::fusion`], docs/MODEL.md §9.4). Off by default. Not used
    /// by the NMS-free `o2o` selection, which has no clusters. Under Soft-NMS (`SoftLinear` / `SoftGaussian`),
    /// `BoxFusion::Sebf` has no hard suppression clusters and keeps each survivor's own box (validation rejects it).
    pub fusion: BoxFusion,
    /// The cluster of a kept box for [`NmsConfig::fusion`]: the candidates of its class (any class with
    /// `agnostic`) whose IoU with it is `>=` this.
    pub fusion_iou: f32,
    /// How the kept boxes are scored ([`ScoreRule`], docs/MODEL.md §9.4). Off by default: the consensus score
    /// changes the score scale. Not used by the NMS-free `o2o` selection or Soft-NMS, which have no hard
    /// suppression clusters.
    pub score: ScoreRule,
    /// Suppression rule ([`NmsKind::Standard`] by default; opt-in DIoU-NMS, Soft-NMS, or Matrix-NMS).
    #[serde(default, skip_serializing_if = "NmsKind::is_standard")]
    pub nms_kind: NmsKind,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            conf: 0.25,
            iou: 0.7,
            max_det: 300,
            max_candidates: 30_000,
            classes: None,
            agnostic: false,
            multi_label: false,
            fusion: BoxFusion::None,
            fusion_iou: FUSION_IOU,
            score: ScoreRule::None,
            nms_kind: NmsKind::Standard,
        }
    }
}

impl NmsConfig {
    /// Validation settings: `conf = 0.001`, `iou = 0.7`, `max_det = 300`, multi-label.
    pub fn val() -> Self {
        Self { conf: 0.001, multi_label: true, ..Self::default() }
    }

    /// Pre-NMS candidate threshold: `0.10` for single-label calibrated scoring (`Elc` / `DflConsensus` /
    /// `DflRepulse` / `DflCgr`) when `conf > 0.10`, else `self.conf` unchanged (for `conf <= 0.10`, post-calibration
    /// scores below `conf` are kept in full).
    #[inline]
    pub fn candidate_conf(&self) -> f32 {
        if self.post_filter_calibrated() { CALIBRATED_PRE_CONF } else { self.conf }
    }

    /// Like [`NmsConfig::candidate_conf`], but returns `self.conf` unchanged for [`HeadMode::O2o`]
    /// (which skips NMS and score rules).
    #[inline]
    pub fn candidate_conf_for(&self, mode: HeadMode) -> f32 {
        if mode == HeadMode::O2m { self.candidate_conf() } else { self.conf }
    }

    /// Whether final detections are filtered by `score > self.conf` after calibrated re-scoring (active for
    /// single-label calibrated scoring with `self.conf > CALIBRATED_PRE_CONF` under hard NMS; when `self.conf <= 0.10`,
    /// `multi_label` is set, or `nms_kind` is Soft-NMS, all surviving detections are retained).
    #[inline]
    pub(crate) fn post_filter_calibrated(&self) -> bool {
        !self.multi_label
            && !self.nms_kind.is_soft()
            && self.score.is_calibrated()
            && self.conf > CALIBRATED_PRE_CONF
    }
}

/// A compact NMS candidate: score, source index (anchor or input detection) and class.
#[derive(Clone, Copy, Debug)]
struct Cand {
    score: f32,
    idx: u32,
    class: u32,
}

/// Per-side precisions ([`crate::decode::side_precision`]) of the candidates' sources, by source index (anchor
/// or input position): what [`BoxFusion::Dfl`] and [`BoxFusion::Sebf`] weigh a cluster member's sides by.
/// `None`: equal precisions.
pub type Precisions<'a> = Option<&'a [[f32; 4]]>;

/// Optional per-source DFL and anchor metadata passed to post-processing ([`BoxFusion::Dfl`], [`BoxFusion::Sebf`],
/// [`ScoreRule::DflConsensus`], [`ScoreRule::DflRepulse`], [`ScoreRule::DflCgr`]), indexed by source position
/// (anchor or candidate).
#[derive(Clone, Copy, Debug, Default)]
pub struct DflInputs<'a> {
    /// Per-side DFL precisions ([`crate::decode::side_precision`]) for [`BoxFusion::Dfl`] / [`BoxFusion::Sebf`].
    pub precision: Precisions<'a>,
    /// Scale-free per-side DFL bin precisions ([`crate::decode::bin_precision`]) for [`ScoreRule::DflConsensus`].
    pub bin_precision: Precisions<'a>,
    /// Mean 4-side DFL Shannon entropy in nats ([`crate::decode::dfl_entropy`]) for [`ScoreRule::DflRepulse`] /
    /// [`ScoreRule::DflCgr`].
    pub entropy: Option<&'a [f32]>,
    /// Anchor stride in input pixels for [`ScoreRule::DflRepulse`] / [`ScoreRule::DflCgr`]'s cross-scale consensus.
    pub strides: Option<&'a [f32]>,
}

/// Reconstruct class-local DIoU suppressor ownership for calibrated score rules when agnostic DIoU-NMS ran.
/// Matches `fusion::reconstruct_owners` semantics (same class, lower-ranked candidate, finite boxes, first owner
/// wins) using `iou(..., IouKind::Diou) > diou_thr`.
fn class_local_diou_owners(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    diou_thr: f32,
) -> Vec<u32> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let score_key = |s: f32| if s.is_nan() { f32::NEG_INFINITY } else { s };
    let mut own = vec![u32::MAX; boxes.len()];
    for (kp, &k) in kept.iter().enumerate() {
        own[k] = kp as u32;
    }
    for (kp, &k) in kept.iter().enumerate() {
        let bk = boxes[k];
        if !finite(&bk) {
            continue;
        }
        let sk = score_key(scores[k]);
        let ck = classes[k];
        for j in 0..boxes.len() {
            if own[j] != u32::MAX || classes[j] != ck || !finite(&boxes[j]) {
                continue;
            }
            let sj = score_key(scores[j]);
            let lower_ranked = sj.total_cmp(&sk).is_lt() || (sj.total_cmp(&sk).is_eq() && j > k);
            if lower_ranked && crate::boxes::iou(&bk, &boxes[j], crate::boxes::IouKind::Diou) > diou_thr {
                own[j] = kp as u32;
            }
        }
    }
    own
}

/// Cap the candidates (given in `(idx, class)` order) at the `cap` best by score, ties to the earlier candidate,
/// then run batched NMS, fuse the kept boxes when [`NmsConfig::fusion`] asks for it (with `dfl.precision`, indexed
/// by source index, for [`BoxFusion::Dfl`]), re-score them when [`NmsConfig::score`] does (sorted by the new score,
/// ties in NMS's order), and build the detections. Every NMS path of every device ends here.
fn finish(
    mut cands: Vec<Cand>,
    cfg: &NmsConfig,
    bbox_of: impl Fn(u32) -> Xyxy,
    dfl: DflInputs<'_>,
) -> Vec<Detection> {
    let use_score_rule = cfg.score != ScoreRule::None && !cfg.nms_kind.is_soft();
    let calibrated = use_score_rule && cfg.score.is_calibrated();
    let (cands, boxes): (Vec<Cand>, Vec<Xyxy>) = if calibrated {
        // Filter non-finite scores (+Inf / NaN) and non-finite boxes before `max_candidates` truncation so
        // malformed high-scoring entries cannot evict valid candidates or survive calibrated NMS.
        let mut valid: Vec<(Cand, Xyxy)> = cands
            .into_iter()
            .filter_map(|c| {
                if !c.score.is_finite() {
                    return None;
                }
                let bx = bbox_of(c.idx);
                bx.to_array().iter().all(|v| v.is_finite()).then_some((c, bx))
            })
            .collect();
        if valid.len() > cfg.max_candidates {
            if cfg.max_candidates == 0 {
                return Vec::new();
            }
            let rank = |a: &(Cand, Xyxy), b: &(Cand, Xyxy)| {
                b.0.score.total_cmp(&a.0.score).then(a.0.idx.cmp(&b.0.idx)).then(a.0.class.cmp(&b.0.class))
            };
            valid.select_nth_unstable_by(cfg.max_candidates - 1, rank);
            valid.truncate(cfg.max_candidates);
            valid.sort_unstable_by_key(|(c, _)| (c.idx, c.class));
        }
        valid.into_iter().unzip()
    } else {
        if cfg.nms_kind.is_soft() {
            // Drop non-finite scores (+Inf / NaN) before `max_candidates` truncation so malformed entries
            // cannot evict valid detections from Soft-NMS or Matrix-NMS.
            cands.retain(|c| c.score.is_finite());
        }
        if cands.len() > cfg.max_candidates {
            if cfg.max_candidates == 0 {
                return Vec::new();
            }
            let key = |c: &Cand| if c.score.is_nan() { f32::NEG_INFINITY } else { c.score };
            let rank = |a: &Cand, b: &Cand| {
                key(b).total_cmp(&key(a)).then(a.idx.cmp(&b.idx)).then(a.class.cmp(&b.class))
            };
            cands.select_nth_unstable_by(cfg.max_candidates - 1, rank);
            cands.truncate(cfg.max_candidates);
            // Back to candidate order so NMS tie-breaks by position.
            cands.sort_unstable_by_key(|c| (c.idx, c.class));
        }
        let boxes = cands.iter().map(|c| bbox_of(c.idx)).collect();
        (cands, boxes)
    };
    let scores: Vec<f32> = cands.iter().map(|c| c.score).collect();
    let classes: Vec<u32> = cands.iter().map(|c| c.class).collect();
    let want_owner = cfg.fusion.wants_owner() || (!cfg.agnostic && calibrated);
    let (kept, kept_scores, owner) = match cfg.nms_kind {
        NmsKind::Standard | NmsKind::Diou => {
            let (kept, owner) = kernel::batched_nms_inner(
                &boxes,
                &scores,
                &classes,
                cfg.iou,
                cfg.max_det,
                cfg.agnostic,
                want_owner,
                cfg.nms_kind == NmsKind::Diou,
            );
            let ks: Vec<f32> = kept.iter().map(|&k| scores[k]).collect();
            (kept, ks, owner)
        }
        NmsKind::SoftLinear | NmsKind::SoftGaussian { .. } => {
            let sigma = match cfg.nms_kind {
                NmsKind::SoftGaussian { sigma } => Some(sigma),
                _ => None,
            };
            let pairs = kernel::batched_soft_nms(
                &boxes,
                &scores,
                &classes,
                cfg.iou,
                cfg.conf,
                sigma,
                cfg.max_det,
                cfg.agnostic,
            );
            let (kept, ks): (Vec<usize>, Vec<f32>) = pairs.into_iter().unzip();
            // Soft-NMS has no hard suppressor clusters; assign self-only owners when SEBF is requested so
            // `BoxFusion::Sebf` never reconstructs hard-IoU owners from `cfg.iou`.
            let self_own = cfg.fusion.wants_owner().then(|| {
                let mut own = vec![u32::MAX; boxes.len()];
                for (kp, &k) in kept.iter().enumerate() {
                    own[k] = kp as u32;
                }
                own
            });
            (kept, ks, self_own)
        }
        NmsKind::MatrixLinear | NmsKind::MatrixGaussian { .. } => {
            let sigma = match cfg.nms_kind {
                NmsKind::MatrixGaussian { sigma } => Some(sigma),
                _ => None,
            };
            let pairs = matrix::batched_matrix_nms(
                &boxes,
                &scores,
                &classes,
                cfg.conf,
                sigma,
                cfg.max_det,
                cfg.agnostic,
            );
            let (kept, ks): (Vec<usize>, Vec<f32>) = pairs.into_iter().unzip();
            let self_own = cfg.fusion.wants_owner().then(|| {
                let mut own = vec![u32::MAX; boxes.len()];
                for (kp, &k) in kept.iter().enumerate() {
                    own[k] = kp as u32;
                }
                own
            });
            (kept, ks, self_own)
        }
    };
    let cands_slice = cands.as_slice();
    let fused = match cfg.fusion {
        BoxFusion::None => None,
        BoxFusion::Vote => {
            Some(fuse_boxes(&kept, &boxes, &scores, &classes, cfg.agnostic, cfg.fusion_iou, None))
        }
        BoxFusion::Dfl => {
            let of = dfl.precision.map(|p| move |k: usize| p[cands_slice[k].idx as usize]);
            let of = of.as_ref().map(|f| f as &dyn Fn(usize) -> [f32; 4]);
            Some(fuse_boxes(&kept, &boxes, &scores, &classes, cfg.agnostic, cfg.fusion_iou, of))
        }
        BoxFusion::Sebf => {
            let of = dfl.precision.map(|p| move |k: usize| p[cands_slice[k].idx as usize]);
            let of = of.as_ref().map(|f| f as &dyn Fn(usize) -> [f32; 4]);
            Some(fuse_boxes_sebf(
                &kept,
                &boxes,
                &scores,
                &classes,
                cfg.agnostic,
                cfg.fusion_iou,
                owner.as_deref(),
                cfg.iou,
                of,
            ))
        }
    };
    let mut dets: Vec<Detection> = kept
        .iter()
        .enumerate()
        .map(|(i, &k)| Detection {
            bbox: fused.as_ref().map_or(boxes[k], |f| f[i]),
            score: kept_scores[i],
            class: classes[k],
        })
        .collect();
    // Calibrated score rules always use class-local ownership (docs/MODEL.md §9.4), so under `agnostic` NMS
    // reconstruct class-local owners matching the active hard-NMS metric (plain IoU inside `fusion`, or DIoU
    // here). Soft-NMS has no hard suppression clusters and already decays/reorders scores, so skip `ScoreRule`.
    let diou_local_owner = (cfg.agnostic && calibrated && cfg.nms_kind == NmsKind::Diou)
        .then(|| class_local_diou_owners(&kept, &boxes, &scores, &classes, cfg.iou));
    let score_owner = if cfg.agnostic { diou_local_owner.as_deref() } else { owner.as_deref() };
    let new_scores = if !use_score_rule {
        None
    } else {
        match cfg.score {
            ScoreRule::None => None,
            ScoreRule::Consensus => Some(consensus_scores(&kept, &boxes, &scores, &classes)),
            ScoreRule::Elc => Some(elc_scores(&kept, &boxes, &scores, &classes, score_owner, cfg.iou)),
            ScoreRule::DflConsensus => {
                let bp = dfl.bin_precision.map(|p| move |k: usize| p[cands_slice[k].idx as usize]);
                let bp = bp.as_ref().map(|f| f as &dyn Fn(usize) -> [f32; 4]);
                Some(dfl_consensus_scores(&kept, &boxes, &scores, &classes, score_owner, cfg.iou, bp))
            }
            ScoreRule::DflRepulse | ScoreRule::DflCgr => {
                let repulse_boxes: Vec<Xyxy> = dets.iter().map(|d| d.bbox).collect();
                let ent = dfl.entropy.map(|e| move |k: usize| e[cands_slice[k].idx as usize]);
                let ent = ent.as_ref().map(|f| f as &dyn Fn(usize) -> f32);
                let st = dfl.strides.map(|s| move |k: usize| s[cands_slice[k].idx as usize]);
                let st = st.as_ref().map(|f| f as &dyn Fn(usize) -> f32);
                let scorer = if cfg.score == ScoreRule::DflCgr { dfl_cgr_scores } else { dfl_repulse_scores };
                Some(scorer(&kept, &boxes, &repulse_boxes, &scores, &classes, score_owner, cfg.iou, ent, st))
            }
        }
    };
    if let Some(new_scores) = new_scores {
        for (d, s) in dets.iter_mut().zip(new_scores) {
            d.score = s;
        }
        // Stable: equal new scores stay in NMS's order.
        dets.sort_by(|a, b| b.score.total_cmp(&a.score));
        if cfg.post_filter_calibrated() {
            dets.retain(|d| d.score > cfg.conf);
        }
    }
    dets
}

pub(crate) fn class_allowed(cfg: &NmsConfig, c: u32) -> bool {
    cfg.classes.as_ref().is_none_or(|cs| cs.contains(&c))
}

/// Apply the confidence filter, class filter, candidate cap and batched NMS to a list of detections.
///
/// # Panics
/// If there are more than `u32::MAX` detections.
pub fn nms_detections(dets: &[Detection], cfg: &NmsConfig) -> Vec<Detection> {
    assert!(dets.len() <= u32::MAX as usize, "nms_detections: too many detections");
    let conf = cfg.candidate_conf();
    let cands: Vec<Cand> = dets
        .iter()
        .enumerate()
        .filter(|(_, d)| d.score > conf && class_allowed(cfg, d.class))
        .map(|(i, d)| Cand { score: d.score, idx: i as u32, class: d.class })
        .collect();
    finish(cands, cfg, |i| dets[i as usize].bbox, DflInputs::default())
}

/// The detector post-process for one image.
///
/// - `boxes`: `[A]` decoded boxes, `xyxy` (see [`crate::decode`]).
/// - `scores`: `[A, num_classes]` row-major class probabilities (post-sigmoid).
///
/// Candidates are one per anchor (its best class; first index on ties) or, with `multi_label`, one per
/// (anchor, class) pair, filtered by `score > conf` and the class filter, capped at `max_candidates`, then batched
/// NMS. Returns at most `max_det` detections sorted by descending score.
///
/// # Panics
/// If `scores.len() != boxes.len() * num_classes`, or there are more than `u32::MAX` anchors.
pub fn non_max_suppression(
    boxes: &[Xyxy],
    scores: &[f32],
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_with_inputs(boxes, scores, DflInputs::default(), num_classes, cfg)
}

/// [`non_max_suppression`] with the anchors' side precisions (`[A]`, [`crate::decode::side_precisions`]) for
/// [`BoxFusion::Dfl`]; `None` weighs every side alike.
///
/// # Panics
/// As [`non_max_suppression`], or if `precision` is not `[A]`.
pub fn non_max_suppression_with(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_with_inputs(
        boxes,
        scores,
        DflInputs { precision, ..Default::default() },
        num_classes,
        cfg,
    )
}

/// [`non_max_suppression_with`] also accepting scale-free per-anchor DFL bin precisions (`[A]`,
/// [`crate::decode::bin_precisions`]) for [`ScoreRule::DflConsensus`].
pub fn non_max_suppression_with_dfl(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    bin_precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    non_max_suppression_with_inputs(boxes, scores, dfl, num_classes, cfg)
}

/// [`non_max_suppression`] accepting all optional per-anchor [`DflInputs`] (`[A]`).
pub fn non_max_suppression_with_inputs(
    boxes: &[Xyxy],
    scores: &[f32],
    dfl: DflInputs<'_>,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let a = boxes.len();
    assert!(dfl.precision.is_none_or(|p| p.len() == a), "non_max_suppression: precision must be [A]");
    assert!(dfl.bin_precision.is_none_or(|p| p.len() == a), "non_max_suppression: bin_precision must be [A]");
    assert!(dfl.entropy.is_none_or(|e| e.len() == a), "non_max_suppression: entropy must be [A]");
    assert!(dfl.strides.is_none_or(|s| s.len() == a), "non_max_suppression: strides must be [A]");
    assert_eq!(scores.len(), a * num_classes, "non_max_suppression: scores must be [A, num_classes]");
    assert!(a <= u32::MAX as usize, "non_max_suppression: too many anchors");
    if num_classes == 0 || a == 0 {
        return Vec::new();
    }
    let cands = extract_candidates(scores, a, num_classes, num_classes, 0, cfg);
    finish(cands, cfg, |i| boxes[i as usize], dfl)
}

/// [`non_max_suppression`] of one image's decoded outputs as rows `[A, 4 + num_classes]` (`x1, y1, x2, y2`, then
/// the class scores: [`crate::model::HeadOutput::decoded_on_device`]'s layout), read in place. Same result.
///
/// # Panics
/// If `rows.len()` is not a multiple of `4 + num_classes`, or there are more than `u32::MAX` anchors.
pub fn non_max_suppression_rows(rows: &[f32], num_classes: usize, cfg: &NmsConfig) -> Vec<Detection> {
    non_max_suppression_rows_with_inputs(rows, DflInputs::default(), num_classes, cfg)
}

/// [`non_max_suppression_rows`] with the anchors' side precisions (`[A]`) for [`BoxFusion::Dfl`].
///
/// # Panics
/// As [`non_max_suppression_rows`], or if `precision` is not `[A]`.
pub fn non_max_suppression_rows_with(
    rows: &[f32],
    precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_rows_with_inputs(
        rows,
        DflInputs { precision, ..Default::default() },
        num_classes,
        cfg,
    )
}

/// [`non_max_suppression_rows_with`] also accepting scale-free DFL bin precisions (`[A]`) for
/// [`ScoreRule::DflConsensus`].
pub fn non_max_suppression_rows_with_dfl(
    rows: &[f32],
    precision: Precisions,
    bin_precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    non_max_suppression_rows_with_inputs(rows, dfl, num_classes, cfg)
}

/// [`non_max_suppression_rows`] accepting all optional per-anchor [`DflInputs`] (`[A]`).
pub fn non_max_suppression_rows_with_inputs(
    rows: &[f32],
    dfl: DflInputs<'_>,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let w = 4 + num_classes;
    assert!(rows.len().is_multiple_of(w), "non_max_suppression_rows: rows must be [A, 4 + num_classes]");
    let a = rows.len() / w;
    assert!(a <= u32::MAX as usize, "non_max_suppression_rows: too many anchors");
    if num_classes == 0 || a == 0 {
        return Vec::new();
    }
    assert!(dfl.precision.is_none_or(|p| p.len() == a), "non_max_suppression_rows: precision must be [A]");
    assert!(
        dfl.bin_precision.is_none_or(|p| p.len() == a),
        "non_max_suppression_rows: bin_precision must be [A]"
    );
    assert!(dfl.entropy.is_none_or(|e| e.len() == a), "non_max_suppression_rows: entropy must be [A]");
    assert!(dfl.strides.is_none_or(|s| s.len() == a), "non_max_suppression_rows: strides must be [A]");
    let cands = extract_candidates(rows, a, num_classes, w, 4, cfg);
    let bbox_of = |i: u32| {
        let r = &rows[i as usize * w..];
        Xyxy::new(r[0], r[1], r[2], r[3])
    };
    finish(cands, cfg, bbox_of, dfl)
}

/// Multi-label [`non_max_suppression_rows`] from the `(anchor, class, score)` pairs whose score is `> conf`, in any
/// order (what folo-gpu's `detect::multi_label_candidates` downloads instead of every anchor's scores), with the
/// box of anchor `i` from `bbox_of(i)`. Given exactly the pairs above `conf` of the same scores, the result is that
/// of [`non_max_suppression_rows`] (the pairs are put back in its `(anchor, class)` order). Needs `multi_label`.
pub fn non_max_suppression_pairs(
    pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_pairs_with_inputs(pairs, bbox_of, DflInputs::default(), cfg)
}

/// [`non_max_suppression_pairs`] with the anchors' side precisions (indexed by anchor) for [`BoxFusion::Dfl`].
pub fn non_max_suppression_pairs_with(
    pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    precision: Precisions,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_pairs_with_inputs(pairs, bbox_of, DflInputs { precision, ..Default::default() }, cfg)
}

/// [`non_max_suppression_pairs_with`] also accepting scale-free DFL bin precisions (indexed by anchor) for
/// [`ScoreRule::DflConsensus`].
pub fn non_max_suppression_pairs_with_dfl(
    pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    precision: Precisions,
    bin_precision: Precisions,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    non_max_suppression_pairs_with_inputs(pairs, bbox_of, dfl, cfg)
}

/// [`non_max_suppression_pairs`] accepting all optional per-anchor [`DflInputs`] (indexed by anchor).
pub fn non_max_suppression_pairs_with_inputs(
    mut pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    dfl: DflInputs<'_>,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    debug_assert!(cfg.multi_label, "non_max_suppression_pairs is the multi-label path");
    let conf = cfg.candidate_conf();
    pairs.sort_unstable_by_key(|&(a, c, _)| (a, c));
    let cands = pairs
        .into_iter()
        .filter(|&(_, c, s)| s > conf && class_allowed(cfg, c))
        .map(|(idx, class, score)| Cand { score, idx, class })
        .collect();
    finish(cands, cfg, bbox_of, dfl)
}

/// The candidates of `anchors` score rows (row `a` is `data[a * stride + offset..][..num_classes]`), in anchor
/// order: every class above `conf` with `multi_label`, else the anchor's best class (first on ties) if above.
fn extract_candidates(
    data: &[f32],
    anchors: usize,
    num_classes: usize,
    stride: usize,
    offset: usize,
    cfg: &NmsConfig,
) -> Vec<Cand> {
    let conf = cfg.candidate_conf();
    let extract = |a: usize, row: &[f32], out: &mut Vec<Cand>| {
        if cfg.multi_label {
            for (c, &s) in row.iter().enumerate() {
                if s > conf && class_allowed(cfg, c as u32) {
                    out.push(Cand { score: s, idx: a as u32, class: c as u32 });
                }
            }
        } else {
            let (mut best, mut bs) = (0usize, row[0]);
            for (c, &s) in row.iter().enumerate().skip(1) {
                if s > bs {
                    best = c;
                    bs = s;
                }
            }
            if bs > conf && class_allowed(cfg, best as u32) {
                out.push(Cand { score: bs, idx: a as u32, class: best as u32 });
            }
        }
    };
    let rows = |lo: usize, hi: usize, out: &mut Vec<Cand>| {
        for a in lo..hi {
            extract(a, &data[a * stride + offset..][..num_classes], out);
        }
    };
    if anchors >= PAR_ROWS {
        const ROWS: usize = 1024;
        let parts: Vec<Vec<Cand>> = (0..anchors.div_ceil(ROWS))
            .into_par_iter()
            .map(|ci| {
                let mut out = Vec::new();
                rows(ci * ROWS, ((ci + 1) * ROWS).min(anchors), &mut out);
                out
            })
            .collect();
        parts.concat()
    } else {
        let mut out = Vec::new();
        rows(0, anchors, &mut out);
        out
    }
}

/// An anchor's best class above the confidence threshold, with its decoded box: what a device decode
/// downloads instead of every anchor's scores (folo-core's `Tensor::detect_candidates`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Anchor index.
    pub anchor: u32,
    /// Its best class (first maximum).
    pub class: u32,
    /// That class's score.
    pub score: f32,
    /// The decoded box, `xyxy`.
    pub bbox: Xyxy,
}

/// [`non_max_suppression`] (single-label) from the candidates already extracted: one per anchor whose best
/// class scores `> conf`, **sorted by anchor**. Applies the class filter, the candidate cap and batched NMS;
/// for the candidates of the same decoded outputs the result is identical to [`non_max_suppression`] with
/// `multi_label` off.
///
/// # Panics
/// If there are more than `u32::MAX` candidates.
pub fn nms_candidates(cands: &[Candidate], cfg: &NmsConfig) -> Vec<Detection> {
    nms_candidates_with(cands, None, cfg)
}

/// [`nms_candidates`] with each candidate's side precisions (parallel to `cands`) for [`BoxFusion::Dfl`].
///
/// # Panics
/// As [`nms_candidates`], or if `precision` is not as long as `cands`.
pub fn nms_candidates_with(cands: &[Candidate], precision: Precisions, cfg: &NmsConfig) -> Vec<Detection> {
    nms_candidates_with_dfl(cands, precision, None, cfg)
}

/// [`nms_candidates_with`] also accepting scale-free DFL bin precisions (parallel to `cands`) for
/// [`ScoreRule::DflConsensus`].
pub fn nms_candidates_with_dfl(
    cands: &[Candidate],
    precision: Precisions,
    bin_precision: Precisions,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    nms_candidates_with_inputs(cands, dfl, cfg)
}

/// [`nms_candidates`] accepting all optional per-candidate [`DflInputs`] (parallel to `cands`).
pub fn nms_candidates_with_inputs(
    cands: &[Candidate],
    dfl: DflInputs<'_>,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let n = cands.len();
    assert!(n <= u32::MAX as usize, "nms_candidates: too many candidates");
    assert!(dfl.precision.is_none_or(|p| p.len() == n), "nms_candidates: precision must match cands");
    assert!(dfl.bin_precision.is_none_or(|p| p.len() == n), "nms_candidates: bin_precision must match cands");
    assert!(dfl.entropy.is_none_or(|e| e.len() == n), "nms_candidates: entropy must match cands");
    assert!(dfl.strides.is_none_or(|s| s.len() == n), "nms_candidates: strides must match cands");
    debug_assert!(cands.windows(2).all(|w| w[0].anchor < w[1].anchor), "nms_candidates: unsorted");
    let conf = cfg.candidate_conf();
    // Positions in anchor order tie-break exactly as anchor indices do.
    let list: Vec<Cand> = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.score > conf && class_allowed(cfg, c.class))
        .map(|(i, c)| Cand { score: c.score, idx: i as u32, class: c.class })
        .collect();
    finish(list, cfg, |i| cands[i as usize].bbox, dfl)
}

/// The post-process of one image's decoded outputs for the branch that produced them: [`non_max_suppression`]
/// (`o2m`) or [`topk_detections`] (`o2o`, no NMS).
pub fn postprocess(
    boxes: &[Xyxy],
    scores: &[f32],
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    postprocess_with_inputs(boxes, scores, DflInputs::default(), num_classes, cfg, mode)
}

/// [`postprocess`] with the anchors' side precisions (`[A]`) for [`BoxFusion::Dfl`] (unused by `o2o`).
pub fn postprocess_with(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    postprocess_with_inputs(
        boxes,
        scores,
        DflInputs { precision, ..Default::default() },
        num_classes,
        cfg,
        mode,
    )
}

/// [`postprocess_with`] also accepting scale-free DFL bin precisions (`[A]`) for [`ScoreRule::DflConsensus`].
pub fn postprocess_with_dfl(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    bin_precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    postprocess_with_inputs(boxes, scores, dfl, num_classes, cfg, mode)
}

/// [`postprocess`] accepting all optional per-anchor [`DflInputs`] (`[A]`, unused by `o2o`).
pub fn postprocess_with_inputs(
    boxes: &[Xyxy],
    scores: &[f32],
    dfl: DflInputs<'_>,
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    match mode {
        HeadMode::O2m => non_max_suppression_with_inputs(boxes, scores, dfl, num_classes, cfg),
        HeadMode::O2o => topk_detections(boxes, scores, num_classes, cfg),
    }
}

#[cfg(test)]
mod tests;
