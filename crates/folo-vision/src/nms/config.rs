//! NMS configuration, suppression kinds, and detection output types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::fusion::{BoxFusion, FUSION_IOU, ScoreRule};
use crate::model::HeadMode;

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
    pub(crate) fn is_standard(&self) -> bool {
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
    /// Keep candidates with `score > conf` (fallback for classes not present in [`NmsConfig::per_class_conf`]).
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
    /// Optional per-class confidence thresholds (`class_id -> conf`). Classes present in this map use their mapped
    /// threshold instead of [`NmsConfig::conf`]; all other classes fall back to [`NmsConfig::conf`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_class_conf: Option<BTreeMap<u32, f32>>,
    /// Use inclusive (`score >= conf`) thresholding and enforce thresholds on final post-rescoring scores under
    /// every [`ScoreRule`], matching [`crate::Evaluator`]'s `score >= best_conf` operating point. Defaults to
    /// `false` (Ultralytics / torchvision strict `>` semantics) and is enabled automatically by
    /// [`crate::Predictor::apply_calibrated_thresholds`] / [`crate::Predictor::with_calibrated_thresholds`].
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub conf_inclusive: bool,
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
            per_class_conf: None,
            conf_inclusive: false,
        }
    }
}

impl NmsConfig {
    /// Validation settings: `conf = 0.001`, `iou = 0.7`, `max_det = 300`, multi-label.
    pub fn val() -> Self {
        Self { conf: 0.001, multi_label: true, ..Self::default() }
    }

    /// Builder attaching per-class confidence thresholds (`class_id -> conf`).
    pub fn with_per_class_conf(mut self, thresholds: BTreeMap<u32, f32>) -> Self {
        self.per_class_conf = Some(thresholds);
        self
    }

    /// Whether `score` passes threshold `thr` under [`NmsConfig::conf_inclusive`] (`score >= thr` when `true`,
    /// `score > thr` when `false`).
    #[inline]
    pub fn score_passes(&self, score: f32, thr: f32) -> bool {
        if self.conf_inclusive { score >= thr } else { score > thr }
    }

    /// Strict lower bound `floor` such that `score > floor` retains every finite `f32` passing `thr` under
    /// [`NmsConfig::score_passes`]. Used by scalar extraction kernels (CPU `single_label_block` and GPU
    /// `detect_candidates`) that implement strict `score > floor`; host filtering then applies exact
    /// [`NmsConfig::score_passes`]. When `thr <= f32::MIN_POSITIVE` under `conf_inclusive`, returns
    /// `-f32::MIN_POSITIVE` (a normal negative float that GPUs never flush to zero, so `0.0 > floor` holds).
    #[inline]
    pub(crate) fn strict_floor(&self, thr: f32) -> f32 {
        if !self.conf_inclusive || !thr.is_finite() {
            return thr;
        }
        if thr <= f32::MIN_POSITIVE { -f32::MIN_POSITIVE } else { thr.next_down() }
    }

    /// Effective confidence threshold for `class`: the entry in [`NmsConfig::per_class_conf`] when present,
    /// otherwise [`NmsConfig::conf`].
    #[inline]
    pub fn conf_for_class(&self, class: u32) -> f32 {
        self.per_class_conf.as_ref().and_then(|m| m.get(&class).copied()).unwrap_or(self.conf)
    }

    /// Minimum effective confidence threshold across [`NmsConfig::conf`] and all entries of
    /// [`NmsConfig::per_class_conf`] (O(1) when `per_class_conf` is `None`). `f32::min` folds symmetrically so
    /// finite entries are preserved even if another entry is `NaN`.
    #[inline]
    pub fn min_conf(&self) -> f32 {
        let Some(map) = &self.per_class_conf else {
            return self.conf;
        };
        map.values().copied().fold(self.conf, f32::min)
    }

    #[inline]
    fn raw_candidate_conf_for_value(&self, c: f32) -> f32 {
        let cap_for_rule = (self.calibrated_post_filter_enabled()
            || (self.conf_inclusive && !self.nms_kind.is_soft() && self.score != ScoreRule::None))
            && c > CALIBRATED_PRE_CONF;
        if cap_for_rule { CALIBRATED_PRE_CONF } else { c }
    }

    /// Pre-NMS candidate threshold for `class`: `0.10` when single-label calibrated scoring (or evaluator-derived
    /// [`NmsConfig::conf_inclusive`] with any [`ScoreRule`]) under hard NMS has `conf_for_class(class) > 0.10`,
    /// else `conf_for_class(class)`.
    #[inline]
    pub fn candidate_conf_for_class(&self, class: u32) -> f32 {
        self.raw_candidate_conf_for_value(self.conf_for_class(class))
    }

    /// Scalar pre-NMS strict extraction floor across all classes: `score > candidate_conf()` captures every
    /// candidate passing `candidate_conf_for_class(c)` for any class `c` (used by scalar class-map scanners and
    /// GPU candidate extraction kernels before per-class filtering).
    #[inline]
    pub fn candidate_conf(&self) -> f32 {
        let base = self.raw_candidate_conf_for_value(self.conf);
        let min_raw = match &self.per_class_conf {
            Some(map) => {
                map.values().copied().fold(base, |acc, v| acc.min(self.raw_candidate_conf_for_value(v)))
            }
            None => base,
        };
        self.strict_floor(min_raw)
    }

    /// Like [`NmsConfig::candidate_conf`], but returns `strict_floor(min_conf())` for [`HeadMode::O2o`]
    /// (which skips NMS and score rules).
    #[inline]
    pub fn candidate_conf_for(&self, mode: HeadMode) -> f32 {
        if mode == HeadMode::O2m { self.candidate_conf() } else { self.strict_floor(self.min_conf()) }
    }

    #[inline]
    fn calibrated_post_filter_enabled(&self) -> bool {
        !self.multi_label && !self.nms_kind.is_soft() && self.score.is_calibrated()
    }

    /// Whether final detections of `class` are filtered by `score_passes(score, conf_for_class(class))` after
    /// [`ScoreRule`] re-scoring:
    /// - Always `true` when [`NmsConfig::conf_inclusive`] is enabled with any non-`None` [`ScoreRule`] under hard
    ///   NMS (evaluator-derived F1 thresholds apply to final scores).
    /// - Otherwise `true` for single-label calibrated scoring under hard NMS when
    ///   `conf_for_class(class) > CALIBRATED_PRE_CONF`.
    #[inline]
    pub(crate) fn post_filter_score_for_class(&self, class: u32) -> bool {
        if self.nms_kind.is_soft() || self.score == ScoreRule::None {
            return false;
        }
        self.conf_inclusive
            || (self.calibrated_post_filter_enabled() && self.conf_for_class(class) > CALIBRATED_PRE_CONF)
    }

    /// Whether any class requires final score filtering after [`ScoreRule`] re-scoring.
    #[inline]
    pub(crate) fn post_filter_score(&self) -> bool {
        if self.nms_kind.is_soft() || self.score == ScoreRule::None {
            return false;
        }
        if self.conf_inclusive {
            return true;
        }
        if !self.calibrated_post_filter_enabled() {
            return false;
        }
        self.conf > CALIBRATED_PRE_CONF
            || self.per_class_conf.as_ref().is_some_and(|m| m.values().any(|&v| v > CALIBRATED_PRE_CONF))
    }
}
