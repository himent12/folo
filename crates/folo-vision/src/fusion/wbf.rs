//! Weighted Boxes Fusion (WBF; Solovyev et al. 2021) for ensembling detections across multiple models or
//! test-time augmentation (TTA) views.
//!
//! Unlike greedy NMS (which discards overlapping predictions), WBF clusters boxes of the same class whose IoU
//! with a running confidence-weighted cluster centroid is `>= iou_thr` and fuses each cluster's coordinates by
//! confidence-weighted averaging in `f64`:
//!
//! ```text
//! s_i = w_{m(i)} * c_i
//! B_k = (Σ_{i in C_k} s_i * b_i) / (Σ_{i in C_k} s_i)
//! ```
//!
//! Cluster confidence is then computed by [`WbfConfType`] (FOLO conventions):
//! - [`WbfConfType::Avg`] (default):
//!   ```text
//!   C_k = (Σ_{i in C_k} w_{m(i)} * c_i) / max(Σ_{m=1}^M w_m, Σ_{i in C_k} w_{m(i)})
//!   ```
//!   For equal weights `w_m = 1`, this matches Solovyev et al. Eq. 3 `(1 / T) * (Σ c_i) * min(T, M) / M`.
//!   Generalising the denominator to `max(Σ_all w_m, Σ_cluster w_m)` keeps unequal weights strictly
//!   scale-invariant and guarantees `C_k <= max_{i in C_k} c_i` even when one model emits duplicate boxes into
//!   a cluster (note: duplicate boxes from one model increase `Σ_cluster w_m`, which can offset the missing-model
//!   penalty from another model while staying capped at the cluster's weighted mean).
//! - [`WbfConfType::Max`]: maximum unweighted raw confidence `max_{i in C_k} c_i` in the cluster (FOLO
//!   convention; unweighted so changing model weight scale never pushes confidence outside `[0, 1]`).
//! - [`WbfConfType::BoxAvg`]: `(Σ_{i in C_k} w_{m(i)} * c_i) / (Σ_{i in C_k} w_{m(i)})` (no absent-model penalty).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::nms::{Detection, exact_iou_f64};
use crate::{Error, Result};

/// How [`weighted_boxes_fusion`] combines member scores into the fused cluster's confidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WbfConfType {
    /// Model-count-aware weighted average (Solovyev et al. 2021 Eq. 3, default):
    /// `Σ(w_m * c_i) / max(Σ_all w_m, Σ_cluster w_m)`.
    #[default]
    Avg,
    /// Maximum unweighted confidence among boxes in the cluster.
    Max,
    /// Weighted average over boxes present in the cluster only (no penalty for absent models).
    BoxAvg,
}

/// Configuration for [`weighted_boxes_fusion`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WbfConfig {
    /// Minimum IoU with a cluster's fused box to join that cluster (default `0.55`).
    pub iou_thr: f32,
    /// Discard input boxes whose score is `<= skip_box_thr` before clustering (default `0.0`).
    pub skip_box_thr: f32,
    /// How fused cluster confidence is computed (default [`WbfConfType::Avg`]).
    pub conf_type: WbfConfType,
}

impl Default for WbfConfig {
    fn default() -> Self {
        Self { iou_thr: 0.55, skip_box_thr: 0.0, conf_type: WbfConfType::Avg }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    bbox: Xyxy,
    raw_score: f64,
    model_weight: f64,
    weighted_score: f64,
    model_idx: usize,
    box_idx: usize,
}

struct Cluster {
    sum_w_coords: [f64; 4],
    sum_weighted_score: f64,
    sum_model_weight: f64,
    max_raw_score: f64,
    fused_box: Xyxy,
}

impl Cluster {
    fn new(e: Entry) -> Self {
        let s = e.weighted_score;
        let coords = [s * e.bbox.x1 as f64, s * e.bbox.y1 as f64, s * e.bbox.x2 as f64, s * e.bbox.y2 as f64];
        Self {
            sum_w_coords: coords,
            sum_weighted_score: s,
            sum_model_weight: e.model_weight,
            max_raw_score: e.raw_score,
            fused_box: e.bbox,
        }
    }

    fn push(&mut self, e: Entry) {
        let s = e.weighted_score;
        self.sum_w_coords[0] += s * e.bbox.x1 as f64;
        self.sum_w_coords[1] += s * e.bbox.y1 as f64;
        self.sum_w_coords[2] += s * e.bbox.x2 as f64;
        self.sum_w_coords[3] += s * e.bbox.y2 as f64;
        self.sum_weighted_score += s;
        self.sum_model_weight += e.model_weight;
        if e.raw_score > self.max_raw_score {
            self.max_raw_score = e.raw_score;
        }
        let inv = 1.0 / self.sum_weighted_score;
        self.fused_box = Xyxy::new(
            (self.sum_w_coords[0] * inv) as f32,
            (self.sum_w_coords[1] * inv) as f32,
            (self.sum_w_coords[2] * inv) as f32,
            (self.sum_w_coords[3] * inv) as f32,
        );
    }

    fn confidence(&self, total_model_weight: f64, conf_type: WbfConfType) -> f32 {
        let c = match conf_type {
            WbfConfType::Avg => self.sum_weighted_score / total_model_weight.max(self.sum_model_weight),
            WbfConfType::Max => self.max_raw_score,
            WbfConfType::BoxAvg => self.sum_weighted_score / self.sum_model_weight,
        };
        c as f32
    }
}

/// Fuse detections from one or more models / TTA views via Weighted Boxes Fusion (Solovyev et al. 2021).
///
/// - `models`: one slice of [`Detection`]s per model or view.
/// - `weights`: optional positive per-model weights (same length as `models`; `None` gives equal weight `1.0`).
/// - `cfg`: IoU matching threshold, score skip threshold, and confidence rule.
///
/// Non-finite scores, non-finite coordinates, non-positive scores (`<= 0` or `<= skip_box_thr`), and inverted
/// or zero-area boxes are ignored. Output is deterministic and sorted by descending confidence.
///
/// # Errors
/// Returns [`Error::Config`] if `iou_thr` is not in `[0, 1]`, `skip_box_thr` is not finite or `< 0`, or
/// `weights` has the wrong length or contains a non-finite / `<= 0` value.
pub fn weighted_boxes_fusion(
    models: &[&[Detection]],
    weights: Option<&[f32]>,
    cfg: &WbfConfig,
) -> Result<Vec<Detection>> {
    if !(cfg.iou_thr.is_finite() && (0.0..=1.0).contains(&cfg.iou_thr)) {
        return Err(Error::Config(format!(
            "weighted_boxes_fusion: iou_thr must be finite and in 0..=1, got {}",
            cfg.iou_thr
        )));
    }
    if !(cfg.skip_box_thr.is_finite() && cfg.skip_box_thr >= 0.0) {
        return Err(Error::Config(format!(
            "weighted_boxes_fusion: skip_box_thr must be finite and >= 0, got {}",
            cfg.skip_box_thr
        )));
    }
    if models.is_empty() {
        if let Some(w) = weights
            && !w.is_empty()
        {
            return Err(Error::Config(format!(
                "weighted_boxes_fusion: weights length {} does not match models length 0",
                w.len()
            )));
        }
        return Ok(Vec::new());
    }
    let model_weights: Vec<f64> = match weights {
        Some(ws) => {
            if ws.len() != models.len() {
                return Err(Error::Config(format!(
                    "weighted_boxes_fusion: weights length {} does not match models length {}",
                    ws.len(),
                    models.len()
                )));
            }
            for (i, &w) in ws.iter().enumerate() {
                if !(w.is_finite() && w > 0.0) {
                    return Err(Error::Config(format!(
                        "weighted_boxes_fusion: weight[{i}] must be finite and > 0, got {w}"
                    )));
                }
            }
            ws.iter().map(|&w| w as f64).collect()
        }
        None => vec![1.0f64; models.len()],
    };
    let total_model_weight: f64 = model_weights.iter().sum();

    let mut by_class: BTreeMap<u32, Vec<Entry>> = BTreeMap::new();
    for (m_idx, (&dets, &mw)) in models.iter().zip(&model_weights).enumerate() {
        for (b_idx, det) in dets.iter().enumerate() {
            if !det.score.is_finite() || det.score <= 0.0 || det.score <= cfg.skip_box_thr {
                continue;
            }
            let b = det.bbox;
            if !(b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()) {
                continue;
            }
            if b.x2 <= b.x1 || b.y2 <= b.y1 {
                continue;
            }
            let raw = det.score as f64;
            by_class.entry(det.class).or_default().push(Entry {
                bbox: b,
                raw_score: raw,
                model_weight: mw,
                weighted_score: raw * mw,
                model_idx: m_idx,
                box_idx: b_idx,
            });
        }
    }

    let mut out = Vec::new();
    for (class, mut entries) in by_class {
        entries.sort_unstable_by(|a, b| {
            b.weighted_score
                .total_cmp(&a.weighted_score)
                .then(a.model_idx.cmp(&b.model_idx))
                .then(a.box_idx.cmp(&b.box_idx))
        });
        let mut clusters: Vec<Cluster> = Vec::new();
        let iou_thr64 = cfg.iou_thr as f64;
        for e in entries {
            let mut best_idx: Option<usize> = None;
            let mut best_iou = -1.0f64;
            for (ci, c) in clusters.iter().enumerate() {
                let ov = exact_iou_f64(&e.bbox, &c.fused_box);
                if ov >= iou_thr64 && ov > best_iou {
                    best_iou = ov;
                    best_idx = Some(ci);
                }
            }
            match best_idx {
                Some(ci) => clusters[ci].push(e),
                None => clusters.push(Cluster::new(e)),
            }
        }
        for c in clusters {
            out.push(Detection {
                bbox: c.fused_box,
                score: c.confidence(total_model_weight, cfg.conf_type),
                class,
            });
        }
    }

    out.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.class.cmp(&b.class))
            .then(a.bbox.x1.total_cmp(&b.bbox.x1))
            .then(a.bbox.y1.total_cmp(&b.bbox.y1))
            .then(a.bbox.x2.total_cmp(&b.bbox.x2))
            .then(a.bbox.y2.total_cmp(&b.bbox.y2))
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wbf_hand_computed_two_model_fusion_and_conf_types() {
        let m1 = [
            Detection { bbox: Xyxy::new(0.0, 0.0, 10.0, 10.0), score: 0.9, class: 0 },
            Detection { bbox: Xyxy::new(50.0, 50.0, 60.0, 60.0), score: 0.8, class: 1 },
        ];
        let m2 = [Detection { bbox: Xyxy::new(1.0, 1.0, 11.0, 11.0), score: 0.6, class: 0 }];
        let cfg = WbfConfig { iou_thr: 0.5, skip_box_thr: 0.0, conf_type: WbfConfType::Avg };
        let fused = weighted_boxes_fusion(&[&m1, &m2], None, &cfg).unwrap();
        assert_eq!(fused.len(), 2);

        // Class 0 fused from both models: weighted coords = (0.9 * [0,0,10,10] + 0.6 * [1,1,11,11]) / 1.5
        // = [0.4, 0.4, 10.4, 10.4], score = (0.9 + 0.6) / 2 = 0.75.
        assert_eq!(fused[0].class, 0);
        assert!((fused[0].score - 0.75).abs() < 1e-6);
        assert!((fused[0].bbox.x1 - 0.4).abs() < 1e-6);
        assert!((fused[0].bbox.x2 - 10.4).abs() < 1e-6);

        // Class 1 seen by only 1 of 2 models: score = 0.8 / 2 = 0.4.
        assert_eq!(fused[1].class, 1);
        assert!((fused[1].score - 0.4).abs() < 1e-6);

        // Max and BoxAvg conf_types:
        let max_f =
            weighted_boxes_fusion(&[&m1, &m2], None, &WbfConfig { conf_type: WbfConfType::Max, ..cfg })
                .unwrap();
        assert!((max_f[0].score - 0.9).abs() < 1e-6);
        assert!((max_f[1].score - 0.8).abs() < 1e-6);

        let bavg =
            weighted_boxes_fusion(&[&m1, &m2], None, &WbfConfig { conf_type: WbfConfType::BoxAvg, ..cfg })
                .unwrap();
        assert_eq!(bavg[0].class, 1);
        assert!((bavg[0].score - 0.8).abs() < 1e-6);
        assert_eq!(bavg[1].class, 0);
        assert!((bavg[1].score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn wbf_model_weights_scale_invariance_and_duplicate_cap() {
        let m1 = [
            Detection { bbox: Xyxy::new(0.0, 0.0, 10.0, 10.0), score: 0.9, class: 0 },
            // Duplicate detection from m1 in the same cluster:
            Detection { bbox: Xyxy::new(0.0, 0.0, 10.0, 10.0), score: 0.85, class: 0 },
        ];
        let m2 = [Detection { bbox: Xyxy::new(1.0, 0.0, 11.0, 10.0), score: 0.7, class: 0 }];
        let cfg = WbfConfig::default();
        let a = weighted_boxes_fusion(&[&m1, &m2], Some(&[2.0, 1.0]), &cfg).unwrap();
        let b = weighted_boxes_fusion(&[&m1, &m2], Some(&[20.0, 10.0]), &cfg).unwrap();
        assert_eq!(a.len(), 1);
        assert!((a[0].score - b[0].score).abs() < 1e-6, "weights must be scale-invariant");
        assert!(a[0].score <= 0.9 + 1e-6, "duplicate boxes must never push WBF score above max input");
        assert!((a[0].bbox.x1 - b[0].bbox.x1).abs() < 1e-6);
    }

    #[test]
    fn wbf_filters_degenerate_and_validates_config() {
        let dirty = [
            Detection { bbox: Xyxy::new(f32::NAN, 0.0, 10.0, 10.0), score: 0.9, class: 0 },
            Detection { bbox: Xyxy::new(0.0, 0.0, 10.0, 10.0), score: f32::NAN, class: 0 },
            Detection { bbox: Xyxy::new(5.0, 5.0, 5.0, 5.0), score: 0.8, class: 0 },
            Detection { bbox: Xyxy::new(10.0, 10.0, 2.0, 2.0), score: 0.8, class: 0 },
            Detection { bbox: Xyxy::new(0.0, 0.0, 10.0, 10.0), score: 0.2, class: 0 },
            Detection { bbox: Xyxy::new(0.0, 0.0, 10.0, 10.0), score: 0.6, class: 0 },
        ];
        let cfg = WbfConfig { iou_thr: 0.55, skip_box_thr: 0.25, conf_type: WbfConfType::Avg };
        let res = weighted_boxes_fusion(&[&dirty], None, &cfg).unwrap();
        assert_eq!(res.len(), 1);
        assert!((res[0].score - 0.6).abs() < 1e-6);

        assert!(weighted_boxes_fusion(&[], None, &cfg).unwrap().is_empty());
        assert!(weighted_boxes_fusion(&[&dirty], Some(&[1.0, 2.0]), &cfg).is_err());
        assert!(weighted_boxes_fusion(&[&dirty], Some(&[0.0]), &cfg).is_err());
        assert!(weighted_boxes_fusion(&[&dirty], Some(&[f32::NAN]), &cfg).is_err());
        assert!(weighted_boxes_fusion(&[&dirty], None, &WbfConfig { iou_thr: 1.5, ..cfg }).is_err());
        assert!(weighted_boxes_fusion(&[&dirty], None, &WbfConfig { skip_box_thr: -0.1, ..cfg }).is_err());

        // Exact f64 IoU at iou_thr = 1.0 must not merge distinct boxes `[0,0,1,1]` and `[1e-8,0,1,1]`.
        let near_m1 = [Detection { bbox: Xyxy::new(0.0, 0.0, 1.0, 1.0), score: 0.9, class: 0 }];
        let near_m2 = [Detection { bbox: Xyxy::new(1e-8, 0.0, 1.0, 1.0), score: 0.8, class: 0 }];
        let strict_cfg = WbfConfig { iou_thr: 1.0, skip_box_thr: 0.0, conf_type: WbfConfType::Max };
        assert_eq!(weighted_boxes_fusion(&[&near_m1, &near_m2], None, &strict_cfg).unwrap().len(), 2);
    }
}
