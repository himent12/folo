//! Mock model output: fake detections and evaluations (PR curves, confusion matrix) consistent with a run's
//! metrics.

use folo_api::{BBox, ClassMetrics, Model, ModelEval, PrCurve, ScoredBox, Split};

use crate::store::ImageQuery;
use crate::util::{Rng, fnv1a, noise};

use super::datasets::DatasetEntry;
use super::runs::{self, SimRun};

/// Fake detections: the truth, jittered and occasionally missed or mislabelled, plus a few false positives. Better
/// `quality` (0..1) gives tighter boxes, higher scores and fewer mistakes.
pub(super) fn fake_detections(
    truth: &[BBox],
    w: u32,
    h: u32,
    classes: &[String],
    quality: f32,
    seed: u64,
) -> Vec<ScoredBox> {
    if classes.is_empty() {
        return Vec::new();
    }
    let mut r = Rng::new(seed);
    let (wf, hf) = (w as f32, h as f32);
    let name = |c: u32| classes.get(c as usize).cloned().unwrap_or_else(|| format!("class{c}"));
    let mut out = Vec::new();
    for t in truth {
        if !r.chance(0.35 + 0.63 * quality) {
            continue;
        }
        let (bw, bh) = (t.x2 - t.x1, t.y2 - t.y1);
        let j = (1.0 - quality) * 0.22 + 0.015;
        let mut class = t.class;
        if classes.len() > 1 && r.chance((1.0 - quality) * 0.25) {
            class = r.below(classes.len() as u64) as u32;
        }
        let score = (0.35 + 0.6 * quality + 0.25 * (r.f32() - 0.5)).clamp(0.05, 0.99);
        out.push(ScoredBox {
            class,
            name: name(class),
            score,
            x1: (t.x1 + bw * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, wf),
            y1: (t.y1 + bh * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, hf),
            x2: (t.x2 + bw * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, wf),
            y2: (t.y2 + bh * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, hf),
        });
    }
    let fp = (r.f32() * 3.5 * (1.0 - quality)).floor() as u32;
    for _ in 0..fp {
        let (bw, bh) = (r.range(0.04, 0.25) * wf, r.range(0.04, 0.25) * hf);
        let (x, y) = (r.range(0.0, wf - bw), r.range(0.0, hf - bh));
        let class = r.below(classes.len().max(1) as u64) as u32;
        out.push(ScoredBox {
            class,
            name: name(class),
            score: r.range(0.08, 0.45),
            x1: x,
            y1: y,
            x2: x + bw,
            y2: y + bh,
        });
    }
    for b in &mut out {
        if b.x2 < b.x1 {
            std::mem::swap(&mut b.x1, &mut b.x2);
        }
        if b.y2 < b.y1 {
            std::mem::swap(&mut b.y1, &mut b.y2);
        }
        b.score = (b.score * 1000.0).round() / 1000.0;
        for v in [&mut b.x1, &mut b.y1, &mut b.x2, &mut b.y2] {
            *v = (*v * 10.0).round() / 10.0;
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

pub(super) struct StaticModel {
    pub(super) model: Model,
    pub(super) basis: EvalBasis,
}

/// What a model's mock evaluation is derived from.
#[derive(Clone)]
pub(super) struct EvalBasis {
    per_class: Vec<ClassMetrics>,
    /// The confidence the per-class precision and recall are taken at.
    best_f1_conf: Option<f32>,
    /// Validation instances per class.
    counts: Vec<u64>,
    images: u32,
    seed: u64,
    /// The evaluated dataset's id and name, the split evaluated and why, when it is not `val`.
    dataset: (String, String),
    split: folo_api::Split,
    split_fallback: Option<folo_api::SplitFallback>,
}

impl EvalBasis {
    /// From a run's best epoch and its dataset's validation split (train when there is none).
    pub(super) fn new(run: &SimRun, ds: &DatasetEntry) -> EvalBasis {
        let best = run.curve.best(run.epochs_done(&run.snap())).map_or(1, |b| b.epoch);
        let split = if ds.meta().splits.val > 0 { Split::Val } else { Split::Train };
        let page =
            ds.list_images(&ImageQuery { split: Some(split), offset: 0, limit: u64::MAX, class: None });
        let mut counts = vec![0u64; run.classes.len()];
        for b in page.images.iter().flat_map(|im| &im.boxes) {
            if let Some(c) = counts.get_mut(b.class as usize) {
                *c += 1;
            }
        }
        let val = run.curve.val(best);
        let meta = ds.meta();
        let used = if split == Split::Val { folo_api::Split::Val } else { folo_api::Split::Train };
        EvalBasis {
            best_f1_conf: val.best_f1_conf,
            per_class: val.per_class,
            counts,
            images: page.total as u32,
            seed: fnv1a(run.id.as_bytes()),
            split: used,
            split_fallback: (used != folo_api::Split::Val).then(|| {
                folo_api::SplitFallback::empty_split(folo_api::Split::Val, used, &meta.name, page.total)
            }),
            dataset: (meta.id.clone(), meta.name.clone()),
        }
    }

    /// PR curves, confusion matrix and per-class metrics, consistent with each other.
    pub(super) fn eval(&self) -> ModelEval {
        let n = self.per_class.len();
        let recall: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let mut per_class = self.per_class.clone();
        let pr = per_class
            .iter_mut()
            .map(|m| {
                // A wiggled `1 - r^k` envelope; its mean grows with k, so bisect k until the curve's AP is the class
                // mAP50 the run reported for this epoch (the model page and the run page then agree).
                let curve = |k: f32| {
                    let mut p: Vec<f32> = recall
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            let wiggle = 1.0 + 0.04 * noise(&[self.seed, m.class as u64, i as u64]);
                            ((1.0 - r.powf(k)) * wiggle).clamp(0.0, 1.0)
                        })
                        .collect();
                    for i in (0..p.len() - 1).rev() {
                        p[i] = p[i].max(p[i + 1]);
                    }
                    p
                };
                let mean = |p: &[f32]| p.iter().sum::<f32>() / p.len() as f32;
                let (mut lo, mut hi) = (1e-3f32, 1e3f32);
                for _ in 0..60 {
                    let mid = (lo * hi).sqrt();
                    if mean(&curve(mid)) < m.map50 { lo = mid } else { hi = mid }
                }
                let precision: Vec<f32> = curve((lo * hi).sqrt()).into_iter().map(runs::r4).collect();
                let ap50 = runs::r4(precision.iter().sum::<f32>() / precision.len() as f32);
                m.map50 = ap50;
                PrCurve { class: m.class, recall: recall.clone(), precision, ap50 }
            })
            .collect();
        // Rows: truth (+ background), columns: prediction (+ background).
        let mut confusion = vec![vec![0u32; n + 1]; n + 1];
        for (c, m) in self.per_class.iter().enumerate() {
            let total = self.counts.get(c).copied().unwrap_or(0) as f32;
            let tp = (total * m.recall).round();
            let wrong = ((total - tp) * 0.3).round();
            confusion[c][c] = tp as u32;
            if n > 1 && wrong > 0.0 {
                // Spread the confusions over the other classes with deterministic weights.
                let others: Vec<usize> = (0..n).filter(|&j| j != c).collect();
                let w: Vec<f32> =
                    others.iter().map(|&j| 1.2 + noise(&[self.seed, c as u64, j as u64, 7])).collect();
                let sum: f32 = w.iter().sum();
                let mut left = wrong as u32;
                for (idx, &j) in others.iter().enumerate() {
                    let share = if idx + 1 == others.len() {
                        left
                    } else {
                        ((wrong * w[idx] / sum).round() as u32).min(left)
                    };
                    confusion[c][j] += share;
                    left -= share;
                }
            }
            let assigned: u32 = confusion[c][..n].iter().sum();
            confusion[c][n] = (total as u32).saturating_sub(assigned);
        }
        for (j, m) in self.per_class.iter().enumerate() {
            // precision = tp / column total, so the column holds tp / p predictions: tp, the confusions from other
            // classes, and false positives on background. Confusions beyond that budget move back to "missed".
            let tp = confusion[j][j];
            let p = m.precision.clamp(0.05, 1.0);
            let budget = ((tp as f32 / p).round() as u32).saturating_sub(tp);
            let from_others: u32 = (0..n).filter(|&c| c != j).map(|c| confusion[c][j]).sum();
            if from_others > budget {
                let mut excess = from_others - budget;
                for c in (0..n).filter(|&c| c != j) {
                    let take = confusion[c][j].min(excess);
                    confusion[c][j] -= take;
                    confusion[c][n] += take;
                    excess -= take;
                }
            }
            let from_others: u32 = (0..n).filter(|&c| c != j).map(|c| confusion[c][j]).sum();
            confusion[n][j] = budget - from_others;
        }
        // Report the precision / recall the matrix actually implies, so every view of the eval agrees.
        for (c, m) in per_class.iter_mut().enumerate() {
            let row: u32 = confusion[c].iter().sum();
            let col: u32 = (0..=n).map(|r| confusion[r][c]).sum();
            if row > 0 {
                m.recall = runs::r4(confusion[c][c] as f32 / row as f32);
            }
            if col > 0 {
                m.precision = runs::r4(confusion[c][c] as f32 / col as f32);
            }
        }
        // The real evaluator's confusion thresholds (docs/METRICS.md §7).
        let thresholds = folo_vision::EvalConfig::default();
        ModelEval {
            pr,
            confusion,
            per_class,
            images: self.images,
            best_f1_conf: self.best_f1_conf,
            dataset_id: Some(self.dataset.0.clone()),
            dataset_name: Some(self.dataset.1.clone()),
            split: Some(self.split),
            split_fallback: self.split_fallback.clone(),
            confusion_conf: Some(thresholds.confusion_conf),
            confusion_iou: Some(thresholds.confusion_iou),
        }
    }
}

#[cfg(test)]
mod eval_tests {
    use super::*;

    fn basis(per_class: &[(f32, f32)], counts: &[u64]) -> EvalBasis {
        let per_class = per_class
            .iter()
            .enumerate()
            .map(|(c, &(precision, recall))| ClassMetrics {
                class: c as u32,
                map50: 0.7,
                map50_95: 0.5,
                precision,
                recall,
            })
            .collect();
        EvalBasis {
            per_class,
            best_f1_conf: Some(0.3),
            counts: counts.to_vec(),
            images: 10,
            seed: 1,
            dataset: ("ds_t".into(), "t".into()),
            split: folo_api::Split::Val,
            split_fallback: None,
        }
    }

    /// Row sums equal instance counts and `per_class` precision / recall match the matrix.
    fn check(b: &EvalBasis) {
        let ev = b.eval();
        let n = b.per_class.len();
        for c in 0..n {
            assert_eq!(ev.confusion[c].iter().map(|&v| v as u64).sum::<u64>(), b.counts[c], "row {c}");
            let row: u32 = ev.confusion[c].iter().sum();
            let col: u32 = (0..=n).map(|r| ev.confusion[r][c]).sum();
            let tp = ev.confusion[c][c] as f32;
            if row > 0 {
                assert!((ev.per_class[c].recall - tp / row as f32).abs() < 1e-3);
            }
            if col > 0 {
                assert!((ev.per_class[c].precision - tp / col as f32).abs() < 1e-3);
            }
        }
        assert_eq!(ev.confusion[n][n], 0);
    }

    #[test]
    fn confusion_is_consistent() {
        // Balanced, imbalanced (a rare class that receives many confusions), zero-count and single-class cases.
        check(&basis(&[(0.9, 0.85), (0.8, 0.7), (0.7, 0.6)], &[400, 380, 390]));
        check(&basis(&[(0.6, 0.4), (0.95, 0.9), (0.5, 0.3)], &[5000, 3, 1200]));
        check(&basis(&[(0.9, 0.8), (0.9, 0.8)], &[100, 0]));
        check(&basis(&[(0.05, 0.02)], &[50]));
        check(&basis(&[(1.0, 1.0), (0.01, 0.01)], &[1, 1]));
    }
}
