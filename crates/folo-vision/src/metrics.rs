//! COCO-style detection metrics: mAP@0.5, mAP@0.5:0.95, precision / recall / F1, PR curves and a confusion matrix.
//!
//! Feed images one at a time into an [`Evaluator`] (or several evaluators in parallel, then [`Evaluator::merge`]),
//! then call [`Evaluator::compute`] for a serde-serialisable [`EvalReport`] whose top-level and `per_class` field names
//! match the `val` object of `docs/API.md`.
//!
//! The AP computation follows **pycocotools** (`COCOeval`, bbox, area range "all", no crowd annotations): per image
//! and class, detections in descending score order (capped at [`EvalConfig::max_dets_per_class`], COCO's
//! `maxDets = 100`) are greedily matched to the unmatched ground truth with the highest IoU `>= t`, for each `t` in
//! `0.50:0.05:0.95`; AP is the mean over 101 recall points `0:0.01:1` of the interpolated precision (the precision
//! envelope sampled with a left-sided search). pycocotools' area-range ignores are applied only with
//! [`EvalConfig::coco_compatible`] (they never fire for valid pixel boxes); `tests/coco_parity.rs` checks the result
//! against pycocotools itself. Precision / recall / F1 are taken at the single confidence threshold that maximises the
//! class-mean F1 at IoU 0.5 (as Ultralytics reports them). `docs/METRICS.md` spells out every step and how this
//! differs from Ultralytics' own AP.

use crate::boxes::Xyxy;
use crate::nms::Detection;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Number of IoU thresholds (`0.50, 0.55, ..., 0.95`).
pub const NUM_IOU_THRESHOLDS: usize = 10;
/// Number of recall sample points for AP (`0.00, 0.01, ..., 1.00`).
pub const NUM_RECALL_POINTS: usize = 101;
/// Resolution of the confidence grid used to pick the best-F1 threshold (`0.000, 0.001, ..., 1.000`).
const NUM_CONF_POINTS: usize = 1001;
/// Every `CONF_EXPORT_STEP`-th confidence point is exported in the curves (101 points, `0.00:0.01:1.00`).
const CONF_EXPORT_STEP: usize = 10;

/// IoU thresholds exactly as `numpy.linspace(0.5, 0.95, 10)` computes them (so matching at the boundaries agrees
/// with pycocotools bit for bit).
pub fn iou_thresholds() -> [f64; NUM_IOU_THRESHOLDS] {
    let step = (0.95f64 - 0.5) / 9.0;
    let mut t: [f64; NUM_IOU_THRESHOLDS] = std::array::from_fn(|i| i as f64 * step + 0.5);
    t[NUM_IOU_THRESHOLDS - 1] = 0.95;
    t
}

/// Recall sample points exactly as `numpy.linspace(0, 1, 101)` computes them.
pub fn recall_points() -> [f64; NUM_RECALL_POINTS] {
    let step = 1.0f64 / 100.0;
    let mut r: [f64; NUM_RECALL_POINTS] = std::array::from_fn(|i| i as f64 * step);
    r[NUM_RECALL_POINTS - 1] = 1.0;
    r
}

/// One ground-truth box. Serialises flat as `{ "x1", "y1", "x2", "y2", "class" }`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    /// The box, `xyxy`, in the same coordinates as the predictions.
    #[serde(flatten)]
    pub bbox: Xyxy,
    /// Class index.
    pub class: u32,
}

/// Evaluation settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EvalConfig {
    /// Per image and class, only the highest-scoring `n` detections count (COCO's `maxDets`, default 100). `None`
    /// keeps all of them.
    pub max_dets_per_class: Option<usize>,
    /// Confusion matrix: only predictions with `score > confusion_conf` are counted (Ultralytics: 0.25).
    pub confusion_conf: f32,
    /// Confusion matrix: a prediction matches a ground truth when IoU `> confusion_iou` (Ultralytics: 0.45).
    pub confusion_iou: f32,
    /// Apply pycocotools' area range "all", `[0, 1e5²]`, to box areas `(x2 - x1) * (y2 - y1)`: ground truths
    /// outside it are ignored (not counted, and a prediction matched to one is neither a TP nor a FP), and so are
    /// unmatched predictions outside it (an inverted box has negative area). Off by default because FOLO's
    /// coordinates can be in any unit; for boxes with area in `[0, 1e10]` (any pixel box on an image under 100,000
    /// px) both settings give identical results. On, AP equals pycocotools' for every input (`docs/METRICS.md`).
    pub coco_compatible: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_dets_per_class: Some(100),
            confusion_conf: 0.25,
            confusion_iou: 0.45,
            coco_compatible: false,
        }
    }
}

impl EvalConfig {
    /// The defaults with [`EvalConfig::coco_compatible`] on: AP exactly as pycocotools' `COCOeval` (bbox, area
    /// range "all", `maxDets = 100`, no crowd boxes) computes it.
    pub fn coco() -> Self {
        Self { coco_compatible: true, ..Self::default() }
    }
}

/// Upper bound of pycocotools' area range "all" (`1e5 ** 2`; the lower bound is 0).
const COCO_MAX_AREA: f64 = 1e10;

/// Whether a box's area, `w * h` in f64 as pycocotools computes it from `xywh` (negative for a box inverted in one
/// axis), lies outside pycocotools' area range "all".
#[inline]
fn outside_coco_area(b: &Xyxy) -> bool {
    let a = (b.x2 as f64 - b.x1 as f64) * (b.y2 as f64 - b.y1 as f64);
    !(0.0..=COCO_MAX_AREA).contains(&a)
}

/// A detection as stored for AP: its score and, per IoU threshold, whether it was a true positive (bit `t` of `tp`)
/// or is ignored (bit `t` of `ig`, only with [`EvalConfig::coco_compatible`]). Neither means a false positive.
#[derive(Clone, Copy, Debug)]
struct DetRecord {
    score: f32,
    tp: u16,
    ig: u16,
}

/// Accumulates per-image matches; see the module docs.
#[derive(Clone, Debug)]
pub struct Evaluator {
    num_classes: usize,
    cfg: EvalConfig,
    thresholds: [f64; NUM_IOU_THRESHOLDS],
    num_images: u64,
    num_preds: u64,
    /// Per class: scored detections in insertion order (image order, then descending score).
    dets: Vec<Vec<DetRecord>>,
    /// Per class: number of ground truths.
    gt_counts: Vec<u64>,
    /// `(num_classes + 1)²`, row-major `[pred][truth]`, index `num_classes` = background.
    confusion: Vec<u64>,
    // Scratch buffers reused across images.
    pred_order: Vec<u32>,
    gt_order: Vec<u32>,
    ious: Vec<f64>,
    gt_matched: Vec<bool>,
    tp_bits: Vec<u16>,
    ig_bits: Vec<u16>,
    pairs: Vec<(f32, u32, u32)>,
    pred_used: Vec<bool>,
}

impl Evaluator {
    /// An evaluator for `num_classes` classes with [`EvalConfig::default`].
    pub fn new(num_classes: usize) -> Self {
        Self::with_config(num_classes, EvalConfig::default())
    }

    /// An evaluator with explicit settings.
    pub fn with_config(num_classes: usize, cfg: EvalConfig) -> Self {
        Self {
            num_classes,
            cfg,
            thresholds: iou_thresholds(),
            num_images: 0,
            num_preds: 0,
            dets: vec![Vec::new(); num_classes],
            gt_counts: vec![0; num_classes],
            confusion: vec![0; (num_classes + 1) * (num_classes + 1)],
            pred_order: Vec::new(),
            gt_order: Vec::new(),
            ious: Vec::new(),
            gt_matched: Vec::new(),
            tp_bits: Vec::new(),
            ig_bits: Vec::new(),
            pairs: Vec::new(),
            pred_used: Vec::new(),
        }
    }

    /// Number of classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Number of images added so far.
    pub fn num_images(&self) -> u64 {
        self.num_images
    }

    /// Add one image's predictions and ground truths (same coordinate system, any order). Predictions with a NaN
    /// score are ignored.
    ///
    /// # Panics
    /// If a class index is `>= num_classes`.
    pub fn add_image(&mut self, preds: &[Detection], truths: &[Truth]) {
        let nc = self.num_classes;
        for p in preds {
            assert!((p.class as usize) < nc, "Evaluator: prediction class {} >= num_classes {nc}", p.class);
        }
        for t in truths {
            assert!((t.class as usize) < nc, "Evaluator: truth class {} >= num_classes {nc}", t.class);
        }
        self.num_images += 1;
        self.update_confusion(preds, truths);

        // One sort per image: predictions by (class, score desc, input order); truths by (class, input order), with
        // COCO-ignored truths last within their class (pycocotools' stable sort on `_ignore`).
        let coco = self.cfg.coco_compatible;
        let pred_order = &mut self.pred_order;
        pred_order.clear();
        pred_order.extend((0..preds.len() as u32).filter(|&i| !preds[i as usize].score.is_nan()));
        pred_order.sort_by(|&a, &b| {
            let (pa, pb) = (&preds[a as usize], &preds[b as usize]);
            pa.class.cmp(&pb.class).then(pb.score.total_cmp(&pa.score))
        });
        let gt_order = &mut self.gt_order;
        gt_order.clear();
        gt_order.extend(0..truths.len() as u32);
        if coco {
            gt_order
                .sort_by_key(|&i| (truths[i as usize].class, outside_coco_area(&truths[i as usize].bbox)));
        } else {
            gt_order.sort_by_key(|&i| truths[i as usize].class);
        }
        for t in truths {
            if !(coco && outside_coco_area(&t.bbox)) {
                self.gt_counts[t.class as usize] += 1;
            }
        }

        let (mut pi, mut gi) = (0, 0);
        while pi < self.pred_order.len() {
            let class = preds[self.pred_order[pi] as usize].class;
            let p_end = pi + self.pred_order[pi..].partition_point(|&i| preds[i as usize].class == class);
            while gi < self.gt_order.len() && truths[self.gt_order[gi] as usize].class < class {
                gi += 1;
            }
            let g_end = gi + self.gt_order[gi..].partition_point(|&i| truths[i as usize].class == class);
            // Regular (not ignored) truths come first: `gt_order[gi..g_reg]`.
            let g_reg = if coco {
                gi + self.gt_order[gi..g_end]
                    .partition_point(|&i| !outside_coco_area(&truths[i as usize].bbox))
            } else {
                g_end
            };
            let n_p = match self.cfg.max_dets_per_class {
                Some(m) => (p_end - pi).min(m),
                None => p_end - pi,
            };
            self.match_class(preds, truths, pi, pi + n_p, gi, g_reg, g_end, class as usize);
            self.num_preds += n_p as u64;
            pi = p_end;
            gi = g_end;
        }
    }

    /// COCO greedy matching for one (image, class): detections `pred_order[p0..p1]` (descending score) against
    /// truths `gt_order[g0..g1]`, of which `gt_order[g_reg..g1]` are COCO-ignored (none unless `coco_compatible`).
    #[allow(clippy::too_many_arguments)]
    fn match_class(
        &mut self,
        preds: &[Detection],
        truths: &[Truth],
        p0: usize,
        p1: usize,
        g0: usize,
        g_reg: usize,
        g1: usize,
        class: usize,
    ) {
        let (np, ng, n_reg) = (p1 - p0, g1 - g0, g_reg - g0);
        self.ious.clear();
        for &pi in &self.pred_order[p0..p1] {
            let pb = &preds[pi as usize].bbox;
            for &gi in &self.gt_order[g0..g1] {
                self.ious.push(iou_f64(pb, &truths[gi as usize].bbox));
            }
        }
        let mut tp = std::mem::take(&mut self.tp_bits);
        tp.clear();
        tp.resize(np, 0);
        let mut ig = std::mem::take(&mut self.ig_bits);
        ig.clear();
        ig.resize(np, 0);
        if ng > 0 {
            for (t, &thr) in self.thresholds.iter().enumerate() {
                self.gt_matched.clear();
                self.gt_matched.resize(ng, false);
                for (d, (tp, ig)) in tp.iter_mut().zip(ig.iter_mut()).enumerate() {
                    let mut best = thr.min(1.0 - 1e-10);
                    let mut m = usize::MAX;
                    for g in 0..ng {
                        if self.gt_matched[g] {
                            continue;
                        }
                        // pycocotools: once matched to a regular truth, stop at the ignored ones (sorted last).
                        if m < n_reg && g >= n_reg {
                            break;
                        }
                        let v = self.ious[d * ng + g];
                        if v < best {
                            continue;
                        }
                        best = v;
                        m = g;
                    }
                    if m != usize::MAX {
                        self.gt_matched[m] = true;
                        // A match to an ignored truth is neither a TP nor a FP.
                        if m < n_reg {
                            *tp |= 1 << t;
                        } else {
                            *ig |= 1 << t;
                        }
                    }
                }
            }
        }
        let coco = self.cfg.coco_compatible;
        let out = &mut self.dets[class];
        for (k, &pi) in self.pred_order[p0..p1].iter().enumerate() {
            let p = &preds[pi as usize];
            let mut ig_k = ig[k];
            if coco && outside_coco_area(&p.bbox) {
                // Unmatched detections outside the area range are ignored too.
                ig_k |= !(tp[k] | ig_k) & ((1 << NUM_IOU_THRESHOLDS) - 1);
            }
            out.push(DetRecord { score: p.score, tp: tp[k], ig: ig_k });
        }
        self.tp_bits = tp;
        self.ig_bits = ig;
    }

    /// Ultralytics-style confusion matrix update with greedy one-to-one class-agnostic matching.
    fn update_confusion(&mut self, preds: &[Detection], truths: &[Truth]) {
        let nc = self.num_classes;
        let stride = nc + 1;
        let conf = self.cfg.confusion_conf;
        let pairs = &mut self.pairs;
        pairs.clear();
        let kept = |p: &Detection| p.score > conf;
        for (pi, p) in preds.iter().enumerate() {
            if !kept(p) {
                continue;
            }
            for (gi, g) in truths.iter().enumerate() {
                let v = crate::boxes::box_iou(&p.bbox, &g.bbox);
                if v > self.cfg.confusion_iou {
                    pairs.push((v, gi as u32, pi as u32));
                }
            }
        }
        // Highest IoU first; ties by truth then prediction index.
        pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let gt_used = &mut self.gt_matched;
        gt_used.clear();
        gt_used.resize(truths.len(), false);
        let pred_used = &mut self.pred_used;
        pred_used.clear();
        pred_used.resize(preds.len(), false);
        for &(_, gi, pi) in pairs.iter() {
            let (gi, pi) = (gi as usize, pi as usize);
            if gt_used[gi] || pred_used[pi] {
                continue;
            }
            gt_used[gi] = true;
            pred_used[pi] = true;
            self.confusion[preds[pi].class as usize * stride + truths[gi].class as usize] += 1;
        }
        for (gi, g) in truths.iter().enumerate() {
            if !gt_used[gi] {
                self.confusion[nc * stride + g.class as usize] += 1;
            }
        }
        for (pi, p) in preds.iter().enumerate() {
            if kept(p) && !pred_used[pi] {
                self.confusion[p.class as usize * stride + nc] += 1;
            }
        }
    }

    /// Fold another evaluator's images into this one, as if they had been added after this evaluator's own.
    ///
    /// # Panics
    /// If the two evaluators differ in class count or configuration.
    pub fn merge(&mut self, other: Evaluator) {
        assert_eq!(self.num_classes, other.num_classes, "Evaluator::merge: class count mismatch");
        assert_eq!(self.cfg, other.cfg, "Evaluator::merge: config mismatch");
        self.num_images += other.num_images;
        self.num_preds += other.num_preds;
        for (a, b) in self.dets.iter_mut().zip(other.dets) {
            a.extend(b);
        }
        for (a, b) in self.gt_counts.iter_mut().zip(other.gt_counts) {
            *a += b;
        }
        for (a, b) in self.confusion.iter_mut().zip(other.confusion) {
            *a += b;
        }
    }

    /// Compute the report. Does not consume the accumulated state, so it can be called again after more images.
    pub fn compute(&self) -> EvalReport {
        let nc = self.num_classes;
        let recall_pts = recall_points();
        let per: Vec<Option<ClassEval>> = (0..nc)
            .into_par_iter()
            .map(|c| {
                let n_gt = self.gt_counts[c];
                (n_gt > 0).then(|| class_eval(&self.dets[c], n_gt, &recall_pts))
            })
            .collect();
        let evaluated: Vec<(usize, &ClassEval)> =
            per.iter().enumerate().filter_map(|(c, e)| e.as_ref().map(|e| (c, e))).collect();
        let n_eval = evaluated.len();
        let mean = |f: &dyn Fn(&ClassEval) -> f64| -> f32 {
            if n_eval == 0 {
                0.0
            } else {
                (evaluated.iter().map(|(_, e)| f(e)).sum::<f64>() / n_eval as f64) as f32
            }
        };

        // Best confidence: argmax of the class-mean F1 over the confidence grid; the highest threshold on ties.
        let mut best_i = 0;
        if n_eval > 0 {
            let mut best_f1 = f64::NEG_INFINITY;
            for i in 0..NUM_CONF_POINTS {
                let f = evaluated.iter().map(|(_, e)| e.f1[i]).sum::<f64>() / n_eval as f64;
                if f >= best_f1 {
                    best_f1 = f;
                    best_i = i;
                }
            }
        }

        let per_class: Vec<ClassReport> = evaluated
            .iter()
            .map(|&(c, e)| ClassReport {
                class: c as u32,
                map50: e.ap[0] as f32,
                map50_95: e.ap_mean() as f32,
                precision: e.p[best_i] as f32,
                recall: e.r[best_i] as f32,
                f1: e.f1[best_i] as f32,
                ap: e.ap.iter().map(|&v| v as f32).collect(),
                truths: self.gt_counts[c],
                preds: self.dets[c].len() as u64,
            })
            .collect();

        let export =
            |v: &[f64]| -> Vec<f32> { v.iter().step_by(CONF_EXPORT_STEP).map(|&x| x as f32).collect() };
        let mean_curve = |f: &dyn Fn(&ClassEval) -> &[f64], len: usize| -> Vec<f64> {
            let mut out = vec![0.0; len];
            for (_, e) in &evaluated {
                for (o, v) in out.iter_mut().zip(f(e)) {
                    *o += v;
                }
            }
            if n_eval > 0 {
                out.iter_mut().for_each(|o| *o /= n_eval as f64);
            }
            out
        };
        let to_f32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let curves = Curves {
            recall: to_f32(&recall_pts),
            confidence: (0..NUM_CONF_POINTS).step_by(CONF_EXPORT_STEP).map(conf_grid).collect(),
            all: CurveSet {
                class: None,
                pr: to_f32(&mean_curve(&|e| &e.pr, NUM_RECALL_POINTS)),
                precision: export(&mean_curve(&|e| &e.p, NUM_CONF_POINTS)),
                recall: export(&mean_curve(&|e| &e.r, NUM_CONF_POINTS)),
                f1: export(&mean_curve(&|e| &e.f1, NUM_CONF_POINTS)),
            },
            per_class: evaluated
                .iter()
                .map(|&(c, e)| CurveSet {
                    class: Some(c as u32),
                    pr: to_f32(&e.pr),
                    precision: export(&e.p),
                    recall: export(&e.r),
                    f1: export(&e.f1),
                })
                .collect(),
        };

        let stride = nc + 1;
        EvalReport {
            map50: mean(&|e| e.ap[0]),
            map50_95: mean(&|e| e.ap_mean()),
            precision: mean(&|e| e.p[best_i]),
            recall: mean(&|e| e.r[best_i]),
            f1: mean(&|e| e.f1[best_i]),
            best_conf: conf_grid(best_i),
            map_per_iou: (0..NUM_IOU_THRESHOLDS).map(|t| mean(&|e| e.ap[t])).collect(),
            iou_thresholds: self.thresholds.iter().map(|&t| t as f32).collect(),
            num_images: self.num_images,
            num_truths: self.gt_counts.iter().sum(),
            num_preds: self.num_preds,
            per_class,
            curves,
            confusion: ConfusionMatrix {
                num_classes: nc,
                conf: self.cfg.confusion_conf,
                iou: self.cfg.confusion_iou,
                matrix: self.confusion.chunks(stride).map(<[u64]>::to_vec).collect(),
            },
        }
    }
}

/// Confidence grid point `i` (`i / 1000`), in f32 like the scores it is compared with.
#[inline]
fn conf_grid(i: usize) -> f32 {
    i as f32 / (NUM_CONF_POINTS - 1) as f32
}

/// IoU in f64, as pycocotools computes it (`bbIou`).
#[inline]
fn iou_f64(a: &Xyxy, b: &Xyxy) -> f64 {
    let (ax1, ay1, ax2, ay2) = (a.x1 as f64, a.y1 as f64, a.x2 as f64, a.y2 as f64);
    let (bx1, by1, bx2, by2) = (b.x1 as f64, b.y1 as f64, b.x2 as f64, b.y2 as f64);
    let w = ax2.min(bx2) - ax1.max(bx1);
    if w <= 0.0 {
        return 0.0;
    }
    let h = ay2.min(by2) - ay1.max(by1);
    if h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let union =
        (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0) + (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Everything computed for one class with at least one ground truth.
struct ClassEval {
    /// AP per IoU threshold.
    ap: [f64; NUM_IOU_THRESHOLDS],
    /// Interpolated precision at the 101 recall points, IoU 0.5.
    pr: Vec<f64>,
    /// Precision, recall and F1 at IoU 0.5 for each confidence grid point.
    p: Vec<f64>,
    r: Vec<f64>,
    f1: Vec<f64>,
}

impl ClassEval {
    fn ap_mean(&self) -> f64 {
        self.ap.iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64
    }
}

fn class_eval(recs: &[DetRecord], n_gt: u64, recall_pts: &[f64; NUM_RECALL_POINTS]) -> ClassEval {
    // Stable sort by descending score: ties keep image order (pycocotools' mergesort over concatenated images).
    let mut sorted = recs.to_vec();
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));
    let nd = sorted.len();
    let n_gt_f = n_gt as f64;

    let mut ap = [0.0; NUM_IOU_THRESHOLDS];
    let mut pr50 = vec![0.0; NUM_RECALL_POINTS];
    let mut rc = vec![0.0f64; nd];
    let mut pr = vec![0.0f64; nd];
    for (t, ap_t) in ap.iter_mut().enumerate() {
        let (mut tp, mut fp) = (0u64, 0u64);
        for (i, d) in sorted.iter().enumerate() {
            if d.tp >> t & 1 == 1 {
                tp += 1;
            } else if d.ig >> t & 1 == 0 {
                fp += 1;
            }
            rc[i] = tp as f64 / n_gt_f;
            pr[i] = tp as f64 / (tp as f64 + fp as f64 + f64::EPSILON);
        }
        // Precision envelope: make precision monotonically non-increasing in recall.
        for i in (1..nd).rev() {
            if pr[i] > pr[i - 1] {
                pr[i - 1] = pr[i];
            }
        }
        // Sample at the recall points with a left-sided search; points beyond the last recall get 0.
        let mut q = [0.0f64; NUM_RECALL_POINTS];
        let mut j = 0;
        for (qi, &r) in q.iter_mut().zip(recall_pts) {
            while j < nd && rc[j] < r {
                j += 1;
            }
            if j == nd {
                break;
            }
            *qi = pr[j];
        }
        *ap_t = q.iter().sum::<f64>() / NUM_RECALL_POINTS as f64;
        if t == 0 {
            pr50.copy_from_slice(&q);
        }
    }

    // Confidence sweep at IoU 0.5: P/R/F1 of the detections with score >= c, for each grid point c.
    let mut p = vec![0.0; NUM_CONF_POINTS];
    let mut r = vec![0.0; NUM_CONF_POINTS];
    let mut f1 = vec![0.0; NUM_CONF_POINTS];
    let (mut j, mut tp, mut n) = (0usize, 0u64, 0u64);
    for i in (0..NUM_CONF_POINTS).rev() {
        let c = conf_grid(i);
        while j < nd && sorted[j].score >= c {
            tp += (sorted[j].tp & 1) as u64;
            n += u64::from(sorted[j].ig & 1 == 0);
            j += 1;
        }
        let prec = if n > 0 { tp as f64 / n as f64 } else { 0.0 };
        let rec = tp as f64 / n_gt_f;
        p[i] = prec;
        r[i] = rec;
        f1[i] = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
    }
    ClassEval { ap, pr: pr50, p, r, f1 }
}

/// Metrics for one class that has at least one ground truth.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassReport {
    /// Class index.
    pub class: u32,
    /// AP at IoU 0.5.
    pub map50: f32,
    /// AP averaged over IoU 0.50:0.05:0.95.
    pub map50_95: f32,
    /// Precision at the report's best-F1 confidence.
    pub precision: f32,
    /// Recall at the report's best-F1 confidence.
    pub recall: f32,
    /// F1 at the report's best-F1 confidence.
    pub f1: f32,
    /// AP at each IoU threshold (`EvalReport::iou_thresholds`).
    pub ap: Vec<f32>,
    /// Number of ground truths (with [`EvalConfig::coco_compatible`], not counting the ignored ones).
    pub truths: u64,
    /// Number of predictions that entered AP (after the per-image cap).
    pub preds: u64,
}

/// A family of curves for one class, or the class mean (`class: None`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurveSet {
    /// Class index, or `None` for the mean over classes.
    pub class: Option<u32>,
    /// Interpolated precision at each point of [`Curves::recall`] (IoU 0.5); the area under it is AP50.
    pub pr: Vec<f32>,
    /// Precision at each point of [`Curves::confidence`] (IoU 0.5).
    pub precision: Vec<f32>,
    /// Recall at each point of [`Curves::confidence`] (IoU 0.5).
    pub recall: Vec<f32>,
    /// F1 at each point of [`Curves::confidence`] (IoU 0.5).
    pub f1: Vec<f32>,
}

/// Curves for charts. All arrays in a [`CurveSet`] align with the axes here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Curves {
    /// The 101 recall points `0.00:0.01:1.00` (x axis of the PR curve).
    pub recall: Vec<f32>,
    /// The 101 confidence thresholds `0.00:0.01:1.00` (x axis of the P / R / F1 curves).
    pub confidence: Vec<f32>,
    /// The mean over classes.
    pub all: CurveSet,
    /// One entry per class with ground truth, in class order.
    pub per_class: Vec<CurveSet>,
}

/// Confusion matrix with a background row and column.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// Number of object classes; the matrix is `(num_classes + 1)²`.
    pub num_classes: usize,
    /// Confidence threshold used (`score > conf`).
    pub conf: f32,
    /// IoU threshold used (`IoU > iou`).
    pub iou: f32,
    /// `matrix[pred][truth]`. Row `num_classes` counts missed truths (background predicted); column `num_classes`
    /// counts unmatched predictions (background truth). `matrix[num_classes][num_classes]` is always 0.
    pub matrix: Vec<Vec<u64>>,
}

/// The full evaluation result. Field names of the first five fields and of `per_class` match `val` in `docs/API.md`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// Mean over classes (with ground truth) of AP at IoU 0.5.
    pub map50: f32,
    /// Mean over classes and IoU thresholds 0.50:0.05:0.95 of AP.
    pub map50_95: f32,
    /// Mean over classes of precision at `best_conf`.
    pub precision: f32,
    /// Mean over classes of recall at `best_conf`.
    pub recall: f32,
    /// Mean over classes of F1 at `best_conf`.
    pub f1: f32,
    /// The confidence threshold (on a 0.001 grid) that maximises the class-mean F1 at IoU 0.5 (the highest one on
    /// ties); detections with `score >= best_conf` count.
    pub best_conf: f32,
    /// mAP at each IoU threshold.
    pub map_per_iou: Vec<f32>,
    /// The IoU thresholds, `0.50:0.05:0.95`.
    pub iou_thresholds: Vec<f32>,
    /// Images evaluated.
    pub num_images: u64,
    /// Ground-truth boxes (with [`EvalConfig::coco_compatible`], not counting the ignored ones).
    pub num_truths: u64,
    /// Predictions that entered AP (after the per-image cap).
    pub num_preds: u64,
    /// Per-class metrics, only for classes with at least one ground truth, in class order.
    pub per_class: Vec<ClassReport>,
    /// PR and confidence curves.
    pub curves: Curves,
    /// Confusion matrix.
    pub confusion: ConfusionMatrix,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_match_numpy() {
        let t = iou_thresholds();
        // numpy: np.linspace(.5, .95, 10)[8] == 0.8999999999999999 (not 0.9).
        assert_eq!(t[0], 0.5);
        assert_eq!(t[1], 0.55);
        assert_eq!(t[8], 0.8999999999999999);
        assert_eq!(t[9], 0.95);
        let r = recall_points();
        // numpy: np.linspace(0, 1, 101)[35] == 0.35000000000000003 (not 0.35).
        assert_eq!(r[7], 0.07);
        assert_eq!(r[35], 0.35000000000000003);
        assert_eq!(r[100], 1.0);
    }
}
