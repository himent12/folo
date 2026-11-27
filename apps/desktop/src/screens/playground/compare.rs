//! Playground Model A/B Visual Compare: state, default Model B selection, delta summaries,
//! and structured JSON export payloads.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use folo_ui::{Bounds, Pixels};

use crate::api::{Model, PredictResponse, ScoredBox, Timing};
use crate::store::Load;

/// Active inspection tab in the Playground side rail while Compare A/B is enabled.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CompareSideTab {
    #[default]
    Summary,
    ModelA,
    ModelB,
}

/// Per-class presence bucket in Compare A/B when both models have fresh predictions.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareClassStatus {
    Both,
    OnlyA,
    OnlyB,
}

impl CompareClassStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Both => "Both",
            Self::OnlyA => "Only A",
            Self::OnlyB => "Only B",
        }
    }
}

/// Filter over the Compare A/B per-class diff table (`All`, `Both`, `Only A`, `Only B`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CompareClassFilter {
    #[default]
    All,
    Both,
    OnlyA,
    OnlyB,
}

/// Completed prediction request parameters so stale cached results (old device, IoU, or floor)
/// are never reused on re-enable or exported under new thresholds.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayPredMeta {
    pub image_gen: u64,
    pub model_id: String,
    pub iou_bits: u32,
    pub target: Option<crate::local::Target>,
    pub floor: f32,
}

impl PlayPredMeta {
    /// True when this completed prediction is still valid for the current Playground parameters.
    pub fn matches_current(
        &self,
        image_gen: u64,
        model_id: &str,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        e2e: bool,
    ) -> bool {
        self.image_gen == image_gen
            && self.model_id == model_id
            && self.target == target
            && (e2e || self.iou_bits == iou.to_bits())
            && conf >= self.floor - 1e-6
    }
}

/// All Model B and comparison state held by `FoloApp` for Playground A/B compare.
pub struct PlayCompareState {
    pub enabled: bool,
    /// Completed parameters of Model A's current `play_pred` (set only on `Ok`, cleared on `Err`/reset).
    pub pred_a_meta: Option<PlayPredMeta>,
    /// Model B id; reconciled against the loaded model list just like `play_model`.
    pub model_b: Option<String>,
    /// Run that produced `model_b`, so deleting that run drops `model_b` immediately.
    pub model_b_run: Option<String>,
    /// Name of a deleted Model B, reported once when reconciled while Playground is active.
    pub model_b_gone: Option<String>,
    pub pred_b: Load<PredictResponse>,
    pub pred_b_pending: bool,
    pub pred_b_dim: bool,
    pub pred_b_for: Option<(u64, String)>,
    /// Completed parameters of Model B's current `pred_b` (set only on `Ok`, cleared on `Err`/reset).
    pub pred_b_meta: Option<PlayPredMeta>,
    pub pred_b_seq: u64,
    pub fetched_floor_b: f32,
    /// Dedicated local-inference ticket line for Model B so A and B never supersede each other.
    pub line_b: Arc<AtomicU64>,
    pub side_tab: CompareSideTab,
    /// Active class-diff filter (`All`, `Both`, `Only A`, `Only B`) in Compare A/B Delta tab.
    pub class_filter: CompareClassFilter,
    pub hover_b: Option<usize>,
    pub hover_b_from_stage: bool,
    pub view_b: Rc<Cell<Option<Bounds<Pixels>>>>,
}

impl Default for PlayCompareState {
    fn default() -> Self {
        Self {
            enabled: false,
            pred_a_meta: None,
            model_b: None,
            model_b_run: None,
            model_b_gone: None,
            pred_b: Load::Idle,
            pred_b_pending: false,
            pred_b_dim: false,
            pred_b_for: None,
            pred_b_meta: None,
            pred_b_seq: 0,
            fetched_floor_b: 0.05,
            line_b: Arc::new(AtomicU64::new(0)),
            side_tab: CompareSideTab::Summary,
            class_filter: CompareClassFilter::All,
            hover_b: None,
            hover_b_from_stage: false,
            view_b: Rc::new(Cell::new(None)),
        }
    }
}

impl PlayCompareState {
    /// Resets transient prediction and hover state when switching backend sources.
    pub fn reset_for_backend(&mut self) {
        self.enabled = false;
        self.pred_a_meta = None;
        self.model_b = None;
        self.model_b_run = None;
        self.model_b_gone = None;
        self.pred_b = Load::Idle;
        self.pred_b_pending = false;
        self.pred_b_dim = false;
        self.pred_b_for = None;
        self.pred_b_meta = None;
        self.pred_b_seq += 1;
        let _ = crate::local::Ticket::take(&self.line_b);
        self.hover_b = None;
        self.hover_b_from_stage = false;
        self.side_tab = CompareSideTab::Summary;
        self.class_filter = CompareClassFilter::All;
    }

    pub fn clear_hover(&mut self) {
        self.hover_b = None;
        self.hover_b_from_stage = false;
    }
}

/// Per-class detection comparison row matched by class name.
#[derive(Clone, Debug, PartialEq)]
pub struct CompareClassRow {
    pub name: String,
    /// Class index for palette color swatch (prefers Model A's class index, then Model B's).
    pub color_class: u32,
    pub count_a: Option<usize>,
    pub count_b: Option<usize>,
    pub delta: Option<isize>,
    /// Class-diff bucket (`Both`, `OnlyA`, `OnlyB`) when both models are loaded.
    pub status: Option<CompareClassStatus>,
    /// Mean confidence of Model A's visible detections in this class (when count_a > 0).
    pub mean_score_a: Option<f32>,
    /// Mean confidence of Model B's visible detections in this class (when count_b > 0).
    pub mean_score_b: Option<f32>,
}

/// Concise A vs B comparison summary: counts, latencies, speedup, and shared/unique classes.
#[derive(Clone, Debug, PartialEq)]
pub struct CompareSummary {
    pub count_a: Option<usize>,
    pub count_b: Option<usize>,
    pub count_delta: Option<isize>,
    pub latency_a_ms: Option<f64>,
    pub latency_b_ms: Option<f64>,
    pub latency_delta_ms: Option<f64>,
    /// Whether Model A's prediction included a cold start / model warm-up.
    pub cold_a: Option<bool>,
    /// Whether Model B's prediction included a cold start / model warm-up.
    pub cold_b: Option<bool>,
    /// `latency_a_ms / latency_b_ms` (> 1 means Model B is faster than Model A). Suppressed (`None`)
    /// whenever either side was a cold start so warm-up overhead is not mistaken for a speed gap.
    pub speedup_ratio: Option<f64>,
    pub shared_classes: Vec<String>,
    pub only_a_classes: Vec<String>,
    pub only_b_classes: Vec<String>,
    pub classes: Vec<CompareClassRow>,
}

impl CompareSummary {
    /// Returns class rows matching `filter` (`All` keeps every row; `Both`/`OnlyA`/`OnlyB`
    /// require both models loaded and match the row's [`CompareClassStatus`]).
    pub fn filtered_classes(&self, filter: CompareClassFilter) -> Vec<&CompareClassRow> {
        match filter {
            CompareClassFilter::All => self.classes.iter().collect(),
            CompareClassFilter::Both => {
                self.classes.iter().filter(|r| r.status == Some(CompareClassStatus::Both)).collect()
            }
            CompareClassFilter::OnlyA => {
                self.classes.iter().filter(|r| r.status == Some(CompareClassStatus::OnlyA)).collect()
            }
            CompareClassFilter::OnlyB => {
                self.classes.iter().filter(|r| r.status == Some(CompareClassStatus::OnlyB)).collect()
            }
        }
    }
}

/// Picks a sensible default Model B for `model_a`: prefers another model with the same class
/// list (e.g. `folo-n` vs `folo-s` on the same dataset), then any other distinct model, falling
/// back to `model_a` in single-model workspaces.
pub fn pick_default_model_b<'a>(model_a: Option<&Model>, models: &'a [Model]) -> Option<&'a Model> {
    if models.is_empty() {
        return None;
    }
    let Some(a) = model_a else {
        return models.get(1).or_else(|| models.first());
    };
    models
        .iter()
        .find(|m| m.id != a.id && m.classes == a.classes)
        .or_else(|| models.iter().find(|m| m.id != a.id))
        .or_else(|| models.iter().find(|m| m.id == a.id))
        .or_else(|| models.first())
}

/// Summarizes detection count delta, latency A vs B, and shared/unique class breakdowns.
/// Pending or failed sides are passed as `None` so unavailable results never fabricate zero counts.
pub fn summarize_comparison(
    boxes_a: Option<&[&ScoredBox]>,
    boxes_b: Option<&[&ScoredBox]>,
    pred_a: Option<&PredictResponse>,
    pred_b: Option<&PredictResponse>,
) -> CompareSummary {
    let has_a = pred_a.is_some() && boxes_a.is_some();
    let has_b = pred_b.is_some() && boxes_b.is_some();

    #[derive(Default)]
    struct ClassAccum {
        color_class: u32,
        count_a: usize,
        sum_a: f64,
        fin_a: usize,
        count_b: usize,
        sum_b: f64,
        fin_b: usize,
    }
    let mut by_name: BTreeMap<String, ClassAccum> = BTreeMap::new();
    if let Some(boxes) = boxes_a.filter(|_| has_a) {
        for b in boxes {
            let key = if b.name.trim().is_empty() { format!("class {}", b.class) } else { b.name.clone() };
            let (s, fin) = if b.score.is_finite() { (f64::from(b.score), 1) } else { (0.0, 0) };
            let entry = by_name
                .entry(key)
                .or_insert_with(|| ClassAccum { color_class: b.class, ..Default::default() });
            entry.count_a += 1;
            entry.sum_a += s;
            entry.fin_a += fin;
        }
    }
    if let Some(boxes) = boxes_b.filter(|_| has_b) {
        for b in boxes {
            let key = if b.name.trim().is_empty() { format!("class {}", b.class) } else { b.name.clone() };
            let (s, fin) = if b.score.is_finite() { (f64::from(b.score), 1) } else { (0.0, 0) };
            let entry = by_name
                .entry(key)
                .or_insert_with(|| ClassAccum { color_class: b.class, ..Default::default() });
            entry.count_b += 1;
            entry.sum_b += s;
            entry.fin_b += fin;
        }
    }

    let mut shared_classes = Vec::new();
    let mut only_a_classes = Vec::new();
    let mut only_b_classes = Vec::new();
    if has_a && has_b {
        for (name, acc) in &by_name {
            match (acc.count_a > 0, acc.count_b > 0) {
                (true, true) => shared_classes.push(name.clone()),
                (true, false) => only_a_classes.push(name.clone()),
                (false, true) => only_b_classes.push(name.clone()),
                (false, false) => {}
            }
        }
    }

    let mut classes: Vec<CompareClassRow> = by_name
        .into_iter()
        .map(|(name, acc)| {
            let status = if has_a && has_b {
                match (acc.count_a > 0, acc.count_b > 0) {
                    (true, true) => Some(CompareClassStatus::Both),
                    (true, false) => Some(CompareClassStatus::OnlyA),
                    (false, true) => Some(CompareClassStatus::OnlyB),
                    (false, false) => None,
                }
            } else {
                None
            };
            CompareClassRow {
                name,
                color_class: acc.color_class,
                count_a: has_a.then_some(acc.count_a),
                count_b: has_b.then_some(acc.count_b),
                delta: (has_a && has_b).then_some(acc.count_b as isize - acc.count_a as isize),
                status,
                mean_score_a: (has_a && acc.fin_a > 0).then(|| (acc.sum_a / acc.fin_a as f64) as f32),
                mean_score_b: (has_b && acc.fin_b > 0).then(|| (acc.sum_b / acc.fin_b as f64) as f32),
            }
        })
        .collect();
    classes.sort_by(|x, y| {
        let dx = x.delta.map_or(-1, isize::abs);
        let dy = y.delta.map_or(-1, isize::abs);
        let sx = x.count_a.unwrap_or(0) + x.count_b.unwrap_or(0);
        let sy = y.count_a.unwrap_or(0) + y.count_b.unwrap_or(0);
        dy.cmp(&dx).then_with(|| sy.cmp(&sx)).then_with(|| x.name.cmp(&y.name))
    });

    let count_a = boxes_a.filter(|_| has_a).map(<[&ScoredBox]>::len);
    let count_b = boxes_b.filter(|_| has_b).map(<[&ScoredBox]>::len);
    let count_delta = count_a.zip(count_b).map(|(a, b)| b as isize - a as isize);

    let latency_a_ms = pred_a.map(|p| p.timing.total_ms).filter(|v| v.is_finite());
    let latency_b_ms = pred_b.map(|p| p.timing.total_ms).filter(|v| v.is_finite());
    let cold_a = pred_a.map(|p| p.cold);
    let cold_b = pred_b.map(|p| p.cold);
    let any_cold = cold_a.unwrap_or(false) || cold_b.unwrap_or(false);
    let latency_delta_ms = latency_a_ms.zip(latency_b_ms).map(|(a, b)| b - a);
    let speedup_ratio = if any_cold {
        None
    } else {
        latency_a_ms.zip(latency_b_ms).filter(|&(a, b)| a > 0.0 && b > 0.0).map(|(a, b)| a / b)
    };

    CompareSummary {
        count_a,
        count_b,
        count_delta,
        latency_a_ms,
        latency_b_ms,
        latency_delta_ms,
        cold_a,
        cold_b,
        speedup_ratio,
        shared_classes,
        only_a_classes,
        only_b_classes,
        classes,
    }
}

/// Formats a signed integer delta (`+3`, `-2`, `0`, or `—` when unavailable).
pub fn format_signed_delta(delta: Option<isize>) -> String {
    match delta {
        None => "—".to_string(),
        Some(d) if d > 0 => format!("+{d}"),
        Some(d) => d.to_string(),
    }
}

/// Formats latency comparison between Model A and Model B (`-4.2 ms · B 1.6× faster`, `+8.4 ms`,
/// or a warm-up notice when either side included a cold start).
pub fn format_latency_comparison(summary: &CompareSummary) -> String {
    let Some(delta) = summary.latency_delta_ms else {
        return "—".to_string();
    };
    match (summary.cold_a.unwrap_or(false), summary.cold_b.unwrap_or(false)) {
        (true, true) => return "Warm-up (A & B)".to_string(),
        (true, false) => return "Warm-up (A)".to_string(),
        (false, true) => return "Warm-up (B)".to_string(),
        (false, false) => {}
    }
    if delta.abs() < 0.05 {
        return "Same latency".to_string();
    }
    let sign = if delta > 0.0 { "+" } else { "-" };
    let diff = format!("{sign}{:.1} ms", delta.abs());
    if let Some(ratio) = summary.speedup_ratio {
        if ratio >= 1.15 {
            return format!("{diff} · B {ratio:.1}× faster");
        }
        if ratio <= 1.0 / 1.15 {
            return format!("{diff} · A {:.1}× faster", 1.0 / ratio);
        }
    }
    diff
}

/// Two-line Compare PNG pane caption: model identity (`title`) and count/timing (`meta`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparePaneCaption {
    pub title: String,
    pub meta: String,
}

impl ComparePaneCaption {
    /// Formats as a single-line caption when header width permits.
    pub fn single_line(&self) -> String {
        format!("{} · {}", self.title, self.meta)
    }
}

/// Structured title and metric caption lines for Compare PNG export headers.
pub fn compare_pane_caption_parts(
    side: &str,
    model_name: &str,
    count: Option<usize>,
    total_ms: Option<f64>,
    cold: bool,
) -> ComparePaneCaption {
    let title = format!("{side} · {model_name}");
    let meta = match count {
        None => "unavailable".to_string(),
        Some(n) => {
            let det = if n == 1 { "1 det".to_string() } else { format!("{n} dets") };
            match total_ms {
                Some(ms) if ms.is_finite() && cold => format!("{det} · {ms:.1} ms (cold)"),
                Some(ms) if ms.is_finite() => format!("{det} · {ms:.1} ms"),
                _ => det,
            }
        }
    };
    ComparePaneCaption { title, meta }
}

/// Side payload for structured Compare JSON export.
pub struct CompareSideExport<'a> {
    pub model: &'a str,
    pub model_id: &'a str,
    pub iou: Option<f32>,
    pub pred: &'a PredictResponse,
    pub boxes: &'a [&'a ScoredBox],
}

/// Formats Compare A/B predictions and delta summary as pretty-printed JSON.
/// Each side includes its `cold` warm-up flag, and `comparison` carries `cold_a`/`cold_b`
/// with `speedup_ratio: null` whenever either side was a cold start.
#[allow(clippy::too_many_arguments)]
pub fn compare_detections_json(
    image: &str,
    image_size: (u32, u32),
    conf: f32,
    iou: Option<f32>,
    class_filter: Option<u32>,
    summary: &CompareSummary,
    a: Option<&CompareSideExport<'_>>,
    b: Option<&CompareSideExport<'_>>,
) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct ImageDims {
        width: u32,
        height: u32,
    }
    #[derive(serde::Serialize)]
    struct ClassDiffCounts {
        both: usize,
        only_a: usize,
        only_b: usize,
    }
    #[derive(serde::Serialize)]
    struct ClassExport<'a> {
        name: &'a str,
        count_a: Option<usize>,
        count_b: Option<usize>,
        delta: Option<isize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<CompareClassStatus>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mean_score_a: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mean_score_b: Option<f32>,
    }
    #[derive(serde::Serialize)]
    struct SummaryExport<'a> {
        count_a: Option<usize>,
        count_b: Option<usize>,
        count_delta: Option<isize>,
        latency_a_ms: Option<f64>,
        latency_b_ms: Option<f64>,
        latency_delta_ms: Option<f64>,
        cold_a: Option<bool>,
        cold_b: Option<bool>,
        speedup_ratio: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        class_diff: Option<ClassDiffCounts>,
        shared_classes: &'a [String],
        only_a_classes: &'a [String],
        only_b_classes: &'a [String],
        classes: Vec<ClassExport<'a>>,
    }
    #[derive(serde::Serialize)]
    struct SideJson<'a> {
        model: &'a str,
        model_id: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        iou: Option<f32>,
        device: &'a str,
        cold: bool,
        timing: &'a Timing,
        detections: &'a [&'a ScoredBox],
    }
    #[derive(serde::Serialize)]
    struct CompareExport<'a> {
        mode: &'static str,
        image: &'a str,
        image_size: ImageDims,
        conf: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        iou: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        class_filter: Option<u32>,
        comparison: SummaryExport<'a>,
        a: Option<SideJson<'a>>,
        b: Option<SideJson<'a>>,
    }

    fn to_side<'a>(s: &CompareSideExport<'a>) -> SideJson<'a> {
        SideJson {
            model: s.model,
            model_id: s.model_id,
            iou: s.iou,
            device: &s.pred.device,
            cold: s.pred.cold,
            timing: &s.pred.timing,
            detections: s.boxes,
        }
    }

    let payload = CompareExport {
        mode: "compare",
        image,
        image_size: ImageDims { width: image_size.0, height: image_size.1 },
        conf,
        iou,
        class_filter,
        comparison: SummaryExport {
            count_a: summary.count_a,
            count_b: summary.count_b,
            count_delta: summary.count_delta,
            latency_a_ms: summary.latency_a_ms,
            latency_b_ms: summary.latency_b_ms,
            latency_delta_ms: summary.latency_delta_ms,
            cold_a: summary.cold_a,
            cold_b: summary.cold_b,
            speedup_ratio: summary.speedup_ratio,
            class_diff: (summary.count_a.is_some() && summary.count_b.is_some()).then_some(ClassDiffCounts {
                both: summary.shared_classes.len(),
                only_a: summary.only_a_classes.len(),
                only_b: summary.only_b_classes.len(),
            }),
            shared_classes: &summary.shared_classes,
            only_a_classes: &summary.only_a_classes,
            only_b_classes: &summary.only_b_classes,
            classes: summary
                .classes
                .iter()
                .map(|c| ClassExport {
                    name: &c.name,
                    count_a: c.count_a,
                    count_b: c.count_b,
                    delta: c.delta,
                    status: c.status,
                    mean_score_a: c.mean_score_a,
                    mean_score_b: c.mean_score_b,
                })
                .collect(),
        },
        a: a.map(to_side),
        b: b.map(to_side),
    };
    Ok(serde_json::to_string_pretty(&payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, ImageSize};

    fn fake_model(id: &str, name: &str, classes: &[&str]) -> Model {
        let mut base = crate::api::mock::MockBackend::new(4200, true).models().unwrap().remove(0);
        base.id = id.into();
        base.name = name.into();
        base.classes = classes.iter().map(|s| (*s).into()).collect();
        base
    }

    fn fake_box(class: u32, name: &str, score: f32) -> ScoredBox {
        ScoredBox { class, name: name.into(), score, x1: 10., y1: 10., x2: 50., y2: 50. }
    }

    fn fake_pred(model_id: &str, total_ms: f64, boxes: Vec<ScoredBox>) -> PredictResponse {
        PredictResponse {
            model_id: model_id.into(),
            image: ImageSize { width: 640, height: 480 },
            boxes,
            timing: Timing {
                preprocess_ms: 1.0,
                inference_ms: (total_ms - 1.5).max(0.1),
                postprocess_ms: 0.5,
                total_ms,
            },
            device: "cpu".into(),
            cold: false,
        }
    }

    #[test]
    fn default_model_b_prefers_same_classes_then_distinct_then_fallback() {
        let a = fake_model("m_a", "shapes-n", &["circle", "square"]);
        let b_other_cls = fake_model("m_b", "coco-n", &["person"]);
        let c_same_cls = fake_model("m_c", "shapes-s", &["circle", "square"]);
        let models = vec![a.clone(), b_other_cls.clone(), c_same_cls.clone()];
        assert_eq!(pick_default_model_b(Some(&a), &models).map(|m| m.id.as_str()), Some("m_c"));
        let diff_only = vec![a.clone(), b_other_cls];
        assert_eq!(pick_default_model_b(Some(&a), &diff_only).map(|m| m.id.as_str()), Some("m_b"));
        let single = vec![a.clone()];
        assert_eq!(pick_default_model_b(Some(&a), &single).map(|m| m.id.as_str()), Some("m_a"));
    }

    #[test]
    fn summarize_comparison_computes_deltas_speedup_and_class_sets() {
        let a_boxes = vec![fake_box(0, "circle", 0.9), fake_box(1, "square", 0.8)];
        let b_boxes =
            vec![fake_box(0, "circle", 0.95), fake_box(0, "circle", 0.85), fake_box(2, "triangle", 0.7)];
        let pa = fake_pred("m_a", 20.0, a_boxes.clone());
        let pb = fake_pred("m_b", 10.0, b_boxes.clone());
        let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
        let rb: Vec<&ScoredBox> = b_boxes.iter().collect();

        let s = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
        assert_eq!((s.count_a, s.count_b, s.count_delta), (Some(2), Some(3), Some(1)));
        assert_eq!(format_signed_delta(s.count_delta), "+1");
        assert_eq!(s.shared_classes, vec!["circle"]);
        assert_eq!(s.only_a_classes, vec!["square"]);
        assert_eq!(s.only_b_classes, vec!["triangle"]);
        assert_eq!(format_latency_comparison(&s), "-10.0 ms · B 2.0× faster");

        // Partial availability (B pending/failed) never fabricates zero counts or deltas.
        let partial = summarize_comparison(Some(&ra), None, Some(&pa), None);
        assert_eq!((partial.count_a, partial.count_b, partial.count_delta), (Some(2), None, None));
        assert_eq!(format_signed_delta(partial.count_delta), "—");
        assert_eq!(format_latency_comparison(&partial), "—");
        assert!(partial.shared_classes.is_empty());
    }

    #[test]
    fn compare_json_serializes_both_sides_and_partial_states() {
        let a_boxes = vec![fake_box(0, "circle", 0.91)];
        let b_boxes = vec![fake_box(0, "circle", 0.88), fake_box(1, "square", 0.77)];
        let pa = fake_pred("m_a", 12.0, a_boxes.clone());
        let pb = fake_pred("m_b", 18.0, b_boxes.clone());
        let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
        let rb: Vec<&ScoredBox> = b_boxes.iter().collect();
        let summary = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
        let json = compare_detections_json(
            "shapes / val_001.png",
            (640, 480),
            0.25,
            Some(0.7),
            None,
            &summary,
            Some(&CompareSideExport {
                model: "shapes-n",
                model_id: "m_a",
                iou: Some(0.7),
                pred: &pa,
                boxes: &ra,
            }),
            Some(&CompareSideExport { model: "shapes-s", model_id: "m_b", iou: None, pred: &pb, boxes: &rb }),
        )
        .expect("serialize compare JSON");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["mode"], "compare");
        assert_eq!(v["comparison"]["count_delta"], 1);
        assert_eq!(v["comparison"]["shared_classes"][0], "circle");
        assert_eq!(v["comparison"]["only_b_classes"][0], "square");
        assert_eq!(v["a"]["model"], "shapes-n");
        assert_eq!(v["a"]["cold"], false);
        assert!(v["a"].get("iou").is_some());
        assert_eq!(v["b"]["model"], "shapes-s");
        assert_eq!(v["b"]["cold"], false);
        assert!(v["b"].get("iou").is_none());
    }

    #[test]
    fn cold_start_prediction_suppresses_speedup_claim_and_serializes_cold_flags() {
        let a_boxes = vec![fake_box(0, "circle", 0.91)];
        let b_boxes = vec![fake_box(0, "circle", 0.88)];
        let mut pa = fake_pred("m_a", 2.0, a_boxes.clone());
        pa.cold = true;
        let pb = fake_pred("m_b", 0.1, b_boxes.clone());
        let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
        let rb: Vec<&ScoredBox> = b_boxes.iter().collect();

        let s_a = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
        assert_eq!(s_a.cold_a, Some(true));
        assert_eq!(s_a.cold_b, Some(false));
        assert_eq!(s_a.speedup_ratio, None, "cold Model A must not produce a speedup ratio");
        assert_eq!(format_latency_comparison(&s_a), "Warm-up (A)");

        let mut pb_cold = pb.clone();
        pb_cold.cold = true;
        let s_both = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb_cold));
        assert_eq!(s_both.speedup_ratio, None);
        assert_eq!(format_latency_comparison(&s_both), "Warm-up (A & B)");

        let mut pa_warm = pa.clone();
        pa_warm.cold = false;
        let s_b = summarize_comparison(Some(&ra), Some(&rb), Some(&pa_warm), Some(&pb_cold));
        assert_eq!(s_b.speedup_ratio, None);
        assert_eq!(format_latency_comparison(&s_b), "Warm-up (B)");

        let cap = compare_pane_caption_parts("A", "shapes-n", Some(1), Some(2.0), true);
        assert_eq!(cap.title, "A · shapes-n");
        assert_eq!(cap.meta, "1 det · 2.0 ms (cold)");

        let json = compare_detections_json(
            "val_001.png",
            (640, 480),
            0.25,
            None,
            None,
            &s_a,
            Some(&CompareSideExport { model: "shapes-n", model_id: "m_a", iou: None, pred: &pa, boxes: &ra }),
            Some(&CompareSideExport { model: "shapes-s", model_id: "m_b", iou: None, pred: &pb, boxes: &rb }),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["comparison"]["cold_a"], true);
        assert_eq!(v["comparison"]["cold_b"], false);
        assert!(v["comparison"]["speedup_ratio"].is_null());
        assert_eq!(v["a"]["cold"], true);
        assert_eq!(v["b"]["cold"], false);
    }

    #[test]
    fn class_diff_status_filtering_and_json_payload_cover_both_only_a_and_only_b() {
        let a_boxes = vec![fake_box(0, "circle", 0.90), fake_box(1, "square", 0.80)];
        let b_boxes =
            vec![fake_box(0, "circle", 0.95), fake_box(0, "circle", 0.85), fake_box(2, "triangle", 0.70)];
        let pa = fake_pred("m_a", 12.0, a_boxes.clone());
        let pb = fake_pred("m_b", 10.0, b_boxes.clone());
        let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
        let rb: Vec<&ScoredBox> = b_boxes.iter().collect();

        let summary = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
        assert_eq!(summary.filtered_classes(CompareClassFilter::All).len(), 3);
        let both = summary.filtered_classes(CompareClassFilter::Both);
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].name, "circle");
        assert_eq!(both[0].status, Some(CompareClassStatus::Both));
        assert!((both[0].mean_score_a.unwrap() - 0.90).abs() < 1e-4);
        assert!((both[0].mean_score_b.unwrap() - 0.90).abs() < 1e-4);

        let only_a = summary.filtered_classes(CompareClassFilter::OnlyA);
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].name, "square");
        assert_eq!(only_a[0].status, Some(CompareClassStatus::OnlyA));

        let only_b = summary.filtered_classes(CompareClassFilter::OnlyB);
        assert_eq!(only_b.len(), 1);
        assert_eq!(only_b[0].name, "triangle");
        assert_eq!(only_b[0].status, Some(CompareClassStatus::OnlyB));

        let json = compare_detections_json(
            "val_001.png",
            (640, 480),
            0.25,
            Some(0.7),
            None,
            &summary,
            Some(&CompareSideExport {
                model: "shapes-n",
                model_id: "m_a",
                iou: Some(0.7),
                pred: &pa,
                boxes: &ra,
            }),
            Some(&CompareSideExport {
                model: "shapes-s",
                model_id: "m_b",
                iou: Some(0.7),
                pred: &pb,
                boxes: &rb,
            }),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["comparison"]["class_diff"]["both"], 1);
        assert_eq!(v["comparison"]["class_diff"]["only_a"], 1);
        assert_eq!(v["comparison"]["class_diff"]["only_b"], 1);
        let rows = v["comparison"]["classes"].as_array().unwrap();
        assert!(rows.iter().any(|r| r["name"] == "circle" && r["status"] == "both"));
        assert!(rows.iter().any(|r| r["name"] == "square" && r["status"] == "only_a"));
        assert!(rows.iter().any(|r| r["name"] == "triangle" && r["status"] == "only_b"));
    }

    #[test]
    fn playground_compare_renders_class_diff_filters_and_exports_ab_json() {
        use crate::app::Route;
        use crate::app::tests::{drive, open};
        use crate::store::Load;

        let (mut app, window) = open();
        let dir = folo_test_support::TestDir::new("desktop-play-cmp-diff-ui");
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Playground, w, cx);
            this.set_play_compare(true, cx);
        });
        for _ in 0..4 {
            app.frame(window.into());
            app.run_until_idle();
        }
        drive(&mut app, window, |this, _, cx| {
            let ds = this.play_sample_dataset().expect("sample dataset for default model");
            let img = this
                .store
                .images
                .get(&ds)
                .and_then(|l| l.ready())
                .and_then(|list| list.first())
                .map(|m| m.id.clone())
                .unwrap_or_else(|| "img_00801".into());
            this.play_dataset_image(&ds, &img, cx);
        });
        for _ in 0..4 {
            app.frame(window.into());
            app.run_until_idle();
        }
        // Inject deterministic real-shaped A & B predictions covering Both, Only A, and Only B.
        drive(&mut app, window, |this, _, cx| {
            let mid_a = this.play_shown_model().map(|m| m.id.clone()).unwrap_or_else(|| "m_a".into());
            let mid_b = this.play_shown_model_b().map(|m| m.id.clone()).unwrap_or_else(|| "m_b".into());
            this.play_pred = Load::Ready(fake_pred(
                &mid_a,
                11.0,
                vec![fake_box(0, "circle", 0.91), fake_box(1, "square", 0.82)],
            ));
            this.play_compare.pred_b = Load::Ready(fake_pred(
                &mid_b,
                8.5,
                vec![fake_box(0, "circle", 0.94), fake_box(2, "triangle", 0.77)],
            ));
            cx.notify();
        });
        app.frame(window.into());

        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            for expected in [
                "play-cmp-cls-all",
                "play-cmp-cls-both",
                "play-cmp-cls-a",
                "play-cmp-cls-b",
                "play-export-json",
                "play-copy-json",
            ] {
                assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
            }
        });

        let out = dir.path().join("compare-ab.json");
        app.platform().answer_prompts(Some(vec![out.clone()]));
        drive(&mut app, window, |this, _, cx| this.export_play_json(cx));
        app.run_until_idle();
        let text = std::fs::read_to_string(&out).expect("compare JSON written");
        let v: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(v["mode"], "compare");
        assert_eq!(v["comparison"]["class_diff"]["both"], 1);
        assert_eq!(v["comparison"]["class_diff"]["only_a"], 1);
        assert_eq!(v["comparison"]["class_diff"]["only_b"], 1);
        drive(&mut app, window, |this, _, _| {
            assert!(this.toasts.iter().any(|t| t.title == "Exported A/B compare JSON"));
        });
    }

    #[test]
    fn playground_compare_delta_ignores_stale_side_during_rerun_interval() {
        use crate::app::Route;
        use crate::app::tests::{drive, open};
        use crate::store::Load;

        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Playground, w, cx);
            this.set_play_compare(true, cx);
        });
        for _ in 0..4 {
            app.frame(window.into());
            app.run_until_idle();
        }
        drive(&mut app, window, |this, _, cx| {
            let ds = this.play_sample_dataset().expect("sample dataset");
            this.play_dataset_image(&ds, "img_00801", cx);
        });
        for _ in 0..4 {
            app.frame(window.into());
            app.run_until_idle();
        }
        // Simulate an in-flight setting change (device switch): Model A already answered fresh for
        // the new target while Model B still holds an older Load::Ready prediction from the old target.
        drive(&mut app, window, |this, _, cx| {
            let mid_a = this.play_model.clone().unwrap();
            let mid_b = this.play_compare.model_b.clone().unwrap();
            let img_gen =
                this.play_compare.pred_a_meta.as_ref().expect("loaded image populates pred_a_meta").image_gen;
            let iou_bits = this.play_iou.to_bits();
            let floor = this.play_conf.min(crate::app::PLAY_CONF_FLOOR);
            this.play_pred_pending = false;
            this.play_compare.pred_b_pending = false;
            this.play_pred = Load::Ready(fake_pred(&mid_a, 9.0, vec![fake_box(0, "circle", 0.91)]));
            this.play_compare.pred_a_meta = Some(PlayPredMeta {
                image_gen: img_gen,
                model_id: mid_a,
                iou_bits,
                target: this.play_target,
                floor,
            });
            // Model B still has a Ready prediction, but from a different device target (stale!).
            let stale_target = Some(crate::local::Target::Cpu);
            assert_ne!(stale_target, this.play_target);
            this.play_compare.pred_b =
                Load::Ready(fake_pred(&mid_b, 12.0, vec![fake_box(1, "square", 0.88)]));
            this.play_compare.pred_b_meta = Some(PlayPredMeta {
                image_gen: img_gen,
                model_id: mid_b,
                iou_bits,
                target: stale_target,
                floor,
            });
            this.focus.borrow_mut().clear();
            cx.notify();
        });
        app.frame(window.into());

        drive(&mut app, window, |this, _, _| {
            assert!(this.pred_a_is_fresh());
            assert!(!this.pred_b_is_fresh(), "stale Model B must not be treated as fresh");
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            assert!(
                !keys.iter().any(|k| k == "play-cmp-cls-both"),
                "Both/Only A/Only B filter chips must not mix fresh A with stale B: {keys:?}"
            );
        });
    }
}
