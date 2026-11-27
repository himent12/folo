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

/// Completed prediction request parameters so stale cached results (old device, IoU, floor, or
/// per-class overrides) are never reused on re-enable or exported under mismatched thresholds.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayPredMeta {
    pub image_gen: u64,
    pub model_id: String,
    pub iou_bits: u32,
    pub target: Option<crate::local::Target>,
    pub floor: f32,
    pub per_class_conf: BTreeMap<u32, f32>,
}

impl PlayPredMeta {
    /// True when this completed prediction is still valid for the current Playground parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn matches_current(
        &self,
        image_gen: u64,
        model_id: &str,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        per_class: &BTreeMap<u32, f32>,
        e2e: bool,
    ) -> bool {
        self.image_gen == image_gen
            && self.model_id == model_id
            && self.target == target
            && (e2e || self.iou_bits == iou.to_bits())
            && super::thresholds::thresholds_match(&self.per_class_conf, self.floor, conf, per_class)
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
    pub fetched_per_class_b: BTreeMap<u32, f32>,
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
            fetched_per_class_b: BTreeMap::new(),
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
        self.fetched_floor_b = 0.05;
        self.fetched_per_class_b.clear();
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
    pub per_class_conf: Option<&'a BTreeMap<u32, f32>>,
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
    per_class_conf: Option<&BTreeMap<u32, f32>>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        per_class_conf: Option<&'a BTreeMap<u32, f32>>,
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
        per_class_conf: Option<&'a BTreeMap<u32, f32>>,
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
            per_class_conf: s.per_class_conf.filter(|m| !m.is_empty()),
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
        per_class_conf: per_class_conf.filter(|m| !m.is_empty()),
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
#[path = "compare_tests.rs"]
mod tests;
