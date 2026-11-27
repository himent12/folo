//! Playground multi-image / folder batch state, folder scanner, and compact filmstrip UI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::thresholds::{filter_detections_with_per_class, thresholds_match};
use crate::api::{ApiError, ApiResult, PredictResponse};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, alpha, radius, ty};
use crate::ui::*;

/// Maximum image files loaded from a single folder pick to keep UI responsive.
pub const MAX_BATCH_FILES: usize = 500;

/// Supported image file extensions matching `IMAGE_FORMATS` / `folo_data::IMAGE_EXTENSIONS`.
pub fn is_supported_image_ext(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|ext| {
        matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp" | "webp")
    })
}

/// Scans `dir` (top-level regular files) for supported images sorted deterministically by filename.
/// Returns `(paths, truncated_count)` where `truncated_count` is how many supported images exceeded
/// [`MAX_BATCH_FILES`].
pub fn scan_image_folder(dir: &Path) -> ApiResult<(Vec<PathBuf>, usize)> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| ApiError::new("read_failed", format!("Could not read folder: {e}")))?;
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && is_supported_image_ext(&path) {
            let key = entry.file_name().to_string_lossy().to_ascii_lowercase();
            found.push((key, path));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let total = found.len();
    let truncated = total.saturating_sub(MAX_BATCH_FILES);
    found.truncate(MAX_BATCH_FILES);
    Ok((found.into_iter().map(|(_, p)| p).collect(), truncated))
}

/// Where a Playground batch item's pixels come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlayBatchSource {
    File(PathBuf),
    Dataset { dataset_id: String, image_id: String },
}

/// Fast 64-bit FNV-1a fingerprint of raw file bytes so in-place file replacements invalidate
/// cached batch predictions instead of painting detections onto replaced pixels.
pub fn bytes_fingerprint(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Completed inference result for one batch item, tagged with its exact request parameters so stale
/// results under another model, device, IoU, or lower confidence floor are never reused or exported.
#[derive(Clone, Debug)]
pub struct BatchItemResult {
    pub pred: PredictResponse,
    pub width: u32,
    pub height: u32,
    pub model_id: String,
    pub iou_bits: u32,
    pub target: Option<crate::local::Target>,
    pub floor: f32,
    pub per_class_conf: BTreeMap<u32, f32>,
    /// Content hash of file bytes used for inference (`None` for server dataset images).
    pub content_hash: Option<u64>,
}

impl BatchItemResult {
    /// True when this cached result is valid for the current Playground parameters.
    pub fn matches_current(
        &self,
        model_id: &str,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        per_class: &BTreeMap<u32, f32>,
        e2e: bool,
    ) -> bool {
        self.model_id == model_id
            && self.target == target
            && (e2e || self.iou_bits == iou.to_bits())
            && thresholds_match(&self.per_class_conf, self.floor, conf, per_class)
    }
}

/// Lifecycle state of one item in the Playground batch filmstrip.
#[derive(Clone, Debug)]
pub enum BatchItemStatus {
    Pending,
    Running,
    Done(BatchItemResult),
    Failed(String),
}

/// One image queued in Playground batch mode.
#[derive(Clone, Debug)]
pub struct PlayBatchItem {
    pub name: SharedString,
    pub source: PlayBatchSource,
    pub status: BatchItemStatus,
}

/// Summary counters for the batch filmstrip header bar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchSummary {
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub running: usize,
    pub total_detections: usize,
}

/// Playground multi-image / folder batch state owned by `FoloApp`.
pub struct PlayBatchState {
    /// Monotonic id identifying the currently loaded batch; bumped on install and clear.
    pub batch_id: u64,
    pub items: Vec<PlayBatchItem>,
    pub active: usize,
    /// Active QA filter (`All` / `With detections` / `Empty` / `By class`) on the filmstrip and
    /// Batch Summary CSV export.
    pub qa_filter: super::batch_qa::BatchQaFilter,
    /// True while sequential `Run All` is actively running.
    pub running: bool,
    /// Generation counter bumped on start, cancel, clear, or any parameter/route/backend change so
    /// in-flight batch callbacks can never write stale results.
    pub run_gen: u64,
    /// Generation counter for async folder scan / multi-file / demo-batch intake.
    pub load_gen: u64,
    /// Dedicated local-inference ticket line for batch runs (bumped on cancel).
    pub line: Arc<AtomicU64>,
}

impl Default for PlayBatchState {
    fn default() -> Self {
        Self {
            batch_id: 0,
            items: Vec::new(),
            active: 0,
            qa_filter: super::batch_qa::BatchQaFilter::All,
            running: false,
            run_gen: 0,
            load_gen: 0,
            line: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Builds display/export names for local file paths, extending ancestor directory components until
/// every duplicate path suffix is uniquely disambiguated.
pub fn disambiguate_file_labels(paths: &[PathBuf]) -> Vec<SharedString> {
    let parts: Vec<Vec<String>> = paths
        .iter()
        .map(|p| {
            p.components()
                .filter_map(|c| match c {
                    std::path::Component::Normal(s) => Some(s.to_string_lossy().to_string()),
                    _ => None,
                })
                .collect()
        })
        .collect();
    let mut depths: Vec<usize> = vec![1; paths.len()];
    let make_label = |i: usize, depth: usize| -> String {
        let comps = &parts[i];
        if comps.is_empty() {
            return paths[i].to_string_lossy().to_string();
        }
        let take = depth.min(comps.len());
        comps[comps.len() - take..].join("/")
    };
    loop {
        let mut counts: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, &depth) in depths.iter().enumerate() {
            counts.entry(make_label(i, depth)).or_default().push(i);
        }
        let mut expanded = false;
        for idxs in counts.values() {
            if idxs.len() > 1 {
                for &i in idxs {
                    if depths[i] < parts[i].len() {
                        depths[i] += 1;
                        expanded = true;
                    }
                }
            }
        }
        if !expanded {
            break;
        }
    }
    (0..paths.len()).map(|i| make_label(i, depths[i]).into()).collect()
}

impl PlayBatchState {
    /// Batch filmstrip activates only when 2 or more images are loaded.
    pub fn is_active(&self) -> bool {
        self.items.len() >= 2
    }

    /// Cancels an active sequential Run All and resets any `Running` item back to `Pending`.
    pub fn cancel(&mut self) -> bool {
        let was_running = self.running;
        self.running = false;
        self.run_gen += 1;
        let _ = crate::local::Ticket::take(&self.line);
        for item in &mut self.items {
            if matches!(item.status, BatchItemStatus::Running) {
                item.status = BatchItemStatus::Pending;
            }
        }
        was_running
    }

    /// Clears the entire batch list and cancels any in-flight scan or Run All.
    pub fn clear(&mut self) {
        self.cancel();
        self.batch_id += 1;
        self.load_gen += 1;
        self.items.clear();
        self.active = 0;
        self.qa_filter = super::batch_qa::BatchQaFilter::All;
    }

    /// Marks all completed/failed items back to `Pending` when model, device, or NMS IoU changes.
    pub fn invalidate_results(&mut self) {
        for item in &mut self.items {
            if !matches!(item.status, BatchItemStatus::Pending) {
                item.status = BatchItemStatus::Pending;
            }
        }
    }

    /// Marks any completed item whose recorded NMS IoU differs from `iou` back to `Pending`,
    /// preserving results whose recorded IoU still matches.
    pub fn invalidate_mismatched_iou(&mut self, iou: f32) {
        let bits = iou.to_bits();
        for item in &mut self.items {
            if let BatchItemStatus::Done(res) = &item.status
                && res.iou_bits != bits
            {
                item.status = BatchItemStatus::Pending;
            }
        }
    }

    /// Marks any completed item whose fetched confidence floor or per-class map no longer matches
    /// `(conf, per_class)` back to `Pending`.
    pub fn invalidate_for_thresholds(&mut self, conf: f32, per_class: &BTreeMap<u32, f32>) {
        for item in &mut self.items {
            if let BatchItemStatus::Done(res) = &item.status
                && !thresholds_match(&res.per_class_conf, res.floor, conf, per_class)
            {
                item.status = BatchItemStatus::Pending;
            }
        }
    }

    /// Returns the fresh `BatchItemResult` for item `idx` if it matches current Playground params.
    #[allow(clippy::too_many_arguments)]
    pub fn fresh_result(
        &self,
        idx: usize,
        model_id: &str,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        per_class: &BTreeMap<u32, f32>,
        e2e: bool,
    ) -> Option<&BatchItemResult> {
        let item = self.items.get(idx)?;
        match &item.status {
            BatchItemStatus::Done(res)
                if res.matches_current(model_id, iou, target, conf, per_class, e2e) =>
            {
                Some(res)
            }
            _ => None,
        }
    }

    /// Computes live done/failed/running/detection counts filtered at `(conf, per_class, class_filter)`.
    #[allow(clippy::too_many_arguments)]
    pub fn summary(
        &self,
        model_id: Option<&str>,
        iou: f32,
        target: Option<crate::local::Target>,
        conf: f32,
        per_class: &BTreeMap<u32, f32>,
        class_filter: Option<u32>,
        e2e: bool,
    ) -> BatchSummary {
        let mut s = BatchSummary { total: self.items.len(), ..Default::default() };
        for item in &self.items {
            match &item.status {
                BatchItemStatus::Running => s.running += 1,
                BatchItemStatus::Failed(_) => s.failed += 1,
                BatchItemStatus::Done(res) => {
                    if model_id.is_some_and(|m| res.matches_current(m, iou, target, conf, per_class, e2e)) {
                        s.done += 1;
                        s.total_detections +=
                            filter_detections_with_per_class(&res.pred.boxes, conf, per_class, class_filter)
                                .len();
                    }
                }
                BatchItemStatus::Pending => {}
            }
        }
        s
    }
}

/// Computes how many compact 124 px batch tiles (with 6 px gaps) fit inside `stage_w` cleanly.
pub fn batch_filmstrip_max_tiles(stage_w: f32) -> usize {
    let inner = (stage_w - 24.).max(0.);
    (((inner + 6.) / 130.).floor() as usize).clamp(2, 10)
}

/// Short display filename for a batch tile (strips optional `"dataset / "` prefix).
pub fn batch_tile_label(name: &str) -> &str {
    name.rsplit_once(" / ").map_or(name, |(_, f)| f)
}

/// Labels for the Playground batch run and export buttons; in Compare A/B mode they explicitly
/// name Model A so users never expect Model B to be included in Model A's batch outputs.
pub(crate) fn batch_action_labels(
    compare_on: bool,
    all_fresh: bool,
) -> (&'static str, &'static str, &'static str) {
    if compare_on {
        (if all_fresh { "Re-run all (A)" } else { "Run all (A)" }, "Batch JSON (A)", "Batch CSV (A)")
    } else {
        (if all_fresh { "Re-run all" } else { "Run all" }, "Batch JSON", "Batch CSV")
    }
}

impl FoloApp {
    /// Persistent failure banner shown whenever the active batch item failed to load or decode,
    /// regardless of whether Playground's stage is empty or still showing a prior image.
    pub(super) fn play_batch_failed_banner(
        &self,
        t: &Theme,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        if !self.play_batch.is_active() {
            return None;
        }
        let idx = self.play_batch.active;
        let item = self.play_batch.items.get(idx)?;
        let BatchItemStatus::Failed(msg) = &item.status else {
            return None;
        };
        if self.play_image.as_ref().and_then(|i| i.batch_ref) == Some((self.play_batch.batch_id, idx)) {
            return None;
        }
        let fname = item.name.clone();
        let detail = match self.play_image.as_ref() {
            Some(prev) => format!(
                "{msg} · Showing previous image ({}) until you retry or switch.",
                batch_tile_label(prev.name.as_ref())
            ),
            None => msg.clone(),
        };
        let retry = self.button(
            "play-batch-retry-active",
            "Retry",
            Btn::Secondary,
            Some("icons/refresh-cw.svg"),
            window,
            cx,
            act(move |this, _, cx| this.select_play_batch_item(idx, cx)),
        );
        let next = self.button(
            "play-batch-skip-next",
            "Next image",
            Btn::Secondary,
            Some("icons/chevron-right.svg"),
            window,
            cx,
            act(|this, _, cx| this.step_play_batch(1, cx)),
        );
        Some(
            div()
                .flex_none()
                .px(px(12.))
                .py(px(8.))
                .rounded(px(radius::MD))
                .bg(t.danger_muted)
                .border_1()
                .border_color(alpha(t.danger_fill, 0.35))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(10.))
                .child(
                    div()
                        .min_w_0()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(text(ty::SM, t.danger_text).child(format!("Could not load {fname}")))
                        .child(text(ty::XS, t.text_secondary).whitespace_normal().child(detail)),
                )
                .child(div().flex().items_center().gap(px(6.)).flex_none().child(retry).child(next))
                .into_any_element(),
        )
    }

    /// Compact batch filmstrip rendered below the Playground stage when 2+ images are loaded.
    pub(super) fn play_batch_filmstrip(
        &mut self,
        t: &Theme,
        stage_w: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let total = self.play_batch.items.len();
        let active = self.play_batch.active.min(total.saturating_sub(1));
        let running = self.play_batch.running;
        let model_id = self.play_model.clone();
        let e2e = self.play_shown_model().is_some_and(|m| self.model_e2e(m));
        let eff_class = self.play_batch.effective_class(self.play_class);
        let summary = self.play_batch.summary(
            model_id.as_deref(),
            self.play_iou,
            self.play_target,
            self.play_conf,
            &self.play_thresholds.per_class,
            eff_class,
            e2e,
        );

        let prev_btn = self.icon_button(
            "play-batch-prev",
            "icons/chevron-left.svg",
            "Previous image (← or [)",
            window,
            cx,
            act(|this, _, cx| this.step_play_batch(-1, cx)),
        );
        let next_btn = self.icon_button(
            "play-batch-next",
            "icons/chevron-right.svg",
            "Next image (→ or ])",
            window,
            cx,
            act(|this, _, cx| this.step_play_batch(1, cx)),
        );
        let clear_btn = self.icon_button(
            "play-batch-clear",
            "icons/x.svg",
            "Close batch strip",
            window,
            cx,
            act(|this, _, cx| {
                this.clear_play_batch();
                cx.notify();
            }),
        );

        // Keep keyboard focus continuous when Run all <-> Cancel <-> disabled swap.
        let run_disabled = !running && model_id.is_none();
        if running && self.fh("play-batch-run-all", cx).is_focused(window) {
            window.focus(&self.fh("play-batch-cancel", cx));
        } else if !running && self.fh("play-batch-cancel", cx).is_focused(window) {
            window.focus(
                &self.fh(if run_disabled { "play-batch-run-all-disabled" } else { "play-batch-run-all" }, cx),
            );
        } else if run_disabled && self.fh("play-batch-run-all", cx).is_focused(window) {
            window.focus(&self.fh("play-batch-run-all-disabled", cx));
        } else if !run_disabled && !running && self.fh("play-batch-run-all-disabled", cx).is_focused(window) {
            window.focus(&self.fh("play-batch-run-all", cx));
        }

        let compare_on = self.play_compare.enabled && self.store.models.ready().is_none_or(|m| !m.is_empty());
        let run_action: AnyElement = if running {
            self.button(
                "play-batch-cancel",
                "Cancel",
                Btn::Danger,
                Some("icons/square.svg"),
                window,
                cx,
                act(|this, _, cx| {
                    this.cancel_play_batch();
                    this.toast(Tone::Neutral, "Batch run cancelled", None, cx);
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip("Cancel remaining batch inference", Some("Esc")))
            .into_any_element()
        } else if model_id.is_none() {
            let (run_label, _, _) = batch_action_labels(compare_on, false);
            self.button_disabled(
                "play-batch-run-all",
                run_label,
                Btn::Secondary,
                Some("icons/play.svg"),
                "Select a model first",
                window,
                cx,
            )
            .into_any_element()
        } else {
            let all_fresh = summary.done == total && summary.failed == 0;
            let (label, _, _) = batch_action_labels(compare_on, all_fresh);
            let tip = if compare_on {
                "Run Model A sequentially across every batch image"
            } else {
                "Run the active model sequentially across every batch image"
            };
            self.button(
                "play-batch-run-all",
                label,
                if all_fresh { Btn::Secondary } else { Btn::Primary },
                Some("icons/play.svg"),
                window,
                cx,
                act(move |this, _, cx| this.start_play_batch(all_fresh, cx)),
            )
            .tooltip(crate::controls::tip(tip, None))
            .into_any_element()
        };

        let model_prefix = if compare_on { "Model A · " } else { "" };
        let status_label = if running {
            let active_run = self
                .play_batch
                .items
                .iter()
                .position(|it| matches!(it.status, BatchItemStatus::Running))
                .map_or(summary.done + 1, |i| i + 1);
            format!("{model_prefix}Running {active_run} / {total}…")
        } else if summary.failed > 0 {
            format!("{model_prefix}{} / {total} run · {} failed", summary.done, summary.failed)
        } else if summary.done > 0 {
            format!(
                "{model_prefix}{} / {total} run · {}",
                summary.done,
                plural(summary.total_detections, "det")
            )
        } else {
            format!("{total} images ready")
        };

        let model_classes = self.play_shown_model().map(|m| m.classes.clone()).unwrap_or_default();
        let qa_counts = self.play_batch.qa_counts(
            model_id.as_deref(),
            self.play_iou,
            self.play_target,
            self.play_conf,
            &self.play_thresholds.per_class,
            self.play_class,
            e2e,
            &model_classes,
        );
        let filtered = self.play_batch.filtered_indices(
            model_id.as_deref(),
            self.play_iou,
            self.play_target,
            self.play_conf,
            &self.play_thresholds.per_class,
            self.play_class,
            e2e,
        );
        let qa_filter_row = self.play_batch_qa_filter_row(t, &qa_counts, window, cx);
        let filter_active = self.play_batch.qa_filter != super::batch_qa::BatchQaFilter::All;
        let counter_label = if filter_active {
            if filtered.is_empty() {
                format!("0 / 0 ({total} total)")
            } else {
                match filtered.iter().position(|&i| i == active) {
                    Some(pos) => format!("{} / {} ({total} total)", pos + 1, filtered.len()),
                    None => format!("{} / {total} · {} match", active + 1, filtered.len()),
                }
            }
        } else {
            format!("{} / {total}", active + 1)
        };

        if !filtered.is_empty() && self.fh("play-batch-qa-reset", cx).is_focused(window) {
            window.focus(&self.fh("play-batch-qa-all", cx));
        }

        let max_tiles = batch_filmstrip_max_tiles(stage_w);
        let tile_row: AnyElement = if filtered.is_empty() {
            let show_all = self.button(
                "play-batch-qa-reset",
                "Show all",
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, w, cx| {
                    this.set_play_batch_qa_filter(super::batch_qa::BatchQaFilter::All, cx);
                    this.focus_key("play-batch-qa-all", w, cx);
                }),
            );
            div()
                .h(px(32.))
                .flex()
                .items_center()
                .gap(px(8.))
                .child(text(ty::XS, t.text_tertiary).child("No batch images match this QA filter."))
                .child(show_all)
                .into_any_element()
        } else {
            let active_pos = filtered.iter().position(|&i| i == active).unwrap_or(0);
            let start =
                active_pos.saturating_sub(max_tiles / 2).min(filtered.len().saturating_sub(max_tiles));
            let end = (start + max_tiles).min(filtered.len());
            let items_slice: Vec<(usize, SharedString, BatchItemStatus)> = filtered[start..end]
                .iter()
                .map(|&idx| {
                    let item = &self.play_batch.items[idx];
                    (idx, item.name.clone(), item.status.clone())
                })
                .collect();

            let tiles: Vec<AnyElement> = items_slice
                .into_iter()
                .map(|(idx, name, status)| {
                    let is_cur = idx == active;
                    let short = batch_tile_label(name.as_ref()).to_string();
                    let (dot_color, count_str, tip_detail) = match &status {
                        BatchItemStatus::Pending => {
                            (t.text_disabled, "—".to_string(), "Not run yet".to_string())
                        }
                        BatchItemStatus::Running => {
                            (t.accent_solid, "…".to_string(), "Detecting…".to_string())
                        }
                        BatchItemStatus::Failed(err) => {
                            (t.danger_solid, "!".to_string(), format!("Failed: {err}"))
                        }
                        BatchItemStatus::Done(res) => {
                            let fresh = model_id.as_deref().is_some_and(|m| {
                                res.matches_current(
                                    m,
                                    self.play_iou,
                                    self.play_target,
                                    self.play_conf,
                                    &self.play_thresholds.per_class,
                                    e2e,
                                )
                            });
                            if fresh {
                                let n = filter_detections_with_per_class(
                                    &res.pred.boxes,
                                    self.play_conf,
                                    &self.play_thresholds.per_class,
                                    eff_class,
                                )
                                .len();
                                (t.success_solid, n.to_string(), plural(n, "detection"))
                            } else {
                                (
                                    t.text_disabled,
                                    "—".to_string(),
                                    "Needs re-run for current settings".to_string(),
                                )
                            }
                        }
                    };
                    let key = format!("play-batch-item-{idx}");
                    let tip_text = format!("{} · {tip_detail}", name.as_ref());
                    let el = div()
                        .id(SharedString::from(key.clone()))
                        .tooltip(crate::controls::tip(tip_text, None))
                        .w(px(124.))
                        .h(px(32.))
                        .flex_none()
                        .px(px(8.))
                        .rounded(px(radius::SM))
                        .bg(if is_cur { t.bg_active } else { t.bg_raised })
                        .border_1()
                        .border_color(if is_cur { t.accent_solid } else { alpha(t.line_default, 0.8) })
                        .cursor_pointer()
                        .hover(|s| s.border_color(if is_cur { t.accent_solid } else { t.line_strong }))
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .child(div().size(px(6.)).rounded_full().flex_none().bg(dot_color))
                        .child(
                            text(ty::XS, if is_cur { t.text_primary } else { t.text_secondary })
                                .font_family(crate::theme::MONO)
                                .font_weight(if is_cur { FontWeight::MEDIUM } else { FontWeight::NORMAL })
                                .flex_1()
                                .min_w_0()
                                .ellipsis()
                                .child(short),
                        )
                        .child(
                            text(
                                ty::XS,
                                if matches!(status, BatchItemStatus::Failed(_)) {
                                    t.danger_text
                                } else {
                                    t.text_tertiary
                                },
                            )
                            .tnum()
                            .flex_none()
                            .child(count_str),
                        );
                    self.bind(
                        el,
                        &key,
                        radius::SM,
                        window,
                        cx,
                        act(move |this, _, cx| this.select_play_batch_item(idx, cx)),
                    )
                    .into_any_element()
                })
                .collect();
            div().flex().items_center().gap(px(6.)).overflow_hidden().children(tiles).into_any_element()
        };

        div()
            .id("play-batch-strip")
            .flex_none()
            .bg(t.bg_panel)
            .border_t_1()
            .border_color(t.line_subtle)
            .px(px(12.))
            .py(px(8.))
            .flex()
            .flex_col()
            .gap(px(6.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .min_w_0()
                            .child(prev_btn)
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .tnum()
                                    .flex_none()
                                    .child(counter_label),
                            )
                            .child(next_btn)
                            .child(
                                text(
                                    ty::XS,
                                    if summary.failed > 0 && !running {
                                        t.danger_text
                                    } else {
                                        t.text_tertiary
                                    },
                                )
                                .tnum()
                                .min_w_0()
                                .ellipsis()
                                .child(status_label),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_none()
                            .items_center()
                            .gap(px(6.))
                            .child(run_action)
                            .child(clear_btn),
                    ),
            )
            .child(qa_filter_row)
            .child(tile_row)
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ImageSize, ScoredBox, Timing};

    fn fake_result(model_id: &str, iou: f32, floor: f32, boxes: Vec<ScoredBox>) -> BatchItemResult {
        BatchItemResult {
            pred: PredictResponse {
                model_id: model_id.into(),
                image: ImageSize { width: 640, height: 480 },
                boxes,
                timing: Timing { preprocess_ms: 1.0, inference_ms: 4.0, postprocess_ms: 0.5, total_ms: 5.5 },
                device: "cpu".into(),
                cold: false,
            },
            width: 640,
            height: 480,
            model_id: model_id.into(),
            iou_bits: iou.to_bits(),
            target: None,
            floor,
            per_class_conf: BTreeMap::new(),
            content_hash: None,
        }
    }

    #[test]
    fn disambiguate_file_labels_expands_multi_level_collisions() {
        let paths = vec![
            PathBuf::from("lot-a/images/frame.png"),
            PathBuf::from("lot-b/images/frame.png"),
            PathBuf::from("lot-a/other/unique.png"),
        ];
        let labels = disambiguate_file_labels(&paths);
        assert_eq!(
            labels.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
            vec!["lot-a/images/frame.png", "lot-b/images/frame.png", "unique.png"]
        );
    }

    #[test]
    fn folder_scanner_filters_supported_extensions_and_sorts_deterministically() {
        let dir = folo_test_support::TestDir::new("desktop-play-batch-scan");
        for f in ["z_img.JPG", "a_img.png", "m_img.webp", "b_img.bmp", "notes.txt", "anim.gif", "c.jpeg"] {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        std::fs::create_dir(dir.path().join("subdir.png")).unwrap();
        let (paths, truncated) = scan_image_folder(dir.path()).unwrap();
        assert_eq!(truncated, 0);
        let names: Vec<String> =
            paths.iter().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert_eq!(names, vec!["a_img.png", "b_img.bmp", "c.jpeg", "m_img.webp", "z_img.JPG"]);
    }

    #[test]
    fn batch_result_freshness_respects_model_iou_target_and_floor() {
        let res = fake_result(
            "m_1",
            0.7,
            0.05,
            vec![
                ScoredBox { class: 0, name: "circle".into(), score: 0.9, x1: 10., y1: 10., x2: 50., y2: 50. },
                ScoredBox { class: 1, name: "square".into(), score: 0.2, x1: 60., y1: 10., x2: 90., y2: 40. },
            ],
        );
        let empty = BTreeMap::new();
        assert!(res.matches_current("m_1", 0.7, None, 0.25, &empty, false));
        assert!(!res.matches_current("m_2", 0.7, None, 0.25, &empty, false));
        assert!(!res.matches_current("m_1", 0.5, None, 0.25, &empty, false));
        assert!(res.matches_current("m_1", 0.5, None, 0.25, &empty, true), "e2e ignores IoU");
        assert!(!res.matches_current("m_1", 0.7, Some(crate::local::Target::Cpu), 0.25, &empty, false));
        assert!(!res.matches_current("m_1", 0.7, None, 0.02, &empty, false), "below fetched floor");
        let ov = BTreeMap::from([(0u32, 0.8f32)]);
        assert!(
            !res.matches_current("m_1", 0.7, None, 0.25, &ov, false),
            "per-class overrides invalidate scalar cache"
        );
    }

    #[test]
    fn batch_state_cancel_and_summary_filter_by_conf() {
        let mut st = PlayBatchState {
            items: vec![
                PlayBatchItem {
                    name: "shapes / a.png".into(),
                    source: PlayBatchSource::Dataset { dataset_id: "ds".into(), image_id: "i1".into() },
                    status: BatchItemStatus::Done(fake_result(
                        "m_1",
                        0.7,
                        0.05,
                        vec![
                            ScoredBox {
                                class: 0,
                                name: "circle".into(),
                                score: 0.9,
                                x1: 10.,
                                y1: 10.,
                                x2: 50.,
                                y2: 50.,
                            },
                            ScoredBox {
                                class: 1,
                                name: "square".into(),
                                score: 0.2,
                                x1: 60.,
                                y1: 10.,
                                x2: 90.,
                                y2: 40.,
                            },
                        ],
                    )),
                },
                PlayBatchItem {
                    name: "b.png".into(),
                    source: PlayBatchSource::File(PathBuf::from("b.png")),
                    status: BatchItemStatus::Running,
                },
            ],
            running: true,
            ..Default::default()
        };
        assert!(st.is_active());
        assert_eq!(batch_tile_label("shapes / a.png"), "a.png");

        let s = st.summary(Some("m_1"), 0.7, None, 0.25, &BTreeMap::new(), None, false);
        assert_eq!((s.total, s.done, s.running, s.total_detections), (2, 1, 1, 1));

        assert!(st.cancel());
        assert!(!st.running);
        assert!(matches!(st.items[1].status, BatchItemStatus::Pending));
    }
}
