//! Playground per-class confidence threshold mutations, cross-model class remapping,
//! JSON import/copy/export, and calibrated Models -> Playground handoff.

use std::collections::BTreeMap;

use folo_ui::{Context, Window};

use crate::screens::playground::{
    MAX_THRESHOLDS_JSON_BYTES, calibration_to_overrides, export_calibration_json, export_filename,
    export_thresholds_json, fmt_threshold, map_per_class_to_classes, parse_thresholds_json, round_conf,
    thresholds_match, write_export_bytes,
};
use crate::store::Load;
use crate::ui::*;

use super::playground::PLAY_CONF_FLOOR;
use super::playground_batch::picked_paths;
use super::playground_compare::{export_default_dir, picked_save};
use super::{FoloApp, Route, act};

impl FoloApp {
    /// Resolves `(request_conf, request_per_class, request_inclusive, cache_floor)` for a
    /// `PredictRequest`. Local inference and servers advertising `threshold_calibration` enforce
    /// `per_class_conf` + inclusive scoring before `max_det` truncation; older servers ignore those
    /// fields, so we fall back to a low scalar floor covering the minimum active class override and
    /// filter per-class client-side.
    pub(crate) fn play_predict_threshold_params(
        &self,
        per_class: &BTreeMap<u32, f32>,
    ) -> (f32, Option<BTreeMap<u32, f32>>, bool, f32) {
        if per_class.is_empty() {
            let floor = self.play_conf.min(PLAY_CONF_FLOOR);
            return (floor, None, false, floor);
        }
        let native_per_class = self.play_target.is_some() || self.store.supports_threshold_calibration();
        if native_per_class {
            (self.play_conf, Some(per_class.clone()), true, self.play_conf)
        } else {
            let min_override =
                per_class.values().copied().filter(|v| v.is_finite()).fold(self.play_conf, f32::min);
            (min_override.min(PLAY_CONF_FLOOR), None, false, self.play_conf)
        }
    }

    /// Model B's effective per-class confidence overrides mapped from Model A's class list by name.
    pub(crate) fn play_per_class_b(&self) -> BTreeMap<u32, f32> {
        if self.play_thresholds.per_class.is_empty() {
            return BTreeMap::new();
        }
        let from = self.play_shown_model().map(|m| m.classes.as_slice()).unwrap_or(&[]);
        let to = self.play_shown_model_b().map(|m| m.classes.as_slice()).unwrap_or(&[]);
        map_per_class_to_classes(&self.play_thresholds.per_class, from, to)
    }

    /// Remaps `play_thresholds.per_class` cleanly when Model A switches from `from_classes` to
    /// `to_model_id` so stale overrides never silently leak across incompatible class lists.
    pub(crate) fn remap_play_thresholds_for_model(
        &mut self,
        from_classes: Option<Vec<String>>,
        to_model_id: Option<&str>,
    ) {
        self.play_thresh_import_seq += 1;
        self.play_thresholds.clear_feedback();
        if self.play_thresholds.per_class.is_empty() {
            return;
        }
        let to_classes =
            to_model_id.and_then(|id| self.store.model(id)).map(|m| m.classes.clone()).unwrap_or_default();
        self.play_thresholds.per_class = match from_classes {
            Some(from) if !from.is_empty() && !to_classes.is_empty() => {
                map_per_class_to_classes(&self.play_thresholds.per_class, &from, &to_classes)
            }
            _ => BTreeMap::new(),
        };
    }

    /// Reconciles hover, batch QA, and threshold cache state after any per-class threshold edit.
    fn after_play_thresholds_changed(&mut self, cx: &mut Context<Self>) {
        let per_b = self.play_per_class_b();
        self.cancel_play_batch();
        self.play_batch.invalidate_for_thresholds(self.play_conf, &self.play_thresholds.per_class);
        self.clear_play_hover();
        self.play_compare.clear_hover();
        let switched = self.reconcile_play_batch_active(cx);
        if !switched {
            if !thresholds_match(
                &self.play_fetched_per_class,
                self.play_fetched_floor,
                self.play_conf,
                &self.play_thresholds.per_class,
            ) {
                self.run_predict(cx);
            }
            if self.play_compare.enabled
                && !thresholds_match(
                    &self.play_compare.fetched_per_class_b,
                    self.play_compare.fetched_floor_b,
                    self.play_conf,
                    &per_b,
                )
            {
                self.run_predict_b(cx);
            }
        }
        cx.notify();
    }

    /// Sets a per-class confidence override for `class_id` (clamped to `[0, 1]`).
    pub fn set_play_class_conf(&mut self, class_id: u32, conf: f32, cx: &mut Context<Self>) {
        let val = round_conf(conf);
        let unchanged =
            self.play_thresholds.per_class.get(&class_id).is_some_and(|&v| (v - val).abs() <= 1e-6);
        self.play_thresholds.clear_feedback();
        if unchanged {
            cx.notify();
            return;
        }
        self.play_thresholds.per_class.insert(class_id, val);
        self.after_play_thresholds_changed(cx);
    }

    /// Removes the per-class override for `class_id` so it inherits global `play_conf`.
    pub fn clear_play_class_conf(&mut self, class_id: u32, cx: &mut Context<Self>) {
        self.play_thresholds.clear_feedback();
        if self.play_thresholds.per_class.remove(&class_id).is_some() {
            self.after_play_thresholds_changed(cx);
        } else {
            cx.notify();
        }
    }

    /// Clears all per-class confidence overrides so every class inherits global `play_conf`.
    pub fn reset_all_play_class_conf(&mut self, cx: &mut Context<Self>) {
        self.play_thresholds.clear_feedback();
        if !self.play_thresholds.per_class.is_empty() {
            self.play_thresholds.per_class.clear();
            self.after_play_thresholds_changed(cx);
        } else {
            cx.notify();
        }
    }

    /// Parses and applies `ThresholdCalibration` or `{ class: conf }` JSON into Playground.
    pub fn apply_play_thresholds_json(&mut self, raw: &str, cx: &mut Context<Self>) -> bool {
        self.play_thresh_import_seq += 1;
        let classes: Vec<String> = self.play_shown_model().map(|m| m.classes.clone()).unwrap_or_default();
        match parse_thresholds_json(raw, &classes) {
            Ok(parsed) => {
                self.play_thresholds.import_error = None;
                self.play_thresholds.import_open = false;
                if let Some(g) = parsed.global_conf {
                    self.play_conf = g;
                }
                self.play_thresholds.per_class = parsed.per_class;
                let mut parts = Vec::new();
                if let Some(g) = parsed.global_conf {
                    parts.push(format!("global {}", fmt_threshold(g)));
                }
                let n = self.play_thresholds.per_class.len();
                parts.push(counted(n as u64, "class override", "class overrides"));
                let summary = if let Some(w) = parsed.warnings.first() {
                    format!("Applied {} ({w})", parts.join(", "))
                } else {
                    format!("Applied {}.", parts.join(", "))
                };
                self.play_thresholds.import_note = Some(summary.clone());
                self.play_thresh_json.update(cx, |i, cx| i.set_text("", cx));
                self.toast(Tone::Success, "Thresholds imported", Some(summary.into()), cx);
                self.after_play_thresholds_changed(cx);
                true
            }
            Err(err) => {
                self.play_thresholds.import_error = Some(err);
                self.play_thresholds.import_note = None;
                cx.notify();
                false
            }
        }
    }

    /// Prompts for a `.json` threshold file on disk and imports it into Playground.
    pub fn pick_play_thresholds_file(&mut self, window: &Window, cx: &mut Context<Self>) {
        self.play_thresh_import_seq += 1;
        let seq = self.play_thresh_import_seq;
        let backend_gen = self.backend_gen;
        let active_model = self.play_model.clone();
        let handle = window.window_handle();
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Load thresholds JSON".into()),
        });
        cx.spawn(async move |this, cx| {
            let Some(paths) = picked_paths(&this, rx.await, cx) else { return };
            let Some(path) = paths.into_iter().next() else { return };
            let read = cx
                .background_executor()
                .spawn(async move {
                    let meta = std::fs::metadata(&path)
                        .map_err(|e| format!("Could not read {}: {e}", path.display()))?;
                    if meta.len() > MAX_THRESHOLDS_JSON_BYTES {
                        return Err(format!(
                            "File is too large ({} bytes, max {MAX_THRESHOLDS_JSON_BYTES}).",
                            meta.len()
                        ));
                    }
                    std::fs::read_to_string(&path)
                        .map_err(|e| format!("Could not read {}: {e}", path.display()))
                })
                .await;
            handle
                .update(cx, |_, window, cx| {
                    this.update(cx, |this, cx| {
                        if this.play_thresh_import_seq != seq
                            || this.backend_gen != backend_gen
                            || this.play_model != active_model
                        {
                            return;
                        }
                        match read {
                            Ok(text) => {
                                if this.apply_play_thresholds_json(&text, cx) {
                                    this.focus_key("play-thr-import", window, cx);
                                }
                            }
                            Err(msg) => {
                                this.play_thresholds.import_error = Some(msg);
                                this.play_thresholds.import_note = None;
                                cx.notify();
                            }
                        }
                    })
                })
                .ok();
        })
        .detach();
    }

    /// Builds `ThresholdCalibration`-compatible JSON from the current Playground thresholds.
    pub(crate) fn build_play_thresholds_export(&self) -> anyhow::Result<(String, String, String)> {
        let shown = self.play_shown_model();
        let classes = shown.map(|m| m.classes.as_slice()).unwrap_or(&[]);
        let model_name =
            shown.map(|m| m.name.as_str()).or(self.play_model.as_deref()).unwrap_or("playground");
        let json = export_thresholds_json(self.play_conf, &self.play_thresholds.per_class, classes)?;
        let fname = export_filename("thresholds", model_name, None, "json");
        let n = self.play_thresholds.per_class.len();
        let summary = format!(
            "global {} · {}",
            fmt_threshold(self.play_conf),
            counted(n as u64, "class override", "class overrides")
        );
        Ok((json, fname, summary))
    }

    /// Copies the current Playground thresholds JSON to the clipboard.
    pub fn copy_play_thresholds_json(&mut self, cx: &mut Context<Self>) {
        match self.build_play_thresholds_export() {
            Ok((json, _, summary)) => {
                cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(json));
                self.toast(Tone::Success, "Copied thresholds JSON", Some(summary.into()), cx);
            }
            Err(e) => {
                self.toast(Tone::Danger, "Threshold export failed", Some(e.to_string().into()), cx);
            }
        }
    }

    /// Prompts for a save path and writes the current Playground thresholds JSON to disk.
    pub fn export_play_thresholds_json(&mut self, cx: &mut Context<Self>) {
        let (json, suggested, summary) = match self.build_play_thresholds_export() {
            Ok(v) => v,
            Err(e) => {
                self.toast(Tone::Danger, "Threshold export failed", Some(e.to_string().into()), cx);
                return;
            }
        };
        let dir = export_default_dir();
        let rx = cx.prompt_for_new_path(&dir, Some(&suggested));
        cx.spawn(async move |this, cx| {
            let Some(path) = picked_save(&this, rx.await, cx) else { return };
            let write_path = path.clone();
            let res = cx
                .background_executor()
                .spawn(async move { write_export_bytes(&write_path, json.as_bytes()) })
                .await;
            this.update(cx, |this, cx| match res {
                Ok(()) => {
                    let reveal = path.clone();
                    let shown = display_path(&path.to_string_lossy());
                    this.toast_with(
                        Tone::Success,
                        "Exported thresholds JSON",
                        Some(format!("{shown} · {summary}").into()),
                        Some(("Show in folder".into(), act(move |_, _, cx| cx.reveal_path(&reveal)))),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(Tone::Danger, "Threshold export failed", Some(e.to_string().into()), cx);
                }
            })
            .ok();
        })
        .detach();
    }

    /// Opens `model_id` in Playground; when `apply_calibrated` is true, carries calibrated
    /// per-class thresholds (if computed via `POST /api/val`) or the model/evaluation's optimal
    /// `best_f1_conf` operating point cleanly into Playground without leaking prior overrides.
    pub fn open_model_in_playground(
        &mut self,
        model_id: String,
        apply_calibrated: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.play_pred = Load::Idle;
        self.navigate(Route::Playground, window, cx);
        self.select_play_model(model_id.clone());
        if apply_calibrated {
            let classes = self.store.model(&model_id).map(|m| m.classes.clone()).unwrap_or_default();
            if let Some(cal) = self
                .store
                .calibrations
                .get(&model_id)
                .and_then(|l| l.ready())
                .filter(|c| !c.is_empty())
                .cloned()
            {
                let (global, per_class) = calibration_to_overrides(&cal, &classes);
                if let Some(g) = global {
                    self.play_conf = g;
                }
                self.play_thresholds.per_class = per_class;
                self.play_thresholds.expanded = !self.play_thresholds.per_class.is_empty();
                let note = format!(
                    "Loaded calibrated thresholds (global {}, {}).",
                    fmt_threshold(self.play_conf),
                    counted(self.play_thresholds.per_class.len() as u64, "class override", "class overrides"),
                );
                self.play_thresholds.import_error = None;
                self.play_thresholds.import_note = Some(note.clone());
                self.toast(Tone::Success, "Opened with calibrated thresholds", Some(note.into()), cx);
            } else {
                // Fallback: carry the model or evaluation's best-F1 operating point when valid.
                let ev_conf = self
                    .store
                    .evals
                    .get(&model_id)
                    .and_then(|l| l.ready())
                    .and_then(|ev| ev.best_f1_conf)
                    .filter(|c| c.is_finite() && (0.0..=1.0).contains(c));
                let model_conf = self
                    .store
                    .model(&model_id)
                    .and_then(|m| m.metrics.as_ref())
                    .and_then(|mm| operating_conf(mm.best_f1_conf, mm.precision, mm.recall));
                if let Some(best) = model_conf.or(ev_conf) {
                    self.play_conf = round_conf(best);
                    self.play_thresholds.per_class.clear();
                    let note = format!("Applied optimal F1 threshold ({}).", fmt_threshold(self.play_conf));
                    self.play_thresholds.import_error = None;
                    self.play_thresholds.import_note = Some(note.clone());
                    self.toast(Tone::Success, "Opened at best F1 threshold", Some(note.into()), cx);
                }
            }
            self.cancel_play_batch();
            self.play_batch.invalidate_for_thresholds(self.play_conf, &self.play_thresholds.per_class);
            self.clear_play_hover();
            self.play_compare.clear_hover();
        }
        let switched = self.reconcile_play_batch_active(cx);
        if !switched {
            self.run_predict(cx);
            if self.play_compare.enabled {
                self.run_predict_b(cx);
            }
        }
        cx.notify();
    }

    /// Runs `POST /api/val` to compute calibrated per-class confidence thresholds for `model_id`.
    pub fn calibrate_model_thresholds(
        &mut self,
        model_id: String,
        dataset_id: String,
        split: crate::api::Split,
        cx: &mut Context<Self>,
    ) {
        if self.hold_loading || matches!(self.store.calibrations.get(&model_id), Some(Load::Loading)) {
            return;
        }
        self.store.calibrations.insert(model_id.clone(), Load::Loading);
        let mid = model_id.clone();
        self.fetch(
            cx,
            move |api| api.val_thresholds(&mid, &dataset_id, split),
            move |this, r, cx| {
                if this.store.model(&model_id).is_none()
                    || !matches!(this.store.calibrations.get(&model_id), Some(Load::Loading))
                {
                    return;
                }
                match r {
                    Ok(Some(cal)) => {
                        let empty = cal.is_empty();
                        let summary = format!(
                            "Global {} · macro F1 {:.3} · {}",
                            fmt_threshold(cal.global.best_conf),
                            cal.macro_f1,
                            counted(cal.per_class.len() as u64, "class", "classes"),
                        );
                        this.store.calibrations.insert(model_id, Load::Ready(cal));
                        if empty {
                            this.toast(
                                Tone::Neutral,
                                "No ground-truth boxes to calibrate",
                                Some(
                                    "This split has no labeled objects to derive per-class thresholds."
                                        .into(),
                                ),
                                cx,
                            );
                        } else {
                            this.toast(Tone::Success, "Thresholds calibrated", Some(summary.into()), cx);
                        }
                    }
                    Ok(None) => {
                        let err = crate::api::ApiError::new(
                            "not_supported",
                            "This server did not return calibrated thresholds from /api/val.",
                        );
                        this.store.calibrations.insert(model_id, Load::Failed(err));
                    }
                    Err(err) => {
                        this.store.calibrations.insert(model_id, Load::Failed(err));
                    }
                }
            },
        );
        cx.notify();
    }

    /// Copies a model's calibrated `ThresholdCalibration` JSON to the clipboard.
    pub fn copy_model_calibration_json(&mut self, model_id: &str, cx: &mut Context<Self>) {
        let Some(cal) =
            self.store.calibrations.get(model_id).and_then(|l| l.ready()).filter(|c| !c.is_empty()).cloned()
        else {
            return;
        };
        let classes = self.store.model(model_id).map(|m| m.classes.as_slice()).unwrap_or(&[]);
        match export_calibration_json(&cal, classes) {
            Ok(json) => {
                cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(json));
                let note = format!(
                    "global {} · {}",
                    fmt_threshold(cal.global.best_conf),
                    counted(cal.per_class.len() as u64, "class", "classes"),
                );
                self.toast(Tone::Success, "Copied calibrated thresholds JSON", Some(note.into()), cx);
            }
            Err(e) => {
                self.toast(Tone::Danger, "Threshold copy failed", Some(e.to_string().into()), cx);
            }
        }
    }
}
