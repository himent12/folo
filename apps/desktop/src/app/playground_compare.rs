//! Playground Model A/B comparison lifecycle, cancellation, freshness checks, and annotated
//! PNG / JSON file export.

use std::path::PathBuf;

use folo_ui::Context;

use crate::api::*;
use crate::screens::playground::{
    ComparePaneCaption, CompareSideExport, ExportPalette, PlayPredMeta, compare_detections_json,
    compare_pane_caption_parts, effective_conf, export_filename, filter_detections_with_per_class,
    pick_default_model_b, render_annotated_compare_png, render_annotated_single_png, render_image_to_rgba,
    summarize_comparison, write_export_bytes,
};
use crate::store::Load;
use crate::theme::theme;
use crate::ui::*;

use super::{FoloApp, PLAY_CONF_FLOOR, Route, act};

impl FoloApp {
    /// True when Model A's `Load::Ready` prediction matches the current image, model, device, IoU,
    /// and confidence floor (and is not superseded by an in-flight rerun).
    pub(crate) fn pred_a_is_fresh(&self) -> bool {
        if self.play_pred_pending || self.play_pred.ready().is_none() {
            return false;
        }
        let Some(model_id) = self.play_model.as_deref() else { return false };
        let e2e = self.play_shown_model().is_some_and(|m| self.model_e2e(m));
        match &self.play_compare.pred_a_meta {
            Some(meta) => meta.matches_current(
                self.play_image_gen,
                model_id,
                self.play_iou,
                self.play_target,
                self.play_conf,
                &self.play_thresholds.per_class,
                e2e,
            ),
            // Legacy/test direct `Load::Ready` injection without meta: still check `play_pred_for` if set.
            None => {
                self.play_pred_for.as_ref().is_none_or(|(g, m)| *g == self.play_image_gen && m == model_id)
            }
        }
    }

    /// True when Model B's `Load::Ready` prediction matches the current image, model, device, IoU,
    /// and threshold configuration (and is not superseded by an in-flight rerun).
    pub(crate) fn pred_b_is_fresh(&self) -> bool {
        if self.play_compare.pred_b_pending || self.play_compare.pred_b.ready().is_none() {
            return false;
        }
        let Some(model_id) = self.play_compare.model_b.as_deref() else { return false };
        let e2e = self.play_shown_model_b().is_some_and(|m| self.model_e2e(m));
        let per_b = self.play_per_class_b();
        self.play_compare.pred_b_meta.as_ref().is_some_and(|meta| {
            meta.matches_current(
                self.play_image_gen,
                model_id,
                self.play_iou,
                self.play_target,
                self.play_conf,
                &per_b,
                e2e,
            )
        })
    }

    /// Cheap predicate for render frames: true when at least one fresh prediction can be exported as JSON.
    pub(crate) fn has_play_export_json(&self) -> bool {
        if self.play_image.is_none() || self.play_stage_excluded_by_qa() {
            return false;
        }
        if !self.play_compare.enabled {
            self.pred_a_is_fresh()
        } else {
            self.pred_a_is_fresh() || self.pred_b_is_fresh()
        }
    }

    /// Enables or disables Compare A/B mode in Playground.
    pub fn set_play_compare(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.play_compare.enabled == enabled {
            return;
        }
        self.cancel_play_batch();
        self.play_compare.enabled = enabled;
        self.play_drag = None;
        self.play_compare.clear_hover();
        if enabled {
            // Compare rail doesn't expose Single-mode class filter chips; clear so detections are never hidden.
            let cleared_class = self.play_class.take().is_some();
            self.clear_play_hover();
            self.ensure_play_model_b();
            let switched = cleared_class && self.reconcile_play_batch_active(cx);
            if !switched && self.play_image.is_some() && !self.pred_b_is_fresh() {
                self.run_predict_b(cx);
            }
        } else {
            // Cancel any in-flight Model B inference (both async response sequence and queued local ticket).
            self.play_compare.pred_b_seq += 1;
            let _ = crate::local::Ticket::take(&self.play_compare.line_b);
            self.play_compare.pred_b_pending = false;
            self.play_compare.pred_b_dim = false;
            self.play_compare.model_b_gone = None;
            // Restore fetched_floor_b to the last completed Model B prediction so cancelling a
            // lowered-confidence rerun cannot mask a later refetch.
            self.play_compare.fetched_floor_b =
                self.play_compare.pred_b_meta.as_ref().map_or(PLAY_CONF_FLOOR, |m| m.floor);
            self.play_compare.class_filter = crate::screens::playground::CompareClassFilter::All;
        }
        cx.notify();
    }

    /// Picks a default Model B if none is currently selected (or the selected one is not in the catalog).
    pub(crate) fn ensure_play_model_b(&mut self) {
        let Some(models) = self.store.models.ready() else { return };
        if let Some(id) = self.play_compare.model_b.as_deref()
            && models.iter().any(|m| m.id == id)
        {
            self.play_compare.model_b_gone = None;
            return;
        }
        let model_a = self.play_shown_model();
        if let Some(m) = pick_default_model_b(model_a, models) {
            self.play_compare.model_b = Some(m.id.clone());
            self.play_compare.model_b_run = m.run_id.clone();
            self.play_compare.model_b_gone = None;
        }
    }

    /// Selects Model B for Compare mode and runs inference on the current image if Compare is active.
    pub fn pick_play_model_b(&mut self, id: String, cx: &mut Context<Self>) {
        if self.play_compare.model_b.as_deref() != Some(id.as_str()) {
            self.play_compare.clear_hover();
        }
        self.play_compare.model_b_run = self.store.model(&id).and_then(|m| m.run_id.clone());
        self.play_compare.model_b = Some(id);
        self.play_compare.model_b_gone = None;
        if self.play_compare.enabled {
            self.run_predict_b(cx);
        }
        cx.notify();
    }

    /// Swaps Model A and Model B (along with their already-fetched predictions and metadata).
    pub fn swap_play_models(&mut self, cx: &mut Context<Self>) {
        if !self.play_compare.enabled {
            return;
        }
        self.ensure_play_model_b();
        if self.play_model == self.play_compare.model_b {
            return;
        }
        // Cancel any in-flight requests and local tickets so pre-swap responses never land in the wrong slot.
        self.invalidate_play_batch();
        self.play_pred_seq += 1;
        self.play_compare.pred_b_seq += 1;
        let _ = crate::local::Ticket::take(&self.play_line);
        let _ = crate::local::Ticket::take(&self.play_compare.line_b);
        let a_pending = std::mem::replace(&mut self.play_pred_pending, false);
        let b_pending = std::mem::replace(&mut self.play_compare.pred_b_pending, false);

        let prev_a_classes = self.play_shown_model().map(|m| m.classes.clone());
        let next_a_id = self.play_compare.model_b.clone();
        std::mem::swap(&mut self.play_model, &mut self.play_compare.model_b);
        std::mem::swap(&mut self.play_model_run, &mut self.play_compare.model_b_run);
        std::mem::swap(&mut self.play_pred, &mut self.play_compare.pred_b);
        std::mem::swap(&mut self.play_pred_for, &mut self.play_compare.pred_b_for);
        std::mem::swap(&mut self.play_compare.pred_a_meta, &mut self.play_compare.pred_b_meta);
        std::mem::swap(&mut self.play_pred_dim, &mut self.play_compare.pred_b_dim);
        std::mem::swap(&mut self.play_fetched_floor, &mut self.play_compare.fetched_floor_b);
        self.remap_play_thresholds_for_model(prev_a_classes, next_a_id.as_deref());
        self.play_compare.class_filter = match self.play_compare.class_filter {
            crate::screens::playground::CompareClassFilter::OnlyA => {
                crate::screens::playground::CompareClassFilter::OnlyB
            }
            crate::screens::playground::CompareClassFilter::OnlyB => {
                crate::screens::playground::CompareClassFilter::OnlyA
            }
            other => other,
        };
        self.clear_play_hover();
        self.play_compare.clear_hover();
        self.play_class = None;

        if b_pending || (self.play_image.is_some() && matches!(self.play_pred, Load::Loading)) {
            self.run_predict(cx);
        }
        if a_pending || (self.play_image.is_some() && matches!(self.play_compare.pred_b, Load::Loading)) {
            self.run_predict_b(cx);
        }
        cx.notify();
    }

    /// The Model B currently selected in Playground.
    pub fn play_shown_model_b(&self) -> Option<&Model> {
        let id = self.play_compare.model_b.as_ref()?;
        self.store.models.ready()?.iter().find(|m| m.id == *id)
    }

    /// Whether Model B was produced by `run`.
    pub(super) fn play_model_b_made_by(&self, run: &str, made: &[String]) -> bool {
        self.play_compare.model_b.as_ref().is_some_and(|m| made.contains(m))
            || self.play_compare.model_b_run.as_deref() == Some(run)
    }

    /// Drops Model B and cancels its in-flight prediction (when deleted locally or externally).
    pub(super) fn drop_play_model_b(&mut self, name: Option<String>) {
        self.play_compare.model_b = None;
        self.play_compare.model_b_run = None;
        self.play_compare.model_b_gone = self.play_compare.enabled.then_some(name).flatten();
        self.play_compare.pred_b_seq += 1;
        let _ = crate::local::Ticket::take(&self.play_compare.line_b);
        self.play_compare.pred_b = Load::Idle;
        self.play_compare.pred_b_for = None;
        self.play_compare.pred_b_meta = None;
        self.play_compare.pred_b_pending = false;
        self.play_compare.pred_b_dim = false;
        self.play_compare.clear_hover();
    }

    /// Reconciles Model B against the loaded model list (called right after `reconcile_play_model`).
    pub(super) fn reconcile_play_model_b(&mut self, cx: &mut Context<Self>) {
        let missing_id = {
            let Some(models) = self.store.models.ready() else { return };
            self.play_compare.model_b.clone().filter(|id| !models.iter().any(|m| m.id == *id))
        };
        if let Some(id) = missing_id {
            let name = self.deleted.get(&id).cloned().unwrap_or(id);
            self.drop_play_model_b(Some(name));
        }
        if !self.play_compare.enabled {
            self.play_compare.model_b_gone = None;
            return;
        }
        if self.play_compare.model_b.is_some() {
            return;
        }
        let next = self.store.models.ready().and_then(|models| {
            pick_default_model_b(self.play_shown_model(), models)
                .map(|m| (m.id.clone(), m.name.clone(), m.run_id.clone()))
        });
        let in_use = self.play_image.is_some() || self.route == Route::Playground;
        if let Some(gone) = self.play_compare.model_b_gone.take().filter(|_| in_use) {
            let body = match &next {
                Some((_, name, _)) => format!("Model B now uses {name}."),
                None => "No other model is left.".to_string(),
            };
            self.toast(Tone::Neutral, format!("{gone} was deleted"), Some(body.into()), cx);
        }
        if let Some((id, _, run)) = next {
            self.play_compare.model_b = Some(id);
            self.play_compare.model_b_run = run;
            self.run_predict_b(cx);
        }
    }

    /// Clears Model B's prediction when the staged image changes or disappears.
    pub(super) fn clear_play_pred_b(&mut self) {
        self.play_compare.pred_b_seq += 1;
        let _ = crate::local::Ticket::take(&self.play_compare.line_b);
        self.play_compare.pred_b = Load::Idle;
        self.play_compare.pred_b_pending = false;
        self.play_compare.pred_b_dim = false;
        self.play_compare.pred_b_for = None;
        self.play_compare.pred_b_meta = None;
        self.play_compare.clear_hover();
    }

    /// Reruns Model B from scratch (showing Loading meanwhile).
    pub fn rerun_predict_b(&mut self, cx: &mut Context<Self>) {
        self.play_compare.pred_b_for = None;
        self.play_compare.pred_b_meta = None;
        self.play_compare.pred_b_dim = false;
        self.run_predict_b(cx);
    }

    /// Runs real inference for Model B (server or local GPU/CPU) with independent sequence cancellation.
    pub fn run_predict_b(&mut self, cx: &mut Context<Self>) {
        if !self.play_compare.enabled {
            return;
        }
        self.ensure_play_model_b();
        let (Some(model), Some(img)) = (self.play_compare.model_b.clone(), self.play_image.as_ref()) else {
            return;
        };
        let per_b = self.play_per_class_b();
        let has_overrides = !per_b.is_empty();
        let floor = if has_overrides { self.play_conf } else { self.play_conf.min(PLAY_CONF_FLOOR) };
        let req = PredictRequest {
            model_id: model.clone(),
            input: img.predict_input(self.play_target.is_some()),
            conf: floor,
            iou: self.play_iou,
            max_det: 300,
            per_class_conf: has_overrides.then(|| per_b.clone()),
            conf_inclusive: has_overrides,
        };
        self.play_compare.fetched_floor_b = floor;
        self.play_compare.fetched_per_class_b = per_b.clone();
        let key = (self.play_image_gen, model.clone());
        let meta = PlayPredMeta {
            image_gen: self.play_image_gen,
            model_id: model,
            iou_bits: self.play_iou.to_bits(),
            target: self.play_target,
            floor,
            per_class_conf: per_b,
        };
        if self.play_compare.pred_b_for.as_ref() != Some(&key) {
            self.play_compare.pred_b = Load::Loading;
            self.play_compare.pred_b_meta = None;
            self.play_compare.clear_hover();
        }
        self.play_compare.pred_b_seq += 1;
        self.play_compare.pred_b_pending = true;
        let seq = self.play_compare.pred_b_seq;
        let target = self.play_target;
        let generation = crate::local::generation();
        let ticket = crate::local::Ticket::take(&self.play_compare.line_b);
        self.fetch(
            cx,
            move |api| match target {
                Some(t) => crate::local::predict(api, &req, t, generation, &ticket),
                None => api.predict(&req),
            },
            move |this, r, _| this.play_pred_b_answered_with_meta(seq, key, Some(meta), r),
        );
    }

    /// Applies a Model B prediction response only if it matches the latest `pred_b_seq`.
    #[cfg(test)]
    pub(super) fn play_pred_b_answered(
        &mut self,
        seq: u64,
        key: (u64, String),
        r: ApiResult<PredictResponse>,
    ) {
        self.play_pred_b_answered_with_meta(seq, key, None, r);
    }

    fn play_pred_b_answered_with_meta(
        &mut self,
        seq: u64,
        key: (u64, String),
        meta: Option<PlayPredMeta>,
        r: ApiResult<PredictResponse>,
    ) {
        // Drop stale/cancelled responses before mutating reachability or UI state.
        if seq != self.play_compare.pred_b_seq {
            return;
        }
        if let Err(e) = &r
            && e.code == "server_unreachable"
        {
            self.store.note_reachability(Some(e));
        }
        let r = r.map(|mut p| {
            for b in &mut p.boxes {
                if b.name.trim().is_empty() {
                    b.name = format!("class {}", b.class);
                }
            }
            p
        });
        self.play_compare.pred_b_meta = if r.is_ok() { meta } else { None };
        self.play_compare.pred_b = r.into();
        self.play_compare.pred_b_for = Some(key);
        self.play_compare.pred_b_pending = false;
        self.play_compare.pred_b_dim = false;
        self.play_compare.clear_hover();
    }

    /// Builds the current Playground JSON export payload (Single or Compare mode) using only fresh
    /// predictions whose completed parameters match the active image/model/device/IoU/floor.
    pub(crate) fn build_play_export_json(&self) -> Option<anyhow::Result<(String, String, String)>> {
        if self.play_stage_excluded_by_qa() {
            return None;
        }
        let img = self.play_image.as_ref()?;
        let image_name = img.name.as_ref();
        let shown_a = self.play_shown_model();
        let model_a_name =
            shown_a.map(|m| m.name.as_str()).or(self.play_model.as_deref()).unwrap_or("model-a");
        let e2e_a = shown_a.is_some_and(|m| self.model_e2e(m));
        let eff_class = self.play_effective_class();

        if !self.play_compare.enabled {
            if !self.pred_a_is_fresh() {
                return None;
            }
            let pred_a = self.play_pred.ready()?;
            let vis = filter_detections_with_per_class(
                &pred_a.boxes,
                self.play_conf,
                &self.play_thresholds.per_class,
                eff_class,
            );
            let iou = (!e2e_a).then_some(self.play_iou);
            let json = crate::screens::playground::detections_json(
                model_a_name,
                image_name,
                self.play_conf,
                Some(&self.play_thresholds.per_class),
                iou,
                eff_class,
                &vis,
            );
            let fname = export_filename(image_name, model_a_name, None, "json");
            let summary = plural(vis.len(), "detection");
            return Some(json.map(|s| (s, fname, summary)));
        }

        let pred_a = self.pred_a_is_fresh().then(|| self.play_pred.ready()).flatten();
        let pred_b = self.pred_b_is_fresh().then(|| self.play_compare.pred_b.ready()).flatten();
        if pred_a.is_none() && pred_b.is_none() {
            return None;
        }
        let shown_b = self.play_shown_model_b();
        let model_b_name =
            shown_b.map(|m| m.name.as_str()).or(self.play_compare.model_b.as_deref()).unwrap_or("model-b");
        let e2e_b = shown_b.is_some_and(|m| self.model_e2e(m));
        let vis_a = pred_a.map(|p| {
            filter_detections_with_per_class(
                &p.boxes,
                self.play_conf,
                &self.play_thresholds.per_class,
                eff_class,
            )
        });
        let vis_b = pred_b.map(|p| self.filter_detections_b(&p.boxes));
        let per_class_b = self.play_per_class_b();
        let cmp = summarize_comparison(vis_a.as_deref(), vis_b.as_deref(), pred_a, pred_b);
        let side_a = pred_a.zip(vis_a.as_ref()).map(|(p, v)| CompareSideExport {
            model: model_a_name,
            model_id: self.play_model.as_deref().unwrap_or(""),
            iou: (!e2e_a).then_some(self.play_iou),
            per_class_conf: (!self.play_thresholds.per_class.is_empty())
                .then_some(&self.play_thresholds.per_class),
            pred: p,
            boxes: v,
        });
        let side_b = pred_b.zip(vis_b.as_ref()).map(|(p, v)| CompareSideExport {
            model: model_b_name,
            model_id: self.play_compare.model_b.as_deref().unwrap_or(""),
            iou: (!e2e_b).then_some(self.play_iou),
            per_class_conf: (!per_class_b.is_empty()).then_some(&per_class_b),
            pred: p,
            boxes: v,
        });
        let top_iou = (!(e2e_a && e2e_b)).then_some(self.play_iou);
        let json = compare_detections_json(
            image_name,
            (img.width, img.height),
            self.play_conf,
            Some(&self.play_thresholds.per_class),
            top_iou,
            eff_class,
            &cmp,
            side_a.as_ref(),
            side_b.as_ref(),
        );
        let fname = export_filename(image_name, model_a_name, Some(model_b_name), "json");
        let summary = if cmp.count_a.is_some() && cmp.count_b.is_some() {
            format!(
                "A {} · B {} · {} both, {} only A, {} only B",
                cmp.count_a.unwrap_or(0),
                cmp.count_b.unwrap_or(0),
                cmp.shared_classes.len(),
                cmp.only_a_classes.len(),
                cmp.only_b_classes.len(),
            )
        } else {
            format!(
                "A {} · B {}",
                cmp.count_a.map_or("—".to_string(), |n| n.to_string()),
                cmp.count_b.map_or("—".to_string(), |n| n.to_string()),
            )
        };
        Some(json.map(|s| (s, fname, summary)))
    }

    /// Filters Model B detections by effective per-class confidence (mapped by class name from
    /// Model A) and active effective class filter.
    pub(crate) fn filter_detections_b<'a>(&self, boxes: &'a [ScoredBox]) -> Vec<&'a ScoredBox> {
        let per_b = self.play_per_class_b();
        let Some(cid) = self.play_effective_class() else {
            return boxes
                .iter()
                .filter(|b| b.score >= effective_conf(self.play_conf, &per_b, b.class))
                .collect();
        };
        let target_name = self.play_class_name(cid);
        boxes
            .iter()
            .filter(|b| {
                b.score >= effective_conf(self.play_conf, &per_b, b.class)
                    && match target_name.as_deref() {
                        Some(name) => b.name == name,
                        None => b.class == cid,
                    }
            })
            .collect()
    }

    /// Prompts for a save path and writes the formatted detections JSON file (Single or Compare).
    pub fn export_play_json(&mut self, cx: &mut Context<Self>) {
        let Some(built) = self.build_play_export_json() else {
            self.toast(
                Tone::Neutral,
                "Nothing to export yet",
                Some("Wait for the current prediction to finish first.".into()),
                cx,
            );
            return;
        };
        let (json, suggested, summary) = match built {
            Ok(v) => v,
            Err(e) => {
                self.toast(Tone::Danger, "JSON export failed", Some(e.to_string().into()), cx);
                return;
            }
        };
        let title =
            if self.play_compare.enabled { "Exported A/B compare JSON" } else { "Exported detections JSON" };
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
                        title,
                        Some(format!("{shown} · {summary}").into()),
                        Some(("Show in folder".into(), act(move |_, _, cx| cx.reveal_path(&reveal)))),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(Tone::Danger, "JSON export failed", Some(e.to_string().into()), cx);
                }
            })
            .ok();
        })
        .detach();
    }

    /// Prompts for a save path and renders + writes an annotated PNG (Single or Compare A/B) off
    /// the UI thread using only fresh predictions.
    pub fn export_play_png(&mut self, cx: &mut Context<Self>) {
        if self.play_stage_excluded_by_qa() {
            let body = if self.play_batch_qa_zero_match() {
                "No batch images match the active QA filter."
            } else {
                "Wait for the matching batch image to finish loading first."
            };
            self.toast(Tone::Neutral, "Nothing to export", Some(body.into()), cx);
            return;
        }
        let Some(img) = self.play_image.as_ref() else {
            self.toast(Tone::Neutral, "Nothing to export yet", Some("Open an image first.".into()), cx);
            return;
        };
        // Clone only the lightweight Arc<RenderImage>; BGRA->RGBA conversion runs off the UI thread after pick.
        let render_img = img.image.clone();
        let t = theme(cx);
        let palette = ExportPalette::from_theme(&t);
        let image_name = img.name.to_string();
        let shown_a = self.play_shown_model();
        let model_a_name = shown_a
            .map(|m| m.name.clone())
            .or_else(|| self.play_model.clone())
            .unwrap_or_else(|| "model-a".to_string());
        let show_boxes = self.play_boxes;
        let show_labels = self.play_labels;
        let compare = self.play_compare.enabled;
        let text_system = cx.text_system().clone();
        let fresh_a = self.pred_a_is_fresh().then(|| self.play_pred.ready()).flatten();
        let eff_class = self.play_effective_class();

        let (suggested, job): (String, PngExportJob) = if !compare {
            let boxes_a: Vec<ScoredBox> = fresh_a
                .map(|p| {
                    filter_detections_with_per_class(
                        &p.boxes,
                        self.play_conf,
                        &self.play_thresholds.per_class,
                        eff_class,
                    )
                    .into_iter()
                    .cloned()
                    .collect()
                })
                .unwrap_or_default();
            let fname = export_filename(&image_name, &model_a_name, None, "png");
            (
                fname,
                PngExportJob::Single {
                    image: render_img,
                    boxes: boxes_a,
                    show_boxes,
                    show_labels,
                    palette,
                    text_system,
                },
            )
        } else {
            let fresh_b = self.pred_b_is_fresh().then(|| self.play_compare.pred_b.ready()).flatten();
            let shown_b = self.play_shown_model_b();
            let model_b_name = shown_b
                .map(|m| m.name.clone())
                .or_else(|| self.play_compare.model_b.clone())
                .unwrap_or_else(|| "model-b".to_string());
            let boxes_a: Vec<ScoredBox> = fresh_a
                .map(|p| {
                    filter_detections_with_per_class(
                        &p.boxes,
                        self.play_conf,
                        &self.play_thresholds.per_class,
                        eff_class,
                    )
                    .into_iter()
                    .cloned()
                    .collect()
                })
                .unwrap_or_default();
            let boxes_b: Vec<ScoredBox> = fresh_b
                .map(|p| self.filter_detections_b(&p.boxes).into_iter().cloned().collect())
                .unwrap_or_default();
            let cap_a = compare_pane_caption_parts(
                "A",
                &model_a_name,
                fresh_a.map(|_| boxes_a.len()),
                fresh_a.map(|p| p.timing.total_ms),
                fresh_a.is_some_and(|p| p.cold),
            );
            let cap_b = compare_pane_caption_parts(
                "B",
                &model_b_name,
                fresh_b.map(|_| boxes_b.len()),
                fresh_b.map(|p| p.timing.total_ms),
                fresh_b.is_some_and(|p| p.cold),
            );
            let fname = export_filename(&image_name, &model_a_name, Some(&model_b_name), "png");
            (
                fname,
                PngExportJob::Compare {
                    image: render_img,
                    caption_a: cap_a,
                    boxes_a,
                    caption_b: cap_b,
                    boxes_b,
                    show_boxes,
                    show_labels,
                    palette,
                    text_system,
                },
            )
        };

        let dir = export_default_dir();
        let rx = cx.prompt_for_new_path(&dir, Some(&suggested));
        cx.spawn(async move |this, cx| {
            let Some(path) = picked_save(&this, rx.await, cx) else { return };
            let write_path = path.clone();
            let res = cx
                .background_executor()
                .spawn(async move {
                    let png = match job {
                        PngExportJob::Single {
                            image,
                            boxes,
                            show_boxes,
                            show_labels,
                            palette,
                            text_system,
                        } => {
                            let base = render_image_to_rgba(&image).ok_or_else(|| {
                                anyhow::anyhow!("Could not read pixels from the current image.")
                            })?;
                            render_annotated_single_png(
                                &base,
                                &boxes,
                                show_boxes,
                                show_labels,
                                &palette,
                                Some(&text_system),
                            )?
                        }
                        PngExportJob::Compare {
                            image,
                            caption_a,
                            boxes_a,
                            caption_b,
                            boxes_b,
                            show_boxes,
                            show_labels,
                            palette,
                            text_system,
                        } => {
                            let base = render_image_to_rgba(&image).ok_or_else(|| {
                                anyhow::anyhow!("Could not read pixels from the current image.")
                            })?;
                            render_annotated_compare_png(
                                &base,
                                &caption_a,
                                &boxes_a,
                                &caption_b,
                                &boxes_b,
                                show_boxes,
                                show_labels,
                                &palette,
                                Some(&text_system),
                            )?
                        }
                    };
                    let nbytes = png.len() as u64;
                    write_export_bytes(&write_path, &png)?;
                    Ok::<u64, anyhow::Error>(nbytes)
                })
                .await;
            this.update(cx, |this, cx| match res {
                Ok(nbytes) => {
                    let (v, u) = bytes(nbytes);
                    let reveal = path.clone();
                    let shown = display_path(&path.to_string_lossy());
                    this.toast_with(
                        Tone::Success,
                        "Exported annotated PNG",
                        Some(format!("{shown} · {v} {u}").into()),
                        Some(("Show in folder".into(), act(move |_, _, cx| cx.reveal_path(&reveal)))),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(Tone::Danger, "PNG export failed", Some(e.to_string().into()), cx);
                }
            })
            .ok();
        })
        .detach();
    }
}

enum PngExportJob {
    Single {
        image: std::sync::Arc<folo_ui::RenderImage>,
        boxes: Vec<ScoredBox>,
        show_boxes: bool,
        show_labels: bool,
        palette: ExportPalette,
        text_system: std::sync::Arc<folo_ui::TextSystem>,
    },
    Compare {
        image: std::sync::Arc<folo_ui::RenderImage>,
        caption_a: ComparePaneCaption,
        boxes_a: Vec<ScoredBox>,
        caption_b: ComparePaneCaption,
        boxes_b: Vec<ScoredBox>,
        show_boxes: bool,
        show_labels: bool,
        palette: ExportPalette,
        text_system: std::sync::Arc<folo_ui::TextSystem>,
    },
}

pub(crate) fn export_default_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_default()
}

pub(crate) fn picked_save(
    this: &folo_ui::WeakEntity<FoloApp>,
    answer: Result<anyhow::Result<Option<PathBuf>>, folo_ui::oneshot::Canceled>,
    cx: &mut folo_ui::AsyncApp,
) -> Option<PathBuf> {
    match answer {
        Ok(Ok(path)) => path,
        Ok(Err(e)) => {
            this.update(cx, |this, cx| {
                this.toast(Tone::Danger, "The save dialog didn't open", Some(e.to_string().into()), cx)
            })
            .ok();
            None
        }
        Err(_) => None,
    }
}
