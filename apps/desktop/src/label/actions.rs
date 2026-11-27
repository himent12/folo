//! Mutations, filtering, form validation, and keyboard review actions for Agent Labeling.

use folo_ui::Context;

use super::load::route_label;
use super::state::{LabelImageFilter, LabelJobFilter, SelectedBox};
use crate::api::*;
use crate::app::{FoloApp, Route};
use crate::ui::Tone;

/// Validation split shares offered when exporting a labeling job to a YOLO dataset.
pub const LABEL_VAL_FRACTIONS: [f32; 3] = [0.1, 0.2, 0.3];

/// Parses a comma- or newline-separated class list into trimmed non-empty names.
pub fn parse_class_list(raw: &str) -> Vec<String> {
    raw.split([',', '\n', '\r']).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

/// Validates the New Labeling Job fields before submitting to `/api/label/jobs`.
pub fn validate_new_job(name: &str, classes_raw: &str, instructions: &str) -> Result<Vec<String>, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("Enter a job name.".into());
    }
    if trimmed.chars().count() > 100 {
        return Err("Job name must be at most 100 characters.".into());
    }
    let classes = parse_class_list(classes_raw);
    if classes.is_empty() {
        return Err("Add at least one class (comma-separated, e.g. car, person).".into());
    }
    if classes.len() > 80 {
        return Err("A job can have at most 80 classes.".into());
    }
    let mut seen = std::collections::HashSet::new();
    for c in &classes {
        if c.chars().count() > 64 || c.contains(':') {
            return Err(format!("Invalid class name `{c}` (max 64 chars, no colons)."));
        }
        if !seen.insert(c.to_lowercase()) {
            return Err(format!("Duplicate class `{c}`."));
        }
    }
    if instructions.chars().count() > 4000 {
        return Err("Instructions must be at most 4000 characters.".into());
    }
    Ok(classes)
}

/// Classifies a job into the list toolbar's status filter buckets.
pub fn job_filter_bucket(job: &LabelJob) -> LabelJobFilter {
    if job.status.is_active() {
        LabelJobFilter::Active
    } else if job.run.as_ref().is_some_and(|r| r.outcome == Some(LabelOutcome::Failed)) {
        LabelJobFilter::Failed
    } else {
        LabelJobFilter::Idle
    }
}

/// Filters labeling jobs by status bucket and a case-insensitive substring query (name, id, or classes).
pub fn filter_label_jobs<'a>(jobs: &'a [LabelJob], filter: LabelJobFilter, query: &str) -> Vec<&'a LabelJob> {
    let q = query.trim().to_lowercase();
    jobs.iter()
        .filter(|j| filter == LabelJobFilter::All || job_filter_bucket(j) == filter)
        .filter(|j| {
            q.is_empty()
                || j.name.to_lowercase().contains(&q)
                || j.id.to_lowercase().contains(&q)
                || j.classes.iter().any(|c| c.to_lowercase().contains(&q))
        })
        .collect()
}

/// Filters a labeling job's images by status/review bucket and optional class index.
pub fn filter_label_images(
    images: &[LabelImage],
    filter: LabelImageFilter,
    class_filter: Option<u32>,
) -> Vec<&LabelImage> {
    images
        .iter()
        .filter(|im| match filter {
            LabelImageFilter::All => true,
            LabelImageFilter::ToReview => im.status == LabelImageStatus::Labeled && !im.reviewed,
            LabelImageFilter::Reviewed => im.status == LabelImageStatus::Labeled && im.reviewed,
            LabelImageFilter::Pending => {
                matches!(im.status, LabelImageStatus::Pending | LabelImageStatus::Labeling)
            }
            LabelImageFilter::Failed => im.status == LabelImageStatus::Failed,
        })
        .filter(|im| class_filter.is_none_or(|c| im.boxes.iter().any(|b| b.class == c)))
        .collect()
}

impl FoloApp {
    /// Returns the filtered images of `job_id` under the current status + class filters.
    pub fn current_filtered_label_images(&self, job_id: &str) -> Vec<LabelImage> {
        let Some(all) = self.label.images.get(job_id).and_then(|l| l.ready()) else {
            return Vec::new();
        };
        filter_label_images(all, self.label.img_filter, self.label.class_filter)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Ensures `label.selected_image` points to a valid image in the current filtered set (or None if empty).
    /// Keeps an image with unsaved local edits selected as long as it still exists in the job so a
    /// concurrent SSE status update never silently discards the user's draft.
    pub fn reconcile_selected_label_image(&mut self, job_id: &str) {
        if self.label.editor.is_dirty()
            && self.label.editor.job_id.as_deref() == Some(job_id)
            && let Some(dirty_id) = self.label.editor.image_id.as_deref()
            && self
                .label
                .images
                .get(job_id)
                .and_then(|l| l.ready())
                .is_some_and(|list| list.iter().any(|im| im.id == dirty_id))
        {
            self.label.selected_image = Some(dirty_id.to_string());
            return;
        }
        let shown = self.current_filtered_label_images(job_id);
        if shown.is_empty() {
            self.label.selected_image = None;
            self.label.selected_box = None;
            return;
        }
        let keep =
            self.label.selected_image.as_deref().is_some_and(|sel| shown.iter().any(|im| im.id == sel));
        if !keep {
            self.label.selected_image = Some(shown[0].id.clone());
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.zoom = 1.0;
            self.label.pan = (0., 0.);
        }
    }

    /// Selects a specific image in the review workspace, resetting box selection, draft, and zoom.
    pub fn select_label_image(&mut self, image_id: String, cx: &mut Context<Self>) {
        self.label.pending_advance = None;
        self.label.editor.pending_switch = None;
        if self.label.selected_image.as_deref() != Some(image_id.as_str()) {
            self.label.selected_image = Some(image_id);
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.zoom = 1.0;
            self.label.pan = (0., 0.);
            self.label.review_error = None;
            self.label.confirm_delete_image = None;
            self.label.editor.drawing = None;
            self.label.editor.transforming = None;
            self.label.editor.hover_handle = None;
            self.label.editor.hover_move = false;
            cx.notify();
        }
    }

    /// Steps through the filtered images (`delta = -1` prev, `+1` next).
    pub fn step_label_image(&mut self, job_id: &str, delta: i32, cx: &mut Context<Self>) {
        let shown = self.current_filtered_label_images(job_id);
        if shown.is_empty() {
            return;
        }
        let cur = self
            .label
            .selected_image
            .as_deref()
            .and_then(|id| shown.iter().position(|im| im.id == id))
            .unwrap_or(0);
        let next = ((cur as i32 + delta).rem_euclid(shown.len() as i32)) as usize;
        let next_id = shown[next].id.clone();
        self.select_label_image(next_id, cx);
    }

    /// Approves (marks reviewed) the current image, optionally advancing to the next image in view.
    ///
    /// If there are unsaved box edits in the local draft, persists those boxes with `reviewed: true`
    /// in a single PUT; otherwise marks the unchanged image reviewed (sending `expect` revision).
    pub fn approve_label_image(
        &mut self,
        job_id: &str,
        image_id: &str,
        advance: bool,
        cx: &mut Context<Self>,
    ) {
        if self.label.mutating {
            return;
        }
        let Some(im) = self.ensure_label_editor_synced(job_id, image_id) else {
            return;
        };
        if im.status == LabelImageStatus::Labeling {
            self.label.review_error =
                Some(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            cx.notify();
            return;
        }
        if self.label.editor.is_dirty_for(job_id, image_id) {
            self.save_label_edits(job_id, image_id, advance, cx);
            return;
        }
        self.label.editor.pending_switch = None;
        let next_after = if advance { self.next_image_after(job_id, image_id) } else { None };
        let expect = Some(label_box_revision(&im.boxes));
        let req = if im.status == LabelImageStatus::Labeled {
            UpdateLabelImage { boxes: None, reviewed: Some(!im.reviewed || advance), expect }
        } else {
            UpdateLabelImage { boxes: Some(im.boxes.clone()), reviewed: Some(true), expect }
        };
        self.send_label_image_update(job_id, image_id, req, next_after, None, false, cx);
    }

    pub(super) fn next_image_after(&self, job_id: &str, current_id: &str) -> Option<String> {
        let shown = self.current_filtered_label_images(job_id);
        let pos = shown.iter().position(|im| im.id == current_id)?;
        // Prefer the next image in view, else wrap to the previous one if this was last.
        shown.get(pos + 1).or_else(|| pos.checked_sub(1).and_then(|p| shown.get(p))).map(|im| im.id.clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn send_label_image_update(
        &mut self,
        job_id: &str,
        image_id: &str,
        req: UpdateLabelImage,
        advance_to: Option<String>,
        then_switch: Option<super::editor::PendingImageSwitch>,
        clear_box: bool,
        cx: &mut Context<Self>,
    ) {
        self.label.mutating = true;
        self.label.mutating_seq += 1;
        let mseq = self.label.mutating_seq;
        let lgen = self.label.epoch;
        let seen_img_seq =
            self.label.image_event_seq.get(&(job_id.to_string(), image_id.to_string())).copied().unwrap_or(0);
        self.label.pending_advance =
            advance_to.as_ref().map(|next| (job_id.to_string(), image_id.to_string(), next.clone()));
        self.label.review_error = None;
        let jid = job_id.to_string();
        let iid = image_id.to_string();
        let jid2 = jid.clone();
        let iid2 = iid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.update_label_image(&jid, &iid, &req),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                if this.label.mutating_seq == mseq {
                    this.label.mutating = false;
                }
                let on_job = route_label(&this.route) == Some(jid2.as_str());
                let still_pending_advance = advance_to.as_ref().is_some_and(|next| {
                    this.label.pending_advance.as_ref() == Some(&(jid2.clone(), iid2.clone(), next.clone()))
                });
                if still_pending_advance {
                    this.label.pending_advance = None;
                }
                match r {
                    Ok(updated) => {
                        let cur_img_seq = this
                            .label
                            .image_event_seq
                            .get(&(jid2.clone(), iid2.clone()))
                            .copied()
                            .unwrap_or(0);
                        let fresh = cur_img_seq == seen_img_seq;
                        let new_rev = label_box_revision(&updated.boxes);
                        let new_len = updated.boxes.len();
                        let saved_boxes = updated.boxes.clone();
                        if fresh {
                            this.upsert_label_image(&jid2, updated);
                        }
                        this.refresh_label_jobs(cx);
                        if on_job {
                            if fresh {
                                if this.label.editor.is_for(&jid2, &iid2) {
                                    this.label.editor.commit_saved(saved_boxes);
                                }
                                if clear_box {
                                    if this.label.selected_box.as_ref().is_some_and(|s| s.image_id == iid2) {
                                        this.label.selected_box = None;
                                        this.label.hover_box = None;
                                    }
                                } else if let Some(sel) = this.label.selected_box.as_mut()
                                    && sel.image_id == iid2
                                {
                                    if sel.index < new_len {
                                        sel.revision = new_rev;
                                    } else {
                                        this.label.selected_box = None;
                                    }
                                }
                            } else if this.label.editor.is_for(&jid2, &iid2) {
                                // SSE for our save (or a newer remote update) landed while `mutating` was
                                // true: advance `base` to our confirmed save first, then re-sync against
                                // the latest cached image so `is_dirty` clears immediately if they match.
                                this.label.editor.commit_saved(saved_boxes);
                                let _ = this.ensure_label_editor_synced(&jid2, &iid2);
                            }
                            if let Some(pending_sw) = then_switch {
                                this.execute_pending_label_switch(&jid2, pending_sw, cx);
                            } else if let Some(next_id) = advance_to
                                && (still_pending_advance
                                    || this.label.selected_image.as_deref() == Some(iid2.as_str()))
                            {
                                this.select_label_image(next_id, cx);
                            }
                            this.reconcile_selected_label_image(&jid2);
                        }
                    }
                    Err(e) => {
                        this.label.editor.pending_switch = None;
                        if e.code == "label_boxes_changed" {
                            if on_job
                                && !this.label.editor.is_dirty_for(&jid2, &iid2)
                                && this.label.selected_box.as_ref().is_some_and(|s| s.image_id == iid2)
                            {
                                this.label.selected_box = None;
                                this.label.hover_box = None;
                            }
                            this.reload_label_images(&jid2, cx);
                        }
                        if on_job {
                            this.label.review_error = Some(e);
                        }
                    }
                }
            },
        );
    }

    /// Removes an image from the labeling job (two-click confirmed).
    pub fn delete_label_image(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
        if self.label.mutating {
            return;
        }
        self.label.confirm_delete_image = None;
        self.label.mutating = true;
        self.label.mutating_seq += 1;
        let mseq = self.label.mutating_seq;
        let lgen = self.label.epoch;
        self.label.review_error = None;
        let next = self.next_image_after(job_id, image_id);
        let jid = job_id.to_string();
        let iid = image_id.to_string();
        let jid2 = jid.clone();
        let iid2 = iid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.delete_label_image(&jid, &iid),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                if this.label.mutating_seq == mseq {
                    this.label.mutating = false;
                }
                let on_job = route_label(&this.route) == Some(jid2.as_str());
                match r {
                    Ok(()) => {
                        if let Some(list) = this.label.images.get_mut(&jid2).and_then(|l| l.ready_mut()) {
                            list.retain(|i| i.id != iid2);
                        }
                        if this.label.editor.is_for(&jid2, &iid2) {
                            this.label.editor.reset();
                        }
                        if on_job {
                            if this.label.selected_image.as_deref() == Some(iid2.as_str()) {
                                this.label.selected_image = next;
                                this.label.selected_box = None;
                                this.label.hover_box = None;
                            } else if this.label.selected_box.as_ref().is_some_and(|s| s.image_id == iid2) {
                                this.label.selected_box = None;
                                this.label.hover_box = None;
                            }
                            this.reconcile_selected_label_image(&jid2);
                        }
                        this.refresh_label_jobs(cx);
                        this.toast(Tone::Neutral, "Removed image from job", None, cx);
                    }
                    Err(e) => {
                        if on_job {
                            this.label.review_error = Some(e);
                        }
                    }
                }
            },
        );
    }

    /// Selects or toggles box `index` on `image_id` bound to its current box-array revision.
    pub fn toggle_label_box(&mut self, image_id: String, index: usize, revision: String) {
        let already = self
            .label
            .selected_box
            .as_ref()
            .is_some_and(|s| s.image_id == image_id && s.index == index && s.revision == revision);
        self.label.selected_box =
            if already { None } else { Some(SelectedBox { image_id, index, revision }) };
    }

    // ---------- Agent Start / Stop & Delete Job ----------

    pub fn start_label_run(&mut self, job_id: &str, scope: LabelScope, cx: &mut Context<Self>) {
        if self.label.starting {
            return;
        }
        let Some(agent_id) = self.label.agent_id else {
            self.label.start_error = Some(ApiError::new("not_supported", "Pick an installed agent first."));
            cx.notify();
            return;
        };
        let agent =
            self.label.agents.ready().and_then(|list| list.iter().find(|a| a.id == agent_id)).cloned();
        let model = agent.as_ref().and_then(|a| {
            if self.label.model_idx == 0 {
                a.default_model.clone()
            } else {
                a.models.get(self.label.model_idx - 1).map(|m| m.id.clone())
            }
        });
        let fast = self.label.fast
            && agent.as_ref().is_some_and(|a| {
                if self.label.model_idx == 0 {
                    a.fast
                } else {
                    a.models.get(self.label.model_idx - 1).is_some_and(|m| m.fast)
                }
            });
        let req = StartLabeling {
            agent: agent_id,
            model,
            effort: Some(self.label.effort),
            fast,
            scope,
            image_ids: None,
            note: String::new(),
            batch: None,
            parallel: None,
        };
        self.label.starting = true;
        self.label.starting_seq += 1;
        let sseq = self.label.starting_seq;
        let lgen = self.label.epoch;
        let seen_job_seq = self.label.job_event_seq.get(job_id).copied().unwrap_or(0);
        self.label.start_error = None;
        let jid = job_id.to_string();
        let jid2 = jid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.start_labeling(&jid, &req),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                if this.label.starting_seq == sseq {
                    this.label.starting = false;
                }
                match r {
                    Ok(job) => {
                        let name = job.name.clone();
                        let active = job.status.is_active();
                        let cur_seq = this.label.job_event_seq.get(&jid2).copied().unwrap_or(0);
                        if cur_seq == seen_job_seq {
                            this.upsert_label_job(job);
                        }
                        this.reload_label_images(&jid2, cx);
                        if active {
                            this.toast(Tone::Accent, format!("Started labeling {name}"), None, cx);
                        } else {
                            this.toast(Tone::Success, format!("Labeled {name}"), None, cx);
                        }
                    }
                    Err(e) => {
                        if route_label(&this.route) == Some(jid2.as_str()) {
                            this.label.start_error = Some(e);
                        }
                    }
                }
            },
        );
    }

    pub fn stop_label_run(&mut self, job_id: &str, cx: &mut Context<Self>) {
        if !self.label.busy.insert(job_id.to_string()) {
            return;
        }
        let lgen = self.label.epoch;
        let seen_job_seq = self.label.job_event_seq.get(job_id).copied().unwrap_or(0);
        let jid = job_id.to_string();
        let jid2 = jid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.stop_labeling(&jid),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                this.label.busy.remove(&jid2);
                match r {
                    Ok(job) => {
                        let cur_seq = this.label.job_event_seq.get(&jid2).copied().unwrap_or(0);
                        if cur_seq == seen_job_seq {
                            this.upsert_label_job(job);
                        }
                        this.toast(Tone::Neutral, "Stopping labeling run", None, cx);
                    }
                    Err(e) if e.code == "label_job_not_active" => {
                        this.refresh_label_jobs(cx);
                    }
                    Err(e) => this.toast(Tone::Danger, "Couldn't stop labeling", Some(e.message.into()), cx),
                }
            },
        );
    }

    pub fn delete_label_job(&mut self, job_id: &str, cx: &mut Context<Self>) {
        if !self.label.busy.insert(job_id.to_string()) {
            return;
        }
        self.label.confirm_delete_job = None;
        let lgen = self.label.epoch;
        let name = self.label_job_name(job_id);
        let jid = job_id.to_string();
        let jid2 = jid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.delete_label_job(&jid),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                this.label.busy.remove(&jid2);
                match r {
                    Ok(()) => {
                        this.label.jobs_seq += 1;
                        this.label.removed.insert(jid2.clone());
                        if let Some(list) = this.label.jobs.ready_mut() {
                            list.retain(|j| j.id != jid2);
                        }
                        this.label.images.remove(&jid2);
                        this.label.logs.remove(&jid2);
                        if matches!(&this.route, Route::LabelJob(open) if open == &jid2) {
                            this.route_pending = Some(Route::LabelJobs);
                        }
                        this.toast(Tone::Info, format!("Deleted {name}"), None, cx);
                    }
                    Err(e) => this.toast(
                        Tone::Danger,
                        format!("Couldn't delete {name}"),
                        Some(e.message.into()),
                        cx,
                    ),
                }
            },
        );
    }
}
