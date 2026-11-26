//! Mutations, filtering, form validation, and keyboard review actions for Agent Labeling.

use folo_ui::{AppContext, Context, Focusable, Window};

use super::state::{AddFolderForm, ExportDatasetForm, LabelImageFilter, LabelJobFilter, NewLabelJobForm};
use crate::api::*;
use crate::app::{FoloApp, Route, act};
use crate::input::TextInput;
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
    /// Handles top-level review keyboard shortcuts while `Route::LabelJob(job_id)` is open and no
    /// modal, menu, or inline form is active. Returns `true` when the key was handled.
    pub(crate) fn handle_label_review_key(
        &mut self,
        job_id: &str,
        key: &str,
        cx: &mut Context<Self>,
    ) -> bool {
        let shown = self.current_filtered_label_images(job_id);
        let active_id = self
            .label
            .selected_image
            .as_deref()
            .and_then(|sel| shown.iter().find(|im| im.id == sel).map(|im| im.id.clone()))
            .or_else(|| shown.first().map(|im| im.id.clone()));
        let Some(active_id) = active_id else {
            return false;
        };
        let n_classes = self.label_job_by_id(job_id).map_or(0, |j| j.classes.len());
        match key {
            "left" => self.step_label_image(job_id, -1, cx),
            "right" | "s" | "S" => self.step_label_image(job_id, 1, cx),
            "a" | "A" | "enter" => self.approve_label_image(job_id, &active_id, true, cx),
            "delete" | "backspace" => self.delete_selected_label_box(job_id, &active_id, cx),
            "b" | "B" => {
                self.label.show_boxes = !self.label.show_boxes;
            }
            "l" | "L" => {
                self.label.show_labels = !self.label.show_labels;
            }
            "+" | "=" => {
                (self.label.zoom, self.label.pan) =
                    crate::viz::zoom_step(self.label.zoom, self.label.pan, 1.4);
            }
            "-" => {
                (self.label.zoom, self.label.pan) =
                    crate::viz::zoom_step(self.label.zoom, self.label.pan, 1.0 / 1.4);
            }
            "0" => {
                self.label.zoom = 1.0;
                self.label.pan = (0., 0.);
            }
            k if k.len() == 1 && k.as_bytes()[0].is_ascii_digit() => {
                let d = (k.as_bytes()[0] - b'0') as usize;
                if d >= 1 && d <= n_classes {
                    self.reclassify_selected_label_box(job_id, &active_id, (d - 1) as u32, cx);
                } else {
                    return false;
                }
            }
            _ => return false,
        }
        true
    }

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
    pub fn reconcile_selected_label_image(&mut self, job_id: &str) {
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

    /// Selects a specific image in the review workspace, resetting box selection and zoom.
    pub fn select_label_image(&mut self, image_id: String, cx: &mut Context<Self>) {
        if self.label.selected_image.as_deref() != Some(image_id.as_str()) {
            self.label.selected_image = Some(image_id);
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.zoom = 1.0;
            self.label.pan = (0., 0.);
            self.label.review_error = None;
            self.label.confirm_delete_image = None;
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
    /// For a `labeled` image with unchanged boxes, sends `reviewed: true` + `expect` revision;
    /// for a `pending`/`failed` image, saves its current (empty) boxes as a reviewed background image.
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
        let Some(im) = self
            .label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .cloned()
        else {
            return;
        };
        if im.status == LabelImageStatus::Labeling {
            self.label.review_error =
                Some(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            cx.notify();
            return;
        }
        let next_after = if advance { self.next_image_after(job_id, image_id) } else { None };
        let expect = Some(label_box_revision(&im.boxes));
        let req = if im.status == LabelImageStatus::Labeled {
            UpdateLabelImage { boxes: None, reviewed: Some(!im.reviewed || advance), expect }
        } else {
            UpdateLabelImage { boxes: Some(im.boxes.clone()), reviewed: Some(true), expect }
        };
        self.send_label_image_update(job_id, image_id, req, next_after, cx);
    }

    /// Deletes the currently selected box from `image_id` and persists via `PUT /api/label/jobs/{id}/images/{img}`.
    pub fn delete_selected_label_box(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
        if self.label.mutating {
            return;
        }
        let Some(sel) = self.label.selected_box else { return };
        let Some(im) = self
            .label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .cloned()
        else {
            return;
        };
        if sel >= im.boxes.len() {
            self.label.selected_box = None;
            cx.notify();
            return;
        }
        let expect = Some(label_box_revision(&im.boxes));
        let mut boxes = im.boxes.clone();
        boxes.remove(sel);
        let req = UpdateLabelImage { boxes: Some(boxes), reviewed: Some(im.reviewed), expect };
        self.label.selected_box = None;
        self.send_label_image_update(job_id, image_id, req, None, cx);
    }

    /// Reclassifies the currently selected box on `image_id` to `class` and persists to `/api/label`.
    pub fn reclassify_selected_label_box(
        &mut self,
        job_id: &str,
        image_id: &str,
        class: u32,
        cx: &mut Context<Self>,
    ) {
        if self.label.mutating {
            return;
        }
        let Some(sel) = self.label.selected_box else { return };
        let Some(im) = self
            .label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .cloned()
        else {
            return;
        };
        if sel >= im.boxes.len() || im.boxes[sel].class == class {
            return;
        }
        let expect = Some(label_box_revision(&im.boxes));
        let mut boxes = im.boxes.clone();
        boxes[sel].class = class;
        let req = UpdateLabelImage { boxes: Some(boxes), reviewed: Some(im.reviewed), expect };
        self.send_label_image_update(job_id, image_id, req, None, cx);
    }

    fn next_image_after(&self, job_id: &str, current_id: &str) -> Option<String> {
        let shown = self.current_filtered_label_images(job_id);
        let pos = shown.iter().position(|im| im.id == current_id)?;
        // Prefer the next image in view, else wrap to the previous one if this was last.
        shown.get(pos + 1).or_else(|| pos.checked_sub(1).and_then(|p| shown.get(p))).map(|im| im.id.clone())
    }

    fn send_label_image_update(
        &mut self,
        job_id: &str,
        image_id: &str,
        req: UpdateLabelImage,
        advance_to: Option<String>,
        cx: &mut Context<Self>,
    ) {
        self.label.mutating = true;
        self.label.review_error = None;
        let jid = job_id.to_string();
        let iid = image_id.to_string();
        let jid2 = jid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.update_label_image(&jid, &iid, &req),
            move |this, r, cx| {
                this.label.mutating = false;
                match r {
                    Ok(updated) => {
                        this.upsert_label_image(&jid2, updated);
                        this.refresh_label_jobs(cx);
                        if let Some(next_id) = advance_to {
                            this.select_label_image(next_id, cx);
                        }
                        this.reconcile_selected_label_image(&jid2);
                    }
                    Err(e) => {
                        if e.code == "label_boxes_changed" {
                            this.reload_label_images(&jid2, cx);
                        }
                        this.label.review_error = Some(e);
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
                this.label.mutating = false;
                match r {
                    Ok(()) => {
                        if let Some(list) = this.label.images.get_mut(&jid2).and_then(|l| l.ready_mut()) {
                            list.retain(|i| i.id != iid2);
                        }
                        if let Some(next_id) = next {
                            this.label.selected_image = Some(next_id);
                        }
                        this.label.selected_box = None;
                        this.reconcile_selected_label_image(&jid2);
                        this.refresh_label_jobs(cx);
                        this.toast(Tone::Neutral, "Removed image from job", None, cx);
                    }
                    Err(e) => {
                        this.label.review_error = Some(e);
                    }
                }
            },
        );
    }

    // ---------- New labeling job form ----------

    pub fn open_new_label_job(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        self.label.form_seq += 1;
        let seq = self.label.form_seq;
        let name = cx.new(|cx| TextInput::new("e.g. street-scenes", cx));
        let classes = cx.new(|cx| TextInput::new("car, person, bicycle", cx));
        let path = cx.new(|cx| TextInput::new("Optional folder path on the server", cx));
        let instructions =
            cx.new(|cx| TextInput::new("Optional instructions for the agent (what to include or skip)", cx));
        let fh = name.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.label.new_job = Some(NewLabelJobForm {
            seq,
            name,
            classes,
            path,
            instructions,
            submitting: false,
            touched: false,
            error: None,
        });
        cx.notify();
    }

    pub fn close_new_label_job(&mut self, cx: &mut Context<Self>) {
        self.label.new_job = None;
        cx.notify();
    }

    pub fn browse_new_label_folder(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.new_job.as_ref() else { return };
        let path_input = form.path.clone();
        let seq = form.seq;
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Select image folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                let text = p.to_string_lossy().to_string();
                let _ = this.update(cx, |this, cx| {
                    if this.label.new_job.as_ref().is_some_and(|f| f.seq == seq) {
                        path_input.update(cx, |i, cx| i.set_text(text, cx));
                        cx.notify();
                    }
                });
            }
        })
        .detach();
    }

    pub fn submit_new_label_job(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.new_job.as_mut() else { return };
        if form.submitting {
            return;
        }
        form.touched = true;
        let name = form.name.read(cx).text().trim().to_string();
        let classes_raw = form.classes.read(cx).text().to_string();
        let path_raw = form.path.read(cx).text().trim().to_string();
        let instructions = form.instructions.read(cx).text().trim().to_string();
        let classes = match validate_new_job(&name, &classes_raw, &instructions) {
            Ok(c) => c,
            Err(msg) => {
                form.error = Some(ApiError::new("invalid_body", msg));
                cx.notify();
                return;
            }
        };
        form.submitting = true;
        form.error = None;
        let seq = form.seq;
        let req =
            CreateLabelJob { name, classes, instructions, path: (!path_raw.is_empty()).then_some(path_raw) };
        cx.notify();
        self.fetch(
            cx,
            move |api| api.create_label_job(&req),
            move |this, r, cx| {
                let mine = this.label.new_job.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(job) => {
                        let id = job.id.clone();
                        let title = job.name.clone();
                        let n_cls = job.classes.len();
                        if mine {
                            this.label.new_job = None;
                        }
                        this.upsert_label_job(job);
                        this.route_pending = Some(Route::LabelJob(id));
                        this.toast(
                            Tone::Accent,
                            format!("Created {title}"),
                            Some(format!("{n_cls} {}", if n_cls == 1 { "class" } else { "classes" }).into()),
                            cx,
                        );
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.label.new_job.as_mut() {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    // ---------- Add folder to existing job ----------

    pub fn open_add_label_folder(&mut self, job_id: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.label.export_ds = None;
        self.label.form_seq += 1;
        let seq = self.label.form_seq;
        let path = cx.new(|cx| TextInput::new("Folder or image path on the server", cx));
        let fh = path.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.label.add_folder =
            Some(AddFolderForm { job: job_id.to_string(), seq, path, submitting: false, error: None });
        cx.notify();
    }

    pub fn browse_add_label_folder(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.add_folder.as_ref() else { return };
        let path_input = form.path.clone();
        let seq = form.seq;
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Select image folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                let text = p.to_string_lossy().to_string();
                let _ = this.update(cx, |this, cx| {
                    if this.label.add_folder.as_ref().is_some_and(|f| f.seq == seq) {
                        path_input.update(cx, |i, cx| i.set_text(text, cx));
                        cx.notify();
                    }
                });
            }
        })
        .detach();
    }

    pub fn submit_add_label_folder(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.add_folder.as_mut() else { return };
        if form.submitting {
            return;
        }
        let path = form.path.read(cx).text().trim().to_string();
        if path.is_empty() {
            form.error = Some(ApiError::new("invalid_body", "Enter a folder path."));
            cx.notify();
            return;
        }
        form.submitting = true;
        form.error = None;
        let job_id = form.job.clone();
        let seq = form.seq;
        let req = ImportLabelImages { path };
        let jid = job_id.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.import_label_images(&jid, &req),
            move |this, r, cx| {
                let mine = this.label.add_folder.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(res) => {
                        if mine {
                            this.label.add_folder = None;
                        }
                        let added_n = res.added.len();
                        let skipped_n = res.skipped.len();
                        this.upsert_label_job(res.job);
                        this.reload_label_images(&job_id, cx);
                        let body =
                            if skipped_n > 0 { Some(format!("{skipped_n} skipped").into()) } else { None };
                        this.toast(
                            Tone::Success,
                            format!("Added {} {}", added_n, if added_n == 1 { "image" } else { "images" }),
                            body,
                            cx,
                        );
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.label.add_folder.as_mut() {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    // ---------- Export / Register as Dataset ----------

    pub fn open_export_label_dataset(&mut self, job: &LabelJob, window: &mut Window, cx: &mut Context<Self>) {
        self.label.add_folder = None;
        self.label.form_seq += 1;
        let seq = self.label.form_seq;
        let default_name = job.name.clone();
        let name = cx.new(|cx| {
            let mut i = TextInput::new("Dataset name", cx);
            i.set_text(default_name, cx);
            i
        });
        let fh = name.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.label.export_ds = Some(ExportDatasetForm {
            job: job.id.clone(),
            seq,
            name,
            val_idx: 1, // 20%
            submitting: false,
            error: None,
        });
        cx.notify();
    }

    pub fn submit_export_label_dataset(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.export_ds.as_mut() else { return };
        if form.submitting {
            return;
        }
        let name = form.name.read(cx).text().trim().to_string();
        if name.is_empty() {
            form.error = Some(ApiError::new("invalid_body", "Enter a dataset name."));
            cx.notify();
            return;
        }
        form.submitting = true;
        form.error = None;
        let job_id = form.job.clone();
        let seq = form.seq;
        let val_fraction = LABEL_VAL_FRACTIONS[form.val_idx.min(LABEL_VAL_FRACTIONS.len() - 1)];
        let req = LabelDatasetRequest { name: Some(name), val_fraction, seed: 0 };
        let jid = job_id.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.create_label_dataset(&jid, &req),
            move |this, r, cx| {
                let mine = this.label.export_ds.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(ds) => {
                        if mine {
                            this.label.export_ds = None;
                        }
                        let ds_id = ds.id.clone();
                        let ds_name = ds.name.clone();
                        let total = ds.splits.total();
                        if let Some(list) = this.store.datasets.ready_mut() {
                            match list.iter_mut().find(|d| d.id == ds.id) {
                                Some(slot) => *slot = ds,
                                None => list.push(ds),
                            }
                        }
                        if let Some(jobs) = this.label.jobs.ready_mut()
                            && let Some(job) = jobs.iter_mut().find(|j| j.id == job_id)
                        {
                            job.dataset_id = Some(ds_id.clone());
                        }
                        this.refresh_label_jobs(cx);
                        this.toast_with(
                            Tone::Success,
                            format!("Exported dataset {ds_name}"),
                            Some(format!("{total} images registered and ready to train.").into()),
                            Some((
                                "Open dataset".into(),
                                act(move |this, w, cx| this.navigate(Route::Dataset(ds_id.clone()), w, cx)),
                            )),
                            cx,
                        );
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.label.export_ds.as_mut() {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
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
        self.label.start_error = None;
        let jid = job_id.to_string();
        let jid2 = jid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.start_labeling(&jid, &req),
            move |this, r, cx| {
                this.label.starting = false;
                match r {
                    Ok(job) => {
                        let name = job.name.clone();
                        let active = job.status.is_active();
                        this.upsert_label_job(job);
                        this.reload_label_images(&jid2, cx);
                        if active {
                            this.toast(Tone::Accent, format!("Started labeling {name}"), None, cx);
                        } else {
                            this.toast(Tone::Success, format!("Labeled {name}"), None, cx);
                        }
                    }
                    Err(e) => {
                        this.label.start_error = Some(e);
                    }
                }
            },
        );
    }

    pub fn stop_label_run(&mut self, job_id: &str, cx: &mut Context<Self>) {
        if !self.label.busy.insert(job_id.to_string()) {
            return;
        }
        let jid = job_id.to_string();
        let jid2 = jid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.stop_labeling(&jid),
            move |this, r, cx| {
                this.label.busy.remove(&jid2);
                match r {
                    Ok(job) => {
                        this.upsert_label_job(job);
                        this.toast(Tone::Neutral, "Stopping labeling run", None, cx);
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
        let name = self.label_job_name(job_id);
        let jid = job_id.to_string();
        let jid2 = jid.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.delete_label_job(&jid),
            move |this, r, cx| {
                this.label.busy.remove(&jid2);
                match r {
                    Ok(()) => {
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

    /// Closes an inline labeling form or delete confirmation on `Esc`.
    pub fn dismiss_label_form(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if self.label.new_job.take().is_some()
            || self.label.add_folder.take().is_some()
            || self.label.export_ds.take().is_some()
            || self.label.confirm_delete_job.take().is_some()
            || self.label.confirm_delete_image.take().is_some()
        {
            window.focus(&self.root_focus);
            cx.notify();
            true
        } else if self.label.selected_box.take().is_some() {
            cx.notify();
            true
        } else {
            false
        }
    }

    /// Submits the open inline labeling form on `Ctrl+Enter`.
    pub fn submit_label_form(&mut self, cx: &mut Context<Self>) -> bool {
        if self.label.new_job.is_some() {
            self.submit_new_label_job(cx);
            true
        } else if self.label.add_folder.is_some() {
            self.submit_add_label_folder(cx);
            true
        } else if self.label.export_ds.is_some() {
            self.submit_export_label_dataset(cx);
            true
        } else {
            false
        }
    }
}
