//! Interactive bounding-box editor and guarded-navigation actions for Agent Labeling.

use folo_ui::Context;

use super::editor::{EditTool, PendingImageSwitch, PendingRouteNav};
use super::state::{LabelImageFilter, SelectedBox};
use crate::api::*;
use crate::app::FoloApp;

impl FoloApp {
    /// True when the currently open `Route::LabelJob` has an unsaved box draft that is actually
    /// active and renderable (the feature is enabled, the job still exists, and the edited image is
    /// still in the loaded image list so the Save / Discard / Stay banner can be shown).
    pub(crate) fn label_route_has_renderable_dirty_draft(&self) -> bool {
        let crate::app::Route::LabelJob(job_id) = &self.route else {
            return false;
        };
        if !self.label_on() || self.label.removed.contains(job_id) || self.label_job_by_id(job_id).is_none() {
            return false;
        }
        if !self.label.editor.is_dirty() || self.label.editor.job_id.as_deref() != Some(job_id.as_str()) {
            return false;
        }
        let Some(img_id) = self.label.editor.image_id.as_deref() else {
            return false;
        };
        self.label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .is_some_and(|list| list.iter().any(|im| im.id == img_id))
    }

    /// Ensures `self.label.editor` is bound to `(job_id, image_id)` and returns the server snapshot.
    pub(crate) fn ensure_label_editor_synced(&mut self, job_id: &str, image_id: &str) -> Option<LabelImage> {
        let im = self
            .label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .cloned()?;
        let n_classes = self.label_job_by_id(job_id).map_or(0, |j| j.classes.len());
        self.label.editor.sync_image(job_id, &im, n_classes, self.label.mutating);
        Some(im)
    }

    /// Resolves the currently active image id in `job_id`'s review workspace (keeping a dirty image
    /// active even if a concurrent SSE update moved it outside the current filter).
    fn active_label_image_id(&self, job_id: &str) -> Option<String> {
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
            return Some(dirty_id.to_string());
        }
        let shown = self.current_filtered_label_images(job_id);
        self.label
            .selected_image
            .as_deref()
            .and_then(|sel| shown.iter().find(|im| im.id == sel).map(|im| im.id.clone()))
            .or_else(|| shown.first().map(|im| im.id.clone()))
    }

    /// Handles Ctrl-modified review shortcuts (`Ctrl+Z` undo, `Ctrl+Shift+Z` / `Ctrl+Y` redo,
    /// `Ctrl+S` save edits) while `Route::LabelJob(job_id)` is open and no modal/form is active.
    pub(crate) fn handle_label_review_shortcut(
        &mut self,
        job_id: &str,
        key: &str,
        shift: bool,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.label.confirm_delete_job.is_some() || self.label.confirm_delete_image.is_some() {
            return false;
        }
        let Some(active_id) = self.active_label_image_id(job_id) else {
            return false;
        };
        match (key, shift) {
            ("z" | "Z", false) => {
                self.undo_label_edit(job_id, &active_id, cx);
                true
            }
            ("z" | "Z", true) | ("y" | "Y", false) => {
                self.redo_label_edit(job_id, &active_id, cx);
                true
            }
            ("s" | "S", false) => {
                if self.label.editor.is_dirty_for(job_id, &active_id) {
                    self.save_label_edits(job_id, &active_id, false, cx);
                }
                true
            }
            _ => false,
        }
    }

    /// Handles top-level review keyboard shortcuts while `Route::LabelJob(job_id)` is open and no
    /// modal, menu, or inline form is active. Returns `true` when the key was handled.
    pub(crate) fn handle_label_review_key(
        &mut self,
        job_id: &str,
        key: &str,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.label.confirm_delete_job.is_some() || self.label.confirm_delete_image.is_some() {
            return false;
        }
        let Some(active_id) = self.active_label_image_id(job_id) else {
            return false;
        };
        let n_classes = self.label_job_by_id(job_id).map_or(0, |j| j.classes.len());
        let has_selected_box = self.label.selected_box.as_ref().is_some_and(|s| s.image_id == active_id);
        match key {
            "left" => self.request_step_label_image(job_id, -1, cx),
            "right" | "s" | "S" => self.request_step_label_image(job_id, 1, cx),
            "a" | "A" => self.approve_label_image(job_id, &active_id, true, cx),
            "d" | "D" => {
                self.label.editor.cancel_draw();
                self.label.editor.cancel_transform();
                self.label.editor.tool = match self.label.editor.tool {
                    EditTool::Select => EditTool::Draw,
                    EditTool::Draw => EditTool::Select,
                };
            }
            "delete" | "backspace" if has_selected_box => {
                self.delete_selected_label_box(job_id, &active_id, cx)
            }
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
                    self.set_label_draw_class(job_id, &active_id, (d - 1) as u32, cx);
                } else {
                    return false;
                }
            }
            _ => return false,
        }
        true
    }

    /// Guards image selection when unsaved box edits exist, prompting Save / Discard / Stay.
    pub fn request_select_label_image(&mut self, image_id: String, cx: &mut Context<Self>) {
        if self.label.selected_image.as_deref() == Some(image_id.as_str()) {
            return;
        }
        if self.label.editor.is_dirty() {
            self.label.editor.pending_switch = Some(PendingImageSwitch::SelectImage(image_id));
            cx.notify();
            return;
        }
        self.select_label_image(image_id, cx);
    }

    /// Guards stepping to prev/next image when unsaved box edits exist.
    pub fn request_step_label_image(&mut self, job_id: &str, delta: i32, cx: &mut Context<Self>) {
        let shown = self.current_filtered_label_images(job_id);
        let cur_in_shown =
            self.label.selected_image.as_deref().is_some_and(|sel| shown.iter().any(|im| im.id == sel));
        // If stepping would stay on the exact same image (0 images, or 1 image that is already open),
        // do nothing instead of raising a false unsaved-edits prompt.
        if shown.is_empty() || (shown.len() == 1 && cur_in_shown) {
            return;
        }
        if self.label.editor.is_dirty() {
            self.label.editor.pending_switch = Some(PendingImageSwitch::Step(delta));
            cx.notify();
            return;
        }
        self.step_label_image(job_id, delta, cx);
    }

    /// Guards status filter changes when unsaved box edits exist.
    pub fn request_label_img_filter(
        &mut self,
        job_id: &str,
        filter: LabelImageFilter,
        cx: &mut Context<Self>,
    ) {
        if self.label.img_filter == filter {
            return;
        }
        if self.label.editor.is_dirty() {
            self.label.editor.pending_switch = Some(PendingImageSwitch::SetFilter(filter));
            cx.notify();
            return;
        }
        self.label.img_filter = filter;
        self.reconcile_selected_label_image(job_id);
        cx.notify();
    }

    /// Guards class filter changes when unsaved box edits exist.
    pub fn request_label_class_filter(&mut self, job_id: &str, class: Option<u32>, cx: &mut Context<Self>) {
        if self.label.class_filter == class {
            return;
        }
        if self.label.editor.is_dirty() {
            self.label.editor.pending_switch = Some(PendingImageSwitch::SetClassFilter(class));
            cx.notify();
            return;
        }
        self.label.class_filter = class;
        self.reconcile_selected_label_image(job_id);
        cx.notify();
    }

    /// Resolves a pending image/filter/route switch: either saves edits first or discards them and continues.
    pub fn resolve_pending_label_switch(&mut self, job_id: &str, save_first: bool, cx: &mut Context<Self>) {
        if self.label.mutating {
            return;
        }
        let Some(switch) = self.label.editor.pending_switch.take() else {
            return;
        };
        if save_first {
            if let Some(active_id) = self.label.editor.image_id.clone() {
                self.save_label_edits_inner(job_id, &active_id, false, false, Some(switch), cx);
                return;
            }
        } else {
            self.label.editor.discard();
            self.label.selected_box = None;
            self.label.hover_box = None;
        }
        self.execute_pending_label_switch(job_id, switch, cx);
    }

    pub(super) fn execute_pending_label_switch(
        &mut self,
        job_id: &str,
        switch: PendingImageSwitch,
        cx: &mut Context<Self>,
    ) {
        match switch {
            PendingImageSwitch::SelectImage(id) => self.select_label_image(id, cx),
            PendingImageSwitch::Step(delta) => self.step_label_image(job_id, delta, cx),
            PendingImageSwitch::SetFilter(f) => {
                self.label.img_filter = f;
                self.reconcile_selected_label_image(job_id);
                cx.notify();
            }
            PendingImageSwitch::SetClassFilter(c) => {
                self.label.class_filter = c;
                self.reconcile_selected_label_image(job_id);
                cx.notify();
            }
            PendingImageSwitch::Navigate(route) => {
                self.label.pending_nav = Some(PendingRouteNav::Navigate(route));
                cx.notify();
            }
            PendingImageSwitch::Back => {
                self.label.pending_nav = Some(PendingRouteNav::Back);
                cx.notify();
            }
            PendingImageSwitch::Forward => {
                self.label.pending_nav = Some(PendingRouteNav::Forward);
                cx.notify();
            }
            PendingImageSwitch::CloseWindow => {
                self.label.pending_nav = Some(PendingRouteNav::CloseWindow);
                cx.notify();
            }
        }
    }

    /// Saves the local draft boxes for `image_id` via `PUT /api/label/jobs/{id}/images/{img}`,
    /// using the revision this draft was originally edited against (so concurrent remote edits
    /// return `409 label_boxes_changed` instead of being silently overwritten).
    pub fn save_label_edits(&mut self, job_id: &str, image_id: &str, advance: bool, cx: &mut Context<Self>) {
        self.save_label_edits_inner(job_id, image_id, advance, false, None, cx);
    }

    /// Explicitly overwrites remote changes with the local draft after the user saw the conflict
    /// warning banner (`expect` is refreshed to the latest server `base_revision()`).
    pub fn save_label_edits_overwrite(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
        self.save_label_edits_inner(job_id, image_id, false, true, None, cx);
    }

    fn save_label_edits_inner(
        &mut self,
        job_id: &str,
        image_id: &str,
        advance: bool,
        overwrite: bool,
        then_switch: Option<PendingImageSwitch>,
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
        self.label.editor.pending_switch = None;
        let expect = Some(if overwrite {
            self.label.editor.base_revision()
        } else {
            self.label.editor.expect_rev.clone()
        });
        let boxes = self.label.editor.draft.clone();
        let next_after = if advance { self.next_image_after(job_id, image_id) } else { None };
        // `Approve` / `Save & next` marks the image reviewed; plain `Save` (`Ctrl+S`) preserves the
        // image's current review state so an image in the `To review` filter doesn't jump away mid-edit.
        let reviewed = if advance { Some(true) } else { Some(im.reviewed) };
        let req = UpdateLabelImage { boxes: Some(boxes), reviewed, expect };
        self.send_label_image_update(job_id, image_id, req, next_after, then_switch, false, cx);
    }

    /// Sets the active drawing class and, when a box is selected, reclassifies it in the local draft.
    pub fn set_label_draw_class(&mut self, job_id: &str, image_id: &str, class: u32, cx: &mut Context<Self>) {
        let n_classes = self.label_job_by_id(job_id).map_or(0, |j| j.classes.len());
        if (class as usize) >= n_classes {
            return;
        }
        self.label.editor.draw_class = class;
        if self.label.selected_box.as_ref().is_some_and(|s| s.image_id == image_id) {
            self.reclassify_selected_label_box(job_id, image_id, class, cx);
        } else {
            cx.notify();
        }
    }

    /// Deletes the currently selected box from the local draft of `image_id` (undoable via `Ctrl+Z`).
    pub fn delete_selected_label_box(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
        if self.label.mutating {
            return;
        }
        let Some(sel) = self.label.selected_box.clone() else { return };
        if sel.image_id != image_id {
            self.label.selected_box = None;
            cx.notify();
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
        if sel.index >= self.label.editor.draft.len() || self.label.editor.draft_revision() != sel.revision {
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.review_error = Some(ApiError::new(
                "label_boxes_changed",
                "This image's boxes changed; reselect the box before deleting.",
            ));
            cx.notify();
            return;
        }
        if self.label.editor.delete_box(sel.index) {
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.review_error = None;
            cx.notify();
        }
    }

    /// Reclassifies the currently selected box in the local draft of `image_id` (undoable via `Ctrl+Z`).
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
        let Some(sel) = self.label.selected_box.clone() else { return };
        if sel.image_id != image_id {
            self.label.selected_box = None;
            cx.notify();
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
        if sel.index >= self.label.editor.draft.len() || self.label.editor.draft_revision() != sel.revision {
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.review_error = Some(ApiError::new(
                "label_boxes_changed",
                "This image's boxes changed; reselect the box before reclassifying.",
            ));
            cx.notify();
            return;
        }
        self.label.editor.draw_class = class;
        if self.label.editor.reclassify_box(sel.index, class) {
            let new_rev = self.label.editor.draft_revision();
            self.label.selected_box =
                Some(SelectedBox { image_id: image_id.to_string(), index: sel.index, revision: new_rev });
            self.label.review_error = None;
            cx.notify();
        }
    }

    /// Undoes the last bounding-box edit on `image_id` (`Ctrl+Z`).
    pub fn undo_label_edit(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
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
        if self.label.editor.undo() {
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.review_error = None;
            cx.notify();
        }
    }

    /// Redoes the last undone bounding-box edit on `image_id` (`Ctrl+Shift+Z` / `Ctrl+Y`).
    pub fn redo_label_edit(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
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
        if self.label.editor.redo() {
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.review_error = None;
            cx.notify();
        }
    }

    /// Reverts unsaved box edits back to the last saved/server boxes (undoable via `Ctrl+Z`).
    pub fn discard_label_edits(&mut self, job_id: &str, image_id: &str, cx: &mut Context<Self>) {
        if self.label.mutating {
            return;
        }
        if self.ensure_label_editor_synced(job_id, image_id).is_none() {
            return;
        }
        if self.label.editor.discard() {
            self.label.selected_box = None;
            self.label.hover_box = None;
            self.label.review_error = None;
            cx.notify();
        }
    }

    /// Commits an in-progress click-drag box draw on the review stage (if moved past threshold and
    /// at least `MIN_BOX_PX` in width/height), selecting the newly created box.
    pub(crate) fn finish_label_box_draw(&mut self, cx: &mut Context<Self>) {
        let Some(job_id) = self.label.editor.job_id.clone() else {
            self.label.editor.cancel_draw();
            return;
        };
        let Some(image_id) = self.label.editor.image_id.clone() else {
            self.label.editor.cancel_draw();
            return;
        };
        let Some((w, h, locked)) = self
            .label
            .images
            .get(&job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .map(|im| {
                (im.width.max(1) as f32, im.height.max(1) as f32, im.status == LabelImageStatus::Labeling)
            })
        else {
            self.label.editor.cancel_draw();
            return;
        };
        let was_moved = self.label.editor.drawing.as_ref().is_some_and(|d| d.moved);
        if self.label.mutating || locked {
            self.label.editor.cancel_draw();
            if locked && was_moved {
                self.label.review_error =
                    Some(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            }
            cx.notify();
            return;
        }
        if let Some(new_idx) = self.label.editor.finish_draw(w, h) {
            let rev = self.label.editor.draft_revision();
            self.label.selected_box = Some(SelectedBox { image_id, index: new_idx, revision: rev });
            self.label.hover_box = None;
            self.label.review_error = None;
            cx.notify();
        } else if !was_moved {
            // Plain click on empty space clears box selection.
            if self.label.selected_box.take().is_some() {
                cx.notify();
            }
        } else {
            cx.notify();
        }
    }

    /// Updates an in-progress click-drag box Move or Resize from a window-level pointer move event.
    pub(crate) fn update_label_box_transform(
        &mut self,
        pos: folo_ui::Point<folo_ui::Pixels>,
        cx: &mut Context<Self>,
    ) {
        let Some(b) = self.label.stage_view.get() else {
            return;
        };
        let (Some(job_id), Some(image_id)) =
            (self.label.editor.job_id.as_deref(), self.label.editor.image_id.as_deref())
        else {
            return;
        };
        let Some(dims) = self
            .label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .map(|im| (im.width.max(1) as f32, im.height.max(1) as f32))
        else {
            return;
        };
        let img_pt =
            crate::viz::to_image(b, dims, crate::viz::Fit::Contain, self.label.zoom, self.label.pan, pos);
        if self.label.editor.update_transform(img_pt, (f32::from(pos.x), f32::from(pos.y))) {
            cx.notify();
        }
    }

    /// Commits an in-progress box Move or Resize as one undoable edit and updates box selection rev.
    pub(crate) fn finish_label_box_transform(&mut self, cx: &mut Context<Self>) {
        let Some(job_id) = self.label.editor.job_id.clone() else {
            self.label.editor.cancel_transform();
            return;
        };
        let Some(image_id) = self.label.editor.image_id.clone() else {
            self.label.editor.cancel_transform();
            return;
        };
        let Some((w, h, locked)) = self
            .label
            .images
            .get(&job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .map(|im| {
                (im.width.max(1) as f32, im.height.max(1) as f32, im.status == LabelImageStatus::Labeling)
            })
        else {
            self.label.editor.cancel_transform();
            return;
        };
        let was_moved = self.label.editor.transforming.as_ref().is_some_and(|t| t.moved);
        if self.label.mutating || locked {
            self.label.editor.cancel_transform();
            if locked && was_moved {
                self.label.review_error =
                    Some(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            }
            cx.notify();
            return;
        }
        if let Some(idx) = self.label.editor.finish_transform(w, h) {
            let rev = self.label.editor.draft_revision();
            self.label.selected_box = Some(SelectedBox { image_id, index: idx, revision: rev });
            self.label.hover_box = None;
            self.label.review_error = None;
            cx.notify();
        } else if was_moved {
            cx.notify();
        }
    }

    /// Updates an in-progress click-drag box draw from a window-level pointer move event.
    pub(crate) fn update_label_box_draw(
        &mut self,
        pos: folo_ui::Point<folo_ui::Pixels>,
        cx: &mut Context<Self>,
    ) {
        let Some(b) = self.label.stage_view.get() else {
            return;
        };
        let (Some(job_id), Some(image_id)) =
            (self.label.editor.job_id.as_deref(), self.label.editor.image_id.as_deref())
        else {
            return;
        };
        let Some(dims) = self
            .label
            .images
            .get(job_id)
            .and_then(|l| l.ready())
            .and_then(|list| list.iter().find(|i| i.id == image_id))
            .map(|im| (im.width.max(1) as f32, im.height.max(1) as f32))
        else {
            return;
        };
        let img_pt =
            crate::viz::to_image(b, dims, crate::viz::Fit::Contain, self.label.zoom, self.label.pan, pos);
        if self.label.editor.update_draw(img_pt, (f32::from(pos.x), f32::from(pos.y))) {
            cx.notify();
        }
    }
}
