//! Image review workspace for Agent Labeling: status & class filters, interactive bounding-box
//! toolbar (Select/Draw, Undo/Redo, Save/Discard), conflict/unsaved-edit notices, stage + box
//! rail, and image filmstrip.

use std::rc::Rc;

use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{LabelImage, LabelImageStatus, LabelJob, LabelSource};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::label::{EditTool, LabelImageFilter, filter_label_images, previous_labeled_frame};
use crate::screens::loading_rows;
use crate::theme::{Theme, ty};
use crate::ui::*;
use crate::viz::zoom_step;

impl FoloApp {
    /// Main column of `Route::LabelJob`: filter bar, review stage, box list, and filmstrip.
    pub(super) fn label_review_workspace(
        &mut self,
        t: &Theme,
        job: &LabelJob,
        review_w: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let jid = job.id.clone();
        let images_load = self.label.images.get(&jid);
        let Some(all_images) = images_load.and_then(|l| l.ready()).cloned() else {
            if let Some(err) = images_load.and_then(|l| l.error()).cloned() {
                let retry_id = jid.clone();
                let retry = self.button(
                    "label-imgs-retry",
                    "Retry",
                    Btn::Primary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_label_job(&retry_id, cx)),
                );
                return panel(t)
                    .child(
                        empty_state(
                            t,
                            "icons/circle-alert.svg",
                            "Could not load images",
                            &err.message,
                            Tone::Danger,
                        )
                        .child(retry),
                    )
                    .into_any_element();
            }
            return panel(t)
                .child(panel_header(t, "Images"))
                .child(loading_rows(t, "label-imgs", 5))
                .into_any_element();
        };

        if all_images.is_empty() {
            let add_id = jid.clone();
            let cta = self.button(
                "label-imgs-empty-add",
                "Add folder",
                Btn::Primary,
                Some("icons/folder-open.svg"),
                window,
                cx,
                act(move |this, w, cx| this.open_add_label_folder(&add_id, w, cx)),
            );
            return panel(t)
                .child(
                    empty_state(
                        t,
                        "icons/images.svg",
                        "No images in this job yet",
                        "Add a folder of JPEG, PNG, BMP, or WebP images on the server's machine, then run an agent to generate draft boxes.",
                        Tone::Neutral,
                    )
                    .child(cta),
                )
                .into_any_element();
        }

        // Counts per status filter bucket.
        let to_review_n =
            all_images.iter().filter(|i| i.status == LabelImageStatus::Labeled && !i.reviewed).count();
        let reviewed_n =
            all_images.iter().filter(|i| i.status == LabelImageStatus::Labeled && i.reviewed).count();
        let pending_n = all_images
            .iter()
            .filter(|i| matches!(i.status, LabelImageStatus::Pending | LabelImageStatus::Labeling))
            .count();
        let failed_n = all_images.iter().filter(|i| i.status == LabelImageStatus::Failed).count();

        let buckets = [
            LabelImageFilter::All,
            LabelImageFilter::ToReview,
            LabelImageFilter::Reviewed,
            LabelImageFilter::Pending,
            LabelImageFilter::Failed,
        ];
        let seg_labels = [
            format!("All\u{a0}{}", int(all_images.len() as u64)),
            format!("To review\u{a0}{}", int(to_review_n as u64)),
            format!("Reviewed\u{a0}{}", int(reviewed_n as u64)),
            format!("Pending\u{a0}{}", int(pending_n as u64)),
            format!("Failed\u{a0}{}", int(failed_n as u64)),
        ];
        let seg_refs: Vec<&str> = seg_labels.iter().map(String::as_str).collect();
        let cur_bucket = buckets.iter().position(|b| *b == self.label.img_filter).unwrap_or(0);
        let jid_filter = jid.clone();
        let status_seg = self.segmented(
            "label-img-filter",
            &seg_refs,
            cur_bucket,
            window,
            cx,
            Rc::new(move |this, idx, _, cx| {
                this.request_label_img_filter(&jid_filter, buckets[idx.min(buckets.len() - 1)], cx);
            }),
        );

        // Class filter chip strip.
        let class_filter = self.label.class_filter;
        let mut class_chips = Vec::with_capacity(job.classes.len() + 1);
        let jid_all_cls = jid.clone();
        class_chips.push(
            self.chip(
                "label-cls-all",
                "All classes",
                class_filter.is_none(),
                None,
                Some(all_images.len() as u64),
                window,
                cx,
                act(move |this, _, cx| this.request_label_class_filter(&jid_all_cls, None, cx)),
            )
            .into_any_element(),
        );
        for (ci, name) in job.classes.iter().enumerate() {
            let cls_idx = ci as u32;
            let count = all_images.iter().filter(|im| im.boxes.iter().any(|b| b.class == cls_idx)).count();
            let jid_cls = jid.clone();
            class_chips.push(
                self.chip(
                    &format!("label-cls-{ci}"),
                    name,
                    class_filter == Some(cls_idx),
                    Some(cls_idx),
                    Some(count as u64),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        let next =
                            if this.label.class_filter == Some(cls_idx) { None } else { Some(cls_idx) };
                        this.request_label_class_filter(&jid_cls, next, cx);
                    }),
                )
                .into_any_element(),
            );
        }

        let shown: Vec<LabelImage> =
            filter_label_images(&all_images, self.label.img_filter, self.label.class_filter)
                .into_iter()
                .cloned()
                .collect();

        let filter_bar = div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .child(status_seg)
                    .child(
                        text(ty::XS, t.text_tertiary).whitespace_normal().child(
                            "Keys: ←/→ step · U next unlabeled · D draw · Ctrl+D dup box · P copy prev · 1–9 class · Del box · Ctrl+Z/⇧Z undo/redo · Ctrl+S save · A approve",
                        ),
                    ),
            )
            .child(div().flex().flex_wrap().items_center().gap(px(6.)).children(class_chips));

        // If the user has unsaved edits on an image that a concurrent SSE update moved outside the
        // current filter, keep that dirty image open so the draft is never silently discarded.
        let dirty_outside = (self.label.editor.is_dirty()
            && self.label.editor.job_id.as_deref() == Some(jid.as_str()))
        .then(|| {
            let dirty_id = self.label.editor.image_id.as_deref()?;
            if shown.iter().any(|im| im.id == dirty_id) {
                None
            } else {
                all_images.iter().find(|im| im.id == dirty_id).cloned()
            }
        })
        .flatten();

        if shown.is_empty() && dirty_outside.is_none() {
            let jid_reset = jid.clone();
            let reset = self.button(
                "label-imgs-reset",
                "Clear filters",
                Btn::Secondary,
                None,
                window,
                cx,
                act(move |this, _, cx| {
                    this.label.img_filter = LabelImageFilter::All;
                    this.label.class_filter = None;
                    this.label.editor.pending_switch = None;
                    this.reconcile_selected_label_image(&jid_reset);
                    cx.notify();
                }),
            );
            let busy = self.label.mutating;
            if pending_n + failed_n > 0 {
                if busy && self.fh("label-next-unlabeled", cx).is_focused(window) {
                    window.focus(&self.fh("label-next-unlabeled-disabled", cx));
                } else if !busy && self.fh("label-next-unlabeled-disabled", cx).is_focused(window) {
                    window.focus(&self.fh("label-next-unlabeled", cx));
                }
            }
            let jump_unlabeled = (pending_n + failed_n > 0).then(|| {
                if busy {
                    self.button_disabled(
                        "label-next-unlabeled",
                        "Next unlabeled",
                        Btn::Primary,
                        Some("icons/target.svg"),
                        "Saving…",
                        window,
                        cx,
                    )
                    .into_any_element()
                } else {
                    let jid_u = jid.clone();
                    self.button(
                        "label-next-unlabeled",
                        "Next unlabeled",
                        Btn::Primary,
                        Some("icons/target.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.request_jump_next_unlabeled(&jid_u, cx)),
                    )
                    .tooltip(crate::controls::tip(
                        "Jump to the next pending or failed image in sequence",
                        Some("U"),
                    ))
                    .into_any_element()
                }
            });
            return div()
                .flex()
                .flex_col()
                .gap(px(12.))
                .child(filter_bar)
                .child(
                    panel(t).child(
                        empty_state(
                            t,
                            "icons/list-filter.svg",
                            "No images match this filter",
                            "Switch status filter or clear the class filter to review other images.",
                            Tone::Neutral,
                        )
                        .child(div().flex().items_center().gap(px(8.)).child(reset).children(jump_unlabeled)),
                    ),
                )
                .into_any_element();
        }

        // Resolve the active image (keeping any dirty outside-filter image visible until resolved)
        // and sync the interactive editor.
        let (cur_pos, active) = if let Some(dirty_im) = dirty_outside {
            (None, dirty_im)
        } else {
            let pos = self
                .label
                .selected_image
                .as_deref()
                .and_then(|sel| shown.iter().position(|im| im.id == sel))
                .unwrap_or(0);
            (Some(pos), shown[pos].clone())
        };
        if self.label.selected_image.as_deref() != Some(active.id.as_str()) {
            self.label.selected_image = Some(active.id.clone());
            self.label.selected_box = None;
        }
        self.label.editor.sync_image(&jid, &active, job.classes.len(), self.label.mutating);
        let draft_boxes = self.label.editor.draft.clone();
        let draft_rev = self.label.editor.draft_revision();
        let dirty = self.label.editor.is_dirty();
        let can_undo = self.label.editor.can_undo();
        let can_redo = self.label.editor.can_redo();
        let changed_elsewhere = self.label.editor.changed_elsewhere;
        let has_pending_switch = self.label.editor.pending_switch.is_some();

        let sel_box = self
            .label
            .selected_box
            .as_ref()
            .filter(|s| s.image_id == active.id && s.index < draft_boxes.len() && s.revision == draft_rev)
            .map(|s| s.index);
        if sel_box.is_none() && self.label.selected_box.is_some() {
            self.label.selected_box = None;
        }

        // Toolbar above the image stage.
        let jid_prev = jid.clone();
        let prev_btn = self.icon_button(
            "label-prev",
            "icons/chevron-left.svg",
            "Previous image (←)",
            window,
            cx,
            act(move |this, _, cx| this.request_step_label_image(&jid_prev, -1, cx)),
        );
        let jid_next = jid.clone();
        let next_btn = self.icon_button(
            "label-next",
            "icons/chevron-right.svg",
            "Next image (→)",
            window,
            cx,
            act(move |this, _, cx| this.request_step_label_image(&jid_next, 1, cx)),
        );
        let counter = match cur_pos {
            Some(pos) => format!("{} of {}", int(pos as u64 + 1), int(shown.len() as u64)),
            None => "Outside filter".to_string(),
        };

        let locked = active.status == LabelImageStatus::Labeling;
        let busy = self.label.mutating;

        // Tool switch: Select (click box / drag empty to draw) vs Draw (drag anywhere to draw).
        let tool_idx = if self.label.editor.tool == EditTool::Draw { 1 } else { 0 };
        let tool_seg = self
            .segmented(
                "label-edit-tool",
                &["Select", "Draw"],
                tool_idx,
                window,
                cx,
                Rc::new(|this, idx, _, cx| {
                    this.label.editor.cancel_draw();
                    this.label.editor.cancel_transform();
                    this.label.editor.tool = if idx == 1 { EditTool::Draw } else { EditTool::Select };
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "Select boxes or drag on empty space to draw; Draw mode lets you drag overlapping boxes (D)",
                None,
            ));

        let undo_disabled = !can_undo || locked || busy;
        let redo_disabled = !can_redo || locked || busy;
        let next_unlabeled_disabled = busy;
        let dup_disabled = sel_box.is_none() || locked || busy;
        let prev_labeled = previous_labeled_frame(&all_images, &active.id);
        let copy_prev_disabled = prev_labeled.is_none() || locked || busy;
        let discard_disabled = !dirty || busy;
        let save_disabled = !dirty || locked || busy;
        for (key, disabled) in [
            ("label-undo", undo_disabled),
            ("label-redo", redo_disabled),
            ("label-next-unlabeled", next_unlabeled_disabled),
            ("label-dup-box", dup_disabled),
            ("label-copy-prev", copy_prev_disabled),
            ("label-discard", discard_disabled),
            ("label-save", save_disabled),
        ] {
            let dis_key = format!("{key}-disabled");
            if disabled && self.fh(key, cx).is_focused(window) {
                window.focus(&self.fh(dis_key, cx));
            } else if !disabled && self.fh(&dis_key, cx).is_focused(window) {
                window.focus(&self.fh(key, cx));
            }
        }
        let undo_depth = self.label.editor.undo.len();
        let redo_depth = self.label.editor.redo.len();
        let undo_btn = if undo_disabled {
            self.button_disabled(
                "label-undo",
                "Undo",
                Btn::Secondary,
                None,
                if locked { "Agent is labeling this image" } else { "Nothing to undo (Ctrl+Z)" },
                window,
                cx,
            )
        } else {
            let (j, im_id) = (jid.clone(), active.id.clone());
            self.button(
                "label-undo",
                "Undo",
                Btn::Secondary,
                None,
                window,
                cx,
                act(move |this, _, cx| this.undo_label_edit(&j, &im_id, cx)),
            )
            .tooltip(crate::controls::tip(format!("Undo ({undo_depth} available)"), Some("Ctrl Z")))
        };

        let redo_btn = if redo_disabled {
            self.button_disabled(
                "label-redo",
                "Redo",
                Btn::Secondary,
                None,
                if locked { "Agent is labeling this image" } else { "Nothing to redo (Ctrl+Shift+Z)" },
                window,
                cx,
            )
        } else {
            let (j, im_id) = (jid.clone(), active.id.clone());
            self.button(
                "label-redo",
                "Redo",
                Btn::Secondary,
                None,
                window,
                cx,
                act(move |this, _, cx| this.redo_label_edit(&j, &im_id, cx)),
            )
            .tooltip(crate::controls::tip(format!("Redo ({redo_depth} available)"), Some("Ctrl Shift Z")))
        };

        let discard_btn = if discard_disabled {
            self.button_disabled(
                "label-discard",
                "Discard",
                Btn::Secondary,
                Some("icons/x.svg"),
                if busy { "Saving…" } else { "No unsaved edits" },
                window,
                cx,
            )
        } else {
            let (j, im_id) = (jid.clone(), active.id.clone());
            self.button(
                "label-discard",
                "Discard",
                Btn::Secondary,
                Some("icons/x.svg"),
                window,
                cx,
                act(move |this, _, cx| this.discard_label_edits(&j, &im_id, cx)),
            )
            .tooltip(crate::controls::tip("Revert unsaved edits (undoable with Ctrl+Z)", None))
        };

        let save_btn = if save_disabled {
            self.button_disabled(
                "label-save",
                if busy { "Saving…" } else { "Save" },
                if dirty { Btn::Primary } else { Btn::Secondary },
                Some("icons/check.svg"),
                if !dirty {
                    "No unsaved edits (Ctrl+S)"
                } else if locked {
                    "Agent is labeling this image"
                } else {
                    "Saving…"
                },
                window,
                cx,
            )
        } else {
            let (j, im_id) = (jid.clone(), active.id.clone());
            self.button(
                "label-save",
                "Save",
                Btn::Primary,
                Some("icons/check.svg"),
                window,
                cx,
                act(move |this, _, cx| this.save_label_edits(&j, &im_id, false, cx)),
            )
            .tooltip(crate::controls::tip("Save box edits and stay on this image", Some("Ctrl S")))
        };

        let approve_label = if dirty {
            "Save & next"
        } else if active.status == LabelImageStatus::Labeled && active.reviewed {
            "Reviewed"
        } else {
            "Approve"
        };
        let approve_btn = if locked || busy {
            self.button_disabled(
                "label-approve",
                approve_label,
                if dirty || !active.reviewed { Btn::Primary } else { Btn::Secondary },
                Some("icons/check.svg"),
                if locked { "Agent is labeling this image" } else { "Saving…" },
                window,
                cx,
            )
        } else {
            let (j, im_id) = (jid.clone(), active.id.clone());
            self.button(
                "label-approve",
                approve_label,
                if !dirty && !active.reviewed { Btn::Primary } else { Btn::Secondary },
                Some("icons/check.svg"),
                window,
                cx,
                act(move |this, _, cx| this.approve_label_image(&j, &im_id, true, cx)),
            )
            .tooltip(crate::controls::tip(
                if dirty {
                    "Save edits, mark reviewed, and open next image"
                } else {
                    "Mark reviewed and open next image"
                },
                Some("A"),
            ))
        };

        let next_unlabeled_btn = if next_unlabeled_disabled {
            self.button_disabled(
                "label-next-unlabeled",
                "Next unlabeled",
                Btn::Secondary,
                Some("icons/target.svg"),
                "Saving…",
                window,
                cx,
            )
            .into_any_element()
        } else {
            let jid_unlabeled = jid.clone();
            self.button(
                "label-next-unlabeled",
                "Next unlabeled",
                Btn::Secondary,
                Some("icons/target.svg"),
                window,
                cx,
                act(move |this, _, cx| this.request_jump_next_unlabeled(&jid_unlabeled, cx)),
            )
            .tooltip(crate::controls::tip("Jump to the next pending or failed image in sequence", Some("U")))
            .into_any_element()
        };
        let jid_skip = jid.clone();
        let skip_btn = self.button(
            "label-skip",
            "Skip",
            Btn::Secondary,
            Some("icons/arrow-right.svg"),
            window,
            cx,
            act(move |this, _, cx| this.request_step_label_image(&jid_skip, 1, cx)),
        );
        let dup_box_btn = if dup_disabled {
            self.button_disabled(
                "label-dup-box",
                "Duplicate",
                Btn::Secondary,
                Some("icons/copy.svg"),
                if locked {
                    "Agent is labeling this image"
                } else if busy {
                    "Saving…"
                } else {
                    "Select a box to duplicate (Ctrl+D)"
                },
                window,
                cx,
            )
        } else {
            let (j, im_id) = (jid.clone(), active.id.clone());
            self.button(
                "label-dup-box",
                "Duplicate",
                Btn::Secondary,
                Some("icons/copy.svg"),
                window,
                cx,
                act(move |this, _, cx| this.duplicate_selected_label_box(&j, &im_id, cx)),
            )
            .tooltip(crate::controls::tip("Duplicate selected box with edge-aware offset", Some("Ctrl D")))
        };
        let copy_prev_btn = if copy_prev_disabled {
            self.button_disabled(
                "label-copy-prev",
                "Copy prev",
                Btn::Secondary,
                Some("icons/layers.svg"),
                if locked {
                    "Agent is labeling this image"
                } else if busy {
                    "Saving…"
                } else {
                    "No earlier labeled image with boxes to copy (P)"
                },
                window,
                cx,
            )
        } else {
            let (j, im_id) = (jid.clone(), active.id.clone());
            let prev_file = prev_labeled.map(|p| p.file.clone()).unwrap_or_default();
            self.button(
                "label-copy-prev",
                "Copy prev",
                Btn::Secondary,
                Some("icons/layers.svg"),
                window,
                cx,
                act(move |this, _, cx| this.copy_prev_frame_label_boxes(&j, &im_id, cx)),
            )
            .tooltip(crate::controls::tip(
                format!("Copy boxes from {prev_file} (scales if dimensions differ)"),
                Some("P"),
            ))
        };
        let boxes_btn = self.icon_button(
            "label-toggle-boxes",
            if self.label.show_boxes { "icons/eye.svg" } else { "icons/eye-off.svg" },
            "Toggle boxes (B)",
            window,
            cx,
            act(|this, _, cx| {
                this.label.show_boxes = !this.label.show_boxes;
                if !this.label.show_boxes {
                    this.label.editor.cancel_transform();
                }
                cx.notify();
            }),
        );
        let zoom_out = self.icon_button(
            "label-zoom-out",
            "icons/zoom-out.svg",
            "Zoom out (-)",
            window,
            cx,
            act(|this, _, cx| {
                (this.label.zoom, this.label.pan) = zoom_step(this.label.zoom, this.label.pan, 1.0 / 1.4);
                cx.notify();
            }),
        );
        let zoom_in = self.icon_button(
            "label-zoom-in",
            "icons/zoom-in.svg",
            "Zoom in (+)",
            window,
            cx,
            act(|this, _, cx| {
                (this.label.zoom, this.label.pan) = zoom_step(this.label.zoom, this.label.pan, 1.4);
                cx.notify();
            }),
        );

        let del_box_btn = match (sel_box, locked || busy) {
            (Some(_), false) => {
                let (j, im_id) = (jid.clone(), active.id.clone());
                self.button(
                    "label-del-box",
                    "Delete box",
                    Btn::Danger,
                    Some("icons/trash-2.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.delete_selected_label_box(&j, &im_id, cx)),
                )
            }
            _ => self.button_disabled(
                "label-del-box",
                "Delete box",
                Btn::Secondary,
                Some("icons/trash-2.svg"),
                "Select a box on the image or list to delete it",
                window,
                cx,
            ),
        };

        let img_status_badge = match (active.status, active.reviewed, active.source) {
            (LabelImageStatus::Labeled, true, Some(LabelSource::Manual)) => {
                badge(t, "Edited · Reviewed", Tone::Success)
            }
            (LabelImageStatus::Labeled, true, _) => badge(t, "Reviewed", Tone::Success),
            (LabelImageStatus::Labeled, false, _) => badge(t, "To review", Tone::Accent),
            (LabelImageStatus::Labeling, _, _) => badge(t, "Labeling…", Tone::Accent),
            (LabelImageStatus::Pending, _, _) => badge(t, "Pending", Tone::Neutral),
            (LabelImageStatus::Failed, _, _) => badge(t, "Failed", Tone::Danger),
        };
        let dirty_badge = dirty.then(|| badge(t, "Unsaved", Tone::Warning));

        // Two stable rows so toggling dirty state never wraps controls or shifts the image stage.
        let review_header = div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .px(px(14.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
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
                            .flex_1()
                            .min_w_0()
                            .child(prev_btn)
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .tnum()
                                    .whitespace_nowrap()
                                    .flex_none()
                                    .child(counter),
                            )
                            .child(next_btn)
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .min_w_0()
                                    .flex_shrink()
                                    .ellipsis()
                                    .max_w(px(240.))
                                    .child(active.file.clone()),
                            )
                            .child(div().flex_none().child(img_status_badge))
                            .children(dirty_badge.map(|b| div().flex_none().child(b))),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_none()
                            .items_center()
                            .gap(px(6.))
                            .child(next_unlabeled_btn)
                            .child(skip_btn)
                            .child(approve_btn),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(px(6.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(6.))
                            .child(tool_seg)
                            .child(undo_btn)
                            .child(redo_btn)
                            .child(dup_box_btn)
                            .child(copy_prev_btn)
                            .child(boxes_btn)
                            .child(zoom_out)
                            .child(zoom_in)
                            .child(del_box_btn),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_none()
                            .items_center()
                            .gap(px(6.))
                            .child(discard_btn)
                            .child(save_btn),
                    ),
            );

        // Error, conflict, and unsaved-switch banners.
        let notices = self.label_review_notices(
            t,
            &jid,
            &active,
            changed_elsewhere,
            has_pending_switch,
            locked,
            busy,
            window,
            cx,
        );

        // Bounding-box stage canvas and right-hand box rail.
        let stage = self.label_stage(t, job, &active, &draft_boxes, &draft_rev, sel_box, cx);
        let box_rail = self.label_box_list(
            t,
            job,
            &active,
            &draft_boxes,
            &draft_rev,
            sel_box,
            dirty,
            locked || busy,
            window,
            cx,
        );
        let filmstrip = self.label_filmstrip(t, &jid, &shown, &active.id, review_w, window, cx);

        let stage_area = div()
            .flex()
            .flex_col()
            .gap(px(12.))
            .p(px(14.))
            .child(
                div()
                    .flex()
                    .items_start()
                    .gap(px(12.))
                    .child(div().flex_1().min_w_0().h(px(360.)).child(stage))
                    .child(div().w(px(220.)).flex_none().h(px(360.)).child(box_rail)),
            )
            .child(filmstrip);

        let stage_panel = panel(t)
            .id("label-review-panel")
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(review_header)
            .children(notices)
            .child(stage_area);

        div().flex().flex_col().gap(px(12.)).child(filter_bar).child(stage_panel).into_any_element()
    }
}
