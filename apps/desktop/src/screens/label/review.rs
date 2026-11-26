//! Image review workspace for Agent Labeling: status & class filters, bounding-box canvas stage,
//! box inspection/relabel/delete controls, quick keyboard review actions, and image filmstrip.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, MouseButton, MouseDownEvent,
    ParentElement, SharedString, StatefulInteractiveElement, Styled, Window, canvas, div, px,
};

use crate::api::{LabelImage, LabelImageStatus, LabelJob, LabelSource};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::label::{LabelImageFilter, filter_label_images};
use crate::screens::loading_rows;
use crate::store::ImgKind;
use crate::theme::{Theme, alpha, radius, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, hit_box, paint_image_with_boxes, to_image, zoom_step};

impl FoloApp {
    /// Main column of `Route::LabelJob`: filter bar, review stage, box list, and filmstrip.
    pub(super) fn label_review_workspace(
        &mut self,
        t: &Theme,
        job: &LabelJob,
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
                this.label.img_filter = buckets[idx.min(buckets.len() - 1)];
                this.reconcile_selected_label_image(&jid_filter);
                cx.notify();
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
                act(move |this, _, cx| {
                    this.label.class_filter = None;
                    this.reconcile_selected_label_image(&jid_all_cls);
                    cx.notify();
                }),
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
                        this.label.class_filter =
                            if this.label.class_filter == Some(cls_idx) { None } else { Some(cls_idx) };
                        this.reconcile_selected_label_image(&jid_cls);
                        cx.notify();
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
                        text(ty::XS, t.text_tertiary)
                            .child("Keys: ←/→ step · A approve · S skip · Del box · B/L toggles"),
                    ),
            )
            .child(div().flex().flex_wrap().items_center().gap(px(6.)).children(class_chips));

        if shown.is_empty() {
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
                    this.reconcile_selected_label_image(&jid_reset);
                    cx.notify();
                }),
            );
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
                        .child(reset),
                    ),
                )
                .into_any_element();
        }

        // Resolve the active image inside `shown`.
        let cur_pos = self
            .label
            .selected_image
            .as_deref()
            .and_then(|sel| shown.iter().position(|im| im.id == sel))
            .unwrap_or(0);
        let active = shown[cur_pos].clone();
        if self.label.selected_image.as_deref() != Some(active.id.as_str()) {
            self.label.selected_image = Some(active.id.clone());
            self.label.selected_box = None;
        }
        let sel_box = self.label.selected_box.filter(|&i| i < active.boxes.len());
        if sel_box != self.label.selected_box {
            self.label.selected_box = sel_box;
        }

        // Toolbar above the image stage.
        let jid_prev = jid.clone();
        let prev_btn = self.icon_button(
            "label-prev",
            "icons/chevron-left.svg",
            "Previous image (←)",
            window,
            cx,
            act(move |this, _, cx| this.step_label_image(&jid_prev, -1, cx)),
        );
        let jid_next = jid.clone();
        let next_btn = self.icon_button(
            "label-next",
            "icons/chevron-right.svg",
            "Next image (→)",
            window,
            cx,
            act(move |this, _, cx| this.step_label_image(&jid_next, 1, cx)),
        );
        let counter = format!("{} of {}", int(cur_pos as u64 + 1), int(shown.len() as u64));

        let locked = active.status == LabelImageStatus::Labeling;
        let busy = self.label.mutating;
        let approve_label = if active.status == LabelImageStatus::Labeled && active.reviewed {
            "Reviewed"
        } else {
            "Approve"
        };
        let approve_btn = if locked || busy {
            self.button_disabled(
                "label-approve",
                approve_label,
                Btn::Primary,
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
                if active.reviewed { Btn::Secondary } else { Btn::Primary },
                Some("icons/check.svg"),
                window,
                cx,
                act(move |this, _, cx| this.approve_label_image(&j, &im_id, true, cx)),
            )
        };
        let jid_skip = jid.clone();
        let skip_btn = self.button(
            "label-skip",
            "Skip",
            Btn::Secondary,
            Some("icons/arrow-right.svg"),
            window,
            cx,
            act(move |this, _, cx| this.step_label_image(&jid_skip, 1, cx)),
        );
        let boxes_btn = self.icon_button(
            "label-toggle-boxes",
            if self.label.show_boxes { "icons/eye.svg" } else { "icons/eye-off.svg" },
            "Toggle boxes (B)",
            window,
            cx,
            act(|this, _, cx| {
                this.label.show_boxes = !this.label.show_boxes;
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

        let review_header = div()
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(8.))
            .px(px(14.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .min_w_0()
                    .child(prev_btn)
                    .child(text(ty::SM, t.text_secondary).tnum().child(counter))
                    .child(next_btn)
                    .child(
                        text(ty::SM, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .ellipsis()
                            .max_w(px(220.))
                            .child(active.file.clone()),
                    )
                    .child(img_status_badge),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(6.))
                    .child(boxes_btn)
                    .child(zoom_out)
                    .child(zoom_in)
                    .child(del_box_btn)
                    .child(skip_btn)
                    .child(approve_btn),
            );

        // Error banners (image-level agent error or mutation conflict error).
        let mut notices = Vec::new();
        if let Some(err) = &active.error {
            notices.push(
                div()
                    .mx(px(14.))
                    .mt(px(10.))
                    .p(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.danger_muted)
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
                    .child(text(ty::SM, t.danger_text).whitespace_normal().child(err.clone())),
            );
        }
        if let Some(err) = &self.label.review_error {
            notices.push(
                div()
                    .mx(px(14.))
                    .mt(px(10.))
                    .p(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.warning_muted)
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(icon("icons/triangle-alert.svg", 14., t.warning_solid))
                    .child(text(ty::SM, t.warning_text).whitespace_normal().child(err.message.clone())),
            );
        }

        // Bounding-box stage canvas.
        let stage = self.label_stage(t, job, &active, sel_box, cx);
        let box_rail = self.label_box_list(t, job, &active, sel_box, locked || busy, window, cx);
        let filmstrip = self.label_filmstrip(t, &jid, &shown, &active.id, window, cx);

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

    fn label_stage(
        &mut self,
        t: &Theme,
        job: &LabelJob,
        active: &LabelImage,
        sel_box: Option<usize>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let img = self
            .label_image_px(&job.id, &active.id, ImgKind::Full, cx)
            .or_else(|| self.label_image_px(&job.id, &active.id, ImgKind::Thumb, cx));
        let failed = img.is_none() && self.label_image_failed(&job.id, &active.id, ImgKind::Full);
        let dims = (active.width.max(1) as f32, active.height.max(1) as f32);
        let class_filter = self.label.class_filter;
        let hover = self.label.hover_box.or(sel_box);
        let draw: Vec<DrawBox> = if self.label.show_boxes {
            active
                .boxes
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let dim_by_filter = class_filter.is_some_and(|c| c != b.class);
                    let dim_by_hover = hover.is_some_and(|h| h != i);
                    DrawBox {
                        label: job.classes.get(b.class as usize).cloned().map(Into::into),
                        highlight: sel_box == Some(i) || self.label.hover_box == Some(i),
                        dim: dim_by_filter || dim_by_hover,
                        ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let draw_for_hit = draw.clone();
        let t_copy = *t;
        let zoom = self.label.zoom;
        let pan = self.label.pan;
        let show_labels = self.label.show_labels;
        let view_cell = self.label.stage_view.clone();
        let view_for_click = self.label.stage_view.clone();
        let canvas_el = canvas(
            move |b, _, _| {
                view_cell.set(Some(b));
            },
            move |b, _, window, cx| {
                paint_image_with_boxes(
                    &t_copy,
                    window,
                    cx,
                    b,
                    img.as_ref(),
                    dims,
                    &draw,
                    Fit::Contain,
                    zoom,
                    pan,
                    show_labels,
                    true,
                    0.,
                );
            },
        )
        .size_full();

        div()
            .id("label-stage")
            .relative()
            .size_full()
            .rounded(px(radius::MD))
            .bg(t.bg_app)
            .border_1()
            .border_color(t.line_subtle)
            .overflow_hidden()
            .cursor_pointer()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &MouseDownEvent, _, cx| {
                    if let Some(b) = view_for_click.get() {
                        let (ix, iy) =
                            to_image(b, dims, Fit::Contain, this.label.zoom, this.label.pan, ev.position);
                        this.label.selected_box = hit_box(&draw_for_hit, ix, iy);
                        cx.notify();
                    }
                }),
            )
            .child(canvas_el)
            .when(failed, |d| d.child(crate::screens::unavailable_mark(t, true)))
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn label_box_list(
        &self,
        t: &Theme,
        job: &LabelJob,
        active: &LabelImage,
        sel_box: Option<usize>,
        disabled: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let jid = job.id.clone();
        let img_id = active.id.clone();
        let confirming_img = self.label.confirm_delete_image.as_ref() == Some(&(jid.clone(), img_id.clone()));
        let remove_img_btn = if disabled {
            self.button_disabled("label-rm-img", "Remove image", Btn::Secondary, None, "Busy", window, cx)
        } else if confirming_img {
            let (j, i) = (jid.clone(), img_id.clone());
            self.button(
                "label-rm-img",
                "Confirm remove",
                Btn::Danger,
                Some("icons/trash-2.svg"),
                window,
                cx,
                act(move |this, _, cx| this.delete_label_image(&j, &i, cx)),
            )
        } else {
            let (j, i) = (jid.clone(), img_id.clone());
            self.button(
                "label-rm-img",
                "Remove image",
                Btn::Secondary,
                Some("icons/trash-2.svg"),
                window,
                cx,
                act(move |this, _, cx| {
                    this.label.confirm_delete_image = Some((j.clone(), i.clone()));
                    cx.notify();
                }),
            )
        };

        let rows: Vec<AnyElement> = active
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let cls_name = job
                    .classes
                    .get(b.class as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("class {}", b.class));
                let is_sel = sel_box == Some(i);
                let coords = format!("{:.0}×{:.0}", (b.x2 - b.x1).max(0.0), (b.y2 - b.y1).max(0.0));
                let row = div()
                    .id(("label-box", i))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .px(px(8.))
                    .h(px(30.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .when(is_sel, |d| d.bg(t.bg_active))
                    .when(!is_sel, |d| d.hover(|s| s.bg(t.bg_hover)))
                    .child(div().size(px(8.)).rounded(px(2.)).flex_none().bg(t.class(b.class)))
                    .child(
                        text(ty::SM, if is_sel { t.text_primary } else { t.text_secondary })
                            .font_weight(if is_sel { FontWeight::MEDIUM } else { FontWeight::NORMAL })
                            .flex_1()
                            .min_w_0()
                            .ellipsis()
                            .child(cls_name),
                    )
                    .child(text(ty::XS, t.text_tertiary).tnum().child(coords));
                self.bind(
                    row,
                    &format!("label-box-{i}"),
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.label.selected_box =
                            if this.label.selected_box == Some(i) { None } else { Some(i) };
                        cx.notify();
                    }),
                )
                .into_any_element()
            })
            .collect();

        // Quick relabel chips when a box is selected.
        let relabel = sel_box.filter(|_| !disabled).map(|_| {
            let chips = job.classes.iter().enumerate().take(9).map(|(ci, name)| {
                let (j, i) = (jid.clone(), img_id.clone());
                let cls_u32 = ci as u32;
                let cur = sel_box.and_then(|s| active.boxes.get(s)).is_some_and(|b| b.class == cls_u32);
                self.chip(
                    &format!("label-recls-{ci}"),
                    format!("{} {name}", ci + 1),
                    cur,
                    Some(cls_u32),
                    None,
                    window,
                    cx,
                    act(move |this, _, cx| this.reclassify_selected_label_box(&j, &i, cls_u32, cx)),
                )
                .into_any_element()
            });
            div()
                .p(px(8.))
                .border_t_1()
                .border_color(t.line_subtle)
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(overline(t, "Reclassify (1–9)"))
                .child(div().flex().flex_wrap().gap(px(4.)).children(chips))
        });

        div()
            .size_full()
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(t.line_subtle)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(
                div()
                    .px(px(10.))
                    .h(px(32.))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(overline(t, "Boxes"))
                    .child(text(ty::XS, t.text_tertiary).tnum().child(active.boxes.len().to_string())),
            )
            .child(
                div()
                    .id("label-boxes-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.label.boxes_scroll)
                    .p(px(6.))
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .when(rows.is_empty(), |d| {
                        d.child(div().p(px(12.)).child(
                            text(ty::XS, t.text_tertiary).whitespace_normal().child(
                                if active.status == LabelImageStatus::Labeled {
                                    "No boxes (background image)."
                                } else {
                                    "Not labeled yet."
                                },
                            ),
                        ))
                    })
                    .children(rows),
            )
            .children(relabel)
            .child(
                div()
                    .p(px(6.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .flex()
                    .justify_end()
                    .child(remove_img_btn),
            )
            .into_any_element()
    }

    fn label_filmstrip(
        &mut self,
        t: &Theme,
        job_id: &str,
        shown: &[LabelImage],
        active_id: &str,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        // Show a focused window of up to 12 thumbnails centered around the active image so the strip
        // stays fast and never overflows vertically.
        let cur = shown.iter().position(|im| im.id == active_id).unwrap_or(0);
        let max_tiles = 10usize;
        let start = cur.saturating_sub(max_tiles / 2).min(shown.len().saturating_sub(max_tiles));
        let end = (start + max_tiles).min(shown.len());
        let tiles: Vec<AnyElement> = shown[start..end]
            .iter()
            .map(|im| {
                let is_cur = im.id == active_id;
                let thumb = self.label_image_px(job_id, &im.id, ImgKind::Thumb, cx);
                let dims = (im.width.max(1) as f32, im.height.max(1) as f32);
                let boxes: Vec<DrawBox> =
                    im.boxes.iter().map(|b| DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)).collect();
                let dot_color = match (im.status, im.reviewed) {
                    (LabelImageStatus::Labeled, true) => t.success_solid,
                    (LabelImageStatus::Labeled, false) => t.accent_solid,
                    (LabelImageStatus::Failed, _) => t.danger_solid,
                    _ => t.text_disabled,
                };
                let target_id = im.id.clone();
                let key = format!("label-thumb-{}", im.id);
                let el = div()
                    .id(SharedString::from(key.clone()))
                    .w(px(72.))
                    .h(px(54.))
                    .flex_none()
                    .relative()
                    .rounded(px(radius::SM))
                    .overflow_hidden()
                    .cursor_pointer()
                    .border_2()
                    .border_color(if is_cur { t.accent_solid } else { alpha(t.line_subtle, 0.8) })
                    .child(crate::screens::image_canvas(t, thumb, dims, boxes, Fit::Cover, false, false))
                    .child(
                        div().absolute().top(px(4.)).right(px(4.)).size(px(7.)).rounded_full().bg(dot_color),
                    );
                self.bind(
                    el,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| this.select_label_image(target_id.clone(), cx)),
                )
                .into_any_element()
            })
            .collect();

        div().flex().items_center().gap(px(6.)).overflow_hidden().children(tiles).into_any_element()
    }
}
