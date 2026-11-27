//! Right-hand box list and class picker rail inside the Agent Labeling review workspace.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{BBox, LabelImage, LabelImageStatus, LabelJob};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    /// Box list rail: draft boxes, active draw/relabel class picker (`Select` + `1–9` chips), and
    /// two-click Remove image action.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn label_box_list(
        &self,
        t: &Theme,
        job: &LabelJob,
        active: &LabelImage,
        draft_boxes: &[BBox],
        draft_rev: &str,
        sel_box: Option<usize>,
        dirty: bool,
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

        let rows: Vec<AnyElement> = draft_boxes
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
                let img_for_row = img_id.clone();
                let rev_for_row = draft_rev.to_string();
                self.bind(
                    row,
                    &format!("label-box-{i}"),
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.toggle_label_box(img_for_row.clone(), i, rev_for_row.clone());
                        cx.notify();
                    }),
                )
                .into_any_element()
            })
            .collect();

        // Class picker: sets active draw class, and when a box is selected reclassifies it too.
        // Uses a compact Select dropdown for > 9 classes (avoiding crowding the 360 px rail), or
        // quick 1–9 chips for <= 9 classes.
        let class_picker = (!disabled && !job.classes.is_empty()).then(|| {
            let active_cls = sel_box
                .and_then(|s| draft_boxes.get(s).map(|b| b.class))
                .unwrap_or_else(|| self.label.editor.clamped_draw_class(job.classes.len()))
                as usize;
            let heading = if sel_box.is_some() { "Reclassify (1–9)" } else { "Draw class (1–9)" };

            let picker_body: AnyElement = if job.classes.len() > 9 {
                let options: Vec<(SharedString, Option<SharedString>)> = job
                    .classes
                    .iter()
                    .enumerate()
                    .map(|(ci, name)| {
                        let hint = (ci < 9).then(|| SharedString::from(format!("Key {}", ci + 1)));
                        (SharedString::from(name.clone()), hint)
                    })
                    .collect();
                let current_name =
                    job.classes.get(active_cls).cloned().unwrap_or_else(|| format!("class {active_cls}"));
                let (j, i) = (jid.clone(), img_id.clone());
                self.select(
                    "label-class-picker",
                    current_name.into(),
                    options,
                    active_cls,
                    window,
                    cx,
                    Rc::new(move |this, idx, _, cx| {
                        this.set_label_draw_class(&j, &i, idx as u32, cx);
                    }),
                )
                .into_any_element()
            } else {
                let chips = job.classes.iter().enumerate().map(|(ci, name)| {
                    let (j, i) = (jid.clone(), img_id.clone());
                    let cls_u32 = ci as u32;
                    let cur = active_cls == ci;
                    self.chip(
                        &format!("label-recls-{ci}"),
                        format!("{} {name}", ci + 1),
                        cur,
                        Some(cls_u32),
                        None,
                        window,
                        cx,
                        act(move |this, _, cx| this.set_label_draw_class(&j, &i, cls_u32, cx)),
                    )
                    .into_any_element()
                });
                div().flex().flex_wrap().gap(px(4.)).children(chips).into_any_element()
            };
            div()
                .flex_none()
                .p(px(8.))
                .border_t_1()
                .border_color(t.line_subtle)
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(overline(t, heading))
                .child(picker_body)
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
                    .child(overline(t, if dirty { "Boxes · edited" } else { "Boxes" }))
                    .child(text(ty::XS, t.text_tertiary).tnum().child(draft_boxes.len().to_string())),
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
                                    "No boxes. Drag on the image to draw one."
                                } else {
                                    "Not labeled yet. Drag on the image to draw a box, or run an agent."
                                },
                            ),
                        ))
                    })
                    .children(rows),
            )
            .children(class_picker)
            .child(
                div()
                    .flex_none()
                    .p(px(6.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .flex()
                    .justify_end()
                    .child(remove_img_btn),
            )
            .into_any_element()
    }
}
