//! Inline forms for Agent Labeling: New labeling job, Add folder, and Export/Register as Dataset.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, FontWeight, IntoElement, ParentElement, Styled, Window, div, px,
};

use crate::api::LabelJob;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::label::{LABEL_VAL_FRACTIONS, parse_class_list};
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;

/// Single-line input box styled consistently with other Desktop inline forms.
fn form_input(
    t: &Theme,
    app: &FoloApp,
    input: folo_ui::Entity<crate::input::TextInput>,
    mono: bool,
    window: &Window,
    cx: &Context<FoloApp>,
) -> folo_ui::Div {
    let fh = input.read(cx).focus_handle(cx);
    let ring = app.ring(&fh, window);
    div()
        .relative()
        .w_full()
        .h(px(34.))
        .px(px(10.))
        .flex()
        .items_center()
        .rounded(px(radius::MD))
        .bg(t.bg_raised)
        .border_1()
        .border_color(if fh.is_focused(window) { t.accent_line } else { t.line_control })
        .when(mono, |d| d.font_family(crate::theme::MONO))
        .text_size(px(ty::SM.size))
        .when(ring, |d| d.child(crate::app::focus_ring(t, radius::MD)))
        .child(input)
}

fn inline_error(t: &Theme, msg: String, hint: Option<String>) -> folo_ui::Div {
    div()
        .flex()
        .items_start()
        .gap(px(8.))
        .p(px(10.))
        .rounded(px(radius::MD))
        .bg(t.danger_muted)
        .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(2.))
                .min_w_0()
                .child(text(ty::SM, t.danger_text).whitespace_normal().child(msg))
                .when_some(hint, |d, h| d.child(text(ty::SM, t.text_secondary).whitespace_normal().child(h))),
        )
}

impl FoloApp {
    /// Inline "New labeling job" card on the Labeling jobs list.
    pub(super) fn new_label_job_form(
        &self,
        t: &Theme,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(form) = self.label.new_job.as_ref() else {
            return div().into_any_element();
        };
        let classes_preview = parse_class_list(&form.classes.read(cx).text());
        let name_el = form_input(t, self, form.name.clone(), false, window, cx);
        let classes_el = form_input(t, self, form.classes.clone(), false, window, cx);
        let path_el = form_input(t, self, form.path.clone(), true, window, cx);
        let instr_el = form_input(t, self, form.instructions.clone(), false, window, cx);
        let browse = self.button(
            "label-new-browse",
            "Browse…",
            Btn::Secondary,
            Some("icons/folder-open.svg"),
            window,
            cx,
            act(|this, _, cx| this.browse_new_label_folder(cx)),
        );
        let cancel = self.button(
            "label-new-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_new_label_job(w, cx)),
        );
        let submit = if form.submitting {
            self.button_disabled(
                "label-new-submit",
                "Creating…",
                Btn::Primary,
                None,
                "Creating labeling job",
                window,
                cx,
            )
        } else {
            self.button(
                "label-new-submit",
                "Create job",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, _, cx| this.submit_new_label_job(cx)),
            )
        };
        let err = form.error.as_ref().map(|e| inline_error(t, e.message.clone(), e.hint.clone()));
        panel(t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(t, "New labeling job"))
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(12.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .gap(px(12.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(220.))
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("Name"),
                                    )
                                    .child(name_el),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(260.))
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("Classes (comma-separated)"),
                                    )
                                    .child(classes_el),
                            ),
                    )
                    .when(!classes_preview.is_empty(), |d| {
                        d.child(div().flex().flex_wrap().gap(px(6.)).children(
                            classes_preview.iter().enumerate().take(16).map(|(i, c)| {
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(5.))
                                    .h(px(22.))
                                    .px(px(8.))
                                    .rounded(px(radius::SM))
                                    .bg(t.bg_active)
                                    .child(div().size(px(7.)).rounded(px(2.)).bg(t.class(i as u32)))
                                    .child(text(ty::XS, t.text_primary).child(c.clone()))
                            }),
                        ))
                    })
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(4.))
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Source folder (optional)"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(div().flex_1().min_w_0().child(path_el))
                                    .child(browse),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(4.))
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Instructions for the agent (optional)"),
                            )
                            .child(instr_el),
                    )
                    .children(err)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_end()
                            .gap(px(8.))
                            .pt(px(4.))
                            .child(cancel)
                            .child(submit),
                    ),
            )
            .into_any_element()
    }

    /// Inline "Add image folder" card on a labeling job page.
    pub(super) fn add_label_folder_form(
        &self,
        t: &Theme,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(form) = self.label.add_folder.as_ref() else {
            return div().into_any_element();
        };
        let path_el = form_input(t, self, form.path.clone(), true, window, cx);
        let browse = self.button(
            "label-add-browse",
            "Browse…",
            Btn::Secondary,
            Some("icons/folder-open.svg"),
            window,
            cx,
            act(|this, _, cx| this.browse_add_label_folder(cx)),
        );
        let cancel = self.button(
            "label-add-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_add_label_folder(w, cx)),
        );
        let submit = if form.submitting {
            self.button_disabled(
                "label-add-submit",
                "Adding…",
                Btn::Primary,
                None,
                "Importing images",
                window,
                cx,
            )
        } else {
            self.button(
                "label-add-submit",
                "Add images",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, _, cx| this.submit_add_label_folder(cx)),
            )
        };
        let err = form.error.as_ref().map(|e| inline_error(t, e.message.clone(), e.hint.clone()));
        panel(t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(t, "Add images from folder"))
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .child(
                        text(ty::SM, t.text_tertiary).whitespace_normal().child(
                            "Images in the folder on the server's machine are read in place, not copied.",
                        ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(div().flex_1().min_w_0().child(path_el))
                            .child(browse),
                    )
                    .children(err)
                    .child(div().flex().justify_end().gap(px(8.)).child(cancel).child(submit)),
            )
            .into_any_element()
    }

    /// Inline "Create dataset" (Export / Register as Dataset) card on a labeling job page.
    pub(super) fn export_label_dataset_form(
        &self,
        t: &Theme,
        job: &LabelJob,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(form) = self.label.export_ds.as_ref() else {
            return div().into_any_element();
        };
        let name_el = form_input(t, self, form.name.clone(), false, window, cx);
        let val_seg = self.segmented(
            "label-export-val",
            &["10%", "20%", "30%"],
            form.val_idx.min(LABEL_VAL_FRACTIONS.len() - 1),
            window,
            cx,
            Rc::new(|this, idx, _, cx| {
                if let Some(f) = this.label.export_ds.as_mut() {
                    f.val_idx = idx.min(LABEL_VAL_FRACTIONS.len() - 1);
                    cx.notify();
                }
            }),
        );
        let frac = LABEL_VAL_FRACTIONS[form.val_idx.min(LABEL_VAL_FRACTIONS.len() - 1)];
        let labeled = job.counts.labeled;
        let val_n = ((labeled as f32 * frac).round() as u64).min(labeled.saturating_sub(1));
        let train_n = labeled.saturating_sub(val_n);
        let left = job.counts.pending + job.counts.failed;
        let summary = if left > 0 {
            format!(
                "{} train · {} val ({} boxes). {} unlabeled left out.",
                int(train_n),
                int(val_n),
                int(job.counts.boxes),
                int(left)
            )
        } else {
            format!("{} train · {} val ({} boxes).", int(train_n), int(val_n), int(job.counts.boxes))
        };
        let cancel = self.button(
            "label-export-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_export_label_dataset(w, cx)),
        );
        let submit = if form.submitting {
            self.button_disabled(
                "label-export-submit",
                "Exporting…",
                Btn::Primary,
                None,
                "Writing YOLO dataset",
                window,
                cx,
            )
        } else {
            self.button(
                "label-export-submit",
                "Export & register dataset",
                Btn::Primary,
                Some("icons/database.svg"),
                window,
                cx,
                act(|this, _, cx| this.submit_export_label_dataset(cx)),
            )
        };
        let err = form.error.as_ref().map(|e| inline_error(t, e.message.clone(), e.hint.clone()));
        let t_now = theme(cx);
        panel(&t_now)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t_now, "Export as YOLO dataset"))
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(12.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_end()
                            .gap(px(16.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(220.))
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("Dataset name"),
                                    )
                                    .child(name_el),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("Validation split"),
                                    )
                                    .child(val_seg),
                            ),
                    )
                    .child(text(ty::SM, t.text_secondary).whitespace_normal().child(summary))
                    .children(err)
                    .child(div().flex().justify_end().gap(px(8.)).child(cancel).child(submit)),
            )
            .into_any_element()
    }
}
