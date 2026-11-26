//! Inline External Predictions import form on a benchmark's Leaderboard tab: browse or paste
//! a JSON/JSONL predictions file, validate against the benchmark before upload, toggle
//! confidences, and submit (single-shot or chunked pass-by-pass).

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use crate::api::Bench;
use crate::app::{FoloApp, act};
use crate::bench::BenchForm;
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    /// Renders the inline External Predictions import form when open for `b`.
    pub(super) fn external_entry_form(
        &mut self,
        t: &Theme,
        b: &Bench,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let Some(f) = self.bench.external.as_ref().filter(|f| f.bench == b.id) else {
            return div().into_any_element();
        };
        let (file_path, name_ent, scored, reading, submitting, show_how, file_name, file_bytes) = (
            f.file_path.clone(),
            f.name.clone(),
            f.scored,
            f.reading,
            f.submitting,
            f.show_how,
            f.file_name.clone(),
            f.file_bytes,
        );
        let cur_path = file_path.read(cx).text().trim().to_string();
        let path_matches = f.checked_path.as_deref().map(str::trim) == Some(cur_path.as_str());
        let path_changed = f.checked_path.is_some() && !path_matches;
        let check_ok = if path_matches {
            f.check.as_ref().and_then(|r| r.as_ref().ok()).map(|c| c.summary.clone())
        } else {
            None
        };
        let check_err =
            if path_matches { f.check.as_ref().and_then(|r| r.as_ref().err()).cloned() } else { None };
        let score_err = if path_matches && scored {
            f.check
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .and_then(|c| crate::bench::validate_box_scores(&c.predictions).err())
        } else {
            None
        };
        let error = f.error.clone();
        let name_val = name_ent.read(cx).text().trim().to_string();

        let pick_btn = if reading || submitting {
            self.button_disabled(
                "bench-ext-pick",
                "Choose file…",
                Btn::Secondary,
                Some("icons/folder-open.svg"),
                "Wait for the current file operation to finish",
                window,
                cx,
            )
        } else {
            self.button(
                "bench-ext-pick",
                if file_name.is_some() { "Choose another…" } else { "Choose file…" },
                Btn::Secondary,
                Some("icons/folder-open.svg"),
                window,
                cx,
                act(|this, w, cx| this.pick_external_file(w, cx)),
            )
        };
        let check_btn = if reading || submitting {
            self.button_disabled(
                "bench-ext-check",
                "Checking…",
                Btn::Secondary,
                None,
                "Checking predictions file",
                window,
                cx,
            )
        } else {
            self.button(
                "bench-ext-check",
                "Check file",
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, _, cx| this.check_typed_external_file(cx)),
            )
        };

        let status_el: AnyElement = if reading {
            text(ty::SM, t.text_secondary)
                .child("Reading and validating predictions against this benchmark…")
                .into_any_element()
        } else if path_changed {
            div()
                .flex()
                .items_start()
                .gap(px(6.))
                .child(icon("icons/triangle-alert.svg", 14., t.warning_text))
                .child(
                    text(ty::SM, t.warning_text).whitespace_normal().child(
                        "File path changed — click Check file to validate the new file before scoring.",
                    ),
                )
                .into_any_element()
        } else if let Some(ref msg) = score_err {
            div()
                .flex()
                .items_start()
                .gap(px(6.))
                .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                .child(text(ty::SM, t.danger_text).whitespace_normal().child(msg.clone()))
                .into_any_element()
        } else if let Some(ref s) = check_ok {
            let cond_part = if s.conditions.is_empty() {
                "unchanged only".to_string()
            } else {
                format!("{} of {} conditions", s.conditions.len(), b.plan.conditions.len())
            };
            let score_part = if s.has_scores { "with confidences" } else { "no confidences" };
            let parts_note = if s.parts_count > 0 {
                format!(" · sent in {} parts", s.parts_count)
            } else {
                String::new()
            };
            let left_out = if !s.missing_conditions.is_empty() && !s.conditions.is_empty() {
                let names: Vec<String> =
                    s.missing_conditions.iter().map(|c| self.bench_condition_name(c)).collect();
                format!(" Left out: {} (partial).", names.join(", "))
            } else {
                String::new()
            };
            let size_tag = if file_bytes > 0 {
                let (v, u) = bytes(file_bytes);
                format!(" ({v} {u})")
            } else {
                String::new()
            };
            let summary_line = format!(
                "{} answered · {cond_part} · {} {score_part}{parts_note}{size_tag}.{left_out}",
                counted(s.images as u64, "image", "images"),
                counted(s.boxes as u64, "box", "boxes"),
            );
            div()
                .flex()
                .items_start()
                .gap(px(6.))
                .child(icon("icons/circle-check.svg", 14., t.success_text))
                .child(text(ty::SM, t.success_text).whitespace_normal().child(summary_line))
                .into_any_element()
        } else if let Some(ref msg) = check_err {
            div()
                .flex()
                .items_start()
                .gap(px(6.))
                .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                .child(text(ty::SM, t.danger_text).whitespace_normal().child(msg.clone()))
                .into_any_element()
        } else {
            text(ty::SM, t.text_tertiary)
                .whitespace_normal()
                .child("A JSON file of {\"predictions\": [...]}, a bare JSON array, or JSONL: one record per image and condition, boxes as corners (x1, y1, x2, y2) in pixels.")
                .into_any_element()
        };

        let scored_ctl = self.segmented(
            "bench-ext-scored",
            &["Uses confidences", "No confidences"],
            if scored { 0 } else { 1 },
            window,
            cx,
            std::rc::Rc::new(|this, i, _, cx| {
                this.set_external_scored(i == 0, cx);
            }),
        );
        let scored_hint = if scored {
            "Every box's score is read; boxes without one count as 1."
        } else {
            "Every score is treated as 1: calibration is not measured and self-labeling has one operating point."
        };

        let how_btn = self.button(
            "bench-ext-how",
            if show_how { "Hide helper command" } else { "How to generate predictions" },
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, _, cx| {
                if let Some(f) = this.bench.external.as_mut() {
                    f.show_how = !f.show_how;
                }
                cx.notify();
            }),
        );
        let raw_url = self.server_url.read(cx).text().trim().trim_end_matches('/').to_string();
        let server_url = if raw_url.is_empty() { "http://127.0.0.1:7878".to_string() } else { raw_url };
        let helper_cmd = format!(
            "python examples/bench_external.py --server {server_url} --bench {} --name my-detector",
            b.id
        );
        let copy_cmd = helper_cmd.clone();
        let how_box = show_how.then(|| {
            let copy = self.button(
                "bench-ext-copy-cmd",
                "Copy",
                Btn::Secondary,
                Some("icons/copy.svg"),
                window,
                cx,
                act(move |this, _, cx| {
                    cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(copy_cmd.clone()));
                    this.toast(Tone::Neutral, "Copied command", None, cx);
                }),
            );
            div()
                .p(px(12.))
                .rounded(px(radius::MD))
                .bg(t.bg_inset)
                .border_1()
                .border_color(t.line_subtle)
                .flex()
                .flex_col()
                .gap(px(8.))
                .child(text(ty::SM, t.text_secondary).whitespace_normal().child(
                    "Wrap any model behind examples/bench_external.py to fetch images under every condition and upload predictions in parts:",
                ))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(
                            text(ty::SM, t.text_primary)
                                .font_family(crate::theme::MONO)
                                .flex_1()
                                .min_w_0()
                                .ellipsis()
                                .child(helper_cmd),
                        )
                        .child(copy),
                )
        });

        let submit_why = if submitting {
            Some("Scoring and uploading predictions")
        } else if reading {
            Some("Checking predictions file")
        } else if path_changed {
            Some("File path changed — click Check file first")
        } else if score_err.is_some() {
            Some("Fix box scores or switch to No confidences")
        } else if check_ok.is_none() {
            Some("Choose and validate a predictions file first")
        } else if name_val.is_empty() {
            Some("Enter a name for the leaderboard")
        } else {
            None
        };
        let submit = match submit_why {
            Some(why) => self.button_disabled(
                "bench-ext-submit",
                if submitting { "Scoring…" } else { "Score and add" },
                Btn::Primary,
                Some("icons/plus.svg"),
                why,
                window,
                cx,
            ),
            None => self.button(
                "bench-ext-submit",
                "Score and add",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, _, cx| this.submit_external_entry(cx)),
            ),
        };
        let cancel = self.button(
            "bench-ext-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_bench_form(BenchForm::External, w, cx)),
        );

        let field = |label: &str, control: AnyElement| {
            div().flex().flex_col().gap(px(6.)).child(caption(&t, label)).child(control)
        };
        panel(&t)
            .child(
                panel_header(&t, "Import external predictions")
                    .child(text(ty::SM, t.text_tertiary).child("Any model run elsewhere, scored here")),
            )
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(14.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(8.))
                            .child(pick_btn)
                            .child(div().flex_1().min_w(px(220.)).child(file_path))
                            .child(check_btn),
                    )
                    .child(status_el)
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_start()
                            .gap(px(16.))
                            .child(field(
                                "Name on the leaderboard",
                                div().w(px(240.)).child(name_ent).into_any_element(),
                            ))
                            .child(field("Confidences", scored_ctl.into_any_element())),
                    )
                    .child(text(ty::SM, t.text_tertiary).whitespace_normal().child(scored_hint))
                    .child(div().flex().child(how_btn))
                    .children(how_box)
                    .when_some(error, |d, e| {
                        d.child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                                .child(text(ty::SM, t.danger_text).whitespace_normal().child(e.message)),
                        )
                        .when_some(e.hint, |d, h| {
                            d.child(text(ty::SM, t.text_tertiary).whitespace_normal().child(h))
                        })
                    })
                    .child(div().flex().gap(px(8.)).child(submit).child(cancel)),
            )
            .into_any_element()
    }
}
