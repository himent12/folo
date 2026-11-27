//! Collapsible per-class confidence threshold controls for the Playground side rail:
//! individual class threshold adjustments, single-class & reset-all actions, and
//! `ThresholdCalibration` JSON import/copy/export.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use super::thresholds::{effective_conf, fmt_threshold, round_conf};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

/// Compact collapsible Per-class thresholds controls inside Playground's side rail.
impl FoloApp {
    /// Compact inline toggle placed on the left of the row below the Confidence slider so
    /// collapsed per-class thresholds add zero extra vertical height to the 800 px rail.
    pub(super) fn play_per_class_toggle(
        &mut self,
        t: &Theme,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let override_count = self.play_thresholds.override_count();
        let expanded = self.play_thresholds.expanded;
        let count_badge = (override_count > 0).then(|| {
            badge(
                t,
                if override_count == 1 { "1".to_string() } else { override_count.to_string() },
                Tone::Accent,
            )
        });
        let el = div()
            .id("play-per-class-toggle")
            .h(px(22.))
            .ml(px(-4.))
            .px(px(4.))
            .rounded(px(radius::SM))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .flex()
            .items_center()
            .gap(px(4.))
            .tooltip(crate::controls::tip(
                "Adjust per-class confidence thresholds or import/export calibrated JSON",
                None,
            ))
            .child(icon(
                if expanded { "icons/chevron-down.svg" } else { "icons/chevron-right.svg" },
                12.,
                t.text_tertiary,
            ))
            .child(
                text(ty::SM, if override_count > 0 || expanded { t.text_primary } else { t.text_secondary })
                    .font_weight(FontWeight::MEDIUM)
                    .child("Per-class"),
            )
            .children(count_badge);
        self.bind(
            el,
            "play-per-class-toggle",
            radius::SM,
            window,
            cx,
            act(|this, _, cx| {
                this.play_thresholds.expanded = !this.play_thresholds.expanded;
                cx.notify();
            }),
        )
        .into_any_element()
    }

    /// Expanded per-class threshold drawer (actions, JSON import/export, class list).
    pub(super) fn play_per_class_drawer(
        &mut self,
        t: &Theme,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        if !self.play_thresholds.expanded {
            return None;
        }
        let classes: Vec<String> = self.play_shown_model().map(|m| m.classes.clone()).unwrap_or_default();
        let override_count = self.play_thresholds.override_count();

        // Action row: Import JSON, Copy JSON, Export JSON, Reset all.
        let import_open = self.play_thresholds.import_open;
        let import_btn = self
            .button(
                "play-thr-import",
                if import_open { "Hide import" } else { "Import" },
                Btn::Secondary,
                Some("icons/file-text.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    this.play_thresholds.import_open = !this.play_thresholds.import_open;
                    this.play_thresholds.import_error = None;
                    if this.play_thresholds.import_open {
                        let fh = this.play_thresh_json.read(cx).focus_handle(cx);
                        w.focus(&fh);
                    }
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "Paste or load ThresholdCalibration JSON from `folo val --save-thresholds`",
                None,
            ));
        let copy_btn = self
            .button(
                "play-thr-copy",
                "Copy",
                Btn::Secondary,
                Some("icons/copy.svg"),
                window,
                cx,
                act(|this, _, cx| this.copy_play_thresholds_json(cx)),
            )
            .tooltip(crate::controls::tip(
                "Copy ThresholdCalibration JSON (compatible with `folo predict --thresholds`)",
                None,
            ));
        let export_btn = self
            .button(
                "play-thr-export",
                "Export",
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(|this, _, cx| this.export_play_thresholds_json(cx)),
            )
            .tooltip(crate::controls::tip(
                "Save ThresholdCalibration JSON file for `folo predict --thresholds`",
                None,
            ));
        let reset_all_btn: AnyElement = if override_count > 0 {
            self.button(
                "play-thr-reset-all",
                "Reset all",
                Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    this.reset_all_play_class_conf(cx);
                    this.focus_key("play-per-class-toggle", w, cx);
                }),
            )
            .into_any_element()
        } else {
            self.button_disabled(
                "play-thr-reset-all",
                "Reset all",
                Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                "No per-class overrides active",
                window,
                cx,
            )
            .into_any_element()
        };

        let import_box = import_open.then(|| {
            let err = self.play_thresholds.import_error.clone();
            let apply = self.button(
                "play-thr-apply",
                "Apply",
                Btn::Primary,
                None,
                window,
                cx,
                act(|this, w, cx| {
                    let raw = this.play_thresh_json.read(cx).text();
                    if this.apply_play_thresholds_json(&raw, cx) {
                        this.focus_key("play-thr-import", w, cx);
                    }
                }),
            );
            let load_file = self.button(
                "play-thr-load-file",
                "Load .json",
                Btn::Secondary,
                Some("icons/folder-open.svg"),
                window,
                cx,
                act(|this, w, cx| this.pick_play_thresholds_file(w, cx)),
            );
            let cancel = self.button(
                "play-thr-cancel",
                "Cancel",
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, w, cx| {
                    this.play_thresh_import_seq += 1;
                    this.play_thresholds.import_open = false;
                    this.play_thresholds.import_error = None;
                    this.focus_key("play-thr-import", w, cx);
                    cx.notify();
                }),
            );
            div()
                .p(px(8.))
                .rounded(px(radius::MD))
                .bg(t.bg_inset)
                .border_1()
                .border_color(t.line_subtle)
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(self.play_thresh_json.clone())
                .when_some(err, |d, msg| d.child(text(ty::XS, t.danger_text).whitespace_normal().child(msg)))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(px(6.))
                        .child(apply)
                        .child(load_file)
                        .child(cancel),
                )
        });

        let note = self.play_thresholds.import_note.clone().map(|msg| {
            div()
                .px(px(8.))
                .py(px(5.))
                .rounded(px(radius::SM))
                .bg(t.accent_muted)
                .child(text(ty::XS, t.accent_text).whitespace_normal().child(msg))
        });

        let filter_query = if classes.len() > 8 {
            self.play_thresh_filter.read(cx).text().trim().to_ascii_lowercase()
        } else {
            String::new()
        };
        let filter_box = (classes.len() > 8).then(|| div().child(self.play_thresh_filter.clone()));

        let mut class_ids: Vec<(u32, String)> =
            classes.iter().enumerate().map(|(i, n)| (i as u32, n.clone())).collect();
        for &cid in self.play_thresholds.per_class.keys() {
            if (cid as usize) >= classes.len() {
                class_ids.push((cid, format!("class {cid}")));
            }
        }
        if !filter_query.is_empty() {
            class_ids.retain(|(cid, name)| {
                name.to_ascii_lowercase().contains(&filter_query) || cid.to_string() == filter_query
            });
        }

        let global_conf = self.play_conf;
        let rows: Vec<AnyElement> = class_ids
            .into_iter()
            .map(|(cid, name)| {
                let overridden = self.play_thresholds.per_class.contains_key(&cid);
                let eff = effective_conf(global_conf, &self.play_thresholds.per_class, cid);
                let dec_key = format!("play-cls-thr-dec-{cid}");
                let inc_key = format!("play-cls-thr-inc-{cid}");
                let rst_key = format!("play-cls-thr-rst-{cid}");
                let dec = self.bind(
                    div()
                        .id(SharedString::from(dec_key.clone()))
                        .size(px(20.))
                        .rounded(px(radius::SM))
                        .bg(t.bg_raised)
                        .border_1()
                        .border_color(t.line_control)
                        .cursor_pointer()
                        .hover(|s| s.bg(t.bg_hover))
                        .active(|s| s.bg(t.bg_active))
                        .flex()
                        .items_center()
                        .justify_center()
                        .tooltip(crate::controls::tip(format!("Lower {name} threshold (−0.05)"), None))
                        .child(text(ty::XS, t.text_primary).child("−")),
                    &dec_key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        let cur = effective_conf(this.play_conf, &this.play_thresholds.per_class, cid);
                        let next = round_conf((cur - 0.05).max(0.0));
                        this.set_play_class_conf(cid, next, cx);
                    }),
                );
                let at_max = eff >= 1.0 - 1e-6;
                let inc: AnyElement = if at_max {
                    div()
                        .id(SharedString::from(inc_key.clone()))
                        .size(px(20.))
                        .rounded(px(radius::SM))
                        .bg(t.bg_raised)
                        .border_1()
                        .border_color(t.line_subtle)
                        .opacity(0.45)
                        .cursor_not_allowed()
                        .flex()
                        .items_center()
                        .justify_center()
                        .tooltip(crate::controls::tip("Already at maximum threshold (1.00)", None))
                        .child(text(ty::XS, t.text_tertiary).child("+"))
                        .into_any_element()
                } else {
                    self.bind(
                        div()
                            .id(SharedString::from(inc_key.clone()))
                            .size(px(20.))
                            .rounded(px(radius::SM))
                            .bg(t.bg_raised)
                            .border_1()
                            .border_color(t.line_control)
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .flex()
                            .items_center()
                            .justify_center()
                            .tooltip(crate::controls::tip(format!("Raise {name} threshold (+0.05)"), None))
                            .child(text(ty::XS, t.text_primary).child("+")),
                        &inc_key,
                        radius::SM,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            let cur = effective_conf(this.play_conf, &this.play_thresholds.per_class, cid);
                            let next = round_conf((cur + 0.05).min(1.0));
                            this.set_play_class_conf(cid, next, cx);
                        }),
                    )
                    .into_any_element()
                };
                let rst = overridden.then(|| {
                    self.bind(
                        div()
                            .id(SharedString::from(rst_key.clone()))
                            .size(px(20.))
                            .rounded(px(radius::SM))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .flex()
                            .items_center()
                            .justify_center()
                            .tooltip(crate::controls::tip(
                                format!("Reset {name} to global ({})", fmt_threshold(global_conf)),
                                None,
                            ))
                            .child(icon("icons/x.svg", 11., t.text_tertiary)),
                        &rst_key,
                        radius::SM,
                        window,
                        cx,
                        act(move |this, w, cx| {
                            this.clear_play_class_conf(cid, cx);
                            let target = if this.play_conf >= 1.0 - 1e-6 {
                                format!("play-cls-thr-dec-{cid}")
                            } else {
                                format!("play-cls-thr-inc-{cid}")
                            };
                            this.focus_key(&target, w, cx);
                        }),
                    )
                });
                let bar_color =
                    if overridden { t.class_on(cid, crate::theme::Surface::Panel) } else { t.text_tertiary };
                div()
                    .h(px(28.))
                    .px(px(6.))
                    .rounded(px(radius::SM))
                    .when(overridden, |d| d.bg(t.bg_inset))
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(class_swatch(t, cid, 8.))
                    .child(
                        text(ty::XS, if overridden { t.text_primary } else { t.text_secondary })
                            .w(px(68.))
                            .flex_none()
                            .ellipsis()
                            .child(name),
                    )
                    .child(
                        div().flex_1().min_w(px(24.)).h(px(4.)).rounded_full().bg(t.line_subtle).child(
                            div().h_full().w(relative(eff.clamp(0.0, 1.0))).rounded_full().bg(bar_color),
                        ),
                    )
                    .child(dec)
                    .child(
                        text(ty::XS, if overridden { t.accent_text } else { t.text_tertiary })
                            .tnum()
                            .w(px(34.))
                            .text_right()
                            .flex_none()
                            .child(fmt_threshold(eff)),
                    )
                    .child(inc)
                    .children(rst)
                    .into_any_element()
            })
            .collect();

        let list_body: AnyElement = if rows.is_empty() {
            let msg = if classes.is_empty() {
                "Select a model to customize class thresholds."
            } else {
                "No classes match that filter."
            };
            div().px(px(4.)).py(px(4.)).child(text(ty::XS, t.text_tertiary).child(msg)).into_any_element()
        } else {
            div()
                .id("play-thr-list")
                .max_h(px(210.))
                .overflow_y_scroll()
                .track_scroll(&self.play_thresh_scroll)
                .flex()
                .flex_col()
                .gap(px(2.))
                .children(rows)
                .into_any_element()
        };

        Some(
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                .p(px(8.))
                .rounded(px(radius::MD))
                .bg(t.bg_inset)
                .border_1()
                .border_color(t.line_subtle)
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(px(4.))
                        .child(import_btn)
                        .child(copy_btn)
                        .child(export_btn)
                        .child(reset_all_btn),
                )
                .children(import_box)
                .children(note)
                .children(filter_box)
                .child(list_body)
                .into_any_element(),
        )
    }
}
