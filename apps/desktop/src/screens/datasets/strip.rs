//! A dataset's stat strip (the Statistics tab's KPI strip): counts, splits and label health. Its
//! Empty images tile shows them in the grid and its Broken labels tile copies the check command.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Div, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::list::names_summary;
use crate::api::{Dataset, SplitsExt};
use crate::app::{DsTab, FoloApp, act};
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn dataset_strip_stats(
        &self,
        ds: &Dataset,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // In the Statistics tab, under the health line (it no longer sits above the image grid, so
        // it keeps its full size in short windows too).
        let (step, pad) = (ty::X2XL, 12.);
        let stat = |t: &Theme, label: &str, value: String, unit: &str, detail: Option<Div>| {
            stat_at(t, label, value, unit, detail, step)
        };
        let total = ds.splits.total().max(1) as f32;
        let s = ds.stats.as_ref();
        let split_c = [
            t.text_tertiary,
            crate::theme::alpha(t.text_tertiary, 0.6),
            crate::theme::alpha(t.text_tertiary, 0.35),
        ];
        let split_bar = div()
            .flex()
            .h(px(4.))
            .w_full()
            .rounded_full()
            .overflow_hidden()
            .gap(px(2.))
            .child(div().h_full().w(folo_ui::relative(ds.splits.train as f32 / total)).bg(split_c[0]))
            .child(div().h_full().w(folo_ui::relative(ds.splits.val as f32 / total)).bg(split_c[1]))
            .when(ds.splits.test > 0, |d| d.child(div().h_full().flex_1().bg(split_c[2])));
        let legend = div().flex().gap(px(10.)).children(
            [
                ("train", ds.splits.train, split_c[0]),
                ("val", ds.splits.val, split_c[1]),
                ("test", ds.splits.test, split_c[2]),
            ]
            .into_iter()
            .filter(|x| x.1 > 0)
            .map(|(n, c, col)| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(5.))
                    .child(swatch(col, 6.))
                    .child(text(ty::SM, t.text_tertiary).tnum().child(format!("{n} {}", int(c))))
            }),
        );
        let tile = |content: folo_ui::Div| div().flex_1().min_w_0().px(px(16.)).py(px(pad)).child(content);
        let boxes = s.map_or(0, |s| s.boxes);
        let per = boxes as f32 / total;
        let broken = s.map_or(0, |s| s.broken_labels);
        let narrow_strip = window.viewport_size().width < px(1120.);
        panel(&t)
            .flex()
            .child(
                div().flex_none().w(px(if narrow_strip { 244. } else { 300. })).px(px(16.)).py(px(pad)).child(
                    // The split bar on the strip's first detail line (where the others' captions
                    // sit), its legend on the second: the same grid as every tile beside it.
                    stat(
                        &t,
                        "Images",
                        int(ds.splits.total()),
                        "",
                        Some(div().flex().flex_col().gap(px(6.)).child(detail_line(split_bar)).child(legend)),
                    ),
                ),
            )
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Boxes",
                int(boxes),
                "",
                Some(text(ty::SM, t.text_tertiary).tnum().child(format!("{per:.1} per image"))),
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Classes",
                ds.classes.len().to_string(),
                "",
                Some(text(ty::SM, t.text_tertiary).whitespace_nowrap().child(names_summary(&ds.classes, 24))),
            )))
            .child(vline(&t))
            .child({
                // Defects are actionable: this tile filters the grid to the images in question.
                let empty = s.map_or(0, |s| s.empty_images);
                let on = self.ds_empty_only;
                let el = div()
                    .id("ds-empty-tile")
                    .flex_1()
                    .min_w_0()
                    .px(px(16.))
                    .py(px(pad))
                    .when(empty > 0, |d| d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).active(|s| s.bg(t.bg_active)))
                    .when(on, |d| d.bg(t.bg_selected))
                    .child(stat(
                        &t,
                        "Empty images",
                        int(empty),
                        "",
                        Some(if empty > 0 {
                            if on {
                                text(ty::SM, t.accent_text).child("Showing them · clear")
                            } else {
                                inline_link(&t, "Show them")
                            }
                        } else if ds.splits.total() == 0 {
                            // Nothing to have labels: not "every image has labels" for no image.
                            text(ty::SM, t.text_tertiary).child("no images")
                        } else {
                            text(ty::SM, t.text_tertiary).child("every image has labels")
                        }),
                    ));
                if empty > 0 {
                    self.bind(
                        el,
                        "ds-empty-tile",
                        0.,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.ds_empty_only = !this.ds_empty_only;
                            // Empty images have no class: the two filters can't both apply.
                            if this.ds_empty_only {
                                this.ds_class = None;
                            }
                            this.ds_tab = DsTab::Images;
                            this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                } else {
                    el.into_any_element()
                }
            })
            .child(vline(&t))
            .child({
                let el = div()
                    .id("ds-broken-tile")
                    .flex_1()
                    .min_w_0()
                    .px(px(16.))
                    .py(px(pad))
                    .when(broken > 0, |d| d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).active(|s| s.bg(t.bg_active)))
                    .when(broken > 0, |d| {
                        d.tooltip(crate::controls::tip(
                            "These label files couldn't be read, so their images are skipped. Click to copy the terminal command that lists them.",
                            None,
                        ))
                    })
                    .child(if broken > 0 {
                        // Actionable: a warning, not one more inventory count.
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(6.))
                            .min_w_0()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(4.))
                                    .child(icon("icons/triangle-alert.svg", 12., t.warning_solid))
                                    .child(
                                        text(ty::SM, t.warning_text)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("Broken labels"),
                                    ),
                            )
                            .child(metric(&t, broken.to_string(), "", step, ty::SM))
                            // Split by severity when the server reports it, then the copy action.
                            .when_some(
                                self.store.issues.get(&ds.id).and_then(|l| l.ready()).filter(|c| c.error + c.warning > 0),
                                |d, c| d.child(severity_detail(&t, c)),
                            )
                            // The action, named (the tile copies `folo dataset check`).
                            .child(broken_detail(&t, narrow_strip))
                    } else {
                        stat(
                            &t,
                            "Broken labels",
                            broken.to_string(),
                            "",
                            Some(div().child(text(ty::SM, t.success_text).child("all labels valid"))),
                        )
                    });
                if broken > 0 {
                    let cmd = format!("folo dataset check \"{}\"", ds.path);
                    self.bind(
                        el,
                        "ds-broken-tile",
                        0.,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(cmd.clone()));
                            this.toast(
                                Tone::Neutral,
                                "Copied the check command",
                                Some(format!("Run `{cmd}` in a terminal to list the broken label files.").into()),
                                cx,
                            );
                        }),
                    )
                    .into_any_element()
                } else {
                    el.into_any_element()
                }
            })
            .into_any_element()
    }
}

/// Under the broken-label count: the check command to copy, or that every label is valid.
fn broken_detail(t: &crate::theme::Theme, narrow: bool) -> folo_ui::Div {
    if narrow { copy_cmd(t, "Copy check") } else { copy_cmd(t, "Copy check command") }
}

/// "2 errors · 1 warning" in their severity colours.
fn severity_detail(t: &crate::theme::Theme, c: &crate::api::IssueCounts) -> folo_ui::Div {
    let n = |k: u64, one: &str, many: &str| format!("{k} {}", if k == 1 { one } else { many });
    div()
        .flex()
        .items_center()
        .gap(px(4.))
        .min_w_0()
        .when(c.error > 0, |d| {
            d.child(
                text(ty::SM, t.danger_text).tnum().whitespace_nowrap().child(n(c.error, "error", "errors")),
            )
        })
        .when(c.error > 0 && c.warning > 0, |d| d.child(text(ty::SM, t.text_tertiary).child("·")))
        .when(c.warning > 0, |d| {
            d.child(
                text(ty::SM, t.warning_text)
                    .tnum()
                    .whitespace_nowrap()
                    .child(n(c.warning, "warning", "warnings")),
            )
        })
}

fn copy_cmd(t: &crate::theme::Theme, label: &'static str) -> folo_ui::Div {
    div()
        .flex()
        .items_center()
        .gap(px(4.))
        .min_w_0()
        .child(text(ty::SM, t.text_secondary).whitespace_nowrap().child(label))
        .child(div().flex_none().child(icon("icons/copy.svg", 12., t.text_secondary)))
}
