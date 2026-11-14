//! The confusion matrix panel: the matrix on one luminance ramp (`confusion_ink`), the share of its
//! row in each cell (or its count), and beside it the Most confused and Most missed lists.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::confusion_ink::{
    FLOOR, Kind, RANK_MAX, cell_fill, label_ink, missed_overall, rank_confusions, share_pct,
};
use crate::api::{Model, ModelEval};
use crate::app::{FoloApp, act};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Row-label column of the confusion matrix (fits "hexagon" and "FP share"); the columns get the
/// rest, so class names head them in full at 1280.
const LABEL_W: f32 = 64.;
/// The widest a matrix column gets: a full-width panel would otherwise stretch six cells into bars.
const CELL_MAX: f32 = 76.;
/// The lists' column beside the matrix.
const RAIL_W: f32 = 300.;
/// A class row needs this many objects to rank: one stray box can't top a list.
const MIN_ROW: u32 = 20;

impl FoloApp {
    pub(super) fn confusion_panel(
        &mut self,
        m: &Model,
        ev: &ModelEval,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let norm = self.confusion_norm;
        // The evaluated dataset and split, as the server reports them (`ModelEval`); an older server
        // leaves them out, and then they're inferred from the model's run.
        let eval_ds = ev.dataset_name.clone().zip(ev.split.map(|s| s.as_str())).or_else(|| {
            m.run_id
                .as_ref()
                .and_then(|r| self.store.run(r))
                .and_then(|r| self.store.dataset_summary(&r.config.dataset_id))
                .map(|d| (d.name.clone(), if d.splits.val == 0 { "train" } else { "val" }))
        });
        // What the matrix counts: predictions above `confusion_conf`, matched at `confusion_iou`.
        let cm_rule = ev
            .confusion_conf
            .zip(ev.confusion_iou)
            .map(|(c, i)| format!(" · conf {} · IoU {:.2}", fmt_conf(c), i))
            .unwrap_or_default();
        let seg = self.segmented(
            "cm-norm",
            &["Row %", "Counts"],
            if norm { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, _| this.confusion_norm = i == 0),
        );
        let n = ev.confusion.len();
        let mut labels: Vec<String> = m.classes.clone();
        labels.truncate(n.saturating_sub(1));
        labels.push("Missed".into());
        // Rows are what was there, columns what was predicted: the last column is objects nothing
        // detected, the last row detections with no object (a share of all false positives).
        let row_sums: Vec<u32> = ev.confusion.iter().map(|r| r.iter().sum::<u32>().max(1)).collect();
        let share = |r: usize, v: u32| v as f32 / row_sums[r] as f32;
        let cell_h = px(32.);
        // Column headings get a measured width (the column area as last laid out, split evenly):
        // a name that doesn't fit shows a short form, in full on hover.
        let cols = self.cm_cols.clone();
        // A heading may use its whole cell less a 2 px gap to the next one (medium weight measures
        // ~4 % wider than the regular face `text_width` sets).
        let wide = |window: &mut Window, s: &str| crate::charts::text_width(window, s, ty::SM.size) * 1.04;
        // "Missed" is never shortened: when an even share can't hold it, it takes the width it needs
        // and the class columns share the rest.
        let missed_need = wide(window, "Missed") + px(4.);
        let missed_w =
            cols.get().filter(|w| (*w - px(6.)) / n.max(1) as f32 <= missed_need).map(|_| missed_need);
        let col_w = cols.get().map(|w| {
            let k = if missed_w.is_some() { n.saturating_sub(1) } else { n };
            (w - px(6.) - missed_w.unwrap_or_default()) / k.max(1) as f32 - px(2.)
        });
        let header_row = div().flex().pl(px(LABEL_W)).child(
            div()
                .relative()
                .flex_1()
                .min_w_0()
                .flex()
                .child(
                    folo_ui::canvas(
                        move |b, window, _| {
                            if cols.get() != Some(b.size.width) {
                                cols.set(Some(b.size.width));
                                window.request_animation_frame();
                            }
                        },
                        |_, _, _, _| {},
                    )
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full(),
                )
                .children(labels.iter().enumerate().map(|(i, l)| {
                    // The full name, else its short form, else its initial (each measured); a
                    // shortened heading shows the full name on hover.
                    // A heading may overhang its cell by 2 px into the 4 px gap.
                    let room =
                        |window: &mut Window, s: &str| col_w.is_none_or(|w| wide(window, s) <= w + px(4.));
                    let fits = i == n - 1 || room(window, l);
                    let shown = if fits {
                        l.clone()
                    } else if room(window, &short_name(l)) {
                        short_name(l)
                    } else {
                        l.chars().next().map(String::from).unwrap_or_default()
                    };
                    div()
                        .id(SharedString::from(format!("cm-col-{i}")))
                        .flex_1()
                        .min_w_0()
                        .px(px(2.))
                        .when(i == n - 1, |d| d.ml(px(6.)))
                        .when(i == n - 1, |d| d.when_some(missed_w, |d, w| d.flex_none().w(w)))
                        .flex()
                        .justify_center()
                        .when(!fits, |d| d.tooltip(crate::controls::tip(l.clone(), None)))
                        .child(
                            text(ty::SM, if i == n - 1 { t.text_tertiary } else { t.text_secondary })
                                .font_weight(FontWeight::MEDIUM)
                                .whitespace_nowrap()
                                // A name that passed the fit check renders whole (a few px into the gap at most);
                                // only a shortened form keeps the hard cap.
                                .when(!fits, |d| {
                                    d.when_some(col_w, |d, w| d.max_w(w + px(4.)).overflow_hidden())
                                })
                                .child(shown),
                        )
                })),
        );
        // One cell at a time can be read out: under the pointer, the matrix's keyboard cursor, or the
        // one a ranked line points at.
        let sel = self.cm_cell.filter(|&(r, c)| r < n && c < n);
        let readout = sel.map(|(r, c)| {
            let v = ev.confusion[r][c];
            let pct = if v == 0 { "0%".to_string() } else { format!("{}%", share_pct(share(r, v))) };
            let name = |i: usize| labels.get(i).cloned().unwrap_or_default();
            let count = int(v as u64);
            match (r == n - 1, c == n - 1) {
                (true, true) => "No value: there is no unmatched detection of no object".to_string(),
                (true, false) => {
                    format!(
                        "False positives predicted as {}: {count} · {pct} of all false positives",
                        name(c)
                    )
                }
                (false, true) => format!("{} missed: {count} · {pct} of {} objects", name(r), name(r)),
                (false, false) if r == c => {
                    format!("{} found: {count} · {pct} of {} objects", name(r), name(r))
                }
                (false, false) => {
                    format!("{} predicted as {}: {count} · {pct} of {} objects", name(r), name(c), name(r))
                }
            }
        });
        let rows = ev.confusion.iter().enumerate().map(|(r, row)| {
            div()
                .flex()
                .items_center()
                .when(r == n - 1, |d| d.mt(px(8.)))
                .child(
                    text(ty::SM, if r == n - 1 { t.text_tertiary } else { t.text_secondary })
                        .font_weight(FontWeight::MEDIUM)
                        .w(px(LABEL_W))
                        .flex_none()
                        .pr(px(8.))
                        .text_right()
                        .ellipsis()
                        .child(if r == n - 1 {
                            // Its denominator differs from the class rows': all false positives.
                            if norm { "FP share".to_string() } else { "FP count".to_string() }
                        } else {
                            labels[r].clone()
                        }),
                )
                .children(row.iter().enumerate().map(|(c, &v)| {
                    let na = r == n - 1 && c == n - 1;
                    // One ramp for every cell, the diagonal told apart by its place: the share of
                    // the row, in both views (Counts changes the label only).
                    let fill = if na { folo_ui::transparent_black() } else { cell_fill(&t, share(r, v)) };
                    let label = if na {
                        "–".to_string()
                    } else if v == 0 {
                        String::new()
                    } else if norm {
                        format!("{}%", share_pct(share(r, v)))
                    } else {
                        int(v as u64)
                    };
                    let ink = if na { t.text_tertiary } else { label_ink(fill) };
                    let picked = sel == Some((r, c));
                    div()
                        .id(SharedString::from(format!("cm-{r}-{c}")))
                        .flex_1()
                        .min_w_0()
                        .h(cell_h)
                        .p(px(1.))
                        .when(c == n - 1, |d| d.ml(px(6.)).when_some(missed_w, |d, w| d.flex_none().w(w)))
                        .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                            if *h {
                                this.cm_cell = Some((r, c));
                                cx.notify();
                            } else if this.cm_cell == Some((r, c)) {
                                this.cm_cell = None;
                                cx.notify();
                            }
                        }))
                        .child(
                            div()
                                .size_full()
                                .rounded(px(3.))
                                .bg(fill)
                                // No objects: the panel itself, edged by the hairline grid.
                                .when(v == 0 && !na, |d| d.border_1().border_color(t.line_subtle))
                                // The cell being read out.
                                .when(picked, |d| d.border_2().border_color(t.text_primary))
                                .flex()
                                .items_center()
                                .justify_center()
                                .child(text(ty::SM, ink).tnum().child(label)),
                        )
                }))
        });
        // The ramp's key: empty, the three floor steps, then the even ramp from 10 % to 100 %.
        let swatch = |v: f32| {
            div()
                .w(px(14.))
                .h(px(10.))
                .rounded(px(2.))
                .bg(cell_fill(&t, v))
                .when(v == 0., |d| d.border_1().border_color(t.line_subtle))
        };
        let tick = |s: &str| text(ty::SM, t.text_tertiary).tnum().child(s.to_string());
        let key = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(6.))
            .child(text(ty::SM, t.text_tertiary).mr(px(4.)).child("Share of the row"))
            .child(swatch(0.))
            .child(tick("0").mr(px(4.)))
            .children([(0.005, "<1%"), (FLOOR[0], "1%"), (FLOOR[1], "5%")].into_iter().map(|(v, s)| {
                div().flex().items_center().gap(px(4.)).mr(px(4.)).child(swatch(v)).child(tick(s))
            }))
            .child(tick("10%"))
            .child(div().flex().h(px(10.)).w(px(96.)).rounded(px(2.)).overflow_hidden().children(
                (0..12).map(|i| {
                    let v = FLOOR[2] + (1. - FLOOR[2]) * i as f32 / 11.;
                    div().flex_1().h_full().bg(cell_fill(&t, v))
                }),
            ))
            .child(tick("100%"));
        // The cells are one Tab stop: arrows move a cursor cell (the readout follows), Esc clears.
        let rows: Vec<AnyElement> = rows.map(|r| r.into_any_element()).collect();
        let fh = self.fh("cm-grid", cx);
        let ring = self.ring(&fh, window);
        let grid = div()
            .id("cm-grid")
            .track_focus(&fh)
            .relative()
            .flex()
            .flex_col()
            .gap(px(2.))
            .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .on_action(cx.listener(|this, _: &crate::app::Dismiss, _, cx| {
                if this.cm_cell.take().is_some() {
                    cx.notify();
                } else {
                    cx.propagate();
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, _, cx| {
                let key = ev.keystroke.key.as_str();
                let arrows = matches!(key, "left" | "right" | "up" | "down" | "home" | "end");
                // A cell from another model's (larger) matrix doesn't count: start over.
                let cur = this.cm_cell.filter(|&(r, c)| r < n && c < n);
                let next = match (cur, key) {
                    // The first press lands on the top-left cell.
                    (None, _) if arrows => Some((0, 0)),
                    (Some((r, c)), "left") => Some((r, c.saturating_sub(1))),
                    (Some((r, c)), "right") => Some((r, (c + 1).min(n - 1))),
                    (Some((r, c)), "up") => Some((r.saturating_sub(1), c)),
                    (Some((r, c)), "down") => Some(((r + 1).min(n - 1), c)),
                    (Some((r, _)), "home") => Some((r, 0)),
                    (Some((r, _)), "end") => Some((r, n - 1)),
                    _ => None,
                };
                if let Some(cell) = next {
                    this.cm_cell = Some(cell);
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .children(rows);
        let matrix = div()
            .flex()
            .flex_col()
            .gap(px(2.))
            .w_full()
            .max_w(px(LABEL_W + 6. + n as f32 * (CELL_MAX + 2.)))
            .child(
                div()
                    .flex()
                    .items_center()
                    .child(text(ty::SM, t.text_tertiary).w(px(LABEL_W)).pr(px(8.)).text_right().child("Actual ↓"))
                    .child(text(ty::SM, t.text_tertiary).flex_1().child("Predicted →")),
            )
            .child(header_row)
            .child(grid)
            // The key sits under the matrix, away from the column labels.
            .child(div().pt(px(10.)).child(key))
            // The selected cell in words, else what the rows mean.
            .child(div().pt(px(4.)).w_full().min_w_0().child(match readout {
                Some(r) => text(ty::SM, t.text_primary).tnum().whitespace_normal().child(r),
                None => text(ty::SM, t.text_tertiary).whitespace_normal().child(if norm {
                    "Rows are actual objects; the bottom row splits false positives by class. Hover or focus a cell to read it."
                } else {
                    "Counts of objects and unmatched detections. Hover or focus a cell to read it."
                }),
            }))
            // The grid is one Tab stop: with keyboard focus, how to move in it (in the strip
            // under the page, as every keyed control says it).
            .when(ring, |d| {
                self.hint_keys("Arrows move the cell · Home / End row ends · Esc clears");
                d
            })
            .child(div().pt(px(4.)).child(text(ty::SM, t.text_tertiary).tnum().child(match &eval_ds {
                Some((d, split)) => format!("{}{cm_rule}", split_label(d, split, Some(ev.images as u64))),
                // Unknown dataset: the split isn't known either.
                None => format!("Evaluation · {} images{cm_rule}", int(ev.images as u64)),
            })))
            // Evaluated on another split than validation: the server's sentence says why.
            .when_some(ev.split_fallback.as_ref(), |d, f| {
                d.child(text(ty::SM, t.text_secondary).child(f.message.clone()))
            });
        let stacked = self.content_width(window) < LABEL_W + n as f32 * 48. + RAIL_W + 80.;
        let rail = self.confusion_rail(ev, &labels, window, cx);
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(panel_header(&t, "Confusion matrix").child(seg))
            .child(
                div()
                    .flex()
                    .when(stacked, |d| d.flex_col())
                    .gap(px(24.))
                    .px(px(16.))
                    .py(px(12.))
                    .child(div().flex_1().min_w_0().child(matrix))
                    .child(div().when(!stacked, |d| d.w(px(RAIL_W)).flex_none()).child(rail)),
            )
            .into_any_element()
    }

    /// What goes wrong most, ranked: the classes most taken for another and most missed, each line
    /// showing its cell in the matrix (click or Enter); then the share missed overall.
    fn confusion_rail(
        &mut self,
        ev: &ModelEval,
        labels: &[String],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let all = rank_confusions(&ev.confusion, usize::MAX, MIN_ROW);
        let any_errors = !rank_confusions(&ev.confusion, 1, 1).is_empty();
        let name = |i: usize| labels.get(i).cloned().unwrap_or_else(|| format!("class {i}"));
        let mut col = div().flex().flex_col().gap(px(12.));
        for (kind, title) in [(Kind::Confused, "Most confused"), (Kind::Missed, "Most missed")] {
            let lines: Vec<_> = all.iter().filter(|e| e.kind == kind).take(RANK_MAX).copied().collect();
            if lines.is_empty() {
                continue;
            }
            let mut list = div().flex().flex_col().child(
                text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).h(px(24.)).child(title),
            );
            for e in lines {
                let key = format!("cm-rank-{}-{}", e.r, e.c);
                let what = if e.kind == Kind::Missed {
                    name(e.r)
                } else {
                    format!("{} → {}", name(e.r), name(e.c))
                };
                let pct = format!("{}%", share_pct(e.v));
                let tip = format!(
                    "{what}: {} of {}, {pct} · show in the matrix",
                    int(e.count as u64),
                    int(e.total as u64)
                );
                let picked = self.cm_cell == Some((e.r, e.c));
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .h(px(30.))
                    .px(px(6.))
                    .mx(px(-6.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .when(picked, |d| d.bg(t.bg_selected))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .tooltip(crate::controls::tip(tip, None))
                    .child(class_swatch(&t, e.r as u32, 8.))
                    .child(text(ty::SM, t.text_primary).flex_1().min_w_0().ellipsis().child(what))
                    .child(text(ty::SM, t.text_tertiary).tnum().flex_none().child(format!(
                        "{} of {}",
                        int(e.count as u64),
                        int(e.total as u64)
                    )))
                    .child(
                        text(ty::SM, t.text_primary).tnum().w(px(36.)).flex_none().text_right().child(pct),
                    );
                let cell = (e.r, e.c);
                list = list.child(self.bind(
                    row,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.cm_cell = if this.cm_cell == Some(cell) { None } else { Some(cell) };
                        cx.notify();
                    }),
                ));
            }
            col = col.child(list);
        }
        if all.is_empty() {
            col = col.child(text(ty::SM, t.text_tertiary).whitespace_normal().child(if any_errors {
                "No class with 20 or more objects is confused or missed: the few errors are in the matrix."
            } else {
                "No confusions and no misses: every object was found as its own class."
            }));
        }
        if let Some(v) = missed_overall(&ev.confusion) {
            let pct = share_pct(v);
            col = col.child(
                div()
                    .flex()
                    .gap(px(4.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .tnum()
                            .font_weight(FontWeight::MEDIUM)
                            .child(format!("{}%", if pct.is_empty() { "0".into() } else { pct })),
                    )
                    .child(text(ty::SM, t.text_tertiary).child("of all objects missed overall.")),
            );
        }
        col.into_any_element()
    }
}

/// A column heading's short form: the first three letters and a full stop ("hexagon" → "hex.").
fn short_name(name: &str) -> String {
    let short: String = name.chars().take(3).collect();
    if short.chars().count() < name.chars().count() { format!("{short}.") } else { short }
}
