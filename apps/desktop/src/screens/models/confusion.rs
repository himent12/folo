//! The confusion matrix panel.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Model, ModelEval};
use crate::app::FoloApp;
use crate::theme::{alpha, radius, theme, ty};
use crate::ui::*;

/// Row-label column of the confusion matrix (fits "hexagon" and "FP share"); the columns get the
/// rest, so class names head them in full at 1280.
const LABEL_W: f32 = 64.;

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
        // detected, the last row detections with no object.
        let row_sums: Vec<u32> = ev.confusion.iter().map(|r| r.iter().sum::<u32>().max(1)).collect();
        let max_count = ev.confusion.iter().flatten().copied().max().unwrap_or(1).max(1);
        // The error ramp spans 0 to the largest error cell, false positives included (their row
        // is a share of all false positives, a row % like the class rows'), so the worst error is
        // the darkest cell and small ones stay apart (the key states that maximum).
        let frac_of =
            |r: usize, v: u32| if norm { v as f32 / row_sums[r] as f32 } else { v as f32 / max_count as f32 };
        let last = ev.confusion.len().saturating_sub(1);
        let max_err = ev
            .confusion
            .iter()
            .enumerate()
            .flat_map(|(r, row)| row.iter().enumerate().map(move |(c, &v)| (r, c, v)))
            .filter(|&(r, c, _)| r != c && !(r == last && c == last))
            .map(|(r, _, v)| frac_of(r, v))
            .fold(0f32, f32::max)
            .max(1e-6);
        let cell_h = px(32.);
        // Correct cells on the success hue: a series or class colour would read as "that class"
        // (series 1 is also the circle class's cyan).
        let base = t.success_solid;
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
        // One cell at a time can be read out: under the pointer, or the matrix's keyboard cursor.
        let sel = self.cm_cell.filter(|&(r, c)| r < n && c < n);
        let readout = sel.map(|(r, c)| {
            let v = ev.confusion[r][c];
            let pct = |d: u32| format!("{:.0}%", v as f32 / d.max(1) as f32 * 100.);
            let name = |i: usize| labels.get(i).cloned().unwrap_or_default();
            let count = int(v as u64);
            match (r == n - 1, c == n - 1) {
                (true, true) => "No value: there is no unmatched detection of no object".to_string(),
                (true, false) => {
                    format!(
                        "False positives predicted as {}: {count} · {} of all false positives",
                        name(c),
                        pct(row_sums[r])
                    )
                }
                (false, true) => {
                    format!("{} missed: {count} · {} of {} objects", name(r), pct(row_sums[r]), name(r))
                }
                (false, false) if r == c => {
                    format!("{} found: {count} · {} of {} objects", name(r), pct(row_sums[r]), name(r))
                }
                (false, false) => format!(
                    "{} predicted as {}: {count} · {} of {} objects",
                    name(r),
                    name(c),
                    pct(row_sums[r]),
                    name(r)
                ),
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
                    let frac = if norm { v as f32 / row_sums[r] as f32 } else { v as f32 / max_count as f32 };
                    let diag = r == c;
                    let na = r == n - 1 && c == n - 1;
                    // Under 1 % reads as noise: leave those cells blank, like zero.
                    let tiny = norm && frac < 0.01;
                    // An error under 2 % of its row is noise next to the ones that matter: it
                    // keeps its number on the empty cells' fill, so the ramp marks real confusions.
                    let faint = !diag && !na && (v as f32 / row_sums[r] as f32) < 0.02;
                    let fill = if na {
                        folo_ui::transparent_black()
                    } else if v == 0 || faint {
                        crate::theme::over(t.bg_panel, t.bg_hover)
                    } else {
                        // Correct cells on the success ramp; errors on a danger ramp (linear from a
                        // faint floor up to the largest error), so small and large confusions stay apart.
                        // The false-positive row is errors too: the same ramp and key.
                        if diag {
                            crate::theme::over(t.bg_inset, alpha(base, (0.35 + 0.6 * frac).min(0.95)))
                        } else {
                            crate::theme::over(t.bg_panel, alpha(t.danger_solid, err_alpha(frac / max_err)))
                        }
                    };
                    let label = if na {
                        "–".to_string()
                    } else if norm {
                        if v == 0 {
                            "·".to_string()
                        } else if tiny {
                            "<1%".to_string()
                        } else {
                            format!("{:.0}%", v as f32 / row_sums[r] as f32 * 100.)
                        }
                    } else {
                        int(v as u64)
                    };
                    let strong = frac > 0.55;
                    // Pick the label colour from the composited cell, not the theme.
                    let ink = if na || v == 0 || faint {
                        t.text_tertiary
                    } else {
                        t.readable_on(crate::theme::over(t.bg_panel, fill))
                    };
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
                                .when(diag && !na, |d| d.border_1().border_color(alpha(base, 0.9)))
                                // The cell being read out.
                                .when(picked, |d| d.border_2().border_color(t.text_primary))
                                .flex()
                                .items_center()
                                .justify_center()
                                .child(
                                    text(ty::SM, ink)
                                        .tnum()
                                        .when(strong, |d| d.font_weight(FontWeight::MEDIUM))
                                        .child(label),
                                ),
                        )
                }))
        });
        // Scale key for the error ramp: a continuous strip from 0 to 100 %.
        let key = div()
            .flex()
            .items_center()
            .gap(px(6.))
            .child(
                div()
                    .w(px(10.))
                    .h(px(10.))
                    .rounded(px(2.))
                    .bg(crate::theme::over(t.bg_inset, alpha(base, 0.8)))
                    .border_1()
                    .border_color(alpha(base, 0.9)),
            )
            .child(text(ty::SM, t.text_tertiary).mr(px(6.)).child("correct"))
            .child(text(ty::SM, t.text_tertiary).ml(px(2.)).child(if norm {
                "Error share 2%"
            } else {
                "Errors 0"
            }))
            .child(div().flex().h(px(8.)).w(px(96.)).rounded(px(2.)).overflow_hidden().children((0..12).map(
                |i| {
                    let f = i as f32 / 11.;
                    div()
                        .flex_1()
                        .h_full()
                        .bg(crate::theme::over(t.bg_inset, alpha(t.danger_solid, err_alpha(f))))
                },
            )))
            .child(text(ty::SM, t.text_tertiary).tnum().child(if norm {
                format!("{:.0}% max", max_err * 100.)
            } else {
                format!("{} max", int((max_err * max_count as f32).round() as u64))
            }));
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
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(panel_header(&t, "Confusion matrix").child(seg))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .px(px(16.))
                    .py(px(12.))
                    .gap(px(2.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .child(
                                text(ty::SM, t.text_tertiary)
                                    .w(px(LABEL_W))
                                    .pr(px(8.))
                                    .text_right()
                                    .child("Actual ↓"),
                            )
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
                    .child(
                        div().pt(px(4.)).child(
                            text(ty::SM, t.text_tertiary)
                                .tnum()
                                .child(match &eval_ds {
                                    Some((d, split)) => {
                                        format!("{}{cm_rule}", split_label(d, split, Some(ev.images as u64)))
                                    }
                                    // Unknown dataset: the split isn't known either.
                                    None => format!("Evaluation · {} images{cm_rule}", int(ev.images as u64)),
                                }),
                        ),
                    )
                    // Evaluated on another split than validation: the server's sentence says why.
                    .when_some(ev.split_fallback.as_ref(), |d, f| {
                        d.child(text(ty::SM, t.text_secondary).child(f.message.clone()))
                    }),
            )
            .into_any_element()
    }
}

/// A column heading's short form: the first three letters and a full stop ("hexagon" → "hex.").
fn short_name(name: &str) -> String {
    let short: String = name.chars().take(3).collect();
    if short.chars().count() < name.chars().count() { format!("{short}.") } else { short }
}

/// Fill opacity of an error cell: linear over `frac`, the cell's share of the largest
/// class-to-class confusion shown, from 0.18 (errors under 2 % of their row aren't filled at all,
/// so the lightest fill is a real confusion) to 0.72.
/// The square-root ramp it replaces lifted 1–4 % confusions toward full red, so the whole
/// off-diagonal read as alarm. The top stops at 0.72 so the strongest cell's label (dark or white,
/// whichever reads better) keeps 4.5:1 in both themes; at 0.8 neither did (4.36:1).
fn err_alpha(frac: f32) -> f32 {
    0.18 + 0.54 * frac.clamp(0., 1.)
}

#[cfg(test)]
mod matrix_tests {
    use crate::theme::{Mode, Theme, alpha, over};
    use folo_ui::{Hsla, Rgba};

    fn contrast(a: Hsla, b: Hsla) -> f32 {
        let lum = |c: Hsla| {
            let c: Rgba = c.into();
            let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
            0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
        };
        let (x, y) = (lum(a), lum(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }

    #[test]
    fn error_cells_keep_their_label_readable() {
        for t in [Theme::new(Mode::Dark), Theme::new(Mode::Light)] {
            for frac in (0..=100).map(|i| i as f32 / 100.) {
                let cell = over(t.bg_panel, alpha(t.danger_solid, super::err_alpha(frac)));
                let c = contrast(t.readable_on(cell), cell);
                assert!(c >= 4.5, "{:?} error cell at {frac}: {c:.2}:1", t.mode);
            }
        }
    }
}
