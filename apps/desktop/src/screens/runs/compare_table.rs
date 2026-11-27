//! Hyperparameter & outcome diff table for Multi-Run Compare (`Route::RunCompare`).

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, FontWeight, IntoElement, ParentElement, Styled, Window, div, px};

use super::compare::{run_slot_color, slot_swatch};
use super::compare_data::{best_map_winners, hyperparam_rows};
use super::duration_of;
use crate::api::{Run, time};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn run_compare_table(
        &mut self,
        t: &Theme,
        chosen: &[Run],
        same_ds: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let now = self.store.api.now();
        let compact = self.content_width(window) < 880.;
        let lead_w = if compact { 120. } else { 164. };
        let col_gap = if compact { 8. } else { 12. };

        // Header row: Setting + each run with Baseline / Make baseline.
        let mut head = div()
            .flex()
            .items_end()
            .gap(px(col_gap))
            .px(px(16.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().w(px(lead_w)).flex_none().child(caption(&t, "Setting")));
        for (slot, r) in chosen.iter().enumerate() {
            let color = run_slot_color(&t, slot);
            let role: AnyElement = if slot == 0 {
                div()
                    .h(px(crate::theme::size::CONTROL_MD))
                    .flex()
                    .items_center()
                    .child(text(ty::SM, t.accent_text).child("Baseline"))
                    .into_any_element()
            } else {
                let rid = r.id.clone();
                self.button(
                    &format!("run-cmp-base-{}", r.id),
                    "Make baseline",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(move |this, _, cx| this.run_compare_make_baseline(&rid, cx)),
                )
                .into_any_element()
            };
            head = head.child(
                div()
                    .flex_1()
                    .min_w(px(110.))
                    .flex()
                    .flex_col()
                    .items_end()
                    .gap(px(4.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_end()
                            .gap(px(6.))
                            .w_full()
                            .min_w_0()
                            .child(slot_swatch(color, slot))
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .flex_shrink()
                                    .min_w_0()
                                    .ellipsis()
                                    .child(r.name.clone()),
                            ),
                    )
                    .child(role),
            );
        }

        let row_frame = |label: &str, bold: bool, cells: Vec<AnyElement>| {
            let mut row = div()
                .flex()
                .items_center()
                .gap(px(col_gap))
                .px(px(16.))
                .py(px(10.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(
                    text(ty::SM, if bold { t.text_primary } else { t.text_secondary })
                        .when(bold, |d| d.font_weight(FontWeight::SEMIBOLD))
                        .w(px(lead_w))
                        .flex_none()
                        .ellipsis()
                        .child(label.to_string()),
                );
            for c in cells {
                row =
                    row.child(div().flex_1().min_w(px(110.)).overflow_hidden().flex().justify_end().child(c));
            }
            row
        };

        // 1. Outcome rows: Status, Best mAP50-95, Precision · Recall, Epochs, Duration.
        let status_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|r| status_dot(&t, r, &format!("cmp-st-{}", r.id)).into_any_element())
            .collect();

        let winners = best_map_winners(chosen);
        let base_best = chosen.first().and_then(|r| r.best.as_ref().map(|b| b.map50_95));
        let best_cells: Vec<AnyElement> =
            chosen
                .iter()
                .enumerate()
                .map(|(i, r)| match &r.best {
                    None => text(ty::SM, t.text_tertiary).child("–").into_any_element(),
                    Some(b) => {
                        let is_win = winners[i];
                        let delta = (i > 0 && same_ds)
                            .then(|| {
                                base_best.map(|base_v| {
                                    let (s, dir) = score_delta(b.map50_95 - base_v);
                                    let col = match dir {
                                        std::cmp::Ordering::Greater => t.success_text,
                                        std::cmp::Ordering::Less => t.danger_text,
                                        std::cmp::Ordering::Equal => t.text_tertiary,
                                    };
                                    text(ty::XS, col).tnum().child(s)
                                })
                            })
                            .flatten();
                        div()
                            .flex()
                            .flex_col()
                            .items_end()
                            .gap(px(1.))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .when(is_win, |d| d.child(badge(&t, "best", Tone::Success)))
                                    .child(
                                        text(ty::MD, t.text_primary)
                                            .tnum()
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .child(f3(b.map50_95)),
                                    ),
                            )
                            .child(
                                div().flex().items_center().gap(px(6.)).children(delta).child(
                                    text(ty::XS, t.text_tertiary).tnum().child(format!("ep {}", b.epoch)),
                                ),
                            )
                            .into_any_element()
                    }
                })
                .collect();

        let pr_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|r| {
                let val = r.best.as_ref().and_then(|b| {
                    self.store
                        .run_data
                        .get(&r.id)?
                        .ready()?
                        .metrics
                        .epochs
                        .iter()
                        .find(|e| e.epoch == b.epoch)
                        .map(|e| &e.val)
                });
                match val {
                    Some(v) => text(ty::SM, t.text_primary)
                        .tnum()
                        .child(format!("{} · {}", f3(v.precision), f3(v.recall)))
                        .into_any_element(),
                    None => text(ty::SM, t.text_tertiary).child("–").into_any_element(),
                }
            })
            .collect();

        let epoch_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|r| {
                text(ty::SM, t.text_primary)
                    .tnum()
                    .child(format!("{} / {}", r.epoch, r.epochs))
                    .into_any_element()
            })
            .collect();

        let dur_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|r| {
                let s = duration_of(r, now).map_or_else(|| "–".into(), time::duration);
                text(ty::SM, t.text_secondary).tnum().child(s).into_any_element()
            })
            .collect();

        // 2. Hyperparameter diff rows.
        let all_hp = hyperparam_rows(
            chosen,
            |r| self.store.run_dataset_name(r),
            |w| {
                self.store
                    .models
                    .ready()
                    .and_then(|ms| ms.iter().find(|m| m.id == w))
                    .map_or_else(|| w.to_string(), |m| m.name.clone())
            },
        );
        let diff_rows: Vec<_> = all_hp.iter().filter(|r| r.differs).collect();
        let same_rows: Vec<_> = all_hp.iter().filter(|r| !r.differs).collect();
        let show_same = self.run_compare.show_identical;

        let mut rows = div()
            .flex()
            .flex_col()
            .child(row_frame("Status", false, status_cells))
            .child(row_frame("Best mAP50-95", true, best_cells))
            .child(row_frame("Precision · recall", false, pr_cells))
            .child(row_frame("Epochs", false, epoch_cells))
            .child(row_frame("Duration", false, dur_cells));

        for hp in &diff_rows {
            let cells: Vec<AnyElement> = hp
                .values
                .iter()
                .enumerate()
                .map(|(i, val)| {
                    let off = hp.off_base[i];
                    div()
                        .flex()
                        .items_center()
                        .justify_end()
                        .gap(px(6.))
                        .min_w_0()
                        .when(off, |d| {
                            d.child(div().size(px(6.)).flex_none().rounded_full().bg(t.accent_solid))
                        })
                        .child(
                            text(ty::SM, if i == 0 || off { t.text_primary } else { t.text_tertiary })
                                .when(off, |d| d.font_weight(FontWeight::MEDIUM))
                                .min_w_0()
                                .tnum()
                                .ellipsis()
                                .child(val.clone()),
                        )
                        .into_any_element()
                })
                .collect();
            rows = rows.child(row_frame(hp.label, false, cells));
        }

        if !same_rows.is_empty() {
            let n_same = same_rows.len();
            let toggle_label = if show_same {
                format!("Hide {} identical {}", n_same, if n_same == 1 { "setting" } else { "settings" })
            } else {
                format!("Show {} identical {}", n_same, if n_same == 1 { "setting" } else { "settings" })
            };
            let toggle = self.button(
                "run-cmp-same-toggle",
                toggle_label,
                Btn::Secondary,
                Some(if show_same { "icons/chevron-up.svg" } else { "icons/chevron-down.svg" }),
                window,
                cx,
                act(|this, _, cx| {
                    this.run_compare.show_identical = !this.run_compare.show_identical;
                    cx.notify();
                }),
            );
            rows = rows
                .child(div().px(px(16.)).py(px(6.)).border_b_1().border_color(t.line_subtle).child(toggle));
            if show_same {
                for hp in &same_rows {
                    let cells: Vec<AnyElement> = hp
                        .values
                        .iter()
                        .map(|val| {
                            text(ty::SM, t.text_tertiary)
                                .min_w_0()
                                .tnum()
                                .ellipsis()
                                .child(val.clone())
                                .into_any_element()
                        })
                        .collect();
                    rows = rows.child(row_frame(hp.label, false, cells));
                }
            }
        }

        let diff_summary = format!(
            "{} {}",
            counted(diff_rows.len() as u64, "setting", "settings"),
            if diff_rows.len() == 1 { "differs" } else { "differ" }
        );
        panel(&t)
            .overflow_hidden()
            .child(
                panel_header(&t, "Hyperparameters & results")
                    .child(text(ty::SM, t.text_tertiary).child(diff_summary)),
            )
            .child(head)
            .child(rows)
            .into_any_element()
    }
}
