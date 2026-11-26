//! An entry's page: its header and its result on the five axes.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use super::cells::{bar_row, partial_badge};
use super::format::{
    axis_name, axis_text, calibration_none, calibration_why_not, contestant_line, entry_frac, partial_why,
    pct, prf_label,
};
use crate::api::{Bench, BenchAxis, BenchEntryStatus, BenchReadiness, BenchResult, Label, time};
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::screens::{detail_header, loading_rows, page_header};
use crate::store::Load;
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

impl FoloApp {
    // ---------- an entry ----------

    pub fn bench_entry_view(
        &mut self,
        bench: &str,
        entry: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        self.ensure_bench(bench, cx);
        if self.bench_unsupported() {
            let header = page_header(&t, "Entry", "Not available on this server", vec![]);
            let body = self.bench_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let e = self
            .bench
            .entries
            .get(bench)
            .and_then(|l| l.ready())
            .and_then(|l| l.iter().find(|x| x.id == entry))
            .cloned();
        let b = self.bench.list.ready().and_then(|l| l.iter().find(|x| x.id == bench)).cloned();
        let (Some(e), Some(b)) = (e, b) else {
            let header = page_header(&t, "Entry", "", vec![]);
            let failed = self
                .bench
                .entries
                .get(bench)
                .and_then(|l| l.error())
                .or_else(|| self.bench.list.error())
                .or_else(|| self.bench_blocked_error())
                .cloned();
            let body = if let Some(err) = failed {
                self.error_panel(&err, "bench-entry-retry", window, cx)
            } else if self.bench.list.is_loading()
                || self.bench_discovering()
                || self.bench.entries.get(bench).is_none_or(|l| l.is_loading())
            {
                panel(&t).child(loading_rows(&t, "entry", 5)).into_any_element()
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Entry not found",
                        "It may have been deleted.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body, cx);
        };
        let now = self.store.api.now();
        let mut title = div()
            .w_full()
            .flex()
            .items_center()
            .gap(px(10.))
            .min_w_0()
            .child(text(ty::XL, t.text_primary).min_w_0().ellipsis().child(e.name.clone()))
            .child(div().flex_none().child(status_badge_of(
                &t,
                StatusKind::of_entry(e.status),
                e.status.label(),
                format!("entry-{}", e.id),
            )));
        if e.score.as_ref().is_some_and(|s| s.partial) {
            title = title.child(div().flex_none().child(partial_badge(
                &t,
                partial_why(&e, &b),
                format!("entry-partial-{}", e.id),
            )));
        }
        let finished = e
            .finished
            .as_deref()
            .and_then(time::parse)
            .map_or(String::new(), |c| format!(" · finished {}", time::ago(c, now)));
        let subtitle = format!(
            "{} · on {} · {} · {} of {} conditions{finished}",
            contestant_line(&e),
            b.name,
            counted(e.images, "image", "images"),
            e.conditions_run,
            b.plan.conditions.len(),
        );
        let finished_count = self
            .bench
            .entries
            .get(bench)
            .and_then(|l| l.ready())
            .map_or(0, |l| l.iter().filter(|x| crate::bench::is_comparable(x)).count());
        let mut actions = Vec::new();
        if crate::bench::is_comparable(&e) && finished_count >= 2 {
            let (bid_cmp, eid_cmp) = (b.id.clone(), e.id.clone());
            actions.push(
                self.button(
                    "entry-compare",
                    "Compare",
                    Btn::Secondary,
                    Some("icons/git-compare-arrows.svg"),
                    window,
                    cx,
                    act(move |this, w, cx| this.open_bench_compare(&bid_cmp, Some(&eid_cmp), w, cx)),
                )
                .into_any_element(),
            );
        }
        let bid = b.id.clone();
        actions.push(
            self.button(
                "entry-board",
                "Leaderboard",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Bench(bid.clone()), w, cx)),
            )
            .into_any_element(),
        );
        let header = detail_header(&t, title, subtitle, actions);
        if e.status != BenchEntryStatus::Done {
            let msg = match e.status {
                BenchEntryStatus::Failed => e.error.clone().unwrap_or_else(|| "It failed.".into()),
                BenchEntryStatus::Stopped => {
                    "It was stopped before it finished, so nothing was scored.".into()
                }
                _ => format!("{} done. Its result shows here when it finishes.", pct(entry_frac(&e))),
            };
            let body = panel(&t)
                .child(empty_state(
                    &t,
                    "icons/clock.svg",
                    e.status.label(),
                    &msg,
                    if e.status == BenchEntryStatus::Failed { Tone::Danger } else { Tone::Neutral },
                ))
                .into_any_element();
            return self.page_frame(&t, header, body, cx);
        }
        self.ensure_bench_result(bench, entry, cx);
        let result =
            self.bench.results.get(&(bench.to_string(), entry.to_string())).cloned().unwrap_or_default();
        let retry_anchor = match self.passing_stop("bench-result-retry", result.error().is_some(), cx) {
            Some(fh) if result.error().is_none() => Some(div().track_focus(&fh)),
            _ => None,
        };
        let body = match result {
            Load::Ready(r) => {
                let images = self.entry_images_panel(&t, &b, &e, &r.accuracy, window, cx);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(div().children(retry_anchor).child(self.entry_result(&t, &b, &r)))
                    .child(images)
                    .into_any_element()
            }
            Load::Failed(err) => {
                let (bid, eid) = (b.id.clone(), e.id.clone());
                let retry = self.button(
                    "bench-result-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_bench_result(&bid, &eid, cx)),
                );
                panel(&t)
                    .child(
                        empty_state(
                            &t,
                            "icons/circle-alert.svg",
                            "The result didn't load",
                            &err.message,
                            Tone::Danger,
                        )
                        .child(retry),
                    )
                    .into_any_element()
            }
            _ => panel(&t).children(retry_anchor).child(loading_rows(&t, "result", 6)).into_any_element(),
        };
        self.page_frame(&t, header, body, cx)
    }

    fn entry_result(&self, t: &Theme, b: &Bench, r: &BenchResult) -> AnyElement {
        let t = *t;
        let e = &r.entry;
        let score = e.score.as_ref();
        let axis = |a: BenchAxis| score.and_then(|s| s.axes.iter().find(|x| x.axis == a)).copied();
        // The score and its five axes, each with its weight.
        let kpi = |content: folo_ui::Div, basis: f32| {
            div().flex_grow().flex_basis(relative(basis)).min_w_0().px(px(16.)).py(px(14.)).child(content)
        };
        let mut kpis = panel(&t).flex().flex_wrap().child(kpi(
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .child(kpi_label(&t, "Score"))
                .child(metric(
                    &t,
                    score.map_or("–".to_string(), |s| format!("{:.1}", s.total)),
                    "/ 100",
                    KPI_HEADLINE,
                    KPI_UNIT,
                ))
                .child(kpi_note(&t).child(if score.is_some_and(|s| s.partial) {
                    "Partial: over what was measured"
                } else {
                    "All five axes measured"
                })),
            0.2,
        ));
        for a in BenchAxis::ALL {
            let s = axis(a);
            let v = s.and_then(|s| s.value);
            let detail = match a {
                BenchAxis::Accuracy => format!("mAP50-95 {}", f3(r.accuracy.map50_95)),
                // Short enough for the tile at 1280 px; the Robustness panel says it in full.
                BenchAxis::Robustness => r
                    .robustness
                    .as_ref()
                    .map_or("conditions not run".into(), |x| format!("{} accuracy kept", pct(x.retention))),
                BenchAxis::SelfLabeling => match r.self_labeling.readiness {
                    BenchReadiness::Ready => "ready to self-label".into(),
                    BenchReadiness::Partial => "partly ready".into(),
                    BenchReadiness::NotReady => "not ready".into(),
                },
                BenchAxis::Calibration => {
                    r.calibration.as_ref().map_or(calibration_none(e), |c| format!("ECE {}", f3(c.ece)))
                }
                BenchAxis::Localization => r
                    .localization
                    .mean_iou
                    .map_or("nothing correct".into(), |m| format!("mean IoU {}", f3(m))),
            };
            kpis = kpis.child(kpi(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .child(kpi_label(
                        &t,
                        &format!("{} · {}", axis_name(a), pct(s.map_or(a.weight(), |s| s.weight))),
                    ))
                    .child(
                        div().h(px(KPI_HEADLINE.line)).flex().items_end().child(
                            metric(&t, axis_text(v), "", KPI_VALUE, KPI_UNIT)
                                .pb(px(baseline_lift(KPI_HEADLINE, KPI_VALUE))),
                        ),
                    )
                    .child(kpi_note(&t).ellipsis().child(detail)),
                0.12,
            ));
        }

        // Accuracy and self-labeling, side by side.
        let acc = &r.accuracy;
        let opt = |v: Option<f32>| v.map_or("–".to_string(), f3);
        let accuracy = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(
                panel_header(&t, "Accuracy")
                    .child(text(ty::SM, t.text_tertiary).child("on the unchanged images")),
            )
            .child(
                div()
                    .px(px(16.))
                    .py(px(8.))
                    .child(kv(
                        &t,
                        "mAP50-95",
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{}  ({} – {}, 95%)",
                            f3(acc.map50_95),
                            f3(acc.ci95[0]),
                            f3(acc.ci95[1])
                        )),
                    ))
                    .child(kv(
                        &t,
                        "mAP50 · mAP75",
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{} · {}",
                            f3(acc.map50),
                            f3(acc.map75)
                        )),
                    ))
                    .child(kv(
                        &t,
                        &prf_label(acc),
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{} · {} · {}",
                            f3(acc.precision),
                            f3(acc.recall),
                            f3(acc.f1)
                        )),
                    ))
                    .child(kv(
                        &t,
                        "Small · medium · large",
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{} · {} · {}",
                            opt(acc.map_small),
                            opt(acc.map_medium),
                            opt(acc.map_large)
                        )),
                    ))
                    .when_some(e.speed.as_ref(), |d, s| {
                        d.child(kv(
                            &t,
                            "Speed",
                            text(ty::SM, t.text_primary).tnum().child(format!(
                                "{:.1} ms/image on {}",
                                s.ms_per_image,
                                device_label(&s.device)
                            )),
                        ))
                    }),
            );
        let sl = &r.self_labeling;
        let tone = match sl.readiness {
            BenchReadiness::Ready => Tone::Success,
            BenchReadiness::Partial => Tone::Warning,
            BenchReadiness::NotReady => Tone::Neutral,
        };
        let self_label = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(panel_header(&t, "Self-labeling").child(badge(&t, sl.readiness.label(), tone)))
            .child(
                div()
                    .px(px(16.))
                    .py(px(8.))
                    .child(kv(
                        &t,
                        "Threshold",
                        text(ty::SM, t.text_primary).tnum().child(
                            sl.threshold.map_or("never reaches the target".into(), |c| {
                                format!("confidence {}", fmt_conf(c))
                            }),
                        ),
                    ))
                    .child(kv(
                        &t,
                        &format!("Precision (target {})", pct(sl.target_precision)),
                        text(ty::SM, t.text_primary).tnum().child(pct(sl.precision)),
                    ))
                    .child(kv(
                        &t,
                        "Objects it labels",
                        text(ty::SM, t.text_primary).tnum().child(pct(sl.recall)),
                    ))
                    .child(kv(
                        &t,
                        "Labels per image",
                        text(ty::SM, t.text_primary).tnum().child(format!("{:.1}", sl.labels_per_image)),
                    ))
                    .child(kv(
                        &t,
                        "Mean IoU of right labels",
                        text(ty::SM, t.text_primary).tnum().child(opt(sl.mean_iou)),
                    )),
            );
        let pair = div().flex().flex_wrap().gap(px(16.)).child(accuracy).child(self_label);

        // Robustness: what each condition keeps.
        let catalog = self.bench.catalog.ready();
        let cond_name = |id: &str| {
            catalog
                .and_then(|c| c.conditions.iter().find(|x| x.id == id))
                .map_or(id.to_string(), |c| c.name.clone())
        };
        let robustness = r.robustness.as_ref().map(|rob| {
            let rows = rob.conditions.iter().map(|c| {
                let sev = c
                    .points
                    .iter()
                    .map(|p| format!("S{} {}", p.severity, f3(p.map50_95)))
                    .collect::<Vec<_>>()
                    .join(" · ");
                div()
                    .flex()
                    .flex_col()
                    .py(px(4.))
                    .child(bar_row(
                        &t,
                        cond_name(&c.condition),
                        c.retention,
                        pct(c.retention),
                        if c.retention >= 0.8 {
                            t.success_solid
                        } else if c.retention >= 0.5 {
                            t.warning_solid
                        } else {
                            t.danger_solid
                        },
                    ))
                    .child(
                        div().pl(px(152.)).child(text(ty::SM, t.text_tertiary).tnum().ellipsis().child(sev)),
                    )
            });
            panel(&t)
                .child(panel_header(&t, "Robustness").child(text(ty::SM, t.text_tertiary).child(format!(
                    "keeps {} of its accuracy under the plan's conditions",
                    pct(rob.retention)
                ))))
                .child(div().px(px(16.)).py(px(8.)).children(rows))
        });

        // Per class.
        let classes = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(panel_header(&t, "Per class").child(text(ty::SM, t.text_tertiary).child("AP50-95")))
            .child(div().px(px(16.)).py(px(8.)).children(acc.per_class.iter().map(|c| {
                let name =
                    b.classes.get(c.class as usize).cloned().unwrap_or_else(|| format!("class {}", c.class));
                div().flex().items_center().gap(px(8.)).child(class_swatch(&t, c.class, 10.)).child(
                    div().flex_1().min_w_0().child(bar_row(
                        &t,
                        format!("{name} · {}", int(c.truths)),
                        c.map50_95,
                        f3(c.map50_95),
                        t.class_on(c.class, crate::theme::Surface::Panel),
                    )),
                )
            })))
            .when(!e.missing_classes.is_empty(), |d| {
                d.child(div().px(px(16.)).pb(px(10.)).child(text(ty::SM, t.warning_text).child(format!(
                    "The model doesn't know {}: AP 0 on those.",
                    e.missing_classes.join(", ")
                ))))
            });

        // Calibration and localization.
        let calib = panel(&t).flex_1().min_w(px(320.)).child(panel_header(&t, "Calibration").child(
            text(ty::SM, t.text_tertiary).child(r.calibration.as_ref().map_or(calibration_none(e), |c| {
                format!("ECE {} over {} detections", f3(c.ece), int(c.detections))
            })),
        ));
        let calib = match &r.calibration {
            Some(c) => calib.child(div().px(px(16.)).py(px(8.)).children(c.bins.iter().filter(|b| b.count > 0).map(|bin| {
                let p = bin.precision.unwrap_or(0.);
                bar_row(
                    &t,
                    format!("{:.1}–{:.1} · {}", bin.lo, bin.hi, int(bin.count)),
                    p,
                    pct(p),
                    if bin.confidence.is_some_and(|c| (c - p).abs() <= 0.1) { t.success_solid } else { t.warning_solid },
                )
            })))
            .child(div().px(px(16.)).pb(px(10.)).child(text(ty::SM, t.text_tertiary).child("Share right per confidence band: a trustworthy model is right about as often as it is sure."))),
            None => calib.child(div().px(px(16.)).py(px(12.)).child(text(ty::SM, t.text_tertiary).child(format!("{} Calibration is left out of the score.", calibration_why_not(e))))),
        };
        let loc = &r.localization;
        let peak = loc.histogram.iter().copied().max().unwrap_or(0).max(1);
        let bins = loc.histogram.len().max(1);
        let hist = div().h(px(64.)).flex().items_end().gap(px(3.)).children(
            loc.histogram.iter().enumerate().map(|(i, &n)| {
                let tip = iou_bin_tooltip(i, bins, n, loc.matches);
                div()
                    .id(SharedString::from(format!("loc-bin-{i}")))
                    .flex_1()
                    .h_full()
                    .flex()
                    .flex_col()
                    .justify_end()
                    .rounded_t(px(2.))
                    .hover(|s| s.bg(t.bg_hover))
                    .tooltip(crate::controls::tip(tip, None))
                    .child(
                        div()
                            .w_full()
                            .h(relative(n as f32 / peak as f32))
                            .min_h(px(1.))
                            .rounded_t(px(2.))
                            .bg(if n > 0 { t.accent_fill() } else { t.line_subtle }),
                    )
            }),
        );
        let localization = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(panel_header(&t, "Localization").child(text(ty::SM, t.text_tertiary).child(
                loc.mean_iou.map_or("nothing correct".to_string(), |m| {
                    format!("mean IoU {} over {} matches", f3(m), int(loc.matches))
                }),
            )))
            .child(
                div().px(px(16.)).py(px(12.)).flex().flex_col().gap(px(6.)).child(hist).child(
                    div()
                        .flex()
                        .justify_between()
                        .child(text(ty::SM, t.text_tertiary).child("IoU 0.5"))
                        .child(text(ty::SM, t.text_tertiary).child("1.0")),
                ),
            );

        div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(kpis)
            .child(pair)
            .children(robustness)
            .child(
                div().flex().flex_wrap().gap(px(16.)).child(classes).child(
                    div()
                        .flex_1()
                        .min_w(px(320.))
                        .flex()
                        .flex_col()
                        .gap(px(16.))
                        .child(calib)
                        .child(localization),
                ),
            )
            .into_any_element()
    }
}

/// Tooltip label for one IoU localization histogram bin spanning `0.50..=1.00`.
pub(crate) fn iou_bin_tooltip(index: usize, bins: usize, count: u64, total_matches: u64) -> String {
    let bins = bins.max(1);
    let lo = 0.5 + 0.5 * (index as f32 / bins as f32);
    let hi = 0.5 + 0.5 * ((index + 1) as f32 / bins as f32);
    if total_matches > 0 {
        let share = (count as f64 / total_matches as f64) * 100.;
        format!("IoU {lo:.2}–{hi:.2} · {} ({share:.0}%)", counted(count, "match", "matches"))
    } else {
        format!("IoU {lo:.2}–{hi:.2} · 0 matches")
    }
}

#[cfg(test)]
mod tests {
    use super::iou_bin_tooltip;

    #[test]
    fn iou_bin_tooltip_formats_range_count_and_share() {
        assert_eq!(iou_bin_tooltip(0, 10, 5, 20), "IoU 0.50–0.55 · 5 matches (25%)");
        assert_eq!(iou_bin_tooltip(9, 10, 1, 20), "IoU 0.95–1.00 · 1 match (5%)");
        assert_eq!(iou_bin_tooltip(3, 10, 0, 0), "IoU 0.65–0.70 · 0 matches");
    }
}
