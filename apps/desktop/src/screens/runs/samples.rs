//! A run's validation samples panel: eight of the fixed evaluation images at an epoch, with the
//! view (Both / Predicted / Truth), the epoch picker and the minimum confidence.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::sample_errors::{
    SampleErrorFilter, SampleErrorStats, SampleSort, analyze_sample, count_sample_filters,
    filter_and_sort_samples,
};
use crate::api::Run;
use crate::app::{FoloApp, SampleView, act};
use crate::controls::Btn;
use crate::screens::image_canvas;
use crate::store::{ImgKind, Load};
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

/// The minimum confidence slider's range and step.
pub(crate) const SAMPLE_CONF: (f32, f32, f32) = (0.25, 0.95, 0.05);

/// The epochs the picker offers below the latest (`latest`, the last validated epoch): the first,
/// then every fifth, newest first, as Studio lists them.
pub(super) fn sample_epochs(latest: u32) -> Vec<u32> {
    (1..latest).rev().filter(|&e| e == 1 || e % 5 == 0).collect()
}

/// The tiles in a row: two rows of four.
const PER_ROW: usize = 4;

impl FoloApp {
    pub(super) fn samples_panel(
        &mut self,
        r: &Run,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // The samples come from each epoch's evaluation: training images when there's no val split.
        let images = if self.eval_split(r) == "train" { "training" } else { "validation" };
        let views = [SampleView::Both, SampleView::Predictions, SampleView::Truth];
        let sel = views.iter().position(|v| *v == self.sample_view).unwrap_or(0);
        let seg = self.segmented(
            "samples-view",
            &["Both", "Predicted", "Truth"],
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, _| this.sample_view = views[i]),
        );
        // The epoch picker: the latest evaluated epoch (followed while the run trains), or a past epoch held still.
        let latest = self.evaluated_epoch(r);
        let past = sample_epochs(latest);
        let picked = self.sample_epoch.filter(|e| *e < latest);
        let epoch_sel = picked.and_then(|e| past.iter().position(|p| *p == e)).map_or(0, |i| i + 1);
        let mut options: Vec<(SharedString, Option<SharedString>)> =
            vec![(format!("Latest (epoch {latest})").into(), None)];
        options.extend(past.iter().map(|e| (SharedString::from(format!("Epoch {e}")), None)));
        let current: SharedString = picked.map_or("Latest".into(), |e| format!("Epoch {e}").into());
        let rid = r.id.clone();
        let picker = (latest > 1).then(|| {
            self.select(
                "samples-epoch",
                current,
                options,
                epoch_sel,
                window,
                cx,
                Rc::new(move |this, i, _, cx| {
                    let e = if i == 0 { None } else { past.get(i - 1).copied() };
                    if e != this.sample_epoch {
                        this.sample_epoch = e;
                        this.store.samples.remove(&rid);
                        this.store.samples_stale.remove(&rid);
                        this.store.samples_refreshing.remove(&rid);
                        this.store.samples_refresh_error.remove(&rid);
                        this.ensure_samples(&rid, cx);
                    }
                    cx.notify();
                }),
            )
        });
        let conf = self.sample_conf;
        let (lo, hi, step) = SAMPLE_CONF;
        let slider = self.slider("sample-conf", conf, lo, hi, step, window, cx);
        let conf_tip = format!("Minimum confidence · {conf:.2}");
        let conf_ring = self.ring(&self.fh("sample-conf", cx), window);
        let classes = self.run_classes(r);
        let refresh_failed =
            latest > 0 && self.sample_epoch.is_none() && self.store.samples_refresh_error.contains_key(&r.id);
        let samples_failed = latest > 0
            && (self.store.samples.get(&r.id).is_some_and(|l| l.error().is_some()) || refresh_failed);
        let retry_anchor = match self.passing_stop("run-samples-retry", samples_failed, cx) {
            Some(fh) if !samples_failed => Some(div().track_focus(&fh)),
            _ => None,
        };
        let body = match self.store.samples.get(&r.id) {
            _ if latest == 0 => div()
                .child(empty_state(
                    &t,
                    "icons/image.svg",
                    "No evaluation yet",
                    &format!("Predictions on up to 8 fixed {images} images appear after the first epoch."),
                    Tone::Neutral,
                ))
                .into_any_element(),
            Some(Load::Ready(s)) => {
                let all_samples = s.samples.clone();
                let total = all_samples.len();
                let epoch = s.epoch;
                let view = self.sample_view;
                let active_filter = self.sample_error_filter;
                let active_sort = self.sample_sort;
                let analyzed: Vec<(usize, &crate::api::Sample, SampleErrorStats)> = all_samples
                    .iter()
                    .enumerate()
                    .map(|(i, smp)| (i, smp, analyze_sample(smp, conf)))
                    .collect();
                let counts = count_sample_filters(&analyzed);
                let matched = filter_and_sort_samples(analyzed, active_filter, active_sort);
                let matched_total = matched.len();
                let shown_samples: Vec<(crate::api::Sample, SampleErrorStats)> =
                    matched.into_iter().take(8).map(|(_, smp, st)| (smp.clone(), st)).collect();

                let filter_chips = [
                    ("smp-filter-all", "All", SampleErrorFilter::All, counts.all as u64),
                    ("smp-filter-fp", "FP", SampleErrorFilter::Fp, counts.fp as u64),
                    ("smp-filter-fn", "FN", SampleErrorFilter::Fn, counts.fn_count as u64),
                    ("smp-filter-tp", "TP", SampleErrorFilter::Tp, counts.tp as u64),
                ]
                .into_iter()
                .map(|(key, label, filter, cnt)| {
                    self.chip(
                        key,
                        label,
                        active_filter == filter,
                        None,
                        Some(cnt),
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.sample_error_filter = filter;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        match filter {
                            SampleErrorFilter::All => "Show all validation samples",
                            SampleErrorFilter::Fp => "Samples with false-positive detections (FP > 0)",
                            SampleErrorFilter::Fn => "Samples with missed ground-truth objects (FN > 0)",
                            SampleErrorFilter::Tp => "Samples with matched true-positive detections (TP > 0)",
                        },
                        None,
                    ))
                    .into_any_element()
                })
                .collect::<Vec<_>>();
                let sort_labels: Vec<&str> = SampleSort::ALL.iter().map(|s| s.label()).collect();
                let sort_idx = SampleSort::ALL.iter().position(|&s| s == active_sort).unwrap_or(0);
                let sort_seg = self.segmented(
                    "smp-sort",
                    &sort_labels,
                    sort_idx,
                    window,
                    cx,
                    Rc::new(|this, i, _, cx| {
                        this.sample_sort = SampleSort::ALL[i.min(SampleSort::ALL.len() - 1)];
                        cx.notify();
                    }),
                );
                let error_bar = (!all_samples.is_empty()).then(|| {
                    div()
                        .pb(px(6.))
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .justify_between()
                        .gap(px(8.))
                        .child(div().flex().flex_wrap().items_center().gap(px(6.)).children(filter_chips))
                        .child(sort_seg)
                });

                let mut tiles: Vec<AnyElement> = shown_samples
                    .iter()
                    .enumerate()
                    .map(|(n, (smp, stats))| {
                        let img = self.image(&smp.dataset_id, &smp.image_id, ImgKind::Thumb, cx);
                        let mut boxes: Vec<DrawBox> = Vec::new();
                        if view != SampleView::Predictions {
                            boxes.extend(smp.truth.iter().map(|b| DrawBox {
                                truth: true,
                                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
                            }));
                        }
                        if view != SampleView::Truth {
                            boxes.extend(smp.pred.iter().filter(|p| p.score >= conf).map(|p| DrawBox {
                                // Class and score where the cell has room (viz drops the score below 140 px).
                                label: classes.get(p.class as usize).cloned().map(Into::into),
                                score: Some(p.score),
                                ..DrawBox::new(p.class, p.x1, p.y1, p.x2, p.y2)
                            }));
                        }
                        let dims = (smp.width as f32, smp.height as f32);
                        // Failed, not loading: said on the cell, as on dataset tiles (its Retry is the
                        // inspector's, which the cell opens: its full size loading asks for it again).
                        let failed = img.is_none() && self.thumb_unavailable(&smp.dataset_id, &smp.image_id);
                        let painted = image_canvas(&t, img, dims, boxes, Fit::Cover, true, false);
                        // Square cells: every sample gets the same frame whatever its aspect. The
                        // ratio sits on an inner full-width box, whose width is definite. The mark
                        // lies on the square itself: a full-size wrapper in it would have no height.
                        let mut square = div()
                            .w_full()
                            .relative()
                            .rounded(px(radius::MD))
                            .overflow_hidden()
                            .bg(t.bg_inset)
                            .child(painted)
                            .when(failed, |s| s.child(crate::screens::unavailable_mark(&t, true)))
                            .child(
                                div()
                                    .absolute()
                                    .left(px(5.))
                                    .bottom(px(5.))
                                    .max_w(folo_ui::relative(0.92))
                                    .px(px(4.))
                                    .py(px(2.))
                                    .rounded(px(radius::SM))
                                    .bg(crate::theme::alpha(t.bg_app, 0.84))
                                    .child(sample_error_badges(&t, stats)),
                            );
                        square.style().aspect_ratio = Some(1.0);
                        // Each sample is a Tab stop: click or Enter opens it in the inspector (the
                        // image in full, its labels, and the dataset to walk from it).
                        let (ds, image, run) = (smp.dataset_id.clone(), smp.image_id.clone(), r.id.clone());
                        if self.ring(&self.fh(format!("smp-{n}"), cx), window) {
                            self.hint_keys(
                                "Enter opens the sample in the inspector, with this epoch's predictions",
                            );
                        }
                        let tile = div()
                            .id(SharedString::from(format!("smp-{n}")))
                            .flex_1()
                            .min_w(px(0.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .tooltip(crate::controls::tip(
                                format!(
                                    "TP {} · FP {} · FN {} · Open in the inspector",
                                    stats.tp, stats.fp, stats.fn_count
                                ),
                                None,
                            ))
                            .child(
                                square
                                    .relative()
                                    // Over an image a fill wouldn't show: the edge answers instead
                                    // (1 px on hover, 2 px while pressed, inside the tile so nothing
                                    // moves).
                                    .child(
                                        div()
                                            .id(SharedString::from(format!("smp-edge-{n}")))
                                            .absolute()
                                            .inset_0()
                                            .rounded(px(radius::MD))
                                            .border_2()
                                            .border_color(folo_ui::transparent_black())
                                            .group_hover(SharedString::from(format!("smp-{n}")), |s| {
                                                s.border_1().border_color(t.line_strong)
                                            })
                                            .group_active(SharedString::from(format!("smp-{n}")), |s| {
                                                s.border_2().border_color(t.text_secondary)
                                            }),
                                    ),
                            )
                            .group(SharedString::from(format!("smp-{n}")));
                        self.bind(
                            tile,
                            &format!("smp-{n}"),
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| {
                                this.open_sample(ds.clone(), image.clone(), run.clone(), w, cx)
                            }),
                        )
                        .into_any_element()
                    })
                    .collect();
                let empty_filter = (tiles.is_empty() && total > 0).then(|| {
                    let reset = self.button(
                        "smp-filter-reset",
                        "Show all samples",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, w, cx| {
                            this.sample_error_filter = SampleErrorFilter::All;
                            // Move focus off the disappearing reset button onto the persistent All chip.
                            let fh = this.fh("smp-filter-all", cx);
                            w.focus(&fh);
                            cx.notify();
                        }),
                    );
                    div()
                        .py(px(20.))
                        .px(px(12.))
                        .rounded(px(radius::MD))
                        .bg(t.bg_inset)
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .justify_between()
                        .gap(px(12.))
                        .child(
                            text(ty::SM, t.text_tertiary)
                                .whitespace_normal()
                                .child(format!("No samples match this filter at confidence ≥ {conf:.2}.")),
                        )
                        .child(reset)
                });
                // A short last row keeps its tiles the grid's size.
                if !tiles.is_empty() {
                    while !tiles.len().is_multiple_of(PER_ROW) {
                        tiles.push(div().flex_1().min_w(px(0.)).into_any_element());
                    }
                }
                let mut rows = Vec::new();
                let mut it = tiles.into_iter().peekable();
                while it.peek().is_some() {
                    rows.push(div().flex().items_start().gap(px(8.)).children(it.by_ref().take(PER_ROW)));
                }
                let refresh_notice = refresh_failed.then(|| {
                    let rid = r.id.clone();
                    let retry = self.button(
                        "run-samples-retry",
                        "Retry",
                        crate::controls::Btn::Secondary,
                        Some("icons/refresh-cw.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.retry_samples(&rid, cx)),
                    );
                    div()
                        .flex_none()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(text(ty::SM, t.danger_text).child("Latest refresh failed"))
                        .child(retry)
                });
                let scope_note = if active_filter == SampleErrorFilter::All {
                    shown_of(shown_samples.len(), total, images)
                } else {
                    format!(
                        "{} of {} matching (from {})",
                        shown_samples.len(),
                        matched_total,
                        shown_of(total, total, images)
                    )
                };
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .p(px(16.))
                    .children(error_bar)
                    .children(empty_filter)
                    .children(rows)
                    // The caption says what the tiles show, view by view (plus an inline Retry if
                    // a background Latest refresh failed while keeping the previous tiles mounted).
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap(px(12.))
                            .child(text(ty::SM, t.text_tertiary).min_w_0().tnum().child(format!(
                                "Epoch {epoch} · {scope_note} · {}",
                                match view {
                                    SampleView::Predictions => format!("predictions with score ≥ {conf:.2}"),
                                    SampleView::Truth => {
                                        format!("ground-truth labels · error badges at score ≥ {conf:.2}")
                                    }
                                    SampleView::Both =>
                                        format!("ground truth and predictions with score ≥ {conf:.2}"),
                                }
                            )))
                            .children(refresh_notice),
                    )
                    .into_any_element()
            }
            Some(Load::Failed(e)) => {
                let rid = r.id.clone();
                let retry = self.button(
                    "run-samples-retry",
                    "Retry",
                    crate::controls::Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_samples(&rid, cx)),
                );
                div()
                    .p(px(16.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(12.))
                    .child(
                        text(ty::SM, t.danger_text)
                            .flex_1()
                            .min_w_0()
                            .whitespace_normal()
                            .child(e.message.clone()),
                    )
                    .child(retry)
                    .into_any_element()
            }
            _ => div()
                .p(px(16.))
                .flex()
                .flex_col()
                .gap(px(8.))
                .children((0..2).map(|row| {
                    div().flex().gap(px(8.)).children((0..PER_ROW).map(move |i| {
                        let mut sq = div().w_full().child(skeleton(&t, format!("smp{}", row * PER_ROW + i)));
                        sq.style().aspect_ratio = Some(1.0);
                        div().flex_1().min_w(px(0.)).child(sq)
                    }))
                }))
                .into_any_element(),
        };
        // The class key and the minimum confidence share the row under the header. Keep the slider
        // mounted in Truth mode too because error counts, filtering, sorting, and badges still use
        // this confidence threshold.
        let tools = div()
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(12.))
            .px(px(16.))
            .pt(px(10.))
            .child(div().flex_1().min_w(px(180.)).flex().flex_wrap().gap(px(12.)).children(
                classes.into_iter().enumerate().map(|(i, name)| {
                    div()
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .child(class_swatch(&t, i as u32, 8.))
                        .child(text(ty::SM, t.text_secondary).child(name))
                }),
            ))
            .child(
                div()
                    .id("sample-conf-field")
                    .relative()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .tooltip(crate::controls::tip(conf_tip.clone(), Some("← →")))
                    .when(conf_ring, |d| {
                        d.child(crate::controls::focus_tip(&t, format!("{conf_tip} · ← → adjust"), 20.))
                    })
                    .child(text(ty::SM, t.text_tertiary).child("Min confidence"))
                    .child(div().w(px(120.)).child(slider))
                    .child(text(ty::SM, t.text_primary).tnum().w(px(28.)).child(format!("{conf:.2}"))),
            );
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(
                panel_header(
                    &t,
                    if images == "training" { "Training samples" } else { "Validation samples" },
                )
                .child(div().flex().items_center().gap(px(8.)).child(seg).children(picker)),
            )
            .child(tools)
            .children(retry_anchor)
            .child(body)
            .into_any_element()
    }
}

/// Compact per-sample TP / FP / FN badge strip overlaid inside each validation thumbnail.
fn sample_error_badges(t: &Theme, st: &SampleErrorStats) -> AnyElement {
    if !st.has_errors() && st.tp == 0 {
        return div().flex().items_center().child(badge(t, "Clean", Tone::Neutral)).into_any_element();
    }
    div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap(px(4.))
        .child(badge(t, format!("TP {}", st.tp), if st.tp > 0 { Tone::Success } else { Tone::Neutral }))
        .when(st.fp > 0, |d| d.child(badge(t, format!("FP {}", st.fp), Tone::Danger)))
        .when(st.fn_count > 0, |d| d.child(badge(t, format!("FN {}", st.fn_count), Tone::Warning)))
        .into_any_element()
}

/// How many of the run's fixed evaluation images the tiles show: "8 of 16 fixed validation
/// images", or all of them when the run has eight or fewer ("3 fixed validation images"). The
/// trainer picks up to 16 (docs/API.md, `/samples`); a small split has fewer.
fn shown_of(shown: usize, total: usize, images: &str) -> String {
    let noun = if total == 1 { "image" } else { "images" };
    if shown < total {
        format!("{shown} of {total} fixed {images} {noun}")
    } else {
        format!("{total} fixed {images} {noun}")
    }
}

#[cfg(test)]
mod tests {
    use super::{sample_epochs, shown_of};

    /// #382: the caption said "8 of 16" whatever the run had.
    #[test]
    fn the_caption_counts_the_images_the_run_has() {
        assert_eq!(shown_of(8, 16, "validation"), "8 of 16 fixed validation images");
        assert_eq!(shown_of(8, 12, "training"), "8 of 12 fixed training images");
        assert_eq!(shown_of(3, 3, "validation"), "3 fixed validation images");
        assert_eq!(shown_of(1, 1, "validation"), "1 fixed validation image");
    }

    #[test]
    fn the_picker_offers_the_first_epoch_and_every_fifth_newest_first() {
        assert_eq!(sample_epochs(23), vec![20, 15, 10, 5, 1]);
        assert_eq!(sample_epochs(20), vec![15, 10, 5, 1]);
        assert_eq!(sample_epochs(2), vec![1]);
        assert!(sample_epochs(1).is_empty());
    }
}
