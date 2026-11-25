//! An entry's images panel: sorted, paged, with clickable inline detail inspection of ground truth
//! and detections.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::format::images_shown;
use crate::api::{Bench, BenchAccuracy, BenchEntry, BenchEntryImage};
use crate::app::{FoloApp, act};
use crate::bench::ImageSort;
use crate::controls::Btn;
use crate::screens::image_canvas;
use crate::store::Load;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

impl FoloApp {
    /// An entry's images: the truth (dashed) and its detections at the confidence they were taken at (its
    /// best-F1 one, or every detection without one), worst first or in the benchmark's order, a page at a time.
    pub(super) fn entry_images_panel(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        accuracy: &BenchAccuracy,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        self.ensure_entry_images(&b.id, &e.id, cx);
        let sort = self.bench.image_sort;
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let sorts = [ImageSort::Worst, ImageSort::Order];
        let seg = self.segmented(
            "entry-img-sort",
            &["Worst first", "In order"],
            sorts.iter().position(|s| *s == sort).unwrap_or(0),
            window,
            cx,
            std::rc::Rc::new(move |this, i, _, cx| {
                this.bench.image_sort = sorts[i.min(1)];
                this.ensure_entry_images(&bid, &eid, cx);
                cx.notify();
            }),
        );
        let header = panel_header(&t, "Images").child(seg);
        let state = self
            .bench
            .entry_images
            .get(&(b.id.clone(), e.id.clone()))
            .map(|x| (x.page.clone(), x.more, x.more_error.clone()));
        let (page, loading_more, more_error) = match state {
            Some((Load::Ready(p), more, err)) => (p, more, err),
            Some((Load::Failed(err), _, _)) => {
                return panel(&t)
                    .child(header)
                    .child(div().p(px(16.)).child(text(ty::SM, t.danger_text).child(err.message)))
                    .into_any_element();
            }
            _ => {
                return panel(&t)
                    .child(header)
                    .child(div().p(px(16.)).flex().flex_wrap().gap(px(8.)).children(
                        (0..8).map(|i| div().w(px(152.)).h(px(152.)).child(skeleton(&t, format!("eimg{i}")))),
                    ))
                    .into_any_element();
            }
        };
        let selected_id: Option<String> = self
            .bench
            .selected_entry_image
            .as_ref()
            .filter(|(sb, se, _)| sb == &b.id && se == &e.id)
            .map(|(_, _, img)| img.clone());
        let selected_idx =
            selected_id.as_deref().and_then(|sid| page.images.iter().position(|im| im.image.id == sid));
        let inspector =
            selected_idx.map(|idx| self.entry_image_detail(&t, b, e, &page.images, idx, window, cx));
        let tiles: Vec<AnyElement> = page
            .images
            .iter()
            .map(|im| {
                let img = self.bench_image(&b.id, &im.image.id, None, cx);
                let mut boxes: Vec<DrawBox> = im
                    .image
                    .boxes
                    .iter()
                    .map(|x| DrawBox { truth: true, ..DrawBox::new(x.class, x.x1, x.y1, x.x2, x.y2) })
                    .collect();
                boxes.extend(im.detections.iter().map(|d| DrawBox {
                    label: b.classes.get(d.class as usize).cloned().map(Into::into),
                    score: e.scored.then_some(d.score),
                    ..DrawBox::new(d.class, d.x1, d.y1, d.x2, d.y2)
                }));
                let dims = (im.image.width as f32, im.image.height as f32);
                // Failed, not loading: said on the frame, which is then a Retry (click, Enter, Space).
                let failed = img.is_none() && self.bench_image_failed(&b.id, &im.image.id, None);
                let is_selected = selected_id.as_deref() == Some(im.image.id.as_str());
                let painted = image_canvas(&t, img, dims, boxes, Fit::Contain, true, false);
                let painted =
                    if failed { crate::screens::unavailable_over(&t, painted, true) } else { painted };
                // The image clips inside; the control itself doesn't, so its focus ring shows whole.
                let frame = div()
                    .w(px(152.))
                    .h(px(152.))
                    .rounded(px(radius::MD))
                    .bg(t.bg_inset)
                    .border_1()
                    .border_color(if is_selected { t.accent_line } else { folo_ui::transparent_black() })
                    .child(div().size_full().rounded(px(radius::MD)).overflow_hidden().child(painted));
                let retry_key = format!("eimg-retry-{}", im.image.id);
                let retry_stop = (!failed).then(|| self.passing_stop(&retry_key, false, cx)).flatten();
                let frame = if failed {
                    self.passing_stop(&retry_key, true, cx);
                    let (bid, iid) = (b.id.clone(), im.image.id.clone());
                    let el = frame.id(SharedString::from(retry_key.clone())).cursor_pointer();
                    self.bind(
                        el,
                        &retry_key,
                        radius::MD,
                        window,
                        cx,
                        act(move |this, _, cx| this.retry_bench_image(&bid, &iid, None, cx)),
                    )
                    .into_any_element()
                } else {
                    let open_key = format!("eimg-open-{}", im.image.id);
                    let (bid, eid, iid) = (b.id.clone(), e.id.clone(), im.image.id.clone());
                    let el = frame
                        .id(SharedString::from(open_key.clone()))
                        .cursor_pointer()
                        .hover(|s| s.border_color(if is_selected { t.accent_line } else { t.line_strong }))
                        .active(|s| s.border_color(t.accent_line));
                    self.bind(
                        el,
                        &open_key,
                        radius::MD,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            let target = (bid.clone(), eid.clone(), iid.clone());
                            if this.bench.selected_entry_image.as_ref() == Some(&target) {
                                this.bench.selected_entry_image = None;
                            } else {
                                this.bench.selected_entry_image = Some(target);
                            }
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                };
                let ok = im.missed == 0 && im.wrong == 0;
                let caption_text = entry_image_caption(im);
                div()
                    .id(SharedString::from(format!("eimg-{}", im.image.id)))
                    .w(px(152.))
                    .flex_none()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .tooltip(crate::controls::tip(
                        if failed {
                            format!("{} · {} · Retry", im.image.file, crate::screens::TILE_UNAVAILABLE)
                        } else if is_selected {
                            format!("{} · Click to close preview", im.image.file)
                        } else {
                            format!("{} · Click to inspect", im.image.file)
                        },
                        None,
                    ))
                    .child(frame)
                    .when_some(retry_stop, |d, fh| d.child(div().track_focus(&fh)))
                    .child(
                        text(ty::SM, if ok { t.text_tertiary } else { t.text_secondary })
                            .tnum()
                            .child(caption_text),
                    )
                    .into_any_element()
            })
            .collect();
        let more = (page.images.len() as u64) < page.total;
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let footer = div()
            .px(px(16.))
            .pb(px(12.))
            .flex()
            .items_center()
            .justify_between()
            .gap(px(12.))
            .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                "{} of {} images · click any image to inspect · dashed: truth · solid: detections{}",
                int(page.images.len() as u64),
                int(page.total),
                if e.scored { images_shown(&page, accuracy) } else { String::new() },
            )))
            .when_some(more_error, |d, err| d.child(text(ty::SM, t.danger_text).child(err.message)))
            .when(more, |d| {
                d.child(if loading_more {
                    self.button_disabled(
                        "entry-img-more",
                        "Loading…",
                        Btn::Secondary,
                        Some("icons/chevron-down.svg"),
                        "Loading the next images",
                        window,
                        cx,
                    )
                } else {
                    self.button(
                        "entry-img-more",
                        "Show more",
                        Btn::Secondary,
                        Some("icons/chevron-down.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.more_entry_images(&bid, &eid, cx)),
                    )
                })
            });
        panel(&t)
            .child(header)
            .children(inspector)
            .child(div().p(px(16.)).flex().flex_wrap().gap(px(12.)).children(tiles))
            .child(footer)
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn entry_image_detail(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        images: &[BenchEntryImage],
        idx: usize,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let im = &images[idx];
        let show_truth = !self.bench.entry_inspect_hide_truth;
        let show_preds = !self.bench.entry_inspect_hide_preds;
        let img = self.bench_image(&b.id, &im.image.id, None, cx);
        let mut boxes = Vec::new();
        if show_truth {
            boxes.extend(
                im.image
                    .boxes
                    .iter()
                    .map(|x| DrawBox { truth: true, ..DrawBox::new(x.class, x.x1, x.y1, x.x2, x.y2) }),
            );
        }
        if show_preds {
            boxes.extend(im.detections.iter().map(|d| DrawBox {
                label: b.classes.get(d.class as usize).cloned().map(Into::into),
                score: e.scored.then_some(d.score),
                ..DrawBox::new(d.class, d.x1, d.y1, d.x2, d.y2)
            }));
        }
        let dims = (im.image.width as f32, im.image.height as f32);
        let stage = div()
            .flex_1()
            .min_w(px(320.))
            .h(px(340.))
            .rounded(px(radius::MD))
            .overflow_hidden()
            .bg(t.bg_inset)
            .border_1()
            .border_color(t.line_subtle)
            .child(image_canvas(t, img, dims, boxes, Fit::Contain, true, true));

        let n = images.len();
        let prev_id = images[(idx + n - 1) % n].image.id.clone();
        let next_id = images[(idx + 1) % n].image.id.clone();
        let (bid1, eid1) = (b.id.clone(), e.id.clone());
        let (bid2, eid2) = (b.id.clone(), e.id.clone());
        let prev = self.icon_button(
            "eimg-prev",
            "icons/chevron-left.svg",
            "Previous image",
            window,
            cx,
            act(move |this, _, cx| {
                this.bench.selected_entry_image = Some((bid1.clone(), eid1.clone(), prev_id.clone()));
                cx.notify();
            }),
        );
        let next = self.icon_button(
            "eimg-next",
            "icons/chevron-right.svg",
            "Next image",
            window,
            cx,
            act(move |this, _, cx| {
                this.bench.selected_entry_image = Some((bid2.clone(), eid2.clone(), next_id.clone()));
                cx.notify();
            }),
        );
        let close_img_id = im.image.id.clone();
        let close = self.icon_button(
            "eimg-close",
            "icons/x.svg",
            "Close preview",
            window,
            cx,
            act(move |this, w, cx| {
                this.bench.selected_entry_image = None;
                let fh = this.fh(format!("eimg-open-{close_img_id}"), cx);
                w.focus(&fh);
                cx.notify();
            }),
        );
        let truth_chip = self
            .chip(
                "eimg-toggle-truth",
                format!("Truth ({})", im.image.boxes.len()),
                show_truth,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.bench.entry_inspect_hide_truth = !this.bench.entry_inspect_hide_truth;
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip("Toggle dashed ground-truth boxes", None));
        let preds_chip = self
            .chip(
                "eimg-toggle-preds",
                format!("Detections ({})", im.detections.len()),
                show_preds,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.bench.entry_inspect_hide_preds = !this.bench.entry_inspect_hide_preds;
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip("Toggle solid detection boxes", None));

        let mut dets: Vec<_> = im.detections.iter().collect();
        dets.sort_by(|a, b| b.score.total_cmp(&a.score));
        let class_name =
            |cls: u32| b.classes.get(cls as usize).cloned().unwrap_or_else(|| format!("class {cls}"));
        let det_rows: Vec<AnyElement> = dets
            .into_iter()
            .map(|d| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .py(px(3.))
                    .child(class_swatch(t, d.class, 8.))
                    .child(
                        text(ty::SM, t.text_primary).flex_1().min_w_0().ellipsis().child(class_name(d.class)),
                    )
                    .when(e.scored, |el| {
                        el.child(text(ty::SM, t.text_primary).tnum().child(format!("{:.2}", d.score)))
                    })
                    .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                        "{:.0}×{:.0}",
                        (d.x2 - d.x1).max(0.),
                        (d.y2 - d.y1).max(0.)
                    )))
                    .into_any_element()
            })
            .collect();
        let truth_rows: Vec<AnyElement> = im
            .image
            .boxes
            .iter()
            .map(|x| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .py(px(3.))
                    .child(class_swatch(t, x.class, 8.))
                    .child(
                        text(ty::SM, t.text_secondary)
                            .flex_1()
                            .min_w_0()
                            .ellipsis()
                            .child(class_name(x.class)),
                    )
                    .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                        "{:.0}×{:.0}",
                        (x.x2 - x.x1).max(0.),
                        (x.y2 - x.y1).max(0.)
                    )))
                    .into_any_element()
            })
            .collect();

        let ok = im.missed == 0 && im.wrong == 0;
        let side = div()
            .id("eimg-detail-side")
            .w(px(270.))
            .flex_none()
            .max_h(px(340.))
            .overflow_y_scroll()
            .flex()
            .flex_col()
            .gap(px(10.))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(6.))
                    .child(badge(t, format!("{} × {}", im.image.width, im.image.height), Tone::Neutral))
                    .child(badge(t, entry_image_caption(im), if ok { Tone::Success } else { Tone::Warning })),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(overline(t, "Detections"))
                    .when(det_rows.is_empty(), |d| {
                        d.child(text(ty::SM, t.text_tertiary).child("No detections above threshold."))
                    })
                    .children(det_rows),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(overline(t, "Ground truth"))
                    .when(truth_rows.is_empty(), |d| {
                        d.child(text(ty::SM, t.text_tertiary).child("No ground-truth labels."))
                    })
                    .children(truth_rows),
            );

        let bar = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(8.))
            .child(prev)
            .child(next)
            .child(text(ty::SM, t.text_tertiary).tnum().child(format!("{} of {n}", idx + 1)))
            .child(
                text(ty::SM, t.text_primary)
                    .font_family(crate::theme::MONO)
                    .font_weight(FontWeight::MEDIUM)
                    .flex_1()
                    .min_w(px(120.))
                    .ellipsis()
                    .child(im.image.file.clone()),
            )
            .child(truth_chip)
            .child(preds_chip)
            .child(close);

        div()
            .px(px(16.))
            .py(px(12.))
            .border_b_1()
            .border_color(t.line_subtle)
            .bg(t.bg_raised)
            .flex()
            .flex_col()
            .gap(px(10.))
            .child(bar)
            .child(div().flex().flex_wrap().gap(px(16.)).child(stage).child(side))
            .into_any_element()
    }
}

fn entry_image_caption(im: &BenchEntryImage) -> String {
    if im.missed == 0 && im.wrong == 0 {
        "all right".to_string()
    } else {
        [(im.missed, "missed"), (im.wrong, "wrong")]
            .iter()
            .filter(|(n, _)| *n > 0)
            .map(|(n, w)| format!("{n} {w}"))
            .collect::<Vec<_>>()
            .join(" · ")
    }
}
