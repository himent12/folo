//! Expanded inline detail inspector for a benchmark entry image: GT vs Prediction / Errors-only
//! overlays, per-image TP / FP / FN and mean IoU badges, and per-box match diagnostics.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::error_analysis::{DetMatch, ImageErrorStats};
use crate::api::{Bench, BenchEntry, BenchEntryImage};
use crate::app::{FoloApp, act};
use crate::bench::EntryImageReveal;
use crate::controls::Btn;
use crate::screens::image_canvas;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

impl FoloApp {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn entry_image_detail(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        items: &[(usize, &BenchEntryImage, ImageErrorStats)],
        idx: usize,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let (_, im, stats) = &items[idx];
        let show_truth = !self.bench.entry_inspect_hide_truth;
        let show_preds = !self.bench.entry_inspect_hide_preds;
        let errors_only = self.bench.entry_inspect_errors_only;
        let img = self.bench_image(&b.id, &im.image.id, None, cx);
        let failed = img.is_none() && self.bench_image_failed(&b.id, &im.image.id, None);
        let mut boxes = Vec::new();
        if show_truth {
            for (gi, x) in im.image.boxes.iter().enumerate() {
                let matched = stats.gt_matched.get(gi).copied().unwrap_or(false);
                if errors_only && matched {
                    continue;
                }
                boxes.push(DrawBox {
                    truth: true,
                    dim: errors_only && matched,
                    ..DrawBox::new(x.class, x.x1, x.y1, x.x2, x.y2)
                });
            }
        }
        if show_preds {
            for (di, d) in im.detections.iter().enumerate() {
                let is_tp = stats.det_matches.get(di).is_some_and(|m| m.is_tp());
                if errors_only && is_tp {
                    continue;
                }
                boxes.push(DrawBox {
                    label: b.classes.get(d.class as usize).cloned().map(Into::into),
                    score: e.scored.then_some(d.score),
                    ..DrawBox::new(d.class, d.x1, d.y1, d.x2, d.y2)
                });
            }
        }
        let dims = (im.image.width as f32, im.image.height as f32);
        let painted = image_canvas(t, img, dims, boxes, Fit::Contain, true, true);
        let painted = if failed { crate::screens::unavailable_over(t, painted, true) } else { painted };
        let stage = div()
            .flex_1()
            .min_w(px(320.))
            .h(px(340.))
            .rounded(px(radius::MD))
            .overflow_hidden()
            .bg(t.bg_inset)
            .border_1()
            .border_color(t.line_subtle)
            .child(painted);

        let n = items.len();
        let prev_id = items[(idx + n - 1) % n].1.image.id.clone();
        let next_id = items[(idx + 1) % n].1.image.id.clone();
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
                this.bench.entry_inspect_scroll.set_offset(folo_ui::point(px(0.), px(0.)));
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
                this.bench.entry_inspect_scroll.set_offset(folo_ui::point(px(0.), px(0.)));
                cx.notify();
            }),
        );
        let retry_btn = failed.then(|| {
            let (bid, iid) = (b.id.clone(), im.image.id.clone());
            self.button(
                "eimg-detail-retry",
                "Retry",
                Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| this.retry_bench_image(&bid, &iid, None, cx)),
            )
        });
        let close_bid = b.id.clone();
        let close_img_id = im.image.id.clone();
        let close = self.icon_button(
            "eimg-close",
            "icons/x.svg",
            "Close preview",
            window,
            cx,
            act(move |this, w, cx| {
                this.bench.selected_entry_image = None;
                this.bench.entry_image_reveal = Some(EntryImageReveal::Tile(close_img_id.clone()));
                let key = if this.bench_image_failed(&close_bid, &close_img_id, None) {
                    format!("eimg-retry-{close_img_id}")
                } else {
                    format!("eimg-open-{close_img_id}")
                };
                let fh = this.fh(key, cx);
                w.focus(&fh);
                cx.notify();
            }),
        );
        let truth_chip = self
            .chip(
                "eimg-toggle-truth",
                format!("GT ({})", im.image.boxes.len()),
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
                format!("Pred ({})", im.detections.len()),
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
            .tooltip(crate::controls::tip("Toggle solid prediction boxes", None));
        let errors_chip = self
            .chip(
                "eimg-toggle-errors",
                "Errors only",
                errors_only,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.bench.entry_inspect_errors_only = !this.bench.entry_inspect_errors_only;
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip("Show only missed GT (FN) and false-positive (FP) boxes", None));

        let mut dets: Vec<(usize, &_)> = im.detections.iter().enumerate().collect();
        dets.sort_by(|(_, a), (_, b)| b.score.total_cmp(&a.score));
        let class_name =
            |cls: u32| b.classes.get(cls as usize).cloned().unwrap_or_else(|| format!("class {cls}"));
        let det_rows: Vec<AnyElement> = dets
            .into_iter()
            .map(|(di, d)| {
                let m =
                    stats.det_matches.get(di).copied().unwrap_or(DetMatch::Fp { best_same_class_iou: 0.0 });
                let (tag_label, tag_tone) = match m {
                    DetMatch::Tp { iou, .. } => (format!("TP {iou:.2}"), Tone::Success),
                    DetMatch::Fp { best_same_class_iou } if best_same_class_iou > 0.0 => {
                        (format!("FP {best_same_class_iou:.2}"), Tone::Danger)
                    }
                    DetMatch::Fp { .. } => ("FP".to_string(), Tone::Danger),
                };
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
                    .child(badge(t, tag_label, tag_tone).flex_none())
                    .into_any_element()
            })
            .collect();
        let truth_rows: Vec<AnyElement> = im
            .image
            .boxes
            .iter()
            .enumerate()
            .map(|(gi, x)| {
                let matched = stats.gt_matched.get(gi).copied().unwrap_or(false);
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
                    .child(
                        badge(
                            t,
                            if matched { "TP" } else { "FN" },
                            if matched { Tone::Success } else { Tone::Warning },
                        )
                        .flex_none(),
                    )
                    .into_any_element()
            })
            .collect();

        let side_fh = self.fh("eimg-detail-side", cx);
        let side_ring = self.ring(&side_fh, window);
        if side_ring {
            self.hint_keys("↑ / ↓ or Page Up / Down scroll detections and ground truth");
        }
        let side_scroll = self.bench.entry_inspect_scroll.clone();
        let side_chrome = self.scroll_chrome(side_scroll.clone(), t.bg_raised, 0., 0., cx);
        let side_list = div()
            .id("eimg-detail-side")
            .track_focus(&side_fh)
            .w_full()
            .max_h(px(340.))
            .pr(px(8.))
            .overflow_y_scroll()
            .track_scroll(&side_scroll)
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _, cx| {
                let m = &ev.keystroke.modifiers;
                if m.control || m.alt || m.platform {
                    return;
                }
                let s = &this.bench.entry_inspect_scroll;
                let max = s.max_offset().height;
                if max <= px(0.) {
                    return;
                }
                let y = -s.offset().y;
                let page = (s.bounds().size.height - px(32.)).max(px(32.));
                let to = match ev.keystroke.key.as_str() {
                    "down" => y + px(32.),
                    "up" => y - px(32.),
                    "pagedown" => y + page,
                    "pageup" => y - page,
                    "home" => px(0.),
                    "end" => max,
                    _ => return,
                };
                s.set_offset(folo_ui::point(s.offset().x, -to.clamp(px(0.), max)));
                cx.stop_propagation();
                cx.notify();
            }))
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
                    .child(badge(
                        t,
                        format!("TP {}", stats.tp),
                        if stats.tp > 0 { Tone::Success } else { Tone::Neutral },
                    ))
                    .child(badge(
                        t,
                        format!("FP {}", stats.fp),
                        if stats.fp > 0 { Tone::Danger } else { Tone::Neutral },
                    ))
                    .child(badge(
                        t,
                        format!("FN {}", stats.fn_count),
                        if stats.fn_count > 0 { Tone::Warning } else { Tone::Neutral },
                    ))
                    .when_some(stats.mean_iou, |d, iou| {
                        d.child(badge(t, format!("IoU {iou:.2}"), Tone::Info))
                    }),
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
        let side = div()
            .relative()
            .w(px(270.))
            .flex_none()
            .rounded(px(radius::SM))
            .when(side_ring, |d| d.child(crate::app::focus_ring(t, radius::SM)))
            .child(side_list)
            .children(side_chrome);

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
            .children(retry_btn)
            .child(truth_chip)
            .child(preds_chip)
            .child(errors_chip)
            .child(close);

        let should_scroll = match self.bench.entry_image_reveal.take() {
            Some(EntryImageReveal::Preview) => true,
            other => {
                self.bench.entry_image_reveal = other;
                false
            }
        };
        let page_scroll = self.page_scroll.clone();
        div()
            .relative()
            .px(px(16.))
            .py(px(12.))
            .border_b_1()
            .border_color(t.line_subtle)
            .bg(t.bg_raised)
            .flex()
            .flex_col()
            .gap(px(10.))
            .when(should_scroll, |d| {
                d.child(
                    folo_ui::canvas(
                        move |b, window, _| crate::app::reveal_bounds_in_scroll(b, &page_scroll, window),
                        |_, _, _, _| {},
                    )
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full(),
                )
            })
            .child(bar)
            .child(div().flex().flex_wrap().gap(px(16.)).child(stage).child(side))
            .into_any_element()
    }
}
