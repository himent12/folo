//! An entry's images panel: sorted, paged, with clickable inline detail inspection of ground truth
//! and detections.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};

use super::format::images_shown;
use crate::api::{Bench, BenchAccuracy, BenchEntry, BenchEntryImage};
use crate::app::{FoloApp, act};
use crate::bench::{EntryImageReveal, ImageSort};
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
                let next = sorts[i.min(1)];
                if this.bench.image_sort != next {
                    this.bench.image_sort = next;
                    this.bench.selected_entry_image = None;
                    this.bench.entry_image_reveal = None;
                }
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
        let retry_shown = matches!(&state, Some((Load::Failed(_), _, _)));
        let retry_anchor = match self.passing_stop("bench-entry-images-retry", retry_shown, cx) {
            Some(fh) if !retry_shown => Some(div().track_focus(&fh)),
            _ => None,
        };
        let (page, loading_more, more_error) = match state {
            Some((Load::Ready(p), more, err)) => (p, more, err),
            Some((Load::Failed(err), _, _)) => {
                let (bid, eid) = (b.id.clone(), e.id.clone());
                let retry = self.button(
                    "bench-entry-images-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_entry_images(&bid, &eid, cx)),
                );
                return panel(&t)
                    .child(header)
                    .child(
                        div()
                            .p(px(16.))
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap(px(16.))
                            .child(text(ty::SM, t.danger_text).flex_1().min_w_0().child(err.message))
                            .child(retry),
                    )
                    .into_any_element();
            }
            _ => {
                return panel(&t)
                    .child(header)
                    .children(retry_anchor)
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
        let reveal_tile = match self.bench.entry_image_reveal.take() {
            Some(EntryImageReveal::Tile(id)) => Some(id),
            other => {
                self.bench.entry_image_reveal = other;
                None
            }
        };
        let page_scroll = self.page_scroll.clone();
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
                        act(move |this, w, cx| {
                            let target = (bid.clone(), eid.clone(), iid.clone());
                            if this.bench.selected_entry_image.as_ref() == Some(&target) {
                                this.bench.selected_entry_image = None;
                                this.bench.entry_image_reveal = None;
                            } else {
                                this.bench.selected_entry_image = Some(target);
                                this.bench.entry_image_reveal = Some(EntryImageReveal::Preview);
                                this.bench.entry_inspect_scroll.set_offset(folo_ui::point(px(0.), px(0.)));
                                let fh = this.fh("eimg-prev", cx);
                                w.focus(&fh);
                            }
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                };
                let ok = im.missed == 0 && im.wrong == 0;
                let caption_text = entry_image_caption(im);
                let should_reveal = reveal_tile.as_deref() == Some(im.image.id.as_str());
                let tile_scroll = page_scroll.clone();
                div()
                    .id(SharedString::from(format!("eimg-{}", im.image.id)))
                    .relative()
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
                    .when(should_reveal, |d| {
                        d.child(
                            folo_ui::canvas(
                                move |b, window, _| {
                                    crate::app::reveal_bounds_in_scroll(b, &tile_scroll, window)
                                },
                                |_, _, _, _| {},
                            )
                            .absolute()
                            .top_0()
                            .left_0()
                            .size_full(),
                        )
                    })
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
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(12.))
            .child(text(ty::SM, t.text_tertiary).tnum().flex_1().min_w(px(220.)).whitespace_normal().child(
                format!(
                    "{} of {} images · click any image to inspect · dashed: truth · solid: detections{}",
                    int(page.images.len() as u64),
                    int(page.total),
                    if e.scored { images_shown(&page, accuracy) } else { String::new() },
                ),
            ))
            .when_some(more_error, |d, err| d.child(text(ty::SM, t.danger_text).child(err.message)))
            .when(more, |d| {
                d.child(div().flex_none().child(if loading_more {
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
                }))
            });
        panel(&t)
            .child(header)
            .children(retry_anchor)
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
        let failed = img.is_none() && self.bench_image_failed(&b.id, &im.image.id, None);
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use folo_ui::{
        AppContext, Bounds, HeadlessApp, HeadlessOptions, KeyDownEvent, Keystroke, Modifiers, MouseButton,
        MouseDownEvent, MouseUpEvent, PlatformInput, WindowBounds, WindowHandle, WindowOptions, point, px,
        size,
    };

    use crate::api::BenchEntryStatus;
    use crate::api::mock::MockBackend;
    use crate::app::tests::drive;
    use crate::app::{FoloApp, Route};
    use crate::theme::{Mode, Theme};

    fn open_at(width: f32, height: f32) -> (HeadlessApp, WindowHandle<FoloApp>) {
        let api = Arc::new(MockBackend::new(4200, true));
        let window_size = size(px(width), px(height));
        let mut app = HeadlessApp::new(HeadlessOptions { size: window_size, ..Default::default() });
        let window = app.update(|cx| {
            crate::app::bind_keys(cx);
            crate::input::bind_keys(cx);
            cx.set_global(Theme::new(Mode::Dark));
            let bounds = Bounds::new(point(px(0.), px(0.)), window_size);
            let options =
                WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
            cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
                .expect("open window")
        });
        drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
        (app, window)
    }

    fn send_key(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str) {
        let ks = Keystroke::parse(key).unwrap();
        app.simulate_input(
            window.into(),
            PlatformInput::KeyDown(KeyDownEvent { keystroke: ks, is_held: false }),
        );
        app.frame(window.into());
    }

    /// Playtest R3 findings #1 & #4 (plus external entry deletion Esc guard):
    /// 1. Opening an entry preview via pointer click while scrolled to the bottom at 960×620
    ///    scrolls the preview into view even though pointer clicks leave keyboard mode (`kb == false`).
    /// 2. The details rail (`eimg-detail-side`) is focusable and scrolls via Up/Down/PageUp/PageDown/
    ///    Home/End without scrolling the outer page.
    /// 3. If the entry is deleted externally while its preview is open, Esc clears the stale
    ///    selection and navigates back instead of focusing a missing tile.
    #[test]
    fn entry_preview_reveals_on_pointer_open_scrolls_side_rail_and_handles_deleted_entry() {
        let (mut app, window) = open_at(960., 620.);
        let bid = "bn_5e12";
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Bench(bid.into()), w, cx));
        for _ in 0..3 {
            app.frame(window.into());
            app.run_until_idle();
        }
        let eid = drive(&mut app, window, |this, w, cx| {
            let eid = this
                .bench
                .entries
                .get(bid)
                .and_then(|l| l.ready())
                .unwrap()
                .iter()
                .find(|e| e.status == BenchEntryStatus::Done)
                .unwrap()
                .id
                .clone();
            this.navigate(Route::BenchEntry(bid.into(), eid.clone()), w, cx);
            eid
        });
        for _ in 0..4 {
            app.frame(window.into());
            app.run_until_idle();
        }

        // Scroll all the way to the bottom of the 960×620 page so the top of the Images panel is offscreen.
        let (bottom_y, last_img) = drive(&mut app, window, |this, _, cx| {
            let page = this
                .bench
                .entry_images
                .get_mut(&(bid.into(), eid.clone()))
                .and_then(|x| x.page.ready_mut())
                .expect("entry images loaded");
            let last = page.images.last().expect("has tiles").image.id.clone();
            // Give the last image 16 ground-truth boxes so its details rail overflows 340 px.
            let sample = page
                .images
                .iter()
                .find_map(|im| im.image.boxes.first().copied())
                .expect("demo benchmark has at least one box");
            let boxes = &mut page.images.last_mut().unwrap().image.boxes;
            while boxes.len() < 16 {
                boxes.push(sample);
            }
            let max_y = this.page_scroll.max_offset().height;
            this.page_scroll.set_offset(point(px(0.), -max_y));
            this.kb = false;
            cx.notify();
            (f32::from(max_y), last)
        });
        app.frame(window.into());
        assert!(bottom_y > 400., "page overflows vertically at 960×620 (bottom_y={bottom_y})");

        // Helper to send a real pointer click (leaves keyboard mode `kb == false`).
        let click_at = |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<FoloApp>, x: f32, y: f32| {
            let pos = point(px(x), px(y));
            app.simulate_input(
                w.into(),
                PlatformInput::MouseDown(MouseDownEvent {
                    button: MouseButton::Left,
                    position: pos,
                    modifiers: Modifiers::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
            );
            app.simulate_input(
                w.into(),
                PlatformInput::MouseUp(MouseUpEvent {
                    button: MouseButton::Left,
                    position: pos,
                    modifiers: Modifiers::default(),
                    click_count: 1,
                }),
            );
        };

        // Pointer-click the last tile in the bottom row (leaves keyboard mode `kb == false`):
        // scan right-to-left across the bottom row so the first hit is the rightmost tile
        // (`last_img`, since 12 tiles fill rows of 3 or 4 evenly), exercising the real tile click
        // handler without manually mutating selection or reveal state.
        let mut clicked_tile = None;
        for &cy in &[460., 500., 420.] {
            for &cx_pos in &[840., 760., 680., 600., 520., 440., 360.] {
                click_at(&mut app, window, cx_pos, cy);
                if drive(&mut app, window, |this, _, _| this.bench.selected_entry_image.is_some()) {
                    clicked_tile = Some((cx_pos, cy));
                    break;
                }
            }
            if clicked_tile.is_some() {
                break;
            }
        }
        let (tile_x, tile_y) = clicked_tile.expect("pointer click hit a tile in the bottom row");
        drive(&mut app, window, |this, _, _| {
            assert!(!this.kb, "pointer click keeps kb == false");
            assert_eq!(
                this.bench.selected_entry_image,
                Some((bid.into(), eid.clone(), last_img.clone())),
                "pointer click on bottom-right tile opens its inline preview"
            );
        });
        app.frame(window.into());
        assert!(
            app.wants_frame(window.into()),
            "prepaint reveal schedules follow-up frame when preview is offscreen"
        );
        app.frame(window.into());
        let revealed_y = drive(&mut app, window, |this, _, _| f32::from(-this.page_scroll.offset().y));
        assert!(
            revealed_y < bottom_y - 100.,
            "opening preview in pointer mode scrolls page up into view (before={bottom_y}, after={revealed_y})"
        );

        // Focus the details rail and verify keyboard scrolling moves `entry_inspect_scroll` without
        // moving `page_scroll`.
        let (side_max, page_before) = drive(&mut app, window, |this, w, cx| {
            this.kb = true;
            this.focus_key("eimg-detail-side", w, cx);
            cx.notify();
            (
                f32::from(this.bench.entry_inspect_scroll.max_offset().height),
                f32::from(-this.page_scroll.offset().y),
            )
        });
        app.frame(window.into());
        assert!(side_max > 40., "16-box details rail overflows its 340 px viewport (max={side_max})");

        send_key(&mut app, window, "down");
        let after_down =
            drive(&mut app, window, |this, _, _| f32::from(-this.bench.entry_inspect_scroll.offset().y));
        assert!(after_down >= 30., "Down scrolls details rail (got {after_down})");

        send_key(&mut app, window, "pagedown");
        let (after_pgdn, page_after) = drive(&mut app, window, |this, _, _| {
            (f32::from(-this.bench.entry_inspect_scroll.offset().y), f32::from(-this.page_scroll.offset().y))
        });
        assert!(
            after_pgdn > after_down,
            "PageDown advances details rail further ({after_pgdn} > {after_down})"
        );
        assert!(
            (page_after - page_before).abs() < 1.0,
            "PageDown on focused details rail does not scroll outer page ({page_before} vs {page_after})"
        );

        send_key(&mut app, window, "end");
        let at_end =
            drive(&mut app, window, |this, _, _| f32::from(-this.bench.entry_inspect_scroll.offset().y));
        assert!(
            (at_end - side_max).abs() < 1.5,
            "End reaches bottom of details rail ({at_end} vs {side_max})"
        );

        send_key(&mut app, window, "home");
        let at_home =
            drive(&mut app, window, |this, _, _| f32::from(-this.bench.entry_inspect_scroll.offset().y));
        assert_eq!(at_home, 0., "Home returns details rail to top");

        // Closing the preview in pointer mode (via Esc while `kb == false`) must reveal the
        // destination tile at the bottom of the gallery instead of leaving it offscreen.
        drive(&mut app, window, |this, _, _| this.kb = false);
        send_key(&mut app, window, "escape");
        drive(&mut app, window, |this, _, _| {
            assert!(!this.kb, "Esc in pointer mode keeps kb == false");
            assert_eq!(
                this.reveal_pending, 0,
                "Esc tile reveal uses EntryImageReveal without reveal_pending"
            );
            assert_eq!(this.bench.selected_entry_image, None, "Esc closes preview");
        });
        assert!(
            app.wants_frame(window.into()),
            "closing preview via Esc schedules follow-up frame to reveal destination tile"
        );
        app.frame(window.into());
        let after_esc_y = drive(&mut app, window, |this, _, _| f32::from(-this.page_scroll.offset().y));
        assert!(
            after_esc_y > revealed_y + 100.,
            "Esc in pointer mode scrolls page back down to destination tile (revealed={revealed_y}, after_esc={after_esc_y})"
        );

        // Round-trip via pointer click on the tile again, then pointer-click on Close preview:
        // must also reveal the destination tile at the bottom while staying in pointer mode.
        drive(&mut app, window, |this, _, cx| {
            let max_y = this.page_scroll.max_offset().height;
            this.page_scroll.set_offset(point(px(0.), -max_y));
            cx.notify();
        });
        app.frame(window.into());
        click_at(&mut app, window, tile_x, tile_y);
        app.frame(window.into());
        app.frame(window.into());
        drive(&mut app, window, |this, _, _| {
            assert_eq!(
                this.bench.selected_entry_image,
                Some((bid.into(), eid.clone(), last_img.clone())),
                "second pointer click reopens last tile preview"
            );
        });
        // Click the Close preview icon button at the top-right of the revealed preview bar.
        let mut closed_by_click = false;
        for &cy in &[76., 84., 92., 100.] {
            for &cx_pos in &[900., 892., 884., 908., 914.] {
                click_at(&mut app, window, cx_pos, cy);
                if drive(&mut app, window, |this, _, _| this.bench.selected_entry_image.is_none()) {
                    closed_by_click = true;
                    break;
                }
            }
            if closed_by_click {
                break;
            }
        }
        assert!(closed_by_click, "pointer click hit Close preview button");
        app.frame(window.into());
        assert!(
            app.wants_frame(window.into()),
            "Close preview click schedules follow-up frame to reveal destination tile"
        );
        app.frame(window.into());
        let after_close_y = drive(&mut app, window, |this, _, _| f32::from(-this.page_scroll.offset().y));
        assert!(
            after_close_y > revealed_y + 100.,
            "Close preview click scrolls page back down to destination tile (revealed={revealed_y}, after_close={after_close_y})"
        );

        // External deletion removes the entry from `bench.entries` while keeping cached `entry_images`:
        // pressing Esc must clear the stale selection and navigate back to `Route::Bench(bid)`.
        drive(&mut app, window, |this, _, cx| {
            this.bench.selected_entry_image = Some((bid.into(), eid.clone(), last_img.clone()));
            if let Some(list) = this.bench.entries.get_mut(bid).and_then(|l| l.ready_mut()) {
                list.retain(|e| e.id != eid);
            }
            cx.notify();
        });
        app.frame(window.into());
        send_key(&mut app, window, "escape");
        drive(&mut app, window, |this, _, _| {
            assert_eq!(this.bench.selected_entry_image, None, "Esc clears stale selection");
            assert_eq!(
                this.route,
                Route::Bench(bid.into()),
                "Esc navigates back when entry was deleted externally"
            );
        });
    }
}
