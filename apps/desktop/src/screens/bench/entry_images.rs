//! An entry's images panel: error-analysis filter chips (All, Missed GT / FN, False Positives / FP,
//! Has errors), error-focused sorting, GT vs Prediction overlay toggles, per-image TP/FP/FN badges,
//! and clickable inline detail inspection.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::error_analysis::{
    EntryImageFilter, EntryImageOrder, ImageErrorStats, count_filter_matches, filter_and_sort_entry_images,
};
use super::format::images_shown;
use crate::api::{Bench, BenchAccuracy, BenchEntry};
use crate::app::{FoloApp, act};
use crate::bench::EntryImageReveal;
use crate::controls::Btn;
use crate::screens::image_canvas;
use crate::store::Load;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

impl FoloApp {
    /// An entry's images: GT (dashed) and detections at operating confidence, with error-analysis
    /// filter chips, sort controls, GT/Prediction overlay toggles, and per-image TP/FP/FN badges.
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
        let order = self.bench.entry_image_order;
        self.bench.image_sort = order.server_sort();
        self.ensure_entry_images(&b.id, &e.id, cx);
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let orders = [
            EntryImageOrder::MostErrors,
            EntryImageOrder::MostFn,
            EntryImageOrder::MostFp,
            EntryImageOrder::LowestIou,
            EntryImageOrder::InOrder,
        ];
        let seg = self.segmented(
            "entry-img-sort",
            &["Most errors", "Most FN", "Most FP", "Lowest IoU", "In order"],
            orders.iter().position(|s| *s == order).unwrap_or(0),
            window,
            cx,
            std::rc::Rc::new(move |this, i, _, cx| {
                let next = orders[i.min(orders.len() - 1)];
                if this.bench.entry_image_order != next {
                    this.bench.entry_image_order = next;
                    let server = next.server_sort();
                    if this.bench.image_sort != server {
                        this.bench.image_sort = server;
                        this.bench.selected_entry_image = None;
                        this.bench.entry_image_reveal = None;
                    }
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

        let counts = count_filter_matches(&page.images);
        let active_filter = self.bench.entry_image_filter;
        let show_truth = !self.bench.entry_inspect_hide_truth;
        let show_preds = !self.bench.entry_inspect_hide_preds;
        let filter_bar =
            self.entry_image_filter_bar(&t, active_filter, counts, show_truth, show_preds, window, cx);

        // Preserve exact server page order for MostErrors/InOrder when server already sorted,
        // and apply client-side error sorting for MostFn/MostFp/LowestIou.
        let filtered = if active_filter == EntryImageFilter::All && order == EntryImageOrder::MostErrors {
            page.images
                .iter()
                .enumerate()
                .map(|(i, im)| (i, im, super::error_analysis::analyze_entry_image(im)))
                .collect::<Vec<_>>()
        } else {
            filter_and_sort_entry_images(&page.images, active_filter, order)
        };

        let selected_id: Option<String> = self
            .bench
            .selected_entry_image
            .as_ref()
            .filter(|(sb, se, _)| sb == &b.id && se == &e.id)
            .map(|(_, _, img)| img.clone());
        let selected_idx =
            selected_id.as_deref().and_then(|sid| filtered.iter().position(|(_, im, _)| im.image.id == sid));
        let inspector = selected_idx.map(|idx| self.entry_image_detail(&t, b, e, &filtered, idx, window, cx));
        let reveal_tile = match self.bench.entry_image_reveal.take() {
            Some(EntryImageReveal::Tile(id)) => Some(id),
            other => {
                self.bench.entry_image_reveal = other;
                None
            }
        };
        let page_scroll = self.page_scroll.clone();
        let tiles: Vec<AnyElement> = filtered
            .iter()
            .map(|(_, im, stats)| {
                let img = self.bench_image(&b.id, &im.image.id, None, cx);
                let mut boxes: Vec<DrawBox> = Vec::new();
                if show_truth {
                    boxes.extend(
                        im.image.boxes.iter().map(|x| DrawBox {
                            truth: true,
                            ..DrawBox::new(x.class, x.x1, x.y1, x.x2, x.y2)
                        }),
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
                let failed = img.is_none() && self.bench_image_failed(&b.id, &im.image.id, None);
                let is_selected = selected_id.as_deref() == Some(im.image.id.as_str());
                let painted = image_canvas(&t, img, dims, boxes, Fit::Contain, true, false);
                let painted =
                    if failed { crate::screens::unavailable_over(&t, painted, true) } else { painted };
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
                let should_reveal = reveal_tile.as_deref() == Some(im.image.id.as_str());
                let tile_scroll = page_scroll.clone();
                let badges = tile_error_badges(&t, stats);
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
                            format!(
                                "{} · TP {} · FP {} · FN {} · Click to inspect",
                                im.image.file, stats.tp, stats.fp, stats.fn_count
                            )
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
                    .child(badges)
                    .into_any_element()
            })
            .collect();

        let empty_filter = tiles.is_empty().then(|| {
            let clear = self.button(
                "entry-img-clear-filter",
                "Show all images",
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.bench.entry_image_filter = EntryImageFilter::All;
                    cx.notify();
                }),
            );
            div()
                .p(px(16.))
                .flex()
                .flex_wrap()
                .items_center()
                .justify_between()
                .gap(px(12.))
                .child(
                    text(ty::SM, t.text_tertiary)
                        .whitespace_normal()
                        .child("No loaded images match this error filter."),
                )
                .child(clear)
        });

        let more = (page.images.len() as u64) < page.total;
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let shown_summary = if active_filter == EntryImageFilter::All && !order.is_client_subsort() {
            format!("{} of {} images", int(page.images.len() as u64), int(page.total))
        } else if active_filter == EntryImageFilter::All {
            if more {
                format!(
                    "{} loaded images sorted (of {} total · Show more to scan remaining)",
                    int(page.images.len() as u64),
                    int(page.total)
                )
            } else {
                format!("{} of {} images", int(page.images.len() as u64), int(page.total))
            }
        } else if more {
            format!(
                "{} matching in {} loaded ({} total · Show more to scan remaining)",
                int(filtered.len() as u64),
                int(page.images.len() as u64),
                int(page.total)
            )
        } else {
            format!("{} matching of {} images", int(filtered.len() as u64), int(page.total))
        };
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
                    "{shown_summary} · click any image to inspect · dashed: GT · solid: predictions{}",
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
            .child(filter_bar)
            .children(inspector)
            .children(empty_filter)
            .when(!tiles.is_empty(), |d| {
                d.child(div().p(px(16.)).flex().flex_wrap().gap(px(12.)).children(tiles))
            })
            .child(footer)
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn entry_image_filter_bar(
        &mut self,
        t: &Theme,
        active: EntryImageFilter,
        counts: super::error_analysis::FilterCounts,
        show_truth: bool,
        show_preds: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let chips = [
            ("eimg-filter-all", "All", EntryImageFilter::All, counts.all as u64),
            ("eimg-filter-fn", "Missed GT (FN)", EntryImageFilter::MissedGt, counts.missed_gt as u64),
            (
                "eimg-filter-fp",
                "False Positives (FP)",
                EntryImageFilter::FalsePositives,
                counts.false_positives as u64,
            ),
            ("eimg-filter-err", "Has errors", EntryImageFilter::HasErrors, counts.has_errors as u64),
        ]
        .into_iter()
        .map(|(key, label, filter, cnt)| {
            self.chip(
                key,
                label,
                active == filter,
                None,
                Some(cnt),
                window,
                cx,
                act(move |this, _, cx| {
                    this.bench.entry_image_filter = filter;
                    if let Some((sb, se, sid)) = this.bench.selected_entry_image.clone() {
                        let still_visible =
                            this.bench.entry_images.get(&(sb, se)).and_then(|x| x.page.ready()).is_some_and(
                                |p| {
                                    p.images.iter().any(|im| {
                                        im.image.id == sid
                                            && super::error_analysis::entry_image_matches_filter(im, filter)
                                    })
                                },
                            );
                        if !still_visible {
                            this.bench.selected_entry_image = None;
                            this.bench.entry_image_reveal = None;
                        }
                    }
                    cx.notify();
                }),
            )
            .into_any_element()
        })
        .collect::<Vec<_>>();

        let gt_toggle = self
            .chip(
                "eimg-grid-gt",
                "GT overlay",
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
            .tooltip(crate::controls::tip(
                "Toggle dashed ground-truth boxes across tiles and inspector",
                None,
            ));
        let pred_toggle = self
            .chip(
                "eimg-grid-preds",
                "Predictions",
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
            .tooltip(crate::controls::tip("Toggle solid prediction boxes across tiles and inspector", None));

        div()
            .px(px(16.))
            .py(px(8.))
            .border_b_1()
            .border_color(t.line_subtle)
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(8.))
            .child(div().flex().flex_wrap().items_center().gap(px(6.)).children(chips))
            .child(div().flex().items_center().gap(px(6.)).child(gt_toggle).child(pred_toggle))
            .into_any_element()
    }
}

/// Compact per-tile TP / FP / FN badge strip computed directly from each image's error stats.
fn tile_error_badges(t: &Theme, st: &ImageErrorStats) -> AnyElement {
    div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap(px(4.))
        .child(badge(t, format!("TP {}", st.tp), if st.tp > 0 { Tone::Success } else { Tone::Neutral }))
        .when(st.fp > 0, |d| d.child(badge(t, format!("FP {}", st.fp), Tone::Danger)))
        .when(st.fn_count > 0, |d| d.child(badge(t, format!("FN {}", st.fn_count), Tone::Warning)))
        .when(!st.has_errors() && st.tp == 0, |d| d.child(badge(t, "Clean", Tone::Neutral)))
        .into_any_element()
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
        let mut closed_by_click = false;
        for &cy in &[76., 84., 92., 100., 112., 120.] {
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
