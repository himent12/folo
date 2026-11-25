//! Frames: an idle window draws nothing and asks for nothing, a resize repaints at the new size, animation
//! frames keep the window dirty, scroll handles report and obey, and (manually) what a frame costs.

use std::time::Instant;

use super::*;
use crate::{
    InteractiveElement, ParentElement, RequestFrameOptions, ScrollHandle, StatefulInteractiveElement, Styled,
    div,
};

fn frame_count(app: &mut HeadlessApp, window: AnyWindowHandle) -> u64 {
    with_window(app, window, |window, _| window.frame_times.frame)
}

fn frame_requested(app: &HeadlessApp) -> bool {
    app.platform().windows.borrow()[0].frame_requested.get()
}

#[test]
fn an_idle_window_draws_no_frame_and_requests_none() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    assert!(!app.is_dirty(window));
    app.platform().windows.borrow()[0].frame_requested.set(false);
    let drawn = frame_count(&mut app, window);
    app.run_until_idle();
    with_window(&mut app, window, |window, cx| window.on_request_frame(RequestFrameOptions::default(), cx));
    assert_eq!(frame_count(&mut app, window), drawn, "a frame request on a clean window draws nothing");
    assert!(!frame_requested(&app), "and nothing asks for another");
    // A mouse move over an element with no hover behaviour costs no frame either.
    mouse_move(&mut app, window, 50., 50.);
    assert!(!app.is_dirty(window) && !frame_requested(&app));
    // A refresh asks the platform exactly once.
    with_window(&mut app, window, |window, _| {
        window.refresh();
        window.refresh();
    });
    assert!(app.is_dirty(window) && frame_requested(&app));
}

#[test]
fn resize_repaints_at_the_new_size() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    let drawn = frame_count(&mut app, window);
    // What both backends do inside a resize: the callback, then at once a frame that must be presented.
    with_window(&mut app, window, |window, cx| {
        window.on_resize(size(px(300.), px(120.)), 1.5, cx);
        let options = RequestFrameOptions { require_presentation: true, force_render: false };
        window.on_request_frame(options, cx);
    });
    assert_eq!(frame_count(&mut app, window), drawn + 1, "drawn once, synchronously");
    assert_bounds(quad_of(&mut app, window, color(1)), device(0., 0., 300., 120., 1.5));
    let framebuffer = app.frame(window);
    assert_eq!((framebuffer.width, framebuffer.height), (450, 180));
}

#[test]
fn a_window_resized_to_nothing_draws_and_comes_back() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    let present = |app: &mut HeadlessApp, to: Size<Pixels>| {
        with_window(app, window, |window, cx| {
            window.on_resize(to, 1.0, cx);
            let options = RequestFrameOptions { require_presentation: true, force_render: false };
            window.on_request_frame(options, cx);
        });
    };
    // Minimized (both sides zero), then collapsed to one side.
    for to in [size(px(0.), px(0.)), size(px(200.), px(0.)), size(px(0.), px(100.))] {
        present(&mut app, to);
        let framebuffer = app.frame(window);
        assert_eq!(framebuffer.width * framebuffer.height, 0);
    }
    present(&mut app, size(px(200.), px(100.)));
    let framebuffer = app.frame(window);
    assert_eq!((framebuffer.width, framebuffer.height), (200, 100));
    let want = framebuffer.pixel(100, 50);
    assert!(framebuffer.pixels.iter().all(|&p| p == want), "repainted whole");
}

#[test]
fn a_required_presentation_presents_the_whole_frame_even_when_nothing_changed() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    app.platform().windows.borrow()[0].last_damage.borrow_mut().clear();
    let drawn = frame_count(&mut app, window);
    with_window(&mut app, window, |window, cx| {
        let options = RequestFrameOptions { require_presentation: true, force_render: false };
        window.on_request_frame(options, cx);
    });
    assert_eq!(frame_count(&mut app, window), drawn, "nothing was dirty: no draw");
    let damage = app.platform().windows.borrow()[0].last_damage.borrow().clone();
    assert_eq!(damage.len(), 1);
    assert_eq!((damage[0].size.width.0, damage[0].size.height.0), (200, 100));
}

#[test]
fn animation_frames_keep_the_window_dirty_until_they_stop() {
    let frames_left = Rc::new(std::cell::Cell::new(2));
    let (mut app, window) = open(size(px(100.), px(100.)), 1.0, {
        let frames_left = frames_left.clone();
        move |window, _| {
            if frames_left.get() > 0 {
                frames_left.set(frames_left.get() - 1);
                window.request_animation_frame();
            }
            div().size_full().into_any_element()
        }
    });
    assert!(app.is_dirty(window), "the first frame asked for another");
    app.frame(window);
    assert!(app.is_dirty(window));
    app.frame(window);
    assert!(!app.is_dirty(window), "no request in the last frame: idle");
}

#[test]
fn scroll_handle_reports_bounds_and_scrolls_to_items() {
    let handle = ScrollHandle::new();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let handle = handle.clone();
        move |_, _| {
            div()
                .id("list")
                .size_full()
                .overflow_y_scroll()
                .track_scroll(&handle)
                .children((0..10).map(|i| div().h(px(30.)).flex_none().bg(color(i))))
                .into_any_element()
        }
    });
    assert_eq!(handle.max_offset(), size(px(0.), px(200.)));
    assert_eq!(handle.bounds(), Bounds::new(point(px(0.), px(0.)), size(px(200.), px(100.))));
    assert_eq!(handle.bounds_for_item(3).unwrap().origin.y, px(90.));
    // Item 5 (150..180) into a 100 px view: its bottom aligns with the bottom.
    handle.scroll_to_item(5);
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(handle.offset().y, px(-80.));
    assert_bounds(quad_of(&mut app, window, color(5)), device(0., 70., 200., 30., 1.));
    handle.scroll_to_bottom();
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(handle.offset().y, px(-200.));
    handle.set_offset(point(px(0.), px(50.)));
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(handle.offset().y, px(0.), "clamped at the next prepaint");
}

/// Measures the pipeline on a mid-sized tree: 40 rows of 12 cells (about 530 divs, 480 with hover styles
/// and listeners). Run with `cargo test -p folo-ui --release --features test-support frame_cost -- --ignored
/// --nocapture`.
#[test]
#[ignore = "measurement: run by hand in release"]
fn frame_cost_of_a_mid_sized_tree() {
    let (mut app, window) = open(size(px(1600.), px(1000.)), 1.5, |_, _| {
        div()
            .size_full()
            .flex()
            .flex_col()
            .p(px(8.))
            .gap(px(2.))
            .children((0..40).map(|row| {
                div().id(("row", row)).flex().gap(px(4.)).h(px(20.)).children((0..12).map(move |col| {
                    div()
                        .id(("cell", col))
                        .flex_1()
                        .h_full()
                        .rounded(px(3.))
                        .border_1()
                        .border_color(color(1))
                        .bg(color((row + col) % 7))
                        .hover(|s| s.bg(color(9)))
                        .on_click(|_, _, _| {})
                }))
            }))
            .into_any_element()
    });
    for _ in 0..20 {
        with_window(&mut app, window, |window, _| window.refresh());
        app.frame(window);
    }
    let runs = 200;
    let mut totals = crate::FrameTimes::default();
    let start = Instant::now();
    for _ in 0..runs {
        with_window(&mut app, window, |window, _| window.refresh());
        app.frame(window);
        let t = with_window(&mut app, window, |window, _| window.last_frame_times());
        totals.build_us += t.build_us;
        totals.layout_us += t.layout_us;
        totals.prepaint_us += t.prepaint_us;
        totals.paint_us += t.paint_us;
        totals.finish_us += t.finish_us;
        totals.raster_us += t.raster_us;
        totals.present_us += t.present_us;
        totals.primitives = t.primitives;
        totals.hitboxes = t.hitboxes;
    }
    let wall = start.elapsed().as_micros() as u32 / runs;
    let r = runs;
    eprintln!(
        "mid-sized tree ({} prims, {} hitboxes), mean over {r} frames: build {} layout {} prepaint {} paint {} finish {} raster {} present {} us; wall {} us/frame (harness included)",
        totals.primitives,
        totals.hitboxes,
        totals.build_us / r,
        totals.layout_us / r,
        totals.prepaint_us / r,
        totals.paint_us / r,
        totals.finish_us / r,
        totals.raster_us / r,
        totals.present_us / r,
        wall,
    );
}

/// A system appearance change reaches the handler `on_appearance_changed` registered through the
/// platform's callback (the path Win32's `WM_SETTINGCHANGE` and winit's theme event take), with the
/// window seeing the new appearance and redrawn; the handler stays registered for the next change,
/// and a handler that sets the frame's appearance reaches the platform window.
#[test]
fn a_system_appearance_change_reaches_the_window_handler() {
    use crate::WindowAppearance;
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    let seen: Rc<RefCell<Vec<WindowAppearance>>> = Rc::default();
    with_window(&mut app, window, |window, cx| {
        assert_eq!(window.appearance(), WindowAppearance::Dark, "the headless system starts dark");
        let seen = seen.clone();
        window.on_appearance_changed(cx, move |window, _| {
            let appearance = window.appearance();
            seen.borrow_mut().push(appearance);
            window.set_frame_appearance(Some(appearance));
        });
    });
    app.frame(window);
    assert!(!app.is_dirty(window));
    app.set_system_appearance(WindowAppearance::Light);
    assert_eq!(*seen.borrow(), [WindowAppearance::Light]);
    assert!(app.is_dirty(window), "the window redraws in the new appearance");
    assert_eq!(app.platform().frame_appearances(), [Some(WindowAppearance::Light)]);
    assert_eq!(app.read(|cx| cx.window_appearance()), WindowAppearance::Light);
    // No change, no call; the next change reaches the same handler.
    app.set_system_appearance(WindowAppearance::Light);
    app.set_system_appearance(WindowAppearance::Dark);
    assert_eq!(*seen.borrow(), [WindowAppearance::Light, WindowAppearance::Dark]);
    assert_eq!(app.platform().frame_appearances(), [Some(WindowAppearance::Dark)]);
}

#[test]
fn text_style_stack_cascades_and_restores_across_nesting() {
    use std::sync::Arc;

    use crate::{FontFeatures, FontWeight, TextAlign, TextStyle, TextStyleRefinement};

    assert!(
        Arc::ptr_eq(&FontFeatures::default().0, &FontFeatures::default().0),
        "default FontFeatures should share a static empty Arc"
    );
    assert!(
        Arc::ptr_eq(&FontFeatures::disable_ligatures().0, &FontFeatures::disable_ligatures().0),
        "disable_ligatures should share a static Arc"
    );

    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().into_any_element());
    with_window(&mut app, window, |window, _| {
        let default_style = TextStyle::default();
        assert!(window.text_style_ref().is_none());
        assert_eq!(window.text_style(), default_style);
        assert_eq!(window.line_height(), default_style.line_height_in_pixels(window.rem_size()));

        // `None` leaves the stack untouched.
        window.with_text_style(None, |window| {
            assert!(window.text_style_ref().is_none());
        });

        let outer = TextStyleRefinement {
            color: Some(color(2)),
            font_size: Some(px(20.).into()),
            text_align: Some(TextAlign::Center),
            ..Default::default()
        };
        let inner = TextStyleRefinement {
            font_weight: Some(FontWeight::BOLD),
            line_height: Some(px(30.).into()),
            ..Default::default()
        };
        window.with_text_style_ref(Some(&outer), |window| {
            let outer_resolved = window.text_style_ref().expect("outer pushed").clone();
            assert_eq!(outer_resolved.color, color(2));
            assert_eq!(outer_resolved.font_size, px(20.).into());
            assert_eq!(outer_resolved.text_align, TextAlign::Center);
            assert_eq!(outer_resolved.font_weight, FontWeight::NORMAL);
            assert_eq!(window.line_height(), outer_resolved.line_height_in_pixels(window.rem_size()));

            window.with_text_style(Some(inner.clone()), |window| {
                let inner_resolved = window.text_style();
                assert_eq!(inner_resolved.color, color(2));
                assert_eq!(inner_resolved.font_size, px(20.).into());
                assert_eq!(inner_resolved.text_align, TextAlign::Center);
                assert_eq!(inner_resolved.font_weight, FontWeight::BOLD);
                assert_eq!(window.line_height(), px(30.));
            });

            // Popping restores the outer resolved style.
            assert_eq!(window.text_style(), outer_resolved);
            assert_eq!(window.line_height(), outer_resolved.line_height_in_pixels(window.rem_size()));
        });

        // Deep nesting beyond SmallVec<[TextStyle; 8]> inline capacity cascades and unwinds cleanly.
        fn nest(window: &mut Window, depth: usize) {
            if depth == 0 {
                assert_eq!(window.text_style().font_size, px(24.).into());
                assert_eq!(window.text_style().color, color(9));
                return;
            }
            let r = TextStyleRefinement {
                color: (depth == 1).then_some(color(9)),
                font_size: (depth == 10).then_some(px(24.).into()),
                ..Default::default()
            };
            window.with_text_style_ref(Some(&r), |window| nest(window, depth - 1));
        }
        nest(window, 10);
        assert!(window.text_style_ref().is_none());
        assert_eq!(window.text_style(), default_style);
    });
}

#[test]
fn deferred_draw_inherits_enclosing_text_style_in_prepaint() {
    use crate::{TextAlign, TextStyle, canvas, deferred};

    let observed: Rc<RefCell<Option<TextStyle>>> = Rc::default();
    let observed_in_canvas = observed.clone();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, move |_, _| {
        let observed = observed_in_canvas.clone();
        div()
            .size_full()
            .text_color(color(4))
            .text_size(px(19.))
            .text_right()
            .child(deferred(
                canvas(
                    move |_bounds, window, _cx| {
                        *observed.borrow_mut() = Some(window.text_style());
                    },
                    |_bounds, (), _window, _cx| {},
                )
                .size_full(),
            ))
            .into_any_element()
    });
    let seen = observed.borrow().clone().expect("deferred canvas prepainted");
    assert_eq!(seen.color, color(4));
    assert_eq!(seen.font_size, px(19.).into());
    assert_eq!(seen.text_align, TextAlign::Right);
    with_window(&mut app, window, |window, _| {
        assert!(window.text_style_ref().is_none(), "stack cleared after deferred prepaint");
    });
}

#[test]
fn steady_frame_reuses_global_element_id_arcs_and_evicts_removed_ids() {
    use std::cell::Cell;
    use std::sync::Arc;

    use crate::{ElementId, GlobalElementId};

    let show_transient = Rc::new(Cell::new(true));
    let seen_ids: Rc<RefCell<Vec<GlobalElementId>>> = Rc::default();
    let show_in_render = show_transient.clone();
    let seen_in_render = seen_ids.clone();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, move |window, _| {
        let seen = seen_in_render.clone();
        let transient = show_in_render.get();
        window.with_id("scope", |window| {
            window.with_global_id(ElementId::from("stable"), |id, _| {
                seen.borrow_mut().push(id.clone());
            });
            if transient {
                window.with_global_id(ElementId::from("transient"), |id, _| {
                    seen.borrow_mut().push(id.clone());
                });
            }
        });
        div()
            .id("pure-id-only")
            .size_full()
            .child(div().id("clickable").size(px(20.)).on_click(|_, _, _| {}))
            .into_any_element()
    });

    let first_stable = seen_ids.borrow()[0].clone();
    assert_eq!(seen_ids.borrow().len(), 2);
    with_window(&mut app, window, |window, _| {
        // Pure-ID `div().id("pure-id-only")` stores no empty InteractiveElementState box, while
        // `clickable` stores its pending_mouse_down state.
        assert_eq!(window.rendered_frame.element_states.len(), 1);
    });

    // Frame 2: same paths reuse the exact same `Arc<[ElementId]>`.
    seen_ids.borrow_mut().clear();
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    let second_stable = seen_ids.borrow()[0].clone();
    assert!(
        Arc::ptr_eq(&first_stable.0, &second_stable.0),
        "steady-state frame should reuse GlobalElementId's Arc<[ElementId]>"
    );

    // Frame 3: removing `transient` drops it from the interned id map while keeping `stable`.
    show_transient.set(false);
    seen_ids.borrow_mut().clear();
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    let third_stable = seen_ids.borrow()[0].clone();
    assert!(Arc::ptr_eq(&first_stable.0, &third_stable.0));
    with_window(&mut app, window, |window, _| {
        let has_transient = window
            .rendered_frame
            .element_ids
            .values()
            .flatten()
            .any(|id| id.path().last() == Some(&ElementId::from("transient")));
        assert!(!has_transient, "removed element ids must be evicted after frame swap");

        // Calling `with_global_id` outside a draw must bypass the frame cache so idle input
        // callbacks cannot grow `next_frame.element_ids` unboundedly.
        assert!(window.next_frame.element_ids.is_empty());
        window.with_global_id(ElementId::from("outside-draw"), |id, _| {
            assert_eq!(id.path(), &[ElementId::from("outside-draw")]);
        });
        assert!(window.next_frame.element_ids.is_empty());
    });
}

#[test]
fn compute_style_preserves_existing_interactive_state_across_frame_swap() {
    use std::cell::Cell;

    use crate::{ElementId, Interactivity};

    let placeholder_frame = Rc::new(Cell::new(false));
    let placeholder_in_render = placeholder_frame.clone();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, move |window, cx| {
        if placeholder_in_render.get() {
            // Render a frame that only calls public `compute_style` with the same element id.
            window.with_global_id(ElementId::from("clickable"), |global_id, window| {
                let _ = Interactivity::default().compute_style(Some(global_id), None, window, cx);
            });
            div().size_full().into_any_element()
        } else {
            div().id("clickable").size_full().on_click(|_, _, _| {}).into_any_element()
        }
    });

    with_window(&mut app, window, |window, _| {
        assert_eq!(window.rendered_frame.element_states.len(), 1);
    });

    // Frame 2: only `compute_style` touches "clickable"; its non-empty state must survive the swap.
    placeholder_frame.set(true);
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    with_window(&mut app, window, |window, _| {
        assert_eq!(
            window.rendered_frame.element_states.len(),
            1,
            "compute_style must carry existing non-empty InteractiveElementState into the next frame"
        );
    });
}

#[test]
fn noop_blur_hover_and_active_callbacks_do_not_schedule_frames() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    let clear = |app: &mut HeadlessApp| {
        app.frame(window);
        app.platform().windows.borrow()[0].frame_requested.set(false);
    };

    // Blurring when nothing is focused is a no-op.
    clear(&mut app);
    with_window(&mut app, window, |window, _| window.blur());
    assert!(!app.is_dirty(window) && !frame_requested(&app), "noop blur must not redraw");

    // Duplicate hover status with the same value does not redraw; flipping it does.
    let initial_hovered = with_window(&mut app, window, |window, _| window.hovered.get());
    clear(&mut app);
    with_window(&mut app, window, |window, cx| window.on_hover_status_change(initial_hovered, cx));
    assert!(!app.is_dirty(window) && !frame_requested(&app), "duplicate hover status must not redraw");
    with_window(&mut app, window, |window, cx| window.on_hover_status_change(!initial_hovered, cx));
    assert!(app.is_dirty(window), "changed hover status must redraw");

    // Duplicate active status with unchanged modifiers does not redraw; flipping active does.
    let initial_active = with_window(&mut app, window, |window, _| window.active.get());
    clear(&mut app);
    app.platform().set_active(initial_active);
    assert!(!app.is_dirty(window) && !frame_requested(&app), "duplicate active status must not redraw");
    app.platform().set_active(!initial_active);
    assert!(app.is_dirty(window), "changed active status must redraw");
}
