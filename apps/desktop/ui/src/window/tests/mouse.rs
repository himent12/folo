//! Mouse: hover styles and their redraws, clicks, click-out, capture / bubble order, `stop_propagation`,
//! `occlude`, group hover, cursor styles, tooltips on the fake clock, window control areas.

use std::time::Duration;

use super::*;
use crate::{
    CursorStyle, InteractiveElement, ParentElement, StatefulInteractiveElement, Styled, WindowControlArea,
    div,
};

#[test]
fn hover_style_appears_and_moves_within_the_hitbox_cost_no_frame() {
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().id("button").w(px(50.)).h(px(20.)).bg(color(1)).hover(|s| s.bg(color(2))))
            .into_any_element()
    });
    // The headless pointer starts at (0, 0), over the button: move it away first.
    mouse_move(&mut app, window, 150., 50.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1);
    mouse_move(&mut app, window, 10., 10.);
    assert!(app.is_dirty(window), "entering the hitbox redraws");
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(2)).len(), 1, "the hover style is painted");
    assert!(quads_of(&mut app, window, color(1)).is_empty());
    mouse_move(&mut app, window, 30., 12.);
    assert!(!app.is_dirty(window), "a move within the same hitbox does not redraw");
    mouse_move(&mut app, window, 150., 50.);
    assert!(app.is_dirty(window), "leaving redraws");
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1);
    // Leaving the window (no `MouseExited` on Windows: the hover callback) clears hover too.
    mouse_move(&mut app, window, 10., 10.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(2)).len(), 1);
    with_window(&mut app, window, |window, cx| window.on_hover_status_change(false, cx));
    assert!(app.is_dirty(window));
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1, "no hover once the mouse left the window");
}

#[test]
fn click_needs_down_and_up_on_the_element_and_click_out_fires_outside() {
    let events = log();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let events = events.clone();
        move |_, _| {
            let (a, b) = (events.clone(), events.clone());
            div()
                .size_full()
                .child(
                    div()
                        .id("button")
                        .w(px(50.))
                        .h(px(20.))
                        .on_click(move |event, _, _| {
                            a.borrow_mut().push(format!("click {}", event.click_count()))
                        })
                        .on_mouse_down_out(move |_, _, _| b.borrow_mut().push("down out".into())),
                )
                .into_any_element()
        }
    });
    mouse_down(&mut app, window, 10., 10.);
    mouse_up(&mut app, window, 12., 12.);
    app.frame(window);
    mouse_down(&mut app, window, 10., 10.);
    mouse_up(&mut app, window, 150., 50.);
    app.frame(window);
    mouse_down(&mut app, window, 150., 50.);
    mouse_up(&mut app, window, 10., 10.);
    assert_eq!(*events.borrow(), ["click 1", "down out"]);
}

#[test]
fn capture_runs_root_first_bubble_leaf_first_and_stop_propagation_ends_it() {
    let events = log();
    let stop = Rc::new(std::cell::Cell::new(false));
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let (events, stop) = (events.clone(), stop.clone());
        move |_, _| {
            let (a, b, c, d) = (events.clone(), events.clone(), events.clone(), events.clone());
            let stop = stop.clone();
            div()
                .size_full()
                .capture_any_mouse_down(move |_, _, _| a.borrow_mut().push("parent capture".into()))
                .on_mouse_down(MouseButton::Left, move |_, _, _| b.borrow_mut().push("parent bubble".into()))
                .child(
                    div()
                        .w(px(50.))
                        .h(px(20.))
                        .capture_any_mouse_down(move |_, _, _| c.borrow_mut().push("child capture".into()))
                        .on_mouse_down(MouseButton::Left, move |_, _, cx| {
                            d.borrow_mut().push("child bubble".into());
                            if stop.get() {
                                cx.stop_propagation();
                            }
                        }),
                )
                .into_any_element()
        }
    });
    mouse_down(&mut app, window, 10., 10.);
    assert_eq!(*events.borrow(), ["parent capture", "child capture", "child bubble", "parent bubble"]);
    events.borrow_mut().clear();
    stop.set(true);
    mouse_down(&mut app, window, 10., 10.);
    assert_eq!(*events.borrow(), ["parent capture", "child capture", "child bubble"]);
}

#[test]
fn occlude_blocks_what_is_under_it() {
    let events = log();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let events = events.clone();
        move |_, _| {
            let (a, b) = (events.clone(), events.clone());
            div()
                .size_full()
                .child(
                    div()
                        .w(px(100.))
                        .h(px(100.))
                        .bg(color(1))
                        .hover(|s| s.bg(color(2)))
                        .on_mouse_down(MouseButton::Left, move |_, _, _| a.borrow_mut().push("under".into())),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(50.))
                        .top(px(0.))
                        .w(px(100.))
                        .h(px(100.))
                        .occlude()
                        .on_mouse_down(MouseButton::Left, move |_, _, _| b.borrow_mut().push("over".into())),
                )
                .into_any_element()
        }
    });
    mouse_move(&mut app, window, 75., 50.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1, "no hover under an occluding element");
    mouse_down(&mut app, window, 75., 50.);
    mouse_down(&mut app, window, 25., 50.);
    assert_eq!(*events.borrow(), ["over", "under"]);
}

#[test]
fn group_hover_styles_a_child_when_the_group_is_hovered() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| {
            div()
                .size_full()
                .child(
                    div().group("row").w(px(100.)).h(px(40.)).child(
                        div().w(px(10.)).h(px(10.)).bg(color(1)).group_hover("row", |s| s.bg(color(2))),
                    ),
                )
                .into_any_element()
        });
    mouse_move(&mut app, window, 150., 80.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1);
    mouse_move(&mut app, window, 80., 30.);
    assert!(app.is_dirty(window), "hovering the group (not the child) redraws");
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(2)).len(), 1);
    mouse_move(&mut app, window, 150., 80.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1);
}

#[test]
fn cursor_comes_from_the_topmost_hovered_hitbox() {
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, |_, _| {
        div()
            .size_full()
            .cursor_pointer()
            .child(div().w(px(50.)).h(px(50.)).cursor(CursorStyle::IBeam))
            .into_any_element()
    });
    mouse_move(&mut app, window, 10., 10.);
    assert_eq!(app.platform().cursor_style.get(), CursorStyle::IBeam);
    mouse_move(&mut app, window, 100., 10.);
    assert_eq!(app.platform().cursor_style.get(), CursorStyle::PointingHand);
}

#[test]
fn tooltip_appears_after_its_delay_and_hides_when_the_mouse_leaves() {
    let (mut app, window) = open(size(px(300.), px(200.)), 1.0, |_, _| {
        div()
            .size_full()
            // A hover style, as the app's buttons have: GPUI redraws on leaving only through it, and the
            // tooltip hides at the next draw.
            .child(div().id("tipped").w(px(50.)).h(px(20.)).hover(|s| s.bg(color(3))).tooltip(|_, cx| {
                let render: RenderFn =
                    Rc::new(|_, _| div().w(px(30.)).h(px(10.)).bg(color(9)).into_any_element());
                cx.new(|_| TestView { render }).into()
            }))
            .into_any_element()
    });
    mouse_move(&mut app, window, 10., 10.);
    app.advance_clock(Duration::from_millis(400));
    app.frame(window);
    assert!(quads_of(&mut app, window, color(9)).is_empty(), "not before the delay");
    app.advance_clock(Duration::from_millis(100));
    assert!(app.is_dirty(window), "the delay elapsed: redraw with the tooltip");
    app.frame(window);
    // Placed 1 px right of and below the mouse.
    assert_bounds(quad_of(&mut app, window, color(9)), device(11., 11., 30., 10., 1.));
    mouse_move(&mut app, window, 200., 150.);
    app.frame(window);
    assert!(quads_of(&mut app, window, color(9)).is_empty(), "hidden once the mouse left");
}

#[test]
fn window_control_area_is_reported_unless_occluded() {
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().w_full().h(px(30.)).window_control_area(WindowControlArea::Drag))
            .child(div().absolute().left(px(150.)).top(px(0.)).w(px(50.)).h(px(30.)).occlude())
            .into_any_element()
    });
    let hit = |app: &mut HeadlessApp, x, y| {
        mouse_move(app, window, x, y);
        with_window(app, window, |window, _| window.hit_test_window_control())
    };
    assert_eq!(hit(&mut app, 20., 10.), Some(WindowControlArea::Drag));
    assert_eq!(hit(&mut app, 170., 10.), None, "under an occluding element");
    assert_eq!(hit(&mut app, 20., 60.), None);
}

#[test]
fn a_file_drop_reaches_on_drop_and_drag_moves_see_the_paths() {
    let events = log();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let events = events.clone();
        move |_, _| {
            let (a, b) = (events.clone(), events.clone());
            div()
                .size_full()
                .child(
                    div()
                        .id("target")
                        .w(px(100.))
                        .h(px(50.))
                        .on_drag_move(move |event: &crate::DragMoveEvent<crate::ExternalPaths>, _, cx| {
                            a.borrow_mut().push(format!("move {}", event.drag(cx).paths().len()))
                        })
                        .on_drop(move |paths: &crate::ExternalPaths, _, _| {
                            b.borrow_mut().push(format!("drop {}", paths.paths()[0].display()))
                        }),
                )
                .into_any_element()
        }
    });
    let paths = crate::ExternalPaths(smallvec::smallvec![std::path::PathBuf::from("a.png")]);
    let drop = |app: &mut HeadlessApp, event| app.simulate_input(window, PlatformInput::FileDrop(event));
    drop(&mut app, crate::FileDropEvent::Entered { position: point(px(150.), px(80.)), paths });
    drop(&mut app, crate::FileDropEvent::Pending { position: point(px(20.), px(20.)) });
    drop(&mut app, crate::FileDropEvent::Submit { position: point(px(20.), px(20.)) });
    assert_eq!(*events.borrow(), ["move 1", "move 1", "drop a.png"]);
    assert!(
        with_window(&mut app, window, |window, _| window.active_drag.is_none()),
        "the drop ends the drag"
    );
}

/// A target that logs its drops, filling a 200 x 100 window's top-left 100 x 50 (hover colour 2).
fn drop_target(
    events: &Log,
) -> impl Fn(&mut Window, &mut crate::Context<TestView>) -> crate::AnyElement + 'static {
    let events = events.clone();
    move |_, _| {
        let events = events.clone();
        div()
            .size_full()
            .child(div().id("target").w(px(100.)).h(px(50.)).bg(color(1)).hover(|s| s.bg(color(2))).on_drop(
                move |paths: &crate::ExternalPaths, _, _| {
                    events.borrow_mut().push(format!("drop {}", paths.paths()[0].display()))
                },
            ))
            .into_any_element()
    }
}

#[test]
fn a_file_drop_from_outside_the_window_reaches_on_drop() {
    let events = log();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, drop_target(&events));
    // The pointer left the window, then drags a file in from outside (no mouse move reaches the window: the
    // platforms report the drag only as file-drop events).
    with_window(&mut app, window, |window, cx| window.on_hover_status_change(false, cx));
    app.frame(window);
    let paths = crate::ExternalPaths(smallvec::smallvec![std::path::PathBuf::from("a.png")]);
    let drop = |app: &mut HeadlessApp, event| app.simulate_input(window, PlatformInput::FileDrop(event));
    drop(&mut app, crate::FileDropEvent::Entered { position: point(px(20.), px(20.)), paths });
    app.frame(window);
    drop(&mut app, crate::FileDropEvent::Pending { position: point(px(25.), px(20.)) });
    drop(&mut app, crate::FileDropEvent::Submit { position: point(px(25.), px(20.)) });
    assert_eq!(*events.borrow(), ["drop a.png"]);
}

#[test]
fn a_file_drag_that_leaves_the_window_ends_its_hover() {
    let events = log();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, drop_target(&events));
    with_window(&mut app, window, |window, cx| window.on_hover_status_change(false, cx));
    app.frame(window);
    let paths = crate::ExternalPaths(smallvec::smallvec![std::path::PathBuf::from("a.png")]);
    let drop = |app: &mut HeadlessApp, event| app.simulate_input(window, PlatformInput::FileDrop(event));
    drop(&mut app, crate::FileDropEvent::Entered { position: point(px(20.), px(20.)), paths });
    app.frame(window);
    assert!(with_window(&mut app, window, |window, _| window.hovered.get()), "the drag hovers the window");
    // Hover styles are off during a drag; once it left, the target must not look hovered at the drag's
    // last position.
    drop(&mut app, crate::FileDropEvent::Exited);
    assert!(app.is_dirty(window), "leaving redraws");
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1, "no hover once the drag left the window");
    assert!(with_window(&mut app, window, |window, _| !window.hovered.get()));
    // The next real mouse move over the window hovers again.
    mouse_move(&mut app, window, 10., 10.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(2)).len(), 1);
    assert!(events.borrow().is_empty());
}
