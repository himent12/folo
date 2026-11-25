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
            // A hover style, as the app's buttons have (without one, see
            // `a_tooltip_without_a_hover_style_hides_when_the_mouse_leaves`).
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

/// A press the platform cancels (the capture went elsewhere, the window was deactivated) arrives as a
/// release outside the window (`platform/windows/events.rs`, `cancel_presses`), where a real release
/// lands when the button comes up elsewhere: it clicks nothing and drops nothing, although the pointer is
/// still over the element pressed, and what ends a press wherever it is released still runs.
/// This proves what the window does with that outside release; it supplies the release itself, so it
/// passes without the Win32 backend's cancellation too. The backend's part (a lost capture, `WM_CANCELMODE`
/// or deactivation reported as that release) is proved by `platform/windows/tests.rs`,
/// `a_cancelled_press_does_not_click`.
#[test]
fn a_cancelled_press_clicks_and_drops_nothing_and_ends_drags() {
    let events = log();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let events = events.clone();
        move |_, _| {
            let push = |name: &'static str| {
                let events = events.clone();
                move || events.borrow_mut().push(name.to_string())
            };
            let (root_up, root_up_out, up, click, drop) =
                (push("root up"), push("root up out"), push("up"), push("click"), push("drop"));
            div()
                .size_full()
                .on_mouse_up(MouseButton::Left, move |_, _, _| root_up())
                .on_mouse_up_out(MouseButton::Left, move |_, _, _| root_up_out())
                .child(
                    div()
                        .id("button")
                        .w(px(50.))
                        .h(px(20.))
                        .on_mouse_up(MouseButton::Left, move |_, _, _| up())
                        .on_click(move |_, _, _| click())
                        .on_drop(move |_: &crate::ExternalPaths, _, _| drop()),
                )
                .into_any_element()
        }
    });
    let take = || std::mem::take(&mut *events.borrow_mut());
    let cancel = |app: &mut HeadlessApp| mouse_up(app, window, -1., -1.);

    // Pressed on the button and cancelled with the pointer still over it.
    mouse_move(&mut app, window, 10., 10.);
    mouse_down(&mut app, window, 10., 10.);
    cancel(&mut app);
    assert_eq!(take(), ["root up out"], "the root's drag end runs, nothing is clicked");
    // The pending press is gone: a later release over the button clicks nothing either.
    app.frame(window);
    mouse_up(&mut app, window, 10., 10.);
    assert_eq!(take(), ["up", "root up"]);

    // A drag over the button, cancelled: no drop, and the drag is over.
    let paths = crate::ExternalPaths(smallvec::smallvec![std::path::PathBuf::from("a.png")]);
    app.simulate_input(
        window,
        PlatformInput::FileDrop(crate::FileDropEvent::Entered { position: point(px(10.), px(10.)), paths }),
    );
    assert!(with_window(&mut app, window, |window, _| window.active_drag.is_some()));
    cancel(&mut app);
    assert_eq!(take(), ["root up out"]);
    assert!(with_window(&mut app, window, |window, _| window.active_drag.is_none()), "the drag ended");

    // A press released over the button still clicks.
    app.frame(window);
    mouse_down(&mut app, window, 10., 10.);
    mouse_up(&mut app, window, 10., 10.);
    assert_eq!(take(), ["click", "up", "root up"]);
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

/// A tooltip view that counts its drops.
struct Tip(Rc<std::cell::Cell<u32>>);

impl crate::Render for Tip {
    fn render(&mut self, _: &mut Window, _: &mut crate::Context<Self>) -> impl IntoElement {
        div().w(px(30.)).h(px(10.)).bg(color(9))
    }
}

impl Drop for Tip {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

/// A window whose 50 x 20 element at the top left has a `Tip` tooltip (no hover style, no cursor), while
/// `shown` is set.
fn tipped_window(
    shown: Rc<std::cell::Cell<bool>>,
    drops: Rc<std::cell::Cell<u32>>,
) -> (HeadlessApp, AnyWindowHandle) {
    open(size(px(300.), px(200.)), 1.0, move |_, _| {
        let drops = drops.clone();
        let tipped = div().id("tipped").w(px(50.)).h(px(20.)).tooltip(move |_, cx| {
            let drops = drops.clone();
            cx.new(|_| Tip(drops)).into()
        });
        div().size_full().children(shown.get().then_some(tipped)).into_any_element()
    })
}

/// Hovers the tipped element until its tooltip shows.
fn show_tip(app: &mut HeadlessApp, window: AnyWindowHandle) {
    mouse_move(app, window, 10., 10.);
    app.advance_clock(Duration::from_millis(500));
    app.frame(window);
    assert_eq!(quads_of(app, window, color(9)).len(), 1, "the tooltip shows");
}

#[test]
fn a_tooltip_without_a_hover_style_hides_when_the_mouse_leaves() {
    let (mut app, window) = tipped_window(Rc::new(true.into()), Rc::default());
    show_tip(&mut app, window);
    mouse_move(&mut app, window, 200., 150.);
    assert!(app.is_dirty(window), "leaving the element redraws");
    app.frame(window);
    assert!(quads_of(&mut app, window, color(9)).is_empty(), "hidden once the mouse left");
}

#[test]
fn a_shown_tooltip_whose_element_goes_away_is_dropped() {
    let (shown, drops) = (Rc::new(std::cell::Cell::new(true)), Rc::new(std::cell::Cell::new(0)));
    let (mut app, window) = tipped_window(shown.clone(), drops.clone());
    show_tip(&mut app, window);
    shown.set(false);
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    app.run_until_idle();
    assert!(quads_of(&mut app, window, color(9)).is_empty());
    assert_eq!(drops.get(), 1, "the tooltip view is dropped with its element");
}

#[test]
fn a_tooltip_whose_element_goes_away_before_it_shows_is_dropped() {
    let (shown, drops) = (Rc::new(std::cell::Cell::new(true)), Rc::new(std::cell::Cell::new(0)));
    let (mut app, window) = tipped_window(shown.clone(), drops.clone());
    mouse_move(&mut app, window, 10., 10.);
    shown.set(false);
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    app.advance_clock(Duration::from_millis(500));
    app.run_until_idle();
    assert!(!app.is_dirty(window), "nothing left to show");
    assert_eq!(drops.get(), 0, "no tooltip is built for an element that is gone");
}

#[test]
fn active_style_listeners_are_skipped_without_active_styles_and_clear_when_removed_mid_press() {
    use std::cell::Cell;

    // 1. A stateful div with hover style only registers no active mouse-down listener and does not
    //    dirty the window on mouse down.
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().id("plain").w(px(60.)).h(px(30.)).bg(color(1)).hover(|s| s.bg(color(2))))
            .into_any_element()
    });
    let (down_len, move_len) = with_window(&mut app, window, |window, _| {
        (window.rendered_frame.mouse_down_listeners.len(), window.rendered_frame.mouse_move_listeners.len())
    });
    assert_eq!(down_len, 0, "no active mouse-down listener should be registered");
    assert_eq!(move_len, 1, "only the hover refresh listener should be registered");
    mouse_move(&mut app, window, 10., 10.);
    app.frame(window);
    assert!(!app.is_dirty(window));
    mouse_down(&mut app, window, 10., 10.);
    assert!(!app.is_dirty(window), "pressing a div without active styles should not trigger a redraw");

    // 2. A div with `.active(...)` applies its active style on press, keeps clearing it on release
    //    even if `.active(...)` is conditionally dropped mid-press, and works with `.group_active(...)`.
    let enable_active = Rc::new(Cell::new(true));
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let enable_active = enable_active.clone();
        move |_, _| {
            let mut btn = div().id("btn").w(px(60.)).h(px(30.)).bg(color(1));
            if enable_active.get() {
                btn = btn.active(|s| s.bg(color(3)));
            }
            div()
                .size_full()
                .child(btn)
                .child(
                    div().group("g").w(px(60.)).h(px(30.)).child(
                        div()
                            .id("g-child")
                            .w(px(20.))
                            .h(px(20.))
                            .bg(color(4))
                            .group_active("g", |s| s.bg(color(5))),
                    ),
                )
                .into_any_element()
        }
    });
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1);
    mouse_down(&mut app, window, 10., 10.);
    assert!(app.is_dirty(window));
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(3)).len(), 1, "active style applied while pressed");

    // Drop `.active(...)` mid-press, redraw, then release: clicked_state must still clear on MouseUp.
    enable_active.set(false);
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1);
    mouse_up(&mut app, window, 10., 10.);
    assert!(app.is_dirty(window), "mouse up still clears in-flight clicked_state");
    app.frame(window);
    enable_active.set(true);
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(1)).len(), 1, "active style stayed cleared after release");

    // Group active triggers when pressing anywhere inside the group.
    mouse_down(&mut app, window, 45., 45.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(5)).len(), 1, "group_active style applied");
    mouse_up(&mut app, window, 45., 45.);
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(4)).len(), 1, "group_active style cleared");
}

#[test]
fn inline_focus_and_hover_refresh_respect_interleaved_listener_order() {
    use std::cell::Cell;

    use crate::{
        DispatchPhase, MouseExitEvent, canvas,
        window::{MouseDownEntry, MouseMoveEntry},
    };

    let events = log();
    let prevent_child_down = Rc::new(Cell::new(true));
    let stop_capture_down = Rc::new(Cell::new(false));
    let parent_focus_slot: Rc<RefCell<Option<crate::FocusHandle>>> = Rc::default();
    let child_focus_slot: Rc<RefCell<Option<crate::FocusHandle>>> = Rc::default();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let events = events.clone();
        let prevent_child_down = prevent_child_down.clone();
        let stop_capture_down = stop_capture_down.clone();
        let parent_focus_slot = parent_focus_slot.clone();
        let child_focus_slot = child_focus_slot.clone();
        move |_, cx| {
            let parent_focus =
                parent_focus_slot.borrow_mut().get_or_insert_with(|| cx.focus_handle()).clone();
            let child_focus = child_focus_slot.borrow_mut().get_or_insert_with(|| cx.focus_handle()).clone();
            let log_parent_down = events.clone();
            let log_child_down = events.clone();
            let log_move = events.clone();
            let prevent_child_down = prevent_child_down.clone();
            let stop_capture_down = stop_capture_down.clone();
            div()
                .size_full()
                .track_focus(&parent_focus)
                .capture_any_mouse_down(move |_, _, cx| {
                    if stop_capture_down.get() {
                        cx.stop_propagation();
                    }
                })
                .on_mouse_down(MouseButton::Left, move |_, window, _| {
                    log_parent_down
                        .borrow_mut()
                        .push(format!("parent down prevented={}", window.default_prevented()));
                })
                .child(
                    canvas(
                        |_, _, _| {},
                        move |_, (), window, _| {
                            let log_move = log_move.clone();
                            window.on_mouse_event(move |_: &MouseMoveEvent, phase, _, cx| {
                                if phase == DispatchPhase::Capture {
                                    log_move.borrow_mut().push("parent capture move".into());
                                    cx.stop_propagation();
                                }
                            });
                        },
                    )
                    .size_0(),
                )
                .child(
                    div()
                        .w(px(60.))
                        .h(px(30.))
                        .bg(color(1))
                        .hover(|s| s.bg(color(2)))
                        .track_focus(&child_focus)
                        .on_mouse_down(MouseButton::Left, move |_, window, _| {
                            log_child_down.borrow_mut().push("child down".into());
                            if prevent_child_down.get() {
                                window.prevent_default();
                            }
                        }),
                )
                .into_any_element()
        }
    });

    // Verify inline variants are used (no boxed closures for focus or hover refresh).
    with_window(&mut app, window, |window, _| {
        let f = &window.rendered_frame;
        assert!(f.mouse_down_listeners.iter().any(|e| matches!(e, MouseDownEntry::Focus { .. })));
        assert!(f.mouse_move_listeners.iter().any(|e| matches!(e, MouseMoveEntry::HoverRefresh { .. })));
    });

    // 1. Parent capture mouse-move stops propagation BEFORE the child's inline HoverRefresh runs.
    mouse_move(&mut app, window, 150., 50.);
    app.frame(window);
    assert!(!app.is_dirty(window));
    events.borrow_mut().clear();
    mouse_move(&mut app, window, 10., 10.);
    assert_eq!(*events.borrow(), ["parent capture move"]);
    assert!(!app.is_dirty(window), "capture stop_propagation blocked child inline HoverRefresh");

    // 2. Child bubble on_mouse_down calls prevent_default(): blocks both inline Focus handlers,
    //    while still allowing parent bubble on_mouse_down to run and observe default_prevented().
    events.borrow_mut().clear();
    mouse_down(&mut app, window, 10., 10.);
    assert_eq!(*events.borrow(), ["child down", "parent down prevented=true"]);
    let focused = with_window(&mut app, window, |window, cx| window.focused(cx).is_some());
    assert!(!focused, "child on_mouse_down prevent_default blocked inline Focus handlers");

    // 3. Without child prevent_default(), child inline Focus focuses the child and prevents parent
    //    inline Focus (parent bubble listener sees default_prevented() == true).
    prevent_child_down.set(false);
    events.borrow_mut().clear();
    mouse_down(&mut app, window, 10., 10.);
    assert_eq!(*events.borrow(), ["child down", "parent down prevented=true"]);
    let focused_child = with_window(&mut app, window, |window, _| {
        child_focus_slot.borrow().as_ref().unwrap().is_focused(window)
    });
    assert!(focused_child, "child inline Focus should win over parent inline Focus");

    // 4. Capture stop_propagation() blocks inline Focus completely.
    with_window(&mut app, window, |window, _| window.blur());
    app.frame(window);
    stop_capture_down.set(true);
    events.borrow_mut().clear();
    mouse_down(&mut app, window, 10., 10.);
    assert!(events.borrow().is_empty());
    let focused_after_stop = with_window(&mut app, window, |window, cx| window.focused(cx).is_some());
    assert!(!focused_after_stop, "capture stop_propagation blocked inline Focus");

    // 5. Fallback `other_mouse_listeners` bucket delivers MouseExitEvent and FileDropEvent::Exited
    //    without crosstalk between the two event types.
    let other_log = log();
    let (mut app2, window2) = open(size(px(100.), px(100.)), 1.0, {
        let other_log = other_log.clone();
        move |_, _| {
            let log_exit = other_log.clone();
            let log_drop = other_log.clone();
            canvas(
                |_, _, _| {},
                move |_, (), window, _| {
                    let log_exit = log_exit.clone();
                    window.on_mouse_event(move |_: &MouseExitEvent, phase, _, _| {
                        if phase == DispatchPhase::Bubble {
                            log_exit.borrow_mut().push("exit".into());
                        }
                    });
                    let log_drop = log_drop.clone();
                    window.on_mouse_event(move |_: &crate::FileDropEvent, phase, _, _| {
                        if phase == DispatchPhase::Bubble {
                            log_drop.borrow_mut().push("drop-exit".into());
                        }
                    });
                },
            )
            .size_full()
            .into_any_element()
        }
    });
    app2.simulate_input(
        window2,
        PlatformInput::MouseExited(MouseExitEvent {
            position: Point::default(),
            pressed_button: None,
            modifiers: Modifiers::default(),
        }),
    );
    app2.simulate_input(window2, PlatformInput::FileDrop(crate::FileDropEvent::Exited));
    assert_eq!(*other_log.borrow(), ["exit", "drop-exit"]);
}

#[test]
fn clipped_offscreen_hitboxes_are_dropped_while_edge_touching_remains_hittable() {
    let clicks = log();
    let (mut app, window) = open(size(px(100.), px(100.)), 1.0, {
        let clicks = clicks.clone();
        move |_, _| {
            let off = clicks.clone();
            let edge = clicks.clone();
            div()
                .id("clip-box")
                .w(px(50.))
                .h(px(50.))
                .overflow_hidden()
                .child(
                    // Strictly outside the 50x50 clip box: clipped size is negative, dropped from Frame::hitboxes.
                    div()
                        .id("offscreen")
                        .absolute()
                        .left(px(60.))
                        .top(px(10.))
                        .w(px(20.))
                        .h(px(20.))
                        .on_click(move |_, _, _| off.borrow_mut().push("off".into())),
                )
                .child(
                    // Touching x = 50: clipped width is 0, height is 20 -> inclusive `contains` matches x = 50.
                    div()
                        .id("edge")
                        .absolute()
                        .left(px(50.))
                        .top(px(10.))
                        .w(px(20.))
                        .h(px(20.))
                        .on_click(move |_, _, _| edge.borrow_mut().push("edge".into())),
                )
                .into_any_element()
        }
    });

    let stored_hitboxes = with_window(&mut app, window, |w, _| w.last_frame_times().hitboxes);
    // Only the edge-touching hitbox survives into Frame::hitboxes (clip-box has no interactivity;
    // offscreen is strictly disjoint from the clip mask).
    assert_eq!(stored_hitboxes, 1);

    mouse_down(&mut app, window, 65., 15.);
    mouse_up(&mut app, window, 65., 15.);
    assert!(clicks.borrow().is_empty(), "strictly clipped hitbox must not fire");

    mouse_down(&mut app, window, 50., 15.);
    mouse_up(&mut app, window, 50., 15.);
    assert_eq!(*clicks.borrow(), ["edge"], "edge-touching zero-width intersection remains clickable");
}

#[test]
fn signed_zero_hitbox_intersection_remains_hittable() {
    let mask_bounds = Bounds::new(point(px(-0.0), px(0.0)), size(px(-0.0), px(20.0)));
    let box_bounds = Bounds::new(point(px(0.0), px(0.0)), size(px(10.0), px(20.0)));
    let clipped = box_bounds.intersect(&mask_bounds);
    assert!(clipped.size.width.0.is_sign_negative() && clipped.size.width.0 == 0.0);

    let recorded = std::rc::Rc::new(std::cell::Cell::new(None));
    let recorded_in_view = recorded.clone();
    let (mut app, window) = open(size(px(100.), px(100.)), 1.0, move |w, _| {
        let id = w.with_content_mask(Some(crate::ContentMask { bounds: mask_bounds }), |w| {
            w.insert_hitbox(box_bounds, crate::HitboxBehavior::Normal).id
        });
        recorded_in_view.set(Some(id));
        div().into_any_element()
    });
    let expected = recorded.get().unwrap();
    let hits = with_window(&mut app, window, |w, _| w.rendered_frame.hit_test(point(px(0.0), px(5.0))).ids);
    assert!(hits.contains(&expected), "signed-zero width intersection must remain hittable at x = 0");
}
