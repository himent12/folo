//! The overlay scroll thumb shows for 800 ms after a scroll, timed on the executor's clock: a headless
//! `--shot` runs on a fake clock, so the thumb's state at the capture depends only on the fake time the
//! shot advanced to, never on how long the process took (the wall clock made `run-done+bottom` and
//! `run-done+sample-focus` flip between two pictures from run to run).

use std::time::Duration;

use folo_ui::{
    AnyWindowHandle, App, AppContext, Bounds, Context, Entity, Framebuffer, HeadlessApp, HeadlessOptions,
    InteractiveElement, IntoElement, ParentElement, Render, ScrollHandle, StatefulInteractiveElement, Styled,
    TextBackend, Window, WindowBounds, WindowOptions, div, point, px, size,
};

use crate::theme::{Mode, Theme};

struct View {
    scroll: ScrollHandle,
    /// The height of the area that clips the list (120: the window; 20: above the list, which it hides).
    clip: f32,
}

impl Render for View {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = crate::theme::theme(cx);
        let paint = self.scroll.clone();
        // A 60 px tall list of 300 px of rows at (40, 40), its thumb painted over its right edge, in an
        // area `clip` px tall that clips it.
        div().size_full().bg(t.bg_app).child(
            div().w_full().h(px(self.clip)).overflow_hidden().child(
                div()
                    .absolute()
                    .left(px(40.))
                    .top(px(40.))
                    .w(px(100.))
                    .h(px(60.))
                    .child(
                        div()
                            .id("list")
                            .size_full()
                            .overflow_y_scroll()
                            .track_scroll(&self.scroll)
                            .children((0..10).map(|_| div().h(px(30.)).flex_none())),
                    )
                    .child(
                        folo_ui::canvas(
                            |_, _, _| (),
                            move |_, _, window, cx| super::paint_thumb(&t, &paint, false, window, cx),
                        )
                        .absolute()
                        .size_full(),
                    ),
            ),
        )
    }
}

fn open(clip: f32) -> (HeadlessApp, AnyWindowHandle, ScrollHandle, Entity<View>) {
    let window_size = size(px(200.), px(120.));
    let mut app =
        HeadlessApp::new(HeadlessOptions { size: window_size, scale_factor: 1.0, text: TextBackend::Native });
    let scroll = ScrollHandle::new();
    let handle = scroll.clone();
    let mut view = None;
    let window = app.update(|cx: &mut App| {
        cx.set_global(Theme::new(Mode::Dark));
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::new(point(px(0.), px(0.)), window_size))),
            ..Default::default()
        };
        cx.open_window(options, |_, cx| {
            let entity = cx.new(|_| View { scroll: handle, clip });
            view = Some(entity.clone());
            entity
        })
        .expect("open window")
        .into()
    });
    (app, window, scroll, view.expect("the view"))
}

/// Whether anything is drawn in the thumb's column (x 134..138) over the list.
fn thumb_shown(frame: &Framebuffer) -> bool {
    let at = |x: u32, y: u32| frame.pixels[(y * frame.width + x) as usize];
    let bg = at(136, 20);
    (40..100).any(|y| at(136, y) != bg)
}

#[test]
fn the_thumb_hides_on_the_executor_s_clock() {
    let (mut app, window, scroll, _) = open(120.);
    assert!(!thumb_shown(app.frame(window)), "at rest, no thumb");
    scroll.set_offset(point(px(0.), px(-120.)));
    app.update(|cx| cx.update_window(window, |_, window, _| window.refresh())).unwrap();
    assert!(thumb_shown(app.frame(window)), "shown after a scroll");
    // 700 ms on the fake clock: still shown.
    for _ in 0..14 {
        app.advance_clock(Duration::from_millis(50));
        app.frame(window);
    }
    assert!(thumb_shown(app.frame(window)), "shown 700 ms after the scroll");
    // Past 800 ms of fake time, and well under 800 ms of real time since the scroll: hidden, and the
    // window stops asking for frames.
    for _ in 0..4 {
        app.advance_clock(Duration::from_millis(50));
        app.frame(window);
    }
    assert!(!thumb_shown(app.frame(window)), "hidden 900 ms after the scroll");
    assert!(!app.wants_frame(window), "idle once hidden");
}

#[test]
fn a_thumb_out_of_view_asks_for_no_frames() {
    let (mut app, window, scroll, view) = open(20.);
    app.frame(window);
    scroll.set_offset(point(px(0.), px(-120.)));
    app.update(|cx| cx.update_window(window, |_, window, _| window.refresh())).unwrap();
    app.frame(window);
    // Clipped away: no frames to keep painting it (it used to ask for 800 ms of them).
    assert!(!app.wants_frame(window), "idle with the thumb out of view");
    // Brought into view while the scroll is recent, it shows and keeps the window drawing until it hides.
    app.advance_clock(Duration::from_millis(300));
    app.update(|cx| view.update(cx, |v, cx| (v.clip = 120., cx.notify())));
    assert!(thumb_shown(app.frame(window)), "shown once in view, 300 ms after the scroll");
    assert!(app.wants_frame(window), "drawing while shown");
    for _ in 0..12 {
        app.advance_clock(Duration::from_millis(50));
        app.frame(window);
    }
    assert!(!thumb_shown(app.frame(window)), "hidden 900 ms after the scroll");
    assert!(!app.wants_frame(window), "idle once hidden");
}
