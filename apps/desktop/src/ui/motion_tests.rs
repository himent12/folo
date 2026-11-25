//! The app's opacity animations (`dot`, `appear`, `skeleton`) run on `with_opacity_animation`, which
//! re-fades them without rebuilding the UI. They must look exactly as their `with_animation` versions did:
//! every frame the new way shows equals the old way's frame at that moment, byte for byte.

use std::time::Duration;

use folo_ui::{
    Animation, AnimationExt, AnyElement, AnyWindowHandle, App, AppContext, Bounds, Context, HeadlessApp,
    HeadlessOptions, IntoElement, ParentElement, Render, SharedString, Styled, TextBackend, Window,
    WindowBounds, WindowOptions, div, point, px, size,
};

use crate::theme::{Mode, Theme, radius};

/// The `with_animation` versions the app used before (`status.rs`, `motion.rs`, `surface.rs`).
fn old_dot(color: folo_ui::Hsla, size: f32, id: &str) -> AnyElement {
    div()
        .size(px(size))
        .flex_none()
        .rounded_full()
        .bg(color)
        .with_animation(
            folo_ui::ElementId::Name(format!("breathe-{id}").into()),
            Animation::new(Duration::from_millis(2400)).repeat(),
            |d, p| d.opacity(0.7 + 0.3 * (p * std::f32::consts::TAU).cos()),
        )
        .into_any_element()
}

fn old_appear<E: IntoElement + Styled + 'static>(el: E, id: &str) -> AnyElement {
    el.with_animation(
        SharedString::from(id.to_string()),
        Animation::new(Duration::from_millis(160)).with_easing(folo_ui::ease_out_quint()),
        |d, p| d.opacity(p),
    )
    .into_any_element()
}

fn old_skeleton(t: &Theme, id: &str) -> AnyElement {
    div()
        .size_full()
        .rounded(px(radius::SM))
        .bg(t.bg_active)
        .with_animation(
            folo_ui::ElementId::Name(format!("skel-{id}").into()),
            Animation::new(Duration::from_millis(1400)).repeat(),
            move |d, p| {
                let w = (p * std::f32::consts::TAU).sin() * 0.5 + 0.5;
                d.opacity(0.55 + 0.45 * w)
            },
        )
        .into_any_element()
}

struct View {
    old: bool,
}

impl Render for View {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = crate::theme::theme(cx);
        let panel =
            div().p(px(12.)).bg(t.bg_raised).border_1().border_color(t.line_subtle).child("Appearing");
        let (dot, panel, skeleton) = if self.old {
            (old_dot(t.accent_solid, 9., "run"), old_appear(panel, "in"), old_skeleton(&t, "s"))
        } else {
            (
                super::dot(t.accent_solid, 9., true, "run"),
                super::appear(panel, "in"),
                super::skeleton(&t, "s"),
            )
        };
        div()
            .size_full()
            .p(px(10.))
            .flex()
            .flex_col()
            .gap(px(8.))
            .bg(t.bg_app)
            .text_color(t.text_primary)
            .child(dot)
            .child(panel)
            .child(div().w(px(80.)).h(px(20.)).child(skeleton))
    }
}

fn open(old: bool) -> (HeadlessApp, AnyWindowHandle) {
    let window_size = size(px(200.), px(120.));
    let mut app =
        HeadlessApp::new(HeadlessOptions { size: window_size, scale_factor: 1.5, text: TextBackend::Native });
    let window = app.update(|cx: &mut App| {
        cx.text_system()
            .add_fonts(crate::FONTS.iter().map(|f| std::borrow::Cow::Borrowed(*f)).collect())
            .expect("bundled fonts load");
        cx.set_global(Theme::new(Mode::Dark));
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::new(point(px(0.), px(0.)), window_size))),
            ..Default::default()
        };
        cx.open_window(options, |_, cx| cx.new(|_| View { old })).expect("open window").into()
    });
    (app, window)
}

#[test]
fn the_app_s_opacity_animations_look_as_they_did() {
    let _reduce_motion = super::motion::override_reduce_motion_for_test(false);
    let (mut old, old_window) = open(true);
    let (mut new, new_window) = open(false);
    let mut shown = old.frame(old_window).clone();
    assert!(new.frame(new_window) == &shown, "the first frame");
    let frame_number = |app: &mut HeadlessApp| {
        app.update(|cx| cx.update_window(new_window, |_, window, _| window.last_frame_times().frame)).unwrap()
    };
    let mut redraws = 0;
    for step in 1..=120 {
        old.advance_clock(Duration::from_millis(23));
        new.advance_clock(Duration::from_millis(23));
        let expected = old.frame(old_window).clone();
        assert!(!new.is_dirty(new_window), "the animations do not rebuild the UI");
        let before = frame_number(&mut new);
        let frame = new.frame(new_window).clone();
        if frame_number(&mut new) != before {
            // A tick redrew: the picture of this moment.
            shown = expected;
            redraws += 1;
        }
        assert!(frame == shown, "the frame at {} ms", step * 23);
    }
    assert!(redraws > 60, "{redraws} redraws");
}
