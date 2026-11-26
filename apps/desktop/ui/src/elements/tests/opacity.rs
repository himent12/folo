//! `with_opacity_animation` paints exactly what `with_animation` setting `.opacity(..)` paints, at every
//! phase, while drawing no frame of the UI after the first.

use std::{rc::Rc, time::Duration};

use super::open_sized;
use crate::{
    Animation, AnimationExt, AnyElement, AnyWindowHandle, AppContext, BoxShadow, Div, HeadlessApp,
    IntoElement, ParentElement, PathBuilder, Styled, canvas, div, ease_out_quint, hsla, point, px, size, svg,
};

/// Which API animates the opacity.
#[derive(Clone, Copy, PartialEq)]
enum Way {
    /// `with_animation(.., |el, p| el.opacity(curve(p)))`: the whole window re-renders each frame.
    Rerender,
    /// `with_opacity_animation(.., curve)`: the window re-fades a group.
    Paint,
}

type Curve = Rc<dyn Fn(f32) -> f32>;

fn animated(way: Way, id: &'static str, animation: Animation, curve: Curve, el: Div) -> AnyElement {
    match way {
        Way::Rerender => {
            el.with_animation(id, animation, move |el, p| el.opacity(curve(p))).into_any_element()
        }
        Way::Paint => el.with_opacity_animation(id, animation, move |p| curve(p)).into_any_element(),
    }
}

/// The app's three opacity animations (the status dot's breathing, an overlay's appear, a skeleton's
/// shimmer), nested, over quads, borders, shadows, text, an icon, a path and an element opacity.
fn tree(way: Way) -> AnyElement {
    let breathe: Curve = Rc::new(|p| 0.7 + 0.3 * (p * std::f32::consts::TAU).cos());
    let appear: Curve = Rc::new(|p| p);
    let shimmer: Curve = Rc::new(|p| 0.55 + 0.45 * ((p * std::f32::consts::TAU).sin() * 0.5 + 0.5));
    let repeat = |ms| Animation::new(Duration::from_millis(ms)).repeat();
    let skeleton = animated(
        way,
        "skeleton",
        repeat(1400),
        shimmer,
        div().size(px(30.)).rounded(px(4.)).bg(hsla(0.6, 0.2, 0.5, 1.)),
    );
    let panel = div()
        .flex()
        .gap(px(6.))
        .p(px(8.))
        .bg(hsla(0.1, 0.3, 0.9, 1.))
        .border_1()
        .border_color(hsla(0.0, 0.5, 0.4, 0.9))
        .rounded(px(6.))
        .shadow(vec![BoxShadow {
            color: hsla(0., 0., 0., 0.4),
            offset: point(px(0.), px(2.)),
            blur_radius: px(6.),
            spread_radius: px(0.),
        }])
        .text_color(hsla(0.7, 0.4, 0.2, 1.))
        .child("Fade in")
        .child(
            div()
                .size(px(14.))
                .opacity(0.6)
                .bg(hsla(0.3, 0.8, 0.4, 1.))
                .child(div().size(px(7.)).opacity(0.35).bg(hsla(0.9, 0.8, 0.4, 0.8))),
        )
        .child(svg().path("icons/square.svg").size(px(12.)).text_color(hsla(0.5, 0.9, 0.3, 1.)))
        .child(
            canvas(
                |_, _, _| (),
                |bounds, _, window, _| {
                    let mut path = PathBuilder::stroke(px(1.5));
                    path.move_to(bounds.origin);
                    path.line_to(bounds.origin + point(px(20.), px(12.)));
                    path.line_to(bounds.origin + point(px(30.), px(2.)));
                    if let Ok(path) = path.build() {
                        window.paint_path(path, hsla(0.0, 0.8, 0.5, 1.));
                    }
                },
            )
            .size(px(30.)),
        )
        .child(skeleton);
    let ease = ease_out_quint();
    div()
        .size_full()
        .flex()
        .flex_col()
        .gap(px(10.))
        .p(px(10.))
        .bg(hsla(0., 0., 1., 1.))
        .child(animated(
            way,
            "dot",
            repeat(2400),
            breathe.clone(),
            div().size(px(9.)).rounded_full().bg(hsla(0.55, 0.9, 0.5, 1.)),
        ))
        .child(div().opacity(0.8).child(animated(
            way,
            "dot-in-opacity",
            repeat(2400),
            breathe,
            div().size(px(12.)).rounded_full().bg(hsla(0.35, 0.9, 0.5, 1.)),
        )))
        .child(animated(
            way,
            "appear",
            Animation::new(Duration::from_millis(160)).with_easing(ease),
            appear,
            panel,
        ))
        .into_any_element()
}

fn open(way: Way) -> (HeadlessApp, AnyWindowHandle) {
    open_sized(size(px(220.), px(130.)), 1.5, move |_, _| tree(way))
}

fn redraw(app: &mut HeadlessApp, window: AnyWindowHandle) {
    app.update(|cx| cx.update_window(window, |_, window, _| window.refresh())).expect("window");
}

#[test]
fn an_opacity_animation_paints_what_a_rerendered_opacity_paints() {
    let (mut old, old_window) = open(Way::Rerender);
    let (mut new, new_window) = open(Way::Paint);
    assert_eq!(old.frame(old_window), new.frame(new_window), "the first frame");
    // Steps of 13 ms: the appear fade's every frame, then the repeating ones through more than a loop. A
    // tick redraws only when a factor moved by 1/255; every frame shown is the old way's frame at the time
    // of the last redraw, byte for byte.
    let frame_number = |app: &HeadlessApp, window: AnyWindowHandle| {
        app.read(|cx| match cx.windows.get(&window.id) {
            Some(Some(window)) => window.frame_times.frame,
            _ => 0,
        })
    };
    let mut shown = old.frame(old_window).clone();
    let mut redraws = 0;
    for step in 1..=200 {
        old.advance_clock(Duration::from_millis(13));
        new.advance_clock(Duration::from_millis(13));
        assert!(new.wants_frame(new_window), "a repeating paint animation keeps asking for frames");
        // Not dirty: the frame below re-fades the groups and builds, lays out and paints nothing.
        assert!(!new.is_dirty(new_window), "a paint animation does not make the window dirty");
        let expected = old.frame(old_window).clone();
        let before = frame_number(&new, new_window);
        let frame = new.frame(new_window).clone();
        if frame_number(&new, new_window) != before {
            shown = expected.clone();
            redraws += 1;
        }
        assert!(frame == shown, "a tick at {} ms", step * 13);
        if step % 25 == 0 {
            // A full redraw continues the animations from their start times.
            redraw(&mut new, new_window);
            assert!(new.frame(new_window) == &expected, "a full redraw at {} ms", step * 13);
            shown = expected;
        }
    }
    assert!(redraws > 150, "most 13 ms ticks move a factor by 1/255: {redraws}");
}

#[test]
fn a_finished_opacity_animation_lands_on_its_last_value_and_stops() {
    let (mut app, window) = open_sized(size(px(60.), px(40.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().size(px(20.)).bg(hsla(0., 0., 0., 1.)).with_opacity_animation(
                "fade",
                Animation::new(Duration::from_millis(100)),
                |p| p * 0.5,
            ))
            .into_any_element()
    });
    app.frame(window);
    assert!(app.wants_frame(window));
    // Past the end in one step (no 1/255 rule can hold back the last value).
    app.advance_clock(Duration::from_millis(250));
    app.frame(window);
    assert!(!app.wants_frame(window), "a finished one-shot animation stops asking");
    let quad = app.scene(window).primitives.iter().find_map(|p| match p {
        crate::Primitive::Quad(q) => Some(q.background.solid.a),
        _ => None,
    });
    assert_eq!(quad, Some(0.5));
}

#[test]
fn zero_duration_and_exact_completion_animations_stay_finite_and_stop() {
    // 1. Zero-duration one-shot finishes immediately on the first frame without NaN or infinite ticks.
    let (mut app, window) = open_sized(size(px(40.), px(40.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().size(px(20.)).bg(hsla(0., 0., 0., 1.)).with_opacity_animation(
                "zero",
                Animation::new(Duration::ZERO),
                |p| p * 0.4,
            ))
            .into_any_element()
    });
    app.frame(window);
    assert!(!app.wants_frame(window), "zero-duration one-shot must not keep requesting frames");
    let alpha = app.scene(window).primitives.iter().find_map(|p| match p {
        crate::Primitive::Quad(q) => Some(q.background.solid.a),
        _ => None,
    });
    assert_eq!(alpha, Some(0.4));

    // 2. Exact duration completion (`elapsed == duration`) finishes on that tick without a trailing frame.
    let (mut app, window) = open_sized(size(px(40.), px(40.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().size(px(20.)).bg(hsla(0., 0., 0., 1.)).with_opacity_animation(
                "exact",
                Animation::new(Duration::from_millis(100)),
                |p| p * 0.6,
            ))
            .into_any_element()
    });
    app.frame(window);
    assert!(app.wants_frame(window));
    app.advance_clock(Duration::from_millis(100));
    app.frame(window);
    assert!(!app.wants_frame(window), "exact completion at duration stops requesting frames");
    let alpha = app.scene(window).primitives.iter().find_map(|p| match p {
        crate::Primitive::Quad(q) => Some(q.background.solid.a),
        _ => None,
    });
    assert_eq!(alpha, Some(0.6));

    // 3. Non-finite easing/animator output is sanitized to finite opacity.
    let (mut app, window) = open_sized(size(px(40.), px(40.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().size(px(20.)).bg(hsla(0., 0., 0., 1.)).with_opacity_animation(
                "nan",
                Animation::new(Duration::from_millis(50)),
                |_| f32::NAN,
            ))
            .into_any_element()
    });
    app.frame(window);
    app.advance_clock(Duration::from_millis(50));
    app.frame(window);
    assert!(!app.wants_frame(window));
    let alpha = app.scene(window).primitives.iter().find_map(|p| match p {
        crate::Primitive::Quad(q) => Some(q.background.solid.a),
        _ => None,
    });
    assert_eq!(alpha, Some(1.0));
}
