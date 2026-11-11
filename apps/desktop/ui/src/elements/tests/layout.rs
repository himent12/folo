//! Views, canvas, anchored, deferred, svg and animation.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::Duration,
};

use super::{draw, open, quads, sb, solid, sprites};
use crate::{
    Animation, AnimationExt, InteractiveElement, IntoElement, ParentElement, SpriteKind, Styled, anchored,
    canvas, deferred, div, fill, hsla, point, px, rgb, svg,
};

#[test]
fn a_view_renders_its_tree_into_the_scene() {
    let (mut app, window) = open(|_, _| {
        div()
            .size_full()
            .bg(rgb(0x102030))
            .child(div().w(px(100.)).h(px(50.)).bg(rgb(0xff0000)))
            .into_any_element()
    });
    let scene = draw(&mut app, window);
    let quads = quads(&scene);
    assert_eq!(quads.len(), 2);
    assert_eq!(quads[0].bounds, sb(0., 0., 400., 300.));
    assert_eq!(quads[1].bounds, sb(0., 0., 100., 50.));
    assert_eq!(solid(&quads[1]), rgb(0xff0000).into());
    assert!(!app.is_dirty(window), "nothing changed: the window is idle after a frame");
}

#[test]
fn a_notify_redraws_and_an_idle_window_does_not() {
    let renders = Rc::new(Cell::new(0));
    let renders2 = renders.clone();
    let (mut app, window) = open(move |_, _| {
        renders2.set(renders2.get() + 1);
        div().size_full().into_any_element()
    });
    draw(&mut app, window);
    draw(&mut app, window);
    assert_eq!(renders.get(), 1, "a clean window does not render again");
    app.update(|cx| {
        window.update(cx, |root, _, cx| cx.notify(root.entity_id())).unwrap();
    });
    assert!(app.is_dirty(window));
    draw(&mut app, window);
    assert_eq!(renders.get(), 2);
}

#[test]
fn canvas_paints_between_its_background_and_border_and_gets_its_bounds() {
    let seen = Rc::new(RefCell::new(None));
    let seen2 = seen.clone();
    let (mut app, window) = open(move |_, _| {
        let seen = seen2.clone();
        div()
            .p(px(10.))
            .child(
                canvas(
                    move |bounds, _, _| {
                        *seen.borrow_mut() = Some(bounds);
                        bounds.size.width
                    },
                    |bounds, width, window, _| {
                        window.paint_quad(fill(
                            crate::Bounds::new(bounds.origin, crate::size(width / 2., px(5.))),
                            rgb(0x00ff00),
                        ))
                    },
                )
                .w(px(40.))
                .h(px(20.))
                .bg(rgb(0x0000ff))
                .border_1()
                .border_color(rgb(0xffffff)),
            )
            .into_any_element()
    });
    let scene = draw(&mut app, window);
    assert_eq!(
        *seen.borrow(),
        Some(crate::Bounds::new(point(px(10.), px(10.)), crate::size(px(40.), px(20.))))
    );
    let quads = quads(&scene);
    let colors: Vec<_> = quads.iter().map(solid).collect();
    // Background, then the canvas's own quad, then the border (four side-clipped quads, GPUI's).
    assert_eq!(colors[0], rgb(0x0000ff).into());
    assert_eq!(colors[1], rgb(0x00ff00).into());
    assert_eq!(quads[1].bounds, sb(10., 10., 20., 5.));
    assert_eq!(quads.len(), 6);
    assert!(quads[2..].iter().all(|q| q.border_color == rgb(0xffffff).into()));
}

#[test]
fn anchored_snaps_inside_the_window_with_a_margin() {
    let (mut app, window) = open(|_, _| {
        div()
            .size_full()
            .child(
                anchored()
                    .position(point(px(380.), px(290.)))
                    .snap_to_window_with_margin(px(8.))
                    .child(div().w(px(100.)).h(px(40.)).bg(rgb(0xff0000))),
            )
            .into_any_element()
    });
    let scene = draw(&mut app, window);
    let quads = quads(&scene);
    // 380 + 100 overflows 400: moved left by the overflow plus the margin; same vertically.
    assert_eq!(quads[0].bounds, sb(292., 252., 100., 40.));
}

#[test]
fn anchored_switches_corner_when_it_does_not_fit() {
    let fit = crate::elements::anchored::fit;
    let limits = crate::Bounds::new(point(px(0.), px(0.)), crate::size(px(400.), px(300.)));
    let size = crate::size(px(100.), px(40.));
    let placed = fit(
        point(px(380.), px(290.)),
        point(px(0.), px(0.)),
        crate::Corner::TopLeft,
        crate::AnchoredFitMode::SwitchAnchor,
        size,
        limits,
    );
    assert_eq!(
        placed,
        crate::Bounds::new(point(px(280.), px(250.)), size),
        "flipped to the bottom-right corner"
    );
}

#[test]
fn deferred_children_paint_after_the_tree_in_priority_order() {
    let (mut app, window) = open(|_, _| {
        div()
            .size_full()
            .child(deferred(div().size(px(10.)).bg(rgb(0x000002))).with_priority(2))
            .child(deferred(div().size(px(10.)).bg(rgb(0x000001))).with_priority(1))
            .child(div().size(px(10.)).bg(rgb(0x000003)))
            .into_any_element()
    });
    let scene = draw(&mut app, window);
    let colors: Vec<_> = quads(&scene).iter().map(solid).collect();
    assert_eq!(colors, [rgb(0x000003).into(), rgb(0x000001).into(), rgb(0x000002).into()]);
    // Laid out in place: the second deferred child sits below the first.
    assert_eq!(quads(&scene)[1].bounds, sb(0., 10., 10., 10.));
}

#[test]
fn svg_paints_a_tinted_mask_at_its_bounds() {
    let (mut app, window) = open(|_, _| {
        div()
            .p(px(4.))
            .child(svg().path("icons/square.svg").size(px(16.)).text_color(rgb(0x00ff00)))
            .child(svg().path("icons/square.svg").size(px(16.)))
            .into_any_element()
    });
    let scene = draw(&mut app, window);
    let sprites = sprites(&scene);
    assert_eq!(sprites.len(), 1, "an svg with no text colour paints nothing (GPUI)");
    assert_eq!(sprites[0].bounds, sb(4., 4., 16., 16.));
    assert_eq!(sprites[0].kind, SpriteKind::Svg);
    assert_eq!(sprites[0].color, rgb(0x00ff00).into());
}

#[test]
fn an_animation_advances_with_the_clock_and_stops_asking_for_frames() {
    let progress = Rc::new(RefCell::new(Vec::new()));
    let progress2 = progress.clone();
    let (mut app, window) = open(move |_, _| {
        let progress = progress2.clone();
        div()
            .id("root")
            .size_full()
            .child(div().size(px(10.)).with_animation(
                "fade",
                Animation::new(Duration::from_millis(100)),
                move |el, p| {
                    progress.borrow_mut().push(p);
                    el.bg(hsla(0., 0., 0., p))
                },
            ))
            .into_any_element()
    });
    draw(&mut app, window);
    assert!(app.is_dirty(window), "a running animation asks for the next frame");
    app.advance_clock(Duration::from_millis(50));
    draw(&mut app, window);
    app.advance_clock(Duration::from_millis(60));
    draw(&mut app, window);
    assert_eq!(*progress.borrow(), [0.0, 0.5, 1.0]);
    assert!(!app.is_dirty(window), "a finished one-shot animation stops asking");
    let scene = draw(&mut app, window);
    assert_eq!(solid(&quads(&scene)[0]).a, 1.0);
}

#[test]
fn a_repeating_animation_wraps() {
    let animation = Animation::new(Duration::from_millis(100)).repeat();
    let (p, done) = crate::elements::animation::progress(&animation, Duration::from_millis(250));
    assert!((p - 0.5).abs() < 1e-6 && !done);
    let eased = Animation::new(Duration::from_millis(100)).with_easing(|t| t * t);
    assert_eq!(crate::elements::animation::progress(&eased, Duration::from_millis(50)).0, 0.25);
}

#[test]
fn a_view_that_notifies_in_render_does_not_redraw_forever() {
    let (mut app, window) = open(|_, cx| {
        cx.notify();
        div().size_full().into_any_element()
    });
    draw(&mut app, window);
    app.run_until_idle();
    assert!(!app.is_dirty(window), "GPUI ignores a view's notify while its window draws");
}
