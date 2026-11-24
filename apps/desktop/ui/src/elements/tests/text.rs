//! The text element with `FixedTextSystem` (every glyph advances 0.6 em; whitespace paints nothing).

use super::{draw, open, sprites};
use crate::{IntoElement, MonoSprite, ParentElement, Styled, StyledText, TextRun, div, font, px, rgb};

/// Glyph sprites grouped by line (their top edge), in order.
fn lines(sprites: &[MonoSprite]) -> Vec<Vec<&MonoSprite>> {
    let mut lines: Vec<Vec<&MonoSprite>> = Vec::new();
    for sprite in sprites {
        match lines.last_mut() {
            Some(line) if line[0].bounds.origin.y == sprite.bounds.origin.y => line.push(sprite),
            _ => lines.push(vec![sprite]),
        }
    }
    lines
}

fn text_box(width: f32) -> crate::Div {
    div().w(px(width)).text_size(px(10.)).line_height(px(20.)).text_color(rgb(0xffffff))
}

#[test]
fn a_string_lays_out_on_one_line_with_fixed_advances() {
    let (mut app, window) = open(|_, _| text_box(200.).child("hello world").into_any_element());
    let scene = draw(&mut app, window);
    let sprites = sprites(&scene);
    assert_eq!(sprites.len(), 10, "eleven characters, the space paints nothing");
    let lines = lines(&sprites);
    assert_eq!(lines.len(), 1);
    let x = |i: usize| lines[0][i].bounds.origin.x.0;
    assert_eq!(x(1) - x(0), 6.0, "0.6 em at 10 px");
    assert_eq!(x(5) - x(0), 36.0, "`w` follows `hello ` (six advances)");
    assert!(sprites.iter().all(|s| s.color == rgb(0xffffff).into()));
}

#[test]
fn text_wraps_at_the_available_width() {
    let (mut app, window) = open(|_, _| text_box(40.).child("hello world").into_any_element());
    let scene = draw(&mut app, window);
    let sprites = sprites(&scene);
    let lines = lines(&sprites);
    assert_eq!(lines.iter().map(Vec::len).collect::<Vec<_>>(), [5, 5]);
    assert_eq!(lines[1][0].bounds.origin.y.0 - lines[0][0].bounds.origin.y.0, 20.0, "one line height apart");
    assert_eq!(
        lines[1][0].bounds.origin.x, lines[0][0].bounds.origin.x,
        "the second line starts at the left"
    );
}

#[test]
fn whitespace_nowrap_keeps_one_line() {
    let (mut app, window) =
        open(|_, _| text_box(40.).whitespace_nowrap().child("hello world").into_any_element());
    let sprites = sprites(&draw(&mut app, window));
    assert_eq!(lines(&sprites).len(), 1);
    assert_eq!(sprites.len(), 10);
}

#[test]
fn truncate_cuts_the_line_and_adds_an_ellipsis() {
    let (mut app, window) = open(|_, _| text_box(40.).truncate().child("hello world").into_any_element());
    let sprites = sprites(&draw(&mut app, window));
    let lines = lines(&sprites);
    assert_eq!(lines.len(), 1);
    // GPUI's rule: keep characters while they and the suffix fit, so `hello` + `…` (six glyphs).
    assert_eq!(lines[0].len(), 6);
    assert_eq!(lines[0][5].bounds.origin.x.0 - lines[0][0].bounds.origin.x.0, 30.0);
}

#[test]
fn line_clamp_limits_the_lines() {
    let (mut app, window) =
        open(|_, _| text_box(40.).line_clamp(1).child("hello world again").into_any_element());
    let sprites = sprites(&draw(&mut app, window));
    assert_eq!(lines(&sprites).len(), 1);
}

#[test]
fn styled_text_runs_colour_their_ranges() {
    let (mut app, window) = open(|_, _| {
        let run = |len, color| TextRun {
            len,
            font: font("Fixed"),
            color: rgb(color).into(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        text_box(200.)
            .child(StyledText::new("abcd").with_runs(vec![run(2, 0xff0000), run(2, 0x00ff00)]))
            .into_any_element()
    });
    let sprites = sprites(&draw(&mut app, window));
    let colors: Vec<_> = sprites.iter().map(|s| s.color).collect();
    assert_eq!(
        colors,
        [rgb(0xff0000).into(), rgb(0xff0000).into(), rgb(0x00ff00).into(), rgb(0x00ff00).into()]
    );
}

#[test]
fn text_height_is_lines_times_line_height() {
    // A parent with no height grows to the text: two wrapped lines of 20 px under a 40 px wide box.
    let (mut app, window) =
        open(|_, _| div().child(text_box(40.).bg(rgb(0x123456)).child("hello world")).into_any_element());
    let scene = draw(&mut app, window);
    let quad = super::quads(&scene).into_iter().find(|q| super::solid(q) == rgb(0x123456).into()).unwrap();
    assert_eq!(quad.bounds, super::sb(0., 0., 40., 40.));
}

/// The reviewer's case: only the clamp changes between frames. Each frame must show what a new window
/// with that clamp shows (the line layout cache once kept the first clamp's wrapping).
#[test]
fn changing_only_the_line_clamp_rewraps_in_the_next_frame() {
    use crate::AppContext as _;
    use std::{cell::Cell, rc::Rc};
    const TEXT: &str = "alpha beta gamma delta epsilon zeta eta theta";
    fn view(
        clamp: Rc<Cell<usize>>,
    ) -> impl FnMut(&mut crate::Window, &mut crate::Context<super::TestView>) -> crate::AnyElement {
        move |_, _| text_box(65.).line_clamp(clamp.get()).child(TEXT).into_any_element()
    }
    let fresh = |clamp: usize| {
        let (mut app, window) = open(view(Rc::new(Cell::new(clamp))));
        let scene = draw(&mut app, window);
        (lines(&sprites(&scene)).len(), app.frame(window).pixels.clone())
    };
    for (from, to) in [(2, 4), (4, 2)] {
        let clamp = Rc::new(Cell::new(from));
        let (mut app, window) = open(view(clamp.clone()));
        draw(&mut app, window);
        clamp.set(to);
        app.update(|cx| cx.update_window(window, |_, window, _| window.refresh())).unwrap();
        let scene = draw(&mut app, window);
        let (want_lines, want_pixels) = fresh(to);
        assert_eq!(want_lines, to, "clamp {to} limits the text");
        assert_eq!(lines(&sprites(&scene)).len(), want_lines, "clamp {from} -> {to}: lines");
        assert!(app.frame(window).pixels == want_pixels, "clamp {from} -> {to}: pixels");
    }
}

/// Two elements with the same text, font and width but different clamps, in one frame: each is wrapped
/// to its own clamp, whichever comes first.
#[test]
fn two_clamps_of_one_text_in_one_frame_wrap_independently() {
    const TEXT: &str = "alpha beta gamma delta epsilon zeta eta theta";
    for (first, second) in [(2, 4), (4, 2)] {
        let (mut app, window) = open(move |_, _| {
            div()
                .flex()
                .flex_row()
                .child(text_box(65.).line_clamp(first).child(TEXT))
                .child(text_box(65.).line_clamp(second).child(TEXT))
                .into_any_element()
        });
        let sprites = sprites(&draw(&mut app, window));
        let column = |left: bool| {
            let sprites: Vec<_> =
                sprites.iter().filter(|s| (s.bounds.origin.x.0 < 65.) == left).cloned().collect();
            lines(&sprites).len()
        };
        assert_eq!((column(true), column(false)), (first, second));
    }
}

#[test]
fn styled_text_paints_background_only_when_runs_have_one() {
    let make_run = |len, bg| TextRun {
        len,
        font: font("Fixed"),
        color: rgb(0xffffff).into(),
        background_color: bg,
        underline: None,
        strikethrough: None,
    };
    let (mut app, window) = open(move |_, _| {
        text_box(200.)
            .child(StyledText::new("abcd").with_runs(vec![make_run(2, None), make_run(2, None)]))
            .into_any_element()
    });
    assert!(super::quads(&draw(&mut app, window)).is_empty());

    let highlight = rgb(0x336699).into();
    let (mut app, window) = open(move |_, _| {
        text_box(200.)
            .child(StyledText::new("abcd").with_runs(vec![
                make_run(1, None),
                make_run(2, Some(highlight)),
                make_run(1, None),
            ]))
            .into_any_element()
    });
    let quads = super::quads(&draw(&mut app, window));
    assert_eq!(quads.len(), 1);
    assert_eq!(super::solid(&quads[0]), highlight);
    assert_eq!(quads[0].bounds, super::sb(6., 0., 12., 20.));
}
