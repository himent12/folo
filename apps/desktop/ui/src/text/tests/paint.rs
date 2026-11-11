//! `paint_line` through a recording `GlyphPainter`: baseline, alignment, wraps, colours, decorations,
//! the content mask and emoji.

use super::{run, runs, window_text_system};
use crate::{
    Bounds, FontId, GlyphId, GlyphPainter, Hsla, LineLayout, Pixels, Point, ShapedGlyph, ShapedRun,
    StrikethroughStyle, TextAlign, UnderlineStyle, paint_line, point, px, size,
};
use anyhow::Result;

#[derive(Debug, PartialEq)]
enum Op {
    Glyph { x: f32, y: f32, ch: char, color: Hsla },
    Emoji { x: f32, y: f32 },
    Underline { x: f32, y: f32, width: f32 },
    Strikethrough { x: f32, y: f32, width: f32 },
    Background { bounds: Bounds<Pixels>, color: Hsla },
}

#[derive(Default)]
struct Recorder(Vec<Op>);

impl GlyphPainter for Recorder {
    fn paint_glyph(
        &mut self,
        origin: Point<Pixels>,
        _: FontId,
        glyph: GlyphId,
        _: Pixels,
        color: Hsla,
    ) -> Result<()> {
        let ch = char::from_u32(glyph.0).unwrap();
        self.0.push(Op::Glyph { x: origin.x.0, y: origin.y.0, ch, color });
        Ok(())
    }

    fn paint_emoji(&mut self, origin: Point<Pixels>, _: FontId, _: GlyphId, _: Pixels) -> Result<()> {
        self.0.push(Op::Emoji { x: origin.x.0, y: origin.y.0 });
        Ok(())
    }

    fn paint_underline(&mut self, origin: Point<Pixels>, width: Pixels, _: &UnderlineStyle) {
        self.0.push(Op::Underline { x: origin.x.0, y: origin.y.0, width: width.0 });
    }

    fn paint_strikethrough(&mut self, origin: Point<Pixels>, width: Pixels, _: &StrikethroughStyle) {
        self.0.push(Op::Strikethrough { x: origin.x.0, y: origin.y.0, width: width.0 });
    }

    fn paint_background(&mut self, bounds: Bounds<Pixels>, color: Hsla) {
        self.0.push(Op::Background { bounds, color });
    }
}

impl Recorder {
    fn glyphs(&self) -> Vec<(f32, f32, char)> {
        self.0
            .iter()
            .filter_map(|op| match op {
                Op::Glyph { x, y, ch, .. } => Some((*x, *y, *ch)),
                _ => None,
            })
            .collect()
    }
}

fn everywhere() -> Bounds<Pixels> {
    Bounds::new(point(px(-1000.), px(-1000.)), size(px(4000.), px(4000.)))
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

// Fixed font at 16 px: 9.6 px advance, ascent 12.8, descent 3.2. In a 20 px line the baseline is
// (20 - 12.8 - 3.2) / 2 + 12.8 = 14.8 below the top.
const W: f32 = 9.6;
const BASELINE: f32 = 14.8;

#[test]
fn glyphs_sit_on_the_baseline_in_their_run_colours() {
    let window = window_text_system();
    let red = crate::red();
    let line = window.shape_line("ab·c".into(), px(16.), &[run(2, Hsla::default()), run(3, red)], None);
    let mut rec = Recorder::default();
    line.paint_with(point(px(10.), px(5.)), px(20.), &window, everywhere(), &mut rec).unwrap();
    let colors: Vec<Hsla> = rec
        .0
        .iter()
        .map(|op| match op {
            Op::Glyph { color, .. } => *color,
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(colors, [Hsla::default(), Hsla::default(), red, red]);
    for (i, (x, y, _)) in rec.glyphs().into_iter().enumerate() {
        assert!(near(x, 10. + W * i as f32), "glyph {i} at {x}");
        assert!(near(y, 5. + BASELINE), "glyph {i} at y {y}");
    }
}

#[test]
fn alignment_moves_each_visual_line() {
    let window = window_text_system();
    let text = "aa bbb cccc";
    let lines = window.shape_text(text.into(), px(16.), &runs(&[text.len()]), Some(px(72.)), None).unwrap();
    let line = &lines[0];
    assert_eq!(line.wrap_boundaries().len(), 1, "\"aa bbb \" | \"cccc\"");
    let bounds = Bounds::new(point(px(0.), px(0.)), size(px(100.), px(40.)));
    for (align, first_x, second_x) in [
        (TextAlign::Left, 0., 0.),
        // Line one is 7 glyphs wide (its trailing space included), line two 4.
        (TextAlign::Center, (100. - 7. * W) / 2., (100. - 4. * W) / 2.),
        (TextAlign::Right, 100. - 7. * W, 100. - 4. * W),
    ] {
        let mut rec = Recorder::default();
        line.paint_with(point(px(0.), px(0.)), px(20.), align, Some(bounds), &window, everywhere(), &mut rec)
            .unwrap();
        let glyphs = rec.glyphs();
        assert_eq!(glyphs.len(), text.len());
        assert!(near(glyphs[0].0, first_x), "{align:?}: line one at {}", glyphs[0].0);
        assert!(near(glyphs[7].0, second_x), "{align:?}: line two at {}", glyphs[7].0);
        assert_eq!(glyphs[7].2, 'c');
        assert!(near(glyphs[7].1, 20. + BASELINE), "line two is one line height down");
        assert!(near(glyphs[8].0 - glyphs[7].0, W));
    }
}

#[test]
fn underline_and_strikethrough_cover_their_runs() {
    let window = window_text_system();
    let underline = UnderlineStyle { thickness: px(1.), color: None, wavy: false };
    let strike = StrikethroughStyle { thickness: px(1.), color: None };
    let mut runs = runs(&[2, 3, 2]);
    runs[1].underline = Some(underline);
    runs[1].strikethrough = Some(strike);
    let line = window.shape_line("abcdefg".into(), px(16.), &runs, None);
    let mut rec = Recorder::default();
    line.paint_with(point(px(0.), px(0.)), px(20.), &window, everywhere(), &mut rec).unwrap();
    let underline = rec.0.iter().find_map(|op| match op {
        Op::Underline { x, y, width } => Some((*x, *y, *width)),
        _ => None,
    });
    let (x, y, width) = underline.expect("an underline");
    assert!(near(x, 2. * W) && near(width, 3. * W));
    assert!(near(y, BASELINE + 3.2 * 0.618), "below the baseline by 0.618 descent: {y}");
    let strike = rec.0.iter().find_map(|op| match op {
        Op::Strikethrough { x, y, width } => Some((*x, *y, *width)),
        _ => None,
    });
    let (x, y, width) = strike.expect("a strikethrough");
    assert!(near(x, 2. * W) && near(width, 3. * W));
    assert!(near(y, (12.8 * 0.5 + BASELINE) * 0.5));
}

#[test]
fn an_underline_to_the_end_of_the_line() {
    let window = window_text_system();
    let mut runs = runs(&[4]);
    runs[0].underline = Some(UnderlineStyle { thickness: px(1.), color: None, wavy: false });
    let line = window.shape_line("abcd".into(), px(16.), &runs, None);
    let mut rec = Recorder::default();
    line.paint_with(point(px(3.), px(0.)), px(20.), &window, everywhere(), &mut rec).unwrap();
    let ops: Vec<_> = rec.0.iter().filter(|op| matches!(op, Op::Underline { .. })).collect();
    assert_eq!(ops.len(), 1);
    let Op::Underline { x, width, .. } = ops[0] else { unreachable!() };
    assert!(near(*x, 3.) && near(*width, 4. * W));
}

#[test]
fn glyphs_outside_the_content_mask_are_skipped() {
    let window = window_text_system();
    let line = window.shape_line("abcdefghij".into(), px(16.), &runs(&[10]), None);
    let mut rec = Recorder::default();
    // Bounding boxes are 9.6 wide from each glyph origin: glyphs 0..=3 reach into x < 35.
    let mask = Bounds::new(point(px(0.), px(0.)), size(px(35.), px(20.)));
    line.paint_with(point(px(0.), px(0.)), px(20.), &window, mask, &mut rec).unwrap();
    let chars: String = rec.glyphs().iter().map(|g| g.2).collect();
    assert_eq!(chars, "abcd");
}

#[test]
fn background_runs_become_quads() {
    let window = window_text_system();
    let blue = crate::blue();
    let mut runs = runs(&[1, 3, 2]);
    runs[1].background_color = Some(blue);
    let line = window.shape_line("abcdef".into(), px(16.), &runs, None);
    let mut rec = Recorder::default();
    line.paint_background_with(point(px(0.), px(10.)), px(20.), &window, &mut rec).unwrap();
    assert_eq!(rec.0.len(), 1);
    let Op::Background { bounds, color } = &rec.0[0] else { panic!("{:?}", rec.0) };
    assert_eq!(*color, blue);
    assert!(near(bounds.origin.x.0, W) && near(bounds.size.width.0, 3. * W));
    assert_eq!((bounds.origin.y, bounds.size.height), (px(10.), px(20.)));
}

#[test]
fn emoji_glyphs_go_to_paint_emoji() {
    let window = window_text_system();
    let font_id = window.resolve_font(&super::dummy_font());
    let glyph = |x: f32, index, is_emoji| ShapedGlyph {
        id: GlyphId('a' as u32),
        position: point(px(x), px(0.)),
        index,
        is_emoji,
    };
    let layout = LineLayout {
        font_size: px(16.),
        width: px(30.),
        ascent: px(12.8),
        descent: px(3.2),
        runs: vec![ShapedRun { font_id, glyphs: vec![glyph(0., 0, false), glyph(10., 1, true)] }],
        len: 5,
    };
    let mut rec = Recorder::default();
    let decorations = [crate::DecorationRun {
        len: 5,
        color: Hsla::default(),
        background_color: None,
        underline: None,
        strikethrough: None,
    }];
    paint_line(
        point(px(0.), px(0.)),
        &layout,
        px(20.),
        TextAlign::Left,
        None,
        &decorations,
        &[],
        &window,
        everywhere(),
        &mut rec,
    )
    .unwrap();
    assert!(matches!(rec.0[0], Op::Glyph { .. }));
    let Op::Emoji { x, y } = rec.0[1] else { panic!("{:?}", rec.0) };
    assert!(near(x, 10.) && near(y, BASELINE));
}
