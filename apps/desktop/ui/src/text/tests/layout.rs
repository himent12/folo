//! `LineLayout` queries, `shape_text` splitting / wrapping / clamping and `WrappedLineLayout` positions.

use super::{dummy_font, run, runs, window_text_system};
use crate::{Hsla, SharedString, TextRun, WrapBoundary, point, px};

const W: f32 = 9.6; // a fixed glyph at 16 px

#[track_caller]
fn assert_near(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-3, "{a} != {b}");
}

#[test]
fn index_and_x_round_trip_on_character_boundaries() {
    let window = window_text_system();
    let text = "héllo · wörld";
    let line = window.shape_line(text.into(), px(16.), &runs(&[text.len()]), None);
    assert_eq!(line.len(), text.len());
    assert_near(line.width.0, W * text.chars().count() as f32);

    let last = text.char_indices().last().unwrap().0;
    for (n, (ix, _)) in text.char_indices().enumerate() {
        let x = line.x_for_index(ix);
        assert_near(x.0, W * n as f32);
        assert_eq!(line.index_for_x(x), Some(ix));
        assert_eq!(line.index_for_x(x + px(W - 0.1)), Some(ix), "inside the glyph");
        assert_eq!(line.closest_index_for_x(x), ix);
        if ix != last {
            assert_eq!(line.closest_index_for_x(x + px(W * 0.4)), ix, "nearer this boundary");
        }
    }
    assert_eq!(line.x_for_index(text.len()), line.width);
    assert_eq!(line.index_for_x(line.width), None);
    assert_eq!(line.closest_index_for_x(line.width + px(5.)), text.len());
    // Anywhere inside the last glyph is the end (GPUI's quirk: no glyph follows to compare with).
    assert_eq!(line.closest_index_for_x(line.x_for_index(last) + px(W * 0.4)), text.len());
    assert_eq!(line.closest_index_for_x(line.x_for_index(last) + px(W * 0.6)), text.len());
}

#[test]
fn a_one_byte_line_rounds_at_its_middle() {
    let window = window_text_system();
    let line = window.shape_line("x".into(), px(16.), &runs(&[1]), None);
    assert_eq!(line.closest_index_for_x(px(4.)), 0);
    assert_eq!(line.closest_index_for_x(px(6.)), 1);
}

#[test]
fn force_width_snaps_glyphs_to_the_grid() {
    let window = window_text_system();
    let layout = window.layout_line("abcd", px(16.), &runs(&[4]), Some(px(12.)));
    let xs: Vec<f32> = layout.runs[0].glyphs.iter().map(|g| g.position.x.0).collect();
    assert_eq!(xs, [0., 12., 24., 36.]);
    // Within a pixel of the grid, positions stay.
    let layout = window.layout_line("abcd", px(16.), &runs(&[4]), Some(px(10.)));
    let xs: Vec<f32> = layout.runs[0].glyphs.iter().map(|g| g.position.x.0).collect();
    for (x, want) in xs.into_iter().zip([0., 9.6, 19.2, 30.]) {
        assert_near(x, want);
    }
}

/// GPUI's `test_wrap_shaped_line` (macOS there), with a backend whose bold font is another font id,
/// so boundaries land in later runs.
#[test]
fn shape_text_wraps_like_gpui() {
    let window = window_text_system();
    let normal = |len| run(len, Hsla::default());
    let bold = |len| TextRun { font: dummy_font().bold(), ..run(len, Hsla::default()) };
    let lines = window
        .shape_text(
            "aa bbb cccc ddddd eeee".into(),
            px(16.),
            &[normal(4), bold(5), normal(6), bold(1), normal(7)],
            Some(px(72.)),
            None,
        )
        .unwrap();
    assert_eq!(lines.len(), 1);
    let layout = &lines[0].layout;
    let starts: Vec<usize> = layout
        .wrap_boundaries()
        .iter()
        .map(|b| layout.unwrapped_layout.runs[b.run_ix].glyphs[b.glyph_ix].index)
        .collect();
    assert_eq!(starts, [7, 12, 18]);
    assert_eq!(
        layout.wrap_boundaries(),
        [
            WrapBoundary { run_ix: 1, glyph_ix: 3 },
            WrapBoundary { run_ix: 2, glyph_ix: 3 },
            WrapBoundary { run_ix: 4, glyph_ix: 2 },
        ]
    );
    assert_eq!(layout.size(px(20.)), crate::size(px(72.), px(80.)));
}

#[test]
fn shape_text_splits_lines_and_their_runs() {
    let window = window_text_system();
    let red = crate::red();
    let text = SharedString::from("ab\ncd\n\nef");
    let lines =
        window.shape_text(text, px(16.), &[run(4, Hsla::default()), run(5, red)], None, None).unwrap();
    let texts: Vec<&str> = lines.iter().map(|l| l.text.as_ref()).collect();
    assert_eq!(texts, ["ab", "cd", "", "ef"]);
    // "ab\n" and "c" are the first run, "d\n\nef" the second.
    let decoration_lens: Vec<Vec<u32>> =
        lines.iter().map(|l| l.decoration_runs().iter().map(|r| r.len).collect()).collect();
    assert_eq!(decoration_lens, [vec![2], vec![1, 1], vec![], vec![2]]);
    assert_eq!(lines[3].decoration_runs()[0].color, red);
    assert_eq!(lines[2].width(), px(0.));
}

#[test]
fn line_clamp_limits_the_wrapped_lines() {
    let window = window_text_system();
    let text = "aa bbb cccc ddddd eeee";
    let lines =
        window.shape_text(text.into(), px(16.), &runs(&[text.len()]), Some(px(72.)), Some(2)).unwrap();
    assert_eq!(lines[0].wrap_boundaries().len(), 1);
    // The clamp is part of the cache key (GPUI's is not, so there the first clamp won).
    let lines =
        window.shape_text(text.into(), px(16.), &runs(&[text.len()]), Some(px(72.)), Some(1)).unwrap();
    assert!(lines[0].wrap_boundaries().is_empty());
}

#[test]
fn wrapped_positions_round_trip() {
    let window = window_text_system();
    let text = "aa bbb cccc ddddd eeee";
    let lines = window.shape_text(text.into(), px(16.), &runs(&[text.len()]), Some(px(72.)), None).unwrap();
    let layout = &lines[0].layout;
    let lh = px(20.);
    // "cccc " starts the second visual line at index 7.
    // Index 7 is the end of line one (GPUI's tie-break), 8 is on line two.
    let at7 = layout.position_for_index(7, lh).unwrap();
    assert_near(at7.x.0, 7. * W);
    assert_eq!(at7.y, px(0.));
    let at8 = layout.position_for_index(8, lh).unwrap();
    assert_near(at8.x.0, W);
    assert_eq!(at8.y, px(20.));
    assert_eq!(layout.index_for_position(point(px(W + 1.), px(25.)), lh), Ok(8));
    assert_eq!(layout.closest_index_for_position(point(px(W * 1.6), px(25.)), lh), Ok(9));
    assert_eq!(layout.index_for_position(point(px(500.), px(5.)), lh), Err(7));
    assert_eq!(layout.index_for_position(point(px(5.), px(500.)), lh), Err(0));
}
