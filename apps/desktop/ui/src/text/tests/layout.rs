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

#[test]
fn shape_text_with_style_matches_shape_text_and_handles_empty_runs() {
    let window = window_text_system();
    let style = crate::TextStyle {
        font_family: "Dummy".into(),
        color: crate::red(),
        background_color: Some(crate::blue()),
        ..Default::default()
    };
    for (text, wrap_width, clamp) in [
        ("", None, None),
        ("\n\n", None, None),
        ("ab\ncd\n\nef", None, None),
        ("aa bbb cccc ddddd eeee", Some(px(72.)), None),
        ("aa bbb cccc ddddd eeee", Some(px(72.)), Some(1)),
        ("aa bbb cccc ddddd eeee", Some(px(72.)), Some(0)),
        ("aa bbb cccc ddddd eeee", Some(px(72.)), Some(2)),
    ] {
        let single_run = [style.to_run(text.len())];
        let expected = window.shape_text(text.into(), px(16.), &single_run, wrap_width, clamp).unwrap();
        let actual = window.shape_text_with_style(text.into(), px(16.), &style, wrap_width, clamp).unwrap();
        assert_eq!(actual.len(), expected.len(), "text={text:?} clamp={clamp:?}");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_eq!(a.text, e.text);
            assert_eq!(a.wrap_boundaries(), e.wrap_boundaries());
            assert_eq!(a.width(), e.width());
            assert_eq!(a.decoration_runs().len(), e.decoration_runs().len());
            for (ad, ed) in a.decoration_runs().iter().zip(e.decoration_runs().iter()) {
                assert_eq!(ad.len, ed.len);
                assert_eq!(ad.color, ed.color);
                assert_eq!(ad.background_color, ed.background_color);
                assert_eq!(ad.underline, ed.underline);
                assert_eq!(ad.strikethrough, ed.strikethrough);
            }
        }
    }

    // Zero-length runs interleaved with multi-line runs must be skipped identically.
    let runs_with_zeros =
        [run(0, Hsla::default()), run(4, Hsla::default()), run(0, crate::red()), run(5, crate::red())];
    let lines = window.shape_text("ab\ncd\n\nef".into(), px(16.), &runs_with_zeros, None, None).unwrap();
    let decoration_lens: Vec<Vec<u32>> =
        lines.iter().map(|l| l.decoration_runs().iter().map(|r| r.len).collect()).collect();
    assert_eq!(decoration_lens, [vec![2], vec![1, 1], vec![], vec![2]]);
}

#[test]
fn shape_text_never_wraps_between_base_letter_and_combining_accent() {
    use unicode_segmentation::UnicodeSegmentation;
    let window = window_text_system();

    // Decomposed "café" ("cafe\u{0301}") after a space: must wrap before "cafe\u{0301}", never between
    // 'e' and '\u{0301}', and never treat '\u{0301}' as a mid-word break candidate.
    let text = "aa cafe\u{0301}s";
    let lines = window.shape_text(text.into(), px(16.), &runs(&[text.len()]), Some(px(72.)), None).unwrap();
    let layout = &lines[0].layout;
    let starts: Vec<usize> = layout
        .wrap_boundaries()
        .iter()
        .map(|b| layout.unwrapped_layout.runs[b.run_ix].glyphs[b.glyph_ix].index)
        .collect();
    assert_eq!(starts, [3], "word breaks at the space before cafe\\u{{0301}}s, not before the accent");

    // Hard-wrap inside a single long word across widths: every wrap boundary must land on a grapheme
    // cluster boundary, even when the width cuts right between 'e' and '\u{0301}' or is narrower than
    // a single cluster.
    for sample in ["aaaae\u{0301}bbbb", "aa o\u{0302}\u{0323}bb", "e\u{0301}ab", "aa 😀\u{FE0F}bb"] {
        let grapheme_starts: Vec<usize> = sample.grapheme_indices(true).map(|(i, _)| i).collect();
        for width in [10., 20., 35., 48., 50., 55., 60.] {
            let lines = window
                .shape_text(sample.into(), px(16.), &runs(&[sample.len()]), Some(px(width)), None)
                .unwrap();
            let layout = &lines[0].layout;
            for b in layout.wrap_boundaries() {
                let ix = layout.unwrapped_layout.runs[b.run_ix].glyphs[b.glyph_ix].index;
                assert!(
                    grapheme_starts.contains(&ix),
                    "wrap inside grapheme cluster for {sample:?} at width {width}: byte {ix} not in {grapheme_starts:?}"
                );
            }
        }
    }

    // Multi-glyph shaped clusters (DirectWrite / HarfBuzz emitting base + mark with the same cluster
    // index, or font fallback splitting base and combining mark into separate runs) must stay atomic.
    let cluster_text = "aaae\u{0301}b";
    let same_index_layout = crate::LineLayout {
        font_size: px(16.),
        width: px(50.),
        ascent: px(12.),
        descent: px(4.),
        len: cluster_text.len(),
        runs: vec![crate::ShapedRun {
            font_id: crate::FontId(0),
            glyphs: vec![
                crate::ShapedGlyph {
                    id: crate::GlyphId(1),
                    position: point(px(0.), px(0.)),
                    index: 0,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(1),
                    position: point(px(10.), px(0.)),
                    index: 1,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(1),
                    position: point(px(20.), px(0.)),
                    index: 2,
                    is_emoji: false,
                },
                // 'e' at 30..38, combining acute mark at same index=3 positioned at 36 (extends to 40).
                crate::ShapedGlyph {
                    id: crate::GlyphId(2),
                    position: point(px(30.), px(0.)),
                    index: 3,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(3),
                    position: point(px(36.), px(0.)),
                    index: 3,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(4),
                    position: point(px(40.), px(0.)),
                    index: 6,
                    is_emoji: false,
                },
            ],
        }],
    };
    let bounds = same_index_layout.compute_wrap_boundaries(cluster_text, px(38.), None);
    assert_eq!(bounds.as_slice(), &[WrapBoundary { run_ix: 0, glyph_ix: 3 }]);

    // Fallback-split runs: 'e' in run 0 (index 3), '\u{0301}' in run 1 (index 4).
    let split_run_layout = crate::LineLayout {
        font_size: px(16.),
        width: px(56.),
        ascent: px(12.),
        descent: px(4.),
        len: cluster_text.len(),
        runs: vec![
            crate::ShapedRun {
                font_id: crate::FontId(0),
                glyphs: vec![
                    crate::ShapedGlyph {
                        id: crate::GlyphId(1),
                        position: point(px(0.), px(0.)),
                        index: 0,
                        is_emoji: false,
                    },
                    crate::ShapedGlyph {
                        id: crate::GlyphId(1),
                        position: point(px(10.), px(0.)),
                        index: 1,
                        is_emoji: false,
                    },
                    crate::ShapedGlyph {
                        id: crate::GlyphId(1),
                        position: point(px(20.), px(0.)),
                        index: 2,
                        is_emoji: false,
                    },
                    crate::ShapedGlyph {
                        id: crate::GlyphId(2),
                        position: point(px(30.), px(0.)),
                        index: 3,
                        is_emoji: false,
                    },
                ],
            },
            crate::ShapedRun {
                font_id: crate::FontId(1),
                glyphs: vec![
                    crate::ShapedGlyph {
                        id: crate::GlyphId(3),
                        position: point(px(40.), px(0.)),
                        index: 4,
                        is_emoji: false,
                    },
                    crate::ShapedGlyph {
                        id: crate::GlyphId(4),
                        position: point(px(48.), px(0.)),
                        index: 6,
                        is_emoji: false,
                    },
                ],
            },
        ],
    };
    let bounds = split_run_layout.compute_wrap_boundaries(cluster_text, px(44.), None);
    assert_eq!(bounds.as_slice(), &[WrapBoundary { run_ix: 0, glyph_ix: 3 }]);

    // Non-monotonic intra-cluster mark x offset: a base glyph at x=20 with a mark positioned at x=31
    // whose next cluster begins at x=28 has total advance 28 <= wrap_width (30), so it must NOT wrap.
    let offset_text = "aae\u{0301}b";
    let offset_layout = crate::LineLayout {
        font_size: px(16.),
        width: px(36.),
        ascent: px(12.),
        descent: px(4.),
        len: offset_text.len(),
        runs: vec![crate::ShapedRun {
            font_id: crate::FontId(0),
            glyphs: vec![
                crate::ShapedGlyph {
                    id: crate::GlyphId(1),
                    position: point(px(0.), px(0.)),
                    index: 0,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(1),
                    position: point(px(10.), px(0.)),
                    index: 1,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(2),
                    position: point(px(20.), px(0.)),
                    index: 2,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(3),
                    position: point(px(31.), px(0.)),
                    index: 3,
                    is_emoji: false,
                },
                crate::ShapedGlyph {
                    id: crate::GlyphId(4),
                    position: point(px(28.), px(0.)),
                    index: 5,
                    is_emoji: false,
                },
            ],
        }],
    };
    let bounds = offset_layout.compute_wrap_boundaries(offset_text, px(30.), None);
    assert_eq!(
        bounds.as_slice(),
        &[WrapBoundary { run_ix: 0, glyph_ix: 4 }],
        "cluster ending at x=28 fits within 30 px; only the following 'b' wraps"
    );

    // max_lines clamping on a multi-glyph cluster stops at the limit without splitting the cluster.
    let clamped = same_index_layout.compute_wrap_boundaries(cluster_text, px(25.), Some(2));
    assert_eq!(clamped.as_slice(), &[WrapBoundary { run_ix: 0, glyph_ix: 2 }]);

    // Descending / RTL glyph indices (as emitted for visual RTL runs): stays linear/fast and keeps
    // combining marks attached to their RTL base character.
    let rtl_text = format!("prefix {}", "א\u{05B8}".repeat(200));
    let mut rtl_glyphs = Vec::new();
    let mut x = 0.0f32;
    for (i, ch) in rtl_text.char_indices().rev() {
        rtl_glyphs.push(crate::ShapedGlyph {
            id: crate::GlyphId(ch as u32),
            position: point(px(x), px(0.)),
            index: i,
            is_emoji: false,
        });
        x += 8.0;
    }
    let rtl_layout = crate::LineLayout {
        font_size: px(16.),
        width: px(x),
        ascent: px(12.),
        descent: px(4.),
        len: rtl_text.len(),
        runs: vec![crate::ShapedRun { font_id: crate::FontId(0), glyphs: rtl_glyphs }],
    };
    let rtl_grapheme_starts: Vec<usize> = rtl_text.grapheme_indices(true).map(|(i, _)| i).collect();
    let rtl_bounds = rtl_layout.compute_wrap_boundaries(&rtl_text, px(80.), None);
    assert!(!rtl_bounds.is_empty());
    for b in &rtl_bounds {
        let g = &rtl_layout.runs[b.run_ix].glyphs[b.glyph_ix];
        // The glyph at the wrap boundary must start a new cluster relative to the previous glyph.
        if b.glyph_ix > 0 {
            let prev = &rtl_layout.runs[b.run_ix].glyphs[b.glyph_ix - 1];
            let cluster_of = |ix: usize| {
                let pos = rtl_grapheme_starts.partition_point(|&s| s <= ix);
                rtl_grapheme_starts[pos.saturating_sub(1)]
            };
            assert_ne!(cluster_of(g.index), cluster_of(prev.index), "RTL wrap split a grapheme cluster");
        }
    }
}
