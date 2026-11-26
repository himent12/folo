// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system/line_wrapper.rs (tests),
// crates/gpui/src/elements/text.rs (folo_truncation_tests)
//! GPUI's `LineWrapper` tests (`text_system/line_wrapper.rs`), and the FOLO truncation tests
//! (`elements/text.rs`, `folo_truncation_tests`) against the non-mutating `truncate_line`.

use super::{CountingTextSystem, mono, runs, text_system};
use crate::text::line_wrapper::runs_after_truncation;
use crate::{Boundary, LineFragment, LineWrapper, Pixels, SharedString, TextRun, TextSystem, px};
use std::sync::Arc;

fn build_wrapper() -> crate::LineWrapperHandle {
    text_system().line_wrapper(mono(), px(16.))
}

fn wrap(wrapper: &mut LineWrapper, fragments: &[LineFragment], width: f32) -> Vec<Boundary> {
    wrapper.wrap_line(fragments, px(width)).collect()
}

#[test]
fn wrap_line() {
    let mut wrapper = build_wrapper();
    let b = Boundary::new;
    let text = LineFragment::text;

    assert_eq!(wrap(&mut wrapper, &[text("aa bbb cccc ddddd eeee")], 72.), [b(7, 0), b(12, 0), b(18, 0)]);
    assert_eq!(wrap(&mut wrapper, &[text("aaa aaaaaaaaaaaaaaaaaa")], 72.), [b(4, 0), b(11, 0), b(18, 0)]);
    assert_eq!(wrap(&mut wrapper, &[text("     aaaaaaa")], 72.), [b(7, 5), b(9, 5), b(11, 5)]);
    assert_eq!(
        wrap(&mut wrapper, &[text("                            ")], 72.),
        [b(7, 0), b(14, 0), b(21, 0)]
    );
    assert_eq!(
        wrap(&mut wrapper, &[text("          aaaaaaaaaaaaaa")], 72.),
        [b(7, 0), b(14, 3), b(18, 3), b(22, 3)]
    );

    // Several text fragments.
    assert_eq!(
        wrap(&mut wrapper, &[text("aa bbb "), text("cccc ddddd eeee")], 72.),
        [b(7, 0), b(12, 0), b(18, 0)]
    );

    // Text and element fragments.
    assert_eq!(
        wrap(
            &mut wrapper,
            &[
                text("aa "),
                LineFragment::element(px(20.), 1),
                text(" bbb "),
                LineFragment::element(px(30.), 1),
                text(" cccc"),
            ],
            72.
        ),
        [b(5, 0), b(9, 0), b(11, 0)]
    );

    // An element first, then text.
    assert_eq!(
        wrap(&mut wrapper, &[LineFragment::element(px(50.), 1), text(" aaaa bbbb cccc dddd")], 72.),
        [b(2, 0), b(7, 0), b(12, 0), b(17, 0)]
    );

    // A large element that forces wrapping by itself.
    assert_eq!(
        wrap(
            &mut wrapper,
            &[text("short text "), LineFragment::element(px(100.), 1), text(" more text")],
            72.
        ),
        [b(6, 0), b(11, 0), b(12, 0), b(18, 0)]
    );
}

#[test]
fn truncate_line() {
    let mut wrapper = build_wrapper();

    fn check(wrapper: &mut LineWrapper, text: &'static str, result: &'static str, ellipsis: &str) {
        let element_runs = runs(&[text.len()]);
        let (cut, cut_runs) = wrapper.truncate_line(text.into(), px(220.), ellipsis, &element_runs);
        assert_eq!(cut, result);
        assert_eq!(cut_runs[0].len, result.len());
        assert_eq!(element_runs[0].len, text.len(), "the caller's runs are untouched");
    }

    check(&mut wrapper, "aa bbb cccc ddddd eeee ffff gggg", "aa bbb cccc ddddd eeee", "");
    check(&mut wrapper, "aa bbb cccc ddddd eeee ffff gggg", "aa bbb cccc ddddd eee…", "…");
    check(&mut wrapper, "aa bbb cccc ddddd eeee ffff gggg", "aa bbb cccc dddd......", "......");
}

#[test]
fn truncate_multiple_runs() {
    let mut wrapper = build_wrapper();

    fn check(
        wrapper: &mut LineWrapper,
        text: &'static str,
        result: &str,
        run_lens: &[usize],
        result_run_len: &[usize],
        line_width: Pixels,
    ) {
        let element_runs = runs(run_lens);
        let (cut, cut_runs) = wrapper.truncate_line(text.into(), line_width, "…", &element_runs);
        assert_eq!(cut, result);
        assert_eq!(cut_runs.iter().map(|r| r.len).collect::<Vec<_>>(), result_run_len);
    }
    // One run: "abcd…", the run grows by the ellipsis.
    check(&mut wrapper, "abcdefghijkl", "abcd…", &[12], &[7], px(50.));
    // Later runs dropped: "abcdef…".
    check(&mut wrapper, "abcdefghijkl", "abcdef…", &[4, 4, 4], &[4, 5], px(70.));
    // Cut at the start of a run: the ellipsis is that run.
    check(&mut wrapper, "abcdefghijkl", "abcdefgh…", &[4, 4, 4], &[4, 4, 3], px(90.));
}

#[test]
fn update_runs_after_truncation() {
    fn check(result: &str, run_lens: &[usize], result_run_lens: &[usize]) {
        let element_runs = runs(run_lens);
        let cut = runs_after_truncation(result, "…", &element_runs);
        assert_eq!(cut.iter().map(|r| r.len).collect::<Vec<_>>(), result_run_lens);
    }
    check("abcd…", &[12], &[7]);
    check("abcdef…", &[4, 4, 4], &[4, 5]);
    check("abcdefgh…", &[4, 4, 4], &[4, 4, 3]);
}

#[test]
fn no_truncation_returns_the_text_and_a_copy_of_the_runs() {
    let mut wrapper = build_wrapper();
    let element_runs = runs(&[3, 2]);
    let (text, cut_runs) = wrapper.truncate_line("abcde".into(), px(200.), "…", &element_runs);
    assert_eq!(text, "abcde");
    assert_eq!(cut_runs, element_runs);

    // Borrowed variant avoids cloning the runs slice when the line already fits.
    let (cow_text, cow_runs) = wrapper.truncate_line_cow("abcde".into(), px(200.), "…", &element_runs);
    assert_eq!(cow_text, "abcde");
    assert!(matches!(cow_runs, std::borrow::Cow::Borrowed(_)));
    assert_eq!(cow_runs.as_ref(), element_runs.as_slice());

    let (trunc_text, trunc_runs) = wrapper.truncate_line_cow("abcde".into(), px(20.), "…", &element_runs);
    assert!(matches!(trunc_runs, std::borrow::Cow::Owned(_)));
    let (expected_text, expected_runs) = wrapper.truncate_line("abcde".into(), px(20.), "…", &element_runs);
    assert_eq!(trunc_text, expected_text);
    assert_eq!(trunc_runs.as_ref(), expected_runs.as_slice());
}

#[test]
fn is_word_char() {
    #[track_caller]
    fn assert_word(word: &str) {
        for c in word.chars() {
            assert!(LineWrapper::is_word_char(c), "assertion failed for '{c}'");
        }
    }

    #[track_caller]
    fn assert_not_word(word: &str) {
        let found = word.chars().any(|c| !LineWrapper::is_word_char(c));
        assert!(found, "assertion failed for '{word}'");
    }

    assert_word("Hello123");
    assert_word("non-English");
    assert_word("var_name");
    assert_word("123456");
    assert_word("3.1415");
    assert_word("10^2");
    assert_word("1~2");
    assert_word("100%");
    assert_word("@mention");
    assert_word("#hashtag");
    assert_word("$variable");
    assert_word("a=1");
    assert_word("Self::is_word_char");
    assert_word("more⋯");

    assert_not_word("foo bar");

    assert_word("github.com");
    assert_not_word("zed-industries/zed");
    assert_not_word("zed-industries\\zed");
    assert_not_word("a=1&b=2");
    assert_not_word("foo?b=2");

    assert_word("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏ");
    assert_word("ĀāĂăĄąĆćĈĉĊċČčĎď");
    assert_word("ƀƁƂƃƄƅƆƇƈƉƊƋƌƍƎƏ");
    assert_word("АБВГДЕЖЗИЙКЛМНОП");

    assert_not_word("你好");
    assert_not_word("안녕하세요");
    assert_not_word("こんにちは");
    assert_not_word("😀😁😂");
    assert_not_word("()[]{}<>");
    // The app's separators: "·", "–" and "…" are not word characters (a line may break before
    // them); "×" (U+00D7) is, being in the Latin-1 Supplement range.
    assert_not_word("·");
    assert_not_word("–");
    assert_not_word("…");
    assert_word("×");
}

/// The runs cover `text` exactly, every boundary on a character boundary.
fn assert_covers(text: &str, runs: &[TextRun]) {
    let mut at = 0;
    for run in runs {
        at += run.len;
        assert!(text.is_char_boundary(at), "run ends inside a character of {text:?} at {at}");
    }
    assert_eq!(at, text.len(), "runs cover {at} of {text:?}'s {} bytes", text.len());
}

/// The FOLO fix: measuring truncated, then untruncated, text never reuses the shortened runs.
#[test]
fn truncated_then_untruncated_with_multibyte_text() {
    let mut wrapper = build_wrapper();
    for text in [
        "best F1 · conf 0.412 · val",
        "1024 × 640 · simulated",
        "café crème · résumé",
        "epochs 91–100 · last…",
        "漢字のラベル · 数据集",
    ] {
        let text = SharedString::from(text);
        // One run, and two (split mid-text on a character boundary), as styled text has.
        let split = text.char_indices().nth(5).map_or(text.len(), |(i, _)| i);
        for element_runs in [runs(&[text.len()]), runs(&[split, text.len() - split])] {
            for width in [40., 80., 120., 200.] {
                // A narrow measure truncates ...
                let (cut, cut_runs) = wrapper.truncate_line(text.clone(), px(width), "…", &element_runs);
                assert_covers(&cut, &cut_runs);
                // ... and the next, unconstrained one starts from the element's own runs.
                assert_covers(&text, &element_runs);
            }
        }
    }
}

#[test]
fn wrappers_return_to_the_pool_with_their_width_cache() {
    let counting = Arc::new(CountingTextSystem::default());
    let ts = Arc::new(TextSystem::new(counting.clone()));
    {
        let mut wrapper = ts.line_wrapper(mono(), px(16.));
        assert_eq!(
            wrap(&mut wrapper, &[LineFragment::text("ab ab ab")], 30.),
            [Boundary::new(3, 0), Boundary::new(6, 0)]
        );
    }
    let shaped = counting.layout_calls();
    assert_eq!(shaped, 3, "one platform layout per distinct character");
    let mut wrapper = ts.line_wrapper(mono(), px(16.));
    let _ = wrap(&mut wrapper, &[LineFragment::text("ba ba")], 30.);
    assert_eq!(counting.layout_calls(), shaped, "the pooled wrapper kept its widths");
    // Another size is another wrapper.
    let mut other = ts.line_wrapper(mono(), px(12.));
    let _ = wrap(&mut other, &[LineFragment::text("a")], 30.);
    assert_eq!(counting.layout_calls(), shaped + 1);
}

#[test]
fn wrappers_from_before_fonts_were_added_are_not_reused() {
    let text_system = |added: bool| {
        let text_system = Arc::new(TextSystem::new(Arc::new(super::WideningTextSystem::default())));
        if added {
            text_system.add_fonts(Vec::new()).unwrap();
        }
        text_system
    };
    let truncate =
        |wrapper: &mut LineWrapper| wrapper.truncate_line("abcdefghij".into(), px(60.), "…", &runs(&[10])).0;
    let want = truncate(&mut text_system(true).line_wrapper(super::dummy_font(), px(10.)));
    assert_eq!(want, "abc…");

    let text_system = text_system(false);
    // One wrapper goes back to the pool before fonts are added, one is held until after.
    let mut pooled = text_system.line_wrapper(super::dummy_font(), px(10.));
    let mut held = text_system.line_wrapper(super::dummy_font(), px(10.));
    assert_eq!((truncate(&mut pooled), truncate(&mut held)), ("abcdefghij".into(), "abcdefghij".into()));
    drop(pooled);
    text_system.add_fonts(Vec::new()).unwrap();
    drop(held);
    let mut first = text_system.line_wrapper(super::dummy_font(), px(10.));
    let mut second = text_system.line_wrapper(super::dummy_font(), px(10.));
    assert_eq!((truncate(&mut first), truncate(&mut second)), (want.clone(), want));
}

#[test]
fn wrap_and_truncate_never_split_grapheme_clusters() {
    use unicode_segmentation::UnicodeSegmentation;
    let mut wrapper = build_wrapper();

    // "aa cafe\u{0301}s": FixedTextSystem gives 9.6 px per char, so "aa cafe\u{0301}" is 8*9.6 = 76.8 px.
    // Must wrap at the space before "cafe\u{0301}s" (byte 3), never between 'e' and '\u{0301}' (byte 7).
    let text = "aa cafe\u{0301}s";
    assert_eq!(wrap(&mut wrapper, &[LineFragment::text(text)], 72.), [Boundary::new(3, 0)]);

    // Across widths, hard-wrapping and truncation must always land on extended grapheme boundaries.
    for sample in ["aaaae\u{0301}bbbb", "cafe\u{0301} cre\u{0300}me", "o\u{0302}\u{0323}abc", "😀\u{FE0F}xyz"]
    {
        let grapheme_starts: Vec<usize> = sample.grapheme_indices(true).map(|(i, _)| i).collect();
        for width in [10., 20., 35., 48., 50., 55., 60., 72.] {
            for b in wrap(&mut wrapper, &[LineFragment::text(sample)], width) {
                assert!(
                    grapheme_starts.contains(&b.ix),
                    "wrap_line split grapheme in {sample:?} at width {width}: byte {} not in {grapheme_starts:?}",
                    b.ix
                );
            }
            let (cut, cut_runs) =
                wrapper.truncate_line(sample.into(), px(width), "…", &runs(&[sample.len()]));
            assert_covers(&cut, &cut_runs);
            if let Some(prefix) = cut.strip_suffix('…') {
                assert!(
                    prefix.len() == sample.len() || grapheme_starts.contains(&prefix.len()),
                    "truncate_line split grapheme in {sample:?} at width {width}: prefix {prefix:?}"
                );
            }
        }
    }

    // Consecutive text fragments that split a grapheme cluster or `\r\n` must be segmented together
    // so neither wraps between base + combining mark nor measures `\r` separately.
    assert_eq!(
        wrap(&mut wrapper, &[LineFragment::text("ae"), LineFragment::text("\u{0301}b")], 20.),
        [Boundary::new(1, 0), Boundary::new(4, 0)],
        "must wrap before 'e' (byte 1) and 'b' (byte 4), never at byte 2 between 'e' and accent"
    );
    assert_eq!(
        wrap(&mut wrapper, &[LineFragment::text("a\r"), LineFragment::text("\nb")], 100.),
        wrap(&mut wrapper, &[LineFragment::text("a\r\nb")], 100.)
    );
    // CRLF truncation stays on a grapheme boundary (never keeps a dangling `\r`).
    let crlf = "ab\r\ncd";
    for width in [10., 20., 25., 30., 40.] {
        let (cut, cut_runs) = wrapper.truncate_line(crlf.into(), px(width), "…", &runs(&[crlf.len()]));
        assert_covers(&cut, &cut_runs);
        if let Some(prefix) = cut.strip_suffix('…') {
            assert!(!prefix.ends_with('\r'), "truncate_line split CRLF at width {width}: {cut:?}");
        }
    }
}
