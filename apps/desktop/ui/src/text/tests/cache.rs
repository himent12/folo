//! The per-window line layout cache: what the previous frame used is reused, everything else is shaped.

use super::{CountingTextSystem, run, runs};
use crate::{Hsla, SharedString, TextSystem, WindowTextSystem, px};
use std::sync::Arc;

fn counting() -> (Arc<CountingTextSystem>, WindowTextSystem) {
    let platform = Arc::new(CountingTextSystem::default());
    let window = WindowTextSystem::new(Arc::new(TextSystem::new(platform.clone())));
    (platform, window)
}

fn shape(window: &WindowTextSystem, text: &'static str, size: f32) -> crate::ShapedLine {
    window.shape_line(SharedString::from(text), px(size), &runs(&[text.len()]), None)
}

#[test]
fn an_unchanged_frame_shapes_nothing() {
    let (platform, window) = counting();
    shape(&window, "best F1 · conf 0.41", 13.);
    shape(&window, "epoch 12", 13.);
    shape(&window, "best F1 · conf 0.41", 13.);
    assert_eq!(platform.layout_calls(), 2, "a repeat within the frame is a hit");
    window.finish_frame();

    for _ in 0..3 {
        shape(&window, "best F1 · conf 0.41", 13.);
        shape(&window, "epoch 12", 13.);
        window.finish_frame();
    }
    assert_eq!(platform.layout_calls(), 2, "frames that show the same text shape nothing");
    assert_eq!(window.shaped_line_count(), 2);
}

#[test]
fn a_layout_unused_for_a_frame_is_dropped() {
    let (platform, window) = counting();
    shape(&window, "a", 13.);
    window.finish_frame();
    shape(&window, "b", 13.);
    window.finish_frame();
    shape(&window, "a", 13.);
    assert_eq!(platform.layout_calls(), 3);
}

#[test]
fn the_key_is_text_size_runs_wrap_and_force_width() {
    let (platform, window) = counting();
    let text = SharedString::from("abc def");
    let one = runs(&[7]);
    window.layout_line(&text, px(13.), &one, None);
    window.layout_line(&text, px(14.), &one, None);
    window.layout_line(&text, px(13.), &one, Some(px(8.)));
    // Another colour splits the font runs (decorations differ), so the key's runs differ.
    let two = vec![run(3, Hsla::default()), run(4, crate::red())];
    window.layout_line(&text, px(13.), &two, None);
    assert_eq!(platform.layout_calls(), 4);
    // A background colour alone does not split font runs.
    let mut bg = runs(&[3, 4]);
    bg[1].background_color = Some(crate::red());
    window.layout_line(&text, px(13.), &bg, None);
    assert_eq!(platform.layout_calls(), 4);

    // Wrapped lines at two widths share one unwrapped layout.
    window.shape_text(text.clone(), px(13.), &one, Some(px(30.)), None).unwrap();
    window.shape_text(text.clone(), px(13.), &one, Some(px(40.)), None).unwrap();
    window.shape_text(text, px(13.), &one, Some(px(40.)), None).unwrap();
    assert_eq!(platform.layout_calls(), 4);
}

#[test]
fn reused_ranges_survive_frames_without_being_laid_out() {
    let (platform, window) = counting();
    let start = window.layout_index();
    shape(&window, "cached element", 13.);
    window.shape_text("wrapped\ntext".into(), px(13.), &runs(&[12]), Some(px(50.)), None).unwrap();
    let end = window.layout_index();
    assert_ne!(start, end);
    window.finish_frame();
    let shaped = platform.layout_calls();

    // Frame 2 reuses the element without laying it out (including a duplicate reuse and a direct
    // hit in the same frame to verify per-frame counter deduplication in `finish_frame`).
    window.reuse_layouts(start.clone()..end.clone());
    window.reuse_layouts(start..end);
    shape(&window, "cached element", 13.);
    window.finish_frame();

    // Frame 3 lays it out again: still cached.
    shape(&window, "cached element", 13.);
    window.shape_text("wrapped\ntext".into(), px(13.), &runs(&[12]), Some(px(50.)), None).unwrap();
    assert_eq!(platform.layout_calls(), shaped);
}

#[test]
fn truncated_layouts_are_not_carried_over_by_reuse() {
    let (platform, window) = counting();
    let start = window.layout_index();
    shape(&window, "kept", 13.);
    let mid = window.layout_index();
    shape(&window, "discarded", 13.);
    window.truncate_layouts(mid.clone());
    assert_eq!(window.layout_index(), mid);
    window.finish_frame();

    // Frame 2 reuses the element's (truncated) range only.
    window.reuse_layouts(start..mid);
    window.finish_frame();

    shape(&window, "kept", 13.);
    assert_eq!(platform.layout_calls(), 2);
    shape(&window, "discarded", 13.);
    assert_eq!(platform.layout_calls(), 3);
}

#[test]
fn the_line_clamp_is_part_of_a_wrapped_key() {
    let (platform, window) = counting();
    let text = SharedString::from("alpha beta gamma delta epsilon zeta eta theta");
    let one = runs(&[text.len()]);
    let lines = |clamp| {
        let lines = window.shape_text(text.clone(), px(10.), &one, Some(px(65.)), clamp).unwrap();
        lines.iter().map(|line| line.wrap_boundaries().len() + 1).sum::<usize>()
    };
    // Within a frame, and in later frames, in both orders.
    assert_eq!((lines(Some(2)), lines(Some(4)), lines(None)), (2, 4, 7));
    window.finish_frame();
    assert_eq!((lines(Some(4)), lines(Some(2))), (4, 2));
    // The clamps share one shaped line.
    assert_eq!(platform.layout_calls(), 1);
    window.finish_frame();
    assert_eq!(lines(Some(4)), 4);
    window.finish_frame();
    assert_eq!(lines(Some(2)), 2);
}

#[test]
fn layouts_from_before_fonts_were_added_are_not_reused() {
    let platform = Arc::new(super::WideningTextSystem::default());
    let text_system = Arc::new(TextSystem::new(platform));
    let window = WindowTextSystem::new(text_system.clone());
    let text = SharedString::from("abc def");
    let wrapped = |window: &WindowTextSystem| {
        let lines = window.shape_text(text.clone(), px(10.), &runs(&[7]), Some(px(50.)), None).unwrap();
        lines[0].wrap_boundaries().len()
    };
    assert_eq!((shape(&window, "abc def", 10.).width, wrapped(&window)), (px(42.), 0));
    window.finish_frame();
    assert_eq!((shape(&window, "abc def", 10.).width, wrapped(&window)), (px(42.), 0));
    // The same font ids now shape twice as wide: in this frame and the next, nothing is reused.
    text_system.add_fonts(Vec::new()).unwrap();
    assert_eq!((shape(&window, "abc def", 10.).width, wrapped(&window)), (px(84.), 1));
    window.finish_frame();
    assert_eq!((shape(&window, "abc def", 10.).width, wrapped(&window)), (px(84.), 1));
}

#[test]
fn borrowed_layout_line_and_shape_line_share_entries_across_frames() {
    let (platform, window) = counting();
    let owned = String::from("square 0.94");
    let one = runs(&[owned.len()]);

    // Populate via borrowed `&str`, then repeat within the same frame.
    let first = window.layout_line(&owned, px(13.), &one, None);
    let idx_after_first = window.layout_index();
    let repeat_same_frame = window.layout_line(&owned, px(13.), &one, None);
    assert!(Arc::ptr_eq(&first, &repeat_same_frame));
    assert_eq!(window.layout_index(), idx_after_first, "same-frame hit does not grow used_lines");
    assert_eq!(platform.layout_calls(), 1);
    window.finish_frame();

    // Next frame: `shape_line` hits the entry populated by `layout_line`, reusing the exact `Arc`.
    let shaped = window.shape_line(SharedString::from(owned.clone()), px(13.), &one, None);
    assert!(Arc::ptr_eq(&first, &shaped.layout));
    assert_eq!(platform.layout_calls(), 1);
    window.finish_frame();

    // Skip one frame: the single-map retain evicts the unused entry.
    window.finish_frame();
    let reshaped = window.layout_line(&owned, px(13.), &one, None);
    assert!(!Arc::ptr_eq(&first, &reshaped));
    assert_eq!(platform.layout_calls(), 2);
}

#[test]
fn reused_layouts_survive_multiple_consecutive_frames() {
    let (platform, window) = counting();
    let start = window.layout_index();
    let line = shape(&window, "persistent", 13.);
    let wrapped = window.shape_text("two\nlines".into(), px(13.), &runs(&[9]), Some(px(40.)), None).unwrap();
    let end = window.layout_index();
    let shaped = platform.layout_calls();
    window.finish_frame();

    // Reuse across two consecutive frames without laying out directly.
    let range = start..end;
    let r1_start = window.layout_index();
    window.reuse_layouts(range.clone());
    let r1_end = window.layout_index();
    window.finish_frame();

    window.reuse_layouts(r1_start..r1_end);
    window.finish_frame();

    let line2 = shape(&window, "persistent", 13.);
    let wrapped2 = window.shape_text("two\nlines".into(), px(13.), &runs(&[9]), Some(px(40.)), None).unwrap();
    assert!(Arc::ptr_eq(&line.layout, &line2.layout));
    assert!(Arc::ptr_eq(&wrapped[0].layout, &wrapped2[0].layout));
    assert!(Arc::ptr_eq(&wrapped[1].layout, &wrapped2[1].layout));
    assert_eq!(platform.layout_calls(), shaped);
}
