//! Text on a canvas: painting a label and measuring text.

use std::sync::Arc;

use folo_ui::{App, Hsla, Pixels, Point, SharedString, TextRun, Window, point, px};

use crate::theme::SANS;

pub fn paint_text(
    window: &mut Window,
    cx: &mut App,
    text: &str,
    origin: Point<Pixels>,
    color: Hsla,
    size_px: f32,
    align_right: bool,
) -> Pixels {
    let run = TextRun {
        len: text.len(),
        font: folo_ui::Font {
            features: folo_ui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])),
            ..folo_ui::font(SANS)
        },
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let line =
        window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None);
    let w = line.width;
    let o = if align_right { point(origin.x - w, origin.y) } else { origin };
    let _ = line.paint(o, px(size_px + 4.), window, cx);
    w
}

/// The width of `text` in Geist Mono at `size_px` (file names and paths), fallbacks included.
pub fn mono_width(window: &mut Window, text: &str, size_px: f32) -> Pixels {
    let run = TextRun {
        len: text.len(),
        font: folo_ui::font(crate::theme::MONO),
        color: folo_ui::black(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None).width
}

pub fn text_width(window: &mut Window, text: &str, size_px: f32) -> Pixels {
    let run = TextRun {
        len: text.len(),
        font: folo_ui::Font {
            features: folo_ui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])),
            ..folo_ui::font(SANS)
        },
        color: folo_ui::black(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None).width
}

/// Fits `text` inside `max_w` (trimming with `"…"` when too wide), or returns `None` when not
/// even a single character plus `"…"` fits.
pub fn elide_text(window: &mut Window, text: &str, size_px: f32, max_w: Pixels) -> Option<String> {
    if max_w <= px(0.) || text.is_empty() {
        return None;
    }
    if text_width(window, text, size_px) <= max_w {
        return Some(text.to_string());
    }
    if text_width(window, "…", size_px) > max_w {
        return None;
    }
    let boundaries: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let (mut lo, mut hi) = (0usize, boundaries.len());
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let end = boundaries.get(mid).copied().unwrap_or(text.len());
        let cand = format!("{}…", text[..end].trim_end());
        if text_width(window, &cand, size_px) <= max_w {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo == 0 {
        return None;
    }
    let end = boundaries.get(lo).copied().unwrap_or(text.len());
    let prefix = text[..end].trim_end();
    if prefix.is_empty() {
        return None;
    }
    Some(format!("{prefix}…"))
}
