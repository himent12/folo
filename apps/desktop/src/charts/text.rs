//! Text on a canvas: painting a label and measuring text.

use std::sync::Arc;

use gpui::{App, Hsla, Pixels, Point, SharedString, TextRun, Window, point, px};

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
        font: gpui::Font {
            features: gpui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])),
            ..gpui::font(SANS)
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
        font: gpui::font(crate::theme::MONO),
        color: gpui::black(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None).width
}

pub fn text_width(window: &mut Window, text: &str, size_px: f32) -> Pixels {
    let run = TextRun {
        len: text.len(),
        font: gpui::Font {
            features: gpui::FontFeatures(Arc::new(vec![("tnum".into(), 1)])),
            ..gpui::font(SANS)
        },
        color: gpui::black(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window.text_system().shape_line(SharedString::from(text.to_string()), px(size_px), &[run], None).width
}
