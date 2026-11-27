//! Off-thread Unicode text rasterization and pixel-blending helpers for Playground PNG exports.
//!
//! Uses `folo-ui`'s `TextSystem` (DirectWrite on Windows, cosmic-text elsewhere) so non-ASCII
//! class and model names (accented Latin, CJK, symbols, emoji) render crisply, with a compact
//! 5×7 bitmap fallback when no `TextSystem` is supplied in pure unit tests.

use std::borrow::Cow;
use std::sync::Arc;

use folo_ui::{
    Font, FontWeight, RenderGlyphParams, SUBPIXEL_VARIANTS_X, SharedString, TextRun, TextSystem,
    WindowTextSystem, font, point, px, white,
};
use image::RgbaImage;

use crate::theme::SANS;

/// Integer pixel rectangle used for canvas clipping and label collision checks.
#[derive(Clone, Copy)]
pub(super) struct RectI {
    pub(super) x: i64,
    pub(super) y: i64,
    pub(super) w: i64,
    pub(super) h: i64,
}

impl RectI {
    pub(super) fn intersects(self, other: RectI, pad: i64) -> bool {
        self.x < other.x + other.w + pad
            && self.x + self.w + pad > other.x
            && self.y < other.y + other.h + pad
            && self.y + self.h + pad > other.y
    }
}

/// Alpha-blends `src` (straight RGBA) over `dst`.
pub(super) fn blend_pixel(dst: &mut image::Rgba<u8>, src: [u8; 4]) {
    let sa = src[3] as u32;
    if sa == 0 {
        return;
    }
    if sa == 255 {
        *dst = image::Rgba(src);
        return;
    }
    let inv = 255 - sa;
    dst[0] = ((src[0] as u32 * sa + dst[0] as u32 * inv) / 255) as u8;
    dst[1] = ((src[1] as u32 * sa + dst[1] as u32 * inv) / 255) as u8;
    dst[2] = ((src[2] as u32 * sa + dst[2] as u32 * inv) / 255) as u8;
    dst[3] = (sa + (dst[3] as u32 * inv) / 255).min(255) as u8;
}

/// Fills a rectangle clipped to both `clip` and `img` dimensions.
pub(super) fn fill_rect_clipped(
    img: &mut RgbaImage,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
    color: [u8; 4],
    clip: RectI,
) {
    if w <= 0 || h <= 0 {
        return;
    }
    let x0 = x.max(clip.x).max(0);
    let y0 = y.max(clip.y).max(0);
    let x1 = x.saturating_add(w).min(clip.x + clip.w).min(img.width() as i64);
    let y1 = y.saturating_add(h).min(clip.y + clip.h).min(img.height() as i64);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    for py in y0..y1 {
        for px in x0..x1 {
            blend_pixel(img.get_pixel_mut(px as u32, py as u32), color);
        }
    }
}

/// Normalizes ASCII and Unicode line/paragraph breaks (CR, LF, VT, FF, FS/GS/RS, NEL, U+2028,
/// U+2029) into single spaces so `WindowTextSystem::shape_line` always receives a single paragraph
/// on every text backend (DirectWrite and cosmic-text).
fn single_line_text(text: &str) -> Cow<'_, str> {
    fn is_line_or_para_break(ch: char) -> bool {
        matches!(
            ch,
            '\n' | '\r' | '\u{000B}' | '\u{000C}' | '\u{001C}'
                ..='\u{001E}' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
    }
    if !text.chars().any(is_line_or_para_break) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut prev_cr = false;
    for ch in text.chars() {
        match ch {
            '\r' => {
                out.push(' ');
                prev_cr = true;
            }
            '\n' => {
                if !prev_cr {
                    out.push(' ');
                }
                prev_cr = false;
            }
            c if is_line_or_para_break(c) => {
                out.push(' ');
                prev_cr = false;
            }
            other => {
                out.push(other);
                prev_cr = false;
            }
        }
    }
    Cow::Owned(out)
}

/// Text rasterizer for exported PNGs: shapes and rasterizes arbitrary Unicode text through
/// `folo-ui`'s `TextSystem` when available, falling back to the 5×7 ASCII bitmap font otherwise.
pub(super) struct ExportText {
    wts: Option<WindowTextSystem>,
}

impl ExportText {
    pub(super) fn new(text_system: Option<&Arc<TextSystem>>) -> Self {
        Self { wts: text_system.map(|ts| WindowTextSystem::new(Arc::clone(ts))) }
    }

    fn font_for(medium: bool) -> Font {
        let mut f = font(SANS);
        if medium {
            f.weight = FontWeight::MEDIUM;
        }
        f
    }

    pub(super) fn line_height(&self, font_px: f32, fallback_scale: u32) -> u32 {
        if let Some(wts) = &self.wts {
            let run = TextRun {
                len: 2,
                font: Self::font_for(true),
                color: white(),
                background_color: None,
                underline: None,
                strikethrough: None,
            };
            let shaped = wts.shape_line(SharedString::from("Ag"), px(font_px), &[run], None);
            let h = (f32::from(shaped.ascent) + f32::from(shaped.descent)).ceil();
            if h.is_finite() && h >= 1.0 {
                return h as u32;
            }
            return font_px.ceil().max(1.0) as u32;
        }
        7 * fallback_scale
    }

    pub(super) fn measure(&self, text: &str, font_px: f32, fallback_scale: u32, medium: bool) -> (f32, f32) {
        let clean = single_line_text(text);
        if clean.is_empty() {
            return (0.0, self.line_height(font_px, fallback_scale) as f32);
        }
        if let Some(wts) = &self.wts {
            let run = TextRun {
                len: clean.len(),
                font: Self::font_for(medium),
                color: white(),
                background_color: None,
                underline: None,
                strikethrough: None,
            };
            let shaped = wts.shape_line(SharedString::from(clean.into_owned()), px(font_px), &[run], None);
            let h = (f32::from(shaped.ascent) + f32::from(shaped.descent)).max(font_px);
            return (f32::from(shaped.width).max(0.0), h);
        }
        (text_width_px(&clean, fallback_scale) as f32, (7 * fallback_scale) as f32)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_clipped(
        &self,
        img: &mut RgbaImage,
        text: &str,
        x: i64,
        box_y: i64,
        box_h: i64,
        max_width: u32,
        font_px: f32,
        fallback_scale: u32,
        medium: bool,
        color: [u8; 4],
        clip: RectI,
    ) {
        let clean = single_line_text(text);
        if clean.is_empty() || max_width == 0 || box_h <= 0 {
            return;
        }
        let Some(wts) = &self.wts else {
            let ty = box_y + ((box_h - (7 * fallback_scale) as i64) / 2).max(0);
            draw_text_clipped_5x7(img, &clean, x, ty, max_width, fallback_scale, color, clip);
            return;
        };

        let fitted = self.truncate_to_width(wts, &clean, max_width as f32, font_px, medium);
        if fitted.is_empty() {
            return;
        }
        let run = TextRun {
            len: fitted.len(),
            font: Self::font_for(medium),
            color: white(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        let shaped = wts.shape_line(SharedString::from(fitted), px(font_px), &[run], None);
        let ascent = f32::from(shaped.ascent);
        let line_h = ascent + f32::from(shaped.descent);
        let baseline_y = box_y as f32 + ((box_h as f32 - line_h) * 0.5) + ascent;
        let max_x = x.saturating_add(max_width as i64).min(clip.x + clip.w).min(img.width() as i64);
        let min_x = clip.x.max(0);
        let min_y = clip.y.max(0);
        let max_y = (clip.y + clip.h).min(img.height() as i64);

        for run in &shaped.runs {
            for glyph in &run.glyphs {
                let gx = x as f32 + f32::from(glyph.position.x);
                let gy = baseline_y + f32::from(glyph.position.y);
                if gx >= max_x as f32 {
                    continue;
                }
                let sub_x = if glyph.is_emoji {
                    0
                } else {
                    ((gx.rem_euclid(1.0) * SUBPIXEL_VARIANTS_X as f32).floor() as u8)
                        .min(SUBPIXEL_VARIANTS_X.saturating_sub(1))
                };
                let params = RenderGlyphParams {
                    font_id: run.font_id,
                    glyph_id: glyph.id,
                    font_size: shaped.font_size,
                    subpixel_variant: point(sub_x, 0),
                    scale_factor: 1.0,
                    is_emoji: glyph.is_emoji,
                };
                let Ok(bounds) = wts.raster_bounds(&params) else { continue };
                if bounds.size.width.0 <= 0 || bounds.size.height.0 <= 0 {
                    continue;
                }
                let Ok((sz, bytes)) = wts.rasterize_glyph(&params) else { continue };
                let bw = sz.width.0 as i64;
                let bh = sz.height.0 as i64;
                if bw <= 0 || bh <= 0 {
                    continue;
                }
                let dst_x0 = gx.floor() as i64 + bounds.origin.x.0 as i64;
                let dst_y0 = gy.floor() as i64 + bounds.origin.y.0 as i64;

                if glyph.is_emoji {
                    if bytes.len() < (bw * bh * 4) as usize {
                        continue;
                    }
                    for py in 0..bh {
                        let ty = dst_y0 + py;
                        if ty < min_y || ty >= max_y {
                            continue;
                        }
                        for px_i in 0..bw {
                            let tx = dst_x0 + px_i;
                            if tx < min_x || tx >= max_x {
                                continue;
                            }
                            let idx = ((py * bw + px_i) * 4) as usize;
                            let (b, g, r, a) = (bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]);
                            if a == 0 {
                                continue;
                            }
                            let unpre = |c: u8| ((c as u32 * 255) / a as u32).min(255) as u8;
                            blend_pixel(
                                img.get_pixel_mut(tx as u32, ty as u32),
                                [unpre(r), unpre(g), unpre(b), a],
                            );
                        }
                    }
                } else {
                    if bytes.len() < (bw * bh) as usize {
                        continue;
                    }
                    for py in 0..bh {
                        let ty = dst_y0 + py;
                        if ty < min_y || ty >= max_y {
                            continue;
                        }
                        for px_i in 0..bw {
                            let tx = dst_x0 + px_i;
                            if tx < min_x || tx >= max_x {
                                continue;
                            }
                            let cov = bytes[(py * bw + px_i) as usize];
                            if cov == 0 {
                                continue;
                            }
                            let alpha = ((color[3] as u32 * cov as u32) + 127) / 255;
                            if alpha > 0 {
                                blend_pixel(
                                    img.get_pixel_mut(tx as u32, ty as u32),
                                    [color[0], color[1], color[2], alpha as u8],
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn truncate_to_width(
        &self,
        wts: &WindowTextSystem,
        text: &str,
        max_width: f32,
        font_px: f32,
        medium: bool,
    ) -> String {
        let measure_str = |s: &str| {
            let run = TextRun {
                len: s.len(),
                font: Self::font_for(medium),
                color: white(),
                background_color: None,
                underline: None,
                strikethrough: None,
            };
            f32::from(wts.shape_line(SharedString::from(s.to_string()), px(font_px), &[run], None).width)
        };
        if measure_str(text) <= max_width {
            return text.to_string();
        }
        let chars: Vec<char> = text.chars().collect();
        let mut lo = 0usize;
        let mut hi = chars.len();
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let mut cand: String = chars[..mid].iter().collect();
            cand.push('…');
            if measure_str(&cand) <= max_width {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo == 0 {
            if measure_str("…") <= max_width { "…".to_string() } else { String::new() }
        } else {
            let mut out: String = chars[..lo].iter().collect();
            out.push('…');
            out
        }
    }
}

fn text_width_px(text: &str, scale: u32) -> u32 {
    let n = text.chars().count() as u32;
    if n == 0 { 0 } else { n * 6 * scale - scale }
}

#[allow(clippy::too_many_arguments)]
fn draw_text_clipped_5x7(
    img: &mut RgbaImage,
    text: &str,
    x: i64,
    y: i64,
    max_width: u32,
    scale: u32,
    color: [u8; 4],
    clip: RectI,
) {
    let step = (6 * scale) as i64;
    let max_x = x.saturating_add(max_width as i64);
    let mut cx = x;
    for ch in text.chars() {
        if cx.saturating_add((5 * scale) as i64) > max_x {
            break;
        }
        let glyph = glyph_5x7(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                if (bits & (1 << (4 - col))) != 0 {
                    fill_rect_clipped(
                        img,
                        cx + (col as u32 * scale) as i64,
                        y + (row as u32 * scale) as i64,
                        scale as i64,
                        scale as i64,
                        color,
                        clip,
                    );
                }
            }
        }
        cx = cx.saturating_add(step);
    }
}

/// Crisp 5×7 bitmap fallback glyphs used only when no `TextSystem` is provided.
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' | '×' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08],
        '-' | '—' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '%' => [0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13],
        '·' => [0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00],
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        _ => [0x0E, 0x11, 0x02, 0x04, 0x04, 0x00, 0x04],
    }
}
