//! Playground annotated PNG rendering and export file helpers (Single and Compare A/B modes).
//!
//! Renders boxes, halos, and crisp label chips onto an RGBA pixel buffer using the active
//! theme's class palette (`design/tokens.json`), then encodes to PNG without extra dependencies.

use std::io::Cursor;
use std::path::Path;

use folo_ui::{Hsla, RenderImage, Rgba};
use image::{ImageFormat, RgbaImage};

use crate::api::ScoredBox;
use crate::theme::{Theme, tokens};

/// Colours derived from the active theme for off-thread annotated PNG rendering.
#[derive(Clone, Debug)]
pub struct ExportPalette {
    pub classes: Vec<[u8; 4]>,
    pub halo: [u8; 4],
    pub chip_text: [u8; 4],
    pub header_bg: [u8; 4],
    pub header_text: [u8; 4],
    pub divider: [u8; 4],
}

impl ExportPalette {
    /// Snapshots theme colours into plain RGBA bytes safe to send to a background thread.
    pub fn from_theme(t: &Theme) -> Self {
        let classes = (0..12).map(|i| hsla_to_rgba(t.class(i))).collect();
        Self {
            classes,
            halo: u32_rgba(tokens::CLASS_HALO),
            chip_text: u32_rgba(tokens::CLASS_CHIP_TEXT),
            header_bg: hsla_to_rgba(t.bg_panel),
            header_text: hsla_to_rgba(t.text_primary),
            divider: hsla_to_rgba(t.line_strong),
        }
    }

    pub fn class_color(&self, class: u32) -> [u8; 4] {
        if self.classes.is_empty() {
            return [0xC5, 0xF8, 0x2A, 0xFF];
        }
        self.classes[(class as usize) % self.classes.len()]
    }
}

fn hsla_to_rgba(c: Hsla) -> [u8; 4] {
    let r: Rgba = c.into();
    let ch = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    [ch(r.r), ch(r.g), ch(r.b), ch(r.a)]
}

fn u32_rgba(v: u32) -> [u8; 4] {
    [((v >> 24) & 0xFF) as u8, ((v >> 16) & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, (v & 0xFF) as u8]
}

/// Converts a `RenderImage` (BGRA frame 0) back to an RGBA `image::RgbaImage`.
pub fn render_image_to_rgba(img: &RenderImage) -> Option<RgbaImage> {
    let sz = img.size(0);
    let (w, h) = (sz.width.0.max(0) as u32, sz.height.0.max(0) as u32);
    if w == 0 || h == 0 {
        return None;
    }
    let src = img.as_bytes(0)?;
    if src.len() < (w as usize) * (h as usize) * 4 {
        return None;
    }
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for px in src[..(w as usize) * (h as usize) * 4].as_chunks::<4>().0 {
        out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    RgbaImage::from_raw(w, h, out)
}

/// Sanitizes an arbitrary image/model label into a safe lowercase filename stem.
pub fn sanitize_export_stem(raw: &str) -> String {
    // Strip path/dataset prefixes like "shapes / val_001.jpg" -> "val_001".
    let tail = raw.rsplit(['/', '\\']).next().unwrap_or(raw).trim();
    let stem = tail.rsplit_once('.').map_or(tail, |(s, _)| if s.is_empty() { tail } else { s });
    let mut out = String::with_capacity(stem.len());
    let mut prev_dash = false;
    for ch in stem.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            out.push(lower);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() { "playground".to_string() } else { out.chars().take(64).collect() }
}

/// Suggested filename for Single or Compare Playground export (`.png` or `.json`).
pub fn export_filename(image: &str, model_a: &str, model_b: Option<&str>, ext: &str) -> String {
    let img = sanitize_export_stem(image);
    let ma = sanitize_export_stem(model_a);
    match model_b {
        Some(b) => {
            let mb = sanitize_export_stem(b);
            format!("{img}-{ma}-vs-{mb}.{ext}")
        }
        None => format!("{img}-{ma}.{ext}"),
    }
}

/// Renders a Single-model annotated PNG to bytes.
pub fn render_annotated_single_png(
    base: &RgbaImage,
    boxes: &[ScoredBox],
    show_boxes: bool,
    show_labels: bool,
    palette: &ExportPalette,
) -> anyhow::Result<Vec<u8>> {
    let mut canvas = base.clone();
    let (w, h) = canvas.dimensions();
    if show_boxes {
        draw_annotations(&mut canvas, 0, 0, w, h, boxes, show_labels, palette);
    }
    encode_png(&canvas)
}

/// Renders a Compare A/B side-by-side annotated PNG (with dark header captions and divider) to bytes.
#[allow(clippy::too_many_arguments)]
pub fn render_annotated_compare_png(
    base: &RgbaImage,
    caption_a: &str,
    boxes_a: &[ScoredBox],
    caption_b: &str,
    boxes_b: &[ScoredBox],
    show_boxes: bool,
    show_labels: bool,
    palette: &ExportPalette,
) -> anyhow::Result<Vec<u8>> {
    let (w, h) = base.dimensions();
    let scale = ((w.min(h) as f32) / 640.0).max(1.0).round() as u32;
    let gap = (4 * scale).max(2);
    let text_scale = scale.clamp(1, 4);
    let head_h = (28 * scale).max(text_scale * 7 + 12);
    let total_w = w.saturating_mul(2).saturating_add(gap);
    let total_h = h.saturating_add(head_h);
    let mut canvas = RgbaImage::from_pixel(total_w, total_h, image::Rgba(palette.header_bg));

    // Copy base image into left (Model A) and right (Model B) panes.
    blit(&mut canvas, base, 0, head_h);
    blit(&mut canvas, base, w + gap, head_h);

    // Draw subtle vertical divider between panes.
    fill_rect(&mut canvas, w as i32, 0, gap as i32, total_h as i32, palette.divider);

    if show_boxes {
        draw_annotations(&mut canvas, 0, head_h, w, h, boxes_a, show_labels, palette);
        draw_annotations(&mut canvas, w + gap, head_h, w, h, boxes_b, show_labels, palette);
    }

    // Header captions clipped to each pane width.
    let pad = (10 * scale) as i32;
    let ty = ((head_h.saturating_sub(7 * text_scale)) / 2) as i32;
    draw_text_clipped(
        &mut canvas,
        caption_a,
        pad,
        ty,
        w.saturating_sub(20 * scale),
        text_scale,
        palette.header_text,
    );
    draw_text_clipped(
        &mut canvas,
        caption_b,
        (w + gap) as i32 + pad,
        ty,
        w.saturating_sub(20 * scale),
        text_scale,
        palette.header_text,
    );

    encode_png(&canvas)
}

/// Writes `bytes` to `path` cleanly.
pub fn write_export_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)
}

fn encode_png(img: &RgbaImage) -> anyhow::Result<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img.clone()).write_to(&mut buf, ImageFormat::Png)?;
    Ok(buf.into_inner())
}

fn blit(dst: &mut RgbaImage, src: &RgbaImage, ox: u32, oy: u32) {
    let (dw, dh) = dst.dimensions();
    let (sw, sh) = src.dimensions();
    let copy_w = sw.min(dw.saturating_sub(ox));
    let copy_h = sh.min(dh.saturating_sub(oy));
    for y in 0..copy_h {
        for x in 0..copy_w {
            dst.put_pixel(ox + x, oy + y, *src.get_pixel(x, y));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_annotations(
    canvas: &mut RgbaImage,
    ox: u32,
    oy: u32,
    pane_w: u32,
    pane_h: u32,
    boxes: &[ScoredBox],
    show_labels: bool,
    palette: &ExportPalette,
) {
    let scale = ((pane_w.min(pane_h) as f32) / 640.0).max(1.0).round() as i32;
    let stroke = (2 * scale).max(2);
    let halo = stroke + (2 * scale).max(2);
    let clip = RectI { x: ox as i32, y: oy as i32, w: pane_w as i32, h: pane_h as i32 };

    // Pass 1: all box outlines (halo then class-colored stroke).
    for b in boxes {
        if !b.x1.is_finite() || !b.y1.is_finite() || !b.x2.is_finite() || !b.y2.is_finite() {
            continue;
        }
        let x1 = ox as i32 + b.x1.round() as i32;
        let y1 = oy as i32 + b.y1.round() as i32;
        let x2 = ox as i32 + b.x2.round() as i32;
        let y2 = oy as i32 + b.y2.round() as i32;
        let (lx, rx) = (x1.min(x2), x1.max(x2));
        let (ty, by) = (y1.min(y2), y1.max(y2));
        let bw = (rx - lx).max(1);
        let bh = (by - ty).max(1);
        let halo_pad = (halo - stroke) / 2;
        stroke_rect_clipped(
            canvas,
            lx - halo_pad,
            ty - halo_pad,
            bw + halo_pad * 2,
            bh + halo_pad * 2,
            halo,
            palette.halo,
            clip,
        );
        stroke_rect_clipped(canvas, lx, ty, bw, bh, stroke, palette.class_color(b.class), clip);
    }

    if !show_labels {
        return;
    }

    // Pass 2: label chips above/inside boxes with simple vertical collision avoidance.
    let text_scale = (scale as u32).clamp(1, 3);
    let pad_x = (4 * text_scale) as i32;
    let pad_y = (3 * text_scale) as i32;
    let chip_h = (7 * text_scale) as i32 + pad_y * 2;
    let mut placed: Vec<RectI> = Vec::with_capacity(boxes.len());

    for b in boxes {
        if !b.x1.is_finite() || !b.y1.is_finite() || !b.x2.is_finite() || !b.y2.is_finite() {
            continue;
        }
        let name = if b.name.trim().is_empty() { format!("class {}", b.class) } else { b.name.clone() };
        let label = format!("{name} {:.2}", b.score.clamp(0.0, 1.0));
        let tw = text_width_px(&label, text_scale) as i32;
        let cw = (tw + pad_x * 2).min((pane_w as i32 - 4).max(8));

        let bx1 = ox as i32 + b.x1.min(b.x2).round() as i32;
        let by1 = oy as i32 + b.y1.min(b.y2).round() as i32;
        let cx = bx1.clamp(clip.x + 2, (clip.x + clip.w - cw - 2).max(clip.x + 2));
        let mut cy = by1 - chip_h - 1;
        if cy < clip.y + 2 {
            cy = by1 + 2;
        }
        cy = cy.clamp(clip.y + 2, (clip.y + clip.h - chip_h - 2).max(clip.y + 2));

        for _ in 0..4 {
            let cand = RectI { x: cx, y: cy, w: cw, h: chip_h };
            if let Some(hit) = placed.iter().find(|r| r.intersects(cand, 2)) {
                let next_y = hit.y + hit.h + 2;
                if next_y + chip_h <= clip.y + clip.h - 2 {
                    cy = next_y;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        let rect = RectI { x: cx, y: cy, w: cw, h: chip_h };
        placed.push(rect);

        fill_rect_clipped(canvas, cx - 1, cy - 1, cw + 2, chip_h + 2, palette.halo, clip);
        fill_rect_clipped(canvas, cx, cy, cw, chip_h, palette.class_color(b.class), clip);
        let avail = (cw - pad_x * 2).max(0) as u32;
        draw_text_clipped(canvas, &label, cx + pad_x, cy + pad_y, avail, text_scale, palette.chip_text);
    }
}

#[derive(Clone, Copy)]
struct RectI {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

impl RectI {
    fn intersects(self, other: RectI, pad: i32) -> bool {
        self.x < other.x + other.w + pad
            && self.x + self.w + pad > other.x
            && self.y < other.y + other.h + pad
            && self.y + self.h + pad > other.y
    }
}

fn blend_pixel(dst: &mut image::Rgba<u8>, src: [u8; 4]) {
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

fn fill_rect(img: &mut RgbaImage, x: i32, y: i32, w: i32, h: i32, color: [u8; 4]) {
    let clip = RectI { x: 0, y: 0, w: img.width() as i32, h: img.height() as i32 };
    fill_rect_clipped(img, x, y, w, h, color, clip);
}

fn fill_rect_clipped(img: &mut RgbaImage, x: i32, y: i32, w: i32, h: i32, color: [u8; 4], clip: RectI) {
    if w <= 0 || h <= 0 {
        return;
    }
    let x0 = x.max(clip.x).max(0);
    let y0 = y.max(clip.y).max(0);
    let x1 = (x + w).min(clip.x + clip.w).min(img.width() as i32);
    let y1 = (y + h).min(clip.y + clip.h).min(img.height() as i32);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    for py in y0..y1 {
        for px in x0..x1 {
            blend_pixel(img.get_pixel_mut(px as u32, py as u32), color);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stroke_rect_clipped(
    img: &mut RgbaImage,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    thickness: i32,
    color: [u8; 4],
    clip: RectI,
) {
    let t = thickness.max(1);
    fill_rect_clipped(img, x, y, w, t, color, clip);
    fill_rect_clipped(img, x, y + h - t, w, t, color, clip);
    fill_rect_clipped(img, x, y + t, t, (h - 2 * t).max(0), color, clip);
    fill_rect_clipped(img, x + w - t, y + t, t, (h - 2 * t).max(0), color, clip);
}

fn text_width_px(text: &str, scale: u32) -> u32 {
    let n = text.chars().count() as u32;
    if n == 0 { 0 } else { n * 6 * scale - scale }
}

fn draw_text_clipped(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    max_width: u32,
    scale: u32,
    color: [u8; 4],
) {
    let step = (6 * scale) as i32;
    let max_x = x + max_width as i32;
    let mut cx = x;
    let clip = RectI { x: 0, y: 0, w: img.width() as i32, h: img.height() as i32 };
    for ch in text.chars() {
        if cx + (5 * scale) as i32 > max_x {
            break;
        }
        let glyph = glyph_5x7(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                if (bits & (1 << (4 - col))) != 0 {
                    fill_rect_clipped(
                        img,
                        cx + (col as u32 * scale) as i32,
                        y + (row as u32 * scale) as i32,
                        scale as i32,
                        scale as i32,
                        color,
                        clip,
                    );
                }
            }
        }
        cx += step;
    }
}

/// Crisp 5×7 bitmap glyphs for ASCII characters (plus `·` and `×`) used in exported PNG labels/captions.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Mode;

    #[test]
    fn filenames_sanitize_dataset_prefixes_and_compare_pairs() {
        assert_eq!(
            export_filename("shapes / val_001.jpg", "shapes-n best", None, "png"),
            "val-001-shapes-n-best.png"
        );
        assert_eq!(
            export_filename("cells.png", "model A", Some("model B"), "json"),
            "cells-model-a-vs-model-b.json"
        );
        assert_eq!(sanitize_export_stem("///..."), "playground");
    }

    #[test]
    fn single_and_compare_annotated_png_render_and_write_to_test_dir() {
        let dir = folo_test_support::TestDir::new("desktop-playground-export");
        let palette = ExportPalette::from_theme(&Theme::new(Mode::Dark));
        let base = RgbaImage::from_pixel(160, 120, image::Rgba([24, 32, 48, 255]));
        let boxes_a = vec![
            ScoredBox { class: 0, name: "circle".into(), score: 0.94, x1: 20., y1: 25., x2: 70., y2: 85. },
            ScoredBox { class: 1, name: "square".into(), score: 0.81, x1: 80., y1: 30., x2: 140., y2: 100. },
        ];
        let boxes_b = vec![ScoredBox {
            class: 0,
            name: "circle".into(),
            score: 0.97,
            x1: 22.,
            y1: 24.,
            x2: 72.,
            y2: 86.,
        }];

        let single_bytes = render_annotated_single_png(&base, &boxes_a, true, true, &palette).unwrap();
        let single_path = dir.path().join("single.png");
        write_export_bytes(&single_path, &single_bytes).unwrap();
        let decoded_single = image::open(&single_path).unwrap().to_rgba8();
        assert_eq!(decoded_single.dimensions(), (160, 120));
        // Bounding box + label chips visibly altered pixels on the base canvas.
        let changed_single = decoded_single.pixels().zip(base.pixels()).filter(|(a, b)| a != b).count();
        assert!(changed_single > 200, "expected boxes and label chips to paint pixels, got {changed_single}");

        // Boxes hidden -> output matches base photo verbatim.
        let clean_bytes = render_annotated_single_png(&base, &boxes_a, false, true, &palette).unwrap();
        let clean_img = image::load_from_memory(&clean_bytes).unwrap().to_rgba8();
        assert_eq!(clean_img, base);

        // Compare side-by-side composite.
        let cmp_bytes = render_annotated_compare_png(
            &base,
            "A · shapes-n · 2 dets · 12.0 ms",
            &boxes_a,
            "B · shapes-s · 1 det · 8.5 ms",
            &boxes_b,
            true,
            true,
            &palette,
        )
        .unwrap();
        let cmp_path = dir.path().join("compare.png");
        write_export_bytes(&cmp_path, &cmp_bytes).unwrap();
        let decoded_cmp = image::open(&cmp_path).unwrap().to_rgba8();
        assert!(decoded_cmp.width() > 320, "side-by-side composite spans both panes + gap");
        assert!(decoded_cmp.height() > 120, "composite includes header caption strip");
    }
}
