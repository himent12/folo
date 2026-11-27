//! Playground annotated PNG rendering and export file helpers (Single and Compare A/B modes).
//!
//! Renders boxes, halos, and crisp Unicode label chips onto an RGBA pixel buffer using the active
//! theme's class palette (`design/tokens.json`) and `folo-ui`'s `TextSystem`, then encodes to PNG.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use folo_ui::{Hsla, RenderImage, Rgba, TextSystem};
use image::{ImageFormat, RgbaImage};

use super::compare::ComparePaneCaption;
use super::export_text::{ExportText, RectI, fill_rect_clipped};
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
    pub header_meta: [u8; 4],
    pub inset_bg: [u8; 4],
    pub divider: [u8; 4],
}

impl ExportPalette {
    /// Snapshots all theme class colours (`tokens::CLASS_PALETTE`) into plain RGBA bytes safe to send off-thread.
    pub fn from_theme(t: &Theme) -> Self {
        let classes = (0..tokens::CLASS_PALETTE.len() as u32).map(|i| hsla_to_rgba(t.class(i))).collect();
        Self {
            classes,
            halo: u32_rgba(tokens::CLASS_HALO),
            chip_text: u32_rgba(tokens::CLASS_CHIP_TEXT),
            header_bg: hsla_to_rgba(t.bg_panel),
            header_text: hsla_to_rgba(t.text_primary),
            header_meta: hsla_to_rgba(t.text_secondary),
            inset_bg: hsla_to_rgba(t.bg_inset),
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
    let byte_len = (w as usize).checked_mul(h as usize)?.checked_mul(4)?;
    if src.len() < byte_len {
        return None;
    }
    let mut out = Vec::with_capacity(byte_len);
    for px in src[..byte_len].as_chunks::<4>().0 {
        out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    RgbaImage::from_raw(w, h, out)
}

fn sanitize_slug(raw: &str, strip_ext: bool) -> String {
    let tail = raw.rsplit(['/', '\\']).next().unwrap_or(raw).trim();
    let base = if strip_ext {
        tail.rsplit_once('.').map_or(tail, |(s, _)| if s.is_empty() { tail } else { s })
    } else {
        tail
    };
    let mut out = String::with_capacity(base.len());
    let mut prev_dash = false;
    for ch in base.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            out.push(lower);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    let mut truncated: String = out.chars().take(64).collect();
    while truncated.ends_with('-') {
        truncated.pop();
    }
    if truncated.is_empty() { "playground".to_string() } else { truncated }
}

/// Suggested filename for Single or Compare Playground export (`.png` or `.json`).
pub fn export_filename(image: &str, model_a: &str, model_b: Option<&str>, ext: &str) -> String {
    let img = sanitize_slug(image, true);
    let ma = sanitize_slug(model_a, false);
    match model_b {
        Some(b) => {
            let mb = sanitize_slug(b, false);
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
    text_system: Option<&Arc<TextSystem>>,
) -> anyhow::Result<Vec<u8>> {
    let mut canvas = base.clone();
    let (w, h) = canvas.dimensions();
    if show_boxes {
        let text_out = ExportText::new(text_system);
        draw_annotations(&mut canvas, 0, 0, w, h, boxes, show_labels, palette, &text_out);
    }
    encode_png(&canvas)
}

/// Renders a Compare A/B side-by-side annotated PNG (with header captions and divider) to bytes.
/// Narrow images are centered inside a minimum-width pane with a two-line header when needed so
/// model identity, detection counts, and timings stay legible.
#[allow(clippy::too_many_arguments)]
pub fn render_annotated_compare_png(
    base: &RgbaImage,
    caption_a: &ComparePaneCaption,
    boxes_a: &[ScoredBox],
    caption_b: &ComparePaneCaption,
    boxes_b: &[ScoredBox],
    show_boxes: bool,
    show_labels: bool,
    palette: &ExportPalette,
    text_system: Option<&Arc<TextSystem>>,
) -> anyhow::Result<Vec<u8>> {
    let (w, h) = base.dimensions();
    let scale = ((w.min(h) as f32) / 640.0).max(1.0).round() as u32;
    let gap = (4 * scale).max(2);
    let text_scale = scale.clamp(1, 4);
    let text_out = ExportText::new(text_system);
    let title_px = 12.0 * text_scale as f32;
    let meta_px = 11.0 * text_scale as f32;

    // Give narrow images a readable minimum pane width so headers keep model identity and metrics.
    let min_pane_w = 240u32.saturating_mul(scale);
    let pane_w = w.max(min_pane_w);
    let pad = 10u32.saturating_mul(scale).min(pane_w / 8);
    let avail_w = pane_w.saturating_sub(pad.saturating_mul(2));

    let single_a = caption_a.single_line();
    let single_b = caption_b.single_line();
    let fits_one_line = text_out.measure(&single_a, title_px, text_scale, true).0 <= avail_w as f32
        && text_out.measure(&single_b, title_px, text_scale, true).0 <= avail_w as f32;

    let head_h = if fits_one_line {
        28u32
            .saturating_mul(scale)
            .max(text_out.line_height(title_px, text_scale).saturating_add(12u32.saturating_mul(scale)))
    } else {
        let l1 = text_out.line_height(title_px, text_scale);
        let l2 = text_out.line_height(meta_px, text_scale);
        46u32.saturating_mul(scale).max(l1.saturating_add(l2).saturating_add(16u32.saturating_mul(scale)))
    };

    let total_w = pane_w.saturating_mul(2).saturating_add(gap);
    let total_h = h.saturating_add(head_h);
    let mut canvas = RgbaImage::from_pixel(total_w, total_h, image::Rgba(palette.inset_bg));
    fill_rect(&mut canvas, 0, 0, total_w as i64, head_h as i64, palette.header_bg);

    let pane_b_x = pane_w.saturating_add(gap);
    let img_ox_a = (pane_w - w) / 2;
    let img_ox_b = pane_b_x.saturating_add((pane_w - w) / 2);
    blit(&mut canvas, base, img_ox_a, head_h);
    blit(&mut canvas, base, img_ox_b, head_h);
    fill_rect(&mut canvas, pane_w as i64, 0, gap as i64, total_h as i64, palette.divider);

    if show_boxes {
        draw_annotations(&mut canvas, img_ox_a, head_h, w, h, boxes_a, show_labels, palette, &text_out);
        draw_annotations(&mut canvas, img_ox_b, head_h, w, h, boxes_b, show_labels, palette, &text_out);
    }

    let head_clip_a = RectI { x: 0, y: 0, w: pane_w as i64, h: head_h as i64 };
    let head_clip_b = RectI { x: pane_b_x as i64, y: 0, w: pane_w as i64, h: head_h as i64 };
    if fits_one_line {
        text_out.draw_clipped(
            &mut canvas,
            &single_a,
            pad as i64,
            0,
            head_h as i64,
            avail_w,
            title_px,
            text_scale,
            true,
            palette.header_text,
            head_clip_a,
        );
        text_out.draw_clipped(
            &mut canvas,
            &single_b,
            (pane_b_x + pad) as i64,
            0,
            head_h as i64,
            avail_w,
            title_px,
            text_scale,
            true,
            palette.header_text,
            head_clip_b,
        );
    } else {
        let top_h = (head_h as i64 * 52) / 100;
        let bot_y = top_h;
        let bot_h = head_h as i64 - top_h;
        for (px_x, cap, clip) in
            [(pad as i64, caption_a, head_clip_a), ((pane_b_x + pad) as i64, caption_b, head_clip_b)]
        {
            text_out.draw_clipped(
                &mut canvas,
                &cap.title,
                px_x,
                2 * scale as i64,
                top_h - 2 * scale as i64,
                avail_w,
                title_px,
                text_scale,
                true,
                palette.header_text,
                clip,
            );
            text_out.draw_clipped(
                &mut canvas,
                &cap.meta,
                px_x,
                bot_y,
                bot_h - 2 * scale as i64,
                avail_w,
                meta_px,
                text_scale,
                false,
                palette.header_meta,
                clip,
            );
        }
    }

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
    img.write_to(&mut buf, ImageFormat::Png)?;
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

/// Validates and clips a box to `[0, pane_w] × [0, pane_h]` in `f64` before converting to canvas
/// integer coordinates; returns `None` for non-finite, degenerate, or fully off-canvas boxes.
fn clip_box_to_pane(b: &ScoredBox, ox: u32, oy: u32, pane_w: u32, pane_h: u32) -> Option<RectI> {
    if !b.x1.is_finite() || !b.y1.is_finite() || !b.x2.is_finite() || !b.y2.is_finite() {
        return None;
    }
    if b.x2 <= b.x1 || b.y2 <= b.y1 || pane_w == 0 || pane_h == 0 {
        return None;
    }
    let pw = pane_w as f64;
    let ph = pane_h as f64;
    let x1 = (b.x1 as f64).clamp(0.0, pw);
    let y1 = (b.y1 as f64).clamp(0.0, ph);
    let x2 = (b.x2 as f64).clamp(0.0, pw);
    let y2 = (b.y2 as f64).clamp(0.0, ph);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let lx = ox as i64 + x1.round() as i64;
    let ty = oy as i64 + y1.round() as i64;
    let rx = ox as i64 + x2.round() as i64;
    let by = oy as i64 + y2.round() as i64;
    let bw = (rx - lx).max(1);
    let bh = (by - ty).max(1);
    Some(RectI { x: lx, y: ty, w: bw, h: bh })
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
    text_out: &ExportText,
) {
    let scale = ((pane_w.min(pane_h) as f32) / 640.0).max(1.0).round() as i64;
    let stroke = (2 * scale).max(2);
    let halo = stroke + (2 * scale).max(2);
    let clip = RectI { x: ox as i64, y: oy as i64, w: pane_w as i64, h: pane_h as i64 };

    let visible_boxes: Vec<(&ScoredBox, RectI)> =
        boxes.iter().filter_map(|b| clip_box_to_pane(b, ox, oy, pane_w, pane_h).map(|r| (b, r))).collect();

    // Pass 1: all visible box outlines (halo then class-colored stroke).
    let halo_pad = (halo - stroke) / 2;
    for &(b, r) in &visible_boxes {
        stroke_rect_clipped(
            canvas,
            r.x - halo_pad,
            r.y - halo_pad,
            r.w + halo_pad * 2,
            r.h + halo_pad * 2,
            halo,
            palette.halo,
            clip,
        );
        stroke_rect_clipped(canvas, r.x, r.y, r.w, r.h, stroke, palette.class_color(b.class), clip);
    }

    if !show_labels {
        return;
    }

    // Pass 2: label chips above/inside visible boxes with simple vertical collision avoidance.
    let text_scale = (scale as u32).clamp(1, 3);
    let font_px = 11.0 * text_scale as f32;
    let pad_x = (4 * text_scale) as i64;
    let pad_y = (3 * text_scale) as i64;
    let chip_h = text_out.line_height(font_px, text_scale) as i64 + pad_y * 2;
    let mut placed: Vec<RectI> = Vec::with_capacity(visible_boxes.len());

    for &(b, r) in &visible_boxes {
        let name = if b.name.trim().is_empty() { format!("class {}", b.class) } else { b.name.clone() };
        let safe_score = if b.score.is_finite() { b.score.clamp(0.0, 1.0) } else { 0.0 };
        let label = format!("{name} {safe_score:.2}");
        let tw = text_out.measure(&label, font_px, text_scale, true).0.ceil() as i64;
        let cw = (tw + pad_x * 2).min((pane_w as i64 - 4).max(8));

        let cx = r.x.clamp(clip.x + 2, (clip.x + clip.w - cw - 2).max(clip.x + 2));
        let mut cy = r.y - chip_h - 1;
        if cy < clip.y + 2 {
            cy = r.y + 2;
        }
        cy = cy.clamp(clip.y + 2, (clip.y + clip.h - chip_h - 2).max(clip.y + 2));

        for _ in 0..4 {
            let cand = RectI { x: cx, y: cy, w: cw, h: chip_h };
            if let Some(hit) = placed.iter().find(|p| p.intersects(cand, 2)) {
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
        text_out.draw_clipped(
            canvas,
            &label,
            cx + pad_x,
            cy,
            chip_h,
            avail,
            font_px,
            text_scale,
            true,
            palette.chip_text,
            clip,
        );
    }
}

fn fill_rect(img: &mut RgbaImage, x: i64, y: i64, w: i64, h: i64, color: [u8; 4]) {
    let clip = RectI { x: 0, y: 0, w: img.width() as i64, h: img.height() as i64 };
    fill_rect_clipped(img, x, y, w, h, color, clip);
}

#[allow(clippy::too_many_arguments)]
fn stroke_rect_clipped(
    img: &mut RgbaImage,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
    thickness: i64,
    color: [u8; 4],
    clip: RectI,
) {
    let t = thickness.max(1);
    fill_rect_clipped(img, x, y, w, t, color, clip);
    fill_rect_clipped(img, x, y.saturating_add(h).saturating_sub(t), w, t, color, clip);
    fill_rect_clipped(img, x, y.saturating_add(t), t, h.saturating_sub(2 * t).max(0), color, clip);
    fill_rect_clipped(
        img,
        x.saturating_add(w).saturating_sub(t),
        y.saturating_add(t),
        t,
        h.saturating_sub(2 * t).max(0),
        color,
        clip,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Mode;

    #[test]
    fn filenames_preserve_model_versions_and_strip_image_extensions() {
        assert_eq!(
            export_filename("shapes / val_001.jpg", "folo-n v1.2", None, "png"),
            "val-001-folo-n-v1-2.png"
        );
        assert_eq!(
            export_filename("cells.png", "model A", Some("model B v2.0"), "json"),
            "cells-model-a-vs-model-b-v2-0.json"
        );
        assert_eq!(sanitize_slug("///...", true), "playground");
    }

    #[test]
    fn single_and_compare_annotated_png_render_and_write_to_test_dir() {
        let dir = folo_test_support::TestDir::new("desktop-playground-export");
        let palette = ExportPalette::from_theme(&Theme::new(Mode::Dark));
        assert_eq!(palette.classes.len(), tokens::CLASS_PALETTE.len());

        let base = RgbaImage::from_pixel(160, 120, image::Rgba([24, 32, 48, 255]));
        let boxes_a = vec![
            ScoredBox { class: 0, name: "circle".into(), score: 0.94, x1: 20., y1: 25., x2: 70., y2: 85. },
            ScoredBox { class: 15, name: "square".into(), score: 0.81, x1: 80., y1: 30., x2: 140., y2: 100. },
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

        let single_bytes = render_annotated_single_png(&base, &boxes_a, true, true, &palette, None).unwrap();
        let single_path = dir.path().join("single.png");
        write_export_bytes(&single_path, &single_bytes).unwrap();
        let decoded_single = image::open(&single_path).unwrap().to_rgba8();
        assert_eq!(decoded_single.dimensions(), (160, 120));
        let changed_single = decoded_single.pixels().zip(base.pixels()).filter(|(a, b)| a != b).count();
        assert!(changed_single > 200, "expected boxes and label chips to paint pixels, got {changed_single}");

        let clean_bytes = render_annotated_single_png(&base, &boxes_a, false, true, &palette, None).unwrap();
        let clean_img = image::load_from_memory(&clean_bytes).unwrap().to_rgba8();
        assert_eq!(clean_img, base);

        let cap_a = ComparePaneCaption { title: "A · shapes-n".into(), meta: "2 dets · 12.0 ms".into() };
        let cap_b = ComparePaneCaption { title: "B · shapes-s".into(), meta: "1 det · 8.5 ms".into() };
        let cmp_bytes = render_annotated_compare_png(
            &base, &cap_a, &boxes_a, &cap_b, &boxes_b, true, true, &palette, None,
        )
        .unwrap();
        let cmp_path = dir.path().join("compare.png");
        write_export_bytes(&cmp_path, &cmp_bytes).unwrap();
        let decoded_cmp = image::open(&cmp_path).unwrap().to_rgba8();
        assert!(decoded_cmp.width() > 320);
        assert!(decoded_cmp.height() > 120);
    }

    fn test_native_text_system() -> Arc<TextSystem> {
        let app = folo_ui::HeadlessApp::new(folo_ui::HeadlessOptions {
            text: folo_ui::TextBackend::Native,
            ..Default::default()
        });
        let ts = app.read(|cx| cx.text_system().clone());
        let fonts = crate::FONTS.iter().map(|f| std::borrow::Cow::Borrowed(*f)).collect();
        ts.add_fonts(fonts).expect("load bundled Geist fonts");
        ts
    }

    #[test]
    fn narrow_compare_png_pads_panes_and_uses_two_line_headers() {
        let ts = test_native_text_system();
        let palette = ExportPalette::from_theme(&Theme::new(Mode::Dark));
        let narrow = RgbaImage::from_pixel(64, 480, image::Rgba([24, 32, 48, 255]));
        let cap_a =
            ComparePaneCaption { title: "A · shapes-hard-s-best".into(), meta: "2 dets · 12.0 ms".into() };
        let cap_b =
            ComparePaneCaption { title: "B · shapes-hard-n-best".into(), meta: "1 det · 8.5 ms".into() };
        for backend in [None, Some(&ts)] {
            let cmp_bytes = render_annotated_compare_png(
                &narrow,
                &cap_a,
                &[],
                &cap_b,
                &[],
                true,
                true,
                &palette,
                backend,
            )
            .unwrap();
            let decoded = image::load_from_memory(&cmp_bytes).unwrap().to_rgba8();
            assert!(
                decoded.width() >= 480,
                "narrow 64x480 compare PNG should widen panes to fit model identity + metrics, got {}",
                decoded.width()
            );
            assert!(
                decoded.height() >= 480 + 46,
                "narrow compare PNG should expand to a two-line header, got height {}",
                decoded.height()
            );
            let head_h = decoded.height() - 480;
            let row2_top = (head_h * 52) / 100;
            let left_meta_px = (0..240)
                .flat_map(|x| (row2_top..head_h).map(move |y| (x, y)))
                .filter(|&(x, y)| decoded.get_pixel(x, y).0 != palette.header_bg)
                .count();
            assert!(
                left_meta_px > 20,
                "expected second header line to render metrics, got {left_meta_px} px"
            );
        }
    }

    #[test]
    fn unicode_and_multiline_labels_and_captions_render_cleanly_via_text_system() {
        let ts = test_native_text_system();
        let palette = ExportPalette::from_theme(&Theme::new(Mode::Dark));
        let base = RgbaImage::from_pixel(220, 120, image::Rgba([24, 32, 48, 255]));
        let mk = |name: &str| {
            vec![ScoredBox { class: 0, name: name.into(), score: 0.91, x1: 20., y1: 35., x2: 180., y2: 100. }]
        };
        let cafe_png =
            render_annotated_single_png(&base, &mk("café"), true, true, &palette, Some(&ts)).unwrap();
        let qmark_png =
            render_annotated_single_png(&base, &mk("caf?"), true, true, &palette, Some(&ts)).unwrap();
        let cjk_png = render_annotated_single_png(&base, &mk("丸"), true, true, &palette, Some(&ts)).unwrap();
        let single_q_png =
            render_annotated_single_png(&base, &mk("?"), true, true, &palette, Some(&ts)).unwrap();
        let cafe = image::load_from_memory(&cafe_png).unwrap().to_rgba8();
        let qmark = image::load_from_memory(&qmark_png).unwrap().to_rgba8();
        let cjk = image::load_from_memory(&cjk_png).unwrap().to_rgba8();
        let single_q = image::load_from_memory(&single_q_png).unwrap().to_rgba8();
        assert_ne!(cafe, qmark, "accented Unicode label 'café' must not degrade to 'caf?'");
        assert_ne!(cjk, single_q, "CJK Unicode label '丸' must not degrade to '?'");

        // Compare captions with Unicode model names must also render distinct header glyphs.
        let cap_cafe =
            ComparePaneCaption { title: "A · café-model".into(), meta: "1 det · 4.0 ms".into() };
        let cap_q = ComparePaneCaption { title: "A · caf?-model".into(), meta: "1 det · 4.0 ms".into() };
        let cap_b = ComparePaneCaption { title: "B · 丸-v2".into(), meta: "1 det · 3.0 ms".into() };
        let cmp_cafe = image::load_from_memory(
            &render_annotated_compare_png(
                &base,
                &cap_cafe,
                &mk("café"),
                &cap_b,
                &mk("丸"),
                true,
                true,
                &palette,
                Some(&ts),
            )
            .unwrap(),
        )
        .unwrap()
        .to_rgba8();
        let cmp_q = image::load_from_memory(
            &render_annotated_compare_png(
                &base,
                &cap_q,
                &mk("café"),
                &cap_b,
                &mk("丸"),
                true,
                true,
                &palette,
                Some(&ts),
            )
            .unwrap(),
        )
        .unwrap()
        .to_rgba8();
        assert_ne!(cmp_cafe, cmp_q, "Unicode Compare caption 'café-model' must not degrade to 'caf?-model'");

        // Embedded CR/LF in class labels and Compare captions must be normalized without panicking.
        let nl_single = render_annotated_single_png(
            &base,
            &mk("traffic\nlight\r\nextra"),
            true,
            true,
            &palette,
            Some(&ts),
        )
        .unwrap();
        assert_eq!(image::load_from_memory(&nl_single).unwrap().to_rgba8().dimensions(), (220, 120));
        let cap_nl_a =
            ComparePaneCaption { title: "A · model\nline2".into(), meta: "1 det\r\n· 2.0 ms".into() };
        let cap_nl_b = ComparePaneCaption { title: "B · model\rB".into(), meta: "0 dets".into() };
        let nl_cmp = render_annotated_compare_png(
            &base,
            &cap_nl_a,
            &mk("sign\nstop"),
            &cap_nl_b,
            &[],
            true,
            true,
            &palette,
            Some(&ts),
        )
        .unwrap();
        assert!(image::load_from_memory(&nl_cmp).unwrap().to_rgba8().width() >= 480);
    }

    #[test]
    fn extreme_nan_degenerate_and_offscreen_boxes_never_panic_or_paint_orphaned_chips() {
        let palette = ExportPalette::from_theme(&Theme::new(Mode::Dark));
        let base = RgbaImage::from_pixel(120, 80, image::Rgba([20, 20, 20, 255]));
        // Only invalid/offscreen/degenerate boxes: output must stay pixel-identical to `base` (no orphaned chips).
        let invisible = vec![
            ScoredBox {
                class: 0,
                name: "nan".into(),
                score: f32::NAN,
                x1: f32::NAN,
                y1: 0.,
                x2: 50.,
                y2: 50.,
            },
            ScoredBox { class: 0, name: "deg".into(), score: 0.9, x1: 50., y1: 10., x2: 50., y2: 40. },
            ScoredBox { class: 0, name: "off".into(), score: 0.9, x1: 200., y1: 10., x2: 260., y2: 40. },
            ScoredBox { class: 0, name: "huge-off".into(), score: 0.9, x1: 1e20, y1: 0., x2: 2e20, y2: 50. },
        ];
        let out_bytes = render_annotated_single_png(&base, &invisible, true, true, &palette, None).unwrap();
        assert_eq!(image::load_from_memory(&out_bytes).unwrap().to_rgba8(), base);

        // Huge finite coordinates spanning the image with NaN score must clip safely without overflow panic.
        let huge_visible = vec![ScoredBox {
            class: 1,
            name: "huge".into(),
            score: f32::NAN,
            x1: -1e20,
            y1: -1e20,
            x2: 1e20,
            y2: 1e20,
        }];
        let cap_a = ComparePaneCaption { title: "A".into(), meta: "1 det".into() };
        let cap_b = ComparePaneCaption { title: "B".into(), meta: "1 det".into() };
        let cmp_bytes = render_annotated_compare_png(
            &base,
            &cap_a,
            &huge_visible,
            &cap_b,
            &huge_visible,
            true,
            true,
            &palette,
            None,
        )
        .unwrap();
        let decoded = image::load_from_memory(&cmp_bytes).unwrap().to_rgba8();
        assert!(decoded.width() > 240);
    }
}
