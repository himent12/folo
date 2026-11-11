//! `text_probe`: lays out lines with the DirectWrite backend and prints, as JSON lines, each line's
//! width, ascent, descent and glyph positions; writes each line's rasterized coverage to a PNG. The
//! orchestrator compares both with GPUI captures.
//!
//! ```text
//! cargo run -p folo-ui --example text_probe -- <lines.txt> <out-dir> [scale]
//! ```
//!
//! `lines.txt` holds one `family|weight|size|features|text` per line (`features`: `-` for none, or
//! `tnum` / `tnum=1,calt=0`; blank lines and lines starting with `#` are skipped). The six bundled Geist
//! fonts are added first. Each PNG (`line-<n>.png`, 8-bit gray) is the glyph masks at `scale`
//! (default 1.0), placed as GPUI's `paint_glyph` places them (device origin floored, subpixel variant
//! from its fraction, plus the raster bounds' origin) and composited source-over as raw coverage, with
//! no contrast or gamma correction; emoji contribute their alpha. The baseline is at `ceil(ascent)`
//! device pixels plus a 2 px margin.

#[cfg(windows)]
fn main() {
    if let Err(err) = probe::run() {
        eprintln!("text_probe: {err:#}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("text_probe needs the DirectWrite backend (Windows)");
    std::process::exit(2);
}

#[cfg(windows)]
mod probe {
    use anyhow::{Context as _, Result, bail};
    use folo_ui::text::dwrite::DirectWriteTextSystem;
    use folo_ui::{
        Font, FontFeatures, FontRun, FontWeight, LineLayout, PlatformTextSystem, Point, RenderGlyphParams,
        SUBPIXEL_VARIANTS_X, SUBPIXEL_VARIANTS_Y, font, px,
    };
    use std::borrow::Cow;
    use std::fmt::Write as _;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const FONTS: [&[u8]; 6] = [
        include_bytes!("../../../../design/fonts/Geist-Regular.ttf"),
        include_bytes!("../../../../design/fonts/Geist-Medium.ttf"),
        include_bytes!("../../../../design/fonts/Geist-SemiBold.ttf"),
        include_bytes!("../../../../design/fonts/GeistMono-Regular.ttf"),
        include_bytes!("../../../../design/fonts/GeistMono-Medium.ttf"),
        include_bytes!("../../../../design/fonts/GeistMono-SemiBold.ttf"),
    ];

    /// Device pixels of margin around each PNG's text.
    const MARGIN: i32 = 2;

    struct Spec {
        family: String,
        weight: f32,
        size: f32,
        features: Vec<(String, u32)>,
        text: String,
    }

    pub(super) fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let [lines, out_dir, rest @ ..] = args.as_slice() else {
            bail!("usage: text_probe <lines.txt> <out-dir> [scale]");
        };
        let scale: f32 = match rest {
            [] => 1.0,
            [scale] => scale.parse().context("scale")?,
            _ => bail!("usage: text_probe <lines.txt> <out-dir> [scale]"),
        };
        let out_dir = PathBuf::from(out_dir);
        std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
        let source = std::fs::read_to_string(lines).with_context(|| format!("reading {lines}"))?;

        let ts = DirectWriteTextSystem::new()?;
        ts.add_fonts(FONTS.iter().map(|f| Cow::Borrowed(*f)).collect())?;

        let mut n = 0;
        for (line_no, line) in source.lines().enumerate() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            n += 1;
            let spec = parse(line).with_context(|| format!("{lines}:{}", line_no + 1))?;
            let font = Font {
                weight: FontWeight(spec.weight),
                features: FontFeatures(Arc::new(spec.features.clone())),
                ..font(spec.family.clone())
            };
            let font_id = ts.font_id(&font)?;
            let layout =
                ts.layout_line(&spec.text, px(spec.size), &[FontRun { len: spec.text.len(), font_id }]);
            let png = out_dir.join(format!("line-{n}.png"));
            rasterize(&ts, &layout, scale, &png)?;
            println!("{}", json(n, &spec, &layout, scale, &png));
        }
        Ok(())
    }

    fn parse(line: &str) -> Result<Spec> {
        let mut parts = line.splitn(5, '|');
        let (Some(family), Some(weight), Some(size), Some(features), Some(text)) =
            (parts.next(), parts.next(), parts.next(), parts.next(), parts.next())
        else {
            bail!("expected family|weight|size|features|text");
        };
        let features = match features.trim() {
            "" | "-" => Vec::new(),
            list => list
                .split(',')
                .map(|feature| match feature.split_once('=') {
                    Some((tag, value)) => Ok((tag.trim().to_string(), value.trim().parse()?)),
                    None => Ok((feature.trim().to_string(), 1)),
                })
                .collect::<Result<_>>()?,
        };
        Ok(Spec {
            family: family.trim().to_string(),
            weight: weight.trim().parse().context("weight")?,
            size: size.trim().parse().context("size")?,
            features,
            text: text.to_string(),
        })
    }

    /// The masks of every glyph, placed as the window places them, composited into a gray PNG.
    fn rasterize(ts: &DirectWriteTextSystem, layout: &LineLayout, scale: f32, png: &Path) -> Result<()> {
        let ascent = (f32::from(layout.ascent) * scale).ceil() as i32;
        let descent = (f32::from(layout.descent) * scale).ceil() as i32;
        let width = (f32::from(layout.width) * scale).ceil() as i32 + 2 * MARGIN;
        let height = ascent + descent + 2 * MARGIN;
        let mut image = image::GrayImage::new(width.max(1) as u32, height.max(1) as u32);
        // The baseline origin in logical pixels (the window's `glyph_origin + baseline_offset`).
        let origin_x = MARGIN as f32 / scale;
        let baseline_y = (MARGIN + ascent) as f32 / scale;
        for run in &layout.runs {
            for glyph in &run.glyphs {
                let device =
                    Point { x: (origin_x + f32::from(glyph.position.x)) * scale, y: baseline_y * scale };
                let subpixel_variant = if glyph.is_emoji {
                    Point::default()
                } else {
                    Point {
                        x: (device.x.fract() * SUBPIXEL_VARIANTS_X as f32).floor() as u8,
                        y: (device.y.fract() * SUBPIXEL_VARIANTS_Y as f32).floor() as u8,
                    }
                };
                let params = RenderGlyphParams {
                    font_id: run.font_id,
                    glyph_id: glyph.id,
                    font_size: layout.font_size,
                    subpixel_variant,
                    scale_factor: scale,
                    is_emoji: glyph.is_emoji,
                };
                let bounds = ts.glyph_raster_bounds(&params)?;
                if bounds.size.width.0 <= 0 || bounds.size.height.0 <= 0 {
                    continue;
                }
                let (size, data) = ts.rasterize_glyph(&params, bounds)?;
                let left = device.x.floor() as i32 + bounds.origin.x.0;
                let top = device.y.floor() as i32 + bounds.origin.y.0;
                let bytes_per_pixel = if glyph.is_emoji { 4 } else { 1 };
                for y in 0..size.height.0 {
                    for x in 0..size.width.0 {
                        let (px, py) = (left + x, top + y);
                        if px < 0 || py < 0 || px >= width || py >= height {
                            continue;
                        }
                        let i = ((y * size.width.0 + x) as usize + 1) * bytes_per_pixel - 1;
                        let a = data[i] as f32 / 255.0;
                        let dst = image.get_pixel_mut(px as u32, py as u32);
                        let d = dst.0[0] as f32 / 255.0;
                        dst.0[0] = ((a + d * (1.0 - a)) * 255.0).round() as u8;
                    }
                }
            }
        }
        image.save(png).with_context(|| format!("writing {}", png.display()))?;
        Ok(())
    }

    fn json(n: usize, spec: &Spec, layout: &LineLayout, scale: f32, png: &Path) -> String {
        let mut out = String::new();
        let features: Vec<String> = spec.features.iter().map(|(t, v)| format!("{t}={v}")).collect();
        let _ = write!(
            out,
            "{{\"line\":{n},\"family\":{},\"weight\":{},\"size\":{},\"features\":{},\"text\":{},\"scale\":{scale},\
             \"width\":{},\"ascent\":{},\"descent\":{},\"len\":{},\"runs\":[",
            quote(&spec.family),
            spec.weight,
            spec.size,
            quote(&features.join(",")),
            quote(&spec.text),
            f32::from(layout.width),
            f32::from(layout.ascent),
            f32::from(layout.descent),
            layout.len,
        );
        for (r, run) in layout.runs.iter().enumerate() {
            let _ =
                write!(out, "{}{{\"font_id\":{},\"glyphs\":[", if r > 0 { "," } else { "" }, run.font_id.0);
            for (g, glyph) in run.glyphs.iter().enumerate() {
                let _ = write!(
                    out,
                    "{}{{\"id\":{},\"x\":{},\"y\":{},\"index\":{},\"emoji\":{}}}",
                    if g > 0 { "," } else { "" },
                    glyph.id.0,
                    f32::from(glyph.position.x),
                    f32::from(glyph.position.y),
                    glyph.index,
                    glyph.is_emoji,
                );
            }
            out.push_str("]}");
        }
        let _ = write!(out, "],\"png\":{}}}", quote(&png.display().to_string()));
        out
    }

    /// A JSON string literal.
    fn quote(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }
}
