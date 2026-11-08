//! A minimal RGB rasteriser for mock images: gradients, value noise, and anti-aliased filled shapes.
//!
//! Shapes are given in *scene* coordinates and drawn through a uniform scale, so a thumbnail is rendered directly at
//! its size instead of being downsampled from the full image.

use crate::util::mix64;

/// An RGB8 image with a scene-to-pixel scale.
pub struct Canvas {
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
    /// Scene units to pixels.
    pub scale: f32,
    /// Row-major RGB.
    pub px: Vec<u8>,
}

/// An RGB colour.
pub type Rgb = [u8; 3];

impl Canvas {
    /// A black canvas of `w x h` pixels drawing scene coordinates times `scale`.
    pub fn new(w: u32, h: u32, scale: f32) -> Canvas {
        Canvas { w, h, scale, px: vec![0; w as usize * h as usize * 3] }
    }

    /// Fills with a vertical gradient from `top` to `bottom` (pixel rows `y0..y1`), plus value noise of amplitude
    /// `noise_amp` (0..255) and feature size `noise_cell` scene units.
    #[allow(clippy::too_many_arguments)]
    pub fn gradient(
        &mut self,
        y0: f32,
        y1: f32,
        top: Rgb,
        bottom: Rgb,
        noise_amp: f32,
        noise_cell: f32,
        seed: u64,
    ) {
        let (w, s) = (self.w as usize, self.scale);
        let py0 = (y0 * s).max(0.0) as usize;
        let py1 = ((y1 * s).ceil() as usize).min(self.h as usize);
        let cell = (noise_cell * s).max(1.0);
        let span = (py1 - py0).max(1) as f32;
        for y in py0..py1 {
            let t = (y - py0) as f32 / span;
            let base = [0, 1, 2].map(|c| top[c] as f32 + (bottom[c] as f32 - top[c] as f32) * t);
            let row = &mut self.px[y * w * 3..(y + 1) * w * 3];
            for x in 0..w {
                let n = if noise_amp > 0.0 {
                    let (fx, fy) = (x as f32 / cell, y as f32 / cell);
                    0.65 * value_noise(fx, fy, seed) + 0.35 * value_noise(fx * 3.7, fy * 3.7, seed ^ 0x9e37)
                } else {
                    0.0
                };
                let d = n * noise_amp;
                for c in 0..3 {
                    row[x * 3 + c] = (base[c] + d).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    /// Blends `color` into pixel `(x, y)` with coverage `a` in `0..=1`.
    #[inline]
    fn blend(&mut self, x: usize, y: usize, color: Rgb, a: f32) {
        let i = (y * self.w as usize + x) * 3;
        for (p, &c) in self.px[i..i + 3].iter_mut().zip(&color) {
            let d = *p as f32;
            *p = (d + (c as f32 - d) * a + 0.5) as u8;
        }
    }

    /// Pixel bounds `[x0, x1) x [y0, y1)` of a scene rectangle, clamped to the canvas.
    fn bounds(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> (usize, usize, usize, usize) {
        let s = self.scale;
        let lo = |v: f32, m: u32| ((v * s).floor().max(0.0) as usize).min(m as usize);
        let hi = |v: f32, m: u32| (((v * s).ceil() + 1.0).max(0.0) as usize).min(m as usize);
        (lo(x1.min(x2), self.w), lo(y1.min(y2), self.h), hi(x1.max(x2), self.w), hi(y1.max(y2), self.h))
    }

    /// Fills an axis-aligned rectangle (anti-aliased edges).
    pub fn rect(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, color: Rgb) {
        let s = self.scale;
        let (px1, py1, px2, py2) = (x1 * s, y1 * s, x2 * s, y2 * s);
        let (bx0, by0, bx1, by1) = self.bounds(x1 - 1.0, y1 - 1.0, x2 + 1.0, y2 + 1.0);
        for y in by0..by1 {
            let fy = y as f32 + 0.5;
            let cy = ((fy + 0.5).min(py2) - (fy - 0.5).max(py1)).clamp(0.0, 1.0);
            if cy <= 0.0 {
                continue;
            }
            for x in bx0..bx1 {
                let fx = x as f32 + 0.5;
                let cx = ((fx + 0.5).min(px2) - (fx - 0.5).max(px1)).clamp(0.0, 1.0);
                if cx > 0.0 {
                    self.blend(x, y, color, cx * cy);
                }
            }
        }
    }

    /// Fills an ellipse centred at `(cx, cy)` with radii `rx`, `ry` (scene units).
    pub fn ellipse(&mut self, cx: f32, cy: f32, rx: f32, ry: f32, color: Rgb) {
        let s = self.scale;
        let (pcx, pcy, prx, pry) = (cx * s, cy * s, (rx * s).max(0.5), (ry * s).max(0.5));
        let (bx0, by0, bx1, by1) = self.bounds(cx - rx - 1.0, cy - ry - 1.0, cx + rx + 1.0, cy + ry + 1.0);
        let r = prx.min(pry);
        for y in by0..by1 {
            let dy = (y as f32 + 0.5 - pcy) / pry;
            for x in bx0..bx1 {
                let dx = (x as f32 + 0.5 - pcx) / prx;
                // Signed distance approximation in pixels.
                let d = ((dx * dx + dy * dy).sqrt() - 1.0) * r;
                let a = (0.5 - d).clamp(0.0, 1.0);
                if a > 0.0 {
                    self.blend(x, y, color, a);
                }
            }
        }
    }

    /// Fills a polygon (even-odd rule) with 4x vertical supersampling for smooth edges.
    pub fn polygon(&mut self, pts: &[(f32, f32)], color: Rgb) {
        if pts.len() < 3 {
            return;
        }
        let s = self.scale;
        let p: Vec<(f32, f32)> = pts.iter().map(|&(x, y)| (x * s, y * s)).collect();
        let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in pts {
            minx = minx.min(x);
            miny = miny.min(y);
            maxx = maxx.max(x);
            maxy = maxy.max(y);
        }
        let (bx0, by0, bx1, by1) = self.bounds(minx, miny, maxx, maxy);
        if bx1 <= bx0 {
            return;
        }
        const SUB: usize = 4;
        let mut cover = vec![0f32; bx1 - bx0];
        let mut xs: Vec<f32> = Vec::with_capacity(8);
        for y in by0..by1 {
            cover.iter_mut().for_each(|c| *c = 0.0);
            for sub in 0..SUB {
                let fy = y as f32 + (sub as f32 + 0.5) / SUB as f32;
                xs.clear();
                for i in 0..p.len() {
                    let (a, b) = (p[i], p[(i + 1) % p.len()]);
                    if (a.1 <= fy) != (b.1 <= fy) {
                        xs.push(a.0 + (fy - a.1) / (b.1 - a.1) * (b.0 - a.0));
                    }
                }
                xs.sort_by(f32::total_cmp);
                for &[a, b] in xs.as_chunks::<2>().0 {
                    let (l, r) = (a.max(bx0 as f32), b.min(bx1 as f32));
                    if r <= l {
                        continue;
                    }
                    let (li, ri) = (l.floor() as usize, (r.ceil() as usize).min(bx1));
                    for x in li..ri {
                        let c = ((x + 1) as f32).min(r) - (x as f32).max(l);
                        cover[x - bx0] += c.max(0.0) / SUB as f32;
                    }
                }
            }
            for (i, &c) in cover.iter().enumerate() {
                if c > 0.0 {
                    self.blend(bx0 + i, y, color, c.min(1.0));
                }
            }
        }
    }

    /// A thick line segment.
    pub fn line(&mut self, a: (f32, f32), b: (f32, f32), width: f32, color: Rgb) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let (nx, ny) = (-dy / len * width * 0.5, dx / len * width * 0.5);
        self.polygon(
            &[(a.0 + nx, a.1 + ny), (b.0 + nx, b.1 + ny), (b.0 - nx, b.1 - ny), (a.0 - nx, a.1 - ny)],
            color,
        );
    }

    /// Encodes as JPEG.
    pub fn to_jpeg(&self, quality: u8) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.px.len() / 8);
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
            .encode(&self.px, self.w, self.h, image::ExtendedColorType::Rgb8)
            .expect("encoding an in-memory RGB buffer cannot fail");
        out
    }
}

/// Smooth value noise in `[-1, 1]` at lattice coordinates `(x, y)`.
fn value_noise(x: f32, y: f32, seed: u64) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (x - xi, y - yi);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let corner = |i: f32, j: f32| {
        let h = mix64(seed ^ mix64(((i as i64 as u64) << 32) ^ (j as i64 as u64 & 0xffff_ffff)));
        (h >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let top = corner(xi, yi) + (corner(xi + 1.0, yi) - corner(xi, yi)) * sx;
    let bot = corner(xi, yi + 1.0) + (corner(xi + 1.0, yi + 1.0) - corner(xi, yi + 1.0)) * sx;
    top + (bot - top) * sy
}

/// Converts HSV (`h` in degrees, `s`, `v` in 0..1) to RGB.
pub fn hsv(h: f32, s: f32, v: f32) -> Rgb {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r, g, b].map(|u| ((u + m) * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// Multiplies a colour's brightness by `f`.
pub fn shade(c: Rgb, f: f32) -> Rgb {
    c.map(|u| (u as f32 * f).clamp(0.0, 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(c: &Canvas, x: u32, y: u32) -> Rgb {
        let i = ((y * c.w + x) * 3) as usize;
        [c.px[i], c.px[i + 1], c.px[i + 2]]
    }

    #[test]
    fn shapes_cover_expected_pixels() {
        let mut c = Canvas::new(100, 100, 1.0);
        c.rect(10.0, 10.0, 30.0, 30.0, [255, 0, 0]);
        c.ellipse(70.0, 70.0, 10.0, 10.0, [0, 255, 0]);
        c.polygon(&[(50.0, 5.0), (60.0, 25.0), (40.0, 25.0)], [0, 0, 255]);
        assert_eq!(at(&c, 20, 20), [255, 0, 0]);
        assert_eq!(at(&c, 5, 5), [0, 0, 0]);
        assert_eq!(at(&c, 70, 70), [0, 255, 0]);
        assert_eq!(at(&c, 85, 85), [0, 0, 0]);
        assert_eq!(at(&c, 50, 20), [0, 0, 255]);
        assert_eq!(at(&c, 41, 6), [0, 0, 0]);
    }

    #[test]
    fn scaled_drawing_and_jpeg() {
        let mut c = Canvas::new(50, 50, 0.5);
        c.gradient(0.0, 100.0, [10, 20, 30], [200, 210, 220], 8.0, 10.0, 1);
        c.rect(20.0, 20.0, 60.0, 60.0, [255, 255, 255]);
        assert_eq!(at(&c, 20, 20), [255, 255, 255]);
        let jpg = c.to_jpeg(80);
        assert_eq!(&jpg[..2], &[0xff, 0xd8]);
        assert_eq!(image::load_from_memory(&jpg).unwrap().width(), 50);
    }
}
