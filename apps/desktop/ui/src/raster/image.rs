// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/shaders.hlsl (polychrome_sprite_fragment)
//! Images: resampled once to their destination size (linear filtering at texel centres, as the GPU's
//! sampler; optional grayscale with GPUI's luminance weights), cached as premultiplied pixels, then
//! blitted with opacity and rounded-corner coverage.

use super::blend::{ONE, alpha16, over_pm8, sat};
use super::geom::IRect;
use super::quad_eval::quad_sdf;
use super::sprite::Axis;
use super::target::Target;
use crate::RenderImage;

/// What a resampled copy depends on.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ImageKey {
    pub id: u64,
    pub frame: usize,
    pub grayscale: bool,
    /// Destination size in pixels, and the bounds' offset and size as `f32` bits.
    pub w: i32,
    pub h: i32,
    pub bits: [u32; 4],
}

/// An image at its destination size: `0xAARRGGBB` premultiplied, rows top-down.
pub(crate) struct ImageCopy {
    pub w: usize,
    pub px: Vec<u32>,
    /// Every pixel has alpha 255.
    pub opaque: bool,
}

impl ImageCopy {
    pub fn bytes(&self) -> usize {
        self.px.len() * 4 + 64
    }

    /// Resamples `image`'s frame `frame` (BGRA, straight alpha) to `w` x `h`, the destination's
    /// `(x, y)` offset of the bounds and their size given as `dst`.
    pub fn new(image: &RenderImage, frame: usize, grayscale: bool, w: i32, h: i32, dst: [f32; 4]) -> Self {
        let (w, h) = (w.max(0) as usize, h.max(0) as usize);
        let Some(bytes) = image.as_bytes(frame) else {
            return ImageCopy { w, px: vec![0; w * h], opaque: false };
        };
        let size = image.size(frame);
        let (iw, ih) = (size.width.0.max(0) as usize, size.height.0.max(0) as usize);
        if iw == 0 || ih == 0 {
            return ImageCopy { w, px: vec![0; w * h], opaque: false };
        }
        let ax = Axis::new(dst[0], dst[2], iw as i32);
        let ay = Axis::new(dst[1], dst[3], ih as i32);
        let cols: Vec<_> = (0..w as i32).map(|i| ax.at(i)).collect();
        let mut px = Vec::with_capacity(w * h);
        let mut opaque = true;
        const GRAY: [f32; 3] = [0.2126, 0.7152, 0.0722];
        for j in 0..h as i32 {
            let (y0, y1, fy) = ay.at(j);
            let r0 = &bytes[y0 * iw * 4..(y0 + 1) * iw * 4];
            let r1 = &bytes[y1 * iw * 4..(y1 + 1) * iw * 4];
            for &(x0, x1, fx) in &cols {
                // Channels in BGRA order, each `value * 65536`.
                let s: [u32; 4] = std::array::from_fn(|c| {
                    let top = r0[x0 * 4 + c] as u32 * (256 - fx) + r0[x1 * 4 + c] as u32 * fx;
                    let bot = r1[x0 * 4 + c] as u32 * (256 - fx) + r1[x1 * 4 + c] as u32 * fx;
                    top * (256 - fy) + bot * fy
                });
                let k = 1. / (255. * 65536.);
                let (mut r, mut g, mut b) = (s[2] as f32 * k, s[1] as f32 * k, s[0] as f32 * k);
                let a = s[3] as f32 * k;
                if grayscale {
                    let l = r * GRAY[0] + g * GRAY[1] + b * GRAY[2];
                    (r, g, b) = (l, l, l);
                }
                let q = |v: f32| (sat(v) * 255. + 0.5) as u32;
                let a8 = q(a);
                opaque &= a8 == 255;
                px.push(a8 << 24 | q(r * a) << 16 | q(g * a) << 8 | q(b * a));
            }
        }
        ImageCopy { w, px, opaque }
    }
}

/// How an image is blitted: its copy, where the copy's (0, 0) lands, opacity, and the rounded corners.
pub(crate) struct ImageDraw {
    pub copy: std::sync::Arc<ImageCopy>,
    pub origin: (i32, i32),
    pub opacity: u32,
    pub bounds_origin: [f32; 2],
    pub bounds_size: [f32; 2],
    /// top-left, top-right, bottom-right, bottom-left.
    pub radii: [f32; 4],
}

impl ImageDraw {
    pub fn new(copy: std::sync::Arc<ImageCopy>, origin: (i32, i32), opacity: f32, img: &crate::Image) -> Self {
        let r = &img.corner_radii;
        ImageDraw {
            copy,
            origin,
            opacity: alpha16(opacity),
            bounds_origin: [img.bounds.origin.x.0, img.bounds.origin.y.0],
            bounds_size: [img.bounds.size.width.0, img.bounds.size.height.0],
            radii: [r.top_left.0, r.top_right.0, r.bottom_right.0, r.bottom_left.0],
        }
    }

    /// Whether pixel (`x`, `y`) lies in a rounded corner's box, where coverage needs the SDF.
    fn in_corner(&self, x: i32, y: i32) -> bool {
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        let [ox, oy] = self.bounds_origin;
        let [w, h] = self.bounds_size;
        let left = px < ox + w / 2.;
        let top = py < oy + h / 2.;
        let r = match (left, top) {
            (true, true) => self.radii[0],
            (false, true) => self.radii[1],
            (false, false) => self.radii[2],
            (true, false) => self.radii[3],
        };
        if r <= 0. {
            return false;
        }
        let dx = if left { px - ox } else { ox + w - px };
        let dy = if top { py - oy } else { oy + h - py };
        dx < r + 1. && dy < r + 1.
    }

    pub fn draw(&self, t: &mut Target, r: IRect) {
        let rounded = self.radii.iter().any(|&v| v > 0.);
        let x0 = (r.x0 - self.origin.0) as usize;
        let n = (r.x1 - r.x0) as usize;
        for y in r.y0..r.y1 {
            let start = (y - self.origin.1) as usize * self.copy.w + x0;
            let src = &self.copy.px[start..start + n];
            let row = t.row(y, r.x0, r.x1);
            if self.opacity >= ONE && self.copy.opaque && !rounded {
                for (p, &s) in row.iter_mut().zip(src) {
                    *p = s & 0x00ff_ffff;
                }
                continue;
            }
            for ((x, p), &s) in (r.x0..r.x1).zip(row.iter_mut()).zip(src) {
                let a = s >> 24;
                if a == 0 {
                    continue;
                }
                let m = if rounded && self.in_corner(x, y) {
                    let d = quad_sdf([x as f32 + 0.5, y as f32 + 0.5], self.bounds_origin, self.bounds_size, &self.radii);
                    ((self.opacity as f32) * sat(0.5 - d) + 0.5) as u32
                } else {
                    self.opacity
                };
                if m != 0 {
                    *p = over_pm8(*p, a, (s >> 16) & 0xff, (s >> 8) & 0xff, s & 0xff, m);
                }
            }
        }
    }
}
