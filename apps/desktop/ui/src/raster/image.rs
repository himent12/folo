// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/shaders.hlsl (polychrome_sprite_fragment)
//! Images: resampled once to their destination size (linear filtering at texel centres, as the GPU's
//! sampler reads GPUI's atlas; optional grayscale with GPUI's luminance weights), cached as
//! premultiplied pixels, then blitted with opacity and rounded-corner coverage.

use super::blend::{ONE, alpha16, over_pm8, sat};
use super::geom::{IRect, ceil_i32, floor_i32};
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
        let cols: Vec<_> = (0..w as i32).map(|i| atlas_texels(&ax, i, iw)).collect();
        let mut px = Vec::with_capacity(w * h);
        let mut opaque = true;
        const GRAY: [f32; 3] = [0.2126, 0.7152, 0.0722];
        let row = |y: Option<usize>| y.map(|y| &bytes[y * iw * 4..(y + 1) * iw * 4]);
        for j in 0..h as i32 {
            let (y0, y1, fy) = atlas_texels(&ay, j, ih);
            let (r0, r1) = (row(y0), row(y1));
            for &(x0, x1, fx) in &cols {
                // A texel outside the image is transparent black (`None`).
                let texel = |r: Option<&[u8]>, x: Option<usize>, c: usize| match (r, x) {
                    (Some(r), Some(x)) => r[x * 4 + c] as u32,
                    _ => 0,
                };
                // Channels in BGRA order, each `value * 65536`.
                let s: [u32; 4] = std::array::from_fn(|c| {
                    let top = texel(r0, x0, c) * (256 - fx) + texel(r0, x1, c) * fx;
                    let bot = texel(r1, x0, c) * (256 - fx) + texel(r1, x1, c) * fx;
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

/// The smallest side of a GPUI atlas texture (`DEFAULT_ATLAS_SIZE`).
const ATLAS_MIN: usize = 1024;

/// The two texels destination pixel `i` blends and the second one's weight, as GPUI's sampler sees
/// them: the image is a tile of a zero-initialised atlas with no padding, sampled linearly with
/// WRAP addressing, so a texel past the image's edge is transparent black (`None`), except along a
/// side of at least `ATLAS_MIN` pixels, where the image spans its own texture and the sampler wraps
/// around to its opposite edge. Magnified images get a darker rim from it, as in the GPUI captures.
/// (A neighbouring tile in a shared atlas would be read instead of black: not modelled.)
#[inline]
fn atlas_texels(axis: &Axis, i: i32, n: usize) -> (Option<usize>, Option<usize>, u32) {
    let (t0, f) = axis.raw(i);
    let at = |t: i64| {
        if (0..n as i64).contains(&t) {
            Some(t as usize)
        } else if n >= ATLAS_MIN {
            Some(t.rem_euclid(n as i64) as usize)
        } else {
            None
        }
    };
    (at(t0), at(t0 + 1), f)
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
    pub fn new(
        copy: std::sync::Arc<ImageCopy>,
        origin: (i32, i32),
        opacity: f32,
        img: &crate::Image,
    ) -> Self {
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

    /// The columns `a..b` of row `y` whose coverage is 1 (outside the rounded corners' boxes); the
    /// pixels left and right of them need the SDF.
    fn plain_columns(&self, y: i32, r: &IRect) -> (i32, i32) {
        let [ox, oy] = self.bounds_origin;
        let [w, h] = self.bounds_size;
        let py = y as f32 + 0.5;
        let top = py < oy + h / 2.;
        let (rl, rr) = if top { (self.radii[0], self.radii[1]) } else { (self.radii[3], self.radii[2]) };
        let dy = if top { py - oy } else { oy + h - py };
        let mut a = r.x0;
        let mut b = r.x1;
        if rl > 0. && dy < rl + 1. {
            a = ceil_i32(ox + rl + 0.5).clamp(r.x0, r.x1);
        }
        if rr > 0. && dy < rr + 1. {
            b = (floor_i32(ox + w - rr - 1.5) + 1).clamp(a, r.x1);
        }
        (a, b)
    }

    pub fn draw(&self, t: &mut Target, r: IRect) {
        let rounded = self.radii.iter().any(|&v| v > 0.);
        let x0 = (r.x0 - self.origin.0) as usize;
        let n = (r.x1 - r.x0) as usize;
        let copy = self.opacity >= ONE && self.copy.opaque;
        for y in r.y0..r.y1 {
            let start = (y - self.origin.1) as usize * self.copy.w + x0;
            let src = &self.copy.px[start..start + n];
            let row = t.row(y, r.x0, r.x1);
            let (a, b) = if rounded { self.plain_columns(y, &r) } else { (r.x0, r.x1) };
            let (ia, ib) = ((a - r.x0) as usize, (b - r.x0) as usize);
            if copy {
                for (p, &s) in row[ia..ib].iter_mut().zip(&src[ia..ib]) {
                    *p = s & 0x00ff_ffff;
                }
            } else {
                for (p, &s) in row[ia..ib].iter_mut().zip(&src[ia..ib]) {
                    let al = s >> 24;
                    if al != 0 {
                        *p = over_pm8(*p, al, (s >> 16) & 0xff, (s >> 8) & 0xff, s & 0xff, self.opacity);
                    }
                }
            }
            for i in (0..ia).chain(ib..n) {
                let s = src[i];
                let al = s >> 24;
                if al == 0 {
                    continue;
                }
                let x = r.x0 + i as i32;
                let d = quad_sdf(
                    [x as f32 + 0.5, y as f32 + 0.5],
                    self.bounds_origin,
                    self.bounds_size,
                    &self.radii,
                );
                let m = ((self.opacity as f32) * sat(0.5 - d) + 0.5) as u32;
                if m != 0 {
                    row[i] = over_pm8(row[i], al, (s >> 16) & 0xff, (s >> 8) & 0xff, s & 0xff, m);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, bgra: [u8; 4]) -> RenderImage {
        let buf = ::image::RgbaImage::from_fn(w, h, |_, _| ::image::Rgba(bgra));
        RenderImage::new([::image::Frame::new(buf)])
    }

    /// A magnified image's edge pixels blend with the transparent texels around its atlas tile, as
    /// GPUI's sampler does: the rim gets `c * (1 - f)` at alpha `1 - f`; the inside stays solid.
    #[test]
    fn magnified_edges_blend_with_the_empty_atlas() {
        // 4 texels over 8 pixels: pixel 0 samples texel coordinate -0.25, a quarter from texel -1.
        let img = solid(4, 4, [200, 100, 50, 255]);
        let copy = ImageCopy::new(&img, 0, false, 8, 8, [0., 0., 8., 8.]);
        assert!(!copy.opaque);
        let (corner, edge, inside) = (copy.px[0], copy.px[3], copy.px[3 * 8 + 3]);
        assert_eq!(inside, 0xff_32_64_c8);
        // Edge (row 0): alpha 0.75, premultiplied colour times 0.75 * 0.75.
        assert_eq!(edge >> 24, 191);
        assert_eq!(edge & 0xff, (200. * 0.75 * 0.75 + 0.5) as u32);
        // Corner: 0.75 on both axes.
        assert_eq!(corner >> 24, (255. * 0.5625 + 0.5) as u32);
        // Drawn at its own size, every pixel samples a texel centre: nothing from outside.
        assert!(ImageCopy::new(&img, 0, false, 4, 4, [0., 0., 4., 4.]).opaque);
    }

    /// A side of 1024 or more fills its own atlas texture, so WRAP reads the opposite edge.
    #[test]
    fn wide_images_wrap() {
        let ax = Axis::new(0., 2048., 1024);
        assert_eq!(atlas_texels(&ax, 0, 1024), (Some(1023), Some(0), 192));
        let ax = Axis::new(0., 1000., 500);
        assert_eq!(atlas_texels(&ax, 0, 500), (None, Some(0), 192));
        assert_eq!(atlas_texels(&ax, 999, 500), (Some(499), None, 64));
    }
}
