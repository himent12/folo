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
        if iw == 0 || ih == 0 || bytes.len() < iw.saturating_mul(ih).saturating_mul(4) {
            return ImageCopy { w, px: vec![0; w * h], opaque: false };
        }
        // Fast 1:1 aligned path: destination covers the exact source image at integer origin with
        // no scaling or grayscale conversion, so every pixel samples a single texel centre.
        if !grayscale
            && w == iw
            && h == ih
            && iw <= (1 << 24)
            && ih <= (1 << 24)
            && dst == [0., 0., iw as f32, ih as f32]
        {
            return Self::copy_one_to_one(bytes, w, h);
        }
        let ax = Axis::new(dst[0], dst[2], iw as i32);
        let ay = Axis::new(dst[1], dst[3], ih as i32);
        let cols: Vec<_> = (0..w as i32).map(|i| atlas_texels(&ax, i, iw)).collect();
        let mut px = Vec::with_capacity(w * h);
        let mut opaque = true;
        let row = |y: Option<usize>| y.map(|y| &bytes[y * iw * 4..(y + 1) * iw * 4]);
        for j in 0..h as i32 {
            let (y0, y1, fy) = atlas_texels(&ay, j, ih);
            let (r0, r1) = (row(y0), row(y1));
            let inv_fy = 256 - fy;
            if !grayscale && let (Some(r0), Some(r1)) = (r0, r1) {
                for &(x0, x1, fx) in &cols {
                    if let (Some(x0), Some(x1)) = (x0, x1) {
                        let p00 = &r0[x0 * 4..x0 * 4 + 4];
                        let p01 = &r0[x1 * 4..x1 * 4 + 4];
                        let p10 = &r1[x0 * 4..x0 * 4 + 4];
                        let p11 = &r1[x1 * 4..x1 * 4 + 4];
                        if (p00[3] & p01[3] & p10[3] & p11[3]) == 255 {
                            let inv_fx = 256 - fx;
                            let w00 = inv_fx * inv_fy;
                            let w01 = fx * inv_fy;
                            let w10 = inv_fx * fy;
                            let w11 = fx * fy;
                            let b = (p00[0] as u32 * w00
                                + p01[0] as u32 * w01
                                + p10[0] as u32 * w10
                                + p11[0] as u32 * w11
                                + 32768)
                                >> 16;
                            let g = (p00[1] as u32 * w00
                                + p01[1] as u32 * w01
                                + p10[1] as u32 * w10
                                + p11[1] as u32 * w11
                                + 32768)
                                >> 16;
                            let r = (p00[2] as u32 * w00
                                + p01[2] as u32 * w01
                                + p10[2] as u32 * w10
                                + p11[2] as u32 * w11
                                + 32768)
                                >> 16;
                            px.push(0xff00_0000 | r << 16 | g << 8 | b);
                            continue;
                        }
                    }
                    let (pixel, is_opaque) = sample_general(r0, r1, x0, x1, fx, fy, false);
                    opaque &= is_opaque;
                    px.push(pixel);
                }
                continue;
            }
            for &(x0, x1, fx) in &cols {
                let (pixel, is_opaque) = sample_general_opt(r0, r1, x0, x1, fx, fy, grayscale);
                opaque &= is_opaque;
                px.push(pixel);
            }
        }
        ImageCopy { w, px, opaque }
    }

    /// Packs 1:1 BGRA straight-alpha source bytes into premultiplied `0xAARRGGBB` without bilinear
    /// filtering or floating-point math.
    fn copy_one_to_one(bytes: &[u8], w: usize, h: usize) -> Self {
        let count = w * h;
        let mut px = Vec::with_capacity(count);
        let mut opaque = true;
        for bgra in bytes[..count * 4].as_chunks::<4>().0 {
            let a = bgra[3] as u32;
            if a == 255 {
                px.push(0xff00_0000 | (bgra[2] as u32) << 16 | (bgra[1] as u32) << 8 | bgra[0] as u32);
            } else if a == 0 {
                opaque = false;
                px.push(0);
            } else {
                opaque = false;
                let r = (bgra[2] as u32 * a + 127) / 255;
                let g = (bgra[1] as u32 * a + 127) / 255;
                let b = (bgra[0] as u32 * a + 127) / 255;
                px.push(a << 24 | r << 16 | g << 8 | b);
            }
        }
        ImageCopy { w, px, opaque }
    }
}

const GRAY: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// General bilinear sample when both rows are in-bounds (handles partial alpha and out-of-bounds columns).
#[inline]
fn sample_general(
    r0: &[u8],
    r1: &[u8],
    x0: Option<usize>,
    x1: Option<usize>,
    fx: u32,
    fy: u32,
    grayscale: bool,
) -> (u32, bool) {
    sample_general_opt(Some(r0), Some(r1), x0, x1, fx, fy, grayscale)
}

/// General bilinear sample with optional rows/columns (out-of-bounds texels are transparent black),
/// optional grayscale conversion, and float premultiplication.
#[inline]
fn sample_general_opt(
    r0: Option<&[u8]>,
    r1: Option<&[u8]>,
    x0: Option<usize>,
    x1: Option<usize>,
    fx: u32,
    fy: u32,
    grayscale: bool,
) -> (u32, bool) {
    let texel = |r: Option<&[u8]>, x: Option<usize>, c: usize| match (r, x) {
        (Some(r), Some(x)) => r[x * 4 + c] as u32,
        _ => 0,
    };
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
    (a8 << 24 | q(r * a) << 16 | q(g * a) << 8 | q(b * a), a8 == 255)
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
                copy_rgb_span(&mut row[ia..ib], &src[ia..ib]);
            } else if self.opacity >= ONE {
                for (p, &s) in row[ia..ib].iter_mut().zip(&src[ia..ib]) {
                    let al = s >> 24;
                    if al == 255 {
                        *p = s & 0x00ff_ffff;
                    } else if al != 0 {
                        *p = over_pm8(*p, al, (s >> 16) & 0xff, (s >> 8) & 0xff, s & 0xff, ONE);
                    }
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

/// Copies `0xAARRGGBB` pixels from `src` into `dst` as `0x00RRGGBB`, masking off the alpha byte.
#[inline]
fn copy_rgb_span(dst: &mut [u32], src: &[u32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= 4 {
        let (dst_chunks, dst_tail) = dst.as_chunks_mut::<4>();
        let (src_chunks, src_tail) = src.as_chunks::<4>();
        // SAFETY: SSE2 is guaranteed on all x86_64 targets; chunk slices have equal length.
        unsafe { copy_rgb_span_sse2(dst_chunks, src_chunks) };
        for (p, &s) in dst_tail.iter_mut().zip(src_tail) {
            *p = s & 0x00ff_ffff;
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if dst.len() >= 4 {
        let (dst_chunks, dst_tail) = dst.as_chunks_mut::<4>();
        let (src_chunks, src_tail) = src.as_chunks::<4>();
        // SAFETY: NEON is guaranteed on all aarch64 targets; chunk slices have equal length.
        unsafe { copy_rgb_span_neon(dst_chunks, src_chunks) };
        for (p, &s) in dst_tail.iter_mut().zip(src_tail) {
            *p = s & 0x00ff_ffff;
        }
        return;
    }
    for (p, &s) in dst.iter_mut().zip(src) {
        *p = s & 0x00ff_ffff;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn copy_rgb_span_sse2(dst: &mut [[u32; 4]], src: &[[u32; 4]]) {
    use std::arch::x86_64::*;

    let rgb_mask = _mm_set1_epi32(0x00ff_ffff);
    for (d, s) in dst.iter_mut().zip(src) {
        // SAFETY: `s` and `d` are 16-byte `[u32; 4]` arrays valid for unaligned 128-bit load/store.
        let v = unsafe { _mm_loadu_si128(s.as_ptr().cast::<__m128i>()) };
        unsafe { _mm_storeu_si128(d.as_mut_ptr().cast::<__m128i>(), _mm_and_si128(v, rgb_mask)) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn copy_rgb_span_neon(dst: &mut [[u32; 4]], src: &[[u32; 4]]) {
    use std::arch::aarch64::*;

    let rgb_mask = vdupq_n_u32(0x00ff_ffff);
    for (d, s) in dst.iter_mut().zip(src) {
        // SAFETY: `s` and `d` are 16-byte `[u32; 4]` arrays valid for 128-bit load/store.
        let v = unsafe { vld1q_u32(s.as_ptr()) };
        unsafe { vst1q_u32(d.as_mut_ptr(), vandq_u32(v, rgb_mask)) };
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

    #[test]
    fn one_to_one_premultiply_matches_float_formula_exhaustively() {
        for a in 0..=255u32 {
            let af = a as f32 / 255.;
            for c in 0..=255u32 {
                let want = (sat((c as f32 / 255.) * af) * 255. + 0.5) as u32;
                let got = if a == 255 {
                    c
                } else if a == 0 {
                    0
                } else {
                    (c * a + 127) / 255
                };
                assert_eq!(got, want, "c={c} a={a}");
            }
        }
    }

    #[test]
    fn fast_image_copy_matches_general_sampler_across_modes() {
        let mut buf = ::image::RgbaImage::new(13, 11);
        for (x, y, p) in buf.enumerate_pixels_mut() {
            let b = ((x * 23 + y * 7) % 256) as u8;
            let g = ((x * 11 + y * 29) % 256) as u8;
            let r = ((x * 41 + y * 13) % 256) as u8;
            let a = if (x + y) % 5 == 0 { ((x * 37) % 256) as u8 } else { 255 };
            *p = ::image::Rgba([b, g, r, a]);
        }
        let mixed = RenderImage::new([::image::Frame::new(buf.clone())]);
        for p in buf.pixels_mut() {
            p.0[3] = 255;
        }
        let opaque_img = RenderImage::new([::image::Frame::new(buf)]);

        let reference = |img: &RenderImage, grayscale: bool, w: i32, h: i32, dst: [f32; 4]| {
            let bytes = img.as_bytes(0).unwrap();
            let size = img.size(0);
            let (iw, ih) = (size.width.0 as usize, size.height.0 as usize);
            let ax = Axis::new(dst[0], dst[2], iw as i32);
            let ay = Axis::new(dst[1], dst[3], ih as i32);
            let cols: Vec<_> = (0..w).map(|i| atlas_texels(&ax, i, iw)).collect();
            let row = |y: Option<usize>| y.map(|y| &bytes[y * iw * 4..(y + 1) * iw * 4]);
            let mut px = Vec::with_capacity((w * h) as usize);
            let mut opaque = true;
            for j in 0..h {
                let (y0, y1, fy) = atlas_texels(&ay, j, ih);
                let (r0, r1) = (row(y0), row(y1));
                for &(x0, x1, fx) in &cols {
                    let (pixel, is_opaque) = sample_general_opt(r0, r1, x0, x1, fx, fy, grayscale);
                    opaque &= is_opaque;
                    px.push(pixel);
                }
            }
            (px, opaque)
        };

        for (img, label) in [(&opaque_img, "opaque"), (&mixed, "mixed")] {
            for grayscale in [false, true] {
                for (w, h, dst) in [
                    (13, 11, [0., 0., 13., 11.]),    // 1:1 exact fast path
                    (9, 7, [0., 0., 9., 7.]),        // downscale (opaque integer bilinear path)
                    (21, 17, [0., 0., 21., 17.]),    // upscale (atlas edge blending)
                    (10, 8, [-1.5, 0.25, 13., 11.]), // fractional offset
                ] {
                    let got = ImageCopy::new(img, 0, grayscale, w, h, dst);
                    let (want_px, want_opaque) = reference(img, grayscale, w, h, dst);
                    assert_eq!(got.opaque, want_opaque, "{label} grayscale={grayscale} {w}x{h} {dst:?}");
                    assert_eq!(got.px.len(), want_px.len());
                    for (idx, (&g, &wn)) in got.px.iter().zip(&want_px).enumerate() {
                        // 1:1 and grayscale are bit-exact; integer bilinear avoids f32 ULP drift on
                        // exact .5 ties and stays within 1/255 of the float formula everywhere.
                        if w == 13 && h == 11 && dst == [0., 0., 13., 11.] || grayscale {
                            assert_eq!(g, wn, "{label} grayscale={grayscale} idx={idx}");
                        } else {
                            assert_eq!(g >> 24, wn >> 24, "{label} alpha idx={idx}");
                            for shift in [16, 8, 0] {
                                let gc = (g >> shift) & 0xff;
                                let wc = (wn >> shift) & 0xff;
                                assert!(
                                    gc.abs_diff(wc) <= 1,
                                    "{label} {w}x{h} idx={idx} shift={shift}: {gc} vs {wc}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn copy_rgb_span_matches_scalar_masking_for_all_lengths() {
        for len in 0..=19 {
            let src: Vec<u32> = (0..len)
                .map(|i| {
                    0xff00_0000u32.wrapping_sub((i as u32) << 24) | ((i as u32 * 0x0013_579b) & 0x00ff_ffff)
                })
                .collect();
            let mut got = vec![0xdead_beefu32; len];
            copy_rgb_span(&mut got, &src);
            let want: Vec<u32> = src.iter().map(|&s| s & 0x00ff_ffff).collect();
            assert_eq!(got, want, "len={len}");
        }
    }
}
