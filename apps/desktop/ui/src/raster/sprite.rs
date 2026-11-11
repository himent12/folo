// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/alpha_correction.hlsl
// (apply_contrast_and_gamma_correction and its helpers)
//! Monochrome sprites (glyphs and SVG icons) and colour sprites (emoji).
//!
//! A glyph's (or an icon's) corrected alpha depends only on its mask byte, the text colour and the `GlyphBlend`,
//! so it is a 256-entry table per (colour, blend), and the blit is "look up, skip zero, blend".

use super::blend::{ONE, Tint, alpha16, over_pm8, sat};
use super::geom::IRect;
use super::target::Target;
use crate::{AlphaMask, ColorMask, GlyphBlend};

fn color_brightness(c: [f32; 3]) -> f32 {
    c[0] * 0.30 + c[1] * 0.59 + c[2] * 0.11
}

fn light_on_dark_contrast(enhanced_contrast: f32, c: [f32; 3]) -> f32 {
    let multiplier = sat(4.0 * (0.75 - color_brightness(c)));
    enhanced_contrast * multiplier
}

fn enhance_contrast(alpha: f32, k: f32) -> f32 {
    alpha * (k + 1.0) / (alpha * k + 1.0)
}

fn apply_alpha_correction(a: f32, b: f32, g: [f32; 4]) -> f32 {
    let brightness_adjustment = g[0] * b + g[1];
    let correction = brightness_adjustment * a + (g[2] * b + g[3]);
    a + a * (1.0 - a) * correction
}

/// `apply_contrast_and_gamma_correction` from GPUI's `alpha_correction.hlsl`.
pub(crate) fn apply_contrast_and_gamma_correction(
    sample: f32,
    c: [f32; 3],
    contrast: f32,
    gamma: [f32; 4],
) -> f32 {
    let enhanced = light_on_dark_contrast(contrast, c);
    let brightness = color_brightness(c);
    apply_alpha_correction(enhance_contrast(sample, enhanced), brightness, gamma)
}

/// The final alpha a glyph pixel of mask value `m` gets, in float (the shader's maths): the
/// corrected sample (or the plain one) times the colour's alpha.
pub(crate) fn glyph_alpha_f32(m: u8, c: [f32; 4], blend: &GlyphBlend) -> f32 {
    let s = m as f32 / 255.;
    let a = if blend.apply {
        apply_contrast_and_gamma_correction(
            s,
            [c[0], c[1], c[2]],
            blend.grayscale_enhanced_contrast,
            blend.gamma_ratios,
        )
    } else {
        s
    };
    sat(c[3] * a)
}

/// Alpha (`0..=ONE`) for each mask byte.
pub(crate) struct AlphaLut(pub [u32; 256]);

impl AlphaLut {
    /// The glyph table for colour `c` under `blend`.
    pub fn glyph(c: [f32; 4], blend: &GlyphBlend) -> Self {
        AlphaLut(std::array::from_fn(|m| alpha16(glyph_alpha_f32(m as u8, c, blend))))
    }

    /// Plain coverage times `alpha` (paths).
    pub fn coverage(alpha: f32) -> Self {
        AlphaLut(std::array::from_fn(|m| alpha16(m as f32 / 255. * sat(alpha))))
    }
}

/// Blits the part of `mask` (at `origin`, 1:1) inside `r`: look up, skip zero, blend.
pub(crate) fn draw_mask(
    t: &mut Target,
    mask: &[u8],
    mask_w: usize,
    origin: (i32, i32),
    lut: &AlphaLut,
    tint: &Tint,
    r: IRect,
) {
    let x0 = (r.x0 - origin.0) as usize;
    let n = (r.x1 - r.x0) as usize;
    for y in r.y0..r.y1 {
        let start = (y - origin.1) as usize * mask_w + x0;
        let src = &mask[start..start + n];
        let row = t.row(y, r.x0, r.x1);
        for (p, &m) in row.iter_mut().zip(src) {
            if m != 0 {
                *p = tint.over(*p, lut.0[m as usize]);
            }
        }
    }
}

/// A glyph: 1:1, its bounds are device-pixel aligned.
pub(crate) fn draw_glyph(
    t: &mut Target,
    mask: &AlphaMask,
    origin: (i32, i32),
    lut: &AlphaLut,
    tint: &Tint,
    r: IRect,
) {
    let r = r.intersect(&mask_rect(origin, mask.size.width.0, mask.size.height.0));
    if !r.is_empty() {
        draw_mask(t, &mask.data, mask.size.width.0 as usize, origin, lut, tint, r);
    }
}

/// The pixels a 1:1 mask of `w` x `h` at `origin` covers.
fn mask_rect(origin: (i32, i32), w: i32, h: i32) -> IRect {
    IRect::new(origin.0, origin.1, origin.0.saturating_add(w), origin.1.saturating_add(h))
}

/// Linear sampling of a mask stretched over a destination, in fixed point: the texel coordinate of
/// destination pixel `i` is `(k0 + i * k) / 2^32` (centres, minus half a texel), as a GPU sampler;
/// the weight keeps 8 bits, as GPU filtering does.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Axis {
    k0: i64,
    k: i64,
    /// The source's size along this axis.
    n: i32,
}

impl Axis {
    /// Maps the destination `dst0 .. dst0 + dst_len` (device px, as `f32`) onto `n` texels.
    pub fn new(dst0: f32, dst_len: f32, n: i32) -> Self {
        let scale = n as f64 / dst_len as f64;
        let scale = if scale.is_finite() { scale } else { 0. };
        // u(i) = ((i + 0.5) - dst0) * scale - 0.5
        let one = (1u64 << 32) as f64;
        let k = (scale * one).round() as i64;
        let k0 = (((0.5 - dst0 as f64) * scale - 0.5) * one).round() as i64;
        Axis { k0, k, n }
    }

    /// Texel indices and the weight of the second (`0..=256`) for destination pixel `i`.
    #[inline(always)]
    pub fn at(&self, i: i32) -> (usize, usize, u32) {
        let u = self.k0 + self.k * i as i64;
        let i0 = u >> 32;
        let f = ((u >> 24) & 0xff) as u32;
        let last = (self.n - 1).max(0) as i64;
        (i0.clamp(0, last) as usize, (i0 + 1).clamp(0, last) as usize, f)
    }
}

/// Bilinear sample of an 8-bit mask; returns `value * 65536` (`0..=255 * 65536`).
#[inline(always)]
fn sample(mask: &[u8], w: usize, ax: (usize, usize, u32), ay: (usize, usize, u32)) -> u32 {
    let (x0, x1, fx) = ax;
    let (y0, y1, fy) = ay;
    let r0 = &mask[y0 * w..];
    let r1 = &mask[y1 * w..];
    let top = r0[x0] as u32 * (256 - fx) + r0[x1] as u32 * fx;
    let bot = r1[x0] as u32 * (256 - fx) + r1[x1] as u32 * fx;
    top * (256 - fy) + bot * fy
}

/// An SVG icon: the mask (often 2x) is sampled linearly over the sprite's bounds, then corrected like
/// a glyph. GPUI's `monochrome_sprite_fragment` applies the contrast and gamma correction to every
/// monochrome sprite, icons included, at the filtered sample: here `lut` (the colour's glyph table,
/// alpha included) is interpolated between the two mask levels around the sample.
pub(crate) fn draw_svg(
    t: &mut Target,
    mask: &AlphaMask,
    ax: &Axis,
    ay: &Axis,
    lut: &AlphaLut,
    tint: &Tint,
    r: IRect,
) {
    let w = mask.size.width.0 as usize;
    if w == 0 || mask.size.height.0 <= 0 {
        return;
    }
    for y in r.y0..r.y1 {
        let sy = ay.at(y);
        let row = t.row(y, r.x0, r.x1);
        for (x, p) in (r.x0..r.x1).zip(row.iter_mut()) {
            let s = sample(&mask.data, w, ax.at(x), sy);
            if s != 0 {
                let a = svg_alpha(lut, s);
                if a != 0 {
                    *p = tint.over(*p, a);
                }
            }
        }
    }
}

/// The table's alpha at a filtered sample `s` (`value * 65536`, `0..=255 * 65536`), linear between
/// the two mask levels around it.
#[inline(always)]
fn svg_alpha(lut: &AlphaLut, s: u32) -> u32 {
    let i = (s >> 16) as usize;
    let f = (s & 0xffff) as u64;
    let a0 = lut.0[i] as u64;
    if f == 0 {
        return a0 as u32;
    }
    let a1 = lut.0[(i + 1).min(255)] as u64;
    ((a0 * (65536 - f) + a1 * f + (1 << 15)) >> 16).min(ONE as u64) as u32
}

/// An emoji: premultiplied BGRA, 1:1, times `opacity` (`0..=ONE`).
pub(crate) fn draw_color(t: &mut Target, mask: &ColorMask, origin: (i32, i32), opacity: u32, r: IRect) {
    let r = r.intersect(&mask_rect(origin, mask.size.width.0, mask.size.height.0));
    if r.is_empty() {
        return;
    }
    let w = mask.size.width.0 as usize;
    let x0 = (r.x0 - origin.0) as usize;
    for y in r.y0..r.y1 {
        let start = ((y - origin.1) as usize * w + x0) * 4;
        let src = &mask.data[start..start + (r.x1 - r.x0) as usize * 4];
        let row = t.row(y, r.x0, r.x1);
        for (p, s) in row.iter_mut().zip(src.as_chunks::<4>().0) {
            if s[3] != 0 {
                *p = over_pm8(*p, s[3] as u32, s[2] as u32, s[1] as u32, s[0] as u32, opacity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::blend::unorm;

    /// The table path (integer alpha, integer blend) against the float formula (the shader's maths
    /// and a float blend rounded like a UNORM target), for every mask value and several colours.
    #[test]
    fn glyph_table_equals_float_formula_for_all_alphas() {
        let blends = [
            GlyphBlend::new([0.149_355_4, -0.226_006_7, 0.026_559_83, -0.034_591_47], 1.0),
            GlyphBlend::new([0.1, -0.1, 0.05, -0.05], 0.5),
            GlyphBlend::IDENTITY,
        ];
        let colours = [
            [1., 1., 1., 1.],
            [0., 0., 0., 1.],
            [0.9, 0.3, 0.2, 0.8],
            [0.45, 0.5, 0.55, 0.6],
            [0.2, 0.6, 1., 1.],
        ];
        let dsts = [0x00_00_00u32, 0xff_ff_ff, 0x20_24_28, 0xe0_d8_d0];
        for blend in &blends {
            for c in colours {
                let lut = AlphaLut::glyph(c, blend);
                let tint = Tint::new(c);
                assert_eq!(lut.0[0], 0, "zero coverage must stay zero (the blit skips it)");
                for m in 0..=255u8 {
                    let a = glyph_alpha_f32(m, c, blend);
                    // The table holds the formula's alpha to 1/65536.
                    assert!((lut.0[m as usize] as f32 / ONE as f32 - a).abs() <= 1. / 65536., "{m} {c:?}");
                    for d in dsts {
                        let ch = [(d >> 16) & 0xff, (d >> 8) & 0xff, d & 0xff];
                        let want: [u32; 3] =
                            std::array::from_fn(|i| unorm(c[i] * a + ch[i] as f32 / 255. * (1. - a)));
                        let p = if m == 0 { d } else { tint.over(d, lut.0[m as usize]) };
                        let got = [(p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff];
                        for i in 0..3 {
                            assert!(
                                want[i].abs_diff(got[i]) <= 1,
                                "m {m} c {c:?} dst {d:#x}: {want:?} vs {got:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn correction_matches_hand_values() {
        // Black text: brightness 0, contrast multiplier saturate(3) = 1.
        let g = [0.1, -0.1, 0.05, -0.05];
        let a = apply_contrast_and_gamma_correction(0.5, [0.; 3], 1.0, g);
        // enhance: 0.5 * 2 / 1.5 = 2/3; correction = (-0.1) * 2/3 + (-0.05); a + a(1-a)c.
        let e = 2. / 3.;
        let want = e + e * (1. - e) * (-0.1 * e - 0.05);
        assert!((a - want).abs() < 1e-6);
        // White text: brightness 1, multiplier saturate(-1) = 0, so no contrast.
        let w = apply_contrast_and_gamma_correction(0.5, [1.; 3], 1.0, g);
        let want = 0.5 + 0.25 * ((0.1 - 0.1) * 0.5 + (0.05 - 0.05));
        assert!((w - want).abs() < 1e-6);
    }

    /// An icon's filtered sample gets the glyph correction (GPUI's mono sprite shader), not plain
    /// alpha: the interpolated table equals the float formula at in-between samples too.
    #[test]
    fn svg_samples_get_the_glyph_correction() {
        let blend = GlyphBlend::new([0.149_355_4, -0.226_006_7, 0.026_559_83, -0.034_591_47], 1.0);
        for c in [[0.93, 0.94, 0.95, 1.], [0.1, 0.12, 0.14, 1.], [0.6, 0.62, 0.66, 0.8]] {
            let lut = AlphaLut::glyph(c, &blend);
            for s in (0..=255 * 65536u32).step_by(65536 / 4 + 7) {
                let v = s as f32 / (255. * 65536.);
                let want = sat(c[3]
                    * apply_contrast_and_gamma_correction(v, [c[0], c[1], c[2]], 1.0, blend.gamma_ratios));
                let got = svg_alpha(&lut, s) as f32 / ONE as f32;
                assert!((got - want).abs() < 0.5 / 255., "{c:?} {v}: {got} vs {want}");
            }
            // A dark icon on a light background is darker than plain alpha would draw it.
            if c[0] < 0.5 {
                assert!(svg_alpha(&lut, 128 << 16) > 128 * ONE / 255);
            }
        }
        // No correction: plain alpha.
        let lut = AlphaLut::glyph([1., 1., 1., 1.], &GlyphBlend::IDENTITY);
        assert_eq!(svg_alpha(&lut, 255 << 16), ONE);
        assert_eq!(svg_alpha(&lut, (51 << 16) + 32768), alpha16(51.5 / 255.));
    }

    #[test]
    fn axis_maps_centres_like_a_sampler() {
        // 2:1 aligned: destination pixel i samples texel coordinate 2i + 0.5, halfway between 2i, 2i+1.
        let a = Axis::new(10., 8., 16);
        assert_eq!(a.at(10), (0, 1, 128));
        assert_eq!(a.at(13), (6, 7, 128));
        // 1:1: exactly texel i.
        let b = Axis::new(5., 4., 4);
        assert_eq!(b.at(5), (0, 1, 0));
        assert_eq!(b.at(8), (3, 3, 0));
    }
}
