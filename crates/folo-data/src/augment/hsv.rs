//! HSV colour jitter with integer hexcone HSV and lookup tables.
//!
//! The arithmetic is integer only, so it has one exact result per pixel. On CPUs with AVX2, explicit 256-bit SIMD
//! kernels (`crate::simd::hsv_fused`) compute 8 pixels at a time using packed `max_lut` gathers and exact division-free
//! `/ 255` and `/ 65280` identities, both standalone ([`apply_hsv`]) and fused into the L1 planar pipeline. Tests check
//! AVX2 against the scalar per-pixel reference on all 2^24 colours.

use crate::image::Image;
use crate::par::for_bands;
use crate::rng::Rng;
pub(crate) use crate::simd::hsv::HsvTables;
use crate::simd::hsv::hsv_row_inplace;

/// Random HSV jitter as in YOLO: hue shifts by `U(-1, 1) * gains[0]` of a full turn, saturation and value scale by
/// `1 + U(-1, 1) * gains[1|2]`. Draws exactly three numbers from `rng`.
pub fn random_hsv(img: &mut Image, gains: [f32; 3], rng: &mut Rng) {
    let Some(t) = random_hsv_tables(gains, rng) else {
        return;
    };
    apply_hsv_tables(img, &t);
}

#[inline]
fn apply_hsv_tables(img: &mut Image, t: &HsvTables) {
    let stride = img.stride();
    let h = img.height as usize;
    if stride == 0 || h == 0 {
        return;
    }
    crate::simd::assert_image_data(img);
    for_bands(&mut img.data[..h * stride], stride, |_, band| hsv_row_inplace(band, t));
}

/// Draws the three RNG numbers of [`random_hsv`] and builds the precomputed [`HsvTables`] (`None` if no-op).
#[inline]
pub(crate) fn random_hsv_tables(gains: [f32; 3], rng: &mut Rng) -> Option<HsvTables> {
    let r = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
    if gains.iter().all(|&g| g == 0.0) {
        return None;
    }
    HsvTables::new(r[0] * gains[0], 1.0 + r[1] * gains[1], 1.0 + r[2] * gains[2])
}

/// Shifts hue by `hue_turns` of a full circle and scales saturation and value, clamping to the valid range.
pub fn apply_hsv(img: &mut Image, hue_turns: f32, sat_gain: f32, val_gain: f32) {
    let Some(t) = HsvTables::new(hue_turns, sat_gain, val_gain) else {
        return;
    };
    apply_hsv_tables(img, &t);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::PlanarRow;
    use crate::simd::hsv::{hsv_row_inplace_scalar, rgb_row_hsv_to_planar_scalar, rgb_row_to_planar_fused};

    fn all_colours() -> Image {
        let mut img = Image::new(64, 64, [0, 0, 0]);
        let mut rng = Rng::new(3);
        for v in img.data.iter_mut() {
            *v = rng.below(256) as u8;
        }
        img
    }

    fn tables(rng: &mut Rng) -> HsvTables {
        let r = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
        HsvTables::new(r[0] * 0.5, 1.0 + r[1] * 0.9, 1.0 + r[2] * 0.6).unwrap()
    }

    /// The AVX2 standalone and fused planar kernels give the scalar pixel kernel's exact bytes (and IEEE-754 f32 bits)
    /// for all 2^24 colours, under a hue shift with saturation and value gains that clamp; rows of any length.
    #[test]
    fn kernels_agree_on_all_colours() {
        let mut rng = Rng::new(12);
        let t = tables(&mut rng);
        for red in 0..256u32 {
            let mut row = vec![0u8; 3 << 16];
            for (i, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                *px = [red as u8, (i >> 8) as u8, i as u8];
            }
            let mut expected = row.clone();
            hsv_row_inplace_scalar(&mut expected, &t);

            // Standalone inplace kernel (all 65536 colours for this red, plus odd prefix lengths).
            let mut got = row.clone();
            hsv_row_inplace(&mut got, &t);
            assert!(got == expected, "AVX2 inplace kernel, red {red}");

            let len = 1 + red as usize * 3;
            let (mut a, mut b) = (row[..len * 3].to_vec(), row[..len * 3].to_vec());
            hsv_row_inplace_scalar(&mut a, &t);
            hsv_row_inplace(&mut b, &t);
            assert_eq!(a, b, "AVX2 inplace kernel, {len} pixels");

            // Fused planar u8 and f32/255 kernels (both flip_lr = false and flip_lr = true).
            for flip_lr in [false, true] {
                let (mut er, mut eg, mut eb) = (vec![0u8; len], vec![0u8; len], vec![0u8; len]);
                let (mut gr, mut gg, mut gb) = (vec![0u8; len], vec![0u8; len], vec![0u8; len]);
                rgb_row_hsv_to_planar_scalar(
                    &row[..len * 3],
                    &t,
                    flip_lr,
                    &mut PlanarRow::U8 { r: &mut er, g: &mut eg, b: &mut eb },
                );
                rgb_row_to_planar_fused(
                    &row[..len * 3],
                    Some(&t),
                    flip_lr,
                    &mut PlanarRow::U8 { r: &mut gr, g: &mut gg, b: &mut gb },
                );
                assert_eq!((&gr, &gg, &gb), (&er, &eg, &eb), "fused u8 len={len} flip={flip_lr}");

                let (mut efr, mut efg, mut efb) = (vec![0f32; len], vec![0f32; len], vec![0f32; len]);
                let (mut gfr, mut gfg, mut gfb) = (vec![0f32; len], vec![0f32; len], vec![0f32; len]);
                rgb_row_hsv_to_planar_scalar(
                    &row[..len * 3],
                    &t,
                    flip_lr,
                    &mut PlanarRow::F32Div255 { r: &mut efr, g: &mut efg, b: &mut efb },
                );
                rgb_row_to_planar_fused(
                    &row[..len * 3],
                    Some(&t),
                    flip_lr,
                    &mut PlanarRow::F32Div255 { r: &mut gfr, g: &mut gfg, b: &mut gfb },
                );
                assert!(
                    gfr.iter().zip(&efr).all(|(x, y)| x.to_bits() == y.to_bits())
                        && gfg.iter().zip(&efg).all(|(x, y)| x.to_bits() == y.to_bits())
                        && gfb.iter().zip(&efb).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "fused f32 len={len} flip={flip_lr}"
                );
            }
        }
    }

    #[test]
    fn near_identity_round_trip() {
        // A zero hue shift with gains of one is a no-op; with a tiny forced shift the round trip stays within a few
        // levels per channel.
        let img = all_colours();
        let mut a = img.clone();
        apply_hsv(&mut a, 0.0, 1.0, 1.0);
        assert_eq!(a, img);
        let mut b = img.clone();
        apply_hsv(&mut b, 0.0, 1.0, 1.0000001);
        let worst = img.data.iter().zip(&b.data).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap();
        assert!(worst <= 3, "round trip error {worst}");
    }

    #[test]
    fn hue_rotation_and_value() {
        let mut img = Image::new(1, 1, [200, 0, 0]);
        apply_hsv(&mut img, 1.0 / 3.0, 1.0, 1.0); // red -> green
        let p = img.pixel(0, 0);
        assert!(p[1] >= 198 && p[0] <= 2 && p[2] <= 2, "{p:?}");
        let mut img = Image::new(1, 1, [100, 50, 20]);
        apply_hsv(&mut img, 0.0, 1.0, 0.5);
        let p = img.pixel(0, 0);
        assert!(
            (p[0] as i32 - 50).abs() <= 1 && (p[1] as i32 - 25).abs() <= 1 && (p[2] as i32 - 10).abs() <= 1
        );
        let mut grey = Image::new(1, 1, [80, 80, 80]);
        apply_hsv(&mut grey, 0.2, 2.0, 1.5);
        assert_eq!(grey.pixel(0, 0), [120, 120, 120]);
        // Desaturate completely.
        let mut img = Image::new(1, 1, [200, 100, 50]);
        apply_hsv(&mut img, 0.0, 0.0, 1.0);
        assert_eq!(img.pixel(0, 0), [200, 200, 200]);
    }

    #[test]
    fn same_colour_same_result() {
        let mut img = Image::new(50, 40, [30, 140, 220]);
        img.fill_rect(10, 10, 20, 20, [250, 10, 90]);
        let mut rng = Rng::new(9);
        random_hsv(&mut img, [0.015, 0.7, 0.4], &mut rng);
        let (a, b) = (img.pixel(0, 0), img.pixel(15, 15));
        for y in 0..40 {
            for x in 0..50 {
                let inside = (10..20).contains(&x) && (10..20).contains(&y);
                assert_eq!(img.pixel(x, y), if inside { b } else { a });
            }
        }
    }

    /// Reference integer hexcone HSV with explicit `/ 255` and `/ (255 * 256)` divisions.
    fn apply_pixel_div_ref(t: &HsvTables, px: [u8; 3]) -> [u8; 3] {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;
        let v = t.vlut[max as usize] as i32;
        if delta == 0 {
            return [v as u8; 3];
        }
        let recip = |d: i32| ((1u32 << 16) + d as u32 / 2) as i32 / d;
        let rd = recip(delta);
        let mut h = if max == r {
            ((g - b) * 256 * rd) >> 16
        } else if max == g {
            512 + (((b - r) * 256 * rd) >> 16)
        } else {
            1024 + (((r - g) * 256 * rd) >> 16)
        };
        h = (h + t.shift).rem_euclid(crate::simd::hsv::HUE_CIRCLE);
        let s0 = ((delta * 255 * recip(max) + (1 << 15)) >> 16).min(255);
        let s = t.slut[s0 as usize] as i32;
        let sector = h >> 8;
        let frac = h & 255;
        let p = (v * (255 - s) + 127) / 255;
        let q = (v * (255 * 256 - s * frac) + 255 * 128) / (255 * 256);
        let w = (v * (255 * 256 - s * (256 - frac)) + 255 * 128) / (255 * 256);
        let (r2, g2, b2) = match sector {
            0 => (v, w, p),
            1 => (q, v, p),
            2 => (p, v, w),
            3 => (p, q, v),
            4 => (w, p, v),
            _ => (v, p, q),
        };
        [r2 as u8, g2 as u8, b2 as u8]
    }

    #[test]
    fn division_free_and_zero_sat_match_reference_bit_exact() {
        let mut rng = Rng::new(77);
        let t_full = tables(&mut rng);
        let t_desat = HsvTables::new(0.25, 0.0, 1.35).unwrap();
        assert!(t_desat.zero_sat);
        for red in (0..256u32).step_by(5) {
            let mut row = Vec::with_capacity(64 * 3);
            for k in 0..64u32 {
                row.extend_from_slice(&[red as u8, ((k * 4) & 255) as u8, ((k * 7 + red) & 255) as u8]);
            }
            for t in [&t_full, &t_desat] {
                let expected: Vec<u8> =
                    row.as_chunks::<3>().0.iter().flat_map(|&px| apply_pixel_div_ref(t, px)).collect();
                let mut got_scalar = row.clone();
                hsv_row_inplace_scalar(&mut got_scalar, t);
                assert_eq!(got_scalar, expected);
                let mut got_fast = row.clone();
                hsv_row_inplace(&mut got_fast, t);
                assert_eq!(got_fast, expected);
            }
        }
    }

    /// Exercises gather-free AVX2 HSV on edge gains (exact halfway `0.5 - 2^-25`, huge `>256`, negative, `0.0`,
    /// `inf`, `NaN`) and odd widths across normal, zero-sat, and fused planar paths.
    #[test]
    fn gather_free_edge_gains_and_zero_sat_match_reference_bit_exact() {
        let near_half = f32::from_bits(0x3eff_ffff); // 0.5 - 2^-25: forces fallback `!fast` rounding path
        let cases = [
            (0.0, near_half, 1.0),
            (0.125, 1.0, near_half),
            (0.3, 0.0, 1.35),
            (0.3, 0.0, near_half),
            (0.2, 300.0, 0.5),
            (0.2, -0.5, 2.0),
            (0.4, f32::INFINITY, 1.2),
            (0.4, 1.2, f32::NAN),
            (0.5, 0.5, 1.5),
        ];
        for &(hue, sg, vg) in &cases {
            let Some(t) = HsvTables::new(hue, sg, vg) else { continue };
            for &w in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 65] {
                let mut row = Vec::with_capacity(w * 3);
                for k in 0..w {
                    row.extend_from_slice(&[
                        ((k * 37 + 1) & 255) as u8,
                        ((k * 73 + 17) & 255) as u8,
                        ((k * 131 + 5) & 255) as u8,
                    ]);
                }
                let expected: Vec<u8> =
                    row.as_chunks::<3>().0.iter().flat_map(|&px| apply_pixel_div_ref(&t, px)).collect();
                let mut got = row.clone();
                hsv_row_inplace(&mut got, &t);
                assert_eq!(got, expected, "inplace hue={hue} sg={sg} vg={vg} w={w}");

                for flip_lr in [false, true] {
                    let (mut er, mut eg, mut eb) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
                    let (mut gr, mut gg, mut gb) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
                    rgb_row_hsv_to_planar_scalar(
                        &row,
                        &t,
                        flip_lr,
                        &mut PlanarRow::U8 { r: &mut er, g: &mut eg, b: &mut eb },
                    );
                    rgb_row_to_planar_fused(
                        &row,
                        Some(&t),
                        flip_lr,
                        &mut PlanarRow::U8 { r: &mut gr, g: &mut gg, b: &mut gb },
                    );
                    assert_eq!(
                        (&gr, &gg, &gb),
                        (&er, &eg, &eb),
                        "fused u8 sg={sg} vg={vg} w={w} flip={flip_lr}"
                    );
                }
            }
        }
        // Empty image is a safe no-op; oversized image leaves trailing bytes untouched.
        let mut empty = Image { width: 0, height: 4, data: Vec::new() };
        apply_hsv(&mut empty, 0.2, 1.5, 0.8);
        let mut oversized = Image { width: 1, height: 1, data: vec![200, 100, 50, 77, 88, 99] };
        apply_hsv(&mut oversized, 0.0, 0.0, 1.0);
        assert_eq!(oversized.data, vec![200, 200, 200, 77, 88, 99]);
    }
}
