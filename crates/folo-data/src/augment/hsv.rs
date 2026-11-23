//! HSV colour jitter with integer hexcone HSV and lookup tables.
//!
//! The arithmetic is integer only, so it has one exact result per pixel. On CPUs with AVX2, explicit 256-bit SIMD
//! kernels (`crate::simd::hsv_fused`) compute 8 pixels at a time using packed `max_lut` gathers and exact division-free
//! `/ 255` and `/ 65280` identities, both standalone ([`apply_hsv`]) and fused into the L1 planar pipeline. Tests check
//! AVX2 against the scalar per-pixel reference on all 2^24 colours.

use crate::image::Image;
use crate::par::for_rows;
use crate::rng::Rng;
pub(crate) use crate::simd::hsv_fused::HsvTables;
use crate::simd::hsv_fused::hsv_row_inplace;

/// Random HSV jitter as in YOLO: hue shifts by `U(-1, 1) * gains[0]` of a full turn, saturation and value scale by
/// `1 + U(-1, 1) * gains[1|2]`. Draws exactly three numbers from `rng`.
pub fn random_hsv(img: &mut Image, gains: [f32; 3], rng: &mut Rng) {
    let Some(t) = random_hsv_tables(gains, rng) else {
        return;
    };
    let stride = img.stride();
    for_rows(&mut img.data, stride, |_, row| hsv_row_inplace(row, &t));
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
    let stride = img.stride();
    for_rows(&mut img.data, stride, |_, row| hsv_row_inplace(row, &t));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::PlanarRow;
    use crate::simd::hsv_fused::{
        hsv_row_inplace_scalar, rgb_row_hsv_to_planar_scalar, rgb_row_to_planar_fused,
    };

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
}
