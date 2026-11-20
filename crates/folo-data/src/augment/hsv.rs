//! HSV colour jitter with integer hexcone HSV and lookup tables.
//!
//! The arithmetic is integer only, so it has one exact result per pixel. Two kernels compute it: a pixel at a
//! time with branches (the hue's sector, the channel holding the maximum), and, on CPUs with AVX2, blocks of 64
//! pixels laid out as channel arrays with every branch turned into a select and every table read into a lookup the
//! compiler gathers, which runs about twice as fast on photographs (docs/notes/hill/r6-loader.md). Tests check
//! both against each other on all 2^24 colours.

use crate::image::Image;
use crate::par::for_rows;
use crate::rng::Rng;

/// Hue units per full circle (6 sectors of 256).
const HUE_CIRCLE: i32 = 6 * 256;

/// Random HSV jitter as in YOLO: hue shifts by `U(-1, 1) * gains[0]` of a full turn, saturation and value scale by
/// `1 + U(-1, 1) * gains[1|2]`. Draws exactly three numbers from `rng`.
pub fn random_hsv(img: &mut Image, gains: [f32; 3], rng: &mut Rng) {
    let r = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
    if gains.iter().all(|&g| g == 0.0) {
        return;
    }
    apply_hsv(img, r[0] * gains[0], 1.0 + r[1] * gains[1], 1.0 + r[2] * gains[2]);
}

/// Shifts hue by `hue_turns` of a full circle and scales saturation and value, clamping to the valid range.
pub fn apply_hsv(img: &mut Image, hue_turns: f32, sat_gain: f32, val_gain: f32) {
    let shift = ((hue_turns * HUE_CIRCLE as f32).round() as i32).rem_euclid(HUE_CIRCLE);
    if shift == 0 && sat_gain == 1.0 && val_gain == 1.0 {
        return;
    }
    let lut = |g: f32| -> [u8; 256] {
        let mut t = [0u8; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = (i as f32 * g).round().clamp(0.0, 255.0) as u8;
        }
        t
    };
    let (slut, vlut) = (lut(sat_gain), lut(val_gain));
    // Reciprocals for the two divisions per pixel, 16-bit fixed point.
    let mut recip = [0u32; 256];
    for (d, r) in recip.iter_mut().enumerate().skip(1) {
        *r = ((1u32 << 16) + d as u32 / 2) / d as u32;
    }
    let t = Tables { shift, slut, vlut, recip };
    let stride = img.stride();
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2.
        for_rows(&mut img.data, stride, |_, row| unsafe { hsv_blocks_avx2(row, &t) });
        return;
    }
    for_rows(&mut img.data, stride, |_, row| hsv_pixels(row, &t));
}

/// What [`apply_hsv`] maps every pixel through: the hue shift in `1 / 256` sectors, the saturation and value
/// curves, and `recip[d] = round(2^16 / d)`.
struct Tables {
    shift: i32,
    slut: [u8; 256],
    vlut: [u8; 256],
    recip: [u32; 256],
}

/// [`hsv_blocks`] compiled for AVX2.
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsv_blocks_avx2(row: &mut [u8], t: &Tables) {
    hsv_blocks(row, t)
}

/// Pixels per block of [`hsv_blocks`].
#[cfg(any(test, target_arch = "x86_64"))]
const BLOCK: usize = 64;

/// [`hsv_pixels`], 64 pixels at a time: the block's channels are copied into `i32` arrays, then one loop without
/// branches computes every output channel (the cases of [`hsv_pixels`] become masks and selects, and a grey pixel
/// needs no case: its saturation is `slut[0] = 0`, which makes all three channels `v`), so the compiler can run
/// it on vectors.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
fn hsv_blocks(row: &mut [u8], t: &Tables) {
    // `u32` tables: what a vector gather reads.
    let slut: [u32; 256] = std::array::from_fn(|i| t.slut[i] as u32);
    let vlut: [u32; 256] = std::array::from_fn(|i| t.vlut[i] as u32);
    let recip = &t.recip;
    for block in row.as_chunks_mut::<3>().0.chunks_mut(BLOCK) {
        let (mut r, mut g, mut b) = ([0i32; BLOCK], [0i32; BLOCK], [0i32; BLOCK]);
        for (i, px) in block.iter().enumerate() {
            (r[i], g[i], b[i]) = (px[0] as i32, px[1] as i32, px[2] as i32);
        }
        let mut out = [[0u8; 3]; BLOCK];
        for i in 0..BLOCK {
            let (r, g, b) = (r[i], g[i], b[i]);
            let max = r.max(g).max(b);
            let delta = max - r.min(g).min(b);
            let v = vlut[(max & 255) as usize];
            // Hexcone hue in [0, 1536), from the channel holding the maximum (red first, then green, then blue).
            let is_r = -((max == r) as i32);
            let is_g = !is_r & -((max == g) as i32);
            let is_b = !(is_r | is_g);
            let num = (is_r & (g - b)) | (is_g & (b - r)) | (is_b & (r - g));
            let off = (is_g & 512) | (is_b & 1024);
            let rd = recip[(delta & 255) as usize] as i32;
            let mut h = off + ((num * 256 * rd) >> 16) + t.shift;
            // `rem_euclid(HUE_CIRCLE)` of a value in [-256, 2 * HUE_CIRCLE).
            h += HUE_CIRCLE & (h >> 31);
            h -= HUE_CIRCLE & -((h >= HUE_CIRCLE) as i32);
            let s0 = ((delta * 255 * recip[(max & 255) as usize] as i32 + (1 << 15)) >> 16).min(255);
            let s = slut[(s0 & 255) as usize];
            let (sector, frac) = (h >> 8, (h & 255) as u32);
            let p = (v * (255 - s) + 127) / 255;
            let q = (v * (255 * 256 - s * frac) + 255 * 128) / (255 * 256);
            let w = (v * (255 * 256 - s * (256 - frac)) + 255 * 128) / (255 * 256);
            // Sectors 0..5: (v, w, p), (q, v, p), (p, v, w), (p, q, v), (w, p, v), (v, p, q).
            let r2 = if sector == 0 || sector == 5 {
                v
            } else if sector == 1 {
                q
            } else if sector == 4 {
                w
            } else {
                p
            };
            let g2 = if sector == 0 {
                w
            } else if sector <= 2 {
                v
            } else if sector == 3 {
                q
            } else {
                p
            };
            let b2 = if sector <= 1 {
                p
            } else if sector == 2 {
                w
            } else if sector <= 4 {
                v
            } else {
                q
            };
            out[i] = [r2 as u8, g2 as u8, b2 as u8];
        }
        block.copy_from_slice(&out[..block.len()]);
    }
}

/// The HSV map a pixel at a time, with branches (the kernel for CPUs without AVX2).
fn hsv_pixels(row: &mut [u8], t: &Tables) {
    let Tables { shift, slut, vlut, recip } = t;
    let shift = *shift;
    for px in row.as_chunks_mut::<3>().0.iter_mut() {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;
        let v = vlut[max as usize] as i32;
        if delta == 0 {
            px.copy_from_slice(&[v as u8; 3]);
            continue;
        }
        // Hexcone hue in [0, 1536).
        let rd = recip[delta as usize] as i32;
        let mut h = if max == r {
            ((g - b) * 256 * rd) >> 16
        } else if max == g {
            512 + (((b - r) * 256 * rd) >> 16)
        } else {
            1024 + (((r - g) * 256 * rd) >> 16)
        };
        h = (h + shift).rem_euclid(HUE_CIRCLE);
        let s0 = ((delta * 255 * recip[max as usize] as i32 + (1 << 15)) >> 16).min(255);
        let s = slut[s0 as usize] as i32;
        // Back to RGB.
        let sector = h >> 8;
        let frac = h & 255;
        let p = (v * (255 - s) + 127) / 255;
        let q = (v * (255 * 256 - s * frac) + 255 * 128) / (255 * 256);
        let t = (v * (255 * 256 - s * (256 - frac)) + 255 * 128) / (255 * 256);
        let (r2, g2, b2) = match sector {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };
        px[0] = r2 as u8;
        px[1] = g2 as u8;
        px[2] = b2 as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_colours() -> Image {
        let mut img = Image::new(64, 64, [0, 0, 0]);
        let mut rng = Rng::new(3);
        for v in img.data.iter_mut() {
            *v = rng.below(256) as u8;
        }
        img
    }

    fn tables(rng: &mut Rng) -> Tables {
        let r = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
        let (hue, sat, val) = (r[0] * 0.5, 1.0 + r[1] * 0.9, 1.0 + r[2] * 0.6);
        let lut =
            |g: f32| -> [u8; 256] { std::array::from_fn(|i| (i as f32 * g).round().clamp(0.0, 255.0) as u8) };
        let mut recip = [0u32; 256];
        for (d, r) in recip.iter_mut().enumerate().skip(1) {
            *r = ((1u32 << 16) + d as u32 / 2) / d as u32;
        }
        let shift = ((hue * HUE_CIRCLE as f32).round() as i32).rem_euclid(HUE_CIRCLE);
        Tables { shift, slut: lut(sat), vlut: lut(val), recip }
    }

    /// The block kernel (as compiled here and, where the CPU has it, for AVX2) gives the pixel kernel's bytes for
    /// all 2^24 colours, under a hue shift with saturation and value gains that clamp; rows of any length.
    #[test]
    fn kernels_agree_on_all_colours() {
        let mut rng = Rng::new(12);
        let t = tables(&mut rng);
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2");
        for red in 0..256u32 {
            let mut row = vec![0u8; 3 << 16];
            for (i, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                *px = [red as u8, (i >> 8) as u8, i as u8];
            }
            let mut expected = row.clone();
            hsv_pixels(&mut expected, &t);
            let mut plain = row.clone();
            hsv_blocks(&mut plain, &t);
            assert!(plain == expected, "block kernel, red {red}");
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                // Rows of every length modulo the block, then the whole row.
                let len = 1 + red as usize * 3;
                let (mut a, mut b) = (row[..len * 3].to_vec(), row[..len * 3].to_vec());
                hsv_pixels(&mut a, &t);
                // SAFETY: the CPU has AVX2.
                unsafe { hsv_blocks_avx2(&mut b, &t) };
                assert_eq!(a, b, "AVX2 kernel, {len} pixels");
                // SAFETY: as above.
                unsafe { hsv_blocks_avx2(&mut row, &t) };
                assert!(row == expected, "AVX2 kernel, red {red}");
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
