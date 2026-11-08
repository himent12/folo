//! HSV colour jitter with integer hexcone HSV and lookup tables.

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
    let stride = img.stride();
    for_rows(&mut img.data, stride, |_, row| {
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
    });
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
