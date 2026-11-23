//! Exhaustive bit-for-bit AVX2 vs scalar parity tests across odd/ragged/large/edge sizes and transforms.

use super::bilinear::{
    WONE, hinterp_taps_to_planar_f32_div255, hinterp_taps_to_planar_f32_scale, hinterp_taps_to_planar_u8,
    hinterp_taps_to_rgb, hinterp_taps_to_rgb_scalar, hinterp_taps_to_rgb0_u32, vblend_rgb_to_rgba32,
    vblend_rgb_to_rgba32_scalar, warp_hinterp_interior_to_rgb, warp_hinterp_interior_to_rgb_scalar,
};
use super::hsv_fused::{HsvTables, warp_interior_to_planar_fused, warp_interior_to_planar_fused_scalar};
use super::planar::{
    flip_lr_row, flip_lr_row_scalar, rgb_to_planar_f32_div255, rgb_to_planar_f32_div255_scalar,
    rgb_to_planar_f32_scale, rgb_to_planar_f32_scale_scalar, rgb_to_planar_u8, rgb_to_planar_u8_scalar,
    rgb_to_rgb0_u32, rgb_to_rgb0_u32_scalar,
};
use super::{PlanarDst, PlanarRow};
use crate::augment::{flip_lr, flip_ud, random_hsv};
use crate::image::{Image, Sample};
use crate::letterbox::{Letterbox, LetterboxOptions, PAD_VALUE};
use crate::resize::{resize, resize_plan};
use crate::rng::Rng;
use crate::warp::{Affine, Plane, Tile, warp_plane, warp_plane_fused};

fn noisy_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.below(256) as u8).collect()
}

fn noisy_image(seed: u64, w: u32, h: u32) -> Image {
    Image { width: w, height: h, data: noisy_bytes(seed, w as usize * h as usize * 3) }
}

#[test]
fn planar_conversions_and_flip_match_scalar_bit_exact() {
    for &w in &[0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 127, 640, 641] {
        let src = noisy_bytes(100 + w as u64, w * 3);
        for flip in [false, true] {
            let (mut r0, mut g0, mut b0) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
            let (mut r1, mut g1, mut b1) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
            rgb_to_planar_u8_scalar(&src, &mut r0, &mut g0, &mut b0, flip);
            rgb_to_planar_u8(&src, &mut r1, &mut g1, &mut b1, flip);
            assert_eq!((&r1, &g1, &b1), (&r0, &g0, &b0), "planar_u8 w={w} flip={flip}");

            let (mut fr0, mut fg0, mut fb0) = (vec![0f32; w], vec![0f32; w], vec![0f32; w]);
            let (mut fr1, mut fg1, mut fb1) = (vec![0f32; w], vec![0f32; w], vec![0f32; w]);
            rgb_to_planar_f32_div255_scalar(&src, &mut fr0, &mut fg0, &mut fb0, flip);
            rgb_to_planar_f32_div255(&src, &mut fr1, &mut fg1, &mut fb1, flip);
            assert!(
                fr1.iter().zip(&fr0).all(|(a, b)| a.to_bits() == b.to_bits())
                    && fg1.iter().zip(&fg0).all(|(a, b)| a.to_bits() == b.to_bits())
                    && fb1.iter().zip(&fb0).all(|(a, b)| a.to_bits() == b.to_bits()),
                "planar_f32_div255 w={w} flip={flip}"
            );
        }
        for &scale in &[1.0 / 255.0_f32, 0.5_f32, 1.0_f32, 0.003921569_f32] {
            let (mut fr0, mut fg0, mut fb0) = (vec![0f32; w], vec![0f32; w], vec![0f32; w]);
            let (mut fr1, mut fg1, mut fb1) = (vec![0f32; w], vec![0f32; w], vec![0f32; w]);
            rgb_to_planar_f32_scale_scalar(&src, &mut fr0, &mut fg0, &mut fb0, scale);
            rgb_to_planar_f32_scale(&src, &mut fr1, &mut fg1, &mut fb1, scale);
            assert!(
                fr1.iter().zip(&fr0).all(|(a, b)| a.to_bits() == b.to_bits())
                    && fg1.iter().zip(&fg0).all(|(a, b)| a.to_bits() == b.to_bits())
                    && fb1.iter().zip(&fb0).all(|(a, b)| a.to_bits() == b.to_bits()),
                "planar_f32_scale w={w} scale={scale}"
            );
        }
        let (mut u0, mut u1) = (vec![0u32; w], vec![0u32; w]);
        rgb_to_rgb0_u32_scalar(&src, &mut u0);
        rgb_to_rgb0_u32(&src, &mut u1);
        assert_eq!(u1, u0, "rgb0_u32 w={w}");

        let (mut f0, mut f1) = (src.clone(), src.clone());
        flip_lr_row_scalar(&mut f0, w);
        flip_lr_row(&mut f1);
        assert_eq!(f1, f0, "flip_lr_row w={w}");
    }
}

#[test]
fn vblend_and_hinterp_match_scalar_bit_exact() {
    for &cols in &[1usize, 2, 3, 5, 6, 7, 13, 16, 17, 63, 64, 65, 321] {
        let top = noisy_bytes(10 + cols as u64, cols * 3);
        let bot = noisy_bytes(20 + cols as u64, cols * 3);
        for &wy in &[0u32, 1, 17, 1024, 2047, WONE] {
            let mut v0 = vec![0u32; (cols + 1) * 4];
            let mut v1 = vec![0u32; (cols + 1) * 4];
            vblend_rgb_to_rgba32_scalar(&top, &bot, wy, &mut v0, cols);
            vblend_rgb_to_rgba32(&top, &bot, wy, &mut v1);
            assert_eq!(&v1[..cols * 4], &v0[..cols * 4], "vblend cols={cols} wy={wy}");
            let last = (cols - 1) * 4;
            v0.copy_within(last..last + 4, cols * 4);
            v1.copy_within(last..last + 4, cols * 4);

            for &dst_w in &[1u32, 3, 5, 6, 7, 8, 9, 15, 16, 17, 63, 64, 131] {
                let plan = resize_plan(cols as u32, 1, dst_w, 1);
                if plan.prefilter != [1, 1] {
                    continue;
                }
                let w = dst_w as usize;
                let (mut o0, mut o1) = (vec![0u8; w * 3], vec![0u8; w * 3]);
                hinterp_taps_to_rgb_scalar(&v0, &plan.x0, &plan.wx, &mut o0);
                // SAFETY: `v1` holds `(cols + 1) * 4` elements and every `plan.x0[x] < cols`.
                unsafe { hinterp_taps_to_rgb(&v1, &plan.x0, &plan.wx, &mut o1) };
                assert_eq!(o1, o0, "hinterp_rgb cols={cols} dst={dst_w}");

                let (mut r0, mut g0, mut b0) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
                rgb_to_planar_u8_scalar(&o0, &mut r0, &mut g0, &mut b0, false);
                let (mut r1, mut g1, mut b1) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
                // SAFETY: as above.
                unsafe { hinterp_taps_to_planar_u8(&v1, &plan.x0, &plan.wx, &mut r1, &mut g1, &mut b1) };
                assert_eq!((&r1, &g1, &b1), (&r0, &g0, &b0), "hinterp_planar_u8 cols={cols} dst={dst_w}");

                let (mut fr0, mut fg0, mut fb0) = (vec![0f32; w], vec![0f32; w], vec![0f32; w]);
                let (mut fr1, mut fg1, mut fb1) = (vec![0f32; w], vec![0f32; w], vec![0f32; w]);
                rgb_to_planar_f32_scale_scalar(&o0, &mut fr0, &mut fg0, &mut fb0, 1.0 / 255.0);
                // SAFETY: as above.
                unsafe {
                    hinterp_taps_to_planar_f32_scale(
                        &v1,
                        &plan.x0,
                        &plan.wx,
                        1.0 / 255.0,
                        &mut fr1,
                        &mut fg1,
                        &mut fb1,
                    );
                }
                assert!(
                    fr1.iter().zip(&fr0).all(|(a, b)| a.to_bits() == b.to_bits())
                        && fg1.iter().zip(&fg0).all(|(a, b)| a.to_bits() == b.to_bits())
                        && fb1.iter().zip(&fb0).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "hinterp_f32_scale cols={cols} dst={dst_w}"
                );

                rgb_to_planar_f32_div255_scalar(&o0, &mut fr0, &mut fg0, &mut fb0, false);
                // SAFETY: as above.
                unsafe {
                    hinterp_taps_to_planar_f32_div255(&v1, &plan.x0, &plan.wx, &mut fr1, &mut fg1, &mut fb1);
                }
                assert!(
                    fr1.iter().zip(&fr0).all(|(a, b)| a.to_bits() == b.to_bits())
                        && fg1.iter().zip(&fg0).all(|(a, b)| a.to_bits() == b.to_bits())
                        && fb1.iter().zip(&fb0).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "hinterp_f32_div255 cols={cols} dst={dst_w}"
                );

                let (mut w0, mut w1) = (vec![0u32; w], vec![0u32; w]);
                rgb_to_rgb0_u32_scalar(&o0, &mut w0);
                // SAFETY: as above.
                unsafe { hinterp_taps_to_rgb0_u32(&v1, &plan.x0, &plan.wx, &mut w1) };
                assert_eq!(w1, w0, "hinterp_rgb0 cols={cols} dst={dst_w}");
            }
        }
    }
}

#[test]
fn warp_hinterp_and_fused_match_scalar_across_steps() {
    let cols = 128usize;
    let top = noisy_bytes(77, cols * 3);
    let bot = noisy_bytes(78, cols * 3);
    let mut v = vec![0u32; cols * 4];
    vblend_rgb_to_rgba32(&top, &bot, 777, &mut v);
    let hsv = HsvTables::new(0.012, 1.45, 0.72).unwrap();

    for &(count, rel_fu0, du) in &[
        (1usize, 0i64, 65536i64),
        (5, 32768, 98304),
        (6, 12345, 65536),
        (7, 12345, 131072),
        (8, 0, 50000),
        (17, 200_000, 110_000),
        (33, 6_000_000, -140_000),
        (63, 7_000_000, -90_000),
        (64, 100_000, 99_999),
    ] {
        let (mut o0, mut o1) = (vec![0u8; count * 3], vec![0u8; count * 3]);
        warp_hinterp_interior_to_rgb_scalar(&v, rel_fu0, du, &mut o0);
        warp_hinterp_interior_to_rgb(&v, rel_fu0, du, &mut o1);
        assert_eq!(o1, o0, "warp_hinterp count={count} du={du}");

        let width = count + 7;
        let x_start = 3;
        let x_end = x_start + count;
        for hsv_opt in [None, Some(&hsv)] {
            for flip_lr in [false, true] {
                let (mut r0, mut g0, mut b0) = (vec![9u8; width], vec![9u8; width], vec![9u8; width]);
                let (mut r1, mut g1, mut b1) = (vec![9u8; width], vec![9u8; width], vec![9u8; width]);
                warp_interior_to_planar_fused_scalar(
                    &v,
                    rel_fu0,
                    du,
                    width,
                    x_start,
                    x_end,
                    hsv_opt,
                    flip_lr,
                    &mut PlanarRow::U8 { r: &mut r0, g: &mut g0, b: &mut b0 },
                );
                warp_interior_to_planar_fused(
                    &v,
                    rel_fu0,
                    du,
                    width,
                    x_start,
                    x_end,
                    hsv_opt,
                    flip_lr,
                    &mut PlanarRow::U8 { r: &mut r1, g: &mut g1, b: &mut b1 },
                );
                assert_eq!((&r1, &g1, &b1), (&r0, &g0, &b0), "fused warp u8 count={count} flip={flip_lr}");

                let (mut fr0, mut fg0, mut fb0) =
                    (vec![-1f32; width], vec![-1f32; width], vec![-1f32; width]);
                let (mut fr1, mut fg1, mut fb1) =
                    (vec![-1f32; width], vec![-1f32; width], vec![-1f32; width]);
                warp_interior_to_planar_fused_scalar(
                    &v,
                    rel_fu0,
                    du,
                    width,
                    x_start,
                    x_end,
                    hsv_opt,
                    flip_lr,
                    &mut PlanarRow::F32Div255 { r: &mut fr0, g: &mut fg0, b: &mut fb0 },
                );
                warp_interior_to_planar_fused(
                    &v,
                    rel_fu0,
                    du,
                    width,
                    x_start,
                    x_end,
                    hsv_opt,
                    flip_lr,
                    &mut PlanarRow::F32Div255 { r: &mut fr1, g: &mut fg1, b: &mut fb1 },
                );
                assert!(
                    fr1.iter().zip(&fr0).all(|(a, b)| a.to_bits() == b.to_bits())
                        && fg1.iter().zip(&fg0).all(|(a, b)| a.to_bits() == b.to_bits())
                        && fb1.iter().zip(&fb0).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "fused warp f32 count={count} flip={flip_lr}"
                );
            }
        }
    }
}

#[test]
fn warp_plane_fused_matches_unfused_pipeline_bit_exact() {
    let mut rng = Rng::new(42);
    let imgs: Vec<Image> = [(97, 61), (64, 113), (33, 29), (128, 96)]
        .into_iter()
        .enumerate()
        .map(|(k, (w, h))| noisy_image(k as u64 + 1, w, h))
        .collect();

    for case in 0..48 {
        let (pw, ph) = (120 + rng.below(50) as u32, 120 + rng.below(50) as u32);
        let (xc, yc) = (30 + rng.below(60) as i64, 30 + rng.below(60) as i64);
        let mut tiles = Vec::new();
        for (k, img) in imgs.iter().enumerate() {
            let (iw, ih) = (img.width as i64, img.height as i64);
            let (x0, x1) = if k % 2 == 0 { ((xc - iw).max(0), xc) } else { (xc, (xc + iw).min(pw as i64)) };
            let (y0, y1) = if k < 2 { ((yc - ih).max(0), yc) } else { (yc, (yc + ih).min(ph as i64)) };
            if x1 > x0 && y1 > y0 {
                let src =
                    [if k % 2 == 0 { iw - (x1 - x0) } else { 0 }, if k < 2 { ih - (y1 - y0) } else { 0 }];
                tiles.push(Tile { img, dst: [x0, y0, x1, y1], src });
            }
        }
        let plane = Plane { width: pw, height: ph, fill: [PAD_VALUE; 3], tiles };
        let (ow, oh) = (1 + rng.below(97) as u32, 1 + rng.below(83) as u32);
        let m = match case % 4 {
            0 => Affine::IDENTITY,
            1 => Affine::translate(-(pw as f64) / 2.0, -(ph as f64) / 2.0)
                .then(&Affine::scale(rng.uniform(0.4, 1.8) as f64, rng.uniform(0.4, 1.8) as f64))
                .then(&Affine::translate(rng.uniform(-20.0, 90.0) as f64, rng.uniform(-20.0, 90.0) as f64)),
            2 => Affine::translate(-(pw as f64) / 2.0, -(ph as f64) / 2.0)
                .then(&Affine::scale(-rng.uniform(0.5, 1.5) as f64, rng.uniform(0.5, 1.5) as f64))
                .then(&Affine::translate(50.0, 40.0)),
            _ => Affine::translate(-(pw as f64) / 2.0, -(ph as f64) / 2.0)
                .then(&Affine::rotate(rng.uniform(-25.0, 25.0) as f64))
                .then(&Affine::shear(rng.uniform(-5.0, 5.0) as f64, rng.uniform(-5.0, 5.0) as f64))
                .then(&Affine::scale(rng.uniform(0.6, 1.4) as f64, rng.uniform(0.6, 1.4) as f64))
                .then(&Affine::translate(ow as f64 * 0.5, oh as f64 * 0.5)),
        };
        let (ow, oh) = if case % 4 == 0 { (pw, ph) } else { (ow, oh) };
        let use_hsv = case % 3 != 0;
        let hsv = use_hsv
            .then(|| {
                HsvTables::new(
                    rng.uniform(-0.02, 0.02),
                    1.0 + rng.uniform(-0.7, 0.7),
                    1.0 + rng.uniform(-0.4, 0.4),
                )
            })
            .flatten();
        let do_ud = case % 2 == 1;
        let do_lr = (case / 2) % 2 == 1;

        // Reference: unfused warp_plane -> apply_hsv -> flip_ud -> flip_lr -> scalar planar.
        let mut ref_sample = Sample { image: warp_plane(&plane, &m, ow, oh), boxes: Vec::new() };
        if let Some(t) = &hsv {
            for px in ref_sample.image.data.as_chunks_mut::<3>().0.iter_mut() {
                *px = t.apply_pixel(*px);
            }
        }
        if do_ud {
            flip_ud(&mut ref_sample);
        }
        if do_lr {
            flip_lr(&mut ref_sample);
        }

        let area = ow as usize * oh as usize;
        let mut want_u8 = vec![0u8; 3 * area];
        let mut want_f32 = vec![0f32; 3 * area];
        for (i, px) in ref_sample.image.data.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                want_u8[c * area + i] = px[c];
                want_f32[c * area + i] = px[c] as f32 / 255.0;
            }
        }

        let mut got_u8 = vec![0xAAu8; 3 * area];
        warp_plane_fused(
            &plane,
            &m,
            ow,
            oh,
            hsv.as_ref(),
            do_ud,
            do_lr,
            &mut PlanarDst::from_u8(&mut got_u8, area),
        );
        assert_eq!(got_u8, want_u8, "fused u8 case={case}");

        let mut got_f32 = vec![f32::NAN; 3 * area];
        warp_plane_fused(
            &plane,
            &m,
            ow,
            oh,
            hsv.as_ref(),
            do_ud,
            do_lr,
            &mut PlanarDst::from_f32_div255(&mut got_f32, area),
        );
        assert!(
            got_f32.iter().zip(&want_f32).all(|(a, b)| a.to_bits() == b.to_bits()),
            "fused f32 case={case}"
        );

        // Also check serial vs parallel agreement.
        let mut got_serial = vec![0x55u8; 3 * area];
        crate::par::serial(|| {
            warp_plane_fused(
                &plane,
                &m,
                ow,
                oh,
                hsv.as_ref(),
                do_ud,
                do_lr,
                &mut PlanarDst::from_u8(&mut got_serial, area),
            );
        });
        assert_eq!(got_serial, want_u8, "fused serial u8 case={case}");
    }
}

#[test]
fn letterbox_planar_dst_matches_apply_and_resize() {
    for &(w, h, s) in
        &[(640u32, 480u32, 640u32), (1280, 720, 640), (1921, 1081, 640), (37, 23, 64), (64, 64, 64)]
    {
        let img = noisy_image(w as u64 * 17 + h as u64, w, h);
        let lb = Letterbox::compute(w, h, s, s, &LetterboxOptions::default());
        let canvas = lb.apply(&img, PAD_VALUE);
        let area = s as usize * s as usize;

        let mut want_u8 = vec![0u8; 3 * area];
        let mut want_f32 = vec![0f32; 3 * area];
        for (i, px) in canvas.data.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                want_u8[c * area + i] = px[c];
                want_f32[c * area + i] = px[c] as f32 / 255.0;
            }
        }
        let mut got_u8 = vec![0u8; 3 * area];
        lb.apply_planar_dst(&img, PAD_VALUE, &mut PlanarDst::from_u8(&mut got_u8, area));
        assert_eq!(got_u8, want_u8, "letterbox planar u8 {w}x{h}");

        let mut got_f32 = vec![0f32; 3 * area];
        lb.apply_planar_dst(&img, PAD_VALUE, &mut PlanarDst::from_f32_div255(&mut got_f32, area));
        assert!(
            got_f32.iter().zip(&want_f32).all(|(a, b)| a.to_bits() == b.to_bits()),
            "letterbox planar f32 {w}x{h}"
        );

        // Smoke-check standalone `resize` + `random_hsv` determinism.
        let mut r = resize(&img, lb.resized[0], lb.resized[1]);
        random_hsv(&mut r, [0.015, 0.7, 0.4], &mut Rng::new(7));
        assert_eq!((r.width, r.height), (lb.resized[0], lb.resized[1]));
    }
}

/// Verifies the widened `i128` step helper (`step_fp`) handles products `x * df` that overflow `i64` while
/// `f0 + x * df` fits in `i64`, and verifies `warp_general` matches scalar on both huge-step rotation and
/// interior-span rotation+shear.
#[test]
fn warp_general_huge_step_with_rotation_does_not_overflow() {
    // Direct test of the widened `i128` offset arithmetic: `256 * (1 << 55) == 2^63` and `257 * -(1 << 55)` both
    // overflow `i64`, while `f0 + x * df` fits cleanly in `i64`.
    let f0 = i64::MIN + (1i64 << 54);
    let df = 1i64 << 55;
    assert!((256i64).checked_mul(df).is_none());
    assert_eq!(crate::warp::step_fp(f0, 256, df), 1i64 << 54);
    assert!((-df).checked_mul(257).is_none());
    assert_eq!(crate::warp::step_fp(i64::MAX - (1i64 << 54), 257, -df), -(3i64 << 54) - 1);

    let src = noisy_image(99, 64, 64);
    let border = [114, 7, 23];
    let m_huge =
        Affine::rotate(1.0).then(&Affine::scale(2f64.powi(-39), 1.0)).then(&Affine::translate(128.5, 10.5));
    let m_rot = Affine::translate(-32.0, -32.0)
        .then(&Affine::rotate(17.0))
        .then(&Affine::shear(4.0, -3.0))
        .then(&Affine::scale(0.8, 1.3))
        .then(&Affine::translate(40.0, 20.0));
    for (m, ow, oh) in [(m_huge, 257u32, 4u32), (m_rot, 97, 63)] {
        let out = crate::warp::warp_affine(&src, &m, ow, oh, border);
        let inv = m.inverse().unwrap();
        let [a, b, c, d, e, f] = inv.0;
        let fone = (1i64 << 16) as f64;
        let (sw, sh) = (src.width as i64, src.height as i64);
        let sstride = src.stride();
        let mut want = vec![0u8; ow as usize * oh as usize * 3];
        for y in 0..oh as usize {
            let yc = y as f64 + 0.5;
            let mut fu = ((a * 0.5 + b * yc + c - 0.5) * fone).round() as i64;
            let mut fv = ((d * 0.5 + e * yc + f - 0.5) * fone).round() as i64;
            let du = (a * fone).round() as i64;
            let dv = (d * fone).round() as i64;
            for x in 0..ow as usize {
                let ix = fu >> 16;
                let iy = fv >> 16;
                let wx = ((fu >> 5) & 2047) as u32;
                let wy = ((fv >> 5) & 2047) as u32;
                fu = fu.wrapping_add(du);
                fv = fv.wrapping_add(dv);
                let dst = &mut want[(y * ow as usize + x) * 3..][..3];
                if ix < -1 || iy < -1 || ix >= sw || iy >= sh {
                    dst.copy_from_slice(&border);
                } else {
                    let fetch = |px: i64, py: i64, ch: usize| -> u32 {
                        if px >= 0 && py >= 0 && px < sw && py < sh {
                            src.data[py as usize * sstride + px as usize * 3 + ch] as u32
                        } else {
                            border[ch] as u32
                        }
                    };
                    let (iwx, iwy) = (2048 - wx, 2048 - wy);
                    for (ch, out) in dst.iter_mut().enumerate() {
                        let t = fetch(ix, iy, ch) * iwx + fetch(ix + 1, iy, ch) * wx;
                        let u = fetch(ix, iy + 1, ch) * iwx + fetch(ix + 1, iy + 1, ch) * wx;
                        *out = ((t * iwy + u * wy + (1 << 21)) >> 22) as u8;
                    }
                }
            }
        }
        assert_eq!(out.data, want, "warp_general mismatch for {m:?}");
    }
}

#[test]
fn malformed_short_image_panics_safely_in_public_apis() {
    let bad = Image { width: 64, height: 64, data: Vec::new() };
    assert!(
        std::panic::catch_unwind(|| crate::warp::warp_affine(&bad, &Affine::rotate(15.0), 64, 64, [0; 3]))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| crate::warp::warp_affine(&bad, &Affine::scale(1.1, 1.1), 64, 64, [0; 3]))
            .is_err()
    );
    assert!(std::panic::catch_unwind(|| resize(&bad, 32, 32)).is_err());
    assert!(std::panic::catch_unwind(|| resize(&bad, 48, 48)).is_err());
    let lb = Letterbox::compute(64, 64, 64, 64, &LetterboxOptions::default());
    assert!(
        std::panic::catch_unwind(|| {
            let mut dst = vec![0f32; 3 * 64 * 64];
            lb.apply_planar_f32(&bad, PAD_VALUE, 1.0 / 255.0, &mut dst);
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            let mut dst = vec![0u32; 64 * 64];
            lb.apply_rgb0_u8(&bad, PAD_VALUE, &mut dst);
        })
        .is_err()
    );
    let bad_lb =
        Letterbox { orig: [16, 16], resized: [16, 16], out: [16, 16], ratio: [1.0, 1.0], pad: [-2.0, 0.0] };
    let ok = noisy_image(1, 16, 16);
    assert!(
        std::panic::catch_unwind(|| {
            let mut dst = vec![0f32; 3 * 16 * 16];
            bad_lb.apply_planar_f32(&ok, PAD_VALUE, 1.0 / 255.0, &mut dst);
        })
        .is_err()
    );
}
