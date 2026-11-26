use super::*;
use crate::rng::Rng;

fn pattern(w: u32, h: u32) -> Image {
    let mut img = Image::new(w, h, [0, 0, 0]);
    for (i, v) in img.data.iter_mut().enumerate() {
        *v = (i * 13 % 251) as u8;
    }
    img
}

#[test]
fn matrix_algebra() {
    let m = Affine::rotate(30.0).then(&Affine::scale(2.0, 1.5)).then(&Affine::translate(3.0, -4.0));
    let inv = m.inverse().unwrap();
    let (x, y) = m.apply(5.0, 7.0);
    let (bx, by) = inv.apply(x, y);
    assert!((bx - 5.0).abs() < 1e-9 && (by - 7.0).abs() < 1e-9);
    assert!(m.then(&inv).is_identity());
    assert!((m.det() - 3.0).abs() < 1e-9);
    // OpenCV convention: +90 degrees takes the x axis to -y (up on screen).
    let (rx, ry) = Affine::rotate(90.0).apply(1.0, 0.0);
    assert!(rx.abs() < 1e-12 && (ry + 1.0).abs() < 1e-12);
}

#[test]
fn identity_and_integer_translation_are_exact() {
    let img = pattern(31, 17);
    assert_eq!(warp_affine(&img, &Affine::IDENTITY, 31, 17, [1, 2, 3]), img);
    let t = warp_affine(&img, &Affine::translate(3.0, 2.0), 31, 17, [1, 2, 3]);
    assert_eq!(t.pixel(3, 2), img.pixel(0, 0));
    assert_eq!(t.pixel(30, 16), img.pixel(27, 14));
    assert_eq!(t.pixel(0, 0), [1, 2, 3]);
    assert_eq!(t.pixel(2, 5), [1, 2, 3]);
}

#[test]
fn mirror_by_matrix() {
    let img = pattern(20, 10);
    let flip = Affine([-1.0, 0.0, 20.0, 0.0, 1.0, 0.0]);
    let out = warp_affine(&img, &flip, 20, 10, [0; 3]);
    for x in 0..20 {
        assert_eq!(out.pixel(x, 4), img.pixel(19 - x, 4));
    }
}

/// Scale (mirrored sometimes, each axis its own) and translate, no rotation or shear: no vertical step.
fn random_upright(rng: &mut Rng, sw: u32, sh: u32) -> Affine {
    let mut scale = || {
        let s = rng.uniform(0.2, 3.0) as f64;
        if rng.chance(0.2) { -s } else { s }
    };
    let (sx, sy) = (scale(), scale());
    let (tx, ty) = (rng.uniform(-40.0, 90.0) as f64, rng.uniform(-40.0, 90.0) as f64);
    Affine::translate(-(sw as f64) / 2.0, -(sh as f64) / 2.0)
        .then(&Affine::scale(sx, sy))
        .then(&Affine::translate(tx, ty))
}

#[test]
fn row_path_matches_the_general_path() {
    let mut rng = Rng::new(5);
    let images: Vec<Image> =
        [(40, 30), (17, 53), (64, 64), (5, 3), (1, 1)].map(|(w, h)| pattern(w, h)).into();
    for case in 0..400 {
        let img = &images[case % images.len()];
        let m = random_upright(&mut rng, img.width, img.height);
        let inv = m.inverse().unwrap();
        assert_eq!((inv.0[3] * (1 << FRAC) as f64).round(), 0.0);
        let (w, h) = (1 + rng.below(70), 1 + rng.below(50));
        let border = [rng.below(256) as u8, 114, 7];
        let (mut rows, mut general) = (vec![0u8; w * h * 3], vec![0u8; w * h * 3]);
        warp_rows(&Plane::image(img, border), &inv, w, &mut rows);
        warp_general(img, &inv, w, border, &mut general);
        assert_eq!(rows, general, "case {case}: {m:?} into {w}x{h}");
    }
}

#[test]
fn plane_warps_as_its_rendered_image() {
    let mut rng = Rng::new(8);
    let images: Vec<Image> = [(30, 20), (25, 41), (12, 12), (50, 9)].map(|(w, h)| pattern(w, h)).into();
    for case in 0..200 {
        // Four tiles in the quadrants around a centre, cropped by the plane's edges, as a mosaic places them.
        let (pw, ph) = (40 + rng.below(30) as i64, 40 + rng.below(30) as i64);
        let (xc, yc) = (rng.below(pw as usize) as i64, rng.below(ph as usize) as i64);
        let mut tiles = PlaneTiles::new();
        for (k, img) in images.iter().enumerate() {
            let (w, h) = (img.width as i64, img.height as i64);
            let (x0, x1) = if k % 2 == 0 { ((xc - w).max(0), xc) } else { (xc, (xc + w).min(pw)) };
            let (y0, y1) = if k < 2 { ((yc - h).max(0), yc) } else { (yc, (yc + h).min(ph)) };
            if x1 > x0 && y1 > y0 {
                let src = [if k % 2 == 0 { w - (x1 - x0) } else { 0 }, if k < 2 { h - (y1 - y0) } else { 0 }];
                tiles.push(Tile { img, dst: [x0, y0, x1, y1], src });
            }
        }
        let plane = Plane { width: pw as u32, height: ph as u32, fill: [114; 3], tiles };
        let rendered = plane.render();
        let (w, h) = (1 + rng.below(60) as u32, 1 + rng.below(60) as u32);
        let m = if case % 4 == 3 {
            // A rotation takes the general path, through the rendered plane.
            Affine::rotate(rng.uniform(-30.0, 30.0) as f64).then(&Affine::translate(10.0, 5.0))
        } else {
            random_upright(&mut rng, plane.width, plane.height)
        };
        assert_eq!(warp_plane(&plane, &m, w, h), warp_affine(&rendered, &m, w, h, [114; 3]), "case {case}");
        let inv = m.inverse().unwrap();
        let mut general = vec![0u8; (w * h * 3) as usize];
        warp_general(&rendered, &inv, w as usize, [114; 3], &mut general);
        assert_eq!(warp_plane(&plane, &m, w, h).data, general, "case {case}");
    }
}

/// Gap-only `Plane::row` and `Plane::render` match naive `fill_rgb + overlay_tiles` for disjoint, unsorted, and
/// overlapping tiles, including non-grey fill colours.
#[test]
fn plane_row_gap_fill_matches_overlay_reference() {
    let a = pattern(12, 10);
    let b = Image::new(10, 8, [10, 200, 50]);
    let c = Image::new(8, 6, [240, 30, 120]);
    for fill in [[114, 114, 114], [17, 99, 211]] {
        // Unsorted disjoint tiles with gaps before, between, and after.
        let mut disjoint = PlaneTiles::new();
        disjoint.push(Tile { img: &b, dst: [20, 2, 30, 10], src: [0, 0] });
        disjoint.push(Tile { img: &a, dst: [3, 0, 15, 10], src: [0, 0] });
        disjoint.push(Tile { img: &c, dst: [34, 4, 42, 10], src: [0, 0] });
        // Overlapping tiles (later tile wins in overlay order).
        let mut overlapping = PlaneTiles::new();
        overlapping.push(Tile { img: &a, dst: [2, 1, 14, 11], src: [0, 0] });
        overlapping.push(Tile { img: &b, dst: [8, 2, 18, 10], src: [0, 0] });

        for tiles in [disjoint, overlapping] {
            let plane = Plane { width: 48, height: 14, fill, tiles };
            for y in -1..15i64 {
                for &(x0, cols) in &[(-4i64, 56usize), (0, 48), (6, 20), (15, 5)] {
                    let mut got = vec![0xFFu8; cols * 3];
                    plane.row(y, x0, &mut got);
                    let mut want = vec![0u8; cols * 3];
                    fill_rgb(&mut want, fill);
                    plane.overlay_tiles(y, x0, &mut want);
                    assert_eq!(got, want, "fill={fill:?} y={y} x0={x0} cols={cols}");
                }
            }
            let rendered = plane.render();
            let serial = crate::par::serial(|| plane.render());
            assert_eq!(rendered, serial);
        }
    }
}

/// A single tile showing part of a larger image (a mosaic's one visible tile, cropped by the canvas), warped
/// with shear: the warp reads only the shown part, as the rendered plane has it.
#[test]
fn a_cropped_tile_shows_only_its_part() {
    let img = pattern(4, 4);
    let tiles = |dst, src| PlaneTiles::single(Tile { img: &img, dst, src });
    for (dst, src) in [([0, 0, 2, 2], [0, 0]), ([0, 0, 2, 2], [2, 2]), ([0, 0, 3, 2], [1, 0])] {
        let plane =
            Plane { width: dst[2] as u32, height: dst[3] as u32, fill: [114; 3], tiles: tiles(dst, src) };
        for m in [
            Affine([1.0, 0.0, -2.0, 1.0, 1.0, -4.0]),
            Affine([1.0, 0.3, 0.0, 0.2, 1.0, 0.0]),
            Affine::rotate(20.0).then(&Affine::scale(2.0, 2.0)),
        ] {
            for (w, h) in [(1, 1), (5, 4)] {
                assert_eq!(
                    warp_plane(&plane, &m, w, h),
                    warp_affine(&plane.render(), &m, w, h, [114; 3]),
                    "{dst:?} {src:?} {m:?}"
                );
            }
        }
    }
}

/// An empty output is returned at once, whatever the other side's size (no per-row work, no scratch).
#[test]
fn empty_outputs() {
    let src = pattern(1, 1);
    for (w, h) in [(0, u32::MAX), (u32::MAX, 0), (0, 0)] {
        let out = warp_affine(&src, &Affine::IDENTITY, w, h, [1; 3]);
        assert_eq!((out.width, out.height, out.data.len()), (w, h, 0));
    }
    let wide = Image { width: 1 << 30, height: 0, data: Vec::new() };
    assert!(warp_affine(&wide, &Affine::IDENTITY, 1, 0, [1; 3]).data.is_empty());
}

/// An empty plane renders as an empty image and warps to the fill.
#[test]
fn empty_planes() {
    for (pw, ph) in [(0, 0), (0, 5), (5, 0)] {
        let plane = Plane { width: pw, height: ph, fill: [9, 8, 7], tiles: PlaneTiles::new() };
        let r = plane.render();
        assert_eq!((r.width, r.height, r.data.len()), (pw, ph, 0));
        for m in [Affine::translate(1.0, 2.0), Affine::rotate(10.0)] {
            assert_eq!(warp_plane(&plane, &m, 3, 2), Image::new(3, 2, [9, 8, 7]));
        }
    }
}

/// A step so large that `du · (width - 1)` overflows `i64` while every running position the general path adds up
/// fits: the warp gives the general path's bytes, without overflowing.
#[test]
fn a_huge_step_does_not_overflow() {
    let src = pattern(1, 1);
    let m = Affine::scale(2f64.powi(-39), 1.0).then(&Affine::translate(128.5, 0.0));
    let inv = m.inverse().unwrap();
    let mut general = vec![0u8; 257 * 3];
    warp_general(&src, &inv, 257, [114; 3], &mut general);
    assert_eq!(warp_affine(&src, &m, 257, 1, [114; 3]).data, general);
}

/// A step whose sampled positions leave `i64` takes the general path, whose sums wrap (release build) as before:
/// here column 511's position wraps round to -0.5, half border, half image.
#[test]
fn positions_beyond_i64_take_the_general_path() {
    let m = Affine::scale(2f64.powi(-39), 1.0).then(&Affine::translate(-0.5, 0.0));
    let inv = m.inverse().unwrap();
    assert!(!rows_fit(&inv, 512, 1));
    // Steps of more than 4 source pixels per output pixel take the general path too.
    assert!(!rows_fit(&inv, 1, 1));
    assert!(!rows_fit(&Affine::scale(0.2, 1.0).inverse().unwrap(), 10, 1));
    assert!(rows_fit(&Affine::scale(0.25, 1.0).inverse().unwrap(), 10, 1));
    // The sums overflow, which a build with overflow checks stops at; compare the bytes where they wrap.
    #[cfg(not(debug_assertions))]
    {
        let src = Image::new(1, 1, [0; 3]);
        let mut general = vec![0u8; 512 * 3];
        warp_general(&src, &inv, 512, [114; 3], &mut general);
        assert_eq!(&general[511 * 3..], &[57; 3]);
        assert_eq!(warp_affine(&src, &m, 512, 1, [114; 3]).data, general);
    }
}

#[test]
fn parallel_matches_serial() {
    let img = pattern(400, 300);
    let m = Affine::translate(-200.0, -150.0)
        .then(&Affine::rotate(17.0))
        .then(&Affine::scale(1.3, 1.3))
        .then(&Affine::translate(320.0, 320.0));
    let a = warp_affine(&img, &m, 640, 640, [114; 3]);
    let b = crate::par::serial(|| warp_affine(&img, &m, 640, 640, [114; 3]));
    assert_eq!(a, b);
}

#[test]
fn warp_rows_fused_reverse_row_cache_and_paired_general_match_reference() {
    // Multi-tile mosaic with `flip_ud = true` exercises downward `iy` stepping through `PlaneRowCache::prepare`.
    let a = pattern(33, 29);
    let b = pattern(27, 31);
    let c = pattern(19, 23);
    let d = pattern(1, 1);
    let mut tiles = PlaneTiles::new();
    tiles.push(Tile { img: &a, dst: [0, 0, 33, 29], src: [0, 0] });
    tiles.push(Tile { img: &b, dst: [33, 0, 60, 29], src: [0, 2] });
    tiles.push(Tile { img: &c, dst: [14, 29, 33, 52], src: [0, 0] });
    tiles.push(Tile { img: &d, dst: [33, 29, 34, 30], src: [0, 0] });
    let plane = Plane { width: 64, height: 64, fill: [114; 3], tiles };
    let m_upright =
        Affine::translate(-32.0, -32.0).then(&Affine::scale(0.85, 0.9)).then(&Affine::translate(30.0, 28.0));
    let m_rot = Affine::translate(-32.0, -32.0)
        .then(&Affine::rotate(19.0))
        .then(&Affine::shear(3.0, -2.0))
        .then(&Affine::translate(31.0, 29.0));
    for m in [m_upright, m_rot] {
        for &(ow, oh) in &[(1u32, 1u32), (5, 7), (17, 19), (63, 49)] {
            let mut ref_img = warp_plane(&plane, &m, ow, oh);
            let stride = ref_img.stride();
            let h = oh as usize;
            for y in 0..h / 2 {
                let (top, bot) = ref_img.data.split_at_mut((h - 1 - y) * stride);
                top[y * stride..(y + 1) * stride].swap_with_slice(&mut bot[..stride]);
            }
            let area = ow as usize * oh as usize;
            let mut want = vec![0u8; 3 * area];
            for (i, px) in ref_img.data.as_chunks::<3>().0.iter().enumerate() {
                want[i] = px[0];
                want[area + i] = px[1];
                want[2 * area + i] = px[2];
            }
            let mut got = vec![0xEEu8; 3 * area];
            warp_plane_fused(&plane, &m, ow, oh, None, true, false, &mut PlanarDst::from_u8(&mut got, area));
            assert_eq!(got, want, "{ow}x{oh} {m:?}");
        }
    }
}
