//! Pixels against values computed independently from GPUI's shader formulas.

use crate::scenes::{Rng, dsize, image, rect, sp};
use folo_ui::raster::Renderer;
use folo_ui::*;
use std::sync::Arc;

const W: u32 = 128;
const H: u32 = 96;

fn render(prims: Vec<Primitive>, blend: GlyphBlend) -> Framebuffer {
    let mut s = Scene::new();
    for p in prims {
        s.push(p);
    }
    let mut fb = Framebuffer::new(W, H);
    Renderer::with_threads(1).render(&s, blend, &mut fb);
    fb
}

fn full() -> Bounds<ScaledPixels> {
    rect(0., 0., W as f32, H as f32)
}

fn quad(b: Bounds<ScaledPixels>, bg: impl Into<Background>) -> Quad {
    Quad {
        bounds: b,
        clip: full(),
        background: bg.into(),
        border_color: transparent_black(),
        corner_radii: Corners::default(),
        border_widths: Edges::default(),
        border_style: BorderStyle::Solid,
    }
}

fn ch(p: u32) -> [i32; 3] {
    [(p >> 16 & 0xff) as i32, (p >> 8 & 0xff) as i32, (p & 0xff) as i32]
}

fn near(p: u32, want: [f32; 3], what: &str) {
    let got = ch(p);
    for i in 0..3 {
        assert!((got[i] as f32 - want[i]).abs() <= 1.0, "{what}: got {got:?}, want {want:?}");
    }
}

fn white() -> Hsla {
    hsla(0., 0., 1., 1.)
}

#[test]
fn integer_aligned_opaque_quad_is_exact() {
    let fb =
        render(vec![Primitive::Quad(quad(rect(10., 20., 30., 15.), rgb(0x336699)))], GlyphBlend::IDENTITY);
    for y in 0..H {
        for x in 0..W {
            let inside = (10..40).contains(&x) && (20..35).contains(&y);
            assert_eq!(fb.pixel(x, y), if inside { 0x336699 } else { 0 }, "({x}, {y})");
        }
    }
}

#[test]
fn half_pixel_offsets_follow_pixel_centres_and_edge_coverage() {
    // A plain quad takes GPUI's fast path: no anti-aliasing, the pixels whose centres are inside
    // (a centre on the left / top edge counts, on the right / bottom edge does not).
    let fb = render(vec![Primitive::Quad(quad(rect(10.5, 10.5, 10., 10.), white()))], GlyphBlend::IDENTITY);
    assert_eq!(fb.pixel(10, 10), 0xffffff);
    assert_eq!(fb.pixel(19, 19), 0xffffff);
    assert_eq!(fb.pixel(20, 15), 0);
    assert_eq!(fb.pixel(9, 15), 0);
    // A bordered quad runs the SDF: the outer edge pixel at 0.25 px inside has alpha 0.75.
    let mut q = quad(rect(10.25, 10., 40., 30.), white());
    q.border_widths = Edges::all(sp(1.));
    q.border_color = white();
    let fb = render(vec![Primitive::Quad(q)], GlyphBlend::IDENTITY);
    let v = 255. * 0.75;
    near(fb.pixel(10, 25), [v, v, v], "left edge");
    near(fb.pixel(11, 25), [255.; 3], "inside");
    // Right edge at 50.25: pixel 49's centre is 0.75 inside: alpha 1; pixel 50 is not rasterized.
    near(fb.pixel(49, 25), [255.; 3], "right edge");
    assert_eq!(fb.pixel(50, 25), 0);
}

#[test]
fn circle_coverage_sums_to_its_area() {
    let mut q = quad(rect(20., 10., 60., 60.), white());
    q.corner_radii = Corners::all(sp(30.));
    let fb = render(vec![Primitive::Quad(q)], GlyphBlend::IDENTITY);
    let sum: f64 = fb.pixels.iter().map(|&p| (p & 0xff) as f64 / 255.).sum();
    let area = std::f64::consts::PI * 30. * 30.;
    assert!((sum - area).abs() / area < 0.002, "coverage {sum} vs area {area}");
    // Symmetric about the centre.
    assert_eq!(fb.pixel(21, 40), fb.pixel(78, 40));
    assert_eq!(fb.pixel(50, 11), fb.pixel(50, 68));
}

#[test]
fn bordered_rounded_quad_inner_and_outer_edges() {
    // Blue background, 2 px red border, radius 8, left edge at x = 10.25. On the middle row
    // (straight border): pixel 10 is outside by -0.25 -> alpha 0.75, red; pixel 11 is in the border;
    // pixel 12's centre is 0.25 inside the inner edge -> lerp(blue, red, 0.25).
    let mut q = quad(rect(10.25, 10., 40., 30.), rgb(0x0000ff));
    q.border_widths = Edges::all(sp(2.));
    q.border_color = rgb(0xff0000).into();
    q.corner_radii = Corners::all(sp(8.));
    let fb = render(vec![Primitive::Quad(q)], GlyphBlend::IDENTITY);
    near(fb.pixel(10, 25), [255. * 0.75, 0., 0.], "outer edge");
    near(fb.pixel(11, 25), [255., 0., 0.], "border");
    near(fb.pixel(12, 25), [255. * 0.25, 0., 255. * 0.75], "inner edge");
    near(fb.pixel(13, 25), [0., 0., 255.], "background");
    // The rounded corner: the pixel at the corner's very tip is outside the circle.
    assert_eq!(fb.pixel(10, 10), 0);
    // On the 45-degree diagonal of the top-left corner (centre (18.25, 18)): the point at distance
    // 8 from the centre is the outer edge.
    let d = |x: f32, y: f32| ((x - 18.25).powi(2) + (y - 18.).powi(2)).sqrt() - 8.;
    let (x, y) = (12u32, 12u32);
    let alpha = (0.5 - d(x as f32 + 0.5, y as f32 + 0.5)).clamp(0., 1.);
    near(fb.pixel(x, y), [255. * alpha, 0., 0.], "corner edge");
}

#[test]
fn blending_rounds_like_the_gpu() {
    let cases = [
        // (base, overlay rgb, overlay alpha)
        (0x000000u32, 0xffffffu32, 0.5f32),
        (0x640000, 0xc80000, 0.25),
        (0x123456, 0xabcdef, 0.1),
        (0xffffff, 0x000000, 0.35),
    ];
    for (base, over, a) in cases {
        let mut c: Hsla = rgb(over).into();
        c.a = a;
        let fb = render(
            vec![Primitive::Quad(quad(full(), rgb(base))), Primitive::Quad(quad(rect(0., 0., 8., 8.), c))],
            GlyphBlend::IDENTITY,
        );
        let (b, o) = (ch(base), ch(over));
        let want: [f32; 3] = std::array::from_fn(|i| o[i] as f32 * a + b[i] as f32 * (1. - a));
        let got = ch(fb.pixel(1, 1));
        for i in 0..3 {
            // Round to nearest, as a UNORM target.
            assert_eq!(got[i], (want[i] + 0.5).floor() as i32, "{base:#x} {over:#x} {a}");
        }
    }
}

#[allow(clippy::excessive_precision)]
fn oklab(c: [f32; 3]) -> [f32; 3] {
    let lin = c.map(|v| v.powf(1. / 2.2));
    let l = (0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2]).cbrt();
    let m = (0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2]).cbrt();
    let s = (0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2]).cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

#[allow(clippy::excessive_precision)]
fn from_oklab(c: [f32; 3]) -> [f32; 3] {
    let l = c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2];
    let m = c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2];
    let s = c[0] - 0.0894841775 * c[1] - 1.2914855480 * c[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
    .map(|v| v.max(0.).powf(2.2))
}

#[test]
fn gradient_ends_and_middle_in_both_colour_spaces() {
    // 90 degrees runs left to right; 101 px wide so pixel 50's centre is t = 0.5 exactly.
    let (c0, c1) = ([1., 0., 0.], [0., 0., 1.]);
    for space in [ColorSpace::Srgb, ColorSpace::Oklab] {
        let g =
            linear_gradient(90., linear_color_stop(rgb(0xff0000), 0.), linear_color_stop(rgb(0x0000ff), 1.))
                .color_space(space);
        let fb = render(vec![Primitive::Quad(quad(rect(0., 0., 101., 20.), g))], GlyphBlend::IDENTITY);
        for x in [0u32, 50, 100] {
            let t = (x as f32 + 0.5) / 101.;
            let want = match space {
                ColorSpace::Srgb => std::array::from_fn(|i| c0[i] + (c1[i] - c0[i]) * t),
                ColorSpace::Oklab => {
                    let (a, b) = (oklab(c0), oklab(c1));
                    from_oklab(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t))
                }
            };
            near(fb.pixel(x, 10), want.map(|v: f32| v * 255.), &format!("{space} x {x}"));
        }
    }
}

#[test]
fn glyph_masks_with_and_without_correction() {
    let values = [0u8, 1, 64, 128, 200, 255];
    let mask = Arc::new(AlphaMask::new(dsize(values.len() as i32, 1), values.to_vec()));
    let sprite = |color: Hsla| {
        Primitive::MonoSprite(MonoSprite {
            bounds: rect(4., 4., 6., 1.),
            clip: full(),
            color,
            mask: mask.clone(),
            kind: SpriteKind::Glyph,
        })
    };
    // Plain alpha: white over black gives the mask value.
    let fb = render(vec![sprite(white())], GlyphBlend::IDENTITY);
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(ch(fb.pixel(4 + i as u32, 4))[0], v as i32);
    }
    // Corrected, with GPUI's formula worked here: black text on white.
    let g = [0.149_355_4f32, -0.226_006_7, 0.026_559_83, -0.034_591_47];
    let k = 1.0f32;
    let bg = Primitive::Quad(quad(full(), rgb(0xffffff)));
    let fb = render(vec![bg, sprite(hsla(0., 0., 0., 1.))], GlyphBlend::new(g, k));
    for (i, &v) in values.iter().enumerate() {
        let s = v as f32 / 255.;
        let contrast = k * (4. * 0.75f32).clamp(0., 1.); // brightness 0
        let e = s * (contrast + 1.) / (s * contrast + 1.);
        let a = e + e * (1. - e) * (g[1] * e + g[3]); // g.x * b + g.y with b = 0
        near(fb.pixel(4 + i as u32, 4), [255. * (1. - a); 3], &format!("mask {v}"));
    }
}

#[test]
fn one_to_one_image_is_a_copy() {
    let mut rng = Rng(42);
    let img = image(&mut rng, 20, 10, false);
    let fb = render(
        vec![Primitive::Image(Image {
            bounds: rect(30., 40., 20., 10.),
            clip: full(),
            corner_radii: Corners::default(),
            image: img.clone(),
            frame: 0,
            grayscale: false,
            opacity: 1.,
        })],
        GlyphBlend::IDENTITY,
    );
    let bytes = img.as_bytes(0).unwrap();
    for y in 0..10u32 {
        for x in 0..20u32 {
            let i = ((y * 20 + x) * 4) as usize;
            // BGRA source.
            let want = (bytes[i + 2] as u32) << 16 | (bytes[i + 1] as u32) << 8 | bytes[i] as u32;
            assert_eq!(fb.pixel(30 + x, 40 + y), want);
        }
    }
}

#[test]
fn svg_mask_at_two_to_one_is_the_2x2_average() {
    let mut rng = Rng(8);
    let data: Vec<u8> = (0..16 * 16).map(|_| rng.i(0, 256) as u8).collect();
    let mask = Arc::new(AlphaMask::new(dsize(16, 16), data.clone()));
    let fb = render(
        vec![Primitive::MonoSprite(MonoSprite {
            bounds: rect(10., 20., 8., 8.),
            clip: full(),
            color: white(),
            mask,
            kind: SpriteKind::Svg,
        })],
        GlyphBlend::new([0.1, -0.1, 0.05, -0.05], 1.0),
    );
    for y in 0..8usize {
        for x in 0..8usize {
            let at = |dx: usize, dy: usize| data[(2 * y + dy) * 16 + 2 * x + dx] as f32;
            let avg = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4.;
            near(fb.pixel(10 + x as u32, 20 + y as u32), [avg; 3], &format!("({x}, {y})"));
        }
    }
}

#[test]
fn clips_follow_the_top_left_rule() {
    let mut q = quad(full(), white());
    q.clip = rect(10.5, 20., 10., 5.25);
    let fb = render(vec![Primitive::Quad(q)], GlyphBlend::IDENTITY);
    // GPUI's clip distances clip the triangles, so the rasterizer's rule applies: columns whose centre
    // lies in [10.5, 20.5): 10..=19 (centre 10.5 on the left edge in, 20.5 on the right edge out);
    // rows with centre in [20, 25.25): 20..=24.
    for y in 0..H {
        for x in 0..W {
            let inside = (10..=19).contains(&x) && (20..=24).contains(&y);
            assert_eq!(fb.pixel(x, y) != 0, inside, "({x}, {y})");
        }
    }
    // Wholly outside the frame: nothing, no panic.
    let fb =
        render(vec![Primitive::Quad(quad(rect(-500., -500., 100., 100.), white()))], GlyphBlend::IDENTITY);
    assert!(fb.pixels.iter().all(|&p| p == 0));
}

#[test]
fn rounded_image_corners_and_opacity_follow_the_sdf() {
    let mut rng = Rng(31);
    let img = image(&mut rng, 40, 30, false);
    let (ox, oy, w, h, r, opacity) = (10.0f32, 12.0f32, 40.0f32, 30.0f32, 8.0f32, 0.75f32);
    let fb = render(
        vec![Primitive::Image(Image {
            bounds: rect(ox, oy, w, h),
            clip: full(),
            corner_radii: Corners::all(sp(r)),
            image: img.clone(),
            frame: 0,
            grayscale: false,
            opacity,
        })],
        GlyphBlend::IDENTITY,
    );
    let bytes = img.as_bytes(0).unwrap();
    for y in 0..30u32 {
        for x in 0..40u32 {
            // Distance to the rounded rectangle (negative inside), at the pixel centre.
            let (px, py) = (ox + x as f32 + 0.5, oy + y as f32 + 0.5);
            let (cx, cy) = ((px - (ox + w / 2.)).abs() - w / 2. + r, (py - (oy + h / 2.)).abs() - h / 2. + r);
            let d = if cx > 0. && cy > 0. { (cx * cx + cy * cy).sqrt() - r } else { cx.max(cy) - r };
            let a = opacity * (0.5 - d).clamp(0., 1.);
            let i = ((y * 40 + x) * 4) as usize;
            let want = [bytes[i + 2], bytes[i + 1], bytes[i]].map(|c| c as f32 * a);
            near(fb.pixel(10 + x, 12 + y), want, &format!("({x}, {y})"));
        }
    }
}

#[test]
fn a_zoomed_image_resamples_only_what_the_frame_shows() {
    let mut rng = Rng(3);
    let img = image(&mut rng, 16, 16, false);
    let mut s = Scene::new();
    // 160 000 x 160 000 device px: a full-size copy would be 100 GB.
    s.push_image(Image {
        bounds: rect(-80_000., -80_000., 160_000., 160_000.),
        clip: full(),
        corner_radii: Corners::default(),
        image: img,
        frame: 0,
        grayscale: false,
        opacity: 1.,
    });
    let mut r = Renderer::with_threads(1);
    let mut fb = Framebuffer::new(W, H);
    r.render(&s, GlyphBlend::IDENTITY, &mut fb);
    assert!(r.cache_bytes() < (W * H * 4 + 4096) as usize, "{} bytes", r.cache_bytes());
    // The frame shows the middle of the image: texels 7 and 8 blend around the centre.
    assert!(fb.pixels.iter().any(|&p| p != 0));
}
