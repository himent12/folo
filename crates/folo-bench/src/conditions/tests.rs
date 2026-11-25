//! Condition tests: ids, sizes kept, the severity range, seeded noise, boxes (kept, or moved by `distance`), effects
//! growing with severity, pixel sizes scaling with the long side, occlusion shares, rendering, and the
//! docs/BENCH.md table kept in step with the code.

use super::*;
use folo_api::{BBox, BENCH_SEVERITIES};
use image::RgbImage;

use super::encoding::encode_jpeg_quality;
use crate::Error;

/// A deterministic test scene: a smooth gradient, a checkerboard, fine stripes and two flat rectangles.
pub(crate) fn scene(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 / w.max(1) as f32, y as f32 / h.max(1) as f32);
        let check = ((x / 16 + y / 16) % 2) as f32;
        let stripe = (x % 4 < 2) as u8 as f32;
        let mut p = [
            (40.0 + 150.0 * fx + 40.0 * check) as u8,
            (60.0 + 120.0 * fy + 30.0 * stripe) as u8,
            (90.0 + 80.0 * (1.0 - fx) + 40.0 * check) as u8,
        ];
        if (w / 5..w / 3).contains(&x) && (h / 4..h / 2).contains(&y) {
            p = [230, 40, 30];
        }
        if (w / 2..w * 4 / 5).contains(&x) && (h / 2..h * 5 / 6).contains(&y) {
            p = [20, 30, 200];
        }
        image::Rgb(p)
    })
}

fn bx(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
    BBox { class: 0, x1, y1, x2, y2 }
}

fn mean(img: &RgbImage) -> f64 {
    img.as_raw().iter().map(|&v| v as f64).sum::<f64>() / img.as_raw().len() as f64
}

fn std(img: &RgbImage) -> f64 {
    let m = mean(img);
    (img.as_raw().iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / img.as_raw().len() as f64).sqrt()
}

/// Mean absolute difference between horizontal and vertical neighbours: high-frequency energy.
fn hf(img: &RgbImage) -> f64 {
    let (w, h) = img.dimensions();
    let mut s = 0.0;
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let (p, r, d) = (img.get_pixel(x, y), img.get_pixel(x + 1, y), img.get_pixel(x, y + 1));
            for c in 0..3 {
                s += (p[c] as f64 - r[c] as f64).abs() + (p[c] as f64 - d[c] as f64).abs();
            }
        }
    }
    s / (w as f64 * h as f64)
}

fn diff(a: &RgbImage, b: &RgbImage) -> f64 {
    a.as_raw().iter().zip(b.as_raw()).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>()
        / a.as_raw().len() as f64
}

fn run(c: Condition, s: u8, img: &RgbImage) -> RgbImage {
    apply(c, s, img, &[bx(10.0, 10.0, 60.0, 50.0)], 7).unwrap().0
}

#[test]
fn ids_parse_exactly() {
    for c in Condition::ALL {
        assert_eq!(Condition::parse(c.id()), Some(c));
    }
    assert_eq!(Condition::parse("Haze"), None);
    assert_eq!(Condition::parse(""), None);
}

#[test]
fn every_condition_keeps_the_size_even_for_tiny_and_odd_images() {
    for (w, h) in [(1, 1), (1, 9), (7, 5), (33, 17), (0, 4)] {
        let img = scene(w, h);
        for c in Condition::ALL {
            for s in 1..=BENCH_SEVERITIES {
                let (out, boxes) = apply(c, s, &img, &[bx(0.0, 0.0, 3.0, 3.0)], 1).unwrap();
                assert_eq!(out.dimensions(), (w, h), "{} {s} {w}x{h}", c.id());
                assert_eq!(boxes.len(), 1);
            }
        }
    }
}

#[test]
fn severity_outside_the_range_is_an_error() {
    let img = scene(8, 8);
    for s in [0, 6, 255] {
        assert_eq!(apply(Condition::Haze, s, &img, &[], 0), Err(Error::Severity(s)));
    }
    assert!(render(&[], Condition::Haze, 0, &[], 0).is_err());
}

#[test]
fn same_seed_same_bytes_and_seeds_move_the_noise() {
    let img = scene(96, 64);
    let boxes = [bx(5.0, 5.0, 60.0, 40.0), bx(30.0, 20.0, 90.0, 60.0)];
    for c in Condition::ALL {
        let a = apply(c, 3, &img, &boxes, 11).unwrap();
        assert!(a == apply(c, 3, &img, &boxes, 11).unwrap(), "{}", c.id());
        let b = apply(c, 3, &img, &boxes, 12).unwrap();
        let seeded = matches!(
            c,
            Condition::LowLight | Condition::SensorNoise | Condition::MotionBlur | Condition::Occlusion
        );
        assert_eq!(a.0 != b.0, seeded, "{}: does the seed matter", c.id());
    }
    let s = variant_seed(0, "img_1", Condition::Haze, 3);
    assert_eq!(s, variant_seed(0, "img_1", Condition::Haze, 3));
    let others = [
        variant_seed(1, "img_1", Condition::Haze, 3),
        variant_seed(0, "img_2", Condition::Haze, 3),
        variant_seed(0, "img_1", Condition::Defocus, 3),
        variant_seed(0, "img_1", Condition::Haze, 4),
    ];
    assert!(others.iter().all(|&o| o != s));
}

#[test]
fn boxes_stay_except_under_distance() {
    let img = scene(64, 48);
    let boxes = [bx(5.0, 5.0, 30.0, 20.0), bx(f32::NAN, 0.0, 3.0, 3.0)];
    for c in Condition::ALL {
        let (_, out) = apply(c, 5, &img, &boxes, 3).unwrap();
        if c.moves_boxes() {
            assert_ne!(out[0], boxes[0]);
            assert!(out[1].x1.is_nan());
        } else {
            assert_eq!(out[0], boxes[0], "{}", c.id());
            assert!(out[1].x1.is_nan());
        }
    }
}

#[test]
fn effects_grow_with_severity() {
    // A long side of 480 px: low_resolution's mildest factor (1.5 at 640) still downscales (1.125).
    let img = scene(480, 360);
    let at = |c: Condition| -> Vec<RgbImage> { (1..=5).map(|s| run(c, s, &img)).collect() };
    let falling = |name: &str, v: Vec<f64>, base: f64| {
        assert!(v[0] < base, "{name}: {base} -> {v:?}");
        assert!(v.windows(2).all(|p| p[1] < p[0]), "{name}: {v:?}");
    };
    let rising = |name: &str, v: Vec<f64>, base: f64| {
        assert!(v[0] > base, "{name}: {base} -> {v:?}");
        assert!(v.windows(2).all(|p| p[1] > p[0]), "{name}: {v:?}");
    };
    falling("low_light brightness", at(Condition::LowLight).iter().map(mean).collect(), mean(&img));
    rising("overexposure brightness", at(Condition::Overexposure).iter().map(mean).collect(), mean(&img));
    falling("low_contrast spread", at(Condition::LowContrast).iter().map(std).collect(), std(&img));
    falling("defocus detail", at(Condition::Defocus).iter().map(hf).collect(), hf(&img));
    falling("motion detail", at(Condition::MotionBlur).iter().map(hf).collect(), hf(&img));
    rising("noise", at(Condition::SensorNoise).iter().map(|v| diff(v, &img)).collect(), 0.0);
    falling("haze spread", at(Condition::Haze).iter().map(std).collect(), std(&img));
    let red = |i: &RgbImage| i.pixels().map(|p| p[0] as f64).sum::<f64>();
    falling("cast red", at(Condition::ColorCast).iter().map(red).collect(), red(&img));
    // JPEG's error is not strictly monotonic in quality on synthetic stripes; it is from 1 to 3 to 5.
    let jpeg: Vec<f64> = at(Condition::Jpeg).iter().map(|v| diff(v, &img)).collect();
    rising("jpeg error", vec![jpeg[0], jpeg[2], jpeg[4]], 0.0);
    let sizes: Vec<f64> =
        JPEG_QUALITY.iter().map(|&q| encode_jpeg_quality(&img, q).unwrap().len() as f64).collect();
    falling("jpeg bytes", sizes, encode_jpeg(&img).unwrap().len() as f64);
    falling("low_resolution detail", at(Condition::LowResolution).iter().map(hf).collect(), hf(&img));
    let changed = |i: &RgbImage| i.as_raw().iter().zip(img.as_raw()).filter(|(a, b)| a != b).count() as f64;
    rising("occluded pixels", at(Condition::Occlusion).iter().map(changed).collect(), 0.0);
    let width = |s: u8| {
        let (_, b) = apply(Condition::Distance, s, &img, &[bx(10.0, 10.0, 110.0, 90.0)], 0).unwrap();
        (b[0].x2 - b[0].x1) as f64
    };
    falling("distance box width", (1..=5).map(width).collect(), 100.0);
}

#[test]
fn distance_moves_boxes_exactly_with_the_pixels() {
    // A white rectangle on black; after the condition, the white pixels are exactly the moved box.
    for (w, h, s) in [(640u32, 480u32, 3u8), (320, 200, 5), (300, 300, 1)] {
        let (x1, y1, x2, y2) = (100u32, 40u32, 220u32, 160u32);
        let img = RgbImage::from_fn(w, h, |x, y| {
            image::Rgb(if (x1..x2).contains(&x) && (y1..y2).contains(&y) { [255; 3] } else { [0; 3] })
        });
        let (out, b) =
            apply(Condition::Distance, s, &img, &[bx(x1 as f32, y1 as f32, x2 as f32, y2 as f32)], 0)
                .unwrap();
        // Pixels at least half covered by the rectangle are at least half white.
        let (mut lo, mut hi) = ((u32::MAX, u32::MAX), (0, 0));
        for (x, y, p) in out.enumerate_pixels() {
            if p[0] >= 128 {
                lo = (lo.0.min(x), lo.1.min(y));
                hi = (hi.0.max(x + 1), hi.1.max(y + 1));
            }
        }
        let b = b[0];
        for (got, want) in [(lo.0, b.x1), (lo.1, b.y1), (hi.0, b.x2), (hi.1, b.y2)] {
            assert!((got as f32 - want).abs() <= 0.5, "{w}x{h} s{s}: pixels {lo:?}..{hi:?}, box {b:?}");
        }
    }
    // Exactly: 640 x 480 at scale 0.5 is 320 x 240 at (160, 120), so x 100 lands on 160 + 50.
    let img = scene(640, 480);
    let (_, b) = apply(Condition::Distance, 3, &img, &[bx(100.0, 40.0, 220.0, 160.0)], 0).unwrap();
    assert_eq!(b[0], bx(210.0, 140.0, 270.0, 200.0));
}

#[test]
fn moved_boxes_are_the_boxes_apply_returns() {
    let boxes = [bx(100.0, 40.0, 220.0, 160.0), bx(0.0, 0.0, 33.5, 17.25)];
    for (w, h) in [(640u32, 480u32), (333, 101)] {
        let img = scene(w, h);
        for c in Condition::ALL {
            for s in 1..=5 {
                let (_, want) = apply(c, s, &img, &boxes, 3).unwrap();
                assert_eq!(moved_boxes(c, s, w, h, &boxes).unwrap(), want, "{} {s} {w}x{h}", c.id());
            }
        }
    }
    assert_eq!(moved_boxes(Condition::Distance, 0, 10, 10, &boxes), Err(Error::Severity(0)));
}

#[test]
fn pixel_sizes_scale_with_the_long_side() {
    // A vertical step edge under defocus, motion blur and low resolution: the width of the transition,
    // Σ min(v, 255 − v) / 255 along a row (proportional to the blur's size), doubles with the image.
    let spread = |c: Condition, w: u32| {
        let img = RgbImage::from_fn(w, w / 2, |x, _| image::Rgb(if x < w / 2 { [0; 3] } else { [255; 3] }));
        let out = run(c, 3, &img);
        (0..w).map(|x| out.get_pixel(x, w / 4)[0]).map(|v| v.min(255 - v) as f64 / 255.0).sum::<f64>()
    };
    for c in [Condition::Defocus, Condition::MotionBlur, Condition::LowResolution] {
        let (a, b) = (spread(c, 640), spread(c, 1280));
        assert!(a > 0.5 && (b / a - 2.0).abs() < 0.2, "{}: {a} {b}", c.id());
    }
}

#[test]
fn occlusion_covers_the_share_of_the_box() {
    let img = RgbImage::from_pixel(200, 200, image::Rgb([0, 255, 0]));
    let boxes = [bx(20.0, 20.0, 120.0, 100.0)];
    for s in 1..=5u8 {
        let (out, _) = apply(Condition::Occlusion, s, &img, &boxes, 9).unwrap();
        let covered = out.pixels().filter(|p| p[1] != 255).count() as f32;
        let want = OCCLUSION_AREA[s as usize - 1] * 100.0 * 80.0;
        assert!((covered - want).abs() <= 0.06 * want + 120.0, "s{s}: {covered} vs {want}");
        // Only inside the box.
        for (x, y, p) in out.enumerate_pixels() {
            if p[1] != 255 {
                assert!((20..120).contains(&x) && (20..100).contains(&y));
            }
        }
    }
    // Boxes outside the image or degenerate get no patch and do not panic.
    let odd = [bx(-50.0, -50.0, -10.0, -10.0), bx(f32::NAN, 0.0, 10.0, 10.0), bx(5.0, 5.0, 5.0, 50.0)];
    let (out, _) = apply(Condition::Occlusion, 5, &img, &odd, 9).unwrap();
    assert!(out == img, "a degenerate box got a patch");
}

#[test]
fn render_decodes_applies_and_encodes() {
    let img = scene(120, 80);
    let png = {
        let mut v = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
        v
    };
    let boxes = [bx(10.0, 10.0, 50.0, 40.0)];
    let r = render(&png, Condition::Distance, 3, &boxes, 5).unwrap();
    assert_eq!((r.width, r.height), (120, 80));
    let (want, want_boxes) = apply(Condition::Distance, 3, &img, &boxes, 5).unwrap();
    assert_eq!(r.boxes, want_boxes);
    assert_eq!(r.jpeg, encode_jpeg(&want).unwrap());
    assert_eq!(decode(&r.jpeg).unwrap().dimensions(), (120, 80));
    assert!(matches!(render(b"not an image", Condition::Haze, 1, &[], 0), Err(Error::Decode(_))));
    assert!(matches!(encode_jpeg(&RgbImage::new(0, 3)), Err(Error::Encode(_))));
}

#[test]
fn oversized_backing_buffer_matches_tightly_packed_image() {
    let base = scene(12, 9);
    for extra in [1usize, 3, 12 * 3] {
        let mut padded = base.as_raw().to_vec();
        padded.extend((0..extra).map(|i| (i * 37) as u8));
        let oversized = RgbImage::from_raw(12, 9, padded).expect("RgbImage accepts oversized buffers");
        for c in Condition::ALL {
            let (got, got_boxes) = apply(c, 3, &oversized, &[bx(1.0, 1.0, 8.0, 6.0)], 7).unwrap();
            let (want, want_boxes) = apply(c, 3, &base, &[bx(1.0, 1.0, 8.0, 6.0)], 7).unwrap();
            assert_eq!(got.dimensions(), want.dimensions(), "{} extra={extra}", c.id());
            assert_eq!(got.as_raw(), want.as_raw(), "{} extra={extra}", c.id());
            assert_eq!(got_boxes, want_boxes, "{} extra={extra}", c.id());
        }
    }
}

#[test]
fn docs_table_is_in_step() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/BENCH.md");
    let doc = std::fs::read_to_string(path).expect("docs/BENCH.md").replace("\r\n", "\n");
    let begin = doc.find("<!-- conditions:begin").expect("begin marker");
    let begin = begin + doc[begin..].find('\n').expect("end of the begin marker") + 1;
    let end = doc.find("<!-- conditions:end -->").expect("end marker");
    let want = markdown_table();
    if std::env::var_os("FOLO_BLESS").is_some() {
        std::fs::write(path, format!("{}{want}{}", &doc[..begin], &doc[end..])).unwrap();
        return;
    }
    assert!(
        doc[begin..end] == want,
        "docs/BENCH.md's conditions table differs from crates/folo-bench/src/conditions/mod.rs; regenerate \
         it with `FOLO_BLESS=1 cargo test -p folo-bench docs_table_is_in_step`"
    );
}
