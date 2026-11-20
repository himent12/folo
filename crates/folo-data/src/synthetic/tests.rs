//! Golden digests of the generator: a seed gives the same pixels and boxes in every build and on every machine.
//! CI builds its tests in the debug profile on Linux and Windows; the values below came from a release build on
//! Windows, so a build or a system that draws one vertex differently fails here (issue #382: release and debug
//! builds once wrote different images and labels for one seed, through libm).

use super::*;

/// Every shape on every kind of background, in the aspect ratios `vary_aspect` mixes. With the platform's libm,
/// release and debug builds drew images 30 and 45 of these differently.
fn config() -> SyntheticConfig {
    SyntheticConfig {
        images: 48,
        classes: SHAPES.iter().map(|s| s.to_string()).collect(),
        imgsz: 128,
        seed: 7,
        vary_aspect: true,
        ..SyntheticConfig::default()
    }
}

/// FNV-1a of an image's pixels and its boxes' bits.
fn digest(s: &Sample) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01B3);
        }
    };
    eat(&s.image.width.to_le_bytes());
    eat(&s.image.height.to_le_bytes());
    eat(&s.image.data);
    for b in &s.boxes {
        eat(&b.class.to_le_bytes());
        for v in [b.cx, b.cy, b.w, b.h] {
            eat(&v.to_bits().to_le_bytes());
        }
    }
    h
}

const GOLDEN: u64 = 0x924dffb2d56fd5c8;

#[test]
fn the_config_covers_every_shape_and_background() {
    let cfg = config();
    let kinds = cfg.kinds().unwrap();
    let scenes: Vec<Scene> = (0..cfg.images).map(|i| layout(&cfg, &kinds, i)).collect();
    for k in 0..4 {
        assert!(scenes.iter().any(|s| s.background.kind == k), "no background of kind {k}");
    }
    for c in 0..SHAPES.len() as u32 {
        assert!(scenes.iter().any(|s| s.objects.iter().any(|o| o.class == c)), "no {}", SHAPES[c as usize]);
    }
}

#[test]
fn a_seed_draws_the_golden_images() {
    let cfg = config();
    let each: Vec<u64> = (0..cfg.images).map(|i| digest(&generate(&cfg, i).unwrap())).collect();
    let got = each.iter().fold(0u64, |a, &h| a.rotate_left(5) ^ h);
    assert_eq!(
        got, GOLDEN,
        "the generator draws something else: if that is intended, re-bless GOLDEN with 0x{got:016x} (each image: {each:x?})"
    );
}
