//! Golden values of the engine (`folo_bench::ENGINE_VERSION`): every condition's pixels on a fixed synthetic image,
//! the canonical JPEG of a variant, and one score end to end. They hold on every machine and thread count (the
//! conditions touch only IEEE basic operations, noise is seeded per row): CI runs them on Linux and Windows.
//!
//! A failure here means the engine computes something else than it did. If that is intended (a condition or a
//! definition of the score changed on purpose), bump `ENGINE_VERSION` in `src/lib.rs`, so benchmark fingerprints
//! change with it, and re-bless the values below with the ones the failure prints.

use folo_api::{BBox, BenchDetection};
use folo_bench::score::{self, Sample};
use folo_bench::{Condition, ENGINE_VERSION, conditions};
use image::RgbImage;

const RE_BLESS: &str = "the engine's output changed: if this is intended, bump folo_bench::ENGINE_VERSION and \
                        re-bless tests/golden.rs with the values printed here";

/// FNV-1a of `bytes`: a stable 64-bit digest.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01B3))
}

/// A 96 x 64 image made of integers only: gradients, a checkerboard, stripes and two flat boxes.
fn scene() -> RgbImage {
    RgbImage::from_fn(96, 64, |x, y| {
        let check = ((x / 8 + y / 8) % 2) * 40;
        let stripe = (x % 4 < 2) as u32 * 30;
        let mut p = [
            (40 + x * 150 / 96 + check) as u8,
            (60 + y * 120 / 64 + stripe) as u8,
            (170 - x * 80 / 96 + check) as u8,
        ];
        if (18..34).contains(&x) && (14..30).contains(&y) {
            p = [230, 40, 30];
        }
        if (50..78).contains(&x) && (34..54).contains(&y) {
            p = [20, 30, 200];
        }
        image::Rgb(p)
    })
}

fn boxes() -> Vec<BBox> {
    vec![
        BBox { class: 0, x1: 18.0, y1: 14.0, x2: 34.0, y2: 30.0 },
        BBox { class: 1, x1: 50.0, y1: 34.0, x2: 78.0, y2: 54.0 },
    ]
}

/// The digest of every condition at severity 3 (seed 7), in `Condition::ALL` order.
const PIXELS: [u64; 12] = [
    0x9a565c8bd54b8900,
    0x59d1815c2b5cb27a,
    0x4cc5dd3fddeff079,
    0x6d943f2a79b087c1,
    0x2fc1df023e9b1fbd,
    0x272f34bf3bcf1417,
    0xd98b6417c50deafe,
    0x68df54c181bd9f76,
    0x29f6e3f65b009f46,
    0x4cf4825bb775bf21,
    0xbcc6ee5c745f0932,
    0x6e3a95a0a573828c,
];
/// The digest of the canonical JPEG of `haze` at severity 3 (and of `distance` at 2, which moves the boxes).
const JPEGS: [u64; 2] = [0x3d58ec8c7fc715a4, 0x3efdde2aef8dabec];
/// The total of the hand-built entry of [`entry`], as f32 bits.
const SCORE_BITS: u32 = 0x4240ace5; // 48.168842

#[test]
fn every_condition_has_its_golden_pixels() {
    // The version is bumped together with the golden values: a bump without new values (or the other way round)
    // is a mistake this points at.
    assert_eq!(ENGINE_VERSION, 1, "{RE_BLESS}");
    let img = scene();
    let got: Vec<u64> = Condition::ALL
        .iter()
        .map(|&c| {
            let (out, moved) = conditions::apply(c, 3, &img, &boxes(), 7).unwrap();
            let mut h = fnv(out.as_raw());
            for b in moved {
                for v in [b.x1, b.y1, b.x2, b.y2] {
                    h = fnv(&[h.to_le_bytes(), (v.to_bits() as u64).to_le_bytes()].concat());
                }
            }
            h
        })
        .collect();
    assert_eq!(
        got,
        PIXELS,
        "{RE_BLESS}: [{}]",
        got.iter().map(|h| format!("0x{h:016x}")).collect::<Vec<_>>().join(", ")
    );
}

#[test]
fn the_canonical_jpeg_is_golden() {
    let png = {
        let mut out = Vec::new();
        image::codecs::png::PngEncoder::new(&mut out)
            .write_image(scene().as_raw(), 96, 64, image::ExtendedColorType::Rgb8)
            .unwrap();
        out
    };
    use image::ImageEncoder;
    let got: Vec<u64> = [(Condition::Haze, 3), (Condition::Distance, 2)]
        .iter()
        .map(|&(c, s)| fnv(&conditions::render(&png, c, s, &boxes(), 7).unwrap().jpeg))
        .collect();
    assert_eq!(
        got,
        JPEGS,
        "{RE_BLESS}: [{}]",
        got.iter().map(|h| format!("0x{h:016x}")).collect::<Vec<_>>().join(", ")
    );
}

/// Two images, two classes: right boxes at several IoUs and confidences, a wrong one, a miss.
fn entry() -> Vec<Sample> {
    let t = |class, x1, y1, x2, y2| BBox { class, x1, y1, x2, y2 };
    let d = |class, score, x1, y1, x2, y2| BenchDetection { class, score, x1, y1, x2, y2 };
    vec![
        Sample {
            truths: vec![
                t(0, 0.0, 0.0, 10.0, 10.0),
                t(1, 20.0, 0.0, 40.0, 20.0),
                t(0, 50.0, 50.0, 60.0, 58.0),
            ],
            detections: vec![
                d(0, 0.92, 0.0, 0.0, 10.0, 9.0),
                d(1, 0.81, 21.0, 1.0, 40.0, 19.0),
                d(0, 0.40, 70.0, 70.0, 80.0, 80.0),
            ],
        },
        Sample {
            truths: vec![t(1, 0.0, 0.0, 30.0, 30.0)],
            detections: vec![d(1, 0.66, 2.0, 0.0, 30.0, 27.0), d(0, 0.12, 0.0, 0.0, 30.0, 30.0)],
        },
    ]
}

#[test]
fn a_hand_built_entry_scores_its_golden_total() {
    let e = score::evaluate(&entry(), 2, true, 11);
    let total = score::score(&e, None, true).total;
    assert_eq!(total.to_bits(), SCORE_BITS, "{RE_BLESS}: 0x{:08x} ({total})", total.to_bits());
}
