//! Variants and scores must be identical for every rayon thread count: rows are split across threads, never a
//! value, noise is seeded per row, and the evaluation merges fixed chunks in order and seeds each bootstrap
//! resample by its index.

use folo_api::{BBox, BenchDetection};
use folo_bench::conditions::{self, Condition};
use folo_bench::score::{self, Sample};
use folo_core::rng::Rng;
use image::RgbImage;

fn in_pool<T: Send>(threads: usize, f: &(dyn Fn() -> T + Sync)) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

fn assert_reproducible<T: PartialEq + Send>(name: &str, f: &(dyn Fn() -> T + Sync)) {
    let want = in_pool(1, f);
    for threads in [2, 3, 8] {
        assert!(in_pool(threads, f) == want, "{name}: result differs with {threads} threads");
    }
}

#[test]
fn variants_do_not_depend_on_the_thread_count() {
    // Above the parallel threshold (65,536 px), odd sizes so rows split unevenly.
    let mut rng = Rng::seed(1);
    let img = RgbImage::from_fn(401, 299, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x * y) % 256) as u8])
    });
    let boxes: Vec<BBox> = (0..6)
        .map(|_| {
            let (x, y) = (rng.uniform(0.0, 300.0), rng.uniform(0.0, 200.0));
            BBox { class: 0, x1: x, y1: y, x2: x + 80.0, y2: y + 60.0 }
        })
        .collect();
    for c in Condition::ALL {
        assert_reproducible(c.id(), &|| conditions::apply(c, 4, &img, &boxes, 99).unwrap());
    }
}

#[test]
fn scores_do_not_depend_on_the_thread_count() {
    let mut rng = Rng::seed(2);
    let samples: Vec<Sample> = (0..150)
        .map(|_| {
            let truths: Vec<BBox> = (0..4)
                .map(|_| {
                    let (x, y) = (rng.uniform(0.0, 300.0), rng.uniform(0.0, 300.0));
                    BBox { class: rng.below(3) as u32, x1: x, y1: y, x2: x + 40.0, y2: y + 30.0 }
                })
                .collect();
            let detections = truths
                .iter()
                .map(|t| BenchDetection {
                    class: if rng.bernoulli(0.9) { t.class } else { 0 },
                    score: rng.next_f32(),
                    x1: t.x1 + rng.uniform(-6.0, 6.0),
                    y1: t.y1 + rng.uniform(-6.0, 6.0),
                    x2: t.x2 + rng.uniform(-6.0, 6.0),
                    y2: t.y2 + rng.uniform(-6.0, 6.0),
                })
                .collect();
            Sample { truths, detections }
        })
        .collect();
    assert_reproducible("evaluate", &|| score::evaluate(&samples, 3, true, 7));
    assert_reproducible("map", &|| score::map(&samples, 3, true));
}

#[test]
fn all_conditions_handle_edge_image_sizes_deterministically() {
    for (w, h) in [(1, 1), (1, 17), (17, 1), (2, 2), (3, 5), (5, 3)] {
        let img = RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([((x * 37) % 256) as u8, ((y * 53) % 256) as u8, (((x + y) * 19) % 256) as u8])
        });
        let boxes = vec![BBox { class: 0, x1: 0.0, y1: 0.0, x2: w as f32, y2: h as f32 }];
        for c in Condition::ALL {
            for severity in 1..=5 {
                let first = conditions::apply(c, severity, &img, &boxes, 42).unwrap();
                let second = conditions::apply(c, severity, &img, &boxes, 42).unwrap();
                assert_eq!(first.0.dimensions(), (w, h), "{} s{severity} on {w}x{h}", c.id());
                assert_eq!(first, second, "{} s{severity} on {w}x{h} must be deterministic", c.id());
            }
        }
    }
}
