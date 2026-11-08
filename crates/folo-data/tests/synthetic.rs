//! The synthetic generator: determinism, label sanity and a round trip through the YOLO reader.

mod common;

use common::TempDir;
use folo_data::synthetic::{SHAPES, SyntheticConfig, SyntheticDataset, generate, write_yolo};
use folo_data::{Dataset, Format, ImageFormat, Split, open};

fn small(seed: u64) -> SyntheticConfig {
    SyntheticConfig { images: 24, imgsz: 96, seed, ..Default::default() }
}

#[test]
fn deterministic_by_seed() {
    let a = generate(&small(1), 5).unwrap();
    assert_eq!(a, generate(&small(1), 5).unwrap());
    assert_ne!(a.image, generate(&small(2), 5).unwrap().image);
    assert_ne!(a.image, generate(&small(1), 6).unwrap().image);
}

#[test]
fn labels_are_sane_and_varied() {
    let cfg = SyntheticConfig {
        images: 120,
        imgsz: 128,
        classes: SHAPES.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let ds = SyntheticDataset::new(cfg.clone(), Split::Train).unwrap();
    assert_eq!(ds.len(), cfg.train_count());
    let stats = ds.stats();
    assert!(stats.class_counts.iter().all(|&c| c > 0), "every class appears: {:?}", stats.class_counts);
    assert!(stats.empty_images > 0, "some empty images");
    assert!(stats.box_size_hist.counts[..3].iter().sum::<u64>() > 0, "some small objects");
    for i in 0..ds.len() {
        let (w, h) = ds.image_size(i).unwrap();
        assert_eq!(w.max(h), 128);
        for b in ds.boxes(i) {
            let [x1, y1, x2, y2] = b.xyxy();
            assert!(x1 >= -1e-6 && y1 >= -1e-6 && x2 <= 1.0 + 1e-6 && y2 <= 1.0 + 1e-6, "{b:?}");
            assert!(b.w * w as f32 >= 2.9 && b.h * h as f32 >= 2.9);
            assert!((b.class as usize) < SHAPES.len());
        }
    }
    let s = ds.get(3).unwrap();
    assert_eq!(s.boxes, ds.boxes(3));
    assert!(
        SyntheticDataset::new(SyntheticConfig { classes: vec!["blob".into()], ..cfg.clone() }, Split::Train)
            .is_err()
    );
    assert!(SyntheticDataset::new(SyntheticConfig { classes: vec![], ..cfg }, Split::Train).is_err());
}

#[test]
fn round_trip_through_the_yolo_reader() {
    let t = TempDir::new("synth");
    let cfg = small(9);
    let written = write_yolo(&cfg, t.path(), ImageFormat::Png).unwrap();
    assert_eq!(written.splits, [cfg.train_count(), cfg.images - cfg.train_count()]);
    let ds = open(t.path()).unwrap();
    assert_eq!(ds.format, Format::Yolo);
    assert_eq!(ds.classes, cfg.classes);
    assert!(ds.report.issues.is_empty(), "{:#?}", ds.report);
    assert_eq!(ds.split_sizes(), [written.splits[0], written.splits[1], 0]);
    for split in [Split::Train, Split::Val] {
        let mem = SyntheticDataset::new(cfg.clone(), split).unwrap();
        let disk = ds.split(split);
        assert_eq!(mem.len(), disk.len());
        for i in 0..mem.len() {
            let (a, b) = (mem.get(i).unwrap(), disk.get(i).unwrap());
            assert_eq!(a.image, b.image, "PNG is lossless");
            assert_eq!(a.boxes.len(), b.boxes.len());
            for (p, q) in a.boxes.iter().zip(&b.boxes) {
                assert_eq!(p.class, q.class);
                for (u, v) in [(p.cx, q.cx), (p.cy, q.cy), (p.w, q.w), (p.h, q.h)] {
                    assert!((u - v).abs() < 2e-6, "label precision");
                }
            }
        }
    }
    let total: usize = ds.stats().boxes as usize;
    assert_eq!(total, written.boxes);
    // JPEG output decodes to the same sizes.
    let t2 = TempDir::new("synth-jpg");
    write_yolo(&cfg, t2.path(), ImageFormat::Jpeg(90)).unwrap();
    let dj = open(t2.path()).unwrap();
    assert_eq!(dj.train.image_size(0), ds.train.image_size(0));
}
