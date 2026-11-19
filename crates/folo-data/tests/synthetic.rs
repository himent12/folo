//! The synthetic generator: determinism, label sanity and a round trip through the YOLO reader.

mod common;

use common::TempDir;
use folo_data::synthetic::{
    SHAPES, SyntheticConfig, SyntheticDataset, generate, write_yolo, write_yolo_with_progress,
};
use folo_data::{DataError, Dataset, Format, ImageFormat, Split, open};

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

/// A write stopped part way (issue #316: `folo dataset synth` killed early) leaves nothing at the folder that reads
/// as a dataset: the images go to `.<name>.partial` beside it until the dataset is complete.
#[test]
fn nothing_is_at_the_folder_until_the_dataset_is_complete() {
    let t = TempDir::new("synth-staged");
    let dir = t.join("shapes");
    let staging = t.join(".shapes.partial");
    let seen = std::sync::Mutex::new(Vec::new());
    write_yolo_with_progress(&small(4), &dir, ImageFormat::Png, |done| {
        seen.lock().unwrap().push((done, dir.exists(), staging.join(".folo-partial").is_file()));
    })
    .unwrap();
    let seen = seen.into_inner().unwrap();
    assert_eq!(seen.len(), 24);
    assert!(seen.iter().all(|&(_, published, marked)| !published && marked), "{seen:?}");
    assert!(dir.join("data.yaml").is_file() && !staging.exists());
    assert!(!dir.join(".folo-partial").exists());
    assert_eq!(open(&dir).unwrap().split_sizes(), [19, 5, 0]);
}

/// The next write after a stopped one replaces its leftover (the marker is there and nobody holds it), refuses a
/// folder of that name that is not FOLO's without touching it, and replaces an empty folder at the destination.
#[test]
fn a_stopped_writes_leftover_is_replaced_and_nothing_else_is_touched() {
    let t = TempDir::new("synth-leftover");
    let leftover = t.join(".a.partial");
    common::write(&leftover.join(".folo-partial"), "stopped");
    common::write(&leftover.join("images/train/000000.png"), "half");
    write_yolo(&small(5), &t.join("a"), ImageFormat::Png).unwrap();
    assert!(!leftover.exists());
    assert_eq!(open(t.join("a")).unwrap().split_sizes(), [19, 5, 0]);

    let theirs = t.join(".b.partial");
    common::write(&theirs.join("notes.txt"), "mine");
    let err = write_yolo(&small(5), &t.join("b"), ImageFormat::Png).unwrap_err();
    assert!(
        matches!(&err, DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::AlreadyExists)
    );
    assert!(err.to_string().contains("not FOLO's"), "{err}");
    assert_eq!(std::fs::read_to_string(theirs.join("notes.txt")).unwrap(), "mine");
    assert!(!t.join("b").exists());

    std::fs::create_dir(t.join("c")).unwrap();
    write_yolo(&small(5), &t.join("c"), ImageFormat::Png).unwrap();
    assert!(t.join("c/data.yaml").is_file());
}

/// While one process writes a folder, a second write of it is refused and leaves the first one's files alone.
#[test]
fn a_folder_being_written_is_not_taken_over() {
    let t = TempDir::new("synth-busy");
    let dir = t.join("shapes");
    let busy = folo_data::staging::Staging::begin(&dir).unwrap();
    let err = write_yolo(&small(6), &dir, ImageFormat::Png).unwrap_err();
    assert!(err.to_string().contains("another process is writing"), "{err}");
    assert!(busy.path().join(".folo-partial").is_file());
    drop(busy);
    assert!(!t.join(".shapes.partial").exists() && !dir.exists());
}
