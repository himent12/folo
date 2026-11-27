//! Integration tests for spec-compliant COCO JSON export (`export_coco`).

mod common;

use common::{TempDir, write, write_png};
use folo_data::{
    BBox, CocoExportOptions, CocoImageMode, Dataset, Format, Split, detect_format, export_coco, open,
    open_coco_ground_truth,
};
use std::path::Path;

/// Builds a small YOLO dataset with train, val, and test splits, including a background image (0 boxes)
/// and same-stem images in distinct subdirectories to exercise collision-safe naming.
fn make_yolo_fixture(root: &Path) {
    write(
        &root.join("data.yaml"),
        "train: images/train\nval: images/val\ntest: images/test\nnames: [cat, dog, bird]\n",
    );

    // Train: 3 images — two colliding case-insensitively across subdirs + one background image with 0 boxes.
    write_png(&root.join("images/train/a/Frame.png"), 40, 20, [200, 10, 10]);
    write(&root.join("labels/train/a/Frame.txt"), "0 0.5 0.5 0.5 0.5\n2 0.25 0.25 0.25 0.5\n");

    write_png(&root.join("images/train/b/frame.png"), 32, 24, [10, 200, 10]);
    write(&root.join("labels/train/b/frame.txt"), "1 0.5 0.5 0.25 0.5\n");

    write_png(&root.join("images/train/bg.png"), 16, 16, [10, 10, 200]);

    // Val: 1 image with 1 box.
    write_png(&root.join("images/val/v0.png"), 20, 30, [100, 100, 10]);
    write(&root.join("labels/val/v0.txt"), "1 0.5 0.5 0.4 0.4\n");

    // Test: 1 image with 1 box.
    write_png(&root.join("images/test/t0.png"), 24, 24, [10, 100, 100]);
    write(&root.join("labels/test/t0.txt"), "2 0.5 0.5 0.5 0.25\n");
}

fn assert_boxes_close(a: &[BBox], b: &[BBox]) {
    assert_eq!(a.len(), b.len(), "box count mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.class, y.class);
        for (p, q) in [(x.cx, y.cx), (x.cy, y.cy), (x.w, y.w), (x.h, y.h)] {
            assert!((p - q).abs() < 1e-4, "{x:?} vs {y:?}");
        }
    }
}

#[test]
fn export_coco_round_trips_all_image_modes_via_open_and_ground_truth() {
    let t = TempDir::new("coco-export-modes");
    let src = t.join("yolo");
    make_yolo_fixture(&src);
    let yolo = open(&src).unwrap();
    assert_eq!(yolo.split_sizes(), [3, 1, 1]);

    for (mode, tag) in [
        (CocoImageMode::Copy, "copy"),
        (CocoImageMode::Link, "link"),
        (CocoImageMode::Reference, "reference"),
    ] {
        let out = t.join(format!("coco-{tag}"));
        let opts = CocoExportOptions {
            splits: Vec::new(),
            images: mode,
            pretty: mode == CocoImageMode::Reference,
            description: Some(format!("test export {tag}")),
        };
        let report = export_coco(&yolo, &out, &opts).unwrap();
        assert_eq!(report.root, out);
        assert_eq!(report.image_mode, mode);
        assert_eq!(report.categories, 3);
        assert_eq!(report.splits, [3, 1, 1]);
        assert_eq!(report.boxes, 5);
        assert_eq!(report.exported_splits.len(), 3);

        // Verify per-split report counts and no leftover temporary files.
        for s in &report.exported_splits {
            assert!(s.annotation_file.is_file(), "missing {}", s.annotation_file.display());
            match mode {
                CocoImageMode::Copy => {
                    assert_eq!((s.copied_images, s.linked_images, s.referenced_images), (s.images, 0, 0));
                }
                CocoImageMode::Link => {
                    assert_eq!(s.linked_images + s.copied_images, s.images);
                    assert_eq!(s.referenced_images, 0);
                }
                CocoImageMode::Reference => {
                    assert_eq!((s.copied_images, s.linked_images, s.referenced_images), (0, 0, s.images));
                }
            }
        }
        if mode == CocoImageMode::Reference {
            assert!(!out.join("images").exists(), "reference mode must not create images/ dir");
        }
        let leftover_tmp: Vec<_> = std::fs::read_dir(out.join("annotations"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftover_tmp.is_empty(), "leftover tmp files: {leftover_tmp:?}");

        // Round-trip via `detect_format` and `folo_data::open`.
        assert_eq!(detect_format(&out, Format::Auto).unwrap(), Format::Coco);
        let reopened = open(&out).unwrap();
        assert_eq!(reopened.classes, yolo.classes);
        assert_eq!(reopened.split_sizes(), [3, 1, 1]);
        assert!(reopened.report.issues.is_empty(), "{tag}: {:?}", reopened.report.issues);

        for split in Split::ALL {
            let orig = yolo.split(split);
            let back = reopened.split(split);
            assert_eq!(orig.len(), back.len(), "{tag} {split}");
            for i in 0..orig.len() {
                assert_eq!(orig.entries()[i].size, back.entries()[i].size, "{tag} {split}[{i}]");
                assert_boxes_close(orig.boxes(i), back.boxes(i));
            }

            // Round-trip via strict `open_coco_ground_truth` (used by `folo val --coco`).
            let gt = open_coco_ground_truth(&out, split).unwrap();
            assert_eq!(
                gt.categories,
                vec![(1, "cat".to_string()), (2, "dog".to_string()), (3, "bird".to_string())]
            );
            assert_eq!(gt.images.len(), orig.len());
            for (idx, im) in gt.images.iter().enumerate() {
                assert_eq!(im.id, (idx + 1) as u64, "1-based deterministic image id");
                assert!(im.path.as_ref().is_some_and(|p| p.is_file()), "resolved image path missing: {im:?}");
                assert!(
                    !im.file_name.contains('\\'),
                    "COCO file_name must use forward slashes: {}",
                    im.file_name
                );
                for ann in &im.annotations {
                    assert!(!ann.iscrowd);
                    assert!(
                        ann.bbox[0] >= 0.0 && ann.bbox[1] >= 0.0 && ann.bbox[2] >= 0.0 && ann.bbox[3] >= 0.0
                    );
                    let expected_area = ann.bbox[2] * ann.bbox[3];
                    assert!((ann.area - expected_area).abs() < 1e-6);
                }
            }
        }
    }
}

#[test]
fn export_coco_split_selection_determinism_and_formatting() {
    let t = TempDir::new("coco-export-det");
    let src = t.join("yolo");
    make_yolo_fixture(&src);
    let yolo = open(&src).unwrap();

    // Unordered duplicate split selection -> canonical [Train, Test] order only, no Val file.
    let out_a = t.join("a");
    let out_b = t.join("b");
    let opts = CocoExportOptions {
        splits: vec![Split::Test, Split::Train, Split::Train],
        images: CocoImageMode::Copy,
        pretty: false,
        description: Some("deterministic-export".into()),
    };
    let ra = export_coco(&yolo, &out_a, &opts).unwrap();
    let rb = export_coco(&yolo, &out_b, &opts).unwrap();
    assert_eq!(ra.splits, [3, 0, 1]);
    assert_eq!(rb.splits, [3, 0, 1]);
    let exported: Vec<Split> = ra.exported_splits.iter().map(|s| s.split).collect();
    assert_eq!(exported, vec![Split::Train, Split::Test]);
    assert!(!out_a.join("annotations/instances_val.json").exists());

    // Byte-for-byte deterministic JSON across repeated exports.
    for split in ["train", "test"] {
        let ja = std::fs::read(out_a.join(format!("annotations/instances_{split}.json"))).unwrap();
        let jb = std::fs::read(out_b.join(format!("annotations/instances_{split}.json"))).unwrap();
        assert_eq!(ja, jb, "non-deterministic JSON for {split}");
        // Compact JSON has no embedded newlines.
        assert!(!ja.contains(&b'\n'), "compact JSON should be single-line");
    }

    // Pretty JSON has newlines and exact `info.description`.
    let out_pretty = t.join("pretty");
    let pretty_opts = CocoExportOptions {
        splits: vec![Split::Val],
        images: CocoImageMode::Reference,
        pretty: true,
        description: Some("custom pretty desc".into()),
    };
    let rp = export_coco(&yolo, &out_pretty, &pretty_opts).unwrap();
    assert_eq!(rp.splits, [0, 1, 0]);
    let raw = std::fs::read_to_string(out_pretty.join("annotations/instances_val.json")).unwrap();
    assert!(raw.contains('\n'), "pretty JSON should contain newlines");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["info"]["description"], "custom pretty desc");
    assert_eq!(v["images"].as_array().unwrap().len(), 1);
    assert_eq!(v["annotations"].as_array().unwrap().len(), 1);
    assert_eq!(v["categories"].as_array().unwrap().len(), 3);
}

#[test]
fn export_coco_rejects_invalid_inputs_before_writing() {
    let t = TempDir::new("coco-export-err");
    let src = t.join("yolo");
    write(&src.join("data.yaml"), "train: images/train\nnames: [only]\n");
    write_png(&src.join("images/train/a.png"), 16, 16, [50, 50, 50]);
    write(&src.join("labels/train/a.txt"), "0 0.5 0.5 0.5 0.5\n");
    let mut ds = open(&src).unwrap();

    // Requesting a split that has no images fails before creating `out`.
    let out_empty = t.join("no-val");
    let err =
        export_coco(&ds, &out_empty, &CocoExportOptions { splits: vec![Split::Val], ..Default::default() })
            .unwrap_err();
    assert!(err.to_string().contains("requested splits"), "{err}");
    assert!(!out_empty.exists(), "must not create output dir on validation error");

    // Non-empty output directory is refused.
    let existing = t.join("occupied");
    write(&existing.join("keep.txt"), "hi");
    assert!(export_coco(&ds, &existing, &CocoExportOptions::default()).is_err());

    // Out-of-range class id fails before creating `out`.
    let mut bad_class_ds = ds.clone();
    bad_class_ds.classes.clear();
    let out_bad_cls = t.join("bad-cls");
    let err = export_coco(&bad_class_ds, &out_bad_cls, &CocoExportOptions::default()).unwrap_err();
    assert!(err.to_string().contains("out of range"), "{err}");
    assert!(!out_bad_cls.exists(), "must not create output dir on invalid class");

    // Non-finite box coordinate fails before creating `out`.
    let mut bad_entry = ds.train.entries()[0].clone();
    bad_entry.boxes = vec![BBox::new(0, f32::NAN, 0.5, 0.5, 0.5)];
    ds.train = folo_data::dataset::FileDataset::new(Split::Train, ds.classes.clone(), vec![bad_entry]);
    let out_nan = t.join("bad-box");
    let err = export_coco(&ds, &out_nan, &CocoExportOptions::default()).unwrap_err();
    assert!(err.to_string().contains("invalid box"), "{err}");
    assert!(!out_nan.exists(), "must not create output dir on invalid box");
}

#[test]
fn export_coco_reference_resolves_unambiguously_even_with_decoy_image() {
    let t = TempDir::new("coco-export-decoy");
    let src = t.join("work/pictures");
    write(&src.join("data.yaml"), "train: .\nnames: [target]\n");
    write_png(&src.join("a.png"), 16, 16, [220, 10, 10]);
    write(&src.join("a.txt"), "0 0.5 0.5 0.5 0.5\n");

    // Place a same-sized decoy at `/work/exports/pictures/a.png` with completely different pixels.
    let out = t.join("work/exports/batch");
    let decoy = t.join("work/exports/pictures/a.png");
    write_png(&decoy, 16, 16, [10, 220, 10]);

    let ds = open(&src).unwrap();
    let report =
        export_coco(&ds, &out, &CocoExportOptions { images: CocoImageMode::Reference, ..Default::default() })
            .unwrap();
    assert_eq!(report.splits, [1, 0, 0]);

    // Both directory open, single-JSON open, and ground-truth open must resolve to the original red image.
    for target in [out.clone(), out.join("annotations/instances_train.json")] {
        let reopened = open(&target).unwrap();
        assert_eq!(reopened.train.len(), 1);
        let sample = reopened.train.get(0).unwrap();
        assert_eq!(
            &sample.image.data[..3],
            &[220, 10, 10],
            "opened decoy instead of original via {}",
            target.display()
        );
    }
    let gt = open_coco_ground_truth(&out, Split::Train).unwrap();
    assert_eq!(gt.images.len(), 1);
    let resolved = std::fs::canonicalize(gt.images[0].path.as_ref().unwrap()).unwrap();
    let expected = std::fs::canonicalize(src.join("a.png")).unwrap();
    assert_eq!(resolved, expected, "ground truth resolved to decoy instead of original");

    // Removing a source image after `open` must cause `Reference` export to fail rather than serialize a dangling path.
    std::fs::remove_file(src.join("a.png")).unwrap();
    let out_missing = t.join("work/exports/missing");
    assert!(
        export_coco(
            &ds,
            &out_missing,
            &CocoExportOptions { images: CocoImageMode::Reference, ..Default::default() }
        )
        .is_err()
    );
}

#[test]
fn export_coco_link_resolves_relative_symlink_target() {
    let t = TempDir::new("coco-export-symlink");
    let blobs = t.join("blobs");
    write_png(&blobs.join("real.png"), 16, 16, [10, 20, 230]);

    let src = t.join("yolo");
    write(&src.join("data.yaml"), "train: images/train\nnames: [thing]\n");
    std::fs::create_dir_all(src.join("images/train")).unwrap();
    let link_path = src.join("images/train/linked.png");

    #[cfg(unix)]
    let link_res = std::os::unix::fs::symlink("../../../blobs/real.png", &link_path);
    #[cfg(windows)]
    let link_res = std::os::windows::fs::symlink_file(r"..\..\..\blobs\real.png", &link_path);
    if let Err(err) = link_res {
        eprintln!("skipped symlink test (platform denied file symlink creation): {err}");
        return;
    }
    write(&src.join("labels/train/linked.txt"), "0 0.5 0.5 0.5 0.5\n");

    let ds = open(&src).unwrap();
    assert_eq!(ds.train.len(), 1);
    let out = t.join("coco-linked");
    let report =
        export_coco(&ds, &out, &CocoExportOptions { images: CocoImageMode::Link, ..Default::default() })
            .unwrap();
    assert_eq!(report.splits, [1, 0, 0]);
    let exported_img = out.join("images/train/linked.png");
    assert!(!std::fs::symlink_metadata(&exported_img).unwrap().file_type().is_symlink());

    let reopened = open(&out).unwrap();
    assert_eq!(reopened.train.len(), 1);
    assert!(reopened.report.issues.is_empty(), "{:?}", reopened.report.issues);
    let sample = reopened.train.get(0).unwrap();
    assert_eq!(&sample.image.data[..3], &[10, 20, 230]);
}
