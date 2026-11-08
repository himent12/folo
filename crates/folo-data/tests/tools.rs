//! `check` (dataset validation), `convert` (YOLO <-> COCO) and synthetic class names.

mod common;

use common::{TempDir, write, write_png};
use folo_data::check::{CheckKind, CheckOptions, Severity, check};
use folo_data::convert::convert;
use folo_data::synthetic::{SyntheticConfig, shapes_for, write_yolo};
use folo_data::{Dataset, Format, ImageFormat, Split, detect_format, open};

fn kinds(report: &folo_data::check::CheckReport) -> Vec<(CheckKind, String, Option<usize>)> {
    report
        .issues
        .iter()
        .map(|i| (i.kind, i.file.file_name().unwrap().to_string_lossy().into_owned(), i.line))
        .collect()
}

#[test]
fn check_finds_every_kind_of_problem() {
    let t = TempDir::new("check");
    write(&t.join("data.yaml"), "train: images/train\nval: images/val\nnames: [a, b]\n");
    write_png(&t.join("images/train/ok.png"), 16, 16, [200, 0, 0]);
    write(&t.join("labels/train/ok.txt"), "0 0.5 0.5 0.5 0.5\n");
    // Same bytes as ok.png.
    write_png(&t.join("images/train/copy.png"), 16, 16, [200, 0, 0]);
    write(&t.join("labels/train/copy.txt"), "1 0.5 0.5 0.5 0.5\n");
    write_png(&t.join("images/train/lines.png"), 16, 8, [0, 200, 0]);
    write(
        &t.join("labels/train/lines.txt"),
        "0 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.2 0.2\n0 40 40 10 10\n0 0.5\n0 0.995 0.5 0.02 0.1\n",
    );
    write(&t.join("images/train/corrupt.jpg"), b"\xFF\xD8\xFF\xE0 definitely not a jpeg");
    write_png(&t.join("images/val/empty.png"), 8, 8, [0, 0, 200]);
    let report = check(t.path(), &CheckOptions::default()).unwrap();
    assert_eq!(report.format, Format::Yolo);
    assert_eq!(report.classes, vec!["a", "b"]);
    assert_eq!(report.splits, [4, 1, 0]);
    let got = kinds(&report);
    let want = [
        (CheckKind::CorruptImage, "corrupt.jpg", None),
        (CheckKind::ClassOverflow, "lines.txt", Some(2)),
        (CheckKind::DuplicateLabel, "lines.txt", Some(3)),
        (CheckKind::OutOfRangeBox, "lines.txt", Some(4)),
        (CheckKind::BrokenLabel, "lines.txt", Some(5)),
        (CheckKind::ClippedBox, "lines.txt", Some(6)),
        (CheckKind::DuplicateImage, "ok.png", None),
        (CheckKind::EmptyImage, "empty.png", None),
    ];
    for w in &want {
        let w = (w.0, w.1.to_string(), w.2);
        assert!(got.contains(&w), "missing {w:?} in {got:?}");
    }
    assert_eq!(got.len(), want.len(), "{got:?}");
    // Images are walked in sorted order, so copy.png comes first and ok.png is the duplicate.
    let dup = report.issues.iter().find(|i| i.kind == CheckKind::DuplicateImage).unwrap();
    assert!(dup.message.contains("copy.png"), "{}", dup.message);
    assert!(!report.is_clean());
    assert_eq!(report.count(Severity::Info), 1);
    assert!(report.issues.iter().all(|i| i.fix.is_some() == (i.severity() != Severity::Info)));
    let json = serde_json::to_value(&report.issues[0]).unwrap();
    assert_eq!(json["kind"], "corrupt_image");
}

#[test]
fn synthetic_datasets_check_clean_and_convert_round_trip() {
    let t = TempDir::new("convert");
    let names = vec!["apple".to_string(), "Square".to_string(), "pear".to_string()];
    let classes = shapes_for(&names).unwrap();
    assert_eq!(classes, vec!["circle", "square", "triangle"]);
    let cfg = SyntheticConfig {
        images: 12,
        imgsz: 64,
        seed: 3,
        classes,
        names: names.clone(),
        ..Default::default()
    };
    write_yolo(&cfg, &t.join("yolo"), ImageFormat::Png).unwrap();
    let report = check(t.join("yolo"), &CheckOptions::default()).unwrap();
    assert!(report.is_clean(), "{:?}", report.issues);
    assert_eq!(report.images(), 12);

    let yolo = open(t.join("yolo")).unwrap();
    assert_eq!(yolo.classes, names);
    let coco = convert(&yolo, &t.join("coco"), Format::Coco).unwrap();
    assert_eq!(coco.splits, [yolo.train.len(), yolo.val.len(), 0]);
    assert_eq!(detect_format(t.join("coco"), Format::Auto).unwrap(), Format::Coco);
    let back = open(t.join("coco")).unwrap();
    assert_eq!(back.classes, names);
    assert!(back.report.issues.is_empty(), "{:?}", back.report.issues);
    let yolo2 = convert(&back, &t.join("yolo2"), Format::Yolo).unwrap();
    assert_eq!(yolo2.boxes, coco.boxes);
    let again = open(t.join("yolo2")).unwrap();
    for split in [Split::Train, Split::Val] {
        let (a, b) = (yolo.split(split), again.split(split));
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a.boxes(i).len(), b.boxes(i).len());
            for (x, y) in a.boxes(i).iter().zip(b.boxes(i)) {
                assert_eq!(x.class, y.class);
                for (p, q) in [(x.cx, y.cx), (x.cy, y.cy), (x.w, y.w), (x.h, y.h)] {
                    assert!((p - q).abs() < 1e-4, "{x:?} vs {y:?}");
                }
            }
        }
    }
    // A non-empty output folder is refused.
    assert!(convert(&yolo, &t.join("coco"), Format::Yolo).is_err());
}

#[test]
fn shape_assignment_rules() {
    assert_eq!(shapes_for(&["star", "x"]).unwrap(), vec!["star", "circle"]);
    assert!(shapes_for(&[] as &[&str]).is_err());
    assert!(shapes_for(&["a", "a"]).is_err());
    assert!(shapes_for(&["circle", "CIRCLE"]).is_err());
    assert!(shapes_for(&["a", "b", "c", "d", "e", "f", "g", "h", "i"]).is_err());
    assert!(shapes_for(&[" "]).is_err());
    let cfg = SyntheticConfig { names: vec!["one".into()], ..Default::default() };
    assert!(
        folo_data::synthetic::SyntheticDataset::new(cfg, Split::Train).is_err(),
        "names must match classes"
    );
}

/// A one-image COCO dataset in the `test` split whose JSON declares `declared` as the image size.
fn coco_test_only(t: &TempDir, declared: (u32, u32)) -> std::path::PathBuf {
    let root = t.join("coco");
    write_png(&root.join("images/test/a.png"), 16, 16, [10, 200, 10]);
    let json = serde_json::json!({
        "images": [{ "id": 1, "file_name": "a.png", "width": declared.0, "height": declared.1 }],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 3, "bbox": [2, 2, 8, 8], "iscrowd": 0 }],
        "categories": [{ "id": 3, "name": "thing" }],
    });
    write(&root.join("annotations/instances_test.json"), json.to_string());
    root
}

#[test]
fn check_reports_coco_size_mismatches() {
    let t = TempDir::new("coco-size");
    let root = coco_test_only(&t, (32, 16));
    for decode in [true, false] {
        let report = check(&root, &CheckOptions { decode, ..Default::default() }).unwrap();
        assert_eq!(report.format, Format::Coco);
        let other: Vec<_> = report.issues.iter().filter(|i| i.kind == CheckKind::Other).collect();
        assert_eq!(other.len(), 1, "decode {decode}: {:?}", report.issues);
        assert!(
            other[0].message.contains("32x16") && other[0].message.contains("16x16"),
            "{}",
            other[0].message
        );
        assert!(!report.is_clean());
    }
    let t = TempDir::new("coco-size-ok");
    let report = check(coco_test_only(&t, (16, 16)), &CheckOptions::default()).unwrap();
    assert!(report.is_clean(), "{:?}", report.issues);
}

#[test]
fn convert_refuses_outputs_it_cannot_read_back() {
    let t = TempDir::new("convert-test-only");
    let ds = open(coco_test_only(&t, (16, 16))).unwrap();
    assert_eq!(ds.split_sizes(), [0, 0, 1]);
    let err = convert(&ds, &t.join("yolo"), Format::Yolo).unwrap_err();
    assert!(err.to_string().contains("test images"), "{err}");
    assert!(!t.join("yolo").exists(), "nothing is written");
    // Test-only COCO is fine and reads back.
    let c = convert(&ds, &t.join("coco2"), Format::Coco).unwrap();
    assert_eq!(c.splits, [0, 0, 1]);
    let back = open(t.join("coco2")).unwrap();
    assert_eq!((back.split_sizes(), back.classes.clone()), ([0, 0, 1], vec!["thing".to_string()]));

    // Every image skipped: nothing to convert.
    let t = TempDir::new("convert-empty");
    write(&t.join("data.yaml"), "train: images/train\nnames: [a]\n");
    write(&t.join("images/train/bad.jpg"), b"not an image");
    let ds = open(t.join("data.yaml")).unwrap();
    assert_eq!(ds.split_sizes(), [0, 0, 0]);
    assert!(
        convert(&ds, &t.join("out"), Format::Coco).unwrap_err().to_string().contains("no readable images")
    );
    assert!(!t.join("out").exists());
}

#[test]
fn convert_gives_extensionless_images_a_readable_extension() {
    let t = TempDir::new("convert-ext");
    let root = t.join("coco");
    write_png(&root.join("images/train/plain"), 16, 16, [10, 10, 200]);
    let jpeg =
        folo_data::encode(&folo_data::Image::new(16, 16, [200, 10, 10]), ImageFormat::Jpeg(90)).unwrap();
    write(&root.join("images/train/odd.bin"), jpeg);
    let json = serde_json::json!({
        "images": [
            { "id": 1, "file_name": "plain", "width": 16, "height": 16 },
            { "id": 2, "file_name": "odd.bin", "width": 16, "height": 16 },
        ],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 8, 8], "iscrowd": 0 }],
        "categories": [{ "id": 1, "name": "thing" }],
    });
    write(&root.join("annotations/instances_train.json"), json.to_string());
    let ds = open(&root).unwrap();
    assert_eq!(ds.split_sizes(), [2, 0, 0], "{:?}", ds.report.issues);
    convert(&ds, &t.join("yolo"), Format::Yolo).unwrap();
    assert!(t.join("yolo/images/train/plain.png").is_file());
    assert!(t.join("yolo/images/train/odd.jpg").is_file());
    let back = open(t.join("yolo")).unwrap();
    assert_eq!(back.split_sizes(), [2, 0, 0]);
    assert_eq!(back.stats().boxes, 1);
}
