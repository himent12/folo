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
fn check_finds_label_files_whose_image_is_gone() {
    let t = TempDir::new("check-orphans");
    let ds = t.join("ds");
    write(&ds.join("classes.txt"), "object\n");
    write_png(&ds.join("images/train/kept.png"), 16, 16, [200, 0, 0]);
    write(&ds.join("labels/train/kept.txt"), "0 0.5 0.5 0.3 0.3\n");
    write(&ds.join("labels/train/missing-photo.txt"), "99 0.5 0.5 0.30 0.30\n");
    // Not orphans: class names, a cache, a label whose image the loader skips by extension or matches by case.
    write(&ds.join("labels/train/classes.txt"), "object\n");
    write(&ds.join("labels/train.cache"), "x");
    write(&ds.join("images/train/raw.heic"), "x");
    write(&ds.join("labels/train/raw.txt"), "0 0.5 0.5 0.3 0.3\n");
    write_png(&ds.join("images/train/Upper.png"), 16, 16, [0, 200, 0]);
    write(&ds.join("labels/train/upper.txt"), "0 0.5 0.5 0.3 0.3\n");
    // A split whose last image is gone is not visited (a known limit: no image pairs its label folder any more).
    std::fs::create_dir_all(ds.join("images/val")).unwrap();
    write(&ds.join("labels/val/gone.txt"), "0 0.5 0.5 0.3 0.3\n");
    let report = check(&ds, &CheckOptions::default()).unwrap();
    let orphans: Vec<_> = report.issues.iter().filter(|i| i.kind == CheckKind::MissingImage).collect();
    let names: Vec<_> = orphans.iter().map(|i| i.file.file_name().unwrap().to_string_lossy()).collect();
    assert_eq!(names, ["missing-photo.txt"], "{:?}", report.issues);
    assert!(orphans[0].message.contains("missing-photo in images/train"), "{}", orphans[0].message);
    assert_eq!(
        orphans[0].fix.as_deref(),
        Some(
            "delete the label file, or put its image back in images/train (and in the split's image list, if it has \
             one)"
        )
    );
    assert!(!report.is_clean());

    // A split given as a list names only some images; labels of the others still have their image.
    let listed = t.join("listed");
    write(&listed.join("data.yaml"), "train: train.txt\nnames: [a]\n");
    write(&listed.join("train.txt"), "images/a.png\n");
    for stem in ["a", "b"] {
        write_png(&listed.join(format!("images/{stem}.png")), 16, 16, [0, 0, 200]);
        write(&listed.join(format!("labels/{stem}.txt")), "0 0.5 0.5 0.3 0.3\n");
    }
    let report = check(&listed, &CheckOptions::default()).unwrap();
    assert!(report.is_clean(), "{:?}", report.issues);
    write(&listed.join("labels/c.txt"), "0 0.5 0.5 0.3 0.3\n");
    let report = check(&listed, &CheckOptions::default()).unwrap();
    assert_eq!(kinds(&report), [(CheckKind::MissingImage, "c.txt".to_string(), None)]);
}

/// Label files the dataset reads, or that sit in folders it does not read, are never orphans.
#[test]
fn check_never_calls_a_label_training_reads_an_orphan() {
    let t = TempDir::new("check-orphans-clean");
    let orphans = |ds: &std::path::Path| {
        let report = check(ds, &CheckOptions::default()).unwrap();
        report.issues.into_iter().filter(|i| i.kind == CheckKind::MissingImage).collect::<Vec<_>>()
    };
    // data.yaml reads `curated/`; an empty `images/train` next to archived labels is not part of the dataset.
    let curated = t.join("curated");
    write(&curated.join("data.yaml"), "train: curated/images\nnames: [a]\n");
    write_png(&curated.join("curated/images/a.png"), 16, 16, [200, 0, 0]);
    write(&curated.join("curated/labels/a.txt"), "0 0.5 0.5 0.3 0.3\n");
    std::fs::create_dir_all(curated.join("images/train")).unwrap();
    write(&curated.join("labels/train/old.txt"), "0 0.5 0.5 0.3 0.3\n");
    assert_eq!(orphans(&curated), []);

    // An image inside the label folder reads its label there too.
    let mixed = t.join("mixed");
    write(&mixed.join("data.yaml"), "train: [images/train, labels/train]\nnames: [a]\n");
    write_png(&mixed.join("images/train/a.png"), 16, 16, [200, 0, 0]);
    write_png(&mixed.join("labels/train/b.png"), 16, 16, [0, 200, 0]);
    for stem in ["a", "b"] {
        write(&mixed.join(format!("labels/train/{stem}.txt")), "0 0.5 0.5 0.3 0.3\n");
    }
    assert_eq!(orphans(&mixed), []);

    // An image beside its label is its image, even when the dataset does not list it.
    let beside = t.join("beside");
    write(&beside.join("data.yaml"), "train: [images/train, labels/train/c.png]\nnames: [a]\n");
    write_png(&beside.join("images/train/a.png"), 16, 16, [200, 0, 0]);
    for stem in ["a", "b", "c"] {
        write(&beside.join(format!("labels/train/{stem}.txt")), "0 0.5 0.5 0.3 0.3\n");
    }
    write_png(&beside.join("labels/train/b.png"), 16, 16, [0, 200, 0]);
    write_png(&beside.join("labels/train/c.png"), 16, 16, [0, 0, 200]);
    assert_eq!(orphans(&beside), []);

    // A list naming only a missing image: its folder holds no image, so its label folder is not visited.
    let gone = t.join("gone");
    write(&gone.join("data.yaml"), "train: train.txt\nval: images/val\nnames: [a]\n");
    write(&gone.join("train.txt"), "images/empty/gone.png\n");
    std::fs::create_dir_all(gone.join("images/empty")).unwrap();
    write(&gone.join("labels/empty/other.txt"), "0 0.5 0.5 0.3 0.3\n");
    write_png(&gone.join("images/val/v.png"), 16, 16, [200, 0, 0]);
    let issues = orphans(&gone);
    assert!(issues.iter().all(|i| !i.file.ends_with("other.txt")), "{issues:?}");
    assert!(issues.iter().any(|i| i.file.ends_with("gone.png")), "{issues:?}");
}

/// Makes `link` a junction to the folder `target`; Err with the reason when this file system cannot.
#[cfg(windows)]
fn link_folder(link: &std::path::Path, target: &std::path::Path) -> Result<(), String> {
    let o = std::process::Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .args([link, target].map(|p| p.to_string_lossy().replace('/', "\\")))
        .output()
        .map_err(|e| e.to_string())?;
    o.status.success().then_some(()).ok_or_else(|| {
        format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
            .trim()
            .to_string()
    })
}

/// Makes `link` a symlink to the folder `target`; Err with the reason when this file system cannot.
#[cfg(unix)]
fn link_folder(link: &std::path::Path, target: &std::path::Path) -> Result<(), String> {
    std::os::unix::fs::symlink(target, link).map_err(|e| e.to_string())
}

/// Two links to one label folder are one folder: a label read through either is not an orphan. A label folder
/// linked to a folder outside the dataset is not visited.
#[test]
fn check_treats_linked_label_folders_as_one() {
    let t = TempDir::new("check-orphans-linked");
    let ds = t.join("ds");
    write(&ds.join("shared/a.txt"), "0 0.5 0.5 0.3 0.3\n");
    write(&ds.join("shared/b.txt"), "0 0.5 0.5 0.3 0.3\n");
    write(&ds.join("shared/c.txt"), "0 0.5 0.5 0.3 0.3\n");
    write_png(&ds.join("images/train/a.png"), 16, 16, [200, 0, 0]);
    write_png(&ds.join("images/val/b.png"), 16, 16, [0, 200, 0]);
    write(&ds.join("data.yaml"), "train: images/train\nval: images/val\nnames: [a]\n");
    std::fs::create_dir_all(ds.join("labels")).unwrap();
    for split in ["train", "val"] {
        if let Err(e) = link_folder(&ds.join("labels").join(split), &ds.join("shared")) {
            eprintln!("skipped: cannot link a folder here: {e}");
            return;
        }
    }
    let report = check(&ds, &CheckOptions::default()).unwrap();
    assert_eq!(kinds(&report), [(CheckKind::MissingImage, "c.txt".to_string(), None)]);

    let out = t.join("linked-out");
    write(&t.join("archive/labels/train/a.txt"), "0 0.5 0.5 0.3 0.3\n");
    write(&t.join("archive/labels/train/b.txt"), "0 0.5 0.5 0.3 0.3\n");
    write_png(&out.join("images/train/a.png"), 16, 16, [200, 0, 0]);
    std::fs::create_dir_all(out.join("labels")).unwrap();
    if let Err(e) = link_folder(&out.join("labels/train"), &t.join("archive/labels/train")) {
        eprintln!("skipped: cannot link a folder here: {e}");
        return;
    }
    let report = check(&out, &CheckOptions::default()).unwrap();
    assert!(report.issues.iter().all(|i| i.kind != CheckKind::MissingImage), "{:?}", report.issues);
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

/// Inserts an EXIF orientation APP1 segment right after the JPEG SOI marker (little-endian TIFF).
fn jpeg_with_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
    let mut seg = b"Exif\0\0II\x2A\x00\x08\x00\x00\x00".to_vec();
    seg.extend_from_slice(&1u16.to_le_bytes());
    seg.extend_from_slice(&0x0112u16.to_le_bytes());
    seg.extend_from_slice(&3u16.to_le_bytes());
    seg.extend_from_slice(&1u32.to_le_bytes());
    seg.extend_from_slice(&orientation.to_le_bytes());
    seg.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&seg);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// `check` on COCO evaluates boxes against actual (EXIF-upright) image dimensions, agreeing with `open`:
/// a valid box outside stale/unrotated JSON dimensions is not flagged as `zero_size_box` / `empty_image`,
/// and a box outside the actual image is flagged even when it fits the JSON's larger declared dimensions.
#[test]
fn check_coco_evaluates_boxes_against_actual_image_size() {
    // 1. Actual 64x64, JSON declares 32x32, valid box [40, 40, 16, 16] inside 64x64.
    let t = TempDir::new("coco-actual-larger");
    write_png(&t.join("images/train/a.png"), 64, 64, [10, 200, 10]);
    let json = serde_json::json!({
        "images": [{ "id": 1, "file_name": "a.png", "width": 32, "height": 32 }],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 16, 16], "iscrowd": 0 }],
        "categories": [{ "id": 1, "name": "thing" }],
    });
    write(&t.join("annotations/instances_train.json"), json.to_string());
    let ds = open(t.path()).unwrap();
    assert_eq!(ds.train.boxes(0).len(), 1);
    for decode in [true, false] {
        let report = check(t.path(), &CheckOptions { decode, ..Default::default() }).unwrap();
        let ks: Vec<CheckKind> = report.issues.iter().map(|i| i.kind).collect();
        assert_eq!(ks, vec![CheckKind::Other], "decode {decode}: {:?}", report.issues);
    }

    // 2. EXIF-rotated JPEG (stored 64x32, orientation 6 -> upright 32x64) with unrotated JSON dims 64x32
    //    and a valid upright box [4, 40, 20, 16].
    let t = TempDir::new("coco-exif-size");
    let raw =
        folo_data::encode(&folo_data::Image::new(64, 32, [10, 10, 200]), ImageFormat::Jpeg(90)).unwrap();
    write(&t.join("images/train/rot.jpg"), jpeg_with_exif_orientation(&raw, 6));
    let json = serde_json::json!({
        "images": [{ "id": 1, "file_name": "rot.jpg", "width": 64, "height": 32 }],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 40, 20, 16], "iscrowd": 0 }],
        "categories": [{ "id": 1, "name": "thing" }],
    });
    write(&t.join("annotations/instances_train.json"), json.to_string());
    let ds = open(t.path()).unwrap();
    assert_eq!(ds.train.image_size(0), Some((32, 64)));
    assert_eq!(ds.train.boxes(0).len(), 1);
    for decode in [true, false] {
        let report = check(t.path(), &CheckOptions { decode, ..Default::default() }).unwrap();
        let ks: Vec<CheckKind> = report.issues.iter().map(|i| i.kind).collect();
        assert_eq!(ks, vec![CheckKind::Other], "decode {decode}: {:?}", report.issues);
    }

    // 3. Inverse: actual 64x64, JSON declares 128x128, box [80, 80, 20, 20] lies outside actual image.
    let t = TempDir::new("coco-actual-smaller");
    write_png(&t.join("images/train/a.png"), 64, 64, [200, 10, 10]);
    let json = serde_json::json!({
        "images": [{ "id": 1, "file_name": "a.png", "width": 128, "height": 128 }],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 1, "bbox": [80, 80, 20, 20], "iscrowd": 0 }],
        "categories": [{ "id": 1, "name": "thing" }],
    });
    write(&t.join("annotations/instances_train.json"), json.to_string());
    let ds = open(t.path()).unwrap();
    assert!(ds.train.boxes(0).is_empty());
    for decode in [true, false] {
        let report = check(t.path(), &CheckOptions { decode, ..Default::default() }).unwrap();
        let ks: Vec<CheckKind> = report.issues.iter().map(|i| i.kind).collect();
        assert_eq!(
            ks,
            vec![CheckKind::Other, CheckKind::ZeroSizeBox, CheckKind::EmptyImage],
            "decode {decode}: {:?}",
            report.issues
        );
    }
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
