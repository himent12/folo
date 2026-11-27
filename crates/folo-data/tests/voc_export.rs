//! Integration tests for configurable Pascal VOC XML export (`export_voc`) and perceptual `dHash`
//! near-duplicate auditing (`dhash_bytes`, `near_duplicate_image_count`, brightness/EXIF/VOC round-trip).

mod common;

use common::{TempDir, write, write_png};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{
    AuditOptions, BBox, Dataset, Format, Image, ImageFormat, Split, VocCoordMode, VocExportOptions,
    VocImageMode, VocOptions, detect_format, dhash_bytes, dhash_file, dhash_image, encode, export_voc,
    hamming_distance, open, open_voc_with, parse_voc_xml,
};
use std::path::Path;

fn assert_boxes_close(a: &[BBox], b: &[BBox], tol: f32) {
    assert_eq!(a.len(), b.len(), "box count mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.class, y.class);
        for (p, q) in [(x.cx, y.cx), (x.cy, y.cy), (x.w, y.w), (x.h, y.h)] {
            assert!((p - q).abs() < tol, "{x:?} vs {y:?} (tol={tol})");
        }
    }
}

/// Builds a multi-split YOLO fixture with XML-special class names, tricky filenames (`&`, `'`, spaces,
/// dots, `#`, cross-split case-insensitive collisions), an empty background image, and an unused class.
fn make_tricky_yolo_fixture(root: &Path) {
    write(
        &root.join("data.yaml"),
        "train: images/train\nval: images/val\ntest: images/test\n\
         names:\n  0: \"cat & dog\"\n  1: \"A<B>\"\n  2: \"bird's \\\"nest\\\"\"\n  3: \"unused_class\"\n",
    );

    // Train: 3 images — XML special chars in filename + tricky stem + background image (0 boxes).
    write_png(&root.join("images/train/a&b'c.png"), 40, 20, [200, 10, 10]);
    write(&root.join("labels/train/a&b'c.txt"), "0 0.5 0.5 0.5 0.5\n1 0.25 0.5 0.25 0.5\n");

    write_png(&root.join("images/train/sub/Frame.jpg.png"), 32, 24, [10, 200, 10]);
    write(&root.join("labels/train/sub/Frame.jpg.txt"), "2 0.5 0.5 0.25 0.5\n");

    write_png(&root.join("images/train/#bg scene.png"), 16, 16, [10, 10, 200]);

    // Val: same stem colliding case-insensitively with Train's `a&b'c` + a distinct image.
    write_png(&root.join("images/val/A&B'C.png"), 20, 30, [100, 100, 10]);
    write(&root.join("labels/val/A&B'C.txt"), "1 0.5 0.5 0.4 0.4\n");

    // Test: 1 image with 1 box.
    write_png(&root.join("images/test/t0.png"), 24, 24, [10, 100, 100]);
    write(&root.join("labels/test/t0.txt"), "2 0.5 0.5 0.5 0.25\n");
}

#[test]
fn export_voc_round_trips_copy_link_none_and_xml_special_chars() {
    let t = TempDir::new("voc-export-modes");
    let src = t.join("yolo");
    make_tricky_yolo_fixture(&src);
    let yolo = open(&src).unwrap();
    assert_eq!(yolo.split_sizes(), [3, 1, 1]);
    assert_eq!(yolo.classes, vec!["cat & dog", "A<B>", "bird's \"nest\"", "unused_class"]);

    for (mode, tag) in
        [(VocImageMode::Copy, "copy"), (VocImageMode::Link, "link"), (VocImageMode::None, "none")]
    {
        let out = t.join(format!("voc-{tag}"));
        let opts = VocExportOptions {
            splits: Vec::new(),
            images: mode,
            coord_mode: VocCoordMode::OneBased,
            write_classes_txt: true,
            write_trainval: true,
            folder: Some("Custom & <Folder>".into()),
        };
        let report = export_voc(&yolo, &out, &opts).unwrap();
        assert_eq!(report.root, out);
        assert_eq!(report.image_mode, mode);
        assert_eq!(report.coord_mode, VocCoordMode::OneBased);
        assert_eq!(report.classes, 4);
        assert_eq!(report.splits, [3, 1, 1]);
        assert_eq!(report.boxes, 5);
        assert_eq!(report.exported_splits.len(), 3);
        assert!(report.classes_file.as_ref().is_some_and(|p| p.is_file()));
        assert!(report.trainval_file.as_ref().is_some_and(|p| p.is_file()));

        // Verify per-split report counts and trainval.txt union (train + val = 4 lines).
        let trainval = std::fs::read_to_string(report.trainval_file.as_ref().unwrap()).unwrap();
        assert_eq!(trainval.lines().count(), 4, "{trainval}");

        for s in &report.exported_splits {
            assert!(s.image_set_file.is_file(), "missing {}", s.image_set_file.display());
            match mode {
                VocImageMode::Copy => {
                    assert_eq!((s.copied_images, s.linked_images, s.skipped_images), (s.images, 0, 0));
                }
                VocImageMode::Link => {
                    assert_eq!(s.linked_images + s.copied_images, s.images);
                    assert_eq!(s.skipped_images, 0);
                }
                VocImageMode::None => {
                    assert_eq!((s.copied_images, s.linked_images, s.skipped_images), (0, 0, s.images));
                }
            }
        }

        // Verify XML escaping and zero leftover `.tmp` files.
        for dir in [out.join("Annotations"), out.join("ImageSets/Main")] {
            let leftover: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
                .collect();
            assert!(leftover.is_empty(), "leftover tmp files in {}: {leftover:?}", dir.display());
        }
        let xml_path = out.join("Annotations/a&b'c.xml");
        let raw_xml = std::fs::read_to_string(&xml_path).unwrap();
        assert!(raw_xml.contains("<folder>Custom &amp; &lt;Folder&gt;</folder>"), "{raw_xml}");
        assert!(raw_xml.contains("<filename>a&amp;b&apos;c.png</filename>"), "{raw_xml}");
        assert!(raw_xml.contains("<name>cat &amp; dog</name>"), "{raw_xml}");
        assert!(raw_xml.contains("<name>A&lt;B&gt;</name>"), "{raw_xml}");
        let parsed = parse_voc_xml(&raw_xml).unwrap();
        assert_eq!(parsed.folder.as_deref(), Some("Custom & <Folder>"));
        assert_eq!(parsed.filename.as_deref(), Some("a&b'c.png"));
        assert_eq!(parsed.objects.len(), 2);
        assert_eq!(parsed.objects[0].name, "cat & dog");
        assert_eq!(parsed.objects[1].name, "A<B>");

        if mode == VocImageMode::None {
            assert!(!out.join("JPEGImages").exists(), "none mode must not create JPEGImages/");
            continue;
        }

        // Round-trip via `detect_format` and `folo_data::open`.
        assert_eq!(detect_format(&out, Format::Auto).unwrap(), Format::Voc);
        let reopened = open(&out).unwrap();
        assert_eq!(reopened.classes, yolo.classes, "{tag}: classes.txt must preserve exact order & unused");
        assert_eq!(reopened.split_sizes(), [3, 1, 1]);
        assert!(reopened.report.issues.is_empty(), "{tag}: {:?}", reopened.report.issues);

        for split in Split::ALL {
            let orig = yolo.split(split);
            let back = reopened.split(split);
            assert_eq!(orig.len(), back.len(), "{tag} {split}");
            // Sort both by image size + box count? Entries are in deterministic sorted order matching `open`.
            let mut orig_entries: Vec<_> = orig.entries().iter().collect();
            orig_entries.sort_by_key(|e| e.image.to_string_lossy().replace('\\', "/"));
            for (o, b) in orig_entries.iter().zip(back.entries()) {
                assert_eq!(o.size, b.size, "{tag} {split}");
                assert_boxes_close(&o.boxes, &b.boxes, 1e-4);
            }
        }
    }
}

#[test]
fn export_voc_zero_based_coords_split_filtering_determinism_and_no_classes_txt() {
    let t = TempDir::new("voc-export-det");
    let src = t.join("yolo");
    make_tricky_yolo_fixture(&src);
    let yolo = open(&src).unwrap();

    // 1. Unordered duplicate split selection -> canonical [Train, Test] only, no Val, no trainval.txt.
    let out_a = t.join("a");
    let out_b = t.join("b");
    let opts = VocExportOptions {
        splits: vec![Split::Test, Split::Train, Split::Train],
        images: VocImageMode::Copy,
        coord_mode: VocCoordMode::ZeroBased,
        write_classes_txt: false,
        write_trainval: true, // Val not exported -> trainval.txt must NOT be created.
        folder: None,
    };
    let ra = export_voc(&yolo, &out_a, &opts).unwrap();
    let rb = export_voc(&yolo, &out_b, &opts).unwrap();
    assert_eq!(ra.splits, [3, 0, 1]);
    assert_eq!(rb.splits, [3, 0, 1]);
    assert_eq!(
        ra.exported_splits.iter().map(|s| s.split).collect::<Vec<_>>(),
        vec![Split::Train, Split::Test]
    );
    assert!(ra.classes_file.is_none());
    assert!(ra.trainval_file.is_none());
    assert!(!out_a.join("classes.txt").exists());
    assert!(!out_a.join("ImageSets/Main/val.txt").exists());
    assert!(!out_a.join("ImageSets/Main/trainval.txt").exists());

    // 2. Byte-for-byte deterministic XML and ImageSets output across repeated exports.
    for rel in
        ["ImageSets/Main/train.txt", "ImageSets/Main/test.txt", "Annotations/a&b'c.xml", "Annotations/t0.xml"]
    {
        assert_eq!(std::fs::read(out_a.join(rel)).unwrap(), std::fs::read(out_b.join(rel)).unwrap(), "{rel}");
    }

    // 3. ZeroBased coordinates round-trip accurately via `open_voc_with(..., ZeroBased)`.
    let reopened = open_voc_with(
        &out_a,
        &VocOptions { include_difficult: true, verify_images: true, coord_mode: VocCoordMode::ZeroBased },
    )
    .unwrap();
    assert_eq!(reopened.split_sizes(), [3, 0, 1]);
    assert!(reopened.report.issues.is_empty(), "{:?}", reopened.report.issues);
    // Without `classes.txt`, only used classes from exported splits are discovered in sorted order.
    assert_eq!(reopened.classes, vec!["A<B>", "bird's \"nest\"", "cat & dog"]);
    assert_eq!(reopened.stats().boxes, 4);
}

#[test]
fn export_voc_rejects_invalid_inputs_before_writing() {
    let t = TempDir::new("voc-export-err");
    let src = t.join("yolo");
    write(&src.join("data.yaml"), "train: images/train\nnames: [only]\n");
    write_png(&src.join("images/train/a.png"), 16, 16, [50, 50, 50]);
    write(&src.join("labels/train/a.txt"), "0 0.5 0.5 0.5 0.5\n");
    let ds = open(&src).unwrap();

    // 1. Empty requested split fails before creating `out`.
    let out_empty = t.join("no-val");
    let err =
        export_voc(&ds, &out_empty, &VocExportOptions { splits: vec![Split::Val], ..Default::default() })
            .unwrap_err();
    assert!(err.to_string().contains("requested splits"), "{err}");
    assert!(!out_empty.exists());

    // 2. Non-empty output directory is refused.
    let occupied = t.join("occupied");
    write(&occupied.join("keep.txt"), "hi");
    assert!(export_voc(&ds, &occupied, &VocExportOptions::default()).is_err());

    // 3. Invalid XML 1.0 control char or untrimmed / multiline class name fails before creating `out`.
    for bad_name in ["", " lead", "line1\nline2", "bad\x01char"] {
        let mut bad_ds = ds.clone();
        bad_ds.classes = vec![bad_name.into()];
        let out_bad = t.join("bad-class");
        assert!(export_voc(&bad_ds, &out_bad, &VocExportOptions::default()).is_err());
        assert!(!out_bad.exists(), "must not create dir for bad class {bad_name:?}");
    }

    // 4. Invalid `<folder>` value fails before creating `out`.
    for bad_folder in ["", " padded", "a\nb", "bad\x07dir"] {
        let out_bad = t.join("bad-folder");
        let opts = VocExportOptions { folder: Some(bad_folder.into()), ..Default::default() };
        assert!(export_voc(&ds, &out_bad, &opts).is_err());
        assert!(!out_bad.exists(), "must not create dir for bad folder {bad_folder:?}");
    }

    // 5. Out-of-range class id and NaN box fail before creating `out`.
    let mut bad_cls_ds = ds.clone();
    bad_cls_ds.classes.clear();
    let out_oob = t.join("bad-oob");
    assert!(export_voc(&bad_cls_ds, &out_oob, &VocExportOptions::default()).is_err());
    assert!(!out_oob.exists());

    let mut bad_box_ds = ds.clone();
    let mut bad_entry = bad_box_ds.train.entries()[0].clone();
    bad_entry.boxes = vec![BBox::new(0, f32::NAN, 0.5, 0.5, 0.5)];
    bad_box_ds.train = folo_data::FileDataset::new(Split::Train, ds.classes.clone(), vec![bad_entry]);
    let out_nan = t.join("bad-nan");
    assert!(export_voc(&bad_box_ds, &out_nan, &VocExportOptions::default()).is_err());
    assert!(!out_nan.exists());
}

#[test]
fn export_voc_link_resolves_relative_symlink_target() {
    let t = TempDir::new("voc-export-symlink");
    let blobs = t.join("blobs");
    write_png(&blobs.join("real.png"), 16, 16, [10, 20, 230]);

    let src = t.join("yolo");
    write(&src.join("data.yaml"), "train: images/train\nnames: [thing]\n");
    std::fs::create_dir_all(src.join("images/train")).unwrap();
    let link_path = src.join("images/train/linked.png");

    #[cfg(unix)]
    let link_res = std::os::unix::fs::symlink("../../../blobs/real.png", &link_path);
    #[cfg(windows)]
    let link_res = std::os::windows::fs::symlink_file("..\\..\\..\\blobs\\real.png", &link_path);
    if link_res.is_err() {
        return; // Unprivileged Windows without Developer Mode cannot create symlinks.
    }
    write(&src.join("labels/train/linked.txt"), "0 0.5 0.5 0.5 0.5\n");

    let ds = open(&src).unwrap();
    let out = t.join("voc-link");
    let report =
        export_voc(&ds, &out, &VocExportOptions { images: VocImageMode::Link, ..Default::default() })
            .unwrap();
    assert_eq!(report.splits, [1, 0, 0]);
    let reopened = open(&out).unwrap();
    assert_eq!(reopened.train.len(), 1);
    assert_eq!(&reopened.train.get(0).unwrap().image.data[..3], &[10, 20, 230]);
}

/// Applies an additive brightness shift and multiplicative contrast factor to an RGB image.
fn adjust_brightness_contrast(img: &Image, gain: f32, bias: f32) -> Image {
    let data = img.data.iter().map(|&v| ((v as f32 * gain + bias).round()).clamp(0.0, 255.0) as u8).collect();
    Image { width: img.width, height: img.height, data }
}

/// Wraps a JPEG byte buffer with an EXIF APP1 header specifying `orientation` (1..=8).
fn inject_jpeg_exif_orientation(jpeg: &[u8], orientation: u8) -> Vec<u8> {
    assert!(jpeg.len() >= 2 && jpeg[0] == 0xFF && jpeg[1] == 0xD8);
    // Minimal little-endian TIFF IFD0 with 1 tag: Orientation (0x0112), SHORT (3), count 1, value = orientation.
    let mut app1 = Vec::new();
    app1.extend_from_slice(b"Exif\0\0");
    app1.extend_from_slice(b"II"); // little-endian
    app1.extend_from_slice(&42u16.to_le_bytes());
    app1.extend_from_slice(&8u32.to_le_bytes()); // offset to IFD0
    app1.extend_from_slice(&1u16.to_le_bytes()); // 1 entry
    app1.extend_from_slice(&0x0112u16.to_le_bytes()); // tag Orientation
    app1.extend_from_slice(&3u16.to_le_bytes()); // type SHORT
    app1.extend_from_slice(&1u32.to_le_bytes()); // count 1
    app1.extend_from_slice(&[orientation, 0, 0, 0]); // inline value
    app1.extend_from_slice(&0u32.to_le_bytes()); // next IFD offset = 0

    let len = (app1.len() + 2) as u16;
    let mut out = Vec::with_capacity(jpeg.len() + app1.len() + 4);
    out.extend_from_slice(&jpeg[..2]);
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&app1);
    out.extend_from_slice(&jpeg[2..]);
    out
}

#[test]
fn perceptual_dhash_audit_brightness_exif_exact_dup_cache_and_voc_roundtrip() {
    let t = TempDir::new("dhash-brightness-exif-voc");
    let r = t.path();
    let yolo_dir = r.join("yolo");
    write(&yolo_dir.join("data.yaml"), "train: images/train\nval: images/val\nnames: [obj]\n");

    let synth = SyntheticDataset::new(
        SyntheticConfig { images: 8, imgsz: 160, seed: 2026, ..Default::default() },
        Split::Train,
    )
    .unwrap();
    let scene_a = synth.get(0).unwrap().image;
    let scene_b = synth.get(1).unwrap().image;
    let scene_distinct = synth.get(3).unwrap().image;

    // 1. Scene A in train (PNG) + exact byte copy in train + slightly brightened/contrast-shifted copy in val.
    let a_png = encode(&scene_a, ImageFormat::Png).unwrap();
    std::fs::create_dir_all(yolo_dir.join("images/train")).unwrap();
    std::fs::create_dir_all(yolo_dir.join("images/val")).unwrap();
    std::fs::write(yolo_dir.join("images/train/a.png"), &a_png).unwrap();
    std::fs::write(yolo_dir.join("images/train/a_exact_copy.png"), &a_png).unwrap();
    let a_bright = adjust_brightness_contrast(&scene_a, 1.08, 12.0);
    let a_bright_jpg = encode(&a_bright, ImageFormat::Jpeg(82)).unwrap();
    std::fs::write(yolo_dir.join("images/val/a_bright.jpg"), &a_bright_jpg).unwrap();

    // `dhash_bytes` matches `dhash_image` and `dhash_file` bit for bit.
    let ha = dhash_image(&scene_a);
    assert_eq!(dhash_bytes(&a_png).unwrap(), ha);
    assert_eq!(dhash_file(&yolo_dir.join("images/train/a.png")).unwrap(), ha);
    let ha_bright = dhash_bytes(&a_bright_jpg).unwrap();
    assert!(
        hamming_distance(ha, ha_bright) <= 6,
        "brightness/contrast + JPEG distance {} too large",
        hamming_distance(ha, ha_bright)
    );

    // 2. Scene B upright PNG in train + 180-deg stored JPEG with EXIF orientation=3 in val (visually identical).
    let b_png = encode(&scene_b, ImageFormat::Png).unwrap();
    std::fs::write(yolo_dir.join("images/train/b.png"), &b_png).unwrap();
    let b_rot180 = scene_b.oriented(3);
    let b_exif_jpg = inject_jpeg_exif_orientation(&encode(&b_rot180, ImageFormat::Jpeg(85)).unwrap(), 3);
    std::fs::write(yolo_dir.join("images/val/b_exif.jpg"), &b_exif_jpg).unwrap();
    assert!(
        hamming_distance(dhash_image(&scene_b), dhash_bytes(&b_exif_jpg).unwrap()) <= 4,
        "EXIF-oriented JPEG should match upright PNG"
    );

    // 3. Distinct scene in val must not match anything.
    let d_png = encode(&scene_distinct, ImageFormat::Png).unwrap();
    std::fs::write(yolo_dir.join("images/val/distinct.png"), &d_png).unwrap();
    assert!(hamming_distance(ha, dhash_image(&scene_distinct)) > 10);

    // Export YOLO -> VOC, reopen as VOC, and run perceptual dHash audit end-to-end.
    let ds = open(&yolo_dir).unwrap();
    let voc_dir = r.join("voc");
    export_voc(&ds, &voc_dir, &VocExportOptions::default()).unwrap();
    let voc_ds = open(&voc_dir).unwrap();

    let report = voc_ds.audit_with(&AuditOptions::perceptual(6));
    assert_eq!(report.duplicate_groups.len(), 1);
    assert_eq!(report.duplicate_image_count(), 1);
    assert_eq!(report.near_duplicate_groups.len(), 2, "{:#?}", report.near_duplicate_groups);
    assert_eq!(report.near_duplicate_leakage.len(), 2, "{:#?}", report.near_duplicate_leakage);
    // Group A has 3 images (2 redundant: a, a_exact_copy, a_bright); Group B has 2 images (1 redundant).
    assert_eq!(report.near_duplicate_image_count(), 3);
    assert!(report.has_near_duplicates());
    assert!(report.has_near_duplicate_leakage());
    assert!(report.has_train_val_leakage());
    assert!(!report.is_clean());
}
