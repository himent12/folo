//! Pascal VOC XML import, deterministic multi-label stratified splitting, and duplicate / split-leakage auditing.

mod common;

use common::{TempDir, write, write_png};
use folo_data::check::{CheckKind, CheckOptions, check};
use folo_data::convert::convert;
use folo_data::{
    DataError, Dataset, Format, IssueKind, OpenOptions, Split, SplitConfig, VocCoordMode, VocOptions,
    detect_format, open, open_splits, open_voc_with, open_with, parse_voc_xml, split_dataset,
    split_file_dataset, split_indices, write_split,
};

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn voc_xml(filename: &str, size: Option<(u32, u32)>, objects: &[(&str, [f32; 4], bool)]) -> String {
    let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<annotation>\n");
    s.push_str(&format!("  <filename>{filename}</filename>\n"));
    if let Some((w, h)) = size {
        s.push_str(&format!("  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>\n"));
    }
    for (name, [x1, y1, x2, y2], diff) in objects {
        let d = if *diff { 1 } else { 0 };
        s.push_str(&format!(
            "  <object><name>{name}</name><difficult>{d}</difficult>\
             <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>\n"
        ));
    }
    s.push_str("</annotation>\n");
    s
}

#[test]
fn voc_classic_layout_with_imagesets_entities_difficult_one_based_and_roundtrip() {
    let t = TempDir::new("voc-classic");
    let r = t.path();
    write_png(&r.join("JPEGImages/000001.png"), 100, 50, [200, 10, 10]);
    write_png(&r.join("JPEGImages/000002.png"), 80, 40, [10, 200, 10]);
    write_png(&r.join("JPEGImages/000003.png"), 60, 60, [10, 10, 200]);
    write_png(&r.join("JPEGImages/000004.png"), 40, 40, [100, 100, 100]);

    // Classic 1-based inclusive VOC coordinates:
    // - (11, 6, 50, 25) -> 0-based [10, 5, 50, 25] -> cx=0.3, cy=0.3, w=0.4, h=0.4
    // - (1, 1, 100, 50) -> full image [0, 0, 100, 50] -> cx=0.5, cy=0.5, w=1.0, h=1.0
    // - (1, 1, 1, 1) -> 1-pixel box [0, 0, 1, 1] -> NOT dropped!
    write(
        &r.join("Annotations/000001.xml"),
        voc_xml(
            "000001.png",
            Some((100, 50)),
            &[
                ("cat &amp; kitten", [11.0, 6.0, 50.0, 25.0], false),
                ("dog", [1.0, 1.0, 100.0, 50.0], true),
                ("dog", [1.0, 1.0, 1.0, 1.0], false),
            ],
        ),
    );
    // Missing <size>: dimensions probed from image; float coords & CDATA class name.
    write(
        &r.join("Annotations/000002.xml"),
        "\u{feff}<!-- comment -->\n<annotation>\n\
         <filename>000002.png</filename>\n\
         <object><name><![CDATA[dog]]></name><difficult>0</difficult>\
         <bndbox><xmin>9.0</xmin><ymin>5.0</ymin><xmax>48.0</xmax><ymax>24.0</ymax></bndbox></object>\n\
         </annotation>\n",
    );
    // Background image (zero objects).
    write(&r.join("Annotations/000003.xml"), voc_xml("000003.png", Some((60, 60)), &[]));
    // Test image.
    write(
        &r.join("Annotations/000004.xml"),
        voc_xml("000004.png", Some((40, 40)), &[("cat &amp; kitten", [5.0, 5.0, 20.0, 20.0], false)]),
    );

    // Classic VOC ships train.txt, val.txt, trainval.txt, and test.txt; trainval.txt must NOT duplicate train+val.
    write(&r.join("ImageSets/Main/train.txt"), "000001 1\n000002 -1\n");
    write(&r.join("ImageSets/Main/val.txt"), "000003\n");
    write(&r.join("ImageSets/Main/trainval.txt"), "000001\n000002\n000003\n");
    write(&r.join("ImageSets/Main/test.txt"), "000004\n");

    assert_eq!(detect_format(r, Format::Auto).unwrap(), Format::Voc);
    let ds = open(r).unwrap();
    assert_eq!(ds.format, Format::Voc);
    assert_eq!(ds.classes, vec!["cat & kitten", "dog"]);
    assert_eq!(ds.split_sizes(), [2, 1, 1], "{:#?}", ds.report);
    assert!(ds.report.issues.is_empty(), "{:#?}", ds.report);

    // Verify 1-based inclusive box geometry: regular, full-image, and 1-pixel boxes.
    let b0 = ds.train.boxes(0);
    assert_eq!(b0.len(), 3);
    assert_eq!(b0[0].class, 0);
    assert!(approx(b0[0].cx, 0.3) && approx(b0[0].cy, 0.3) && approx(b0[0].w, 0.4) && approx(b0[0].h, 0.4));
    assert_eq!(b0[1].class, 1);
    assert!(approx(b0[1].cx, 0.5) && approx(b0[1].cy, 0.5) && approx(b0[1].w, 1.0) && approx(b0[1].h, 1.0));
    assert_eq!(b0[2].class, 1);
    assert!(approx(b0[2].w, 0.01) && approx(b0[2].h, 0.02));
    assert_eq!(ds.train.image_size(1), Some((80, 40)));
    assert!(ds.val.boxes(0).is_empty());

    // Zero-based coordinate mode via `open_voc_with` drops the (1,1,1,1) box as zero-area and keeps (11..50).
    let zb = open_voc_with(
        r,
        &VocOptions { include_difficult: true, verify_images: true, coord_mode: VocCoordMode::ZeroBased },
    )
    .unwrap();
    assert_eq!(zb.train.boxes(0).len(), 2);
    assert!(approx(zb.train.boxes(0)[0].w, 0.39));

    // Filtering out <difficult>1</difficult> objects via `open_voc_with`.
    let no_diff = open_voc_with(
        r,
        &VocOptions { include_difficult: false, verify_images: true, coord_mode: VocCoordMode::OneBased },
    )
    .unwrap();
    assert_eq!(no_diff.classes, vec!["cat & kitten", "dog"]);
    assert_eq!(no_diff.train.boxes(0).len(), 2);
    assert_eq!(no_diff.train.boxes(0)[0].class, 0);

    // `open_splits` loads only the requested split while keeping consistent classes.
    let val_only = open_splits(r, &OpenOptions::default(), &[Split::Val]).unwrap();
    assert_eq!(val_only.classes, ds.classes);
    assert_eq!(val_only.split_sizes(), [0, 1, 0]);

    // Round-trip VOC -> YOLO -> open, VOC -> COCO -> open, and YOLO -> VOC -> open.
    let yolo_dir = t.join("as_yolo");
    convert(&ds, &yolo_dir, Format::Yolo).unwrap();
    let from_yolo = open(&yolo_dir).unwrap();
    assert_eq!(from_yolo.classes, ds.classes);
    assert_eq!(from_yolo.split_sizes(), ds.split_sizes());

    let coco_dir = t.join("as_coco");
    convert(&ds, &coco_dir, Format::Coco).unwrap();
    let from_coco = open(&coco_dir).unwrap();
    assert_eq!(from_coco.classes, ds.classes);
    assert_eq!(from_coco.split_sizes(), ds.split_sizes());

    let voc2_dir = t.join("as_voc2");
    let conv_voc = convert(&from_yolo, &voc2_dir, Format::Voc).unwrap();
    assert_eq!(conv_voc.splits, ds.split_sizes());
    assert_eq!(detect_format(&voc2_dir, Format::Auto).unwrap(), Format::Voc);
    let from_voc2 = open(&voc2_dir).unwrap();
    assert_eq!(from_voc2.classes, ds.classes);
    assert_eq!(from_voc2.split_sizes(), ds.split_sizes());
    for split in Split::ALL {
        let (a_ds, b_ds) = (ds.split(split), from_voc2.split(split));
        assert_eq!(a_ds.len(), b_ds.len(), "split {split}");
        for i in 0..a_ds.len() {
            assert_eq!(a_ds.boxes(i).len(), b_ds.boxes(i).len(), "split {split} img {i}");
            for (a, b) in a_ds.boxes(i).iter().zip(b_ds.boxes(i)) {
                assert_eq!(a.class, b.class);
                assert!(approx(a.cx, b.cx) && approx(a.cy, b.cy) && approx(a.w, b.w) && approx(a.h, b.h));
            }
        }
    }
}

#[test]
fn voc_convert_roundtrip_handles_spaces_dots_and_hash_filenames() {
    let t = TempDir::new("voc-tricky-names");
    let src = t.join("yolo");
    write(&src.join("data.yaml"), "train: images/train\nval: images/val\nnames: [zebra, apple]\n");
    for (split, name, rgb) in [
        ("train", "street scene.png", [10, 20, 30]),
        ("train", "frame.jpg.png", [40, 50, 60]),
        ("train", "#1.png", [70, 80, 90]),
        ("val", "foo 1.png", [100, 110, 120]),
        ("val", "street scene.png", [130, 140, 150]),
    ] {
        write_png(&src.join(format!("images/{split}/{name}")), 40, 40, rgb);
        let stem = std::path::Path::new(name).file_stem().unwrap().to_string_lossy();
        write(&src.join(format!("labels/{split}/{stem}.txt")), "0 0.5 0.5 0.5 0.5\n");
    }
    let ds = open(&src).unwrap();
    assert_eq!(ds.split_sizes(), [3, 2, 0]);
    let voc_out = t.join("voc");
    let c = convert(&ds, &voc_out, Format::Voc).unwrap();
    assert_eq!(c.splits, [3, 2, 0]);
    assert_eq!(c.boxes, 5);
    let back = open(&voc_out).unwrap();
    assert!(back.report.issues.is_empty(), "{:#?}", back.report);
    assert_eq!(back.classes, vec!["zebra", "apple"], "classes.txt preserves order");
    assert_eq!(back.split_sizes(), [3, 2, 0]);
    assert_eq!(back.stats().boxes, 5);
}

#[test]
fn voc_trainval_fallback_colocated_split_dirs_stem_only_and_unfinished_marker() {
    // 1. trainval.txt fallback + stem-only background image + missing stem.
    let t = TempDir::new("voc-layouts");
    let tv = t.join("trainval");
    write_png(&tv.join("JPEGImages/a.png"), 32, 32, [1, 2, 3]);
    write_png(&tv.join("JPEGImages/bg_only.png"), 32, 32, [4, 5, 6]);
    write(
        &tv.join("Annotations/a.xml"),
        voc_xml("a.png", Some((32, 32)), &[("bird", [2.0, 2.0, 16.0, 16.0], false)]),
    );
    write(&tv.join("ImageSets/Main/trainval.txt"), "a\nbg_only\nmissing_id\n");
    let ds = open(&tv).unwrap();
    assert_eq!(ds.format, Format::Voc);
    assert_eq!(ds.split_sizes(), [2, 0, 0]);
    assert!(ds.train.boxes(1).is_empty(), "stem-only image becomes background");
    assert_eq!(ds.report.count(IssueKind::MissingImage), 1);

    // 2. Colocated Roboflow-style split directories (train/, valid/).
    let rf = t.join("roboflow");
    write_png(&rf.join("train/x.png"), 20, 20, [10, 20, 30]);
    write(
        &rf.join("train/x.xml"),
        voc_xml("x.png", Some((20, 20)), &[("car", [1.0, 1.0, 10.0, 10.0], false)]),
    );
    write_png(&rf.join("valid/y.png"), 20, 20, [40, 50, 60]);
    write(
        &rf.join("valid/y.xml"),
        voc_xml("y.png", Some((20, 20)), &[("car", [5.0, 5.0, 15.0, 15.0], false)]),
    );
    let ds = open(&rf).unwrap();
    assert_eq!(ds.format, Format::Voc);
    assert_eq!(ds.split_sizes(), [1, 1, 0]);

    // 3. Flat colocated folder without ImageSets.
    let flat = t.join("flat");
    write_png(&flat.join("img.png"), 16, 16, [5, 5, 5]);
    write(&flat.join("img.xml"), voc_xml("img", None, &[("cone", [2.0, 2.0, 10.0, 10.0], false)]));
    let ds = open(&flat).unwrap();
    assert_eq!(ds.format, Format::Voc);
    assert_eq!(ds.split_sizes(), [1, 0, 0]);

    // 4. Opening a single XML file directly and format-mismatch errors.
    let single = open(flat.join("img.xml")).unwrap();
    assert_eq!(single.format, Format::Voc);
    assert_eq!(single.split_sizes(), [1, 0, 0]);
    let err = open_with(flat.join("img.xml"), &OpenOptions { format: Format::Yolo, verify_images: true })
        .unwrap_err()
        .to_string();
    assert!(err.contains("is a Pascal VOC .xml file, not YOLO"), "{err}");

    // 5. Random non-VOC XML in a folder is NOT falsely detected as a VOC dataset.
    let other = t.join("not-voc");
    write(&other.join("pom.xml"), "<project><modelVersion>4.0.0</modelVersion></project>");
    assert!(detect_format(&other, Format::Auto).is_err());

    // 6. Unfinished marker refuses VOC root, `Annotations/` dir, and single `.xml` file.
    write(&tv.join(folo_data::unfinished::MARKER), "wip");
    for p in [tv.clone(), tv.join("Annotations"), tv.join("Annotations/a.xml")] {
        assert!(matches!(detect_format(&p, Format::Auto), Err(DataError::Unfinished(_))), "{}", p.display());
        assert!(matches!(open(&p), Err(DataError::Unfinished(_))), "{}", p.display());
        assert!(matches!(open_voc_with(&p, &VocOptions::default()), Err(DataError::Unfinished(_))));
    }
}

#[test]
fn voc_rejects_path_traversal_and_drive_prefix_escape() {
    let t = TempDir::new("voc-traversal");
    write_png(&t.join("outside.png"), 16, 16, [255, 0, 0]);
    let r = t.join("ds");
    write_png(&r.join("JPEGImages/legit.png"), 16, 16, [0, 255, 0]);
    write(
        &r.join("Annotations/legit.xml"),
        voc_xml("legit.png", Some((16, 16)), &[("ok", [1.0, 1.0, 10.0, 10.0], false)]),
    );
    // XML attempting path traversal or Windows drive-relative path.
    write(
        &r.join("Annotations/evil_rel.xml"),
        "<annotation><folder>..</folder><filename>../outside.png</filename>\
         <object><name>bad</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>",
    );
    write(
        &r.join("Annotations/evil_drive.xml"),
        "<annotation><filename>C:outside.png</filename>\
         <object><name>bad</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>",
    );
    write(
        &r.join("ImageSets/Main/train.txt"),
        "legit\n../../outside\n/abs/outside\nC:outside\nevil_rel\nevil_drive\n",
    );
    let ds = open(&r).unwrap();
    // Only legit.png loads; none of the traversal/drive-prefix entries escape to t/outside.png.
    assert_eq!(ds.train.len(), 1);
    assert!(ds.train.entries()[0].image.ends_with("legit.png"));
    assert!(ds.report.issues.iter().any(|i| i.message.contains("unsafe image ID")));
}

#[test]
fn voc_malformed_xml_errors_check_reconciliation_and_shared_file_leakage() {
    let err = parse_voc_xml(
        "<annotation><object><name>a</name><bndbox><xmin>10</xmax></bndbox></object></annotation>",
    )
    .unwrap_err();
    assert!(err.contains("mismatched"), "{err}");
    assert!(parse_voc_xml("<root/>").unwrap_err().contains("expected root element <annotation>"));
    assert!(
        parse_voc_xml(
            "<annotation><object><name>a</name><bndbox><xmin>20</xmin><ymin>0</ymin><xmax>10</xmax><ymax>5</ymax></bndbox></object></annotation>"
        )
        .unwrap_err()
        .contains("max < min")
    );
    assert!(
        parse_voc_xml(
            "<annotation><object><name>a</name><difficult>maybe</difficult><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>"
        )
        .unwrap_err()
        .contains("invalid <difficult>")
    );

    let t = TempDir::new("voc-check");
    let r = t.path();
    write_png(&r.join("JPEGImages/good.png"), 32, 32, [200, 0, 0]);
    write_png(&r.join("JPEGImages/mismatch.png"), 16, 16, [0, 200, 0]);
    write_png(&r.join("JPEGImages/bad.png"), 16, 16, [0, 0, 200]);
    write(
        &r.join("Annotations/good.xml"),
        voc_xml("good.png", Some((32, 32)), &[("cat", [2.0, 2.0, 12.0, 12.0], false)]),
    );
    // Declares 64x64 for a 16x16 image, with a box entirely past x=16 (becomes zero-area & empty image).
    write(
        &r.join("Annotations/mismatch.xml"),
        voc_xml("mismatch.png", Some((64, 64)), &[("cat", [25.0, 2.0, 40.0, 12.0], false)]),
    );
    write(&r.join("Annotations/bad.xml"), "<annotation><object><name>cat</object></annotation>");
    // Put `good` in BOTH train.txt and val.txt to test shared-file leakage advice.
    write(&r.join("ImageSets/Main/train.txt"), "good\nmismatch\nbad\n");
    write(&r.join("ImageSets/Main/val.txt"), "good\n");

    let ds = open(r).unwrap();
    assert_eq!(ds.split_sizes(), [2, 1, 0]);
    assert_eq!(ds.report.count(IssueKind::BrokenLabel), 1);
    assert_eq!(ds.report.count(IssueKind::Other), 1);
    assert!(ds.train.boxes(1).is_empty(), "box outside 16x16 image is dropped on open");

    let rep = check(r, &CheckOptions::default()).unwrap();
    assert_eq!(rep.format, Format::Voc);
    assert!(rep.issues.iter().any(|i| i.kind == CheckKind::BrokenLabel));
    assert!(rep.issues.iter().any(|i| {
        i.kind == CheckKind::Other && i.message.contains("XML says 64x64") && i.message.contains("16x16")
    }));
    // Reconciled diagnostics for mismatch.png match `open`: ZeroSizeBox + EmptyImage.
    assert!(
        rep.issues.iter().any(|i| i.kind == CheckKind::ZeroSizeBox && i.file.ends_with("mismatch.xml")),
        "{:#?}",
        rep.issues
    );
    assert!(
        rep.issues.iter().any(|i| i.kind == CheckKind::EmptyImage && i.file.ends_with("mismatch.png")),
        "{:#?}",
        rep.issues
    );
    // Shared-file leakage advises removing split membership, NOT deleting the file!
    let leak = rep.issues.iter().find(|i| i.kind == CheckKind::DuplicateImage).unwrap();
    assert!(leak.message.contains("shares the same file as train"), "{}", leak.message);
    assert!(
        leak.fix.as_deref().unwrap().contains("from the val split list")
            && !leak.fix.as_deref().unwrap().contains("remove this val copy"),
        "{:?}",
        leak.fix
    );
}

#[test]
fn stratified_split_end_to_end_and_materialize_on_disk() {
    let t = TempDir::new("split-e2e");
    let src = t.join("src");
    write(&src.join("data.yaml"), "train: images/train\nnames: [common, medium, rare]\n");

    // 24 distinct images: 14 common-only, 6 medium+common, 2 rare+common, 2 background.
    for i in 0..24u8 {
        let name = format!("img_{i:02}.png");
        write_png(&src.join("images/train").join(&name), 16, 16, [i * 10, i * 5, 255 - i * 8]);
        let label = match i {
            0..14 => "0 0.5 0.5 0.4 0.4\n",
            14..20 => "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n",
            20..22 => "0 0.2 0.2 0.2 0.2\n2 0.8 0.8 0.2 0.2\n",
            _ => "",
        };
        write(&src.join("labels/train").join(format!("img_{i:02}.txt")), label);
    }

    let ds = open(&src).unwrap();
    assert_eq!(ds.split_sizes(), [24, 0, 0]);

    let cfg = SplitConfig::train_val_test(0.75, 0.125, 0.125, 123);
    let split = ds.split_stratified(&cfg).unwrap();
    assert_eq!(split.split_sizes(), [18, 3, 3]);
    assert_eq!(split.split_sizes(), split_dataset(&ds, &cfg).unwrap().split_sizes());

    // Exact determinism on split indices.
    let idxs_a = split_indices(&ds.train, &cfg).unwrap();
    let idxs_b = split_indices(&ds.train, &cfg).unwrap();
    assert_eq!(idxs_a, idxs_b);

    // Rare class (2 samples) must not be concentrated in a single split.
    let has_class = |f: &folo_data::FileDataset, c: u32| {
        (0..f.len()).filter(|&i| f.boxes(i).iter().any(|b| b.class == c)).count()
    };
    let rare_splits = [has_class(&split.train, 2), has_class(&split.val, 2), has_class(&split.test, 2)];
    assert_eq!(rare_splits.iter().sum::<usize>(), 2);
    assert!(
        rare_splits.iter().all(|&n| n <= 1),
        "rare class should be spread across splits: {rare_splits:?}"
    );

    // split_file_dataset works on a single FileDataset.
    let (tr, va, te) = split_file_dataset(&ds.train, &SplitConfig::train_val(0.75, 0.25, 5)).unwrap();
    assert_eq!((tr.len(), va.len(), te.len()), (18, 6, 0));

    // Materialise the stratified split on disk as YOLO and verify zero leakage on reload.
    let out = t.join("materialized");
    let written = write_split(&ds, &out, Format::Yolo, &SplitConfig::train_val(0.75, 0.25, 9)).unwrap();
    assert_eq!(written.splits, [18, 6, 0]);
    let reloaded = open(&out).unwrap();
    assert_eq!(reloaded.split_sizes(), [18, 6, 0]);
    let audit = reloaded.audit();
    assert!(audit.is_clean(), "{audit:#?}");
    assert!(!audit.has_split_leakage());
    assert!(audit.empty_classes.is_empty());
}

#[test]
fn audit_and_check_detect_duplicates_and_cross_split_leakage() {
    let t = TempDir::new("audit-leakage");
    let r = t.path();
    write(
        &r.join("data.yaml"),
        "train: images/train\nval: images/val\ntest: images/test\nnames: [cat, dog, unused]\n",
    );

    // Distinct train image + intra-split duplicate in train.
    write_png(&r.join("images/train/t1.png"), 16, 16, [10, 20, 30]);
    write(&r.join("labels/train/t1.txt"), "0 0.5 0.5 0.4 0.4\n");
    write_png(&r.join("images/train/t1_copy.png"), 16, 16, [10, 20, 30]);
    write(&r.join("labels/train/t1_copy.txt"), "0 0.5 0.5 0.4 0.4\n");

    // Cross-split train <-> val leakage.
    write_png(&r.join("images/train/leaked.png"), 16, 16, [200, 100, 50]);
    write(&r.join("labels/train/leaked.txt"), "1 0.5 0.5 0.3 0.3\n");
    write_png(&r.join("images/val/leaked_in_val.png"), 16, 16, [200, 100, 50]);
    write(&r.join("labels/val/leaked_in_val.txt"), "1 0.5 0.5 0.3 0.3\n");

    // Cross-split train <-> test leakage + unique val image.
    write_png(&r.join("images/test/leaked_in_test.png"), 16, 16, [200, 100, 50]);
    write(&r.join("labels/test/leaked_in_test.txt"), "1 0.5 0.5 0.3 0.3\n");
    write_png(&r.join("images/val/unique_val.png"), 16, 16, [7, 8, 9]);
    write(&r.join("labels/val/unique_val.txt"), "0 0.5 0.5 0.2 0.2\n");

    let ds = open(r).unwrap();
    let audit = ds.audit();
    assert!(!audit.is_clean());
    assert!(audit.has_duplicates());
    assert!(audit.has_split_leakage());
    assert!(audit.has_train_val_leakage());
    assert_eq!(audit.duplicate_groups.len(), 2);
    assert_eq!(audit.duplicate_image_count(), 3); // 1 intra-train + 2 copies in val/test
    assert_eq!(audit.split_leakage.len(), 1);
    assert_eq!(audit.split_leakage[0].splits, vec![Split::Train, Split::Val, Split::Test]);
    assert_eq!(audit.split_leakage[0].images.len(), 3);
    assert_eq!(audit.empty_classes, vec![2]); // `unused` class has 0 boxes

    // `check::check` also reports cross-split leakage explicitly so CLI/Studio/Desktop surface it immediately.
    let rep = check(r, &CheckOptions::default()).unwrap();
    let dups: Vec<_> = rep.issues.iter().filter(|i| i.kind == CheckKind::DuplicateImage).collect();
    assert_eq!(dups.len(), 3, "{dups:#?}");
    assert!(dups.iter().any(|i| i.file.ends_with("t1_copy.png") && !i.message.contains("split leakage")));
    assert!(dups.iter().any(|i| {
        i.file.ends_with("leaked_in_val.png") && i.message.contains("split leakage (val duplicates train)")
    }));
    assert!(dups.iter().any(|i| {
        i.file.ends_with("leaked_in_test.png") && i.message.contains("split leakage (test duplicates train)")
    }));

    // Audit handles an image deleted after `open` gracefully without failing.
    std::fs::remove_file(r.join("images/val/unique_val.png")).unwrap();
    let after = ds.audit();
    assert_eq!(after.unreadable.len(), 1);
    assert_eq!(after.unreadable[0].split, Split::Val);
}
