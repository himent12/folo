//! YOLO and COCO parsing, auto-detection, robustness and EXIF orientation. Fixtures are written by the tests.

mod common;

use common::*;
use folo_data::{
    BBox, Dataset, Format, Image, ImageFormat, IssueKind, OpenOptions, Split, encode, jpeg_exif_orientation,
    open, open_splits, open_with, probe_size, read_image,
};

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn yolo_yaml_with_lists_segments_and_broken_files() {
    let t = TempDir::new("yolo");
    let r = t.path();
    write(
        &r.join("data.yaml"),
        "path: .\ntrain: images/train\nval: val.txt\ntest:\nnames: ['cat', 'dog', \"bird\"]\n",
    );
    // train: good box, polygon, background (no label file), broken token, class out of range, corrupt image.
    write_png(&r.join("images/train/a.png"), 40, 20, [10, 10, 10]);
    write(&r.join("labels/train/a.txt"), "0 0.5 0.5 0.5 0.5\n1 0.25 0.5 0.4 0.5\n");
    write_png(&r.join("images/train/b.png"), 30, 30, [10, 10, 10]);
    write(&r.join("labels/train/b.txt"), "2 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n");
    write_png(&r.join("images/train/c.png"), 30, 30, [10, 10, 10]);
    write_png(&r.join("images/train/d.png"), 30, 30, [10, 10, 10]);
    write(&r.join("labels/train/d.txt"), "0 0.5 0.5 0.2 zz\n");
    write_png(&r.join("images/train/e.png"), 30, 30, [10, 10, 10]);
    write(&r.join("labels/train/e.txt"), "5 0.5 0.5 0.2 0.2\n");
    write(&r.join("images/train/f.jpg"), b"\xFF\xD8 definitely not a jpeg");
    write(&r.join("labels/train/f.txt"), "0 0.5 0.5 0.2 0.2\n");
    // val: an image list with ./ and root-relative entries, one missing.
    write_png(&r.join("images/val/v1.png"), 16, 16, [0, 0, 0]);
    write(&r.join("labels/val/v1.txt"), "1 0.5 0.5 1 1\n1 0.5 0.5 1 1\n");
    write_png(&r.join("images/val/v2.png"), 16, 16, [0, 0, 0]);
    write(&r.join("val.txt"), "./images/val/v1.png\nimages/val/v2.png\nimages/val/missing.png\n");

    let ds = open(r).unwrap();
    assert_eq!(ds.format, Format::Yolo);
    assert_eq!(ds.classes, ["cat", "dog", "bird"]);
    assert_eq!(ds.split_sizes(), [3, 2, 0]);
    let rep = &ds.report;
    assert_eq!(rep.broken_labels(), 2, "{rep:#?}");
    assert_eq!(rep.count(IssueKind::CorruptImage), 1);
    assert_eq!(rep.count(IssueKind::MissingImage), 1);
    assert_eq!(rep.count(IssueKind::LabelWarning), 1, "duplicate in v1");

    let train = &ds.train;
    let names: Vec<_> = train.entries().iter().map(|e| e.image.file_name().unwrap().to_owned()).collect();
    assert_eq!(names, ["a.png", "b.png", "c.png"]);
    assert_eq!(train.boxes(0), &[BBox::new(0, 0.5, 0.5, 0.5, 0.5), BBox::new(1, 0.25, 0.5, 0.4, 0.5)]);
    let poly = train.boxes(1)[0];
    assert!(
        poly.class == 2
            && approx(poly.cx, 0.3)
            && approx(poly.cy, 0.4)
            && approx(poly.w, 0.4)
            && approx(poly.h, 0.4)
    );
    assert!(train.boxes(2).is_empty());
    assert_eq!(train.image_size(0), Some((40, 20)));
    let s = train.get(0).unwrap();
    assert_eq!((s.image.width, s.image.height), (40, 20));
    assert_eq!(ds.val.boxes(0).len(), 1);

    let stats = ds.stats();
    assert_eq!(stats.boxes, 4);
    assert_eq!(stats.empty_images, 2);
    assert_eq!(stats.broken_labels, 2);
    assert_eq!(stats.class_counts, vec![1, 2, 1]);
    assert_eq!(stats.boxes_per_image_hist.counts, vec![2, 2, 1, 0, 0, 0]);
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("\"box_size_hist\":{\"edges\":[0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0]"), "{json}");
}

#[test]
fn yolo_names_map_roboflow_paths_and_no_verify() {
    let t = TempDir::new("yolo-map");
    let r = t.path();
    write(
        &r.join("data.yaml"),
        "train: ../train/images\nval: ../valid/images\nnc: 3\nnames:\n  0: a\n  2: c\n  1: b\n",
    );
    write_png(&r.join("train/images/x.png"), 10, 10, [1, 2, 3]);
    write(&r.join("train/labels/x.txt"), "2 0.5 0.5 0.5 0.5\n");
    write_png(&r.join("valid/images/y.png"), 10, 10, [1, 2, 3]);
    let ds = open(r.join("data.yaml")).unwrap();
    assert_eq!(ds.classes, ["a", "b", "c"]);
    assert_eq!(ds.split_sizes(), [1, 1, 0]);
    assert_eq!(ds.train.boxes(0)[0].class, 2);
    // Without verification sizes stay unknown but labels still load.
    let ds = open_with(r, &OpenOptions { verify_images: false, ..Default::default() }).unwrap();
    assert_eq!(ds.train.image_size(0), None);
    assert_eq!(ds.train.boxes(0).len(), 1);
    // nc/names mismatch is an error.
    write(&r.join("data.yaml"), "train: ../train/images\nnc: 2\nnames: [a, b, c]\n");
    assert!(open(r).is_err());
}

/// A YAML found by its `train` key is found whatever the case of its extension (`.YAML`, `.Yml`): on a
/// case-sensitive file system `DATA.YAML` was not looked at.
#[test]
fn a_file_read_in_the_wrong_format_says_so_in_plain_words() {
    let t = TempDir::new("wrong-format");
    let r = t.path();
    write(&r.join("instances_train.json"), "{}");
    write(&r.join("data.yaml"), "names: [a]\n");
    let msg = |p: &str, format| match open_with(r.join(p), &OpenOptions { format, ..Default::default() }) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{p} opened as {format:?}"),
    };
    let json = msg("instances_train.json", Format::Yolo);
    assert!(json.contains("is a COCO .json file, not YOLO"), "{json}");
    assert!(msg("data.yaml", Format::Coco).contains("is a YOLO data.yaml, not COCO"));
    assert!(!json.contains("Yolo"), "no enum names: {json}");
}

#[test]
fn yaml_extension_case_does_not_matter() {
    for name in ["Shapes.YAML", "shapes.Yml"] {
        let t = TempDir::new("yolo-yaml-case");
        let r = t.path();
        write_png(&r.join("images/train/1.png"), 8, 8, [0; 3]);
        write(
            &r.join("labels/train/1.txt"),
            "1 0.5 0.5 0.5 0.5
",
        );
        write(
            &r.join(name),
            "train: images/train
val: images/train
names: [ring, cross]
",
        );
        let ds = open(r).unwrap();
        assert_eq!(ds.classes, ["ring", "cross"], "{name}");
        assert_eq!(ds.split_sizes(), [1, 1, 0], "{name}");
    }
}

#[test]
fn yolo_dir_without_yaml() {
    let t = TempDir::new("yolo-dir");
    let r = t.path();
    write_png(&r.join("images/train/1.png"), 8, 8, [0; 3]);
    write(&r.join("labels/train/1.txt"), "3 0.5 0.5 0.5 0.5\n");
    write_png(&r.join("images/val/2.png"), 8, 8, [0; 3]);
    let ds = open(r).unwrap();
    assert_eq!(ds.classes, ["class0", "class1", "class2", "class3"]);
    assert_eq!(ds.split_sizes(), [1, 1, 0]);
    write(&r.join("classes.txt"), "w\nx\ny\nz\nextra\n");
    assert_eq!(open(r).unwrap().classes.len(), 5);
    // Format::Coco on a YOLO folder fails clearly.
    assert!(open_with(r, &OpenOptions { format: Format::Coco, ..Default::default() }).is_err());
}

#[test]
fn coco_instances_with_category_remap() {
    let t = TempDir::new("coco");
    let r = t.path();
    write_png(&r.join("images/train2017/1.png"), 100, 50, [5, 5, 5]);
    write_png(&r.join("images/train2017/2.png"), 64, 64, [5, 5, 5]);
    write_png(&r.join("images/train2017/3.png"), 64, 64, [5, 5, 5]);
    write_png(&r.join("images/train2017/4.png"), 64, 64, [5, 5, 5]);
    write_png(&r.join("images/train2017/5.png"), 64, 64, [5, 5, 5]);
    write_png(&r.join("images/val2017/9.png"), 20, 20, [5, 5, 5]);
    let train = serde_json::json!({
        "images": [
            {"id": 1, "file_name": "1.png", "width": 100, "height": 50},
            {"id": 2, "file_name": "2.png", "width": 64, "height": 64},
            {"id": 3, "file_name": "3.png", "width": 64, "height": 64},
            {"id": 4, "file_name": "4.png"},
            {"id": 5, "file_name": "5.png", "width": 64, "height": 64}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 5, 20, 10], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [90, 40, 30, 30]},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [0, 0, 100, 50], "iscrowd": 1},
            {"id": 4, "image_id": 2, "category_id": 3, "bbox": [1, 2, 3]},
            {"id": 5, "image_id": 3, "category_id": 99, "bbox": [1, 2, 3, 4]},
            {"id": 6, "image_id": 4, "category_id": 3, "bbox": [0, 0, 32, 16]},
            {"id": 7, "image_id": 5, "category_id": 3, "bbox": [10, null, 20, 30, 40]}
        ],
        "categories": [{"id": 7, "name": "dog"}, {"id": 1, "name": "cat"}, {"id": 3, "name": "bird"}]
    });
    write(&r.join("annotations/instances_train2017.json"), serde_json::to_vec(&train).unwrap());
    let val = serde_json::json!({
        "images": [{"id": "a", "file_name": "9.png", "width": 20, "height": 20}],
        "annotations": [{"image_id": "a", "category_id": 7, "bbox": [0, 0, 10, 10]}],
        "categories": [{"id": 7, "name": "dog"}, {"id": 1, "name": "cat"}, {"id": 3, "name": "bird"}]
    });
    write(&r.join("annotations/instances_val2017.json"), serde_json::to_vec(&val).unwrap());

    let ds = open(r).unwrap();
    assert_eq!(ds.format, Format::Coco);
    assert_eq!(ds.classes, ["cat", "bird", "dog"]);
    assert_eq!(ds.split_sizes(), [2, 1, 0], "{:#?}", ds.report);
    assert_eq!(ds.report.broken_labels(), 3, "bad bbox, unknown category, bbox with null");
    let b = ds.train.boxes(0);
    assert_eq!(b.len(), 2, "crowd box skipped");
    assert_eq!(b[0].class, 2);
    assert!(approx(b[0].cx, 0.2) && approx(b[0].cy, 0.2) && approx(b[0].w, 0.2) && approx(b[0].h, 0.2));
    // Clipped to the image: x 90..100, y 40..50.
    assert_eq!(b[1].class, 0);
    assert!(approx(b[1].w, 0.1) && approx(b[1].h, 0.2) && approx(b[1].cx, 0.95));
    // Image 4 had no declared size: probed.
    assert_eq!(ds.train.image_size(1), Some((64, 64)));
    assert_eq!(ds.train.boxes(1)[0], BBox::new(1, 0.25, 0.125, 0.5, 0.25));
    assert_eq!(ds.val.boxes(0)[0].class, 2);
    assert_eq!(ds.stats().class_counts, vec![1, 1, 2]);
}

/// An integral float id is the integer: `"image_id": 1.0` names the image `"id": 1` (and the other way round).
#[test]
fn coco_integral_float_ids_match_integer_ids() {
    let t = TempDir::new("coco-float-id");
    let r = t.path();
    write_png(&r.join("train/a.png"), 10, 10, [0; 3]);
    write_png(&r.join("train/b.png"), 10, 10, [0; 3]);
    let j = serde_json::json!({
        "images": [
            {"id": 1, "file_name": "a.png", "width": 10, "height": 10},
            {"id": 2.0, "file_name": "b.png", "width": 10, "height": 10}
        ],
        "annotations": [
            {"image_id": 1.0, "category_id": 0, "bbox": [0, 0, 5, 5]},
            {"image_id": 2, "category_id": 0, "bbox": [5, 5, 5, 5]}
        ],
        "categories": [{"id": 0, "name": "thing"}]
    });
    write(&r.join("train/_annotations.coco.json"), serde_json::to_vec(&j).unwrap());
    let ds = open(r).unwrap();
    assert!(ds.report.issues.is_empty(), "{:#?}", ds.report);
    assert_eq!(ds.split_sizes(), [2, 0, 0]);
    assert_eq!((ds.train.boxes(0).len(), ds.train.boxes(1).len()), (1, 1));
}

/// Two `images` entries with one id: the first keeps its boxes, the second is reported (not silently made a
/// background image) and left out.
#[test]
fn coco_duplicate_image_ids_are_reported() {
    let t = TempDir::new("coco-dup-id");
    let r = t.path();
    write_png(&r.join("train/a.png"), 10, 10, [0; 3]);
    write_png(&r.join("train/b.png"), 10, 10, [0; 3]);
    let j = serde_json::json!({
        "images": [
            {"id": 7, "file_name": "a.png", "width": 10, "height": 10},
            {"id": 7, "file_name": "b.png", "width": 10, "height": 10}
        ],
        "annotations": [{"image_id": 7, "category_id": 0, "bbox": [0, 0, 5, 5]}],
        "categories": [{"id": 0, "name": "thing"}]
    });
    write(&r.join("train/_annotations.coco.json"), serde_json::to_vec(&j).unwrap());
    let ds = open(r).unwrap();
    assert_eq!(ds.split_sizes(), [1, 0, 0], "{:#?}", ds.report);
    assert!(ds.train.entries()[0].image.ends_with("a.png"));
    assert_eq!(ds.train.boxes(0).len(), 1);
    let issue =
        ds.report.issues.iter().find(|i| i.message.contains("b.png")).expect("the duplicate is reported");
    assert_eq!(issue.kind, IssueKind::Other);
    assert!(issue.message.contains("a.png") && issue.message.contains('7'), "{}", issue.message);
}

#[test]
fn coco_roboflow_layout_and_single_json() {
    let t = TempDir::new("coco-rf");
    let r = t.path();
    write_png(&r.join("train/a.png"), 10, 10, [0; 3]);
    write_png(&r.join("valid/b.png"), 10, 10, [0; 3]);
    let j = |f: &str| {
        serde_json::to_vec(&serde_json::json!({
            "images": [{"id": 0, "file_name": f, "width": 10, "height": 10}],
            "annotations": [{"image_id": 0, "category_id": 0, "bbox": [0, 0, 5, 5]}],
            "categories": [{"id": 0, "name": "thing"}]
        }))
        .unwrap()
    };
    write(&r.join("train/_annotations.coco.json"), j("a.png"));
    write(&r.join("valid/_annotations.coco.json"), j("b.png"));
    let ds = open(r).unwrap();
    assert_eq!(ds.format, Format::Coco);
    assert_eq!(ds.split_sizes(), [1, 1, 0]);
    let single = open(r.join("valid/_annotations.coco.json")).unwrap();
    assert_eq!(single.split_sizes(), [0, 1, 0]);
    // Broken JSON is reported, and with nothing readable the open fails.
    write(&r.join("valid/_annotations.coco.json"), "{ not json");
    let ds = open(r).unwrap();
    assert_eq!(ds.split_sizes(), [1, 0, 0]);
    assert_eq!(ds.report.broken_labels(), 1);
}

#[test]
fn unknown_layout_is_an_error() {
    let t = TempDir::new("empty");
    assert!(open(t.path()).is_err());
    assert!(open(t.join("nope")).is_err());
}

/// Inserts an EXIF APP1 segment with the given orientation right after the JPEG SOI marker.
fn with_exif_orientation(jpeg: &[u8], orientation: u16, big_endian: bool) -> Vec<u8> {
    let mut tiff = Vec::new();
    if big_endian {
        tiff.extend_from_slice(b"MM\x00\x2A\x00\x00\x00\x08");
        tiff.extend_from_slice(&1u16.to_be_bytes());
        tiff.extend_from_slice(&0x0112u16.to_be_bytes());
        tiff.extend_from_slice(&3u16.to_be_bytes());
        tiff.extend_from_slice(&1u32.to_be_bytes());
        tiff.extend_from_slice(&orientation.to_be_bytes());
        tiff.extend_from_slice(&[0, 0]);
        tiff.extend_from_slice(&0u32.to_be_bytes());
    } else {
        tiff.extend_from_slice(b"II\x2A\x00\x08\x00\x00\x00");
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x0112u16.to_le_bytes());
        tiff.extend_from_slice(&3u16.to_le_bytes());
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&orientation.to_le_bytes());
        tiff.extend_from_slice(&[0, 0]);
        tiff.extend_from_slice(&0u32.to_le_bytes());
    }
    let mut seg = b"Exif\0\0".to_vec();
    seg.extend_from_slice(&tiff);
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&seg);
    out.extend_from_slice(&jpeg[2..]);
    out
}

#[test]
fn jpeg_exif_orientation_is_applied() {
    // 64x32: left half red, right half blue.
    let mut img = Image::new(64, 32, [0, 0, 250]);
    img.fill_rect(0, 0, 32, 32, [250, 0, 0]);
    let jpeg = encode(&img, ImageFormat::Jpeg(95)).unwrap();
    assert_eq!(jpeg_exif_orientation(&jpeg), None);
    let t = TempDir::new("exif");
    for (o, be) in [(6u16, false), (6, true), (8, false), (3, true)] {
        let bytes = with_exif_orientation(&jpeg, o, be);
        assert_eq!(jpeg_exif_orientation(&bytes), Some(o as u8));
        let p = t.join(format!("o{o}{be}.jpg"));
        write(&p, &bytes);
        let d = read_image(&p).unwrap();
        let rotated = o != 3;
        assert_eq!((d.width, d.height), if rotated { (32, 64) } else { (64, 32) });
        assert_eq!(probe_size(&p).unwrap(), (d.width, d.height));
        let red = |p: [u8; 3]| p[0] > 200 && p[2] < 60;
        let blue = |p: [u8; 3]| p[2] > 200 && p[0] < 60;
        match o {
            // Rotate 90 CW: the left (red) half ends up on top.
            6 => assert!(red(d.pixel(16, 8)) && blue(d.pixel(16, 56))),
            // Rotate 90 CCW: the left half ends up at the bottom.
            8 => assert!(blue(d.pixel(16, 8)) && red(d.pixel(16, 56))),
            _ => assert!(blue(d.pixel(8, 16)) && red(d.pixel(56, 16))),
        }
    }
}

#[test]
fn exif_after_large_app_segments() {
    let mut img = Image::new(40, 20, [0, 0, 250]);
    img.fill_rect(0, 0, 20, 20, [250, 0, 0]);
    let jpeg = encode(&img, ImageFormat::Jpeg(95)).unwrap();
    let with_exif = with_exif_orientation(&jpeg, 6, false);
    // Three ~60 KB APP2 segments (like a big ICC profile) before the EXIF segment.
    let mut bytes = with_exif[..2].to_vec();
    for _ in 0..3 {
        bytes.extend_from_slice(&[0xFF, 0xE2]);
        bytes.extend_from_slice(&60_002u16.to_be_bytes());
        bytes.extend(std::iter::repeat_n(0u8, 60_000));
    }
    bytes.extend_from_slice(&with_exif[2..]);
    let t = TempDir::new("exif-far");
    let p = t.join("far.jpg");
    write(&p, &bytes);
    assert_eq!(probe_size(&p).unwrap(), (20, 40));
    let d = read_image(&p).unwrap();
    assert_eq!((d.width, d.height), (20, 40));
}

#[test]
fn coco_image_dir_found_past_missing_files() {
    let t = TempDir::new("coco-missing");
    let r = t.path();
    let images: Vec<_> = (0..40)
        .map(|i| serde_json::json!({"id": i, "file_name": format!("{i}.png"), "width": 8, "height": 8}))
        .collect();
    // Only the last few files exist (under images/train2017); everything before them is missing.
    for i in 36..40 {
        write_png(&r.join(format!("images/train2017/{i}.png")), 8, 8, [0; 3]);
    }
    let j = serde_json::json!({"images": images, "annotations": [], "categories": [{"id": 1, "name": "x"}]});
    write(&r.join("annotations/instances_train2017.json"), serde_json::to_vec(&j).unwrap());
    let ds = open(r).unwrap();
    assert_eq!(ds.train.len(), 4);
    assert_eq!(ds.report.count(IssueKind::MissingImage), 36);
}

#[test]
fn absurd_class_counts_are_rejected_not_allocated() {
    let t = TempDir::new("yolo-huge");
    let r = t.path();
    write_png(&r.join("images/train/a.png"), 8, 8, [0; 3]);
    write(&r.join("labels/train/a.txt"), "1000000000 0.5 0.5 0.2 0.2\n");
    write_png(&r.join("images/train/b.png"), 8, 8, [0; 3]);
    write(&r.join("labels/train/b.txt"), "1 0.5 0.5 0.2 0.2\n");
    let ds = open(r).unwrap();
    assert_eq!(ds.classes.len(), 2);
    assert_eq!(ds.report.broken_labels(), 1);
    write(&r.join("data.yaml"), "train: images/train\nnc: 4000000000\n");
    assert!(open(r).is_err());
}

#[test]
fn coco_ground_truth_paths_match_the_loader() {
    // Nested file names deeper than three levels, and `a.png` next to `dir/a.png`: every dataset entry must map
    // to exactly the COCO image it was loaded from.
    let t = TempDir::new("coco-gt-paths");
    let r = t.path();
    let names = ["a.png", "dir/a.png", "x/y/z/w/d.png"];
    for n in names {
        write_png(&r.join("val2017").join(n), 16, 16, [10, 10, 10]);
    }
    let images: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, n)| format!(r#"{{"id":{},"file_name":"{n}","width":16,"height":16}}"#, i + 10))
        .collect();
    write(
        &r.join("annotations/instances_val2017.json"),
        format!(
            r#"{{"images":[{}],"annotations":[{{"id":1,"image_id":11,"category_id":1,"bbox":[1,1,5,5],"area":20,"iscrowd":1}}],"categories":[{{"id":1,"name":"x"}}]}}"#,
            images.join(",")
        ),
    );
    let ds = open(r).unwrap();
    let gt = folo_data::open_coco_ground_truth(r, Split::Val).unwrap();
    assert_eq!(ds.val.len(), 3);
    for e in ds.val.entries() {
        let hits: Vec<_> =
            gt.images.iter().filter(|im| im.path.as_deref() == Some(e.image.as_path())).collect();
        assert_eq!(hits.len(), 1, "{}", e.image.display());
        assert!(e.image.ends_with(&hits[0].file_name));
    }
    let crowd = &gt.images.iter().find(|im| im.id == 11).unwrap().annotations[0];
    assert!(crowd.iscrowd && crowd.area == 20.0 && crowd.bbox == [1.0, 1.0, 5.0, 5.0]);
    assert!(folo_data::open_coco_ground_truth(r, Split::Train).is_err());
}

#[test]
fn coco_ground_truth_of_a_yolo_dataset_says_it_is_yolo() {
    // `folo val --coco` on a YOLO folder: the error names the format it found, not "could not detect a dataset".
    let t = TempDir::new("coco-gt-of-yolo");
    let r = t.path();
    write_png(&r.join("images/val/a.png"), 8, 8, [0; 3]);
    write(&r.join("labels/val/a.txt"), "0 0.5 0.5 0.2 0.2\n");
    write(&r.join("data.yaml"), "val: images/val\nnames: [a]\n");
    for err in [
        folo_data::open_coco_ground_truth(r, Split::Val).unwrap_err().to_string(),
        folo_data::coco_annotation_files(r).unwrap_err().to_string(),
    ] {
        assert!(err.contains("is a YOLO dataset, not a COCO one"), "{err}");
        assert!(!err.contains(folo_data::UNDETECTED), "{err}");
    }
    // Nothing that looks like a dataset still says so.
    let empty = TempDir::new("coco-gt-of-nothing");
    let err = folo_data::open_coco_ground_truth(empty.path(), Split::Val).unwrap_err().to_string();
    assert!(err.contains(folo_data::UNDETECTED), "{err}");
}

/// `open_splits` builds the asked splits exactly as a full open does and never reads the others' images.
#[test]
fn open_splits_reads_only_the_asked_splits() {
    let t = TempDir::new("splits");
    // YOLO with names: a corrupt training image is neither probed nor reported when only val is opened.
    let y = t.join("yolo");
    write(&y.join("data.yaml"), "train: images/train\nval: images/val\ntest: images/test\nnames: [a, b]\n");
    write_png(&y.join("images/train/1.png"), 8, 8, [0; 3]);
    write(&y.join("images/train/2.jpg"), b"\xFF\xD8 not a jpeg");
    write_png(&y.join("images/val/3.png"), 12, 10, [0; 3]);
    write(&y.join("labels/val/3.txt"), "1 0.5 0.5 0.5 0.5\n");
    write_png(&y.join("images/test/4.png"), 8, 8, [0; 3]);
    let opts = OpenOptions::default();
    let full = open(&y).unwrap();
    let val = open_splits(&y, &opts, &[Split::Val]).unwrap();
    assert_eq!(val.classes, full.classes);
    assert_eq!(val.split_sizes(), [0, 1, 0]);
    assert_eq!(val.val.entries(), full.val.entries());
    assert_eq!(full.report.count(IssueKind::CorruptImage), 1);
    assert_eq!(val.report.count(IssueKind::CorruptImage), 0);
    let both = open_splits(&y, &opts, &[Split::Train, Split::Val]).unwrap();
    assert_eq!(both.split_sizes(), [1, 1, 0]);
    assert_eq!(both.train.entries(), full.train.entries());
    // Without names the classes come from every split's labels, so the whole dataset is opened.
    write(&y.join("data.yaml"), "train: images/train\nval: images/val\n");
    write(&y.join("labels/train/1.txt"), "4 0.5 0.5 0.5 0.5\n");
    let val = open_splits(&y, &opts, &[Split::Val]).unwrap();
    assert_eq!(val.classes.len(), 5);
    assert_eq!(val.classes, open(&y).unwrap().classes);

    // COCO: the class ids count the categories of every file, also the ones of the split not opened.
    let c = t.join("coco");
    write_png(&c.join("val2017/9.png"), 20, 20, [5; 3]);
    let train = serde_json::json!({
        "images": [{"id": 1, "file_name": "missing.png", "width": 10, "height": 10}],
        "annotations": [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5]}],
        "categories": [{"id": 2, "name": "zebra"}, {"id": 7, "name": "dog"}]
    });
    write(&c.join("annotations/instances_train2017.json"), serde_json::to_vec(&train).unwrap());
    let val = serde_json::json!({
        "images": [{"id": "a", "file_name": "9.png", "width": 20, "height": 20}],
        "annotations": [{"image_id": "a", "category_id": 7, "bbox": [0, 0, 10, 10]}],
        "categories": [{"id": 7, "name": "dog"}]
    });
    write(&c.join("annotations/instances_val2017.json"), serde_json::to_vec(&val).unwrap());
    let full = open(&c).unwrap();
    assert_eq!(full.report.count(IssueKind::MissingImage), 1);
    let only = open_splits(&c, &opts, &[Split::Val]).unwrap();
    assert_eq!(only.classes, ["zebra", "dog"]);
    assert_eq!(only.classes, full.classes);
    assert_eq!(only.split_sizes(), [0, 1, 0]);
    assert_eq!(only.val.entries(), full.val.entries());
    assert_eq!(only.val.boxes(0)[0].class, 1);
    assert_eq!(only.report.count(IssueKind::MissingImage), 0, "train images are not checked");
    // A train file the full open would skip (no `images` array) adds no categories either way, and one read
    // through the fallback (a duplicate key) gives the same classes.
    for text in [
        r#"{"images": {"not": "an array"}, "categories": [{"id": 1, "name": "cat"}]}"#,
        r#"{"images": [], "images": [], "categories": [{"id": 1, "name": "cat"}]}"#,
        r#"[1, 2]"#,
        r#"{ broken"#,
    ] {
        write(&c.join("annotations/instances_train2017.json"), text);
        let full = open(&c).unwrap();
        let only = open_splits(&c, &opts, &[Split::Val]).unwrap();
        assert_eq!(only.classes, full.classes, "{text}");
        assert_eq!(only.val.entries(), full.val.entries(), "{text}");
    }
    // Only a train file: asking for val opens an empty val split, as the full open does.
    std::fs::remove_file(c.join("annotations/instances_val2017.json")).unwrap();
    write(&c.join("annotations/instances_train2017.json"), serde_json::to_vec(&train).unwrap());
    let only = open_splits(&c, &opts, &[Split::Val]).unwrap();
    assert_eq!((only.split_sizes(), only.classes.len()), ([0, 0, 0], 2));
}

/// Text files written with a UTF-8 byte order mark (Notepad, PowerShell 5.1's `Set-Content -Encoding UTF8`) read
/// like the same files without one: label files, `classes.txt`, `data.yaml` and image lists.
#[test]
fn utf8_bom_is_ignored() {
    const BOM: &str = "\u{feff}";
    let t = TempDir::new("bom");
    let r = t.path();
    write(&r.join("data.yaml"), format!("{BOM}train: images/train\nval: val.txt\nnames: ['cat', 'dog']\n"));
    write_png(&r.join("images/train/a.png"), 20, 20, [10, 10, 10]);
    write(&r.join("labels/train/a.txt"), format!("{BOM}1 0.5 0.5 0.5 0.5\r\n"));
    write_png(&r.join("images/val/v.png"), 20, 20, [10, 10, 10]);
    write(&r.join("val.txt"), format!("{BOM}images/val/v.png\r\n"));
    let ds = open(r).unwrap();
    assert_eq!(ds.report.broken_labels(), 0, "{:#?}", ds.report);
    assert_eq!(ds.split_sizes(), [1, 1, 0], "{:#?}", ds.report);
    assert_eq!(ds.train.boxes(0), &[BBox::new(1, 0.5, 0.5, 0.5, 0.5)]);
    let c = folo_data::check::check(r, &folo_data::check::CheckOptions::default()).unwrap();
    assert!(c.issues.iter().all(|i| i.kind != folo_data::check::CheckKind::BrokenLabel), "{:#?}", c.issues);

    let d = TempDir::new("bom-dir");
    let r = d.path();
    write_png(&r.join("images/train/a.png"), 20, 20, [10, 10, 10]);
    write(&r.join("labels/train/a.txt"), "0 0.5 0.5 0.5 0.5\n");
    write(&r.join("labels/classes.txt"), format!("{BOM}person\r\ncar\r\n"));
    assert_eq!(open(r).unwrap().classes, ["person", "car"]);
}

/// An empty or whitespace split path (`test: ""`, `val: "  "`, `train: ["images/train", ""]`) leaves that source
/// unset like `null`, rather than resolving `root.join("")` to the dataset root and walking every split's images.
#[test]
fn yolo_empty_split_paths_do_not_scan_dataset_root() {
    let t = TempDir::new("yolo-empty-split");
    let r = t.path();
    write_png(&r.join("images/train/a.png"), 16, 16, [10, 20, 30]);
    write(&r.join("labels/train/a.txt"), "0 0.5 0.5 0.4 0.4\n");
    write_png(&r.join("images/val/v.png"), 16, 16, [30, 20, 10]);
    write(&r.join("labels/val/v.txt"), "0 0.5 0.5 0.4 0.4\n");
    write(
        &r.join("data.yaml"),
        "path: \"   \"\ntrain: [images/train, \"\", \"   \"]\nval: images/val\ntest: \"\"\nnames: [thing]\n",
    );

    let ds = open(r).unwrap();
    assert_eq!(ds.split_sizes(), [1, 1, 0], "{:#?}", ds.report);
    assert!(ds.report.issues.is_empty(), "{:#?}", ds.report);
    let report = folo_data::check::check(r, &folo_data::check::CheckOptions::default()).unwrap();
    assert_eq!(report.splits, [1, 1, 0]);
    assert!(report.is_clean(), "{:?}", report.issues);

    // Whitespace val leaves val empty without leaking train images into it.
    write(&r.join("data.yaml"), "train: images/train\nval: \"   \"\ntest: [\"\"]\nnames: [thing]\n");
    let ds = open(r).unwrap();
    assert_eq!(ds.split_sizes(), [1, 0, 0], "{:#?}", ds.report);
    assert!(ds.report.issues.is_empty(), "{:#?}", ds.report);

    // An explicit `.` split path still walks the dataset folder when intended.
    let flat = TempDir::new("yolo-dot-split");
    write_png(&flat.join("a.png"), 16, 16, [10, 20, 30]);
    write(&flat.join("a.txt"), "0 0.5 0.5 0.4 0.4\n");
    write(&flat.join("data.yaml"), "train: .\ntest: \"\"\nnames: [thing]\n");
    let ds = open(flat.path()).unwrap();
    assert_eq!(ds.split_sizes(), [1, 0, 0], "{:#?}", ds.report);
}
