//! YOLO and COCO parsing, auto-detection, robustness and EXIF orientation. Fixtures are written by the tests.

mod common;

use common::*;
use folo_data::{
    BBox, Dataset, Format, Image, ImageFormat, IssueKind, OpenOptions, Split, encode, jpeg_exif_orientation,
    open, open_with, probe_size, read_image,
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
fn split_and_format_serde() {
    assert_eq!(serde_json::to_string(&Split::Val).unwrap(), "\"val\"");
    assert_eq!(serde_json::from_str::<Format>("\"auto\"").unwrap(), Format::Auto);
    assert_eq!(serde_json::to_string(&IssueKind::BrokenLabel).unwrap(), "\"broken_label\"");
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
