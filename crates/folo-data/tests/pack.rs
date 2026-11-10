//! `pack`: a dataset as one tar stream in the canonical YOLO layout. Unpacked and opened again it is the same
//! dataset: class names, split sizes, image order and bytes, and boxes.

mod common;

use common::TempDir;
use folo_data::convert::convert;
use folo_data::pack::{pack, packed_len};
use folo_data::synthetic::{SyntheticConfig, write_yolo};
use folo_data::{DataError, Dataset, Format, ImageFormat, LoadedDataset, Split, open};

/// A small synthetic YOLO dataset under `t/yolo` (train and val, some images without boxes).
fn synthetic(t: &TempDir) -> LoadedDataset {
    let cfg = SyntheticConfig {
        images: 14,
        imgsz: 48,
        seed: 11,
        empty_fraction: 0.4,
        vary_aspect: true,
        ..Default::default()
    };
    write_yolo(&cfg, &t.join("yolo"), ImageFormat::Jpeg(85)).unwrap();
    open(t.join("yolo")).unwrap()
}

/// Packs `ds`, checks the stream's length and determinism, unpacks it into `out` and opens it.
fn round_trip(ds: &LoadedDataset, out: &std::path::Path) -> LoadedDataset {
    let mut bytes = Vec::new();
    let written = pack(ds, &mut bytes).unwrap();
    assert_eq!(written, bytes.len() as u64);
    assert_eq!(packed_len(ds).unwrap(), written, "packed_len is exact");
    let mut again = Vec::new();
    pack(ds, &mut again).unwrap();
    assert!(again == bytes, "the stream depends only on the dataset");
    tar::Archive::new(bytes.as_slice()).unpack(out).unwrap();
    let back = open(out).unwrap();
    assert_eq!(back.format, Format::Yolo);
    assert!(back.report.issues.is_empty(), "{:?}", back.report.issues);
    back
}

/// The same classes, splits, image order and bytes, image sizes and boxes (exactly).
fn assert_same(a: &LoadedDataset, b: &LoadedDataset) {
    assert_eq!(a.classes, b.classes);
    assert_eq!(a.split_sizes(), b.split_sizes());
    for split in Split::ALL {
        let (x, y) = (a.split(split), b.split(split));
        for (i, (ex, ey)) in x.entries().iter().zip(y.entries()).enumerate() {
            let name = ey.image.file_name().unwrap().to_string_lossy().into_owned();
            let ext = ex.image.extension().unwrap().to_string_lossy();
            assert_eq!(name, format!("{i:07}.{ext}"), "{split} image {i}");
            assert_eq!(std::fs::read(&ex.image).unwrap(), std::fs::read(&ey.image).unwrap(), "{name}");
            assert_eq!(ex.size, ey.size, "{name}");
            assert_eq!(ex.boxes, ey.boxes, "{name}");
        }
    }
}

#[test]
fn a_yolo_dataset_round_trips() {
    let t = TempDir::new("pack-yolo");
    let ds = synthetic(&t);
    assert!(ds.train.len() > 0 && ds.val.len() > 0);
    let boxes = |i| ds.train.boxes(i).len();
    assert!((0..ds.train.len()).any(|i| boxes(i) == 0) && (0..ds.train.len()).any(|i| boxes(i) > 1));
    let back = round_trip(&ds, &t.join("unpacked"));
    assert_same(&ds, &back);
    // An image without boxes gets an empty label file.
    let empty = (0..ds.train.len()).find(|&i| ds.train.boxes(i).is_empty()).unwrap();
    let label = t.join(format!("unpacked/labels/train/{empty:07}.txt"));
    assert_eq!(std::fs::read(label).unwrap().len(), 0);
}

#[test]
fn a_coco_dataset_round_trips() {
    let t = TempDir::new("pack-coco");
    convert(&synthetic(&t), &t.join("coco"), Format::Coco).unwrap();
    let coco = open(t.join("coco")).unwrap();
    assert_eq!(coco.format, Format::Coco);
    let back = round_trip(&coco, &t.join("unpacked"));
    assert_same(&coco, &back);
}

#[test]
fn a_dataset_without_train_or_val_images_is_refused() {
    let t = TempDir::new("pack-empty");
    let mut ds = synthetic(&t);
    ds.train = folo_data::FileDataset::new(Split::Train, ds.classes.clone(), Vec::new());
    ds.val = folo_data::FileDataset::new(Split::Val, ds.classes.clone(), Vec::new());
    let mut out = Vec::new();
    assert!(matches!(pack(&ds, &mut out), Err(DataError::Config(_))));
    assert!(out.is_empty(), "nothing written");
    assert!(matches!(packed_len(&ds), Err(DataError::Config(_))));
}
