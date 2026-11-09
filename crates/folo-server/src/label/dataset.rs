//! Writing a job's labeled images as a YOLO dataset folder (`images/`, `labels/`, `data.yaml`), the layout
//! folo-data reads and trains from.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use folo_api::BBox;

use crate::util::Rng;

/// A labeled image to write.
pub struct Item {
    /// The image file to copy.
    pub source: PathBuf,
    /// Its name in the dataset (stems unique).
    pub file: String,
    pub width: u32,
    pub height: u32,
    /// Its boxes, pixel `xyxy`.
    pub boxes: Vec<BBox>,
}

/// How many of `n` images go to `val`: `round(n * val_fraction)`, but at least one image stays in `train`.
pub fn val_count(n: usize, val_fraction: f32) -> usize {
    ((n as f64 * f64::from(val_fraction)).round() as usize).min(n.saturating_sub(1))
}

/// Writes `items` into `dir` (which exists and is empty): a seeded shuffle, the first [`val_count`] images to
/// `val`, the rest to `train`; the files copied byte for byte, one label file per image (an empty one for an image
/// without boxes) and a `data.yaml` naming `classes`. Returns `[train, val]` image counts.
pub fn write(
    dir: &Path,
    classes: &[String],
    items: &[Item],
    val_fraction: f32,
    seed: u64,
) -> std::io::Result<[usize; 2]> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    let mut rng = Rng::new(seed);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let val = val_count(items.len(), val_fraction);
    for (split, part) in [("val", &order[..val]), ("train", &order[val..])] {
        if part.is_empty() {
            continue;
        }
        let (images, labels) = (dir.join("images").join(split), dir.join("labels").join(split));
        std::fs::create_dir_all(&images)?;
        std::fs::create_dir_all(&labels)?;
        for &i in part {
            let item = &items[i];
            std::fs::copy(&item.source, images.join(&item.file))?;
            let stem = Path::new(&item.file).file_stem().unwrap_or_default().to_string_lossy();
            std::fs::write(labels.join(format!("{stem}.txt")), label_text(item))?;
        }
    }
    let mut yaml = String::from("# made with FOLO agent labeling\ntrain: images/train\n");
    if val > 0 {
        yaml.push_str("val: images/val\n");
    }
    yaml.push_str("names:\n");
    for (i, c) in classes.iter().enumerate() {
        // JSON strings are valid YAML double-quoted scalars, so any name reads back unchanged.
        let _ = writeln!(yaml, "  {i}: {}", serde_json::to_string(c).expect("strings serialize"));
    }
    std::fs::write(dir.join("data.yaml"), yaml)?;
    Ok([items.len() - val, val])
}

/// An image's YOLO label file: one `class cx cy w h` line per box, normalised to the image.
fn label_text(item: &Item) -> String {
    let (w, h) = (f64::from(item.width), f64::from(item.height));
    let mut text = String::new();
    for b in &item.boxes {
        let (x1, y1, x2, y2) = (f64::from(b.x1), f64::from(b.y1), f64::from(b.x2), f64::from(b.y2));
        let _ = writeln!(
            text,
            "{} {:.6} {:.6} {:.6} {:.6}",
            b.class,
            (x1 + x2) / 2.0 / w,
            (y1 + y2) / 2.0 / h,
            (x2 - x1) / w,
            (y2 - y1) / h
        );
    }
    text
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    use super::*;

    #[test]
    fn val_share_never_empties_train() {
        assert_eq!(val_count(10, 0.2), 2);
        assert_eq!(val_count(3, 0.2), 1);
        assert_eq!(val_count(2, 0.2), 0);
        assert_eq!(val_count(2, 0.5), 1);
        assert_eq!(val_count(1, 0.5), 0);
        assert_eq!(val_count(0, 0.5), 0);
        assert_eq!(val_count(10, 0.0), 0);
    }

    #[test]
    fn labels_are_normalised_centre_boxes() {
        let item = Item {
            source: PathBuf::new(),
            file: "a.jpg".into(),
            width: 200,
            height: 100,
            boxes: vec![
                BBox { class: 2, x1: 20.0, y1: 10.0, x2: 120.0, y2: 60.0 },
                BBox { class: 0, x1: 0.0, y1: 0.0, x2: 200.0, y2: 100.0 },
            ],
        };
        assert_eq!(
            label_text(&item),
            "2 0.350000 0.350000 0.500000 0.500000\n0 0.500000 0.500000 1.000000 1.000000\n"
        );
    }

    #[test]
    fn a_dataset_folder_folo_data_reads() {
        let dir = TestDir::new("label-dataset-write");
        let source = dir.join("src");
        std::fs::create_dir_all(&source).unwrap();
        let items: Vec<Item> = (0..10)
            .map(|i| {
                let path = source.join(format!("in{i}.png"));
                let image = folo_data::Image::new(32, 24, [i as u8 * 20, 90, 200]);
                folo_data::write_image(&image, &path, folo_data::ImageFormat::Png).unwrap();
                Item {
                    source: path,
                    file: format!("img{i}.png"),
                    width: 32,
                    height: 24,
                    // Every third image is a background image.
                    boxes: if i % 3 == 0 {
                        vec![]
                    } else {
                        vec![BBox { class: i % 2, x1: 4.0, y1: 4.0, x2: 20.0, y2: 16.0 }]
                    },
                }
            })
            .collect();
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let classes = vec!["cat".to_string(), "traffic: \"light\"".to_string()];
        assert_eq!(write(&out, &classes, &items, 0.2, 7).unwrap(), [8, 2]);

        let ds = folo_data::open(&out).unwrap();
        assert_eq!(ds.classes, classes);
        assert_eq!(ds.split_sizes(), [8, 2, 0]);
        let boxes: usize = folo_data::Split::ALL
            .into_iter()
            .flat_map(|s| ds.split(s).entries().iter().map(|e| e.boxes.len()))
            .sum();
        assert_eq!(boxes, 6);
        // The same seed gives the same split; another seed another one.
        let split_of = |seed: u64| {
            let again = dir.join(format!("again{seed}"));
            std::fs::create_dir_all(&again).unwrap();
            write(&again, &classes, &items, 0.2, seed).unwrap();
            let mut val: Vec<String> = std::fs::read_dir(again.join("images/val"))
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            val.sort();
            val
        };
        assert_eq!(split_of(7), split_of(7));
        assert_ne!(split_of(7), split_of(8));
    }

    #[test]
    fn without_val_images_there_is_no_val_split() {
        let dir = TestDir::new("label-dataset-train-only");
        let source = dir.join("only.png");
        folo_data::write_image(&folo_data::Image::new(8, 8, [1, 2, 3]), &source, folo_data::ImageFormat::Png)
            .unwrap();
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let items = [Item { source, file: "only.png".into(), width: 8, height: 8, boxes: vec![] }];
        assert_eq!(write(&out, &["a".to_string()], &items, 0.5, 0).unwrap(), [1, 0]);
        assert!(!out.join("images/val").exists());
        let ds = folo_data::open(&out).unwrap();
        assert_eq!(ds.split_sizes(), [1, 0, 0]);
    }
}
