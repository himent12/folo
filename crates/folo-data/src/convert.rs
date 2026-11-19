//! Format conversion (`folo dataset convert`): writes an opened dataset as YOLO or COCO.
//!
//! Images are copied byte for byte (no re-encoding) to `images/<split>/`; images the loader skipped (corrupt,
//! missing, broken labels) are left out. File names keep their stem, made unique within a split (`a.jpg` and
//! `a.png` become `a.jpg` and `a_1.png`, so YOLO label files never collide).

use crate::dataset::{Dataset, LoadedDataset};
use crate::error::{DataError, Result};
use crate::image::probe_size;
use crate::{Format, Split};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// What [`convert`] wrote.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Converted {
    /// Output root.
    pub root: PathBuf,
    /// Output format.
    pub format: Format,
    /// Images written per split `[train, val, test]`.
    pub splits: [usize; 3],
    /// Boxes written.
    pub boxes: usize,
}

/// Writes `ds` to `out` in `to` format (`Yolo` or `Coco`). `out` must not exist or be an empty directory: it is
/// claimed before the first file ([`crate::unfinished::claim`]). Fails
/// with [`DataError::Config`] before writing anything when the result would be unreadable: no images at all, or
/// YOLO output without train or val images.
pub fn convert(ds: &LoadedDataset, out: &Path, to: Format) -> Result<Converted> {
    if to == Format::Auto {
        return Err(DataError::Config("convert needs a target format (yolo or coco)".into()));
    }
    let [train, val, test] = ds.split_sizes();
    if train + val + test == 0 {
        return Err(DataError::Config(format!(
            "nothing to convert: {} has no readable images",
            crate::paths::shown(&ds.root)
        )));
    }
    if to == Format::Yolo && train + val == 0 {
        // The YOLO reader needs a train or val split; writing test-only YOLO would produce an unreadable dataset.
        return Err(DataError::Config(
            "a YOLO dataset needs train or val images, and this one only has test images (convert to coco instead)"
                .into(),
        ));
    }
    let claim = crate::unfinished::claim(out)?;
    let mut splits = [0usize; 3];
    let mut boxes = 0;
    let mut yaml_splits = String::new();
    for split in Split::ALL {
        let d = ds.split(split);
        if d.is_empty() {
            continue;
        }
        let img_dir = out.join("images").join(split.as_str());
        mkdir(&img_dir)?;
        let names = unique_names(d.entries().iter().map(|e| e.image.as_path()));
        d.entries().par_iter().zip(&names).try_for_each(|(e, name)| copy(&e.image, &img_dir.join(name)))?;
        match to {
            Format::Yolo => {
                let lbl_dir = out.join("labels").join(split.as_str());
                mkdir(&lbl_dir)?;
                d.entries().par_iter().zip(&names).try_for_each(|(e, name)| {
                    let mut text = String::with_capacity(e.boxes.len() * 40);
                    for b in &e.boxes {
                        let _ = writeln!(text, "{} {:.6} {:.6} {:.6} {:.6}", b.class, b.cx, b.cy, b.w, b.h);
                    }
                    let p = lbl_dir.join(Path::new(name).with_extension("txt"));
                    std::fs::write(&p, text).map_err(|e| DataError::io(&p, e))
                })?;
                let _ = writeln!(yaml_splits, "{split}: images/{split}");
            }
            Format::Coco => {
                let sizes: Vec<(u32, u32)> = d
                    .entries()
                    .par_iter()
                    .map(|e| e.size.map_or_else(|| probe_size(&e.image), Ok))
                    .collect::<Result<_>>()?;
                let mut images = Vec::with_capacity(names.len());
                let mut anns = Vec::new();
                for (i, ((e, name), &(w, h))) in d.entries().iter().zip(&names).zip(&sizes).enumerate() {
                    images.push(json!({ "id": i + 1, "file_name": name, "width": w, "height": h }));
                    for b in &e.boxes {
                        let [x1, y1, x2, y2] = b.pixel_xyxy(w, h);
                        let (bw, bh) = (x2 - x1, y2 - y1);
                        anns.push(json!({
                            "id": anns.len() + 1, "image_id": i + 1, "category_id": b.class + 1,
                            "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
                        }));
                    }
                }
                let categories: Vec<_> = ds
                    .classes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| json!({ "id": i + 1, "name": n, "supercategory": "" }))
                    .collect();
                let doc = json!({
                    "info": { "description": format!("converted by FOLO from {}", ds.root.display()) },
                    "images": images, "annotations": anns, "categories": categories,
                });
                let ann_dir = out.join("annotations");
                mkdir(&ann_dir)?;
                let p = ann_dir.join(format!("instances_{split}.json"));
                let bytes = serde_json::to_vec(&doc).expect("JSON values serialize");
                std::fs::write(&p, bytes).map_err(|e| DataError::io(&p, e))?;
            }
            Format::Auto => unreachable!("checked above"),
        }
        splits[split as usize] = d.len();
        boxes += d.entries().iter().map(|e| e.boxes.len()).sum::<usize>();
    }
    if to == Format::Yolo {
        let mut yaml = format!("# converted by FOLO from {}\n{yaml_splits}names:\n", ds.root.display());
        for (i, c) in ds.classes.iter().enumerate() {
            // JSON strings are valid YAML double-quoted scalars.
            let _ = writeln!(yaml, "  {i}: {}", serde_json::to_string(c).expect("strings serialize"));
        }
        mkdir(out)?;
        let p = out.join("data.yaml");
        std::fs::write(&p, yaml).map_err(|e| DataError::io(&p, e))?;
    }
    claim.finish()?;
    Ok(Converted { root: out.to_path_buf(), format: to, splits, boxes })
}

fn mkdir(p: &Path) -> Result<()> {
    std::fs::create_dir_all(p).map_err(|e| DataError::io(p, e))
}

fn copy(from: &Path, to: &Path) -> Result<()> {
    std::fs::copy(from, to).map(|_| ()).map_err(|e| DataError::io(from, e))
}

/// The extension an image is written with: its own when folo-data reads that extension, otherwise one for the
/// format its content has (a COCO `file_name` may be `a.bin` or have no extension; the YOLO reader would not see it).
pub(crate) fn image_extension(p: &Path) -> String {
    if crate::is_image_path(p) {
        return p.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    }
    let mut head = Vec::with_capacity(64);
    if let Ok(f) = std::fs::File::open(p) {
        use std::io::Read;
        let _ = f.take(64).read_to_end(&mut head);
    }
    match image::guess_format(&head) {
        Ok(image::ImageFormat::Png) => "png",
        Ok(image::ImageFormat::Bmp) => "bmp",
        Ok(image::ImageFormat::WebP) => "webp",
        _ => "jpg",
    }
    .to_string()
}

/// File names for images in one folder: the original name, with `_N` added to the stem when another image
/// already took that stem (compared case-insensitively, for Windows and macOS file systems).
fn unique_names<'a>(paths: impl Iterator<Item = &'a Path>) -> Vec<String> {
    let mut taken = HashSet::new();
    paths
        .map(|p| {
            let stem =
                p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
            let ext = format!(".{}", image_extension(p));
            let mut candidate = stem.clone();
            let mut n = 0;
            while !taken.insert(candidate.to_lowercase()) {
                n += 1;
                candidate = format!("{stem}_{n}");
            }
            format!("{candidate}{ext}")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_by_stem() {
        let paths = [Path::new("x/a.jpg"), Path::new("y/a.png"), Path::new("z/A.jpg"), Path::new("a_1.jpg")];
        assert_eq!(unique_names(paths.into_iter()), vec!["a.jpg", "a_1.png", "A_2.jpg", "a_1_1.jpg"]);
    }
}
