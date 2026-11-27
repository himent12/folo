//! Format conversion (`folo dataset convert`): writes an opened dataset as YOLO, COCO, or Pascal VOC.
//!
//! Images are copied byte for byte (no re-encoding) to `images/<split>/` (YOLO / COCO) or `JPEGImages/` (VOC);
//! images the loader skipped (corrupt, missing, broken labels) are left out. File names keep their stem, made unique
//! within a split (or across all splits for Pascal VOC, where `JPEGImages/` and `Annotations/` are shared).

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

/// Writes `ds` to `out` in `to` format (`Yolo`, `Coco`, or `Voc`). `out` must not exist or be an empty directory:
/// it is claimed before the first file ([`crate::unfinished::claim`]). Fails with [`DataError::Config`] before
/// writing anything when the result would be unreadable: no images at all, or YOLO output without train or val
/// images.
pub fn convert(ds: &LoadedDataset, out: &Path, to: Format) -> Result<Converted> {
    if to == Format::Auto {
        return Err(DataError::Config("convert needs a target format (yolo, coco, or voc)".into()));
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
    if to == Format::Voc {
        for (i, c) in ds.classes.iter().enumerate() {
            if c.trim().is_empty() || c.trim() != c || c.contains(['\n', '\r']) {
                return Err(DataError::Config(format!(
                    "cannot convert class #{i} ({c:?}) to Pascal VOC: class names must be non-empty, trimmed, and single-line"
                )));
            }
        }
    }
    let claim = crate::unfinished::claim(out)?;
    let mut splits = [0usize; 3];
    let mut boxes = 0;
    let mut yaml_splits = String::new();
    let mut voc_taken = HashSet::new();
    if to == Format::Voc {
        mkdir(&out.join("JPEGImages"))?;
        mkdir(&out.join("Annotations"))?;
        mkdir(&out.join("ImageSets").join("Main"))?;
    }
    for split in Split::ALL {
        let d = ds.split(split);
        if d.is_empty() {
            continue;
        }
        if to == Format::Voc {
            let img_dir = out.join("JPEGImages");
            let ann_dir = out.join("Annotations");
            let names = unique_voc_names(&mut voc_taken, d.entries().iter().map(|e| e.image.as_path()));
            d.entries()
                .par_iter()
                .zip(&names)
                .try_for_each(|(e, name)| copy(&e.image, &img_dir.join(name)))?;
            let sizes: Vec<(u32, u32)> = d
                .entries()
                .par_iter()
                .map(|e| e.size.map_or_else(|| probe_size(&e.image), Ok))
                .collect::<Result<_>>()?;
            d.entries().par_iter().zip(&names).zip(&sizes).try_for_each(|((e, name), &(w, h))| {
                let xml =
                    crate::dataset::voc::format_voc_xml(name, "JPEGImages", w, h, &ds.classes, &e.boxes);
                let p = ann_dir.join(Path::new(name).with_extension("xml"));
                std::fs::write(&p, xml).map_err(|err| DataError::io(&p, err))
            })?;
            let mut list = String::with_capacity(names.len() * 16);
            for name in &names {
                let stem = Path::new(name).file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
                let _ = writeln!(list, "{stem}");
            }
            let set_path = out.join("ImageSets").join("Main").join(format!("{split}.txt"));
            std::fs::write(&set_path, list).map_err(|err| DataError::io(&set_path, err))?;
            splits[split as usize] = d.len();
            boxes += d.entries().iter().map(|e| e.boxes.len()).sum::<usize>();
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
            Format::Auto | Format::Voc => unreachable!("handled above"),
        }
        splits[split as usize] = d.len();
        boxes += d.entries().iter().map(|e| e.boxes.len()).sum::<usize>();
    }
    if to == Format::Voc && !ds.classes.is_empty() {
        let mut names_txt = String::new();
        for c in &ds.classes {
            let _ = writeln!(names_txt, "{c}");
        }
        let p = out.join("classes.txt");
        std::fs::write(&p, names_txt).map_err(|err| DataError::io(&p, err))?;
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

pub(crate) fn mkdir(p: &Path) -> Result<()> {
    std::fs::create_dir_all(p).map_err(|e| DataError::io(p, e))
}

pub(crate) fn copy(from: &Path, to: &Path) -> Result<()> {
    crate::guard::check(from).map_err(|e| DataError::io(from, e))?;
    std::fs::copy(from, to).map(|_| ()).map_err(|e| DataError::io(from, e))
}

/// The extension an image is written with: its own when folo-data reads that extension, otherwise one for the
/// format its content has (a COCO `file_name` may be `a.bin` or have no extension; the YOLO reader would not see it).
pub(crate) fn image_extension(p: &Path) -> String {
    if crate::is_image_path(p) {
        return p.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    }
    let mut head = Vec::with_capacity(64);
    if let Ok(f) = crate::guard::check(p).and_then(|_| std::fs::File::open(p)) {
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
    unique_names_with(&mut taken, paths)
}

fn unique_names_with<'a>(taken: &mut HashSet<String>, paths: impl Iterator<Item = &'a Path>) -> Vec<String> {
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

/// Generates whitespace-free, `#`-free, extension-safe VOC image filenames (unique across all splits) so
/// `ImageSets/Main/*.txt` IDs round-trip cleanly in both FOLO and external Pascal VOC tools.
fn unique_voc_names<'a>(taken: &mut HashSet<String>, paths: impl Iterator<Item = &'a Path>) -> Vec<String> {
    paths
        .map(|p| {
            let raw_stem =
                p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
            let mut stem: String = raw_stem
                .chars()
                .map(|c| if c.is_whitespace() || matches!(c, '.' | '/' | '\\' | ':') { '_' } else { c })
                .collect();
            while stem.starts_with('#') {
                stem.replace_range(..1, "_");
            }
            if stem.is_empty() {
                stem = "image".into();
            }
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
