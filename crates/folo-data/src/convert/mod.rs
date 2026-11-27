//! Format conversion (`folo dataset convert`), COCO export ([`export_coco`]), and Pascal VOC export
//! ([`export_voc`]): writes an opened dataset as YOLO, COCO, or Pascal VOC.
//!
//! Under [`convert`], images are copied byte for byte (no re-encoding) to `images/<split>/` (YOLO / COCO) or
//! `JPEGImages/` (VOC); images the loader skipped (corrupt, missing, broken labels) are left out. File names keep
//! their stem, made unique within a split (or across all splits for Pascal VOC, where `JPEGImages/` and
//! `Annotations/` are shared). For finer-grained export (split selection, hard-linking, referencing or skipping
//! images, coordinate conventions, pretty-printed atomic manifests), use [`export_coco`] or [`export_voc`].

mod coco;
mod voc;

pub use coco::{CocoExportOptions, CocoExportReport, CocoImageMode, CocoSplitReport, export_coco};
pub use voc::{VocExportOptions, VocExportReport, VocImageMode, VocSplitReport, export_voc};

use crate::dataset::{Dataset, ImageEntry, LoadedDataset};
use crate::error::{DataError, Result};
use crate::{Format, Split};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::{BufWriter, ErrorKind, Write as _};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic counter for unique sibling temporary filenames during atomic writes.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

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
    if to == Format::Coco {
        let report = export_coco(
            ds,
            out,
            &CocoExportOptions {
                description: Some(format!("converted by FOLO from {}", ds.root.display())),
                ..CocoExportOptions::default()
            },
        )?;
        return Ok(Converted {
            root: report.root,
            format: Format::Coco,
            splits: report.splits,
            boxes: report.boxes,
        });
    }
    if to == Format::Voc {
        let report = export_voc(ds, out, &VocExportOptions::default())?;
        return Ok(Converted {
            root: report.root,
            format: Format::Voc,
            splits: report.splits,
            boxes: report.boxes,
        });
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
        let lbl_dir = out.join("labels").join(split.as_str());
        mkdir(&lbl_dir)?;
        d.entries().par_iter().zip(&names).try_for_each(|(e, name)| {
            let mut text = String::with_capacity(e.boxes.len() * 48);
            for b in &e.boxes {
                let _ = writeln!(text, "{} {} {} {} {}", b.class, b.cx, b.cy, b.w, b.h);
            }
            let p = lbl_dir.join(Path::new(name).with_extension("txt"));
            std::fs::write(&p, text).map_err(|e| DataError::io(&p, e))
        })?;
        let _ = writeln!(yaml_splits, "{split}: images/{split}");
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

pub(crate) fn mkdir(p: &Path) -> Result<()> {
    std::fs::create_dir_all(p).map_err(|e| DataError::io(p, e))
}

pub(super) fn copy(from: &Path, to: &Path) -> Result<()> {
    crate::guard::check(from).map_err(|e| DataError::io(from, e))?;
    std::fs::copy(from, to).map(|_| ()).map_err(|e| DataError::io(from, e))
}

/// Resolves requested splits in canonical `[Train, Val, Test]` order, rejecting empty selections before writing.
pub(super) fn select_splits(ds: &LoadedDataset, requested: &[Split]) -> Result<Vec<Split>> {
    let candidates: Vec<Split> = if requested.is_empty() {
        Split::ALL.into_iter().filter(|&s| !ds.split(s).is_empty()).collect()
    } else {
        let mut out = Vec::new();
        for s in Split::ALL {
            if requested.contains(&s) && !ds.split(s).is_empty() {
                out.push(s);
            }
        }
        out
    };
    if candidates.is_empty() {
        if requested.is_empty() {
            return Err(DataError::Config(format!(
                "nothing to export: {} has no readable images",
                crate::paths::shown(&ds.root)
            )));
        }
        let names: Vec<&str> = requested.iter().map(|s| s.as_str()).collect();
        return Err(DataError::Config(format!(
            "nothing to export: {} has no readable images in requested splits [{}]",
            crate::paths::shown(&ds.root),
            names.join(", ")
        )));
    }
    Ok(candidates)
}

/// Validates image paths (no NUL bytes), class indices, and finite non-negative box dimensions before claiming `out`.
pub(super) fn validate_common_entries(ds: &LoadedDataset, splits: &[Split]) -> Result<()> {
    let nc = ds.classes.len();
    for &split in splits {
        for e in ds.split(split).entries() {
            if e.image.to_string_lossy().contains('\0') {
                return Err(DataError::Config(format!(
                    "cannot export image path containing NUL byte: {}",
                    e.image.display()
                )));
            }
            for b in &e.boxes {
                if (b.class as usize) >= nc {
                    return Err(DataError::Config(format!(
                        "cannot export {}: box class {} is out of range for {} classes",
                        crate::paths::shown(&e.image),
                        b.class,
                        nc
                    )));
                }
                if ![b.cx, b.cy, b.w, b.h].iter().all(|v| v.is_finite()) || b.w < 0.0 || b.h < 0.0 {
                    return Err(DataError::Config(format!(
                        "cannot export {}: invalid box coordinates ({}, {}, {}, {})",
                        crate::paths::shown(&e.image),
                        b.cx,
                        b.cy,
                        b.w,
                        b.h
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Orders entries deterministically so exports do not depend on filesystem directory iteration order.
pub(super) fn deterministic_entries(entries: &[ImageEntry]) -> Vec<&ImageEntry> {
    #[cfg(windows)]
    {
        let mut ordered: Vec<(&ImageEntry, String)> =
            entries.iter().map(|e| (e, e.image.to_string_lossy().replace('\\', "/"))).collect();
        ordered.sort_by(|a, b| a.1.cmp(&b.1));
        ordered.into_iter().map(|(e, _)| e).collect()
    }
    #[cfg(not(windows))]
    {
        let mut ordered: Vec<&ImageEntry> = entries.iter().collect();
        ordered.sort_by(|a, b| a.image.as_os_str().cmp(b.image.as_os_str()));
        ordered
    }
}

pub(super) fn validate_single_component(name: &str) -> Result<()> {
    let p = Path::new(name);
    let mut comps = p.components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(_)), None) if !name.contains(['/', '\\', '\0']) => Ok(()),
        _ => Err(DataError::Config(format!("unsafe exported filename {name:?}"))),
    }
}

/// Resolves a source path for hard-linking: checks the read guard on `from`, and if `from` is a symlink,
/// canonicalizes it and re-checks the read guard on the resolved target so relative symlinks are never
/// hard-linked as broken symlinks and confinement cannot be bypassed.
pub(super) fn resolve_link_source(from: &Path) -> Result<PathBuf> {
    crate::guard::check(from).map_err(|e| DataError::io(from, e))?;
    let meta = std::fs::symlink_metadata(from).map_err(|e| DataError::io(from, e))?;
    if meta.file_type().is_symlink() {
        let target = std::fs::canonicalize(from).map_err(|e| DataError::io(from, e))?;
        crate::guard::check(&target).map_err(|e| DataError::io(&target, e))?;
        Ok(target)
    } else {
        Ok(from.to_path_buf())
    }
}

pub(super) fn should_fallback_from_link(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::CrossesDevices | ErrorKind::Unsupported | ErrorKind::PermissionDenied)
        || (cfg!(windows) && matches!(e.raw_os_error(), Some(1 | 17 | 50 | 1142)))
}

/// Writes `path` atomically via a unique sibling temporary file and rename, cleaning up only the temporary file
/// created by this call if writing, flushing, or renaming fails.
pub(super) fn write_file_atomic(
    path: &Path,
    write_fn: impl FnOnce(&mut BufWriter<std::fs::File>) -> Result<()>,
) -> Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("export.tmp");
    let pid = std::process::id();
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(".{name}.{pid}-{seq}.tmp"));

    // Acquire the temporary file before enabling cleanup so a pre-existing file is never removed on `AlreadyExists`.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| DataError::io(&tmp, e))?;

    let write_res = (|| -> Result<()> {
        let mut writer = BufWriter::with_capacity(256 << 10, file);
        write_fn(&mut writer)?;
        writer.flush().map_err(|e| DataError::io(&tmp, e))?;
        Ok(())
    })();

    if let Err(err) = write_res {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }

    if let Err(err) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(DataError::io(path, err));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_by_stem() {
        let paths = [Path::new("x/a.jpg"), Path::new("y/a.png"), Path::new("z/A.jpg"), Path::new("a_1.jpg")];
        assert_eq!(unique_names(paths.into_iter()), vec!["a.jpg", "a_1.png", "A_2.jpg", "a_1_1.jpg"]);
    }
}
