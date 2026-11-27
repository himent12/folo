//! `dataset_export`: spec-compliant COCO instances JSON export (`folo_data::convert::export_coco`) for
//! registered datasets, supporting split filtering, `copy` / `link` / `reference` image modes, confinement, and
//! safe cleanup on failure.

use std::path::{Path, PathBuf};

use folo_api::{Split, Splits};
use folo_data::convert::{CocoExportOptions, CocoImageMode, export_coco};
use folo_data::{DataError, LoadedDataset, OpenOptions, open_with};
use http::StatusCode;

use super::curate::{check_unreadable_splits, data_error, slug, to_data_format, to_data_split};
use super::results::{Output, out};
use crate::errors::{Error, invalid_arguments};
use crate::paths::PathPolicy;
use crate::server::Inner;
use crate::types::*;

fn from_data_split(s: folo_data::Split) -> Split {
    match s {
        folo_data::Split::Train => Split::Train,
        folo_data::Split::Val => Split::Val,
        folo_data::Split::Test => Split::Test,
    }
}

fn to_coco_image_mode(m: DatasetExportImageMode) -> CocoImageMode {
    match m {
        DatasetExportImageMode::Copy => CocoImageMode::Copy,
        DatasetExportImageMode::Link => CocoImageMode::Link,
        DatasetExportImageMode::Reference => CocoImageMode::Reference,
    }
}

fn from_coco_image_mode(m: CocoImageMode) -> DatasetExportImageMode {
    match m {
        CocoImageMode::Copy => DatasetExportImageMode::Copy,
        CocoImageMode::Link => DatasetExportImageMode::Link,
        CocoImageMode::Reference => DatasetExportImageMode::Reference,
    }
}

/// Picks and creates a fresh empty directory under `<base>/exports/<slug>-coco[-N]`.
fn allocate_export_dir(
    base: &Path,
    name: &str,
    confine_policy: Option<&PathPolicy>,
) -> Result<PathBuf, Error> {
    let root = base.join("exports");
    if let Some(paths) = confine_policy {
        paths.check_dir_inside(&root)?;
    }
    std::fs::create_dir_all(&root)
        .map_err(|e| Error::internal(format!("creating {}: {e}", folo_api::json::path(&root))))?;
    if let Some(paths) = confine_policy {
        paths.check_dir_inside(&root)?;
    }
    let stem = format!("{}-coco", slug(name));
    for n in 1u32..10_000 {
        let candidate = if n == 1 { root.join(&stem) } else { root.join(format!("{stem}-{n}")) };
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(Error::internal(format!("creating {}: {e}", folo_api::json::path(&candidate))));
            }
        }
    }
    Err(Error::internal(format!(
        "could not allocate a unique export folder under {}",
        folo_api::json::path(&root)
    )))
}

/// Canonical path of `p`, resolving each existing prefix with filesystem semantics before applying any subsequent
/// `..` (so Windows `\\?\` prefixes, symlinks, and junctions match `std::fs::canonicalize` on the dataset root
/// even when the target folder does not exist yet).
fn canonical_target(p: &Path) -> Option<PathBuf> {
    let p = if p.as_os_str().is_empty() { Path::new(".") } else { p };
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let mut cur = PathBuf::new();
    let mut missing = false;
    for c in abs.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::Prefix(prefix) => {
                cur.push(prefix.as_os_str());
            }
            std::path::Component::RootDir => {
                cur.push(std::path::Component::RootDir);
                if let Ok(canon) = std::fs::canonicalize(&cur) {
                    cur = canon;
                    missing = false;
                } else {
                    missing = true;
                }
            }
            std::path::Component::ParentDir => {
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(cur.join("..")) {
                        cur = canon;
                    } else {
                        let _ = cur.pop();
                        missing = true;
                    }
                } else {
                    let _ = cur.pop();
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                        missing = false;
                    }
                }
            }
            std::path::Component::Normal(part) => {
                cur.push(part);
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                    } else {
                        missing = true;
                    }
                }
            }
        }
    }
    if cur.as_os_str().is_empty() {
        return None;
    }
    if missing && !cur.ancestors().skip(1).any(Path::exists) {
        return None;
    }
    Some(cur)
}

/// Enforces `--confine` on a caller-supplied export destination with a write-specific hint.
fn check_export_out_inside(paths: &PathPolicy, dir: &Path) -> Result<(), Error> {
    paths.check_dir_inside(dir).map_err(|err| {
        if err.body.code == "path_outside_data_dir" {
            err.hint(
                "pick an output directory inside --data-dir, omit `out` to export under <data_dir>/exports/, \
                 or restart without --confine",
            )
        } else {
            err
        }
    })
}

/// Checks that `out_dir` does not overlap the input dataset or its source folders.
fn check_output_outside_input(out_dir: &Path, input: &Path, ds: &LoadedDataset) -> Result<(), Error> {
    let Some(canon_out) = canonical_target(out_dir) else {
        return Ok(());
    };
    let input_dir: PathBuf = if input.is_dir() {
        input.to_path_buf()
    } else {
        let abs = folo_data::paths::normalize(input);
        abs.parent().map_or(abs.clone(), Path::to_path_buf)
    };
    for p in ds.sources.iter().map(PathBuf::as_path).chain([ds.root.as_path(), input_dir.as_path()]) {
        if let Ok(canon_in) = std::fs::canonicalize(p)
            && (canon_out.starts_with(&canon_in) || canon_in.starts_with(&canon_out))
        {
            return Err(invalid_arguments(format!(
                "out ({}) overlaps the input dataset ({}); pick a separate output folder",
                folo_api::json::path(out_dir),
                folo_api::json::path(p)
            )));
        }
    }
    Ok(())
}

/// Refuses an existing file or non-empty directory before starting export.
fn ensure_empty_export_target(out_dir: &Path) -> Result<(), Error> {
    match std::fs::symlink_metadata(out_dir) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::internal(format!("{}: {e}", folo_api::json::path(out_dir)))),
        Ok(m) if !m.is_dir() => Err(Error::new(
            StatusCode::CONFLICT,
            "output_exists",
            format!("{} already exists and is not a folder", folo_api::json::path(out_dir)),
        )),
        Ok(_) => {
            let mut entries = std::fs::read_dir(out_dir)
                .map_err(|e| Error::internal(format!("{}: {e}", folo_api::json::path(out_dir))))?;
            if entries.next().is_some() {
                return Err(Error::new(
                    StatusCode::CONFLICT,
                    "output_exists",
                    format!("{} already exists and is not empty", folo_api::json::path(out_dir)),
                ));
            }
            Ok(())
        }
    }
}

fn export_error(e: DataError) -> Error {
    match &e {
        DataError::Io { source, .. }
            if matches!(
                source.kind(),
                std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::DirectoryNotEmpty
            ) =>
        {
            Error::new(StatusCode::CONFLICT, "output_exists", e.to_string())
        }
        DataError::Config(m) => Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", m.clone()),
        _ => data_error(e),
    }
}

impl Inner {
    pub(super) async fn dataset_export(&self, a: DatasetExportArgs) -> Result<Output, Error> {
        let DatasetExportFormat::Coco = a.format;
        let explicit_out = match a.out {
            Some(raw) => {
                if raw.trim().is_empty() {
                    return Err(invalid_arguments("out must not be empty"));
                }
                let p = Path::new(&raw);
                let resolved = if p.is_absolute() { p.to_path_buf() } else { self.paths.data_dir.join(p) };
                Some(resolved)
            }
            None => None,
        };
        if let Some(ref out_dir) = explicit_out {
            check_export_out_inside(&self.paths, out_dir)?;
        }
        let write_root = self.dataset_write_root("dataset_export").await?;
        let (ds, path) = self.resolve_local_dataset("dataset_export", &a.dataset_id).await?;
        let src_fmt = to_data_format(ds.format);
        let ds_name = ds.name.clone();
        let confine_paths = self.confinement.active().then(|| self.paths.clone());
        let splits: Vec<folo_data::Split> =
            a.splits.unwrap_or_default().into_iter().map(to_data_split).collect();
        let image_mode = to_coco_image_mode(a.images);
        let opts =
            CocoExportOptions { splits, images: image_mode, pretty: a.pretty, description: a.description };

        let report =
            tokio::task::spawn_blocking(move || -> Result<folo_data::convert::CocoExportReport, Error> {
                let loaded = open_with(&path, &OpenOptions { format: src_fmt, verify_images: true })
                    .map_err(data_error)?;
                check_unreadable_splits(&loaded, &path)?;
                let (out_dir, cleanup_on_err) = match explicit_out {
                    Some(dir) => {
                        if let Some(paths) = confine_paths.as_ref() {
                            check_export_out_inside(paths, &dir)?;
                        }
                        check_output_outside_input(&dir, &path, &loaded)?;
                        ensure_empty_export_target(&dir)?;
                        (dir, false)
                    }
                    None => (allocate_export_dir(&write_root, &ds_name, confine_paths.as_ref())?, true),
                };
                export_coco(&loaded, &out_dir, &opts).map_err(|e| {
                    if cleanup_on_err {
                        let _ = std::fs::remove_dir_all(&out_dir);
                    }
                    export_error(e)
                })
            })
            .await
            .map_err(super::join)??;

        let exported_splits: Vec<DatasetExportSplitSummary> = report
            .exported_splits
            .iter()
            .map(|s| DatasetExportSplitSummary {
                split: from_data_split(s.split),
                annotation_file: folo_api::json::path(&s.annotation_file),
                images: s.images,
                annotations: s.annotations,
                linked_images: s.linked_images,
                copied_images: s.copied_images,
                referenced_images: s.referenced_images,
            })
            .collect();
        let total_images: usize = report.splits.iter().sum();
        let res = DatasetExportResult {
            dataset_id: a.dataset_id.clone(),
            path: folo_api::json::path(&report.root),
            format: DatasetExportFormat::Coco,
            images: from_coco_image_mode(report.image_mode),
            categories: report.categories,
            splits: Splits {
                train: report.splits[0] as u64,
                val: report.splits[1] as u64,
                test: report.splits[2] as u64,
            },
            boxes: report.boxes,
            exported_splits,
        };
        let mode_str = match res.images {
            DatasetExportImageMode::Copy => "copy",
            DatasetExportImageMode::Link => "link",
            DatasetExportImageMode::Reference => "reference",
        };
        let summary = format!(
            "exported {} to {} as COCO ({mode_str}): {total_images} images (train {}, val {}, test {}), {} boxes, \
             {} categories",
            a.dataset_id,
            res.path,
            res.splits.train,
            res.splits.val,
            res.splits.test,
            res.boxes,
            res.categories,
        );
        out(&res, summary)
    }
}
