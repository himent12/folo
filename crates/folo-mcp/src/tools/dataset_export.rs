//! `dataset_export`: spec-compliant COCO instances JSON export (`folo_data::convert::export_coco`) for
//! registered datasets, supporting split filtering, `copy` / `link` / `reference` image modes, confinement, and
//! safe cleanup on failure.

use std::path::{Path, PathBuf};

use folo_api::{Split, Splits};
use folo_data::convert::{CocoExportOptions, CocoImageMode, export_coco};
use folo_data::{DataError, LoadedDataset, OpenOptions, open_with};
use folo_server::Store;
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
        check_export_out_inside(paths, &root)?;
    }
    std::fs::create_dir_all(&root)
        .map_err(|e| Error::internal(format!("creating {}: {e}", folo_api::json::path(&root))))?;
    if let Some(paths) = confine_policy {
        check_export_out_inside(paths, &root)?;
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

/// Checks that `dir` is not equal to or inside the input dataset (used before creating `<write_root>/exports`
/// when `out` is omitted, while still allowing disjoint sibling folders under `exports/`).
fn check_dir_not_inside_input(dir: &Path, input: &Path, ds: &LoadedDataset) -> Result<(), Error> {
    let Some(canon_out) =
        std::fs::canonicalize(dir).ok().or_else(|| crate::paths::resolve_nonexistent(dir, 0))
    else {
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
            && canon_out.starts_with(&canon_in)
        {
            return Err(invalid_arguments(format!(
                "default export folder ({}) lies inside the input dataset ({}); pass an explicit `out` outside it",
                folo_api::json::path(dir),
                folo_api::json::path(p)
            )));
        }
    }
    Ok(())
}

/// Checks that `out_dir` does not overlap the input dataset or its source folders, reusing `paths::resolve_nonexistent`
/// so Windows `\\?\` prefixes, `..` traversals, case-insensitive ancestors, symlinks, and junctions all resolve
/// identically to `std::fs::canonicalize` on the dataset paths.
fn check_output_outside_input(out_dir: &Path, input: &Path, ds: &LoadedDataset) -> Result<(), Error> {
    let Some(canon_out) =
        std::fs::canonicalize(out_dir).ok().or_else(|| crate::paths::resolve_nonexistent(out_dir, 0))
    else {
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

fn export_error(e: DataError, confined: bool) -> Error {
    match &e {
        DataError::Io { source, .. }
            if matches!(
                source.kind(),
                std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::DirectoryNotEmpty
            ) =>
        {
            Error::new(StatusCode::CONFLICT, "output_exists", e.to_string())
        }
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::PermissionDenied && confined => {
            Error::new(StatusCode::FORBIDDEN, "path_outside_data_dir", e.to_string())
        }
        DataError::Io { .. } => Error::internal(e.to_string()),
        DataError::Config(m) => Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", m.clone()),
        _ => data_error(e),
    }
}

impl Inner {
    /// Resolves the write root `data_dir` for `dataset_export` without checking `<data_dir>/datasets` or touching
    /// disk in `--mock` / remote proxy modes.
    async fn export_write_root(&self) -> Result<PathBuf, Error> {
        let Some(proxy) = &self.proxy else {
            if self.blocking(|s| Ok(s.is_mock())).await? {
                return Err(Error::new(
                    StatusCode::NOT_IMPLEMENTED,
                    "not_supported",
                    "dataset_export writes dataset files and is not supported in --mock mode",
                )
                .hint("run `folo mcp` without --mock"));
            }
            return Ok(self.opts.data_dir.clone());
        };
        let proxy = proxy.clone();
        let fallback = self.paths.data_dir.clone();
        tokio::task::spawn_blocking(move || -> Result<PathBuf, Error> {
            if !crate::confine::is_local_url(proxy.base()) {
                return Err(Error::new(
                    StatusCode::NOT_IMPLEMENTED,
                    "not_supported",
                    format!("dataset_export is not available through remote server {}", proxy.base()),
                )
                .hint("run `folo mcp` locally without --server, or run `folo dataset` on the server host"));
            }
            if proxy.is_mock() {
                return Err(Error::new(
                    StatusCode::NOT_IMPLEMENTED,
                    "not_supported",
                    "dataset_export writes dataset files and is not supported against a --mock server",
                )
                .hint("run against a real `folo serve` (without --mock) or embedded `folo mcp`"));
            }
            Ok(proxy.remote_data_dir()?.map(PathBuf::from).unwrap_or(fallback))
        })
        .await
        .map_err(super::join)?
    }

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
        let write_root = self.export_write_root().await?;
        let (ds, path) = self.resolve_local_dataset("dataset_export", &a.dataset_id).await?;
        let src_fmt = to_data_format(ds.format);
        let ds_name = ds.name.clone();
        let confine_paths = self.confinement.active().then(|| self.paths.clone());
        let confined = confine_paths.is_some();
        let splits: Vec<folo_data::Split> =
            a.splits.unwrap_or_default().into_iter().map(to_data_split).collect();
        let image_mode = to_coco_image_mode(a.images);
        let opts =
            CocoExportOptions { splits, images: image_mode, pretty: a.pretty, description: a.description };

        let report =
            tokio::task::spawn_blocking(move || -> Result<folo_data::convert::CocoExportReport, Error> {
                if let (Some(dir), Some(paths)) = (explicit_out.as_ref(), confine_paths.as_ref()) {
                    check_export_out_inside(paths, dir)?;
                }
                let loaded = open_with(&path, &OpenOptions { format: src_fmt, verify_images: true })
                    .map_err(|e| export_error(e, confined))?;
                check_unreadable_splits(&loaded, &path)?;
                let (out_dir, cleanup_on_err) = match explicit_out {
                    Some(dir) => {
                        check_output_outside_input(&dir, &path, &loaded)?;
                        ensure_empty_export_target(&dir)?;
                        (dir, false)
                    }
                    None => {
                        check_dir_not_inside_input(&write_root.join("exports"), &path, &loaded)?;
                        let dir = allocate_export_dir(&write_root, &ds_name, confine_paths.as_ref())?;
                        if let Err(err) = check_output_outside_input(&dir, &path, &loaded) {
                            let _ = std::fs::remove_dir_all(&dir);
                            return Err(err);
                        }
                        (dir, true)
                    }
                };
                export_coco(&loaded, &out_dir, &opts).map_err(|e| {
                    if cleanup_on_err {
                        let _ = std::fs::remove_dir_all(&out_dir);
                    }
                    export_error(e, confined)
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
