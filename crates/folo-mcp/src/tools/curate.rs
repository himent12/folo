//! Dataset audit & curation tools: `audit_dataset` (exact + perceptual dHash leakage), `merge_datasets`, and
//! `filter_dataset`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use folo_api::{DatasetFormat, RegisterDataset, Split};
use folo_data::{
    AuditOptions, ClassNormalize, CollisionPolicy, DataError, FilterConfig, Format, MergeConfig,
    MergeSplitPolicy, OpenOptions, SplitConfig, convert, filter_dataset, open_with, write_merged,
};
use folo_server::Store as _;
use http::StatusCode;

use super::arguments::check_name;
use super::results::{MAX_RESULT_BYTES, Output, json_len, out};
use crate::errors::{Error, invalid_arguments};
use crate::server::Inner;
use crate::types::*;

const MAX_AUDIT_GROUPS: usize = 50;
const DEFAULT_AUDIT_GROUPS: usize = 20;

fn to_class_normalize(m: ClassNormalizeMode) -> ClassNormalize {
    match m {
        ClassNormalizeMode::Exact => ClassNormalize::Exact,
        ClassNormalizeMode::CaseAndWhitespace => ClassNormalize::CaseAndWhitespace,
    }
}

fn to_data_format(f: DatasetFormat) -> Format {
    match f {
        DatasetFormat::Auto => Format::Auto,
        DatasetFormat::Yolo => Format::Yolo,
        DatasetFormat::Coco => Format::Coco,
    }
}

fn to_data_split(s: Split) -> folo_data::Split {
    match s {
        Split::Train => folo_data::Split::Train,
        Split::Val => folo_data::Split::Val,
        Split::Test => folo_data::Split::Test,
    }
}

fn slug(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
        if s.len() >= 48 {
            break;
        }
    }
    let trimmed = s.trim_matches('-');
    if trimmed.is_empty() { "dataset".to_string() } else { trimmed.to_string() }
}

/// Picks and creates a fresh empty directory under `<base>/datasets/<slug>[-N]`.
fn allocate_dataset_dir(base: &Path, name: &str) -> Result<PathBuf, Error> {
    let root = base.join("datasets");
    std::fs::create_dir_all(&root)
        .map_err(|e| Error::internal(format!("creating {}: {e}", folo_api::json::path(&root))))?;
    let stem = slug(name);
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
    Err(Error::internal(format!("could not allocate a unique folder under {}", folo_api::json::path(&root))))
}

/// Maps a [`DataError`] to an MCP tool error.
fn data_error(e: DataError) -> Error {
    match &e {
        DataError::Config(m) => {
            if m.contains("nothing to merge")
                || m.contains("only has test images")
                || m.contains("no readable images")
            {
                Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", m.clone())
            } else {
                invalid_arguments(m.clone())
            }
        }
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::PermissionDenied => {
            Error::new(StatusCode::FORBIDDEN, "path_outside_data_dir", e.to_string())
        }
        _ => Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", e.to_string()),
    }
}

fn dataset_counts(ds: &folo_data::LoadedDataset) -> (usize, usize) {
    let mut images = 0usize;
    let mut boxes = 0usize;
    for split in [folo_data::Split::Train, folo_data::Split::Val, folo_data::Split::Test] {
        let entries = ds.split(split).entries();
        images += entries.len();
        boxes += entries.iter().map(|e| e.boxes.len()).sum::<usize>();
    }
    (images, boxes)
}

impl Inner {
    /// Resolves a registered dataset's on-disk path for local disk tools (`audit_dataset`, `merge_datasets`,
    /// `filter_dataset`). Refuses mock procedural datasets and remote non-local proxies.
    async fn resolve_local_dataset(
        &self,
        tool: &str,
        dataset_id: &str,
    ) -> Result<(folo_api::Dataset, PathBuf), Error> {
        if let Some(proxy) = &self.proxy
            && !crate::confine::is_local_url(proxy.base())
        {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_supported",
                format!(
                    "{tool} requires local filesystem access and is not available through remote server {}",
                    proxy.base()
                ),
            )
            .hint("run `folo mcp` locally without --server, or run `folo dataset` on the server host"));
        }
        self.ensure_dataset(dataset_id).await?;
        let id = dataset_id.to_string();
        let (ds, root) = self
            .blocking(move |s| {
                let ds = s.get_dataset(&id)?;
                let root = s.dataset_root(&id)?;
                Ok((ds, root))
            })
            .await?;
        if ds.path.starts_with("mock://") {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_supported",
                format!(
                    "{tool} requires on-disk files, but dataset '{}' is a procedural mock dataset",
                    ds.id
                ),
            )
            .hint("use a real dataset registered from disk (or `folo mcp` without --mock)"));
        }
        let path = root.unwrap_or_else(|| PathBuf::from(&ds.path));
        if self.confinement.active() {
            let paths = self.paths.clone();
            let check_path = path.clone();
            tokio::task::spawn_blocking(move || paths.resolve(&check_path.to_string_lossy()))
                .await
                .map_err(super::join)??;
        }
        Ok((ds, path))
    }

    /// Resolves the root `data_dir` where newly created datasets (`merge_datasets`, `filter_dataset`) are written.
    async fn dataset_write_root(&self, tool: &str) -> Result<PathBuf, Error> {
        let Some(proxy) = &self.proxy else {
            return Ok(self.opts.data_dir.clone());
        };
        let proxy = proxy.clone();
        let paths = self.paths.clone();
        let confine = self.confinement.active();
        let tool = tool.to_string();
        tokio::task::spawn_blocking(move || -> Result<PathBuf, Error> {
            if !crate::confine::is_local_url(proxy.base()) {
                return Err(Error::new(
                    StatusCode::NOT_IMPLEMENTED,
                    "not_supported",
                    format!("{tool} is not available through remote server {}", proxy.base()),
                )
                .hint("run `folo mcp` locally without --server, or run `folo dataset` on the server host"));
            }
            if proxy.is_mock() {
                return Err(Error::new(
                    StatusCode::NOT_IMPLEMENTED,
                    "not_supported",
                    format!("{tool} writes dataset files and is not supported against a --mock server"),
                )
                .hint("run against a real `folo serve` (without --mock) or embedded `folo mcp`"));
            }
            let dir = proxy.remote_data_dir()?.map(PathBuf::from).unwrap_or_else(|| paths.data_dir.clone());
            if confine {
                paths.check_dir_inside(&dir.join("datasets"))?;
            }
            Ok(dir)
        })
        .await
        .map_err(super::join)?
    }

    /// Registers a newly written dataset folder and cleans up the folder if registration or confinement fails.
    async fn register_written_dataset(
        &self,
        name: String,
        out_dir: PathBuf,
        format: DatasetFormat,
    ) -> Result<folo_api::Dataset, Error> {
        let req = RegisterDataset { name: Some(name), path: folo_api::json::path(&out_dir), format };
        let reg = self.blocking(move |s| s.register_dataset_tracked(req)).await;
        let (dataset, rollback) = match reg {
            Ok(pair) => pair,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&out_dir);
                return Err(e);
            }
        };
        if let Err(e) = self.ensure_dataset(&dataset.id).await {
            if let Some(token) = rollback {
                let (id, c) = (dataset.id.clone(), self.confinement.clone());
                let _ = self.blocking(move |s| s.delete_dataset_if_unused(&id, &token)).await;
                c.forget(&dataset.id);
            } else if self.proxy.is_none() {
                let (id, c) = (dataset.id.clone(), self.confinement.clone());
                let _ = self.blocking(move |s| s.delete_dataset(&id)).await;
                c.forget(&dataset.id);
            }
            let _ = std::fs::remove_dir_all(&out_dir);
            return Err(e);
        }
        Ok(dataset)
    }

    pub(super) async fn audit_dataset(&self, a: AuditDatasetArgs) -> Result<Output, Error> {
        if !a.perceptual && a.dhash_threshold.is_some() {
            return Err(invalid_arguments("dhash_threshold requires `perceptual: true`"));
        }
        if let Some(t) = a.dhash_threshold
            && t > 64
        {
            return Err(invalid_arguments(format!("dhash_threshold must be in 0..=64, got {t}")));
        }
        let max_groups = a.max_groups.unwrap_or(DEFAULT_AUDIT_GROUPS);
        if !(1..=MAX_AUDIT_GROUPS).contains(&max_groups) {
            return Err(invalid_arguments(format!(
                "max_groups must be in 1..={MAX_AUDIT_GROUPS}, got {max_groups}"
            )));
        }
        let (ds, path) = self.resolve_local_dataset("audit_dataset", &a.dataset_id).await?;
        let threshold = a.perceptual.then(|| a.dhash_threshold.unwrap_or(folo_data::DEFAULT_DHASH_THRESHOLD));
        let fmt = to_data_format(ds.format);
        let audit_res = tokio::task::spawn_blocking(move || -> Result<AuditDatasetResult, Error> {
            let loaded =
                open_with(&path, &OpenOptions { format: fmt, verify_images: false }).map_err(data_error)?;
            let images = dataset_counts(&loaded).0;
            let opts = match threshold {
                Some(t) => AuditOptions::perceptual(t),
                None => AuditOptions::default(),
            };
            let report = loaded.audit_with(&opts);
            let rel = |p: &Path| folo_data::paths::relative_to(p, &loaded.root);
            let exact_leakage_examples: Vec<Vec<String>> = report
                .split_leakage
                .iter()
                .take(max_groups)
                .map(|g| g.images.iter().map(|i| rel(&i.path)).collect())
                .collect();
            let near_leakage_examples: Vec<Vec<String>> = report
                .near_duplicate_leakage
                .iter()
                .take(max_groups)
                .map(|g| g.images.iter().map(|i| rel(&i.path)).collect())
                .collect();
            let near_duplicate_examples: Vec<Vec<String>> = report
                .near_duplicate_groups
                .iter()
                .take(max_groups)
                .map(|g| g.images.iter().map(|i| rel(&i.path)).collect())
                .collect();
            let has_split_leakage =
                !report.split_leakage.is_empty() || !report.near_duplicate_leakage.is_empty();
            Ok(AuditDatasetResult {
                dataset_id: ds.id,
                images,
                exact_duplicate_groups: report.duplicate_groups.len(),
                exact_leakage_groups: report.split_leakage.len(),
                perceptual_enabled: threshold.is_some(),
                dhash_threshold: threshold,
                near_duplicate_groups: report.near_duplicate_groups.len(),
                near_leakage_groups: report.near_duplicate_leakage.len(),
                has_split_leakage,
                exact_leakage_examples,
                near_leakage_examples,
                near_duplicate_examples,
            })
        })
        .await
        .map_err(super::join)??;

        let mut res = audit_res;
        while json_len(&res) > MAX_RESULT_BYTES {
            if !res.near_duplicate_examples.is_empty() {
                res.near_duplicate_examples.pop();
            } else if !res.near_leakage_examples.is_empty() {
                res.near_leakage_examples.pop();
            } else if !res.exact_leakage_examples.is_empty() {
                res.exact_leakage_examples.pop();
            } else {
                break;
            }
        }
        let summary = if res.perceptual_enabled {
            format!(
                "{}: {} images, {} exact leakage / {} near-duplicate leakage (dHash <= {}) / {} near-dup groups",
                res.dataset_id,
                res.images,
                res.exact_leakage_groups,
                res.near_leakage_groups,
                res.dhash_threshold.unwrap_or(0),
                res.near_duplicate_groups,
            )
        } else {
            format!(
                "{}: {} images, {} exact duplicate groups, {} exact leakage groups",
                res.dataset_id, res.images, res.exact_duplicate_groups, res.exact_leakage_groups
            )
        };
        out(&res, summary)
    }

    pub(super) async fn merge_datasets(&self, a: MergeDatasetsArgs) -> Result<Output, Error> {
        check_name("name", &a.name)?;
        if !(2..=32).contains(&a.dataset_ids.len()) {
            return Err(invalid_arguments("dataset_ids must contain 2 to 32 dataset ids"));
        }
        if a.drop_unmapped && a.class_map.as_ref().is_none_or(std::collections::BTreeMap::is_empty) {
            return Err(invalid_arguments("drop_unmapped requires a non-empty class_map"));
        }
        if a.splits != MergeSplitMode::Stratified && (a.val.is_some() || a.test.is_some() || a.seed.is_some())
        {
            return Err(invalid_arguments("val, test and seed require `splits: \"stratified\"`"));
        }
        if let Some(prefixes) = &a.source_prefixes
            && prefixes.len() != a.dataset_ids.len()
        {
            return Err(invalid_arguments(format!(
                "source_prefixes length ({}) must match dataset_ids length ({})",
                prefixes.len(),
                a.dataset_ids.len()
            )));
        }
        let splits = match a.splits {
            MergeSplitMode::Preserve => MergeSplitPolicy::Preserve,
            MergeSplitMode::PoolToTrain => MergeSplitPolicy::PoolToTrain,
            MergeSplitMode::Stratified => {
                let val = a.val.unwrap_or(0.2);
                let test = a.test.unwrap_or(0.0);
                if !(0.0..=1.0).contains(&val) || !(0.0..=1.0).contains(&test) || val + test >= 1.0 {
                    return Err(invalid_arguments(format!(
                        "val ({val}) and test ({test}) must be in 0..=1 with val + test < 1"
                    )));
                }
                let sc = SplitConfig::train_val_test(1.0 - val - test, val, test, a.seed.unwrap_or(42));
                sc.validate().map_err(|e| invalid_arguments(e.to_string()))?;
                MergeSplitPolicy::Stratified(sc)
            }
        };

        let write_root = self.dataset_write_root("merge_datasets").await?;
        let mut sources = Vec::with_capacity(a.dataset_ids.len());
        for id in &a.dataset_ids {
            let (ds, path) = self.resolve_local_dataset("merge_datasets", id).await?;
            sources.push((path, to_data_format(ds.format)));
        }

        let class_map: HashMap<String, Option<String>> =
            a.class_map.unwrap_or_default().into_iter().collect();
        let cfg = MergeConfig {
            normalize: to_class_normalize(a.normalize),
            class_map,
            drop_unmapped: a.drop_unmapped,
            drop_empty_images: a.drop_empty_images,
            splits,
            collisions: match a.collisions {
                MergeCollisionMode::PrefixOnCollision => CollisionPolicy::PrefixOnCollision,
                MergeCollisionMode::AlwaysPrefix => CollisionPolicy::AlwaysPrefix,
            },
            source_prefixes: a.source_prefixes.unwrap_or_default(),
        };
        let out_name = a.name.clone();
        let (out_dir, renamed_collisions) =
            tokio::task::spawn_blocking(move || -> Result<(PathBuf, usize), Error> {
                let mut loaded = Vec::with_capacity(sources.len());
                for (path, fmt) in sources {
                    loaded.push(
                        open_with(&path, &OpenOptions { format: fmt, verify_images: false })
                            .map_err(data_error)?,
                    );
                }
                let refs: Vec<&folo_data::LoadedDataset> = loaded.iter().collect();
                let out_dir = allocate_dataset_dir(&write_root, &out_name)?;
                let report = write_merged(&refs, &out_dir, &cfg).map_err(|e| {
                    let _ = std::fs::remove_dir_all(&out_dir);
                    data_error(e)
                })?;
                Ok((out_dir, report.renamed_collisions))
            })
            .await
            .map_err(super::join)??;

        let dataset = self.register_written_dataset(a.name, out_dir, DatasetFormat::Yolo).await?;
        let summary = format!(
            "merged {} datasets into {} ({}): {} classes, {} train / {} val / {} test images",
            a.dataset_ids.len(),
            dataset.name,
            dataset.id,
            dataset.classes.len(),
            dataset.splits.train,
            dataset.splits.val,
            dataset.splits.test,
        );
        out(&MergeDatasetsResult { dataset, renamed_collisions }, summary)
    }

    pub(super) async fn filter_dataset(&self, a: FilterDatasetArgs) -> Result<Output, Error> {
        check_name("name", &a.name)?;
        let out_fmt = match a.format.unwrap_or(DatasetFormat::Yolo) {
            DatasetFormat::Yolo => (DatasetFormat::Yolo, Format::Yolo),
            DatasetFormat::Coco => (DatasetFormat::Coco, Format::Coco),
            DatasetFormat::Auto => {
                return Err(invalid_arguments(
                    "filter_dataset output format must be `yolo` or `coco`, not `auto`",
                ));
            }
        };
        for (name, val) in [("min_box_area", a.min_box_area), ("max_box_area", a.max_box_area)] {
            if let Some(v) = val
                && !(0.0..=1.0).contains(&v)
            {
                return Err(invalid_arguments(format!("{name} must be in 0..=1, got {v}")));
            }
        }
        for (name, val) in [
            ("min_box_width_px", a.min_box_width_px),
            ("min_box_height_px", a.min_box_height_px),
            ("max_box_width_px", a.max_box_width_px),
            ("max_box_height_px", a.max_box_height_px),
        ] {
            if let Some(v) = val
                && (!v.is_finite() || v < 0.0)
            {
                return Err(invalid_arguments(format!("{name} must be finite and >= 0, got {v}")));
            }
        }
        let cfg = FilterConfig {
            include_classes: a.include_classes,
            exclude_classes: a.exclude_classes.unwrap_or_default(),
            normalize: to_class_normalize(a.normalize),
            compact_classes: a.compact_classes,
            splits: a.splits.map(|v| v.into_iter().map(to_data_split).collect()),
            min_box_area: a.min_box_area,
            max_box_area: a.max_box_area,
            min_box_width_px: a.min_box_width_px,
            min_box_height_px: a.min_box_height_px,
            max_box_width_px: a.max_box_width_px,
            max_box_height_px: a.max_box_height_px,
            min_image_width: a.min_width,
            min_image_height: a.min_height,
            max_image_width: a.max_width,
            max_image_height: a.max_height,
            probe_image_sizes: false,
            drop_empty_images: a.drop_empty_images,
        };
        cfg.validate().map_err(|e| invalid_arguments(e.to_string()))?;

        let write_root = self.dataset_write_root("filter_dataset").await?;
        let (ds, path) = self.resolve_local_dataset("filter_dataset", &a.dataset_id).await?;
        let src_fmt = to_data_format(ds.format);
        let out_name = a.name.clone();
        let target_fmt = out_fmt.1;
        let (out_dir, src_images, src_boxes, kept_images, kept_boxes) =
            tokio::task::spawn_blocking(move || -> Result<(PathBuf, usize, usize, usize, usize), Error> {
                let loaded = open_with(&path, &OpenOptions { format: src_fmt, verify_images: false })
                    .map_err(data_error)?;
                let (src_images, src_boxes) = dataset_counts(&loaded);
                let filtered = filter_dataset(&loaded, &cfg).map_err(data_error)?;
                let (kept_images, kept_boxes) = dataset_counts(&filtered);
                let out_dir = allocate_dataset_dir(&write_root, &out_name)?;
                convert::convert(&filtered, &out_dir, target_fmt).map_err(|e| {
                    let _ = std::fs::remove_dir_all(&out_dir);
                    data_error(e)
                })?;
                Ok((out_dir, src_images, src_boxes, kept_images, kept_boxes))
            })
            .await
            .map_err(super::join)??;

        let dataset = self.register_written_dataset(a.name, out_dir, out_fmt.0).await?;
        let dropped_images = src_images.saturating_sub(kept_images);
        let dropped_boxes = src_boxes.saturating_sub(kept_boxes);
        let summary = format!(
            "filtered {} into {} ({}): kept {kept_images}/{src_images} images, {kept_boxes}/{src_boxes} boxes, {} classes",
            a.dataset_id,
            dataset.name,
            dataset.id,
            dataset.classes.len(),
        );
        out(
            &FilterDatasetResult {
                dataset,
                source_images: src_images,
                source_boxes: src_boxes,
                kept_images,
                kept_boxes,
                dropped_images,
                dropped_boxes,
            },
            summary,
        )
    }
}
