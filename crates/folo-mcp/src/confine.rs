//! `--confine` for datasets. A dataset root inside the data dir can still reference files outside it (a
//! `data.yaml` pointing elsewhere, image lists, linked folders, YOLO labels derived from image paths). Two layers:
//!
//! - **Enforcement**: [`install_read_guard`] installs a [`folo_data::guard`] for the process, so every file and folder
//!   the dataset reader (and the server's image reads) opens is checked first: its canonical path (symlinks and
//!   junctions resolved) must be inside the data dir, or the read fails. This holds for registration and for every
//!   later re-index, whatever the manifest says, because the check sits where the files are opened.
//! - **Reporting**: [`Confinement::dataset`], before any tool or resource reads a dataset: the root, every image the
//!   store reports, each image's label, and the dataset's issues (a refused read shows up as one) are checked, so the
//!   agent gets a clear `path_outside_data_dir` error, and such datasets (also ones registered in an earlier session
//!   without `--confine`) are hidden. Results are cached per dataset id; the guard keeps enforcing after a dataset
//!   changes.
//!
//! The checks guard what can be reached through FOLO's tools; they do not defend against another process re-pointing
//! links inside the data dir between a check and the open (FOLO itself never creates links). In proxy mode the
//! datasets are read by `folo serve`, so only the files the MCP process reads itself are confined. Procedural mock
//! datasets (`mock://`) read no files.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use folo_server::Store;
use folo_server::store::{ImageQuery, IssueQuery};
use http::StatusCode;

use crate::errors::Error;
use crate::paths::PathPolicy;

/// Page size used to walk a dataset's images and issues.
const WALK_PAGE: u64 = 1000;

/// Installs the process-wide read guard for `data_dir` (see the module docs). Paths that cannot be resolved are let
/// through: they cannot be opened either.
pub fn install_read_guard(policy: &PathPolicy) -> Result<(), Error> {
    let root = policy.canonical_root()?;
    folo_data::guard::install(move |p: &Path| {
        std::fs::canonicalize(p).map_or(true, |c| c.starts_with(&root))
    });
    Ok(())
}

/// The `path_outside_data_dir` error for a read the guard refused (recognised by [`folo_data::guard::DENIED`]).
pub fn map_denied(e: Error) -> Error {
    if e.body.code != "path_outside_data_dir" && e.body.message.contains(folo_data::guard::DENIED) {
        Error::new(StatusCode::FORBIDDEN, "path_outside_data_dir", e.body.message)
            .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine")
    } else {
        e
    }
}

/// The dataset check of `--confine`. Cheap to clone.
#[derive(Clone)]
pub struct Confinement {
    policy: PathPolicy,
    cache: Arc<Mutex<HashMap<String, Result<(), Error>>>>,
}

impl Confinement {
    /// A checker for `policy` (a no-op without `confine`).
    pub fn new(policy: PathPolicy) -> Confinement {
        Confinement { policy, cache: Arc::default() }
    }

    /// True when the check does anything.
    pub fn active(&self) -> bool {
        self.policy.confine
    }

    /// `Ok` when dataset `id` reads nothing outside the data dir (always without `--confine`). Blocking.
    pub fn dataset(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        if !self.policy.confine {
            return Ok(());
        }
        if let Some(r) = self.cache.lock().unwrap().get(id) {
            return r.clone();
        }
        let r = self.scan(store, id).map_err(map_denied);
        // Unknown datasets are not cached: they may be registered later under that id.
        if r.as_ref().err().is_none_or(|e| e.body.code != "dataset_not_found") {
            self.cache.lock().unwrap().insert(id.to_string(), r.clone());
        }
        r
    }

    /// Drops the cached result of `id` (after it was unregistered).
    pub fn forget(&self, id: &str) {
        self.cache.lock().unwrap().remove(id);
    }

    fn scan(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        let d = store.get_dataset(id)?;
        if d.path.starts_with("mock://") {
            return Ok(());
        }
        let root = PathBuf::from(&d.path);
        self.policy.check_files([root.as_path()]).map_err(|e| in_dataset(e, id))?;
        let mut offset = 0;
        loop {
            let q = ImageQuery { split: None, offset, limit: WALK_PAGE, class: None };
            let page = store.list_images(id, &q)?;
            let n = page.images.len() as u64;
            let mut files = Vec::with_capacity(page.images.len() * 2);
            for im in &page.images {
                let f = Path::new(&im.file);
                let img = if f.is_absolute() { f.to_path_buf() } else { root.join(f) };
                files.push(label_path(&img));
                files.push(img);
            }
            self.policy.check_files(files.iter().map(PathBuf::as_path)).map_err(|e| in_dataset(e, id))?;
            offset += n;
            if n == 0 || offset >= page.total {
                break;
            }
        }
        // A read the guard refused while indexing is reported as an issue (an unreadable list, image or label).
        let mut offset = 0;
        loop {
            let page =
                store.list_issues(id, &IssueQuery { offset, limit: WALK_PAGE, ..IssueQuery::default() })?;
            if let Some(i) = page.issues.iter().find(|i| i.message.contains(folo_data::guard::DENIED)) {
                return Err(in_dataset(
                    Error::new(StatusCode::FORBIDDEN, "path_outside_data_dir", i.message.clone()),
                    id,
                ));
            }
            let n = page.issues.len() as u64;
            offset += n;
            if n == 0 || offset >= page.total {
                return Ok(());
            }
        }
    }
}

/// The YOLO label of an image: the last `images` component becomes `labels`, the extension `.txt` (the reader's
/// convention, docs/DATA.md).
fn label_path(img: &Path) -> PathBuf {
    let comps: Vec<Component> = img.components().collect();
    let idx = comps.iter().rposition(|c| c.as_os_str() == "images");
    let mut p = PathBuf::new();
    for (i, c) in comps.iter().enumerate() {
        p.push(if Some(i) == idx { "labels".as_ref() } else { c.as_os_str() });
    }
    p.set_extension("txt");
    p
}

fn in_dataset(mut e: Error, id: &str) -> Error {
    if e.body.code == "path_outside_data_dir" {
        e.body.message = format!("dataset '{id}' reads files outside the data dir: {}", e.body.message);
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_follow_the_last_images_component() {
        let p = label_path(Path::new("/d/images/x/images/train/a.jpg"));
        assert_eq!(p, Path::new("/d/images/x/labels/train/a.txt"));
        assert_eq!(label_path(Path::new("/d/a.jpg")), Path::new("/d/a.txt"));
    }

    #[test]
    fn denied_reads_become_path_outside_data_dir() {
        let e = Error::internal(format!("reading x: x {}", folo_data::guard::DENIED));
        assert_eq!(map_denied(e).body.code, "path_outside_data_dir");
        assert_eq!(map_denied(Error::internal("other")).body.code, "internal");
    }
}
