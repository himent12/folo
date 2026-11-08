//! `--confine` for datasets. A dataset root inside the data dir can still reference files outside it (a
//! `data.yaml` pointing elsewhere, image lists, symlinked or junctioned folders, YOLO label paths derived from image
//! paths). Two checks keep reads inside, both over-approximations that do not depend on how the store parses a
//! dataset:
//!
//! - [`Confinement::precheck_root`], before `register_dataset` lets the store open anything:
//!   1. no link or junction anywhere in the data dir points outside it, so every path that stays under the data dir
//!      lexically also stays there on disk;
//!   2. every token of every manifest in the root (`data.yaml`, `dataset.yaml`, `data.yml`; any key, any
//!      indentation, scalars, flow and block lists), taken as a path relative to the root and to every other token,
//!      either does not exist or resolves inside, and so does every line of every image-list file such a token names;
//!   3. no component of the data dir's own path is named `images`, because the YOLO label of an image is found by
//!      replacing the last `images` component of its path with `labels`, which would then leave the data dir.
//! - [`Confinement::dataset`], before any tool or resource reads a dataset (also ones registered in an earlier
//!   session without `--confine`): the root, every image file the store reports and each image's derived label
//!   resolve inside. Results are cached per dataset id.
//!
//! Paths are compared canonically (symlinks and junctions resolved). The checks guard what can be reached through
//! FOLO's tools; they do not defend against another process re-pointing links inside the data dir after a check
//! (FOLO itself never creates links). Procedural mock datasets (`mock://`) read no files.

use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use folo_server::Store;
use folo_server::store::ImageQuery;
use http::StatusCode;

use crate::errors::Error;
use crate::paths::PathPolicy;

/// Page size used to walk a dataset's images.
const WALK_PAGE: u64 = 1000;
/// Most directory entries [`Confinement::precheck_root`] visits.
const MAX_ENTRIES: usize = 5_000_000;
/// YOLO manifests a dataset root may hold.
const MANIFESTS: [&str; 3] = ["data.yaml", "dataset.yaml", "data.yml"];
/// Largest manifest or image-list file read by the pre-check.
const MAX_LIST_BYTES: u64 = 64 * 1024 * 1024;

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
        let r = self.scan(store, id);
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

    /// Checks a folder before it is registered (see the module docs). Blocking.
    pub fn precheck_root(&self, root: &Path) -> Result<(), Error> {
        if !self.policy.confine {
            return Ok(());
        }
        let data = self.policy.canonical_root()?;
        if data.components().any(|c| c.as_os_str() == "images") {
            return Err(Error::new(
                StatusCode::FORBIDDEN,
                "path_outside_data_dir",
                format!(
                    "the data dir {} is inside a folder named `images`, so YOLO label paths (images → labels) \
                     would leave it",
                    data.display()
                ),
            )
            .hint("use a --data-dir whose path has no `images` component"));
        }
        self.check_links(&data)?;
        for m in MANIFESTS {
            let manifest = root.join(m);
            if manifest.is_file() {
                let text = read_small(&manifest)?;
                self.check_tokens(root, &manifest_tokens(&text))?;
            }
        }
        Ok(())
    }

    /// Every link and junction under `dir` (not followed) must resolve inside the data dir.
    fn check_links(&self, dir: &Path) -> Result<(), Error> {
        let mut stack = vec![dir.to_path_buf()];
        let mut entries = 0usize;
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                entries += 1;
                if entries > MAX_ENTRIES {
                    return Err(Error::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_dataset",
                        format!(
                            "{} has more than {MAX_ENTRIES} entries to verify under --confine",
                            dir.display()
                        ),
                    ));
                }
                let Ok(ft) = e.file_type() else { continue };
                // `is_symlink` is also true for Windows junctions (name-surrogate reparse points).
                if ft.is_symlink() {
                    self.policy.check_files([e.path().as_path()])?;
                } else if ft.is_dir() {
                    stack.push(e.path());
                }
            }
        }
        Ok(())
    }

    /// Takes every token as a path relative to the root and to every other token (any of them may be the store's
    /// base), and checks every one that exists; files among them are read as image lists and their lines checked
    /// the same way.
    fn check_tokens(&self, root: &Path, tokens: &BTreeSet<String>) -> Result<(), Error> {
        // Possible bases: the root, and every token that is a folder or could reach one lexically (`..`, absolute).
        // Class names and other plain words are neither, which keeps this linear in practice.
        let mut bases = vec![root.to_path_buf()];
        bases.extend(
            tokens
                .iter()
                .filter(|t| t.contains("..") || Path::new(t.as_str()).is_absolute() || root.join(t).is_dir())
                .map(|t| root.join(t)),
        );
        let mut lists = Vec::new();
        for base in &bases {
            for t in tokens {
                let p = base.join(t);
                self.policy.check_files([p.as_path(), base.as_path()])?;
                if p.is_file() {
                    lists.push(p);
                }
            }
        }
        for list in lists {
            let text = read_small(&list)?;
            let dir = list.parent().unwrap_or(root).to_path_buf();
            for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
                for base in bases.iter().chain(std::iter::once(&dir)) {
                    self.policy.check_files([base.join(line).as_path()])?;
                }
            }
        }
        Ok(())
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
                return Ok(());
            }
        }
    }
}

fn read_small(path: &Path) -> Result<String, Error> {
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if len > MAX_LIST_BYTES {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_dataset",
            format!("{} is too large to verify under --confine", path.display()),
        ));
    }
    std::fs::read(path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .map_err(|e| Error::internal(format!("{}: {e}", path.display())))
}

/// Every value-like token of a YAML manifest, whatever its key or indentation. A superset of what a YAML reader
/// can make of a line: each line whole and cut at every ` #` (comments), the text after every `key: ` and `- `, both
/// whole and split at commas (flow lists and maps), with quotes and brackets trimmed. Extra tokens only cost a
/// check; a token that names no existing file is skipped.
fn manifest_tokens(text: &str) -> BTreeSet<String> {
    let clean = |s: &str| {
        s.trim().trim_matches(|c| matches!(c, '[' | ']' | '{' | '}' | '"' | '\'')).trim().to_string()
    };
    let mut out = BTreeSet::new();
    let mut add = |s: &str| {
        let t = clean(s);
        if !t.is_empty() {
            out.insert(t);
        }
    };
    for raw in text.lines() {
        // The line cut at each ` #` (a YAML comment, unless it is inside quotes), and whole.
        let cuts = raw.match_indices(" #").map(|(i, _)| &raw[..i]);
        for line in std::iter::once(raw).chain(cuts) {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut rest = line;
            while let Some(r) = rest.strip_prefix("- ") {
                rest = r.trim_start();
            }
            let mut values = vec![rest, rest.strip_suffix(':').unwrap_or(rest)];
            values.extend(rest.match_indices(": ").map(|(i, _)| &rest[i + 2..]));
            for v in values {
                add(v);
                for part in v.split(',') {
                    add(part);
                    if let Some((_, after)) = part.rsplit_once(": ") {
                        add(after);
                    }
                }
            }
        }
    }
    out
}

/// The YOLO label of an image: the last `images` component becomes `labels`, the extension `.txt` (the stores'
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
    fn manifest_tokens_take_every_value() {
        let yaml = "# comment\npath: ../elsewhere  # base\n  train: ../../private.txt\nval: [a.txt, 'b']\ntest:\n  \
                    - one\n  - \"two\"\nnames:\n  0: person\nnc: 2\n";
        let got: Vec<String> = manifest_tokens(yaml).into_iter().collect();
        // Bare `key:` lines count too (harmless: a token that names no existing file is skipped).
        for want in ["../../private.txt", "../elsewhere", "a.txt", "b", "one", "person", "two"] {
            assert!(got.iter().any(|g| g == want), "{want} missing from {got:?}");
        }
        // Whole values are kept next to their comma parts, and comments are tried both ways.
        let got =
            manifest_tokens("train: ../../secret,list.txt\nval: 'a #b'  # c\nnames: {0: cat, 1: dog}\n");
        for want in ["../../secret,list.txt", "../../secret", "list.txt", "a #b", "a", "cat", "dog"] {
            assert!(got.contains(want), "{want} missing from {got:?}");
        }
    }

    #[test]
    fn labels_follow_the_last_images_component() {
        let p = label_path(Path::new("/d/images/x/images/train/a.jpg"));
        assert_eq!(p, Path::new("/d/images/x/labels/train/a.txt"));
        assert_eq!(label_path(Path::new("/d/a.jpg")), Path::new("/d/a.txt"));
    }
}
