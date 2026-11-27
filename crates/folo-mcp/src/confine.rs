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
//! datasets (`mock://`) read no files. [`Confinement::label_job`] checks a labeling job's images the same way.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use folo_server::Store;
use folo_server::store::{ImageQuery, IssueQuery};
use http::StatusCode;

use crate::errors::Error;
use crate::paths::PathPolicy;
use crate::proxy::HttpStore;

/// Parses legacy/non-canonical IPv4 literals accepted by POSIX `inet_aton` / OS resolvers (`127.1`, `0`,
/// `2130706433`, `0x7f000001`, `0177.0.0.1`) so they never slip past `std::net::IpAddr::from_str`.
fn parse_legacy_ipv4(host: &str) -> Option<std::net::Ipv4Addr> {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.is_empty() || parts.len() > 4 || parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let parse_part = |s: &str| -> Option<u64> {
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return None;
            }
            u64::from_str_radix(hex, 16).ok()
        } else if s.len() > 1 && s.starts_with('0') {
            if !s.bytes().all(|b| matches!(b, b'0'..=b'7')) {
                return None;
            }
            u64::from_str_radix(s, 8).ok()
        } else {
            if !s.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            s.parse::<u64>().ok()
        }
    };
    let nums: Vec<u64> = parts.into_iter().map(parse_part).collect::<Option<_>>()?;
    let addr = match nums.as_slice() {
        [a] if *a <= 0xffff_ffff => *a as u32,
        [a, b] if *a <= 0xff && *b <= 0x00ff_ffff => ((*a as u32) << 24) | (*b as u32),
        [a, b, c] if *a <= 0xff && *b <= 0xff && *c <= 0xffff => {
            ((*a as u32) << 24) | ((*b as u32) << 16) | (*c as u32)
        }
        [a, b, c, d] if *a <= 0xff && *b <= 0xff && *c <= 0xff && *d <= 0xff => {
            ((*a as u32) << 24) | ((*b as u32) << 16) | ((*c as u32) << 8) | (*d as u32)
        }
        _ => return None,
    };
    Some(std::net::Ipv4Addr::from(addr))
}

/// True when `host` (trimmed of trailing dots) matches this machine's hostname (`COMPUTERNAME` / `HOSTNAME`),
/// optionally suffixed with `.local`.
fn is_local_hostname(host: &str) -> bool {
    let bare = host.strip_suffix(".local").or_else(|| host.strip_suffix(".LOCAL")).unwrap_or(host);
    ["COMPUTERNAME", "HOSTNAME"].iter().any(|var| {
        std::env::var(var)
            .ok()
            .is_some_and(|name| !name.is_empty() && name.trim_end_matches('.').eq_ignore_ascii_case(bare))
    })
}

/// True when `url` targets this machine (loopback, unspecified bind address, `localhost`, `*.localhost`, legacy
/// IPv4 loopback literals, local hostname, empty host, or non-http(s) scheme), meaning a proxied `folo serve` may
/// share the local filesystem with this MCP process.
pub(crate) fn is_local_url(url: &str) -> bool {
    let rest = match url.split_once("://") {
        Some((scheme, after))
            if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") =>
        {
            after
        }
        Some(_) => return true, // non-http(s) scheme: fail closed as local
        None => url,
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let host = if let Some(after) = authority.strip_prefix('[') {
        let raw = after.split(']').next().unwrap_or("");
        // Strip IPv6 zone identifier (`%eth0` or percent-encoded `%25lo`).
        raw.split('%').next().unwrap_or("")
    } else {
        authority.split(':').next().unwrap_or("")
    };
    let trimmed = host.trim_end_matches('.');
    if trimmed.is_empty() {
        return true; // empty host: fail closed as local
    }
    let suffix = b".localhost";
    if trimmed.eq_ignore_ascii_case("localhost")
        || (trimmed.len() > suffix.len()
            && trimmed.as_bytes()[trimmed.len() - suffix.len()..].eq_ignore_ascii_case(suffix))
        || is_local_hostname(trimmed)
    {
        return true;
    }
    match trimmed.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_loopback() || v4.is_unspecified(),
        Ok(std::net::IpAddr::V6(v6)) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback() || v4.is_unspecified())
        }
        Err(_) => parse_legacy_ipv4(trimmed).is_some_and(|v4| v4.is_loopback() || v4.is_unspecified()),
    }
}

/// Local-proxy cache validator: binds a cached verdict to the dataset's path, format, an on-disk
/// [`folo_server::Fingerprint`] over its minimized watched directories, AND a [`Confinement::canonical_signature`]
/// over all directory/symlink targets so both ordinary manifest edits and directory-to-symlink replacements
/// invalidate cached verdicts immediately (even within `folo serve`'s 5-second recheck window).
#[derive(Clone)]
struct ProxyTag {
    path: String,
    format: folo_api::DatasetFormat,
    watched: Vec<PathBuf>,
    fingerprint: folo_server::Fingerprint,
    canon_hash: u64,
}

/// Cached confinement verdict for one dataset id: optional [`ProxyTag`] (used in local proxy mode) and result.
type DatasetVerdict = (Option<ProxyTag>, Result<(), Error>);

/// A labeling job's check: a hash of its image files when checked, and the result.
type JobCheck = (u64, Result<(), Error>);

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
    /// Present in proxy mode (`--server`): a non-local server whose reported `data_dir` does not exist on this
    /// machine has its own separate filesystem; any local/loopback server or server whose `data_dir` exists on
    /// this filesystem stays strictly confined to `policy.data_dir`.
    proxy: Option<Arc<HttpStore>>,
    cache: Arc<Mutex<HashMap<String, DatasetVerdict>>>,
    /// Labeling job id -> (a hash of its image files when checked, the result).
    jobs: Arc<Mutex<HashMap<String, JobCheck>>>,
}

impl Confinement {
    /// A checker for `policy` (a no-op without `confine`).
    pub fn new(policy: PathPolicy, proxy: Option<Arc<HttpStore>>) -> Confinement {
        Confinement { policy, proxy, cache: Arc::default(), jobs: Arc::default() }
    }

    /// True when the check does anything.
    pub fn active(&self) -> bool {
        self.policy.confine
    }

    /// True when proxying a non-local URL (fast syntactic check, no network calls).
    pub fn has_non_local_proxy_url(&self) -> bool {
        self.proxy.as_ref().is_some_and(|p| !is_local_url(p.base()))
    }

    /// True ONLY when proxying a non-local server verified via `GET /api/system` to serve a separate remote
    /// `data_dir` that does not exist on this local filesystem. Fails closed (`Ok(false)`) for loopback/local
    /// URLs, servers omitting `data_dir`, and servers whose `data_dir` exists locally (e.g., a local `folo serve`
    /// reached over a LAN IP or hostname, or a shared mount). Transient `/api/system` errors propagate without
    /// being cached.
    fn is_verified_remote_proxy(&self) -> Result<bool, Error> {
        let Some(proxy) = &self.proxy else {
            return Ok(false);
        };
        if is_local_url(proxy.base()) {
            return Ok(false);
        }
        let Some(remote_dir) = proxy.remote_data_dir()? else {
            return Ok(false);
        };
        let p = Path::new(&remote_dir);
        let absent = matches!(
            std::fs::symlink_metadata(p),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound
        ) && matches!(
            std::fs::canonicalize(p),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound
        );
        Ok(absent)
    }

    /// Checks every directory entry that `folo-data`'s YOLO/COCO format discovery (or `folo-server`'s
    /// `dataset_below` hint) could probe before opening `root`, so guard-skipped symlinks (such as custom
    /// `*.yaml`/`*.yml`/`*.json`, symlinked `annotations`/`train`/`val`/`valid`/`test` folders, or symlinked
    /// subdirs) are rejected before unguarded `folo serve` ever inspects them.
    fn check_discovery_candidates(&self, root: &Path, id: &str) -> Result<(), Error> {
        if !root.is_dir() {
            return Ok(());
        }
        let wrap = |e: Error| if id.is_empty() { e } else { in_dataset(e, id) };
        let read_err = |dir: &Path, e: std::io::Error| {
            wrap(
                Error::new(
                    StatusCode::FORBIDDEN,
                    "path_outside_data_dir",
                    format!("cannot inspect dataset folder {}: {e}", dir.display()),
                )
                .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine"),
            )
        };
        let entries = std::fs::read_dir(root).map_err(|e| read_err(root, e))?;
        for entry in entries {
            let entry = entry.map_err(|e| read_err(root, e))?;
            let p = entry.path();
            self.policy.check_path_if_present(&p).map_err(&wrap)?;
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if matches!(
                name.as_str(),
                "annotations" | "jpegimages" | "imagesets" | "train" | "val" | "valid" | "test"
            ) && p.is_dir()
            {
                let sub_entries = std::fs::read_dir(&p).map_err(|e| read_err(&p, e))?;
                for sub in sub_entries {
                    let sub = sub.map_err(|e| read_err(&p, e))?;
                    self.policy.check_path_if_present(&sub.path()).map_err(&wrap)?;
                }
            }
        }
        Ok(())
    }

    /// Computes an FNV-1a hash over the canonical targets of every watched root, directory, and symlink under
    /// `watched`. Returns `None` if any target cannot be canonicalized or escapes `policy.data_dir` (failing closed
    /// so replacing a split folder with a symlink outside `--data-dir` invalidates cached allows immediately).
    fn canonical_signature(&self, watched: &[PathBuf]) -> Option<u64> {
        let canon_root = self.policy.canonical_root().ok()?;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut visited = std::collections::HashSet::new();
        fn feed(h: &mut u64, data: &[u8]) {
            for &b in data {
                *h ^= b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        fn walk_dir(
            dir: &Path,
            canon_root: &Path,
            depth: usize,
            visited: &mut std::collections::HashSet<PathBuf>,
            h: &mut u64,
        ) -> Option<()> {
            if depth > 64 {
                return None;
            }
            let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
                .ok()?
                .collect::<std::io::Result<Vec<_>>>()
                .ok()?
                .into_iter()
                .map(|e| e.path())
                .collect();
            entries.sort();
            for p in entries {
                let md = std::fs::symlink_metadata(&p).ok()?;
                if md.file_type().is_symlink() || md.is_dir() {
                    let c = std::fs::canonicalize(&p).ok()?;
                    if !c.starts_with(canon_root) {
                        return None;
                    }
                    feed(h, p.to_string_lossy().as_bytes());
                    feed(h, b"->");
                    feed(h, c.to_string_lossy().as_bytes());
                    if c.is_dir() && visited.insert(c.clone()) {
                        walk_dir(&p, canon_root, depth + 1, visited, h)?;
                    }
                }
            }
            Some(())
        }
        for w in watched {
            match std::fs::symlink_metadata(w) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    feed(&mut h, w.to_string_lossy().as_bytes());
                    feed(&mut h, &[0]);
                }
                Err(_) => return None,
                Ok(_) => {
                    let c = std::fs::canonicalize(w).ok()?;
                    if !c.starts_with(&canon_root) {
                        return None;
                    }
                    feed(&mut h, w.to_string_lossy().as_bytes());
                    feed(&mut h, b"->");
                    feed(&mut h, c.to_string_lossy().as_bytes());
                    if c.is_dir() && visited.insert(c.clone()) {
                        walk_dir(w, &canon_root, 0, &mut visited, &mut h)?;
                    }
                }
            }
        }
        Some(h)
    }

    /// Registry-only dataset lookup via `Store::list_datasets` so local proxy confinement never triggers
    /// `RealStore::fresh_index` on `folo serve` before validating the on-disk manifest locally.
    fn lookup_registry(&self, store: &dyn Store, id: &str) -> Result<folo_api::Dataset, Error> {
        let datasets = store.list_datasets().map_err(map_denied)?;
        datasets.into_iter().find(|d| d.id == id).ok_or_else(|| {
            self.cache.lock().unwrap().remove(id);
            Error::not_found("dataset", id)
        })
    }

    /// Validates a local dataset path BEFORE `POST /api/datasets` is sent to a local `folo serve` in confined
    /// proxy mode, so an unguarded server never opens files outside `--data-dir` while building an index.
    /// No-op outside confined local real proxy mode. Blocking.
    pub fn pre_register_local(&self, path: &Path, format: folo_api::DatasetFormat) -> Result<(), Error> {
        if !self.policy.confine {
            return Ok(());
        }
        let Some(proxy) = &self.proxy else {
            return Ok(());
        };
        if proxy.is_mock() || self.is_verified_remote_proxy()? {
            return Ok(());
        }
        if matches!(
            std::fs::symlink_metadata(path),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound
        ) {
            return Ok(());
        }
        self.policy.check_existing_inside(path)?;
        self.check_discovery_candidates(path, "")?;
        let fmt = match format {
            folo_api::DatasetFormat::Auto => folo_data::Format::Auto,
            folo_api::DatasetFormat::Yolo => folo_data::Format::Yolo,
            folo_api::DatasetFormat::Coco => folo_data::Format::Coco,
        };
        let opts = folo_data::OpenOptions { format: fmt, verify_images: false };
        let ds = match folo_data::open_with(path, &opts) {
            Ok(ds) => ds,
            Err(e) if e.to_string().contains(folo_data::guard::DENIED) => {
                return Err(Error::new(
                    StatusCode::FORBIDDEN,
                    "path_outside_data_dir",
                    format!("cannot verify dataset manifest locally: {e}"),
                )
                .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine"));
            }
            Err(e @ (folo_data::DataError::Format(_) | folo_data::DataError::Unfinished(_))) => {
                return Err(folo_server::data_error(e, &path.to_string_lossy()));
            }
            Err(e) => {
                return Err(Error::new(
                    StatusCode::FORBIDDEN,
                    "path_outside_data_dir",
                    format!("cannot verify dataset manifest locally: {e}"),
                )
                .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine"));
            }
        };
        let (_, watched) = self.verify_loaded_dataset(path, format, &ds, "")?;
        if self.canonical_signature(&watched).is_none() {
            return Err(Error::new(
                StatusCode::FORBIDDEN,
                "path_outside_data_dir",
                format!("dataset {} contains a symlink outside the data dir", path.display()),
            )
            .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine"));
        }
        Ok(())
    }

    /// `Ok` when dataset `id` reads nothing outside the data dir (always without `--confine`, and for datasets
    /// hosted on a verified separate remote server). In local proxy mode, registry-only lookup obtains the
    /// dataset's path without triggering `fresh_index` on `folo serve`, and cached verdicts are verified against
    /// [`Self::canonical_signature`] first (before [`folo_server::Fingerprint`], so outside links are never
    /// walked) on every call. Blocking.
    pub fn dataset(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        if !self.policy.confine {
            return Ok(());
        }
        if self.is_verified_remote_proxy()? {
            return Ok(());
        }
        if self.proxy.is_some() {
            let d = self.lookup_registry(store, id)?;
            if d.path.starts_with("mock://") {
                return Ok(());
            }
            let cached = self.cache.lock().unwrap().get(id).cloned();
            if let Some((Some(tag), verdict)) = cached
                && tag.path == d.path
                && tag.format == d.format
                && !tag.watched.is_empty()
                && self.canonical_signature(&tag.watched) == Some(tag.canon_hash)
                && folo_server::Fingerprint::of(&tag.watched, &[]) == tag.fingerprint
            {
                return verdict;
            }
            return self.scan_and_cache(store, id, Some(d));
        }
        if let Some((_, r)) = self.cache.lock().unwrap().get(id) {
            return r.clone();
        }
        self.dataset_fresh(store, id)
    }

    /// Revalidates dataset `id` against `store` without trusting a cached verdict, refreshing the cache.
    /// Used before exposing datasets in fuzzy `"did you mean ...?"` suggestions so a dataset whose
    /// manifest changed mid-session to read outside `--data-dir` is never disclosed. Blocking.
    pub fn dataset_fresh(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        if !self.policy.confine {
            return Ok(());
        }
        if self.is_verified_remote_proxy()? {
            return Ok(());
        }
        if self.proxy.is_some() {
            let d = self.lookup_registry(store, id)?;
            return self.scan_and_cache(store, id, Some(d));
        }
        self.scan_and_cache(store, id, None)
    }

    fn scan_and_cache(
        &self,
        store: &dyn Store,
        id: &str,
        prefetched: Option<folo_api::Dataset>,
    ) -> Result<(), Error> {
        let d = match prefetched {
            Some(d) => d,
            None => match store.get_dataset(id).map_err(map_denied) {
                Ok(d) => d,
                Err(e) => {
                    if e.body.code == "dataset_not_found" {
                        self.cache.lock().unwrap().remove(id);
                    }
                    return Err(e);
                }
            },
        };
        let res = self.scan(store, id, &d).map_err(map_denied);
        // Cache definitive confinement verdicts; in local proxy mode never cache denials so `Fingerprint::of`
        // never walks a rejected dataset path and repaired datasets recover immediately without restart.
        match &res {
            Ok(watched) => {
                let tag = watched.as_ref().map(|(w, canon_hash)| ProxyTag {
                    path: d.path.clone(),
                    format: d.format,
                    watched: w.clone(),
                    fingerprint: folo_server::Fingerprint::of(w, &[]),
                    canon_hash: *canon_hash,
                });
                self.cache.lock().unwrap().insert(id.to_string(), (tag, Ok(())));
                Ok(())
            }
            Err(e) if e.body.code == "path_outside_data_dir" => {
                if self.proxy.is_some() {
                    self.cache.lock().unwrap().remove(id);
                } else {
                    self.cache.lock().unwrap().insert(id.to_string(), (None, Err(e.clone())));
                }
                Err(e.clone())
            }
            Err(e) if e.body.code == "dataset_not_found" => {
                self.cache.lock().unwrap().remove(id);
                Err(e.clone())
            }
            Err(e) => Err(e.clone()),
        }
    }

    /// `Ok` when every image of labeling job `id` is inside the data dir (always without `--confine`; through a
    /// server, which reads them itself, too). Cached until the job's image files change (not only their number: one
    /// image removed and another added keeps the count). Blocking.
    pub fn label_job(&self, labels: &crate::label::Labels, id: &str) -> Result<(), Error> {
        if !self.policy.confine || labels.is_remote() {
            return Ok(());
        }
        let files = labels.files(id)?.unwrap_or_default();
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            files.hash(&mut h);
            h.finish()
        };
        if let Some((k, r)) = self.jobs.lock().unwrap().get(id)
            && *k == key
        {
            return r.clone();
        }
        let r = self.policy.check_files(files.iter().map(PathBuf::as_path)).map_err(|mut e| {
            if e.body.code == "path_outside_data_dir" {
                e.body.message =
                    format!("labeling job '{id}' has images outside the data dir: {}", e.body.message);
            }
            e
        });
        if r.as_ref().err().is_none_or(|e| e.body.code == "path_outside_data_dir") {
            self.jobs.lock().unwrap().insert(id.to_string(), (key, r.clone()));
        }
        r
    }

    /// Drops the cached result of `id` (after it was unregistered).
    pub fn forget(&self, id: &str) {
        self.cache.lock().unwrap().remove(id);
    }

    fn verify_loaded_dataset(
        &self,
        root: &Path,
        format: folo_api::DatasetFormat,
        ds: &folo_data::LoadedDataset,
        id: &str,
    ) -> Result<(PathBuf, Vec<PathBuf>), Error> {
        let wrap = |e: Error| if id.is_empty() { e } else { in_dataset(e, id) };
        if let Some(denied) = ds.report.issues.iter().find(|i| i.message.contains(folo_data::guard::DENIED)) {
            return Err(wrap(Error::new(
                StatusCode::FORBIDDEN,
                "path_outside_data_dir",
                denied.message.clone(),
            )));
        }
        let ds_root = resolve_dotdot(&ds.root);
        self.policy.check_existing_inside(&ds_root).map_err(&wrap)?;
        let root_dir = if root.is_file() { root.parent().unwrap_or(root) } else { root };
        let mut watched = vec![root_dir.to_path_buf(), ds_root.clone()];
        for s in &ds.sources {
            let resolved = resolve_dotdot(s);
            self.policy.check_path_if_present(&resolved).map_err(&wrap)?;
            watched.push(resolved);
        }
        let is_yolo = format == folo_api::DatasetFormat::Yolo
            || (format == folo_api::DatasetFormat::Auto && ds.format == folo_data::Format::Yolo);
        let yolo_labels = is_yolo.then(|| LabelPaths::new(&ds_root));
        for split in folo_data::Split::ALL {
            for entry in ds.split(split).entries() {
                self.policy.check_path_if_present(&entry.image).map_err(&wrap)?;
                if let Some(lbl) = &entry.label {
                    self.policy.check_path_if_present(lbl).map_err(&wrap)?;
                }
                if let Some(labels) = &yolo_labels {
                    self.policy.check_path_if_present(&labels.of(&entry.image)).map_err(&wrap)?;
                    self.policy
                        .check_path_if_present(&folo_data::label_path_for(&entry.image))
                        .map_err(&wrap)?;
                }
            }
        }
        for issue in &ds.report.issues {
            self.policy.check_path_if_present(&issue.path).map_err(&wrap)?;
        }
        if is_yolo {
            for rel in ["classes.txt", "labels/classes.txt"] {
                self.policy.check_path_if_present(&ds_root.join(rel)).map_err(&wrap)?;
                if root_dir != ds_root {
                    self.policy.check_path_if_present(&root_dir.join(rel)).map_err(&wrap)?;
                }
            }
        }
        let watched = minimize_watched(watched);
        Ok((ds_root, watched))
    }

    fn scan(
        &self,
        store: &dyn Store,
        id: &str,
        d: &folo_api::Dataset,
    ) -> Result<Option<(Vec<PathBuf>, u64)>, Error> {
        if d.path.starts_with("mock://") {
            return Ok(None);
        }
        let root = PathBuf::from(&d.path);
        // Fail closed if `root` cannot be canonicalized locally (permission denied, container/other mount) or
        // resolves outside `--data-dir`.
        self.policy.check_existing_inside(&root).map_err(|e| in_dataset(e, id))?;
        let (base, watched) = match store.dataset_root(id)? {
            Some(r) => {
                self.policy.check_existing_inside(&r).map_err(|e| in_dataset(e, id))?;
                (r, None)
            }
            None => {
                // In local proxy mode `HttpStore` does not expose `Index::root` or run `folo_data::guard` inside
                // `folo serve`. Inspect discovery candidates and the manifest locally (with our process read
                // guard installed) BEFORE calling any endpoint that triggers `fresh_index`, and fail closed if it
                // cannot be inspected or references anything outside `--data-dir`.
                self.check_discovery_candidates(&root, id)?;
                let format = match d.format {
                    folo_api::DatasetFormat::Auto => folo_data::Format::Auto,
                    folo_api::DatasetFormat::Yolo => folo_data::Format::Yolo,
                    folo_api::DatasetFormat::Coco => folo_data::Format::Coco,
                };
                let opts = folo_data::OpenOptions { format, verify_images: false };
                let ds = folo_data::open_with(&root, &opts).map_err(|e| {
                    in_dataset(
                        Error::new(
                            StatusCode::FORBIDDEN,
                            "path_outside_data_dir",
                            format!("cannot verify dataset manifest locally: {e}"),
                        )
                        .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine"),
                        id,
                    )
                })?;
                let (ds_root, watched) = self.verify_loaded_dataset(&root, d.format, &ds, id)?;
                let Some(canon_hash) = self.canonical_signature(&watched) else {
                    return Err(in_dataset(
                        Error::new(
                            StatusCode::FORBIDDEN,
                            "path_outside_data_dir",
                            format!("dataset {} contains a symlink outside the data dir", root.display()),
                        )
                        .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine"),
                        id,
                    ));
                };
                (ds_root, Some((watched, canon_hash)))
            }
        };
        // Check YOLO class-name sidecar files (`classes.txt`, `labels/classes.txt`), which are omitted from
        // `ds.sources` and echoed verbatim into `dataset.classes`.
        if d.format == folo_api::DatasetFormat::Yolo {
            for rel in ["classes.txt", "labels/classes.txt"] {
                self.policy.check_path_if_present(&base.join(rel)).map_err(|e| in_dataset(e, id))?;
                let root_dir = if root.is_file() { root.parent().unwrap_or(&root) } else { root.as_path() };
                if root_dir != base {
                    self.policy.check_path_if_present(&root_dir.join(rel)).map_err(|e| in_dataset(e, id))?;
                }
            }
        }
        let yolo_labels = (d.format == folo_api::DatasetFormat::Yolo).then(|| LabelPaths::new(&base));
        let mut offset = 0;
        loop {
            let q = ImageQuery { split: None, offset, limit: WALK_PAGE, class: None };
            let page = store.list_images(id, &q)?;
            let n = page.images.len() as u64;
            for im in &page.images {
                let f = Path::new(&im.file);
                let img = if f.is_absolute() { f.to_path_buf() } else { base.join(f) };
                if let Some(labels) = &yolo_labels {
                    // Check both the normalized mirror path and the raw `folo_data::label_path_for` path so a
                    // `..` spelling difference can never skip a label file that `folo-data` opens.
                    self.policy.check_path_if_present(&labels.of(&img)).map_err(|e| in_dataset(e, id))?;
                    self.policy
                        .check_path_if_present(&folo_data::label_path_for(&img))
                        .map_err(|e| in_dataset(e, id))?;
                }
                self.policy.check_path_if_present(&img).map_err(|e| in_dataset(e, id))?;
            }
            offset += n;
            if n == 0 || offset >= page.total {
                break;
            }
        }
        // A read the guard refused while indexing is reported as an issue (an unreadable list, image or label),
        // and any skipped/broken file reported in issues (e.g. a symlinked label whose first token is echoed) must
        // also lie inside `--data-dir`.
        let mut offset = 0;
        loop {
            let page =
                store.list_issues(id, &IssueQuery { offset, limit: WALK_PAGE, ..IssueQuery::default() })?;
            for i in &page.issues {
                if i.message.contains(folo_data::guard::DENIED) {
                    return Err(in_dataset(
                        Error::new(StatusCode::FORBIDDEN, "path_outside_data_dir", i.message.clone()),
                        id,
                    ));
                }
                if !i.file.is_empty() {
                    let f = Path::new(&i.file);
                    let path = if f.is_absolute() { f.to_path_buf() } else { base.join(f) };
                    self.policy.check_path_if_present(&path).map_err(|e| in_dataset(e, id))?;
                }
            }
            let n = page.issues.len() as u64;
            offset += n;
            if n == 0 || offset >= page.total {
                return Ok(watched);
            }
        }
    }
}

/// Pops one `ParentDir` (`..`) component onto `out` safely: preserves root (`/`, `C:\`), preserves leading `..`
/// segments on relative paths (`../../a`), and resolves symlinks before popping on Unix.
fn push_parent(out: &mut PathBuf) {
    #[cfg(unix)]
    {
        if !out.as_os_str().is_empty()
            && !matches!(out.components().next_back(), Some(Component::ParentDir))
            && let Ok(canon) = std::fs::canonicalize(&*out)
        {
            *out = canon;
        }
    }
    match out.components().next_back() {
        Some(Component::Normal(_)) => {
            let _ = out.pop();
        }
        Some(Component::RootDir | Component::Prefix(_)) => {}
        None | Some(Component::ParentDir | Component::CurDir) => out.push(".."),
    }
}

/// Normalizes `p` to an absolute path while following symlinks before `..` on Unix and collapsing `..` lexically
/// on Windows (matching OS path resolution and `folo-server`'s `Index::build`).
fn resolve_dotdot(p: &Path) -> PathBuf {
    let spelled = Path::new(".").join(p);
    let abs = std::path::absolute(&spelled).unwrap_or(spelled);
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => push_parent(&mut out),
            // Windows `\\?\` verbatim paths yield `Normal(".")` / `Normal("..")` in `Path::components()`.
            Component::Normal(s) if s == "." => {}
            Component::Normal(s) if s == ".." => push_parent(&mut out),
            other => out.push(other),
        }
    }
    out
}

/// The YOLO label files of one dataset: [`folo_data::label_path_for`], except that an `images` folder
/// at or above the dataset root counts only when the root's mirror exists (`x/labels/birds` for `x/images/birds`,
/// `x/labels` for `x/images`). Without that mirror, labels of images below the root sit beside them (`a.txt`),
/// matching `folo-data`'s YOLO reader.
struct LabelPaths {
    /// The dataset root, absolute and normalised (with symlink-aware `..` resolution on Unix).
    root: PathBuf,
    /// The root has an `images` ancestor whose `labels` mirror does not exist.
    beside: bool,
}

impl LabelPaths {
    fn new(root: &Path) -> Self {
        let resolved = resolve_dotdot(root);
        let comps: Vec<Component> = resolved.components().collect();
        let mirror = comps.iter().rposition(|c| c.as_os_str() == std::ffi::OsStr::new("images")).map(|i| {
            let mut mirror: PathBuf = comps[..i].iter().collect();
            mirror.push("labels");
            mirror.extend(&comps[i + 1..]);
            mirror
        });
        let beside = mirror.as_ref().is_some_and(|m| !m.is_dir());
        LabelPaths { root: resolved, beside }
    }

    /// The label file of `image`. An image outside the root (or under its own `images/` subfolder) follows
    /// [`folo_data::label_path_for`].
    fn of(&self, image: &Path) -> PathBuf {
        let resolved = resolve_dotdot(image);
        let beside = self.beside
            && resolved.strip_prefix(&self.root).is_ok_and(|below| {
                !below.components().any(|c| c.as_os_str() == std::ffi::OsStr::new("images"))
            });
        if beside { resolved.with_extension("txt") } else { folo_data::label_path_for(&resolved) }
    }
}

/// Minimizes a watched-path list so no watched directory is walked twice when it is already nested under another.
fn minimize_watched(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for d in dirs {
        if !out.iter().any(|w| d.starts_with(w)) {
            out.retain(|w| !w.starts_with(&d));
            out.push(d);
        }
    }
    out.sort();
    out
}

fn in_dataset(mut e: Error, id: &str) -> Error {
    if e.body.code == "path_outside_data_dir" {
        e.body.message = format!("dataset '{id}' reads files outside the data dir: {}", e.body.message);
    }
    e
}

#[cfg(test)]
#[path = "confine_tests.rs"]
mod tests;
