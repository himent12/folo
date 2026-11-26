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

/// Cached confinement verdict for one dataset id: optional `(path, revision)` tag (used in proxy mode so a
/// manifest or file change on the separate `folo serve` immediately invalidates a cached `Ok(())`) and result.
type DatasetVerdict = (Option<(String, String)>, Result<(), Error>);

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

    /// `Ok` when dataset `id` reads nothing outside the data dir (always without `--confine`, and for datasets
    /// hosted on a verified separate remote server). In local proxy mode, cached verdicts are keyed by the
    /// server's current `(path, revision)` so manifest changes on `folo serve` immediately trigger a fresh scan.
    /// Blocking.
    pub fn dataset(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        if !self.policy.confine {
            return Ok(());
        }
        if self.is_verified_remote_proxy()? {
            return Ok(());
        }
        if self.proxy.is_some() {
            let d = store.get_dataset(id)?;
            if let Some(rev) = &d.revision
                && let Some((Some((cached_path, cached_rev)), verdict)) = self.cache.lock().unwrap().get(id)
                && cached_path == &d.path
                && cached_rev == rev
            {
                return verdict.clone();
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
        let tag = d.revision.as_ref().map(|rev| (d.path.clone(), rev.clone()));
        let r = self.scan(store, id, &d).map_err(map_denied);
        // Only cache definitive confinement verdicts (allowed, or explicitly outside the data dir).
        // Unknown datasets and transient store/IO errors are never cached so recovery works without a restart.
        match &r {
            Ok(()) => {
                self.cache.lock().unwrap().insert(id.to_string(), (tag, Ok(())));
            }
            Err(e) if e.body.code == "path_outside_data_dir" => {
                self.cache.lock().unwrap().insert(id.to_string(), (tag, Err(e.clone())));
            }
            Err(e) if e.body.code == "dataset_not_found" => {
                self.cache.lock().unwrap().remove(id);
            }
            Err(_) => {}
        }
        r
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

    fn scan(&self, store: &dyn Store, id: &str, d: &folo_api::Dataset) -> Result<(), Error> {
        if d.path.starts_with("mock://") {
            return Ok(());
        }
        let root = PathBuf::from(&d.path);
        // Fail closed if `root` cannot be canonicalized locally (permission denied, container/other mount) or
        // resolves outside `--data-dir`.
        self.policy.check_existing_inside(&root).map_err(|e| in_dataset(e, id))?;
        let base = match store.dataset_root(id)? {
            Some(r) => {
                self.policy.check_existing_inside(&r).map_err(|e| in_dataset(e, id))?;
                r
            }
            None => {
                // In local proxy mode `HttpStore` does not expose `Index::root` or run `folo_data::guard` inside
                // `folo serve`. Inspect the manifest locally (with our process read guard installed) and fail
                // closed if it cannot be inspected or references anything outside `--data-dir`.
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
                if let Some(denied) =
                    ds.report.issues.iter().find(|i| i.message.contains(folo_data::guard::DENIED))
                {
                    return Err(in_dataset(
                        Error::new(StatusCode::FORBIDDEN, "path_outside_data_dir", denied.message.clone()),
                        id,
                    ));
                }
                let ds_root = resolve_dotdot(&ds.root);
                self.policy.check_existing_inside(&ds_root).map_err(|e| in_dataset(e, id))?;
                for s in &ds.sources {
                    let resolved = resolve_dotdot(s);
                    self.policy.check_path_if_present(&resolved).map_err(|e| in_dataset(e, id))?;
                }
                for split in folo_data::Split::ALL {
                    for entry in ds.split(split).entries() {
                        self.policy.check_path_if_present(&entry.image).map_err(|e| in_dataset(e, id))?;
                        if let Some(lbl) = &entry.label {
                            self.policy.check_path_if_present(lbl).map_err(|e| in_dataset(e, id))?;
                        }
                    }
                }
                for issue in &ds.report.issues {
                    self.policy.check_path_if_present(&issue.path).map_err(|e| in_dataset(e, id))?;
                }
                ds_root
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
                return Ok(());
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
    fn labels_follow_yolo_beside_images_and_mirror_rules() {
        let dir = folo_test_support::TestDir::new("mcp-confine-label-paths");
        // Dataset below an `images` ancestor with NO `labels` mirror: labels sit beside images.
        let ds = dir.join("images").join("birds");
        std::fs::create_dir_all(&ds).unwrap();
        let lp = LabelPaths::new(&ds);
        assert_eq!(lp.of(&ds.join("a.jpg")), ds.join("a.txt"));
        // An `images/` subfolder inside the dataset still maps to `labels/` inside the dataset.
        assert_eq!(
            lp.of(&ds.join("images").join("train").join("a.jpg")),
            ds.join("labels").join("train").join("a.txt")
        );

        // Once the mirror `<dir>/labels/birds` exists, labels resolve through the mirror instead of beside.
        let mirror = dir.join("labels").join("birds");
        std::fs::create_dir_all(&mirror).unwrap();
        let lp_mirror = LabelPaths::new(&ds);
        assert_eq!(lp_mirror.of(&ds.join("a.jpg")), mirror.join("a.txt"));

        // A root spelled with `..` through a nested `images` segment (`images/images/../birds`, or
        // `images\images\..\birds` on Windows) resolves both its mirror probe and label lookup consistently.
        let nested_images = dir.join("images").join("images");
        std::fs::create_dir_all(&nested_images).unwrap();
        let dotdot_root = nested_images.join("..").join("birds");
        let lp_dotdot = LabelPaths::new(&dotdot_root);
        assert_eq!(lp_dotdot.of(&dotdot_root.join("a.jpg")), mirror.join("a.txt"));
        std::fs::remove_dir(&mirror).unwrap();
        let lp_dotdot_beside = LabelPaths::new(&dotdot_root);
        assert_eq!(lp_dotdot_beside.of(&dotdot_root.join("a.jpg")), ds.join("a.txt"));

        // On Unix, a `data.yaml` folder reached through a symlink (`work/current -> releases/v7/meta`) with
        // `../images/birds` resolves its mirror in the link target's parent (`releases/v7/labels/birds`).
        #[cfg(unix)]
        {
            let release = dir.join("releases/v7");
            std::fs::create_dir_all(release.join("images/birds")).unwrap();
            std::fs::create_dir_all(release.join("meta")).unwrap();
            std::fs::create_dir_all(dir.join("work")).unwrap();
            std::os::unix::fs::symlink(release.join("meta"), dir.join("work/current")).unwrap();
            let linked_root = dir.join("work/current/../images/birds");
            let linked_img = linked_root.join("a.jpg");
            let canon_img = std::fs::canonicalize(release.join("images/birds")).unwrap().join("a.jpg");
            let canon_mirror =
                std::fs::canonicalize(&release).unwrap().join("labels").join("birds").join("a.txt");
            assert_eq!(LabelPaths::new(&linked_root).of(&linked_img), canon_img.with_extension("txt"));
            std::fs::create_dir_all(release.join("labels/birds")).unwrap();
            assert_eq!(LabelPaths::new(&linked_root).of(&linked_img), canon_mirror);
        }
    }

    #[test]
    fn local_url_detection_covers_loopback_unspecified_and_remote_hosts() {
        for local in [
            "http://127.0.0.1:7878",
            "HTTP://127.0.0.1:7878",
            "http://localhost:7878/api",
            "http://LOCALHOST.:7878",
            "http://foo.localhost:7878",
            "https://[::1]:7878",
            "http://[::1%25lo]:7878",
            "http://0.0.0.0:7878",
            "http://[::]:7878",
            "http://[::ffff:127.0.0.1]:7878",
            // Empty hosts and non-canonical IPv4 loopback/unspecified literals fail closed as local.
            "",
            "http://:7878",
            "http:///api",
            "http://127.1:7878",
            "http://0:7878",
            "http://2130706433:7878",
            "http://0x7f000001:7878",
            "http://0177.0.0.1:7878",
        ] {
            assert!(is_local_url(local), "expected local: {local}");
        }
        for remote in [
            "http://192.168.1.5:7878",
            "https://gpu.example.com:7878",
            "https://[2001:db8::1]:7878",
            "http://notlocalhost.example.com:7878",
            // Multi-byte UTF-8 hostname near the `.localhost` suffix length must never panic.
            "http://éééééa:7878",
        ] {
            assert!(!is_local_url(remote), "expected remote: {remote}");
        }
    }

    #[test]
    fn push_parent_preserves_root_and_leading_relative_parents() {
        let mut at_root = PathBuf::from(if cfg!(windows) { r"C:\" } else { "/" });
        push_parent(&mut at_root);
        assert_eq!(at_root, PathBuf::from(if cfg!(windows) { r"C:\" } else { "/" }));

        let mut rel = PathBuf::from("..");
        push_parent(&mut rel);
        assert_eq!(rel, PathBuf::from("../.."));
        rel.push("a");
        assert_eq!(rel, PathBuf::from("../../a"));
    }

    #[test]
    fn denied_reads_become_path_outside_data_dir() {
        let e = Error::internal(format!("reading x: x {}", folo_data::guard::DENIED));
        assert_eq!(map_denied(e).body.code, "path_outside_data_dir");
        assert_eq!(map_denied(Error::internal("other")).body.code, "internal");
    }

    #[test]
    fn dataset_does_not_cache_transient_store_errors() {
        use folo_api::{
            Dataset, ExportResult, ExportSpec, GpuInfo, ImagePage, IssuePage, Model, ModelEval,
            PredictResponse, RegisterDataset, Run, RunLogs, RunMetrics, RunSamples, SyntheticDataset,
            TrainConfig,
        };
        use folo_server::store::{ImageData, LogsQuery, MetricsQuery, PredictInput, Subscription};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FlakyStore {
            inner: folo_server::MockStore,
            calls: AtomicUsize,
        }
        impl Store for FlakyStore {
            fn is_mock(&self) -> bool {
                self.inner.is_mock()
            }
            fn gpus(&self) -> Vec<GpuInfo> {
                self.inner.gpus()
            }
            fn list_datasets(&self) -> Result<Vec<Dataset>, Error> {
                self.inner.list_datasets()
            }
            fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset, Error> {
                self.inner.register_dataset(req)
            }
            fn create_synthetic(&self, req: SyntheticDataset) -> Result<Dataset, Error> {
                self.inner.create_synthetic(req)
            }
            fn get_dataset(&self, id: &str) -> Result<Dataset, Error> {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(Error::new(
                        StatusCode::BAD_GATEWAY,
                        "server_unreachable",
                        "temporary outage",
                    ));
                }
                self.inner.get_dataset(id)
            }
            fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage, Error> {
                self.inner.list_images(id, q)
            }
            fn list_issues(&self, id: &str, q: &IssueQuery) -> Result<IssuePage, Error> {
                self.inner.list_issues(id, q)
            }
            fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData, Error> {
                self.inner.image(id, image_id, thumb)
            }
            fn delete_dataset(&self, id: &str) -> Result<(), Error> {
                self.inner.delete_dataset(id)
            }
            fn list_models(&self) -> Result<Vec<Model>, Error> {
                self.inner.list_models()
            }
            fn get_model(&self, id: &str) -> Result<Model, Error> {
                self.inner.get_model(id)
            }
            fn model_eval(&self, id: &str) -> Result<ModelEval, Error> {
                self.inner.model_eval(id)
            }
            fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult, Error> {
                self.inner.export_model(id, spec)
            }
            fn list_runs(&self) -> Result<Vec<Run>, Error> {
                self.inner.list_runs()
            }
            fn create_run(&self, config: TrainConfig) -> Result<Run, Error> {
                self.inner.create_run(config)
            }
            fn get_run(&self, id: &str) -> Result<Run, Error> {
                self.inner.get_run(id)
            }
            fn stop_run(&self, id: &str) -> Result<Run, Error> {
                self.inner.stop_run(id)
            }
            fn delete_run(&self, id: &str) -> Result<(), Error> {
                self.inner.delete_run(id)
            }
            fn run_metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics, Error> {
                self.inner.run_metrics(id, q)
            }
            fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples, Error> {
                self.inner.run_samples(id, epoch)
            }
            fn run_logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs, Error> {
                self.inner.run_logs(id, q)
            }
            fn subscribe(&self, id: &str) -> Result<Subscription, Error> {
                self.inner.subscribe(id)
            }
            fn predict(&self, input: PredictInput) -> Result<PredictResponse, Error> {
                self.inner.predict(input)
            }
        }

        let dir = folo_test_support::TestDir::new("mcp-confine-transient");
        let c = Confinement::new(PathPolicy { data_dir: dir.to_path_buf(), confine: true }, None);
        let store = FlakyStore { inner: folo_server::MockStore::new(), calls: AtomicUsize::new(0) };
        let id = store.list_datasets().unwrap().remove(0).id;

        let first = c.dataset(&store, &id).unwrap_err();
        assert_eq!(first.body.code, "server_unreachable");
        assert!(c.dataset(&store, &id).is_ok(), "transient error must not be permanently cached");
        assert_eq!(store.calls.load(Ordering::SeqCst), 2);
        // Now that Ok(()) was cached, a third call uses the cache without hitting the store again.
        assert!(c.dataset(&store, &id).is_ok());
        assert_eq!(store.calls.load(Ordering::SeqCst), 2);
    }
}
