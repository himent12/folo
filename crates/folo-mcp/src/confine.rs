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

/// True when `url` targets this machine (loopback, unspecified bind address, `localhost`, or `*.localhost`),
/// meaning a proxied `folo serve` may share the local filesystem with this MCP process.
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
    if trimmed.eq_ignore_ascii_case("localhost")
        || trimmed.len() > ".localhost".len()
            && trimmed[trimmed.len() - ".localhost".len()..].eq_ignore_ascii_case(".localhost")
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
        Err(_) => false,
    }
}

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
    /// Present in proxy mode (`--server`): a non-local remote server has its own separate filesystem, whereas a
    /// local/loopback server may share this machine's filesystem and stays strictly confined to `policy.data_dir`.
    proxy: Option<Arc<HttpStore>>,
    cache: Arc<Mutex<HashMap<String, Result<(), Error>>>>,
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

    /// True when proxying a non-local remote server whose filesystem is separate from this machine's `--data-dir`.
    /// Loopback / local URLs always return `false` so paths outside `--data-dir` on a shared filesystem are never
    /// exposed even when the local server uses a different `--data-dir`.
    pub fn is_remote_proxy(&self) -> bool {
        self.proxy.as_ref().is_some_and(|p| !is_local_url(p.base()))
    }

    /// `Ok` when dataset `id` reads nothing outside the data dir (always without `--confine`, and for datasets
    /// hosted on a separate non-local remote server). Blocking.
    pub fn dataset(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        if !self.policy.confine || self.is_remote_proxy() {
            return Ok(());
        }
        if let Some(r) = self.cache.lock().unwrap().get(id) {
            return r.clone();
        }
        self.dataset_fresh(store, id)
    }

    /// Revalidates dataset `id` against `store` without trusting a cached verdict, refreshing the cache.
    /// Used before exposing datasets in fuzzy `"did you mean ...?"` suggestions so a dataset whose
    /// manifest changed mid-session to read outside `--data-dir` is never disclosed. Blocking.
    pub fn dataset_fresh(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        if !self.policy.confine || self.is_remote_proxy() {
            return Ok(());
        }
        let r = self.scan(store, id).map_err(map_denied);
        // Only cache definitive confinement verdicts (allowed, or explicitly outside the data dir).
        // Unknown datasets and transient store/IO errors are never cached so recovery works without a restart.
        match &r {
            Ok(()) => {
                self.cache.lock().unwrap().insert(id.to_string(), Ok(()));
            }
            Err(e) if e.body.code == "path_outside_data_dir" => {
                self.cache.lock().unwrap().insert(id.to_string(), Err(e.clone()));
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

    fn scan(&self, store: &dyn Store, id: &str) -> Result<(), Error> {
        let d = store.get_dataset(id)?;
        if d.path.starts_with("mock://") {
            return Ok(());
        }
        let root = PathBuf::from(&d.path);
        self.policy.check_files([root.as_path()]).map_err(|e| in_dataset(e, id))?;
        let base = match store.dataset_root(id)? {
            Some(r) => {
                self.policy.check_files([r.as_path()]).map_err(|e| in_dataset(e, id))?;
                r
            }
            None => {
                // In local proxy mode `HttpStore` does not expose `Index::root` or install `folo_data::guard` in
                // `folo serve`. Inspect the manifest locally (without decoding images) so `data.yaml`'s `path:`,
                // external split lists, linked annotation folders, and effective `ds.root` are all verified.
                let format = match d.format {
                    folo_api::DatasetFormat::Auto => folo_data::Format::Auto,
                    folo_api::DatasetFormat::Yolo => folo_data::Format::Yolo,
                    folo_api::DatasetFormat::Coco => folo_data::Format::Coco,
                };
                let opts = folo_data::OpenOptions { format, verify_images: false };
                if let Ok(ds) = folo_data::open_with(&root, &opts) {
                    let ds_root = resolve_dotdot(&ds.root);
                    let sources: Vec<PathBuf> = ds.sources.iter().map(|s| resolve_dotdot(s)).collect();
                    self.policy
                        .check_files(
                            std::iter::once(ds_root.as_path()).chain(sources.iter().map(PathBuf::as_path)),
                        )
                        .map_err(|e| in_dataset(e, id))?;
                    ds_root
                } else if root.is_file() {
                    root.parent().unwrap_or(&root).to_path_buf()
                } else {
                    root
                }
            }
        };
        let yolo_labels = (d.format == folo_api::DatasetFormat::Yolo).then(|| LabelPaths::new(&base));
        let mut offset = 0;
        loop {
            let q = ImageQuery { split: None, offset, limit: WALK_PAGE, class: None };
            let page = store.list_images(id, &q)?;
            let n = page.images.len() as u64;
            let mut files = Vec::with_capacity(page.images.len() * if yolo_labels.is_some() { 2 } else { 1 });
            for im in &page.images {
                let f = Path::new(&im.file);
                let img = if f.is_absolute() { f.to_path_buf() } else { base.join(f) };
                if let Some(labels) = &yolo_labels {
                    files.push(labels.of(&img));
                }
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

/// Normalizes `p` to an absolute path while following symlinks before `..` on Unix and collapsing `..` lexically
/// on Windows (matching OS path resolution and `folo-server`'s `Index::build`).
fn resolve_dotdot(p: &Path) -> PathBuf {
    let spelled = Path::new(".").join(p);
    let abs = std::path::absolute(&spelled).unwrap_or(spelled);
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                #[cfg(unix)]
                {
                    if let Ok(canon) = std::fs::canonicalize(&out) {
                        out = canon;
                    }
                }
                if !out.pop() {
                    out.push("..");
                }
            }
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
        ] {
            assert!(is_local_url(local), "expected local: {local}");
        }
        for remote in [
            "http://192.168.1.5:7878",
            "https://gpu.example.com:7878",
            "https://[2001:db8::1]:7878",
            "http://notlocalhost.example.com:7878",
        ] {
            assert!(!is_local_url(remote), "expected remote: {remote}");
        }
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
