//! Client-side state: what has been fetched from the backend, live run data, and decoded images.

mod best;
mod image_queue;
mod images;
mod live;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::api::*;

pub use best::better_model;
pub use image_queue::LoadQueue;
pub use images::{
    ImageCache, ImgKey, ImgKind, ImgSlot, decode_upright, to_render_image, to_render_image_within,
};

/// A queued image fetch: the backend call that answers it (run off the UI thread).
pub type ImageJob = Box<dyn FnOnce(&dyn Backend) -> ApiResult<ImageData> + Send>;
pub use live::{Derived, LogHistory, RunData};

/// A value fetched from the backend.
#[derive(Clone, Debug, Default)]
pub enum Load<T> {
    #[default]
    Idle,
    Loading,
    Ready(T),
    Failed(ApiError),
}

impl<T> Load<T> {
    pub fn ready(&self) -> Option<&T> {
        match self {
            Load::Ready(v) => Some(v),
            _ => None,
        }
    }

    pub fn ready_mut(&mut self) -> Option<&mut T> {
        match self {
            Load::Ready(v) => Some(v),
            _ => None,
        }
    }

    pub fn is_loading(&self) -> bool {
        matches!(self, Load::Idle | Load::Loading)
    }

    pub fn error(&self) -> Option<&ApiError> {
        match self {
            Load::Failed(e) => Some(e),
            _ => None,
        }
    }

    pub fn needs_fetch(&self) -> bool {
        matches!(self, Load::Idle)
    }
}

impl<T> From<ApiResult<T>> for Load<T> {
    fn from(r: ApiResult<T>) -> Self {
        match r {
            Ok(v) => Load::Ready(v),
            Err(e) => Load::Failed(e),
        }
    }
}

pub struct Store {
    pub api: Arc<dyn Backend>,
    pub health: Load<Health>,
    pub system: Load<SystemInfo>,
    /// What runs can use on the server (from the same `/system` answer; absent from older servers).
    pub capabilities: Option<crate::api::Capabilities>,
    /// Optional API features the server reports (`/system` `features`, e.g. `queue_control`).
    pub features: Vec<String>,
    /// The server's data dir (`/system` `data_dir`; absent in mock mode and from older servers).
    pub data_dir: Option<String>,
    pub datasets: Load<Vec<Dataset>>,
    pub dataset: HashMap<String, Load<Dataset>>,
    /// All image metadata of a dataset (cheap: boxes only), for filtering and the grid.
    pub images: HashMap<String, Load<Vec<ImageInfo>>>,
    /// A dataset card's first images, for a dataset whose full list isn't read
    /// (`FoloApp::ensure_dataset_card`).
    pub previews: HashMap<String, Load<Vec<ImageInfo>>>,
    pub models: Load<Vec<Model>>,
    pub evals: HashMap<String, Load<ModelEval>>,
    /// Per dataset, its issue counts by severity (fetched when it has broken labels).
    pub issues: HashMap<String, Load<IssueCounts>>,
    pub runs: Load<Vec<Run>>,
    pub run_data: HashMap<String, Load<RunData>>,
    pub samples: HashMap<String, Load<RunSamples>>,
    /// Runs whose latest validation samples have a newer epoch available; refreshed in place so
    /// existing tiles stay on screen instead of flashing placeholders and shifting the Logs panel.
    pub samples_stale: std::collections::HashSet<String>,
    /// Runs with an in-place Latest sample refresh currently in flight while older tiles stay visible.
    pub samples_refreshing: std::collections::HashSet<String>,
    /// Last error from an in-place Latest sample refresh while older tiles stay visible.
    pub samples_refresh_error: HashMap<String, ApiError>,
    pub logs: HashMap<String, Load<LogHistory>>,
    /// Bumped on every log change; the length stops changing once the 2,000-line cap is hit.
    pub log_rev: u64,
    /// Live events that arrived while a run's history was loading, merged in when it lands.
    pending: HashMap<String, Vec<RunEvent>>,
    /// Runs resumed and still active: every history load is cut back to the last finished epoch.
    resumed: std::collections::HashSet<String>,
    /// Each run's length as its live events last reported it (steps per epoch, epochs; 0 where
    /// unknown). Kept apart from the replaceable run list, so an older snapshot that still says
    /// 0 can't undo it.
    lengths: HashMap<String, (u64, u32)>,
    pub images_px: ImageCache,
    /// Image fetches waiting to start, a few at a time, newest first.
    pub image_queue: LoadQueue<ImageJob>,
    /// Throughput history for the system card (GPU util samples), newest last.
    pub gpu_util: VecDeque<f32>,
    /// How many GPUs the server's last `/system` answer listed: what holds while it is asked again.
    pub last_gpu_count: Option<usize>,
    /// Canonical architecture overrides (`folo2-*` or resolved `folo-*`) keyed by run id and model id.
    pub run_archs: HashMap<String, crate::arch::ModelArch>,
    pub model_archs: HashMap<String, crate::arch::ModelArch>,
    /// Model ids already probed via `GET /api/models/{id}` for `x-folo-arch` on this backend.
    pub probed_model_archs: std::collections::HashSet<String>,
}

impl Store {
    pub fn new(api: Arc<dyn Backend>) -> Self {
        let mut store = Self {
            api,
            health: Load::Idle,
            system: Load::Idle,
            capabilities: None,
            data_dir: None,
            features: Vec::new(),
            datasets: Load::Idle,
            dataset: HashMap::new(),
            images: HashMap::new(),
            previews: HashMap::new(),
            models: Load::Idle,
            evals: HashMap::new(),
            issues: HashMap::new(),
            runs: Load::Idle,
            run_data: HashMap::new(),
            samples: HashMap::new(),
            samples_stale: Default::default(),
            samples_refreshing: Default::default(),
            samples_refresh_error: HashMap::new(),
            logs: HashMap::new(),
            log_rev: 0,
            pending: HashMap::new(),
            resumed: Default::default(),
            lengths: HashMap::new(),
            images_px: ImageCache::default(),
            image_queue: LoadQueue::default(),
            gpu_util: VecDeque::new(),
            last_gpu_count: None,
            run_archs: HashMap::new(),
            model_archs: HashMap::new(),
            probed_model_archs: Default::default(),
        };
        store.sync_known_archs();
        store
    }

    /// Whether the connected server advertises `folo2_arch` (`folo2-n..x`) support.
    pub fn supports_folo2(&self) -> bool {
        self.features.iter().any(|f| f == crate::arch::FEATURE_FOLO2_ARCH)
    }

    /// Pull canonical `folo2-*` run/model architectures known to the backend into the store cache.
    pub fn sync_known_archs(&mut self) {
        let (runs, models) = self.api.known_archs();
        for (id, raw) in runs {
            if let Some(arch) = crate::arch::ModelArch::parse(&raw) {
                self.run_archs.insert(id, arch);
            }
        }
        for (id, raw) in models {
            if let Some(arch) = crate::arch::ModelArch::parse(&raw) {
                self.model_archs.insert(id, arch);
            }
        }
    }

    /// Record a run's canonical architecture (and its linked model's, if known).
    pub fn note_run_arch(&mut self, run_id: &str, arch: crate::arch::ModelArch) {
        self.run_archs.insert(run_id.to_string(), arch);
        if let Some(model_id) = self.run(run_id).and_then(|r| r.model_id.clone()) {
            self.model_archs.insert(model_id, arch);
        }
    }

    /// Record a model's canonical architecture (and its source run's, if known).
    pub fn note_model_arch(&mut self, model_id: &str, arch: crate::arch::ModelArch) {
        self.model_archs.insert(model_id.to_string(), arch);
        if let Some(run_id) = self.model(model_id).and_then(|m| m.run_id.clone()) {
            self.run_archs.insert(run_id, arch);
        }
    }

    /// Canonical architecture for a run (`folo-*` or `folo2-*`).
    pub fn run_arch(&self, r: &Run) -> crate::arch::ModelArch {
        if let Some(&arch) = self.run_archs.get(&r.id) {
            return arch;
        }
        if let Some(mid) = r.model_id.as_deref()
            && let Some(&arch) = self.model_archs.get(mid)
        {
            return arch;
        }
        crate::arch::ModelArch::v1(r.config.arch)
    }

    /// Canonical architecture for a model (`folo-*` or `folo2-*`).
    pub fn model_arch(&self, m: &Model) -> crate::arch::ModelArch {
        if let Some(&arch) = self.model_archs.get(&m.id) {
            return arch;
        }
        if let Some(rid) = m.run_id.as_deref()
            && let Some(&arch) = self.run_archs.get(rid)
        {
            return arch;
        }
        crate::arch::ModelArch::v1(m.arch)
    }

    pub fn run(&self, id: &str) -> Option<&Run> {
        self.runs.ready()?.iter().find(|r| r.id == id)
    }

    pub fn live_run(&self) -> Option<&Run> {
        self.runs.ready()?.iter().find(|r| r.status == RunStatus::Running)
    }

    pub fn model(&self, id: &str) -> Option<&Model> {
        self.models.ready()?.iter().find(|m| m.id == id)
    }

    pub fn dataset_summary(&self, id: &str) -> Option<&Dataset> {
        self.datasets.ready()?.iter().find(|d| d.id == id)
    }

    /// How a run names its dataset: the registered name; once the datasets are in without it (it was
    /// removed), the name recorded with the run (`Run.dataset`) marked "(removed)", as Studio does. The
    /// raw id only for a run from a server that recorded no name.
    pub fn run_dataset_name(&self, r: &Run) -> String {
        if let Some(d) = self.dataset_summary(&r.config.dataset_id) {
            return d.name.clone();
        }
        match (&r.dataset, self.datasets.ready().is_some()) {
            (Some(d), true) => format!("{} (removed)", d.name),
            (Some(d), false) => d.name.clone(),
            (None, true) => "removed dataset".into(),
            (None, false) => r.config.dataset_id.clone(),
        }
    }

    /// How a dataset is named by its id where no run is at hand (a model's row, a crumb): the
    /// registered name, else as a run trained on it names it ([`Self::run_dataset_name`]: the
    /// recorded name marked "(removed)"); the raw id only when no run knows it either.
    pub fn dataset_name(&self, id: &str) -> String {
        if let Some(d) = self.dataset_summary(id) {
            return d.name.clone();
        }
        let run = self.runs.ready().and_then(|runs| {
            // A run that recorded the name, if any did.
            let on = || runs.iter().filter(|r| r.config.dataset_id == id);
            on().find(|r| r.dataset.is_some()).or_else(|| on().next())
        });
        run.map_or(id.to_string(), |r| self.run_dataset_name(r))
    }

    /// Whether the server wrote this dataset into its own data dir (a generated set, a labeling job's
    /// export, an upload), so its folder can go with it (`?files=true`), as Studio decides.
    pub fn written_by_server(&self, d: &Dataset) -> bool {
        written_by_server(&d.path, self.data_dir.as_deref())
    }

    /// The server stopped answering (the last `/system` refresh couldn't connect).
    pub fn unreachable(&self) -> bool {
        matches!(&self.health, Load::Failed(e) if e.code == "server_unreachable")
    }

    /// The address is not one at all ([`ApiError::is_bad_address`]): nothing was sent.
    pub fn bad_address(&self) -> bool {
        matches!(&self.health, Load::Failed(e) if e.is_bad_address())
    }

    /// What the periodic `/system` refresh says about the connection (`err`: its error, if any).
    /// `health`, which the status line shows, is fetched once per connection; without this a server
    /// that went away mid-session still read "Connected". Unreachable marks it down; the first answer
    /// after that has health checked again.
    pub fn note_reachability(&mut self, err: Option<&ApiError>) {
        match err {
            // `--token` leaves `/api/health` open: the status says what every other call answers.
            Some(e) if e.code == "server_unreachable" || e.code == "unauthorized" => {
                self.health = Load::Failed(e.clone())
            }
            None if matches!(self.health, Load::Failed(_)) => self.health = Load::Idle,
            _ => {}
        }
    }
}

/// True when `path` is inside `<data_dir>/datasets/`, where the server writes the datasets it makes:
/// `/` and `\` alike, a trailing separator ignored, and a Windows path compared ignoring case.
pub fn written_by_server(path: &str, data_dir: Option<&str>) -> bool {
    let norm = |p: &str| {
        let s = p.replace('\\', "/").trim_end_matches('/').to_string();
        let drive = s.as_bytes().get(1) == Some(&b':') && s.as_bytes()[0].is_ascii_alphabetic();
        if drive { s.to_lowercase() } else { s }
    };
    data_dir.is_some_and(|dir| norm(path).starts_with(&format!("{}/datasets/", norm(dir))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::mock::MockBackend;

    #[test]
    fn a_server_that_goes_away_mid_session_shows_as_unreachable_until_it_is_back() {
        let api = MockBackend::new(4200, true);
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        store.health = api.health().into();
        let down = ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7911");
        // The 2 s refresh fails to connect: the status line must stop saying "Connected".
        store.note_reachability(Some(&down));
        assert!(store.unreachable());
        // It answers again: health is checked again (Idle is fetched by the next ensure_core).
        store.note_reachability(None);
        assert!(store.health.needs_fetch() && !store.unreachable());
        // A connected server's answers and its other errors (a 500) leave a good status alone.
        store.health = api.health().into();
        store.note_reachability(None);
        store.note_reachability(Some(&ApiError::new("http_error", "folo serve answered HTTP 500")));
        assert!(store.health.ready().is_some());
        // A `--token` server answers its health but nothing else: the status says so, not "Connected".
        store.note_reachability(Some(&crate::api::http::needs_token()));
        assert!(matches!(&store.health, Load::Failed(e) if e.code == "unauthorized") && !store.unreachable());
    }

    #[test]
    fn a_run_names_its_removed_dataset_by_the_name_it_recorded_never_by_id() {
        let api = Arc::new(MockBackend::new(4200, true));
        let mut store = Store::new(api.clone());
        let mut run = api.runs().unwrap().into_iter().next().unwrap();
        let datasets = api.datasets().unwrap();
        let name = datasets.iter().find(|d| d.id == run.config.dataset_id).unwrap().name.clone();
        run.dataset = Some(RunDataset { name: name.clone(), classes: Vec::new() });
        // Before the datasets load: the recorded name.
        assert_eq!(store.run_dataset_name(&run), name);
        store.datasets = Load::Ready(datasets.clone());
        assert_eq!(store.run_dataset_name(&run), name);
        // Removed: the recorded name, marked; an older server's run without one is not shown as an id.
        store.datasets =
            Load::Ready(datasets.into_iter().filter(|d| d.id != run.config.dataset_id).collect());
        assert_eq!(store.run_dataset_name(&run), format!("{name} (removed)"));
        run.dataset = None;
        assert_eq!(store.run_dataset_name(&run), "removed dataset");
    }

    #[test]
    fn a_dataset_under_the_data_dirs_datasets_folder_was_written_by_the_server() {
        let dir = Some("D:/folo-ux/data2-v-desktop-e2e");
        assert!(written_by_server("D:\\folo-ux\\data2-v-desktop-e2e\\datasets\\Été 🎉 shapes", dir));
        assert!(written_by_server("d:/FOLO-UX/data2-v-desktop-e2e/datasets/shapes/", dir));
        assert!(!written_by_server("D:/folo-data/finetune/BCCD", dir));
        assert!(!written_by_server("D:/folo-ux/data2-v-desktop-e2e/datasets", dir), "the folder itself");
        assert!(!written_by_server("D:/folo-ux/data2-v-desktop-e2e-old/datasets/x", dir));
        assert!(!written_by_server("D:/folo-ux/data2-v-desktop-e2e/datasets/x", None), "no data dir: never");
        // Not a Windows path: case counts.
        assert!(!written_by_server("/Srv/data/datasets/x", Some("/srv/data")));
    }

    #[test]
    fn a_removed_dataset_is_named_by_what_its_runs_recorded() {
        let api = Arc::new(MockBackend::new(4200, true));
        let mut store = Store::new(api.clone());
        let datasets = api.datasets().unwrap();
        let mut runs = api.runs().unwrap();
        let id = runs[0].config.dataset_id.clone();
        let name = datasets.iter().find(|d| d.id == id).unwrap().name.clone();
        for r in runs.iter_mut().filter(|r| r.config.dataset_id == id) {
            r.dataset = Some(RunDataset { name: name.clone(), classes: Vec::new() });
        }
        store.runs = Load::Ready(runs);
        store.datasets = Load::Ready(datasets.clone());
        assert_eq!(store.dataset_name(&id), name);
        store.datasets = Load::Ready(datasets.into_iter().filter(|d| d.id != id).collect());
        assert_eq!(store.dataset_name(&id), format!("{name} (removed)"), "never the raw id {id}");
        // Nothing knows it: only then the id.
        assert_eq!(store.dataset_name("ds_none"), "ds_none");
    }
}
