//! Client-side state: what has been fetched from the backend, live run data, and decoded images.

mod best;
mod images;
mod live;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::api::*;

pub use best::better_model;
pub use images::{ImageCache, ImgKey, ImgKind, ImgSlot, to_render_image};
pub use live::{Derived, RunData};

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
    pub datasets: Load<Vec<Dataset>>,
    pub dataset: HashMap<String, Load<Dataset>>,
    /// All image metadata of a dataset (cheap: boxes only), for filtering and the grid.
    pub images: HashMap<String, Load<Vec<ImageInfo>>>,
    pub models: Load<Vec<Model>>,
    pub evals: HashMap<String, Load<ModelEval>>,
    /// Per dataset, its issue counts by severity (fetched when it has broken labels).
    pub issues: HashMap<String, Load<IssueCounts>>,
    pub runs: Load<Vec<Run>>,
    pub run_data: HashMap<String, Load<RunData>>,
    pub samples: HashMap<String, Load<RunSamples>>,
    pub logs: HashMap<String, Load<Vec<LogLine>>>,
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
    /// Throughput history for the system card (GPU util samples), newest last.
    pub gpu_util: VecDeque<f32>,
}

impl Store {
    pub fn new(api: Arc<dyn Backend>) -> Self {
        Self {
            api,
            health: Load::Idle,
            system: Load::Idle,
            capabilities: None,
            features: Vec::new(),
            datasets: Load::Idle,
            dataset: HashMap::new(),
            images: HashMap::new(),
            models: Load::Idle,
            evals: HashMap::new(),
            issues: HashMap::new(),
            runs: Load::Idle,
            run_data: HashMap::new(),
            samples: HashMap::new(),
            logs: HashMap::new(),
            log_rev: 0,
            pending: HashMap::new(),
            resumed: Default::default(),
            lengths: HashMap::new(),
            images_px: ImageCache::default(),
            gpu_util: VecDeque::new(),
        }
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

    /// The server stopped answering (the last `/system` refresh couldn't connect).
    pub fn unreachable(&self) -> bool {
        matches!(&self.health, Load::Failed(e) if e.code == "server_unreachable")
    }

    /// What the periodic `/system` refresh says about the connection (`err`: its error, if any).
    /// `health`, which the status line shows, is fetched once per connection; without this a server
    /// that went away mid-session still read "Connected". Unreachable marks it down; the first answer
    /// after that has health checked again.
    pub fn note_reachability(&mut self, err: Option<&ApiError>) {
        match err {
            Some(e) if e.code == "server_unreachable" => self.health = Load::Failed(e.clone()),
            None if matches!(self.health, Load::Failed(_)) => self.health = Load::Idle,
            _ => {}
        }
    }
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
