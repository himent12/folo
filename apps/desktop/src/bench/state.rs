//! What the Benchmarks screens hold: the tabs, the forms, an entry's paged images and the state per server.

use std::collections::{HashMap, HashSet};

use folo_ui::Entity;

use crate::api::*;
use crate::input::TextInput;
use crate::store::Load;

/// The tabs of a benchmark's page.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BenchTab {
    #[default]
    Leaderboard,
    Plan,
    Loops,
}

/// The splits a benchmark can be made from, in the order the form offers them (`val` first: a
/// benchmark should be held out from training).
pub const BENCH_SPLITS: [Split; 3] = [Split::Val, Split::Test, Split::Train];

/// The New benchmark form on the Benchmarks page.
pub struct NewBenchForm {
    pub name: Entity<TextInput>,
    /// Bumped on every opening: a late answer only touches its own form.
    pub seq: u64,
    /// Index into [`FoloApp::bench_sources`]: the datasets, then the labeling jobs with labeled images.
    ///
    /// [`FoloApp::bench_sources`]: crate::app::FoloApp::bench_sources
    pub dataset: usize,
    /// Index into [`BENCH_SPLITS`] (a dataset's split; a labeling job has none).
    pub split: usize,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

/// Who runs on a benchmark from its leaderboard: a model of the server, or a coding agent.
#[derive(Clone, Debug, PartialEq)]
pub enum Contender {
    Model(String),
    Agent(LabelAgentId),
}

/// The New loop form on a benchmark's Loops tab.
pub struct NewLoopForm {
    /// The benchmark it was opened on, and its opening (a late answer only touches its own form).
    pub bench: String,
    pub seq: u64,
    /// The teacher, a model id.
    pub teacher: Option<String>,
    /// Index into [`FoloApp::bench_sources`].
    ///
    /// [`FoloApp::bench_sources`]: crate::app::FoloApp::bench_sources
    pub pool: usize,
    /// Index into [`LOOP_GENERATIONS`].
    pub generations: usize,
    /// Index into [`LOOP_EPOCHS`].
    pub epochs: usize,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

pub const LOOP_GENERATIONS: [u32; 3] = [1, 2, 3];
pub const LOOP_EPOCHS: [u32; 3] = [10, 30, 60];

/// "Design with an agent" on a benchmark's Plan tab: the request, then the proposal beside the plan.
pub struct DesignForm {
    pub bench: String,
    pub agent: LabelAgentId,
    pub note: Entity<TextInput>,
    pub asking: bool,
    /// The agent's plan, waiting to be used or discarded.
    pub proposal: Option<BenchDesign>,
    /// "Use this plan" is out.
    pub applying: bool,
    pub error: Option<ApiError>,
    /// Bumped on every opening: a late answer only touches its own form.
    pub seq: u64,
}

/// How an entry's images are ordered.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ImageSort {
    /// Most missed plus wrong boxes first.
    #[default]
    Worst,
    /// The benchmark's order.
    Order,
}

impl ImageSort {
    pub fn as_str(self) -> &'static str {
        match self {
            ImageSort::Worst => "worst",
            ImageSort::Order => "order",
        }
    }
}

/// Images an entry page shows at first, and how many more each Show more adds.
pub const ENTRY_IMAGES_PAGE: u32 = 24;

/// An entry's images as far as they are paged in.
pub struct EntryImages {
    pub sort: ImageSort,
    /// The pages so far, in one: `total` is the entry's whole count.
    pub page: Load<BenchEntryImagePage>,
    /// A next page is out.
    pub more: bool,
    pub more_error: Option<ApiError>,
    /// This load's number: an answer for an earlier sort or load is dropped.
    pub seq: u64,
}

/// What a reload fetches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reload {
    Entries,
    Loops,
    List,
}

/// Everything the Benchmarks screens show, per server (cleared on a source switch).
#[derive(Default)]
pub struct BenchState {
    pub list: Load<Vec<Bench>>,
    pub catalog: Load<BenchCatalog>,
    /// Per benchmark: its leaderboard, as the server orders it.
    pub entries: HashMap<String, Load<Vec<BenchEntry>>>,
    /// Per benchmark: its loops, newest first.
    pub loops: HashMap<String, Load<Vec<BenchLoop>>>,
    /// Per (benchmark, entry): everything measured.
    pub results: HashMap<(String, String), Load<BenchResult>>,
    pub tab: BenchTab,
    /// The open benchmark's event stream (dropping it closes the stream).
    pub events: Option<(String, BenchSubscription)>,
    /// A stream being opened (off the UI thread), for this benchmark.
    pub subscribing: Option<String>,
    /// The benchmark the last stream was opened for: a page shown anew opens its stream at once, a
    /// stream that failed or dropped is retried every 2 s.
    pub sub_for: Option<String>,
    pub new_bench: Option<NewBenchForm>,
    pub new_loop: Option<NewLoopForm>,
    pub design: Option<DesignForm>,
    /// Bumped on every form opening and every image load (their late answers check it).
    pub form_seq: u64,
    /// Events applied so far, counted; a reload remembers the count it was asked at.
    pub ev_seq: u64,
    /// Per entry, loop or benchmark id: the count at the event (or answer) that last changed it.
    pub touched: HashMap<String, u64>,
    /// Ids the stream removed, with the count, so an older snapshot doesn't bring them back.
    pub removed: HashMap<String, u64>,
    /// Stream values that arrived while their list was still on its first load: the first answer is
    /// merged with them as a later one is with the list on screen.
    pub early_list: Vec<Bench>,
    pub early_entries: HashMap<String, Vec<BenchEntry>>,
    pub early_loops: HashMap<String, Vec<BenchLoop>>,
    /// Reloads out, by (benchmark, or "" for the list; what): whether another is wanted after it.
    pub reloading: HashMap<(String, Reload), bool>,
    /// The coding agents of the server and its labeling jobs (servers with `agent_labeling`).
    pub agents: Load<Vec<LabelAgent>>,
    pub jobs: Load<Vec<LabelJob>>,
    /// Per (benchmark, entry): its images in an order, as far as they are paged in.
    pub entry_images: HashMap<(String, String), EntryImages>,
    pub image_sort: ImageSort,
    /// Per benchmark: its first image, shown under each condition on the Plan tab.
    pub previews: HashMap<String, Load<BenchImagePage>>,
    /// Who runs on the open benchmark (the newest model until one is picked).
    pub run_model: Option<Contender>,
    /// A Run model request is out.
    pub starting: bool,
    /// What waits on a second click to be deleted: (benchmark, an entry or loop id, or `None` for the
    /// benchmark itself).
    pub confirm_delete: Option<(String, Option<String>)>,
    /// Entries, loops and benchmarks with a stop or delete request out.
    pub busy: HashSet<String>,
    /// The list is shown again: refresh it in place.
    pub list_stale: bool,
}
