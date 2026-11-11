//! Vision benchmarks (servers with `vision_bench`, docs/BENCH.md): what the Benchmarks screens
//! hold, how they load, the open benchmark's live stream and the actions they take.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::TryRecvError;

use gpui::{AppContext, Context, Entity, Focusable, Window};

use crate::api::*;
use crate::app::{FoloApp, Route};
use crate::input::TextInput;
use crate::store::{ImgKey, ImgKind, ImgSlot, Load};
use crate::ui::Tone;

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
    pub dataset: usize,
    /// Index into [`BENCH_SPLITS`] (a dataset's split; a labeling job has none).
    pub split: usize,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

/// Where a benchmark's images, or a loop's pool, can come from.
#[derive(Clone, Debug, PartialEq)]
pub enum SourcePick {
    Dataset {
        id: String,
        name: String,
        splits: Splits,
    },
    /// A labeling job's labeled images.
    Job {
        id: String,
        name: String,
        labeled: u64,
    },
}

impl SourcePick {
    pub fn name(&self) -> &str {
        match self {
            SourcePick::Dataset { name, .. } | SourcePick::Job { name, .. } => name,
        }
    }

    /// How many images it holds (a dataset in `split`).
    pub fn images(&self, split: Split) -> u64 {
        match self {
            SourcePick::Dataset { splits, .. } => match split {
                Split::Train => splits.train,
                Split::Val => splits.val,
                Split::Test => splits.test,
            },
            SourcePick::Job { labeled, .. } => *labeled,
        }
    }

    /// The source a request sends.
    pub fn source(&self, split: Split) -> BenchSource {
        match self {
            SourcePick::Dataset { id, .. } => {
                BenchSource::Dataset { dataset_id: id.clone(), split: Some(split) }
            }
            SourcePick::Job { id, .. } => BenchSource::LabelJob { job_id: id.clone() },
        }
    }

    /// The name the New benchmark form suggests for it.
    pub fn suggested_name(&self, split: Split) -> String {
        match self {
            SourcePick::Dataset { name, .. } => format!("{name} {}", split.as_str()),
            SourcePick::Job { name, .. } => name.clone(),
        }
    }
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

/// Installs a reloaded list over the one on screen. A snapshot can be older than the events applied
/// since it was asked for (count `since`): an item the stream changed after that keeps the stream's
/// version (unless the snapshot has it ended and the stream doesn't: an end is final), one the stream
/// removed stays removed, and one the stream added stays. Returns the list and whether it disagrees
/// with the snapshot in a way that can change the order (another reload then settles it).
#[allow(clippy::too_many_arguments)]
pub fn merge_snapshot<T: Clone>(
    current: Option<&[T]>,
    fresh: Vec<T>,
    since: u64,
    touched: &HashMap<String, u64>,
    removed: &HashMap<String, u64>,
    id: impl Fn(&T) -> &str,
    ended: impl Fn(&T) -> bool,
    place: impl Fn(&mut Vec<T>, T),
) -> (Vec<T>, bool) {
    let newer = |i: &str| touched.get(i).is_some_and(|&s| s > since);
    let gone = |i: &str| removed.get(i).is_some_and(|&s| s > since);
    let mut differs = false;
    let mut out: Vec<T> = Vec::with_capacity(fresh.len());
    for f in fresh {
        if gone(id(&f)) {
            continue;
        }
        let cur = current.and_then(|c| c.iter().find(|x| id(x) == id(&f)));
        match cur {
            Some(c) if newer(id(&f)) && !(ended(&f) && !ended(c)) => {
                differs |= ended(c) != ended(&f);
                out.push(c.clone());
            }
            _ => out.push(f),
        }
    }
    for c in current.into_iter().flatten() {
        if newer(id(c)) && !gone(id(c)) && !out.iter().any(|o| id(o) == id(c)) {
            differs |= ended(c);
            place(&mut out, c.clone());
        }
    }
    (out, differs)
}

/// The severity a condition is previewed at: the middle of the ones the plan runs.
pub fn preview_severity(severities: &[u8]) -> u8 {
    let mut s = severities.to_vec();
    s.sort_unstable();
    s.get(s.len() / 2).copied().unwrap_or(3)
}

/// The image cache's key for a benchmark image: one "dataset" per benchmark, and the condition in the
/// image's key, so every variant has its own slot.
pub fn bench_img_key(bench: &str, image: &str, condition: Option<(&str, u8)>) -> (String, String) {
    let img = match condition {
        Some((c, s)) => format!("{image}@{c}:{s}"),
        None => image.to_string(),
    };
    (format!("bench:{bench}"), img)
}

/// A class name as the server compares them: trimmed, without case.
fn class_key(s: &str) -> String {
    s.trim().to_lowercase()
}

/// A model can run on a benchmark when it knows at least one of its classes (by name, without
/// case); the server refuses one that knows none (`bench_class_mismatch`).
pub fn model_fits(model: &[String], bench: &[String]) -> bool {
    model.iter().any(|m| bench.iter().any(|b| class_key(b) == class_key(m)))
}

/// A model can teach a loop on a benchmark when its classes are the benchmark's, the same names in
/// the same order: its labels are written by class index.
pub fn model_teaches(model: &[String], bench: &[String]) -> bool {
    model.len() == bench.len() && model.iter().zip(bench).all(|(m, b)| class_key(m) == class_key(b))
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

/// A benchmark route's benchmark id, if the route is one.
pub fn route_bench(route: &Route) -> Option<&str> {
    match route {
        Route::Bench(id) | Route::BenchEntry(id, _) => Some(id),
        _ => None,
    }
}

/// Inserts `e` where it belongs in a leaderboard that already has the rest, or replaces its old
/// version in place. A new entry goes first among the unfinished ones (the server lists those
/// newest first, after every finished one).
pub fn upsert_entry(list: &mut Vec<BenchEntry>, e: BenchEntry) {
    if let Some(old) = list.iter_mut().find(|x| x.id == e.id) {
        // An entry that has ended never runs again: an older answer can't bring it back.
        if !old.status.is_active() && e.status.is_active() {
            return;
        }
        *old = e;
    } else {
        let at = list.iter().position(|x| x.status != BenchEntryStatus::Done).unwrap_or(list.len());
        list.insert(at, e);
    }
}

/// Inserts `l` first (newest first), or replaces its old version in place.
pub fn upsert_loop(list: &mut Vec<BenchLoop>, l: BenchLoop) {
    if let Some(old) = list.iter_mut().find(|x| x.id == l.id) {
        if !old.status.is_active() && l.status.is_active() {
            return;
        }
        *old = l;
    } else {
        list.insert(0, l);
    }
}

impl FoloApp {
    /// The server has the `/api/bench` endpoints.
    pub fn bench_on(&self) -> bool {
        self.store.features.iter().any(|f| f == FEATURE_VISION_BENCH)
    }

    /// The server is known not to have them (its system info is in and doesn't list the feature).
    pub fn bench_unsupported(&self) -> bool {
        self.store.system.ready().is_some() && !self.bench_on()
    }

    /// Loads the benchmark list and the catalog, once each.
    pub fn ensure_benches(&mut self, cx: &mut Context<Self>) {
        if !self.bench_on() {
            return;
        }
        if self.bench.list.needs_fetch() {
            // Through the same merge as every later reload: an answer older than the stream's
            // events never undoes them.
            self.bench.list = Load::Loading;
            self.refresh_benches(cx);
        }
        if self.bench.catalog.needs_fetch() {
            self.bench.catalog = Load::Loading;
            self.fetch(cx, |api| api.bench_catalog(), |this, r, _| this.bench.catalog = r.into());
        }
        self.ensure_agents(cx);
    }

    /// Loads a benchmark's entries and loops, once each.
    pub fn ensure_bench(&mut self, id: &str, cx: &mut Context<Self>) {
        self.ensure_benches(cx);
        if !self.bench_on() {
            return;
        }
        if self.bench.entries.get(id).is_none_or(|l| l.needs_fetch()) {
            self.bench.entries.insert(id.to_string(), Load::Loading);
            self.load_bench_entries(id, cx);
        }
        if self.bench.loops.get(id).is_none_or(|l| l.needs_fetch()) {
            self.bench.loops.insert(id.to_string(), Load::Loading);
            self.load_bench_loops(id, cx);
        }
    }

    /// (Re)loads a benchmark's leaderboard, merged with what the stream said meanwhile.
    pub fn load_bench_entries(&mut self, id: &str, cx: &mut Context<Self>) {
        self.reload(id, Reload::Entries, cx);
    }

    /// (Re)loads a benchmark's loops, merged with what the stream said meanwhile.
    pub fn load_bench_loops(&mut self, id: &str, cx: &mut Context<Self>) {
        self.reload(id, Reload::Loops, cx);
    }

    /// Reloads the benchmark list in place (no loading state once it has one).
    pub fn refresh_benches(&mut self, cx: &mut Context<Self>) {
        self.reload("", Reload::List, cx);
    }

    /// One reload per list at a time: asked for while one is out, it runs once more after it (and
    /// so does one whose snapshot disagreed with the stream on what has ended). A failed reload keeps
    /// the list there is.
    fn reload(&mut self, bench: &str, what: Reload, cx: &mut Context<Self>) {
        let key = (bench.to_string(), what);
        if let Some(again) = self.bench.reloading.get_mut(&key) {
            *again = true;
            return;
        }
        self.bench.reloading.insert(key.clone(), false);
        let since = self.bench.ev_seq;
        let b = bench.to_string();
        let finish = move |this: &mut Self, differs: bool, cx: &mut Context<Self>| {
            let again = this.bench.reloading.remove(&key).unwrap_or(false);
            if again || differs {
                this.reload(&key.0, key.1, cx);
            }
        };
        match what {
            Reload::Entries => self.fetch(
                cx,
                {
                    let b = b.clone();
                    move |api| api.bench_entries(&b)
                },
                move |this, r, cx| {
                    let mut differs = false;
                    match r {
                        Ok(fresh) => {
                            let s = &this.bench;
                            let current = s
                                .entries
                                .get(&b)
                                .and_then(|l| l.ready())
                                .or_else(|| s.early_entries.get(&b))
                                .map(Vec::as_slice);
                            let (list, d) = merge_snapshot(
                                current,
                                fresh,
                                since,
                                &s.touched,
                                &s.removed,
                                |e: &BenchEntry| e.id.as_str(),
                                |e| !e.status.is_active(),
                                upsert_entry,
                            );
                            differs = d;
                            this.bench.early_entries.remove(&b);
                            this.bench.entries.insert(b, Load::Ready(list));
                        }
                        Err(e) => {
                            if this.bench.entries.get(&b).is_none_or(|l| l.ready().is_none()) {
                                this.bench.entries.insert(b, Load::Failed(e));
                            }
                        }
                    }
                    finish(this, differs, cx);
                },
            ),
            Reload::Loops => self.fetch(
                cx,
                {
                    let b = b.clone();
                    move |api| api.bench_loops(&b)
                },
                move |this, r, cx| {
                    match r {
                        Ok(fresh) => {
                            let s = &this.bench;
                            let current = s
                                .loops
                                .get(&b)
                                .and_then(|l| l.ready())
                                .or_else(|| s.early_loops.get(&b))
                                .map(Vec::as_slice);
                            let (list, _) = merge_snapshot(
                                current,
                                fresh,
                                since,
                                &s.touched,
                                &s.removed,
                                |l: &BenchLoop| l.id.as_str(),
                                |l| !l.status.is_active(),
                                upsert_loop,
                            );
                            this.bench.early_loops.remove(&b);
                            this.bench.loops.insert(b, Load::Ready(list));
                        }
                        Err(e) => {
                            if this.bench.loops.get(&b).is_none_or(|l| l.ready().is_none()) {
                                this.bench.loops.insert(b, Load::Failed(e));
                            }
                        }
                    }
                    // Loops are listed newest first: a kept version never moves one.
                    finish(this, false, cx);
                },
            ),
            Reload::List => self.fetch(
                cx,
                |api| api.benches(),
                move |this, r, cx| {
                    match r {
                        Ok(fresh) => {
                            let s = &this.bench;
                            let (list, _) = merge_snapshot(
                                s.list.ready().map(Vec::as_slice).or(Some(s.early_list.as_slice())),
                                fresh,
                                since,
                                &s.touched,
                                &s.removed,
                                |b: &Bench| b.id.as_str(),
                                |_| false,
                                |l, b| l.insert(0, b),
                            );
                            this.bench.early_list.clear();
                            this.bench.list = Load::Ready(list);
                        }
                        Err(e) => {
                            if this.bench.list.ready().is_none() {
                                this.bench.list = Load::Failed(e);
                            }
                        }
                    }
                    finish(this, false, cx);
                },
            ),
        }
    }

    /// Notes that `id` is gone (deleted here, or removed on the stream): older snapshots don't bring
    /// it back.
    fn forget(&mut self, id: &str) {
        self.touch(id);
        self.bench.removed.insert(id.to_string(), self.bench.ev_seq);
    }

    /// Notes that the stream (or an answer) changed `id` now: older snapshots keep this version.
    fn touch(&mut self, id: &str) {
        self.bench.ev_seq += 1;
        self.bench.touched.insert(id.to_string(), self.bench.ev_seq);
    }

    /// Loads a finished entry's full result, once.
    pub fn ensure_bench_result(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        let key = (bench.to_string(), entry.to_string());
        if self.bench.results.get(&key).is_some_and(|l| !l.needs_fetch()) {
            return;
        }
        self.bench.results.insert(key.clone(), Load::Loading);
        let (b, e) = key.clone();
        self.fetch(
            cx,
            move |api| api.bench_result(&b, &e),
            move |this, r, _| {
                this.bench.results.insert(key, r.into());
            },
        );
    }

    /// Forgets everything benchmark (a new data source, or Retry).
    pub fn clear_bench(&mut self) {
        let (tab, form_seq, image_sort) = (self.bench.tab, self.bench.form_seq, self.bench.image_sort);
        self.bench = BenchState { tab, form_seq, image_sort, ..BenchState::default() };
    }

    /// The Benchmarks section shows the server as it is now (also when going back to it): the list
    /// refreshes in place on the next render; a benchmark's page reloads when its stream opens.
    pub fn after_bench_navigate(&mut self) {
        self.bench.confirm_delete = None;
        // Forms belong to the page they were opened on.
        if self.bench.design.as_ref().is_some_and(|f| route_bench(&self.route) != Some(f.bench.as_str())) {
            self.bench.design = None;
        }
        if !matches!(self.route, Route::Bench(_)) {
            self.bench.new_loop = None;
        }
        if matches!(self.route, Route::Benches) {
            self.bench.list_stale = true;
        }
        if let Route::Bench(_) = self.route {
            // A benchmark opens on its leaderboard, whatever tab the last one was on.
            self.bench.tab = BenchTab::Leaderboard;
        }
    }

    /// Called from the app's 100 ms pump: applies the open benchmark's events, keeps its stream open
    /// while a benchmark page shows and closes it when none does.
    pub fn bench_pump(&mut self, tick: u64, cx: &mut Context<Self>) -> bool {
        let open = route_bench(&self.route).map(str::to_string);
        if self.bench.events.as_ref().is_some_and(|(id, _)| Some(id) != open.as_ref()) {
            self.bench.events = None;
        }
        let Some(id) = open else {
            self.bench.sub_for = None;
            return false;
        };
        if !self.bench_on() {
            return false;
        }
        let mut changed = false;
        let mut ended = false;
        let mut reload = false;
        loop {
            let ev = match self.bench.events.as_ref().map(|(_, rx)| rx.try_recv()) {
                Some(Ok(ev)) => ev,
                Some(Err(TryRecvError::Empty)) | None => break,
                Some(Err(TryRecvError::Disconnected)) => {
                    ended = true;
                    break;
                }
            };
            changed = true;
            match ev {
                BenchEvent::Bench(b) => {
                    self.touch(&b.id);
                    let list = match self.bench.list.ready_mut() {
                        Some(list) => list,
                        None => &mut self.bench.early_list,
                    };
                    match list.iter_mut().find(|x| x.id == b.id) {
                        Some(old) => *old = *b,
                        None => list.insert(0, *b),
                    }
                    // Sent when the stream opens, at a benchmark's milestones and to a subscriber that
                    // fell behind (whose missed events are gone): the lists are reconciled from the
                    // server.
                    reload = true;
                }
                BenchEvent::Entry(e) => {
                    self.touch(&e.id);
                    let finished = !e.status.is_active();
                    let known = self
                        .bench
                        .entries
                        .get(&id)
                        .and_then(|l| l.ready())
                        .and_then(|l| l.iter().find(|x| x.id == e.id))
                        .map(|x| x.status.is_active());
                    // An entry that ends here, or one first seen ended (predictions sent in one
                    // request are scored at once): its place on the leaderboard comes from the
                    // server's order.
                    if finished && known != Some(false) {
                        reload = true;
                    }
                    if finished && known == Some(true) {
                        if e.status == BenchEntryStatus::Done {
                            let detail = e.score.as_ref().map(|s| format!("Score {:.1}", s.total).into());
                            self.toast(Tone::Success, format!("{} finished", e.name), detail, cx);
                        } else if e.status == BenchEntryStatus::Failed {
                            self.toast(
                                Tone::Danger,
                                format!("{} failed", e.name),
                                e.error.clone().map(Into::into),
                                cx,
                            );
                        }
                        self.bench.results.remove(&(id.clone(), e.id.clone()));
                    }
                    match self.bench.entries.get_mut(&id).and_then(|l| l.ready_mut()) {
                        Some(list) => upsert_entry(list, *e),
                        None => upsert_entry(self.bench.early_entries.entry(id.clone()).or_default(), *e),
                    }
                }
                BenchEvent::Loop(l) => {
                    self.touch(&l.id);
                    match self.bench.loops.get_mut(&id).and_then(|x| x.ready_mut()) {
                        Some(list) => upsert_loop(list, *l),
                        None => upsert_loop(self.bench.early_loops.entry(id.clone()).or_default(), *l),
                    }
                }
                BenchEvent::Removed(r) => {
                    self.forget(&r.id);
                    if let Some(list) = self.bench.entries.get_mut(&id).and_then(|l| l.ready_mut()) {
                        list.retain(|x| x.id != r.id);
                    }
                    if let Some(list) = self.bench.loops.get_mut(&id).and_then(|l| l.ready_mut()) {
                        list.retain(|x| x.id != r.id);
                    }
                }
            }
        }
        if ended {
            // The stream dropped: what happened in the gap comes from a reload; it reopens below.
            self.bench.events = None;
            reload = true;
        }
        if reload {
            self.load_bench_entries(&id, cx);
            self.load_bench_loops(&id, cx);
            self.refresh_benches(cx);
        }
        // Open (or reopen, at most every 2 s) the stream of the benchmark on screen.
        let first = self.bench.sub_for.as_deref() != Some(id.as_str());
        if self.bench.events.is_none()
            && self.bench.subscribing.is_none()
            && (first || tick.is_multiple_of(20))
        {
            self.subscribe_bench(id, cx);
        }
        changed
    }

    fn subscribe_bench(&mut self, id: String, cx: &mut Context<Self>) {
        self.bench.subscribing = Some(id.clone());
        self.bench.sub_for = Some(id.clone());
        let a = id.clone();
        self.fetch(
            cx,
            move |api| api.bench_events(&a),
            move |this, r, _| {
                if this.bench.subscribing.as_deref() == Some(id.as_str()) {
                    this.bench.subscribing = None;
                }
                // Only the benchmark still on screen keeps its stream. What changed before it opened
                // (or while it was closed) comes from the reload its first event, the benchmark,
                // starts.
                if let Ok(sub) = r
                    && route_bench(&this.route) == Some(id.as_str())
                {
                    this.bench.events = Some((id, sub));
                }
            },
        );
    }

    // ---------- actions ----------

    /// The server runs coding agents (`agent_labeling`): agents can be contestants and design plans,
    /// and labeling jobs can be sources.
    pub fn agents_on(&self) -> bool {
        self.store.features.iter().any(|f| f == FEATURE_AGENT_LABELING)
    }

    /// Loads the server's coding agents and labeling jobs, once each (servers with `agent_labeling`).
    pub fn ensure_agents(&mut self, cx: &mut Context<Self>) {
        if !self.agents_on() {
            return;
        }
        if self.bench.agents.needs_fetch() {
            self.bench.agents = Load::Loading;
            self.fetch(cx, |api| api.label_agents(), |this, r, _| this.bench.agents = r.into());
        }
        if self.bench.jobs.needs_fetch() {
            self.bench.jobs = Load::Loading;
            self.fetch(cx, |api| api.label_jobs(), |this, r, _| this.bench.jobs = r.into());
        }
    }

    /// The agents that are installed on the server's machine.
    pub fn bench_agents(&self) -> Vec<LabelAgent> {
        self.bench
            .agents
            .ready()
            .map(|l| l.iter().filter(|a| a.available).cloned().collect())
            .unwrap_or_default()
    }

    /// Where images can come from: the datasets, then the labeling jobs with labeled images.
    pub fn bench_sources(&self) -> Vec<SourcePick> {
        let datasets = self.store.datasets.ready().into_iter().flatten().map(|d| SourcePick::Dataset {
            id: d.id.clone(),
            name: d.name.clone(),
            splits: d.splits,
        });
        let jobs =
            self.bench.jobs.ready().into_iter().flatten().filter(|j| j.counts.labeled > 0).map(|j| {
                SourcePick::Job { id: j.id.clone(), name: j.name.clone(), labeled: j.counts.labeled }
            });
        datasets.chain(jobs).collect()
    }

    /// Opens the New benchmark form, named after the first source.
    pub fn open_new_bench(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.ensure_agents(cx);
        let name = self
            .bench_sources()
            .first()
            .map_or_else(|| "benchmark".to_string(), |s| s.suggested_name(BENCH_SPLITS[0]));
        let input = cx.new(|cx| {
            let mut i = TextInput::new("Benchmark name", cx);
            i.set_text(name, cx);
            i
        });
        let fh = input.read(cx).focus_handle(cx);
        self.bench.form_seq += 1;
        let seq = self.bench.form_seq;
        self.bench.new_bench =
            Some(NewBenchForm { name: input, seq, dataset: 0, split: 0, submitting: false, error: None });
        window.focus(&fh);
        cx.notify();
    }

    /// Sets the New benchmark form's source and split, and its name when the name still follows them.
    pub fn pick_bench_source(&mut self, source: Option<usize>, split: Option<usize>, cx: &mut Context<Self>) {
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_bench.as_mut() else { return };
        let suggested = |i: usize, s: usize| sources.get(i).map(|x| x.suggested_name(BENCH_SPLITS[s]));
        let old = suggested(f.dataset, f.split);
        f.dataset = source.unwrap_or(f.dataset);
        f.split = split.unwrap_or(f.split).min(BENCH_SPLITS.len() - 1);
        f.error = None;
        let current = f.name.read(cx).text().to_string();
        if old.is_none_or(|o| o == current)
            && let Some(next) = suggested(f.dataset, f.split)
        {
            f.name.update(cx, |input, cx| input.set_text(next, cx));
        }
        cx.notify();
    }

    /// Creates the benchmark the form describes (the default plan) and opens it.
    pub fn submit_new_bench(&mut self, cx: &mut Context<Self>) {
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_bench.as_mut() else { return };
        if f.submitting {
            return;
        }
        let name = f.name.read(cx).text().trim().to_string();
        if name.is_empty() {
            f.error = Some(ApiError::new("invalid_body", "Name the benchmark"));
            cx.notify();
            return;
        }
        let Some(src) = sources.get(f.dataset) else {
            f.error =
                Some(ApiError::new("invalid_body", "Add a dataset first: a benchmark copies its images"));
            cx.notify();
            return;
        };
        let split = BENCH_SPLITS[f.split];
        if src.images(split) == 0 {
            f.error = Some(ApiError::new("invalid_body", "That source has no images: pick another"));
            cx.notify();
            return;
        }
        let req = CreateBench {
            name,
            description: String::new(),
            source: src.source(split),
            max_images: None,
            seed: 0,
            plan: None,
        };
        f.submitting = true;
        f.error = None;
        let seq = f.seq;
        self.fetch(
            cx,
            move |api| api.create_bench(&req),
            move |this, r, cx| {
                // The form that asked, still open (a late answer leaves a newer form alone).
                let mine = this.bench.new_bench.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(b) => {
                        this.touch(&b.id);
                        if let Some(list) = this.bench.list.ready_mut()
                            && !list.iter().any(|x| x.id == b.id)
                        {
                            list.insert(0, b.clone());
                        }
                        this.toast(Tone::Success, format!("Created {}", b.name), None, cx);
                        if mine {
                            this.bench.new_bench = None;
                            this.route_pending = Some(Route::Bench(b.id));
                        }
                    }
                    Err(e) => {
                        if let Some(f) = this.bench.new_bench.as_mut().filter(|f| f.seq == seq) {
                            f.submitting = false;
                            f.error = Some(e);
                        } else {
                            this.toast(
                                Tone::Danger,
                                "Couldn't create the benchmark",
                                Some(e.message.into()),
                                cx,
                            );
                        }
                    }
                }
            },
        );
        cx.notify();
    }

    /// Runs the picked contestant on a benchmark: a model on every image and the plan's conditions, an
    /// agent as the server runs one by default (its first images, unchanged only).
    pub fn run_bench_model(&mut self, bench: &str, cx: &mut Context<Self>) {
        let Some(b) = self.bench.list.ready().and_then(|l| l.iter().find(|x| x.id == bench)).cloned() else {
            return;
        };
        let Some(who) = self.bench_run_model(&b) else { return };
        if self.bench.starting {
            return;
        }
        self.bench.starting = true;
        let contestant = match who {
            Contender::Model(model_id) => BenchContestantSpec::Model { model_id },
            Contender::Agent(agent) => BenchContestantSpec::Agent { agent, model: None, effort: None },
        };
        let req = StartBenchEntry { contestant, name: None, max_images: None, conditions: None };
        let (a, b) = (bench.to_string(), bench.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.start_bench_entry(&a, &req),
            move |this, r, cx| {
                this.bench.starting = false;
                match r {
                    Ok(e) => {
                        let name = e.name.clone();
                        this.apply_entry(&b, e, since);
                        this.toast(
                            Tone::Info,
                            format!("{name} started"),
                            Some("It appears on the leaderboard as it runs.".into()),
                            cx,
                        );
                        this.refresh_benches(cx);
                    }
                    Err(e) => {
                        this.toast(Tone::Danger, "Couldn't start the entry", Some(e.message.into()), cx)
                    }
                }
            },
        );
    }

    /// Who Run would start on `b`: the contestant picked while it can still run there, else the newest
    /// model that knows its classes, else the first installed agent.
    pub fn bench_run_model(&self, b: &Bench) -> Option<Contender> {
        let options = self.bench_contenders(b);
        self.bench
            .run_model
            .as_ref()
            .filter(|c| options.contains(c))
            .cloned()
            .or_else(|| options.into_iter().next())
    }

    /// Everyone the leaderboard of `b` can run: the models that know its classes, newest first, then
    /// the installed agents (which are asked for its classes by name).
    pub fn bench_contenders(&self, b: &Bench) -> Vec<Contender> {
        let models = self
            .store
            .models
            .ready()
            .into_iter()
            .flatten()
            .filter(|m| model_fits(&m.classes, &b.classes))
            .map(|m| Contender::Model(m.id.clone()));
        let agents = self.bench_agents().into_iter().map(|a| Contender::Agent(a.id));
        models.chain(agents).collect()
    }

    /// Applies an entry the server answered with (a start or a stop), unless the stream has said
    /// something about it since the request went out (its answer is at least as new).
    pub fn apply_entry(&mut self, bench: &str, e: BenchEntry, since: u64) {
        if self.bench.touched.get(&e.id).is_some_and(|&s| s > since) {
            return;
        }
        self.touch(&e.id);
        if let Some(list) = self.bench.entries.get_mut(bench).and_then(|l| l.ready_mut()) {
            upsert_entry(list, e);
        }
    }

    /// Applies a loop the server answered with, as [`Self::apply_entry`] does.
    pub fn apply_loop(&mut self, bench: &str, l: BenchLoop, since: u64) {
        if self.bench.touched.get(&l.id).is_some_and(|&s| s > since) {
            return;
        }
        self.touch(&l.id);
        if let Some(list) = self.bench.loops.get_mut(bench).and_then(|x| x.ready_mut()) {
            upsert_loop(list, l);
        }
    }

    /// An entry's first page of images in the chosen order (again when the order changes).
    pub fn ensure_entry_images(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        let key = (bench.to_string(), entry.to_string());
        let sort = self.bench.image_sort;
        if self.bench.entry_images.get(&key).is_some_and(|x| x.sort == sort && !x.page.needs_fetch()) {
            return;
        }
        self.bench.form_seq += 1;
        let seq = self.bench.form_seq;
        let state = EntryImages { sort, page: Load::Loading, more: false, more_error: None, seq };
        self.bench.entry_images.insert(key.clone(), state);
        let (b, e) = key.clone();
        self.fetch(
            cx,
            move |api| api.bench_entry_images(&b, &e, sort.as_str(), 0, ENTRY_IMAGES_PAGE),
            move |this, r, _| {
                if let Some(x) = this.bench.entry_images.get_mut(&key).filter(|x| x.seq == seq) {
                    x.page = r.into();
                }
            },
        );
    }

    /// The next page of an entry's images, after the ones there are.
    pub fn more_entry_images(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        let key = (bench.to_string(), entry.to_string());
        let Some(x) = self.bench.entry_images.get_mut(&key) else { return };
        let Some(page) = x.page.ready() else { return };
        let offset = page.images.len() as u32;
        if x.more || u64::from(offset) >= page.total {
            return;
        }
        x.more = true;
        x.more_error = None;
        let (sort, seq) = (x.sort, x.seq);
        let (b, e) = key.clone();
        self.fetch(
            cx,
            move |api| api.bench_entry_images(&b, &e, sort.as_str(), offset, ENTRY_IMAGES_PAGE),
            move |this, r, _| {
                let Some(x) = this.bench.entry_images.get_mut(&key).filter(|x| x.seq == seq) else { return };
                x.more = false;
                match (r, x.page.ready_mut()) {
                    (Ok(next), Some(page)) if page.images.len() as u32 == offset => {
                        page.images.extend(next.images);
                        page.total = next.total;
                    }
                    (Err(e), _) => x.more_error = Some(e),
                    _ => {}
                }
            },
        );
    }

    /// A benchmark's first image, the one the Plan tab shows under each condition.
    pub fn ensure_previews(&mut self, bench: &str, cx: &mut Context<Self>) {
        if self.bench.previews.get(bench).is_some_and(|l| !l.needs_fetch()) {
            return;
        }
        self.bench.previews.insert(bench.to_string(), Load::Loading);
        let (a, b) = (bench.to_string(), bench.to_string());
        self.fetch(
            cx,
            move |api| api.bench_images(&a, 0, 1, None),
            move |this, r, _| {
                this.bench.previews.insert(b, r.into());
            },
        );
    }

    /// A benchmark image (under a condition, or unchanged) as a thumbnail, decoded and cached with the
    /// app's other images; `None` until it is in.
    pub fn bench_image(
        &mut self,
        bench: &str,
        image: &str,
        condition: Option<(&str, u8)>,
        cx: &mut Context<Self>,
    ) -> Option<Arc<gpui::RenderImage>> {
        let (dataset, img) = bench_img_key(bench, image, condition);
        let key = ImgKey { dataset, image: img, kind: ImgKind::Thumb };
        match self.store.images_px.get(&key) {
            Some(ImgSlot::Ready(img)) => return Some(img.clone()),
            Some(_) => return None,
            None => {}
        }
        if self.hold_loading || !self.store.images_px.begin(&key) {
            return None;
        }
        let (b, i, c) = (bench.to_string(), image.to_string(), condition.map(|(c, s)| (c.to_string(), s)));
        self.load_cached_image(
            key,
            move |api| api.bench_image(&b, &i, c.as_ref().map(|(c, s)| (c.as_str(), *s)), Some(320)),
            cx,
        );
        None
    }

    /// The models that can teach a loop on `b` (its classes, in its order), newest first.
    pub fn bench_teachers(&self, b: &Bench) -> Vec<Model> {
        let models = self.store.models.ready().into_iter().flatten();
        models.filter(|m| model_teaches(&m.classes, &b.classes)).cloned().collect()
    }

    /// Opens the New loop form for the benchmark on screen: its best model that can teach (else the
    /// newest that can), on the first source's train split.
    pub fn open_new_loop(&mut self, cx: &mut Context<Self>) {
        self.ensure_agents(cx);
        let Some(bench) = route_bench(&self.route).map(str::to_string) else { return };
        let b = self.bench.list.ready().and_then(|l| l.iter().find(|x| x.id == bench)).cloned();
        let teachers = b.as_ref().map(|b| self.bench_teachers(b)).unwrap_or_default();
        let best = self.bench.entries.get(&bench).and_then(|l| l.ready()).and_then(|l| {
            l.iter().filter(|e| e.status == BenchEntryStatus::Done).find_map(|e| match &e.contestant {
                BenchContestant::Model { model_id, .. } if teachers.iter().any(|m| &m.id == model_id) => {
                    Some(model_id.clone())
                }
                _ => None,
            })
        });
        let teacher = best.or_else(|| teachers.first().map(|m| m.id.clone()));
        self.bench.form_seq += 1;
        self.bench.new_loop = Some(NewLoopForm {
            bench,
            seq: self.bench.form_seq,
            teacher,
            pool: 0,
            generations: 0,
            epochs: 1,
            submitting: false,
            error: None,
        });
        cx.notify();
    }

    /// Starts the loop the New loop form describes.
    pub fn submit_new_loop(&mut self, bench: &str, cx: &mut Context<Self>) {
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_loop.as_mut() else { return };
        if f.submitting {
            return;
        }
        let Some(teacher) = f.teacher.clone() else {
            f.error = Some(ApiError::new("invalid_body", "Pick a teacher: a model of this server"));
            cx.notify();
            return;
        };
        let Some(pool) = sources.get(f.pool) else {
            f.error =
                Some(ApiError::new("invalid_body", "Add a dataset first: the teacher labels its images"));
            cx.notify();
            return;
        };
        let req = StartBenchLoop {
            name: None,
            teacher,
            pool: pool.source(Split::Train),
            generations: Some(LOOP_GENERATIONS[f.generations]),
            min_confidence: None,
            student: BenchStudent { epochs: Some(LOOP_EPOCHS[f.epochs]), ..BenchStudent::default() },
        };
        f.submitting = true;
        f.error = None;
        let seq = f.seq;
        let (a, b) = (bench.to_string(), bench.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.start_bench_loop(&a, &req),
            move |this, r, cx| {
                // Only the form that asked (same benchmark, same opening) closes or shows the error.
                let mine = |f: &NewLoopForm| f.seq == seq && f.bench == b;
                match r {
                    Ok(l) => {
                        if this.bench.new_loop.as_ref().is_some_and(mine) {
                            this.bench.new_loop = None;
                        }
                        let name = l.name.clone();
                        this.apply_loop(&b, l, since);
                        this.toast(
                            Tone::Info,
                            format!("{name} started"),
                            Some(
                                "Its generations appear on its benchmark's Loops tab as they finish.".into(),
                            ),
                            cx,
                        );
                        this.refresh_benches(cx);
                    }
                    Err(e) => match this.bench.new_loop.as_mut().filter(|f| mine(f)) {
                        Some(f) => {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                        None => {
                            this.toast(Tone::Danger, "Couldn't start the loop", Some(e.message.into()), cx)
                        }
                    },
                }
            },
        );
        cx.notify();
    }

    /// Opens "Design with an agent" for a benchmark, with the first installed agent.
    pub fn open_design(&mut self, bench: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.ensure_agents(cx);
        let agent = self.bench_agents().first().map_or(LabelAgentId::Claude, |a| a.id);
        let note = cx.new(|cx| TextInput::new("Optional: what matters most, what to leave out", cx));
        let fh = note.read(cx).focus_handle(cx);
        self.bench.form_seq += 1;
        self.bench.design = Some(DesignForm {
            bench: bench.to_string(),
            agent,
            note,
            asking: false,
            proposal: None,
            applying: false,
            error: None,
            seq: self.bench.form_seq,
        });
        window.focus(&fh);
        cx.notify();
    }

    /// Asks the chosen agent for a plan.
    pub fn ask_design(&mut self, cx: &mut Context<Self>) {
        let Some(f) = self.bench.design.as_mut() else { return };
        if f.asking {
            return;
        }
        let req = DesignBench {
            agent: f.agent,
            model: None,
            effort: None,
            note: f.note.read(cx).text().trim().to_string(),
        };
        f.asking = true;
        f.error = None;
        let (id, seq) = (f.bench.clone(), f.seq);
        self.fetch(
            cx,
            move |api| api.design_bench(&id, &req),
            move |this, r, _| {
                let Some(f) = this.bench.design.as_mut().filter(|f| f.seq == seq) else { return };
                f.asking = false;
                match r {
                    Ok(d) => f.proposal = Some(d),
                    Err(e) => f.error = Some(e),
                }
            },
        );
        cx.notify();
    }

    /// Makes the proposed plan the benchmark's.
    pub fn apply_design(&mut self, cx: &mut Context<Self>) {
        let Some(f) = self.bench.design.as_mut() else { return };
        let Some(plan) = f.proposal.as_ref().map(|d| d.plan.clone()) else { return };
        if f.applying {
            return;
        }
        f.applying = true;
        f.error = None;
        let (id, seq) = (f.bench.clone(), f.seq);
        let req = UpdateBench { name: None, description: None, plan: Some(plan) };
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.update_bench(&id, &req),
            move |this, r, cx| match r {
                Ok(b) => {
                    if this.bench.design.as_ref().is_some_and(|f| f.seq == seq) {
                        this.bench.design = None;
                    }
                    // The benchmark as the answer has it, unless the stream has said something newer.
                    if !this.bench.touched.get(&b.id).is_some_and(|&s| s > since) {
                        this.touch(&b.id);
                        if let Some(list) = this.bench.list.ready_mut()
                            && let Some(old) = list.iter_mut().find(|x| x.id == b.id)
                        {
                            *old = b;
                        }
                    }
                    this.toast(Tone::Success, "Plan updated", None, cx);
                }
                Err(e) => {
                    if let Some(f) = this.bench.design.as_mut().filter(|f| f.seq == seq) {
                        f.applying = false;
                        f.error = Some(e);
                    }
                }
            },
        );
        cx.notify();
    }

    /// Stops a queued or running entry.
    pub fn stop_bench_entry(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        if !self.bench.busy.insert(entry.to_string()) {
            return;
        }
        let (a, e1, b, e2) = (bench.to_string(), entry.to_string(), bench.to_string(), entry.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.stop_bench_entry(&a, &e1),
            move |this, r, cx| {
                this.bench.busy.remove(&e2);
                match r {
                    Ok(e) => this.apply_entry(&b, e, since),
                    Err(e) => this.toast(Tone::Danger, "Couldn't stop the entry", Some(e.message.into()), cx),
                }
            },
        );
        cx.notify();
    }

    /// Stops a running loop.
    pub fn stop_bench_loop(&mut self, bench: &str, id: &str, cx: &mut Context<Self>) {
        if !self.bench.busy.insert(id.to_string()) {
            return;
        }
        let (a, l1, b, l2) = (bench.to_string(), id.to_string(), bench.to_string(), id.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.stop_bench_loop(&a, &l1),
            move |this, r, cx| {
                this.bench.busy.remove(&l2);
                match r {
                    Ok(l) => this.apply_loop(&b, l, since),
                    Err(e) => this.toast(Tone::Danger, "Couldn't stop the loop", Some(e.message.into()), cx),
                }
            },
        );
        cx.notify();
    }

    /// Deletes what [`BenchState::confirm_delete`] names: an entry or loop of a benchmark, or the
    /// benchmark itself (which then closes).
    pub fn confirm_bench_delete(&mut self, cx: &mut Context<Self>) {
        let Some((bench, item)) = self.bench.confirm_delete.take() else { return };
        let key = item.clone().unwrap_or_else(|| bench.clone());
        if !self.bench.busy.insert(key.clone()) {
            return;
        }
        let is_loop = item.as_ref().is_some_and(|id| {
            self.bench
                .loops
                .get(&bench)
                .and_then(|l| l.ready())
                .is_some_and(|l| l.iter().any(|x| &x.id == id))
        });
        let (b, i) = (bench.clone(), item.clone());
        self.fetch(
            cx,
            move |api| match &i {
                None => api.delete_bench(&b),
                Some(id) if is_loop => api.delete_bench_loop(&b, id),
                Some(id) => api.delete_bench_entry(&b, id),
            },
            move |this, r, cx| {
                this.bench.busy.remove(&key);
                match (r, &item) {
                    (Ok(()), None) => {
                        this.forget(&bench);
                        if let Some(list) = this.bench.list.ready_mut() {
                            list.retain(|x| x.id != bench);
                        }
                        this.bench.entries.remove(&bench);
                        this.bench.loops.remove(&bench);
                        this.bench.results.retain(|(b, _), _| *b != bench);
                        if route_bench(&this.route) == Some(bench.as_str()) {
                            this.route_pending = Some(Route::Benches);
                        }
                        this.toast(Tone::Info, "Benchmark deleted", None, cx);
                    }
                    (Ok(()), Some(id)) => {
                        this.forget(id);
                        if let Some(list) = this.bench.entries.get_mut(&bench).and_then(|l| l.ready_mut()) {
                            list.retain(|x| &x.id != id);
                        }
                        if let Some(list) = this.bench.loops.get_mut(&bench).and_then(|l| l.ready_mut()) {
                            list.retain(|x| &x.id != id);
                        }
                        this.bench.results.remove(&(bench.clone(), id.clone()));
                        if matches!(&this.route, Route::BenchEntry(b, e) if *b == bench && e == id) {
                            this.route_pending = Some(Route::Bench(bench.clone()));
                        }
                        this.refresh_benches(cx);
                    }
                    (Err(e), _) => this.toast(Tone::Danger, "Couldn't delete it", Some(e.message.into()), cx),
                }
            },
        );
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, status: BenchEntryStatus) -> BenchEntry {
        BenchEntry {
            id: id.into(),
            bench_id: "bn_1".into(),
            name: id.into(),
            contestant: BenchContestant::External,
            status,
            created: "2026-11-10T00:00:00.000Z".into(),
            finished: None,
            progress: BenchProgress::default(),
            images: 10,
            conditions: true,
            scored: true,
            fingerprint: "f".into(),
            missing_classes: vec![],
            score: None,
            map50_95: None,
            ci95: None,
            conditions_run: 0,
            speed: None,
            usage: None,
            error: None,
            hint: None,
        }
    }

    #[test]
    fn previews_use_the_middle_planned_severity_and_their_own_cache_slot() {
        assert_eq!(preview_severity(&[5, 1, 3]), 3);
        assert_eq!(preview_severity(&[2, 4]), 4);
        assert_eq!(preview_severity(&[]), 3);
        let plain = bench_img_key("bn_1", "img_7", None);
        let hazy = bench_img_key("bn_1", "img_7", Some(("haze", 3)));
        assert_eq!(plain, ("bench:bn_1".to_string(), "img_7".to_string()));
        assert_ne!(plain, hazy);
        assert_ne!(hazy, bench_img_key("bn_1", "img_7", Some(("haze", 4))));
    }

    #[test]
    fn sources_name_and_count_what_a_benchmark_would_copy() {
        let ds = SourcePick::Dataset {
            id: "ds_1".into(),
            name: "pond".into(),
            splits: Splits { train: 800, val: 200, test: 0 },
        };
        assert_eq!(ds.suggested_name(Split::Val), "pond val");
        assert_eq!(ds.images(Split::Train), 800);
        assert_eq!(ds.images(Split::Test), 0);
        assert_eq!(
            ds.source(Split::Test),
            BenchSource::Dataset { dataset_id: "ds_1".into(), split: Some(Split::Test) }
        );
        // A labeling job has no splits: its labeled images, whatever split the form shows.
        let job = SourcePick::Job { id: "lb_1".into(), name: "rejects".into(), labeled: 120 };
        assert_eq!(job.suggested_name(Split::Val), "rejects");
        assert_eq!(job.images(Split::Test), 120);
        assert_eq!(job.source(Split::Val), BenchSource::LabelJob { job_id: "lb_1".into() });
    }

    /// `merge_snapshot` for entries, as the leaderboard's reload calls it.
    fn merge(
        current: &[BenchEntry],
        fresh: Vec<BenchEntry>,
        since: u64,
        touched: &[(&str, u64)],
        removed: &[(&str, u64)],
    ) -> (Vec<BenchEntry>, bool) {
        let map = |v: &[(&str, u64)]| v.iter().map(|(k, s)| (k.to_string(), *s)).collect::<HashMap<_, _>>();
        merge_snapshot(
            Some(current),
            fresh,
            since,
            &map(touched),
            &map(removed),
            |e: &BenchEntry| e.id.as_str(),
            |e| !e.status.is_active(),
            upsert_entry,
        )
    }

    #[test]
    fn an_older_snapshot_never_undoes_what_the_stream_said() {
        use BenchEntryStatus::*;
        // Asked at event 3; the stream then said `b` finished (event 4). The snapshot still has it
        // running: the end stays, and the order is reloaded.
        let current = vec![entry("a", Done), entry("b", Done)];
        let (list, again) = merge(&current, vec![entry("a", Done), entry("b", Running)], 3, &[("b", 4)], &[]);
        assert_eq!(list[1].status, Done);
        assert!(again);
        // Progress the stream reported later is kept too, without another reload (same order).
        let mut ahead = entry("c", Running);
        ahead.progress.done = 9;
        let (list, again) = merge(&[ahead], vec![entry("c", Running)], 3, &[("c", 5)], &[]);
        assert_eq!((list[0].progress.done, again), (9, false));
        // A snapshot that has an end the stream hasn't sent yet wins: ends are final.
        let (list, _) = merge(&[entry("d", Running)], vec![entry("d", Failed)], 3, &[("d", 4)], &[]);
        assert_eq!(list[0].status, Failed);
        // Removed after the request: it stays gone. Added by the stream after it: it stays.
        let (list, _) = merge(
            &[entry("e", Queued)],
            vec![entry("a", Done), entry("gone", Done)],
            3,
            &[("e", 6), ("gone", 5)],
            &[("gone", 5)],
        );
        let ids: Vec<&str> = list.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["a", "e"]);
        // Nothing said since the request: the snapshot is the list.
        let (list, again) = merge(&[entry("x", Running)], vec![entry("y", Done)], 9, &[("x", 4)], &[]);
        assert_eq!((list.len(), list[0].id.as_str(), again), (1, "y", false));
    }

    #[test]
    fn an_ended_entry_or_loop_never_runs_again() {
        let mut list = vec![entry("a", BenchEntryStatus::Done)];
        upsert_entry(&mut list, entry("a", BenchEntryStatus::Running));
        assert_eq!(list[0].status, BenchEntryStatus::Done);
    }

    #[test]
    fn classes_match_as_the_server_matches_them() {
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let bench = names(&["Person", "car", "dog"]);
        // A contestant needs one class in common (by name, without case or edge spaces).
        assert!(model_fits(&names(&["cat", " person "]), &bench));
        assert!(!model_fits(&names(&["circle", "square"]), &bench));
        // A teacher needs all of them, in the same order.
        assert!(model_teaches(&names(&["person", "Car", "dog"]), &bench));
        assert!(!model_teaches(&names(&["car", "person", "dog"]), &bench));
        assert!(!model_teaches(&names(&["person", "car"]), &bench));
    }

    #[test]
    fn new_entries_go_first_among_the_unfinished() {
        use BenchEntryStatus::*;
        let mut list = vec![entry("a", Done), entry("b", Done), entry("c", Failed)];
        upsert_entry(&mut list, entry("d", Queued));
        let ids: Vec<_> = list.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "d", "c"]);
        // An update replaces in place, whatever its status now.
        upsert_entry(&mut list, entry("d", Running));
        assert_eq!(list[2].status, Running);
        assert_eq!(list.len(), 4);
    }
}
