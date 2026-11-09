//! Client-side state: what has been fetched from the backend, live run data, and decoded images.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[allow(unused_imports)]
use crate::api::{Label, RunExt, SplitsExt};
use gpui::RenderImage;

use crate::api::*;

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

pub type Pts = Arc<Vec<(f32, f32)>>;

/// Chart-ready series derived from a run's metric history. x is fractional epochs for steps and
/// the epoch number for validation results.
#[derive(Clone, Default)]
pub struct Derived {
    pub loss: [Pts; 4],
    pub lr: Pts,
    pub map50: Pts,
    pub map50_95: Pts,
    pub precision: Pts,
    pub recall: Pts,
    /// Highest learning rate in the history (for the LR chart's scale).
    pub lr_max: f32,
    /// Highest value of each loss part (box, cls, dfl, total), for the loss axis.
    pub loss_max: [f32; 4],
    /// Smallest positive, finite value per loss series (infinite when there is none): the floor of
    /// the log-scale loss axis.
    pub loss_min: [f32; 4],
}

/// A run's full metric history, extended live by SSE events.
pub struct RunData {
    pub metrics: RunMetrics,
    pub spe: u64,
    derived: RefCell<Option<Derived>>,
}

impl RunData {
    pub fn new(metrics: RunMetrics, spe: u64) -> Self {
        Self { metrics, spe: spe.max(1), derived: RefCell::new(None) }
    }

    /// The run's steps per epoch became known (or changed): step positions are recomputed.
    pub fn set_spe(&mut self, spe: u64) {
        if spe > 0 && spe != self.spe {
            self.spe = spe;
            self.derived.replace(None);
        }
    }

    pub fn push_step(&mut self, s: Step) {
        if self.metrics.steps.last().is_none_or(|l| s.step > l.step) {
            self.metrics.steps.push(s);
            self.derived.replace(None);
        }
    }

    /// A resumed run restarts after its last finished (validated) epoch, so steps past that epoch
    /// may be the interrupted attempt's: keep history only up to the boundary; live steps refill
    /// the rest. (The server's history may still list the old tail until the new one is flushed.)
    pub fn cut_to_last_epoch(&mut self) {
        let last = self.metrics.epochs.last().map_or(0, |e| e.epoch);
        self.metrics.steps.retain(|s| s.epoch <= last);
        self.derived.replace(None);
    }

    pub fn push_epoch(&mut self, e: Epoch) {
        if self.metrics.epochs.last().is_none_or(|l| e.epoch > l.epoch) {
            self.metrics.epochs.push(e);
            self.derived.replace(None);
        }
    }

    pub fn derived(&self) -> Derived {
        if let Some(d) = self.derived.borrow().as_ref() {
            return d.clone();
        }
        let spe = self.spe as f32;
        let steps = &self.metrics.steps;
        let pts = |f: &dyn Fn(&Step) -> f32| -> Pts {
            Arc::new(steps.iter().map(|s| (s.step as f32 / spe, f(s))).collect())
        };
        let eps = &self.metrics.epochs;
        let epts = |f: &dyn Fn(&Epoch) -> f32| -> Pts {
            Arc::new(eps.iter().map(|e| (e.epoch as f32, f(e))).collect())
        };
        let d = Derived {
            loss: [pts(&|s| s.loss.box_), pts(&|s| s.loss.cls), pts(&|s| s.loss.dfl), pts(&|s| s.loss.total)],
            lr: pts(&|s| s.lr),
            map50: epts(&|e| e.val.map50),
            map50_95: epts(&|e| e.val.map50_95),
            precision: epts(&|e| e.val.precision),
            recall: epts(&|e| e.val.recall),
            lr_max: steps.iter().map(|s| s.lr).fold(0f32, f32::max),
            loss_max: [
                steps.iter().map(|s| s.loss.box_).fold(0f32, f32::max),
                steps.iter().map(|s| s.loss.cls).fold(0f32, f32::max),
                steps.iter().map(|s| s.loss.dfl).fold(0f32, f32::max),
                steps.iter().map(|s| s.loss.total).fold(0f32, f32::max),
            ],
            loss_min: {
                let min = |f: &dyn Fn(&Step) -> f32| {
                    steps.iter().map(f).filter(|v| v.is_finite() && *v > 0.).fold(f32::INFINITY, f32::min)
                };
                [min(&|s| s.loss.box_), min(&|s| s.loss.cls), min(&|s| s.loss.dfl), min(&|s| s.loss.total)]
            },
        };
        self.derived.replace(Some(d.clone()));
        d
    }

    pub fn last_step(&self) -> Option<&Step> {
        self.metrics.steps.last()
    }

    pub fn last_epoch(&self) -> Option<&Epoch> {
        self.metrics.epochs.last()
    }

    /// Recent throughput samples (imgs/s), newest last.
    pub fn recent_ips(&self, n: usize) -> Vec<f32> {
        let s = &self.metrics.steps;
        let start = s.len().saturating_sub(n);
        s[start..].iter().map(|s| s.imgs_per_s).collect()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ImgKind {
    Thumb,
    Full,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ImgKey {
    pub dataset: String,
    pub image: String,
    pub kind: ImgKind,
}

pub enum ImgSlot {
    Loading,
    Ready(Arc<RenderImage>),
    Failed,
}

/// Decoded images keyed by dataset/image/size, evicted oldest-first past a budget.
#[derive(Default)]
pub struct ImageCache {
    pub slots: HashMap<ImgKey, ImgSlot>,
    order: VecDeque<ImgKey>,
}

impl ImageCache {
    pub const BUDGET: usize = 600;

    pub fn get(&self, k: &ImgKey) -> Option<&ImgSlot> {
        self.slots.get(k)
    }

    /// Mark `k` as loading; returns false if it is already known.
    pub fn begin(&mut self, k: &ImgKey) -> bool {
        if self.slots.contains_key(k) {
            return false;
        }
        self.slots.insert(k.clone(), ImgSlot::Loading);
        self.order.push_back(k.clone());
        true
    }

    /// Forget everything (a new data source); returns the decoded images to free on the GPU.
    pub fn clear(&mut self) -> Vec<Arc<RenderImage>> {
        self.order.clear();
        self.slots
            .drain()
            .filter_map(|(_, s)| match s {
                ImgSlot::Ready(img) => Some(img),
                _ => None,
            })
            .collect()
    }

    /// Store a result; returns images evicted to stay within budget (free them on the GPU).
    pub fn finish(&mut self, k: ImgKey, img: Option<Arc<RenderImage>>) -> Vec<Arc<RenderImage>> {
        // A request evicted while it was still loading has no slot (and no place in `order`) any
        // more: drop its result instead of resurrecting an untracked entry.
        if !matches!(self.slots.get(&k), Some(ImgSlot::Loading)) {
            return img.into_iter().collect();
        }
        self.slots.insert(k, img.map_or(ImgSlot::Failed, ImgSlot::Ready));
        let mut evicted = Vec::new();
        while self.order.len() > Self::BUDGET {
            if let Some(old) = self.order.pop_front()
                && let Some(ImgSlot::Ready(img)) = self.slots.remove(&old)
            {
                evicted.push(img);
            }
        }
        evicted
    }
}

/// Turn backend pixels into a GPUI image (BGRA frames).
pub fn to_render_image(data: ImageData) -> Option<Arc<RenderImage>> {
    let (w, h, mut px) = match data {
        ImageData::Rgba { width, height, pixels } => (width, height, pixels),
        ImageData::Encoded(bytes) => {
            let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
            let (w, h) = img.dimensions();
            (w, h, img.into_raw())
        }
    };
    for p in px.as_chunks_mut::<4>().0 {
        p.swap(0, 2);
    }
    let buf = image::RgbaImage::from_raw(w, h, px)?;
    Some(Arc::new(RenderImage::new(vec![image::Frame::new(buf)])))
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

    /// Install a run's metric history, merging live events that arrived while it loaded.
    pub fn install_run_data(&mut self, run_id: &str, data: ApiResult<RunData>) {
        let mut load: Load<RunData> = data.into();
        if let Load::Ready(d) = &mut load {
            if self.resumed.contains(run_id) {
                d.cut_to_last_epoch();
            }
            // The fetch may have started before the run's length was known; events since have it.
            let known = self.lengths.get(run_id).map_or(0, |l| l.0);
            d.set_spe(if known > 0 { known } else { self.run(run_id).map_or(0, |r| r.steps_per_epoch) });
            for ev in self.pending.get(run_id).into_iter().flatten() {
                match ev {
                    RunEvent::Step(s) => d.push_step(*s),
                    RunEvent::Epoch(e) => d.push_epoch(e.clone()),
                    _ => {}
                }
            }
        }
        self.run_data.insert(run_id.to_string(), load);
        self.prune_pending(run_id);
    }

    /// Install a run's log history, appending live lines that arrived while it loaded.
    pub fn install_logs(&mut self, run_id: &str, lines: ApiResult<Vec<LogLine>>) {
        let mut load: Load<Vec<LogLine>> = lines.into();
        if let Load::Ready(lines) = &mut load {
            let mut seen: std::collections::HashSet<(String, String)> =
                lines.iter().map(|l| (l.t.clone(), l.message.clone())).collect();
            for ev in self.pending.get(run_id).into_iter().flatten() {
                if let RunEvent::Log(l) = ev
                    && seen.insert((l.t.clone(), l.message.clone()))
                {
                    lines.push(l.clone());
                }
            }
            // RFC 3339 UTC timestamps sort lexically; stable, so equal times keep arrival order.
            lines.sort_by(|a, b| a.t.cmp(&b.t));
        }
        self.logs.insert(run_id.to_string(), load);
        self.log_rev += 1;
        self.prune_pending(run_id);
    }

    /// A run was resumed: drop its history and buffered events; the next history load and the
    /// first new step rebuild it.
    pub fn reset_for_resume(&mut self, run_id: &str) {
        self.run_data.remove(run_id);
        self.logs.remove(run_id);
        self.samples.remove(run_id);
        self.pending.remove(run_id);
        self.resumed.insert(run_id.to_string());
    }

    /// Forget live events buffered for histories still loading (a new data source: ids may repeat).
    pub fn clear_pending(&mut self) {
        self.pending.clear();
        self.resumed.clear();
        self.lengths.clear();
    }

    /// After a run list lands: lengths the stream reported fill in snapshot zeros, and loaded
    /// histories adopt their run's length.
    pub fn sync_run_list(&mut self) {
        self.sync_resumed();
        let ids: Vec<String> =
            self.runs.ready().map_or(Vec::new(), |r| r.iter().map(|r| r.id.clone()).collect());
        for id in ids {
            self.set_length(&id, 0, 0);
        }
    }

    /// The run list is authoritative: a resumed run it reports finished needs no more cutting
    /// (its terminal status event may never have arrived).
    pub fn sync_resumed(&mut self) {
        let Some(runs) = self.runs.ready() else { return };
        let active: std::collections::HashSet<&str> =
            runs.iter().filter(|r| r.status.is_active()).map(|r| r.id.as_str()).collect();
        self.resumed.retain(|id| active.contains(id.as_str()));
    }

    fn prune_pending(&mut self, run_id: &str) {
        let waiting = |l: Option<bool>| l.unwrap_or(false);
        let data_loading = waiting(self.run_data.get(run_id).map(|l| matches!(l, Load::Loading)));
        let logs_loading = waiting(self.logs.get(run_id).map(|l| matches!(l, Load::Loading)));
        if !data_loading && !logs_loading {
            self.pending.remove(run_id);
        }
    }

    /// Apply one live event to the run list and its metric history.
    pub fn apply(&mut self, run_id: &str, ev: RunEvent) {
        let loading = matches!(self.run_data.get(run_id), Some(Load::Loading))
            || matches!(self.logs.get(run_id), Some(Load::Loading));
        if loading && matches!(ev, RunEvent::Step(_) | RunEvent::Epoch(_) | RunEvent::Log(_)) {
            let buf = self.pending.entry(run_id.to_string()).or_default();
            if buf.len() < 20_000 {
                buf.push(ev.clone());
            }
        }
        // Step and status events carry the run's length (0 from older servers and before the
        // trainer is built), so a snapshot taken before it was known catches up from the stream.
        let length = match &ev {
            RunEvent::Step(s) => (s.steps_per_epoch, s.epochs),
            RunEvent::Status(s) => (s.steps_per_epoch, s.epochs),
            _ => (0, 0),
        };
        self.set_length(run_id, length.0, length.1);
        match ev {
            RunEvent::Step(s) => {
                if let Some(runs) = self.runs.ready_mut()
                    && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
                {
                    r.step = s.step;
                    r.epoch = s.epoch.min(r.epochs);
                    r.eta_s = Some(s.eta_s);
                }
                if let Some(Load::Ready(d)) = self.run_data.get_mut(run_id) {
                    d.push_step(s);
                }
            }
            RunEvent::Epoch(e) => {
                if let Some(runs) = self.runs.ready_mut()
                    && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
                    && r.best.as_ref().is_none_or(|b| e.val.map50_95 > b.map50_95)
                {
                    r.best = Some(BestEpoch { epoch: e.epoch, map50_95: e.val.map50_95 });
                }
                if let Some(Load::Ready(d)) = self.run_data.get_mut(run_id) {
                    d.push_epoch(e);
                }
                // New validation predictions exist; refetch samples lazily.
                if let Some(s) = self.samples.get_mut(run_id)
                    && matches!(s, Load::Ready(_))
                {
                    *s = Load::Idle;
                }
            }
            RunEvent::Queue(q) => {
                if let Some(runs) = self.runs.ready_mut()
                    && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
                {
                    r.queue_position = Some(q.position);
                }
            }
            // State, not a change: the latest replaces the run's (phase, transfer, cost).
            RunEvent::Cloud(c) => {
                if let Some(runs) = self.runs.ready_mut()
                    && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
                {
                    r.cloud = Some(*c);
                }
            }
            RunEvent::Log(l) => {
                if let Some(Load::Ready(lines)) = self.logs.get_mut(run_id) {
                    lines.push(l);
                    if lines.len() > 2000 {
                        lines.drain(..lines.len() - 2000);
                    }
                    self.log_rev += 1;
                }
            }
            RunEvent::Status(st) => {
                if let Some(runs) = self.runs.ready_mut()
                    && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
                {
                    r.status = st.status;
                    // Only a waiting run has a place in the queue (`queue` events set it).
                    if st.status != RunStatus::Queued {
                        r.queue_position = None;
                    }
                    if st.error.is_some() {
                        r.error = st.error;
                    }
                    // The code drives the failure's recovery (not_supported → no Resume, AMP off), so
                    // a live failure needn't wait for the run list's refresh.
                    if st.error_code.is_some() {
                        r.error_code = st.error_code;
                    }
                    // Active again (resumed, requeued): an earlier failure's message and code no
                    // longer describe it.
                    if st.status.is_active() {
                        r.error = None;
                        r.error_code = None;
                    }
                    if !st.status.is_active() {
                        r.eta_s = None;
                    }
                }
                // Finished: the server's history is complete again.
                if !st.status.is_active() {
                    self.resumed.remove(run_id);
                }
            }
        }
    }

    /// Adopt a run length from an event; zeros mean unknown and change nothing.
    /// Zeros mean unknown and change nothing; the latest known length is applied to the run and
    /// its loaded history (`set_length(id, 0, 0)` just re-applies it).
    fn set_length(&mut self, run_id: &str, spe: u64, epochs: u32) {
        if spe > 0 || epochs > 0 {
            let l = self.lengths.entry(run_id.to_string()).or_default();
            if spe > 0 {
                l.0 = spe;
            }
            if epochs > 0 {
                l.1 = epochs;
            }
        }
        let (spe, epochs) = self.lengths.get(run_id).copied().unwrap_or_default();
        let mut run_spe = 0;
        if let Some(runs) = self.runs.ready_mut()
            && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
        {
            if spe > 0 {
                r.steps_per_epoch = spe;
            }
            if epochs > 0 {
                r.epochs = epochs;
            }
            run_spe = r.steps_per_epoch;
        }
        if let Some(Load::Ready(d)) = self.run_data.get_mut(run_id) {
            d.set_spe(run_spe);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::mock::MockBackend;

    #[test]
    fn image_cache_clear_forgets_slots_and_returns_images() {
        let mut c = ImageCache::default();
        let k = |i: &str| ImgKey { dataset: "ds".into(), image: i.into(), kind: ImgKind::Thumb };
        let img = Arc::new(RenderImage::new(vec![image::Frame::new(image::RgbaImage::new(1, 1))]));
        assert!(c.begin(&k("a")));
        c.finish(k("a"), Some(img));
        assert!(c.begin(&k("b")));
        c.finish(k("b"), None); // a failure
        assert!(c.begin(&k("c"))); // still loading
        let freed = c.clear();
        assert_eq!(freed.len(), 1);
        // Nothing is remembered: every key can be fetched again, failures included.
        assert!(c.get(&k("a")).is_none() && c.get(&k("b")).is_none());
        assert!(c.begin(&k("b")));
        // A request that was in flight across the clear lands nowhere.
        let late = Arc::new(RenderImage::new(vec![image::Frame::new(image::RgbaImage::new(1, 1))]));
        assert_eq!(c.finish(k("c"), Some(late)).len(), 1);
    }

    #[test]
    fn live_events_extend_history() {
        let api = MockBackend::new(4200, true);
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        store.runs = api.runs().into();
        let id = "run_a19f";
        store.run_data.insert(id.into(), Load::Ready(RunData::new(api.metrics(id).unwrap(), 100)));
        let before = store.run_data[id].ready().unwrap().derived().loss[3].len();
        let mut next = *api.metrics(id).unwrap().steps.last().unwrap();
        next.step += 1;
        store.apply(id, RunEvent::Step(next));
        let d = store.run_data[id].ready().unwrap().derived();
        assert_eq!(d.loss[3].len(), before + 1);
        assert_eq!(store.run(id).unwrap().step, 4201);
        // Stale or duplicate steps are ignored.
        let mut stale = api.metrics(id).unwrap().steps[10];
        stale.step = 5;
        store.apply(id, RunEvent::Step(stale));
        assert_eq!(store.run_data[id].ready().unwrap().derived().loss[3].len(), before + 1);
        store.apply(id, RunEvent::Status(StatusEvent::new(RunStatus::Completed, None)));
        assert_eq!(store.run(id).unwrap().status, RunStatus::Completed);
    }

    #[test]
    fn resumed_history_is_cut_to_the_last_finished_epoch() {
        // Two finished epochs of 10 steps, then 4 steps of epoch 3 before the stop.
        let loss = || crate::api::Loss { box_: 1., cls: 1., dfl: 1., total: 3. };
        let step = |n: u64| Step {
            epoch: ((n - 1) / 10 + 1) as u32,
            step: n,
            lr: 0.01,
            loss: loss(),
            imgs_per_s: 1.,
            eta_s: 1.,
            steps_per_epoch: 10,
            epochs: 3,
        };
        let epoch = |n: u32| Epoch {
            epoch: n,
            train_loss: loss(),
            val: crate::api::ValMetrics {
                map50: 0.5,
                map50_95: 0.3,
                precision: 0.5,
                recall: 0.5,
                per_class: vec![],
                best_f1_conf: None,
            },
            duration_s: 1.,
        };
        let stale = || RunMetrics { steps: (1..=24).map(step).collect(), epochs: vec![epoch(1), epoch(2)] };
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        let id = "run_r";
        store.reset_for_resume(id);
        // The new stream missed steps 21-22; the server still lists the old 21-24.
        store.run_data.insert(id.into(), Load::Loading);
        store.apply(id, RunEvent::Step(step(23)));
        store.install_run_data(id, Ok(RunData::new(stale(), 10)));
        let d = store.run_data[id].ready().unwrap();
        assert_eq!(d.metrics.steps.iter().map(|s| s.step).collect::<Vec<_>>().last(), Some(&23));
        assert!(!d.metrics.steps.iter().any(|s| s.step == 24), "old tail kept");
        assert_eq!(d.metrics.steps.len(), 21);
        // A reconnect mid-epoch reloads the stale history again: still cut.
        store.run_data.insert(id.into(), Load::Loading);
        store.install_run_data(id, Ok(RunData::new(stale(), 10)));
        assert_eq!(store.run_data[id].ready().unwrap().metrics.steps.len(), 20);
        // Once the run ends, histories are taken as they are.
        store.apply(id, RunEvent::Status(StatusEvent::new(RunStatus::Completed, None)));
        store.run_data.insert(id.into(), Load::Loading);
        store.install_run_data(id, Ok(RunData::new(stale(), 10)));
        assert_eq!(store.run_data[id].ready().unwrap().metrics.steps.len(), 24);
    }

    #[test]
    fn the_run_length_comes_from_the_stream() {
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        let mut runs = store.api.runs().unwrap();
        // A real run's snapshot before its trainer is built: the length isn't known yet.
        runs[0].steps_per_epoch = 0;
        let id = runs[0].id.clone();
        store.runs = Load::Ready(runs);
        let empty = || RunMetrics { steps: vec![], epochs: vec![] };
        store.install_run_data(&id, Ok(RunData::new(empty(), 0)));
        assert_eq!(store.run_data[&id].ready().unwrap().spe, 1);
        // Older servers send zeros: nothing changes.
        store.apply(&id, RunEvent::Status(StatusEvent::new(RunStatus::Running, None)));
        assert_eq!(store.run(&id).unwrap().steps_per_epoch, 0);
        store.apply(&id, RunEvent::Status(StatusEvent::new(RunStatus::Running, None).with_length(7, 25)));
        assert_eq!(store.run(&id).unwrap().steps_per_epoch, 25);
        assert_eq!(store.run(&id).unwrap().epochs, 7);
        assert_eq!(store.run_data[&id].ready().unwrap().spe, 25);
        // A history fetched with the old length is corrected when it lands.
        store.install_run_data(&id, Ok(RunData::new(empty(), 0)));
        assert_eq!(store.run_data[&id].ready().unwrap().spe, 25);
    }

    #[test]
    fn a_stale_run_list_cannot_undo_the_streamed_length() {
        let loss = || crate::api::Loss { box_: 1., cls: 1., dfl: 1., total: 3. };
        let step = |n: u64, spe: u64| Step {
            epoch: 1,
            step: n,
            lr: 0.01,
            loss: loss(),
            imgs_per_s: 1.,
            eta_s: 1.,
            steps_per_epoch: spe,
            epochs: 4,
        };
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        let mut runs = store.api.runs().unwrap();
        runs[0].steps_per_epoch = 0;
        let id = runs[0].id.clone();
        let stale = runs.clone();
        store.runs = Load::Ready(runs);
        // History loading; a length-bearing step and the terminal status arrive meanwhile.
        store.run_data.insert(id.clone(), Load::Loading);
        store.apply(&id, RunEvent::Step(step(5, 25)));
        store.apply(&id, RunEvent::Status(StatusEvent::new(RunStatus::Completed, None).with_length(4, 25)));
        // An older run-list response lands, still saying 0, then the history.
        store.runs = Load::Ready(stale);
        store.sync_run_list();
        assert_eq!(store.run(&id).unwrap().steps_per_epoch, 25);
        let hist = RunMetrics { steps: (1..=4).map(|n| step(n, 0)).collect(), epochs: vec![] };
        store.install_run_data(&id, Ok(RunData::new(hist, 0)));
        let d = store.run_data[&id].ready().unwrap();
        assert_eq!(d.spe, 25);
        // Steps sit at step / steps-per-epoch on the epoch axis.
        assert_eq!(d.derived().loss[3].last().map(|p| p.0), Some(5. / 25.));
    }

    #[test]
    fn a_learned_length_moves_points_already_drawn() {
        let loss = || crate::api::Loss { box_: 1., cls: 1., dfl: 1., total: 3. };
        let step = |n: u64| Step {
            epoch: 1,
            step: n,
            lr: 0.01,
            loss: loss(),
            imgs_per_s: 1.,
            eta_s: 1.,
            steps_per_epoch: 0,
            epochs: 0,
        };
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        let mut runs = store.api.runs().unwrap();
        runs[0].steps_per_epoch = 0;
        let id = runs[0].id.clone();
        store.runs = Load::Ready(runs);
        store.install_run_data(
            &id,
            Ok(RunData::new(RunMetrics { steps: (1..=10).map(step).collect(), epochs: vec![] }, 0)),
        );
        // Drawn once with the fallback length: raw step numbers.
        assert_eq!(store.run_data[&id].ready().unwrap().derived().lr.last().map(|p| p.0), Some(10.));
        store.apply(&id, RunEvent::Status(StatusEvent::new(RunStatus::Running, None).with_length(4, 5)));
        assert_eq!(store.run_data[&id].ready().unwrap().derived().lr.last().map(|p| p.0), Some(2.));
    }

    #[test]
    fn a_run_list_reporting_the_resumed_run_finished_clears_the_cut() {
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        let mut runs = store.api.runs().unwrap();
        let id = runs[0].id.clone();
        store.reset_for_resume(&id);
        // The terminal status event was missed; the next run-list refresh says it stopped.
        runs[0].status = RunStatus::Stopped;
        store.runs = Load::Ready(runs);
        store.sync_resumed();
        assert!(!store.resumed.contains(&id));
    }

    #[test]
    fn a_cloud_event_replaces_the_runs_cloud_state() {
        let mut store = Store::new(Arc::new(MockBackend::with_cloud_runs(4200, true)));
        store.runs = store.api.runs().into();
        let id = "run_c1d0";
        let mut cloud = store.run(id).and_then(|r| r.cloud.clone()).expect("a cloud run");
        assert_eq!(cloud.phase, crate::api::CloudPhase::Syncing);
        cloud.phase = crate::api::CloudPhase::Training;
        cloud.progress = None;
        cloud.cost_usd = Some(0.31);
        store.apply(id, RunEvent::Cloud(Box::new(cloud.clone())));
        assert_eq!(store.run(id).unwrap().cloud.as_ref(), Some(&cloud));
        // Another run's state stays as it was.
        assert!(store.run("run_a19f").unwrap().cloud.is_none());
    }

    #[test]
    fn derived_loss_maxima_match_the_history_and_follow_new_steps() {
        let api = MockBackend::new(4200, true);
        let m = api.metrics("run_a19f").unwrap();
        let expect = m.steps.iter().map(|s| s.loss.total).fold(0f32, f32::max);
        let mut d = RunData::new(m.clone(), 100);
        assert_eq!(d.derived().loss_max[3], expect);
        let mut big = *m.steps.last().unwrap();
        big.step += 1;
        big.loss.total = expect + 10.0;
        d.push_step(big);
        assert_eq!(d.derived().loss_max[3], expect + 10.0, "a new step invalidates the cached maxima");
    }

    #[test]
    fn buffered_events_do_not_cross_sources() {
        let api = MockBackend::new(4200, true);
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        store.runs = api.runs().into();
        let id = "run_a19f";
        store.run_data.insert(id.into(), Load::Loading);
        store.logs.insert(id.into(), Load::Loading);
        let history = api.metrics(id).unwrap();
        let mut from_a = *history.steps.last().unwrap();
        from_a.step += 1;
        store.apply(id, RunEvent::Step(from_a));
        store.apply(
            id,
            RunEvent::Log(LogLine {
                t: "2026-11-08T07:30:00Z".into(),
                level: LogLevel::Info,
                message: "a".into(),
            }),
        );
        // Switch to source B, which has a run with the same id and its own history.
        store.clear_pending();
        store.run_data.insert(id.into(), Load::Loading);
        store.logs.insert(id.into(), Load::Loading);
        store.install_run_data(id, Ok(RunData::new(history.clone(), 100)));
        store.install_logs(id, Ok(vec![]));
        let last = store.run_data[id].ready().unwrap().last_step().unwrap().step;
        assert_eq!(last, history.steps.last().unwrap().step, "source A's step leaked into B");
        assert!(store.logs[id].ready().unwrap().is_empty(), "source A's log leaked into B");
    }

    #[test]
    fn events_during_load_are_merged() {
        let api = MockBackend::new(4200, true);
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        store.runs = api.runs().into();
        let id = "run_a19f";
        store.run_data.insert(id.into(), Load::Loading);
        store.logs.insert(id.into(), Load::Loading);
        let history = api.metrics(id).unwrap();
        let mut late = *history.steps.last().unwrap();
        late.step += 1;
        store.apply(id, RunEvent::Step(late));
        let line =
            LogLine { t: "2026-11-08T07:30:00Z".into(), level: LogLevel::Info, message: "late".into() };
        store.apply(id, RunEvent::Log(line.clone()));
        // The snapshot was taken before the event: the event must not be lost.
        store.install_run_data(id, Ok(RunData::new(history.clone(), 100)));
        assert_eq!(store.run_data[id].ready().unwrap().last_step().unwrap().step, late.step);
        store.install_logs(id, Ok(vec![]));
        assert_eq!(store.logs[id].ready().unwrap(), &vec![line]);
        // Duplicates of events already in the snapshot are ignored.
        let mut store2 = Store::new(Arc::new(MockBackend::new(4200, true)));
        store2.run_data.insert(id.into(), Load::Loading);
        store2.apply(id, RunEvent::Step(history.steps[100]));
        store2.install_run_data(id, Ok(RunData::new(history.clone(), 100)));
        assert_eq!(store2.run_data[id].ready().unwrap().metrics.steps.len(), history.steps.len());
    }

    #[test]
    fn log_merge_dedupes_large_overlaps_in_order() {
        let mut store = Store::new(Arc::new(MockBackend::new(0, true)));
        let id = "run_x";
        store.logs.insert(id.into(), Load::Loading);
        let line = |i: i64| LogLine {
            t: crate::api::time::format(1_000 + i),
            level: LogLevel::Info,
            message: format!("line {i}"),
        };
        // 80 buffered lines, of which the first 60 are already in the snapshot.
        for i in 40..120 {
            store.apply(id, RunEvent::Log(line(i)));
        }
        store.install_logs(id, Ok((0..100).map(line).collect()));
        let got = store.logs[id].ready().unwrap();
        assert_eq!(got.len(), 120);
        assert!(got.windows(2).all(|w| w[0].t <= w[1].t));
    }

    #[test]
    fn late_completion_of_evicted_request_is_dropped() {
        let mut c = ImageCache::default();
        let key = |i: usize| ImgKey { dataset: "d".into(), image: i.to_string(), kind: ImgKind::Thumb };
        for i in 0..=ImageCache::BUDGET {
            assert!(c.begin(&key(i)));
        }
        // A newer request finishes first and evicts the oldest, still-loading one.
        c.finish(key(ImageCache::BUDGET), None);
        assert!(c.get(&key(0)).is_none());
        // Its late completion must not reappear outside the eviction order.
        c.finish(key(0), None);
        assert!(c.get(&key(0)).is_none());
        assert!(c.slots.len() <= ImageCache::BUDGET);
    }

    #[test]
    fn cache_evicts_oldest() {
        let mut c = ImageCache::default();
        for i in 0..ImageCache::BUDGET + 5 {
            let k = ImgKey { dataset: "d".into(), image: i.to_string(), kind: ImgKind::Thumb };
            assert!(c.begin(&k));
            c.finish(k, None);
        }
        assert_eq!(c.slots.len(), ImageCache::BUDGET);
        let first = ImgKey { dataset: "d".into(), image: "0".into(), kind: ImgKind::Thumb };
        assert!(c.get(&first).is_none());
    }

    #[test]
    fn rgba_converts_to_bgra() {
        let img =
            to_render_image(ImageData::Rgba { width: 1, height: 1, pixels: vec![10, 20, 30, 255] }).unwrap();
        assert_eq!(img.as_bytes(0).unwrap(), &[30, 20, 10, 255]);
    }
}
