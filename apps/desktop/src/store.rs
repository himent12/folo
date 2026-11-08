//! Client-side state: what has been fetched from the backend, live run data, and decoded images.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

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
}

/// A run's full metric history, extended live by SSE events.
#[derive(Default)]
pub struct RunData {
    pub metrics: Metrics,
    pub spe: u32,
    derived: RefCell<Option<Derived>>,
}

impl RunData {
    pub fn new(metrics: Metrics, spe: u32) -> Self {
        Self { metrics, spe: spe.max(1), derived: RefCell::new(None) }
    }

    pub fn push_step(&mut self, s: Step) {
        if self.metrics.steps.last().is_none_or(|l| s.step > l.step) {
            self.metrics.steps.push(s);
            self.derived.replace(None);
        }
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
    pub datasets: Load<Vec<Dataset>>,
    pub dataset: HashMap<String, Load<Dataset>>,
    /// All image metadata of a dataset (cheap: boxes only), for filtering and the grid.
    pub images: HashMap<String, Load<Vec<ImageMeta>>>,
    pub models: Load<Vec<Model>>,
    pub evals: HashMap<String, Load<ModelEval>>,
    pub runs: Load<Vec<Run>>,
    pub run_data: HashMap<String, Load<RunData>>,
    pub samples: HashMap<String, Load<Samples>>,
    pub logs: HashMap<String, Load<Vec<LogLine>>>,
    /// Live events that arrived while a run's history was loading, merged in when it lands.
    pending: HashMap<String, Vec<RunEvent>>,
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
            datasets: Load::Idle,
            dataset: HashMap::new(),
            images: HashMap::new(),
            models: Load::Idle,
            evals: HashMap::new(),
            runs: Load::Idle,
            run_data: HashMap::new(),
            samples: HashMap::new(),
            logs: HashMap::new(),
            pending: HashMap::new(),
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
            for ev in self.pending.get(run_id).into_iter().flatten() {
                match ev {
                    RunEvent::Step(s) => d.push_step(s.clone()),
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
        self.prune_pending(run_id);
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
        let spe = self.run(run_id).map_or(1, |r| r.steps_per_epoch.max(1));
        match ev {
            RunEvent::Step(s) => {
                if let Some(runs) = self.runs.ready_mut()
                    && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
                {
                    r.step = s.step;
                    r.epoch = s.epoch.min(r.epochs);
                    r.eta_s = s.eta_s;
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
                    r.best = Some(Best { epoch: e.epoch, map50_95: e.val.map50_95 });
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
            RunEvent::Log(l) => {
                if let Some(Load::Ready(lines)) = self.logs.get_mut(run_id) {
                    lines.push(l);
                    if lines.len() > 2000 {
                        lines.drain(..lines.len() - 2000);
                    }
                }
            }
            RunEvent::Status(st) => {
                if let Some(runs) = self.runs.ready_mut()
                    && let Some(r) = runs.iter_mut().find(|r| r.id == run_id)
                {
                    r.status = st.status;
                    if st.error.is_some() {
                        r.error = st.error;
                    }
                    if !st.status.is_active() {
                        r.eta_s = None;
                    }
                }
            }
        }
        let _ = spe;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::mock::MockBackend;

    #[test]
    fn live_events_extend_history() {
        let api = MockBackend::new(4200, true);
        let mut store = Store::new(Arc::new(MockBackend::new(4200, true)));
        store.runs = api.runs().into();
        let id = "run_a19f";
        store.run_data.insert(id.into(), Load::Ready(RunData::new(api.metrics(id).unwrap(), 100)));
        let before = store.run_data[id].ready().unwrap().derived().loss[3].len();
        let mut next = api.metrics(id).unwrap().steps.last().unwrap().clone();
        next.step += 1;
        store.apply(id, RunEvent::Step(next));
        let d = store.run_data[id].ready().unwrap().derived();
        assert_eq!(d.loss[3].len(), before + 1);
        assert_eq!(store.run(id).unwrap().step, 4201);
        // Stale or duplicate steps are ignored.
        let mut stale = api.metrics(id).unwrap().steps[10].clone();
        stale.step = 5;
        store.apply(id, RunEvent::Step(stale));
        assert_eq!(store.run_data[id].ready().unwrap().derived().loss[3].len(), before + 1);
        store.apply(id, RunEvent::Status(StatusEvent { status: RunStatus::Completed, error: None }));
        assert_eq!(store.run(id).unwrap().status, RunStatus::Completed);
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
        let mut late = history.steps.last().unwrap().clone();
        late.step += 1;
        store.apply(id, RunEvent::Step(late.clone()));
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
        store2.apply(id, RunEvent::Step(history.steps[100].clone()));
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
