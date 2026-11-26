//! Live run data: a run's metric history and the chart series derived from it, and how the store
//! installs histories and logs and applies the run's SSE events to them.

use std::cell::RefCell;
use std::sync::Arc;

use crate::api::*;

use super::{Load, Store};

pub type Pts = Arc<Vec<(f32, f32)>>;

/// Maximum log lines retained in memory per run before older head lines are dropped.
pub const LOG_BUFFER_CAP: usize = 2000;

/// A run's loaded log lines plus how many earlier lines were omitted (server `limit` or client cap).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogHistory {
    pub lines: Vec<LogLine>,
    /// Earlier lines omitted before `lines[0]` (from the server's `total > lines.len()` or client head trimming).
    pub dropped: u64,
    /// Latest timestamp from the installed snapshot; used to reconcile overlapping SSE lines that
    /// arrive in the first drain after installation without suppressing genuinely repeated lines.
    snapshot_cutoff: Option<String>,
    snapshot_overlap: std::collections::HashMap<(String, String), usize>,
}

/// Chart-ready series derived from a run's metric history. x is fractional epochs for steps and
/// the epoch number for validation results.
#[derive(Clone, Default)]
pub struct Derived {
    pub loss: [Pts; 4],
    pub lr: Pts,
    pub ips: Pts,
    pub map50: Pts,
    pub map50_95: Pts,
    pub precision: Pts,
    pub recall: Pts,
    /// Highest learning rate in the history (for the LR chart's scale).
    pub lr_max: f32,
    /// Highest finite throughput (imgs/s) in the history (for the Throughput chart's scale).
    pub ips_max: f32,
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
            ips: Arc::new(
                steps
                    .iter()
                    .filter(|s| s.imgs_per_s.is_finite())
                    .map(|s| (s.step as f32 / spe, s.imgs_per_s))
                    .collect(),
            ),
            map50: epts(&|e| e.val.map50),
            map50_95: epts(&|e| e.val.map50_95),
            precision: epts(&|e| e.val.precision),
            recall: epts(&|e| e.val.recall),
            lr_max: steps.iter().map(|s| s.lr).filter(|v| v.is_finite()).fold(0f32, f32::max),
            ips_max: steps.iter().map(|s| s.imgs_per_s).filter(|v| v.is_finite()).fold(0f32, f32::max),
            loss_max: {
                let max = |f: &dyn Fn(&Step) -> f32| {
                    steps.iter().map(f).filter(|v| v.is_finite()).fold(0f32, f32::max)
                };
                [max(&|s| s.loss.box_), max(&|s| s.loss.cls), max(&|s| s.loss.dfl), max(&|s| s.loss.total)]
            },
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

impl Store {
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
    pub fn install_logs(&mut self, run_id: &str, logs: ApiResult<RunLogs>) {
        let mut load: Load<LogHistory> = logs
            .map(|l| {
                let dropped = l.total.saturating_sub(l.lines.len() as u64);
                let snapshot_cutoff = l.lines.iter().map(|x| x.t.as_str()).max().map(str::to_string);
                let mut snapshot_overlap: std::collections::HashMap<(String, String), usize> =
                    std::collections::HashMap::new();
                for x in &l.lines {
                    *snapshot_overlap.entry((x.t.clone(), x.message.clone())).or_insert(0) += 1;
                }
                LogHistory { lines: l.lines, dropped, snapshot_cutoff, snapshot_overlap }
            })
            .into();
        if let Load::Ready(hist) = &mut load {
            for ev in self.pending.get(run_id).into_iter().flatten() {
                if let RunEvent::Log(l) = ev {
                    let key = (l.t.clone(), l.message.clone());
                    if let Some(count) = hist.snapshot_overlap.get_mut(&key)
                        && *count > 0
                    {
                        *count -= 1;
                        if *count == 0 {
                            hist.snapshot_overlap.remove(&key);
                        }
                    } else {
                        hist.lines.push(l.clone());
                    }
                }
            }
            // RFC 3339 UTC timestamps sort lexically; stable, so equal times keep arrival order.
            hist.lines.sort_by(|a, b| a.t.cmp(&b.t));
            if hist.lines.len() > LOG_BUFFER_CAP {
                let cut = hist.lines.len() - LOG_BUFFER_CAP;
                hist.lines.drain(..cut);
                hist.dropped = hist.dropped.saturating_add(cut as u64);
            }
            if hist.snapshot_overlap.is_empty() {
                hist.snapshot_cutoff = None;
            }
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
        self.samples_stale.remove(run_id);
        self.samples_refreshing.remove(run_id);
        self.samples_refresh_error.remove(run_id);
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
                {
                    r.epoch = r.epoch.max(e.epoch);
                    if r.best.as_ref().is_none_or(|b| e.val.map50_95 > b.map50_95) {
                        r.best = Some(BestEpoch { epoch: e.epoch, map50_95: e.val.map50_95 });
                    }
                }
                if let Some(Load::Ready(d)) = self.run_data.get_mut(run_id) {
                    d.push_epoch(e);
                }
                // New validation predictions exist: keep ready tiles visible while refreshing in
                // place, and clear any earlier failure (e.g. a premature epoch-0 error).
                self.samples_refresh_error.remove(run_id);
                match self.samples.get_mut(run_id) {
                    Some(Load::Ready(_)) | Some(Load::Loading) => {
                        self.samples_stale.insert(run_id.to_string());
                    }
                    Some(s @ Load::Failed(_)) => {
                        *s = Load::Idle;
                        self.samples_stale.remove(run_id);
                    }
                    _ => {}
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
                if let Some(Load::Ready(hist)) = self.logs.get_mut(run_id) {
                    if let Some(cutoff) = &hist.snapshot_cutoff {
                        if l.t.as_str() > cutoff.as_str() {
                            hist.snapshot_cutoff = None;
                            hist.snapshot_overlap.clear();
                        } else {
                            let key = (l.t.clone(), l.message.clone());
                            if let Some(count) = hist.snapshot_overlap.get_mut(&key)
                                && *count > 0
                            {
                                *count -= 1;
                                if *count == 0 {
                                    hist.snapshot_overlap.remove(&key);
                                    if hist.snapshot_overlap.is_empty() {
                                        hist.snapshot_cutoff = None;
                                    }
                                }
                                return;
                            }
                        }
                    }
                    hist.lines.push(l);
                    if hist.lines.len() > LOG_BUFFER_CAP {
                        let cut = hist.lines.len() - LOG_BUFFER_CAP;
                        hist.lines.drain(..cut);
                        hist.dropped = hist.dropped.saturating_add(cut as u64);
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
                    match self.samples.get_mut(run_id) {
                        Some(s @ Load::Failed(_)) => {
                            *s = Load::Idle;
                            self.samples_stale.remove(run_id);
                            self.samples_refresh_error.remove(run_id);
                        }
                        Some(Load::Loading) => {
                            self.samples_stale.insert(run_id.to_string());
                        }
                        Some(Load::Ready(_))
                            if self.samples_refresh_error.remove(run_id).is_some()
                                || self.samples_refreshing.contains(run_id) =>
                        {
                            self.samples_stale.insert(run_id.to_string());
                        }
                        _ => {}
                    }
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
        // A real run's snapshot before its trainer is built: the length isn't known yet.
        runs[0].steps_per_epoch = 0;
        let id = runs[0].id.clone();
        store.runs = Load::Ready(runs);
        let empty = || RunMetrics { steps: vec![], epochs: vec![] };
        store.install_run_data(
            &id,
            Ok(RunData::new(RunMetrics { steps: (1..=10).map(step).collect(), epochs: vec![] }, 0)),
        );
        assert_eq!(store.run_data[&id].ready().unwrap().spe, 1);
        // Drawn with the fallback length: raw step numbers.
        let last_x = |store: &Store| store.run_data[&id].ready().unwrap().derived().lr.last().map(|p| p.0);
        assert_eq!(last_x(&store), Some(10.));
        // Older servers send zeros: nothing changes.
        store.apply(&id, RunEvent::Status(StatusEvent::new(RunStatus::Running, None)));
        assert_eq!(store.run(&id).unwrap().steps_per_epoch, 0);
        store.apply(&id, RunEvent::Status(StatusEvent::new(RunStatus::Running, None).with_length(7, 25)));
        assert_eq!(store.run(&id).unwrap().steps_per_epoch, 25);
        assert_eq!(store.run(&id).unwrap().epochs, 7);
        assert_eq!(store.run_data[&id].ready().unwrap().spe, 25);
        // The learned length moves the points already drawn.
        assert_eq!(last_x(&store), Some(10. / 25.));
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
        store.install_logs(id, Ok(RunLogs { lines: vec![], total: 0 }));
        let last = store.run_data[id].ready().unwrap().last_step().unwrap().step;
        assert_eq!(last, history.steps.last().unwrap().step, "source A's step leaked into B");
        assert!(store.logs[id].ready().unwrap().lines.is_empty(), "source A's log leaked into B");
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
        store.install_logs(id, Ok(RunLogs { lines: vec![], total: 0 }));
        assert_eq!(store.logs[id].ready().unwrap().lines, vec![line]);
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
        store.install_logs(id, Ok(RunLogs { lines: (0..100).map(line).collect(), total: 100 }));
        let got = &store.logs[id].ready().unwrap().lines;
        assert_eq!(got.len(), 120);
        assert_eq!(store.logs[id].ready().unwrap().dropped, 0);
        assert!(got.windows(2).all(|w| w[0].t <= w[1].t));
    }

    #[test]
    fn log_history_tracks_server_and_ring_buffer_drops() {
        let mut store = Store::new(Arc::new(MockBackend::new(0, true)));
        let id = "run_cap";
        let line = |i: i64| LogLine {
            t: crate::api::time::format(1_000 + i),
            level: LogLevel::Info,
            message: format!("line {i}"),
        };
        // Server truncated 250 earlier lines (total = 2250, returned = 2000).
        let initial: Vec<LogLine> = (0..LOG_BUFFER_CAP as i64).map(line).collect();
        store.install_logs(id, Ok(RunLogs { lines: initial, total: LOG_BUFFER_CAP as u64 + 250 }));
        assert_eq!(store.logs[id].ready().unwrap().dropped, 250);
        // Streaming 5 more pushes past LOG_BUFFER_CAP and increments dropped by 5.
        for i in 0..5 {
            store.apply(id, RunEvent::Log(line(LOG_BUFFER_CAP as i64 + i)));
        }
        let hist = store.logs[id].ready().unwrap();
        assert_eq!(hist.lines.len(), LOG_BUFFER_CAP);
        assert_eq!(hist.dropped, 255);
    }

    #[test]
    fn post_install_overlapping_sse_log_is_not_duplicated_while_repeated_lines_are_kept() {
        let mut store = Store::new(Arc::new(MockBackend::new(0, true)));
        let id = "run_overlap";
        let t1 = crate::api::time::format(1_000);
        let t2 = crate::api::time::format(1_001);
        let snap_line = LogLine { t: t1.clone(), level: LogLevel::Info, message: "epoch 1".into() };
        store.install_logs(id, Ok(RunLogs { lines: vec![snap_line.clone()], total: 1 }));
        // An SSE event that was already in the snapshot arrives in the next drain: skipped.
        store.apply(id, RunEvent::Log(snap_line.clone()));
        assert_eq!(store.logs[id].ready().unwrap().lines.len(), 1);
        // A genuinely repeated line with the same timestamp/message after overlap is consumed: kept.
        store.apply(id, RunEvent::Log(snap_line));
        assert_eq!(store.logs[id].ready().unwrap().lines.len(), 2);
        // A newer event advances past the snapshot cutoff and is kept.
        let newer = LogLine { t: t2, level: LogLevel::Info, message: "epoch 2".into() };
        store.apply(id, RunEvent::Log(newer));
        assert_eq!(store.logs[id].ready().unwrap().lines.len(), 3);
    }

    #[test]
    fn non_finite_throughput_and_loss_steps_do_not_poison_derived_series() {
        let s1 = Step {
            step: 1,
            epoch: 1,
            epochs: 5,
            steps_per_epoch: 10,
            loss: Loss { box_: 1.2, cls: 0.8, dfl: 0.5, total: 2.5 },
            lr: 0.01,
            imgs_per_s: 120.0,
            eta_s: 0.0,
        };
        let s2 = Step {
            step: 2,
            epoch: 1,
            epochs: 5,
            steps_per_epoch: 10,
            loss: Loss { box_: f32::INFINITY, cls: f32::NAN, dfl: 0.4, total: f32::INFINITY },
            lr: f32::NAN,
            imgs_per_s: f32::NAN,
            eta_s: 0.0,
        };
        let s3 = Step {
            step: 3,
            epoch: 1,
            epochs: 5,
            steps_per_epoch: 10,
            loss: Loss { box_: 0.9, cls: 0.6, dfl: 0.3, total: 1.8 },
            lr: 0.008,
            imgs_per_s: 145.0,
            eta_s: 0.0,
        };
        let data = RunData::new(RunMetrics { steps: vec![s1, s2, s3], epochs: vec![] }, 10);
        let d = data.derived();
        assert_eq!(d.ips.len(), 2, "NaN throughput step is omitted from plotted series");
        assert_eq!(d.ips_max, 145.0);
        assert_eq!(d.lr_max, 0.01);
        assert_eq!(d.loss_max, [1.2, 0.8, 0.5, 2.5], "inf/NaN step does not poison loss_max");
    }
}
