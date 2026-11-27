//! Simulated training runs: deterministic metric curves, a timeline (fixed for finished runs, wall-clock driven for
//! live ones) and an event ticker that feeds SSE subscribers.
//!
//! Runs created through the API share one training slot, like the real server's: [`reschedule`] gives each waiting
//! run the moment it starts (when the run ahead of it ends), so its clock, its queue position and its `queue`
//! events follow from the wall clock alone. The seeded looping run is outside the queue.
//!
//! Files: `curve` (metric curves by step and epoch), `clock` (the wall-clock timeline of a live run), `events` (log
//! lines, events between two snapshots and the ticker that sends them), `queue` (the shared training slot) and
//! `tests`.

mod clock;
mod curve;
mod events;
mod queue;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_api::{
    Arch, CloudInstance, LogLevel, LogLine, ResolvedTrain, Run, RunCloud, RunEvent, RunMetrics, RunStatus,
    StatusEvent, Step, TrainConfig,
};
use tokio::sync::broadcast;

use super::cloud::SimCloud;
use crate::store::{RunSseResync, RunSseStream, RunSseSubscription, SseFrame, Subscription};
use crate::util::{noise, rfc3339, unix_now};

use events::ticker;
pub use queue::{move_waiting, reschedule};

/// Seconds a stop request spends in `stopping`.
const STOP_S: f64 = 1.5;
/// Tick of the live event pump.
pub const TICK: Duration = Duration::from_millis(100);

/// Per-architecture constants: (params, GFLOPs, train imgs/s at 640 on the mock GPU, inference ms at 640).
pub fn arch_spec(arch: Arch) -> (u64, f64, f32, f64) {
    match arch {
        Arch::N => (2_620_000, 6.5, 410.0, 3.1),
        Arch::S => (9_410_000, 21.5, 265.0, 5.4),
        Arch::M => (20_050_000, 68.0, 132.0, 10.8),
        Arch::L => (25_310_000, 86.9, 88.0, 16.2),
        Arch::X => (56_870_000, 194.9, 46.0, 26.0),
    }
}

/// Rounds to 3 decimals (the confidence grid of docs/METRICS.md).
pub(super) fn r3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

/// Rounds to 4 decimals (keeps the JSON short; metrics are not more precise than that anyway).
pub(super) fn r4(v: f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

/// Deterministic metric curves of one run.
#[derive(Clone, Debug)]
pub struct Curve {
    seed: u64,
    /// Final validation mAP@0.5:0.95 the run converges to.
    pub quality: f32,
    nc: usize,
    epochs: u32,
    spe: u64,
    lr0: f32,
    ips: f32,
    batch: u32,
    close_mosaic: u32,
}

/// A point-in-time view of a run's timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snap {
    /// Status.
    pub status: RunStatus,
    /// Global steps completed.
    pub step: u64,
    /// Unix seconds.
    pub created: f64,
    /// Unix seconds.
    pub started: Option<f64>,
    /// Unix seconds.
    pub finished: Option<f64>,
    /// Seconds left while active.
    pub eta_s: Option<f64>,
    /// Incremented each time a looping run restarts.
    pub cycle: u64,
    /// Unix seconds of the stop request, if one was made in this cycle.
    pub stop_requested: Option<f64>,
    /// `(position, waiting)` while the run waits in the training queue
    /// ([`QueueEvent`](folo_api::QueueEvent)).
    pub queue: Option<(u32, u32)>,
    /// Unix seconds of the view.
    pub at: f64,
}

/// Wall-clock driven timeline of a live run.
#[derive(Debug)]
pub struct LiveClock {
    origin: Instant,
    origin_wall: f64,
    /// Seconds per epoch (compressed).
    pub epoch_s: f64,
    queue_s: f64,
    hold_s: f64,
    looping: bool,
    stop_at: Option<f64>,
    cycle: u64,
    /// The earliest start (seconds after `origin`): `queue_s` as created, or the moment of a `start` request.
    min_start: f64,
    /// Set by [`reschedule`] for runs in the training queue: the scheduled starts of the waiting runs ahead of this
    /// one, and of every waiting run.
    queue: Option<(Vec<Instant>, Vec<Instant>)>,
    /// Seconds the run stays `running` after its last epoch (a cloud run collects and releases then).
    tail_s: f64,
    /// Seconds after `origin` its cloud instance was terminated by hand: the run fails then.
    lost_at: Option<f64>,
}

/// How a run's state evolves.
#[derive(Debug)]
pub enum Timeline {
    /// A finished run frozen at `last_step`.
    Static {
        /// Unix seconds.
        created: f64,
        /// Final status.
        status: RunStatus,
        /// Steps done.
        last_step: u64,
        /// Error when failed.
        error: Option<String>,
    },
    /// A run that advances in real time.
    Live(Mutex<LiveClock>),
}

#[derive(Default)]
struct Hub {
    tx: Option<broadcast::Sender<RunEvent>>,
    sse_tx: Option<broadcast::Sender<SseFrame>>,
    ticking: bool,
}

impl Hub {
    fn receiver_count(&self) -> usize {
        self.tx.as_ref().map_or(0, broadcast::Sender::receiver_count)
            + self.sse_tx.as_ref().map_or(0, broadcast::Sender::receiver_count)
    }

    fn broadcast(&self, ev: RunEvent) {
        if let Some(sse) = &self.sse_tx
            && sse.receiver_count() > 0
        {
            let _ = sse.send(SseFrame::run(&ev));
        }
        if let Some(tx) = &self.tx
            && tx.receiver_count() > 0
        {
            let _ = tx.send(ev);
        }
    }
}

/// A simulated run.
pub struct SimRun {
    /// Run id.
    pub id: String,
    /// Config (defaults filled in).
    pub config: TrainConfig,
    /// Class names.
    pub classes: Vec<String>,
    /// Steps per epoch.
    pub spe: u64,
    /// Metric curves.
    pub curve: Curve,
    /// Device string.
    pub device: String,
    /// Model produced when completed (or fixed for finished runs).
    pub model_id: Option<String>,
    /// True for runs created via `POST /api/runs` (their model only exists once completed).
    pub user: bool,
    /// What the trainer would resolve the config to ([`SimRun::resolved_for`]); shown once the run started.
    resolved: Option<ResolvedTrain>,
    timeline: Timeline,
    hub: Mutex<Hub>,
    deleted: AtomicBool,
    /// Log lines of actions outside the timeline (queue moves), in the order they happened.
    notes: Mutex<Vec<LogLine>>,
    /// A cloud run's hand-over ([`SimRun::with_cloud`]).
    cloud: Option<SimCloud>,
    /// Opt-in GPU VRAM-budget auto-batch settings requested when the run was created.
    pub auto_batch: crate::store::AutoBatchOptions,
    /// Optional FOLO v2-Iso architecture override (`"folo2-n"`..`"folo2-x"`).
    pub arch_override: Option<String>,
}

/// Why a live (cloud) run failed.
const LOST: &str = "the instance was terminated by hand before the run ended; the weights copied so far stay \
                    (resume the run to continue from them)";

impl SimRun {
    /// A new simulated run.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        config: TrainConfig,
        classes: Vec<String>,
        spe: u64,
        curve: Curve,
        device: String,
        model_id: Option<String>,
        user: bool,
        timeline: Timeline,
    ) -> SimRun {
        SimRun {
            id,
            config,
            classes,
            spe,
            curve,
            device,
            model_id,
            user,
            resolved: None,
            timeline,
            hub: Mutex::default(),
            deleted: AtomicBool::new(false),
            notes: Mutex::default(),
            cloud: None,
            auto_batch: crate::store::AutoBatchOptions::default(),
            arch_override: None,
        }
    }

    /// Attaches opt-in auto-batch settings to this simulated run.
    pub fn with_auto_batch(mut self, auto_batch: crate::store::AutoBatchOptions) -> SimRun {
        self.auto_batch = auto_batch;
        self
    }

    /// Attaches an optional `folo2-*` architecture override to this simulated run.
    pub fn with_arch_override(mut self, arch_override: Option<String>) -> SimRun {
        self.arch_override = arch_override;
        self
    }

    /// The same run as a cloud run with this hand-over (a live one stays `running` while it collects and releases:
    /// its clock needs [`LiveClock::with_tail`]).
    pub fn with_cloud(mut self, cloud: SimCloud) -> SimRun {
        self.cloud = Some(cloud);
        self
    }

    /// A cloud run's hand-over.
    pub fn cloud(&self) -> Option<&SimCloud> {
        self.cloud.as_ref()
    }

    /// `Run.cloud` at `snap` (cloud runs only).
    pub fn cloud_at(&self, snap: &Snap) -> Option<RunCloud> {
        let c = self.cloud.as_ref()?;
        let epoch_s = match &self.timeline {
            Timeline::Static { .. } => self.curve.real_epoch_s(),
            Timeline::Live(clock) => clock.lock().unwrap().epoch_s(),
        };
        let trained = snap.started.map(|s| s + self.config.epochs as f64 * epoch_s);
        Some(c.at(snap, trained, epoch_s, self.epochs_done(snap)))
    }

    /// Its cloud instance now (cloud runs only).
    pub fn instance(&self) -> Option<CloudInstance> {
        let snap = self.snap();
        let cloud = self.cloud_at(&snap)?;
        self.cloud.as_ref().map(|c| c.instance(&self.id, &snap, &cloud))
    }

    /// Its cloud instance is terminated by hand: the run fails now unless it ended already. Returns whether it was
    /// still alive.
    pub fn lose(&self) -> bool {
        let alive = self.instance().is_some_and(|i| i.status.is_alive());
        if alive && let Timeline::Live(c) = &self.timeline {
            let mut c = c.lock().unwrap();
            c.lost_at = Some(Instant::now().saturating_duration_since(c.origin).as_secs_f64());
        }
        alive
    }

    /// Logs an action outside the timeline (a queue move): kept for `/logs` and sent to live subscribers.
    pub fn note(&self, level: LogLevel, message: String) {
        let line = LogLine { t: rfc3339(unix_now()), level, message };
        self.notes.lock().unwrap().push(line.clone());
        // Under the hub lock, as the ticker sends, so the line is ordered with the run's other events.
        let hub = self.hub.lock().unwrap();
        hub.broadcast(RunEvent::Log(line));
    }

    /// Sets `Run.resolved` to what the real trainer resolves this run's config to on `train_images` training images.
    pub fn resolved_for(mut self, train_images: u64) -> SimRun {
        let mut cfg = folo_vision::TrainConfig::from_api(&self.config).expect("mock configs are known");
        if let Some(arch) = &self.arch_override {
            cfg.arch = arch.clone();
        }
        self.resolved = Some(cfg.resolved_train(train_images as usize, self.classes.len()));
        self
    }

    /// Current state.
    pub fn snap(&self) -> Snap {
        match &self.timeline {
            Timeline::Static { created, status, last_step, .. } => {
                let started = created + 40.0;
                let epoch_s = self.curve.real_epoch_s();
                let finished = started + *last_step as f64 / self.spe as f64 * epoch_s;
                Snap {
                    status: *status,
                    step: *last_step,
                    created: *created,
                    started: Some(started),
                    finished: Some(finished),
                    eta_s: None,
                    cycle: 0,
                    stop_requested: None,
                    queue: None,
                    at: unix_now(),
                }
            }
            Timeline::Live(clock) => clock.lock().unwrap().snap(&self.curve, Instant::now()),
        }
    }

    /// A `status` event (with the run's length).
    fn status_event(&self, status: RunStatus) -> StatusEvent {
        StatusEvent::new(status, self.error(status)).with_length(self.config.epochs, self.spe)
    }

    fn error(&self, status: RunStatus) -> Option<String> {
        match &self.timeline {
            Timeline::Static { error, .. } if status == RunStatus::Failed => error.clone(),
            // A live run fails only when its cloud instance is terminated by hand.
            Timeline::Live(_) if status == RunStatus::Failed => Some(LOST.into()),
            _ => None,
        }
    }

    /// The stable code of [`SimRun::error`].
    fn error_code(&self, status: RunStatus) -> Option<String> {
        let lost = matches!(self.timeline, Timeline::Live(_));
        self.error(status).map(|_| if lost { "cloud_instance_lost" } else { "training_failed" }.to_string())
    }

    /// Wall time of an epoch as reported in `Epoch::duration_s`.
    fn epoch_duration(&self, e: u32) -> f64 {
        let base = match &self.timeline {
            Timeline::Static { .. } => self.curve.real_epoch_s(),
            Timeline::Live(c) => c.lock().unwrap().epoch_s,
        };
        ((base * (1.0 + 0.02 * noise(&[self.curve.seed, e as u64, 50]) as f64)) * 100.0).round() / 100.0
    }

    fn eta_at(&self, step: u64) -> f64 {
        let left = self.curve.total_steps().saturating_sub(step) as f64 / self.spe as f64;
        let per_epoch = match &self.timeline {
            Timeline::Static { .. } => self.curve.real_epoch_s(),
            Timeline::Live(c) => c.lock().unwrap().epoch_s,
        };
        (left * per_epoch).round()
    }

    /// Epochs with validation done at `snap`.
    pub fn epochs_done(&self, snap: &Snap) -> u32 {
        (snap.step / self.spe) as u32
    }

    /// The model id if it exists now.
    ///
    /// A simulated run writes a checkpoint after each validated epoch, so its model exists once the run completed,
    /// or was stopped after at least one finished epoch.
    pub fn current_model(&self, snap: &Snap) -> Option<String> {
        let saved = match snap.status {
            RunStatus::Completed => true,
            RunStatus::Stopped => self.epochs_done(snap) >= 1,
            _ => false,
        };
        if self.user && !saved { None } else { self.model_id.clone() }
    }

    /// API view.
    pub fn to_api(&self) -> Run {
        let snap = self.snap();
        let done = self.epochs_done(&snap);
        Run {
            id: self.id.clone(),
            name: self.config.name.clone(),
            status: snap.status,
            config: self.config.clone(),
            created: rfc3339(snap.created),
            started: snap.started.map(rfc3339),
            finished: snap.finished.map(rfc3339),
            paused_s: None,
            epoch: snap.step.div_ceil(self.spe) as u32,
            epochs: self.config.epochs,
            step: snap.step,
            steps_per_epoch: self.spe,
            eta_s: snap.eta_s.map(f64::round),
            best: self.curve.best(done),
            device: self.device.clone(),
            model_id: self.current_model(&snap),
            error: self.error(snap.status),
            error_code: self.error_code(snap.status),
            resolved: snap.started.and(self.resolved.clone()),
            queue_position: snap.queue.map(|q| q.0),
            cloud: self.cloud_at(&snap),
            dataset: None,
        }
    }

    /// Full history.
    pub fn metrics(&self) -> RunMetrics {
        let snap = self.snap();
        let steps: Vec<Step> = (1..=snap.step).map(|s| self.curve.step(s, self.eta_at(s))).collect();
        let epochs =
            (1..=self.epochs_done(&snap)).map(|e| self.curve.epoch(e, self.epoch_duration(e))).collect();
        RunMetrics { steps, epochs }
    }

    /// True while the run can still change.
    pub fn is_active(&self) -> bool {
        self.snap().status.is_active()
    }

    /// Requests a stop. Returns false if the run is not active.
    pub fn stop(&self) -> bool {
        match &self.timeline {
            Timeline::Static { .. } => false,
            Timeline::Live(c) => {
                let mut c = c.lock().unwrap();
                let now = Instant::now();
                let snap = c.snap(&self.curve, now);
                if !matches!(snap.status, RunStatus::Queued | RunStatus::Running) {
                    return snap.status == RunStatus::Stopping;
                }
                c.stop_at = Some(now.saturating_duration_since(c.origin).as_secs_f64());
                true
            }
        }
    }

    /// Lets `d` pass for a live run at once (test support, [`super::MockStore::pass_time`]).
    pub fn pass_time(&self, d: Duration) {
        if let Timeline::Live(c) = &self.timeline {
            c.lock().unwrap().pass(d);
        }
    }

    /// Marks the run deleted: the ticker exits and open streams end.
    pub fn delete(&self) {
        self.deleted.store(true, Ordering::SeqCst);
        let mut hub = self.hub.lock().unwrap();
        hub.tx = None;
        hub.sse_tx = None;
    }

    /// Subscribes to events, starting the ticker if this is the first subscriber of a live run.
    ///
    /// The receiver is created before the initial snapshot, and a newly started ticker diffs from that same
    /// snapshot, so the first subscriber never misses a transition (later subscribers may see one event twice).
    pub fn subscribe(self: &Arc<Self>) -> Subscription {
        let mut hub = self.hub.lock().unwrap();
        let tx = hub.tx.get_or_insert_with(|| broadcast::channel(256).0);
        let rx = tx.subscribe();
        let snap = self.snap();
        if !hub.ticking && matches!(self.timeline, Timeline::Live(_)) && !self.deleted.load(Ordering::SeqCst)
        {
            hub.ticking = true;
            tokio::spawn(ticker(Arc::clone(self), snap));
        }
        drop(hub);
        let events = rx;
        Subscription { initial: self.status_event(snap.status), events }
    }

    /// Subscribes to pre-serialized SSE frames, sharing one JSON serialization per event across concurrent HTTP
    /// clients and holding a weak resync callback for lag recovery.
    pub fn subscribe_sse(self: &Arc<Self>) -> RunSseSubscription {
        let mut hub = self.hub.lock().unwrap();
        let sse_tx = hub.sse_tx.get_or_insert_with(|| broadcast::channel(256).0);
        let rx = sse_tx.subscribe();
        let snap = self.snap();
        if !hub.ticking && matches!(self.timeline, Timeline::Live(_)) && !self.deleted.load(Ordering::SeqCst)
        {
            hub.ticking = true;
            tokio::spawn(ticker(Arc::clone(self), snap));
        }
        drop(hub);
        let weak = Arc::downgrade(self);
        let resync: RunSseResync = Arc::new(move || {
            let run = weak.upgrade()?;
            let hub = run.hub.lock().unwrap_or_else(|p| p.into_inner());
            if run.deleted.load(Ordering::SeqCst) {
                return None;
            }
            let rx = hub.sse_tx.as_ref()?.subscribe();
            Some((run.status_event(run.snap().status), rx))
        });
        RunSseSubscription {
            initial: SseFrame::run(&RunEvent::Status(self.status_event(snap.status))),
            stream: RunSseStream::Frames(rx),
            resync: Some(resync),
        }
    }

    /// Number of live subscribers (for tests).
    pub fn subscribers(&self) -> usize {
        self.hub.lock().unwrap().receiver_count()
    }

    /// True while a ticker task runs (for tests).
    pub fn ticking(&self) -> bool {
        self.hub.lock().unwrap().ticking
    }
}

#[cfg(test)]
mod tests;
