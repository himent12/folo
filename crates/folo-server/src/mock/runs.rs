//! Simulated training runs: deterministic metric curves, a timeline (fixed for finished runs, wall-clock driven for
//! live ones) and an event ticker that feeds SSE subscribers.
//!
//! Runs created through the API share one training slot, like the real server's: [`reschedule`] gives each waiting
//! run the moment it starts (when the run ahead of it ends), so its clock, its queue position and its `queue`
//! events follow from the wall clock alone. The seeded looping run is outside the queue.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_api::{
    Arch, BestEpoch, ClassMetrics, CloudInstance, CloudPhase, Epoch, LogLevel, LogLine, Loss, QueueEvent,
    ResolvedTrain, Run, RunCloud, RunEvent, RunMetrics, RunStatus, StatusEvent, Step, TrainConfig,
    ValMetrics,
};
use tokio::sync::broadcast;

use super::cloud::{PHASE_S, SimCloud};
use crate::store::{QueueMove, Subscription};
use crate::util::{noise, rfc3339, unix_now};

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

impl Curve {
    /// Curves for `config` with `spe` steps per epoch converging to `quality`.
    pub fn new(seed: u64, quality: f32, nc: usize, config: &TrainConfig, spe: u64) -> Curve {
        let scale = (640.0 / config.imgsz.max(32) as f32).powi(2);
        Curve {
            seed,
            quality,
            nc,
            epochs: config.epochs,
            spe,
            lr0: config.lr0,
            ips: arch_spec(config.arch).2 * scale,
            batch: config.batch,
            close_mosaic: config.augment.close_mosaic,
        }
    }

    /// Total steps.
    pub fn total_steps(&self) -> u64 {
        self.epochs as u64 * self.spe
    }

    fn epoch_of(&self, step: u64) -> u32 {
        step.div_ceil(self.spe).max(1) as u32
    }

    /// Learning rate at a 1-based global step: linear warmup over 3 epochs, then cosine to `lr0 / 100`.
    pub fn lr(&self, step: u64) -> f32 {
        let warm = (3.min(self.epochs.div_ceil(2)) as u64 * self.spe).max(1);
        let s = step.saturating_sub(1);
        if s < warm {
            self.lr0 * (s + 1) as f32 / warm as f32
        } else {
            let t = (s - warm) as f32 / (self.total_steps() - warm).max(1) as f32;
            self.lr0 * (0.01 + 0.99 * 0.5 * (1.0 + (std::f32::consts::PI * t).cos()))
        }
    }

    fn base_loss(&self, e: f32) -> (f32, f32, f32) {
        let q = self.quality;
        let tau = (self.epochs as f32 * 0.15).max(1.5);
        let p = e / self.epochs as f32;
        let mut boxl = 0.95 + 0.35 * (1.0 - q) + 1.5 * (-e / tau).exp() - 0.15 * p;
        let mut cls = 0.42 + 0.6 * (1.0 - q) + 2.8 * (-e / (0.7 * tau)).exp() - 0.1 * p;
        let mut dfl = 0.88 + 0.15 * (1.0 - q) + 0.65 * (-e / tau).exp() - 0.05 * p;
        if e > (self.epochs.saturating_sub(self.close_mosaic)) as f32 {
            boxl *= 0.93;
            cls *= 0.9;
            dfl *= 0.96;
        }
        (boxl, cls, dfl)
    }

    /// Training loss of one batch.
    pub fn step_loss(&self, step: u64) -> Loss {
        let e = step as f32 / self.spe as f32;
        let (b, c, d) = self.base_loss(e);
        let n = |k: u64| 1.0 + 0.055 * noise(&[self.seed, step, k]);
        let (b, c, d) = (b * n(1), c * n(2), d * n(3));
        Loss { box_: r4(b), cls: r4(c), dfl: r4(d), total: r4(b + c + d) }
    }

    /// Step record with a given ETA.
    pub fn step(&self, step: u64, eta_s: f64) -> Step {
        let warm = (step as f32 / 8.0).min(1.0);
        let ips = self.ips * (0.6 + 0.4 * warm) * (1.0 + 0.03 * noise(&[self.seed, step, 9]));
        let lr = (self.lr(step) as f64 * 1e7).round() / 1e7;
        let imgs_per_s = (ips * 10.0).round() / 10.0;
        Step {
            epoch: self.epoch_of(step),
            step,
            lr: lr as f32,
            loss: self.step_loss(step),
            imgs_per_s,
            eta_s,
            steps_per_epoch: self.spe,
            epochs: self.epochs,
        }
    }

    /// Simulated wall time of one epoch in a "real" (uncompressed) run, including validation.
    pub fn real_epoch_s(&self) -> f64 {
        self.spe as f64 * self.batch as f64 / self.ips as f64 * 1.12
    }

    /// Validation mAP@0.5:0.95 after epoch `e`.
    fn map50_95(&self, e: u32) -> f32 {
        let tau = self.epochs as f32 * 0.14 + 1.0;
        let x = e as f32;
        let mut m = self.quality * (1.0 - (-x / tau).exp()).powf(1.6);
        if e > self.epochs.saturating_sub(self.close_mosaic) {
            m += 0.012 * self.quality;
        }
        let jitter = 0.014 * (1.0 - 0.6 * x / self.epochs as f32);
        r4((m + jitter * noise(&[self.seed, e as u64, 20])).clamp(0.0, 0.99))
    }

    /// Validation metrics after epoch `e`.
    pub fn val(&self, e: u32) -> ValMetrics {
        let to50 = |m: f32| (1.0 - (1.0 - m).powf(1.6)).clamp(0.0, 0.995);
        let pr = |m50: f32, k: u64| {
            let p = (m50 * 1.04 + 0.02 * noise(&[self.seed, e as u64, k])).clamp(0.0, 0.995);
            let r = (m50 * 0.93 + 0.02 * noise(&[self.seed, e as u64, k + 1])).clamp(0.0, 0.995);
            (p, r)
        };
        let m = self.map50_95(e);
        // Per-class mAP50-95 deviates from the mean by zero-mean factors (scaled down so no class passes 0.99), so
        // the overall numbers are the class means, as the engine reports them (docs/METRICS.md).
        let raw: Vec<f32> = (0..self.nc as u64).map(|c| 0.16 * noise(&[self.seed, c, 30])).collect();
        let mean_dev = raw.iter().sum::<f32>() / raw.len().max(1) as f32;
        let dev: Vec<f32> = raw.iter().map(|d| d - mean_dev).collect();
        let room = dev
            .iter()
            .filter(|&&d| d > 0.0)
            .map(|d| (0.99 - m).max(0.0) / (m.max(1e-6) * d))
            .fold(1.0, f32::min);
        let per_class: Vec<ClassMetrics> = dev
            .iter()
            .enumerate()
            .map(|(c, d)| {
                let cm = (m * (1.0 + room * d)).clamp(0.0, 0.99);
                let c50 = to50(cm);
                let (precision, recall) = pr(c50, 40 + 2 * c as u64);
                ClassMetrics {
                    class: c as u32,
                    map50: r4(c50),
                    map50_95: r4(cm),
                    precision: r4(precision),
                    recall: r4(recall),
                }
            })
            .collect();
        let mean = |f: fn(&ClassMetrics) -> f32| {
            r4(per_class.iter().map(f).sum::<f32>() / per_class.len().max(1) as f32)
        };
        let m50 = if per_class.is_empty() { to50(m) } else { mean(|c| c.map50) };
        let (precision, recall) =
            if per_class.is_empty() { pr(m50, 21) } else { (mean(|c| c.precision), mean(|c| c.recall)) };
        // The best-F1 confidence on the 0.001 grid: low while the model is unsure, settling near 0.3-0.45.
        let conf = 0.12 + 0.35 * m50 + 0.02 * noise(&[self.seed, e as u64, 22]);
        ValMetrics {
            map50: r4(m50),
            map50_95: r4(m),
            precision: r4(precision),
            recall: r4(recall),
            per_class,
            best_f1_conf: Some(r3(conf.clamp(0.001, 0.999))),
        }
    }

    /// Epoch record.
    pub fn epoch(&self, e: u32, duration_s: f64) -> Epoch {
        let first = (e as u64 - 1) * self.spe + 1;
        let n = self.spe as f32;
        let mut sum = Loss { box_: 0.0, cls: 0.0, dfl: 0.0, total: 0.0 };
        for s in first..first + self.spe {
            let l = self.step_loss(s);
            sum.box_ += l.box_ / n;
            sum.cls += l.cls / n;
            sum.dfl += l.dfl / n;
            sum.total += l.total / n;
        }
        let train_loss =
            Loss { box_: r4(sum.box_), cls: r4(sum.cls), dfl: r4(sum.dfl), total: r4(sum.total) };
        Epoch { epoch: e, train_loss, val: self.val(e), duration_s }
    }

    /// Best epoch among `1..=done`.
    pub fn best(&self, done: u32) -> Option<BestEpoch> {
        (1..=done)
            .map(|e| BestEpoch { epoch: e, map50_95: self.map50_95(e) })
            .max_by(|a, b| a.map50_95.total_cmp(&b.map50_95).then(b.epoch.cmp(&a.epoch)))
    }
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
    /// `(position, waiting)` while the run waits in the training queue ([`QueueEvent`]).
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

impl LiveClock {
    /// A clock that started `offset_s` seconds ago.
    pub fn new(epoch_s: f64, queue_s: f64, hold_s: f64, looping: bool, offset_s: f64) -> LiveClock {
        let now = Instant::now();
        let origin = now.checked_sub(Duration::from_secs_f64(offset_s)).unwrap_or(now);
        let origin_wall = unix_now() - now.duration_since(origin).as_secs_f64();
        LiveClock {
            origin,
            origin_wall,
            epoch_s,
            queue_s,
            hold_s,
            looping,
            stop_at: None,
            cycle: 0,
            min_start: queue_s,
            queue: None,
            tail_s: 0.0,
            lost_at: None,
        }
    }

    /// The same clock, `running` for `tail_s` more seconds after the last epoch (a cloud run's collecting and
    /// releasing).
    pub fn with_tail(mut self, tail_s: f64) -> LiveClock {
        self.tail_s = tail_s;
        self
    }

    /// Seconds per epoch.
    pub fn epoch_s(&self) -> f64 {
        self.epoch_s
    }

    /// When a started run frees the training slot.
    fn ends_at(&self, curve: &Curve) -> Instant {
        let end = match self.stop_at {
            Some(sa) => sa + STOP_S,
            None => self.queue_s + curve.epochs as f64 * self.epoch_s,
        };
        self.origin + Duration::from_secs_f64(end.max(0.0))
    }

    /// The clock as if `d` had passed: its origin and the planned starts it knows move back by `d`.
    fn pass(&mut self, d: Duration) {
        self.origin -= d;
        self.origin_wall -= d.as_secs_f64();
        if let Some((ahead, all)) = &mut self.queue {
            for start in ahead.iter_mut().chain(all.iter_mut()) {
                *start -= d;
            }
        }
    }

    fn advance(&mut self, by: f64) {
        self.origin += Duration::from_secs_f64(by);
        self.origin_wall += by;
        self.stop_at = None;
        self.cycle += 1;
    }

    fn snap(&mut self, curve: &Curve, now: Instant) -> Snap {
        let step_s = self.epoch_s / curve.spe as f64;
        let total = curve.total_steps();
        let run_end = self.queue_s + curve.epochs as f64 * self.epoch_s + self.tail_s;
        let queue_s = self.queue_s;
        let steps_at = |t: f64| (((t - queue_s) / step_s).floor().max(0.0) as u64).min(total);
        loop {
            let t = now.saturating_duration_since(self.origin).as_secs_f64();
            let base = Snap {
                status: RunStatus::Queued,
                step: 0,
                created: self.origin_wall,
                started: None,
                finished: None,
                eta_s: None,
                cycle: self.cycle,
                stop_requested: self.stop_at.map(|sa| self.origin_wall + sa),
                queue: None,
                at: self.origin_wall + t,
            };
            let started = (t >= self.queue_s).then_some(self.origin_wall + self.queue_s);
            if let Some(lost) = self.lost_at.filter(|&l| t >= l) {
                let started = (lost >= self.queue_s).then_some(self.origin_wall + self.queue_s);
                let finished = Some(self.origin_wall + lost);
                return Snap { status: RunStatus::Failed, step: steps_at(lost), started, finished, ..base };
            }
            if let Some(sa) = self.stop_at {
                let started = (sa >= self.queue_s).then_some(self.origin_wall + self.queue_s);
                let step = steps_at(sa);
                if t < sa + STOP_S {
                    return Snap { status: RunStatus::Stopping, step, started, ..base };
                }
                if self.looping && t >= sa + STOP_S + self.hold_s {
                    self.advance(sa + STOP_S + self.hold_s);
                    continue;
                }
                let finished = Some(self.origin_wall + sa + STOP_S);
                return Snap { status: RunStatus::Stopped, step, started, finished, ..base };
            }
            if t < self.queue_s {
                let queue = self.queue.as_ref().map(|(ahead, all)| {
                    let later = |v: &[Instant]| v.iter().filter(|&&s| s > now).count() as u32;
                    (1 + later(ahead), later(all).max(1))
                });
                return Snap { queue, ..base };
            }
            if t < run_end {
                let step = steps_at(t);
                return Snap { status: RunStatus::Running, step, started, eta_s: Some(run_end - t), ..base };
            }
            if self.looping && t >= run_end + self.hold_s {
                self.advance(run_end + self.hold_s);
                continue;
            }
            let finished = Some(self.origin_wall + run_end);
            return Snap { status: RunStatus::Completed, step: total, started, finished, ..base };
        }
    }
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
    ticking: bool,
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
        }
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
        if let Some(tx) = &hub.tx {
            let _ = tx.send(RunEvent::Log(line));
        }
    }

    /// Sets `Run.resolved` to what the real trainer resolves this run's config to on `train_images` training images.
    pub fn resolved_for(mut self, train_images: u64) -> SimRun {
        let cfg = folo_vision::TrainConfig::from_api(&self.config).expect("mock configs are known");
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

    /// Number of live subscribers (for tests).
    pub fn subscribers(&self) -> usize {
        self.hub.lock().unwrap().tx.as_ref().map_or(0, |t| t.receiver_count())
    }

    /// True while a ticker task runs (for tests).
    pub fn ticking(&self) -> bool {
        self.hub.lock().unwrap().ticking
    }

    fn log(&self, (level, message): (LogLevel, String)) -> RunEvent {
        RunEvent::Log(LogLine { t: rfc3339(unix_now()), level, message })
    }

    fn msg_start(&self) -> (LogLevel, String) {
        let c = &self.config;
        let message = format!(
            "training {} on {}: {} epochs, batch {}, imgsz {}, {} steps/epoch",
            c.arch.as_str(),
            self.device,
            c.epochs,
            c.batch,
            c.imgsz,
            self.spe
        );
        (LogLevel::Info, message)
    }

    fn msg_stopping() -> (LogLevel, String) {
        (LogLevel::Warn, "stop requested; finishing the current step".into())
    }

    fn msg_epoch(&self, ep: &Epoch) -> Vec<(LogLevel, String)> {
        let e = ep.epoch;
        let l = &ep.train_loss;
        let mut out = vec![(
            LogLevel::Info,
            format!(
                "epoch {e}/{}  box {:.3}  cls {:.3}  dfl {:.3}  mAP50 {:.3}  mAP50-95 {:.3}",
                self.config.epochs, l.box_, l.cls, l.dfl, ep.val.map50, ep.val.map50_95
            ),
        )];
        if self.curve.best(e).is_some_and(|b| b.epoch == e) {
            out.push((LogLevel::Debug, format!("saved best.folo (mAP50-95 {:.3})", ep.val.map50_95)));
        }
        if e > 0 && e == self.config.epochs.saturating_sub(self.config.augment.close_mosaic) {
            out.push((LogLevel::Info, "closing mosaic augmentation for the final epochs".into()));
        }
        out
    }

    fn msg_end(&self, snap: &Snap) -> Option<(LogLevel, String)> {
        match snap.status {
            RunStatus::Completed => {
                let best = self.curve.best(self.epochs_done(snap));
                let message = format!(
                    "training complete: best epoch {} (mAP50-95 {:.3})",
                    best.map_or(0, |b| b.epoch),
                    best.map_or(0.0, |b| b.map50_95)
                );
                Some((LogLevel::Info, message))
            }
            RunStatus::Stopped => {
                Some((LogLevel::Info, format!("stopped at step {}; last checkpoint kept", snap.step)))
            }
            RunStatus::Failed => Some((
                LogLevel::Error,
                format!(
                    "training failed at step {}: {}",
                    snap.step,
                    self.error(snap.status).unwrap_or_default()
                ),
            )),
            _ => None,
        }
    }

    /// Log history of the current cycle, oldest first: the same lines the event stream sends, stamped with the
    /// simulated times they happened at.
    pub fn logs(&self) -> Vec<LogLine> {
        let snap = self.snap();
        let per_epoch = match &self.timeline {
            Timeline::Static { .. } => self.curve.real_epoch_s(),
            Timeline::Live(c) => c.lock().unwrap().epoch_s,
        };
        let line = |t: f64, (level, message): (LogLevel, String)| LogLine { t: rfc3339(t), level, message };
        let mut out = Vec::new();
        // Training lines only exist once it started; a stop (even while queued) and the end are logged regardless.
        if let Some(started) = snap.started {
            out.push(line(started, self.msg_start()));
            for e in 1..=self.epochs_done(&snap) {
                let t = started + e as f64 * per_epoch;
                let ep = self.curve.epoch(e, self.epoch_duration(e));
                out.extend(self.msg_epoch(&ep).into_iter().map(|m| line(t, m)));
            }
        }
        if let Some(t) = snap.stop_requested {
            out.push(line(t, Self::msg_stopping()));
        }
        if let (Some(m), Some(t)) = (self.msg_end(&snap), snap.finished) {
            out.push(line(t, m));
        }
        out.extend(self.cloud_lines(&snap).into_iter().map(|(t, m)| line(t, (LogLevel::Info, m))));
        // Queue moves, merged in by time (the sort is stable, so lines of the same second keep their order).
        out.extend(self.notes.lock().unwrap().iter().cloned());
        out.sort_by(|a, b| a.t.cmp(&b.t));
        out
    }

    /// A cloud run's phase lines up to `snap`, with the times their phases began: before the training at the fixed
    /// steps of [`PHASE_S`], then when it starts, ends (or is stopped) and the run is released.
    fn cloud_lines(&self, snap: &Snap) -> Vec<(f64, String)> {
        let (Some(c), Some(cloud)) = (&self.cloud, self.cloud_at(snap)) else { return Vec::new() };
        let epoch_s = match &self.timeline {
            Timeline::Static { .. } => self.curve.real_epoch_s(),
            Timeline::Live(clock) => clock.lock().unwrap().epoch_s(),
        };
        let trained = snap.started.map(|s| s + self.config.epochs as f64 * epoch_s);
        let releasing = match snap.stop_requested {
            Some(stop) => Some(stop),
            None => trained.map(|t| t + PHASE_S),
        };
        let until = snap.finished.unwrap_or(snap.at);
        [
            (CloudPhase::Provisioning, Some(snap.created)),
            (CloudPhase::Booting, Some(snap.created + PHASE_S)),
            (CloudPhase::Syncing, Some(snap.created + 2.0 * PHASE_S)),
            (CloudPhase::Training, snap.started),
            (CloudPhase::Collecting, trained.filter(|_| snap.stop_requested.is_none())),
            (CloudPhase::Releasing, releasing),
            (CloudPhase::Released, snap.finished),
        ]
        .into_iter()
        .filter_map(|(p, t)| {
            let t = t.filter(|&t| p <= cloud.phase && t <= until)?;
            Some((t, c.phase_line(p, cloud.cost_usd)?))
        })
        .collect()
    }

    /// Events describing the change from `prev` to `now`.
    fn diff(&self, prev: &Snap, now: &Snap) -> Vec<RunEvent> {
        let mut out = Vec::new();
        let status = |s: &Snap| RunEvent::Status(self.status_event(s.status));
        let mut prev = *prev;
        if now.cycle != prev.cycle {
            out.push(
                self.log((LogLevel::Info, "mock run finished its loop; restarting from epoch 1".into())),
            );
            prev = Snap { status: RunStatus::Queued, step: 0, ..*now };
            out.push(status(&prev));
        }
        let entering =
            now.status != prev.status && !matches!(now.status, RunStatus::Completed | RunStatus::Stopped);
        if now.queue != prev.queue
            && let Some((position, waiting)) = now.queue
        {
            out.push(RunEvent::Queue(QueueEvent { position, waiting }));
        }
        // A cloud run's hand-over: a line when its phase changes, its state whenever it changed (progress moves in
        // tenths, the cost in cents).
        if let (Some(c), Some(before), Some(after)) = (&self.cloud, self.cloud_at(&prev), self.cloud_at(now))
            && before != after
        {
            if before.phase != after.phase
                && let Some(m) = c.phase_line(after.phase, after.cost_usd)
            {
                out.push(self.log((LogLevel::Info, m)));
            }
            out.push(RunEvent::Cloud(Box::new(after)));
        }
        if entering {
            out.push(status(now));
            match now.status {
                RunStatus::Running => out.push(self.log(self.msg_start())),
                RunStatus::Stopping => out.push(self.log(Self::msg_stopping())),
                _ => {}
            }
        }
        if now.step > prev.step {
            out.push(RunEvent::Step(self.curve.step(now.step, now.eta_s.unwrap_or(0.0).round())));
        }
        let (done_prev, done_now) = (self.epochs_done(&prev), self.epochs_done(now));
        for e in done_prev + 1..=done_now {
            let ep = self.curve.epoch(e, self.epoch_duration(e));
            out.extend(self.msg_epoch(&ep).into_iter().map(|m| self.log(m)));
            out.push(RunEvent::Epoch(ep));
        }
        if now.status != prev.status && !entering {
            out.extend(self.msg_end(now).map(|m| self.log(m)));
            out.push(status(now));
        }
        out
    }
}

/// Whether `run` waits in the training queue now (queued, not stopped).
pub fn is_waiting(run: &SimRun) -> bool {
    run.snap().queue.is_some()
}

/// Plans the training slot shared by the runs in `queue` (front first) as of `now`: runs that started keep their
/// timeline; each waiting run starts when the slot frees (not before its `min_start`), in queue order. Every
/// waiting run also learns the starts of the runs ahead of it, so its queue position follows from the clock.
pub fn reschedule(queue: &[Arc<SimRun>], now: Instant) {
    let mut free_at = now;
    let mut starts: Vec<(Arc<SimRun>, Instant)> = Vec::new();
    for run in queue {
        let Timeline::Live(c) = &run.timeline else { continue };
        let mut c = c.lock().unwrap();
        let snap = c.snap(&run.curve, now);
        match snap.status {
            // A run stopped while it waited never held the slot.
            RunStatus::Running | RunStatus::Stopping if snap.started.is_some() => {
                free_at = free_at.max(c.ends_at(&run.curve));
                c.queue = None;
            }
            RunStatus::Queued => {
                let start = free_at.max(c.origin + Duration::from_secs_f64(c.min_start.max(0.0)));
                c.queue_s = start.saturating_duration_since(c.origin).as_secs_f64();
                free_at = start + Duration::from_secs_f64(run.curve.epochs as f64 * c.epoch_s);
                starts.push((run.clone(), start));
            }
            _ => c.queue = None,
        }
    }
    let all: Vec<Instant> = starts.iter().map(|s| s.1).collect();
    for (i, (run, _)) in starts.iter().enumerate() {
        if let Timeline::Live(c) = &run.timeline {
            c.lock().unwrap().queue = Some((all[..i].to_vec(), all.clone()));
        }
    }
}

/// Moves run `id` among the waiting runs of `queue` (runs that are not waiting keep their order, ahead of the
/// waiting ones) and plans the slot again. `start` also lets the run start at once when the slot is free. Returns
/// `(from, to, waiting)` (0-based places), or `None` when the run does not wait in the queue.
pub fn move_waiting(queue: &mut Vec<Arc<SimRun>>, id: &str, mv: QueueMove) -> Option<(usize, usize, usize)> {
    let now = Instant::now();
    reschedule(queue, now);
    let (mut waiting, rest): (Vec<Arc<SimRun>>, Vec<Arc<SimRun>>) =
        queue.drain(..).partition(|r| is_waiting(r));
    let from = waiting.iter().position(|r| r.id == id);
    let moved = from.map(|from| {
        let to = mv.target(from, waiting.len());
        let run = waiting.remove(from);
        if mv == QueueMove::Start
            && let Timeline::Live(c) = &run.timeline
        {
            let mut c = c.lock().unwrap();
            c.min_start = now.saturating_duration_since(c.origin).as_secs_f64();
        }
        waiting.insert(to, run);
        (from, to, waiting.len())
    });
    queue.extend(rest);
    queue.extend(waiting);
    reschedule(queue, now);
    moved
}

/// Publishes a live run's events while it has subscribers; exits when the last one disconnects.
async fn ticker(run: Arc<SimRun>, mut last: Snap) {
    let mut interval = tokio::time::interval(TICK);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let now = run.snap();
        let events = run.diff(&last, &now);
        last = now;
        let mut hub = run.hub.lock().unwrap();
        let alive = !run.deleted.load(Ordering::SeqCst);
        match &hub.tx {
            Some(tx) if alive && tx.receiver_count() > 0 => {
                for ev in events {
                    let _ = tx.send(ev);
                }
            }
            _ => {
                hub.ticking = false;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(epochs: u32) -> TrainConfig {
        serde_json::from_value(serde_json::json!({ "dataset_id": "ds", "epochs": epochs })).unwrap()
    }

    fn user_run(offset_s: f64) -> SimRun {
        let cfg = config(10);
        let curve = Curve::new(3, 0.5, 1, &cfg, 10);
        let clock = LiveClock::new(1.0, 0.0, 0.0, false, offset_s);
        let t = Timeline::Live(Mutex::new(clock));
        SimRun::new(
            "run_x".into(),
            cfg,
            vec!["a".into()],
            10,
            curve,
            "cpu".into(),
            Some("m_x".into()),
            true,
            t,
        )
    }

    #[test]
    fn stopped_runs_keep_a_checkpoint_after_an_epoch() {
        // Stopped mid-epoch 1: nothing validated, no model.
        let early = user_run(0.5);
        assert!(early.stop());
        assert_eq!(early.snap().status, RunStatus::Stopping);
        early.pass_time(Duration::from_secs_f64(STOP_S + 0.1));
        let snap = early.snap();
        assert_eq!(snap.status, RunStatus::Stopped);
        assert_eq!(early.current_model(&snap), None);
        // Stopped in epoch 3: the last checkpoint is a model.
        let later = user_run(2.5);
        assert_eq!(later.current_model(&later.snap()), None, "running runs expose no model yet");
        assert!(later.stop());
        later.pass_time(Duration::from_secs_f64(STOP_S + 0.1));
        let snap = later.snap();
        assert_eq!(snap.status, RunStatus::Stopped);
        assert_eq!(later.to_api().model_id.as_deref(), Some("m_x"));
    }

    #[test]
    fn curves_are_plausible() {
        let c = Curve::new(1, 0.7, 3, &config(100), 50);
        // Warmup then cosine.
        assert!(c.lr(1) < c.lr(100));
        assert!((c.lr(150) - 0.01).abs() < 1e-4);
        assert!(c.lr(5000) < 0.0002);
        // Loss decays.
        let e1 = c.epoch(1, 1.0);
        let e100 = c.epoch(100, 1.0);
        assert!(e100.train_loss.total < e1.train_loss.total * 0.6);
        assert!(
            (e100.train_loss.total - (e100.train_loss.box_ + e100.train_loss.cls + e100.train_loss.dfl))
                .abs()
                < 1e-3
        );
        // mAP rises and saturates near quality.
        assert!(c.val(1).map50_95 < 0.1);
        assert!((c.val(100).map50_95 - 0.7).abs() < 0.04);
        assert!(c.val(100).map50 > c.val(100).map50_95);
        assert_eq!(c.val(5).per_class.len(), 3);
        let best = c.best(100).unwrap();
        assert!(best.epoch > 60);
    }

    #[test]
    fn live_clock_phases_and_stop() {
        let cfg = config(10);
        let curve = Curve::new(2, 0.5, 1, &cfg, 10);
        let mut clock = LiveClock::new(1.0, 2.0, 3.0, true, 0.0);
        let t0 = clock.origin;
        let at = |c: &mut LiveClock, s: f64| c.snap(&curve, t0 + Duration::from_secs_f64(s));
        assert_eq!(at(&mut clock, 1.0).status, RunStatus::Queued);
        let s = at(&mut clock, 4.55);
        assert_eq!((s.status, s.step), (RunStatus::Running, 25));
        assert_eq!(at(&mut clock, 12.5).status, RunStatus::Completed);
        // Loops after the hold.
        let s = at(&mut clock, 15.5);
        assert_eq!((s.status, s.cycle), (RunStatus::Queued, 1));
        // A stopped looping run restarts after the hold, too.
        clock.stop_at = Some(0.5);
        assert_eq!(at(&mut clock, 18.0).status, RunStatus::Stopped);
        let s = at(&mut clock, 21.0);
        assert_eq!((s.status, s.cycle), (RunStatus::Queued, 2));
    }

    #[test]
    fn stop_sequence() {
        let cfg = config(10);
        let curve = Curve::new(2, 0.5, 1, &cfg, 10);
        let mut clock = LiveClock::new(1.0, 0.0, 100.0, false, 0.0);
        let t0 = clock.origin;
        clock.stop_at = Some(5.0);
        let s = clock.snap(&curve, t0 + Duration::from_secs_f64(5.5));
        assert_eq!((s.status, s.step), (RunStatus::Stopping, 50));
        let s = clock.snap(&curve, t0 + Duration::from_secs_f64(8.0));
        assert_eq!((s.status, s.step), (RunStatus::Stopped, 50));
        assert!(s.finished.is_some());
    }
}
