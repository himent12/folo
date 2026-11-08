//! Simulated training runs: deterministic metric curves, a timeline (fixed for finished runs, wall-clock driven for
//! live ones) and an event ticker that feeds SSE subscribers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_api::{
    Arch, BestEpoch, ClassMetrics, Epoch, LogLevel, LogLine, Loss, Run, RunEvent, RunMetrics, RunStatus,
    StatusEvent, Step, TrainConfig, ValMetrics,
};
use tokio::sync::broadcast;

use crate::store::Subscription;
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

/// Rounds to 4 decimals (keeps the JSON short; metrics are not more precise than that anyway).
fn r4(v: f32) -> f32 {
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
        let m50 = to50(m);
        let (precision, recall) = pr(m50, 21);
        let per_class = (0..self.nc as u32)
            .map(|c| {
                let f = 1.0 + 0.16 * noise(&[self.seed, c as u64, 30]);
                let cm = (m * f).clamp(0.0, 0.99);
                let c50 = to50(cm);
                let (precision, recall) = pr(c50, 40 + 2 * c as u64);
                ClassMetrics {
                    class: c,
                    map50: r4(c50),
                    map50_95: r4(cm),
                    precision: r4(precision),
                    recall: r4(recall),
                }
            })
            .collect();
        ValMetrics {
            map50: r4(m50),
            map50_95: r4(m),
            precision: r4(precision),
            recall: r4(recall),
            per_class,
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
}

impl LiveClock {
    /// A clock that started `offset_s` seconds ago.
    pub fn new(epoch_s: f64, queue_s: f64, hold_s: f64, looping: bool, offset_s: f64) -> LiveClock {
        let now = Instant::now();
        let origin = now.checked_sub(Duration::from_secs_f64(offset_s)).unwrap_or(now);
        let origin_wall = unix_now() - now.duration_since(origin).as_secs_f64();
        LiveClock { origin, origin_wall, epoch_s, queue_s, hold_s, looping, stop_at: None, cycle: 0 }
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
        let run_end = self.queue_s + curve.epochs as f64 * self.epoch_s;
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
            };
            let started = (t >= self.queue_s).then_some(self.origin_wall + self.queue_s);
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
                return base;
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
    timeline: Timeline,
    hub: Mutex<Hub>,
    deleted: AtomicBool,
}

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
            timeline,
            hub: Mutex::default(),
            deleted: AtomicBool::new(false),
        }
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
                }
            }
            Timeline::Live(clock) => clock.lock().unwrap().snap(&self.curve, Instant::now()),
        }
    }

    fn error(&self, status: RunStatus) -> Option<String> {
        match &self.timeline {
            Timeline::Static { error, .. } if status == RunStatus::Failed => error.clone(),
            _ => None,
        }
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
        }
    }

    /// Full history.
    pub fn metrics(&self, every: u64) -> RunMetrics {
        let snap = self.snap();
        let every = every.max(1);
        let mut steps: Vec<Step> = (1..=snap.step)
            .filter(|s| s % every == 0 || *s == snap.step)
            .map(|s| self.curve.step(s, self.eta_at(s)))
            .collect();
        steps.dedup_by_key(|s| s.step);
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
        Subscription { initial: StatusEvent { status: snap.status, error: self.error(snap.status) }, events }
    }

    /// Number of live subscribers (for tests).
    pub fn subscribers(&self) -> usize {
        self.hub.lock().unwrap().tx.as_ref().map_or(0, |t| t.receiver_count())
    }

    /// True while a ticker task runs (for tests).
    pub fn ticking(&self) -> bool {
        self.hub.lock().unwrap().ticking
    }

    fn log(&self, level: LogLevel, message: String) -> RunEvent {
        RunEvent::Log(LogLine { t: rfc3339(unix_now()), level, message })
    }

    /// Events describing the change from `prev` to `now`.
    fn diff(&self, prev: &Snap, now: &Snap) -> Vec<RunEvent> {
        let mut out = Vec::new();
        let status =
            |s: &Snap| RunEvent::Status(StatusEvent { status: s.status, error: self.error(s.status) });
        let mut prev = *prev;
        if now.cycle != prev.cycle {
            out.push(self.log(LogLevel::Info, "mock run finished its loop; restarting from epoch 1".into()));
            prev = Snap { status: RunStatus::Queued, step: 0, ..*now };
            out.push(status(&prev));
        }
        let entering =
            now.status != prev.status && !matches!(now.status, RunStatus::Completed | RunStatus::Stopped);
        if entering {
            out.push(status(now));
            match now.status {
                RunStatus::Running => out.push(self.log(
                    LogLevel::Info,
                    format!(
                        "training {} on {}: {} epochs, batch {}, imgsz {}, {} steps/epoch",
                        self.config.arch.as_str(),
                        self.device,
                        self.config.epochs,
                        self.config.batch,
                        self.config.imgsz,
                        self.spe
                    ),
                )),
                RunStatus::Stopping => {
                    out.push(self.log(LogLevel::Warn, "stop requested; finishing the current step".into()))
                }
                _ => {}
            }
        }
        if now.step > prev.step {
            out.push(RunEvent::Step(self.curve.step(now.step, now.eta_s.unwrap_or(0.0).round())));
        }
        let (done_prev, done_now) = (self.epochs_done(&prev), self.epochs_done(now));
        for e in done_prev + 1..=done_now {
            let ep = self.curve.epoch(e, self.epoch_duration(e));
            out.push(self.log(
                LogLevel::Info,
                format!(
                    "epoch {e}/{}  box {:.3}  cls {:.3}  dfl {:.3}  mAP50 {:.3}  mAP50-95 {:.3}",
                    self.config.epochs,
                    ep.train_loss.box_,
                    ep.train_loss.cls,
                    ep.train_loss.dfl,
                    ep.val.map50,
                    ep.val.map50_95
                ),
            ));
            if self.curve.best(e).is_some_and(|b| b.epoch == e) {
                out.push(
                    self.log(LogLevel::Debug, format!("saved best.folo (mAP50-95 {:.3})", ep.val.map50_95)),
                );
            }
            if e == self.config.epochs.saturating_sub(self.config.augment.close_mosaic) && e > 0 {
                out.push(self.log(LogLevel::Info, "closing mosaic augmentation for the final epochs".into()));
            }
            out.push(RunEvent::Epoch(ep));
        }
        if now.status != prev.status && !entering {
            if now.status == RunStatus::Completed {
                let best = self.curve.best(done_now);
                out.push(self.log(
                    LogLevel::Info,
                    format!(
                        "training complete: best epoch {} (mAP50-95 {:.3})",
                        best.map_or(0, |b| b.epoch),
                        best.map_or(0.0, |b| b.map50_95)
                    ),
                ));
            } else {
                out.push(
                    self.log(LogLevel::Info, format!("stopped at step {}; last checkpoint kept", now.step)),
                );
            }
            out.push(status(now));
        }
        out
    }
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
        std::thread::sleep(Duration::from_millis(1600));
        let snap = early.snap();
        assert_eq!(snap.status, RunStatus::Stopped);
        assert_eq!(early.current_model(&snap), None);
        // Stopped in epoch 3: the last checkpoint is a model.
        let later = user_run(2.5);
        assert_eq!(later.current_model(&later.snap()), None, "running runs expose no model yet");
        assert!(later.stop());
        std::thread::sleep(Duration::from_millis(1600));
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
