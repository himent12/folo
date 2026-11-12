//! Simulated training: a run's learning curves, the demo clock, a run's log, the settings a
//! config resolves to, and the `folo train` command for it.

use std::time::Instant;

use crate::api::time;
use crate::api::*;
use crate::synth::Rng;

use super::cloud::CloudSim;
use super::{LIVE_ID, MOCK_NOW, MockBackend, STEP_DT};

/// Shape of a run's simulated learning curves.
#[derive(Clone)]
pub(super) struct Curve {
    pub(super) seed: u64,
    pub(super) epochs: u32,
    pub(super) spe: u64,
    pub(super) lr0: f32,
    pub(super) batch: u32,
    pub(super) map_max: f32,
    pub(super) ips: f32,
    pub(super) classes: u32,
    pub(super) close_mosaic: u32,
}

impl Curve {
    fn noise(&self, i: u64, k: u64) -> f32 {
        Rng::new(self.seed ^ i.wrapping_mul(0x2545_F491) ^ k).normal()
    }

    fn lr(&self, frac: f32) -> f32 {
        let warm = 3.0;
        if frac < warm {
            self.lr0 * (0.1 + 0.9 * frac / warm)
        } else {
            // Linear decay to 1% (lrf 0.01, cos_lr off), as the run's resolved settings say.
            let t = ((frac - warm) / (self.epochs as f32 - warm)).clamp(0.0, 1.0);
            self.lr0 * (1.0 - 0.99 * t)
        }
    }

    fn loss_at(&self, frac: f32) -> Loss {
        let e = self.epochs as f32;
        let closing = (frac - (e - self.close_mosaic as f32)).clamp(0.0, 1.0);
        let k = 1.0 - 0.1 * closing;
        let box_ = (0.92 + 1.38 * (-frac / (e * 0.2)).exp()) * k;
        let cls = (0.40 + 2.7 * (-frac / (e * 0.1)).exp()) * k;
        let dfl = (0.86 + 0.58 * (-frac / (e * 0.24)).exp()) * k;
        Loss { box_, cls, dfl, total: box_ + cls + dfl }
    }

    pub(super) fn step(&self, i: u64) -> Step {
        let frac = i as f32 / self.spe as f32;
        let base = self.loss_at(frac);
        let jitter = |k: u64| 1.0 + 0.045 * self.noise(i, k);
        let box_ = base.box_ * jitter(1);
        let cls = base.cls * jitter(2);
        let dfl = base.dfl * jitter(3);
        let ips = self.ips * (1.0 + 0.025 * self.noise(i, 4));
        let remaining = (self.epochs as u64 * self.spe).saturating_sub(i) as f64;
        Step {
            epoch: ((i.max(1) - 1) / self.spe + 1) as u32,
            step: i,
            lr: self.lr(frac),
            loss: Loss { box_, cls, dfl, total: box_ + cls + dfl },
            imgs_per_s: ips,
            eta_s: remaining * STEP_DT,
            steps_per_epoch: self.spe,
            epochs: self.epochs,
        }
    }

    pub(super) fn quality(&self, epoch: u32) -> f32 {
        let e = epoch as f32;
        let n = self.epochs as f32;
        // Without a close-mosaic phase there is no late boost (and no 0 / 0).
        let closing = if self.close_mosaic == 0 {
            0.0
        } else {
            (e - (n - self.close_mosaic as f32)).clamp(0.0, self.close_mosaic as f32)
                / self.close_mosaic as f32
        };
        ((1.0 - (-e / (n * 0.17)).exp()) * (1.0 + 0.012 * self.noise(epoch as u64, 9)) + 0.025 * closing)
            .min(1.02)
    }

    pub(super) fn epoch(&self, epoch: u32) -> Epoch {
        let q = self.quality(epoch);
        let m = (self.map_max * q).clamp(0.0, 0.99);
        let map50 = (m * 1.22 + 0.05 * (1.0 - m)).min(0.985);
        let per_class = (0..self.classes)
            .map(|c| {
                let d = 1.0 + 0.09 * Rng::new(self.seed ^ c as u64).normal() - 0.03 * c as f32;
                let cm = (m * d).clamp(0.0, 0.99);
                ClassMetrics {
                    class: c,
                    map50: (cm * 1.22 + 0.05 * (1.0 - cm)).min(0.99),
                    map50_95: cm,
                    precision: (0.42 + 0.56 * cm / self.map_max.max(0.01) * 0.93).min(0.99),
                    recall: (0.35 + 0.6 * cm / self.map_max.max(0.01) * 0.9).min(0.99),
                }
            })
            .collect();
        Epoch {
            epoch,
            train_loss: self.loss_at(epoch as f32 - 0.5),
            val: ValMetrics {
                map50,
                map50_95: m,
                precision: (0.42 + 0.52 * q).min(0.98),
                recall: (0.33 + 0.56 * q).min(0.97),
                per_class,
                best_f1_conf: Some(best_f1_conf(map50, self.noise(epoch as u64, 22))),
            },
            duration_s: (self.spe * self.batch as u64) as f64 / self.ips as f64 + 2.4,
        }
    }
}

/// The best-F1 confidence (docs/METRICS.md) of a validation with mAP@0.5 `map50`, on the server's 0.001 grid:
/// low while the model is unsure, settling near 0.3-0.45 (as `folo serve --mock`'s).
pub(super) fn best_f1_conf(map50: f32, jitter: f32) -> f32 {
    ((0.12 + 0.35 * map50 + 0.02 * jitter).clamp(0.001, 0.999) * 1000.0).round() / 1000.0
}

pub(super) struct RunFixture {
    pub(super) run: Run,
    pub(super) curve: Curve,
    /// Live runs: steps done when the clock started, and when that was.
    pub(super) live: Option<(u64, Instant)>,
    /// Frozen at this step (finished, stopped, failed, or a frozen clock).
    pub(super) step: u64,
    pub(super) logs: Vec<LogLine>,
    /// A cloud run's simulated life on its instance; its steps and status come from there.
    pub(super) cloud: Option<CloudSim>,
}

/// The demo's clock as the simulators read it: frozen at [`MOCK_NOW`] for screenshots, else
/// running from when the backend was made.
#[derive(Clone, Copy)]
pub(super) struct Clock {
    pub(super) frozen: bool,
    pub(super) started: Instant,
}

impl Clock {
    /// Seconds since the Unix epoch, with a fraction.
    pub(super) fn now(&self) -> f64 {
        MOCK_NOW as f64 + if self.frozen { 0. } else { self.started.elapsed().as_secs_f64() }
    }
}

/// An epoch's summary line in a run's log.
pub(super) fn epoch_line(e: u32, epochs: u32, ep: &Epoch) -> String {
    format!(
        "epoch {e}/{epochs} · mAP50-95 {:.3} · mAP50 {:.3} · {:.1} s",
        ep.val.map50_95, ep.val.map50, ep.duration_s
    )
}

pub(super) fn initial_logs(f: &RunFixture, created: i64) -> Vec<LogLine> {
    let c = &f.run.config;
    // What the run trains with, as the trainer's first log line says: its resolution, or the one a new run gets
    // (`steps_per_epoch` batches of `batch` images; the class count of its dataset).
    let r = f.run.resolved.clone().unwrap_or_else(|| {
        let classes = f.run.dataset.as_ref().map_or(1, |d| d.classes.len().max(1));
        resolve(c, f.run.steps_per_epoch * c.batch.max(1) as u64, classes)
    });
    let mut logs = vec![
        (
            0,
            LogLevel::Info,
            format!(
                "folo train {} on {} ({} epochs, batch {}, imgsz {})",
                c.arch.as_str(),
                c.dataset_id,
                c.epochs,
                c.batch,
                c.imgsz
            ),
        ),
        // The worker's, for a cloud run.
        (1, LogLevel::Info, format!("device: {}", MockBackend::snapshot(f).device)),
        (2, LogLevel::Debug, "dataloader: 8 workers, prefetch 4 batches, mosaic 1.0".to_string()),
        (
            3,
            LogLevel::Info,
            format!(
                "model: {} · {} params · fused BN off (training)",
                c.arch.as_str(),
                match c.arch.as_str() {
                    "folo-n" => "2.63 M",
                    "folo-s" => "9.42 M",
                    "folo-m" => "20.1 M",
                    _ => "43.7 M",
                }
            ),
        ),
        (
            4,
            LogLevel::Info,
            format!(
                "optimizer: {}(lr={}, wd={}){}, {} schedule, warmup {} steps",
                match r.optimizer {
                    ResolvedOptimizer::Sgd => "SGD",
                    ResolvedOptimizer::Adamw => "AdamW",
                    ResolvedOptimizer::Musgd => "MuSGD",
                    _ => "unknown",
                },
                r.lr0,
                r.weight_decay,
                if c.optimizer == Optimizer::Auto { " (auto)" } else { "" },
                if r.cos_lr { "cosine" } else { "linear" },
                r.warmup_iters
            ),
        ),
    ];
    if f.run.status == RunStatus::Failed {
        logs.push((40, LogLevel::Warn, "GPU memory at 97% after epoch 2 (15.5 / 16.0 GiB)".into()));
        logs.push((48, LogLevel::Error, f.run.error.clone().unwrap_or_default()));
    }
    if f.run.status == RunStatus::Stopped {
        logs.push((900, LogLevel::Warn, "stop requested; finishing current step".into()));
        logs.push((901, LogLevel::Info, "saved last checkpoint (epoch 37)".into()));
    }
    if f.run.id == LIVE_ID {
        logs.push((120, LogLevel::Warn, "label 'img_03417' has a box outside the image; clipped".into()));
    }
    let lines = |logs: Vec<(i64, LogLevel, String)>, from: i64| {
        logs.into_iter().map(move |(dt, level, message)| LogLine {
            t: time::format(from + dt),
            level,
            message,
        })
    };
    let Some(sim) = &f.cloud else { return lines(logs, created + 5).collect() };
    // A cloud run's own lines lead, up to where it is; the trainer's follow once it trains there.
    let [provision, boot, _] = sim.pre.map(|s| s as i64);
    let train_at = sim.train_at() as i64;
    let pod = sim.instance(&f.run.id).and_then(|i| i.provider_ref).unwrap_or_default();
    let (size, unit) = crate::ui::bytes(sim.upload_bytes);
    let cloud = vec![
        (
            0,
            LogLevel::Info,
            format!(
                "cloud: launching {} at {}",
                crate::ui::offer_line(&sim.base.offer),
                crate::ui::fmt_price(sim.base.offer.price_per_hour)
            ),
        ),
        (provision, LogLevel::Info, format!("cloud: instance {pod} is up; waiting for its worker")),
        (
            provision + boot,
            LogLevel::Info,
            format!("cloud: worker ready; uploading the dataset ({size} {unit})"),
        ),
        (train_at, LogLevel::Info, "cloud: upload done; training on the instance".to_string()),
    ];
    let elapsed = sim.elapsed() as i64;
    let trained = if elapsed >= train_at { logs } else { Vec::new() };
    lines(cloud.into_iter().filter(|(dt, _, _)| *dt <= elapsed).collect(), created)
        .chain(lines(trained, created + train_at + 5))
        .collect()
}

/// Plausible detections for known ground truth at model quality `q` (0..1): jittered hits,
/// occasional class confusions and misses, and a few low-score false positives.
/// The settings a run resolves to, as the trainer would (docs/MODEL.md §8.2 and §8.4).
pub(super) fn resolve(config: &TrainConfig, train_images: u64, classes: usize) -> ResolvedTrain {
    let nbs = 64u32;
    let batch = config.batch.max(1);
    let accumulate = ((nbs as f64 / batch as f64).round() as u32).max(1);
    let batches = train_images.div_ceil(batch as u64).max(1);
    // A chosen optimizer keeps the configured momentum 0.937 and bias warmup 0.1; `auto` sets its own.
    let (optimizer, lr0, momentum, warmup_bias_lr) = match config.optimizer {
        Optimizer::Sgd => (ResolvedOptimizer::Sgd, config.lr0 as f64, 0.937, 0.1),
        Optimizer::Adamw => (ResolvedOptimizer::Adamw, config.lr0 as f64, 0.937, 0.1),
        Optimizer::Musgd => (ResolvedOptimizer::Musgd, config.lr0 as f64, 0.937, 0.1),
        // The engine's own rule (docs/MODEL.md §8.2); the real server refuses an unknown optimizer.
        Optimizer::Auto | Optimizer::Unknown => {
            let a = folo_nn::optim::auto_optimizer(
                train_images as usize,
                batch as usize,
                nbs as usize,
                config.epochs as usize,
                classes,
            );
            let kind = match a.kind {
                folo_nn::optim::OptimKind::Sgd => ResolvedOptimizer::Sgd,
                folo_nn::optim::OptimKind::AdamW => ResolvedOptimizer::Adamw,
                folo_nn::optim::OptimKind::MuSgd => ResolvedOptimizer::Musgd,
            };
            (kind, a.lr0, a.momentum, a.warmup_bias_lr)
        }
    };
    // Warmup in optimizer steps: the trainer's own rule (docs/MODEL.md §8.4; new runs have no warmup floor).
    let spe = batches.div_ceil(accumulate as u64);
    let warmup_iters = folo_nn::warmup_iters_floor(3.0, config.epochs as u64, spe, 0);
    ResolvedTrain {
        optimizer,
        lr0,
        lrf: 0.01,
        cos_lr: false,
        momentum,
        betas: (optimizer == ResolvedOptimizer::Adamw).then_some([momentum, 0.999]),
        weight_decay: 0.0005 * batch as f64 * accumulate as f64 / nbs as f64,
        batch,
        accumulate,
        nbs,
        optimizer_steps_per_epoch: spe,
        warmup_iters,
        warmup_epochs: 3.0,
        warmup_momentum: 0.8,
        warmup_bias_lr,
        // The trainer's cap: at most half the run.
        close_mosaic: Some(config.augment.close_mosaic.min(config.epochs / 2)),
    }
}

impl MockBackend {
    /// The demo's `folo train` command for a config (the flags the CLI takes for what the form sets).
    pub(super) fn cli_for(&self, c: &TrainConfig) -> CliCommand {
        let ds = self
            .datasets
            .iter()
            .find(|d| d.ds.id == c.dataset_id)
            .map_or(c.dataset_id.clone(), |d| d.ds.path.replace('\\', "/"));
        let mut argv: Vec<String> = vec![
            "folo".into(),
            "train".into(),
            "--data".into(),
            ds,
            "--arch".into(),
            c.arch.as_str().into(),
            "--epochs".into(),
            c.epochs.to_string(),
            "--batch".into(),
            c.batch.to_string(),
            "--imgsz".into(),
            c.imgsz.to_string(),
            "--device".into(),
            c.device.clone(),
        ];
        if c.amp {
            argv.push("--amp".into());
        }
        if c.head == Head::O2o {
            argv.extend(["--head".into(), "o2o".into()]);
        }
        let cli = argv
            .iter()
            .map(|a| if a.contains(' ') { format!("'{a}'") } else { a.clone() })
            .collect::<Vec<_>>()
            .join(" ");
        CliCommand { cli, argv }
    }
}
