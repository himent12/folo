//! In-process fixtures behind [`Backend`]: three synthetic datasets, a handful of models, finished,
//! failed, stopped and queued runs, and one live run whose metrics advance in real time.
//!
//! Everything is a pure function of seeds and the simulated clock, so a frozen clock gives
//! pixel-identical screenshots.

use std::collections::HashMap;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::time;
use super::*;
use crate::synth::{Rng, Scene, hash};

/// The fixtures' "now": 2026-11-08T07:30:00Z.
pub const MOCK_NOW: i64 = 1_794_123_000;

/// Seconds per training step of a live mock run.
const STEP_DT: f64 = 0.12;

struct DatasetFixture {
    ds: Dataset,
    seed: u64,
    max_objects: u32,
    small: bool,
    /// Per-image boxes, computed once from the scenes.
    images: Vec<ImageInfo>,
}

impl DatasetFixture {
    fn new(
        id: &str,
        name: &str,
        path: &str,
        format: &str,
        classes: &[&str],
        splits: (u32, u32, u32),
        max_objects: u32,
        small: bool,
        sizes: &'static [(u32, u32)],
        created: i64,
    ) -> Self {
        let seed = hash(id);
        let total = splits.0 + splits.1 + splits.2;
        let mut images = Vec::with_capacity(total as usize);
        for i in 0..total {
            let split = if i < splits.0 {
                Split::Train
            } else if i < splits.0 + splits.1 {
                Split::Val
            } else {
                Split::Test
            };
            let (w, h) = sizes[(i as usize * 7 + 3) % sizes.len()];
            let scene = Scene::generate(
                seed ^ (i as u64).wrapping_mul(0x9E37),
                classes.len() as u32,
                max_objects,
                small,
            );
            images.push(ImageInfo {
                // 1-based ids, so "7 of 4,400" in the inspector shows img_00007.
                id: format!("img_{:05}", i + 1),
                file: format!("images/{}/img_{:05}.jpg", split.label(), i + 1),
                split,
                width: w,
                height: h,
                boxes: scene.boxes(w, h),
            });
        }
        let mut class_counts = vec![0u64; classes.len()];
        let size_edges = vec![0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
        let mut size_counts = vec![0u64; size_edges.len() - 1];
        let per_edges = vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let mut per_counts = vec![0u64; per_edges.len() - 1];
        let mut boxes = 0u64;
        let mut empty = 0u64;
        for im in &images {
            boxes += im.boxes.len() as u64;
            if im.boxes.is_empty() {
                empty += 1;
            }
            let n = im.boxes.len() as f64;
            let slot =
                per_edges.windows(2).position(|w| n >= w[0] && n < w[1]).unwrap_or(per_counts.len() - 1);
            per_counts[slot] += 1;
            for b in &im.boxes {
                class_counts[b.class as usize] += 1;
                // Relative size: sqrt(box area / image area), as docs/API.md defines it.
                let area = (((b.x2 - b.x1) * (b.y2 - b.y1)) as f64 / (im.width * im.height) as f64).sqrt();
                let slot = size_edges
                    .windows(2)
                    .position(|w| area >= w[0] && area < w[1])
                    .unwrap_or(size_counts.len() - 1);
                size_counts[slot] += 1;
            }
        }
        let ds = Dataset {
            id: id.into(),
            name: name.into(),
            path: path.into(),
            format: if format == "coco" { DatasetFormat::Coco } else { DatasetFormat::Yolo },
            classes: classes.iter().map(|c| c.to_string()).collect(),
            created: time::format(created),
            splits: Splits { train: splits.0 as u64, val: splits.1 as u64, test: splits.2 as u64 },
            stats: Some(DatasetStats {
                boxes,
                empty_images: empty,
                broken_labels: if small { 3 } else { 0 },
                class_counts,
                box_size_hist: Histogram { edges: size_edges, counts: size_counts },
                boxes_per_image_hist: Histogram { edges: per_edges, counts: per_counts },
            }),
        };
        Self { ds, seed, max_objects, small, images }
    }

    fn scene(&self, index: u32) -> Scene {
        Scene::generate(
            self.seed ^ (index as u64).wrapping_mul(0x9E37),
            self.ds.classes.len() as u32,
            self.max_objects,
            self.small,
        )
    }
}

/// Shape of a run's simulated learning curves.
#[derive(Clone)]
struct Curve {
    seed: u64,
    epochs: u32,
    spe: u64,
    lr0: f32,
    batch: u32,
    map_max: f32,
    ips: f32,
    classes: u32,
    close_mosaic: u32,
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

    fn step(&self, i: u64) -> Step {
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

    fn quality(&self, epoch: u32) -> f32 {
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

    fn epoch(&self, epoch: u32) -> Epoch {
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
fn best_f1_conf(map50: f32, jitter: f32) -> f32 {
    ((0.12 + 0.35 * map50 + 0.02 * jitter).clamp(0.001, 0.999) * 1000.0).round() / 1000.0
}

struct RunFixture {
    run: Run,
    curve: Curve,
    /// Live runs: steps done when the clock started, and when that was.
    live: Option<(u64, Instant)>,
    /// Frozen at this step (finished, stopped, failed, or a frozen clock).
    step: u64,
    logs: Vec<LogLine>,
    /// A cloud run's simulated life on its instance; its steps and status come from there.
    cloud: Option<CloudSim>,
}

/// The demo's clock as the simulators read it: frozen at [`MOCK_NOW`] for screenshots, else
/// running from when the backend was made.
#[derive(Clone, Copy)]
struct Clock {
    frozen: bool,
    started: Instant,
}

impl Clock {
    /// Seconds since the Unix epoch, with a fraction.
    fn now(&self) -> f64 {
        MOCK_NOW as f64 + if self.frozen { 0. } else { self.started.elapsed().as_secs_f64() }
    }
}

/// How a cloud run ended early.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ended {
    /// Stopped through the API: the run keeps its checkpoint.
    Stopped,
    /// Its instance was terminated by hand: the run fails with `cloud_instance_lost`.
    InstanceLost,
}

/// A cloud run's life (docs/CLOUD.md, *A cloud run, step by step*) as a function of the clock:
/// provisioning, booting and uploading, then the training curve on the instance, then collecting
/// the weights and releasing the instance. Everything is measured in seconds since the instance
/// was asked for.
#[derive(Clone)]
struct CloudSim {
    /// `Run.cloud` as the run started: connection, offer, instance and time limit.
    base: RunCloud,
    clock: Clock,
    /// When the instance was asked for (the run's creation), Unix seconds.
    t0: f64,
    /// How long provisioning, booting and uploading take.
    pre: [f64; 3],
    /// The dataset's upload, bytes.
    upload_bytes: u64,
    /// Seconds per training step.
    step_dt: f64,
    /// Where the training clock is anchored: (seconds since `t0`, steps done then).
    anchor: (f64, u64),
    /// The run's length in steps.
    total: u64,
    /// Ended early: when, and how.
    ended: Option<(f64, Ended)>,
    /// Why the provider hasn't confirmed the termination (the instance stays `terminating`).
    release_error: Option<String>,
}

/// Seconds of collecting the weights, then of releasing the instance.
const COLLECT_S: f64 = 1.;
const RELEASE_S: f64 = 0.5;

impl CloudSim {
    /// A run launched as `base` says at `t0` (Unix seconds) that trains `total` steps, at the
    /// demo's pace: about 2 s for each phase before training, then as fast as a local mock run.
    fn new(base: RunCloud, clock: Clock, t0: f64, total: u64) -> Self {
        let pre = [2., 2., 2.];
        Self {
            base,
            clock,
            t0,
            pre,
            upload_bytes: 600 << 20,
            step_dt: STEP_DT,
            anchor: (pre.iter().sum(), 0),
            total,
            ended: None,
            release_error: None,
        }
    }

    fn elapsed(&self) -> f64 {
        (self.clock.now() - self.t0).max(0.)
    }

    fn train_at(&self) -> f64 {
        self.pre.iter().sum()
    }

    /// Training steps done `t` seconds in.
    fn steps_at(&self, t: f64) -> u64 {
        let t = self.ended.map_or(t, |(e, _)| t.min(e));
        if t < self.train_at() {
            return 0;
        }
        (self.anchor.1 + ((t - self.anchor.0).max(0.) / self.step_dt) as u64).min(self.total)
    }

    fn steps(&self) -> u64 {
        self.steps_at(self.elapsed())
    }

    /// When training ended (all steps done, or stopped), if it has.
    fn trained_at(&self) -> Option<f64> {
        let natural = self.anchor.0 + self.total.saturating_sub(self.anchor.1) as f64 * self.step_dt;
        let end = self.ended.map_or(natural, |(e, _)| e.min(natural)).max(self.train_at());
        (self.elapsed() >= end).then_some(end)
    }

    /// When the instance was released: right away for a run that ended before training or lost its
    /// instance, else after collecting the weights.
    fn released_at(&self) -> Option<f64> {
        match self.ended {
            Some((e, how)) if how == Ended::InstanceLost || e < self.train_at() => Some(e),
            _ => self.trained_at().map(|t| t + COLLECT_S + RELEASE_S).filter(|r| self.elapsed() >= *r),
        }
    }

    fn phase(&self) -> (CloudPhase, Option<TransferProgress>) {
        let t = self.elapsed();
        let [provision, boot, sync] = self.pre;
        if self.released_at().is_some() {
            return (CloudPhase::Released, None);
        }
        if let Some(done) = self.trained_at() {
            return if t < done + COLLECT_S {
                // The two weights files, about 40 MB each for these models.
                let total = 80 << 20;
                let done_bytes = (total as f64 * (t - done) / COLLECT_S) as u64;
                (CloudPhase::Collecting, Some(TransferProgress { done_bytes, total_bytes: total }))
            } else {
                (CloudPhase::Releasing, None)
            };
        }
        if t < provision {
            (CloudPhase::Provisioning, None)
        } else if t < provision + boot {
            (CloudPhase::Booting, None)
        } else if t < self.train_at() {
            let done_bytes = (self.upload_bytes as f64 * (t - provision - boot) / sync) as u64;
            (CloudPhase::Syncing, Some(TransferProgress { done_bytes, total_bytes: self.upload_bytes }))
        } else {
            (CloudPhase::Training, None)
        }
    }

    /// Whether the provider hasn't confirmed the termination: the instance stays `terminating` (and
    /// billed) after the run was released.
    fn stuck(&self) -> bool {
        self.release_error.is_some() && self.trained_at().is_some_and(|t| self.elapsed() >= t + COLLECT_S)
    }

    /// Seconds the instance has existed: until it was released, or until now while it isn't.
    fn lifetime(&self) -> f64 {
        if self.stuck() { self.elapsed() } else { self.released_at().unwrap_or(self.elapsed()) }
    }

    /// Price times `secs`, without a price none.
    fn cost(&self, secs: f64) -> Option<f64> {
        self.base.offer.price_per_hour.map(|p| p * secs / 3600.)
    }

    fn at(&self, secs: f64) -> String {
        time::format((self.t0 + secs) as i64)
    }

    /// `Run.cloud` now.
    fn state(&self) -> RunCloud {
        let (phase, progress) = self.phase();
        let t = self.elapsed().min(self.ended.map_or(f64::MAX, |e| e.0));
        let train_at = self.train_at();
        // `last.folo` is copied every 10 minutes of training, and both files at the end.
        let synced = match self.trained_at() {
            Some(done) if phase >= CloudPhase::Releasing => Some(done + COLLECT_S),
            _ if t >= train_at + 600. => Some(train_at + ((t - train_at) / 600.).floor() * 600.),
            _ => None,
        };
        RunCloud {
            phase,
            progress,
            deadline: (self.elapsed() >= self.pre[0]).then(|| self.at(self.base.max_hours as f64 * 3600.)),
            // Final once released, even while the provider hasn't confirmed the termination.
            cost_usd: self.cost(self.released_at().unwrap_or(self.elapsed())),
            synced: synced.map(|s| self.at(s)),
            ..self.base.clone()
        }
    }

    /// The run as its life says now: status, times, device and `cloud`.
    fn apply(&self, run: &mut Run) {
        let cloud = self.state();
        let lost = self.ended.is_some_and(|(_, how)| how == Ended::InstanceLost);
        let stopped = self.ended.is_some_and(|(_, how)| how == Ended::Stopped);
        run.status = match cloud.phase {
            _ if lost => RunStatus::Failed,
            CloudPhase::Released if stopped => RunStatus::Stopped,
            CloudPhase::Released => RunStatus::Completed,
            _ if stopped => RunStatus::Stopping,
            p if p >= CloudPhase::Training => RunStatus::Running,
            _ => RunStatus::Queued,
        };
        let training =
            self.elapsed() >= self.train_at() && !self.ended.is_some_and(|(e, _)| e < self.train_at());
        run.started = training.then(|| self.at(self.train_at()));
        run.finished = self.released_at().map(|r| self.at(r));
        // The worker's GPU once it trains (the offer's catalogue name, as wgpu reports it).
        run.device = if training {
            format!("{} (vulkan)", self.base.offer.id.split('@').next().unwrap_or(&self.base.offer.gpu))
        } else {
            String::new()
        };
        if lost {
            run.error = Some(format!(
                "The instance {} was terminated by hand; the checkpoints copied so far are kept.",
                self.base.instance_id.as_deref().unwrap_or("of this run")
            ));
            run.error_code = Some("cloud_instance_lost".into());
        }
        if !run.status.is_active() {
            run.eta_s = None;
        }
        run.cloud = Some(cloud);
    }

    /// The run's instance as `/api/cloud/instances` lists it.
    fn instance(&self, run_id: &str) -> Option<CloudInstance> {
        let id = self.base.instance_id.clone()?;
        let (phase, _) = self.phase();
        let stuck = self.stuck();
        let t = self.elapsed();
        let status = match phase {
            _ if stuck => CloudInstanceStatus::Terminating,
            CloudPhase::Released => CloudInstanceStatus::Terminated,
            CloudPhase::Releasing => CloudInstanceStatus::Terminating,
            CloudPhase::Provisioning => CloudInstanceStatus::Launching,
            CloudPhase::Booting => CloudInstanceStatus::Starting,
            _ => CloudInstanceStatus::Running,
        };
        Some(CloudInstance {
            id,
            connection_id: self.base.connection_id.clone(),
            provider: self.base.provider.clone(),
            provider_ref: (t >= self.pre[0])
                .then(|| format!("{:016x}", hash(&format!("pod{run_id}")))[..14].to_string()),
            name: format!("folo-{run_id}-{:04x}", hash(run_id) & 0xFFFF),
            run_id: Some(run_id.to_string()),
            offer: self.base.offer.clone(),
            status,
            created: self.at(0.),
            ended: (status == CloudInstanceStatus::Terminated).then(|| self.at(self.lifetime())),
            deadline: (t >= self.pre[0]).then(|| self.at(self.base.max_hours as f64 * 3600.)),
            cost_usd: self.cost(self.lifetime()),
            error: self.release_error.clone().filter(|_| stuck),
        })
    }
}

/// `Run.cloud` as a run is launched on `offer`: phase and costs follow from its [`CloudSim`]. A
/// provider that provisions gets an instance per run.
fn launch_state(
    run_id: &str,
    compute: &Compute,
    provider: &str,
    offer: CloudOffer,
    provisions: bool,
) -> RunCloud {
    RunCloud {
        connection_id: compute.connection_id.clone(),
        provider: provider.into(),
        offer,
        instance_id: provisions.then(|| format!("ci_{:04x}", hash(run_id) & 0xFFFF)),
        phase: CloudPhase::Provisioning,
        progress: None,
        max_hours: compute.max_hours,
        deadline: None,
        cost_usd: None,
        synced: None,
    }
}

/// The demo's one RunPod connection.
const RUNPOD_CONNECTION: &str = "cc_4d2a";
const RTX4090_OFFER: &str = "NVIDIA GeForce RTX 4090@secure";
const L40S_OFFER: &str = "NVIDIA L40S@secure";
const A100_OFFER: &str = "NVIDIA A100 80GB PCIe@secure";

/// The providers as `folo serve` lists them (docs/CLOUD.md, *Providers*).
fn cloud_providers() -> Vec<CloudProvider> {
    let field =
        |key: &str, label: &str, secret: bool, required: bool, placeholder: Option<&str>, help: &str| {
            CloudField {
                key: key.into(),
                label: label.into(),
                secret,
                required,
                placeholder: placeholder.map(Into::into),
                help: Some(help.into()),
            }
        };
    vec![
        CloudProvider {
            id: "runpod".into(),
            name: "RunPod".into(),
            about: "Rents a GPU pod by the hour for each run and terminates it when the run ends.".into(),
            docs_url: Some("https://www.runpod.io/console/user/settings".into()),
            fields: vec![
                field(
                    "api_key",
                    "API key",
                    true,
                    true,
                    Some("rpa_…"),
                    "RunPod console › Settings › API Keys, with write access to pods.",
                ),
                field(
                    "worker_base_url",
                    "Worker download URL",
                    false,
                    false,
                    Some("https://github.com/himent12/folo/releases"),
                    "Where pods download FOLO from; leave empty for this version's release.",
                ),
            ],
            provisions: true,
            self_terminates: true,
        },
        CloudProvider {
            id: "remote".into(),
            name: "FOLO server".into(),
            about: "Another machine of yours that runs folo serve with a token; runs wait in its own queue."
                .into(),
            docs_url: None,
            fields: vec![
                field(
                    "url",
                    "Server URL",
                    false,
                    true,
                    Some("https://gpu-box.example.com:7878"),
                    "Use https:// across networks you don't control.",
                ),
                field("token", "Token", true, true, None, "The --token that server was started with."),
            ],
            provisions: false,
            self_terminates: false,
        },
    ]
}

/// RunPod's catalogue as the demo has it, cheapest first: both clouds, one GPU out of stock and one
/// low on it.
fn runpod_offers() -> Vec<CloudOffer> {
    let gib = 1u64 << 30;
    let offer = |id: &str, gpu: &str, vram: u64, price: f64, availability: CloudAvailability| CloudOffer {
        id: id.into(),
        gpu: gpu.into(),
        gpu_count: 1,
        vram_bytes: vram * gib,
        price_per_hour: Some(price),
        region: None,
        detail: Some(if id.ends_with("@secure") { "Secure Cloud" } else { "Community Cloud" }.into()),
        availability,
    };
    vec![
        offer("NVIDIA RTX A4000@community", "RTX A4000", 16, 0.17, CloudAvailability::High),
        offer("NVIDIA GeForce RTX 3090@community", "RTX 3090", 24, 0.22, CloudAvailability::Medium),
        offer("NVIDIA GeForce RTX 4090@community", "RTX 4090", 24, 0.34, CloudAvailability::None),
        offer(RTX4090_OFFER, "RTX 4090", 24, 0.69, CloudAvailability::High),
        offer("NVIDIA RTX 6000 Ada Generation@secure", "RTX 6000 Ada", 48, 0.77, CloudAvailability::Low),
        offer(L40S_OFFER, "L40S", 48, 0.86, CloudAvailability::Medium),
        offer(A100_OFFER, "A100 80GB", 80, 1.64, CloudAvailability::Medium),
        offer("NVIDIA H100 80GB HBM3@secure", "H100 SXM", 80, 2.99, CloudAvailability::Low),
    ]
}

/// The demo's RunPod connection, checked 20 minutes ago.
fn runpod_connection() -> CloudConnection {
    CloudConnection {
        id: RUNPOD_CONNECTION.into(),
        provider: "runpod".into(),
        name: "RunPod".into(),
        created: time::format(MOCK_NOW - 3 * 86_400),
        status: CloudConnectionStatus::Ok,
        account: Some("you@example.com".into()),
        balance_usd: Some(42.17),
        error: None,
        note: None,
        checked: Some(time::format(MOCK_NOW - 20 * 60)),
        values: [("api_key".to_string(), "••••a1b2".to_string())].into(),
    }
}

/// A secret as answers show it: `••••` and its last 4 characters.
fn secret_hint(s: &str) -> String {
    let tail: String = s.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("••••{tail}")
}

/// An `http://` URL to a host that isn't loopback or a private address: the connection's `note`
/// warns about it.
fn unencrypted(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else { return false };
    let host = rest.split(['/', ':']).next().unwrap_or_default();
    let private = ["localhost", "127.", "10.", "192.168.", "["].iter().any(|p| host.starts_with(p))
        || host
            .strip_prefix("172.")
            .and_then(|r| r.split('.').next()?.parse::<u8>().ok())
            .is_some_and(|b| (16..32).contains(&b));
    !private
}

/// What a connection can run on: RunPod's catalogue, or a remote server's own GPU (not billed).
/// A connection whose last check failed can't be asked.
fn offers_of(conn: &CloudConnection) -> ApiResult<Vec<CloudOffer>> {
    if conn.status == CloudConnectionStatus::Error {
        return Err(ApiError::new("cloud_auth_failed", conn.error.clone().unwrap_or_default())
            .with_hint("Check the connection again in Settings › Cloud."));
    }
    if conn.provider == "runpod" {
        return Ok(runpod_offers());
    }
    Ok(vec![CloudOffer {
        id: "gpu:0".into(),
        gpu: "RTX 4080 SUPER".into(),
        gpu_count: 1,
        vram_bytes: 16 << 30,
        price_per_hour: None,
        region: None,
        detail: conn.account.clone(),
        availability: CloudAvailability::Unknown,
    }])
}

fn no_connection(id: &str) -> ApiError {
    ApiError::new("connection_not_found", format!("No cloud connection {id}"))
        .with_hint("Settings › Cloud lists the connections.")
}

fn no_instance(id: &str) -> ApiError {
    ApiError::new("instance_not_found", format!("No cloud instance {id}"))
        .with_hint("Settings › Cloud lists the instances.")
}

type PixelCache = Mutex<HashMap<(String, String, u32), Arc<Vec<u8>>>>;

pub struct MockBackend {
    datasets: Vec<DatasetFixture>,
    models: Vec<Model>,
    runs: Arc<Mutex<Vec<RunFixture>>>,
    frozen: bool,
    started: Instant,
    image_cache: PixelCache,
    /// Inject an error into every list call (error-state screenshots).
    fail: bool,
    /// Pretend there is nothing on disk yet (empty-state screenshots).
    empty: bool,
    /// The scripted runs: they don't block a benchmark (runs created through the API do, as in
    /// `folo serve --mock`).
    fixture_runs: Vec<String>,
    /// What the demo "measured": training and inference rows.
    measured: Mutex<(Vec<TrainThroughput>, Vec<InferThroughput>)>,
    /// Cloud connections (one RunPod connection unless a variant says otherwise).
    connections: Mutex<Vec<CloudConnection>>,
    clock: Clock,
}

const LIVE_ID: &str = "run_a19f";

impl MockBackend {
    /// `live_step`: where the live run starts. `frozen`: keep the clock still (screenshots).
    pub fn new(live_step: u32, frozen: bool) -> Self {
        let h = |hours: f64| MOCK_NOW - (hours * 3600.0) as i64;
        let datasets = vec![
            DatasetFixture::new(
                "ds_8f2c",
                "shapes",
                "D:/data/shapes",
                "yolo",
                &["circle", "square", "triangle"],
                (800, 200, 0),
                6,
                false,
                &[(640, 640)],
                h(70.0),
            ),
            DatasetFixture::new(
                "ds_3b91",
                "shapes-hard",
                "D:/data/shapes-hard",
                "coco",
                &["circle", "square", "triangle", "star", "hexagon"],
                (3200, 800, 400),
                12,
                true,
                &[(640, 480), (800, 600), (480, 640), (640, 640)],
                h(30.0),
            ),
            DatasetFixture::new(
                "ds_c07a",
                "shapes-mini",
                "D:/data/shapes-mini",
                "yolo",
                &["circle", "square", "triangle"],
                (240, 60, 0),
                4,
                false,
                &[(320, 320)],
                h(4.0),
            ),
            DatasetFixture::new(
                "ds_5e21",
                "shapes-crowded",
                "D:/data/shapes-crowded",
                "yolo",
                &["circle", "square", "triangle", "star", "hexagon"],
                (1600, 400, 200),
                24,
                false,
                &[(640, 640), (800, 600)],
                h(200.0),
            ),
            DatasetFixture::new(
                "ds_a3d0",
                "shapes-tiny",
                "D:/data/shapes-tiny",
                "yolo",
                &["circle", "square", "triangle"],
                (1200, 300, 0),
                10,
                true,
                &[(512, 512)],
                h(120.0),
            ),
        ];
        let classes = |i: usize| datasets[i].ds.classes.clone();
        let model = |id: &str,
                     name: &str,
                     arch: &str,
                     imgsz,
                     ds: usize,
                     params,
                     flops_g,
                     bytes,
                     hours,
                     run: &str,
                     m: [f32; 4]| Model {
            id: id.into(),
            name: name.into(),
            arch: Arch::parse(arch).unwrap_or_default(),
            imgsz,
            classes: classes(ds),
            params,
            flops_g,
            bytes,
            created: time::format(h(hours)),
            run_id: Some(run.into()),
            metrics: Some(ModelMetrics {
                map50: m[0],
                map50_95: m[1],
                precision: m[2],
                recall: m[3],
                // Where F1 peaks: lower for the stronger models (their scores sit higher).
                best_f1_conf: Some(((0.42 - 0.18 * m[1]) * 100.).round() / 100.),
            }),
            head: folo_api::Head::O2m,
            trained_with: None,
            precision: Some(folo_api::ModelPrecision::F32),
            fused: Some(false),
            source: Some(folo_api::ModelSource::Run),
            about: None,
            path: None,
        };
        let mut models = vec![
            model(
                "m_44c0",
                "shapes-hard-s-best",
                "folo-s",
                640,
                1,
                9_420_000,
                21.5,
                18_900_000,
                6.5,
                "run_52e8",
                [0.874, 0.624, 0.861, 0.792],
            ),
            model(
                "m_31aa",
                "shapes-n-best",
                "folo-n",
                640,
                0,
                2_630_000,
                6.6,
                5_400_000,
                52.0,
                "run_7d1e",
                [0.952, 0.781, 0.928, 0.903],
            ),
            model(
                "m_31ab",
                "shapes-n-f16",
                "folo-n",
                640,
                0,
                2_630_000,
                6.6,
                2_700_000,
                51.5,
                "run_7d1e",
                [0.951, 0.780, 0.927, 0.903],
            ),
            model(
                "m_90d2",
                "shapes-mini-n-last",
                "folo-n",
                320,
                2,
                2_630_000,
                1.7,
                5_400_000,
                3.2,
                "run_1a2b",
                [0.803, 0.544, 0.817, 0.716],
            ),
            model(
                "m_7c11",
                "shapes-crowded-s-best",
                "folo-s",
                640,
                3,
                9_420_000,
                21.5,
                18_900_000,
                128.0,
                "run_c9f1",
                [0.801, 0.553, 0.790, 0.742],
            ),
            model(
                "m_7c12",
                "shapes-tiny-n-best",
                "folo-n",
                512,
                4,
                2_630_000,
                4.3,
                5_400_000,
                98.0,
                "run_d204",
                [0.661, 0.402, 0.712, 0.584],
            ),
        ];
        // shapes-hard-s-best comes from the end-to-end, mixed-precision run below: its record says so.
        if let Some(m) = models.iter_mut().find(|m| m.id == "m_44c0") {
            m.head = folo_api::Head::O2o;
            m.trained_with = Some(folo_api::TrainedWith { amp: true, imgsz: 640, epochs: 120 });
        }
        // shapes-n-f16 is the fused f16 export of shapes-n-best.
        if let Some(m) = models.iter_mut().find(|m| m.id == "m_31ab") {
            m.precision = Some(folo_api::ModelPrecision::F16);
            m.fused = Some(true);
        }

        let cfg = |name: &str, ds: &str, arch: &str, epochs, batch, imgsz, device: &str, lr0| TrainConfig {
            name: name.into(),
            dataset_id: ds.into(),
            arch: Arch::parse(arch).unwrap_or_default(),
            epochs,
            batch,
            imgsz,
            device: device.into(),
            optimizer: Optimizer::Auto,
            lr0,
            weights: None,
            augment: Augment::default(),
            seed: 0,
            amp: false,
            head: Default::default(),
            compute: None,
        };
        // Trains on a cloud connection's offer (the cloud fixtures below).
        let on = |offer: &str, max_hours: f32, c: TrainConfig| TrainConfig {
            compute: Some(Compute {
                connection_id: RUNPOD_CONNECTION.into(),
                offer_id: offer.into(),
                max_hours,
                disk_gb: None,
            }),
            device: "auto".into(),
            ..c
        };
        let gpu = GPU_DEVICE.to_string();
        struct Spec {
            id: &'static str,
            config: TrainConfig,
            status: RunStatus,
            step: u64,
            spe: u64,
            created_h: f64,
            map_max: f32,
            ips: f32,
            model: Option<&'static str>,
            error: Option<&'static str>,
        }
        let specs = vec![
            Spec {
                id: LIVE_ID,
                config: cfg("shapes-hard-m", "ds_3b91", "folo-m", 100, 32, 640, "gpu:0", 0.01),
                status: RunStatus::Running,
                step: live_step as u64,
                spe: 100,
                created_h: 0.25,
                map_max: 0.71,
                ips: 268.0,
                model: None,
                error: None,
            },
            Spec {
                id: "run_b7e2",
                config: cfg("shapes-mini-s", "ds_c07a", "folo-s", 60, 16, 320, "gpu:0", 0.01),
                status: RunStatus::Queued,
                step: 0,
                spe: 19,
                created_h: 0.1,
                map_max: 0.6,
                ips: 900.0,
                model: None,
                error: None,
            },
            Spec {
                id: "run_52e8",
                // The end-to-end head (no NMS) in mixed precision, so its model shows both.
                config: TrainConfig {
                    head: Head::O2o,
                    amp: true,
                    ..cfg("shapes-hard-s", "ds_3b91", "folo-s", 120, 32, 640, "gpu:0", 0.01)
                },
                status: RunStatus::Completed,
                step: 120 * 100,
                spe: 100,
                created_h: 9.0,
                map_max: 0.612,
                ips: 412.0,
                model: Some("m_44c0"),
                error: None,
            },
            Spec {
                id: "run_e4f0",
                config: cfg("shapes-hard-l", "ds_3b91", "folo-l", 100, 64, 1024, "gpu:0", 0.01),
                status: RunStatus::Failed,
                step: 112,
                spe: 50,
                created_h: 10.5,
                map_max: 0.7,
                ips: 96.0,
                model: None,
                error: Some(
                    "GPU out of memory: tried to allocate 2.1 GiB with 0.6 GiB free (batch 64, imgsz 1024)",
                ),
            },
            Spec {
                id: "run_1a2b",
                config: cfg("shapes-mini-n", "ds_c07a", "folo-n", 100, 16, 320, "cpu", 0.01),
                status: RunStatus::Stopped,
                step: 37 * 15,
                spe: 15,
                created_h: 4.0,
                map_max: 0.58,
                ips: 140.0,
                model: Some("m_90d2"),
                error: None,
            },
            Spec {
                id: "run_7d1e",
                config: cfg("shapes-n", "ds_8f2c", "folo-n", 100, 16, 640, "gpu:0", 0.01),
                status: RunStatus::Completed,
                step: 100 * 50,
                spe: 50,
                created_h: 55.0,
                map_max: 0.781,
                ips: 610.0,
                model: Some("m_31aa"),
                error: None,
            },
            // Cloud runs (docs/CLOUD.md): one uploading its dataset, one training, one released.
            Spec {
                id: "run_c1d0",
                config: on(
                    L40S_OFFER,
                    24.,
                    cfg("shapes-crowded-l", "ds_5e21", "folo-l", 120, 32, 800, "auto", 0.01),
                ),
                status: RunStatus::Queued,
                step: 0,
                spe: 50,
                created_h: 0.1,
                map_max: 0.62,
                ips: 120.0,
                model: None,
                error: None,
            },
            Spec {
                id: "run_c2e1",
                // The out-of-memory run above, again on a 24 GB GPU.
                config: on(
                    RTX4090_OFFER,
                    12.,
                    cfg("shapes-hard-l-2", "ds_3b91", "folo-l", 100, 32, 1024, "auto", 0.01),
                ),
                status: RunStatus::Running,
                step: 4400,
                spe: 100,
                created_h: 2.4,
                map_max: 0.7,
                ips: 96.0,
                model: None,
                error: None,
            },
            Spec {
                id: "run_c9f1",
                config: on(
                    A100_OFFER,
                    12.,
                    cfg("shapes-crowded-s", "ds_5e21", "folo-s", 80, 32, 640, "auto", 0.01),
                ),
                status: RunStatus::Completed,
                step: 80 * 50,
                spe: 50,
                created_h: 131.0,
                map_max: 0.553,
                ips: 398.0,
                model: Some("m_7c11"),
                error: None,
            },
            Spec {
                id: "run_d204",
                config: cfg("shapes-tiny-n", "ds_a3d0", "folo-n", 60, 32, 512, "gpu:0", 0.01),
                status: RunStatus::Completed,
                step: 60 * 38,
                spe: 38,
                created_h: 101.0,
                map_max: 0.402,
                ips: 720.0,
                model: Some("m_7c12"),
                error: None,
            },
            Spec {
                id: "run_0b5e",
                config: cfg("shapes-crowded-m", "ds_5e21", "folo-m", 120, 32, 800, "gpu:0", 0.01),
                status: RunStatus::Stopped,
                step: 23 * 50,
                spe: 50,
                created_h: 30.0,
                map_max: 0.6,
                ips: 241.0,
                model: None,
                error: None,
            },
        ];
        let now = Instant::now();
        // Each run resolved against its own dataset's class count (Auto's AdamW rate depends on it).
        let nc = |ds: &str| datasets.iter().find(|d| d.ds.id == ds).map_or(5, |d| d.ds.classes.len());
        let mut runs = specs
            .into_iter()
            .map(|s| {
                let curve = Curve {
                    seed: hash(s.id),
                    epochs: s.config.epochs,
                    spe: s.spe,
                    // The rate the run resolves to (Auto's pick), not the form's.
                    lr0: resolve(&s.config, s.spe * s.config.batch as u64, nc(&s.config.dataset_id)).lr0
                        as f32,
                    batch: s.config.batch,
                    map_max: s.map_max,
                    ips: s.ips,
                    classes: datasets
                        .iter()
                        .find(|d| d.ds.id == s.config.dataset_id)
                        .map_or(3, |d| d.ds.classes.len() as u32),
                    close_mosaic: s.config.augment.close_mosaic,
                };
                let created = h(s.created_h);
                let total = s.config.epochs as u64 * s.spe;
                let run_secs = |steps: u64| (steps as f64 * (s.config.batch as f64 / s.ips as f64)) as i64;
                let finished =
                    matches!(s.status, RunStatus::Completed | RunStatus::Stopped | RunStatus::Failed).then(
                        || time::format(created + 5 + run_secs(s.step) + s.step as i64 / s.spe as i64 * 3),
                    );
                let started = (s.status != RunStatus::Queued).then(|| time::format(created + 5));
                let mut fixture = RunFixture {
                    run: Run {
                        id: s.id.into(),
                        name: s.config.name.clone(),
                        status: s.status,
                        created: time::format(created),
                        started,
                        finished,
                        epoch: 0,
                        epochs: s.config.epochs,
                        step: s.step,
                        steps_per_epoch: s.spe,
                        eta_s: None,
                        best: None,
                        device: if s.config.device == "cpu" { CPU_DEVICE.into() } else { gpu.clone() },
                        model_id: s.model.map(Into::into),
                        error: s.error.map(Into::into),
                        error_code: s.error.map(|_| "training_failed".into()),
                        resolved: (s.status != RunStatus::Queued).then(|| {
                            resolve(&s.config, s.spe * s.config.batch as u64, nc(&s.config.dataset_id))
                        }),
                        queue_position: (s.status == RunStatus::Queued).then_some(1),
                        cloud: None,
                        config: s.config,
                    },
                    curve,
                    live: (s.status == RunStatus::Running && !frozen).then_some((s.step, now)),
                    step: s.step.min(total),
                    logs: Vec::new(),
                    cloud: None,
                };
                fixture.logs = initial_logs(&fixture, created);
                fixture
            })
            .collect::<Vec<RunFixture>>();
        // The cloud fixtures live on their instances: 6 minutes into uploading 4.8 GiB, 2.4 hours
        // into training (at step 4,400), and released 5 days ago after about an hour on an A100.
        let clock = Clock { frozen, started: now };
        let gib = 1u64 << 30;
        for f in &mut runs {
            let Some(compute) = f.run.config.compute.clone() else { continue };
            let t0 = time::parse(&f.run.created).unwrap_or(MOCK_NOW);
            let total = f.run.epochs as u64 * f.run.steps_per_epoch;
            let offer = runpod_offers().into_iter().find(|o| o.id == compute.offer_id);
            let Some(offer) = offer else { continue };
            let mut sim = CloudSim::new(
                launch_state(&f.run.id, &compute, "runpod", offer, true),
                clock,
                t0 as f64,
                total,
            );
            match f.run.id.as_str() {
                "run_c1d0" => (sim.pre, sim.upload_bytes) = ([40., 150., 680.], gib * 24 / 5),
                "run_c2e1" => {
                    (sim.pre, sim.upload_bytes) = ([45., 230., 290.], gib * 21 / 10);
                    sim.anchor = ((MOCK_NOW - t0) as f64, f.step);
                }
                _ => (sim.pre, sim.upload_bytes, sim.step_dt) = ([45., 240., 360.], gib * 13 / 10, 0.85),
            }
            // Unless anchored mid-run, training starts from step 0 when the upload ends.
            if sim.anchor.1 == 0 {
                sim.anchor.0 = sim.train_at();
            }
            f.cloud = Some(sim);
            f.live = None;
            f.run.queue_position = None;
            f.logs = initial_logs(f, t0);
        }
        // A model's operating point is its run's best epoch's (as `folo serve --mock`'s), so the run and model
        // pages name the same confidence.
        for model in &mut models {
            let Some(f) = runs.iter().find(|f| Some(&*f.run.id) == model.run_id.as_deref()) else { continue };
            let (Some(best), Some(metrics)) = (Self::snapshot(f).best, model.metrics.as_mut()) else {
                continue;
            };
            metrics.best_f1_conf = f.curve.epoch(best.epoch).val.best_f1_conf;
        }
        let fixture_runs = runs.iter().map(|f| f.run.id.clone()).collect();
        Self {
            datasets,
            models,
            fixture_runs,
            measured: Mutex::new(mock_measured()),
            runs: Arc::new(Mutex::new(runs)),
            frozen,
            started: now,
            image_cache: Mutex::new(HashMap::new()),
            fail: false,
            empty: false,
            connections: Mutex::new(vec![runpod_connection()]),
            clock,
        }
    }

    pub fn failing(mut self) -> Self {
        self.fail = true;
        self
    }

    pub fn empty(mut self) -> Self {
        self.empty = true;
        self
    }

    /// A server whose cloud was never used: no connection, no cloud run, no instance (Settings ›
    /// Cloud's empty state). The released cloud run trained on this machine instead.
    pub fn without_cloud(self) -> Self {
        self.connections.lock().unwrap().clear();
        let mut runs = self.runs.lock().unwrap();
        runs.retain(|f| f.cloud.as_ref().is_none_or(|s| s.released_at().is_some()));
        for f in runs.iter_mut().filter(|f| f.cloud.is_some()) {
            f.cloud = None;
            f.run.config.compute = None;
            f.run.config.device = "gpu:0".into();
            f.run.device = GPU_DEVICE.into();
            f.logs = initial_logs(f, time::parse(&f.run.created).unwrap_or(MOCK_NOW));
        }
        drop(runs);
        self
    }

    /// The RunPod key revoked, and the training cloud run stopped ten minutes ago with a termination
    /// RunPod hasn't confirmed: Settings › Cloud's problem states.
    pub fn cloud_trouble(self) -> Self {
        if let Some(c) = self.connections.lock().unwrap().first_mut() {
            c.status = CloudConnectionStatus::Error;
            c.balance_usd = None;
            c.error = Some(
                "RunPod rejected the API key (401 Unauthorized): it was revoked or lost write access to pods."
                    .into(),
            );
            c.checked = Some(time::format(MOCK_NOW - 4 * 60));
        }
        for f in self.runs.lock().unwrap().iter_mut() {
            if let Some(sim) =
                f.cloud.as_mut().filter(|s| s.elapsed() >= s.train_at() && s.released_at().is_none())
            {
                sim.ended = Some((sim.elapsed() - 600., Ended::Stopped));
                sim.release_error = Some("RunPod's API answered 502 Bad Gateway".into());
            }
        }
        self
    }

    fn check(&self) -> ApiResult<()> {
        if self.fail {
            Err(ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878")
                .with_hint("Start it with `folo serve`, or choose Built-in demo in Settings."))
        } else {
            Ok(())
        }
    }

    fn fixture(&self, id: &str) -> ApiResult<&DatasetFixture> {
        self.datasets
            .iter()
            .find(|d| d.ds.id == id)
            .ok_or_else(|| ApiError::new("dataset_not_found", format!("No dataset with id {id}")))
    }

    /// Current step of a run fixture, advancing live runs with the clock.
    fn current_step(f: &RunFixture) -> u64 {
        let total = f.run.epochs as u64 * f.run.steps_per_epoch;
        if let Some(sim) = &f.cloud {
            return sim.steps().min(total);
        }
        match f.live {
            Some((base, at)) => (base + (at.elapsed().as_secs_f64() / STEP_DT) as u64).min(total),
            None => f.step.min(total),
        }
    }

    fn snapshot(f: &RunFixture) -> Run {
        let mut run = f.run.clone();
        if let Some(sim) = &f.cloud {
            sim.apply(&mut run);
        }
        let step = Self::current_step(f);
        let total = run.epochs as u64 * run.steps_per_epoch;
        run.step = step;
        run.epoch =
            if step == 0 { 0 } else { (((step - 1) / run.steps_per_epoch + 1) as u32).min(run.epochs) };
        if f.live.is_some() && step >= total {
            run.status = RunStatus::Completed;
        }
        if run.status == RunStatus::Running {
            run.eta_s = Some((total - step) as f64 * STEP_DT + (run.epochs - run.epoch) as f64 * 0.4);
        }
        let done_epochs = (step / run.steps_per_epoch) as u32;
        run.best = (1..=done_epochs)
            .map(|e| (e, f.curve.epoch(e).val.map50_95))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(epoch, map50_95)| BestEpoch { epoch, map50_95 });
        run
    }

    fn sim_now(&self) -> i64 {
        if self.frozen { MOCK_NOW } else { MOCK_NOW + self.started.elapsed().as_secs() as i64 }
    }

    fn val_images(&self, dataset_id: &str) -> Vec<(u32, &ImageInfo)> {
        let Ok(f) = self.fixture(dataset_id) else { return Vec::new() };
        f.images
            .iter()
            .enumerate()
            .filter(|(_, im)| im.split == Split::Val && !im.boxes.is_empty())
            .map(|(i, im)| (i as u32, im))
            .collect()
    }
}

/// An epoch's summary line in a run's log.
fn epoch_line(e: u32, epochs: u32, ep: &Epoch) -> String {
    format!(
        "epoch {e}/{epochs} · mAP50-95 {:.3} · mAP50 {:.3} · {:.1} s",
        ep.val.map50_95, ep.val.map50, ep.duration_s
    )
}

fn initial_logs(f: &RunFixture, created: i64) -> Vec<LogLine> {
    let c = &f.run.config;
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
            "optimizer: AdamW(lr=0.01, wd=5e-4), cosine schedule, warmup 3 epochs".to_string(),
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

impl Backend for MockBackend {
    fn describe(&self) -> String {
        // The failing variant stands in for an unreachable folo serve (the error scenario): it
        // names the address the app tried, as the HTTP backend would.
        if self.fail { "127.0.0.1:7878".into() } else { "Built-in demo".into() }
    }

    fn now(&self) -> i64 {
        self.sim_now()
    }

    fn health(&self) -> ApiResult<Health> {
        self.check()?;
        Ok(Health {
            status: "ok".into(),
            version: "0.1.0".into(),
            api: folo_api::API_VERSION.into(),
            mock: true,
        })
    }

    fn system(&self) -> ApiResult<SystemResponse> {
        self.check()?;
        let t = self.started.elapsed().as_secs_f32();
        let wobble = if self.frozen { 0.0 } else { (t * 0.7).sin() * 0.02 };
        // A run on this machine's GPU (a cloud run trains elsewhere).
        let live = !self.empty
            && self
                .runs
                .lock()
                .unwrap()
                .iter()
                .any(|f| f.cloud.is_none() && Self::snapshot(f).status == RunStatus::Running);
        let system = SystemInfo {
            cpu: CpuInfo {
                name: "AMD Ryzen 9 7950X".into(),
                threads: 32,
                util: if live { 0.31 + wobble } else { 0.04 },
            },
            memory: MemoryInfo { used_bytes: 23_100_000_000, total_bytes: 68_719_476_736 },
            gpus: vec![GpuInfo {
                ordinal: 0,
                name: "NVIDIA GeForce RTX 4080 SUPER".into(),
                backend: GpuBackend::Vulkan,
                memory_bytes: 17_171_480_576,
                util: Some(if live { 0.94 + wobble } else { 0.02 }),
            }],
        };
        // The demo's simulated GPU has tensor cores, so Mixed precision reads as this server's (a
        // run trains the same simulated curve either way).
        let capabilities = Capabilities {
            device: "gpu:0".into(),
            shader_f16: true,
            tensor_cores: TensorCoreSupport { available: true, enabled: true, reason: None },
            amp: AmpSupport { available: true, reason: None },
        };
        Ok(SystemResponse {
            system,
            data_dir: None,
            features: vec![
                "queue_control".into(),
                "weights_onnx".into(),
                "run_estimate".into(),
                "run_cli".into(),
                "throughput".into(),
                "benchmark".into(),
                folo_api::FEATURE_CLOUD.into(),
            ],
            capabilities: Some(capabilities),
            gpu_pool: None,
        })
    }

    fn datasets(&self) -> ApiResult<Vec<Dataset>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        Ok(self.datasets.iter().map(|d| Dataset { stats: None, ..d.ds.clone() }).collect())
    }

    fn dataset(&self, id: &str) -> ApiResult<Dataset> {
        self.check()?;
        Ok(self.fixture(id)?.ds.clone())
    }

    fn dataset_issues(&self, id: &str) -> ApiResult<IssuePage> {
        self.check()?;
        // The broken labels split as folo dataset check would: most are errors (lines the loader
        // skips), a third warnings (boxes clipped to the image).
        let broken = self.fixture(id)?.ds.stats.as_ref().map_or(0, |s| s.broken_labels);
        let warning = broken / 3;
        let error = broken - warning;
        let mut counts = IssueCounts { error, warning, info: 0, by_kind: Default::default() };
        if error > 0 {
            counts.by_kind.insert("class_out_of_range".into(), error);
        }
        if warning > 0 {
            counts.by_kind.insert("box_out_of_bounds".into(), warning);
        }
        Ok(IssuePage { total: broken, issues: Vec::new(), counts })
    }

    fn images(&self, id: &str, q: &ImageQuery) -> ApiResult<ImagePage> {
        self.check()?;
        let f = self.fixture(id)?;
        let matching: Vec<&ImageInfo> = f
            .images
            .iter()
            .filter(|im| q.split.is_none_or(|s| im.split == s))
            .filter(|im| q.class.is_none_or(|c| im.boxes.iter().any(|b| b.class == c)))
            .collect();
        let limit = if q.limit == 0 { 100 } else { q.limit };
        Ok(ImagePage {
            total: matching.len() as u64,
            images: matching.into_iter().skip(q.offset as usize).take(limit as usize).cloned().collect(),
        })
    }

    fn image(&self, dataset_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData> {
        let f = self.fixture(dataset_id)?;
        let index: u32 = image_id
            .strip_prefix("img_")
            .and_then(|n| n.parse::<u32>().ok())
            .and_then(|n| n.checked_sub(1))
            .filter(|&i| (i as usize) < f.images.len())
            .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
        let meta = &f.images[index as usize];
        let (mut w, mut h) = (meta.width, meta.height);
        if let Some(t) = thumb {
            let s = t as f32 / w.max(h) as f32;
            if s < 1.0 {
                w = ((w as f32 * s).round() as u32).max(1);
                h = ((h as f32 * s).round() as u32).max(1);
            }
        }
        let key = (dataset_id.to_string(), image_id.to_string(), w);
        if let Some(px) = self.image_cache.lock().unwrap().get(&key) {
            return Ok(ImageData::Rgba { width: w, height: h, pixels: px.to_vec() });
        }
        let pixels = f.scene(index).render(w, h);
        let mut cache = self.image_cache.lock().unwrap();
        if cache.len() > 600 {
            cache.clear();
        }
        cache.insert(key, Arc::new(pixels.clone()));
        Ok(ImageData::Rgba { width: w, height: h, pixels })
    }

    fn models(&self) -> ApiResult<Vec<Model>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        Ok(self.models.clone())
    }

    fn model_eval(&self, id: &str) -> ApiResult<ModelEval> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {id}")))?;
        let metrics = m.metrics.unwrap_or(ModelMetrics {
            map50: 0.5,
            map50_95: 0.3,
            precision: 0.5,
            recall: 0.5,
            best_f1_conf: None,
        });
        // What is evaluated: the model's run's dataset (else one with its classes), its val split (train when
        // it has none, saying so).
        let run_ds = {
            let runs = self.runs.lock().unwrap();
            runs.iter()
                .find(|f| m.run_id.as_deref() == Some(&*f.run.id))
                .map(|f| f.run.config.dataset_id.clone())
        };
        let ds = run_ds
            .and_then(|id| self.fixture(&id).ok())
            .or_else(|| self.datasets.iter().find(|d| d.ds.classes == m.classes))
            .map(|d| &d.ds);
        let split = if ds.is_some_and(|d| d.splits.val == 0) { Split::Train } else { Split::Val };
        let images =
            ds.map_or(800, |d| if split == Split::Val { d.splits.val } else { d.splits.train }) as u32;
        let split_fallback = (split != Split::Val).then(|| {
            folo_api::SplitFallback::empty_split(
                Split::Val,
                split,
                ds.map_or("?", |d| &d.name),
                images as u64,
            )
        });
        let n = m.classes.len();
        let mut rng = Rng::new(hash(id));
        // Each class's PR curve has its AP50 as its area: precision 1 - (r / r1)^k up to the
        // recall r1 it reaches (k = 6, flat then falling, as detector curves run; a class too
        // good for that reaches recall 1 with a steeper k). Its precision and recall are the
        // curve's best-F1 point, so the dots sit on their curves.
        let curve = |ap: f32| {
            let (r1, k) =
                if ap * 7. / 6. <= 1. { (ap * 7. / 6., 6.) } else { (1., ap / (1. - ap).max(1e-3)) };
            move |r: f32| if r > r1 { 0. } else { 1. - (r / r1).powf(k) }
        };
        // Per-class spread around the model's figures, scaled so the classes average to them
        // (the legend's "All classes" row is their mean).
        let spread: Vec<f32> = (0..n).map(|c| 1.0 + 0.07 * rng.normal() - 0.025 * c as f32).collect();
        let mean = (spread.iter().sum::<f32>() / n.max(1) as f32).max(1e-3);
        let per_class: Vec<ClassMetrics> = (0..n as u32)
            .map(|c| {
                let d = spread[c as usize] / mean;
                let ap = (metrics.map50 * d).min(0.995);
                let p_at = curve(ap);
                let (recall, precision) = (1..=200)
                    .map(|i| {
                        let r = i as f32 / 200.;
                        (r, p_at(r))
                    })
                    .max_by(|a, b| {
                        let f1 = |(r, p): &(f32, f32)| 2. * r * p / (r + p).max(1e-6);
                        f1(a).total_cmp(&f1(b))
                    })
                    .unwrap_or((metrics.recall, metrics.precision));
                ClassMetrics {
                    class: c,
                    map50: ap,
                    map50_95: (metrics.map50_95 * d).min(0.99),
                    precision,
                    recall,
                }
            })
            .collect();
        let pr = per_class
            .iter()
            .map(|cm| {
                let p_at = curve(cm.map50);
                // Sampled up to where the curve ends, its end point (precision 0 at r1) included.
                let r1 = if cm.map50 * 7. / 6. <= 1. { cm.map50 * 7. / 6. } else { 1. };
                let mut recall: Vec<f32> =
                    (0..=100).map(|i| i as f32 / 100.).take_while(|&r| r == 0. || r < r1).collect();
                recall.push(r1);
                let precision: Vec<f32> =
                    recall.iter().map(|&r| if r >= r1 { 0. } else { p_at(r) }).collect();
                PrCurve { class: cm.class, recall, precision, ap50: cm.map50 }
            })
            .collect();
        let mut confusion = vec![vec![0u32; n + 1]; n + 1];
        for t in 0..n {
            let count = 900 + rng.below(500);
            let recall = per_class[t].recall;
            let hit = (count as f32 * recall) as u32;
            confusion[t][t] = hit;
            let mut left = count - hit;
            for (p, row_cell) in confusion[t].iter_mut().enumerate().take(n) {
                if p != t {
                    let share = (left as f32 * rng.range(0.05, 0.25)) as u32;
                    *row_cell += share;
                    left -= share;
                }
            }
            confusion[t][n] = left;
            confusion[n][t] = (count as f32 * (1.0 - per_class[t].precision) * 0.6) as u32;
        }
        Ok(ModelEval {
            pr,
            confusion,
            per_class,
            images,
            best_f1_conf: metrics.best_f1_conf,
            dataset_id: ds.map(|d| d.id.clone()),
            dataset_name: ds.map(|d| d.name.clone()),
            split: Some(split),
            split_fallback,
            // folo-vision's confusion thresholds (docs/METRICS.md §7).
            confusion_conf: Some(0.25),
            confusion_iou: Some(0.45),
        })
    }

    fn export_model(&self, id: &str, format: &str) -> ApiResult<ExportResult> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {id}")))?;
        let (ext, bytes) = match format {
            "onnx" => ("onnx", m.bytes + 120_000),
            _ => ("f16.folo", m.bytes / 2),
        };
        Ok(ExportResult { path: format!("D:/folo/models/{}.{ext}", m.name), bytes })
    }

    fn runs(&self) -> ApiResult<Vec<Run>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        Ok(self.runs.lock().unwrap().iter().map(Self::snapshot).collect())
    }

    fn run(&self, id: &str) -> ApiResult<Run> {
        self.check()?;
        self.runs
            .lock()
            .unwrap()
            .iter()
            .find(|f| f.run.id == id)
            .map(Self::snapshot)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))
    }

    fn create_run(&self, config: TrainConfig) -> ApiResult<Run> {
        self.check()?;
        let ds = self.fixture(&config.dataset_id)?;
        // A cloud run is checked like any run plus its compute (docs/API.md, Cloud runs).
        let target = match &config.compute {
            Some(c) => {
                let (conn, offer) = self.compute_offer(c, &config)?;
                if offer.availability == CloudAvailability::None {
                    return Err(ApiError::new(
                        "cloud_refused",
                        format!("{} has no {} available right now", conn.name, crate::ui::offer_line(&offer)),
                    )
                    .with_hint("Pick another GPU, or try again in a few minutes."));
                }
                Some((c.clone(), conn, offer))
            }
            None => None,
        };
        let mut runs = self.runs.lock().unwrap();
        let id = format!("run_{:04x}", hash(&format!("{}{}", config.name, runs.len())) & 0xFFFF);
        let spe = (ds.ds.splits.train / config.batch.max(1) as u64).max(1);
        let curve = Curve {
            seed: hash(&id),
            epochs: config.epochs,
            spe,
            lr0: resolve(&config, ds.ds.splits.train, ds.ds.classes.len()).lr0 as f32,
            batch: config.batch,
            map_max: 0.66,
            ips: 380.0,
            classes: ds.ds.classes.len() as u32,
            close_mosaic: config.augment.close_mosaic,
        };
        let now = self.sim_now();
        let mut f = RunFixture {
            run: Run {
                id: id.clone(),
                name: config.name.clone(),
                status: RunStatus::Running,
                created: time::format(now),
                started: Some(time::format(now)),
                finished: None,
                epoch: 0,
                epochs: config.epochs,
                step: 0,
                steps_per_epoch: spe,
                eta_s: None,
                best: None,
                device: if config.device == "cpu" { CPU_DEVICE.into() } else { GPU_DEVICE.into() },
                model_id: None,
                error: None,
                error_code: None,
                resolved: None,
                queue_position: None,
                cloud: None,
                config,
            },
            curve,
            live: (!self.frozen && target.is_none()).then(|| (0, Instant::now())),
            step: 0,
            logs: Vec::new(),
            cloud: None,
        };
        if let Some((compute, conn, offer)) = target {
            let base = launch_state(&id, &compute, &conn.provider, offer, conn.provider == "runpod");
            let mut sim = CloudSim::new(base, self.clock, self.clock.now(), f.run.epochs as u64 * spe);
            // The dataset as FOLO sends it: about 180 kB an image.
            sim.upload_bytes = ds.ds.splits.total() * 180_000;
            f.cloud = Some(sim);
        }
        // A local run's lines follow its start; a cloud run's its launch, now.
        f.logs = initial_logs(&f, if f.cloud.is_some() { now } else { now - 5 });
        let run = Self::snapshot(&f);
        runs.insert(0, f);
        Ok(run)
    }

    fn run_cli(&self, id: &str) -> ApiResult<CliCommand> {
        self.check()?;
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        Ok(self.cli_for(&f.run.config))
    }

    fn throughput(&self) -> ApiResult<ThroughputTable> {
        self.check()?;
        let (training, inference) = self.measured.lock().unwrap().clone();
        Ok(with_reference(training, inference))
    }

    fn benchmark(&self, req: &BenchmarkRequest) -> ApiResult<BenchmarkReport> {
        self.check()?;
        let busy = self.runs().unwrap_or_default().into_iter().any(|r| {
            !self.fixture_runs.contains(&r.id) && matches!(r.status, RunStatus::Queued | RunStatus::Running)
        });
        if busy {
            return Err(ApiError::new("run_active", "A run is queued or training")
                .with_hint("Measure when the queue is empty."));
        }
        if !req.imgsz.is_multiple_of(32)
            || !(64..=1280).contains(&req.imgsz)
            || !(1..=64).contains(&req.batch)
            || !(req.budget_s.is_finite() && (1.0..=10.0).contains(&req.budget_s))
        {
            return Err(ApiError::new(
                "invalid_body",
                "imgsz must be a multiple of 32 in 64..=1280, batch 1..=64, budget_s 1..=10",
            ));
        }
        // Near the reference, a little off, as a real probe lands.
        let r = reference_train(req.arch, req.imgsz, req.amp, false) * 0.97;
        let lat = reference_latency(req.arch, req.imgsz, req.amp, false) * 1.04;
        let now = time::format(self.sim_now());
        let mut m = self.measured.lock().unwrap();
        let same_train = |t: &TrainThroughput| {
            t.device == "gpu:0"
                && t.arch == req.arch
                && t.imgsz == req.imgsz
                && t.batch == req.batch
                && t.amp == req.amp
        };
        let train = match m.0.iter().position(same_train) {
            Some(i) => {
                let mut t = m.0.remove(i);
                t.imgs_per_s = (t.imgs_per_s * t.samples as f64 + r) / (t.samples + 1) as f64;
                t.samples += 1;
                t.probes += 1;
                t.updated = Some(now.clone());
                t
            }
            None => TrainThroughput {
                samples: 1,
                probes: 1,
                updated: Some(now.clone()),
                ..train_row(
                    "gpu:0",
                    GPU_NAME,
                    req.arch,
                    req.imgsz,
                    req.batch,
                    req.amp,
                    r,
                    ThroughputSource::Measured,
                )
            },
        };
        m.0.insert(0, train.clone());
        // Inference counts predictions (training counts one sample per probe).
        let predictions = (req.budget_s * 0.4 * 1000. / lat).round() as u64;
        let precision = if req.amp { InferPrecision::F16 } else { InferPrecision::F32 };
        let same_infer = |i: &InferThroughput| {
            i.device == "gpu:0" && i.arch == req.arch && i.imgsz == req.imgsz && i.precision == precision
        };
        let infer = match m.1.iter().position(same_infer) {
            Some(i) => {
                let mut row = m.1.remove(i);
                let n = row.samples as f64;
                row.latency_ms = (row.latency_ms * n + lat * predictions as f64) / (n + predictions as f64);
                row.imgs_per_s = 1000. / row.latency_ms;
                row.samples += predictions;
                row.updated = Some(now.clone());
                row
            }
            None => InferThroughput {
                samples: predictions,
                updated: Some(now),
                ..infer_row("gpu:0", GPU_NAME, req.arch, req.imgsz, req.amp, lat, ThroughputSource::Measured)
            },
        };
        m.1.insert(0, infer.clone());
        Ok(BenchmarkReport {
            device: "gpu:0".into(),
            device_name: GPU_NAME.into(),
            train_imgs_per_s: r,
            train_batches: (r * req.budget_s * 0.42 / req.batch as f64).round() as u64,
            latency_ms: lat,
            predictions,
            elapsed_s: req.budget_s + 4.0,
            training: train,
            inference: infer,
        })
    }

    fn estimate_run(&self, config: &TrainConfig) -> ApiResult<RunEstimate> {
        self.check()?;
        let ds = self
            .datasets
            .iter()
            .find(|d| d.ds.id == config.dataset_id)
            .ok_or_else(|| ApiError::new("dataset_not_found", "No such dataset"))?;
        let train = ds.ds.splits.train;
        // No validation split: validation runs on the training images (docs/API.md).
        let val = if ds.ds.splits.val > 0 { ds.ds.splits.val } else { train };
        // Reference rates (the app's own reference model) unless the table has a measured row of
        // this device with the run's arch, image size, batch and precision (docs/API.md: the
        // measured row first).
        let arch = match config.arch.as_str() {
            "folo-n" => 0,
            "folo-s" => 1,
            "folo-m" => 2,
            "folo-l" => 3,
            _ => 4,
        };
        let cpu = config.device == "cpu";
        let (secs, vram_gib) =
            crate::overlays::estimate(arch, config.epochs as f32, config.batch, config.imgsz, train as u32);
        let secs = if cpu { secs * 25. } else { secs };
        let vram_gib = if config.amp { vram_gib * 0.58 } else { vram_gib };
        let epochs = config.epochs.max(1);
        // Without mosaic every epoch is a close-mosaic one; else the last ones, capped at half.
        let close =
            if config.augment.mosaic <= 0. { epochs } else { config.augment.close_mosaic.min(epochs / 2) };
        let startup = 6.0;
        // A cloud config trains on its offer's GPU: nothing measured here applies (no row has
        // that device), so the estimate is the reference, plus what the run would cost.
        let offer = match &config.compute {
            Some(c) => Some(self.compute_offer(c, config)?.1),
            None => None,
        };
        let (device, here) = match &offer {
            Some(o) => ("cloud", o.gpu.as_str()),
            None if cpu => ("cpu", CPU_NAME),
            None => ("gpu:0", GPU_NAME),
        };
        // The basis (docs/API.md): the measured row of this device with the run's arch, image
        // size, batch and precision; else the nearest measured row on it by compute (the same
        // precision preferred), carried over by the reference's ratio between the two settings
        // (the reference has no batch dependence); else the reference itself.
        let rows: Vec<TrainThroughput> = self
            .measured
            .lock()
            .unwrap()
            .0
            .iter()
            .filter(|t| t.device == device && t.measured_on == here)
            .cloned()
            .collect();
        // Model compute: the architecture's GFLOPs at 640 px, scaled by image area.
        let compute = |arch: Arch, imgsz: u32| gflops(arch) * (imgsz as f64 / 640.).powi(2);
        let want = compute(config.arch, config.imgsz);
        let exact = rows.iter().find(|t| {
            t.arch == config.arch && t.imgsz == config.imgsz && t.batch == config.batch && t.amp == config.amp
        });
        // Nearest by compute, then by batch; the same precision before any other.
        let nearest = exact.or_else(|| {
            rows.iter().min_by(|a, b| {
                let key = |t: &TrainThroughput| {
                    (
                        t.amp != config.amp,
                        (compute(t.arch, t.imgsz) / want).ln().abs(),
                        (t.batch.max(1) as f64 / config.batch.max(1) as f64).ln().abs(),
                    )
                };
                let (ka, kb) = (key(a), key(b));
                ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(ka.2.total_cmp(&kb.2))
            })
        });
        // (row, rate carried to this config, within 4x compute with the same precision)
        let measured = nearest.map(|m| {
            let carried = if exact.is_some() {
                m.imgs_per_s
            } else {
                m.imgs_per_s * reference_train(config.arch, config.imgsz, config.amp, cpu)
                    / reference_train(m.arch, m.imgsz, m.amp, cpu)
            };
            let near = m.amp == config.amp && (compute(m.arch, m.imgsz) / want).ln().abs() <= 4f64.ln();
            (m.clone(), carried, near)
        });
        let exact = exact.is_some();
        let ref_epoch_s = secs / epochs as f64;
        // Validation stays at the reference rate (the table's rows time training).
        let val_s = ref_epoch_s * 0.08;
        let (rate, train_s) = match &measured {
            Some((_, r, _)) => (*r, train as f64 / r.max(1e-6)),
            None => (train as f64 / (ref_epoch_s * 0.92), ref_epoch_s * 0.92),
        };
        let epoch_s = train_s + val_s;
        let secs = epoch_s * epochs as f64;
        let phase = |kind: PhaseKind, n: u32| EstimatePhase {
            phase: kind,
            epochs: n,
            train_imgs_per_s: rate,
            train_s,
            val_s,
            epoch_s,
        };
        let phases = [(PhaseKind::Mosaic, epochs - close), (PhaseKind::CloseMosaic, close)]
            .into_iter()
            .filter(|p| p.1 > 0)
            .map(|(k, n)| phase(k, n))
            .collect();
        let device_name = here.to_string();
        // Confidence and range as docs/API.md has them: an exact row, high from 3 samples (±10 %)
        // else medium (-20 / +30 %); a carried row within 4x the compute with the same precision,
        // medium (-25 / +40 %); further, low (-40 / +80 %); the reference, low: -30 / +60 % on the
        // reference machine, -60 / +200 % elsewhere (the demo's CPU isn't the reference CPU).
        let (source, confidence, lo, hi) = match &measured {
            Some((m, _, _)) if exact && m.samples >= 3 => {
                (ThroughputSource::Measured, Confidence::High, 0.9, 1.1)
            }
            Some(_) if exact => (ThroughputSource::Measured, Confidence::Medium, 0.8, 1.3),
            Some((_, _, true)) => (ThroughputSource::Measured, Confidence::Medium, 0.75, 1.4),
            Some(_) => (ThroughputSource::Measured, Confidence::Low, 0.6, 1.8),
            None if (if cpu { REF_CPU } else { REF_GPU }) == here => {
                (ThroughputSource::Reference, Confidence::Low, 0.7, 1.6)
            }
            None => (ThroughputSource::Reference, Confidence::Low, 0.4, 3.0),
        };
        let basis = match &measured {
            Some((m, _, _)) => EstimateBasis {
                source: ThroughputSource::Measured,
                arch: m.arch,
                imgsz: m.imgsz,
                batch: m.batch,
                amp: m.amp,
                measured_on: m.measured_on.clone(),
                samples: m.samples,
                updated: m.updated.clone(),
                exact,
                note: if exact {
                    format!(
                        "Measured on this machine at these settings ({} {}).",
                        m.samples,
                        if m.samples == 1 { "sample" } else { "samples" }
                    )
                } else {
                    format!(
                        "Carried from {} at {} px, batch {}{} measured on this machine, by the reference's ratio.",
                        m.arch.as_str(),
                        m.imgsz,
                        m.batch,
                        if m.amp { " (mixed precision)" } else { "" }
                    )
                },
            },
            None => EstimateBasis {
                source: ThroughputSource::Reference,
                arch: config.arch,
                imgsz: 640,
                batch: 16,
                amp: false,
                measured_on: if cpu { REF_CPU } else { REF_GPU }.into(),
                samples: 0,
                updated: None,
                exact: false,
                note: match &offer {
                    Some(o) => format!(
                        "Reference rates of FOLO's reference machine ({}), scaled to this config; the {} may \
                         be faster or slower.",
                        crate::ui::short_gpu(REF_GPU),
                        o.gpu
                    ),
                    None => "Reference rates for this architecture at 640 px and batch 16, scaled to this \
                             config; nothing has been measured at these settings on this machine."
                        .into(),
                },
            },
        };
        let total_s = secs + startup;
        let cost = offer.as_ref().zip(config.compute.as_ref()).and_then(|(o, c)| {
            let price = o.price_per_hour?;
            let usd = |s: f64| price * s / 3600.;
            Some(CostEstimate {
                price_per_hour: price,
                est_usd: usd(total_s),
                est_usd_low: usd(total_s * lo),
                est_usd_high: usd(total_s * hi),
                max_usd: price * c.max_hours as f64,
            })
        });
        Ok(RunEstimate {
            device: device.into(),
            device_name,
            train_images: train,
            val_images: val,
            batches_per_epoch: train.div_ceil(config.batch.max(1) as u64),
            epochs,
            phases,
            // The rate the phases' validation time is at (val_images / val_imgs_per_s = val_s).
            val_imgs_per_s: val as f64 / val_s.max(1e-6),
            startup_s: startup,
            est_epoch_s: epoch_s,
            est_total_s: secs + startup,
            est_total_s_low: (secs + startup) * lo,
            est_total_s_high: (secs + startup) * hi,
            source,
            confidence,
            basis,
            val_source: ThroughputSource::Reference,
            est_peak_vram_bytes: (!cpu).then_some((vram_gib as f64 * (1u64 << 30) as f64) as u64),
            vram_source: (!cpu).then_some(ThroughputSource::Reference),
            close_mosaic: close,
            cli: Some(self.cli_for(config).cli),
            cost,
        })
    }

    fn queue_run(&self, id: &str, _req: QueueRequest) -> ApiResult<Run> {
        self.check()?;
        let mut runs = self.runs.lock().unwrap();
        let f = runs
            .iter_mut()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        if f.run.status != RunStatus::Queued {
            return Err(ApiError::new("run_not_queued", "This run is no longer waiting in the queue"));
        }
        // The demo queues one run at a time: every move leaves it first.
        f.run.queue_position = Some(1);
        Ok(f.run.clone())
    }

    fn stop_run(&self, id: &str) -> ApiResult<Run> {
        self.check()?;
        let mut runs = self.runs.lock().unwrap();
        let f = runs
            .iter_mut()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        if !Self::snapshot(f).status.is_active() {
            return Err(ApiError::new("run_not_active", "This run is not running"));
        }
        // A cloud run stops on its instance: its checkpoint is collected and the instance released
        // (at once before training).
        if let Some(sim) = f.cloud.as_mut() {
            sim.ended = Some((sim.elapsed(), Ended::Stopped));
            f.logs.push(LogLine {
                t: time::format(self.sim_now()),
                level: LogLevel::Warn,
                message: "stop requested; collecting the checkpoint and releasing the instance".into(),
            });
            return Ok(Self::snapshot(f));
        }
        f.step = Self::current_step(f);
        f.live = None;
        f.run.status = RunStatus::Stopped;
        f.run.finished = Some(time::format(self.sim_now()));
        let epoch = f.step / f.run.steps_per_epoch;
        f.logs.push(LogLine {
            t: time::format(self.sim_now()),
            level: LogLevel::Warn,
            message: "stop requested; finishing current step".into(),
        });
        f.logs.push(LogLine {
            t: time::format(self.sim_now()),
            level: LogLevel::Info,
            message: format!("saved last checkpoint (epoch {epoch})"),
        });
        Ok(Self::snapshot(f))
    }

    fn metrics(&self, id: &str) -> ApiResult<RunMetrics> {
        self.check()?;
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        let step = Self::current_step(f);
        Ok(RunMetrics {
            steps: (1..=step).map(|i| f.curve.step(i)).collect(),
            epochs: (1..=(step / f.run.steps_per_epoch) as u32).map(|e| f.curve.epoch(e)).collect(),
        })
    }

    fn samples(&self, id: &str, epoch: Option<u32>) -> ApiResult<RunSamples> {
        self.check()?;
        let (curve, dataset_id, done) = {
            let runs = self.runs.lock().unwrap();
            let f = runs
                .iter()
                .find(|f| f.run.id == id)
                .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
            (
                f.curve.clone(),
                f.run.config.dataset_id.clone(),
                (Self::current_step(f) / f.run.steps_per_epoch) as u32,
            )
        };
        let epoch = epoch.unwrap_or(done).min(done);
        let q = (curve.quality(epoch) * curve.map_max / 0.75).clamp(0.05, 1.0);
        let class_names = self.fixture(&dataset_id).map(|f| f.ds.classes.clone()).unwrap_or_default();
        let samples = self
            .val_images(&dataset_id)
            .into_iter()
            .take(16)
            .map(|(i, im)| Sample {
                image_id: im.id.clone(),
                dataset_id: dataset_id.clone(),
                width: im.width,
                height: im.height,
                truth: im.boxes.clone(),
                pred: fake_predictions(
                    &im.boxes,
                    im.width,
                    im.height,
                    q,
                    &class_names,
                    hash(id) ^ i as u64 ^ (epoch as u64) << 20,
                ),
            })
            .collect();
        Ok(RunSamples { epoch, samples })
    }

    fn logs(&self, id: &str) -> ApiResult<Vec<LogLine>> {
        self.check()?;
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        // Epoch summaries up to now, merged with the fixture's lines in time order (a cloud run's
        // from when it started training on its instance).
        let created = time::parse(&f.run.created).unwrap_or(MOCK_NOW);
        let trains_at = created + f.cloud.as_ref().map_or(10, |s| s.train_at() as i64 + 10);
        let done = (Self::current_step(f) / f.run.steps_per_epoch) as u32;
        let per_epoch = (f.run.steps_per_epoch as f64 * STEP_DT) as i64 + 1;
        let mut lines = f.logs.clone();
        let from = done.saturating_sub(60) + 1;
        for e in from..=done {
            lines.push(LogLine {
                t: time::format(trains_at + e as i64 * per_epoch),
                level: LogLevel::Info,
                message: epoch_line(e, f.run.epochs, &f.curve.epoch(e)),
            });
        }
        lines.sort_by(|a, b| a.t.cmp(&b.t));
        Ok(lines)
    }

    fn events(&self, id: &str) -> ApiResult<Subscription> {
        self.check()?;
        let (tx, rx) = channel();
        let (sub, cancel) = Subscription::new(rx);
        let (live, curve, spe, epochs, snap, log_mark, cloud) = {
            let runs = self.runs.lock().unwrap();
            let f = runs
                .iter()
                .find(|f| f.run.id == id)
                .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
            (
                f.live,
                f.curve.clone(),
                f.run.steps_per_epoch,
                f.run.epochs,
                Self::snapshot(f),
                f.logs.len(),
                f.cloud.is_some(),
            )
        };
        let _ = tx.send(RunEvent::Status(StatusEvent::new(snap.status, None).with_length(epochs, spe)));
        if cloud && !self.frozen {
            self.stream_cloud(id, snap, curve, log_mark, tx, cancel);
            return Ok(sub);
        }
        let Some((base, at)) = live else { return Ok(sub) };
        let total = epochs as u64 * spe;
        let start_secs = self.sim_now();
        let runs = self.runs.clone();
        let run_id = id.to_string();
        // The stream starts where the history a client just fetched ends: capture it now, not
        // whenever the worker thread happens to start.
        let mut sent = (base + (at.elapsed().as_secs_f64() / STEP_DT) as u64).min(total);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(100));
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                // One lock for the whole decision, so a stop can't land between "is it stopped?"
                // and "how far has it got?".
                let (target, stop) = {
                    let runs = runs.lock().unwrap();
                    match runs.iter().find(|f| f.run.id == run_id) {
                        None => (sent, Some(Vec::new())),
                        Some(f) if f.live.is_none() => (
                            f.step.min(total),
                            Some(f.logs.get(log_mark..).map(<[LogLine]>::to_vec).unwrap_or_default()),
                        ),
                        Some(f) => (Self::current_step(f), None),
                    }
                };
                // Stopped through the API: flush the steps up to the stop point and the stop's log
                // lines, then report the terminal status and end the stream.
                if let Some(lines) = stop {
                    let stop_step = target;
                    while sent < stop_step {
                        sent += 1;
                        let _ = tx.send(RunEvent::Step(curve.step(sent)));
                        if sent.is_multiple_of(spe) {
                            let _ = tx.send(RunEvent::Epoch(curve.epoch((sent / spe) as u32)));
                        }
                    }
                    for l in lines {
                        let _ = tx.send(RunEvent::Log(l));
                    }
                    let _ = tx.send(RunEvent::Status(
                        StatusEvent::new(RunStatus::Stopped, None).with_length(epochs, spe),
                    ));
                    return;
                }
                let now = target;
                while sent < now {
                    sent += 1;
                    if tx.send(RunEvent::Step(curve.step(sent))).is_err() {
                        return;
                    }
                    if sent.is_multiple_of(spe) {
                        let e = (sent / spe) as u32;
                        let ep = curve.epoch(e);
                        let t = time::format(start_secs + at.elapsed().as_secs() as i64);
                        let msg = epoch_line(e, epochs, &ep);
                        if tx.send(RunEvent::Epoch(ep)).is_err()
                            || tx
                                .send(RunEvent::Log(LogLine { t, level: LogLevel::Info, message: msg }))
                                .is_err()
                        {
                            return;
                        }
                    }
                }
                if sent >= total {
                    let _ = tx.send(RunEvent::Status(
                        StatusEvent::new(RunStatus::Completed, None).with_length(epochs, spe),
                    ));
                    return;
                }
            }
        });
        Ok(sub)
    }

    fn cloud_providers(&self) -> ApiResult<Vec<CloudProvider>> {
        self.check()?;
        Ok(cloud_providers())
    }

    fn cloud_connections(&self) -> ApiResult<Vec<CloudConnection>> {
        self.check()?;
        Ok(self.connections.lock().unwrap().clone())
    }

    fn connect_cloud(&self, req: &ConnectCloud) -> ApiResult<CloudConnection> {
        self.check()?;
        let provider = cloud_providers().into_iter().find(|p| p.id == req.provider).ok_or_else(|| {
            ApiError::new("provider_not_found", format!("No cloud provider {}", req.provider))
                .with_hint("GET /api/cloud/providers lists them.")
        })?;
        let value = |key: &str| req.values.get(key).map(|v| v.trim()).filter(|v| !v.is_empty());
        if let Some(f) = provider.fields.iter().find(|f| f.required && value(&f.key).is_none()) {
            return Err(ApiError::new("invalid_body", format!("{} is missing", f.label)));
        }
        // The demo's provider rejects any secret containing "bad".
        if provider.fields.iter().any(|f| f.secret && value(&f.key).is_some_and(|v| v.contains("bad"))) {
            return Err(ApiError::new(
                "cloud_auth_failed",
                format!("{} rejected these credentials", provider.name),
            )
            .with_hint("Check that the key is current and allowed to manage pods."));
        }
        let now = time::format(self.sim_now());
        let url = value("url").map(str::to_string);
        let mut conns = self.connections.lock().unwrap();
        let base =
            if req.name.trim().is_empty() { provider.name.clone() } else { req.name.trim().to_string() };
        let name = (1..)
            .map(|n| if n == 1 { base.clone() } else { format!("{base} {n}") })
            .find(|n| !conns.iter().any(|c| &c.name == n))
            .unwrap_or(base);
        let conn = CloudConnection {
            id: format!("cc_{:04x}", hash(&format!("{name}{}", conns.len())) & 0xFFFF),
            provider: provider.id.clone(),
            name,
            created: now.clone(),
            status: CloudConnectionStatus::Ok,
            account: match &url {
                Some(u) => Some(u.split("://").nth(1).unwrap_or(u).trim_end_matches('/').to_string()),
                None => Some("you@example.com".into()),
            },
            balance_usd: provider.provisions.then_some(42.17),
            error: None,
            note: url.as_deref().filter(|u| unencrypted(u)).map(|_| {
                "Unencrypted http://: the token and the data cross the network in the clear. Use https:// \
                 (a TLS proxy or a tunnel)."
                    .to_string()
            }),
            checked: Some(now),
            values: provider
                .fields
                .iter()
                .filter_map(|f| {
                    let v = value(&f.key)?;
                    Some((f.key.clone(), if f.secret { secret_hint(v) } else { v.to_string() }))
                })
                .collect(),
        };
        conns.push(conn.clone());
        Ok(conn)
    }

    fn check_cloud(&self, id: &str) -> ApiResult<CloudConnection> {
        self.check()?;
        let mut conns = self.connections.lock().unwrap();
        let c = conns.iter_mut().find(|c| c.id == id).ok_or_else(|| no_connection(id))?;
        c.checked = Some(time::format(self.sim_now()));
        Ok(c.clone())
    }

    fn disconnect_cloud(&self, id: &str) -> ApiResult<()> {
        self.check()?;
        let name = self.connection(id)?.name;
        let in_use = self.runs.lock().unwrap().iter().any(|f| {
            f.cloud.as_ref().is_some_and(|s| s.base.connection_id == id)
                && (Self::snapshot(f).status.is_active()
                    || f.cloud
                        .as_ref()
                        .and_then(|s| s.instance(&f.run.id))
                        .is_some_and(|i| i.status.is_alive()))
        });
        if in_use {
            return Err(ApiError::new(
                "connection_in_use",
                format!("{name} has a live instance or an active run"),
            )
            .with_hint("Terminate its instances in Settings › Cloud, or wait for its runs to end."));
        }
        self.connections.lock().unwrap().retain(|c| c.id != id);
        Ok(())
    }

    fn cloud_offers(&self, connection: &str, _refresh: bool) -> ApiResult<CloudOfferList> {
        self.check()?;
        let conn = self.connection(connection)?;
        Ok(CloudOfferList { offers: offers_of(&conn)?, fetched: time::format(self.sim_now()) })
    }

    fn cloud_instances(&self) -> ApiResult<Vec<CloudInstance>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        let week_ago = time::format(self.sim_now() - 7 * 86_400);
        let mut list: Vec<CloudInstance> = self
            .runs
            .lock()
            .unwrap()
            .iter()
            .filter_map(|f| f.cloud.as_ref()?.instance(&f.run.id))
            .filter(|i| i.ended.as_ref().is_none_or(|e| *e >= week_ago))
            .collect();
        // Alive first, then the ended ones; newest first within each.
        list.sort_by(|a, b| b.status.is_alive().cmp(&a.status.is_alive()).then(b.created.cmp(&a.created)));
        Ok(list)
    }

    fn terminate_instance(&self, id: &str) -> ApiResult<CloudInstance> {
        self.check()?;
        let now = time::format(self.sim_now());
        let mut runs = self.runs.lock().unwrap();
        let f = runs
            .iter_mut()
            .find(|f| f.cloud.as_ref().is_some_and(|s| s.base.instance_id.as_deref() == Some(id)))
            .ok_or_else(|| no_instance(id))?;
        let run_id = f.run.id.clone();
        if let Some(sim) =
            f.cloud.as_mut().filter(|s| s.instance(&run_id).is_some_and(|i| i.status.is_alive()))
        {
            let at = sim.elapsed();
            // The provider confirms at once; a run still using it fails.
            sim.release_error = None;
            if sim.released_at().is_none() {
                sim.ended = Some((at, Ended::InstanceLost));
                f.logs.push(LogLine {
                    t: now,
                    level: LogLevel::Error,
                    message: format!("cloud: instance {id} was terminated by hand"),
                });
            }
        }
        f.cloud.as_ref().and_then(|s| s.instance(&run_id)).ok_or_else(|| no_instance(id))
    }

    fn predict(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == req.model_id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {}", req.model_id)))?;
        let q = m.metrics.as_ref().map_or(0.5, |m| m.map50_95 / 0.8).clamp(0.1, 1.0);
        let (w, h, raw, seed) = match &req.input {
            PredictInput::Dataset { dataset_id, image_id } => {
                let f = self.fixture(dataset_id)?;
                let im = f
                    .images
                    .iter()
                    .find(|im| &im.id == image_id)
                    .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
                let p = fake_predictions(
                    &im.boxes,
                    im.width,
                    im.height,
                    q,
                    &m.classes,
                    hash(image_id) ^ hash(&m.id),
                );
                (im.width, im.height, p, hash(image_id))
            }
            PredictInput::File { name, bytes } => {
                let (w, h) = image::ImageReader::new(std::io::Cursor::new(bytes.as_slice()))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok())
                    .ok_or_else(|| {
                        ApiError::new("unsupported_image", format!("Could not decode {name}"))
                            .with_hint("FOLO reads JPEG and PNG images.")
                    })?;
                let seed = bytes.iter().step_by(97).fold(hash(name), |h, &b| h.rotate_left(5) ^ b as u64);
                let mut rng = Rng::new(seed);
                let n = 2 + rng.below(4);
                let s = w.min(h) as f32;
                let p = (0..n)
                    .map(|_| {
                        let bw = rng.range(0.12, 0.32) * s;
                        let bh = bw * rng.range(0.7, 1.4);
                        let x1 = rng.range(0.0, w as f32 - bw);
                        let y1 = rng.range(0.0, (h as f32 - bh).max(1.0));
                        let class = rng.below(m.classes.len() as u32);
                        ScoredBox {
                            class,
                            name: m.classes.get(class as usize).cloned().unwrap_or_default(),
                            score: rng.range(0.3, 0.92),
                            x1,
                            y1,
                            x2: x1 + bw,
                            y2: (y1 + bh).min(h as f32),
                        }
                    })
                    .collect();
                (w, h, p, seed)
            }
        };
        let mut boxes: Vec<ScoredBox> = raw.into_iter().filter(|b| b.score >= req.conf).collect();
        boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
        boxes.truncate(req.max_det as usize);
        let scale = match m.arch.as_str() {
            "folo-n" => 1.0,
            "folo-s" => 1.7,
            "folo-m" => 3.1,
            _ => 5.4,
        } * (m.imgsz as f32 / 640.0).powi(2);
        let mut rng = Rng::new(seed ^ 0x71);
        let pre = 0.62 + 0.1 * rng.f32();
        let inf = 2.9 * scale * (1.0 + 0.05 * rng.normal());
        let post = 0.14 + 0.02 * boxes.len() as f32;
        Ok(PredictResponse {
            model_id: m.id.clone(),
            image: ImageSize { width: w, height: h },
            boxes,
            timing: Timing {
                preprocess_ms: pre as f64,
                inference_ms: inf as f64,
                postprocess_ms: post as f64,
                total_ms: (pre + inf + post) as f64,
            },
            device: "gpu:0".into(),
        })
    }
}

/// Plausible detections for known ground truth at model quality `q` (0..1): jittered hits,
/// occasional class confusions and misses, and a few low-score false positives.
/// The settings a run resolves to, as the trainer would (docs/MODEL.md §8.2 and §8.4).
fn resolve(config: &TrainConfig, train_images: u64, classes: usize) -> ResolvedTrain {
    let nbs = 64u32;
    let batch = config.batch.max(1);
    let accumulate = ((nbs as f64 / batch as f64).round() as u32).max(1);
    let batches = train_images.div_ceil(batch as u64).max(1);
    let steps = train_images.div_ceil(batch.max(nbs) as u64) as f64 * config.epochs as f64;
    let (optimizer, lr0) = match config.optimizer {
        Optimizer::Sgd => (ResolvedOptimizer::Sgd, config.lr0 as f64),
        Optimizer::Adamw => (ResolvedOptimizer::Adamw, config.lr0 as f64),
        Optimizer::Auto if steps > 10_000. => (ResolvedOptimizer::Sgd, 0.01),
        Optimizer::Auto => {
            let lr = 0.01 / (4.0 + classes as f64);
            (ResolvedOptimizer::Adamw, (lr * 1e6).round() / 1e6)
        }
    };
    let auto = config.optimizer == Optimizer::Auto;
    // Auto's momentum (AdamW β1) is 0.9; a chosen optimizer keeps the configured 0.937.
    let momentum = if auto { 0.9 } else { 0.937 };
    // Warmup in optimizer steps: the trainer's own rule (docs/MODEL.md §8.4).
    let spe = batches.div_ceil(accumulate as u64);
    let warmup_iters = folo_nn::warmup_iters(3.0, config.epochs as u64, spe);
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
        warmup_bias_lr: if auto { 0.0 } else { 0.1 },
        // The trainer's cap: at most half the run.
        close_mosaic: Some(config.augment.close_mosaic.min(config.epochs / 2)),
    }
}

impl MockBackend {
    /// A cloud run's live events, from `first` (the snapshot the stream started at) on: the steps
    /// and epochs it trains, a `status` when its status changes, and its `cloud` state on a new
    /// phase at once and otherwise at most once a second (transfer progress, cost). Ends once the
    /// run is released and finished.
    fn stream_cloud(
        &self,
        id: &str,
        first: Run,
        curve: Curve,
        log_mark: usize,
        tx: std::sync::mpsc::Sender<RunEvent>,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) {
        let runs = self.runs.clone();
        let run_id = id.to_string();
        let clock = self.clock;
        std::thread::spawn(move || {
            let (spe, epochs) = (first.steps_per_epoch, first.epochs);
            let (mut sent, mut status, mut logs) = (first.step, first.status, log_mark);
            let mut cloud = first.cloud;
            let mut cloud_at = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(100));
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                let (now, lines) = {
                    let runs = runs.lock().unwrap();
                    let Some(f) = runs.iter().find(|f| f.run.id == run_id) else { return };
                    (Self::snapshot(f), f.logs.get(logs..).map(<[LogLine]>::to_vec).unwrap_or_default())
                };
                logs += lines.len();
                while sent < now.step {
                    sent += 1;
                    let _ = tx.send(RunEvent::Step(curve.step(sent)));
                    if sent.is_multiple_of(spe) {
                        let e = (sent / spe) as u32;
                        let ep = curve.epoch(e);
                        let line = LogLine {
                            t: time::format(clock.now() as i64),
                            level: LogLevel::Info,
                            message: epoch_line(e, epochs, &ep),
                        };
                        let _ = tx.send(RunEvent::Epoch(ep));
                        let _ = tx.send(RunEvent::Log(line));
                    }
                }
                for l in lines {
                    let _ = tx.send(RunEvent::Log(l));
                }
                let phase_changed = now.cloud.as_ref().map(|c| c.phase) != cloud.as_ref().map(|c| c.phase);
                if now.cloud != cloud && (phase_changed || cloud_at.elapsed() >= Duration::from_secs(1)) {
                    cloud.clone_from(&now.cloud);
                    cloud_at = Instant::now();
                    if let Some(c) = cloud.clone()
                        && tx.send(RunEvent::Cloud(Box::new(c))).is_err()
                    {
                        return;
                    }
                }
                if now.status != status {
                    status = now.status;
                    let ev = StatusEvent::new(status, now.error.clone())
                        .with_length(epochs, spe)
                        .with_error_code(now.error_code.clone());
                    let _ = tx.send(RunEvent::Status(ev));
                }
                if !status.is_active() && cloud.as_ref().is_some_and(|c| c.phase == CloudPhase::Released) {
                    return;
                }
            }
        });
    }

    fn connection(&self, id: &str) -> ApiResult<CloudConnection> {
        self.connections.lock().unwrap().iter().find(|c| c.id == id).cloned().ok_or_else(|| no_connection(id))
    }

    /// The connection and offer a run's `compute` names, checked as `POST /api/runs` checks them.
    fn compute_offer(&self, c: &Compute, config: &TrainConfig) -> ApiResult<(CloudConnection, CloudOffer)> {
        if config.device != "auto" {
            return Err(ApiError::new(
                "invalid_config",
                format!(
                    "device must be auto for a cloud run (got {}): the worker picks its GPU",
                    config.device
                ),
            ));
        }
        if !(folo_api::MIN_MAX_HOURS..=folo_api::MAX_MAX_HOURS).contains(&c.max_hours) {
            return Err(ApiError::new(
                "invalid_config",
                format!("max_hours must be from 0.25 to 720 (got {})", c.max_hours),
            ));
        }
        let conn = self.connection(&c.connection_id)?;
        let offer = offers_of(&conn)?.into_iter().find(|o| o.id == c.offer_id).ok_or_else(|| {
            ApiError::new("offer_not_found", format!("{} has no offer {}", conn.name, c.offer_id))
                .with_hint("Offers change: pick one from the list again.")
        })?;
        Ok((conn, offer))
    }

    /// The demo's `folo train` command for a config (the flags the CLI takes for what the form sets).
    fn cli_for(&self, c: &TrainConfig) -> CliCommand {
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

fn fake_predictions(truth: &[BBoxT], w: u32, h: u32, q: f32, names: &[String], seed: u64) -> Vec<ScoredBox> {
    let classes = names.len().max(1) as u32;
    let name = |c: u32| names.get(c as usize).cloned().unwrap_or_default();
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for b in truth {
        // A model can't find a class it wasn't trained on (a 3-class model on a 5-class image).
        if b.class >= classes || rng.f32() > 0.55 + 0.44 * q {
            continue;
        }
        let (bw, bh) = (b.x2 - b.x1, b.y2 - b.y1);
        let j = (1.0 - q) * 0.12 + 0.015;
        let x1 = (b.x1 + rng.normal() * j * bw).clamp(0.0, w as f32);
        let y1 = (b.y1 + rng.normal() * j * bh).clamp(0.0, h as f32);
        let x2 = (b.x2 + rng.normal() * j * bw).clamp(x1 + 1.0, w as f32);
        let y2 = (b.y2 + rng.normal() * j * bh).clamp(y1 + 1.0, h as f32);
        let class = if rng.f32() < (1.0 - q) * 0.18 { rng.below(classes) } else { b.class };
        out.push(ScoredBox {
            class,
            name: name(class),
            score: (0.45 + 0.5 * q * rng.range(0.75, 1.0) + 0.04 * rng.normal()).clamp(0.26, 0.99),
            x1,
            y1,
            x2,
            y2,
        });
    }
    let fps = if rng.f32() < (1.0 - q) * 0.9 { 1 + rng.below(2) } else { 0 };
    for _ in 0..fps {
        let s = w.min(h) as f32 * rng.range(0.05, 0.15);
        let x1 = rng.range(0.0, w as f32 - s);
        let y1 = rng.range(0.0, h as f32 - s);
        let class = rng.below(classes);
        out.push(ScoredBox {
            class,
            name: name(class),
            score: rng.range(0.26, 0.48),
            x1,
            y1,
            x2: x1 + s,
            y2: y1 + s,
        });
    }
    out
}

type BBoxT = super::types::BBox;

const GPU_NAME: &str = "NVIDIA GeForce RTX 4080 SUPER";
/// `Run.device` of the demo's GPU runs, and of its CPU runs.
const GPU_DEVICE: &str = "NVIDIA GeForce RTX 4080 SUPER (vulkan)";
const CPU_DEVICE: &str = "AMD Ryzen 9 7950X (32 threads)";
const CPU_NAME: &str = "AMD Ryzen 9 7950X";
/// The reference machine (docs/BENCHMARKS.md), named on every reference row.
const REF_GPU: &str = "NVIDIA GeForce RTX 4080 SUPER";
const REF_CPU: &str = "Intel Core i7-14700KF";
const REF_NOTE: &str =
    "FOLO's measurement on its reference machine (docs/BENCHMARKS.md), scaled by compute to this size";

/// An architecture's compute at 640 px, GFLOPs (as New run lists them).
fn gflops(arch: Arch) -> f64 {
    match arch.as_str() {
        "folo-n" => 6.6,
        "folo-s" => 21.5,
        "folo-m" => 48.0,
        "folo-l" => 104.0,
        _ => 165.0,
    }
}

/// Reference training rate, img/s at batch 16 (the demo's stand-in for docs/BENCHMARKS.md).
fn reference_train(arch: Arch, imgsz: u32, amp: bool, cpu: bool) -> f64 {
    let base = match arch.as_str() {
        "folo-n" => 420.,
        "folo-s" => 260.,
        "folo-m" => 140.,
        "folo-l" => 80.,
        _ => 52.,
    };
    let px = (640. / imgsz as f64).powi(2).min(3.6);
    let amp = if amp { 1.15 } else { 1. };
    base * px * amp / if cpu { 25. } else { 1. }
}

/// Reference latency of one prediction, ms (decode included).
fn reference_latency(arch: Arch, imgsz: u32, f16: bool, cpu: bool) -> f64 {
    let base = match arch.as_str() {
        "folo-n" => 4.2,
        "folo-s" => 6.1,
        "folo-m" => 9.8,
        "folo-l" => 15.0,
        _ => 22.0,
    };
    let px = (imgsz as f64 / 640.).powi(2).max(0.3);
    base * px * if f16 { 0.7 } else { 1. } * if cpu { 12. } else { 1. }
}

#[allow(clippy::too_many_arguments)]
fn train_row(
    device: &str,
    on: &str,
    arch: Arch,
    imgsz: u32,
    batch: u32,
    amp: bool,
    rate: f64,
    source: ThroughputSource,
) -> TrainThroughput {
    TrainThroughput {
        device: device.into(),
        measured_on: on.into(),
        arch,
        imgsz,
        batch,
        amp,
        imgs_per_s: rate,
        mosaic: None,
        close_mosaic: None,
        val: None,
        startup_s: None,
        peak_vram_bytes: None,
        source,
        samples: 0,
        probes: 0,
        updated: None,
        note: (source == ThroughputSource::Reference).then(|| REF_NOTE.to_string()),
    }
}

fn infer_row(
    device: &str,
    on: &str,
    arch: Arch,
    imgsz: u32,
    f16: bool,
    latency: f64,
    source: ThroughputSource,
) -> InferThroughput {
    InferThroughput {
        device: device.into(),
        measured_on: on.into(),
        arch,
        imgsz,
        batch: 1,
        precision: if f16 { InferPrecision::F16 } else { InferPrecision::F32 },
        tensor_cores: f16,
        latency_ms: latency,
        latency_ms_low: latency * 0.9,
        latency_ms_high: latency * 1.2,
        inference_ms: Some(latency * 0.8),
        imgs_per_s: 1000. / latency,
        source,
        samples: 0,
        updated: None,
        note: (source == ThroughputSource::Reference).then(|| REF_NOTE.to_string()),
    }
}

/// What the demo has measured, as `folo serve --mock` has it: folo-n and folo-s at 640 from its
/// runs, a folo-n 320 px amp probe, and folo-n 640 predictions.
fn mock_measured() -> (Vec<TrainThroughput>, Vec<InferThroughput>) {
    let at = |hours: f64| Some(time::format(MOCK_NOW - (hours * 3600.) as i64));
    let training = vec![
        TrainThroughput {
            samples: 37,
            updated: at(4.0),
            ..train_row("gpu:0", GPU_NAME, Arch::N, 640, 16, false, 431.8, ThroughputSource::Measured)
        },
        TrainThroughput {
            samples: 118,
            updated: at(9.0),
            ..train_row("gpu:0", GPU_NAME, Arch::S, 640, 32, false, 247.3, ThroughputSource::Measured)
        },
        TrainThroughput {
            samples: 1,
            probes: 1,
            updated: at(26.0),
            ..train_row("gpu:0", GPU_NAME, Arch::N, 320, 16, true, 1392.0, ThroughputSource::Measured)
        },
    ];
    let inference = vec![InferThroughput {
        samples: 54,
        updated: at(2.0),
        ..infer_row("gpu:0", GPU_NAME, Arch::N, 640, false, 4.6, ThroughputSource::Measured)
    }];
    (training, inference)
}

/// The table as the server answers it: measured rows first (newest first), then a reference row
/// for every arch at 640 and 320 px (batch 16), f32 and (GPU) amp / f16, on the GPU and the CPU,
/// unless a measured row of that device has the same settings.
fn with_reference(
    mut training: Vec<TrainThroughput>,
    mut inference: Vec<InferThroughput>,
) -> ThroughputTable {
    training.sort_by(|a, b| b.updated.cmp(&a.updated));
    inference.sort_by(|a, b| b.updated.cmp(&a.updated));
    let mut ref_train = Vec::new();
    let mut ref_infer = Vec::new();
    for (device, cpu) in [("gpu:0", false), ("cpu", true)] {
        let on = if cpu { REF_CPU } else { REF_GPU };
        // Only a row measured on this very adapter / CPU replaces the reference (one from
        // another adapter under the same ordinal stays listed beside it).
        let here = if cpu { CPU_NAME } else { GPU_NAME };
        for arch in Arch::ALL {
            for imgsz in [640, 320] {
                for amp in [false, true] {
                    if cpu && amp {
                        continue;
                    }
                    let have = training.iter().any(|t| {
                        t.device == device
                            && t.measured_on == here
                            && t.arch == arch
                            && t.imgsz == imgsz
                            && t.batch == 16
                            && t.amp == amp
                    });
                    if !have {
                        let r = reference_train(arch, imgsz, amp, cpu);
                        ref_train.push(train_row(
                            device,
                            on,
                            arch,
                            imgsz,
                            16,
                            amp,
                            r,
                            ThroughputSource::Reference,
                        ));
                    }
                    let precision = if amp { InferPrecision::F16 } else { InferPrecision::F32 };
                    let have = inference.iter().any(|i| {
                        i.device == device
                            && i.measured_on == here
                            && i.arch == arch
                            && i.imgsz == imgsz
                            && i.precision == precision
                    });
                    if !have {
                        let l = reference_latency(arch, imgsz, amp, cpu);
                        ref_infer.push(infer_row(
                            device,
                            on,
                            arch,
                            imgsz,
                            amp,
                            l,
                            ThroughputSource::Reference,
                        ));
                    }
                }
            }
        }
    }
    training.extend(ref_train);
    inference.extend(ref_infer);
    ThroughputTable { device: "gpu:0".into(), device_name: GPU_NAME.into(), training, inference }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_as_the_trainer_does() {
        let cfg = |batch, epochs, optimizer| TrainConfig {
            name: "r".into(),
            dataset_id: "ds".into(),
            arch: Arch::N,
            epochs,
            batch,
            imgsz: 640,
            device: "auto".into(),
            optimizer,
            lr0: 0.01,
            weights: None,
            augment: Augment::default(),
            seed: 0,
            amp: false,
            head: Default::default(),
            compute: None,
        };
        // Warmup in optimizer steps, as folo_nn computes it.
        assert_eq!(resolve(&cfg(16, 30, Optimizer::Auto), 800, 3).warmup_iters, 100);
        assert_eq!(resolve(&cfg(16, 10, Optimizer::Auto), 800, 3).warmup_iters, 39);
        assert_eq!(resolve(&cfg(16, 1, Optimizer::Auto), 800, 3).warmup_iters, 0);
        // Auto: AdamW at 0.01 / (4 + nc) for short runs, with β1 0.9; SGD past 10,000 iterations
        // (the iteration count rounds batches up).
        let short = resolve(&cfg(16, 100, Optimizer::Auto), 3200, 3);
        assert_eq!((short.optimizer, short.lr0, short.momentum), (ResolvedOptimizer::Adamw, 0.001429, 0.9));
        assert_eq!(resolve(&cfg(32, 527, Optimizer::Auto), 1200, 3).optimizer, ResolvedOptimizer::Sgd);
    }

    #[test]
    fn frozen_mock_is_deterministic() {
        let a = MockBackend::new(4200, true);
        let b = MockBackend::new(4200, true);
        assert_eq!(a.runs().unwrap(), b.runs().unwrap());
        assert_eq!(a.metrics(LIVE_ID).unwrap(), b.metrics(LIVE_ID).unwrap());
        let live = a.run(LIVE_ID).unwrap();
        assert_eq!(live.status, RunStatus::Running);
        assert_eq!(live.epoch, 42);
        assert_eq!(a.metrics(LIVE_ID).unwrap().epochs.len(), 42);
    }

    #[test]
    fn every_epoch_and_eval_says_its_operating_point_and_source() {
        let m = MockBackend::new(4200, true);
        for r in m.runs().unwrap() {
            for e in m.metrics(&r.id).unwrap().epochs {
                let c = e
                    .val
                    .best_f1_conf
                    .unwrap_or_else(|| panic!("{} epoch {}: no best_f1_conf", r.id, e.epoch));
                assert!((0.001..1.0).contains(&c) && (c * 1000.0 - (c * 1000.0).round()).abs() < 1e-3, "{c}");
            }
        }
        for model in m.models().unwrap() {
            let ev = m.model_eval(&model.id).unwrap();
            assert_eq!(ev.best_f1_conf, model.metrics.unwrap().best_f1_conf, "{}", model.id);
            assert!(ev.best_f1_conf.is_some(), "{}", model.id);
            let run = m.run(model.run_id.as_deref().unwrap()).unwrap();
            if let Some(best) = run.best {
                let epochs = m.metrics(&run.id).unwrap().epochs;
                let best = epochs.iter().find(|e| e.epoch == best.epoch).unwrap();
                assert_eq!(
                    ev.best_f1_conf, best.val.best_f1_conf,
                    "{}: model vs its run's best epoch",
                    model.id
                );
            }
            assert_eq!(ev.dataset_id.as_deref(), Some(&*run.config.dataset_id), "{}", model.id);
            let ds = m.fixture(&run.config.dataset_id).unwrap();
            assert_eq!(ev.dataset_name.as_deref(), Some(&*ds.ds.name));
            assert_eq!((ev.split, ev.split_fallback.as_ref()), (Some(Split::Val), None));
            assert_eq!(ev.images as u64, ds.ds.splits.val);
            assert_eq!((ev.confusion_conf, ev.confusion_iou), (Some(0.25), Some(0.45)));
        }
    }

    #[test]
    fn learning_curves_improve() {
        let m = MockBackend::new(0, true);
        let metrics = m.metrics("run_7d1e").unwrap();
        let first = &metrics.epochs[0];
        let last = metrics.epochs.last().unwrap();
        assert!(last.val.map50_95 > first.val.map50_95 + 0.3);
        assert!(last.train_loss.total < first.train_loss.total);
        assert!(metrics.epochs.iter().all(|e| e.val.map50 >= e.val.map50_95));
    }

    #[test]
    fn image_pages_and_filters() {
        let m = MockBackend::new(0, true);
        let all = m.images("ds_8f2c", &ImageQuery { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(all.total, 1000);
        assert_eq!(all.images.len(), 10);
        let val = m
            .images("ds_8f2c", &ImageQuery { split: Some(Split::Val), limit: 500, ..Default::default() })
            .unwrap();
        assert_eq!(val.total, 200);
        let squares =
            m.images("ds_8f2c", &ImageQuery { class: Some(1), limit: 5, ..Default::default() }).unwrap();
        assert!(squares.images.iter().all(|im| im.boxes.iter().any(|b| b.class == 1)));
        match m.image("ds_8f2c", "img_00003", Some(128)).unwrap() {
            ImageData::Rgba { width, height, pixels } => {
                assert_eq!((width, height), (128, 128));
                assert_eq!(pixels.len(), 128 * 128 * 4);
            }
            ImageData::Encoded(_) => panic!("mock returns raw pixels"),
        }
        assert!(m.image("ds_8f2c", "img_99999", None).is_err());
    }

    #[test]
    fn stats_match_images() {
        let m = MockBackend::new(0, true);
        let ds = m.dataset("ds_3b91").unwrap();
        let stats = ds.stats.unwrap();
        assert_eq!(stats.class_counts.iter().sum::<u64>(), stats.boxes);
        assert_eq!(stats.boxes_per_image_hist.counts.iter().sum::<u64>(), ds.splits.total());
    }

    #[test]
    fn stop_and_errors() {
        let m = MockBackend::new(4200, true);
        let stopped = m.stop_run(LIVE_ID).unwrap();
        assert_eq!(stopped.status, RunStatus::Stopped);
        assert!(m.stop_run(LIVE_ID).is_err());
        assert!(MockBackend::new(0, true).failing().runs().is_err());
        assert!(MockBackend::new(0, true).empty().datasets().unwrap().is_empty());
    }

    #[test]
    fn stopping_flushes_the_stream() {
        let m = MockBackend::new(500, false);
        let rx = m.events(LIVE_ID).unwrap();
        std::thread::sleep(Duration::from_millis(150));
        let stopped = m.stop_run(LIVE_ID).unwrap();
        let mut last_step = 0;
        let mut saw_log = false;
        loop {
            match rx.recv_timeout(Duration::from_secs(5)).expect("stream ends with a status") {
                RunEvent::Step(s) => last_step = s.step,
                RunEvent::Log(l) => saw_log |= l.message.contains("checkpoint"),
                RunEvent::Status(s) if s.status == RunStatus::Stopped => break,
                _ => {}
            }
        }
        assert_eq!(last_step, stopped.step, "every step up to the stop point is delivered");
        assert!(saw_log, "the stop's log lines are delivered");
    }

    #[test]
    fn no_close_mosaic_still_learns() {
        let mut c = MockBackend::new(0, true).runs.lock().unwrap()[0].curve.clone();
        c.close_mosaic = 0;
        let (early, late) = (c.quality(1), c.quality(c.epochs));
        assert!(early.is_finite() && late.is_finite());
        assert!(late > early + 0.3, "quality rises over training: {early} -> {late}");
    }

    #[test]
    fn immediate_stop_flushes_exactly_to_the_stop_point() {
        let m = MockBackend::new(500, false);
        let rx = m.events(LIVE_ID).unwrap();
        // Stop before the worker's first tick.
        let stopped = m.stop_run(LIVE_ID).unwrap();
        let mut steps = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(5)).expect("stream ends with a status") {
                RunEvent::Step(s) => steps.push(s.step),
                RunEvent::Status(s) if s.status == RunStatus::Stopped => break,
                _ => {}
            }
        }
        assert!(steps.iter().all(|&s| s <= stopped.step), "no step beyond the stop point");
        assert!(steps.windows(2).all(|w| w[1] == w[0] + 1), "no gaps");
        if let Some(&last) = steps.last() {
            assert_eq!(last, stopped.step);
        }
    }

    fn cloud_config(offer: &str, max_hours: f32) -> TrainConfig {
        let mut c = MockBackend::new(0, true).run("run_7d1e").unwrap().config;
        c.name = "cloudy".into();
        c.device = "auto".into();
        c.compute = Some(Compute {
            connection_id: RUNPOD_CONNECTION.into(),
            offer_id: offer.into(),
            max_hours,
            disk_gb: None,
        });
        c
    }

    #[test]
    fn cloud_fixtures_upload_train_and_release() {
        let m = MockBackend::new(0, true);
        let syncing = m.run("run_c1d0").unwrap();
        let c = syncing.cloud.as_ref().expect("a cloud run");
        assert_eq!(
            (syncing.status, c.phase, syncing.queue_position),
            (RunStatus::Queued, CloudPhase::Syncing, None)
        );
        let p = c.progress.expect("an upload in progress");
        assert!((0.2..0.3).contains(&(p.done_bytes as f64 / p.total_bytes as f64)), "{p:?}");
        assert!(syncing.device.is_empty() && syncing.started.is_none());
        let training = m.run("run_c2e1").unwrap();
        let c = training.cloud.as_ref().unwrap();
        assert_eq!(
            (training.status, c.phase, training.step),
            (RunStatus::Running, CloudPhase::Training, 4400)
        );
        // 2.4 hours at $0.69/h so far.
        assert!((c.cost_usd.unwrap() - 0.69 * 2.4).abs() < 0.01, "{:?}", c.cost_usd);
        assert!(training.device.contains("RTX 4090") && c.deadline.is_some() && c.synced.is_some());
        let released = m.run("run_c9f1").unwrap();
        let c = released.cloud.as_ref().unwrap();
        assert_eq!((released.status, c.phase), (RunStatus::Completed, CloudPhase::Released));
        assert!(released.model_id.is_some() && released.finished.is_some());
        assert!((1.5..2.5).contains(&c.cost_usd.unwrap()), "final cost {:?}", c.cost_usd);
        // Instances: the two alive first, then the released one's, terminated.
        let instances = m.cloud_instances().unwrap();
        let alive: Vec<bool> = instances.iter().map(|i| i.status.is_alive()).collect();
        assert_eq!(alive, [true, true, false]);
        assert_eq!(instances[2].status, CloudInstanceStatus::Terminated);
        // Its estimate is the reference's, with what it would cost; the offer's price times the limit.
        let e = m.estimate_run(&cloud_config(RTX4090_OFFER, 12.)).unwrap();
        let cost = e.cost.expect("a priced offer has a cost");
        assert_eq!((e.device.as_str(), e.source), ("cloud", ThroughputSource::Reference));
        assert!(
            (cost.max_usd - 8.28).abs() < 1e-9
                && cost.est_usd_low < cost.est_usd
                && cost.est_usd < cost.est_usd_high
        );
        assert!(m.estimate_run(&m.run("run_7d1e").unwrap().config).unwrap().cost.is_none());
    }

    #[test]
    fn cloud_requests_are_checked_as_the_server_checks_them() {
        let m = MockBackend::new(0, true);
        let refused = m.create_run(cloud_config("NVIDIA GeForce RTX 4090@community", 12.)).unwrap_err();
        assert_eq!(refused.code, "cloud_refused");
        let too_long = m.create_run(cloud_config(RTX4090_OFFER, 1000.)).unwrap_err();
        assert!(too_long.code == "invalid_config" && too_long.message.contains("max_hours"), "{too_long:?}");
        let mut on_gpu = cloud_config(RTX4090_OFFER, 12.);
        on_gpu.device = "gpu:0".into();
        assert_eq!(m.create_run(on_gpu).unwrap_err().code, "invalid_config");
        assert_eq!(m.create_run(cloud_config("nope", 12.)).unwrap_err().code, "offer_not_found");
        let run = m.create_run(cloud_config(RTX4090_OFFER, 12.)).unwrap();
        assert_eq!(
            (run.status, run.cloud.map(|c| c.phase)),
            (RunStatus::Queued, Some(CloudPhase::Provisioning))
        );
        // Connections: a secret with "bad" is rejected; a required field missing names it; the
        // rest is kept with the secret as a hint only.
        let req = |key: &str| ConnectCloud {
            provider: "runpod".into(),
            name: String::new(),
            values: [("api_key".to_string(), key.to_string())].into(),
        };
        assert_eq!(m.connect_cloud(&req("rpa_bad_9")).unwrap_err().code, "cloud_auth_failed");
        let missing = m.connect_cloud(&req("")).unwrap_err();
        assert!(missing.code == "invalid_body" && missing.message.contains("API key"), "{missing:?}");
        let c = m.connect_cloud(&req("rpa_SECRETa1b2")).unwrap();
        assert_eq!((c.name.as_str(), c.values["api_key"].as_str()), ("RunPod 2", "••••a1b2"));
        assert!(!format!("{c:?}").contains("SECRET"));
        // A connection with live instances stays; one without goes.
        assert_eq!(m.disconnect_cloud(RUNPOD_CONNECTION).unwrap_err().code, "connection_in_use");
        m.disconnect_cloud(&c.id).unwrap();
        assert_eq!(m.check_cloud(&c.id).unwrap_err().code, "connection_not_found");
    }

    #[test]
    fn terminating_an_instance_fails_its_run() {
        let m = MockBackend::new(0, true);
        let training = m.run("run_c2e1").unwrap();
        let id = training.cloud.unwrap().instance_id.unwrap();
        let gone = m.terminate_instance(&id).unwrap();
        assert_eq!(gone.status, CloudInstanceStatus::Terminated);
        let run = m.run("run_c2e1").unwrap();
        assert_eq!((run.status, run.error_code.as_deref()), (RunStatus::Failed, Some("cloud_instance_lost")));
        assert_eq!(m.terminate_instance("ci_nope").unwrap_err().code, "instance_not_found");
    }

    #[test]
    fn cloud_variants() {
        let none = MockBackend::new(0, true).without_cloud();
        assert!(none.cloud_connections().unwrap().is_empty() && none.cloud_instances().unwrap().is_empty());
        assert!(none.runs().unwrap().iter().all(|r| r.cloud.is_none() && r.config.compute.is_none()));
        let trouble = MockBackend::new(0, true).cloud_trouble();
        let conn = &trouble.cloud_connections().unwrap()[0];
        assert!(conn.status == CloudConnectionStatus::Error && conn.error.is_some());
        assert_eq!(trouble.cloud_offers(&conn.id, false).unwrap_err().code, "cloud_auth_failed");
        let stuck =
            trouble.cloud_instances().unwrap().into_iter().find(|i| i.error.is_some()).expect("a stuck one");
        assert_eq!(stuck.status, CloudInstanceStatus::Terminating);
        let run = trouble.run(stuck.run_id.as_deref().unwrap()).unwrap();
        assert_eq!((run.status, run.cloud.unwrap().phase), (RunStatus::Stopped, CloudPhase::Released));
    }

    #[test]
    fn a_cloud_run_streams_its_phases() {
        let m = MockBackend::new(0, false);
        let mut config = cloud_config(RTX4090_OFFER, 12.);
        config.epochs = 1;
        let run = m.create_run(config).unwrap();
        // The demo's 2 s phases and training, shortened.
        if let Some(sim) =
            m.runs.lock().unwrap().iter_mut().find(|f| f.run.id == run.id).and_then(|f| f.cloud.as_mut())
        {
            (sim.pre, sim.step_dt) = ([0.05, 0.05, 0.3], 0.01);
            sim.anchor = (sim.train_at(), 0);
        }
        let rx = m.events(&run.id).unwrap();
        let (mut phases, mut statuses, mut uploads) = (Vec::new(), Vec::new(), 0);
        while let Ok(ev) = rx.recv_timeout(Duration::from_secs(5)) {
            match ev {
                RunEvent::Cloud(c) => {
                    uploads += usize::from(c.phase == CloudPhase::Syncing && c.progress.is_some());
                    if phases.last() != Some(&c.phase) {
                        phases.push(c.phase);
                    }
                }
                RunEvent::Status(s) => statuses.push(s.status),
                _ => {}
            }
        }
        assert!(phases.windows(2).all(|w| w[0] < w[1]), "phases in order: {phases:?}");
        assert!(
            phases.contains(&CloudPhase::Training) && phases.last() == Some(&CloudPhase::Released),
            "{phases:?}"
        );
        assert!(uploads > 0, "the upload reports its progress");
        assert_eq!(statuses.first(), Some(&RunStatus::Queued));
        assert_eq!(statuses.last(), Some(&RunStatus::Completed), "{statuses:?}");
        assert!(statuses.contains(&RunStatus::Running));
    }

    #[test]
    fn predictions_respect_threshold() {
        let m = MockBackend::new(0, true);
        let req = PredictRequest {
            model_id: "m_31aa".into(),
            input: PredictInput::Dataset { dataset_id: "ds_8f2c".into(), image_id: "img_00810".into() },
            conf: 0.5,
            iou: 0.7,
            max_det: 300,
        };
        let p = m.predict(&req).unwrap();
        assert!(p.boxes.iter().all(|b| b.score >= 0.5));
        assert!(p.timing.total_ms > p.timing.inference_ms);
    }
}
