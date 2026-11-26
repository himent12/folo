//! In-process fixtures behind [`Backend`]: three synthetic datasets, a handful of models, finished,
//! failed, stopped and queued runs, and one live run whose metrics advance in real time.
//!
//! Everything is a pure function of seeds and the simulated clock, so a frozen clock gives
//! pixel-identical screenshots.

mod backend;
mod cloud;
mod datasets;
mod events;
mod fixtures;
mod hold;
mod predictions;
mod throughput;
mod training;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::time;
use super::*;
use cloud::Ended;
pub(super) use datasets::DatasetFixture;
use training::{Clock, RunFixture, initial_logs};

/// The fixtures' "now": 2026-11-08T07:30:00Z.
pub const MOCK_NOW: i64 = 1_794_123_000;

/// Seconds per training step of a live mock run.
const STEP_DT: f64 = 0.12;

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
    /// Vision benchmarks and coding-agent fixtures (lazily seeded on first use).
    benches: Arc<super::mock_bench::MockBench>,
    /// Datasets unregistered and runs deleted through the API (a deleted run's models go with it).
    removed: Mutex<std::collections::HashSet<String>>,
    /// Calls a test holds back once answered ([`MockBackend::hold`]).
    holds: Mutex<Vec<(&'static str, Arc<hold::Gate>)>>,
    /// An image-list call from this offset on fails (once when `once` is true, else until cleared).
    fail_images_from: Mutex<Option<(u32, ApiError, bool)>>,
    /// Every image-list call made, in order ([`MockBackend::image_lists`]).
    image_lists: Mutex<Vec<(String, ImageQuery)>>,
    /// `/system` calls made ([`MockBackend::system_calls`]).
    system_calls: std::sync::atomic::AtomicUsize,
    /// Every image's pixels fail while set, as a cut-off answer would ([`MockBackend::fail_pixels`]).
    fail_pixels: std::sync::atomic::AtomicBool,
    /// The next run-history read fails ([`MockBackend::fail_metrics_once`]).
    fail_metrics: std::sync::atomic::AtomicBool,
    /// The next run-stream open fails ([`MockBackend::fail_events_once`]).
    fail_events: std::sync::atomic::AtomicBool,
    /// The data dir `/system` names ([`MockBackend::with_data_dir`]; none by default, as the demo).
    data_dir: Option<String>,
    /// The next instance-list read fails ([`MockBackend::fail_instances_once`]).
    fail_instances: std::sync::atomic::AtomicBool,
    /// Datasets whose own read fails with this (code, message) until mended
    /// ([`MockBackend::break_dataset`]).
    broken: Mutex<HashMap<String, (String, String)>>,
    /// Per-dataset revision overrides returned by `dataset(id)` ([`MockBackend::set_dataset_revision`]).
    revisions: Mutex<HashMap<String, String>>,
    clock: Clock,
}

const LIVE_ID: &str = "run_a19f";
/// [`MockBackend::with_cloud_runs`]'s runs: uploading its dataset, and training on its instance.
const CLOUD_SYNCING_ID: &str = "run_c1d0";
const CLOUD_TRAINING_ID: &str = "run_c2e1";

impl MockBackend {
    /// `live_step`: where the live run starts. `frozen`: keep the clock still (screenshots).
    pub fn new(live_step: u32, frozen: bool) -> Self {
        Self::build(live_step, frozen, false)
    }

    /// The demo with two cloud runs under way besides: one uploading its dataset, one training on
    /// its instance (the cloud screenshots).
    pub fn with_cloud_runs(live_step: u32, frozen: bool) -> Self {
        Self::build(live_step, frozen, true)
    }

    #[allow(dead_code)]
    pub fn without_bench(mut self) -> Self {
        self.benches = Arc::new(super::mock_bench::MockBench::disabled(self.frozen, self.started));
        self
    }

    /// `/system` names `dir` as its data dir, as a local `folo serve` does.
    #[cfg(test)]
    pub fn with_data_dir(mut self, dir: &str) -> Self {
        self.data_dir = Some(dir.into());
        self
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

    /// The RunPod key revoked, and a training cloud run ([`MockBackend::with_cloud_runs`]) stopped ten
    /// minutes ago with a termination RunPod hasn't confirmed: Settings › Cloud's problem states.
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

    /// The training cloud run's worker ([`MockBackend::with_cloud_runs`]) silent for three minutes:
    /// the run page's "Worker not answering".
    pub fn cloud_silent(self) -> Self {
        for f in self.runs.lock().unwrap().iter_mut() {
            if let Some(sim) =
                f.cloud.as_mut().filter(|s| s.elapsed() >= s.train_at() && s.released_at().is_none())
            {
                sim.silent_since = Some(sim.elapsed() - 180.);
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
            .find(|d| d.ds.id == id && !self.is_removed(id))
            .ok_or_else(|| ApiError::new("dataset_not_found", format!("No dataset with id {id}")))
    }

    /// Grows a dataset to `n` val images past its own (copies of its last image under new ids), as
    /// large as a test needs (a COCO-sized list).
    #[cfg(test)]
    pub fn with_more_val_images(mut self, id: &str, n: usize) -> Self {
        let f = self.datasets.iter_mut().find(|d| d.ds.id == id).expect("a demo dataset");
        let last = f.images.last().cloned().expect("an image");
        let start = f.images.len();
        f.images.extend((0..n).map(|i| ImageInfo {
            id: format!("img_{}", start + i + 1),
            split: Split::Val,
            ..last.clone()
        }));
        f.ds.splits.val += n as u64;
        self
    }

    /// Replaces dataset `id`'s images with `train` train images followed by `val` val images, each
    /// with a unique relative `file` path (e.g. to place all val images on a later image page).
    #[cfg(test)]
    pub fn with_paged_splits(mut self, id: &str, train: usize, val: usize) -> Self {
        let f = self.datasets.iter_mut().find(|d| d.ds.id == id).expect("a demo dataset");
        let template = f.images.first().cloned().expect("an image");
        let mut images = Vec::with_capacity(train + val);
        for i in 0..train {
            images.push(ImageInfo {
                id: format!("img_{}", i + 1),
                file: format!("images/train/{:05}.jpg", i + 1),
                split: Split::Train,
                ..template.clone()
            });
        }
        for i in 0..val {
            images.push(ImageInfo {
                id: format!("img_{}", train + i + 1),
                file: format!("images/val/{:05}.jpg", i + 1),
                split: Split::Val,
                ..template.clone()
            });
        }
        f.images = images;
        f.ds.splits.train = train as u64;
        f.ds.splits.val = val as u64;
        f.ds.splits.test = 0;
        self
    }

    /// Takes the boxes off a dataset's first `n` images: background images listed first.
    #[cfg(test)]
    pub fn with_background_first(mut self, id: &str, n: usize) -> Self {
        let f = self.datasets.iter_mut().find(|d| d.ds.id == id).expect("a demo dataset");
        f.images.iter_mut().take(n).for_each(|im| im.boxes.clear());
        self
    }

    /// The next run-history read fails as an unreachable server would.
    #[cfg(test)]
    pub fn fail_metrics_once(&self) {
        self.fail_metrics.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// The next run-stream open fails as an unreachable server would.
    #[cfg(test)]
    pub fn fail_events_once(&self) {
        self.fail_events.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// The next instance-list read fails as an unreachable server would.
    #[cfg(test)]
    pub fn fail_instances_once(&self) {
        self.fail_instances.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// The next image-list call at `offset` or later fails as an unreachable server would.
    #[cfg(test)]
    pub fn fail_images_once_from(&self, offset: u32) {
        self.fail_images_from_with(
            offset,
            ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878"),
            true,
        );
    }

    /// Image-list calls at `offset` or later fail with `error` (once if `once`, else until cleared).
    #[cfg(test)]
    pub fn fail_images_from_with(&self, offset: u32, error: ApiError, once: bool) {
        *self.fail_images_from.lock().unwrap() = Some((offset, error, once));
    }

    /// Stops failing image-list calls installed by [`Self::fail_images_from_with`].
    #[cfg(test)]
    pub fn clear_fail_images(&self) {
        *self.fail_images_from.lock().unwrap() = None;
    }

    /// The image-list calls made so far (dataset id and query), oldest first.
    #[cfg(test)]
    pub fn image_lists(&self) -> Vec<(String, ImageQuery)> {
        self.image_lists.lock().unwrap().clone()
    }

    /// The `/system` calls made so far.
    #[cfg(test)]
    pub fn system_calls(&self) -> usize {
        self.system_calls.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// From now on (`true`) or no longer, every image's pixels (dataset and benchmark ones, either
    /// size) fail as an answer cut off would, while the rest of the server answers.
    #[cfg(test)]
    pub fn fail_pixels(&self, on: bool) {
        self.fail_pixels.store(on, std::sync::atomic::Ordering::SeqCst);
    }

    /// The error an image fetch answers with while [`Self::fail_pixels`] is on.
    fn pixels_check(&self) -> ApiResult<()> {
        if self.fail_pixels.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(ApiError::new("bad_response", "folo serve's answer was cut off"));
        }
        Ok(())
    }

    /// A dataset whose training images all failed to read, as the server would count it.
    #[cfg(test)]
    pub fn with_no_training_images(mut self, id: &str) -> Self {
        let f = self.datasets.iter_mut().find(|d| d.ds.id == id).expect("a demo dataset");
        f.ds.splits.train = 0;
        self
    }

    /// Reading dataset `id` fails with `code` from now on, as a server answers for a folder moved
    /// away (`path_not_found`) or one that no longer reads as a dataset (`invalid_dataset`).
    #[cfg(test)]
    pub fn break_dataset(&self, id: &str, code: &str, message: &str) {
        self.broken.lock().unwrap().insert(id.to_string(), (code.to_string(), message.to_string()));
    }

    /// Dataset `id` reads again (its folder is back).
    #[cfg(test)]
    pub fn mend_dataset(&self, id: &str) {
        self.broken.lock().unwrap().remove(id);
    }

    /// Sets or clears the `revision` reported by `dataset(id)` for tests verifying revision checks.
    #[cfg(test)]
    pub fn set_dataset_revision(&self, id: &str, rev: Option<&str>) {
        let mut map = self.revisions.lock().unwrap();
        match rev {
            Some(r) => {
                map.insert(id.to_string(), r.to_string());
            }
            None => {
                map.remove(id);
            }
        }
    }

    /// Lists a removed dataset (or run) again, as registering its folder again does on a server.
    #[cfg(test)]
    pub fn restore(&self, id: &str) {
        self.removed.lock().unwrap().remove(id);
    }

    /// A dataset or run deleted through the API.
    fn is_removed(&self, id: &str) -> bool {
        self.removed.lock().unwrap().contains(id)
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::cloud::{COLLECT_S, CloudSim, RELEASE_S, RTX4090_OFFER, RUNPOD_CONNECTION, cloud_providers};
    use super::training::resolve;
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
        // Warmup in optimizer steps, as folo_nn computes it (3 epochs, no floor for new runs).
        assert_eq!(resolve(&cfg(16, 30, Optimizer::Auto), 800, 3).warmup_iters, 39);
        assert_eq!(resolve(&cfg(16, 10, Optimizer::Auto), 800, 3).warmup_iters, 39);
        assert_eq!(resolve(&cfg(16, 1, Optimizer::Auto), 800, 3).warmup_iters, 0);
        // Auto (rule 4): MuSGD at lr0 0.01, momentum 0.937, bias warmup 0.1, short or long; no betas.
        let short = resolve(&cfg(16, 100, Optimizer::Auto), 3200, 3);
        assert_eq!(
            (short.optimizer, short.lr0, short.momentum, short.warmup_bias_lr, short.betas),
            (ResolvedOptimizer::Musgd, 0.01, 0.937, 0.1, None)
        );
        assert_eq!(resolve(&cfg(32, 527, Optimizer::Auto), 1200, 3).optimizer, ResolvedOptimizer::Musgd);
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
        // Stop once the worker is streaming live steps.
        let mut last_step = 0;
        while last_step == 0 {
            if let RunEvent::Step(s) = rx.recv_timeout(Duration::from_secs(5)).expect("a live step") {
                last_step = s.step;
            }
        }
        let stopped = m.stop_run(LIVE_ID).unwrap();
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
    fn the_demo_lists_the_real_providers() {
        let real: Vec<CloudProvider> = folo_cloud::providers().iter().map(|p| p.info()).collect();
        assert_eq!(cloud_providers(), real);
    }

    #[test]
    fn the_default_demo_has_one_released_cloud_run() {
        let m = MockBackend::new(0, true);
        // The demo's runs as they were before cloud compute, one live; the oldest finished one
        // trained on RunPod.
        let runs = m.runs().unwrap();
        let ids: Vec<&str> = runs.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "run_a19f", "run_b7e2", "run_52e8", "run_e4f0", "run_1a2b", "run_7d1e", "run_c9f1",
                "run_d204", "run_0b5e"
            ]
        );
        assert_eq!(
            runs.iter().filter(|r| r.status.is_active()).count(),
            2,
            "the live run and the queued one"
        );
        assert_eq!(runs.iter().filter(|r| r.cloud.is_some()).count(), 1);
        let released = m.run("run_c9f1").unwrap();
        let c = released.cloud.as_ref().unwrap();
        assert_eq!((released.status, c.phase), (RunStatus::Completed, CloudPhase::Released));
        assert!(released.model_id.is_some() && released.device.contains("A100"));
        // It trained as long as it did on this machine (the Runs table's Duration, 9m 21s).
        let at = |t: &Option<String>| t.as_deref().and_then(time::parse).unwrap();
        assert_eq!(at(&released.finished) - at(&released.started), 561);
        // About 20 minutes of an A100 at $1.64/h, final.
        assert!((0.5..0.6).contains(&c.cost_usd.unwrap()), "final cost {:?}", c.cost_usd);
        let instances = m.cloud_instances().unwrap();
        assert_eq!(instances.len(), 1);
        assert_eq!(
            (instances[0].status, instances[0].run_id.as_deref()),
            (CloudInstanceStatus::Terminated, Some("run_c9f1"))
        );
        assert_eq!(m.cloud_connections().unwrap().len(), 1);
        assert!(m.system().unwrap().features.iter().any(|f| f == folo_api::FEATURE_CLOUD));
        // #382: its log ends as the real control plane's does, the weights copied back before the
        // instance goes (Studio's demo, `folo serve --mock`, says the same).
        let log: Vec<String> = m.logs("run_c9f1").unwrap().lines.into_iter().map(|l| l.message).collect();
        let at = |s: &str| log.iter().position(|l| l.contains(s)).unwrap_or_else(|| panic!("{s}: {log:#?}"));
        assert!(at("training on the instance") < at("copying its weights"));
        // Review round 1: its epochs were stamped on another clock than the end's lines, so the
        // last two came after the instance went. Each end line once, after the final epoch.
        let last = log.iter().rposition(|l| l.starts_with("epoch ")).unwrap_or_else(|| panic!("{log:#?}"));
        assert!(log[last].starts_with(&format!("epoch {}/", released.epochs)), "{}", log[last]);
        assert!(last < at("copying its weights"), "{log:#?}");
        let again: Vec<String> = m.logs("run_c9f1").unwrap().lines.into_iter().map(|l| l.message).collect();
        for end in ["copying its weights", "terminating the instance"] {
            assert_eq!(again.iter().filter(|l| l.contains(end)).count(), 1, "{end}");
        }
        assert!(at("copying its weights") < at("terminating the instance"));
    }

    /// Review round 1 (#382): the end of a cloud run's log follows its instance. On a remote FOLO
    /// server there is none: the weights are copied back, nothing is terminated. One terminated by
    /// hand while its weights were copied says so after "copying", whether its log was read before.
    #[test]
    fn a_cloud_runs_end_lines_follow_its_instance() {
        for (remote, read_first) in [(true, false), (false, false), (false, true)] {
            let m = MockBackend::new(0, true);
            let run = m.create_run(cloud_config(RTX4090_OFFER, 12.)).unwrap();
            let instance = {
                let mut runs = m.runs.lock().unwrap();
                let s = runs.iter_mut().find(|f| f.run.id == run.id).and_then(|f| f.cloud.as_mut()).unwrap();
                // Trained at once, and half way through copying the weights back.
                s.pre = [0., 0., 0.];
                s.anchor = (0., s.total);
                s.t0 -= if remote { COLLECT_S + 1. } else { COLLECT_S / 2. };
                if remote {
                    s.base.instance_id = None;
                }
                s.base.instance_id.clone()
            };
            let log =
                || -> Vec<String> { m.logs(&run.id).unwrap().lines.into_iter().map(|l| l.message).collect() };
            if read_first {
                log();
            }
            if let Some(id) = &instance {
                m.terminate_instance(id).unwrap();
            }
            let lines = log();
            let at = |s: &str| lines.iter().position(|l| l.contains(s));
            let case = format!("remote {remote}, read first {read_first}: {lines:#?}");
            assert!(at("copying its weights").is_some(), "{case}");
            assert_eq!(at("terminating the instance"), None, "{case}");
            if !remote {
                assert!(at("copying its weights") < at("terminated by hand"), "{case}");
            }
        }
    }

    /// Review round 2: terminating the instance of a run stopped while its checkpoint is copied
    /// keeps the step the run stopped at.
    #[test]
    fn terminating_a_stopped_runs_instance_keeps_its_stop() {
        let m = MockBackend::new(0, true);
        let run = m.create_run(cloud_config(RTX4090_OFFER, 12.)).unwrap();
        let ended = || {
            let runs = m.runs.lock().unwrap();
            runs.iter().find(|f| f.run.id == run.id).and_then(|f| f.cloud.as_ref()).unwrap().ended
        };
        let (instance, stop) = {
            let mut runs = m.runs.lock().unwrap();
            let s = runs.iter_mut().find(|f| f.run.id == run.id).and_then(|f| f.cloud.as_mut()).unwrap();
            s.pre = [0., 0., 0.];
            s.t0 -= 30.;
            let stop = s.elapsed();
            s.ended = Some((stop, Ended::Stopped));
            // Half a second later, still copying.
            s.t0 -= 0.5;
            (s.base.instance_id.clone().unwrap(), stop)
        };
        m.terminate_instance(&instance).unwrap();
        assert_eq!(ended(), Some((stop, Ended::InstanceLost)));
        // Its instance ended at the termination, not at the stop.
        let i = m.cloud_instances().unwrap().into_iter().find(|i| i.id == instance).unwrap();
        assert_eq!(i.status, CloudInstanceStatus::Terminated);
        let runs = m.runs.lock().unwrap();
        let s = runs.iter().find(|f| f.run.id == run.id).and_then(|f| f.cloud.as_ref()).unwrap();
        assert_eq!(i.ended, Some(s.at(stop + 0.5)));
    }

    /// Final check (#386): Desktop's merge of a run's log (the history, and the stream's lines
    /// buffered while it loads or added after) is left as it was; the demo's history and stream
    /// agree as a real server's do. Read as the run ends, its last epoch and end lines show once, in
    /// order: a line a read of the log returned is not streamed again.
    #[test]
    fn a_cloud_runs_log_read_as_it_ends_shows_each_line_once() {
        use crate::store::{Load, Store};
        let m = MockBackend::new(0, false);
        let run = m.create_run(cloud_config(RTX4090_OFFER, 12.)).unwrap();
        let sim = |edit: &dyn Fn(&mut CloudSim)| {
            let mut runs = m.runs.lock().unwrap();
            edit(runs.iter_mut().find(|f| f.run.id == run.id).and_then(|f| f.cloud.as_mut()).unwrap());
        };
        // Streamed from its last epoch; then trained, its weights copied and its instance gone.
        let spe = run.steps_per_epoch;
        sim(&|s| {
            s.pre = [0.; 3];
            s.anchor = (s.elapsed(), s.total - spe);
        });
        let rx = m.events(&run.id).unwrap();
        sim(&|s| s.t0 -= spe as f64 * s.step_dt + COLLECT_S + RELEASE_S + 1.);
        let mut store = Store::new(Arc::new(MockBackend::new(0, true)));
        store.logs.insert(run.id.clone(), Load::Loading);
        let history = m.logs(&run.id);
        // What the stream sent before the read is buffered while the history loads, then the rest.
        while let Ok(ev) = rx.try_recv() {
            store.apply(&run.id, ev);
        }
        store.install_logs(&run.id, history);
        loop {
            match rx.recv_timeout(Duration::from_secs(60)) {
                Ok(ev) => store.apply(&run.id, ev),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(e) => panic!("the stream did not end: {e}"),
            }
        }
        let log: Vec<&str> =
            store.logs[&run.id].ready().unwrap().lines.iter().map(|l| l.message.as_str()).collect();
        let last = format!("epoch {}/", run.epochs);
        let at = |s: &str| {
            let found: Vec<usize> = (0..log.len()).filter(|&i| log[i].contains(s)).collect();
            assert_eq!(found.len(), 1, "{s}: {log:#?}");
            found[0]
        };
        assert!(at(&last) < at("copying its weights"), "{log:#?}");
        assert!(at("copying its weights") < at("terminating the instance"), "{log:#?}");
    }

    #[test]
    fn cloud_runs_upload_and_train() {
        let m = MockBackend::with_cloud_runs(0, true);
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
        // Instances: the two alive first, then the released run's, terminated.
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
        let m = MockBackend::with_cloud_runs(0, true);
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
        let none = MockBackend::with_cloud_runs(0, true).without_cloud();
        assert!(none.cloud_connections().unwrap().is_empty() && none.cloud_instances().unwrap().is_empty());
        let runs = none.runs().unwrap();
        assert!(runs.iter().all(|r| r.cloud.is_none() && r.config.compute.is_none()));
        assert_eq!(
            runs.len(),
            MockBackend::new(0, true).runs().unwrap().len(),
            "the cloud runs under way go"
        );
        let trouble = MockBackend::with_cloud_runs(0, true).cloud_trouble();
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
        // The demo's phases take seconds. Here each lasts until the stream has shown it, then the
        // test moves the simulation on: no wait on the clock, however slow the machine.
        let sim = |edit: &dyn Fn(&mut CloudSim)| {
            let mut runs = m.runs.lock().unwrap();
            edit(runs.iter_mut().find(|f| f.run.id == run.id).and_then(|f| f.cloud.as_mut()).unwrap());
        };
        // Provisioning when the stream starts, then uploading: the stream sends a new phase at once.
        sim(&|s| s.pre = [1e6, 0., 1e6]);
        let rx = m.events(&run.id).unwrap();
        sim(&|s| {
            s.pre = [0., 0., 1e6];
            s.anchor = (s.train_at(), 0);
        });
        let (mut phases, mut statuses, mut uploads) = (Vec::new(), Vec::new(), 0);
        while let Ok(ev) = rx.recv_timeout(Duration::from_secs(5)) {
            match ev {
                RunEvent::Cloud(c) => {
                    uploads += usize::from(c.phase == CloudPhase::Syncing && c.progress.is_some());
                    if phases.last() != Some(&c.phase) {
                        phases.push(c.phase);
                        match c.phase {
                            // Uploading: finish it now, and train until told otherwise.
                            CloudPhase::Syncing => sim(&|s| {
                                s.pre[2] = s.elapsed();
                                s.step_dt = 1e6;
                                s.anchor = (s.train_at(), 0);
                            }),
                            // Training: finish every step now, and skip collecting and releasing.
                            CloudPhase::Training => sim(&|s| {
                                s.step_dt = 1e-6;
                                s.anchor = (s.elapsed(), 0);
                                s.t0 -= COLLECT_S + RELEASE_S + 1.;
                            }),
                            _ => {}
                        }
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
