//! Deterministic in-process Vision Benchmarks fixtures behind [`super::mock::MockBackend`].
//!
//! Uses `folo_bench` directly for the catalog, default plan, plan validation, score computation
//! and pixel-accurate condition variants on the procedural `Scene` images. Initialized lazily on
//! the first benchmark or agent-labeling call so non-benchmark tests pay zero setup cost.

mod benches;
mod entries;
mod images;
mod live;
mod loops;
mod scoring;
mod seed;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::mock::{DatasetFixture, MOCK_NOW};
use super::*;
use crate::synth::Scene;

use live::advance;
use seed::seed_data;

#[derive(Clone)]
struct BenchImgSource {
    id: String,
    file: String,
    width: u32,
    height: u32,
    boxes: Vec<BBox>,
    seed: u64,
    classes: u32,
    max_objects: u32,
    small: bool,
}

impl BenchImgSource {
    fn scene(&self) -> Scene {
        Scene::generate(self.seed, self.classes, self.max_objects, self.small)
    }
}

#[derive(Clone, Copy)]
struct EntrySkill {
    map: f32,
    fragility: f32,
    ece: f32,
    iou: f32,
    self_recall: f32,
}

#[derive(Clone)]
struct EntryFixture {
    entry: BenchEntry,
    skill: EntrySkill,
    /// Open external entry records accumulated before `finish`.
    open_records: Option<Vec<BenchPrediction>>,
    /// Precomputed result when scored from external predictions, or lazily built from `skill`.
    cached_result: Option<BenchResult>,
    /// Live start offset (done at `t = 0`, rate in passes/sec) when `!frozen`.
    live: Option<(u64, f64)>,
}

struct BenchFixture {
    bench: Bench,
    images: Vec<BenchImgSource>,
    entries: Vec<EntryFixture>,
    loops: Vec<BenchLoop>,
    /// Loops started through the API, by id: when they started (seconds of the demo's clock). They
    /// run through their generations as [`loops::step_loop`] says; the seeded ones keep their state.
    loop_live: HashMap<String, f64>,
}

/// Rendered images by (benchmark, image, condition, severity, width): width, height, RGBA pixels.
type VariantCache = HashMap<(String, String, String, u8, u32), (u32, u32, Arc<Vec<u8>>)>;

pub(super) struct MockBench {
    pub enabled: bool,
    frozen: bool,
    started: Instant,
    state: Mutex<Option<MockBenchData>>,
    variant_cache: Mutex<VariantCache>,
}

struct MockBenchData {
    seq: u64,
    benches: Vec<BenchFixture>,
    agents: Vec<LabelAgent>,
    jobs: Vec<LabelJob>,
}

impl MockBench {
    pub fn new(frozen: bool, started: Instant) -> Self {
        Self {
            enabled: true,
            frozen,
            started,
            state: Mutex::new(None),
            variant_cache: Mutex::new(HashMap::new()),
        }
    }

    /// A server without `vision_bench`: every call answers `not_supported`.
    pub fn disabled(frozen: bool, started: Instant) -> Self {
        Self { enabled: false, ..Self::new(frozen, started) }
    }

    fn elapsed(&self) -> f64 {
        if self.frozen { 0.0 } else { self.started.elapsed().as_secs_f64() }
    }

    fn sim_now(&self) -> i64 {
        if self.frozen { MOCK_NOW } else { MOCK_NOW + self.started.elapsed().as_secs() as i64 }
    }

    fn with<T>(
        &self,
        datasets: &[DatasetFixture],
        models: &[Model],
        f: impl FnOnce(&mut MockBenchData, f64, i64) -> ApiResult<T>,
    ) -> ApiResult<T> {
        if !self.enabled {
            return Err(super::no_bench());
        }
        let mut guard = self.state.lock().unwrap();
        let data = guard.get_or_insert_with(|| seed_data(datasets, models));
        let dt = self.elapsed();
        let now = self.sim_now();
        advance(data, dt, now);
        f(data, dt, now)
    }

    pub fn catalog(&self) -> ApiResult<BenchCatalog> {
        if !self.enabled {
            return Err(super::no_bench());
        }
        Ok(folo_bench::catalog())
    }

    pub fn agents(&self, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<Vec<LabelAgent>> {
        self.with(datasets, models, |d, _, _| Ok(d.agents.clone()))
    }

    pub fn jobs(&self, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<Vec<LabelJob>> {
        self.with(datasets, models, |d, _, _| Ok(d.jobs.clone()))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Helpers & Seed Data
// ---------------------------------------------------------------------------------------------------------------

fn find_bench<'a>(benches: &'a [BenchFixture], id: &str) -> ApiResult<&'a BenchFixture> {
    benches
        .iter()
        .find(|b| b.bench.id == id)
        .ok_or_else(|| ApiError::new("bench_not_found", format!("No benchmark {id}")))
}

fn find_bench_mut<'a>(benches: &'a mut [BenchFixture], id: &str) -> ApiResult<&'a mut BenchFixture> {
    benches
        .iter_mut()
        .find(|b| b.bench.id == id)
        .ok_or_else(|| ApiError::new("bench_not_found", format!("No benchmark {id}")))
}

fn find_entry<'a>(entries: &'a [EntryFixture], id: &str) -> ApiResult<&'a EntryFixture> {
    entries
        .iter()
        .find(|e| e.entry.id == id)
        .ok_or_else(|| ApiError::new("bench_entry_not_found", format!("No entry {id}")))
}

fn find_entry_mut<'a>(entries: &'a mut [EntryFixture], id: &str) -> ApiResult<&'a mut EntryFixture> {
    entries
        .iter_mut()
        .find(|e| e.entry.id == id)
        .ok_or_else(|| ApiError::new("bench_entry_not_found", format!("No entry {id}")))
}

fn sorted_entries(b: &BenchFixture) -> Vec<BenchEntry> {
    let fp = &b.bench.fingerprint;
    let tier = |e: &BenchEntry| -> u8 {
        if e.status != BenchEntryStatus::Done {
            3
        } else if e.fingerprint != *fp {
            2
        } else if e.score.as_ref().is_some_and(|s| s.partial) {
            1
        } else {
            0
        }
    };
    let mut list: Vec<BenchEntry> = b.entries.iter().map(|e| e.entry.clone()).collect();
    list.sort_by(|a, b| {
        let (ta, tb) = (tier(a), tier(b));
        if ta != tb {
            return ta.cmp(&tb);
        }
        if ta < 3 {
            let sa = a.score.as_ref().map_or(0.0, |s| s.total);
            let sb = b.score.as_ref().map_or(0.0, |s| s.total);
            sb.total_cmp(&sa).then_with(|| a.created.cmp(&b.created)).then_with(|| a.id.cmp(&b.id))
        } else {
            b.created.cmp(&a.created).then_with(|| a.id.cmp(&b.id))
        }
    });
    list
}

fn snapshot_bench(b: &BenchFixture) -> Bench {
    let sorted = sorted_entries(b);
    let best = sorted
        .iter()
        .find(|e| e.status == BenchEntryStatus::Done && e.fingerprint == b.bench.fingerprint)
        .and_then(|e| {
            let s = e.score.as_ref()?;
            Some(BenchBest {
                entry_id: e.id.clone(),
                name: e.name.clone(),
                score: s.total,
                partial: s.partial,
            })
        });
    let active =
        b.entries.iter().any(|e| e.entry.status.is_active()) || b.loops.iter().any(|l| l.status.is_active());
    Bench { entries: b.entries.len() as u64, best, active, ..b.bench.clone() }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::api::mock::MockBackend;
    use crate::api::*;

    fn model_entry(model_id: &str) -> StartBenchEntry {
        StartBenchEntry {
            contestant: BenchContestantSpec::Model { model_id: model_id.into() },
            name: None,
            max_images: None,
            conditions: None,
        }
    }

    #[test]
    fn demo_lists_three_benchmarks_and_ranks_finished_entries_first() {
        let m = MockBackend::new(0, true);
        let list = m.benches().unwrap();
        let ids: Vec<&str> = list.iter().map(|b| b.id.as_str()).collect();
        assert!(ids.contains(&"bn_3c90") && ids.contains(&"bn_5e12") && ids.contains(&"bn_7a04"), "{ids:?}");
        let entries = m.bench_entries("bn_3c90").unwrap();
        // Finished entries lead, best first; the unfinished ones follow.
        let first_open = entries.iter().position(|e| e.status != BenchEntryStatus::Done).unwrap();
        assert!(entries[first_open..].iter().all(|e| e.status != BenchEntryStatus::Done));
        let done: Vec<&BenchEntry> = entries.iter().filter(|e| e.status == BenchEntryStatus::Done).collect();
        let complete: Vec<f32> = done
            .iter()
            .filter(|e| !e.score.as_ref().unwrap().partial)
            .map(|e| e.score.as_ref().unwrap().total)
            .collect();
        assert!(complete.windows(2).all(|w| w[0] >= w[1]), "{complete:?}");
        // Without the feature, every call says so.
        let off = MockBackend::new(0, true).without_bench();
        assert_eq!(off.benches().unwrap_err().code, "not_supported");
        assert!(!off.system().unwrap().features.iter().any(|f| f == FEATURE_VISION_BENCH));
    }

    #[test]
    fn a_plan_changes_only_until_an_entry_runs_and_a_refusal_changes_nothing() {
        let m = MockBackend::new(0, true);
        let req = DesignBench { agent: LabelAgentId::Claude, model: None, effort: None, note: String::new() };
        let design = m.design_bench("bn_7a04", &req).unwrap();
        assert!(!design.plan.conditions.is_empty());
        let before = m.benches().unwrap().into_iter().find(|b| b.id == "bn_7a04").unwrap();
        let after = m
            .update_bench(
                "bn_7a04",
                &UpdateBench { name: None, description: None, plan: Some(design.plan.clone()) },
            )
            .unwrap();
        assert_eq!(after.plan.conditions.len(), design.plan.conditions.len());
        assert_ne!(after.fingerprint, before.fingerprint, "a new plan is a new test");
        // Once an entry runs, the plan is fixed; the refused request doesn't rename it either.
        let models = m.models().unwrap();
        m.start_bench_entry("bn_7a04", &model_entry(&models[0].id)).unwrap();
        let refused = m
            .update_bench(
                "bn_7a04",
                &UpdateBench { name: Some("renamed".into()), description: None, plan: Some(before.plan) },
            )
            .unwrap_err();
        assert_eq!(refused.code, "bench_frozen");
        let now = m.benches().unwrap().into_iter().find(|b| b.id == "bn_7a04").unwrap();
        assert_eq!(now.name, before.name);
    }

    #[test]
    fn external_boxes_with_reversed_corners_or_no_number_are_refused_not_scored() {
        let m = MockBackend::new(0, true);
        let mut images = Vec::new();
        loop {
            let page = m.bench_images("bn_7a04", images.len() as u32, 100, None).unwrap();
            let done = page.images.is_empty() || images.len() + page.images.len() >= page.total as usize;
            images.extend(page.images);
            if done {
                break;
            }
        }
        let upload = |x1: f32, x2: f32| StartBenchEntry {
            contestant: BenchContestantSpec::External {
                scored: false,
                open: false,
                predictions: images
                    .iter()
                    .map(|im| BenchPrediction {
                        image: im.id.clone(),
                        condition: None,
                        severity: None,
                        boxes: vec![BenchDetection { class: 0, score: 1.0, x1, y1: 10.0, x2, y2: 40.0 }],
                    })
                    .collect(),
            },
            name: Some("external".into()),
            max_images: None,
            conditions: None,
        };
        // An (x, y, width, height) box sent as corners, and an overflowed coordinate.
        let swapped = m.start_bench_entry("bn_7a04", &upload(50.0, 10.0)).unwrap_err();
        assert_eq!(swapped.code, "invalid_body");
        assert!(swapped.message.contains("is less than"), "{}", swapped.message);
        let inf = m.start_bench_entry("bn_7a04", &upload(10.0, f32::INFINITY)).unwrap_err();
        assert!(inf.message.contains("not a finite number"), "{}", inf.message);
        assert!(m.bench_entries("bn_7a04").unwrap().iter().all(|e| e.name != "external"));
        // A box of no area is a box.
        m.start_bench_entry("bn_7a04", &upload(10.0, 10.0)).unwrap();
    }

    #[test]
    fn entry_images_come_worst_first_and_conditions_keep_the_size() {
        let m = MockBackend::new(0, true);
        let page = m.bench_entry_images("bn_3c90", "be_3c01", "worst", 0, 24).unwrap();
        assert_eq!(page.images.len(), 24);
        let bad: Vec<u32> = page.images.iter().map(|i| i.missed + i.wrong).collect();
        assert!(bad.windows(2).all(|w| w[0] >= w[1]), "{bad:?}");
        assert_eq!(
            m.bench_entry_images("bn_3c90", "be_3c01", "best", 0, 24).unwrap_err().code,
            "invalid_query"
        );
        let first = m.bench_images("bn_3c90", 0, 1, None).unwrap().images.remove(0);
        let size = |d: ImageData| match d {
            ImageData::Rgba { width, height, pixels } => (width, height, pixels),
            ImageData::Encoded(_) => panic!("the mock renders pixels"),
        };
        let plain = size(m.bench_image("bn_3c90", &first.id, None, Some(160)).unwrap());
        let hazy = size(m.bench_image("bn_3c90", &first.id, Some(("haze", 3)), Some(160)).unwrap());
        assert_eq!((plain.0, plain.1), (hazy.0, hazy.1));
        assert_ne!(plain.2, hazy.2, "a condition changes the pixels");
    }

    #[test]
    fn entry_images_page_by_offset_past_the_first_hundred() {
        let m = MockBackend::new(0, true);
        let all = m.bench_entry_images("bn_3c90", "be_3c01", "order", 0, 1000).unwrap();
        assert!(all.total > 100 && all.images.len() as u64 == all.total, "{}", all.total);
        let page = m.bench_entry_images("bn_3c90", "be_3c01", "order", 120, 24).unwrap();
        assert_eq!(page.images[0].image.id, all.images[120].image.id);
    }

    #[test]
    fn the_stream_sends_what_changes_after_it_opens() {
        let m = MockBackend::new(0, false);
        let models = m.models().unwrap();
        let entry = m.start_bench_entry("bn_7a04", &model_entry(&models[0].id)).unwrap();
        let sub = m.bench_events("bn_7a04").unwrap();
        // The benchmark as it is when the stream opens.
        assert!(matches!(sub.recv_timeout(Duration::from_secs(3)).unwrap(), BenchEvent::Bench(_)));
        m.stop_bench_entry("bn_7a04", &entry.id).unwrap();
        let stopped = std::iter::from_fn(|| sub.recv_timeout(Duration::from_secs(3)).ok())
            .find_map(|ev| match ev {
                BenchEvent::Entry(e) if e.id == entry.id && !e.status.is_active() => Some(e),
                _ => None,
            })
            .expect("the stop is streamed");
        assert_eq!(stopped.status, BenchEntryStatus::Stopped);
        m.delete_bench_entry("bn_7a04", &entry.id).unwrap();
        let removed = std::iter::from_fn(|| sub.recv_timeout(Duration::from_secs(3)).ok())
            .any(|ev| matches!(ev, BenchEvent::Removed(r) if r.id == entry.id));
        assert!(removed, "the deletion is streamed");
    }
}
