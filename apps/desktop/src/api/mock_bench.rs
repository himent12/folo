//! Deterministic in-process Vision Benchmarks fixtures behind [`super::mock::MockBackend`].
//!
//! Uses `folo_bench` directly for the catalog, default plan, plan validation, score computation
//! and pixel-accurate condition variants on the procedural `Scene` images. Initialized lazily on
//! the first benchmark or agent-labeling call so non-benchmark tests pay zero setup cost.

use std::collections::HashMap;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_bench::Condition;

use super::mock::{DatasetFixture, MOCK_NOW};
use super::time;
use super::*;
use crate::synth::{Rng, Scene, hash};

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
    /// run through their generations as [`step_loop`] says; the seeded ones keep their state.
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

    pub fn list(&self, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<Vec<Bench>> {
        self.with(datasets, models, |d, _, _| {
            let mut out: Vec<Bench> = d.benches.iter().map(snapshot_bench).collect();
            out.sort_by(|a, b| b.created.cmp(&a.created));
            Ok(out)
        })
    }

    pub fn get(&self, id: &str, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<Bench> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            Ok(snapshot_bench(b))
        })
    }

    pub fn create(
        &self,
        req: &CreateBench,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<Bench> {
        let name = req.name.trim();
        if name.is_empty() || name.len() > 100 {
            return Err(ApiError::new("invalid_body", "Benchmark name must be 1 to 100 characters."));
        }
        if req.description.len() > 4000 {
            return Err(ApiError::new("invalid_body", "Description must be at most 4,000 characters."));
        }
        let max_images = req.max_images.unwrap_or(500);
        if !(1..=5000).contains(&max_images) {
            return Err(ApiError::new("invalid_body", "max_images must be between 1 and 5,000."));
        }
        let plan = match &req.plan {
            Some(p) => folo_bench::check_plan(p).map_err(|m| ApiError::new("invalid_body", m))?,
            None => folo_bench::default_plan(),
        };
        self.with(datasets, models, |d, _, now| {
            let (source, source_name, classes, pool) =
                resolve_source(&req.source, datasets, &d.jobs, Split::Val)?;
            if pool.is_empty() {
                return Err(ApiError::new("invalid_dataset", "The chosen source has no images."));
            }
            let sampled = sample_source(&pool, req.seed, max_images as usize);
            d.seq += 1;
            let id = format!("bn_{:012x}", hash(&format!("bn-{}-{name}", d.seq)) & 0xFFFF_FFFF_FFFF);
            let class_boxes = count_class_boxes(&sampled, classes.len());
            let total_boxes = class_boxes.iter().sum();
            let fp = fingerprint_of(req.seed, &classes, sampled.len(), &plan);
            let bench = Bench {
                id,
                name: name.to_string(),
                description: req.description.trim().to_string(),
                classes,
                created: time::format(now),
                source,
                source_name,
                images: sampled.len() as u64,
                boxes: total_boxes,
                class_boxes,
                plan,
                seed: req.seed,
                fingerprint: fp,
                entries: 0,
                best: None,
                active: false,
            };
            let out = bench.clone();
            d.benches.insert(
                0,
                BenchFixture {
                    bench,
                    images: sampled,
                    entries: Vec::new(),
                    loops: Vec::new(),
                    loop_live: HashMap::new(),
                },
            );
            Ok(out)
        })
    }

    pub fn update(
        &self,
        id: &str,
        req: &UpdateBench,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<Bench> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench_mut(&mut d.benches, id)?;
            // Everything is checked before anything changes: a refused request changes nothing.
            let plan = match &req.plan {
                Some(plan) => {
                    let frozen = b
                        .entries
                        .iter()
                        .any(|e| e.entry.status == BenchEntryStatus::Done || e.entry.status.is_active());
                    if frozen {
                        return Err(ApiError::new(
                            "bench_frozen",
                            "The plan is fixed once an entry has finished or is running.",
                        )
                        .with_hint(
                            "Delete the finished entries and stop active ones, or make a new benchmark.",
                        ));
                    }
                    Some(folo_bench::check_plan(plan).map_err(|m| ApiError::new("invalid_body", m))?)
                }
                None => None,
            };
            if req.name.as_ref().is_some_and(|n| n.trim().is_empty() || n.trim().len() > 100) {
                return Err(ApiError::new("invalid_body", "Name must be 1 to 100 characters."));
            }
            if req.description.as_ref().is_some_and(|d| d.len() > 4000) {
                return Err(ApiError::new("invalid_body", "Description must be at most 4,000 characters."));
            }
            if let Some(name) = &req.name {
                b.bench.name = name.trim().to_string();
            }
            if let Some(desc) = &req.description {
                b.bench.description = desc.trim().to_string();
            }
            if let Some(checked) = plan {
                b.bench.fingerprint =
                    fingerprint_of(b.bench.seed, &b.bench.classes, b.images.len(), &checked);
                b.bench.plan = checked;
            }
            Ok(snapshot_bench(b))
        })
    }

    pub fn delete(&self, id: &str, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<()> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            if snapshot_bench(b).active {
                return Err(ApiError::new(
                    "bench_active",
                    "Stop running entries and loops before deleting this benchmark.",
                ));
            }
            d.benches.retain(|x| x.bench.id != id);
            Ok(())
        })
    }

    pub fn design(
        &self,
        id: &str,
        req: &DesignBench,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchDesign> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let plan = propose_plan(&b.bench, req);
            let cost = (req.agent == LabelAgentId::Claude).then_some(0.0018);
            Ok(BenchDesign {
                plan,
                usage: LabelUsage {
                    calls: 1,
                    input_tokens: 1_840,
                    cached_input_tokens: 0,
                    output_tokens: 420,
                    cost_usd: cost,
                },
            })
        })
    }

    pub fn images(
        &self,
        id: &str,
        offset: u32,
        limit: u32,
        condition: Option<(&str, u8)>,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchImagePage> {
        let parsed = parse_cond_query(condition)?;
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let total = b.images.len() as u64;
            let lim = if limit == 0 { 100 } else { limit.min(1000) } as usize;
            let seed = b.bench.seed;
            let images = b
                .images
                .iter()
                .skip(offset as usize)
                .take(lim)
                .map(|im| {
                    let boxes = match parsed {
                        Some((Condition::Distance, sev)) => {
                            distance_boxes(&im.boxes, im.width, im.height, sev)
                        }
                        _ => im.boxes.clone(),
                    };
                    let _ = seed;
                    BenchImage {
                        id: im.id.clone(),
                        file: im.file.clone(),
                        width: im.width,
                        height: im.height,
                        boxes,
                    }
                })
                .collect();
            Ok(BenchImagePage { total, images })
        })
    }

    pub fn image_pixels(
        &self,
        id: &str,
        image_id: &str,
        condition: Option<(&str, u8)>,
        thumb: Option<u32>,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<ImageData> {
        let parsed = parse_cond_query(condition)?;
        let (bench_seed, src) = self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let im = b
                .images
                .iter()
                .find(|i| i.id == image_id)
                .cloned()
                .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
            Ok((b.bench.seed, im))
        })?;

        let (mut w, mut h) = (src.width, src.height);
        let mut scale = 1.0f32;
        if let Some(t) = thumb {
            let s = t as f32 / w.max(h) as f32;
            if s < 1.0 {
                scale = s;
                w = ((w as f32 * s).round() as u32).max(1);
                h = ((h as f32 * s).round() as u32).max(1);
            }
        }
        let cond_key = parsed.map_or((String::new(), 0), |(c, s)| (c.id().to_string(), s));
        let key = (id.to_string(), image_id.to_string(), cond_key.0, cond_key.1, w);
        if let Some((cw, ch, px)) = self.variant_cache.lock().unwrap().get(&key).cloned() {
            return Ok(ImageData::Rgba { width: cw, height: ch, pixels: px.to_vec() });
        }

        let rgba = src.scene().render(w, h);
        let pixels = match parsed {
            None => rgba,
            Some((cond, sev)) => {
                let mut rgb_raw = Vec::with_capacity((w * h * 3) as usize);
                for p in rgba.as_chunks::<4>().0 {
                    rgb_raw.extend_from_slice(&p[..3]);
                }
                let rgb = image::RgbImage::from_raw(w, h, rgb_raw)
                    .ok_or_else(|| ApiError::new("decode_failed", "Invalid scene dimensions"))?;
                let scaled_boxes: Vec<BBox> = src
                    .boxes
                    .iter()
                    .map(|b| BBox {
                        class: b.class,
                        x1: b.x1 * scale,
                        y1: b.y1 * scale,
                        x2: b.x2 * scale,
                        y2: b.y2 * scale,
                    })
                    .collect();
                let vseed = folo_bench::conditions::variant_seed(bench_seed, &src.id, cond, sev);
                let (out_rgb, _) = folo_bench::conditions::apply(cond, sev, &rgb, &scaled_boxes, vseed)
                    .map_err(|e| ApiError::new("invalid_query", e.to_string()))?;
                let mut out_rgba = Vec::with_capacity((w * h * 4) as usize);
                for p in out_rgb.as_raw().as_chunks::<3>().0 {
                    out_rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
                out_rgba
            }
        };
        let mut cache = self.variant_cache.lock().unwrap();
        if cache.len() > 400 {
            cache.clear();
        }
        cache.insert(key, (w, h, Arc::new(pixels.clone())));
        Ok(ImageData::Rgba { width: w, height: h, pixels })
    }

    pub fn entries(
        &self,
        id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<Vec<BenchEntry>> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            Ok(sorted_entries(b))
        })
    }

    pub fn start_entry(
        &self,
        id: &str,
        req: &StartBenchEntry,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchEntry> {
        self.with(datasets, models, |d, dt, now| {
            d.seq += 1;
            let seq = d.seq;
            let b = find_bench_mut(&mut d.benches, id)?;
            let eid = format!("be_{:04x}", (hash(&format!("be-{seq}")) & 0xFFFF).max(1));
            let created = format!("{}.{:03}Z", time::format(now).trim_end_matches('Z'), seq % 1000);
            let passes = plan_passes(&b.bench.plan) as u64;

            match &req.contestant {
                BenchContestantSpec::Model { model_id } => {
                    let m = models
                        .iter()
                        .find(|m| m.id == *model_id)
                        .ok_or_else(|| ApiError::new("model_not_found", format!("No model {model_id}")))?;
                    let missing = missing_classes(&b.bench.classes, &m.classes);
                    if missing.len() == b.bench.classes.len() {
                        return Err(ApiError::new(
                            "bench_class_mismatch",
                            format!("None of {}'s classes match {}.", m.name, b.bench.name),
                        ));
                    }
                    let imgs = req
                        .max_images
                        .map_or(b.images.len() as u64, |n| (n as u64).min(b.images.len() as u64));
                    if imgs == 0 {
                        return Err(ApiError::new("invalid_body", "max_images must be at least 1."));
                    }
                    let run_conds = req.conditions.unwrap_or(true) && !b.bench.plan.conditions.is_empty();
                    let total = imgs * (1 + if run_conds { passes } else { 0 });
                    let busy = b.entries.iter().any(|e| {
                        matches!(e.entry.contestant, BenchContestant::Model { .. })
                            && e.entry.status.is_active()
                    });
                    let base_map = m.metrics.as_ref().map_or(0.58, |x| x.map50_95)
                        * ((b.bench.classes.len() - missing.len()) as f32 / b.bench.classes.len() as f32);
                    let skill = EntrySkill {
                        map: base_map.clamp(0.15, 0.94),
                        fragility: 0.92,
                        ece: 0.042,
                        iou: 0.79,
                        self_recall: (base_map * 0.92).clamp(0.1, 0.88),
                    };
                    let entry = BenchEntry {
                        id: eid,
                        bench_id: b.bench.id.clone(),
                        name: req
                            .name
                            .clone()
                            .filter(|s| !s.trim().is_empty())
                            .unwrap_or_else(|| m.name.clone()),
                        contestant: BenchContestant::Model {
                            model_id: m.id.clone(),
                            arch: m.arch.as_str().to_string(),
                            params: m.params,
                            imgsz: m.imgsz,
                        },
                        status: if busy { BenchEntryStatus::Queued } else { BenchEntryStatus::Running },
                        created,
                        finished: None,
                        progress: BenchProgress { done: 0, total, condition: None, severity: None },
                        images: imgs,
                        conditions: run_conds,
                        scored: true,
                        fingerprint: b.bench.fingerprint.clone(),
                        missing_classes: missing,
                        score: None,
                        map50_95: None,
                        ci95: None,
                        conditions_run: 0,
                        speed: Some(BenchSpeed {
                            device: "gpu:0".into(),
                            ms_per_image: if m.arch == Arch::N { 2.4 } else { 4.1 },
                        }),
                        usage: None,
                        error: None,
                        hint: None,
                    };
                    let out = entry.clone();
                    b.entries.push(EntryFixture {
                        entry,
                        skill,
                        open_records: None,
                        cached_result: None,
                        live: Some((0, dt)),
                    });
                    Ok(out)
                }
                BenchContestantSpec::Agent { agent, model, effort } => {
                    let eff = effort.unwrap_or(LabelEffort::Medium);
                    let imgs = req.max_images.unwrap_or(48).min(b.images.len() as u32) as u64;
                    if imgs == 0 {
                        return Err(ApiError::new("invalid_body", "max_images must be at least 1."));
                    }
                    let run_conds = req.conditions.unwrap_or(false) && !b.bench.plan.conditions.is_empty();
                    let total = imgs * (1 + if run_conds { passes } else { 0 });
                    let default_name = match model.as_deref() {
                        Some(m) if !m.is_empty() => format!("{} ({m})", agent.display_name()),
                        _ => agent.display_name().to_string(),
                    };
                    let calls = total.div_ceil(if *agent == LabelAgentId::Claude { 4 } else { 8 });
                    let entry = BenchEntry {
                        id: eid,
                        bench_id: b.bench.id.clone(),
                        name: req.name.clone().filter(|s| !s.trim().is_empty()).unwrap_or(default_name),
                        contestant: BenchContestant::Agent {
                            agent: *agent,
                            model: model.clone(),
                            effort: eff,
                        },
                        status: BenchEntryStatus::Running,
                        created,
                        finished: None,
                        progress: BenchProgress { done: 0, total, condition: None, severity: None },
                        images: imgs,
                        conditions: run_conds,
                        scored: false,
                        fingerprint: b.bench.fingerprint.clone(),
                        missing_classes: Vec::new(),
                        score: None,
                        map50_95: None,
                        ci95: None,
                        conditions_run: 0,
                        speed: None,
                        usage: Some(LabelUsage {
                            calls,
                            input_tokens: calls * 1_600,
                            cached_input_tokens: 0,
                            output_tokens: calls * 260,
                            cost_usd: (*agent == LabelAgentId::Claude).then_some(calls as f64 * 0.0025),
                        }),
                        error: None,
                        hint: None,
                    };
                    let out = entry.clone();
                    b.entries.push(EntryFixture {
                        entry,
                        skill: EntrySkill {
                            map: 0.49,
                            fragility: 1.05,
                            ece: 0.0,
                            iou: 0.75,
                            self_recall: 0.34,
                        },
                        open_records: None,
                        cached_result: None,
                        live: Some((0, dt)),
                    });
                    Ok(out)
                }
                BenchContestantSpec::External { scored, open, predictions } => {
                    let name = req
                        .name
                        .as_deref()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| ApiError::new("invalid_body", "External entries require a name."))?
                        .to_string();
                    if *open {
                        let max_total = b.images.len() as u64 * (1 + passes);
                        let entry = BenchEntry {
                            id: eid,
                            bench_id: b.bench.id.clone(),
                            name,
                            contestant: BenchContestant::External,
                            status: BenchEntryStatus::Running,
                            created,
                            finished: None,
                            progress: BenchProgress {
                                done: predictions.len() as u64,
                                total: max_total,
                                condition: None,
                                severity: None,
                            },
                            images: b.images.len() as u64,
                            conditions: false,
                            scored: *scored,
                            fingerprint: b.bench.fingerprint.clone(),
                            missing_classes: Vec::new(),
                            score: None,
                            map50_95: None,
                            ci95: None,
                            conditions_run: 0,
                            speed: None,
                            usage: None,
                            error: None,
                            hint: None,
                        };
                        let out = entry.clone();
                        b.entries.push(EntryFixture {
                            entry,
                            skill: EntrySkill {
                                map: 0.6,
                                fragility: 0.95,
                                ece: 0.05,
                                iou: 0.78,
                                self_recall: 0.5,
                            },
                            open_records: Some(predictions.clone()),
                            cached_result: None,
                            live: None,
                        });
                        Ok(out)
                    } else {
                        let (entry, res) = score_external_entry(
                            b,
                            eid,
                            name,
                            created,
                            time::format(now),
                            *scored,
                            predictions,
                        )?;
                        let out = entry.clone();
                        b.entries.push(EntryFixture {
                            entry,
                            skill: EntrySkill {
                                map: 0.6,
                                fragility: 0.95,
                                ece: 0.05,
                                iou: 0.78,
                                self_recall: 0.5,
                            },
                            open_records: None,
                            cached_result: Some(res),
                            live: None,
                        });
                        Ok(out)
                    }
                }
            }
        })
    }

    pub fn result(
        &self,
        id: &str,
        entry_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchResult> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench_mut(&mut d.benches, id)?;
            let idx = b
                .entries
                .iter()
                .position(|e| e.entry.id == entry_id)
                .ok_or_else(|| ApiError::new("bench_entry_not_found", format!("No entry {entry_id}")))?;
            if b.entries[idx].entry.status != BenchEntryStatus::Done {
                return Err(ApiError::new("bench_entry_not_done", "The entry has not finished yet."));
            }
            if let Some(res) = &b.entries[idx].cached_result {
                return Ok(res.clone());
            }
            let res = synthesize_result(&b.bench, &b.images, &b.entries[idx].entry, b.entries[idx].skill);
            b.entries[idx].cached_result = Some(res.clone());
            Ok(res)
        })
    }

    pub fn entry_images(
        &self,
        id: &str,
        entry_id: &str,
        sort: &str,
        offset: u32,
        limit: u32,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchEntryImagePage> {
        if sort != "worst" && sort != "order" {
            return Err(ApiError::new("invalid_query", "sort must be `worst` or `order`."));
        }
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let ef = find_entry(&b.entries, entry_id)?;
            if ef.entry.status != BenchEntryStatus::Done {
                return Err(ApiError::new("bench_entry_not_done", "The entry has not finished yet."));
            }
            let n = (ef.entry.images as usize).min(b.images.len());
            let best_conf = if ef.entry.scored { (0.42 - 0.18 * ef.skill.map).clamp(0.15, 0.5) } else { 0.0 };
            let mut items: Vec<(usize, BenchEntryImage)> = b.images[..n]
                .iter()
                .enumerate()
                .map(|(idx, im)| {
                    let dets = synth_detections(im, &ef.entry, ef.skill, best_conf);
                    let out = folo_bench::score::outcome(
                        &folo_bench::Sample { truths: im.boxes.clone(), detections: dets },
                        b.bench.classes.len(),
                        ef.entry.scored,
                        best_conf,
                    );
                    (
                        idx,
                        BenchEntryImage {
                            image: BenchImage {
                                id: im.id.clone(),
                                file: im.file.clone(),
                                width: im.width,
                                height: im.height,
                                boxes: im.boxes.clone(),
                            },
                            detections: out.detections,
                            missed: out.missed,
                            wrong: out.wrong,
                        },
                    )
                })
                .collect();
            if sort == "worst" {
                items.sort_by(|a, b| {
                    (b.1.missed + b.1.wrong).cmp(&(a.1.missed + a.1.wrong)).then_with(|| a.0.cmp(&b.0))
                });
            }
            // As the server pages them: 100 by default, at most 1,000 per request.
            let lim = if limit == 0 { 100 } else { limit.min(1000) } as usize;
            Ok(BenchEntryImagePage {
                total: items.len() as u64,
                images: items.into_iter().skip(offset as usize).take(lim).map(|(_, x)| x).collect(),
            })
        })
    }

    pub fn stop_entry(
        &self,
        id: &str,
        entry_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchEntry> {
        self.with(datasets, models, |d, _, now| {
            let b = find_bench_mut(&mut d.benches, id)?;
            let ef = find_entry_mut(&mut b.entries, entry_id)?;
            if !ef.entry.status.is_active() {
                return Err(ApiError::new("bench_entry_not_active", "The entry is no longer running."));
            }
            ef.entry.status = BenchEntryStatus::Stopped;
            ef.entry.finished = Some(time::format(now));
            ef.open_records = None;
            ef.live = None;
            Ok(ef.entry.clone())
        })
    }

    pub fn delete_entry(
        &self,
        id: &str,
        entry_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<()> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench_mut(&mut d.benches, id)?;
            let ef = find_entry(&b.entries, entry_id)?;
            if ef.entry.status.is_active() {
                return Err(ApiError::new("bench_entry_active", "Stop the entry before deleting it."));
            }
            if b.loops.iter().any(|l| {
                l.status.is_active() && l.generations.iter().any(|g| g.entry_id.as_deref() == Some(entry_id))
            }) {
                return Err(ApiError::new(
                    "bench_entry_in_loop",
                    "A running self-improvement loop uses this entry.",
                ));
            }
            b.entries.retain(|e| e.entry.id != entry_id);
            Ok(())
        })
    }

    pub fn loops(
        &self,
        id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<Vec<BenchLoop>> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let mut out = b.loops.clone();
            out.sort_by(|a, b| b.created.cmp(&a.created));
            Ok(out)
        })
    }

    pub fn start_loop(
        &self,
        id: &str,
        req: &StartBenchLoop,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchLoop> {
        self.with(datasets, models, |d, dt, now| {
            let teacher = models
                .iter()
                .find(|m| m.id == req.teacher)
                .ok_or_else(|| ApiError::new("model_not_found", format!("No model {}", req.teacher)))?;
            d.seq += 1;
            let seq = d.seq;
            let b = find_bench_mut(&mut d.benches, id)?;
            if teacher.classes.len() != b.bench.classes.len()
                || !teacher
                    .classes
                    .iter()
                    .zip(&b.bench.classes)
                    .all(|(a, c)| a.eq_ignore_ascii_case(c))
            {
                return Err(ApiError::new(
                    "bench_class_mismatch",
                    format!("{} does not have {}'s classes in order.", teacher.name, b.bench.name),
                ));
            }
            let gens = req.generations.unwrap_or(1);
            if !(1..=5).contains(&gens) {
                return Err(ApiError::new("invalid_body", "generations must be 1 to 5."));
            }
            let lid = format!("bl_{:04x}", (hash(&format!("bl-{seq}")) & 0xFFFF).max(1));
            let name = req
                .name
                .clone()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| format!("{} loop", teacher.name));
            let teacher_entry = b.entries.iter().find(|e| {
                matches!(&e.entry.contestant, BenchContestant::Model { model_id, .. } if *model_id == teacher.id)
                    && e.entry.status == BenchEntryStatus::Done
                    && e.entry.fingerprint == b.bench.fingerprint
            });
            let g0 = BenchGeneration {
                generation: 0,
                labels: None,
                run_id: teacher.run_id.clone(),
                model_id: Some(teacher.id.clone()),
                model_name: Some(teacher.name.clone()),
                entry_id: teacher_entry.map(|e| e.entry.id.clone()),
                score: teacher_entry.and_then(|e| e.entry.score.as_ref().map(|s| s.total)),
                map50_95: teacher_entry.and_then(|e| e.entry.map50_95),
                ci95: teacher_entry.and_then(|e| e.entry.ci95),
            };
            let phase = if g0.map50_95.is_some() {
                BenchLoopPhase::Labeling
            } else {
                BenchLoopPhase::Benchmarking
            };
            let mut generations = vec![g0];
            if phase == BenchLoopPhase::Labeling {
                generations.push(BenchGeneration {
                    generation: 1,
                    labels: None,
                    run_id: None,
                    model_id: None,
                    model_name: None,
                    entry_id: None,
                    score: None,
                    map50_95: None,
                    ci95: None,
                });
            }
            let mut config = req.clone();
            config.generations = Some(gens);
            if config.student.arch.is_none() {
                config.student.arch = Some(teacher.arch.as_str().to_string());
            }
            if config.student.epochs.is_none() {
                config.student.epochs = Some(30);
            }
            if config.student.imgsz.is_none() {
                config.student.imgsz = Some(teacher.imgsz);
            }
            let lp = BenchLoop {
                id: lid,
                bench_id: b.bench.id.clone(),
                name,
                status: BenchLoopStatus::Running,
                phase: Some(phase),
                created: time::format(now),
                finished: None,
                config,
                generations,
                verdict: None,
                error: None,
                hint: None,
            };
            b.loops.insert(0, lp.clone());
            b.loop_live.insert(lp.id.clone(), dt);
            Ok(lp)
        })
    }

    pub fn stop_loop(
        &self,
        id: &str,
        loop_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchLoop> {
        self.with(datasets, models, |d, _, now| {
            let b = find_bench_mut(&mut d.benches, id)?;
            let lp = b
                .loops
                .iter_mut()
                .find(|l| l.id == loop_id)
                .ok_or_else(|| ApiError::new("bench_loop_not_found", format!("No loop {loop_id}")))?;
            if !lp.status.is_active() {
                return Err(ApiError::new("bench_loop_not_active", "The loop is no longer running."));
            }
            lp.status = BenchLoopStatus::Stopped;
            lp.phase = None;
            lp.finished = Some(time::format(now));
            Ok(lp.clone())
        })
    }

    pub fn delete_loop(
        &self,
        id: &str,
        loop_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<()> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench_mut(&mut d.benches, id)?;
            let lp = b
                .loops
                .iter()
                .find(|l| l.id == loop_id)
                .ok_or_else(|| ApiError::new("bench_loop_not_found", format!("No loop {loop_id}")))?;
            if lp.status.is_active() {
                return Err(ApiError::new("bench_loop_active", "Stop the loop before deleting it."));
            }
            b.loops.retain(|l| l.id != loop_id);
            Ok(())
        })
    }

    pub fn events(
        this: Arc<Self>,
        id: &str,
        datasets: Vec<DatasetFixture>,
        models: Vec<Model>,
    ) -> ApiResult<BenchSubscription> {
        let initial = this.get(id, &datasets, &models)?;
        let (tx, rx) = channel();
        let (sub, cancel) = BenchSubscription::new(rx);
        let _ = tx.send(BenchEvent::Bench(Box::new(initial)));
        if this.frozen {
            return Ok(sub);
        }
        let bid = id.to_string();
        // What the subscriber has seen, taken now (a change right after this call is streamed): each
        // tick sends what changed since, as the server does (an entry or loop that moved, one that is
        // gone, the benchmark's own record).
        let (bench0, entries0, loops0) = this.with(&datasets, &models, |d, _, _| {
            let b = find_bench(&d.benches, &bid)?;
            Ok((
                snapshot_bench(b),
                b.entries.iter().map(|e| e.entry.clone()).collect::<Vec<_>>(),
                b.loops.clone(),
            ))
        })?;
        let mut seen_bench = Some(bench0);
        let mut seen: HashMap<String, Option<BenchEntry>> =
            entries0.into_iter().map(|e| (e.id.clone(), Some(e))).collect();
        let mut seen_loops: HashMap<String, Option<BenchLoop>> =
            loops0.into_iter().map(|l| (l.id.clone(), Some(l))).collect();
        std::thread::spawn(move || {
            while !cancel.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(450));
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let snap = this.with(&datasets, &models, |d, _, _| {
                    let b = find_bench(&d.benches, &bid)?;
                    let entries: Vec<BenchEntry> = b.entries.iter().map(|e| e.entry.clone()).collect();
                    Ok((snapshot_bench(b), entries, b.loops.clone()))
                });
                let Ok((bench_snap, entries, loops)) = snap else { break };
                let mut out = Vec::new();
                for e in &entries {
                    if seen.get(&e.id).and_then(Option::as_ref) != Some(e) {
                        seen.insert(e.id.clone(), Some(e.clone()));
                        out.push(BenchEvent::Entry(Box::new(e.clone())));
                    }
                }
                for l in &loops {
                    if seen_loops.get(&l.id).and_then(Option::as_ref) != Some(l) {
                        seen_loops.insert(l.id.clone(), Some(l.clone()));
                        out.push(BenchEvent::Loop(Box::new(l.clone())));
                    }
                }
                // An entry or loop that is gone: one removal event names it (ids don't collide).
                let gone: Vec<String> = seen
                    .iter()
                    .filter(|(id, v)| v.is_some() && !entries.iter().any(|e| &e.id == *id))
                    .map(|(id, _)| id.clone())
                    .chain(
                        seen_loops
                            .iter()
                            .filter(|(id, v)| v.is_some() && !loops.iter().any(|l| &l.id == *id))
                            .map(|(id, _)| id.clone()),
                    )
                    .collect();
                for id in gone {
                    seen.insert(id.clone(), None);
                    seen_loops.insert(id.clone(), None);
                    out.push(BenchEvent::Removed(BenchRemoved { id }));
                }
                if seen_bench.as_ref() != Some(&bench_snap) {
                    seen_bench = Some(bench_snap.clone());
                    out.push(BenchEvent::Bench(Box::new(bench_snap)));
                }
                for ev in out {
                    if tx.send(ev).is_err() {
                        return;
                    }
                }
            }
        });
        Ok(sub)
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

fn parse_cond_query(q: Option<(&str, u8)>) -> ApiResult<Option<(Condition, u8)>> {
    let Some((c, s)) = q else { return Ok(None) };
    if c.is_empty() {
        return Ok(None);
    }
    let cond = Condition::parse(c)
        .ok_or_else(|| ApiError::new("invalid_query", format!("Unknown condition `{c}`.")))?;
    if !(1..=BENCH_SEVERITIES).contains(&s) {
        return Err(ApiError::new("invalid_query", format!("Severity {s} must be 1..={BENCH_SEVERITIES}.")));
    }
    Ok(Some((cond, s)))
}

fn distance_boxes(boxes: &[BBox], w: u32, h: u32, severity: u8) -> Vec<BBox> {
    const SCALES: [f32; 5] = [0.8, 0.65, 0.5, 0.375, 0.25];
    let k = SCALES[(severity.saturating_sub(1) as usize).min(4)];
    let (ox, oy) = (w as f32 * (1.0 - k) * 0.5, h as f32 * (1.0 - k) * 0.5);
    boxes
        .iter()
        .map(|b| BBox {
            class: b.class,
            x1: ox + b.x1 * k,
            y1: oy + b.y1 * k,
            x2: ox + b.x2 * k,
            y2: oy + b.y2 * k,
        })
        .collect()
}

fn plan_passes(plan: &BenchPlan) -> usize {
    plan.conditions.iter().map(|c| c.severities.len()).sum()
}

fn fingerprint_of(seed: u64, classes: &[String], images: usize, plan: &BenchPlan) -> String {
    let mut h1 = hash(&format!("v{}:{seed}:{images}:{}", folo_bench::ENGINE_VERSION, classes.join(",")));
    for c in &plan.conditions {
        h1 ^= hash(&format!("{}:{:?}:{:.3}", c.condition, c.severities, c.weight));
        h1 = h1.rotate_left(13).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    let h2 = hash(&format!("fp2:{h1:016x}:{}", plan.conditions.len()));
    let h3 = hash(&format!("fp3:{h2:016x}:{seed}"));
    let h4 = hash(&format!("fp4:{h3:016x}:{images}"));
    format!("{h1:016x}{h2:016x}{h3:016x}{h4:016x}")
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

fn advance(data: &mut MockBenchData, dt: f64, now: i64) {
    if dt <= 0.0 {
        return;
    }
    for b in &mut data.benches {
        let mut model_slot_busy = false;
        for ef in &mut b.entries {
            let is_model = matches!(ef.entry.contestant, BenchContestant::Model { .. });
            if is_model && ef.entry.status == BenchEntryStatus::Queued && !model_slot_busy {
                ef.entry.status = BenchEntryStatus::Running;
                ef.live = Some((0, dt));
            }
            if let Some((base, t0)) = ef.live
                && ef.entry.status == BenchEntryStatus::Running
            {
                let rate = if is_model { 180.0 } else { 14.0 };
                let done = (base + ((dt - t0).max(0.0) * rate) as u64).min(ef.entry.progress.total);
                ef.entry.progress.done = done;
                if done >= ef.entry.progress.total {
                    ef.entry.status = BenchEntryStatus::Done;
                    ef.entry.finished = Some(time::format(now));
                    ef.live = None;
                    let res = synthesize_result(&b.bench, &b.images, &ef.entry, ef.skill);
                    ef.entry.score = Some(res.entry.score.clone().unwrap());
                    ef.entry.map50_95 = Some(res.accuracy.map50_95);
                    ef.entry.ci95 = Some(res.accuracy.ci95);
                    ef.entry.conditions_run =
                        res.robustness.as_ref().map_or(0, |r| r.conditions.len() as u32);
                    ef.cached_result = Some(res);
                } else {
                    let (cond, sev) = pass_at(&b.bench.plan, ef.entry.images, ef.entry.conditions, done);
                    ef.entry.progress.condition = cond;
                    ef.entry.progress.severity = sev;
                }
            }
            if is_model && ef.entry.status.is_active() {
                model_slot_busy = true;
            }
        }
        for lp in &mut b.loops {
            if let Some(&t0) = b.loop_live.get(&lp.id)
                && lp.status == BenchLoopStatus::Running
            {
                step_loop(lp, dt - t0, now);
            }
        }
    }
}

/// Seconds a demo loop spends on one generation: labeling the pool, training the student, then
/// benchmarking it (a teacher without an entry here is benchmarked first, for [`LOOP_TEACHER_S`]).
const LOOP_GEN_S: f64 = 4.0;
const LOOP_TEACHER_S: f64 = 2.0;

/// A loop started `t` seconds ago, as the demo runs it: each generation's student scores a little
/// higher than the last, so a long enough loop improves.
fn step_loop(lp: &mut BenchLoop, t: f64, now: i64) {
    let gens = lp.config.generations.unwrap_or(1).max(1) as usize;
    let mut t = t.max(0.0);
    if lp.generations[0].map50_95.is_none() {
        if t < LOOP_TEACHER_S {
            lp.phase = Some(BenchLoopPhase::Benchmarking);
            return;
        }
        let g0 = &mut lp.generations[0];
        g0.score = Some(50.0);
        g0.map50_95 = Some(0.40);
        g0.ci95 = Some([0.37, 0.43]);
        t -= LOOP_TEACHER_S;
    }
    let base = lp.generations[0].clone();
    let (base_map, base_score) = (base.map50_95.unwrap_or(0.4), base.score.unwrap_or(50.0));
    let done = ((t / LOOP_GEN_S) as usize).min(gens);
    lp.generations.truncate(1);
    for k in 1..=done {
        let map = base_map + 0.03 * k as f32;
        lp.generations.push(BenchGeneration {
            generation: k as u32,
            labels: Some(BenchLabels {
                threshold: 0.5,
                images: 400,
                labeled_images: 380,
                boxes: 1500 + 20 * k as u64,
                excluded: 0,
                dataset_id: format!("ds_{}g{k}", lp.id),
            }),
            run_id: None,
            model_id: None,
            model_name: Some(format!("{} gen {k}-best", lp.name)),
            entry_id: None,
            score: Some(base_score + 2.0 * k as f32),
            map50_95: Some(map),
            ci95: Some([map - 0.02, map + 0.02]),
        });
    }
    if done == gens {
        // As docs/BENCH.md defines it: the last generation's mAP against the first teacher's interval,
        // its bounds included in Held.
        let last = lp.generations.last().and_then(|g| g.map50_95);
        let ci0 = base.ci95.unwrap_or([base_map, base_map]);
        lp.verdict = Some(match last {
            Some(m) if m > ci0[1] => BenchLoopVerdict::Improved,
            Some(m) if m < ci0[0] => BenchLoopVerdict::Regressed,
            _ => BenchLoopVerdict::Held,
        });
        lp.status = BenchLoopStatus::Done;
        lp.phase = None;
        lp.finished = Some(time::format(now));
        return;
    }
    let within = t - done as f64 * LOOP_GEN_S;
    lp.phase = Some(if within < 1.5 {
        BenchLoopPhase::Labeling
    } else if within < 3.5 {
        BenchLoopPhase::Training
    } else {
        BenchLoopPhase::Benchmarking
    });
    lp.generations.push(BenchGeneration {
        generation: done as u32 + 1,
        labels: None,
        run_id: None,
        model_id: None,
        model_name: None,
        entry_id: None,
        score: None,
        map50_95: None,
        ci95: None,
    });
}

fn pass_at(plan: &BenchPlan, images: u64, conditions: bool, done: u64) -> (Option<String>, Option<u8>) {
    let imgs = images.max(1);
    let pass = (done / imgs) as usize;
    if pass == 0 || !conditions {
        return (None, None);
    }
    let mut k = 1usize;
    for c in &plan.conditions {
        for &s in &c.severities {
            if k == pass {
                return (Some(c.condition.clone()), Some(s));
            }
            k += 1;
        }
    }
    (None, None)
}

fn missing_classes(bench_classes: &[String], model_classes: &[String]) -> Vec<String> {
    bench_classes
        .iter()
        .filter(|bc| !model_classes.iter().any(|mc| mc.eq_ignore_ascii_case(bc)))
        .cloned()
        .collect()
}

fn count_class_boxes(images: &[BenchImgSource], classes: usize) -> Vec<u64> {
    let mut counts = vec![0u64; classes];
    for im in images {
        for b in &im.boxes {
            if let Some(slot) = counts.get_mut(b.class as usize) {
                *slot += 1;
            }
        }
    }
    counts
}

fn sample_source(pool: &[BenchImgSource], seed: u64, max_images: usize) -> Vec<BenchImgSource> {
    let mut order: Vec<usize> = (0..pool.len()).collect();
    let mut rng = Rng::new(seed ^ 0x5A17_BEEF);
    for i in (1..order.len()).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        order.swap(i, j);
    }
    order
        .into_iter()
        .take(max_images)
        .enumerate()
        .map(|(idx, src_idx)| {
            let mut im = pool[src_idx].clone();
            im.id = (idx + 1).to_string();
            im
        })
        .collect()
}

fn resolve_source(
    source: &BenchSource,
    datasets: &[DatasetFixture],
    jobs: &[LabelJob],
    default_split: Split,
) -> ApiResult<(BenchSource, String, Vec<String>, Vec<BenchImgSource>)> {
    match source {
        BenchSource::Dataset { dataset_id, split } => {
            let sp = split.unwrap_or(default_split);
            let f = datasets
                .iter()
                .find(|d| d.ds.id == *dataset_id)
                .ok_or_else(|| ApiError::new("dataset_not_found", format!("No dataset {dataset_id}")))?;
            let pool: Vec<BenchImgSource> = f
                .images
                .iter()
                .enumerate()
                .filter(|(_, im)| im.split == sp)
                .map(|(i, im)| BenchImgSource {
                    id: im.id.clone(),
                    file: im.file.rsplit('/').next().unwrap_or(&im.file).to_string(),
                    width: im.width,
                    height: im.height,
                    boxes: im.boxes.clone(),
                    seed: f.seed ^ (i as u64).wrapping_mul(0x9E37),
                    classes: f.ds.classes.len() as u32,
                    max_objects: f.max_objects,
                    small: f.small,
                })
                .collect();
            Ok((
                BenchSource::Dataset { dataset_id: f.ds.id.clone(), split: Some(sp) },
                f.ds.name.clone(),
                f.ds.classes.clone(),
                pool,
            ))
        }
        BenchSource::LabelJob { job_id } => {
            let job = jobs
                .iter()
                .find(|j| j.id == *job_id)
                .ok_or_else(|| ApiError::new("label_job_not_found", format!("No labeling job {job_id}")))?;
            let seed = hash(&job.id);
            let pool: Vec<BenchImgSource> = (0..job.counts.labeled)
                .map(|i| {
                    let s = seed ^ i.wrapping_mul(0x9E37);
                    let scene = Scene::generate(s, job.classes.len() as u32, 8, false);
                    BenchImgSource {
                        id: (i + 1).to_string(),
                        file: format!("frame_{:04}.jpg", i + 1),
                        width: 640,
                        height: 480,
                        boxes: scene.boxes(640, 480),
                        seed: s,
                        classes: job.classes.len() as u32,
                        max_objects: 8,
                        small: false,
                    }
                })
                .collect();
            Ok((
                BenchSource::LabelJob { job_id: job.id.clone() },
                job.name.clone(),
                job.classes.clone(),
                pool,
            ))
        }
    }
}

fn propose_plan(bench: &Bench, req: &DesignBench) -> BenchPlan {
    let desc = format!("{} {}", bench.description, req.note).to_lowercase();
    let murky =
        desc.contains("water") || desc.contains("pond") || desc.contains("haze") || desc.contains("fog");
    let night =
        desc.contains("night") || desc.contains("dusk") || desc.contains("dark") || desc.contains("light");
    let conds = vec![
        BenchPlanCondition {
            condition: "low_light".into(),
            severities: vec![2, 3, 4, 5],
            weight: if night { 2.5 } else { 1.5 },
            why: Some("Dim shifts and shadows raise shot noise and hide low-contrast edges.".into()),
        },
        BenchPlanCondition {
            condition: "haze".into(),
            severities: vec![1, 3, 5],
            weight: if murky { 3.0 } else { 1.5 },
            why: Some("Scattering lowers contrast across the frame.".into()),
        },
        BenchPlanCondition {
            condition: "motion_blur".into(),
            severities: vec![1, 3, 4],
            weight: 2.0,
            why: Some("Moving parts smear during longer exposures.".into()),
        },
        BenchPlanCondition {
            condition: "defocus".into(),
            severities: vec![1, 3],
            weight: 1.0,
            why: Some("Objects step slightly out of the focal plane.".into()),
        },
        BenchPlanCondition {
            condition: "occlusion".into(),
            severities: vec![1, 2, 3, 4],
            weight: 2.5,
            why: Some("Clutter and overlapping objects hide part of the target.".into()),
        },
        BenchPlanCondition {
            condition: "distance".into(),
            severities: vec![1, 3, 4],
            weight: 2.0,
            why: Some("Objects seen farther from the camera shrink in pixel area.".into()),
        },
        BenchPlanCondition {
            condition: "sensor_noise".into(),
            severities: vec![2, 4],
            weight: 1.0,
            why: Some("High-ISO sensor grain at low illumination.".into()),
        },
    ];
    BenchPlan {
        conditions: conds,
        rationale: Some(format!(
            "Tailored for {}: weighs low-light grain, partial occlusion, motion smear and scale loss over compression artifacts.",
            bench.name
        )),
        designed_by: Some(BenchDesigner {
            agent: req.agent,
            model: req.model.clone().or_else(|| {
                Some(if req.agent == LabelAgentId::Claude { "sonnet".into() } else { "o3".into() })
            }),
        }),
    }
}

fn synth_detections(
    im: &BenchImgSource,
    entry: &BenchEntry,
    skill: EntrySkill,
    floor: f32,
) -> Vec<BenchDetection> {
    let mut rng = Rng::new(hash(&entry.id) ^ im.seed);
    let mut out = Vec::new();
    for b in &im.boxes {
        let missing = entry.missing_classes.iter().any(|_| false); // checked by class index in caller when needed
        let _ = missing;
        let hit_prob = (skill.map * 1.18).clamp(0.15, 0.96);
        if rng.f32() <= hit_prob {
            let (bw, bh) = ((b.x2 - b.x1).max(2.0), (b.y2 - b.y1).max(2.0));
            let jitter = (1.0 - skill.iou).clamp(0.02, 0.35) * 0.45;
            let dx1 = rng.range(-jitter, jitter) * bw;
            let dy1 = rng.range(-jitter, jitter) * bh;
            let dx2 = rng.range(-jitter, jitter) * bw;
            let dy2 = rng.range(-jitter, jitter) * bh;
            let score = if entry.scored { rng.range(0.45, 0.97).clamp(floor, 0.99) } else { 1.0 };
            out.push(BenchDetection {
                class: b.class,
                score,
                x1: (b.x1 + dx1).max(0.0),
                y1: (b.y1 + dy1).max(0.0),
                x2: (b.x2 + dx2).min(im.width as f32),
                y2: (b.y2 + dy2).min(im.height as f32),
            });
        }
    }
    // Occasional false positive on tougher images.
    if rng.f32() < (1.0 - skill.map) * 0.42 {
        let cx = rng.range(0.15, 0.85) * im.width as f32;
        let cy = rng.range(0.15, 0.85) * im.height as f32;
        let r = rng.range(18.0, 52.0);
        let score = if entry.scored { rng.range(floor.max(0.28), 0.72) } else { 1.0 };
        out.push(BenchDetection {
            class: rng.below(im.classes.max(1)),
            score,
            x1: (cx - r).max(0.0),
            y1: (cy - r).max(0.0),
            x2: (cx + r).min(im.width as f32),
            y2: (cy + r).min(im.height as f32),
        });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

fn score_external_entry(
    b: &BenchFixture,
    eid: String,
    name: String,
    created: String,
    finished: String,
    scored: bool,
    predictions: &[BenchPrediction],
) -> ApiResult<(BenchEntry, BenchResult)> {
    let nc = b.bench.classes.len();
    let mut clean_map: HashMap<&str, &[BenchDetection]> = HashMap::new();
    for p in predictions.iter().filter(|p| p.condition.is_none()) {
        clean_map.insert(p.image.as_str(), &p.boxes);
    }
    if clean_map.len() < b.images.len() {
        return Err(ApiError::new(
            "invalid_body",
            format!(
                "Missing unchanged-image predictions: got {} of {} images.",
                clean_map.len(),
                b.images.len()
            ),
        ));
    }
    let samples: Vec<folo_bench::Sample> = b
        .images
        .iter()
        .map(|im| folo_bench::Sample {
            truths: im.boxes.clone(),
            detections: clean_map.get(im.id.as_str()).map(|s| s.to_vec()).unwrap_or_default(),
        })
        .collect();
    let eval = folo_bench::score::evaluate(&samples, nc, scored, b.bench.seed);

    let mut variant_runs = Vec::new();
    for pc in &b.bench.plan.conditions {
        let Some(cond) = Condition::parse(&pc.condition) else { continue };
        for &sev in &pc.severities {
            let recs: Vec<&BenchPrediction> = predictions
                .iter()
                .filter(|p| p.condition.as_deref() == Some(pc.condition.as_str()) && p.severity == Some(sev))
                .collect();
            if recs.len() == b.images.len() {
                let mut by_img: HashMap<&str, &[BenchDetection]> = HashMap::new();
                for r in recs {
                    by_img.insert(r.image.as_str(), &r.boxes);
                }
                let vsamples: Vec<folo_bench::Sample> = b
                    .images
                    .iter()
                    .map(|im| folo_bench::Sample {
                        truths: if cond == Condition::Distance {
                            distance_boxes(&im.boxes, im.width, im.height, sev)
                        } else {
                            im.boxes.clone()
                        },
                        detections: by_img.get(im.id.as_str()).map(|s| s.to_vec()).unwrap_or_default(),
                    })
                    .collect();
                variant_runs.push((cond, sev, folo_bench::score::map(&vsamples, nc, scored)));
            }
        }
    }
    let rob = folo_bench::score::robustness(eval.accuracy.map50_95, &b.bench.plan, &variant_runs);
    let conds_run = rob.as_ref().map_or(0, |r| r.conditions.len() as u32);
    let complete = conds_run as usize == b.bench.plan.conditions.len();
    let sc = folo_bench::score::score(&eval, rob.as_ref(), complete);
    let entry = BenchEntry {
        id: eid,
        bench_id: b.bench.id.clone(),
        name,
        contestant: BenchContestant::External,
        status: BenchEntryStatus::Done,
        created,
        finished: Some(finished),
        progress: BenchProgress {
            done: predictions.len() as u64,
            total: predictions.len() as u64,
            condition: None,
            severity: None,
        },
        images: b.images.len() as u64,
        conditions: conds_run > 0,
        scored,
        fingerprint: b.bench.fingerprint.clone(),
        missing_classes: Vec::new(),
        score: Some(sc),
        map50_95: Some(eval.accuracy.map50_95),
        ci95: Some(eval.accuracy.ci95),
        conditions_run: conds_run,
        speed: None,
        usage: None,
        error: None,
        hint: None,
    };
    let res = BenchResult {
        entry: entry.clone(),
        accuracy: eval.accuracy,
        robustness: rob,
        calibration: eval.calibration,
        localization: eval.localization,
        self_labeling: eval.self_labeling,
    };
    Ok((entry, res))
}

fn synthesize_result(
    bench: &Bench,
    images: &[BenchImgSource],
    entry: &BenchEntry,
    skill: EntrySkill,
) -> BenchResult {
    let mut rng = Rng::new(hash(&entry.id) ^ bench.seed);
    let n_imgs = entry.images.max(1) as f32;
    let half_w = (0.38 / n_imgs.sqrt()).clamp(0.008, 0.12);
    let map = skill.map.clamp(0.05, 0.96);
    let ci95 = [(map - half_w).max(0.0), (map + half_w * 1.04).min(1.0)];
    let map50 = (map * 1.24 + 0.04 * (1.0 - map)).min(0.985);
    let map75 = (map * 1.05).min(map50);
    let precision = (0.52 + 0.44 * map).min(0.98);
    let recall = (0.44 + 0.50 * map).min(0.96);
    let f1 = 2.0 * precision * recall / (precision + recall).max(1e-6);
    let best_conf = if entry.scored { ((0.42 - 0.18 * map) * 100.0).round() / 100.0 } else { 1.0 };

    let missing_lc: Vec<String> = entry.missing_classes.iter().map(|c| c.to_lowercase()).collect();
    let per_class: Vec<BenchClassAccuracy> = bench
        .classes
        .iter()
        .enumerate()
        .filter_map(|(i, cname)| {
            let truths = bench.class_boxes.get(i).copied().unwrap_or(0);
            if truths == 0 {
                return None;
            }
            let unk = missing_lc.contains(&cname.to_lowercase());
            let jitter = 1.0 + 0.08 * rng.normal() - 0.02 * i as f32;
            let cm = if unk { 0.0 } else { (map * jitter).clamp(0.04, 0.98) };
            Some(BenchClassAccuracy {
                class: i as u32,
                truths,
                map50: if unk { 0.0 } else { (cm * 1.22 + 0.04).min(0.99) },
                map50_95: cm,
            })
        })
        .collect();

    let accuracy = BenchAccuracy {
        map50_95: map,
        ci95,
        map50,
        map75,
        precision,
        recall,
        f1,
        best_conf,
        map_small: Some((map * 0.68).clamp(0.02, 0.92)),
        map_medium: Some((map * 1.02).clamp(0.05, 0.96)),
        map_large: Some((map * 1.14).clamp(0.08, 0.98)),
        per_class,
    };

    let robustness = if entry.conditions && !bench.plan.conditions.is_empty() {
        let mut w_sum = 0.0f64;
        let mut r_sum = 0.0f64;
        let mut conds = Vec::new();
        for pc in &bench.plan.conditions {
            let base_drop = match pc.condition.as_str() {
                "occlusion" => 0.115,
                "low_light" => 0.095,
                "distance" => 0.105,
                "motion_blur" => 0.085,
                "haze" => 0.080,
                "low_resolution" => 0.075,
                "defocus" => 0.070,
                "sensor_noise" => 0.065,
                "color_cast" => 0.045,
                "jpeg" => 0.050,
                "low_contrast" => 0.055,
                _ => 0.035,
            } * skill.fragility;
            let points: Vec<BenchSeverityPoint> = pc
                .severities
                .iter()
                .map(|&s| {
                    let drop = (base_drop * (s as f32).powf(1.22)).clamp(0.0, 0.88);
                    let ret = (1.0 - drop).clamp(0.08, 1.0);
                    let m = map * ret;
                    BenchSeverityPoint {
                        severity: s,
                        map50_95: m,
                        map50: (m * 1.24).min(0.98),
                        retention: ret,
                    }
                })
                .collect();
            let mean_ret =
                points.iter().map(|p| p.retention as f64).sum::<f64>() / points.len().max(1) as f64;
            w_sum += pc.weight as f64;
            r_sum += pc.weight as f64 * mean_ret;
            conds.push(BenchConditionResult {
                condition: pc.condition.clone(),
                weight: pc.weight,
                retention: mean_ret as f32,
                points,
            });
        }
        Some(BenchRobustness {
            retention: if w_sum > 0.0 { (r_sum / w_sum) as f32 } else { 0.0 },
            conditions: conds,
        })
    } else {
        None
    };

    let total_truths: u64 =
        images.iter().take(entry.images as usize).map(|i| i.boxes.len() as u64).sum::<u64>().max(40);
    let calibration = entry.scored.then(|| {
        let dets_total = (total_truths as f64 * 1.15) as u64;
        let shares = [0.0, 0.06, 0.08, 0.09, 0.11, 0.12, 0.14, 0.15, 0.14, 0.11];
        let mut bins = Vec::with_capacity(10);
        let mut ece = 0.0f64;
        let mut counted = 0u64;
        for (b, &sh) in shares.iter().enumerate() {
            let lo = b as f32 * 0.1;
            let hi = (b + 1) as f32 * 0.1;
            let c = if b == 0 { 0 } else { ((dets_total as f64 * sh).round() as u64).max(2) };
            counted += c;
            if c == 0 {
                bins.push(BenchCalibrationBin { lo, hi, count: 0, confidence: None, precision: None });
            } else {
                let mid = (lo + hi) * 0.5;
                let bias = -skill.ece * (1.2 - 0.5 * mid);
                let prec = (mid + bias).clamp(0.04, 0.99);
                ece += c as f64 * (mid - prec).abs() as f64;
                bins.push(BenchCalibrationBin {
                    lo,
                    hi,
                    count: c,
                    confidence: Some(mid),
                    precision: Some(prec),
                });
            }
        }
        BenchCalibration { ece: (ece / counted.max(1) as f64) as f32, detections: counted, bins }
    });

    let matches = ((total_truths as f32 * recall).round() as u64).max(12);
    let mean_iou = skill.iou.clamp(0.55, 0.94);
    let peak = ((mean_iou - 0.5) / 0.05).clamp(0.5, 9.0);
    let weights: Vec<f32> = (0..10).map(|k| (-0.45 * (k as f32 - peak).powi(2)).exp()).collect();
    let w_tot: f32 = weights.iter().sum::<f32>().max(1e-6);
    let histogram: Vec<u64> = weights.iter().map(|w| (matches as f32 * w / w_tot).round() as u64).collect();
    let localization = BenchLocalization { mean_iou: Some(mean_iou), matches, histogram };

    let self_labeling = if entry.scored {
        let reaches = skill.self_recall >= 0.05;
        let t_star = reaches.then(|| ((0.76 - 0.34 * skill.self_recall) * 100.0).round() / 100.0);
        let curve: Vec<BenchLabelPoint> = (0..=100)
            .map(|k| {
                let c = k as f32 / 100.0;
                let p = if reaches {
                    let t = t_star.unwrap_or(0.65);
                    if c >= t {
                        (0.90 + 0.09 * ((c - t) / (1.0 - t).max(0.05))).min(0.995)
                    } else {
                        (0.42 + 0.48 * (c / t.max(0.05)).powf(0.85)).min(0.899)
                    }
                } else {
                    (0.28 + 0.58 * c).min(0.88)
                };
                let r = (skill.self_recall
                    * ((1.0 - c) / (1.0 - t_star.unwrap_or(0.65)).max(0.1)).powf(0.75))
                .clamp(0.0, 0.96);
                let labels = ((total_truths as f32 * r / p.max(0.2)).round() as u64).max(u64::from(c < 0.92));
                BenchLabelPoint {
                    confidence: c,
                    precision: (labels > 0).then_some(p),
                    recall: if labels > 0 { r } else { 0.0 },
                    labels,
                }
            })
            .collect();
        let at_idx = t_star.map_or(0, |t| (t * 100.0).round() as usize);
        let pt = curve[at_idx.min(100)];
        let axis = if reaches { skill.self_recall } else { 0.0 };
        let readiness = if axis >= 0.5 {
            BenchReadiness::Ready
        } else if axis >= 0.25 {
            BenchReadiness::Partial
        } else {
            BenchReadiness::NotReady
        };
        BenchSelfLabeling {
            target_precision: 0.9,
            threshold: t_star,
            precision: pt.precision.unwrap_or(0.0),
            recall: axis,
            mean_iou: Some(mean_iou),
            labels_per_image: pt.labels as f32 / n_imgs,
            labels: pt.labels,
            readiness,
            curve,
        }
    } else {
        let labels = ((total_truths as f32 * 0.72).round() as u64).max(14);
        let prec = (0.58 + 0.36 * skill.map).clamp(0.4, 0.93);
        let rec = (labels as f32 * prec / total_truths as f32).clamp(0.1, 0.85);
        let axis = if prec >= 0.9 && labels >= 10 { rec } else { 0.0 };
        let readiness = if axis >= 0.5 {
            BenchReadiness::Ready
        } else if axis >= 0.25 {
            BenchReadiness::Partial
        } else {
            BenchReadiness::NotReady
        };
        BenchSelfLabeling {
            target_precision: 0.9,
            threshold: None,
            precision: prec,
            recall: axis,
            mean_iou: Some(mean_iou),
            labels_per_image: labels as f32 / n_imgs,
            labels,
            readiness,
            curve: vec![BenchLabelPoint { confidence: 0.0, precision: Some(prec), recall: rec, labels }],
        }
    };

    let complete = entry.images == bench.images
        && (bench.plan.conditions.is_empty()
            || robustness.as_ref().is_some_and(|r| r.conditions.len() == bench.plan.conditions.len()));
    let eval = folo_bench::Evaluation {
        accuracy: accuracy.clone(),
        calibration: calibration.clone(),
        localization: localization.clone(),
        self_labeling: self_labeling.clone(),
    };
    let sc = folo_bench::score::score(&eval, robustness.as_ref(), complete);
    let mut full_entry = entry.clone();
    full_entry.score = Some(sc);
    full_entry.map50_95 = Some(accuracy.map50_95);
    full_entry.ci95 = Some(accuracy.ci95);
    full_entry.conditions_run = robustness.as_ref().map_or(0, |r| r.conditions.len() as u32);
    BenchResult { entry: full_entry, accuracy, robustness, calibration, localization, self_labeling }
}

#[allow(clippy::too_many_arguments)]
fn make_done_entry(
    bench: &Bench,
    images: &[BenchImgSource],
    id: &str,
    name: &str,
    contestant: BenchContestant,
    created_s: i64,
    dur_s: i64,
    skill: EntrySkill,
    imgs: u64,
    conditions: bool,
    scored: bool,
    missing_classes: Vec<String>,
    speed: Option<BenchSpeed>,
    usage: Option<LabelUsage>,
    fp_override: Option<String>,
) -> EntryFixture {
    let passes = if conditions { plan_passes(&bench.plan) as u64 } else { 0 };
    let total = imgs * (1 + passes);
    let mut entry = BenchEntry {
        id: id.into(),
        bench_id: bench.id.clone(),
        name: name.into(),
        contestant,
        status: BenchEntryStatus::Done,
        created: time::format(created_s),
        finished: Some(time::format(created_s + dur_s)),
        progress: BenchProgress { done: total, total, condition: None, severity: None },
        images: imgs,
        conditions,
        scored,
        fingerprint: fp_override.unwrap_or_else(|| bench.fingerprint.clone()),
        missing_classes,
        score: None,
        map50_95: None,
        ci95: None,
        conditions_run: 0,
        speed,
        usage,
        error: None,
        hint: None,
    };
    let res = synthesize_result(bench, images, &entry, skill);
    entry.score = res.entry.score.clone();
    entry.map50_95 = res.entry.map50_95;
    entry.ci95 = res.entry.ci95;
    entry.conditions_run = res.entry.conditions_run;
    EntryFixture { entry, skill, open_records: None, cached_result: Some(res), live: None }
}

fn seed_data(datasets: &[DatasetFixture], models: &[Model]) -> MockBenchData {
    let h = |hours: f64| MOCK_NOW - (hours * 3600.0) as i64;
    let agents = vec![
        LabelAgent {
            id: LabelAgentId::Claude,
            name: "Claude Code".into(),
            available: true,
            version: Some("2.1.30".into()),
            models: vec![
                LabelModel { id: "sonnet".into(), label: "Sonnet".into() },
                LabelModel { id: "haiku".into(), label: "Haiku".into() },
                LabelModel { id: "opus".into(), label: "Opus".into() },
            ],
            default_model: Some("sonnet".into()),
            default_effort: LabelEffort::Medium,
            hint: None,
        },
        LabelAgent {
            id: LabelAgentId::Codex,
            name: "Codex".into(),
            available: true,
            version: Some("0.42.0".into()),
            models: vec![
                LabelModel { id: "o3".into(), label: "o3".into() },
                LabelModel { id: "codex-mini".into(), label: "codex-mini".into() },
            ],
            default_model: Some("o3".into()),
            default_effort: LabelEffort::Medium,
            hint: None,
        },
    ];
    let jobs = vec![LabelJob {
        id: "lb_4f10".into(),
        name: "line-rejects".into(),
        classes: vec!["circle".into(), "square".into(), "triangle".into()],
        instructions: "Shapes on the reject chute under uneven LEDs.".into(),
        created: time::format(h(48.0)),
        status: LabelJobStatus::Idle,
        counts: LabelCounts { images: 140, pending: 20, labeled: 120, failed: 0, boxes: 492 },
        run: None,
        usage: LabelUsage {
            calls: 30,
            input_tokens: 48_000,
            cached_input_tokens: 0,
            output_tokens: 7_200,
            cost_usd: Some(0.084),
        },
        dataset_id: None,
    }];

    let model_spec = |mid: &str, fallback_arch: &str, fallback_params: u64| -> BenchContestant {
        match models.iter().find(|m| m.id == mid) {
            Some(m) => BenchContestant::Model {
                model_id: m.id.clone(),
                arch: m.arch.as_str().to_string(),
                params: m.params,
                imgsz: m.imgsz,
            },
            None => BenchContestant::Model {
                model_id: mid.into(),
                arch: fallback_arch.into(),
                params: fallback_params,
                imgsz: 640,
            },
        }
    };

    // 1. Shapes-hard val (`bn_3c90`): agent-designed plan, entries of every kind & state, 1 active loop.
    let (_, src_name_1, classes_1, pool_1) = resolve_source(
        &BenchSource::Dataset { dataset_id: "ds_3b91".into(), split: Some(Split::Val) },
        datasets,
        &jobs,
        Split::Val,
    )
    .unwrap();
    let imgs_1 = sample_source(&pool_1, 0, 500);
    let boxes_1 = count_class_boxes(&imgs_1, classes_1.len());
    let mut b1 = Bench {
        id: "bn_3c90".into(),
        name: "Shapes-hard val".into(),
        description: "Overhead inspection camera over mixed shapes: clutter, overlapping parts, motion smear on fast belts and dim corners at night shift.".into(),
        classes: classes_1.clone(),
        created: time::format(h(26.0)),
        source: BenchSource::Dataset { dataset_id: "ds_3b91".into(), split: Some(Split::Val) },
        source_name: src_name_1,
        images: imgs_1.len() as u64,
        boxes: boxes_1.iter().sum(),
        class_boxes: boxes_1,
        plan: folo_bench::default_plan(),
        seed: 0,
        fingerprint: String::new(),
        entries: 0,
        best: None,
        active: true,
    };
    b1.plan = propose_plan(
        &b1,
        &DesignBench {
            agent: LabelAgentId::Claude,
            model: Some("sonnet".into()),
            effort: Some(LabelEffort::Medium),
            note: "night shift".into(),
        },
    );
    b1.fingerprint = fingerprint_of(b1.seed, &b1.classes, imgs_1.len(), &b1.plan);

    let e1_lead = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c01",
        "shapes-hard-s-best",
        model_spec("m_44c0", "folo-s", 9_420_000),
        h(22.0),
        54,
        EntrySkill { map: 0.624, fragility: 0.86, ece: 0.038, iou: 0.81, self_recall: 0.58 },
        imgs_1.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let e1_onnx = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c02",
        "shapes-hard-s.onnx",
        BenchContestant::External,
        h(20.0),
        40,
        EntrySkill { map: 0.622, fragility: 0.87, ece: 0.040, iou: 0.81, self_recall: 0.57 },
        imgs_1.len() as u64,
        true,
        true,
        Vec::new(),
        None,
        None,
        None,
    );
    let e1_crowd = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c03",
        "shapes-crowded-s-best",
        model_spec("m_7c11", "folo-s", 9_420_000),
        h(18.0),
        52,
        EntrySkill { map: 0.553, fragility: 0.94, ece: 0.052, iou: 0.78, self_recall: 0.46 },
        imgs_1.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.2 }),
        None,
        None,
    );
    let e1_partial_cls = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c04",
        "shapes-n-best",
        model_spec("m_31aa", "folo-n", 2_630_000),
        h(15.0),
        28,
        EntrySkill { map: 0.432, fragility: 0.98, ece: 0.061, iou: 0.76, self_recall: 0.31 },
        imgs_1.len() as u64,
        true,
        true,
        vec!["star".into(), "hexagon".into()],
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.3 }),
        None,
        None,
    );
    let e1_agent = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c05",
        "Claude Code (sonnet)",
        BenchContestant::Agent {
            agent: LabelAgentId::Claude,
            model: Some("sonnet".into()),
            effort: LabelEffort::Medium,
        },
        h(12.0),
        95,
        EntrySkill { map: 0.492, fragility: 1.0, ece: 0.0, iou: 0.75, self_recall: 0.34 },
        48,
        false,
        false,
        Vec::new(),
        None,
        Some(LabelUsage {
            calls: 12,
            input_tokens: 19_400,
            cached_input_tokens: 0,
            output_tokens: 3_120,
            cost_usd: Some(0.031),
        }),
        None,
    );
    let e1_failed = EntryFixture {
        entry: BenchEntry {
            id: "be_3c06".into(),
            bench_id: b1.id.clone(),
            name: "Codex (o3)".into(),
            contestant: BenchContestant::Agent {
                agent: LabelAgentId::Codex,
                model: Some("o3".into()),
                effort: LabelEffort::Medium,
            },
            status: BenchEntryStatus::Failed,
            created: time::format(h(6.0)),
            finished: Some(time::format(h(5.98))),
            progress: BenchProgress { done: 0, total: 48, condition: None, severity: None },
            images: 48,
            conditions: false,
            scored: false,
            fingerprint: b1.fingerprint.clone(),
            missing_classes: Vec::new(),
            score: None,
            map50_95: None,
            ci95: None,
            conditions_run: 0,
            speed: None,
            usage: None,
            error: Some("codex exited with status 1: not signed in.".into()),
            hint: Some("Run `codex login` on the server's machine, then add the entry again.".into()),
        },
        skill: EntrySkill { map: 0.4, fragility: 1.0, ece: 0.0, iou: 0.7, self_recall: 0.0 },
        open_records: None,
        cached_result: None,
        live: None,
    };
    let run_total = imgs_1.len() as u64 * (1 + plan_passes(&b1.plan) as u64);
    let run_done = (run_total as f64 * 0.56) as u64;
    let (r_cond, r_sev) = pass_at(&b1.plan, imgs_1.len() as u64, true, run_done);
    let e1_running = EntryFixture {
        entry: BenchEntry {
            id: "be_3c07".into(),
            bench_id: b1.id.clone(),
            name: "shapes-hard-s-f16".into(),
            contestant: BenchContestant::Model {
                model_id: "m_44c0".into(),
                arch: "folo-s".into(),
                params: 9_420_000,
                imgsz: 640,
            },
            status: BenchEntryStatus::Running,
            created: time::format(h(0.02)),
            finished: None,
            progress: BenchProgress { done: run_done, total: run_total, condition: r_cond, severity: r_sev },
            images: imgs_1.len() as u64,
            conditions: true,
            scored: true,
            fingerprint: b1.fingerprint.clone(),
            missing_classes: Vec::new(),
            score: None,
            map50_95: None,
            ci95: None,
            conditions_run: 0,
            speed: Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.9 }),
            usage: None,
            error: None,
            hint: None,
        },
        skill: EntrySkill { map: 0.623, fragility: 0.86, ece: 0.039, iou: 0.81, self_recall: 0.58 },
        open_records: None,
        cached_result: None,
        live: Some((run_done, 0.0)),
    };
    let loop_1 = BenchLoop {
        id: "bl_3c10".into(),
        bench_id: b1.id.clone(),
        name: "shapes-hard-s-best loop".into(),
        status: BenchLoopStatus::Running,
        phase: Some(BenchLoopPhase::Training),
        created: time::format(h(0.6)),
        finished: None,
        config: StartBenchLoop {
            name: Some("shapes-hard-s-best loop".into()),
            teacher: "m_44c0".into(),
            pool: BenchSource::Dataset { dataset_id: "ds_3b91".into(), split: Some(Split::Train) },
            generations: Some(2),
            min_confidence: None,
            student: BenchStudent {
                arch: Some("folo-m".into()),
                epochs: Some(40),
                imgsz: Some(640),
                batch: Some(16),
                device: Some("auto".into()),
                weights: None,
                seed: 0,
            },
        },
        generations: vec![
            BenchGeneration {
                generation: 0,
                labels: None,
                run_id: Some("run_52e8".into()),
                model_id: Some("m_44c0".into()),
                model_name: Some("shapes-hard-s-best".into()),
                entry_id: Some("be_3c01".into()),
                score: e1_lead.entry.score.as_ref().map(|s| s.total),
                map50_95: e1_lead.entry.map50_95,
                ci95: e1_lead.entry.ci95,
            },
            BenchGeneration {
                generation: 1,
                labels: Some(BenchLabels {
                    threshold: 0.56,
                    images: 3200,
                    labeled_images: 2940,
                    boxes: 14_820,
                    excluded: 0,
                    dataset_id: "ds_3b91".into(),
                }),
                run_id: Some("run_a19f".into()),
                model_id: None,
                model_name: None,
                entry_id: None,
                score: None,
                map50_95: None,
                ci95: None,
            },
        ],
        verdict: None,
        error: None,
        hint: None,
    };

    // 2. Shapes val (`bn_5e12`): 3 loops (Improved, Held, Failed) + finished entries.
    let (_, src_name_2, classes_2, pool_2) = resolve_source(
        &BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Val) },
        datasets,
        &jobs,
        Split::Val,
    )
    .unwrap();
    let imgs_2 = sample_source(&pool_2, 0, 200);
    let boxes_2 = count_class_boxes(&imgs_2, classes_2.len());
    let plan_2 = folo_bench::default_plan();
    let fp_2 = fingerprint_of(0, &classes_2, imgs_2.len(), &plan_2);
    let b2 = Bench {
        id: "bn_5e12".into(),
        name: "Shapes val".into(),
        description: "Coloured shapes on textured backgrounds: held-out validation split grading clean accuracy, robustness and multi-generation self-labeling.".into(),
        classes: classes_2,
        created: time::format(h(160.0)),
        source: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Val) },
        source_name: src_name_2,
        images: imgs_2.len() as u64,
        boxes: boxes_2.iter().sum(),
        class_boxes: boxes_2,
        plan: plan_2,
        seed: 0,
        fingerprint: fp_2,
        entries: 0,
        best: None,
        active: false,
    };
    let n0 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a10",
        "shapes-n-best",
        model_spec("m_31aa", "folo-n", 2_630_000),
        h(140.0),
        24,
        EntrySkill { map: 0.781, fragility: 0.88, ece: 0.035, iou: 0.83, self_recall: 0.69 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.3 }),
        None,
        None,
    );
    let n1 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a11",
        "shapes-n-best loop gen 1-best",
        BenchContestant::Model {
            model_id: "m_5a11".into(),
            arch: "folo-s".into(),
            params: 9_420_000,
            imgsz: 640,
        },
        h(138.0),
        32,
        EntrySkill { map: 0.808, fragility: 0.86, ece: 0.033, iou: 0.84, self_recall: 0.72 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let n2 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a12",
        "shapes-n-best loop gen 2-best",
        BenchContestant::Model {
            model_id: "m_5a12".into(),
            arch: "folo-s".into(),
            params: 9_420_000,
            imgsz: 640,
        },
        h(136.0),
        32,
        EntrySkill { map: 0.829, fragility: 0.85, ece: 0.031, iou: 0.85, self_recall: 0.75 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let n3 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a13",
        "shapes-n-best loop gen 3-best",
        BenchContestant::Model {
            model_id: "m_5a13".into(),
            arch: "folo-s".into(),
            params: 9_420_000,
            imgsz: 640,
        },
        h(134.0),
        32,
        EntrySkill { map: 0.846, fragility: 0.84, ece: 0.030, iou: 0.86, self_recall: 0.77 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let s0 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a20",
        "shapes-n-f16",
        model_spec("m_31ab", "folo-n", 2_630_000),
        h(100.0),
        19,
        EntrySkill { map: 0.780, fragility: 0.88, ece: 0.036, iou: 0.83, self_recall: 0.68 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 1.7 }),
        None,
        None,
    );
    let s1 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a21",
        "shapes-n-f16 loop gen 1-best",
        BenchContestant::Model {
            model_id: "m_5a21".into(),
            arch: "folo-n".into(),
            params: 2_630_000,
            imgsz: 640,
        },
        h(98.0),
        24,
        EntrySkill { map: 0.786, fragility: 0.88, ece: 0.035, iou: 0.83, self_recall: 0.69 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.3 }),
        None,
        None,
    );
    let weak = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a30",
        "shapes-mini-n-last",
        model_spec("m_90d2", "folo-n", 2_630_000),
        h(60.0),
        18,
        EntrySkill { map: 0.242, fragility: 1.18, ece: 0.14, iou: 0.70, self_recall: 0.0 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 1.4 }),
        None,
        None,
    );

    let gen_from = |gen_idx: u32,
                    ef: &EntryFixture,
                    labels: Option<BenchLabels>,
                    run_id: Option<&str>,
                    model_id: &str| {
        BenchGeneration {
            generation: gen_idx,
            labels,
            run_id: run_id.map(Into::into),
            model_id: Some(model_id.into()),
            model_name: Some(ef.entry.name.clone()),
            entry_id: Some(ef.entry.id.clone()),
            score: ef.entry.score.as_ref().map(|s| s.total),
            map50_95: ef.entry.map50_95,
            ci95: ef.entry.ci95,
        }
    };
    let mk_labels = |th: f32, labeled: u64, boxes: u64| BenchLabels {
        threshold: th,
        images: 800,
        labeled_images: labeled,
        boxes,
        excluded: 0,
        dataset_id: "ds_8f2c".into(),
    };
    let loops_2 = vec![
        BenchLoop {
            id: "bl_5l01".into(),
            bench_id: b2.id.clone(),
            name: "shapes-n-best loop".into(),
            status: BenchLoopStatus::Done,
            phase: None,
            created: time::format(h(139.5)),
            finished: Some(time::format(h(133.8))),
            config: StartBenchLoop {
                name: Some("shapes-n-best loop".into()),
                teacher: "m_31aa".into(),
                pool: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Train) },
                generations: Some(3),
                min_confidence: None,
                student: BenchStudent {
                    arch: Some("folo-s".into()),
                    epochs: Some(30),
                    imgsz: Some(640),
                    batch: Some(16),
                    device: Some("auto".into()),
                    weights: None,
                    seed: 0,
                },
            },
            generations: vec![
                gen_from(0, &n0, None, Some("run_7d1e"), "m_31aa"),
                gen_from(1, &n1, Some(mk_labels(0.53, 756, 2_890)), None, "m_5a11"),
                gen_from(2, &n2, Some(mk_labels(0.51, 772, 3_010)), None, "m_5a12"),
                gen_from(3, &n3, Some(mk_labels(0.50, 781, 3_094)), None, "m_5a13"),
            ],
            verdict: Some(BenchLoopVerdict::Improved),
            error: None,
            hint: None,
        },
        BenchLoop {
            id: "bl_5l02".into(),
            bench_id: b2.id.clone(),
            name: "shapes-n-f16 loop".into(),
            status: BenchLoopStatus::Done,
            phase: None,
            created: time::format(h(99.5)),
            finished: Some(time::format(h(97.8))),
            config: StartBenchLoop {
                name: Some("shapes-n-f16 loop".into()),
                teacher: "m_31ab".into(),
                pool: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Train) },
                generations: Some(1),
                min_confidence: None,
                student: BenchStudent {
                    arch: Some("folo-n".into()),
                    epochs: Some(30),
                    imgsz: Some(640),
                    batch: Some(16),
                    device: Some("auto".into()),
                    weights: None,
                    seed: 0,
                },
            },
            generations: vec![
                gen_from(0, &s0, None, Some("run_7d1e"), "m_31ab"),
                gen_from(1, &s1, Some(mk_labels(0.53, 752, 2_860)), None, "m_5a21"),
            ],
            verdict: Some(BenchLoopVerdict::Held),
            error: None,
            hint: None,
        },
        BenchLoop {
            id: "bl_5l03".into(),
            bench_id: b2.id.clone(),
            name: "shapes-mini-n-last loop".into(),
            status: BenchLoopStatus::Failed,
            phase: None,
            created: time::format(h(58.0)),
            finished: Some(time::format(h(57.95))),
            config: StartBenchLoop {
                name: Some("shapes-mini-n-last loop".into()),
                teacher: "m_90d2".into(),
                pool: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Train) },
                generations: Some(1),
                min_confidence: None,
                student: BenchStudent {
                    arch: Some("folo-n".into()),
                    epochs: Some(30),
                    imgsz: Some(320),
                    batch: Some(16),
                    device: Some("auto".into()),
                    weights: None,
                    seed: 0,
                },
            },
            generations: vec![
                gen_from(0, &weak, None, Some("run_1a2b"), "m_90d2"),
                BenchGeneration {
                    generation: 1,
                    labels: None,
                    run_id: None,
                    model_id: None,
                    model_name: None,
                    entry_id: None,
                    score: None,
                    map50_95: None,
                    ci95: None,
                },
            ],
            verdict: None,
            error: Some(
                "shapes-mini-n-last never reaches 90% precision on at least 10 labels on this benchmark, so its entry found no threshold."
                    .into(),
            ),
            hint: Some(
                "Start a loop with a stronger teacher, or set a minimum confidence explicitly.".into(),
            ),
        },
    ];

    // 3. Night-shift tray (`bn_7a04`): fresh benchmark with 0 entries (editable plan!).
    let (_, src_name_3, classes_3, pool_3) = resolve_source(
        &BenchSource::Dataset { dataset_id: "ds_c07a".into(), split: Some(Split::Val) },
        datasets,
        &jobs,
        Split::Val,
    )
    .unwrap();
    let imgs_3 = sample_source(&pool_3, 0, 60);
    let boxes_3 = count_class_boxes(&imgs_3, classes_3.len());
    let plan_3 = folo_bench::default_plan();
    let fp_3 = fingerprint_of(0, &classes_3, imgs_3.len(), &plan_3);
    let b3 = Bench {
        id: "bn_7a04".into(),
        name: "Night-shift tray".into(),
        description: "Sorting tray camera under warm low-wattage LEDs: shadows at the rim and occasional motion blur when the tray indexes.".into(),
        classes: classes_3,
        created: time::format(h(1.5)),
        source: BenchSource::Dataset { dataset_id: "ds_c07a".into(), split: Some(Split::Val) },
        source_name: src_name_3,
        images: imgs_3.len() as u64,
        boxes: boxes_3.iter().sum(),
        class_boxes: boxes_3,
        plan: plan_3,
        seed: 0,
        fingerprint: fp_3,
        entries: 0,
        best: None,
        active: false,
    };

    MockBenchData {
        seq: 100,
        benches: vec![
            BenchFixture {
                bench: b3,
                images: imgs_3,
                entries: Vec::new(),
                loops: Vec::new(),
                loop_live: HashMap::new(),
            },
            BenchFixture {
                bench: b1,
                images: imgs_1,
                entries: vec![e1_lead, e1_onnx, e1_crowd, e1_partial_cls, e1_agent, e1_failed, e1_running],
                loops: vec![loop_1],
                loop_live: HashMap::new(),
            },
            BenchFixture {
                bench: b2,
                images: imgs_2,
                entries: vec![n3, n2, n1, n0, s1, s0, weak],
                loops: loops_2,
                loop_live: HashMap::new(),
            },
        ],
        agents,
        jobs,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::mock::MockBackend;
    use super::super::*;

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
    fn a_loop_starts_first_in_the_list() {
        let m = MockBackend::new(0, true);
        // The benchmark's best model that the server still has (a loop's students may be gone).
        let models = m.models().unwrap();
        let teacher = m.bench_entries("bn_5e12").unwrap().into_iter().find_map(|e| match e.contestant {
            BenchContestant::Model { model_id, .. }
                if e.status == BenchEntryStatus::Done && models.iter().any(|x| x.id == model_id) =>
            {
                Some(model_id)
            }
            _ => None,
        });
        let ds = m.datasets().unwrap().into_iter().find(|d| d.splits.train > 0).unwrap();
        let req = StartBenchLoop {
            name: None,
            teacher: teacher.unwrap(),
            pool: BenchSource::Dataset { dataset_id: ds.id, split: Some(Split::Train) },
            generations: Some(2),
            min_confidence: None,
            student: BenchStudent { epochs: Some(10), ..BenchStudent::default() },
        };
        let l = m.start_bench_loop("bn_5e12", &req).unwrap();
        assert!(l.status.is_active());
        assert_eq!(m.bench_loops("bn_5e12").unwrap()[0].id, l.id);
    }

    #[test]
    fn a_started_loop_runs_its_generations_and_ends() {
        let mut lp = MockBackend::new(0, true).bench_loops("bn_5e12").unwrap().remove(0);
        lp.status = BenchLoopStatus::Running;
        lp.config.generations = Some(2);
        lp.generations.truncate(1);
        lp.generations[0].map50_95 = None;
        lp.generations[0].score = None;
        // A teacher without an entry is benchmarked first.
        super::step_loop(&mut lp, 1.0, 0);
        assert_eq!(lp.phase, Some(BenchLoopPhase::Benchmarking));
        // Then each generation labels, trains and is benchmarked.
        super::step_loop(&mut lp, super::LOOP_TEACHER_S + 2.0, 0);
        assert_eq!((lp.phase, lp.generations.len()), (Some(BenchLoopPhase::Training), 2));
        assert!(lp.generations[0].map50_95.is_some() && lp.generations[1].map50_95.is_none());
        super::step_loop(&mut lp, super::LOOP_TEACHER_S + 2.0 * super::LOOP_GEN_S, 0);
        assert_eq!(lp.status, BenchLoopStatus::Done);
        assert_eq!(lp.generations.len(), 3);
        assert!(lp.generations.iter().all(|g| g.map50_95.is_some()));
        assert_eq!(lp.verdict, Some(BenchLoopVerdict::Improved));
    }

    #[test]
    fn a_loop_verdict_compares_the_students_map_with_the_teachers_interval() {
        let mut lp = MockBackend::new(0, true).bench_loops("bn_5e12").unwrap().remove(0);
        lp.status = BenchLoopStatus::Running;
        lp.config.generations = Some(1);
        lp.generations.truncate(1);
        // A teacher at 0.624 (0.607–0.642): the student's 0.654 is above the interval, though its own
        // interval's lower bound (0.634) is not.
        lp.generations[0].map50_95 = Some(0.624);
        lp.generations[0].ci95 = Some([0.607, 0.642]);
        lp.generations[0].score = Some(67.0);
        super::step_loop(&mut lp, super::LOOP_GEN_S, 0);
        assert_eq!(lp.verdict, Some(BenchLoopVerdict::Improved));
        // On the bound itself it holds.
        lp.status = BenchLoopStatus::Running;
        lp.generations.truncate(1);
        lp.generations[0].map50_95 = Some(0.40);
        lp.generations[0].ci95 = Some([0.37, 0.43]);
        super::step_loop(&mut lp, super::LOOP_GEN_S, 0);
        assert_eq!(lp.generations[1].map50_95, Some(0.43));
        assert_eq!(lp.verdict, Some(BenchLoopVerdict::Held));
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
