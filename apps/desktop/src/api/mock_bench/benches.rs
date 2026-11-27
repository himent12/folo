//! A benchmark itself: listing, creating, editing and deleting benchmarks, and designing a plan for one.

use std::collections::HashMap;

use crate::api::mock::DatasetFixture;
use crate::api::time;
use crate::api::*;
use crate::synth::hash;

use super::images::{count_class_boxes, resolve_source, sample_source};
use super::{BenchFixture, BenchImgSource, MockBench, find_bench, find_bench_mut, snapshot_bench};

impl MockBench {
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

    pub fn create_generated(
        &self,
        req: &CreateGeneratedBench,
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
        let spec = folo_bench::GeneratedSourceSpec {
            images: req.source.images,
            classes: req.source.classes.clone(),
            imgsz: req.source.imgsz,
            seed: req.source.seed,
            vary_aspect: req.source.vary_aspect,
        };
        let cfg = spec.to_synthetic_config().map_err(|m| ApiError::new("invalid_body", m))?;
        let plan = match &req.plan {
            Some(p) => folo_bench::check_plan(p).map_err(|m| ApiError::new("invalid_body", m))?,
            None => folo_bench::default_plan(),
        };
        let classes = cfg.class_names().to_vec();
        let n_classes = classes.len() as u32;
        let source_name = spec.display_name();
        const ASPECTS: [(u32, u32); 5] = [(1, 1), (4, 3), (3, 4), (16, 9), (3, 2)];
        let imgs: Vec<BenchImgSource> = (0..spec.images)
            .map(|i| {
                let (w, h) = if spec.vary_aspect {
                    let (aw, ah) = ASPECTS[(i as usize) % ASPECTS.len()];
                    if aw >= ah {
                        (spec.imgsz, ((spec.imgsz * ah) / aw).max(32))
                    } else {
                        (((spec.imgsz * aw) / ah).max(32), spec.imgsz)
                    }
                } else {
                    (spec.imgsz, spec.imgsz)
                };
                let s = spec.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64).wrapping_mul(0x517C_C1B7);
                let scene = crate::synth::Scene::generate(s, n_classes, 6, i % 20 == 0);
                BenchImgSource {
                    id: (i + 1).to_string(),
                    file: format!("{i:06}.jpg"),
                    width: w,
                    height: h,
                    boxes: scene.boxes(w, h),
                    seed: s,
                    classes: n_classes,
                    max_objects: 6,
                    small: i % 20 == 0,
                }
            })
            .collect();
        self.with(datasets, models, |d, _, now| {
            d.seq += 1;
            let id = format!("bn_{:012x}", hash(&format!("bn-gen-{}-{name}", d.seq)) & 0xFFFF_FFFF_FFFF);
            let class_boxes = count_class_boxes(&imgs, classes.len());
            let total_boxes = class_boxes.iter().sum();
            let fp = fingerprint_of(spec.seed, &classes, imgs.len(), &plan);
            let bench = Bench {
                id,
                name: name.to_string(),
                description: req.description.trim().to_string(),
                classes,
                created: time::format(now),
                source: BenchSource::Dataset { dataset_id: GENERATED_BENCH_SOURCE_ID.into(), split: None },
                source_name,
                images: imgs.len() as u64,
                boxes: total_boxes,
                class_boxes,
                plan,
                seed: spec.seed,
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
                    images: imgs,
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
}

pub(super) fn plan_passes(plan: &BenchPlan) -> usize {
    plan.conditions.iter().map(|c| c.severities.len()).sum()
}

pub(super) fn fingerprint_of(seed: u64, classes: &[String], images: usize, plan: &BenchPlan) -> String {
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

pub(super) fn propose_plan(bench: &Bench, req: &DesignBench) -> BenchPlan {
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
                Some(match req.agent {
                    LabelAgentId::Claude => "sonnet".into(),
                    LabelAgentId::Codex => "o3".into(),
                    LabelAgentId::Antigravity => "gemini-4-neon".into(),
                })
            }),
        }),
    }
}
