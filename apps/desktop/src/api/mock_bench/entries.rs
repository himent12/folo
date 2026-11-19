//! A benchmark's entries: listing, starting (a model, an agent, or uploaded predictions), results,
//! per-image pages, stopping and deleting.

use crate::api::mock::DatasetFixture;
use crate::api::time;
use crate::api::*;
use crate::synth::hash;

use super::benches::plan_passes;
use super::scoring::{
    check_external_boxes, missing_classes, score_external_entry, synth_detections, synthesize_result,
};
use super::{
    EntryFixture, EntrySkill, MockBench, find_bench, find_bench_mut, find_entry, find_entry_mut,
    sorted_entries,
};

impl MockBench {
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
                    check_external_boxes(predictions)?;
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
                conf: Some(best_conf),
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
}
