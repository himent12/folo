//! Results and scores: synthetic detections from an entry's skill, scoring uploaded predictions
//! through folo-bench, and finished entries for the seed data.

use std::collections::HashMap;

use folo_bench::Condition;

use crate::api::time;
use crate::api::*;
use crate::synth::{Rng, hash};

use super::benches::plan_passes;
use super::images::distance_boxes;
use super::{BenchFixture, BenchImgSource, EntryFixture, EntrySkill};

pub(super) fn missing_classes(bench_classes: &[String], model_classes: &[String]) -> Vec<String> {
    bench_classes
        .iter()
        .filter(|bc| !model_classes.iter().any(|mc| mc.eq_ignore_ascii_case(bc)))
        .cloned()
        .collect()
}

pub(super) fn synth_detections(
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

/// Refuses a box the scoring would take as a wrong guess without a word, as the server does
/// (`crates/folo-server/src/bench/external.rs`): a coordinate that is not a finite number, or corners
/// the wrong way round (most often an (x, y, width, height) box sent as corners). A box of no area is a box.
pub(super) fn check_external_boxes(predictions: &[BenchPrediction]) -> ApiResult<()> {
    for (i, p) in predictions.iter().enumerate() {
        for (n, d) in p.boxes.iter().enumerate() {
            let what = format!("Record {} (image {}), box {}", i + 1, p.image, n + 1);
            if ![d.x1, d.y1, d.x2, d.y2].iter().all(|v| v.is_finite()) {
                return Err(ApiError::new(
                    "invalid_body",
                    format!("{what}: a coordinate is not a finite number"),
                ));
            }
            if d.x2 < d.x1 || d.y2 < d.y1 {
                return Err(ApiError::new(
                    "invalid_body",
                    format!("{what}: x2 {} or y2 {} is less than x1 {} or y1 {}", d.x2, d.y2, d.x1, d.y1),
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn score_external_entry(
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

pub(super) fn synthesize_result(
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
pub(super) fn make_done_entry(
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
