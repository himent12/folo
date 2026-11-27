//! Head-to-head comparison of two finished benchmark entries, shared by `folo bench compare` and MCP
//! `compare_bench_entries`.

use std::collections::{BTreeMap, BTreeSet};

use folo_api::{
    Bench, BenchAxis, BenchConditionResult, BenchContestant, BenchEntry, BenchEntryStatus, BenchResult,
    BenchScore,
};
use http::StatusCode;

use super::Benchmarks;
use crate::errors::Error;
use crate::types::bench::{
    BenchAccuracyDelta, BenchAxisDelta, BenchCompareResult, BenchCompareSide, BenchConditionDelta,
    BenchConditionMatch, BenchSeverityDelta, BenchSpeedDelta,
};

impl Benchmarks {
    /// Compares two finished, scored entries (`candidate - baseline`) of benchmark `bench_id`.
    ///
    /// Fails with `409 bench_entry_not_done` if either entry is not `done` or lacks a score.
    pub fn compare(
        &self,
        bench_id: &str,
        baseline_id: &str,
        candidate_id: &str,
    ) -> Result<BenchCompareResult, Error> {
        let bench = self.get(bench_id)?;
        let baseline_entry = self.entry(bench_id, baseline_id)?;
        ensure_done("baseline", &baseline_entry)?;
        let candidate_entry = if candidate_id == baseline_id {
            baseline_entry.clone()
        } else {
            let e = self.entry(bench_id, candidate_id)?;
            ensure_done("candidate", &e)?;
            e
        };
        let baseline_result = self.result(bench_id, baseline_id)?;
        let candidate_result = if candidate_id == baseline_id {
            baseline_result.clone()
        } else {
            self.result(bench_id, candidate_id)?
        };
        build_compare(&bench, &baseline_entry, &baseline_result, &candidate_entry, &candidate_result)
    }
}

fn status_word(s: BenchEntryStatus) -> &'static str {
    match s {
        BenchEntryStatus::Queued => "queued",
        BenchEntryStatus::Running => "running",
        BenchEntryStatus::Stopping => "stopping",
        BenchEntryStatus::Done => "done",
        BenchEntryStatus::Failed => "failed",
        BenchEntryStatus::Stopped => "stopped",
    }
}

fn ensure_done(role: &str, entry: &BenchEntry) -> Result<(), Error> {
    if entry.status != BenchEntryStatus::Done {
        let terminal = matches!(entry.status, BenchEntryStatus::Failed | BenchEntryStatus::Stopped);
        let mut msg = if terminal {
            format!(
                "{role} entry '{}' ({}) is {} and cannot be compared",
                entry.id,
                entry.name,
                status_word(entry.status)
            )
        } else {
            format!(
                "{role} entry '{}' ({}) is {} and cannot be compared yet",
                entry.id,
                entry.name,
                status_word(entry.status)
            )
        };
        if let Some(err) = &entry.error {
            msg.push_str(&format!(": {err}"));
        }
        let err = Error::new(StatusCode::CONFLICT, "bench_entry_not_done", msg);
        return Err(if terminal { err.hint(super::hints::COMPARE_TERMINAL_HINT) } else { err });
    }
    if entry.score.is_none() {
        return Err(Error::new(
            StatusCode::CONFLICT,
            "bench_entry_not_done",
            format!("{role} entry '{}' ({}) has no score to compare", entry.id, entry.name),
        )
        .hint(super::hints::COMPARE_TERMINAL_HINT));
    }
    Ok(())
}

fn kind_word(c: &BenchContestant) -> &'static str {
    match c {
        BenchContestant::Model { .. } => "model",
        BenchContestant::Agent { .. } => "agent",
        BenchContestant::External => "external",
    }
}

fn score_of<'a>(role: &str, entry: &'a BenchEntry, res: &'a BenchResult) -> Result<&'a BenchScore, Error> {
    entry.score.as_ref().or(res.entry.score.as_ref()).ok_or_else(|| {
        Error::new(
            StatusCode::CONFLICT,
            "bench_entry_not_done",
            format!("{role} entry '{}' ({}) has no score to compare", entry.id, entry.name),
        )
    })
}

fn side_of(bench: &Bench, entry: &BenchEntry, score: &BenchScore, res: &BenchResult) -> BenchCompareSide {
    BenchCompareSide {
        entry_id: entry.id.clone(),
        name: entry.name.clone(),
        kind: kind_word(&entry.contestant).into(),
        score: score.total,
        partial: score.partial,
        images: entry.images,
        conditions: entry.conditions,
        stale: entry.fingerprint != bench.fingerprint,
        map50_95: res.accuracy.map50_95,
        ci95: res.accuracy.ci95,
        ms_per_image: entry.speed.as_ref().map(|s| s.ms_per_image),
    }
}

fn build_compare(
    bench: &Bench,
    baseline_entry: &BenchEntry,
    baseline_res: &BenchResult,
    candidate_entry: &BenchEntry,
    candidate_res: &BenchResult,
) -> Result<BenchCompareResult, Error> {
    let base_score = score_of("baseline", baseline_entry, baseline_res)?;
    let cand_score = score_of("candidate", candidate_entry, candidate_res)?;
    let baseline = side_of(bench, baseline_entry, base_score, baseline_res);
    let candidate = side_of(bench, candidate_entry, cand_score, candidate_res);

    let ba = &baseline_res.accuracy;
    let ca = &candidate_res.accuracy;
    let ci_overlap = ba.ci95[0] <= ca.ci95[1] && ca.ci95[0] <= ba.ci95[1];
    let accuracy = BenchAccuracyDelta {
        baseline_map50_95: ba.map50_95,
        candidate_map50_95: ca.map50_95,
        map50_95_delta: ca.map50_95 - ba.map50_95,
        ci_overlap,
        map50_delta: ca.map50 - ba.map50,
        map75_delta: ca.map75 - ba.map75,
        precision_delta: ca.precision - ba.precision,
        recall_delta: ca.recall - ba.recall,
    };

    let axes: Vec<BenchAxisDelta> = BenchAxis::ALL
        .iter()
        .map(|&axis| {
            let b_axis = base_score.axes.iter().find(|a| a.axis == axis);
            let c_axis = cand_score.axes.iter().find(|a| a.axis == axis);
            let weight = c_axis.or(b_axis).map_or_else(|| axis.weight(), |a| a.weight);
            let b_val = b_axis.and_then(|a| a.value);
            let c_val = c_axis.and_then(|a| a.value);
            let delta = match (b_val, c_val) {
                (Some(b), Some(c)) => Some(c - b),
                _ => None,
            };
            BenchAxisDelta { axis, weight, baseline: b_val, candidate: c_val, delta }
        })
        .collect();

    let ece_delta = match (&baseline_res.calibration, &candidate_res.calibration) {
        (Some(b), Some(c)) => Some(c.ece - b.ece),
        _ => None,
    };
    let mean_iou_delta = match (baseline_res.localization.mean_iou, candidate_res.localization.mean_iou) {
        (Some(b), Some(c)) => Some(c - b),
        _ => None,
    };
    let self_labeling_recall_delta = candidate_res.self_labeling.recall - baseline_res.self_labeling.recall;

    let speed = match (&baseline_entry.speed, &candidate_entry.speed) {
        (None, None) => None,
        (b, c) => {
            let b_ms = b.as_ref().map(|s| s.ms_per_image);
            let c_ms = c.as_ref().map(|s| s.ms_per_image);
            let (delta_ms, speedup) = match (b_ms, c_ms) {
                (Some(bm), Some(cm)) => {
                    (Some(cm - bm), (bm > 0.0 && cm > 0.0).then_some(bm / cm).filter(|v| v.is_finite()))
                }
                _ => (None, None),
            };
            Some(BenchSpeedDelta {
                baseline_ms_per_image: b_ms,
                candidate_ms_per_image: c_ms,
                delta_ms_per_image: delta_ms,
                speedup,
                baseline_device: b.as_ref().map(|s| s.device.clone()),
                candidate_device: c.as_ref().map(|s| s.device.clone()),
            })
        }
    };

    let conditions = compare_conditions(
        bench,
        baseline_res.robustness.as_ref().map(|r| r.conditions.as_slice()).unwrap_or(&[]),
        candidate_res.robustness.as_ref().map(|r| r.conditions.as_slice()).unwrap_or(&[]),
    );

    let mut warnings = Vec::new();
    if baseline.entry_id == candidate.entry_id {
        warnings.push("baseline and candidate refer to the same entry".into());
    }
    if baseline_entry.fingerprint != candidate_entry.fingerprint {
        warnings
            .push("entries were graded against different benchmark versions (fingerprints differ)".into());
    }
    if baseline.stale {
        warnings
            .push(format!("baseline {} was graded on an older version of this benchmark", baseline.entry_id));
    }
    if candidate.stale && candidate.entry_id != baseline.entry_id {
        warnings.push(format!(
            "candidate {} was graded on an older version of this benchmark",
            candidate.entry_id
        ));
    }
    if baseline.partial || candidate.partial {
        warnings.push(
            "at least one entry has a partial score (missing axes or subset of images); compare totals cautiously"
                .into(),
        );
    }
    if baseline.images != candidate.images {
        warnings.push(format!(
            "image counts differ: baseline ran {} images, candidate ran {} images",
            baseline.images, candidate.images
        ));
    }
    let unmatched = conditions.iter().filter(|c| c.match_kind != BenchConditionMatch::Shared).count();
    if unmatched > 0 {
        warnings.push(format!(
            "{unmatched} stress condition(s) were run by only one entry and have no retention delta"
        ));
    }
    if let (Some(bs), Some(cs)) = (&baseline_entry.speed, &candidate_entry.speed)
        && bs.device != cs.device
    {
        warnings.push(format!(
            "speed was measured on different devices: baseline on {}, candidate on {}",
            bs.device, cs.device
        ));
    }

    Ok(BenchCompareResult {
        benchmark_id: bench.id.clone(),
        benchmark_name: bench.name.clone(),
        score_delta: candidate.score - baseline.score,
        baseline,
        candidate,
        accuracy,
        axes,
        ece_delta,
        mean_iou_delta,
        self_labeling_recall_delta,
        speed,
        conditions,
        warnings,
    })
}

fn compare_conditions(
    bench: &Bench,
    baseline: &[BenchConditionResult],
    candidate: &[BenchConditionResult],
) -> Vec<BenchConditionDelta> {
    let base_map: BTreeMap<&str, &BenchConditionResult> =
        baseline.iter().map(|c| (c.condition.as_str(), c)).collect();
    let cand_map: BTreeMap<&str, &BenchConditionResult> =
        candidate.iter().map(|c| (c.condition.as_str(), c)).collect();

    // Plan order first (for conditions present in either entry), then any extra conditions sorted by name.
    let mut order: Vec<String> = Vec::new();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for plan_c in &bench.plan.conditions {
        let id = plan_c.condition.as_str();
        if (base_map.contains_key(id) || cand_map.contains_key(id)) && seen.insert(id) {
            order.push(plan_c.condition.clone());
        }
    }
    for &id in base_map.keys().chain(cand_map.keys()) {
        if seen.insert(id) {
            order.push(id.to_string());
        }
    }

    order
        .into_iter()
        .map(|cond_id| {
            let b = base_map.get(cond_id.as_str()).copied();
            let c = cand_map.get(cond_id.as_str()).copied();
            let (match_kind, weight, baseline_retention, candidate_retention, retention_delta) = match (b, c)
            {
                (Some(b), Some(c)) => (
                    BenchConditionMatch::Shared,
                    c.weight,
                    Some(b.retention),
                    Some(c.retention),
                    Some(c.retention - b.retention),
                ),
                (Some(b), None) => {
                    (BenchConditionMatch::BaselineOnly, b.weight, Some(b.retention), None, None)
                }
                (None, Some(c)) => {
                    (BenchConditionMatch::CandidateOnly, c.weight, None, Some(c.retention), None)
                }
                (None, None) => unreachable!("order only holds keys present in base_map or cand_map"),
            };

            let mut sev_set: BTreeSet<u8> = BTreeSet::new();
            if let Some(b) = b {
                sev_set.extend(b.points.iter().map(|p| p.severity));
            }
            if let Some(c) = c {
                sev_set.extend(c.points.iter().map(|p| p.severity));
            }
            let severities = sev_set
                .into_iter()
                .map(|severity| {
                    let bp = b.and_then(|x| x.points.iter().find(|p| p.severity == severity));
                    let cp = c.and_then(|x| x.points.iter().find(|p| p.severity == severity));
                    BenchSeverityDelta {
                        severity,
                        baseline_map50_95: bp.map(|p| p.map50_95),
                        candidate_map50_95: cp.map(|p| p.map50_95),
                        map50_95_delta: match (bp, cp) {
                            (Some(bp), Some(cp)) => Some(cp.map50_95 - bp.map50_95),
                            _ => None,
                        },
                        baseline_retention: bp.map(|p| p.retention),
                        candidate_retention: cp.map(|p| p.retention),
                        retention_delta: match (bp, cp) {
                            (Some(bp), Some(cp)) => Some(cp.retention - bp.retention),
                            _ => None,
                        },
                    }
                })
                .collect();

            BenchConditionDelta {
                condition: cond_id,
                weight,
                match_kind,
                baseline_retention,
                candidate_retention,
                retention_delta,
                severities,
            }
        })
        .collect()
}
