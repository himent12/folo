//! Pure state, curve extraction, and hyperparameter diff logic for Multi-Run Compare (`Route::RunCompare`).

use std::collections::HashSet;

use crate::api::{Head, Label, Optimizer, ResolvedOptimizer, Run, RunStatus};
use crate::store::RunData;

/// Maximum number of runs compared side by side.
pub const RUN_COMPARE_MAX: usize = 4;

/// Which validation mAP series to plot on the overlaid mAP chart.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CompareMapMetric {
    #[default]
    Map50_95,
    Map50,
}

/// X-axis alignment mode for overlaid comparison curves.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CompareXAxis {
    /// Epoch index (`0..=max_epochs`).
    #[default]
    Epochs,
    /// Percentage of each run's planned epochs (`0..=100%`), helpful when comparing runs of different lengths.
    Progress,
}

/// Persistent client state for the Multi-Run Compare workflow.
#[derive(Clone, Debug)]
pub struct RunCompareState {
    /// Pinned/selected run IDs in slot order (`ids[0]` is the baseline).
    pub ids: Vec<String>,
    /// True once default runs have been seeded for this session (so removing the last chip shows
    /// the empty state instead of immediately re-seeding defaults).
    pub seeded: bool,
    /// Whether the Runs table shows row selection checkboxes for multi-run compare.
    pub select_mode: bool,
    /// Optional run ID highlighted/isolated across overlaid charts.
    pub pinned_run: Option<String>,
    pub map_metric: CompareMapMetric,
    /// Loss component index matching `Derived::loss` (`0 = box`, `1 = cls`, `2 = dfl`, `3 = total`).
    pub loss_part: usize,
    pub loss_smooth: bool,
    pub loss_log_y: bool,
    pub x_axis: CompareXAxis,
    /// Expand identical hyperparameter rows in the comparison table.
    pub show_identical: bool,
    /// Shared hover cursor fraction `0..=1` on the comparison charts.
    pub hover: Option<f32>,
    /// Which comparison chart owns the active hover cursor (`"cmp-map"` or `"cmp-loss"`).
    pub hover_on: Option<&'static str>,
}

impl Default for RunCompareState {
    fn default() -> Self {
        Self {
            ids: Vec::new(),
            seeded: false,
            select_mode: false,
            pinned_run: None,
            map_metric: CompareMapMetric::Map50_95,
            loss_part: 3, // total loss
            loss_smooth: true,
            loss_log_y: true,
            x_axis: CompareXAxis::Epochs,
            show_identical: false,
            hover: None,
            hover_on: None,
        }
    }
}

impl RunCompareState {
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// One row of the hyperparameter comparison table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperparamRow {
    pub label: &'static str,
    pub values: Vec<String>,
    /// True when at least two compared runs have distinct formatted values.
    pub differs: bool,
    /// Per column: true when that column's value differs from column 0 (the baseline).
    pub off_base: Vec<bool>,
}

/// Picks up to `RUN_COMPARE_MAX` sensible default runs to compare, starting with `preferred` if
/// provided. Prefers runs that have recorded epochs/metrics and share the baseline run's dataset.
pub fn default_compare_runs(runs: &[Run], preferred: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    let baseline = preferred
        .and_then(|id| runs.iter().find(|r| r.id == id))
        .or_else(|| {
            runs.iter()
                .find(|r| r.best.is_some() && matches!(r.status, RunStatus::Completed | RunStatus::Running))
        })
        .or_else(|| runs.iter().find(|r| r.epoch > 0 || r.best.is_some()))
        .or_else(|| runs.first());
    let Some(base) = baseline else {
        return out;
    };
    out.push(base.id.clone());
    let base_ds = base.config.dataset_id.as_str();

    // Rank remaining runs: same dataset first, then has metrics/epochs, then highest best mAP.
    let mut rest: Vec<&Run> = runs.iter().filter(|r| r.id != base.id).collect();
    rest.sort_by(|a, b| {
        let a_same = a.config.dataset_id == base_ds;
        let b_same = b.config.dataset_id == base_ds;
        let a_has = a.best.is_some() || a.epoch > 0;
        let b_has = b.best.is_some() || b.epoch > 0;
        let a_map =
            a.best.as_ref().map_or(-1.0f32, |x| if x.map50_95.is_finite() { x.map50_95 } else { -1.0 });
        let b_map =
            b.best.as_ref().map_or(-1.0f32, |x| if x.map50_95.is_finite() { x.map50_95 } else { -1.0 });
        b_same.cmp(&a_same).then(b_has.cmp(&a_has)).then(b_map.total_cmp(&a_map))
    });
    for r in rest {
        if out.len() >= 3.min(RUN_COMPARE_MAX) {
            break;
        }
        out.push(r.id.clone());
    }
    out
}

/// Deduplicates `ids`, drops runs no longer present in `runs`, and enforces `RUN_COMPARE_MAX`.
pub fn reconcile_compare_runs(ids: &[String], runs: &[Run]) -> Vec<String> {
    let known: HashSet<&str> = runs.iter().map(|r| r.id.as_str()).collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for id in ids {
        if known.contains(id.as_str()) && seen.insert(id.as_str()) {
            out.push(id.clone());
            if out.len() >= RUN_COMPARE_MAX {
                break;
            }
        }
    }
    out
}

/// True when all compared runs trained on the same dataset (or fewer than 2 runs are compared).
pub fn same_dataset(runs: &[Run]) -> bool {
    let Some(first) = runs.first() else {
        return true;
    };
    runs.iter().all(|r| r.config.dataset_id == first.config.dataset_id)
}

/// Maps a raw epoch coordinate `x` into the active `CompareXAxis` domain (`Epochs` or `0..=100` Progress %).
pub fn map_x(x: f32, planned_epochs: u32, x_axis: CompareXAxis) -> f32 {
    match x_axis {
        CompareXAxis::Epochs => x,
        CompareXAxis::Progress => {
            let denom = planned_epochs.max(1) as f32;
            (x / denom * 100.0).clamp(0.0, 100.0)
        }
    }
}

/// Builds the overlaid validation mAP series points for one run under `x_axis`.
pub fn build_map_points(
    run: &Run,
    data: &RunData,
    metric: CompareMapMetric,
    x_axis: CompareXAxis,
) -> Vec<(f32, f32)> {
    let d = data.derived();
    let src = match metric {
        CompareMapMetric::Map50_95 => &d.map50_95,
        CompareMapMetric::Map50 => &d.map50,
    };
    src.iter()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|&(x, y)| (map_x(x, run.epochs, x_axis), y))
        .collect()
}

/// Builds the overlaid training loss series points for one run under `x_axis`.
pub fn build_loss_points(run: &Run, data: &RunData, part: usize, x_axis: CompareXAxis) -> Vec<(f32, f32)> {
    let d = data.derived();
    let idx = part.min(3);
    d.loss[idx]
        .iter()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|&(x, y)| (map_x(x, run.epochs, x_axis), y))
        .collect()
}

/// Computes the shared x-axis ceiling across `runs` and their loaded series.
pub fn shared_x_max(runs: &[Run], series: &[Vec<(f32, f32)>], x_axis: CompareXAxis) -> f32 {
    match x_axis {
        CompareXAxis::Progress => 100.0,
        CompareXAxis::Epochs => {
            let from_series = series
                .iter()
                .flat_map(|pts| pts.iter())
                .map(|p| p.0)
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max);
            let from_runs = runs.iter().map(|r| r.epoch.max(1) as f32).fold(0.0f32, f32::max);
            from_series.max(from_runs).max(1.0)
        }
    }
}

/// Marks which runs hold the highest `best.map50_95` within their dataset group (at 3-decimal
/// display precision, matching `best_in_row` on Benchmarks). A dataset with only one run in the
/// comparison gets no winner badge.
pub fn best_map_winners(runs: &[Run]) -> Vec<bool> {
    let mut winners = vec![false; runs.len()];
    let mut ds_ids: Vec<&str> = Vec::new();
    for r in runs {
        if !ds_ids.contains(&r.config.dataset_id.as_str()) {
            ds_ids.push(&r.config.dataset_id);
        }
    }
    for ds in ds_ids {
        let idxs: Vec<usize> =
            runs.iter().enumerate().filter(|(_, r)| r.config.dataset_id == ds).map(|(i, _)| i).collect();
        if idxs.len() < 2 {
            continue;
        }
        let max_v = idxs
            .iter()
            .filter_map(|&i| runs[i].best.as_ref().map(|b| b.map50_95).filter(|v| v.is_finite()))
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_v.is_finite() {
            continue;
        }
        let max_key = format!("{max_v:.3}");
        for i in idxs {
            if let Some(b) = &runs[i].best
                && b.map50_95.is_finite()
                && format!("{:.3}", b.map50_95) == max_key
            {
                winners[i] = true;
            }
        }
    }
    winners
}

fn fmt_optimizer(r: &Run) -> String {
    match &r.resolved {
        Some(s) => {
            let opt = match s.optimizer {
                ResolvedOptimizer::Sgd => "SGD",
                ResolvedOptimizer::Adamw => "AdamW",
                ResolvedOptimizer::Musgd => "MuSGD",
                ResolvedOptimizer::Unknown => "Unknown",
            };
            if r.config.optimizer == Optimizer::Auto { format!("{opt} (auto)") } else { opt.to_string() }
        }
        None => r.config.optimizer.label().to_string(),
    }
}

fn fmt_lr_value(r: &Run) -> String {
    let lr0 = r.resolved.as_ref().map_or(r.config.lr0 as f64, |s| s.lr0);
    crate::ui::fmt_lr(lr0)
}

fn fmt_batch(r: &Run) -> String {
    r.config.batch.to_string()
}

/// Builds the full hyperparameter diff table rows across `runs`.
///
/// Compares canonical IDs / exact numeric values (not formatted display labels) so distinct
/// datasets, pretrained models with matching display names, or fine-grained numeric settings never
/// falsely collapse to identical.
pub fn hyperparam_rows(
    runs: &[Run],
    dataset_name: impl Fn(&Run) -> String,
    model_name: impl Fn(&str) -> String,
) -> Vec<HyperparamRow> {
    let make = |label: &'static str, f: &dyn Fn(&Run) -> (String, String)| -> HyperparamRow {
        let pairs: Vec<(String, String)> = runs.iter().map(f).collect();
        let base_key = pairs.first().map(|p| p.0.as_str()).unwrap_or("");
        let off_base: Vec<bool> =
            pairs.iter().enumerate().map(|(i, (k, _))| i > 0 && k.as_str() != base_key).collect();
        let differs = pairs.iter().any(|(k, _)| k.as_str() != base_key);
        let values = pairs.into_iter().map(|(_, v)| v).collect();
        HyperparamRow { label, values, differs, off_base }
    };

    vec![
        make("Dataset", &|r| (r.config.dataset_id.clone(), dataset_name(r))),
        make("Architecture", &|r| {
            let a = r.config.arch.as_str().to_string();
            (a.clone(), a)
        }),
        make("Image size", &|r| (r.config.imgsz.to_string(), format!("{} px", r.config.imgsz))),
        make("Batch", &|r| (r.config.batch.to_string(), fmt_batch(r))),
        make("Learning rate", &|r| {
            let lr0 = r.resolved.as_ref().map_or(r.config.lr0 as f64, |s| s.lr0);
            (lr0.to_bits().to_string(), fmt_lr_value(r))
        }),
        make("Optimizer", &|r| {
            let key = format!("{:?}:{:?}", r.config.optimizer, r.resolved.as_ref().map(|s| s.optimizer));
            (key, fmt_optimizer(r))
        }),
        make("Precision", &|r| {
            (r.config.amp.to_string(), if r.config.amp { "mixed (f16)".into() } else { "f32".into() })
        }),
        make("Head", &|r| {
            let disp = if r.config.head == Head::O2o {
                "end-to-end (no NMS)".into()
            } else {
                "one-to-many + NMS".into()
            };
            (format!("{:?}", r.config.head), disp)
        }),
        make("Pretrained weights", &|r| {
            let key = r.config.weights.clone().unwrap_or_default();
            let disp = r.config.weights.as_deref().map_or_else(|| "none".into(), &model_name);
            (key, disp)
        }),
        make("Mosaic / mixup", &|r| {
            let a = &r.config.augment;
            let key = format!("{}:{}", a.mosaic.to_bits(), a.mixup.to_bits());
            let disp = format!(
                "{} / {}",
                if a.mosaic <= 0.0 { "off".into() } else { crate::ui::fmt_aug(a.mosaic) },
                crate::ui::fmt_aug(a.mixup)
            );
            (key, disp)
        }),
        make("Seed", &|r| (r.config.seed.to_string(), r.config.seed.to_string())),
        make("Device", &|r| {
            let d = if r.device.is_empty() { r.config.device.clone() } else { r.device.clone() };
            (d.clone(), d)
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Backend;
    use crate::api::mock::MockBackend;

    #[test]
    fn default_and_reconcile_compare_runs_prioritize_same_dataset_and_prune_deleted() {
        let api = MockBackend::new(4200, true);
        let runs = api.runs().unwrap();

        // Preferred run becomes slot 0 (baseline); same-dataset runs follow first.
        let picked = default_compare_runs(&runs, Some("run_52e8"));
        assert!(picked.len() >= 2 && picked.len() <= RUN_COMPARE_MAX, "{picked:?}");
        assert_eq!(picked[0], "run_52e8");
        let ds0 = &runs.iter().find(|r| r.id == "run_52e8").unwrap().config.dataset_id;
        let ds1 = &runs.iter().find(|r| r.id == picked[1]).unwrap().config.dataset_id;
        assert_eq!(ds0, ds1, "second pick prefers same dataset as baseline");

        // Reconcile drops missing IDs, deduplicates, and caps at RUN_COMPARE_MAX.
        let rec = reconcile_compare_runs(
            &[
                "run_52e8".into(),
                "missing".into(),
                "run_52e8".into(),
                "run_a19f".into(),
                "run_7d1e".into(),
                "run_1a2b".into(),
                "run_c9f1".into(),
            ],
            &runs,
        );
        assert_eq!(rec, vec!["run_52e8", "run_a19f", "run_7d1e", "run_1a2b"]);
    }

    #[test]
    fn hyperparam_diff_detects_changed_settings_and_best_map_winners() {
        let api = MockBackend::new(4200, true);
        let runs = api.runs().unwrap();
        let r_s = runs.iter().find(|r| r.id == "run_52e8").unwrap().clone(); // ds_3b91, folo-s, amp=true, Head::O2o
        let r_m = runs.iter().find(|r| r.id == "run_a19f").unwrap().clone(); // ds_3b91, folo-m, amp=false, Head::O2m
        let pair = vec![r_s.clone(), r_m.clone()];

        assert!(same_dataset(&pair));
        let rows = hyperparam_rows(&pair, |r| r.config.dataset_id.clone(), |m| m.to_string());
        let find = |name: &str| rows.iter().find(|row| row.label == name).unwrap();

        assert!(!find("Dataset").differs);
        assert!(find("Architecture").differs);
        assert_eq!(find("Architecture").off_base, vec![false, true]);
        assert!(find("Precision").differs);
        assert_eq!(find("Precision").values, vec!["mixed (f16)", "f32"]);
        assert!(find("Head").differs);
        assert!(!find("Image size").differs);

        // Best mAP winner marked on same dataset; different dataset alone gets no winner badge.
        let r_other = runs.iter().find(|r| r.id == "run_7d1e").unwrap().clone(); // ds_8f2c
        assert!(!same_dataset(&[r_s.clone(), r_other.clone()]));
        assert_eq!(best_map_winners(&[r_s.clone(), r_other]), vec![false, false]);

        let winners = best_map_winners(&pair);
        assert!(winners.iter().any(|&w| w), "same-dataset pair marks highest mAP50-95");
    }

    #[test]
    fn curve_extraction_supports_epochs_and_progress_axes() {
        let api = MockBackend::new(4200, true);
        let runs = api.runs().unwrap();
        let r = runs.iter().find(|r| r.id == "run_52e8").unwrap();
        let data = RunData::new(api.metrics(&r.id).unwrap(), r.steps_per_epoch);

        let ep_map = build_map_points(r, &data, CompareMapMetric::Map50_95, CompareXAxis::Epochs);
        let pct_map = build_map_points(r, &data, CompareMapMetric::Map50_95, CompareXAxis::Progress);
        assert!(!ep_map.is_empty());
        assert_eq!(ep_map.len(), pct_map.len());
        assert!((ep_map.last().unwrap().0 - r.epochs as f32).abs() < 1.0);
        assert!((pct_map.last().unwrap().0 - 100.0).abs() < 1e-3);

        let loss = build_loss_points(r, &data, 3, CompareXAxis::Epochs);
        assert!(!loss.is_empty());
        assert!(loss.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
    }
}
