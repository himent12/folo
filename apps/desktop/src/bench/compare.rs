//! Benchmark Compare: selecting 2 to 4 finished entries of one benchmark, apples-to-apples caveats,
//! 95% confidence interval verdicts, row best markers, and per-condition / per-class aggregation.

use std::collections::HashSet;

use folo_ui::{Context, Window};

use crate::api::{Bench, BenchEntry, BenchEntryStatus, BenchResult};
use crate::app::{FoloApp, Route};

/// Maximum entries compared side by side at once (matches Studio's `COMPARE_MAX`).
pub const COMPARE_MAX: usize = 4;

/// Default number of hardest classes shown before expanding "Show all classes".
pub const COMPARE_CLS_SHOWN: usize = 12;

/// Chart palette indices (`Theme::chart_series`) for slots 0..4: sky, amber, magenta, teal.
pub const COMPARE_SLOTS: [usize; COMPARE_MAX] = [0, 1, 4, 3];

/// Whether `e` is finished and has a score, so it can be compared.
pub fn is_comparable(e: &BenchEntry) -> bool {
    e.status == BenchEntryStatus::Done && e.score.is_some()
}

/// Picks the initial entries when opening Compare: `preferred` (e.g. pinned baseline or entry page)
/// first if finished, then top ranked finished entries up to [`COMPARE_MAX`].
pub fn default_compare_entries(entries: &[BenchEntry], preferred: Option<&str>) -> Vec<String> {
    let mut out = Vec::with_capacity(COMPARE_MAX);
    if let Some(pid) = preferred
        && entries.iter().any(|e| e.id == pid && is_comparable(e))
    {
        out.push(pid.to_string());
    }
    for e in entries {
        if out.len() >= COMPARE_MAX {
            break;
        }
        if is_comparable(e) && !out.iter().any(|x| x == &e.id) {
            out.push(e.id.clone());
        }
    }
    out
}

/// Filters `selected` to distinct finished entries that still exist in `entries`, capped at
/// [`COMPARE_MAX`].
pub fn reconcile_compare_entries(selected: &[String], entries: &[BenchEntry]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(COMPARE_MAX);
    for id in selected {
        if out.len() >= COMPARE_MAX {
            break;
        }
        if entries.iter().any(|e| e.id == *id && is_comparable(e)) && seen.insert(id.as_str()) {
            out.push(id.clone());
        }
    }
    out
}

/// Plain-language caveats when compared entries are not like for like (matches Studio's
/// `compareCaveats` rules).
pub fn compare_caveats(shown: &[BenchEntry], bench: &Bench) -> Vec<String> {
    if shown.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();

    let stale: Vec<&str> =
        shown.iter().filter(|e| e.fingerprint != bench.fingerprint).map(|e| e.name.as_str()).collect();
    if !stale.is_empty() {
        out.push(format!(
            "{} ran on an earlier version of this benchmark — deltas and best-in-row exclude earlier versions.",
            stale.join(", ")
        ));
    }

    let min_imgs = shown.iter().map(|e| e.images).min().unwrap_or(0);
    let max_imgs = shown.iter().map(|e| e.images).max().unwrap_or(0);
    if min_imgs != max_imgs || min_imgs < bench.images {
        let parts: Vec<String> = shown
            .iter()
            .filter(|e| e.images < bench.images)
            .map(|e| format!("{} ({} of {})", e.name, e.images, bench.images))
            .collect();
        if !parts.is_empty() {
            out.push(format!("Fewer images graded: {}.", parts.join(", ")));
        }
    }

    let planned = bench.plan.conditions.len();
    if planned > 0 {
        let partial_conds: Vec<String> = shown
            .iter()
            .filter_map(|e| {
                if !e.conditions || e.conditions_run == 0 {
                    Some(format!("{} (unchanged only)", e.name))
                } else if (e.conditions_run as usize) < planned {
                    Some(format!("{} ({} of {planned} conditions)", e.name, e.conditions_run))
                } else {
                    None
                }
            })
            .collect();
        if !partial_conds.is_empty() {
            out.push(format!("Conditions left out: {}.", partial_conds.join(", ")));
        }
    }

    let unscored: Vec<&str> = shown.iter().filter(|e| !e.scored).map(|e| e.name.as_str()).collect();
    if !unscored.is_empty() {
        out.push(format!("No confidences used for {} — calibration is not measured.", unscored.join(", ")));
    }

    for e in shown {
        if !e.missing_classes.is_empty() {
            out.push(format!("{} does not know {} (AP 0 on those).", e.name, e.missing_classes.join(", ")));
        }
    }
    out
}

/// Scales `v` to the integer represented by `format!("{v:.decimals$}")` so best-in-row matches
/// printed values on screen.
fn displayed_scaled(v: f32, decimals: usize) -> Option<i64> {
    if !v.is_finite() {
        return None;
    }
    let formatted = format!("{v:.decimals$}");
    let neg = formatted.starts_with('-');
    let digits: String = formatted.chars().filter(char::is_ascii_digit).collect();
    let mag = digits.parse::<i64>().ok()?;
    Some(if neg { -mag } else { mag })
}

/// Marks the best displayed value(s) in a comparison row, skipping entries from an earlier
/// benchmark version (`stale`). Returns all `false` if fewer than 2 non-stale entries have a value.
pub fn best_in_row(
    values: &[Option<f32>],
    stale: &[bool],
    decimals: usize,
    higher_is_better: bool,
) -> Vec<bool> {
    let mut scaled: Vec<Option<i64>> = Vec::with_capacity(values.len());
    let mut valid_count = 0usize;
    let mut best: Option<i64> = None;
    for (i, v) in values.iter().enumerate() {
        let is_stale = stale.get(i).copied().unwrap_or(false);
        let s = if is_stale { None } else { v.and_then(|x| displayed_scaled(x, decimals)) };
        if let Some(sv) = s {
            valid_count += 1;
            best = Some(match best {
                None => sv,
                Some(b) => {
                    if higher_is_better {
                        b.max(sv)
                    } else {
                        b.min(sv)
                    }
                }
            });
        }
        scaled.push(s);
    }
    if valid_count < 2 {
        return vec![false; values.len()];
    }
    scaled.into_iter().map(|s| s.is_some() && s == best).collect()
}

/// Shared `[lo, hi]` domain for all compared entries' 95% mAP confidence intervals.
pub fn intervals_domain(shown: &[BenchEntry]) -> Option<(f32, f32)> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for e in shown {
        if let Some([a, b]) = e.ci95 {
            if a.is_finite() && b.is_finite() {
                lo = lo.min(a).min(b);
                hi = hi.max(a).max(b);
            }
        } else if let Some(m) = e.map50_95
            && m.is_finite()
        {
            lo = lo.min(m);
            hi = hi.max(m);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let span = (hi - lo).max(0.06);
    let pad = span * 0.08;
    let a = (lo - pad).clamp(0.0, 1.0);
    let b = (hi + pad).clamp(0.0, 1.0);
    if (b - a) < 1e-4 { Some((0.0, 1.0)) } else { Some((a, b)) }
}

/// Plain-English verdict on whether the top current-version entries' 95% intervals separate them.
pub fn intervals_verdict(shown: &[BenchEntry], bench: &Bench) -> String {
    let mut cur: Vec<(&BenchEntry, f32, [f32; 2])> = shown
        .iter()
        .filter(|e| e.fingerprint == bench.fingerprint)
        .filter_map(|e| {
            let m = e.map50_95.filter(|v| v.is_finite())?;
            let ci = e.ci95.filter(|c| c[0].is_finite() && c[1].is_finite())?;
            Some((e, m, ci))
        })
        .collect();
    if cur.len() < 2 {
        return "Need at least two current-version entries with 95% intervals to judge separation."
            .to_string();
    }
    cur.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (a, am, aci) = cur[0];
    let (b, bm, bci) = cur[1];
    if aci[0] > bci[1] {
        format!("{} ({am:.3}) leads {} ({bm:.3}) with non-overlapping 95% intervals.", a.name, b.name)
    } else {
        format!(
            "{} ({am:.3}) and {} ({bm:.3}) have overlapping 95% intervals — the difference may be sampling noise.",
            a.name, b.name
        )
    }
}

/// One condition row across the compared entries: accuracy retention (`0..=1`) per entry slot.
#[derive(Clone, Debug, PartialEq)]
pub struct CompareCondRow {
    pub condition: String,
    pub values: Vec<Option<f32>>,
}

/// Builds per-condition retention rows in the benchmark's plan order.
pub fn compare_condition_rows(bench: &Bench, results: &[Option<&BenchResult>]) -> Vec<CompareCondRow> {
    bench
        .plan
        .conditions
        .iter()
        .map(|pc| {
            let values = results
                .iter()
                .map(|r| {
                    r.and_then(|res| res.robustness.as_ref())
                        .and_then(|rob| rob.conditions.iter().find(|c| c.condition == pc.condition))
                        .map(|c| c.retention)
                })
                .collect();
            CompareCondRow { condition: pc.condition.clone(), values }
        })
        .collect()
}

/// One class row across the compared entries: AP50-95 (`0..=1`) per entry slot.
#[derive(Clone, Debug, PartialEq)]
pub struct CompareClassRow {
    pub class: u32,
    pub name: String,
    pub truths: u64,
    pub values: Vec<Option<f32>>,
}

/// Builds per-class AP50-95 rows sorted hardest-first (lowest mean AP across measured entries).
pub fn compare_class_rows(bench: &Bench, results: &[Option<&BenchResult>]) -> Vec<CompareClassRow> {
    let mut rows: Vec<(f32, CompareClassRow)> = bench
        .classes
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            let class = idx as u32;
            let truths = bench.class_boxes.get(idx).copied().unwrap_or(0);
            let values: Vec<Option<f32>> = results
                .iter()
                .map(|r| {
                    r.and_then(|res| res.accuracy.per_class.iter().find(|c| c.class == class))
                        .map(|c| c.map50_95)
                })
                .collect();
            let measured: Vec<f32> = values.iter().flatten().copied().filter(|v| v.is_finite()).collect();
            if truths == 0 && measured.is_empty() {
                return None;
            }
            let mean = if measured.is_empty() {
                f32::INFINITY
            } else {
                measured.iter().sum::<f32>() / measured.len() as f32
            };
            Some((mean, CompareClassRow { class, name: name.clone(), truths, values }))
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.class.cmp(&b.1.class)));
    rows.into_iter().map(|(_, r)| r).collect()
}

impl FoloApp {
    /// Opens the Benchmark Compare page for `bench_id`, seeding its compared entries if none are
    /// chosen yet (or placing `preferred_baseline` first when requested).
    pub fn open_bench_compare(
        &mut self,
        bench_id: &str,
        preferred_baseline: Option<&str>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let entries = self.bench.entries.get(bench_id).and_then(|l| l.ready()).cloned().unwrap_or_default();
        let pin_fallback =
            self.bench.pinned_entry.as_ref().filter(|(b, _)| b == bench_id).map(|(_, e)| e.as_str());
        let preferred = preferred_baseline.or(pin_fallback);
        let existing = self.bench.compare.get(bench_id).cloned().unwrap_or_default();
        let mut next = if existing.is_empty() {
            default_compare_entries(&entries, preferred)
        } else {
            let mut reconciled = reconcile_compare_entries(&existing, &entries);
            if let Some(pref) = preferred_baseline
                && entries.iter().any(|e| e.id == pref && is_comparable(e))
            {
                reconciled.retain(|x| x != pref);
                reconciled.insert(0, pref.to_string());
                reconciled.truncate(COMPARE_MAX);
            }
            if reconciled.len() < 2 { default_compare_entries(&entries, preferred) } else { reconciled }
        };
        if next.is_empty()
            && let Some(pref) = preferred
        {
            next.push(pref.to_string());
        }
        self.bench.compare.insert(bench_id.to_string(), next);
        self.bench.compare_all_classes = false;
        self.navigate(Route::BenchCompare(bench_id.to_string()), window, cx);
    }

    /// Resolves the active compared entries for `bench_id` against loaded `entries`, lazily seeding
    /// defaults on first visit (e.g. deep-link via `--open bench-compare:ID`).
    pub fn resolved_compare_entries(&mut self, bench_id: &str, entries: &[BenchEntry]) -> Vec<BenchEntry> {
        let pin = self.bench.pinned_entry.as_ref().filter(|(b, _)| b == bench_id).map(|(_, e)| e.as_str());
        let ids = match self.bench.compare.get(bench_id) {
            Some(saved) => {
                let reconciled = reconcile_compare_entries(saved, entries);
                if reconciled != *saved {
                    self.bench.compare.insert(bench_id.to_string(), reconciled.clone());
                }
                reconciled
            }
            None => {
                let defaults = default_compare_entries(entries, pin);
                self.bench.compare.insert(bench_id.to_string(), defaults.clone());
                defaults
            }
        };
        ids.iter()
            .filter_map(|id| entries.iter().find(|e| e.id == *id && is_comparable(e)).cloned())
            .collect()
    }

    /// Appends a finished entry to the comparison for `bench_id` (up to [`COMPARE_MAX`]).
    pub fn compare_add_entry(&mut self, bench_id: &str, entry_id: String, cx: &mut Context<Self>) {
        let list = self.bench.compare.entry(bench_id.to_string()).or_default();
        if list.len() < COMPARE_MAX && !list.iter().any(|x| x == &entry_id) {
            list.push(entry_id);
            cx.notify();
        }
    }

    /// Removes `entry_id` from the comparison for `bench_id`.
    pub fn compare_remove_entry(&mut self, bench_id: &str, entry_id: &str, cx: &mut Context<Self>) {
        if let Some(list) = self.bench.compare.get_mut(bench_id) {
            list.retain(|x| x != entry_id);
            cx.notify();
        }
    }

    /// Moves `entry_id` to slot 0 (baseline) for `bench_id`.
    pub fn compare_make_baseline(&mut self, bench_id: &str, entry_id: &str, cx: &mut Context<Self>) {
        if let Some(list) = self.bench.compare.get_mut(bench_id)
            && let Some(pos) = list.iter().position(|x| x == entry_id)
        {
            let id = list.remove(pos);
            list.insert(0, id);
            cx.notify();
        }
    }
}
