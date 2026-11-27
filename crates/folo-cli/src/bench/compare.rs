//! `folo bench compare`: head-to-head comparison of two finished benchmark entries in table, Markdown, CSV,
//! or `--json` format, with atomic `--out` writing and optional `--fail-on-regression` CI gating.

use std::fmt::Write as _;

use folo_mcp::types::{BenchCompareResult, BenchConditionMatch};
use serde_json::json;

use super::{api_error, open, usage};
use crate::cli::{BenchCompareArgs, BenchCompareFormat};
use crate::output::{CliError, ExitCode, Out};
use crate::output_file::OutputFile;

pub(super) fn compare(a: BenchCompareArgs, out: Out) -> Result<(), CliError> {
    if let Some(d) = a.min_delta
        && !d.is_finite()
    {
        return Err(usage("--min-delta must be a finite number"));
    }
    let out_file =
        a.out.as_ref().map(|p| OutputFile::new_explicit("bench compare", p.clone())).transpose()?;
    let session = open(&a.conn, None)?;
    let r = session.benchmarks.compare(&a.bench, &a.baseline, &a.candidate).map_err(api_error)?;
    let rendered = render(&r, a.format);
    let written = match out_file {
        Some(mut f) => {
            f.write_bytes(rendered.as_bytes())?;
            Some(f.publish()?)
        }
        None => None,
    };

    let min_delta = a.min_delta.unwrap_or(0.0);
    let regression = a.fail_on_regression && (f64::from(r.score_delta) < min_delta);

    let mut record = folo_api::json::to_value(&r).unwrap_or_default();
    if let Some(obj) = record.as_object_mut() {
        obj.insert("format".into(), json!(a.format.as_str()));
        if let Some(path) = &written {
            obj.insert("output_path".into(), json!(crate::output::record_path(path)));
        }
        if a.fail_on_regression {
            obj.insert("min_delta".into(), json!(min_delta));
            obj.insert("regression".into(), json!(regression));
        }
    }
    out.record(&record);

    if let Some(path) = &written {
        out.human(format!(
            "wrote {} ({}; score {:+.1}, mAP50-95 {:+.3})",
            crate::output::record_path(path),
            a.format.as_str(),
            r.score_delta,
            r.accuracy.map50_95_delta
        ));
    } else {
        out.human(rendered.trim_end_matches('\n'));
    }

    if regression {
        return Err(CliError::new(
            ExitCode::Runtime,
            "bench_entry_failed",
            format!(
                "regression: candidate '{}' ({}) scored {:.1} vs baseline '{}' ({}) {:.1} (delta {:+.2} < min_delta {:+.2})",
                r.candidate.entry_id,
                r.candidate.name,
                r.candidate.score,
                r.baseline.entry_id,
                r.baseline.name,
                r.baseline.score,
                r.score_delta,
                min_delta
            ),
        )
        .hint("adjust --min-delta to allow small tolerances, or inspect the per-axis and per-condition deltas above"));
    }
    Ok(())
}

/// Renders `r` in the requested format (always ending with a trailing newline).
pub(crate) fn render(r: &BenchCompareResult, format: BenchCompareFormat) -> String {
    match format {
        BenchCompareFormat::Table => render_table(r),
        BenchCompareFormat::Markdown => render_markdown(r),
        BenchCompareFormat::Csv => render_csv(r),
    }
}

fn opt_f(v: Option<f32>, prec: usize) -> String {
    v.map_or_else(|| "-".into(), |x| format!("{x:.prec$}"))
}

fn opt_signed(v: Option<f32>, prec: usize) -> String {
    v.map_or_else(|| "-".into(), |x| format!("{x:+.prec$}"))
}

fn opt_pct(v: Option<f32>) -> String {
    v.map_or_else(|| "-".into(), |x| format!("{:.1}%", x * 100.0))
}

fn opt_signed_pct(v: Option<f32>) -> String {
    v.map_or_else(|| "-".into(), |x| format!("{:+.1}pp", x * 100.0))
}

fn match_label(m: BenchConditionMatch) -> &'static str {
    match m {
        BenchConditionMatch::Shared => "shared",
        BenchConditionMatch::BaselineOnly => "baseline_only",
        BenchConditionMatch::CandidateOnly => "candidate_only",
    }
}

fn render_table(r: &BenchCompareResult) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "benchmark {} '{}'", r.benchmark_id, r.benchmark_name);
    let _ = writeln!(
        s,
        "baseline  {} '{}' ({}, {} images, score {:.1}{})",
        r.baseline.entry_id,
        r.baseline.name,
        r.baseline.kind,
        r.baseline.images,
        r.baseline.score,
        if r.baseline.partial { "*" } else { "" }
    );
    let _ = writeln!(
        s,
        "candidate {} '{}' ({}, {} images, score {:.1}{})",
        r.candidate.entry_id,
        r.candidate.name,
        r.candidate.kind,
        r.candidate.images,
        r.candidate.score,
        if r.candidate.partial { "*" } else { "" }
    );
    let _ = writeln!(
        s,
        "score     {:.1} -> {:.1} ({:+.1} pts)",
        r.baseline.score, r.candidate.score, r.score_delta
    );
    let ci_note = if r.accuracy.ci_overlap { "95% CIs overlap" } else { "95% CIs disjoint" };
    let _ = writeln!(
        s,
        "accuracy  mAP50-95 {:.3} -> {:.3} ({:+.3}; {ci_note})  mAP50 {:+.3}  mAP75 {:+.3}  P {:+.3}  R {:+.3}",
        r.accuracy.baseline_map50_95,
        r.accuracy.candidate_map50_95,
        r.accuracy.map50_95_delta,
        r.accuracy.map50_delta,
        r.accuracy.map75_delta,
        r.accuracy.precision_delta,
        r.accuracy.recall_delta,
    );
    let _ = writeln!(
        s,
        "details   ECE delta {}  mean IoU delta {}  self-labeling recall {:+.3}",
        opt_signed(r.ece_delta, 3),
        opt_signed(r.mean_iou_delta, 3),
        r.self_labeling_recall_delta
    );
    if let Some(sp) = &r.speed {
        let speedup = sp.speedup.map_or_else(|| "-".into(), |x| format!("{x:.2}x"));
        let _ = writeln!(
            s,
            "speed     {} -> {} ms/img (delta {} ms/img, speedup {speedup})",
            opt_f(sp.baseline_ms_per_image, 2),
            opt_f(sp.candidate_ms_per_image, 2),
            opt_signed(sp.delta_ms_per_image, 2),
        );
    }

    let _ = writeln!(s);
    let _ =
        writeln!(s, "{:<14} {:>6} {:>10} {:>10} {:>10}", "axis", "weight", "baseline", "candidate", "delta");
    for ax in &r.axes {
        let _ = writeln!(
            s,
            "{:<14} {:>5.0}% {:>10} {:>10} {:>10}",
            ax.axis.as_str(),
            ax.weight * 100.0,
            opt_f(ax.baseline, 3),
            opt_f(ax.candidate, 3),
            opt_signed(ax.delta, 3),
        );
    }

    if !r.conditions.is_empty() {
        let _ = writeln!(s);
        let width = r.conditions.iter().map(|c| c.condition.len()).max().unwrap_or(9).max(9);
        let _ = writeln!(
            s,
            "{:<width$} {:<14} {:>10} {:>10} {:>10}  severities",
            "condition", "status", "baseline", "candidate", "delta"
        );
        for c in &r.conditions {
            let sev: Vec<String> = c
                .severities
                .iter()
                .map(|sv| {
                    format!(
                        "s{}:{}",
                        sv.severity,
                        sv.retention_delta.map_or_else(
                            || opt_pct(sv.candidate_retention.or(sv.baseline_retention)),
                            |d| format!("{:+.0}pp", d * 100.0)
                        )
                    )
                })
                .collect();
            let _ = writeln!(
                s,
                "{:<width$} {:<14} {:>10} {:>10} {:>10}  {}",
                c.condition,
                match_label(c.match_kind),
                opt_pct(c.baseline_retention),
                opt_pct(c.candidate_retention),
                opt_signed_pct(c.retention_delta),
                sev.join(" "),
            );
        }
    }

    if !r.warnings.is_empty() {
        let _ = writeln!(s);
        for w in &r.warnings {
            let _ = writeln!(s, "warning: {w}");
        }
    }
    s
}

fn md_escape(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

fn render_markdown(r: &BenchCompareResult) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Benchmark Comparison: {} (`{}`)\n", md_escape(&r.benchmark_name), r.benchmark_id);
    let _ = writeln!(
        s,
        "| Metric | Baseline (`{}`) | Candidate (`{}`) | Delta |",
        r.baseline.entry_id, r.candidate.entry_id
    );
    let _ = writeln!(s, "| --- | ---: | ---: | ---: |");
    let _ = writeln!(s, "| Name | {} | {} | - |", md_escape(&r.baseline.name), md_escape(&r.candidate.name));
    let _ = writeln!(
        s,
        "| Overall score | {:.1}{} | {:.1}{} | {:+.1} |",
        r.baseline.score,
        if r.baseline.partial { "*" } else { "" },
        r.candidate.score,
        if r.candidate.partial { "*" } else { "" },
        r.score_delta
    );
    let _ = writeln!(
        s,
        "| mAP@0.5:0.95 | {:.3} [{:.3}, {:.3}] | {:.3} [{:.3}, {:.3}] | {:+.3} ({}) |",
        r.accuracy.baseline_map50_95,
        r.baseline.ci95[0],
        r.baseline.ci95[1],
        r.accuracy.candidate_map50_95,
        r.candidate.ci95[0],
        r.candidate.ci95[1],
        r.accuracy.map50_95_delta,
        if r.accuracy.ci_overlap { "CIs overlap" } else { "CIs disjoint" }
    );
    let _ = writeln!(s, "| mAP@0.5 delta | - | - | {:+.3} |", r.accuracy.map50_delta);
    let _ = writeln!(s, "| mAP@0.75 delta | - | - | {:+.3} |", r.accuracy.map75_delta);
    let _ = writeln!(s, "| Precision delta | - | - | {:+.3} |", r.accuracy.precision_delta);
    let _ = writeln!(s, "| Recall delta | - | - | {:+.3} |", r.accuracy.recall_delta);
    let _ = writeln!(s, "| ECE delta | - | - | {} |", opt_signed(r.ece_delta, 3));
    let _ = writeln!(s, "| Mean IoU delta | - | - | {} |", opt_signed(r.mean_iou_delta, 3));
    let _ = writeln!(s, "| Self-labeling recall delta | - | - | {:+.3} |", r.self_labeling_recall_delta);
    if let Some(sp) = &r.speed {
        let speedup = sp.speedup.map_or_else(String::new, |x| format!(" ({x:.2}x)"));
        let _ = writeln!(
            s,
            "| Speed (ms/image) | {} | {} | {}{speedup} |",
            opt_f(sp.baseline_ms_per_image, 2),
            opt_f(sp.candidate_ms_per_image, 2),
            opt_signed(sp.delta_ms_per_image, 2),
        );
    }

    let _ = writeln!(s, "\n## Axes\n");
    let _ = writeln!(s, "| Axis | Weight | Baseline | Candidate | Delta |");
    let _ = writeln!(s, "| --- | ---: | ---: | ---: | ---: |");
    for ax in &r.axes {
        let _ = writeln!(
            s,
            "| {} | {:.0}% | {} | {} | {} |",
            ax.axis.as_str(),
            ax.weight * 100.0,
            opt_f(ax.baseline, 3),
            opt_f(ax.candidate, 3),
            opt_signed(ax.delta, 3),
        );
    }

    if !r.conditions.is_empty() {
        let _ = writeln!(s, "\n## Stress Conditions\n");
        let _ = writeln!(s, "| Condition | Status | Baseline Retention | Candidate Retention | Delta |");
        let _ = writeln!(s, "| --- | --- | ---: | ---: | ---: |");
        for c in &r.conditions {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} |",
                md_escape(&c.condition),
                match_label(c.match_kind),
                opt_pct(c.baseline_retention),
                opt_pct(c.candidate_retention),
                opt_signed_pct(c.retention_delta),
            );
        }
    }

    if !r.warnings.is_empty() {
        let _ = writeln!(s, "\n## Warnings\n");
        for w in &r.warnings {
            let _ = writeln!(s, "- {}", md_escape(w));
        }
    }
    s
}

fn csv_cell(v: &str) -> String {
    if v.contains([',', '"', '\n', '\r']) { format!("\"{}\"", v.replace('"', "\"\"")) } else { v.to_string() }
}

fn push_csv_row(out: &mut String, cols: [&str; 8]) {
    for (i, col) in cols.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&csv_cell(col));
    }
    out.push('\n');
}

fn render_csv(r: &BenchCompareResult) -> String {
    let mut s = String::new();
    push_csv_row(&mut s, ["section", "metric", "status", "baseline", "candidate", "delta", "unit", "notes"]);
    let f4 = |v: Option<f32>| v.map_or_else(String::new, |x| format!("{x:.4}"));
    push_csv_row(
        &mut s,
        [
            "overall",
            "score",
            "shared",
            &format!("{:.4}", r.baseline.score),
            &format!("{:.4}", r.candidate.score),
            &format!("{:.4}", r.score_delta),
            "points",
            &format!("{} vs {}", r.baseline.entry_id, r.candidate.entry_id),
        ],
    );
    push_csv_row(
        &mut s,
        [
            "accuracy",
            "map50_95",
            "shared",
            &format!("{:.4}", r.accuracy.baseline_map50_95),
            &format!("{:.4}", r.accuracy.candidate_map50_95),
            &format!("{:.4}", r.accuracy.map50_95_delta),
            "ratio",
            if r.accuracy.ci_overlap { "ci_overlap=true" } else { "ci_overlap=false" },
        ],
    );
    for (name, delta) in [
        ("map50", r.accuracy.map50_delta),
        ("map75", r.accuracy.map75_delta),
        ("precision", r.accuracy.precision_delta),
        ("recall", r.accuracy.recall_delta),
        ("self_labeling_recall", r.self_labeling_recall_delta),
    ] {
        push_csv_row(&mut s, ["accuracy", name, "shared", "", "", &format!("{delta:.4}"), "ratio", ""]);
    }
    push_csv_row(
        &mut s,
        [
            "calibration",
            "ece",
            if r.ece_delta.is_some() { "shared" } else { "unmeasured" },
            "",
            "",
            &f4(r.ece_delta),
            "ratio",
            "lower_is_better",
        ],
    );
    push_csv_row(
        &mut s,
        [
            "localization",
            "mean_iou",
            if r.mean_iou_delta.is_some() { "shared" } else { "unmeasured" },
            "",
            "",
            &f4(r.mean_iou_delta),
            "ratio",
            "",
        ],
    );
    if let Some(sp) = &r.speed {
        let note = sp.speedup.map_or_else(String::new, |x| format!("speedup={x:.3}x"));
        push_csv_row(
            &mut s,
            [
                "speed",
                "ms_per_image",
                if sp.delta_ms_per_image.is_some() { "shared" } else { "partial" },
                &f4(sp.baseline_ms_per_image),
                &f4(sp.candidate_ms_per_image),
                &f4(sp.delta_ms_per_image),
                "ms",
                &note,
            ],
        );
    }
    for ax in &r.axes {
        let st = match (ax.baseline, ax.candidate) {
            (Some(_), Some(_)) => "shared",
            (Some(_), None) => "baseline_only",
            (None, Some(_)) => "candidate_only",
            (None, None) => "unmeasured",
        };
        push_csv_row(
            &mut s,
            [
                "axis",
                ax.axis.as_str(),
                st,
                &f4(ax.baseline),
                &f4(ax.candidate),
                &f4(ax.delta),
                "ratio",
                &format!("weight={:.2}", ax.weight),
            ],
        );
    }
    for c in &r.conditions {
        push_csv_row(
            &mut s,
            [
                "condition",
                &c.condition,
                match_label(c.match_kind),
                &f4(c.baseline_retention),
                &f4(c.candidate_retention),
                &f4(c.retention_delta),
                "retention",
                &format!("weight={:.2}", c.weight),
            ],
        );
        for sv in &c.severities {
            let st = match (sv.baseline_retention, sv.candidate_retention) {
                (Some(_), Some(_)) => "shared",
                (Some(_), None) => "baseline_only",
                (None, Some(_)) => "candidate_only",
                (None, None) => "unmeasured",
            };
            push_csv_row(
                &mut s,
                [
                    "condition_severity",
                    &format!("{}@s{}", c.condition, sv.severity),
                    st,
                    &f4(sv.baseline_retention),
                    &f4(sv.candidate_retention),
                    &f4(sv.retention_delta),
                    "retention",
                    &format!(
                        "baseline_map={};candidate_map={};map_delta={}",
                        f4(sv.baseline_map50_95),
                        f4(sv.candidate_map50_95),
                        f4(sv.map50_95_delta)
                    ),
                ],
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use folo_api::BenchAxis;
    use folo_mcp::types::{
        BenchAccuracyDelta, BenchAxisDelta, BenchCompareResult, BenchCompareSide, BenchConditionDelta,
        BenchConditionMatch, BenchSeverityDelta, BenchSpeedDelta,
    };

    use super::*;
    use crate::cli::BenchServerArgs;

    fn sample_compare() -> BenchCompareResult {
        BenchCompareResult {
            benchmark_id: "bn_1".into(),
            benchmark_name: "Outdoor | Smoke".into(),
            baseline: BenchCompareSide {
                entry_id: "be_1".into(),
                name: "folo-n, v1".into(),
                kind: "model".into(),
                score: 70.0,
                partial: false,
                images: 100,
                conditions: true,
                stale: false,
                map50_95: 0.60,
                ci95: [0.55, 0.65],
                ms_per_image: Some(10.0),
            },
            candidate: BenchCompareSide {
                entry_id: "be_2".into(),
                name: "folo2-n \"v2\"".into(),
                kind: "model".into(),
                score: 76.5,
                partial: false,
                images: 100,
                conditions: true,
                stale: false,
                map50_95: 0.72,
                ci95: [0.68, 0.76],
                ms_per_image: Some(8.0),
            },
            score_delta: 6.5,
            accuracy: BenchAccuracyDelta {
                baseline_map50_95: 0.60,
                candidate_map50_95: 0.72,
                map50_95_delta: 0.12,
                ci_overlap: false,
                map50_delta: 0.10,
                map75_delta: 0.11,
                precision_delta: 0.05,
                recall_delta: 0.08,
            },
            axes: vec![BenchAxisDelta {
                axis: BenchAxis::Accuracy,
                weight: 0.35,
                baseline: Some(0.60),
                candidate: Some(0.72),
                delta: Some(0.12),
            }],
            ece_delta: Some(-0.02),
            mean_iou_delta: Some(0.03),
            self_labeling_recall_delta: 0.09,
            speed: Some(BenchSpeedDelta {
                baseline_ms_per_image: Some(10.0),
                candidate_ms_per_image: Some(8.0),
                delta_ms_per_image: Some(-2.0),
                speedup: Some(1.25),
                baseline_device: Some("gpu:0".into()),
                candidate_device: Some("gpu:0".into()),
            }),
            conditions: vec![
                BenchConditionDelta {
                    condition: "haze".into(),
                    weight: 1.0,
                    match_kind: BenchConditionMatch::Shared,
                    baseline_retention: Some(0.80),
                    candidate_retention: Some(0.90),
                    retention_delta: Some(0.10),
                    severities: vec![BenchSeverityDelta {
                        severity: 2,
                        baseline_map50_95: Some(0.48),
                        candidate_map50_95: Some(0.65),
                        map50_95_delta: Some(0.17),
                        baseline_retention: Some(0.80),
                        candidate_retention: Some(0.90),
                        retention_delta: Some(0.10),
                    }],
                },
                BenchConditionDelta {
                    condition: "rain".into(),
                    weight: 1.0,
                    match_kind: BenchConditionMatch::CandidateOnly,
                    baseline_retention: None,
                    candidate_retention: Some(0.75),
                    retention_delta: None,
                    severities: vec![],
                },
            ],
            warnings: vec![
                "1 stress condition(s) were run by only one entry and have no retention delta".into(),
            ],
        }
    }

    #[test]
    fn renders_table_markdown_and_csv_cleanly() {
        let r = sample_compare();
        let t = render(&r, BenchCompareFormat::Table);
        assert!(t.contains("70.0 -> 76.5 (+6.5 pts)"), "{t}");
        assert!(t.contains("95% CIs disjoint"), "{t}");
        assert!(t.contains("1.25x"), "{t}");
        assert!(t.contains("haze"), "{t}");
        assert!(t.contains("candidate_only"), "{t}");

        let md = render(&r, BenchCompareFormat::Markdown);
        assert!(md.contains("Outdoor \\| Smoke"), "{md}");
        assert!(md.contains("| Overall score |"), "{md}");
        assert!(md.contains("## Stress Conditions"), "{md}");

        let csv = render(&r, BenchCompareFormat::Csv);
        assert!(csv.starts_with("section,metric,status,baseline,candidate,delta,unit,notes\n"), "{csv}");
        assert!(csv.contains("overall,score,shared,70.0000,76.5000,6.5000,points,be_1 vs be_2\n"), "{csv}");
        assert!(csv.contains("condition,rain,candidate_only,,0.7500,,retention,weight=1.00\n"), "{csv}");
    }

    #[test]
    fn compare_rejects_non_finite_min_delta_and_existing_out_before_network() {
        let err = compare(
            BenchCompareArgs {
                bench: "bn_1".into(),
                baseline: "be_1".into(),
                candidate: "be_2".into(),
                format: BenchCompareFormat::Table,
                out: None,
                fail_on_regression: true,
                min_delta: Some(f64::NAN),
                conn: BenchServerArgs {
                    data_dir: None,
                    server: Some("http://127.0.0.1:9".into()),
                    token: None,
                    tls_ca: None,
                },
            },
            Out { json: true },
        )
        .unwrap_err();
        assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"));

        let dir = folo_test_support::TestDir::new("cli-bench-compare-out-exists");
        let existing = dir.join("report.md");
        std::fs::write(&existing, "old").unwrap();
        let err = compare(
            BenchCompareArgs {
                bench: "bn_1".into(),
                baseline: "be_1".into(),
                candidate: "be_2".into(),
                format: BenchCompareFormat::Markdown,
                out: Some(existing.clone()),
                fail_on_regression: false,
                min_delta: None,
                conn: BenchServerArgs {
                    data_dir: None,
                    server: Some("http://127.0.0.1:9".into()),
                    token: None,
                    tls_ca: None,
                },
            },
            Out { json: true },
        )
        .unwrap_err();
        assert_eq!(err.code, "output_exists");
        assert_eq!(std::fs::read_to_string(&existing).unwrap(), "old");
    }
}
