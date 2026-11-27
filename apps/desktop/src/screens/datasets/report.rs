//! Dataset Health & Class-Balance Report Export (RFC 4180 CSV with formula-injection protection
//! and structured JSON), reusing `ClassBalanceSummary` and `health_checks` without duplicating
//! thresholds or metrics.

use std::collections::BTreeMap;

use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use super::balance::{BalanceSeverity, ClassBalanceSummary};
use super::health::{Finding, Severity, health_checks};
use crate::api::{Dataset, DatasetStats, IssueCounts, Label, SplitsExt};
use crate::app::{FoloApp, act, export_default_dir, picked_save};
use crate::controls::Btn;
use crate::screens::playground::write_export_bytes;
use crate::theme::{theme, ty};
use crate::ui::*;

/// Underrepresented class threshold matching `health.rs` and `balance.rs`.
const UNDERREPRESENTED_BOXES: u64 = 100;

/// One row of the sorted per-class distribution in a dataset health report.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct DatasetClassReportRow {
    pub class_id: u32,
    pub name: String,
    pub boxes: u64,
    pub share_pct: f64,
    pub status: &'static str,
}

fn severity_str(s: Severity) -> &'static str {
    match s {
        Severity::Danger => "danger",
        Severity::Warning => "warning",
        Severity::Ok => "ok",
        Severity::Info => "info",
    }
}

fn balance_severity_str(s: BalanceSeverity) -> &'static str {
    match s {
        BalanceSeverity::SingleOrEmpty => "single_or_empty",
        BalanceSeverity::Balanced => "balanced",
        BalanceSeverity::Skewed => "skewed",
        BalanceSeverity::Severe => "severe",
    }
}

/// Builds the per-class distribution sorted by `boxes DESC, class_id ASC`.
pub fn sorted_class_distribution(ds: &Dataset, s: &DatasetStats) -> Vec<DatasetClassReportRow> {
    let total_boxes: u64 =
        ds.classes.iter().enumerate().map(|(i, _)| s.class_counts.get(i).copied().unwrap_or(0)).sum();
    let denom = total_boxes.max(1) as f64;
    let mut rows: Vec<DatasetClassReportRow> = ds
        .classes
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let boxes = s.class_counts.get(i).copied().unwrap_or(0);
            let share_pct = if total_boxes > 0 { (boxes as f64 / denom) * 100.0 } else { 0.0 };
            let status = if boxes == 0 {
                "zero_boxes"
            } else if boxes < UNDERREPRESENTED_BOXES {
                "underrepresented"
            } else {
                "ok"
            };
            DatasetClassReportRow { class_id: i as u32, name: name.clone(), boxes, share_pct, status }
        })
        .collect();
    rows.sort_by(|a, b| b.boxes.cmp(&a.boxes).then_with(|| a.class_id.cmp(&b.class_id)));
    rows
}

/// Suggested filename for a dataset health report (`<slug>-health-report.csv` / `.json`).
pub fn dataset_report_filename(ds: &Dataset, ext: &str) -> String {
    let raw = if ds.name.trim().is_empty() { ds.id.as_str() } else { ds.name.as_str() };
    let mut slug = String::with_capacity(raw.len());
    let mut prev_dash = false;
    for ch in raw.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            slug.push(lower);
            prev_dash = false;
        } else if !prev_dash && !slug.is_empty() {
            slug.push('-');
            prev_dash = true;
        }
    }
    let mut slug: String = slug.chars().take(64).collect();
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug = "dataset".to_string();
    }
    format!("{slug}-health-report.{ext}")
}

fn push_report_row(
    out: &mut String,
    section: &str,
    key: &str,
    value: &str,
    count: Option<u64>,
    share_pct: Option<f64>,
    severity: &str,
    detail: &str,
) {
    push_csv_safe_text(out, section);
    out.push(',');
    push_csv_safe_text(out, key);
    out.push(',');
    push_csv_safe_text(out, value);
    out.push(',');
    if let Some(c) = count {
        out.push_str(&c.to_string());
    }
    out.push(',');
    if let Some(p) = share_pct.filter(|v| v.is_finite()) {
        out.push_str(&format!("{p:.2}"));
    }
    out.push(',');
    push_csv_safe_text(out, severity);
    out.push(',');
    push_csv_safe_text(out, detail);
    out.push('\n');
}

/// Builds a unified RFC 4180 CSV dataset health & class-balance report with spreadsheet
/// formula-injection protection across all text fields.
pub fn build_dataset_health_csv(ds: &Dataset, stats: &DatasetStats, issues: Option<&IssueCounts>) -> String {
    let summary = ClassBalanceSummary::compute(ds, stats);
    let findings = health_checks(ds, stats, issues);
    let classes = sorted_class_distribution(ds, stats);

    let mut out = String::from("section,key,value,count,share_pct,severity,detail\n");

    // 1. Summary metrics.
    push_report_row(&mut out, "summary", "dataset_id", &ds.id, None, None, "", "");
    push_report_row(&mut out, "summary", "dataset_name", &ds.name, None, None, "", "");
    push_report_row(&mut out, "summary", "format", ds.format.label(), None, None, "", "");
    push_report_row(&mut out, "summary", "total_images", "", Some(summary.total_images), None, "", "");
    push_report_row(&mut out, "summary", "total_boxes", "", Some(summary.total_boxes), None, "", "");
    push_report_row(&mut out, "summary", "class_count", "", Some(ds.classes.len() as u64), None, "", "");
    push_report_row(
        &mut out,
        "summary",
        "empty_images",
        "",
        Some(summary.empty_images),
        Some(summary.empty_share * 100.0),
        if summary.empty_share > 0.10 { "warning" } else { "ok" },
        "",
    );
    push_report_row(
        &mut out,
        "summary",
        "broken_labels",
        "",
        Some(stats.broken_labels),
        None,
        if stats.broken_labels > 0 { "danger" } else { "ok" },
        "",
    );
    let ratio_str = summary.imbalance_ratio.map(|r| format!("{r:.1}x")).unwrap_or_default();
    push_report_row(
        &mut out,
        "summary",
        "imbalance_ratio",
        &ratio_str,
        None,
        None,
        balance_severity_str(summary.severity),
        "",
    );

    // 2. Split breakdown.
    let img_denom = summary.total_images.max(1) as f64;
    for (split_name, n) in [("train", ds.splits.train), ("val", ds.splits.val), ("test", ds.splits.test)] {
        let pct = if summary.total_images > 0 { (n as f64 / img_denom) * 100.0 } else { 0.0 };
        push_report_row(&mut out, "split", split_name, "", Some(n), Some(pct), "", "");
    }

    // 3. Sorted class distribution.
    for row in &classes {
        let detail = if row.boxes == 0 {
            "No boxes in dataset"
        } else if row.boxes < UNDERREPRESENTED_BOXES {
            "Under 100 boxes"
        } else {
            ""
        };
        push_report_row(
            &mut out,
            "class",
            &row.class_id.to_string(),
            &row.name,
            Some(row.boxes),
            Some(row.share_pct),
            row.status,
            detail,
        );
    }

    // 4. Health findings.
    for f in &findings {
        push_report_row(&mut out, "finding", &f.short, "", None, None, severity_str(f.severity), f.detail);
    }

    // 5. Check issue breakdown (when `folo dataset check` counts are loaded).
    if let Some(c) = issues {
        push_report_row(&mut out, "check_issues", "errors", "", Some(c.error), None, "error", "");
        push_report_row(&mut out, "check_issues", "warnings", "", Some(c.warning), None, "warning", "");
        push_report_row(&mut out, "check_issues", "info", "", Some(c.info), None, "info", "");
        let sorted_kinds: BTreeMap<&str, u64> = c.by_kind.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        for (kind, count) in sorted_kinds {
            push_report_row(&mut out, "check_issue_kind", kind, "", Some(count), None, "", "");
        }
    }

    out
}

/// Builds a structured JSON dataset health & class-balance report.
pub fn build_dataset_health_json(
    ds: &Dataset,
    stats: &DatasetStats,
    issues: Option<&IssueCounts>,
) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct DatasetMeta<'a> {
        id: &'a str,
        name: &'a str,
        path: &'a str,
        format: &'a str,
        created: &'a str,
    }
    #[derive(serde::Serialize)]
    struct SplitBreakdown {
        train: u64,
        val: u64,
        test: u64,
        total: u64,
    }
    #[derive(serde::Serialize)]
    struct SummaryExport {
        total_images: u64,
        total_boxes: u64,
        class_count: usize,
        empty_images: u64,
        empty_share_pct: f64,
        broken_labels: u64,
        imbalance_ratio: Option<f64>,
        balance_severity: &'static str,
        dominant_class: Option<String>,
        rarest_class: Option<String>,
        zero_box_classes: Vec<String>,
        underrepresented_class_count: usize,
    }
    #[derive(serde::Serialize)]
    struct FindingExport<'a> {
        severity: &'static str,
        summary: &'a str,
        detail: &'static str,
    }
    #[derive(serde::Serialize)]
    struct IssuesExport<'a> {
        error: u64,
        warning: u64,
        info: u64,
        by_kind: BTreeMap<&'a str, u64>,
    }
    #[derive(serde::Serialize)]
    struct Payload<'a> {
        dataset: DatasetMeta<'a>,
        summary: SummaryExport,
        splits: SplitBreakdown,
        classes: Vec<DatasetClassReportRow>,
        health_findings: Vec<FindingExport<'a>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        check_issues: Option<IssuesExport<'a>>,
    }

    let summary = ClassBalanceSummary::compute(ds, stats);
    let findings: Vec<Finding> = health_checks(ds, stats, issues);
    let classes = sorted_class_distribution(ds, stats);

    let check_issues = issues.map(|c| IssuesExport {
        error: c.error,
        warning: c.warning,
        info: c.info,
        by_kind: c.by_kind.iter().map(|(k, &v)| (k.as_str(), v)).collect(),
    });
    let payload = Payload {
        dataset: DatasetMeta {
            id: &ds.id,
            name: &ds.name,
            path: &ds.path,
            format: ds.format.label(),
            created: &ds.created,
        },
        summary: SummaryExport {
            total_images: summary.total_images,
            total_boxes: summary.total_boxes,
            class_count: ds.classes.len(),
            empty_images: summary.empty_images,
            empty_share_pct: (summary.empty_share * 10_000.0).round() / 100.0,
            broken_labels: stats.broken_labels,
            imbalance_ratio: summary.imbalance_ratio,
            balance_severity: balance_severity_str(summary.severity),
            dominant_class: summary.dominant.map(|c| c.name),
            rarest_class: summary.rarest_present.map(|c| c.name),
            zero_box_classes: summary.zero_classes.into_iter().map(|(_, n)| n).collect(),
            underrepresented_class_count: summary.underrepresented_count,
        },
        splits: SplitBreakdown {
            train: ds.splits.train,
            val: ds.splits.val,
            test: ds.splits.test,
            total: ds.splits.total(),
        },
        classes,
        health_findings: findings
            .iter()
            .map(|f| FindingExport {
                severity: severity_str(f.severity),
                summary: &f.short,
                detail: f.detail,
            })
            .collect(),
        check_issues,
    };
    Ok(serde_json::to_string_pretty(&payload)?)
}

impl FoloApp {
    /// Prompts for a save path and exports the dataset health & class-balance report as CSV.
    pub fn export_dataset_health_csv(&mut self, ds_id: &str, cx: &mut Context<Self>) {
        let Some(ds) = self.store.dataset.get(ds_id).and_then(|l| l.ready()).cloned() else { return };
        let Some(stats) = ds.stats.as_ref() else {
            self.toast(
                Tone::Neutral,
                "No statistics to export",
                Some("Wait for the dataset scan to finish first.".into()),
                cx,
            );
            return;
        };
        let issues = self.store.issues.get(ds_id).and_then(|l| l.ready());
        let csv = build_dataset_health_csv(&ds, stats, issues);
        let suggested = dataset_report_filename(&ds, "csv");
        let summary = format!(
            "{} · {}",
            counted(ds.splits.total(), "image", "images"),
            counted(ds.classes.len() as u64, "class", "classes")
        );
        self.save_dataset_report_file(suggested, csv, "Exported health report CSV", summary, cx);
    }

    /// Prompts for a save path and exports the dataset health & class-balance report as JSON.
    pub fn export_dataset_health_json(&mut self, ds_id: &str, cx: &mut Context<Self>) {
        let Some(ds) = self.store.dataset.get(ds_id).and_then(|l| l.ready()).cloned() else { return };
        let Some(stats) = ds.stats.as_ref() else {
            self.toast(
                Tone::Neutral,
                "No statistics to export",
                Some("Wait for the dataset scan to finish first.".into()),
                cx,
            );
            return;
        };
        let issues = self.store.issues.get(ds_id).and_then(|l| l.ready());
        let json = match build_dataset_health_json(&ds, stats, issues) {
            Ok(s) => s,
            Err(e) => {
                self.toast(Tone::Danger, "JSON report export failed", Some(e.to_string().into()), cx);
                return;
            }
        };
        let suggested = dataset_report_filename(&ds, "json");
        let summary = format!(
            "{} · {}",
            counted(ds.splits.total(), "image", "images"),
            counted(ds.classes.len() as u64, "class", "classes")
        );
        self.save_dataset_report_file(suggested, json, "Exported health report JSON", summary, cx);
    }

    fn save_dataset_report_file(
        &mut self,
        suggested: String,
        content: String,
        ok_title: &'static str,
        summary: String,
        cx: &mut Context<Self>,
    ) {
        let dir = export_default_dir();
        let rx = cx.prompt_for_new_path(&dir, Some(&suggested));
        cx.spawn(async move |this, cx| {
            let Some(path) = picked_save(&this, rx.await, cx) else { return };
            let write_path = path.clone();
            let res = cx
                .background_executor()
                .spawn(async move { write_export_bytes(&write_path, content.as_bytes()) })
                .await;
            this.update(cx, |this, cx| match res {
                Ok(()) => {
                    let reveal = path.clone();
                    let shown = display_path(&path.to_string_lossy());
                    this.toast_with(
                        Tone::Success,
                        ok_title,
                        Some(format!("{shown} · {summary}").into()),
                        Some(("Show in folder".into(), act(move |_, _, cx| cx.reveal_path(&reveal)))),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(Tone::Danger, "Export failed", Some(e.to_string().into()), cx);
                }
            })
            .ok();
        })
        .detach();
    }

    /// Compact export footer inside the Class balance panel (mirrors Run Detail's export footer).
    pub(super) fn dataset_report_export_footer(
        &self,
        ds: &Dataset,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let did_csv = ds.id.clone();
        let did_json = ds.id.clone();
        let csv_btn = self
            .button(
                "ds-export-health-csv",
                "Health report CSV",
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(move |this, _, cx| this.export_dataset_health_csv(&did_csv, cx)),
            )
            .tooltip(crate::controls::tip(
                "Save summary metrics, split breakdown, sorted class distribution, and health findings as CSV",
                None,
            ));
        let json_btn = self
            .button(
                "ds-export-health-json",
                "Health report JSON",
                Btn::Secondary,
                Some("icons/file-text.svg"),
                window,
                cx,
                act(move |this, _, cx| this.export_dataset_health_json(&did_json, cx)),
            )
            .tooltip(crate::controls::tip(
                "Save structured dataset health, class balance, and check issues as JSON",
                None,
            ));
        div()
            .w_full()
            .flex_none()
            .border_t_1()
            .border_color(t.line_subtle)
            .px(px(12.))
            .py(px(8.))
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(8.))
            .child(text(ty::XS, t.text_tertiary).child("Export health & class-balance report"))
            .child(div().flex().flex_wrap().items_center().gap(px(6.)).child(csv_btn).child(json_btn))
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, mock::MockBackend};

    #[test]
    fn health_report_csv_and_json_include_sorted_classes_splits_findings_and_formula_protection() {
        let api = MockBackend::new(4200, true);
        let mut ds = api.dataset("ds_3b91").unwrap();
        let mut s = ds.stats.clone().unwrap();
        ds.name = "=EVIL(1,2) \"Road\"".into();
        ds.classes = vec!["+car".into(), "person, ped".into(), "@bike".into()];
        ds.splits.train = 80;
        ds.splits.val = 20;
        ds.splits.test = 10;
        s.class_counts = vec![50, 400, 0];
        s.empty_images = 12;
        s.broken_labels = 2;

        let mut by_kind = BTreeMap::new();
        by_kind.insert("duplicate_image".into(), 3);
        let issues = IssueCounts { error: 2, warning: 3, info: 12, by_kind };

        let csv = build_dataset_health_csv(&ds, &s, Some(&issues));
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "section,key,value,count,share_pct,severity,detail");
        // Formula-like dataset name and class names are neutralized with leading apostrophe + RFC 4180 quotes.
        assert!(
            lines.iter().any(|l| l.contains("\"'=EVIL(1,2) \"\"Road\"\"\"")),
            "expected neutralized dataset name in CSV:\n{csv}"
        );
        // Sorted class distribution: person (400) first, then +car (50), then @bike (0).
        let class_lines: Vec<&&str> = lines.iter().filter(|l| l.starts_with("class,")).collect();
        assert_eq!(class_lines.len(), 3);
        assert!(class_lines[0].starts_with("class,1,\"person, ped\",400,"));
        assert!(class_lines[1].starts_with("class,0,'+car,50,"));
        assert!(class_lines[2].starts_with("class,2,'@bike,0,"));
        assert!(lines.iter().any(|l| l.starts_with("split,train,,80,")));
        assert!(lines.iter().any(|l| l.starts_with("finding,2 broken labels,")));
        assert!(lines.iter().any(|l| l.starts_with("check_issue_kind,duplicate_image,,3,")));

        // JSON keeps raw names verbatim and carries all structured sections.
        let json_str = build_dataset_health_json(&ds, &s, Some(&issues)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["dataset"]["name"], ds.name);
        assert_eq!(v["splits"]["total"], 110);
        assert_eq!(v["classes"][0]["name"], "person, ped");
        assert_eq!(v["classes"][0]["boxes"], 400);
        assert_eq!(v["classes"][1]["name"], "+car");
        assert_eq!(v["classes"][2]["name"], "@bike");
        assert_eq!(v["classes"][2]["status"], "zero_boxes");
        assert_eq!(v["check_issues"]["by_kind"]["duplicate_image"], 3);
        assert!(!v["health_findings"].as_array().unwrap().is_empty());

        assert_eq!(dataset_report_filename(&ds, "csv"), "evil-1-2-road-health-report.csv");
    }
}
