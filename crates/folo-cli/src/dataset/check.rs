//! `folo dataset check`: dataset diagnostics, dangling symlink detection, and COCO declared-vs-actual dimension
//! reconciliation on top of `folo_data::check::check`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use folo_data::check::{CheckIssue, CheckKind, CheckOptions, CheckReport, Severity};
use folo_data::{AuditOptions, DEFAULT_DHASH_THRESHOLD, Format, IssueKind, OpenOptions, Split};
pub(super) use folo_server::{
    coco_dangling_annotation_links, dangling_link_message, enrich_dangling_symlinks,
};
use serde_json::{Value, json};

use crate::cli::{CheckArgs, SeverityArg, bounds};
use crate::output::{CliError, ExitCode, Out};

use super::{data_error, format_name, format_of, rel, splits_json};

fn issue_json(i: &CheckIssue, root: &Path, with_fix: bool) -> Value {
    // The same object as `GET /api/datasets/{id}/issues` (folo_api::DatasetIssue), plus `fix`.
    let mut v = json!({
        "kind": i.kind,
        "severity": i.severity(),
        "file": rel(&i.file, root),
        "line": i.line,
        "message": i.message,
    });
    if with_fix && let Some(f) = &i.fix {
        v["fix"] = json!(f);
    }
    v
}

pub(super) fn check(args: CheckArgs, out: Out) -> Result<(), CliError> {
    let perceptual_threshold = if args.perceptual || args.dhash_threshold.is_some() {
        let thr = args.dhash_threshold.unwrap_or(DEFAULT_DHASH_THRESHOLD);
        if !bounds::DHASH_THRESHOLD.contains_u64(u64::from(thr)) {
            return Err(CliError::new(
                ExitCode::Usage,
                "invalid_config",
                format!("--dhash-threshold must be in 0..=64, got {thr}"),
            )
            .hint("see `folo describe dataset check --json` for the accepted ranges"));
        }
        Some(thr)
    } else {
        None
    };
    let t0 = Instant::now();
    let opts = CheckOptions { format: format_of(args.format), decode: !args.headers_only };
    let mut report = folo_data::check::check(&args.path, &opts).map_err(|e| data_error(e, &args.path))?;
    if report.format == Format::Coco {
        reconcile_coco_size_mismatches(&args.path, &mut report);
    }
    enrich_dangling_symlinks(&args.path, &mut report);
    let perceptual_json = if let Some(thr) = perceptual_threshold {
        Some(append_perceptual_audit(&args.path, &mut report, thr))
    } else {
        None
    };
    let secs = t0.elapsed().as_secs_f64();
    let (errors, warnings, infos) =
        (report.count(Severity::Error), report.count(Severity::Warning), report.count(Severity::Info));
    let total = errors + warnings + infos;
    let split_leakage = report.issues.iter().filter(|i| is_split_leakage(i)).count();
    let by_kind: serde_json::Map<String, Value> =
        report.counts().into_iter().map(|(k, n)| (k.as_str().to_string(), json!(n))).collect();
    let json_min = args.min_severity.unwrap_or(SeverityArg::Info);
    let human_min = args.min_severity.unwrap_or(SeverityArg::Warning);
    let issues: Vec<Value> = report
        .issues
        .iter()
        .filter(|i| severity_allows(json_min, i.severity()))
        .map(|i| issue_json(i, &report.root, args.fix_dry_run))
        .collect();
    let filtered_out = report.issues.len().saturating_sub(issues.len());
    let mut record = json!({
        "path": folo_api::json::path(&args.path),
        "root": folo_api::json::path(&report.root),
        "format": format_name(report.format),
        "classes": report.classes,
        "images": report.images(),
        "splits": splits_json(report.splits),
        "clean": report.is_clean(),
        "min_severity": severity_arg_name(json_min),
        "counts": {
            "error": errors,
            "warning": warnings,
            "info": infos,
            "total": total,
            "split_leakage": split_leakage,
            "by_kind": by_kind,
        },
        "filtered_out": filtered_out,
        "issues": issues,
        "seconds": secs,
    });
    if let Some(p) = perceptual_json {
        record["perceptual"] = p;
    }
    out.record(&record);
    for line in check_lines(&report, args.fix_dry_run, human_min, secs) {
        out.human(line);
    }
    if report.is_clean() {
        return Ok(());
    }
    Err(CliError::new(
        ExitCode::Runtime,
        "dataset_issues",
        format!("{} has {errors} error(s) and {warnings} warning(s)", folo_api::json::path(&args.path)),
    )
    .hint(if args.fix_dry_run {
        "nothing was changed; the `fix` of each issue says what a cleanup would do"
    } else {
        "see the issues above; --fix-dry-run shows what a cleanup would change"
    }))
}

/// Runs `folo_data::audit_dataset_with` with 64-bit `dHash` perceptual near-duplicate detection, appends
/// `DuplicateImage` warnings for every near-duplicate / cross-split leakage group, and returns a structured
/// JSON summary.
fn append_perceptual_audit(path: &Path, report: &mut CheckReport, threshold: u32) -> Value {
    let Ok(ds) = folo_data::open_with(path, &OpenOptions { format: report.format, verify_images: false })
    else {
        return json!({
            "threshold": threshold,
            "near_duplicate_groups": 0,
            "near_duplicate_leakage": 0,
        });
    };
    let audit = ds.audit_with(&AuditOptions::perceptual(threshold));
    let leakage_hashes: HashSet<&str> =
        audit.near_duplicate_leakage.iter().map(|g| g.dhash.as_str()).collect();
    for leak in &audit.near_duplicate_leakage {
        let splits: Vec<&str> = leak.splits.iter().map(|s| s.as_str()).collect();
        let Some(first) = leak.images.first() else { continue };
        let first_rel = rel(&first.path, &report.root);
        for img in leak.images.iter().skip(1) {
            report.issues.push(CheckIssue {
                kind: CheckKind::DuplicateImage,
                file: img.path.clone(),
                line: None,
                message: format!(
                    "perceptual split leakage ({}, dHash max distance {}): near-duplicate of {first_rel}",
                    splits.join(", "),
                    leak.max_distance,
                ),
                fix: Some("remove the near-duplicate image from one of the splits to prevent leakage".into()),
            });
        }
    }
    for group in &audit.near_duplicate_groups {
        if leakage_hashes.contains(group.dhash.as_str()) {
            continue;
        }
        let Some(first) = group.images.first() else { continue };
        let first_rel = rel(&first.path, &report.root);
        for img in group.images.iter().skip(1) {
            report.issues.push(CheckIssue {
                kind: CheckKind::DuplicateImage,
                file: img.path.clone(),
                line: None,
                message: format!(
                    "perceptual near-duplicate of {first_rel} (dHash max distance {})",
                    group.max_distance,
                ),
                fix: Some("remove or deduplicate the near-duplicate copy".into()),
            });
        }
    }
    json!({
        "threshold": threshold,
        "near_duplicate_groups": audit.near_duplicate_groups.len(),
        "near_duplicate_leakage": audit.near_duplicate_leakage.len(),
    })
}

/// Whether `i` is a cross-split duplicate leakage warning (`folo_data::audit` / `folo_data::check`).
fn is_split_leakage(i: &CheckIssue) -> bool {
    i.kind == CheckKind::DuplicateImage
        && (i.message.starts_with("split leakage (") || i.message.starts_with("perceptual split leakage ("))
}

/// Whether `i` is `folo_data::check`'s COCO declared-vs-actual dimension mismatch warning.
fn is_coco_size_mismatch(i: &CheckIssue) -> bool {
    i.kind == CheckKind::Other
        && i.message.starts_with("JSON says ")
        && i.message.contains(" but the image is ")
}

/// Reconciles `ZeroSizeBox` and `EmptyImage` issues for COCO images whose JSON-declared dimensions disagree with
/// their actual image header dimensions (#422).
///
/// `folo_data::check::check` opens COCO with `verify_images: false`, so boxes are clamped against the declared
/// JSON size rather than the real image size used during training/validation. When (and only when) at least one
/// image has a declared-vs-actual dimension mismatch, we re-evaluate via `open_with(..., verify_images: true)` and
/// replace only those mismatched images' `ZeroSizeBox` and `EmptyImage` issues so `folo dataset check` agrees with
/// the loader while leaving clean checks and normal loading untouched.
fn reconcile_coco_size_mismatches(path: &Path, report: &mut CheckReport) {
    let mismatched: HashSet<PathBuf> =
        report.issues.iter().filter(|i| is_coco_size_mismatch(i)).map(|i| i.file.clone()).collect();
    if mismatched.is_empty() {
        return;
    }
    let Ok(ds) = folo_data::open_with(path, &OpenOptions { format: Format::Coco, verify_images: true })
    else {
        return;
    };
    let mut real_zero: HashMap<PathBuf, Vec<CheckIssue>> = HashMap::new();
    for issue in &ds.report.issues {
        if issue.kind == IssueKind::LabelWarning && mismatched.contains(&issue.path) {
            real_zero.entry(issue.path.clone()).or_default().push(CheckIssue {
                kind: CheckKind::ZeroSizeBox,
                file: issue.path.clone(),
                line: None,
                message: issue.message.clone(),
                fix: Some("remove the degenerate annotation".into()),
            });
        }
    }
    let mut real_empty: HashMap<PathBuf, Vec<CheckIssue>> = HashMap::new();
    for split in Split::ALL {
        for entry in ds.split(split).entries() {
            if mismatched.contains(&entry.image) && entry.boxes.is_empty() {
                real_empty.entry(entry.image.clone()).or_default().push(CheckIssue {
                    kind: CheckKind::EmptyImage,
                    file: entry.image.clone(),
                    line: None,
                    message: "no annotations (background image)".into(),
                    fix: None,
                });
            }
        }
    }
    // Drop stale ZeroSizeBox / EmptyImage issues computed against the wrong declared size, and splice the
    // real-dimension issues right next to each mismatched image's size-mismatch warning.
    let mut out = Vec::with_capacity(report.issues.len());
    for issue in std::mem::take(&mut report.issues) {
        if mismatched.contains(&issue.file)
            && matches!(issue.kind, CheckKind::ZeroSizeBox | CheckKind::EmptyImage)
        {
            continue;
        }
        let anchor = is_coco_size_mismatch(&issue);
        let file = issue.file.clone();
        out.push(issue);
        if anchor {
            if let Some(zeros) = real_zero.remove(&file) {
                out.extend(zeros);
            }
            if let Some(empties) = real_empty.remove(&file) {
                out.extend(empties);
            }
        }
    }
    report.issues = out;
}

/// `Severity` in `SeverityArg` order (0 error, 1 warning, 2 info).
fn severity_rank(s: Severity) -> u8 {
    match s {
        Severity::Error => 0,
        Severity::Warning => 1,
        Severity::Info => 2,
    }
}

/// Whether an issue of severity `actual` passes a `--min-severity` threshold of `min`.
fn severity_allows(min: SeverityArg, actual: Severity) -> bool {
    let cutoff = match min {
        SeverityArg::Error => 0,
        SeverityArg::Warning => 1,
        SeverityArg::Info => 2,
    };
    severity_rank(actual) <= cutoff
}

fn severity_arg_name(s: SeverityArg) -> &'static str {
    match s {
        SeverityArg::Error => "error",
        SeverityArg::Warning => "warning",
        SeverityArg::Info => "info",
    }
}

fn check_lines(r: &CheckReport, fixes: bool, min_severity: SeverityArg, secs: f64) -> Vec<String> {
    const SHOWN: usize = 200;
    let mut l = vec![format!(
        "checked {} images ({}, {}) in {secs:.1}s",
        r.images(),
        format_name(r.format),
        folo_api::json::path(&r.root)
    )];
    let listed: Vec<_> = r.issues.iter().filter(|i| severity_allows(min_severity, i.severity())).collect();
    if !listed.is_empty() {
        l.push(String::new());
    }
    for i in listed.iter().take(SHOWN) {
        let at = match i.line {
            Some(n) => format!("{}:{n}", rel(&i.file, &r.root)),
            None => rel(&i.file, &r.root),
        };
        let sev = match i.severity() {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        };
        l.push(format!("{sev:<8} {:<17} {at}  {}", i.kind.as_str(), i.message));
        if fixes && let Some(f) = &i.fix {
            l.push(format!("{:26}would {f}", ""));
        }
    }
    if listed.len() > SHOWN {
        l.push(format!("... {} more (--json lists every issue)", listed.len() - SHOWN));
    }
    let errors = r.count(Severity::Error);
    let warnings = r.count(Severity::Warning);
    let infos = r.count(Severity::Info);
    let empty = r.counts().get(&CheckKind::EmptyImage).copied().unwrap_or(0);
    l.push(String::new());
    let summary: Vec<String> = r
        .counts()
        .into_iter()
        .filter(|(k, _)| *k != CheckKind::EmptyImage)
        .map(|(k, n)| format!("{n} {}", k.as_str()))
        .collect();
    if r.is_clean() {
        l.push(format!("clean: 0 error(s), 0 warning(s), {infos} info(s)"));
    } else {
        l.push(format!("{errors} error(s), {warnings} warning(s), {infos} info(s): {}", summary.join(", ")));
    }
    if min_severity == SeverityArg::Error && warnings > 0 {
        l.push(format!("({warnings} warning(s) hidden by --min-severity error)"));
    }
    let split_leakage = r.issues.iter().filter(|i| is_split_leakage(i)).count();
    if split_leakage > 0 {
        l.push(format!(
            "{split_leakage} cross-split duplicate leakage warning(s) (identical image content across splits; remove duplicate copies to keep evaluation unbiased)"
        ));
    }
    if empty > 0 {
        if min_severity == SeverityArg::Info {
            l.push(format!(
                "{empty} empty image(s) without boxes (background images; fine unless unintended)"
            ));
        } else {
            l.push(format!(
                "{empty} empty image(s) without boxes (background images; pass --min-severity info to list them)"
            ));
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_filter_orders_error_warning_info() {
        assert_eq!(severity_arg_name(SeverityArg::Error), "error");
        assert_eq!(severity_arg_name(SeverityArg::Warning), "warning");
        assert_eq!(severity_arg_name(SeverityArg::Info), "info");

        assert!(severity_allows(SeverityArg::Error, Severity::Error));
        assert!(!severity_allows(SeverityArg::Error, Severity::Warning));
        assert!(!severity_allows(SeverityArg::Error, Severity::Info));

        assert!(severity_allows(SeverityArg::Warning, Severity::Error));
        assert!(severity_allows(SeverityArg::Warning, Severity::Warning));
        assert!(!severity_allows(SeverityArg::Warning, Severity::Info));

        assert!(severity_allows(SeverityArg::Info, Severity::Error));
        assert!(severity_allows(SeverityArg::Info, Severity::Warning));
        assert!(severity_allows(SeverityArg::Info, Severity::Info));
    }
}
