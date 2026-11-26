//! `folo dataset check`: dataset diagnostics, dangling symlink detection, and COCO declared-vs-actual dimension
//! reconciliation on top of `folo_data::check::check`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use folo_data::check::{CheckIssue, CheckKind, CheckOptions, CheckReport, Severity};
use folo_data::{Format, IssueKind, OpenOptions, Split};
use serde_json::{Value, json};

use crate::cli::{CheckArgs, SeverityArg};
use crate::output::{CliError, ExitCode, Out};

use super::{data_error, format_name, format_of, rel, splits_json};

/// Returns `true` when `meta` indicates a symlink or (on Windows) a reparse point such as a directory junction,
/// avoiding extra `read_link` syscalls on regular files and folders.
fn maybe_link(meta: &std::fs::Metadata) -> bool {
    if meta.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        (meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT) != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Returns a clear diagnostic when `p` is a symlink (or Windows directory junction) whose target cannot be
/// resolved (missing target, symlink loop, or inaccessible target). Returns `None` for regular files/dirs,
/// working symlinks, or paths that do not exist at all.
pub(super) fn dangling_link_message(p: &Path) -> Option<String> {
    let meta = std::fs::symlink_metadata(p).ok()?;
    if !maybe_link(&meta) {
        return None;
    }
    let target = std::fs::read_link(p).ok();
    if !meta.file_type().is_symlink() && target.is_none() {
        return None;
    }
    let err = match std::fs::metadata(p) {
        Ok(_) => return None,
        Err(e) => e,
    };
    let arrow = target.as_deref().map(|t| format!(" -> {}", folo_api::json::path(t))).unwrap_or_default();
    if err.kind() == std::io::ErrorKind::NotFound {
        Some(format!("dangling symlink{arrow} (target does not exist)"))
    } else if is_symlink_loop(&err) {
        Some(format!("broken symlink{arrow} (symlink loop)"))
    } else {
        Some(format!("broken symlink{arrow} ({err})"))
    }
}

/// Portable check for OS symlink-loop errors without depending on unstable `ErrorKind::FilesystemLoop`.
fn is_symlink_loop(err: &std::io::Error) -> bool {
    match err.raw_os_error() {
        #[cfg(target_os = "linux")]
        Some(40) => return true, // ELOOP
        #[cfg(target_os = "macos")]
        Some(62) => return true, // ELOOP
        #[cfg(windows)]
        Some(1921) => return true, // ERROR_CANT_RESOLVE_FILENAME
        _ => {}
    }
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("too many levels of symbolic links") || msg.contains("symbolic link")
}

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
    let t0 = Instant::now();
    let opts = CheckOptions { format: format_of(args.format), decode: !args.headers_only };
    let mut report = folo_data::check::check(&args.path, &opts).map_err(|e| data_error(e, &args.path))?;
    if report.format == Format::Coco {
        reconcile_coco_size_mismatches(&args.path, &mut report);
    }
    enrich_dangling_symlinks(&args.path, &mut report);
    let secs = t0.elapsed().as_secs_f64();
    let (errors, warnings, infos) =
        (report.count(Severity::Error), report.count(Severity::Warning), report.count(Severity::Info));
    let total = errors + warnings + infos;
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
    out.record(&json!({
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
            "by_kind": by_kind,
        },
        "filtered_out": filtered_out,
        "issues": issues,
        "seconds": secs,
    }));
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

/// Detects dangling / broken symlinks (and Windows directory junctions) inside the dataset that `folo-data`'s
/// directory walks silently skip or misreport as missing files / empty background images (#382).
fn enrich_dangling_symlinks(input_path: &Path, report: &mut CheckReport) {
    // 1. Upgrade any existing loader issue whose file is a broken symlink (including orphan YOLO `.txt` labels,
    //    symlink-loop labels / listed images, and COCO missing images) in place, tracking normalized paths so
    //    each broken link is reported exactly once.
    let mut reported_files: HashSet<PathBuf> = HashSet::new();
    let is_yolo = report.format == Format::Yolo;
    for issue in &mut report.issues {
        let Some(detail) = dangling_link_message(&issue.file) else { continue };
        let is_txt = issue.file.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt"));
        if is_yolo && is_txt && matches!(issue.kind, CheckKind::MissingImage | CheckKind::BrokenLabel) {
            // Convert orphan dangling `.txt` labels (from `orphan_labels`) or ELOOP labels in place to BrokenLabel.
            issue.kind = CheckKind::BrokenLabel;
            issue.message = detail;
            issue.fix = Some("fix the symlink target or delete the broken label symlink".into());
            reported_files.insert(folo_data::paths::normalize(&issue.file));
        } else if matches!(issue.kind, CheckKind::MissingImage | CheckKind::CorruptImage) {
            issue.kind = CheckKind::MissingImage;
            issue.message = detail;
            issue.fix = Some("fix the symlink target or remove the broken symlink".into());
            reported_files.insert(folo_data::paths::normalize(&issue.file));
        } else if issue.kind == CheckKind::BrokenLabel {
            issue.message = detail;
            issue.fix = Some("fix the symlink target or delete the broken label symlink".into());
            reported_files.insert(folo_data::paths::normalize(&issue.file));
        }
    }

    match report.format {
        Format::Yolo => enrich_yolo_dangling_symlinks(input_path, report, &mut reported_files),
        Format::Coco => enrich_coco_dangling_symlinks(report, &mut reported_files),
        Format::Auto => {}
    }
}

/// Scans YOLO split image directories and their paired label directories for dangling symlinks / junctions.
fn enrich_yolo_dangling_symlinks(
    input_path: &Path,
    report: &mut CheckReport,
    reported_files: &mut HashSet<PathBuf>,
) {
    let root = &report.root;
    let mut image_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    let mut label_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    let mut extra_links: BTreeSet<PathBuf> = BTreeSet::new();

    // Collect candidate split dirs from `data.yaml` (if any) and standard YOLO layout dirs.
    let yaml_path = find_yolo_yaml(input_path, root);
    if let Some(yaml) = &yaml_path {
        collect_yaml_split_dirs(yaml, root, &mut image_dirs, &mut extra_links);
    }
    let images_root = root.join("images");
    if dangling_link_message(&images_root).is_some() {
        extra_links.insert(images_root.clone());
    }
    for name in ["train", "val", "valid", "test"] {
        for cand in [images_root.join(name), root.join(name).join("images")] {
            if dangling_link_message(&cand).is_some() {
                extra_links.insert(cand);
            } else if cand.is_dir() {
                image_dirs.insert(cand);
            }
        }
        for lbl in [root.join("labels").join(name), root.join(name).join("labels")] {
            if dangling_link_message(&lbl).is_some() {
                extra_links.insert(lbl);
            } else if lbl.is_dir() {
                label_dirs.insert(lbl);
            }
        }
    }
    if images_root.is_dir() && image_dirs.is_empty() {
        image_dirs.insert(images_root);
    }
    let labels_root = root.join("labels");
    if dangling_link_message(&labels_root).is_some() {
        extra_links.insert(labels_root);
    } else if labels_root.is_dir() {
        label_dirs.insert(labels_root);
    }

    // For every image directory we discovered, also pair its mirrored label directory.
    let discovered_image_dirs: Vec<PathBuf> = image_dirs.iter().cloned().collect();
    for dir in &discovered_image_dirs {
        let probe = folo_data::label_path_for(&dir.join("probe.jpg"));
        if let Some(lbl_dir) = probe.parent() {
            if dangling_link_message(lbl_dir).is_some() {
                extra_links.insert(lbl_dir.to_path_buf());
            } else if lbl_dir.is_dir() {
                label_dirs.insert(lbl_dir.to_path_buf());
            }
        }
    }

    // Walk every image dir once (capturing dangling image links, beside-image `.txt` label links, and broken
    // subdir links in one pass without double-walking `image_dirs`).
    let mut dangling_images: Vec<(PathBuf, String)> = Vec::new();
    let mut dangling_labels: Vec<(PathBuf, String)> = Vec::new();
    let mut dangling_other: Vec<(PathBuf, String)> = Vec::new();
    for p in extra_links {
        if let Some(msg) = dangling_link_message(&p) {
            dangling_other.push((p, msg));
        }
    }
    let mut visited_dirs: HashSet<PathBuf> = HashSet::new();
    for dir in &image_dirs {
        walk_dangling(dir, &mut visited_dirs, &mut |p, msg| {
            if folo_data::is_image_path(p) {
                dangling_images.push((p.to_path_buf(), msg));
            } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt")) {
                dangling_labels.push((p.to_path_buf(), msg));
            } else {
                dangling_other.push((p.to_path_buf(), msg));
            }
        });
    }

    // Walk dedicated label dirs (skipping any already visited via `image_dirs`) for dangling `.txt` symlinks.
    for dir in &label_dirs {
        walk_dangling(dir, &mut visited_dirs, &mut |p, msg| {
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt")) {
                dangling_labels.push((p.to_path_buf(), msg));
            }
        });
    }

    // Suppress bogus `EmptyImage` ("no label file (background image)") emitted when an image's label file is a
    // dangling symlink.
    if !dangling_labels.is_empty() {
        let dangling_label_set: HashSet<PathBuf> =
            dangling_labels.iter().map(|(p, _)| folo_data::paths::normalize(p)).collect();
        report.issues.retain(|issue| {
            !(issue.kind == CheckKind::EmptyImage
                && issue.message.starts_with("no label file")
                && matches_dangling_label(&issue.file, &dangling_label_set))
        });
    }

    dangling_images.sort_by(|a, b| a.0.cmp(&b.0));
    for (p, msg) in dangling_images {
        if reported_files.insert(folo_data::paths::normalize(&p)) {
            report.issues.push(CheckIssue {
                kind: CheckKind::MissingImage,
                file: p,
                line: None,
                message: msg,
                fix: Some("fix the symlink target or remove the broken symlink".into()),
            });
        }
    }

    dangling_labels.sort_by(|a, b| a.0.cmp(&b.0));
    for (p, msg) in dangling_labels {
        if reported_files.insert(folo_data::paths::normalize(&p)) {
            report.issues.push(CheckIssue {
                kind: CheckKind::BrokenLabel,
                file: p,
                line: None,
                message: msg,
                fix: Some("fix the symlink target or delete the broken label symlink".into()),
            });
        }
    }

    dangling_other.sort_by(|a, b| a.0.cmp(&b.0));
    for (p, msg) in dangling_other {
        if reported_files.insert(folo_data::paths::normalize(&p)) {
            report.issues.push(CheckIssue {
                kind: CheckKind::Other,
                file: p,
                line: None,
                message: msg,
                fix: Some("fix the symlink target or remove the broken symlink".into()),
            });
        }
    }
}

/// Checks whether `image`'s YOLO label path (either mirrored `images` -> `labels` or beside-image `.txt`) is in
/// `dangling_labels`.
fn matches_dangling_label(image: &Path, dangling_labels: &HashSet<PathBuf>) -> bool {
    let mirrored = folo_data::paths::normalize(&folo_data::label_path_for(image));
    if dangling_labels.contains(&mirrored) {
        return true;
    }
    let beside = folo_data::paths::normalize(&image.with_extension("txt"));
    dangling_labels.contains(&beside)
}

/// Scans COCO `annotations/` and root/split directories for dangling `.json` or directory symlinks that
/// `coco::find_jsons` skips because `p.is_file()` is false for dangling links.
fn enrich_coco_dangling_symlinks(report: &mut CheckReport, reported_files: &mut HashSet<PathBuf>) {
    let root = &report.root;
    let mut dirs = vec![root.to_path_buf(), root.join("annotations")];
    for sub in ["train", "val", "valid", "test"] {
        dirs.push(root.join(sub));
    }
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    for dir in dirs {
        if let Some(msg) = dangling_link_message(&dir) {
            found.push((dir, msg));
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json"))
                && let Some(msg) = dangling_link_message(&p)
            {
                found.push((p, msg));
            }
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    for (p, msg) in found {
        if reported_files.insert(folo_data::paths::normalize(&p)) {
            report.issues.push(CheckIssue {
                kind: CheckKind::BrokenLabel,
                file: p,
                line: None,
                message: msg,
                fix: Some("fix the symlink target or remove the broken symlink".into()),
            });
        }
    }
}

/// Recursively walks `dir`, invoking `on_dangling` for every broken symlink or Windows junction encountered, while
/// tracking canonical visited directories to avoid symlink cycles.
fn walk_dangling(dir: &Path, visited: &mut HashSet<PathBuf>, on_dangling: &mut impl FnMut(&Path, String)) {
    if let Ok(canon) = std::fs::canonicalize(dir)
        && !visited.insert(canon)
    {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if let Some(msg) = dangling_link_message(&p) {
            on_dangling(&p, msg);
        } else if p.is_dir() {
            walk_dangling(&p, visited, on_dangling);
        }
    }
}

fn find_yolo_yaml(input_path: &Path, root: &Path) -> Option<PathBuf> {
    if input_path.is_file()
        && input_path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml"))
    {
        return Some(input_path.to_path_buf());
    }
    // Check `input_path` first (handles `config/data.yaml` with `path: ../actual_data`), then `root`.
    for dir in [input_path, root] {
        if !dir.is_dir() {
            continue;
        }
        for name in ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"] {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Extracts split path strings (`train`, `val`, `valid`, `test`) from YOLO `data.yaml` text, handling quoted
/// scalars, flow sequences with quoted commas (`['custom,train', images/train]`), and both indented and column-0
/// block sequences (`train:\n- dir`).
fn parse_yaml_split_paths(text: &str) -> Vec<String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut active_split = false;
    let mut raw_paths: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // A YAML sequence item (`- value` or `-`) may appear either indented or at column 0 under `train:`.
        if active_split && (trimmed == "-" || trimmed.starts_with("- ") || trimmed.starts_with("-\t")) {
            let item = trimmed.strip_prefix('-').unwrap_or("");
            let val = unquote_yaml_scalar(strip_yaml_comment(item).trim());
            if !val.is_empty() {
                raw_paths.push(val);
            }
            continue;
        }
        let indented = line.starts_with(' ') || line.starts_with('\t');
        if !indented {
            active_split = false;
            if let Some((key, rest)) = trimmed.split_once(':') {
                let key = key.trim();
                if matches!(key, "train" | "val" | "valid" | "test") {
                    active_split = true;
                    let val = strip_yaml_comment(rest).trim();
                    if let Some(inner) = val.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                        for item in split_yaml_flow_items(inner) {
                            let s = unquote_yaml_scalar(item.trim());
                            if !s.is_empty() {
                                raw_paths.push(s);
                            }
                        }
                        active_split = false;
                    } else if !val.is_empty() && !matches!(val, "null" | "~") {
                        let s = unquote_yaml_scalar(val);
                        if !s.is_empty() {
                            raw_paths.push(s);
                        }
                        active_split = false;
                    }
                }
            }
        }
    }
    raw_paths
}

/// Splits a YAML flow sequence body (inside `[...]`) by commas outside single/double quotes.
fn split_yaml_flow_items(inner: &str) -> Vec<&str> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for (i, ch) in inner.char_indices() {
        if in_double {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_double = false;
            }
        } else if in_single {
            if ch == '\'' {
                in_single = false;
            }
        } else {
            match ch {
                '\'' => in_single = true,
                '"' => in_double = true,
                ',' => {
                    items.push(&inner[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
    }
    items.push(&inner[start..]);
    items
}

/// Reads split paths declared in a YOLO YAML (`train`, `val`, `valid`, `test`) so custom split folders (outside
/// `images/{train,val,test}`) are scanned for dangling symlinks too.
fn collect_yaml_split_dirs(
    yaml: &Path,
    root: &Path,
    image_dirs: &mut BTreeSet<PathBuf>,
    extra_links: &mut BTreeSet<PathBuf>,
) {
    let Ok(text) = std::fs::read_to_string(yaml) else { return };
    let yaml_dir = yaml.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    for raw in parse_yaml_split_paths(&text) {
        let p = Path::new(&raw);
        let candidates: Vec<PathBuf> = if p.is_absolute() {
            vec![p.to_path_buf()]
        } else {
            let stripped: PathBuf = p
                .components()
                .skip_while(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::CurDir))
                .collect();
            vec![root.join(p), yaml_dir.join(p), root.join(&stripped), yaml_dir.join(&stripped)]
        };
        for cand in candidates {
            if dangling_link_message(&cand).is_some() {
                extra_links.insert(cand);
                break;
            }
            if cand.is_dir() {
                image_dirs.insert(cand);
                break;
            }
        }
    }
}

fn strip_yaml_comment(s: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if in_double {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_double = false;
            }
        } else if in_single {
            if b == b'\'' {
                in_single = false;
            }
        } else {
            match b {
                b'\'' => in_single = true,
                b'"' => in_double = true,
                b'#' if i == 0 || bytes[i - 1].is_ascii_whitespace() => return &s[..i],
                _ => {}
            }
        }
    }
    s
}

fn unquote_yaml_scalar(s: &str) -> String {
    if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        return s[1..s.len() - 1].replace("''", "'");
    }
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        return serde_json::from_str::<String>(s).unwrap_or_else(|_| s[1..s.len() - 1].to_string());
    }
    s.to_string()
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

    #[test]
    fn dangling_link_message_is_none_for_regular_or_missing_paths() {
        let dir = folo_test_support::TestDir::new("cli-check-dangling-unit");
        let regular = dir.join("a.txt");
        std::fs::write(&regular, "ok").unwrap();
        assert_eq!(dangling_link_message(&regular), None);
        assert_eq!(dangling_link_message(dir.path()), None);
        assert_eq!(dangling_link_message(&dir.join("absent.txt")), None);
    }

    #[test]
    fn parse_yaml_split_paths_handles_quoted_commas_and_column_zero_sequences() {
        let yaml = "\
train: ['custom,train', \"quoted,dir\", images/train]
val:
- col0/val
- 'col0,val2'
test:
  - indented/test # trailing comment
names: [circle]
";
        assert_eq!(
            parse_yaml_split_paths(yaml),
            vec!["custom,train", "quoted,dir", "images/train", "col0/val", "col0,val2", "indented/test"]
        );
    }
}
