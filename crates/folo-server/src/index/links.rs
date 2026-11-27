//! Dangling symlink and Windows directory junction detection for YOLO and COCO datasets.
//!
//! Shared by `folo-server`'s dataset index (`GET /api/datasets/{id}/issues`) and `folo dataset check` so both
//! report identical broken-link diagnostics that `folo-data`'s directory walks otherwise skip (#382).

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use folo_data::Format;
use folo_data::check::{CheckIssue, CheckKind, CheckReport};

/// Returns `true` when `meta` indicates a symlink or (on Windows) a reparse point such as a directory junction,
/// avoiding extra `read_link` syscalls on regular files and folders.
fn maybe_link(meta: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        meta.file_type().is_symlink() || (meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT) != 0
    }
    #[cfg(not(windows))]
    {
        meta.file_type().is_symlink()
    }
}

/// Returns `true` when `folo_data::guard` allows probing `p`.
///
/// For resolvable paths (`canonicalize(p).is_ok()`), trusts `folo_data::guard::check(p)` directly so a dataset
/// registered at the confinement root itself (whose parent is outside the root) stays allowed. For unresolvable
/// paths (dangling symlinks or multi-level absent paths that `folo_data::guard` lets through because
/// `canonicalize` fails), normalizes `..` and walks up to the nearest resolvable ancestor, requiring every
/// checked ancestor to pass `folo_data::guard::check`.
pub(crate) fn guard_allows_probe(p: &Path) -> bool {
    if !folo_data::guard::installed() {
        return true;
    }
    if folo_data::guard::check(p).is_err() {
        return false;
    }
    if std::fs::canonicalize(p).is_ok() {
        return true;
    }
    let abs = super::resolve_dotdot(&std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()));
    for anc in abs.ancestors().skip(1) {
        if anc.as_os_str().is_empty() {
            break;
        }
        if folo_data::guard::check(anc).is_err() {
            return false;
        }
        if std::fs::canonicalize(anc).is_ok() {
            return true;
        }
    }
    false
}

/// Returns a clear diagnostic when `p` is a symlink (or Windows directory junction) whose target cannot be
/// resolved (missing target, symlink loop, or inaccessible target). Returns `None` for regular files/dirs,
/// working symlinks, paths refused by `folo_data::guard`, or paths that do not exist at all.
pub fn dangling_link_message(p: &Path) -> Option<String> {
    if !guard_allows_probe(p) {
        return None;
    }
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

/// Detects dangling / broken symlinks (and Windows directory junctions) inside the dataset that `folo-data`'s
/// directory walks silently skip or misreport as missing files / empty background images (#382).
pub fn enrich_dangling_symlinks(input_path: &Path, report: &mut CheckReport) {
    // 1. Upgrade any existing loader issue whose file is a broken symlink (including orphan YOLO `.txt` labels,
    //    symlink-loop labels / listed images, and COCO missing images) in place, tracking normalized paths so
    //    each broken link is reported exactly once.
    let mut reported_files: HashSet<PathBuf> = HashSet::new();
    let is_yolo = report.format == Format::Yolo;
    for issue in &mut report.issues {
        if issue.message.contains(folo_data::guard::DENIED) {
            continue;
        }
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
        Format::Auto | Format::Voc => {}
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
            } else if guard_allows_probe(&cand) && cand.is_dir() {
                image_dirs.insert(cand);
            }
        }
        for lbl in [root.join("labels").join(name), root.join(name).join("labels")] {
            if dangling_link_message(&lbl).is_some() {
                extra_links.insert(lbl);
            } else if guard_allows_probe(&lbl) && lbl.is_dir() {
                label_dirs.insert(lbl);
            }
        }
    }
    if guard_allows_probe(&images_root) && images_root.is_dir() && image_dirs.is_empty() {
        image_dirs.insert(images_root);
    }
    let labels_root = root.join("labels");
    if dangling_link_message(&labels_root).is_some() {
        extra_links.insert(labels_root);
    } else if guard_allows_probe(&labels_root) && labels_root.is_dir() {
        label_dirs.insert(labels_root);
    }

    // For every image directory we discovered, also pair its mirrored label directory.
    let discovered_image_dirs: Vec<PathBuf> = image_dirs.iter().cloned().collect();
    for dir in &discovered_image_dirs {
        let probe = folo_data::label_path_for(&dir.join("probe.jpg"));
        if let Some(lbl_dir) = probe.parent() {
            if dangling_link_message(lbl_dir).is_some() {
                extra_links.insert(lbl_dir.to_path_buf());
            } else if guard_allows_probe(lbl_dir) && lbl_dir.is_dir() {
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

    // Also probe the label path of any image reported as `EmptyImage` ("no label file ...") so dangling label
    // symlinks referenced through `.txt` image lists or direct image paths in `data.yaml` are never missed.
    for issue in &report.issues {
        if issue.kind == CheckKind::EmptyImage && issue.message.starts_with("no label file") {
            for cand in [folo_data::label_path_for(&issue.file), issue.file.with_extension("txt")] {
                if let Some(msg) = dangling_link_message(&cand) {
                    dangling_labels.push((cand, msg));
                }
            }
        }
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

/// Returns every dangling symlink / junction for COCO annotation files or split/annotations directories under
/// `root`, following `folo-data`'s COCO discovery rules (`annotations/instances*.json`, ignoring `captions*` /
/// `person_keypoints*`, plus root and `{train,val,valid,test}` directories).
pub fn coco_dangling_annotation_links(root: &Path) -> Vec<(PathBuf, String)> {
    let ann_dir = root.join("annotations");
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    if let Some(msg) = dangling_link_message(&ann_dir) {
        found.push((ann_dir.clone(), msg));
    } else if folo_data::guard::check(&ann_dir).is_ok()
        && let Ok(rd) = std::fs::read_dir(&ann_dir)
    {
        let entries: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")))
            .collect();
        let has_instances = entries
            .iter()
            .any(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("instances")));
        for p in entries {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let relevant = if has_instances {
                name.starts_with("instances")
            } else {
                !name.starts_with("captions") && !name.starts_with("person_keypoints")
            };
            if relevant && let Some(msg) = dangling_link_message(&p) {
                found.push((p, msg));
            }
        }
    }

    let mut other_dirs = vec![root.to_path_buf()];
    for sub in ["train", "val", "valid", "test"] {
        other_dirs.push(root.join(sub));
    }
    for dir in other_dirs {
        if let Some(msg) = dangling_link_message(&dir) {
            found.push((dir, msg));
            continue;
        }
        if folo_data::guard::check(&dir).is_err() {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
                if (name.contains("coco") || name.starts_with("instances") || name.contains("annotations"))
                    && let Some(msg) = dangling_link_message(&p)
                {
                    found.push((p, msg));
                }
            }
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found
}

/// Scans COCO `annotations/` and root/split directories for dangling annotation `.json` or directory symlinks
/// matching `folo-data`'s COCO discovery rules (`instances*.json`, ignoring `captions*` / `person_keypoints*` and
/// unrelated root JSON files).
fn enrich_coco_dangling_symlinks(report: &mut CheckReport, reported_files: &mut HashSet<PathBuf>) {
    for (p, msg) in coco_dangling_annotation_links(&report.root) {
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
/// tracking canonical visited directories to avoid symlink cycles and respecting `folo_data::guard`.
fn walk_dangling(dir: &Path, visited: &mut HashSet<PathBuf>, on_dangling: &mut impl FnMut(&Path, String)) {
    if folo_data::guard::check(dir).is_err() {
        return;
    }
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
        } else if guard_allows_probe(&p) && p.is_dir() {
            walk_dangling(&p, visited, on_dangling);
        }
    }
}

fn find_yolo_yaml(input_path: &Path, root: &Path) -> Option<PathBuf> {
    if guard_allows_probe(input_path)
        && input_path.is_file()
        && input_path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml"))
    {
        return Some(input_path.to_path_buf());
    }
    // Check `input_path` first (handles `config/data.yaml` with `path: ../actual_data`), then `root`.
    for dir in [input_path, root] {
        if !guard_allows_probe(dir) || !dir.is_dir() {
            continue;
        }
        for name in ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"] {
            let p = dir.join(name);
            if guard_allows_probe(&p) && p.is_file() {
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

/// Candidate paths for one split entry in `yaml` (`root`-relative, `yaml_dir`-relative, and leading-`..`-stripped
/// fallbacks), filtered by [`guard_allows_probe`].
fn yaml_split_candidates(raw: &str, yaml_dir: &Path, root: &Path) -> Vec<PathBuf> {
    let p = Path::new(raw);
    let candidates: Vec<PathBuf> = if p.is_absolute() {
        vec![p.to_path_buf()]
    } else {
        let stripped: PathBuf = p
            .components()
            .skip_while(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::CurDir))
            .collect();
        vec![root.join(p), yaml_dir.join(p), root.join(&stripped), yaml_dir.join(&stripped)]
    };
    candidates.into_iter().filter(|cand| guard_allows_probe(cand)).collect()
}

/// Guarded diagnostic split candidate paths that [`enrich_dangling_symlinks`] may inspect for a YOLO dataset,
/// returned so [`Index::build`](super::Index::build) watches them before fingerprinting (including while absent,
/// so creating, removing, or retargeting a manifest-relative fallback link invalidates the cached index).
pub(crate) fn diagnostic_watch_paths(input_path: &Path, root: &Path, format: Format) -> Vec<PathBuf> {
    if format != Format::Yolo {
        return Vec::new();
    }
    let Some(yaml) = find_yolo_yaml(input_path, root) else {
        return Vec::new();
    };
    if folo_data::guard::check(&yaml).is_err() {
        return Vec::new();
    }
    let Ok(text) = std::fs::read_to_string(&yaml) else {
        return Vec::new();
    };
    let yaml_dir = yaml.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for raw in parse_yaml_split_paths(&text) {
        for cand in yaml_split_candidates(&raw, yaml_dir, root) {
            let probe = folo_data::label_path_for(&cand.join("probe.jpg"));
            if let Some(lbl_dir) = probe.parent()
                && guard_allows_probe(lbl_dir)
            {
                out.push(lbl_dir.to_path_buf());
            }
            out.push(cand);
        }
    }
    out
}

/// Reads split paths declared in a YOLO YAML (`train`, `val`, `valid`, `test`) so custom split folders (outside
/// `images/{train,val,test}`) are scanned for dangling symlinks too.
fn collect_yaml_split_dirs(
    yaml: &Path,
    root: &Path,
    image_dirs: &mut BTreeSet<PathBuf>,
    extra_links: &mut BTreeSet<PathBuf>,
) {
    if folo_data::guard::check(yaml).is_err() {
        return;
    }
    let Ok(text) = std::fs::read_to_string(yaml) else { return };
    let yaml_dir = yaml.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    for raw in parse_yaml_split_paths(&text) {
        for cand in yaml_split_candidates(&raw, yaml_dir, root) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dangling_link_message_is_none_for_regular_or_missing_paths() {
        let dir = folo_test_support::TestDir::new("server-index-dangling-unit");
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
