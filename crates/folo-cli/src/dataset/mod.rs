//! `folo dataset check | stats | synth | convert | split | merge | filter | export`, on top of folo-data.
//!
//! - `mod.rs`: command dispatch, `stats`, `convert`, and shared error / path helpers.
//! - `check.rs`: `folo dataset check` (dangling symlink diagnostics, COCO declared-vs-actual dimension
//!   reconciliation, exact & perceptual `dHash` split-leakage reporting, severity filtering and human/JSON formatting).
//! - `synth.rs`: `folo dataset synth` (atomic sibling-staging for new folders and clean-on-failure rollback for
//!   pre-existing empty folders).
//! - `split.rs`: `folo dataset split` (deterministic multi-label stratified train/val/test splitting).
//! - `merge.rs`: `folo dataset merge` (multi-dataset class unification/remap/drop, split policies, collision-safe YOLO).
//! - `filter.rs`: `folo dataset filter` (class whitelist/blacklist, split, box area/size, image dims, drop-empty).
//! - `export.rs`: `folo dataset export` (COCO Detection JSON export with split selection and copy/link/reference modes).

mod check;
mod export;
mod filter;
mod merge;
mod split;
mod synth;

use std::path::{Path, PathBuf};
use std::time::Instant;

use folo_data::{DataError, DatasetStats, Format, Histogram, LoadedDataset, OpenOptions};
use serde_json::{Value, json};

use crate::cli::{ConvertArgs, DatasetCommand, FormatArg, StatsArgs, TargetFormatArg};
use crate::output::{CliError, ExitCode, Out};

/// Runs a `folo dataset` subcommand.
pub fn run(cmd: DatasetCommand, out: Out) -> Result<(), CliError> {
    match cmd {
        DatasetCommand::Check(a) => check::check(a, out),
        DatasetCommand::Stats(a) => stats(a, out),
        DatasetCommand::Synth(a) => synth::synth(a, out),
        DatasetCommand::Convert(a) => convert(a, out),
        DatasetCommand::Split(a) => split::split(a, out),
        DatasetCommand::Merge(a) => merge::merge(a, out),
        DatasetCommand::Filter(a) => filter::filter(a, out),
        DatasetCommand::Export(a) => export::export(a, out),
    }
}

pub(crate) fn format_of(f: FormatArg) -> Format {
    match f {
        FormatArg::Auto => Format::Auto,
        FormatArg::Yolo => Format::Yolo,
        FormatArg::Coco => Format::Coco,
        FormatArg::Voc => Format::Voc,
    }
}

pub(super) fn target_format_of(f: TargetFormatArg) -> Format {
    match f {
        TargetFormatArg::Yolo => Format::Yolo,
        TargetFormatArg::Coco => Format::Coco,
        TargetFormatArg::Voc => Format::Voc,
    }
}

pub(super) fn format_name(f: Format) -> &'static str {
    match f {
        Format::Auto => "auto",
        Format::Yolo => "yolo",
        Format::Coco => "coco",
        Format::Voc => "voc",
    }
}

/// Maps a folo-data error to the CLI's codes and exit codes.
pub(super) fn data_error(e: DataError, path: &Path) -> CliError {
    match e {
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            if let Some(detail) = check::dangling_link_message(path) {
                return CliError::new(
                    ExitCode::NotFound,
                    "path_not_found",
                    format!("{}: {detail}", folo_api::json::path(path)),
                )
                .hint("fix the symlink target or pass the real dataset path");
            }
            CliError::new(
                ExitCode::NotFound,
                "path_not_found",
                format!("{} does not exist", folo_api::json::path(path)),
            )
            .hint("pass a dataset folder, a data.yaml, a COCO instances .json or a Pascal VOC .xml")
        }
        // An output folder that `folo_data::unfinished::claim` refused: another writer's, or not empty.
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::AlreadyExists => {
            output_exists(path, true)
        }
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::DirectoryNotEmpty => {
            output_exists(path, false)
        }
        e @ DataError::Unfinished(_) => CliError::new(ExitCode::Runtime, "dataset_unfinished", e.to_string())
            .hint(folo_data::unfinished::HINT),
        DataError::Format(m) if m.starts_with(folo_data::UNDETECTED) => {
            CliError::new(ExitCode::Runtime, "unknown_format", m).hint(
                "expected data.yaml or images/ + labels/ (YOLO), annotations/instances_*.json (COCO), or Annotations/*.xml + JPEGImages/ (Pascal VOC)",
            )
        }
        DataError::Format(m) => CliError::new(ExitCode::Runtime, "invalid_dataset", m),
        DataError::Config(m) => {
            CliError::new(ExitCode::Usage, "usage", m).hint("see `folo dataset <command> --help`")
        }
        e => CliError::new(ExitCode::Runtime, "io_error", e.to_string()),
    }
}

/// Maps a folo-data error from writing an output dataset (`synth`, `convert`) so a missing output drive or parent
/// reports an output-path hint instead of the missing *input* dataset hint.
pub(super) fn output_data_error(e: DataError, out_dir: &Path) -> CliError {
    match e {
        DataError::Io { path: err_path, source } if source.kind() == std::io::ErrorKind::NotFound => {
            if err_path.starts_with(out_dir) || out_dir.starts_with(&err_path) {
                CliError::new(
                    ExitCode::NotFound,
                    "path_not_found",
                    format!("cannot create {}: {source}", folo_api::json::path(&err_path)),
                )
                .hint("check that the output drive and parent path exist and are writable")
            } else {
                CliError::new(
                    ExitCode::NotFound,
                    "path_not_found",
                    format!("{} does not exist", folo_api::json::path(&err_path)),
                )
                .hint("an input file disappeared while writing; run `folo dataset check` on the input")
            }
        }
        other => data_error(other, out_dir),
    }
}

/// Canonical path of `p`, resolving each existing prefix with filesystem semantics before applying any subsequent
/// `..` (so `<symlink>/../out` resolves `<symlink>` first instead of collapsing `<symlink>/..` lexically), and
/// keeping any remaining non-existent suffix components. Returns `None` when not even an ancestor root exists.
pub(super) fn canonical_target(p: &Path) -> Option<PathBuf> {
    let p = if p.as_os_str().is_empty() { Path::new(".") } else { p };
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let mut cur = PathBuf::new();
    let mut missing = false;
    for c in abs.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::Prefix(prefix) => {
                // Wait for RootDir before canonicalising so drive-relative `D:` is never resolved to cwd.
                cur.push(prefix.as_os_str());
            }
            std::path::Component::RootDir => {
                cur.push(std::path::Component::RootDir);
                if let Ok(canon) = std::fs::canonicalize(&cur) {
                    cur = canon;
                    missing = false;
                } else {
                    missing = true;
                }
            }
            std::path::Component::ParentDir => {
                // When `!missing`, `cur` is already canonical (no symlinks remain), so popping avoids Win32
                // rejecting `..` after a `\\?\` verbatim prefix while preserving symlink-before-`..` semantics.
                let _ = cur.pop();
                if let Ok(canon) = std::fs::canonicalize(&cur) {
                    cur = canon;
                    missing = false;
                } else {
                    missing = true;
                }
            }
            std::path::Component::Normal(part) => {
                if part == "." {
                    continue;
                }
                if part == ".." {
                    let _ = cur.pop();
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                        missing = false;
                    } else {
                        missing = true;
                    }
                    continue;
                }
                cur.push(part);
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                    } else {
                        missing = true;
                    }
                }
            }
        }
    }
    if cur.as_os_str().is_empty() {
        return None;
    }
    if missing && !cur.ancestors().skip(1).any(Path::exists) {
        return None;
    }
    Some(cur)
}

/// Refuses a `convert` or `split` output inside (or containing) the input dataset, its supplied folder, or any of
/// its source folders, so output images are never written into the input and picked up on its next open.
pub(super) fn check_convert_output_outside(
    output: &Path,
    input: &Path,
    ds: &LoadedDataset,
) -> Result<(), CliError> {
    let Some(dir) = canonical_target(output) else {
        return Ok(());
    };
    let input_dir: PathBuf = if input.is_dir() {
        input.to_path_buf()
    } else {
        let abs = folo_data::paths::normalize(input);
        abs.parent().map_or(abs.clone(), Path::to_path_buf)
    };
    let mut dirs: Vec<(&Path, PathBuf)> = Vec::new();
    for p in ds.sources.iter().map(PathBuf::as_path).chain([ds.root.as_path(), input_dir.as_path()]) {
        if p.is_dir()
            && let Ok(canon) = std::fs::canonicalize(p)
        {
            dirs.push((p, canon));
        }
    }
    // Prefer the deepest (most specific) input folder that contains `output`, e.g. `ds/images/train` over `ds`.
    dirs.sort_by_key(|(_, c)| std::cmp::Reverse(c.components().count()));
    let out_shown = folo_api::json::path(output);
    for (raw, src) in dirs {
        let src_shown = folo_api::json::path(raw);
        let message = if dir.starts_with(&src) {
            let rel = if dir == src { "is" } else { "is inside" };
            format!(
                "output {out_shown} {rel} the input dataset folder {src_shown}: writing into the input would \
                 mix output files into it"
            )
        } else if src.starts_with(&dir) {
            format!(
                "the input dataset folder {src_shown} is inside output {out_shown}: pick a separate output folder"
            )
        } else {
            continue;
        };
        return Err(CliError::new(ExitCode::Usage, "usage", message)
            .hint("pick an output folder outside the input dataset"));
    }
    Ok(())
}

/// Refuses an output folder that exists and is not empty (or is a file / symlink to something non-directory).
pub(super) fn ensure_empty_output(dir: &Path) -> Result<(), CliError> {
    let busy = if dir.is_dir() {
        std::fs::read_dir(dir).map(|mut rd| rd.next().is_some()).unwrap_or(true)
    } else {
        // Also refuse a dangling symlink in `dir`'s place rather than silently racing or failing late.
        dir.exists() || std::fs::symlink_metadata(dir).is_ok()
    };
    if busy {
        return Err(output_exists(dir, dir.join(folo_data::unfinished::MARKER).exists()));
    }
    Ok(())
}

/// `output_exists` for `dir`; `unfinished` when it holds another writer's marker.
pub(super) fn output_exists(dir: &Path, unfinished: bool) -> CliError {
    CliError::new(
        ExitCode::Runtime,
        "output_exists",
        format!("{} exists and is not an empty folder", folo_api::json::path(dir)),
    )
    .hint(if unfinished {
        folo_data::unfinished::HINT
    } else {
        "pick a new folder, or delete this one first"
    })
}

/// `p` relative to `root` with `/` separators, or `p` as is when it is outside `root`.
pub(super) fn rel(p: &Path, root: &Path) -> String {
    // The server names issue files the same way (GET /api/datasets/{id}/issues).
    folo_data::paths::relative_to(p, root)
}

pub(super) fn splits_json(s: [usize; 3]) -> Value {
    json!({ "train": s[0], "val": s[1], "test": s[2] })
}

// ---- stats -------------------------------------------------------------------------------------------------------

pub(super) fn open(path: &Path, format: Format) -> Result<LoadedDataset, CliError> {
    folo_data::open_with(path, &OpenOptions { format, verify_images: true }).map_err(|e| data_error(e, path))
}

fn stats(args: StatsArgs, out: Out) -> Result<(), CliError> {
    let ds = open(&args.path, format_of(args.format))?;
    let stats = ds.stats();
    let skipped = ds.report.skipped_images();
    let name = crate::engine::dataset_name(&args.path);
    out.record(&json!({
        "name": name,
        "path": folo_api::json::path(&args.path),
        "root": folo_api::json::path(&ds.root),
        "format": format_name(ds.format),
        "classes": ds.classes,
        "splits": splits_json(ds.split_sizes()),
        "skipped_images": skipped,
        "stats": stats,
    }));
    for line in stats_table(&name, &ds, &stats) {
        out.human(line);
    }
    Ok(())
}

const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// One character per value, scaled to the largest: zero is the baseline `▁`, any other value at least `▂`.
fn sparkline(values: &[u64]) -> String {
    let max = values.iter().copied().max().unwrap_or(0);
    values
        .iter()
        .map(|&v| if v == 0 || max == 0 { BLOCKS[0] } else { BLOCKS[((v * 7).div_ceil(max)) as usize] })
        .collect()
}

/// A horizontal bar `width` cells long at `max`, with eighth-cell resolution; non-zero values show at least a sliver.
fn bar(v: u64, max: u64, width: usize) -> String {
    const PARTS: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    if v == 0 || max == 0 {
        return String::new();
    }
    let eighths = ((v as f64 / max as f64) * (width * 8) as f64).round().max(1.0) as usize;
    let mut s = "█".repeat(eighths / 8);
    if !eighths.is_multiple_of(8) {
        s.push(PARTS[eighths % 8]);
    }
    s
}

fn hist_rows(h: &Histogram, label: impl Fn(f64, f64, bool) -> String) -> Vec<String> {
    let max = h.counts.iter().copied().max().unwrap_or(0);
    let total = h.total().max(1);
    let n = h.counts.len();
    h.counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let last = i + 1 == n;
            format!(
                "  {:<13} {:>8}  {:>5.1}%  {}",
                label(h.edges[i], h.edges[i + 1], last),
                c,
                100.0 * c as f64 / total as f64,
                bar(c, max, 30)
            )
        })
        .collect()
}

/// Maximum character width of a class label in the human `folo dataset stats` table.
const MAX_CLASS_NAME_CHARS: usize = 24;

/// Truncates `name` to at most `max_chars` Unicode scalar values, marking truncated labels with `…`.
fn truncate_class_name(name: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if name.chars().count() <= max_chars {
        return name.to_string();
    }
    let mut s: String = name.chars().take(max_chars - 1).collect();
    s.push('…');
    s
}

fn stats_table(name: &str, ds: &LoadedDataset, s: &DatasetStats) -> Vec<String> {
    let [train, val, test] = ds.split_sizes();
    let images = train + val + test;
    let mut l = vec![
        format!("{name}  ({}, {})", format_name(ds.format), folo_api::json::path(&ds.root)),
        format!(
            "images   {images} (train {train} · val {val} · test {test})   boxes {}   empty {}   broken labels {}{}",
            s.boxes,
            s.empty_images,
            s.broken_labels,
            match ds.report.skipped_images() {
                0 => String::new(),
                n => format!("   unreadable images {n}"),
            }
        ),
        String::new(),
    ];
    let max = s.class_counts.iter().copied().max().unwrap_or(0);
    let total = s.boxes.max(1);
    const SHOWN: usize = 40;
    let mut order: Vec<usize> = (0..s.class_counts.len()).collect();
    if order.len() > SHOWN {
        order.sort_by_key(|&i| std::cmp::Reverse(s.class_counts[i]));
    }
    let idx_width = s.class_counts.len().saturating_sub(1).to_string().len().max(2);
    let shown_names: Vec<(usize, String)> = order
        .iter()
        .take(SHOWN)
        .map(|&i| {
            let raw = ds.classes.get(i).cloned().unwrap_or_else(|| format!("class{i}"));
            (i, truncate_class_name(&raw, MAX_CLASS_NAME_CHARS))
        })
        .collect();
    let width =
        shown_names.iter().map(|(_, n)| n.chars().count()).max().unwrap_or(5).clamp(5, MAX_CLASS_NAME_CHARS);
    l.push(format!("  {:>idx_width$}  {:<width$} {:>8}  {:>6}", "id", "class", "boxes", "share"));
    for (i, label) in shown_names {
        let c = s.class_counts[i];
        l.push(format!(
            "  {i:>idx_width$}  {label:<width$} {c:>8}  {:>5.1}%  {}",
            100.0 * c as f64 / total as f64,
            bar(c, max, 30)
        ));
    }
    if order.len() > SHOWN {
        l.push(format!("  ... {} more classes (--json lists all)", order.len() - SHOWN));
    }
    l.push(String::new());
    l.push(format!("box size  sqrt(box area / image area)   {}", sparkline(&s.box_size_hist.counts)));
    l.extend(hist_rows(&s.box_size_hist, |a, b, last| {
        if last { format!("≥ {a:.2}") } else { format!("{a:.2} – {b:.2}") }
    }));
    l.push(String::new());
    l.push(format!("boxes per image   {}", sparkline(&s.boxes_per_image_hist.counts)));
    l.extend(hist_rows(&s.boxes_per_image_hist, |a, b, last| {
        let (a, b) = (a as u64, b as u64);
        if last {
            format!("≥ {a}")
        } else if b == a + 1 {
            format!("{a}")
        } else {
            format!("{a} – {}", b - 1)
        }
    }));
    l
}

// ---- convert -----------------------------------------------------------------------------------------------------

/// Whether `path` is a confirmed COCO annotation location (explicitly passed input file, inside `annotations/` or
/// a split folder `train|val|valid|test/`, a split/annotations directory itself, or a root `instances*.json` /
/// `_annotations.coco.json`) as opposed to a root-level heuristic candidate like `coco_predictions.json` whose lack
/// of an `images` array is benign.
fn is_confirmed_coco_annotation(path: &Path, input: &Path, root: &Path) -> bool {
    if input.is_file() {
        return true;
    }
    if path == root.join("annotations")
        || ["train", "val", "valid", "test"].iter().any(|s| path == root.join(s))
    {
        return true;
    }
    let parent = path.parent();
    if parent.is_some_and(|p| p == root.join("annotations"))
        || ["train", "val", "valid", "test"].iter().any(|s| parent.is_some_and(|p| p == root.join(s)))
    {
        return true;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    name.starts_with("instances") || name == "_annotations.coco.json"
}

/// Whole COCO annotation files or Pascal VOC `ImageSets/Main/*.txt` split manifests that `folo-data` could not
/// read or skipped because of a broken symlink, returning `(path, message)` pairs so `convert` and `split` never
/// silently drop an entire split.
pub(super) fn unreadable_split_manifests(ds: &LoadedDataset, input: &Path) -> Vec<(PathBuf, String)> {
    match ds.format {
        Format::Coco => {
            let mut bad: Vec<(PathBuf, String)> = ds
                .report
                .issues
                .iter()
                .filter(|i| {
                    i.kind == folo_data::IssueKind::BrokenLabel
                        && (i.message.starts_with("invalid JSON:")
                            || (i.message == "no `images` array: not a COCO file"
                                && is_confirmed_coco_annotation(&i.path, input, &ds.root)))
                })
                .map(|i| (i.path.clone(), i.message.clone()))
                .collect();
            if !input.is_file() {
                for (p, msg) in check::coco_dangling_annotation_links(&ds.root) {
                    if is_confirmed_coco_annotation(&p, input, &ds.root) {
                        bad.push((p, msg));
                    }
                }
            }
            bad
        }
        Format::Voc => {
            let mut bad: Vec<(PathBuf, String)> = ds
                .report
                .issues
                .iter()
                .filter(|i| {
                    i.kind == folo_data::IssueKind::Other
                        && i.message.starts_with("cannot read ImageSets split file:")
                })
                .map(|i| (i.path.clone(), i.message.clone()))
                .collect();
            for rel in ["ImageSets/Main", "imagesets/main", "ImageSets/main", "imagesets/Main"] {
                let dir = ds.root.join(rel);
                if !dir.is_dir() {
                    continue;
                }
                for name in ["train.txt", "val.txt", "trainval.txt", "test.txt"] {
                    let p = dir.join(name);
                    if let Some(msg) = check::dangling_link_message(&p) {
                        bad.push((p, msg));
                    }
                }
                break;
            }
            bad
        }
        _ => Vec::new(),
    }
}

fn convert(args: ConvertArgs, out: Out) -> Result<(), CliError> {
    let to = target_format_of(args.to);
    let ds = open(&args.input, format_of(args.from))?;
    let bad_splits = unreadable_split_manifests(&ds, &args.input);
    if !bad_splits.is_empty() {
        let (first_path, first_msg) = &bad_splits[0];
        let mut msg = format!("{}: {first_msg}", rel(first_path, &ds.root));
        if bad_splits.len() > 1 {
            msg.push_str(&format!(" (+{} more unreadable split file(s))", bad_splits.len() - 1));
        }
        return Err(CliError::new(ExitCode::Runtime, "invalid_dataset", msg).hint(
            folo_api::json::cli_command(
                "fix the corrupt annotation or split file; run `folo dataset check {dataset}` to inspect all issues",
                &[("dataset", &args.input)],
            ),
        ));
    }
    check_convert_output_outside(&args.output, &args.input, &ds)?;
    ensure_empty_output(&args.output)?;
    let skipped = ds.report.skipped_images() + ds.report.skipped_label_images();
    let t0 = Instant::now();
    let done = folo_data::convert::convert(&ds, &args.output, to).map_err(|e| match e {
        // The input cannot be written in that format (no images, or only test images for YOLO): not a usage error.
        DataError::Config(m) => CliError::new(ExitCode::Runtime, "invalid_dataset", m),
        e => output_data_error(e, &args.output),
    })?;
    let secs = t0.elapsed().as_secs_f64();
    out.record(&json!({
        "input": folo_api::json::path(&args.input),
        "output": folo_api::json::path(&done.root),
        "from": format_name(ds.format),
        "to": format_name(to),
        "classes": ds.classes,
        "splits": splits_json(done.splits),
        "boxes": done.boxes,
        "skipped_images": skipped,
        "seconds": secs,
    }));
    let [train, val, test] = done.splits;
    out.human(format!(
        "converted {} {} → {} at {} ({train} train / {val} val / {test} test images, {} boxes) in {secs:.1}s",
        format_name(ds.format),
        folo_api::json::path(&args.input),
        format_name(to),
        folo_api::json::path(&done.root),
        done.boxes
    ));
    if skipped > 0 {
        out.human(format!(
            "  {skipped} image(s) were skipped (unreadable or broken labels); {}",
            folo_api::json::cli_command("run `folo dataset check {dataset}`", &[("dataset", &args.input)])
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparklines_and_bars() {
        assert_eq!(sparkline(&[0, 1, 7, 14]), "▁▂▅█");
        assert_eq!(sparkline(&[0, 0]), "▁▁");
        assert_eq!(bar(0, 10, 4), "");
        assert_eq!(bar(10, 10, 4), "████");
        assert_eq!(bar(1, 1000, 4), "▏", "non-zero values show a sliver");
        assert_eq!(bar(5, 10, 3), "█▌");
        assert_eq!(rel(Path::new("/d/x/images/a.jpg"), Path::new("/d/x")), "images/a.jpg");
    }

    #[test]
    fn truncate_class_name_marks_long_labels_with_ellipsis() {
        assert_eq!(truncate_class_name("circle", 24), "circle");
        assert_eq!(truncate_class_name("a".repeat(24).as_str(), 24), "a".repeat(24));
        let truncated = truncate_class_name("industrial_safety_helmet_yellow", 24);
        assert_eq!(truncated.chars().count(), 24);
        assert!(truncated.ends_with('…'), "{truncated}");
        assert_eq!(truncated, "industrial_safety_helme…");
        // Unicode-safe truncation.
        assert_eq!(truncate_class_name("héllo_wörld", 5), "héll…");
        assert_eq!(truncate_class_name("abc", 1), "…");
        assert_eq!(truncate_class_name("abc", 0), "");
    }

    #[test]
    fn output_data_error_gives_output_hint_for_missing_output_path_and_input_hint_for_missing_input() {
        let out_dir = Path::new("Z:/no-such-drive/out");
        let missing_out = DataError::Io {
            path: out_dir.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "the system cannot find the path specified",
            ),
        };
        let err = output_data_error(missing_out, out_dir);
        assert_eq!(err.exit, ExitCode::NotFound);
        assert_eq!(err.code, "path_not_found");
        let hint = err.hint.as_deref().unwrap_or("");
        assert!(hint.contains("output"), "expected output-path hint, got: {hint}");
        assert!(!hint.contains("data.yaml"), "must not emit input-dataset hint: {hint}");

        let vanished_input = DataError::Io {
            path: PathBuf::from("ds/images/train/a.png"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        let err = output_data_error(vanished_input, out_dir);
        assert_eq!(err.exit, ExitCode::NotFound);
        assert_eq!(err.code, "path_not_found");
        assert!(err.message.contains("ds/images/train/a.png"), "{}", err.message);
        assert!(err.hint.as_deref().unwrap_or("").contains("input"), "{:?}", err.hint);
    }

    #[test]
    fn confirmed_coco_annotation_distinguishes_splits_from_root_predictions() {
        let root = Path::new("/data/coco");
        assert!(is_confirmed_coco_annotation(&root.join("annotations/instances_val.json"), root, root));
        assert!(is_confirmed_coco_annotation(&root.join("val/_annotations.coco.json"), root, root));
        assert!(is_confirmed_coco_annotation(&root.join("instances_val.json"), root, root));
        assert!(is_confirmed_coco_annotation(&root.join("_annotations.coco.json"), root, root));
        assert!(is_confirmed_coco_annotation(&root.join("annotations"), root, root));
        assert!(is_confirmed_coco_annotation(&root.join("val"), root, root));
        assert!(!is_confirmed_coco_annotation(&root.join("coco_predictions.json"), root, root));
        assert!(!is_confirmed_coco_annotation(&root.join("extra_annotations.json"), root, root));
    }
}
