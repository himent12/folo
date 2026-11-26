//! `folo dataset check | stats | synth | convert`, on top of folo-data.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use folo_data::check::{CheckKind, CheckOptions, CheckReport, Severity};
use folo_data::synthetic::{SyntheticConfig, shapes_for, write_yolo_with_progress};
use folo_data::{DataError, DatasetStats, Format, Histogram, ImageFormat, LoadedDataset, OpenOptions};
use serde_json::{Value, json};

use crate::cli::{
    CheckArgs, ConvertArgs, DatasetCommand, FormatArg, SeverityArg, StatsArgs, SynthArgs, SynthImageFormat,
    TargetFormatArg,
};
use crate::output::{CliError, ExitCode, Out};

/// Runs a `folo dataset` subcommand.
pub fn run(cmd: DatasetCommand, out: Out) -> Result<(), CliError> {
    match cmd {
        DatasetCommand::Check(a) => check(a, out),
        DatasetCommand::Stats(a) => stats(a, out),
        DatasetCommand::Synth(a) => synth(a, out),
        DatasetCommand::Convert(a) => convert(a, out),
    }
}

pub(crate) fn format_of(f: FormatArg) -> Format {
    match f {
        FormatArg::Auto => Format::Auto,
        FormatArg::Yolo => Format::Yolo,
        FormatArg::Coco => Format::Coco,
    }
}

fn format_name(f: Format) -> &'static str {
    match f {
        Format::Auto => "auto",
        Format::Yolo => "yolo",
        Format::Coco => "coco",
    }
}

/// Maps a folo-data error to the CLI's codes and exit codes.
fn data_error(e: DataError, path: &Path) -> CliError {
    match e {
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => CliError::new(
            ExitCode::NotFound,
            "path_not_found",
            format!("{} does not exist", folo_api::json::path(path)),
        )
        .hint("pass a dataset folder, a data.yaml or a COCO instances .json"),
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
                "expected data.yaml or images/ + labels/ (YOLO), or annotations/instances_*.json (COCO)",
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
fn output_data_error(e: DataError, out_dir: &Path) -> CliError {
    match e {
        DataError::Io { path: err_path, source } if source.kind() == std::io::ErrorKind::NotFound => {
            if err_path.starts_with(out_dir) {
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
fn canonical_target(p: &Path) -> Option<PathBuf> {
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
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(cur.join("..")) {
                        cur = canon;
                    } else {
                        let _ = cur.pop();
                        missing = true;
                    }
                } else {
                    let _ = cur.pop();
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                        missing = false;
                    }
                }
            }
            std::path::Component::Normal(part) => {
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

/// Refuses a `convert` output inside (or containing) the input dataset, its supplied folder, or any of its source
/// folders, so converted images are never written into the input and picked up on its next open.
fn check_convert_output_outside(output: &Path, input: &Path, ds: &LoadedDataset) -> Result<(), CliError> {
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
                "output {out_shown} {rel} the input dataset folder {src_shown}: converting into the input would \
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

/// Refuses an output folder that exists and is not empty (or is a file).
fn ensure_empty_output(dir: &Path) -> Result<(), CliError> {
    let busy = if dir.is_dir() {
        std::fs::read_dir(dir).map(|mut rd| rd.next().is_some()).unwrap_or(true)
    } else {
        dir.exists()
    };
    if busy {
        return Err(output_exists(dir, dir.join(folo_data::unfinished::MARKER).exists()));
    }
    Ok(())
}

/// `output_exists` for `dir`; `unfinished` when it holds another writer's marker.
fn output_exists(dir: &Path, unfinished: bool) -> CliError {
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
fn rel(p: &Path, root: &Path) -> String {
    // The server names issue files the same way (GET /api/datasets/{id}/issues).
    folo_data::paths::relative_to(p, root)
}

fn splits_json(s: [usize; 3]) -> Value {
    json!({ "train": s[0], "val": s[1], "test": s[2] })
}

// ---- stats -------------------------------------------------------------------------------------------------------

fn open(path: &Path, format: Format) -> Result<LoadedDataset, CliError> {
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

// ---- check -------------------------------------------------------------------------------------------------------

fn issue_json(i: &folo_data::check::CheckIssue, root: &Path, with_fix: bool) -> Value {
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

fn check(args: CheckArgs, out: Out) -> Result<(), CliError> {
    let t0 = Instant::now();
    let opts = CheckOptions { format: format_of(args.format), decode: !args.headers_only };
    let report = folo_data::check::check(&args.path, &opts).map_err(|e| data_error(e, &args.path))?;
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

// ---- synth -------------------------------------------------------------------------------------------------------

/// Throttled progress output: `--json` records, or a `\r` line on an interactive stderr.
struct Progress {
    out: Out,
    total: usize,
    tty: bool,
    last: Mutex<Option<Instant>>,
}

impl Progress {
    fn update(&self, done: usize) {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let due = last.is_none_or(|t| now.duration_since(t) >= Duration::from_millis(200));
        if !(due || done == self.total) {
            return;
        }
        *last = Some(now);
        self.out.record(&json!({ "event": "progress", "done": done, "total": self.total }));
        if !self.out.json && self.tty {
            let pct = 100.0 * done as f64 / self.total.max(1) as f64;
            let mut e = std::io::stderr().lock();
            let _ = write!(e, "\rgenerating {done}/{} images ({pct:.0}%)", self.total);
            if done == self.total {
                let _ = write!(e, "\r{:60}\r", "");
            }
            let _ = e.flush();
        }
    }
}

fn synth(args: SynthArgs, out: Out) -> Result<(), CliError> {
    if !crate::cli::bounds::SYNTH_IMAGES.contains_u64(args.images) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--images must be in 1..=1000000"));
    }
    if !crate::cli::bounds::SYNTH_IMGSZ.contains_u64(u64::from(args.imgsz)) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--imgsz must be in 32..=8192"));
    }
    let names: Vec<String> = args.classes.iter().map(|c| c.trim().to_string()).collect();
    let classes = shapes_for(&names).map_err(|e| data_error(e, &args.out_dir))?;
    let cfg = SyntheticConfig {
        images: args.images as usize,
        classes: classes.clone(),
        names: names.clone(),
        imgsz: args.imgsz,
        seed: args.seed,
        val_fraction: args.val_frac,
        vary_aspect: args.vary_aspect,
        ..Default::default()
    };
    ensure_empty_output(&args.out_dir)?;
    cfg.check().map_err(|e| data_error(e, &args.out_dir))?;
    let format = match args.image_format {
        SynthImageFormat::Jpeg => ImageFormat::Jpeg(90),
        SynthImageFormat::Png => ImageFormat::Png,
    };
    let t0 = Instant::now();
    let progress =
        Progress { out, total: cfg.images, tty: std::io::stderr().is_terminal(), last: Mutex::new(None) };
    let claim =
        folo_data::unfinished::claim(&args.out_dir).map_err(|e| output_data_error(e, &args.out_dir))?;
    let written = write_yolo_with_progress(&cfg, claim, format, |d| progress.update(d))
        .map_err(|e| output_data_error(e, &args.out_dir))?;
    let secs = t0.elapsed().as_secs_f64();
    let [train, val] = written.splits;
    out.record(&json!({
        "event": "done",
        "path": folo_api::json::path(&written.root),
        "yaml": folo_api::json::path(&written.yaml),
        "images": cfg.images,
        "splits": splits_json([train, val, 0]),
        "boxes": written.boxes,
        "classes": names,
        "shapes": classes,
        "imgsz": cfg.imgsz,
        "seed": cfg.seed,
        "seconds": secs,
    }));
    out.human(format!(
        "wrote {} images ({train} train / {val} val, {} boxes) to {} in {secs:.1}s",
        cfg.images,
        written.boxes,
        folo_api::json::path(&written.root)
    ));
    let mapping: Vec<String> = names
        .iter()
        .zip(&classes)
        .map(|(n, s)| if n.eq_ignore_ascii_case(s) { n.clone() } else { format!("{n} ({s})") })
        .collect();
    out.human(format!("  classes  {}", mapping.join(", ")));
    out.human(format!("  yaml     {}", folo_api::json::path(&written.yaml)));
    out.human(folo_api::json::cli_command(
        "next: folo dataset stats {dataset}",
        &[("dataset", &args.out_dir)],
    ));
    Ok(())
}

// ---- convert -----------------------------------------------------------------------------------------------------

fn convert(args: ConvertArgs, out: Out) -> Result<(), CliError> {
    let to = match args.to {
        TargetFormatArg::Yolo => Format::Yolo,
        TargetFormatArg::Coco => Format::Coco,
    };
    let ds = open(&args.input, format_of(args.from))?;
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
    fn severity_filter_orders_error_warning_info() {
        assert_eq!(severity_arg_name(SeverityArg::Error), "error");
        assert_eq!(severity_arg_name(SeverityArg::Warning), "warning");
        assert_eq!(severity_arg_name(SeverityArg::Info), "info");

        // Error threshold: only errors pass.
        assert!(severity_allows(SeverityArg::Error, Severity::Error));
        assert!(!severity_allows(SeverityArg::Error, Severity::Warning));
        assert!(!severity_allows(SeverityArg::Error, Severity::Info));

        // Warning threshold: errors and warnings pass; info is filtered.
        assert!(severity_allows(SeverityArg::Warning, Severity::Error));
        assert!(severity_allows(SeverityArg::Warning, Severity::Warning));
        assert!(!severity_allows(SeverityArg::Warning, Severity::Info));

        // Info threshold: every severity passes.
        assert!(severity_allows(SeverityArg::Info, Severity::Error));
        assert!(severity_allows(SeverityArg::Info, Severity::Warning));
        assert!(severity_allows(SeverityArg::Info, Severity::Info));
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
}
