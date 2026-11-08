//! `folo dataset check | stats | synth | convert`, on top of folo-data.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use folo_data::check::{CheckKind, CheckOptions, CheckReport, Severity};
use folo_data::synthetic::{SyntheticConfig, shapes_for, write_yolo_with_progress};
use folo_data::{DataError, DatasetStats, Format, Histogram, ImageFormat, LoadedDataset, OpenOptions};
use serde_json::{Value, json};

use crate::cli::{
    CheckArgs, ConvertArgs, DatasetCommand, FormatArg, StatsArgs, SynthArgs, SynthImageFormat,
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

fn format_of(f: FormatArg) -> Format {
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
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            CliError::new(ExitCode::NotFound, "path_not_found", format!("{} does not exist", path.display()))
                .hint("pass a dataset folder, a data.yaml or a COCO instances .json")
        }
        DataError::Format(m) if m.starts_with(folo_data::UNDETECTED) => {
            CliError::new(ExitCode::Runtime, "unknown_format", m).hint(
                "expected data.yaml or images/ + labels/ (YOLO), or annotations/instances_*.json (COCO)",
            )
        }
        DataError::Format(m) => CliError::new(ExitCode::Runtime, "invalid_dataset", m),
        DataError::Config(m) => {
            CliError::new(ExitCode::Usage, "usage", m).hint("see `folo dataset <command> --help`")
        }
        e => CliError::new(ExitCode::Runtime, "io", e.to_string()),
    }
}

/// Refuses an output folder that exists and is not empty (or is a file).
fn ensure_empty_output(dir: &Path) -> Result<(), CliError> {
    let busy = if dir.is_dir() {
        std::fs::read_dir(dir).map(|mut rd| rd.next().is_some()).unwrap_or(true)
    } else {
        dir.exists()
    };
    if busy {
        return Err(CliError::new(
            ExitCode::Runtime,
            "output_exists",
            format!("{} exists and is not an empty folder", dir.display()),
        )
        .hint("pick a new folder, or delete this one first"));
    }
    Ok(())
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

fn dataset_name(path: &Path) -> String {
    let p = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let p = if p.is_file() { p.parent().map(Path::to_path_buf).unwrap_or(p) } else { p };
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "dataset".into())
}

fn stats(args: StatsArgs, out: Out) -> Result<(), CliError> {
    let ds = open(&args.path, format_of(args.format))?;
    let stats = ds.stats();
    let skipped = ds.report.skipped_images();
    out.record(&json!({
        "name": dataset_name(&args.path),
        "path": args.path,
        "root": ds.root,
        "format": format_name(ds.format),
        "classes": ds.classes,
        "splits": splits_json(ds.split_sizes()),
        "skipped_images": skipped,
        "stats": stats,
    }));
    for line in stats_table(&dataset_name(&args.path), &ds, &stats) {
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

fn stats_table(name: &str, ds: &LoadedDataset, s: &DatasetStats) -> Vec<String> {
    let [train, val, test] = ds.split_sizes();
    let images = train + val + test;
    let mut l = vec![
        format!("{name}  ({}, {})", format_name(ds.format), ds.root.display()),
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
    let width = ds.classes.iter().map(|c| c.chars().count()).max().unwrap_or(5).clamp(5, 24);
    l.push(format!("  {:<width$} {:>8}  {:>6}", "class", "boxes", "share"));
    const SHOWN: usize = 40;
    let mut order: Vec<usize> = (0..s.class_counts.len()).collect();
    if order.len() > SHOWN {
        order.sort_by_key(|&i| std::cmp::Reverse(s.class_counts[i]));
    }
    for &i in order.iter().take(SHOWN) {
        let c = s.class_counts[i];
        let name = ds.classes.get(i).cloned().unwrap_or_else(|| format!("class{i}"));
        let name: String = name.chars().take(width).collect();
        l.push(format!(
            "  {name:<width$} {c:>8}  {:>5.1}%  {}",
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
    let by_kind: serde_json::Map<String, Value> =
        report.counts().into_iter().map(|(k, n)| (k.as_str().to_string(), json!(n))).collect();
    out.record(&json!({
        "path": args.path,
        "root": report.root,
        "format": format_name(report.format),
        "classes": report.classes,
        "images": report.images(),
        "splits": splits_json(report.splits),
        "clean": report.is_clean(),
        "counts": { "error": errors, "warning": warnings, "info": infos, "by_kind": by_kind },
        "issues": report.issues.iter().map(|i| issue_json(i, &report.root, args.fix_dry_run)).collect::<Vec<_>>(),
        "seconds": secs,
    }));
    for line in check_lines(&report, args.fix_dry_run, secs) {
        out.human(line);
    }
    if report.is_clean() {
        return Ok(());
    }
    Err(CliError::new(
        ExitCode::Runtime,
        "dataset_issues",
        format!("{} has {errors} error(s) and {warnings} warning(s)", args.path.display()),
    )
    .hint(if args.fix_dry_run {
        "nothing was changed; the `fix` of each issue says what a cleanup would do"
    } else {
        "see the issues above; --fix-dry-run shows what a cleanup would change"
    }))
}

fn check_lines(r: &CheckReport, fixes: bool, secs: f64) -> Vec<String> {
    const SHOWN: usize = 200;
    let mut l = vec![format!(
        "checked {} images ({}, {}) in {secs:.1}s",
        r.images(),
        format_name(r.format),
        r.root.display()
    )];
    let problems: Vec<_> = r.issues.iter().filter(|i| i.severity() != Severity::Info).collect();
    if !problems.is_empty() {
        l.push(String::new());
    }
    for i in problems.iter().take(SHOWN) {
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
    if problems.len() > SHOWN {
        l.push(format!("... {} more (--json lists every issue)", problems.len() - SHOWN));
    }
    let empty = r.counts().get(&CheckKind::EmptyImage).copied().unwrap_or(0);
    l.push(String::new());
    let summary: Vec<String> = r
        .counts()
        .into_iter()
        .filter(|(k, _)| *k != CheckKind::EmptyImage)
        .map(|(k, n)| format!("{n} {}", k.as_str()))
        .collect();
    if r.is_clean() {
        l.push("clean: no errors or warnings".into());
    } else {
        l.push(format!(
            "{} error(s), {} warning(s): {}",
            r.count(Severity::Error),
            r.count(Severity::Warning),
            summary.join(", ")
        ));
    }
    if empty > 0 {
        l.push(format!("{empty} empty image(s) without boxes (background images; fine unless unintended)"));
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
    if !(1..=1_000_000).contains(&args.images) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--images must be in 1..=1000000"));
    }
    if !(32..=8192).contains(&args.imgsz) {
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
    let format = match args.image_format {
        SynthImageFormat::Jpeg => ImageFormat::Jpeg(90),
        SynthImageFormat::Png => ImageFormat::Png,
    };
    let t0 = Instant::now();
    let progress =
        Progress { out, total: cfg.images, tty: std::io::stderr().is_terminal(), last: Mutex::new(None) };
    let written = write_yolo_with_progress(&cfg, &args.out_dir, format, |d| progress.update(d))
        .map_err(|e| data_error(e, &args.out_dir))?;
    let secs = t0.elapsed().as_secs_f64();
    let [train, val] = written.splits;
    out.record(&json!({
        "event": "done",
        "path": written.root,
        "yaml": written.yaml,
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
        written.root.display()
    ));
    let mapping: Vec<String> = names
        .iter()
        .zip(&classes)
        .map(|(n, s)| if n.eq_ignore_ascii_case(s) { n.clone() } else { format!("{n} ({s})") })
        .collect();
    out.human(format!("  classes  {}", mapping.join(", ")));
    out.human(format!("  yaml     {}", written.yaml.display()));
    out.human(format!("next: folo dataset stats {}", args.out_dir.display()));
    Ok(())
}

// ---- convert -----------------------------------------------------------------------------------------------------

fn convert(args: ConvertArgs, out: Out) -> Result<(), CliError> {
    let to = match args.to {
        TargetFormatArg::Yolo => Format::Yolo,
        TargetFormatArg::Coco => Format::Coco,
    };
    ensure_empty_output(&args.output)?;
    let ds = open(&args.input, format_of(args.from))?;
    let skipped = ds.report.skipped_images() + ds.report.broken_labels();
    let t0 = Instant::now();
    let done = folo_data::convert::convert(&ds, &args.output, to).map_err(|e| match e {
        // The input cannot be written in that format (no images, or only test images for YOLO): not a usage error.
        DataError::Config(m) => CliError::new(ExitCode::Runtime, "invalid_dataset", m),
        e => data_error(e, &args.output),
    })?;
    let secs = t0.elapsed().as_secs_f64();
    out.record(&json!({
        "input": args.input,
        "output": done.root,
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
        args.input.display(),
        format_name(to),
        done.root.display(),
        done.boxes
    ));
    if skipped > 0 {
        out.human(format!(
            "  {skipped} image(s) were skipped (unreadable or broken labels); run `folo dataset check {}`",
            args.input.display()
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
}
