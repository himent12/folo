//! `folo train`: a run under `<project>/<name>/` with a live progress view (or JSON lines with `--json`).

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use folo_api::{RunEvent, RunStatus};
use folo_data::{AugmentConfig, Dataset, Split};
use folo_vision::train::{OptimizerChoice, ResumeOverrides, TrainSummary, event_record};
use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};
use serde_json::json;

use crate::cli::{JsonEvents, OptimizerArg, TrainArgs};
use crate::engine::{engine_error, open_dataset};
use crate::output::{CliError, ExitCode, Out};

/// `folo stop <run_dir>` creates this file in the run folder; `folo train` polls for it and stops after the
/// current batch.
pub const STOP_FILE: &str = "stop";

pub fn train(args: TrainArgs, out: Out) -> Result<(), CliError> {
    if args.hsv.len() != 3 {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--hsv takes three gains h,s,v (e.g. 0.015,0.7,0.4), got {}", args.hsv.len()),
        ));
    }
    // Mixed precision needs tensor cores, which the GPU device must be created with: ask before anything creates
    // the shared context (also when resuming a mixed-precision run without repeating --amp).
    if args.amp || args.resume.as_deref().is_some_and(checkpoint_uses_amp) {
        folo_gpu::context::request_tensor_cores();
    }
    crate::engine::init_device(&args.device);
    let stop = Arc::new(AtomicBool::new(false));
    watch_ctrl_c(stop.clone());
    let nb = Arc::new(AtomicU64::new(0));
    let fresh = args.resume.is_none();
    let (mut trainer, run_dir) = match &args.resume {
        Some(ckpt) => resume(&args, ckpt, out, stop.clone(), nb.clone())?,
        None => start(&args, out, stop.clone(), nb.clone())?,
    };
    watch_stop_file(&run_dir, stop);
    nb.store(trainer.batches_per_epoch() as u64, Ordering::Relaxed);
    if !out.json {
        out.human(format!("Run folder: {}", folo_api::json::path(&run_dir)));
    }
    let summary = match trainer.run() {
        Ok(s) => s,
        Err(e) => {
            drop(trainer);
            if fresh {
                discard_if_empty(&run_dir, out);
            }
            return Err(engine_error(e));
        }
    };
    finish(&summary, &run_dir, trainer.config(), &trainer.resolved_train(), out);
    Ok(())
}

/// Whether the training checkpoint at `ckpt` belongs to a mixed-precision run (`folo.train` has `amp: true`);
/// `false` for anything unreadable (the resume then reports the problem).
fn checkpoint_uses_amp(ckpt: &Path) -> bool {
    let Ok(file) = folo_nn::io::ModelFile::open(ckpt) else { return false };
    file.checkpoint().is_some_and(|c| c.train.get("amp").and_then(|v| v.as_bool()) == Some(true))
}

/// Removes the folder of a new run that failed before its first checkpoint, so nothing half-made is left behind
/// (the error says what happened).
fn discard_if_empty(run_dir: &Path, out: Out) {
    if !run_dir.join("weights").join("last.folo").exists() && std::fs::remove_dir_all(run_dir).is_ok() {
        out.human(format!("removed {} (no epoch finished)", folo_api::json::path(run_dir)));
    }
}

/// Polls `<run_dir>/stop` (written by `folo stop`) every 200 ms and raises `stop` when it appears. A stop file
/// left from an earlier process is removed first.
fn watch_stop_file(run_dir: &Path, stop: Arc<AtomicBool>) {
    let file = run_dir.join(STOP_FILE);
    let _ = std::fs::remove_file(&file);
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            if file.exists() {
                let _ = std::fs::remove_file(&file);
                eprintln!(
                    "stop requested ({}); stopping after the current batch…",
                    folo_api::json::path(&file)
                );
                stop.store(true, Ordering::Relaxed);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    });
}

/// `folo stop <run_dir>`: asks the `folo train` writing that run folder to stop after its current batch.
pub fn request_stop(run_dir: &Path, out: Out) -> Result<(), CliError> {
    let not_a_run = |code: &str, exit, msg: String| {
        CliError::new(exit, code, msg).hint("pass the run_dir that `folo train` printed, e.g. runs/shapes-n")
    };
    if !run_dir.is_dir() {
        let msg = format!("{} does not exist", folo_api::json::path(run_dir));
        return Err(not_a_run("path_not_found", ExitCode::NotFound, msg));
    }
    if !run_dir.join("config.json").is_file() && !run_dir.join("weights").is_dir() {
        let msg = format!("{} is not a run folder (it has no config.json)", folo_api::json::path(run_dir));
        return Err(not_a_run("run_not_found", ExitCode::NotFound, msg));
    }
    let file = run_dir.join(STOP_FILE);
    std::fs::write(&file, b"stop\n").map_err(|e| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot write {}: {e}", folo_api::json::path(&file)),
        )
    })?;
    out.record(&json!({
        "event": "stop_requested",
        "run_dir": folo_api::json::path(run_dir),
        "stop_file": folo_api::json::path(&file),
    }));
    out.human(format!(
        "Asked the run in {} to stop after its current batch; its `folo train` ends with status `stopped`.",
        folo_api::json::path(run_dir)
    ));
    Ok(())
}

fn start(
    args: &TrainArgs,
    out: Out,
    stop: Arc<AtomicBool>,
    nb: Arc<AtomicU64>,
) -> Result<(Trainer, PathBuf), CliError> {
    let data_arg = args.data.as_deref().expect("clap requires --data without --resume");
    let (ds, ds_name) = open_dataset(data_arg, crate::dataset::format_of(args.format))?;
    let mut weights = None;
    let mut weights_head = folo_vision::model::HeadMode::O2m;
    // Initial weights: --weights, or --model naming a .folo file or published weights rather than an arch.
    let init = match &args.weights {
        Some(w) => Some(w.as_str()),
        None if args.model.ends_with(".folo") || crate::weights::is_published(&args.model) => {
            Some(args.model.as_str())
        }
        None => None,
    };
    let arch = if let Some(init) = init {
        let p = crate::weights::resolve(Path::new(init))?;
        let file = folo_nn::io::ModelFile::open(&p).map_err(|e| engine_error(e.into()))?;
        weights = Some(p);
        if file.metadata().config.has_o2o() {
            weights_head = folo_vision::model::HeadMode::O2o;
        }
        file.metadata().arch.clone()
    } else {
        args.model.clone()
    };
    let suffix = arch.strip_prefix("folo-").unwrap_or(&arch).to_string();
    let name = args.name.clone().unwrap_or_else(|| format!("{ds_name}-{suffix}"));
    let mut cfg = config_of(args, name.clone(), absolute(data_arg), arch, weights_head);
    // Validate before creating the run folder, so a typo leaves nothing behind.
    cfg.validate().map_err(engine_error)?;
    let data = train_data(&ds)?;
    let run_dir = reserve_dir(&args.project, &name).map_err(|e| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot create a run folder under {}: {e}", folo_api::json::path(&args.project)),
        )
    })?;
    cfg.name = run_dir.file_name().map_or(name, |n| n.to_string_lossy().into_owned());
    let opts = TrainOptions {
        run_dir: Some(run_dir.clone()),
        events: Some(progress_sink(out, args.json_events, cfg.epochs, nb)),
        stop,
        weights,
        ..TrainOptions::default()
    };
    match Trainer::new(cfg, data, opts) {
        Ok(t) => Ok((t, run_dir)),
        Err(e) => {
            discard_if_empty(&run_dir, out);
            Err(engine_error(e))
        }
    }
}

/// The trainer's config of `folo train` with `args`, the run `name`, the `dataset` path, the architecture `arch`
/// (the weights' when fine-tuning) and the head a run without `--head` gets (the weights file's, else o2m).
fn config_of(
    args: &TrainArgs,
    name: String,
    dataset: String,
    arch: String,
    default_head: folo_vision::model::HeadMode,
) -> TrainConfig {
    TrainConfig {
        name,
        dataset,
        data_format: crate::dataset::format_of(args.format),
        arch,
        epochs: args.epochs,
        batch: args.batch as usize,
        imgsz: args.imgsz,
        device: args.device.clone(),
        optimizer: match args.optimizer {
            OptimizerArg::Auto => OptimizerChoice::Auto,
            OptimizerArg::Sgd => OptimizerChoice::Sgd,
            OptimizerArg::Adamw => OptimizerChoice::Adamw,
        },
        lr0: folo_vision::train::f32_decimal(args.lr0),
        lrf: args.lrf,
        cos_lr: args.cos_lr,
        amp: args.amp,
        head: args.head.map_or(default_head, Into::into),
        warmup_epochs: args.warmup_epochs,
        patience: args.patience,
        seed: args.seed,
        workers: args.workers.unwrap_or_else(folo_vision::train::default_workers),
        augment: AugmentConfig {
            mosaic: args.mosaic,
            mixup: args.mixup,
            hsv: [args.hsv[0], args.hsv[1], args.hsv[2]],
            fliplr: args.fliplr,
            scale: args.scale,
            translate: args.translate,
            close_mosaic: args.close_mosaic,
            ..AugmentConfig::default()
        },
        ..TrainConfig::default()
    }
}

fn resume(
    args: &TrainArgs,
    ckpt: &Path,
    out: Out,
    stop: Arc<AtomicBool>,
    nb: Arc<AtomicU64>,
) -> Result<(Trainer, PathBuf), CliError> {
    if !ckpt.is_file() {
        return Err(CliError::new(
            ExitCode::NotFound,
            "path_not_found",
            format!("{} does not exist", folo_api::json::path(ckpt)),
        ));
    }
    let file = folo_nn::io::ModelFile::open(ckpt).map_err(|e| engine_error(e.into()))?;
    let Some(c) = file.checkpoint() else {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("{} is a weights file, not a training checkpoint", folo_api::json::path(ckpt)),
        )
        .hint("resume from runs/<name>/weights/last.folo"));
    };
    let stored: TrainConfig = serde_json::from_value(c.train.clone()).map_err(|e| {
        CliError::new(ExitCode::Runtime, "invalid_weights", format!("folo.train is not a TrainConfig: {e}"))
    })?;
    // As MCP and `POST /api/runs`: a completed run is not resumed, and other flags must match the run.
    let early_stopped =
        c.best.fitness.is_some() && (c.epoch as i64 - 1) - c.best.epoch >= stored.patience as i64;
    if c.epoch >= stored.epochs as u64 || early_stopped {
        let best = ckpt.with_file_name("best.folo");
        return Err(CliError::new(
            ExitCode::Runtime,
            "run_not_resumable",
            format!(
                "the run of {} already completed ({} of {} epochs{})",
                folo_api::json::path(ckpt),
                c.epoch,
                stored.epochs,
                if early_stopped { ", stopped early" } else { "" }
            ),
        )
        .hint(format!(
            "to train further, start a new run from its weights: folo train --model {} --data {} --epochs <n>",
            folo_api::json::path(if best.is_file() { &best } else { ckpt }),
            if stored.dataset.is_empty() { "<dataset>" } else { &stored.dataset }
        )));
    }
    check_resume_flags(args, &stored)?;
    // weights/last.folo lives in <run>/weights/.
    let run_dir = ckpt
        .parent()
        .filter(|p| p.file_name().is_some_and(|n| n == "weights"))
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| args.project.join(&stored.name));
    let data_arg = match &args.data {
        Some(d) => d.clone(),
        None if !stored.dataset.is_empty() => stored.dataset.clone(),
        None => {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                "the checkpoint does not say which dataset the run used",
            )
            .hint("pass --data with the dataset the run was started on"));
        }
    };
    // The format the run was started with, unless --format says otherwise (as --data may).
    let format = if args.explicit.iter().any(|f| f == "format") {
        crate::dataset::format_of(args.format)
    } else {
        stored.data_format
    };
    let (ds, _) = open_dataset(&data_arg, format)?;
    let data = train_data(&ds)?;
    let epochs = stored.epochs;
    let opts = TrainOptions {
        run_dir: Some(run_dir.clone()),
        events: Some(progress_sink(out, args.json_events, epochs, nb)),
        stop,
        ..TrainOptions::default()
    };
    let overrides = ResumeOverrides {
        workers: args.workers,
        device: Some(args.device.clone()),
        data_format: Some(format),
    };
    Ok((Trainer::resume(ckpt, data, opts, overrides).map_err(engine_error)?, run_dir))
}

/// Flags a resume may change: where and how fast it runs, where its data is, and how it reports.
const RESUME_MAY_CHANGE: &[&str] =
    &["resume", "data", "format", "workers", "device", "json_events", "project"];

/// With `--resume`, every other flag given on the command line must repeat the run's stored setting (as MCP
/// and `POST /api/runs` require): a changed one is `invalid_config`, naming the flag and both values.
fn check_resume_flags(args: &TrainArgs, stored: &TrainConfig) -> Result<(), CliError> {
    let have = serde_json::to_value(stored).expect("configs serialize");
    for id in args.explicit.iter().filter(|id| !RESUME_MAY_CHANGE.contains(&id.as_str())) {
        let field = |k: &str| have[k].clone();
        let aug = |k: &str| have["augment"][k].clone();
        let (stored_value, given) = match id.as_str() {
            "model" => (field("arch"), json!(args.model)),
            "name" => (field("name"), json!(args.name)),
            "epochs" => (field("epochs"), json!(args.epochs)),
            "batch" => (field("batch"), json!(args.batch)),
            "imgsz" => (field("imgsz"), json!(args.imgsz)),
            "optimizer" => (field("optimizer"), json!(format!("{:?}", args.optimizer).to_lowercase())),
            "lr0" => (field("lr0"), json!(folo_vision::train::f32_decimal(args.lr0))),
            "lrf" => (field("lrf"), json!(args.lrf)),
            "cos_lr" => (field("cos_lr"), json!(args.cos_lr)),
            "amp" => (field("amp"), json!(args.amp)),
            "head" => (field("head"), json!(args.head.map(folo_vision::model::HeadMode::from))),
            "warmup_epochs" => (field("warmup_epochs"), json!(args.warmup_epochs)),
            "patience" => (field("patience"), json!(args.patience)),
            "seed" => (field("seed"), json!(args.seed)),
            "mosaic" => (aug("mosaic"), json!(args.mosaic)),
            "mixup" => (aug("mixup"), json!(args.mixup)),
            "fliplr" => (aug("fliplr"), json!(args.fliplr)),
            "scale" => (aug("scale"), json!(args.scale)),
            "translate" => (aug("translate"), json!(args.translate)),
            "close_mosaic" => (aug("close_mosaic"), json!(args.close_mosaic)),
            "hsv" => (aug("hsv"), json!(args.hsv)),
            _ => continue,
        };
        // `--lrf` and `--warmup-epochs` are f64 settings: compared exactly; the rest are integers or f32s.
        let f32_setting = !matches!(id.as_str(), "lrf" | "warmup_epochs");
        if !same(&stored_value, &given, f32_setting) {
            let flag = format!("--{}", id.replace('_', "-"));
            return Err(CliError::new(
                ExitCode::Usage,
                "invalid_config",
                format!(
                    "{flag} {} differs from the run's {} (a resumed run keeps its settings)",
                    show(&given),
                    show(&stored_value)
                ),
            )
            .hint(format!(
                "leave {flag} out to resume the run as it was started (only --workers, --device, --data, \
                 --format, --project and --json-events may change), or start a new run from its weights with --model \
                 <run>/weights/best.folo"
            )));
        }
    }
    Ok(())
}

/// Equal JSON values: integers exactly, floats exactly or, for `f32` settings (stored widened to f64), as the
/// same f32.
fn same(a: &serde_json::Value, b: &serde_json::Value, f32_setting: bool) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Number(x), Value::Number(y)) if !x.is_f64() && !y.is_f64() => x == y,
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
            x == y || (f32_setting && x as f32 == y as f32)
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y, f32_setting))
        }
        _ => a == b,
    }
}

/// A setting as an error message shows it.
fn show(v: &serde_json::Value) -> String {
    let mut v = v.clone();
    folo_api::json::clean_floats(&mut v);
    match v {
        serde_json::Value::String(s) => s,
        serde_json::Value::Null => "(none)".into(),
        other => other.to_string(),
    }
}

/// `p` made absolute (so a resume from another folder finds the dataset).
fn absolute(p: &str) -> String {
    std::path::absolute(p).map_or_else(|_| p.to_string(), folo_api::json::path)
}

fn train_data(ds: &folo_data::LoadedDataset) -> Result<TrainData, CliError> {
    if ds.train.is_empty() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            "the dataset has no training images",
        )
        .hint("YOLO datasets need images/train (or a `train:` entry in data.yaml)"));
    }
    Ok(TrainData {
        train: Arc::new(ds.split(Split::Train).clone()),
        val: Arc::new(ds.split(Split::Val).clone()),
        classes: ds.classes.clone(),
    })
}

/// Creates `<project>/<name>`, or `<name>2`, `<name>3`, … if it exists, and returns it. `create_dir` claims the
/// folder atomically, so concurrent `folo train`s never share one (and a failed run only removes its own).
fn reserve_dir(project: &Path, name: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(project)?;
    for i in 1.. {
        let dir = if i == 1 { project.join(name) } else { project.join(format!("{name}{i}")) };
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    unreachable!("some suffix is free")
}

/// First Ctrl+C (on Windows also Ctrl+Break, which is what a script can send to a child started in its own
/// process group): stop after the current batch (the last epoch's checkpoint is kept). A second one exits.
fn watch_ctrl_c(stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return };
        rt.block_on(async {
            #[cfg(windows)]
            let Ok(mut brk) = tokio::signal::windows::ctrl_break() else { return };
            for first in [true, false] {
                #[cfg(windows)]
                let got = tokio::select! {
                    r = tokio::signal::ctrl_c() => r.is_ok(),
                    r = brk.recv() => r.is_some(),
                };
                #[cfg(not(windows))]
                let got = tokio::signal::ctrl_c().await.is_ok();
                if !got {
                    return;
                }
                if !first {
                    std::process::exit(130);
                }
                eprintln!("\nstopping after the current batch (interrupt again to abort)…");
                stop.store(true, Ordering::Relaxed);
            }
        });
    });
}

/// The event consumer: JSON lines with `--json` (plus one-line epoch summaries on stderr), otherwise a live view.
/// `nb` (batches per epoch) is filled in once the trainer exists.
fn progress_sink(
    out: Out,
    records: JsonEvents,
    epochs: u32,
    nb: Arc<AtomicU64>,
) -> folo_vision::train::EventSink {
    let tty = std::io::stdout().is_terminal();
    let mut view = Progress { out, records, epochs, nb, tty, header: false, live: false };
    Box::new(move |e: &RunEvent| view.event(e))
}

struct Progress {
    out: Out,
    /// `--json-events`.
    records: JsonEvents,
    epochs: u32,
    nb: Arc<AtomicU64>,
    tty: bool,
    header: bool,
    /// A `\r` progress line is on screen.
    live: bool,
}

const BAR: usize = 24;

impl Progress {
    fn event(&mut self, e: &RunEvent) {
        if self.out.json {
            if !(self.records == JsonEvents::Epoch && matches!(e, RunEvent::Step(_))) {
                self.out.record(&event_record(e));
            }
            if let RunEvent::Epoch(ep) = e {
                eprintln!(
                    "epoch {}/{}  loss {:.3}  mAP50 {:.3}  mAP50-95 {:.3}  {:.1}s",
                    ep.epoch, self.epochs, ep.train_loss.total, ep.val.map50, ep.val.map50_95, ep.duration_s
                );
            }
            return;
        }
        match e {
            RunEvent::Step(s) => {
                let nb = self.nb.load(Ordering::Relaxed).max(1);
                let k = s.step - (s.epoch as u64 - 1) * nb;
                if !self.tty && k != nb {
                    return;
                }
                let filled = ((k as f64 / nb.max(1) as f64) * BAR as f64).round() as usize;
                let line = format!(
                    "epoch {:>3}/{} [{}{}] {:>4}/{nb}  box {:.3}  cls {:.3}  dfl {:.3}  {:>6.1} img/s  ETA {}",
                    s.epoch,
                    self.epochs,
                    "#".repeat(filled.min(BAR)),
                    "-".repeat(BAR - filled.min(BAR)),
                    k,
                    s.loss.box_,
                    s.loss.cls,
                    s.loss.dfl,
                    s.imgs_per_s,
                    duration(s.eta_s)
                );
                let mut stdout = std::io::stdout().lock();
                if self.tty {
                    let _ = write!(stdout, "\r\x1b[2K{line}");
                    self.live = true;
                } else {
                    let _ = writeln!(stdout, "{line}");
                }
                let _ = stdout.flush();
            }
            RunEvent::Epoch(ep) => {
                self.end_line();
                if !self.header {
                    println!(
                        "{:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>7}",
                        "epoch", "box", "cls", "dfl", "P", "R", "mAP50", "mAP50-95", "time"
                    );
                    self.header = true;
                }
                let l = &ep.train_loss;
                let v = &ep.val;
                println!(
                    "{:>7} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>9.3} {:>6.1}s",
                    format!("{}/{}", ep.epoch, self.epochs),
                    l.box_,
                    l.cls,
                    l.dfl,
                    v.precision,
                    v.recall,
                    v.map50,
                    v.map50_95,
                    ep.duration_s
                );
            }
            RunEvent::Log(l) => {
                self.end_line();
                let tag = match l.level {
                    folo_api::LogLevel::Warn => "warning: ",
                    folo_api::LogLevel::Error => "error: ",
                    _ => "",
                };
                println!("{tag}{}", l.message);
            }
            // Queue and cloud events come from a server's runs, never from this trainer.
            RunEvent::Status(_) | RunEvent::Queue(_) | RunEvent::Cloud(_) => {}
        }
    }

    fn end_line(&mut self) {
        if self.live {
            println!();
            self.live = false;
        }
    }
}

fn duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s / 60) % 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

fn finish(s: &TrainSummary, run_dir: &Path, cfg: &TrainConfig, resolved: &folo_api::ResolvedTrain, out: Out) {
    let path = |p: &Option<PathBuf>| p.as_ref().map(folo_api::json::path);
    out.record(&json!({
        "event": "done",
        "status": s.status,
        "run_dir": folo_api::json::path(run_dir),
        "epochs_completed": s.epochs_completed,
        "epochs": cfg.epochs,
        "early_stopped": s.early_stopped,
        "best": s.best.map(|(epoch, fitness)| json!({ "epoch": epoch, "map50_95": fitness })),
        "best_metrics": s.best_metrics,
        "weights": { "best": path(&s.best_weights), "last": path(&s.last) },
        "resolved": resolved,
    }));
    let verb = match s.status {
        RunStatus::Stopped => "Stopped",
        _ if s.early_stopped => "Stopped early",
        _ => "Finished",
    };
    out.human(format!("{verb} after {} of {} epochs.", s.epochs_completed, cfg.epochs));
    if let (Some((epoch, _)), Some(m)) = (s.best, &s.best_metrics) {
        out.human(format!(
            "Best epoch {epoch}: mAP50 {:.3}  mAP50-95 {:.3}  P {:.3}  R {:.3}{}",
            m.map50,
            m.map50_95,
            m.precision,
            m.recall,
            m.best_f1_conf.map_or(String::new(), |c| format!(" (at conf {c:.3})"))
        ));
        if !m.per_class.is_empty() {
            out.human(format!("  {:<16} {:>7} {:>9} {:>7} {:>7}", "class", "mAP50", "mAP50-95", "P", "R"));
            let names = read_classes(run_dir);
            for c in &m.per_class {
                let name = names.get(c.class as usize).cloned().unwrap_or_else(|| c.class.to_string());
                out.human(format!(
                    "  {:<16} {:>7.3} {:>9.3} {:>7.3} {:>7.3}",
                    name, c.map50, c.map50_95, c.precision, c.recall
                ));
            }
        }
    }
    if let Some(b) = &s.best_weights {
        out.human(format!("Weights: {} (and last.folo)", folo_api::json::path(b)));
    }
}

fn read_classes(run_dir: &Path) -> Vec<String> {
    std::fs::read_to_string(run_dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| serde_json::from_value(v["classes"].clone()).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::same;
    use folo_test_support::TestDir;
    use serde_json::json;

    #[test]
    fn resume_settings_compare_integers_exactly() {
        let f32s = |a, b| same(&a, &b, true);
        assert!(!f32s(json!(20_261_108u64), json!(20_261_109u64)));
        assert!(!f32s(json!(u64::MAX), json!(u64::MAX - 1)));
        assert!(f32s(json!(16), json!(16u32)));
        // f32 settings widened to f64 still match what the flag parses to.
        assert!(f32s(json!(0.015f32 as f64), json!(0.015f32)));
        assert!(f32s(json!([0.015f32 as f64, 0.7]), json!([0.015f32, 0.7f32])));
        assert!(!f32s(json!(0.5), json!(0.6)));
        // Small learning rates are told apart.
        assert!(!f32s(json!(1e-7), json!(1e-6)));
        assert!(f32s(json!(folo_vision::train::f32_decimal(1e-7)), json!(1e-7f32)));
        // f64 settings compare exactly (1e-50 is 0 as an f32).
        assert!(!same(&json!(1e-50), &json!(0.0), false));
        assert!(same(&json!(0.01), &json!(0.01), false));
    }

    #[test]
    fn concurrent_runs_reserve_distinct_folders() {
        let project = TestDir::new("reserve");
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let project = project.to_path_buf();
                std::thread::spawn(move || super::reserve_dir(&project, "shapes-n").unwrap())
            })
            .collect();
        let mut dirs: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        dirs.sort();
        dirs.dedup();
        assert_eq!(dirs.len(), 8);
        assert!(dirs.contains(&project.join("shapes-n")) && dirs.contains(&project.join("shapes-n8")));
        let _ = std::fs::remove_dir_all(&project);
    }

    /// The server's `folo train` command (`RunEstimate.cli`, `GET /api/runs/{id}/cli`) parses with the real
    /// arguments, and maps to the trainer config the server builds from the same `TrainConfig` (`from_api`).
    #[test]
    fn server_cli_commands_train_the_servers_config() {
        use clap::Parser;
        use folo_vision::model::HeadMode;
        let base: folo_api::TrainConfig =
            serde_json::from_value(json!({ "dataset_id": "ds_1", "name": "r6-n" })).unwrap();
        let mut all = base.clone();
        all.name = "my run 'two'".into();
        all.arch = folo_api::Arch::S;
        (all.epochs, all.batch, all.imgsz, all.seed) = (37, 8, 320, 7);
        all.device = "gpu:0".into();
        all.optimizer = folo_api::Optimizer::Sgd;
        all.lr0 = 0.0123;
        all.amp = true;
        all.head = folo_api::Head::O2o;
        all.augment = folo_api::Augment {
            mosaic: 0.5,
            mixup: 0.15,
            hsv: [0.0, 0.25, 0.333],
            fliplr: 0.0,
            scale: 0.9,
            translate: 0.05,
            close_mosaic: 3,
        };
        let mut adamw = base.clone();
        adamw.optimizer = folo_api::Optimizer::Adamw;
        // A name that looks like a flag.
        let mut flaglike = base.clone();
        flaglike.name = "--amp".into();
        let data = "C:/data sets/r6";
        let (auto, coco) = (folo_api::DatasetFormat::Auto, folo_api::DatasetFormat::Coco);
        // (config, registered format, --weights, the head of the weights file: the CLI's default without --head)
        let cases = [
            (base.clone(), auto, None, HeadMode::O2m),
            (all.clone(), coco, None, HeadMode::O2m),
            (adamw, folo_api::DatasetFormat::Yolo, None, HeadMode::O2m),
            (flaglike, auto, None, HeadMode::O2m),
            (base.clone(), auto, Some("folo-n-coco"), HeadMode::O2o),
            (all.clone(), coco, Some("D:/folo/runs/run_1a2b/weights/best.folo"), HeadMode::O2o),
        ];
        for (api, format, weights, file_head) in cases {
            let cmd = folo_server::cli::train_command(&api, data, format, weights);
            let cli =
                crate::cli::Cli::try_parse_from(&cmd.argv).unwrap_or_else(|e| panic!("{}: {e}", cmd.cli));
            let crate::cli::Command::Train(args) = cli.command else { panic!("not train: {}", cmd.cli) };
            assert_eq!(args.data.as_deref(), Some(data), "{}", cmd.cli);
            assert_eq!(args.weights.as_deref(), weights, "{}", cmd.cli);
            // Read as the server reads the registered dataset (runs.rs `data_format`).
            let want_format = match format {
                folo_api::DatasetFormat::Auto => folo_data::Format::Auto,
                folo_api::DatasetFormat::Yolo => folo_data::Format::Yolo,
                folo_api::DatasetFormat::Coco => folo_data::Format::Coco,
            };
            assert_eq!(crate::dataset::format_of(args.format), want_format, "{}", cmd.cli);
            // With --weights the CLI takes the architecture from the file; the server already set it to the weights'.
            let arch = if weights.is_some() { api.arch.as_str().to_string() } else { args.model.clone() };
            let got = super::config_of(&args, args.name.clone().unwrap(), data.into(), arch, file_head);
            let mut want = folo_vision::TrainConfig::from_api(&api);
            (want.dataset, want.data_format, want.workers) = (data.into(), want_format, got.workers);
            assert_eq!(got, want, "{}", cmd.cli);
        }
    }
}
