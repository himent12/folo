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

use crate::cli::{OptimizerArg, TrainArgs};
use crate::engine::{engine_error, open_dataset};
use crate::output::{CliError, ExitCode, Out};

pub fn train(args: TrainArgs, out: Out) -> Result<(), CliError> {
    crate::engine::init_device(&args.device);
    let stop = Arc::new(AtomicBool::new(false));
    watch_ctrl_c(stop.clone());
    let nb = Arc::new(AtomicU64::new(0));
    let (mut trainer, run_dir) = match &args.resume {
        Some(ckpt) => resume(&args, ckpt, out, stop, nb.clone())?,
        None => start(&args, out, stop, nb.clone())?,
    };
    nb.store(trainer.batches_per_epoch() as u64, Ordering::Relaxed);
    if !out.json {
        out.human(format!("Run folder: {}", run_dir.display()));
    }
    let summary = trainer.run().map_err(engine_error)?;
    finish(&summary, &run_dir, trainer.config(), out);
    Ok(())
}

fn start(
    args: &TrainArgs,
    out: Out,
    stop: Arc<AtomicBool>,
    nb: Arc<AtomicU64>,
) -> Result<(Trainer, PathBuf), CliError> {
    let data_arg = args.data.as_deref().expect("clap requires --data without --resume");
    let (ds, ds_name) = open_dataset(data_arg)?;
    let mut weights = None;
    let arch = if args.model.ends_with(".folo") {
        let p = PathBuf::from(&args.model);
        if !p.is_file() {
            return Err(CliError::new(
                ExitCode::NotFound,
                "model_not_found",
                format!("{} does not exist", p.display()),
            ));
        }
        let file = folo_nn::io::ModelFile::open(&p).map_err(|e| engine_error(e.into()))?;
        weights = Some(p);
        file.metadata().arch.clone()
    } else {
        args.model.clone()
    };
    let suffix = arch.strip_prefix("folo-").unwrap_or(&arch).to_string();
    let name = args.name.clone().unwrap_or_else(|| format!("{ds_name}-{suffix}"));
    let run_dir = unique_dir(&args.project, &name);
    let cfg = TrainConfig {
        name: run_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(name),
        dataset: absolute(data_arg),
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
    };
    // Validate before creating the run folder, so a typo leaves nothing behind.
    cfg.validate().map_err(engine_error)?;
    let data = train_data(&ds)?;
    std::fs::create_dir_all(&run_dir).map_err(|e| {
        CliError::new(ExitCode::Runtime, "io_error", format!("cannot create {}: {e}", run_dir.display()))
    })?;
    let opts = TrainOptions {
        run_dir: Some(run_dir.clone()),
        events: Some(progress_sink(out, cfg.epochs, nb)),
        stop,
        weights,
        ..TrainOptions::default()
    };
    Ok((Trainer::new(cfg, data, opts).map_err(engine_error)?, run_dir))
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
            format!("{} does not exist", ckpt.display()),
        ));
    }
    let file = folo_nn::io::ModelFile::open(ckpt).map_err(|e| engine_error(e.into()))?;
    let Some(c) = file.checkpoint() else {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("{} is a weights file, not a training checkpoint", ckpt.display()),
        )
        .hint("resume from runs/<name>/weights/last.folo"));
    };
    let stored: TrainConfig = serde_json::from_value(c.train.clone()).map_err(|e| {
        CliError::new(ExitCode::Runtime, "invalid_weights", format!("folo.train is not a TrainConfig: {e}"))
    })?;
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
    let (ds, _) = open_dataset(&data_arg)?;
    let data = train_data(&ds)?;
    let epochs = stored.epochs;
    let opts = TrainOptions {
        run_dir: Some(run_dir.clone()),
        events: Some(progress_sink(out, epochs, nb)),
        stop,
        ..TrainOptions::default()
    };
    let overrides = ResumeOverrides { workers: args.workers, device: Some(args.device.clone()) };
    Ok((Trainer::resume(ckpt, data, opts, overrides).map_err(engine_error)?, run_dir))
}

/// `p` made absolute (so a resume from another folder finds the dataset).
fn absolute(p: &str) -> String {
    std::path::absolute(p).map_or_else(|_| p.to_string(), |a| a.display().to_string())
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

/// `<project>/<name>`, or `<name>2`, `<name>3`, … if it exists.
fn unique_dir(project: &Path, name: &str) -> PathBuf {
    let first = project.join(name);
    if !first.exists() {
        return first;
    }
    (2..).map(|i| project.join(format!("{name}{i}"))).find(|p| !p.exists()).expect("some suffix is free")
}

/// First Ctrl+C: stop after the current batch (the last epoch's checkpoint is kept). A second one exits.
fn watch_ctrl_c(stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return };
        rt.block_on(async {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\nstopping after the current batch (Ctrl+C again to abort)…");
                stop.store(true, Ordering::Relaxed);
            }
            if tokio::signal::ctrl_c().await.is_ok() {
                std::process::exit(130);
            }
        });
    });
}

/// The event consumer: JSON lines with `--json` (plus one-line epoch summaries on stderr), otherwise a live view.
/// `nb` (batches per epoch) is filled in once the trainer exists.
fn progress_sink(out: Out, epochs: u32, nb: Arc<AtomicU64>) -> folo_vision::train::EventSink {
    let mut view =
        Progress { out, epochs, nb, tty: std::io::stdout().is_terminal(), header: false, live: false };
    Box::new(move |e: &RunEvent| view.event(e))
}

struct Progress {
    out: Out,
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
            self.out.record(&event_record(e));
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
            RunEvent::Status(_) => {}
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

fn finish(s: &TrainSummary, run_dir: &Path, cfg: &TrainConfig, out: Out) {
    out.record(&json!({
        "event": "done",
        "status": s.status,
        "run_dir": run_dir,
        "epochs_completed": s.epochs_completed,
        "epochs": cfg.epochs,
        "early_stopped": s.early_stopped,
        "best": s.best.map(|(epoch, fitness)| json!({ "epoch": epoch, "map50_95": fitness })),
        "best_metrics": s.best_metrics,
        "weights": { "best": s.best_weights, "last": s.last },
    }));
    let verb = match s.status {
        RunStatus::Stopped => "Stopped",
        _ if s.early_stopped => "Stopped early",
        _ => "Finished",
    };
    out.human(format!("{verb} after {} of {} epochs.", s.epochs_completed, cfg.epochs));
    if let (Some((epoch, _)), Some(m)) = (s.best, &s.best_metrics) {
        out.human(format!(
            "Best epoch {epoch}: mAP50 {:.3}  mAP50-95 {:.3}  P {:.3}  R {:.3}",
            m.map50, m.map50_95, m.precision, m.recall
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
        out.human(format!("Weights: {} (and last.folo)", b.display()));
    }
}

fn read_classes(run_dir: &Path) -> Vec<String> {
    std::fs::read_to_string(run_dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| serde_json::from_value(v["classes"].clone()).ok())
        .unwrap_or_default()
}
