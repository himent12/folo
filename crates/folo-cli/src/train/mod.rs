//! `folo train`: a run under `<project>/<name>/` with a live progress view (or JSON lines with `--json`).
//!
//! This file checks the flags, reserves the run folder and starts the trainer. The rest of the folder:
//!
//! - `config.rs`: the trainer config of a new run, from the flags and `--set KEY=VALUE`.
//! - `resume.rs`: `--resume`, and which flags a resumed run may change.
//! - `progress.rs`: the live view or JSON lines while it trains, and the summary at the end.
//! - `stop.rs`: `folo stop` (the stop file) and Ctrl+C.
//! - `tests.rs`: unit tests of `--set`, `--resume` and run-folder reservation.

mod config;
mod progress;
mod resume;
mod stop;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use folo_api::RunStatus;
use folo_data::{Dataset, Split};
use folo_vision::{TrainData, TrainOptions, Trainer};

use crate::cli::TrainArgs;
use crate::engine::{engine_error, open_dataset};
use crate::output::{CliError, ExitCode, Out};
use config::{apply_sets, config_of};
use progress::{finish, log_line, progress_sink};
use resume::resume;
use stop::{late_stop, watch_ctrl_c, watch_stop_file};

pub use stop::request_stop;
pub(crate) use stop::watch_interrupt;

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
    watch_stop_file(&run_dir, stop.clone());
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
            return Err(engine_error("train", e));
        }
    };
    // A stop that came after the trainer's last look (the stop file written after the last poll, or Ctrl+C) found
    // nothing left to stop either; its stop file is not left behind.
    let late = late_stop(&run_dir, &stop);
    if late && summary.status == RunStatus::Completed {
        log_line(out, folo_vision::train::STOP_AFTER_LAST_BATCH);
    }
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
        let file = folo_nn::io::ModelFile::open(&p).map_err(|e| engine_error("train", e.into()))?;
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
    let mut base = config_of(args, name.clone(), absolute(data_arg), arch, weights_head);
    if let Some(t) = &args.teacher {
        // Resolved now for the error and its `folo weights pull` hint; the trainer resolves it again (and on resume).
        crate::weights::resolve(Path::new(t))?;
        base.distill = Some(folo_vision::DistillConfig::new(teacher_setting(t)));
    }
    let mut cfg = apply_sets(base, &args.set)?;
    // Validate before creating the run folder, so a typo leaves nothing behind.
    cfg.validate().map_err(|e| engine_error("train", e))?;
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
        cache: args.cache.into(),
        ..TrainOptions::default()
    };
    match Trainer::new(cfg, data, opts) {
        Ok(t) => Ok((t, run_dir)),
        Err(e) => {
            discard_if_empty(&run_dir, out);
            Err(engine_error("train", e))
        }
    }
}

/// `p` made absolute (so a resume from another folder finds the dataset).
fn absolute(p: &str) -> String {
    std::path::absolute(p).map_or_else(|_| p.to_string(), folo_api::json::path)
}

/// `distill.teacher` of `--teacher t`: an existing file made absolute (so a resume from another folder finds it),
/// else `t` as given (a published name, resolved in the weights cache whenever the run starts or resumes).
pub(super) fn teacher_setting(t: &str) -> String {
    if Path::new(t).is_file() { absolute(t) } else { t.to_string() }
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

#[cfg(test)]
mod tests;
