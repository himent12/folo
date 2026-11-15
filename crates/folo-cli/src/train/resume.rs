//! `folo train --resume`: continuing a run from its checkpoint, with the flags it may and may not change.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};

use folo_vision::train::ResumeOverrides;
use folo_vision::{TrainConfig, TrainOptions, Trainer};
use serde_json::json;

use crate::cli::TrainArgs;
use crate::engine::{engine_error, open_dataset};
use crate::output::{CliError, ExitCode, Out};

use super::progress::progress_sink;
use super::{teacher_setting, train_data};

pub(super) fn resume(
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
    let file = folo_nn::io::ModelFile::open(ckpt).map_err(|e| engine_error("train", e.into()))?;
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
    if let Some(s) = args.set.iter().find(|s| !s.trim_start().starts_with("distill.")) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--set {s} cannot change a resumed run (a resumed run keeps its settings; only distill.* may change)"),
        )
        .hint("leave --set out, or start a new run from its weights with --model <run>/weights/best.folo"));
    }
    check_resume_flags(args, &stored)?;
    let distill = resume_distill(args, &stored)?;
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
        cache: args.cache.into(),
        ..TrainOptions::default()
    };
    let overrides = ResumeOverrides {
        workers: args.workers,
        // The run's device unless --device says otherwise (its default, `auto`, is not the run's choice).
        device: args.explicit.iter().any(|f| f == "device").then(|| args.device.clone()),
        data_format: Some(format),
        distill,
    };
    Ok((Trainer::resume(ckpt, data, opts, overrides).map_err(|e| engine_error("train", e))?, run_dir))
}

/// The resume override of distillation (docs/CLI.md, docs/MODEL.md §8.9): `None` (keep what the run stored) without
/// `--teacher` and `--set distill.*`; else the stored `distill` (or, for a run without one, the starting values) with
/// the teacher of `--teacher` and the `--set distill.*` values, validated.
pub(super) fn resume_distill(
    args: &TrainArgs,
    stored: &TrainConfig,
) -> Result<Option<folo_vision::DistillConfig>, CliError> {
    if args.teacher.is_none() && args.set.is_empty() {
        return Ok(None);
    }
    let mut base = TrainConfig { distill: stored.distill.clone(), ..TrainConfig::default() };
    if let Some(t) = &args.teacher {
        crate::weights::resolve(Path::new(t))?;
        let teacher = teacher_setting(t);
        base.distill = Some(match base.distill.take() {
            Some(d) => folo_vision::DistillConfig { teacher, ..d },
            None => folo_vision::DistillConfig::new(teacher),
        });
    } else if base.distill.is_none() {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            "--set distill.* on a resumed run that does not distil needs --teacher",
        )
        .hint("add --teacher <weights> to distil from the resumed epoch on"));
    }
    let cfg = super::config::apply_sets(base, &args.set)?;
    let d = cfg.distill.expect("distill is set");
    d.validate().map_err(|e| CliError::new(ExitCode::Usage, "invalid_config", e))?;
    Ok(Some(d))
}

/// Flags a resume may change: where and how fast it runs, where its data is, how it reports, and (§8.9) the
/// teacher it distils from.
const RESUME_MAY_CHANGE: &[&str] =
    &["resume", "data", "format", "workers", "device", "json_events", "project", "cache", "teacher"];

/// With `--resume`, every other flag given on the command line must repeat the run's stored setting (as MCP
/// and `POST /api/runs` require): a changed one is `invalid_config`, naming the flag and both values.
pub(super) fn check_resume_flags(args: &TrainArgs, stored: &TrainConfig) -> Result<(), CliError> {
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
            "weights" => {
                return Err(CliError::new(
                    ExitCode::Usage,
                    "invalid_config",
                    "--weights cannot change a resumed run (it continues from its checkpoint)",
                )
                .hint("leave --weights out to resume the run, or start a new run from those weights without --resume"));
            }
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
                 --format, --cache, --project and --json-events may change), or start a new run from its weights \
                 with --model <run>/weights/best.folo"
            )));
        }
    }
    Ok(())
}

/// Equal JSON values: integers exactly, floats exactly or, for `f32` settings (stored widened to f64), as the
/// same f32.
pub(super) fn same(a: &serde_json::Value, b: &serde_json::Value, f32_setting: bool) -> bool {
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
