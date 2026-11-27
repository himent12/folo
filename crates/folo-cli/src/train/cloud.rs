//! `folo train --cloud`: launches and follows a cloud training run (via `--server`, the `folo serve`
//! owning the data dir, or an in-process `RealStore`) with per-run spot-preemption auto-resume controls
//! (`--max-preempt-retries`) and live preemption status reporting.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_api::{
    Augment, Compute, DatasetFormat, Head, Optimizer, RegisterDataset, Run, RunStatus, TrainConfig,
};
use folo_mcp::proxy::HttpStore;
use folo_server::RealStore;
use folo_server::store::{
    AutoBatchOptions, CloudPreemptionStatus, DEFAULT_MAX_PREEMPT_RETRIES, RunOptions, Store,
    check_auto_batch_options, check_max_preempt_retries, parse_arch_override,
};
use serde_json::json;

use crate::cli::bounds::{
    CLOUD_DISK_GB, CLOUD_MAX_HOURS, CLOUD_MAX_PREEMPT_RETRIES, NON_NEGATIVE_FLOAT, TRAIN_AUTO_BATCH_TARGET,
    TRAIN_EPOCHS, TRAIN_IMGSZ, TRAIN_LR0, TRAIN_LRF, TRAIN_SCALE, UNIT_INTERVAL,
};
use crate::cli::{FormatArg, HeadArg, OptimizerArg, TrainArgs};
use crate::output::{CliError, ExitCode, Out};
use crate::serve::{data_dir_in_use, data_dir_lock_failed};
use crate::serve_lock::{self, Acquire, ServeLock};

const POLL_LOCAL: Duration = Duration::from_millis(250);
const POLL_SERVER: Duration = Duration::from_millis(1000);

/// Local-only flags that are rejected when explicitly passed alongside `--cloud`.
const LOCAL_ONLY_FLAGS: [(&str, &str); 6] = [
    ("project", "--project"),
    ("workers", "--workers"),
    ("cache", "--cache"),
    ("teacher", "--teacher"),
    ("set", "--set"),
    ("device", "--device"),
];

/// Cloud-store connection flags that must not be explicitly passed on a local `folo train` run.
const CLOUD_ONLY_CONN_FLAGS: [(&str, &str); 4] =
    [("data_dir", "--data-dir"), ("server", "--server"), ("token", "--token"), ("tls_ca", "--tls-ca")];

/// Rejects explicit `--server` / `--data-dir` / `--token` / `--tls-ca` on a local (non-`--cloud`) run
/// while ignoring ambient environment variables such as `FOLO_SERVER` or `FOLO_DATA_DIR`.
pub(super) fn reject_explicit_cloud_conn_flags(args: &TrainArgs) -> Result<(), CliError> {
    for (id, flag) in CLOUD_ONLY_CONN_FLAGS {
        if args.explicit.iter().any(|e| e == id) {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                format!("{flag} on `folo train` requires `--cloud <CONNECTION_ID>`"),
            )
            .hint("pass `--cloud <CONNECTION_ID>` to train on a cloud connection, or omit the flag for local training"));
        }
    }
    Ok(())
}

/// Validates cloud-mode arguments before touching disk or the network.
pub(super) fn validate_cloud_args(args: &TrainArgs) -> Result<(folo_api::Arch, Option<String>), CliError> {
    if args.resume.is_some() {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            "--cloud cannot be combined with local checkpoint --resume",
        )
        .hint("resume stopped/failed cloud runs from Studio or via MCP `train` with `resume_run_id`"));
    }
    for (id, flag) in LOCAL_ONLY_FLAGS {
        if args.explicit.iter().any(|e| e == id) {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                format!("{flag} is for local training and is not supported with `--cloud`"),
            )
            .hint("omit local-only flags when using `--cloud`"));
        }
    }
    if let Some(h) = args.max_hours
        && !CLOUD_MAX_HOURS.contains_f64(h)
    {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--max-hours must be finite in 0.25..=720, got {h}"),
        ));
    }
    if let Some(gb) = args.disk_gb
        && !CLOUD_DISK_GB.contains_u64(u64::from(gb))
    {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--disk-gb must be in 10..=4096, got {gb}"),
        ));
    }
    if let Some(n) = args.max_preempt_retries
        && !CLOUD_MAX_PREEMPT_RETRIES.contains_u64(u64::from(n))
    {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!(
                "--max-preempt-retries must be in 0..={}, got {n}",
                folo_server::store::MAX_PREEMPT_RETRIES
            ),
        ));
    }
    check_max_preempt_retries(args.max_preempt_retries, true).map_err(api_error)?;

    if !TRAIN_EPOCHS.contains_u64(u64::from(args.epochs)) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--epochs must be in 1..=10000, got {}", args.epochs),
        ));
    }
    if !args.batch.is_auto() {
        let b = args.batch.fixed_or_default();
        if !(1..=1024).contains(&b) {
            return Err(CliError::new(
                ExitCode::Usage,
                "invalid_config",
                format!("--batch must be in 1..=1024 or `auto`, got {b}"),
            ));
        }
    }
    let auto_batch =
        AutoBatchOptions { enabled: args.batch.is_auto(), target_utilization: args.auto_batch_target };
    if let Some(t) = args.auto_batch_target
        && !TRAIN_AUTO_BATCH_TARGET.contains_f64(t)
    {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!(
                "--auto-batch-target must be finite in {}..={}, got {t}",
                folo_vision::train::MIN_AUTO_BATCH_UTILIZATION,
                folo_vision::train::MAX_AUTO_BATCH_UTILIZATION
            ),
        ));
    }
    check_auto_batch_options(&auto_batch).map_err(api_error)?;
    if !TRAIN_IMGSZ.contains_u64(u64::from(args.imgsz)) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--imgsz must be in 64..=4096 and a multiple of 32, got {}", args.imgsz),
        ));
    }
    if !TRAIN_LR0.contains_f64(f64::from(args.lr0)) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--lr0 must be in (0, 10), got {}", args.lr0),
        ));
    }
    if !TRAIN_LRF.contains_f64(args.lrf) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--lrf must be in (0, 1], got {}", args.lrf),
        ));
    }
    if !NON_NEGATIVE_FLOAT.contains_f64(args.warmup_epochs) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--warmup-epochs must be >= 0, got {}", args.warmup_epochs),
        ));
    }
    if !UNIT_INTERVAL.contains_f64(f64::from(args.mosaic))
        || !UNIT_INTERVAL.contains_f64(f64::from(args.mixup))
        || !UNIT_INTERVAL.contains_f64(f64::from(args.fliplr))
        || !TRAIN_SCALE.contains_f64(f64::from(args.scale))
        || !NON_NEGATIVE_FLOAT.contains_f64(f64::from(args.translate))
        || args.hsv.iter().any(|&v| !NON_NEGATIVE_FLOAT.contains_f64(f64::from(v)))
    {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            "augmentation hyperparameters out of range (check --mosaic, --mixup, --fliplr, --scale, --translate, --hsv)",
        ));
    }
    if let Some(name) = args.name.as_deref() {
        super::check_run_name(name)?;
    }

    if args.weights.is_some() {
        Ok((folo_api::Arch::N, None))
    } else {
        parse_arch_override(&args.model).map_err(|e| {
            CliError::new(ExitCode::Usage, "invalid_config", format!("--model: {}", e.body.message)).hint(
                "pass a model architecture (folo-n..folo-x or folo2-n..folo2-x) or use --weights <MODEL_ID>",
            )
        })
    }
}

struct CloudSession {
    store: Arc<dyn Store>,
    local: Option<ServeLock>,
}

impl Drop for CloudSession {
    fn drop(&mut self) {
        if self.local.is_some() {
            self.store.shutdown();
        }
    }
}

fn open_session(args: &TrainArgs) -> Result<CloudSession, CliError> {
    let tls_ca = serve_lock::load_tls_ca(args.tls_ca.as_deref())?;
    let http = |url: &str| {
        let h: Arc<dyn Store> = Arc::new(HttpStore::with_tls_ca(url, args.token.as_deref(), tls_ca.as_ref()));
        CloudSession { store: h, local: None }
    };
    if let Some(url) = &args.server {
        if let Err(why) = folo_client::check_base_url(url) {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                format!("--server (or FOLO_SERVER) must be a valid http:// or https:// URL ({why})"),
            ));
        }
        return Ok(http(url));
    }
    let data_dir = crate::data_dir(args.data_dir.clone())?;
    match serve_lock::acquire(&data_dir) {
        Ok(Acquire::Held(lock)) => {
            let store = RealStore::open(data_dir).published_weights(folo_api::weights::cache_dir());
            Ok(CloudSession { store: Arc::new(store), local: Some(lock) })
        }
        Ok(Acquire::InUse { url: Some(url) })
            if serve_lock::serves(&url, &data_dir, args.token.as_deref(), tls_ca.as_ref()) =>
        {
            Ok(http(&url))
        }
        Ok(Acquire::InUse { url }) => Err(data_dir_in_use(&data_dir, url.as_deref())),
        Err(e) => Err(data_dir_lock_failed(&data_dir, e)),
    }
}

pub(super) fn train_cloud(args: TrainArgs, out: Out) -> Result<(), CliError> {
    let (mut arch, mut arch_override) = validate_cloud_args(&args)?;
    let connection_id = args.cloud.clone().expect("train_cloud requires --cloud");
    let data_arg = args.data.as_deref().expect("clap requires --data without --resume");
    let session = open_session(&args)?;

    // Resolve dataset: an existing registered dataset id, else register the path on the store/server.
    let dataset_id = match session.store.get_dataset(data_arg) {
        Ok(ds) => ds.id,
        Err(e) if e.body.code == "dataset_not_found" => {
            let path = if args.server.is_some() { data_arg.to_string() } else { super::absolute(data_arg) };
            let format = match args.format {
                FormatArg::Auto | FormatArg::Voc => DatasetFormat::Auto,
                FormatArg::Yolo => DatasetFormat::Yolo,
                FormatArg::Coco => DatasetFormat::Coco,
            };
            let ds = session
                .store
                .register_dataset(RegisterDataset { name: None, path, format })
                .map_err(api_error)?;
            out.human(format!("dataset {} '{}' ({})", ds.id, ds.name, ds.path));
            ds.id
        }
        Err(e) => return Err(api_error(e)),
    };

    // Resolve weights architecture if `--weights` was provided.
    if let Some(w) = &args.weights {
        let model_arch = session.store.model_arch(w).map_err(api_error)?;
        let (parsed, ov) = parse_arch_override(&model_arch).map_err(api_error)?;
        arch = parsed;
        arch_override = ov;
    }

    // Resolve offer: explicit `--offer`, else cheapest available offer on the connection.
    let offer_id = match &args.offer {
        Some(id) => id.clone(),
        None => {
            let list = session.store.cloud_offers(&connection_id, false).map_err(api_error)?;
            list.offers.first().map(|o| o.id.clone()).ok_or_else(|| {
                CliError::new(
                    ExitCode::NotFound,
                    "offer_not_found",
                    format!("connection '{connection_id}' returned no available offers"),
                )
                .hint("run `folo cloud offers <CONNECTION_ID>` to inspect availability")
            })?
        }
    };

    let cfg = TrainConfig {
        name: args.name.clone().unwrap_or_default(),
        dataset_id,
        arch,
        epochs: args.epochs,
        batch: args.batch.fixed_or_default(),
        imgsz: args.imgsz,
        device: "auto".into(),
        optimizer: match args.optimizer {
            OptimizerArg::Auto => Optimizer::Auto,
            OptimizerArg::Sgd => Optimizer::Sgd,
            OptimizerArg::Adamw => Optimizer::Adamw,
            OptimizerArg::Musgd => Optimizer::Musgd,
        },
        lr0: args.lr0,
        weights: args.weights.clone(),
        augment: Augment {
            mosaic: args.mosaic,
            mixup: args.mixup,
            hsv: [args.hsv[0], args.hsv[1], args.hsv[2]],
            fliplr: args.fliplr,
            scale: args.scale,
            translate: args.translate,
            close_mosaic: args.close_mosaic,
        },
        seed: args.seed,
        amp: args.amp,
        head: match args.head {
            Some(HeadArg::O2o) => Head::O2o,
            Some(HeadArg::O2m) | None => Head::O2m,
        },
        compute: Some(Compute {
            connection_id: connection_id.clone(),
            offer_id: offer_id.clone(),
            max_hours: args.max_hours.map_or(folo_api::DEFAULT_MAX_HOURS, |h| h as f32),
            disk_gb: args.disk_gb,
        }),
    };
    let options = RunOptions {
        auto_batch: AutoBatchOptions {
            enabled: args.batch.is_auto(),
            target_utilization: args.auto_batch_target,
        },
        arch: arch_override,
        max_preempt_retries: args.max_preempt_retries,
    };

    let stop = Arc::new(AtomicBool::new(false));
    super::watch_interrupt(stop.clone(), "stopping the cloud run (interrupt again to abort)…");

    let run = session.store.create_run_with_options(cfg, options).map_err(api_error)?;
    let configured_retries = args.max_preempt_retries.unwrap_or(DEFAULT_MAX_PREEMPT_RETRIES);
    out.record(&json!({
        "event": "started",
        "run_id": run.id,
        "name": run.name,
        "status": run.status,
        "connection_id": connection_id,
        "offer_id": offer_id,
        "max_preempt_retries": configured_retries,
    }));
    out.human(format!(
        "started cloud run {} '{}' on {} (offer {}, max spot retries: {})",
        run.id, run.name, connection_id, offer_id, configured_retries
    ));

    follow_cloud_run(&session, &run.id, &stop, out)
}

#[derive(PartialEq, Eq)]
struct ProgressSig {
    status: RunStatus,
    phase: Option<String>,
    epoch: u32,
    step: u64,
    preemption: Option<(u32, u32, bool)>,
}

fn follow_cloud_run(
    session: &CloudSession,
    run_id: &str,
    stop: &AtomicBool,
    out: Out,
) -> Result<(), CliError> {
    let poll = if session.local.is_some() { POLL_LOCAL } else { POLL_SERVER };
    let mut last_sig: Option<ProgressSig> = None;
    let mut asked_stop = false;

    loop {
        if stop.load(Ordering::Relaxed) && !asked_stop {
            asked_stop = true;
            let _ = session.store.stop_run(run_id);
        }
        let (run, preemption) = session.store.get_run_with_preemption(run_id).map_err(api_error)?;
        let sig = ProgressSig {
            status: run.status,
            phase: run.cloud.as_ref().map(|c| c.phase.as_str().to_string()),
            epoch: run.epoch,
            step: run.step,
            preemption: preemption.as_ref().map(|p| (p.retries, p.max_retries, p.recovering)),
        };
        if last_sig.as_ref() != Some(&sig) {
            emit_progress(&run, preemption.as_ref(), out);
            last_sig = Some(sig);
        }
        match run.status {
            RunStatus::Completed => {
                emit_done(&run, preemption.as_ref(), out);
                return Ok(());
            }
            RunStatus::Stopped => {
                emit_done(&run, preemption.as_ref(), out);
                return Ok(());
            }
            RunStatus::Failed => {
                emit_done(&run, preemption.as_ref(), out);
                return Err(CliError::new(
                    ExitCode::Runtime,
                    "internal",
                    run.error.unwrap_or_else(|| format!("cloud run {run_id} failed")),
                ));
            }
            RunStatus::Queued | RunStatus::Running | RunStatus::Stopping => {
                std::thread::sleep(poll);
            }
        }
    }
}

fn emit_progress(run: &Run, preemption: Option<&CloudPreemptionStatus>, out: Out) {
    let mut rec = json!({
        "event": "progress",
        "run_id": run.id,
        "status": run.status,
        "epoch": run.epoch,
        "epochs": run.epochs,
        "step": run.step,
        "cloud_phase": run.cloud.as_ref().map(|c| c.phase.as_str()),
        "cost_usd": run.cloud.as_ref().and_then(|c| c.cost_usd),
    });
    if let Some(p) = preemption {
        rec["cloud_preemption"] = json!(p);
    }
    out.record(&rec);

    let phase = run.cloud.as_ref().map_or("-", |c| c.phase.as_str());
    let mut line = format!("{} [{}] epoch {}/{} step {}", run.id, phase, run.epoch, run.epochs, run.step);
    if let Some(b) = run.best {
        line.push_str(&format!(" best mAP50-95 {:.3}", b.map50_95));
    }
    if let Some(p) = preemption {
        if p.recovering {
            line.push_str(&format!(
                " | spot recovery in progress (preemptions {}/{})",
                p.retries, p.max_retries
            ));
        } else if p.retries > 0 {
            line.push_str(&format!(" | spot preemptions {}/{}", p.retries, p.max_retries));
        }
    }
    out.human(line);
}

fn emit_done(run: &Run, preemption: Option<&CloudPreemptionStatus>, out: Out) {
    let mut rec = json!({
        "event": "done",
        "run_id": run.id,
        "name": run.name,
        "status": run.status,
        "epoch": run.epoch,
        "epochs": run.epochs,
        "model_id": run.model_id,
        "best": run.best,
        "cost_usd": run.cloud.as_ref().and_then(|c| c.cost_usd),
        "error": run.error,
    });
    if let Some(p) = preemption {
        rec["cloud_preemption"] = json!(p);
    }
    out.record(&rec);
    let mut msg = format!("cloud run {} finished ({:?})", run.id, run.status);
    if let Some(m) = &run.model_id {
        msg.push_str(&format!("; model {m}"));
    }
    if let Some(p) = preemption
        && p.retries > 0
    {
        msg.push_str(&format!("; survived {} spot preemption(s) (limit {})", p.retries, p.max_retries));
    }
    out.human(msg);
}

fn api_error(e: folo_server::error::Error) -> CliError {
    let err = CliError::new(ExitCode::of(&e.body.code), &e.body.code, e.body.message);
    match e.body.hint {
        Some(h) => err.hint(h),
        None => err,
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::cli::{Cli, Command};

    fn parse_train(argv: &[&str]) -> TrainArgs {
        let cli = Cli::try_parse_from(argv).expect("parse");
        let Command::Train(mut args) = cli.command else { panic!("not train") };
        let matches = <Cli as clap::CommandFactory>::command().try_get_matches_from(argv).unwrap();
        if let Some(("train", m)) = matches.subcommand() {
            args.explicit = m
                .ids()
                .filter(|id| m.value_source(id.as_str()) == Some(clap::parser::ValueSource::CommandLine))
                .map(|id| id.to_string())
                .collect();
        }
        args
    }

    #[test]
    fn cloud_flag_validation_catches_invalid_ranges_and_local_only_flags() {
        // Explicit `--server` without `--cloud` is refused.
        let local_with_server =
            parse_train(&["folo", "train", "--data", "d", "--server", "http://127.0.0.1:7878"]);
        let e = reject_explicit_cloud_conn_flags(&local_with_server).unwrap_err();
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::Usage, "usage"));
        assert!(e.message.contains("--cloud"), "{}", e.message);

        // Out-of-range `--max-preempt-retries` (> 20) fails before network.
        let oob =
            parse_train(&["folo", "train", "--data", "d", "--cloud", "cc_1", "--max-preempt-retries", "21"]);
        let e = validate_cloud_args(&oob).unwrap_err();
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::Usage, "invalid_config"));

        // Local-only flag `--project` with `--cloud` is refused.
        let local_flag =
            parse_train(&["folo", "train", "--data", "d", "--cloud", "cc_1", "--project", "out"]);
        let e = validate_cloud_args(&local_flag).unwrap_err();
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::Usage, "usage"));
        assert!(e.message.contains("--project"), "{}", e.message);

        // Valid cloud args pass validation.
        let ok = parse_train(&[
            "folo",
            "train",
            "--data",
            "d",
            "--cloud",
            "cc_1",
            "--offer",
            "rtx4090",
            "--max-hours",
            "2.5",
            "--disk-gb",
            "80",
            "--max-preempt-retries",
            "5",
            "--model",
            "folo2-n",
        ]);
        let (arch, ov) = validate_cloud_args(&ok).unwrap();
        assert_eq!(arch, folo_api::Arch::N);
        assert_eq!(ov.as_deref(), Some("folo2-n"));
    }
}
