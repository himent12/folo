//! `folo train --cloud`: launches and follows a cloud training run (via `--server`, the `folo serve`
//! owning the data dir, or an in-process `RealStore`) with per-run spot-preemption auto-resume controls
//! (`--max-preempt-retries`) and live preemption status reporting.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_api::{
    Augment, CloudAvailability, CloudOffer, Compute, DatasetFormat, Head, Optimizer, RegisterDataset, Run,
    RunStatus, TrainConfig,
};
use folo_mcp::proxy::HttpStore;
use folo_server::RealStore;
use folo_server::store::{
    AutoBatchOptions, CloudPreemptionStatus, DEFAULT_MAX_PREEMPT_RETRIES, RunOptions, Store,
    check_max_preempt_retries, parse_arch_override,
};
use serde_json::json;

use crate::cli::bounds::{
    CLOUD_DISK_GB, CLOUD_MAX_HOURS, CLOUD_MAX_PREEMPT_RETRIES, NON_NEGATIVE_FLOAT, TRAIN_EPOCHS, TRAIN_IMGSZ,
    TRAIN_LR0, TRAIN_SCALE, UNIT_INTERVAL,
};
use crate::cli::{FormatArg, HeadArg, JsonEvents, OptimizerArg, TrainArgs};
use crate::output::{CliError, ExitCode, Out};
use crate::serve::{data_dir_in_use, data_dir_lock_failed};
use crate::serve_lock::{self, Acquire, ServeLock};

const POLL_LOCAL: Duration = Duration::from_millis(250);
const POLL_SERVER: Duration = Duration::from_millis(1000);
const MAX_STOP_RETRIES: u32 = 5;

/// Local-only flags that are rejected when explicitly passed alongside `--cloud`.
const LOCAL_ONLY_FLAGS: [(&str, &str); 11] = [
    ("project", "--project"),
    ("workers", "--workers"),
    ("cache", "--cache"),
    ("teacher", "--teacher"),
    ("set", "--set"),
    ("device", "--device"),
    ("lrf", "--lrf"),
    ("cos_lr", "--cos-lr"),
    ("warmup_epochs", "--warmup-epochs"),
    ("patience", "--patience"),
    ("auto_batch_target", "--auto-batch-target"),
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
    if matches!(args.format, FormatArg::Voc) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            "`--format voc` is not supported with `--cloud` (cloud training accepts `auto`, `yolo`, or `coco`)",
        )
        .hint("convert the dataset with `folo dataset convert` or omit `--format voc`"));
    }
    if args.batch.is_auto() {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            "`--batch auto` is not supported for cloud runs yet (pass an explicit integer `--batch`)",
        )
        .hint("pick an integer `--batch` such as `--batch 16` for `--cloud`, or omit `--cloud` to auto-batch locally"));
    }
    let b = args.batch.fixed_or_default();
    if !(1..=1024).contains(&b) {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--batch must be in 1..=1024, got {b}"),
        ));
    }
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
        let (arch, ov) = parse_arch_override(&args.model).map_err(|e| {
            CliError::new(ExitCode::Usage, "invalid_config", format!("--model: {}", e.body.message))
                .hint("pass a v1 model architecture (folo-n..folo-x) or use --weights <MODEL_ID>")
        })?;
        if ov.is_some() {
            return Err(CliError::new(
                ExitCode::Usage,
                "invalid_config",
                format!("`--model {}` (FOLO v2-Iso) is not supported for cloud runs yet", args.model),
            )
            .hint("pick a v1 architecture (`folo-n`..`folo-x`) for `--cloud`, or omit `--cloud` to train locally"));
        }
        Ok((arch, None))
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
    let (mut arch, _) = validate_cloud_args(&args)?;
    let connection_id = args.cloud.clone().expect("train_cloud requires --cloud");
    let data_arg = args.data.as_deref().expect("clap requires --data without --resume");
    let session = open_session(&args)?;

    // Resolve weights architecture and head before registering any dataset so invalid or v2 `--weights` fails cleanly.
    let mut weights_head = None;
    if let Some(w) = &args.weights {
        let (model, model_arch) = session.store.get_model_with_arch(w).map_err(api_error)?;
        let (parsed, ov) = parse_arch_override(&model_arch).map_err(api_error)?;
        if ov.is_some() {
            return Err(CliError::new(
                ExitCode::Usage,
                "invalid_config",
                format!("`--weights {w}` uses {model_arch} (FOLO v2-Iso), which is not supported for cloud runs yet"),
            )
            .hint("use v1 weights (`folo-n`..`folo-x`) for `--cloud`, or omit `--cloud` to train locally"));
        }
        arch = parsed;
        weights_head = Some(model.head);
    }

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

    // Resolve offer: explicit `--offer`, else cheapest in-stock offer on the connection.
    let offer_id = match &args.offer {
        Some(id) => id.clone(),
        None => {
            let list = session.store.cloud_offers(&connection_id, false).map_err(api_error)?;
            select_default_offer(&list.offers, &connection_id)?
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
        head: resolve_cloud_head(args.head, weights_head),
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
        arch: None,
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

    follow_cloud_run(&session, &run.id, &stop, args.json_events, out)
}

fn resolve_cloud_head(explicit: Option<HeadArg>, weights_head: Option<Head>) -> Head {
    match explicit {
        Some(HeadArg::O2o) => Head::O2o,
        Some(HeadArg::O2m) => Head::O2m,
        None => weights_head.unwrap_or(Head::O2m),
    }
}

fn select_default_offer(offers: &[CloudOffer], connection_id: &str) -> Result<String, CliError> {
    offers.iter().find(|o| o.availability != CloudAvailability::None).map(|o| o.id.clone()).ok_or_else(|| {
        CliError::new(
            ExitCode::NotFound,
            "offer_not_found",
            format!("connection '{connection_id}' has no in-stock offers right now"),
        )
        .hint("run `folo cloud offers <CONNECTION_ID>` to inspect availability or pass `--offer <OFFER_ID>`")
    })
}

fn failed_run_error(run: &Run) -> CliError {
    let code = run.error_code.as_deref().unwrap_or("training_failed");
    CliError::new(
        ExitCode::of(code),
        code,
        run.error.clone().unwrap_or_else(|| format!("cloud run {} failed", run.id)),
    )
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
    json_events: JsonEvents,
    out: Out,
) -> Result<(), CliError> {
    let poll = if session.local.is_some() { POLL_LOCAL } else { POLL_SERVER };
    follow_cloud_run_with_poll(session, run_id, stop, json_events, out, poll)
}

fn follow_cloud_run_with_poll(
    session: &CloudSession,
    run_id: &str,
    stop: &AtomicBool,
    json_events: JsonEvents,
    out: Out,
    poll: Duration,
) -> Result<(), CliError> {
    let mut last_sig: Option<ProgressSig> = None;
    let mut stop_acknowledged = false;
    let mut stop_failures: u32 = 0;

    loop {
        if stop.load(Ordering::Relaxed) && !stop_acknowledged {
            match session.store.stop_run(run_id) {
                Ok(_) => {
                    stop_acknowledged = true;
                    stop_failures = 0;
                }
                Err(e) if e.body.code == "run_not_active" => {
                    stop_acknowledged = true;
                }
                Err(e) => {
                    stop_failures += 1;
                    out.record(&json!({
                        "event": "stop_retry",
                        "run_id": run_id,
                        "attempt": stop_failures,
                        "max_attempts": MAX_STOP_RETRIES,
                        "code": e.body.code,
                        "error": e.body.message,
                    }));
                    out.human(format!(
                        "warning: stop request for cloud run {run_id} failed ({}/{MAX_STOP_RETRIES}): {}",
                        stop_failures, e.body.message
                    ));
                    if stop_failures >= MAX_STOP_RETRIES {
                        return Err(CliError::new(
                            ExitCode::of(&e.body.code),
                            &e.body.code,
                            format!(
                                "failed to stop cloud run {run_id} after Ctrl+C ({MAX_STOP_RETRIES} attempts): {}; \
                                 the cloud instance may still be running",
                                e.body.message
                            ),
                        )
                        .hint(format!(
                            "stop it manually in Studio, via `POST /api/runs/{run_id}/stop`, or check `folo cloud instances`"
                        )));
                    }
                    std::thread::sleep(poll);
                    continue;
                }
            }
        }
        let (run, preemption) = match session.store.get_run_with_preemption(run_id) {
            Ok(v) => v,
            Err(_) if stop.load(Ordering::Relaxed) && !stop_acknowledged => {
                // Ctrl+C arrived while the status request was in flight and failed; route into the stop retry loop
                // rather than exiting without attempting to stop the paid cloud run.
                continue;
            }
            Err(e) => return Err(api_error(e)),
        };
        let sig = ProgressSig {
            status: run.status,
            phase: run.cloud.as_ref().map(|c| c.phase.as_str().to_string()),
            epoch: run.epoch,
            step: if json_events == JsonEvents::Epoch { 0 } else { run.step },
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
                return Err(failed_run_error(&run));
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
        "error_code": run.error_code,
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
        if run.status == RunStatus::Completed {
            msg.push_str(&format!("; survived {} spot preemption(s) (limit {})", p.retries, p.max_retries));
        } else {
            msg.push_str(&format!(
                "; after {} spot preemption retry/retries (limit {})",
                p.retries, p.max_retries
            ));
        }
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

        // Local-only flags conflict with `--cloud` at clap parse time so `folo describe train` surfaces them too.
        for extra in [
            &["--project", "out"][..],
            &["--workers", "2"],
            &["--cache", "ram"],
            &["--teacher", "m.folo"],
            &["--set", "momentum=0.9"],
            &["--device", "cpu"],
            &["--lrf", "0.1"],
            &["--cos-lr"],
            &["--warmup-epochs", "5"],
            &["--patience", "20"],
            &["--auto-batch-target", "0.7"],
        ] {
            let mut argv = vec!["folo", "train", "--data", "d", "--cloud", "cc_1"];
            argv.extend_from_slice(extra);
            let err = Cli::try_parse_from(&argv).unwrap_err();
            assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict, "{extra:?}: {err}");
        }

        // Explicit `--format voc`, `--batch auto`, and `folo2-*` models fail upfront before opening the store.
        for (extra, needle) in [
            (&["--format", "voc"][..], "--format voc"),
            (&["--batch", "auto"], "--batch auto"),
            (&["--model", "folo2-n"], "folo2-n"),
        ] {
            let mut argv = vec!["folo", "train", "--data", "d", "--cloud", "cc_1"];
            argv.extend_from_slice(extra);
            let parsed = parse_train(&argv);
            let e = validate_cloud_args(&parsed).unwrap_err();
            assert_eq!((e.exit, e.code.as_str()), (ExitCode::Usage, "invalid_config"), "{extra:?}");
            assert!(e.message.contains(needle), "{extra:?}: {}", e.message);
        }

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
            "folo-s",
        ]);
        let (arch, ov) = validate_cloud_args(&ok).unwrap();
        assert_eq!(arch, folo_api::Arch::S);
        assert_eq!(ov, None);
    }

    #[test]
    fn default_offer_skips_out_of_stock_and_failed_run_preserves_error_code() {
        let offer = |id: &str, price: f64, availability: CloudAvailability| CloudOffer {
            id: id.into(),
            gpu: "GPU".into(),
            vram_bytes: 24 * 1024 * 1024 * 1024,
            gpu_count: 1,
            price_per_hour: Some(price),
            region: None,
            availability,
            detail: None,
        };
        let offers = vec![
            offer("cheap-oos", 0.10, CloudAvailability::None),
            offer("in-stock", 0.45, CloudAvailability::Low),
            offer("high-stock", 0.80, CloudAvailability::High),
        ];
        assert_eq!(select_default_offer(&offers, "cc_1").unwrap(), "in-stock");

        assert_eq!(resolve_cloud_head(None, None), Head::O2m);
        assert_eq!(resolve_cloud_head(None, Some(Head::O2o)), Head::O2o);
        assert_eq!(resolve_cloud_head(Some(HeadArg::O2m), Some(Head::O2o)), Head::O2m);
        assert_eq!(resolve_cloud_head(Some(HeadArg::O2o), Some(Head::O2m)), Head::O2o);

        let only_oos = vec![offer("cheap-oos", 0.10, CloudAvailability::None)];
        let err = select_default_offer(&only_oos, "cc_1").unwrap_err();
        assert_eq!((err.exit, err.code.as_str()), (ExitCode::NotFound, "offer_not_found"));

        let mut run: Run = serde_json::from_value(json!({
            "id": "run_fail",
            "name": "spot",
            "status": "failed",
            "config": { "dataset_id": "ds_1" },
            "created": "2026-11-27T00:00:00Z",
            "started": null,
            "finished": null,
            "epoch": 0,
            "epochs": 5,
            "step": 0,
            "steps_per_epoch": 0,
            "eta_s": null,
            "best": null,
            "device": "gpu:0",
            "model_id": null,
            "error": "worker preempted",
            "error_code": "cloud_instance_lost"
        }))
        .unwrap();
        let e = failed_run_error(&run);
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::Runtime, "cloud_instance_lost"));
        assert!(e.message.contains("worker preempted"));

        run.error_code = Some("not_supported".into());
        let e = failed_run_error(&run);
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::NotImplemented, "not_supported"));

        run.error_code = None;
        let e = failed_run_error(&run);
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::Runtime, "training_failed"));
    }

    #[test]
    fn follow_retries_transient_stop_error_and_reports_persistent_stop_failure() {
        use folo_server::error::Error as ServerError;
        use std::sync::atomic::AtomicU32;

        struct StopStore {
            fail_stops: AtomicU32,
            stop_calls: AtomicU32,
            stopped: AtomicBool,
            /// Optional flag flipped inside the first `get_run` error to simulate Ctrl+C arriving mid-poll.
            trigger_stop_on_poll: Option<Arc<AtomicBool>>,
        }
        impl Store for StopStore {
            fn is_mock(&self) -> bool {
                true
            }
            fn gpus(&self) -> Vec<folo_api::GpuInfo> {
                vec![]
            }
            fn list_datasets(&self) -> folo_server::error::Result<Vec<folo_api::Dataset>> {
                unimplemented!()
            }
            fn register_dataset(&self, _: RegisterDataset) -> folo_server::error::Result<folo_api::Dataset> {
                unimplemented!()
            }
            fn get_dataset(&self, _: &str) -> folo_server::error::Result<folo_api::Dataset> {
                unimplemented!()
            }
            fn delete_dataset(&self, _: &str) -> folo_server::error::Result<()> {
                unimplemented!()
            }
            fn list_images(
                &self,
                _: &str,
                _: &folo_server::store::ImageQuery,
            ) -> folo_server::error::Result<folo_api::ImagePage> {
                unimplemented!()
            }
            fn create_synthetic(
                &self,
                _: folo_api::SyntheticDataset,
            ) -> folo_server::error::Result<folo_api::Dataset> {
                unimplemented!()
            }
            fn list_issues(
                &self,
                _: &str,
                _: &folo_server::store::IssueQuery,
            ) -> folo_server::error::Result<folo_api::IssuePage> {
                unimplemented!()
            }
            fn image(
                &self,
                _: &str,
                _: &str,
                _: Option<u32>,
            ) -> folo_server::error::Result<folo_server::store::ImageData> {
                unimplemented!()
            }
            fn list_models(&self) -> folo_server::error::Result<Vec<folo_api::Model>> {
                unimplemented!()
            }
            fn get_model(&self, _: &str) -> folo_server::error::Result<folo_api::Model> {
                unimplemented!()
            }
            fn model_eval(&self, _: &str) -> folo_server::error::Result<folo_api::ModelEval> {
                unimplemented!()
            }
            fn export_model(
                &self,
                _: &str,
                _: folo_api::ExportSpec,
            ) -> folo_server::error::Result<folo_api::ExportResult> {
                unimplemented!()
            }
            fn list_runs(&self) -> folo_server::error::Result<Vec<Run>> {
                unimplemented!()
            }
            fn create_run(&self, _: TrainConfig) -> folo_server::error::Result<Run> {
                unimplemented!()
            }
            fn get_run(&self, id: &str) -> folo_server::error::Result<Run> {
                if let Some(flag) = &self.trigger_stop_on_poll
                    && !flag.swap(true, Ordering::SeqCst)
                {
                    return Err(ServerError::internal("poll failed as Ctrl+C arrived"));
                }
                // Simulate a full server outage while `stop_calls <= fail_stops` so `get_run` also fails during
                // the outage window and only succeeds once the server recovers.
                if !self.stopped.load(Ordering::SeqCst)
                    && self.stop_calls.load(Ordering::SeqCst) <= self.fail_stops.load(Ordering::SeqCst)
                {
                    return Err(ServerError::internal("server temporarily unreachable"));
                }
                let status = if self.stopped.load(Ordering::SeqCst) { "stopped" } else { "running" };
                Ok(serde_json::from_value(json!({
                    "id": id,
                    "name": "spot",
                    "status": status,
                    "config": { "dataset_id": "ds_1" },
                    "created": "2026-11-27T00:00:00Z",
                    "started": null,
                    "finished": null,
                    "epoch": 1,
                    "epochs": 5,
                    "step": 2,
                    "steps_per_epoch": 2,
                    "eta_s": null,
                    "best": null,
                    "device": "gpu:0",
                    "model_id": null,
                    "error": null
                }))
                .unwrap())
            }
            fn stop_run(&self, id: &str) -> folo_server::error::Result<Run> {
                let call = self.stop_calls.fetch_add(1, Ordering::SeqCst) + 1;
                if call <= self.fail_stops.load(Ordering::SeqCst) {
                    return Err(ServerError::internal("transient stop error"));
                }
                self.stopped.store(true, Ordering::SeqCst);
                self.get_run(id)
            }
            fn move_run(&self, _: &str, _: folo_server::store::QueueMove) -> folo_server::error::Result<Run> {
                unimplemented!()
            }
            fn delete_run(&self, _: &str) -> folo_server::error::Result<()> {
                unimplemented!()
            }
            fn run_metrics(
                &self,
                _: &str,
                _: &folo_server::store::MetricsQuery,
            ) -> folo_server::error::Result<folo_api::RunMetrics> {
                unimplemented!()
            }
            fn run_logs(
                &self,
                _: &str,
                _: &folo_server::store::LogsQuery,
            ) -> folo_server::error::Result<folo_api::RunLogs> {
                unimplemented!()
            }
            fn run_samples(
                &self,
                _: &str,
                _: Option<u32>,
            ) -> folo_server::error::Result<folo_api::RunSamples> {
                unimplemented!()
            }
            fn subscribe(&self, _: &str) -> folo_server::error::Result<folo_server::store::Subscription> {
                unimplemented!()
            }
            fn predict(
                &self,
                _: folo_server::store::PredictInput,
            ) -> folo_server::error::Result<folo_api::PredictResponse> {
                unimplemented!()
            }
        }

        // Case 1: first stop (and status GET during the same outage) fails transiently, second succeeds -> cleanly stops.
        let transient = Arc::new(StopStore {
            fail_stops: AtomicU32::new(1),
            stop_calls: AtomicU32::new(0),
            stopped: AtomicBool::new(false),
            trigger_stop_on_poll: None,
        });
        let session = CloudSession { store: transient.clone(), local: None };
        let stop = AtomicBool::new(true);
        follow_cloud_run_with_poll(
            &session,
            "run_t1",
            &stop,
            JsonEvents::All,
            Out { json: false },
            Duration::from_millis(1),
        )
        .unwrap();
        assert_eq!(transient.stop_calls.load(Ordering::SeqCst), 2);

        // Case 2: persistent outage (both stop and status fail) fails with run_id after MAX_STOP_RETRIES attempts.
        let persistent = Arc::new(StopStore {
            fail_stops: AtomicU32::new(100),
            stop_calls: AtomicU32::new(0),
            stopped: AtomicBool::new(false),
            trigger_stop_on_poll: None,
        });
        let session2 = CloudSession { store: persistent.clone(), local: None };
        let err = follow_cloud_run_with_poll(
            &session2,
            "run_p1",
            &stop,
            JsonEvents::Epoch,
            Out { json: false },
            Duration::from_millis(1),
        )
        .unwrap_err();
        assert_eq!(persistent.stop_calls.load(Ordering::SeqCst), MAX_STOP_RETRIES);
        assert_eq!(err.code, "internal");
        assert!(err.message.contains("run_p1"), "{}", err.message);

        // Case 3: Ctrl+C arrives while an in-flight status GET fails -> routes into stop_run instead of exiting.
        let late_stop = Arc::new(AtomicBool::new(false));
        let in_flight = Arc::new(StopStore {
            fail_stops: AtomicU32::new(0),
            stop_calls: AtomicU32::new(0),
            stopped: AtomicBool::new(false),
            trigger_stop_on_poll: Some(late_stop.clone()),
        });
        let session3 = CloudSession { store: in_flight.clone(), local: None };
        follow_cloud_run_with_poll(
            &session3,
            "run_f1",
            &late_stop,
            JsonEvents::All,
            Out { json: false },
            Duration::from_millis(1),
        )
        .unwrap();
        assert_eq!(in_flight.stop_calls.load(Ordering::SeqCst), 1);
    }
}
