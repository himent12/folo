//! Resuming a stopped or failed run, here or on a cloud connection.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::http::StatusCode;
use folo_api::{BestEpoch, LogLevel, Run, RunCloud, RunEvent, RunStatus, TrainConfig};

use crate::engine;
use crate::error::{Error, Result};

use crate::util::{parse_rfc3339, unix_now};

use super::entry::status_event;
use super::{RunEntry, RunState, Runs, add_paused, mark_resume};

impl Runs {
    /// What `POST /api/runs` with `resume_run_id` asks of a `stopped` or `failed` run: the run, its config with
    /// the request's `fields` (the body's other fields), and any optional `max_preempt_retries` override.
    /// Every field must equal the run's config (an empty `name` counts as omitted), except `device`, `compute`,
    /// and `max_preempt_retries`, which may change. Nothing changes yet: [`Runs::resume`] (on this machine) or
    /// [`Runs::resume_cloud`] does.
    pub fn resume_config(
        &self,
        id: &str,
        fields: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(Arc<RunEntry>, TrainConfig, Option<Option<u32>>)> {
        self.check_open()?;
        let e = self.get(id)?;
        let st = e.state.lock().unwrap();
        resumable(&st, id)?;
        let stored = st.file.run.config.clone();
        let stored_auto = st.file.auto_batch;
        let stored_arch =
            crate::store::effective_arch(stored.arch, st.file.arch_override.as_deref()).to_string();
        let mut merged = serde_json::to_value(&stored).expect("configs serialize");
        let mut extra_differ: Vec<&str> = Vec::new();
        let mut max_preempt_retries_update: Option<Option<u32>> = None;
        let auto_batch_flag = match fields.get("auto_batch").filter(|v| !v.is_null()) {
            None => None,
            Some(v) => Some(v.as_bool().ok_or_else(|| {
                Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_body", "auto_batch: expected a boolean")
            })?),
        };
        if let Some(b) = auto_batch_flag
            && b != stored_auto.enabled
        {
            extra_differ.push("auto_batch");
        }
        for (k, v) in fields {
            if k == "resume_run_id" || k == "auto_batch" || (k == "name" && v.as_str() == Some("")) {
                continue;
            }
            if k == "max_preempt_retries" {
                if v.is_null() {
                    max_preempt_retries_update = Some(None);
                    continue;
                }
                let Some(n) = v.as_u64().and_then(|n| u32::try_from(n).ok()) else {
                    return Err(Error::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "invalid_body",
                        "max_preempt_retries: expected a non-negative integer",
                    ));
                };
                max_preempt_retries_update = Some(Some(n));
                continue;
            }
            if k == "arch" {
                if v.is_null() {
                    continue;
                }
                let Some(s) = v.as_str() else {
                    return Err(Error::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "invalid_body",
                        "arch: expected a string",
                    ));
                };
                crate::store::parse_arch_override(s)?;
                if s != stored_arch {
                    extra_differ.push("arch");
                }
                merged["arch"] = serde_json::to_value(stored.arch).expect("arch serializes");
                continue;
            }
            if k == "batch"
                && let Some(s) = v.as_str()
            {
                if s.eq_ignore_ascii_case("auto") {
                    if !stored_auto.enabled {
                        extra_differ.push("batch");
                    }
                    continue;
                }
                return Err(Error::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "invalid_body",
                    format!("batch: expected a positive integer in 1..=1024 or \"auto\", got {s:?}"),
                ));
            }
            if k == "auto_batch_target" {
                if v.is_null() {
                    continue;
                }
                let Some(t) = v.as_f64() else {
                    return Err(Error::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "invalid_body",
                        "auto_batch_target: expected a number",
                    ));
                };
                let have_t = stored_auto
                    .target_utilization
                    .unwrap_or(folo_vision::train::DEFAULT_AUTO_BATCH_UTILIZATION);
                if !stored_auto.enabled || (t - have_t).abs() > 1e-12 {
                    extra_differ.push("auto_batch_target");
                }
                continue;
            }
            merged[k.as_str()] = v.clone();
        }
        let wanted: TrainConfig = serde_json::from_value(merged)
            .map_err(|err| Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_body", err.to_string()))?;
        let (a, b) = (serde_json::to_value(&stored).unwrap(), serde_json::to_value(&wanted).unwrap());
        let mut differ: Vec<&str> = a
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, v)| !matches!(k.as_str(), "device" | "compute") && b.get(k.as_str()) != Some(v))
            .map(|(k, _)| k.as_str())
            .collect();
        for k in extra_differ {
            if !differ.contains(&k) {
                differ.push(k);
            }
        }
        if !differ.is_empty() {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("resuming keeps the run's config; {} differ(s) from it", differ.join(", ")),
            )
            .hint("omit the fields, or start a new run (with `weights` to start from this run's model)"));
        }
        if !Path::new(&st.file.dataset.path).exists() {
            return Err(Error::not_found("dataset", &st.file.dataset.id)
                .hint(format!("the run's dataset is gone from {}", st.file.dataset.path)));
        }
        if let Some(Some(n)) = max_preempt_retries_update {
            crate::store::check_max_preempt_retries(Some(n), wanted.compute.is_some())?;
        }
        drop(st);
        Ok((e, wanted, max_preempt_retries_update))
    }

    /// Resumes run `e` on this machine with `wanted` ([`Runs::resume_config`]): from `weights/last.folo` when it
    /// exists (the full checkpoint state, same id), from scratch otherwise.
    pub fn resume(self: &Arc<Self>, e: &Arc<RunEntry>, wanted: TrainConfig) -> Result<Run> {
        let device = engine::resolve(&wanted.device)?;
        // A mixed-precision run resumes in mixed precision (`amp` must match): refused before it is queued where this
        // server cannot train it, as a new run is.
        if wanted.amp
            && let Err(err) = folo_vision::train::amp_available(device)
        {
            let mut err = engine::engine_error(err);
            err.body.hint = Some(
                "the run trained in mixed precision and resumes in it: resume it on a server with tensor cores \
                 (`folo serve --tensor-cores` / `folo mcp --tensor-cores` on a GPU that has them), or start a new \
                 f32 run with `weights` set to its model"
                    .into(),
            );
            return Err(err);
        }
        let (_, attempt) = self.restart(e, wanted, None, Some(None))?;
        self.enqueue(e, attempt)?;
        self.record(e)
    }

    /// Resumes run `e` on a cloud connection with `wanted` ([`Runs::resume_config`]; `compute` set): queued again
    /// with a fresh `cloud`, and claimed for [`crate::cloud`] to execute.
    pub(crate) fn resume_cloud(
        &self,
        e: &Arc<RunEntry>,
        wanted: TrainConfig,
        cloud: RunCloud,
        max_preempt_retries: Option<Option<u32>>,
    ) -> Result<Run> {
        let (run, _) = self.restart(e, wanted, Some(cloud), max_preempt_retries)?;
        Ok(run)
    }

    /// Queues run `e` again with `wanted`'s `device` and `compute` (and `cloud`, for a cloud run), unless it became
    /// active since [`Runs::resume_config`] looked at it. Returns the updated record and the new `attempt` number.
    pub(super) fn restart(
        &self,
        e: &Arc<RunEntry>,
        wanted: TrainConfig,
        cloud: Option<RunCloud>,
        max_preempt_retries: Option<Option<u32>>,
    ) -> Result<(Run, u64)> {
        let mut st = e.state.lock().unwrap();
        resumable(&st, &e.id)?;
        let last = e.dir.join("weights").join("last.folo");
        let ckpt = if last.is_file() {
            crate::engine::read_model_file(&last).ok().and_then(|f| f.checkpoint().map(|c| (c.epoch, c.best)))
        } else {
            None
        };
        let resume_epoch = if last.is_file() {
            ckpt.map_or(st.file.epochs_done, |(ep, _)| ep.min(u32::MAX as u64) as u32)
        } else {
            0
        };
        let ckpt_best = ckpt.and_then(|(_, b)| {
            let f = b.fitness.map(|v| v as f32)?;
            if !f.is_finite() {
                return None;
            }
            let ep = b.epoch.saturating_add(1).clamp(1, i64::from(u32::MAX)) as u32;
            (ep <= resume_epoch).then_some(BestEpoch { epoch: ep, map50_95: f })
        });
        mark_resume(&e.metrics_path(), resume_epoch);
        remove_samples_after(&e.dir, resume_epoch);
        let retained = super::history::read_metrics(&e.metrics_path());
        let prev_epochs_done = st.file.epochs_done;
        if resume_epoch != prev_epochs_done {
            st.file.epochs_done = resume_epoch;
            st.file.last_metrics = retained
                .epochs
                .last()
                .filter(|ep| ep.epoch == resume_epoch)
                .map(|ep| super::entry::model_metrics(&ep.val));
        }
        st.claimed = cloud.is_some();
        // Not training from its end (or from a resume it has not trained since: it was stopped while it waited) until
        // it trains again: the first part now, the wait in the queue once it starts (`RunEntry::on_event`).
        let at = unix_now().floor();
        let file = &mut st.file;
        if file.run.started.is_some() {
            let since = file.resumed_at.or_else(|| file.run.finished.as_deref().and_then(parse_rfc3339));
            if let Some(since) = since {
                add_paused(&mut file.run, at - since);
            }
            file.resumed_at = Some(at);
        }
        if let Some(updated_retries) = max_preempt_retries {
            file.max_preempt_retries = updated_retries;
        }
        let r = &mut file.run;
        r.epoch = resume_epoch;
        r.step = if resume_epoch == 0 {
            0
        } else {
            retained
                .steps
                .last()
                .filter(|s| s.epoch == resume_epoch)
                .map(|s| s.step)
                .unwrap_or_else(|| u64::from(resume_epoch).saturating_mul(r.steps_per_epoch))
        };
        if resume_epoch == 0 {
            r.best = None;
        } else {
            let mut best = r.best.filter(|b| b.epoch <= resume_epoch);
            for ep in &retained.epochs {
                if best.is_none_or(|b| ep.val.map50_95 > b.map50_95) {
                    best = Some(BestEpoch { epoch: ep.epoch, map50_95: ep.val.map50_95 });
                }
            }
            if let Some(cb) = ckpt_best
                && best.is_none_or(|b| cb.map50_95 > b.map50_95)
            {
                best = Some(cb);
            }
            r.best = best;
        }
        r.config.device = wanted.device;
        r.config.compute = wanted.compute;
        // `cloud` is there exactly for a cloud run: one resumed here is a run of this machine from now on (what its
        // instances cost is in its log and in the ledger).
        r.cloud = cloud;
        r.status = RunStatus::Queued;
        r.finished = None;
        r.error = None;
        r.error_code = None;
        r.eta_s = None;
        st.tail.clear();
        st.outcome = None;
        st.interrupted = false;
        st.attempt += 1;
        let attempt = st.attempt;
        e.stop.store(false, Ordering::Relaxed);
        e.persist(&st);
        e.send(RunEvent::Status(status_event(&st.file.run)));
        let run = st.file.shown();
        drop(st);
        e.log(
            LogLevel::Info,
            if last.is_file() {
                "resuming from weights/last.folo".to_string()
            } else {
                "no epoch finished before; starting over".to_string()
            },
        );
        Ok((run, attempt))
    }
}

/// Refuses to resume a run that is deleted, still active (or claimed) or completed.
fn resumable(st: &RunState, id: &str) -> Result<()> {
    if st.deleted {
        return Err(Error::not_found("run", id));
    }
    let status = st.file.run.status;
    if status.is_active() || st.claimed {
        return Err(Error::conflict("run_active", format!("run '{id}' is still active")));
    }
    if status == RunStatus::Completed {
        return Err(Error::conflict("run_not_resumable", format!("run '{id}' already completed"))
            .hint("start a new run with `weights` set to its model to train further"));
    }
    Ok(())
}

/// Removes any mirrored validation sample files (`samples/epoch_NNN.json`) for epochs strictly after
/// `resume_epoch` so rolled-back epochs are never served after resuming from an earlier checkpoint.
fn remove_samples_after(run_dir: &Path, resume_epoch: u32) {
    let dir = run_dir.join("samples");
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for path in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(num) = name.strip_prefix("epoch_").and_then(|s| s.strip_suffix(".json")) else { continue };
        if let Ok(epoch) = num.parse::<u32>()
            && epoch > resume_epoch
        {
            let _ = std::fs::remove_file(&path);
        }
    }
}

impl RunEntry {
    /// Checks whether `weights/last.folo` in this run's folder holds a valid, incomplete training checkpoint
    /// matching the same schedule, best-epoch consistency, and early-stopping (`patience`) checks enforced by
    /// `RealStore::continue_run`. Returns `Some((resume_epoch, best))` when a replacement cloud worker can
    /// continue training from it.
    pub(crate) fn resumable_checkpoint(&self) -> Option<(u32, Option<BestEpoch>)> {
        let last = self.dir.join("weights").join("last.folo");
        if !last.is_file() {
            return None;
        }
        let file = crate::engine::read_model_file(&last).ok()?;
        let ckpt = file.checkpoint()?;
        let ckpt_cfg = serde_json::from_value::<folo_vision::TrainConfig>(ckpt.train.clone()).ok();
        let done = ckpt.epoch;
        let run_epochs = self.state.lock().unwrap().file.run.epochs as u64;
        let sched_total =
            ckpt_cfg.as_ref().map_or(ckpt.sched.epochs, |c| ckpt.sched.epochs.min(c.epochs as u64));
        let total = sched_total.min(run_epochs);
        if done > i64::MAX as u64 {
            return None;
        }
        if ckpt.best.fitness.is_some()
            && (ckpt.best.epoch < 0 || done == 0 || (ckpt.best.epoch as u64) >= done)
        {
            return None;
        }
        if done >= total {
            return None;
        }
        if done > 0
            && ckpt.best.fitness.is_some()
            && ckpt_cfg
                .as_ref()
                .is_some_and(|c| (done as i64 - 1).saturating_sub(ckpt.best.epoch) >= c.patience as i64)
        {
            return None;
        }
        let resume_epoch = done.min(u32::MAX as u64) as u32;
        let best = ckpt.best.fitness.map(|v| v as f32).filter(|f| f.is_finite()).and_then(|f| {
            let ep = ckpt.best.epoch.saturating_add(1).clamp(1, i64::from(u32::MAX)) as u32;
            (ep <= resume_epoch).then_some(BestEpoch { epoch: ep, map50_95: f })
        });
        Some((resume_epoch, best))
    }

    /// Rolls `metrics.jsonl`, validation sample files, and live `Run` progress (`epoch`, `step`, `best`,
    /// `epochs_done`, `last_metrics`, `eta_s`, `attempt`) back to match a synced checkpoint at `resume_epoch`
    /// before a replacement cloud worker continues training from `weights/last.folo`, keeping `Run.status` active.
    pub(crate) fn rollback_to_checkpoint(&self, resume_epoch: u32, ckpt_best: Option<BestEpoch>) {
        let mut st = self.state.lock().unwrap();
        mark_resume(&self.metrics_path(), resume_epoch);
        remove_samples_after(&self.dir, resume_epoch);
        let retained = super::history::read_metrics(&self.metrics_path());
        if resume_epoch != st.file.epochs_done {
            st.file.epochs_done = resume_epoch;
            st.file.last_metrics = retained
                .epochs
                .last()
                .filter(|ep| ep.epoch == resume_epoch)
                .map(|ep| super::entry::model_metrics(&ep.val));
        }
        let r = &mut st.file.run;
        r.epoch = resume_epoch;
        r.step = if resume_epoch == 0 {
            0
        } else {
            retained
                .steps
                .last()
                .filter(|s| s.epoch == resume_epoch)
                .map(|s| s.step)
                .unwrap_or_else(|| u64::from(resume_epoch).saturating_mul(r.steps_per_epoch))
        };
        if resume_epoch == 0 {
            r.best = None;
        } else {
            let mut best = r.best.filter(|b| b.epoch <= resume_epoch);
            for ep in &retained.epochs {
                if best.is_none_or(|b| ep.val.map50_95 > b.map50_95) {
                    best = Some(BestEpoch { epoch: ep.epoch, map50_95: ep.val.map50_95 });
                }
            }
            if let Some(cb) = ckpt_best
                && best.is_none_or(|b| cb.map50_95 > b.map50_95)
            {
                best = Some(cb);
            }
            r.best = best;
        }
        r.eta_s = None;
        st.tail.clear();
        // Bump `attempt` so any concurrent `Runs::metrics` reader that snapshotted `st.tail` from the dead worker
        // retries instead of merging discarded steps onto the rolled-back history.
        st.attempt += 1;
        self.persist(&st);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::remove_samples_after;
    use crate::Store as _;
    use crate::runs::{DatasetRef, Runs};
    use crate::store::AutoBatchOptions;
    use crate::throughput::Throughput;
    use axum::http::StatusCode;
    use folo_api::{DatasetFormat, RunStatus, SyntheticDataset, TrainConfig};
    use folo_test_support::TestDir;

    #[test]
    fn remove_samples_after_prunes_only_epochs_past_resume_epoch() {
        let dir = TestDir::new("resume-prune-samples");
        let samples = dir.join("samples");
        std::fs::create_dir_all(&samples).unwrap();
        for name in ["epoch_001.json", "epoch_002.json", "epoch_003.json", "notes.txt"] {
            std::fs::write(samples.join(name), b"{}").unwrap();
        }
        remove_samples_after(&dir, 1);
        assert!(samples.join("epoch_001.json").is_file());
        assert!(!samples.join("epoch_002.json").exists());
        assert!(!samples.join("epoch_003.json").exists());
        assert!(samples.join("notes.txt").is_file());
    }

    #[test]
    fn resume_config_validates_auto_batch_fields_and_nulls() {
        let dir = TestDir::new("resume-auto-batch-config");
        let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
        let ds_dir = dir.join("ds");
        std::fs::create_dir_all(&ds_dir).unwrap();
        let dataset = DatasetRef {
            id: "ds_1".into(),
            name: "d".into(),
            path: ds_dir.to_string_lossy().into_owned(),
            format: DatasetFormat::Yolo,
            classes: vec!["c".into()],
        };
        let cfg: TrainConfig =
            serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1", "batch": 24 })).unwrap();
        let ab = AutoBatchOptions { enabled: true, target_utilization: Some(0.75) };
        let opts = crate::store::RunOptions { auto_batch: ab, arch: None, max_preempt_retries: None };
        let entry =
            runs.register_with_options(cfg, dataset, None, None, 0, "cpu".into(), None, opts).unwrap();
        {
            let mut st = entry.state.lock().unwrap();
            st.file.run.status = RunStatus::Stopped;
        }
        let id = entry.id();

        // Explicit `null` for `auto_batch` and `auto_batch_target` counts as omitted.
        let nulls =
            serde_json::json!({ "auto_batch": null, "auto_batch_target": null }).as_object().unwrap().clone();
        runs.resume_config(id, &nulls).unwrap();

        // Matching `batch: 24` + `auto_batch: true` + `auto_batch_target: 0.75` succeeds.
        let same = serde_json::json!({ "batch": 24, "auto_batch": true, "auto_batch_target": 0.75 })
            .as_object()
            .unwrap()
            .clone();
        runs.resume_config(id, &same).unwrap();

        // Repeating matching `batch: 24` alone (omitting `auto_batch`, e.g. echoing `Run.config`) succeeds.
        let repeat_batch = serde_json::json!({ "batch": 24 }).as_object().unwrap().clone();
        runs.resume_config(id, &repeat_batch).unwrap();

        // A differing integer `batch` or `auto_batch: false` is rejected.
        let wrong_batch = serde_json::json!({ "batch": 16 }).as_object().unwrap().clone();
        let err = runs.resume_config(id, &wrong_batch).err().unwrap();
        assert_eq!(err.body.code, "invalid_config");
        assert!(err.body.message.contains("batch"), "{}", err.body.message);

        let disable_auto = serde_json::json!({ "auto_batch": false }).as_object().unwrap().clone();
        let err = runs.resume_config(id, &disable_auto).err().unwrap();
        assert_eq!(err.body.code, "invalid_config");
        assert!(err.body.message.contains("auto_batch"), "{}", err.body.message);

        // Non-auto string `batch` is 422 `invalid_body`.
        let bad = serde_json::json!({ "batch": "huge" }).as_object().unwrap().clone();
        let err = runs.resume_config(id, &bad).err().unwrap();
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.body.code, "invalid_body");

        runs.shutdown(None);
    }

    #[test]
    fn queued_auto_batch_run_cli_estimate_and_cloud_resume_rejection() {
        let dir = TestDir::new("real-store-auto-batch-cli-cloud");
        let store = crate::RealStore::open(dir.to_path_buf()).device("cpu");
        let ds = store
            .create_synthetic(SyntheticDataset {
                name: "ab".into(),
                images: 4,
                classes: vec!["circle".into()],
                imgsz: 64,
                seed: 1,
            })
            .unwrap();
        let cfg: TrainConfig = serde_json::from_value(serde_json::json!({
            "dataset_id": ds.id,
            "epochs": 1,
            "batch": 24,
            "imgsz": 64,
            "device": "cpu"
        }))
        .unwrap();
        let ab = AutoBatchOptions { enabled: true, target_utilization: Some(0.75) };

        // `estimate_run_with_auto_batch` accepts auto-batch and emits the matching CLI flags.
        let est = store.estimate_run_with_auto_batch(cfg.clone(), ab).unwrap();
        let est_cli = est.cli.unwrap();
        assert!(est_cli.contains("--batch 24"), "{est_cli}");
        assert!(est_cli.contains("--set auto_batch=true"), "{est_cli}");
        assert!(est_cli.contains("--auto-batch-target 0.75"), "{est_cli}");

        // Unqueued fixture with no `config.json`: deterministically exercises the persisted `run.json` fallback.
        let dataset = DatasetRef {
            id: ds.id.clone(),
            name: ds.name.clone(),
            path: ds.path.clone(),
            format: ds.format,
            classes: ds.classes.clone(),
        };
        let opts = crate::store::RunOptions { auto_batch: ab, arch: None, max_preempt_retries: None };
        let unqueued = store
            .runs_for_test()
            .register_with_options(cfg.clone(), dataset, None, None, 0, "cpu".into(), None, opts)
            .unwrap();
        assert!(!unqueued.dir().join("config.json").exists(), "fixture must lack config.json");
        let cli = store.run_cli(unqueued.id()).unwrap();
        assert!(cli.cli.contains("--batch 24"), "{}", cli.cli);
        assert!(cli.cli.contains("--set auto_batch=true"), "{}", cli.cli);
        assert!(cli.cli.contains("--auto-batch-target 0.75"), "{}", cli.cli);

        // Stopped auto-batch run rejects cloud resume with `invalid_config`.
        unqueued.state.lock().unwrap().file.run.status = RunStatus::Stopped;
        let cloud_fields = serde_json::json!({
            "device": "auto",
            "compute": { "connection_id": "cc_missing", "offer_id": "off_1", "max_hours": 1.0 }
        })
        .as_object()
        .unwrap()
        .clone();
        let err = store.resume_run(unqueued.id(), &cloud_fields).unwrap_err();
        assert_eq!(err.body.code, "invalid_config");
        assert!(err.body.message.contains("auto_batch"), "{}", err.body.message);
        store.shutdown();
    }
}
