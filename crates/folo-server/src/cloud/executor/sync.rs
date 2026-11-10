//! Syncing the dataset and the starting weights to the worker, and starting the run there.

use std::io::Write;
use std::time::Instant;

use folo_api::{
    CloudPhase, CreateUpload, DatasetFormat, LogLevel, Run, RunList, TransferProgress, UploadKind, Uploaded,
};
use folo_client::Client;
use folo_client::transfer::{self, Control, TransferError};
use folo_data::Dataset as _;
use folo_data::LoadedDataset;

use super::{
    ATTEMPTS, Exec, Halt, PROGRESS_EVERY, Synced, TRANSFER_LOG_EVERY, failed, file_fingerprint, size,
};

impl Exec {
    pub(super) fn sync(&mut self, client: &Client, ds: &LoadedDataset) -> std::result::Result<Synced, Halt> {
        let d = self.entry.dataset();
        let total = folo_data::pack::packed_len(ds)
            .map_err(|e| failed("data_error", format!("measuring the dataset: {e}")))?;
        let images = ds.train.len() + ds.val.len() + ds.test.len();
        self.phase(
            CloudPhase::Syncing,
            format!("uploading the dataset {} ({images} images, {})", d.name, size(total)),
        );
        let req = CreateUpload {
            kind: UploadKind::Dataset,
            name: d.name.clone(),
            format: DatasetFormat::Yolo,
            fingerprint: self.fingerprint.clone(),
            bytes: Some(total),
        };
        let write = |w: &mut dyn Write| {
            folo_data::pack::pack(ds, w).map(drop).map_err(|e| std::io::Error::other(e.to_string()))
        };
        let (uploaded, sent) = self.upload("the dataset", client, &req, write)?;
        let dataset = uploaded.dataset.ok_or_else(|| {
            failed("cloud_sync_failed", "the worker's answer to the dataset upload has no dataset")
        })?;
        if sent == 0 {
            self.entry.log(LogLevel::Info, "the worker already has this dataset; nothing to send");
        }
        let last = self.dir.join("weights").join("last.folo");
        let start = if last.is_file() {
            Some((last, format!("{}-last", self.entry.run().name), true))
        } else {
            self.entry.weights().map(|w| {
                let name = w.file_stem().map_or("weights".into(), |s| s.to_string_lossy().into_owned());
                (w, name, false)
            })
        };
        let model = match start {
            None => None,
            Some((path, name, checkpoint)) => {
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let what =
                    if checkpoint { "the checkpoint weights/last.folo" } else { "the weights to start from" };
                self.entry.log(LogLevel::Info, format!("uploading {what} ({})", size(bytes)));
                let req = CreateUpload {
                    kind: UploadKind::Model,
                    name,
                    format: DatasetFormat::Auto,
                    fingerprint: file_fingerprint(&path),
                    bytes: Some(bytes),
                };
                let write = |w: &mut dyn Write| {
                    std::fs::File::open(&path).and_then(|mut f| std::io::copy(&mut f, w)).map(drop)
                };
                let (uploaded, _) = self.upload(what, client, &req, write)?;
                let m = uploaded.model.ok_or_else(|| {
                    failed("cloud_sync_failed", "the worker's answer to the weights upload has no model")
                })?;
                Some((m.id, checkpoint))
            }
        };
        self.save();
        Ok(Synced { dataset_id: dataset.id, model })
    }

    /// Sends what `write` produces as upload `req` to the worker, with progress in `Run.cloud`; the bytes sent.
    fn upload(
        &self,
        what: &str,
        client: &Client,
        req: &CreateUpload,
        write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
    ) -> std::result::Result<(Uploaded, u64), Halt> {
        let total = req.bytes.unwrap_or(0);
        let mut sent = 0;
        let (mut shown, mut logged) = (None::<Instant>, Instant::now());
        let entry = &self.entry;
        let mut progress = |p: TransferProgress| {
            sent = p.done_bytes;
            if shown.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY) || p.done_bytes >= total {
                entry.update_cloud(|c| c.progress = Some(p));
                shown = Some(Instant::now());
            }
            if logged.elapsed() >= TRANSFER_LOG_EVERY {
                entry.log(
                    LogLevel::Info,
                    format!("uploading {what}: {} of {}", size(p.done_bytes), size(total)),
                );
                logged = Instant::now();
            }
        };
        let stop = || self.interrupted().is_err();
        let mut ctl = Control { progress: &mut progress, stop: &stop, backoff: transfer::FIRST_BACKOFF };
        let up = transfer::upload(client, req, write, &mut ctl);
        match up {
            Ok(u) => Ok((u, sent)),
            Err(TransferError::Stopped) => Err(self.interrupted().err().unwrap_or(Halt::LetGo)),
            Err(e) => Err(failed("cloud_sync_failed", format!("uploading {what}: {e}"))),
        }
    }

    /// Starts the run on the worker: the run's config without `compute`, on the worker's device and with the ids the
    /// worker gave the dataset and the starting weights. A request whose answer was lost is found again by the run's
    /// name there (`<name> (<run id>)`), so the worker never trains it twice.
    pub(super) fn start_worker_run(
        &mut self,
        client: &Client,
        synced: Synced,
    ) -> std::result::Result<(), Halt> {
        let run = self.entry.run();
        let name = format!("{} ({})", run.name, run.id);
        let mut body = serde_json::to_value(&run.config).expect("configs serialize");
        let fields = body.as_object_mut().expect("a config is an object");
        fields.remove("compute");
        fields.insert("name".into(), name.clone().into());
        fields.insert("device".into(), "auto".into());
        fields.insert("dataset_id".into(), synced.dataset_id.into());
        fields.insert("weights".into(), serde_json::Value::Null);
        match synced.model {
            Some((id, true)) => fields.insert("checkpoint".into(), id.into()),
            Some((id, false)) => fields.insert("weights".into(), id.into()),
            None => None,
        };
        // A `remote` server keeps the runs of the run's earlier attempts under the same name: those are not this
        // attempt's. After a request whose answer was lost, a run of the name that is not one of them is the one it
        // created, however it went since (it may have ended already).
        let earlier: Vec<String> = self.named(client, &name)?.into_iter().map(|r| r.id).collect();
        let mut attempt = 1;
        let created = loop {
            match client.post::<_, Run>("/runs", &body) {
                Ok(r) => break r,
                Err(e) if e.is_transient() && attempt < ATTEMPTS => {
                    if let Some(r) = self.named(client, &name)?.into_iter().find(|r| !earlier.contains(&r.id))
                    {
                        break r;
                    }
                    attempt += 1;
                    self.wait(self.core.tuning.poll)?;
                }
                Err(e) => {
                    let code = if e.is_transient() { "cloud_sync_failed" } else { e.code() };
                    return Err(failed(code, format!("starting the run on the worker: {e}")));
                }
            }
        };
        self.private.run_id = Some(created.id.clone());
        (self.private.after_step, self.private.after_epoch, self.private.after_log) = (0, 0, 0);
        self.save();
        self.phase(CloudPhase::Training, format!("the worker took the run as {}", created.id));
        Ok(())
    }

    /// The worker's runs named `name`, asked for again while it fails for a reason of the moment. When it does not
    /// answer, the run fails: without the list, a run started now could be one the worker already trains.
    fn named(&self, client: &Client, name: &str) -> std::result::Result<Vec<Run>, Halt> {
        let mut attempt = 1;
        loop {
            match client.get::<RunList>("/runs") {
                Ok(list) => return Ok(list.runs.into_iter().filter(|r| r.name == name).collect()),
                Err(e) if e.is_transient() && attempt < ATTEMPTS => {
                    attempt += 1;
                    self.wait(self.core.tuning.poll)?;
                }
                Err(e) => {
                    let code = if e.is_transient() { "cloud_sync_failed" } else { e.code() };
                    return Err(failed(code, format!("listing the worker's runs: {e}")));
                }
            }
        }
    }
}
