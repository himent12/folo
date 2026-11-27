//! The first phases: provisioning the instance (or following a run again) and booting its worker.

use std::path::Path;
use std::time::{Duration, Instant};

use folo_api::{CloudInstanceStatus, CloudPhase, Health, LogLevel, SystemResponse};
use folo_client::Client;
use folo_cloud::{ErrorKind, InstanceState, LaunchSpec};
use folo_data::Dataset as _;
use folo_data::LoadedDataset;

use crate::cloud::files::{self};
use crate::cloud::state::Outcome;
use crate::cloud::worker;
use crate::util::rfc3339;

use super::{Endpoint, Exec, GONE_POLLS, GoneSince, Halt, Release, STATE_POLLS, disk_for, failed};

impl Exec {
    // ---- phases ----------------------------------------------------------------------------------------------------

    pub(super) fn start(&mut self, release: &mut Release) -> std::result::Result<Outcome, Halt> {
        let ds = self.open_dataset()?;
        let endpoint = self.provision(&ds, release)?;
        let client = self.boot(endpoint)?;
        let synced = self.sync(&client, &ds)?;
        drop(ds);
        self.start_worker_run(&client, synced)?;
        self.train(&client)
    }

    pub(super) fn recover(&mut self, release: &mut Release) -> std::result::Result<Outcome, Halt> {
        // Repair any torn trailing line left by a crash before any recovery, interruption, or release path appends
        // logs/metrics (this thread already holds `self._lock`, so no other writer is active on these files).
        crate::runs::ensure_trailing_newline(&self.entry.metrics_path());
        crate::runs::ensure_trailing_newline(&self.dir.join("log.jsonl"));
        // Collected before the restart: what is left is the release.
        if let Some(o) = self.private.outcome.clone() {
            return Ok(o);
        }
        // Crashed during preemption recovery backoff (after releasing the old instance, before launching the
        // replacement): honor stop/deadline and continue auto-resume from the synced checkpoint instead of
        // abandoning the run.
        if self.private.recovering_preemption && self.private.instance_id.is_none() {
            if self.is_stopping() {
                self.private.recovering_preemption = false;
                self.save();
                return Err(Halt::Stopped(
                    "stopped during preemption recovery; synced checkpoint weights/last.folo kept".into(),
                ));
            }
            if self.past_deadline() {
                self.private.recovering_preemption = false;
                self.save();
                return Err(Halt::Stopped(format!(
                    "the time limit of {} ended the run during preemption recovery",
                    self.hours()
                )));
            }
            if self.entry.resumable_checkpoint().is_none() {
                self.private.recovering_preemption = false;
                self.save();
                return Err(failed(
                    "cloud_instance_lost",
                    "server restarted during preemption recovery, but no incomplete weights/last.folo is \
                     available to resume from",
                ));
            }
            self.entry.log(
                LogLevel::Info,
                "this server restarted during preemption recovery; continuing auto-resume from weights/last.folo",
            );
            return self.start(release);
        }
        let phase = self.phase_now();
        let (Some(url), Some(_)) = (self.private.worker_url.clone(), &self.private.run_id) else {
            return Err(Halt::Interrupted(phase));
        };
        if phase < CloudPhase::Training {
            return Err(Halt::Interrupted(phase));
        }
        let (token, use_tls_ca) = match self.c.provider.fixed_endpoint(&self.c.connection.values()) {
            Some(e) => (e.token, true),
            None => (self.private.token.clone().unwrap_or_default(), false),
        };
        self.entry.log(
            LogLevel::Info,
            "this server restarted: following the run on its worker again and fetching what it missed",
        );
        let client = worker::client(&url, &token, use_tls_ca).map_err(|e| failed("cloud_boot_failed", e))?;
        self.train(&client)
    }

    /// The local dataset, opened as the local trainer opens it, before anything is rented.
    fn open_dataset(&mut self) -> std::result::Result<LoadedDataset, Halt> {
        let d = self.entry.dataset();
        let opts = folo_data::OpenOptions { format: crate::runs::data_format(d.format), verify_images: true };
        let ds = folo_data::open_with(Path::new(&d.path), &opts)
            .map_err(|e| failed("data_error", format!("opening the dataset {}: {e}", d.path)))?;
        if ds.train.is_empty() {
            return Err(failed("invalid_dataset", "the dataset has no training images"));
        }
        self.private.classes = ds.classes.clone();
        Ok(ds)
    }

    fn provision(
        &mut self,
        ds: &LoadedDataset,
        release: &mut Release,
    ) -> std::result::Result<Endpoint, Halt> {
        let values = self.c.connection.values();
        if let Some(e) = self.c.provider.fixed_endpoint(&values) {
            return Ok(Endpoint::Fixed(e));
        }
        let info = self.c.provider.info();
        let offer = self.offer();
        let compute = self.entry.run().config.compute.expect("a cloud run has compute");
        let len = folo_data::pack::packed_len(ds)
            .map_err(|e| failed("data_error", format!("measuring the dataset: {e}")))?;
        let disk_gb = compute.disk_gb.unwrap_or_else(|| disk_for(len));
        // Recheck shutdown/stop/deadline after opening and measuring the dataset so an accepted cancellation never
        // proceeds into a paid launch.
        self.interrupted()?;
        self.entry
            .log(LogLevel::Info, format!("launching {} at {} ({disk_gb} GB disk)", offer.gpu, info.name));
        let internal = |e: std::io::Error| failed("internal", e.to_string());
        let token = files::random_hex(32).map_err(internal)?;
        let remaining = self.ensure_deadline()?;
        let deadline = self.private.deadline.map(rfc3339);
        let record = self
            .core
            .ledger
            .create(&self.c.connection.id, &info.id, self.entry.id(), &offer, deadline, Some(remaining))
            .map_err(|e| failed(&e.body.code, e.body.message))?;
        // Nothing could terminate an instance launched on a connection removed since the run started. Removing one
        // looks for live records under the cloud lock this one was written under, so from here on it is refused.
        if let Err(e) = self.core.connected(&self.c.connection.id) {
            self.core.ledger.remove(&record.id);
            return Err(failed(&e.body.code, e.body.message));
        }
        release.arm(&record.id);
        self.private.recovering_preemption = false;
        self.private.instance_id = Some(record.id.clone());
        self.private.token = Some(token.clone());
        self.save();
        self.entry.update_cloud(|c| c.instance_id = Some(record.id.clone()));
        let spec = LaunchSpec {
            name: record.name.clone(),
            offer_id: offer.id.clone(),
            disk_gb,
            token,
            lifetime_s: remaining.as_secs().max(1),
            worker_base_url: folo_cloud::worker_base_url(&values, env!("CARGO_PKG_VERSION")),
        };
        let nothing = |release: &mut Release, e: folo_cloud::Error| {
            self.core.ledger.remove(&record.id);
            release.disarm();
            self.entry.update_cloud(|c| c.instance_id = None);
            failed(e.code(), e.message)
        };
        let provider_ref = match self.c.provider.launch(&values, &spec) {
            Ok(r) => r,
            // The request may have gone through and only its answer been lost.
            Err(e) if e.kind == ErrorKind::Unavailable => match self.c.provider.find(&values, &spec.name) {
                Ok(refs) if !refs.is_empty() => {
                    self.entry.log(
                        LogLevel::Warn,
                        format!("the launch request failed ({e}), but the instance exists; using it"),
                    );
                    // A name is unique to one launch, so more than one is a provider's retry: those go.
                    for extra in &refs[1..] {
                        if let Err(e) = self.c.provider.terminate(&values, extra) {
                            self.entry
                                .log(LogLevel::Warn, format!("terminating the extra instance {extra}: {e}"));
                        }
                    }
                    refs[0].clone()
                }
                // Not listed (yet) or unknown: the release looks for it by its name again, and the reaper after
                // that (crate::cloud::reaper::LISTED_WITHIN).
                Ok(_) | Err(_) => return Err(failed(e.code(), e.message)),
            },
            Err(e) => return Err(nothing(release, e)),
        };
        let record = self
            .core
            .ledger
            .update(&record.id, |i| {
                i.provider_ref = Some(provider_ref.clone());
                if i.status == CloudInstanceStatus::Launching {
                    i.status = CloudInstanceStatus::Starting;
                }
            })
            .map_err(|e| failed(&e.body.code, e.body.message))?;
        if !matches!(record.status, CloudInstanceStatus::Starting) {
            // Terminated by hand while the launch was in flight: the instance it created goes too.
            let _ = self.core.ledger.update(&record.id, |i| {
                i.status = CloudInstanceStatus::Terminating;
                i.ended = None;
            });
            return Err(self.lost_by_hand());
        }
        self.interrupted()?;
        Ok(Endpoint::Instance(provider_ref))
    }

    fn boot(&mut self, endpoint: Endpoint) -> std::result::Result<Client, Halt> {
        let provisioned = matches!(endpoint, Endpoint::Instance(_));
        let timeout =
            if provisioned { self.core.tuning.boot_timeout } else { self.core.tuning.remote_boot_timeout };
        let boot_failed = |m: String| failed("cloud_boot_failed", m);
        let minutes = |d: Duration| match d.as_secs() {
            s if s >= 120 => format!("{} minutes", s / 60),
            s => format!("{s} s"),
        };
        let t0 = Instant::now();
        let values = self.c.connection.values();
        // A provider may not show an instance it has only just created: gone (or stopped) is believed once it has
        // been the answer for a while, without another answer in between.
        let mut gone = GoneSince::default();
        let believe_after = self.core.tuning.poll * GONE_POLLS;
        let (url, token, instance) = match endpoint {
            Endpoint::Fixed(e) => {
                self.phase(CloudPhase::Booting, format!("connecting to the FOLO server at {}", e.url));
                self.arm_deadline();
                (e.url, e.token, None)
            }
            Endpoint::Instance(r) => {
                self.phase(CloudPhase::Booting, "waiting for the instance to start");
                let url = loop {
                    self.interrupted()?;
                    match self.c.provider.state(&values, &r) {
                        Ok(state) if gone.believed(&state, believe_after) => {
                            // Terminated by hand meanwhile: lost, not a boot failure.
                            self.interrupted()?;
                            return Err(boot_failed("the instance stopped before it started".into()));
                        }
                        Ok(InstanceState::Running { url }) => break url,
                        Ok(_) => {}
                        Err(e) => tracing::debug!(run = %self.entry.id(), "instance state: {e}"),
                    }
                    if t0.elapsed() >= timeout {
                        return Err(boot_failed(format!(
                            "the instance did not start within {}",
                            minutes(timeout)
                        )));
                    }
                    self.wait(self.core.tuning.poll)?;
                };
                (url, self.private.token.clone().unwrap_or_default(), Some(r))
            }
        };
        if provisioned {
            self.entry.log(
                LogLevel::Info,
                format!(
                    "the instance runs; it installs FOLO (a few minutes), then its worker answers at {url}"
                ),
            );
        }
        let client = worker::client(&url, &token, !provisioned).map_err(boot_failed)?;
        let mut asked = Instant::now();
        let (health, system) = loop {
            self.interrupted()?;
            let answer = client
                .get::<Health>("/health")
                .and_then(|h| Ok((h, client.get::<SystemResponse>("/system")?)));
            match answer {
                Ok(x) => break x,
                Err(e) if e.status.map(|s| s.as_u16()) == Some(401) => {
                    return Err(boot_failed(format!("the worker at {url} rejected the token")));
                }
                Err(e) if t0.elapsed() >= timeout => {
                    return Err(boot_failed(format!(
                        "the worker at {url} did not answer within {} ({e})",
                        minutes(timeout)
                    )));
                }
                Err(_) => {}
            }
            // An instance whose setup fails terminates itself (docs/CLOUD.md, `runpod`): it is gone long before
            // the timeout, and the run says so instead of waiting for a worker that will never answer.
            if let Some(r) =
                instance.as_ref().filter(|_| asked.elapsed() >= self.core.tuning.poll * STATE_POLLS)
            {
                asked = Instant::now();
                match self.c.provider.state(&values, r) {
                    Ok(state) if gone.believed(&state, believe_after) => {
                        self.interrupted()?;
                        return Err(boot_failed(self.setup_failed(&values)));
                    }
                    Ok(_) => {}
                    Err(e) => tracing::debug!(run = %self.entry.id(), "instance state: {e}"),
                }
            }
            self.wait(self.core.tuning.poll)?;
        };
        let amp = self.entry.run().config.amp;
        worker::check(&system, &health.version, provisioned && self.core.tuning.require_gpu, amp)
            .map_err(boot_failed)?;
        if let Some(id) = &self.private.instance_id {
            let _ = self.core.ledger.update(id, |i| {
                if i.status == CloudInstanceStatus::Starting {
                    i.status = CloudInstanceStatus::Running;
                }
            });
            // Its setup has started, so its own clock runs: the latest it can live is known from here.
            let _ = self.core.ledger.booted(id);
        }
        let device = system.capabilities.map_or_else(|| "its default device".to_string(), |c| c.device);
        self.entry
            .log(LogLevel::Info, format!("the worker answers: FOLO {} training on {device}", health.version));
        self.private.worker_url = Some(url);
        self.save();
        Ok(client)
    }

    /// Why an instance that is gone before its worker answered most likely is: its setup failed and it terminated
    /// itself. Names where it installs FOLO from, the step that fails when a release is not published.
    fn setup_failed(&self, values: &folo_cloud::Values) -> String {
        format!(
            "the instance stopped before its worker answered: its setup failed, and it terminated itself. It \
             installs FOLO from {} (the connection's worker download URL names another place); the instance's log \
             at {} says which step failed",
            folo_cloud::worker_base_url(values, env!("CARGO_PKG_VERSION")),
            self.c.provider.info().name
        )
    }
}
