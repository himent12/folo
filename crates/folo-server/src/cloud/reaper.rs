//! Terminating instances, and the reaper: one thread that, at start and then every [`super::Tuning::reaper_every`],
//! looks at the instances of runs no process follows: it retries the terminations the provider has not confirmed,
//! terminates the instances that are still alive (also a `launching` record a crash left: looked up by its name),
//! keeps looking for the instances that were never found ([`LISTED_WITHIN`]) and prunes old records. A process
//! follows a run while it holds the run's `cloud.lock` and releases the run's instance itself, so an instance whose
//! launch is in flight (in this process or another) is never touched.

use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_api::{CloudInstance, CloudInstanceStatus};
use folo_cloud::InstanceState;

use super::ledger::Paces;
use super::{Core, files};
use crate::error::Result;
use crate::util::{parse_rfc3339, unix_now};

/// A provider may list an instance only a while after the launch request that created it (one whose answer was
/// lost may still have been in the works). An instance without a provider id is looked for by its name in two
/// stages:
/// - until this long after its record was written, finding none leaves it `terminating`, looked for at every
///   sweep;
/// - after that it ends (`terminated`, at no cost: nobody saw it) and stays marked, and the reaper keeps looking
///   for it: every [`super::Tuning::unseen_every`] until its watchdog is due, every
///   [`super::Tuning::unseen_late_every`] after (a watchdog can fail), for as long as an ended record is kept, and
///   once more at the end; a look that fails does not count. One that shows up is terminated as any other, and
///   costs what it ran from its launch ([`super::ledger::Ledger::unseen`]).
pub(crate) const LISTED_WITHIN: Duration = Duration::from_secs(5 * 60);

/// Whether a record written at `created` was written less than [`LISTED_WITHIN`] before `now`. A time that does not
/// parse, or lies in the future (a clock that went back), cannot say how long ago that was: not recent, so the
/// record moves on to the slower watch, which ends.
fn recent(created: &str, now: f64) -> bool {
    parse_rfc3339(created).is_some_and(|t| t <= now && now - t < LISTED_WITHIN.as_secs_f64())
}

impl Core {
    /// Asks the provider once to terminate instance `id`: `terminated` when it confirms (also when the instance was
    /// gone already), else `terminating` with the error, for the reaper to retry. An instance without a provider id
    /// (its launch is in flight, or a crash interrupted it) is looked up by its name; when there is none, see
    /// [`LISTED_WITHIN`].
    pub(crate) fn terminate_once(&self, id: &str) -> Result<CloudInstance> {
        let record = self.ledger.get(id)?;
        if !record.status.is_alive() {
            return Ok(record);
        }
        let record = self.ledger.update(id, |i| {
            if i.status.is_alive() {
                i.status = CloudInstanceStatus::Terminating;
            }
        })?;
        let now = unix_now();
        // `None`: never found.
        let terminated = self.connected(&record.connection_id).map_err(|e| e.body.message).and_then(|c| {
            let values = c.connection.values();
            let refs = match &record.provider_ref {
                Some(r) => vec![r.clone()],
                None => c.provider.find(&values, &record.name).map_err(|e| e.message)?,
            };
            if refs.is_empty() && record.provider_ref.is_none() {
                if recent(&record.created, now) {
                    return Err(format!(
                        "{} lists no instance named {} yet; looking again",
                        c.provider.info().name,
                        record.name
                    ));
                }
                return Ok(None);
            }
            // Whether the provider still had it: an instance that was gone before this ended at a time nobody
            // saw, one that is terminated here ends now (the ledger tells them apart for its cost).
            let was_gone =
                refs.iter().all(|r| matches!(c.provider.state(&values, r), Ok(InstanceState::Gone)));
            refs.iter().try_for_each(|r| c.provider.terminate(&values, r)).map_err(|e| e.message)?;
            Ok(Some(was_gone))
        });
        match terminated {
            Ok(Some(was_gone)) => {
                tracing::info!(instance = %id, "terminated the cloud instance");
                self.ledger.terminated(id, was_gone)
            }
            Ok(None) => {
                tracing::info!(instance = %id, name = %record.name, "the provider never listed the instance");
                self.ledger.unseen(id, now)
            }
            Err(message) => {
                tracing::warn!(instance = %id, "terminating the cloud instance failed (retried): {message}");
                self.ledger.update(id, |i| i.error = Some(message))
            }
        }
    }

    /// [`Core::terminate_once`] until the provider confirms, shutdown halts the loop, or `wait` has passed; the
    /// record as it is then.
    pub(crate) fn terminate_within(&self, id: &str, wait: Duration) -> Result<CloudInstance> {
        let t0 = Instant::now();
        let mut pause = self.tuning.poll;
        loop {
            let record = self.terminate_once(id)?;
            if !record.status.is_alive() || self.halt.is_on() || t0.elapsed() >= wait {
                return Ok(record);
            }
            let remaining = pause.min(wait.saturating_sub(t0.elapsed()));
            if self.halt.pause(remaining) {
                return Ok(record);
            }
            pause = (pause * 2).min(self.tuning.poll_max);
        }
    }

    /// One pass of the reaper (see the module docs).
    pub(crate) fn sweep(&self) {
        for record in self.ledger.list() {
            if self.halt.is_on() {
                return;
            }
            if !record.status.is_alive() {
                self.look_for_unseen(&record);
                continue;
            }
            // Followed: its run's process releases it. (A run folder that is gone is followed by nobody.)
            let run_dir = record.run_id.as_deref().map(|r| self.run_dir(r));
            match run_dir.as_deref().map(files::try_lock_run).transpose() {
                Ok(Some(None)) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(instance = %record.id, "checking who follows its run: {e}");
                    continue;
                }
                Ok(_) => {}
            }
            if let Err(e) = self.terminate_once(&record.id) {
                tracing::warn!(instance = %record.id, "terminating it: {}", e.body.message);
            }
        }
    }

    /// The second stage of [`LISTED_WITHIN`] for `record`, when it is one whose instance was never found and it is
    /// time to look again: an instance that shows up is terminated through the normal path; a look that fails
    /// (the connection is gone, the provider does not answer) is the record's error.
    fn look_for_unseen(&self, record: &CloudInstance) {
        let id = &record.id;
        let paces = Paces { fast: self.tuning.unseen_every, slow: self.tuning.unseen_late_every };
        let at = unix_now();
        match self.ledger.due(id, at, paces) {
            Ok(true) => {}
            Ok(false) => return,
            Err(e) => {
                tracing::warn!(instance = %id, "looking for it: {}", e.body.message);
                return;
            }
        }
        let found = self
            .connected(&record.connection_id)
            .map_err(|e| e.body.message)
            .and_then(|c| c.provider.find(&c.connection.values(), &record.name).map_err(|e| e.message));
        self.answered(record, at, found);
    }

    /// What the look for the never found instance of `record` taken at `at` ([`super::ledger::Ledger::due`])
    /// answered: the provider ids of the instances its name lists, or why there was no answer. An instance that was
    /// found is terminated, whatever another look's answer did to the record meanwhile: a look that found it wins.
    fn answered(&self, record: &CloudInstance, at: f64, found: std::result::Result<Vec<String>, String>) {
        let id = &record.id;
        let looked = match found {
            Ok(refs) if refs.is_empty() => self.ledger.looked(id, at, Ok(())),
            Ok(refs) => {
                tracing::warn!(
                    instance = %id,
                    "the provider lists the instance {} ({}) after all, though it never confirmed its launch: \
                     terminating it",
                    record.name,
                    refs.join(", ")
                );
                // One is its provider id from now on; several (a provider's retries) are found by the name again.
                let one = (refs.len() == 1).then(|| refs[0].clone());
                match self.ledger.reopen(id, one) {
                    // The record went (the watch ended with another look's empty answer, and it was pruned):
                    // nothing would terminate them, so this does.
                    Err(e) if e.body.code == "instance_not_found" => {
                        self.terminate_unrecorded(record, &refs);
                        return;
                    }
                    reopened => reopened.and_then(|_| self.terminate_once(id)),
                }
            }
            // Does not count: looked for again at the next sweep.
            Err(message) => self.ledger.looked(
                id,
                at,
                Err(format!(
                    "looking for the instance, which its provider never listed: {message}; check the provider's \
                     console"
                )),
            ),
        };
        if let Err(e) = looked {
            tracing::warn!(instance = %id, "looking for it: {}", e.body.message);
        }
    }
}

impl Core {
    /// Terminates the instances `refs` of `record`, whose ledger record is gone: once, with the outcome in the
    /// server's log (there is no record left to keep retrying from).
    fn terminate_unrecorded(&self, record: &CloudInstance, refs: &[String]) {
        let terminated = self.connected(&record.connection_id).map_err(|e| e.body.message).and_then(|c| {
            let values = c.connection.values();
            refs.iter().try_for_each(|r| c.provider.terminate(&values, r)).map_err(|e| e.message)
        });
        match terminated {
            Ok(()) => {
                tracing::warn!(instance = %record.id, "terminated {} ({}) after its record went", record.name, refs.join(", "))
            }
            Err(e) => tracing::error!(
                instance = %record.id,
                "could not terminate {} ({}), whose record went: {e}; terminate it in the provider's console",
                record.name,
                refs.join(", ")
            ),
        }
    }
}

/// Starts the reaper thread; it sweeps at once, then every `reaper_every`, until the process shuts down.
pub(crate) fn spawn(core: std::sync::Arc<Core>) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new().name("folo-cloud-reaper".into()).spawn(move || {
        loop {
            core.sweep();
            if core.halt.pause(core.tuning.reaper_every) {
                return;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::KEEP_ENDED;
    use crate::util::rfc3339;
    use folo_api::{CloudAvailability, CloudField, CloudOffer, CloudProvider, ConnectCloud};
    use folo_cloud::{Account, Provider, Values};
    use std::sync::{Arc, Mutex};

    /// A provider whose instances are always running, recording what it is asked to terminate.
    #[derive(Default)]
    struct Recording(Mutex<Vec<String>>);

    impl Provider for Recording {
        fn info(&self) -> CloudProvider {
            let key = CloudField {
                key: "api_key".into(),
                label: "Key".into(),
                secret: true,
                required: true,
                placeholder: None,
                help: None,
            };
            CloudProvider {
                id: "rec".into(),
                name: "Recording".into(),
                about: String::new(),
                docs_url: None,
                fields: vec![key],
                provisions: true,
                self_terminates: false,
            }
        }
        fn check(&self, _: &Values) -> folo_cloud::Result<Account> {
            Ok(Account::default())
        }
        fn offers(&self, _: &Values) -> folo_cloud::Result<Vec<CloudOffer>> {
            Ok(Vec::new())
        }
        fn state(&self, _: &Values, _: &str) -> folo_cloud::Result<InstanceState> {
            Ok(InstanceState::Running { url: "http://w".into() })
        }
        fn find(&self, _: &Values, _: &str) -> folo_cloud::Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn terminate(&self, _: &Values, r: &str) -> folo_cloud::Result<()> {
            self.0.lock().unwrap().push(r.to_string());
            Ok(())
        }
    }

    #[test]
    fn a_look_that_found_the_instance_wins_over_an_empty_answer_that_ended_the_watch() {
        let dir = folo_test_support::TestDir::new("cloud-reaper-found-late");
        let provider = Arc::new(Recording::default());
        let cloud = super::super::Cloud::with(&dir, vec![provider.clone()], super::super::Tuning::default());
        let values = [("api_key".to_string(), "key-0123456789".to_string())].into_iter().collect();
        let conn =
            cloud.connect(ConnectCloud { provider: "rec".into(), name: String::new(), values }).unwrap();
        let core = &cloud.core;
        let offer = CloudOffer {
            id: "o".into(),
            gpu: "G".into(),
            gpu_count: 1,
            vram_bytes: 0,
            price_per_hour: Some(3.6),
            region: None,
            detail: None,
            availability: CloudAvailability::High,
        };
        let paces = Paces { fast: Duration::from_secs(600), slow: Duration::from_secs(3600) };
        // Never found, launched `age` ago; two looks at its end overlap (a look that outlasted its hold): one
        // answers empty and ends the watch, then the other answers that the provider lists it.
        let overlapping = |run: &str, age: f64, pod: &str| {
            let now = unix_now().floor();
            let i = core.ledger.create(&conn.id, "rec", run, &offer, None, None).unwrap();
            core.ledger.update(&i.id, |i| i.created = rfc3339(now - age)).unwrap();
            let i = core.ledger.unseen(&i.id, now).unwrap();
            let end = now.max(now - age + KEEP_ENDED.as_secs_f64());
            assert!(core.ledger.due(&i.id, end, paces).unwrap());
            core.answered(&i, end, Ok(Vec::new()));
            core.ledger.list();
            core.answered(&i, end - 1.0, Ok(vec![pod.to_string()]));
            i.id
        };
        // An hour old: the record is reopened and the instance terminated, with what it ran.
        let id = overlapping("run_1", 3600.0, "pod-1");
        assert_eq!(*provider.0.lock().unwrap(), ["pod-1"]);
        let i = core.ledger.get(&id).unwrap();
        assert_eq!((i.status, i.provider_ref.as_deref()), (CloudInstanceStatus::Terminated, Some("pod-1")));
        assert!(i.cost_usd.unwrap() >= 3.6 - 1e-3, "{i:?}");
        // Eight days old: the empty answer let the record go, and the instance found is terminated all the same.
        overlapping("run_2", 8.0 * 86_400.0, "pod-2");
        assert_eq!(*provider.0.lock().unwrap(), ["pod-1", "pod-2"]);
    }

    #[test]
    fn recent_is_bounded_whatever_the_record_says() {
        let now = 1_800_000_000.0;
        assert!(recent(&rfc3339(now - 240.0), now), "4 minutes");
        assert!(!recent(&rfc3339(now - 360.0), now), "6 minutes");
        assert!(!recent("yesterday", now), "a time that does not parse is not recent for ever");
        assert!(!recent(&rfc3339(now + 60.0), now), "nor is one in the future");
        assert!(recent(&rfc3339(now), now));
    }

    #[test]
    fn terminate_within_and_sweep_exit_promptly_on_halt() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FailingTerminate {
            calls: AtomicUsize,
            first_called: Mutex<Option<std::sync::mpsc::Sender<()>>>,
        }
        impl Provider for FailingTerminate {
            fn info(&self) -> CloudProvider {
                CloudProvider {
                    id: "fail".into(),
                    name: "Failing".into(),
                    about: String::new(),
                    docs_url: None,
                    fields: vec![CloudField {
                        key: "api_key".into(),
                        label: "Key".into(),
                        secret: true,
                        required: true,
                        placeholder: None,
                        help: None,
                    }],
                    provisions: true,
                    self_terminates: false,
                }
            }
            fn check(&self, _: &Values) -> folo_cloud::Result<Account> {
                Ok(Account::default())
            }
            fn offers(&self, _: &Values) -> folo_cloud::Result<Vec<CloudOffer>> {
                Ok(Vec::new())
            }
            fn terminate(&self, _: &Values, _: &str) -> folo_cloud::Result<()> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if let Some(tx) = self.first_called.lock().unwrap().take() {
                    let _ = tx.send(());
                }
                Err(folo_cloud::Error::new(folo_cloud::ErrorKind::Unavailable, "provider busy"))
            }
        }

        let dir = folo_test_support::TestDir::new("cloud-reaper-halt");
        let (tx, rx) = std::sync::mpsc::channel();
        let provider =
            Arc::new(FailingTerminate { calls: AtomicUsize::new(0), first_called: Mutex::new(Some(tx)) });
        let tuning = super::super::Tuning {
            poll: Duration::from_secs(5),
            poll_max: Duration::from_secs(10),
            release_wait: Duration::from_secs(10),
            ..super::super::Tuning::default()
        };
        let cloud = super::super::Cloud::with(&dir, vec![provider.clone()], tuning);
        let values = [("api_key".to_string(), "key-0123456789".to_string())].into_iter().collect();
        let conn =
            cloud.connect(ConnectCloud { provider: "fail".into(), name: String::new(), values }).unwrap();
        let offer = CloudOffer {
            id: "o".into(),
            gpu: "G".into(),
            gpu_count: 1,
            vram_bytes: 0,
            price_per_hour: None,
            region: None,
            detail: None,
            availability: CloudAvailability::High,
        };
        let inst = cloud.core.ledger.create(&conn.id, "fail", "run_1", &offer, None, None).unwrap();
        cloud.core.ledger.update(&inst.id, |i| i.provider_ref = Some("pod-1".into())).unwrap();

        let core = cloud.core.clone();
        let id = inst.id.clone();
        let t0 = Instant::now();
        let handle = std::thread::spawn(move || core.terminate_within(&id, Duration::from_secs(10)));
        rx.recv_timeout(Duration::from_secs(2)).expect("terminate_once was not invoked");
        cloud.core.halt.set();
        let rec = handle.join().unwrap().unwrap();
        assert!(t0.elapsed() < Duration::from_secs(2), "terminate_within stalled for {:?}", t0.elapsed());
        assert_eq!(rec.status, CloudInstanceStatus::Terminating);

        let before = provider.calls.load(Ordering::SeqCst);
        cloud.core.sweep();
        assert_eq!(provider.calls.load(Ordering::SeqCst), before, "sweep must stop immediately when halted");
    }
}
