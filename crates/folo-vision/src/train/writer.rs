//! The epoch's files written off the training thread: `weights/last.folo`, `best.folo` when the epoch improved,
//! and the validation samples.
//!
//! The training thread takes the checkpoint's snapshot (every tensor downloaded to the host, the bytes
//! [`Trainer::save_checkpoint`] writes) and hands it to a writer thread, which writes each file to a temporary
//! name, syncs it and renames it over the previous one, so a crash or stop mid-write leaves the previous checkpoint
//! intact. The next epoch trains meanwhile. The promise of docs/API.md stays: when the `epoch` event is delivered,
//! that epoch's files are on disk. Until the write is done every event (the next epoch's steps and logs) is held
//! in order behind the epoch event; the training thread releases them at its next step after the writer finished
//! (or waits for it at the end of the run, before the next epoch's write, and when the run stops or fails). A write
//! that fails fails the run there, and the events held behind it are dropped, as if the run had failed at the
//! epoch's end. One write at a time: a second epoch end waits for the first write.

use std::path::PathBuf;
use std::thread::JoinHandle;

use folo_api::RunEvent;
use folo_nn::io::FileContents;

use super::Trainer;
use crate::error::{Error, Result};
use crate::val::ValSample;

/// What the writer thread writes for one epoch.
pub(super) struct EpochFiles {
    last: PathBuf,
    /// `best.folo`, a copy of `last` (the epoch improved).
    best: Option<PathBuf>,
    contents: FileContents,
    /// `samples/epoch_NNN.json` and its text.
    samples: (PathBuf, String),
}

/// A write in flight, the epoch event waiting for it, and what it saves (for [`Trainer::last_saved`]).
pub(super) struct PendingWrite {
    handle: JoinHandle<Result<()>>,
    event: RunEvent,
    last: PathBuf,
    epoch: u32,
}

impl EpochFiles {
    fn write(self) -> Result<()> {
        #[cfg(test)]
        tests::WRITERS
            .lock()
            .unwrap()
            .get_or_insert_with(Default::default)
            .insert(self.last.clone(), std::thread::current().id());
        let (map, tensors) = &self.contents;
        folo_nn::io::safetensors::write_file(&self.last, Some(map), tensors)?;
        if let Some(best) = &self.best {
            let (tmp, mut f) = folo_nn::io::safetensors::create_temp_beside(best)?;
            // The file is closed before the rename (on wasm `File` has nothing to close, hence the allow).
            #[cfg_attr(target_arch = "wasm32", allow(clippy::drop_non_drop))]
            let copied = std::fs::File::open(&self.last)
                .and_then(|mut src| std::io::copy(&mut src, &mut f))
                .and_then(|_| f.sync_all())
                .and_then(|_| {
                    drop(f);
                    std::fs::rename(&tmp, best)
                });
            if let Err(err) = copied {
                let _ = std::fs::remove_file(&tmp);
                return Err(Error::io(best, err));
            }
        }
        let (p, text) = &self.samples;
        if let Some(sdir) = p.parent() {
            std::fs::create_dir_all(sdir).map_err(|err| Error::io(sdir, err))?;
        }
        std::fs::write(p, text).map_err(|err| Error::io(p, err))
    }
}

impl Trainer {
    /// Epoch `e`'s files (`weights/last.folo`, `best.folo` when `improved`, the samples) with the checkpoint's
    /// snapshot taken now, at the epoch boundary; `None` without a run folder. Waits for the previous epoch's write
    /// first (one write at a time).
    pub(super) fn epoch_files(
        &mut self,
        e: u32,
        improved: bool,
        samples: &[ValSample],
    ) -> Result<Option<EpochFiles>> {
        self.settle_write()?;
        let Some(dir) = &self.opts_dir else { return Ok(None) };
        let weights = dir.join("weights");
        let body = serde_json::json!({
            "epoch": e + 1,
            "classes": &self.data.classes,
            "samples": samples,
        });
        let samples = (
            dir.join("samples").join(format!("epoch_{:03}.json", e + 1)),
            serde_json::to_string(&body).expect("samples serialise"),
        );
        Ok(Some(EpochFiles {
            last: weights.join("last.folo"),
            best: improved.then(|| weights.join("best.folo")),
            contents: self.checkpoint_snapshot()?,
            samples,
        }))
    }

    /// Writes `files` on the writer thread and delivers `event` once they are on disk (without files, now). When
    /// no thread can be started (in the browser, or out of resources) the files are written here.
    pub(super) fn write_epoch_files(&mut self, files: Option<EpochFiles>, event: RunEvent) -> Result<()> {
        let Some(files) = files else {
            self.emitter.emit(event);
            return Ok(());
        };
        let (last, epoch) = (files.last.clone(), self.epoch);
        let files = std::sync::Arc::new(std::sync::Mutex::new(Some(files)));
        let take = files.clone();
        let spawned = std::thread::Builder::new()
            .name("folo-checkpoint-writer".into())
            .spawn(move || take.lock().ok().and_then(|mut f| f.take()).map_or(Ok(()), EpochFiles::write));
        match spawned {
            Ok(handle) => {
                self.emitter.hold();
                self.pending_write = Some(PendingWrite { handle, event, last, epoch });
            }
            Err(_) => {
                // The closure never ran: the files are still there.
                if let Some(f) = files.lock().ok().and_then(|mut f| f.take()) {
                    f.write()?;
                }
                self.last_saved = Some((last, epoch));
                self.emitter.emit(event);
            }
        }
        Ok(())
    }

    /// Delivers a finished write's epoch event and the events held behind it; nothing while the write runs. Called
    /// at every step.
    pub(super) fn poll_write(&mut self) -> Result<()> {
        if self.pending_write.as_ref().is_some_and(|p| p.handle.is_finished()) {
            self.settle_write()?;
        }
        Ok(())
    }

    /// Waits for the write in flight, if any. Done: its epoch event, then the held events, are delivered. Failed:
    /// the held events are dropped and the error is returned (the run fails as if at the epoch's end).
    pub(super) fn settle_write(&mut self) -> Result<()> {
        let Some(p) = self.pending_write.take() else { return Ok(()) };
        let done =
            p.handle.join().unwrap_or_else(|_| Err(Error::Failed("the checkpoint writer panicked".into())));
        match done {
            Ok(()) => {
                self.last_saved = Some((p.last, p.epoch));
                self.emitter.release(Some(p.event));
                Ok(())
            }
            Err(e) => {
                self.emitter.release_dropping();
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::thread::ThreadId;

    use folo_api::RunEvent;
    use folo_test_support::TestDir;

    use super::super::TrainOptions;
    use super::super::tests::micro;

    /// The thread that wrote each `last.folo`.
    pub(super) static WRITERS: Mutex<Option<HashMap<PathBuf, ThreadId>>> = Mutex::new(None);

    /// A run in `dir` whose events go to `seen` as `epoch N`, `step N` (the step's epoch) and `status <s>`.
    fn run_in(
        dir: &Path,
        epochs: u32,
        check: bool,
    ) -> (crate::error::Result<super::super::TrainSummary>, Vec<String>) {
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let (log, rd) = (seen.clone(), dir.to_path_buf());
        let sink = move |e: &RunEvent| {
            let line = match e {
                RunEvent::Epoch(ep) => {
                    if check {
                        // The promise of docs/API.md: the epoch's files are on disk when its event arrives.
                        let f = folo_nn::io::ModelFile::open(rd.join("weights/last.folo")).unwrap();
                        assert_eq!(f.checkpoint().unwrap().epoch, u64::from(ep.epoch));
                        assert!(rd.join(format!("samples/epoch_{:03}.json", ep.epoch)).is_file());
                    }
                    format!("epoch {}", ep.epoch)
                }
                RunEvent::Step(s) => format!("step {}", s.epoch),
                RunEvent::Status(s) => format!("status {:?}", s.status),
                _ => return,
            };
            log.lock().unwrap().push(line);
        };
        let opts = TrainOptions {
            run_dir: Some(dir.to_path_buf()),
            events: Some(Box::new(sink)),
            step_interval: std::time::Duration::ZERO,
            ..TrainOptions::default()
        };
        let mut t = micro(epochs, 100, opts);
        let r = t.run();
        let seen = seen.lock().unwrap().clone();
        (r, seen)
    }

    /// The epoch's files are written on another thread, are on disk (with that epoch's state) when the epoch event
    /// arrives, and the events keep their order: every step of epoch n + 1 after epoch n's event.
    #[test]
    fn epoch_files_are_written_off_the_training_thread_before_their_event() {
        WRITERS.lock().unwrap().get_or_insert_with(HashMap::new);
        let dir = TestDir::new("train-writer");
        let (r, seen) = run_in(&dir, 3, true);
        assert_eq!(r.unwrap().epochs_completed, 3);
        let epochs: Vec<&String> = seen.iter().filter(|l| l.starts_with("epoch")).collect();
        assert_eq!(epochs, ["epoch 1", "epoch 2", "epoch 3"]);
        for (i, line) in seen.iter().enumerate() {
            if let Some(n) = line.strip_prefix("step ").and_then(|n| n.parse::<u32>().ok())
                && n > 1
            {
                assert!(
                    seen[..i].contains(&format!("epoch {}", n - 1)),
                    "step of epoch {n} before epoch {}: {seen:?}",
                    n - 1
                );
            }
        }
        assert_eq!(seen.last().map(String::as_str), Some("status Completed"), "{seen:?}");
        let writers = WRITERS.lock().unwrap();
        let writer = writers.as_ref().and_then(|w| w.get(&dir.join("weights").join("last.folo"))).copied();
        assert!(writer.is_some_and(|id| id != std::thread::current().id()), "written on a writer thread");
        // metrics.jsonl keeps the same order.
        let text = std::fs::read_to_string(dir.join("metrics.jsonl")).unwrap();
        let kinds: Vec<&str> = text
            .lines()
            .filter_map(|l| l.split('"').nth(3))
            .filter(|k| *k == "epoch" || *k == "step")
            .collect();
        let first_epoch = kinds.iter().position(|k| *k == "epoch").unwrap();
        assert!(kinds[..first_epoch].iter().all(|k| *k == "step"));
    }

    /// A write that fails fails the run as if at the epoch's end: no epoch event, the steps held behind it dropped,
    /// a failed status.
    #[test]
    fn a_failed_write_fails_the_run_without_its_epoch_event() {
        let dir = TestDir::new("train-writer-fail");
        // `last.folo` is a non-empty folder: renaming the written file over it fails.
        std::fs::create_dir_all(dir.join("weights").join("last.folo").join("x")).unwrap();
        let (r, seen) = run_in(&dir, 3, false);
        assert!(r.is_err(), "{seen:?}");
        assert!(!seen.iter().any(|l| l.starts_with("epoch")), "{seen:?}");
        assert!(!seen.iter().any(|l| l == "step 2" || l == "step 3"), "{seen:?}");
        assert_eq!(seen.last().map(String::as_str), Some("status Failed"), "{seen:?}");
    }

    /// The snapshot owns its tensors: training on (which updates the live CPU weights, BatchNorm statistics and
    /// optimizer state in place) leaves it as it was, so the writer thread writes the boundary's state.
    #[test]
    fn the_snapshot_does_not_change_when_training_goes_on() {
        let bytes = |t: &[(String, folo_core::Tensor)]| -> Vec<(String, Vec<u8>)> {
            t.iter().map(|(n, x)| (n.clone(), x.to_cpu_storage().unwrap().as_bytes().to_vec())).collect()
        };
        let mut t = micro(1, 100, TrainOptions::default());
        let (_, snap) = t.checkpoint_snapshot().unwrap();
        let before = bytes(&snap);
        t.run().unwrap();
        assert_eq!(bytes(&snap), before, "training changed the snapshot");
        let (_, fresh) = micro(1, 100, TrainOptions::default()).checkpoint_snapshot().unwrap();
        assert_eq!(bytes(&fresh), before);
        let (_, trained) = t.checkpoint_snapshot().unwrap();
        assert_ne!(bytes(&trained), before, "the run trained");
    }

    /// The epoch's `last.folo` holds the same bytes as `save_checkpoint` at that boundary (but the creation time).
    #[test]
    fn epoch_checkpoint_matches_save_checkpoint() {
        let dir = TestDir::new("train-writer-bytes");
        let opts = TrainOptions { run_dir: Some(dir.to_path_buf()), ..TrainOptions::default() };
        let mut t = micro(2, 100, opts);
        t.run().unwrap();
        let other = dir.join("again.folo");
        t.save_checkpoint(&other).unwrap();
        let blank = |mut b: Vec<u8>| {
            let key = b"folo.created";
            if let Some(i) = b.windows(key.len()).position(|w| w == key) {
                b[i..i + key.len() + 26].fill(0);
            }
            b
        };
        let (a, b) =
            (std::fs::read(dir.join("weights").join("last.folo")).unwrap(), std::fs::read(&other).unwrap());
        assert_eq!(a.len(), b.len());
        assert!(blank(a) == blank(b), "the epoch checkpoint differs from save_checkpoint's");
    }
}
