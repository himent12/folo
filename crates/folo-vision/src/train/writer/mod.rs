//! The epoch's files written off the training thread: `weights/last.folo`, `best.folo` when the epoch improved,
//! and the validation samples.
//!
//! The training thread takes the checkpoint's snapshot (every tensor downloaded to the host, the bytes
//! [`Trainer::save_checkpoint`] writes) and hands it to a writer thread, which writes each file to a temporary
//! name, syncs it and renames it over the previous one, so a crash or stop mid-write leaves the previous files
//! intact. The next epoch trains meanwhile. The promise of docs/API.md stays: when the `epoch` event is delivered,
//! that epoch's files are on disk. Until the write is done every event (the next epoch's steps and logs) is held
//! in order behind the epoch event; the training thread releases them at its next step after the writer finished
//! (or waits for it at the end of the run, before the next epoch's write, when the run stops or fails, and at a
//! boundary where [`TrainOptions::after_epoch`](super::TrainOptions::after_epoch) pauses). A write that fails fails
//! the run there, and the events held behind it are dropped, as if the run had failed at the epoch's end. One write
//! at a time: a second epoch end waits for the first write. A trainer dropped with a write in flight (a panic
//! unwinding mid-epoch) waits for it without delivering any event, so nothing writes into the run folder after.

use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use folo_api::RunEvent;
use folo_nn::io::FileContents;

use super::Trainer;
use crate::error::{Error, Result};
use crate::val::ValSample;

#[cfg(test)]
mod tests;

/// What the writer thread writes for one epoch.
pub(super) struct EpochFiles {
    last: PathBuf,
    /// `best.folo`, a copy of `last` (the epoch improved).
    best: Option<PathBuf>,
    contents: FileContents,
    /// `samples/epoch_NNN.json` and its text.
    samples: (PathBuf, String),
    /// Test support: holds the write pending.
    #[cfg(test)]
    gate: Option<std::sync::Arc<tests::Gate>>,
    /// Test support: written on the training thread, as when no thread can be started.
    #[cfg(test)]
    inline: bool,
}

/// A write in flight, the epoch event waiting for it, and what it saves (for [`Trainer::last_saved`]).
pub(super) struct PendingWrite {
    thread: WriterThread,
    event: RunEvent,
    last: PathBuf,
    epoch: u32,
}

/// The writer thread, joined when dropped unjoined (its trainer went away mid-write) without delivering anything.
struct WriterThread {
    handle: Option<JoinHandle<Result<()>>>,
    #[cfg(test)]
    gate: Option<std::sync::Arc<tests::Gate>>,
}

impl WriterThread {
    fn finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }

    fn join(&mut self) -> Result<()> {
        match self.handle.take() {
            Some(h) => {
                h.join().unwrap_or_else(|_| Err(Error::Failed("the checkpoint writer panicked".into())))
            }
            None => Ok(()),
        }
    }
}

impl Drop for WriterThread {
    fn drop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        #[cfg(test)]
        if let Some(g) = &self.gate {
            g.dropped();
        }
        let _ = self.join();
    }
}

impl EpochFiles {
    fn write(self) -> Result<()> {
        #[cfg(test)]
        tests::wrote(&self);
        let (map, tensors) = &self.contents;
        folo_nn::io::safetensors::write_file(&self.last, Some(map), tensors)?;
        if let Some(best) = &self.best {
            promote_best(&self.last, best)?;
        }
        let (p, text) = &self.samples;
        if let Some(sdir) = p.parent() {
            std::fs::create_dir_all(sdir).map_err(|err| Error::io(sdir, err))?;
        }
        write_atomic(p, |f| std::io::Write::write_all(f, text.as_bytes()))
    }
}

/// Promotes `last` to `best` atomically.
///
/// On Unix, prefers an O(1) same-filesystem hardlink (`hard_link` to a sibling temporary path followed by
/// [`folo_nn::io::rename_over_with`]) and falls back to an atomic byte copy when hardlinks are unavailable
/// (cross-device, unsupported filesystem, etc.). Safe because [`folo_nn::io::safetensors::write_file`] always
/// writes `last` to a fresh temporary file and atomically renames it over `last`, leaving any previous inode
/// linked by `best` immutable.
///
/// On Windows, uses an independent atomic copy so a reader holding `best.folo` open without
/// `FILE_SHARE_DELETE` cannot share an NTFS file object with `last.folo` and block future epoch writes.
fn promote_best(last: &Path, best: &Path) -> Result<()> {
    #[cfg(not(windows))]
    if let Ok(tmp) = try_hardlink_temp(last, best) {
        if folo_nn::io::rename_over_with(&tmp, best, folo_nn::io::RenameRetry::BRIEF).is_ok() {
            return Ok(());
        }
        let _ = std::fs::remove_file(&tmp);
    }
    write_atomic(best, |f| std::fs::File::open(last).and_then(|mut src| std::io::copy(&mut src, f).map(drop)))
}

#[cfg(not(windows))]
fn try_hardlink_temp(src: &Path, dst: &Path) -> std::io::Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = dst.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let name = dst.file_name().and_then(|n| n.to_str()).unwrap_or("best.folo");
    let pid = std::process::id();
    for _ in 0..8 {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!(".{name}.link-{pid}-{seq}.tmp"));
        match std::fs::hard_link(src, &tmp) {
            Ok(()) => return Ok(tmp),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate unique hardlink temp path",
    ))
}

/// Writes `path` through `fill` to a temporary file beside it, synced, then renamed over it.
fn write_atomic(path: &Path, fill: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>) -> Result<()> {
    let (tmp, mut f) = folo_nn::io::safetensors::create_temp_beside(path)?;
    // The file is closed before the rename (on wasm `File` has nothing to close, hence the allow).
    #[cfg_attr(target_arch = "wasm32", allow(clippy::drop_non_drop))]
    let written = fill(&mut f).and_then(|_| f.sync_all()).and_then(|_| {
        drop(f);
        folo_nn::io::rename_over(&tmp, path)
    });
    written.map_err(|err| {
        let _ = std::fs::remove_file(&tmp);
        Error::io(path, err)
    })
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
        let last = weights.join("last.folo");
        #[cfg(test)]
        let (inline, gate) = tests::knobs(&last, e + 1);
        Ok(Some(EpochFiles {
            last,
            best: improved.then(|| weights.join("best.folo")),
            contents: self.checkpoint_snapshot()?,
            samples,
            #[cfg(test)]
            gate,
            #[cfg(test)]
            inline,
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
        #[cfg(test)]
        let (inline, gate) = (files.inline, files.gate.clone());
        #[cfg(not(test))]
        let inline = false;
        let files = std::sync::Arc::new(std::sync::Mutex::new(Some(files)));
        let take = files.clone();
        let spawned = if inline {
            Err(std::io::Error::other("test support: written on the training thread"))
        } else {
            std::thread::Builder::new()
                .name("folo-checkpoint-writer".into())
                .spawn(move || take.lock().ok().and_then(|mut f| f.take()).map_or(Ok(()), EpochFiles::write))
        };
        match spawned {
            Ok(handle) => {
                self.emitter.hold();
                let thread = WriterThread {
                    handle: Some(handle),
                    #[cfg(test)]
                    gate,
                };
                self.pending_write = Some(PendingWrite { thread, event, last, epoch });
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
        let Some(p) = &self.pending_write else { return Ok(()) };
        if p.thread.finished() {
            return self.settle_write();
        }
        #[cfg(test)]
        if let Some(g) = &p.thread.gate {
            g.polled();
        }
        Ok(())
    }

    /// Waits for the write in flight, if any. Done: its epoch event, then the held events, are delivered. Failed:
    /// the held events are dropped and the error is returned (the run fails as if at the epoch's end).
    pub(super) fn settle_write(&mut self) -> Result<()> {
        let Some(mut p) = self.pending_write.take() else { return Ok(()) };
        match p.thread.join() {
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
