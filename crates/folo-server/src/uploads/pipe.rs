//! Streaming bridge between HTTP upload part requests and a dataset session's tar unpacking thread.

use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use axum::body::Bytes;

use crate::error::{Error, Result};

use super::truncated;
use super::unpack::unpack;

/// How a stream's end went ([`Pipe::finish`]).
pub(super) enum Finish {
    /// The archive (or file) is whole.
    Done,
    /// The archive has not ended yet; the session stays open for more parts.
    Incomplete(Error),
    /// It cannot be used; the session is dropped.
    Failed(Error),
}

/// Ensures [`Pipe`] waiters are always woken with an error if the unpacking thread unwinds unexpectedly.
pub(super) struct UnpackGuard {
    pipe: Arc<Pipe>,
    completed: bool,
}

impl UnpackGuard {
    pub(super) fn new(pipe: Arc<Pipe>) -> UnpackGuard {
        UnpackGuard { pipe, completed: false }
    }

    pub(super) fn complete(mut self, outcome: Result<()>) {
        self.completed = true;
        let mut f = self.pipe.lock();
        f.reading = None;
        f.outcome = Some(outcome);
        self.pipe.changed.notify_all();
    }
}

impl Drop for UnpackGuard {
    fn drop(&mut self) {
        if !self.completed {
            let mut f = self.pipe.lock();
            f.reading = None;
            if f.outcome.is_none() {
                f.outcome = Some(Err(Error::internal("upload unpacking thread aborted unexpectedly")));
            }
            self.pipe.changed.notify_all();
        }
    }
}

/// A dataset session's unpacking thread. Dropping it stops the thread and waits for it, so the session's files
/// are closed when its folder is removed.
pub(super) struct Unpacker {
    pipe: Arc<Pipe>,
    thread: Option<JoinHandle<()>>,
}

impl Unpacker {
    pub(super) fn start(id: &str, root: &Path) -> Result<Unpacker> {
        let pipe = Arc::new(Pipe::default());
        let (p, root) = (pipe.clone(), root.to_path_buf());
        let thread = std::thread::Builder::new()
            .name(format!("folo-{id}"))
            .spawn(move || {
                let guard = UnpackGuard::new(p.clone());
                let outcome = unpack(&p, &root);
                guard.complete(outcome);
            })
            .map_err(|e| Error::internal(format!("starting the unpacking thread: {e}")))?;
        Ok(Unpacker { pipe, thread: Some(thread) })
    }

    pub(super) fn pipe(&self) -> &Arc<Pipe> {
        &self.pipe
    }
}

impl Drop for Unpacker {
    fn drop(&mut self) {
        self.pipe.cancel();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Hands one part at a time from the request that received it to the unpacking thread.
#[derive(Default)]
pub(super) struct Pipe {
    flow: Mutex<Flow>,
    changed: Condvar,
}

#[derive(Default)]
struct Flow {
    /// The part being read and how far.
    chunk: Option<Bytes>,
    pos: usize,
    /// The stream is complete: the reader sees its end after the last part.
    closed: bool,
    /// The session was dropped: the reader fails.
    cancelled: bool,
    /// The file whose bytes the thread is reading, if it is inside one.
    reading: Option<String>,
    /// The last file the thread read.
    last: Option<String>,
    /// The end-of-archive marker arrived: the archive is whole (what follows it is padding).
    ended: bool,
    /// How the thread ended.
    outcome: Option<Result<()>>,
}

impl Pipe {
    fn lock(&self) -> MutexGuard<'_, Flow> {
        self.flow.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Gives `bytes` to the thread and waits until it has unpacked all of them, or refused the archive.
    pub(super) fn feed(&self, bytes: Bytes) -> Result<()> {
        let mut f = self.lock();
        if let Some(Err(e)) = &f.outcome {
            return Err(e.clone());
        }
        if f.cancelled {
            return Err(Error::internal("the upload was dropped"));
        }
        f.chunk = Some(bytes);
        f.pos = 0;
        self.changed.notify_all();
        while f.chunk.is_some() && f.outcome.is_none() && !f.cancelled {
            f = self.changed.wait(f).unwrap_or_else(|p| p.into_inner());
        }
        match &f.outcome {
            Some(Err(e)) => Err(e.clone()),
            _ if f.cancelled => Err(Error::internal("the upload was dropped")),
            _ => Ok(()),
        }
    }

    /// Ends the stream once the archive is whole (its end-of-archive marker arrived) and waits for the thread.
    /// Before that the archive is incomplete and the stream stays open.
    pub(super) fn finish(&self) -> Finish {
        let mut f = self.lock();
        if let Some(Err(e)) = &f.outcome {
            return Finish::Failed(e.clone());
        }
        if f.cancelled {
            return Finish::Failed(Error::internal("the upload was dropped"));
        }
        if !f.ended {
            return Finish::Incomplete(truncated(match (&f.reading, &f.last) {
                (Some(file), _) => format!("the stream ends inside {file}"),
                (None, Some(file)) => format!("the stream ends after {file}, before the end of the archive"),
                (None, None) => "the stream ends before the archive's first file".into(),
            }));
        }
        f.closed = true;
        self.changed.notify_all();
        while f.outcome.is_none() {
            f = self.changed.wait(f).unwrap_or_else(|p| p.into_inner());
        }
        match &f.outcome {
            Some(Err(e)) => Finish::Failed(e.clone()),
            _ => Finish::Done,
        }
    }

    pub(super) fn cancel(&self) {
        self.lock().cancelled = true;
        self.changed.notify_all();
    }

    /// The thread starts reading file `name`.
    pub(super) fn reading(&self, name: &str) {
        self.lock().reading = Some(name.to_string());
    }

    /// The thread has read the file it was reading.
    pub(super) fn read(&self) {
        let mut f = self.lock();
        f.last = f.reading.take();
    }

    /// The end-of-archive marker arrived.
    pub(super) fn end(&self) {
        self.lock().ended = true;
    }
}

/// The thread's end of a [`Pipe`].
pub(super) struct PipeReader(pub(super) Arc<Pipe>);

impl Read for PipeReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let pipe = &self.0;
        let mut f = pipe.lock();
        loop {
            if f.cancelled {
                return Err(std::io::Error::other("the upload was dropped"));
            }
            if let Some(chunk) = &f.chunk {
                if f.pos < chunk.len() {
                    let n = buf.len().min(chunk.len() - f.pos);
                    buf[..n].copy_from_slice(&chunk[f.pos..f.pos + n]);
                    f.pos += n;
                    return Ok(n);
                }
                // Every byte of the part was read, and so unpacked: its request can be answered.
                f.chunk = None;
                pipe.changed.notify_all();
            }
            if f.closed {
                return Ok(0);
            }
            f = pipe.changed.wait(f).unwrap_or_else(|p| p.into_inner());
        }
    }
}
