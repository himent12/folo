//! Upload sessions of the real store (`/api/uploads`, docs/API.md *Uploads*): a byte stream cut into parts that
//! arrive one at a time, in order, into the session's folder `<data dir>/uploads/<id>/`.
//!
//! - `mod.rs`: [`Uploads`], [`Finished`], per-session state, expiry sweeper, and shared upload error helpers.
//! - `fingerprints.rs`: persisted `<data dir>/uploads.json` fingerprint memory with in-memory caching.
//! - `pipe.rs`: [`pipe::Unpacker`] and [`pipe::Pipe`] streaming one part at a time to a dataset unpacking thread.
//! - `unpack.rs`: streaming tar archive extraction, path-traversal / collision checks, and partial-file cleanup.
//! - `tests.rs`: unit tests for upload sessions, fingerprints, streaming pipe, and archive validation.
//!
//! - A dataset's stream is a tar archive, unpacked as it arrives by one thread per session: a part is handed to the
//!   thread through a [`pipe::Pipe`] that holds that part only, and its request is answered once the thread has
//!   written every byte of it (or refused the archive), so memory stays at about one part and a hostile entry fails
//!   the part that carries it. Only regular files and folders with relative paths inside the session folder are
//!   accepted, and the archive is whole once its end-of-archive marker arrived (completing it earlier is a conflict
//!   that leaves the session open).
//! - A model's stream is one `.folo` file, written as it arrives.
//!
//! [`crate::RealStore`] moves what a completed session holds into place and registers it ([`Uploads::finish`]
//! hands it over). Sessions live in memory: one without a part for the idle limit ([`UPLOAD_IDLE`]) expires, a
//! dropped session takes its folder with it, and the folder of every session is removed on start. Uploads that had
//! a fingerprint are remembered in `<data dir>/uploads.json` (written atomically), so a later upload of the same
//! content is answered with what it became.

mod fingerprints;
mod pipe;
#[cfg(test)]
mod tests;
mod unpack;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::http::StatusCode;
use folo_api::{CreateUpload, DatasetFormat, Upload, UploadKind};

use crate::error::{Error, Result};
use crate::util::{fnv1a, rfc3339, unique_id, unix_now};

use fingerprints::{Fingerprints, Known};
use pipe::{Finish, Unpacker};

/// The largest part a session takes (`Upload.max_part_bytes`; the part route's body limit).
pub const MAX_PART_BYTES: usize = 32 * 1024 * 1024;
/// A session without a part for this long expires.
pub const UPLOAD_IDLE: Duration = Duration::from_secs(3600);
/// Longest fingerprint a session takes.
pub const MAX_FINGERPRINT_CHARS: usize = 128;
/// The file a model upload is written to, inside its session folder.
const MODEL_FILE: &str = "model.folo";

/// The upload sessions and the remembered fingerprints of one data dir. See the module docs.
pub struct Uploads {
    dir: PathBuf,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    idle_ms: AtomicU64,
    known: Fingerprints,
    sweeper: Arc<SweeperGate>,
    sweeper_thread: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Default)]
struct SweeperState {
    closed: bool,
    generation: u64,
}

/// Coordinates the upload expiry sweeper so it sleeps with zero idle wakeups and exits immediately on shutdown.
#[derive(Default)]
struct SweeperGate {
    state: Mutex<SweeperState>,
    wake: Condvar,
}

impl SweeperGate {
    fn notify(&self) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.generation = st.generation.wrapping_add(1);
        self.wake.notify_all();
    }

    fn close(&self) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.closed = true;
        st.generation = st.generation.wrapping_add(1);
        self.wake.notify_all();
    }
}

/// Removes a newly created staging directory if setup returns an error or unwinds before [`Session`] owns it.
struct DirGuard {
    dir: Option<PathBuf>,
}

impl DirGuard {
    fn new(dir: PathBuf) -> DirGuard {
        DirGuard { dir: Some(dir) }
    }

    fn disarm(mut self) -> PathBuf {
        self.dir.take().expect("dir guard already disarmed")
    }
}

impl Drop for DirGuard {
    fn drop(&mut self) {
        if let Some(dir) = self.dir.take() {
            remove_dir(&dir);
        }
    }
}

/// One session. Its folder is removed when the session is dropped.
struct Session {
    id: String,
    kind: UploadKind,
    name: String,
    format: DatasetFormat,
    fingerprint: Option<String>,
    dir: PathBuf,
    /// Held while a part is received, so the parts of one session are taken one at a time.
    state: Mutex<Progress>,
}

/// Where a session's stream is.
struct Progress {
    /// Index of the next part.
    part: u32,
    received: u64,
    /// Length of the last part received (a retry of it is accepted when it has this length).
    last_len: Option<u64>,
    touched: Instant,
    /// Where the bytes go; `None` once the session ended (completed, dropped or expired).
    sink: Option<Sink>,
}

enum Sink {
    /// A dataset's tar archive, unpacked by a thread.
    Archive(Unpacker),
    /// A model's file.
    File(std::fs::File),
}

/// What a completed session holds, for [`crate::RealStore`] to move into place. The session's folder (and anything
/// left in it) is removed when this is dropped.
pub struct Finished {
    session: Arc<Session>,
}

impl Finished {
    /// What was uploaded.
    pub fn kind(&self) -> UploadKind {
        self.session.kind
    }

    /// The dataset's or model's name.
    pub fn name(&self) -> &str {
        &self.session.name
    }

    /// The dataset's format (`kind: dataset`).
    pub fn format(&self) -> DatasetFormat {
        self.session.format
    }

    /// The fingerprint the upload was started with.
    pub fn fingerprint(&self) -> Option<&str> {
        self.session.fingerprint.as_deref()
    }

    /// The unpacked dataset's folder (`kind: dataset`).
    pub fn root(&self) -> &Path {
        &self.session.dir
    }

    /// The uploaded weights file (`kind: model`).
    pub fn file(&self) -> PathBuf {
        self.session.dir.join(MODEL_FILE)
    }
}

/// `404 upload_not_found`.
pub(super) fn not_found(id: &str) -> Error {
    Error::new(StatusCode::NOT_FOUND, "upload_not_found", format!("upload '{id}' does not exist"))
        .hint("it may have expired (an hour without a part) or been completed; start a new one with POST /api/uploads")
}

/// `422 invalid_upload`.
pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_upload", message).hint(
        "a dataset upload is a tar archive of the dataset's folder: regular files and folders with relative paths, \
         no links or devices",
    )
}

/// `409 upload_conflict` for a stream that ends before the archive does (`message` says where).
pub(super) fn truncated(message: String) -> Error {
    Error::conflict("upload_conflict", message)
        .hint("send the rest of the stream (GET /api/uploads/{id} names the next part), then complete it")
}

/// `500 io_error` for a filesystem operation on `path`.
pub(super) fn io_error(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error::io(format!("{what} {}: {e}", path.display()))
}

impl Uploads {
    /// The uploads of `data_dir`. Session folders a previous process left in `<data dir>/uploads/` and any stale
    /// atomic temp files of `uploads.json` are removed. A thread expires idle sessions while the returned value lives.
    pub fn open(data_dir: &Path) -> Arc<Uploads> {
        let dir = data_dir.join("uploads");
        if let Err(e) = std::fs::remove_dir_all(&dir)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("removing the leftover uploads in {}: {e}", dir.display());
        }
        let known = Fingerprints::new(data_dir.join("uploads.json"));
        known.cleanup_stale_temps();
        let sweeper = Arc::new(SweeperGate::default());
        let uploads = Arc::new(Uploads {
            dir,
            sessions: Mutex::default(),
            idle_ms: AtomicU64::new(UPLOAD_IDLE.as_millis() as u64),
            known,
            sweeper: sweeper.clone(),
            sweeper_thread: Mutex::default(),
        });
        if let Some(handle) = spawn_sweeper(Arc::downgrade(&uploads), sweeper) {
            *uploads.sweeper_thread.lock().unwrap() = Some(handle);
        }
        uploads
    }

    /// Changes how long a session may go without a part before it expires (default [`UPLOAD_IDLE`]).
    pub fn set_idle(&self, idle: Duration) {
        self.idle_ms.store(idle.as_millis() as u64, Ordering::Relaxed);
        self.sweeper.notify();
    }

    fn idle(&self) -> Duration {
        Duration::from_millis(self.idle_ms.load(Ordering::Relaxed))
    }

    /// `400 invalid_body` for a request no session can be made of: a blank name, a dataset format for a model, a
    /// fingerprint over [`MAX_FINGERPRINT_CHARS`].
    pub fn check(req: &CreateUpload) -> Result<()> {
        let bad = |m: &str| Error::new(StatusCode::BAD_REQUEST, "invalid_body", m);
        if req.name.trim().is_empty() {
            return Err(bad("name must not be empty"));
        }
        if req.kind == UploadKind::Model && req.format != DatasetFormat::Auto {
            return Err(bad("format only applies to kind dataset"));
        }
        if req.fingerprint.as_ref().is_some_and(|f| f.chars().count() > MAX_FINGERPRINT_CHARS) {
            return Err(bad(&format!("fingerprint is longer than {MAX_FINGERPRINT_CHARS} characters")));
        }
        Ok(())
    }

    /// A new session for `req` ([`Uploads::check`] first), with its folder and, for a dataset, its unpacking thread.
    pub fn create(&self, req: &CreateUpload) -> Result<Upload> {
        Uploads::check(req)?;
        std::fs::create_dir_all(&self.dir).map_err(|e| io_error("creating", &self.dir, e))?;
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        let seed = fnv1a(req.name.as_bytes()) ^ unix_now().to_bits();
        let id = unique_id("up", seed, |id| sessions.contains_key(id) || self.dir.join(id).exists());
        let dir = self.dir.join(&id);
        std::fs::create_dir(&dir).map_err(|e| io_error("creating", &dir, e))?;
        let guard = DirGuard::new(dir.clone());
        let sink = match req.kind {
            UploadKind::Dataset => Unpacker::start(&id, &dir).map(Sink::Archive),
            UploadKind::Model => {
                let path = dir.join(MODEL_FILE);
                std::fs::File::create(&path).map(Sink::File).map_err(|e| io_error("creating", &path, e))
            }
        }?;
        let progress =
            Progress { part: 0, received: 0, last_len: None, touched: Instant::now(), sink: Some(sink) };
        let session = Arc::new(Session {
            id: id.clone(),
            kind: req.kind,
            name: req.name.trim().to_string(),
            format: req.format,
            fingerprint: req.fingerprint.clone(),
            dir: guard.disarm(),
            state: Mutex::new(progress),
        });
        let upload = session.upload(&lock(&session.state), self.idle());
        sessions.insert(id, session);
        drop(sessions);
        self.sweeper.notify();
        tracing::info!(upload = %upload.id, kind = ?req.kind, name = %upload.name, "upload started");
        Ok(upload)
    }

    /// Session `id` as the API shows it.
    pub fn get(&self, id: &str) -> Result<Upload> {
        self.live(id, |s, st| Ok(s.upload(st, self.idle())))
    }

    /// Takes part `part` of session `id`: the next part, or a repeat of the last one with its length (a retry after
    /// a lost answer, which changes nothing). Anything else is `409 upload_conflict`. A part the content of which is
    /// refused (`422 invalid_upload`) or cannot be written (`500 io_error`) drops the session.
    pub fn put(&self, id: &str, part: u32, bytes: Bytes) -> Result<Upload> {
        if bytes.len() > MAX_PART_BYTES {
            return Err(Error::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                format!("a part holds at most {MAX_PART_BYTES} bytes, this one {}", bytes.len()),
            )
            .hint("send the stream in smaller parts: Upload.max_part_bytes is the largest part this server takes"));
        }
        self.live(id, |s, st| {
            let len = bytes.len() as u64;
            if part == st.part {
                let Some(next_part) = st.part.checked_add(1) else {
                    return Err(Error::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "payload_too_large",
                        "upload exceeded the maximum number of parts",
                    ));
                };
                let Some(next_received) = st.received.checked_add(len) else {
                    return Err(Error::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "payload_too_large",
                        "upload exceeded the maximum total byte count",
                    ));
                };
                let written = match st.sink.as_mut() {
                    Some(Sink::Archive(u)) => u.pipe().feed(bytes),
                    Some(Sink::File(f)) => {
                        f.write_all(&bytes).map_err(|e| io_error("writing", &s.dir.join(MODEL_FILE), e))
                    }
                    None => Err(not_found(id)),
                };
                if let Err(e) = written {
                    tracing::warn!(upload = %id, "upload dropped at part {part}: {e}");
                    s.close(st);
                    self.forget(s);
                    return Err(e);
                }
                st.part = next_part;
                st.received = next_received;
                st.last_len = Some(len);
            } else if part.checked_add(1) != Some(st.part) || st.last_len != Some(len) {
                return Err(Error::conflict(
                    "upload_conflict",
                    format!("upload '{id}' expects part {}, not part {part} ({len} bytes)", st.part),
                )
                .hint(format!(
                    "send part {} next (GET /api/uploads/{id} says where the stream is)",
                    st.part
                )));
            }
            st.touched = Instant::now();
            Ok(s.upload(st, self.idle()))
        })
    }

    /// Ends session `id`'s stream and hands over what it holds (the session leaves the list). A dataset archive
    /// that has not ended yet is `409 upload_conflict` and the session stays open (more parts may follow); one that
    /// cannot be read to its end drops the session.
    pub fn finish(&self, id: &str) -> Result<Finished> {
        self.live(id, |s, st| {
            let ended = match st.sink.as_mut() {
                Some(Sink::Archive(u)) => u.pipe().finish(),
                Some(Sink::File(f)) => match f.sync_all() {
                    Ok(()) => Finish::Done,
                    Err(e) => Finish::Failed(io_error("writing", &s.dir.join(MODEL_FILE), e)),
                },
                None => Finish::Failed(not_found(id)),
            };
            match ended {
                Finish::Done => {
                    // Ends the thread or closes the file; the folder stays for the caller.
                    st.sink = None;
                    let session = self.forget(s).ok_or_else(|| not_found(id))?;
                    Ok(Finished { session })
                }
                Finish::Incomplete(e) => Err(e),
                Finish::Failed(e) => {
                    s.close(st);
                    self.forget(s);
                    Err(e)
                }
            }
        })
    }

    /// Drops session `id` and what it received.
    pub fn delete(&self, id: &str) -> Result<()> {
        let session = self
            .sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(id)
            .ok_or_else(|| not_found(id))?;
        session.close(&mut lock(&session.state));
        Ok(())
    }

    /// Stops the sweeper thread and drops every session (ending their threads and removing their folders).
    pub fn shutdown(&self) {
        self.stop_sweeper();
        let sessions: Vec<_> =
            self.sessions.lock().unwrap_or_else(|p| p.into_inner()).drain().map(|(_, s)| s).collect();
        for s in sessions {
            s.close(&mut lock(&s.state));
        }
    }

    fn stop_sweeper(&self) {
        self.sweeper.close();
        if let Some(handle) = self.sweeper_thread.lock().unwrap_or_else(|p| p.into_inner()).take()
            && handle.thread().id() != std::thread::current().id()
        {
            let _ = handle.join();
        }
    }

    /// Drops the sessions that went without a part for the idle limit (not one receiving a part right now).
    fn expire(&self) {
        let idle = self.idle();
        let sessions: Vec<_> =
            self.sessions.lock().unwrap_or_else(|p| p.into_inner()).values().cloned().collect();
        for s in sessions {
            let Ok(mut st) = s.state.try_lock() else { continue };
            if st.sink.is_some() && st.touched.elapsed() >= idle {
                tracing::info!(upload = %s.id, "upload expired");
                s.close(&mut st);
                self.forget(&s);
            }
        }
    }

    /// Runs `f` on live session `id` with its state locked: `404 upload_not_found` for an unknown, ended or expired
    /// session (an expired one is dropped here, should the sweeper not have come by yet).
    fn live<T>(&self, id: &str, f: impl FnOnce(&Arc<Session>, &mut Progress) -> Result<T>) -> Result<T> {
        let session = self
            .sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(id)
            .cloned()
            .ok_or_else(|| not_found(id))?;
        let mut st = lock(&session.state);
        if st.sink.is_none() {
            return Err(not_found(id));
        }
        if st.touched.elapsed() >= self.idle() {
            session.close(&mut st);
            self.forget(&session);
            return Err(not_found(id));
        }
        f(&session, &mut st)
    }

    /// Removes `session` from the list (when it is still the one listed under its id) and returns it.
    fn forget(&self, session: &Arc<Session>) -> Option<Arc<Session>> {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        match sessions.get(&session.id) {
            Some(s) if Arc::ptr_eq(s, session) => sessions.remove(&session.id),
            _ => None,
        }
    }

    /// What a finished upload of `kind` with `fingerprint` became (a dataset or model id), if one was remembered.
    pub fn known(&self, kind: UploadKind, fingerprint: &str) -> Option<String> {
        self.known.find(kind, fingerprint)
    }

    /// Remembers that the upload of `kind` with `fingerprint` became `id`.
    pub fn remember(&self, kind: UploadKind, fingerprint: &str, id: &str) {
        self.known.write(Known { kind, fingerprint: fingerprint.to_string(), id: id.to_string() });
    }
}

impl Drop for Uploads {
    fn drop(&mut self) {
        self.stop_sweeper();
    }
}

impl Session {
    fn upload(&self, st: &Progress, idle: Duration) -> Upload {
        let left = idle.saturating_sub(st.touched.elapsed());
        Upload {
            id: self.id.clone(),
            kind: self.kind,
            name: self.name.clone(),
            part: st.part,
            received_bytes: st.received,
            max_part_bytes: MAX_PART_BYTES as u64,
            expires: rfc3339(unix_now() + left.as_secs_f64()),
        }
    }

    /// Ends the session: its thread (or file) first, then its folder.
    fn close(&self, st: &mut Progress) {
        st.sink = None;
        remove_dir(&self.dir);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // A completed session's folder outlives its sink until the store has moved its content out.
        let st = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        st.sink = None;
        remove_dir(&self.dir);
    }
}

fn remove_dir(dir: &Path) {
    if let Err(e) = std::fs::remove_dir_all(dir)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!("removing {}: {e}", dir.display());
    }
}

/// A session's state, also after a panic while it was held (the state is plain numbers and a sink).
fn lock(m: &Mutex<Progress>) -> MutexGuard<'_, Progress> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Expires idle sessions every half idle limit (at most every minute) while `uploads` lives, parking on
/// `gate`'s condvar when no sessions are open and exiting immediately on shutdown or drop.
fn spawn_sweeper(uploads: Weak<Uploads>, gate: Arc<SweeperGate>) -> Option<JoinHandle<()>> {
    let spawned = std::thread::Builder::new().name("folo-uploads".into()).spawn(move || {
        let mut since = Instant::now();
        let mut st = gate.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if st.closed {
                return;
            }
            let epoch = st.generation;
            drop(st);
            let Some(u) = uploads.upgrade() else { return };
            let has_sessions = !u.sessions.lock().unwrap_or_else(|p| p.into_inner()).is_empty();
            let tick = (u.idle() / 2).clamp(Duration::from_millis(10), Duration::from_secs(60));
            drop(u);

            st = gate.state.lock().unwrap_or_else(|p| p.into_inner());
            if st.closed {
                return;
            }
            if st.generation != epoch {
                continue;
            }
            if !has_sessions {
                st = gate.wake.wait(st).unwrap_or_else(|p| p.into_inner());
                since = Instant::now();
                continue;
            }
            let left = tick.saturating_sub(since.elapsed());
            if left.is_zero() {
                drop(st);
                let Some(u) = uploads.upgrade() else { return };
                u.expire();
                drop(u);
                since = Instant::now();
                st = gate.state.lock().unwrap_or_else(|p| p.into_inner());
                continue;
            }
            st = gate.wake.wait_timeout(st, left).unwrap_or_else(|p| p.into_inner()).0;
        }
    });
    match spawned {
        Ok(h) => Some(h),
        Err(e) => {
            tracing::warn!(
                "could not start the upload sweeper (idle uploads expire when they are next used): {e}"
            );
            None
        }
    }
}
