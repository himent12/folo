//! Upload sessions of the real store (`/api/uploads`, docs/API.md *Uploads*): a byte stream cut into parts that
//! arrive one at a time, in order, into the session's folder `<data dir>/uploads/<id>/`.
//!
//! - A dataset's stream is a tar archive, unpacked as it arrives by one thread per session: a part is handed to the
//!   thread through a [`Pipe`] that holds that part only, and its request is answered once the thread has written
//!   every byte of it (or refused the archive), so memory stays at about one part and a hostile entry fails the part
//!   that carries it. Only regular files and folders with relative paths inside the session folder are accepted,
//!   and the archive is whole once its end-of-archive marker arrived (completing it earlier is a conflict that
//!   leaves the session open).
//! - A model's stream is one `.folo` file, written as it arrives.
//!
//! [`crate::RealStore`] moves what a completed session holds into place and registers it ([`Uploads::finish`]
//! hands it over). Sessions live in memory: one without a part for the idle limit ([`UPLOAD_IDLE`]) expires, a
//! dropped session takes its folder with it, and the folder of every session is removed on start. Uploads that had
//! a fingerprint are remembered in `<data dir>/uploads.json` (written atomically), so a later upload of the same
//! content is answered with what it became.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::http::StatusCode;
use folo_api::{CreateUpload, DatasetFormat, Upload, UploadKind};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::util::{fnv1a, rfc3339, unique_id, unix_now};

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
fn not_found(id: &str) -> Error {
    Error::new(StatusCode::NOT_FOUND, "upload_not_found", format!("upload '{id}' does not exist"))
        .hint("it may have expired (an hour without a part) or been completed; start a new one with POST /api/uploads")
}

/// `422 invalid_upload`.
fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_upload", message).hint(
        "a dataset upload is a tar archive of the dataset's folder: regular files and folders with relative paths, \
         no links or devices",
    )
}

/// `409 upload_conflict` for a stream that ends before the archive does (`message` says where).
fn truncated(message: String) -> Error {
    Error::conflict("upload_conflict", message)
        .hint("send the rest of the stream (GET /api/uploads/{id} names the next part), then complete it")
}

fn io_error(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error::io(format!("{what} {}: {e}", path.display()))
}

impl Uploads {
    /// The uploads of `data_dir`. Session folders a previous process left in `<data dir>/uploads/` are removed. A
    /// thread expires idle sessions while the returned value lives.
    pub fn open(data_dir: &Path) -> Arc<Uploads> {
        let dir = data_dir.join("uploads");
        if let Err(e) = std::fs::remove_dir_all(&dir)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("removing the leftover uploads in {}: {e}", dir.display());
        }
        let sweeper = Arc::new(SweeperGate::default());
        let uploads = Arc::new(Uploads {
            dir,
            sessions: Mutex::default(),
            idle_ms: AtomicU64::new(UPLOAD_IDLE.as_millis() as u64),
            known: Fingerprints::new(data_dir.join("uploads.json")),
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
        let mut sessions = self.sessions.lock().unwrap();
        let seed = fnv1a(req.name.as_bytes()) ^ unix_now().to_bits();
        let id = unique_id("up", seed, |id| sessions.contains_key(id) || self.dir.join(id).exists());
        let dir = self.dir.join(&id);
        std::fs::create_dir(&dir).map_err(|e| io_error("creating", &dir, e))?;
        let sink = match req.kind {
            UploadKind::Dataset => Unpacker::start(&id, &dir).map(Sink::Archive),
            UploadKind::Model => {
                let path = dir.join(MODEL_FILE);
                std::fs::File::create(&path).map(Sink::File).map_err(|e| io_error("creating", &path, e))
            }
        };
        let sink = sink.inspect_err(|_| {
            let _ = std::fs::remove_dir_all(&dir);
        })?;
        let session = Arc::new(Session {
            id: id.clone(),
            kind: req.kind,
            name: req.name.trim().to_string(),
            format: req.format,
            fingerprint: req.fingerprint.clone(),
            dir,
            state: Mutex::new(Progress {
                part: 0,
                received: 0,
                last_len: None,
                touched: Instant::now(),
                sink: Some(sink),
            }),
        });
        let upload = session.upload(&session.state.lock().unwrap(), self.idle());
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
            ));
        }
        self.live(id, |s, st| {
            let len = bytes.len() as u64;
            if part == st.part {
                let written = match st.sink.as_mut() {
                    Some(Sink::Archive(u)) => u.pipe.feed(bytes),
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
                st.part += 1;
                st.received += len;
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
                Some(Sink::Archive(u)) => u.pipe.finish(),
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
        let session = self.sessions.lock().unwrap().remove(id).ok_or_else(|| not_found(id))?;
        session.close(&mut lock(&session.state));
        Ok(())
    }

    /// Stops the sweeper thread and drops every session (ending their threads and removing their folders).
    pub fn shutdown(&self) {
        self.stop_sweeper();
        let sessions: Vec<_> = self.sessions.lock().unwrap().drain().map(|(_, s)| s).collect();
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
        let sessions: Vec<_> = self.sessions.lock().unwrap().values().cloned().collect();
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
        let session = self.sessions.lock().unwrap().get(id).cloned().ok_or_else(|| not_found(id))?;
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
        let mut sessions = self.sessions.lock().unwrap();
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

// ---- remembered fingerprints -------------------------------------------------------------------------------------

/// One finished upload with a fingerprint (`uploads.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Known {
    kind: UploadKind,
    fingerprint: String,
    /// The dataset id or model id it became.
    id: String,
}

/// `<data dir>/uploads.json`, cached in memory after its first read and updated atomically on write. A missing or
/// malformed file remembers nothing (the next upload sends its content).
struct Fingerprints {
    path: PathBuf,
    cache: Mutex<Option<Vec<Known>>>,
    #[cfg(test)]
    disk_reads: std::sync::atomic::AtomicUsize,
}

impl Fingerprints {
    fn new(path: PathBuf) -> Fingerprints {
        Fingerprints {
            path,
            cache: Mutex::default(),
            #[cfg(test)]
            disk_reads: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Reads `uploads.json` from disk: `Some(Vec::new())` when missing or malformed (safe to cache/replace),
    /// `None` on transient I/O errors so callers retry later and never overwrite from an unknown baseline.
    fn read_disk(&self) -> Option<Vec<Known>> {
        #[cfg(test)]
        self.disk_reads.fetch_add(1, Ordering::Relaxed);
        let bytes = match std::fs::read(&self.path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(Vec::new()),
            Err(e) => {
                tracing::warn!("reading {}: {e}", self.path.display());
                return None;
            }
        };
        Some(serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(
                "{} is malformed ({e}); it is replaced on the next finished upload",
                self.path.display()
            );
            Vec::new()
        }))
    }

    fn load<'a>(&self, cache: &'a mut Option<Vec<Known>>) -> Option<&'a mut Vec<Known>> {
        if cache.is_none() {
            *cache = self.read_disk();
        }
        cache.as_mut()
    }

    fn find(&self, kind: UploadKind, fingerprint: &str) -> Option<String> {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        let list = self.load(&mut cache)?;
        list.iter().find(|k| k.kind == kind && k.fingerprint == fingerprint).map(|k| k.id.clone())
    }

    /// Records `k`, replacing an older record of its kind and fingerprint.
    fn write(&self, k: Known) {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        let Some(list) = self.load(&mut cache) else {
            return;
        };
        let mut next = list.clone();
        next.retain(|x| x.kind != k.kind || x.fingerprint != k.fingerprint);
        next.push(k);
        let bytes = serde_json::to_vec_pretty(&next).expect("fingerprints serialize");
        match write_atomic(&self.path, &bytes) {
            Ok(()) => *list = next,
            Err(e) => tracing::warn!("writing {}: {e}", self.path.display()),
        }
    }
}

// ---- unpacking ---------------------------------------------------------------------------------------------------

/// How a stream's end went ([`Pipe::finish`]).
enum Finish {
    /// The archive (or file) is whole.
    Done,
    /// The archive has not ended yet; the session stays open for more parts.
    Incomplete(Error),
    /// It cannot be used; the session is dropped.
    Failed(Error),
}

/// A dataset session's unpacking thread. Dropping it stops the thread and waits for it, so the session's files
/// are closed when its folder is removed.
struct Unpacker {
    pipe: Arc<Pipe>,
    thread: Option<JoinHandle<()>>,
}

impl Unpacker {
    fn start(id: &str, root: &Path) -> Result<Unpacker> {
        let pipe = Arc::new(Pipe::default());
        let (p, root) = (pipe.clone(), root.to_path_buf());
        let thread = std::thread::Builder::new()
            .name(format!("folo-{id}"))
            .spawn(move || {
                let outcome = unpack(&p, &root);
                let mut f = p.lock();
                f.reading = None;
                f.outcome = Some(outcome);
                p.changed.notify_all();
            })
            .map_err(|e| Error::internal(format!("starting the unpacking thread: {e}")))?;
        Ok(Unpacker { pipe, thread: Some(thread) })
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
struct Pipe {
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
    fn feed(&self, bytes: Bytes) -> Result<()> {
        let mut f = self.lock();
        if let Some(Err(e)) = &f.outcome {
            return Err(e.clone());
        }
        f.chunk = Some(bytes);
        f.pos = 0;
        self.changed.notify_all();
        while f.chunk.is_some() && f.outcome.is_none() {
            f = self.changed.wait(f).unwrap_or_else(|p| p.into_inner());
        }
        match &f.outcome {
            Some(Err(e)) => Err(e.clone()),
            _ => Ok(()),
        }
    }

    /// Ends the stream once the archive is whole (its end-of-archive marker arrived) and waits for the thread.
    /// Before that the archive is incomplete and the stream stays open.
    fn finish(&self) -> Finish {
        let mut f = self.lock();
        if let Some(Err(e)) = &f.outcome {
            return Finish::Failed(e.clone());
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

    fn cancel(&self) {
        self.lock().cancelled = true;
        self.changed.notify_all();
    }

    /// The thread starts reading file `name`.
    fn reading(&self, name: &str) {
        self.lock().reading = Some(name.to_string());
    }

    /// The thread has read the file it was reading.
    fn read(&self) {
        let mut f = self.lock();
        f.last = f.reading.take();
    }

    /// The end-of-archive marker arrived.
    fn end(&self) {
        self.lock().ended = true;
    }
}

/// The thread's end of a [`Pipe`].
struct PipeReader(Arc<Pipe>);

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

/// An error of the tar reader: the stream is not a readable archive.
fn unreadable(e: std::io::Error) -> Error {
    invalid(format!("not a readable tar archive: {e}"))
}

/// Unpacks the archive arriving through `pipe` into `root`, then reads the stream to its end (writers pad archives
/// after the end-of-archive marker). The stream ends only once the marker arrived ([`Pipe::finish`]), so the reader
/// never runs out inside the archive.
fn unpack(pipe: &Arc<Pipe>, root: &Path) -> Result<()> {
    let mut archive = tar::Archive::new(PipeReader(pipe.clone()));
    let mut buf = vec![0u8; 64 * 1024];
    // Each file's name as Windows compares it ([`folded`]) -> the entry's name.
    let mut files: HashMap<String, String> = HashMap::new();
    for entry in archive.entries().map_err(unreadable)? {
        let mut entry = entry.map_err(unreadable)?;
        let kind = entry.header().entry_type();
        if kind.is_pax_global_extensions() {
            // Archive-wide metadata (`git archive` writes one), not a file.
            continue;
        }
        let rel = entry_path(&entry)?;
        let path = root.join(&rel);
        if let Some(clash) = file_folder_clash(root, &rel, kind.is_dir()) {
            return Err(invalid(format!(
                "{clash} is both a file and a folder in the archive; rename one and pack again"
            )));
        }
        if kind.is_dir() {
            std::fs::create_dir_all(&path).map_err(|e| io_error("creating", &path, e))?;
        } else if kind.is_file() || kind.is_contiguous() {
            if rel.as_os_str().is_empty() {
                return Err(invalid("a file entry without a name"));
            }
            let name = folo_api::json::path(&rel);
            if let Some(other) = files.insert(folded(&rel), name.clone())
                && other != name
            {
                return Err(invalid(format!(
                    "{other} and {name} differ only by letter case or a trailing dot or space, so one would \
                     overwrite the other on Windows; rename one (and the labels that refer to it) and pack again"
                )));
            }
            pipe.reading(&name);
            write_entry(&mut entry, &path, &name, &mut buf)?;
            pipe.read();
        } else {
            let what = if kind.is_symlink() {
                "a symbolic link"
            } else if kind.is_hard_link() {
                "a hard link"
            } else {
                "a special file (device, FIFO, sparse file)"
            };
            return Err(invalid(format!("{} is {what}; only files and folders are accepted", rel.display())));
        }
    }
    pipe.end();
    std::io::copy(&mut archive.into_inner(), &mut std::io::sink()).map_err(unreadable)?;
    Ok(())
}

/// The relative path an entry unpacks to: its normal components (`.` skipped). An absolute path, `..`, a drive,
/// or a `:` (an NTFS stream) is `422 invalid_upload` on every OS; on Windows, components that fail to unpack or
/// lose data there (`< > " | ? *`, ASCII control bytes `< 0x20`, dot/space-only names, `NUL` with optional trailing
/// dots/spaces, or a parent folder ending in a space or more than one dot/space) are `422 invalid_upload` too.
fn entry_path(entry: &tar::Entry<impl Read>) -> Result<PathBuf> {
    let raw = entry.path().map_err(|e| invalid(format!("an entry's name is not a path: {e}")))?;
    let mut parts = Vec::new();
    for c in raw.components() {
        match c {
            Component::CurDir => {}
            Component::Normal(n) => {
                if n.to_string_lossy().contains(':') {
                    return Err(invalid(format!(
                        "{}: only relative paths inside the dataset folder are accepted",
                        raw.display()
                    )));
                }
                parts.push(n.to_os_string());
            }
            _ => {
                return Err(invalid(format!(
                    "{}: only relative paths inside the dataset folder are accepted",
                    raw.display()
                )));
            }
        }
    }
    #[cfg(windows)]
    for (i, n) in parts.iter().enumerate() {
        let s = n.to_string_lossy();
        let is_parent = i + 1 < parts.len();
        if let Some(why) = windows_unsafe_component(&s, is_parent) {
            return Err(invalid(format!("{}: {why}; rename it and pack again", raw.display())));
        }
    }
    Ok(parts.into_iter().collect())
}

/// Why a path component fails to unpack or loses data on Windows (`os error 123`/`5`/`3` or writing to
/// `\Device\Null`). Narrowed to what fails on Windows 11 with `std::fs`: forbidden characters (`< > " | ? *` and
/// ASCII control bytes `< 0x20`), dot/space-only components (`...`, `   `), bare `NUL` (with optional trailing
/// dots/spaces), and parent folders ending in a space or multiple dots/spaces (`foo /a.png`, `foo../a.png`).
/// Names that unpack cleanly on Windows 11 (`CON.png`, `NUL.png`, `aux/`, bare `CON`/`AUX`/`PRN`/`COM*`/`LPT*`,
/// and files with a trailing dot or space) stay accepted; collisions remain caught by [`folded`].
#[cfg(windows)]
fn windows_unsafe_component(name: &str, is_parent: bool) -> Option<&'static str> {
    if name.bytes().any(|b| b < 0x20 || matches!(b, b'<' | b'>' | b'"' | b'|' | b'?' | b'*')) {
        return Some("contains characters Windows forbids in file names");
    }
    let trimmed = name.trim_end_matches(['.', ' ']);
    if trimmed.is_empty() {
        return Some("is empty on Windows (only dots or spaces)");
    }
    if trimmed.eq_ignore_ascii_case("nul") {
        return Some("uses the reserved Windows device name NUL");
    }
    if is_parent && trimmed != name && !name.strip_suffix('.').is_some_and(|s| s == trimmed) {
        return Some(
            "has a folder name ending in spaces or multiple dots that Windows cannot open files under",
        );
    }
    None
}

/// The name of what an entry at `rel` (a folder when `dir`) collides with in `root`: an earlier file where it or one
/// of its parent folders goes, or an earlier folder where a file goes. Without this check the unpacking fails with an
/// operating system error (`500 io_error`) instead of refusing the archive.
fn file_folder_clash(root: &Path, rel: &Path, dir: bool) -> Option<String> {
    let mut at = root.to_path_buf();
    let parts: Vec<_> = rel.components().collect();
    for (i, c) in parts.iter().enumerate() {
        at.push(c);
        let last = i + 1 == parts.len();
        let Ok(meta) = std::fs::symlink_metadata(&at) else { return None };
        let wants_dir = !last || dir;
        if wants_dir != meta.is_dir() {
            let name: PathBuf = parts[..=i].iter().collect();
            return Some(folo_api::json::path(&name));
        }
    }
    None
}

/// `rel` as Windows compares names: each component without trailing dots and spaces, in lower case.
fn folded(rel: &Path) -> String {
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().trim_end_matches(['.', ' ']).to_lowercase())
        .collect();
    parts.join("/")
}

/// Writes file entry `name` to `path`.
fn write_entry(entry: &mut tar::Entry<impl Read>, path: &Path, name: &str, buf: &mut [u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| io_error("creating", dir, e))?;
    }
    let mut file = std::fs::File::create(path).map_err(|e| io_error("creating", path, e))?;
    let mut left = entry.size();
    while left > 0 {
        let n = entry.read(buf).map_err(unreadable)?;
        if n == 0 {
            return Err(truncated(format!("the stream ends inside {name}")));
        }
        file.write_all(&buf[..n]).map_err(|e| io_error("writing", path, e))?;
        left = left.saturating_sub(n as u64);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    /// `Fingerprints` reads `uploads.json` at most once and serves subsequent lookups and updates from memory.
    #[test]
    fn fingerprints_cache_reads_uploads_json_once() {
        let dir = TestDir::new("uploads-fp-cache");
        let uploads = Uploads::open(dir.path());
        assert_eq!(uploads.known(UploadKind::Model, "fp1"), None);
        assert_eq!(uploads.known.disk_reads.load(Ordering::Relaxed), 1);

        uploads.remember(UploadKind::Model, "fp1", "m_1");
        assert_eq!(uploads.known(UploadKind::Model, "fp1").as_deref(), Some("m_1"));
        uploads.remember(UploadKind::Model, "fp1", "m_2");
        assert_eq!(uploads.known(UploadKind::Model, "fp1").as_deref(), Some("m_2"));
        assert_eq!(uploads.known.disk_reads.load(Ordering::Relaxed), 1, "no repeat disk reads after warmup");

        // Fresh instance loads persisted state on first lookup.
        uploads.shutdown();
        let reopened = Uploads::open(dir.path());
        assert_eq!(reopened.known(UploadKind::Model, "fp1").as_deref(), Some("m_2"));
        assert_eq!(reopened.known.disk_reads.load(Ordering::Relaxed), 1);
        reopened.shutdown();

        // A transient non-NotFound read error (e.g. a directory in place of `uploads.json`) is not cached.
        let bad = TestDir::new("uploads-fp-ioerr");
        std::fs::create_dir(bad.join("uploads.json")).unwrap();
        let fp = Fingerprints::new(bad.join("uploads.json"));
        assert_eq!(fp.find(UploadKind::Model, "fp1"), None);
        assert_eq!(fp.find(UploadKind::Model, "fp1"), None);
        assert_eq!(fp.disk_reads.load(Ordering::Relaxed), 2, "transient I/O error is retried, not latched");
    }

    /// `Uploads::shutdown` wakes the condvar sweeper and joins it immediately (no 100 ms sleep-loop lag).
    #[test]
    fn upload_sweeper_wakes_and_shuts_down_promptly() {
        let dir = TestDir::new("uploads-sweeper-wake");
        let uploads = Uploads::open(dir.path());
        let t0 = Instant::now();
        uploads.shutdown();
        assert!(t0.elapsed() < Duration::from_secs(2), "shutdown stalled: {:?}", t0.elapsed());
        assert!(uploads.sweeper_thread.lock().unwrap().is_none());
    }
}
