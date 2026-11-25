//! Moving files to and from a FOLO server in parts: [`upload`] (docs/API.md, *Uploads*) sends a stream a caller
//! produces, [`download`] (*Models*, Ranges) copies a file into place. A cloud control plane sends a worker its
//! dataset and checkpoint this way and copies the weights back (docs/CLOUD.md). Every request is short and retried
//! when it fails for a reason of the moment ([`Error::is_transient`]), so a proxy that cuts connections or loses an
//! answer costs a retry, not the transfer. Memory stays at about one part.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use folo_api::{ApiError, CreateUpload, TransferProgress, Upload, UploadCreated, Uploaded};
use http::StatusCode;

use crate::{Client, Error, StreamError, StreamedPart, encode};

/// The largest part sent or fetched (less when the server takes less, `Upload.max_part_bytes`).
pub const PART_BYTES: u64 = 8 * 1024 * 1024;
/// Tries of one request before a transient failure counts as final.
pub const ATTEMPTS: u32 = 6;
/// Times a download starts over because the file was replaced while it was fetched.
pub const RESTARTS: u32 = 3;
/// After a download's first part, the rest of the file is asked for in one streamed request when it is at most this
/// large ([`crate::MAX_BODY`] bounds an answer) and, at the first part's rate, arrives within [`REST_WITHIN`].
pub const REST_MAX_BYTES: u64 = 192 * 1024 * 1024;
/// See [`REST_MAX_BYTES`]: well inside the 100 seconds after which provider proxies cut a connection.
pub const REST_WITHIN: Duration = Duration::from_secs(30);
/// The wait before the first retry ([`Control::backoff`]); it doubles with every retry, up to [`MAX_BACKOFF`].
pub const FIRST_BACKOFF: Duration = Duration::from_millis(500);
const MAX_BACKOFF: Duration = Duration::from_secs(8);
/// Interval at which [`pause`] polls [`Control::stop`] while sleeping through a retry backoff.
const STOP_POLL: Duration = Duration::from_millis(50);

/// Why a transfer did not finish.
#[derive(Debug)]
pub enum TransferError {
    /// The caller asked it to stop ([`Control::stop`]).
    Stopped,
    /// A request failed for good: an error answer, or a transient failure that outlived the retries.
    Failed(Error),
    /// The file was replaced every time it was fetched ([`RESTARTS`] restarts): it changes faster than it downloads.
    Replaced,
    /// The stream could not be produced, or the destination file not written.
    Io(std::io::Error),
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferError::Stopped => f.write_str("the transfer was stopped"),
            TransferError::Failed(e) => write!(f, "{e}"),
            TransferError::Replaced => {
                write!(f, "the file was replaced {} times while it was downloaded", RESTARTS + 1)
            }
            TransferError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TransferError {}

/// What a transfer tells its caller and asks it.
pub struct Control<'a> {
    /// Called after every part with the bytes moved so far (`total_bytes` 0 when unknown).
    pub progress: &'a mut dyn FnMut(TransferProgress),
    /// Asked before every part and retry: `true` ends the transfer with [`TransferError::Stopped`].
    pub stop: &'a dyn Fn() -> bool,
    /// The wait before the first retry of a request: [`FIRST_BACKOFF`] (tests shorten it).
    pub backoff: Duration,
}

/// What [`download`] fetched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Downloaded {
    /// Length of the file.
    pub bytes: u64,
    /// The ETag of the version fetched, quoted as the server sent it.
    pub etag: String,
}

/// Sends what `write` produces as upload `req` and completes it. When the server already holds an upload with
/// `req.fingerprint`, its answer is returned at once and `write` is not called. The stream is cut into parts of
/// `min(Upload.max_part_bytes, PART_BYTES)` bytes sent in order; after a failed part the server is asked where the
/// stream is (`GET /api/uploads/{id}`), so a part whose answer was lost is not sent twice. On a final failure or a
/// stop the session is deleted (best effort).
pub fn upload(
    client: &Client,
    req: &CreateUpload,
    write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
    ctl: &mut Control<'_>,
) -> Result<Uploaded, TransferError> {
    let created: UploadCreated = retry(ctl, || client.post("/uploads", req))?;
    let session = match created {
        UploadCreated { existing: Some(existing), .. } => return Ok(existing),
        UploadCreated { upload: Some(u), .. } => u,
        UploadCreated { upload: None, existing: None } => {
            return Err(TransferError::Failed(bad_gateway(
                client,
                "POST /api/uploads answered neither an upload nor what exists",
            )));
        }
    };
    let id = encode(&session.id);
    let result = send(client, &session, req, write, ctl).and_then(|()| complete(client, &id, req, ctl));
    if result.is_err() {
        let _ = client.delete(&format!("/uploads/{id}"));
    }
    result
}

/// Sends the stream `write` produces as the parts of `session`.
fn send(
    client: &Client,
    session: &Upload,
    req: &CreateUpload,
    write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
    ctl: &mut Control<'_>,
) -> Result<(), TransferError> {
    let size = session.max_part_bytes.clamp(1, PART_BYTES) as usize;
    let mut parts = Parts {
        client,
        id: encode(&session.id),
        size,
        buf: Vec::with_capacity(size),
        next: session.part,
        sent: session.received_bytes,
        total: req.bytes.unwrap_or(0),
        ctl,
        failure: None,
    };
    let written = write(&mut parts);
    if let Some(failure) = parts.failure.take() {
        return Err(failure);
    }
    written.map_err(TransferError::Io)?;
    if !parts.buf.is_empty() {
        parts.send_buffered()?;
    }
    Ok(())
}

/// `POST /api/uploads/{id}/complete`. When an attempt's answer was lost the session is gone on the retry (`404`);
/// with a fingerprint the server then names what the upload became.
fn complete(
    client: &Client,
    id: &str,
    req: &CreateUpload,
    ctl: &mut Control<'_>,
) -> Result<Uploaded, TransferError> {
    let mut lost = false;
    let completed = retry(ctl, || {
        client.post_empty(&format!("/uploads/{id}/complete")).inspect_err(|e| lost |= e.is_transient())
    });
    match completed {
        Err(TransferError::Failed(e))
            if lost && e.code() == "upload_not_found" && req.fingerprint.is_some() =>
        {
            match retry(ctl, || client.post::<_, UploadCreated>("/uploads", req))? {
                UploadCreated { existing: Some(existing), .. } => Ok(existing),
                // The server does not know the content after all: the session this question opened is not needed.
                created => {
                    if let Some(u) = created.upload {
                        let _ = client.delete(&format!("/uploads/{}", encode(&u.id)));
                    }
                    Err(TransferError::Failed(e))
                }
            }
        }
        other => other,
    }
}

/// The stream being uploaded: buffers one part, sends it when it is full.
struct Parts<'c, 'a> {
    client: &'c Client,
    /// The session id, encoded for paths.
    id: String,
    size: usize,
    buf: Vec<u8>,
    /// Index of the next part.
    next: u32,
    sent: u64,
    total: u64,
    ctl: &'c mut Control<'a>,
    /// Why the upload failed; the writer refuses everything after.
    failure: Option<TransferError>,
}

impl Parts<'_, '_> {
    /// Sends the buffered bytes as the next part and reports the progress.
    fn send_buffered(&mut self) -> Result<(), TransferError> {
        if (self.ctl.stop)() {
            return Err(TransferError::Stopped);
        }
        self.put(self.next)?;
        self.next += 1;
        self.sent += self.buf.len() as u64;
        self.buf.clear();
        (self.ctl.progress)(TransferProgress { done_bytes: self.sent, total_bytes: self.total });
        Ok(())
    }

    /// `PUT` part `n`, retried. After a transient failure the server says where the stream is: one part further
    /// (the part arrived, its answer did not), the same part (send it again), or elsewhere (the upload is broken).
    fn put(&mut self, n: u32) -> Result<(), TransferError> {
        let path = format!("/uploads/{}/parts/{n}", self.id);
        let mut wait = self.ctl.backoff;
        let mut attempt = 1;
        loop {
            let e = match self.client.put::<Upload>(&path, &self.buf) {
                Ok(_) => return Ok(()),
                Err(e) if e.is_transient() && attempt < ATTEMPTS => e,
                Err(e) => return Err(TransferError::Failed(e)),
            };
            attempt += 1;
            pause(self.ctl, &mut wait)?;
            match self.client.get::<Upload>(&format!("/uploads/{}", self.id)) {
                Ok(u) if u.part == n + 1 => return Ok(()),
                Ok(u) if u.part == n => {}
                Ok(u) => {
                    let message = format!(
                        "the server expects part {} of the upload, the stream is at part {n}",
                        u.part
                    );
                    return Err(TransferError::Failed(Error {
                        status: Some(StatusCode::CONFLICT),
                        error: ApiError {
                            code: "upload_conflict".into(),
                            message,
                            hint: Some(format!(
                                "start the upload again (the failed part: {})",
                                e.error.message
                            )),
                        },
                    }));
                }
                Err(e) if e.is_transient() => {}
                Err(e) => return Err(TransferError::Failed(e)),
            }
        }
    }
}

impl Write for Parts<'_, '_> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        if self.failure.is_some() {
            return Err(std::io::Error::other("the upload failed"));
        }
        let n = data.len().min(self.size - self.buf.len());
        self.buf.extend_from_slice(&data[..n]);
        if self.buf.len() == self.size
            && let Err(e) = self.send_buffered()
        {
            self.failure = Some(e);
            return Err(std::io::Error::other("the upload failed"));
        }
        Ok(n)
    }

    /// Parts are sent when they are full; the last, shorter one when the stream ends.
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Fetches `GET /api{path}` (a weights file) into `dest` with `Range` requests, each after the first with
/// `If-Match` on the first one's ETag: a file replaced meanwhile (`412`) is fetched again from the start, at most
/// [`RESTARTS`] times. The bytes go to a temporary file next to `dest`, which is renamed into place at the end, so
/// `dest` is never a partial file.
///
/// The first request asks for [`PART_BYTES`] and tells the file's length and how fast the link is. When the rest
/// would arrive within [`REST_WITHIN`] (and is at most [`REST_MAX_BYTES`]), it is asked for in one request: the
/// server streams one request from one open file, so a file that is replaced often (a run's `last.folo`, every
/// epoch) can only get in the way between the first answer and the second request, not once per part. Otherwise,
/// and when that one request fails for a reason of the moment, the rest comes in parts of [`PART_BYTES`].
///
/// A file that was replaced in that moment too (epochs of half a second) is then asked for whole in one request,
/// when the whole of it fits the same bounds: one request is one version, whatever happens to the file meanwhile.
pub fn download(
    client: &Client,
    path: &str,
    dest: &Path,
    ctl: &mut Control<'_>,
) -> Result<Downloaded, TransferError> {
    let tmp = temp_path(dest);
    let fetched = fetch(client, path, &tmp, ctl)
        .and_then(|d| std::fs::rename(&tmp, dest).map(|()| d).map_err(TransferError::Io));
    if fetched.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    fetched
}

/// Removes the temporary files of downloads into `dir` that did not finish (a process killed mid-download leaves
/// its `.<name>.<pid>.<n>.download`). For a directory nothing else downloads into at the moment.
pub fn remove_partial(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for path in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if name.starts_with('.') && name.ends_with(".download") {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// `.<name>.<pid>.<n>.download` next to `dest`.
fn temp_path(dest: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let name = dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    dest.with_file_name(format!(
        ".{name}.{}.{}.download",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// [`download`] into `tmp`.
fn fetch(
    client: &Client,
    path: &str,
    tmp: &Path,
    ctl: &mut Control<'_>,
) -> Result<Downloaded, TransferError> {
    // The file's length, once a first part showed that all of it fits one request: what the next version is
    // asked for at once after this one was replaced (tried once).
    let mut whole: Option<u64> = None;
    'version: for _ in 0..=RESTARTS {
        let mut file = std::fs::File::create(tmp).map_err(TransferError::Io)?;
        let mut etag: Option<String> = None;
        let mut done = 0u64;
        // The end of the next request when it is the whole rest (decided after the first part, tried once).
        let mut rest: Option<u64> = whole.take();
        loop {
            if (ctl.stop)() {
                return Err(TransferError::Stopped);
            }
            let asked = std::time::Instant::now();
            let answer = match rest.take() {
                Some(end) => {
                    match client.get_range_to(path, done..end, etag.as_deref(), &mut file, ctl.stop) {
                        Ok(p) => Ok(p),
                        // Too much for one request on this link after all: discard any partial bytes written and
                        // continue in parts from `done`.
                        Err(StreamError::Http(e)) if e.is_transient() => {
                            rewind(&mut file, done)?;
                            continue;
                        }
                        Err(StreamError::Http(e)) => Err(TransferError::Failed(e)),
                        Err(StreamError::Stopped) => Err(TransferError::Stopped),
                        Err(StreamError::Io(e)) => Err(TransferError::Io(e)),
                    }
                }
                None => {
                    fetch_part(client, path, done..done + PART_BYTES, etag.as_deref(), &mut file, done, ctl)
                }
            };
            let part = match answer {
                Ok(p) => p,
                Err(TransferError::Failed(e)) if e.status == Some(StatusCode::PRECONDITION_FAILED) => {
                    continue 'version;
                }
                // An empty file has no first byte to ask for.
                Err(TransferError::Failed(e))
                    if e.status == Some(StatusCode::RANGE_NOT_SATISFIABLE) && done == 0 =>
                {
                    let (headers, bytes) = retry(ctl, || client.get_bytes(path))?;
                    file.write_all(&bytes).map_err(TransferError::Io)?;
                    let etag =
                        headers.get(http::header::ETAG).and_then(|v| v.to_str().ok()).unwrap_or_default();
                    return Ok(Downloaded { bytes: bytes.len() as u64, etag: etag.to_string() });
                }
                Err(e) => return Err(e),
            };
            if part.written == 0 && done < part.total {
                return Err(TransferError::Failed(bad_gateway(
                    client,
                    "a range of the file came back empty",
                )));
            }
            let first = done == 0;
            done += part.written;
            (ctl.progress)(TransferProgress { done_bytes: done, total_bytes: part.total });
            let version = etag.get_or_insert(part.etag);
            if done >= part.total {
                file.sync_all().map_err(TransferError::Io)?;
                return Ok(Downloaded { bytes: done, etag: version.clone() });
            }
            let (got, took) = (part.written, asked.elapsed());
            if first && rest_fits(part.total - done, got, took) {
                rest = Some(part.total);
                whole = rest_fits(part.total, got, took).then_some(part.total);
            }
        }
    }
    Err(TransferError::Replaced)
}

/// Streams one `Range` part into `file`, rewinding to `done` before each transient retry so partial bytes from a
/// dropped connection never corrupt the file.
fn fetch_part(
    client: &Client,
    path: &str,
    range: std::ops::Range<u64>,
    etag: Option<&str>,
    file: &mut std::fs::File,
    done: u64,
    ctl: &Control<'_>,
) -> Result<StreamedPart, TransferError> {
    let mut wait = ctl.backoff;
    let mut attempt = 1;
    loop {
        match client.get_range_to(path, range.clone(), etag, file, ctl.stop) {
            Ok(p) => return Ok(p),
            Err(StreamError::Stopped) => return Err(TransferError::Stopped),
            Err(StreamError::Io(e)) => return Err(TransferError::Io(e)),
            Err(StreamError::Http(e)) if e.is_transient() && attempt < ATTEMPTS => {
                rewind(file, done)?;
                attempt += 1;
                pause(ctl, &mut wait)?;
            }
            Err(StreamError::Http(e)) => return Err(TransferError::Failed(e)),
        }
    }
}

/// Truncates `file` back to `pos` and seeks there so a retried range overwrites any bytes streamed before a
/// mid-body failure.
fn rewind(file: &mut std::fs::File, pos: u64) -> Result<(), TransferError> {
    file.set_len(pos).map_err(TransferError::Io)?;
    file.seek(SeekFrom::Start(pos)).map_err(TransferError::Io)?;
    Ok(())
}

/// Whether the `rest` bytes of a download are asked for in one request, after a first part of `got` bytes that
/// took `took`: at most [`REST_MAX_BYTES`], and within [`REST_WITHIN`] at that rate.
fn rest_fits(rest: u64, got: u64, took: Duration) -> bool {
    let rate = got as f64 / took.as_secs_f64().max(1e-3);
    rest <= REST_MAX_BYTES && rest as f64 <= rate * REST_WITHIN.as_secs_f64()
}

/// Calls `call` until it succeeds, fails for good, or has failed transiently [`ATTEMPTS`] times, waiting longer
/// before each retry.
fn retry<T>(ctl: &Control<'_>, mut call: impl FnMut() -> crate::Result<T>) -> Result<T, TransferError> {
    let mut wait = ctl.backoff;
    let mut attempt = 1;
    loop {
        match call() {
            Ok(v) => return Ok(v),
            Err(e) if e.is_transient() && attempt < ATTEMPTS => {
                attempt += 1;
                pause(ctl, &mut wait)?;
            }
            Err(e) => return Err(TransferError::Failed(e)),
        }
    }
}

/// Waits `wait` (doubling it for the next time, up to [`MAX_BACKOFF`]) unless the caller asks to stop.
fn pause(ctl: &Control<'_>, wait: &mut Duration) -> Result<(), TransferError> {
    let deadline = std::time::Instant::now() + *wait;
    loop {
        if (ctl.stop)() {
            return Err(TransferError::Stopped);
        }
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            break;
        }
        std::thread::sleep(left.min(STOP_POLL));
    }
    *wait = (*wait * 2).min(MAX_BACKOFF);
    Ok(())
}

/// A `502 bad_gateway` for an answer that does not follow the API.
fn bad_gateway(client: &Client, message: &str) -> Error {
    Error {
        status: Some(StatusCode::BAD_GATEWAY),
        error: ApiError {
            code: "bad_gateway".into(),
            message: format!("{}: {message}", client.base()),
            hint: Some("the server may be a different FOLO version".into()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rest_is_one_request_when_it_arrives_in_time_and_fits_in_memory() {
        let mib = 1024 * 1024;
        let secs = Duration::from_secs_f64;
        // 8 MiB in a second: 30 s carry 240 MiB, more than the cap of 192.
        assert!(rest_fits(34 * mib, PART_BYTES, secs(1.0)));
        assert!(rest_fits(REST_MAX_BYTES, PART_BYTES, secs(1.0)));
        assert!(!rest_fits(REST_MAX_BYTES + 1, PART_BYTES, secs(0.01)), "more than is held in memory");
        // 8 MiB in 16 s is half a MiB a second: 15 MiB arrive within 30 s, 16 do not.
        assert!(rest_fits(15 * mib, PART_BYTES, secs(16.0)));
        assert!(!rest_fits(16 * mib, PART_BYTES, secs(16.0)));
        // A part timed at zero does not divide by it.
        assert!(rest_fits(mib, PART_BYTES, Duration::ZERO));
    }

    #[test]
    fn pause_stops_promptly_when_cancelled_mid_backoff() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();
        let stopper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            stop_flag.store(true, Ordering::SeqCst);
        });

        let stop_fn = || stop.load(Ordering::SeqCst);
        let ctl = Control { progress: &mut |_| {}, stop: &stop_fn, backoff: Duration::from_secs(10) };
        let mut wait = ctl.backoff;
        let started = std::time::Instant::now();
        let result = pause(&ctl, &mut wait);
        let elapsed = started.elapsed();
        stopper.join().unwrap();

        assert!(matches!(result, Err(TransferError::Stopped)));
        assert!(elapsed < Duration::from_secs(2), "pause waited {elapsed:?} despite stop signal");
    }
}
