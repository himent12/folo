//! Moving files to and from a FOLO server in parts: [`upload`] (docs/API.md, *Uploads*) sends a stream a caller
//! produces, [`download`] (*Models*, Ranges) copies a file into place. A cloud control plane sends a worker its
//! dataset and checkpoint this way and copies the weights back (docs/CLOUD.md). Every request is short and retried
//! when it fails for a reason of the moment ([`Error::is_transient`]), so a proxy that cuts connections or loses an
//! answer costs a retry, not the transfer. Memory stays at about one part.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use folo_api::{ApiError, CreateUpload, TransferProgress, Upload, UploadCreated, Uploaded};
use http::StatusCode;

use crate::{Client, Error, encode};

/// The largest part sent or fetched (less when the server takes less, `Upload.max_part_bytes`).
pub const PART_BYTES: u64 = 8 * 1024 * 1024;
/// Tries of one request before a transient failure counts as final.
pub const ATTEMPTS: u32 = 6;
/// Times a download starts over because the file was replaced while it was fetched.
pub const RESTARTS: u32 = 3;
/// The wait before the first retry; it doubles with every retry, up to [`MAX_BACKOFF`].
const FIRST_BACKOFF: Duration = Duration::from_millis(500);
const MAX_BACKOFF: Duration = Duration::from_secs(8);

/// Why a transfer did not finish.
#[derive(Debug)]
pub enum TransferError {
    /// The caller asked it to stop ([`Control::stop`]).
    Stopped,
    /// A request failed for good: an error answer, or a transient failure that outlived the retries.
    Failed(Error),
    /// The stream could not be produced, or the destination file not written.
    Io(std::io::Error),
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferError::Stopped => f.write_str("the transfer was stopped"),
            TransferError::Failed(e) => write!(f, "{e}"),
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
                _ => Err(TransferError::Failed(e)),
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
        let mut wait = FIRST_BACKOFF;
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

/// Fetches `GET /api{path}` (a weights file) into `dest` in [`PART_BYTES`] `Range` parts, each after the first
/// with `If-Match` on the first one's ETag: a file replaced meanwhile (`412`) is fetched again from the start, at
/// most [`RESTARTS`] times. The bytes go to a temporary file next to `dest`, which is renamed into place at the end,
/// so `dest` is never a partial file.
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
    'version: for _ in 0..=RESTARTS {
        let mut file = std::fs::File::create(tmp).map_err(TransferError::Io)?;
        let mut etag: Option<String> = None;
        let mut done = 0u64;
        loop {
            if (ctl.stop)() {
                return Err(TransferError::Stopped);
            }
            let range = done..done + PART_BYTES;
            let part = match retry(ctl, || client.get_range(path, range.clone(), etag.as_deref())) {
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
            if part.bytes.is_empty() && done < part.total {
                return Err(TransferError::Failed(bad_gateway(
                    client,
                    "a range of the file came back empty",
                )));
            }
            file.write_all(&part.bytes).map_err(TransferError::Io)?;
            done += part.bytes.len() as u64;
            (ctl.progress)(TransferProgress { done_bytes: done, total_bytes: part.total });
            let version = etag.get_or_insert(part.etag);
            if done >= part.total {
                file.sync_all().map_err(TransferError::Io)?;
                return Ok(Downloaded { bytes: done, etag: version.clone() });
            }
        }
    }
    let message = format!("{path} was replaced {} times while it was downloaded", RESTARTS + 1);
    Err(TransferError::Failed(Error {
        status: Some(StatusCode::PRECONDITION_FAILED),
        error: ApiError {
            code: "cloud_sync_failed".into(),
            message,
            hint: Some("the file changes faster than it can be copied; try again later".into()),
        },
    }))
}

/// Calls `call` until it succeeds, fails for good, or has failed transiently [`ATTEMPTS`] times, waiting longer
/// before each retry.
fn retry<T>(ctl: &Control<'_>, mut call: impl FnMut() -> crate::Result<T>) -> Result<T, TransferError> {
    let mut wait = FIRST_BACKOFF;
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
    if (ctl.stop)() {
        return Err(TransferError::Stopped);
    }
    std::thread::sleep(*wait);
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
