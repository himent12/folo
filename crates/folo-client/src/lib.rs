//! A blocking client for a FOLO server's HTTP API (docs/API.md).
//!
//! [`Client`] is the transport every FOLO-to-FOLO caller shares: `folo mcp --server` (the proxy store) and the
//! cloud control plane talking to a worker (docs/CLOUD.md). It knows the API's conventions and nothing about its
//! routes: paths are relative to `/api`, bodies are JSON, a non-2xx answer is the error envelope
//! `{ "error": { code, message, hint } }`, and a server started with `--token` gets `Authorization: Bearer`.
//! `http://` and `https://` (rustls, the bundled web PKI roots) both work.
//!
//! Calls block; async callers run them on a blocking pool. [`transfer`] moves files in parts on top of it.
//!
//! This file holds [`Client`] (JSON calls, raw bytes, `Range` parts), [`Error`] and its codes ([`UNREACHABLE`],
//! [`NOT_FOLO`]) and [`encode`] for path segments.

pub mod transfer;

use std::ops::Range;
use std::time::{Duration, Instant};

use folo_api::{ApiError, ErrorBody};
use http::header::{AUTHORIZATION, CONTENT_RANGE, CONTENT_TYPE, ETAG, IF_MATCH, RANGE};
use http::{HeaderMap, StatusCode};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Largest response body read into memory (images, metric histories, one part of a download).
pub const MAX_BODY: u64 = 256 * 1024 * 1024;
/// Largest error response body read into memory on a non-2xx response.
const MAX_ERROR_BODY: u64 = 1024 * 1024;
/// Shortest token blanked out of error messages (shorter strings would shred ordinary words).
const MIN_SECRET_CHARS: usize = 8;
/// Chunk size used when streaming a `Range` response directly to a writer.
const STREAM_CHUNK: usize = 64 * 1024;

/// The code of an [`Error`] that never reached the server (refused, timed out, TLS failure, a cut connection).
pub const UNREACHABLE: &str = "server_unreachable";
/// The code of an answer that is not FOLO's (no error envelope): a proxy's page, or another server.
pub const NOT_FOLO: &str = "bad_gateway";

/// A failed request: the server's error envelope with its status, or a transport failure.
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    /// HTTP status of the answer; `None` when there was none ([`UNREACHABLE`]).
    pub status: Option<StatusCode>,
    /// The error, as the API reports errors.
    pub error: ApiError,
}

impl Error {
    /// True when the request got no answer, so trying again may work.
    pub fn is_unreachable(&self) -> bool {
        self.status.is_none()
    }

    /// True for a failure that may pass when the same request is sent again: no answer, or an answer that blames
    /// the moment rather than the request (`429`, `502`, `503`, `504`, and the `408` / `520`–`524` of proxies).
    /// So does a `404` that is not FOLO's own (no error envelope): a proxy in front of the server saying it cannot
    /// reach it right now. RunPod's answers so for a pod's port, also for a moment after the port has answered.
    pub fn is_transient(&self) -> bool {
        match self.status {
            None => true,
            Some(s) if s.as_u16() == 404 => self.error.code == NOT_FOLO,
            Some(s) => matches!(s.as_u16(), 408 | 429 | 502 | 503 | 504 | 520..=524),
        }
    }

    /// The stable error code (`folo_api::codes`).
    pub fn code(&self) -> &str {
        &self.error.code
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error.message)
    }
}

impl std::error::Error for Error {}

/// `Result` of a [`Client`] call.
pub type Result<T> = std::result::Result<T, Error>;

/// One part of a file fetched with a `Range` request ([`Client::get_range`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// The bytes of the requested range (fewer at the end of the file).
    pub bytes: Vec<u8>,
    /// Length of the whole file.
    pub total: u64,
    /// The file's ETag, quoted as the server sent it.
    pub etag: String,
}

/// Metadata of one part streamed with a `Range` request ([`Client::get_range_to`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamedPart {
    /// Number of bytes written to the destination sink.
    pub written: u64,
    /// Length of the whole file.
    pub total: u64,
    /// The file's ETag, quoted as the server sent it.
    pub etag: String,
}

/// Why [`Client::get_range_to`] did not finish streaming a range.
#[derive(Debug)]
pub enum StreamError {
    /// The caller's stop callback returned `true` while streaming.
    Stopped,
    /// An HTTP error answer or transport failure.
    Http(Error),
    /// Writing to the destination sink failed.
    Io(std::io::Error),
}

/// A client of one FOLO server. Cheap to clone (the connection pool is shared).
#[derive(Clone)]
pub struct Client {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
    /// When set, every request ends by then ([`Client::until`]).
    deadline: Option<Instant>,
}

// Not derived: the token must not reach a log through `{:?}`.
impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client").field("base", &self.base).field("token", &self.token.is_some()).finish()
    }
}

/// Percent-encodes one path segment or query value.
pub fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

impl Client {
    /// A client of the server at `base` (e.g. `http://127.0.0.1:7878`): 5 s to connect, 10 minutes per request
    /// (validation and exports answer when they are done).
    pub fn new(base: &str) -> Client {
        Client::with_timeouts(base, Duration::from_secs(5), Duration::from_secs(600))
    }

    /// [`Client::new`] with its own timeouts: `connect` to establish a connection, `request` for a whole call,
    /// body included.
    pub fn with_timeouts(base: &str, connect: Duration, request: Duration) -> Client {
        // Never follow HTTP redirects: FOLO endpoints do not redirect, and following one would risk treating a
        // proxy or captive portal's answer as the server's (or sending requests to an unexpected host).
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .max_redirects_will_error(false)
            .timeout_connect(Some(connect))
            .timeout_global(Some(request))
            .build()
            .into();
        Client { base: base.trim_end_matches('/').to_string(), token: None, agent, deadline: None }
    }

    /// Sends `token` as `Authorization: Bearer` (a server started with `--token`). An empty token sends none.
    pub fn token(mut self, token: impl Into<String>) -> Client {
        let token = token.into();
        self.token = (!token.is_empty()).then_some(token);
        self
    }

    /// This client with every request ending by `deadline` (body included), for a caller that has promised an
    /// answer by then: one past it fails as [`UNREACHABLE`]. Shares the connection pool.
    pub fn until(&self, deadline: Instant) -> Client {
        Client { deadline: Some(deadline), ..self.clone() }
    }

    /// The server's base URL, without a trailing slash.
    pub fn base(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}/api{path}", self.base)
    }

    fn scrub(&self, text: &str) -> String {
        match &self.token {
            Some(t) if t.chars().count() >= MIN_SECRET_CHARS => text.replace(t.as_str(), "[redacted]"),
            _ => text.to_string(),
        }
    }

    fn check_deadline(&self) -> Result<()> {
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(self.unreachable("request deadline exceeded"));
        }
        Ok(())
    }

    fn unreachable(&self, e: impl std::fmt::Display) -> Error {
        Error {
            status: None,
            error: ApiError {
                code: UNREACHABLE.into(),
                message: format!("{}: {}", self.base, self.scrub(&e.to_string())),
                hint: Some("is `folo serve` running there, and reachable from this machine?".into()),
            },
        }
    }

    /// `request` with the token, if there is one, and what is left until the deadline, if there is one.
    fn auth<B>(&self, request: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        let request = match self.deadline {
            Some(d) => {
                let left = d.saturating_duration_since(Instant::now()).max(Duration::from_millis(1));
                request.config().timeout_global(Some(left)).build()
            }
            None => request,
        };
        match &self.token {
            Some(t) => request.header(AUTHORIZATION, format!("Bearer {t}")),
            None => request,
        }
    }

    /// Reads an answer: status, headers and body, whatever the status. Non-2xx bodies are capped at
    /// [`MAX_ERROR_BODY`] so a huge proxy error page cannot exhaust memory or hide the HTTP status.
    fn read(
        &self,
        res: std::result::Result<http::Response<ureq::Body>, ureq::Error>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>)> {
        let res = res.map_err(|e| self.unreachable(e))?;
        let (status, headers) = (res.status(), res.headers().clone());
        let bytes = if status.is_success() {
            res.into_body().with_config().limit(MAX_BODY).read_to_vec().map_err(|e| self.unreachable(e))?
        } else {
            let mut bytes = Vec::new();
            let mut limited = std::io::Read::take(res.into_body().into_reader(), MAX_ERROR_BODY);
            let _ = std::io::Read::read_to_end(&mut limited, &mut bytes);
            bytes
        };
        Ok((status, headers, bytes))
    }

    /// The error of a non-2xx answer: its envelope, or (an answer that is not FOLO's, e.g. a proxy's page) the
    /// start of its text under the code `bad_gateway`. Any echoed bearer token is scrubbed before truncation.
    fn failure(&self, status: StatusCode, bytes: &[u8]) -> Error {
        let error = match serde_json::from_slice::<ErrorBody>(bytes) {
            Ok(mut b) => {
                // Never forward a remote `error.code` that echoes our bearer token (of any length), while
                // keeping known FOLO error codes intact.
                if let Some(t) = &self.token
                    && folo_api::codes::find(&b.error.code).is_none()
                    && b.error.code.contains(t.as_str())
                {
                    b.error.code = NOT_FOLO.into();
                }
                b.error.message = self.scrub(&b.error.message);
                b.error.hint = b.error.hint.map(|h| self.scrub(&h));
                b.error
            }
            Err(_) => {
                let text = self.scrub(&String::from_utf8_lossy(bytes));
                let text = folo_api::codes::cap(text.trim(), 200);
                let what = if text.is_empty() { String::new() } else { format!(": {text}") };
                ApiError {
                    code: NOT_FOLO.into(),
                    message: format!("{} answered {status} without a FOLO error{what}", self.base),
                    hint: None,
                }
            }
        };
        Error { status: Some(status), error }
    }

    /// The headers and body of a 2xx answer; any other status is its error.
    fn finish(
        &self,
        res: std::result::Result<http::Response<ureq::Body>, ureq::Error>,
    ) -> Result<(HeaderMap, Vec<u8>)> {
        let (status, headers, bytes) = self.read(res)?;
        if status.is_success() { Ok((headers, bytes)) } else { Err(self.failure(status, &bytes)) }
    }

    fn parse<T: DeserializeOwned>(&self, bytes: &[u8]) -> Result<T> {
        serde_json::from_slice(bytes).map_err(|e| Error {
            status: Some(StatusCode::BAD_GATEWAY),
            error: ApiError {
                code: "bad_gateway".into(),
                message: format!("unexpected response from {}: {}", self.base, self.scrub(&e.to_string())),
                hint: Some("the server may be a different FOLO version".into()),
            },
        })
    }

    /// `GET /api{path}`, parsed.
    pub fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.check_deadline()?;
        let (_, b) = self.finish(self.auth(self.agent.get(self.url(path))).call())?;
        self.parse(&b)
    }

    /// `GET /api{path}` as bytes with the answer's headers (images).
    pub fn get_bytes(&self, path: &str) -> Result<(HeaderMap, Vec<u8>)> {
        self.check_deadline()?;
        self.finish(self.auth(self.agent.get(self.url(path))).call())
    }

    /// `POST /api{path}` with a JSON body, parsed.
    pub fn post<B: Serialize + ?Sized, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        self.send(path, "application/json", &json)
    }

    /// `POST /api{path}` without a body, parsed.
    pub fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.check_deadline()?;
        let (_, b) = self.finish(self.auth(self.agent.post(self.url(path))).send_empty())?;
        self.parse(&b)
    }

    /// `POST /api{path}` with `body` as `content_type` (multipart forms), parsed.
    pub fn send<T: DeserializeOwned>(&self, path: &str, content_type: &str, body: &[u8]) -> Result<T> {
        self.check_deadline()?;
        let req = self.auth(self.agent.post(self.url(path))).header(CONTENT_TYPE, content_type);
        let (_, b) = self.finish(req.send(body))?;
        self.parse(&b)
    }

    /// `PUT /api{path}` with `body` as `application/octet-stream` (an upload part), parsed.
    pub fn put<T: DeserializeOwned>(&self, path: &str, body: &[u8]) -> Result<T> {
        self.check_deadline()?;
        let req = self.auth(self.agent.put(self.url(path))).header(CONTENT_TYPE, "application/octet-stream");
        let (_, b) = self.finish(req.send(body))?;
        self.parse(&b)
    }

    /// `PUT /api{path}` with a JSON body, parsed.
    pub fn put_json<B: Serialize + ?Sized, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        self.check_deadline()?;
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        let req = self.auth(self.agent.put(self.url(path))).header(CONTENT_TYPE, "application/json");
        let (_, b) = self.finish(req.send(&json[..]))?;
        self.parse(&b)
    }

    /// `DELETE /api{path}`.
    pub fn delete(&self, path: &str) -> Result<()> {
        self.check_deadline()?;
        self.finish(self.auth(self.agent.delete(self.url(path))).call()).map(|_| ())
    }

    /// `DELETE /api{path}` of an endpoint that answers with a body (`DELETE /api/cloud/instances/{id}`), parsed.
    pub fn delete_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.check_deadline()?;
        let (_, b) = self.finish(self.auth(self.agent.delete(self.url(path))).call())?;
        self.parse(&b)
    }

    /// `GET /api{path}` for the bytes `range` of a file, with `If-Match: etag` when `etag` is given: a file that
    /// was replaced since answers `412 Precondition Failed` (an [`Error`] with that status) instead of bytes of
    /// another version. A range starting at or past the end of the file is `416`.
    pub fn get_range(&self, path: &str, range: Range<u64>, etag: Option<&str>) -> Result<Part> {
        let mut bytes = Vec::new();
        let meta = self.get_range_to(path, range, etag, &mut bytes, || false).map_err(|e| match e {
            StreamError::Http(e) => e,
            StreamError::Io(e) => self.unreachable(e),
            StreamError::Stopped => unreachable!("stop closure always returns false"),
        })?;
        Ok(Part { bytes, total: meta.total, etag: meta.etag })
    }

    /// `GET /api{path}` for the bytes `range` of a file, streaming the body directly into `out` in small chunks
    /// instead of buffering the whole part in memory. Checks `stop` between chunks so a large range stops promptly.
    ///
    /// On a mid-stream transport failure (`StreamError::Http`) or stop (`StreamError::Stopped`), some prefix of the
    /// range may already have been written to `out`; callers that retry must rewind or discard those bytes first.
    pub fn get_range_to<W: std::io::Write>(
        &self,
        path: &str,
        range: Range<u64>,
        etag: Option<&str>,
        out: &mut W,
        stop: impl Fn() -> bool,
    ) -> std::result::Result<StreamedPart, StreamError> {
        if stop() {
            return Err(StreamError::Stopped);
        }
        if range.is_empty() {
            return Err(StreamError::Http(Error {
                status: Some(StatusCode::BAD_REQUEST),
                error: ApiError {
                    code: "bad_request".into(),
                    message: format!("{}: empty byte range {}..{}", self.base, range.start, range.end),
                    hint: None,
                },
            }));
        }
        self.check_deadline().map_err(StreamError::Http)?;
        let mut req = self
            .auth(self.agent.get(self.url(path)))
            .header(RANGE, format!("bytes={}-{}", range.start, range.end - 1));
        if let Some(etag) = etag {
            req = req.header(IF_MATCH, etag);
        }
        let res = req.call().map_err(|e| StreamError::Http(self.unreachable(e)))?;
        let status = res.status();
        let headers = res.headers().clone();
        if !status.is_success() {
            // Read only a bounded diagnostic prefix so a huge non-2xx proxy body cannot exhaust memory or hide
            // the HTTP status code behind a transport error.
            let mut bytes = Vec::new();
            let mut limited = std::io::Read::take(res.into_body().into_reader(), MAX_ERROR_BODY);
            let _ = std::io::Read::read_to_end(&mut limited, &mut bytes);
            return Err(StreamError::Http(self.failure(status, &bytes)));
        }
        let etag = headers.get(ETAG).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
        // `206` must advertise `Content-Range: bytes <start>-<end>/<total>` starting at `range.start`; a server
        // that ignores `Range` answers `200` with the whole file (only valid when `range.start == 0`). Validate
        // headers before writing any bytes to `out`.
        let declared = match status {
            StatusCode::PARTIAL_CONTENT => Some(
                headers
                    .get(CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| parse_content_range(v, range.start))
                    .ok_or_else(|| {
                        StreamError::Http(
                            self.failure(StatusCode::BAD_GATEWAY, b"a 206 answer with invalid Content-Range"),
                        )
                    })?,
            ),
            _ if range.start == 0 => None,
            _ => {
                return Err(StreamError::Http(
                    self.failure(StatusCode::BAD_GATEWAY, b"the server ignored the Range header"),
                ));
            }
        };
        if let Some((expected_len, _)) = declared
            && expected_len > MAX_BODY
        {
            return Err(StreamError::Http(
                self.unreachable(format!("response body exceeded {MAX_BODY} bytes")),
            ));
        }

        let mut reader = res.into_body().into_reader();
        let mut buf = vec![0u8; STREAM_CHUNK];
        let mut written = 0u64;
        loop {
            if stop() {
                return Err(StreamError::Stopped);
            }
            let n = match std::io::Read::read(&mut reader, &mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(StreamError::Http(self.unreachable(e))),
            };
            written = written.saturating_add(n as u64);
            if written > MAX_BODY {
                return Err(StreamError::Http(
                    self.unreachable(format!("response body exceeded {MAX_BODY} bytes")),
                ));
            }
            if let Some((expected_len, _)) = declared
                && written > expected_len
            {
                return Err(StreamError::Http(
                    self.failure(StatusCode::BAD_GATEWAY, b"206 body exceeded Content-Range length"),
                ));
            }
            out.write_all(&buf[..n]).map_err(StreamError::Io)?;
        }
        let total = match declared {
            Some((expected_len, total)) => {
                if written != expected_len {
                    return Err(StreamError::Http(
                        self.failure(StatusCode::BAD_GATEWAY, b"206 body length did not match Content-Range"),
                    ));
                }
                total
            }
            None => written,
        };
        Ok(StreamedPart { written, total, etag })
    }
}

/// Parses `Content-Range: bytes <start>-<end>/<total>`, requiring `<start> == expected_start` and
/// `start <= end < total`. Returns `(expected_len, total)`.
fn parse_content_range(header: &str, expected_start: u64) -> Option<(u64, u64)> {
    let rest = header.trim();
    let (unit, spec) = rest.split_once(char::is_whitespace)?;
    if !unit.eq_ignore_ascii_case("bytes") {
        return None;
    }
    let (range_part, total_part) = spec.trim().split_once('/')?;
    let (start_part, end_part) = range_part.trim().split_once('-')?;
    let parse_u64 = |s: &str| {
        let s = s.trim();
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<u64>().ok()).flatten()
    };
    let start = parse_u64(start_part)?;
    let end = parse_u64(end_part)?;
    let total = parse_u64(total_part)?;
    if start != expected_start || start > end || end >= total {
        return None;
    }
    let expected_len = end.checked_sub(start)?.checked_add(1)?;
    Some((expected_len, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_segments() {
        assert_eq!(encode("ds_1"), "ds_1");
        assert_eq!(encode("a/b c"), "a%2Fb%20c");
    }

    #[test]
    fn an_unreachable_server_is_transient_and_keeps_the_token_out_of_debug() {
        // Port 9 (discard) is essentially never served on loopback. Windows retries a refused loopback connection
        // for about 2 s, so the connection times out first: refused and timed out are both unreachable.
        let c = Client::with_timeouts(
            "http://127.0.0.1:9/",
            Duration::from_millis(100),
            Duration::from_millis(100),
        )
        .token("s3cret");
        assert_eq!(c.base(), "http://127.0.0.1:9");
        assert!(!format!("{c:?}").contains("s3cret"));
        let e = c.get::<serde_json::Value>("/health").unwrap_err();
        assert_eq!((e.code(), e.status), (UNREACHABLE, None));
        assert!(e.is_unreachable() && e.is_transient());
        assert!(!format!("{e}").contains("s3cret"));
    }

    #[test]
    fn a_deadline_ends_a_call_the_server_never_answers() {
        // Connections wait in the listener's backlog: accepted by the OS, never answered.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let c = Client::new(&format!("http://{}", listener.local_addr().unwrap()));
        // On a thread: without the deadline the call holds for the 10-minute timeout, the watchdog fails the test,
        // and closing the listener ends the call, so no thread outlives the test.
        let r = std::thread::scope(|scope| {
            let (done, answer) = std::sync::mpsc::channel();
            scope.spawn(move || {
                let r =
                    c.until(Instant::now() + Duration::from_millis(50)).get::<serde_json::Value>("/health");
                let _ = done.send(r);
            });
            let got = answer.recv_timeout(Duration::from_secs(60));
            drop(listener);
            got.expect("the deadline did not end the call")
        });
        assert!(r.unwrap_err().is_unreachable());
    }

    #[test]
    fn only_some_statuses_are_transient() {
        let c = Client::new("http://127.0.0.1:9");
        let body = br#"{"error":{"code":"run_not_found","message":"no run 'x'","hint":null}}"#;
        let e = c.failure(StatusCode::NOT_FOUND, body);
        assert_eq!((e.code(), e.is_transient()), ("run_not_found", false));
        // A proxy's own page, not FOLO's envelope.
        let e = c.failure(StatusCode::BAD_GATEWAY, b"<html>502 Bad Gateway</html>");
        assert_eq!((e.code(), e.is_transient()), ("bad_gateway", true));
        assert!(e.error.message.contains("502 Bad Gateway"), "{}", e.error.message);
        // A proxy's 404 for a server it cannot reach at the moment (RunPod's, for a pod's port) passes; so do
        // Cloudflare's 52x. FOLO's own 404 and a plain 400 or 500 do not.
        let e = c.failure(StatusCode::NOT_FOUND, b"404 page not found");
        assert_eq!((e.code(), e.is_transient()), ("bad_gateway", true));
        for (status, transient) in
            [(400, false), (500, false), (520, true), (522, true), (524, true), (525, false)]
        {
            let e = c.failure(StatusCode::from_u16(status).unwrap(), b"");
            assert_eq!(e.is_transient(), transient, "{status}");
        }
    }

    #[test]
    fn get_range_to_streams_validates_headers_and_checks_stop() {
        use std::io::{Read, Write};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        fn serve_raw(responses: Vec<Vec<u8>>) -> (String, std::thread::JoinHandle<()>) {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let handle = std::thread::spawn(move || {
                for resp in responses {
                    let (mut stream, _) = listener.accept().unwrap();
                    let mut req_buf = [0u8; 1024];
                    let _ = stream.read(&mut req_buf);
                    let _ = stream.write_all(&resp);
                    let _ = stream.flush();
                }
            });
            (format!("http://{addr}"), handle)
        }

        let payload: Vec<u8> = (0..(STREAM_CHUNK * 2 + 17)).map(|i| (i % 251) as u8).collect();
        let total = payload.len();

        let mut ok_206 = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-{}/{total}\r\nETag: \"v1\"\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n",
            total - 1
        )
        .into_bytes();
        ok_206.extend_from_slice(&payload);

        let missing_cr =
            b"HTTP/1.1 206 Partial Content\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd"
                .to_vec();
        let mismatched_offset =
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 4-7/8\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd"
                .to_vec();
        let mismatched_len =
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-5/8\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd"
                .to_vec();
        let ignored_range =
            b"HTTP/1.1 200 OK\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd".to_vec();
        let huge_403_len = (MAX_ERROR_BODY as usize) + 64 * 1024;
        let mut huge_403 =
            format!("HTTP/1.1 403 Forbidden\r\nContent-Length: {huge_403_len}\r\nConnection: close\r\n\r\n")
                .into_bytes();
        huge_403.resize(huge_403.len() + huge_403_len, b'e');
        let truncated = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-99/100\r\nETag: \"v1\"\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{}",
            "x".repeat(20)
        )
        .into_bytes();
        let stop_body = ok_206.clone();

        let (base, handle) = serve_raw(vec![
            ok_206,
            missing_cr,
            mismatched_offset,
            mismatched_len,
            ignored_range,
            huge_403,
            truncated,
            stop_body,
        ]);
        let client = Client::new(&base);

        // 1. Streams full multi-chunk 206 response directly to writer.
        let mut out = Vec::new();
        let meta = client.get_range_to("/m/weights", 0..total as u64, None, &mut out, || false).unwrap();
        assert_eq!(meta, StreamedPart { written: total as u64, total: total as u64, etag: "\"v1\"".into() });
        assert_eq!(out, payload);

        // 2. Rejects 206 without Content-Range before writing anything to `out`.
        let mut out = Vec::new();
        let err = client.get_range_to("/m/weights", 0..4, None, &mut out, || false).unwrap_err();
        assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));
        assert!(out.is_empty());

        // 3. Rejects 206 whose Content-Range start offset does not match requested start.
        let mut out = Vec::new();
        let err = client.get_range_to("/m/weights", 0..4, None, &mut out, || false).unwrap_err();
        assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));
        assert!(out.is_empty());

        // 4. Rejects 206 whose body length does not match the advertised Content-Range span.
        let mut out = Vec::new();
        let err = client.get_range_to("/m/weights", 0..6, None, &mut out, || false).unwrap_err();
        assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));

        // 5. Rejects 200 for non-zero start before writing anything to `out`.
        let mut out = Vec::new();
        let err = client.get_range_to("/m/weights", 10..14, None, &mut out, || false).unwrap_err();
        assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));
        assert!(out.is_empty());

        // 6. Preserves HTTP status (and non-transient classification) even when an error body exceeds MAX_ERROR_BODY.
        let mut out = Vec::new();
        let err = client.get_range_to("/m/weights", 0..4, None, &mut out, || false).unwrap_err();
        assert!(
            matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::FORBIDDEN) && !e.is_transient())
        );
        assert!(out.is_empty());

        // 7. Mid-stream truncated TCP body returns a transient transport error.
        let mut out = Vec::new();
        let err = client.get_range_to("/m/weights", 0..100, None, &mut out, || false).unwrap_err();
        assert!(matches!(err, StreamError::Http(e) if e.is_transient()));

        // 8. Stopping mid-stream aborts between chunks without reading the entire body.
        let stopped = Arc::new(AtomicBool::new(false));
        struct StopAfterFirstChunk {
            stopped: Arc<AtomicBool>,
            bytes: usize,
        }
        impl std::io::Write for StopAfterFirstChunk {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.bytes += buf.len();
                self.stopped.store(true, Ordering::SeqCst);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut sink = StopAfterFirstChunk { stopped: stopped.clone(), bytes: 0 };
        let err = client
            .get_range_to("/m/weights", 0..total as u64, None, &mut sink, || stopped.load(Ordering::SeqCst))
            .unwrap_err();
        assert!(matches!(err, StreamError::Stopped));
        assert!(sink.bytes > 0 && sink.bytes < total, "stopped mid-stream at {} / {total}", sink.bytes);

        handle.join().unwrap();
    }

    #[test]
    fn redirects_are_not_followed() {
        use std::io::{Read, Write};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct StopOnDrop {
            flag: Arc<AtomicBool>,
            addr: std::net::SocketAddr,
        }
        impl Drop for StopOnDrop {
            fn drop(&mut self) {
                self.flag.store(true, Ordering::Relaxed);
                let _ = std::net::TcpStream::connect_timeout(&self.addr, Duration::from_secs(1));
            }
        }

        let io_timeout = Some(Duration::from_secs(5));
        // A second server that must never be contacted if redirects are disabled.
        let target_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let target_addr = target_listener.local_addr().unwrap();
        let target_hit = Arc::new(AtomicBool::new(false));
        let stop_target = Arc::new(AtomicBool::new(false));

        let redirect_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let redirect_addr = redirect_listener.local_addr().unwrap();
        let stop_redirect = Arc::new(AtomicBool::new(false));

        std::thread::scope(|scope| {
            let hit_clone = target_hit.clone();
            let stop_t = stop_target.clone();
            scope.spawn(move || {
                for stream in target_listener.incoming() {
                    if stop_t.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Ok(mut s) = stream {
                        let _ = s.set_read_timeout(io_timeout);
                        let _ = s.set_write_timeout(io_timeout);
                        hit_clone.store(true, Ordering::Relaxed);
                        let mut buf = [0u8; 512];
                        let _ = s.read(&mut buf);
                        let _ = s.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                        );
                    }
                }
            });
            let _stop_t = StopOnDrop { flag: stop_target.clone(), addr: target_addr };

            let stop_r = stop_redirect.clone();
            scope.spawn(move || {
                if let Ok((mut s, _)) = redirect_listener.accept()
                    && !stop_r.load(Ordering::Relaxed)
                {
                    let _ = s.set_read_timeout(io_timeout);
                    let _ = s.set_write_timeout(io_timeout);
                    let mut buf = [0u8; 1024];
                    let _ = s.read(&mut buf);
                    let resp = format!(
                        "HTTP/1.1 302 Found\r\nLocation: http://{target_addr}/api/health\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = s.write_all(resp.as_bytes());
                }
            });
            let _stop_r = StopOnDrop { flag: stop_redirect.clone(), addr: redirect_addr };

            let client = Client::with_timeouts(
                &format!("http://{redirect_addr}"),
                Duration::from_secs(2),
                Duration::from_secs(5),
            )
            .token("s3cret");
            let err = client.get::<serde_json::Value>("/health").unwrap_err();
            assert_eq!(err.status, Some(StatusCode::FOUND), "{err:?}");
            assert_eq!(err.code(), NOT_FOLO, "{err:?}");
            assert!(!err.is_transient(), "{err:?}");
            assert!(!target_hit.load(Ordering::Relaxed), "redirect target must not be contacted");
        });
    }

    #[test]
    fn non_2xx_error_bodies_are_capped_tokens_scrubbed_and_expired_deadlines_never_send() {
        use std::io::{Read, Write};

        const SECRET: &str = "tok_secret_1234567890";
        let c = Client::new("http://127.0.0.1:9").token(SECRET);

        // Echoed bearer token is scrubbed from FOLO error envelopes (including arbitrary `error.code`) and raw
        // proxy bodies (even across the 200-char truncation cut).
        let env = format!(
            r#"{{"error":{{"code":"err_{SECRET}","message":"bad token {SECRET}","hint":"check {SECRET}"}}}}"#
        );
        let e = c.failure(StatusCode::UNAUTHORIZED, env.as_bytes());
        assert_eq!(e.code(), NOT_FOLO, "{e:?}");
        assert!(!e.error.message.contains(SECRET) && e.error.message.contains("[redacted]"), "{e:?}");
        assert!(
            e.error.hint.as_deref().is_some_and(|h| !h.contains(SECRET) && h.contains("[redacted]")),
            "{e:?}"
        );
        let raw = format!("{} {SECRET}", "x".repeat(195));
        let e = c.failure(StatusCode::BAD_GATEWAY, raw.as_bytes());
        assert!(!e.error.message.contains(&SECRET[..8]), "{e:?}");

        // Empty byte range is rejected locally without touching the network.
        let mut out = Vec::new();
        let err = c.get_range_to("/m/weights", 5..5, None, &mut out, || false).unwrap_err();
        assert!(
            matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_REQUEST) && !e.is_transient())
        );

        // Already-expired deadline fails immediately without contacting the server.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap_or_else(Instant::now);
        let expired = Client::new(&format!("http://{addr}")).until(past);
        let e = expired.post_empty::<serde_json::Value>("/runs").unwrap_err();
        assert!(e.is_unreachable());
        assert!(listener.accept().is_err(), "expired deadline must not connect");

        // Huge non-2xx body on normal JSON call is bounded and preserves the real non-transient HTTP status.
        listener.set_nonblocking(false).unwrap();
        let huge_len = (MAX_ERROR_BODY as usize) + 64 * 1024;
        let srv = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 512];
            let _ = s.read(&mut buf);
            let mut resp =
                format!("HTTP/1.1 403 Forbidden\r\nContent-Length: {huge_len}\r\nConnection: close\r\n\r\n")
                    .into_bytes();
            resp.resize(resp.len() + huge_len, b'e');
            let _ = s.write_all(&resp);
        });
        let e = Client::new(&format!("http://{addr}")).get::<serde_json::Value>("/health").unwrap_err();
        srv.join().unwrap();
        assert_eq!(e.status, Some(StatusCode::FORBIDDEN));
        assert!(!e.is_transient());
    }
}
