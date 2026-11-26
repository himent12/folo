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
use http::header::{
    AUTHORIZATION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, ETAG, IF_MATCH, RANGE,
};
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

/// Validates that `url` is a well-formed `http://` or `https://` base URL with a non-empty host, optional
/// numeric port (`1..=65535`), no embedded credentials, no query or fragment, and no spaces or control characters.
pub fn check_base_url(url: &str) -> std::result::Result<(), String> {
    if !url.bytes().all(|b| b.is_ascii_graphic()) {
        return Err("URL must not contain spaces or control characters".into());
    }
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or_else(|| "URL must start with http:// or https://".to_string())?;
    if rest.contains('?') || rest.contains('#') {
        return Err("URL must not contain a query or fragment".into());
    }
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.contains('@') {
        return Err("URL must not contain embedded credentials".into());
    }
    let (host, port) = if let Some(after_open) = authority.strip_prefix('[') {
        let (h, tail) =
            after_open.split_once(']').ok_or_else(|| "URL has an invalid IPv6 host".to_string())?;
        if h.parse::<std::net::Ipv6Addr>().is_err() {
            return Err("URL has an invalid IPv6 host".into());
        }
        let port = match tail {
            "" => None,
            s => Some(s.strip_prefix(':').ok_or_else(|| "URL has an invalid host or port".to_string())?),
        };
        (h, port)
    } else if let Some((h, p)) = authority.split_once(':') {
        (h, Some(p))
    } else {
        (authority, None)
    };
    if host.is_empty() || host.contains(['[', ']']) {
        return Err("URL is missing a host".into());
    }
    if let Some(p) = port {
        let valid_port = (!p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
            .then(|| p.parse::<u16>().ok())
            .flatten()
            .is_some_and(|n| n > 0);
        if !valid_port {
            return Err("URL has an invalid port".into());
        }
    }
    let probe = format!("{}/api/health", url.trim_end_matches('/'));
    let uri = http::Uri::try_from(probe.as_str()).map_err(|_| "URL is not a valid HTTP URI".to_string())?;
    if uri.host().is_none_or(str::is_empty) {
        return Err("URL is missing a host".into());
    }
    Ok(())
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

    /// Maps a `ureq::Error` from sending a request: deterministic local request-construction errors (`Http`,
    /// `BadUri`) are non-transient `400 bad_request` so transfers do not waste retries on them; network, DNS, TLS,
    /// and timeout errors remain transient [`UNREACHABLE`].
    fn transport_error(&self, e: ureq::Error) -> Error {
        match e {
            ureq::Error::Http(_) | ureq::Error::BadUri(_) => Error {
                status: Some(StatusCode::BAD_REQUEST),
                error: ApiError {
                    code: "bad_request".into(),
                    message: format!("{}: {}", self.base, self.scrub(&e.to_string())),
                    hint: Some(
                        "check the server URL and token (they must not contain spaces or control characters)"
                            .into(),
                    ),
                },
            },
            other => self.unreachable(other),
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
        let res = res.map_err(|e| self.transport_error(e))?;
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

    /// `POST /api{path}` with a JSON body, returning response headers and the parsed body.
    pub fn post_with_headers<B: Serialize + ?Sized, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<(HeaderMap, T)> {
        self.check_deadline()?;
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        let req = self.auth(self.agent.post(self.url(path))).header(CONTENT_TYPE, "application/json");
        let (headers, b) = self.finish(req.send(&json[..]))?;
        Ok((headers, self.parse(&b)?))
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
        let res = req.call().map_err(|e| StreamError::Http(self.transport_error(e)))?;
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
            _ if range.start == 0 => {
                let encoded = headers
                    .get(CONTENT_ENCODING)
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|v| !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("identity"));
                if encoded {
                    None
                } else if let Some(cl) = headers.get(CONTENT_LENGTH) {
                    let len = cl.to_str().ok().and_then(parse_ascii_u64).ok_or_else(|| {
                        StreamError::Http(
                            self.failure(StatusCode::BAD_GATEWAY, b"invalid Content-Length header"),
                        )
                    })?;
                    Some((len, len))
                } else {
                    None
                }
            }
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
                let msg: &[u8] = if status == StatusCode::PARTIAL_CONTENT {
                    b"206 body exceeded Content-Range length"
                } else {
                    b"response body exceeded Content-Length"
                };
                return Err(StreamError::Http(self.failure(StatusCode::BAD_GATEWAY, msg)));
            }
            out.write_all(&buf[..n]).map_err(StreamError::Io)?;
        }
        let total = match declared {
            Some((expected_len, total)) => {
                if written != expected_len {
                    let msg: &[u8] = if status == StatusCode::PARTIAL_CONTENT {
                        b"206 body length did not match Content-Range"
                    } else {
                        b"response body length did not match Content-Length"
                    };
                    return Err(StreamError::Http(self.failure(StatusCode::BAD_GATEWAY, msg)));
                }
                total
            }
            None => written,
        };
        Ok(StreamedPart { written, total, etag })
    }
}

fn parse_ascii_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<u64>().ok()).flatten()
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
    let parse_u64 = parse_ascii_u64;
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
mod tests;
