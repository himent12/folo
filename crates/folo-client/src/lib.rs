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
use std::time::Duration;

use folo_api::{ApiError, ErrorBody};
use http::header::{AUTHORIZATION, CONTENT_RANGE, CONTENT_TYPE, ETAG, IF_MATCH, RANGE};
use http::{HeaderMap, StatusCode};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Largest response body read into memory (images, metric histories, one part of a download).
pub const MAX_BODY: u64 = 256 * 1024 * 1024;

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

/// A client of one FOLO server. Cheap to clone (the connection pool is shared).
#[derive(Clone)]
pub struct Client {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
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
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(connect))
            .timeout_global(Some(request))
            .build()
            .into();
        Client { base: base.trim_end_matches('/').to_string(), token: None, agent }
    }

    /// Sends `token` as `Authorization: Bearer` (a server started with `--token`). An empty token sends none.
    pub fn token(mut self, token: impl Into<String>) -> Client {
        let token = token.into();
        self.token = (!token.is_empty()).then_some(token);
        self
    }

    /// The server's base URL, without a trailing slash.
    pub fn base(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}/api{path}", self.base)
    }

    fn unreachable(&self, e: impl std::fmt::Display) -> Error {
        Error {
            status: None,
            error: ApiError {
                code: UNREACHABLE.into(),
                message: format!("{}: {e}", self.base),
                hint: Some("is `folo serve` running there, and reachable from this machine?".into()),
            },
        }
    }

    /// `request` with the token, if there is one.
    fn auth<B>(&self, request: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.token {
            Some(t) => request.header(AUTHORIZATION, format!("Bearer {t}")),
            None => request,
        }
    }

    /// Reads an answer: status, headers and body, whatever the status.
    fn read(
        &self,
        res: std::result::Result<http::Response<ureq::Body>, ureq::Error>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>)> {
        let res = res.map_err(|e| self.unreachable(e))?;
        let (status, headers) = (res.status(), res.headers().clone());
        let bytes =
            res.into_body().with_config().limit(MAX_BODY).read_to_vec().map_err(|e| self.unreachable(e))?;
        Ok((status, headers, bytes))
    }

    /// The error of a non-2xx answer: its envelope, or (an answer that is not FOLO's, e.g. a proxy's page) the
    /// start of its text under the code `bad_gateway`.
    fn failure(&self, status: StatusCode, bytes: &[u8]) -> Error {
        let error = match serde_json::from_slice::<ErrorBody>(bytes) {
            Ok(b) => b.error,
            Err(_) => {
                let text = String::from_utf8_lossy(bytes);
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
                message: format!("unexpected response from {}: {e}", self.base),
                hint: Some("the server may be a different FOLO version".into()),
            },
        })
    }

    /// `GET /api{path}`, parsed.
    pub fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let (_, b) = self.finish(self.auth(self.agent.get(self.url(path))).call())?;
        self.parse(&b)
    }

    /// `GET /api{path}` as bytes with the answer's headers (images).
    pub fn get_bytes(&self, path: &str) -> Result<(HeaderMap, Vec<u8>)> {
        self.finish(self.auth(self.agent.get(self.url(path))).call())
    }

    /// `POST /api{path}` with a JSON body, parsed.
    pub fn post<B: Serialize + ?Sized, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        self.send(path, "application/json", &json)
    }

    /// `POST /api{path}` without a body, parsed.
    pub fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let (_, b) = self.finish(self.auth(self.agent.post(self.url(path))).send_empty())?;
        self.parse(&b)
    }

    /// `POST /api{path}` with `body` as `content_type` (multipart forms), parsed.
    pub fn send<T: DeserializeOwned>(&self, path: &str, content_type: &str, body: &[u8]) -> Result<T> {
        let req = self.auth(self.agent.post(self.url(path))).header(CONTENT_TYPE, content_type);
        let (_, b) = self.finish(req.send(body))?;
        self.parse(&b)
    }

    /// `PUT /api{path}` with `body` as `application/octet-stream` (an upload part), parsed.
    pub fn put<T: DeserializeOwned>(&self, path: &str, body: &[u8]) -> Result<T> {
        let req = self.auth(self.agent.put(self.url(path))).header(CONTENT_TYPE, "application/octet-stream");
        let (_, b) = self.finish(req.send(body))?;
        self.parse(&b)
    }

    /// `PUT /api{path}` with a JSON body, parsed.
    pub fn put_json<B: Serialize + ?Sized, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        let req = self.auth(self.agent.put(self.url(path))).header(CONTENT_TYPE, "application/json");
        let (_, b) = self.finish(req.send(&json[..]))?;
        self.parse(&b)
    }

    /// `DELETE /api{path}`.
    pub fn delete(&self, path: &str) -> Result<()> {
        self.finish(self.auth(self.agent.delete(self.url(path))).call()).map(|_| ())
    }

    /// `DELETE /api{path}` of an endpoint that answers with a body (`DELETE /api/cloud/instances/{id}`), parsed.
    pub fn delete_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let (_, b) = self.finish(self.auth(self.agent.delete(self.url(path))).call())?;
        self.parse(&b)
    }

    /// `GET /api{path}` for the bytes `range` of a file, with `If-Match: etag` when `etag` is given: a file that
    /// was replaced since answers `412 Precondition Failed` (an [`Error`] with that status) instead of bytes of
    /// another version. A range starting at or past the end of the file is `416`.
    pub fn get_range(&self, path: &str, range: Range<u64>, etag: Option<&str>) -> Result<Part> {
        let mut req = self
            .auth(self.agent.get(self.url(path)))
            .header(RANGE, format!("bytes={}-{}", range.start, range.end.saturating_sub(1)));
        if let Some(etag) = etag {
            req = req.header(IF_MATCH, etag);
        }
        let (status, headers, bytes) = self.read(req.call())?;
        if !status.is_success() {
            return Err(self.failure(status, &bytes));
        }
        let etag = headers.get(ETAG).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
        // `206` names the file's length after the slash of `Content-Range: bytes a-b/total`; a server that ignores
        // `Range` answers `200` with the whole file.
        let total = match status {
            StatusCode::PARTIAL_CONTENT => headers
                .get(CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.rsplit('/').next())
                .and_then(|t| t.trim().parse().ok())
                .ok_or_else(|| {
                    self.failure(StatusCode::BAD_GATEWAY, b"a 206 answer without Content-Range")
                })?,
            _ if range.start == 0 => bytes.len() as u64,
            _ => return Err(self.failure(StatusCode::BAD_GATEWAY, b"the server ignored the Range header")),
        };
        Ok(Part { bytes, total, etag })
    }
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
}
