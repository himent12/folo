//! The transport: HTTP agents, path encoding, and the JSON requests and error decoding every call
//! goes through.

use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::api::*;

use super::HttpBackend;

/// The scheme of a request to an address that isn't one: ureq knows only http and https, so it
/// fails such a request before a lookup or a connect.
pub(super) const NOT_SENT: &str = "folo-not-sent";

/// The most of a body [`json`] reads unless told otherwise: ureq's own default.
pub(super) const BODY_LIMIT: u64 = 10 << 20;

/// A JSON body as `T`: read whole (at most `limit` bytes), then parsed from the bytes. ureq's `read_json` would
/// compile two parsers of every response type, one over a slice and one over the socket, and parsing a slice is
/// the faster one; those copies were most of the HTTP client's machine code (and build time).
pub(super) fn json<T: DeserializeOwned>(body: &mut ureq::Body, limit: u64) -> Result<T, ureq::Error> {
    let bytes = body.with_config().limit(limit).read_to_vec()?;
    serde_json::from_slice(&bytes).map_err(ureq::Error::Json)
}

pub(super) fn agent(timeout: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_global(timeout)
        .build()
        .into()
}

/// An agent for event streams: the handshake (connect, response headers) is bounded by `headers`, and
/// every read of the body by `silence`. The server sends a keep-alive every 15 s (docs/API.md), so a
/// stream that says nothing for longer has a dead link under it (a stuck proxy, a sleeping laptop, a
/// NAT that forgot it): the read fails and the reader ends, instead of blocking for good while its
/// page waits for events that never come.
pub(super) fn stream_agent(headers: Duration, silence: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_recv_response(Some(headers))
        .timeout_per_read(Some(silence))
        .build()
        .into()
}

/// A server message or hint as a sentence, as Studio shows them (`web/src/lib/format.ts`,
/// `sentence`): a capital first letter and a closing full stop ("the dataset has no training
/// images" → "The dataset has no training images."). One that starts with an identifier in
/// backticks keeps its case: it is a name, not a word.
pub(super) fn sentence(s: &str) -> String {
    let t = s.trim();
    let mut chars = t.chars();
    let mut out: String = match chars.next() {
        None => return String::new(),
        Some('`') => t.to_string(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    };
    if !out.ends_with(['.', '!', '?']) {
        out.push('.');
    }
    out
}

/// A server started with `--token` (every `/api` call but `/api/health` answers `401 unauthorized`): the app
/// has no way to send a token, so it says what works instead.
pub(crate) fn needs_token() -> ApiError {
    ApiError::new("unauthorized", "This server needs a token").with_hint(
        "It was started with `folo serve --token`, and FOLO Desktop can't send one. Start it without `--token`, or connect to another server.",
    )
}

/// A 4xx or 5xx answer without FOLO's error JSON (none, or not whole).
pub(super) fn status_error(status: u16) -> ApiError {
    match status {
        401 => needs_token(),
        _ => ApiError::new("http_error", format!("folo serve answered HTTP {status}")).with_status(status),
    }
}

/// FOLO's error JSON as an [`ApiError`], its words as sentences.
pub(super) fn server_error(b: folo_api::ErrorBody, status: u16) -> ApiError {
    // `folo serve --token`: the server's words are for an HTTP client that can send one.
    if b.error.code == "unauthorized" {
        return needs_token();
    }
    ApiError {
        code: b.error.code,
        message: sentence(&b.error.message),
        hint: b.error.hint.as_deref().map(sentence),
        status: Some(status),
    }
}

/// Percent-encode one path segment (ids are opaque strings).
pub(super) fn seg(s: &str) -> String {
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

/// Whether an address as typed in Settings (the scheme optional) names an HTTP server: http or
/// https, a host (a bracketed one a real IPv6 address), and a port from 1 to 65535 when one is
/// typed. ureq refuses any other scheme before sending a byte, and quietly sends to port 80 when
/// the typed port isn't a number it can use ("http://127.0.0.1:99999", ":abc", a bare ":"); the
/// URI parser counts a bracketed host's colons but doesn't read it ("[:::]", "[1::2::3]").
pub(crate) fn address_ok(typed: &str) -> bool {
    let typed = typed.trim();
    let url = if typed.contains("://") { typed.to_string() } else { format!("http://{typed}") };
    url.parse::<ureq::http::Uri>().is_ok_and(|u| {
        matches!(u.scheme_str(), Some("http" | "https"))
            && u.host().is_some_and(host_ok)
            && u.authority().is_some_and(|a| port_ok(a.as_str()))
    })
}

/// A host as the URI parser gives it: not empty, and `[v6]` an IPv6 address.
fn host_ok(host: &str) -> bool {
    match host.strip_prefix('[') {
        Some(v6) => v6.strip_suffix(']').is_some_and(|a| a.parse::<std::net::Ipv6Addr>().is_ok()),
        None => !host.is_empty(),
    }
}

/// The port of an authority (`user@host:port`, `[v6]:port`): none typed, or one from 1 to 65535.
fn port_ok(authority: &str) -> bool {
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    // An IPv6 host has its own colons, inside the brackets.
    let after_host = match host_port.strip_prefix('[') {
        Some(v6) => v6.split_once(']').map_or("", |(_, rest)| rest),
        None => host_port.find(':').map_or("", |i| &host_port[i..]),
    };
    match after_host.strip_prefix(':') {
        None => after_host.is_empty(),
        Some(port) => {
            !port.is_empty()
                && port.bytes().all(|b| b.is_ascii_digit())
                && port.parse::<u16>().is_ok_and(|p| p > 0)
        }
    }
}

impl HttpBackend {
    /// Every request's URL. For an address that isn't one, a scheme ureq turns down before it
    /// connects: "http://127.0.0.1:99999" as typed would go to port 80 (review round 2).
    pub(super) fn url(&self, path: &str) -> String {
        match &self.not_an_address {
            Some(_) => format!("{NOT_SENT}:///api{path}"),
            None => format!("{}/api{path}", self.base),
        }
    }

    /// Nothing answered: said for a person (ureq's own words, "timeout: connect" for a refused port on
    /// Windows, would read as a fault of the server).
    pub(super) fn unreachable(&self, _e: &ureq::Error) -> ApiError {
        // Garbage typed in Settings ("abc def", "http://", "foo://x:1") never left the app: say so,
        // not that a server is down.
        if let Some(typed) = &self.not_an_address {
            let what = if typed.is_empty() { "An empty address".to_string() } else { typed.clone() };
            return ApiError::new("server_unreachable", format!("{what} {NOT_AN_ADDRESS}"))
                .with_hint("Type it as http://host:port, like http://127.0.0.1:7878.");
        }
        ApiError::new("server_unreachable", format!("Could not reach folo serve at {}", self.host()))
            .with_hint("Start it with `folo serve`, or switch to the built-in demo in Settings.")
    }

    /// Whether the address names an HTTP server, so a request can be sent to it at all.
    #[cfg(test)]
    pub(super) fn address_ok(&self) -> bool {
        self.not_an_address.is_none()
    }

    pub(super) fn host(&self) -> String {
        self.base.trim_start_matches("http://").trim_start_matches("https://").to_string()
    }

    /// The address as scheme, host and port, for the log: a pasted URL's path, query or user can hold a token.
    pub(super) fn origin(&self) -> String {
        let uri = self.base.parse::<ureq::http::Uri>().ok();
        let (scheme, host) =
            uri.as_ref().map_or(("", ""), |u| (u.scheme_str().unwrap_or(""), u.host().unwrap_or("")));
        let port = uri.as_ref().and_then(|u| u.port_u16()).map(|p| format!(":{p}")).unwrap_or_default();
        format!("{scheme}://{host}{port}")
    }

    /// `/api/health` answered `status` without FOLO's shape (see `health`): a web page (another
    /// server, a sign-in page or a proxy's error page, or folo serve's own Studio page when a path
    /// follows the port, "http://127.0.0.1:7878/subpath"), another JSON, or nothing. Retrying can't
    /// help; the address is what to check, and with a path after the port, the address without it.
    pub(super) fn not_folo(&self, status: u16) -> ApiError {
        eprintln!("{}", self.not_folo_log(status));
        let authority = self.base.find("://").map_or(0, |i| i + 3);
        let hint = match self.base[authority..].find('/') {
            Some(path) => format!(
                "Check the address: folo serve answers at its base address. Try {}.",
                &self.base[..authority + path]
            ),
            None => "Check the address: it should be folo serve's base address, like http://127.0.0.1:7878."
                .into(),
        };
        ApiError::new("not_folo", format!("{} answered HTTP {status}, but not as a FOLO server", self.host()))
            .with_hint(hint)
    }

    /// [`Self::not_folo`]'s log line: the origin, the status and a fixed reason. Nothing of the
    /// answer: a proxy's error page or JSON can quote the request, a pasted `?token=` with it.
    pub(super) fn not_folo_log(&self, status: u16) -> String {
        format!("folo-desktop: {} answered HTTP {status}, not as a FOLO server", self.origin())
    }

    /// Turn a response into `T`, or the API's error body into an [`ApiError`].
    pub(super) fn decode<T: DeserializeOwned>(
        &self,
        mut resp: ureq::http::Response<ureq::Body>,
    ) -> ApiResult<T> {
        let status = resp.status().as_u16();
        if status >= 400 {
            let body: Option<folo_api::ErrorBody> = json(resp.body_mut(), BODY_LIMIT).ok();
            return Err(match body {
                Some(b) => server_error(b, status),
                None => status_error(status),
            });
        }
        if status == 204 {
            // Parsed from bytes, as every body is: one parser of `T`, not a second over a `&str`.
            return serde_json::from_slice(b"null").map_err(|e| ApiError::new("bad_response", e.to_string()));
        }
        json(resp.body_mut(), 256 << 20)
            .map_err(|e| self.body_failed(e, "Unexpected response from folo serve: "))
    }

    /// A response body that didn't read. The link breaking under it (the server went away, a proxy
    /// cut it, a read timed out: a body shorter than it announced) is the server being out of reach,
    /// as a request that got no answer is, so reconnecting asks for it again; a body that arrived
    /// whole but isn't what was asked for (not JSON, the wrong shape, too big) is a bad response.
    pub(super) fn body_failed(&self, e: ureq::Error, what: &str) -> ApiError {
        match e {
            ureq::Error::Io(_) | ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed => {
                self.unreachable(&e)
            }
            e => ApiError::new("bad_response", format!("{what}{e}")),
        }
    }

    pub(super) fn get<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self.agent.get(&self.url(path)).call().map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    pub(super) fn get_with_header<T: DeserializeOwned>(
        &self,
        path: &str,
        header: &str,
    ) -> ApiResult<(T, Option<String>)> {
        let resp = self.agent.get(&self.url(path)).call().map_err(|e| self.unreachable(&e))?;
        let hdr = resp
            .headers()
            .get(header)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let body = self.decode(resp)?;
        Ok((body, hdr))
    }

    pub(super) fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = self.agent.post(&self.url(path)).send_json(body).map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    /// A POST that may legitimately take minutes (indexing a large folder, writing images).
    pub(super) fn post_slow<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = agent(Some(Duration::from_secs(600)))
            .post(&self.url(path))
            .send_json(body)
            .map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    pub(super) fn post_empty<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self
            .agent
            .post(&self.url(path))
            .header("Content-Type", "application/json")
            .send("{}")
            .map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    pub(super) fn put<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = self.agent.put(&self.url(path)).send_json(body).map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    /// A DELETE: `204` decodes as `()`, a body as `T`.
    pub(super) fn delete<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self.agent.delete(&self.url(path)).call().map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_segments_are_encoded() {
        assert_eq!(seg("ds_8f2c"), "ds_8f2c");
        assert_eq!(seg("a b/c"), "a%20b%2Fc");
    }

    #[test]
    fn server_messages_read_as_sentences() {
        assert_eq!(sentence("the dataset has no training images"), "The dataset has no training images.");
        assert_eq!(sentence("  'D:/x' does not exist. "), "'D:/x' does not exist.");
        assert_eq!(sentence("`claude` printed no result"), "`claude` printed no result.");
        assert_eq!(sentence("étiquettes manquantes!"), "Étiquettes manquantes!");
        assert_eq!(sentence(""), "");
    }
}
