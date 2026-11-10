//! JSON over HTTPS for provider APIs: a bearer key per call, timeouts for interactive use, a bounded body, and
//! `429 Too Many Requests` waited out briefly. A request that gets no answer is [`ErrorKind::Unavailable`]; any
//! answer, whatever its status, is a [`Reply`] for the provider to map.
//!
//! Keys travel only in the `Authorization` header, never in a URL, so a transport error (which may quote the URL)
//! cannot carry one, and a key a provider echoes in a body is blanked out before anything reads the body.

use std::time::Duration;

use http::Method;
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderValue, RETRY_AFTER};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{Error, ErrorKind, Result};

/// Time to establish a connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Time for a whole request, body included: these calls answer someone waiting in a form or a run's start.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Largest answer read (a page of pods, the GPU catalog).
const MAX_BODY: u64 = 16 * 1024 * 1024;
/// Requests sent at most for one call when the provider answers `429`.
const MAX_ATTEMPTS: u32 = 3;
/// Longest single wait for a `429`, whatever `Retry-After` asks for.
const MAX_WAIT: Duration = Duration::from_secs(10);
/// Longest wait for `429`s over one call.
const MAX_TOTAL_WAIT: Duration = Duration::from_secs(20);
/// The wait for a `429` without a readable `Retry-After`.
const DEFAULT_WAIT: Duration = Duration::from_secs(2);

/// What a key echoed in an answer is replaced with.
const REDACTED: &[u8] = b"[redacted]";

/// One provider's API: a connection pool with its timeouts, and the provider's name for messages.
#[derive(Clone, Debug)]
pub(crate) struct Api {
    agent: ureq::Agent,
    service: &'static str,
}

/// An answer: its status and its body (with the key blanked out).
#[derive(Clone, Debug)]
pub(crate) struct Reply {
    pub status: u16,
    pub body: Vec<u8>,
    service: &'static str,
}

impl Reply {
    /// A `2xx` answer.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// The body, parsed; [`ErrorKind::Unavailable`] when it is not what the provider documents.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.body).map_err(|e| {
            Error::new(ErrorKind::Unavailable, format!("{} answered something unexpected: {e}", self.service))
        })
    }
}

impl Api {
    /// A client of the API `service` names (e.g. `RunPod`).
    pub fn new(service: &'static str) -> Api {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .into();
        Api { agent, service }
    }

    /// `GET url`.
    pub fn get(&self, url: &str, key: &str) -> Result<Reply> {
        self.call(Method::GET, url, key, None)
    }

    /// `POST url` with `body` as JSON.
    pub fn post<B: Serialize + ?Sized>(&self, url: &str, key: &str, body: &B) -> Result<Reply> {
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        self.call(Method::POST, url, key, Some(&json))
    }

    /// `DELETE url`.
    pub fn delete(&self, url: &str, key: &str) -> Result<Reply> {
        self.call(Method::DELETE, url, key, None)
    }

    fn call(&self, method: Method, url: &str, key: &str, body: Option<&[u8]>) -> Result<Reply> {
        let mut waited = Duration::ZERO;
        let mut attempt = 1;
        loop {
            let res = self.send(&method, url, key, body).map_err(|e| self.unavailable(e))?;
            let status = res.status().as_u16();
            // A `429` was not acted on, so sending the same request again is safe, also for a `POST`.
            if status == 429
                && attempt < MAX_ATTEMPTS
                && let Some(wait) = retry_wait(res.headers().get(RETRY_AFTER), waited)
            {
                std::thread::sleep(wait);
                waited += wait;
                attempt += 1;
                continue;
            }
            let body = res
                .into_body()
                .with_config()
                .limit(MAX_BODY)
                .read_to_vec()
                .map_err(|e| self.unavailable(e))?;
            return Ok(Reply { status, body: redact(body, key), service: self.service });
        }
    }

    fn send(
        &self,
        method: &Method,
        url: &str,
        key: &str,
        body: Option<&[u8]>,
    ) -> std::result::Result<http::Response<ureq::Body>, ureq::Error> {
        let request = http::Request::builder()
            .method(method)
            .uri(url)
            .header(AUTHORIZATION, format!("Bearer {key}"))
            .header(ACCEPT, "application/json");
        match body {
            Some(body) => self.agent.run(request.header(CONTENT_TYPE, "application/json").body(body)?),
            None => self.agent.run(request.body(())?),
        }
    }

    fn unavailable(&self, e: ureq::Error) -> Error {
        Error::new(
            ErrorKind::Unavailable,
            format!("{} does not answer ({e}); check this machine's internet connection", self.service),
        )
    }
}

/// Time for [`status_of`] to get its answer's headers: it runs before a launch someone waits for.
const STATUS_TIMEOUT: Duration = Duration::from_secs(15);

/// The status `GET url` is answered with, redirects followed; `Err` (what happened, for people) when there is no
/// answer. Sends no credentials: the URL is a download anyone may host, not a provider's API. The body is not
/// read.
pub(crate) fn status_of(url: &str) -> std::result::Result<u16, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_global(Some(STATUS_TIMEOUT))
        .build()
        .into();
    agent.get(url).call().map(|res| res.status().as_u16()).map_err(|e| e.to_string())
}

/// How long to wait before sending a request again that was answered `429` with `retry_after`, after `waited`
/// already; `None` when the call has waited long enough.
fn retry_wait(retry_after: Option<&HeaderValue>, waited: Duration) -> Option<Duration> {
    let asked = retry_after
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(DEFAULT_WAIT, Duration::from_secs);
    let wait = asked.min(MAX_WAIT);
    (waited + wait <= MAX_TOTAL_WAIT).then_some(wait)
}

/// `body` with every occurrence of `key` replaced by [`REDACTED`].
fn redact(body: Vec<u8>, key: &str) -> Vec<u8> {
    let key = key.as_bytes();
    if key.is_empty() || !body.windows(key.len()).any(|w| w == key) {
        return body;
    }
    let mut out = Vec::with_capacity(body.len());
    let mut rest = &body[..];
    while !rest.is_empty() {
        if rest.starts_with(key) {
            out.extend_from_slice(REDACTED);
            rest = &rest[key.len()..];
        } else {
            out.push(rest[0]);
            rest = &rest[1..];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Answer, Fixture};

    #[test]
    fn retry_waits_follow_retry_after_within_bounds() {
        let h = |s: &'static str| HeaderValue::from_static(s);
        assert_eq!(retry_wait(Some(&h("3")), Duration::ZERO), Some(Duration::from_secs(3)));
        assert_eq!(retry_wait(Some(&h("0")), Duration::ZERO), Some(Duration::ZERO));
        // An hour is cut to the longest single wait; a date or garbage is the default.
        assert_eq!(retry_wait(Some(&h("3600")), Duration::ZERO), Some(MAX_WAIT));
        assert_eq!(retry_wait(Some(&h("Wed, 21 Oct 2026 07:28:00 GMT")), Duration::ZERO), Some(DEFAULT_WAIT));
        assert_eq!(retry_wait(None, Duration::ZERO), Some(DEFAULT_WAIT));
        // The call gives up rather than wait past its total.
        assert_eq!(retry_wait(Some(&h("10")), Duration::from_secs(15)), None);
        assert_eq!(retry_wait(Some(&h("5")), Duration::from_secs(15)), Some(Duration::from_secs(5)));
    }

    #[test]
    fn redact_blanks_every_occurrence_of_the_key() {
        assert_eq!(redact(b"key k3y and k3y".to_vec(), "k3y"), b"key [redacted] and [redacted]");
        assert_eq!(redact(b"nothing".to_vec(), "k3y"), b"nothing");
        assert_eq!(redact(b"nothing".to_vec(), ""), b"nothing");
    }

    #[test]
    fn sends_the_key_as_a_bearer_header_and_retries_a_429() {
        let fixture = Fixture::serve(vec![
            Answer::json(429, r#"{"title":"Too Many Requests"}"#).header("Retry-After", "0"),
            Answer::json(200, r#"{"echo":"rpa_SECRET"}"#),
        ]);
        let api = Api::new("Test");
        let reply = api.post(&fixture.url("/things"), "rpa_SECRET", &serde_json::json!({ "a": 1 })).unwrap();
        assert_eq!(reply.status, 200);
        assert_eq!(reply.body, br#"{"echo":"[redacted]"}"#);
        let requests = fixture.requests();
        assert_eq!(requests.len(), 2);
        for r in &requests {
            assert_eq!((r.method.as_str(), r.path.as_str()), ("POST", "/things"));
            assert_eq!(r.header("authorization"), Some("Bearer rpa_SECRET"));
            assert_eq!(r.header("content-type"), Some("application/json"));
            assert_eq!(r.body, br#"{"a":1}"#);
        }
    }

    #[test]
    fn a_429_that_persists_is_answered_after_the_last_attempt() {
        let busy = || Answer::json(429, "{}").header("Retry-After", "0");
        let fixture = Fixture::serve(vec![busy(), busy(), busy(), Answer::json(200, "{}")]);
        let reply = Api::new("Test").get(&fixture.url("/x"), "k").unwrap();
        assert_eq!(reply.status, 429);
        assert_eq!(fixture.requests().len(), MAX_ATTEMPTS as usize);
    }

    #[test]
    fn no_answer_is_unavailable_and_names_the_service() {
        let fixture = Fixture::hang_up();
        let e = Api::new("Test").get(&fixture.url("/x"), "rpa_SECRET").unwrap_err();
        assert_eq!(e.kind, ErrorKind::Unavailable);
        assert!(e.message.starts_with("Test does not answer"), "{}", e.message);
        assert!(!format!("{e:?}").contains("rpa_SECRET"));
    }

    #[test]
    fn status_of_asks_without_credentials() {
        let fixture = Fixture::serve(vec![Answer::json(404, "{}"), Answer::json(200, "ok")]);
        assert_eq!(status_of(&fixture.url("/install.sh")), Ok(404));
        assert_eq!(status_of(&fixture.url("/install.sh")), Ok(200));
        for r in fixture.requests() {
            assert_eq!((r.method.as_str(), r.path.as_str()), ("GET", "/install.sh"));
            assert_eq!(r.header("authorization"), None);
        }
        assert!(status_of(&Fixture::hang_up().url("/x")).is_err());
    }

    #[test]
    fn an_unexpected_body_is_unavailable() {
        let fixture = Fixture::serve(vec![Answer::json(200, "<html>")]);
        let reply = Api::new("Test").get(&fixture.url("/x"), "k").unwrap();
        let e = reply.json::<serde_json::Value>().unwrap_err();
        assert_eq!(e.kind, ErrorKind::Unavailable);
        assert!(e.message.starts_with("Test answered something unexpected"), "{}", e.message);
    }
}
