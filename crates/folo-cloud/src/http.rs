//! JSON over HTTPS for provider APIs: a bearer key per call, timeouts for interactive use, a bounded body, and
//! `429 Too Many Requests` (plus idempotent `502`/`503`/`504`) waited out briefly. A request that gets no answer
//! is [`ErrorKind::Unavailable`]; any answer, whatever its status, is a [`Reply`] for the provider to map.
//!
//! Keys travel only in the `Authorization` header, never in a URL, so a transport error (which may quote the URL)
//! cannot carry one, and a key a provider echoes in an answer is blanked out of every message made from it,
//! however the body spells it ([`Reply::scrub`]; so is any other credential the request carried via
//! [`Api::post_with_secrets`]). The body itself is read as it came: the ids in it are used again.

use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
/// Largest error body read on a non-2xx answer (keeps memory bounded and preserves HTTP status).
const MAX_ERROR_BODY: u64 = 1024 * 1024;
/// Requests sent at most for one call when the provider answers a retryable status (`429`, or idempotent `502`/`503`/`504`).
const MAX_ATTEMPTS: u32 = 3;
/// Longest single wait before a retry, whatever `Retry-After` asks for.
const MAX_WAIT: Duration = Duration::from_secs(10);
/// Longest cumulative retry wait over one call.
const MAX_TOTAL_WAIT: Duration = Duration::from_secs(20);
/// The wait for a retryable status without a readable `Retry-After`.
const DEFAULT_WAIT: Duration = Duration::from_secs(2);

/// One provider's API: a connection pool with its timeouts, and the provider's name for messages.
#[derive(Clone, Debug)]
pub(crate) struct Api {
    agent: ureq::Agent,
    service: &'static str,
}

/// An answer: its status and its body, as it came.
#[derive(Clone)]
pub(crate) struct Reply {
    pub status: u16,
    pub body: Vec<u8>,
    service: &'static str,
    /// The credentials the request carried: the key, and any extra secrets passed to [`Api::post_with_secrets`].
    /// Text read out of the body (decoded, so whatever its spelling) goes through [`Reply::scrub`] before it
    /// becomes a message.
    secrets: Vec<String>,
}

// Not derived: the secrets must not reach a log through `{:?}`, and the body may spell them.
impl std::fmt::Debug for Reply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reply")
            .field("status", &self.status)
            .field("body_bytes", &self.body.len())
            .field("service", &self.service)
            .finish_non_exhaustive()
    }
}

impl Reply {
    /// A `2xx` answer.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// `text` for a person, read out of the body (decoded), with every credential the request carried blanked out
    /// ([`crate::scrub`]). Not for a value that is used again.
    pub fn scrub(&self, text: &str) -> String {
        crate::scrub(text, self.secrets.iter().map(String::as_str))
    }

    /// The body, parsed; [`ErrorKind::Unavailable`] when it is not what the provider documents.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.body).map_err(|e| {
            let what = self.scrub(&e.to_string());
            Error::new(
                ErrorKind::Unavailable,
                format!("{} answered something unexpected: {what}", self.service),
            )
        })
    }
}

impl Api {
    /// A client of the API `service` names (e.g. `RunPod`). Never follows HTTP redirects so bearer keys and
    /// request payloads (such as worker tokens) cannot be forwarded to an unexpected host.
    pub fn new(service: &'static str) -> Api {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .max_redirects_will_error(false)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .into();
        Api { agent, service }
    }

    /// `GET url`.
    pub fn get(&self, url: &str, key: &str) -> Result<Reply> {
        self.call(Method::GET, url, key, None, &[])
    }

    /// `POST url` with `body` as JSON.
    pub fn post<B: Serialize + ?Sized>(&self, url: &str, key: &str, body: &B) -> Result<Reply> {
        self.post_with_secrets(url, key, body, &[])
    }

    /// `POST url` with `body` as JSON, also scrubbing `extra_secrets` from transport errors and the reply.
    pub fn post_with_secrets<B: Serialize + ?Sized>(
        &self,
        url: &str,
        key: &str,
        body: &B,
        extra_secrets: &[&str],
    ) -> Result<Reply> {
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        self.call(Method::POST, url, key, Some(&json), extra_secrets)
    }

    /// `DELETE url`.
    pub fn delete(&self, url: &str, key: &str) -> Result<Reply> {
        self.call(Method::DELETE, url, key, None, &[])
    }

    fn call(
        &self,
        method: Method,
        url: &str,
        key: &str,
        body: Option<&[u8]>,
        extra_secrets: &[&str],
    ) -> Result<Reply> {
        let mut waited = Duration::ZERO;
        let mut attempt = 1;
        loop {
            let res =
                self.send(&method, url, key, body).map_err(|e| self.unavailable(e, key, extra_secrets))?;
            let status = res.status().as_u16();
            if should_retry(&method, status)
                && attempt < MAX_ATTEMPTS
                && let Some(wait) = retry_wait(res.headers().get(RETRY_AFTER), waited)
            {
                std::thread::sleep(wait);
                waited += wait;
                attempt += 1;
                continue;
            }
            let body = if (200..300).contains(&status) {
                res.into_body()
                    .with_config()
                    .limit(MAX_BODY)
                    .read_to_vec()
                    .map_err(|e| self.unavailable(e, key, extra_secrets))?
            } else {
                // Bound non-2xx error bodies so a huge proxy error page never exhausts memory or masks a
                // 401/403/400 status behind `Unavailable`.
                let mut bytes = Vec::new();
                let mut limited = std::io::Read::take(res.into_body().into_reader(), MAX_ERROR_BODY);
                let _ = std::io::Read::read_to_end(&mut limited, &mut bytes);
                bytes
            };
            let mut secrets = Vec::with_capacity(1 + extra_secrets.len());
            secrets.push(key.to_string());
            secrets.extend(extra_secrets.iter().map(|s| (*s).to_string()));
            return Ok(Reply { status, body, service: self.service, secrets });
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

    fn unavailable(&self, e: ureq::Error, key: &str, extra_secrets: &[&str]) -> Error {
        let why = crate::scrub(&e.to_string(), std::iter::once(key).chain(extra_secrets.iter().copied()));
        Error::new(
            ErrorKind::Unavailable,
            format!("{} does not answer ({why}); check this machine's internet connection", self.service),
        )
    }
}

/// Whether `status` on `method` is safe and worthwhile to retry.
///
/// `429` was not acted on so every method (including `POST`) may be retried; `502`/`503`/`504` are transient
/// gateway or overload failures, retried only for idempotent methods so a `POST` never launches a second instance.
fn should_retry(method: &Method, status: u16) -> bool {
    status == 429
        || (matches!(*method, Method::GET | Method::DELETE | Method::HEAD | Method::PUT)
            && matches!(status, 502..=504))
}

/// Time for [`status_of`] to get its answer's headers: it runs before a launch someone waits for.
const STATUS_TIMEOUT: Duration = Duration::from_secs(15);
/// Smallest body prefix [`status_of`] drains so short responses (`install.sh`, 404 pages) return their keep-alive
/// connection to the shared pool without downloading large release archives.
const STATUS_DRAIN_BYTES: u64 = 16 * 1024;

/// Shared connection pool for [`status_of`] probes (no credentials, short timeout).
fn status_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_global(Some(STATUS_TIMEOUT))
            .build()
            .into()
    })
}

/// The status `GET url` is answered with, redirects followed; `Err` (what happened, for people) when there is no
/// answer. Sends no credentials: the URL is a download anyone may host, not a provider's API. Drains only a tiny
/// body prefix so small responses recycle their pooled keep-alive connection while large archives are not fetched.
pub(crate) fn status_of(url: &str) -> std::result::Result<u16, String> {
    let res = status_agent().get(url).call().map_err(|e| e.to_string())?;
    let status = res.status().as_u16();
    // Only drain when the body is small enough to finish within `STATUS_DRAIN_BYTES` (or chunked/unknown length);
    // skipping large `Content-Length` bodies avoids pulling megabytes of a release archive over the wire.
    let known_len = res
        .headers()
        .get(http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    if known_len.is_none_or(|len| len <= STATUS_DRAIN_BYTES) {
        let mut reader = std::io::Read::take(res.into_body().into_reader(), STATUS_DRAIN_BYTES + 1);
        let _ = std::io::copy(&mut reader, &mut std::io::sink());
    }
    Ok(status)
}

/// How long to wait before sending a request again that was answered `429` (or idempotent `502`/`503`/`504`) with
/// `retry_after`, after `waited` already; `None` when the call has waited long enough.
fn retry_wait(retry_after: Option<&HeaderValue>, waited: Duration) -> Option<Duration> {
    retry_wait_at(retry_after, waited, SystemTime::now())
}

fn retry_wait_at(retry_after: Option<&HeaderValue>, waited: Duration, now: SystemTime) -> Option<Duration> {
    let asked = retry_after
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .and_then(|s| {
            parse_delay_seconds(s)
                .map(Duration::from_secs)
                .or_else(|| parse_http_date(s).map(|at| at.duration_since(now).unwrap_or(Duration::ZERO)))
        })
        .unwrap_or(DEFAULT_WAIT);
    let wait = asked.min(MAX_WAIT);
    (waited + wait <= MAX_TOTAL_WAIT).then_some(wait)
}

fn parse_delay_seconds(s: &str) -> Option<u64> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok()).flatten()
}

fn parse_digits<T: std::str::FromStr>(s: &str, min_len: usize, max_len: usize) -> Option<T> {
    if (min_len..=max_len).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit()) {
        s.parse().ok()
    } else {
        None
    }
}

/// Parses an RFC 9110 §5.6.7 HTTP-date (`IMF-fixdate`, obsolete `RFC 850`, or `asctime`) as a UTC [`SystemTime`].
fn parse_http_date(s: &str) -> Option<SystemTime> {
    let (day, month, year, hms) = if let Some((wday, rest)) = s.split_once(", ") {
        let mut parts = rest.split_ascii_whitespace();
        let first = parts.next()?;
        if let Some((d, m, y)) = split3(first, '-') {
            // RFC 850: `Sunday, 06-Nov-94 08:49:37 GMT`
            if !is_long_weekday(wday) {
                return None;
            }
            let hms = parts.next()?;
            if parts.next()? != "GMT" || parts.next().is_some() {
                return None;
            }
            let yy: i32 = parse_digits(y, 2, 2)?;
            let year = if yy >= 70 { 1900 + yy } else { 2000 + yy };
            (parse_digits(d, 2, 2)?, parse_month(m)?, year, hms)
        } else {
            // IMF-fixdate: `Wed, 21 Oct 2026 07:28:00 GMT`
            if !is_short_weekday(wday) {
                return None;
            }
            let m = parts.next()?;
            let y = parts.next()?;
            let hms = parts.next()?;
            if parts.next()? != "GMT" || parts.next().is_some() {
                return None;
            }
            (parse_digits(first, 2, 2)?, parse_month(m)?, parse_digits(y, 4, 4)?, hms)
        }
    } else {
        // asctime: `Sun Nov  6 08:49:37 1994`
        let mut parts = s.split_ascii_whitespace();
        let wday = parts.next()?;
        if !is_short_weekday(wday) {
            return None;
        }
        let m = parts.next()?;
        let d = parts.next()?;
        let hms = parts.next()?;
        let y = parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        (parse_digits(d, 1, 2)?, parse_month(m)?, parse_digits(y, 4, 4)?, hms)
    };
    let (h, min, sec) = split3(hms, ':')?;
    let (hour, minute, second): (u32, u32, u32) =
        (parse_digits(h, 2, 2)?, parse_digits(min, 2, 2)?, parse_digits(sec, 2, 2)?);
    if hour > 23 || minute > 59 || second > 59 || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let secs = days
        .checked_mul(86_400)?
        .checked_add(i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second))?;
    if secs >= 0 {
        UNIX_EPOCH.checked_add(Duration::from_secs(secs as u64))
    } else {
        Some(UNIX_EPOCH.checked_sub(Duration::from_secs(secs.unsigned_abs())).unwrap_or(UNIX_EPOCH))
    }
}

fn split3(s: &str, sep: char) -> Option<(&str, &str, &str)> {
    let mut parts = s.split(sep);
    let a = parts.next()?;
    let b = parts.next()?;
    let c = parts.next()?;
    parts.next().is_none().then_some((a, b, c))
}

fn is_short_weekday(s: &str) -> bool {
    matches!(s, "Mon" | "Tue" | "Wed" | "Thu" | "Fri" | "Sat" | "Sun")
}

fn is_long_weekday(s: &str) -> bool {
    matches!(s, "Monday" | "Tuesday" | "Wednesday" | "Thursday" | "Friday" | "Saturday" | "Sunday")
}

fn parse_month(s: &str) -> Option<u32> {
    match s {
        "Jan" => Some(1),
        "Feb" => Some(2),
        "Mar" => Some(3),
        "Apr" => Some(4),
        "May" => Some(5),
        "Jun" => Some(6),
        "Jul" => Some(7),
        "Aug" => Some(8),
        "Sep" => Some(9),
        "Oct" => Some(10),
        "Nov" => Some(11),
        "Dec" => Some(12),
        _ => None,
    }
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Days since 1970-01-01 (Howard Hinnant's civil-from-days inverse).
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let y = i64::from(year) - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u64;
    let m = u64::from(month);
    let d = u64::from(day);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i64 - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Answer, Fixture};

    #[test]
    fn retry_waits_follow_retry_after_within_bounds() {
        let h = |s: &'static str| HeaderValue::from_static(s);
        // 2026-10-21T07:28:00Z = 1_792_567_680 s since epoch.
        let now = UNIX_EPOCH + Duration::from_secs(1_792_567_680);
        assert_eq!(retry_wait_at(Some(&h("3")), Duration::ZERO, now), Some(Duration::from_secs(3)));
        assert_eq!(retry_wait_at(Some(&h("0")), Duration::ZERO, now), Some(Duration::ZERO));
        // An hour is cut to the longest single wait; invalid values fall back to the default.
        assert_eq!(retry_wait_at(Some(&h("3600")), Duration::ZERO, now), Some(MAX_WAIT));
        assert_eq!(retry_wait_at(Some(&h("+5")), Duration::ZERO, now), Some(DEFAULT_WAIT));
        assert_eq!(retry_wait_at(Some(&h("not-a-date")), Duration::ZERO, now), Some(DEFAULT_WAIT));
        assert_eq!(retry_wait_at(None, Duration::ZERO, now), Some(DEFAULT_WAIT));
        // HTTP-dates: future (within cap), far future (capped), past (zero), RFC 850, asctime, and invalid day.
        assert_eq!(
            retry_wait_at(Some(&h("Wed, 21 Oct 2026 07:28:04 GMT")), Duration::ZERO, now),
            Some(Duration::from_secs(4))
        );
        assert_eq!(
            retry_wait_at(Some(&h("Wed, 21 Oct 2026 08:28:00 GMT")), Duration::ZERO, now),
            Some(MAX_WAIT)
        );
        assert_eq!(
            retry_wait_at(Some(&h("Wed, 21 Oct 2026 07:27:55 GMT")), Duration::ZERO, now),
            Some(Duration::ZERO)
        );
        assert_eq!(
            retry_wait_at(Some(&h("Wednesday, 21-Oct-26 07:28:06 GMT")), Duration::ZERO, now),
            Some(Duration::from_secs(6))
        );
        assert_eq!(
            retry_wait_at(Some(&h("Wed Oct 21 07:28:07 2026")), Duration::ZERO, now),
            Some(Duration::from_secs(7))
        );
        assert!(parse_http_date("Thu, 29 Feb 2024 00:00:00 GMT").is_some());
        assert_eq!(
            retry_wait_at(Some(&h("Sat, 29 Feb 2025 00:00:00 GMT")), Duration::ZERO, now),
            Some(DEFAULT_WAIT)
        );
        // The call gives up rather than wait past its total.
        assert_eq!(retry_wait_at(Some(&h("10")), Duration::from_secs(15), now), None);
        assert_eq!(retry_wait_at(Some(&h("5")), Duration::from_secs(15), now), Some(Duration::from_secs(5)));
    }

    #[test]
    fn scrub_blanks_every_secret_long_enough_to_be_one() {
        let secrets = ["k3y-0123456", "abc", ""];
        assert_eq!(crate::scrub("k3y-0123456 and k3y-0123456", secrets), "[redacted] and [redacted]");
        // Shorter than a real key or token: blanking it would shred ordinary words.
        assert_eq!(crate::scrub("abc and abcdef", secrets), "abc and abcdef");
        assert_eq!(crate::scrub("12345678", ["12345678"]), "[redacted]", "the minimum itself counts");
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
        // The body is read as it came; what a person reads from it is scrubbed.
        assert_eq!(reply.body, br#"{"echo":"rpa_SECRET"}"#);
        assert_eq!(reply.scrub(r#"echo: rpa_SECRET"#), "echo: [redacted]");
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
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        assert!(std::ptr::eq(status_agent(), status_agent()), "status_of shares one pooled ureq::Agent");
        let fixture = Fixture::serve(vec![Answer::json(404, "{}"), Answer::json(200, "ok")]);
        assert_eq!(status_of(&fixture.url("/install.sh")), Ok(404));
        assert_eq!(status_of(&fixture.url("/install.sh")), Ok(200));
        for r in fixture.requests() {
            assert_eq!((r.method.as_str(), r.path.as_str()), ("GET", "/install.sh"));
            assert_eq!(r.header("authorization"), None);
        }
        assert!(status_of(&Fixture::hang_up().url("/x")).is_err());

        // Two probes to a keep-alive server with small non-empty bodies reuse a single TCP connection.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let accepts = Arc::new(AtomicUsize::new(0));
        let accepts_srv = accepts.clone();
        let srv = std::thread::spawn(move || {
            let mut served = 0usize;
            while served < 2 {
                let (mut stream, _) = listener.accept().unwrap();
                accepts_srv.fetch_add(1, Ordering::SeqCst);
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                while served < 2 {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    loop {
                        let mut hdr = String::new();
                        if reader.read_line(&mut hdr).unwrap_or(0) == 0 || hdr == "\r\n" || hdr == "\n" {
                            break;
                        }
                    }
                    served += 1;
                    let conn = if served == 2 { "close" } else { "keep-alive" };
                    let resp =
                        format!("HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: {conn}\r\n\r\nhello");
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                }
            }
        });
        assert_eq!(status_of(&format!("http://{addr}/a")), Ok(200));
        assert_eq!(status_of(&format!("http://{addr}/b")), Ok(200));
        srv.join().unwrap();
        assert_eq!(accepts.load(Ordering::SeqCst), 1, "keep-alive connection should be reused across probes");
    }

    #[test]
    fn idempotent_requests_retry_transient_5xx_and_post_does_not() {
        let api = Api::new("Test");

        // GET retries 502 and 503 (honoring Retry-After) and succeeds on the third attempt.
        let fixture = Fixture::serve(vec![
            Answer::empty(502).header("Retry-After", "0"),
            Answer::empty(503).header("Retry-After", "0"),
            Answer::json(200, r#"{"ok":true}"#),
        ]);
        let reply = api.get(&fixture.url("/items"), "k").unwrap();
        assert_eq!(reply.status, 200);
        assert_eq!(fixture.requests().len(), 3);

        // DELETE retries 504 and succeeds on retry.
        let fixture = Fixture::serve(vec![Answer::empty(504).header("Retry-After", "0"), Answer::empty(204)]);
        let reply = api.delete(&fixture.url("/items/1"), "k").unwrap();
        assert_eq!(reply.status, 204);
        assert_eq!(fixture.requests().len(), 2);

        // POST does NOT retry 502/503/504 (could duplicate a side effect like launching an instance).
        for status in [502, 503, 504] {
            let fixture = Fixture::serve(vec![
                Answer::empty(status).header("Retry-After", "0"),
                Answer::json(200, "{}"),
            ]);
            let reply = api.post(&fixture.url("/launch"), "k", &serde_json::json!({})).unwrap();
            assert_eq!(reply.status, status);
            assert_eq!(fixture.requests().len(), 1, "POST must not retry {status}");
        }

        // GET does NOT retry 500, and stops retrying persistent 503 after MAX_ATTEMPTS.
        let fixture = Fixture::serve(vec![Answer::empty(500), Answer::json(200, "{}")]);
        assert_eq!(api.get(&fixture.url("/x"), "k").unwrap().status, 500);
        assert_eq!(fixture.requests().len(), 1);

        let down = || Answer::empty(503).header("Retry-After", "0");
        let fixture = Fixture::serve(vec![down(), down(), down(), Answer::json(200, "{}")]);
        assert_eq!(api.get(&fixture.url("/x"), "k").unwrap().status, 503);
        assert_eq!(fixture.requests().len(), MAX_ATTEMPTS as usize);
    }

    #[test]
    fn an_unexpected_body_is_unavailable() {
        let fixture = Fixture::serve(vec![Answer::json(200, "<html>")]);
        let reply = Api::new("Test").get(&fixture.url("/x"), "k").unwrap();
        let e = reply.json::<serde_json::Value>().unwrap_err();
        assert_eq!(e.kind, ErrorKind::Unavailable);
        assert!(e.message.starts_with("Test answered something unexpected"), "{}", e.message);
    }

    #[test]
    fn api_does_not_follow_redirects_and_caps_error_bodies() {
        let target = Fixture::serve(vec![Answer::json(200, r#"{"leaked":true}"#)]);
        let redirect = Fixture::serve(vec![Answer::empty(302).header("Location", &target.url("/steal"))]);
        let reply = Api::new("Test").get(&redirect.url("/pods"), "rpa_SECRET_KEY").unwrap();
        assert_eq!(reply.status, 302);
        assert!(target.requests().is_empty(), "redirect target must not be contacted");

        let huge_401 = "x".repeat(MAX_ERROR_BODY as usize + 64 * 1024);
        let f = Fixture::serve(vec![Answer::json(401, huge_401)]);
        let reply = Api::new("Test").get(&f.url("/pods"), "rpa_SECRET_KEY").unwrap();
        assert_eq!(reply.status, 401);
        assert_eq!(reply.body.len(), MAX_ERROR_BODY as usize);
    }

    #[test]
    fn post_with_secrets_redacts_extra_secrets_in_reply_and_transport_errors() {
        let key = "rpa_KEY_SECRET_1234";
        let worker_tok = "wrk_TOKEN+SECRET/5678";
        let f =
            Fixture::serve(vec![Answer::json(400, format!(r#"{{"error":"bad {worker_tok} and {key}"}}"#))]);
        let reply = Api::new("Test")
            .post_with_secrets(&f.url("/pods"), key, &serde_json::json!({}), &[worker_tok])
            .unwrap();
        let text = reply.scrub(&String::from_utf8_lossy(&reply.body));
        assert!(!text.contains(worker_tok) && !text.contains(key), "{text}");

        // Even a transport error before any reply is created scrubs `extra_secrets` (including percent-encoded).
        let err = Api::new("Test")
            .post_with_secrets(
                &Fixture::hang_up().url(&format!("/pods?t={}", folo_client::encode(worker_tok))),
                key,
                &serde_json::json!({}),
                &[worker_tok],
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unavailable);
        assert!(!err.message.contains(worker_tok) && !err.message.contains(&folo_client::encode(worker_tok)));
    }
}
