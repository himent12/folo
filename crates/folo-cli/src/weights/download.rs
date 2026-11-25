//! How `folo weights pull` gets a release's files: over plain HTTPS without credentials first (all a public release
//! needs, so the `folo` binary is enough), else with the GitHub CLI (`gh release download`), which reaches the
//! releases of a private repository the user can see, drafts included for its writers.

use std::fs::File;
use std::io::{IsTerminal, Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::output::{CliError, ExitCode, Out};

/// Downloads from `<base>/<tag>/<asset>` instead of the GitHub release (a mirror, or a test's server).
pub const BASE_URL_ENV: &str = "FOLO_WEIGHTS_BASE_URL";
/// The GitHub CLI program the fallback runs (default `gh`).
pub const GH_ENV: &str = "FOLO_GH";
/// Base backoff delay in milliseconds between transient download retries (default 500 ms; `0` disables sleeping).
pub const RETRY_DELAY_ENV: &str = "FOLO_WEIGHTS_RETRY_DELAY_MS";

/// The longest wait for the host name to resolve, and then for the connection (TLS included) to be set up.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// The longest wait for the request's head to be sent, for an answer's head, and for each read of its body: a
/// stalled download fails, a slow one does not (no whole-request deadline).
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Files smaller than this get no progress line (`SHA256SUMS`).
const PROGRESS_MIN_BYTES: u64 = 1 << 20;
/// Maximum number of retries after an initial transient download failure (4 total attempts).
const MAX_RETRIES: u32 = 3;
/// Default initial backoff delay before the first retry.
const DEFAULT_RETRY_BASE: Duration = Duration::from_millis(500);
/// Upper bound on any single retry wait (including server `Retry-After` headers).
const RETRY_MAX_DELAY: Duration = Duration::from_secs(10);
/// Wait at least this long before showing throughput and ETA so the first read does not print a spike.
const RATE_MIN_ELAPSED: Duration = Duration::from_millis(100);

/// `$FOLO_WEIGHTS_BASE_URL`, else the release downloads of `repo` on GitHub, without a trailing slash.
pub fn base_url(repo: &str) -> String {
    match std::env::var(BASE_URL_ENV) {
        Ok(u) if !u.trim().is_empty() => u.trim().trim_end_matches('/').to_string(),
        _ => format!("https://github.com/{repo}/releases/download"),
    }
}

/// How a release's files reached the download folder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Via {
    /// Plain HTTPS, no credentials.
    Https,
    /// `gh release download`.
    Gh,
}

impl Via {
    /// The `via` of `folo weights pull --json`.
    pub fn name(self) -> &'static str {
        match self {
            Via::Https => "https",
            Via::Gh => "gh",
        }
    }
}

/// A file the plain download was refused: `401`, `403` or `404`. GitHub answers `404` for a release of a private
/// repository, for a draft, and for a release or file that does not exist.
struct Refused {
    url: String,
    status: u16,
}

/// Downloads `files` of release `tag` into `dir` (which must not hold them yet): over HTTPS from `base`
/// ([`base_url`]), or, when that is refused, with `gh` from `repo`. Any other failure of the HTTPS download (no
/// network, a timeout, a server error, a cut body) is an error of its own, not a reason to try `gh`.
pub fn release_files(
    base: &str,
    repo: &str,
    tag: &str,
    files: &[&str],
    dir: &Path,
    out: Out,
) -> Result<Via, CliError> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .user_agent(format!("folo/{}", env!("CARGO_PKG_VERSION")))
        .timeout_resolve(Some(CONNECT_TIMEOUT))
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_send_request(Some(READ_TIMEOUT))
        .timeout_per_write(Some(READ_TIMEOUT))
        .timeout_recv_response(Some(READ_TIMEOUT))
        .timeout_per_read(Some(READ_TIMEOUT))
        .build()
        .into();
    let mut refused = None;
    for f in files {
        refused = fetch(&agent, &format!("{base}/{tag}/{f}"), f, &dir.join(f), out)?;
        if refused.is_some() {
            break;
        }
    }
    let Some(refused) = refused else { return Ok(Via::Https) };
    out.human(format!(
        "{} answered {}: the release is not public (or does not exist); trying the GitHub CLI",
        refused.url, refused.status
    ));
    // `gh` does not overwrite: start from an empty folder.
    for f in files {
        let _ = std::fs::remove_file(dir.join(f));
    }
    gh(repo, tag, files, dir, &refused)?;
    Ok(Via::Gh)
}

/// Outcome of one download attempt.
enum Attempt {
    /// Download completed and matched `Content-Length` (if announced).
    Done,
    /// `401`, `403`, or `404` — do not retry; caller falls back to `gh`.
    Refused(Refused),
    /// Transient transport error, retryable HTTP status (`408`/`429`/`5xx`), or broken/truncated body.
    Retryable { err: CliError, summary: String, retry_after: Option<Duration> },
    /// Permanent failure (non-retryable `4xx`, malformed URI, or local filesystem error).
    Fatal(CliError),
}

fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500..=599)
}

fn is_retryable_io(kind: std::io::ErrorKind) -> bool {
    use std::io::ErrorKind::*;
    matches!(
        kind,
        ConnectionReset
            | ConnectionAborted
            | ConnectionRefused
            | NotConnected
            | BrokenPipe
            | TimedOut
            | Interrupted
            | UnexpectedEof
            | WouldBlock
            | HostUnreachable
            | NetworkUnreachable
            | NetworkDown
    )
}

fn is_retryable_transport(err: &ureq::Error) -> bool {
    match err {
        ureq::Error::Io(io) => is_retryable_io(io.kind()),
        ureq::Error::Timeout(_)
        | ureq::Error::HostNotFound
        | ureq::Error::ConnectionFailed
        | ureq::Error::BodyStalled => true,
        _ => false,
    }
}

fn parse_retry_after(raw: Option<&str>, now: SystemTime) -> Option<Duration> {
    let raw = raw?.trim();
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    let at = parse_http_date(raw, now)?;
    Some(at.duration_since(now).unwrap_or(Duration::ZERO))
}

fn parse_fixed_digits(s: &str, len: usize) -> Option<i64> {
    if s.len() == len && s.bytes().all(|b| b.is_ascii_digit()) { s.parse().ok() } else { None }
}

/// Parses RFC 9110 §5.6.7 HTTP-date formats (IMF-fixdate, RFC 850, and asctime) as UTC `SystemTime`.
fn parse_http_date(raw: &str, now: SystemTime) -> Option<SystemTime> {
    let (day, month, year, hms) = if let Some((_, rest)) = raw.split_once(", ") {
        let parts: Vec<&str> = rest.split_whitespace().collect();
        match parts.as_slice() {
            // IMF-fixdate: "06 Nov 1994 08:49:37 GMT"
            [d, mon, y, hms, "GMT"] => {
                (d.parse::<u32>().ok()?, parse_month(mon)?, parse_fixed_digits(y, 4)?, *hms)
            }
            // RFC 850: "06-Nov-94 08:49:37 GMT" (two-digit year resolved via RFC 9110 §5.6.7 50-year rule)
            [dmy, hms, "GMT"] => {
                let mut segs = dmy.split('-');
                let d = segs.next()?.parse::<u32>().ok()?;
                let mon = parse_month(segs.next()?)?;
                let y2 = parse_fixed_digits(segs.next()?, 2)?;
                if segs.next().is_some() {
                    return None;
                }
                let cur_year = civil_year_from_time(now);
                let mut y = (cur_year / 100) * 100 + y2;
                if y > cur_year + 50 {
                    y -= 100;
                } else if y <= cur_year - 50 {
                    y += 100;
                }
                (d, mon, y, *hms)
            }
            _ => return None,
        }
    } else {
        // asctime: "Sun Nov  6 08:49:37 1994"
        let parts: Vec<&str> = raw.split_whitespace().collect();
        match parts.as_slice() {
            [_, mon, d, hms, y] => {
                (d.parse::<u32>().ok()?, parse_month(mon)?, parse_fixed_digits(y, 4)?, *hms)
            }
            _ => return None,
        }
    };
    let mut t = hms.split(':');
    let hour: u32 = t.next()?.parse().ok()?;
    let min: u32 = t.next()?.parse().ok()?;
    let sec: u32 = t.next()?.parse().ok()?;
    if t.next().is_some() || hour > 23 || min > 59 || sec > 60 {
        return None;
    }
    let max_day = days_in_month(year, month)?;
    if day == 0 || day > max_day {
        return None;
    }
    let days = days_from_civil(year, month, day);
    if days < 0 {
        return Some(UNIX_EPOCH);
    }
    let total_secs = (days as u64)
        .checked_mul(86_400)?
        .checked_add(u64::from(hour) * 3600 + u64::from(min) * 60 + u64::from(sec.min(59)))?;
    UNIX_EPOCH.checked_add(Duration::from_secs(total_secs))
}

/// UTC calendar year of `time` (Howard Hinnant's civil-from-days algorithm).
fn civil_year_from_time(time: SystemTime) -> i64 {
    let days = time.duration_since(UNIX_EPOCH).map_or(0, |d| (d.as_secs() / 86_400) as i64);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    y + i64::from(m <= 2)
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

fn days_in_month(year: i64, month: u32) -> Option<u32> {
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if leap { 29 } else { 28 }),
        _ => None,
    }
}

/// Days since 1970-01-01 (Howard Hinnant's civil-to-days algorithm).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u64;
    let m = u64::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + u64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i64 - 719_468
}

fn retry_delay_base() -> Duration {
    match std::env::var(RETRY_DELAY_ENV) {
        Ok(raw) => raw
            .trim()
            .parse::<u64>()
            .map_or(DEFAULT_RETRY_BASE, |ms| Duration::from_millis(ms).min(RETRY_MAX_DELAY)),
        Err(_) => DEFAULT_RETRY_BASE,
    }
}

fn backoff_delay(attempt: u32, base: Duration, retry_after: Option<Duration>) -> Duration {
    if base.is_zero() {
        return Duration::ZERO;
    }
    let exp = base.saturating_mul(1u32 << attempt.min(10)).min(RETRY_MAX_DELAY);
    retry_after.map_or(exp, |ra| exp.max(ra).min(RETRY_MAX_DELAY))
}

fn format_delay(d: Duration) -> String {
    if d < Duration::from_secs(1) {
        format!("{}ms", d.as_millis())
    } else if d.subsec_millis() == 0 {
        format!("{}s", d.as_secs())
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

/// `GET url` into `dest`, streamed, redirects followed, retrying transient failures with bounded exponential
/// backoff; `Some` when it is refused ([`Refused`]).
fn fetch(
    agent: &ureq::Agent,
    url: &str,
    name: &str,
    dest: &Path,
    out: Out,
) -> Result<Option<Refused>, CliError> {
    let base = retry_delay_base();
    for attempt in 0..=MAX_RETRIES {
        match fetch_once(agent, url, name, dest, out) {
            Attempt::Done => return Ok(None),
            Attempt::Refused(r) => return Ok(Some(r)),
            Attempt::Fatal(e) => return Err(e),
            Attempt::Retryable { err, summary, retry_after } => {
                if attempt == MAX_RETRIES {
                    return Err(err);
                }
                let delay = backoff_delay(attempt, base, retry_after);
                out.human(format!(
                    "{name}: {summary}; retrying ({}/{MAX_RETRIES}) in {} ...",
                    attempt + 1,
                    format_delay(delay)
                ));
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
            }
        }
    }
    unreachable!()
}

/// One `GET url` attempt into `dest`. Removes any partial `dest` before returning an error.
fn fetch_once(agent: &ureq::Agent, url: &str, name: &str, dest: &Path, out: Out) -> Attempt {
    let res = match agent.get(url).call() {
        Ok(res) => res,
        Err(e) => {
            let summary = e.to_string();
            let retryable = is_retryable_transport(&e);
            let err = CliError::new(
                ExitCode::Runtime,
                "download_failed",
                format!("could not download {url}: {summary}"),
            )
            .hint(format!(
                "check this machine's network connection (or proxy) and try again; {BASE_URL_ENV} downloads \
                 from a mirror instead"
            ));
            return if retryable {
                Attempt::Retryable { err, summary, retry_after: None }
            } else {
                Attempt::Fatal(err)
            };
        }
    };
    let status = res.status().as_u16();
    if matches!(status, 401 | 403 | 404) {
        return Attempt::Refused(Refused { url: url.to_string(), status });
    }
    if !res.status().is_success() {
        let retry_after = parse_retry_after(
            res.headers().get("retry-after").and_then(|v| v.to_str().ok()),
            SystemTime::now(),
        );
        let err = CliError::new(ExitCode::Runtime, "download_failed", format!("{url} answered {status}"))
            .hint(format!(
                "the server's trouble: try again later; {BASE_URL_ENV} downloads from a mirror instead"
            ));
        return if is_retryable_status(status) {
            Attempt::Retryable { err, summary: format!("HTTP {status}"), retry_after }
        } else {
            Attempt::Fatal(err)
        };
    }
    let total = res.body().content_length();
    let mut body = res.into_body().into_reader();
    let mut file = match File::create(dest) {
        Ok(file) => file,
        Err(e) => {
            return Attempt::Fatal(CliError::new(
                ExitCode::Runtime,
                "io_error",
                format!("cannot create {}: {e}", folo_api::json::path(dest)),
            ));
        }
    };
    let broke_off = |done: u64, why: String| {
        let of = total.map_or(String::new(), |t| format!(" of {}", mb(t)));
        CliError::new(
            ExitCode::Runtime,
            "download_failed",
            format!("the download of {url} broke off after {}{of}: {why}", mb(done)),
        )
        .hint("the connection was cut or stalled: try again (nothing was installed)")
    };
    let mut progress = Progress::new(out, name, total);
    let mut buf = vec![0u8; 256 << 10];
    let mut done = 0u64;
    loop {
        let n = match body.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                progress.finish();
                drop(file);
                let _ = std::fs::remove_file(dest);
                let why = e.to_string();
                return Attempt::Retryable {
                    err: broke_off(done, why.clone()),
                    summary: format!("connection broke off: {why}"),
                    retry_after: None,
                };
            }
        };
        if let Err(e) = file.write_all(&buf[..n]) {
            progress.finish();
            drop(file);
            let _ = std::fs::remove_file(dest);
            return Attempt::Fatal(CliError::new(
                ExitCode::Runtime,
                "io_error",
                format!("cannot write {}: {e}", folo_api::json::path(dest)),
            ));
        }
        done += n as u64;
        progress.update(done);
    }
    progress.finish();
    if let Some(t) = total.filter(|&t| t != done) {
        drop(file);
        let _ = std::fs::remove_file(dest);
        return Attempt::Retryable {
            err: broke_off(done, format!("the server announced {t} bytes")),
            summary: format!("incomplete download ({done}/{t} bytes)"),
            retry_after: None,
        };
    }
    Attempt::Done
}

/// `gh release download` of `files` into `dir`, after the plain download was `refused`: its errors say both.
fn gh(repo: &str, tag: &str, files: &[&str], dir: &Path, refused: &Refused) -> Result<(), CliError> {
    let program = std::env::var_os(GH_ENV).filter(|p| !p.is_empty()).unwrap_or_else(|| "gh".into());
    let not_public =
        format!("{} answered {} (the release is not public, or does not exist)", refused.url, refused.status);
    let mut cmd = Command::new(&program);
    cmd.args(["release", "download", tag, "--repo", repo]);
    for f in files {
        cmd.args(["--pattern", f]);
    }
    // stdout carries `--json` records: what `gh` prints goes to stderr.
    let status = cmd.arg("--dir").arg(dir).stdout(std::io::stderr()).status().map_err(|e| {
        CliError::new(
            ExitCode::Runtime,
            "gh_failed",
            format!("{not_public}, and the GitHub CLI `{}` could not run: {e}", program.to_string_lossy()),
        )
        .hint(format!(
            "if {repo} is a private repository you can see, install the GitHub CLI (https://cli.github.com), run \
             `gh auth login` and pull again; a public release needs neither. `folo weights list` shows the names"
        ))
    })?;
    if !status.success() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "gh_failed",
            format!("{not_public}, and `gh release download {tag} --repo {repo}` failed ({status})"),
        )
        .hint(format!(
            "check `gh auth status`, and that `gh release view {tag} --repo {repo}` shows the release (a draft only \
             to the repository's writers)"
        )));
    }
    Ok(())
}

fn mb(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

fn format_rate(bytes_per_sec: f64) -> String {
    if bytes_per_sec >= 1e6 {
        format!("{:.1} MB/s", bytes_per_sec / 1e6)
    } else if bytes_per_sec >= 1e3 {
        format!("{:.0} KB/s", bytes_per_sec / 1e3)
    } else {
        format!("{:.0} B/s", bytes_per_sec)
    }
}

fn format_eta(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

fn format_progress(name: &str, done: u64, total: Option<u64>, elapsed: Duration) -> String {
    let of = match total {
        Some(t) => {
            let pct = (100.0 * done as f64 / t.max(1) as f64).min(100.0);
            format!(" / {} ({pct:.0}%)", mb(t))
        }
        None => String::new(),
    };
    let mut line = format!("{name}: {}{of}", mb(done));
    if done > 0 && elapsed >= RATE_MIN_ELAPSED {
        let rate = done as f64 / elapsed.as_secs_f64();
        if rate.is_finite() && rate > 0.0 {
            line.push_str(&format!(", {}", format_rate(rate)));
            if let Some(remaining) = total.and_then(|t| t.checked_sub(done)).filter(|&r| r > 0) {
                let eta_secs = (remaining as f64 / rate).ceil().min(u64::MAX as f64) as u64;
                line.push_str(&format!(", eta {}", format_eta(eta_secs)));
            }
        }
    }
    line
}

/// Maximum single-row width for progress updates (leaves one column free so the cursor never auto-wraps).
fn progress_max_width() -> usize {
    let cols = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&w| w > 0)
        .unwrap_or(80);
    cols.saturating_sub(1).max(1)
}

fn fit_progress_line(line: &str, max_width: usize) -> String {
    let max_width = max_width.max(1);
    if line.chars().count() <= max_width {
        return line.to_string();
    }
    if max_width <= 3 {
        return line.chars().take(max_width).collect();
    }
    let mut truncated: String = line.chars().take(max_width - 3).collect();
    truncated.push_str("...");
    truncated
}

fn render_progress_frame(line: &str, prev_width: usize, max_width: usize) -> (String, usize) {
    let max_width = max_width.max(1);
    let fitted = fit_progress_line(line, max_width);
    let width = fitted.chars().count();
    let pad = prev_width.min(max_width).saturating_sub(width);
    (format!("\r{fitted}{:pad$}", ""), width)
}

fn render_progress_clear(prev_width: usize, max_width: usize) -> String {
    let width = prev_width.min(max_width.max(1));
    format!("\r{:width$}\r", "")
}

/// A download's progress line for people: on a terminal, not with `--json`, for files of a MB or more.
struct Progress {
    name: String,
    total: Option<u64>,
    show: bool,
    started: Instant,
    last: Option<Instant>,
    last_width: usize,
    max_width: usize,
}

impl Progress {
    fn new(out: Out, name: &str, total: Option<u64>) -> Progress {
        let show =
            !out.json && std::io::stderr().is_terminal() && total.is_none_or(|t| t >= PROGRESS_MIN_BYTES);
        Progress {
            name: name.to_string(),
            total,
            show,
            started: Instant::now(),
            last: None,
            last_width: 0,
            max_width: progress_max_width(),
        }
    }

    fn update(&mut self, done: u64) {
        let now = Instant::now();
        if !self.show || self.last.is_some_and(|t| now.duration_since(t) < Duration::from_millis(200)) {
            return;
        }
        self.last = Some(now);
        let line = format_progress(&self.name, done, self.total, now.duration_since(self.started));
        let (frame, width) = render_progress_frame(&line, self.last_width, self.max_width);
        self.last_width = width;
        let mut e = std::io::stderr().lock();
        let _ = write!(e, "{frame}");
        let _ = e.flush();
    }

    fn finish(&mut self) {
        if self.show && self.last.take().is_some() {
            let clear = render_progress_clear(self.last_width, self.max_width);
            self.last_width = 0;
            let mut e = std::io::stderr().lock();
            let _ = write!(e, "{clear}");
            let _ = e.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_status_classification() {
        for s in [408, 429, 500, 502, 503, 504] {
            assert!(is_retryable_status(s), "status {s} should be retryable");
        }
        for s in [200, 302, 400, 401, 403, 404, 410, 451] {
            assert!(!is_retryable_status(s), "status {s} should not be retryable");
        }
    }

    #[test]
    fn retryable_transport_classification() {
        assert!(is_retryable_transport(&ureq::Error::HostNotFound));
        assert!(is_retryable_transport(&ureq::Error::ConnectionFailed));
        assert!(is_retryable_transport(&ureq::Error::BodyStalled));
        assert!(is_retryable_transport(&ureq::Error::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "eof"
        ))));
        assert!(is_retryable_transport(&ureq::Error::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "reset"
        ))));
        // Rustls surfaces certificate verification errors as io::ErrorKind::InvalidData.
        assert!(!is_retryable_transport(&ureq::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid peer certificate"
        ))));
        assert!(!is_retryable_transport(&ureq::Error::BadUri("bad".into())));
        assert!(!is_retryable_transport(&ureq::Error::TooManyRedirects));
        assert!(!is_retryable_transport(&ureq::Error::InvalidProxyUrl));
        assert!(!is_retryable_transport(&ureq::Error::Tls("cert rejected")));
    }

    #[test]
    fn backoff_delay_grows_exponentially_and_caps() {
        let base = Duration::from_millis(500);
        assert_eq!(backoff_delay(0, base, None), Duration::from_millis(500));
        assert_eq!(backoff_delay(1, base, None), Duration::from_secs(1));
        assert_eq!(backoff_delay(2, base, None), Duration::from_secs(2));
        assert_eq!(backoff_delay(3, base, None), Duration::from_secs(4));
        assert_eq!(backoff_delay(8, base, None), RETRY_MAX_DELAY);
        assert_eq!(backoff_delay(0, base, Some(Duration::from_secs(3))), Duration::from_secs(3));
        assert_eq!(backoff_delay(0, base, Some(Duration::from_secs(60))), RETRY_MAX_DELAY);
        assert_eq!(backoff_delay(2, Duration::ZERO, Some(Duration::from_secs(3))), Duration::ZERO);
    }

    #[test]
    fn parse_retry_after_handles_seconds_and_http_dates() {
        // RFC 9110 §5.6.7 canonical example: Sun, 06 Nov 1994 08:49:37 GMT == 784_111_777.
        let base_ts = UNIX_EPOCH + Duration::from_secs(784_111_767); // 10s earlier
        assert_eq!(parse_retry_after(Some("3"), base_ts), Some(Duration::from_secs(3)));
        assert_eq!(parse_retry_after(Some(" 12 "), base_ts), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_after(Some("0"), base_ts), Some(Duration::ZERO));
        assert_eq!(
            parse_retry_after(Some("Sun, 06 Nov 1994 08:49:37 GMT"), base_ts),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            parse_retry_after(Some("Sunday, 06-Nov-94 08:49:37 GMT"), base_ts),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            parse_retry_after(Some("Sun Nov  6 08:49:37 1994"), base_ts),
            Some(Duration::from_secs(10))
        );
        // Past dates clamp to zero rather than failing.
        let later_ts = UNIX_EPOCH + Duration::from_secs(784_111_800);
        assert_eq!(parse_retry_after(Some("Sun, 06 Nov 1994 08:49:37 GMT"), later_ts), Some(Duration::ZERO));
        // RFC 850 50-year rule: in 2040, "94" resolves to 1994 (past -> 0s), while "41" resolves to 2041.
        let year_2040 = UNIX_EPOCH + Duration::from_secs(days_from_civil(2040, 1, 1) as u64 * 86_400);
        assert_eq!(
            parse_retry_after(Some("Sunday, 06-Nov-94 08:49:37 GMT"), year_2040),
            Some(Duration::ZERO)
        );
        assert!(
            parse_retry_after(Some("Tuesday, 01-Jan-41 00:00:00 GMT"), year_2040)
                .is_some_and(|d| d > Duration::from_secs(86_400 * 300))
        );
        // Invalid or out-of-range values return None without panicking.
        assert_eq!(parse_retry_after(Some("not-a-date"), base_ts), None);
        assert_eq!(parse_retry_after(Some("Sun, 31 Feb 1994 08:49:37 GMT"), base_ts), None);
        assert_eq!(parse_retry_after(Some("Sun, 06 Nov 9223372036854775807 08:49:37 GMT"), base_ts), None);
        assert_eq!(parse_retry_after(Some("Sun, 06 Nov -1994 08:49:37 GMT"), base_ts), None);
        assert_eq!(parse_retry_after(Some("Sun Nov  6 08:49:37 999999999999999999"), base_ts), None);
        assert_eq!(parse_retry_after(Some("-1"), base_ts), None);
        assert_eq!(parse_retry_after(None, base_ts), None);
    }

    #[test]
    fn format_progress_reports_rate_and_eta() {
        // Early reads omit rate and ETA to avoid spikes.
        assert_eq!(
            format_progress("m.folo", 1_000_000, Some(10_000_000), Duration::from_millis(50)),
            "m.folo: 1.0 MB / 10.0 MB (10%)"
        );
        // Mid-download shows MB/s and seconds ETA.
        assert_eq!(
            format_progress("m.folo", 12_500_000, Some(50_000_000), Duration::from_secs(2)),
            "m.folo: 12.5 MB / 50.0 MB (25%), 6.2 MB/s, eta 6s"
        );
        // Slower download formats KB/s and minute/hour ETA.
        assert_eq!(
            format_progress("m.folo", 500_000, Some(33_000_000), Duration::from_secs(1)),
            "m.folo: 0.5 MB / 33.0 MB (2%), 500 KB/s, eta 1m05s"
        );
        assert_eq!(
            format_progress("m.folo", 1_000, Some(3_721_000), Duration::from_secs(1)),
            "m.folo: 0.0 MB / 3.7 MB (0%), 1 KB/s, eta 1h02m"
        );
        // Completed download shows rate without ETA.
        assert_eq!(
            format_progress("m.folo", 50_000_000, Some(50_000_000), Duration::from_secs(4)),
            "m.folo: 50.0 MB / 50.0 MB (100%), 12.5 MB/s"
        );
        // Unknown content-length shows rate without percentage or ETA.
        assert_eq!(
            format_progress("m.folo", 10_000_000, None, Duration::from_secs(2)),
            "m.folo: 10.0 MB, 5.0 MB/s"
        );
        // Overshoot never panics or exceeds 100%.
        assert_eq!(
            format_progress("m.folo", 12_000_000, Some(10_000_000), Duration::from_secs(2)),
            "m.folo: 12.0 MB / 10.0 MB (100%), 6.0 MB/s"
        );
    }

    #[test]
    fn progress_frames_fit_narrow_terminals_and_clear_trailing_chars() {
        let max_w = 36;
        let long = "folo-n-coco-v2.folo: 12.5 MB / 50.0 MB (25%), 6.2 MB/s, eta 1m05s";
        let (frame1, w1) = render_progress_frame(long, 0, max_w);
        assert_eq!(w1, max_w);
        assert!(frame1.starts_with('\r') && frame1.ends_with("..."));
        assert_eq!(frame1.chars().count() - 1, max_w);

        let short = "m.folo: 5.0 MB (50%)";
        let (frame2, w2) = render_progress_frame(short, w1, max_w);
        assert_eq!(w2, short.chars().count());
        // Padded to previous width to overwrite leftover characters, never wider than max_w.
        assert_eq!(frame2.chars().count() - 1, w1);
        assert_eq!(frame2, format!("\r{short}{}", " ".repeat(w1 - w2)));

        // Clearing never exceeds max_w and matches the last drawn width.
        let clear = render_progress_clear(w2, max_w);
        assert_eq!(clear, format!("\r{}\r", " ".repeat(w2)));
    }

    #[test]
    fn format_delay_formats_millis_and_seconds() {
        assert_eq!(format_delay(Duration::ZERO), "0ms");
        assert_eq!(format_delay(Duration::from_millis(250)), "250ms");
        assert_eq!(format_delay(Duration::from_secs(1)), "1s");
        assert_eq!(format_delay(Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_delay(Duration::from_secs(10)), "10s");
    }
}
