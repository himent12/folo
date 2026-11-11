//! How `folo weights pull` gets a release's files: over plain HTTPS without credentials first (all a public release
//! needs, so the `folo` binary is enough), else with the GitHub CLI (`gh release download`), which reaches the
//! releases of a private repository the user can see, drafts included for its writers.

use std::fs::File;
use std::io::{IsTerminal, Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::output::{CliError, ExitCode, Out};

/// Downloads from `<base>/<tag>/<asset>` instead of the GitHub release (a mirror, or a test's server).
pub const BASE_URL_ENV: &str = "FOLO_WEIGHTS_BASE_URL";
/// The GitHub CLI program the fallback runs (default `gh`).
pub const GH_ENV: &str = "FOLO_GH";

/// The longest wait for the host name to resolve, and then for the connection (TLS included) to be set up.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// The longest wait for the request's head to be sent, for an answer's head, and for each read of its body: a
/// stalled download fails, a slow one does not (no whole-request deadline).
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Files smaller than this get no progress line (`SHA256SUMS`).
const PROGRESS_MIN_BYTES: u64 = 1 << 20;

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

/// `GET url` into `dest`, streamed, redirects followed; `Some` when it is refused ([`Refused`]).
fn fetch(
    agent: &ureq::Agent,
    url: &str,
    name: &str,
    dest: &Path,
    out: Out,
) -> Result<Option<Refused>, CliError> {
    let res = agent.get(url).call().map_err(|e| {
        CliError::new(ExitCode::Runtime, "download_failed", format!("could not download {url}: {e}")).hint(format!(
            "check this machine's network connection (or proxy) and try again; {BASE_URL_ENV} downloads from a \
             mirror instead"
        ))
    })?;
    let status = res.status().as_u16();
    if matches!(status, 401 | 403 | 404) {
        return Ok(Some(Refused { url: url.to_string(), status }));
    }
    if !res.status().is_success() {
        return Err(CliError::new(ExitCode::Runtime, "download_failed", format!("{url} answered {status}"))
            .hint(format!(
                "the server's trouble: try again later; {BASE_URL_ENV} downloads from a mirror instead"
            )));
    }
    let total = res.body().content_length();
    let mut body = res.into_body().into_reader();
    let mut file = File::create(dest).map_err(|e| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot create {}: {e}", folo_api::json::path(dest)),
        )
    })?;
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
                return Err(broke_off(done, e.to_string()));
            }
        };
        file.write_all(&buf[..n]).map_err(|e| {
            CliError::new(
                ExitCode::Runtime,
                "io_error",
                format!("cannot write {}: {e}", folo_api::json::path(dest)),
            )
        })?;
        done += n as u64;
        progress.update(done);
    }
    progress.finish();
    if let Some(t) = total.filter(|&t| t != done) {
        return Err(broke_off(done, format!("the server announced {t} bytes")));
    }
    Ok(None)
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

/// A download's progress line for people: on a terminal, not with `--json`, for files of a MB or more.
struct Progress {
    name: String,
    total: Option<u64>,
    show: bool,
    last: Option<Instant>,
}

impl Progress {
    fn new(out: Out, name: &str, total: Option<u64>) -> Progress {
        let show =
            !out.json && std::io::stderr().is_terminal() && total.is_none_or(|t| t >= PROGRESS_MIN_BYTES);
        Progress { name: name.to_string(), total, show, last: None }
    }

    fn update(&mut self, done: u64) {
        let now = Instant::now();
        if !self.show || self.last.is_some_and(|t| now.duration_since(t) < Duration::from_millis(200)) {
            return;
        }
        self.last = Some(now);
        let of = match self.total {
            Some(t) => format!(" / {} ({:.0}%)", mb(t), 100.0 * done as f64 / t.max(1) as f64),
            None => String::new(),
        };
        let mut e = std::io::stderr().lock();
        let _ = write!(e, "\r{}: {}{of}   ", self.name, mb(done));
        let _ = e.flush();
    }

    fn finish(&mut self) {
        if self.show && self.last.take().is_some() {
            let mut e = std::io::stderr().lock();
            let _ = write!(e, "\r{:72}\r", "");
            let _ = e.flush();
        }
    }
}
