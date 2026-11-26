//! A local `folo serve`: finding the binary, starting it on a free port, and stopping it with the app.

use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::api::*;
use ureq::unversioned::transport::{Buffers as _, Connector as _};

use super::refused::{ConnectedStream, Found};
use super::transport::{BODY_LIMIT, agent, json};

/// A local `folo serve` started by the app; stopped when dropped.
pub struct ServerProcess {
    child: Child,
    pub url: String,
}

impl ServerProcess {
    /// Whether the server is still running (it may have crashed, or been stopped by hand).
    pub fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

/// The runs of the server at `url` now, asked with a one-second limit (closing the window waits on it);
/// `None` when it doesn't answer in time.
pub fn runs_now(url: &str) -> Option<Vec<Run>> {
    // `url()`, not the base: an address that isn't one is never sent (port 80 for ":99999").
    let runs = super::HttpBackend::new(url).url("/runs");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_millis(500)))
        .timeout_global(Some(Duration::from_secs(1)))
        .build()
        .into();
    let mut r = agent.get(&runs).call().ok().filter(|r| r.status().is_success())?;
    json::<folo_api::RunList>(r.body_mut(), 64 << 20).ok().map(|l| l.runs)
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Arguments passed to `folo` when starting a local server: the real engine (never `--mock`), on a
/// free port, emitting JSON Lines so we read its `"listening"` record.
pub const SERVE_ARGS: &[&str] = &["--json", "serve", "--port", "0"];

/// Where to find the `folo` binary: `$FOLO_BIN`, next to this executable, or on `PATH`.
pub fn find_folo() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("FOLO_BIN").map(std::path::PathBuf::from)
        && p.is_file()
    {
        return Some(p);
    }
    let exe = if cfg!(windows) { "folo.exe" } else { "folo" };
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        let p = dir.join(exe);
        if p.is_file() {
            return Some(p);
        }
    }
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(exe)).find(|p| p.is_file()))
}

/// Why no `folo` binary was found, for a person: `$FOLO_BIN` names a file that isn't there (`fobin`), or
/// nothing was set and none is next to the app or on `PATH`.
fn not_found(fobin: Option<&std::ffi::OsStr>) -> String {
    match fobin.filter(|p| !p.is_empty()) {
        Some(p) => format!(
            "FOLO_BIN names {}, which isn't a file, and there's no folo next to FOLO Desktop or on PATH",
            std::path::Path::new(p).display()
        ),
        None => "folo was not found next to FOLO Desktop or on PATH (set FOLO_BIN to it)".into(),
    }
}

/// Whether `url` (as typed: a scheme, user info, a port and a path are all optional) names this machine:
/// `localhost`, `127.x.y.z` or `[::1]`. Only there can the app start the server itself.
pub fn is_loopback(url: &str) -> bool {
    let url = url.trim();
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let authority = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
    let host = match authority.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => authority.split(':').next().unwrap_or(""),
    };
    host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok_and(|a| a.is_loopback())
}

/// What answered a probe of an address ([`probe`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    /// A `folo serve`: its health, `status: "ok"`.
    Folo,
    /// An HTTP answer that is not a healthy FOLO health: a web page, a redirect, a proxy's sign-in, a
    /// server's error. The address's own answer says what is wrong there.
    Answered,
    /// No answer: nothing listens, or the connection failed, was reset or timed out.
    Nothing,
}

/// Whether a `folo serve` answers `GET /api/health` at `url`. On this machine the connection gets half a
/// second: a listening port accepts at once however busy the server is, and a port nothing listens on is
/// told first without waiting ([`super::refused::check`]: Windows refuses it only after its SYN retries, 2 s).
/// Through the proxy the environment names (`ALL_PROXY`, `HTTPS_PROXY`, `HTTP_PROXY`, minus `NO_PROXY`),
/// as every other call of the app goes.
pub fn probe(url: &str) -> Probe {
    probe_through(url, ureq::Proxy::try_from_env())
}

/// [`probe`] with the HTTP client's proxy given: `None` connects directly.
pub(super) fn probe_through(url: &str, proxy: Option<ureq::Proxy>) -> Probe {
    let connect = if is_loopback(url) { Duration::from_millis(500) } else { Duration::from_secs(2) };
    // `url()`, not the base: an address that isn't one is never sent (port 80 for ":99999").
    let health = super::HttpBackend::new(url).url("/health");
    // Nothing listens there (Windows says so only after 2 s otherwise; the connect limit below is 0.5 s).
    // Asked only when the client below connects to that address itself (ureq's own test: no proxy, or one
    // whose NO_PROXY names the host): through a proxy, a refused direct connection says nothing about what
    // the proxy reaches. Before the client is made, which costs more than the check.
    let direct = proxy.as_ref().is_none_or(|p| health.parse().is_ok_and(|uri| p.is_no_proxy(&uri)));
    let found = if direct { super::refused::check(&health) } else { Found::Unsure };
    if found.is_nothing() {
        return Probe::Nothing;
    }
    let config = ureq::Agent::config_builder()
        .http_status_as_error(false)
        // `folo serve` answers its health itself: a redirect is something else's (and a hop that
        // timed out connecting would look like the first one, so the request went again, #308).
        .max_redirects(0)
        .timeout_connect(Some(connect))
        .timeout_global(Some(Duration::from_secs(10)))
        .proxy(proxy)
        .build();
    let (agent, retry_connect) = match found {
        Found::Listener(Some(conn)) => {
            // Reuse the loopback socket `refused::check` already connected (#308): no throwaway
            // connection hits the listener's accept queue, so no second attempt is needed.
            let connector =
                ().chain(PreconnectedConnector::new(conn))
                    .chain(ureq::unversioned::transport::TcpConnector::default())
                    .chain(ureq::unversioned::transport::RustlsConnector::default());
            (
                ureq::Agent::with_parts(
                    config,
                    connector,
                    ureq::unversioned::resolver::DefaultResolver::default(),
                ),
                false,
            )
        }
        Found::Listener(None) => (config.into(), true),
        Found::Unsure | Found::Nothing => (config.into(), false),
    };
    let (answer, _) = ask_health(retry_connect, || agent.get(&health).call());
    let Ok(mut r) = answer else { return Probe::Nothing };
    // A FOLO server's health, not any web server that answers 200 (a dev server's index page).
    if r.status().is_success() && json::<Health>(r.body_mut(), BODY_LIMIT).is_ok_and(|h| h.status == "ok") {
        Probe::Folo
    } else {
        Probe::Answered
    }
}

/// The health request (`call`), and how many times it was sent. When `retry_connect` is set (the
/// listening check saw a listener but could not hand its socket over, so a closed empty connection may
/// still occupy the last accept-queue slot) and the first attempt never connected, it is sent once more
/// before nobody is said to be there. Only a failure to connect is tried again (#308): an answer is
/// final, and so is a connection that was made and then reset, aborted or timed out.
pub(super) fn ask_health<T>(
    retry_connect: bool,
    mut call: impl FnMut() -> Result<T, ureq::Error>,
) -> (Result<T, ureq::Error>, u32) {
    let answer = call();
    if retry_connect && answer.as_ref().is_err_and(not_connected) {
        return (call(), 2);
    }
    (answer, 1)
}

/// Whether a call never connected: the connect limit passed, or the client found no connection to make.
fn not_connected(e: &ureq::Error) -> bool {
    matches!(e, ureq::Error::Timeout(ureq::Timeout::Connect) | ureq::Error::ConnectionFailed)
}

/// Hands a single already-connected loopback socket from [`super::refused::check`] into a `ureq`
/// connector chain, falling through to [`ureq::unversioned::transport::TcpConnector`] if empty.
#[derive(Debug)]
struct PreconnectedConnector {
    conn: Mutex<Option<ConnectedStream>>,
}

impl PreconnectedConnector {
    fn new(conn: ConnectedStream) -> Self {
        Self { conn: Mutex::new(Some(conn)) }
    }
}

impl ureq::unversioned::transport::Connector<()> for PreconnectedConnector {
    type Out = PreconnectedTransport;

    fn connect(
        &self,
        details: &ureq::unversioned::transport::ConnectionDetails,
        _chained: Option<()>,
    ) -> Result<Option<Self::Out>, ureq::Error> {
        let Some(conn) = self.conn.lock().unwrap().take() else {
            return Ok(None);
        };
        if details.config.no_delay() {
            let _ = conn.stream().set_nodelay(true);
        }
        let buffers = ureq::unversioned::transport::LazyBuffers::new(
            details.config.input_buffer_size(),
            details.config.output_buffer_size(),
        );
        let transport = PreconnectedTransport {
            stream: Arc::new(conn),
            buffers,
            timeout_read: None,
            timeout_write: None,
        };
        let handle =
            Arc::downgrade(&transport.stream) as Weak<dyn ureq::unversioned::transport::ShutdownHandle>;
        (details.register_shutdown)(handle)?;
        Ok(Some(transport))
    }
}

/// One-shot `ureq` [`Transport`](ureq::unversioned::transport::Transport) over a [`ConnectedStream`].
#[derive(Debug)]
struct PreconnectedTransport {
    stream: Arc<ConnectedStream>,
    buffers: ureq::unversioned::transport::LazyBuffers,
    timeout_read: Option<Duration>,
    timeout_write: Option<Duration>,
}

impl PreconnectedTransport {
    fn apply_timeout(
        &self,
        cached: &mut Option<Duration>,
        next: Option<Duration>,
        setter: fn(&std::net::TcpStream, Option<Duration>) -> std::io::Result<()>,
    ) -> Result<(), ureq::Error> {
        if next != *cached {
            setter(self.stream.stream(), next)?;
            *cached = next;
        }
        Ok(())
    }
}

impl ureq::unversioned::transport::Transport for PreconnectedTransport {
    fn buffers(&mut self) -> &mut dyn ureq::unversioned::transport::Buffers {
        &mut self.buffers
    }

    fn transmit_output(
        &mut self,
        amount: usize,
        timeout: ureq::unversioned::transport::NextTimeout,
    ) -> Result<(), ureq::Error> {
        let timeout = timeout.for_write();
        if timeout.after.is_zero() {
            return Err(ureq::Error::Timeout(timeout.reason));
        }
        let limit = timeout.not_zero().map(|d| *d);
        let mut cached = self.timeout_write;
        self.apply_timeout(&mut cached, limit, std::net::TcpStream::set_write_timeout)?;
        self.timeout_write = cached;
        let output = &self.buffers.output()[..amount];
        match self.stream.stream().write_all(output) {
            Ok(()) => Ok(()),
            Err(e) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) => {
                Err(ureq::Error::Timeout(timeout.reason))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn await_input(
        &mut self,
        timeout: ureq::unversioned::transport::NextTimeout,
    ) -> Result<bool, ureq::Error> {
        let timeout = timeout.for_read();
        if timeout.after.is_zero() {
            return Err(ureq::Error::Timeout(timeout.reason));
        }
        let limit = timeout.not_zero().map(|d| *d);
        let mut cached = self.timeout_read;
        self.apply_timeout(&mut cached, limit, std::net::TcpStream::set_read_timeout)?;
        self.timeout_read = cached;
        let started = Instant::now();
        let buf = self.buffers.input_append_buf();
        let amount = loop {
            match self.stream.stream().read(buf) {
                Ok(n) => break n,
                Err(e) if e.kind() == ErrorKind::Interrupted => {
                    if let Some(total) = limit {
                        let left = total.saturating_sub(started.elapsed());
                        if left.is_zero() {
                            return Err(ureq::Error::Timeout(timeout.reason));
                        }
                        self.stream.stream().set_read_timeout(Some(left))?;
                        self.timeout_read = Some(left);
                    }
                }
                Err(e) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) => {
                    return Err(ureq::Error::Timeout(timeout.reason));
                }
                Err(e) => return Err(e.into()),
            }
        };
        self.buffers.input_appended(amount);
        Ok(amount > 0)
    }

    fn is_open(&mut self) -> bool {
        // Single-use probe transport: never returned to a connection pool.
        false
    }
}

/// Make the OS stop `child` when this process exits, even if it is killed (no Drop runs then).
#[cfg(windows)]
fn bind_to_app(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    // SAFETY: plain Win32 calls on handles we own. The job handle is deliberately never closed:
    // the OS closes it when this process exits, and KILL_ON_JOB_CLOSE then ends the child.
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok != 0 {
            AssignProcessToJobObject(job, child.as_raw_handle() as _);
        }
    }
}

#[cfg(not(windows))]
fn bind_to_app(_child: &Child) {}

/// What starting a local `folo serve` gave.
pub enum Spawned {
    /// A server this app started (and stops when the [`ServerProcess`] drops).
    Started(ServerProcess),
    /// The data dir already has a `folo serve` (another app's, or one started by hand), listening here: use it,
    /// don't start a second one on the same data.
    Running(String),
}

/// What the starting server's JSON Lines told us.
enum Startup {
    Listening(String),
    /// Where the data dir's owner listens (when it says), and the data dir.
    InUse(Option<String>, Option<String>),
    Failed(String),
}

/// Reads one JSON Lines record of `folo --json serve`.
fn startup_record(line: &str) -> Option<Startup> {
    let v = serde_json::from_str::<serde_json::Value>(line).ok()?;
    let url = v["url"].as_str().map(str::to_string);
    match v["event"].as_str() {
        Some("listening") => url.map(Startup::Listening),
        Some("data_dir_in_use") => Some(Startup::InUse(url, v["data_dir"].as_str().map(str::to_string))),
        _ => v["error"]["message"].as_str().map(|m| Startup::Failed(m.to_string())),
    }
}

/// Start `folo serve` (the real engine, bare until datasets/runs exist) on a free port and wait
/// (up to `timeout`) for its address.
/// When the data dir already has an owner, connect to it once it is listening there: retried until `timeout`,
/// as the owner may still be starting (or be an embedded `folo mcp`, which has no address).
pub fn spawn_server(timeout: Duration) -> Result<Spawned, String> {
    let bin = find_folo().ok_or_else(|| not_found(std::env::var_os("FOLO_BIN").as_deref()))?;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let busy = match spawn_server_bin(&bin, left)? {
            Attempt::Started(proc_) => return Ok(Spawned::Started(proc_)),
            Attempt::InUse(Some(url), Some(dir))
                if serves(&url, &dir, deadline.saturating_duration_since(std::time::Instant::now())) =>
            {
                return Ok(Spawned::Running(url));
            }
            Attempt::InUse(..) => "another FOLO process (folo serve or folo mcp) is using the data dir",
        };
        if std::time::Instant::now() + Duration::from_millis(500) >= deadline {
            return Err(busy.into());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Whether the server at `url` serves the data dir `dir` (`serve.json` can be stale, and its port reused), asked
/// within `budget` (the start deadline's remainder).
fn serves(url: &str, dir: &str, budget: Duration) -> bool {
    let same = |a: &str, b: &str| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    };
    if budget.is_zero() {
        return false;
    }
    let system = agent(Some(budget.min(Duration::from_secs(5))))
        .get(&format!("{}/api/system", url.trim_end_matches('/')))
        .call()
        .ok()
        .filter(|r| r.status().is_success())
        .and_then(|mut r| json::<SystemResponse>(r.body_mut(), BODY_LIMIT).ok());
    system.and_then(|s| s.data_dir).is_some_and(|d| same(&d, dir))
}

/// One start of `folo serve`.
enum Attempt {
    Started(ServerProcess),
    /// The data dir has an owner (the child has exited): where it listens, when it says, and the data dir.
    InUse(Option<String>, Option<String>),
}

/// The command that starts the app's own `folo serve`. Without `FOLO_TOKEN`: a server started with
/// one (`folo serve` reads it from the environment) answers every call `401`, and the app can't send a
/// token, so a token set for the CLI or for a remote server would leave every screen saying "This
/// server needs a token" about a server the person never started.
fn serve_command(bin: &std::path::Path) -> Command {
    let mut cmd = Command::new(bin);
    cmd.args(SERVE_ARGS)
        .env_remove("FOLO_TOKEN")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Why `bin` could not be started, said for a person: a file that is no program (a renamed text file,
/// a program for another machine) reads "isn't a program", not Windows' "This version of %1 is not
/// compatible…".
fn start_failed(bin: &std::path::Path, e: &std::io::Error) -> String {
    // ERROR_BAD_EXE_FORMAT, ERROR_EXE_MACHINE_TYPE_MISMATCH; ENOEXEC elsewhere.
    let not_a_program = if cfg!(windows) { [193, 216].as_slice() } else { [8].as_slice() };
    if e.raw_os_error().is_some_and(|c| not_a_program.contains(&c)) {
        return format!(
            "{} isn't a program this PC can run: FOLO_BIN should name FOLO's folo{}",
            bin.display(),
            if cfg!(windows) { ".exe" } else { "" }
        );
    }
    format!("could not start {}: {e}", bin.display())
}

fn spawn_server_bin(bin: &std::path::Path, timeout: Duration) -> Result<Attempt, String> {
    let mut child = serve_command(bin).spawn().map_err(|e| start_failed(bin, &e))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(rec) = startup_record(&line)
                && tx.send(rec).is_err()
            {
                break;
            }
        }
    });
    bind_to_app(&child);
    let outcome = rx.recv_timeout(timeout);
    if !matches!(outcome, Ok(Startup::Listening(_))) {
        let _ = child.kill();
        let _ = child.wait();
    }
    match outcome {
        Ok(Startup::Listening(url)) => Ok(Attempt::Started(ServerProcess { child, url })),
        Ok(Startup::InUse(url, dir)) => Ok(Attempt::InUse(url, dir)),
        Ok(Startup::Failed(message)) => Err(message),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err("folo serve did not start in time".into()),
        // No record at all: whatever ran isn't a folo that serves.
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err(format!("{} stopped before it was listening (is it FOLO's folo?)", bin.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Answers each request on `listener` with `answer` until `stop`, and says how many it had (a
    /// connection closed without a request, the probe's check, is none).
    fn serve(listener: &std::net::TcpListener, answer: &str, stop: &std::sync::atomic::AtomicBool) -> usize {
        use std::io::{Read, Write};
        listener.set_nonblocking(true).unwrap();
        let mut asked = 0;
        while !stop.load(std::sync::atomic::Ordering::SeqCst) {
            let Ok((mut sock, _)) = listener.accept() else {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            };
            sock.set_nonblocking(false).unwrap();
            // A connection that sends nothing cannot keep the test from stopping.
            sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut buf = [0u8; 4096];
            if sock.read(&mut buf).unwrap_or(0) > 0 {
                asked += 1;
                let _ = sock.write_all(answer.as_bytes());
            }
        }
        asked
    }

    /// #390 review: a health answered with a redirect was followed (and a hop that timed out
    /// connecting sent the health request again). It is not: here the redirect leads to a working
    /// health, which is never asked, and the address that redirects is no `folo serve`.
    #[test]
    fn a_redirected_health_is_not_followed() {
        let (redirects, health) = (
            std::net::TcpListener::bind("127.0.0.1:0").unwrap(),
            std::net::TcpListener::bind("127.0.0.1:0").unwrap(),
        );
        let (from, to) = (redirects.local_addr().unwrap(), health.local_addr().unwrap());
        let moved = format!(
            "HTTP/1.1 302 Found\r\nlocation: http://{to}/api/health\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
        );
        let body = r#"{ "status": "ok", "version": "0.1.0", "mock": false }"#;
        let ok = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let stop = std::sync::atomic::AtomicBool::new(false);
        // Sets `stop` when dropped: the workers end however the scope is left (a spawn that fails).
        struct Stop<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let (found, asked) = std::thread::scope(|s| {
            let guard = Stop(&stop);
            let first = s.spawn(|| serve(&redirects, &moved, &stop));
            let second = s.spawn(|| serve(&health, &ok, &stop));
            let found = probe_through(&format!("http://{from}"), None);
            drop(guard);
            (found, (first.join().unwrap(), second.join().unwrap()))
        });
        assert_eq!((found, asked), (Probe::Answered, (1, 0)), "asked once, and the redirect not followed");
    }

    #[test]
    fn only_this_machines_addresses_are_loopback() {
        for url in [
            "http://127.0.0.1:7878",
            "127.0.0.1:1",
            " localhost:7878/ ",
            "http://LOCALHOST",
            "https://127.4.5.6:9/api",
            "http://[::1]:7878",
            "http://me@127.0.0.1:7878",
        ] {
            assert!(is_loopback(url), "{url}");
        }
        for url in
            ["http://192.168.9.9:7878", "10.0.0.2", "http://example.com", "abc def", "", "http://[::2]:1"]
        {
            assert!(!is_loopback(url), "{url}");
        }
    }

    #[test]
    fn a_missing_folo_names_what_was_looked_for() {
        let set = not_found(Some(std::ffi::OsStr::new(r"D:\nope\folo.exe")));
        assert!(set.starts_with(r"FOLO_BIN names D:\nope\folo.exe, which isn't a file"), "{set}");
        assert!(not_found(None).contains("set FOLO_BIN"));
        assert_eq!(not_found(Some(std::ffi::OsStr::new(""))), not_found(None));
    }

    #[test]
    fn startup_records_are_read() {
        let url = |r: Option<Startup>| match r {
            Some(Startup::Listening(u)) => format!("listening {u}"),
            Some(Startup::InUse(u, d)) => format!("in use {u:?} {d:?}"),
            Some(Startup::Failed(m)) => format!("failed {m}"),
            None => "none".into(),
        };
        assert_eq!(
            url(startup_record(r#"{"event":"listening","url":"http://127.0.0.1:5000","mock":false}"#)),
            "listening http://127.0.0.1:5000"
        );
        assert_eq!(
            url(startup_record(
                r#"{"event":"data_dir_in_use","data_dir":"C:/x","url":"http://127.0.0.1:7"}"#
            )),
            r#"in use Some("http://127.0.0.1:7") Some("C:/x")"#
        );
        assert_eq!(url(startup_record(r#"{"event":"data_dir_in_use","url":null}"#)), "in use None None");
        assert_eq!(
            url(startup_record(r#"{"error":{"code":"bind_failed","message":"could not bind","hint":null}}"#)),
            "failed could not bind"
        );
        assert_eq!(url(startup_record("FOLO serving")), "none");
    }

    #[test]
    fn the_started_server_never_inherits_a_token() {
        let cmd = serve_command(std::path::Path::new("folo"));
        let token = cmd.get_envs().find(|(k, _)| *k == std::ffi::OsStr::new("FOLO_TOKEN"));
        assert_eq!(token, Some((std::ffi::OsStr::new("FOLO_TOKEN"), None)), "FOLO_TOKEN is removed");
    }

    #[test]
    fn a_file_that_is_no_program_is_named_so() {
        let bin = std::path::Path::new(r"D:\tools\folo.exe");
        let code = if cfg!(windows) { 216 } else { 8 };
        let said = start_failed(bin, &std::io::Error::from_raw_os_error(code));
        assert!(said.starts_with(r"D:\tools\folo.exe isn't a program this PC can run"), "{said}");
        assert!(!said.contains("%1"), "{said}");
        if cfg!(windows) {
            assert!(start_failed(bin, &std::io::Error::from_raw_os_error(193)).contains("isn't a program"));
        }
        // Anything else keeps the system's words.
        let other = start_failed(bin, &std::io::Error::other("busy"));
        assert_eq!(other, r"could not start D:\tools\folo.exe: busy");
    }

    #[test]
    fn spawned_server_uses_the_real_store_not_mock() {
        assert_eq!(SERVE_ARGS, ["--json", "serve", "--port", "0"]);
        assert!(!SERVE_ARGS.contains(&"--mock"));
    }
}
