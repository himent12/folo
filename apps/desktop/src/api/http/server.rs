//! A local `folo serve`: finding the binary, starting it on a free port, and stopping it with the app.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::channel;
use std::time::Duration;

use crate::api::*;

use super::transport::agent;

/// A local `folo serve` started by the app; stopped when dropped.
pub struct ServerProcess {
    child: Child,
    pub url: String,
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
    let bin = find_folo().ok_or("folo was not found (set FOLO_BIN or put folo on PATH)")?;
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
        .and_then(|mut r| r.body_mut().read_json::<SystemResponse>().ok());
    system.and_then(|s| s.data_dir).is_some_and(|d| same(&d, dir))
}

/// One start of `folo serve`.
enum Attempt {
    Started(ServerProcess),
    /// The data dir has an owner (the child has exited): where it listens, when it says, and the data dir.
    InUse(Option<String>, Option<String>),
}

fn spawn_server_bin(bin: &std::path::Path, timeout: Duration) -> Result<Attempt, String> {
    let mut cmd = Command::new(bin);
    cmd.args(SERVE_ARGS).stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", bin.display()))?;
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
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err("folo serve exited before it was listening".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn spawned_server_uses_the_real_store_not_mock() {
        assert_eq!(SERVE_ARGS, ["--json", "serve", "--port", "0"]);
        assert!(!SERVE_ARGS.contains(&"--mock"));
    }
}
