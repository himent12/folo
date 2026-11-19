//! Stopping a run: `folo stop` and its stop file, the run folder's claim it relies on, and Ctrl+C.
//!
//! `folo stop` decides from two things only the trainer makes: [`RUN_MARKER`] says a folder is a `folo train` run
//! folder, and the OS lock on [`RUN_LOCK`] (taken the way `serve.lock` is) says a `folo train` process still owns
//! it. The lock goes with the process, so a killed trainer leaves no live-looking run behind.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::json;

use crate::output::{CliError, ExitCode, Out};

use super::STOP_FILE;

/// Written by the `folo train` that reserves (or resumes) a run folder: what makes it one for `folo stop`.
pub const RUN_MARKER: &str = "folo-train.json";
/// Locked by the `folo train` process that owns the run folder, from reservation to its terminal cleanup.
pub const RUN_LOCK: &str = "train.lock";
const MARKER_BODY: &str = "{\"owner\":\"folo train\"}\n";

/// A `folo train` process's hold on its run folder; dropping it (the run ended, or failed) removes a stop request
/// still there, then lets go of the lock.
pub(super) struct RunClaim {
    dir: PathBuf,
    /// Locked; dropping it unlocks (the OS also releases it when the process dies).
    _lock: File,
}

impl RunClaim {
    /// Takes `run_dir`'s lock and marks the folder, before the trainer sets up; `None` while another process holds
    /// it. A stop file there is from an earlier process (`folo stop` writes one only while the lock is held), so it
    /// is removed.
    pub(super) fn take(run_dir: &Path) -> std::io::Result<Option<RunClaim>> {
        let lock =
            OpenOptions::new().create(true).truncate(false).write(true).open(run_dir.join(RUN_LOCK))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Error(e)) => return Err(e),
        }
        match std::fs::remove_file(run_dir.join(STOP_FILE)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
        std::fs::write(run_dir.join(RUN_MARKER), MARKER_BODY)?;
        Ok(Some(RunClaim { dir: run_dir.to_path_buf(), _lock: lock }))
    }
}

impl Drop for RunClaim {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.dir.join(STOP_FILE));
    }
}

/// Polls `<run_dir>/stop` (written by `folo stop`) every 200 ms and raises `stop` when it appears. The file stays
/// until the run's [`RunClaim`] is dropped, so asking again gets the same answer.
pub(super) fn watch_stop_file(run_dir: &Path, stop: Arc<AtomicBool>) {
    let file = run_dir.join(STOP_FILE);
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            if file.exists() {
                eprintln!(
                    "stop requested ({}); stopping after the current batch…",
                    folo_api::json::path(&file)
                );
                stop.store(true, Ordering::Relaxed);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    });
}

/// `folo stop <run_dir>`: asks the `folo train` that owns that run folder to stop after its current batch.
pub fn request_stop(run_dir: &Path, out: Out) -> Result<(), CliError> {
    let shown = folo_api::json::path(run_dir);
    if !run_dir.is_dir() {
        return Err(CliError::new(ExitCode::NotFound, "path_not_found", format!("{shown} does not exist"))
            .hint("pass the run_dir that `folo train` printed, e.g. runs/shapes-n"));
    }
    let io_error = |p: &Path, e: std::io::Error| {
        CliError::new(ExitCode::Runtime, "io_error", format!("cannot use {}: {e}", folo_api::json::path(p)))
    };
    let marked = std::fs::read(run_dir.join(RUN_MARKER))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["owner"] == "folo train");
    let lock = run_dir.join(RUN_LOCK);
    let file = run_dir.join(STOP_FILE);
    if marked && lock_held(&lock).map_err(|e| io_error(&lock, e))? {
        // A request already there stays as it is: the trainer removes it when the run ends.
        match OpenOptions::new().write(true).create_new(true).open(&file) {
            Ok(mut f) => f.write_all(b"stop\n"),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && file.is_file() => Ok(()),
            Err(e) => Err(e),
        }
        .map_err(|e| io_error(&file, e))?;
        out.record(&json!({
            "event": "stop_requested",
            "run_dir": shown,
            "stop_file": folo_api::json::path(&file),
        }));
        out.human(format!(
            "Asked the run in {shown} to stop after its current batch; its `folo train` ends with status `stopped` \
             (`completed` when the request comes after its last batch)."
        ));
        return Ok(());
    }
    if let Some(id) = server_run_id(run_dir) {
        return Err(CliError::new(
            ExitCode::Runtime,
            "run_owned_by_server",
            format!("{shown} is the folder of the FOLO server run `{id}`, which `folo stop` cannot stop"),
        )
        .hint(format!("stop it in Studio, or with POST /api/runs/{id}/stop on that server")));
    }
    if !marked {
        return Err(CliError::new(
            ExitCode::NotFound,
            "run_not_found",
            format!("{shown} is not a `folo train` run folder (it has no {RUN_MARKER})"),
        )
        .hint("pass the run_dir that `folo train` printed, e.g. runs/shapes-n"));
    }
    // No `folo train` owns the run: it ended, or its process died. A request left from then is stale.
    match std::fs::remove_file(&file) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io_error(&file, e)),
        _ => {}
    }
    Err(CliError::new(
        ExitCode::Runtime,
        "run_not_active",
        format!("no `folo train` is running the run in {shown}; there is nothing to stop"),
    ))
}

/// Whether a process holds the lock on `lock` (a missing file: nobody does).
fn lock_held(lock: &Path) -> std::io::Result<bool> {
    let f = match File::open(lock) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    match f.try_lock() {
        Ok(()) => Ok(false),
        Err(TryLockError::WouldBlock) => Ok(true),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// The run id in `run_dir/run.json` when the folder is a FOLO server's run (the server writes `{"run": {"id"…}}`
/// there and stops its runs through its own API, never through a stop file).
fn server_run_id(run_dir: &Path) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(run_dir.join("run.json")).ok()?).ok()?;
    v["run"]["id"].as_str().map(str::to_string)
}

/// First Ctrl+C (on Windows also Ctrl+Break, which is what a script can send to a child started in its own
/// process group): stop after the current batch (the last epoch's checkpoint is kept). A second one exits.
pub(super) fn watch_ctrl_c(stop: Arc<AtomicBool>) {
    watch_interrupt(stop, "stopping after the current batch (interrupt again to abort)…");
}

/// The first Ctrl+C (or Ctrl+Break) sets `stop` and prints `message`; a second one exits (130). Returns once the
/// handlers are registered (or after a second, if that stalls), so an interrupt right after the call is not lost
/// (a handler registered later let the default one kill the process).
pub(crate) fn watch_interrupt(stop: Arc<AtomicBool>, message: &'static str) {
    let (ready, registered) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return };
        rt.block_on(async {
            // Streams register their handler when made, not when first polled.
            #[cfg(windows)]
            let (Ok(mut int), Ok(mut brk)) =
                (tokio::signal::windows::ctrl_c(), tokio::signal::windows::ctrl_break())
            else {
                return;
            };
            #[cfg(not(windows))]
            let Ok(mut int) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            else {
                return;
            };
            let _ = ready.send(());
            for first in [true, false] {
                #[cfg(windows)]
                let got = tokio::select! {
                    r = int.recv() => r.is_some(),
                    r = brk.recv() => r.is_some(),
                };
                #[cfg(not(windows))]
                let got = int.recv().await.is_some();
                if !got {
                    return;
                }
                if !first {
                    std::process::exit(130);
                }
                eprintln!("\n{message}");
                stop.store(true, Ordering::Relaxed);
            }
        });
    });
    // A failed registration drops `ready`, which ends the wait too.
    let _ = registered.recv_timeout(std::time::Duration::from_secs(1));
}
