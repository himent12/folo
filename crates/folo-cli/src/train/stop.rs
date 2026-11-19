//! Stopping a run: `folo stop` and its stop file, and Ctrl+C.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::json;

use crate::output::{CliError, ExitCode, Out};

use super::STOP_FILE;

/// Polls `<run_dir>/stop` (written by `folo stop`) every 200 ms and raises `stop` when it appears. A stop file
/// left from an earlier process is removed first.
pub(super) fn watch_stop_file(run_dir: &Path, stop: Arc<AtomicBool>) {
    let file = run_dir.join(STOP_FILE);
    let _ = std::fs::remove_file(&file);
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            if file.exists() {
                let _ = std::fs::remove_file(&file);
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

/// After the run: whether a stop was asked for that the trainer did not take (the stop file written after the last
/// poll, which is removed here, or a flag raised after the trainer's last look).
pub(super) fn late_stop(run_dir: &Path, stop: &AtomicBool) -> bool {
    let file = std::fs::remove_file(run_dir.join(STOP_FILE)).is_ok();
    stop.swap(false, Ordering::Relaxed) | file
}

/// `folo stop <run_dir>`: asks the `folo train` writing that run folder to stop after its current batch.
pub fn request_stop(run_dir: &Path, out: Out) -> Result<(), CliError> {
    let not_a_run = |code: &str, exit, msg: String| {
        CliError::new(exit, code, msg).hint("pass the run_dir that `folo train` printed, e.g. runs/shapes-n")
    };
    if !run_dir.is_dir() {
        let msg = format!("{} does not exist", folo_api::json::path(run_dir));
        return Err(not_a_run("path_not_found", ExitCode::NotFound, msg));
    }
    if !run_dir.join("config.json").is_file() && !run_dir.join("weights").is_dir() {
        let msg = format!("{} is not a run folder (it has no config.json)", folo_api::json::path(run_dir));
        return Err(not_a_run("run_not_found", ExitCode::NotFound, msg));
    }
    let file = run_dir.join(STOP_FILE);
    std::fs::write(&file, b"stop\n").map_err(|e| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot write {}: {e}", folo_api::json::path(&file)),
        )
    })?;
    out.record(&json!({
        "event": "stop_requested",
        "run_dir": folo_api::json::path(run_dir),
        "stop_file": folo_api::json::path(&file),
    }));
    out.human(format!(
        "Asked the run in {} to stop after its current batch; its `folo train` ends with status `stopped` \
         (`completed` if its last batch is already done).",
        folo_api::json::path(run_dir)
    ));
    Ok(())
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
