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
        "Asked the run in {} to stop after its current batch; its `folo train` ends with status `stopped`.",
        folo_api::json::path(run_dir)
    ));
    Ok(())
}

/// First Ctrl+C (on Windows also Ctrl+Break, which is what a script can send to a child started in its own
/// process group): stop after the current batch (the last epoch's checkpoint is kept). A second one exits.
pub(super) fn watch_ctrl_c(stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return };
        rt.block_on(async {
            #[cfg(windows)]
            let Ok(mut brk) = tokio::signal::windows::ctrl_break() else { return };
            for first in [true, false] {
                #[cfg(windows)]
                let got = tokio::select! {
                    r = tokio::signal::ctrl_c() => r.is_ok(),
                    r = brk.recv() => r.is_some(),
                };
                #[cfg(not(windows))]
                let got = tokio::signal::ctrl_c().await.is_ok();
                if !got {
                    return;
                }
                if !first {
                    std::process::exit(130);
                }
                eprintln!("\nstopping after the current batch (interrupt again to abort)…");
                stop.store(true, Ordering::Relaxed);
            }
        });
    });
}
