//! OS subprocess lifecycle, stdio pipe reader/writer threads, and strict process-tree cleanup for `agy acp`.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_api::LabelUsage;
use serde_json::Value;

use super::acp::{CANCEL_GRACE, EXIT_DRAIN, drive_acp_inner};
use super::protocol::{
    Isolation, LABEL_AGENT_NAME, LABEL_CLI_DIR, UsageTracker, prepare_isolation, resolve_executable,
};
use crate::label::agent::{AgentError, AgentReply, AgentRequest, Failure, last_line};
use crate::label::tree::Tree;

/// How long to wait for `agy acp` to exit cleanly after closing stdin before killing its process tree.
const CLOSE_GRACE: Duration = Duration::from_millis(250);
/// Maximum bytes retained from child stderr (most recent tail).
pub(super) const MAX_STDERR_BYTES: usize = 64 * 1024;

/// Bounded stderr tail with EOF tracking so unexpected exits wait briefly for diagnostics without unbounded memory.
pub(super) struct StderrTail {
    bytes: Mutex<VecDeque<u8>>,
    ended: AtomicBool,
}

impl StderrTail {
    pub(super) fn new() -> Arc<StderrTail> {
        Arc::new(StderrTail { bytes: Mutex::new(VecDeque::new()), ended: AtomicBool::new(false) })
    }

    pub(super) fn push(&self, chunk: &[u8]) {
        let mut buf = self.bytes.lock().unwrap();
        if chunk.len() >= MAX_STDERR_BYTES {
            buf.clear();
            buf.extend(&chunk[chunk.len() - MAX_STDERR_BYTES..]);
            return;
        }
        let excess = (buf.len() + chunk.len()).saturating_sub(MAX_STDERR_BYTES);
        if excess > 0 {
            buf.drain(..excess);
        }
        buf.extend(chunk);
    }

    pub(super) fn mark_ended(&self) {
        self.ended.store(true, Ordering::Release);
    }

    pub(super) fn ended(&self) -> bool {
        self.ended.load(Ordering::Acquire)
    }

    pub(super) fn drain_text(&self, wait: Duration) -> String {
        let deadline = Instant::now() + wait;
        while !self.ended() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let buf = self.bytes.lock().unwrap();
        let (a, b) = buf.as_slices();
        let mut all = Vec::with_capacity(a.len() + b.len());
        all.extend_from_slice(a);
        all.extend_from_slice(b);
        String::from_utf8_lossy(&all).into_owned()
    }
}

/// Process-tree guard for `agy acp`: tracks child exit separately from process-group termination so descendants on
/// Unix (and Windows job members, required via [`Tree::start_strict`]) are always killed even when the immediate
/// child exits cleanly.
struct AcpProcess {
    child: Child,
    tree: Tree,
    child_exited: bool,
    tree_killed: bool,
}

impl AcpProcess {
    fn spawn(path: &Path, iso: &Isolation) -> Result<AcpProcess, AgentError> {
        let mut cmd = acp_command(path, iso);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        Tree::prepare(&mut cmd);
        let program = cmd.get_program().to_string_lossy().into_owned();
        let unavailable = |e: std::io::Error| AgentError::Unavailable(format!("starting `{program}`: {e}"));
        let mut child = cmd.spawn().map_err(unavailable)?;
        let tree = Tree::start_strict(&child).map_err(|e| {
            let _ = child.kill();
            let _ = child.wait();
            unavailable(e)
        })?;
        Ok(AcpProcess { child, tree, child_exited: false, tree_killed: false })
    }

    fn poll_exited(&mut self) -> bool {
        if !self.child_exited && matches!(self.child.try_wait(), Ok(Some(_))) {
            self.child_exited = true;
        }
        self.child_exited
    }

    fn finish_gracefully(&mut self, stdout_ended: &AtomicBool, stderr_ended: &AtomicBool) {
        let deadline = Instant::now() + CLOSE_GRACE;
        while Instant::now() < deadline {
            if self.poll_exited()
                && stdout_ended.load(Ordering::Acquire)
                && stderr_ended.load(Ordering::Acquire)
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // Always terminate the process group / job even after a clean parent exit so no helper outlives the call.
        self.kill();
    }

    fn kill(&mut self) {
        if !self.tree_killed {
            self.tree.kill(&mut self.child);
            self.tree_killed = true;
            self.child_exited = true;
        }
    }
}

impl Drop for AcpProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Builds the isolated `agy acp` command in the scratch workspace with the restricted `folo-labeler` agent, strict
/// `folo-label` CLI settings, terminal sandbox, default customizations off, and telemetry off.
pub(super) fn acp_command(path: &Path, iso: &Isolation) -> Command {
    let resolved = resolve_executable(path);
    let mut cmd = Command::new(resolved);
    cmd.args(["acp", "--agent", LABEL_AGENT_NAME, "--cli_dir_name", LABEL_CLI_DIR, "--log-file"])
        .arg(&iso.log_file)
        .args(["--sandbox", "--include-default-customizations=false", "--disable-telemetry"])
        .current_dir(&iso.work);
    cmd
}

/// Spawns `agy acp`, drives one ACP v1 labeling turn, and cleans up the entire child tree and reader threads.
pub(super) fn run_acp(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
) -> Result<AgentReply, AgentError> {
    run_acp_checked(path, req, timeout, None).0
}

/// Same as [`run_acp`], returning `(result, pipes_closed)` for deterministic descendant-cleanup tests.
pub(super) fn run_acp_checked(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
    home: Option<&Path>,
) -> (Result<AgentReply, AgentError>, bool) {
    if req.cancel.load(Ordering::Relaxed) {
        return (Err(AgentError::Cancelled(LabelUsage::default())), true);
    }
    let iso = match prepare_isolation(req.scratch, home) {
        Ok(iso) => iso,
        Err(e) => return (Err(e), true),
    };

    let mut proc = match AcpProcess::spawn(path, &iso) {
        Ok(p) => p,
        Err(e) => return (Err(e), true),
    };
    let stdin = proc.child.stdin.take().expect("stdin is piped");
    let stdout = proc.child.stdout.take().expect("stdout is piped");
    let stderr = proc.child.stderr.take().expect("stderr is piped");

    // Dedicated stdin writer thread so a wedged child never blocks deadline/cancel checks on large image batches.
    let (in_tx, in_rx) = mpsc::channel::<Vec<u8>>();
    let _in_thread = std::thread::spawn(move || {
        let mut stdin = stdin;
        while let Ok(bytes) = in_rx.recv() {
            if stdin.write_all(&bytes).is_err() || stdin.flush().is_err() {
                break;
            }
        }
    });

    // Dedicated stdout line reader thread.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let stdout_ended = Arc::new(AtomicBool::new(false));
    let stdout_done = stdout_ended.clone();
    let out_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if out_tx.send(line).is_err() {
                break;
            }
        }
        stdout_done.store(true, Ordering::Release);
    });

    // Dedicated bounded stderr tail reader thread.
    let stderr_tail = StderrTail::new();
    let stderr_out = stderr_tail.clone();
    let err_thread = std::thread::spawn(move || {
        let mut stderr = stderr;
        let mut chunk = [0u8; 8 * 1024];
        while let Ok(n @ 1..) = stderr.read(&mut chunk) {
            stderr_out.push(&chunk[..n]);
        }
        stderr_out.mark_ended();
    });

    let mut send = |msg: &Value| -> bool {
        let mut bytes = serde_json::to_vec(msg).expect("JSON values serialize");
        bytes.push(b'\n');
        in_tx.send(bytes).is_ok()
    };
    let read_stderr = || stderr_tail.drain_text(EXIT_DRAIN);
    let mut parent_exited = || proc.poll_exited();

    let outcome = drive_acp_inner(
        req,
        &iso.work,
        timeout,
        CANCEL_GRACE,
        &mut send,
        &out_rx,
        &read_stderr,
        &mut parent_exited,
    );

    // Close stdin; kill immediately on cancel, timeout, or policy violation, otherwise allow a brief clean exit
    // before always killing the process group / job so no descendant outlives the call.
    drop(in_tx);
    if outcome.kill_immediately {
        proc.kill();
    } else {
        proc.finish_gracefully(&stdout_ended, &stderr_tail.ended);
    }
    let out_closed = join_bounded(out_thread, &stdout_ended, EXIT_DRAIN);
    let err_closed = join_bounded(err_thread, &stderr_tail.ended, EXIT_DRAIN);
    (outcome.result, out_closed && err_closed)
}

fn join_bounded(handle: JoinHandle<()>, ended: &AtomicBool, wait: Duration) -> bool {
    let deadline = Instant::now() + wait;
    while !ended.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    if ended.load(Ordering::Acquire) {
        let _ = handle.join();
        true
    } else {
        false
    }
}

pub(super) fn unexpected_exit(usage: &UsageTracker, stderr_text: &dyn Fn() -> String) -> AgentError {
    if let Some(why) = last_line(&stderr_text()) {
        return AgentError::Failed(Failure { used: usage.to_usage(0), ..Failure::new(why) });
    }
    if usage.seen() {
        return AgentError::Failed(Failure {
            used: usage.to_usage(0),
            ..Failure::new("`agy` exited before replying")
        });
    }
    AgentError::Unavailable("`agy` exited before replying".into())
}
