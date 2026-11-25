//! OS subprocess lifecycle, stdio pipe reader/writer threads, and strict process-tree cleanup for `agy acp`.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_api::LabelUsage;
use serde_json::Value;

use super::acp::{CANCEL_GRACE, EXIT_DRAIN, StdoutLine, drive_acp_inner};
use super::protocol::{
    Isolation, LABEL_AGENT_NAME, LABEL_CLI_DIR, UsageTracker, prepare_isolation, resolve_executable,
};
use crate::label::agent::{AgentError, AgentReply, AgentRequest, Failure, last_line};
use crate::label::tree::Tree;

/// How long to wait for `agy acp` to exit cleanly after closing stdin before killing its process tree.
const CLOSE_GRACE: Duration = Duration::from_millis(250);
/// Maximum bytes retained from child stderr (most recent tail).
pub(super) const MAX_STDERR_BYTES: usize = 64 * 1024;
/// Maximum bytes accepted for a single stdout JSON-RPC line before failing closed (16 MiB).
pub(super) const MAX_STDOUT_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Condvar-backed one-shot completion signal so pipe drains and bounded thread joins wake immediately on EOF
/// instead of sleeping in fixed intervals.
pub(super) struct DoneSignal {
    done: Mutex<bool>,
    cv: Condvar,
}

impl DoneSignal {
    pub(super) fn new() -> DoneSignal {
        DoneSignal { done: Mutex::new(false), cv: Condvar::new() }
    }

    pub(super) fn mark(&self) {
        let mut done = self.done.lock().unwrap_or_else(|p| p.into_inner());
        *done = true;
        self.cv.notify_all();
    }

    pub(super) fn is_done(&self) -> bool {
        *self.done.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(super) fn wait_timeout(&self, wait: Duration) -> bool {
        let mut done = self.done.lock().unwrap_or_else(|p| p.into_inner());
        if *done || wait.is_zero() {
            return *done;
        }
        let Some(deadline) = Instant::now().checked_add(wait) else {
            while !*done {
                done = self.cv.wait(done).unwrap_or_else(|p| p.into_inner());
            }
            return true;
        };
        while !*done {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let (guard, _) = self.cv.wait_timeout(done, remaining).unwrap_or_else(|p| p.into_inner());
            done = guard;
        }
        *done
    }
}

/// Bounded stderr tail with EOF tracking so unexpected exits wait briefly for diagnostics without unbounded memory.
pub(super) struct StderrTail {
    bytes: Mutex<VecDeque<u8>>,
    done: DoneSignal,
}

impl StderrTail {
    pub(super) fn new() -> Arc<StderrTail> {
        Arc::new(StderrTail { bytes: Mutex::new(VecDeque::new()), done: DoneSignal::new() })
    }

    pub(super) fn push(&self, chunk: &[u8]) {
        let mut buf = self.bytes.lock().unwrap_or_else(|p| p.into_inner());
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
        self.done.mark();
    }

    pub(super) fn drain_text(&self, wait: Duration) -> String {
        let _ = self.done.wait_timeout(wait);
        let buf = self.bytes.lock().unwrap_or_else(|p| p.into_inner());
        let (a, b) = buf.as_slices();
        let mut all = Vec::with_capacity(a.len() + b.len());
        all.extend_from_slice(a);
        all.extend_from_slice(b);
        String::from_utf8_lossy(&all).into_owned()
    }
}

/// Reads one UTF-8 line (stripping trailing `\n` and optional `\r`) from `reader`, rejecting any single line whose
/// payload exceeds `max_bytes` with [`ErrorKind::InvalidData`] without unbounded allocation.
pub(super) fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    max_bytes: usize,
) -> std::io::Result<Option<String>> {
    let mut buf = Vec::new();
    let mut saw_any = false;
    loop {
        let avail = match reader.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if avail.is_empty() {
            if !saw_any {
                return Ok(None);
            }
            if buf.len() > max_bytes {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("stdout line exceeded {max_bytes} bytes"),
                ));
            }
            break;
        }
        saw_any = true;
        if let Some(pos) = avail.iter().position(|&b| b == b'\n') {
            let chunk = &avail[..pos];
            // Allow one extra byte here for a trailing `\r` right before `\n`, checked after stripping.
            if buf.len().saturating_add(chunk.len()) > max_bytes.saturating_add(1) {
                reader.consume(pos + 1);
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("stdout line exceeded {max_bytes} bytes"),
                ));
            }
            buf.extend_from_slice(chunk);
            reader.consume(pos + 1);
            if buf.last() == Some(&b'\r') {
                buf.pop();
            }
            if buf.len() > max_bytes {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("stdout line exceeded {max_bytes} bytes"),
                ));
            }
            break;
        }
        let n = avail.len();
        let next_len = buf.len().saturating_add(n);
        let allow_trailing_cr = next_len == max_bytes.saturating_add(1) && avail.last() == Some(&b'\r');
        if next_len > max_bytes && !allow_trailing_cr {
            reader.consume(n);
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("stdout line exceeded {max_bytes} bytes"),
            ));
        }
        buf.extend_from_slice(avail);
        reader.consume(n);
    }
    String::from_utf8(buf).map(Some).map_err(|e| std::io::Error::new(ErrorKind::InvalidData, e))
}

/// Streams bounded UTF-8 lines from `reader` into `out_tx`, forwarding framing or I/O errors in stream order so the
/// ACP driver reports the actual cause and kills the child immediately instead of misclassifying EOF.
pub(super) fn read_stdout_loop<R: BufRead>(
    mut reader: R,
    max_bytes: usize,
    marker: Option<&str>,
    saw_marker: &AtomicBool,
    out_tx: &mpsc::Sender<StdoutLine>,
) {
    loop {
        match read_bounded_line(&mut reader, max_bytes) {
            Ok(Some(line)) => {
                if let Some(m) = marker
                    && line.contains(m)
                {
                    saw_marker.store(true, Ordering::Release);
                }
                if out_tx.send(StdoutLine::Line(line)).is_err() {
                    break;
                }
            }
            Ok(None) => break,
            Err(e) => {
                let msg = if e.kind() == ErrorKind::InvalidData {
                    format!("`agy acp` stdout framing error: {e}")
                } else {
                    format!("reading `agy acp` stdout failed: {e}")
                };
                let _ = out_tx.send(StdoutLine::Error(msg));
                break;
            }
        }
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

    fn finish_gracefully(&mut self, stdout_done: &DoneSignal, stderr_done: &DoneSignal) {
        let deadline = Instant::now() + CLOSE_GRACE;
        while Instant::now() < deadline {
            if self.poll_exited() && stdout_done.is_done() && stderr_done.is_done() {
                break;
            }
            let step = deadline.saturating_duration_since(Instant::now()).min(Duration::from_millis(10));
            if !stdout_done.is_done() {
                let _ = stdout_done.wait_timeout(step);
            } else if !stderr_done.is_done() {
                let _ = stderr_done.wait_timeout(step);
            } else {
                std::thread::sleep(step);
            }
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

/// Diagnostics returned by [`run_acp_checked`] for deterministic subprocess-cleanup tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct AcpRunCheck {
    pub result: Result<AgentReply, AgentError>,
    pub pipes_closed: bool,
    pub saw_stdout_marker: bool,
    pub stderr: String,
}

/// Spawns `agy acp`, drives one ACP v1 labeling turn, and cleans up the entire child tree and reader threads.
pub(super) fn run_acp(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
) -> Result<AgentReply, AgentError> {
    run_acp_checked(path, req, timeout, None, None).result
}

/// Same as [`run_acp`], returning pipe-EOF and stream-marker observations for deterministic cleanup tests.
pub(super) fn run_acp_checked(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
    home: Option<&Path>,
    stdout_marker: Option<&str>,
) -> AcpRunCheck {
    let early = |err: AgentError| AcpRunCheck {
        result: Err(err),
        pipes_closed: true,
        saw_stdout_marker: false,
        stderr: String::new(),
    };
    if req.cancel.load(Ordering::Relaxed) {
        return early(AgentError::Cancelled(LabelUsage::default()));
    }
    let iso = match prepare_isolation(req.scratch, home) {
        Ok(iso) => iso,
        Err(e) => return early(e),
    };

    let mut proc = match AcpProcess::spawn(path, &iso) {
        Ok(p) => p,
        Err(e) => return early(e),
    };
    let stdin = proc.child.stdin.take().expect("stdin is piped");
    let stdout = proc.child.stdout.take().expect("stdout is piped");
    let stderr = proc.child.stderr.take().expect("stderr is piped");

    // Dedicated stdin writer thread so a wedged child never blocks deadline/cancel checks on large image batches.
    let (in_tx, in_rx) = mpsc::channel::<Vec<u8>>();
    let stdin_done = Arc::new(DoneSignal::new());
    let stdin_done_out = stdin_done.clone();
    let in_thread = std::thread::spawn(move || {
        let mut stdin = stdin;
        while let Ok(bytes) = in_rx.recv() {
            if stdin.write_all(&bytes).is_err() || stdin.flush().is_err() {
                break;
            }
        }
        drop(stdin);
        stdin_done_out.mark();
    });

    // Dedicated bounded stdout line reader thread.
    let (out_tx, out_rx) = mpsc::channel::<StdoutLine>();
    let stdout_done = Arc::new(DoneSignal::new());
    let stdout_done_out = stdout_done.clone();
    let saw_marker = Arc::new(AtomicBool::new(false));
    let saw_marker_out = saw_marker.clone();
    let marker = stdout_marker.map(str::to_string);
    let out_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        read_stdout_loop(reader, MAX_STDOUT_LINE_BYTES, marker.as_deref(), &saw_marker_out, &out_tx);
        drop(out_tx);
        stdout_done_out.mark();
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
        drop(stderr);
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
        proc.finish_gracefully(&stdout_done, &stderr_tail.done);
    }
    let in_closed = join_bounded(in_thread, &stdin_done, EXIT_DRAIN);
    let out_closed = join_bounded(out_thread, &stdout_done, EXIT_DRAIN);
    let err_closed = join_bounded(err_thread, &stderr_tail.done, EXIT_DRAIN);
    AcpRunCheck {
        result: outcome.result,
        pipes_closed: in_closed && out_closed && err_closed,
        saw_stdout_marker: saw_marker.load(Ordering::Acquire),
        stderr: stderr_tail.drain_text(Duration::ZERO),
    }
}

pub(super) fn join_bounded(handle: JoinHandle<()>, done: &DoneSignal, wait: Duration) -> bool {
    if done.wait_timeout(wait) {
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
