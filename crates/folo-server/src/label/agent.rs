//! The [`Agent`] trait (one labeling call to a coding-agent CLI) and what its implementations share: finding the
//! CLI on this machine and running it as a child process with a deadline and a cancel switch.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use folo_api::{LabelAgent, LabelAgentId, LabelEffort, LabelUsage};

/// How long one agent call may take.
pub const CALL_TIMEOUT: Duration = Duration::from_secs(300);
/// How long a CLI may take to answer a question about itself (`--version`, its model list).
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a probe's result is reused.
const PROBE_TTL: Duration = Duration::from_secs(60);
/// How often a running child is checked for its exit, the deadline and the cancel switch.
const POLL: Duration = Duration::from_millis(40);

/// One call: a prompt and the images it is about.
pub struct AgentRequest<'a> {
    /// The standing instructions ([`super::prompt::SYSTEM`]).
    pub system: &'a str,
    /// The request ([`super::prompt::user_prompt`]).
    pub prompt: &'a str,
    /// The images, JPEG-encoded, in the order the prompt numbers them.
    pub images: &'a [Vec<u8>],
    /// The JSON Schema of the reply ([`super::prompt::reply_schema`]), for agents that enforce one.
    pub schema: &'a serde_json::Value,
    /// The model to ask for (`None`: the CLI's default).
    pub model: Option<&'a str>,
    /// Reasoning effort.
    pub effort: LabelEffort,
    /// An existing folder of this call's own, for files the CLI needs on disk; the caller removes it.
    pub scratch: &'a Path,
    /// Set to end the call early ([`AgentError::Cancelled`]).
    pub cancel: &'a AtomicBool,
}

/// What a call returned.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentReply {
    /// The model's final message.
    pub text: String,
    /// What the call used (`calls` is 1).
    pub usage: LabelUsage,
}

/// Why a call returned nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentError {
    /// The CLI could not be started, or said something FOLO does not understand. Calling again will not help.
    Unavailable(String),
    /// The CLI ran and reported an error (not signed in, an unknown model, a rate limit): its message.
    Failed(String),
    /// No answer within [`CALL_TIMEOUT`]; the process was killed.
    Timeout,
    /// The cancel switch was set; the process was killed.
    Cancelled,
}

/// A coding agent that can label images. Implementations: [`super::claude::ClaudeCode`], [`super::codex::Codex`];
/// tests script their own.
pub trait Agent: Send + Sync + 'static {
    /// Which agent this is.
    fn id(&self) -> LabelAgentId;
    /// Whether it can be used on this machine, with its models. May run the CLI's `--version` (cached).
    fn info(&self) -> LabelAgent;
    /// Images per call when a run names no batch size: larger for an agent with a larger fixed cost per call.
    fn default_batch(&self) -> usize;
    /// Labels one batch: one model request, no tools. Blocks until the reply, the deadline or `cancel`.
    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError>;
    /// What to do about a failure with this `message` ([`AgentError::Failed`] / [`AgentError::Unavailable`]),
    /// when the agent can tell (sign in, pick another model, ...).
    fn hint(&self, message: &str) -> Option<String> {
        let _ = message;
        None
    }
}

/// A value that is expensive to find out (it takes a child process) and is remembered for [`PROBE_TTL`].
pub struct Memo<T>(Mutex<Option<(Instant, T)>>);

impl<T: Clone> Memo<T> {
    /// An empty memo.
    pub const fn new() -> Memo<T> {
        Memo(Mutex::new(None))
    }

    /// The remembered value, or `find()`'s when there is none or it is older than [`PROBE_TTL`]. Concurrent
    /// callers wait for one `find`.
    pub fn get(&self, find: impl FnOnce() -> T) -> T {
        let mut slot = self.0.lock().unwrap();
        match &*slot {
            Some((at, value)) if at.elapsed() < PROBE_TTL => value.clone(),
            _ => {
                let value = find();
                *slot = Some((Instant::now(), value.clone()));
                value
            }
        }
    }
}

impl<T: Clone> Default for Memo<T> {
    fn default() -> Self {
        Memo::new()
    }
}

/// Where an agent's CLI is and what version it says it is, probed on demand.
pub struct Cli {
    name: &'static str,
    env: &'static str,
    probed: Memo<Option<Found>>,
}

#[derive(Clone)]
struct Found {
    path: PathBuf,
    version: String,
}

impl Cli {
    /// The CLI called `name` on `PATH`, unless the environment variable `env` names another executable.
    pub const fn new(name: &'static str, env: &'static str) -> Cli {
        Cli { name, env, probed: Memo::new() }
    }

    fn found(&self) -> Option<Found> {
        self.probed.get(|| self.probe())
    }

    fn probe(&self) -> Option<Found> {
        let path = match std::env::var_os(self.env).filter(|v| !v.is_empty()) {
            Some(custom) => PathBuf::from(custom),
            None => which(self.name)?,
        };
        let mut cmd = Command::new(&path);
        cmd.arg("--version");
        let out = run(cmd, None, PROBE_TIMEOUT, &AtomicBool::new(false)).ok()?;
        let version = version_of(&out.stdout)?;
        out.success.then_some(Found { path, version })
    }

    /// The executable to start, or why there is none.
    pub fn path(&self) -> Result<PathBuf, AgentError> {
        match self.found() {
            Some(f) => Ok(f.path),
            None => Err(AgentError::Unavailable(format!("`{}` was not found on this machine", self.name))),
        }
    }

    /// The version `--version` printed (`None`: not installed).
    pub fn version(&self) -> Option<String> {
        self.found().map(|f| f.version)
    }

    /// How to make the CLI available: `install` (a sentence on getting it), or the override variable.
    pub fn hint(&self, install: &str) -> String {
        format!("{install}, or point {} at its executable; then reload", self.env)
    }
}

/// The version number in a `--version` line: the first word that starts with a digit (`2.1.302 (Claude Code)`,
/// `codex-cli 0.167.0`).
fn version_of(stdout: &str) -> Option<String> {
    stdout.split_whitespace().find(|w| w.starts_with(|c: char| c.is_ascii_digit())).map(str::to_string)
}

/// The first `name` (with any of Windows' executable extensions) in a `PATH` folder.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    // `.cmd` covers npm's shims. Elsewhere the bare name is the executable.
    let exts: &[&str] = if cfg!(windows) { &[".exe", ".cmd", ".bat"] } else { &[""] };
    std::env::split_paths(&path)
        .flat_map(|dir| exts.iter().map(move |ext| dir.join(format!("{name}{ext}"))))
        .find(|p| p.is_file())
}

/// What a finished child printed.
pub struct Output {
    /// Whether it exited with status 0.
    pub success: bool,
    /// Its standard output.
    pub stdout: String,
    /// Its standard error.
    pub stderr: String,
}

/// Runs `cmd` to its end, feeding it `stdin` (then closing it), and returns what it printed. The child and the
/// processes it started are killed when `timeout` passes or `cancel` is set. Output is read on threads of its own
/// while the input is written, so neither side can block the other on a full pipe.
pub fn run(
    mut cmd: Command,
    stdin: Option<Vec<u8>>,
    timeout: Duration,
    cancel: &AtomicBool,
) -> Result<Output, AgentError> {
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    no_window(&mut cmd);
    #[cfg(unix)]
    {
        // A group of its own, so the whole tree can be killed (see `kill_tree`).
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let program = cmd.get_program().to_string_lossy().into_owned();
    let mut child = cmd.spawn().map_err(|e| AgentError::Unavailable(format!("starting `{program}`: {e}")))?;
    let writer = stdin.map(|bytes| {
        let mut pipe = child.stdin.take().expect("stdin is piped");
        // A child that exits early closes the pipe; its exit status tells the story, not the write error.
        std::thread::spawn(move || drop(pipe.write_all(&bytes)))
    });
    let reader = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut text = Vec::new();
            let _ = pipe.read_to_end(&mut text);
            String::from_utf8_lossy(&text).into_owned()
        })
    };
    let stdout = reader(Box::new(child.stdout.take().expect("stdout is piped")));
    let stderr = reader(Box::new(child.stderr.take().expect("stderr is piped")));
    let deadline = Instant::now() + timeout;
    let ended = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status.success()),
            Ok(None) if cancel.load(Ordering::Relaxed) => break Err(AgentError::Cancelled),
            Ok(None) if Instant::now() >= deadline => break Err(AgentError::Timeout),
            Ok(None) => std::thread::sleep(POLL),
            Err(e) => break Err(AgentError::Unavailable(format!("waiting for `{program}`: {e}"))),
        }
    };
    let success = match ended {
        Ok(success) => success,
        Err(e) => {
            // The pipe threads end by themselves once every process holding the pipes is gone.
            kill_tree(&mut child);
            return Err(e);
        }
    };
    if let Some(w) = writer {
        let _ = w.join();
    }
    Ok(Output {
        success,
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

/// Keeps a console child from opening a window of its own when the server has none (started by a desktop app).
fn no_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

/// Kills `child` and everything it started: a CLI installed through npm is a shim (a batch file, a node script)
/// around the process that does the work, and killing only the shim would leave that one running.
fn kill_tree(child: &mut std::process::Child) {
    let pid = child.id().to_string();
    let mut killer = if cfg!(windows) {
        let mut c = Command::new("taskkill");
        c.args(["/T", "/F", "/PID", &pid]);
        c
    } else {
        // The child leads a process group of its own (see `run`); a negative pid names the group.
        let mut c = Command::new("kill");
        c.args(["-KILL", "--", &format!("-{pid}")]);
        c
    };
    no_window(&mut killer);
    let _ = killer.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
    let _ = child.kill();
    let _ = child.wait();
}

/// The last non-empty line of `text`, shortened: what a CLI's error output boils down to.
pub fn last_line(text: &str) -> Option<String> {
    let line = text.lines().rev().map(str::trim).find(|l| !l.is_empty())?;
    Some(folo_api::codes::cap(line, folo_api::codes::MAX_MESSAGE_CHARS))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A command that runs `script` in the platform's shell.
    fn shell(script: &str) -> Command {
        let mut cmd = if cfg!(windows) { Command::new("cmd") } else { Command::new("sh") };
        cmd.arg(if cfg!(windows) { "/C" } else { "-c" }).arg(script);
        cmd
    }

    #[test]
    fn versions_are_read_from_either_cli() {
        assert_eq!(version_of("2.1.302 (Claude Code)\n").as_deref(), Some("2.1.302"));
        assert_eq!(version_of("codex-cli 0.167.0\n").as_deref(), Some("0.167.0"));
        assert_eq!(version_of("command not found"), None);
    }

    #[test]
    fn run_returns_output_and_status() {
        let out = run(shell("echo hello"), None, PROBE_TIMEOUT, &AtomicBool::new(false)).unwrap();
        assert!(out.success);
        assert_eq!(out.stdout.trim(), "hello");
        let out =
            run(shell("echo oops 1>&2 && exit 3"), None, PROBE_TIMEOUT, &AtomicBool::new(false)).unwrap();
        assert!(!out.success);
        assert_eq!(last_line(&out.stderr).as_deref(), Some("oops"));
    }

    #[test]
    fn run_feeds_stdin_larger_than_a_pipe_buffer() {
        // More than any pipe buffer in both directions, so the write must overlap the reads.
        let input = "x\n".repeat(1 << 19).into_bytes();
        // Either one copies its input to its output.
        let echo = if cfg!(windows) { shell("findstr x") } else { Command::new("cat") };
        let out = run(echo, Some(input.clone()), PROBE_TIMEOUT, &AtomicBool::new(false)).unwrap();
        assert!(out.success, "{}", out.stderr);
        assert_eq!(out.stdout.replace('\r', "").len(), input.len());
    }

    #[test]
    fn run_kills_on_timeout_and_on_cancel() {
        let sleep = if cfg!(windows) { "ping -n 30 127.0.0.1 > NUL" } else { "sleep 30" };
        let t0 = Instant::now();
        let r = run(shell(sleep), None, Duration::from_millis(200), &AtomicBool::new(false));
        assert_eq!(r.err(), Some(AgentError::Timeout));
        let r = run(shell(sleep), None, PROBE_TIMEOUT, &AtomicBool::new(true));
        assert_eq!(r.err(), Some(AgentError::Cancelled));
        assert!(t0.elapsed() < Duration::from_secs(10), "the children were not killed");
    }

    #[test]
    fn a_missing_program_is_unavailable() {
        let r = run(Command::new("folo-no-such-agent-cli"), None, PROBE_TIMEOUT, &AtomicBool::new(false));
        assert!(matches!(r.err(), Some(AgentError::Unavailable(m)) if m.contains("folo-no-such-agent-cli")));
        let cli = Cli::new("folo-no-such-agent-cli", "FOLO_TEST_NO_SUCH_AGENT_BIN");
        assert!(cli.version().is_none() && cli.path().is_err());
    }
}
