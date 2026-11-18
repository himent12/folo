//! The [`Agent`] trait (one labeling call to a coding-agent CLI) and what its implementations share: finding the
//! CLI on this machine and running it as a child process with a deadline and a cancel switch.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_api::{LabelAgent, LabelAgentId, LabelEffort, LabelUsage};

use super::tree::Tree;

/// How long one agent call may take.
pub const CALL_TIMEOUT: Duration = Duration::from_secs(300);
/// How long a CLI may take to answer a question about itself (`--version`, its model list).
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a probe's result is reused.
const PROBE_TTL: Duration = Duration::from_secs(60);
/// How often a running child is checked for its exit, the deadline and the cancel switch.
const POLL: Duration = Duration::from_millis(40);
/// How long a child's output pipes may stay open after it exited (a process it started still holds them) before
/// what has been read is taken as its output.
const PIPE_GRACE: Duration = Duration::from_secs(2);

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
    /// Ask for the provider's faster service tier, which uses more of the account's usage
    /// ([`LabelAgent::fast`]); agents without one ignore it.
    pub fast: bool,
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
    /// The provider failed the call twice on a passing fault ([`is_transient`]: overloaded, a server error, a
    /// dropped stream): its message. Only [`label_with_retry`] returns it; the next call may well go through.
    Transient(String),
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
    /// The longest side, in pixels, of an image as this agent is sent it ([`super::agent_jpeg`]): what the
    /// provider bills per image grows with it, so it is as small as labels its images about as well.
    fn image_side(&self) -> u32 {
        super::run::AGENT_IMAGE_SIDE
    }
    /// Labels one batch: one model request, no tools. Blocks until the reply, the deadline or `cancel`.
    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError>;
    /// What to do about a failure with this `message` ([`AgentError::Failed`] / [`AgentError::Unavailable`]),
    /// when the agent can tell (sign in, pick another model, ...).
    fn hint(&self, message: &str) -> Option<String> {
        let _ = message;
        None
    }
}

/// Whether a CLI's failure `message` is a passing fault on the provider's side that the same call can get past a
/// little later: the provider is overloaded (Anthropic's 529, OpenAI's "high demand"), answered with a server
/// error (500, 502, 503, 504), or the connection or the stream broke off. Both CLIs retry such faults themselves
/// before they give up, so one that reaches FOLO has lasted a while. A rate limit or a usage limit never is one:
/// asking again only spends more of a limit that has run out.
pub fn is_transient(message: &str) -> bool {
    let m = message.to_lowercase();
    const LIMITS: [&str; 9] = [
        "rate limit",
        "rate_limit",
        "usage limit",
        "session limit",
        "spend",
        "quota",
        "credits",
        "too many requests",
        "hit your",
    ];
    const PASSING: [&str; 18] = [
        "overloaded",
        "high demand",
        "high load",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "server_error",
        "api_error",
        "stream disconnected",
        "stream error",
        "connection error",
        "connection reset",
        "connection closed",
        "econnreset",
        "socket hang up",
        "temporarily unavailable",
        "request timed out",
    ];
    let numbers = || m.split(|c: char| !c.is_ascii_digit());
    if LIMITS.iter().any(|w| m.contains(w)) || numbers().any(|n| n == "429") {
        return false;
    }
    numbers().any(|n| matches!(n, "500" | "502" | "503" | "504" | "529"))
        || PASSING.iter().any(|w| m.contains(w))
}

/// When a usage limit that `message` reports resets, in the CLI's own words: Claude Code's "resets 3pm
/// (Europe/Berlin)", Codex's "try again at 3:04 PM", or the seconds since 1970 after a `|` that older Claude Code
/// versions print (as an RFC 3339 UTC time).
pub fn reset_time(message: &str) -> Option<String> {
    // ASCII lowercase keeps every byte offset of `message`.
    let lower = message.to_ascii_lowercase();
    for marker in ["resets ", "try again at ", "try again in "] {
        if let Some(at) = lower.find(marker) {
            let rest = &message[at + marker.len()..];
            // To the end of the clause: a sentence's end, a `·` or `|` separator, a line break.
            let end = rest
                .char_indices()
                .find(|&(i, c)| {
                    matches!(c, '·' | '|' | '\n')
                        || (c == '.' && rest[i + 1..].starts_with(char::is_whitespace))
                })
                .map_or(rest.len(), |(i, _)| i);
            let when = rest[..end].trim().trim_end_matches('.');
            if !when.is_empty() {
                return Some(when.to_string());
            }
        }
    }
    let (_, epoch) = message.rsplit_once('|')?;
    let secs: u64 = epoch.trim().parse().ok().filter(|&s| s > 1_000_000_000)?;
    Some(crate::util::rfc3339(secs as f64))
}

/// The hint of a run that ended on a usage or rate limit, with when it resets if the CLI said.
pub fn limit_hint(message: &str) -> String {
    match reset_time(message) {
        Some(at) => format!(
            "the account's usage limit was reached and resets {at}: start again then, or use another agent"
        ),
        None => {
            "the account's usage limit was reached; start again once it resets, or use another agent".into()
        }
    }
}

/// The hint of a run that ended on passing faults ([`is_transient`]).
pub const BUSY: &str = "the provider is overloaded or failing right now (each call was asked twice); start again \
                        in a few minutes, or use another agent";

/// [`Agent::label`], asked once more after `backoff` when the call failed on a passing fault ([`is_transient`]);
/// `on_retry` is told the fault first. A second passing fault is returned as [`AgentError::Transient`], any other
/// outcome as it is. The wait ends early, as [`AgentError::Cancelled`], when `req.cancel` is set.
pub fn label_with_retry(
    agent: &dyn Agent,
    req: &AgentRequest<'_>,
    backoff: Duration,
    on_retry: impl FnOnce(&str),
) -> Result<AgentReply, AgentError> {
    match agent.label(req) {
        Err(AgentError::Failed(why)) if is_transient(&why) => {
            on_retry(&why);
            let until = Instant::now() + backoff;
            while Instant::now() < until {
                if req.cancel.load(Ordering::Relaxed) {
                    return Err(AgentError::Cancelled);
                }
                std::thread::sleep(POLL.min(until.saturating_duration_since(Instant::now())));
            }
            match agent.label(req) {
                Err(AgentError::Failed(why)) if is_transient(&why) => Err(AgentError::Transient(why)),
                other => other,
            }
        }
        other => other,
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

/// What a pipe has delivered so far, read on a thread of its own.
#[derive(Default)]
struct Piped {
    bytes: Mutex<Vec<u8>>,
    /// The pipe reached its end: every process that held it is gone.
    ended: AtomicBool,
}

impl Piped {
    fn read(mut pipe: impl Read + Send + 'static) -> Arc<Piped> {
        let piped = Arc::new(Piped::default());
        let out = piped.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 16 * 1024];
            while let Ok(n @ 1..) = pipe.read(&mut chunk) {
                out.bytes.lock().unwrap().extend_from_slice(&chunk[..n]);
            }
            out.ended.store(true, Ordering::Release);
        });
        piped
    }

    fn ended(&self) -> bool {
        self.ended.load(Ordering::Acquire)
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes.lock().unwrap()).into_owned()
    }
}

/// Runs `cmd` to its end, feeding it `stdin` (then closing it), and returns what it printed. Nothing is started
/// once `cancel` is set, and the child and the processes it started ([`Tree`]) are killed when `timeout` passes
/// or `cancel` is set. Input and output move on threads of their own, so neither side can block the other on a
/// full pipe, and none of them is waited for without a limit: a process the child left behind may hold its pipes
/// open, but not this call (after [`PIPE_GRACE`], what was read is the output, and that process is killed too).
pub fn run(
    cmd: Command,
    stdin: Option<Vec<u8>>,
    timeout: Duration,
    cancel: &AtomicBool,
) -> Result<Output, AgentError> {
    let ended = run_to_end(cmd, stdin, timeout, cancel, PIPE_GRACE)?;
    Ok(Output { success: ended.outcome?, stdout: ended.stdout.text(), stderr: ended.stderr.text() })
}

/// A child that has ended, or was ended: whether it succeeded, and its pipes.
struct Ended {
    outcome: Result<bool, AgentError>,
    stdout: Arc<Piped>,
    stderr: Arc<Piped>,
}

/// [`run`], up to the point where the child (and what it started, if that had to be killed) is gone. `grace` is
/// [`PIPE_GRACE`] (tests shorten it).
fn run_to_end(
    mut cmd: Command,
    stdin: Option<Vec<u8>>,
    timeout: Duration,
    cancel: &AtomicBool,
    grace: Duration,
) -> Result<Ended, AgentError> {
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    Tree::prepare(&mut cmd);
    if cancel.load(Ordering::Relaxed) {
        return Err(AgentError::Cancelled);
    }
    let program = cmd.get_program().to_string_lossy().into_owned();
    let unavailable = |e: std::io::Error| AgentError::Unavailable(format!("starting `{program}`: {e}"));
    let mut child = cmd.spawn().map_err(unavailable)?;
    let tree = Tree::start(&child).map_err(|e| {
        let _ = child.kill();
        let _ = child.wait();
        unavailable(e)
    })?;
    if let Some(bytes) = stdin {
        let mut pipe = child.stdin.take().expect("stdin is piped");
        // A child that exits early closes the pipe; its exit status tells the story, not the write error.
        std::thread::spawn(move || drop(pipe.write_all(&bytes)));
    }
    let stdout = Piped::read(child.stdout.take().expect("stdout is piped"));
    let stderr = Piped::read(child.stderr.take().expect("stderr is piped"));
    let deadline = Instant::now() + timeout;
    // The child's exit status and when it was seen.
    let mut exited: Option<(bool, Instant)> = None;
    let outcome = loop {
        if exited.is_none() {
            match child.try_wait() {
                Ok(Some(status)) => exited = Some((status.success(), Instant::now())),
                Ok(None) => {}
                Err(e) => break Err(AgentError::Unavailable(format!("waiting for `{program}`: {e}"))),
            }
        }
        match exited {
            Some((success, at)) if (stdout.ended() && stderr.ended()) || at.elapsed() >= grace => {
                break Ok(success);
            }
            _ if cancel.load(Ordering::Relaxed) => break Err(AgentError::Cancelled),
            _ if Instant::now() >= deadline => break Err(AgentError::Timeout),
            _ => std::thread::sleep(POLL),
        }
    };
    if outcome.is_err() || !(stdout.ended() && stderr.ended()) {
        // Whatever still runs (the child, or what it started and left holding a pipe) goes.
        tree.kill(&mut child);
    }
    Ok(Ended { outcome, stdout, stderr })
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
        assert!(t0.elapsed() < Duration::from_secs(10), "the child was not killed");
        // Cancelled while it runs: killed. Cancelled before it starts: never started (the program is not even
        // looked for).
        let cancel = Arc::new(AtomicBool::new(false));
        let switch = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            switch.store(true, Ordering::Relaxed);
        });
        let r = run(shell(sleep), None, PROBE_TIMEOUT, &cancel);
        assert_eq!(r.err(), Some(AgentError::Cancelled));
        let r = run(Command::new("folo-no-such-agent-cli"), None, PROBE_TIMEOUT, &cancel);
        assert_eq!(r.err(), Some(AgentError::Cancelled));
        assert!(t0.elapsed() < Duration::from_secs(10), "the child was not killed");
    }

    #[test]
    fn a_process_left_holding_the_pipes_neither_holds_the_call_nor_outlives_it() {
        // A shell that starts a process in the background, prints and exits. The process lives for two minutes
        // and has the shell's output pipes, which stay open for as long as anything holds them: their end is
        // the proof that nothing of the call is left.
        const LEFT_BEHIND_S: u64 = 120;
        let script = if cfg!(windows) {
            format!("start /B ping -n {LEFT_BEHIND_S} 127.0.0.1 > NUL & echo done")
        } else {
            format!("sleep {LEFT_BEHIND_S} & echo done")
        };
        let t0 = Instant::now();
        let grace = Duration::from_millis(500);
        let ended = run_to_end(shell(&script), None, Duration::from_secs(60), &AtomicBool::new(false), grace)
            .unwrap();
        // The call is back long before that process would have ended, with what the shell printed...
        assert_eq!(ended.outcome, Ok(true));
        assert_eq!(ended.stdout.text().trim(), "done");
        assert!(t0.elapsed() < Duration::from_secs(30), "the call waited {:?} for the pipes", t0.elapsed());
        // ... and the process went with it: its pipes reach their end.
        while !(ended.stdout.ended() && ended.stderr.ended()) {
            assert!(t0.elapsed() < Duration::from_secs(60), "the process left behind was not killed");
            std::thread::sleep(POLL);
        }
    }

    #[test]
    fn passing_faults_are_told_from_limits_and_other_errors() {
        // What the CLIs print once their own retries are spent (Claude Code, then Codex).
        for transient in [
            r#"API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            r#"API Error: 500 {"type":"error","error":{"type":"api_error","message":"Internal server error"}}"#,
            "Repeated 529 Overloaded errors",
            "API Error: Connection error.",
            "Opus is experiencing high load, please use /model to switch to Sonnet",
            "exceeded retry limit, last status: 503 Service Unavailable",
            "stream disconnected before completion: error sending request for url",
            "We're currently experiencing high demand, which may cause temporary errors.",
            "unexpected status 502 Bad Gateway",
        ] {
            assert!(is_transient(transient), "{transient}");
        }
        for not in [
            "You've hit your session limit · resets 3pm (Europe/Berlin)",
            "Claude AI usage limit reached|1763481600",
            "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or try again at 3:04 PM.",
            "exceeded retry limit, last status: 429 Too Many Requests",
            "rate limit exceeded: 500 requests per minute",
            "Invalid API key · Please run /login",
            "There's an issue with the selected model (nope). It may not exist or you may not have access to it.",
            "The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account.",
            "the image is 15000 px wide",
        ] {
            assert!(!is_transient(not), "{not}");
        }
    }

    #[test]
    fn reset_times_are_read_in_the_clis_words() {
        let reset = |m: &str| reset_time(m);
        assert_eq!(
            reset("You've hit your session limit · resets 3pm (Europe/Berlin)").as_deref(),
            Some("3pm (Europe/Berlin)")
        );
        assert_eq!(
            reset("You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or try again at 3:04 PM.")
                .as_deref(),
            Some("3:04 PM")
        );
        assert_eq!(
            reset("Usage limit reached. Try again in 2 hours 5 minutes. Thanks").as_deref(),
            Some("2 hours 5 minutes")
        );
        assert_eq!(
            reset("Claude AI usage limit reached|1763481600").as_deref(),
            Some("2025-11-18T16:00:00Z")
        );
        assert_eq!(reset("You've hit your usage limit. Try again later."), None);
        assert_eq!(reset("rate limit exceeded"), None);
        assert!(limit_hint("rate limit exceeded").contains("once it resets"));
        assert!(limit_hint("limit · resets 5am").contains("resets 5am: start again then"));
    }

    /// An agent that fails with the messages it is given, in turn, then replies.
    struct Flaky(Mutex<Vec<&'static str>>, std::sync::atomic::AtomicUsize);

    impl Agent for Flaky {
        fn id(&self) -> LabelAgentId {
            LabelAgentId::Claude
        }
        fn info(&self) -> LabelAgent {
            unreachable!()
        }
        fn default_batch(&self) -> usize {
            1
        }
        fn label(&self, _: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
            self.1.fetch_add(1, Ordering::SeqCst);
            match self.0.lock().unwrap().pop() {
                Some(why) => Err(AgentError::Failed(why.into())),
                None => Ok(AgentReply { text: "{}".into(), usage: LabelUsage::default() }),
            }
        }
    }

    #[test]
    fn a_passing_fault_is_asked_once_more_and_no_more() {
        let schema = serde_json::json!({});
        let cancel = AtomicBool::new(false);
        let req = AgentRequest {
            system: "",
            prompt: "",
            images: &[],
            schema: &schema,
            model: None,
            effort: LabelEffort::Low,
            fast: false,
            scratch: Path::new("."),
            cancel: &cancel,
        };
        let wait = Duration::from_millis(30);
        let ask = |fails: Vec<&'static str>| {
            let agent = Flaky(Mutex::new(fails), Default::default());
            let mut heard = Vec::new();
            let t0 = Instant::now();
            let r = label_with_retry(&agent, &req, wait, |why| heard.push(why.to_string()));
            (r, agent.1.load(Ordering::SeqCst), heard, t0.elapsed())
        };
        // Overloaded once: asked again after the wait, and the reply comes back.
        let (r, calls, heard, took) = ask(vec!["API Error: 529 Overloaded"]);
        assert!(r.is_ok() && calls == 2 && heard == ["API Error: 529 Overloaded"] && took >= wait);
        // Overloaded twice: a transient failure, not a third call. (The list pops from its end.)
        let (r, calls, _, _) = ask(vec!["503 again", "API Error: 529 Overloaded"]);
        assert_eq!((r, calls), (Err(AgentError::Transient("503 again".into())), 2));
        // Overloaded, then a real error: that error.
        let (r, calls, _, _) = ask(vec!["Please run /login", "API Error: 529 Overloaded"]);
        assert_eq!((r, calls), (Err(AgentError::Failed("Please run /login".into())), 2));
        // A limit is never asked again.
        let (r, calls, heard, _) = ask(vec!["You've hit your session limit"]);
        assert_eq!(
            (r, calls, heard.len()),
            (Err(AgentError::Failed("You've hit your session limit".into())), 1, 0)
        );
        // Cancelled while it waits: no second call.
        cancel.store(true, Ordering::SeqCst);
        let (r, calls, _, took) = ask(vec!["API Error: 529 Overloaded"]);
        assert_eq!((r, calls), (Err(AgentError::Cancelled), 1));
        assert!(took < Duration::from_secs(5));
    }

    #[test]
    fn a_missing_program_is_unavailable() {
        let r = run(Command::new("folo-no-such-agent-cli"), None, PROBE_TIMEOUT, &AtomicBool::new(false));
        assert!(matches!(r.err(), Some(AgentError::Unavailable(m)) if m.contains("folo-no-such-agent-cli")));
        let cli = Cli::new("folo-no-such-agent-cli", "FOLO_TEST_NO_SUCH_AGENT_BIN");
        assert!(cli.version().is_none() && cli.path().is_err());
    }
}
