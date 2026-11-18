//! One batch's calls ([`ask`]): at most [`MAX_CALLS`] model calls, whatever made the first one fail (a passing
//! fault of the provider or a reply that cannot be read), and every call's usage counted whatever the outcome.
//! Labeling runs, benchmark agent entries, plan design and the eval all ask this way.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use folo_api::LabelUsage;

use super::agent::{Agent, AgentError, AgentRequest, Failure, POLL};
use super::faults::{fault_hint, retry_after};

/// The most model calls one batch may make, all reasons together: a passing fault and an unreadable reply share
/// the one retry.
pub const MAX_CALLS: u32 = 2;

/// What [`ask`] tells its caller as it goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event<'a> {
    /// A reply could not be read: why (every such reply, the last one too).
    Unreadable(&'a str),
    /// The batch is asked once more: why, and the wait first (`Some` after a passing fault).
    Again { why: &'a str, wait: Option<Duration> },
}

/// Why a batch got no answer.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchError {
    /// The cancel switch was set (during a call or the wait before the next).
    Cancelled,
    /// A call did not answer within its deadline.
    Timeout,
    /// The CLI could not be started, or said something FOLO does not understand.
    Unavailable(String),
    /// A failure another call would only repeat (not signed in, an unknown model, a limit): the run ends.
    Failed(Failure),
    /// A passing fault on the last call the batch may make: its images fail, the next batch may go through.
    Transient(Failure),
    /// The last reply the batch may get could not be read: why.
    Unreadable(String),
}

/// Asks `agent` about one batch: each reply is read with `read`, and a passing fault ([`Failure::is_transient`])
/// or a reply `read` refuses is asked once more, after `backoff` (or the wait a passing fault asks for, if longer:
/// at most [`super::faults::SHORT_RATE_LIMIT`]) and no wait after an unreadable reply. Never more than
/// [`MAX_CALLS`] calls. Everything the calls report using is added to `used`, whatever the outcome; `on` hears
/// each [`Event`]. A set `req.cancel` ends it before the next call and during the wait.
pub fn ask<T>(
    agent: &dyn Agent,
    req: &AgentRequest<'_>,
    backoff: Duration,
    used: &mut LabelUsage,
    mut read: impl FnMut(&str) -> Result<T, String>,
    mut on: impl FnMut(Event<'_>),
) -> Result<T, BatchError> {
    let cancelled = || req.cancel.load(Ordering::Relaxed);
    let mut calls = 0;
    loop {
        if cancelled() {
            return Err(BatchError::Cancelled);
        }
        calls += 1;
        let last = calls >= MAX_CALLS;
        match agent.label(req) {
            Ok(reply) => {
                used.add(&reply.usage);
                match read(&reply.text) {
                    Ok(answer) => return Ok(answer),
                    Err(why) => {
                        on(Event::Unreadable(&why));
                        // The caller may have given up on the run after this reply.
                        if last || cancelled() {
                            return Err(BatchError::Unreadable(why));
                        }
                        on(Event::Again { why: &why, wait: None });
                    }
                }
            }
            Err(AgentError::Failed(failure)) => {
                used.add(&failure.used);
                if !failure.is_transient() {
                    return Err(BatchError::Failed(failure));
                }
                if last {
                    return Err(BatchError::Transient(failure));
                }
                let wait = retry_after(&failure.message).map_or(backoff, |asked| asked.max(backoff));
                on(Event::Again { why: &failure.shown(), wait: Some(wait) });
                let until = Instant::now() + wait;
                while Instant::now() < until {
                    if cancelled() {
                        return Err(BatchError::Cancelled);
                    }
                    std::thread::sleep(POLL.min(until.saturating_duration_since(Instant::now())));
                }
            }
            Err(AgentError::Timeout(spent)) => {
                used.add(&spent);
                return Err(BatchError::Timeout);
            }
            Err(AgentError::Cancelled(spent)) => {
                used.add(&spent);
                return Err(BatchError::Cancelled);
            }
            Err(AgentError::Unavailable(why)) => return Err(BatchError::Unavailable(why)),
        }
    }
}

/// What to do about `failure`: the hint its own status gives (a limit, a passing fault), else the agent's
/// ([`Agent::hint`], which reads the message). With a status of its own the failure is no limit or passing fault
/// unless that status says so, so the agent's hint does not count when it is the one the text alone would give
/// ([`fault_hint`]: a 400 whose text names a 503 is not "overloaded").
pub fn failure_hint(agent: &dyn Agent, failure: &Failure) -> Option<String> {
    let message = &failure.message;
    match failure.status {
        Some(status) => fault_hint(message, Some(status))
            .or_else(|| agent.hint(message).filter(|hint| fault_hint(message, None).as_ref() != Some(hint))),
        None => agent.hint(message),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use folo_api::{LabelAgent, LabelAgentId, LabelEffort};

    use super::*;
    use crate::label::agent::AgentReply;
    use crate::label::faults::BUSY;

    /// One call of a scripted agent.
    enum Step {
        Fail(&'static str),
        Reply(&'static str),
        Timeout,
    }

    /// An agent that takes its steps in turn (then replies "{}"); each call reports 100 input tokens.
    struct Scripted(Mutex<Vec<Step>>, AtomicUsize);

    fn scripted(mut steps: Vec<Step>) -> Scripted {
        steps.reverse();
        Scripted(Mutex::new(steps), AtomicUsize::new(0))
    }

    impl Agent for Scripted {
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
            let usage = LabelUsage { calls: 1, input_tokens: 100, ..LabelUsage::default() };
            match self.0.lock().unwrap().pop() {
                Some(Step::Fail(why)) => {
                    Err(AgentError::Failed(Failure { used: usage, ..Failure::new(why) }))
                }
                Some(Step::Timeout) => Err(AgentError::Timeout(usage)),
                Some(Step::Reply(text)) => Ok(AgentReply { text: text.into(), usage }),
                None => Ok(AgentReply { text: "{}".into(), usage }),
            }
        }
    }

    fn used(calls: u64) -> LabelUsage {
        LabelUsage { calls, input_tokens: 100 * calls, ..LabelUsage::default() }
    }

    const WAIT: Duration = Duration::from_millis(30);
    const OVERLOADED: &str = "API Error: 529 Overloaded";

    /// What asking `agent` came to: the outcome, the calls made, what they used, the events and the time taken.
    struct Asked {
        result: Result<String, BatchError>,
        calls: usize,
        used: LabelUsage,
        events: Vec<String>,
        took: Duration,
    }

    /// Asks `agent`; with `stop_on_again`, the run is stopped as the first call is asked again (during its wait).
    fn ask_with(agent: &Scripted, stop_on_again: bool) -> Asked {
        let cancel = &AtomicBool::new(false);
        let schema = serde_json::json!({});
        let req = AgentRequest {
            system: "",
            prompt: "",
            images: &[],
            schema: &schema,
            model: None,
            effort: LabelEffort::Low,
            fast: false,
            scratch: Path::new("."),
            cancel,
        };
        let (mut used, mut events) = (LabelUsage::default(), Vec::new());
        let t0 = Instant::now();
        // A reply reads when it is "{}".
        let read =
            |text: &str| if text == "{}" { Ok(text.to_string()) } else { Err(format!("unreadable: {text}")) };
        let on = |e: Event<'_>| {
            if stop_on_again && matches!(e, Event::Again { .. }) {
                cancel.store(true, Ordering::SeqCst);
            }
            events.push(format!("{e:?}"));
        };
        let result = ask(agent, &req, WAIT, &mut used, read, on);
        Asked { result, calls: agent.1.load(Ordering::SeqCst), used, events, took: t0.elapsed() }
    }

    fn ask_steps(steps: Vec<Step>) -> Asked {
        ask_with(&scripted(steps), false)
    }

    #[test]
    fn a_passing_fault_is_asked_once_more_and_no_more() {
        // Overloaded once: asked again after the wait, and both calls are counted.
        let a = ask_steps(vec![Step::Fail(OVERLOADED)]);
        assert_eq!((a.result, a.calls, a.used), (Ok("{}".into()), 2, used(2)));
        assert!(a.took >= WAIT && a.events.len() == 1 && a.events[0].contains("Again"), "{:?}", a.events);
        // Overloaded twice: a passing failure, not a third call.
        let a = ask_steps(vec![Step::Fail(OVERLOADED), Step::Fail("status 503 again")]);
        let failure = Failure { used: used(1), ..Failure::new("status 503 again") };
        assert_eq!((a.result, a.calls, a.used), (Err(BatchError::Transient(failure)), 2, used(2)));
        // Overloaded, then a real error: that error.
        let a = ask_steps(vec![Step::Fail(OVERLOADED), Step::Fail("Please run /login")]);
        assert!(matches!(a.result, Err(BatchError::Failed(f)) if f.message == "Please run /login"));
        assert_eq!((a.calls, a.used), (2, used(2)));
        // A limit is never asked again.
        let a = ask_steps(vec![Step::Fail("You've hit your session limit")]);
        assert!(matches!(a.result, Err(BatchError::Failed(_))) && a.calls == 1 && a.events.is_empty());
        // A short rate limit is waited out, for as long as it asks when that is longer than the pause.
        let a = ask_steps(vec![Step::Fail("Rate limit reached. Please try again in 0.2s.")]);
        assert!(a.result.is_ok() && a.calls == 2 && a.took >= Duration::from_millis(200), "{:?}", a.took);
    }

    #[test]
    fn one_retry_serves_every_reason_a_batch_fails() {
        // A passing fault, then garbage, then another passing fault: two calls, not four.
        let a = ask_steps(vec![Step::Fail(OVERLOADED), Step::Reply("garbage"), Step::Fail(OVERLOADED)]);
        assert_eq!(
            (a.result, a.calls, a.used),
            (Err(BatchError::Unreadable("unreadable: garbage".into())), 2, used(2))
        );
        // Garbage, then a passing fault: the fault is the last call's.
        let a = ask_steps(vec![Step::Reply("garbage"), Step::Fail(OVERLOADED), Step::Reply("{}")]);
        assert!(matches!(a.result, Err(BatchError::Transient(_))) && a.calls == 2, "{:?}", a.result);
        assert_eq!(a.events.len(), 2, "unreadable, then asked again without a wait: {:?}", a.events);
        // Garbage twice: unreadable, every reply reported.
        let a = ask_steps(vec![Step::Reply("garbage"), Step::Reply("still garbage")]);
        assert!(matches!(a.result, Err(BatchError::Unreadable(_))) && a.calls == 2);
        assert_eq!(a.events.iter().filter(|e| e.starts_with("Unreadable")).count(), 2);
    }

    #[test]
    fn what_the_calls_used_survives_every_outcome() {
        // The second call times out: the first one's usage stays, and what the second printed before it hung
        // counts too.
        let a = ask_steps(vec![Step::Fail(OVERLOADED), Step::Timeout]);
        assert_eq!((a.result, a.calls, a.used), (Err(BatchError::Timeout), 2, used(2)));
        // Stopped during the wait before the second call (as it is announced, so before any second call).
        let agent = scripted(vec![Step::Fail("Rate limit reached. Please try again in 30s.")]);
        let a = ask_with(&agent, true);
        assert_eq!((a.result, a.calls, a.used), (Err(BatchError::Cancelled), 1, used(1)));
        // It waited for the 30 s the limit asked for, and ended promptly on the stop.
        assert!(a.events[0].contains("30s"), "{:?}", a.events);
        assert!(a.took < Duration::from_secs(5), "{:?}", a.took);
    }

    #[test]
    fn a_reported_status_is_what_counts() {
        struct Status;
        impl Agent for Status {
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
                unreachable!()
            }
            fn hint(&self, _: &str) -> Option<String> {
                Some("the agent's own".into())
            }
        }
        let failure =
            Failure { status: Some(503), ..Failure::new("upstream failed after processing 503 tokens") };
        assert!(failure.is_transient());
        assert_eq!(failure_hint(&Status, &failure).as_deref(), Some(BUSY));
        let failure = Failure { status: None, ..failure };
        assert!(!failure.is_transient());
        assert_eq!(failure_hint(&Status, &failure).as_deref(), Some("the agent's own"));
        // A 400 whose text names a 503: no passing fault, so not the hint the text alone would give, whatever
        // the agent reads into it.
        struct ByText;
        impl Agent for ByText {
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
                unreachable!()
            }
            fn hint(&self, message: &str) -> Option<String> {
                fault_hint(message, None).or_else(|| message.contains("model").then(|| "pick another".into()))
            }
        }
        let failure = Failure { status: Some(400), ..Failure::new("API Error: 503 Service Unavailable") };
        assert!(!failure.is_transient());
        assert_eq!(failure_hint(&ByText, &failure), None);
        assert_eq!(failure_hint(&ByText, &Failure { status: None, ..failure }).as_deref(), Some(BUSY));
        // A hint of the agent's own (not a fault's) stays.
        let failure = Failure { status: Some(404), ..Failure::new("no such model") };
        assert_eq!(failure_hint(&ByText, &failure).as_deref(), Some("pick another"));
        // What is shown is cut short; what is classified is not.
        let long = format!("{} API Error: 529 Overloaded", "x".repeat(5000));
        let failure = Failure::new(long);
        assert!(
            failure.is_transient()
                && failure.shown().chars().count() <= folo_api::codes::MAX_MESSAGE_CHARS + 1
        );
    }
}
