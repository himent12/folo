//! What a CLI's failure says: a passing fault of the provider that the same call can get past a little later
//! ([`is_transient`]), a limit of the account or of the provider ([`limit_hint`], with when it passes in the CLI's
//! own words), or something else. A failure is its message and, when the CLI reported one in a field of its own,
//! the HTTP status: that status is what counts then, never a number in the text. Both agents' hints and the
//! retry of [`super::agent::ask`] read their failures with these.

use std::time::Duration;

/// The longest wait a passing fault may ask for: a rate limit (or any fault) that asks for at most this long is
/// asked once more after the wait, instead of ending a run of thousands of images on a per-minute limit. A fault
/// that asks for longer ends the run like a limit, with when to start again in the hint.
pub const SHORT_RATE_LIMIT: Duration = Duration::from_secs(60);

/// The hint of a run that ended on passing faults ([`is_transient`]). How often each call was asked is the run's
/// to say (its log has every retry), so the hint does not count.
pub const BUSY: &str =
    "the provider is overloaded or failing right now; start again in a few minutes, or use another agent";

/// Words of a limit of the account (a plan's usage, a spending cap): asking again only spends more of a limit
/// that has run out, however soon it says to try again. Each is matched at the start of a word ([`has_word`]), so
/// "spend" is not found in "suspended".
const USAGE_LIMIT: [&str; 11] = [
    "usage limit",
    "session limit",
    "weekly limit",
    "hour limit",
    "hit your",
    "quota",
    "credits",
    "credit balance",
    "spend",
    "billing",
    "out of extra usage",
];
/// Words of the provider's rate limit (requests or tokens per minute), and its status, 429.
const RATE_LIMIT: [&str; 3] = ["rate limit", "rate_limit", "too many requests"];
/// Words of a passing fault: the provider overloaded, a server error, the connection broken off. (Codex's "stream
/// disconnected before completion" is not one of them: it wraps any failed response, a client error too, so it is
/// a passing fault only when the rest of its message says so.)
const PASSING: [&str; 21] = [
    "overloaded",
    "high demand",
    "high load",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "server_error",
    "api_error",
    "api error: terminated",
    "stream error",
    "connection error",
    "connection reset",
    "connection closed",
    "error sending request",
    "econnreset",
    "econnrefused",
    "etimedout",
    "socket hang up",
    "temporarily unavailable",
    "request timed out",
];
/// The server errors that are passing faults (529: Anthropic's overloaded).
const PASSING_STATUS: [u16; 5] = [500, 502, 503, 504, 529];

/// Whether a CLI's failure is a passing fault on the provider's side that the same call can get past a little
/// later: the provider is overloaded (Anthropic's 529, OpenAI's "high demand"), answered with a server error (500,
/// 502, 503, 504), the connection broke off, or a rate limit asks for a short wait. Both CLIs retry such faults
/// themselves before they give up, so one that reaches FOLO has lasted a while. `status` is the HTTP status the CLI
/// reported in a field of its own (it then stands for the statuses in the text).
///
/// In this order: a usage limit never is one; a client error (a 4xx status other than 429) is not, whatever else
/// the message says; nor is a fault that asks for a wait longer than [`SHORT_RATE_LIMIT`]; a rate limit is one when
/// it asks for a wait (a short one, then); anything else when it has a server error's status or words.
pub fn is_transient(message: &str, status: Option<u16>) -> bool {
    let m = message.to_lowercase();
    if is_usage_limit(&m) {
        return false;
    }
    let statuses = status.map_or_else(|| statuses(&m), |s| vec![s]);
    if statuses.iter().any(|&s| (400..500).contains(&s) && s != 429) {
        return false;
    }
    let wait = retry_after(message);
    if wait.is_some_and(|w| w > SHORT_RATE_LIMIT) {
        return false;
    }
    if is_rate_limit(&m, &statuses) {
        return wait.is_some();
    }
    statuses.iter().any(|s| PASSING_STATUS.contains(s)) || PASSING.iter().any(|w| m.contains(w))
}

/// Whether a failure reports a limit: the account's usage, the provider's rate limit, or a fault that asks for a
/// wait longer than [`SHORT_RATE_LIMIT`].
pub fn is_limit(message: &str, status: Option<u16>) -> bool {
    let m = message.to_lowercase();
    let statuses = status.map_or_else(|| statuses(&m), |s| vec![s]);
    is_usage_limit(&m)
        || is_rate_limit(&m, &statuses)
        || retry_after(message).is_some_and(|w| w > SHORT_RATE_LIMIT)
}

fn is_usage_limit(lower: &str) -> bool {
    USAGE_LIMIT.iter().any(|w| has_word(lower, w))
}

fn is_rate_limit(lower: &str, statuses: &[u16]) -> bool {
    RATE_LIMIT.iter().any(|w| lower.contains(w)) || statuses.contains(&429)
}

/// Whether `words` stand in `lower` at the start of a word (not after a letter or a digit).
fn has_word(lower: &str, words: &str) -> bool {
    lower
        .match_indices(words)
        .any(|(at, _)| !lower[..at].chars().next_back().is_some_and(char::is_alphanumeric))
}

/// The HTTP statuses `lower` (a lowercase message) names where a status stands: `status 503`, `status: 503`,
/// `status code 503`, `"status":503`, `API Error: 529`, `HTTP/1.1 502`. A number anywhere else ("at most 500
/// items") is not one.
fn statuses(lower: &str) -> Vec<u16> {
    const MARKERS: [&str; 5] = ["status code", "status", "api error", "http/1.1", "http/2"];
    let mut found = Vec::new();
    for marker in MARKERS {
        for (at, _) in lower.match_indices(marker) {
            let rest = lower[at + marker.len()..].trim_start_matches([' ', ':', '"', '=']);
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            if digits == 3
                && let Ok(status) = rest[..3].parse::<u16>()
                && (100..600).contains(&status)
            {
                found.push(status);
            }
        }
    }
    found
}

/// The words that start a wait in numbers and units ([`retry_after`]).
const WAIT_MARKERS: [&str; 4] = ["try again in ", "retry after ", "retry in ", "retry-after: "];

/// How long a `message` says to wait before trying again ("try again in 1.5s", "retry after 20 seconds", "try
/// again in 2 hours 5 minutes", "try again in 1m12.5s"), when it says so in numbers and units: every amount of a
/// compound duration counts, and of several waits (Claude Code joins its error fields into one message) the
/// longest.
pub fn retry_after(message: &str) -> Option<Duration> {
    longest_wait(message).map(|(_, _, wait)| wait)
}

/// The longest wait `message` asks for in numbers and units: where its marker starts (a byte offset of
/// `message`), the marker and the wait. The first of equal ones.
fn longest_wait(message: &str) -> Option<(usize, &'static str, Duration)> {
    // ASCII lowercase keeps every byte offset of `message`.
    let lower = message.to_ascii_lowercase();
    let mut longest: Option<(usize, &'static str, Duration)> = None;
    for marker in WAIT_MARKERS {
        for (at, _) in lower.match_indices(marker) {
            let Some(wait) = read_wait(&lower[at + marker.len()..]) else { continue };
            if longest.is_none_or(|(first, _, most)| wait > most || (wait == most && at < first)) {
                longest = Some((at, marker, wait));
            }
        }
    }
    longest
}

/// The wait at the start of `rest` (lowercase), in numbers and units.
fn read_wait(mut rest: &str) -> Option<Duration> {
    let (mut total, mut read) = (0.0, false);
    // Up to six amounts, each a number and its unit, written together ("1m30s") or apart ("2 hours 5 minutes").
    for _ in 0..6 {
        rest = rest.trim_start_matches([' ', ',']);
        if read && let Some(after) = rest.strip_prefix("and ") {
            rest = after.trim_start();
        }
        let digits = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
        let Ok(amount) = rest[..digits].trim_end_matches('.').parse::<f64>() else { break };
        let after = rest[digits..].trim_start();
        let letters = after.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(after.len());
        let millis = match &after[..letters] {
            "ms" | "msec" | "millisecond" | "milliseconds" => 1.0,
            "s" | "sec" | "secs" | "second" | "seconds" => 1e3,
            "m" | "min" | "mins" | "minute" | "minutes" => 60e3,
            "h" | "hr" | "hrs" | "hour" | "hours" => 3600e3,
            // A bare number at the end: seconds (a Retry-After header's).
            "" if after.is_empty() => 1e3,
            _ => break,
        };
        total += amount * millis;
        read = true;
        rest = &after[letters..];
    }
    (read && total.is_finite() && total < 1e12).then(|| Duration::from_millis(total.round() as u64))
}

/// When a limit that `message` reports passes, in the CLI's own words with what it says before them: Claude
/// Code's "resets 3pm (Europe/Berlin)", Codex's "try again at 3:04 PM" or a rate limit's "try again in 1.5s" (of
/// several waits, the one [`retry_after`] reads: the longest). Older Claude Code versions print the seconds since
/// 1970 after a `|`: "resets at" and that time in RFC 3339 (UTC).
pub fn reset_time(message: &str) -> Option<String> {
    // ASCII lowercase keeps every byte offset of `message`.
    let lower = message.to_ascii_lowercase();
    // The clause after the `marker` at `at`: to a sentence's end, a `·`, `∙` or `|` separator, a line break.
    let clause = |at: usize, marker: &str| {
        let rest = &message[at + marker.len()..];
        let end = rest
            .char_indices()
            .find(|&(i, c)| {
                matches!(c, '·' | '∙' | '|' | '\n')
                    || (c == '.' && rest[i + 1..].chars().next().is_none_or(char::is_whitespace))
            })
            .map_or(rest.len(), |(i, _)| i);
        let when = rest[..end].trim();
        (!when.is_empty()).then(|| format!("{marker}{when}"))
    };
    for marker in ["resets ", "try again at "] {
        if let Some(when) = lower.find(marker).and_then(|at| clause(at, marker)) {
            return Some(when);
        }
    }
    if let Some(when) = longest_wait(message).and_then(|(at, marker, _)| clause(at, marker)) {
        return Some(when);
    }
    let (_, epoch) = message.rsplit_once('|')?;
    let secs: u64 = epoch.trim().parse().ok().filter(|&s| s > 1_000_000_000)?;
    Some(format!("resets at {}", crate::util::rfc3339(secs as f64)))
}

/// The hint of a run that ended on a limit ([`is_limit`]): which limit (the account's usage only when the message
/// says so), and when it passes if the CLI said, in its own words.
pub fn limit_hint(message: &str, status: Option<u16>) -> String {
    let m = message.to_lowercase();
    let statuses = status.map_or_else(|| statuses(&m), |s| vec![s]);
    let usage = is_usage_limit(&m);
    let wait = retry_after(message);
    let (what, then) = if usage {
        ("the account's usage limit was reached", "start again once it resets")
    } else if wait.is_some_and(|w| w > SHORT_RATE_LIMIT) {
        // A rate limit too, maybe: what counts is that it is no short wait.
        ("the provider asked for a long wait", "start again later")
    } else if is_rate_limit(&m, &statuses) {
        ("the provider's rate limit was reached", "start again in a little while")
    } else if wait.is_some() {
        ("the provider asked for a long wait", "start again later")
    } else {
        ("the CLI reported a limit", "start again once it has passed")
    };
    let fewer = if usage { "" } else { " with fewer calls in flight (`parallel`)" };
    match reset_time(message) {
        Some(when) => format!("{what} ({when}): start again then{fewer}, or use another agent"),
        None => format!("{what}: {then}{fewer}, or use another agent"),
    }
}

/// The hint every agent gives for a limit ([`limit_hint`]) or a passing fault ([`BUSY`]); `None` for anything
/// else.
pub fn fault_hint(message: &str, status: Option<u16>) -> Option<String> {
    if is_limit(message, status) {
        Some(limit_hint(message, status))
    } else if is_transient(message, status) {
        Some(BUSY.into())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transient(message: &str) -> bool {
        is_transient(message, None)
    }

    #[test]
    fn passing_faults_are_told_from_limits_and_other_errors() {
        // What the CLIs print once their own retries are spent (Claude Code, then Codex).
        for passing in [
            r#"API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            r#"API Error: 500 {"type":"error","error":{"type":"api_error","message":"Internal server error"}}"#,
            "Repeated 529 Overloaded errors",
            "API Error: Connection error.",
            "API Error: terminated",
            "API Error: connect ETIMEDOUT 160.79.104.10:443",
            "API Error: connect ECONNREFUSED 127.0.0.1:443",
            "Opus is experiencing high load, please use /model to switch to Sonnet",
            "exceeded retry limit, last status: 503 Service Unavailable",
            "stream disconnected before completion: error sending request for url",
            "stream disconnected before completion: Transport error: connection reset by peer",
            "stream disconnected before completion: unexpected status 502 Bad Gateway",
            "We're currently experiencing high demand, which may cause temporary errors.",
            "unexpected status 502 Bad Gateway",
            r#"{"status":504,"message":"upstream"}"#,
            // A per-minute rate limit that asks for a short wait.
            "Rate limit reached for gpt-6.2-sol on tokens per min (TPM): Limit 30000, Used 29500, Requested 1200. \
             Please try again in 1.5s.",
            "429 Too Many Requests: retry after 20 seconds",
            "Rate limit reached. Please try again in 59s.",
        ] {
            assert!(transient(passing), "{passing}");
        }
        for not in [
            "You've hit your session limit · resets 3pm (Europe/Berlin)",
            "5-hour limit reached ∙ resets 3am",
            "Claude AI usage limit reached|1763481600",
            "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or try again at 3:04 PM.",
            "exceeded retry limit, last status: 429 Too Many Requests",
            "rate limit exceeded: 500 requests per minute",
            "Rate limit reached. Please try again in 20 minutes.",
            // Compound waits count in full: a minute and a half is past the cutoff.
            "Rate limit reached. Please try again in 1m30s.",
            "You exceeded your current quota. Please try again in 1.5s.",
            // A server error that asks for a long wait ends the run like a limit.
            "unexpected status 503 Service Unavailable; try again in 2 hours",
            "Invalid API key · Please run /login",
            "There's an issue with the selected model (nope). It may not exist or you may not have access to it.",
            "The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account.",
            "the image is 15000 px wide",
            // Codex's wrapper of any failed response, here a client error's.
            "stream disconnected before completion: Incorrect API key provided",
            "stream disconnected before completion: unexpected status 400 Bad Request: Invalid image",
            // A number that is not in a status's place, and a client error that names a server fault or a rate
            // limit: the 4xx wins.
            "API Error: 400 the request has at most 500 items",
            r#"API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"Overloaded context"}}"#,
            "API Error: 400 the rate limit settings are invalid; try again in 20 seconds",
        ] {
            assert!(!transient(not), "{not}");
        }
    }

    #[test]
    fn a_status_of_its_own_stands_for_the_numbers_in_the_text() {
        let text = "upstream failed after processing 503 tokens";
        assert!(!is_transient(text, None), "503 is not in a status's place");
        assert!(is_transient(text, Some(503)));
        assert!(is_transient(text, Some(529)));
        assert!(!is_transient("API Error: 503 Service Unavailable", Some(400)), "the reported status wins");
        assert!(!is_transient(text, Some(429)), "a rate limit without a wait");
        assert!(is_limit(text, Some(429)));
        assert!(is_transient("Please try again in 2s.", Some(429)));
        assert_eq!(fault_hint(text, Some(503)).as_deref(), Some(BUSY));
        assert_eq!(fault_hint(text, None), None);
    }

    #[test]
    fn waits_are_read_in_numbers_and_units() {
        let wait = |m: &str| retry_after(m).map(|d| d.as_millis());
        assert_eq!(wait("Please try again in 1.5s."), Some(1500));
        assert_eq!(wait("Please try again in 340ms"), Some(340));
        assert_eq!(wait("retry after 20 seconds"), Some(20_000));
        assert_eq!(wait("Try again in 2 hours 5 minutes. Thanks"), Some(7_500_000));
        assert_eq!(wait("Try again in 2 hours and 5 minutes."), Some(7_500_000));
        assert_eq!(wait("try again in 1m30s"), Some(90_000));
        assert_eq!(wait("Please try again in 1m12.5s."), Some(72_500));
        assert_eq!(wait("try again in 2h5m"), Some(7_500_000));
        assert_eq!(wait("try again in 20m0s."), Some(1_200_000));
        assert_eq!(wait("retry-after: 20"), Some(20_000));
        assert_eq!(wait("Try again later."), None);
        assert_eq!(wait("try again at 3:04 PM"), None);
        assert_eq!(wait("try again in a moment"), None);
        // Several waits in one message (Claude Code joins its error fields): the longest, whichever comes first.
        assert_eq!(wait("Try again in 1s · Retry after 90 seconds"), Some(90_000));
        assert_eq!(wait("retry after 90 seconds · try again in 1s"), Some(90_000));
        assert_eq!(wait("try again in a moment · retry in 2s"), Some(2000));
        assert!(!transient("Rate limit reached. Try again in 1s · Retry after 90 seconds"));
    }

    #[test]
    fn reset_times_are_read_in_the_clis_words() {
        let reset = |m: &str| reset_time(m);
        assert_eq!(
            reset("You've hit your session limit · resets 3pm (Europe/Berlin)").as_deref(),
            Some("resets 3pm (Europe/Berlin)")
        );
        assert_eq!(reset("5-hour limit reached ∙ resets 3am ∙ /upgrade").as_deref(), Some("resets 3am"));
        assert_eq!(
            reset("You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or try again at 3:04 PM.")
                .as_deref(),
            Some("try again at 3:04 PM")
        );
        assert_eq!(
            reset("Usage limit reached. Try again in 2 hours 5 minutes. Thanks").as_deref(),
            Some("try again in 2 hours 5 minutes")
        );
        assert_eq!(
            reset("Rate limit reached. Please try again in 1.5s.").as_deref(),
            Some("try again in 1.5s")
        );
        assert_eq!(
            reset("Claude AI usage limit reached|1763481600").as_deref(),
            Some("resets at 2025-11-18T16:00:00Z")
        );
        assert_eq!(
            reset("Rate limit reached. Try again in 1s · Retry after 90 seconds").as_deref(),
            Some("retry after 90 seconds"),
            "the wait that counts"
        );
        assert_eq!(reset("You've hit your usage limit. Try again later."), None);
        assert_eq!(reset("rate limit exceeded"), None);
    }

    #[test]
    fn a_limit_hint_names_the_limit_the_message_names() {
        let hint = |m: &str| limit_hint(m, None);
        assert_eq!(
            hint("You've hit your session limit · resets 3pm (Europe/Berlin)"),
            "the account's usage limit was reached (resets 3pm (Europe/Berlin)): start again then, or use another \
             agent"
        );
        assert_eq!(
            hint("5-hour limit reached ∙ resets 3am"),
            "the account's usage limit was reached (resets 3am): start again then, or use another agent"
        );
        assert_eq!(
            hint("Rate limit reached for gpt-6.2-sol on tokens per min. Please try again in 1.5s."),
            "the provider's rate limit was reached (try again in 1.5s): start again then with fewer calls in flight \
             (`parallel`), or use another agent"
        );
        assert_eq!(
            hint("You've hit your usage limit. Try again later."),
            "the account's usage limit was reached: start again once it resets, or use another agent"
        );
        assert!(
            hint("rate limit exceeded").starts_with("the provider's rate limit was reached: start again")
        );
        let long = "unexpected status 503 Service Unavailable; try again in 2 hours";
        assert_eq!(
            fault_hint(long, None).as_deref(),
            Some(
                "the provider asked for a long wait (try again in 2 hours): start again then with fewer calls in \
                 flight (`parallel`), or use another agent"
            )
        );
        // A rate limit that asks for a long wait is that long wait, whether its words or its status say 429.
        for long in
            ["429 Too Many Requests; try again in 1m30s", "Rate limit reached. Please try again in 1m30s."]
        {
            assert_eq!(
                fault_hint(long, None).as_deref(),
                Some(
                    "the provider asked for a long wait (try again in 1m30s): start again then with fewer calls \
                     in flight (`parallel`), or use another agent"
                ),
                "{long}"
            );
        }
        assert!(fault_hint("slow down; try again in 1m30s", Some(429)).unwrap().contains("long wait"));
        assert!(
            fault_hint("Try again in 1s · Retry after 90 seconds", Some(429))
                .unwrap()
                .contains("(retry after 90 seconds)")
        );
        assert_eq!(fault_hint("API Error: 529 Overloaded", None).as_deref(), Some(BUSY));
        assert!(
            fault_hint("Rate limit reached. Please try again in 1.5s.", None).unwrap().contains("rate limit")
        );
        assert_eq!(fault_hint("There's an issue with the selected model", None), None);
        // "spend" in "suspended" is no spending cap.
        assert_eq!(fault_hint("Your organization has been suspended", None), None);
        assert!(fault_hint("You have reached your spending cap", None).unwrap().contains("usage limit"));
    }
}
