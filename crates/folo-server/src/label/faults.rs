//! What a CLI's failure message says: a passing fault of the provider that the same call can get past a little
//! later ([`is_transient`]), a limit of the account or of the provider ([`limit_hint`], with when it resets in the
//! CLI's own words), or something else. Both agents' hints and the retry of [`super::agent::label_with_retry`]
//! read their messages with these.

use std::time::Duration;

/// A rate limit that asks for a wait of at most this long is a passing fault: the call is asked once more after
/// the wait, instead of ending a run of thousands of images on a per-minute limit. Longer than this it is a limit
/// to start again after (the run ends with when, in the hint).
pub const SHORT_RATE_LIMIT: Duration = Duration::from_secs(60);

/// The hint of a run that ended on passing faults ([`is_transient`]).
pub const BUSY: &str = "the provider is overloaded or failing right now (each call was asked twice); start again \
                        in a few minutes, or use another agent";

/// Words of a limit of the account (a plan's usage, a spending cap): asking again only spends more of a limit
/// that has run out, however soon it says to try again.
const USAGE_LIMIT: [&str; 10] = [
    "usage limit",
    "session limit",
    "weekly limit",
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
/// Words of a passing fault: the provider overloaded, a server error, the connection or the stream broken off.
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
/// The server errors that are passing faults (529: Anthropic's overloaded).
const PASSING_STATUS: [u16; 5] = [500, 502, 503, 504, 529];

/// Whether a CLI's failure `message` is a passing fault on the provider's side that the same call can get past a
/// little later: the provider is overloaded (Anthropic's 529, OpenAI's "high demand"), answered with a server
/// error (500, 502, 503, 504), the connection or the stream broke off, or a rate limit asks for a wait of at most
/// [`SHORT_RATE_LIMIT`]. Both CLIs retry such faults themselves before they give up, so one that reaches FOLO has
/// lasted a while. A usage limit never is one, nor a longer rate limit; and a client error (a 4xx status) wins over
/// any word of a server fault in the same message.
pub fn is_transient(message: &str) -> bool {
    let m = message.to_lowercase();
    if is_usage_limit(&m) {
        return false;
    }
    let statuses = statuses(&m);
    if is_rate_limit(&m, &statuses) {
        return retry_after(message).is_some_and(|wait| wait <= SHORT_RATE_LIMIT);
    }
    if statuses.iter().any(|s| (400..500).contains(s)) {
        return false;
    }
    statuses.iter().any(|s| PASSING_STATUS.contains(s)) || PASSING.iter().any(|w| m.contains(w))
}

/// Whether `message` reports a limit: the account's usage or the provider's rate limit.
pub fn is_limit(message: &str) -> bool {
    let m = message.to_lowercase();
    is_usage_limit(&m) || is_rate_limit(&m, &statuses(&m))
}

fn is_usage_limit(lower: &str) -> bool {
    USAGE_LIMIT.iter().any(|w| lower.contains(w))
}

fn is_rate_limit(lower: &str, statuses: &[u16]) -> bool {
    RATE_LIMIT.iter().any(|w| lower.contains(w)) || statuses.contains(&429)
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

/// How long a `message` says to wait before trying again ("try again in 1.5s", "retry after 20 seconds", "try
/// again in 2 hours 5 minutes"), when it says so in a number and a unit.
pub fn retry_after(message: &str) -> Option<Duration> {
    let lower = message.to_lowercase();
    let at = ["try again in ", "retry after ", "retry in ", "retry-after: "]
        .iter()
        .find_map(|marker| lower.find(marker).map(|at| at + marker.len()))?;
    let mut words = lower[at..].split_whitespace().peekable();
    let mut total = 0.0;
    let mut read = false;
    // Up to three amounts, each a number and its unit, written together ("1.5s") or apart ("2 hours").
    for _ in 0..3 {
        let Some(&word) = words.peek() else { break };
        let split = word.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(word.len());
        let Ok(amount) = word[..split].parse::<f64>() else { break };
        words.next();
        let unit = match &word[split..] {
            "" => words.next().unwrap_or("s"),
            unit => unit,
        };
        let unit = &unit[..unit.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(unit.len())];
        let millis = match unit {
            "ms" | "msec" | "millisecond" | "milliseconds" => 1.0,
            "s" | "sec" | "secs" | "second" | "seconds" => 1e3,
            "m" | "min" | "mins" | "minute" | "minutes" => 60e3,
            "h" | "hr" | "hrs" | "hour" | "hours" => 3600e3,
            _ => break,
        };
        total += amount * millis;
        read = true;
        if words.peek() == Some(&"and") {
            words.next();
        }
    }
    (read && total.is_finite() && total < 1e12).then(|| Duration::from_millis(total.round() as u64))
}

/// When a limit that `message` reports passes, in the CLI's own words with what it says before them: Claude
/// Code's "resets 3pm (Europe/Berlin)", Codex's "try again at 3:04 PM" or a rate limit's "try again in 1.5s". Older
/// Claude Code versions print the seconds since 1970 after a `|`: "resets at" and that time in RFC 3339 (UTC).
pub fn reset_time(message: &str) -> Option<String> {
    // ASCII lowercase keeps every byte offset of `message`.
    let lower = message.to_ascii_lowercase();
    for marker in ["resets ", "try again at ", "try again in ", "retry after "] {
        if let Some(at) = lower.find(marker) {
            let rest = &message[at + marker.len()..];
            // To the end of the clause: a sentence's end, a `·` or `|` separator, a line break.
            let end = rest
                .char_indices()
                .find(|&(i, c)| {
                    matches!(c, '·' | '|' | '\n')
                        || (c == '.' && rest[i + 1..].chars().next().is_none_or(char::is_whitespace))
                })
                .map_or(rest.len(), |(i, _)| i);
            let when = rest[..end].trim();
            if !when.is_empty() {
                return Some(format!("{marker}{when}"));
            }
        }
    }
    let (_, epoch) = message.rsplit_once('|')?;
    let secs: u64 = epoch.trim().parse().ok().filter(|&s| s > 1_000_000_000)?;
    Some(format!("resets at {}", crate::util::rfc3339(secs as f64)))
}

/// The hint of a run that ended on a limit: which limit (the account's usage only when the message says so),
/// and when it passes if the CLI said, in its own words.
pub fn limit_hint(message: &str) -> String {
    let m = message.to_lowercase();
    let (what, then) = if is_usage_limit(&m) {
        ("the account's usage limit was reached", "start again once it resets")
    } else if is_rate_limit(&m, &statuses(&m)) {
        ("the provider's rate limit was reached", "start again in a little while")
    } else {
        ("the CLI reported a limit", "start again once it has passed")
    };
    let fewer = if is_usage_limit(&m) { "" } else { " with fewer calls in flight (`parallel`)" };
    match reset_time(message) {
        Some(when) => format!("{what} ({when}): start again then{fewer}, or use another agent"),
        None => format!("{what}: {then}{fewer}, or use another agent"),
    }
}

/// The hint both agents give for a limit ([`limit_hint`]) or a passing fault ([`BUSY`]); `None` for anything else.
pub fn fault_hint(message: &str) -> Option<String> {
    if is_limit(message) {
        Some(limit_hint(message))
    } else if is_transient(message) {
        Some(BUSY.into())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            r#"{"status":504,"message":"upstream"}"#,
            // A per-minute rate limit that asks for a short wait.
            "Rate limit reached for gpt-6.2-sol on tokens per min (TPM): Limit 30000, Used 29500, Requested 1200. \
             Please try again in 1.5s.",
            "429 Too Many Requests: retry after 20 seconds",
        ] {
            assert!(is_transient(transient), "{transient}");
        }
        for not in [
            "You've hit your session limit · resets 3pm (Europe/Berlin)",
            "Claude AI usage limit reached|1763481600",
            "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or try again at 3:04 PM.",
            "exceeded retry limit, last status: 429 Too Many Requests",
            "rate limit exceeded: 500 requests per minute",
            "Rate limit reached. Please try again in 20 minutes.",
            "You exceeded your current quota. Please try again in 1.5s.",
            "Invalid API key · Please run /login",
            "There's an issue with the selected model (nope). It may not exist or you may not have access to it.",
            "The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account.",
            "the image is 15000 px wide",
            // A number that is not in a status's place, and a client error that names a server fault.
            "API Error: 400 the request has at most 500 items",
            r#"API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"Overloaded context"}}"#,
        ] {
            assert!(!is_transient(not), "{not}");
        }
    }

    #[test]
    fn waits_are_read_in_numbers_and_units() {
        let wait = |m: &str| retry_after(m).map(|d| d.as_millis());
        assert_eq!(wait("Please try again in 1.5s."), Some(1500));
        assert_eq!(wait("Please try again in 340ms"), Some(340));
        assert_eq!(wait("retry after 20 seconds"), Some(20_000));
        assert_eq!(wait("Try again in 2 hours 5 minutes. Thanks"), Some(7_500_000));
        assert_eq!(wait("Try again later."), None);
        assert_eq!(wait("try again at 3:04 PM"), None);
    }

    #[test]
    fn reset_times_are_read_in_the_clis_words() {
        let reset = |m: &str| reset_time(m);
        assert_eq!(
            reset("You've hit your session limit · resets 3pm (Europe/Berlin)").as_deref(),
            Some("resets 3pm (Europe/Berlin)")
        );
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
        assert_eq!(reset("You've hit your usage limit. Try again later."), None);
        assert_eq!(reset("rate limit exceeded"), None);
    }

    #[test]
    fn a_limit_hint_names_the_limit_the_message_names() {
        assert_eq!(
            limit_hint("You've hit your session limit · resets 3pm (Europe/Berlin)"),
            "the account's usage limit was reached (resets 3pm (Europe/Berlin)): start again then, or use another \
             agent"
        );
        assert_eq!(
            limit_hint("Rate limit reached for gpt-6.2-sol on tokens per min. Please try again in 1.5s."),
            "the provider's rate limit was reached (try again in 1.5s): start again then with fewer calls in flight \
             (`parallel`), or use another agent"
        );
        assert_eq!(
            limit_hint("You've hit your usage limit. Try again later."),
            "the account's usage limit was reached: start again once it resets, or use another agent"
        );
        assert!(
            limit_hint("rate limit exceeded")
                .starts_with("the provider's rate limit was reached: start again")
        );
        assert_eq!(fault_hint("API Error: 529 Overloaded").as_deref(), Some(BUSY));
        assert!(fault_hint("Rate limit reached. Please try again in 1.5s.").unwrap().contains("rate limit"));
        assert_eq!(fault_hint("There's an issue with the selected model"), None);
    }
}
