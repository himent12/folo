//! The hints of benchmark errors in each front end's words. The server's hints name HTTP requests; [`decorate`]
//! replaces them by code, naming the MCP tools or the `folo bench` commands that help.

use crate::errors::Error;
pub use crate::label::hints::Front;

/// Stands in an upload's hint for the commands that show and stop its entry: [`decorate`] puts them in the front end's
/// words.
pub(super) const ENTRY_COMMANDS: &str = "<entry commands>";

/// True when `hint` was written by an upload operation (`ops.rs`) rather than by the HTTP API.
fn is_upload_hint(hint: &str) -> bool {
    hint.starts_with("entry '") || hint.starts_with("the server may have opened an entry named '")
}

/// `e` with a hint in `front`'s words (an error whose code has nothing to add keeps its hint).
pub fn decorate(mut e: Error, front: Front) -> Error {
    let mcp = front == Front::Mcp;
    let upload_hint = e.body.hint.as_deref().is_some_and(is_upload_hint);
    let hint = if upload_hint {
        None
    } else {
        match e.body.code.as_str() {
            "bench_not_found" => Some(if mcp {
                "list_benchmarks lists them"
            } else {
                "`folo bench list` lists the benchmarks"
            }),
            "bench_entry_not_found" => Some(if mcp {
                "get_benchmark lists its entries"
            } else {
                "`folo bench show BENCH` lists its entries"
            }),
            "bench_active" => Some(if mcp {
                "wait for its entries to end (bench_entry), or stop them (stop_bench_entry)"
            } else {
                "wait for its entries to end (`folo bench status`), or stop them (`folo bench stop`)"
            }),
            "bench_entry_not_active" => Some(if mcp {
                "bench_entry says how it ended"
            } else {
                "`folo bench status BENCH ENTRY` says how it ended"
            }),
            "bench_entry_not_done" => Some(if mcp {
                "wait for the entry to finish with bench_entry, or pick a `done` entry from get_benchmark"
            } else {
                "wait for it with `folo bench status BENCH ENTRY --wait`, or pick a `done` entry from `folo bench show BENCH`"
            }),
            "model_not_found" => Some(if mcp {
                "list_models lists the models; published weights are models once pulled (`folo weights pull NAME`)"
            } else {
                "`folo weights pull NAME` makes published weights a model; the models are those Studio lists"
            }),
            // Only when the error has none: a failed submit's hint says what became of its entry.
            "invalid_arguments" if e.body.hint.is_none() => Some(if mcp {
                "check the tool's inputSchema (tools/list)"
            } else {
                "see `folo bench <command> --help`"
            }),
            "invalid_body" if !mcp && e.body.hint.is_none() => Some("see `folo bench <command> --help`"),
            _ => None,
        }
    };
    if let Some(h) = hint {
        e.body.hint = Some(h.to_string());
    }
    if let Some(h) = &mut e.body.hint {
        let commands = if mcp {
            "bench_entry shows it and stop_bench_entry stops it"
        } else {
            "`folo bench status BENCH ENTRY` shows it and `folo bench stop BENCH ENTRY` stops it"
        };
        if let Some(prefix) = h.strip_suffix(ENTRY_COMMANDS) {
            *h = format!("{prefix}{commands}");
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn hints_name_the_front_ends_own_commands() {
        let e =
            || Error::conflict("bench_active", "benchmark 'bn_1' has an active entry").hint("POST /api/…");
        assert!(decorate(e(), Front::Mcp).body.hint.unwrap().contains("stop_bench_entry"));
        assert!(decorate(e(), Front::Cli).body.hint.unwrap().contains("folo bench stop"));
        let other = Error::conflict("bench_frozen", "x").hint("kept");
        assert_eq!(decorate(other, Front::Cli).body.hint.unwrap(), "kept");
        let left = || Error::conflict("x", "y").hint(format!("entry left; {ENTRY_COMMANDS}"));
        assert!(decorate(left(), Front::Mcp).body.hint.unwrap().ends_with("stop_bench_entry stops it"));
        assert!(decorate(left(), Front::Cli).body.hint.unwrap().contains("`folo bench stop BENCH ENTRY`"));
    }

    #[test]
    fn upload_hints_survive_code_decoration_and_preserve_entry_names() {
        // Upload hint on `bench_entry_not_found` must not be overwritten with the generic lookup hint.
        let upload_err = || {
            Error::new(StatusCode::NOT_FOUND, "bench_entry_not_found", "benchmark 'bn_1' has no entry 'be_1'")
                .hint(format!(
                    "entry 'my {ENTRY_COMMANDS} run' (be_1) may be left on the server: its state could not be read; {ENTRY_COMMANDS}"
                ))
        };
        let mcp = decorate(upload_err(), Front::Mcp).body.hint.unwrap();
        assert!(
            mcp.starts_with(&format!("entry 'my {ENTRY_COMMANDS} run' (be_1) may be left")),
            "entry name placeholder must stay untouched: {mcp}"
        );
        assert!(mcp.ends_with("bench_entry shows it and stop_bench_entry stops it"), "{mcp}");
        let cli = decorate(upload_err(), Front::Cli).body.hint.unwrap();
        assert!(
            cli.starts_with(&format!("entry 'my {ENTRY_COMMANDS} run' (be_1) may be left")),
            "entry name placeholder must stay untouched: {cli}"
        );
        assert!(cli.ends_with("`folo bench stop BENCH ENTRY` stops it"), "{cli}");

        // Without a trailing placeholder, a literal `<entry commands>` inside the entry name is never replaced.
        let removed = Error::new(StatusCode::BAD_REQUEST, "invalid_body", "bad")
            .hint(format!("entry '{ENTRY_COMMANDS}' (be_1) was removed"));
        assert_eq!(
            decorate(removed, Front::Mcp).body.hint.unwrap(),
            format!("entry '{ENTRY_COMMANDS}' (be_1) was removed")
        );

        // Non-upload `bench_entry_not_found` still gets the front-end lookup hint.
        let plain = Error::new(StatusCode::NOT_FOUND, "bench_entry_not_found", "no entry")
            .hint("GET /api/bench/benchmarks/{id}/entries lists its entries");
        assert_eq!(decorate(plain, Front::Mcp).body.hint.unwrap(), "get_benchmark lists its entries");

        // Front-end-neutral `invalid_arguments` from bench ops gets each front end's own hint (never MCP's on CLI).
        let cli_arg =
            decorate(Error::new(StatusCode::BAD_REQUEST, "invalid_arguments", "bad line"), Front::Cli);
        assert_eq!(cli_arg.body.hint.as_deref(), Some("see `folo bench <command> --help`"));
        let mcp_arg =
            decorate(Error::new(StatusCode::BAD_REQUEST, "invalid_arguments", "bad line"), Front::Mcp);
        assert_eq!(mcp_arg.body.hint.as_deref(), Some("check the tool's inputSchema (tools/list)"));
    }
}
