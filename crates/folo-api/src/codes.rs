//! The one error-code vocabulary of FOLO's front ends: the HTTP API (`docs/API.md`), the CLI (`docs/CLI.md`) and
//! the MCP server (`docs/MCP.md`) report the same failure with the same `code`. Each code has the HTTP status
//! the server answers with and the exit code `folo` exits with; `crates/folo-cli/tests/codes.rs` checks that every
//! code the sources use is listed here, with the exit code the CLI uses for it.

/// One error code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Code {
    /// The stable snake_case code.
    pub code: &'static str,
    /// HTTP status (`folo serve`; also the status MCP tool errors carry internally).
    pub status: u16,
    /// `folo` exit code: 1 runtime error, 2 usage error, 3 not found, 4 not implemented.
    pub exit: u8,
    /// When it is used.
    pub meaning: &'static str,
}

const fn c(code: &'static str, status: u16, exit: u8, meaning: &'static str) -> Code {
    Code { code, status, exit, meaning }
}

/// The codes an agent may see, grouped by kind.
pub const CODES: &[Code] = &[
    // Bad input.
    c("usage", 400, 2, "invalid command-line arguments or flags (CLI)"),
    c("invalid_arguments", 400, 2, "tool arguments that fail the inputSchema or a documented range (MCP)"),
    c("bad_request", 400, 2, "a malformed request (HTTP)"),
    c("invalid_path", 400, 2, "a malformed path parameter (HTTP)"),
    c("invalid_query", 400, 2, "a malformed query parameter (HTTP)"),
    c(
        "invalid_body",
        400,
        2,
        "a malformed or wrongly typed JSON body (HTTP; MCP reports it as invalid_arguments)",
    ),
    c(
        "invalid_config",
        400,
        2,
        "a TrainConfig value out of range, or a resume that changes the run's settings",
    ),
    c("unknown_format", 400, 1, "nothing at the path looks like a dataset"),
    c(
        "invalid_dataset",
        400,
        1,
        "a dataset that cannot be used (malformed data.yaml, no images, an empty split)",
    ),
    c("dataset_issues", 400, 1, "`folo dataset check` found errors or warnings (CLI)"),
    c("too_many_inputs", 400, 2, "more images than predict_batch takes (MCP)"),
    c("forbidden_origin", 403, 1, "a state-changing request from a disallowed browser origin (HTTP)"),
    c("read_only", 403, 1, "a mutating tool on a --read-only server (MCP)"),
    c("path_outside_data_dir", 403, 1, "a read outside --data-dir with --confine (MCP)"),
    // Not found.
    c("dataset_not_found", 404, 3, "no such dataset"),
    c("image_not_found", 404, 3, "no such image"),
    c("model_not_found", 404, 3, "no such model or weights file"),
    c("run_not_found", 404, 3, "no such run"),
    c("path_not_found", 404, 3, "no such file or folder"),
    c(
        "device_not_found",
        404,
        3,
        "a device (gpu:<n>) this machine does not have; the hint lists the ones it has",
    ),
    c("epoch_not_available", 404, 3, "an epoch that has not been validated yet"),
    c(
        "weights_not_found",
        404,
        3,
        "published weights that are unknown or not downloaded; the hint says `folo weights pull <name>` (CLI)",
    ),
    c("route_not_found", 404, 3, "no such HTTP route"),
    c("method_not_allowed", 405, 2, "wrong HTTP method"),
    // Conflicts.
    c("run_active", 409, 1, "the run is still queued, running or stopping"),
    c("run_not_active", 409, 1, "stopping a run that already finished"),
    c("run_not_resumable", 409, 1, "resuming a run that completed; start a new run from its weights instead"),
    c("output_exists", 409, 1, "the output folder exists and is not empty (CLI)"),
    // Too large.
    c("payload_too_large", 413, 1, "an upload or image over the limit"),
    c("result_too_large", 413, 1, "a tool result over 32 KB; the hint names the resource to read (MCP)"),
    c(
        "unsupported_media_type",
        415,
        2,
        "a JSON endpoint called without Content-Type: application/json (HTTP)",
    ),
    c("image_decode_failed", 422, 1, "an image that cannot be decoded"),
    c("invalid_weights", 422, 1, "a file that is not a readable .folo weights file or checkpoint"),
    // Runtime failures.
    c("internal", 500, 1, "a bug or an unexpected failure"),
    c("io_error", 500, 1, "a file could not be read or written"),
    c("data_error", 500, 1, "reading the dataset failed while training or validating"),
    c("training_failed", 500, 1, "training diverged or produced no usable batch"),
    c("gpu_out_of_memory", 507, 1, "the GPU ran out of memory; lower batch or imgsz, or use a smaller arch"),
    c("not_implemented", 501, 4, "the feature exists in the interface but not yet in the engine"),
    c("server_unreachable", 502, 1, "proxy mode: `folo serve` does not answer (MCP)"),
    c("shutting_down", 503, 1, "the server is shutting down"),
    c("runtime", 500, 1, "the async runtime could not start (CLI)"),
    c("bind_failed", 500, 1, "`folo serve` could not bind its address (CLI)"),
    c("serve_failed", 500, 1, "`folo serve` stopped with an error (CLI)"),
    c("mcp_failed", 500, 1, "`folo mcp` stopped with an error (CLI)"),
    c("gh_failed", 502, 1, "`folo weights pull`: the GitHub CLI is missing or its download failed (CLI)"),
    c(
        "checksum_missing",
        502,
        1,
        "`folo weights pull`: the release's SHA256SUMS does not list the file (CLI)",
    ),
    c(
        "checksum_mismatch",
        502,
        1,
        "`folo weights pull`: the download's SHA-256 differs from SHA256SUMS; nothing was installed (CLI)",
    ),
];

/// The entry for `code`, if it is part of the vocabulary.
pub fn find(code: &str) -> Option<&'static Code> {
    CODES.iter().find(|c| c.code == code)
}

/// The `folo` exit code for `code` (1, a runtime error, for codes outside the vocabulary).
pub fn exit_code(code: &str) -> u8 {
    find(code).map_or(1, |c| c.exit)
}

/// `message` cut to at most `max` characters, ending in `…` when cut. Error messages that wrap a driver's report
/// (a wgpu validation error can run to tens of kilobytes) go through this before an agent sees them.
pub fn cap(message: &str, max: usize) -> String {
    let message = message.trim_end();
    match message.char_indices().nth(max) {
        None => message.to_string(),
        Some((i, _)) => format!("{}…", &message[..i]),
    }
}

/// Longest error message the front ends report ([`cap`]).
pub const MAX_MESSAGE_CHARS: usize = 500;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_snake_case() {
        let mut seen = std::collections::HashSet::new();
        for c in CODES {
            assert!(seen.insert(c.code), "{} twice", c.code);
            assert!(c.code.chars().all(|ch| ch.is_ascii_lowercase() || ch == '_'), "{}", c.code);
            assert!((1..=4).contains(&c.exit) && (400..600).contains(&c.status), "{}", c.code);
            // Exit 3 is exactly the 404s.
            assert_eq!(c.exit == 3, c.status == 404, "{}", c.code);
        }
        assert_eq!(exit_code("device_not_found"), 3);
        assert_eq!(exit_code("not_implemented"), 4);
        assert_eq!(exit_code("something_else"), 1);
        assert_eq!(cap("héllo\n", 2), "hé…");
        assert_eq!(cap("hi\n", 5), "hi");
    }
}
