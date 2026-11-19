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
    c(
        "invalid_arguments",
        400,
        2,
        "tool arguments that fail the inputSchema or a documented range (MCP; `folo label` reports them as usage)",
    ),
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
    c(
        "dataset_unfinished",
        400,
        1,
        "a dataset folder holding `.folo-unfinished`: `folo dataset synth` / `convert` is writing it or was stopped",
    ),
    c("dataset_issues", 400, 1, "`folo dataset check` found errors or warnings (CLI)"),
    c(
        "bench_class_mismatch",
        400,
        1,
        "a model entry of a vision benchmark whose model knows none of the benchmark's classes",
    ),
    c("too_many_inputs", 400, 2, "more images than predict_batch takes (MCP)"),
    c("unauthorized", 401, 1, "a request without the server's token (`folo serve --token`)"),
    c("forbidden_origin", 403, 1, "a state-changing request from a disallowed browser origin (HTTP)"),
    c(
        "forbidden_host",
        403,
        1,
        "a request to a server on loopback for a Host that is not loopback nor --allow-host (HTTP)",
    ),
    c("read_only", 403, 1, "a mutating tool on a --read-only server (MCP)"),
    c("path_outside_data_dir", 403, 1, "a read outside --data-dir with --confine (MCP)"),
    // Not found.
    c("dataset_not_found", 404, 3, "no such dataset"),
    c("image_not_found", 404, 3, "no such image"),
    c("model_not_found", 404, 3, "no such model or weights file"),
    c("run_not_found", 404, 3, "no such run"),
    c(
        "no_matching_dataset",
        404,
        3,
        "no registered dataset has a models-folder model's classes, so it cannot be evaluated",
    ),
    c("label_job_not_found", 404, 3, "no such labeling job"),
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
    c("upload_not_found", 404, 3, "no such upload session (it may have expired)"),
    c("provider_not_found", 404, 3, "no such cloud provider"),
    c("connection_not_found", 404, 3, "no such cloud connection"),
    c("offer_not_found", 404, 3, "the connection has no such offer (any more)"),
    c("instance_not_found", 404, 3, "no such cloud instance"),
    c("bench_not_found", 404, 3, "no such vision benchmark"),
    c("bench_entry_not_found", 404, 3, "no such entry of the benchmark"),
    c("bench_loop_not_found", 404, 3, "no such self-improvement loop of the benchmark"),
    c("method_not_allowed", 405, 2, "wrong HTTP method"),
    // Conflicts.
    c("run_active", 409, 1, "the run is still queued, running or stopping"),
    c("run_not_active", 409, 1, "stopping a run that already finished"),
    c(
        "dataset_in_use",
        409,
        1,
        "deleting a dataset's files while another dataset, or runs of an unregistered one, read files there",
    ),
    c("run_not_resumable", 409, 1, "resuming a run that completed; start a new run from its weights instead"),
    c(
        "run_not_queued",
        409,
        1,
        "moving a run that is not waiting in the queue (it is starting, running or finished)",
    ),
    c(
        "label_job_active",
        409,
        1,
        "changing a labeling job, or an image an agent call holds, while its agent run is active",
    ),
    c("label_job_not_active", 409, 1, "stopping a labeling job that has no agent run"),
    c(
        "label_boxes_changed",
        409,
        1,
        "an image's boxes changed since the revision a box edit or review was made on (view the image again)",
    ),
    c("bench_frozen", 409, 1, "changing the plan of a vision benchmark that has entries"),
    c("bench_active", 409, 1, "deleting a vision benchmark while one of its entries or loops is active"),
    c("bench_entry_active", 409, 1, "deleting a benchmark entry that is still queued or running"),
    c("bench_entry_not_active", 409, 1, "stopping a benchmark entry that has ended"),
    c("bench_entry_not_done", 409, 1, "the result or images of a benchmark entry that has not finished"),
    c(
        "bench_entry_not_open",
        409,
        1,
        "sending predictions to, or finishing, a benchmark entry that is not an open external entry",
    ),
    c("bench_entry_in_loop", 409, 1, "deleting a benchmark entry that a running self-improvement loop uses"),
    c("bench_loop_active", 409, 1, "deleting a self-improvement loop that is still running"),
    c("bench_loop_not_active", 409, 1, "stopping a self-improvement loop that has ended"),
    c("output_exists", 409, 1, "the output folder exists and is not empty (CLI, Generate synthetic)"),
    c("upload_conflict", 409, 1, "an upload part out of order, or completing an upload that is not whole"),
    c(
        "connection_in_use",
        409,
        1,
        "removing a cloud connection that still has live instances or active runs",
    ),
    c(
        "cloud_refused",
        409,
        1,
        "the cloud provider refused the request (no capacity for the offer, no balance, a quota)",
    ),
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
    c("invalid_upload", 422, 1, "an upload that is not a readable tar archive, or holds something unsafe"),
    c("cloud_auth_failed", 422, 1, "the cloud provider rejected the connection's credentials"),
    // Runtime failures.
    c("internal", 500, 1, "a bug or an unexpected failure"),
    c("io_error", 500, 1, "a file could not be read or written"),
    c("data_error", 500, 1, "reading the dataset failed while training or validating"),
    c("training_failed", 500, 1, "training diverged or produced no usable batch"),
    c("gpu_out_of_memory", 507, 1, "the GPU ran out of memory; lower batch or imgsz, or use a smaller arch"),
    c("not_implemented", 501, 4, "the feature exists in the interface but not yet in the engine"),
    c(
        "not_supported",
        501,
        4,
        "valid, but this machine cannot run it (mixed-precision training without GPU tensor cores, a labeling \
         agent whose CLI is not installed)",
    ),
    c(
        "server_unreachable",
        502,
        1,
        "another FOLO server (`folo mcp --server`, a cloud worker) does not answer",
    ),
    c(
        "bad_gateway",
        502,
        1,
        "another FOLO server answered with something that is not this API (a proxy's page, another version)",
    ),
    c("cloud_unavailable", 502, 1, "the cloud provider's API does not answer, or answers with an error"),
    c(
        "agent_failed",
        502,
        1,
        "a coding agent's call failed or its answer could not be used (designing a benchmark plan)",
    ),
    c(
        "cloud_boot_failed",
        502,
        1,
        "a cloud instance never served a usable worker (its setup failed or timed out, too old, no GPU adapter), or \
         the worker cannot be downloaded (nothing is rented then); Run.error_code",
    ),
    c("cloud_sync_failed", 502, 1, "copying data to or from a cloud worker failed; Run.error_code"),
    c("cloud_instance_lost", 502, 1, "a cloud run's instance is gone before the run ended; Run.error_code"),
    c("shutting_down", 503, 1, "the server is shutting down"),
    c(
        "interrupted",
        503,
        1,
        "the server stopped (or restarted) while the run trained or waited; Run.error_code of a stopped run",
    ),
    c("runtime", 500, 1, "the async runtime could not start (CLI)"),
    c("bind_failed", 500, 1, "`folo serve` could not bind its address (CLI)"),
    c("serve_failed", 500, 1, "`folo serve` stopped with an error (CLI)"),
    c(
        "data_dir_in_use",
        409,
        1,
        "another `folo serve` or embedded `folo mcp` already uses this data dir (CLI)",
    ),
    c("data_dir_lock_failed", 500, 1, "the data dir's `serve.lock` could not be locked (CLI)"),
    c("mcp_failed", 500, 1, "`folo mcp` stopped with an error (CLI)"),
    c(
        "label_run_failed",
        502,
        1,
        "`folo label start` waited for an agent run that ended failed (the agent CLI's message; CLI)",
    ),
    c(
        "download_failed",
        502,
        1,
        "`folo weights pull`: the HTTPS download failed (no network, a timeout, a server error, a cut body); nothing          was installed (CLI)",
    ),
    c(
        "gh_failed",
        502,
        1,
        "`folo weights pull`: the release is not public, and the GitHub CLI fallback is missing or failed (CLI)",
    ),
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
