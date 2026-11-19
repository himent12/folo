# folo-api

The serde request and response types of [docs/API.md](../../docs/API.md): data types only, no IO and no async. It
depends on no workspace crate (serde, serde_json, and schemars behind the `schema` feature). Its users are
folo-server, folo-cli, folo-client, folo-mcp, folo-cloud, folo-vision (the trainer takes its `TrainConfig` and sends
its `RunEvent`s), folo-bench, folo-wasm and FOLO Desktop (`apps/desktop`, a separate cargo workspace that depends on
it by path). Field names and enum spellings are the wire format.

## Where things are

Every module but `codes`, `json` and `weights` is private and re-exported at the crate root (`folo_api::Run`).

| Path | What it holds | docs/API.md section |
|---|---|---|
| `src/system.rs` | Health, host CPU, memory and GPUs, `capabilities`, `API_VERSION`, `CliCommand`, `FEATURE_*` names (others sit next to their types) | System |
| `src/datasets.rs` | Datasets, their statistics, images, boxes and issues | Datasets |
| `src/models.rs` | Models, their validation results and exports | Models |
| `src/runs.rs` | `TrainConfig`, `Run`, what the trainer resolved, the training queue | Training runs |
| `src/run_events.rs` | Steps, epochs, samples, logs and `RunEvent` (the SSE events of `/events`) | Training runs |
| `src/inference.rs` | `POST /api/predict` | Inference |
| `src/validation.rs` | `POST /api/val` | Validation |
| `src/throughput.rs` | The throughput table, benchmark probes and run estimates | Throughput and estimates |
| `src/label.rs` | Labeling jobs, their images, agents and events | Agent labeling |
| `src/bench/` | Benchmarks (`benchmarks.rs`), `catalog.rs`, `entries.rs`, `results.rs`, `loops.rs`, `events.rs` | Vision benchmarks |
| `src/cloud.rs` | Providers, connections, offers, instances, a run's `compute` and `cloud`, upload sessions | Cloud compute, Uploads |
| `src/errors.rs` | `ErrorBody` and `ApiError`: the body of every non-2xx response | Errors |
| `src/codes.rs` | `CODES`: every error code with its HTTP status, CLI exit code and meaning | Errors |
| `src/json.rs` | JSON presentation helpers of the CLI's `--json` records and the MCP results; `cli_command`, a command with paths FOLO prints for the user to paste: it pastes into PowerShell, cmd or a POSIX shell when no path holds `"`, `$`, `` ` ``, `%`, `!`, `\`, a typographic double quote or a control character; for a path that does, it shows the path and leaves the quoting to the person | |
| `src/weights.rs` | The published weights `folo weights pull` downloads, their cache folder, `home_dir` | |
| `src/tests.rs`, `src/bench/tests.rs` | Unit tests of defaults, round trips and compatibility | |

## Start here

1. `src/lib.rs`: the module list and the conventions (RFC 3339 times, `*_s` / `*_ms` durations, pixel `xyxy`
   boxes, opaque ids).
2. `TrainConfig` and `Run` (`src/runs.rs`): what a training run is asked to do and what the API says about it.
3. `RunEvent` (`src/run_events.rs`): what a run streams, and what folo-vision's trainer emits.
4. `CODES` (`src/codes.rs`) and `ApiError` (`src/errors.rs`): how every front end reports a failure.

## The `schema` feature

`schema` derives `schemars::JsonSchema` for the wire types (`#[cfg_attr(feature = "schema", ...)]`), so the MCP server
(docs/MCP.md) and `folo describe` can publish JSON Schemas generated from the wire types. It is off by default to
keep the crate light for FOLO Desktop. folo-mcp turns it on in its `Cargo.toml`; the `folo` binary gets it through
folo-mcp.

## Changing a type

`docs/API.md` is the contract: change it in the same PR as the type, and keep `web/src/lib/api/types/` (one file per
section, re-exported by `types.ts`) in sync. The code and docs follow these rules for compatibility between versions:

- Unknown fields are ignored, and missing optional fields take their defaults (docs/API.md, Conventions). A field
  added to a response gets `#[serde(default)]`, so a response from an older server still parses, and its doc says
  what the missing field reads as (see the `Additive:` notes in `src/models.rs` and `src/throughput.rs`).
- Enum values grow. An enum a newer server may extend gets an `Unknown` variant with `#[serde(other)]`
  (`CloudAvailability`, `CloudInstanceStatus`, `CloudPhase` in `src/cloud.rs`). `Optimizer` and `ResolvedOptimizer`
  (`src/runs.rs`) also keep `Unknown` out of the schema with `schemars(skip)`, and a run that would train with an
  `Unknown` optimizer is refused with `invalid_config`.
- `ApiError.hint` is always serialized (`null` when absent).
- A new optional capability of a server is announced as a `FEATURE_*` name in `/api/system`'s `features`.

`src/tests.rs` holds a test for each of these (`unknown_optimizers_decode`, `system_capabilities_are_additive`,
`error_hint_is_always_present`, ...): add one when you add a field or a variant that older or newer peers must read.

## Common changes

| To do this | Touch these files |
|---|---|
| Add a field to a request or response | The type in `src/<section>.rs` (`#[serde(default)]` for a response field); a test in `src/tests.rs` when older servers lack it; the section in `docs/API.md`; `web/src/lib/api/types/<section>.ts`; the server code that fills it (crates/folo-server) |
| Add an enum value | The variant (and its `as_str` / `ALL` where the enum has them, as `LabelAgentId` in `src/label.rs`); `docs/API.md`; the web types; every `match` the compiler then points at, also in `apps/desktop` |
| Add an error code | `CODES` in `src/codes.rs`; the Errors table in `docs/API.md`; `crates/folo-cli/tests/codes.rs` fails on a code the sources use that is not listed |
| Announce an optional API feature | A `FEATURE_*` constant next to its types; the server lists it in `/api/system` and `/api/health` (`features` in crates/folo-server/src/routes/system.rs, or `Store::features`) |

## Tests and benchmarks

- `cargo test -p folo-api`: the unit tests in `src/tests.rs` and `src/bench/tests.rs`.
- `cargo test -p folo-api --features schema`: also `schema_leaves_unknown_out`, which needs the feature.

No GPU needed, nothing is `#[ignore]`d, and the crate has no benchmarks.

## Further reading

- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [docs/API.md](../../docs/API.md): the HTTP contract these types model.
- [docs/MCP.md](../../docs/MCP.md): the MCP server and its schemas.
- [docs/CLI.md](../../docs/CLI.md): the CLI's `--json` records and exit codes.
- [folo-server](../folo-server/README.md): the server that implements the contract.
