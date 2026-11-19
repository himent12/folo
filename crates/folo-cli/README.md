# folo-cli

The `folo` binary: training, evaluation, inference, export, dataset tools, published weights, cloud compute, agent
labeling, the Studio server (`folo serve`) and the MCP server (`folo mcp`). It sits on top of the stack and depends on folo-api,
folo-client, folo-data, folo-gpu, folo-nn, folo-server, folo-mcp and folo-vision; no workspace crate depends on it.
Its contract is [docs/CLI.md](../../docs/CLI.md). The library target (`src/lib.rs`) holds only the clap command tree
and two constants, so tests can parse commands with the real parser; every command is a module of the binary
(`src/main.rs`).

## Where things are

| path | what it holds |
|---|---|
| `src/main.rs` | `main`: parses the command line, dispatches each subcommand (`run`), turns on the caches, resolves `--data-dir` |
| `src/lib.rs` | `VERSION` (`folo --version`), `WEIGHTS_REPO` (default of `weights pull --repo`), `pub mod cli` |
| `src/cli.rs` | The clap tree: `Cli`, `Command` and every `*Args` struct (`src/cli/label.rs`: `folo label`'s). Help strings and defaults here are what `folo describe` prints |
| `src/output.rs` | `Out` (human text vs `--json` records), `CliError` and its JSON envelope, `ExitCode` |
| `src/describe.rs` | `folo describe`: the clap tree as JSON, plus the MCP tool schemas from folo-mcp |
| `src/engine.rs` | Opening datasets, registering the GPU backend for `--device`, engine errors as `CliError` |
| `src/serve.rs` | `folo serve`; also tracing to stderr, the data-dir lock errors and where the Studio comes from |
| `src/serve_lock.rs` | One owner per data dir: the OS lock on `<data dir>/serve.lock` and the address in `serve.json` |
| `src/mcp.rs` | `folo mcp`: picks the folo-mcp backend (embedded, `--mock`, `--server`) and runs it on stdio |
| `src/doctor.rs` | `folo doctor`: host, GPU adapters, the adapter FOLO picks, the Studio build |
| `src/train/` | `folo train` and `folo stop`: `mod.rs` (flags, run folder, trainer), `config.rs` (`--set`), `resume.rs` (`--resume`), `progress.rs` (live view, JSON lines, summary), `stop.rs` (stop file, the run folder's marker and lock, Ctrl+C), `tests.rs` |
| `src/infer.rs` | `folo val`, `folo predict` and `folo export` |
| `src/coco.rs` | `val --coco` / `--save-json`: pycocotools-exact evaluation and COCO results files |
| `src/draw.rs` | Annotated images for `predict --save` (class colours, a built-in digit font) |
| `src/dataset.rs` | `folo dataset check \| stats \| synth \| convert`, on folo-data |
| `src/weights/` | `folo weights list \| pull` and published-name resolution (`mod.rs`); `download.rs` fetches a release over HTTPS, else with `gh` |
| `src/cloud.rs` | `folo cloud ...` on the data dir, through `folo_server::cloud` |
| `src/label/` | `folo label ...`: `mod.rs` opens the data dir's `folo serve` (or `--server`, else the data dir's labeler in this process, holding its lock; the store only for `export`) and runs folo-mcp's labeling operations (`folo_mcp::label::Labels`) with CLI hints; `print.rs` is the human output |
| `build.rs` | Bakes `FOLO_GIT_SHA` and `FOLO_BUILD_DATE` into `folo --version` |
| `tests/` | Integration tests that run the built binary; see [Tests](#tests-and-benchmarks) |

### Subcommands

From the `Command` enum in `src/cli.rs` and the `run` match in `src/main.rs`.

| command | implemented in |
|---|---|
| `serve` | `src/serve.rs` |
| `describe` | `src/describe.rs` (printed by `run` in `src/main.rs`) |
| `doctor` | `src/doctor.rs` |
| `train` | `src/train/mod.rs` |
| `stop` | `src/train/stop.rs` (`request_stop`) |
| `val`, `predict`, `export` | `src/infer.rs` |
| `bench` | none: `run` answers `not_implemented` (exit 4) |
| `dataset check \| stats \| synth \| convert` | `src/dataset.rs` |
| `weights list \| pull` | `src/weights/mod.rs`, `src/weights/download.rs` |
| `cloud providers \| connect \| connections \| check \| disconnect \| offers \| instances \| terminate` | `src/cloud.rs` |
| `label agents \| jobs \| create \| add \| start \| stop \| status \| images \| view \| set \| review \| export` | `src/label/`, on folo-mcp's `label` module |
| `mcp` | `src/mcp.rs`, on [folo-mcp](../folo-mcp/README.md) |

### The `--json` contract (`src/output.rs`)

- `--json` is a global flag. With it, `Out::record` writes one compact JSON object per line on stdout and
  `Out::human` writes to stderr; without it, `record` writes nothing and `human` writes to stdout.
- An error is `{"error":{"code","message","hint"}}`. `Out::error` writes it on stdout in `--json` mode, and always
  writes `error: <message>` and `hint: <hint>` on stderr.
- Exit codes (`ExitCode`): 0 `ok`, 1 `runtime_error`, 2 `usage_error`, 3 `not_found`, 4 `not_implemented`.
  `ExitCode::of` takes an error code's exit code from `folo_api::codes`, the vocabulary the CLI shares with the
  HTTP API and MCP.
- A clap parse error is a `usage` error (exit 2); `--help` and `--version` exit 0 (`usage_error` in `src/main.rs`).
- `folo mcp` never writes records on stdout, also with `--json`: stdout belongs to the protocol.

### Features and the cargo alias

- `embed-web` compiles the built Studio into the binary (it turns on `folo-server/embed-web`); see
  [docs/CLI.md](../../docs/CLI.md#the-embedded-studio-embed-web). It is the only feature.
- `.cargo/config.toml` defines `cargo folo` as `cargo run --release -p folo-cli --`.

## Start here

1. `Command` in `src/cli.rs`: every subcommand and its flags, in one file.
2. `run` in `src/main.rs`: which function each subcommand calls.
3. `Out` and `CliError` in `src/output.rs`: how every command reports results and failures.
4. `engine_error` in `src/engine.rs`: how engine errors get their code, exit code and hint.
5. `train` in `src/train/mod.rs` or `val` / `predict` in `src/infer.rs`: a full command on the engine.
6. `acquire` in `src/serve_lock.rs`, then `src/serve.rs` and `src/mcp.rs`: who owns a data dir.

## Common changes

| to do this | touch these files |
|---|---|
| Add a top-level command | A variant of `Command` and its `Args` struct (doc comment on every field) in `src/cli.rs`; a module in `src/` declared in `src/main.rs` and an arm in `run`; a `--json` test in `tests/cli.rs` (or a topic file in `tests/`); the Commands table of [docs/CLI.md](../../docs/CLI.md) |
| Add a flag | The field and its doc comment in the `*Args` struct in `src/cli.rs` (the help text is `folo describe`'s); the command's module; docs/CLI.md. For `train`, `train_defaults_match_the_api` in `src/main.rs` compares defaults with the API's `TrainConfig` |
| Add a `dataset`, `weights` or `cloud` subcommand | The variant in `DatasetCommand`, `WeightsCommand` or `CloudCommand` (`src/cli.rs`); the match in `src/dataset.rs`, `src/weights/mod.rs` or `src/cloud.rs`; `tests/dataset.rs`, `tests/weights.rs` or `tests/cloud.rs`; docs/CLI.md |
| Add an error code | A `c(code, status, exit, meaning)` entry in `CODES` (`crates/folo-api/src/codes.rs`); `CliError::new(ExitCode::<exit>, "<code>", ...)` where it fails. `tests/codes.rs` fails when a code is missing from `CODES` or exits with another code |
| Change what `folo describe` prints | `src/describe.rs`; the `describe_covers_every_command` and `help_states_the_computed_defaults` tests in `src/main.rs` |

## Tests and benchmarks

```sh
cargo test -p folo-cli                      # unit tests and every integration test below
cargo test -p folo-cli --test mcp           # only the folo mcp end-to-end tests
bash tools/test-tmp.sh cargo test -p folo-cli   # as CI runs them: fails if a test leaves temp folders behind
```

| test | covers |
|---|---|
| `src/main.rs`, `src/train/tests.rs`, inline `mod tests` | Unit tests: the clap tree, `train` defaults against the API's `TrainConfig` and `describe` (`src/main.rs`), `--set` and `--resume` (`src/train/tests.rs`), and the modules' own `mod tests` |
| `tests/cli.rs` | Global conventions: JSON lines on stdout, stable exit codes |
| `tests/cli_values.rs` | Multi-value flags in every documented form |
| `tests/studio_cli.rs` | Studio's "CLI command" strings (`web/src/lib/trainCli.fixtures.json`) parse with the real tree |
| `tests/engine.rs` | `train`, `val`, `predict`, `export` and the weights plumbing end to end on a tiny synthetic dataset |
| `tests/dataset.rs` | `dataset synth \| stats \| check \| convert` end to end |
| `tests/weights.rs` | `weights pull` against a release server on loopback, and the `gh` fallback |
| `tests/cloud.rs` | `folo cloud` against a `folo serve --token` child as a `remote` connection |
| `tests/label.rs` | `folo label`'s flags, JSON records and exit codes with a fake `claude` script, in process and through a `folo serve` child (found by its data dir, and with `--server`); every child has a deadline |
| `tests/codes.rs` | Every error code the CLI, server and MCP sources construct is in `folo_api::codes` |
| `tests/version.rs` | Every version in the repository matches the workspace version |
| `tests/onnx.rs` | ONNX export parity against onnxruntime (see below) |
| `tests/mcp/` | `folo mcp` over raw JSON-RPC: `main.rs` (the binary), `harness.rs` (client, schema checks, `ALL_TOOLS`), `protocol.rs`, `real_store.rs`, `proxy.rs`, `confine.rs` |

- No test needs a GPU: `tests/engine.rs` passes `--device cpu` to the engine commands.
- `tests/onnx.rs` needs a Python with `onnx`, `onnxruntime` and `numpy` (`FOLO_ONNX_PYTHON`, else the venv at
  `FOLO_ONNX_VENV`). Without one it prints why and passes, unless `FOLO_ONNX_REQUIRED=1`, which CI's `rust (linux)`
  job sets.
- No test in this crate is `#[ignore]`d.
- The crate has no benchmarks.

## Further reading

- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [docs/CLI.md](../../docs/CLI.md): every command, flag, record and error.
- [docs/MCP.md](../../docs/MCP.md) and [folo-mcp](../folo-mcp/README.md): what `folo mcp` serves.
- [docs/API.md](../../docs/API.md) and [folo-server](../folo-server/README.md): what `folo serve` serves.
- [docs/CLOUD.md](../../docs/CLOUD.md) and [folo-cloud](../folo-cloud/README.md): what `folo cloud` manages.
- [docs/CI.md](../../docs/CI.md): how CI runs these tests.
