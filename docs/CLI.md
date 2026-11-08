# The `folo` CLI

`folo` is the single binary for training, evaluation, inference, export, the Studio server and the MCP server. It is
built for people and for agents: every command follows the conventions below, and `folo describe --json` prints the
whole command surface (generated from the clap command tree, so it never drifts from the code).

```sh
cargo run -p folo-cli -- serve --mock     # mock Studio API on http://127.0.0.1:7878
cargo folo --help                         # alias for: cargo run --release -p folo-cli --
folo describe --json                      # every command, argument, type, default and help string
folo doctor                               # version, OS, CPU, memory, web app build
folo --version                            # folo 0.1.0 (1a2b3c4d5e6f 2026-11-08): version, commit, build date
```

`folo --version` prints the workspace version (`[workspace.package] version` in the root `Cargo.toml`, the single
source of truth), the 12-character commit and the UTC build date, baked in by `crates/folo-cli/build.rs`. Outside a
git checkout the commit is `unknown` unless `FOLO_GIT_SHA` is set; `SOURCE_DATE_EPOCH` pins the date for
reproducible builds.

### The embedded Studio (`embed-web`)
Release binaries are built with `cargo build --release -p folo-cli --features embed-web`, which compiles the built
Studio into the binary (about 1 MB), so `folo serve` works anywhere. The feature needs a built `web/dist`
(`npm ci && npm run build` in `web/`), or `FOLO_WEB_DIST` pointing at one (relative paths are resolved against the
repository root); the build fails with a hint otherwise. Because of that, `cargo test --workspace --all-features`
also needs a built web app. Without the feature, `folo serve` serves `web/dist` from disk as before.

## Conventions (every command)

### `--json`
A global flag. With it, **stdout carries only JSON, one object per line** (JSON Lines), and all human text (progress,
tables, hints, logs) goes to **stderr**. Without it, human text goes to stdout and errors to stderr. Logs from the
server (`tracing`) always go to stderr; tune them with `RUST_LOG` (default `info`).

Long-running commands emit one record per event, each with an `"event"` field (e.g. `folo serve --json` prints
`{"event":"listening","url":"http://127.0.0.1:7878",...}` once bound and `{"event":"stopped"}` on shutdown).
One-shot commands print one object.

### Errors
A failing command prints its error as the last stdout line in `--json` mode, and always in human form on stderr:

```json
{"error":{"code":"not_implemented","message":"`folo train` is not implemented yet","hint":"it waits on ..."}}
```

`code` is a stable snake_case identifier (the same vocabulary as the HTTP API's error codes where they overlap),
`message` is for humans, `hint` says what to do next and is `null` when there is nothing to add.

### Exit codes
| code | name | meaning |
|---|---|---|
| 0 | `ok` | success (also `--help` / `--version`) |
| 1 | `runtime_error` | the command ran and failed (IO, bind failure, bad data, ...) |
| 2 | `usage_error` | invalid arguments or flags (error code `usage`) |
| 3 | `not_found` | a named file, dataset, model or run does not exist |
| 4 | `not_implemented` | the command exists but is not implemented yet |

## Commands

| command | status | what it does |
|---|---|---|
| `serve [--mock] [--host 127.0.0.1] [--port 7878] [--web-dir DIR] [--data-dir DIR] [--cors-any]` | works | Runs the HTTP API (`docs/API.md`) and serves the built Studio with an SPA fallback: from `--web-dir` when given, else the Studio embedded in the binary (`embed-web` builds), else `web/dist`. The `listening` record says which (`web_source`: `dir` or `embedded`, `web_dir`, `web_ready`). `--mock` serves deterministic mock data. `--port 0` picks a free port. `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`) holds registered datasets. Ctrl+C shuts down gracefully (open SSE streams are closed first). |
| `describe` | works | Prints the CLI surface as JSON: `global_args`, `conventions` (exit codes, error shape) and `commands` (recursive, each with `path`, `about`, `args` and `subcommands`). Each argument has `name`, `long`, `short`, `positional`, `type` (`string`, `path`, `integer`, `number`, `boolean`, `enum`, `count`), `value_name`, `required`, `multiple`, `default`, `possible_values`, `env` and `help`. Pretty-printed without `--json`, one line with it. |
| `doctor [--web-dir web/dist]` | works | Version (`version`, `git_sha`, `build_date`), OS, CPU name and threads, memory, whether the web app is built (`web_dist`), whether one is embedded (`web_embedded`: `present`, `files`) and which one `folo serve` would use (`web_source`). GPU detection lands with `folo-gpu`. |
| `train --data D [--model folo-n] [--epochs 100] [--batch 16] [--imgsz 640] [--device auto] [--optimizer auto] [--lr0 0.01] [augmentation flags] [--seed 0] [--project runs]` | planned (exit 4) | Arguments mirror `TrainConfig` in `docs/API.md`. |
| `val --model W --data D [--split val] ...` | planned (exit 4) | mAP / precision / recall of weights on a split. |
| `predict --model W SOURCES... [--conf 0.25] [--iou 0.7] [--max-det 300] [--save DIR]` | planned (exit 4) | Inference on images or folders. |
| `export --model W [--format folo-f16\|onnx] [--output FILE]` | planned (exit 4) | Weight export. |
| `bench [--model folo-n] [--mode infer\|train] [--imgsz 640] [--batch 1] [--iters 100] [--warmup 10]` | planned (exit 4) | Speed benchmarks. |
| `dataset check PATH`, `dataset stats PATH`, `dataset synth NAME [--images 1000] [--classes ...]` | planned (exit 4) | Dataset validation, statistics and synthetic generation (`folo-data`). |
| `mcp [--mock]` | planned (exit 4) | Model Context Protocol server over stdio. |

`folo describe --json` is the authoritative list of arguments and defaults.

## Adding a command
1. Add the clap `Args` struct in `crates/folo-cli/src/cli.rs` with doc comments on every field (they become the help
   text and `describe` output) and accurate defaults.
2. Return `Result<(), CliError>`; pick the exit code from the table above. Emit machine records with `Out::record`
   and human text with `Out::human`, never `println!` directly.
3. Add a test in `crates/folo-cli/tests/cli.rs` that runs it with `--json` and checks stdout parses line by line.
