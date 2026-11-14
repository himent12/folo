# folo-mcp

`folo mcp`: a Model Context Protocol server over stdio, so agents can build datasets, train, evaluate, predict and
export through FOLO. The protocol layer is the `rmcp` SDK; this crate adds the FOLO tools, `folo://` resources and
prompts on top of folo-server's `Store` trait. It depends on folo-api (with its `schema` feature), folo-server,
folo-data and folo-client. folo-cli is the only crate that depends on it: `crates/folo-cli/src/mcp.rs` builds the
server from the `folo mcp` flags, and `folo describe` includes the tool schemas. The contract is
[docs/MCP.md](../../docs/MCP.md).

## Where things are

| path | what it holds |
|---|---|
| `src/lib.rs` | `Options` (the flags), `Backend` (`Real`, `Mock`, `Proxy`), `build`, `serve_stdio`, `describe_tools` |
| `src/server.rs` | `FoloMcp`, the rmcp `ServerHandler`: `tools/list` and `tools/call`, resources and subscriptions, prompts, logging, stopping the session's runs on exit |
| `src/tools/` | The tools; see below |
| `src/types.rs` | Argument and result types only MCP has (paging, summaries, `run_status` extras); everything else is a folo-api type |
| `src/resources.rs` | `folo://` URIs (`FoloUri`), the resource listing and templates, JSON snapshots |
| `src/render.rs` | `folo://runs/{id}/samples/{epoch}`: an epoch's validation samples as one annotated JPEG grid |
| `src/prompts.rs` | The prompts `train_on_folder` and `diagnose_run` |
| `src/proxy.rs` | `HttpStore`: the `Store` of `--server`, forwarding the request / response calls to a running `folo serve` through `folo_client::Client`; `subscribe` answers `not_implemented` |
| `src/paths.rs` | `PathPolicy`: relative input paths resolve against `--data-dir`; with `--confine`, canonical paths must stay inside it |
| `src/confine.rs` | `--confine` for datasets: a read guard on every file the dataset reader opens, and the report before a tool reads a dataset |
| `src/errors.rs` | Tool errors: folo-server's error type, plus `invalid_arguments`, `read_only` and `not_implemented` |

### `src/tools/`

One `ToolDef` per tool in `TOOLS` (`src/tools/mod.rs`), dispatched by name in `src/tools/dispatch.rs`.

| file | what it holds |
|---|---|
| `mod.rs` | `TOOLS` (name, description, `mutating`, input and output schema) in docs/MCP.md order; `listed`, `find`, `describe` |
| `dispatch.rs` | `Inner::call`: the `--read-only` refusal, unknown-argument check, the match onto handlers, progress notifications, error-code mapping |
| `schemas.rs` | Input and output schemas generated with schemars, then shrunk for `tools/list` |
| `arguments.rs` | Unknown argument names (with a suggestion), parsing, names and `0..=1` values |
| `results.rs` | Size limits, one-line summaries and pages of a result |
| `info.rs` | `folo_info` |
| `datasets.rs` | `list_datasets`, `get_dataset`, `list_dataset_issues`, `list_dataset_images`, `register_dataset`, `create_synthetic_dataset` |
| `models.rs` | `list_models`, `val`, `predict`, `predict_batch`, `export` |
| `runs.rs` | `train` (and resume), `run_status`, `list_runs`, `stop_run`, `reorder_run`, `compare_runs` |
| `cloud.rs` | `cloud_connections`, `cloud_offers`, `cloud_instances`, `cloud_terminate` |
| `tests.rs` | Unit tests, including the tool count the docs state |

### How the flags are applied

- **Backend.** `build` (`src/lib.rs`) opens a `RealStore` on the data dir (`Backend::Real`), a `MockStore`
  (`Backend::Mock`, `--mock`) or an `HttpStore` (`Backend::Proxy`, `--server`, which must be an `http://` or
  `https://` URL). folo-cli picks the backend, and switches an embedded server to `Proxy` when a `folo serve`
  already owns the data dir and `--confine` is off (`crates/folo-cli/src/mcp.rs`).
- **`--read-only`.** `tools::listed` leaves the `mutating` tools out of `tools/list`, `Inner::call` refuses them
  with the `read_only` error, and `prompts::list` leaves out `train_on_folder`.
- **`--confine`.** `build` installs the dataset read guard (`confine::install_read_guard`) unless the backend is
  `Proxy`, where `folo serve` reads the datasets; `PathPolicy` checks every input path.
- **Schemas.** The tool types derive `schemars::JsonSchema`: those in `src/types.rs` directly, the folo-api types
  through its `schema` feature, which this crate's `Cargo.toml` turns on. `schemas.rs` generates each schema with
  rmcp's `schema_for_input` / `schema_for_output`, removes what does not help validation and marks every object
  `additionalProperties: false`; `train_input` makes `dataset_id` optional when `resume_run_id` is given.

## Start here

1. `build` and `serve_stdio` in `src/lib.rs`: how a session starts and ends.
2. `TOOLS` in `src/tools/mod.rs`: every tool and its types.
3. `Inner::call` in `src/tools/dispatch.rs`: what happens to a `tools/call`.
4. A handler, for example `train` in `src/tools/runs.rs`: a tool on top of the `Store`.
5. `FoloMcp` in `src/server.rs`: the rest of the protocol surface (resources, subscriptions, prompts).
6. `HttpStore` in `src/proxy.rs`: the same `Store` over HTTP.

## Common changes

| to do this | touch these files |
|---|---|
| Add a tool | A `ToolDef` in `TOOLS` (`src/tools/mod.rs`); its argument and result types in `src/types.rs` (or a folo-api type); an arm in `Inner::call` (`src/tools/dispatch.rs`); the handler in `src/tools/<area>.rs`; `ALL_TOOLS` in `crates/folo-cli/tests/mcp/harness.rs`; the Tools table of [docs/MCP.md](../../docs/MCP.md); the "N tools" count in README.md, docs/QUICKSTART.md, docs/MCP.md and llms.txt, and llms.txt's tool list (`docs_name_every_tool` in `src/tools/tests.rs` checks them) |
| A tool that needs a new store operation | Also the `Store` trait (`crates/folo-server/src/store.rs`), folo-server's `RealStore` and `MockStore`, and `HttpStore` in `src/proxy.rs` |
| Add a resource | A `FoloUri` variant with its parse and display, and the listing or template, in `src/resources.rs`; `read_resource` in `src/server.rs`; the Resources section of docs/MCP.md |
| Add an error code | `CODES` in `crates/folo-api/src/codes.rs`; `crates/folo-cli/tests/codes.rs` scans this crate's sources too |

## Tests and benchmarks

```sh
cargo test -p folo-mcp                  # unit tests: schemas, arguments, paging, URIs, paths, an unreachable --server
cargo test -p folo-cli --test mcp       # end to end: spawns `folo mcp` and speaks raw JSON-RPC
```

- The end-to-end tests live in `crates/folo-cli/tests/mcp/` because they run the `folo` binary: `harness.rs`
  (JSON-RPC client, schema checks), `protocol.rs` (each supported revision, size limits), `real_store.rs` (the
  real store on the CPU), `proxy.rs` (against `folo serve --mock`, with a token), `confine.rs` (`--read-only`
  and `--confine`). docs/MCP.md, Tests, lists what they cover.
- Nothing needs a GPU; `real_store.rs` runs with `--device cpu`. No test is `#[ignore]`d. No benchmarks.

## Further reading

- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [docs/MCP.md](../../docs/MCP.md): the contract (tools, resources, prompts, limits, safety).
- [docs/AGENT_COOKBOOK.md](../../docs/AGENT_COOKBOOK.md): recipes for agents.
- [docs/API.md](../../docs/API.md): the HTTP API behind `--server` and the shared types.
- [folo-cli](../folo-cli/README.md), [folo-server](../folo-server/README.md),
  [folo-client](../folo-client/README.md).
