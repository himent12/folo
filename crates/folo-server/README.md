# folo-server

The HTTP + SSE server behind FOLO Studio: it implements the contract of [docs/API.md](../../docs/API.md) on axum and
serves the built Studio. Handlers sit on the `Store` trait, with two implementations: `RealStore` (datasets on disk,
training runs, models, inference, cloud runs, agent labeling and vision benchmarks on the FOLO engine) and
`MockStore` (deterministic synthetic data). It depends on folo-api (the wire types), folo-core, folo-gpu, folo-nn,
folo-data and folo-vision (the engine), folo-bench (benchmark conditions and scores), folo-cloud (cloud providers)
and folo-client (to reach cloud workers). folo-cli (`folo serve`, `folo cloud`) and folo-mcp (`folo mcp`) depend on
it.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | `Server` (the router and its middleware: token, origin, `Host`, CORS, compression, logging), `Options`, `WebSource` |
| `src/routes/` | The `/api` router (`mod.rs`: `api()`, `blocking()`, `OriginalUri` in `404`/`405` fallbacks) and handler files grouped by area: `system.rs` (health, system, throughput, benchmark probes), `datasets.rs` (datasets + shared `serve_image` with strong `ETag` / `304` / `412`), `models.rs` (models with conditional `ETag` / `304`, weights downloads, and `POST /api/val`), `runs.rs` (runs + `/runs/estimate`), `inference.rs`, `uploads.rs`, `cloud.rs`, `label.rs`, `bench.rs` |
| `src/store.rs` | The `Store` trait (everything a handler can ask, with `501 not_implemented` defaults for optional parts) and pre-serialized `SseFrame` payloads |
| `src/real/` | `RealStore`: `mod.rs` (the struct, `open_with_cloud`, parallel startup indexing, the data dir layout), `store.rs` (its `Store` impl), `datasets.rs`, `registry.rs`, `deletion.rs`, `training.rs`, `models.rs`, `uploads.rs`, `gpu_idle.rs`, `tests.rs` |
| `src/mock/` | `MockStore`: `store.rs` (its `Store` impl), `catalog.rs` (the seeded data), `runs/` (simulated runs), `scene.rs`, `datasets.rs`, `models.rs`, `val.rs`, `weights.rs`, `throughput.rs`, `cloud.rs` |
| `src/runs/` | The real store's training runs: `mod.rs`, `queue.rs` (FIFO queue, worker thread), `train.rs` (one run on the trainer), `entry.rs` (a run's record and event sink), `history.rs` (typed incremental `metrics.jsonl` cache and line-by-line `/logs` disk reader), `registry.rs`, `resume.rs`, `tests.rs` |
| `src/cloud/` | The cloud control plane (docs/CLOUD.md): `Cloud` in `mod.rs`, `connections.rs`, `ledger.rs`, `reaper.rs`, `worker.rs`, `state.rs`, `files.rs`, and `executor/` (a cloud run's thread: `launch.rs`, `sync.rs`, `train.rs`, `mirror.rs`) |
| `src/label/` | Agent labeling: `Labeler` in `mod.rs`, the `Agent` trait (`agent.rs`), `ask.rs` (one batch's calls: two at most, usage kept whatever the outcome), `faults.rs` (passing faults and limits in a CLI's failure), `claude.rs`, `codex.rs`, `antigravity/` (`mod.rs`, `protocol.rs`, `process.rs`, `acp.rs`, `tests/`), `labeling.rs` (starting and stopping a run), `run.rs`, `prompt.rs`, `job.rs`, `images.rs`, `export.rs`, `dataset.rs`, `tree.rs`, `validate.rs` |
| `src/bench/` | Vision benchmarks: `Benches` in `mod.rs`, `state.rs`, `benchmarks.rs`, `entries.rs`, `model.rs`, `agent.rs`, `external.rs`, `results.rs`, `images.rs` (single-flight rendered-variant and decoded-base-image LRU caches), `source.rs`, `design.rs`, `loops.rs` |
| `src/throughput/` | The measured throughput table and run-time estimates: `mod.rs`, `recording.rs`, `api_table.rs`, `reference.rs`, `run_estimate.rs`, `tests.rs` |
| `src/engine.rs` | Device selection, GPU info, `engine_error` (engine errors to API errors), the single-flight predictor cache of `/api/predict` |
| `src/index.rs` | A dataset's `Index` of images and boxes, its `Fingerprint`, thumbnails (tests in `src/index/tests.rs`) |
| `src/models/` | The real store's models: `mod.rs` (`Models`, bounded LRU caches, `<data dir>/models` scan and `by_id` index, published weights), `meta.rs` (`.folo` header metadata, `ModelRef`, FLOPs), `weights.rs` (cached derived `/weights?format=...` with per-`(model, variant)` locks), `eval.rs` (single-flight bounded `/eval` cache and `POST /api/val`), `export.rs` (`/export` publishing and ONNX export), `tests.rs` |
| `src/uploads/` | Upload sessions (`/api/uploads`): `mod.rs` (sessions into `<data dir>/uploads/<id>/`, condvar-gated expiry sweeper), `fingerprints.rs` (cached `uploads.json` fingerprints), `pipe.rs` (streaming part bridge), `unpack.rs` (tar extraction and path validation), `tests.rs` |
| `src/error.rs` | `Error` (status + `{ "error": { code, message, hint } }`), `ApiJson`, `ApiPath`, `ApiQuery`, `ApiBytes` |
| `src/web.rs` | The Studio compiled in by the `embed-web` feature and how it is served |
| `src/probe.rs` | `POST /api/system/benchmark`: a short training and inference probe |
| `src/cli.rs` | The `folo train` command of a run (`GET /api/runs/{id}/cli`) |
| `src/system.rs`, `src/stats.rs`, `src/raster.rs`, `src/util.rs` | Host CPU and memory (the CPU counters are set up on a thread of their own, so `serve` listens without waiting ~150 ms for them; the first `/api/system` starts a preparation of its slow parts, side by side, that the server owns and every request waits for, so a request that goes away cancels nothing; no CPU reading is measured over less than sysinfo's minimum interval: one asked sooner gives the last again), dataset statistics, the mock's rasteriser, small helpers (`Flight` single-flight deduplication, FNV-1a, RFC 3339 timestamps) |
| `build.rs` | With `embed-web`: writes the table of the built Studio's files (`web/dist` or `$FOLO_WEB_DIST`) |
| `tests/` | Integration tests (see below) |

Every folder's `mod.rs` starts with a doc comment that lists its files.

## Start here

1. `Server::with_embedded` (`src/lib.rs`): how the router is put together around a `Store`.
2. `routes::api` (`src/routes/mod.rs`): every path and the handler it calls; the handler file is named for the
   docs/API.md section.
3. `Store` (`src/store.rs`): what handlers can ask. `folo serve` picks `MockStore::new()` with `--mock`, else
   `RealStore::open(data_dir)` (crates/folo-cli/src/serve.rs).
4. `RealStore::open_with_cloud` (`src/real/mod.rs`): opens the data dir and builds the services the real store uses
   (`Runs`, `Models`, `Uploads`, `Cloud`, `Labeler`, `Benches`). Its module doc lists every file under the data dir
   (`$FOLO_HOME`, else `~/.folo`, unless `--data-dir` names another).
5. `src/runs/mod.rs`: the layout of a run folder and how live events reach SSE subscribers.

## A training run, end to end

1. `POST /api/runs` reaches `create_run` (`src/routes/runs.rs`), which calls `Store::create_run` on the blocking pool.
2. `RealStore::start_run` (`src/real/training.rs`) checks the config against the dataset's index and resolves where
   it trains (`run_settings`): a local device, or a cloud connection when `TrainConfig.compute` is set.
3. `Runs::create` (`src/runs/queue.rs`) registers the run (folder `<data dir>/runs/<id>/`, `run.json`) and queues
   it; `enqueue` starts the `folo-runs` worker thread when it is not running.
4. The worker (`Runs::work`) takes runs one at a time in FIFO order and calls `execute` (`src/runs/train.rs`), which
   builds a folo-vision `Trainer` (or resumes one from `weights/last.folo`) with the run folder and an event sink.
5. The sink is `RunEntry::on_event` (`src/runs/entry.rs`): it updates the run, persists it and sends each event on
   the run's broadcast channel.
6. `GET /api/runs/{id}/events` (`run_events` in `src/routes/runs.rs`) subscribes through `Store::subscribe_sse`
   (`Runs::subscribe_sse`, `src/runs/history.rs`) and streams a `status` event, then shared pre-serialized live frames,
   as SSE (recovering broadcast lag with a warning log and fresh `status` snapshot).
7. The run folder ends up with `run.json`, `log.jsonl`, `config.json`, `metrics.jsonl`, `weights/{last,best}.folo`
   and `samples/epoch_NNN.json`.

A cloud run takes the same first two steps, then `Runs::create_cloud` registers it outside the queue and
`src/cloud/` executes it on a thread of its own through the same `RunEntry::execute`.

## Where other things hang off

- **Cloud, labeling, benchmarks**: `RealStore` holds a `Cloud` (`src/cloud/`), a `Labeler` (`src/label/`) and
  `Benches` (`src/bench/`). Handlers reach the last two through `Store::labeler` and `Store::benches`, and the cloud
  through the cloud methods of `Store` (`cloud_providers`, `cloud_offers`, `terminate_instance`, ...). The mock
  store simulates the cloud (`src/mock/cloud.rs`) and has no labeler and no benches (`501 not_implemented`).
- **The web app**: outside `/api` the router serves `WebSource::Dir` (`folo serve --web-dir`, else `web/dist` when
  it has an `index.html`) or `WebSource::Embedded` (built with the `embed-web` feature, `src/web.rs` and `build.rs`).
  A folder's file is kept for a year only under Vite's hashed name while the folder's `index.html` is a Vite page;
  everything else is `no-cache` (docs/SERVER.md, *Router*).
- **Errors**: handlers return `error::Error`, which answers with its status and the JSON envelope (`IntoResponse` in
  `src/error.rs`). The `Api*` extractors turn axum's rejections into the same envelope, `engine::engine_error` maps
  engine errors, and `routes/mod.rs` answers unknown paths with `route_not_found`. Every code is listed in
  `folo_api::codes::CODES`.

## Common changes

| To do this | Touch these files |
|---|---|
| Add an endpoint | The request / response types in `crates/folo-api/src/<section>.rs`; a `Store` method in `src/store.rs` (a default that answers `501` if not every store has it); its implementations in `src/real/store.rs` and `src/mock/store.rs`, and in `crates/folo-mcp/src/proxy.rs` (`HttpStore`) when it has no default; the handler in `src/routes/<area>.rs` and its path in `routes::api`; tests in `tests/api/` (mock) and the real-store test binary of the area; the section in `docs/API.md`; `web/src/lib/api/types/<section>.ts` |
| Announce an optional API feature | A `FEATURE_*` constant in folo-api; list it in `system` (`src/routes/system.rs`) or in the store's `Store::features` |
| Add an error code | `CODES` in `crates/folo-api/src/codes.rs`; raise it with `Error::new(status, "code", message).hint(..)`; the Errors table in `docs/API.md` (`crates/folo-cli/tests/codes.rs` fails on a code that is not listed) |
| Add a labeling agent | The variant and its names in `LabelAgentId` (`crates/folo-api/src/label.rs`); an `Agent` implementation next to `src/label/claude.rs`, `src/label/codex.rs` and `src/label/antigravity/`; the `agents` list in `Labeler::open` (`src/label/mod.rs`); `AgentArg` in `crates/folo-cli/src/cli/label.rs` and `crates/folo-cli/src/label/mod.rs`; `agent_name` in `apps/desktop/src/screens/bench/format.rs` (an exhaustive match); Studio's `LabelAgentId` in `web/src/lib/api/types/label.ts`, `AGENT_NAME` in `web/src/lib/label.ts` and `DEFAULT_BATCH` in `web/src/lib/api/mock/label.ts`; docs/LABELING.md |
| Add a file under the data dir | The code that writes it, and the data dir list in the module doc of `src/real/mod.rs` |

## Tests and benchmarks

Run them as CI does: `bash tools/test-tmp.sh cargo test -p folo-server` (or `cargo test -p folo-server`). Unit tests
live next to the code (`src/**/tests.rs` and `mod tests` blocks). Each folder under `tests/` is one test binary
(`main.rs` plus topic files); each single file is one binary too. Select one with `--test <name>`.

| Binary | What it covers |
|---|---|
| `api` | Every endpoint in mock mode through `tower::ServiceExt::oneshot`, plus SSE, the real store, static files, CORS and compression |
| `real_runs` | The real store's runs, models and inference on a tiny synthetic dataset (CPU, 64 px); `hardware.rs` trains on a GPU and skips without one |
| `real_datasets`, `issues`, `confined_diagnostics` | The real store's datasets, index cache and re-indexing; `GET /api/datasets/{id}/issues`; failed dataset registration hints honouring the `folo mcp --confine` process-wide read guard (separate binary) |
| `uploads`, `transfer`, `weights` | `/api/uploads`; `folo_client::transfer` against a real server; `GET /api/models/{id}/weights` |
| `cloud`, `mock_cloud`, `worker` | The cloud control plane against in-process workers; the mock's cloud; what a cloud worker serves |
| `label`, `label_live` | Agent labeling with scripted agents; the real Claude Code, Codex and Antigravity CLIs |
| `bench` | Vision benchmarks end to end with tiny datasets, untrained models and scripted agents |
| `token`, `real_system` | `folo serve --token`; `/api/system` `capabilities` (checks the refusal without a GPU) |

The tests in `label_live.rs` are `#[ignore]`d: they need the CLI installed and signed in, and they spend tokens.
Run one with `cargo test -p folo-server --test label_live -- --ignored --nocapture claude` (or `codex` /
`antigravity`). The crate
has no benchmarks; its one example, `examples/label_eval.rs`, measures agent labeling with the real CLIs on 60 COCO
val2017 photos (found, right, IoU, tokens, cost and time per setting: docs/LABELING.md, How good the labels are).

## Further reading

- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [docs/SERVER.md](../../docs/SERVER.md): the server's internals.
- [docs/API.md](../../docs/API.md): the HTTP contract this crate implements.
- [docs/CLOUD.md](../../docs/CLOUD.md), [docs/LABELING.md](../../docs/LABELING.md),
  [docs/BENCH.md](../../docs/BENCH.md): cloud runs, agent labeling and vision benchmarks.
- [folo-api](../folo-api/README.md): the wire types.
