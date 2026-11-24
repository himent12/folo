# folo-server: the HTTP server and its services

`crates/folo-server` (axum + tokio) implements [API.md](API.md): datasets, models, training runs with live SSE
metrics, inference, cloud compute, agent labeling and vision benchmarks. This file covers its internals and how
folo-cli and folo-mcp sit on it. The contracts live elsewhere and are not repeated here: [API.md](API.md) (HTTP),
[CLOUD.md](CLOUD.md), [LABELING.md](LABELING.md), [BENCH.md](BENCH.md) and [MCP.md](MCP.md);
[ARCHITECTURE.md](ARCHITECTURE.md) shows where the crate sits. The crate's file map and its common changes are in
[crates/folo-server/README.md](../crates/folo-server/README.md).

```text
lib.rs             Server: the router and its middleware, Options, AppState, serve
routes/            one module per area of API.md (datasets, models, runs, inference, system, uploads, label, cloud,
                   bench); routes::blocking runs a Store call on tokio's blocking pool
store.rs           the Store trait: everything a handler needs, whatever the data source
real/              RealStore: the glue between the trait and the services below
mock/              MockStore: deterministic data for `folo serve --mock` (mock/runs/ simulates training)
runs/              the run registry, the FIFO queue and its worker thread, history and resume
engine.rs          device selection, GPU info, engine error mapping, the predictor cache of /api/predict
index.rs           dataset indexes and their Fingerprint
models.rs          models: finished runs' weights, <data dir>/models, published weights, /eval, exports
uploads.rs         upload sessions (/api/uploads)
cloud/             the cloud control plane: connections, ledger, reaper, executor/ (one thread per cloud run)
label/             agent labeling: jobs, the Agent trait, claude.rs, codex.rs and antigravity/
bench/             vision benchmarks: storage, entries, scheduling and self-improvement loops
throughput/        the measured throughput table and run-time estimates
cli.rs             the `folo train` command of a TrainConfig (GET /api/runs/{id}/cli)
probe.rs           POST /api/system/benchmark
system.rs, web.rs, error.rs   host information, the embedded Studio (`embed-web`), the error envelope
```

`models.rs` / `real/models.rs` and `uploads.rs` / `real/uploads.rs` are not duplicates: the top-level file is the
service, the one under `real/` connects it to `RealStore`.

## Router
`Server::new(store, options)` builds the router (`Server::with_embedded`), from the inside out:
1. `routes::api()` nested under `/api`, behind the token check (`require_token`) when the server has a token
   (`folo serve --token`);
2. the web app as the fallback: a built `web/dist`-style folder, or the copy compiled in by the `embed-web` feature;
   both cache the same way (`web::cache_control`): Vite's content-hashed `assets/` for a year (`public,
   max-age=31536000, immutable`), and `no-cache` (stored, revalidated before every use) for `index.html`, the SPA
   fallback for any other path (an asset name that is no file included), source maps (any case of `.map`) and the
   unhashed files of `public/`, so a new build shows on the next load. The embedded copy assumes `assets/` holds only
   Vite's hashed output (`web/public` has no `assets/` folder) and revalidates by `ETag`. On the folder, a `200` or
   `206` gets the year only for a plain path (its leading `/`s trimmed: ASCII letters, digits, `.`, `_`, `-` and `/`
   only, no empty, `.` or `..` segment, none ending in `.`), so the file sent is the one the path names, also on a
   case-insensitive volume, whose name is Vite's hashed form (`assets/<name>-<8 characters>.<ext>`), while the
   folder's current `index.html` is a Vite page: a whole module script tag with such a name within its first 64 KiB
   (`web::VITE_HEAD_BYTES`), read only for a hashed name whose file exists. A name merely shaped like a hash, in a
   folder whose page loads such a module, is trusted. Any other file is `no-cache`. Other answers (a redirect to a
   folder's `/`, an error) carry no `Cache-Control`. The folder ignores `If-Modified-Since`, since a date from another file (the
   fallback's, a newer build's before a rollback) could get a `304` for this one;
3. the origin guard, compression (not for `application/octet-stream`: weights files are sent as they are), and CORS
   (loopback origins only unless the server was started to allow any);
4. the Host guard when the server listens on loopback (no bind address given, or a loopback one), so nothing runs for
   a foreign Host;
5. the request log and trace layer (without headers, so a token never reaches the log).

Handlers are thin: they parse the request and call the `Store`, through `routes::blocking` for calls that may touch
the disk or burn CPU. An SSE stream (`GET /api/runs/{id}/events`, `routes/runs.rs`, `run_events`) sends the run's
`status` first, then live events from a broadcast receiver; a client that falls behind gets a `log` warning, the
stream sends a keep-alive comment every 15 s (`KEEP_ALIVE`) and ends when the server shuts down.

```text
  HTTP / SSE client (Studio, Desktop, folo-client, folo mcp --server)
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Middleware (outer → inner, Server::with_embedded in lib.rs)          │
  │  TraceLayer (no headers/query) → log_request → guard_host (loopback) │
  │  → CorsLayer → CompressionLayer (skip octet-stream) → guard_origin   │
  └───────────────────┬──────────────────────────────┬───────────────────┘
                      │ /api/*                       │ /* (fallback)
                      ▼                              ▼
           require_token (--token;             Studio web app
           /api/health & OPTIONS open)         (WebSource::Dir / Embedded)
                      │
                      ▼
           routes::api() handlers
             ├── request/response ──► direct Store call or routes::blocking (spawn_blocking)
             ├── SSE streams (/runs/{id}/events, /label/jobs/{id}/events, /bench/benchmarks/{id}/events)
             │      └── subscribe via Store / service ──► initial snapshot + tokio::sync::broadcast
             │                                            + 15 s KEEP_ALIVE + shutdown watch
             ▼
           Arc<dyn Store> (RealStore / MockStore)
             ├── Datasets  ◄── folo-index
             ├── Runs      ◄── folo-runs (FIFO local trainer)
             ├── Models / Predictors ◄── folo-gpu-idle
             ├── Uploads   ◄── folo-uploads + folo-<upload_id>
             ├── Cloud     ◄── folo-cloud-reaper, folo-cloud-<run_id>,
             │                 folo-cloud-copy-<run_id>
             ├── Labeler   ◄── folo-label-<job_id> (agent CLI calls)
             └── Benches   ◄── folo-bench-models, folo-bench-expiry,
                               folo-bench-agent-<id>, folo-bench-loop-<id>
```

## The Store trait
`store.rs` defines `Store`, the one interface behind every handler. Its methods are synchronous. Implementations:
- `RealStore` (`real/`): the engine (the sections below).
- `MockStore` (`mock/`): deterministic synthetic data for `folo serve --mock`, for UI work before or without a
  trained model.
- `folo_mcp::proxy::HttpStore` (in folo-mcp): forwards the request / response calls to a running `folo serve` over
  HTTP. `subscribe` (a run's event stream) answers `not_implemented`, and it has no labeler or benchmarks.

The services hang off the real store, all built in `RealStore::open_with_cloud`: `Runs` (training), `Models`,
`Predictors`, `Throughput`, `Uploads`, `Cloud`, `Labeler` and `Benches`. The store hands the labeler and the benchmarks
out through `Store::labeler` and `Store::benches`; the mock has neither.

## Datasets, models and inference
- The real store's datasets are read with folo-data and indexed into a JSON cache under the data dir, invalidated by
  file mtimes (`index.rs`, `Fingerprint`: file sizes and modification times under the dataset folder).
- Finished runs' checkpoints and `.folo` files under `<data dir>/models` are its models; `/eval` runs folo-vision's
  validator, cached.
- `/predict` keeps an LRU of loaded, fused `Predictor`s (`engine.rs`, `PREDICTOR_CACHE` models).
- Devices go through folo-vision's `resolve_device` / `can_place` (`auto` is the GPU when folo-gpu finds an adapter,
  else the CPU) and GPU info comes from folo-gpu.
- The request and response types live in `folo-api`, a dependency-light crate (serde and serde_json; schemars behind
  its `schema` feature) that the CLI, folo-mcp and FOLO Desktop reuse.

## Training runs
`runs/` holds the real store's runs:
- A registry persisted under `<data dir>/runs/<id>/`, a FIFO queue and one background worker thread (`folo-runs`,
  started by `Runs::ensure_worker`) that trains one run at a time with folo-vision's `Trainer` (`runs/train.rs`,
  `execute` and `build`).
- Each run folder holds `run.json` (the API `Run` plus what the server needs to restart it), `log.jsonl`, and what the
  trainer writes: `config.json`, `metrics.jsonl`, `weights/{last,best}.folo` and `samples/epoch_NNN.json`. At start a
  folder whose `run.json` cannot be read is skipped with a warning, and so is a copy of a run folder (a `run.json` with
  an id already loaded: the folder named after the id is the run). Labeling jobs (`label/`) and benchmarks (`bench/`)
  skip copies the same way, picked by the id in `job.json` / `bench.json` before anything loads them
  (`util::folders_by_id`): a copy whose record the loader takes wins over an original it refuses. A file in place of
  `runs/`, `label/`, `bench/`, `datasets/`, `models/` or `uploads/` would be read as an empty folder: `folo serve`
  refuses to start instead, naming it (`RealStore::check_layout`).
- The trainer's event sink is `RunEntry::on_event` (`runs/entry.rs`): it updates the run, persists what matters and
  forwards each event to SSE subscribers through a per-run broadcast channel (`Runs::subscribe`, `runs/history.rs`).
- `RunEntry::execute` is the one claim-execute-publish path, shared by local and cloud runs.
- A stop (`Runs::stop`) sets the run's flag, which the trainer checks after each batch and before each epoch, and logs
  `runs::STOP_REQUESTED`. One that comes after the last batch changes nothing (the run ends `completed` with all its
  results), so that line names both outcomes.
- On open, **local** runs a previous process left `queued`, `running` or `stopping` become `stopped`;
  `Runs::resume` continues one from `weights/last.folo`. An active cloud run is left for `crate::cloud` to follow
  again or end (`runs/registry.rs`, `Runs::open`).

## The data dir
`real/mod.rs` documents the layout in full. In short:

| path | what |
|---|---|
| `datasets.json`, `datasets.removed.json` | the dataset registry and the tombstones of unregistered datasets |
| `cache/datasets/<id>.json`, `cache/thumbs/`, `cache/eval/`, `cache/weights/` | dataset indexes, thumbnails, cached `/eval` results, f16 exports |
| `datasets/<name>/` | synthetic datasets made by the server, and finished dataset uploads |
| `runs/<id>/` | training runs |
| `models/`, `exports/` | `.folo` files listed as models; export output |
| `throughput.json` | the measured throughput table |
| `uploads/<id>/`, `uploads.json` | upload sessions in progress |
| `cloud/connections.json`, `cloud/instances/` | cloud connections and the instance ledger |
| `label/<id>/` | labeling jobs |
| `bench/<id>/` | vision benchmarks |
| `serve.lock`, `serve.json` | the lock of the one store that owns the data dir, and the address of the server that holds it |

## Cloud compute
`folo_server::cloud` ([CLOUD.md](CLOUD.md)): a run with `TrainConfig.compute` trains on another machine's GPU and stays
a run of this server.
- The worker is an ordinary `folo serve` reached through `folo-client` with the public API: a token (`--token`),
  uploads in parts (`/api/uploads`, a dataset as the tar stream `folo_data::pack` writes), cursors on a run's history
  and `Range` downloads of its weights are all it needs, so any FOLO server can be one.
- `folo-cloud` holds the providers behind one trait (`Provider`: check credentials, list offers, launch, look up, find
  by name, terminate): `RunPod` rents a pod per run and starts the worker with a bootstrap script; `Remote` is a
  server that already runs.
- `Cloud` is the control plane of a data dir: connections (`cloud/connections.json`, owner-only, never served), the
  instance ledger (`cloud/instances/<id>.json`, written before the provider is called), a reaper thread that retries
  terminations and cleans up after crashes, and one executor thread per cloud run through the phases provisioning →
  booting → syncing → training → collecting → releasing.
- A cloud run shares the run registry, folder and event sink with local runs (`RunEntry::execute` is the one
  claim-execute-publish path) but not the queue or the device; its worker's history is mirrored poll by poll into the
  same `metrics.jsonl`, `log.jsonl` and `samples/`, so everything downstream of a run is unchanged.
- A guard releases the instance on every way out, including a panic; a run that trains survives this server
  (`runs/<id>/cloud.json` and an OS lock on `cloud.lock` let the next process follow it again).
- Tests drive the whole lifecycle against an in-process worker (`tests/cloud/`).

## Agent labeling
`folo_server::label` ([LABELING.md](LABELING.md)): labeling jobs under `<data dir>/label/<id>/` (unlabeled images,
classes, instructions, boxes so far), labeled by a coding-agent CLI on the server's machine and exported as a YOLO
dataset that the store registers.
- `Labeler` is the service behind `/api/label` (the store hands it out through `Store::labeler`; the mock has none).
- Agents sit behind the `Agent` trait (`label/agent.rs`): `ClaudeCode` runs `claude -p`, `Codex` runs `codex exec`,
  and `Antigravity` runs `agy acp` as child processes, one model request per batch of images with tools, config files
  and sessions switched off (and any ACP permission request rejected), so a call costs little more than its images.
- A run thread feeds batches to a few workers, checks each reply (JSON shape, class names, boxes clipped to the image)
  before storing it, and ends the run at once on a failure that another call would repeat. A provider's passing
  fault (overloaded, a 5xx, a dropped stream, a short rate limit: `label/faults.rs`) or an unreadable reply is
  asked once more, a batch making two calls at most whatever the reason (`label/ask.rs`, which benchmark agent
  entries, plan design and the eval use too, with the same pause); `faults.rs` also words a limit's hint from the
  CLI's message and the status it reported.
- Events (`job`, `image`, `removed`, `reload`, `log`: `folo_api::LabelEvent`) fan out to SSE subscribers through a
  per-job broadcast channel, as for runs.
- Tests script the agents; `tests/label_live.rs` (ignored by default) runs the real CLIs against synthetic shapes,
  and `examples/label_eval.rs` measures them on COCO photos.

## Vision benchmarks
`folo_server::bench` ([BENCH.md](BENCH.md)): benchmarks under `<data dir>/bench/<id>/` (the definition with every
image's SHA-256 and truth, the copied images, one file per entry and per loop), made from a dataset split or a
labeling job read through the `Store` and graded by folo-bench. `Benches` is the service behind `/api/bench`
(`Store::benches`; the mock has none); its handlers are in `routes/bench.rs`.
- **Model entries** queue for one worker thread that runs them on the store's inference device with validation's
  predictor settings, a preparing thread making the next batch of variants (decode, condition, canonical JPEG,
  decode) on the rayon pool while one is predicted, each variant graded (to its mAP pair) as soon as its pass ends.
- **Agent entries** ask the labeler's agents (so scripted agents script them too) with the labeling prompt and
  failure policy, a thread each.
- **External entries** come in one request or in parts (an open entry: each part checked whole and taken, repeats of
  the same content ignored, a variant graded and dropped once all its images are in; a sweeper thread fails open
  entries that get no part for 30 minutes).
- Every contestant is graded by the same code (`bench::results`).
- A **self-improvement loop** is a thread of its own that drives everything else through what exists: its entries go
  through the model queue, its datasets and training runs through the `Store` (held weakly, so a loop never keeps the
  store alive), so the student's run is an ordinary run.
- Events (`bench`, `entry`, `loop`, `removed`) fan out through a per-benchmark broadcast channel; shutdown fails the
  active entries and loops and joins their threads.

## Locks, background threads, and shutdown order

### Locks
| lock | kind & location | what it guards |
|---|---|---|
| `<data dir>/serve.lock` | Exclusive OS file lock (`folo-cli/src/serve_lock.rs`) | Single-owner access to a data dir by `folo serve` or embedded `folo mcp` (`serve.json` records the bound address). |
| `<data dir>/cloud/.lock` | Exclusive OS file lock (`cloud/files.rs`) | Read-modify-write serialization of `cloud/connections.json` and `cloud/instances/<id>.json` across `folo serve`, `folo cloud` and in-process threads. |
| `<data dir>/runs/<id>/cloud.lock` | Non-blocking OS file lock (`cloud/files.rs`, `try_lock_run`) | Exclusive ownership of an active cloud run by one executor; `folo-cloud-reaper` skips locked runs and another process returns `409 run_active` on stop. |
| `RealStore::files` | `RwLock<()>` (`real/mod.rs`) | Shared during dataset registration and run creation/resume; exclusive during dataset unregister and `DELETE /api/datasets/{id}?files=true` so tombstones and file deletion cannot race run admission. |
| `RunEntry::state` → `Runs::queue` | `Mutex` lock order (`runs/queue.rs`) | `move_queued` and `stop_as` always lock the run's `state` before `Runs::queue` so queue moves, starts, stops and deletes cannot deadlock. |
| Service registries & slots | `RwLock` + per-item `Mutex` (`real/`, `label/`, `bench/`) | Read locks on `entries`, `jobs` and `benches` are held briefly for lookups to clone an `Arc`; write locks serialize additions/removals (including registry writes). Per-item locks (`Entry::{reg, slot, deleted}` + `Condvar`, `Job`, `Slot`) guard state mutations and atomic saves. |
| Upload sessions | `Mutex` + per-session `Mutex` (`uploads.rs`) | `Uploads::sessions` (`Mutex<HashMap<…>>`) guards session creation (including session folder creation and tar unpacker startup), lookup and expiry; per-session `Session::state` serializes parts of one upload. |

### Background threads
| thread name | started by | lifecycle & job |
|---|---|---|
| `folo-system-probe` | `SystemProbe::default()` (`system.rs`) | One-shot setup of `sysinfo` CPU counters so `folo serve` binds without waiting ~150 ms. |
| `folo-index` | `RealStore::open_with_cloud` (`real/mod.rs`) | Checks fingerprints and loads or rebuilds index caches for registered datasets on startup; exits once done. |
| `folo-runs` | `Runs::ensure_worker` (`runs/queue.rs`) | Spawned lazily on the first local run; trains queued local runs one at a time in FIFO order. |
| `folo-gpu-idle` | `RealStore::open_with_cloud` (`real/gpu_idle.rs`) | Holds `Weak` refs; clears cached predictors after 300 s (`PREDICTOR_IDLE`) and releases GPU pool buffers after 30 s (`GPU_IDLE_RELEASE`) when idle. Exits when `RealStore` drops. |
| `folo-uploads`, `folo-<upload_id>` | `Uploads::open` / `Unpacker::start` (`uploads.rs`) | Sweeper expires idle upload sessions (`UPLOAD_IDLE` = 1 h; holds `Weak` ref). Each dataset tar upload spawns one streaming unpacker thread `folo-<id>` joined on session close/drop. |
| `folo-cloud-reaper`, `folo-cloud-<run_id>`, `folo-cloud-copy-<run_id>` | `Cloud::recover_and_reap` / `executor::spawn` / `Exec::start_copy` (`cloud/`) | Reaper sweeps every 30 s (`reaper_every`) for unfollowed/terminating/unseen instances; one executor thread per active cloud run; optional background checkpoint-download thread (`Exec::start_copy`) during `training`. |
| `folo-label-<job_id>` | `Labeler::start` (`label/labeling.rs`) | One thread per active labeling job, driving `parallel` worker calls in a thread scope and child CLI processes. |
| `folo-bench-models`, `folo-bench-expiry`, `folo-bench-agent-<id>`, `folo-bench-loop-<id>` | `Benches::open` / `entries.rs` / `bench/mod.rs` | Persistent FIFO model-entry worker and open external-entry expiry sweeper (30 min timeout); one thread per active agent entry or self-improvement loop. |

### Shutdown order
When `Ctrl+C` or `SIGTERM` resolves `shutdown_signal()`:
1. `Server::serve` (`lib.rs`) sets the `watch` shutdown channel (`shutdown.send_replace(true)`), ending all open SSE streams, and lets `axum::serve` drain in-flight HTTP requests.
2. `Server::serve` runs `Store::shutdown()` on tokio's blocking pool (`RealStore::shutdown` in `real/store.rs`):
   1. `uploads.shutdown()` — closes and deletes in-progress upload sessions and joins their unpacker threads.
   2. `cloud.shutdown()` — sets `Halt`, lets runs in `training` or `collecting` continue on their workers (earlier phases release their instances), stops `folo-cloud-reaper`, waits up to `cloud::SHUTDOWN_WAIT` (30 s) for tracked executors and the reaper, and joins those that finished (in-flight `folo-cloud-copy-*` threads are cancelled separately and may outlive their executor).
   3. `benches.shutdown()` — sets `closing`, fails open external entries, cancels active entries and loops, drains the model queue, unparks `folo-bench-expiry` and joins benchmark threads (before `runs`, because a loop may stop the run it follows).
   4. `labeler.shutdown()` — cancels active labeling jobs (killing in-flight agent CLI process trees) and joins `folo-label-*` threads.
   5. `runs.shutdown(Some(SHUTDOWN_WAIT))` — rejects new runs (`503 shutting_down`), marks queued runs `stopped` (`interrupted`), signals the active local run to stop after its current batch (keeping its last checkpoint) and waits up to `SHUTDOWN_WAIT` (60 s) for the active-run slot to clear (the `folo-runs` thread's handle is not retained or joined).
   6. `throughput.flush()` — writes buffered throughput measurements to `throughput.json`.
3. Dropping `RealStore` lets weak-referenced sweepers (`folo-gpu-idle`, `folo-uploads`) exit; `Drop for RealStore` (and `Drop for Cloud`) also run `benches`, `labeler`, `cloud` and non-waiting `runs` shutdown as a safety net when a store is dropped without `Server::serve`.

## folo-cli on the server
- `folo serve` (`crates/folo-cli/src/serve.rs`) opens `RealStore` on the data dir (or `MockStore` with `--mock`) and
  runs `Server`. It serves Studio from `--web-dir` when given, else from the copy compiled into the binary by the
  `embed-web` feature (release builds, [RELEASE.md](RELEASE.md)), else from `web/dist`.
- `serve_lock.rs` keeps one owner per data dir: `folo serve` (real store) and an embedded `folo mcp` hold an
  exclusive OS lock on `<data dir>/serve.lock`, and a server writes its local address to `serve.json` (wildcard binds
  `0.0.0.0` and `::` become loopback `127.0.0.1` and `[::1]`, and legacy wildcard records are normalized on read;
  specific binds, including scoped IPv6 addresses, stay as bound).
- `folo cloud` (`cloud.rs`) opens `folo_server::cloud::Cloud` on the data dir directly, so it works with or without a
  server running.
- `GET /api/runs/{id}/cli` comes from `folo_server::cli`, which turns a `TrainConfig` back into `folo train`
  arguments; folo-cli's tests parse the result with the real CLI.

## folo-mcp on the Store
`folo mcp` (`crates/folo-cli/src/mcp.rs`) builds `folo_mcp::FoloMcp` over one of three stores (`folo_mcp::Backend`):
the embedded `RealStore` on the data dir, `MockStore` (`--mock`), or `proxy::HttpStore` (`--server URL`). When an
embedded start finds a `folo serve` that owns the data dir (and `--confine` is off), it goes through that server
instead. Tool calls go through the same `Store` methods as the HTTP handlers, and the tool schemas come from the
`folo-api` types through schemars, so they match the HTTP API. `HttpStore` does not follow event streams: the MCP
layer polls runs instead. The tools, resources and prompts are in [MCP.md](MCP.md).

## Tests
`crates/folo-server/tests/` holds the integration tests, each folder one test binary:
- `api/`: every endpoint of API.md in mock mode through `tower::ServiceExt::oneshot`, plus SSE, the real store,
  static files, CORS and compression;
- `real_runs/`: the real store's runs, models and inference end to end on a tiny synthetic dataset;
- `cloud/`: the cloud control plane against in-process workers;
- `label/` and `bench/`: agent labeling and vision benchmarks over HTTP, with scripted agents;
- single files: `issues.rs`, `mock_cloud.rs`, `real_datasets.rs`, `real_system.rs`, `token.rs`, `transfer.rs`,
  `uploads.rs`, `weights.rs`, `worker.rs` and the ignored `label_live.rs`.
