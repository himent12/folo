# FOLO architecture

FOLO ("**F**OLO **O**nly **L**ooks **O**nce" — a play on YOLO) is a vision framework written from scratch in Rust: a
tensor engine with autograd, CPU and GPU kernels, neural-network layers, a YOLO-style real-time object detector, a
training and evaluation pipeline, a server, a CLI, a web studio and a native desktop app on its own UI framework.
This file is the map: how the parts fit, how a request travels through them and which files to open for a change.
Each crate's details are in its deep-dive doc and its README, linked from [The crates](#the-crates).

## Design principles
These are stated in [AGENTS.md](../AGENTS.md) and in the code, and they explain most of the layout.
- **The ML stack is FOLO's own.** Tensors, kernels, autograd, layers, optimizers, the detector, losses, assigners,
  NMS, metrics, augmentation and the GPU kernels are written here; no ML framework is a dependency. Commodity crates
  do commodity jobs (`image` / `zune-jpeg`, `wgpu`, `rayon`, `serde`, `axum` / `tokio`, `clap`).
- **The CPU backend is the reference.** folo-core's CPU kernels define the results; every GPU kernel is tested
  against them within a stated tolerance, and every differentiable op has a finite-difference gradient check.
- **Deterministic by construction.** CPU results are bitwise identical for every thread count, randomness comes from
  a seeded RNG, data-loader batches do not depend on the worker count, resuming a run is bit-exact, and a benchmark
  variant and a synthetic dataset have the same bytes on every machine and in every build.
- **Plain data at the edges.** folo-data does not depend on folo-core: it produces plain buffers that the trainer wraps
  in tensors. folo-vision's detection math (boxes, decode, NMS, metrics, assigner) works on plain slices, so the
  trainer, the predictor, the server and the browser share it.
- **Front ends talk to the HTTP API.** Studio uses only [API.md](API.md). FOLO Desktop uses the same API through a
  local `folo serve`; it links the engine crates only to run a model in process in its Playground and to grade its
  built-in demo benchmarks.
- **Agent-first.** Every command takes `--json`, errors carry stable codes, and `folo mcp` exposes the same `Store`
  the HTTP handlers use ([Agent-first design](#agent-first-design)).

## System diagram
The crates and their cargo dependencies. An arrow points at a direct dependency; not every edge is drawn (the
table in [The crates](#the-crates) lists them all).

```text
 folo-cli ─────────────────────► folo-mcp
 (the `folo` binary)                 │
     │                               │
     ▼                               │
 folo-server ◄───────────────────────┘
     │   │                 │
     │   │                 └──► folo-cloud ──► folo-client ──► folo-api
     │   └──► folo-bench
     │            │                folo-wasm        folo-desktop (apps/desktop/, on folo-ui)
     ▼            ▼                    │                 │
 folo-vision ◄────┴────────────────────┴─────────────────┘
     │
     ├──► folo-data
     ├──► folo-nn ───┐
     └──► folo-gpu ──┴──► folo-core

 Among the edges not drawn: folo-api is a direct dependency of folo-cli, folo-mcp, folo-server, folo-cloud,
 folo-bench, folo-vision, folo-wasm and folo-desktop; folo-wasm and folo-desktop also use folo-nn, folo-gpu,
 folo-data and folo-core. folo-test-support is a dev-dependency only.
```

The running processes. `┄` lines are not cargo dependencies: they are HTTP, SSE, stdio or spawned processes.

```text
 web/ FOLO Studio (in a browser) ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ HTTP + SSE ┄┄┐
   runs models in the page with folo-wasm                        ┊
 apps/desktop/ FOLO Desktop ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ HTTP + SSE ┄┄┤  Desktop can spawn the server it talks to
   runs models in process with folo-vision                       ┊
                                                                 ▼
 an MCP client ┄┄ stdio ┄┄► folo mcp ┄┄ HTTP, with --server ┄┄► folo serve ┄┄ HTTP (folo-client) ┄┄► folo serve
                            (else embedded)                      ┊      ┊                            on a cloud worker
                                                                 ┊      └┄┄ HTTP (folo-cloud) ┄┄► a provider's API
                                                                 └┄┄ spawns ┄┄► claude -p / codex exec (agent labeling)
```

## Repository map

| path | what it is | read more |
|---|---|---|
| `crates/` | the cargo workspace of the engine and services: the 14 crates below | [The crates](#the-crates) |
| `web/` | FOLO Studio: TypeScript + Vite on fw, FOLO's own UI framework (`web/framework/`) | [web/README.md](../web/README.md) |
| `apps/desktop/` | FOLO Desktop, its own cargo workspace: the app and `ui/` (folo-ui, its UI framework) | [apps/desktop/README.md](../apps/desktop/README.md) |
| `design/` | the design system both apps share: `tokens.json` (with the built-in colour themes under `themes`), fonts, the logo, `UNIFIED.md` (screen by screen), `THEMES.md` (themes and the custom theme format, with examples in `themes/` and the shared cases in `theme-cases.json`), `contrast.mjs` (the colour checks) | [design/README.md](../design/README.md) |
| `docs/` | the documentation; `docs/notes/` holds the engineering notebooks, `docs/img/` the README's images | [docs/README.md](README.md) |
| `site/` | the docs website, generated from `docs/*.md`, `README.md` and `examples/*/README.md` | [site/README.md](../site/README.md) |
| `examples/` | the shapes and COCO walkthroughs, and `bench_external.py` (any model on a FOLO benchmark) | [examples/README.md](../examples/README.md) |
| `bench/` | measurement scripts outside cargo: incumbent baselines, fine-tuning comparisons, ONNX parity | [bench/README.md](../bench/README.md), [BENCHMARKS.md](BENCHMARKS.md) |
| `tools/` | CI and development helpers: `file-size.sh`, `test-tmp.sh`, `ci-retry.sh` (+ `ci-retry-test.sh`), `bench-ab.ps1`, `boat.py`, `charts/` | [tools/README.md](../tools/README.md) |
| `scripts/` | `build-wasm.sh` / `.ps1` (folo-wasm into `web/public/folo-wasm/`), `install.sh` / `.ps1` (the CLI from a release) | [scripts/README.md](../scripts/README.md) |
| `.github/` | CI workflows (`ci`, `web`, `wasm`, `desktop`, `site`, `release`), issue and PR templates | [CI.md](CI.md), [RELEASE.md](RELEASE.md) |
| `Cargo.toml`, `.cargo/`, `rust-toolchain.toml`, `rustfmt.toml` | the workspace and its build settings | [AGENTS.md](../AGENTS.md) |
| `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `llms.txt` | rules and entry points for contributors and agents | [AGENTS.md](../AGENTS.md) |

## The crates

| crate | responsibility | depends on | entry points | read more |
|---|---|---|---|---|
| folo-core | tensors, dtypes, CPU kernels, autograd, autocast, RNG, the `Backend` trait | none | `Tensor`, `backend::Backend`, `no_grad`, `autocast`, `rng::Rng`, `testing::grad_check` | [CORE.md](CORE.md), [README](../crates/folo-core/README.md) |
| folo-gpu | wgpu device, buffer pool, WGSL kernels; implements `Backend` | core | `GpuContext`, `GpuBackend`, `init`, `GpuTensor`, `ForwardPlans`, `kernels` | [GPU.md](GPU.md), [README](../crates/folo-gpu/README.md) |
| folo-nn | modules, layers, init, optimizers, schedules, EMA, loss scaling, `.folo` files | core | `Module`, `Forward`, `ConvBnAct`, `optim`, `LrSchedule`, `ModelEma`, `GradScaler`, `io::ModelFile` | [NN.md](NN.md), [FORMAT.md](FORMAT.md), [README](../crates/folo-nn/README.md) |
| folo-data | image IO, YOLO / COCO datasets, augmentation, the data loader, synthetic data | none | `open`, `open_with`, `Dataset`, `DataLoader`, `AugmentConfig`, `synthetic`, `pack` | [DATA.md](DATA.md), [README](../crates/folo-data/README.md) |
| folo-vision | the detector, loss, assigner, box ops, NMS, metrics, trainer, predictor, ONNX export | api, core, data, gpu, nn | `FoloModel`, `detection_loss`, `Evaluator`, `Trainer`, `Predictor`, `export::onnx` | [VISION.md](VISION.md), [MODEL.md](MODEL.md), [README](../crates/folo-vision/README.md) |
| folo-bench | stress conditions, plans and scores of vision benchmarks | api, core, data, vision | `Condition`, `plan`, `score`, `ENGINE_VERSION` | [BENCH.md](BENCH.md), [README](../crates/folo-bench/README.md) |
| folo-wasm | detectors in the browser (wasm-bindgen) | api, core, data, gpu, nn, vision | `FoloDetector` (JS), `detector::Detector` | [WASM.md](WASM.md), [README](../crates/folo-wasm/README.md) |
| folo-api | the serde types of API.md and the error codes, shared by every service and Desktop | none | `TrainConfig`, `Run`, `RunEvent`, `codes::CODES` | [API.md](API.md), [README](../crates/folo-api/README.md) |
| folo-server | the HTTP + SSE server, the `Store` trait and the services behind it | api, bench, client, cloud, core, data, gpu, nn, vision | `Server`, `Store`, `RealStore`, `MockStore` | [SERVER.md](SERVER.md), [README](../crates/folo-server/README.md) |
| folo-client | blocking HTTP client of API.md; transfers in parts | api | `Client`, `transfer` | [SERVER.md](SERVER.md), [README](../crates/folo-client/README.md) |
| folo-cloud | cloud compute providers behind one trait | api, client | `Provider`, `providers()`, `runpod`, `remote`, `bootstrap` | [CLOUD.md](CLOUD.md), [README](../crates/folo-cloud/README.md) |
| folo-mcp | the MCP server (rmcp, stdio) over the server's `Store`; the labeling operations `folo label` shares | api, client, data, server | `build`, `serve_stdio`, `Backend`, `proxy::HttpStore`, `label::Labels` | [MCP.md](MCP.md), [README](../crates/folo-mcp/README.md) |
| folo-cli | the `folo` binary | api, client, data, gpu, mcp, nn, server, vision | `cli::Command`, `main.rs` (`run`) | [CLI.md](CLI.md), [README](../crates/folo-cli/README.md) |
| folo-test-support | test-only helpers (a dev-dependency) | none | `TestDir`, `TestFile`, `ChildGuard`, `FOLO_TEST_TMP` | [CI.md](CI.md#test-temp-dirs), [README](../crates/folo-test-support/README.md) |

FOLO Desktop (`apps/desktop/`, crate `folo-desktop`) depends on folo-ui, folo-api, folo-bench, folo-core, folo-data,
folo-gpu, folo-nn and folo-vision by path.

## How things flow
Each step names the file and the function or type that does it. Paths are under `crates/` unless they start with
`web/`.

### `folo train`
1. `folo-cli/src/main.rs`: `parse` (also records which flags were given explicitly), then `run` →
   `train::train` (`folo-cli/src/train/mod.rs`).
2. `train::start`: `open_dataset` (`engine.rs`), `config_of` + `apply_sets` (`train/config.rs`), `cfg.validate()`
   before the run folder is created, then `reserve_dir` (`<project>/<name>`, or `<name>2`, ...), which takes the
   folder's `train.lock` and writes `folo-train.json` (`train/stop.rs`: what `folo stop` checks).
3. `TrainOptions { run_dir, events: progress_sink(..) }` → `folo_vision::Trainer::new`
   (`folo-vision/src/train/setup.rs`) → `Trainer::run` (`train/run.rs`), which runs the epochs (`train/epoch.rs`)
   and writes `config.json`, `metrics.jsonl`, `weights/{last,best}.folo` and `samples/` (`train/writer/`).
4. Every `RunEvent` reaches `progress_sink` (`folo-cli/src/train/progress.rs`): a progress view, or JSON lines with
   `--json`; `finish` prints the summary.

### `folo predict`
1. `main.rs` `run` → `infer::predict` (`folo-cli/src/infer.rs`).
2. `open_model`: resolves the weights (`weights::resolve`), the device (`resolve_device`, `can_place`) and the
   precision, starts the GPU precompile (`folo_vision::precompile`), then `FoloModel::from_file_for_inference`.
3. `Predictor::new(model, PredictConfig { .. })` (`folo-vision/src/predict.rs`).
4. Per image: `folo_data::read_image` → `Predictor::predict` → optional `annotate` and `write_new` → a `prediction`
   record; a `done` record at the end.

### A training run started from Studio
1. `web/src/lib/api/client.ts` `createRun` → `POST /api/runs`.
2. `folo-server/src/routes/runs.rs` `create_run` (also handles `resume_run_id` and `checkpoint`) → `Store::create_run`
   on the blocking pool.
3. `RealStore::create_run` (`real/store.rs`) → `RealStore::start_run` (`real/training.rs`: checks, then the target)
   → `Runs::create` (`runs/queue.rs`): register the run, enqueue it, start the worker if needed.
4. The `folo-runs` worker thread (`Runs::work`) takes the run, `RunEntry::claim` and `RunEntry::execute` →
   `runs/train.rs` `execute` → `build` → `Trainer::new` (or `Trainer::resume` from `weights/last.folo`) with
   `events` set to `RunEntry::on_event` → `Trainer::run`.
5. `RunEntry::on_event` (`runs/entry.rs`) updates and persists the run and sends each event on the run's broadcast
   channel.
6. The page follows: `client.ts` `runEvents` → `openEventStream` (`web/src/lib/api/sse.ts`, `fetch` read as SSE, reopened when silent for 45 s) →
   `GET /api/runs/{id}/events` → `routes/runs.rs` `run_events` → `Runs::subscribe` (`runs/history.rs`): the run's
   `status` first, then live events. `web/src/lib/api/live.ts` keeps the open run's view current from them.

### An MCP tool call
1. `folo mcp` (`folo-cli/src/mcp.rs`) picks a `folo_mcp::Backend`: `Real` (embedded), `Mock` or `Proxy` (`--server`);
   an embedded start becomes `Proxy` when a `folo serve` already owns the data dir (`serve_lock.rs`) and `--confine`
   is off. `folo_mcp::build` → `serve_stdio`.
2. `FoloMcp::call_tool` (`folo-mcp/src/server.rs`) → `Inner::call` (`tools/dispatch.rs`: read-only and
   unknown-argument checks, then a match on the tool name) → for `train`, `Inner::train` (`tools/runs.rs`) →
   `Store::create_run` on the blocking pool.
3. Embedded, that is `RealStore::create_run` and the rest of the Studio flow from step 3; a local run is remembered
   and stopped when the client disconnects. With `--server`, `proxy::HttpStore::create_run` (`folo-mcp/src/proxy.rs`)
   sends `POST /runs` through folo-client to the running server, whose runs outlive the session.

### In-browser inference
1. `scripts/build-wasm.sh` builds `crates/folo-wasm` into `web/public/folo-wasm/`; a server whose web app includes
   it lists the `web_wasm` feature in `GET /api/system`.
2. Studio's Playground (`web/src/routes/Playground.tsx`) uses `BrowserEngine` (`web/src/lib/inference/browser.ts`),
   which imports `folo-wasm/folo_wasm.js` and gets the weights with `client.ts` `modelWeights`
   (`GET /api/models/{id}/weights?format=f16`).
3. `FoloDetector.load` (`folo-wasm/src/lib.rs`) → `Detector::load` (`detector.rs`): parse the `.folo` file, build the
   fused eval-mode model, place it on WebGPU or the CPU (`auto` checks WebGPU with one small forward pass).
4. `FoloDetector.detect` → `Detector::detect`: folo-vision's `Predictor` steps (letterbox, forward, decode, NMS,
   unmap), with the GPU readback awaited (`folo_gpu::download_async`); the result is `/api/predict`'s response shape
   plus `backend`.

### A cloud run
1. `POST /api/runs` with `compute` → `RealStore::start_run` → `run_settings`: `cloud::check_device` and
   `Cloud::target` (`folo-server/src/cloud/mod.rs`) choose the connection and the offer.
2. `Runs::create_cloud` registers the run, claimed and outside the queue → `RealStore::start_cloud` → `Cloud::start`
   (takes the run's `cloud.lock`) → `executor::spawn` (`cloud/executor/mod.rs`): a `folo-cloud-<id>` thread runs
   `Exec::drive` inside `RunEntry::execute`.
3. `Exec::start` (`executor/launch.rs`): `open_dataset` (before anything is rented) → `provision` (a ledger record,
   then the provider's launch) → `boot` (the instance, its worker, the worker check) → `sync` (`executor/sync.rs`:
   dataset and starting weights) → `start_worker_run` → `train` (`executor/train.rs`: the worker's run mirrored into
   the local run folder, then `collect`, `executor/mirror.rs`).
4. A `Release` guard terminates the instance through the ledger on every way out, except one: when the server shuts
   down while the worker already trains, the instance keeps running and the next `folo serve` on the data dir
   follows the run again (`Exec::drive`, `executor/mod.rs`). The run's events reach Studio exactly as for a local
   run.

## Where do I change ...?
Paths are under `crates/` unless they start with `web/`, `apps/` or `site/`. Each row follows an existing example.

| task | change | test | docs |
|---|---|---|---|
| add a tensor op | `folo-core/src/ops/<area>.rs` (the `Tensor` method, its checks and its backward), `backend.rs` (a `Backend` method or an op enum such as `UnaryOp`), the CPU kernel in `folo-core/src/cpu/`; for the GPU, `folo-gpu/src/backend/ops.rs` and `convert.rs` | a finite-difference check in `folo-core/tests/grad.rs`; `tests/determinism.rs` for a reduction or scatter; `folo-gpu/src/tests/backend/` against the CPU | [CORE.md](CORE.md), [GPU.md](GPU.md) |
| add a GPU kernel | `folo-gpu/src/shaders/<name>.wgsl`, a `KernelDef` (listed in the module's `DEFS`) and entry point in `folo-gpu/src/kernels/<area>.rs`, a `GpuTensor` method if tensor-level | `folo-gpu/src/tests/<area>.rs` against a CPU loop | [GPU.md](GPU.md#adding-a-kernel) |
| add a layer | `folo-nn/src/layers/<file>.rs` (`Module`, plus `Forward` for one-in, one-out), re-exported from `layers/mod.rs` and `lib.rs` | `folo-nn/tests/modules.rs` | [NN.md](NN.md) |
| add an optimizer | `folo-nn/src/optim/<name>.rs` (as `sgd.rs`, `adamw.rs`, `musgd.rs`) and `OptimKind` in `optim/mod.rs`; to make it selectable: `Optimizer` / `ResolvedOptimizer` in `folo-api/src/runs.rs`, `OptimizerChoice` in `folo-vision/src/train/config.rs` and the constructor match in `train/setup.rs`, the CLI's value in `folo-cli/src/cli.rs` and its match in `config_of` (`folo-cli/src/train/config.rs`), `train_command` in `folo-server/src/cli.rs`, Studio's `web/src/lib/api/types/runs.ts`, `web/src/routes/run/resolved.ts` and `web/src/routes/runnew/`, Desktop's exhaustive matches in `apps/desktop/src/api/types.rs`, `apps/desktop/src/api/mock/training.rs`, `apps/desktop/src/screens/runs/config.rs` and `charts.rs`, and `OPTIMIZERS` in `apps/desktop/src/app/forms.rs`; its state tensor names in `folo-nn/src/io/checkpoint.rs`; the exhaustive optimizer match in `apps/desktop/src/overlays/new_run.rs` | `folo-nn/tests/optim.rs`; also update the exhaustive optimizer matches in `folo-cli/tests/studio_cli.rs`, `folo-nn/tests/folo_io.rs` and `folo-nn/tests/gpu/main.rs` | [NN.md](NN.md), [FORMAT.md](FORMAT.md) (`folo.optim`), [MODEL.md](MODEL.md) §8.2 |
| add an augmentation | `folo-data/src/augment/` (the transform, its knob in `AugmentConfig` and `validate`, the chain in `pipeline.rs`); `Augment` in `folo-api/src/runs.rs`, `AugmentConfig` use in `folo-vision/src/train/config.rs`, the flag in `folo-cli/src/cli.rs` and `train/config.rs`, `folo-server/src/cli.rs` | `folo-data/tests/augment.rs`; `train_defaults_match_the_api` in `folo-cli/src/main.rs` | [DATA.md](DATA.md), [API.md](API.md), [CLI.md](CLI.md) |
| add a dataset format | a reader in `folo-data/src/dataset/` (as `yolo.rs`, `coco.rs`), `Format` and `detect_format` in `dataset/mod.rs`, `check.rs`, `convert.rs`; `DatasetFormat` in `folo-api/src/datasets.rs`; `FormatArg` in `folo-cli/src/cli.rs` with `format_of` and `format_name` in `folo-cli/src/dataset.rs`; in folo-server, `data_format` in `index.rs` and `runs/mod.rs`, the `ds.format` match in `index.rs` (its `_ => Yolo` arm would label a new format YOLO) and the format match in `cli.rs`; the `DatasetFormat` match in `server_cli_commands_train_the_servers_config` (`folo-cli/src/train/tests.rs`); for a format that can be written, `TargetFormatArg` in `folo-cli/src/cli.rs` and its match in `convert` (`folo-cli/src/dataset.rs`); `Label for DatasetFormat` in `apps/desktop/src/api/types.rs`; Studio's `DatasetFormat` in `web/src/lib/api/types/datasets.ts` | `folo-data/tests/formats.rs` | [DATA.md](DATA.md), [API.md](API.md) |
| change the detector | `folo-vision/src/model/` (`blocks.rs`, `head.rs`, `mod.rs`); the ONNX graph mirrors `forward` in `export/onnx/blocks.rs` | `folo-vision/tests/model.rs`; ONNX parity `folo-cli/tests/onnx.rs` | [MODEL.md](MODEL.md) (normative), [VISION.md](VISION.md) |
| change the loss or assigner | `folo-vision/src/loss/` and `assign.rs`; the GPU path in `folo-gpu/src/detect.rs`, `kernels/detect.rs` and the `det_*` / `tal_*` shaders | `folo-vision/tests/loss_cases.rs`, `assign_cases.rs`, `gpu_loss.rs` | [MODEL.md](MODEL.md) §7, [VISION.md](VISION.md) |
| change the `.folo` format | `folo-nn/src/io/` (`safetensors.rs`, `meta.rs`, `file.rs`, `checkpoint.rs`, `export.rs`); the trainer's side in `folo-vision/src/train/checkpoint.rs` | `folo-nn/tests/folo_io.rs` | [FORMAT.md](FORMAT.md), [NN.md](NN.md) |
| add a metric | `folo-vision/src/metrics.rs` (`Evaluator`), `cocoeval.rs` for COCOeval numbers | `folo-vision/tests/metrics_known.rs`, `coco_parity.rs` | [METRICS.md](METRICS.md) |
| add a CLI command | a variant and `Args` in `folo-cli/src/cli.rs` (its help is the `folo describe` output), dispatch in `main.rs` `run`, a module of its own | the name in `describe_covers_every_command` (`main.rs`); `folo-cli/tests/cli.rs` | [CLI.md](CLI.md#adding-a-command) |
| add an HTTP route | the wire type in `folo-api` (a `FEATURE_*` constant in `system.rs` if clients must detect it, listed in `routes/system.rs`); the handler in `folo-server/src/routes/<area>.rs`, wired in `routes/mod.rs`; a `Store` method in `store.rs` with `real/store.rs` and `mock/store.rs`; `HttpStore` in `folo-mcp/src/proxy.rs` must implement a `Store` method that has no default (for one with a default, forward it there only when MCP needs it); Studio: `web/src/lib/api/client.ts`, `types/`, `mock/`, `share.ts`; Desktop: `apps/desktop/src/api/mod.rs`, `http/mod.rs`, `mock/backend.rs`, and a new type in the `pub use folo_api::{…}` lists of `types.rs`, which re-exports by name. `folo-client::Client` has generic request methods; upload-protocol changes also touch `folo-client/src/transfer.rs` | `folo-server/tests/api/` | [API.md](API.md) |
| add an SSE run event | a `RunEvent` variant in `folo-api/src/run_events.rs` (`name`, `data_json`, `from_sse`); emit it from `folo-server/src/runs/` and `mock/runs/events.rs`; the exhaustive matches in `runs/entry.rs` and `folo-cli/src/train/progress.rs`; `RUN_EVENTS` in `web/src/lib/api/sse.ts` and `types/runs.ts`; `apps/desktop/src/store/live.rs` | `folo-server/tests/api/` (SSE) | [API.md](API.md) |
| add an error code | `c(code, status, exit, meaning)` in `CODES` (`folo-api/src/codes.rs`); raise it with `Error::new(status, code, message).hint(..)` | `folo-cli/tests/codes.rs` fails until it is listed | [API.md](API.md#errors), [CLI.md](CLI.md) |
| add an MCP tool | a `ToolDef` in `TOOLS` (`folo-mcp/src/tools/mod.rs`), its types in `types.rs` or `folo-api`, a match arm in `tools/dispatch.rs`, a handler in `tools/<area>.rs`; `ALL_TOOLS` in `folo-cli/tests/mcp/harness.rs` | `folo-mcp/src/tools/tests.rs` checks the tool count and list stated in README.md, QUICKSTART.md, MCP.md and `llms.txt`; `folo-cli/tests/mcp/` compares `tools/list` with `ALL_TOOLS` | [MCP.md](MCP.md) |
| change the labeling tools or `folo label` | the operations in `folo-mcp/src/label/` (`ops.rs`; `compact.rs` for boxes in pixels of a view; `view.rs` for the drawn view; `hints.rs` for each front end's hints), the tool handlers in `folo-mcp/src/tools/label.rs` and types in `types/label.rs`, the CLI in `folo-cli/src/cli/label.rs` and `folo-cli/src/label/`; a new server filter or check in `folo-server/src/label/images.rs` and `routes/label.rs`, with a `FEATURE_*` and the fallback in `Labels` (`label/mod.rs`) | `folo-mcp/src/tools/label_tests.rs`, `folo-cli/tests/label.rs`, `folo-server/tests/label/` | [MCP.md](MCP.md), [CLI.md](CLI.md#labeling), [LABELING.md](LABELING.md) |
| add a cloud provider | a type implementing `Provider` in `folo-cloud`, listed in `providers()` (`lib.rs`); mock offers in `folo-server/src/mock/`; Desktop's demo list in `apps/desktop/src/api/mock/cloud.rs` | the provider id lists asserted in `folo-cloud/src/lib.rs`, `folo-cli/tests/cloud.rs`, `folo-server/tests/mock_cloud.rs` | [CLOUD.md](CLOUD.md#adding-a-provider) |
| add a labeling agent | a `LabelAgentId` variant with its entries in `ALL`, `as_str` and `display_name` (`folo-api/src/label.rs`); an `Agent` (`folo-server/src/label/agent.rs`) as `claude.rs` / `codex.rs`, added in `label/mod.rs`; Desktop's exhaustive `agent_name` in `apps/desktop/src/screens/bench/format.rs`; Studio's `LabelAgentId` in `web/src/lib/api/types/label.ts`, `AGENT_NAME` in `web/src/lib/label.ts` and `DEFAULT_BATCH` in `web/src/lib/api/mock/label.ts` | `folo-server/tests/label/` (scripted agents) | [LABELING.md](LABELING.md), [API.md](API.md) |
| add a benchmark condition | `Condition` and its parameter table in `folo-bench/src/conditions/mod.rs`, `transform.rs`, `catalog.rs`; bump `ENGINE_VERSION` (`lib.rs`); the mock catalogs in `web/src/lib/api/mock/benchCatalog.ts` and `apps/desktop/src/api/mock_bench/` | `folo-bench/tests/golden.rs`; `conditions/tests.rs` checks BENCH.md's table; update the condition-count and default-plan assertions in `folo-bench/src/plan.rs`, `folo-server/tests/bench/benchmarks.rs`, `web/src/lib/api/mock/bench.test.ts`, `web/src/lib/bench/plan.test.ts`, `web/src/routes/bench/results.test.tsx`, and the ignored `real_server_bench_calls` test in `apps/desktop/src/api/http/mod.rs` | [BENCH.md](BENCH.md) |
| add a Studio screen | `RouteName` and its pattern in `web/src/lib/router.ts`, a line in `ROUTES` (`web/src/routes.ts`), `web/src/routes/<Name>.tsx`; `web/src/lib/shell/nav.ts` for a top-level entry | `web/src/routes/<Name>.test.tsx` | [web/README.md](../web/README.md), `design/UNIFIED.md` |
| add a Desktop screen | `Route` (with `parse` and `section`) in `apps/desktop/src/app/mod.rs`, `FoloApp::page` in `app/page.rs`, the exhaustive `breadcrumbs` match in `app/titlebar.rs`, a module in `apps/desktop/src/screens/` | a `--scenario` state in `apps/desktop/src/scenario.rs` and its screenshot (`apps/desktop/tools/shoot.ps1`) | [apps/desktop/README.md](../apps/desktop/README.md), `design/UNIFIED.md` |
| add a theme | an entry in `themes` of `design/tokens.json` (id, name, mode, description, the colours it overrides); `node design/contrast.mjs` until every check passes; `npm run tokens` in `web/` (Studio's `tokens.css` / `tokens.ts`, and the data line of `web/index.html`'s first-paint script); Desktop picks it up through `apps/desktop/build.rs` at the next build | `design/contrast.mjs` checks it; the built-in id lists in `design/contrast.test.mjs`, `web/src/lib/prefs.themes.test.ts`, `web/src/routes/settings/Appearance.test.tsx` and `apps/desktop/src/theme/catalog.rs`; `npm run check` fails while the first-paint data is stale and `firstPaint.test.ts` holds it to `tokens.ts`; the rest of Studio's (`web/src/lib/design/themes.test.ts`, `web/scripts/themes.test.mjs`) and Desktop's (`apps/desktop/src/theme/`, `app/theme_tests.rs`) theme tests run unchanged | `design/THEMES.md` (the table of built-in themes) |
| add a docs page | `docs/<NAME>.md`, its entry in `NAV` (`site/scripts/config.ts`), the docs table of `README.md`, `docs/README.md`, `llms.txt` | `npm run check` in `site/` (links, anchors, images) | [site/README.md](../site/README.md) |

## The engine crates

### folo-core
The tensor engine: `Tensor` is a cheap `Arc` handle over shared storage plus a `Layout`; views do not copy; ops return
`Result` with errors that name the op and both shapes. Kernels sit behind the `Backend` trait, and the CPU backend
(BLIS-style GEMM, im2col-free convolution, Winograd for 3x3 inference convs, a buffer cache, a dedicated rayon pool)
is built in. Autograd is define-by-run with `Grads` keyed by tensor id; `autocast` adds f16 mixed precision on
backends that compute in f16.
- **Backend integration contract, in short:** implement `Backend` and register it with `register_backend`; only
  `name` and the memory methods are required, every compute op and fused hook defaults to `NotSupported` so a
  backend grows op by op, and the CPU backend is the reference each GPU kernel is tested against. Full text:
  [CORE.md](CORE.md#backend-integration-contract).
- Determinism, autograd, in-place updates, custom ops and the RNG: [CORE.md](CORE.md).

### folo-gpu
The wgpu backend ([GPU.md](GPU.md)).
- One `GpuContext` per adapter (Vulkan / DX12 / Metal natively; WebGPU in the browser through wasm), with a buffer
  pool in size classes to avoid allocation churn and command batching, so a forward pass is a few submissions, not
  hundreds. Forward plans (`folo_gpu::ForwardPlans`) record an inference call once (a forward, or the whole device
  side of candidate-first prediction) and replay it as a flat list of dispatches over buffers the plan owns,
  bit-identically, retired when a listed tensor changes; `Predictor` turns them on for its model.
- Kernels in WGSL: tiled GEMM (shared-memory tiles, vec4 loads), convolution as implicit GEMM (no im2col buffer) with
  direct depthwise kernels, elementwise + fused activations, reductions, softmax, pooling, upsample, concat. Each
  kernel is tested against the CPU reference. f16 storage and compute when the adapter supports `shader-f16`, with
  f32 accumulation.
- `folo_gpu::GpuBackend` implements folo-core's `Backend`: `folo_gpu::init()` registers the default adapter as
  `Device::Gpu(0)`, after which `tensor.to_device(Device::Gpu(0))` keeps data and every op (forward and the gradient
  ops autograd records) on the GPU. It also implements the fused hooks: Conv + bias + activation in the conv epilogue,
  train-mode BatchNorm + activation forward / backward kernels, and in-place optimizer updates. Work is encoded into
  the context's open batch and only waits on readback, so a forward + backward pass is a handful of queue
  submissions. Every dtype is stored (`u8` / `f16` packed into words, `i64` as word pairs); strided views are read in
  place by elementwise ops, copies and matmul.
- A standalone `GpuTensor` over the same kernel layer stays for kernel-level work. Kernel list, layout conventions,
  the backend's op mapping, GEMM tuning notes and benchmarks: [GPU.md](GPU.md).

### folo-nn
Layers and training utilities ([NN.md](NN.md)).
- `Module` trait: `parameters()` / `named_parameters()` / `named_buffers()` (stable, dotted names like
  `backbone.stem.conv.weight`), `train()` / `eval()` mode, a submodule visitor, `to_device()` (in place, identities
  kept; optimizers and EMAs have their own `to_device`); `Forward` adds `forward` for one-in, one-out layers.
- Layers: `Conv2d` (groups, stride, padding, dilation), `BatchNorm2d` (running stats, momentum; fused into the conv
  for inference), `Linear`, activations (SiLU, ReLU, LeakyReLU, Sigmoid, Hardswish, ...), `MaxPool2d`, `Upsample`
  (nearest), `Concat`, `Sequential`.
- Init: Kaiming / Xavier uniform / normal, seeded. Optimizers: SGD (momentum, Nesterov, weight decay), AdamW and
  MuSGD, with parameter groups (no decay for BN and bias). Schedules: warmup + cosine / linear, EMA of weights.
- `.folo` weight files ([FORMAT.md](FORMAT.md)): a safetensors-compatible layout (u64 header length, JSON header, raw
  little-endian tensor data) with a `__metadata__` block holding the model config, class names and training info, so
  standard safetensors tools can read FOLO weights.

### folo-data
The data pipeline ([DATA.md](DATA.md): formats, augmentation parameters, determinism).
- Formats: YOLO txt (`images/` + `labels/`, `data.yaml`), COCO JSON; auto-detected. Dataset statistics (class counts,
  box-size distribution, empty images, broken labels) for Studio and `folo dataset stats`; a line-level checker
  (`check`) for `folo dataset check` and YOLO / COCO conversion (`convert`).
- Image IO through `image` / `zune-jpeg`, decoded straight to `u8` RGB. Letterbox resize with exact box remapping.
- Augmentation (all box-aware, seeded): mosaic (4-image), random affine (scale / translate / shear / rotate), HSV
  jitter, horizontal flip, mixup, copy-paste later. Closing-mosaic for the last N epochs.
- `DataLoader`: worker threads that decode + augment ahead, a bounded channel, batches as contiguous `NCHW f32`
  (`0..1`) plus targets `[batch_idx, class, cx, cy, w, h]` (normalised). folo-data does not depend on folo-core; the
  trainer wraps the buffers in tensors.
- `synthetic`: procedurally generated shapes datasets (circles / squares / triangles / stars on textured backgrounds,
  random colour, scale, occlusion) for tests, demos and benchmarks with no downloads.

### folo-vision
The detector ([VISION.md](VISION.md) for the implementation, [MODEL.md](MODEL.md) for the normative spec). A CSP-style
backbone, SPPF, C2PSA, a PAN-FPN neck and an anchor-free decoupled head on strides 8 / 16 / 32, per scale
`n/s/m/l/x`; `head = "o2o"` adds the NMS-free one-to-one branch. The loss is TAL + BCE + CIoU + DFL, on folo-core ops
on the CPU and as folo-gpu kernels on a GPU. Box ops, NMS and metrics work on plain slices
([METRICS.md](METRICS.md) defines mAP). The `Trainer` runs MODEL.md §8 end to end and reports through docs/API.md's
event types; the `Predictor` evaluates the head candidate-first on both devices; `export::onnx` writes an opset-17
graph that mirrors `forward`.

### folo-bench
Vision benchmarks ([BENCH.md](BENCH.md)): a pure library (no files, no server types) that folo-server's `/api/bench`
builds on.
- `conditions`: the twelve stress conditions (`Condition`, ids as in the catalog), each a deterministic, size-keeping
  change at severities 1 to 5, with pixel sizes given at a 640 px long side and scaled with the image; `apply` makes a
  variant (only `distance` moves the true boxes), `render` goes from an image file (decoded as datasets are,
  `folo_data::decode`) to the canonical quality-95 JPEG.
- Pixels touch only IEEE basic operations (the sRGB curve and Gaussian weights come from tables built without libm)
  and noise is seeded per row, so a variant has the same bytes on every machine and thread count: `tests/golden.rs`
  pins a digest of every condition, two canonical JPEGs and a score, on the Linux and Windows CI runners.
  `ENGINE_VERSION` names what the crate computes and goes into every benchmark's fingerprint; a change the golden
  tests catch is a bump.
- `plan`: the catalog, the default plan, `check_plan` (docs/API.md's rules) and `variants`. `score`: every number of
  BENCH.md's *Definitions* from plain per-image truths and detections, returned as the `folo_api` wire types. AP is
  folo-vision's `Evaluator` (and `CocoEval` for the size buckets), never a second implementation; the bootstrap
  interval reruns it on 200 seeded resamples in parallel.

### folo-wasm
FOLO in the browser ([WASM.md](WASM.md): JS API, build and limits). `crates/folo-wasm` (wasm-bindgen) loads a `.folo`
file from bytes and runs `folo_vision::Predictor`'s steps on WebGPU (folo-gpu's browser backend, async device
creation and readback) or on folo-core's CPU kernels compiled to wasm (single-threaded, built with SIMD128). `detect`
resolves to `/api/predict`'s response shape plus the backend, so Studio can run models in the page.
`scripts/build-wasm.sh` builds it into `web/public/folo-wasm/`.

## The service crates
How they fit together, with the server's internals: [SERVER.md](SERVER.md).

### folo-api
The request and response types of [API.md](API.md), a dependency-light crate (serde and serde_json, with schemars
behind its `schema` feature for the MCP tool schemas) that folo-server, folo-cli, folo-mcp, folo-vision (its run
events) and FOLO Desktop share. `codes::CODES` lists every error code with its HTTP status and the CLI's exit code.

### folo-server
axum + tokio, implementing [API.md](API.md): datasets, models, training runs with live SSE metrics, inference, cloud
compute, agent labeling and vision benchmarks. Handlers sit on a `Store` trait: `MockStore` for `--mock`, `RealStore`
for the engine, and folo-mcp's HTTP-backed store. The real store trains with folo-vision's `Trainer` on one background
worker thread (a FIFO queue, one run at a time), fans the trainer's events out to SSE subscribers through a per-run
broadcast channel and persists each run under `<data dir>/runs/<id>/`. `folo serve` runs it and serves the built
Studio; `folo serve --mock` serves realistic synthetic data for UI work before or without a trained model.
Internals, the data dir, cloud compute, labeling and benchmarks: [SERVER.md](SERVER.md).

### folo-client
The blocking HTTP client of [API.md](API.md), with uploads and downloads in parts (`transfer`). It knows nothing about
individual routes. `folo mcp --server` (through `proxy::HttpStore`) and the cloud control plane (talking to a worker)
use it.

### folo-cloud
Cloud compute providers behind one trait ([CLOUD.md](CLOUD.md)): `Provider` (check credentials, list offers, launch,
look up, find by name, terminate), with `RunPod`, which rents a pod per run and starts the worker with a bootstrap
script, and `Remote`, a server that already runs. `providers()` lists them. The control plane that uses them is
`folo_server::cloud`.

### folo-mcp
`folo mcp` ([MCP.md](MCP.md)): an MCP server over stdio on the official `rmcp` SDK, with tools for datasets, training,
runs, validation, prediction, export and cloud compute, `folo://` resources and prompts. It sits on the same `Store`
trait as the HTTP handlers: the embedded `RealStore` / `MockStore`, or `proxy::HttpStore`, which forwards to a running
`folo serve` (`--server`).

### folo-cli
The `folo` binary ([CLI.md](CLI.md)): `serve`, `describe`, `doctor`, `train`, `stop`, `val`, `predict`, `export`,
`bench`, `dataset check|stats|synth|convert`, `weights`, `cloud` and `mcp` (Model Context Protocol server over stdio
so agents can train and run models). `folo bench` is a placeholder that answers `not_implemented`. Every command takes
`--json` (one JSON object per line on stdout, human text on stderr; `folo mcp` keeps stdout for the protocol, also
with `--json`) and uses stable exit codes. `cli.rs` is the clap tree and `main.rs` dispatches. Most commands have a
module of their own; `val`, `predict` and `export` share `infer.rs`, `stop` shares `train/`, and `bench` is answered
in `main.rs`.

### folo-test-support
Test-only helpers, used as a dev-dependency: `TestDir`, `TestFile` and `ChildGuard`, which keep test scratch folders
under `FOLO_TEST_TMP` and remove them, also when a test fails ([CI.md](CI.md#test-temp-dirs)).

## Front ends

### FOLO Studio
`web/`: TypeScript + Vite on fw, FOLO's own UI framework (`web/framework/`: JSX compiled to cloned DOM templates with
fine-grained signals, scoped CSS; it replaced Svelte 5), a custom design system (tokens, components) and no UI kit.
It talks only to [API.md](API.md); `web/src/lib/api/` holds the typed client and a dev mock, `web/src/routes/` the
screens. See [web/README.md](../web/README.md).

### FOLO Desktop
`apps/desktop/`: a native Rust app on folo-ui, its own UI framework (below), with the same design language and main
screens as Studio (Overview, Datasets, Runs, Models, Benchmarks, Playground and Settings; agent labeling, run compare,
benchmark compare and importing external predictions stay in Studio). It talks to a local `folo serve` over the same API
(it can spawn one), so the engine and the UI stay
decoupled. Its Playground can also run a model in process ("Run on: Local GPU / Local CPU"): it fetches the f16
weights from `GET /api/models/{id}/weights` (revalidated by ETag) and runs folo-vision's `Predictor` on folo-gpu or
the CPU. Its Benchmarks screens use `folo-bench` only in the built-in demo, to grade its fixtures as the server would.
It is a separate cargo workspace (the app and folo-ui) so the main workspace does not build the UI. See
[apps/desktop/README.md](../apps/desktop/README.md).

### folo-ui — FOLO Desktop's UI framework
`apps/desktop/ui` (crate `folo-ui`, a member of the desktop workspace; design record
[ui/DESIGN.md](../apps/desktop/ui/DESIGN.md), how to work on it [ui/README.md](../apps/desktop/ui/README.md)). Its
public API is GPUI 0.2.2's for exactly the subset the app uses (same names, signatures and behaviour, re-exported flat
from the crate root), so the app reads as it did on GPUI; the engine under it is its own. Layers, bottom up:
platform, raster (the renderer), text, the app runtime, the window, the elements and styles.
- Runtime (`app/`): one UI thread; `App` in an `Rc<RefCell<…>>` holds entities, globals, observers and the effect
  queue, flushed after every platform callback. Tasks are `async-task` runnables on the platform's dispatcher, so an
  idle app has no timer and no thread awake.
- Window (`window/`): a frame renders the root view, lays the elements out with taffy (pinned to the version and the
  style mapping GPUI used, so every box lands where it did), prepaints (bounds, hitboxes, scroll offsets, deferred
  draws, the tooltip), paints a `Scene` and dispatches input through the hitboxes and the key-context tree. A window
  draws only when something made it dirty, at most once per display refresh; while nothing changes nothing runs.
- Renderer (`raster/`): the scene (quads, shadows, glyph and SVG sprites, images, paths, underlines, each with its
  clip) is binned into 64 px tiles; each tile's ordered primitives are hashed, cut at the topmost opaque cover, and
  only tiles whose hash changed are rasterized, on the calling thread or on the renderer's own small pool (not
  rayon's, which belongs to the engine). Coverage formulas are GPUI's; blending is 8-bit source-over in gamma space as
  GPUI's DX11 renderer did. Rendering frame A then B into one buffer gives the same bytes as B into a fresh one, for
  every thread count (tested on random scenes). Caches (shadows, paths, resampled images, glyph masks) are bounded in
  bytes. No GPU device, swap chain or shader: the GPU stays with training and inference.
- Text (`text/`): line layout, wrapping and truncation with GPUI's rules, a per-window line layout cache; glyphs from
  DirectWrite on Windows (as GPUI used it: same masks, subpixel variants, contrast and gamma) and from cosmic-text
  elsewhere and in deterministic tests (feature `portable-text`).
- Platforms (`platform/`): Win32 (own window class and message loop, custom title bar through `WM_NCCALCSIZE` /
  `WM_NCHITTEST`, IME, clipboard, file dialogs, per-monitor DPI, presenting damage rectangles with GDI); winit +
  softbuffer on Linux and macOS (also built on Windows with feature `winit`); headless, driven by `HeadlessApp` (fake
  clock, synthetic input, a framebuffer to read back), which the tests and `folo-desktop --shot <png>` use. Windows is
  where it runs; Linux runs the headless tests in CI; macOS is type-checked (`apps/desktop/tools/cross-check.sh`).

## Agent-first design
- Machine-readable everything: `--json`, JSON Schemas for configs, `folo describe` prints the CLI surface as JSON.
- `folo mcp` (crate `folo-mcp`, on the official `rmcp` SDK) is an MCP server over stdio with tools for datasets,
  training, runs, validation, prediction and export, `folo://` resources and prompts ([MCP.md](MCP.md)). It sits on
  the same `Store` trait as the HTTP handlers: the embedded `RealStore` / `MockStore`, or an HTTP-backed store that
  forwards to a running `folo serve` (`--server`). Tool schemas come from the `folo-api` types through schemars (the
  crate's `schema` feature), so they match the HTTP API.
- Errors carry a stable `code`, a human message and a hint.
- `llms.txt` at the repo root summarises the project for agents.
