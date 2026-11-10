# AGENTS.md — how to work on FOLO

FOLO is built almost entirely by AI agents coordinated from one orchestrator thread. These rules apply to every agent
(human contributors too). Read this file, then `docs/ARCHITECTURE.md`, then the crate you are touching.

## What FOLO is
A from-scratch, very fast vision framework in Rust (a YOLO-style object detector, plus training, evaluation and export)
with two front ends sharing one design language: **FOLO Studio**, a web app (`web/`), and **FOLO Desktop**, a native
GPUI app (`apps/desktop/`), for datasets, training runs and live inference. It is meant to be pleasant for people and
easy for agents to drive (`--json` everywhere, an MCP server, stable exit codes).

## Ground rules
- **Rust stable, edition 2024.** `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo test --workspace` must pass before you open a PR. CI runs the same on Linux and Windows.
- **Custom where it matters.** The ML stack is ours: tensors, kernels, autograd, layers, optimizers, the detector,
  losses, assigners, NMS, metrics, augmentation and the GPU kernels. Do **not** pull in ML frameworks (candle, burn, tch,
  ndarray-based NN crates, ort, ...). Commodity crates are fine for commodity jobs: `image`/`zune-jpeg` (decoding),
  `wgpu` (GPU access), `rayon` (threads), `serde`/`serde_json`, `axum`/`tokio` (server), `clap` (CLI), `bytemuck`,
  `thiserror`/`anyhow`. Add a dependency only to the crate that needs it and declare it in `[workspace.dependencies]`.
- **Fast is a feature.** No per-element allocations in hot loops, no needless clones of tensor storage, prefer
  contiguous `&[f32]` loops the compiler can vectorise, parallelise with rayon above a size threshold. Benchmarks live
  in `crates/<crate>/benches` (plain `harness = false` binaries are fine) and PRs that touch hot paths report numbers.
- **Correct first.** Every op has tests; every differentiable op has a finite-difference gradient check; GPU kernels are
  tested against the CPU reference within a stated tolerance. Tests must be deterministic (seeded RNG) and quick
  (whole workspace `cargo test` under ~2 min on a desktop; mark longer ones `#[ignore]`).
- **Docs as you go.** Public items get a doc comment. When you change behaviour or an interface, update the matching file
  in `docs/` in the same PR.
- **No secrets in the repo, ever.** Keys live outside the repo (for example `~/.folo/boat.key`). Never print them.

## Workflow (every change is a PR) — details in docs/PROCESS.md
1. Work in your own git worktree on a branch named `<area>/<short-topic>` (e.g. `core/autograd`, `ui/run-view`).
2. Keep PRs focused and reviewable; a big feature can be a stack of PRs.
3. Commit messages and PR titles: `area: imperative summary` (e.g. `core: add broadcasting binary ops`).
4. Open the PR against `main` with a description: what, why, how it was tested, benchmark numbers if relevant.
5. Review: a reviewer agent (GPT-6.1 Sol, high effort) reviews every PR for bugs; an implementer agent
   (Claude Opus 5.5, high effort) fixes the findings. Repeat until the reviewer has no blocking findings.
6. The orchestrator merges (squash) once CI is green and review is clean. Do not merge your own PR unless your task says so.
7. Rebase on `main` if you fall behind; never force-push someone else's branch.

## Build hygiene on the dev machine
- The C: drive is small. Put build output on D:: `export CARGO_TARGET_DIR=D:/folo-build/<branch-name-with-dashes>`
  (PowerShell: `$env:CARGO_TARGET_DIR="D:\folo-build\<name>"`). Delete that directory when your PR is merged.
- Tests put their scratch folders (`folo-*`) under `FOLO_TEST_TMP` when it is set, else the system temp dir, and
  remove them when they finish, also when they fail (`folo_test_support::TestDir`). On the dev machine, keep them off
  C: too: `export FOLO_TEST_TMP=D:/folo-build/test-tmp` (PowerShell: `$env:FOLO_TEST_TMP="D:\folo-build\test-tmp"`).
  A new test that needs a folder takes a `TestDir` (or `TestFile`), and a process it spawns goes in a `ChildGuard`
  declared after that folder; `crates/folo-test-support/tests/hygiene.rs` fails on a direct `std::env::temp_dir()`.
  `bash tools/test-tmp.sh cargo test --workspace` runs the tests the way CI does and fails if they leave anything
  behind (docs/CI.md, Test temp dirs).
- Many agents build at once: keep the default `jobs = 8` from `.cargo/config.toml`, and don't run more than one heavy
  build at a time from one agent.
- The GPU (RTX 4080 SUPER) is shared. GPU test runs should be short (< 2 min) unless your task says otherwise.
- Offloading CPU-heavy work (Linux builds, long CPU benchmarks, dataset preprocessing) to boat.dev sandboxes is allowed:
  see `docs/BOAT.md`.

## Layout
```
crates/folo-core     tensors, dtypes, shapes, CPU kernels, autograd tape
crates/folo-gpu      wgpu device, buffer pool, WGSL kernels; implements the core Backend trait
crates/folo-nn       modules (Conv2d, BatchNorm2d, ...), init, optimizers, LR schedules, .folo weight files
crates/folo-data     image IO, YOLO/COCO dataset formats, augmentation, batching data loader, synthetic datasets
crates/folo-vision   FOLO detector (n/s/m/l), loss, task-aligned assigner, box ops, NMS, mAP, trainer, predictor
crates/folo-api      serde types of docs/API.md, shared by the server, the CLI and FOLO Desktop
crates/folo-server   axum HTTP + SSE API used by Studio and by agents; serves the built web app
crates/folo-client   blocking HTTP client of docs/API.md: `folo mcp --server`, cloud workers; transfers in parts
crates/folo-cloud    cloud compute providers behind one trait (RunPod, a remote FOLO server), docs/CLOUD.md
crates/folo-mcp      `folo mcp`: the MCP server (rmcp, stdio) over the server's Store, embedded or proxying folo serve
crates/folo-cli      the `folo` binary (train / val / predict / export / serve / cloud / mcp / doctor)
crates/folo-wasm     FOLO detectors in the browser (wasm-bindgen; WebGPU or the wasm CPU), docs/WASM.md
crates/folo-test-support  test-only helpers (a dev-dependency): TestDir / TestFile / ChildGuard, FOLO_TEST_TMP
web/                 FOLO Studio (Svelte 5 + TypeScript + Vite, custom design system, no UI kit)
apps/desktop/        FOLO Desktop (Rust + GPUI, its own cargo workspace; talks to folo-server over docs/API.md)
bench/baselines/     incumbent (Ultralytics) benchmark scripts + raw results; numbers and FOLO targets in docs/BENCHMARKS.md
docs/                QUICKSTART, ARCHITECTURE, MODEL (detector spec), API (HTTP contract), CLI, MCP, FORMAT (.folo), ROADMAP, BOAT,
                     BENCHMARKS, WASM, CLOUD (training on rented or remote GPUs), ...; docs/img holds the README screenshots
```
