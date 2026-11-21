# AGENTS.md — how to work on FOLO

FOLO is built almost entirely by a team of specialized AI coding agents, and every change lands as a reviewed pull
request. These rules apply to every contributor, agent or human. Read this file, then
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), then the README of the crate or folder you are touching.
[CONTRIBUTING.md](CONTRIBUTING.md) is the short version for a first pull request.

## Finding your way
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the map of the repository and crates, the main execution flows, and
  the **"Where do I change ...?"** table from a change to the exact files it touches.
- [crates/README.md](crates/README.md): the crates in dependency order and the dependency rules; each
  `crates/<name>/README.md` maps that crate's modules to what they hold. Most top-level folders have a README too
  (listed under [Layout](#layout)).
- [docs/README.md](docs/README.md): every doc, grouped by what you want to do, with a reading order for newcomers.

## What FOLO is
A from-scratch, real-time vision framework in Rust (a YOLO-style object detector with its own tensors, autograd, CPU
and WGSL GPU kernels, training, evaluation, benchmarking, and export) with two front ends sharing one design language:
**FOLO Studio**, a web app (`web/`), and **FOLO Desktop**, a native app on its own UI framework, `folo-ui`
(`apps/desktop/`). It is built to be pleasant for people and effortless for agents to drive (`--json` everywhere,
`folo mcp`, stable exit codes).

## Agentic team & subagents

FOLO is developed agentically in T3 Code using four specialized roles. The main orchestrator coordinates work and
delegates to dedicated subagents so implementation, play-testing/review, and writing each run on the model best
suited for the job:

| Role | Model & settings | Harness / provider | What they own |
|---|---|---|---|
| **Main orchestrator** | **Claude Opus 5.5** (`effort: high`) or **Claude Fable 5.2** (`effort: high`) | `claudeAgent` (`claude-opus-5-5` / `claude-fable-5-2`) | Top-level planning, breaking work into focused PRs, dispatching subagents, tracking CI/reviews/play-testing, merging approved PRs, and updating `docs/ROADMAP.md`. |
| **Main code implementer & doer** *(subagent)* | **Claude Opus 5.5** (`effort: high`) | `claudeAgent` (`claude-opus-5-5`) | The primary builder and doer: implements engine, CLI, server, and UI code, writes tests, runs checks, opens PRs, and fixes review and play-test findings. |
| **App play-tester & reviewer** *(subagent)* | **GPT-6.2 Sol** (`reasoningEffort: high`, **fast** / `serviceTier: priority`) or **GPT-6.2 Astra** (`reasoningEffort: high`, **standard — not fast** / `serviceTier: default`) | `codex` (`gpt-6.2-sol` / `gpt-6.2-astra`) | Play-tests the real app end to end (FOLO Studio, FOLO Desktop, CLI, MCP, edge cases, live states) like a demanding user, and runs independent code/PR reviews against `origin/main`. |
| **Writer & creative writer** *(subagent)* | **Gemini 4 Neon** (`thought_level: high`) | Antigravity CLI in T3 Code (`antigravity`, `gemini-4-neon`) | Owns documentation clarity, `README.md` storytelling, user guides, release notes, changelogs, and creative/UX copy so the project is clear and pleasant to read. When the owner asks for it, takes any role Opus 5.5 has (see below). |

Subagent guidelines:
- Delegate by role rather than having one model do everything: let **Opus 5.5** implement and fix code, **GPT-6.2 Sol
  (high, fast)** or **GPT-6.2 Astra (high, standard — not fast)** play-test the app and review diffs, and **Gemini 4
  Neon** (via the Antigravity CLI in T3 Code) write and polish docs and creative copy.
- **Gemini 4 Neon can replace Opus 5.5 in everything when the owner asks for it**: main orchestration, implementing
  and fixing code, research. The table's models are the defaults and Opus 5.5 stays one; the swap is the owner's
  call, per task or per wave, and every other rule here (PRs, disjoint worktrees) applies to Neon as it does to Opus.
  A Neon agent splits its own work with Antigravity's agent spawning and checks what its sub-agents return; PR
  reviews still run on a fresh GPT reviewer through T3 Code, never on one of its own sub-agents.
- Keep worktrees disjoint when running subagents in parallel so two subagents never edit the same checkout or files.
- Never grade your own PR: code review and app play-testing always run on a fresh play-tester/reviewer subagent.
- [docs/notes/PROCESS.md](docs/notes/PROCESS.md) has the exact T3 Code `delegate_task` target configs, prompts, and
  PR review loop.

## Ground rules
- **Rust stable, edition 2024.** `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo test --workspace` must pass before you open a PR. CI runs the same on Linux and Windows.
- **Custom where it matters.** The ML stack is ours: tensors, kernels, autograd, layers, optimizers, the detector,
  losses, assigners, NMS, metrics, augmentation and the GPU kernels. Do **not** pull in ML frameworks (candle, burn,
  tch, ndarray-based NN crates, ort, ...). Commodity crates are fine for commodity jobs: `image`/`zune-jpeg`
  (decoding), `wgpu` (GPU access), `rayon` (threads), `serde`/`serde_json`, `axum`/`tokio` (server), `clap` (CLI),
  `bytemuck`, `thiserror`/`anyhow`. Add a dependency only to the crate that needs it and declare it in
  `[workspace.dependencies]`.
- **Fast is a feature.** No per-element allocations in hot loops, no needless clones of tensor storage, prefer
  contiguous `&[f32]` loops the compiler can vectorise, parallelise with rayon above a size threshold. Benchmarks live
  in `crates/<crate>/benches` (plain `harness = false` binaries are fine) and PRs that touch hot paths report numbers.
- **Correct first.** Every op has tests; every differentiable op has a finite-difference gradient check; GPU kernels
  are tested against the CPU reference within a stated tolerance. Tests must be deterministic (seeded RNG) and quick:
  about a second each at most (the smallest model, a few small images, a few steps, no fixed sleeps), and each one
  there because it protects something that matters (docs/CI.md, Keeping CI fast). `#[ignore]` is for a test that
  needs what CI does not have, or a documented manual acceptance run, not for a slow one.
- **Docs as you go.** Public items get a doc comment. When you change behaviour or an interface, update the matching
  file in `docs/` in the same PR. A change that adds, moves or removes a module updates that crate's (or folder's)
  README in the same PR, and `docs/ARCHITECTURE.md` when the map or the change table no longer matches. For any
  Markdown change, run `cd site && npm run check` to verify every link, anchor, and image.
- **No god files.** A source file stays under 1000 lines, every line counted (`bash tools/file-size.sh`, which CI runs
  on every pull request; docs and data are exempt). Before a file gets there, split it by responsibility
  into a folder of modules named for what they hold (`runs.rs` → `runs/mod.rs` + `runs/queue.rs`, a route's parts in
  its folder, an integration test as `tests/api/main.rs` + topic files so it stays one binary), never by line count
  into `part2` or `misc`.
- **No secrets in the repo, ever.** Keys live outside the repo (for example under `~/.folo/`). Never print them.

## Workflow (every change is a PR)
1. Work on a branch named `<area>/<short-topic>` (e.g. `core/autograd`, `ui/run-view`, `docs/readme-polish`), in your
   own git worktree when several agents share a machine.
2. Keep PRs focused and reviewable; a big feature can be a stack of PRs.
3. Commit messages and PR titles: `area: imperative summary` (e.g. `core: add broadcasting binary ops`).
4. Open the PR against `main` with a description: what, why, how it was tested, benchmark numbers if relevant.
5. Play-test & review: app-facing changes are play-tested and every PR is reviewed for bugs before it merges; findings
   are fixed or answered until the reviewer has no blocking findings.
6. The orchestrator or a maintainer merges (squash) once CI is green and review is clean. Do not merge your own PR
   unless your task says so.
7. Rebase on `main` if you fall behind; never force-push someone else's branch.

[docs/notes/PROCESS.md](docs/notes/PROCESS.md) has the exact T3 Code subagent procedure.

## Tests and temp dirs
- Tests put their scratch folders (`folo-*`) under `FOLO_TEST_TMP` when it is set, else the system temp dir, and
  remove them when they finish, also when they fail (`folo_test_support::TestDir`).
  A new test that needs a folder takes a `TestDir` (or `TestFile`), and a process it spawns goes in a `ChildGuard`
  declared after that folder; `crates/folo-test-support/tests/hygiene.rs` fails on a direct `std::env::temp_dir()`.
  `bash tools/test-tmp.sh cargo test --workspace` runs the tests the way CI does and fails if they leave anything
  behind (docs/CI.md, Test temp dirs).
- Keep the default `jobs = 8` from `.cargo/config.toml`, and don't run more than one heavy build at a time from one
  agent: when several agents build on one machine, they share its cores.
- GPU test runs should be short (< 2 min) unless your task says otherwise: a GPU is often shared.

## On the maintainers' machine
These apply to agents working on the maintainers' Windows machine (an RTX 4080 SUPER and an i7-14700KF, shared by
many agents); skip them anywhere else.
- The C: drive is small. Put build output on D:: `export CARGO_TARGET_DIR=D:/folo-build/<branch-name-with-dashes>`
  (PowerShell: `$env:CARGO_TARGET_DIR="D:\folo-build\<name>"`). Delete that directory when your PR is merged.
- Keep test scratch folders off C: too: `export FOLO_TEST_TMP=D:/folo-build/test-tmp`
  (PowerShell: `$env:FOLO_TEST_TMP="D:\folo-build\test-tmp"`).
- Offloading CPU-heavy work (Linux builds, long CPU benchmarks, dataset preprocessing) to boat.dev sandboxes is
  allowed: see `docs/notes/BOAT.md`.

## Layout
Every crate has a README, and so do `crates/`, `web/`, `apps/desktop/`, `design/`, `examples/`, `bench/`, `tools/`,
`scripts/`, `site/` and `docs/`; `crates/README.md` gives the crates' dependency edges and rules.
```
crates/              the root cargo workspace (members = crates/*), engine at the bottom, services on top
crates/folo-core     tensors, dtypes, shapes, CPU kernels, autograd tape, the Backend trait
crates/folo-gpu      wgpu device, buffer pool, WGSL kernels; implements the core Backend trait
crates/folo-nn       modules (Conv2d, BatchNorm2d, ...), init, optimizers, LR schedules, .folo weight files
crates/folo-data     image IO, YOLO/COCO dataset formats, augmentation, batching data loader, synthetic datasets
crates/folo-vision   FOLO detector (n/s/m/l/x), loss, task-aligned assigner, box ops, NMS, mAP, trainer, predictor,
                     val, ONNX export
crates/folo-api      serde types of docs/API.md, shared by folo-vision, the service crates, the CLI and FOLO Desktop
crates/folo-server   axum HTTP + SSE API used by Studio and by agents: the Store, the run queue, labeling, cloud
                     runs, benchmarks; serves the built web app; docs/SERVER.md
crates/folo-client   blocking HTTP client of docs/API.md: `folo mcp --server`, cloud workers; transfers in parts
crates/folo-cloud    cloud compute providers behind one trait (RunPod, a remote FOLO server), docs/CLOUD.md
crates/folo-bench    vision benchmarks, docs/BENCH.md: stress conditions, plans, every number of an entry's score
crates/folo-mcp      `folo mcp`: the MCP server (rmcp, stdio) over the server's Store, embedded or proxying folo serve
crates/folo-cli      the `folo` binary (serve / describe / doctor / train / stop / val / predict / export / bench /
                     dataset / weights / cloud / mcp), docs/CLI.md
crates/folo-wasm     FOLO detectors in the browser (wasm-bindgen; WebGPU or the wasm CPU), docs/WASM.md
crates/folo-test-support  test-only helpers (a dev-dependency): TestDir / TestFile / ChildGuard, FOLO_TEST_TMP
web/                 FOLO Studio (TypeScript + Vite on fw, our own UI framework in web/framework; custom design system, no UI kit)
apps/desktop/        FOLO Desktop (Rust, its own cargo workspace: the app and ui/ = folo-ui, its UI framework;
                     talks to folo-server over docs/API.md)
design/              the design system both apps share: tokens.json, fonts, the logo
examples/            shapes (the convergence test), coco (COCO pretraining), bench_external.py (an external
                     benchmark entry, docs/BENCH.md)
bench/               Python harnesses outside cargo; bench/README.md says what each folder holds:
bench/baselines/     incumbent (Ultralytics) benchmark scripts + raw results; numbers and FOLO targets in docs/BENCHMARKS.md
bench/finetune/      fine-tuning accuracy: folo-n vs yolo26n and folo-s vs yolo26s on small public datasets, one
                     pycocotools scorer
bench/...            fusion/, learn/, parity/, refine/, trainer/
tools/               CI and development helpers; tools/charts draws the charts of the README and docs
scripts/             build-wasm.sh / build-wasm.ps1 (the browser module), install.sh / install.ps1 (the release installers)
site/                the docs website, generated from docs/, README.md and the examples
.github/             the CI workflows (docs/CI.md), the issue and pull request templates
docs/                README (the index), QUICKSTART, ARCHITECTURE, CORE, VISION, SERVER, MODEL (detector spec), API (HTTP
                     contract), CLI, MCP, FORMAT (.folo), ROADMAP, BENCHMARKS, WASM, CLOUD (training on rented or
                     remote GPUs), ...; docs/img holds the README's screenshots and charts; docs/notes holds the
                     engineering notebooks and the agents' process
```
