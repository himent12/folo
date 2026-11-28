<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="design/logo-wordmark.svg">
    <img src="design/logo-wordmark-light.svg" alt="folo" width="207" height="93">
  </picture>
</p>

# FOLO

**FOLO Only Looks Once.** A from-scratch, real-time vision framework in Rust: train, evaluate, benchmark, track, and
deploy YOLO-style object detectors from one native binary — with custom tensors, autograd, CPU/GPU kernels, Studio,
Desktop, and first-class agent tooling.

- **One binary, zero Python or CUDA.** Custom Rust engine and `wgpu`/WGSL shaders run on Vulkan, DX12, Metal, AVX2/VNNI
  CPUs (including INT8), or WebGPU/Wasm in the browser (~14 ms on WebGPU for `folo-n` at 640×640 in Chrome).
- **Full vision workflow.** Curate YOLO/COCO/VOC datasets, label raw images with coding agents, train `folo` and
  `folo2` models locally (or `folo` models on cloud GPUs with spot auto-resume), evaluate with pycocotools-exact mAP,
  and export `.folo` or ONNX.
- **Production inference built in.** Run f32, GPU f16/tensor-core, or CPU INT8 inference with SAHI tiling, TTA,
  ByteTrack multi-object tracking, calibrated per-class thresholds, and 5-axis vision benchmarks.
- **Built for agents.** CLI commands support `--json` with stable exit codes, `folo describe --json` dumps the full
  schema, and `folo mcp` gives Claude Code, Codex, Antigravity, or any MCP client 45 tools.

> **Status: early (v0.1).** Trains and runs on CPU and GPU with published COCO weights (`folo weights pull folo-s-coco`
> / `folo-n-coco-v5`). Faster than Ultralytics YOLO26 on every measured row; COCO accuracy is not yet level, and
> interfaces may still change. See [Project status](#project-status).

**[Benchmarks](#why-folo-the-numbers)** · **[Quickstart](#quickstart)** · **[Studio](#folo-studio)** ·
**[Desktop](#folo-desktop)** · **[Agents & MCP](#use-folo-from-an-agent)** · **[Docs](#docs)**

## Why FOLO: the numbers

`folo-n` vs Ultralytics **yolo26n** (8.4.198, torch 2.11 + CUDA 12.8) on one machine (RTX 4080 SUPER, i7-14700KF) at
640×640, against each `yolo26n` row's fastest configuration (`folo-s` also beats `yolo26s` on every measured row). Full
method, `folo-s` numbers, and raw data: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/charts/speed-dark.svg">
    <img src="docs/img/charts/speed-light.svg" alt="Speed and memory, folo-n against yolo26n at 640x640. Batch-1 latency (GPU on 0-7 ms scale, CPU on 0-25 ms scale): GPU forward 1.09 vs 4.95 ms (4.5x), GPU end to end 1.60 vs 6.92 ms (4.3x), CPU forward 12.7 vs 22.8 ms (1.8x), CPU end to end 13.6 vs 24.8 ms (1.8x). Batch-16 GPU inference: forward f32 2081 vs 1454 img/s (1.4x), forward f16 4159 vs 2370 img/s (1.8x), end to end f16 2428 vs 615 img/s (3.9x). Batch-16 training: step f32 380 vs 202 img/s (1.9x), step AMP 594 vs 252 img/s (2.4x), whole trainer f32 388 vs 199 img/s (1.9x), whole trainer AMP 613 vs 224 img/s (2.7x). Peak GPU memory: 3467 vs 4517 MiB in f32 (-23%), 2022 vs 2646 MiB in AMP (-24%)." width="840">
  </picture>
</p>

<details>
<summary>Full benchmark table and measurement notes</summary>

| metric (640×640) | folo-n | yolo26n (best config) | ratio |
|---|---:|---:|---|
| GPU forward, batch 1, f32 (median) | **1.09 ms** | 4.95 ms (6.04 ms default) | **4.5×** faster |
| GPU end to end, batch 1 (decoded image → boxes) | **1.60 ms** | 6.92 ms | **4.3×** faster |
| CPU forward, batch 1, f32 | **12.7 ms** (28 threads) | 22.8 ms (8 threads) | **1.8×** faster |
| CPU end to end, batch 1 | **13.6 ms** (28 threads) | 24.8 ms | **1.8×** faster |
| GPU forward, batch 16, f32 | **2081 img/s** | 1454 img/s (TF32) | **1.4×** faster |
| GPU forward, batch 16, f16 (tensor cores) | **4159 img/s** | 2370 img/s (fp16) | **1.8×** faster |
| GPU end to end, batch 16, f16 (tensor cores) | **2428 img/s** | 615 img/s (fp16) | **3.9×** faster |
| Training step, batch 16 (f32 / AMP) | **380** / **594 img/s** | 202 (TF32) / 252 img/s | **1.9× / 2.4×** faster |
| Whole trainer, sustained (mosaic, 8 workers) | **388** / **613 img/s** | 199 / 224 img/s | **1.9× / 2.7×** faster |
| Training step peak GPU memory, batch 16 | **3467** / **2022 MiB** | 4517 / 2646 MiB | **23% / 24%** less |

FOLO computes in strict f32 unless a row notes f16/AMP (~19% more FLOPs than `yolo26n`: 6.54 vs 5.5 GFLOPs). FOLO rows
are medians of alternated runs on `main` (2026-11-12/13; end-to-end rows on published COCO weights); `yolo26n` was
measured 2026-11-07. GPU batch-1 latency warms up 25 forward / 10 end-to-end calls on both sides, then FOLO keeps
warming until 1 s elapses ([method](docs/BENCHMARKS.md#method)).

</details>

### Accuracy

Trained from random weights on COCO alone, `folo-n` (`folo-n-coco-v5`, 128 epochs, 96 distilled from `folo-s`) reaches
**0.370** val2017 AP and `folo-s` reaches **0.416** (56 epochs); opt-in SEBF + DFL-CGR post-processing
(`--fusion sebf --score dfl-cgr --iou 0.78 --max-det 1000`) reaches **0.378** and **0.423**. On one 20-epoch COCO
subset from scratch, `folo-n` matches Ultralytics **YOLO11n** per epoch with the same optimizer, and fine-tuned on four
small datasets (`folo-n-coco-v2` vs `yolo26n`) `folo-n` is level on three of four ([examples/coco](examples/coco/README.md),
[bench/finetune](bench/finetune/README.md)).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/charts/accuracy-dark.svg">
    <img src="docs/img/charts/accuracy-light.svg" alt="Accuracy, FOLO against YOLO26 (AP@[.5:.95], pycocotools, 0 to 1 shared scale). COCO val2017 (not like for like): folo-n v5 0.370 vs yolo26n 0.409 (-0.039), folo-s 0.416 vs yolo26s 0.480 (-0.064). Fine-tuned on small datasets (folo-n vs yolo26n, 3-seed mean): Blood cells 0.607 vs 0.626 (-0.020), Brain tumours 0.452 vs 0.436 (level), Signatures 0.989 vs 0.982 (level), Medical pills 0.804 vs 0.803 (level)." width="840">
  </picture>
</p>

## Quickstart

### Install

Build from source on Linux, Windows, or macOS (requires Rust stable; `cargo folo` is a workspace alias for
`cargo run --release -p folo-cli --`):

```sh
git clone https://github.com/himent12/folo && cd folo
npm --prefix web ci && npm --prefix web run build          # build Studio once
cargo install --path crates/folo-cli --features embed-web  # or omit --features embed-web for CLI/API only
```

For in-browser WebGPU/Wasm inference in Studio, build `folo-wasm` before `npm --prefix web run build`
(`scripts/build-wasm.sh` / `scripts/build-wasm.ps1`, see [docs/WASM.md](docs/WASM.md)).

<details>
<summary>Prebuilt release installers (when a binary release is published on Releases)</summary>

Once a tagged binary release is published as `latest` on [Releases](https://github.com/himent12/folo/releases)
([docs/RELEASE.md](docs/RELEASE.md)), prebuilt `x86_64` binaries for Linux (glibc 2.35+) and Windows embed Studio out
of the box:

```sh
# Linux (~/.local/bin)
curl -fsSL https://github.com/himent12/folo/releases/latest/download/install.sh | sh
```

```powershell
# Windows (adds to PATH; save and pass -Desktop to also install FOLO Desktop)
irm https://github.com/himent12/folo/releases/latest/download/install.ps1 | iex
```

</details>

### Train and run in 4 commands

```sh
folo doctor                                                               # check GPU/SIMD readiness
folo dataset synth datasets/shapes --images 1000 --imgsz 320              # synthetic YOLO dataset (~1 s)
folo train --data datasets/shapes --model folo-n --imgsz 320 --epochs 30  # ~83 s on RTX 4080 SUPER (mAP50 0.955)
folo predict --model runs/shapes-n/weights/best.folo datasets/shapes/images/val --save runs/shapes-n/pred
```

Start the web UI with demo data (or copy `runs/shapes-n/weights/best.folo` into `<data dir>/models/`, default
`~/.folo/models/`, and run `folo serve` without `--mock` to test your trained model in Studio):

```sh
folo serve --mock   # serves FOLO Studio at http://127.0.0.1:7878 (open in your browser)
```

See **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for custom YOLO/COCO/VOC folders, fine-tuning from COCO weights
(`folo weights pull folo-n-coco-v5`), ONNX/INT8 export, and in-browser inference.

## FOLO Studio

![FOLO Studio: a live training run](docs/img/studio-run-live.webp)

`folo serve` hosts Studio at `http://127.0.0.1:7878` (built on `fw`, FOLO's own zero-VDOM TypeScript framework and
design system):
- **Datasets & Labeling** — inspect YOLO/COCO/VOC datasets, audit duplicates/leakage, generate CLI commands to
  filter/merge/export, or label raw images with Claude Code, Codex, or Antigravity ([docs/LABELING.md](docs/LABELING.md)).
- **Runs** — launch, compare, queue, stop, or resume runs with live SSE charts and validation grids, locally, on remote
  FOLO servers, or on RunPod with automatic spot-preemption checkpoint recovery ([docs/CLOUD.md](docs/CLOUD.md)).
- **Models** — inspect per-class PR curves, calibrated F1 thresholds, and confusion matrices; export `.folo` or ONNX.
- **Benchmarks** — freeze labeled images into 5-axis vision benchmarks across 12 stress conditions, inspect worst
  errors, compare entries head-to-head, and run self-improvement loops ([docs/BENCH.md](docs/BENCH.md)).
- **Playground** — test images, video, or live camera on the server or in-browser via WebGPU/Wasm
  ([docs/WASM.md](docs/WASM.md)), with A/B compare, per-class thresholds, and instant filtering.

![FOLO Studio: the Playground](docs/img/studio-playground.webp)

Frontend dev guide: [web/README.md](web/README.md).

## FOLO Desktop

![FOLO Desktop: a live training run](docs/img/desktop-run-live.webp)

A native Rust app on `folo-ui` sharing Studio's design system and screens (Overview, Datasets, Labeling, Runs & Compare,
Models, Benchmarks & Compare, Playground A/B & per-class thresholds, and Settings). The UI renders on the CPU so your
GPU stays 100% free for training and inference, includes a command palette and custom themes, and auto-starts a local
`folo serve` (or connect with `--server URL`, or `--mock` for demo data):

```sh
cd apps/desktop && cargo run --release -- --mock
```

Lives in its own cargo workspace so engine builds stay fast ([apps/desktop/README.md](apps/desktop/README.md)).

## Use FOLO from an agent

**MCP.** `folo mcp` speaks [Model Context Protocol](https://modelcontextprotocol.io) over stdio with **45 tools**
covering datasets, curation/dedup, agent labeling, local & cloud training, evaluation, tiled/TTA inference with
calibrated thresholds, export, and vision benchmarks — plus `folo://` resources and prompts.

```sh
claude mcp add folo -- folo mcp   # Claude Code
codex mcp add folo -- folo mcp    # Codex
agy mcp add folo -- folo mcp      # Antigravity
```

**CLI with `--json`.** `--json` is global: stdout emits pure JSON Lines and stderr carries progress (except `folo mcp`,
which reserves stdout for MCP, and `--help` / `--version`, which stay plain text). Errors return
`{"error":{"code","message","hint"}}` with stable exit codes (`0` ok, `1` runtime, `2` usage, `3` not found, `4`
unsupported), and `folo describe --json` dumps the full CLI and MCP schema.

```sh
folo train --data datasets/shapes --model folo-n --epochs 30 --json
folo val --model runs/shapes-n/weights/best.folo --data datasets/shapes --diagnostics --json
```

Guides: [docs/AGENT_COOKBOOK.md](docs/AGENT_COOKBOOK.md) · [docs/MCP.md](docs/MCP.md) · [llms.txt](llms.txt).

## Architecture at a glance

Every ML layer — tensors, autograd, CPU/WGSL kernels, layers, optimizers, detectors, losses, assigner, NMS, metrics,
and augmentation — is written from scratch in Rust; commodity crates handle IO and serving (`wgpu`, `rayon`, `serde`,
`axum`, `clap`). Each row depends only on rows below it:

```text
 FOLO Studio (web/)   FOLO Desktop (apps/desktop/)     the apps: HTTP + SSE to folo-server (docs/API.md)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
 folo-cli      the `folo` binary: train, val, predict, export, quantize, bench, label, serve, mcp, cloud, ...
 folo-mcp      `folo mcp`: 45-tool MCP server over stdio (embedded or proxying `folo serve`)
 folo-server   axum HTTP + SSE: datasets, runs queue, models, labeling, cloud runs, benchmarks; serves Studio
 folo-bench    vision benchmarks       folo-cloud    RunPod, remote servers   folo-wasm    WebGPU / Wasm build
 folo-client   blocking HTTP client with chunked resumable transfers
 folo-vision   detectors (folo / folo2), loss, assigner, NMS/fusion, tracking, mAP, trainer, predictor, ONNX
 folo-nn       layers, optimizers      folo-data     datasets, augmentation   folo-gpu     wgpu, WGSL kernels
 folo-core     tensors, autograd, CPU/SIMD/INT8 kernels, Backend trait        folo-api     shared wire types
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (including **"Where do I change ...?"**) and
[crates/README.md](crates/README.md).

<details>
<summary>Repository layout (every folder's README)</summary>

| folder | what is there |
|---|---|
| [crates/](crates/README.md) | Root cargo workspace: engine and services, with a README per crate |
| [web/](web/README.md) | FOLO Studio (TypeScript + Vite) and `web/framework/` (`fw`, custom UI framework) |
| [apps/desktop/](apps/desktop/README.md) | FOLO Desktop and `folo-ui` (`ui/`), custom native CPU-rendered UI framework |
| [design/](design/README.md) | Shared design tokens, fonts, themes, contrast checks, and logos |
| [docs/](docs/README.md) | User guides, specs, and `docs/notes/` engineering notebooks |
| [examples/](examples/README.md) | Worked runs: shapes convergence test, COCO pretraining, external bench entry |
| [bench/](bench/README.md) | Python harnesses: Ultralytics baselines, fine-tuning accuracy, parity |
| [tools/](tools/README.md) | CI/dev helpers and the zero-dep SVG chart generator (`tools/charts/`) |
| [scripts/](scripts/README.md) | WebAssembly build scripts and release installers |
| [site/](site/README.md) | Static documentation site generated directly from this repo's Markdown |
| [.github/](.github/) | CI workflows ([docs/CI.md](docs/CI.md)) and issue/PR templates |

</details>

## Project status

**Works today (v0.1):**
- `folo-n/s/m/l/x` and `folo2-n/s/m/l/x` training on CPU or GPU (`--amp`, `--batch auto`, `--head o2o`, `--teacher`
  distillation, MuSGD, bit-exact resume) plus GPU f16/tensor-core and CPU INT8 (`folo quantize`) inference.
- Published COCO weights via `folo weights pull` (`folo-s-coco` AP 0.416, `folo-n-coco-v5` AP 0.370; `+0.0065–0.0076`
  AP with `--fusion sebf --score dfl-cgr --iou 0.78 --max-det 1000`).
- YOLO/COCO/VOC curation & conversion, pycocotools-exact `val` with error diagnostics & per-class threshold calibration,
  SAHI/TTA/ByteTrack `predict`, and `.folo` / ONNX export.
- FOLO Studio (`folo serve`), FOLO Desktop, `folo mcp` (45 tools), in-browser WebGPU/Wasm, agent labeling, cloud GPU
  training (v1 models, with RunPod spot auto-resume), vision benchmarks, and release tooling ([CHANGELOG.md](CHANGELOG.md)).

**Not yet:**
- COCO accuracy level with Objects365-pretrained incumbents or an equal-budget full-COCO run ([Accuracy](#accuracy));
- a CLI speed-benchmark subcommand (speed numbers come from `cargo bench`, [docs/BENCHMARKS.md](docs/BENCHMARKS.md);
  `folo bench` runs vision benchmarks);
- prebuilt macOS release binaries (and published `latest` binary release artifacts).

Live checklist: [docs/ROADMAP.md](docs/ROADMAP.md).

## Docs

Browse by task in [docs/README.md](docs/README.md), try the examples ([shapes](examples/shapes/README.md),
[coco](examples/coco/README.md)), or run the docs site locally (`cd site && npm ci && npm run dev`).

<details>
<summary>All documentation</summary>

| doc | what it covers |
|---|---|
| [QUICKSTART](docs/QUICKSTART.md) | Walkthrough: your own data, fine-tuning, export, in-browser inference |
| [CLI](docs/CLI.md) | Every `folo` command, `--json`, exit codes, dataset tools |
| [LABELING](docs/LABELING.md) | Unlabeled images to a dataset with Claude Code, Codex or Antigravity |
| [BENCH](docs/BENCH.md) | Vision benchmarks from your own images: conditions, 5-axis scoring, loops |
| [CLOUD](docs/CLOUD.md) | Training on rented GPUs (RunPod) or remote FOLO servers with spot auto-resume |
| [AGENT_COOKBOOK](docs/AGENT_COOKBOOK.md) | Recipes for agents as MCP tool calls and `--json` commands |
| [MCP](docs/MCP.md) | The `folo mcp` contract: 45 tools, resources, prompts, errors |
| [API](docs/API.md) | The HTTP + SSE API behind Studio and Desktop |
| [SERVER](docs/SERVER.md) | Inside `folo-server`: router, Store, and background services |
| [BENCHMARKS](docs/BENCHMARKS.md) | FOLO vs Ultralytics on one machine: method, numbers, accuracy |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Repository map: crates, execution flows, where to change what |
| [MODEL](docs/MODEL.md) | Detector architecture, losses, and training recipe (normative spec) |
| [VISION](docs/VISION.md) | Tour of `folo-vision`: models, loss, assigner, trainer, predictor |
| [DATA](docs/DATA.md) | Dataset formats, augmentation, curation, and the data loader |
| [WASM](docs/WASM.md) | FOLO in the browser: build, JS API, WebGPU & SIMD128 performance |
| [CORE](docs/CORE.md) | `folo-core` in depth: tensors, CPU/SIMD/INT8 kernels, autograd |
| [GPU](docs/GPU.md) | The `wgpu` backend, buffer pool, and WGSL kernels |
| [NN](docs/NN.md) | Layers, optimizers, LR schedules, EMA, and `.folo` IO |
| [FORMAT](docs/FORMAT.md) | The `.folo` weights and checkpoint format |
| [METRICS](docs/METRICS.md) | How mAP, precision, recall, and COCOeval are computed |
| [ROADMAP](docs/ROADMAP.md) | What is done, in progress, and next |
| [RELEASE](docs/RELEASE.md) | How releases are cut, install scripts, and embedded Studio |
| [CI](docs/CI.md) | What each CI workflow runs and checks |
| [CHANGELOG](CHANGELOG.md) | What each release changed |

</details>

## Contributing & License

Issues, hardware measurements, and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md), and [AGENTS.md](AGENTS.md). Licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.
