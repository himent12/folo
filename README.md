# FOLO

**FOLO Only Looks Once.** A from-scratch, very fast vision framework in Rust: a YOLO-style real-time object detector
with its own tensors, autograd, CPU kernels and GPU kernels (wgpu: Vulkan, DX12, Metal, and WebGPU in the browser),
plus training, evaluation and export. You drive it from one `folo` binary, an HTTP API, an MCP server for agents,
**FOLO Studio** (a web app) or **FOLO Desktop** (a native GPUI app).

- **One binary, no Python.** `folo train`, `val`, `predict`, `export`, `serve` and `mcp` run on the same Rust engine.
  No PyTorch or CUDA toolkit is needed: the GPU kernels are our own WGSL, so a Vulkan, DX12 or Metal GPU that meets
  wgpu's compute limits works (`folo doctor` tells you which one FOLO picks).
- **Built for agents.** `--json` on every command, stable exit codes and error codes, `folo describe --json` for the
  whole CLI surface, and `folo mcp` for Claude Code, Codex or any MCP client.
- **Runs in the browser.** The same detector compiles to wasm and runs on WebGPU in the page (about 14 ms for folo-n
  at 640 in Chrome).

> **Status: early (v0.1).** The detector trains and infers on CPU and GPU, and it passes its convergence gate. There
> are no pretrained weights yet. See [Project status](#project-status).

## Why FOLO: the numbers

folo-n against Ultralytics **yolo26n** (8.4.198, torch 2.11 + CUDA 12.8) on the same machine: an RTX 4080 SUPER and an
i7-14700KF, both at 640×640. FOLO computes in strict f32 (no TF32, no tensor cores) and does about 19% more FLOPs per
image (6.54 vs 5.5 GFLOPs). Every number here comes from [docs/BENCHMARKS.md](docs/BENCHMARKS.md), which also covers
the method, the raw JSON and how much noise to expect.

| metric | folo-n | yolo26n (best config) | |
|---|---:|---:|---|
| GPU forward, batch 1, f32 (median latency) | **2.99 ms** | 4.95 ms (6.04 ms default) | FOLO 1.66× faster |
| GPU forward, batch 16, f32 (throughput) | 1385 img/s | 1454 img/s (TF32) | tie (within run-to-run noise) |
| Training step, batch 16, f32 (fwd + loss + bwd + SGD) | **219 img/s** | 202 img/s (fp32 with TF32) / 252 img/s (AMP) | FOLO faster than fp32; AMP still ahead |
| Training step peak GPU memory, batch 16, f32 | **3474 MiB** | 4517 MiB (2646 MiB AMP) | FOLO uses 23% less in f32 |
| GPU end to end, batch 1 (decoded image in, boxes out) | **5.18 ms** | 6.92 ms | FOLO 1.34× faster |
| GPU end to end, batch 16 | **722 img/s** (f32) | 615 img/s (fp16) / 536 (fp32) | FOLO faster than both |
| CPU forward, batch 1, f32 | **21.4 ms** (28 threads, contended machine) | 22.8 ms (8 threads) | FOLO slightly faster |
| CPU end to end, batch 1 | 26.9 ms | **24.8 ms** | FOLO behind |

FOLO is not ahead everywhere: the CPU end-to-end path (letterbox, decode and NMS around the forward) is still behind, and
the CPU numbers were measured on a contended machine, so treat them as upper bounds. f16 inference and the full trainer loop (data loader included) have not been benchmarked against the incumbent yet.

With the whole trainer (data loader, augmentation, validation), folo-n trains the shapes gate set (2,000 images
at 320 px, 30 epochs) to mAP50 0.980 in **169 s on the GPU**, 25.6× faster than
on the CPU ([examples/shapes](examples/shapes/README.md)).

## Quickstart

You need Rust (stable) and a GPU with Vulkan, DX12 or Metal. Without a GPU, `--device auto` falls back to the CPU,
which is much slower. `cargo folo` is an alias for `cargo run --release -p folo-cli --`, and the first call builds
the CLI (a few minutes):

```sh
git clone https://github.com/himent12/folo && cd folo
cargo folo doctor                                                         # the GPU FOLO picks
cargo folo dataset synth datasets/shapes --images 1000 --imgsz 320        # a labelled YOLO dataset, ~1 s
cargo folo train --data datasets/shapes --model folo-n --imgsz 320 --epochs 30   # ~2 min on the GPU
cargo folo predict --model runs/shapes-n/weights/best.folo datasets/shapes/images/val --save runs/shapes-n/pred
(cd web && npm ci && npm run build)                                       # build Studio's web app, once
cargo folo serve --mock                                                   # Studio on http://127.0.0.1:7878, demo data
```

`train` prints one row per epoch and writes `runs/shapes-n/`: `weights/best.folo`, `weights/last.folo` and
`metrics.jsonl`. On the dev machine this run ended at mAP50 0.948 / mAP50-95 0.814 after 137 s. `predict --save`
writes copies of the images with the boxes drawn on them.

To see real runs in Studio, drop `--mock`. `folo serve` keeps datasets, runs and models in its data dir
(`--data-dir`, default `~/.folo`), and lists every `.folo` file under `<data dir>/models/` as a model. To try your
model in the Playground, copy it there: `mkdir -p ~/.folo/models && cp runs/shapes-n/weights/best.folo
~/.folo/models/shapes-n.folo`.

**[docs/QUICKSTART.md](docs/QUICKSTART.md)** goes further: training on your own YOLO or COCO folder, fine-tuning,
evaluation, export, and running a model in the browser.

### Install a prebuilt binary

Tagged releases ship `folo` for x86_64 Windows and Linux and Apple-silicon macOS, with Studio built in, so `folo serve`
works anywhere. **No release has been published yet**, so build from source for now. Once one is published:

```sh
sh scripts/install.sh                     # Linux / macOS: into ~/.local/bin
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1            # into %LOCALAPPDATA%\Programs\folo\bin, added to PATH
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Desktop   # also FOLO Desktop
```

The scripts download the release for your platform, check its SHA-256 and install `folo`. The repository is private,
so they need a token: `GH_TOKEN` / `GITHUB_TOKEN`, or a logged-in [`gh`](https://cli.github.com). Options:
`--version` / `-Version`, `--dir` / `-InstallDir`, `--repo` / `-Repo` and `--base-url` / `-BaseUrl`, or the
environment variables `FOLO_VERSION`, `FOLO_INSTALL_DIR`, `FOLO_REPO` and `FOLO_BASE_URL`. To install from a checkout
instead: `cargo install --path crates/folo-cli` (add `--features embed-web` after building `web/`). See
[docs/RELEASE.md](docs/RELEASE.md).

## FOLO Studio

![FOLO Studio: a live training run](docs/img/studio-run-live.webp)

`folo serve` hosts Studio at `http://127.0.0.1:7878` (Svelte 5, our own design system). It has these screens:
- **Datasets**: register a YOLO or COCO folder or generate a synthetic one, browse the images with their boxes, and
  see class and box-size statistics and label issues.
- **Runs**: start a training run, follow it live (metrics over SSE, validation samples, per-class mAP, logs), stop
  or resume it, and compare runs.
- **Models**: evaluation results, per-class metrics, and export.
- **Playground**: run a model on any image, on the server or **in the browser** (WebGPU or the wasm CPU). The
  in-browser backends need the wasm module (`sh scripts/build-wasm.sh`) built before Studio. Thresholds re-filter
  instantly.

![FOLO Studio: the Playground](docs/img/studio-playground.webp)

`folo serve --mock` serves deterministic demo data, which is what these screenshots show. For UI work, run
`npm run dev` in `web/`, which starts the Vite dev server on :5173 with `/api` proxied to `folo serve`. See
[web/README.md](web/README.md).

## FOLO Desktop

![FOLO Desktop: a live training run](docs/img/desktop-run-live.webp)

The same screens, built natively in Rust on [GPUI](https://gpui.rs) (Zed's GPU-accelerated UI framework). It also has
a command palette, keyboard navigation, and dark and light themes. It talks to a local `folo serve` over the same
HTTP API. If no server answers on `$FOLO_SERVER` (default `http://127.0.0.1:7878`), it starts `folo serve --mock`
for you. If it can't, it falls back to its built-in demo data, and `--server` points it at a real server.

```powershell
cd apps/desktop
cargo run --release                                   # a running folo serve, else folo serve --mock, else demo data
cargo run --release -- --server http://127.0.0.1:7878 # a specific server
```

![FOLO Desktop: the image inspector](docs/img/desktop-inspector.webp)

Desktop is its own cargo workspace, so the engine's builds and CI stay GPUI-free. See
[apps/desktop/README.md](apps/desktop/README.md).

## Use FOLO from an agent

**MCP.** `folo mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio (MCP 2026-07-28
and 2025-11-25). It has 17 tools: build, register and check datasets, train, follow, stop and resume runs, evaluate,
predict, export, and compare runs. It also has `folo://` resources (dataset images, live run status and metrics,
validation sample grids, the docs) and two prompts. It runs the real engine on `--data-dir`.

```sh
claude mcp add folo -- folo mcp --data-dir D:/folo-data                 # Claude Code
codex mcp add folo -- folo mcp --data-dir D:/folo-data                  # Codex
```

Or in `.mcp.json` (`claude_desktop_config.json` for Claude Desktop):

```json
{ "mcpServers": { "folo": { "command": "folo", "args": ["mcp", "--data-dir", "D:/folo-data"] } } }
```

From a checkout without an installed `folo`, use `"command": "cargo", "args": ["run", "-q", "--release", "-p",
"folo-cli", "--", "mcp", "--data-dir", "D:/folo-data"]`. These flags change how it runs:
- `--device auto|cpu|gpu:<n>`: where inference and `val` run.
- `--mock`: simulated data and runs, to try the tools.
- `--read-only`: only tools that don't change state.
- `--confine`: only read files inside `--data-dir`.
- `--server http://127.0.0.1:7878`: share runs with a running `folo serve` and Studio.

**CLI with `--json`.** Every command takes `--json`: stdout is then JSON Lines only, and human text goes to stderr.
Failures end with `{"error":{"code","message","hint"}}` and a stable exit code (0 ok, 1 runtime error, 2 usage,
3 not found, 4 not implemented). `folo describe --json` prints every command, argument and default, plus every MCP
tool's schemas.

```sh
folo train --data datasets/shapes --model folo-n --epochs 30 --json   # {"event":"epoch",...} per epoch, then {"event":"done",...}
folo val --model runs/shapes-n/weights/best.folo --data datasets/shapes --json   # {"map50":...,"map50_95":...,"per_class":[...]}
```

Read on in [docs/AGENT_COOKBOOK.md](docs/AGENT_COOKBOOK.md) (recipes for both forms), [docs/MCP.md](docs/MCP.md)
(the MCP contract) and [llms.txt](llms.txt) (an index for agents).

## Architecture at a glance

```
 folo (CLI) ─────┬──────────────► folo-server (axum, HTTP + SSE) ◄── FOLO Studio (web/), FOLO Desktop (apps/desktop/)
 folo mcp ───────┘                     │  docs/API.md, types in folo-api
                                       ▼
                folo-vision: detector (folo-n/s/m/l/x), loss, assigner, NMS, mAP, trainer, predictor
                  │                      │                        │
             folo-nn: layers,       folo-data: YOLO / COCO,   folo-wasm: the predictor in the
             optimizers, .folo      augmentation, loader      browser (WebGPU or wasm CPU)
                  │
             folo-core: tensors, autograd, CPU kernels ◄── folo-gpu: wgpu device, WGSL kernels
```

Everything ML is written here: the kernels, autograd, layers, optimizers, detector, loss, assigner, NMS, metrics,
augmentation and GPU kernels. Commodity crates do the commodity jobs: wgpu, rayon, image decoding, serde, axum and
clap. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the full picture.

## Project status

What works today (v0.1):
- folo-n/s/m/l/x, trained from scratch on CPU or GPU, with resume, EMA, mosaic and the usual augmentations. On the
  shapes gate it reaches mAP50 0.980 / mAP50-95 0.896. On a real 2,030-frame dataset folo-n reaches 0.397 mAP50-95
  on the test split, trained from scratch ([examples/r6](examples/r6/README.md)).
- YOLO and COCO datasets: reading, stats, checks and conversion. mAP that matches pycocotools.
- `val`, `predict` and export to `.folo` (f32 or f16, a safetensors-compatible format) and ONNX (opset 17, f32 or
  f16, optional in-graph decode and dynamic shapes; parity-tested in onnxruntime).
- `folo serve` (Studio and the HTTP API), `folo mcp`, in-browser inference, and release builds and install scripts.

Not yet:
- pretrained weights;
- f16 compute on the GPU;
- `folo bench`;
- a published release.

The live checklist is [docs/ROADMAP.md](docs/ROADMAP.md).

## Docs

| doc | what it covers |
|---|---|
| [QUICKSTART](docs/QUICKSTART.md) | Walkthrough: your own data, fine-tuning, export, in-browser inference |
| [CLI](docs/CLI.md) | Every `folo` command, `--json`, exit codes, dataset tools |
| [AGENT_COOKBOOK](docs/AGENT_COOKBOOK.md) | Recipes for agents, as MCP tool calls and as `--json` commands |
| [MCP](docs/MCP.md) | The `folo mcp` contract: tools, resources, prompts, errors |
| [API](docs/API.md) | The HTTP + SSE API behind Studio and Desktop |
| [BENCHMARKS](docs/BENCHMARKS.md) | FOLO vs Ultralytics on the dev machine: method, numbers, targets |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | The crates and how they fit together |
| [MODEL](docs/MODEL.md) | The detector, loss and training recipe (normative spec) |
| [DATA](docs/DATA.md) | Dataset formats, augmentation, the data loader |
| [WASM](docs/WASM.md) | FOLO in the browser: the build, JS API, performance |
| [GPU](docs/GPU.md) | The wgpu backend, the buffer pool, the WGSL kernels |
| [NN](docs/NN.md) | Layers, optimizers, LR schedules, `.folo` IO |
| [FORMAT](docs/FORMAT.md) | The `.folo` weights and checkpoint format |
| [METRICS](docs/METRICS.md) | How mAP, precision and recall are computed |
| [ROADMAP](docs/ROADMAP.md) | What is done, in progress and next |
| [RELEASE](docs/RELEASE.md) · [CI](docs/CI.md) · [PROCESS](docs/PROCESS.md) · [AGENTS](AGENTS.md) | Releases, CI runners, how changes land, rules for contributors |

Examples: [shapes](examples/shapes/README.md) (the convergence gate, on CPU and GPU) and
[r6](examples/r6/README.md) (a real dataset trained from scratch).

To read these as a website with search: `cd site && npm ci && npm run dev` ([site/README.md](site/README.md)).

## License

Licensed under either of MIT or Apache-2.0 at your option.
