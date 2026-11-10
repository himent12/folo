# Changelog

All notable changes to FOLO. Versions follow [Semantic Versioning](https://semver.org); the release process is in
[docs/RELEASE.md](docs/RELEASE.md).

## Unreleased

### Engine hill-climb
- Faster everywhere the GPU is used, measured against the previous build on the same machine
  ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#hill-climb-2026-11-09), notebooks in `docs/hill/`): folo-n
  batch-1 GPU forward 2.88 → 2.09 ms, end to end at batch 16 960 → 1898 img/s on tensor cores, the training step 242
  → 327 img/s in f32 and 345 → 537 with `--amp` (folo-s 121 → 151 / 194 → 273), a whole `folo train` epoch 216 → 347
  img/s (f32) and 302 → 500 (AMP), `folo val` on COCO val2017 29.6 → 8.9 s with the same mAP. On the CPU the training
  step is 25% faster and `predict_batch` 32–47%; batch-1 latency is unchanged.
- **`--optimizer musgd`** (Ultralytics 8.4's MuSGD: a Muon-orthogonalised update plus SGD) and two `auto` rule
  changes: rule 2 floors the AdamW LR of short runs at 0.001 (80-class runs from scratch learned almost nothing at
  the old 0.000119), rule 3 picks MuSGD for runs over 10,000 iterations (its +0.031 AP over SGD on the COCO proxy was measured with `--optimizer musgd` at momentum 0.937 and warmup_bias_lr 0.1; rule 3 ships momentum 0.9 and warmup_bias_lr 0, and no measured run has resolved `auto` to MuSGD (the proxy's 6,260 iterations and the shapes gate resolve AdamW), so the shipped rule-3 recipe itself has not been validated by a run; a long COCO run under rule 3 is the follow-up). Checkpoints resume with the rule they were
  started with (`TrainConfig.auto_rule`; a missing field is rule 1), so resumed runs keep their optimizer;
  `--set auto_rule=2` keeps SGD for a new long run. `--set KEY=VALUE` sets any `train` config key.
- The data loader defaults to up to 16 workers (was 8); the next epoch's loader starts once the previous epoch has taken
  its last batch (it works during validation and the checkpoint writes), and checkpoints download in one transfer
  per dtype and 128 MiB chunk.
- The GPU submits its command batch every 64 dispatches (was 1024), starting at 16 after an idle wait so the GPU
  starts early (`FOLO_GPU_FLUSH_EVERY` overrides it); `FOLO_GPU_PROFILE=1` turns on the per-kernel GPU timer
  (docs/GPU.md, Profiling).
- Gradient clipping runs in three multi-tensor kernels. A GPU run resumed from a checkpoint written by an earlier
  build continues normally, but not bit-identically to how that build would have continued while clipping is active
  (the sum of squares adds in another fixed order); resume within one build stays bit-exact.
- CPU `predict_batch` runs a batch as concurrent forwards of two images; each image's predictions are those it gets
  alone.

### Cloud compute
- Train a run on a GPU that is not in this machine: one rented from **RunPod** for the length of the run, or
  another FOLO server of yours (`remote`). The run stays a run of this server, with the same metrics, logs,
  samples and models ([docs/CLOUD.md](docs/CLOUD.md); `TrainConfig.compute`, HTTP API under `/api/cloud`,
  feature `cloud`; `folo cloud`; MCP `cloud_*` tools; Settings → Cloud and New run → Run on in Studio and
  Desktop).
- Nothing rented is left behind: every instance is written to a ledger before the provider is asked for it,
  released on every way out of a run (also a crash or a bug), terminated with retries until the provider confirms,
  and bounded by the run's time limit (`compute.max_hours`), which the instance also enforces on itself.
- A cloud run that trains does not need this server: close the laptop, and the next `folo serve` follows it again
  and fetches what it missed. A stopped or failed one resumes on a new instance from the checkpoint copied here.
- `folo serve --token` (bearer auth for a server that listens beyond loopback), `/api/uploads` (datasets and
  weights in retryable parts), cursors on a run's metrics and logs, `Range` on weights downloads and `checkpoint`
  on `POST /api/runs`: what makes any FOLO server usable as a worker, and useful on their own. `folo mcp --server`
  takes `--token` and `https://`.
- The RunPod provider is written against RunPod's REST API v2 and tested against a fixture of it; it has not been
  run against a live account yet (`crates/folo-cloud/tests/runpod_live.rs`).

### Agent labeling
- Unlabeled images to a dataset: `folo serve` hands a set of images to **Claude Code** or **Codex** on the
  server's machine, Studio shows every box for review and correction, and the labeled images become a
  registered YOLO dataset ([docs/LABELING.md](docs/LABELING.md); HTTP API under `/api/label`, feature
  `agent_labeling`).
- Built to be cheap: one model request per batch of images with the agent's tools, config, MCP servers and
  session switched off (a Claude Code call's fixed cost is a few hundred tokens; Codex's drops from about
  20000 to about 8000), images scaled to at most 1024 px, and a run that ends on the first failure another call
  would only repeat. On 24 COCO photos the default (Claude Code, Haiku) found 79% of the objects with 92% of its
  boxes right for under a cent.

## 0.1.0 (2026-11)

The first release: a from-scratch Rust vision framework with a YOLO-style detector, its own tensors, autograd, CPU
and GPU kernels, training, evaluation and export, driven from one `folo` binary, an HTTP API, an MCP server, FOLO
Studio (web) and FOLO Desktop (GPUI).

Headline numbers, folo-n against Ultralytics yolo26n on an RTX 4080 SUPER and an i7-14700KF at 640×640
([docs/BENCHMARKS.md](docs/BENCHMARKS.md); FOLO in strict f32 unless noted):

| | folo-n | yolo26n (best config) |
|---|---:|---:|
| GPU forward, batch 1 (median) | **2.99 ms** | 4.95 ms |
| GPU forward, batch 16 | 1385 img/s | 1454 img/s (TF32) |
| GPU end to end, batch 1 / batch 16 | **5.18 ms** / **722 img/s** | 6.92 ms / 615 img/s (fp16) |
| Training step, batch 16, f32 / AMP | **219** / **307 img/s** | 202 (TF32) / 252 img/s |
| Training step peak memory, f32 / AMP | **3474** / **2022 MiB** | 4517 / 2646 MiB |
| CPU forward, batch 1 | **21.4 ms** (contended) | 22.8 ms |
| CPU end to end, batch 1 | **21.7 ms** (quietest normal-priority round) | 24.8 ms |
| COCO val2017 AP, 50 epochs from scratch | 0.314 | 0.409 (Objects365 + 245 COCO epochs) |

### Engine
- folo-core: tensors, views, dtypes, autograd with finite-difference gradient checks, CPU kernels (GEMM about
  1 TFLOPS, AVX2 epilogues, pack-free conv, deterministic reductions) and a backend registry (#11, #39, #55, #137).
- folo-gpu: wgpu compute (Vulkan, DX12, Metal, WebGPU) with its own WGSL kernels: tiled SGEMM (about 31 TFLOPS),
  implicit-GEMM conv, fused Conv+BN+SiLU, BatchNorm and optimizer kernels, a buffer pool, whole models on the GPU
  (#10, #31, #38). f16 inference (#80) and opt-in tensor-core f16 GEMM and conv on cooperative matrices (#89).
- folo-nn: layers, init, SGD and AdamW, LR schedules, EMA, `.folo` weight files (a safetensors-compatible format,
  docs/FORMAT.md) (#1, #21).
- folo-vision: the folo-n/s/m/l/x detector family (#4, #28), box ops, IoU family, NMS, decode, task-aligned assigner,
  mAP and confusion matrix with pycocotools parity (#5, #15, #20), the NMS-free one-to-one head, `--head o2o` (#109).
- CPU end to end under the yolo26n bar: fused letterbox, a direct stem conv, in-place parallel decode (#127); per-core
  conv efficiency (B packing, stride-2 phase split, register epilogue) (#137).

### Training
- Detection loss (TAL + BCE + CIoU + DFL), the trainer and predictor; the shapes gate passes on CPU and GPU
  (#27, #33). Resume (bit-exact), EMA, mosaic and the usual augmentations; the folo-data pipeline with YOLO and COCO
  readers, statistics, checks, conversion and synthetic datasets (#8, #29).
- GPU training end to end with the loss and assigner on the device (#46); mixed-precision training, `--amp`, on
  tensor cores with dynamic loss scaling (#101).
- COCO 2017 support with 6x less loader memory, a pycocotools-exact COCOeval and `folo weights` to pull pretrained
  weights (#72); `folo-n-coco`, val2017 AP 0.314 / AP50 0.455 after 50 epochs from scratch, in the draft release
  `weights-v1` (#128). Fine-tuned from it, the R6 example reaches 0.441 test mAP50-95 in 100 epochs (0.350 from
  scratch).
- Short-run schedules, timing split and other dogfooding fixes (#69, #99).

### Inference and export
- `folo predict` and `folo val` on CPU or GPU, `--half` and `--tensor-cores` for f16 (#33, #80, #89).
- Export to `.folo` (f32 or f16) and to ONNX (opset 17, f32 or f16, raw or decoded, static or dynamic shapes),
  parity-tested against onnxruntime in CI (#87).
- folo-wasm: the detectors in the browser on WebGPU or the wasm CPU, about 14 ms for folo-n at 640 in Chrome (#51);
  shipped inside the release builds' Studio (#78) and, from this release, as the `folo-wasm.tar.gz` asset.

### Server, API and MCP
- `folo serve`: an axum HTTP + SSE API (docs/API.md, types in folo-api) with real datasets, runs, models, evaluation,
  export and predict, plus `--mock` demo data (#7, #42, #60).
- Capability reporting in `/api/system` (f16, tensor cores, AMP) (#108); `head`, `trained_with`, `precision` and
  `fused` on models (#117, #123); operating-point confidence, the resolved optimizer, queue control and ONNX download
  (#129); measured throughput, run estimates and a benchmark probe (#130); evaluation provenance (#138).
- `folo serve` releases a run's GPU memory when it ends and an idle model's after inference, lists published weights
  as models and refuses `--amp` up front where it cannot run (#140).
- `folo mcp`: an MCP server (rmcp, stdio) with 18 tools, `folo://` resources and prompts, embedded or proxying a
  running `folo serve`, able to drive real training end to end (#2, #36, #49).
- The CLI is agent-friendly: `--json` everywhere, stable exit and error codes, `folo describe --json`, `folo doctor`
  (#69).

### Studio
- FOLO Studio (Svelte 5, its own design system) embedded in the `folo` binary: Overview, Datasets, Runs (live
  metrics over SSE, compare, resume), Models (evaluation, export) and the Playground, which runs models on the server
  or in the browser (#3, #6, #13, #19, #57, #70).
- Design-review rounds 1 to 23; one responsive system from phone to desktop, with blank-section and overflow QA in CI
  (#126, #135, #141).

### Desktop
- FOLO Desktop, a native GPUI app with the same screens, a command palette, keyboard navigation and dark and light
  themes. It talks to `folo serve` over HTTP + SSE, starts `folo serve --mock` when no server answers, and runs local
  GPU or CPU inference in its Playground (#9, #32, #73).
- Design-review rounds 3 to 51 (#26 to #142).

### Docs
- README, QUICKSTART, CLI, API, MCP, MODEL, FORMAT, DATA, GPU, NN, WASM, METRICS and BENCHMARKS, an agent cookbook
  and `llms.txt` (#71); a static docs site generated from the Markdown with search and link checking (#122).
- Three dogfooding reports and their fixes (#59, #93, #134).
- Release builds for Windows and Linux with the embedded Studio, install scripts, checksums and a draft-release
  workflow on self-hosted runners (#22, #78, #104).

### Known issues
- No macOS build: the macOS job needs GitHub-hosted runners, which are unavailable for this repository, so
  `install.sh` stops on macOS. Build from source there.
- The repository is private: the install scripts need a token (`GH_TOKEN`, `GITHUB_TOKEN` or a logged-in `gh`).
- `weights-v1` (`folo-n-coco`) is still a draft release, so only repository writers can `folo weights pull` it.
