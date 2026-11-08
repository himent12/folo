# FOLO architecture

FOLO ("**F**OLO **O**nly **L**ooks **O**nce" — a play on YOLO) is a vision framework written from scratch in Rust: a
tensor engine with autograd, CPU and GPU kernels, neural-network layers, a YOLO-style real-time object detector, a
training and evaluation pipeline, a server, a CLI and a web studio.

```
            folo-cli  ──────────────┐            web/ (FOLO Studio)
                │                   │                   │ HTTP + SSE (docs/API.md)
                ▼                   ▼                   ▼
          folo-vision ◄──────── folo-server ◄───────────┘
         ╱     │     ╲
   folo-nn  folo-data  (box ops / NMS / metrics: plain slices)
      │
  folo-core ◄── folo-gpu (registers a GPU backend at runtime)
```

## folo-core — tensors and autograd
- `Tensor` is a cheap-to-clone handle (`Arc` inside) over shared storage plus a view (shape, strides, offset). Views
  (reshape of contiguous data, transpose, permute, narrow/slice, broadcast, squeeze/unsqueeze) do not copy.
- `DType`: `F32` (compute), `F16` (storage / GPU inference), `U8` (images), `U32`/`I64` (indices). Most ops are `F32`.
- `Device`: `Cpu` or `Gpu(ordinal)`. Kernels sit behind a `Backend` trait. The CPU backend lives in folo-core;
  folo-gpu implements the same trait and registers itself at runtime, so folo-core never depends on wgpu.
- CPU kernels: contiguous fast paths with rayon parallelism above a size threshold, a cache-blocked SIMD-friendly
  GEMM, conv2d via im2col + GEMM (with a direct path for depthwise and 1x1), pooling, reductions, elementwise and
  broadcasting binary ops. Strided inputs are made contiguous once, not walked per element in hot loops.
- Autograd: define-by-run, reverse mode. Ops on tensors that require grad record a node (inputs + backward closure)
  on the result. `loss.backward()` returns `Grads` (gradient per tensor id) via a topological sort. A thread-local
  `no_grad` guard disables recording (inference, optimizer updates). Every differentiable op ships a
  finite-difference gradient test.
- Seeded RNG (`folo_core::rng`) for reproducible init, augmentation and tests.

## folo-gpu — wgpu backend
- One `GpuDevice` per adapter (Vulkan / DX12 / Metal natively; WebGPU in the browser through wasm). Buffer pool with
  size buckets to avoid allocation churn; command batching so a forward pass is a few submissions, not hundreds.
- Kernels in WGSL: tiled GEMM (shared-memory tiles, vec4 loads), im2col / direct conv, elementwise + fused activations,
  reductions, softmax, pooling, upsample, concat. Each kernel is tested against the CPU reference.
- f16 storage/compute when the adapter supports `shader-f16`, with f32 accumulation.

## folo-nn — layers and training utilities
- `Module` trait: `forward`, `parameters()`/`named_parameters()` (stable, dotted names like `backbone.stem.conv.weight`),
  `train()`/`eval()` mode. Layers: `Conv2d` (groups, stride, padding, dilation), `BatchNorm2d` (running stats,
  momentum; fused into the conv for inference), `Linear`, activations (SiLU, ReLU, LeakyReLU, Sigmoid, Hardswish),
  `MaxPool2d`, `Upsample` (nearest), `Concat`, `Sequential`.
- Init: Kaiming / Xavier uniform/normal, seeded. Optimizers: SGD (momentum, Nesterov, weight decay) and AdamW, with
  parameter groups (no decay for BN and bias). Schedules: warmup + cosine / linear, EMA of weights.
- `.folo` weight files (`docs/FORMAT.md`): safetensors-compatible layout (u64 header length, JSON header, raw
  little-endian tensor data) with a `__metadata__` block holding the model config, class names and training info, so
  standard safetensors tools can read FOLO weights.

## folo-data — data pipeline
- Formats: YOLO txt (`images/` + `labels/`, `data.yaml`), COCO JSON; auto-detected. Dataset statistics (class counts,
  box-size distribution, empty images, broken labels) for Studio and `folo dataset check`.
- Image IO through `image` / `zune-jpeg`, decoded straight to `u8` RGB. Letterbox resize with exact box remapping.
- Augmentation (all box-aware, seeded): mosaic (4-image), random affine (scale / translate / shear / rotate), HSV
  jitter, horizontal flip, mixup, copy-paste later. Closing-mosaic for the last N epochs.
- `DataLoader`: worker threads that decode + augment ahead, a bounded channel, batches as contiguous `NCHW f32`
  (`0..1`) plus targets `[batch_idx, class, cx, cy, w, h]` (normalised). folo-data does not depend on folo-core; the
  trainer wraps the buffers in tensors.
- `synthetic` module: procedurally generated shapes datasets (circles / squares / triangles / stars on textured
  backgrounds, random colour, scale, occlusion) for tests, demos and benchmarks with no downloads.
- Details (formats, augmentation parameters, determinism): `docs/DATA.md`.

## folo-vision — the detector
- **Architecture** (per scale `n/s/m/l/x` via depth and width multipliers): a CSP-style backbone (Conv-BN-SiLU stem,
  C2f-like blocks with split + bottlenecks + concat), SPPF, a PAN-FPN neck, and an anchor-free decoupled head on three
  scales (strides 8 / 16 / 32). Box regression uses distribution focal loss bins (`reg_max`, default 16).
  An NMS-free one-to-one head (dual assignment during training, as in YOLOv10) is a later milestone.
- **Loss**: task-aligned assigner (TAL: `score^alpha * IoU^beta`, top-k), BCE classification on aligned targets,
  CIoU box loss and DFL, weighted 7.5 / 0.5 / 1.5 by default.
- **Box ops, NMS, metrics** work on plain slices so they are reusable everywhere: IoU / GIoU / DIoU / CIoU,
  xywh↔xyxy, class-aware batched NMS, COCO-style mAP@0.5 and mAP@0.5:0.95 (101-point interpolation), precision /
  recall / F1 curves, confusion matrix. `docs/METRICS.md` defines exactly how mAP is computed.
- **Trainer**: epochs, AMP-free f32 training on CPU or GPU, warmup, cosine LR, EMA, gradient clipping, validation every
  epoch, best/last checkpoints, metrics streamed as events (consumed by the CLI's progress UI, `--json` lines and the
  server's SSE).
- **Predictor**: letterbox, forward, decode, NMS, rescale; batch and streaming APIs; fused BN for inference.

## folo-server and folo-cli
- `folo-server` (axum + tokio) implements `docs/API.md`: datasets, models, training runs with live SSE metrics, and
  inference. `folo serve` runs it and serves the built Studio from `web/dist`. `folo serve --mock` serves realistic
  synthetic data for UI work before or without a trained model.
- `folo` CLI (`docs/CLI.md`): `train`, `val`, `predict`, `export`, `bench`, `dataset check|stats|synth`, `serve`,
  `mcp` (Model Context Protocol server over stdio so agents can train and run models), `doctor`. Every command takes
  `--json` (one JSON object per line on stdout, human text on stderr) and uses stable exit codes.

## Front ends
- **FOLO Studio** (`web/`): Svelte 5 + TypeScript + Vite, a custom design system (tokens, components) and no UI kit.
  Talks only to `docs/API.md`; `web/src/lib/api/` holds the typed client and a dev mock.
- **FOLO Desktop** (`apps/desktop/`): native Rust app on GPUI (Zed's GPU-accelerated UI framework). Same screens and
  design language as Studio. It talks to a local `folo serve` over the same API (it can spawn one), so the engine and
  the UI stay decoupled. It is a separate cargo workspace so the main workspace and its CI stay GPUI-free.

## Agent-first design
- Machine-readable everything: `--json`, JSON Schemas for configs, `folo describe` prints the CLI surface as JSON.
- `folo mcp` exposes tools: `list_datasets`, `dataset_stats`, `train`, `run_status`, `predict`, `val`, `export`.
- Errors carry a stable `code`, a human message and a hint.
- `llms.txt` at the repo root summarises the project for agents.
