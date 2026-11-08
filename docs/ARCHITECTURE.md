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
- `Tensor` is a cheap-to-clone handle (`Arc` inside) over shared storage plus a `Layout` (shape, strides, offset).
  Views (reshape of compatible layouts, transpose/permute, narrow/slice with step, broadcast/expand,
  squeeze/unsqueeze, flatten) do not copy. Ops return `Result`; `+ - * /` (on `Tensor`, `&Tensor`, `f32`) return
  `Result<Tensor>` too. Errors name the op and both shapes.
- `DType`: `F32` (compute), `F16` (storage / GPU inference), `U8` (images), `U32`/`I64` (indices). Arithmetic is `F32`;
  copies, views, `cat`, `index_select`, `gather` and casts work for every dtype. Comparisons return 0/1 `F32` masks.
- `Device`: `Cpu` or `Gpu(ordinal)`. Kernels sit behind the `Backend` trait (below); the CPU backend lives in folo-core.
- CPU kernels: contiguous fast paths with rayon above ~32K elements; strided inputs are walked row by row after merging
  dimensions, or made contiguous once. A BLIS-style GEMM (`cpu::gemm`, packed panels, 6x16 AVX2/FMA micro-kernel
  chosen at runtime, tile- or K-split parallelism) backs `matmul` and conv2d. conv2d packs an implicit im2col operand
  straight into GEMM panels (no unfolded matrix), with direct paths for 1x1 and depthwise convolutions. Large `f32`
  buffers are recycled through a size-keyed cache (`cpu::set_buffer_cache_limit`, default 1 GiB) so training loops do
  not pay page faults every step.
- Autograd: define-by-run, reverse mode. An op whose input requires grad records a node (inputs, their storage
  versions, a backward closure built from ordinary tensor ops) on the result. `loss.backward()` returns `Grads` (keyed
  by tensor id, leaves only) via a topological sort; graphs are retained, so backward can run twice. Graph teardown is
  iterative (deep graphs do not overflow the stack). A thread-local `no_grad()` guard disables recording.
- In-place updates for optimizers (`sub_assign`, `add_scaled_assign`, `copy_from`, `fill_assign`, `update_f32`, ...)
  refuse to touch a tensor that requires grad unless `no_grad` is active, and bump the storage version: backward
  through a graph that saved the old values fails with `Error::ModifiedInPlace` instead of using stale data.
- `testing::grad_check` compares autograd against central finite differences (f64 loss accumulation); every
  differentiable op has a check in `crates/folo-core/tests/grad.rs`.
- Custom ops: `Tensor::custom_op(name, inputs, output, backward)` lets downstream crates record fused kernels
  (Conv+BN+SiLU, BatchNorm, DFL / CIoU losses) with a hand-written backward; inputs and output are version-checked
  like built-in ops and a wrong gradient count or shape is an error at `backward()`.
- Seeded RNG (`folo_core::rng::Rng`, xoshiro256++) for reproducible init, augmentation and tests. `Rng::ALGO`
  (`"xoshiro256pp"`), `state()` and `from_state()` save and resume the exact stream (the `folo.rng` checkpoint entry).

### Backend integration contract (for folo-gpu and other devices)
- Implement `folo_core::backend::Backend` and call `register_backend(Device::Gpu(i), Arc::new(backend))` at startup
  (for example from `folo_gpu::init()`). `backend(device)` looks it up; tensors on an unregistered device fail with
  `Error::BackendNotRegistered`.
- Device memory is `Storage::Device(DeviceStorage)`: folo-core stores the device, dtype and element count and an opaque
  `Box<dyn Any + Send + Sync>` buffer that only the backend downcasts. Build results with `DeviceStorage::new` and
  wrap foreign buffers into tensors with `Tensor::from_storage`.
- Every method receives `View`s (storage + `Layout`). Layouts may be strided or broadcast (stride 0); outputs are fresh
  contiguous storages on the same device. folo-core validates shapes, dtypes and devices before calling; binary,
  `where_cond` and `matmul` operands arrive already broadcast to the output shape.
- Required: `name`, `full`, `upload`, `download` (to host, logical order), `copy_strided`, `copy_into` (strided write
  used by `cat`, `pad`, view gradients and `copy_from`), `cast`. Everything else (unary + `unary_backward`, binary,
  `where_cond`, `reduce`, `softmax`, `matmul`, conv2d forward / backward-input / backward-weight, pooling and
  upsampling with their backwards, `index_select`/`index_add`/`gather`/`scatter_add`, `topk`) defaults to
  `Error::NotSupported { op, backend }`, so a backend can grow op by op. Gradients are composed from the same methods,
  so an op is trainable on a device once its forward and `*_backward` methods exist.
- `max_pool2d_backward` also receives the pooling window (`Pool2dParams`), so a backend can compute the gradient as a
  gather (each input element visits only the windows that cover it) instead of a scatter with float atomics; the
  CPU backend ignores it.
- `synchronize` (default no-op) waits for queued device work and reports asynchronous errors;
  `folo_core::backend::synchronize(device)` calls it. Backends may queue work lazily: only `download` (`to_vec`,
  `to_device(Cpu)`, `item`) and `synchronize` have to wait.
- Tensors move between devices with `to_device` (autograd flows back across the transfer). The CPU backend is the
  reference: each GPU kernel should be tested against it within a stated tolerance.

## folo-gpu — wgpu backend
- One `GpuDevice` per adapter (Vulkan / DX12 / Metal natively; WebGPU in the browser through wasm). Buffer pool with
  size buckets to avoid allocation churn; command batching so a forward pass is a few submissions, not hundreds.
- Kernels in WGSL: tiled GEMM (shared-memory tiles, vec4 loads), im2col / direct conv, elementwise + fused activations,
  reductions, softmax, pooling, upsample, concat. Each kernel is tested against the CPU reference.
- f16 storage/compute when the adapter supports `shader-f16`, with f32 accumulation.
- Implements folo-core's `Backend` (`folo_gpu::GpuBackend`): `folo_gpu::init()` registers the default adapter as
  `Device::Gpu(0)`, after which `tensor.to_device(Device::Gpu(0))` keeps data and every op (forward and the gradient
  ops autograd records) on the GPU. Work is encoded into the context's open batch and only waits on readback, so
  a forward + backward pass is a handful of queue submissions. Every dtype is stored (`u8` / `f16` packed into
  words, `i64` as word pairs); strided views are read in place by elementwise ops, copies and matmul.
- A standalone `GpuTensor` over the same kernel layer stays for kernel-level work. Kernel list, layout
  conventions, the backend's op mapping, GEMM tuning notes and benchmarks: `docs/GPU.md`.

## folo-nn — layers and training utilities
- `Module` trait: `parameters()`/`named_parameters()`/`named_buffers()` (stable, dotted names like
  `backbone.stem.conv.weight`), `train()`/`eval()` mode, submodule visitor; `Forward` adds `forward` for one-in,
  one-out layers. Overview: [docs/NN.md](NN.md). Layers: `Conv2d` (groups, stride, padding, dilation), `BatchNorm2d` (running stats,
  momentum; fused into the conv for inference), `Linear`, activations (SiLU, ReLU, LeakyReLU, Sigmoid, Hardswish),
  `MaxPool2d`, `Upsample` (nearest), `Concat`, `Sequential`.
- Init: Kaiming / Xavier uniform/normal, seeded. Optimizers: SGD (momentum, Nesterov, weight decay) and AdamW, with
  parameter groups (no decay for BN and bias). Schedules: warmup + cosine / linear, EMA of weights.
- `.folo` weight files (`docs/FORMAT.md`): safetensors-compatible layout (u64 header length, JSON header, raw
  little-endian tensor data) with a `__metadata__` block holding the model config, class names and training info, so
  standard safetensors tools can read FOLO weights.

## folo-data — data pipeline
- Formats: YOLO txt (`images/` + `labels/`, `data.yaml`), COCO JSON; auto-detected. Dataset statistics (class counts,
  box-size distribution, empty images, broken labels) for Studio and `folo dataset stats`; a line-level checker
  (`check`) for `folo dataset check` and YOLO / COCO conversion (`convert`).
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
The exact block definitions, layer tables, parameter names, head, loss, training recipe and verification plan are in
**[docs/MODEL.md](MODEL.md)**, the normative spec for the detector. In short:
- **Architecture** (per scale `n/s/m/l/x` via depth and width multipliers): a CSP-style backbone (Conv-BN-SiLU stem,
  C3k2 blocks with split + bottlenecks + concat), SPPF, one C2PSA attention block at P5, a PAN-FPN neck, and an
  anchor-free decoupled head on three scales (strides 8 / 16 / 32). Box regression uses distribution focal loss bins
  (`reg_max`, default 16). An NMS-free one-to-one head (dual assignment during training, as in YOLOv10) is specified
  in MODEL.md but reserved for a later milestone. Implemented in `folo_vision::model`: `FoloModel::new(config)` for
  any scale and class count, `forward` → `HeadOutput { box_raw [B, 64, A], cls_raw [B, nc, A], anchors }`, `fuse()`,
  `predict_raw` (forward + CPU decode + NMS), and `.folo` save / load / transfer through folo-nn.
- **Loss**: task-aligned assigner (TAL: `score^alpha * IoU^beta`, top-k), BCE classification on aligned targets,
  CIoU box loss and DFL, weighted 7.5 / 0.5 / 1.5 by default. `folo_vision::loss` runs the assigner on host copies of the detached
  head outputs, then builds the terms from folo-core tensor ops (fused custom ops for the full-size BCE and gather on
  the CPU), so gradients flow into both head branches.
- **Box ops, NMS, metrics** work on plain slices so they are reusable everywhere: IoU / GIoU / DIoU / CIoU,
  xywh↔xyxy, class-aware batched NMS, COCO-style mAP@0.5 and mAP@0.5:0.95 (101-point interpolation), precision /
  recall / F1 curves, confusion matrix. `docs/METRICS.md` defines exactly how mAP is computed.
- **Trainer**: epochs, AMP-free f32 training on CPU or GPU, warmup, cosine LR, EMA, gradient clipping, validation every
  epoch, best/last checkpoints, metrics streamed as events (consumed by the CLI's progress UI, `--json` lines and the
  server's SSE).
- **Predictor**: letterbox, forward, decode, NMS, rescale; batch and streaming APIs; fused BN for inference.

## folo-server and folo-cli
- `folo-server` (axum + tokio) implements `docs/API.md`: datasets, models, training runs with live SSE metrics, and
  inference. Handlers sit on a `Store` trait (`MockStore` for `--mock`, `RealStore` for the engine; its datasets
  are read with folo-data and indexed into a JSON cache under the data dir, invalidated by file mtimes). The request and
  response types live in `folo-api`, a dependency-light crate (serde only) that FOLO Desktop reuses. `folo serve` runs it and serves the built Studio from `web/dist`, or from the copy compiled into the binary by
  the `embed-web` feature (release builds, see `docs/RELEASING.md`). `folo serve --mock` serves realistic
  synthetic data for UI work before or without a trained model.
- `folo` CLI (`docs/CLI.md`): `train`, `val`, `predict`, `export`, `bench`, `dataset check|stats|synth|convert`, `serve`,
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
