# FOLO architecture

FOLO ("**F**OLO **O**nly **L**ooks **O**nce" — a play on YOLO) is a vision framework written from scratch in Rust: a
tensor engine with autograd, CPU and GPU kernels, neural-network layers, a YOLO-style real-time object detector, a
training and evaluation pipeline, a server, a CLI and a web studio.

```
            folo-cli  ──────────────┐            web/ (FOLO Studio)
                │                   │                   │ HTTP + SSE (docs/API.md)
                ▼                   ▼                   ▼
          folo-vision ◄──────── folo-server ◄───────────┘
         ╱     │     ╲              │  ╲
   folo-nn  folo-data  (box ops)    │   folo-cloud ──► a provider's API (launch, look up, terminate)
      │                             ▼
  folo-core ◄── folo-gpu       folo-client ══► another folo serve (a cloud worker), over docs/API.md
```

## folo-core — tensors and autograd
- `Tensor` is a cheap-to-clone handle (`Arc` inside) over shared storage plus a `Layout` (shape, strides, offset).
  Views (reshape of compatible layouts, transpose/permute, narrow/slice with step, broadcast/expand,
  squeeze/unsqueeze, flatten) do not copy. Ops return `Result`; `+ - * /` (on `Tensor`, `&Tensor`, `f32`) return
  `Result<Tensor>` too. Errors name the op and both shapes.
- `DType`: `F32` (compute), `F16` (storage / GPU inference), `U8` (images), `U32`/`I64` (indices). Arithmetic is `F32`;
  copies, views, `cat`, `index_select`, `gather` and casts work for every dtype. Comparisons return 0/1 masks of their
  operands' dtype. On a backend that computes in f16 (`Backend::computes_f16`: the GPU with `shader-f16`) the forward
  ops also take `F16` operands and return `F16` (half-precision inference); every float operand of an op has one dtype.
  The dtype lives with the storage, like the device, so `Tensor::to_dtype_` / `Module::to_dtype` convert in place.
- `Device`: `Cpu` or `Gpu(ordinal)`. Kernels sit behind the `Backend` trait (below); the CPU backend lives in folo-core.
- CPU kernels: contiguous fast paths with rayon above ~32K elements; strided inputs are walked row by row after merging
  dimensions, or made contiguous once. A BLIS-style GEMM (`cpu::gemm`, packed panels, 6x16 AVX2/FMA micro-kernel
  chosen at runtime, tile- or K-split parallelism) backs `matmul` and conv2d. Each GEMM task packs the B panels of
  its own columns into an L2-sized buffer one `KC` block at a time, reading row-contiguous operands row by row so
  the prefetcher can follow (column tiles first, rows only when there are too few columns, about three tasks per
  thread and none under ~1 MFLOP). conv2d (any kernel, stride, padding or
  dilation, not depthwise) never builds an im2col matrix: the input is copied once, zero-padded and split into its
  stride phases, so every `(channel, ky, kx)` row of B is a contiguous run of that copy, read in place by a variant
  of the micro-kernel through a row-pointer table (a few dropped columns per output row; planes are spaced an odd
  number of cache lines apart so the rows do not alias in L1; stride-2 rows are padded once and both column phases
  dealt out in one pass). Pointwise convs pack B as a plain copy; depthwise convs have a direct kernel (with AVX2 at
  stride 1: the plane copied once with its padding, every output vector its taps' FMAs in registers), and so do convs with at most 4 input channels (the 3-channel stem:
  per output row the `c·kh·kw` taps are gathered into L1 and every output channel is that many FMAs per pixel in
  the GEMM's order, bit for bit the GEMM result, without the per-channel phase copy). Bias and activation run as an
  epilogue on each finished output (`Tensor::conv2d_act`, used by fused `ConvBnAct` blocks and by eval-mode blocks
  under `no_grad`): bias and SiLU on the micro-kernel's accumulators before they are stored, other activations as
  a pass over each finished block; SiLU there is an 8-lane AVX2 `x · sigmoid(x)` with relative error below
  `cpu::SILU_MAX_REL_ERR` (5e-7, tested).
  `Tensor::conv2d_act_cat` runs the same conv over the channel concatenation of several inputs, reading each part
  (or a channel slice of a tensor) in place, so `cat` + conv never writes the concatenation. Max pooling is
  separable (row pass, then column pass; same first-maximum argmax as a direct scan). Batches smaller than the
  thread pool split the pool between images. The stride-1 input gradient is the forward kernel run on the output gradient with the flipped,
  channel-swapped kernel (no col2im); strided convs use an unfolded gradient and col2im. `f32` buffers from 16 KiB
  are recycled through a size-keyed cache (`cpu::set_buffer_cache_limit`, default 1 GiB; `cpu::take_buffer` hands one
  to custom ops) so training and inference loops do not pay page faults every step. `cpu::in_pool` runs a closure on
  a worker of a dedicated rayon pool (so the per-thread state cannot reach other jobs of the global pool) with the
  caller's grad mode and autocast state; CPU forwards and predictions use it so their many parallel kernels start
  inside a pool instead of each waking one from outside.
- Determinism: CPU results are bitwise identical from run to run and for every rayon thread count. Full and
  long-row reductions use fixed blocks (8192 elements) combined by a fixed-order pairwise tree; reductions over
  inner dims give each output element to one task; the GEMM's K split depends only on the shape; accumulating
  scatters (`index_add`, `scatter_add` and the `index_select` / `gather` / `topk` gradients) partition work by
  destination and add sources in index order. `crates/folo-core/tests/determinism.rs` checks this across pool
  sizes.
- Autograd: define-by-run, reverse mode. An op whose input requires grad records a node (inputs, their storage
  versions, a backward closure built from ordinary tensor ops) on the result. `loss.backward()` returns `Grads` (keyed
  by tensor id, leaves only) via a topological sort; graphs are retained, so backward can run twice. Graph teardown is
  iterative (deep graphs do not overflow the stack). A thread-local `no_grad()` guard disables recording. Fan-in
  gradients are summed in place when the running sum is a fresh tensor nothing else holds.
- In-place updates for optimizers (`sub_assign`, `add_scaled_assign`, `copy_from`, `fill_assign`, `update_f32`, ...)
  refuse to touch a tensor that requires grad unless `no_grad` is active, and bump the storage version: backward
  through a graph that saved the old values fails with `Error::ModifiedInPlace` instead of using stale data.
- `testing::grad_check` compares autograd against central finite differences (f64 loss accumulation); every
  differentiable op has a check in `crates/folo-core/tests/grad.rs`.
- Custom ops: `Tensor::custom_op(name, inputs, output, backward)` lets downstream crates record fused kernels
  (Conv+BN+SiLU, BatchNorm, DFL / CIoU losses) with a hand-written backward; inputs and output are version-checked
  like built-in ops and a wrong gradient count or shape is an error at `backward()`.
- Fused ops with composite fallbacks (`ops/fused.rs`): `Tensor::conv2d_act` (conv + bias + activation),
  `Tensor::conv2d_act_cat` (the same over a channel concatenation, without writing it),
  `Tensor::batch_norm_act_train` / `batch_norm_act_train_backward` (train-mode BatchNorm + activation, running
  update in place) and `Tensor::optimizer_step_` (in-place SGD / AdamW / EMA update) call optional `Backend` hooks;
  folo-gpu implements them, the CPU backend implements `conv2d_act` and `conv2d_act_cat`, other backends fall
  back.
- Detector pre- and post-processing hooks (`ops/vision.rs`, `Ok(None)` without a kernel so callers keep their host
  code): `Tensor::letterbox_rgb8` (raw `u8` images → letterboxed `f32` NCHW input, sampled with folo-data's exact
  resize taps) and `Tensor::detect_candidates` (head → per-anchor best class above `conf` with its decoded box, as
  a compact list). folo-gpu implements both.
- `Tensor::to_device_(device)` moves a storage to another device **in place**: the device lives on the shared storage
  cell, so every clone and view follows and the tensor keeps its id (gradients and optimizer state stay keyed). This
  is how whole models move (`Module::to_device`).
- Mixed precision (`folo_core::autocast`, like `torch.autocast`): while an `autocast()` guard is alive on the thread,
  `conv2d_with` and `matmul` of tensors on a backend that computes in f16 run in f16 (`F32` operands cast by
  recorded casts; a conv uses f16 copies of its weight and bias but is recorded against the f32 ones, and its
  backward writes the weight gradient straight to f32 with `Backend::conv2d_backward_weight_as`), and softmax runs
  in f32. Everything downstream runs in the dtype it receives; the fused train-mode BatchNorm takes f16 data with
  f32 parameters and statistics. `Tensor::unscale_check_` (a `Backend::unscale_check` hook) is the loss scaler's
  fused unscale and overflow flag.
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
- Optional gradient fast paths with composing defaults: `index_select_backward` and `gather_backward` (default
  `full` zeros then `index_add` / `scatter_add`).
- `max_pool2d_backward` also receives the pooling window (`Pool2dParams`), so a backend can compute the gradient as a
  gather (each input element visits only the windows that cover it) instead of a scatter with float atomics; the
  CPU backend ignores it.
- Optional fused hooks (`conv2d_act`, `conv2d_act_cat`, `batch_norm_act`, `batch_norm_act_backward`,
  `optimizer_step`, `letterbox_rgb8`, `detect_candidates`) default to
  `NotSupported`; folo-core's fused entry points then compose ordinary ops (or report "not available" so folo-nn
  uses its own path). `batch_norm_act` and `optimizer_step` write into `&mut Storage` arguments (running statistics,
  parameters, optimizer state) that folo-core has write-locked, checked to be contiguous and distinct.
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
  ops autograd records) on the GPU. It also implements the fused hooks: Conv + bias + activation in the conv
  epilogue, train-mode BatchNorm + activation forward / backward kernels, and in-place optimizer updates. Work is encoded into the context's open batch and only waits on readback, so
  a forward + backward pass is a handful of queue submissions. Every dtype is stored (`u8` / `f16` packed into
  words, `i64` as word pairs); strided views are read in place by elementwise ops, copies and matmul.
- A standalone `GpuTensor` over the same kernel layer stays for kernel-level work. Kernel list, layout
  conventions, the backend's op mapping, GEMM tuning notes and benchmarks: `docs/GPU.md`.

## folo-nn — layers and training utilities
- `Module` trait: `parameters()`/`named_parameters()`/`named_buffers()` (stable, dotted names like
  `backbone.stem.conv.weight`), `train()`/`eval()` mode, submodule visitor, `to_device()` (in place, identities
  kept; optimizers and EMAs have their own `to_device`); `Forward` adds `forward` for one-in, one-out layers. Overview: [docs/NN.md](NN.md). Layers: `Conv2d` (groups, stride, padding, dilation), `BatchNorm2d` (running stats,
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
  (`reg_max`, default 16). `head = "o2o"` adds the NMS-free one-to-one branch (`head.o2o.*`, MODEL.md §6.6, as in
  YOLOv10 / YOLO26): trained with dual assignment on detached features, it predicts with a top-k selection instead
  of NMS (`nms::topk_detections`), and only the predicting branch runs in eval mode (`FoloModel::head_mode`).
  Implemented in `folo_vision::model`: `FoloModel::new(config)` for
  any scale and class count, `forward` → `HeadOutput { box_raw [B, 64, A], cls_raw [B, nc, A], anchors }`, `fuse()`,
  `predict_raw` (forward + decode + NMS; decode runs on the outputs' device, NMS on the CPU),
  `predict_detections` / `HeadOutput::detections` (on a GPU the decode runs in one kernel that keeps only the
  candidates above `conf`, and only those are downloaded), `to_device()`, and
  `.folo` save / load (`open_on(path, device)`) / transfer through folo-nn.
- **Loss**: task-aligned assigner (TAL: `score^alpha * IoU^beta`, top-k), BCE classification on aligned targets,
  CIoU box loss and DFL, weighted 7.5 / 0.5 / 1.5 by default. On the CPU `folo_vision::loss` runs the assigner on the
  detached head outputs, then builds the terms from folo-core tensor ops (fused custom ops for the full-size BCE and
  gather), so gradients flow into both head branches. On a GPU the whole loss runs as folo-gpu kernels
  (`folo_gpu::detect`: assigner, fused BCE, fused CIoU + DFL with a hand-written backward, as one custom op) and
  only the reported items are read back.
- **Box ops, NMS, metrics** work on plain slices so they are reusable everywhere: IoU / GIoU / DIoU / CIoU,
  xywh↔xyxy, class-aware batched NMS, COCO-style mAP@0.5 and mAP@0.5:0.95 (101-point interpolation), precision /
  recall / F1 curves, confusion matrix. `docs/METRICS.md` defines exactly how mAP is computed.
- **Trainer** (`folo_vision::train::Trainer`, MODEL.md §8): folo-data's `DataLoader` (seeded per epoch and sample, so
  batches do not depend on the worker count) → tensors → train-mode forward → loss → backward into a
  `GradAccumulator` (nominal batch 64) → schedule (warmup + linear / cosine, in optimizer steps), global-norm
  clipping, SGD / AdamW with `decay` / `bn` / `bias` groups (`auto` picks per §8.2), EMA. Batches with a non-finite
  loss or BatchNorm statistic are skipped and their running statistics restored. After every epoch: validation of the
  EMA model with `metrics::Evaluator`, fitness mAP@0.5:0.95, `last.folo` / `best.folo` checkpoints (complete
  FORMAT.md state, atomic), early stopping. Resume from a checkpoint is bit-exact. Events (`status`, rate-limited
  `step`, `epoch`, `log`: the API.md payloads as `folo_api::RunEvent`) go to a callback and to the run's
  `metrics.jsonl`; the CLI's progress view, `--json` and the server consume them. The device is a parameter:
  on a GPU (`place_model` moves the model; optimizer, EMA and gradient buffers follow) a feeder thread stages each
  `u8` batch for upload while the previous step runs, accumulation and clipping stay on the device, and the step
  waits only for the loss items and one BatchNorm finiteness check. Mixed precision (`amp`, MODEL.md §8.8; GPU
  tensor cores only, else `not_supported`): the forward under `autocast` (f16 convs and products on tensor cores),
  the loss in f32 on f32 copies of the head, backward of the loss times folo-nn's `GradScaler` scale, and at the end
  of a window the unscale, clip and overflow check with one flag readback (an overflowing window skips its step and
  backs the scale off); master weights, optimizer state, EMA and BatchNorm statistics stay f32, and the scaler is
  the checkpoint's `folo.amp`. `folo_vision::val::validate`
  is the same validation for `folo val`.
- **Predictor** (`folo_vision::predict::Predictor`): loads `.folo` (EMA when present), fuses BN, letterboxes (rect for
  one image, square for a batch), forward, decode, NMS, rescale to the original image, with a preprocess /
  inference / postprocess timing split. On a GPU the raw `u8` images are uploaded and letterboxed there
  (bit-exact with the CPU letterbox) and only the decoded candidates come back: one upload per image, one queue of
  work, one small readback. On the CPU the whole prediction runs inside the rayon pool (`cpu::in_pool`), the
  letterbox is one fused pass per image (`Letterbox::apply_planar_f32`: resize, padding and `/ 255` straight into
  planar f32, rows in parallel) into an input buffer from folo-core's
  buffer cache, the head's six branch chains run side by side, the neck's Concat layers are never written
  (`C3k2::forward_cat`), and the decode reads the head outputs in place and extracts candidates in blocks of 1024
  anchors in parallel.
- **ONNX export** (`folo_vision::export::onnx`): `FoloModel::to_onnx` turns a fused model into an opset-17 graph
  that mirrors `forward` op for op (raw head outputs, or the MODEL.md §6.4 decode in the graph, which for an `o2o`
  model ends in the §6.6 top-k: end-to-end `[B, K, 6]` detections; fixed or dynamic
  shapes; f32 or f16), encoded by a small hand-written protobuf writer (`export::onnx::proto`), with the class names
  and `folo.*` metadata in `metadata_props`. Any model tensor no node consumed, or an unfused BatchNorm, is an error
  naming it. `folo export --format onnx` and `POST /api/models/{id}/export` call `export_file`; parity with
  onnxruntime is `crates/folo-cli/tests/onnx.rs` + `bench/parity/onnx_check.py` (docs/CLI.md, ONNX export).

## folo-wasm — FOLO in the browser
- `crates/folo-wasm` (wasm-bindgen) loads a `.folo` file from bytes and runs `folo_vision::Predictor`'s steps on
  WebGPU (folo-gpu's browser backend, async device creation and readback) or on folo-core's CPU kernels compiled to
  wasm (single-threaded, built with SIMD128). `detect` resolves to `/api/predict`'s response shape plus the
  backend, so Studio can run models in the page. `scripts/build-wasm.sh` builds it into `web/public/folo-wasm/`.
  JS API, build and limits: `docs/WASM.md`.

## folo-server and folo-cli
- `folo-server` (axum + tokio) implements `docs/API.md`: datasets, models, training runs with live SSE metrics, and
  inference. Handlers sit on a `Store` trait (`MockStore` for `--mock`, `RealStore` for the engine; its datasets
  are read with folo-data and indexed into a JSON cache under the data dir, invalidated by file mtimes). The real
  store trains with folo-vision's `Trainer` on one background worker thread (a FIFO queue, one run at a time), fans
  the trainer's events out to SSE subscribers through a per-run broadcast channel and persists each run under
  `<data dir>/runs/<id>/`; finished runs' checkpoints and `.folo` files under `<data dir>/models` are its models
  (`/eval` runs folo-vision's validator, cached); `/predict` keeps an LRU of loaded, fused `Predictor`s. Devices go
  through folo-vision's `resolve_device` / `can_place` (`auto` is the GPU when folo-gpu finds an adapter, else the CPU) and GPU info
  comes from folo-gpu. The request and response types live in `folo-api`, a dependency-light crate (serde only) that
  FOLO Desktop reuses. `folo serve` runs it and serves the built Studio from `web/dist`, or from the copy compiled into the binary by
  the `embed-web` feature (release builds, see `docs/RELEASE.md`). `folo serve --mock` serves realistic
  synthetic data for UI work before or without a trained model.
- **Cloud compute** (`folo_server::cloud`, docs/CLOUD.md): a run with `TrainConfig.compute` trains on another
  machine's GPU and stays a run of this server. The worker is an ordinary `folo serve` reached through
  `folo-client` with the public API: a token (`--token`), uploads in parts (`/api/uploads`, a dataset as the tar
  stream `folo_data::pack` writes), cursors on a run's history and `Range` downloads of its weights are all it
  needs, so any FOLO server can be one. `folo-cloud` holds the providers behind one trait (`Provider`: check
  credentials, list offers, launch, look up, find by name, terminate): `RunPod` rents a pod per run and starts the
  worker with a bootstrap script; `Remote` is a server that already runs. `Cloud` is the control plane of a data
  dir: connections (`cloud/connections.json`, owner-only, never served), the instance ledger
  (`cloud/instances/<id>.json`, written before the provider is called), a reaper thread that retries terminations
  and cleans up after crashes, and one executor thread per cloud run through the phases provisioning → booting →
  syncing → training → collecting → releasing. A cloud run shares the run registry, folder and event sink with
  local runs (`RunEntry::execute` is the one claim-execute-publish path) but not the queue or the device; its
  worker's history is mirrored poll by poll into the same `metrics.jsonl`, `log.jsonl` and `samples/`, so everything
  downstream of a run is unchanged. A guard releases the instance on every way out, including a panic; a run that
  trains survives this server (`runs/<id>/cloud.json` and an OS lock on `cloud.lock` let the next process follow it
  again). Tests drive the whole lifecycle against an in-process worker (`tests/cloud.rs`).
- **Agent labeling** (`folo_server::label`, docs/LABELING.md): labeling jobs under `<data dir>/label/<id>/`
  (unlabeled images, classes, instructions, boxes so far), labeled by a coding-agent CLI on the server's machine
  and exported as a YOLO dataset that the store registers. `Labeler` is the service behind `/api/label` (the
  store hands it out through `Store::labeler`; the mock has none). Agents sit behind the `Agent` trait:
  `ClaudeCode` runs `claude -p` and `Codex` runs `codex exec` as child processes, one model request per batch of
  images with every tool, config file and session switched off, so a call costs little more than its images; a
  run thread feeds batches to a few workers, checks each reply (JSON shape, class names, boxes clipped to the
  image) before storing it, and ends the run at once on a failure that another call would repeat. Events (`job`,
  `image`, `log`) fan out to SSE subscribers through a per-job broadcast channel, as for runs. Tests script
  the agents; `tests/label_live.rs` (ignored by default) runs the real CLIs against synthetic shapes.
- `folo` CLI (`docs/CLI.md`): `train`, `val`, `predict`, `export`, `bench`, `dataset check|stats|synth|convert`, `serve`,
  `mcp` (Model Context Protocol server over stdio so agents can train and run models), `doctor`. Every command takes
  `--json` (one JSON object per line on stdout, human text on stderr) and uses stable exit codes.

## Front ends
- **FOLO Studio** (`web/`): Svelte 5 + TypeScript + Vite, a custom design system (tokens, components) and no UI kit.
  Talks only to `docs/API.md`; `web/src/lib/api/` holds the typed client and a dev mock.
- **FOLO Desktop** (`apps/desktop/`): native Rust app on GPUI (Zed's GPU-accelerated UI framework). Same screens and
  design language as Studio. It talks to a local `folo serve` over the same API (it can spawn one), so the engine and
  the UI stay decoupled. Its Playground can also run a model in process ("Run on: Local GPU / Local CPU"): it fetches
  the f16 weights from `GET /api/models/{id}/weights` (revalidated by ETag) and runs folo-vision's `Predictor` on
  folo-gpu or the CPU. It is a separate cargo workspace so the main workspace and its CI stay GPUI-free.

## Agent-first design
- Machine-readable everything: `--json`, JSON Schemas for configs, `folo describe` prints the CLI surface as JSON.
- `folo mcp` (crate `folo-mcp`, on the official `rmcp` SDK) is an MCP server over stdio with tools for datasets,
  training, runs, validation, prediction and export, `folo://` resources and prompts (`docs/MCP.md`). It sits on
  the same `Store` trait as the HTTP handlers: the embedded `RealStore` / `MockStore`, or an HTTP-backed store that
  forwards to a running `folo serve` (`--server`). Tool schemas come from the `folo-api` types through schemars
  (the crate's `schema` feature), so they match the HTTP API.
- Errors carry a stable `code`, a human message and a hint.
- `llms.txt` at the repo root summarises the project for agents.
