# folo-gpu: GPU compute on wgpu

`crates/folo-gpu` is FOLO's GPU layer. It runs on [wgpu](https://wgpu.rs) 30: Vulkan, DX12 and Metal natively,
WebGPU in the browser (through folo-wasm, see [In the browser](#in-the-browser) and docs/WASM.md). It implements folo-core's `Backend` trait, so folo-core tensors train and infer on
the GPU (see [folo-core backend](#folo-core-backend)).

```
GpuBackend (backend/)          folo-core's Backend: Storage::Device(GpuStorage) for every dtype, views → kernels
GpuTensor (tensor.rs)          standalone shape + dtype + pooled buffer handle for kernel-level work
   │
kernels/* (kernels/mod.rs)     one Rust entry point per WGSL kernel; raw wgpu::Buffer + explicit sizes
   │
GpuContext (context/)          adapter, device, queue, buffer pool, pipeline cache, command batch
```

The kernel layer sees neither a `GpuTensor` nor a folo-core tensor: both front ends call it with raw buffers.

## Device: `GpuContext`

- **Adapter selection.** All adapters on Vulkan, Metal, DX12 and WebGPU are enumerated and ranked: discrete >
  integrated > virtual > CPU (reversed for `PowerPreference::LowPower`), then by backend, Vulkan/Metal first and
  DX12 second. On the RTX 4080 SUPER, Vulkan SGEMM is ~20% faster than DX12 with FXC: 31 vs 25 TFLOPS at 4096,
  and DX12 also lacks `shader-f16`.
  - `FOLO_GPU_ADAPTER=<index | name substring>` picks an adapter (`FOLO_GPU_ADAPTER=1`, `FOLO_GPU_ADAPTER=4080`).
  - `FOLO_GPU_BACKEND=vulkan,dx12,metal,webgpu` restricts the backends.
  - `FOLO_GPU_NO_F16=1` turns the f16 kernels off even when the adapter supports them.
  - `FOLO_GPU_TENSOR_CORES=1` (or `GpuOptions::tensor_cores`, `folo_gpu::request_tensor_cores()`, the CLI's
    `--tensor-cores`) runs the f16 GEMMs and convolutions on tensor cores; see [Tensor cores](#tensor-cores-cooperative-matrices).
- **Features and limits.** `shader-f16` is requested when the adapter has it. The device asks for the adapter's
  full limits: the largest storage buffers and bindings, and all the workgroup memory (48 KiB on NVIDIA Vulkan).
  Adapters with fewer than 4 storage buffers per stage, less than 16 KiB of workgroup memory or fewer than 256
  invocations per workgroup are rejected.
- **`folo doctor`.** `GpuContext::enumerate()` lists every adapter as an `AdapterSummary` (name, backend, type,
  driver, `shader-f16`, tensor cores, key limits, subgroup size); `ctx.summary()` describes the one in use and
  `ctx.tensor_cores_status()` says whether its tensor cores are on (or why not).
- **Shared context.** `GpuContext::shared()` is a process-wide context created once from the environment. Tests
  and the folo-core backend (`folo_gpu::init()`) use it.
- **Errors.** Validation and out-of-memory errors are captured, not panicked on. `sync()` and readbacks return
  them as `GpuError::Wgpu`. A kernel that fails to compile returns an error from the call that needed it.
  A lost device (driver reset / TDR, `destroy`) is `GpuError::DeviceLost`: wgpu panics inside `Device::poll` then, so
  FOLO polls under `catch_unwind`, remembers the loss (`lost_reason()`) and fails every later wait; folo-core sees it
  as a backend error that `folo_gpu::is_device_lost` recognises.

### Buffer pool

`ctx.alloc(bytes)` rounds the request up to a size class and takes the most recently freed buffer of that class,
or creates one. Size classes are quarter steps between powers of two, with a 256-byte floor (…, 1 MiB, 1.25 MiB,
1.5 MiB, 1.75 MiB, 2 MiB, …), so at most 25% of a buffer is slack. A dropped `GpuBuffer` goes back to its free
list. The free lists hold at most the larger of 2 GiB and the high-water mark of live bytes, so a training step
whose working set is bigger than 2 GiB still reuses every buffer on the next step (as PyTorch's caching allocator
does); when a returned buffer does not fit, the least recently freed buffers are evicted first. `trim_pool()`
empties the lists and `pool_stats()` reports hits, misses and bytes. With a fixed 2 GiB cap that dropped the
buffer being returned, batch-16 training thrashed the driver allocator (171 GB allocated in one benchmark run,
batch-1 inference after it 3.5x slower). `PoolStats::peak_live_bytes` is the largest `live_bytes` seen, PyTorch's
`max_memory_allocated` counterpart (in size-class bytes); `ctx.reset_peak_memory()` restarts it.

The high-water mark is what lets a training loop keep its working set, and it is also why a long-lived process must
give the pool back between jobs: after a run the free lists hold up to the run's peak (about 7 GB for a folo-n
fine-tune at 640 px, batch 16, report 3's X1), which other processes on the GPU then page around (a CLI run beside
an idle `folo serve` took 2.5–4× longer per epoch). `ctx.release_memory()` drops every cached buffer, lowers the
cap back to the live bytes (`trim_pool()`), submits the open batch and waits for the GPU, so wgpu frees the memory
now rather than at its next submission; it returns the bytes released. `folo serve` and `folo mcp` call it when a
training run ends (before the run is reported finished) and after 30 s without inference, and drop their cached
predictors after 5 minutes idle (docs/API.md, `gpu_pool`). Buffers in use are untouched, and a dropped buffer that
encoded work still reads is freed only once that work is done.

Reusing a buffer while earlier, unsubmitted work still reads it is safe. All work runs on one queue in encoding
order, and wgpu puts barriers between dispatches that touch the same buffer. The one exception is
`Queue::write_buffer`, which runs before the next submission's commands. That is why every upload first submits
the open batch.

### Command batching

Kernels never submit. Each call encodes a dispatch into one long-lived compute pass of an open command encoder.
The batch is submitted when:

- the batch's dispatch limit is reached: 16 after the device was idle (a wait), doubling with every submission up
  to the flush interval of 64 (`ctx.set_flush_interval(n)`, 1 to 1024, or `FOLO_GPU_FLUSH_EVERY=n` for contexts
  created afterwards), so the GPU starts on the head of a pass while the host still encodes the rest (a
  synchronised batch-1 forward otherwise spent a third of its latency encoding before the GPU had anything to do);
- an upload or readback needs ordering;
- you call `ctx.flush()` (submit without waiting) or `ctx.sync()` (submit, wait, report errors).

A forward pass is therefore a handful of submissions. Each dispatch's parameters take a 256-byte slot in a
64 KiB uniform arena, which is written once per submission. `ctx.work_stats()` counts dispatches and submissions
(the backend benchmark prints them per pass).

The interval was 1024 until the training-step profile showed the GPU idle for most of the host's encoding and
submission time: a step's 1300 dispatches went out in 3 submissions, so the GPU waited for each one to be built
whole. Submitting every 64 lets the GPU run one batch while the host encodes the next: the folo-n training step
went from 295 to 366 img/s with AMP (median of 3 alternated rounds) and from 231 / 245 to 259 / 277 img/s in f32
(the undisturbed rounds; docs/hill/gpu-train.md). 32 to 128 perform alike; 16 is slower again (every submission
costs host time) and 256 gives back half the gain.

### Profiling

`FOLO_GPU_PROFILE=1` (on an adapter with timestamp queries inside passes) makes a context write a timestamp
before and after every dispatch, copy and clear and add the differences up per label: the caller's
`profile::scope` tag (folo-gpu's backend tags each op with its kind and shape, e.g. `conv_bwd_in 16->32 k3 s2 g1
16x320x320 F16`) plus the kernel name. `ctx.profile_take()` returns and resets the totals. Resolving the queries
waits at every submission, so a profiled run is slower end to end and the per-kernel times are upper bounds (the
timestamps serialise kernels that would otherwise overlap). The folo-vision `gpu` benchmark prints a per-phase
table of a profiled training step when the variable is set.

**Dispatch trace.** `ctx.trace_begin()` / `trace_end()` record every dispatch and buffer copy encoded in between
(`TracedOp`: kernel name, spec, workgroups, and what is needed to encode it again) and `ctx.replay(&op)` re-encodes
one. `cargo bench -p folo-vision --bench gpu_profile` uses them on a fused forward: the host split of a
synchronised call (encode, submit, wait), then every recorded dispatch replayed alone, summed per kernel and per
kernel variant (`FOLO_BENCH_OPS=1` lists every dispatch). Host-side, a dispatch costs about 8–12 µs on this
machine (bind group creation about 3–4 µs; the rest lands in wgpu's submission), so batch-1 forwards are
host-bound: folo-n's 117 dispatches take about as long to encode and submit as the GPU takes to run them.

### Pipeline cache and specialisation constants

A kernel is a `KernelDef`: a name, the access mode of each storage binding, and a WGSL source (static, or
generated from the spec). A `Spec` is a list of specialisation constants. They are rendered as WGSL `const`
declarations in front of the source, together with `alias T = f32|f16` and `enable f16;` when needed. Constants
can therefore size workgroup arrays, select code paths and compile bounds checks out. Pipelines are cached by
`(kernel name, spec)`, and compilation happens on first use. Rendering constants as source text works the same
on every backend, including browser WebGPU; pipeline-overridable constants cannot size arrays on every backend.

### In the browser

On `wasm32` the same crate runs on the browser's WebGPU. Nothing may block the page's thread there, so:

- **Async device creation.** `GpuContext::new_async`, `with_options_async` and `shared_async`, and
  `folo_gpu::init_async()`, are the async forms of the blocking constructors (which `pollster` wraps natively). In
  the browser the blocking ones return `GpuError::Unsupported` with a "use the async variant" message;
  `GpuContext::shared()` / `init()` return the context `shared_async()` / `init_async()` created.
- **Async readback.** A readback is split into `begin_read` (encode the copy into a staging buffer, submit, request
  the map) and a wait. `read_with` waits by polling the device; `read_async` / `read_with_async` and
  `folo_gpu::download_async(ctx, &tensor)` (a GPU tensor → a CPU tensor, strided views made dense on the GPU first)
  await a future the map callback completes, which WebGPU runs from the event loop. Natively the async forms poll to
  completion first, so they work (and are tested) there too. In the browser a blocking readback (`to_vec`,
  `to_device(Cpu)` of a GPU tensor) fails with that error instead of hanging. The staging buffer is unmapped on
  every path, including an abandoned wait.
- **Errors.** Natively, pipeline creation sits in a validation error scope whose result is awaited with `pollster`.
  In the browser that result is asynchronous, so no scope is pushed: compile errors go to the device's
  uncaptured-error sink and are returned by the next readback.
- **`sync()`** submits but cannot wait in the browser (WebGPU's `poll` is a no-op); an awaited readback is the wait.
- **`Send + Sync`.** wgpu's WebGPU objects are `Send + Sync` only with its `fragile-send-sync-non-atomic-wasm`
  feature, which folo-gpu enables on wasm32. That is sound because the wasm build has no threads.
- **WGSL.** The kernels are unchanged. Chrome's Tint is stricter than Naga in places: it rejects float literals
  beyond `f32::MAX` that Naga rounds (`-3.4028235e38` in `softmax` is now `bitcast<f32>(0xff7fffffu)`). Keep new
  kernels to literals inside the `f32` range.
- `GpuContext::enumerate` is empty in the browser (WebGPU hands out one adapter on request), and the adapter name
  is usually hidden (empty).

## Buffer layout conventions

- Every operand is a **dense row-major array starting at byte 0** of its buffer. Buffers can be larger than the
  operand (size classes); kernels never read or write the slack. Views (offsets, strides) are not part of
  `GpuTensor`. The GEMM layer takes explicit leading dimensions, offsets and batch strides; everything else
  works on dense data.
- Images are **NCHW**. Conv weights are **`[C_out, C_in / groups, KH, KW]`**, which is the row-major `A` matrix
  of each group's GEMM. Biases are `[C_out]`.
- **f16** is IEEE binary16, little-endian, two per 32-bit word. Byte sizes are rounded up to 4 for wgpu's copy
  alignment.
- **Binding layout** is the same for every kernel. `@binding(0)` is the 256-byte uniform parameter block, read in
  WGSL with `pu(i)` / `pi(i)` / `pf(i)` (u32 / i32 / f32 word `i`). `@binding(1..)` are the storage buffers in
  `KernelDef::storage` order. An absent optional input (no bias) is bound to a small dummy buffer.
- **1-D launches** over more than 65535 workgroups are folded into x/y (`GpuContext::grid_1d`). Shaders recover
  the linear id with `linear_id(...)` from the prelude.

## Kernels

| Kernel | Entry point | dtypes | Notes |
|---|---|---|---|
| unary | `kernels::elementwise::unary` | f32 (the backend's ops go through the strided kernel below, f16 too) | exp, log, sqrt, rsqrt, neg, abs, sigmoid, silu, relu, relu6, tanh, gelu (tanh), leaky_relu, hardswish, hardsigmoid, square, recip, sin, cos, floor, ceil; vec4 path when `n % 4 == 0` |
| binary | `kernels::elementwise::binary` | f32 | add, sub, mul, div, max, min, pow (C `powf` semantics). NumPy broadcasting; dims are coalesced first (`[N,C,H,W] + [1,C,1,1]` becomes rank 3); same-shape vec4 fast path; rank ≤ 8 after coalescing |
| bias + activation | `kernels::elementwise::bias_act` | f32 | `act(x + bias[(i / inner) % C])`, or the activation alone |
| reduce | `kernels::reduce::reduce`, `reduce_as` | f32, f16 (f32 accumulation) | sum, mean, max, min over the middle axis of `[outer, R, inner]`, i.e. any dim. `inner == 1`: one workgroup per row with a tree in workgroup memory; otherwise each thread owns a column and loads are coalesced |
| softmax / log_softmax | `kernels::reduce::softmax`, `softmax_rows` | f32, f16 (f32 max, sum and exponentials) | last dim, one workgroup per row, max-shifted |
| GEMM | `kernels::gemm::gemm` | f32, f16 | `alpha·op(A)·op(B) (+ row/col bias) → act`; A, Aᵀ, B, Bᵀ; two-level batch with arbitrary offsets and strides (stride 0 broadcasts); f16 storage with f32 accumulation |
| conv2d | `kernels::conv::conv2d` | f32, f16 | NCHW; stride, padding, dilation, groups; fused bias + activation. Implicit GEMM, a pointwise GEMM fast path for 1×1, and a direct depthwise kernel (4 adjacent outputs of a row per invocation, the window unrolled by the generator, so the input columns they share load once) |
| tensor-core GEMM / conv2d | `kernels::coop::{gemm, conv2d}` | f16 in; f16 or f32 out | cooperative matrices (16×16×16, f32 accumulation), the same GEMM / implicit-GEMM conv contract and fused `alpha` + bias + activation; `gemm::gemm` and `conv::conv2d` route f16 problems here when the context has [tensor cores](#tensor-cores-cooperative-matrices) on, and so do the f16 conv gradients (`CONV_MODE` 2 / 3, split-K, below) |
| max_pool2d | `kernels::pooling::max_pool2d` | f32, f16 | kernel / stride / padding / dilation; padding never wins (−∞); floor mode |
| upsample_nearest2d | `kernels::layout::upsample_nearest2d` | f32, f16 | any output size; `src = dst · in / out`, as in PyTorch `nearest` |
| concat | `kernels::layout::copy_rows` | f32, f16 | any dim, via row-blocked copies; a single block becomes a buffer copy; f16 blocks need even lengths and offsets |
| permute / transpose | `kernels::layout::permute` | f32, f16 | dims are coalesced; batched 2-D transposes (NCHW ↔ NHWC) go through a padded 32×32 tile, everything else through a generic strided copy |
| cast | `kernels::layout::cast` | f32 ↔ f16 | packed words; needs no `shader-f16` |
| strided gather | `kernels::strided::gather_strided` | any (1/2/4/8-byte) | make-contiguous of any view (offset, strides, stride-0 broadcast); dims coalesced; contiguous views become a buffer copy; sub-word views whose innermost run is word-aligned (even f16 runs, u8 runs of four, every offset and outer stride too) are copied as `u32` views; other sub-word elements assembled one output word per thread |
| strided scatter | `kernels::strided::scatter_strided` | any | `copy_into` (cat, pad, view gradients, `copy_from`); word-aligned sub-word views (as for the gather, e.g. concatenating f16 feature maps with even `H·W`) are written as `u32` words; other sub-word elements with `atomicAnd` / `atomicOr` on their own bits |
| fill / cast | `kernels::strided::{fill_words, cast}` | any | `full` with a two-word pattern (i64) or `clear_buffer` for zeros; casts between all five dtypes with Rust `as` semantics (truncate, saturate, NaN → 0) |
| index_select / gather | `kernels::strided::{index_select, gather}` | any | `i64` or `u32` indices; out-of-range indices give zeros |
| index_add / scatter_add | `kernels::strided::{index_add, scatter_add}` | f32 | compare-and-swap float atomics; out-of-range indices skipped |
| strided elementwise | `kernels::ew::elementwise` | f32, f16 (f32 arithmetic; a `where` condition stays f32 / u8) | folo-core's unary ops and their gradients, binary ops and comparisons, `where` (f32 or u8 mask); every operand a strided view read in place, linear + vec4 path when all are contiguous |
| arg-reduce | `kernels::reduce::arg_reduce` | f32 → i64 | first max / min, NaN wins (the backend runs f16 inputs on an f32 copy) |
| split reduce | `kernels::reduce::reduce_split` | f32, f16 | long reductions over few outputs (BatchNorm statistics) in two passes; the partials between them are f32 |
| softmax (any dim) | `kernels::reduce::softmax_dim` | f32, f16 | rows kernel for the last dim, a column kernel otherwise |
| top-k | `kernels::reduce::topk` | f32 → (f32, i64) | sorted, ties by index, `total_cmp` order; one thread per row |
| pooling + gradients | `kernels::pool_grad::*` | f32 (max pooling with indices, its backward and the upsample backward also f16, summed in f32) | max pooling with indices and its gather backward, avg / adaptive avg pooling and backward, upsample backward |
| conv2d gradients | `kernels::conv::{conv2d_backward_input, conv2d_backward_weight, conv2d_backward_weight_as}` | f32, f16 (f32 accumulation; the weight gradient also f16 → f32) | implicit GEMM (`CONV_MODE` 2 / 3) with split-K for the weight gradient, a transposed-A GEMM for 1×1 input gradients, direct depthwise kernels. f16 runs on tensor cores when they are on (1×1 input gradients as a GEMM over the once-transposed weight), else on the regular generator with f16 storage; split-K partials are f32 either way |
| train-mode BatchNorm + activation | `kernels::norm::{batch_norm_forward, batch_norm_backward}` | f32 or f16 data (`BnDesc::dtype`); parameters, statistics and sums f32 | NCHW, statistics over `N·H·W`. Forward: per-(channel, chunk) Welford partials merged with Chan's formula, a per-channel finalize (mean, `invstd`, the running-statistics update in place), then `act(x · scale + shift)`; backward: partial sums of `dz` and `dz · x̂` with the activation's derivative at the recomputed pre-activation, a finalize (`dγ`, `dβ`), then `dx`. Identity, SiLU or ReLU; 3 dispatches each way |
| optimizer updates | `kernels::optim::optimizer_step`, `optimizer_step_many` | f32 | in place, one thread per element: SGD (momentum, Nesterov, coupled decay), AdamW (decoupled decay, bias corrections), and `p = a·p + b·g` for EMAs and gradient accumulation, in the op order of folo-nn's CPU loops. `_many` updates up to 16 tensors with one update per dispatch: each workgroup finds its tensor in a table of first workgroups (uniforms 16..31, element counts at 32..47) and runs that tensor's copy of the update (a generated `switch` over per-tensor bindings, so the per-element arithmetic is the single kernel's, bit for bit); devices that bind fewer than `16 · (2 + states)` storage buffers fall back to one dispatch per tensor |
| clipping reductions + scale | `kernels::optim::{norm_partials_many, scale_many}` | f32 | global-norm clipping over many gradient buffers, 16 per dispatch: one partial per 256 elements (a fixed tree in workgroup memory) of the largest `|x|` (NaN propagating) or of `((x · h) · h)²`, reduced by the caller; and `x = (x · f1) · f2` with device factors |
| unscale + overflow check | `kernels::optim::unscale_check`, `unscale_check_many` | f32 | mixed precision's loss-scaler step: `x *= 1 / S` in place and an `atomicOr` into a shared `u32` flag when a result is inf or NaN (exponent bits all set); `_many` takes 16 gradient buffers per dispatch as above; one flag readback per optimizer step |
| task-aligned assigner | `kernels::detect::assign` | f32 / u32 | docs/MODEL.md §7.3 on the raw head outputs, five dispatches: DFL decode of every anchor; one workgroup per gt that scans its candidates, scores `u = max(CIoU, 0)` and `m = score^α · u^β` and keeps its top k by `(m desc, anchor asc)` (per-thread sorted lists merged by a k-round tournament); two atomic passes for the conflicts (largest `u`, then the smallest gt index); the soft targets `t = m · u_max / (m_max + eps)` per gt and a dense per-anchor `(label, t)` map. Same decisions as `folo_vision::assign` |
| detection loss | `kernels::detect::{bce_forward, bce_backward, box_terms, combine}` | f32 | §7.4: BCE over every logit against the dense targets, each element formed whole; CIoU + DFL per positive (softmax expectation, both terms and their hand-written backward in one thread); a one-workgroup fixed-order reduction to the reported items, the total and the backward scales. `exp`, `ln(1 + x)` and `atan` are written out in plain arithmetic (folo-core's `fast_exp`, an atanh series, Cephes `atanf`), since WGSL only promises an absolute error of 2^-21 for `log` near 1 |
| letterbox | `kernels::vision::letterbox_rgb8` | u8 → f32, f16 (the f32 value rounded once) | one thread per canvas pixel: the padding grey, or the resized image sampled with folo-data's `ResizePlan` (integer box prefilter computed on the fly, then 11-bit fixed-point bilinear taps uploaded by the host), in integers, times `scale`: bit-exact with the CPU letterbox. Prefilter blocks whose sum could overflow u32 are refused (the caller letterboxes on the host) |
| detection candidates | `kernels::vision::detect_candidates` | f32, f16 → u32 (decoded in f32) | one thread per (image, anchor): the first maximum of the `nc` sigmoid scores; above `conf`, the DFL softmax expectation of each side, the box around the anchor times the stride, and a 7-word record `(anchor, class, score, x1, y1, x2, y2)` appended with an atomic counter per image (the output is cleared first) |

The f16 kernels need `shader-f16`; without it they return `GpuError::Unsupported`. Everything not listed as f16
is f32-only. f16 is storage: every f16 kernel widens to f32 on load, computes and accumulates in f32, and rounds once
on the store (the GEMM and conv tiles are f32 in workgroup memory).

### GEMM design

`kernels/gemm.rs` holds a single WGSL generator that covers every variant.

- A workgroup of 256 threads computes a `BM × BN` tile of C. Each thread owns a `TM × TN` block of `vec4<f32>`
  accumulators held in individual registers, arranged as interleaved vec4 groups so that workgroup reads have no
  bank conflicts.
- K is walked in steps of `BK`. The next A and B tiles are prefetched from global memory into registers while
  the current ones are consumed from double-buffered workgroup memory, with one barrier per step. The main loop
  handles two K tiles per iteration, so every buffer index into workgroup memory is a constant.
- Tiles are stored k-major (`As[k][m/4]`, `Bs[k][n/4]`), so the inner product reads vec4s of both operands. Global
  loads are vec4 whenever the leading dimension, offsets and strides are multiples of 4; otherwise they are
  scalar. Bounds checks are compiled out when M, N and K are tile multiples.
- Transposes only change how a tile is fetched. Aᵀ and B load vec4s along the contiguous direction; A and Bᵀ
  load along k and scatter four scalars down the k-major tile.
- Tile configs: `LARGE` 128×128×8 with 8×8 per thread, `MEDIUM` 64×128×16 with 4×8, `SMALL` 64×64×16 with 4×4.
  `pick_config` takes the largest config that still yields at least 64 workgroups and wastes no more than 25% of
  its rows. The forward convs pick through `pick_config_conv`, which first tries two narrow configs for few output
  channels: `ROWS32` 32×128×32 (4×4, 256 threads) for `M <= 32` and `ROWS16` 16×128×32 (4×4, 128 threads) for
  `M <= 16`, where a 64-row tile leaves half or three quarters of its rows empty (folo-n's 16- and 32-channel early
  layers at 320 and 160 px: the stem 0.42 → 0.30 ms, 16 → 32 at stride 2 0.44 → 0.29 ms, a 16-channel 3×3 0.38 →
  0.19 ms at batch 16).
- `CONV_MODE = 1` turns the same generator into implicit-GEMM conv. The B tile is gathered straight from the NCHW
  image. Each thread keeps one output pixel for the whole K loop, so the per-element index math is a few
  divisions by constants. No im2col buffer is ever written.
- **Forward split-K** (`kernels::conv::split`, both generators). folo's 3×3 convs at 40 and 20 px have few output
  tiles but a long `K` (64–256 workgroups at batch 16 for 80 SMs, `K` up to 4608), and at batch 1 every layer
  below 160 px has only a handful. When a forward implicit-GEMM conv has fewer than 320 workgroups and at least 18
  `BK` tiles of `K`, `K` is split into up to 8 chunks, aiming at about 480 workgroups. The count is capped at one
  chunk per 9 tiles of `K` (`splits <= k_tiles / 9`); that caps the count, not every chunk's length: the chunks
  are one length, a multiple of `BK`, and the last one ends short (`K = 441` at `BK = 16`: 28 tiles, 3 chunks of
  160, the last 121 long). Every chunk is a workgroup layer (`SPLIT = true`) that writes an f32 partial
  slab of the whole output without bias or activation, and `conv_split_finish` sums the slabs (unrolled, `vec4`
  when `OH·OW` is a multiple of 4) and applies the bias and activation, rounding to f16 for an f16 output. The
  slabs come from the buffer pool (at most 64 MiB). `ConvPlan::splits` forces a chunk count (tests, tuning);
  `cargo bench -p folo-gpu --bench conv_model` times folo-n / folo-s's weakest forward convs, and with
  `FOLO_CONV_TUNE=1` every tile config at 1–8 chunks. 1×1 stride-1 convs (the pointwise GEMM) and depthwise
  convs are not split; a strided 1×1 conv is implicit GEMM and is.
- `CONV_MODE = 2` is the input gradient as a transposed conv: each thread owns one input pixel and, for
  `k = (c_out, kh, kw)`, reads the output-gradient pixel `((ih + PH - kh·DH) / SH, (iw + PW - kw·DW) / SW)` when
  the divisions are exact and in range (a strided conv leaves the rest zero). `A` is the weight permuted once to
  `[groups, C_in/g, C_out/g, KH·KW]`.
- **Stride phases** (`CONV_MODE = 2` with `PHASE`): for a strided conv with dilation 1 (not depthwise), only the
  taps `kh ≡ (ih + PH) mod SH` (and likewise for `kw`) reach input row `ih`, so the plain transposed conv
  multiplies by zero for 3/4 of its `K` at 3×3 stride 2. The input pixels split into `SH·SW` phases by
  `((ih + PH) mod SH, (iw + PW) mod SW)`; each phase is a dense implicit GEMM with `K = C_out/g · th · tw` (1, 2, 2
  or 4 taps of the 9 at 3×3 s2, constants `PTH` / `PTW`), its `B` gathered at output pixel `(bh + a − t, bw + b −
  u)` for phase pixel `(a, b)` and tap `(t, u)` with no divisibility test, and its epilogue writing column `j` to
  plane offset `(IH0 + SH·(j / WR))·W + IW0 + SW·(j mod WR)` (`ppix`). `conv2d_phase_pack` packs the weight per
  phase once (`[phase][g][C_in/g][C_out/g][th·tw]`, the `A` of each phase), replacing the permute. One dispatch
  per phase with taps: only the `min(SH, KH) · min(SW, KW)` such phases are packed and run, so scratch and host
  work do not grow with the stride; when the kernel is smaller than the stride, the input gradient is cleared
  first and the other phases' pixels stay zero. Both the regular and the tensor-core generator have it.
- `CONV_MODE = 3` is the weight gradient: `K = N·OH·OW` folds the batch into the reduction, `A` rows are output-
  gradient planes read across images, and `B` is the image gathered as a transposed im2col (each thread owns one
  `(c_in, kh, kw)` column). `SPLIT = true` splits `K` into chunks, one per extra workgroup layer, each writing a
  partial weight slab that a column reduction sums: early layers have only a few output tiles but hundreds of
  thousands of `K` steps. The spatial sizes (`IH`, `IW`, `OH`, `OW`) are specialisation constants here (in both
  generators), because the gather divides every `k = (n, oh, ow)` by `OH·OW` and `OW`: a constant divisor is a
  multiply and a shift, a runtime one a dozen instructions. That made the tensor-core weight gradients of a folo-n
  step about 5 ms faster (the AMP step 366 → 449 img/s); the f32 kernels are bound elsewhere and did not change.
  A training run compiles one weight-gradient pipeline per layer shape (a few dozen).

#### What was tried (RTX 4080 SUPER, Vulkan, square SGEMM, GFLOPS)

| Step | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| v1: 128×128×16 tiles, 8×8 per thread, register prefetch, double buffering, `for k` loop | 7 084 | 12 703 | 18 893 | 23 239 |
| + unroll the k loop in the generator | 10 120 | 16 003 | 24 582 | 29 568 |
| + 2-D workgroup tiles, two K tiles per iteration (constant buffer indices), retuned configs (final) | 12 577 | 23 651 | 28 037 | 31 308 |

Notes from the tuning runs:

- **Naga's loop guard was the biggest cost.** wgpu guards every WGSL loop with an iteration counter (infinite-loop
  protection, `force_loop_bounding`), and that stops the driver from unrolling the inner k loop. Turning the guard
  off through `create_shader_module_trusted`, which is `unsafe`, was worth 20–70%. Unrolling the loop in the
  generator gets all of that and more while staying safe, so no shader is created unchecked.
- **Index clamping on workgroup arrays.** With robust buffer access on Vulkan, wgpu leaves storage-buffer accesses
  unchecked, but every dynamic index into a workgroup array is still clamped. Turning those checks off too was
  worth a further 3–20%. Most of that was recovered safely by making the buffer and k indices constants and the
  remaining column index loop-invariant, so the clamps fold away or are hoisted out of the loop.
- **Tile shapes.** 128×256 and 256×128 tiles with 8×16 per thread were best before unrolling (24.3 TFLOPS at 4096).
  After unrolling they fell behind 128×128, presumably because of register pressure. BK = 8 beats 16 for
  128×128; BK = 32 never helped for 64×64.
- **Padding the workgroup-tile rows** by one vec4 removes a 2-way bank conflict in the A scatter, but it was
  slightly slower overall and was dropped.
- **DX12 (FXC)** reaches 9.7 / 15.4 / 20.9 / 25.2 TFLOPS on the same kernels.
- **Not tried yet:** subgroup (warp) shuffles, split-K for skinny plain GEMMs (the forward convs have it, above),
  and a persistent-kernel / stream-K schedule. Cooperative matrices (tensor cores) are the next section; they are native-only (not in stable WebGPU).

### Tensor cores (cooperative matrices)

`kernels/coop/` runs f16 GEMMs and convolutions on NVIDIA tensor cores through wgpu 30's experimental cooperative
matrices (`EXPERIMENTAL_COOPERATIVE_MATRIX`; WGSL `coop_mat16x16<f16, A>`, `coopLoadT`, `coopMultiplyAdd`,
`coopStoreT`; `SPV_KHR_cooperative_matrix` on Vulkan). f16 inputs, f32 accumulation, f16 or f32 output, and the
regular kernels' fused epilogue (`alpha`, row or column bias, any activation).

- **Opt-in, then capability.** A context uses them only when asked (`FOLO_GPU_TENSOR_CORES=1`,
  `GpuOptions::tensor_cores`, `folo_gpu::request_tensor_cores()` before the shared context exists, or
  `--tensor-cores` on `folo predict` / `val` / `serve`, `folo train --amp`) **and** the adapter reports a 16×16×16 f16 × f16 → f32
  configuration (`adapter.cooperative_matrix_properties()`), `shader-f16`, subgroups and 32-wide subgroups. The
  RTX 4080 SUPER on Vulkan has them (driver 621.01); DX12, Metal and the browser do not in wgpu 30.
- **Self-test, then routing.** The device is created with the experimental feature, then a probe fragment and a
  ragged GEMM (70×90×75, row bias + SiLU, f16 and f32 output) run against the CPU before `caps().coop_f16` is set.
  After that, `gemm::gemm` and `conv::conv2d` send every f16 problem to the tensor-core kernel (1×1 stride-1 convs
  as its GEMM, other non-depthwise convs as its implicit-GEMM conv), and `conv::conv2d_backward_input` /
  `conv2d_backward_weight_as` send f16 gradients to its `CONV_MODE` 2 / 3 kernels (mixed-precision training,
  below). Depthwise convs and f32 keep the regular kernels; so do calls with an explicit `TileConfig`.
- **Gradients (mixed-precision training).** `CONV_MODE` 2 is the input gradient as a transposed conv (each
  invocation owns an input pixel; the `B` tile gathers the output-gradient pixels that read it, `A` is the weight
  permuted once to `[groups, C_in/g, C_out/g, KH·KW]`; 1×1 convs are the plain tensor-core GEMM over that permuted
  weight). `CONV_MODE` 3 is the weight gradient with `K = N·OH·OW` (the `A` rows are output-gradient planes read
  across images, the `B` tile the image gathered as a transposed im2col), tiles picked by `M = C_out/groups` alone
  and `SPLIT` cutting `K` into chunks (one z-layer each; each chunk writes an f32 partial slab, a column reduction
  sums them) until about 512 workgroups run. The loop bounds derive from `workgroup_id` and uniforms, so they stay
  uniform. The weight gradient is written as f32 (straight to the f32 master weight, no f16 rounding or
  overflow) or f16. `src/tests/coop.rs` checks both against an f64 CPU reference with the bound below, on every conv
  shape folo-n / s use plus dilation, rectangular kernels, groups, depthwise and a split-K case, with tensor cores
  and with the regular f16 kernels.
- **Fallback.** Anything unexpected leaves the regular kernels in place: an adapter without the configuration, a
  device that refuses the feature (it is created again without it), a failed self-test (the experimental device is
  dropped and the context rebuilt on a plain one), or a tensor-core kernel
  that fails to build (tensor cores turn off for the context and the call runs on the regular kernel).
  `ctx.tensor_cores_status()` says which; `ctx.set_tensor_cores(bool)` switches them at run time (A/B tests and
  benchmarks).
- **The one `unsafe`.** `ExperimentalFeatures::enabled()` is called in exactly one place,
  `coop::experimental_opt_in`, whose doc comment states the safety conditions: the opt-in and capability checks
  above; cooperative loads and stores touch only workgroup arrays of compile-time size, at offsets the generator
  proves in bounds (asserted for every fragment; the subgroup index is clamped to 0..8), never a storage buffer, so
  wgpu's missing bounds checks on cooperative accesses cannot reach outside a binding (global memory goes through
  ordinary, bounds-checked loads and stores); every cooperative op runs in workgroup-uniform control flow (naga
  checks this and rejects the shader otherwise); the self-test passes first; and no other code gets the
  experimental device safely: `ctx.device()` is `None` on such a context (the raw device is behind the `unsafe`
  `device_unchecked`, whose contract is the two conditions before), and `ctx.destroy_device()` covers what tests
  used the raw device for.
- **Kernel.** 256 invocations = 8 subgroups per workgroup, each owning `FM × FN` 16×16 f32 accumulators of a
  `BM × BN` tile (`subgroup_id` picks the subgroup's fragments, so they are uniform per subgroup). `K` is walked
  32 at a time with the f32 generator's scheme: the next `A` / `B` tiles are loaded into registers (vec4 when
  aligned, the conv's `B` gathered straight from the NCHW image) while the current ones, double-buffered f16 in
  workgroup memory, row-major with 8 halves of padding per row, feed `coopLoadT` + `coopMultiplyAdd`; one barrier
  per step. The K loop starts each half with a `break` check instead of wrapping it in an `if`, because naga
  treats values derived from the loop counter as non-uniform. Epilogue: cooperative matrices have no per-element
  access, so each subgroup stores one fragment at a time into its own 1 KiB f32 scratch slice, and its 32 lanes
  apply `alpha`, bias and activation to 8 elements each and store them with bounds checks. Configs: 128×128 (4×2
  fragments per subgroup), 64×128, 64×64, 32×128 for `M ≤ 32` and 16×256 for `M ≤ 16` (folo-n's 16- and 32-channel
  early layers); `coop::pick_config` picks by `M` and occupancy. 45 KiB of workgroup memory at 128×128.
- **Accuracy.** Products of f16 values are exact in f32, so the only difference from the regular f16 kernels is
  the f32 summation order. `src/tests/coop.rs` checks the kernels against the f64 CPU on the same f16 inputs with a
  per-element bound of `K · 2^-23 · Σ|a·b|` (one ulp per addition, which also covers tensor cores that truncate
  while aligning addends) plus the activation's few ulps and, for f16 output, one rounding (`2^-11 · |y|`), with 2x
  slack: every config and transpose on ragged and aligned shapes, odd shapes from 1×1×1 up, a `K = 4608`
  reduction, every bias kind with SiLU / ReLU / sigmoid / identity and `alpha`, two-level batches with strides,
  offsets and padded leading dimensions, attention-shaped batched products, and every conv variant folo-n / s use
  (3×3 s1 / s2, the 3-channel stem, 1×1, ragged channel counts, groups, dilation, rectangular kernels) in f16 and
  f32 output, through the routing and explicitly. `crates/folo-vision/tests/gpu_tensor_cores.rs` checks the whole
  detector (decoded scores within 1.1e-5 of f32 on a random folo-n) and the end-to-end predictor; trained-model
  mAP is within 0.002 of f32 (docs/BENCHMARKS.md).

## Benchmarks

`cargo bench -p folo-gpu --bench gemm | conv | elementwise`. Times are wall-clock, best of three runs. Submissions
go out in groups of 8 so the GPU runs while the CPU encodes. RTX 4080 SUPER, Vulkan, driver 621.01, release
build:

| GEMM | GFLOPS |
|---|---|
| SGEMM 512³ | 12 577 |
| SGEMM 1024³ | 23 651 |
| SGEMM 2048³ | 28 037 |
| SGEMM 4096³ | 31 308 |
| 2048³ Aᵀ·B / A·Bᵀ / Aᵀ·Bᵀ | 28 098 / 27 150 / 27 017 |
| 1000³ (not a multiple of 4) | 18 095 |
| batched 64 × 256³ | 24 817 |
| HGEMM (f16 storage, f32 accumulation) 1024 / 2048 / 4096 | 20 603 / 26 582 / 31 997 |
| tensor-core HGEMM, under 60–70% load from another job (regular HGEMM in the same run: 17 269 / 15 239 / 18 072) 1024 / 2048 / 4096 | **22 565 / 30 298 / 37 091** |
| tensor-core 2048 Aᵀ·B / A·Bᵀ / Aᵀ·Bᵀ (same run) | 19 535 / 25 821 / 18 436 |

| conv2d, N = 8, 80×80, 64 → 128 | ms | GFLOPS |
|---|---|---|
| 3×3 s1 p1 | 0.291 | 25 919 |
| 3×3 s2 p1 | 0.111 | 16 965 |
| 1×1 s1 | 0.041 | 20 579 |
| depthwise 3×3 s1, 128 ch | 0.069 | 1 707 (memory-bound) |
| f16: 3×3 s1 / 3×3 s2 / 1×1 / depthwise | 0.270 / 0.110 / 0.039 / 0.113 | 27 935 / 17 107 / 21 272 / 1 048 |
| depthwise with 4 outputs per invocation (hill/gpu-infer; old kernel in the same alternated runs): f32 / f16 | 0.063–0.068 / **0.044–0.051** (0.070–0.080 / 0.127–0.130) | 1 865 / 2 700 at best |
| tensor cores, under ~65% load (regular f16 in the same run: 0.584 / 0.177 / 0.101 ms): 3×3 s1 / 3×3 s2 / 1×1 | **0.288 / 0.118 / 0.044** | 26 191 / 15 937 / 19 239 |
| tensor cores, folo-n's stem at batch 16 (same run, regular f16 0.831 / 0.780 / 0.309 ms): 3→16 at 640, 16→32 at 320, 32→64 at 160, all 3×3 s2 | **0.245 / 0.250 / 0.097** | 5 783 / 15 104 / 39 097 |

| Memory-bound, 64 Mi f32 | GB/s |
|---|---|
| buffer copy (reference) | 598 |
| unary relu / silu / exp | 577–579 |
| binary add (same shape) / mul by a `[1,C,1,1]` broadcast | 596 / 577 |
| bias + SiLU | 578 |
| sum over last dim / middle dim | 483 / 590 |
| softmax rows of 1024 | 446 |
| permute NCHW → NHWC / generic | 593 / 550 |
| concat, upsample 2× | 590, 576 |

## Adding a kernel

1. **Write the WGSL** in `src/shaders/<name>.wgsl`. Read parameters with `pu(i)` / `pf(i)` (words of the uniform
   block at `@binding(0)`) and bind storage buffers at `@binding(1..)`. Use `T` for the element type if the
   kernel should also have an f16 variant. Read specialisation constants as plain identifiers; they are declared
   for you. For a launch over many elements use `@workgroup_size(256)` and `linear_id(wid, nwg, lid, 256u)`.
2. **Declare it** in the matching `src/kernels/*.rs` as a `static KernelDef { name, storage, source }`.
   `storage` lists `Access::Read` / `Access::ReadWrite` per binding. `source` is `Source::Static(include_str!(..))`,
   or `Source::Generated(fn)` when the code depends on the spec, like GEMM.
3. **Add the Rust entry point.** Validate sizes with `check_buf`, check dtype support with `check_dtype`, return
   early on empty problems, then:
   ```rust
   let k = ctx.kernel(&MY_KERNEL, Spec::new().u32("OP", code).f16(dtype == DType::F16))?;
   let p = Params::new().uz(n)?.f(alpha);
   ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
   ```
4. **Expose it on `GpuTensor`** if it is a tensor-level op: validate shapes, allocate the output with
   `GpuTensor::empty`, and call the entry point.
5. **Test it** in `src/tests/<area>.rs` against a straightforward CPU loop. Use seeded random data
   (`tests::Rng`), include sizes that are not multiples of the tile or vector width, and state the tolerance. Start
   each test with `let ctx = gpu!();` so it skips when there is no adapter. Avoid WGSL loops in hot paths when the
   trip count is a constant: unroll them in the generator (see the loop-guard note above).
6. **If it is on a hot path**, add a case to a bench and put the numbers in the PR.

## Testing

`cargo test -p folo-gpu` runs every kernel against a CPU reference on seeded random data. That includes odd
sizes, ragged tiles, unaligned offsets, every transpose and batch layout, f16 variants and error paths. It takes
about 5 s on the RTX 4080 SUPER, most of it compiling pipelines.

- With no adapter (CI), each test prints `note: skipping GPU test, no adapter` and passes.
- Software adapters (WARP on Windows CI images, llvmpipe) are skipped too, because compiling shaders there is
  slow. Set `FOLO_GPU_TEST_SOFTWARE=1` to run on them: the suite passes on WARP in about 30 s.
- To cross-check another backend locally, run with `FOLO_GPU_BACKEND=dx12` (passes) or
  `FOLO_GPU_ADAPTER=<name>`.

## folo-core backend

`folo_gpu::init()` registers `GpuBackend` over `GpuContext::shared()` as `Device::Gpu(0)`; custom setups build
`GpuBackend::new(ctx, device)` and call `folo_core::backend::register_backend` themselves.

```rust
let dev = folo_gpu::init()?;                          // Device::Gpu(0)
let x = Tensor::randn([16, 3, 640, 640], 0.0, 1.0, &mut rng).to_device(dev)?;
let w = w_cpu.to_device(dev)?.detach().requires_grad_();
let loss = x.conv2d(&w, None, 1, 1, 1, 1)?.silu()?.square()?.mean_all()?;
let grads = loss.backward()?;                         // every gradient op runs on the GPU too
println!("{}", loss.item::<f32>()?);                  // the only wait for the GPU
```

- **Storage.** `Storage::Device(DeviceStorage)` holds a `GpuStorage` (one pooled `GpuBuffer`) for every folo-core
  dtype: `f32` / `u32` one word per element, `f16` two and `u8` four per word (little-endian), `i64` two words.
  Elements start at byte 0; views add offsets and strides.
- **Views.** Elementwise ops (`unary`, `unary_backward`, `binary`, `where_cond`), `copy_strided`, `copy_into`,
  casts and `matmul` read strided and broadcast layouts in place: the shape is coalesced across all operands and a
  kernel walks up to 8 dims (a linear, vec4 path when everything is contiguous). `matmul` maps each operand to the
  GEMM's leading dimension / transpose / two-level batch strides (transposed views cost nothing; a shared right
  operand folds the batch into `M`) and copies an operand only when no such mapping exists. Non-depthwise convs
  read any input whose images are contiguous (a channel slice, `narrow` along dim 1) in place through the GEMM's
  `B` offset and batch stride. Every other kernel takes dense data, so a strided input is copied once first.
- **Concatenations read in place.** Every `forward_cat` of the detector (C3k2, C3k, SPPF, C2PSA and the neck's
  Concat + C3k2) is a 1×1 conv over a channel concatenation. `conv2d_act_cat` runs it as one GEMM whose `B` is
  the parts' rows read from their own buffers (`kernels::gemm_cat`, `CAT` in the GEMM and tensor-core
  generators): part 0 binds where `B` does, parts 1–3 at `@binding(5..7)`, and uniform words 40–51 hold each
  part's row end, offset and image stride; the generated `cat_b(k, i, b1)` picks the part by row. Up to four parts
  (`MAX_CAT_PARTS`), each a dense tensor or a channel slice with contiguous images; parts without channels are left
  out (one part left is the plain conv on it). Anything else (3×3 or grouped convs, more parts, strided images, a
  device with fewer than 7 storage buffers per stage) is `NotSupported`, and folo-core concatenates first. The
  output must not share a buffer with a part, the weight or the bias (the backend always allocates a fresh one).
  Inference only: a recorded (training) forward keeps `cat` then the conv.
- **Op mapping.** Reductions: sum / max / min through the reduce kernels (split into two passes when long rows
  meet few outputs), arg-reductions to `i64`. Softmax over any dim. conv2d forward is the existing kernel; its
  gradients are the `CONV_MODE` 2 / 3 GEMMs or the depthwise kernels. Pooling, upsampling, index ops and top-k use
  the kernels in the table above.
- **Fused ops** (folo-core's optional `Backend` hooks, each with a composite fallback on other backends):
  `conv2d_act` is the conv kernel with bias and activation in its epilogue (`Tensor::conv2d_act`, used by folo-nn's
  `ConvBnAct` in eval / fused mode, so a fused folo-n forward is one dispatch per conv), and reads a channel
  slice (`narrow` along dim 1, any view whose images are contiguous) in place through the GEMM's `B` offset and
  batch stride instead of gathering it first (depthwise convs still take dense inputs); `conv2d_act_cat` (see
  Concatenations read in place, above) is the 1×1 conv over a concatenation without writing it;
  `batch_norm_act` / `batch_norm_act_backward` are the BatchNorm kernels above (`Tensor::batch_norm_act_train`,
  recorded by folo-nn's `BatchNorm2d` as one custom op); `optimizer_step` is the update kernel
  (`Tensor::optimizer_step_`) and `optimizer_step_many` / `unscale_check_many` its multi-tensor form
  (`Tensor::optimizer_step_many_`, `Tensor::unscale_check_many_`), which folo-nn's SGD, AdamW, EMA, gradient
  accumulation and loss scaler use on the GPU: a folo-n step's 255 parameters take 16 dispatches per pass instead
  of 255 (the AMP step's copy + unscale + SGD phase was 766 dispatches).
- **Half precision** (`Backend::computes_f16`, true with `shader-f16` unless `FOLO_GPU_NO_F16` is set): the forward
  ops folo-core lets `F16` through (elementwise and `where`, reductions, softmax, matmul, conv2d and the fused
  conv, max / average / adaptive pooling, upsampling, the letterbox output and `detect_candidates`) take f16
  operands and return f16 results, f32 inside as above; average pooling and the arg-reductions run on an f32 copy.
  `Tensor::to_dtype_` converts a storage in place (the dtype lives in the storage cell, like the device), so
  folo-nn's `Module::to_dtype(F16)` turns a fused model into an f16 one; folo-vision's `Predictor::new_f16` /
  `folo predict --half` build on it. With [tensor cores](#tensor-cores-cooperative-matrices) on, the matmuls and
  convs among them run on the tensor-core kernels. **Mixed-precision training** (folo-core's `autocast`, docs/MODEL.md §8.8)
  adds the f16 gradients it needs: conv2d input and weight gradients (tensor cores when on; the weight gradient
  written as f32 through `Backend::conv2d_backward_weight_as`), the fused BatchNorm on f16 data with f32
  parameters and statistics, the max-pooling and nearest-upsampling gradients, and the elementwise gradients (the
  strided kernel was f16 already). The other gradient kernels (average pooling, the index-op scatters) refuse f16
  (`UnsupportedDType`); the detector does not use them. `src/tests/amp.rs` checks the BatchNorm, pooling and
  upsampling gradients and the autocast conv / matmul against f32 on the same f16-rounded operands. Without `shader-f16` folo-core refuses f16 compute (`DTypeMismatch`) and the predictor falls back
  to f32 (`predict::f16_unavailable` says why). `src/tests/f16.rs` checks every such op on f16 GPU tensors
  against the f32 CPU on the same f16-rounded inputs (`1e-3` relative for single roundings, `2e-3` plus a small
  absolute term for accumulations), and `crates/folo-vision/tests/gpu_f16.rs` the detector (decoded boxes and
  scores, the end-to-end predictor, f16 exports loading as f16, the CPU fallback).
- **Prediction** (folo-core's detector hooks): `letterbox_rgb8` and `detect_candidates` are the two kernels above.
  `folo_vision::predict::Predictor` on the GPU uploads the decoded images as raw `u8` (a quarter of the f32
  bytes) in one `upload_with` (below: the pixels copied in parallel straight into reused mapped staging memory, each
  image at a 4-byte aligned offset of one device buffer, which the letterbox kernel reads in place), letterboxes on
  the device, runs the forward, decodes into candidates, and downloads the per-image counts
  plus up to `CANDIDATE_READBACK` (1024) records per image in one readback (a second one only for an image with
  more): about 28 KiB per image instead of the 2.8 MB of a decoded `[A, 4 + nc]` head. Nothing waits on the GPU
  before that readback. `src/tests` has no tests for these two; they are tested against the CPU path in
  `crates/folo-vision/tests/gpu.rs` (letterbox bit for bit, candidates and detections to float rounding).
- **Readback.** `download` decodes the mapped staging buffer straight into typed host storage
  (`GpuContext::read_with`), and `Tensor::to_vec` takes that storage without another copy.
- **No round trips.** Kernels only encode; nothing waits until a download (`to_vec`, `item`, `to_device(Cpu)`)
  or `folo_core::backend::synchronize(dev)`. Uploads flush the open batch; they do not wait. A YOLOv8n-shaped
  forward at batch 1 is 131 dispatches in one submission; forward + backward is about 2 200 dispatches in 3
  submissions.
- **Semantics.** The CPU backend is the reference and every op is tested against it (`src/tests/backend/`, with
  the tolerances stated there): stable sigmoid / softplus, Rust's NaN-ignoring `max` / `min`, C `powf`, rounding
  half away from zero, NaN-aware comparisons, first-index ties for max pooling, arg-reductions and top-k. Two
  documented differences: index ops cannot report out-of-range indices without a host round trip, so
  `index_select` / `gather` return zeros and `index_add` / `scatter_add` skip such entries; and the two
  scatter-adds use float atomics, so their summation order is unspecified (results agree to rounding).
- **Autograd** is folo-core's, unchanged: `src/tests/backend_net.rs` runs a small conv net (conv, BatchNorm from
  composite ops, SiLU, max pooling, upsample, concat) on both devices and compares every gradient, runs
  folo-core's finite-difference `grad_check` through the GPU, and checks that folo-nn's SGD and AdamW steps on GPU
  parameters match the CPU. The fused ops are tested from folo-nn (`crates/folo-nn/tests/gpu.rs`: fused BatchNorm
  against composite ops and finite differences, Conv + BN + act inference parity, every optimizer variant, a moved
  training setup against the CPU) and folo-vision (`crates/folo-vision/tests/gpu.rs`: the detector's eval forward,
  a training step's loss and gradients, `.folo` loading onto the GPU).
- **Moving whole models**: `Tensor::to_device_` moves a storage in place (identity kept), and folo-nn's
  `Module::to_device` / `Optimizer::to_device` / `ModelEma::to_device` build on it (docs/NN.md).
- **Kernels outside the `Backend` trait.** `folo_gpu::register(ctx, device)` (which `init()` calls) also records the
  context, so `context_for(device)` finds it; `dense_buffer(&tensor)` returns the wgpu buffer of a dense GPU tensor
  and `tensor_from_buffer` wraps a pooled buffer as one. `folo_gpu::detect` uses them to run the task-aligned
  assigner and the detection loss straight on the head outputs: `detect::assign` returns the assignment as device
  tensors, `detect::detection_loss` the total as a custom op (its backward is the BCE and box backward kernels)
  plus a 12-float `stats` tensor (items, positives, target score sum) that the caller reads once. folo-vision's
  `detection_loss` takes this path for GPU tensors; `crates/folo-vision/tests/gpu_loss.rs` checks the assignment
  against the host assigner (identical positives, gts and labels, soft targets to 1e-5) and the loss and both
  gradients against the host loss.
- **Staged uploads.** `GpuContext::stage(&[u8])` copies host data into a new mapped-at-creation staging buffer and
  touches neither the command batch nor the queue, so another thread can prepare the next batch while the GPU runs;
  `upload_staged` then encodes one buffer copy into a pooled buffer in stream order (no flush, no wait). The
  trainer's feeder thread stages each `u8` batch this way (docs/MODEL.md §8.3).
- **Reused staging.** `GpuContext::upload_with(len, fill)` hands `fill` the mapped bytes of a host-visible staging
  buffer (`wgpu::WriteOnly`, re-exported as `folo_gpu::WriteOnly`; its fixed-size chunks are `Send`, so a caller
  can fill it from several threads), encodes the copy into a pooled buffer, submits, and maps the staging buffer
  again once the GPU is done with it; the last few (`STAGING_KEEP` = 4) stay mapped for the next upload of the same
  size class, and `release_memory` drops them. Against `upload` (`Queue::write_buffer`: a fresh wgpu staging
  allocation and a serial copy per call) the predictor's batch-16 upload went from about 12 ms to 3 ms
  (docs/hill/gpu-infer.md).

### Benchmark: a YOLOv8n-shaped network

`cargo bench -p folo-gpu --bench backend` builds a YOLOv8n-shaped network from plain folo-core ops (Conv-BN-SiLU,
C2f, SPPF, PAN neck with upsample + concat, a decoupled 3-scale head; 3.16 M parameters, 8.7 conv GFLOPs per
640×640 image) and times it, synchronised every iteration; medians of 30 runs on the GPU, 5 (batch 1) or 2
(batch 16) on the CPU:

- *infer*: inference forward with BatchNorm folded into the conv bias (conv + bias, then SiLU), `no_grad`;
- *train fwd*: forward with BatchNorm from batch statistics as composite ops, graph recorded;
- *train fwd+bwd*: the same plus `backward()` of a scalar loss over the three head outputs.

| 640×640 | device | infer | train fwd | train fwd+bwd |
|---|---|---|---|---|
| batch 1 | RTX 4080 SUPER (wgpu, Vulkan) | **4.3 ms** | 9.9 ms | **17.8 ms** |
| batch 1 | i7-14700KF (folo-core CPU, 28 threads) | 43.7 ms | 112 ms | 355 ms |
| batch 16 | RTX 4080 SUPER | **17.7 ms** (906 img/s) | 37.5 ms | **113 ms** (142 img/s) |
| batch 16 | i7-14700KF | 1 007 ms | 2 340 ms | 9 139 ms |

The GPU is 10x the CPU at batch 1 and 57x (inference) to 81x (training) at batch 16. Batch-1 inference spends
about 0.4 ms encoding on the CPU; repeated runs vary by about ±15% (GPU clocks on a shared desktop). This network is
built from plain ops: BatchNorm as composite ops is about ten full passes over each activation forward and twice
that backward, and SiLU is a separate pass. The real detector uses the fused kernels instead (folo-nn's `ConvBnAct`
on the GPU). `FOLO_BENCH_CPU=0` skips the CPU rows; `FOLO_BENCH_PROFILE=1` prints the encode / submit / wait split
of batch-1 inference.

### Benchmark: the folo-n / folo-s detector

`cargo bench -p folo-vision --bench gpu` runs the real detector (fused inference forward, end-to-end predict,
training step, peak memory); its numbers and the comparison with Ultralytics are in
[BENCHMARKS.md](BENCHMARKS.md#folo-on-the-gpu). In short, folo-n at 640 on the hill-climb build (2026-11-09): 2.09 ms batch-1
forward and 1648 img/s at batch 16 in f32 (1.65 ms / 3078 img/s in f16 on tensor cores), 2.63 ms / 1239 img/s end to
end through the predictor at batch 1 / 16 in f32 (2.21 ms / 1898 img/s on tensor cores), a 327 img/s training step at
batch 16 with a 3.5 GiB peak, and 537 img/s with mixed precision (`--amp`) at 2.0 GiB. The fused kernels' effect on the training step:
the SGD update went from 62 ms (about ten elementwise dispatches and copies per parameter) to 5 ms with the fused
update kernel, and to about 2 ms with the multi-tensor one. The training-step hill-climb (submission every 64
dispatches, constant-size weight gradients, stride-phase input gradients, multi-tensor updates, weight-gradient
tiles by shape; docs/hill/gpu-train.md) took the folo-n step from 226 to 307 img/s in f32 and 292 to 502 with AMP,
and folo-s from 105 to 140 / 167 to 246, on a host shared with other jobs (alternated A/B, medians of 3 rounds).

## Known limitations / follow-ups

- **Tensor cores are opt-in.** Without `FOLO_GPU_TENSOR_CORES` / `--tensor-cores` / `folo train --amp` the f16
  GEMM and convs keep f32 tiles and f32 FMAs (f16 saves memory traffic, not arithmetic: on Ada, non-tensor f16
  math is no faster than f32), which is why mixed-precision training refuses to run without them. The
  transposed-operand tensor-core GEMMs scatter into workgroup memory with scalar stores and are 15–40% slower than
  the plain layout (the attention products use them).
- Timings are wall-clock; GPU timestamp queries would make small-kernel numbers more precise.
- `max_pool2d` has no `ceil_mode`; upsampling is nearest-only.
- Backend: no general elementwise fusion. The fused paths are Conv + bias + activation (inference), BatchNorm +
  activation (training), the optimizer updates and the detection loss (`folo_gpu::detect`); other chains (residual
  adds, attention, gradient accumulation and clipping) are separate memory passes.
- Backend: strided convolution input gradients with dilation > 1 still run the transposed-conv GEMM over every
  tap (dilated strided convs keep the old path; the detector has none).
- Backend: index ops do not report out-of-range indices (see [folo-core backend](#folo-core-backend)), and top-k
  is one thread per row (`O(k · d)`), sized for assigner-style `k`.
- Backend: `Device::Gpu(0)` only; registering further adapters as `Gpu(1..)` is a few lines with
  `GpuBackend::new` but untested.
