# folo-gpu: GPU compute on wgpu

`crates/folo-gpu` is FOLO's GPU layer. It runs on [wgpu](https://wgpu.rs) 30: Vulkan, DX12 and Metal natively,
WebGPU in the browser (through folo-wasm, see [In the browser](#in-the-browser) and docs/WASM.md). It implements folo-core's `Backend` trait, so folo-core tensors train and infer on
the GPU (see [folo-core backend](#folo-core-backend)).

```
GpuBackend (backend.rs)        folo-core's Backend: Storage::Device(GpuStorage) for every dtype, views → kernels
GpuTensor (tensor.rs)          standalone shape + dtype + pooled buffer handle for kernel-level work
   │
kernels/* (kernels/mod.rs)     one Rust entry point per WGSL kernel; raw wgpu::Buffer + explicit sizes
   │
GpuContext (context.rs)        adapter, device, queue, buffer pool, pipeline cache, command batch
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
- **Features and limits.** `shader-f16` is requested when the adapter has it. The device asks for the adapter's
  full limits: the largest storage buffers and bindings, and all the workgroup memory (48 KiB on NVIDIA Vulkan).
  Adapters with fewer than 4 storage buffers per stage, less than 16 KiB of workgroup memory or fewer than 256
  invocations per workgroup are rejected.
- **`folo doctor`.** `GpuContext::enumerate()` lists every adapter as an `AdapterSummary` (name, backend, type,
  driver, `shader-f16`, key limits, subgroup size); `ctx.summary()` describes the one in use.
- **Shared context.** `GpuContext::shared()` is a process-wide context created once from the environment. Tests
  and the folo-core backend (`folo_gpu::init()`) use it.
- **Errors.** Validation and out-of-memory errors are captured, not panicked on. `sync()` and readbacks return
  them as `GpuError::Wgpu`. A kernel that fails to compile returns an error from the call that needed it.

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

Reusing a buffer while earlier, unsubmitted work still reads it is safe. All work runs on one queue in encoding
order, and wgpu puts barriers between dispatches that touch the same buffer. The one exception is
`Queue::write_buffer`, which runs before the next submission's commands. That is why every upload first submits
the open batch.

### Command batching

Kernels never submit. Each call encodes a dispatch into one long-lived compute pass of an open command encoder.
The batch is submitted when:

- 1024 dispatches have been encoded;
- an upload or readback needs ordering;
- you call `ctx.flush()` (submit without waiting) or `ctx.sync()` (submit, wait, report errors).

A forward pass is therefore a handful of submissions. Each dispatch's parameters take a 256-byte slot in a
64 KiB uniform arena, which is written once per submission. `ctx.work_stats()` counts dispatches and submissions
(the backend benchmark prints them per pass).

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
| unary | `kernels::elementwise::unary` | f32 | exp, log, sqrt, rsqrt, neg, abs, sigmoid, silu, relu, relu6, tanh, gelu (tanh), leaky_relu, hardswish, hardsigmoid, square, recip, sin, cos, floor, ceil; vec4 path when `n % 4 == 0` |
| binary | `kernels::elementwise::binary` | f32 | add, sub, mul, div, max, min, pow (C `powf` semantics). NumPy broadcasting; dims are coalesced first (`[N,C,H,W] + [1,C,1,1]` becomes rank 3); same-shape vec4 fast path; rank ≤ 8 after coalescing |
| bias + activation | `kernels::elementwise::bias_act` | f32 | `act(x + bias[(i / inner) % C])`, or the activation alone |
| reduce | `kernels::reduce::reduce` | f32 | sum, mean, max, min over the middle axis of `[outer, R, inner]`, i.e. any dim. `inner == 1`: one workgroup per row with a tree in workgroup memory; otherwise each thread owns a column and loads are coalesced |
| softmax / log_softmax | `kernels::reduce::softmax` | f32 | last dim, one workgroup per row, max-shifted |
| GEMM | `kernels::gemm::gemm` | f32, f16 | `alpha·op(A)·op(B) (+ row/col bias) → act`; A, Aᵀ, B, Bᵀ; two-level batch with arbitrary offsets and strides (stride 0 broadcasts); f16 storage with f32 accumulation |
| conv2d | `kernels::conv::conv2d` | f32, f16 | NCHW; stride, padding, dilation, groups; fused bias + activation. Implicit GEMM, a pointwise GEMM fast path for 1×1, and a direct depthwise kernel |
| max_pool2d | `kernels::pooling::max_pool2d` | f32, f16 | kernel / stride / padding / dilation; padding never wins (−∞); floor mode |
| upsample_nearest2d | `kernels::layout::upsample_nearest2d` | f32, f16 | any output size; `src = dst · in / out`, as in PyTorch `nearest` |
| concat | `kernels::layout::copy_rows` | f32, f16 | any dim, via row-blocked copies; a single block becomes a buffer copy; f16 blocks need even lengths and offsets |
| permute / transpose | `kernels::layout::permute` | f32, f16 | dims are coalesced; batched 2-D transposes (NCHW ↔ NHWC) go through a padded 32×32 tile, everything else through a generic strided copy |
| cast | `kernels::layout::cast` | f32 ↔ f16 | packed words; needs no `shader-f16` |
| strided gather | `kernels::strided::gather_strided` | any (1/2/4/8-byte) | make-contiguous of any view (offset, strides, stride-0 broadcast); dims coalesced; contiguous views become a buffer copy; sub-word elements assembled one output word per thread |
| strided scatter | `kernels::strided::scatter_strided` | any | `copy_into` (cat, pad, view gradients, `copy_from`); sub-word elements written with `atomicAnd` / `atomicOr` on their own bits |
| fill / cast | `kernels::strided::{fill_words, cast}` | any | `full` with a two-word pattern (i64) or `clear_buffer` for zeros; casts between all five dtypes with Rust `as` semantics (truncate, saturate, NaN → 0) |
| index_select / gather | `kernels::strided::{index_select, gather}` | any | `i64` or `u32` indices; out-of-range indices give zeros |
| index_add / scatter_add | `kernels::strided::{index_add, scatter_add}` | f32 | compare-and-swap float atomics; out-of-range indices skipped |
| strided elementwise | `kernels::ew::elementwise` | f32 | folo-core's unary ops and their gradients, binary ops and comparisons, `where` (f32 or u8 mask); every operand a strided view read in place, linear + vec4 path when all are contiguous |
| arg-reduce | `kernels::reduce::arg_reduce` | f32 → i64 | first max / min, NaN wins |
| split reduce | `kernels::reduce::reduce_split` | f32 | long reductions over few outputs (BatchNorm statistics) in two passes |
| softmax (any dim) | `kernels::reduce::softmax_dim` | f32 | rows kernel for the last dim, a column kernel otherwise |
| top-k | `kernels::reduce::topk` | f32 → (f32, i64) | sorted, ties by index, `total_cmp` order; one thread per row |
| pooling + gradients | `kernels::pool_grad::*` | f32 | max pooling with indices and its gather backward, avg / adaptive avg pooling and backward, upsample backward |
| conv2d gradients | `kernels::conv::{conv2d_backward_input, conv2d_backward_weight}` | f32 | implicit GEMM (`CONV_MODE` 2 / 3) with split-K for the weight gradient, a transposed-A GEMM for 1×1 input gradients, direct depthwise kernels |
| train-mode BatchNorm + activation | `kernels::norm::{batch_norm_forward, batch_norm_backward}` | f32 | NCHW, statistics over `N·H·W`. Forward: per-(channel, chunk) Welford partials merged with Chan's formula, a per-channel finalize (mean, `invstd`, the running-statistics update in place), then `act(x · scale + shift)`; backward: partial sums of `dz` and `dz · x̂` with the activation's derivative at the recomputed pre-activation, a finalize (`dγ`, `dβ`), then `dx`. Identity, SiLU or ReLU; 3 dispatches each way |
| optimizer updates | `kernels::optim::optimizer_step` | f32 | in place, one thread per element: SGD (momentum, Nesterov, coupled decay), AdamW (decoupled decay, bias corrections), and `p = a·p + b·g` for EMAs, in the op order of folo-nn's CPU loops |
| task-aligned assigner | `kernels::detect::assign` | f32 / u32 | docs/MODEL.md §7.3 on the raw head outputs, five dispatches: DFL decode of every anchor; one workgroup per gt that scans its candidates, scores `u = max(CIoU, 0)` and `m = score^α · u^β` and keeps its top k by `(m desc, anchor asc)` (per-thread sorted lists merged by a k-round tournament); two atomic passes for the conflicts (largest `u`, then the smallest gt index); the soft targets `t = m · u_max / (m_max + eps)` per gt and a dense per-anchor `(label, t)` map. Same decisions as `folo_vision::assign` |
| detection loss | `kernels::detect::{bce_forward, bce_backward, box_terms, combine}` | f32 | §7.4: BCE over every logit against the dense targets, each element formed whole; CIoU + DFL per positive (softmax expectation, both terms and their hand-written backward in one thread); a one-workgroup fixed-order reduction to the reported items, the total and the backward scales. `exp`, `ln(1 + x)` and `atan` are written out in plain arithmetic (folo-core's `fast_exp`, an atanh series, Cephes `atanf`), since WGSL only promises an absolute error of 2^-21 for `log` near 1 |
| letterbox | `kernels::vision::letterbox_rgb8` | u8 → f32 | one thread per canvas pixel: the padding grey, or the resized image sampled with folo-data's `ResizePlan` (integer box prefilter computed on the fly, then 11-bit fixed-point bilinear taps uploaded by the host), in integers, times `scale`: bit-exact with the CPU letterbox. Prefilter blocks whose sum could overflow u32 are refused (the caller letterboxes on the host) |
| detection candidates | `kernels::vision::detect_candidates` | f32 → u32 | one thread per (image, anchor): the first maximum of the `nc` sigmoid scores; above `conf`, the DFL softmax expectation of each side, the box around the anchor times the stride, and a 7-word record `(anchor, class, score, x1, y1, x2, y2)` appended with an atomic counter per image (the output is cleared first) |

The f16 kernels need `shader-f16`; without it they return `GpuError::Unsupported`. Everything not listed as f16
is f32-only.

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
  its rows.
- `CONV_MODE = 1` turns the same generator into implicit-GEMM conv. The B tile is gathered straight from the NCHW
  image. Each thread keeps one output pixel for the whole K loop, so the per-element index math is a few
  divisions by constants. No im2col buffer is ever written.
- `CONV_MODE = 2` is the input gradient as a transposed conv: each thread owns one input pixel and, for
  `k = (c_out, kh, kw)`, reads the output-gradient pixel `((ih + PH - kh·DH) / SH, (iw + PW - kw·DW) / SW)` when
  the divisions are exact and in range (a strided conv leaves the rest zero). `A` is the weight permuted once to
  `[groups, C_in/g, C_out/g, KH·KW]`.
- `CONV_MODE = 3` is the weight gradient: `K = N·OH·OW` folds the batch into the reduction, `A` rows are output-
  gradient planes read across images, and `B` is the image gathered as a transposed im2col (each thread owns one
  `(c_in, kh, kw)` column). `SPLIT = true` splits `K` into chunks, one per extra workgroup layer, each writing a
  partial weight slab that a column reduction sums: early layers have only a few output tiles but hundreds of
  thousands of `K` steps.

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
- **Not tried yet:** subgroup (warp) shuffles, `subgroupMatrix` / cooperative matrices (not in stable WebGPU),
  split-K for skinny problems, and a persistent-kernel / stream-K schedule.

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

| conv2d, N = 8, 80×80, 64 → 128 | ms | GFLOPS |
|---|---|---|
| 3×3 s1 p1 | 0.291 | 25 919 |
| 3×3 s2 p1 | 0.111 | 16 965 |
| 1×1 s1 | 0.041 | 20 579 |
| depthwise 3×3 s1, 128 ch | 0.069 | 1 707 (memory-bound) |
| f16: 3×3 s1 / 3×3 s2 / 1×1 / depthwise | 0.270 / 0.110 / 0.039 / 0.113 | 27 935 / 17 107 / 21 272 / 1 048 |

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
  operand folds the batch into `M`) and copies an operand only when no such mapping exists. Every other kernel
  takes dense data, so a strided input is copied once first.
- **Op mapping.** Reductions: sum / max / min through the reduce kernels (split into two passes when long rows
  meet few outputs), arg-reductions to `i64`. Softmax over any dim. conv2d forward is the existing kernel; its
  gradients are the `CONV_MODE` 2 / 3 GEMMs or the depthwise kernels. Pooling, upsampling, index ops and top-k use
  the kernels in the table above.
- **Fused ops** (folo-core's optional `Backend` hooks, each with a composite fallback on other backends):
  `conv2d_act` is the conv kernel with bias and activation in its epilogue (`Tensor::conv2d_act`, used by folo-nn's
  `ConvBnAct` in eval / fused mode, so a fused folo-n forward is one dispatch per conv);
  `batch_norm_act` / `batch_norm_act_backward` are the BatchNorm kernels above (`Tensor::batch_norm_act_train`,
  recorded by folo-nn's `BatchNorm2d` as one custom op); `optimizer_step` is the update kernel
  (`Tensor::optimizer_step_`, used by folo-nn's SGD, AdamW and EMA on the GPU).
- **Prediction** (folo-core's detector hooks): `letterbox_rgb8` and `detect_candidates` are the two kernels above.
  `folo_vision::predict::Predictor` on the GPU uploads each decoded image as raw `u8` (a quarter of the f32
  bytes), letterboxes on the device, runs the forward, decodes into candidates, and downloads the per-image counts
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
- **Semantics.** The CPU backend is the reference and every op is tested against it (`src/tests/backend.rs`, with
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
[BENCHMARKS.md](BENCHMARKS.md#folo-on-the-gpu). In short, folo-n at 640: 2.99 ms batch-1 forward, 1385–1461 img/s
at batch 16, 5.18 ms / 722 img/s end to end through the predictor at batch 1 / 16, 219 img/s training step at batch 16 (143 before the loss moved to the GPU), 3.5 GiB training peak. The fused kernels' effect on the training step:
the SGD update went from 62 ms (about ten elementwise dispatches and copies per parameter) to 5 ms with the fused
update kernel.

## Known limitations / follow-ups

- Elementwise, reduce and softmax are f32-only; the f16 coverage is GEMM, conv, pooling, upsample, concat,
  permute and casts.
- The f16 depthwise conv is slower than the f32 one (2-byte scalar loads); vectorising it is a follow-up.
- Timings are wall-clock; GPU timestamp queries would make small-kernel numbers more precise.
- `max_pool2d` has no `ceil_mode`; upsampling is nearest-only.
- Backend: no general elementwise fusion. The fused paths are Conv + bias + activation (inference), BatchNorm +
  activation (training), the optimizer updates and the detection loss (`folo_gpu::detect`); other chains (residual
  adds, attention, gradient accumulation and clipping) are separate memory passes.
- Backend: strided convolution input gradients run the transposed-conv GEMM over every `(kh, kw)` tap and skip
  the positions the stride never touched (3/4 of the B tile is zero at stride 2); a stride-phase decomposition
  would remove that waste.
- Backend: index ops do not report out-of-range indices (see [folo-core backend](#folo-core-backend)), and top-k
  is one thread per row (`O(k · d)`), sized for assigner-style `k`.
- Backend: `Device::Gpu(0)` only; registering further adapters as `Gpu(1..)` is a few lines with
  `GpuBackend::new` but untested.
