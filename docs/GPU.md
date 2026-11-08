# folo-gpu: GPU compute on wgpu

`crates/folo-gpu` is FOLO's GPU layer. It runs on [wgpu](https://wgpu.rs) 30: Vulkan, DX12 and Metal natively,
WebGPU in the browser later. It is standalone for now: it does not depend on folo-core. A later PR plugs it into
folo-core's `Backend` trait (see [Backend integration](#planned-folo-core-backend-integration)).

```
GpuTensor (tensor.rs)          shape + dtype + pooled buffer; validates, allocates, calls kernels
   │
kernels/* (kernels/mod.rs)     one Rust entry point per WGSL kernel; raw wgpu::Buffer + explicit sizes
   │
GpuContext (context.rs)        adapter, device, queue, buffer pool, pipeline cache, command batch
```

The kernel layer never sees a `GpuTensor`. The future folo-core backend calls it directly with its own storage
type.

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
  use it, and the folo-core backend will too.
- **Errors.** Validation and out-of-memory errors are captured, not panicked on. `sync()` and readbacks return
  them as `GpuError::Wgpu`. A kernel that fails to compile returns an error from the call that needed it.

### Buffer pool

`ctx.alloc(bytes)` rounds the request up to a size class and takes a buffer from that class's free list, or
creates one. Size classes are quarter steps between powers of two, with a 256-byte floor (…, 1 MiB, 1.25 MiB,
1.5 MiB, 1.75 MiB, 2 MiB, …), so at most 25% of a buffer is slack. A dropped `GpuBuffer` goes back to its free
list. The free lists hold at most 2 GiB; `trim_pool()` empties them and `pool_stats()` reports hits, misses and
bytes.

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
64 KiB uniform arena, which is written once per submission.

### Pipeline cache and specialisation constants

A kernel is a `KernelDef`: a name, the access mode of each storage binding, and a WGSL source (static, or
generated from the spec). A `Spec` is a list of specialisation constants. They are rendered as WGSL `const`
declarations in front of the source, together with `alias T = f32|f16` and `enable f16;` when needed. Constants
can therefore size workgroup arrays, select code paths and compile bounds checks out. Pipelines are cached by
`(kernel name, spec)`, and compilation happens on first use. Rendering constants as source text works the same
on every backend, including browser WebGPU; pipeline-overridable constants cannot size arrays on every backend.

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
- `CONV = true` turns the same generator into implicit-GEMM conv. The B tile is gathered straight from the NCHW
  image. Each thread keeps one output pixel for the whole K loop, so the per-element index math is a few
  divisions by constants. No im2col buffer is ever written.

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

## Planned folo-core Backend integration

folo-core is being written alongside this crate. The plan for the follow-up PR:

- folo-gpu depends on folo-core (never the other way round) and implements its `Backend` trait for a
  `GpuBackend { ctx: GpuContext }`. At runtime it registers itself for `Device::Gpu(ordinal)`.
- GPU tensor storage in folo-core is an opaque handle around `Arc<GpuBuffer>` plus dtype. The backend's kernel
  methods take folo-core's (shape, strides, offset) views. Contiguous views map directly onto the kernel layer's
  dense-buffer convention. Strided views are made contiguous first with `kernels::layout::permute` (one copy),
  except for GEMM, which takes leading dimensions, transposes and batch strides natively.
- Elementwise / broadcast / reduce / softmax / conv / pool / upsample / concat map one-to-one onto the
  `kernels::*` entry points. Fused bias + activation and the conv and GEMM epilogues let folo-nn fuse
  Conv-BN-SiLU for inference.
- Autograd stays in folo-core. Backward kernels (conv weight/input gradients via the transposed GEMM variants,
  activation gradients, pooling argmax) come in the same PR or right after it.
- Host ↔ device transfers go through `GpuContext::upload` / `read`. Batching means a whole forward pass encodes
  into a few submissions. folo-core calls `sync()` only at readback or explicit synchronisation points.

## Known limitations / follow-ups

- Elementwise, reduce and softmax are f32-only; the f16 coverage is GEMM, conv, pooling, upsample, concat,
  permute and casts.
- The f16 depthwise conv is slower than the f32 one (2-byte scalar loads); vectorising it is a follow-up.
- Timings are wall-clock; GPU timestamp queries would make small-kernel numbers more precise.
- `max_pool2d` has no `ceil_mode`; upsampling is nearest-only.
