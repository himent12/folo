# folo-core: tensors, CPU kernels and autograd

`crates/folo-core` is the tensor engine every other FOLO crate builds on: n-d tensors over shared storage, the CPU
backend (the reference every other device is tested against), reverse-mode autograd, mixed precision and the seeded
RNG, plus the `Backend` trait a device implements. The GPU side of that trait is [GPU.md](GPU.md), the layers and
optimizers built on the tensors are [NN.md](NN.md), and [ARCHITECTURE.md](ARCHITECTURE.md) shows where the crate sits.
The crate's file map and its common changes are in [crates/folo-core/README.md](../crates/folo-core/README.md).

```text
tensor.rs, layout.rs, shape.rs   the Tensor handle, Layout (shape, strides, offset), shapes and broadcasting
dtype.rs, device.rs, storage.rs  DType, Device, Storage (CpuStorage, or a backend's DeviceStorage)
ops/                             the Tensor ops: view, copy, elementwise, reduce, nn (matmul, conv, pooling),
                                 fused, vision (detector hooks), inplace, muon, overload (+ - * /)
backend.rs                       the Backend trait and the device registry
cpu/                             the CPU backend: gemm/, conv/, elementwise, reduce, index, pool (pooling and
                                 upsampling), pool_alloc (the buffer cache)
autograd.rs, autocast.rs         the tape (Grads, no_grad) and mixed precision
rng.rs, testing.rs               Rng, grad_check
```

## Tensors, layouts and views
- `Tensor` is a cheap-to-clone handle (`Arc` inside) over shared storage plus a `Layout` (shape, strides, offset).
- Views (reshape of compatible layouts, transpose/permute, narrow/slice with step, broadcast/expand,
  squeeze/unsqueeze, flatten) do not copy.
- Ops return `Result`; `+ - * /` (on `Tensor`, `&Tensor`, `f32`) return `Result<Tensor>` too. Errors name the op and
  both shapes.

## Dtypes
- `DType`: `F32` (compute), `F16` (storage / GPU inference), `U8` (images), `U32`/`I64` (indices).
- Arithmetic is `F32`; copies, views, `cat`, `index_select`, `gather` and casts work for every dtype. Comparisons
  return 0/1 masks of their operands' dtype.
- On a backend that computes in f16 (`Backend::computes_f16`: the GPU with `shader-f16`) the forward ops also take
  `F16` operands and return `F16` (half-precision inference); every float operand of an op has one dtype.
- The dtype lives with the storage, like the device, so `Tensor::to_dtype_` / `Module::to_dtype` convert in place.

## Devices and the Backend trait
- `Device`: `Cpu` or `Gpu(ordinal)`. Kernels sit behind the `Backend` trait (the contract is the last section of this
  file); the CPU backend lives in folo-core and is built in: `register_backend` refuses `Device::Cpu`.
- `Tensor::to_device_(device)` moves a storage to another device **in place**: the device lives on the shared storage
  cell, so every clone and view follows and the tensor keeps its id (gradients and optimizer state stay keyed). This
  is how whole models move (`Module::to_device`).
- Tensors also move between devices with `to_device` (autograd flows back across the transfer).

## CPU kernels
Contiguous fast paths run with rayon above ~32K elements; strided inputs are walked row by row after merging
dimensions, or made contiguous once.

### GEMM
- A BLIS-style GEMM (`cpu::gemm`, packed panels, 6x16 AVX2/FMA micro-kernel chosen at runtime, tile- or K-split
  parallelism) backs `matmul` and conv2d.
- Each GEMM task packs the B panels of its own columns into an L2-sized buffer one `KC` block at a time, reading
  row-contiguous operands row by row so the prefetcher can follow (column tiles first, rows only when there are too
  few columns, about three tasks per thread and none under ~1 MFLOP).
- A row-major A whose packed form fits one K super-block (16 MiB: every conv weight) is not packed at all: the
  micro-kernels broadcast its six rows through row pointers, the same FMAs in the same order (bit for bit the packed
  path), so a conv forward has no serial packing pass.
- A batched `matmul` splits the pool between its matrices.

### conv2d forward
- conv2d (any kernel, stride, padding or dilation, not depthwise) never builds an im2col matrix: the input is copied
  once, zero-padded and split into its stride phases (`cpu/conv/shifted.rs`), so every `(channel, ky, kx)` row of B
  is a contiguous run of that copy, read in place by a variant of the micro-kernel through a row-pointer table (a few
  dropped columns per output row; planes are spaced an odd number of cache lines apart so the rows do not alias in
  L1; stride-2 rows are padded once and both column phases dealt out in one pass).
- Pointwise convs pack B as a plain copy.
- Depthwise convs have a direct kernel (`cpu/conv/depthwise.rs`; with AVX2 at stride 1: the plane copied once with
  its padding, every output vector its taps' FMAs in registers).
- So do convs with at most 4 input channels (one group, not pointwise, `c·kh·kw <= 256`: the 3-channel stem,
  `cpu/conv/small_c.rs`): per output row the
  `c·kh·kw` taps are gathered into L1 and every output channel is that many FMAs per pixel in the GEMM's order, bit
  for bit the GEMM result, without the per-channel phase copy.
- Bias and activation run as an epilogue on each finished output (`Tensor::conv2d_act`, used by fused `ConvBnAct`
  blocks and by eval-mode blocks under `no_grad`): bias and SiLU on the micro-kernel's accumulators before they are
  stored, other activations as a pass over each finished block. SiLU there is an 8-lane AVX2 `x · sigmoid(x)` with
  relative error below `cpu::SILU_MAX_REL_ERR` (5e-7, tested).
- A batch of several images splits the pool between them: only as many images run at once (`lanes`) as their
  shifted-rows copies or unfolded gradients fit in 16 MiB, each lane taking its images one after another, and each
  image's GEMMs get `threads / lanes` threads (work stealing spreads the tiles).

### Winograd F(2x2, 3x3)
- The fused entry points (`conv2d_act`, `conv2d_act_cat`: nothing is recorded) run a dense 3x3 stride-1 conv with at
  least 16 input and 8 output channels (at most 4096 together, which bounds the per-thread scratch), and at least
  256 2x2 output tiles (`⌈OH/2⌉ · ⌈OW/2⌉ >= 256`) or `c · o >= 4096`, as Winograd F(2x2, 3x3)
  (`cpu/conv/winograd.rs`): 16 multiplies per 2x2 output tile and channel pair instead of 36.
- The weights are transformed once per weight tensor and kept with its storage until the storage is next written
  (`View::derived`: every write lock clears it; a view whose `storage` field was pointed elsewhere neither uses nor
  fills it).
- The tiles are cut into blocks; a block transforms its input patches, runs 16 small GEMMs through the row-pointer
  micro-kernel and transforms the results into its output tiles while they are in L2. An image with too few blocks
  for the pool runs in two passes: the input transforms split by input channels, then the GEMMs and output transforms
  by output channels.
- Not bit for bit the direct conv: an output is within `(C + 8) · 2^-23 · 9 · Σ_c D_c · G_c` of the exact sum
  (`D_c`, `G_c`: the sums of |inputs| over its tile's 4x4 input patch and of |weights| over its 3x3 kernel in channel
  c; derived in the module doc, tested; it assumes no step overflows or underflows, and inputs within a factor ~4 of
  `f32::MAX` can overflow in the input transform where the direct sum would not), an error that grows with C like the
  direct conv's `K · 2^-23 · Σ|terms|`; COCO val AP of folo-n-coco changed by less than 1e-6.
- The same bits for any thread count or batch. The autograd conv (`conv2d_with`) keeps the direct conv.

### Convs at some output sites
- `Tensor::conv2d_act_at` (`Backend::conv2d_act_at`) is the same inference conv needed only at some output sites
  (flat `n · OH·OW + oy · OW + ox`, a `U32` vector): the result has the full output shape, the sites hold the dense
  conv's values and every other element is unspecified.
- On the CPU a Winograd conv runs the block code over the list of 2x2 tiles that hold a site
  (`winograd_forward_at`: a tile's arithmetic and its GEMM sums do not depend on which tiles share its block) and a
  pointwise conv (one group, fewer than 2048 input channels) runs the dense GEMM over the sites' columns gathered
  into planes of their own (`cpu/conv/sites.rs`), both bit for bit the dense values; any other conv runs densely, as
  does a backend without the kernel.
- A site reads only its receptive field: in Winograd F(2x2, 3x3) each output of a tile reads only its own 3x3 window
  of the 4x4 patch (tested with NaN everywhere else).
- On the GPU, `Tensor::conv2d_act_sites` takes per image a fixed-capacity list of output pixels (a count, then the
  pixels): folo-gpu's implicit GEMM with `SITES` (`kernels/conv/sites.rs`; regular and tensor-core generators) maps
  column `j` to the image's `j`-th listed pixel, runs the dense conv's tile config and split-K chunks (planned for the
  full output, so each value is the dense one bit for bit) and retires workgroups past the image's count; the result
  is compact `[N, O, cap]` or scattered back into a full map. `detect_class_sites` and `detect_decode_sites`
  (`kernels/detect_sites.rs`) are the head's class pass into such lists and the decode of the boxes computed there.

### Concatenated inputs
`Tensor::conv2d_act_cat` runs the same conv over the channel concatenation of several inputs, reading each part (or a
channel slice of a tensor) in place, so `cat` + conv never writes the concatenation.

### conv2d gradients
- The input gradient (`cpu/conv/backward_input.rs`): pointwise convs multiply by `Wᵀ` directly (`Wᵀ · gy`). Other
  non-depthwise convs with stride 1 and `p <= d (k - 1)` on both axes run the forward kernel on the output gradient
  with the flipped, channel-swapped kernel (no col2im); the rest use an unfolded gradient and col2im. Depthwise convs
  have their own kernel (`cpu/conv/depthwise.rs`).
- The weight gradient of a non-pointwise stride-1 conv, or of a pointwise one with `C/groups >= 64`, is
  `dWᵀ = X_unfolded · gyᵀ` with the input's shifted rows as A, read in place by a 6x16 micro-kernel through six row
  pointers (`gemm_arows`), and the output gradient transposed once per image into packed B panels (zeros in the
  dropped columns); K is cut into parts and the rows into tiles by shape alone, partials added in order. Other convs
  pack the unfolded input as they go (`cpu/conv/backward_weight.rs`).
- In a batch, the weight gradient makes its images' operands in waves of `lanes` images (above), each wave tiling its
  rows over the K parts it touches.

### Pooling
Max pooling is separable (row pass, then column pass; same first-maximum argmax as a direct scan). When nothing is
recorded (`no_grad`, or an input that does not require grad) it makes no index tensor (`Backend::max_pool2d_values`,
on the CPU a select over `-inf`-padded rows, the same bits; a window far wider than the plane takes the clipped
indexed scan instead).

### BatchNorm statistics
folo-nn's train-mode BatchNorm on the CPU (`crates/folo-nn/src/layers/batchnorm.rs`, [NN.md](NN.md)) is built on
these kernels:
- It takes its statistics in one read of x, in `f64` and in blocks of 256 floats per plane: the 8-lane plane sums of
  the old two-pass code (so the mean is bit for bit its mean), and each block's sum of squared deviations from its own
  mean, merged per plane and per channel with Chan's formula (no cancellation).
- It applies SiLU / ReLU in its own passes (forward and backward), bit for bit BatchNorm then `Tensor::silu` / `relu`
  (`cpu::silu` / `cpu::silu_backward` are the shared formulas).
- Its per-plane loops run as AVX2 when the CPU has it (the same loops compiled under `target_feature`, bit for bit).

### The buffer cache
`f32` buffers from 16 KiB are recycled through a size-keyed cache (`cpu/pool_alloc.rs`;
`cpu::set_buffer_cache_limit`, default 1 GiB; `cpu::take_buffer` hands one to custom ops) so training and inference
loops do not pay page faults every step. When it is full, lengths unused for two limits of traffic, and for twice the
longest they ever went unused before coming back, are evicted for the ones now asked for, so a later workload
(another model, another input shape) gets cached too while one that cycles through more than the cache holds keeps a
cache-sized part.

### The dedicated pool
`cpu::in_pool` runs a closure on a worker of a dedicated rayon pool (so the per-thread state cannot reach other jobs of
the global pool) with the caller's grad mode and autocast state. CPU forwards and predictions use it so their many
parallel kernels start inside a pool instead of each waking one from outside.

## Determinism
CPU results are bitwise identical from run to run and for every rayon thread count:
- full and long-row reductions use fixed blocks (8192 elements) combined by a fixed-order pairwise tree;
- reductions over inner dims give each output element to one task;
- the GEMM's K split depends only on the shape;
- accumulating scatters (`index_add`, `scatter_add` and the `index_select` / `gather` / `topk` gradients) partition
  work by destination and add sources in index order.

`crates/folo-core/tests/determinism.rs` checks this across pool sizes.

## Autograd
- Define-by-run, reverse mode. An op whose input requires grad records a node (inputs, their storage versions, a
  backward closure built from ordinary tensor ops) on the result.
- `loss.backward()` returns `Grads` (keyed by tensor id, leaves only) via a topological sort; graphs are retained, so
  backward can run twice. Graph teardown is iterative (deep graphs do not overflow the stack).
- A thread-local `no_grad()` guard disables recording.
- Fan-in gradients are summed in place when the running sum is a fresh tensor nothing else holds.
- `testing::grad_check` compares autograd against central finite differences (f64 loss accumulation); every
  differentiable op has a check in `crates/folo-core/tests/grad.rs`.

## In-place updates
In-place updates for optimizers (`sub_assign`, `add_scaled_assign`, `copy_from`, `fill_assign`, `update_f32`, ...)
refuse to touch a tensor that requires grad unless `no_grad` is active, and bump the storage version: backward
through a graph that saved the old values fails with `Error::ModifiedInPlace` instead of using stale data.

## Custom and fused ops
- `Tensor::custom_op(name, inputs, output, backward)` lets downstream crates record fused kernels (Conv+BN+SiLU,
  BatchNorm, DFL / CIoU losses) with a hand-written backward; inputs and output are version-checked like built-in ops
  and a wrong gradient count or shape is an error at `backward()`.
- Fused ops with composite fallbacks (`ops/fused.rs`): `Tensor::conv2d_act` (conv + bias + activation),
  `Tensor::conv2d_act_cat` (the same over a channel concatenation, without writing it),
  `Tensor::batch_norm_act_train` / `batch_norm_act_train_backward` (train-mode BatchNorm + activation, running update
  in place) and `Tensor::optimizer_step_` (in-place SGD / AdamW / EMA update) call optional `Backend` hooks. folo-gpu
  implements them, the CPU backend implements `conv2d_act` and `conv2d_act_cat`, other backends fall back.

## Detector pre- and post-processing hooks
`ops/vision.rs` holds four hooks that return `Ok(None)` without a kernel, so callers keep their host code; folo-gpu
implements all four:
- `Tensor::letterbox_rgb8`: raw `u8` images → letterboxed `f32` NCHW input, sampled with folo-data's exact resize
  taps.
- `Tensor::detect_candidates`: head → per-anchor best class above `conf` with its decoded box, as a compact list.
- `Tensor::detect_class_sites` and `Tensor::detect_decode_sites`: the class pass of one head level into a list of
  candidate sites, and the decode of the boxes computed at those sites (candidate-first prediction, above).

## Mixed precision
`folo_core::autocast` works like `torch.autocast`:
- While an `autocast()` guard is alive on the thread, `conv2d_with` and `matmul` of tensors on a backend that computes
  in f16 run in f16 (`F32` operands cast by recorded casts; a conv uses f16 copies of its weight and bias but is
  recorded against the f32 ones, and its backward writes the weight gradient straight to f32 with
  `Backend::conv2d_backward_weight_as`), and softmax runs in f32.
- Everything downstream runs in the dtype it receives; the fused train-mode BatchNorm takes f16 data with f32
  parameters and statistics.
- `Tensor::unscale_check_` (a `Backend::unscale_check` hook) is the loss scaler's fused unscale and overflow flag.
- `StepGuard::new(&flag)` (a `Backend::set_step_guard` hook; `src/guard.rs`) makes the optimizer updates that follow
  on that device (`optimizer_step`, the `*_many` updates, Muon's parameter update) change nothing when the one-element
  f32 `flag` is not finite, until the guard drops: mixed precision steps without waiting for the overflow check.
  `Ok(None)` where the backend has none; updates outside those kernels ask `step_guard_allows()` first.

## RNG
Seeded RNG (`folo_core::rng::Rng`, xoshiro256++) for reproducible init, augmentation and tests. `Rng::ALGO`
(`"xoshiro256pp"`), `state()` and `from_state()` save and resume the exact stream (the `folo.rng` checkpoint entry).

## Backend integration contract
This is the contract for folo-gpu and any other device.
- **Registration.** Implement `folo_core::backend::Backend` and call `register_backend(Device::Gpu(i),
  Arc::new(backend))` at startup (for example from `folo_gpu::init()`). `backend(device)` looks it up; tensors on an
  unregistered device fail with `Error::BackendNotRegistered`.
- **Storage.** Device memory is `Storage::Device(DeviceStorage)`: folo-core stores the device, dtype and element count
  and an opaque `Box<dyn Any + Send + Sync>` buffer that only the backend downcasts. Build results with
  `DeviceStorage::new` and wrap foreign buffers into tensors with `Tensor::from_storage`.
- **Views.** Every method receives `View`s (storage + `Layout`). Layouts may be strided or broadcast (stride 0);
  outputs are fresh contiguous storages on the same device. folo-core validates shapes, dtypes and devices before
  calling; binary and `where_cond` operands arrive already broadcast to the output shape, and `matmul` operands have
  identical, already broadcast batch dims.
- **Required methods:** `name`, `full`, `upload`, `download` (to host, logical order), `copy_strided`, `copy_into`
  (strided write used by `cat`, `pad`, view gradients and `copy_from`), `cast`.
- **Compute methods** (unary + `unary_backward`, binary, `where_cond`, `reduce`, `softmax`, `matmul`, conv2d forward /
  backward-input / backward-weight, pooling and upsampling with their backwards,
  `index_select`/`index_add`/`gather`/`scatter_add`, `topk`) default to `Error::NotSupported { op, backend }`, so a
  backend can grow op by op. Gradients are composed from the same methods, so an op is trainable on a device once its
  forward and `*_backward` methods exist.
- **Gradient fast paths** with composing defaults: `index_select_backward` and `gather_backward` (default `full`
  zeros then `index_add` / `scatter_add`).
- `max_pool2d_backward` also receives the pooling window (`Pool2dParams`), so a backend can compute the gradient as a
  gather (each input element visits only the windows that cover it) instead of a scatter with float atomics; the CPU
  backend ignores it.
- **Fused hooks** (`conv2d_act`, `conv2d_act_cat`, `batch_norm_act`, `batch_norm_act_backward`, `optimizer_step`,
  `letterbox_rgb8`, `detect_candidates`) default to `NotSupported`; folo-core's fused entry points then compose
  ordinary ops (or report "not available" so folo-nn uses its own path). `batch_norm_act` and `optimizer_step` write
  into `&mut Storage` arguments (running statistics, parameters, optimizer state) that folo-core has write-locked,
  checked to be contiguous and distinct. The trait has further optional hooks with the same `NotSupported` default
  (`conv2d_act_at`, `conv2d_act_sites`, the multi-tensor `*_many` updates, `unscale_check`, `set_step_guard`,
  `detect_class_sites`, `detect_decode_sites`); each one's doc comment in `backend.rs` names the `Tensor` method (or
  type) that reaches it.
- **Synchronisation.** `synchronize` (default no-op) waits for queued device work and reports asynchronous errors;
  `folo_core::backend::synchronize(device)` calls it. Backends may queue work lazily: only `download` (`to_vec`,
  `to_device(Cpu)`, `item`) and `synchronize` have to wait.
- **Testing.** The CPU backend is the reference: each GPU kernel should be tested against it within a stated
  tolerance.
