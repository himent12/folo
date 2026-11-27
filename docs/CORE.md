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
cpu/                             the CPU backend: gemm/, conv/ (direct, depthwise, Winograd F(2x2)/F(4x4)),
                                 elementwise/ (AVX2 & Wasm SIMD128 transcendentals, 8x8 transpose),
                                 softmax/ (cache-tiled online softmax & fused backward), reduce, cast (F16C), index,
                                 pool/ (pooling and 2x2 upsampling), q8/ (int8 convs),
                                 pool_alloc/ (two-level TLS + shared buffer cache)
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

Kernels fan out through `cpu::par_chunks` / `cpu::par_for`, which take the per-task work as a `&dyn Fn`, rather than
through rayon's iterators with their own closure: rayon's splitting and joining is then compiled once instead of once
per op and kernel (the elementwise maps alone are instantiated for every unary and binary op), which halved this
crate's code to optimise and its release build time, for one indirect call per task of thousands of elements. The
GEMM driver takes its operands as `&dyn Operand` for the same reason. Keep new kernels on these helpers.

### GEMM
- A BLIS-style GEMM (`cpu::gemm`, packed panels, runtime-selected AVX2/FMA micro-kernels with 4× K-unrolling in exact
  sequential K order and specialized `2×16`, `4×16`, and `6×16` register tiles across both packed `kernel.rs` and
  row-pointer `rows.rs`/`arows.rs`, tile- or K-split parallelism) backs `matmul` and conv2d.
- Each GEMM task packs the B panels of its own columns into an L2-sized buffer one `KC` block at a time, reading
  row-contiguous operands row by row so the prefetcher can follow (column tiles first, rows only when there are too
  few columns, about three tasks per thread and none under `PAR_FLOPS` = `2²¹` ≈ 2.1 MFLOPs / 1.05M MACs). `Strided::pack`
  takes a contiguous copy path when `so == 1`; for transposed operands (`sk == 1`, `so != 1`, e.g. `X @ Wᵀ` in `Linear`),
  it packs through `transpose_f32_2d_serial` (8×8 AVX2 blocks when AVX2 is available and both panel dimensions are at
  least 8, with an L1-blocked scalar fallback).
- A row-major A whose packed form fits one K super-block (16 MiB: every conv weight) is not packed at all: the
  micro-kernels broadcast its rows through row pointers, the same FMAs in the same order (bit for bit the packed
  path), so a conv forward has no serial packing pass.
- Register epilogues (`RegEpi`) fuse bias plus identity, SiLU, ReLU, LeakyReLU, or Hardswish directly onto the
  micro-kernel accumulators before storing; other activations run as a pass over each finished row segment in cache.
- A batched `matmul` splits the pool between its matrices.

### conv2d forward
- conv2d (any kernel, stride, padding or dilation, not depthwise) never builds an im2col matrix: the input is copied
  once, zero-padded and split into its stride phases (`cpu/conv/shifted.rs`), so every `(channel, ky, kx)` row of B
  is a contiguous run of that copy, read in place by a variant of the micro-kernel through a row-pointer table (a few
  dropped columns per output row; planes are spaced an odd number of cache lines apart so the rows do not alias in
  L1; stride-2 rows are padded once and both column phases dealt out in one pass).
- Pointwise convs pack B as a plain copy.
- Depthwise convs have a direct kernel (`cpu/conv/depthwise.rs`; at stride 1, the AVX2/FMA path pads the plane once,
  accumulates every output vector's taps in registers, and fuses bias / SiLU in registers — applying other activations
  such as ReLU in a post-store pass — while the Wasm SIMD128 `f32x4 × 4` path for overwriting stride-1 convs with
  finite weights and padded scratch ≤ 16 MiB fuses bias, SiLU, and ReLU in registers before storing).
- So do convs with at most 4 input channels (one group, not pointwise, `c·kh·kw <= 256`: the 3-channel stem,
  `cpu/conv/small_c.rs`): each worker reuses a thread-local scratch buffer across row bands, gathering the `c·kh·kw`
  taps into L1 — for `3×3 s2 p1`, three rotating `(even, odd)` AVX2 de-interleaved row slots reuse `ky = 2 -> ky = 0`
  across consecutive rows and a 4-output-channel AVX2+FMA kernel (`small_c_s2k3p1_row4_avx2`) shares each loaded input
  vector across 4 output channels — and computes every output channel's FMAs in the GEMM's order with bias and SiLU
  fused in registers before storing directly to the output, bit for bit the GEMM result, without the per-channel
  phase copy.
- Bias and activation run as an epilogue on each finished output (`Tensor::conv2d_act`, used by fused `ConvBnAct`
  blocks and by eval-mode blocks under `no_grad`): bias plus identity, SiLU, ReLU, LeakyReLU, or Hardswish on the
  micro-kernel's accumulators before they are stored, other activations as a pass over each finished block. SiLU there
  is a division-free, bit-exact scalar and
  8-lane AVX2 `x · sigmoid(x)` (Cody-Waite reduction with `1.5·2^23` rounding, degree-5 minimax `exp(r)`, and
  bit-hack seed + 4-FMA factored Householder-6 reciprocal, zero `vdivps`/`vroundps`) with relative error below
  `cpu::SILU_MAX_REL_ERR` (5e-7, tested).
- A batch of several images splits the pool between them: only as many images run at once (`lanes`) as their
  shifted-rows copies or unfolded gradients fit in 16 MiB, each lane taking its images one after another, and each
  image's GEMMs get `threads / lanes` threads (work stealing spreads the tiles).

### Winograd F(2x2, 3x3)
- The fused entry points (`conv2d_act`, `conv2d_act_cat`: nothing is recorded) run dense 3x3 stride-1 convs with at
  least 16 input and 8 output channels through Winograd F(4x4, 3x3) (below) or Winograd F(2x2, 3x3)
  (`cpu/conv/winograd.rs`, for remaining shapes with `C + O <= 4096` and either `⌈OH/2⌉ · ⌈OW/2⌉ >= 256` or
  `C · O >= 4096`, such as 20x20 feature maps): 16 multiplies per 2x2 output tile and channel pair instead of 36.
- The weights are transformed once per weight tensor (in `f64`, rounded once to `f32`) and kept with its storage until
  the storage is next written (`View::derived`: every write lock clears it; a view whose `storage` field was pointed
  elsewhere neither uses nor fills it). `Tensor::derived_f32` offers the same store outside the backend (folo-vision's
  int8 plan keeps its packed convs there): unlike a check of the storage version, which a write bumps before it waits
  for the lock, a value made under the read lock never outlives the data it was made from.
- Input (`Bᵀ d B`) and output (`Aᵀ M A`) transforms have explicit 8-tile AVX2/FMA (`cpu/conv/winograd.rs`) and 4-tile
  Wasm SIMD128 (`cpu/conv/winograd_wasm.rs`) paths with register-fused bias + SiLU/ReLU epilogue, bit-exact with the
  scalar transforms.
- Tiles are cut into L2-sized blocks (transforming inputs, running 16 small GEMMs through the row-pointer micro-kernel,
  and transforming outputs while in L2) and switch to a two-pass split (`C` then `O`, keeping per-task `M` in L1/L2
  scratch) when an image has fewer than half as many blocks as threads.
- Not bit for bit the direct conv: an F(2x2) output is within `(C + 8) · 2^-23 · 9 · Σ_c D_c · G_c` of the exact sum
  (`D_c`, `G_c`: the sums of |inputs| over its tile's 4x4 input patch and of |weights| over its 3x3 kernel in channel
  c; derived in the module doc and tested); COCO val AP of folo-n-coco changed by less than 1e-6.
- The same bits for any thread count or batch. The autograd conv (`conv2d_with`) keeps the direct conv.

### Winograd F(4x4, 3x3)
- On larger dense 3x3 stride-1 inference convs (`OH, OW >= 24`, `C >= 16`, `O >= 8`, `C + O <= 2048`, and either at
  least 56 4x4 tiles with `C · O >= 2048` or at least 200 4x4 tiles), `cpu/conv/winograd4.rs` runs Winograd
  F(4x4, 3x3) before F(2x2): 36 multiplies per 4x4 output tile and channel pair instead of 144 (4.0x FLOP reduction vs
  direct, 1.78x vs F(2x2)).
- Uses well-conditioned interpolation points `(0, ±1/2, ±1, ∞)` rather than `(0, ±1, ±2, ∞)` so every multiplier in
  the factored `Bᵀ` and `Aᵀ` butterflies is an exact power of two (`±1, ±1/2, ±1/4, ±1/8`; unfactored `Bᵀ` has
  `-5/4` entries), with explicit 8-tile AVX2/FMA + 4-tile SSE/FMA (`winograd4/transform.rs`) and 4-tile Wasm SIMD128
  (`winograd4/wasm.rs`) transforms bit-exact with their scalar paths, register-fused bias + SiLU/ReLU direct stores,
  and the same two-pass L1/L2 split and `View::derived` weight cache as F(2x2).
- Error is bounded by `(C + 16) · 2^-23 · 140 · Σ_c D_c · G_c` (`D_c` over the 6x6 patch; tested including 512-channel
  all-ones), and results are bitwise deterministic across thread counts and batches.

### Convs at some output sites
- `Tensor::conv2d_act_at` (`Backend::conv2d_act_at`) is the inference conv needed only at some output sites
  (flat `n · OH·OW + oy · OW + ox`, a `U32` vector): the result has the full output shape, every site depends strictly
  on its kernel's own receptive field in the input (never F(4x4)), and every other element is unspecified.
- `Tensor::conv2d_act_at_tiles` (`Backend::conv2d_act_at_tiles`) is the opt-in tile-patched variant for callers (such
  as candidate-first prediction) that populate each requested 4x4 tile's full 6x6 input patch and want bit-for-bit
  parity with dense `conv2d_act` including Winograd F(4x4, 3x3).
- On the CPU a Winograd conv runs the block code over the list of 2x2 or 4x4 tiles that hold a site
  (`winograd_forward_at` for `conv2d_act_at`; `winograd4_forward_at` also permitted for `conv2d_act_at_tiles`: a
  tile's arithmetic and its GEMM sums do not depend on which tiles share its block) and a pointwise conv (one group,
  fewer than 2048 input channels) runs the dense GEMM over the sites' columns gathered into planes of their own
  (`cpu/conv/sites.rs`); any other conv runs densely, as does a backend without the kernel.
- Patch isolation: in `conv2d_act_at` (Winograd F(2x2, 3x3), pointwise, or direct) each site depends strictly on its
  own 3x3 (or 1x1) receptive field — garbage, `1e20`, or `NaN` anywhere else never affects the site. In
  `conv2d_act_at_tiles` with Winograd F(4x4, 3x3), the transform combines inputs across the 4x4 tile's 6x6 input patch
  (`4·ty − pad_h .. 4·ty − pad_h + 6` by `4·tx − pad_w .. 4·tx − pad_w + 6`), so inputs must be valid across every
  requested tile's 6x6 patch (candidate-first prediction populates that patch in `conv0` whenever `conv1` may take
  F(4x4)). For image-contiguous inputs, the sparse kernels never read outside those tiles/columns (tested with NaN);
  non-contiguous views are materialized before dispatch, though elements outside those patches still cannot affect the
  requested sites.
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
  with the flipped, channel-swapped kernel (no col2im); the rest use an unfolded gradient and `col2im` with hoisted
  valid-column bounds and 16-wide AVX2 row accumulation (`add_row_avx2`, bit-exact with scalar `+`). Depthwise convs
  have their own backward kernels (`cpu/conv/depthwise.rs`) with non-FMA bit-exact AVX2 row updates
  (`mul_add_row_avx2`) and 8-lane dot products (`dot_avx2`, plus 8-lane Wasm SIMD128 `dot_wasm`).
- The weight gradient of a non-pointwise stride-1 conv, or of a pointwise one with `C/groups >= 64`, is
  `dWᵀ = X_unfolded · gyᵀ` with the input's shifted rows as A, read in place by a 6x16 micro-kernel through six row
  pointers (`gemm_arows`), and the output gradient transposed once per image into packed B panels (zeros in the
  dropped columns); K is cut into parts and the rows into tiles by shape alone, partials added in order. Other convs
  pack the unfolded input as they go (`cpu/conv/backward_weight.rs`).
- In a batch, the weight gradient makes its images' operands in waves of `lanes` images (above), each wave tiling its
  rows over the K parts it touches.

### Pooling and upsampling
- **Max pooling** (`cpu/pool/mod.rs`) is separable (row pass, then column pass; same first-maximum argmax as a direct
  scan). When nothing is recorded (`no_grad`, or an input that does not require grad) it makes no index tensor
  (`Backend::max_pool2d_values`, on the CPU a select over `-inf`-padded rows, the same bits; a window far wider than
  the plane takes the clipped indexed scan instead).
- **Average pooling** (`cpu/pool/mod.rs`) uses separable AVX2 row/column accumulation for contiguous `NCHW` planes.
- **Nearest upsampling** (`cpu/pool/upsample.rs`) includes specialized AVX2 `2×2` forward and backward fast paths
  (forward writes both output rows per input row in one pass via `unpacklo/hi` + `permute2f128`, 8 → 16 floats per
  step).

### Elementwise, softmax, reductions, transpose and casts
- **AVX2 & Wasm SIMD128 transcendentals (`cpu/elementwise/simd.rs`, `simd_log_tanh.rs`, `wasm_simd.rs`):** standalone
  `Exp`, `Sigmoid`, `Silu`, `Log`, `Tanh`, `Softplus`, and their backward kernels use 8-lane non-FMA division-free AVX2
  implementations on `x86_64` and 4-lane non-FMA Wasm SIMD128 implementations (plus 4-lane binary ops) on
  `wasm32 +simd128` (except `log_bwd`, which runs exact IEEE-754 `g / x`), paired with bit-exact scalar twins for all
  non-NaN inputs (and preserving NaN payloads on single-operand ops). `ln_scalar` / AVX2 `ln8` (`simd_log_tanh.rs`) and
  Wasm SIMD128 `ln4` (`wasm_simd.rs`) evaluate the degree-10 `ln(1 + f)` polynomial (`LN_C2..LN_C10`) with a
  depth-reduced Estrin tree (`f4 = f2 * f2`, parallel degree-1 pairs `q0..q3`, then `r0`, `t3`, `r1`) instead of a
  serial Horner chain, reducing multiply-add dependency depth while preserving exact scalar/AVX2/Wasm-SIMD128 bit
  parity. Omitting FMA guarantees identical bits across
  contiguous/strided paths, slice boundaries, thread counts, and `avx2`-only callers such as `folo-nn`'s `BatchNorm2d`.
  The fused conv/GEMM epilogue (`silu8` / `silu_bias_inplace`) keeps its FMA formulation.
- **Unit-stride fast paths & 8×8 AVX2 transpose (`cpu/elementwise/mod.rs`, `transpose.rs`):** `strided_update` and
  `where3` take unit-stride fast paths when operands are contiguous along the inner axis, and 2-D `f32` transposes run
  through a cache-blocked (`32×32` L1 tile) 8×8 AVX2 kernel (`transpose_f32_2d`).
- **Cache-tiled online softmax and fused backward (`cpu/softmax/`):** contiguous last-dim (`inner == 1`) rows up to
  `4096` floats (`ROW_TILE`) stay in L1 (`log_softmax` accumulates `sum(fast_exp(v − m))` in registers; `softmax`
  fuses `exp` + sum), using 8-wide AVX2 `fast_exp8` and the 16-lane sum order of `sum_slice`; longer rows run an
  online `(m, s)` pass over `2048`-float L1 tiles (`ONLINE_TILE`) to avoid an extra DRAM pass. Non-last-dim
  (`inner > 1`) softmax tiles columns in stack-allocated `256`-float strips (`COL_TILE`, zero heap allocations).
  `Backend::softmax_backward` (`cpu/softmax/backward.rs`) is a fused single-allocation kernel eliminating 3
  intermediate tensors and 2 full-tensor memory passes compared to the composite autograd path.
- **AVX2 max/min reductions & bounded-k `topk` (`cpu/reduce.rs`):** `Max` and `Min` reductions run 8-wide AVX2 loops
  preserving exact scalar NaN and signed-zero semantics. `topk` uses a zero-allocation linear scan for `k == 1`, a
  bounded `k`-element binary heap when `1 < k <= 1024` and `4·k < d`, and `select_nth_unstable_by` otherwise.
- **F16C casts (`cpu/cast.rs`):** CPU `f32 ↔ f16` conversions borrow contiguous inputs without copying, parallelise
  above `PAR_MIN`, and use hardware `F16C` (`VCVTPS2PH` / `VCVTPH2PS`, round-to-nearest-even) on `x86_64` CPUs with
  `avx + f16c`.

### BatchNorm statistics
folo-nn's train-mode BatchNorm on the CPU (`crates/folo-nn/src/layers/batchnorm.rs`, [NN.md](NN.md)) is built on
these kernels:
- It takes its statistics in one read of x, in `f64` and in blocks of 256 floats per plane: the 8-lane plane sums of
  the old two-pass code (so the mean is bit for bit its mean), and each block's sum of squared deviations from its own
  mean, merged per plane and per channel with Chan's formula (no cancellation).
- It applies SiLU / ReLU in its own passes (forward and backward), bit for bit BatchNorm then `Tensor::silu` / `relu`
  (`cpu::silu` / `cpu::silu_backward` are the shared formulas).
- Its per-plane loops run as AVX2 when the CPU has it (the same loops compiled under `target_feature`, bit for bit).

### int8 convs
`cpu::q8` holds the kernels of int8 inference (docs/MODEL.md §9.6 has the arithmetic; folo-vision's
`model::quant` drives them). They work on raw buffers, not `Tensor`s:
- **Layout.** Activations are u8 codes in C4 words, `[N][C/4][H][W]` `u32` (byte `t` of a word is channel
  `4g + t`), each with an affine grid (`Grid { scale, zero }`). A channel slice on a multiple of 4 is a view (an
  offset), so split and concatenation copy nothing. Buffers come from the buffer cache (`q8::alloc`, `q8::recycle`)
  with 64 words of slack after the data.
- **Conv.** `q8::conv` takes the input parts of a concatenation (each with its zero point, which must be the one
  the weights were packed with), an optional residual and a `QConv` (s8 weights packed `[O/4][C/4·k·k][4]`, four
  to a word, the zero-point corrections, per-channel scales and biases; `C·k·k` at most `MAX_REDUCTION` = 2¹⁵ so
  the i32 sums cannot overflow, stride at most `MAX_STRIDE` = 64, padding at most `MAX_PAD` = 4096:
  `QConv::supports` names these limits, `q8::out_size` says whether a kernel fits an input, and `q8::conv_size`
  also whether the input, padded copy and output each fit one allocation (`isize::MAX` bytes), on 32-bit
  targets too) and writes u8 C4 codes or `f32` NCHW. A micro-kernel computes 4 output channels × 24 output
  pixels in i32: AVX-VNNI `vpdpbusd` (u8 × s8, 4 products per lane, no saturation) when the CPU has
  `avxvnni` (and AVX2, FMA), AVX-512 VNNI VL (`avx512vnni` + `avx512vl`) where supported, exact AVX2 `vpmaddwd`, Wasm SIMD128, detected at run time, else portable scalar code with the same sums. On x86-64 with AVX2 + FMA, `fused.rs` + `fused_epi.rs` execute each task's L1-blocked `(c_tile, g, chunk)` tile loop and split-pair zero-spill epilogue inside a single `#[target_feature]` function (zero per-tile `vzeroupper`, zero indirect calls, direct strided single- and multi-part 1×1 K-loops that skip `image_rows` completely, and contiguous slice copies for non-contiguous cross-row chunks). Single-part `3×3, pad = 1` convs at stride 2 (except huge-weight / small-input shapes better served by channel-first splitting) and large-input stride 1 (`> 384 KiB`, `oh >= 8`) run through `fused_3x3.rs` in a single rayon pass over L1-resident 3-slot rotating row strips (reusing per-thread TLS scratch, avoiding global padded buffers and a second barrier). Remaining non-pointwise convs read a padded, phase-split copy of each image's rows (`rows.rs`: `image_rows`, padding
  with the zero point, parallelised across channel groups and row bands for few-group high-resolution layers, and rounding each channel group's phase block up to a 64-byte / 16-`u32` multiple to reduce
  cache-line false sharing across parallel `fill` workers) so each output pixel's taps are contiguous columns. The epilogue (`(acc − corr) · scale + bias` as one
  FMA, SiLU, the residual as one FMA, requantise) is the same `f32` operations in the scalar and AVX2 code, so
  results are bit-identical across kernels and thread counts; the AVX2 form runs on any CPU with AVX2 and FMA
  (`FOLO_Q8_SCALAR=1` turns off both vector paths). Its SiLU (`q8::silu_q8`) is a division-free, bit-exact scalar and
  AVX2 `x / (1 + e^{−x})` using base-2 range reduction (clamping `x >= −85.0` so `−x·log₂(e) < 124` without an extra upper clamp, `1.5·2^23 + 127` magic rounding, single-shift `2^n`), a
  degree-5 Estrin-form minimax `2^f` polynomial evaluated in register-spill-free 2-vector pairs (`silu_q8_avx2_x2`), and a bit-hack seed + 4-FMA factored Householder-6 reciprocal (`< 2.5e-7` hybrid
  error on `[-85, 85]`); the specialised AVX2 epilogue writes contiguous 24-pixel tiles directly to the C4 output
  (for both 1×1 and within-row `k×k` chunks), packing four channels' codes with saturating `packs` / `packus` and
  one byte shuffle. Work is split into (image, pixel block, channel block) tasks: capped at `3 · threads`, none
  under `2¹⁸` multiply-adds (and small convs below the serial floor run without waking the pool), with a heuristic that rounds multi-task pixel blocks to even chunk counts (192-byte / 3-cache-line multiples on contiguous planes) to reduce output-plane false sharing. An image whose
  input fits in 384 KiB (a core's L2) is split by channel blocks first, so tasks write whole output planes; pixel
  blocks of a few 24-column chunks would share cache lines with their neighbours on every plane. Bigger inputs
  are split by pixel blocks first, so each task reads its own part once. `q8::stem_u8` runs 3-channel `u8` `[R, G, B, 0]`
  stems directly in L1-resident rotating row strips with AVX2 even/odd de-interleaving and a fused per-row x86-64 tile pass.
- **Others.** 2× unrolled AVX2-vectorised `quantize` / `dequantize` (16 pixels / full 64-byte cache line per iteration, 4 channels per C4 word) between `f32` NCHW and C4
  codes, separable AVX2 max pooling on codes (`max` of bytes; exact since the grid is monotonic, reusing a thread-local
  scratch buffer for small serial planes below `MIN_POOLED`, a single pooled `ScratchWords` for larger serial planes,
  and per-worker pooled `ScratchWords` via `for_each_init` on the parallel path), and AVX2 nearest 2× upsampling, all
  with serial fast paths below `PAR_MIN`.

Tests (`cpu/q8/tests.rs`) check convs against an i64 reference at 1, 3 and 8 threads, with the real task size and
with one-MAC tasks (every split): 3×3 stride 1 and 2, 1×1, several input parts on different grids, residuals, `f32`
outputs, two images, and offset views with a longer image stride. They also check the task plan's bounds, the
refusals (zero points, long reductions, wrapping views, kernels larger than the input), `supports`, `out_size` and
`conv_size`,
VNNI against the scalar kernel, the SiLU against SiLU in `f64` and its scalar form against its AVX2 form bit for
bit, the epilogue's exact ties (to even, on both paths), `quantize_weights`' known answers (ties away from zero,
±127 at the row maximum), quantise round trips and the pools. `cargo bench -p folo-core --bench
q8_infer` times each folo-n conv in int8 against the fused `f32` `conv2d_act`.

### The buffer cache
`f32` buffers from 16 KiB (`MIN_POOLED = 4096` floats) are recycled through a two-level cache (`cpu/pool_alloc/`;
`cpu::set_buffer_cache_limit`, default 1 GiB; `cpu::take_buffer` hands one to custom ops) so training and inference
loops do not pay page faults every step: a per-thread TLS magazine (guarded by an uncontended per-thread mutex so
`clear_buffer_cache` and limit changes can drain live slots immediately; at most 8 buffers, each ≤ 4 MiB, ≤ 16 MiB
per thread) sits in front of the shared size-keyed pool to avoid global-pool contention, coordinated by a shared atomic
total-bytes budget and clear epoch so `set_buffer_cache_limit` and pressure drains bound both levels together. When the
shared pool is full, lengths
unused for two limits of traffic, and for twice the
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

Release and debug builds are not checked against each other for training. Rust never contracts or reorders float
operations at any optimisation level, and one short CPU run (`folo train`, folo-n, 2 epochs at 128 px, seed 3, on
Windows) wrote the same losses, validation metrics and weights with both builds (timestamps and timings, such as
images per second, ETA and epoch time, aside). What can still make a build differ is a libm call (`exp`, `ln`, `tanh`,
`sin`, ...): where inlining lets the compiler see its argument, it may fold the call into the correctly rounded value,
elsewhere the platform's libm computes it, and the two can differ in the last bit (the synthetic generator drew its
squares this way until its sine and cosine stopped coming from libm). The platform's libm also differs between
operating systems, so two systems need not agree. The synthetic datasets ([DATA.md](DATA.md#synthetic-shapes)) and
benchmark variants ([BENCH.md](BENCH.md)) take no approximation from libm and are the same everywhere.

## Autograd
- Define-by-run, reverse mode. An op whose input requires grad records a node (`SmallVec` storage for inputs, versions,
  and saved cells, plus a backward closure built from ordinary tensor ops) on the result.
- `loss.backward()` builds a dense slot-indexed post-order graph (`topo_order`, slots `0..nodes.len()`), accumulates
  intermediate gradients in a flat `Vec<Option<Tensor>>` without `HashMap` lookups on the reverse walk, and returns
  `Grads` (keyed by tensor id, leaves only); graphs are retained, so backward can run twice. Graph teardown is
  iterative (deep graphs do not overflow the stack).
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
  (`conv2d_act_at`, `conv2d_act_sites`, the multi-tensor `*_many` updates, `unscale_check`, `detect_class_sites`,
  `detect_decode_sites`); each one's doc comment in `backend.rs` names the `Tensor` method that reaches it.
- **Synchronisation.** `synchronize` (default no-op) waits for queued device work and reports asynchronous errors;
  `folo_core::backend::synchronize(device)` calls it. Backends may queue work lazily: only `download` (`to_vec`,
  `to_device(Cpu)`, `item`) and `synchronize` have to wait.
- **Testing.** The CPU backend is the reference: each GPU kernel should be tested against it within a stated
  tolerance.
