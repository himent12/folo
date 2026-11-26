# folo-gpu

FOLO's GPU layer on [wgpu](https://wgpu.rs): Vulkan, DX12 and Metal natively, WebGPU in the browser. It owns the
device (`GpuContext`: adapter, queue, buffer pool, pipeline cache, command batching), the WGSL compute kernels with
their Rust entry points, and `GpuBackend`, which implements folo-core's `Backend` so folo-core tensors (and autograd)
run on the GPU. Its one workspace dependency is folo-core (folo-nn and folo-test-support are dev-dependencies, for
tests). folo-vision, folo-server, folo-cli and folo-wasm depend on it, folo-nn as a dev-dependency for its GPU tests,
and FOLO Desktop (`apps/desktop`) for local inference.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | Crate doc, module map and re-exports. |
| `src/context/` | `GpuContext`: `device.rs` (adapter selection, device creation), `memory.rs` (allocation, uploads), `batch.rs` (the open command batch; `kernel` and `dispatch`), `binds.rs` (bind-group cache), `readback.rs`, `wait.rs` (`sync`, device loss), `plan.rs` (frozen dispatch lists, forward plans), `precompile.rs`, `trace.rs`. |
| `src/pool.rs` | The size-class buffer pool (`GpuBuffer`, `PoolStats`). |
| `src/pipeline.rs` | `KernelDef`, `Spec` (specialisation constants), `Params` (the per-dispatch uniform block), the WGSL `PRELUDE`, `render` and pipeline compilation. |
| `src/pipeline_cache.rs` | Compiled pipelines kept on disk between processes. |
| `src/kernel_list.rs` | The kernel lists `GpuContext::precompile` records and compiles on worker threads in the next run. |
| `src/kernels/` | One Rust entry point per kernel, on raw `wgpu::Buffer`s: validation, specialisation, dispatch. `kernels/mod.rs` lists the modules and which front end uses each. |
| `src/shaders/` | The hand-written WGSL files (below). Many kernels have no file: their source is generated in `src/kernels/`. |
| `src/backend/` | `GpuBackend`: `ops.rs` (every `Backend` method), `encode.rs` (calls several ops share), `norm.rs` (fused `LayerNorm`, `RMSNorm`, and `softmax_backward` hooks), `sites.rs` (candidate-first detection hooks), `convert.rs`, `storage.rs`, `plan.rs` (`ForwardPlans`). |
| `src/tensor.rs` | `GpuTensor`: a standalone buffer + shape + dtype handle for kernel-level work. |
| `src/detect.rs` | The task-aligned assigner and the detection loss on folo-core GPU tensors. |
| `src/distill.rs` | The fused distillation term (class and DFL KL against a teacher, docs/MODEL.md §8.9) on folo-core GPU tensors. |
| `src/profile.rs` | Per-kernel GPU timings (`FOLO_GPU_PROFILE=1`). |
| `src/dtype.rs`, `src/error.rs` | Buffer element types; `GpuError`. |
| `src/tests/` | The GPU tests (unit tests, compiled with `cargo test`; `kernel_list.rs`, `pool.rs` and `pipeline_cache.rs` also have unit tests of their own): `backend/` (the backend against folo-core's CPU backend, by area) and kernel-level files (`gemm.rs`, `conv.rs`, `coop.rs`, `elementwise.rs`, ...) plus runtime tests (`runtime.rs`, `pool.rs`, `binds.rs`, `plan.rs`, `precompile.rs`, `pipeline_cache.rs`). |
| `benches/` | `gemm.rs`, `conv.rs`, `conv_model.rs`, `elementwise.rs`, `backend.rs`, `pooling.rs`, `norm_softmax.rs`. |

There is no `tests/` folder at the crate root and no `build.rs`.

### How shader source is assembled

A kernel is a `KernelDef { name, storage, source }` in a `src/kernels/` module, listed in that module's `DEFS`.
Its `source` is either `Source::Static(include_str!("../shaders/<file>.wgsl"))` or `Source::Generated(fn)`, a Rust
function that returns WGSL for a `Spec`. Generators either concatenate files with shared snippets
(`unary_fn.wgsl`, `bn_common.wgsl`, `bn_welford.wgsl`, `detect_common.wgsl` — shared across `tal_*`, `det_*`, `dfl_stats.wgsl`, `detect.wgsl`, `sites_class.wgsl`, `sites_decode.wgsl`, and `kd_*.wgsl`, `kd_common.wgsl`, the `E` / `V4` / `OUT` aliases of
`kernels::vec_prelude`) or write the whole source in Rust: the GEMM and implicit-GEMM convs (`kernels/gemm/`,
`kernels/conv/`), tensor cores (`kernels/coop/shader.rs`), `kernels/ew.rs`, `kernels/strided/`,
`kernels/pool_grad.rs`, Muon and the multi-tensor optimizer kernels. There is no include mechanism inside WGSL.
`pipeline::render` then puts `Spec::header` (`enable f16;` and `enable wgpu_cooperative_matrix;` when asked,
`alias T = f32|f16`, one `const` per specialisation constant) and `PRELUDE` (the uniform block and `pu` / `pi` /
`pf` / `linear_id`) in front. `Spec::header` omits `enable subgroups;` because Naga 30 rejects that directive; native
subgroup built-ins are enabled via `wgpu::Features::SUBGROUP` alone, while `Spec::subgroups(true)` selects the subgroup
WGSL variant and tags the pipeline cache key. When `kernels::use_subgroups(ctx)` is true (native adapters with
`SUBGROUP` and subgroup width `(32, 32)`; always `false` on `wasm32` / browser WebGPU), reductions (`reduce_rows`,
`softmax`), BatchNorm (`bn_stats`, `bn_finalize`, `bn_bwd_reduce`, `bn_bwd_finalize`), depthwise weight gradients
(`conv2d_depthwise_grad`), detection loss reductions (`det_bce_fwd`, `det_combine`), and gradient-norm partials
(`norm_partials_many`) dispatch 1-barrier 32-wide subgroup variants (and `topk` selection passes for `d > 512`
dispatch a 2-barrier 32-wide subgroup reduction per pass, skipping the second barrier on the final pass) with a
portable shared-memory tree fallback otherwise, and `unscale_check*` coalesces overflow flags via `subgroupAny` with
no workgroup barrier (direct per-element `atomicOr` otherwise). Every file in `src/shaders/` opens with a
comment saying what it computes, in most files followed by its `Spec` constants and its parameter words; the bindings
come next.

### From an op to its files

There is no strict naming convention, but the chain is regular. A folo-core op reaches `GpuBackend`'s method of the
same name in `src/backend/ops.rs`, which calls an entry point in the `src/kernels/` module for that kind of work
(not one module per op). That entry point dispatches one or more `KernelDef`s; a kernel's `name` is usually the stem
of its shader file where it has one (`softmax` and `softmax.wgsl`, `tal_topk` and `tal_topk.wgsl`), with a few
exceptions: `max_pool2d` (`maxpool.wgsl`), `conv2d_depthwise` (`depthwise.wgsl`), and the `det_kd_*` distillation
kernels (`kd_*.wgsl`). Tests are grouped by area, not by op.

| Op (folo-core) | `GpuBackend` method | Kernel entry point | WGSL | Test against the CPU |
|---|---|---|---|---|
| unary, binary, comparisons, `where` | `unary`, `unary_backward`, `binary`, `where_cond` (through `GpuBackend::ew` in `encode.rs`) | `kernels::ew::elementwise` | generated in `ew.rs` | `src/tests/backend/elementwise.rs` |
| sum / mean / max / min, argmax / argmin | `reduce` | `kernels::reduce::{reduce_split, arg_reduce}` (in `reduce/mod.rs`, `reduce/arg.rs`: ceiling-division split-K for `Sum`/`Max`/`Min` from `GpuBackend`, plus `Mean` in `reduce_split` directly; `vec4` loads/stores on rows and columns, 32-wide subgroup reductions on rows; workgroup-parallel `vec4` + subgroup + split-K `arg_reduce_rows` for contiguous rows) | `reduce_rows.wgsl`, `reduce_cols.wgsl`, `arg_reduce.wgsl`, `arg_reduce_rows.wgsl` | `src/tests/backend/reduce.rs`, `src/tests/reduce.rs` |
| softmax, log-softmax | `softmax`, `softmax_backward` | `kernels::reduce::softmax_dim` (online row softmax when `inner == 1`, 1 read up to 1,024 elems / 2-pass online above; `softmax_cols` otherwise, register-cached for `D ≤ 32` and 2-pass online above); `kernels::softmax_bwd::softmax_backward_dim` (fused `vec4` register-cached rows with 1-barrier subgroups / portable tree fallback, and `vec4` coalesced columns) | `softmax.wgsl`, `softmax_cols.wgsl`; generated in `softmax_bwd.rs` | `src/tests/backend/reduce.rs`, `src/tests/backend/norm.rs` |
| LayerNorm, RMSNorm | `layer_norm`, `layer_norm_backward`, `rms_norm`, `rms_norm_backward` | `kernels::layer_norm::*` (fused `vec4` register-cached rows with 1-barrier 32-wide subgroups / portable tree fallback, extreme-scale normalization guard, and split-K `vec4` column parameter gradients) | `layer_norm_fwd.wgsl`, `layer_norm_bwd_dx.wgsl`; generated in `layer_norm.rs` | `src/tests/backend/norm.rs` |
| top-k | `topk` | `kernels::reduce::topk` (workgroup-parallel bitonic sort for `d ≤ 512`; `k`-pass selection with 32-wide subgroup reductions and a portable shared-memory tree fallback above) | `topk.wgsl` | `src/tests/backend/reduce.rs`, `src/tests/backend/index.rs` |
| matmul | `matmul` | `kernels::gemm::gemm` | generated in `gemm/shader.rs` | `src/tests/backend/matmul.rs`, `src/tests/gemm.rs` |
| conv2d and its gradients | `conv2d`, `conv2d_act`, `conv2d_backward_input`, `conv2d_backward_weight(_as)` | `kernels::conv::{conv2d, conv2d_input, conv2d_backward_input, conv2d_backward_weight_as}` (16×16 shared-memory tiled + `vec4` depthwise forward for stride-1 `2..=7` kernels, plus 16×16 tiled stride-1, phase-unrolled stride-2, and `vec4`-store `VX4` depthwise input-gradient paths) | generated (`gemm/shader.rs`, `conv/`, `coop/shader.rs`); `depthwise.wgsl` | `src/tests/backend/conv.rs`, `src/tests/conv.rs`, `src/tests/coop_conv/` |
| 1x1 conv over a concatenation | `conv2d_act_cat` | `kernels::conv::conv2d_cat` (`conv/forward.rs`, calling `gemm_cat.rs`) | generated | `src/tests/backend/conv_cat.rs` |
| max / avg / adaptive pooling and gradients | `max_pool2d`, `max_pool2d_values` (inference, skips `i64` indices within `maxpool` limits), `avg_pool2d`, `adaptive_avg_pool2d`, their `_backward` | `kernels::pooling::max_pool2d` (values only; 16×16 separable shared-memory tiled stride-1 `2..=7` and `vec4` 2×2 stride-2), `kernels::pool_grad::*` (16×16 tiled stride-1 max fwd/bwd, `vec4` 2×2 stride-2 avg fwd/bwd, GAP 1×1 via `reduce_split` + `adaptive_gap_bwd`) | `maxpool.wgsl`; generated in `pooling.rs`, `pool_grad.rs` | `src/tests/backend/pooling.rs` |
| nearest upsampling | `upsample_nearest2d`, `upsample_nearest2d_backward` | `kernels::layout::upsample_nearest2d` (`upsample_2x_vec4` fast path for 2×2 even widths), `kernels::pool_grad::upsample_nearest2d_backward` (`pool_2x2_s2_vec4` fast path for 2×2 even widths) | `upsample.wgsl`; generated | `src/tests/backend/pooling.rs` |
| index ops | `index_select`, `index_add`, `gather`, `scatter_add` | `kernels::strided::*` | generated in `strided/` | `src/tests/backend/index.rs` |
| copies, casts, fills | `copy_strided`, `copy_into`, `cast`, `full` | `kernels::strided::*` | generated in `strided/` | `src/tests/backend/transfer.rs` |
| train-mode BatchNorm + activation | `batch_norm_act`, `batch_norm_act_backward` | `kernels::norm::{batch_norm_forward, batch_norm_backward}` (in `norm/mod.rs`, `norm/fused.rs`: single-dispatch register-cached `bn_fused_fwd`/`bn_fused_bwd` up to 8,192 elems/ch with 1-barrier subgroups / portable fallback, plus 3-pass split-K for larger planes) | `bn_*.wgsl`; generated in `norm/fused.rs` | `src/tests/amp.rs`; folo-nn `tests/gpu/main.rs` |
| optimizer updates, clipping, loss-scaler unscale | `optimizer_step(_many)`, `norm_partials_many`, `scale_many`, `unscale_check(_many)` | `kernels::optim::*` (`vec4` fast paths on 4-aligned tensors, partitioning mixed-alignment multi-tensor batches so 4-aligned tensors stay vectorized) | `optim.wgsl`; generated `_many` kernels | `src/tests/reduce.rs`, `src/tests/backend/reduce.rs`; folo-nn `tests/gpu/main.rs` |
| Muon (MuSGD) | `muon_step_many` | `kernels::muon::muon_step_many` (Polar-FOLO + `Polar3-CS9` / `muon_s8`) | generated in `muon/shader.rs` | `src/tests/muon.rs`; folo-nn `tests/gpu/muon.rs` |
| letterbox, packed `u8` upload/normalize | `letterbox_rgb8` | `kernels::vision::{letterbox_rgb8, u8_to_float_scale, rgb8_hwc_to_chw_norm, upload_u8_scaled, upload_rgb8_hwc_to_chw_norm}` | `letterbox.wgsl`; generated in `vision.rs` | `src/tests/vision.rs`; folo-vision `tests/gpu.rs` |
| detection candidates | `detect_candidates` | `kernels::vision::detect_candidates` | `detect.wgsl` (after `detect_common.wgsl`, register-cached DFL for `R <= 32`) | `src/tests/sites.rs` |
| candidate-first detection | `conv2d_act_sites`, `detect_class_sites`, `detect_decode_sites` (in `backend/sites.rs`) | `kernels::conv::conv2d_sites`, `kernels::detect_sites::*` | `sites_*.wgsl` (`sites_class` / `sites_decode` after `detect_common.wgsl`); generated | `src/tests/sites.rs` |
| assigner, detection loss & DFL stats (not `Backend` methods) | `folo_gpu::detect::{assign, detection_loss, multi_label_candidates, dfl_stats}` (device-cached static anchors) | `kernels::detect::*` (32-wide subgroup `det_bce_fwd` & `det_combine`, fused VFL/QFL/S-AEIoU/CRPS-DFL, 128-bit `det_candidates`, single-pass `det_dfl_stats`) | `tal_*.wgsl`, `det_*.wgsl`, `dfl_stats.wgsl`, `detect_common.wgsl` | `src/tests/reduce.rs`; folo-vision `tests/gpu_loss/`, `tests/gpu_lazy_head.rs` |
| distillation term (not a `Backend` method) | `folo_gpu::distill::{distill_loss, distill_loss_with_context}` | `kernels::distill::*` | `kd_*.wgsl` (after `detect_common.wgsl`) | folo-vision `tests/gpu_distill.rs` |

`GpuTensor` (`src/tensor.rs`) has one kernel module of its own that the backend does not use: `kernels::elementwise`
(`unary.wgsl`, `binary.wgsl`, `bias_act.wgsl`, tested in `src/tests/elementwise.rs`). It shares `kernels::layout`,
`kernels::pooling` (`maxpool.wgsl`, tested in `src/tests/layout.rs`), `kernels::reduce`, `kernels::gemm` and
`kernels::conv` with the backend.

## Start here

1. `docs/GPU.md`, the first diagram: the three layers and who calls whom.
2. `KernelDef`, `Spec` and `render` in `src/pipeline.rs`: what a kernel is and how its WGSL is built.
3. A small module end to end: `src/kernels/layout.rs` (`upsample_nearest2d`) and `src/shaders/upsample.wgsl`.
4. `GpuContext::kernel` and `GpuContext::dispatch` in `src/context/batch.rs`: compile or fetch, then encode.
5. `impl Backend for GpuBackend` in `src/backend/ops.rs`: how each folo-core op maps onto the kernels.
6. `src/tests/backend/fixtures.rs`, `check`: how an op is compared with the CPU backend, forward and gradients.

## Common changes

| To do this | Touch these files |
|---|---|
| Add a kernel | The WGSL in `src/shaders/<name>.wgsl` (or a generator in Rust); a `KernelDef` and the entry point in `src/kernels/<area>.rs`, added to that module's `DEFS` (a new module also goes into `all_defs` in `src/kernels/mod.rs`); a bound in `src/kernels/bounds.rs` if the generated source grows with a constant; a test in `src/tests/<area>.rs`; the kernel table in `docs/GPU.md`. `docs/GPU.md`, Adding a kernel, has the steps. |
| Run a folo-core op on the GPU | The method in `src/backend/ops.rs` (or `src/backend/sites.rs`), calling a `src/kernels/` entry point; a `check` case in `src/tests/backend/<area>.rs`; the op mapping in `docs/GPU.md`, folo-core backend. |
| Support a new folo-core unary op | `unary_code` and `push_ew_op` in `src/kernels/ew.rs`; the op list in `src/tests/backend/elementwise.rs`. |

A kernel missing from its module's `DEFS` cannot be precompiled from a kernel list. Once such a kernel has run,
`GpuContext::kernels_without_registry` names it, and `src/tests/precompile.rs` and GPU training tests in folo-nn and
folo-vision assert that list is empty.

## Tests and benchmarks

```text
cargo test -p folo-gpu                        # every kernel and the backend against CPU references
cargo bench -p folo-gpu --bench gemm          # also: conv, conv_model, elementwise, backend, pooling
```

- The GPU tests ask for the shared context (`gpu!()`, or `dev!()` for the backend tests) and return early, printing a
  note, when there is no hardware adapter. CI runners have none, so there they pass without running: run them on a
  machine with a GPU.
- Software adapters (WARP, llvmpipe) are skipped too unless `FOLO_GPU_TEST_SOFTWARE=1`.
- `FOLO_GPU_BACKEND=dx12` (or `vulkan`, `metal`) and `FOLO_GPU_ADAPTER=<index | name>` pick what the tests run on.
- Nothing is `#[ignore]`d. Keep GPU test runs short (`AGENTS.md`): the GPU is often shared.
- Benches: `gemm` (GFLOPS, f32 and f16, transposed, batched), `conv` (conv2d at YOLO-typical shapes), `conv_model`
  (folo-n and folo-s forward convs at batch 16), `elementwise` (bandwidth of memory-bound kernels), `backend` (a
  YOLOv8n-shaped network on the GPU backend against the CPU backend), `pooling` (max/avg/adaptive pooling and 2×2
  upsample backward).

## Further reading

- [docs/GPU.md](../../docs/GPU.md): the device, buffer pool, batching, pipeline cache, every kernel, GEMM and
  tensor-core design, the folo-core backend.
- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): where this crate sits.
- [docs/WASM.md](../../docs/WASM.md): the browser build.
- [crates/folo-core](../folo-core/README.md): the `Backend` trait this crate implements and the CPU reference.
