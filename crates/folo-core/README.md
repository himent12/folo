# folo-core

The bottom of FOLO's stack: n-d tensors, dtypes, shapes and layouts, define-by-run reverse-mode autograd,
mixed precision, a seeded RNG, the `Backend` trait every device implements with its registry, and the built-in CPU
backend with its kernels. It depends on no other workspace crate. folo-gpu, folo-nn, folo-vision, folo-bench,
folo-server and folo-wasm depend on it (folo-cli as a dev-dependency), and FOLO Desktop (`apps/desktop`) uses it for
local inference. folo-gpu implements its `Backend` and registers itself at run time; folo-core never depends on a GPU
API.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | Crate doc with a runnable example, the module map and the re-exports. |
| `src/tensor.rs` | `Tensor`: construction, inspection, data access; `Tensor::track` / `track_saved` (attach a backward node to an op's result) and the public `Tensor::custom_op`. |
| `src/ops/` | Every tensor op, as `Tensor` methods: `elementwise.rs`, `reduce.rs`, `nn.rs` (matmul, conv, pooling, upsampling), `view.rs`, `copy.rs`, `inplace.rs`, `overload.rs` (`+ - * /`), `fused.rs`, `muon.rs`, `vision.rs`. `ops/mod.rs` lists them. |
| `src/autograd.rs` | Grad mode (`no_grad`, `is_grad_enabled`), `Grads`, and the backward walk over the recorded graph. |
| `src/autocast.rs` | Mixed precision for the forward pass (`autocast`, like `torch.autocast`). |
| `src/backend.rs` | The `Backend` trait (memory methods required; compute, fused and detector methods optional), `View`, the op descriptions (`UnaryOp`, `BinaryOp`, `ReduceOp`, `Conv2dParams`, `Pool2dParams`, ...) and the registry (`register_backend`, `backend`). |
| `src/cpu/` | `CpuBackend` and its kernels: `elementwise/` (`mod.rs` strided dispatch and scalar ops, `simd.rs` AVX2 + FMA `exp`/`sigmoid`/`silu` kernels, `simd_log_tanh.rs` AVX2 `log`/`tanh`/`softplus` kernels, `tests.rs` unit tests), `reduce.rs`, `index.rs`, `pool.rs`, `pool_alloc.rs` (buffer cache), `gemm/` (the GEMM), `conv/` (conv2d forward and gradients, Winograd F(2×2, 3×3) in `winograd.rs` and well-conditioned Winograd F(4×4, 3×3) in `winograd4.rs` + `winograd4/transform.rs`), `q8/` (int8 inference convs, direct `u8` stem in `stem.rs`, fast SiLU in `silu.rs`, per-channel C4 grids, decoupled concat, pools and quantisation on u8 C4 buffers, docs/CORE.md "int8 convs"; tests in `q8/tests.rs` and `tests/q8_fast.rs`). `cpu/mod.rs` lists them. |
| `src/storage.rs` | `Storage`, `CpuStorage`, `DeviceStorage`: the raw element buffers behind tensors. |
| `src/layout.rs` | `Layout` (shape, strides, offset) and the row iterator CPU kernels walk strided data with. |
| `src/shape.rs` | `Shape`, dimension arguments and broadcasting rules. |
| `src/dtype.rs`, `src/device.rs`, `src/error.rs` | `DType` and `Element`, `Device`, `Error` and `Result`. |
| `src/rng.rs` | `Rng`: xoshiro256++ seeded through SplitMix64, with a savable state. |
| `src/testing.rs` | `grad_check` / `grad_check_with`: finite-difference gradient checks. |
| `tests/grad.rs` | A finite-difference gradient check for every differentiable op. |
| `tests/reference.rs` | Forward results against naive reference implementations, at sizes that take the parallel paths. |
| `tests/determinism.rs` | Bitwise identical results from run to run and for every rayon thread count. |
| `tests/autograd.rs` | Autograd semantics (leaves, repeated backward, `no_grad`, version checks) and the backend registry contract. |
| `tests/custom_op.rs` | `Tensor::custom_op` used the way a downstream crate would. |
| `tests/in_pool.rs` | `cpu::in_pool`: grad mode and autocast travel with the closure. |
| `benches/` | `gemm.rs`, `conv.rs`, `conv_infer.rs`, `q8_infer.rs` (int8 against `f32` per folo-n conv), `conv_wgrad.rs`, `index.rs` (plain `harness = false` binaries). |

The split to keep in mind: an op's public surface, argument checks and autograd recording are in `src/ops/`; the
computation is a `Backend` method, declared in `src/backend.rs` and implemented for the CPU in `src/cpu/mod.rs`, which
calls a kernel in `src/cpu/<area>.rs`. An op's gradient is the closure it passes to `Tensor::track` (or
`track_saved`) right after the backend call; the closure is built from ordinary ops, often a `*_backward` backend
method. `src/autograd.rs` only walks the graph.

## Start here

1. The example in `src/lib.rs`: a conv, a loss, `backward()` and an in-place SGD step.
2. `Tensor::unary` in `src/ops/elementwise.rs`: a whole op in a few lines (dtype check, backend call, `track` with
   its backward).
3. `Tensor::track` and `Tensor::custom_op` in `src/tensor.rs`: how a result gets its backward node.
4. `backward` in `src/autograd.rs`: the reverse topological walk that returns `Grads`.
5. The `Backend` trait in `src/backend.rs`: its module doc is the contract every device implements.
6. `impl Backend for CpuBackend` in `src/cpu/mod.rs`, then the kernel file of the op you care about.
7. `grad_check_with` in `src/testing.rs`: how gradients are verified.

## Common changes

| To do this | Touch these files |
|---|---|
| Add a unary op | `src/backend.rs` (a `UnaryOp` variant and its `UnaryOp::name`), `src/cpu/elementwise/mod.rs` (`unary` and `unary_backward`), `src/ops/elementwise.rs` (an entry in `unary_fns!`), `tests/grad.rs` (`unary_ops`). For the GPU: `crates/folo-gpu/src/kernels/ew.rs` (`unary_code` and the `un` / `unb` cases) and `crates/folo-gpu/src/tests/backend/elementwise.rs`; without them the GPU backend returns an error for the op, as it does today for `Atan`, `Asin`, `Acos` and `Erf`. |
| Add an op with its own backend method | `src/backend.rs` (the method, with a default that returns `Error::NotSupported`), `src/cpu/mod.rs` and a kernel in `src/cpu/<area>.rs`, the `Tensor` method in `src/ops/<area>.rs` (checks, backend call, `track`), `tests/grad.rs` and `tests/reference.rs`, the GPU side in `crates/folo-gpu/src/backend/ops.rs` (see [its README](../folo-gpu/README.md)), and the method list of the [backend contract](../../docs/CORE.md#backend-integration-contract). |
| Add a fused op behind an optional hook | `src/backend.rs` (the hook, defaulting to `NotSupported`), `src/ops/fused.rs` (the entry point with its composite fallback), optionally `src/cpu/mod.rs`; a test that the fused result matches the composite one (`fused_conv_act_matches_composite` in `tests/reference.rs` is the pattern); the hook list in the [backend contract](../../docs/CORE.md#backend-integration-contract). |
| Change a CPU kernel | `src/cpu/<area>.rs` (or `gemm/`, `conv/`), fanning out through `cpu::par_chunks` / `cpu::par_for` rather than rayon iterators (they keep the build short: [docs/CORE.md](../../docs/CORE.md#cpu-kernels)); `tests/reference.rs` and, for reductions, scatters, the GEMM or convs, `tests/determinism.rs`; the matching bench in `benches/`, with numbers in the PR. |
| Record a fused kernel from another crate | Nothing here: call `Tensor::custom_op` (`tests/custom_op.rs` shows it) and check it with `testing::grad_check`. |

## Tests and benchmarks

```text
cargo test -p folo-core                       # unit tests in src/, tests/*.rs and the doctests; no GPU needed
cargo bench -p folo-core --bench gemm         # also: conv, conv_infer, conv_wgrad, index
```

Nothing in this crate is `#[ignore]`d. The GPU backend is checked against this crate's CPU backend in folo-gpu's own
tests (`crates/folo-gpu/src/tests/backend/`).

## Further reading

- [docs/CORE.md](../../docs/CORE.md): this crate in depth (tensors, kernels, autograd, determinism).
- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [docs/GPU.md](../../docs/GPU.md) and [crates/folo-gpu](../folo-gpu/README.md): the GPU backend.
- [docs/NN.md](../../docs/NN.md): folo-nn, the layers and optimizers built on these tensors.
