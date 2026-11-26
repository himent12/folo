# folo-nn: layers, optimizers, schedules and `.folo` files

`crates/folo-nn` is everything the detector and trainer need below the model definition. It implements the
building blocks and recipe of [MODEL.md](MODEL.md) and the file format of [FORMAT.md](FORMAT.md); those two files
are normative, this one is the map.

## Modules
- **`Module`** (structure) and **`Forward`** (`fn forward(&self, &Tensor) -> Result<Tensor>`) are separate traits,
  because not every module maps one tensor to one tensor (the detector head returns several). `Layer` is any
  `Module + Forward`, the element type of `Sequential`.
- A module reports its own tensors (`visit_tensors`, each a `Parameter` or a `Buffer`) and its direct children
  (`visit_children`) in **canonical order**: construction order, MODEL.md §5. Everything else is provided:
  `named_tensors()`, `named_parameters()`, `named_buffers()`, `parameters()`, `num_parameters()` (parameters only,
  as MODEL.md §4 counts), `for_each_module(f)` (pre-order visitor with dotted paths; downcast through `&dyn Any`
  to act on one layer type), and `train()` / `eval()` / `is_training()`.
- Names are dotted paths built from the child names: `backbone.stem.conv.weight`. A child with an empty name is
  transparent. `ModuleList::with_prefix("block", items)` names its items `block0, block1, …`, and a parent that
  forwards the list's `visit_children` gets MODEL.md's `<p>.block0.*` names without an extra segment.
- Mode is an atomic flag owned by the modules that need it (`BatchNorm2d`), so `train()` / `eval()` take `&self`
  and propagate with the same visitor as names.
- **Devices.** `to_device(&mut self, Device)` moves every parameter and buffer in place (folo-core's
  `Tensor::to_device_`): each tensor keeps its identity (`TensorId`), name, dtype and `requires_grad`, and the mode is
  unchanged, so parameter groups, `GradAccumulator` slots and an EMA's source built earlier stay valid.
  `Optimizer::to_device` moves the optimizer's parameters and state (SGD momentum buffers, AdamW moments),
  `GradAccumulator::to_device` its buffers and `ModelEma::to_device` the averaged model (`ModelEma::new` also moves
  its copy to the source's device). A whole training setup can therefore be built on the CPU and moved, even
  mid-run (tested against an all-CPU run in `tests/gpu/main.rs`).

## Layers
| layer | notes |
|---|---|
| `Conv2d` | groups, stride, padding, dilation, optional bias. Init per MODEL.md §2.9: weight then bias, `U(±1/sqrt(fan_in))` |
| `BatchNorm2d` | `eps = 1e-3`, `momentum = 0.03`. Train: batch mean and biased variance, running update with the unbiased variance under `no_grad`; `N·H·W < 2` is an error. Eval: running statistics. On the CPU, train mode is one fused kernel (`f64` statistics in `256`-float L1 blocks, hand-written L1-blocked backward in `256`-float tiles — `bwd_act_and_sum` + `bwd_write_dx` — fused per channel when a channel's planes fit in `32_768` floats, via `Tensor::custom_op`), about 5× faster than composite ops. On the GPU, train mode is folo-gpu's fused BatchNorm kernels (Welford statistics, the running update inside the kernel, subgroup-reduced finalization, analytic backward) recorded as one `batch_norm2d_act` custom op; `forward_act(x, act)` fuses the following SiLU / ReLU in as well. Other devices compose tensor ops |
| `ConvBnAct` | MODEL.md's `Conv` (§2.1): bias-free conv, BN, activation. `fuse_()` folds the BN into the conv; `new_fused` builds the fused structure for loading `folo.fused` files. Forward: fused → `Tensor::conv2d_act` (on the GPU one kernel with bias and activation in the epilogue when nothing is recorded); train → conv then `BatchNorm2d::forward_act`; eval on a non-CPU device (or on the CPU under `no_grad`) → the BN folded into the conv weights on the fly (small per-channel ops, still differentiable) then `conv2d_act`; eval on the CPU with recording → conv, BN, activation. `forward_cat(&[..])` is `forward(cat(.., 1))` that reads the parts in place when the conv runs fused (`Tensor::conv2d_act_cat`) |
| `Linear`, `Activation` (Identity, SiLU, ReLU, LeakyReLU, Sigmoid, Tanh, Hardswish), `MaxPool2d`, `Upsample` (nearest), `Concat` (several inputs, so not a `Forward`), `Identity`, `Sequential`, `ModuleList` | |

`RunningStats` (`layers/batchnorm/stats.rs`, alongside `mod.rs`, `cpu.rs`, and `tests.rs` in the split
`layers/batchnorm/` module) implements the non-finite batch guard of MODEL.md §8.3: `RunningStats::capture(model)`
snapshots every floating buffer, while `RunningStats::capture_training(model)` captures only active train-mode
`BatchNorm2d` buffers once per epoch, refreshes them in place via `recapture()` without reallocating snapshot tensors,
checks live finiteness via `all_finite_live()` without re-walking the module tree, and pairs with
`RunningStats::all_finite_eval(model)` to validate frozen eval-mode BN buffers at epoch entry (`restore()` undoes a
non-finite batch). Buffers on a device are checked with one readback per device via multi-tensor
`Tensor::norm_partials_many(bufs, NormReduce::AbsMax, None)` + `max_all()` (falling back to `Σ 0 · x == 0` when
`norm_partials_many` is unsupported).

`fuse_conv_bn(w, b, γ, β, mean, var, eps)` is the reusable fusion (`k = γ / sqrt(var + eps)`, `W' = W·k`,
`b' = β + (b − mean)·k`), in `F32`.

## Init (`folo_nn::init`)
`kaiming_uniform` / `kaiming_normal` (gain from the leaky-ReLU slope, fan-in or fan-out), `xavier_uniform` /
`xavier_normal`, `uniform`, `normal`, PyTorch fans, and `constant_` / `copy_` for re-initialising a parameter in
place. Every sampler draws row-major from a `folo_core::rng::Rng`, so initialising in canonical order from one seed
is reproducible everywhere. Head priors (MODEL.md §6.5): `BOX_BIAS_PRIOR = 2.0`, `cls_bias_prior(nc, stride)`.

## Optimizers (`folo_nn::optim`)
- `ParamGroup`: name, `(name, tensor)` list, base LR / momentum (μ or β1) / weight decay / Nesterov / β2 / ε, plus
  the current `lr` and `momentum` the schedule writes.
- `StandardGroups::from_module(model)` splits MODEL.md §8.2's `decay` (weights), `bn` (BatchNorm γ plus 1-D no-decay
  scale/gate parameters `gamma_id` and `alpha_raw` used by FOLO v2-Iso) and `bias` (every `*.bias`) groups;
  `into_groups(lr, momentum, wd)` gives decay to `decay` only.
- `Sgd`: L2 (coupled) decay, `buf = μ·buf + g` (zero-initialised, so the first step is `buf = g`), Nesterov
  `p −= lr·(g + μ·buf)`. `AdamW`: decoupled decay `p *= 1 − lr·wd`, bias-corrected moments, global step counter.
  Both skip parameters without a gradient, update CPU tensors with non-FMA AVX2-vectorized fused loops (`optim/simd/`,
  bitwise identical to scalar and also used by Muon updates, gradient accumulation/clipping, and CPU AMP
  unscaling/norm checks; parallel over parameters for large steps), update each parameter group on the GPU with one
  multi-tensor call (`Tensor::optimizer_step_many_`: 16 parameters per dispatch, the per-element op order of the
  single-parameter kernel `Tensor::optimizer_step_`, which is the fallback), and fall back to tensor ops with the same
  arithmetic elsewhere.
- `GradAccumulator` sums gradients over a window of backward passes into private buffers (reused across windows),
  implements `GradSource` for the step, and does MODEL.md's global-norm clipping (`clip_grad_norm`). On the GPU a pass
  adds with one multi-tensor `buf = buf + g` call (16 buffers per dispatch), and clipping stays on the device: the norm (in `f32`) and
  the factor `max_norm / (n + 1e-6)` (or 1) are device scalars that scale every buffer, so no value is read back
  (`clip_grad_norm` then returns NaN instead of the norm). On the GPU the largest magnitude, the sum of squares and
  the scaling are multi-tensor calls (`Tensor::norm_partials_many`, `Tensor::scale_many_`: 16 buffers per
  dispatch, one partial per 256 elements, then one reduction), about 60 dispatches per window instead of ten tensor
  ops per buffer. The maximum and the scaling equal the tensor ops bit for bit; the sum of squares adds the same
  f32 terms in another (fixed) order.
- Mixed precision: `GradAccumulator::unscale_clip(inv_scale, max_norm)` unscales the accumulated gradients, clips
  them and returns whether all are finite, with one readback on the GPU: with clipping the unscale is folded into the
  clip factors and the finiteness check is the clipping norm itself (the overflow-safe sum makes it non-finite
  exactly when a gradient is); without, it is `amp::unscale_` (below).
- Recipe helpers: `accumulate_steps(nbs, batch)`, `effective_weight_decay`, `auto_optimizer` (§8.2 `auto`).
- `Optimizer::contract()` / `apply_contract()` / `state_tensors()` / `load_state()` are the `folo.optim` contract
  and `optim.*` tensors of FORMAT.md. `MuSgd::with_weights(groups, MuSgdWeights { muon, sgd })` sets MuSGD's step
  weights (MODEL.md §8.2; default 0.2 / 1.0), `MuSgd::with_ortho(groups, weights, ortho)` sets its
  orthogonalisation schedule (`MuonOrtho::Polar4` default, `Polar5`, `Polar3Cs9`, or legacy `Ns5`; `orthogonalize_with` /
  `orthogonalize_many_with` expose the schedules directly), and `MuSgd::with_ortho_and_geo(groups, weights, ortho, geo)`
  selects Conv-GeoMuon (`MuonGeoMode::Geo`: true 2-D flattened conv aspect scaling + exact zero-GEMM per-channel Polar
  LMO for depthwise convs; default `Legacy`). `MuonOrtho::Polar3Cs9` (`POLAR3_CS9_COEFFS: [Cs9StepCoeffs; 3]`,
  `cs9_schedule()`, `is_cs9()`) runs a 3-step Completed-Square Paterson–Stockmeyer degree-9 odd polynomial
  (`S = δ·I + γ·A + A²`, `q₄(A) = ξ·I + η·A + c₄·S²`, `X' = q₄(A)·X`, 6 wide `r × c` GEMMs vs 8 in `Polar4`, total
  degree 729) paired with zero-extra-GEMM step-0 Schatten-8 spectral rescaling
  `α₈ = ‖A₀²‖_F^(1/4) + eps / (‖u‖_F + eps)`. Weights other than the defaults (`musgd`), schedules other than legacy
  `Ns5` (`muon_ortho`: `"polar4"`, `"polar5"`, `"polar3_cs9"`), and non-legacy geometric modes (`muon_geo`, plus
  `muon_depthwise` naming the depthwise conv weights when `Geo` is active) are part of its contract; `apply_contract`
  adopts the file's weights when present (keeping the optimizer's otherwise) and restores `muon_ortho` (defaulting
  absent `muon_ortho` to `Ns5` so pre-Polar checkpoints resume with legacy 5-step Newton–Schulz), `muon_geo`
  (defaulting absent `muon_geo` to `Legacy`), and the depthwise parameter mask from `muon_depthwise`.

## Mixed precision (`folo_nn::amp`)
- `GradScaler`: dynamic loss scaling as `torch.cuda.amp.GradScaler` (docs/MODEL.md §8.8): `scale()` (initially
  `2^16`), `inv_scale()`, `update(found_inf)` (a skip backs off ×0.5 and restarts the count; 2000 finite steps in a
  row grow ×2) returning `ScaleUpdate::{Kept, Grew, BackedOff}`, `skipped_steps()`, `skip_streak()` (skips in a
  row; the trainer fails a run at 50), and the `folo.amp` checkpoint entry (`contract()` / `from_contract()`,
  `to_json()` / `from_json()`, exact round trip). The scale stays within `[MIN_SCALE, MAX_SCALE] = [2^-126, 2^126]`
  (`usable_scale`: it and its inverse are normal f32 values); restored states outside it are refused.
- `amp::unscale_(tensors, inv_scale)`: multiplies private gradient buffers by `1 / S` in place and returns whether all
  are finite. On a device the buffers take one multi-tensor `Tensor::unscale_check_many_` call (16 per dispatch,
  else one `Tensor::unscale_check_` kernel each) that also raises a shared device flag, read back once; elsewhere
  tensor ops.
- The forward side is folo-core's `autocast` ([CORE.md](CORE.md#mixed-precision)); layers need no changes: `Conv2d` and
  `ConvBnAct` call `conv2d_with`, which autocast runs in f16 against the f32 weights, and `BatchNorm2d`'s fused GPU
  path takes f16 activations with its f32 parameters and statistics (its dγ, dβ are f32, dx f16).

## Schedule and EMA
- `LrSchedule::new(kind, epochs, steps_per_epoch, lrf, warmup_epochs, warmup_bias_lr, warmup_momentum)` computes
  `W` with MODEL.md §8.4's formula at a floor of 100 (`warmup_iters`, the floor of runs made before
  `warmup_floor`); `LrSchedule::with_warmup_iters` takes `W` as given, e.g. from `warmup_iters_floor(.., floor)`,
  which the trainer calls with `TrainConfig.warmup_floor` (default 0: no floor). Supported decay shapes (`DecayKind`):
  - `Linear` (default) and `Cosine` (`cos_lr`), decaying across the full run after warmup.
  - `Wsd` (opt-in Warmup–Stable–Decay): holds `lf(t) = 1.0` through the stable plateau and decays over the final
    `decay_fraction` (default `0.2`, in `(0, 1]`) down to `lrf` with `WsdTail::{Linear, Cosine, Sqrt}`.
  - `CosineRestarts` (opt-in SGDR): after warmup, repeats cosine cycles starting at `restart_period` iterations and
    multiplying cycle length by `restart_mult` (`T_mult >= 1`) each restart (defaults: `3` equal cycles, `T_mult = 1`).
  Each optimizer step: `sched.apply(&mut opt)` (LR and momentum of every group at iteration `t`; the `bias` group
  warms up from `warmup_bias_lr`), `opt.step(..)`, `sched.advance()`. `contract()` / `from_contract()` are
  `folo.sched` (optional WSD / restart fields serialize only when set).
- `ModelEma<M>` keeps a second instance of the model (eval mode, no grad, on the source's device), caches its tracked
  parameter and floating-buffer handles to avoid repeated module-tree walks on every step, and updates them with
  either the legacy exponential ramp `d = decay · (1 − exp(−updates / tau))` (`EmaMode::Legacy`, default) or opt-in
  `PC-PEMA` (`EmaMode::Power`, `ModelEma::with_power(module, source, EmaPowerConfig)`). `EmaPowerConfig` holds
  `{ gamma_0, rho, lf_min, t_0 }` (defaults `PEMA_GAMMA_0 = 6.94`, `PEMA_RHO = 0.25`, `PEMA_LF_MIN = 0.01`,
  constructor `EmaPowerConfig::new(t_0)`, `validate()`, and `decay_at(u, lf)` computing
  `exp((γ₀ · clamp(lf, lf_min, 1)^(-ρ) + 1) · ln(1 − 1 / (u + t₀)))`, or `0.0` when `u + t₀ <= 1`); callers drive
  `PC-PEMA` with `ModelEma::update_with_lf(source, lf)` (`decay_at_with_lf(u, lf)`; `update` / `decay_at` use
  `lf = 1.0`, identical in legacy mode). `EmaContract` persists `power: Option<EmaPowerConfig>` (omitted when `None`
  so legacy `folo.ema` stays byte-identical). On the CPU, contiguous `f32` updates run non-FMA AVX2 loops (parallelised
  across tensors above `32_768` total elements — with storage-alias checking — and across `16_384`-element chunks
  inside single tensors above `65_536` elements); on the GPU, all tracked tensors update in one multi-tensor
  `a·p + b·g` call (16 per dispatch). `ModelEma::new` takes MODEL.md §8.5's 0.9999 / 2000;
  `ema_for_run(total_updates, fraction)` is the `(decay, tau)` of a window of `fraction` of a run (a fine-tune's,
  capped at the default window), for `ModelEma::with_params`.
  The object-safe `EmaState` trait is what checkpoints use.

## `.folo` files (`folo_nn::io`)
- `io::safetensors`: the container. `Container::parse` applies FORMAT.md's acceptance rules exactly (header limit,
  leading `{`, trailing spaces only, no duplicate keys, string-only metadata, exact coverage of the data section);
  every safetensors dtype parses, and `decode` refuses those FOLO does not use with a clear error. `write` uses
  the FOLO layout (descending dtype size, then name; header padded so data starts 8-aligned); `write_file` writes
  a fresh, uniquely named temporary file next to the target (`create_temp_beside`, `create_new`, so it never
  truncates an existing file) and renames it.
- `Metadata` / `ModelConfig`: typed `folo.*` keys, validated (format major version, MODEL.md §3 config rules,
  `num_classes == len(classes)`); unknown keys are kept in `extra` and written back.
- `ModelFile::open` (memory-mapped) / `from_bytes` / `import` (plain safetensors plus caller metadata) validate
  the namespaces: weights files hold only the model namespace; checkpoints must have consistent step counters,
  a complete `ema.` set, and every group parameter in the model; `folo.dtype` must match the tensors.
- `ModelFile::load(model, &LoadOptions)`: source selection (EMA for inference when present, model namespace for
  training), strict (every offending name listed; class lists must be identical) or partial (load matching names,
  shapes and convertible dtypes; skip and report the rest; skip the class-output layers `head.cls.*.out.*` whenever
  the class lists differ), `F16 → F32` conversion (and `F16` files into an `F16` model as they are), fused files for
  inference only. `LoadReport` lists the loaded,
  skipped (with reasons) and missing names.
- Checkpoints: `save_checkpoint(path, &meta, &CheckpointParts { model, ema, optimizer, schedule, rng, epoch, best,
  train_config })` writes the model, `ema.*` + `ema.__updates`, `optim.*` + `optim.__step`, `sched.__step` and the
  `folo.train/optim/sched/ema/rng/epoch/best` contracts. `ModelFile::resume(model, optimizer, ema, classes)` checks
  and restores all of it and returns the schedule, RNG streams, epoch and early-stopping state. Resuming then
  stepping is bit-identical to never having stopped (tested).
- Export: `ModelFile::export(path, F32 | F16)` takes the EMA (or model) tensors, fuses Conv+BN in `F32`
  (`fuse_bn_tensors`), optionally casts to `F16`, and writes a fused `weights` file. `save_weights`,
  `export_module` and `export_tensors` do the same from in-memory modules and tensor lists. `write_weights`,
  `write_export` and `ModelFile::export_to` write the same bytes to any `io::Write` (a `.folo` file held in
  memory, e.g. the mock server's weights).

## Example
See the crate docs (`cargo doc -p folo-nn --open`): a two-block model with SGD, the schedule, gradient
accumulation, clipping and EMA in a step loop.

## Benchmarks
`cargo bench -p folo-nn --bench nn` reports BatchNorm train forward + backward (fused vs composite), the SGD, AdamW,
MuSGD and EMA steps over a folo-n-sized parameter set, and `.folo` save / load (reference release numbers on an RTX
4080 SUPER host: MuSGD `Polar4` 2.64M `10.15 ms`, EMA `0.11 ms`, BN fused + SiLU `8.68 / 3.54 / 0.75 ms`). The GPU
paths are measured on the real detector by `cargo bench -p folo-vision --bench gpu` (docs/BENCHMARKS.md, "FOLO on the
GPU").
