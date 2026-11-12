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
  mid-run (tested against an all-CPU run in `tests/gpu.rs`).

## Layers
| layer | notes |
|---|---|
| `Conv2d` | groups, stride, padding, dilation, optional bias. Init per MODEL.md §2.9: weight then bias, `U(±1/sqrt(fan_in))` |
| `BatchNorm2d` | `eps = 1e-3`, `momentum = 0.03`. Train: batch mean and biased variance, running update with the unbiased variance under `no_grad`; `N·H·W < 2` is an error. Eval: running statistics. On the CPU, train mode is one fused kernel (`f64` statistics, hand-written backward via `Tensor::custom_op`), about 5× faster than composite ops. On the GPU, train mode is folo-gpu's fused BatchNorm kernels (Welford statistics, the running update inside the kernel, analytic backward) recorded as one `batch_norm2d_act` custom op; `forward_act(x, act)` fuses the following SiLU / ReLU in as well. Other devices compose tensor ops |
| `ConvBnAct` | MODEL.md's `Conv` (§2.1): bias-free conv, BN, activation. `fuse_()` folds the BN into the conv; `new_fused` builds the fused structure for loading `folo.fused` files. Forward: fused → `Tensor::conv2d_act` (on the GPU one kernel with bias and activation in the epilogue when nothing is recorded); train → conv then `BatchNorm2d::forward_act`; eval on a non-CPU device (or on the CPU under `no_grad`) → the BN folded into the conv weights on the fly (small per-channel ops, still differentiable) then `conv2d_act`; eval on the CPU with recording → conv, BN, activation. `forward_cat(&[..])` is `forward(cat(.., 1))` that reads the parts in place when the conv runs fused (`Tensor::conv2d_act_cat`) |
| `Linear`, `Activation` (Identity, SiLU, ReLU, LeakyReLU, Sigmoid, Tanh, Hardswish), `MaxPool2d`, `Upsample` (nearest), `Concat` (several inputs, so not a `Forward`), `Identity`, `Sequential`, `ModuleList` | |

`RunningStats::capture(model)` / `restore()` / `all_finite(model)` implement the non-finite batch guard of
MODEL.md §8.3: snapshot the running statistics before a train-mode forward and restore them if the loss or a
statistic is not finite. Buffers on a device are checked there with a single readback (`Σ 0 · x` over all of them
is 0 exactly when every value is finite).

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
- `StandardGroups::from_module(model)` splits MODEL.md §8.2's `decay` (weights), `bn` (BatchNorm γ) and `bias`
  (every `*.bias`) groups; `into_groups(lr, momentum, wd)` gives decay to `decay` only.
- `Sgd`: L2 (coupled) decay, `buf = μ·buf + g` (zero-initialised, so the first step is `buf = g`), Nesterov
  `p −= lr·(g + μ·buf)`. `AdamW`: decoupled decay `p *= 1 − lr·wd`, bias-corrected moments, global step counter.
  Both skip parameters without a gradient, update CPU tensors with fused per-element loops (parallel over
  parameters for large steps), update each parameter group on the GPU with one multi-tensor call
  (`Tensor::optimizer_step_many_`: 16 parameters per dispatch, the per-element op order of the single-parameter
  kernel `Tensor::optimizer_step_`, which is the fallback), and fall back to tensor ops with the same arithmetic
  elsewhere.
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
  weights (MODEL.md §8.2; default 0.2 / 1.0); weights other than the defaults are part of its contract (`musgd`),
  and `apply_contract` adopts the file's (when it has none, the optimizer keeps the weights it was built with).

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
- The forward side is folo-core's `autocast` (docs/ARCHITECTURE.md); layers need no changes: `Conv2d` and
  `ConvBnAct` call `conv2d_with`, which autocast runs in f16 against the f32 weights, and `BatchNorm2d`'s fused GPU
  path takes f16 activations with its f32 parameters and statistics (its dγ, dβ are f32, dx f16).

## Schedule and EMA
- `LrSchedule::new(kind, epochs, steps_per_epoch, lrf, warmup_epochs, warmup_bias_lr, warmup_momentum)` computes
  `W` with MODEL.md §8.4's formula at a floor of 100 (`warmup_iters`, the floor of runs made before
  `warmup_floor`); `LrSchedule::with_warmup_iters` takes `W` as given, e.g. from `warmup_iters_floor(.., floor)`,
  which the trainer calls with `TrainConfig.warmup_floor` (default 0: no floor). Each optimizer step: `sched.apply(&mut opt)` (LR and momentum of every group at
  iteration `t`; the `bias` group warms up from `warmup_bias_lr`), `opt.step(..)`, `sched.advance()`.
  `contract()` / `from_contract()` are `folo.sched`.
- `ModelEma<M>` keeps a second instance of the model (eval mode, no grad, on the source's device) and updates every
  parameter and floating buffer with `d = decay · (1 − exp(−updates / tau))` (on the GPU one multi-tensor
  `a·p + b·g` call for all of them, 16 per dispatch).
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
`cargo bench -p folo-nn --bench nn` reports BatchNorm train forward + backward (fused vs composite), the SGD, AdamW
and EMA steps over a folo-n-sized parameter set, and `.folo` save / load. The GPU paths are measured on the real
detector by `cargo bench -p folo-vision --bench gpu` (docs/BENCHMARKS.md, "FOLO on the GPU").
