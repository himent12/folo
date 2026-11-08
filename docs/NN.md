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

## Layers
| layer | notes |
|---|---|
| `Conv2d` | groups, stride, padding, dilation, optional bias. Init per MODEL.md §2.9: weight then bias, `U(±1/sqrt(fan_in))` |
| `BatchNorm2d` | `eps = 1e-3`, `momentum = 0.03`. Train: batch mean and biased variance, running update with the unbiased variance under `no_grad`; `N·H·W < 2` is an error. Eval: running statistics. On the CPU, train mode is one fused kernel (`f64` statistics, hand-written backward via `Tensor::custom_op`), about 5× faster than the composite ops other devices use |
| `ConvBnAct` | MODEL.md's `Conv` (§2.1): bias-free conv, BN, activation. `fuse_()` folds the BN into the conv; `new_fused` builds the fused structure for loading `folo.fused` files |
| `Linear`, `Activation` (Identity, SiLU, ReLU, LeakyReLU, Sigmoid, Tanh, Hardswish), `MaxPool2d`, `Upsample` (nearest), `Concat` (several inputs, so not a `Forward`), `Identity`, `Sequential`, `ModuleList` | |

`RunningStats::capture(model)` / `restore()` / `all_finite(model)` implement the non-finite batch guard of
MODEL.md §8.3: snapshot the running statistics before a train-mode forward and restore them if the loss or a
statistic is not finite.

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
  parameters for large steps) and fall back to tensor ops on other devices with bit-identical arithmetic.
- `GradAccumulator` sums gradients over a window of backward passes into private buffers (reused across windows),
  implements `GradSource` for the step, and does MODEL.md's global-norm clipping (`clip_grad_norm`).
- Recipe helpers: `accumulate_steps(nbs, batch)`, `effective_weight_decay`, `auto_optimizer` (§8.2 `auto`).
- `Optimizer::contract()` / `apply_contract()` / `state_tensors()` / `load_state()` are the `folo.optim` contract
  and `optim.*` tensors of FORMAT.md.

## Schedule and EMA
- `LrSchedule::new(kind, epochs, steps_per_epoch, lrf, warmup_epochs, warmup_bias_lr, warmup_momentum)` computes
  `W` with MODEL.md §8.4's formula. Each optimizer step: `sched.apply(&mut opt)` (LR and momentum of every group at
  iteration `t`; the `bias` group warms up from `warmup_bias_lr`), `opt.step(..)`, `sched.advance()`.
  `contract()` / `from_contract()` are `folo.sched`.
- `ModelEma<M>` keeps a second instance of the model (eval mode, no grad) and updates every parameter and floating
  buffer with `d = decay · (1 − exp(−updates / tau))`. The object-safe `EmaState` trait is what checkpoints use.

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
  the class lists differ), `F16 → F32` conversion, fused files for inference only. `LoadReport` lists the loaded,
  skipped (with reasons) and missing names.
- Checkpoints: `save_checkpoint(path, &meta, &CheckpointParts { model, ema, optimizer, schedule, rng, epoch, best,
  train_config })` writes the model, `ema.*` + `ema.__updates`, `optim.*` + `optim.__step`, `sched.__step` and the
  `folo.train/optim/sched/ema/rng/epoch/best` contracts. `ModelFile::resume(model, optimizer, ema, classes)` checks
  and restores all of it and returns the schedule, RNG streams, epoch and early-stopping state. Resuming then
  stepping is bit-identical to never having stopped (tested).
- Export: `ModelFile::export(path, F32 | F16)` takes the EMA (or model) tensors, fuses Conv+BN in `F32`
  (`fuse_bn_tensors`), optionally casts to `F16`, and writes a fused `weights` file. `save_weights`,
  `export_module` and `export_tensors` do the same from in-memory modules and tensor lists.

## Example
See the crate docs (`cargo doc -p folo-nn --open`): a two-block model with SGD, the schedule, gradient
accumulation, clipping and EMA in a step loop.

## Benchmarks
`cargo bench -p folo-nn --bench nn` reports BatchNorm train forward + backward (fused vs composite), the SGD, AdamW
and EMA steps over a folo-n-sized parameter set, and `.folo` save / load.
