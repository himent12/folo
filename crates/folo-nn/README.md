# folo-nn

folo-nn is everything the detector and the trainer need below the model definition: the `Module` trait, layers,
weight initialisation, optimizers, the LR schedule, the weight EMA, loss scaling for mixed precision and the `.folo`
weight format. It depends on one workspace crate, folo-core (tensors and autograd). folo-vision, folo-server,
folo-cli, folo-wasm and FOLO Desktop (`apps/desktop`) depend on it; folo-gpu has it as a dev-dependency only. The
normative specs it implements are [MODEL.md](../../docs/MODEL.md) (blocks and training recipe) and
[FORMAT.md](../../docs/FORMAT.md) (the file format); [NN.md](../../docs/NN.md) is the overview.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | Crate docs with a worked step loop (a doctest), and the re-exports |
| `src/module.rs` | `Module` (named parameters and buffers, train / eval mode, traversal), `Forward`, `Layer` |
| `src/layers/` | `conv.rs` (`Conv2d`), `ws_conv.rs` (`WeightStandardizedConv2d`, `standardize_weight`), `repconv.rs` (`RepConv`, `RepConvConfig`: multi-branch RepVGG block with `switch_to_deploy()` fusing into a single `3×3` `Conv2d`), `rope2d.rs` (`RoPE2d`: 2D axial Rotary Positional Embedding), `cross_attn.rs` (`MultiHeadCrossAttention`, `MultiHeadCrossAttentionConfig` with optional QK-RMSNorm, additive masks, and `RoPE2d`), `tests_repconv_rope_attn.rs`, `attention.rs` (`SqueezeExcitation`, `Eca`, `eca_kernel_size`, `ChannelAttention`, `SpatialAttention`, `Cbam`), `tests_attention.rs`, `convnext.rs` (`LayerNorm`, `ConvNeXtBlock`, `ConvNeXtConfig`, `ConvNeXtNorm`), `mbconv.rs` (`InvertedResidual` MBConv, `InvertedResidualConfig`), `dropblock.rs` (`DropBlock2d`, `dropblock_gamma`), `tests_blocks.rs`, `batchnorm/` (`mod.rs`: `BatchNorm2d`, `cpu.rs`: fused CPU training kernel, `stats.rs`: `RunningStats`, `tests.rs`), `conv_bn_act.rs` (`ConvBnAct`, MODEL.md's `Conv`, and `fuse_conv_bn`), `glu.rs` (`GatedFfn` SwiGLU / GeGLU, `GluKind`, `swiglu`, `geglu`, cancellation-free `gelu_tanh`), `rmsnorm.rs` (`RmsNorm`), `regularize.rs` (`DropPath` stochastic depth, `LayerScale` per-channel residual scaling, `ChannelLayout`), `spectral_norm.rs` (`SpectralNorm` wrapper for `Linear`/`Conv2d`, `power_iteration`, `spectral_norm`, `spectral_normalize`), `simple.rs` (`Linear`, `Activation`, `MaxPool2d`, `Upsample`, `Concat`, `Identity`), `container.rs` (`Sequential`, `ModuleList`) |
| `src/init.rs` | Seeded Kaiming / Xavier / Orthogonal (Householder QR with Mezzadri positive-diagonal sign correction) initialisation and the head bias priors (MODEL.md §2.9, §6.5) |
| `src/optim/` | `mod.rs` (`Optimizer`, `OptimKind`, `ParamGroup`, `StandardGroups`, the `auto` optimizer rule), `dispatch.rs` (zero-allocation CPU parameter scheduling, O(1) storage identity, and multi-group GPU coalescing), `sgd.rs`, `adamw.rs`, `muon/` (`mod.rs`, `tests.rs`: standalone `Muon` Momentum Orthogonalized by Newton–Schulz optimizer + `MuonConfig` / `MuonAspectScale` / `MuonFallback` / `MuonContract` / `newton_schulz_quintic`), `adafactor/` (`mod.rs`, `tests.rs`: `Adafactor` memory-efficient factored second-moment optimizer Shazeer & Stern 2018 + `AdafactorConfig` / `AdafactorContract` / `adafactor_beta2` / `adafactor_relative_lr`), `prodigy/` (`mod.rs`, `contract.rs`, `tests.rs`: `Prodigy` parameter-free adaptive D-estimation AdamW Mishchenko & Defazio 2024 + `ProdigyConfig` / `ProdigyContract`), `sophia/` (`mod.rs`, `contract.rs`, `tests.rs`: `SophiaG` second-order clipped Gauss–Newton–Bartlett / seeded Hutchinson optimizer Liu et al. 2024 + `SophiaConfig` / `SophiaContract` / `HutchinsonDist`), `radam/` (`mod.rs`, `tests.rs`: `RAdam` Rectified Adam optimizer Liu et al. 2020 + `radam_rectification` / `RAdamConfig` / `RAdamContract`), `lookahead/` (`mod.rs`, `tests.rs`: `Lookahead<O>` k-step fast/slow wrapper Zhang et al. 2019 + `LookaheadConfig` / `LookaheadContract`), `adan/` (`mod.rs`, `tests.rs`: `Adan` Adaptive Nesterov Momentum optimizer Xie et al. 2024 + `AdanConfig` / `AdanContract`), `lion.rs` (opt-in `Lion` Evolved Sign Momentum optimizer + `LionContract`), `schedulefree/` (`mod.rs`, `tests.rs`: opt-in `ScheduleFreeAdamW` Defazio et al. 2024 + `ScheduleFreeConfig` / `ScheduleFreeContract`), `cautious.rs` (opt-in Cautious update masking for `AdamW` / `Sgd`), `musgd.rs` (`MuSgd`), `ortho.rs` (Newton–Schulz `orthogonalize`, Polar-FOLO `MuonOrtho` / `orthogonalize_with`, and `Polar3-CS9` `POLAR3_CS9_COEFFS`), `grads.rs` (`GradAccumulator`, clipping), `agc.rs` (Adaptive Gradient Clipping `clip_grad_agc_tensor` / `AgcStats`), `simd/` (`mod.rs`, `norm.rs`, `vec_ops.rs`, `tests.rs`: non-FMA AVX2 / scalar CPU optimizer, 1-pass rescaled norm/clipping, and AMP unscale kernels) |
| `src/sched.rs` | `LrSchedule`: warmup, then linear, cosine, WSD, cosine-with-restarts, polynomial, or OneCycle decay (MODEL.md §8.4, `folo.sched` in FORMAT.md) |
| `src/sgdr.rs` | `SgdrSchedule`: standalone unbounded Cosine Annealing with Warm Restarts (SGDR, Loshchilov & Hutter 2017) with `t_0`, `t_mult`, `eta_min`, optional linear warmup, and `SgdrCycleInfo` |
| `src/ema.rs` | `ModelEma` (with `update_with_decay` / `update_with_power_law` / `update_with_edm2_power_law`), `ema_for_run` (a fine-tune's EMA window), `EmaMode` (`Legacy`, `Power`) and `EmaPowerConfig` (`PC-PEMA`; MODEL.md §8.5) |
| `src/power_ema.rs` | `PowerLawEmaSchedule` (`1 - (1 + step / t0)^(-gamma)`), `power_law_ema_decay`, and `Edm2PowerEmaSchedule` (Karras et al. 2024 EDM2 `(1 - 1 / (step + t0))^(gamma + 1)` with relative-width inversion) |
| `src/swa.rs` | `SwaModel`: Stochastic Weight Averaging (equal-weight running arithmetic mean across snapshots) |
| `src/amp.rs` | `GradScaler`, dynamic loss scaling for mixed-precision training (`folo.amp` in FORMAT.md) |
| `src/io/` | `.folo` files: `safetensors.rs` (the container), `meta.rs` (`__metadata__`, `ModelConfig`, `scale_row`), `file.rs` (`ModelFile`, loading rules), `checkpoint.rs` (training state), `export.rs` (Conv+BN fusion, `F16` export), `replace.rs` (`rename_over`), `publish.rs` (`publish_new`: a finished file moved to a name only while it is free) |
| `src/error.rs` | `Error` and `Result` |
| `tests/modules.rs` | Layers and `Module`: names, modes, gradient checks, hand-computed BatchNorm values, fusion |
| `tests/optim/` | `main.rs` + topic files (`step.rs`, `grads.rs`, `muon.rs`, `ema.rs`): optimizers, groups, accumulation, clipping, schedule application, MuSGD/orthogonalisation, and EMA against hand-computed values |
| `tests/folo_io/` | `main.rs` + topic files (`format.rs`, `checkpoint.rs`, `musgd_resume.rs`): `.folo` files, safetensors compatibility both ways, loading rules, export, and checkpoint resume |
| `tests/gpu/` | `main.rs`: modules, optimizer state and EMAs moved between devices, the fused BatchNorm and Conv paths on the GPU; `muon.rs`: MuSGD's fused Muon step; `optim.rs`: multi-group coalesced GPU optimizer steps and cross-group alias fallback |
| `benches/nn.rs` | BatchNorm train forward + backward, optimizer and EMA steps, `.folo` save / load |
| `benches/train_step.rs` | One CPU training step of a folo-n-sized network |

## Start here

1. `Module` and `Forward` in `src/module.rs`: every layer and the whole detector (`FoloModel`) implement `Module`,
   whose names come from `visit_tensors` and `visit_children`. Only one-tensor-in, one-tensor-out layers implement
   `Forward`: `Concat` and `FoloModel` (whose `forward` returns a `HeadOutput`) do not, so they cannot go in a
   `Sequential`.
2. `ConvBnAct` in `src/layers/conv_bn_act.rs`: the block the detector is made of, and how its forward picks a path
   (fused, train, eval).
3. `StandardGroups` and the `Optimizer` trait in `src/optim/mod.rs`, then `Sgd` in `src/optim/sgd.rs`.
4. `GradAccumulator` in `src/optim/grads.rs` and `LrSchedule` in `src/sched.rs`: what one training step does around
   the optimizer.
5. `ModelFile` in `src/io/file.rs`, then `save_checkpoint` in `src/io/checkpoint.rs`: how weights and training
   state are stored.

## Common changes

| To do this | Touch these files |
|---|---|
| Add a layer | A file in `src/layers/` (a small one can go in `simple.rs`); `mod` and `pub use` in `src/layers/mod.rs`; the re-export in `src/lib.rs`; `impl Module` (an empty impl for a layer without tensors, as `MaxPool2d`) and, for a one-input layer, `impl Forward`; tests in `tests/modules.rs`; the Layers table in `docs/NN.md` |
| Add an optimizer | A file in `src/optim/` implementing `Optimizer`; a variant of `OptimKind` and its `as_str` in `src/optim/mod.rs`; its state tensor names in `src/io/checkpoint.rs`; tests in `tests/optim/`, `tests/folo_io/` (resume), `tests/gpu/main.rs`, and `crates/folo-cli/tests/studio_cli.rs`. To make training use it: `OptimizerChoice` and `ResolvedOptim` in `crates/folo-vision/src/train/config.rs`, the constructor match in `crates/folo-vision/src/train/setup.rs`, `Optimizer` and `ResolvedOptimizer` in `crates/folo-api/src/runs.rs`, `OptimizerArg` in `crates/folo-cli/src/cli.rs` and its match in `config_of` (`crates/folo-cli/src/train/config.rs`), the `config.optimizer` match in `train_command` (`crates/folo-server/src/cli.rs`). Studio: `Optimizer` and `ResolvedTrain` in `web/src/lib/api/types/runs.ts`, `OPTIMIZER` in `web/src/routes/run/resolved.ts`, the option lists in `web/src/routes/runnew/draft.ts` and `TrainingSection.tsx`. Desktop's exhaustive matches: `Label for Optimizer` in `apps/desktop/src/api/types.rs`, `resolve` in `apps/desktop/src/api/mock/training.rs` (also on `OptimKind`), `apps/desktop/src/screens/runs/config.rs` and `charts.rs`, `apps/desktop/src/overlays/new_run/mod.rs`, and `OPTIMIZERS` in `apps/desktop/src/app/forms.rs`. Docs: MODEL.md §8.2, FORMAT.md (`folo.optim`), API.md |
| Change the LR schedule | `src/sched.rs`; `tests/optim/`; MODEL.md §8.4 and FORMAT.md (`folo.sched`) |
| Change `.folo` metadata or the loading rules | `src/io/meta.rs` or `src/io/file.rs`; `tests/folo_io/`; FORMAT.md |

## Tests and benchmarks

```sh
cargo test -p folo-nn                       # unit tests, doctests and tests/
cargo test -p folo-nn --test gpu            # the GPU tests only
cargo bench -p folo-nn --bench nn
cargo bench -p folo-nn --bench train_step   # -- --steps N, or -- --loop for a profiler
```

`tests/gpu/` needs a hardware GPU adapter (Vulkan, DX12 or Metal) and skips with a note without one;
`FOLO_GPU_TEST_SOFTWARE=1` runs it on a software adapter. No test in this crate is `#[ignore]`d.

## Further reading

- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [NN.md](../../docs/NN.md): the overview of this crate.
- [MODEL.md](../../docs/MODEL.md) §2, §8 and [FORMAT.md](../../docs/FORMAT.md): the specs.
- Neighbours: [folo-core](../folo-core/README.md) below, [folo-gpu](../folo-gpu/README.md) for the GPU kernels,
  [folo-vision](../folo-vision/README.md) above.
