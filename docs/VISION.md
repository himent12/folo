# folo-vision: the detector, its training and its inference

`crates/folo-vision` is the FOLO detector family and everything around it: the model code, the loss on the CPU and
the GPU, box ops, NMS and metrics, the `Trainer`, the `Predictor` and ONNX export. This file is the implementation
tour. [MODEL.md](MODEL.md) is the normative spec of the detector (blocks, layer tables, parameter names, head, loss,
training recipe), [METRICS.md](METRICS.md) defines how mAP is computed, and neither is repeated here; the building
blocks below the model are in [CORE.md](CORE.md), [NN.md](NN.md), [DATA.md](DATA.md) and [GPU.md](GPU.md). The
crate's README has its file map, its common changes and a table [from each MODEL.md section to the
code](../crates/folo-vision/README.md#from-modelmd-to-the-code).

```text
model/             FoloModel: blocks.rs (MODEL.md §2), head.rs (§6), io.rs (.folo through folo-nn),
                   predict.rs (decode and the raw prediction path), lazy.rs / lazy_gpu.rs (candidate-first head),
                   quant/ (int8 inference, §9.6), v2/ (FOLO v2-Iso clean-sheet model family)
loss/              the detection loss: assignment.rs (host assigner), terms/ (BCE/MAL/VFL/QFL,
                   CIoU/R-PIoU/S-AEIoU/Inner-CIoU/WIoU v3/NWD, DFL/OT-DFL/CRPS-DFL), device.rs (GPU),
                   distill/ (dense/BCKD/GT-verified distillation, §8.9)
assign.rs          the task-aligned assigner (including eps-free TAL & NWD-IoU) and DFL target encoding
boxes.rs, decode.rs, nms/{mod.rs, kernel.rs, topk.rs}, fusion/{mod.rs, tests.rs}   boxes and IoU, anchors and DFL decode, NMS (standard/AVX2, DIoU-NMS, Soft-NMS) / o2o top-k, cluster fusion (vote/DFL/SEBF) & ELC/DFL-consensus/DFL-repulse/DFL-CGR
metrics.rs, cocoeval.rs                  Evaluator (mAP, PR curves, confusion matrix), the full COCOeval
train/             Trainer: config/, setup (new / resume), run (epochs), epoch, feed (GPU staging), device,
                   teacher (distillation), gauge (Floor Conv+BN Gauge Normalization), checkpoint, events, writer/
val.rs             validate: a model on a dataset split
predict.rs         Predictor
export/onnx/       ONNX export: graph.rs, blocks.rs (mirrors forward), decode.rs, proto.rs
precompile.rs      GPU pipeline precompile for a run's kernels
```

The detection math (boxes, decode, NMS, metrics, assigner) works on plain slices and small `Copy` structs with no
tensor types, so the trainer, the predictor, the server and the browser share it. The loss is the exception: it is
differentiable, so it works on folo-core tensors.

## The model
- **Architecture** (per scale `n/s/m/l/x` via depth and width multipliers): a CSP-style backbone (Conv-BN-SiLU stem,
  C3k2 blocks with split + bottlenecks + concat), SPPF, one C2PSA attention block at P5, a PAN-FPN neck, and an
  anchor-free decoupled head on three scales (strides 8 / 16 / 32). Box regression uses distribution focal loss bins
  (`reg_max`, default 16).
- `head = "o2o"` adds the NMS-free one-to-one branch (`head.o2o.*`, MODEL.md §6.6, as in YOLOv10 / YOLO26): trained
  with dual assignment on detached features, it predicts with a top-k selection instead of NMS
  (`nms::topk_detections`), and only the predicting branch runs in eval mode (`FoloModel::head_mode`).
- Implemented in `folo_vision::model`: `FoloModel::new(config)` for any scale and class count,
  `forward` → `HeadOutput { box_raw [B, 64, A], cls_raw [B, nc, A], anchors }`, `fuse()`, `predict_raw` (forward +
  decode + NMS; decode runs on the outputs' device, NMS on the CPU), `predict_detections` / `HeadOutput::detections`
  (on a GPU the decode runs in one kernel that keeps only the candidates above `conf`, and only those are
  downloaded), `to_device()`, and `.folo` save / load (`open_on(path, device)`) / transfer through folo-nn.

### FOLO v2-Iso (`model/v2/`)
Clean-sheet, hardware-co-designed detector family (`folo2-n/s/m/l/x`) built on standard CPU/GPU autograd primitives and
ONNX opset-17-expressible ops (`export::onnx` and `model::quant` currently target v1 `FoloModel`):
- **Config & scaling (`config.rs`):** `FoloV2Config` and `ScalingV2` enforce `ceil16`-aligned widths at every stage
  and lock shared head widths (`c_box = 48`, `c_cls = 64`) across `n` and `s` for zero-adapter CrossKD; `plus = true`
  selects the deeper `v2-Plus` tier.
- **Blocks (`blocks.rs`):** `StemV2` (`space_to_depth_2x2` + `Conv1×1`, or `Conv3×3 s=2` when `spd_stem = false`),
  `WsQARepConv3x3` (sums `3×3 + 1×3 + 3×1 + 1×1`, plus per-channel `γ_id·I` when `c_in == c_out`, in weight space
  before one post-BN + SiLU during training, and collapses into a single dense `3×3` conv on `fuse()`), `FoloBlock`
  (`ceil16`-aligned QARep-CSP / Rep-ELAN whose `cv2` reads at most three `forward_cat` parts, eliding concatenation on
  fused paths), `HyperContext` (linear `O(N·M)` P5 hypergraph global context replacing C2PSA, returning `(P5, g_scene)`,
  with softmax, FFN residual, and scene mean computed in `F32` for stability under `F16`/AMP),
  and `FullPadGate` (zero-init residual FiLM gate `feat * (1 + tanh(proj(g_scene)))`, multiplied in `F32` before casting
  back to `feat.dtype()` so small gate updates are not rounded away under `F16`/AMP).
- **Backbone, SPD-PAN neck & `LazyAlignHead` (`mod.rs`, `head.rs`, `iou.rs`):** `NeckV2` projects lossless
  `space_to_depth_2x2(P2)` through `p2_proj` (`4·c1 → c1`) into P3 (keeping 8,400 anchors at 640×640) and applies
  `pad_p3`/`pad_p4` strictly to the `P3`/`P4` head inputs *after* `neck.down3`/`neck.down4` taps so the bottom-up neck
  stays free of float barriers by design. `LazyAlignHead` pairs a depthwise-free dense `ClsBranchV2` with `BoxBranchV2`
  (`2 × WsQARepConv3x3` + `4·reg_max` `out` — `64` channels at default `reg_max = 16` — and opt-in separate
  `iou_out: Conv1×1(c_box → 1)` with standalone `iou_quality_loss` helper to supervise it against detached
  predicted-vs-GT plain IoU and `fuse_cls_iou_scores` `p_cls^(1 − α) · p_iou^α` used by `HeadOutputV2::detections` /
  `decode_images` with an explicit `alpha`, e.g. `DEFAULT_IOU_FUSION_ALPHA = 0.5`), plus optional `o2o`
  (`O2oClsLevel` with `PeakSuppress` 3×3 local-peak-preserving lateral inhibition sharing `BoxBranchV2`).
  `HeadOutputV2::into_head_output()` converts directly to v1's `HeadOutput` (dropping `iou_raw`).

### int8 inference (`model/quant/`)
Post-training CPU int8 quantisation of the v1 backbone and neck (layers 0–22; the head stays `f32`, MODEL.md §9.6).
A shared graph walk (`walk.rs`) drives three executors:
- **Calibration (`calib.rs`):** `Calibrator` runs two `f32` passes over sample images (min/max, then 2048-bin
  per-tensor and 256-bin per-channel histograms) and `Calibrator::finish` returns a `VERSION_V2` `Calibration` with
  per-channel `channel_ranges` and `u8_stem: true` (`f32_convs` empty, `weights: None`). `calibrate` attaches the
  64-bit FNV-1a weight fingerprint, `calibrate_sample` also adds the P5-tail `default_sensitivity_f32_convs`
  (`neck.out5.*`), and `Calibration::store` serializes `folo.quant` (used by `folo quantize`). Loading remains
  backward-compatible with `VERSION_V1` per-tensor files.
- **Execution (`exec.rs`):** `Int8Plan` executes on `folo_core::cpu::q8` C4 buffers and `QConv` kernels (AVX-VNNI
  when available, else portable scalar fallback; with per-part concat scale decoupling, packed weights cached on
  each weight tensor via `Tensor::derived_f32`, bias fingerprints cached on each bias tensor, and cached uniform
  grids for `VERSION_V1` checkpoints).
  Activations produced by `f32` convs (such as `neck.out5.*` in `f32_convs`), an `f32` stem, or deferred
  `quantize` boundaries stay in unquantised `f32` until an int8 conv consumes them (avoiding redundant
  `f32 → u8 → f32` ping-pong across consecutive `f32` convs and into the `f32` head), and the head-only P5 output
  (`neck.out5.cv2`) writes `f32` directly (`QOut::F32` when int8).
  In `Attention` (`blocks.rs`, shared with `walk.rs`), eval-mode CPU `F32` forwards fuse `q^T k`, in-place scaled
  softmax, `v @ attn^T`, and depthwise 3×3 `pe(v)` accumulation without intermediate Q/K/V/transpose/PE allocations.
  When `u8_stem` is enabled, the stem is eligible (`stem_int8_able`), and it is not listed in `f32_convs`,
  `features_rgb0_u8` runs `q8::stem_u8` directly on packed `[R, G, B, 0]` bytes from `Letterbox::apply_rgb0_u8`
  without materialising an `f32` image (falling back to `f32` otherwise).
- **Reference (`fake.rs`):** `FakeQuant` runs `f32` convolutions on the same quantisation grids and lazy `f32`
  boundaries for parity checks.

## The loss
- Task-aligned assigner (TAL: `score^alpha * max(CIoU, 0)^beta`, top-k), BCE classification on aligned targets,
  CIoU box loss and DFL, weighted 7.5 / 0.5 / 1.5 by default.
- On the CPU `folo_vision::loss` (`loss/assignment.rs`, `assign.rs`) runs a grid-indexed, candidate-only TAL
  assigner on the detached head outputs: each ground-truth box directly indexes only the anchor rectangle covering its
  centre region per feature level (`level_candidate_rect`), and `decode_masked_channel_major` lazily decodes DFL box
  logits (streaming active runs linearly with `fast_exp`) only at anchors inside at least one GT candidate mask —
  bit-identical to a full-grid decode while skipping background anchors. It then builds the loss terms from folo-core
  tensor ops (fused custom ops for the full-size BCE and gather), so gradients flow into both head branches.
- On a GPU, when `device_loss_supported` holds (`cls_loss ∈ {Bce, Vfl, Qfl}`, `box_loss ∈ {Ciou, Rpiou, Saeiou}`,
  `topk ∈ 1..=64`, `topk2 != Some(0)`, `reg_max ∈ 2..=32`), the whole loss runs as folo-gpu kernels
  (`folo_gpu::detect`: assigner, fused BCE/MAL/VFL/QFL classification, fused CIoU/R-PIoU/S-AEIoU + DFL/OT-DFL/CRPS-DFL
  with a hand-written backward, as one custom op) and only the reported items are read back.
- **Opt-in assigner, loss & distillation options (`assign.rs`, `loss/terms/`, `loss/device.rs`, `loss/distill/`):**
  all default to off:
  - *Assigner (CPU + GPU parity):* `TalConfig.eps_free` removes `eps` attenuation on small-box targets via
    `(m / m_max) * u_max`; `TalConfig.nwd_weight` blends Plain-IoU with Normalized Wasserstein Distance (`C = 12.8 px`)
    using scale weight `λ(s_g) = λ₀ · exp(−(s_g / 24)²)`.
  - *Classification terms:* `LossConfig.mal_gamma > 0` enables Matchability-Aware Loss (`p^γ · softplus(x)`
    on background negatives, unweighted BCE on positives; requires `cls_loss == Bce`; CPU + GPU fused);
    `LossConfig.cls_loss = ClsLossKind::Vfl` selects Varifocal Loss (quality-weighted `t · BCE(x, t)` on positives and
    `vfl_alpha · p^vfl_gamma · softplus(x)` on negatives; defaults `0.75` / `2.0`, requires `mal_gamma == 0`; CPU + GPU
    fused in `det_bce_fwd.wgsl` / `det_bce_bwd.wgsl`); `ClsLossKind::Qfl` selects Quality Focal Loss
    (`|t − p|^qfl_beta · BCE(x, t)`, default `qfl_beta = 2.0`, requires `mal_gamma == 0`; CPU + GPU fused in
    `det_bce_fwd.wgsl` / `det_bce_bwd.wgsl`).
  - *Box & DFL terms:* `LossConfig.box_loss` selects `Ciou` (default), `Rpiou` (purely algebraic Rational-Focal
    Edge-Decoupled IoU: `rpiou_beta`, `rpiou_kappa`, `rpiou_delta`; CPU + GPU fused), `Saeiou` (Scale-Normalized
    Algebraic Edge IoU, `saeiou_loss`: scale-normalized focal-IoU + shape-weighted per-edge inverse-sqrt penalty
    reusing `rpiou_beta = 0.5`, `rpiou_kappa = 2.0`, `rpiou_delta = 0.5`, zero `atan`/`exp`/exponent-split; CPU + GPU
    fused), `InnerCiou` (Inner-CIoU using auxiliary boxes scaled by `inner_ratio` in `[0.5, 1.5]`, default `0.7`),
    or `WiouV3` (Wise-IoU v3 dynamic non-monotonic focusing: `wiou_alpha = 1.9`, `wiou_delta = 3.0`,
    `wiou_momentum = 0.01`, running `L_IoU_bar` persisted in checkpoints as `folo.wiou_iou_mean`);
    `LossConfig.nwd_weight` adds the scale-adaptive NWD box blend (using the `sqrt(W_2² + 1e-6)` smoothed form
    matching `det_box.wgsl` for CPU/GPU parity); `LossConfig.dfl_ot_weight > 0` adds an exact `O(R)`
    log-space 1D optimal-transport cumulative tail barrier to DFL (when `reg_max > 2`: host `f64` custom op
    `ot_dfl_tail` on CPU, fused in `folo_gpu::detect` on GPU when `device_loss_supported` holds, or host `f64` readback
    on the GPU fallback path when foreground anchors exist); `LossConfig.dfl_crps_weight > 0` adds a closed-form `O(R)`
    1D excess CRPS / squared Energy Distance term over dual-ended CDF masses `F_k` and `S_k` (any `reg_max >= 2`: host
    `f64` custom op `crps_dfl` on CPU, fused in `det_box.wgsl` on GPU when `device_loss_supported` holds, or host `f64`
    readback on the fallback path). `InnerCiou` and `WiouV3` run through the host-assigner + `loss_with_assignment`
    path (device-generic tensor composites with foreground host `atan` / focusing-gain readback; `Vfl` and `Qfl` only
    fall back there when paired with a host-only box loss or out-of-range `topk`/`reg_max`).
  - *Distillation (`loss/distill/`):* dense teacher output KD (MODEL.md §8.9, fused GPU path in `distill/fused.rs`),
    plus `DistillConfig.bckd_gamma > 0` (BCKD focal disagreement weighting `sg(|p_S − p_T|^γ)` with positive teacher
    clamping) and `DistillConfig.gt_verify` (positive-only GT-verified DFL KD gating where `IoU(b_T, g) > IoU(b_S, g) +
    0.02` and `IoU(b_T, g) > 0.50`, weighted by `IoU_T − IoU_S`).

## Box ops, NMS and metrics
They work on plain slices so they are reusable everywhere: IoU / GIoU / DIoU / CIoU (`boxes::iou` computes enclosing
boxes and overlaps in `f64` for stability on degenerate, tiny, and huge boxes; `boxes::ciou_eps` preserves the exact
§7.2 `f32` order on valid in-range boxes `≤ 1e15` with an `f64` guard outside it), xywh↔xyxy, DFL decode (`decode.rs`:
on `x86_64` with runtime-detected AVX2, `avx2_16` accelerates `reg_max == 16` `softmax_expectation`, `softmax_moments`,
and `softmax_moments_and_entropy` in two 8-lane registers using degree-5 minimax `exp_neg8` with no libm calls, falling
back to scalar on non-finite logits; `folo-vision --bench post` `decode_head` median `1.049 ms`, `TAL assign`
`0.215 ms`), class-aware batched NMS (`nms/kernel.rs`: AVX2-vectorised SoA suppression loops on x86-64, plus opt-in
`NmsKind::Diou` DIoU-NMS and `NmsKind::{SoftLinear, SoftGaussian { sigma }}` Soft-NMS), COCO-style mAP@0.5 and
mAP@0.5:0.95 (101-point interpolation), precision / recall / F1 curves, confusion matrix.
`Evaluator` (`metrics.rs`) and `CocoEval` (`cocoeval.rs`) reuse scratch buffers across images (packing sort keys into
register-sized `(u64, u32)` tuples sorted in place without indirection) and accumulate AP in a single pass.
[METRICS.md](METRICS.md) defines exactly how mAP is computed.

### Cluster box fusion and calibrated scoring (`fusion/`, `nms/mod.rs`)
Opt-in post-processing on `o2m` NMS clusters (MODEL.md §9.4, configured via `NmsConfig::fusion` and `NmsConfig::score`):
- **Box fusion (`BoxFusion`):** `Vote` (score-weighted box voting), `Dfl` (score × per-side DFL precision from
  `decode::side_precision`), and `Sebf` (Student-t Exclusive Box Fusion: 1-step Student-t `ν = 3` IRLS weighted by
  `score · IoU^4 · side_precision` restricted to each kept box's NMS-owned cluster from `nms_sorted_with_owner`).
- **Score rules (`ScoreRule`):** `Consensus` (cluster mass scaling), `Elc` (Exclusive Logit-Space Consensus preserving
  the `[0, 1]` score scale), `DflConsensus` (adds scale-free fused DFL excess-std), `DflRepulse` (power-weighted
  support `s^1.5`, 4-side DFL Shannon entropy, cross-stride agreement, and greedy sequential log-survival soft
  repulsion `γ = 2.5`), and `DflCgr` (Confidence-Gated Repulsion: `p_m^0.25` suppressor confidence gating with
  `γ = 3.0`, best paired with `Sebf`). Single-label calibrated rules lower pre-NMS candidate extraction to
  `CALIBRATED_PRE_CONF = 0.10` when `conf > 0.10` and post-filter at `conf`.

## The Trainer
`folo_vision::train::Trainer` implements MODEL.md §8.
- **Data to step:** folo-data's `DataLoader` (seeded per epoch and sample, so batches do not depend on the worker
  count) → tensors → train-mode forward → loss → backward into a `GradAccumulator` (nominal batch 64) → schedule
  (warmup + linear / cosine, in optimizer steps), global-norm clipping, SGD / AdamW / MuSGD with `decay` / `bn` /
  `bias` groups (`auto` picks per §8.2), EMA (`TrainConfig.ema_mode`: `EmaMode::Legacy` exponential ramp by default, or
  opt-in `EmaMode::Power` `PC-PEMA` with prior shift `t_0 = 0.0` from scratch or `warmup_iters` /
  `max(round(0.03 * total), 1)` on `--weights` transfer, stepped via `ema.update_with_lf(&self.model, lf)`).
- **Floor Conv+BN Gauge Normalization (`train/gauge.rs`):** opt-in (`TrainConfig.gauge_norm`), applied once at setup
  when a new run starts from transferred weights (skipped on checkpoint resume) via `apply_floor_gauge_norm`. For every
  unfused, bias-free `F32` `ConvBnAct` output channel with `1e-4 < ‖W_c‖₂ < √2`, it rescales `W_c` and `running_mean`
  by `α_c = clamp(√2 / ‖W_c‖₂, 1.0, 4.0)` and sets `running_var` to `α_c² · (v_c + ε) − ε` without altering `γ` or `β`,
  preserving eval-mode forward outputs while undoing coupled weight-decay filter shrinkage.
- **Differential backbone LR & backbone freeze (`train/setup.rs`, `train/epoch.rs`):** opt-in
  `TrainConfig.backbone_lr_scale` (default `1.0`, must be finite and `> 0`) splits `backbone.*` parameters into
  `<group>.backbone` groups at `base_lr * backbone_lr_scale` (also scaling `bias.backbone`'s warmup start from
  `warmup_bias_lr`); `TrainConfig.freeze_backbone_epochs` (default `0`) sets `model.backbone` to eval mode and disables
  `requires_grad` on backbone parameters for epochs `0..freeze_backbone_epochs` (restored via RAII `RestoreGrad` on
  epoch exit).
- **Non-finite batches:** batches with a non-finite loss or BatchNorm statistic are skipped and their running
  statistics restored. At epoch entry, `RunningStats::all_finite_eval` validates any frozen eval-mode BN buffers once
  and `RunningStats::capture_training` allocates snapshot buffers once for active train-mode BNs, refreshing them in
  place with `recapture()` only when marked dirty after a finite batch and checking `all_finite_live()` without
  per-batch snapshot tensor allocations or module-tree walks.
- **After every epoch:** validation runs on a pre-folded eval clone (`FoloModel::fused_eval_clone()`, with GPU
  `ForwardPlans` enabled) of the EMA model (or live model when `ema = false`) so Conv+BN is folded once per epoch
  instead of per batch while leaving the live and EMA models untouched, scored with `metrics::Evaluator` (fitness
  mAP@0.5:0.95), followed by `last.folo` / `best.folo` checkpoints (complete FORMAT.md state, atomic — `best.folo` uses
  an O(1) same-filesystem hardlink + atomic rename on non-Windows with atomic-copy fallback, and an independent atomic
  copy on Windows so open readers on `best.folo` cannot block future `last.folo` writes) and early stopping. Resume from
  a checkpoint is bit-exact.
  On Windows, where the rename over `last.folo` fails while another program (a script, a virus scan) has the file
  open, it is retried for up to 10 s before the run fails with an error saying another program may hold the file
  or its permissions may not allow replacing it.
- **Events** (`status`, rate-limited `step`, `epoch`, `log`: the API.md payloads as `folo_api::RunEvent`) go to a
  callback and to the run's `metrics.jsonl`; the CLI's progress view, `--json` and the server consume them.
- **Devices:** the device is a parameter. On a GPU (`place_model` moves the model; optimizer, EMA and gradient
  buffers follow) a feeder thread stages each `u8` batch for upload while the previous step runs, accumulation and
  clipping stay on the device, and the step waits only for the loss items and one BatchNorm finiteness check.
- **Mixed precision** (`amp`, MODEL.md §8.8; GPU tensor cores only, else `not_supported`): the forward under
  `autocast` (f16 convs and products on tensor cores), the loss in f32 on f32 copies of the head, backward of the loss
  times folo-nn's `GradScaler` scale, and at the end of a window the unscale, clip and overflow check with one flag
  readback (an overflowing window skips its step and backs the scale off). Master weights, optimizer state, EMA and
  BatchNorm statistics stay f32, and the scaler is the checkpoint's `folo.amp`.
- `folo_vision::val::validate` is the same validation for `folo val`.

## The Predictor
`folo_vision::predict::Predictor` loads `.folo` (EMA when present), fuses BN, letterboxes (rect for one image and for a
batch whose images share a rect canvas, square for a batch of mixed canvases), runs the forward (candidate-first: the
box branch only at the candidates, MODEL.md §9.5), decodes, runs NMS and rescales to the original image, with a
preprocess / inference / postprocess timing split.

### On a GPU
- The raw `u8` images are uploaded and letterboxed there (bit-exact with the CPU letterbox) and only the decoded
  candidates come back: the images of a batch share one upload (a batch larger than the device's largest storage
  buffer takes several), one queue of work, one small readback.
- The head is candidate-first too (`model/lazy_gpu.rs`): per level the class branch, the class pass into
  fixed-capacity site lists, `conv0` on the listed windows (scattered back into a map), `conv1` and `out` at the
  candidates, then one decode of every level and one readback of the records. The dispatches do not depend on the
  candidates, and an image whose candidates overflow a list takes the dense head on the same features.
- With DFL box fusion / DFL scoring, `dfl_stats_with` (`model/predict.rs`) dispatches the fused single-pass GPU kernel
  `folo_gpu::detect::dfl_stats` (`det_dfl_stats` / `dfl_stats.wgsl`, `reg_max <= 32`) on the candidates' box-logit
  columns (all site lists are packed with the records into one `u32` buffer, so the whole readback is at most two
  transfers: one `u32` buffer and one combined `f32` stats buffer).

### On the CPU
- The whole prediction runs inside the rayon pool (`cpu::in_pool`).
- The letterbox is one fused pass per image (`Letterbox::apply_planar_f32`: resize, padding and `/ 255` straight into
  planar f32, rows in parallel) into an input buffer from folo-core's buffer cache.
- The neck's Concat layers are never written (`C3k2::forward_cat`).
- The head is evaluated candidate-first (MODEL.md §9.5, `FoloModel::predict_candidates`, `model/lazy.rs`): the three
  class branches run side by side, each level's candidates come straight from its class map (blocks of 1024 anchors
  in parallel), and the three box branches then run only at the candidates (`conv2d_act_at_tiles` for the 3x3 convs
  and `conv2d_act_at` for 1x1 `out`; `conv0` on the input patches `conv1` reads — their 3x3 receptive fields, or the
  6x6 input patches of their 4x4 tiles when `conv1` may take Winograd F(4x4); with DFL box fusion also the candidates'
  side precisions), so the dense box outputs and the
  head's `cat` are never made; the detections are the dense path's bit for bit.
- The candidate extraction and the box branch count as inference, NMS and the letterbox inverse as postprocess.
- `Predictor::predict_batch` runs chunks of two images as concurrent forwards, the same predictions as one pass.

### The dense path
An unfused model, multi-label settings or f16 on the CPU take the dense forward (the six branch chains side by side)
and `HeadOutput::detections`, which reads the head outputs in place and extracts candidates the same way.

### Tiled inference (SAHI)
`folo_vision::tile` wires Slicing Aided Hyper Inference into `Predictor::predict_tiled`,
`Predictor::predict_batch_tiled`, and `Predictor::predict_file_tiled` (`TileConfig`):
- High-resolution images are sliced into uniform overlapping `TileRect` windows (`compute_tiles`, clamped back on the
  trailing edges so every tile shares the same `(width, height)` and `rect` letterbox canvas), cropped with row-slice
  copies (`crop_tile`), and run through `Predictor::predict_batch` in bounded chunks (`tile_batch`), automatically
  inheriting CPU/int8/GPU/f16/`o2o`/candidate-first fast paths.
- Images already fitting within a single tile degrade directly to `Predictor::predict` with zero slicing overhead.
- An optional full-image pass (`full_image`, on by default) preserves large objects spanning tile boundaries.
- `merge_tiled_detections` maps and clips boxes to original-image coordinates, sorts deterministically independent of
  tile order, clusters duplicates via `TileMergeMetric::Ios` (default, suppressing truncated tile-edge fragments) or
  `TileMergeMetric::Iou`, and refines agreeing boxes with `BoxFusion::Vote` (default), `BoxFusion::Sebf`, or
  `BoxFusion::None` without letting IoS-suppressed fragments shrink a full box.

## ONNX export
- `folo_vision::export::onnx`: `FoloModel::to_onnx` turns a fused model into an opset-17 graph that mirrors `forward`
  op for op (raw head outputs, or the MODEL.md §6.4 decode in the graph, which for an `o2o` model ends in the §6.6
  top-k: end-to-end `[B, K, 6]` detections; fixed or dynamic shapes; f32 or f16).
- It is encoded by a small hand-written protobuf writer (`export::onnx::proto`), with the class names and `folo.*`
  metadata in `metadata_props`.
- Apart from the head branch the export does not use (`head.o2o.*`, or `head.box.*` / `head.cls.*` for end-to-end),
  any model tensor no node consumed, or an unfused BatchNorm, is an error naming it.
- `folo export --format onnx` and `POST /api/models/{id}/export` call `export_file`; parity with onnxruntime is
  `crates/folo-cli/tests/onnx.rs` + `bench/parity/onnx_check.py` ([CLI.md](CLI.md#onnx-export)).
- `.folo` export itself lives in folo-nn ([FORMAT.md](FORMAT.md), [NN.md](NN.md)).

## Tests
The integration tests in `crates/folo-vision/tests/` are named for what they protect: `model.rs`, `model_v2.rs`,
`loss_cases.rs`, `assign_cases.rs`, `distill.rs`, `wave7_train_algos.rs`, `nms_reference.rs`, `fusion_reference.rs`,
`elc_postproc.rs`, `metrics_known.rs`, `coco_parity.rs`, `coco_full.rs`, `train/` (`main.rs`, `ema.rs`), `o2o.rs`, `lazy_head.rs`,
`int8.rs`, and the GPU ones (`gpu*.rs`).
