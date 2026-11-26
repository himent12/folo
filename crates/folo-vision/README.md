# folo-vision

folo-vision is the FOLO detector and everything that works with detections: the model family `folo-n/s/m/l/x`, the
detection loss and the task-aligned assigner, box maths, decoding, NMS and box fusion, the metrics, the trainer, the
validator, the predictor and ONNX export. It depends on folo-core (tensors), folo-gpu (the assigner and loss kernels
on a GPU), folo-nn (layers, optimizers, `.folo` files), folo-data (datasets and the loader) and folo-api (the run
event types of docs/API.md). folo-bench, folo-server, folo-cli, folo-wasm and FOLO Desktop (`apps/desktop`) depend on
it. The box maths, decode, NMS, metrics and assigner work on plain slices, with no tensor types.
[MODEL.md](../../docs/MODEL.md) is the spec it implements, [METRICS.md](../../docs/METRICS.md) defines the metrics,
and [VISION.md](../../docs/VISION.md) is the tour of the implementation.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | Crate docs and the re-exports |
| `src/model/` | `mod.rs` (`FoloModel`: construction, forward, `fuse`, `to_device`; `HeadOutput`), `blocks.rs` (`Bottleneck`, `C3k`, `C3k2`, `Sppf`, `Attention`, `PsaBlock`, `C2Psa`), `head.rs` (the detect head, and the `o2o` branch), `v2/` (FOLO v2-Iso architecture: `config.rs`, `blocks.rs` with `StemV2`/`WsQARepConv3x3`/`FoloBlock`/`HyperContext`/`FullPadGate`, `head.rs` with `LazyAlignHead`/`PeakSuppress` and opt-in `iou_out`, `iou.rs` with `iou_quality_loss` and `fuse_cls_iou_scores`, `mod.rs` with `FoloV2Model`), `io.rs` (`.folo` weight IO), `predict.rs` (decode and `predict_raw`), `lazy.rs` / `lazy_gpu.rs` (candidate-first prediction on the CPU / a GPU), `quant/` (int8 inference: `walk.rs` the quantised graph, `calib.rs` calibration and the stored `Calibration`, `exec.rs` the int8 executor on folo-core's `cpu::q8`, `fake.rs` the fake-quant reference) |
| `src/boxes.rs` | `Xyxy` / `Cxcywh`, letterbox mapping, IoU / GIoU / DIoU / CIoU |
| `src/decode.rs` | Anchor grids (`make_anchors`) and the DFL decode of raw head outputs (including 2×8 AVX2 `reg_max == 16` fast path `avx2_16`) |
| `src/assign.rs` | The task-aligned assigner and the DFL target encoding |
| `src/loss/` | `mod.rs` (`detection_loss`, `LossConfig`, `ClsLossKind`, `BoxLossKind`, `dual_loss`), `assignment.rs` (host assignment), `terms/` (`mod.rs`, `cls.rs` for BCE/MAL/VFL/QFL, `box_loss.rs` for CIoU/Inner-CIoU/WIoU v3/R-PIoU/S-AEIoU/NWD, `geom_loss.rs` for MPDIoU/Shape-IoU/Focaler-IoU, `siou_eiou.rs` for EIoU/SIoU, `dfl.rs` for OT-DFL/CRPS-DFL), `device.rs` (the folo-gpu kernels), `distill/` (the opt-in distillation term, `fused.rs` its fused folo-gpu path, and their tests), `tests.rs`, `tests_math.rs` (`S-AEIoU` and `CRPS-DFL` finite-difference gradient and edge-case tests), `tests_geom.rs` (`MPDIoU`, `Shape-IoU`, and `Focaler-IoU` finite-difference gradient and edge-case tests), `tests_siou_eiou.rs` (`EIoU` and `SIoU` finite-difference gradient and edge-case tests) |
| `src/nms/` | `mod.rs` (NMS config/pipeline and ELC / consensus / DFL-consensus / DFL-repulse / DFL-CGR post-processing), `kernel.rs` (AVX2-accelerated standard IoU NMS, DIoU-NMS, Soft-NMS, and class-aware bucketing), `matrix.rs` (Linear & Gaussian Matrix-NMS), `topk.rs` (the NMS-free top-k of the `o2o` head), `tests.rs` |
| `src/fusion/` | `mod.rs` (cluster box fusion after NMS — `vote`, `dfl`, `sebf` — opt-in through `NmsConfig::fusion`, and the `consensus` / `elc` / `dfl-consensus` / `dfl-repulse` / `dfl-cgr` score modes opt-in through `NmsConfig::score`), `wbf.rs` (multi-model / TTA Weighted Boxes Fusion), `tests.rs` |
| `src/metrics.rs` | `Evaluator`: mAP, precision / recall / F1, PR curves, confusion matrix |
| `src/calibration.rs` | Post-hoc Temperature Scaling (`TemperatureScaler`, damped Newton-Raphson convex NLL minimization in inverse temperature `1/T`) and calibration metrics (ECE, MCE, Brier score, NLL) |
| `src/cocoeval.rs` | `CocoEval`: the full pycocotools `COCOeval` (crowd regions, area buckets, AR) |
| `src/train/` | `mod.rs` (`Trainer`), `config/` (`mod.rs` for `TrainConfig`, `tests.rs`), `setup.rs` (new and resumed runs), `epoch.rs` (one epoch), `run.rs` (the run loop), `checkpoint.rs`, `writer/` (epoch files off the training thread), `events.rs`, `feed.rs` (batches staged for a GPU), `device.rs`, `teacher.rs` (the distillation teacher), `gauge.rs` (BN-Conv Floor Gauge Normalisation), `tests.rs` |
| `src/val.rs` | `validate`: a model on a dataset split, scored with `Evaluator` |
| `src/predict.rs` | `Predictor`: `.folo` weights to detections in image pixels, with a timing breakdown |
| `src/export/` | `onnx/`: `mod.rs` (`export`, `OnnxOptions`, the op table), `graph.rs`, `blocks.rs`, `decode.rs`, `proto.rs` (protobuf encoder) |
| `src/precompile.rs` | GPU pipeline precompile for a run's kernels |
| `src/error.rs` | `Error`: the trainer's, the validator's and the predictor's |
| `tests/` | Integration tests, listed below; `common/` (helpers, also used by benches), `golden/folo-n-names.txt`, `fixtures/` (`coco/`, `coco_full/`: from `bench/parity/coco_reference.py`; `fusion/`: from `bench/fusion/make_fixture.py`) |
| `benches/` | Benchmarks, listed below |
| `examples/candidate_dump.rs` | Dumps a model's pre-NMS candidates on a COCO split, for `bench/fusion/` |

## From MODEL.md to the code

| MODEL.md | Code |
|---|---|
| §2.1 Conv | `ConvBnAct` in folo-nn `src/layers/conv_bn_act.rs` |
| §2.2 to §2.7 blocks | `src/model/blocks.rs` |
| §2.8 Glue | `Upsample`, `Concat` in folo-nn `src/layers/simple.rs` |
| §2.9 Initialisation | folo-nn `src/init.rs`; `Init` in `src/model/blocks.rs` |
| §3 Scaling and model config | `ModelConfig` and `scale_row` in folo-nn `src/io/meta.rs` |
| §4 Layer tables | `FoloModel` in `src/model/mod.rs`; mirrored for ONNX in `src/export/onnx/blocks.rs` |
| §5 Parameter names | `src/model/blocks.rs`, `head.rs`; pinned by `tests/golden/folo-n-names.txt` |
| §6.1, §6.3 Head branches and raw outputs | `src/model/head.rs`, `HeadOutput` in `src/model/mod.rs` |
| §6.2 Anchors, §6.4 Decode | `src/decode.rs`; `src/model/predict.rs` |
| §6.5 Bias priors | folo-nn `src/init.rs`, `src/model/head.rs` |
| §6.6 One-to-one head | `src/model/head.rs`, `src/assign.rs`, `dual_loss` in `src/loss/mod.rs`, the top-k in `src/nms/topk.rs` |
| §7.1, §7.3 Assigner | `src/assign.rs`, `src/loss/assignment.rs` |
| §7.2 CIoU, §7.4 Terms | `src/boxes.rs`, `src/loss/terms/`; on a GPU `src/loss/device.rs` and folo-gpu's `src/detect.rs` |
| §8.1 Defaults, §8.2 Optimizer | `src/train/config/`, `src/train/setup.rs`; folo-nn `src/optim/` |
| §8.3 Step loop | `src/train/epoch.rs` |
| §8.4 Schedule, §8.5 EMA | folo-nn `src/sched.rs`, `src/ema.rs` |
| §8.6 Data and augmentation | folo-data `src/augment/`, `src/loader.rs` |
| §8.7 Validation and checkpoints | `src/val.rs`, `src/train/run.rs`, `src/train/checkpoint.rs`, `src/train/writer/` |
| §8.8 Mixed precision | `src/train/epoch.rs`; folo-nn `src/amp.rs` |
| §8.9 Distillation | `src/loss/distill/`, `src/loss/mod.rs` (`step_loss`), `src/train/teacher.rs`, `src/train/epoch.rs` |
| §9.1 Fusion | `FoloModel::fuse` in `src/model/mod.rs`; folo-nn `fuse_conv_bn` |
| §9.2 Letterbox | `src/predict.rs`; folo-data `src/letterbox.rs` |
| §9.3 NMS, §9.4 Box fusion | `src/nms/` (`mod.rs`, `kernel.rs`, `topk.rs`), `src/fusion/mod.rs` |
| §9.5 Candidate-first evaluation | `src/model/lazy.rs`, `src/model/lazy_gpu.rs` |
| §9.6 int8 inference | `src/model/quant/`, `FoloModel::set_int8`; folo-core `src/cpu/q8/` |
| §10 Verification plan | `tests/model.rs` (§10.1, §10.2), `tests/assign_cases.rs` and `tests/loss_cases.rs` (§10.3), `tests/train/` (§10.4) |

## Start here

1. `FoloModel` in `src/model/mod.rs`: the detector, its forward pass and `HeadOutput`.
2. `Predictor` in `src/predict.rs`: letterbox, forward, decode, NMS, rescale; what `folo predict` runs.
3. `detection_loss` in `src/loss/mod.rs`: its docs walk through the loss step by step against MODEL.md §7.
4. `Trainer` in `src/train/mod.rs` (its docs describe a run), then `setup.rs`, `epoch.rs` and `run.rs`.
5. `Evaluator` in `src/metrics.rs` and `validate` in `src/val.rs`: how a model is scored.
6. `export` in `src/export/onnx/mod.rs`: the ONNX graph, op by op.

## Common changes

| To do this | Touch these files |
|---|---|
| Change a block or a layer table | `src/model/blocks.rs` or `src/model/mod.rs`; the same change in `src/export/onnx/blocks.rs`; `tests/model.rs` (when names change, re-bless `tests/golden/folo-n-names.txt` with `FOLO_BLESS=1 cargo test -p folo-vision --test model golden_names`); the parameter count in the `src/model/mod.rs` doctest; MODEL.md §2, §4, §5 |
| Add a training recipe key | The field and its default in `TrainConfig` (`src/train/config/mod.rs`); where it acts (`setup.rs`, `epoch.rs` or `run.rs`); `tests/train/`; MODEL.md §8.1. To set it over the API: `TrainConfig` in `crates/folo-api/src/runs.rs`, `TrainConfig::from_api`, and API.md |
| Change a loss term | `src/loss/terms/`; on a GPU, folo-gpu's `src/detect.rs`, `src/kernels/detect.rs` and the shaders `det_bce_fwd.wgsl`, `det_bce_bwd.wgsl`, `det_box.wgsl` and `detect_common.wgsl` behind `src/loss/device.rs`; `tests/loss_cases.rs` and `tests/gpu_loss/`; MODEL.md §7.4 |
| Change the post-process | `src/nms/` (`mod.rs`, `kernel.rs`, `topk.rs`) or `src/fusion/mod.rs`; `tests/nms_reference.rs`, `tests/fusion_reference.rs`, `tests/elc_postproc.rs`, and `tests/lazy_head.rs` / `tests/gpu_lazy_head.rs` (candidate-first must equal the dense path); MODEL.md §9.3, §9.4 |
| Change a metric | `src/metrics.rs` or `src/cocoeval.rs`; `tests/metrics_known.rs`, `tests/coco_parity.rs`, `tests/coco_full.rs`; METRICS.md |
| Change the ONNX export | `src/export/onnx/graph.rs` (layer helpers), `blocks.rs` or `decode.rs`; the unit tests and the op table in `src/export/onnx/mod.rs`; CLI.md (`folo export`) |

## Tests and benchmarks

```sh
cargo test -p folo-vision                    # unit tests, doctests and tests/
cargo test -p folo-vision --test train       # one integration test binary
```

| Test | Covers |
|---|---|
| `model.rs`, `model_v2.rs` | Blocks and the whole model (MODEL.md §10.1, §10.2), the golden tensor names, and FOLO v2-Iso blocks/head/fusion/GPU parity |
| `assign_cases.rs`, `loss_cases.rs`, `wave7_train_algos.rs` | The assigner and the loss on hand-checked cases, against an `f64` reference, gradient checks, and Wave-7 MAL / NWD-IoU / CWD / Floor Gauge / BCKD parity |
| `o2o.rs` | The one-to-one head: assignment, dual loss, `.folo` round trips, the top-k post-process |
| `nms_reference.rs`, `fusion_reference.rs`, `elc_postproc.rs` | NMS against a naive reference; DFL fusion against `bench/fusion/post.py`; ELC, DFL-consensus, and DFL-repulse post-processing |
| `metrics_known.rs`, `coco_parity.rs`, `coco_full.rs` | The evaluator against hand-computed answers and against pycocotools |
| `lazy_head.rs` | Candidate-first prediction on the CPU, bit for bit against the dense path |
| `int8.rs` | int8 inference: the executor against the fake-quant reference, the same bits at every thread count, the calibration's JSON, metadata and weights fingerprint |
| `train/` (`main.rs`, `ema.rs`) | The trainer on tiny synthetic sets: learning, checkpoints, bit-exact resume, legacy & `PC-PEMA` EMA contracts/resumes, the predictor on trained weights |
| `distill.rs` | Distillation in the trainer: a run without a teacher pinned to its pre-distillation weights, determinism, resume with the teacher, incompatible teachers |
| `gpu.rs`, `gpu_f16.rs`, `gpu_tensor_cores.rs`, `gpu_in_place.rs`, `gpu_plan.rs`, `gpu_lazy_head.rs`, `gpu_loss/` (`main.rs`, `new_math.rs`, `math_winners.rs`), `gpu_train.rs`, `gpu_amp.rs`, `gpu_distill.rs` | The same on the GPU against the CPU: forward, f16, tensor cores, forward plans, candidate-first & fused `dfl_stats`, the loss (including `S-AEIoU`, `CRPS-DFL`, and GPU-fused `VFL`/`QFL` parity), training, mixed precision, distillation (and its fused kernel against the tensor path) |

The `gpu*.rs` tests need a hardware GPU adapter and skip with a note without one (`FOLO_GPU_TEST_SOFTWARE=1` runs
most of them on a software adapter); `gpu_f16.rs` also skips without f16 compute, and `gpu_tensor_cores.rs` and the
GPU tests of `gpu_amp.rs` without tensor cores. Two tests are `#[ignore]`d:

- `tests/train/main.rs` `overfit_eight_images`: the MODEL.md §10.4 overfit run, minutes on a desktop CPU:
  `cargo test --release -p folo-vision --test train -- --ignored overfit`.
- `gpu.rs` `letterbox_on_device_splits_a_batch_above_the_buffer_limit`: uploads more than the device's buffer limit
  (about 2 GiB).

Benchmarks (`cargo bench -p folo-vision --bench <name>`; each file's header lists its environment variables).
A change to a hot path reports numbers measured against the build before it; CONTRIBUTING.md names `gpu`, `cpu_e2e`
and `train_step`.

| Bench | Measures | Needs |
|---|---|---|
| `gpu` | The detector on the GPU: forward, predict and training step per scale and precision | GPU |
| `cpu_e2e` | The CPU predictor end to end at batch 1 | |
| `int8_e2e` | int8 against `f32` CPU prediction end to end (rows 7 and 8), in-process alternation | `FOLO_BENCH_WEIGHTS` (a `folo quantize` output), `FOLO_BENCH_IMAGES` |
| `train_step` | One CPU training step: folo-n, 4 classes, 320 px, batch 16 (MODEL.md §10.6's configuration) | |
| `model` | The CPU forward pass (unfused and fused), `predict_raw` and the predictor | |
| `post` | Decode + NMS, the `o2o` top-k, batched NMS, the assigner, the evaluator | |
| `loss` | `assign_batch`, the loss forward, forward + backward, on the CPU | |
| `gpu_profile` | The GPU time of a fused inference forward, dispatch by dispatch | GPU |
| `kd_step` | The distillation term of one COCO batch: the tensor path against the fused kernel (`FOLO_GPU_PROFILE=1`: per kernel) | GPU |
| `gpu_latency` | Where the time of a synchronised batch-1 forward goes, call by call | GPU |
| `dispatch_ab` | Host cost of a fused GPU forward with the bind-group cache on and off | GPU |
| `plan_gate` | A fused GPU forward against replays of its recorded trace | GPU |
| `gpu_trained` | GPU prediction with trained weights on real images | GPU, `FOLO_BENCH_WEIGHTS`, `FOLO_BENCH_IMAGES` |
| `candidates` | Candidate-first prediction with trained weights on real images | `FOLO_BENCH_WEIGHTS`, `FOLO_BENCH_IMAGES` |
| `head` | The `o2o` head (top-k) against the one-to-many head with NMS | an `o2o` model in `FOLO_BENCH_WEIGHTS`, `FOLO_BENCH_IMAGES` |

## Further reading

- [VISION.md](../../docs/VISION.md): this crate in depth. [MODEL.md](../../docs/MODEL.md) and
  [METRICS.md](../../docs/METRICS.md): the specs it implements.
- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [FORMAT.md](../../docs/FORMAT.md) for `.folo` files, [CLI.md](../../docs/CLI.md) for `folo train / val / predict /
  export`.
- Neighbours: [folo-nn](../folo-nn/README.md), [folo-data](../folo-data/README.md),
  [folo-gpu](../folo-gpu/README.md), [folo-bench](../folo-bench/README.md), [folo-wasm](../folo-wasm/README.md).
