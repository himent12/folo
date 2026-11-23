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
loss/              the detection loss: assignment.rs (host assigner), terms.rs (BCE/MAL, CIoU/NWD-IoU, DFL),
                   device.rs (GPU), distill/ (dense/BCKD/GT-verified distillation, §8.9)
assign.rs          the task-aligned assigner (including eps-free TAL & NWD-IoU) and DFL target encoding
boxes.rs, decode.rs, nms/{mod.rs, topk.rs}, fusion.rs   boxes and IoU, anchors and DFL decode, NMS / o2o top-k, cluster fusion & ELC/DFL-consensus/DFL-repulse
metrics.rs, cocoeval.rs                  Evaluator (mAP, PR curves, confusion matrix), the full COCOeval
train/             Trainer: config, setup (new / resume), run (epochs), epoch, feed (GPU staging), device,
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

## The loss
- Task-aligned assigner (TAL: `score^alpha * max(CIoU, 0)^beta`, top-k), BCE classification on aligned targets,
  CIoU box loss and DFL, weighted 7.5 / 0.5 / 1.5 by default.
- On the CPU `folo_vision::loss` runs the assigner on the detached head outputs, then builds the terms from
  folo-core tensor ops (fused custom ops for the full-size BCE and gather), so gradients flow into both head
  branches.
- On a GPU the whole loss runs as folo-gpu kernels (`folo_gpu::detect`: assigner, fused BCE, fused CIoU + DFL with a
  hand-written backward, as one custom op) and only the reported items are read back.

## Box ops, NMS and metrics
They work on plain slices so they are reusable everywhere: IoU / GIoU / DIoU / CIoU, xywh↔xyxy, class-aware batched
NMS, COCO-style mAP@0.5 and mAP@0.5:0.95 (101-point interpolation), precision / recall / F1 curves, confusion matrix.
[METRICS.md](METRICS.md) defines exactly how mAP is computed.

## The Trainer
`folo_vision::train::Trainer` implements MODEL.md §8.
- **Data to step:** folo-data's `DataLoader` (seeded per epoch and sample, so batches do not depend on the worker
  count) → tensors → train-mode forward → loss → backward into a `GradAccumulator` (nominal batch 64) → schedule
  (warmup + linear / cosine, in optimizer steps), global-norm clipping, SGD / AdamW / MuSGD with `decay` / `bn` /
  `bias` groups (`auto` picks per §8.2), EMA.
- **Non-finite batches:** batches with a non-finite loss or BatchNorm statistic are skipped and their running
  statistics restored.
- **After every epoch:** validation of the EMA model with `metrics::Evaluator`, fitness mAP@0.5:0.95, `last.folo` /
  `best.folo` checkpoints (complete FORMAT.md state, atomic), early stopping. Resume from a checkpoint is bit-exact.
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
- With DFL box fusion / DFL scoring the dense path's DFL moment/entropy ops also run on the candidates' box-logit
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
The integration tests in `crates/folo-vision/tests/` are named for what they protect: `model.rs`, `loss_cases.rs`,
`assign_cases.rs`, `nms_reference.rs`, `metrics_known.rs`, `coco_parity.rs`, `coco_full.rs`, `train.rs`, `o2o.rs`,
`lazy_head.rs`, `fusion_reference.rs`, and the GPU ones (`gpu*.rs`).
