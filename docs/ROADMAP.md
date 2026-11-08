# FOLO roadmap

Status keys: `[ ]` todo, `[~]` in progress, `[x]` merged. The orchestrator keeps this file current.

## M0 — scaffold
- [x] Cargo workspace, crate skeletons, AGENTS.md, ARCHITECTURE.md, API.md, CI
- [x] specs: .folo format (#1), MCP (#2), detector + training recipe (#4); CI on self-hosted runners (#17, #23)

## M1 — foundations
- [x] core: tensor, views, dtypes, CPU kernels (GEMM ~1 TFLOPS), autograd, grad checks, custom ops (#11); deterministic reductions, faster conv, fused conv epilogue (#39)
- [x] data: image IO, YOLO + COCO formats, stats, letterbox, augmentation, DataLoader, synthetic shapes (#8)
- [x] vision-post: box ops, IoU family, NMS, mAP / PR metrics, confusion matrix, TAL assigner (#5, #15); pycocotools parity (#20)
- [x] server + CLI skeleton with `--mock` (#7); real dataset commands and store (#29)
- [x] gpu: wgpu device, buffer pool, WGSL kernels (SGEMM ~31 TFLOPS on RTX 4080 SUPER) (#10)
- [~] Studio (web/): all screens merged (#3, #6, #13, #19, #24, #34, #37, #40, #41, #44, #48); hill-climb round 9 (~7–8)
- [~] Desktop (apps/desktop, GPUI): #9, #26, #32, #35, #43, #45, #47, #52; hill-climb round 15 (8.3 / 8.3)

## M2 — learning end to end
- [x] nn: layers, init, SGD/AdamW, schedules, EMA, .folo IO (#21)
- [x] vision: FOLO-n/s/m/l/x (#28), detection loss (#27), trainer + predictor + train/val/predict/export CLI (#33)
- [x] `folo train` on synthetic shapes passes the MODEL.md gate on CPU (mAP50 0.975 / mAP50-95 0.878 at epoch 23; examples/shapes)

## M3 — fast
- [x] gpu backend implements core Backend (#31); whole models on the GPU, fused Conv+BN+SiLU / BN / optimizer kernels (#38)
- [x] benchmarks: Ultralytics baselines (#14); folo-n GPU forward b1 3.98 ms vs yolo26n 4.95 ms (docs/BENCHMARKS.md)
- [~] GPU training end to end with the loss and assigner on the device; R6 real-data run (#46)
- [~] inference round 2: CPU forward ≤ 22.8 ms, GPU end-to-end predict (perf/inference-2)
- [ ] f16 whole-model inference

## M4 — real use
- [x] real datasets from disk (YOLO, COCO), export folo-f16
- [x] server runs real training + inference (#42); `folo mcp` (#36) driving real training (#49); agent cookbook
- [x] in-browser inference (wasm + WebGPU, ~14 ms folo-n @640 in Chrome) (#51); Studio Playground integration pending
- [x] release pipeline: tagged binaries with embedded Studio, install scripts (#22)
- [~] dogfooding: a fresh agent uses FOLO via MCP + CLI and reports friction (dogfood/report-1)
- [ ] ONNX export; pretrained weights

## M5 — polish
- [ ] docs site, more examples, R6 example recipe (in #46)
