# FOLO roadmap

Status keys: `[ ]` todo, `[~]` in progress, `[x]` merged. The orchestrator keeps this file current.

## M0 — scaffold
- [x] Cargo workspace, crate skeletons, AGENTS.md, ARCHITECTURE.md, API.md, CI

## M1 — foundations (parallel)
- [x] core: tensor, views, dtypes, CPU kernels (elementwise, broadcast, reduce, GEMM ~1 TFLOPS, conv2d, pool), autograd, grad checks, custom ops (#11)
- [x] data: image IO, YOLO + COCO formats, stats, letterbox, augmentation, DataLoader, synthetic shapes datasets (#8)
- [x] vision-post: box ops, IoU family, NMS, mAP / PR metrics, confusion matrix, TAL assigner (#5, #15)
- [x] server + CLI skeleton: API.md endpoints with `--mock`, `folo serve`, `folo describe`, `--json` conventions (#7)
- [x] gpu: wgpu device, buffer pool, WGSL kernels (SGEMM ~31 TFLOPS on RTX 4080 SUPER), CPU-parity tests (#10)
- [~] Studio v1 (web/): all screens merged (#3, #6, #13, #19); UI hill-climb in progress (round 1: 5–8)
- [~] Desktop v1 (apps/desktop, GPUI): PR #9; UI hill-climb in progress (round 1: ~6.5)

- [x] specs: .folo format (#1), MCP (#2), detector + training recipe (#4); CI on self-hosted runners (#17)

## M2 — learning end to end (CPU)
- [x] nn: Module, Conv2d, BatchNorm2d, activations, pooling, upsample, concat, init, SGD/AdamW, schedules, EMA, .folo IO (#21)
- [~] vision: FOLO-n/s/m/l/x model, CIoU + DFL + BCE loss (in progress), trainer, predictor
- [ ] `folo train` on synthetic shapes converges (mAP50 > 0.9 on a small shapes set)

## M3 — fast
- [~] gpu backend implements core Backend; training on GPU (in progress); fused conv+bn+silu; f16 inference
- [~] benchmarks: Ultralytics baselines measured (#14, docs/BENCHMARKS.md); FOLO columns pending

## M4 — real use
- [ ] real datasets (YOLO format from disk, COCO), pretrained weights path, export (folo-f16, ONNX)
- [ ] server runs real training + inference; Studio live training dashboard on real runs
- [ ] `folo mcp` server, `llms.txt`, agent cookbook
- [ ] in-browser inference (wasm + WebGPU) in Studio

## M5 — polish
- [ ] docs site, examples, release binaries, R6 example dataset recipe
