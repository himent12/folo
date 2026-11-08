# FOLO roadmap

Status keys: `[ ]` todo, `[~]` in progress, `[x]` merged. The orchestrator keeps this file current.

## M0 — scaffold
- [x] Cargo workspace, crate skeletons, AGENTS.md, ARCHITECTURE.md, API.md, CI

## M1 — foundations (parallel)
- [ ] core: tensor, views, dtypes, CPU kernels (elementwise, broadcast, reduce, GEMM, conv2d, pool), autograd, grad checks
- [ ] data: image IO, YOLO + COCO formats, stats, letterbox, augmentation, DataLoader, synthetic shapes datasets
- [ ] vision-post: box ops, IoU family, NMS, mAP / PR metrics, confusion matrix (plain slices)
- [ ] server + CLI skeleton: API.md endpoints with `--mock`, `folo serve`, `folo describe`, `--json` conventions
- [ ] gpu: wgpu device, buffer pool, WGSL GEMM / elementwise / reduce kernels with CPU-parity tests
- [ ] Studio v1 (web/) against the mock server, UI hill-climb loop to 9/10

## M2 — learning end to end (CPU)
- [ ] nn: Module, Conv2d, BatchNorm2d, activations, pooling, upsample, concat, init, SGD/AdamW, schedules, EMA, .folo IO
- [ ] vision: FOLO-n/s/m model, TAL assigner, CIoU + DFL + BCE loss, decode, trainer, predictor
- [ ] `folo train` on synthetic shapes converges (mAP50 > 0.9 on a small shapes set)

## M3 — fast
- [ ] gpu backend implements core Backend; training on GPU; fused conv+bn+silu; f16 inference
- [ ] benchmarks vs PyTorch/Ultralytics on the same model size (inference latency, train imgs/s)

## M4 — real use
- [ ] real datasets (YOLO format from disk, COCO), pretrained weights path, export (folo-f16, ONNX)
- [ ] server runs real training + inference; Studio live training dashboard on real runs
- [ ] `folo mcp` server, `llms.txt`, agent cookbook
- [ ] in-browser inference (wasm + WebGPU) in Studio

## M5 — polish
- [ ] docs site, examples, release binaries, R6 example dataset recipe
