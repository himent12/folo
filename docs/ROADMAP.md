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
- [~] Studio (web/): all screens, in-browser Playground, real-server smoke; hill-climb round 15 (r12: min 8 from both reviewers); contrast + CVD token checks in CI (#84)
- [~] Desktop (apps/desktop, GPUI): all screens, real server, local GPU inference (#73); hill-climb round 32 (r31: Opus 8.6 / Sol 8.3)

## M2 — learning end to end
- [x] nn: layers, init, SGD/AdamW, schedules, EMA, .folo IO (#21)
- [x] vision: FOLO-n/s/m/l/x (#28), detection loss (#27), trainer + predictor + train/val/predict/export CLI (#33)
- [x] `folo train` on synthetic shapes passes the MODEL.md gate on CPU (mAP50 0.975 / mAP50-95 0.878 at epoch 23; examples/shapes)

## M3 — fast
- [x] gpu backend implements core Backend (#31); whole models on the GPU, fused Conv+BN+SiLU / BN / optimizer kernels (#38)
- [x] benchmarks: Ultralytics baselines (#14); folo-n beats yolo26n on GPU fwd b1 (2.99 vs 4.95 ms), GPU e2e b1/b16 and CPU fwd (#55; docs/BENCHMARKS.md)
- [x] GPU training end to end with the loss and assigner on the device: 219 img/s vs yolo26n 202 (#46); R6 runs (#46, #67)
- [x] inference round 2: pack-free CPU conv, GPU letterbox and candidate decode (#55)
- [x] f16 whole-model inference (#80); tensor-core f16 GEMM/conv, opt-in (#89)
- [~] AMP training on tensor cores (train/amp)

## M4 — real use
- [x] real datasets from disk (YOLO, COCO), export folo-f16
- [x] server runs real training + inference (#42); `folo mcp` (#36) driving real training (#49); agent cookbook
- [x] in-browser inference (wasm + WebGPU, ~14 ms folo-n @640 in Chrome) (#51), in Studio's Playground (#57, #70), shipped in release builds (#78)
- [x] release pipeline: tagged binaries with embedded Studio, install scripts (#22)
- [x] dogfooding: reports 1 (#59, fixed in #69) and 2 (#93; fixes in flight on fix/dogfood-2)
- [x] ONNX export with onnxruntime parity in CI (#87)
- [~] COCO-pretrained folo-n: loader + COCOeval + `folo weights` (#72); 50-epoch run in progress (epoch 29: mAP50-95 0.273)

## M5 — polish
- [x] README showcase, QUICKSTART, screenshots, llms.txt (#71); R6 example (examples/r6)
- [ ] docs site
