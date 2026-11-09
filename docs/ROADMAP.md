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
- [~] Studio (web/): all screens, in-browser Playground, real-server smoke, mixed-precision and capability-aware New run; one breakpoint system and blank-section QA in CI (#126); hill-climb round 22 (r21: Sol min 8, Opus min 7.5)
- [~] Desktop (apps/desktop, GPUI): all screens, real server, local GPU inference, capability-aware New run; hill-climb round 48 (r47 #131: Opus 8.4 with 6/12 lines at 9, Sol 8.6 with 7/12)

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
- [x] AMP training on tensor cores: folo-n 307 img/s vs yolo26n AMP 252 (#101); /api/system capabilities (#108)
- [x] NMS-free one-to-one head, `--head o2o` (#109)
- [x] CPU end-to-end latency below yolo26n at b1 (fused letterbox, direct stem conv, in-place parallel decode; #127)
- [~] CPU GEMM per-thread throughput (forward at 8 threads ~37 ms vs Ultralytics 22.8)

## M4 — real use
- [x] real datasets from disk (YOLO, COCO), export folo-f16
- [x] server runs real training + inference (#42); `folo mcp` (#36) driving real training (#49); agent cookbook
- [x] in-browser inference (wasm + WebGPU, ~14 ms folo-n @640 in Chrome) (#51), in Studio's Playground (#57, #70), shipped in release builds (#78)
- [x] release pipeline: tagged binaries with embedded Studio, install scripts (#22)
- [x] dogfooding: reports 1 (#59 → #69) and 2 (#93 → #99)
- [x] ONNX export with onnxruntime parity in CI (#87)
- [x] COCO-pretrained folo-n: loader + COCOeval + `folo weights` (#72); 50 epochs from scratch, val2017 AP 0.314 / AP50 0.455 (pycocotools parity); R6 fine-tune from COCO beats the 410-epoch scratch run in 100 epochs; `weights-v1` draft release (#128)
- [x] `Model.head` / `trained_with` (#117); [x] `Model.precision` / `fused` (#123); [x] operating-point conf, resolved optimizer/LR, queue control, ONNX download (#129); [x] measured throughput, run estimates, benchmark probe (#130)

## M5 — polish
- [x] README showcase, QUICKSTART, screenshots, llms.txt (#71); R6 example (examples/r6)
- [~] dogfooding report 3 (pretrained weights, o2o, AMP, docs site, MCP)
- [x] docs site: `site/`, generated from docs/, README.md and the examples (search, link checking, llms.txt; #122); build
  and serve it locally with `cd site && npm ci && npm run dev` (site/README.md)
