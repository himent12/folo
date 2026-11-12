# FOLO roadmap

Status keys: `[ ]` todo, `[~]` in progress, `[x]` merged. `#N` is the pull request an item landed in. The milestones
are the order the project was built in; what is open is marked `[ ]` or `[~]`, most of it under M4.

## M0 — scaffold
- [x] Cargo workspace, crate skeletons, AGENTS.md, ARCHITECTURE.md, API.md, CI
- [x] specs: .folo format (#1), MCP (#2), detector + training recipe (#4); CI on self-hosted runners (#17, #23)

## M1 — foundations
- [x] core: tensor, views, dtypes, CPU kernels (GEMM ~1 TFLOPS), autograd, grad checks, custom ops (#11); deterministic reductions, faster conv, fused conv epilogue (#39)
- [x] data: image IO, YOLO + COCO formats, stats, letterbox, augmentation, DataLoader, synthetic shapes (#8)
- [x] vision-post: box ops, IoU family, NMS, mAP / PR metrics, confusion matrix, TAL assigner (#5, #15); pycocotools parity (#20)
- [x] server + CLI skeleton with `--mock` (#7); real dataset commands and store (#29)
- [x] gpu: wgpu device, buffer pool, WGSL kernels (SGEMM ~31 TFLOPS on RTX 4080 SUPER) (#10)
- [x] Studio (web/): all screens, in-browser Playground, real-server smoke, mixed-precision and capability-aware New run; one breakpoint system and blank-section QA in CI (#126); 28 design review rounds (the log is web/notes/HILLCLIMB.md)
- [x] Desktop (apps/desktop, on its own UI framework folo-ui, which replaced GPUI): all screens, real server, local GPU inference, capability-aware New run; 56 design review rounds (the log is apps/desktop/notes/HILLCLIMB.md)

## M2 — learning end to end
- [x] nn: layers, init, SGD/AdamW, schedules, EMA, .folo IO (#21)
- [x] vision: FOLO-n/s/m/l/x (#28), detection loss (#27), trainer + predictor + train/val/predict/export CLI (#33)
- [x] `folo train` on synthetic shapes passes the MODEL.md gate on CPU (mAP50 0.975 / mAP50-95 0.878 at epoch 23; examples/shapes)

## M3 — fast
- [x] gpu backend implements core Backend (#31); whole models on the GPU, fused Conv+BN+SiLU / BN / optimizer kernels (#38)
- [x] benchmarks: Ultralytics baselines (#14); folo-n beats yolo26n on GPU fwd b1 (2.99 vs 4.95 ms), GPU e2e b1/b16 and CPU fwd (#55; docs/BENCHMARKS.md)
- [x] GPU training end to end with the loss and assigner on the device: 219 img/s vs yolo26n 202 (#46, #67)
- [x] inference round 2: pack-free CPU conv, GPU letterbox and candidate decode (#55)
- [x] f16 whole-model inference (#80); tensor-core f16 GEMM/conv, opt-in (#89)
- [x] AMP training on tensor cores: folo-n 307 img/s vs yolo26n AMP 252 (#101); /api/system capabilities (#108)
- [x] NMS-free one-to-one head, `--head o2o` (#109)
- [x] CPU end-to-end latency below yolo26n at b1 (fused letterbox, direct stem conv, in-place parallel decode; #127)
- [x] CPU per-core efficiency: B packing, depthwise AVX2, fused bias+SiLU, dedicated pool (#137; folo-n forward 14.8-18.7 ms at 8 threads vs Ultralytics 22.8)
- [x] engine hill-climb, five areas merged in `hill/integration` (docs/BENCHMARKS.md, Hill-climb; notebooks in docs/notes/hill/): folo-n GPU forward b1 2.09 ms, e2e b16 1898 img/s on tensor cores, train step 327 / 537 img/s (f32 / AMP, yolo26n 202 / 252), trainer sustained 347 / 500 img/s (row 10, bars 199 / 224), COCO `folo val` 29.6 → 8.9 s, CPU train step −25% and `predict_batch` +32–47%; MuSGD and `auto` rules 2 and 3. Not moved: CPU batch-1 latency
- [x] engine hill-climb 2: nine lanes, 27 PRs (#181–#207), re-measured on `main` against 647b981f (docs/BENCHMARKS.md, Hill-climb 2; notebooks docs/notes/hill/r2-*.md, fold-in docs/notes/hill/r2-fold-in.md): folo-n GPU forward b1 2.13 → 1.14 ms (f32) and 1.85 → 0.95 ms (f16 tensor cores), b16 1619 → 2081 img/s (f32) and 3081 → 4159 (tensor cores), e2e b16 2155 img/s; folo-s now beats yolo26s at b16 (819 / 1799 img/s, were ties); train step 333 → 380 / 544 → 594 img/s (f32 / AMP), MuSGD's step overhead +23% → +0.2% under AMP; trainer row 10 388 / 613 img/s (bars 199 / 224); CPU forward b1 15.2 → 12.7 ms (Winograd, in-place weights); COCO `folo train` to the first step 36.5 → 1.3 s with warm caches, first-epoch compile 1–2 s → 0.01 s, `folo val` 8.0 → 4.5 s, `--cache ram` 6.5x on 1080p JPEGs; `auto` rules 4 and 5 (MuSGD for every run, proxy AP 0.0757 → 0.1234). Worse: a first-ever AMP epoch (no kernel list yet, cold driver cache), +1.6 s

## M4 — real use
- [x] real datasets from disk (YOLO, COCO), export folo-f16
- [x] server runs real training + inference (#42); `folo mcp` (#36) driving real training (#49); agent cookbook
- [x] in-browser inference (wasm + WebGPU, ~14 ms folo-n @640 in Chrome) (#51), in Studio's Playground (#57, #70), shipped in release builds (#78)
- [x] release pipeline: tagged binaries with embedded Studio, install scripts (#22); self-hosted runners (#104); v0.1.0 CHANGELOG, draft-by-dispatch, wasm and install-script assets (#145). GitHub's "latest" release has to be a FOLO version for `releases/latest/download/...` and the install scripts to find binaries, so weights releases are created with `--latest=false` (docs/RELEASE.md, Weights releases)
- [~] COCO long schedule: folo-n 300 epochs from scratch, `weights-v2` (#144); paused, see below
- [x] dogfooding: reports 1 (#59 → #69) and 2 (#93 → #99)
- [x] ONNX export with onnxruntime parity in CI (#87)
- [x] COCO-pretrained folo-n: loader + COCOeval + `folo weights` (#72); 50 epochs from scratch, val2017 AP 0.314 / AP50 0.455 (pycocotools parity); `weights-v1` release (#128; published 2026-11-11, #175)
- [~] COCO 300-epoch folo-n (#144): **paused at epoch 62 of 300** (8.0 h, AMP), val2017 AP 0.317 / AP50 0.458; interim `folo-n-coco-v2` in the `weights-v2` release (published 2026-11-11). About 29 h of training are left on an RTX 4080 SUPER; it resumes from its checkpoint (examples/coco/README.md, Resume (v2)) and then needs evaluating and exporting again
- [~] accuracy on your own data (#175): `weights-v1` and `weights-v2` published, `folo weights pull` over plain HTTPS with a `gh` fallback; a fine-tuning benchmark against Ultralytics on four small public datasets (`bench/finetune/`, docs/BENCHMARKS.md, Fine-tuning on small datasets); **folo-s from scratch on COCO**, 56 epochs on a rented RTX PRO 4000 Blackwell (9.5 h of epochs, about $5.60 for the pod), val2017 AP 0.416 / AP50 0.580, `folo-s-coco` in `weights-v3`, the first full-scale run of `auto` rule 3 (MuSGD)
- [ ] still open after #175 and hill-climb 2: a full COCO run under `auto` rule 5 (the next accuracy step: the folo-s run used rule 3, and rule 5's proxy gain, 0.1135 → 0.1234 AP over rule 4, is unmeasured at full scale); resume the folo-n 300-epoch run (paused at epoch 62); folo-m / folo-l weights; an equal-budget COCO comparison with Ultralytics (same schedule and data on both trainers); MuSGD against SGD at full scale; the incumbent's GPU batch-1 latency measured again with FOLO's warm-up (at least a second of calls, not only the 25 or 10; docs/BENCHMARKS.md, Summary)
- [x] `Model.head` / `trained_with` (#117); [x] `Model.precision` / `fused` (#123); [x] operating-point conf, resolved optimizer/LR, queue control, ONNX download (#129); [x] measured throughput, run estimates, benchmark probe (#130)

- [x] agent labeling: unlabeled images → boxes by Claude Code or Codex (one tool-free model request per batch) →
  review in Studio → a registered YOLO dataset (`/api/label`, docs/LABELING.md; #165). Next: the same screen in FOLO
  Desktop, `folo label` and MCP tools

- [x] vision benchmarks: a frozen set of labeled images plus a plan of stress conditions, any vision model as a
  contestant (FOLO models, Claude Code / Codex, external predictions), a five-axis score with exact definitions,
  agent-designed plans, and self-improvement loops (`crates/folo-bench`, `/api/bench`, Benchmarks in Studio;
  docs/BENCH.md). Next: the screens in FOLO Desktop, a `folo` CLI command and MCP tools; confidences from agents
  (so they get a calibration axis); generated images as a benchmark source; masks and keypoints

- [x] cloud compute: training runs on a rented GPU (RunPod) or on another FOLO server, as ordinary runs of this
  server (`TrainConfig.compute`, `/api/cloud`, `folo cloud`, Settings → Cloud and Run on in Studio and Desktop;
  docs/CLOUD.md). The worker side is plain API: `folo serve --token`, `/api/uploads`, run cursors, weight ranges.
  Next: check the RunPod provider against a live account (`crates/folo-cloud/tests/runpod_live.rs`) once a
  release with this is published for workers to install; TLS on the worker, then providers that expose a raw
  port (Vast.ai, Lambda); spot instances with automatic resume

## M5 — polish
- [x] README showcase, QUICKSTART, screenshots, llms.txt (#71)
- [x] dogfooding report 3 (#134) and its fixes: GPU memory released after runs, published weights as models, AMP refused up front, server-built CLI command (#140); eval provenance (#138)
- [x] test hygiene: every test cleans up its temp dirs, `FOLO_TEST_TMP`, CI fails on leftovers (#160)
- [x] docs site: `site/`, generated from docs/, README.md and the examples (search, link checking, llms.txt; #122); build
  and serve it locally with `cd site && npm ci && npm run dev` (site/README.md)
