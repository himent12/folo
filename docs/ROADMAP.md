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
- [x] engine hill-climb 3: new algorithms, four lanes, #253–#260, re-measured on `main` against e5a6ccf0 (docs/BENCHMARKS.md, Hill-climb 3; notebooks docs/notes/hill/r3-*.md, fold-in docs/notes/hill/r3-fold-in.md): candidate-first prediction (box branch only at the candidates, bit-identical; #254 CPU, #255 GPU), recorded GPU forward plans (#258), `predict_batch` on a shared rect canvas (#259), opt-in box fusion `--fusion vote|dfl` (#257: +0.0025 to +0.0043 COCO AP on the published weights), the validation clip (#253). On trained weights folo-n GPU end to end b1 1.77 → 1.60 ms (f32) and 1.51 → 1.33 ms (tensor cores), b16 2174 → 2428 img/s (tensor cores); forward b1 on tensor cores 0.94 → 0.845 ms with plans; 16 × 1280×720 frames −18 to −35% on the GPU and −27 to −36% on the CPU. Not moved: the forward at batch 16, the training step. folo-n's CPU row 8 on trained weights: a tie at the fold-in's 60 calls, −6% at 200 calls (13.88 → 13.04 ms, notes/hill/r3-cpufix.md). Worse, then fixed: folo-s single 1280×720 CPU frames +24–29% after another model's rect batch in one process; #259 exposed a CPU buffer cache that never evicted, #264 evicts stale lengths (26.4 → 21.3 ms). Tried and lost: a learned box refiner (#260, best +0.0007 AP of a +0.005 gate), C3k sibling fusion, lattice pooling, plans at batch 16
- [x] engine hill-climb 5, four lanes (docs/BENCHMARKS.md, Hill-climb 5; notebooks docs/notes/hill/r5-*.md): the fused distillation kernel (#297: `kd_step` 6.84 → 1.23 ms, 5.6x, 53 → 7 GPU commands; a training step with the teacher −4 / −7 / −12% by median step time in three loaded rounds, measured on the kernel's first version; GPU distillation numerics change by rounding only) and opt-in int8 CPU inference of the backbone and neck (#300: `folo quantize`, `--precision int8`; val2017 AP −0.0019 folo-n, −0.0025 folo-s; folo-n row 8 14.91 → 12.74 ms under load, folo-s 1.82x in one loaded in-process run, an upper estimate). Closed, not merged: #294 (training glue, a tie: f32 +2.4%, AMP +1.9%) and #298 (sync-free AMP optimizer window: +13% img/s GPU-bound, about nothing on loader-bound full COCO, blocking error-path bugs after four review rounds); their branches stay on origin. Tried and lost: class-wise rank distillation (#295, +0.0002 AP at its gate), BatchNorm statistics in the conv epilogue (+0.6%), the decode-once mosaic window (1.56x against a 1.8x gate), `train/amp-step` again (a tie), carving folo-n out of folo-s (rejected before any code). Not reached: the row 9 targets (the backward GEMMs are 12.7 of the backward's 22.5 GPU-ms per AMP step) and folo-n's int8 row 8 of 10 ms

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
- [x] folo-n v3: v2's weights annealed for 16 more epochs (78 in all, 1.58 h; hill-climb 3), val2017 AP 0.317 → 0.340 / AP50 0.487 (`main`'s `folo val`, 2026-11-13), `folo-n-coco-v3` in `weights-v4` (#261); 0.343 with the opt-in `--fusion dfl --iou 0.65 --max-det 1000`
- [~] accuracy on your own data (#175): `weights-v1` and `weights-v2` published, `folo weights pull` over plain HTTPS with a `gh` fallback; a fine-tuning benchmark against Ultralytics on four small public datasets (`bench/finetune/`, docs/BENCHMARKS.md, Fine-tuning on small datasets); **folo-s from scratch on COCO**, 56 epochs on a rented RTX PRO 4000 Blackwell (9.5 h of epochs, about $5.60 for the pod), val2017 AP 0.416 / AP50 0.580, `folo-s-coco` in `weights-v3`, the first full-scale run of `auto` rule 3 (MuSGD)
- [x] hill-climb 4, accuracy: opt-in dense distillation from a frozen teacher, `folo train --teacher` (#279, docs/MODEL.md §8.9), and the first full COCO folo-n under `auto` rule 5 (docs/BENCHMARKS.md, Hill-climb 4; notebook docs/notes/hill/r4-distill.md; #284): from scratch, 64 epochs, MuSGD, val2017 AP 0.3507; its epoch-32 checkpoint continued with the folo-s teacher 0.3559, +0.0052 [+0.0032, +0.0075] (proxy P1: 0.1550 against 0.1281). One seed per side; the fork published as `folo-n-coco-v4` in `weights-v5` (val2017 AP 0.356 / AP50 0.509; 0.359 with the fusion flags), the control not published. Tried and lost: a mirrored-view consensus teacher, a blended class target
- [x] hill-climb 5, the long COCO run: `folo-n-coco-v4` continued for 64 epochs with the folo-s teacher (fused kernel, no rank term; about 6 h), val2017 AP 0.3701 / AP50 0.5250, +0.0141 [+0.0114, +0.0166] over v4 and +0.0194 [+0.0160, +0.0223] over hill-climb 4's control (paired bootstrap, one seed; no teacher-free continuation, so the gain is the epochs and the teacher together; docs/notes/hill/r5-accuracy.md, Long run). Published as `folo-n-coco-v5` in `weights-v6` (2026-11-18; the f16 file 0.370, the f32 file 0.373 with the fusion flags; examples/coco/README.md, v5)
- [ ] after hill-climb 5: a teacher-free continuation of v4 of the same length, to split the long run's gain between the 64 epochs and the teacher; faster backward GEMM shapes for row 9 (the stride-2 input gradient as one sub-pixel conv, about 0.7 ms per AMP step, and a batched-chunk 1×1 weight gradient, about 0.4 ms; docs/notes/hill/r5-gputrain.md); the int8 leftovers for folo-n's row 8 at 10 ms or less (an int8 stem, fewer quantise boundaries around attention, f32 for the conv shapes int8 loses, a cheaper bit-exact SiLU epilogue, an AVX-512 VNNI kernel; published int8 weights; docs/notes/hill/r5-cpuq8.md); the AMP window's host wait deferred without a device guard (docs/notes/hill/r5-trainer.md)
- [ ] still open after #175 and hill-climb 4: a whole distilled COCO run with the teacher from epoch 1 (hill-climb 5 ran a continuation of v4 instead); a second seed per side; the teacher's training cost without other load; resume the folo-n 300-epoch run (paused at epoch 62); folo-m / folo-l weights; an equal-budget COCO comparison with Ultralytics (same schedule and data on both trainers); MuSGD against SGD at full scale; the incumbent's GPU batch-1 latency measured again with FOLO's warm-up (at least a second of calls, not only the 25 or 10; docs/BENCHMARKS.md, Summary)
- [x] `Model.head` / `trained_with` (#117); [x] `Model.precision` / `fused` (#123); [x] operating-point conf, resolved optimizer/LR, queue control, ONNX download (#129); [x] measured throughput, run estimates, benchmark probe (#130)

- [x] agent labeling: unlabeled images → boxes by Claude Code or Codex (one tool-free model request per batch) →
  review in Studio, `folo label` or MCP tools → a registered YOLO dataset (`/api/label`, `folo label`, 10 MCP tools,
  docs/LABELING.md; #165). Next: the same screen in FOLO Desktop

- [x] vision benchmarks: a frozen set of labeled images plus a plan of stress conditions, any vision model as a
  contestant (FOLO models, Claude Code / Codex, external predictions), a five-axis score with exact definitions,
  agent-designed plans, and self-improvement loops (`crates/folo-bench`, `/api/bench`, Benchmarks in Studio;
  docs/BENCH.md). Next: the screens in FOLO Desktop, a `folo` CLI command and MCP tools; confidences from agents
  (so they get a calibration axis); generated images as a benchmark source; masks and keypoints

- [x] cloud compute: training runs on a rented GPU (RunPod) or on another FOLO server, as ordinary runs of this
  server (`TrainConfig.compute`, `/api/cloud`, `folo cloud`, Settings → Cloud and Run on in Studio and Desktop;
  docs/CLOUD.md). The worker side is plain API: `folo serve --token`, `/api/uploads`, run cursors, weight ranges.
  Next: TLS on the worker, then providers that expose a raw port (Vast.ai, Lambda); spot instances with automatic
  resume

## M5 — polish
- [x] README showcase, QUICKSTART, screenshots, llms.txt (#71)
- [x] dogfooding report 3 (#134) and its fixes: GPU memory released after runs, published weights as models, AMP refused up front, server-built CLI command (#140); eval provenance (#138)
- [x] test hygiene: every test cleans up its temp dirs, `FOLO_TEST_TMP`, CI fails on leftovers (#160)
- [x] docs site: `site/`, generated from docs/, README.md and the examples (search, link checking, llms.txt; #122); build
  and serve it locally with `cd site && npm ci && npm run dev` (site/README.md)
