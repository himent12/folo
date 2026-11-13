# Hill-climb 3, lane `lazy`: candidate-first prediction (the lazy box head)

Branch `infer/lazy-box-head` (PR 1, CPU). Design: `D:/folo-build/hill3/briefs/lazy.md`; counsel's audit
`D:/folo-build/hill3/counsel-a.md`. Logs: `D:/folo-data/runs-hill3/lazy/logs/`.

## What it is
Single-label prediction needs a box only for the anchors whose best class scores `> conf`. The head's box and
class branches (MODEL.md §6.1) are independent functions of the same level feature, so on the CPU the predictor now
runs the three class branches densely, extracts each level's candidates straight from its class map, and runs the
box branch (`conv0`, `conv1` 3×3, `out` 1×1, then the DFL decode) only at the candidates (MODEL.md §9.5,
`FoloModel::predict_candidates`, `crates/folo-vision/src/model/lazy.rs`). The dense raw outputs (`HeadOutput`) and the
head's `cat` are never made on this path; the forward rows (dense by definition) are unchanged.

### Bit for bit, and why
- `Tensor::conv2d_act_at` / `Backend::conv2d_act_at` (folo-core): the inference conv needed only at some output
  sites. The result has the full output shape; sites hold the dense values, the rest is unspecified.
- CPU, 3×3 convs: the dense path runs these as Winograd F(2×2, 3×3) (every box conv of every FOLO scale passes
  `uses_winograd`: `c_box ≥ 64`, so `c·o ≥ 4096`). `winograd_forward_at` runs the existing block code over a list of
  2×2 tiles. Per tile, the input transform, the 16 GEMM sums (whole 16-wide panels, the same `KC` blocks, the
  same micro-kernel) and the output transform do not depend on which tiles share the block, so a listed tile gets
  the dense bits. The two-pass split and the per-thread block size move with the list length, and neither changes
  a value (the dense path already relies on that across thread counts).
- CPU, the 1×1 `out`: the dense call is `gemm_ex(O, OH·OW, C)` with the weights read in place and `B` packed from
  planes; the sites run the same call over their columns gathered into planes of their own. Each element is the
  same K sum in the same order, wherever its column is (`k_split_parts` is 1 below `K = 2048`, which the code
  checks).
- Receptive field: `conv1` and `out` are evaluated at the candidate pixel, `conv0` on its 3×3 window. In Winograd
  F(2×2, 3×3) an output reads only its own 3×3 window of the tile's 4×4 patch, structurally: output row 0 is
  `m0 + m1 + m2`, row 1 `m1 − m2 − m3`, and `m_i` of the input transform mixes only rows `{0,2}`, `{1,2}`,
  `{2,1}`, `{1,3}`; the same holds for columns. So garbage, even NaN, in the rest of the patch reaches only the
  tile's other outputs. (First cut used the tile's 4×4 receptive field; a mutation to the pixel's 3×3 window kept
  every test green, which is how this was found; it is about 4 instead of 9 `conv0` tiles per isolated
  candidate.)
- Tests: `crates/folo-core/src/cpu/sites_tests.rs` (sites against the dense conv, every channel; a tile list writes
  exactly its tiles, at 1, 3 and all threads); `crates/folo-vision/tests/lazy_head.rs` (records and detections
  against the dense path bit for bit: 640², 320² b2, 384×640, 352², 352×640 b2, folo-s, a b5 batch with a constant
  image, conf with every anchor passing and none, o2o top-k with `max_det` 0..100000 and with ties, class filters,
  the predictor, the dense forward unchanged); `model::lazy::tests::isolated_sites_get_the_dense_boxes` (corners,
  borders, odd/even coordinates, odd-sided P5). `folo_core::cpu::poison_uninit` fills uninitialised CPU buffers with
  NaN, so a test can only pass by computing what it reads (recycled buffers otherwise held the dense values).
  Mutation checks: the 1×1 receptive field fails the isolated-sites and o2o tests.

### Box fusion (after the rebase onto #257)
#257 added opt-in box fusion after NMS (MODEL.md §9.4: `none`, `vote`, `dfl`). `dfl` weighs each cluster member's
sides by their precisions, which the dense CPU path computes from the same softmax moments as the box
(`dfl_box`: `softmax_moments`, `side_precision`). Candidate-first serves every setting with the dense result:
- `none` and `vote` read only the boxes and scores, which are already the dense ones.
- `dfl`: `predict_candidates` returns `ImageCandidates`, the candidates plus, with `dfl` on an `o2m` head, each
  candidate's side precisions. `boxes_at` computes them from the candidate's own box logits with `dfl_box`, so
  they are the dense bits. `finish_candidates` passes them to `nms_candidates_with`.
- Tests: `box_fusion_matches_dense` in `tests/lazy_head.rs` (`vote` and `dfl`, `o2m` and `o2o`, a few percent and
  every anchor passing; every candidate's precisions against `HeadOutput::side_precisions` bit for bit; it asserts
  that fusion moves boxes) and the predictor with `dfl`. Dropping the precisions in `finish_candidates` fails both.
- All of val2017 again, folo-n on the CPU (`FOLO_BENCH_VERIFY=1`): 0 of 5000 images mismatched with fusion off,
  and 0 with `FOLO_BENCH_FUSION=dfl` (`logs/verify2-cpu-*`).

## The GPU path (PR 2, `infer/lazy-box-head-gpu`)
Class pass (`sites_class.wgsl`) → fixed-capacity site lists per image and level (1024 / 512 / 256; val2017 at 0.25
never needs more than 117 / 201 / 82) plus a deduplicated field list of the sites' 3×3 windows → `conv0` on the
field (scattered into a full map), `conv1` and `out` at the sites (compact `[B, C, cap]`) → decode
(`sites_decode.wgsl`, anchors computed in the kernel) → one readback. An image that overflows a level is answered
by the dense head on the same features.
- The `SITES` conv (`kernels/conv/sites.rs`) is the implicit GEMM with each column's pixel read from the site list.
  The per-element sum is sequential over K, so the column's position and the tile config do not change bits; the
  split-K plan and the tensor-core choice do, so the sites conv takes the dense shape's. Bit for bit in f32, f16 and
  on tensor cores (tests at the box convs' shapes, and the whole predictor against the dense GPU path).
- Tensor cores: naga refuses cooperative-matrix ops after a `return` that depends on a buffer (uniformity). The first
  cut retired dead workgroups that way; the kernel failed to build, which silently turned tensor cores off for the
  whole context, and the f16 test then failed by 1 ulp against the dense tensor-core conv. The sites variant now runs
  zero K tiles instead of returning, and the tests assert the tensor-core state after the run.

## Numbers
### Candidates on COCO val2017
Candidates per image that need a box, at conf 0.25 with the published weights, over all 5000 val2017 images
(`benches/candidates.rs`, rect letterbox at 640, as the predictor runs). With `FOLO_BENCH_VERIFY=1` every image's
detections were also compared with the dense path: **0 mismatched** for both models.

| model | level (anchors at 640²) | median | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| folo-n | P3 (6400) | 0 | 19 | 57 | 104 | 6.0 |
| folo-n | P4 (1600) | 5 | 29 | 71 | 201 | 10.3 |
| folo-n | P5 (400) | 10 | 26 | 46 | 77 | 12.6 |
| folo-n | total (8400) | 21 | 63 | 119 | 227 | 28.9 |
| folo-s | P3 | 0 | 19 | 55 | 117 | 5.9 |
| folo-s | P4 | 8 | 40 | 90 | 166 | 14.5 |
| folo-s | P5 | 11 | 31 | 55 | 82 | 15.5 |
| folo-s | total | 26 | 77 | 143 | 249 | 35.9 |

So the median image boxes about 0.3% of its anchors, and the worst image 3%.

### CPU A/B (first run, under load)
`benches/cpu_e2e.rs` trained rows (the first 16 val2017 images, 28 threads, 100 iterations), `main` (e5a6ccf0)
against this branch, both built from this bench source, alternated in 3 rounds in one `cpu` lock hold. The machine
was 32–55% busy with other work (another lane's evaluation, a JVM), so only the folo-n rows are readable:

| row | base | new | new/base |
|---|---|---|---|
| folo-n 640² b1 (row 8) | 17.49 ms (17.12–17.67) | 15.02 (14.65–15.91) | 0.859 |
| folo-n 1280×720 (rect 640×384) | 12.83 (12.29–12.84) | 11.79 (11.73–12.19) | 0.919 |
| folo-n `predict_batch` 16 × 640² | 202.2 (190.6–208.2) | 177.4 (169.9–191.4) | 0.877 |
| folo-s 640² b1 | 39.87 (34.52–51.15) | 40.74 (33.57–53.12) | noise |
| folo-s 1280×720 | 26.57 (22.50–31.65) | 24.96 (22.72–28.22) | noise |
| folo-s `predict_batch` 16 | 458.8 (444.2–551.8) | 508.1 (438.9–509.8) | noise |

Counsel's estimate for row 8 was 13.12 → 11.6–12.0 ms (−9 to −12%); under this load folo-n measured −14% at 640² and
−8% on the rect input (fewer anchors, so the box branch is a smaller share). A quiet repeat is pending.

### GPU A/B (PR 2)
Bit for bit on real data: `benches/candidates.rs` with `FOLO_BENCH_DEVICE=gpu FOLO_BENCH_VERIFY=1`, all 5000
val2017 images, folo-n and folo-s, f32, f16 on the regular kernels and f16 on tensor cores: **0 mismatched** in every
config (`logs/verify-gpu-*.txt`).

`benches/gpu_trained.rs` (the public `Predictor`, conf 0.25, the first 16 val2017 images; b1 = `predict` per image,
rect letterbox; b16 = `predict_batch` at 640²), `main` against the branch, 3 alternating rounds in one GPU lock hold,
RTX 4080 SUPER (Vulkan). Median of the per-round medians, ms (min–max):

| row | base | new | new/base |
|---|---|---|---|
| folo-n f32 b16 (row 6) | 10.66 (10.27–11.07) | 9.54 (9.51–9.86) | 0.895 |
| folo-n f16 b16 | 9.97 (9.96–13.41) | 8.98 (8.82–9.37) | 0.901 |
| folo-n f16 tensor cores b16 (row 6) | 7.69 (7.17–60.28) | 6.79 (6.59–7.78) | 0.883 |
| folo-s f32 b16 | 22.83 (22.71–60.65) | 20.79 (20.64–21.34) | 0.911 |
| folo-s f16 b16 | 20.60 (20.49–22.50) | 18.94 (18.93–19.53) | 0.919 |
| folo-s f16 tensor cores b16 | 12.08 (11.94–15.65) | 11.58 (11.30–12.64) | 0.959 |

b1 (row 5) is bimodal on this machine. Within one run of either build, calls jump between about 1.5 and 5 ms (GPU clocks), so medians move by a factor of 2 between runs. The per-run minimums are steadier (median of 3 runs, base → new):

| config | base → new | change |
|---|---|---|
| n f32 | 1.54 → 1.48 | −4% |
| n f16 | 1.71 → 1.62 | −5% |
| n f16 tensor cores | 1.60 → 1.53 | −4% |
| s f32 | 2.45 → 2.20 | −10% |
| s f16 | 2.49 → 2.15 | −14% |
| s f16 tensor cores | 2.03 → 1.77 | −13% |

Against counsel:
- Row 6 measured −4 to −12%, against an estimate of about +10% throughput.
- Row 5 measured −4 to −14% at the minimum, where counsel expected a change within noise.

## Negative results and notes
- No lazy/dense switch. Counsel's break-even (400 / 80 / 16 candidates for P3 / P4 / P5) compared a per-candidate
  GEMM with dense Winograd. With the tile list, lazy runs the dense arithmetic on a subset of tiles, so its work
  is at most the dense work plus building the site lists; it can only lose by that overhead (see the sweep).
- The random-init benches pass no candidate at `conf = 0.25`, so their e2e rows would show the box branch vanish;
  the trained rows (`FOLO_BENCH_WEIGHTS`, `FOLO_BENCH_IMAGES`) are the ones to read.
- **C3k sibling fusion (PR 3): a measured tie; not shipped.** A fused C3k ran `cv1` and `cv2` as one conv with their
  weights stacked (`Tensor::conv2d_act_stacked`: the stacked copy was a derived value kept with the first weight's
  storage and keyed by the other parts' storage and version, so writes to any part made it again). Bit for bit the
  unstacked block on the CPU and the GPU (f32, f16, tensor cores), checked at folo-n/s C3k shapes; a mutation failed
  both. A/B against `main`, both built from the same bench sources:
  - GPU forward (`benches/gpu.rs`, 3 rounds): row 1 (b1 f32) 1.14 → 1.13 ms (n), 2.24 → 2.23 (s); row 3 (b16
    f32) 7.96 → 7.96 (n), 19.70 → 19.70 (s). The pipelined numbers, b16 f16 and folo-s b1 f16 were inside ±2.5%.
    folo-n b1 f16 (regular kernels) was 1.17 → 1.13 (−3.4%; not a row the criterion names).
  - CPU row 7 (`forward 640 b1`, 28 threads): folo-n 15.21 → 15.12 ms over 7 rounds (0.994, load 17–43%). The
    folo-s rounds ran at 37–86% load and are unreadable.
  - Per layer (`FOLO_BENCH_LAYERS=1`, 5 rounds): the three C3k-bearing layers (stage4, stage5, out5) got
    0.07–0.12 ms faster each on folo-n, 0.32 ms in all, about 2.5% of the forward. On folo-s they saved about
    0.4 ms, about 1.4%. That fits what fusing saves: one pass over the input and one pool wait (CPU) or one
    dispatch (GPU) per C3k, three times per model.
  - Inside ±3% on rows 7, 1 and 3, so not shipped (the brief's rule). The commit is `05cf15a1` on the local
    branch `infer/c3k-sibling-fusion` if a later round wants it (folo-m/l/x have 8 such pairs, not measured).

## Leftovers
- A quiet CPU repeat of PR 1's rows (folo-s was unreadable under load) and the break-even sweeps on the CPU and the
  GPU (`benches/candidates.rs`, `FOLO_BENCH_SWEEP=1`): pending a quiet window.
- The validation decode (one kernel emitting multi-label pairs plus boxes only where a pair is): not started.
