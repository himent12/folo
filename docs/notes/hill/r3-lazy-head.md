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

## Numbers
(pending: measured once the fusion lane's training run released the CPU)

## Negative results and notes
- No lazy/dense switch. Counsel's break-even (400 / 80 / 16 candidates for P3 / P4 / P5) compared a per-candidate
  GEMM with dense Winograd. With the tile list, lazy runs the dense arithmetic on a subset of tiles, so its work
  is at most the dense work plus building the site lists; it can only lose by that overhead (see the sweep).
- The random-init benches pass no candidate at `conf = 0.25`, so their e2e rows would show the box branch vanish;
  the trained rows (`FOLO_BENCH_WEIGHTS`, `FOLO_BENCH_IMAGES`) are the ones to read.

## Leftovers
- PR 2: the GPU path (`SITES` conv variant, class pass split from the decode, fixed capacity).
- PR 3: C3k sibling fusion.
