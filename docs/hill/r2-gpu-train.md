# Hill-climb 2: the GPU training step (lane `gpu-train`, branches `train/*`)

Round 2 of docs/hill/gpu-train.md. Goal: a faster folo-n / folo-s training step (batch 16, 640 px, forward + loss
+ backward + optimizer) on the RTX 4080 SUPER, without changing what is computed beyond summation order.
Baseline: `main` 647b981f.

## Method

- **Profile** (`FOLO_GPU_PROFILE=1`, docs/GPU.md "Profiling"): GPU time per kernel label and phase. Timestamps
  serialise kernels, so profiled wall times are not the real ones; per-label times swing by about ±15 % between
  runs on the shared GPU, so a kernel change is judged on the group totals of several runs, and claimed only
  through the end-to-end A/B.
- **A/B**: `cargo bench -p folo-vision --bench gpu` (`FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=train
  FOLO_BENCH_SCALES=n`, plus `FOLO_BENCH_CLIP=1` / `FOLO_BENCH_OPTIM=musgd` where stated), one binary per side
  (`base` = main 647b981f, built from this worktree before any change), run alternately in one GPU-lock hold
  (`D:/folo-build/hill/lock.sh`), each run's figure the median of its 10 steps; tables give the median of the
  rounds and the range. Host CPU load (other lanes' builds and jobs) is logged before each run.
  Script: `D:/folo-build/hill2-gpu-train/bin/ab.sh <tag> <rounds> base <new>`; raw outputs in
  `D:/folo-data/runs-hill2/gpu-train/ab-<tag>/`.
- **Correctness**: every changed kernel against the CPU reference (the existing conv-gradient tests in
  `folo-gpu` cover the f32 and tensor-core weight gradients on every folo-n / s layer shape plus edge shapes; the
  stem shape and split-K included), the gradient checks, the GPU-vs-CPU training-step test.

## Profile of main (folo-n, b16, 640)

Wall-clock split (unprofiled run, host 0-1 % GPU load): f32 48.1 ms per step (332 img/s): forward 13.1, loss 0.6,
backward 34.2, SGD 1.2 ms; AMP 29.6 ms (541 img/s): 7.9 / 0.7 / 19.5 / 2.4 ms. GPU time per step (profiled):

| phase | f32 GPU ms | AMP GPU ms | largest items |
|---|---|---|---|
| forward | 13.8 | 8.8 | convs 7.1 / 3.7, BatchNorm 3.6 / 2.6, concat copies 1.4 / 0.6, AMP casts – / 0.8 (98 dispatches) |
| backward | 33.0 | 20.4 | weight grads 11.9 + 0.9 split-K reduce / 7.0 + 0.7, input grads 8.6 / 5.4, BatchNorm 6.3 / 3.6 |
| step | 0.1 | 1.2 | AMP: 255 gradient copies into the accumulator (0.9 ms, 255 dispatches) |

MuSGD (`FOLO_BENCH_OPTIM=musgd`): f32 50.7 ms (315 img/s, +5 % over SGD), step phase 6.9 ms (SGD 1.2), 7.7 GPU-ms
in 1225 dispatches (255 matmuls, 393 + 187 elementwise, 224 copies), host encode of the step 6.4 ms.

## PR 1: weight-gradient B tiles in runs of pixels, a stem tile (`train/weight-grad-kernels`)

**Finding.** The weight gradient's `B` tile (the image as a transposed im2col, `B[k = (n, oh, ow)][j = (ci, kh,
kw)]`) was loaded with neighbouring invocations on neighbouring columns `j`, which sit a whole image plane apart
(`IH · IW` elements; for 3×3 layers 3 taps share a row). Every load instruction of a warp touched up to 32 memory
sectors. 1×1 layers' weight gradients ran at 2-5 TFLOPS (e.g. 48→64 at 160×160: 2.5 GFLOP in 0.77 ms).

**Change.** Both generators (regular and tensor-core) load the `B` tile in runs: neighbouring invocations take
`run` neighbouring pixels of one column, then the next columns (`conv::wgrad_b_units`; run 16 in the regular
generator where the tile is that deep, 8 on tensor cores). The pixel `(n, oh, ow)` of each run is decoded once
per load, the tap per element. The k-major workgroup tile is padded so the stores of a run hit distinct banks.
The stem shape (`C_out ≤ 16`, `N ≤ 32`: folo's 3→16 stem, `N = 27`) takes a 16×32 regular tile with 32
invocations, also under AMP (the narrowest tensor-core tile is 16×256, 89 % padding), split-K into up to 1024
chunks.

**Tried on the way** (profiled, alternated with main):

| variant | f32 weight grads (GPU ms) | tensor-core weight grads | note |
|---|---|---|---|
| main | 11.8-12.0 (+0.9 reduce) | 7.0-9.0 (+0.8) | |
| whole warp along `k` (32 / `bk` lanes per column) | 11.5 | 11.0 | tensor cores **3x slower** on 3×3 layers (4-way bank conflicts on the f16 tile stores) |
| runs of 8, padded tile | 10.9 | 6.5 | |
| runs of 16 (f32) / 8 (f16) + stem tile (shipped) | 10.8 (+1.2 reduce) | 5.8 + 0.4 stem | stem f32 0.83 → 0.35 ms, AMP 0.70 → 0.31 + 0.13 reduce |
| runs of 4 / 8 everywhere | 10.6-11.2 | 5.2-5.4 | inside the noise of the shipped variant |

**A/B** (end to end): TODO

## Negative results

- A warp-wide run along `k` (above): coalesced loads, but the tensor-core tile stores conflict 4-way.

## Leftovers

TODO
