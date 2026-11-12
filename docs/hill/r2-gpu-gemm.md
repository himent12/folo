# Hill-climb 2: forward GEMM / conv kernels (lane `gpu-gemm`)

Goal: faster forward GEMM / conv kernels (f32 and f16 tensor cores) for folo-n / folo-s at 640 px, batch 16
first, without a slower batch 1 or train step. Baseline `main` 647b981f. RTX 4080 SUPER, Vulkan, driver 621.01.

## Method

- **Where the time goes**: `cargo bench -p folo-vision --bench gpu_profile` (`FOLO_BENCH_SCALES=n,s
  FOLO_BENCH_PRECISION=f32,f16tc FOLO_BENCH_BATCH=1,16 FOLO_BENCH_OPS=1`, `FOLO_GPU_TENSOR_CORES=1` for f16tc):
  every dispatch of one fused forward replayed on its own. Its replay sums are the per-kernel numbers below.
- **Per layer**: `cargo bench -p folo-gpu --bench conv_model` (added here): the 17 weakest forward convs of
  folo-n / folo-s at batch 16 with their occurrence count; `FOLO_CONV_TUNE=1` times every tile config at 1–8
  split-K chunks, `FOLO_CONV_GEMM=1` the plain GEMM of the same `M, N, K` (the conv without its gather).
- **Headline**: `cargo bench -p folo-vision --bench gpu` (`FOLO_BENCH_SECTIONS=forward,train
  FOLO_BENCH_PRECISION=f32,f16tc FOLO_BENCH_TRAIN=f32,amp`), the binaries of both trees copied aside and run
  alternately (order flipped each round) inside one hold of the GPU lock, 3 rounds, `nvidia-smi` under 10%
  before each run (`D:/folo-data/runs-hill2/gpu-gemm/ab.sh`, summaries by `summ.py`, raw outputs in `ab/`).
  Tables give the median and the min–max of the three rounds.

## Profile of `main` (batch 16, f32, replay sums)

| | folo-n | folo-s |
|---|---|---|
| all dispatches | 8.50 ms | 22.02 ms |
| GEMM / conv (82 dispatches) | 6.75 ms (79%) | 17.44 ms (79%) |
| concat / split copies (`scatter`) | 0.89 ms (10.5%) | 2.31 ms (10.5%) |
| everything else | < 2.3% each | < 2.8% each |
| f16 tensor cores: `coop_gemm` | 3.53 ms of 4.47 (79%) | 9.55 ms of 11.69 (82%) |

Per layer (`conv_model`, `main`, f32, batch 16, `tune_base.txt`): the convs at 80 px and above run at 21–23
TFLOPS; the 3×3 convs at 40 px at 12–19 (17–19 but for folo-n's stride-2 128 → 128 at 14 and its narrow 32 → 32
and 64 → 32 at 12–13) and at 20 px at 10–14 (folo-n's 64 → 64 at 20 px: 64 workgroups for 80 SMs, 10 TFLOPS;
folo-s's 512 → 64 head conv at 20 px: 12 TFLOPS). Tile shape alone does not fix them: the best of the five
configs is at most about 11% better than the picked one (64×64 on folo-n's 3×3 s2 128 → 128 at 40 px, 131.8 →
117.2 µs, and on folo-s's 3×3 128 → 128 at 20 px, 131.6 → 116.8 µs), and no better on most cases.

The plain GEMM of the same `M, N, K` (`FOLO_CONV_GEMM=1`, `gemm_cmp.txt` / `gemm_cmp2.txt`; the plain GEMM never
splits) against `main`'s unsplit conv (`tune_base.txt`, a separate run: the conv rows of `gemm_cmp*.txt` ran on a
split-K build and are partly split, so they are no like-for-like comparison): at 20 px it is slower on three of
the four 3×3 convs (2–12%) and 5–7% faster on folo-s's 128 → 128. At 40 px it is slower on folo-n's two narrow
convs (32 → 32 and 64 → 32, 14–22%), level on folo-n's head 128 → 64, and 5–17% faster on the other seven, most
on the 64 → 64 convs (folo-n 107.7 vs 91.7 / 89.2 µs, folo-s 105.3 vs 92.0 / 94.5 µs). So the gather costs up to
~15% on some 40 px layers, but it is not what makes the 20 px layers slow. At 80 px the plain GEMM is 3–5% faster
for folo-n's 64 → 128 and 12–13% for folo-s's 128 → 256 (665.7 vs 581.4 / 583.1 µs, 23 vs 26 TFLOPS).

## PR 1: split-K for forward convs with few output tiles (f32, f16, f16 tensor cores)

A forward implicit-GEMM conv with fewer than 320 workgroups and at least 18 `BK` tiles of `K` splits `K` into
up to 8 chunks, aiming at ~480 workgroups (six per SM: the last partial wave of 2–3 resident workgroups then
costs little); each chunk writes an f32 partial slab without bias or activation and `conv_split_finish` sums the
slabs with the bias and activation. Same plan on tensor cores. docs/GPU.md, *Forward split-K*.

Tuning (`conv_model`, f32, batch 16, µs; noisy runs). Columns: **main** is the default row of `tune_base.txt`
(`main`, no split). **Best forced split** is the fastest `<tiles> split <n>` row of `tune_split1.txt`
(`FOLO_CONV_TUNE=1`; main's tiles unless named). **Earlier-rule build** is the default rows of `gemm_cmp.txt` /
`gemm_cmp2.txt`: a split-K build with an earlier plan than the current 480 / 320 one (it did not split folo-s's
256 → 512; the default rows of `tune_split1.txt` are an earlier plan too, `TARGET` 320 / `MIN` 160, and noisier,
so they are not used). **Chunks now** is what the current plan (`split::plan`) gives with main's tiles; no file
holds a run of the current build's default rows.

| case (main's tiles, workgroups) | main | best forced split | earlier-rule build | chunks now |
|---|---|---|---|---|
| n 3×3 64 → 64 @ 20 (64 × 128, 64 wg) | 46.1 | 41.6 (2 chunks) | 40.5–43.5 | 4 |
| n head 256 → 64 @ 20 (64 × 128, 64 wg) | 156.1 | 118.4 (64×64, 8) | 111.5–120.2 | 8 |
| n 3×3 s2 128 → 128 @ 40 (128 × 128, 64 wg) | 131.8 | 105.6 (6) | 99.5–100.3 | 8 |
| s 3×3 128 → 128 @ 20 (128 × 128, 64 wg) | 131.6 | 105.6 (6) | 98.3–99.6 | 8 |
| s head 512 → 64 @ 20 (64 × 128, 64 wg) | 307.6 | 224.4 (64×64, 8) | 212.8–213.2 | 8 |
| s 3×3 s2 256 → 512 @ 40 (128 × 128, 256 wg) | 846.0 | 738.2 (3) | 846.7–847.4 (not split) | 2; forced `128x128x8 split 2`: 762.6 |
| s 3×3 s2 256 → 256 @ 40 (128 × 128, 128 wg) | 433.9 | 370.1 (3) | 368.5–369.0 | 4 |
| sum of the 17 cases × occurrences | 6652.4 | | 6034.1–6080.0 | |

A/B, f32, `main` (A) against the split-K build (B), 3 rounds, GPU 0–7% before every run, 2026-11-11:

| row | main | split-K | |
|---|---|---|---|
| folo-n forward b1, `gpu` median | 3.23 ms (2.60–3.25) | 2.23 ms (1.75–2.26) | −31% |
| folo-n forward b1, `gpu_profile` pipelined | 343 img/s (335–393) | 381 img/s (354–396) | +11% |
| folo-n forward b1, GEMM replay | 1.14 ms (1.03–1.50) | 0.62 ms (0.54–0.83) | −45% |
| folo-n forward b16, `gpu_profile` pipelined | 1652 img/s (1627–1658) | 1666 img/s (1650–1684) | tie |
| folo-n forward b16, `gpu` pipelined | 1475 img/s (1280–1618) | 1624 img/s (1610–1640) | A had a slow round |
| folo-n train step b16 f32 | 48.52 ms (48.04–48.95) | 49.12 ms (47.14–49.13) | tie |
| folo-s forward b1, `gpu` median | 4.67 ms (4.62–4.69) | 3.89 ms (3.44–6.68) | −17% |
| folo-s forward b1, `gpu_profile` pipelined | 272 img/s (267–278) | 316 img/s (296–346) | +16% |
| folo-s forward b16, `gpu_profile` pipelined | 668 img/s (666–668) | 684 img/s (683–687) | +2.4% |
| folo-s forward b16, GEMM replay | 22.17 ms (17.97–22.30) | 16.99 ms (16.95–17.55) | min −6% |
| folo-s train step b16 f32 | 113.30 ms (109.98–114.43) | 109.91 ms (107.53–110.25) | −3% |

Second A/B with the tensor-core split too (`gpu`, `FOLO_GPU_TENSOR_CORES=1`, f32 and f16tc forward, f32 and AMP
train step), 3 rounds, GPU 0–1% before every run, 2026-11-11:

| row | main | split-K | |
|---|---|---|---|
| folo-n f32 forward b1 median | 3.06 ms (2.27–3.29) | 2.28 ms (2.12–2.56) | −26% |
| folo-n f32 forward b16 pipelined | 1652 img/s (1652–1669) | 1632 img/s (1600–1695) | tie |
| folo-n f16 TC forward b1 pipelined | 419 img/s (365–766) | 538 img/s (327–929) | bimodal, see round 1 |
| folo-n f16 TC forward b16 pipelined | 3062 img/s (2733–3085) | 3079 img/s (2826–3084) | tie |
| folo-n train step f32 | 47.36 ms (46.34–47.60) | 46.61 ms (45.62–93.74) | tie |
| folo-n train step AMP | 29.39 ms (28.41–30.17) | 30.34 ms (29.12–32.57) | +3.2%, inconclusive (below) |
| folo-s f32 forward b1 median | 4.26 ms (4.07–4.62) | 2.82 ms (2.67–4.04) | −34% |
| folo-s f32 forward b16 pipelined | 612 img/s (607–673) | 657 img/s (622–684) | +7% |
| folo-s f16 TC forward b1 median | 2.97 ms (2.23–3.90) | 2.04 ms (2.01–3.21) | −31% |
| folo-s f16 TC forward b16 pipelined | 1193 img/s (1123–1254) | 1222 img/s (1166–1252) | +2.4%, tie |
| folo-s train step f32 | 105.25 ms (103.23–159.07) | 108.62 ms (108.38–143.25) | tie |
| folo-s train step AMP | 57.48 ms (57.38–57.72) | 59.82 ms (57.48–62.08) | +4.1%, inconclusive (below) |

The AMP medians are 3.2% (folo-n) and 4.1% (folo-s) higher with split-K: inconclusive, neither a tie nor a
shown loss. The slow split-K rounds are slow also in the phases this change does not touch, the loss and the
SGD step (folo-n round B1: loss 0.9 vs 0.6 ms in every `main` round, unscale + SGD 3.9 vs 2.1–2.3 ms; folo-s round
B1: loss 0.8 vs 0.6–0.7 ms, unscale + SGD 3.6 vs 2.2–2.6 ms, leaving out `main`'s 19.2 ms outlier in A2), and
round B3 matches `main` phase by phase (folo-n forward 7.8 vs 7.7–7.9 ms, backward 18.6 vs 18.7–19.1 ms; folo-s
forward 16.9 vs 16.9–17.5 ms, backward 38.8 vs 38.5–41.7 ms). Train mode splits the same convs (18–23 more
dispatches per step).

Train-only A/B to settle it (`gpu`, `FOLO_BENCH_SECTIONS=train FOLO_BENCH_TRAIN=amp,f32`, 5 rounds, GPU 0–2%
before every run, `ab/train5`):

| row | main | split-K |
|---|---|---|
| folo-n train step AMP, median | 32.03 ms (29.69–55.49) | 35.65 ms (30.95–65.66) |
| folo-n train step AMP, forward phase | 7.7–8.5 ms | 7.8–8.5 ms |
| folo-n train step AMP, backward phase | 19.2–21.3 ms | 19.5–21.8 ms |
| folo-n train step AMP, unscale + check + SGD | 2.0–21.8 ms | 2.0–39.0 ms |
| folo-n train step f32 | 50.38 ms (48.44–84.27) | 51.02 ms (49.06–55.89) |
| folo-s train step AMP | 61.08 ms (59.75–98.52) | 61.55 ms (60.16–104.47) |
| folo-s train step f32 | 121.39 ms (115.23–135.78) | 122.19 ms (121.71–207.24) |

The folo-n AMP median difference is all in the unscale + check + SGD phase (the overflow-flag readback), which
swings 2 → 39 ms within each binary and which this change does not touch; the forward phase, the only one split-K
changes, is the same (7.7–8.5 vs 7.8–8.5 ms), and folo-s AMP and both f32 steps are ties. No train-step
regression.

**Accuracy** (`folo val --model D:/folo-weights/folo-n-coco.folo --data D:/folo-data/coco`, COCO val2017, both
builds, GPU lock held):

| | mAP50 | mAP50-95 |
|---|---|---|
| main, f32 | 0.44945 | 0.30999 |
| split-K, f32 | 0.44945 | 0.30999 |
| main, `--half --tensor-cores` | 0.44944 | 0.31003 |
| split-K, `--half --tensor-cores` | 0.44939 | 0.31005 |

Largest gap 0.00005 (bound 0.002).

`gpu_profile` on tensor cores (f16tc, `FOLO_GPU_TENSOR_CORES=1`), the same binaries, 3 rounds, GPU 0–2% before
every run (`ab/prof_tc`):

| row | main | split-K | |
|---|---|---|---|
| folo-n b1, `coop_gemm` replay | 0.63 ms (0.51–4.51) | 0.37 ms (0.33–0.38) | −40% |
| folo-n b16, `coop_gemm` replay | 3.56 ms (3.51–4.24) | 3.50 ms (3.49–4.98) | tie |
| folo-n b16 pipelined | 3085 img/s (2363–3087) | 3065 img/s (2457–3066) | tie |
| folo-s b1, `coop_gemm` replay | 1.01 ms (1.01–1.18) | 0.67 ms (0.61–0.77) | −34% |
| folo-s b16, `coop_gemm` replay | 9.63 ms (9.57–12.46) | 9.41 ms (9.40–12.88) | −2%, tie |
| folo-s b16 pipelined | 1235 img/s (1217–1241) | 1214 img/s (991–1240) | tie |

The batch-1 pipelined rows of this run are bimodal for both binaries (folo-n 53–801 vs 400–867 img/s, folo-s
323–509 vs 305–619): host scheduling, round 1's leftover 2; they settle nothing either way. On tensor cores the
split shortens batch-1 GPU time by a third and leaves batch 16 unchanged: the coop tiles already reach their
peak at 40 px.

At batch 1 every layer below 160 px has only a handful of output tiles, so split-K halves the GEMM time there.
At batch 16 it is a small gain: only the 40 and 20 px layers split, and they are a quarter of the GEMM time.
(The replay medians of `main` swing between rounds, 8.8–11.6 ms for folo-n; the minima are the steadier
comparison.)

Tensor-core split sweep (`conv_model`, `FOLO_BENCH_PRECISION=f16tc FOLO_CONV_TUNE=1`, µs, one noisy run:
the same forced plan measured 70 and 100 µs on one case, so only large differences count): splitting pays for a
long `K` (folo-s head 512 → 64 @ 20 129 → 96, head 256 → 64 @ 40 227 → 174, folo-n head 256 → 64 @ 20 75 → 61,
bottleneck 64 → 32 @ 40 62 → 44) and is flat or a little worse for `K` ≤ 1152 (folo-s c3k 128 → 128 @ 20 62 →
71 with 4 chunks, folo-n down4 64 → 128 s2 @ 80 104 → 125 with 3); the 17 cases sum to 3.62 ms with the plan.
Kept one rule for both paths: batch-16 tensor-core GEMM time is a tie either way (above).

## Fix: `tiny_amp_run_learns` (#193)

Main's Windows CI went red after #186: folo-vision `gpu_amp` `tiny_amp_run_learns` (40 AdamW steps, AMP tail
loss ≤ 1.15 × f32's) failed with seed 0 at ratio 1.19. Not a numerics error: split-K forced off on main reproduces
c0464b29 bit for bit, and there seeds 0–5 give 0.97–1.16 (seed 5 already failed); with split-K 0.91–1.19. One AMP
step of folo-n (128 px, batch 4, same weights): AMP gradients 6.29% (split off) and 6.21% (split on) from f32;
split on vs off moves the f32 gradient by 4.7e-5. #193 set that test's bound to 1.3 (the dual-head test keeps
1.15) and added a tensor-core split-K test on a channel slice read in place.

## PR 2: the forward gather through a per-column tap mask (regular and tensor-core generators)

The implicit-GEMM gather rebuilt, per loaded element, `ci = k / 9`, `r = k % 9`, the row and column, four
bounds checks and the address (about 15 integer operations). Now each thread builds, once before the K loop, its
output pixel's tap mask (bit `kh·KW + kw` set when the tap lies inside the image) and the address of its
window's top-left element; an element is then one division by the constant `KH·KW`, a bit test and an add.
Windows of more than 32 taps (a 7×7 stem) keep the old gather. Both generators. As before, the gather's
`k < K` test is compiled out only when M, N and K are all tile multiples (`CHECK` false), not whenever `K` is.

Tests: the existing CPU-reference conv cases that take this gather (3×3 stride 1 and 2, padding, dilation,
groups, 1×3, 1×1 stride 2, ragged M and N, split-K, channel slices; their 5×5 cases are depthwise, a separate
kernel) plus new ones on both generators: a 4×8 window (exactly 32 taps, so bit 31 is a live tap), 5×5 with
groups 1, and a conv with a ragged `K` on a channel slice read in place whose next channels are NaN (a missing
`k < K` test pulls 0 × NaN in; both this and a dropped bit 31 were checked to fail the tests). On tensor cores
also 32 → 128 channels at 16×16, where M, N and K are all tile multiples (`CHECK` false), whole and in 2 chunks.
The tensor-core conv tests moved to `tests/coop_conv.rs`.

A/B, `main` 5246c0c0 (A) against it (B), `gpu` (forward f32 and f16tc, train f32 and AMP) and `gpu_profile`
(f16tc, batch 16), 3 rounds each, GPU 0–1% before every run, 2026-11-12 (`ab/tc_mask`, `ab/proftc_mask`):

| row | main | tap mask | |
|---|---|---|---|
| folo-n f32 forward b16 pipelined | 2061 img/s (2060–2069) | 2116 img/s (2102–2117) | +2.7% |
| folo-s f32 forward b16 pipelined | 815 img/s (814–815) | 830 img/s (794–832) | +1.8% |
| folo-n f16 TC forward b16 pipelined | 3606 img/s (3592–3629) | 3560 img/s (2718–3688) | tie |
| folo-s f16 TC forward b16 pipelined | 1406 img/s (1405–1408) | 1459 img/s (1450–1459) | +3.8% |
| folo-n f16 TC b16, `coop_gemm` replay | 2.78 ms (2.73–2.79) | 2.67 ms (2.63–2.68) | −4.0% |
| folo-s f16 TC b16, `coop_gemm` replay | 6.93 ms (6.91–6.95) | 6.55 ms (6.54–6.57) | −5.5% |
| folo-n / folo-s f32 forward b1 median | 1.34 / 2.45 ms | 1.29 / 2.28 ms | −4% / −7% (noisy) |
| folo-n / folo-s f16 TC forward b1 median | 1.18 / 1.79 ms | 1.09 / 1.51 ms | −8% / −16% (noisy) |
| folo-n / folo-s train step f32 | 44.30 / 104.33 ms | 44.49 / 103.81 ms | tie |
| folo-n / folo-s train step AMP | 27.79 / 56.21 ms | 27.21 / 55.78 ms | tie |

How far to trust it: the min–max ranges are apart on folo-n f32 b16, folo-s f16 TC b16 and both `coop_gemm`
replays. folo-s f32 b16 overlaps (main 814–815, tap mask 794–832: one slow round), so its +1.8% is a median gain
only. The batch-1 rows are medians of noisy rows whose ranges overlap between A and B: lower observed medians, not
a measured gain, and no proof that batch 1 is not slower. The train steps are ties.

An earlier f32-only A/B of the regular generator alone (`ab/fwd_mask`, 3 rounds, GPU 0–3% before every run)
gave folo-n f32 b16 1993 (1843–2061) → 2107 (2094–2110) img/s and folo-s 808 (796–811) → 827 (821–827).

## Leftovers (by expected gain)

1. **Tensor-core split threshold by `K`**: on tensor cores, a chunk of ≤ 4 `BK = 32` tiles does not pay. A
   coop-specific `MIN_CHUNK_TILES` (about 18) would keep the gains on the long-`K` heads and drop the small
   losses; needs a quiet, repeated sweep (worth ≤ 2% of the f16 batch-16 forward).

## Negative results

- **Tile shape for the 20 / 40 px layers** (`tune_base.txt`, no split): the best alternative to the picked config
  is at most 11% faster on any weak case (64×64: n 3×3 s2 128 → 128 @ 40 131.8 → 117.2 µs, s 3×3 128 → 128 @ 20
  131.6 → 116.8 µs, n head 64 → 64 @ 40 107.7 → 96.4 µs) and no better on most; the others are often much
  slower (128×128 and the narrow configs up to 2.2×). Not worth a picker change on its own.
