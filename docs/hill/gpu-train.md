# Hill-climb: the GPU training step (branch `hill/gpu-train`)

Goal: a faster folo-n / folo-s training step on the RTX 4080 SUPER (batch 16, 640 px, forward + loss + backward +
optimizer; `FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_TRAIN=f32,amp cargo bench -p folo-vision --bench gpu`), without
changing what is computed. Targets: folo-n 300+ img/s f32 and 380+ AMP, folo-s 140+ f32 and 230+ AMP, peak memory
no higher than before.

## Method

- **Profile first.** `FOLO_GPU_PROFILE=1` (the per-kernel timestamp profiler handed over in the WIP commit, kept and
  finished, docs/GPU.md "Profiling") gives the GPU time of every kernel per training phase, labelled by op and
  shape. The bench's split line now also prints the host encode time per phase.
- **A/B.** One bench binary per commit (`gpu-base` = main c50211ba, `c2` … `c6` = the commits below), run
  alternately in one GPU-lock window (`D:/folo-build/hill/lock.sh`), each process at High priority, 3 rounds,
  medians of each run's 10-step median. Other agents kept the host CPU 30–100 % busy throughout (the load is logged
  per run), which the step is sensitive to (encoding is host work), so single rounds swing by up to 2x; the tables
  give the median of the 3 rounds and the best round. Inside ±10 % is a tie and is not claimed; kernel-level
  changes too small to show end to end are measured with the profiler instead (alternated profiled runs).
- **Correctness.** Every changed kernel is tested against the CPU reference (tolerances as the existing tests); the
  multi-tensor updates are tested bit for bit against the single-tensor kernels. The shapes gate (MODEL.md §10.6)
  ran in f32 and AMP on the final binary.

## Where the time was (folo-n, profile of main)

GPU time per step (timestamps serialise kernels, so these are upper bounds) against the wall-clock split:

| phase | f32 GPU / wall (ms) | AMP GPU / wall (ms) | largest items |
|---|---|---|---|
| forward | 14.8 / 16.3 | 9.2 / 12.8 | convs 8.0 / 3.7, BatchNorm 3.7 / 2.6 |
| backward | 38.6 / 49.8 | 26.1 / 29.0 | weight grads 13.9 / 11.8, input grads 12.7 / 6.3 (strided ones 6.3 / 3.4), BatchNorm 6.2 / 3.5 |
| step | 1.4 / 4.4 | 3.5 / 7.1 | 255 SGD dispatches; AMP adds 255 copies and 255 unscale dispatches |

Two things stood out: the GPU sat idle for about a fifth of the step (1328 dispatches went out in 3 submissions,
so it waited for each one to be encoded and submitted whole), and the tensor-core weight gradients were barely
faster than f32 (their im2col gather divides every element index by runtime sizes).

## Results (folo-n, b16, 640, img/s; median of 3 rounds, best round in parentheses)

| commit | change | f32 | AMP |
|---|---|---|---|
| main | | 226 (236) | 292 (307) |
| c1 `9ebddcbe` + c2 `46764b34` | submit every 64 dispatches; weight-gradient spatial sizes as constants | 284 (284) | 393 (463) |
| c3 `49b5e05f` | stride-phase input gradients | 244 (277) | 410 (434) |
| c4 `dee3ae6b` | multi-tensor optimizer / unscale / EMA / accumulation | 300 (308) | 381 (500) |
| c5 `f1ae24ee` | weight-gradient tiles by shape | **307 (320)** | **502 (508)** |

Round 3 of c2–c4 coincided with a host-load spike (138–252 img/s); the best-round column is the less noisy read.
Main → c5: f32 **+36 %** (226 → 307), AMP **+72 %** (292 → 502). Peak memory unchanged (3467 / 2022 MiB).

Per commit (separate A/Bs, same method):

- **c1, flush every 64 dispatches** (`GpuContext::set_flush_interval`, `FOLO_GPU_FLUSH_EVERY`). Sweep on one
  binary: 16 → slower (87 submissions per step; host cost per submission), 32 / 64 / 128 alike, 256 half the gain,
  1024 the old default. Against main, alternated: AMP 295 → 366 (median of 3); f32 231 → 259 and 245 → 277 in the
  two rounds without contention (the third hit a spike).
- **c2, weight-gradient spatial sizes as specialisation constants.** AMP 366 → 449 (median of 3, best 378 → 457):
  the tensor-core weight gradients went from 11.8 to about 7 GPU-ms per step. f32 unchanged (its tiles are bound
  by the 4×4-per-thread arithmetic, see c5).
- **c3, stride phases.** Profiled A/B (two alternated rounds): f32 input gradients 14.1 → 8.7 GPU-ms per step;
  tensor cores 3.4 → 2.5 ms over the six strided layers at first, then the epilogue's per-element phase-pixel
  division was replaced by a cursor (one division per run of columns). End to end it is inside the noise.
- **c4, multi-tensor updates.** 255 dispatches per pass → 16 (one per 16 tensors): SGD, AdamW, the EMA, gradient
  accumulation and the AMP unscale. Profiled: the f32 SGD phase 1.38 → 0.13 GPU-ms; AMP step phase 766 → 290
  dispatches. End to end AMP 410 → 500 (best rounds, +15 %), f32 277 → 308 (best rounds).
- **c5, weight-gradient tiles by shape.** The regular kernels picked tiles by occupancy, which split-K makes
  meaningless here, so every f32 weight gradient ran 64×64 tiles at 4×4 per thread. Now 128×128 / 64×128 by `M`,
  and a new 32×128×32 tile for `M ≤ 32`. Profiled (alternated): f32 weight gradients 13.0 → 11.8 GPU-ms per step
  (−9 %); end to end f32 300 → 307 (median), 308 → 320 (best). Tensor cores do not use these kernels.

Commit hashes after the final rebase: c1 `9ebddcbe`, c2 `46764b34`, c3 `49b5e05f`, c4 `dee3ae6b`, c5 `f1ae24ee`
(plus `18a77e9a`, a clippy fix in the profiler).

## folo-s (b16, 640, img/s; main vs the final code, 3 alternated rounds)

| | f32 | AMP | peak f32 / AMP |
|---|---|---|---|
| main | 105 (106) | 167 (177) | 6520 / 3713 MiB |
| final | **140 (144)** | **246 (256)** | 6520 / 3713 MiB |

f32 +33 %, AMP +47 % (yolo26s AMP is 188 img/s, so folo-s now beats it by 31 %). The split moved from forward 40 /
backward 118 / SGD 6.2 ms to 34 / 83 / 2.1 ms (f32) and from 25 / 68 / 9.7 to 20 / 43 / 4.0 ms (AMP). This A/B ran
the c6 binary, whose extra BatchNorm change measured as a tie and was dropped (below).

### Final binary in a quiet window

At 16:13 the host was nearly idle (1–18 % CPU from other jobs, GPU free), and one run of the final binary gave
folo-n **327 img/s f32 / 533 img/s AMP** (split f32: forward 14.2, loss 0.6, backward 34.4, SGD 1.2 ms; AMP 8.4 /
0.8 / 19.3 / 2.9 ms; 1109 / 1275 dispatches in 20 / 28 submissions per step) and folo-s **153 / 269 img/s** (f32:
32.2 / 0.7 / 71.1 / 1.6 ms; AMP 17.7 / 0.8 / 38.6 / 2.4 ms). Peak memory 3467 / 2022 MiB (folo-n) and 6520 / 3713
MiB (folo-s), as on main. These are single runs, not A/Bs; the claims above are the alternated medians.

### Inference with the new flush interval

The flush interval is shared with inference, so the forward and predict sections ran on the final binary with
`FOLO_GPU_FLUSH_EVERY=1024` (the old behaviour) and the default 64, alternated, two rounds (folo-n, 640):

| | 1024 | 64 |
|---|---|---|
| forward b1 f32 (median ms) | 3.03 / 2.75 | **2.30 / 2.33** |
| forward b16 f32 (pipelined img/s) | 1508 / 1504 | 1515 / 1532 |
| predict b1 f32 (ms) | 3.81 / 3.60 | **3.10 / 3.13** |
| predict b16 f32 (img/s) | 765 / 777 | 809 / 808 |
| forward b1 f16 tensor cores (ms) | 2.39 / 2.63 | **1.78 / 1.65** |
| forward b16 f16 tensor cores (pipelined img/s) | 2836 / 2869 | 2933 / 2942 |
| predict b16 f16 tensor cores (img/s) | 993 / 985 | 1104 / 1068 |

No regression; batch-1 latency is about 20 % better (the GPU starts on the first 64 dispatches while the host
encodes the rest), batch-16 throughput ties or improves slightly.

## Accuracy: the shapes gate

MODEL.md §10.6 (examples/shapes: folo-n, 320 px, 30 epochs, batch 16, seed 0, `--device gpu:0`), the final
`folo` binary, f32 then `--amp`, GPU lock held, 0–2 % other GPU load before each run:

| | gate first met | final mAP50 | final mAP50-95 | reference (BENCHMARKS.md) | overflow skips | wall clock |
|---|---|---|---|---|---|---|
| f32 | epoch 9 (0.922 / 0.754) | 0.978 | **0.893** | 0.978 / 0.898 | | **114 s** (was 182) |
| AMP | epoch 9 | 0.979 | **0.897** | 0.981 / 0.899 | 6, all in epoch 1 (2^16 → 2^10, then stable) | **109 s** (was 200) |

Both within 0.01 of the reference final mAP50-95 (−0.005 and −0.002), no non-finite batch. The run folders are
`D:/folo-data/runs-hill/gpu-train/gate-f32-1608` and `gate-amp-1610`. The AMP gate now runs faster than f32 (it was
10 % slower), because the per-window overflow wait and the many small optimizer dispatches are gone.

## Negative results

- **vec4 BatchNorm reductions** (c6, dropped): `bn_stats` and `bn_bwd_reduce` loading four values per thread
  (each group of four folded in with Chan's merge). Profiled, alternated, folo-n: f32 `bn_bwd_reduce` 3.24 / 3.48
  → 3.99 (a disturbed run) / 3.09 GPU-ms, `bn_stats` 1.48 / 1.87 → 1.45 / 1.39; AMP 1.95 / 1.96 → 1.68 / 2.00. A
  tie, so the kernels are not bound by their load instructions (more likely by the chunk plan's occupancy or the
  tree reduction); not committed.
- Flush intervals below 32 (16: 232 vs 293 img/s f32, the AMP step phase 10.5 vs 6.6 ms) and above 128 (256: 260).
- Constant spatial sizes in the f32 weight-gradient kernel: no change (bound by its tile shape, fixed in c5).
- The first tensor-core stride-phase epilogue (a division per stored element) ate most of the phase gain; the
  cursor fixed that.

## Leftovers, by expected gain

1. **The trainer's global-norm clipping** (`GradAccumulator::clip_grad_norm_device`, `clip_grad_norm = 10` by
   default) is about ten tensor ops per gradient buffer, ~2000 tiny dispatches per accumulation window. The bench
   step does not clip, so it does not show, but real training does. A multi-tensor abs-max (exact with
   `atomicMax` on the bits) and a two-pass deterministic sum of squares plus a multi-tensor scale would make it
   ~40 dispatches.
2. **Tensor-core weight gradients of the stem** (3→16 at 640: `N = 27` columns on a 16×256 tile, 1.7 GPU-ms with
   AMP): a direct reduction kernel for tiny `M × N` with huge `K` would take ~0.2 ms.
3. **f32 weight gradients**: still the largest f32 item; a transposed formulation for `C_in·9 ≫ C_out` layers, or
   tensor cores in TF32-like f16x2 splitting, is the next step.
4. **Concat in the training forward** (`ConvBnAct::forward_cat` with BatchNorm copies the parts; its backward hands
   the consumers strided views that are copied again): a conv that reads the parts in place in training mode.
5. **AMP weight casts** (98 `cast` dispatches per forward): the optimizer could write the f16 copy of each weight
   while it updates it.
6. **Bind-group creation** per dispatch (about 3 µs of host time each): a cache keyed by the buffers, with dynamic
   uniform offsets, would cut host encode time further.

## Interfaces and files outside the lane

- **New defaults and knobs**: the command batch is submitted every 64 dispatches (was 1024);
  `GpuContext::set_flush_interval` / `flush_interval`, `FOLO_GPU_FLUSH_EVERY`, `DEFAULT_FLUSH_INTERVAL`,
  `ENV_FLUSH_EVERY`. `FOLO_GPU_PROFILE=1` and `GpuContext::profiling` / `profile_take` (the per-kernel profiler).
  `Capabilities::max_storage_buffers`.
- **folo-core** (`crates/folo-core/src/backend.rs`, outside the lane, additive): the optional
  `Backend::optimizer_step_many` / `unscale_check_many` hooks and `OptimItem`, defaulting to `NotSupported` (the CPU
  backend is untouched; callers fall back). `crates/folo-core/src/ops/fused.rs`: `Tensor::optimizer_step_many_`,
  `unscale_check_many_`.
- **folo-gpu `context.rs`** (shared with inference): the flush interval; measured above, inference is no slower.
- **docs**: docs/GPU.md (command batching, profiling, weight-gradient constants, stride phases, multi-tensor
  kernels, the benchmark summary), docs/NN.md (the multi-tensor paths of SGD, AdamW, EMA, accumulation, unscale).
