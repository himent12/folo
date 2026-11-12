# Hill-climb 2: FOLO on the CPU (lane `cpu`, 2026-11-11)

Goal: CPU batch-1 latency, the headline row round 1 ([cpu.md](cpu.md)) did not move (folo-n forward 18.3 ms,
end to end 18.6 ms at 28 threads; target 12 ms), then per-layer outliers and the CPU training leftovers.
Baseline: `main` at 647b981f. Machine: i7-14700KF (8 P + 12 E cores, 28 threads, AVX2 + FMA), shared with five
other lanes' builds, benches and training runs.

## Method

- Baseline binaries are the bench binaries of this branch built before any change (the branch started at
  647b981f), copied to `D:/folo-build/hill2-cpu-bin/base-<bench>.exe`; each change's binaries are copied next to
  them (`<label>-<bench>.exe`). Square-sgemm baseline: `D:/folo-wt/hill2-base` (647b981f).
- `tools/bench-ab.ps1` alternates the binaries run by run at High priority, inside the shared `cpu` lock
  (`D:/folo-build/hill/lock.sh acquire cpu cpu`), 3 rounds unless stated. A copy of it (`ab2.ps1`, not committed)
  takes per-run environment variables (`label=exe|K=V,K=V`), for sweeping knobs inside one binary.
- Benches: `cargo bench -p folo-vision --bench cpu_e2e`, `--bench train_step` (folo-n, 320 px, batch 16),
  `cargo bench -p folo-core --bench gemm`. Rows below are medians of the rounds' medians, ranges in brackets.
- The PR 1 table's runs, each `bash D:/folo-build/hill2-cpu-bin/ab.sh 3 <threads> <affinity> <env>
  "base=base-<bench>.exe;pr1=pr1-<bench>.exe"` (threads 0 = all 28; raw logs in the same folder):

  | rows | bench | threads, affinity | env | log |
  |---|---|---|---|---|
  | folo-n forward, e2e | `cpu_e2e` | 28 | `FOLO_BENCH_ITERS=60` | `pr1_n28.txt` |
  | folo-n forward, 8 P-cores | `cpu_e2e` | 8, `5555` | `FOLO_BENCH_ITERS=60` | `pr1_n8.txt` |
  | folo-s forward, e2e | `cpu_e2e` | 28 | `FOLO_BENCH_SCALE=s`, `FOLO_BENCH_ITERS=40` | `pr1_s28.txt` |
  | folo-m forward, e2e | `cpu_e2e` | 28 | `FOLO_BENCH_SCALE=m`, `FOLO_BENCH_ITERS=20` | `pr1_m28.txt` |
  | `predict_batch` 16 | `cpu_e2e` | 28 | `FOLO_BENCH_BATCH=16`, `FOLO_BENCH_ITERS=40` | `pr1_b16.txt` |
  | train step | `train_step` | 28 | none (defaults) | `pr1_train.txt` |

  The square-sgemm numbers (`--bench gemm`) and the steps on the way are observations from runs whose output
  was not saved to a file.
- "busy" is the CPU time of every other process over the run's wall time, in cores. **The machine was never
  quiet: 6–21 cores busy elsewhere in every A/B and sweep run**, 11–16 in most. The 8-pinned-P-core rows
  (`-Threads 8 -Affinity 5555`) were the noisiest (17–20 busy), so the claims use 28-thread rows.

### Profiling

samply records nothing on this machine, so, as in round 1, an op timer was patched in for the profile runs and
never committed (`prof.patch`, `prof2.patch` in `D:/folo-build/hill2-cpu-bin`): a guard on the CPU backend methods (conv label, flops), sub-phase guards inside the conv forward
(shifted-rows copy, A packing, the tile loop), and a per-task record (thread index, start, end) inside every GEMM
tile loop, printed by a temporary `FOLO_PROF=1` section of `cpu_e2e`. Findings on `main`, folo-n, 640², batch 1:

- **28 threads, 16 cores busy elsewhere, forward 17.2 ms.** Before the head (13.4 ms of it): GEMM tile loops 8.6
  ms, shifted-rows copies 1.3, **A packing 1.1** (a serial pass on the calling thread before every conv GEMM:
  `pack_lhs_panels` goes parallel only from 128K floats, and conv weights are 2K–295K floats), small ops 1.6
  (3 max pools 0.40, the PSA attention's two batched matmuls 0.45, softmax 0.17, `mul_scalar` 0.14, adds 0.42).
  The head ran 2.7 ms, all six branches side by side, the P3 box branch (the two largest convs of the model,
  64→64 3×3 at 80², 0.47 GFLOP each) its critical path.
- **Per-task records.** In a typical conv region about 16 workers join within 1–5 µs (spinning) and the rest
  20–130 µs late (woken from sleep); busy / (wall × workers used) over all regions was 0.48. Summed over the
  backbone and neck, task time was 52.8 ms at 1 thread, 68.6 ms at 8 pinned threads and 113.5 ms at 28 threads:
  a task runs about 2.15× slower at 28 threads, which matches hyper-threads sharing a P-core and E-cores at about
  half a P-core's AVX2 rate, not cache misses or tiling.
- **Thread count per layer does not help under this load** (sweeps below): fewer tasks for small layers, larger
  tasks, more tasks per thread, a smaller pool, and running the neck's tail beside the head were all ties or
  losses. With 11–16 other cores busy the forward is bound by the CPU time it needs, not by fork-join waits:
  only less work per forward moved it.

## 1. Conv forward without serial packing (PR 1)

Three changes, all bit for bit the old results:

- **A read in place.** A row-major A (`Operand::rows_in_place`; conv weights are `Strided` with unit K stride)
  whose packed form fits one K super-block (`PACK_BUDGET`, 16 MiB) is no longer packed: two new micro-kernels
  broadcast the six A values of each K step from six row pointers (`rows_a_kernel` for the shifted-rows B,
  `arows_kernel` with an epilogue for packed B and for the ragged last column panel). Same FMAs in the same
  order, same `KC` block boundaries, rows past `mr` repeat row 0 and are never stored. Removes the serial A pass
  from every conv forward and pointwise conv, and the packed copy (peak memory of the train step −30 MB).
  Larger A (a 2048² sgemm) keeps the packed path.
- **Max pooling without indices** when nothing is recorded (`no_grad`, or an input that does not require grad):
  `Backend::max_pool2d_values` (default: `max_pool2d` and drop the indices), on the CPU two passes of a select
  over `-inf`-padded rows (same `beats` rule; `-inf` padding never changes a value, see the doc comment), planes
  grouped into ~4K-output tasks. The indexed version allocated a fresh 410 KB `i64` buffer per SPPF pool (page
  faults) and wrote indices nobody read.
- **Batched `matmul` splits the pool between its matrices** (`threads / batch` each) instead of running small
  ones (< 16 MFLOP) one thread each: the PSA attention's two heads were two single-thread 10 MFLOP GEMMs
  (0.25 → 0.05 ms and 0.20 → 0.09 ms).

Tests: `a_in_place_is_bitwise_the_packed_path` (gemm_ex and gemm_rows, row-major vs column-major A, ragged
blocks, K over one block, the K split, 1 and 8 threads, with and without bias + SiLU),
`a_rows_kernels_are_bitwise_the_packed_kernel` (portable and AVX2 kernels, ragged rows, epilogue),
`max_pool_values_are_the_indexed_values_bit_for_bit` (±0 ties, NaN payloads, `-inf` windows, planes smaller
than the window, strides 1–3, paddings), `max_pool_with_huge_windows_or_empty_planes` (windows up to `usize::MAX`
wide fall back to the clipped scan instead of padding rows that long; empty planes give `-inf` on both paths),
`max_pool_makes_indices_only_when_recorded` (a counting backend: indices only when the result is recorded),
`rejects_overflowing_a_row_stride` (A's extent is checked without overflow before its rows are read in place).

| A/B, 28 threads, 3 rounds alternated | `main` | PR 1 | |
|---|---|---|---|
| folo-n forward b1 (busy 9–13) | 16.37 ms (16.19–16.68) | **14.55 ms** (13.99–14.75) | −11% |
| folo-n e2e 640×640 | 16.78 ms (16.58–17.06) | **14.79 ms** (14.27–14.99) | −12% |
| folo-n e2e 1280×720 | 12.43 ms (12.36–13.08) | **11.10 ms** (10.60–11.25) | −11% |
| folo-s forward b1 (busy 11–12) | 36.34 ms (36.21–36.85) | **32.23 ms** (32.05–32.56) | −11% |
| folo-s e2e 640×640 | 35.71 ms (35.59–36.23) | **32.21 ms** (31.97–32.37) | −10% |
| folo-m forward b1 (busy 8–10) | 87.58 ms (85.96–88.17) | **79.31 ms** (78.18–80.70) | −9.4% |
| folo-m e2e 640×640 | 85.81 ms (84.30–88.17) | **79.63 ms** (78.59–80.19) | −7.2% |
| folo-n `predict_batch` 16 (busy 11–12) | 153.9 ms (150.6–156.9) | 153.5 ms (150.9–154.5) | tie |
| folo-n train step 320 px b16 (busy 11–15) | 263.0 / 263.7 / 256.6 ms | 256.0 / 255.8 / 256.2 ms | tie (−2%) |
| folo-n forward b1, 8 pinned P-cores (busy 17–20) | 20.49 ms (19.30–22.96) | 16.18 ms (15.27–18.96) | noisy |

Square sgemm (`--bench gemm`, best of 1.5 s, 3 rounds at 28 threads, 2 at 1 thread on one P-core): 28 threads
256³ 431–436 → 689–723 GFLOP/s, 512³ 883–917 → 1090–1102, 1024³ 1208–1217 → 1248–1304, 2048³ (packed, unchanged
path) 1174–1197 → 1170–1181. One thread: 256³ 116–119 → 121–127, 512³ 102–104 → 108–111, 1024³ and 2048³ within
the noise of a loaded core (2048³ 79–80 vs 64–73 when it was read in place too, which is why A over one
super-block keeps packing).

Steps on the way (forward b1, 28 threads, same session each): in-place A alone 16.98 → 15.47 ms; the first
values-only max pool (a scalar flag scan, planes in 8K-output tasks) made the pools slower (3 × 0.24 ms against
3 × 0.13), the padded select version 15.45 → 15.01 together with the matmul split.

## Negative results

All in one binary with temporary environment knobs, uncommitted like the op timer (the tiling knobs are in
`prof.patch`, the head overlap and its thread budget in `overlap-budget.patch`, both in
`D:/folo-build/hill2-cpu-bin`), alternated, folo-n forward b1 at 28 threads (11–17 busy):

| knob | forward b1 |
|---|---|
| `main` tiling (3 tasks per thread, ≥ 1 MFLOP each) | 16.84 / 17.00 ms |
| tasks ≥ 2 / 4 MFLOP | 17.79 / 21.41 |
| tasks ≥ 0.5 / 0.25 MFLOP | 16.68 / 17.01 |
| 6 tasks per thread (and ≥ 0.5 MFLOP) | 17.05 (16.90) |
| conv threads capped at flops / (2, 4, 8 MFLOP) | 17.11 / 18.03 / 19.59 |

| pool size (`RAYON_NUM_THREADS`, same binary) | 28 | 24 | 20 | 16 | 12 |
|---|---|---|---|---|---|
| forward b1 | **17.02** | 18.01 | 18.63 | 19.69 | 23.55 |

- **The neck's tail beside the head**: on CPU inference, start each level's head branches as soon as the neck
  has made its feature (`rayon::join` of the P3 branches with layers 17–22, then P4 with 20–22): 14.71 vs 14.68
  ms, a tie. The overlap happens, but the neck's convs, the critical path, then share the pool (down3 0.28 → 1.08
  ms). Giving the P3 / P4 branches a thread budget (a thread-local cap on the tiling of the kernels they start:
  P3 box 2–12, cls 1–4) made it worse: 15.34–17.85 ms against 14.86.
- **Shifted-rows copy in parallel from 8K / 16K / 32K floats** instead of 64K: 15.50 / 14.94 / 15.81 ms against
  15.01, a tie.
- Not tried: pinning workers to P-cores first. Every pool smaller than 28 threads lost under this load, and a
  pinned 8-thread pool was the noisiest configuration measured.

## 2. CPU training: BatchNorm statistics in one pass (PR 2)

**Profile of the train step** (folo-n, 320 px, batch 16; the op timer of PR 1 plus guards on the conv input
and weight gradients and on folo-nn's train-mode BatchNorm, `prof3.patch`). At 28 threads (257 ms a step,
12 cores busy elsewhere): BatchNorm forward 29 ms and backward 34 ms (64 M activations a step), weight
gradients 66 ms (25 of them pointwise), input gradients 41 ms (15 of them strided), forward convs 37 ms,
`cat` copies 9 ms, adds 8 ms. On one core (1215 ms): BatchNorm 335 ms (28%), about 11 cycles per element in
the forward. The build targets baseline x86-64, so its loops ran as 4-lane SSE2. At 28 threads the backward
streams 6 floats per element at ~60 GB/s: memory-bound.

Changes, in folo-nn's `train_cpu` (`crates/folo-nn/src/layers/batchnorm.rs`):

- **One read of x for the statistics** (`plane_stats`), so the forward reads x twice instead of three times.
  Each plane is taken in blocks of 256 floats. A block first feeds the plane's sum: the old code's 8 `f64`
  accumulators across the whole plane, then the tail, so the per-channel sum (planes added in image order) and
  the mean `Σ / n` are bit for bit the old ones. Then, while the block is in L1, its own mean and
  `M2 = Σ(v − mean_b)²` in `f64`. The blocks are merged per plane, and the planes per channel in image order,
  with Chan's formula (`M2a + M2b + d² · na · nb / n`, `d` the difference of the means); the variance is
  `M2 / n`. Every term is non-negative, so nothing cancels and nothing needs a clamp, and an infinity or a NaN
  makes the variance NaN as before. The variance is not bit for bit the old two-pass sum (it differs in the
  last f64 bits). A first draft shifted each plane by its first value and took `Σd² − (Σd)² / n`, which
  cancels when that value is extreme (a 1000×1000 plane of 0.1 with 10⁶ first: variance 1000007 against
  999999) and lost small terms of the mean (`[1e16, −1e16, 1, 1]`: 0 against 0.5); review caught it. Test
  `one_pass_statistics_match_two_pass_reference` checks the mean bit for bit against the old formula and the
  variance within a relative 2·ε (f32; 2 to 4 ulp) of a two-pass f64 reference: means 10⁴ and 10⁷ times the spread (the second fails
  `Σv² − n·mean²`, asserted), a constant channel, 1×1 planes, one outlier per plane, both review cases,
  infinities and NaNs.
- **AVX2 for the per-plane loops** (`with_avx2`): the same loop bodies compiled under
  `#[target_feature(enable = "avx2")]`, chosen at run time. The same IEEE operations lane by lane, and Rust
  never fuses a multiply and an add without being asked, so the result is bit for bit the SSE2 one. Test
  `avx2_and_fallback_are_bitwise_equal` runs the SiLU kernel forward and backward both ways (a test-only
  switch forces the plain build) and compares output, statistics and gradients bit for bit.
  `fused_activation_is_bitwise_bn_then_act` (now with a 33×17 plane, three statistics blocks and a ragged
  tail) still holds bit for bit.

| folo-n train step, 320 px b16, 28 threads, 5 rounds alternated (9–15 busy) | `main` (62f80c4a) | PR 2 |
|---|---|---|
| forward + loss | 80.0–85.6 ms (median 84.4) | **76.3–77.8 ms (76.8), −9%** |
| backward | 170.2–178.7 ms | 173.3–183.6 ms (tie) |
| total | 251.5–265.4 ms (median 263.9) | 251.0–262.8 ms (256.0), −3.0% (tie: inside the ±3% band) |

On one core with 27 cores busy elsewhere, forward + loss went from 689 to 612 ms and backward from 1296 to 1285
ms. Log: `pr2_train.txt` (`ab.sh 5 0 "" "" 'main=...pr1-train_step.exe;pr2=...pr2-train_step.exe'`); the `main`
binary is PR 1's head, whose train-step code is `main`'s. These numbers are of the first draft of the
statistics (shifted sums, one loop per plane). The block design above (two loops over each block in L1), the
one merged, measured again the same way (log `pr2b_train.txt`, 12–15 cores busy elsewhere):

| folo-n train step, 320 px b16, 28 threads, 5 rounds alternated | `main` | PR 2 (block statistics) |
|---|---|---|
| forward + loss | 80.9–83.4 ms (median 81.4) | **74.6–76.5 ms (75.8), −7%** |
| backward | 167.5–175.1 ms | 171.2–175.8 ms (tie) |
| total | 249.8–258.1 ms (median 256.9) | 247.9–253.9 ms (250.9), −2.3% (tie) |

### Negative results (PR 2)

- **The backward without the stored `g'`**: both passes recompute `g' = g · act'(z)` from g and x a block at a
  time (one write of the tensor less, bit for bit the same sums): backward 171–177 → 186–201 ms. The passes
  are compute-bound once `act'` runs twice, as in round 1. Reverted.
- **Strided-conv input gradients by stride phases** (each input phase a stride-1 GEMM over a padded copy of the
  output gradient, read in place, with no unfolded gradient and no col2im; `phases-dx.patch`). Same flops,
  within the f32 summation bound, and it passed the reference tests. `cargo bench -p folo-core --bench conv`,
  2 rounds at 28 threads, input gradient ms:

  | shape (batch 8) | `main` | phases | phases, weights built once, rows written once |
  |---|---|---|---|
  | 64→128 3×3 s2 at 80² | 2.38–2.46 | 2.78–2.78 | 2.93–3.08 |
  | 16→32 3×3 s2 at 160² (down2) | 1.25–1.26 | 1.09–1.15 | 1.03–1.06 |
  | 3→16 3×3 s2 at 320² (stem) | 1.31–1.35 | 1.03–1.06 | 1.15–1.17 |

  On one core it was faster everywhere but the inference stem (64→128: 26.3 → 20.5 ms). At 28 threads it wins
  only where the input has few channels; folo-n's strided convs mostly have 64 or more. Train step with the
  phases 260.8–268.2 ms (median 262.9) against 258.3–261.9 (median 261.2) without it (log `ph_train.txt`); the
  log of the conv-bench table above was not saved. Dropped.

## Leftovers (by expected gain)

- **3×3 stride-1 convs** are 37% of the forward's flops (2.39 of 6.48 GFLOP); Winograd F(2×2, 3×3) would cut
  their multiplies 2.25× (not bit for bit; within an f32 tolerance).
- **Stride-2 convs** run at 41–73 GF/s on one core against 76–88 for stride 1 (16→32 at 320², 64→64 at 160²):
  their phase copies leave L2 and `m = 32` wastes a third of the last micro-panel.
- The shifted-rows copy (1.3 ms of the 28-thread forward) could be written by the conv that produces the input
  (a padded output layout), the head's depthwise convs scale poorly (0.26 ms at 28 threads for 0.5 ms at one),
  and softmax over the PSA's 800 rows of 400 is 0.17 ms.
- **Training on the CPU**: the weight gradients (66 ms of the 28-thread step; pointwise ones at 31–83 GF/s for
  64→32 at 20² and 128→64 at 10²), and BatchNorm's backward (34 ms, memory-bound). Folding the statistics into
  the producing conv's epilogue would remove another read of every BatchNorm input. The stride-phase input
  gradient could be kept for convs with few input channels (down2: −0.2 ms a step).
- On a quiet machine the scheduling questions (fewer threads for small layers, P-cores first, the head beside
  the neck) may come out differently; under the load of this session none won.

# Second wave (lane `cpu2`, 2026-11-12)

Goal: folo-n CPU forward b1 14.7 → ≤ 12.5 ms, folo-s and folo-m −12%, by fewer flops and faster kernels (lane
`cpu` above showed the forward is bound by the CPU time it needs, not by scheduling).

## Method

- Three builds of the same bench sources: `main` at 647b981f (`D:/folo-wt/hill2-base`, label `m647`), the `main`
  this lane branched from (ed265317, private worktree `D:/folo-wt/hill2-cpu2-base`, label `base`) and the branch,
  copied to `D:/folo-build/hill2-cpu2-bin/<label>-<bench>.exe`.
- `ab.sh` and `ab2.ps1` there are lane `cpu`'s scripts (`tools/bench-ab.ps1` with per-run environment variables),
  holding the `cpu` lock; `summ.py` there prints each row's median and range per label from a log. Every A/B
  table below names its log in that folder. Rows are medians of the rounds' medians, ranges in brackets, 28
  threads, "busy" as above (cores used by other processes).
- Per-layer numbers: `conv_infer` (GF/s at 28 threads). **Not reproducible from the committed code**: the
  per-layer table, the single-core phase timings and the `tb` / two-pass sweeps below came from temporary knobs
  that are not in the code. The binaries with the knobs are kept: `D:/folo-build/hill2-cpu2-bin/w4-conv_infer.exe`
  and `w6-conv_infer.exe`, environment `FOLO_WINOGRAD=0|1` (direct / forced Winograd), `FOLO_WINO_2P=N` (two
  passes when blocks · N ≤ threads; w6), `FOLO_WINO_TB=n` (forced block size; w4, w6) and `FOLO_WINO_SMALL=x`
  (the `c · o` threshold; `w6-cpu_e2e.exe`). The per-layer runs were
  `bash D:/folo-build/hill2-cpu2-bin/ab.sh 2 0 "" "FOLO_BENCH_BUDGET_MS=300" "<label>=<exe>|<env>;..." k3s1` (and
  group `wide`, which w6 has with the extra folo-s / folo-m shapes); their raw output was not saved
  (`w6_groups.txt`, the per-group run, is). The single-core phase probe was a temporary timer, not kept.

## 3. Winograd F(2x2, 3x3) for 3x3 stride-1 inference convs (PR 3)

The inference conv (`Backend::conv2d_act` / `conv2d_act_cat`, which never record a gradient: fused models, and
eval-mode blocks under `no_grad`) runs a dense 3x3 stride-1 conv as Winograd F(2x2, 3x3)
(`crates/folo-core/src/cpu/conv/winograd.rs`): 16 multiplies per 2x2 output tile and channel pair instead of 36.

- **Weights** `U = G g Gᵀ` (`[16][O][C]`, in f64, rounded once) are made on the first call and kept with the
  weight tensor's storage (`StorageCell::derived`, reached through `View::derived`, keyed by type, purpose and
  layout). Every write lock of the storage (`copy_from`, `update_f32`, optimizer steps, `to_dtype_`, ...) clears
  them; they are made only under the read lock, so a value can never outlive the data it came from. A fused model
  transforms each weight once; an unfused eval model (BatchNorm folded per call) transforms per call (about 1–2%
  of the conv).
- **Blocks.** An image's tiles are cut into blocks of `tb` tiles (a multiple of 16, at most 128, small enough
  for `V` and `M` to stay in ~1 MiB of L2 and to give every thread a block). A block transforms its input patches
  into `V` (`16 x C x tb`; each input row deinterleaved once into (even, odd) pairs, row and column transforms
  vectorised across tiles), runs the 16 GEMMs `M[ξ] = U[ξ] · V[ξ]` on its own thread through the row-pointer
  AVX2 micro-kernel (`gemm_rows_serial`: A and B both read in place, one `KC` block at a time), and transforms `M`
  into its 2x2 output tiles, then bias and activation (the same `BiasAct::apply` as the direct path).
- **Small images** (fewer than half as many blocks as threads: a 20x20 output has 100 tiles, 7 blocks) run in two
  passes instead, so more tasks than blocks share the work without repeating any: every block's input transform,
  split by input channels, into one `V`; then the GEMMs and output transforms, split by output channels.
- **Which convs**: at least 16 input and 8 output channels, at most 4096 input plus output channels (a block's
  per-thread scratch, `16 · (c + o) · tb` floats kept between calls, stays within 4 MiB) and either 256 tiles (a
  32x32 output) or `c · o >= 4096`. The autograd conv (`conv2d_with`, the training forward) and the stride-1 input gradient (which
  runs the forward kernel) keep the direct conv, so training is unchanged.
- **Numerics.** Not bit for bit the direct conv. Error bound per output, before bias and activation (derived in
  the module doc of `winograd.rs`): `(C + 8) · 2⁻²³ · 9 · Σ_c D_c · G_c`, with `D_c` the sum of |inputs| over
  the output's tile's 4x4 input patch in channel c (padding 0) and `G_c` the sum of |weights| of its 3x3 kernel
  from channel c; the bias add adds `2⁻²³ · |bias|`. An earlier version of this PR claimed `2 · 2⁻²³ · Σ|terms|`
  with no factor of C; that is wrong (review): with all-constant inputs (1.0) and weights (0.1) over 512
  channels at 4x4 (no padding) every product has one sign, and the f64 sum 460.8000069 came out 460.7989197, an
  error of 1.09·10⁻³ = 19.8 · 2⁻²³ · Σ|terms| (the direct conv: 21.3). That is 0.43% of the direct conv's own
  bound `K · 2⁻²³ · Σ|terms|` (K = 9C = 4608) and 2.6·10⁻⁴ of the new bound. On the test cases (pseudo-random
  data, 1 to 300 channels, no epilogue) the largest error was 0.36–2.55 · 2⁻²³ · Σ|terms| (direct conv 0–1.81
  on the same cases; the larger Winograd ratios are on 1–3 channel cases with few terms), 1.4·10⁻⁴–0.14 of the
  direct conv's bound and at most 1.4·10⁻³ of the new one. Tests: `winograd_matches_reference` (f64 reference
  within the bound, plus the activation's error only when one is applied, every epilogue; odd and even output
  sizes, 1x1 / 1x2 / 2x2 images, no and 3-pixel padding, 1 to 300 channels, a 300-pixel-wide row, a batch, and
  the all-constant 512-channel case), `winograd_is_deterministic` (1, 3 and 28 threads, and each image of a
  batch alone: the same bits), `cpu/winograd_tests.rs` (counting the transforms and Winograd forwards on the
  test's thread: the inference conv takes Winograd and the autograd conv does not; one transform for repeated
  calls, a new one after `copy_from`, `update_f32` and `sub_assign`; two narrowed halves of one storage each
  transform once; a view whose `storage` field was pointed at another storage neither uses nor fills the first
  storage's transform).
- **COCO val2017, folo-n-coco, `folo val --device cpu`**: mAP50-95 0.30999154 → 0.30999145, mAP50 0.44944698 →
  0.44944635 (logs `D:/folo-data/runs-hill2/cpu2/val_*.json`): changes below 10⁻⁶.

Single core (`RAYON_NUM_THREADS=1`, phases timed by a temporary probe that was not kept), 64→64 at 80x80: input transform 0.74 ms
with a scalar gather → 0.42 ms deinterleaved, GEMMs 1.6–1.8 ms (116–129 GF/s of GEMM flops), output transform
0.18–0.23, about 2.5 ms against the direct conv's ~5.4 ms on a loaded core. 128→64 at 40x40: 1.17 ms (direct
~2.6).

Per layer, 28 threads, GF/s of direct-conv flops (`w6-conv_infer.exe` with the knobs of "Method", 2 rounds
each, 13 cores busy elsewhere, raw output not saved; the folo-n rows are `k3s1`, the wider ones are folo-s /
folo-m shapes of its group `wide`):

| conv | direct | Winograd (one pass) | Winograd, two passes when blocks · 2 ≤ threads |
|---|---|---|---|
| 16→8 at 160² | 326–334 | 343–371 | (one pass) |
| 16→32 at 80² | 419–426 | 495–498 | (one pass) |
| 32→16 at 80² | 404–411 | 525–530 | (one pass) |
| 64→64 at 80² | 772–773 | 1083–1099 | (one pass) |
| 32→32 at 40² | 312–319 | 413–417 | (one pass) |
| 128→64 at 40² | 675–684 | 956–980 | (one pass) |
| 64→64 at 20² | 324–326 | 201–213 | 336–341 |
| 256→64 at 20² | 542–554 | 319–321 | 638–645 |
| 128→128 at 20² | 522–534 | 330–331 | 681–684 |
| 512→64 at 20² | 618–635 | 347–359 | 741–770 |
| 128→128 at 24² | 651–666 | 463–464 | 820–836 |
| 64→64 at 24² | 374–382 | 269–310 | 436–439 |
| 64→64 at 12² | 158–161 | 77–78 | 170–187 |

Forward A/B of the merged rule (logs `pr1_n28.txt`, `pr1_s28.txt`, `pr1_m28.txt`, `pr1_b16.txt`,
`pr1_train5.txt`; 3 rounds alternated, `FOLO_BENCH_ITERS` 60 / 30 / 12 / 8 for n / s / m / batch 16; 2–6 cores
busy elsewhere, the quietest session of the climb):

| 28 threads | `main` 647b981f | `main` ed265317 | PR 3 | vs ed265317 |
|---|---|---|---|---|
| folo-n forward b1 | 15.81 ms (15.51–16.16) | 13.51 (13.36–13.66) | **12.33** (12.25–12.70) | −8.7% |
| folo-n e2e 640×640 | 16.76 (16.35–16.88) | 14.13 (13.81–14.34) | **13.17** (13.05–13.67) | −6.8% |
| folo-n e2e 1280×720 | 12.90 (12.18–12.91) | 10.78 (10.32–10.91) | **10.00** (9.83–10.80) | −7.2% |
| folo-s forward b1 | 34.30 (33.32–36.20) | 29.05 (27.57–29.52) | **27.34** (27.21–28.04) | −5.9% |
| folo-m forward b1 | 79.41 (76.78–79.72) | 76.86 (73.93–77.37) | **68.39** (67.56–68.64) | −11.0% |
| folo-n `predict_batch` 16 | 149.7 (145.1–158.8) | 150.7 (147.5–153.1) | 145.8 (145.5–153.5) | tie (−3%) |
| folo-n train step 320 px b16 (5 rounds, 2–9 busy) | | 245.4 (212.9–248.5) | 247.5 (217.8–248.2) | tie (+0.9%) |

Earlier the same day, under more load, the same comparison measured folo-n 15.17 → 14.66 ms (−3.4%; 13.7–15.3
cores busy elsewhere), folo-s 33.5 → 31.9 (−4.8%; 10.0–12.8 busy) and folo-m 77.9 → 72.9 (−6.5%; 5.5–10.0 busy)
(logs `w6_n28.txt`, `w6_s28.txt`, `w6_m28.txt`, label `base` against `s4k`, busy ranges over all labels of each
log; that binary had the rule behind environment knobs): under load less of the per-layer gain reaches the
forward.

After the review fixes (per-row writes into `V`, the cache checked against the view's storage, at most 4096
channels), the fixed head (`pr1b`) against the first one (`pr1`), 3 rounds, 14–16 cores busy elsewhere (logs
`pr1b_n28.txt`, `pr1b_m28.txt`): folo-n forward 15.10 (`base`) / 14.06 / **14.09** ms, folo-m 86.57 / 81.34 /
**79.90** ms: the fixes cost nothing.

### Negative results (PR 3)

(Measured with the knob binaries of "Method": `FOLO_WINO_2P`, `FOLO_WINO_TB`, `FOLO_WINOGRAD`; raw output not
saved.)

- **Two passes for every image with fewer blocks than threads** (40x40 outputs: 25 blocks of 16 tiles): 32→32 at
  40² 413 → 298–304 GF/s, 64→64 at 40² 784 → 670–686, worse than one pass and, for 32→32, than the direct conv;
  hence two passes only from `blocks · 2 ≤ threads`.
- **Smaller or larger blocks** (`tb` forced to 16 / 32 / 48 for every layer at 28 threads): no layer faster than
  the per-layer choice; 64→64 at 80² 1091–1147 GF/s at 16 tiles and 931 at 48, against 1090–1100 by the rule (that
  run got ~14.7 cores: the layer is bound by the CPU it gets, not by load balance).
- **8 input channels** (8→16 at 160²): 324–348 GF/s against 347–361 direct; kept direct.
- **The weight transform per call** (before the cache): 0.08 ms for 64→64 and 0.18 ms for 128→64 on one thread,
  serial: a quarter of a 28-thread layer.

## Leftovers (second wave, by expected gain)

- **Pointwise convs** are the largest conv group of the folo-n forward at 28 threads (4.9 ms of 12.4 ms of convs
  timed alone, `conv_infer`, log `w6_groups.txt`): those at 40² and 20² run at 50–300 GF/s, waiting on the pool
  more than computing.
- **Winograd F(4x4, 3x3)**: 2.25 multiplies per output against F(2x2, 3x3)'s 4 (1.8x fewer GEMM flops, less
  transform data per output); larger rounding error (transform constants up to 8), to be checked against AP.
- Stride-2 convs (2.8 ms of convs timed alone) and the depthwise convs and shifted-rows copy, as above.
