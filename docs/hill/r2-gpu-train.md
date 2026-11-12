# Hill-climb 2: the GPU training step (lane `gpu-train`, branches `train/*`)

Round 2 of docs/hill/gpu-train.md. Goal: a faster folo-n / folo-s training step (batch 16, 640 px, forward + loss
+ backward + optimizer) on the RTX 4080 SUPER, without changing what is computed beyond summation order.
Baseline: `main` 647b981f.

## Method

- **Profile** (`FOLO_GPU_PROFILE=1`, docs/GPU.md "Profiling"): GPU time per kernel label and phase. Timestamps
  serialise kernels, so profiled wall times are not the real ones; per-label times swing by about ±15 % between
  runs on the shared GPU, so a kernel change is judged on the group totals of several runs, and claimed only
  through the end-to-end A/B. Script: `D:/folo-build/hill2-gpu-train/bin/prof.sh <tag> <bins...>`
  (`FOLO_GPU_PROFILE=1 FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=train FOLO_BENCH_SCALES=n`, the f32 and the AMP
  step of each binary in turn), output `D:/folo-data/runs-hill2/gpu-train/prof-<tag>-<bin>.txt`.
- **Binaries** (`D:/folo-build/hill2-gpu-train/bin/<name>.exe`, the `folo-vision` `gpu` bench): `base` = main
  647b981f; `c1` = the whole warp along `k`; `c2` = runs of 8; `c3` = runs of 16 / 8 plus the stem tile (with a
  `FOLO_WG_RUN` override of the run length, since removed, which gave the runs-of-4 and runs-of-8 rows below);
  `wg` = this branch at 5af98913 (the PR's first commit). Profiles: `prof-w1-*` (`base`, `c1`), `prof-w2-*`
  (`base`, `c2`), `prof-w3-*` (`base`, `c3`, and `r4` / `r8` = `c3` with runs of 4 / 8).
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
`run` neighbouring pixels of one column, then the next columns (`conv::wgrad_b_units`; run `min(bk, 16)` in the
regular generator, so 8 on the `LARGE` tile, whose depth is 8, and 8 on tensor cores). The pixel `(n, oh, ow)` of
each run is decoded once per load, the tap per element. The k-major workgroup tile is padded so the stores of a run
hit distinct banks (in the regular generator only when the padded tile still fits the device's workgroup memory,
`WG_PAD`, which every weight-gradient tile does in the 4080's 48 KiB; the tensor-core tile always has its padding).
The stem shape (per group `C_out/groups ≤ 16` and `C_in/groups·KH·KW ≤ 32`: folo's 3→16 stem, `N = 27`) takes a
16×32 regular tile with 32 invocations, also under AMP (the narrowest tensor-core tile is 16×256, 89 % padding),
split-K into up to 1024 chunks.

**Tried on the way** (profiled, alternated with main):

| variant | f32 weight grads (GPU ms) | tensor-core weight grads | note |
|---|---|---|---|
| main | 11.8-12.0 (+0.9 reduce) | 7.0-9.0 (+0.8) | |
| whole warp along `k` (32 / `bk` lanes per column) | 11.5 | 11.0 | tensor cores **3x slower** on 3×3 layers (4-way bank conflicts on the f16 tile stores) |
| runs of 8, padded tile | 10.9 | 6.5 | |
| runs of 16 (f32) / 8 (f16) + stem tile (shipped) | 10.8 (+1.2 reduce) | 5.8 + 0.4 stem | stem f32 0.83 → 0.35 ms, AMP 0.70 → 0.41 + 0.13 reduce (`prof-w3-c3`) |
| runs of 4 / 8 everywhere | 10.6-11.2 | 5.2-5.4 | inside the noise of the shipped variant; AMP stem 0.35 / 0.31 + 0.13 reduce (`prof-w3-r4` / `-r8`) |

**A/B** (end to end, folo-n b16 640, base = main, alternated; ms per step, median of the run's 10 steps and,
in brackets, its fastest step). The host was 33-100 % busy with other lanes' builds and jobs during both holds
and one run saw 84 % foreign GPU load, so the medians swing by up to 2x; the fastest-step column is the steadier
read.

| hold, host load | | f32 base | f32 new | AMP base | AMP new |
|---|---|---|---|---|---|
| 1 (33-71 %) | r1 | 48.4 (47.8) | 54.0 (49.8) | 48.3 (33.7) | 36.2 (32.5) |
| | r2 | 55.8 (49.5) | 53.0 (48.6) | 59.5 (41.3) | 34.3 (28.3) |
| | r3 | 51.6 (48.4) | 48.8 (47.6) | 37.3 (32.7) | 31.8 (28.4) |
| 2 (68-100 %) | r1 | 59.3 (49.3) | 46.0 (45.9) | 31.9 (29.4) | 29.7 (28.5) |
| | r2 | 81.7 (50.0) | 58.8 (49.4) | 31.6 (28.8) | 40.9 (27.8) |
| | r3 | 74.4 (48.6) | 56.9 (52.5) | 31.3 (30.6) | 90.6 (49.5) |

Hold 3, each bench process at High priority (`ab.sh` sets it), host 2-57 %, GPU 0 % before nearly every run
(`ab-wg3/`), img/s at the median of each run:

| round | f32 base | f32 new | AMP base | AMP new |
|---|---|---|---|---|
| 1 | 331 | 342 | 547 | 584 |
| 2 | 334 | 353 | 492 | 596 |
| 3 | 332 | 348 | 573 | 562 |
| 4 | 337 | 349 | 538 | 560 |
| **median (range)** | **333 (331-337)** | **348.5 (342-353), +4.6 %** | **542.5 (492-573)** | **573 (560-596), +5.6 %** |

f32 wins every round; AMP three of four. Peak memory unchanged (3467 / 2022 MiB, the bench's own line). This is
the PR's claim; holds 1-2 only say the direction under heavy contention.

folo-s (`FOLO_BENCH_SCALES=s ab.sh wgs 3 base wg`, High priority, host 82-100 %, merged as #184): f32 147 / 145 /
146 (main) vs 146 / 147 / 147, a tie; AMP 267 / 270 / 268 vs 251 / 275 / 278: median 268 → 275 (+2.6 %, inside the
±3 % tie band; fastest steps 58.7-59.2 → 57.1-57.6 ms). folo-s's weight gradients have wider channels, where the
old gather cost relatively less. Peak memory unchanged (6520 / 3713 MiB).

## PR 2: MuSGD's Muon step in fused kernels (`train/musgd-fused`)

**Finding.** On main the Muon half of MuSGD is tensor ops: per column bucket a pad and stack, the norm (four ops),
five Newton–Schulz steps of nine ops each (a transpose copy, three matmuls, five elementwise), then per weight a
narrow / transpose / contiguous / scale, plus the Nesterov mix as one `mul_scalar` per weight. Profiled (folo-n,
f32): 7.7 GPU-ms in 1225 dispatches per step (255 matmuls, 580 elementwise, 224 copies), 6.4 ms of host encoding,
step phase 6.9 ms (SGD: 1.2 ms). Under AMP the step phase is 12.1 ms against SGD's 2.4.

**Change.** `Tensor::muon_step_many_` (a new optional `Backend::muon_step_many` hook; the CPU keeps the tensor
ops) and `folo_gpu::kernels::muon`: a multi-tensor gather (Nesterov mix, each matrix oriented rows ≤ cols into one
workspace, a partial sum of squares per 256 elements), the Frobenius norms from the partials and the division over
all elements, per Newton–Schulz step three grouped GEMMs over all matrices at once (each workgroup binary-searches
its matrix and 64×64 tile in a table; `A = X·Xᵀ` is symmetric bit for bit, so `A·A` reads `A` as its own transpose;
no zero padding), and a multi-tensor `p = p · 1 + (X · scale) · (−lr)`. About `2 · ⌈n / 16⌉ + 17` dispatches for
`n` matrices. Same arithmetic per element as the tensor ops except the summation order of the products and of the
norm. The workspaces of all matrices must fit one storage binding, so the matrices run in consecutive groups that
each fit the device's largest binding (one group on this card); a device with fewer than 5 storage bindings per
kernel, or a matrix that alone is past the limit, declines the call before anything is written
(`NotSupported`), and MuSGD falls back to the tensor ops.

**Steps** (profiled, folo-n f32, optimizer phase GPU time per step; `FOLO_BENCH_OPTIM=musgd bash
D:/folo-build/hill2-gpu-train/bin/prof.sh <tag> <bin>`: main's `prof-base-musgd.txt`, `prof-m1-mu.txt`,
`prof-m2-mu2.txt`). The binaries: `mu` is the first fused version (commit `821c5dd1` after the rebase, `214e562f`
before it), `mu2` the version shipped before review (`eb09c3db`, was `0893ff56`).

| version | GPU ms | dispatches | largest items |
|---|---|---|---|
| main | 7.67 | 1225 | matmuls 2.95, elementwise 2.68, copies 1.19 |
| fused v1 (`mu`: one-workgroup norm, scalar GEMM tiles, result as views + `axpby_many`) | 3.13 | 139 | GEMMs 2.14, 86 view copies 0.35, norm 0.33 |
| fused (`mu2`, shipped: unrolled vec4 GEMM tiles, parallel norm, update fused into the last kernel) | **1.47** | **53** | GEMMs 1.14, SGD 0.17 |

The view copies of v1: the GPU backend densifies gradient views at a nonzero offset, so `p −= lr · O` over views
of one result copied every matrix; the shipped version writes the parameters from the workspace directly.

**A/B, main vs this branch, both MuSGD** (`FOLO_BENCH_OPTIM=musgd ab.sh mu3 3 base mu2`, High priority, host
26-99 %), img/s at the median of each run:

| round | f32 main | f32 fused | AMP main | AMP fused |
|---|---|---|---|---|
| 1 | 303 | 321 | 355 | 488 |
| 2 | 315 | 338 | 367 | 487 |
| 3 | 306 | 324 | 359 | 499 |
| **median** | **306** | **324, +6 %** | **359** | **488, +36 %** |

**MuSGD's overhead over SGD** (the step's img/s with SGD over with MuSGD, minus 1, same binary, same hold):

| | f32 | AMP | source |
|---|---|---|---|
| main | 333 / 318: **+4.7 %** | 542.5 / 453: **+20 %** | hold 3 of PR 1 (`ab-wg3`, `ab-mu2`, base runs, host 2-65 %) |
| fused | 321 / 336: none (MuSGD measured 4 % faster) | 518 / 495: **+4.6 %** | `ovh.sh mu2 3 mu2` (`ovh-mu2/`, alternated SGD / MuSGD, host 73-87 %) |

The fused row comes from one hold under load (host 73-87 %), where the AMP overhead of the single rounds ranged from
about −4 % to +5 %: MuSGD within 6 % of SGD in both precisions on the 640 px bench is what was observed there, not a
bound. The "17 %" of docs/hill/learn.md was measured
on the 320 px proxy, where the step is a quarter of the work and the Muon half (whose cost does not scale with the
image size) weighs four times as much; the fused step's ~1.5 GPU-ms against main's 7.7 should take most of that
back too (not measured at 320 px). Peak memory: 3478 / 2033 MiB, as main's MuSGD.

**Correctness.** `crates/folo-nn/tests/gpu/muon.rs`: `fused_muon_matches_tensor_ops` (the fused update against the
Nesterov mix, `orthogonalize_many` and `p − lr·O` as tensor ops on the CPU, with `lr = 1` so the parameter carries
`O` at full size; wide, tall (transposed), multi-tile, single-row / single-column and square matrices, scales
other than 1, 20 matrices so the multi-tensor kernels take two dispatches; Nesterov and plain; momenta and
gradients unchanged; a broken transpose fails it), `fused_muon_edge_cases` (a zero momentum and gradient leave `p`
as it was, one near zero has a norm about the eps, a 160×1152 matrix makes the norm loop run three times per
invocation, a NaN gradient turns that parameter NaN exactly where the tensor ops do and nothing else, an empty
matrix; dropping the eps or the norm loop fails it), and `crates/folo-nn/tests/gpu/main.rs`'s
`training_steps_on_gpu_match_cpu` (MuSGD: two steps on the CPU, then three on the GPU, against three more on the
CPU). folo-gpu's `tests/muon.rs`: split into three groups by a small limit with 5 bindings, the step equals the
one-group step bit for bit; 4 bindings or a matrix past the limit decline it and leave the parameters as they
were. `partition_fits_the_limit` (CPU) checks the groups against the limit. `tests/optim.rs` (two steps against an f64 re-implementation of the reference) is the CPU path,
unchanged.

## Not shipped: AMP step overhead (branch `train/amp-step`, 0b74e5f7 after rebasing on #191, no PR)

Item 3 of the brief plus the accumulator's copies. Two changes:
- **Cached f16 weights.** Autocast's f16 copy of each conv weight / bias is cached on its storage with the storage
  version it was cast at (`Tensor::autocast_f16`). The optimizers recast the copies of the parameters they updated
  right after their step, in multi-tensor casts (`Tensor::refresh_autocast_f16`, `Backend::cast_f16_many`,
  `kernels::cast_many`; same rounding as `cast_any`).
- **Accumulator first pass.** A window's first pass overwrites the reused buffers with `OptimStep::Axpby { a: 0 }`
  (the stale buffer is not read) in 16-tensor dispatches, instead of 255 copies.

Both are tested: the cached copies equal fresh casts bit for bit after optimizer steps and after a direct write
(a check that ignores the version fails it), and stale infinities / NaNs do not survive the overwrite.

Profiled (folo-n AMP, `prof-a2-amp2.txt`):
- forward: 526 → 433 dispatches (98 casts → 5);
- optimizer phase: 1.15 → 0.34 GPU-ms, 290 → 57 dispatches.

End to end it is a **tie**:
- `ab-amp2`, 4 rounds, host 54-100 %: AMP median 507.5 (main) vs 495.5 img/s, with one disturbed run.
- `ab-amp2-clip`, 3 rounds: 526 vs 507.
- `ab-amp3`, AMP only, 6 rounds, host 21-85 %: 522 vs 529 (+1.3 %).

Rebased on main 912da99d and measured again against it (`prof.sh p1|p2 main2 amp3`, `ab.sh amp4 4 main2 amp3`,
host 15-97 %):
- Profiled pair p2: forward 8.83 → 8.42 GPU-ms, casts 0.83 → 0.41 ms; optimizer phase 1.15 → 0.32 GPU-ms,
  290 → 57 dispatches.
- End to end, AMP img/s: main 567 / 548 / 587 / 499 (median 557.5) vs 535 / 572 / 545 / 521 (median 540).

Still a tie. About 1 GPU-ms of a 27-29 ms step is inside this machine's noise. The branch is left for a
re-measure in a quiet window: it adds a cache to folo-core's storage cells, which is not worth it without a
measured win.

## Final: current main against 647b981f

Run after every lane's merges so far: main 56a2ba38, which includes this lane's #184, #191 and #195 and other
lanes' #181-#196. The rows therefore show the whole round, not this lane alone; this lane's share is the PR 1 / PR 2
A/Bs above.

Method: folo-n / folo-s, batch 16, 640 px, `ab.sh fin-n|fin-clip|fin-mu 3 base main3` and `ab.sh fin-s1|s2|s3 1
base main3`, the bench at High priority. `main3` is the bench built at 56a2ba38. Raw outputs are under
`D:/folo-data/runs-hill2/gpu-train/ab-fin-*/` (lines of an earlier, interrupted attempt with `main2` are in the
same summaries). Host load was 31-97 %. Figures are img/s, median of 3 rounds (range).

| row | 647b981f | main 56a2ba38 | change |
|---|---|---|---|
| folo-n f32 | 318 (316-336) | 333 (326-351) | +4.7 % |
| folo-n AMP | 513 (512-520) | 596 (545-599) | +16 % |
| folo-n f32, clip 10 | 334 (327-343) | 350 (340-361) | +4.8 % |
| folo-n AMP, clip 10 | 520 (502-525) | 552 (395-584) | +6 % |
| folo-n f32, MuSGD | 299 (182-308) | 347 (330-354) | +16 % |
| folo-n AMP, MuSGD | 411 (372-421) | 510 (343-525) | +24 % |
| folo-s f32 | 148 (108-151) | 151 (144-154) | +2 % (tie) |
| folo-s AMP | 274 (270-278) | 279 (230-291) | +2 % (tie) |
| peak memory, folo-n f32 / AMP (MuSGD) | 3467 / 2022 (3478 / 2033) MiB | the same | |
| peak memory, folo-s f32 / AMP | 6520 / 3713 MiB | the same | |

Disturbed runs, kept in the ranges:
- base round 3 of f32 MuSGD (64 % foreign GPU load);
- main3 round 3 of AMP MuSGD (64 %);
- base round 1 of folo-s f32 (15 %);
- main3 round 2 of folo-s AMP (230).

**MuSGD's overhead over SGD.** Measured on the same binary, alternated SGD / MuSGD (`ovh.sh main3 3 main3`,
host 0-94 %):

| | SGD | MuSGD | overhead |
|---|---|---|---|
| f32 | 337 / 338 / 338 | 359 / 358 / 354 | none (MuSGD measures about 6 % faster, as in `ovh-mu2`) |
| AMP | 567 / 444 / 549 (median 549) | 413 / 513 / 521 (median 513) | +7 % on medians; fastest steps 27.3-27.9 vs 28.1-29.2 ms, +5 % |

On 647b981f (PR 1's hold 3) the overhead was +4.7 % (f32) and +20 % (AMP). The faster f32 MuSGD step is
consistent across holds but not explained; it is the same bench with another optimizer, so the loss and assigner
see other weights.

**Against the brief's targets** (f32 ≥ 390, AMP ≥ 620 img/s, MuSGD overhead ≤ 6 %): AMP reached 596, f32 333 and
350 with clipping. MuSGD's overhead is gone in f32 and +4-7 % under AMP. folo-s moved by about 2 %.

## Negative results

- A warp-wide run along `k` (PR 1 above): coalesced loads, but the tensor-core tile stores conflict 4-way.
- Run lengths 4 / 8 / 16 for the weight-gradient B tile: within the profiler's noise of each other.
- The AMP step changes above: real GPU and dispatch savings, no measurable end-to-end gain (a tie).
- The first fused Muon version: one workgroup per matrix for the norm, and the result returned as views. Views
  at a nonzero offset are densified (copied) by the GPU backend's multi-tensor update: 86 copies, 0.35 ms.

## Leftovers, by expected gain

1. **f32 weight gradients**: still 10.6-10.8 GPU-ms plus 1.0-1.2 ms of split-K reductions per folo-n step. The
   forward convs do the same FLOPs in about 7 ms.
   - The regular tiles (4×4 / 4×8 per invocation) and the column reduction of the partial slabs (`reduce_cols`
     over up to 1024 × N partials) are the next places.
   - f16x2 splitting on tensor cores was not tried: gradient magnitudes below f16's range need per-tensor scaling.
   - The f32 targets of the brief (390 img/s f32) were not reached: PR 1 gave +4.6 % (333 → 348.5).
2. **Concat in the training forward** (item 5): `copy_into` 1.4 / 0.6 GPU-ms (f32 / AMP) in the forward plus the
   backward's strided copies. #185 (gpu-dispatch) made convs read concatenations in place in inference; the
   training-mode `ConvBnAct::forward_cat` with BatchNorm still copies (`Tensor::cat`, then the conv).
   - Not started this round: the expected 2-3 % of the step would not measure clearly on this shared host.
   - A design: a recorded conv over the parts whose forward is #185's `gemm_cat` (B rows from up to four part
     buffers). Its input gradient is written once and split as views, which every consumer reads in place. Its
     weight gradient gathers its B tile from the parts too (a `CAT` variant of `CONV_MODE` 3 in both generators,
     choosing the part buffer by the `ci` range of each tile column).
3. **The AMP step branch** above (about 1 GPU-ms and 330 dispatches per step): re-measure when the host is quiet.
4. **Muon GEMMs** (1.14 GPU-ms per step): the gram stage's tiles on the largest matrices (256 × 2304 in folo-n)
   run their whole K serially. Split-K or narrower tiles would help, or tensor cores (the reference computes in
   bf16).
5. **MuSGD at 320 px** (the proxy's 17 %): not measured after the change.
6. **Test infrastructure**:
   - `gpu!()` (folo-nn) and `coop_ctx` (folo-gpu) skip silently when no adapter / no tensor cores; CI should
     require them on the GPU runner.
   - Timing-sensitive tests fail under load: folo-cli `format_picks_the_labels_and_resume_keeps_it` (its `stop`
     races a 6-epoch CPU run) and folo-server `cloud::ledger` (a 5 s tolerance).
   - folo-vision `tiny_amp_run_learns` failed on main after #186: its bound sat inside its own seed spread
     (fixed by gpu-gemm in #193). `amp_trains_the_dual_head` had the same problem: seeds 0-5 gave 0.93-1.25
     against a bound of 1.15. #195 (this lane) moved it to 1.3, added a no-skipped-batch check and a direct
     autocast check of the one-to-one branch.
