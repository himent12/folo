# Hill-climb 2: GPU dispatch cost (lane `gpu-dispatch`)

Goal: cheaper and fewer dispatches in the batch-1 forward, which is host-bound (folo-n f16 tensor cores: about
0.8 ms of GPU work in a 1.65 ms call, 117 dispatches at 8–19 µs of host time each; docs/hill/gpu-infer.md).
Baseline `main` 647b981f, RTX 4080 SUPER, 640×640.

## Method

- **Benches**: `cargo bench -p folo-vision --bench gpu` (rows: forward b1 / b16, e2e predict b1 / b16, train step)
  and `--bench gpu_profile` (host split of a synchronised b1 forward: encode / submit / wait, then every dispatch
  replayed alone). Each binary is built from its own tree (baseline: `D:/folo-wt/hill2-base`, target
  `D:/folo-build/hill2-base`) and copied aside; `ab.sh` alternates the binaries run by run (order reversed every
  other round), waits for `nvidia-smi` under 10% before each run and records the GPU and CPU load; `parse.py` /
  `parse_prof.py` give the median and min–max over rounds of each binary's median. GPU lock
  (`D:/folo-build/hill/lock.sh`) around every measured command. Raw runs: `D:/folo-data/runs-hill2/gpu-dispatch/`.
- **Noise**: the machine was shared with five other lanes (builds, CPU at 60–97%) and four `folo serve` instances
  outside the lock that put 17–19% bursts on the GPU between runs. Wall-clock rows of the same binary swung up to 2×
  between rounds (folo-n f32 forward b16, which is GPU-bound and untouched by PR 1, read 10.4 and 20.1 ms in the
  first two runs), so claims rest on rows that stayed within a few percent across rounds, on the host-side encode
  time of `gpu_profile` and on the bind-group counters.

## PR 1: bind-group cache (`gpu/bind-group-cache`)

**What.** Every dispatch created a bind group (3–4 µs of host time each, round 1's timers). Now the uniform
parameter block is bound with a dynamic offset (its slot in the 64 KiB arena chunk), so a bind group depends only
on `(kernel layout, arena chunk, storage buffers)`, and `context/binds.rs` caches them under that key. An
inference forward meets the same combinations every call and creates no bind group after its first calls; a
training step reuses only some (see Training below). The `gpu` bench now prints new bind groups per call / per
step.

Admission: a miss creates the bind group as before but caches it only if its key missed before (a bounded set of
recent keys, cleared past 8192), so a combination seen once costs one hash and a set insert besides the bind group,
and a forward hits from its third call on. Keys and the cache's maps use a small FxHash-style hasher instead of
SipHash.

Invalidation and bounds:
- A cached bind group keeps its buffers alive, so only buffers the pool owns (created by it and not yet let go of)
  and the context's dummy buffer are cached. A raw buffer passed to a public kernel call, or a trace replaying
  buffers the pool already let go of, is never cached: nothing would report its release.
- The pool records the id of every buffer it lets go of (evicted from a free list, dropped by `trim_pool`, or too
  large to cache) and sets a flag; the next dispatch or submission (so `flush` and `sync` too), or `trim_pool`,
  drops every cached bind group that binds one of them, through an exact index from buffer id to keys: a key leaves
  its buffers' sets whenever its entry leaves the cache (retired, dropped at a rollover, replaced after a
  collision, cleared), so the index holds no more than the entries do. Past 4096 pending ids the pool asks for the
  whole cache to be dropped instead. `release_memory` empties the cache with the uniform arena (whose chunks the
  bind groups hold).
- Two generations of at most 4096 bind groups: when the current one fills it becomes the old one and the previous
  old one is dropped; a hit in the old generation moves the entry back.
- Hits compare the stored handles (layout, uniform chunk, every buffer), so a hash collision is a miss, never a
  wrong bind group; the stored clones also keep a buffer's identity from being reused while an entry holds it.

Touches `pipeline.rs` (lane `trainer`): one line, `has_dynamic_offset: true` on the uniform binding.

Tests (`src/tests/binds.rs`): 300 dispatches on one buffer create two bind groups (first sighting, then cached)
and each reads its own parameters (sum exact), a second buffer gets its own, a 1024-dispatch batch spans four
uniform chunks (two bind groups per chunk; every dispatch adds its own value, so a wrong chunk breaks the exact
sum); `trim_pool` and `release_memory` drop the entries of the buffers they free and later dispatches still compute
right; `sync()` alone drops the entries of an evicted buffer; a buffer made on the device directly is never cached,
the dummy buffer is; the pool reports exactly the buffers it evicts or clears, switches to `All` past its bound and
owns exactly what it holds or lends; the cache never holds more than two generations, keeps a hot entry across
rollovers, and retiring a buffer drops its entries in both generations and nothing else; the index stays as small
as the cache when a persistent buffer is paired with many retired short-lived ones.

### Numbers

**In-process toggle** (`FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SCALES=n,s cargo bench -p folo-vision --bench
dispatch_ab`: one binary, cache on and off interleaved, 12 rounds, batch 1, CPU 30–90% busy with other lanes;
median over rounds of each round's median, range in brackets):

| folo-? b1 | metric | cache on | cache off | |
|---|---|---|---|---|
| n f32 | encode | 1.82 (0.95–3.14) | 2.06 (1.23–2.99) | −12% |
| n f32 | synchronised call | 2.70 (2.15–4.34) | 2.95 (2.24–4.06) | −8% |
| n f16 TC | encode | 2.19 (1.07–3.12) | 2.76 (2.11–3.55) | −21% |
| n f16 TC | synchronised call | 2.68 (1.65–3.68) | 3.29 (2.59–4.03) | −19% |
| s f32 | encode | 2.53 (2.25–3.28) | 2.80 (1.61–3.27) | −10% |
| s f32 | synchronised call | 4.48 (4.08–6.55) | 4.51 (4.05–4.80) | −1% |
| s f16 TC | encode | 1.47 (1.12–2.28) | 2.30 (1.28–3.17) | −36% |
| s f16 TC | synchronised call | 2.59 (2.23–3.13) | 3.17 (2.28–4.01) | −18% |

That is 2–7 µs less host time per dispatch (117 per forward), about what round 1 timed for bind group creation.
Pipelined b1 rows move less (−8%, −16%, +2%, +2%): with the host faster, folo-s is bound by the GPU.

**Against `main`** (`gpu` bench, forward section, 3 rounds alternated, GPU 0–3% before each run, CPU 42–77%):

| row | main | PR 1 | |
|---|---|---|---|
| folo-n f32 forward b1 | 2.30 ms (2.21–2.96) | 2.25 (2.16–2.27) | −2% (tie) |
| folo-n f32 forward b16 | 10.27 ms (10.15–10.43) | 10.09 (9.92–10.12) | −2% (tie) |
| folo-n f16 TC forward b1 | 2.54 ms (2.35–3.39) | 2.18 (2.01–2.27) | −14% |
| folo-n f16 TC forward b16 | 5.99 ms (5.76–6.23) | 5.70 (5.69–5.77) | −5% |
| folo-s f32 forward b1 | 4.62 ms (4.38–5.35) | 4.19 (4.09–4.28) | −9% |
| folo-s f32 forward b16 | 25.17 ms (24.98–25.21) | 24.52 (24.48–32.38) | −3% (tie) |
| folo-s f16 TC forward b1 | 3.80 ms (3.44–3.88) | 3.42 (3.04–3.80) | −10% |
| folo-s f16 TC forward b16 | 13.82 ms (13.75–14.45) | 13.56 (13.44–14.02) | −2% (tie) |

An earlier 3-round run of forward + predict in a busier window (CPU 60–97%, GPU bursts from `folo serve`
processes outside the lock) swung up to 2× on GPU-bound rows of either binary and is not used
(`pr1-infer/`). The `gpu_profile` replay sums read 10–40% *higher* on PR 1: that bench times only the wait after
encoding 200 replays of a dispatch, so a faster host leaves more of the GPU work to wait for (an artifact; the
GPU-bound b16 rows above do not move).

**Training.** The first A/B (3 rounds against `main`, folo-n b16) showed the step faster overall (f32 54.6 →
49.8 ms, AMP 32.4 → 30.8) but the optimizer phase 4× slower (f32 1.8 → 7.0 ms of synchronised optimizer phase
time, its host encode 1.6 → 6.7 ms). The pool evicts buffers throughout a step, and each eviction made the next
dispatch sweep the whole cache (`retain` over thousands of entries). A first fix kept an index from buffer id to
the entries that bind it, so retiring a buffer cost only its own entries, but the optimizer phase's host encode
still went 1.0 → 3.8 ms (f32): only 27% (f32) / 56% (AMP) of a training step's dispatches reuse a bind group (805
of 1109 / 556 of 1275 create one), because the pool's buffers do not come back in the same order every step, and
every miss paid for an entry (buffer clones, index pushes) while generation rollovers rebuilt the index and dropped
4096 bind groups at a time. The fix is implemented: second-sight admission (a miss caches nothing unless its key
missed before), the FxHash-style hasher, an exact index that shrinks with the cache, pool-owned buffers only, and
the sweep on submission.

After the fix (`pr1-train4/`: 4 rounds, `main`, cache on and cache off in one binary via `FOLO_GPU_NO_BIND_CACHE=1`
alternated, folo-n b16, CPU 56–100%) training is a tie:

| | main | cache on | cache off |
|---|---|---|---|
| f32 step | 49.81 ms (49.33–50.19) | 49.48 (47.87–79.46) | 50.48 (48.21–51.34) |
| f32 optimizer phase, host encode | 1.30 (1.00–1.70) | 1.50 (1.00–2.10) | 1.50 (1.00–2.20) |
| AMP step | 30.10 ms (29.71–30.66) | 30.10 (28.34–35.52) | 29.84 (29.46–31.68) |
| AMP optimizer phase, host encode | 3.10 (2.00–3.20) | 2.35 (1.90–3.40) | 2.50 (2.00–3.50) |

An earlier 3-round run (`pr1-train3/`, CPU 82–100%) had AMP steps of 46–77 ms with the cache on, but `main` and
cache-off also had 52–69 ms rounds there, an instrumented run (temporary timers around rollovers, sweeps and slow
lookups) found no cache work above 0.2 ms, and the clean run above does not reproduce it: machine noise. A training
step still creates about 850 (f32) / 760 (AMP) bind groups, and the f32 step allocates 55 new pool buffers per step
(AMP none): see Leftovers.

The in-process toggle after the fix (`pr1-dispatch-ab2.txt`, 12 rounds, CPU busier than the first run): folo-n f32
encode −13%, pipelined −12%; folo-n f16 TC encode −19%, synchronised call −18%, pipelined −20%; folo-s f32 encode
+15%, calls ±1% (a tie within the ranges); folo-s f16 TC encode −12%, synchronised call −16%, pipelined +12%.

## PR 2: convs read concatenations and channel slices in place (#185, merged 4cfdb648)

**What.** Every `forward_cat` of the detector (C3k2, C3k, SPPF, C2PSA, the neck's Concat + C3k2) is a 1×1 conv over
a channel concatenation. The GPU backend's `conv2d_act_cat` now runs it as one GEMM whose `B` rows come from up to
four part buffers (`kernels/gemm_cat.rs`, `CAT` in the GEMM and tensor-core generators), and a non-depthwise conv
reads a channel-slice input (`narrow` along dim 1) through the GEMM's `B` offset and image stride instead of
gathering it. Inference only (a recorded forward keeps `cat` then the conv). Review found an empty-batch panic, a
bias extent check that ignored batch strides, no check of the device's storage-binding limit, empty parts and
output aliasing not handled, and routing tests that could not prove the path taken; all fixed before merge (the
tests now trace the dispatches, and `crates/folo-vision/tests/gpu_in_place.rs` checks that a fused folo-n forward
at batch 1 encodes no copy).

**Op counts** (`gpu_profile` trace, folo-n): 165 → 117 operations per forward; at b1 the 48 buffer copies are gone,
at b16 42 `scatter` → 6 and 13 `move` → 1 (0.92 of 8.47 ms replayed GPU time in f32, 0.34 of 4.51 ms in f16 TC).

**A/B against `main` 647b981f** (`gpu` bench, forward + predict, 3 rounds alternated, GPU 0–5% before each run,
CPU 10–100%; `pr2-infer/`, binaries `base-gpu` / `new2-gpu`). Pipelined rows (calls back to back, one wait):

| row | main | PR 2 | |
|---|---|---|---|
| folo-n f32 forward b1 | 2.25 ms (2.05–3.02) | 1.63 (1.55–1.71) | −28% |
| folo-n f32 forward b16 | 10.03 ms (9.68–10.74) | 7.98 (7.94–8.89) | −20% |
| folo-n f16 TC forward b1 | 3.12 ms (2.07–3.21) | 1.28 (0.99–1.51) | −59% |
| folo-n f16 TC forward b16 | 5.19 ms (5.11–7.11) | 4.53 (4.43–4.70) | −13% |
| folo-s f32 forward b1 | 3.69 ms (3.66–3.95) | 3.33 (3.30–3.35) | −10% |
| folo-s f32 forward b16 | 24.04 ms (23.96–37.10) | 20.33 (20.16–24.17) | −15% |
| folo-s f16 TC forward b16 | 13.07 ms (13.06–13.21) | 11.57 (11.50–11.60) | −11% |

End to end: folo-n f32 e2e b16 18.28 → 13.00 ms (875 → 1231 img/s), f16 TC e2e b16 12.92 → 9.27 ms, folo-s f32 e2e
b16 31.21 → 25.64 ms. The spread across rounds reached 20–55% on some rows (contention), but the direction held in
every round of every row except folo-s f32 b16 round 3. A later run against `main` with this PR merged
(`pr3-infer/`, binary `main185-gpu`) read folo-n f32 b1 pipelined 1.56 ms, b16 7.96 ms, f16 TC b16 4.44 ms.

**Accuracy** (`folo val --model D:/folo-weights/folo-n-coco.folo --data D:/folo-data/coco`, val2017, 5000 images,
a build with this PR and the residual change below): f32 mAP50 0.44944638 / mAP50-95 0.30999148, identical to
`main`; `--half` 0.44942 / 0.31000 against `main`'s 0.44944 / 0.31003 (`val.txt`).

## Negative results

- **Residual adds in the conv epilogue** (item 2, branch `gpu/fused-residual`, not merged).
  `Backend::conv2d_act_add`, `ConvBnAct::forward_add`, `RESID` in the GEMM / tensor-core epilogues (a fifth
  binding read like a conv input); Bottleneck and PSABlock route their residuals through it, 13 of the 14
  elementwise adds of a folo-n forward go (104 operations per forward). Tests passed, f32 matches conv + add to an
  ulp (bit for bit where the driver does not contract `act(v) + r` into an FMA). Against `main` with PR 2 merged
  (`pr3-infer/`, 3 rounds): replayed GPU time −1–2% (the adds were only 0.1–0.2 ms of a b16 forward), wall clock
  a tie or worse: folo-n f32 b16 pipelined 7.96 → 8.04 ms, f16 TC b16 pipelined 4.44 → 4.79 ms, b1 0.95 → 1.23 ms,
  folo-s within ±3% (an earlier window read f16 TC b16 4.53 → 4.41). Not a clear win, so not shipped; the 13
  dispatches it removes are worth ~0.05–0.1 ms of host time at b1 at most.
- **Pipelined image upload** (item 4, branch `gpu/upload-pipeline`, not merged): a predict batch over 4 MiB uploaded
  in four runs of equal image counts, each run's copy submitted as soon as it is written so it overlaps writing the
  next. `pr4-predict/` (3 rounds against `main`): upload + letterbox b16 went 2.6–3.0 → 3.2–3.3 ms and e2e b16 +2–4%
  (f32) / tie (f16 TC). The fill is already parallel and memory-bound; four flushes and maps cost more than the
  overlap saves.
- **Sweeping the whole bind cache on every pool eviction** (PR 1's first version): see Training above.

## Leftovers (by expected gain)

1. **Bimodal batch-1 latency** (item 4, not started): synchronised b1 rows of the same binary still swing 2× between
   rounds on this machine (e.g. folo-n f32 forward b1 1.97–9.43 ms); pipelined rows do not. Needs a quiet machine
   and GPU clock logging (`nvidia-smi -q -d CLOCK`) around idle gaps to confirm or rule out clock ramp-down.
2. **Training bind-group reuse**: only ~23% (f32) / ~40% (AMP) of a step's dispatches hit the cache, because the
   pool's buffers come back in a different order each step; and the f32 step creates 55 new pool buffers per step
   (evict + reallocate churn). Making the free order deterministic (autograd's traversal) or pinning a step's
   buffers would turn both into hits.
3. **Residual add fusion** (negative result above): worth retrying only if a GEMM epilogue change makes the extra
   load free in the tensor-core kernel (the f16 TC rows were the ones that lost).
4. **Concatenations with more than four parts or 3×3 consumers** read in place: none in the current detector.

## Second wave (lane `gpu-pool`, 2026-11-12)

The two leftovers above: training bind-group reuse and the bimodal batch-1 latency. Same machine, same method
(binaries copied aside, alternated run by run with the order reversed every other round, GPU lock, `nvidia-smi`
under 10% before each run); baselines `main` 647b981f (`m647`, the shared build) and the `main` this lane branched
from, ed265317 (`base`: a private worktree with only the bench's new counters added, so its rows print the same
numbers). Scripts and raw runs: `D:/folo-data/runs-hill2/gpu-pool/` (`bin/ab.sh`, `bin/parse_train.py`).

### Pool PR 1: the pool hands out the same buffers every step (`gpu/pool-order`)

**Why the bind groups missed.** The pool popped the most recently freed buffer of a size class. A training step
frees its tensors in another order than it allocates them (autograd's traversal, gradients in a hash map whose
order changes per step), so the free list of a class was a different permutation at the start of every step and
request *k* got a different buffer each time: only 22% (f32) / 24% (AMP) of a folo-n b16 step's dispatches found
their bind group. A second cause made it worse in f32: the free lists held at most the live high-water mark
(3467 MiB), but a step needs 4191 MiB of buffers, because its size classes peak at different times (forward
activations, then backward gradients). So every step the pool evicted and recreated 57 buffers, each new one a
new bind-group key, and each new buffer zero-filled by wgpu on first use (GPU time). AMP in a process of its own
churned 46 buffers per step the same way (its cap was the 2 GiB floor, its need 2472 MiB); after an f32 section in
the same process it did not, which is why lane gpu-dispatch saw 0 there.

**What.**
- Each free list is ordered by the buffer's creation number and hands out the oldest free buffer. The choice then
  depends only on which buffers are free, not on the order they came back: the free set at the start of a step is
  the same every step, so by induction every request gets the buffer it got the step before. Where a step
  allocates differently (other shapes), the assignment differs for that stretch and falls back in step when the
  free sets agree again.
- The cache bound adds the *need*: per size class, the most buffers lent out at once in the current or the
  previous window of 4096 requests (about three folo-n steps), times the class size, summed. The free lists may
  hold the need less the live bytes (besides the old 2 GiB floor and live high-water mark), counted at most up to
  twice the live high-water mark. Classes no longer requested (validation, a last partial batch) leave the need
  after two windows and are then evicted least recently freed first, as before.
- The bound is enforced on every allocation miss (before the new buffer is created), on every free, and at every
  need-window rollover after a hit has left the cache (second review round: without it a lower need waited for the
  next miss or free; third round: with the hit still counted as cached, the rollover evicted a buffer the step
  needed, churn every 4096 requests, caught by running the steady-step test past two rollovers)
  (review finding: checked on frees only, a miss could stack a new buffer on a cache filled for the need, about
  3× the high-water mark held across successive workloads). With `M` the 2 GiB floor and `H` the high-water mark
  the pool now holds at most `max(live + max(M, H), min(need, 2H)) ≤ 2·max(M, H)`, never more than the old rule
  (a cache of `max(M, H)` on top of the live bytes).
- `PoolStats` gains `peak_held_bytes` (live + cached at most: the pool's device memory) and `need_bytes`;
  `GpuContext::device_memory()` returns wgpu's allocator report (allocated, reserved). The `gpu` bench prints them
  per training precision, the reserved bytes sampled at the end of every phase of the split steps.

**Numbers** (`pr1-train/`: 3 rounds, f32 and AMP in separate processes, GPU 0–1% before each run, CPU 0–94%
busy (one sample read 0%); median and range of each binary's per-run median). These train numbers were measured
before the bound was enforced on allocation misses too (above); the re-measurement follows the table:

| folo-n b16 | `main` 647b981f | `main` ed265317 | PR | vs ed265317 |
|---|---|---|---|---|
| f32 train step | 49.89 ms (49.42–50.84) | 48.26 (46.34–48.72) | **43.21** (42.94–43.23) | −10.5% |
| f32 host encode, sum of the four phases | 15.5 ms (13.9–17.6) | 12.5 (9.8–17.7) | **8.1** (7.9–13.6) | −35% |
| f32 new bind groups / buffers per step | n/a | 872 / 57 (22% reuse) | **0 / 0** (100%) | |
| AMP train step | 30.61 ms (28.64–31.03) | 28.55 (26.88–28.66) | **27.14** (26.37–27.88) | −4.9% |
| AMP host encode, sum | 13.7 ms (8.8–18.0) | 14.4 (8.7–21.4) | **8.8** (7.1–14.0) | −39% |
| AMP new bind groups / buffers per step | n/a | 979 / 46 (24% reuse) | **0 / 0** (100%) | |
| f32 peak live / held by the pool / device memory | 3467 MiB / – / – | 3467 / 4191 / 4608 | 3467 / 4196 / 4608 | held +5 MiB (+0.1%), rest same |
| AMP peak live / held / device memory | 2022 MiB / – / – | 2022 / 2472 / 2816 | 2022 / 2472 / 2816 | same |

The pool already reached the step's need at its peak before (it held 4191 MiB while the step ran, then evicted
at the end of the backward), so keeping those buffers barely moves it: the pool's held peak is 4191 → 4196 MiB in
f32 (+5 MiB, +0.1%) and 2472 → 2472 in AMP, the peak live bytes are unchanged, and wgpu's allocator reserves the same
4608 / 2816 MiB on both. The split rows show the f32 backward phase 32.6–35.0 → 29.2–30.2 ms (the per-run medians;
`main` 35.0, 32.6, 34.1, the PR 30.2, 29.2, 29.9) with each phase
synchronised: the recreated buffers cost GPU time too (wgpu zero-fills a new buffer before its first use).

**Re-measured after the review fixes** (the bound enforced on allocation misses and at window rollovers; binaries
`pr1b` = 312c080d, `pr1c` = 5484bca0, which differ only on misses and rollovers, neither of which a steady step
has), in busier windows (CPU 59–100%; the encode of both binaries doubled against the first run):

| folo-n b16 | `main` ed265317 | PR | |
|---|---|---|---|
| f32 step (`pr1b-train/`, 3 rounds) | 48.24 ms (48.16–51.02) | **43.47** (42.75–43.93) | −9.9% |
| f32 host encode | 21.2 ms (18.0–23.9) | **14.9** (14.2–15.5) | −30% |
| AMP step (`pr1c-amp/`, 4 rounds, `pr1` = 36ed569c too) | 29.58 (29.35–107.12) | `pr1` 27.94 (27.58–30.62), `pr1c` 29.97 (28.53–32.57) | tie to −5.5% |
| AMP host encode | 19.85 ms (15.5–49.4) | `pr1` 13.65, `pr1c` 15.6 | −21 to −31% |
| new bind groups / buffers per step, memory | as above | 0 / 0, held and device memory as above | |

f32 holds its −10% in every window. **AMP does not clearly meet the −4% target**: −4.9% in the first window,
−5.5% (`pr1`) and +1.3% (`pr1c`) in the second, with base and PR rounds of 50–107 ms from other load. The AMP step
is GPU-bound (about 28 ms of GPU work against 13–20 ms of encode), so its 25–35% less host encode hardly shows; the
f32 step also gained GPU time (no zero-filled new buffers, about 700 MB per step).

The last review fix (a rollover hit leaves the cache before the eviction, b60d69bf, `pr1d-train/`) was measured in a
window where the machine was saturated (CPU 91–100%, `main`'s own f32 rounds read 88–403 ms): the PR's f32 rounds
read 43.0, 45.1 and 46.4 ms, as in the earlier windows, with 0 new bind groups and buffers per step; the AMP rounds of
both binaries (28–60 ms) say nothing in that window.

**Inference** (`pr1-infer/`, `gpu` bench forward + predict, folo-n and folo-s, f32 and f16 TC, 3 rounds, CPU 76–97%):
nothing to change there (an inference forward already hit the cache on every call) and nothing lost. The GPU-bound
batch-16 rows tie: folo-n f32 forward b16 8.23 → 8.23 ms (pipelined 7.76 → 7.77), folo-s f32 20.11 → 20.13
(19.66 → 19.61), folo-n f16 TC pipelined 4.48 → 4.45, folo-s f16 TC pipelined 11.66 → 11.42. Batch-1 rows of both
binaries swing between a fast and a slow mode by run (the next item); one row read worse in that run, folo-n f32
predict b1 2.41 → 2.70 ms, and a 5-round rerun of it (`pr1-predict/`) read 4.11 (2.07–5.01) → 3.27 (1.80–5.36),
both binaries bimodal (fast runs 1.8–2.1 ms, slow ones 3–5 ms).

Tests (`src/tests/pool.rs`): twelve requests of four classes per step, the held ones freed in a shuffled order
mid-step and at the end, give the same buffer for every request in all twelve steps and create nothing after the
first (it fails with the newest-first rule, checked); a step whose two classes peak apart (4 MiB, then 5 MiB) keeps
all 9 MiB and creates nothing after its first run; a class not requested for two windows is evicted, and forty
classes requested one at a time keep at most twice the live high-water mark. Three older tests that evicted
inside a window now pass two windows first. After review: a step of elementwise ops on a private `GpuContext`, its
temporaries dropped in a shuffled order, creates no buffer and no bind group after warm-up (it fails with
most-recently-freed reuse, checked); a miss evicts before it allocates (4 MiB, 5 MiB, then 6 MiB: 11 MiB held, what
the old rule held, 15 without the fix); three workloads of 16 MiB live (4 × 4, 8 × 2, 16 × 1 MiB) hold at most
32 MiB, twice the high-water mark (48 without the fix); `clear()` drops the need except buffers still lent out;
`reset_peak` restarts the held peak from live plus cached.
### Pool PR 2: where the slow batch-1 rounds come from, and a wait that polls first (`gpu/b1-latency`)

**Diagnosis.** A new bench, `gpu_latency`, times synchronised folo-n b1 forwards call by call (encode, submit, wait),
records the logical processor at the start of the call and at the end of the wait (`rdtscp`; since the review
also right before the wait), samples `nvidia-smi` clocks / P-state / utilisation every 50 ms, can space calls out
(`FOLO_BENCH_GAP_MS`: a sleep after each call), trace the recovery after two idle seconds (`FOLO_BENCH_RAMP=1`;
since the review with the clocks sampled during the trace too) and compare wait spins in one process
(`FOLO_BENCH_SPINS`, interleaved in blocks of 50 calls).
Raw runs: `D:/folo-data/runs-hill2/gpu-pool/lat/`.

- **GPU clocks (the machine's).** Back to back, the GPU sits at P0 (2730 MHz) and a call takes 1.2–1.3 ms (f32) /
  1.0–1.1 ms (f16 TC) at the median. After two idle seconds the first 150–300 ms of calls were slower in three of
  the four traces (`ramp-r*.txt`: f32 r1 2.75 → ~2.0 ms, f32 r2 3.65 → ~1.8, f16 TC r1 3.8–3.9 → ~1.0) and not in
  the fourth (f16 TC r2 starts at 1.54–1.88 ms and stays around there). With a 200 ms sleep after each call the GPU
  stays at P8 (210 MHz graphics, 405 MHz memory: 912 of 1038 samples) and a f32 call takes 16.3 ms at the median.
  With a 30 ms sleep after each call (not a fixed 33 fps schedule; `gap30.txt`, 150 calls per setting) a f32 call
  took 5.78 ms p50 / 15.70 ms p90 with spin 0 and 10.36 / 15.69 with the 2 ms spin: the spin did not help at
  spaced-out calls (a worse median) in that run. The ramp re-run with the clocks sampled (`v2-r1.txt`,
  `v2-r2.txt`, the fixed bench; latency medians and `nvidia-smi` clock readings per 50 ms, two separate
  timelines): in the two traces where the GPU had dropped its clocks while idle, the first calls were 3–9× slower:
  f16 TC r1 read 210–255 MHz (P5/P8) and calls took 7.6–9.0 ms for the first 300 ms, then 1.0–1.5 ms, while the
  reported clock stayed at 255 MHz until 400 ms and read 2550 MHz from 450 ms; f32 r2 read 795–855 MHz, calls took
  3.9–4.2 ms for the first 250 ms, then 1.3–1.6 ms, while the reported clock read 855 MHz until 400 ms. So the
  reported clock rises 150–200 ms after the latency recovers (`nvidia-smi` reports with a lag), and the two do not
  line up to the bucket. f32 r1 kept 2550 MHz through the idle seconds and had no slow start (2.15 ms in the first
  bucket, 1.5–1.6 ms after); its clock read 2730 MHz from 550 ms and its calls dropped to 1.17 ms only at 800–850
  ms, not aligned either. In the fourth trace (f16 TC r2) the GPU read 73% utilisation and P0
  during our idle seconds: another process ran on it outside the lock, and every call took ~3 ms throughout (the
  next bullet). So the slow b1 rounds of the `gpu` bench, which idles a second before every section, warmed up for
  25 calls (~40 ms) and then measured 100 calls (~150 ms), come from the clocks when the GPU had dropped them,
  and from other processes on the GPU otherwise. NVIDIA's driver power management decides the clocks; FOLO cannot set them through
  Vulkan (NVML's locked clocks need admin rights). A machine that serves spaced-out streams should use the driver's
  "Prefer maximum performance" mode for the process. This is not ours, and it is not fixed here.
- **Other load (the machine's).** Encode time doubles when the CPU is busy with other lanes (0.55 → 1.2–1.45 ms
  p50 in the same binary across windows), and in one run the wait alone reached 3–3.7 ms for a tenth of the calls
  while `nvidia-smi` read 95% utilisation against our usual 70–80% (another process on the GPU).
- **Thread wake-up (ours).** The wait blocked in the driver, and the thread often woke on another logical processor:
  measured right before and right after `synchronize` (`v2-r*.txt`, 1000 calls per setting), 463–763 of 1000 calls
  changed processor across the wait, regularly onto an E-core (16–27 on this i7-14700KF; 112–911 calls per
  setting touched one), with cold caches for the next encode. (The first runs counted changes between the start of the call and
  the end of the wait: 525 of 1000.)

**What.**
- `wait_idle` (behind `sync()` and every readback) first polls the device without blocking (`PollType::Poll`) until
  the queue is empty, for up to 2 ms (`GpuContext::set_wait_spin`, `FOLO_GPU_WAIT_SPIN_US`; 0 restores the old
  wait), then blocks as before. The poll is loss-aware like every other (a lost device ends it with `DeviceLost`).
  Calls whose thread changes processor across the wait drop from 463–763 to 27–269 of 1000 (`v2-r*.txt`: f32
  757 → 269 and 645 → 106, f16 TC 463 → 27 and 763 → 96).
- The `gpu` bench's latency rows warm up for at least 25 calls *and* a second of back-to-back calls (half a second in
  the A/B below; raised to a second after the clock-sampled traces showed up to ~450 ms of ramp), so they
  measure the steady state (the clocks are up), as a long Ultralytics run does. This changes the measurement, not
  FOLO: rows before and after this PR are not comparable without it, so the A/B below uses this bench on both sides
  and toggles only the spin.

**Numbers.** In-process (`gpu_latency`, spins interleaved in blocks of 50 calls, 600–1500 calls per setting, 9
runs over four windows; v2 = the bench after review: starting setting rotated per cycle, processor read around the
wait, CPU busy with other lanes throughout), total per synchronised call, spin 0 → 2 ms:

| run | f32 p50 | f32 p90 | f32 p99 | f16 TC p50 | f16 TC p90 | f16 TC p99 |
|---|---|---|---|---|---|---|
| spin-r1 | 1.29 → 1.22 | 1.76 → 1.71 | 1.96 → 2.24 | 1.07 → 1.00 | 1.69 → 1.54 | 2.66 → 2.51 |
| spin-r2 | 1.26 → 1.21 | 1.44 → 1.40 | 4.40 → 2.12 | 1.08 → 0.98 | 1.35 → 1.32 | 1.93 → 1.73 |
| ramp-r1 (busy CPU) | 2.10 → 1.81 | 2.72 → 2.10 | 4.21 → 3.10 | 1.06 → 0.97 | 1.18 → 1.14 | 1.74 → 1.40 |
| ramp-r2 (busy CPU) | 1.93 → 1.72 | 2.22 → 2.07 | 3.66 → 2.55 | 1.77 → 1.55 | 2.23 → 1.97 | 4.86 → 3.65 |
| final-r1 | 1.27 → 1.21 | 1.98 → 1.84 | 3.12 → 2.35 | 1.38 → 1.14 | 1.99 → 1.76 | 2.71 → 2.27 |
| final-r2 | 1.27 → 1.20 | 1.68 → 1.48 | 2.18 → 1.92 | 1.68 → 1.55 | 2.31 → 2.08 | 4.47 → 3.58 |
| final-r3 | 1.79 → 1.63 | 2.71 → 2.08 | 18.38 → 4.17 | 1.78 → 1.54 | 2.16 → 1.94 | 5.19 → 2.92 |
| v2-r1 | 1.73 → 1.59 | 2.57 → 2.50 | 3.00 → 2.95 | 1.17 → 1.02 | 1.62 → 1.44 | 2.09 → 1.91 |
| v2-r2 (another process on the GPU) | 1.91 → 1.81 | 2.35 → 2.42 | 3.58 → 3.49 | 1.85 → 1.77 | 2.89 → 2.74 | 3.41 → 3.36 |

The median drops 4–14% (f32) and 4–17% (f16 TC) in every run, p99 in all but one (spin-r1 f32); the wait phase itself hardly moves
(it is GPU time), the encode of the next call gets faster (the thread keeps its core).

`gpu` bench, forward + predict, folo-n, 6 rounds alternated (`pr2-infer/` rounds 1–3 and `pr2-infer-b/` as 4–6,
merged in `pr2-all/`; CPU 21–88%, GPU 0–9% before each run). Same binary, `FOLO_GPU_WAIT_SPIN_US=0` against the
default; `main` 647b981f with its own (old) bench from the first 3 rounds for reference:

| row | `main` 647b981f, old bench | spin 0 | spin 2 ms | |
|---|---|---|---|---|
| f32 forward b1 | 3.01 ms (2.27–3.12), p90 4.08 | 1.83 (1.35–2.37), p90 1.96 | **1.44** (1.17–1.72), p90 **1.73** | −21% |
| f16 TC forward b1 | 3.42 (3.03–3.54), p90 4.69 | 1.25 (1.12–3.55), p90 1.72 | 1.50 (0.89–1.85), p90 1.97 | +20%, unresolved, see below |
| f32 predict b1 (e2e) | 3.63 (2.99–4.23) | 2.54 (2.07–3.22), p90 3.00 | **2.17** (1.74–3.21), p90 2.52 | −15% |
| f16 TC predict b1 (e2e) | 3.38 (3.03–4.06) | 2.69 (2.03–4.00), p90 3.29 | **1.86** (1.57–2.42), p90 2.29 | −31% |
| f32 forward b16 | 10.34 (10.07–10.48) | 8.34 (7.92–9.61) | 8.15 (7.90–8.26) | tie |
| f16 TC forward b16 | 5.97 (5.90–5.98) | 4.91 (4.77–4.98) | 4.84 (4.65–5.05) | tie |
| f32 / f16 TC predict b16 | 13.81 / 9.49 | 11.73 / 8.39 | 11.79 / 8.27 | tie |

The f16 TC forward b1 row is the one exception, and it is an unresolved regression in these rows (1.25 → 1.50
ms, p90 1.72 → 1.97, 6 rounds). The interleaved in-process runs above, which see the same machine state for both
settings, favour the spin in all nine runs, and this row's per-run medians (0.89–1.85 ms with the spin, 1.12–3.55
without) and its pipelined row, which the spin cannot touch (one wait after 100 calls; 0.77–1.92 ms), swing
widely, which suggests machine load. But round 1 got slower synchronised (1.12 → 1.51) while its pipelined row
improved (1.19 → 1.09), which load alone does not explain, so it is not settled.

Targets (`gpu` bench rows, steady state): folo-n f32 forward b1 1.44 ms median (≤ 1.6) with p90 1.73 (≤ 2.2): met;
f16 TC forward b1 ≤ 1.3: met by the in-process medians in quiet windows (0.97–1.14 ms) but not by the `gpu` bench
row in these busy windows (1.50); e2e b1 ≤ 2.2: met (f32 2.17, f16 TC 1.86). Cold, right after two idle seconds, none
of them is met while the clocks are down (3.9–9 ms for the first 250–300 ms in the clock-sampled traces whose GPU
had dropped its clocks), and not while another process shares the GPU (~3 ms throughout in v2-r2 f16 TC). Calls spaced out in time stay at 6–16 ms at the median
(P8 measured with a 200 ms sleep after each call) unless the driver keeps the clocks up.

**Training step** (the spin also covers its waits: the loss readback, the AMP overflow flag). Same binary, spin off
against on, folo-n b16, built before #198 (the pool still churned): f32, 5 rounds (`pr2-train-f32/`, CPU 11–100%
before the runs): 48.30 ms (47.17–50.05) → 46.72 (45.30–92.14), the spin ahead in rounds 2–4 and two rounds of
either side dragged by other load (p90 103–191 ms); 3 rounds of both precisions (`pr2-train/`): f32 49.39 → 59.81
and AMP 32.98 → 29.87 ms (CPU 76–97%), with rounds from 48 to 66 ms on both sides. No sign of a cost to training; no gain claimed.

**Negative / not done.** Keeping the readback's staging buffer mapped is not possible in WebGPU (a buffer must be
unmapped before the GPU copies into it). The submit phase is 0.08–0.2 ms and does not sleep; nothing to win there.
A keep-alive dispatch to hold the clocks up between spaced-out calls would burn GPU power behind the user's back:
not done, documented as the driver setting instead.
