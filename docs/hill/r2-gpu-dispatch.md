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
- The bound is enforced on every allocation miss (before the new buffer is created) as well as on every free
  (review finding: checked on frees only, a miss could stack a new buffer on a cache filled for the need, about
  3× the high-water mark held across successive workloads). With `M` the 2 GiB floor and `H` the high-water mark
  the pool now holds at most `max(live + max(M, H), min(need, 2H)) ≤ 2·max(M, H)`, never more than the old rule
  (a cache of `max(M, H)` on top of the live bytes).
- `PoolStats` gains `peak_held_bytes` (live + cached at most: the pool's device memory) and `need_bytes`;
  `GpuContext::device_memory()` returns wgpu's allocator report (allocated, reserved). The `gpu` bench prints them
  per training precision, the reserved bytes sampled at the end of every phase of the split steps.

**Numbers** (`pr1-train/`: 3 rounds, f32 and AMP in separate processes, GPU 0–1% before each run, CPU 0–94%
busy (one sample read 0%); median and range of each binary's per-run median). These train numbers were measured
before the bound was enforced on allocation misses too (above) and are to be re-measured:

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
