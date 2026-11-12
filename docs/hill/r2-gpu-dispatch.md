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
the sweep on submission. Its measurement is pending.

## Negative results

## Leftovers
