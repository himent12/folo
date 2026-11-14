# Hill-climb 3: forward plans (lane `plan`)

Goal: rows 1 and 2 of docs/BENCHMARKS.md "FOLO targets" (folo-n GPU forward latency at batch 1, 1.14 ms f32 and
0.95 ms f16 on tensor cores) toward about 0.8 / 0.65 ms. An inference forward of one model on one input shape
issues the same dispatches with the same bind groups every call, yet each call walks the whole tensor-op layer
again (shape checks, pool lookups, bind-group cache lookups, parameter packing): record the forward once as a flat
list of dispatches over pinned buffers, replay it bit-identically, and invalidate conservatively.
Branch point `main` e5a6ccf0, RTX 4080 SUPER, i7-14700KF, 640×640.

## Method

- **Benches.** `plan_gate` (new, see Step 0) for the gate and the copy / rebind comparison; the folo-vision `gpu` bench
  (forward and predict sections, rows 1–6 of the targets table, and the train section, row 9) and `gpu_latency`
  (1000 synchronised b1 calls per configuration: encode / submit / wait percentiles) as two binaries, the branch
  point (`D:/folo-wt/hill3-plan-base`, target `D:/folo-build/hill3-plan-base`) against the branch, copied aside to
  `D:/folo-build/hill3-plan-bins/{base,new}/`; the in-process toggle (`FOLO_GPU_NO_PLAN=1` on the new `gpu` binary)
  against the same binary with plans on. `FOLO_GPU_TENSOR_CORES=1`, precisions f32 and f16 on tensor cores.
- **Runs.** `ab.sh` alternates the two binaries run by run (order reversed every other round) under the GPU lock
  and first waits up to 2 minutes for the CPU load to read under 30% three times in a row (`quiet.sh`, added after
  the first three rounds), then logs the GPU and CPU load. `parse.py`, `parse_lat.py` and `parse_gate.py` give the
  median over runs of each side's per-run median, with min–max. Raw runs and scripts:
  `D:/folo-data/runs-hill3/plan/`.
- **Noise.** The machine was shared all afternoon: the other lanes' builds and a 10-worker Python job kept the CPU
  at 60–100% most of the time, the owner played a game (10–16% GPU), and in one window something outside the lock
  held the GPU at 100% with 5.4 GiB (it made the first train A/B worthless: steps 3–30× slower on both sides). The
  CPU read under 30% at the start of only a few runs. A host-bound batch-1 call is exactly what CPU contention
  slows, so each table says what load it ran under, and the quietest measurement (`gpu_latency`, 4 of its first 6
  runs at 3–20% CPU) is reported separately from the loaded ones.

## Step 0: the gate

`cargo bench -p folo-vision --bench plan_gate` (new) builds two identical fused models per configuration, one with
plans off, and times four ways of issuing the same folo-n / folo-s forward in one process, 12 rounds, the order of
the four rotating every round, 60 synchronised calls each per round:
- **forward**: the normal call;
- **replay**: the call traced once (`trace_begin` / `trace_end`), then every traced op re-encoded with `ctx.replay`
  (no tensor-op layer, but still the bind-group cache and the uniform arena per dispatch);
- **frozen**: the trace made into a flat list once (`ctx.freeze`: one bind group per dispatch, every parameter block
  in one uniform buffer) and encoded with `ctx.replay_frozen`: the ceiling of this idea;
- **planned**: the second model's forward through its plan (the shipped path, with its per-call checks).

Run 2026-11-13 10:00, right after the GPU training marker went; GPU idle at the start, CPU load 93–97% from the
other lanes' builds (`D:/folo-data/runs-hill3/plan/gate1/`, `gate.sh`). Median of the 720 calls of each mode, ms:

| config | forward p50 / p90 / p99 | frozen (ceiling) | planned | ceiling | planned |
|---|---|---|---|---|---|
| folo-n f32 b1 | 1.862 / 2.525 / 3.830 | 1.288 / 1.630 / 2.830 | 1.347 / 1.868 / 3.592 | −30.8% | −27.7% |
| folo-n f16 tc b1 | 1.818 / 3.210 / 7.377 | 1.109 / 1.811 / 3.692 | 1.139 / 1.924 / 3.315 | −39.0% | −37.3% |
| folo-s f32 b1 | 2.590 / 3.468 / 7.736 | 2.337 / 2.844 / 4.019 | 2.391 / 3.024 / 4.949 | −9.8% | −7.7% |
| folo-s f16 tc b1 | 1.942 / 2.557 / 4.180 | 1.444 / 1.999 / 3.504 | 1.449 / 1.855 / 2.933 | −25.7% | −25.4% |
| folo-n f32 b16 | 8.239 | 8.060 | 8.133 | −2.2% | −1.3% |
| folo-n f16 tc b16 | 4.337 | 4.178 | 4.218 | −3.7% | −2.7% |
| folo-s f32 b16 | 19.841 | 19.744 | 19.825 (no plan: over the cap) | −0.5% | −0.1% |
| folo-s f16 tc b16 | 9.440 | 9.323 | 9.387 | −1.2% | −0.6% |

The ceiling is 26–39% at batch 1 for folo-n and 10–26% for folo-s: **go on**. At batch 16 the GPU is the
bottleneck and the ceiling is 0.5–3.7%. This gate ran on a loaded host; measured again at quiet starts later, the
f32 ceilings fell under 10% (see the end of Numbers).

**Host-time split** of the synchronised folo-n b1 call (medians, ms; enc = host time until the forward returns,
sub = the final `flush`, wait = `sync`):

| | f32 enc / sub / wait | f16 tc enc / sub / wait |
|---|---|---|
| forward | 1.310 / 0.269 / 0.268 | 1.369 / 0.282 / 0.131 |
| replay | 0.678 / 0.215 / 0.470 | 0.766 / 0.226 / 0.235 |
| frozen | 0.593 / 0.199 / 0.494 | 0.628 / 0.204 / 0.262 |
| planned | 0.679 / 0.218 / 0.450 | 0.709 / 0.217 / 0.207 |

- tensor-op layer (shape and dtype checks, autograd bookkeeping, pool lookups, parameter packing, output tensors):
  forward − replay = 0.60–0.63 ms, about half of the encode time;
- bind-group cache lookups and uniform-arena writes: replay − frozen = 0.09–0.14 ms;
- wgpu encoding plus the batch's own submissions: frozen = 0.59–0.63 ms for 138–146 dispatches, about 4 µs each,
  which no host-side change in folo-gpu can remove;
- the plan's own per-call work (checking 300–400 tensors' versions, allocating the results, the dynamic bind groups
  of the last dispatches): planned − frozen = 0.03–0.09 ms;
- sub and wait: the GPU's work and the tail of the batch; wait grows when the host gets faster because less of the
  GPU's work overlaps encoding.

The machine was loaded (CPU 93–97%), so the absolute times are 30–60% above the quiet ones of docs/BENCHMARKS.md
(1.14 / 0.95 ms); the relative gains are what the gate needs. The A/B below has its own load numbers.

## Step 1: the plan

### Design

- **Recording.** `GpuContext::record` (every dispatch, copy and clear the context encodes) also feeds a
  thread-local recorder when one is open for that context; the pool reports every buffer it hands out
  (`alloc_usage`), and uploads (`write`, `upload_with`, `upload_staged`) and readbacks (`begin_read`) taint the
  recording. A global counter of open recorders keeps the thread-local lookup off the path when nothing records.
  The recorder is a guard: a panic inside the recorded call drops it and the next call starts clean.
- **Buffer roles.** Every buffer the recorded operations touch gets one role, or the call is refused:
  - the caller's input: bound per replay;
  - a result (a buffer handed out during the call that one of the returned tensors owns whole and dense): bound per
    replay to a fresh pool buffer of the same size, which becomes the caller's result, so a replay never writes a
    buffer a caller still holds (the "output aliasing" trap: there is nothing to detect, the plan never owns an
    output);
  - an intermediate (handed out by the pool during the call, not a result): the plan takes it out of the pool's
    free list (`BufferPool::adopt`), or allocates one of the same size when another thread holds it right now, and
    owns it from then on;
  - a model tensor (listed by the caller) or the context's dummy buffer: kept as is;
  - anything else (a buffer the call read but nobody vouches for): refusal, "not listed".
- **Freezing.** Dispatches that bind neither the input nor a result get their bind group made once. The others
  ("dynamic") get theirs made at replay, and kept while the same buffers come back (a pooled input usually is the
  same buffer call after call; results never are, so the last few dispatches make a bind group per replay). Every
  dispatch's parameter block gets its own slot in one uniform buffer written once, so a replay writes nothing.
- **Replay** encodes the steps into the open batch under one hold of the batch lock and keeps the batch's
  submission schedule (`bump` per step), so it overlaps host and GPU as the normal call does.
- **Where it engages.** `FoloModel::forward_in_pool` on a GPU, when the model's plans are on, it is fused and in
  eval mode, and gradients are off. A plan is recorded at the second sighting of a key (the first call warms the
  pool and the caches as before) and replayed from the third. Plans are off on a `FoloModel` on its own and
  `Predictor::new` turns them on (so `folo predict`, the server and `predict_batch` use them; `folo val` and training
  do not). The reason is the contract below: a plan checks the model's tensors, not the layers' settings.
- **Contract.** The call must be a pure function of its input, the listed tensors and the key. What it picks from
  elsewhere is not checked: a layer's activation, a conv's stride, a block's `residual`, SPPF's `k` are public
  fields of `FoloModel`, and changing one after recording would replay the old kernels (round 1 of review found
  this). `Predictor` owns its model and only lends it out as `&FoloModel`, so none of them can change under its
  plans; a bare model opts in with `set_forward_plans(true)` and must turn plans off before changing a setting.
- **Nesting.** A `ForwardPlans::run` made while the same thread records another call runs as it is: its work
  (dispatches and buffers) becomes part of the outer recording, instead of a replay the outer recorder would not
  see.
- **Key**: the caller's key (the head mode), the input's dims, dtype and buffer size, and the tensor-core switch,
  hashed for the lookup; the plan keeps the whole signature and a replay needs it equal, so two keys whose hashes
  collide never share a plan.
- **Refusals** (the call runs normally, and a refused key is not tried again until the plans are cleared): grad
  enabled, autocast, a profiling or tracing context, a lost device, an input that is not dense from byte 0 of its
  buffer, a call that uploaded or read back, a result that is a view or shares a buffer, a buffer nobody listed, a
  plan over the per-plan cap (256 MiB, see Memory). A plan whose recording overlapped a `release_memory()` (the
  epoch at the recording's start differs from the one under the store's lock at insertion) is dropped, not kept.

### Inputs and outputs: copy slot against rebind

The brief asks to measure both. The first version owned an input slot and output buffers, and a replay copied the
caller's input into the slot and the plan's outputs into fresh result buffers (three copies). The second binds the
caller's buffers directly (dynamic bind groups, see above). Both binaries, alternated, 3 rounds of 4 rounds × 60
calls each (CPU 63–100%), the planned call's p50 against the forward of the same run (median of the 3 runs,
min–max):

| config | copy | rebind |
|---|---|---|
| folo-n f32 b1 | −19.2% (−23.7..+0.6) | −25.5% (−27.4..−7.1) |
| folo-n f16 tc b1 | −32.5% (−34.8..−20.2) | −27.4% (−36.5..−13.7) |
| folo-s f32 b1 | −6.7% (−10.5..−5.9) | −3.9% (−5.7..−2.6) |
| folo-s f16 tc b1 | −17.8% (−18.9..−4.4) | −23.9% (−27.1..−23.5) |
| folo-n f32 b16 | **+6.2%** (+2.4..+8.6) | −1.0% (−1.9..+0.5) |
| folo-n f16 tc b16 | **+3.0%** (+2.5..+3.4) | +1.3% (−0.9..+2.2) |
| folo-s f32 b16 | (no plan) | (no plan) |
| folo-s f16 tc b16 | **+2.7%** (−0.2..+2.7) | −0.1% (−2.8..+0.6) |

At batch 1 the two are inside each other's noise (the copies are small); at batch 16 the copies (a 79 MB input in
f32, 39 MB in f16, and the head outputs) make the copy variant slower than the normal call. The rebind variant
ships; the copy variant's binary was kept (`D:/folo-build/hill3-plan-variants/copy/plan_gate.exe`).

### Invalidation, one by one

| Trap | What the plan does | Test |
|---|---|---|
| A weight written in place (an optimizer step, `mul_scalar_assign`) | every model tensor's `(TensorId, storage_version)` is compared before each replay (a visitor, no allocation); any change retires the plan and the call runs normally | `weight_changes_retire_the_plan`: the same change is made to a reference model that never plans, and the first call after it must not be a replay and must match the reference |
| A tensor replaced by another (`fuse`, loading weights) | the id or the count of tensors differs: retired | same test; an unfused model never plans |
| `to_device` / `to_dtype` (there and back) | new storage, new versions: retired | same test (CPU round trip, f16 round trip whose rounding must show in the replay) |
| A layer setting changed (activation, stride, `residual`, SPPF `k`) | not seen by the plan: plans are off on a bare `FoloModel`, and a `Predictor` never lends its model out mutably | `a_model_on_its_own_does_not_plan_and_a_predictor_does` (the default); the contract is documented on `set_forward_plans` |
| Another input shape, dtype, batch or buffer size; another head mode | in the key: another plan (or a recording) | `two_shapes_two_models_and_two_head_modes_alternate`, `replays_are_bit_identical_…` (b1 / b16, 64×96 / 96×64) |
| Two models on one context | each model owns its plans (`ForwardPlans` id in the key) | `two_shapes_two_models_…` |
| Pool eviction, `trim_pool` | the plan owns its intermediates (taken out of the pool), so the pool cannot hand them out or drop them | `context_switches_retire_plans_and_release_frees_them` |
| `release_memory` | every plan dropped (its bytes count as released) and the epoch bumped; a recording keeps the epoch it started in and is checked again under the store's lock, so a plan recorded across a release is not kept | same test (a release between calls); the overlap itself is not tested (it needs a second thread timed into a recording) |
| Pipeline cache | a plan holds its pipelines and bind groups (refcounted wgpu handles), so nothing the cache does can free them; the kernel a call picks depends only on the shapes, the adapter and the tensor-core switch | (no state to change besides the switch) |
| The tensor-core switch | in the key, so each setting has its own plan and switching back reuses the old one; `disable_tensor_cores` (a kernel failed) bumps the epoch and retires all | same test for the switch (`set_tensor_cores`); the `disable_tensor_cores` path is inspected, not tested (it needs a failing kernel build) |
| Device lost | no recording and no replay on a lost device: the call runs as it is and reports the loss | `a_lost_device_is_an_error_not_a_replay` (folo-gpu) |
| Another thread forwarding the same model | a replay is encoded whole under the batch lock, so two replays cannot interleave on the shared intermediates; each gets fresh result buffers | `threads_forwarding_one_model_get_their_own_results` (4 threads × 100 calls, bits checked) |
| A result held while the next replay runs | results are fresh pool buffers per replay, never the plan's | `a_thousand_replays_keep_their_bits_and_results_stay_put`; `a_pure_call_replays_and_its_results_are_its_own` (folo-gpu: held results rechecked after a replay of another input) |
| A model dropped while its plans live | `ForwardPlans::drop` removes its plans from every context it planned on | `context_switches_…` (plan count drops by one) |
| A panic inside the recorded call | the recorder is a guard; the panic drops it and the next call records again | `a_panic_while_recording_leaves_the_owner_usable` (folo-gpu) |
| Gradients on, training mode | no plan | `no_plan_with_gradients_on_or_in_training_mode` |
| Autocast, a tracing context | no plan | `no_plan_while_tracing`; autocast in folo-gpu's `uploads_and_unlisted_tensors_make_a_call_unplannable` on the small call (a fused `FoloModel` forward fails under autocast, `DTypeMismatch` in `add`, whatever the plans do: `eligible` passes an autocast call straight through; see Leftovers) |
| A profiling context | no plan | not tested (the same check as tracing, `eligible` in `backend/plan.rs`) |
| An upload or readback inside the call, an unlisted tensor, a view as result | refused with the reason (`ForwardPlans::refusal`) | `uploads_and_unlisted_tensors_make_a_call_unplannable` (folo-gpu) |
| A planned call made inside another call's recording | runs as it is, so the outer plan carries its work | `a_call_inside_a_recording_runs_as_part_of_it` (folo-gpu: the outer plan replayed with another input) |

### Memory

A plan pins its intermediates (the buffers the call's pool handed out, taken out of the pool) and one uniform buffer
of 256 bytes per dispatch (37 KiB). Its kept bind groups also hold the last input and results until the next
replay, so a plan's byte count (and the caps) include the input's and the results' sizes since round 1 of review;
the table below was measured before that and leaves them out (a folo-n b1 f32 input is 4.7 MiB).
`ctx.plan_stats()` after the gate's calls:

| | folo-n f32 | folo-n f16 tc | folo-s f32 | folo-s f16 tc |
|---|---|---|---|---|
| b1 | 45.6 MiB | 22.8 MiB | 101.6 MiB | 53.8 MiB |
| b16 | 616.5 MiB | 316.3 MiB | 1036 MiB | 526.2 MiB |

That is about the forward's peak activation memory: the pool hands a call the same buffers every time, and the plan
takes exactly those, so the pool does not hold them besides. Bounds: 8 plans and 1 GiB per context, least recently
used first out; one plan at most 256 MiB (`MAX_PLAN_BYTES_EACH`). The per-plan cap came after the gate: at batch 16
the plans tied the normal call (above) while pinning 316–617 MiB, so they are not made; the size is checked before
any buffer leaves the pool. A plan goes with its owner (`ForwardPlans::drop`) and with `release_memory()`.

## Numbers

### Rows 1–6, two binaries (`gpu` bench, rounds 1–6, CPU load 5–88% at the runs' starts)

Rounds 7–9 were run too and are left out: from 12:48 something outside the GPU lock held the GPU at 52–88% with
4.5–6 GiB (logged before each run), and both sides' batch-1 medians read 4–23 ms. Median over the 6 runs of each
run's median, ms, min–max over runs (`parse.py ab-infer-r1to6`):

| row | config | base | new | |
|---|---|---|---|---|
| 1 | folo-n forward b1 f32 | 1.615 (1.350–1.960) | 1.240 (1.180–1.380) | −23.2% |
| 2 | folo-n forward b1 f16 tc | 1.610 (1.140–2.180) | 1.050 (0.850–3.860) | −34.8% |
| 1 | folo-s forward b1 f32 | 2.435 (2.300–2.590) | 2.430 (2.240–3.020) | tie |
| 2 | folo-s forward b1 f16 tc | 1.920 (1.390–2.360) | 1.500 (1.420–1.840) | −21.9% |
| 5 | folo-n predict b1 f32 | 2.460 (2.060–14.31) | 2.360 (2.070–2.720) | −4.1% |
| 5 | folo-n predict b1 f16 tc | 2.245 (1.570–2.950) | 2.160 (1.430–11.15) | −3.8% |
| 5 | folo-s predict b1 f32 | 3.685 (2.880–6.790) | 3.345 (2.830–3.970) | −9.2% |
| 5 | folo-s predict b1 f16 tc | 2.765 (2.190–3.300) | 2.455 (1.970–3.270) | −11.2% |
| 3 | folo-n / folo-s forward b16 f32 | 8.510 / 20.97 | 8.485 / 21.59 | −0.3% / +3.0% |
| 4 | folo-n / folo-s forward b16 f16 tc | 4.415 / 9.700 | 4.535 / 9.935 | +2.7% / +2.4% |
| 6 | folo-n / folo-s predict b16 f32 | 12.09 / 24.90 | 12.72 / 25.64 | +5.3% / +3.0% |
| 6 | folo-n / folo-s predict b16 f16 tc | 8.555 / 13.39 | 8.550 / 13.54 | −0.1% / +1.1% |

At batch 16 no plan is made (over the per-plan cap): both sides run the same code but for the plan's lookup (a
hash of the signature and a look into the store, refused keys are not recorded again), and the +3.0–5.3% rows have
ranges that overlap the base's (folo-n predict b16 f32: base 11.39–18.82, new 11.39–13.39). The p90 of folo-n
predict b1 f16 tc reads +16.8% (2.865 → 3.345) from two loaded runs (11.80 and 4.15 ms against 1.64–2.54 in the
other four); the toggle below and the latency bench show that configuration faster. The best runs are closer to the
quiet numbers: folo-n forward b1 1.35 → 1.18 ms in f32 and 1.14 → 0.85 ms in f16 on tensor cores.

### The in-process toggle (3 rounds, CPU 10–99%)

The new `gpu` binary with `FOLO_GPU_NO_PLAN=1` against itself, median of 3: folo-n forward b1 +6.0% f32 (1.16 →
1.23, ranges overlapping) and −21.6% f16 tc; folo-s −20.9% f32 and −26.3% f16 tc; predict b1 −40.1% / −39.5%
(folo-n) and +2.5% / −20.1% (folo-s).

### `gpu_latency` (1000 synchronised b1 calls per configuration)

Rounds 1–3 first, the ones that started at a lower CPU load (CPU 3–20% at 5 of 6 run starts; selected by that, not
quiet: the GPU snapshots before them read 76–93% busy), then all 6 rounds:

| config | rounds 1–3 p50 base → new | p90 | p99 | encode p50 | all 6: p50 / p90 / p99 |
|---|---|---|---|---|---|
| folo-n f32 | 1.14 → 1.10 (−3.5%) | −4.1% | −6.0% | 0.42 → 0.21 | −4.7% / −6.9% / −26.3% |
| folo-n f16 tc | 0.87 → 0.82 (−5.7%) | −12.2% | −11.5% | 0.45 → 0.23 | −21.9% / −26.6% / −54.0% |
| folo-s f32 | 2.24 → 2.21 (−1.3%) | +26.9% (2.30–3.71 base) | +13.7% | 0.59 → 0.35 | −1.7% / −2.1% / −5.4% |
| folo-s f16 tc | 1.35 → 1.32 (−2.2%) | −3.5% | −4.3% | 0.53 → 0.28 | −7.3% / −7.0% / −37.4% |

**This is the main finding.** In the lower-load rounds the plan halves the host's encode time (0.42 → 0.21 ms for
folo-n f32), but the call barely gets shorter, because the wait grows by almost as much. My reading (an
interpretation, not measured directly: the GPU was not idle in these rounds either) is that a batch-1 forward is
GPU-bound once its host work is gone. The GPU starts on the first 16 dispatches about 30 µs into the call, and from there the
call takes as long as the GPU's own work: about 1.0 ms for folo-n f32 and 0.8 ms on tensor cores, the quiet call
times with the plan (kernel time, the barriers between dependent dispatches and the submission boundaries). Under
host load the plan is worth 20–36% at the median and up to 54% at p99, which fits the host being the bottleneck
then.
**The 0.8 / 0.65 ms targets of rows 1 and 2 need the GPU's time per forward to shrink** (fewer dispatches, cheaper
kernels), not only the host's; the plan made the host side small enough that it no longer hides that.

The gate was measured under load (CPU 93–97%) and said 26–39% for folo-n. Measured again at quiet starts
(`gate3/`), the ceiling was −8.0% (folo-n f32), −22.9% (f16 tc), −5.4% (folo-s f32) and −20.3% (f16 tc). By the
brief's rule the f32 rows would stop the idea on a quiet machine, the tensor-core rows would not, and the plan was
already built when the quiet numbers came in. The lead should weigh that the quiet f32 gain is 2–5%, and the rows
below that read slower.

### Rows that read slower

Several rows read 3–6% slower at the median, and one +27% at p90, all on a shared machine:

| measurement | row | reading |
|---|---|---|
| in-process toggle | folo-n forward b1 f32 | +6.0% (1.16 → 1.23, ranges overlapping) |
| `gpu_latency` rounds 1–3 | folo-s f32 | p90 +26.9% (base 2.30–3.71 across runs), p99 +13.7%; p50 −1.3% |
| rows 1–6 | folo-n predict b16 f32 | +5.3% (base 11.39–18.82, new 11.39–13.39) |
| rows 1–6 | folo-s forward b16 f32, folo-n f16 tc b16 | +3.0%, +2.7% |
| training step | folo-n b16 f32 | +5.4%; AMP p90 +19.2% while its median read −8.6% |
| `folo val` b32 | second pair | +4.1% (the first pair −0.7%) |

None of them is consistent across the measurements: the same configuration reads faster in another one (folo-n f32
b1 is −23% in rows 1–6 and −3.5% in `gpu_latency`; folo-s f32 is −1.7% / −2.1% / −5.4% over all 6 latency rounds),
and the batch-16, val and training rows run no plan at all (the same code but a lookup, or two atomic loads per
dispatch). I read them as the shared machine, but they are unresolved, not shown to be noise: a quiet re-run is the
first leftover below.

### Validation and training

Measured before review round 1, when a bare model planned by default; since then `folo val` and training do not use
plans at all (a bare model has them off).

- **`folo val`** (folo-n COCO, `D:/folo-weights/folo-n-coco.folo`, 5000 val2017 images, GPU f32), two binaries
  alternated, CPU 19–65% and another process holding 3.3 GiB and 27–37% of the GPU before the runs (outside the
  lock): batch 32 (the default; its plan would pin over 1 GiB, so none is made) 6.06 / 4.82 s base against 6.02 /
  5.02 s new, a tie. Batch 1 (plans replay for every image after the second): 12.87 → 11.77 s. **mAP50-95 is the same
  to the last digit on both sides**: 0.30999136 at batch 32, 0.30999124 at batch 1 (a replay is bit-identical, so the
  whole evaluation is).
- **Training step** (`gpu` bench train section, folo-n b16, 3 rounds, the same outside GPU load, CPU 15–43%): f32
  52.5 → 55.3 ms (+5.4%; base 44.7–55.0, new 51.8–58.6), AMP 38.3 → 35.0 ms (−8.6%). A training step never
  records or replays; what it runs of this change is two relaxed atomic loads per dispatch and one per pool
  allocation (the recorder hooks return at once when no thread records), nanoseconds against a 50 ms step, so I
  expect both differences to be the shared GPU's; that is a hypothesis until a controlled run (see "Rows that read
  slower"). An earlier train A/B (`ab-train-contended/`) ran while something held the GPU
  at 100% and read steps 3–30× slower on both sides; it is not used.
- The machine rebooted at 13:46 during the measurements (no cause seen in this lane's logs; the GPU lock this lane
  held at 13:39 was left with an empty owner file, which `lock.sh` would have treated as fresh forever, and was
  removed by hand).

## Step 2: the candidate-first plan

Main moved under the branch: since f04dce45 (#254, #255) a `Predictor` on a GPU calls
`FoloModel::predict_candidates`, not `forward`, so the forward's plan no longer ran inside a predictor. The
branch was rebased onto it and a second planned call added; the forward's hook stays for callers of `forward`.

### What one plan holds

`predict_candidates_gpu` (model/lazy_gpu.rs) puts the whole device side of a prediction through one
`ForwardPlans::run`: the features, the class branches, the class pass (`detect_class_sites`), the box branch at the
sites (`conv2d_act_sites`), the decode (`detect_decode_sites`), and with DFL fusion `dfl_moments` of the site
raws. Its results are the record tensor and, with DFL fusion, what `site_precisions` reads: the moments and the
three site lists. A replayed prediction is one flat list of dispatches, then the host reads the records back (with DFL fusion
also the moments and the three site lists: five readbacks).

That is possible because the site lists have fixed capacities (`GPU_SITE_SLOTS`, 1024 / 512 / 256 per level): the
dispatches do not depend on the image, only the counts in the lists do, and those live in buffers.

- **Key**: `[SITES_PLAN, o2o branch, moments wanted, conf bits, the three slots]`. What of `NmsConfig` reaches a
  kernel before the readback is only `conf` (the class pass's parameter block); the IoU, `max_det`, the class
  filter, class-agnostic NMS and the fusion settings act on the host after it, except whether DFL fusion wants the
  moments (`wants_precisions(nms)`), which adds dispatches. The slots are constants; they are in the key so that a
  change to them cannot meet an old plan. `ForwardPlans::run` now takes the key as a slice whose length is part of
  the signature, so the forward's one-word key and this one cannot collide.
- **Fallback**: an image with more candidates than slots on some level is answered by the dense head on features
  computed again, unplanned, and merged per image with the plan's answers for the others (as before the plan). A
  device without the site kernels returns no results from the recorded call; a call with no results is now refused
  as a plan ("the call returned no results"), and the dense path runs. The features are not a plan result just for
  the rare overflow.
- **One fix so the recording is accepted**: `dfl_moments` uploaded its bins (`0..rm` and their squares) every call,
  and a plan refuses an upload. The bins are now made once per registered context and `rm` (`device_bins`, a small
  static list keyed by the context serving the device, so a context registered again at the same ordinal gets its
  own; review round 3 found the first version keyed by the ordinal) and listed with the model's tensors, so a
  change to them would retire the plan like any other. No other refusal
  came up: `plan_refusal()` read `None` in every run below.

### Proof

- **val2017, plans on against off** (`FOLO_BENCH_VERIFY=plan` on the `candidates` bench, `verify-plan.sh`):
  the predictor's model (plans on) against a copy with plans off, every candidate field and every precision
  compared as bits, all 5000 images, published folo-n weights. **0 mismatched in all six configurations** (f32, f16,
  f16 on tensor cores, each with `--fusion none` and `dfl`), 4397 of the 5000 answered from a plan in each, no
  refusal. The first run of f16 tc `dfl`
  exited with status 127 right after its header line, with no message; the rerun passed. I have no explanation
  for it (127 from bash is "command not found", which the identical command just before and after was not).
- **Tests** (`tests/gpu_plan.rs`): candidate replays bit-identical with and without DFL precisions (f32 / f16 /
  f16 tc, two shapes, batches 1 and 2); a `conf` change gets its own plan (0.45 against 0.5, both replayed and
  both right); an image over its slots after a replay is answered by the dense head and matches plans off; DFL bins follow a
  context registered again at the same device (fails on the ordinal-keyed cache, `predict_candidates` errors).

### Numbers

Measured once at the end, as asked, after the integration. The owner was expected to game on the GPU tonight;
in fact the `nvidia-smi` snapshots before and after both measurements list no other GPU process (GPU 0% / P5–P8
before), and the CPU read 2–21% before them. That shows a quiet start and end, not that nothing competed in
between. A first toggle run was contaminated (an orphaned A/B job of this
lane, which survived a `TaskStop`, took the lock as "already held" and ran beside it); it is kept as
`toggle-cand-contended/` and not used.

**`gpu_latency` on the candidate-first path** (`FOLO_BENCH_CALL=candidates FOLO_BENCH_PLANS=toggle`, published
weights, 640×640, 1000 synchronised calls per side, the two models interleaved in blocks of 50; CPU 2% before,
13% after; `latency-cand/`). The call reads its candidates back, so it is all "encode":

| config | p50 on / off | | p90 | p99 |
|---|---|---|---|---|
| folo-n f32 | 1.05 / 1.12 | −6.3% | −7.5% | −6.8% |
| folo-n f16 tc | 0.88 / 0.97 | −9.3% | −14.0% | −19.0% |
| folo-s f32 | 2.06 / 2.10 | −1.9% | −2.3% | −3.0% |
| folo-s f16 tc | 1.35 / 1.37 | −1.5% (about a tie) | −0.7% | −3.0% |

**`gpu_trained` toggle** (`FOLO_BENCH_PLANS=toggle`, two predictors of the published weights, 6 rounds of 100
calls per row and side, order alternated; CPU 21% before, 15% after; `toggle-cand/`). **b1** is `Predictor::predict`
over the first 16 val2017 images (row 5), **b1 one** the same image every call, **step** `predict_candidates` on a
prepared input. p50 of all calls, ms, and the median and range of the round medians:

| config | row | on | off | change | round medians on / off |
|---|---|---|---|---|---|
| folo-n f32 | b1 | 1.458 | 1.483 | −1.7%, overlap | 1.458 (1.375–1.582) / 1.511 (1.404–1.518) |
| folo-n f32 | b1 one | 1.415 | 1.480 | −4.4%, overlap | 1.436 (1.329–1.708) / 1.505 (1.397–1.522) |
| folo-n f32 | step | 1.095 | 1.141 | −4.0%, overlap | 1.103 (1.044–1.201) / 1.143 (1.083–1.166) |
| folo-n f16 tc | b1 | 1.232 | 1.365 | −9.8% | 1.232 (1.205–1.255) / 1.379 (1.339–1.395) |
| folo-n f16 tc | b1 one | 1.231 | 1.377 | −10.6% | 1.232 (1.215–1.259) / 1.375 (1.349–1.450) |
| folo-n f16 tc | step | 0.894 | 1.001 | −10.6% | 0.888 (0.879–0.914) / 1.012 (0.978–1.028) |
| folo-s f32 | b1 | 2.262 | 2.294 | −1.4%, overlap | 2.271 (2.211–2.365) / 2.309 (2.255–2.371) |
| folo-s f32 | b1 one | 2.264 | 2.317 | −2.3%, overlap | 2.267 (2.222–2.331) / 2.335 (2.276–2.375) |
| folo-s f32 | step | 1.914 | 1.945 | −1.6%, overlap | 1.908 (1.865–1.991) / 1.943 (1.921–3.176) |
| folo-s f16 tc | b1 | 1.665 | 1.654 | +0.7%, overlap | 1.657 (1.630–1.742) / 1.658 (1.618–1.677) |
| folo-s f16 tc | b1 one | 1.622 | 1.664 | −2.6%, barely overlap | 1.628 (1.592–1.636) / 1.669 (1.632–1.676) |
| folo-s f16 tc | step | 1.252 | 1.315 | −4.8%, overlap | 1.258 (1.222–1.541) / 1.319 (1.285–1.322) |

"Overlap": the two sides' round-median ranges overlap. Only folo-n f16 tc separates cleanly in this toggle
(all three rows, ranges apart); every other row, taken alone, is a tie leaning faster. `gpu_latency`'s blocks
interleave the two models more finely, and there the whole distribution moved for folo-n f32 too (p10 1.03 against
1.09, p50 −6.3%), so I count folo-n f32 at about −4 to −6%. folo-s reads −1.5 to −2.6% in both benches: small, and on
this machine I call it a tie.

Tails: in `gpu_latency` p90 and p99 are better with plans in every configuration. In the toggle (600 calls a side,
so a p99 is the 6 slowest calls) seven tail readings are worse with plans: folo-n f16 tc b1 p99 +11.6% (1.968
against 1.763) and step p99 +65% (1.960 against 1.189), folo-s f32 b1 p99 +13.4% (3.168 against 2.794), folo-s f16
tc step p90 +13.2% (1.539 against 1.360) and p99 +7.5% (1.587 against 1.476), folo-s f16 tc b1 one p99 +8.5% (2.090
against 1.927), folo-n f32 b1 one p90 +7.8% (1.703 against 1.580); the same configurations' other tails read better (folo-n f16 tc b1 one p99 1.492
against 1.835). I read them as a few slow calls, not a pattern, but that is not shown.

Summary: **about 10% for folo-n on tensor cores (−9 to −11% at p50 in both benches), about 5% for folo-n f32, a tie
for folo-s**; the one p50 that reads slower, folo-s f16 tc b1 (+0.7%, 1.665 against 1.654), is a tie
(its round-median ranges overlap). Rows 1 and 2's 0.8 / 0.65 ms are not reached: the quiet candidate call is 1.05 /
0.88 ms for folo-n, and, as in step 1, what is left is mostly the GPU's own time.

## Training steps (not in this lane)

A training step at a fixed batch shape (about 1100 dispatches) has the same structure every step, but a plan of it
would need:

- a static batch shape (the trainer's last partial batch, and multi-scale, would get their own plans or run
  normally);
- the GT-dependent parts at a fixed capacity: the assigner's candidate lists and the loss's per-target buffers
  sized by the batch's targets would have to be padded to a capacity known at recording, with the count passed in
  a buffer instead of in parameter blocks;
- no readbacks inside the step: the loss items and the AMP overflow flag are read back today; they would have to
  stay on the GPU (accumulated and read every N steps) or be read after the replay;
- the plan's own in-place writes to its parameters, moments and BN statistics accepted as the expected storage
  version bumps (today any version change retires a plan), while a write from outside still invalidates it;
- the AMP skip (an overflowing step leaves the weights alone) predicated on the GPU instead of branched on the host,
  and the scaler's state in a buffer;
- the learning rate and other schedule values in a buffer the replay writes, not in recorded parameter blocks.

## Second item: shape-bucketed batches

In its own PR (#259, `gpu/rect-batches`). `Predictor::predict_batch` padded every batch to the square
`imgsz × imgsz` canvas. Now a batch whose images all get the same `rect` canvas (frames of one camera) runs on it:
16 frames of 1280×720 at `imgsz` 640 run on 640×384 instead of 640×640. Any other batch is padded square as before.

- **A first version grouped a mixed batch by canvas**, one pass per group. On the CPU four passes of four images
  read slower than one square pass of sixteen (+4.7% median, +11.5% best run, 6 rounds; `ab-rect-cpu-v1/`), so
  mixed batches stay square.
- **Results.** Every image of a same-canvas batch gets the input `predict` gives it alone: on the CPU the
  predictions are then identical to `predict`'s; on a GPU the last bits can differ (a kernel may sum in another
  order at another batch size). Validation pads square, so mAP does not move.

**CPU** (`cpu_e2e`, folo-n, batch 16, two binaries alternated, 6 rounds under the `cpu` lock, CPU 4–100% at the
starts; `ab-rect-cpu/`): 16 × 1280×720 155.2 → 129.5 ms (**−16.6%**, best run 140.3 → 104.2). The 640×640 (+3.4%)
and mixed (+5.9%) rows run the same code on both sides, with ranges overlapping; they are in line with the
unchanged single-image rows on the same runs (+2–4%), so I read them as the loaded machine.

**GPU** (`gpu` bench predict section, RTX 4080 SUPER, random weights; `ab-rect-gpu-main/`): two
binaries on origin/main ca37efc5 (candidate-first), the base with only the new bench rows, alternated, 3 rounds,
GPU idle and CPU 1–13% at every start. Median over runs of the runs' medians, ms:

| config | 16 × 1280×720 base → new | 16 × 640×640 | 16 × mixed | b1 |
|---|---|---|---|---|
| folo-n f32 | 11.90 → 9.35 (**−21.4%**) | −0.1% | −0.1% | −0.7% |
| folo-n f16 tc | 8.93 → 7.70 (**−13.8%**) | −1.3% | −0.9% | −2.1% |
| folo-s f32 | 22.86 → 15.52 (**−32.1%**) | +0.1% | +0.3% | +0.0% |
| folo-s f16 tc | 13.67 → 10.77 (**−21.2%**) | −0.5% | +0.4% | −1.7% |

Every row but the 1280×720 one is a tie (the ranges over runs are 0.01–0.1 ms wide). Two earlier GPU A/Bs are
not used: one ran beside a game (`ab-rect-gpu-contended/`), and one quiet one (`ab-rect-gpu/`, `ab-rect-gpu-nf32/`)
used binaries built before #255 and showed a slow mode on the new side's mixed row (27–38 ms against 12 ms in 3 of
7 runs; the same ~27 ms mode once on the base's 1280×720 row). On current main the mixed row read 10.4–10.5 ms in
6 of 6 diagnostic runs, with no buffer created during the row (pool counters), and the A/B above ties it. I did
not find the old binaries' cause; it no longer reproduces.

## Files touched outside the lane

At the lead's direction (step 2): lane `lazy`'s `model/lazy_gpu.rs` (the planned call, its key and fallback),
`model/predict.rs` (`dfl_moments` split so the bins can be made once) and the benches `candidates.rs`
(`FOLO_BENCH_VERIFY=plan`) and `gpu_trained.rs` (`FOLO_BENCH_PLANS=toggle`); `gpu_latency.rs`
(`FOLO_BENCH_CALL=candidates`, the toggle, `FOLO_BENCH_WEIGHTS`) was this lane's bench already.

## Leftovers

By expected gain:
- **GPU time of the batch-1 forward** (the floor the plan exposed, about 1.0 ms for folo-n f32 and 0.8 ms on
  tensor cores): fewer dispatches (138–146 per call; lane `lazy`'s fused head), and a look at what the GPU does
  between dependent dispatches (`gpu_profile` sums kernel times; the gap between that sum and the wait is the
  barriers and submission boundaries). Not in this lane.
- **The device letterbox in the plan**: step 2 plans everything from the letterboxed input to the records; the
  letterbox itself still runs unplanned because it uploads the image, which a plan refuses. With the upload moved
  before the planned part (the image into a staging buffer the plan binds as its input) its few dispatches would
  join the plan. b1 and step differ by 0.3–0.4 ms in the toggle (the letterbox, the host's preparation and the
  NMS); how much of that is the letterbox's encoding is not measured.
- **Fewer per-replay checks**: 0.03–0.09 ms of a replay is the plan's own work (300–400 tensors' versions checked,
  results allocated, the last dispatches' bind groups made). A model-wide version counter bumped by every write would
  replace the walk.
- **Training-step plans** (above): the larger prize if the host matters there, but a design of its own.
- **Measure again on a quiet machine**: every number here ran next to other lanes' builds, and some next to GPU work
  outside the lock, including the rows that read slower. The lower-load `gpu_latency` rounds are the closest to it
  for rows 1 and 2.
- **A fused model under autocast** fails (`DTypeMismatch { op: "add", expected: F16, got: F32 }`), found while
  writing the autocast test; plans never see such a call. Not this lane's (lane `lazy` owns `model/`); AMP training
  runs the unfused model.
- **`folo val` at batch 1 with plans**: it read 12.87 → 11.77 s while bare models planned by default. `validate`
  holds the model as `&FoloModel` for the whole run, so it could turn plans on for its duration; not done, since its
  default batch (32) is over the per-plan cap anyway.
