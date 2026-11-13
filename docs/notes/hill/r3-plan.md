# Hill-climb 3: forward plans (lane `plan`)

Goal: rows 1 and 2 of docs/BENCHMARKS.md "FOLO targets" (folo-n GPU forward latency at batch 1, 1.14 ms f32 and
0.95 ms f16 on tensor cores) toward about 0.8 / 0.65 ms. An inference forward of one model on one input shape
issues the same dispatches with the same bind groups every call, yet each call walks the whole tensor-op layer
again (shape checks, pool lookups, bind-group cache lookups, parameter packing): record the forward once as a flat
list of dispatches over pinned buffers, replay it bit-identically, and invalidate conservatively.
Branch point `main` e5a6ccf0, RTX 4080 SUPER, i7-14700KF, 640×640.

## Method

<!-- METHOD -->

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
bottleneck and the ceiling is 0.5–3.7%.

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
- **Where it engages.** `DetectionModel::forward_in_pool` (so `forward`, `Predictor`, `folo val` and the server)
  on a GPU, when the model is fused and in eval mode and gradients are off. A plan is recorded at the second sighting
  of a key (the first call warms the pool and the caches as before) and replayed from the third.
- **Key**: the caller's key (the head mode), the input's dims, dtype and buffer size, and the tensor-core switch.
- **Refusals** (the call runs normally, and a refused key is not tried again until the plans are cleared): grad
  enabled, autocast, a profiling or tracing context, a lost device, an input that is not dense from byte 0 of its
  buffer, a call that uploaded or read back, a result that is a view or shares a buffer, a buffer nobody listed, a
  plan over the memory cap.

### Inputs and outputs: copy slot against rebind

The brief asks to measure both. The first version owned an input slot and output buffers, and a replay copied the
caller's input into the slot and the plan's outputs into fresh result buffers (three copies). The second binds the
caller's buffers directly (dynamic bind groups, see above). Both binaries, alternated, 3 rounds of 4 rounds × 60
calls each (CPU 63–100%), the planned call's p50 against the forward of the same run (median of the 3 runs, min–max):

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

At batch 1 the two are inside each other's noise (the copies are small); at batch 16 the copies (a 79 MB input in f32, 39 MB in f16, and
the head outputs) make the copy variant slower than the normal call. The rebind variant ships; the copy variant's
binary was kept (`D:/folo-build/hill3-plan-variants/copy/plan_gate.exe`).

### Invalidation, one by one

| Trap | What the plan does | Test |
|---|---|---|
| A weight written in place (an optimizer step, `mul_scalar_assign`) | every model tensor's `(TensorId, storage_version)` is compared before each replay (a visitor, no allocation); any change retires the plan and the call runs normally | `weight_changes_retire_the_plan` |
| A tensor replaced by another (`fuse`, loading weights) | the id or the count of tensors differs: retired | same test; an unfused model never plans |
| `to_device` / `to_dtype` (there and back) | new storage, new versions: retired | same test (CPU round trip, f16 round trip whose rounding must show in the replay) |
| Another input shape, dtype, batch or buffer size; another head mode | in the key: another plan (or a recording) | `two_shapes_two_models_and_two_head_modes_alternate`, `replays_are_bit_identical_…` (b1 / b16, 64×96 / 96×64) |
| Two models on one context | each model owns its plans (`ForwardPlans` id in the key) | `two_shapes_two_models_…` |
| Pool eviction, `trim_pool` | the plan owns its intermediates (taken out of the pool), so the pool cannot hand them out or drop them | `context_switches_retire_plans_and_release_frees_them` |
| `release_memory` | every plan dropped (its bytes count as released) and the epoch bumped, so a plan built while it ran is not kept | same test |
| Pipeline cache | a plan holds its pipelines and bind groups (refcounted wgpu handles), so nothing the cache does can free them; the kernel a call picks depends only on the shapes, the adapter and the tensor-core switch | (no state to change besides the switch) |
| The tensor-core switch | in the key, so each setting has its own plan and switching back reuses the old one; `disable_tensor_cores` (a kernel failed) bumps the epoch and retires all | same test |
| Device lost | `ForwardPlans::run` checks before each replay and returns the error | `a_lost_device_is_an_error_not_a_replay` (folo-gpu) |
| Another thread forwarding the same model | a replay is encoded whole under the batch lock, so two replays cannot interleave on the shared intermediates; each gets fresh result buffers | `threads_forwarding_one_model_get_their_own_results` (4 threads × 100 calls, bits checked) |
| A result held while the next replay runs | results are fresh pool buffers per replay, never the plan's | `a_thousand_replays_keep_their_bits_and_results_stay_put` |
| A model dropped while its plans live | `ForwardPlans::drop` removes its plans from every context it planned on | `context_switches_…` (plan count drops by one) |
| A panic inside the recorded call | the recorder is a guard; the panic drops it and the next call records again | `a_panic_while_recording_leaves_the_owner_usable` (folo-gpu) |
| Gradients on, autocast, training mode, a profiling or tracing context | no plan | `no_plan_with_gradients_on_or_in_training_mode` |
| An upload or readback inside the call, an unlisted tensor, a view as result | refused with the reason (`ForwardPlans::refusal`) | `uploads_and_unlisted_tensors_make_a_call_unplannable` (folo-gpu) |

### Memory

A plan pins its intermediates (the buffers the call's pool handed out, taken out of the pool) and one uniform buffer
of 256 bytes per dispatch (37 KiB). `ctx.plan_stats()` after the gate's calls:

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

<!-- NUMBERS -->

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

<!-- RECT -->

## Leftovers

<!-- LEFTOVERS -->
