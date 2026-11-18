# Hill-climb 5: the trainer's host side (lane `trainer`)

Goal: training images per second on full COCO, folo-n at 640 with AMP, from the host side of the training loop: the
data loader (decode once per epoch) and the optimizer window (no host wait for the overflow check). Baseline `main`
cf65f7a5, built in a detached worktree (`D:/folo-wt/hill5-trainer-base`, binary
`D:/folo-data/runs-hill5/trainer/bin/folo-base.exe`). Logs in `D:/folo-data/runs-hill5/trainer/logs/`, scripts in
`.../scripts/`.

**Status on `main`: nothing of this lane is merged.** Item 2's PR, #298 (branch `train/sync-free-window`), was
**closed, not merged**. It measured +13 % img/s on the GPU-bound mini320 set and +5.7 % in the one GPU-bound
full-COCO pair, but about nothing on full COCO with mosaic, which is loader-bound on this machine; and a fourth
review round (GPT-6.2 Sol on 5e6714a5, after the three below) still found two blocking error-path bugs (see
Design). The branch stays on origin for reference. Items 1 and 3 got no PR: `train/mosaic-window` and
`train/amp-step-r5` (the rebase) are branches on origin (5644cff3 and cf4c7f63), with no PR; the original
`train/amp-step` (0b74e5f7) is on origin. The code, tests and docs this notebook describes exist only on those
branches.

## Method

- **Machine.** i7-14700KF (28 threads), RTX 4080 SUPER (Vulkan), shared with the other hill-climb lanes: GPU arms
  run inside one `lock.sh acquire gpu trainer` hold, alternated, at least 3 rounds; CPU-only loader timings inside
  the `cpu` lock. "Load" is what typeperf saw beside each arm: total CPU %, the cores Defender used, the cores
  `folo` used. Builds and tests of other lanes ran beside most of these numbers; the load is stated per run.
- **COCO arms.** `scripts/coco-arms.sh <tag> <secs> <arm>...`: `folo train --data D:/folo-data/coco --model folo-n
  --imgsz 640 --batch 16 --amp --seed 0` killed by PID after `<secs>`, with nvidia-smi and typeperf beside it.
  `scripts/ana.py <dir> [skip]` reads each arm's `metrics.jsonl` step events (per-step wall `t_k = 16 k /
  imgs_per_s_k` from consecutive events) and prints the mean img/s over steps > skip, the median non-window step,
  the median window-end step (every 4th batch at batch 16, nbs 64) and the **cycle**: the sum of the four
  per-position step medians, i.e. the typical time of one whole window. The cycle is the measure of the window
  change; the window-end step alone overstates it, because on `main` the batch after a synchronous window end starts
  on an idle GPU and is fast (position 1 is ~20 ms on `main`, ~26 ms deferred).
- **Prototype.** Branch `train/window-proto` (not for a PR): `FOLO_PROTO_TIMING=1` prints per-phase host times of
  the batch loop; `FOLO_PROTO_DEFER=1` encodes the step without reading the overflow flag first and reads it with
  the next batch's loss (from batch 300 on, at a fixed loss scale `FOLO_PROTO_SCALE=1024`, so that no step
  overflows: the prototype measures the wait only, it does not skip correctly).

## PR #298 (closed): sync-free optimizer window (item 2)

### Where the window's time went (synchronous, main)

`win-mini1`, mini320 images at 640 px (20k COCO train images resized to a 320 px long side, so decode is cheap),
`--mosaic 0 --workers 12`, prototype timing on. Per window (4 batches), host-side phases of the window-end batch:

| phase | ms (s1 / s2 / s3; each arm's last prototype print, epoch 1 or 2) |
|---|---|
| `unscale_clip` (norm partials, clip factor, unscale, **one readback**) | 10.2 / 10.0 / 10.4 |
| optimizer step (MuSGD, ~70 dispatches, 2 submissions) | 2.2 - 8.4 |
| EMA | 0.4 |

The `unscale_clip` readback waits for the window's last backward to finish on the GPU: the GPU then idles while the
host encodes the optimizer, the EMA and the next batch's forward.

### Gate: the prototype that only defers the readback

Same arms, 3 alternated rounds (`logs/win-mini1/`, s = synchronous, d = deferred, steps 301-1250 of the epoch):

| round | cycle sync | cycle deferred | saving | window-end extra sync / deferred | mean img/s sync / deferred | load: CPU, Defender (sync / deferred) |
|---|---|---|---|---|---|---|
| 1 | 131.6 ms | 112.8 ms | −18.8 ms | +14.0 / +4.1 ms | 411 / 549 | 90%, 5.5 / 99%, 2.5 cores |
| 2 | 116.1 ms | 111.7 ms | −4.4 ms | +14.2 / +5.0 ms | 537 / 568 | 100%, 1.3 / 99%, 1.1 cores |
| 3 | 114.3 ms | 110.1 ms | −4.2 ms | +13.2 / +4.0 ms | 555 / 496 | 96%, 1.3 / 90%, 1.1 cores |

Positions 1-3 of each window take 75.0 ms (sync, rounds 2-3) against 79.8 ms (deferred): the deferred
window keeps the GPU fed, so its window-end step is 9-10 ms shorter, but the next batch no longer starts on an idle
GPU and is ~5 ms slower. Per whole window the prototype saves 4.2-4.4 ms in the two quiet rounds (round 1 had
Defender on 5.5 cores during the sync arm). **Against the 5 ms gate this is a miss by the cycle measure** (median
4.4 ms) and a pass by the window-end measure I first used (9.2-9.9 ms); I went on with the implementation on the
window-end number, and found the error in review (below). The full implementation measures better than the
prototype (it also drops the second submission of the window end): see the A/B.

### Design

- **Step guard** (folo-core `StepGuard`, `Backend::set_step_guard`; folo-gpu `context/guard.rs`): while a guard
  is set, the optimizer updates (`optimizer_step`, `optimizer_step_many`, Muon's apply) bind a copy of a
  one-element flag and return before writing when it is not finite (exponent bits all set). The copy is taken in
  order with the queued work, so the flag buffer can be reused. The guard is per thread (a first version kept it
  on the shared context, and the workspace tests caught a guard test's NaN guard skipping another test's steps on
  the same device; a server with two trainings would have done the same). Guarded kernels are separate pipelines
  (`optim_guarded`, `optim_many_guarded`, `muon_apply_guarded`), so nothing changes for unguarded callers.
- **The flag** is the clipping norm: `clip_grad_norm_device` already returns a device scalar that is finite exactly
  when every gradient is (its overflow-safe sum). `GradAccumulator::unscale_clip_deferred` returns it without the
  readback.
- **Settling** (`Trainer::resolve_step`): after the next batch's loss readback (the flag was computed long before,
  so this read does not wait) and before that batch's backward, which uses the loss scale. Then the scaler moves
  and the schedule advances; a skipped step takes back the optimizer's step counter (`Optimizer::unstep`) and the
  EMA's update count (`ModelEma::unstep_update`): the guarded kernels did not touch the tensors. A stop or the end
  of an epoch settles before the checkpoint, so `folo.amp`, `optim.__step` and `ema.__updates` are those of the
  synchronous path and a resume is bit-exact.
- **Fallbacks** that are not guarded kernels (tensor-op paths: a device without the multi-tensor kernels) ask
  `step_guard_allows(device)` (a readback) first. CPU training, AMP without clipping (`clip_grad_norm = 0`) and the
  f32 path are unchanged.
- **Review fixes** (round 1, Sol 6.2 and Opus 5.5): a stop the settle raises (the 50th skip) restores the batch
  snapshot and returns before that batch's backward, as the synchronous path stopped before its forward; a loader
  error settles the pending step before it returns; SGD's `unstep` drops the momentum buffers a skipped first step
  allocated; the per-parameter fallback runs on the guarded thread (the guard is a thread-local, rayon workers do
  not see it); `StepGuard` is not `Send`; the guard binds its own copy of the flag in every path; a device with
  fewer than 5 storage bindings per stage reports no guard and keeps the synchronous path; a stale guard of a
  dropped context is keyed by a `Weak`, not an address. A skipped step still bumps the parameter versions (the
  kernel ran); that only costs a cache refill and is documented in the branch's CORE.md.
- **Review fixes** (round 2): every failure between receiving a batch and settling the previous window's step (the
  batch's input, forward and loss, the checkpoint writer's poll) settles that step first, so its own failure (the
  50th skip) is the one reported and a stop it raises ends the epoch as a stop (`Trainer::fail_after_settling`);
  the guard takes one snapshot of the flag, which the device's guard and the fallbacks' check both read; the
  fallbacks' check covers the guard's device only; a guard's drop clears the backend that set it, not whichever
  one the device maps to by then; `GpuContext::set_step_guard` itself refuses a device with fewer than five storage
  bindings; the guard tests skip on such a device. Kept: a run that fails on the 50th skip in a row has reported
  one more `Step` event (the window's last batch, whose loss was computed) than a synchronous run.
- **Review fixes** (round 3, the last the round's rules allow): a checkpoint writer's failure settles the pending step and is still the
  one reported, even where settling raises a stop; a failed BatchNorm restore no longer skips settling or masks
  the step's own error; the MuSGD and EMA tensor-op fallbacks refuse tensors on more than one device under a guard
  (`optim::guard_allows`). Kept and documented (the branch's CORE.md): two device ordinals sharing one GPU context
  are told apart by ordinal in the fallbacks' check but not in the kernels' guard (the trainer uses one device).
- **Round 4** (GPT-6.2 Sol on 5e6714a5, the lead's final review; not fixed, the PR was closed after it): two
  blocking error-path defects. An encoding error inside a guarded window returns before the overflow flag is
  settled, so there is no loss-scale backoff and no skip-streak count (`main` skips the step and keeps the 50-skip
  rule). Two GPU ordinals sharing one context can apply a partial optimizer update: the fused kernel skips on the
  context's guard while the per-parameter fallback updates. Two should-fixes: an overflow event emitted after a
  failed checkpoint write, and tests that derive their expectation from the guard itself. The lead's note on
  closing: a future attempt should keep the host-side flag read and only defer it, without a device guard.

### Tests (on the branch)

- `crates/folo-nn/tests/gpu/guard.rs`: SGD, AdamW and MuSGD with an EMA on the GPU, under a guard on NaN, +inf and
  -inf, leave every model tensor, optimizer state and EMA value bit-identical; under a finite flag the step is the
  unguarded step bit for bit. The guarded kernels are in the kernel registry.
- `guard_rules`: one F32 element, one guard per thread, `step_guard_allows` (for the guard's device only), a write
  to the flag after the guard started moves neither the device's decision nor the fallbacks', and an update another
  thread encodes meanwhile on the same device still runs.
- `per_parameter_fallback_stays_guarded`: a group the multi-tensor update declines (a transposed parameter beside a
  90,000-element one, above the parallel threshold) still skips under a NaN guard, for SGD and AdamW (it fails when
  the per-parameter path runs on worker threads).
- `crates/folo-vision/tests/gpu_amp.rs` `deferred_steps_end_bit_identical_to_synchronous_ones`: 4-epoch AMP runs
  with SGD, AdamW and MuSGD, deferred against synchronous (`Trainer::set_deferred_steps(false)`): every checkpoint
  tensor and the loss scaler equal. The initial scale overflows the first windows; the test asserts that at least
  one step is skipped and at least one taken (with 2 epochs every window skipped, which a review caught), and that
  all 8 windows went through the guard. A mutation that drops `unstep` fails it.
- `amp_run_resumes_bit_exactly_with_its_loss_scaler` (existing, now on the deferred path): resume, an injected
  overflow epoch and the 50-skip rule across a resume.

### A/B

`folo-base.exe` (main cf65f7a5) against this branch, `scripts/coco-arms.sh`: 130 s arms of folo-n AMP at 640,
batch 16, seed 0, alternated inside one GPU-lock hold each, steps 301-1250 of epoch 1. "cycle" is defined under
Method. Other lanes' builds and tests ran beside every set; the load is per arm.

**GPU-bound: mini320 at 640, `--mosaic 0 --workers 12`.** The measure that resolves the change. Three sets:
`sync-mini2` (the branch before the review fixes, `folo-sync.exe`, base first), `sync-mini3` (after the fixes,
`folo-sync2.exe`, new first) and `sync-mini4` (`folo-sync2.exe`, one discarded warm-up arm, base first).

| set | arm | mean img/s | cycle | window-end extra | GPU util | load: CPU total, Defender, folo |
|---|---|---|---|---|---|---|
| mini2 | b1 main | 438 | 113.6 ms | +14.6 ms | 89% | 80%, 1.0, 6.0 cores |
| mini2 | n1 new | 481 | 112.5 ms | +5.0 ms | 88% | 91%, 0.6, 6.2 cores |
| mini2 | b2 main | 469 | 116.0 ms | +15.4 ms | 91% | 80%, 0.5, 6.1 cores |
| mini2 | n2 new | 586 | 108.1 ms | +2.8 ms | 93% | 81%, 0.6, 7.0 cores |
| mini2 | b3 main | 482 | 113.6 ms | +14.9 ms | 90% | 50%, 0.8, 5.2 cores |
| mini2 | n3 new | 583 | 108.4 ms | +2.0 ms | 96% | 69%, 0.5, 7.2 cores |
| mini3 | n1 new | (303) | 111.0 ms | +2.9 ms | 2% | cold pipeline cache, see below |
| mini3 | b1 main | 382 | 127.7 ms | +21.2 ms | 88% | 79%, 0.6, 5.3 cores |
| mini3 | n2 new | (185) | 109.9 ms | +3.0 ms | 54% | cold pipeline cache, see below |
| mini3 | b2 main | 455 | 110.5 ms | +15.2 ms | 87% | 78%, 0.5, 6.4 cores |
| mini3 | n3 new | 537 | 112.3 ms | +4.0 ms | 92% | 89%, 0.4, 6.7 cores |
| mini3 | b3 main | 520 | 119.0 ms | +15.1 ms | 84% | 92%, 0.3, 6.6 cores |
| mini4 | b1 main | 496 | 122.6 ms | +14.8 ms | 90% | 83%, 0.5, 6.2 cores |
| mini4 | n1 new | 573 | 109.9 ms | +2.6 ms | 85% | 100%, 1.2, 6.7 cores |
| mini4 | b2 main | 436 | 134.6 ms | +24.6 ms | 69% | 100%, 0.9, 5.2 cores |
| mini4 | n2 new | (235) | (238.8 ms) | +8.8 ms | 35% | 100%, 1.0, **2.8** cores: CPU-starved |
| mini4 | b3 main | 390 | 148.5 ms | +20.9 ms | 64% | 100%, 0.5, 5.1 cores |
| mini4 | n3 new | 568 | 110.1 ms | +2.7 ms | 80% | 100%, 0.8, 6.1 cores |
| mini4 | b4 main | 464 | 130.8 ms | +20.3 ms | 69% | 100%, 0.4, 5.6 cores |
| mini4 | n4 new | 521 | 119.2 ms | +3.5 ms | 82% | 100%, 0.5, 6.2 cores |

**Result.** Cycle median **110.1 ms (108.1-119.2) against 120.8 ms (110.5-148.5): −9% per window, about +10%
throughput.** Mean img/s with the same stall rule for every arm (steps longer than 0.3 s dropped: both binaries
have them, from the environment, e.g. mini2 b1 7 gaps / 6.9 s and n1 4 gaps / 3.5 s): **568 (521-586, 7 arms)
against 501 (390-556, 10 arms), +13%.** The raw means (459 against 568, +24%) leave more stall time in the `main`
arms and overstate the gain. The three bracketed arms are excluded from img/s; the two cold-cache arms (mini3
n1, n2) keep their cycles in the median, the CPU-starved mini4 n2 does not (with it the median is 110.55 ms). Every pair's cycle favours the branch (−1.1 to −38.4 ms per window without
the bracketed arms) except mini4's starved n2. The gain grows with the CPU load, likely because `main`'s host
thread waits for the GPU at every window end and then has to be scheduled again before it encodes anything, and
with the CPU at 100% that costs more (mini4's base cycle 123-149 ms against 114-116 ms in the quieter mini2). In
the quietest set (mini2, CPU 50-91%) the saving is −1.1 / −7.9 / −5.2 ms per window and the stall-free img/s
change −1% / +16% / +13% (the first pair is a tie).

Excluded arms: `mini3` n1 and n2 were the first two launches of a freshly built binary. The arms are killed by
PID, so the run never writes its kernel list, and new pipelines compiled on demand: gaps of 6.5 s (n1) and 15-16 s
(n2, three of them) with the GPU at 0%, `folo` at exactly one core and the machine at 12-20% CPU. Their cycle
medians are unaffected and are shown. `mini4` n2 lost the loader to other lanes (folo 2.8 cores at 100% CPU, GPU
35%). `mini4` therefore began with a discarded warm-up arm of the new binary (w0: 527 img/s, cycle 116.6 ms, no
gap over 300 ms). Logs: `logs/sync-mini2/`, `logs/sync-mini3/`, `logs/sync-mini4/`.

**Full COCO, 16 workers** (`folo-sync.exe`, before the review fixes; `logs/sync-mosaic/`, `logs/sync-nomosaic/`).
Full COCO's large JPEGs make these runs loader-bound on this machine, and every arm reads the same seed-0 order, so
later arms find more of it in the page cache: mean img/s does not resolve the change.

| setting | cycle main (b1 / b2 / b3) | cycle new (n1 / n2 / n3) | mean img/s main | mean img/s new | GPU util |
|---|---|---|---|---|---|
| mosaic on | 219.2 / 219.5 / 130.1 ms | 213.0 / 176.6 / 121.6 ms | 212 / 252 / 433 | 239 / 353 / 351 | 40-75% |
| `--mosaic 0` | 120.1 / 119.8 / 150.0 ms | 121.9 / 117.3 / 150.3 ms | 230 / 491 / 378 | 327 / 519 / 363 | 9-89% |

Load: CPU 57-100%, Defender 1.2-4.6 cores. With mosaic on, the window is a small part of a 120-220 ms cycle the
loader sets; with `--mosaic 0` the pairs are +1.8 / −2.5 / +0.3 ms: a tie. The one GPU-bound pair (`--mosaic 0`
round 2, 85-89% GPU) gave +5.7% img/s. The brief's targets for this lane (+20% with mosaic at 16 workers, +8%
GPU-bound) are the sum of items 1 and 2; this item alone moves the GPU-bound case only.

## Item 1: mosaic window (branch `train/mosaic-window`, no PR)

Each image is decoded about once per epoch: a mosaic's partners come from the K samples before it in the epoch's
order, through a window cache keyed by position (`folo-data/src/window.rs`). Batches do not depend on the worker
count (tested). The code, its tests and DATA.md's "Mosaic window" section are on the branch (d111ff99, 5644cff3); it
is opt-in (`mosaic_window = 0` default).

**Gate (b), proxy P1 seed 0** (r4-distill.md protocol: mini320, folo-n, 320 px, 20 epochs, batch 64, AMP, 10
workers, `folo val --coco`): control (`main` cf65f7a5) 0.1255, the same as hill-climb 4's control (no drift); K = 512
**0.1232** (below the control range 0.1238-0.1255: fail); K = 2048 **0.1274** (above the range: pass).

**Gate (a), `benches/pipeline.rs` on COCO train2017, 16 workers**, alternated arms, one process per arm, each its
own epoch:

| run | batches per arm | load | K = 0 img/s | K img/s | ratio per pair |
|---|---|---|---|---|---|
| `loader-ab1`, K = 512 | 600 | other lanes' builds | 658.9 / 631.0 / 650.5 | 978.5 / 1008.0 / 1047.6 | 1.49 / 1.60 / 1.61 |
| `loader-ab2`, K = 512 (cold arms) | 2000 | heavy | 413.5 / 283.4 / 341.7 | 846.7 / 812.4 / 821.4 | 2.05 / 2.87 / 2.40 |
| `loader-ab-k2048b` | 2000 | heavy, CPU 73-98% | 288.1 / 373.2 / 350.2 | 594.4 / 867.8 / 459.8 | 2.06 / 2.33 / 1.31 |
| `loader-ab-k2048c` | 2000 | heavy | 248.5 / 358.5 / 400.9 | 652.4 / 494.3 / 655.8 | 2.63 / 1.38 / 1.64 |
| `loader-ab-k2048-full` | whole epoch | CPU 80-95% incl. the bench | 247.8 / 417.4 / 583.5 | 639.0 / 652.8 / 812.9 | 2.58 / 1.56 / 1.39 |

The ratio depends on the load (K = 0 decodes about 4 images per sample, so it suffers more when other processes
take the cores) and on the bench length (a measured epoch starts with the partners of its first K samples, which
wrap around to the end of the order and are decoded on top: about 20% more decodes in 2000 batches at K = 2048,
about 5% over a whole epoch). The non-decode part of a mosaic sample (the canvas 2.5 ms, the affine warp 3.1 ms,
HSV 1.1 ms, flips; about 8 ms of 16 ms per sample at K = 512) bounds the ratio on a quiet machine.

**Verdict: gate (a) fails.** The whole-epoch bench is the one that matches training; its ratio falls as the machine
quietens (K = 0 at 248 → 417 → 584 img/s, ratio 2.58 → 1.56 → 1.39), median 1.56x. The window reaches 1.8x only when
other processes take the cores K = 0 needs for its extra decodes. With (b) passing at K = 2048 but (a) failing, item
1 stays opt-in on its branch and gets no PR. It remains useful as an opt-in for runs where the loader competes for
the CPU (16 workers beside other work), which the lead may want to ship separately.

## Item 3: `train/amp-step` rebased (branch `train/amp-step-r5`, not shipped)

`train/amp-step` (0b74e5f7, on origin) rebased on `main` cf65f7a5 as cf4c7f63 on `train/amp-step-r5` (on origin) (conflicts in `storage.rs`, `kernels/mod.rs`, ARCHITECTURE.md; clippy clean). A/B
`amp-mini2` (mini320 at 640, `--mosaic 0 --workers 12`, new first, 130 s arms; an earlier set `amp-mini` overlapped a
loader bench of mine and is discarded):

| arm | mean img/s | cycle | window-end extra | load: CPU total, Defender, folo |
|---|---|---|---|---|
| n4 amp-step | 501 | 111.8 ms | +15.5 ms | 72%, 1.3, 7.3 cores |
| b4 main | 574 | 110.9 ms | +15.5 ms | 80%, 0.4, 6.6 cores |
| n5 amp-step | 573 | 110.8 ms | +15.7 ms | 77%, 0.3, 6.8 cores |
| b5 main | 475 | 115.8 ms | +14.5 ms | 72%, 1.0, 6.3 cores |
| n6 amp-step | 435 | 112.2 ms | +15.1 ms | 81%, 0.6, 6.5 cores |
| b6 main | 409 | 120.0 ms | +15.2 ms | 98%, 1.8, 5.4 cores |

Cycle median 111.8 against 115.8 ms (−3.5%), but `main`'s cycle drifted up with the load (110.9 → 120.0 ms as the
CPU went from 80% to 98%) and the first pair is a tie (+0.9 ms); mean img/s pairs −13% / +21% / +6%. Row 10 (the
512-image synthetic trainer set, folo-n AMP, 3 rounds, `logs/amp-row10/ab.txt`): epoch-2 training img/s, median of
the rounds, 289.7 → 294.2 (+1.6%). Not resolved outside ±3%: not
shipped. A quiet-machine re-measure could settle whether the ~3% is real.
