# Hill-climb 5: the trainer's host side (lane `trainer`)

Goal: training images per second on full COCO, folo-n at 640 with AMP, from the host side of the training loop: the
data loader (decode once per epoch) and the optimizer window (no host wait for the overflow check). Baseline `main`
cf65f7a5, built in a detached worktree (`D:/folo-wt/hill5-trainer-base`, binary
`D:/folo-data/runs-hill5/trainer/bin/folo-base.exe`). Logs in `D:/folo-data/runs-hill5/trainer/logs/`, scripts in
`.../scripts/`.

## Method

- **Machine.** i7-14700KF (28 threads), RTX 4080 SUPER (Vulkan), shared with the other hill-climb lanes: GPU arms
  run inside one `lock.sh acquire gpu trainer` hold, alternated, at least 3 rounds; CPU-only loader timings inside
  the `cpu` lock. "Load" is what typeperf saw beside each arm: total CPU %, the cores Defender used, the cores
  `folo` used. Builds and tests of other lanes ran beside most of these numbers; the load is stated per run.
- **COCO arms.** `scripts/coco-arms.sh <tag> <secs> <arm>...`: `folo train --data D:/folo-data/coco --model folo-n
  --imgsz 640 --batch 16 --amp --seed 0` killed by PID after `<secs>`, with nvidia-smi and typeperf beside it.
  `scripts/ana.py <dir> [skip]` reads each arm's `metrics.jsonl` step events (per-step wall `t_k = 16 k /
  imgs_per_s_k` from consecutive events) and prints the mean img/s over steps > skip, the median non-window step
  and the median window-end step (every 4th batch at batch 16, nbs 64).
- **Prototype.** Branch `train/window-proto` (not for a PR): `FOLO_PROTO_TIMING=1` prints per-phase host times of
  the batch loop; `FOLO_PROTO_DEFER=1` encodes the step without reading the overflow flag first and reads it with
  the next batch's loss (from batch 300 on, at a fixed loss scale `FOLO_PROTO_SCALE=1024`, so that no step
  overflows: the prototype measures the wait only, it does not skip correctly).

## PR: sync-free optimizer window (item 2)

### Where the window's time went (synchronous, main)

`win-mini1`, mini320 images at 640 px (20k COCO train images resized to a 320 px long side, so decode is cheap),
`--mosaic 0 --workers 12`, prototype timing on. Per window (4 batches), host-side phases of the window-end batch:

| phase | ms (s1 / s2 / s3, epoch 2) |
|---|---|
| `unscale_clip` (norm partials, clip factor, unscale, **one readback**) | 10.2 / 10.0 / 10.4 |
| optimizer step (MuSGD, ~70 dispatches, 2 submissions) | 2.2 - 8.4 |
| EMA | 0.4 |

The `unscale_clip` readback waits for the window's last backward to finish on the GPU: the GPU then idles while the
host encodes the optimizer, the EMA and the next batch's forward.

### Gate: the prototype that only defers the readback

Same arms, 3 alternated rounds (`logs/win-mini1/`, s = synchronous, d = deferred, steps 301-1250 of the epoch):

| round | window-end extra, sync | window-end extra, deferred | mean img/s sync | mean img/s deferred |
|---|---|---|---|---|
| 1 | +14.0 ms | +4.1 ms | 411 | 549 |
| 2 | +14.2 ms | +5.0 ms | 537 | 568 |
| 3 | +13.2 ms | +4.0 ms | 555 | 496 |

"Window-end extra" is the median window-end step less the median other step. Deferring saves 9.2-10.1 ms per window
(gate: 5 ms): **go**. Load: CPU 90-100% busy (other lanes building), Defender 1-5.5 cores; the mean img/s swing
with that load (round 1 sync and round 3 deferred lost the loader for stretches), the window-end delta does not.

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
  `step_guard_allows()` (a readback) first. CPU training, AMP without clipping (`clip_grad_norm = 0`) and the f32
  path are unchanged.

### Tests

- `crates/folo-nn/tests/gpu/guard.rs`: SGD, AdamW and MuSGD with an EMA on the GPU, under a guard on NaN, +inf and
  -inf, leave every model tensor, optimizer state and EMA value bit-identical; under a finite flag the step is the
  unguarded step bit for bit. The guarded kernels are in the kernel registry.
- `crates/folo-vision/tests/gpu_amp.rs` `deferred_steps_end_bit_identical_to_synchronous_ones`: 2-epoch AMP runs
  with SGD, AdamW and MuSGD, deferred against synchronous (`Trainer::set_deferred_steps(false)`): every checkpoint
  tensor and the loss scaler equal. The initial scale overflows the first windows, so skipped and taken guarded
  steps both occur (asserted). A mutation that drops `unstep` fails it (at the deferred run's checkpoint write).
- `amp_run_resumes_bit_exactly_with_its_loss_scaler` (existing, now on the deferred path): resume, an injected
  overflow epoch and the 50-skip rule across a resume.

### A/B (full COCO)

`folo-base.exe` (main cf65f7a5) against `folo-sync.exe` (this branch), `scripts/ab-sync.sh`: 130 s arms of
`coco-arms.sh` with `--workers 16`, alternated, 3 rounds per setting inside one GPU-lock hold each, steps 301+ of
epoch 1.

**Mosaic on** (`logs/sync-mosaic/`):

| arm | mean img/s | non-window median | window-end median (extra) | GPU util | load: CPU total, Defender, folo |
|---|---|---|---|---|---|
| b1 main | 212 | 50.6 ms | 67.9 ms (+17.3) | 40% | 80%, 3.1, 7.0 cores |
| n1 new | 239 | 51.8 ms | 58.2 ms (+6.4) | 46% | 92%, 4.0, 7.4 cores |
| b2 main | 252 | 50.4 ms | 68.2 ms (+17.8) | 48% | 96%, 4.4, 7.7 cores |
| n2 new | 353 | 42.0 ms | 50.7 ms (+8.7) | 59% | 99%, 4.6, 10.2 cores |
| b3 main | 433 | 29.0 ms | 43.5 ms (+14.5) | 75% | 99%, 1.2, 12.5 cores |
| n3 new | 351 | 29.0 ms | 34.5 ms (+5.5) | 68% | 67%, 1.9, 11.0 cores |

With mosaic on, full COCO at 16 workers is loader-bound on this machine, and the mean img/s does not resolve the
change: every arm reads the same seed-0 order, so later arms find more of it in the page cache (the base arms rise
212 → 252 → 433), and other lanes' builds held the CPU at 80-99%. The window-end step is the robust measure: it is
9.0-10.9 ms shorter per window (round by round: −10.9, −9.1, −9.0 ms), the same saving as the gate prototype.

**Mosaic off** (`--mosaic 0`, `logs/sync-nomosaic/`):

| arm | mean img/s | non-window median | window-end median (extra) | GPU util | load: CPU total, Defender, folo |
|---|---|---|---|---|---|
| b1 main | 230 | 27.4 ms | 40.9 ms (+13.5) | 9% | 57%, 1.3, 3.4 cores |
| n1 new | 327 | 28.9 ms | 35.0 ms (+6.1) | 59% | 81%, 1.4, 5.4 cores |
| b2 main | 491 | 27.3 ms | 39.9 ms (+12.6) | 85% | 99%, 2.4, 7.4 cores |
| n2 new | 519 | 28.2 ms | 32.8 ms (+4.6) | 89% | 100%, 3.0, 7.8 cores |
| b3 main | 378 | 34.2 ms | 47.2 ms (+13.1) | 63% | 100%, 3.3, 5.7 cores |
| n3 new | 363 | 34.6 ms | 46.4 ms (+11.7) | 64% | 100%, 2.9, 5.3 cores |

Not GPU-bound either: full COCO's large JPEGs still starved the GPU for stretches (b1's median utilisation 9%)
while other lanes held the CPU at 100%. The one GPU-bound pair (round 2, 85-89%) gave +5.7% img/s and a window
7.3 ms shorter; round 3, with the CPU saturated, kept most of the stall (−1.4 ms). The clean GPU-bound comparison
is the mini320 one below.
