# Wave 6, lane distill: feature imitation and foreground-gated distillation at the fork gate (lab notebook)

Lane `distill` of wave 6 (2026-11-21). **Result: no arm passed, so no code ships.** The gate was at least +0.0010 COCO
AP over the control with the bootstrap interval's lower end above 0. Temperature 2 is a tie (−0.0002), the two
foreground-gated arms are about 0.0006 below the control, and feature imitation is 0.0011 below it with an interval
entirely under 0. Scratch, scripts, logs and evaluation files: `D:/folo-wave6/lanes/distill/`. Run folders:
`D:/folo-data/runs-wave6/distill/<arm>`.

Terms: **AP** is COCO's mAP50-95; AP50, AP75, AP90 and AP95 are AP at one IoU threshold; AP_S, AP_M and AP_L are AP for
small, medium and large objects. `L_kc` and `L_kd` are the class and DFL distillation terms of docs/MODEL.md §8.9.

## The two candidates

Both came from research reports of this wave, as hypotheses with a gate.

- **F1, feature imitation** (`D:/folo-wave6/research/formula.md`, F1; expected +0.003, a guess from the literature,
  range −0.002 to +0.008). A training-only 1×1 adapter per level maps the student's neck map to the teacher's width.
  With `ẑ = (x − μ_c) / sqrt(σ_c² + 1e-12)`, a map standardised per channel over the batch and the positions:

      L_f = Σ_l (1/2) · mean (ẑ_S − ẑ_T)²  ≈  Σ_l (1 − mean_c r_c)

  where `r_c` is the Pearson correlation of a channel of the adapted student map and the teacher's, and `λf = 0.5`
  multiplies the term. The two sides are equal only while the stabiliser `1e-12` is negligible against the channel
  variances. The adapters had their own AdamW (lr 1e-3) and were outside the model, its optimizer, the EMA and the
  checkpoints.
- **C3, foreground-gated terms and T = 2** (`D:/folo-wave6/research/formula-neon.md`, Candidate 3; expected +0.004 to
  +0.010 AP, concentrated in AP75 to AP95). The per-element `L_kc` and `L_kd` are multiplied by the assigner's
  foreground mask, with the same gains and the same normaliser as the dense terms, and the DFL temperature drops from
  4 to 2. The report's argument is that the teacher's false positives teach the student background, and that T = 4
  costs tight boxes (AP95 0.0561 to 0.0527 when distillation was turned on, one pair of single-seed runs).

The foreground mask needs a correction to the report: it says `Assignment.dense[0, b, a] == 1` iff foreground. The
value is `label + 1`, with 0 for background, so the gate code clamped it to [0, 1].

## The gate

Fixed before any arm ran, with the protocol of [r5-accuracy.md](r5-accuracy.md):

- Copies of `D:/folo-data/runs-hill4/coco/n-r5-ctl-e32` (folo-n at epoch 32 of 64), resumed for epochs 33–36 with
  `--teacher folo-s-coco`, AMP, seed 0, on the GPU.
- One binary for every arm: `bin/folo-gate.exe`, commit `061d14b4`, sha256 `089c60cd55db…`.
- Exported EMA weights, `folo val --coco --save-json`, pycocotools on val2017.
- Paired bootstrap over the 5,000 images (`bench/fusion/bootstrap.py`, 1,000 resamples, seed 0, 95% percentile
  interval). AP75, AP90 and AP95 use the same resamples (`bin/tight.py`), whose full-sample values equal pycocotools'.
- **Pass: at least +0.0010 AP over `ctl` and the interval's lower end above 0.** No second setting of the same idea.

With the new keys absent, the gate binary matched its base commit `7b8ce7e3` exactly: 18 of 18 loss records and all
1,179 tensors, on the CPU path and on the GPU path with AMP and the fused kernel (`bin/identity.sh`,
`bin/identity_gpu.sh`, logs `logs/identity.log` and `logs/identity-gpu.log`). On the GPU the base was run twice, around
the gate binary, and its two runs were identical too.

## Results

| arm | what | AP | − ctl [95% interval] | P(≤ 0) | verdict |
|---|---|---:|---:|---:|---|
| `ctl` | today's distillation (dense, T = 4) | 0.3251 | | | control |
| `f-t4` | the same recipe, hill 5's binary and run | 0.3249 | −0.0002 [−0.0006, +0.0003] | 0.778 | cross-binary control gap |
| `t2` | `distill.temperature=2` | 0.3249 | −0.0002 [−0.0007, +0.0005] | 0.521 | fail |
| `fg4` | `distill.foreground=true` | 0.3245 | −0.0006 [−0.0013, +0.0001] | 0.940 | fail |
| `fg2` | `distill.foreground=true`, `distill.temperature=2` | 0.3244 | −0.0006 [−0.0012, +0.0001] | 0.939 | fail |
| `feat` | `distill.feature_gain=0.5`, fresh adapters at epoch 33 | 0.3240 | **−0.0011 [−0.0016, −0.0004]** | 1.000 | fail |

No arm passed, so no pair was run together.

By threshold and size (`eval/stats-<arm>.json`, `logs/analyse-final.log`):

| arm | AP50 | AP75 | AP90 | AP95 | AP_S | AP_M | AP_L | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ctl` | 0.4677 | 0.3507 | 0.1530 | 0.0454 | 0.1553 | 0.3534 | 0.4654 | 0.5487 |
| `f-t4` | 0.4680 | 0.3503 | 0.1537 | 0.0449 | 0.1557 | 0.3530 | 0.4651 | 0.5473 |
| `t2` | 0.4682 | 0.3505 | 0.1530 | 0.0459 | 0.1553 | 0.3532 | 0.4663 | 0.5485 |
| `fg4` | 0.4671 | 0.3507 | 0.1534 | 0.0447 | 0.1560 | 0.3530 | 0.4646 | 0.5504 |
| `fg2` | 0.4670 | 0.3512 | 0.1541 | 0.0447 | 0.1552 | 0.3530 | 0.4645 | 0.5496 |
| `feat` | 0.4664 | 0.3492 | 0.1538 | 0.0451 | 0.1563 | 0.3523 | 0.4648 | 0.5471 |

The tight thresholds against `ctl`, on the same resamples (`logs/tight.log`):

| arm | AP75 − ctl | AP90 − ctl | AP95 − ctl |
|---|---:|---:|---:|
| `f-t4` | −0.0004 [−0.0014, +0.0007] | +0.0007 [−0.0006, +0.0022] | −0.0005 [−0.0014, +0.0003] |
| `t2` | −0.0002 [−0.0011, +0.0008] | +0.0001 [−0.0016, +0.0023] | +0.0005 [−0.0006, +0.0017] |
| `fg4` | +0.0000 [−0.0015, +0.0016] | +0.0004 [−0.0018, +0.0025] | −0.0007 [−0.0024, +0.0007] |
| `fg2` | +0.0005 [−0.0009, +0.0020] | +0.0011 [−0.0004, +0.0031] | −0.0007 [−0.0023, +0.0010] |
| `feat` | **−0.0015 [−0.0024, −0.0001]** | +0.0008 [−0.0010, +0.0027] | −0.0003 [−0.0015, +0.0011] |

Internal mAP50-95 of the trainer, and the reported distillation terms of epoch 36 (weighted, without `B`):

| arm | e33 | e34 | e35 | e36 | `L_kc` | `L_kd` | feature term |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ctl` | 0.31677 | 0.31837 | 0.31981 | 0.32191 | 0.47156 | 0.49668 | |
| `t2` | 0.31682 | 0.31822 | 0.31983 | 0.32168 | 0.47175 | 0.36145 | |
| `fg4` | 0.31642 | 0.31781 | 0.31934 | 0.32134 | 0.19327 | 0.25250 | |
| `fg2` | 0.31656 | 0.31795 | 0.31958 | 0.32128 | 0.19359 | 0.17992 | |
| `feat` | 0.31657 | 0.31795 | 0.31938 | 0.32083 | 0.46948 | 0.49407 | 0.46263 |

## The reports' claims, answered

- **Claim: foreground gating adds +0.004 to +0.010 AP.** Not at this gate. Both gated arms are 0.0006 below the
  control, and the upper end of both intervals is +0.0001.
- **Claim: gating and T = 2 recover the AP95 lost to distillation (−0.0034).** Not shown at this gate. The −0.0034 is
  a historical size: it compares a distilled and an undistilled model at epoch 64, and the report proposed its own
  test at epoch 44. This gate compares each arm with dense distillation at epoch 36 and has no undistilled arm. There,
  AP95 is −0.0007 for both gated arms, with upper ends +0.0007 and +0.0010, and `t2` alone is +0.0005
  [−0.0006, +0.0017]. These differences are the size of the gap between the two controls (`f-t4` − `ctl` is −0.0005 at
  AP95 and +0.0007 at AP90). The largest tight-IoU gain of the gated arms, `fg2` at AP90 (+0.0011), has an interval
  that includes 0, and it is one of many comparisons read here.
- **Claim: T = 4 is what costs tight boxes.** `t2` changes AP90 by +0.0001 and AP95 by +0.0005, both with intervals
  that include 0. At T = 2 the logged DFL term is 0.73 of its T = 4 value, and AP does not move.
- **Claim: hill-climb 4 already screened T in {2, 4}** (the neon report, repeated in this lane's brief). The table in
  [r4-distill.md](r4-distill.md) has T = 1 and T = 4 only, so `t2` is the first measurement of T = 2.
- **Claim: feature imitation adds about +0.003.** Measured −0.0011 [−0.0016, −0.0004], with AP75 −0.0015. The term was
  not mis-scaled: at epoch 36 it is 0.463 against `L_kc` 0.469 and `L_kd` 0.494, so the one extra `λf` the brief
  allowed for a mis-scaled term was not justified and not run. The term fell during training. Read through
  `1 − loss / (0.5 × 3)`, with the stabiliser taken as negligible, the logged values give a correlation proxy for the
  **adapted** maps of 0.568 (first quarter of epoch 33) and 0.693 (last quarter of epoch 36). The adapters train too,
  so this rise cannot be assigned to the student alone. It did not come with a higher AP.
- **The size of the gated terms.** At epoch 36 the logged `fg4` terms are 0.41 (class) and 0.51 (DFL) of `ctl`'s. These
  are ratios of two arms' means, from students that trained on different objectives for four epochs. They do not
  measure the foreground share of the dense loss, which would need both losses on the same predictions and batches.

## What is not shown

- **A full run.** Each arm is four epochs from epoch 32. The reports' expected gains are for longer training (C3's own
  test was at epoch 44), so this gate rejects them at its own scale and under its fixed bar, and says nothing about
  epoch 64.
- **Run-to-run noise.** The intervals resample images for fixed weights, so they do not include training variance,
  which this lane did not measure. Each arm has one seed. The one related number is the cross-binary gap between the
  two controls (`f-t4` and `ctl`, 0.0002 apart), a single difference that mixes the binaries with anything else that
  differed between the runs. `feat`'s −0.0011 is therefore a measured difference at this gate, not proof that the
  term harms a long run.
- **Cost.** Epoch times are not comparable: other lanes' builds and trainings shared the machine (`ctl` 285–308 s per
  epoch, `fg2` 1,183–2,994 s with the GPU waiting on a loaded host). The foreground arms also leave the fused GPU
  kernel for the tensor path, so their time would not represent a shipped version anyway.
- **The teacher's false positives.** The report's "35–39%" figure was not re-measured.

## Runs and failures

| arm | run folder | detections |
|---|---|---|
| `ctl`, `t2`, `fg4`, `fg2`, `feat` | `D:/folo-data/runs-wave6/distill/<arm>` | `D:/folo-wave6/lanes/distill/eval/dets-<arm>.json` |
| `f-t4` (hill 5) | see [r5-accuracy.md](r5-accuracy.md) | `D:/folo-data/runs-hill5/accuracy/eval/dets-f-t4-e32.json` |

`feat` needed three attempts. The first two died on a failed host allocation: 39,321,600 bytes after epoch 34's
training log lines, before its validation was recorded, and 19,660,800 bytes 21 seconds after the second attempt took
the lock. A memory check by hand just after the second crash read 92% load (not saved in a log). Their run folders are
`feat-crash1` and `feat-crash2`, their logs `logs/feat-crash{1,2}.*`. The third attempt waited for 12 GB of commit and
6 GB of physical memory before taking the GPU lock, and is the only attempt that reached epoch 36. It reproduced what
the first attempt had recorded: the loss fields of all 14,786 overlapping step records, the feature terms of epochs 33
and 34 (0.54180 and 0.48346) and the internal mAP of epoch 33 (0.31657).

Since no arm passed, the keys `distill.foreground` and `distill.feature_gain` are not in the code. The implementation
and its tests are kept outside the repository's history, in `D:/folo-wave6/lanes/distill/distill-gates-code.patch`
(commit `061d14b4`, patch sha256 `1fd98a85cf7caa61…`).
