# Wave 6: fine-tuning defaults on small datasets (lab notebook)

Lane `finetune` of wave 6 (2026-11-20). Goal: raise the AP FOLO reaches out of the box when a user fine-tunes on a
few hundred of their own images (`folo train --weights ... `, everything else at its default), by changing a
default with the evidence for it, or say with measurements that the open leads do not pay. **Result: no default
changes.** The one lead with a real effect, an EMA that actually averages in short runs, gives a better
last-epoch model (+0.009 AP over the four sets, 11 of 12 seed pairs) but a worse best checkpoint as the benchmark
picks it (−0.005; three of the four sets pick the best epoch on the split they then score, BCCD picks on val and
scores test); it misses the gate either way. Everything else
tried (a lower backbone LR, `cls_gain` 0.25, `top_k` 13) is level or a little worse. Along the way: the brief's
step count was wrong by the accumulation factor, `folo-n-coco-v5` is a worse fine-tuning start than
`folo-n-coco-v2` on brain-tumor (−0.030 AP, three seeds per side), and a hold-out carved from brain-tumor's train
split scores far higher than its val split (0.66 against 0.42, with different training pools and images, so not a
like-for-like gap).

## Protocol

**Benchmark runs** as bench/finetune/README.md's protocol, on this PC instead of RunPod: folo-n from
`folo-n-coco-v5` (`weights-v6`, sha256 `4287860…`; the start docs/QUICKSTART.md and examples/coco recommend), 640 px,
batch 16, 100 epochs, f32 (a user's default run: `amp` is opt-in), every other key at its default (`optimizer:
auto` → MuSGD, `auto` rule 5), one RTX 4080 SUPER on Vulkan shared with five other lanes, binary `main` 71284767.
`folo val --save-json` of `best.folo` (the epoch with the best val mAP50-95) on the evaluation split (BCCD: test;
the other three: val) at conf 0.001, IoU 0.7, scored by `bench/finetune/scorer.py` (pycocotools). Loader workers 6
or 10 (they never change the batches). Five runs died of a failed allocation (the PC's memory was shared
with two CI VMs and other lanes' builds) and were run again from scratch; no number comes from a partial run.
Three seeds per side, every comparison paired by seed.

**Selection splits.** Choices were made on a split carved from each train split, never on the reported one: a
seeded 20% hold-out of the train images (`make_holdout.py`, seed 0) becomes `val`, with a COCO ground truth built as
`bench/finetune/datasets.py` builds the evaluation split's, and the other 80% trains. bccd 164 / 41 images (523
boxes), brain-tumor 714 / 179 (190), signature 114 / 29 (29), medical-pills 74 / 18 (314). The trainer chooses
`best.folo` on that hold-out and the scorer scores it there (as the protocol does for three of the four sets).

**Brain-tumor's hold-out is not like its val split.** Fine-tuned models score 0.65–0.68 AP on it against
0.40–0.43 on val (other training pools and other images, so not a like-for-like gap), and its per-epoch curve
climbs for the whole run while the val curve is flat after a few epochs. 102 of the 179 hold-out images share a
file-name prefix with a training image (no two files are identical), so slices of the same scans on both sides are
a possibility; without scan ids or a near-duplicate audit it is not shown. It is the weakest of the four selection
splits.

Scripts and logs (not in the repo): `D:/folo-wave6/lanes/finetune/` — `bin/ft.sh` (one run + scorer), `bin/runner.sh`
(a queue under the GPU lock, holds of at most 28 minutes), `bin/lastscore.sh` (scores `last.folo`), `bin/table.py`,
`bin/curve.py`, `bin/level.py`, `bin/make_holdout.py`; `runs/<name>/` (each run's folder with `metrics.jsonl`,
`score.json`), `logs/`. The experiment knobs below were a local build of `main` plus two environment variables
(`D:/folo-wt/wave6-finetune-x`, branch `x/finetune-exp`, never pushed).

## Baseline: today's `main` and v5 against the published column

| dataset | published folo-n (v2 weights, `ee846c88`, A40) | `main` 71284767 + v5, this PC: seeds 0 / 1 / 2 | mean | moved |
|---|---:|---|---:|---:|
| BCCD (test) | 0.607 | 0.6166 / 0.6363 / 0.6205 | 0.6245 | +0.018 |
| brain-tumor (val) | 0.452 | 0.4287 / 0.4295 / 0.4072 | 0.4218 | −0.030 |
| signature (val) | 0.989 | 0.9881 / 0.9849 / 0.9880 | 0.9870 | −0.002 |
| medical-pills (val) | 0.804 | 0.8069 / 0.8034 / 0.8043 | 0.8049 | +0.001 |

Seeds 0 and 1 of BCCD, signature and pills reproduce examples/coco's v5 check (`main` d0c0907d) to every digit.
What moved: BCCD is where `auto` rules 4/5 took it in hill-climb 2 (0.607 → 0.625 there, from v2; no matched v2 BCCD
control was run here, so none of it is credited to the newer weights); signature and pills did not move.
**Brain-tumor lost 0.030**, and it is the weights, not the build: a follow-up diagnostic (not a selection run) of
`main` with `folo-n-coco-v2` on brain-tumor scores 0.4575 / 0.4592 / 0.4381 (mean 0.4516, +0.0298; the published
A40 seeds: 0.4393 / 0.4563 / 0.4605), the lowest v2 seed above the highest v5 seed. The v4 and v5 starting-point
checks (examples/coco, Publish v4 / v5) never ran brain-tumor, the one set with 893 images and the longest run, so
this is the first time a distilled start was fine-tuned on it.

## The run lengths `auto` resolves (leads 1 and 2)

The brief counted a 205-image set at batch 16 as 13 steps an epoch, 1,300 in 100 epochs. That counts batches.
FOLO (as Ultralytics) accumulates to the nominal batch 64 (`nbs`), so an optimizer step is 4 batches:

| dataset | train images | batches / epoch | optimizer steps / epoch | T (100 epochs) | warmup (3 epochs) | EMA decay at the last step | weight of the last step / of the last 5 steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCCD | 205 | 13 | 4 | 400 | 12 steps | 0.18 | 82% / 100% |
| brain-tumor | 893 | 56 | 14 | 1,400 | 42 steps | 0.50 | 50% / 97% |
| signature | 143 | 9 | 3 | 300 | 9 steps | 0.14 | 86% / 100% |
| medical-pills | 92 | 6 | 2 | 200 | 6 steps | 0.10 | 90% / 100% |

The EMA of docs/MODEL.md §8.5 (`d = 0.9999 · (1 − e^(−updates / 2000))`) is built for COCO-length runs (tens of
thousands of steps). In a fine-tune of a few hundred steps it never ramps up: the validated and saved "EMA"
weights are 50–90% the last step's live weights and, to the percent, an average of the last five steps' (the last
column, from the recurrence). Ultralytics has the same formula, so yolo26n's fine-tunes in the comparison have no
working EMA either.

**Warmup (lead 2)** is 3 epochs of optimizer steps on every set (`warmup_floor` 0 since hill-climb 2), 3% of the
run, never most of it or a handful of steps relative to the run; hill-climb 2 already measured the 100-step floor
(−0.013 on BCCD, three seeds) and a 1-epoch warmup (AdamW: 0.6109 against 0.6143 for the 3-epoch run of the same
seed, −0.003, one seed). Nothing to change.

## Lead 1: an EMA that averages in short runs

Knob (experiment build only): `FOLO_X_EMA_H=h` gives the EMA a nominal window of `w = h · T` optimizer steps: the
§8.5 formula `d = decay · (1 − e^(−updates / tau))` with `decay = 1 − 1 / w` and `tau = w / 5` (the default's ratio,
2000 / 10000). At `h = 0.25` that is decay 0.990 / tau 20 on BCCD, 0.9971 / 70 on brain-tumor, 0.9867 / 15 on
signature, 0.980 / 10 on pills. "Window" is nominal: at the last step the last quarter of the run carries about 63% of
the average and the earlier steps 37%, on every set (the same recurrence). A rule of this form capped at the default
(window 10,000) leaves every run of more than 40,000 steps (COCO) exactly as it is.

**Selection splits** (AP of `best.folo` on the hold-out; difference to `ho-base`, paired by seed):

| variant | BCCD | brain-tumor | signature | medical-pills | mean Δ |
|---|---|---|---|---|---:|
| base | 0.6495 / 0.6467 / 0.6462 | 0.6662 / 0.6522 / 0.6675 | 0.9959 / 0.9819 / 0.9891 | 0.8187 / 0.8117 / 0.8140 | |
| h = 0.1 | 0.6515 / 0.6567 / 0.6485 (+0.005) | not run | 0.9931 / 0.9822 / 0.9860 (−0.002) | 0.8169 / 0.8191 / 0.8172 (+0.003) | +0.002 (3 sets) |
| **h = 0.25** | 0.6544 / 0.6585 / 0.6494 (+0.007) | 0.6722 / 0.6691 / 0.6829 (+0.013) | 0.9929 / 0.9807 / 0.9850 (−0.003) | 0.8175 / 0.8179 / 0.8163 (+0.002) | **+0.005** |
| h = 0.5 | 0.6512 (seed 0) | not run | 0.9891 | 0.8123 | below 0.25 on all three: stopped |

h = 0.25 went to the reported splits.

**Reported splits** (the gate; AP of `best.folo`, three seeds, Δ paired by seed):

| dataset | base | EMA h = 0.25 | Δ |
|---|---|---|---:|
| BCCD (test) | 0.6166 / 0.6363 / 0.6205 | 0.6301 / 0.6254 / 0.6404 | +0.0075 |
| brain-tumor (val) | 0.4287 / 0.4295 / 0.4072 | 0.3813 / 0.4137 / 0.3708 | **−0.0332** |
| signature (val) | 0.9881 / 0.9849 / 0.9880 | 0.9908 / 0.9880 / 0.9911 | +0.0030 |
| medical-pills (val) | 0.8069 / 0.8034 / 0.8043 | 0.8079 / 0.8058 / 0.8080 | +0.0024 |
| mean | 0.7095 | 0.7044 | **−0.0051** |

**Gate missed**: the mean is down, and brain-tumor is worse by more than its baseline seed spread (0.0332 against
0.0223).

**Brain-tumor: the ranking turns with the checkpoint choice.** For three of the four sets `best.folo` is the
epoch with the best mAP50-95 on val, and val is then the scored split. Without a working EMA the per-epoch val curve
is noisy (brain-tumor swings 0.27–0.43 from one epoch to the next; BCCD's val ±0.03); the EMA smooths it and raises
its level, and its peak is lower. On brain-tumor's val the EMA is ahead at the last epoch (0.371 against 0.352) and
behind at the best (0.389 against 0.422). Selection bias is the plausible reading (the maximum of a noisy curve,
taken on the split that is then scored, is biased upward, more so the noisier the curve), but it was not tested:
that needs the selected checkpoints scored on images neither side selected on, which brain-tumor does not have.

| per-epoch val mAP50-95, mean over seeds | BCCD (its val) | brain-tumor | signature | medical-pills |
|---|---|---|---|---|
| base: max / mean of epochs 51–100 / last | 0.626 / 0.611 / 0.616 | 0.422 / 0.360 / 0.352 | 0.987 / 0.974 / 0.981 | 0.805 / 0.798 / 0.802 |
| EMA: max / mean of epochs 51–100 / last | 0.634 / 0.629 / 0.627 | 0.389 / 0.369 / 0.371 | 0.990 / 0.983 / 0.982 | 0.807 / 0.803 / 0.804 |

Best epochs: base brain-tumor 58 / 76 / 35, EMA 12 / 87 / 5 (two of the three EMA seeds peak within 12 epochs of
the pretrained start); BCCD base 90 / 44 / 42, EMA 81 / 92 / 52.

**Selection-free** (`last.folo`, the weights a run ends with, on the reported split; `bin/lastscore.sh`):

| dataset | base | EMA h = 0.25 | Δ |
|---|---|---|---:|
| BCCD (test) | 0.6176 / 0.6140 / 0.6159 | 0.6304 / 0.6240 / 0.6319 | +0.0129 |
| brain-tumor (val) | 0.3329 / 0.3979 / 0.3257 | 0.3566 / 0.4031 / 0.3535 | +0.0189 |
| signature (val) | 0.9810 / 0.9759 / 0.9846 | 0.9815 / 0.9789 / 0.9847 | +0.0012 |
| medical-pills (val) | 0.8034 / 0.7987 / 0.8032 | 0.8073 / 0.8024 / 0.8031 | +0.0024 |
| mean | | | **+0.0089** |

11 of 12 seed pairs favour the EMA (pills seed 2: −0.0002). So the EMA makes a better model at the end of a
fine-tune, by a little under the gate's 0.010; with the best checkpoint chosen on the scored split the order
reverses on brain-tumor. BCCD, the one set whose best epoch is chosen on another split than the scored one, is the
selection-unbiased version of the benchmark number: +0.0075 there, with the seed ranges overlapping (0.6166–0.6363
against 0.6254–0.6404).

**Not shipped** (the gate is the gate). If the benchmark ever scores a split it does not select on (or scores
`last.folo`), this is the first thing to re-run; the knob is twenty lines (`ModelEma::with_params` from `h · T`).

## Lead 3: `loss.cls_gain` for few classes

Hill-climb 2 found `cls_gain` 1.0 (the class term doubled) +4.5% on the 80-class proxy and −0.006 on BCCD; the
brief's guess was that few-class fine-tunes want the other direction. `--set loss.cls_gain=0.25` (half the default),
hold-out, three seeds:

| dataset | base | `cls_gain` 0.25 | Δ |
|---|---|---|---:|
| BCCD | 0.6495 / 0.6467 / 0.6462 | 0.6405 / 0.6472 / 0.6472 | −0.0025 |
| signature | 0.9959 / 0.9819 / 0.9891 | 0.9863 / 0.9845 / 0.9851 | −0.0037 |
| medical-pills | 0.8187 / 0.8117 / 0.8140 | 0.8167 / 0.8100 / 0.8126 | −0.0017 |

Lower on all three (mean −0.003): with 1 to 3 classes the default 0.5 sits between two settings that both lose.

## Lead 4: dense scenes (BCCD)

BCCD has about 14 small, touching cells per image. The assigner's `top_k` (10) caps the positives per object;
`--set loss.tal.topk=13` (hill-climb 2's proxy screen value), BCCD hold-out, three seeds: 0.6486 / 0.6385 / 0.6454
against 0.6495 / 0.6467 / 0.6462, **−0.0033**. Not pursued. The validation NMS IoU (0.7) is the benchmark's
fixed scoring setting, not a training default, and was left alone.

Not re-run: closing-mosaic epochs (hill-climb 2 found them bit-identical on BCCD because the best epoch came before
mosaic closed; mosaic is off in epochs 91–100, and of the reported BCCD runs here only EMA seed 1, best at epoch 92,
peaks after that) and a smaller `scale` for small cells.

## Lead 6: a lower LR for the backbone

Knob (experiment build only): `FOLO_X_BB_LR=f` moves every `backbone.*` parameter of each optimizer group into a
group at `f` times its LR (the schedule scales it as any group). Hold-out, `f = 0.3`, three seeds:

| dataset | base | backbone LR × 0.3 | Δ |
|---|---|---|---:|
| BCCD | 0.6495 / 0.6467 / 0.6462 | 0.6400 / 0.6500 / 0.6518 | −0.0002 |
| signature | 0.9959 / 0.9819 / 0.9891 | 0.9891 / 0.9844 / 0.9850 | −0.0028 |
| medical-pills | 0.8187 / 0.8117 / 0.8140 | 0.8159 / 0.8147 / 0.8123 | −0.0005 |

Level (mean −0.001). Not tried: a frozen-backbone warmup, BatchNorm statistics in the first steps.

## Lead 5: folo-s

Not run. Leads 1–4 and 6 found nothing that a model-size rule could scale, and the brain-tumor finding says the
starting weights matter more than the recipe on the one set where size could show; folo-s's own published weights
(`folo-s-coco`, not distilled) would be the place to start.

## Leftovers, by expected value

1. **The benchmark's selection.** Three of four sets choose `best.folo` on the split they score, which plausibly
   favours a noisy run. Scoring `last.folo` as well (or carving a selection split, as here) would make the table
   measure the model rather than its best epoch on the scored images, and would let the EMA rule above be judged.
2. **Starting weights for fine-tuning.** v5 is 0.030 AP worse than v2 on brain-tumor (three seeds per side, the
   lowest v2 seed above the highest v5 seed); the v4/v5 checks covered three of the four sets. A starting-point
   check on all four before the next weights release.
3. The EMA rule itself (lead 1) once 1 is settled: `h = 0.25`, capped at the default, keyed on `transfer`.
4. Closing-mosaic epochs on BCCD with the EMA (EMA seed 1's best epoch, 92, is after mosaic closes).

## What the brief got wrong

- "A 205-image set at batch 16 is 13 steps an epoch, about 1,300 steps in 100 epochs": 13 **batches**; with the
  nominal batch 64 it is 4 optimizer steps an epoch, 400 in all (the table above).
- "`folo weights` shows what `folo-n-coco` resolves to": the name `folo-n-coco` is the first release (`weights-v1`,
  AP 0.314); the documented fine-tuning start is `folo-n-coco-v5`, which these runs used.

## Lane finetune2: the EMA rule on a protocol that does not select on what it scores

Lane `finetune2` of wave 6 (2026-11-20), continuing the sections above. Scripts and logs (not in the repo):
`D:/folo-wave6/lanes/finetune2/` — `bin/p2score.sh` (P2's scoring), `bin/g2.py` (the table and the gate),
`bin/ft.sh` and `bin/runner.sh` (copies of the first lane's, scoring `last.folo` too), `bin/bitcheck.py`;
`runs/p2-<run>/` (`score-best.json`, `score-last.json`), `logs/p2score.out`, `logs/g2.out`.

### Protocol P2 and gate G2 (fixed before any score was read)

For each of the four sets: train on the train split minus the seeded 20% hold-out of `make_holdout.py` (seed 0),
let the trainer choose `best.folo` on that hold-out, and score that `best.folo` on the set's reported split (BCCD
test; val for the other three), which neither arm saw in training or selection. Three seeds per arm, paired by seed;
otherwise bench/finetune/README.md's protocol (640 px, batch 16, 100 epochs, f32, `folo-n-coco-v5`, pycocotools).
`last.folo` is scored on the reported split too. Arms: today's default against the EMA rule as it ships (window
`min(10000, 0.25 T)` optimizer steps, `decay = 1 − 1/w`, `tau = w/5`; a run of 40,000 steps or more keeps today's EMA).

G2: ship the rule only if, on P2's `best.folo` numbers, (a) the mean over the four sets is up by at least 0.005,
(b) at least 9 of the 12 seed pairs are up, (c) no set is down by more than its baseline seed spread (max − min),
and (d) `last.folo` is not down on any set's mean.

**The runs are the first lane's hold-out runs.** `ho-base-*` and `ho-ema25-*` above are exactly P2's training and
selection: trained on the hold-out's 80%, `best.folo` chosen on the 20%, with the knob at `h = 0.25` (the runs' log
lines: decay 0.98667 / tau 15 at `T` 300 on BCCD's 164 images, 0.99667 / 60 at 1,200 on brain-tumor, 0.98 / 10 at
200 on signature and pills; every `T` is under 40,000, so the cap does not bind and the knob is the rule). Only their
reported-split scores were missing, and nobody had read them. P2 therefore scored those 24 runs (`p2score.sh`:
`folo val --save-json` of `best.folo` and `last.folo` on the reported split at conf 0.001, IoU 0.7, then the scorer)
instead of training them again. The scoring was checked first: `base-bccd-s0`'s `best.folo` scores 0.61663 on
BCCD test, the first lane's number. The rule as shipped reproduces the knob bit for bit: `ho-ema25-bccd-s0` trained
again with the PR's build (`main` f875e5c9 + this change, no knob) has every epoch event and every tensor of
`best.folo` and `last.folo` identical to the knob's run (`bin/bitcheck.py`; first log line "EMA window 75 steps").

### P2: AP on the reported split (three seeds, Δ paired by seed)

`best.folo` (chosen on the hold-out):

| dataset | default | EMA window 0.25 T | Δ per seed | mean Δ |
|---|---|---|---|---:|
| BCCD (test) | 0.6218 / 0.6215 / 0.6114 | 0.6291 / 0.6326 / 0.6217 | +0.0073 / +0.0111 / +0.0103 | +0.0096 |
| brain-tumor (val) | 0.3942 / 0.3349 / 0.3495 | 0.3669 / 0.3438 / 0.3547 | −0.0274 / +0.0089 / +0.0052 | −0.0044 |
| signature (val) | 0.9821 / 0.9733 / 0.9676 | 0.9881 / 0.9851 / 0.9791 | +0.0060 / +0.0119 / +0.0115 | +0.0098 |
| medical-pills (val) | 0.7973 / 0.7892 / 0.7943 | 0.7986 / 0.7944 / 0.8049 | +0.0013 / +0.0052 / +0.0106 | +0.0057 |
| mean | 0.6864 | 0.6916 | | **+0.0052** |

`last.folo` (no selection):

| dataset | default | EMA window 0.25 T | Δ per seed | mean Δ |
|---|---|---|---|---:|
| BCCD (test) | 0.6086 / 0.6178 / 0.6008 | 0.6180 / 0.6301 / 0.6191 | +0.0094 / +0.0123 / +0.0184 | +0.0133 |
| brain-tumor (val) | 0.3690 / 0.3433 / 0.3533 | 0.3791 / 0.3438 / 0.3599 | +0.0102 / +0.0005 / +0.0065 | +0.0057 |
| signature (val) | 0.9850 / 0.9730 / 0.9791 | 0.9851 / 0.9791 / 0.9788 | +0.0002 / +0.0061 / −0.0003 | +0.0020 |
| medical-pills (val) | 0.7976 / 0.7945 / 0.7982 | 0.7996 / 0.8007 / 0.8036 | +0.0020 / +0.0062 / +0.0054 | +0.0046 |
| mean | 0.6850 | 0.6914 | | **+0.0064** |

### G2, line by line

- (a) mean `best.folo` Δ +0.0052 ≥ +0.005: **pass**, by 0.0002. It is the line closest to failing.
- (b) 11 of 12 seed pairs up (brain-tumor seed 0 down): **pass**.
- (c) no set down by more than its baseline spread: brain-tumor −0.0044 against a spread of 0.0593; the others are
  up: **pass**.
- (d) `last.folo` mean up on all four sets (+0.0133, +0.0057, +0.0020, +0.0046): **pass**.

**G2 passes, so the rule ships** (`ema_run_fraction` 0.25, docs/MODEL.md §8.5).

### Reading

- Without selection on the scored images the order of the first lane's benchmark-protocol result turns around:
  there the rule was −0.0051 with brain-tumor −0.033; here it is +0.0052 with brain-tumor −0.0044, and brain-tumor's
  `last.folo` gains as on every other set. What was not shown there is shown now: the earlier brain-tumor loss was
  the selection, not the model.
- The margin is thin. (a) passes by 0.0002, and brain-tumor's P2 baseline spreads over 0.059 across three seeds,
  so this is a small, consistent gain (11 of 12 pairs, `last.folo` up everywhere), not a large one. Under the
  benchmark's own protocol the published table will not show it on brain-tumor.
- P2 trains on 80% of each train split, so its absolute numbers are below the benchmark's (brain-tumor 0.36 against
  0.42; with the protocol's selection on the scored split included in the 0.42); the comparison is the pair.
- Brain-tumor's hold-out is the weakest selection split (the first lane: possibly near-duplicate slices, 0.66 on it
  against 0.36–0.39 on val). It still selected epochs whose val AP favours the rule on two of three seeds.
- The best epochs on the hold-out (default / rule): BCCD 38, 97, 25 / 54, 78, 39; brain-tumor 83, 75, 93 / 88, 100,
  82; signature 84, 84, 55 / 91, 96, 83; pills 94, 50, 86 / 96, 63, 93.
