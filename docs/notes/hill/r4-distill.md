# Hill-climb 4: dense output distillation (lab notebook)

Lane `distill` of hill-climb 4 (2026-11-14). Goal: folo-n learns from a frozen folo-s teacher on the same augmented
batches, through its raw head outputs, without changing inference, export or the `.folo` format. The method is
docs/MODEL.md §8.9 (`distill`, opt-in); branch `vision/distill`. This notebook has the measurements that motivated
it, the proxy runs that chose its configuration, and what is not shown yet.

## Why: what a folo-s teacher can give folo-n (CPU, hill-climb 3's dumps)

Both measurements were run by the counsels of this round on hill-climb 3's candidate dumps of val2017 at 640 px
(`D:/folo-data/runs-hill3/fusion/dumps/land-val` = folo-n-coco-v3, COCO AP 0.340, and `.../s-val` = folo-s-coco,
0.416), default post-processing, pycocotools. Scripts: [`bench/distill/`](../../../bench/distill/README.md).

**Swap at shared anchors** ([`swap.py`](../../../bench/distill/swap.py); log `D:/folo-build/hill4/counsel-a-swap.log`).
At anchors that are candidates (class-max score > 0.001) in both dumps (89% of the student's pair-score mass, 95% of
the teacher's), one model's boxes or scores are replaced by the other's:

| detections | AP | AP50 | AP75 | AP_S / M / L |
|---|---:|---:|---:|---|
| folo-n (student) | 0.3396 | 0.4874 | 0.3671 | 0.163 / 0.368 / 0.494 |
| student scores, teacher boxes | 0.3553 (+0.016) | 0.4886 | 0.3870 | 0.173 / 0.388 / 0.507 |
| student scores, halfway boxes | 0.3505 (+0.011) | 0.4915 | 0.3801 | 0.173 / 0.380 / 0.504 |
| teacher scores, student boxes | 0.3706 (+0.031) | 0.5335 | 0.3995 | 0.195 / 0.414 / 0.533 |
| folo-s (teacher) | 0.4162 | 0.5798 | 0.4519 | 0.229 / 0.459 / 0.585 |

**Swap of each truth's primary detection** ([`nvs.py`](../../../bench/distill/nvs.py); log
`D:/folo-build/hill4/counsel-b/nvs.log`): per non-crowd truth, each model's top-scoring same-class detection at
IoU >= 0.5; on the truths both models find, folo-s's box or score is put on folo-n's primary.

| configuration | AP | AP50 | AP75 |
|---|---:|---:|---:|
| folo-n | 0.3396 | 0.4874 | 0.3671 |
| folo-n with folo-s's primary boxes | 0.3546 (+0.015) | 0.4874 | 0.3851 |
| folo-n with folo-s's primary scores | 0.3655 (+0.026) | 0.5378 | 0.3896 |

On the same objects folo-s's median edge error is 17-19% lower at every size (S 1.87 / 1.56 px, M 3.30 / 2.67,
L 4.81 / 3.94; n / s). Both tables say the same: the teacher's ranking is worth about twice its boxes
(+0.026..+0.031 against +0.015..+0.016), the boxes show in AP75 only, the scores in AP50 and AP75. Swapping in the
teacher's boxes at shared anchors (the student's scores and candidates kept) gave +0.016; whether edge-only
(localisation) distillation in training can exceed that is untested.

**The DFL distributions carry no uncertainty.** Hill 3's refiner lane (r3-refine.md, Part 5) measured median DFL
widths of 3.45 / 5.97 / 12.52 px for S / M / L primaries, about 0.43 / 0.37 / 0.39 bins at strides 8 / 16 / 32. A
two-bin DFL target with a uniform sub-bin phase has a mean width of pi/8 = 0.393 bins: the measured widths are the
encoding floor of a certain edge. At T = 1 a single teacher's distribution is therefore little more than its box.
That is what a mirrored second teacher view (two views averaged in probability space) was meant to add: where it
disagrees about an edge the averaged target is wider. It measured as useless on the proxy and was removed (below).

## Protocol

**Proxy P1** as hill-climb 2 ([r2-learn.md](r2-learn.md), Protocol): `D:/folo-data/runs-hill/learn/mini320` (20,000
train2017 / 2,000 val2017 images at a 320 px long side, 80 classes), folo-n from scratch, 320 px, 20 epochs, real
batch 64, close-mosaic 5, AMP, `--patience 1000`, the defaults (`auto` rule 5); score: pycocotools-exact COCO AP of
`best.folo` on the 2,000 proxy val images (`folo val --coco`). Teacher: `folo-s-coco` (a copy in
`D:/folo-data/runs-hill4/teachers/`), proxy val AP at 320 px 0.3220. Harness `D:/folo-data/runs-hill4/proxy/bin/`
(`p1.sh <binary> <name> <seed> [flags]`, `queue.sh`, `table.py`), every run one hold of the shared GPU lock; queue
file of this lane `D:/folo-data/runs-hill4/proxy/queue-distill.txt`, logs `proxy/logs/<name>.log`, results
`proxy/results/<name>.json`, runs `proxy/runs/<name>/`.

Binaries: `folo-main.exe` = origin/main 1436ad68 (lane `base`'s controls), `folo-kd.exe` = `vision/distill`.
Distillation arms add `--teacher D:/folo-data/runs-hill4/teachers/folo-s-coco.folo` and the `--set` shown.

Commands. One arm is one queue line `p1 <binary> <name> <seed> [flags]`, which runs
```
folo train --data D:/folo-data/runs-hill/learn/mini320 --format coco --model folo-n --imgsz 320 --batch 64 \
  --epochs 20 --close-mosaic 5 --amp --workers 10 --device gpu:0 --seed <seed> --patience 1000 \
  --project D:/folo-data/runs-hill4/proxy/runs --name <name> [--teacher D:/folo-data/runs-hill4/teachers/folo-s-coco.folo --set ...]
folo val --model <run>/weights/best.folo --data D:/folo-data/runs-hill/learn/mini320 --imgsz 320 --coco \
  --device gpu:0 --workers 10 --json
```
inside one hold of the GPU lock; the queue runner is `LOCK_OWNER=distill bash proxy/bin/queue.sh
proxy/queue-distill.txt`, the tables `python proxy/bin/table.py --runs` (`proxy/logs/table-distill.md`), the cost
lines `python D:/folo-build/hill4/distill-cost.py <runs>` (epoch seconds and img/s from each run's log).

Fair controls. The class term shares the supervised normaliser, so supervised BCE at gain g plus `L_kc` at
`lambda_c` is exactly BCE against a blended target at gain g + lambda_c (MODEL.md §8.9). The starting values
(`loss.cls_gain` 0.5 + `distill.cls_gain` 0.5) are therefore compared against `cls1` (`--set loss.cls_gain=1.0`, no
teacher), `blend` (0.25 + 0.25) against `ctl` (0.5, no teacher).

## Runs

**Opt-in check.** `off-s0` (`folo-kd.exe`, no teacher, seed 0) against lane `base`'s `ctl-s0` (`folo-main.exe`):
all 6,260 step losses and the 20 epoch lines (train loss, internal validation) identical to the last printed digit,
the same `best.folo` AP 0.12551666, `config.json` different only in the run name
(`D:/folo-data/runs-hill4/proxy/logs/cmp-ctl-off-s0.log`, script `D:/folo-build/hill4/distill-cmp.py`). With the
teacher off the branch trains bit-identically to origin/main.

The proxy arms all ran one binary, `folo-kd.exe` at `dc727599` (before the mirrored view was removed in
`751d6d84`). That the removal changes nothing for one view was checked on CPU, not on the proxy: both binaries
trained folo-n for 2 epochs (16 batches of 8, 2 optimizer steps) on 64 mini320 images at 160 px with the teacher at
T = 4 and seed 0. Every printed loss and distillation line is the same, `metrics.jsonl` differs only in timings and
the teacher log line (which loses its `consensus none`), and the tensor payload of `last.folo` is byte-identical
(the last 42,045,980 bytes; only the safetensors header with the stored config differs). Script
`D:/folo-build/hill4/equiv.sh`, output `D:/folo-build/hill4/equiv/` (`cmp.out`). This does not cover the GPU / f16
path.

Every run of the lane and lane `base`'s controls, P1, proxy-val COCO AP of `best.folo` (best epoch 20 in every run):

| run | flags (after `--teacher …` for distillation arms) | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---:|---:|---:|---:|---:|---:|
| ctl-s0 / s1 / s2 | none, folo-main (base) | 0.1255 / 0.1238 / 0.1249 | 0.204 | 0.130 | 0.062 | 0.190 | 0.281 |
| cls1-s0 / s1 / s2 | `loss.cls_gain=1.0`, folo-main (base) | 0.1267 / 0.1280 / 0.1296 | 0.213 | 0.131 | 0.061 | 0.195 | 0.288 |
| off-s0 | none, folo-kd | 0.1255 | 0.206 | 0.130 | 0.062 | 0.190 | 0.289 |
| kd-s0 | starting values (cls 0.5, dfl 1.5, T 1, one view) | 0.1536 | 0.248 | 0.160 | 0.071 | 0.243 | 0.349 |
| kc-s0 | `distill.dfl_gain=0` | 0.1493 | 0.238 | 0.158 | 0.069 | 0.230 | 0.322 |
| kdl-s0 | `distill.cls_gain=0` | 0.1454 | 0.232 | 0.153 | 0.072 | 0.220 | 0.310 |
| flip-s0 | `distill.consensus=flip` (removed since) | 0.1533 | 0.248 | 0.158 | 0.070 | 0.236 | 0.338 |
| t4-s0 | `distill.temperature=4` | 0.1565 | 0.248 | 0.164 | 0.071 | 0.245 | 0.341 |
| t4-s1 | `distill.temperature=4`, seed 1 | 0.1548 | 0.247 | 0.162 | 0.074 | 0.241 | 0.347 |
| t4-s2 | `distill.temperature=4`, seed 2 | 0.1538 | 0.248 | 0.158 | 0.072 | 0.235 | 0.353 |
| blend-s0 | `loss.cls_gain=0.25`, `distill.cls_gain=0.25` | 0.1478 | 0.234 | 0.158 | 0.072 | 0.225 | 0.333 |

(the `ctl` and `cls1` rows give AP per seed and the 3-seed means of the other columns.)

Seed-0 reading so far. The starting values clear the kill criterion (0.1297) by 0.024. Against its fair control
(`cls1-s0`, the same total class gain 1.0) `kd-s0` is +0.027; the class term alone (`kc-s0`, also total gain 1.0) is
+0.023 over `cls1-s0`; the DFL term alone (`kdl-s0`, total class gain 0.5) is +0.020 over `ctl-s0`, more than the
+0.016 the box swap measured on the frozen full-scale models (the proxy student is far weaker than the teacher, see "What is not
shown"). Both terms add: kd is +0.004 over class-only and +0.008 over DFL-only. The design predicted the class
term in AP50 and the DFL term in AP75; on the proxy neither separates: `kc-s0` over `cls1-s0` is AP50 +0.027 /
AP75 +0.029, `kdl-s0` over `ctl-s0` AP50 +0.026 / AP75 +0.023. At this distance from the teacher both terms
improve the student everywhere (the prediction came from a 0.34 student next to a 0.42 teacher at full scale).

Internal mAP50-95 (the trainer's own validation) at epochs 2 / 5 / 10 / 15 / 20: ctl-s0 0.0049 / 0.0397 / 0.0768 /
0.1049 / 0.1250; kd-s0 0.0134 / 0.0590 / 0.1041 / 0.1345 / 0.1529; kc-s0 0.0095 / 0.0515 / 0.1007 / 0.1289 /
0.1489; kdl-s0 0.0076 / 0.0438 / 0.0940 / 0.1232 / 0.1446. The distillation terms (weighted, per batch, without
B) of kd-s0 fall from cls 5.91 / dfl 3.75 in epoch 1 to 1.32 / 0.49 in epoch 5 and 0.73 / 0.32 in epoch 20, against
supervised items of box 3.8 / cls 5.0 / dfl 3.6 in epoch 1.

**Negative result: the flip consensus.** `flip-s0` (the teacher also run on the batch mirrored along W, the outputs
mapped back exactly and both views averaged in probability space) gave AP 0.1533 against 0.1536 for one view
(`kd-s0`): below the +0.003 bar the protocol set for keeping it, so seeds 1 and 2 were not run and the option was
removed from the branch (code, tests, docs). Its averaged targets are softer, as intended (epoch-20 terms cls 0.668
/ dfl 0.290 against 0.726 / 0.315 with one view), but the student does not profit; AP_M and AP_L are lower (0.236 /
0.338 against 0.243 / 0.349), AP_S equal. It cost a second teacher forward: 43.8 s per epoch against 38.5 s
(469 against 528 img/s at epoch end, both with the COCO control on the same GPU). Horizontal-flip augmentation
(`fliplr` 0.5) already shows the student both orientations, which may be why the mirrored view adds no information.

**T = 4.** `t4-s0` gave 0.1565 against 0.1536 for `kd-s0` (T = 1): +0.0029, above the 0.002 seed noise, all of it
in AP75 (0.164 against 0.160; AP50 equal, AP_L 0.341 against 0.349). The temperature acts on the DFL term only (the
class term has none); at T = 4 the softened teacher distributions spread over more bins than the two-bin floor and
the T² factor makes the term about 2.6 times larger (epoch 20: dfl 0.811 against 0.315). Internal mAP50-95 e10 / e20
0.1115 / 0.1559 against 0.1041 / 0.1529. Seeds 1 and 2 gave 0.1548 and 0.1538, closer to `kd-s0`: the 3-seed mean
of `t4` (0.1550) is only +0.0014 over the single `kd-s0`, so T = 4 over T = 1 is a seed-0 choice, not a shown gain
(`kd` has one seed; its seed 1 was stopped at its start to give `t4`'s seeds the GPU, seed 2 was dropped).

**Blend.** `blend-s0` keeps the total class gain at 0.5 and only moves the class target halfway to the teacher's
scores (`loss.cls_gain=0.25` + `distill.cls_gain=0.25`, DFL term at 1.5, T = 1): 0.1478, +0.022 over `ctl-s0`
(its fair control, same total class gain) and +0.002 over `kdl-s0` (the DFL term alone at class gain 0.5, the
label target untouched). Against `kd-s0` (0.1536, total class gain 1.0) it is -0.006: on the proxy the class term
works more by adding gain on the teacher's target than by replacing part of the label target. Not chosen.

**Choice.** `t4` (`--teacher` with T = 4, class gain 0.5, DFL gain 1.5, one view): the best seed-0 arm by more
than the 0.002 noise over `kd-s0`, and no simpler arm comes within 0.002 of it (`kc` -0.007, `kdl` -0.011, `kd`
-0.003). Both terms stay (each alone is clearly worse); the temperature stays a tunable key; the mirrored view is
removed. Since the gate passed, `--teacher` alone now starts at these values (T = 4 in `DistillConfig::default`,
docs/MODEL.md §8.9, docs/CLI.md). After the proxy, for a `head = "o2o"` student the term became part of the
one-to-many loss, weighted by the dual loss's `w_m` (§6.6, §8.9); every arm here is an `o2m` student, which that
change leaves bit-identical, and no `o2o` student was measured with a teacher.

**Gate** (3 seeds per side, proxy-val COCO AP of `best.folo`; table `proxy/logs/table-distill.md` from
`proxy/bin/table.py --runs`):

| arm | seeds | AP mean | AP min-max | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t4 (chosen) | 0 1 2 | 0.1550 | 0.1538-0.1565 | 0.248 | 0.161 | 0.072 | 0.240 | 0.347 |
| cls1 (matching control, total class gain 1.0) | 0 1 2 | 0.1281 | 0.1267-0.1296 | 0.213 | 0.131 | 0.061 | 0.195 | 0.288 |
| ctl (defaults) | 0 1 2 | 0.1247 | 0.1238-0.1255 | 0.204 | 0.130 | 0.062 | 0.190 | 0.281 |

`t4` is +0.0269 AP over its matching control `cls1` (bar +0.010) and +0.0303 over `ctl`; its lowest seed (0.1538)
is above the highest control seed (0.1296): **the gate passes** on the proxy. The gain is spread over every column
(AP50 +0.035, AP75 +0.030, AP_S +0.011, AP_M +0.045, AP_L +0.059 over `cls1`): the AP50-for-class, AP75-for-DFL split
the design predicted does not show at this distance from the teacher.

## What is not shown

- **The proxy flatters distillation.** P1 is a 20-epoch, 320 px folo-n trained from scratch on 20,000 images next to
  a teacher trained at 640 px on all of train2017, which includes these 20,000 images: the
  teacher has seen every proxy training image with its labels, and the student is far from converged (proxy val
  AP 0.125 against the teacher's 0.322 at the same 320 px). A full-scale student (folo-n-coco-v3: 0.340 against
  folo-s's 0.416 on val2017) is much closer to its teacher; on the frozen models the counsel swaps measured +0.031
  (scores) and +0.016 (boxes), which do not bound what training can give. The full-scale test (the COCO control's twin with `--teacher`) is pending.
- No COCO val2017 number: val2017 is read once per final configuration, and no final configuration has been
  trained at full scale.
- The single-seed arms (`kc`, `kdl`, `flip`, `t4`, `blend`) differ from `kd-s0` by less than the gaps that matter
  only where stated; the seed noise of P1 is about 0.002 AP (the `ctl` and `cls1` ranges).
- Speed is measured with the full-COCO control training on the same GPU (lane `base`'s run, hours long); the
  teacher's cost alone, unloaded, is not measured.

## Leftovers: the full-scale test (proposed, not run)

Smaller leftovers: `kd` (T = 1) has one seed, so T = 4 over T = 1 is not shown beyond seed 0; doubled gains
(the optional arm) were not tried; the cost split between the teacher forward and the loss path is not profiled; no
unloaded speed A/B.

The control is lane `base`'s full-COCO run `D:/folo-data/runs-hill4/coco/n-r5-ctl` (folo-n from scratch, 640 px,
64 epochs, batch 16 x accumulate 4, AMP, seed 0, origin/main 1436ad68). Two forms of its distillation twin:

- **(A) the fork.** A watcher copies the control's epoch-48 checkpoint to `D:/folo-data/runs-hill4/coco/n-r5-ctl-e48/`.
  Copy that folder (the copy is the fork, the snapshot stays untouched) and resume it with the teacher for the last
  16 epochs (the close-mosaic phase starts at epoch 55):
  ```
  folo train --resume <copy>/weights/last.folo --teacher D:/folo-data/runs-hill4/teachers/folo-s-coco.folo \
    --data D:/folo-data/coco --workers 16 --device gpu:0
  ```
  The control continuing past epoch 48 without a teacher is its exact twin (same weights, same batches), so the
  difference is distillation over the last quarter of the schedule only; how it relates to a whole run is untested.
  It is available a few hours after the control ends.
- **(B) a whole run.** The control's command with `--teacher ... --name n-r5-kd`: the same seed and batches from
  epoch 1, the clean comparison, about four times the wall time of (A).

Cost projection. The proxy's teacher forward took an epoch from 14 s to 38 s (1,898 to 528 img/s at epoch end,
`off-s0` against `kd-s0`, both with the COCO control on the same GPU, so the ratio is uncertain); the folo-s
forward at 640 px costs about as much per pixel. The control runs about 270 s per epoch alone, so a distillation
epoch at full scale is projected at 540-730 s alone (2-2.7x): (A) 16 epochs in about 2.5-3.5 h, (B) 64 epochs in
about 10-13 h, each with the GPU to itself. Both read COCO val2017 once at the end (`folo val --coco` of
`best.folo`, against the control's same read).

## Cost

From each run's own log (`D:/folo-build/hill4/distill-cost.py`, log `proxy/logs/cost-distill.log`): mean epoch wall
time over the 20 epochs and the median img/s at the end of an epoch. Load: every run below shared the GPU with the
full-COCO control (lane `base`, folo-n at 640 px, batch 16); the controls of lane `base` ran under a different
share of it, so only the runs of this lane are compared with each other.

| run | teacher | epoch s (min-max) | img/s at epoch end |
|---|---|---:|---:|
| off-s0 | none | 14.1 (10.3-25.5) | 1,898 |
| kd-s0 | folo-s, f16, one view | 38.5 (36.1-42.1) | 528 |
| kc-s0 | one view, DFL term off | 38.0 (29.0-41.5) | 535 |
| kdl-s0 | one view, class term off | 35.7 (34.2-38.9) | 587 |
| flip-s0 | two views (removed) | 43.8 (43.0-45.3) | 469 |

The teacher makes a proxy epoch about 2.7 times as long under this load. The mirrored second view (one more
teacher forward, mapping and averaging) added only 5.3 s, so the teacher's forward itself is a small part of the
24 s; the rest sits in building the targets and the two loss terms with their backward over `[B, 144, A]` f32
tensors (not profiled). Turning one term off saves little (kc, kdl). GPU memory: the whole board (WDDM reports no
per-process numbers) read 13.5-13.7 GiB during `kc-s0` plus the COCO control; the teacher's own share is not
measured. No unloaded A/B was taken (the COCO control occupied the GPU for this lane's whole time).

From about 17:30 a game ran on the same GPU as well (`nvidia-smi`): `t4-s0` took 33-903 s per epoch, so its wall
time is not a cost number, and runs after it are slower several-fold; accuracy does not depend on load. `t4-s1`
(104.7 s per epoch, 239 img/s) and `t4-s2` (66.4 s, 393 img/s) and `blend-s0` (57.4 s, 416 img/s, `proxy/logs/cost-blend.log`) ran under the same mixed load plus other agents'
builds and test suites; T = 4 itself adds only a division before the softmax (not measured
separately).
