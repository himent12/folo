# Hill-climb 2: learning (lab notebook)

Lane `learn` of hill-climb 2 (2026-11-11). Goal: FOLO learns more per epoch than main's recipe and the Ultralytics
reference, on the round-1 COCO proxy and on the fine-tuning benchmark (folo-n behind yolo26n on BCCD). Round 1:
[learn.md](learn.md) (protocol, reference diff, `auto` rules 2 and 3, MuSGD).

## Protocol

**Proxy P1** exactly as round 1 (learn.md, Protocol): `D:/folo-data/runs-hill/learn/mini320` (20,000 train2017 /
2,000 val2017 images at a 320 px long side, all 80 classes), folo-n from scratch, 320 px, 20 epochs, real batch 64,
close-mosaic 5, AMP, `--patience 1000`; score: pycocotools-exact COCO AP of `best.folo` on the 2,000 proxy val
images (`folo val --coco`). 313 optimizer steps per epoch, 6,260 in all.

**Fine-tune** as the benchmark (bench/finetune/README.md, Protocol) on this PC's GPU instead of RunPod: folo-n from
`folo-n-coco-v2`, 640 px, batch 16, 100 epochs, f32, everything else at its default; `folo val --save-json` of
`best.folo` (FOLO's best val mAP50-95 epoch) on the evaluation split (BCCD: test; the others: val) at conf 0.001,
IoU 0.7, scored by `bench/finetune/scorer.py` (pycocotools). The main binary reproduces the published A40 result of
BCCD seed 0 exactly (0.6062), so the published main and yolo26n rows (docs/BENCHMARKS.md) are the baselines.

**Shapes gate**: docs/MODEL.md §10.6 on the GPU (folo-n, 320 px, 30 epochs, batch 16, seed 0; pass: mAP50 ≥ 0.90,
mAP50-95 ≥ 0.75).

Wrappers in `D:/folo-data/runs-hill2/learn/bin`: `p1.sh` (proxy), `p1o.sh` (proxy with `--head o2o`, both branches
scored), `ft.sh` (fine-tune + scorer), `shapes.sh`, `runner.sh` (a queue file run under the shared GPU lock, several
runs per hold of at most 28 minutes), `ap.py` / `ftap.py` (tables). Binaries: `folo-main.exe` = `main` 647b981f;
`folo-r2a.exe` adds the opt-in keys `warmup_floor` and `musgd.*`; `folo-r2b.exe` adds `o2o_attached_epochs`;
`folo-r4.exe` is `auto` rule 4 with the floor default 0 (momentum 0.9 in its rule; the `r5` rows pass momentum
0.937 explicitly, which is what the shipped rule resolves to). Every run is one GPU hold of the lock
(`D:/folo-build/hill/lock.sh`), with the GPU otherwise shared by five lanes' short tests; wall times are
not speed measurements.

## Runs

### Proxy P1 (AP of best.folo on the 2,000 proxy val images)

| run | binary | seed | settings | AP | AP50 | AP_S | AP_M | AP_L |
|---|---|---:|---|---:|---:|---:|---:|---:|
| (round 1) base-adamw | | 0 | main's `auto` on P1: AdamW lr0 0.001 (rule 2/3) | 0.0747 | 0.125 | 0.034 | 0.118 | 0.176 |
| (round 1) ultra-musgd | Ultralytics 8.4.198 | 0 | `optimizer=MuSGD` (momentum 0.937, warmup_bias_lr 0.1) | 0.1133 | 0.188 | 0.060 | 0.172 | 0.244 |
| mus-r3-s0 | main | 0 | `--optimizer musgd`, momentum 0.9, warmup_bias_lr 0 (= `auto` rule 3's MuSGD) | 0.1027 | 0.173 | 0.050 | 0.154 | 0.227 |
| mus-r3-s1 | r2a | 1 | the same | 0.1010 | 0.168 | 0.050 | 0.153 | 0.225 |
| mus-r3-s2 | r2a | 2 | the same | 0.1027 | 0.169 | 0.054 | 0.156 | 0.217 |
| mus-937-s0 | main | 0 | momentum 0.937, warmup_bias_lr 0.1 (round 1's MuSGD rows) | 0.1133 | 0.187 | 0.057 | 0.174 | 0.239 |
| mus-m937w0-s0 | r2a | 0 | momentum 0.937, warmup_bias_lr 0 | 0.1048 | 0.175 | 0.050 | 0.158 | 0.224 |
| mus-m09w01-s0 | r2a | 0 | momentum 0.9, warmup_bias_lr 0.1 | 0.1139 | 0.188 | 0.060 | 0.171 | 0.265 |
| mus-m09w01-s1 | r2a | 1 | the same | 0.1091 | 0.180 | 0.055 | 0.167 | 0.236 |
| mus-m09w01-s2 | r2a | 2 | the same | 0.1123 | 0.185 | 0.056 | 0.171 | 0.245 |

**Bias warmup, three seeds per side:** MuSGD at momentum 0.9 with `warmup_bias_lr = 0` (what `auto` rule 3 resolves to)
0.1027 / 0.1010 / 0.1027, mean 0.1021; with 0.1: 0.1139 / 0.1091 / 0.1123, mean **0.1118**: +0.0097 AP (+9.5%),
every seed of one side above every seed of the other. Momentum 0.937 against 0.9: one seed per side at each bias
warmup (0.1048 vs 0.1027 at 0, 0.1133 vs 0.1139 at 0.1), not distinguishable.

### Proxy P1: the defaults, and screens on top of rule 4 (binary `folo-r4f` = PR #188, `auto` rule 4)

| run | seed | settings (beyond P1) | AP | AP50 | AP_S | AP_M | AP_L |
|---|---:|---|---:|---:|---:|---:|---:|
| auto-main-s0 | 0 | main 647b981f defaults: `auto` → AdamW lr0 0.001 | 0.0761 | 0.129 | 0.033 | 0.119 | 0.170 |
| auto-main-s1 | 1 | the same | 0.0751 | 0.129 | 0.036 | 0.119 | 0.165 |
| auto-main-s2 | 2 | the same | 0.0758 | 0.127 | 0.035 | 0.125 | 0.171 |
| mus-937-s0 | 0 | MuSGD momentum 0.937, warmup_bias_lr 0.1 (= rule 4's `auto` here) | 0.1133 | 0.187 | 0.057 | 0.174 | 0.239 |
| auto-r4-s1 | 1 | rule 4 defaults | 0.1129 | 0.186 | 0.055 | 0.171 | 0.252 |
| auto-r4-s2 | 2 | rule 4 defaults | 0.1143 | 0.187 | 0.064 | 0.171 | 0.250 |
| r4-w55-s0 | 0 | rule 4, `--set musgd.muon=0.5 --set musgd.sgd=0.5` (= rule 5 from scratch) | 0.1248 | 0.203 | 0.057 | 0.191 | 0.294 |
| r4-w55-s1 | 1 | the same | 0.1229 | 0.201 | 0.056 | 0.186 | 0.276 |
| r4-w55-s2 | 2 | the same | 0.1225 | 0.201 | 0.068 | 0.182 | 0.255 |
| r4-cls1-s0 | 0 | rule 4, `--set loss.cls_gain=1.0` | 0.1203 | 0.204 | 0.059 | 0.178 | 0.246 |
| r4-cls1-s1 | 1 | the same | 0.1149 | 0.194 | 0.055 | 0.174 | 0.256 |
| r4-cls1-s2 | 2 | the same | 0.1206 | 0.201 | 0.059 | 0.180 | 0.272 |
| r4-topk13-s0 | 0 | rule 4, `--set loss.tal.topk=13` | 0.1126 | 0.185 | 0.056 | 0.176 | 0.250 |
| r4-cos-s0 | 0 | rule 4, `--cos-lr` | 0.1129 | 0.188 | 0.064 | 0.162 | 0.239 |

Means over seeds 0–2: main's `auto` **0.0757**; rule 4 **0.1135**; rule 4 with MuSGD weights 0.5 / 0.5 (rule 5
from scratch) **0.1234**; rule 4 with `cls_gain` 1.0 0.1186. Both 0.5 / 0.5 and `cls_gain` 1.0 are above rule 4 on
every seed (0.5 / 0.5: +0.012, +0.010, +0.008; `cls_gain`: +0.007, +0.002, +0.006).

**o2o (`--head o2o`, `p1o.sh`, binary r2b; MuSGD momentum 0.937, warmup_bias_lr 0.1), seed 0**, AP of the one-to-one
head (NMS-free, what an o2o model predicts with) and of the same model's one-to-many branch (`--head o2m`, with NMS):

| run | settings | o2o AP | o2o AP50 | o2m AP | o2m AP50 |
|---|---|---:|---:|---:|---:|
| o2o-det-s0 | default: the o2o branch's inputs detached | 0.0952 | 0.158 | 0.1144 | 0.188 |
| o2o-att10-s0 | `o2o_attached_epochs=10` (attached for the first half) | 0.0966 | 0.161 | 0.1132 | 0.189 |
| o2o-att20-s0 | `o2o_attached_epochs=20` (attached throughout) | 0.1009 | 0.170 | 0.1093 | 0.185 |

Attaching the one-to-one branch throughout helps the NMS-free head (+0.006) and costs the one-to-many branch
(−0.005); for the first 10 epochs only, both are within a seed's noise of detached (+0.001, −0.001). One seed each,
not shipped as a default (it stays opt-in).

### Fine-tune on BCCD (test AP of best.folo, benchmark scorer)

| run | seeds | settings | test AP per seed | mean | AP50 mean |
|---|---|---|---|---:|---:|
| main (published, A40) | 0 1 2 | `auto` → AdamW lr0 0.001429, warmup 100 steps (25 of 100 epochs) | 0.6062 0.6074 0.6061 | 0.607 | 0.883 |
| ft-main | 0 | the same, this PC | 0.6062 | | 0.892 |
| ft-nbs16 | 0 | `nbs=16` (no accumulation: 13 steps per epoch, warmup 100 steps = 8 epochs) | 0.6069 | | 0.898 |
| ft-wf0 | 0 1 2 | `warmup_floor=0` (warmup 12 steps = 3 epochs, as Ultralytics 8.4) | 0.6143 0.6211 0.6225 | 0.619 | 0.897 |
| ft-wf0lr2 | 0 1 | floor 0, AdamW lr0 0.002 (round 1's AdamW floor 0.002) | 0.6049 0.6151 | 0.610 | 0.889 |
| ft-wf0we1 | 0 | floor 0, `warmup_epochs=1` | 0.6109 | | 0.890 |
| ft-musgd | 0 | `--optimizer musgd --lr0 0.01` (momentum 0.937, warmup_bias_lr 0.1), floor 100 | 0.6209 | | 0.901 |
| ft-wf0mus | 0 1 2 | MuSGD lr0 0.01, momentum 0.937, warmup_bias_lr 0.1, floor 0 | 0.6294 0.6223 0.6235 | **0.625** | 0.910 |
| ft-muslr005 | 0 | the same at lr0 0.005 | 0.6079 | | 0.893 |
| ft-muslr02 | 0 | the same at lr0 0.02 | 0.6074 | | 0.896 |
| ft-musy26w | 0 | the same with YOLO26's published MuSGD weights (muon 0.528, sgd 0.674, cls_lr 2.737) | 0.6085 | | 0.893 |
| ft-musm09 / ft-r4 | 0 1 2 | MuSGD lr0 0.01, momentum 0.9, warmup_bias_lr 0.1, floor 0 (`auto` rule 4 draft) | 0.6231 0.6158 0.6269 | 0.622 | 0.906 |
| yolo26n (published) | 0 1 2 | Ultralytics defaults from yolo26n.pt | 0.6212 0.6339 0.6235 | 0.626 | 0.911 |

**On top of rule 4 on BCCD** (three seeds each): MuSGD weights 0.5 / 0.5 0.6036 / 0.6254 / 0.6185 (mean 0.616,
−0.009 against rule 4's 0.6251); `cls_gain` 1.0 0.6167 / 0.6177 / 0.6233 (mean 0.619, −0.006). Both help from
scratch on the proxy and cost the fine-tune, so rule 5 uses 0.5 / 0.5 only from scratch and `cls_gain` stays 0.5.

### Fine-tune on the other three sets (rule 4's settings: MuSGD lr0 0.01, momentum 0.937, warmup_bias_lr 0.1, floor 0)

| dataset | main (published, seeds 0 1 2) | rule 4 (this PC, seeds 0 1 2) | mean main → rule 4 | yolo26n (published) |
|---|---|---|---|---:|
| medical-pills (val) | 0.8011 0.8049 0.8067 | 0.8080 0.8089 0.8052 | 0.8042 → **0.8074** | 0.8027 |
| signature (val) | 0.9921 0.9874 0.9881 | 0.9873 0.9909 0.9874 | 0.9892 → 0.9885 | 0.9818 |
| brain-tumor (val) | 0.4393 0.4563 0.4605 | 0.4649 0.4369 0.4470 | 0.4520 → 0.4496 | 0.4355 |

AP50: pills 0.997–0.998 for the rule-4 seeds (mean 0.998; main 0.995), signature 1.000 for every run of both (saturated, 35 images),
brain-tumor 0.619 (main 0.613). Pills a little better, signature and brain-tumor level (differences of 0.001 and
0.002, inside seed ranges of 0.004 and 0.03): no loss on the other three sets.

### Shapes gate (seed 0)

| run | settings | final mAP50 | final mAP50-95 | gate first met |
|---|---|---:|---:|---:|
| shapes-main | main: `auto` → AdamW lr0 0.00125, warmup 100 steps | 0.980 | 0.892 | epoch 9 |
| shapes-mus | MuSGD lr0 0.01, momentum 0.9, warmup_bias_lr 0.1, floor 0 | 0.982 | 0.917 | epoch 10 |
| shapes-r5 | MuSGD lr0 0.01, momentum 0.937, warmup_bias_lr 0.1, floor 0 (= rule 4) | 0.985 | 0.926 | epoch 8 |
| shapes-w55 | rule 4 with `musgd.muon=0.5`, `musgd.sgd=0.5` (= rule 5 from scratch) | 0.988 | **0.934** | epoch 8 |
| shapes-cls1 | rule 4 with `loss.cls_gain=1.0` | 0.987 | 0.924 | epoch 9 |

No run had a non-finite batch.

## Commands

Every run is one line of `D:/folo-data/runs-hill2/learn/bin/queue.txt`, run by `runner.sh` as
`bash <wrapper> <binary> <name> <seed> [flags]` (wrappers above; `logs/<name>.log` has the output, `logs/runner.out`
the order). Flags per row:

| rows | flags |
|---|---|
| mus-r3 | `--optimizer musgd --set momentum=0.9 --set warmup_bias_lr=0` |
| mus-937 | `--optimizer musgd` (momentum 0.937 and warmup_bias_lr 0.1 are the config defaults) |
| mus-m937w0 | `--optimizer musgd --set warmup_bias_lr=0` |
| mus-m09w01 | `--optimizer musgd --set momentum=0.9` |
| ft-main | none (`ft.sh` with `folo-main.exe`) |
| ft-nbs16 | `--set nbs=16` |
| ft-wf0 | `--set warmup_floor=0` |
| ft-wf0lr2 | `--set warmup_floor=0 --optimizer adamw --lr0 0.002 --set momentum=0.9 --set warmup_bias_lr=0` |
| ft-wf0we1 | `--set warmup_floor=0 --set warmup_epochs=1` |
| ft-musgd | `--optimizer musgd --lr0 0.01` (floor 100: binary r2a, whose default floor is still 100) |
| ft-wf0mus | `--set warmup_floor=0 --optimizer musgd --lr0 0.01` |
| ft-muslr005 / ft-muslr02 | `--set warmup_floor=0 --optimizer musgd --lr0 0.005` / `--lr0 0.02` |
| ft-musy26w | `--set warmup_floor=0 --optimizer musgd --lr0 0.01 --set musgd.muon=0.528 --set musgd.sgd=0.674 --set musgd.cls_lr=2.737` |
| ft-musm09 | `--set warmup_floor=0 --optimizer musgd --lr0 0.01 --set momentum=0.9` |
| ft-r4 | none, binary r4 (its rule-4 draft: MuSGD momentum 0.9, floor 0), the same settings as ft-musm09 (seed 0 bit-identical) |
| ft-r5 | `--optimizer musgd --lr0 0.01 --set momentum=0.937`, binary r4 (floor 0): the shipped rule's settings |
| auto-main / auto-r4 | none (`p1.sh` with `folo-main.exe` / `folo-r4f.exe`, the PR #188 build: `auto` rule 4) |
| r4-w55 / r4-cls1 / r4-topk13 / r4-cos | `--set musgd.muon=0.5 --set musgd.sgd=0.5` / `--set loss.cls_gain=1.0` / `--set loss.tal.topk=13` / `--cos-lr` (r4f) |
| ft-w55 / ft-cls1, shapes-w55 / shapes-cls1 | the same flags with `ft.sh` / `shapes.sh` (r4f) |
| o2o-det / o2o-att10 / o2o-att20 | `p1o.sh` (adds `--head o2o`): `--optimizer musgd` / and `--set o2o_attached_epochs=10` / `=20` (r2b) |
| shapes-main / shapes-mus / shapes-r5 | none / `--optimizer musgd --set momentum=0.9 --set warmup_floor=0` / `--optimizer musgd --lr0 0.01 --set momentum=0.937` (r4) |

(`bin/chain1.sh` and `chain2.sh` were the first queues and were stopped after `mus-937-s0`; none of their later lines
ran as written there.)

## Shipped (PR #188)

- **`auto` rule 4**: MuSGD at lr0 0.01, momentum 0.937, warmup_bias_lr 0.1 for every run. Rule 3 (and Ultralytics
  8.4.198) picks AdamW at or below 10,000 iterations and zeroes the bias warmup LR. On the proxy the bias warmup is
  worth +9.5% to MuSGD (three seeds per side); MuSGD beats `auto`'s AdamW on the proxy (round 1: 0.0747 vs about
  0.112), on the shapes gate (0.892 vs 0.926 mAP50-95, one seed) and fine-tuning BCCD (0.607 vs 0.625, three seeds),
  and is level or better on the other three fine-tuning sets.
- **`warmup_floor` default 0**: the 100-iteration floor (dropped by Ultralytics 8.4) made a 205-image fine-tune warm
  up for 25 of its 100 epochs; without it BCCD gains 0.013 AP with AdamW (three seeds). Old configs keep 100.
- Opt-in, defaults unchanged: `musgd.{muon,sgd,cls_lr}` (YOLO26's published weights lost on BCCD, one seed) and
  `o2o_attached_epochs`.

## Shipped (PR #194: `auto` rule 5)

- **MuSGD step weights 0.5 / 0.5 for a run from scratch** (`AUTO_MUSGD_SCRATCH_WEIGHTS`), the reference trainer's
  0.2 / 1.0 for one from pretrained weights (`TrainConfig.transfer`, set by the trainer from `--weights` and stored
  with the run). Proxy 0.1234 vs 0.1135 (three seeds each), shapes gate 0.934 vs 0.926 (one seed); fine-tunes resolve
  exactly as under rule 4, so their numbers above stand.

**Check of the PR build** (`folo-r5.exe`, rule 5, no flags; one seed each): the proxy resolves `auto → musgd lr0 0.01
momentum 0.937 muon 0.5 sgd 0.5` and scores 0.1242 (seed 0; the `r4-w55-s0` screen: 0.1248); BCCD (from
`folo-n-coco-v2`) resolves `muon 0.2 sgd 1` and scores 0.6172 (seed 0; the same settings on the earlier binary:
0.6294, rule-4 seeds over both momentums 0.616–0.629). The two binaries differ in main's GPU kernels (#184–#186),
so these are other rounding trajectories of the same recipe.

## Targets

- **Proxy P1, +5% over main's recipe and over Ultralytics' rows**: rule 5's `auto` 0.1234 against main's `auto`
  0.0757 (**+63%**; three seeds per side), and against Ultralytics 8.4.198's round-1 rows on the same images, one seed
  each: `auto` 0.0166, SGD 0.0853 and MuSGD 0.1133 (**+8.9%** over its best row). **Met against main** (replicated)
  **and above the Ultralytics rows** (+8.9%, about four times FOLO's own seed spread of 0.002, but the reference rows
  are single runs, not replicated here). (Rule 4 alone: 0.1135, level with Ultralytics' MuSGD.)
- **BCCD ≥ yolo26n (0.626)**: 0.625 (0.6251 vs 0.6262): level, not above. Not met by its letter. (+0.018 over main's
  0.607; the other three sets level or better.)

## Negative results

- AdamW lr0 0.002 on a fine-tune (round 1's open "floor 0.002"): BCCD 0.605 / 0.615 against 0.614 / 0.621 at
  `auto`'s 0.00143 (two seeds, floor 0). Not shipped.
- MuSGD lr0 0.005 or 0.02 on BCCD (0.608, 0.607), YOLO26's published MuSGD weights (0.609), a 1-epoch warmup (0.611),
  `nbs=16` (0.607): all below MuSGD lr0 0.01 / floor 0 (0.622–0.629), one seed each.
- On rule 4, one seed each on the proxy: `loss.tal.topk=13` 0.1126 and `--cos-lr` 0.1129 (rule 4: 0.1129–0.1143):
  nothing. `loss.cls_gain=1.0`: +4.5% on the proxy (three seeds) but level on the shapes gate and −0.006 on BCCD:
  not a default; a class-count effect (80 classes) is the likely reading and needs full COCO to tell.
- Mosaic on BCCD (rule-5 build, seeds 0 and 1; test AP 0.6172 / 0.6268 at the default `close_mosaic` 10):
  `--close-mosaic` 3, 20 or 50 give bit-identical results, because the best checkpoint is from epoch 37 / 26, before
  mosaic closes in any of them (the closing epochs never decide the score); `--set augment.mosaic=0` 0.6196 / 0.6256,
  level. A fine-tune peaks around epoch 17–37 of 100 and holds there, whatever the mosaic schedule.
- ProgLoss and STAL (YOLO26): FOLO already has both (the o2m/o2o loss-weight decay of MODEL.md §6.6 and the 16 px
  candidate enlargement of §7.3); nothing to add.

## Leftovers

- `o2o_attached_epochs`: one seed each at 10 and 20 (above); for NMS-free models, three seeds of 20 and an o2o
  fine-tune would tell whether to make it the `--head o2o` default (+0.006 on the o2o head, one seed).
- `loss.cls_gain` by class count (80-class proxy +4.5%, 3- and 4-class sets not): full COCO under rule 5.
- The Studio and Desktop offline estimates (`web/src/routes/runnew/estimate.ts`, `web/src/lib/api/mock/runs.ts`,
  `web/src/routes/run/resolved.ts`, `apps/desktop/src/api/mock/training.rs`) still mirror rule 3 and the 100-step
  floor; other lanes own them.
- Full-scale COCO under rule 5 (the folo-s run used rule 3: momentum 0.9, bias warmup 0, weights 0.2 / 1.0).
- EMA for short fine-tunes (item 4): with 400 optimizer steps the EMA decay `0.9999 · (1 − e^{−t/2000})` ends at
  0.18, so BCCD's validated weights are nearly the live model; a shorter `tau` needs a config key (none today).
- BCCD stays level with yolo26n (0.625 vs 0.626): its fine-tunes peak by epoch 17–37 and the remaining 60+ epochs add
  nothing; what is left is the pretrained weights (folo-n-coco-v2, COCO AP 0.317, vs yolo26n's 0.409).
- `GET /api/runs/{id}/cli` replays `musgd.*` values through the CLI's JSON parsing without `float_roundtrip`, so an
  unusual value can differ in its last bit; a non-default `musgd.cls_lr` is not replayed (before #194 too).

## Headline (three seeds per side unless noted)

| | main 647b981f | after #188 + #194 | reference |
|---|---:|---:|---:|
| Proxy P1 AP (folo-n from scratch, 20 epochs) | 0.0757 | **0.1234** (+63%) | Ultralytics 8.4.198, one run each: MuSGD 0.1133, SGD 0.0853, `auto` 0.0166 |
| Shapes gate mAP50-95 (one seed) | 0.892 | **0.934** | pass bar 0.75 |
| BCCD fine-tune test AP | 0.607 | **0.625** (+0.018) | yolo26n 0.626 |
| medical-pills / signature / brain-tumor | 0.804 / 0.989 / 0.452 | 0.807 / 0.989 / 0.450 | yolo26n 0.803 / 0.982 / 0.436 |
