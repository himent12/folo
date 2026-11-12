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

(The rest of the proxy screens, the o2o runs and seed 2 are added below as they finish.)

### Fine-tune on BCCD (test AP of best.folo, benchmark scorer)

| run | seeds | settings | test AP per seed | mean | AP50 mean |
|---|---|---|---|---:|---:|
| main (published, A40) | 0 1 2 | `auto` → AdamW lr0 0.001429, warmup 100 steps (25 of 100 epochs) | 0.6062 0.6074 0.6061 | 0.607 | 0.883 |
| ft-main | 0 | the same, this PC | 0.6062 | | 0.892 |
| ft-nbs16 | 0 | `nbs=16` (no accumulation: 13 steps per epoch, warmup 100 steps = 8 epochs) | 0.6069 | | 0.898 |
| ft-wf0 | 0 1 2 | `warmup_floor=0` (warmup 12 steps = 3 epochs, as Ultralytics 8.4) | 0.6143 0.6211 0.6225 | 0.619 | 0.897 |
| ft-wf0lr2 | 0 1 | floor 0, AdamW lr0 0.002 (round 1's AdamW floor 0.002) | 0.6049 0.6151 | 0.610 | 0.889 |
| ft-wf0we1 | 0 | floor 0, `warmup_epochs=1` | 0.6109 | | 0.891 |
| ft-musgd | 0 | `--optimizer musgd --lr0 0.01` (momentum 0.937, warmup_bias_lr 0.1), floor 100 | 0.6209 | | 0.901 |
| ft-wf0mus | 0 1 2 | MuSGD lr0 0.01, momentum 0.937, warmup_bias_lr 0.1, floor 0 | 0.6294 0.6223 0.6235 | **0.625** | 0.910 |
| ft-muslr005 | 0 | the same at lr0 0.005 | 0.6079 | | 0.893 |
| ft-muslr02 | 0 | the same at lr0 0.02 | 0.6074 | | 0.896 |
| ft-musy26w | 0 | the same with YOLO26's published MuSGD weights (muon 0.528, sgd 0.674, cls_lr 2.737) | 0.6085 | | 0.893 |
| ft-musm09 / ft-r4 | 0 1 2 | MuSGD lr0 0.01, momentum 0.9, warmup_bias_lr 0.1, floor 0 (`auto` rule 4 draft) | 0.6231 0.6158 0.6269 | 0.622 | 0.906 |
| yolo26n (published) | 0 1 2 | Ultralytics defaults from yolo26n.pt | 0.6212 0.6339 0.6235 | 0.626 | 0.911 |

### Shapes gate (seed 0)

| run | settings | final mAP50 | final mAP50-95 | gate first met |
|---|---|---:|---:|---:|
| shapes-main | main: `auto` → AdamW lr0 0.00125, warmup 100 steps | 0.980 | 0.892 | epoch 9 |
| shapes-mus | MuSGD lr0 0.01, momentum 0.9, warmup_bias_lr 0.1, floor 0 | 0.982 | 0.917 | epoch 10 |
| shapes-r5 | MuSGD lr0 0.01, momentum 0.937, warmup_bias_lr 0.1, floor 0 (= rule 4) | 0.985 | **0.926** | epoch 8 |

No run had a non-finite batch.
