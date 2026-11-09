# COCO: folo-n pretrained from scratch, and fine-tuning from it

folo-n trained **from scratch** on COCO train2017 at 640 px on one RTX 4080 SUPER for **50 epochs** (8.7 h of
epochs), evaluated on val2017 with FOLO's pycocotools-exact evaluator, published as `folo-n-coco` (a draft
release for now, [Publish](#publish)), and used to fine-tune the R6 detector ([Fine-tuning](#fine-tuning-r6)).

**Result: AP 0.314 / AP50 0.455 on val2017** (f32; f16 the same to three decimals), cross-checked number for number
against pycocotools. Ultralytics' published nano models reach 0.395 (YOLO11n) and 0.409 (YOLO26n). YOLO26n was
pretrained for 150 epochs on Objects365, then fine-tuned for 245 epochs on COCO with evolved hyperparameters. This run
is **50 epochs on COCO alone, from random init**. Data, schedule, recipe and architecture all differ, so this is not a
like-for-like comparison, and how much of the gap each accounts for is not known ([Results](#results-val2017)).

## Data

COCO 2017 from the official servers, into `D:/folo-data/coco` (about 20 GB of zips, 26 GB unpacked):

```sh
mkdir -p D:/folo-data/coco/zips && cd D:/folo-data/coco/zips
for f in annotations/annotations_trainval2017.zip zips/val2017.zip zips/train2017.zip; do
  curl -fL -C - --retry 5 -O "http://images.cocodataset.org/$f"     # -C -: resumable
done
md5sum *.zip
#   f4bbac642086de4f52a3fdda2de5fa2c  annotations_trainval2017.zip
#   442b8da7639aecaf257c1dceb8ba8c80  val2017.zip
#   cced6f7f71b7629ddf16f17bbcfab6b2  train2017.zip
cd .. && for z in zips/*.zip; do unzip -q "$z"; done                # annotations/, train2017/, val2017/
```

cocodataset.org publishes no checksums; the MD5s above are the widely cited ones (torchvision, TFDS) and all three
matched. `unzip` checks every member's CRC as well.

folo-data reads that folder as is (`annotations/instances_{train,val}2017.json` next to `train2017/` and `val2017/`),
no conversion:

```sh
cargo folo dataset stats D:/folo-data/coco
#   coco: 80 classes, 118,287 train / 5,000 val images, 886,282 boxes (849,947 / 36,335; crowd regions excluded)
```

Opening it reads every image header in parallel (about 50 s) and, since this PR, parses the 448 MB train JSON with a
1.2 GB peak instead of 7 GB (docs/DATA.md). No pre-pass is needed: COCO images are already at most 640 px on the
long side, so a resized cache would change nothing, and the loader keeps up with the GPU (about 200 img/s with
mosaic, 16 workers, on a CPU shared with other jobs; the trainer's GPU peak for folo-n at B=16 is 219 img/s).

## Train

```sh
cargo folo train --data D:/folo-data/coco --model folo-n --imgsz 640 --batch 16 --epochs 50 --cos-lr \
  --patience 1000 --workers 16 --device gpu:0 --seed 0 --project D:/folo-data/runs-coco --name coco-folo-n
# after an interruption:
cargo folo train --resume D:/folo-data/runs-coco/coco-folo-n/weights/last.folo --workers 16 --device gpu:0
```

The MODEL.md §8 recipe: SGD (`auto` picks it: 92,450 optimizer steps > 10,000) at lr0 0.01, momentum 0.9, nominal
batch 64 (batch 16 × accumulate 4), 3 warmup epochs, weight decay 5e-4, EMA, mosaic + affine + HSV + flips, mosaic
closed for the last 10 epochs. No mixup, copy-paste, multi-scale or distillation. Two choices fit the budget: the
epoch count, and cosine decay (`--cos-lr`) so the schedule reaches `lr0 · lrf` exactly at the last epoch. A first
5-minute probe measured 195 img/s, about 10.5 min per epoch with validation, so 50 epochs ≈ 9 h was the budget.
`--patience 1000` keeps early stopping from cutting the schedule short.

### Run

2026-11-08, 09:11 to 20:44 local time on the dev machine (RTX 4080 SUPER, Vulkan, 16 loader workers, a CPU shared with
other agents' builds). The 50 epochs took **8.7 h** (518–850 s each; one slow epoch of 1765 s while other jobs held the
machine), about **11.5 h** of wall clock in all. The rest was lost to three interruptions, each resumed from
`last.folo` at the epoch boundary with the stored optimizer, schedule and EMA:
- after epoch 9, a console Ctrl+C meant for another process reached the trainer (now it runs in its own hidden
  process group);
- after epochs 29 and 41, the process died without an error message. The machine's CPU is unstable under long
  all-core load, and other jobs died the same night. Any work since the last checkpoint was redone. After the epoch-29
  crash the machine sat idle for about 2 h before the resume.

No non-finite batch. 7393 batches (1849 optimizer steps) per epoch, 219 img/s peak and about 200 img/s typical: the
loader and the GPU are close to balanced.

### Curve (internal val: network-input boxes, no crowd regions, `best.folo` fitness)

| epoch | box | cls | dfl | val P | val R | val mAP50 | val mAP50-95 | epoch time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.064 | 5.016 | 3.488 | 0.008 | 0.033 | 0.005 | 0.002 | 724 s |
| 3 (end of warmup) | 1.772 | 3.109 | 1.891 | 0.182 | 0.164 | 0.112 | 0.064 | 736 s |
| 5 | 1.594 | 2.535 | 1.676 | 0.294 | 0.249 | 0.215 | 0.134 | 547 s |
| 10 | 1.457 | 2.115 | 1.510 | 0.395 | 0.324 | 0.309 | 0.203 | 586 s |
| 20 | 1.366 | 1.856 | 1.405 | 0.473 | 0.348 | 0.359 | 0.242 | 612 s |
| 30 | 1.306 | 1.703 | 1.347 | 0.510 | 0.385 | 0.403 | 0.275 | 581 s |
| 40 | 1.259 | 1.582 | 1.307 | 0.526 | 0.411 | 0.432 | 0.298 | 581 s |
| 41 (mosaic off) | 1.248 | 1.479 | 1.294 | 0.531 | 0.413 | 0.433 | 0.299 | 707 s |
| 45 | 1.222 | 1.417 | 1.276 | 0.551 | 0.416 | 0.442 | 0.306 | 530 s |
| **50** (best) | 1.212 | 1.392 | 1.267 | 0.535 | 0.429 | 0.449 | **0.310** | 574 s |

Every epoch improved on the one before, so the best epoch is the last. The gains shrink as the cosine LR decays:
about 0.0022 mAP50-95 per epoch from epoch 30 to 40, 0.0009 per epoch from 45 to 50 (0.3055 → 0.3100), and 0.0005
for the last epoch. The run ends at its schedule's floor, not at a measured plateau. A longer schedule would
probably gain more, but how much has not been measured.

## Evaluate

```sh
cargo folo val --model D:/folo-data/runs-coco/coco-folo-n/weights/best.folo --data D:/folo-data/coco \
  --coco --save-json D:/folo-data/runs-coco/eval/val2017-dets-f32.json --device gpu:0 --workers 16
cargo folo val --model D:/folo-data/runs-coco/coco-folo-n/weights/best.folo --data D:/folo-data/coco \
  --coco --half --save-json D:/folo-data/runs-coco/eval/val2017-dets-f16.json --device gpu:0 --workers 16
# cross-check with pycocotools on exactly the same detections:
D:/folo-venvs/parity/Scripts/python -I -c "from pycocotools.coco import COCO; from pycocotools.cocoeval import COCOeval; \
  g = COCO('D:/folo-data/coco/annotations/instances_val2017.json'); d = g.loadRes('D:/folo-data/runs-coco/eval/val2017-dets-f32.json'); \
  e = COCOeval(g, d, 'bbox'); e.evaluate(); e.accumulate(); e.summarize()"
```

Each `val` takes about 18 s for the 5000 images (plus about 15 s to open the dataset) and writes about 910,000
detections (`conf > 0.001`, up to 300 per image; COCOeval keeps 100).

### Results (val2017)

`best.folo` (EMA, epoch 50), 640 px, rect letterbox, the MODEL.md §9.3 validation settings, 5000 images, 36,335 boxes
plus 446 crowd regions:

| | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR1 | AR10 | AR100 | AR_S | AR_M | AR_L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **f32** | **0.314** | **0.455** | 0.339 | 0.144 | 0.335 | 0.458 | 0.284 | 0.486 | 0.544 | 0.311 | 0.598 | 0.745 |
| f16 (`--half`) | 0.314 | 0.455 | 0.339 | 0.144 | 0.335 | 0.458 | 0.284 | 0.486 | 0.544 | 0.310 | 0.597 | 0.745 |

**pycocotools agrees on all 12 numbers.** Its `COCOeval` on the `--save-json` file gives the same stats as FOLO's
`coco` object to within 3.3e-16 (f32: AP 0.313810, AP50 0.455412, AP75 0.339179) and 5.6e-16 (f16: AP 0.313924).
f16 differs from f32 by at most 0.0004 (AP_L), well inside the 0.002 bound in docs/BENCHMARKS.md. The internal
val that picks `best.folo` (network-input boxes, crowd regions dropped) reads 0.310 / 0.449. The official numbers
are slightly higher, partly because crowd regions absorb some false positives.

**Against the incumbent.** Ultralytics publishes AP 0.395 for YOLO11n (2.6 M params) and 0.409 for YOLO26n (2.4 M;
0.401 with its end-to-end head), with the same evaluation (val2017, 640, single scale). folo-n is 2.6 M params. The
released YOLO26n was pretrained for 150 epochs on Objects365-v1, then fine-tuned for 245 epochs on COCO at batch 128
with size-specific hyperparameters from evolutionary search ([the YOLO26 paper](https://arxiv.org/abs/2606.03748), §4.1 and supplementary S3).
YOLO11n's recipe is not given on its model page. **This run is 50 epochs on COCO alone, from random init**, with the
plain MODEL.md §8 recipe. The data, the schedule, the recipe and the architectures all differ, so 0.314 against
0.395–0.409 cannot be split into those causes. The shorter schedule and the missing Objects365 pretraining are
plausible large contributors, but neither has been isolated. The result does not show that FOLO matches YOLO11n, and
it does not show that FOLO falls short at equal budget. That needs an equal-budget run (a much longer FOLO schedule,
or Ultralytics for 50 epochs from scratch on COCO on this machine), and neither has been done.

Per class (AP / AP50, f32): large, distinctive objects do well; small, thin or ambiguous ones do not.

| best | AP | AP50 | | worst | AP | AP50 |
|---|---:|---:|---|---|---:|---:|
| giraffe | 0.635 | 0.848 | | hair drier (11 boxes) | 0.002 | 0.005 |
| bear | 0.632 | 0.773 | | spoon | 0.036 | 0.070 |
| zebra | 0.625 | 0.863 | | toothbrush | 0.053 | 0.104 |
| train | 0.621 | 0.802 | | handbag | 0.055 | 0.100 |
| elephant | 0.597 | 0.814 | | knife | 0.056 | 0.098 |
| airplane | 0.594 | 0.786 | | backpack | 0.070 | 0.131 |
| cat | 0.590 | 0.787 | | apple | 0.092 | 0.136 |
| fire hydrant | 0.589 | 0.745 | | book | 0.097 | 0.219 |

`person`, the most common class (10,777 boxes), gets AP 0.484 / AP50 0.732; `car` 0.316 / 0.519, `traffic light`
0.164 / 0.321. Small objects are the weak spot (AP_S 0.144 vs AP_L 0.458), as for every nano model at 640.

## Publish

```sh
cargo folo export --model D:/folo-data/runs-coco/coco-folo-n/weights/best.folo --format weights \
  --output D:/folo-weights/folo-n-coco.folo                         # EMA, unfused, f32, no optimizer state
cargo folo export --model D:/folo-weights/folo-n-coco.folo --format folo-f16 --output D:/folo-weights/folo-n-coco-f16.folo
cd D:/folo-weights && sha256sum folo-n-coco.folo folo-n-coco-f16.folo > SHA256SUMS
gh release create weights-v1 --draft --repo himent12/folo --title "FOLO pretrained weights v1" \
  --notes-file notes.md folo-n-coco.folo folo-n-coco-f16.folo SHA256SUMS
```

| asset | what | bytes | SHA-256 |
|---|---|---:|---|
| `folo-n-coco.folo` | folo-n, 80 classes, EMA weights, unfused f32: fine-tune or predict | 10,601,456 | `5d5429ae2eb6050363a7f2253449cbe99a1dd2a36e3cd988b2011e2bc6bd5b01` |
| `folo-n-coco-f16.folo` | the same, BatchNorm fused, f16: inference only | 5,251,088 | `de863020b3542f0c16b94d0c2e9a03b8d5956e339eedaa875db65521de2d5c47` |

The release `weights-v1` is a **draft**: it is not public yet, and its tag is created when it is published. Until
then, only accounts with write access to the repository can see it. For them, `folo weights pull` already works, since
`gh release download` finds draft releases by tag for repository writers (checked: both files pull and verify):

```sh
folo weights pull folo-n-coco          # → ~/.folo/weights/folo-n-coco.folo, sha256 verified
folo weights pull folo-n-coco-f16
folo train --data <your dataset> --weights folo-n-coco ...
folo predict --model folo-n-coco-f16 --half <images>
```

Pulled and passed by name, the files give the val2017 numbers above. `folo val --model folo-n-coco --data D:/folo-data/coco --coco`
is identical to `best.folo` in f32 (all 12 stats). `folo val --model folo-n-coco-f16 --data D:/folo-data/coco --half
--coco` matches the f16 row
to four decimals.

Once the release is published, everyone who can read the repository uses the same commands. Without `folo`:
`gh release download weights-v1 --repo himent12/folo --pattern 'folo-n-coco*' --pattern SHA256SUMS`, then
`sha256sum -c --ignore-missing SHA256SUMS`.

## Fine-tuning: R6

`--weights folo-n-coco` on the 2-class R6 set (examples/r6) against a from-scratch control with the same flags,
seed and epoch budget, both evaluated on r6-clean test and the held-out r6-fresh. Results:
[examples/r6, Fine-tuning from COCO](../r6/README.md#fine-tuning-from-coco).

| folo-n, 100 epochs on R6 (best.folo, 640 px) | r6-clean test mAP50-95 | r6-fresh (held out) mAP50-95 | r6-fresh Head |
|---|---:|---:|---:|
| from scratch | 0.350 | 0.118 | 0.047 |
| **from folo-n-coco** | **0.441** | **0.201** | **0.105** |
| *from scratch, 410 epochs (examples/r6)* | *0.397* | *0.130* | *0.053* |

Pretraining is worth more than four times the epochs. In 100 epochs the fine-tune beats the 410-epoch from-scratch
run on both splits, and it generalises much better to unseen footage (+70% on r6-fresh over the same budget).
