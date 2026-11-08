# COCO: folo-n pretrained from scratch, and fine-tuning from it

folo-n trained **from scratch** on COCO train2017 at 640 px on one RTX 4080 SUPER, under a **~10 hour** wall-clock
budget, evaluated on val2017 with FOLO's pycocotools-exact evaluator, published as `folo-n-coco`, and used to
fine-tune the R6 detector (examples/r6).

> Results are filled in when the run finishes (this file lands with the code first).

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
closed for the last 10 epochs. Two choices fit the budget: the epoch count, and cosine decay (`--cos-lr`) so the
schedule reaches `lr0 · lrf` exactly at the last epoch. A first 5-minute probe measured 195 img/s, i.e. about
10.5 min per epoch with validation, so **50 epochs ≈ 9 h** (plus contention) is the budget. `--patience 1000` keeps
early stopping from cutting the schedule short.

## Evaluate

```sh
cargo folo val --model D:/folo-data/runs-coco/coco-folo-n/weights/best.folo --data D:/folo-data/coco \
  --coco --save-json D:/folo-data/runs-coco/val2017-dets.json
# cross-check with pycocotools on exactly the same detections:
D:/folo-venvs/parity/Scripts/python -I -c "from pycocotools.coco import COCO; from pycocotools.cocoeval import COCOeval; \
  g = COCO('D:/folo-data/coco/annotations/instances_val2017.json'); d = g.loadRes('D:/folo-data/runs-coco/val2017-dets.json'); \
  e = COCOeval(g, d, 'bbox'); e.evaluate(); e.accumulate(); e.summarize()"
```

## Publish

```sh
cargo folo export --model D:/folo-data/runs-coco/coco-folo-n/weights/best.folo --format weights \
  --output D:/folo-weights/folo-n-coco.folo
cargo folo export --model D:/folo-weights/folo-n-coco.folo --format folo-f16 --output D:/folo-weights/folo-n-coco-f16.folo
cd D:/folo-weights && sha256sum folo-n-coco.folo folo-n-coco-f16.folo > SHA256SUMS
gh release create weights-v1 --draft --repo himent12/folo --title "Pretrained weights v1" \
  folo-n-coco.folo folo-n-coco-f16.folo SHA256SUMS
```

Users then run `folo weights pull folo-n-coco` (docs/CLI.md, Weights) and `folo train --weights folo-n-coco ...`.
