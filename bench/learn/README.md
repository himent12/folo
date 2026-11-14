# bench/learn: learning from scratch, FOLO against Ultralytics

The incumbent's side of the learning comparison of the hill-climb notebook
[docs/notes/hill/learn.md](../../docs/notes/hill/learn.md), which records the runs and their results: a fixed COCO
subset both tools train on, an Ultralytics trainer run from scratch on it, one pycocotools scoring for both, and the
box statistics of Ultralytics' training augmentation.

| file | what it does |
|---|---|
| [`make_coco_proxy.py`](make_coco_proxy.py) | builds the proxy set: the first N ids of a seeded shuffle of COCO 2017's image ids, as COCO JSON (FOLO, pycocotools) and YOLO labels (Ultralytics) over the same image files, optionally shrunk (`--long-side`), plus the Ultralytics data YAML beside the folder |
| [`ultra_train.py`](ultra_train.py) | trains an Ultralytics model from scratch (`pretrained=False` forced) on that set, validates the best weights and writes `<work-dir>/runs/<name>/dets.json` with the dataset's own category ids, and `summary.json` |
| [`coco_eval.py`](coco_eval.py) | scores a COCO detections file (`folo val --coco --save-json`, or `ultra_train.py`'s `dets.json`) with pycocotools; prints the 12 COCO stats as one JSON line |
| [`ultra_aug_stats.py`](ultra_aug_stats.py) | boxes per image and box-size quantiles of Ultralytics' training transform; FOLO's side is `crates/folo-data/examples/aug_stats.rs` |

## What it needs

- A COCO 2017 folder for `--coco`: `annotations/instances_{train,val}2017.json` next to `train2017/` and `val2017/`,
  the layout of [examples/coco](../../examples/coco/README.md#data).
- `make_coco_proxy.py`: `numpy` and `opencv-python`.
- `ultra_train.py` and `ultra_aug_stats.py`: `ultralytics` (and torch). `ultra_train.py` trains and validates on CUDA
  device 0, which is hard-coded.
- `coco_eval.py`: `pycocotools`.

## Run it

The usage lines of the scripts, with the maintainers' `D:/` paths replaced by placeholders:

```sh
python bench/learn/make_coco_proxy.py --coco <coco dir> --out <dir>/mini320 \
    --train 20000 --val 2000 --long-side 320 --seed 0
python bench/learn/ultra_train.py --model yolo11n.yaml --data <dir>/mini320.yaml \
    --epochs 30 --imgsz 320 --batch 16 --seed 0 --workers 8 --work-dir <dir>/ultra --name y11n-s0
python bench/learn/coco_eval.py --gt <dir>/mini320/annotations/instances_val2017.json \
    --dt <dir>/ultra/runs/y11n-s0/dets.json
YOLO_CONFIG_DIR=<scratch> python bench/learn/ultra_aug_stats.py --data <dir>/mini320.yaml --imgsz 320 --samples 4000
```

`ultra_train.py` keeps Ultralytics' settings, downloads and runs inside `--work-dir`; `--resume`, `--stop-after N` and
`--eval-only` continue, pause and re-score a run, and `--set key=value ...` passes other Ultralytics overrides. The
scripts write only to the paths they are given. There are no tests in this folder.

[Up: bench/](../README.md)
