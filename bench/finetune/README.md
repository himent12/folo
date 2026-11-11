# Fine-tuning on small datasets: folo-n against yolo26n

Most people train on a few hundred of their own images, starting from pretrained weights. This harness measures
that: it fine-tunes **folo-n** from FOLO's COCO weights and **Ultralytics YOLO26n** from `yolo26n.pt` on four small
public datasets and scores both with **one scorer** (pycocotools against one ground truth). The numbers go into
[docs/BENCHMARKS.md](../../docs/BENCHMARKS.md), *Fine-tuning on small datasets*. The harness is not tied to the
nano size: `run_folo.py --arch folo-s --weights <folo-s weights>` and `run_ultralytics.py --weights yolo26s.pt` add
an `s` row, and the renderer pairs models by size letter (folo-n with yolo26n, folo-s with yolo26s), one table each.

| file | what it does |
|---|---|
| `datasets.py` | fetch (cached, sha256-checked), convert and lay out the datasets; `datasets.json` is its metadata |
| `scorer.py` | the common scorer: COCO detections JSON against `annotations/instances_<split>.json`, pycocotools |
| `run_ultralytics.py` | Ultralytics runs (one subprocess each), scored by `scorer.py`; self-contained for a pod |
| `run_ultralytics_pod.sh` | the Ultralytics matrix on a plain PyTorch GPU pod: installs, fetches, runs, packs `results.tgz` |
| `run_folo.py` | FOLO runs through a running `folo serve` (local GPU or cloud pods), evaluated with `folo val --save-json` |
| `render_table.py` | `results/*.json` → `results/summary.json` and the tables (`--update docs/BENCHMARKS.md`) |
| `crosscheck_coco.py` | checks the scorer against both tools' own COCO val2017 numbers |
| `common.py` | run keys, result files, the summary |
| `test_finetune.py` | bookkeeping tests: `python -m unittest discover -s bench/finetune -p "test_*.py"` |

## Datasets

Four small object-detection sets from different domains, all with a licence that allows this use and a stable
download URL that needs no account. Three come from Ultralytics' asset releases (and have only train and val, so
their scores are on val); BCCD is not curated by Ultralytics and has its own test split, which is what it is scored
on. Counts are after the layout step (`datasets.json` has the per-class counts):

| name | domain | source | licence | classes | train images / boxes | scored on: images / boxes |
|---|---|---|---|---:|---:|---|
| `bccd` | blood cells, microscopy | [Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset) at `d272fb1` | MIT | 3 (RBC, WBC, Platelets) | 205 / 2804 | test: 72 / 945 |
| `brain-tumor` | brain MRI | [Ultralytics](https://docs.ultralytics.com/datasets/detect/brain-tumor/) | AGPL-3.0 | 2 (negative, positive) | 893 / 925 | val: 223 / 241 |
| `signature` | signatures on documents | [Ultralytics](https://docs.ultralytics.com/datasets/detect/signature/) | AGPL-3.0 | 1 | 143 / 143 | val: 35 / 35 |
| `medical-pills` | pills, product photos | [Ultralytics](https://docs.ultralytics.com/datasets/detect/medical-pills/) | AGPL-3.0 | 1 | 92 / 1623 | val: 23 / 399 |

Considered and left out: Ultralytics' `african-wildlife` (its source on Kaggle states no licence), Oxford-IIIT Pet
and PASCAL VOC subsets (larger than "a few hundred images" without an arbitrary cut).

`python bench/finetune/datasets.py fetch --root D:/folo-data/finetune` lays each one out as
`<root>/<name>/{data.yaml, images/<split>/, labels/<split>/, annotations/instances_<eval split>.json, meta.json}`:
images byte for byte; YOLO labels rewritten from the parsed boxes (6 decimals; BCCD's VOC XML converted with the
convention of Ultralytics' VOC.yaml, its 2 inverted boxes dropped; boxes are clipped to the image and exact
repeats dropped, none of which the other three needed); `data.yaml` without `path:`, so both tools read the folder
the YAML is in; and the COCO ground truth of the evaluation split built from those same labels (image ids 1.. in
file order, category id = class + 1, boxes in pixels, area = w · h, no crowd regions). The layout has a digest
(sha256 over every file the tools read) that is recorded in `datasets.py`, so a pod lays out byte-identical
datasets or fails. No image has an EXIF orientation (the fetch refuses one rather than letting two tools rotate
it differently). Nothing but the code and `datasets.json` is committed.

## Protocol

- **Both tools at their out-of-the-box fine-tuning defaults**, the same for every dataset: 640 px, batch 16,
  **100 epochs**, seeds 0, 1 and 2. 100 epochs is both tools' default (`folo train --epochs 100`, Ultralytics'
  `epochs: 100`), long enough for a few hundred images (600 to 5,600 batches of 16 per run) and cheap enough to run 28
  times.
- **No early stopping** for either: both default to `patience: 100`, which cannot stop a 100-epoch run (FOLO's is
  not settable through the API; `run_ultralytics.py` passes `patience=max(100, epochs)`).
- **FOLO**: `POST /api/runs` with `weights: "folo-n-coco-v2"` (COCO AP 0.317, epoch 62 of a paused 300-epoch
  run; these are the weights of this comparison), `arch`, `epochs`, `batch`, `imgsz`, `seed`, everything else
  FOLO's default (`optimizer: auto`, f32, its augmentation). What `auto` resolved to is in each result's `versions.resolved`.
- **Ultralytics 8.4.198**: `YOLO("yolo26n.pt").train(data=..., epochs=100, imgsz=640, batch=16, seed=s,
  deterministic=True)` with its defaults (`optimizer: auto`, AMP on, its augmentation); `plots=False` changes no
  number.
- **Control**: folo-n from scratch (no `weights`), seed 0, per dataset, to show what the pretrained weights buy.
- **Scoring**: each run's best checkpoint (each tool's own choice: FOLO's best mAP50-95 epoch, Ultralytics'
  `best.pt` by its fitness) predicts the evaluation split with each tool's validation pipeline at the same settings
  (conf 0.001, at most 300 detections per image, 640 px; FOLO with its NMS at IoU 0.7, YOLO26n with its NMS-free
  end-to-end head), written as COCO detections JSON (`folo val --save-json`; Ultralytics `val(save_json=True)`,
  mapped onto the ground truth's ids by file name, since it writes numeric stems as integers and categories as
  `class + 1`). Both are scored by pycocotools against the same `instances_<split>.json`: AP@[.5:.95], AP50, AP75
  (COCO's protocol: at most 100 detections per image and category count). Each tool's own number is kept in
  `tool_reported` for reference and never compared.
- **Training time** is the wall time of training (all epochs, with the per-epoch validation both tools run): FOLO
  from the run's `started` to `finished`, Ultralytics around `train()`.
- **Hardware**: the real matrix runs on rented RunPod GPUs of one type (FOLO through its own cloud feature, one pod
  per run; Ultralytics on one plain PyTorch pod). Nothing trains on the dev PC's GPU: smoke runs there train on
  the CPU (`run_folo.py --device cpu`, `run_ultralytics.py --device cpu`; `--device` has no default in
  `run_ultralytics.py` for that reason). Each result records the device it trained on.

## Running it

```sh
# datasets (any machine)
python bench/finetune/datasets.py fetch --root D:/folo-data/finetune

# FOLO: a folo serve as the control plane of a RunPod connection (docs/CLOUD.md), one pod per run
folo weights pull folo-n-coco-v2
folo serve --port 7979 --data-dir D:/folo-data/runs-finetune/serve
python bench/finetune/run_folo.py --server http://127.0.0.1:7979 --data-root D:/folo-data/finetune \
    --work-dir D:/folo-data/runs-finetune/folo --results bench/finetune/results --folo folo \
    --compute <connection id>:<offer id> --max-hours 0.5 --parallel 3 --budget-usd 1.0
# a CPU smoke run on that server instead (into a scratch results folder)
python bench/finetune/run_folo.py ... --results D:/folo-verify/smoke --datasets medical-pills --seeds 0 \
    --epochs 2 --imgsz 320 --device cpu --val-device cpu

# Ultralytics: on a pod, with run_ultralytics_pod.sh next to this folder's .py files (WEIGHTS=yolo26s.pt: an s row)
SEEDS=0,1,2 EPOCHS=100 SERVE_PORT=8888 bash run_ultralytics_pod.sh /workspace/finetune
# or anywhere else; a CPU smoke run:
python bench/finetune/run_ultralytics.py --data-root D:/folo-data/finetune --work-dir D:/folo-verify/smoke-ul \
    --results D:/folo-verify/smoke --weights yolo26n.pt --datasets medical-pills --seeds 0 --epochs 2 \
    --imgsz 320 --device cpu

# the table
python bench/finetune/render_table.py bench/finetune/results --update docs/BENCHMARKS.md
```

Python 3.10+ with `requests`, Pillow and pycocotools (plus torch and ultralytics for the Ultralytics side). Both
runners skip a run whose result file exists, so a matrix can be stopped and started again; `run_folo.py` follows a
run it started before (its `state.json`, or a run of the same name on the server) instead of starting it twice,
and on Ctrl+C, an error or `--budget-usd` it stops every run it started that is still active and terminates any
cloud instance of those runs that is still live before it exits. Smoke runs belong in a scratch results folder:
the renderer leaves out runs of another epoch count anyway. Result files are named
`<model>-<dataset>-<variant>-s<seed>.json` (`folo-n-bccd-pretrained-s0.json`, `yolo26n-bccd-pretrained-s0.json`).

## The 2026-11-11 matrix

`results/` holds one file per run of the first matrix (27 of 28 runs; the brain-tumor from-scratch control did not
fit the budget) and `summary.json`; the tables and their reading are in docs/BENCHMARKS.md. It ran on RunPod A40s
(Secure Cloud, $0.59/h), about $1.45 in all:
- FOLO: `run_folo.py --compute cc_…:"NVIDIA A40@secure" --max-hours 0.5 --parallel 4` against a local control
  plane serving the Linux build of `ee846c88`. RunPod often refused a launch for capacity right after offering
  "high" stock; `run_folo.py` retries those (nothing is rented). A fine-tune cost $0.025–0.11 (pills to
  brain-tumor), about 40 s of it the pod's boot and upload.
- Ultralytics: one `ubuntu:22.04` pod whose start command arms a self-terminating watchdog, installs python3 and
  torch 2.5.1+cu124, fetches this folder as a tarball and runs `PASSES="0:1 1,2:2" bash run_ultralytics_pod.sh`;
  the results were copied off its HTTP port and the pod terminated. A first attempt on a `runpod/pytorch` devel
  image never started its container within 12 minutes (the image pull) and was terminated.

## The scorer, checked

`crosscheck_coco.py` scores both tools' COCO models on COCO val2017 (all 5,000 images, the original annotations
with crowd regions) through exactly the paths the runners use (2026-11-10, RTX 4080 SUPER,
`results/crosscheck-coco.json`):

| model | the tool's own number | common scorer AP | AP50 | published |
|---|---:|---:|---:|---:|
| folo-n-coco-v2 | 0.31682 (`folo val --coco`, pycocotools-exact) | 0.31682 | 0.4578 | 0.317 |
| yolo26n (`yolo26n.pt`) | 0.4023 (Ultralytics' own metric, on YOLO labels) | 0.4080 | 0.5686 | 0.409 |

FOLO's detections score identically (to 1e-16); Ultralytics' detections score within 0.001 of its published
COCO AP. Its own metric is lower by 0.006 because it is computed differently (YOLO labels without crowd regions,
its own matching), which is why the comparison never uses either tool's own number.
