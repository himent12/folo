# Agent cookbook

Short recipes for driving FOLO from an agent, each in two forms: **MCP tool calls** (`folo mcp`, docs/MCP.md) and
the **CLI with `--json`** (docs/CLI.md). Tool calls are written `→ tool { arguments }` with the interesting part
of the result after `←`; every result also has a one-line text summary. Errors are `isError` results whose text is
`{ "code", "message", "hint" }` (MCP) or the last stdout line `{"error":{...}}` plus a non-zero exit code (CLI).

Setup: `folo mcp` ([README, "Use FOLO from an agent"](../README.md#use-folo-from-an-agent)). That is the real engine:
training, `val`, `predict` and `export` run in the `folo mcp` process, and runs persist under `~/.folo/runs/` (the
data dir; `--data-dir` picks another). Add `--device cpu` to keep inference off the GPU, `--mock` to try the tools on
simulated data, or `--server http://127.0.0.1:7878` to share runs with FOLO Studio. The examples use a dataset at
`/data/cats` and images under `/data/photos`; tool arguments take absolute paths or paths relative to the data dir
(`~` is not expanded).

## 1. Train a detector on a folder of images
The folder must be a labelled dataset: YOLO (`images/` + `labels/`, or a `data.yaml`) or COCO (an instances
`.json`). No labels yet? `create_synthetic_dataset` makes a shapes dataset to practise on.

```
→ folo_info {}                                    ← devices.gpus, scales (params, GFLOPs at 640 / 320)
→ register_dataset { "path": "/data/cats", "name": "cats" }
                                                  ← { "id": "ds_8f21", "classes": ["cat", "dog"], "splits": {...} }
→ get_dataset { "dataset_id": "ds_8f21" }         ← stats (class counts, box sizes), issue_counts, issues (recipe 3)
→ train { "dataset_id": "ds_8f21", "arch": "folo-s", "epochs": 50, "imgsz": 640, "device": "auto" }
                                                  ← { "run_id": "run_4c1a", "status": "queued" }
→ run_status { "run_id": "run_4c1a" }             ← status, epoch / epochs, best, eta_s, last_epoch.val, log
```

Instead of polling `run_status` (every 30 s is plenty), subscribe to `folo://runs/run_4c1a`; the server sends a
resource-updated notification (at most one per second) on every step, epoch and status change. For the loss and
mAP curves read `folo://runs/run_4c1a/epochs` (about 1 KB per epoch), not `/metrics`, which holds every step.
Arguments a tool does not know are rejected (`lr` → "did you mean `lr0`?"), so a typo never trains with defaults;
calling `register_dataset` again on the same folder returns the dataset it already is. When the status is
`completed`, `model_id` names the best weights (`m_4c1a`; `m_4c1a-last` is the last epoch). The prompt
`train_on_folder { "folder_path": "/data/cats" }` returns these steps as instructions.

Stop and resume: `stop_run { "run_id": "run_4c1a" }` stops after the current batch (status `stopping`, then
`stopped`, the last epoch's checkpoint kept). `train { "resume_run_id": "run_4c1a" }` continues the same run from
that checkpoint (optimizer, schedule, EMA and RNG included); leave the other fields out, only `device` and `compute`
may change.
Runs are also stopped this way when the MCP client disconnects, so a new session can resume them. To train further
after a run **completed**, start a new run with `"weights": "m_4c1a"`.

CLI:
```sh
folo train --data /data/cats --model folo-s --epochs 50 --imgsz 640 --json --json-events epoch > train.jsonl
# one {"event":"status"|"epoch"|"log",...} record per event (drop --json-events epoch for per-batch "step"s), then
# {"event":"done","status":"completed","run_dir":"runs/cats-s","best":{"epoch":41,"map50_95":0.62},
#  "weights":{"best":"runs/cats-s/weights/best.folo","last":".../last.folo"}}
folo stop runs/cats-s                                      # from another process: stop after the current batch
folo train --resume runs/cats-s/weights/last.folo --json   # after a stop, Ctrl+C or a killed process
folo train --weights runs/cats-s/weights/best.folo --data /data/cats --epochs 20 --json   # after it completed
```

## 2. Evaluate and compare two runs
```
→ list_runs { "status": "completed" }             ← items: [{ "id", "name", "best", "config", ... }]
→ compare_runs { "run_ids": ["run_4c1a", "run_9e07"] }
    ← runs: [{ run_id, best: { epoch, map50_95 }, best_metrics, final_train_loss, duration_s }, ...],
      config_diff: { "arch": ["folo-s", "folo-n"], "imgsz": [640, 320] }
→ val { "model_id": "m_4c1a", "dataset_id": "ds_8f21", "split": "test" }
→ val { "model_id": "m_9e07", "dataset_id": "ds_8f21", "split": "test" }
    ← { images, imgsz, val: { map50, map50_95, precision, recall, per_class: [...] }, timing: { total_s, per_image_ms } }
```

`compare_runs` compares what the runs measured on their own validation split while training; `val` puts both
models on the same split, thresholds (`conf` 0.001 and `iou` 0.7 by default, the mAP settings) and image size
(`imgsz`, default the model's). Read `folo://runs/{id}/metrics` for full loss and mAP curves, and
`folo://runs/{id}/samples/{epoch}` for a JPEG grid of validation predictions (truth outlined white). The
`diagnose_run { "run_id": "run_4c1a" }` prompt asks for an over / underfitting analysis and a TrainConfig diff.

CLI:
```sh
folo val --model runs/cats-s/weights/best.folo --data /data/cats --split test --json > a.json
folo val --model runs/cats-n/weights/best.folo --data /data/cats --split test --json > b.json
jq -s 'map({model, map50, map50_95, precision, recall})' a.json b.json
```

## 3. Find label problems in a dataset
```
→ get_dataset { "dataset_id": "ds_8f21" }
    ← stats: { class_counts, box_size_hist, empty_images, broken_labels },
      issue_counts: { "error": 3, "warning": 12, "info": 40, "by_kind": { "out_of_range_box": 1, ... } },
      issues: [{ "kind": "out_of_range_box", "severity": "error", "file": "labels/train/0412.txt", "line": 3,
                 "message": "...", "fix": "delete line 3" }, ...] (errors, then warnings; first 50)
→ list_dataset_issues { "dataset_id": "ds_8f21", "severity": "error" }   ← every error, paged (also by kind)
→ list_dataset_images { "dataset_id": "ds_8f21", "class": 1, "include_boxes": true, "limit": 20 }
→ (read folo://datasets/ds_8f21/images/{image_id} to look at it: a 512 px JPEG, ?size=original for the file)
```

`get_dataset` lists errors first, then warnings; infos (`empty_image`, background images) are only counted in
`issue_counts`, so a dataset with many background images cannot hide an error. When there are more than 50,
page with `list_dataset_issues` (`severity`, `kind`, `cursor`). Look for: errors (`broken_label`, `class_overflow`, `out_of_range_box`, `corrupt_image`, `missing_image`; the
loader skips those images), warnings (`clipped_box`, `zero_size_box`, `duplicate_label`, `duplicate_image`), a
class with far fewer boxes than the others, and many tiny boxes for a small `imgsz`. After training, `val`'s
`per_class` and the sample grid show which classes the labels let down. FOLO never edits a dataset; fix the files
and the next `get_dataset` re-indexes them.

CLI (every issue, not just 50, with a suggested fix for each):
```sh
folo dataset check /data/cats --fix-dry-run --json   # exit 0 when clean, 1 with errors or warnings
folo dataset check /data/cats --fix-dry-run --json --min-severity warning   # leave the background-image infos out
# {"clean":false,"counts":{"error":3,"warning":12,"info":40,"by_kind":{...}},
#  "issues":[{"kind","severity","file","line","message","fix"}, ...]}
folo dataset stats /data/cats --json                 # the same stats object as get_dataset
```

## 4. Export and run inference
```
→ export { "model_id": "m_4c1a", "format": "folo-f16" }
    ← { "path": "/home/me/.folo/exports/cats-s-best-f16.folo", "bytes": 18874368 }
→ predict { "model_id": "m_4c1a", "image_path": "/data/photos/garden.jpg", "conf": 0.25 }
    ← { image: { width, height }, boxes: [{ class, name, score, x1, y1, x2, y2 }], timing: { ..., total_ms }, device }
→ predict_batch { "model_id": "m_4c1a", "glob": "/data/photos/*.jpg" }
    ← { processed, failed, boxes_total, per_class_counts, timing, details_uri: "folo://predictions/pb_1" }
```

Boxes are pixel `xyxy` in the original image, highest score first. `predict` also takes `image_base64`, or
`dataset_id` + `image_id` for a dataset image; `predict_batch` takes up to 256 images (`paths` or a `glob`) and
puts the per-image boxes in the linked `folo://predictions/{batch_id}` resource. Exports never overwrite a file
(`-2`, `-3`, ... are added). The exported `.folo` file is a fused f16 inference model; `folo predict` and
`folo val` load it directly.

CLI:
```sh
folo export --model runs/cats-s/weights/best.folo --json   # {"path":"runs/cats-s/weights/best-f16.folo","bytes",...}
                                                           # (never overwrites: a second export is best-f16-2.folo)
folo predict --model runs/cats-s/weights/best-f16.folo /data/photos --conf 0.25 --save annotated --json
# {"event":"prediction","file":"/data/photos/garden.jpg","image":{...},"boxes":[...],"timing":{...},"saved":"..."}
# ... one per image, then {"event":"done","images":12,"failed":0,"boxes":31,"mean_timing":{...}}
```

## 5. Use FOLO from a script via `--json`
With `--json`, stdout is JSON Lines only (human text and logs go to stderr), and the exit code says how it went:
0 ok, 1 runtime error (bad data, a failed run), 2 usage error, 3 not found, 4 not implemented. A failure's last
stdout line is `{"error":{"code","message","hint"}}`. `folo describe --json` lists every command and argument, and
its `mcp_tools` holds every MCP tool with its input and output schemas.

```python
import json, subprocess

def folo(*args):
    """Runs `folo ... --json`; returns its records, raises with the error envelope on failure."""
    p = subprocess.run(["folo", *args, "--json"], capture_output=True, text=True)
    records = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    if p.returncode != 0:
        err = records[-1].get("error", {}) if records else {}
        raise RuntimeError(f"folo {args[0]} exited {p.returncode}: {err.get('code')}: {err.get('message')}")
    return records

done = folo("train", "--data", "/data/cats", "--model", "folo-n", "--epochs", "20", "--imgsz", "320")[-1]
best = done["weights"]["best"]
[report] = folo("val", "--model", best, "--data", "/data/cats")
print(f"mAP50-95 {report['map50_95']:.3f} on {report['images']} images")
for r in folo("predict", "--model", best, "/data/photos"):
    if r.get("event") == "prediction":
        print(r["file"], [(b["name"], round(b["score"], 2)) for b in r["boxes"]])
```

`folo dataset check` is the one command that exits 1 with a useful report (issues found), so read its records
before treating exit 1 as a failure. For a long `folo train`, read stdout line by line instead of waiting for the
process: `step` records are throttled to about 10 per second (a slow run gets one per batch) and `epoch`
records arrive after each validation. Most of the output is `step`s (in one test, 15 epochs of 50 batches made
about 150 KB): pass `--json-events epoch` to leave them out (about 1 KB per epoch). To stop a run from the
script, `folo stop <run_dir>` (or Ctrl+Break on Windows, Ctrl+C elsewhere) ends it after the current batch with
`status: stopped` (`completed` when it comes after the last batch). Error codes are the same as MCP's (a `--device gpu:7` the machine lacks is
`device_not_found`, a GPU out of memory `gpu_out_of_memory`), and JSON paths use `/` on every OS.

## 6. Run the same flow headless, then look at it in Studio
Point the agent at a running server instead of an embedded store, so its runs show up in FOLO Studio and keep
going after the agent disconnects:

```sh
folo serve                                  # Studio on http://127.0.0.1:7878, data dir ~/.folo
folo mcp --server http://127.0.0.1:7878     # the MCP server the agent starts (same tools)
```

Every recipe above works unchanged; `folo_info` reports `"mode": "proxy"`. Run subscriptions are polled once per
second through the HTTP API. The same data dir also works the other way round: runs an embedded `folo mcp`
trained are listed by a later `folo serve` on the same data dir (only one process should use a data dir at a
time).

## 7. Label an unlabeled folder of images
Have Claude Code (`claude`), Codex (`codex`) or Antigravity (`antigravity`) draw draft boxes, inspect and fix them,
then export a registered YOLO dataset ready for `train` ([LABELING.md](LABELING.md)).

```
→ create_label_job { "name": "street", "classes": ["car", "person"],
                     "instructions": "Label every car and pedestrian.", "paths": ["/data/street"] }
                                                  ← { "job_id": "lb_3f9a", "images": 240, "added": 240, "skipped": 0 }
→ start_labeling { "job_id": "lb_3f9a", "agent": "claude", "image_ids": ["1", "2", "3", "4", "5"] }
                                                  ← { "job_id": "lb_3f9a", "total": 5, "model": ... }
→ label_status { "job_id": "lb_3f9a" }            ← status: "idle", counts, reviewed, run: { done, failed, outcome }, log
→ view_label_image { "job_id": "lb_3f9a" }        ← image: { id: "1", size: [1024, 576], revision: "3-5d01c2e7", boxes }
                                                    + image/jpeg content item with numbered boxes
→ review_label_images { "job_id": "lb_3f9a", "image_ids": ["1"], "expect": { "1": "3-5d01c2e7" } }
                                                  ← { "updated": 1, "to_review": 4 }
→ set_label_boxes { "job_id": "lb_3f9a", "image_id": "2", "expect": "2-1c9e04aa",
                    "size": [1024, 576], "relabel": [[2, 1]], "add": [[0, 120, 80, 340, 400]] }
                                                  ← updated image view (source: "manual", reviewed: true, new revision)
→ start_labeling { "job_id": "lb_3f9a", "agent": "claude" }          # or "codex" / "antigravity"; labels the rest
→ export_label_dataset { "job_id": "lb_3f9a", "val_fraction": 0.2, "seed": 0 }
                                                  ← { "dataset_id": "ds_8c1e", "name": "street", "path": "...", "train": 192, "val": 48 }
```

Start with a small sample (`image_ids`) and check it before labeling the whole folder; if the agent made a systematic
mistake, re-run those ids with `"note"` explaining what to change. Boxes are `[class, x1, y1, x2, y2]` in pixels of
the view's `size` (1024 px on the long side by default). Pass the view's `revision` as `expect` on every
`set_label_boxes` and `review_label_images` call: if another run or user changed the boxes in the meantime, FOLO
returns `label_boxes_changed` instead of applying edits to stale box numbers. The `label_folder` prompt returns this
walk-through as instructions.

CLI (`folo label` shares the same operations and records; without a `folo serve` owning the data dir, `start` waits
until the run ends):
```sh
folo label agents --json                                   # check which CLIs (claude, codex, agy) are installed
folo label create street --classes car,person --instructions "Label every car and pedestrian." \
  --path /data/street --json                               # {"job_id":"lb_3f9a","images":240,"added":240,"skipped":0}
folo label start lb_3f9a --agent antigravity --images 1,2,3,4,5 --wait --json
folo label view lb_3f9a -o next.jpg --json                 # writes next.jpg; prints size, revision and numbered boxes
folo label set lb_3f9a 2 --expect 2-1c9e04aa --size 1024x576 --relabel 2=person --add car,120,80,340,400 --json
folo label review lb_3f9a 1 --expect 1=3-5d01c2e7 --json
folo label start lb_3f9a --agent antigravity --wait --json # label the remaining unlabeled images
folo label export lb_3f9a --val-fraction 0.2 --seed 0 --json
```

## 8. Create and run a vision benchmark
Freeze a labeled split (or labeling job) with a stress-condition plan and score models or external predictions across
accuracy, robustness, self-labeling, calibration and localization ([BENCH.md](BENCH.md)).

```
→ bench_catalog {}                                ← conditions: [{ id, group, description }], severities: 5,
                                                    default_plan: [...], axes: [{ axis, weight }]
→ create_benchmark { "name": "cats-stress", "dataset_id": "ds_8f21", "split": "val", "max_images": 300, "seed": 0 }
                                                  ← { "id": "bn_3f9a0c1e5b7d", "images": 300, "conditions": ..., "entries": 0 }
→ start_bench_entry { "benchmark_id": "bn_3f9a0c1e5b7d", "model_id": "m_4c1a" }
                                                  ← { "benchmark_id": "bn_3f9a0c1e5b7d", "entry_id": "be_91c2", "status": "queued", ... }
→ bench_entry { "benchmark_id": "bn_3f9a0c1e5b7d", "entry_id": "be_91c2", "wait": 50 }
                                                  ← status: "done", result: { score, partial, axes, accuracy: { map50_95, ci95 },
                                                    conditions: [...], ece, mean_iou, self_labeling, ms_per_image }
→ get_benchmark { "benchmark_id": "bn_3f9a0c1e5b7d" }
                                                  ← plan, fingerprint, leaderboard (best first, up to 50 rows)
```

Pass `plan: { "conditions": [{ "condition": "haze", "severities": [1, 3], "weight": 1 }] }` to override
`default_plan`, or `label_job_id` instead of `dataset_id` to benchmark a labeling job's labeled images. To score
another detector, pass `predictions_file` and `name` to `start_bench_entry` (JSON Lines of
`{"image", "condition"?, "severity"?, "boxes": [{"class", "score", "x1", "y1", "x2", "y2"}]}` records; scored at
once). Call `bench_entry` with `wait: 50` in a loop until `result` is present (`stop_bench_entry` cancels a running
entry). Note that benchmark tools are unavailable under `folo mcp --confine` (`not_supported`).

CLI:
```sh
folo bench catalog --json
folo bench create cats-stress --data /data/cats --split val --max-images 300 --seed 0 --json
folo bench run bn_3f9a0c1e5b7d --model m_4c1a --wait --json      # progress records, then finished state with result
folo bench run bn_3f9a0c1e5b7d --predictions yolo.jsonl --name yolo26n --json
folo bench status bn_3f9a0c1e5b7d be_91c2 --wait --json
folo bench show bn_3f9a0c1e5b7d --json                           # benchmark plan and ranked leaderboard
```

## 9. Fine-tune on a rented cloud GPU
Pull published pretrained weights once into the local cache (`folo weights pull`), connect a cloud provider once in
Studio, Desktop or with `folo cloud connect` (MCP never handles credentials), then start a cloud run with `compute`
([CLOUD.md](CLOUD.md)).

```
→ list_models {}                                  ← includes { "id": "folo-n-coco-v5", "source": "published", ... } once pulled
→ cloud_connections {}                            ← { "connections": [{ "id": "cc_4d2a", "provider": "runpod", "status": "ok", ... }] }
→ cloud_offers { "connection_id": "cc_4d2a" }     ← { "offers": [{ "id", "gpu", "vram_bytes", "price_per_hour", "availability" }], "fetched" }
→ train { "dataset_id": "ds_8f21", "weights": "folo-n-coco-v5", "epochs": 50, "imgsz": 640, "amp": true,
          "compute": { "connection_id": "cc_4d2a", "offer_id": "NVIDIA GeForce RTX 4090@secure", "max_hours": 4 } }
                                                  ← { "run_id": "run_7b2e", "status": "queued" }
→ run_status { "run_id": "run_7b2e" }             ← status, epoch, best, cloud: { phase, instance_id, deadline, cost_usd, synced }
→ cloud_instances {}                              ← alive instances first, then those ended in the last 7 days
→ cloud_terminate { "instance_id": "ci_8a2f" }    ← manual termination if needed (the run fails with cloud_instance_lost)
```

Leave `arch` out when `weights` is set: fine-tuning takes the weights' architecture (`"folo-n-coco-v5"` or
`"folo-s-coco"`). `compute.max_hours` (default 12, range 0.25–720) caps spend and arms the worker watchdog. Once in
`training`, a cloud run keeps going even if the MCP session closes; `train { "resume_run_id": "run_7b2e", "compute":
... }` resumes a stopped or lost run on a fresh instance (or `"compute": null` to resume locally).

CLI (`folo weights` and `folo cloud` manage the local cache, connections, offers and instances; `folo train` runs
locally, so launch cloud runs via MCP `train` above or `POST /api/runs` on `folo serve`):
```sh
folo weights pull folo-n-coco-v5 --json
RUNPOD_API_KEY=rpa_... folo cloud connect runpod --secret api_key=env:RUNPOD_API_KEY --json
folo cloud offers cc_4d2a --json                  # one offer per line, cheapest first
curl -s http://127.0.0.1:7878/api/runs -H 'Content-Type: application/json' -d '{
  "dataset_id": "ds_8f21", "weights": "folo-n-coco-v5", "epochs": 50, "imgsz": 640, "amp": true,
  "compute": { "connection_id": "cc_4d2a", "offer_id": "NVIDIA GeForce RTX 4090@secure", "max_hours": 4 }
}'
folo cloud instances --json
folo cloud terminate ci_8a2f --json
```

## 10. CPU int8 calibration and SEBF + DFL-CGR post-processing
For faster CPU inference on AVX-VNNI processors, calibrate a trained model with `folo quantize` and run it with
`--precision int8` ([MODEL.md](MODEL.md#96-int8-inference-opt-in-cpu)). For higher AP on one-to-many (`o2m`) models at any
precision, enable Student-t Exclusive Box Fusion (`--fusion sebf`) with confidence-gated DFL repulsion
(`--score dfl-cgr`) ([MODEL.md](MODEL.md#94-box-fusion-opt-in)).

CLI:
```sh
# 1. Calibrate int8 activation ranges on 512 sampled training images (writes best-int8.folo; never overwrites)
folo quantize --model runs/cats-s/weights/best.folo --data /data/cats --images 512 --method percentile --json
# {"path":"runs/cats-s/weights/best-int8.folo","bytes":...,"images":512,"imgsz":640,"method":"percentile","tail":0.0001,"boundaries":...,"seconds":...}

# 2. Evaluate and predict in int8 on the CPU (--device auto picks cpu; a GPU device is a usage error)
folo val --model runs/cats-s/weights/best-int8.folo --data /data/cats --precision int8 --json
folo predict --model runs/cats-s/weights/best-int8.folo /data/photos --precision int8 --conf 0.25 --json

# 3. Higher-AP post-processing on o2m models (pairs with f32, --half / --precision f16, or --precision int8)
folo val --model runs/cats-s/weights/best.folo --data /data/cats \
  --fusion sebf --score dfl-cgr --iou 0.78 --max-det 1000 --json
folo predict --model runs/cats-s/weights/best.folo /data/photos \
  --fusion sebf --score dfl-cgr --iou 0.78 --max-det 1000 --conf 0.25 --json
```

MCP note: `quantize`, `--precision int8`, `--fusion` and `--score` are CLI flags only — `folo mcp`'s `val` and
`predict` run standard NMS in the server's configured precision (`--device` / `--precision`). Use MCP `val` or
`predict` for the default `f32`/`f16` baseline, then run the CLI commands above when you need calibrated int8 CPU
inference or `sebf` + `dfl-cgr` post-processing:

```
→ val { "model_id": "m_4c1a", "dataset_id": "ds_8f21", "split": "val" }   ← default f32/f16 NMS baseline
```

On CPUs without AVX-VNNI (including AVX-512 VNNI), `--precision int8` falls back to a portable path slower than `f32`
and prints a `note:` on stderr. Models trained with `--head o2o` predict end-to-end without NMS and ignore `--fusion`
and `--score`.
