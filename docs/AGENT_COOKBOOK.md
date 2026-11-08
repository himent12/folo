# Agent cookbook

Short recipes for driving FOLO from an agent, each in two forms: **MCP tool calls** (`folo mcp`, docs/MCP.md) and
the **CLI with `--json`** (docs/CLI.md). Tool calls are written `→ tool { arguments }` with the interesting part
of the result after `←`; every result also has a one-line text summary. Errors are `isError` results whose text is
`{ "code", "message", "hint" }` (MCP) or the last stdout line `{"error":{...}}` plus a non-zero exit code (CLI).

Setup: `folo mcp --data-dir D:/folo-data` (README, "Use FOLO from Claude Code / Codex"). That is the real engine:
training, `val`, `predict` and `export` run in the `folo mcp` process, and runs persist under
`D:/folo-data/runs/`. Add `--device cpu` to keep inference off the GPU, `--mock` to try the tools on simulated
data, or `--server http://127.0.0.1:7878` to share runs with FOLO Studio.

## 1. Train a detector on a folder of images
The folder must be a labelled dataset: YOLO (`images/` + `labels/`, or a `data.yaml`) or COCO (an instances
`.json`). No labels yet? `create_synthetic_dataset` makes a shapes dataset to practise on.

```
→ folo_info {}                                    ← devices.gpus, scales (params, GFLOPs at 640 / 320)
→ register_dataset { "path": "D:/data/cats", "name": "cats" }
                                                  ← { "id": "ds_8f21", "classes": ["cat", "dog"], "splits": {...} }
→ get_dataset { "dataset_id": "ds_8f21" }         ← stats (class counts, box sizes), issues (see recipe 4)
→ train { "dataset_id": "ds_8f21", "arch": "folo-s", "epochs": 50, "imgsz": 640, "device": "auto" }
                                                  ← { "run_id": "run_4c1a", "status": "queued" }
→ run_status { "run_id": "run_4c1a" }             ← status, epoch / epochs, best, eta_s, last_epoch.val, log
```

Instead of polling `run_status` (every 30 s is plenty), subscribe to `folo://runs/run_4c1a`; the server sends a
resource-updated notification (at most one per second) on every step, epoch and status change. When the status is
`completed`, `model_id` names the best weights (`m_4c1a`; `m_4c1a-last` is the last epoch). The prompt
`train_on_folder { "folder_path": "D:/data/cats" }` returns these steps as instructions.

Stop and resume: `stop_run { "run_id": "run_4c1a" }` stops after the current batch (status `stopping`, then
`stopped`, the last epoch's checkpoint kept). `train { "resume_run_id": "run_4c1a" }` continues the same run from
that checkpoint (optimizer, schedule, EMA and RNG included); leave the other fields out, only `device` may change.
Runs are also stopped this way when the MCP client disconnects, so a new session can resume them. To train further
after a run **completed**, start a new run with `"weights": "m_4c1a"`.

CLI:
```sh
folo train --data D:/data/cats --model folo-s --epochs 50 --imgsz 640 --json > train.jsonl
# one {"event":"status"|"step"|"epoch"|"log",...} record per event, then
# {"event":"done","status":"completed","run_dir":"runs/cats-s","best":{"epoch":41,"map50_95":0.62},
#  "weights":{"best":"runs/cats-s/weights/best.folo","last":".../last.folo"}}
folo train --resume runs/cats-s/weights/last.folo --json   # after Ctrl+C (or a killed process)
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
folo val --model runs/cats-s/weights/best.folo --data D:/data/cats --split test --json > a.json
folo val --model runs/cats-n/weights/best.folo --data D:/data/cats --split test --json > b.json
jq -s 'map({model, map50, map50_95, precision, recall})' a.json b.json
```

## 3. Find label problems in a dataset
```
→ get_dataset { "dataset_id": "ds_8f21" }
    ← stats: { class_counts, box_size_hist, empty_images, broken_labels },
      issues: [{ "kind": "out_of_range_box", "file": "labels/train/0412.txt", "message": "..." }, ...] (first 50),
      issues_total
→ list_dataset_images { "dataset_id": "ds_8f21", "class": 1, "include_boxes": true, "limit": 20 }
→ (read folo://datasets/ds_8f21/images/{image_id} to look at an image)
```

`get_dataset` lists issues in file order, so on a dataset with many background images the 50 it returns can
be mostly `empty_image` infos while an error further down is left out: when `issues_total` is over 50, run
`folo dataset check` (below) for the full list. Look for: errors (`broken_label`, `class_overflow`, `out_of_range_box`, `corrupt_image`, `missing_image`; the
loader skips those images), warnings (`clipped_box`, `zero_size_box`, `duplicate_label`, `duplicate_image`), a
class with far fewer boxes than the others, and many tiny boxes for a small `imgsz`. After training, `val`'s
`per_class` and the sample grid show which classes the labels let down. FOLO never edits a dataset; fix the files
and the next `get_dataset` re-indexes them.

CLI (every issue, not just 50, with a suggested fix for each):
```sh
folo dataset check D:/data/cats --fix-dry-run --json   # exit 0 when clean, 1 with errors or warnings
# {"clean":false,"counts":{"error":3,"warning":12,"info":40,"by_kind":{...}},
#  "issues":[{"kind","severity","file","line","message","fix"}, ...]}
folo dataset stats D:/data/cats --json                 # the same stats object as get_dataset
```

## 4. Export and run inference
```
→ export { "model_id": "m_4c1a", "format": "folo-f16" }
    ← { "path": "D:/folo-data/exports/cats-s-best-f16.folo", "bytes": 18874368 }
→ predict { "model_id": "m_4c1a", "image_path": "D:/photos/garden.jpg", "conf": 0.25 }
    ← { image: { width, height }, boxes: [{ class, name, score, x1, y1, x2, y2 }], timing: { ..., total_ms }, device }
→ predict_batch { "model_id": "m_4c1a", "glob": "D:/photos/*.jpg" }
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
folo predict --model runs/cats-s/weights/best-f16.folo D:/photos --conf 0.25 --save annotated --json
# {"event":"prediction","file":"D:/photos/garden.jpg","image":{...},"boxes":[...],"timing":{...},"saved":"..."}
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

done = folo("train", "--data", "D:/data/cats", "--model", "folo-n", "--epochs", "20", "--imgsz", "320")[-1]
best = done["weights"]["best"]
[report] = folo("val", "--model", best, "--data", "D:/data/cats")
print(f"mAP50-95 {report['map50_95']:.3f} on {report['images']} images")
for r in folo("predict", "--model", best, "D:/photos"):
    if r.get("event") == "prediction":
        print(r["file"], [(b["name"], round(b["score"], 2)) for b in r["boxes"]])
```

`folo dataset check` is the one command that exits 1 with a useful report (issues found), so read its records
before treating exit 1 as a failure. For a long `folo train`, read stdout line by line instead of waiting for the
process: `step` records are throttled to about 10 per second (a slow run gets one per batch) and `epoch`
records arrive after each validation. Most of the output is `step`s (in one test, 15 epochs of 50 batches made
about 150 KB); keep only the `epoch` and `done` records in an agent's context.

## 6. Run the same flow headless, then look at it in Studio
Point the agent at a running server instead of an embedded store, so its runs show up in FOLO Studio and keep
going after the agent disconnects:

```sh
folo serve --data-dir D:/folo-data          # Studio on http://127.0.0.1:7878
folo mcp --server http://127.0.0.1:7878     # the MCP server the agent starts (same tools)
```

Every recipe above works unchanged; `folo_info` reports `"mode": "proxy"`. Run subscriptions are polled once per
second through the HTTP API. The same data dir also works the other way round: runs an embedded `folo mcp`
trained are listed by a later `folo serve --data-dir D:/folo-data` (only one process should use a data dir at a
time).
