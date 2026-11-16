# Stress round, lane s4-models: Studio Models, the Model page and the Playground

Base: `origin/main` 1436ad68. Shared prebuilt `folo` (ENV.md), `folo serve` on the CPU device with the embedded Studio,
seed data copied to the lane's data dir, headless Chrome (playwright-core, `channel: 'chrome'`). No GPU training; the
in-browser WebGPU check is single-image inference.

## What was tested and how

- **Every way an image gets in**, with the seed kit's odd files (zero bytes, text named `.jpg`, truncated JPEG, 1x1,
  8000 px, CMYK, 16-bit and 8-bit gray, palette PNG, animated GIF, WebP, a 30 MB PNG, EXIF orientations 1-8, unicode
  and spaced names): all 22 through the file picker (`setInputFiles`) and `POST /api/predict`; a subset by drop (a
  dispatched `DataTransfer`: the unicode JPEG, the text named `.jpg`, the zero-byte JPEG, GIF + WebP together, a
  text-and-link-only drop) and paste (a dispatched `ClipboardEvent`, never the real clipboard: WebP, the spaced JPEG,
  text only). The other file × drop / paste combinations were not run.
- **Results against the CLI**: `folo predict --json` (CPU) for the same model, image and thresholds against the
  server's answer, for folo-n-coco on four COCO photos and with the thresholds at both ends (conf 0.01, IoU 0.05 and
  0.95, max 1 and 300): the sorted lists of class name and score (rounded to 0.01) were compared, box coordinates
  were not. The Playground's rows (server, wasm CPU, WebGPU f16) were recorded and their count checked against the
  server's; they were not compared box by box.
- **Races**: a second image while the first decodes (8000 px); twenty images at 100 ms and 700 ms gaps (requests
  aborted, the last image's answer shown, heap watched); switching the model mid-inference; deleting the run whose
  model the Playground uses, in a visible and in a hidden tab.
- **Models and the Model page**: the table against `GET /api/models` (published weights, imported files, run models,
  zero metrics), the evaluation, per-class table and confusion matrix against `/eval`, a model whose dataset is
  unregistered or whose folder was moved away, unknown and odd ids in the URL and the API (`..%2F..`, NUL, 5000
  characters, emoji).
- **Export and download**: every ONNX switch combination and f16 (status, `0x08` ModelProto header, Content-Length,
  half about 50% of the size, a bad `format`), downloads aborted at 50 ms, 400 ms and after the first chunk (no partial
  file left in `cache/weights`, the next full download valid), Range requests (206 / 416 / 200 as docs/API.md says).

## Findings

| id | severity | surface | finding | status |
| --- | --- | --- | --- | --- |
| s4-models-1 | annoying | Model page, Details | A run stopped after epoch 1 of 50: "Trained for 50 epochs at 320 px" and "Weights from: last finished epoch" (no number). `trained_with.epochs` is the planned count. | fixed in this PR; SIBLING Desktop `apps/desktop/src/screens/models/details.rs:142` |
| s4-models-2 | polish | Playground intake | A truncated JPEG: "could not be opened as an image. Use a JPEG, PNG, BMP or WebP file." for a JPEG. The server reads it (200, 640x480); the browser's `createImageBitmap` refuses incomplete data. | fixed in this PR (message); SIBLING Desktop `apps/desktop/src/app/playground.rs:97-99` (a damaged PNG is called unsupported) |
| s4-models-3 | polish | model downloads | `?format=onnx&dynamic=true` downloads as `folo-n-coco.onnx`, the static graph's name (different bytes); `folo-n-coco-f16` exports as `folo-n-coco-f16-f16.folo` / `-f16.onnx`. | deferred: naming is a design decision across Studio, Desktop and the CLI |
| s4-models-4 | annoying | Model page, `GET /api/models/{id}/eval` | Model of a run whose registered dataset folder was moved away: `404 path_not_found` "'D:/…/datasets/shapes' does not exist" with the hint "pass the dataset root: the folder holding data.yaml, …", which the user cannot do here. `GET /api/datasets/{id}` answers the right "the folder of dataset 'shapes' is gone … move the folder back". | fixed in this PR: `crates/folo-server/src/real/models.rs` `eval_dataset` dropped the entry's `folder_gone` error and fell back to the run's recorded path, whose index build failed with the registration error of `src/index.rs:188-192` |

### Repros

- **s4-models-1**: seed data, open `#/models/m_c527-last` (`GET /api/runs/run_c527` → `status: stopped, epoch: 1,
  epochs: 50`). Before: "Trained for 50 epochs at 320 px", "Weights from last finished epoch". After: "1 of 50 epochs
  at 320 px", "last finished epoch, 1 of 50, same weights as best". The count (`finishedEpochs`) is a `completed` or
  `stopped` run's `Run.epoch`, which the server brings up to `weights/last.folo` after a crash between the checkpoint
  and its history line (`crates/folo-server/src/runs/history.rs` `reconcile`; re-review 1); for any other status the
  highest epoch of the run's history (`GET /api/runs/{id}/metrics`) when `Run.epoch` is not past it (review round 1:
  a failed resume that never trained again, a queued resume; an early-stopped completed run has `epoch < epochs`).
  Past it, or when it is not known (loading, the run or a needed history did not load, a checkpoint copied out of its
  run), Details says "50 epochs planned at 320 px". The page reads the run and its history again when the models
  change (a run resumed in another window that ends). Tests: `web/src/routes/model/weightsEpoch.test.ts`,
  `web/src/routes/Model.test.tsx` (`Model: how long it trained`).
- **s4-models-2**: Playground, Browse files → a JPEG cut in half. After: "truncated-half.jpg could not be decoded: the
  file may be truncated or damaged." (the format is told by the first bytes, as elsewhere in `intake.ts`). Test:
  `web/src/routes/playground/intake.test.ts` (`calls a file of a readable format that does not decode damaged…`).
- **s4-models-4**: seed data; move `data/datasets/shapes` away; open `#/models/m_c527-last` (cached or not). After:
  the registered dataset's "the folder of dataset 'shapes' is gone … move the folder back"; unregistered too: "the
  folder of dataset 'shapes' this model trained on is gone", hint "move the folder back … or POST /api/val". The
  Model page's evaluation panels show "Move the dataset's folder back where it was, then Retry." in place of that
  hint, which names endpoints and so stays under Technical details (re-review 1). Tests: the end of
  `crates/folo-server/tests/real_runs/training.rs` (`train_evaluate_predict_and_reload`: moved away, unregistered,
  moved back); `web/src/routes/Model.test.tsx` (`Model: evaluation when the dataset folder is gone`).

### Checked, not bugs

- All 22 odd files through the picker, and the drop / paste subset above: the image or a clear toast; EXIF 1-8
  upright with the same box (by eye, on the screenshots).
- Server and `folo predict --json`: identical sorted class-and-score lists (0.01) at every threshold tried; the
  Playground showed as many rows as the server answered, on each backend. Coordinates and the in-browser backends'
  scores were not compared detection by detection (see Not done).
- Twenty images in quick succession: earlier requests aborted, the last image and its own answer shown, heap ~9 MB.
- Deleting the model in use: the picker follows to another model; a hidden tab refreshes on return.
- Deleting a run removes its exports in `cache/weights`. Interrupted downloads leave nothing behind.

### Notes, not filed

- A dataset whose images are gone still lists its old split counts (datasets lane's area; known broken seed dataset).
- A run model whose dataset is unregistered: the Models list says "split unknown" while the Model page says "val"
  (its `/eval` provenance); by design of the list's summary.
- A drop of a link or text only (no file) does nothing (Chrome's image drags carry a File).

## Not done

- Desktop's siblings of s4-models-1 (`apps/desktop/src/screens/models/details.rs:142-149`) and s4-models-2
  (`apps/desktop/src/app/playground.rs:97-99`).
- An ONNX validity check beyond the header (no onnx runtime on this machine).
- Box coordinates against the CLI, and the in-browser backends against the server detection by detection.
- The odd files not in the drop / paste subset, by drop and by paste.
