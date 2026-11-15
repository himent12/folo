# Stress round, lane d3-models: Desktop Models, model page, Playground and local inference

Base `1436ad68`. FOLO Desktop (shared prebuilt release binary) driven off screen with window messages
(`FOLO_UI_OFFSCREEN=1`, `tools/drive.ps1`), against `folo serve` on the CPU (`FOLO_GPU_BACKEND=metal`) with a copy
of the seed data, and against `folo predict --json` for the same model, image and thresholds. Local GPU was checked
on single images only (the GPU was busy training).

## What was tested

- Models list against `GET /api/models`; a model's page (headline tiles, PR curves, class table, confusion matrix)
  against `GET /api/models/{id}/eval`; Export ONNX twice in a row.
- Playground with each Run on target (Server, Local GPU, Local CPU) and the CLI on the same images: the EXIF-6
  image, a CMYK JPEG, an 8000x6000 PNG, a 1x1 PNG, shapes validation images; the seed kit's odd files (zero-byte,
  truncated, GIF, text named .jpg, 16-bit, WebP) through the server and the CLI. Boxes, scores and class names agree
  (EXIF-6: umbrella 0.58, 183x263 on all three targets and the CLI); GIF and the text file are refused by both.
- Switching the target, the model and the image while a run is in flight (8000 px image): the newest answer wins.
- Threshold sliders at both ends, dragged and stepped with the arrow keys (held down), and "Use best F1" (0.001:
  300 detections, Server and Local CPU agree).
- A model's weights replaced on the server (Local reloads on the next run: ETag revalidation), the server killed
  (the loaded copy still runs; banner "Lost connection"), the shown model's run deleted on the server (Playground
  falls back to another model and runs it; the model page goes back to Models with a toast).
- 100 local runs in a row (four images in turn, Local CPU): working set 130 -> 199 MB after the first 20, then flat;
  handles 269 -> 291 and threads 78 -> 81, flat.

## Findings

| id | severity | what | status |
|---|---|---|---|
| d3-models-1 | annoying | Local CPU/GPU: every superseded run (IoU arrow key held, slider steps) still decodes and infers, one after the other; the wanted answer comes 6-15 s after the last key instead of after one inference | fix written, not in this PR |
| d3-models-2 | annoying | Model page of a published model: headline tiles stay "–" after it was evaluated, while the caption names the evaluated dataset and split | deferred |
| d3-models-3 | polish | Playground header "300 detections ≥ 0.00" while the threshold is 0.001 | fix written, not in this PR |
| d3-models-4 | polish | Models list: mAP50 at 3 decimals ("0.000") beside mAP50-95 widened to 6 ("0.000008") reads as mAP50 < mAP50-95 | deferred |
| d3-models-5 | polish | 80-class confusion matrix: one-letter column heads, the last overlaps "Missed"; 1280 px window: class table truncates "circle" to "cir…" | deferred |
| d3-models-6 | polish | A 1x1 image fills the stage and its thumbnail as a radial glow fading to black (linear upscale against a dark edge) instead of a flat square | deferred |

### d3-models-1: superseded local runs all execute

Repro: Playground, model folo-n-coco, a validation image (EXIF-6, or the 8000x6000 one), Run on Local CPU; when the
result is in, focus the NMS IoU slider and press Right 10-25 times quickly. Each press runs `run_predict`
(`apps/desktop/src/controls/slider.rs:73`), which schedules a local prediction on its own pool thread (threads
76 -> 96). They queue on the predictor's mutex and all run; the app only dropped their answers
(`play_pred_answered`, seq check). 25 presses on EXIF-6: the answer 10-15 s after the last key (one run 100-500 ms);
10 presses on the 8000 px image: 6-10 s (each decodes the 144 MB image again). On Local GPU the same queue lands on a
GPU busy training. Fix: `local::Ticket`, one per Playground request on the app's line; `local::predict` checks it
before anything, again under the load lock before revalidating the weights (each queued run asked the server again,
up to the 2 s connect / 15 s answer timeout apiece against a server out of reach), and `run` again after taking the
predictor lock, answering `superseded` (an answer the app drops by its seq anyway). Test: `local::tests::superseded_runs_queued_behind_the_predictor_are_skipped`. SIBLING: the Server
target queues every `/api/predict` on the server too (not changed: the server is another lane's area).

### d3-models-2: published model's headline tiles stay "–" (deferred)

`Model.metrics` is `null` for `source: published`, and the tiles read only it
(`apps/desktop/src/screens/models/detail.rs:141,221,234`), while the caption under mAP50-95 names the evaluation's
dataset and split. `ModelEval` has no overall metrics (`crates/folo-api/src/models.rs:202`): the fix is either an
overall block in `/eval` (API change) or a client-side class mean, which must match the server's averaging rule
(classes without ground truth). A design call: deferred.

### d3-models-3: "≥ 0.00" at a 0.001 threshold

`apps/desktop/src/screens/playground/stage.rs:178,363` formatted the threshold with two decimals while the slider
and every other surface use `fmt_conf` (three where the third carries information). Fix: `fmt_conf` (written, not in this PR; see below).

## Fixes not in this PR

The code fixes for d3-models-1 and d3-models-3 (and one README sentence on skipped superseded local runs) are written but were never compiled: no build slot came in three turns (both queues 4-5 deep for 25-36 minutes each turn). Gates come before a push, so the code is not pushed. It lives uncommitted in the lane worktree `D:/folo-wt/stress-d3-models` and as a patch at `D:/folo-build/stress/lanes/d3-models/d3-models-fixes.patch` (`git apply` on this branch; run `cargo fmt --all` in `apps/desktop` after it, because the patch reorders one `use` line in `local.rs`). Files: `apps/desktop/src/local.rs` (`Ticket` and the test), `src/app/playground.rs`, `src/app/mod.rs`, `src/screens/playground/stage.rs`, `apps/desktop/README.md`.

## Not reached

Weights download interrupted mid-way through the real app (the HTTP layer's cut-download test,
`api::http::tests::weights_revalidate_by_etag_and_a_cut_download_is_unreachable`, covers the mechanism), two windows
on the same model. Ctrl+O with the real file dialog on the sandbox desktop was checked with the odd files: the text
file named .jpg, the GIF, the zero-byte and the truncated JPEG and README.txt each give a toast "<name> is not an image
FOLO can read · Use a JPEG, PNG, BMP or WebP file" and keep the image on screen; the unicode and the double-space
names, and the WebP, open and run. `folo serve --mock` with Local CPU was checked: the mock's random-weight file loads and
runs (0 detections, by design of `crates/folo-server/src/mock/weights.rs`).

## Wishes

- The Local GPU button reads greyed out for a few seconds after launch while the GPU is probed, and a click then does
  nothing: a "Looking for a GPU…" hint would say why.
