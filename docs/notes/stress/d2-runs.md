# Stress round (dogfood 4): lane d2-runs

FOLO Desktop › Runs, the New run dialog, the live training dashboard and the cloud run pages, driven off screen
(`FOLO_UI_OFFSCREEN=1`, window messages through `apps/desktop/tools/drive.ps1`) against isolated `folo serve`
instances on the CPU (`FOLO_GPU_BACKEND=metal`), `folo serve --mock` for cloud, and the built-in demo (`--mock`,
`--scenario`). Base `1436ad68`. Every value shown was checked against the API JSON and `metrics.jsonl` at the same
moment; every finding was reproduced twice.

## What was tested

- **Runs list**: the seed runs (completed, stopped, failed `data_error`) against `GET /api/runs`; 400 fabricated
  finished runs (scroll, sparklines: 150 MB working set, handles 274 → 828 → 420, back down); an empty data dir
  (Runs and Overview empty states).
- **New run** (Ctrl N): epochs and close-mosaic with 0, -5, 1e9, letters, empty, 20 digits (inline errors, Start
  disabled, no estimate asked); Device GPU 0 on a server without a GPU; Esc closes it; Enter on the
  Stop confirmation keeps training; double submit (Ctrl Enter twice, double click) creates one run
  (`NewRunForm.submitting`); 960×620 (the body scrolls, the footer stays); no dataset at all.
- **Live CPU runs** (folo-n, 160–320 px, 3 epochs on the 240-image shapes set): Epoch, mAP, Precision/Recall,
  throughput, time left and every chart against `metrics.jsonl`; leaving the page mid-training and coming back (36/36
  steps, 3 epochs, Duration = API `started..finished`); a second run queued behind the first (queue text, Start next,
  Remove, the first stopped by API: the second runs within 8 s); a running run deleted (409, page unchanged) and a
  stopped one deleted by API while shown (the page gives way with a toast); `--open run:run_nope`.
- **A run failing mid-way** (its dataset's images deleted after epoch 1): "Failed in epoch 2 of 3" with the io error,
  the failure marker at step 21 of 12 per epoch, the last loss equal to `metrics.jsonl`'s last step, Duration equal to
  the API's `started..finished`, Resume from epoch 2. The loss card's "smoothed 15.348 · EMA 0.98" is the debiased
  EMA of the 21 logged totals (15.3478); the throughput is the last step's `imgs_per_s`.
- **Run names**: a ~434-character name with accents, CJK and emoji is refused in the dialog ("At most 64 characters
  (this one has 434).") and nothing is sent.
- **Server killed by PID mid-epoch** and restarted: "Lost connection… retrying", "Server unreachable", then without a
  click "Interrupted: the server stopped during training" with Resume from epoch 2, within about 20 s.
- **Cloud** (`folo serve --mock`): a live cloud run through Provisioning, Booting, Uploading data (32.0 of 45.8 MiB =
  the API's 38.4 / 48.0 MB), Training, Collecting weights, Releasing, Released, Completed with its toast; its
  instance deleted mid-training (Failed in epoch 3 of 3 with the provider's message, Resume from epoch 3).

## Findings

| id | severity | what | status |
|---|---|---|---|
| d2-runs-1 | annoying | A run's Validation mAP axis reads `0.000042` under `× 10⁻⁵` (= 4.2×10⁻¹⁰): every run whose mAP is under about 0.03 (the first epochs of every real run) gets a power-of-ten caption over labels that are already exact. Root cause `apps/desktop/src/charts/line.rs:201`: the caption ignored the fitted 0–1 axis, whose labels are written in full. | fixed: `charts::axis::label_pow`, test `charts::axis::tests::small_values_share_a_power_of_ten` |
| d2-runs-2 | polish | New run › Device offers GPU 0 on a server whose `/api/system` lists no GPU; Start then answers "Device gpu:0 is not available". `apps/desktop/src/overlays/new_run.rs:192`. | fixed: GPU 0 disabled with the reason as its tooltip (arrow keys skip it), `FoloApp::nr_no_gpu`, test `app::tests::new_run_offers_gpu_0_only_on_a_server_that_has_a_gpu` |
| d2-runs-3 | polish | After the server is killed mid-epoch, the interrupted run's Duration is the last `alive_at` (35 s for ~58 s trained) and the loss chart ends at the last flushed step (12, where 16 had been streamed). `crates/folo-server/src/runs/registry.rs:56-61`. | deferred: server internals, approximate by design |
| d2-runs-4 | annoying | Ctrl N (and the palette's New run) does nothing at all on a server without datasets. `apps/desktop/src/app/new_run.rs:124` returned silently. | fixed: one "Add a dataset first" toast with an Add dataset action, test `app::tests::new_run_on_a_server_without_datasets_says_so_once` |

Repro for each, from a fresh state:

- d2-runs-1: `folo serve` on the seed data dir, `folo-desktop --server … --open run:run_560a` (mAP50 3.9×10⁻⁵ in
  epoch 1): the Validation mAP y axis shows `0.000042` with `× 10⁻⁵` above it.
- d2-runs-2: `FOLO_GPU_BACKEND=metal folo serve …` (gpus = []), Ctrl N, click GPU 0, Ctrl Enter.
- d2-runs-3: 3-epoch CPU run, kill the server PID in epoch 2, restart it, open the run.
- d2-runs-4: `folo serve --data-dir <empty dir>`, Desktop on it, Ctrl N.

## Not reached

Linked hover across charts and the smoothing control on a live run (window-message mouse moves did not raise the
hover readout off screen, so it was looked at only on the demo); memory over a long live run (beyond the 400-run
list); two Desktop windows on the same run (Stop in one).
