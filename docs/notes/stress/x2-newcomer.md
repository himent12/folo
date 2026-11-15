# Stress round, lane x2-newcomer: the first hour of a new user

Surface: the first hour of someone who has never seen FOLO, in Studio and in Desktop, from an empty `FOLO_HOME` and an
empty data dir. Base `1436ad68`. Shared prebuilt binaries, servers on the CPU device, lane ports 18130-18133.

## What was tested, and how

- **Studio** (headless Chrome via playwright-core, scripts in the lane dir): every empty state on a fresh data dir
  (Overview, Datasets, Runs, Models, Benchmarks, Playground); New run with no dataset; Generate sample set (200 images,
  320 px); the New run form defaults, then a tiny run (folo-n, 3 epochs) watched to completion; Models, the model page,
  Playground on a val image and on a text file named `.jpg`; Add dataset with a nonexistent path, an images-only folder,
  a drive root, a folder and its `data.yaml`.
- **Desktop** (`FOLO_UI_OFFSCREEN=1`, window messages only): the empty tour on an empty server (Overview Getting started,
  `first_step.rs` cards on Runs, Models, Playground); Ctrl+N and the palette's New training run with no dataset;
  Generate synthetic by click and by Enter; New run defaults after a synthetic set; a dead `--server` port from the
  start; the find-or-start-a-server path (checked first that it runs `folo serve --port 0` on the lane's
  `FOLO_DATA_DIR`, never 7878 or `~/.folo`), a run started there, the app closed mid-training and reopened.
- **Careless**: deleting a dataset while its run trains (API), registering a drive root, the same dataset twice by two
  spellings.
- **Docs**: QUICKSTART's train flags against `folo train --help` and the loader.

## Findings

| id | severity | surface | finding | status |
|---|---|---|---|---|
| x2-newcomer-1 | annoying | Desktop | Ctrl+N / New training run with no dataset does nothing at all | fixed |
| x2-newcomer-2 | polish | Studio | empty Runs, Models, Playground offer New run when no dataset exists (one dead hop; Desktop offers Add dataset) | fixed |
| x2-newcomer-3 | polish | Desktop | Enter in the Generate synthetic dialog does nothing | fixed |
| x2-newcomer-4 | annoying | server | registering a drive root (`D:\`) answers after 389 s: the folder is fingerprinted before format detection | duplicate of x3-server's claim |
| x2-newcomer-5 | polish | server | a YOLO folder and its `data.yaml` register as two datasets | deferred |
| x2-newcomer-6 | polish | Desktop | New run defaults to 640 px right after a 320 px synthetic set (estimate ≈ 1 h 49 min on the CPU) | deferred |
| x2-newcomer-7 | polish | Desktop | server unreachable: Datasets subtitle stays "Loading…", Runs says "0 runs · 0 active · 0 failed" | deferred |
| docs | polish | docs | QUICKSTART says `--workers` defaults to `min(cores, 8)`; it is `min(cores, 16)` | fixed |

### x2-newcomer-1: New run with no dataset silently does nothing (Desktop)

Repro: empty server, `folo-desktop --server http://127.0.0.1:<port>`, press Ctrl+N (or Ctrl+K, New training run).
Nothing happens: no form, no toast, no navigation. Root cause: `apps/desktop/src/app/new_run.rs` `open_new_run`
returned early when the dataset list was empty, the same as while it was still loading. Fix: with a loaded empty list
it goes to Datasets and toasts "Add a dataset first" (Studio's `#/runs/new` says the same); while loading it still
waits. Test: `new_run_without_any_dataset_leads_to_datasets` in `apps/desktop/src/app/tests.rs`. Review round 1:
`--open new-run` against an empty server waited silently, and opened the dialog out of the blue once a dataset was
added (`apps/desktop/src/app/render.rs` consumed the request only for a nonempty list); it now answers once the list
is in, through the same path. Test: `open_new_run_on_launch_without_any_dataset_leads_to_datasets`.

### x2-newcomer-2: Studio's empty pages ask for New run with zero datasets

Runs, Models and Playground empty states hard-coded New run; with no dataset that lands on "Add a dataset first".
Desktop's `first_step.rs` already offers Add dataset in that case. Fix: `web/src/routes/firstStep.tsx`
(`FirstStepAction`, `noDatasetsYet`), used by `Runs.tsx`, `Models.tsx`, `Playground.tsx`: Add dataset
(`#/datasets?add=folder`) when the server answered with no dataset, New run otherwise (also while loading or failed).
Test: `web/src/routes/firstStep.test.tsx` (the three no-dataset cases fail on the old code).

Review round 1 made it hold in every order and over time: `FirstStepAction` is one Button whose link and words follow
`noDatasets` (it picked its branch once, so datasets answering after the runs left it on New run); the pages read their
dataset list again after a dataset action in the tab and when the tab is shown again (`refreshOnDatasetChange`); and
the header's create action of Runs and Overview (past runs, no dataset left) is Add dataset too, as Desktop's headers.
The 11 new cases in the same test file fail with those changes reverted. Re-review 1: Overview's header reads its
dataset list the same way (`refreshOnDatasetChange` in `Overview.tsx`), so the last dataset deleted or the first one
added in another tab turns its action over when the tab is shown again; the two Overview cases fail without it.
Re-review 2: the tab shown again while the first dataset read is still out reads the list again too (that read
supersedes the first, whose old "none" would otherwise land last and stay); the three new cases fail without it.

### x2-newcomer-3: Enter does not submit Generate synthetic (Desktop)

The Name field has focus when the dialog opens; Enter did nothing (Ctrl+Enter and the button worked), unlike Connect
cloud and Studio's form. Root cause: `apps/desktop/src/overlays/synth.rs` `input_box` had no Enter handler. Fix:
`on_key_down` Enter calls `submit_synth`, which validates the fields. Test: `enter_in_the_synth_dialog_generates`
(opens the dialog as against folo serve, sends a real Enter key down, checks the request went with the name).

### Deferred

- **x2-newcomer-4**: `crates/folo-server/src/index.rs` fingerprints the whole folder before it detects the format;
  `POST /api/datasets {"path":"D:\\"}` ends after 389.5 s with a correct `400 unknown_format`. Left to x3-server, who
  claimed it.
- **x2-newcomer-5**: `real/datasets.rs` canonicalizes the path as given, so `<folder>` and `<folder>/data.yaml` are two
  paths. Whether a yaml is "the same folder" (API.md: "the same folder by its canonical path, whatever the spelling")
  is a design call.
- **x2-newcomer-6**: the New run form's image size does not follow the dataset's native size. A default change, not a
  bug fix.
- **x2-newcomer-7**: the Datasets and Runs subtitles do not know the list failed; the cards below them are right
  ("Could not reach folo serve" with Connection settings and Retry). Overlaps the connection lane.
- A run killed with the app (own-started server) reopens as "Stopped" 0/3 with no reason: the runs lane's claim on
  `runs/registry.rs`.

### Checked, not bugs

- Deleting a dataset while its run trains: 204, the run completes and its model is listed.
- Studio Playground: detection on a val image works; a text file named `.jpg` gives "Unsupported image".
- Studio Add dataset: a nonexistent path and an images-only folder get clear messages with a next step.
- Desktop with a dead server: Overview, Runs and Models cards say what happened and offer Connection settings + Retry.
- Desktop's own server: `--port 0` on its own data dir, dies with the app (job object).
- Renaming a registered dataset's folder on disk: after the 5 s re-check `GET /api/datasets/{id}`, its images and
  `POST /api/runs` answer `404 path_not_found` ("the folder of dataset 'my set' is gone", with the hint to move it
  back or unregister); renamed back, the dataset answers 200 again.
- A second `folo serve` on the same data dir: "another FOLO process is using ...", names the running server's URL,
  exits non-zero.
- QUICKSTART §2-§6 literally (synth, stats, train 1 epoch on the CPU, predict, val, export f16 and ONNX) with the
  shared binary: every command works and writes the files the text names.
- Studio export from the model page: FOLO f16 and ONNX both download (4.9 MiB, 9.9 MiB) and the dialog says where
  the file went; no console errors.

## Not done

- Desktop's Playground and export steps of the journey.
- Closing the browser tab mid-upload, Desktop started twice, the server moved to another port under a running app.
- The close-with-a-running-run confirm dialog was seen to hold the window but not captured.

## Wishes

- The New run form says fine-tuning converges faster, but a fresh install has nothing to pick and no hint about
  `folo weights pull`.
