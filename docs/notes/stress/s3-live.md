# Stress round, lane s3-live: the Run page while a run trains and after it ends, and Run compare

Base `1436ad68`. Surface: FOLO Studio › Run page (`web/src/routes/Run.tsx`, `run/`) and Run compare
(`RunCompare.tsx`, `compare/`), fed by `web/src/lib/api/live.ts` (the run's SSE stream).

## How it was tested

- The shared prebuilt `folo.exe` of the round, `folo serve --data-dir D:/folo-data/stress/s3-live/... --port 18020`,
  on the CPU (the GPU belonged to a training run), with the embedded Studio; tiny runs (folo-n, 160 to 320 px,
  synthetic 200-image datasets, 1 to 3 epochs).
- Studio driven in headless Chrome (playwright-core, `channel: 'chrome'`): two tabs on the same run, reloads mid-run,
  stop and delete from the other tab, the server killed by PID mid-epoch and restarted, a dataset's images deleted
  mid-epoch, a dataset unregistered while its run waited in the queue, Run compare with a live run, with 2 to 6 runs,
  with a deleted id, the same id twice, an empty list and an encoded `../` id. Every number on the page was read
  against `GET /api/runs/{id}`, `/metrics` and `metrics.jsonl` on disk at the same moment.
- Forged runs (hand-written `run.json` / `metrics.jsonl` / `log.jsonl`) for a log of 6000 lines and non-finite losses.

## Findings

| id | severity | what | status |
|---|---|---|---|
| s3-live-1 | annoying | The Epoch caption says "Mosaic off · final epoch" while epoch 2 of 3 still trains with mosaic, and a run that failed in epoch 2 says "Last 1 epoch: mosaic off" for an epoch it never trained. | fixed |
| s3-live-4 | annoying | The Run page of a run deleted in another tab (or the Runs list, the CLI) shows the last-known run with "Reconnecting" forever, still offering Resume and Open model, while the stream retries a 404 every 15 s. | fixed |
| s3-live-5 | annoying | Run compare shows a compared run that trains as "— · live" with no curve, and keeps saying so after it completed, until a reload. | fixed |
| s3-live-2 | polish | A run interrupted by a server kill reports `step` past its last checkpoint (step 60 with 40 steps flushed); the page shows 50 %, Resume restarts at 33 %. | deferred |
| s3-live-3 | polish | A run whose dataset was deleted shows six sample tiles "Image unavailable · Retry" whose Retry can never succeed, and nothing says the dataset is gone. | deferred |
| s3-live-6 | polish | The Logs header counts the whole log (6000), the Warn / Error filters only the 2000 lines held (Warn 40 of 120). | deferred |

### s3-live-1: mosaic off one epoch early

Repro: a 3-epoch run (close_mosaic 10, capped to 1: epoch 3 trains without mosaic); open its page while epoch 2
trains. Seen `Epoch 2/3 · 50% · Mosaic off · final epoch`; the trainer logs "closing mosaic" only after epoch 2 and the
chart's own mark sits at x = 2. Same run failing in epoch 2 (images deleted): `Last 1 epoch: mosaic off`.
Cause: `run/runModel.ts` `closeText` passed `Run.epoch` to `mosaicOffText`, which takes the epochs done, but `Run.epoch`
is the epoch in progress while queued / running / stopping and the one a failed run failed in (docs/API.md `Run.epoch`).
Fix: `epochsDone(status, epoch)` in `run/readings.ts`. Tests: `runParts.test.tsx` (the readings) and
`run/RunCharts.test.tsx` "says mosaic is off only from the epoch the trainer closes it" (through `RunModel`; fails
without the fix). FOLO Desktop has no such caption.

### s3-live-4: a deleted run reconnects forever

Repro: stop a run, open `#/runs/<id>` in tabs A and B, delete it in A. B keeps the whole run with "Reconnecting";
after 20 s still the same, `GET /api/runs/<id>/events` answers 404 five times then every 15 s. Only a reload says
"Run not found". Cause: `web/src/lib/api/live.ts` LiveRun waited for the stream on a drop, and an EventSource exposes no
status, so the 404 was invisible. Fix: on each drop LiveRun asks `GET /api/runs/{id}` (one request at a time); a 404
clears the run, sets the error (the page's "Run not found") and closes the stream; any other answer leaves the
reconnect to the stream. A failing history load that answers 404 does the same. Test: `live.test.ts` "notices a run
deleted elsewhere" (fails without the fix).

### s3-live-5: Run compare never follows a live run

Repro: start a 2-epoch run, open `#/runs/compare?ids=<it>,<a finished run>` while it trains, let it complete, wait
20 s: still `— · live`, no curve, and the axis note "one run trained far longer" stays. Cause: `RunCompare.tsx` read
the runs and their metrics once. Fix: while a compared run is queued, running or stopping the runs are read again every
3 s, and a run's metrics again whenever its status or epoch moves. Test: `RunCompare.test.tsx` "a run that trains
while compared" (fails without the fix).

### Deferred

- **s3-live-2.** `crates/folo-server/src/runs/history.rs` `reconcile` returns early when the epochs done did not change,
  so the `step` that the 10 s `alive_at` write of `run.json` carried survives the restart. A run stopped by the user
  also keeps `step` past its epoch (12 steps per epoch, step 15), so this may be the documented meaning of `step`:
  a server-semantics decision, not a page fix.
- **s3-live-3.** Deleting a dataset leaves its files and lets queued runs on it train; the run page cannot tell
  "dataset gone" from a slow image. Needs the run to know its dataset is unregistered (server) before the page can say it.
- **s3-live-6.** By design of `run/Logs.tsx` (older lines live in `log.jsonl`, said above the first line); an honest
  Warn / Error count needs a server-side count per level.

## Checked, no bug

- Completed run: KPIs, best epoch, precision / recall, log count and per-class rows match `metrics.jsonl` and the API.
- Server killed by PID mid-epoch, restarted: "Interrupted: the server stopped during training", Resume offered; Resume
  continues from the last finished epoch.
- A run failing mid-way: "Failed in epoch 2 · Files of dataset midfail went missing", Check dataset and Resume offered.
- Bad weights are refused at `POST /api/runs` (`404 model_not_found`), no run is created.
- Non-finite losses never reach the page from a real server: the trainer skips such batches
  (`crates/folo-vision/src/train/epoch.rs`).
- Run compare: the same run twice draws it once; a deleted id is skipped and the others drawn; an empty list opens the
  picker; an encoded `../` id is a 404 and ignored.
- Two tabs on one live run: both follow; Stop in one shows in both; a double-clicked Stop sends one stop.

## Not reached

A tab in the background for ten minutes, chart hover and keyboard focus on a one-point chart, narrow widths (360 px)
of the Run and compare pages, lineage after the parent model is deleted, and the queue position of a queued run's page.
