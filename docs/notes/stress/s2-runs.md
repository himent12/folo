# Stress round (dogfood 4), lane s2-runs: Runs list, the queue, New run

Surface: FOLO Studio › Runs (`web/src/routes/Runs.tsx`, `runs/`), the run queue, and New run
(`web/src/routes/RunNew.tsx`, `runnew/`). Base `1436ad68`. Shared prebuilt `folo.exe` of the round, CPU device,
isolated data dirs, headless Chrome through playwright-core (scripts under `D:/folo-build/stress/lanes/s2-runs/scripts/`).

## What was tested, and how

- **Every New run field pushed past its edge** (headless Chrome, typing as a user): epochs 0, -1, 1e9, 1.5, empty,
  letters, 10001; names in unicode, 500 characters, only spaces, duplicates, `--amp`, quotes; learning rate 0, 10,
  1e-46, 1e-300; `?from=` retry links with junk (`imgsz=33`, unknown run). The Start button, the message under each
  field and the POST body (or its absence) were checked against the server's answer.
- **Submitting**: double-click Start, Ctrl+Enter twice, two tabs at once, Start then Back before the answer, Start
  with the server down then back up.
- **The CLI command the form shows**: the server's `cli` copied and run for real with the shared `folo.exe` (aug
  flags, `--hsv '0,0,0'`, `--name=--amp`, a dataset path with a space) in Windows PowerShell 5.1: it parses, trains,
  and its `config.json` matches the run the form would have started.
- **The queue**: ten queued runs on a mock server, reordered through the row menu (Move up / down, Start next),
  stop a queued run, stop the running one, delete in several orders; the UI's places against the API's
  `queue_position` after every step. Restart of a real server with a queue (killed mid-run).
- **The Runs table with 307 runs** (mock server, runs made through the API): sort every column, filter, search
  with regex characters and 2000 characters, heap and DOM size over time, request counts.
- **A dataset with no val split** (YOLO folder with `images/train` only).
- **Cloud "Run on" with the mock server only** (RunPod connection `cc_4d2a`): one run through every phase, then
  Stop in provisioning, booting, syncing, training and collecting.

## Findings

| id | sev | what | status |
|---|---|---|---|
| s2-runs-1 | annoying | New run's name field turns every non-ASCII letter into a dash (`ünïcødé-名前` → `n-c-d`, `名前` refused); lengths counted in UTF-16 units. Desktop and the server keep them. Root cause `runnew/name.ts` (`\w` is ASCII). | fixed |
| s2-runs-2 | polish | Start then Back (or another page) before the POST answers: when it answers the app jumps to the new run from wherever the user went. Root cause `RunNew.tsx` `router.go` after `await createRun`. | fixed |
| s2-runs-3 | polish | Learning rate below the f32 range (`1e-46`) passes the form, the server refuses it as "got 0". Root cause `runnew/jump.ts` checked `lr0 > 0` in f64. | fixed |
| s2-runs-4 | annoying | Runs list with more than 200 runs refetches every run's sparkline metrics on every re-render (1310 requests for five filter/clear cycles). Root cause `lib/api/metricsCache.ts` FIFO bound of 200. | duplicate: x4-soak owns it |
| s2-runs-5 | annoying | A row's ⋯ menu near the window's bottom opens below the window: Delete run / Remove from queue unreachable (wheel closes it). Root cause `lib/ui/Menu.tsx` `place()` never flips. | fixed |
| s2-runs-6 | annoying | New run says nothing when the dataset has no val split, though every epoch's mAP is then on the training images (docs/API.md "Split fallback"). | fixed |
| s2-runs-7 | polish | `#/runs/new?from=<unknown run>` toast shows the server's agent hint ("List_runs (MCP) or GET /api/runs…") and stays after the link is gone. | open |
| s2-runs-9 | annoying | Found in review: a start answered after the user left and opened New run again deletes the tab's draft, so the newer form's edits are lost on reload (one sessionStorage key per tab). | fixed |
| s2-runs-8 | polish | Mock server only: a cloud run stopped in provisioning or booting reports `cloud.ended_in: "syncing"` (`crates/folo-server/src/mock/cloud.rs:227`); the real executor records the right phase. | deferred: mock fidelity |

Polish noted, not filed as a bug: after a server restart with a queue every waiting run is `stopped`
(`interrupted`), as documented; runs created in the same second can change list order after the restart.

### The fixes

- **s2-runs-1** `web/src/routes/runnew/name.ts`: names keep letters, marks and digits of every script
  (`\p{L}\p{M}\p{N}`), and lengths are counted in characters as the server does
  (`crates/folo-server/src/store.rs`). Test `runnew/name.test.ts`.
- **s2-runs-2** `web/src/routes/RunNew.tsx`: after the run is created the app opens it only while that form is still
  mounted and the URL's route (`router.location`, not the route on display, which App holds on the form while the
  next screen's chunk loads) is still the link Start was pressed under; otherwise the toast says it started. Tests
  `runnew/bindings.test.tsx`: leave before the answer, leave while the next screen's chunk loads, leave and open New
  run again.
- **s2-runs-9** `web/src/routes/RunNew.tsx`, `runnew/draft.ts` (review round 1): a start answered after the user left
  and opened New run again cleared the tab's one draft, the newer form's edits with it. A start now clears the draft
  only while it is still the one its form wrote (`clearOwnDraft`). Test `runnew/bindings.test.tsx`.
- **s2-runs-3** `web/src/routes/runnew/jump.ts`: lr0 is checked as the server's 32-bit float (`Math.fround`), so a
  value that rounds to 0 is refused in the form. Test `runnew/bindings.test.tsx`.
- **s2-runs-5** `web/src/lib/ui/Menu.tsx`: a fixed list opens above its button when it would leave the window below
  and fits above; where it fits on neither side (a short window) it takes the side with more room, its height capped
  at the window's edge, and scrolls inside itself. Test `lib/ui/Menu.test.tsx`.
- **s2-runs-6** `web/src/routes/runnew/valSplit.ts`, `DatasetSection.tsx`: a note under the dataset picker says a
  dataset with no val images is scored on its training images (mAP measures fit); it describes both pickers (cards
  and the phone select) and is announced politely when a pick brings it. Tests `runnew/DatasetSection.test.tsx`
  (renders the picker, picks a train-only dataset, then one with a val split) and `runnew/valSplit.test.ts`.
  SIBLING: Desktop's New run (`apps/desktop/src/overlays/new_run.rs`, dataset picker) shows the counts only; not
  changed here.

Every regression assertion was run against the unfixed code and failed there (round 1's against the round-0 code:
`logs/r1-counterfactual.txt` in the lane dir). Some tests also hold controls that pass either way (the menu that
fits below, the form that is still showing opens the run).

## Checked and fine

Epoch, name and lr validation messages and the disabled Start; one POST for a double-click and for Ctrl+Enter twice;
server down on Start (toast, form kept, retry works); the server's CLI command runs as shown; queue reorder / stop /
start next through the row menu match the API; sort and filter on 307 rows in 250-500 ms, no heap growth after GC;
every cloud phase shown as the API reports it, and Stop in each phase ends in `stopped` with the instance terminated
and the right words ("Stopped before it started", "Stopped during its first epoch", "Stopped at epoch 2 … Resume").

## Not done

- A revoked cloud key: the mock cannot revoke a stored key (only a `bad` secret at creation, which is Settings').
- Draft restore with invalid values after a reload; the time estimate at epochs 10000 / imgsz 1280.
- Narrow widths (390, 768) of New run's summary sheet.
- s2-runs-7 (small; the hint text is the server's).
- Desktop's sibling of s2-runs-6 (no train-only note beside its dataset picker): a Rust change in `apps/desktop`,
  left to a Desktop lane.
