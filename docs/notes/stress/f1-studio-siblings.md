# Stress follow-up f1-studio-siblings: Studio siblings of Desktop findings

Follow-up lane of the stress round (no hunting). Base: main 069935d6. Surface: FOLO Studio (`web/`). Each item below was
confirmed by another lane or its reviewer and left open because it sat outside that lane's surface. Every item was
re-checked on main before fixing; none had been fixed by another lane (s5-label and s6-shell had touched
`liveLabel.ts`, not the 404 path). Testing was by vitest with fake timers and mocked fetch; no server or browser was run.

## Fixed

| id | origin | severity | what | files | test |
|---|---|---|---|---|---|
| f1-studio-siblings-1 | d6-connection review | blocker | A stream that delivers events and then goes silent without closing stayed "live" forever: EventSource fires no error and hides the server's keep-alive comments. `openEventStream` now reads the stream with `fetch`; any byte (the 15 s `: keep-alive` included) is a sign of life; after `STREAM_SILENCE_MS` (45 s, three keep-alives missed) the stream is aborted, the view says reconnecting, it reopens with backoff and `reconnect` makes the run, benchmark and labeling views read their history again. An answer that is not `text/event-stream` counts as a failed connection. Sibling of Desktop PR #273. | `web/src/lib/api/sse.ts`, `live.ts` (comment), `web/README.md`, `docs/ARCHITECTURE.md` | `web/src/lib/api/sse.test.ts` |
| f1-studio-siblings-2 | d2-runs | annoying | New run restored from a draft with Device `gpu:0` on a server listing no GPU let Start through; only the server's refusal set the error. A kept GPU missing from `/api/system` is now a Device error before Start (local runs only), as Desktop's `nr_device_gone`. | `web/src/routes/runnew/device.ts` (`missingDevice`), `web/src/routes/RunNew.tsx` | `web/src/routes/RunNew.test.tsx` |
| f1-studio-siblings-3 | s3-live review | annoying | A labeling job deleted in another tab left its page on "Reconnecting" forever. A 404 on the job check (after a dropped stream) or on the history load now ends the stream and shows not found, as `LiveRun`. | `web/src/lib/api/liveLabel.ts` | `web/src/lib/api/liveLabel.test.ts` |
| f1-studio-siblings-4 | s3-live-3 | polish | The Run page read the datasets list once, so a dataset deleted while the page was open was not shown as gone. The list is read again every 5 s. | `web/src/routes/Run.tsx` | `web/src/routes/runDatasetGone.test.tsx` |
| f1-studio-siblings-5 | s2-runs-7 | polish | `#/runs/new?from=<unknown run>` toasted the server's agent hint ("list_runs (MCP) or GET /api/runs ...") and the toast outlived the link. `errorDetail` leaves agent hints out (`forAgents`, now in `toast.ts`, shared with `ErrorState`); a `run_not_found` says "Run to copy not found"; the toast is keyed and cleared when the form goes. | `web/src/lib/ui/toast.ts`, `web/src/lib/ui/ErrorState.tsx`, `web/src/routes/RunNew.tsx` | `web/src/routes/RunNew.test.tsx` |
| f1-studio-siblings-6 | d1-datasets review | polish | "1 images" in dataset health; the same plural slip in the New run summary, Overview's also-running card, the Label jobs list and the command palette. | `web/src/routes/dataset/health.ts`, `runnew/summary.ts`, `overview/AlsoRunning.tsx`, `LabelJobs.tsx`, `lib/shell/CommandPalette.tsx` | `web/src/routes/dataset/health.test.ts` |

## Review round 1 (PR #287)

- f1-2 compared the device as a string, so a run created over the API with `gpu:00` (or `GPU:0`, `cuda:0`, `gpu`,
  all read by the trainer as GPU 0) was called gone on copy. `canonicalDevice` (`runnew/device.ts`) now names every
  spelling `resolve_device` accepts as the select does; the copy, the draft and `missingDevice` use it
  (`runnew/device.test.ts`, `RunNew.test.tsx`).
- f1-5 dropped a whole hint that named an endpoint, remedy and all ("label images first: POST /api/…/start, or draw
  boxes by hand"). `humanHint` (`lib/ui/toast.ts`) keeps the clauses a user can act on and drops the endpoints; toasts,
  dialog feedback (`errorDetail`) and `ErrorState` use it; the full hint stays in ErrorState's details
  (`lib/ui/toast.test.ts`).
- SIBLING of f1-4 on New run: the form read the datasets once, so one removed in another window stayed picked with
  Start on. It now polls the list alone every 5 s (the request the Run page polls) and reads every dataset's stats
  again only when the list changed, and again on a `dataset_not_found` from Start; the picker names the removed dataset
  and Start stays off (`RunNew.test.tsx`).
- f1-6 left "1 images" in the Label jobs footer and in six more Studio strings (labeling agent progress, the New run
  estimate's validation row, the loop dialog's job options, a bench entry's running pill, accuracy and self-label
  captions): all use `count` now (`label/labeling.test.tsx`). The same slip in Desktop (`apps/desktop/src/ui/format.rs:102`,
  `screens/models/confusion.rs:344`, `screens/bench/page.rs:57`, `screens/bench/entry.rs:81`,
  `screens/bench/entry_images.rs:134`) is SIBLING, deferred: outside this Studio lane.
- f1-3's tests now fail on main: the 404 closes the stream before the page goes, late events are ignored, and a 404 on
  the history load of a `reload` ends the job without the re-sync retrying.

## Re-review 1 (PR #287)

- The New run datasets follow (`followedDatasets`, `lib/api/datasetReads.ts`) re-read the stats whenever the shown
  ids differed from the polled list and nothing was in flight. A failed re-read kept the old ids and cleared
  `refreshing`, so it started again at once, without bound (and while the tab was hidden); stats read ahead of the list
  (Start's `dataset_not_found`) were read again and again until the next poll. Each list answer is now looked at once:
  a failed re-read is tried again at the next poll, stats newer than the list wait for it
  (`lib/api/datasetReads.test.ts`: 303 list requests in 3 s and 45 extra stats reads before, 3 polls and none after).
- Disconnecting a cloud connection in use lost the whole remedy of its hint ("stop its runs and wait until
  GET /api/cloud/instances shows its instances terminated": the endpoint is the subject). `userHint` (`lib/ui/toast.ts`)
  gives such a code plain words ("stop its runs and wait until its instances are terminated"); toasts, dialog feedback
  and `ErrorState` use it (`lib/ui/toast.test.ts`). It is the only server hint of that shape today.

## Notes and what stays open
- The deadline (45 s) follows the server's keep-alive interval (15 s, docs/API.md) and Desktop's choice in PR #273. A
  server or proxy that strips SSE comments and sends no events for 45 s gets its stream reopened every 45 s: harmless
  (the view re-reads its history) but visible as a short "reconnecting".
- `fetch` streams send the page's cookies like EventSource did (same origin); no auth header was involved before or after.
- The datasets poll costs one `GET /api/datasets` per 5 s per open Run or New run page; a server-pushed datasets
  change would be better but is a design change (deferred).
- Not run: the app against a live server, a real stalling proxy. The behaviour is covered by fake-timer tests only.

## Gates
`npm run check`, `npm test` (224 files, 1888 tests at re-review 1), `npm run build` in `web/`: pass. `bash tools/file-size.sh`: pass.
No Rust was touched.
