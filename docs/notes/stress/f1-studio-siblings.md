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

## Notes and what stays open
- The deadline (45 s) follows the server's keep-alive interval (15 s, docs/API.md) and Desktop's choice in PR #273. A
  server or proxy that strips SSE comments and sends no events for 45 s gets its stream reopened every 45 s: harmless
  (the view re-reads its history) but visible as a short "reconnecting".
- `fetch` streams send the page's cookies like EventSource did (same origin); no auth header was involved before or after.
- The Run page's datasets poll costs one `GET /api/datasets` per 5 s per open Run page; a server-pushed datasets
  change would be better but is a design change (deferred).
- Not run: the app against a live server, a real stalling proxy. The behaviour is covered by fake-timer tests only.

## Gates
`npm run check`, `npm test` (221 files, 1875 tests), `npm run build` in `web/`: pass. `bash tools/file-size.sh`: pass.
No Rust was touched.
