# Stress round, lane d6-connection: Desktop connecting, starting a server, surviving a bad one

Surface: FOLO Desktop (`apps/desktop`) › finding or starting `folo serve`, the HTTP transport and SSE streams
(`src/api/http/`), the list fetching and live event pump (`src/app/data.rs`), and the unreachable and reconnecting
states of every screen. Base commit `1436ad68`.

## How it was tested

- Desktop off screen (`FOLO_UI_OFFSCREEN=1`), driven with window messages (`tools/drive.ps1`), the shared prebuilt
  `folo.exe` / `folo-desktop.exe`, CPU-device servers on the lane's ports, `FOLO_HOME`, `FOLO_DESKTOP_STATE`,
  `FOLO_DATA_DIR` and `FOLO_SERVER` always pointing at the lane (the find-or-start path was read first:
  `src/main.rs` FOLO_SERVER default, `FoloApp::autoconnect`, `server.rs` `spawn_server`), stderr of every launch kept.
- A small Node proxy in front of the server with switchable modes: pass, 10 s delay, 500 (empty and HTML), garbage,
  truncated JSON, wrong types, connection reset, hang, an SSE stream that stalls after its headers, malformed SSE
  events; with path filters so one endpoint misbehaves while the rest pass.
- Scenarios: `--server` to the lane server; the server killed by PID while idle, while a run trains (tiny200, folo-n,
  320 px, 3 epochs) and restarted; Desktop opened before the server exists; auto mode with `FOLO_BIN` missing, a
  directory, a text file, a console program that is not FOLO, a script that never prints, a script that prints a
  listening record and dies; auto mode with `FOLO_TOKEN` in the environment; a `--token` server; the spawned server
  killed by PID; window close and hard kill of Desktop (process tree checked: no `folo.exe` left behind).

## Findings

| id | severity | what | status |
|---|---|---|---|
| d6-connection-4 | blocker | Desktop opened before the server: once it answers, Runs and Overview keep "Could not reach folo serve" forever (only Retry helps); Models show every model as "Unknown dataset" | fixed |
| d6-connection-5 | blocker | a run's event stream that goes silent (stuck proxy, NAT timeout) freezes the run page for good, also after the run ended: one loss point, no final mAP, reopening the page does not help; a reader thread leaks per stall | fixed |
| d6-connection-1 | blocker | a `FOLO_TOKEN` in the environment makes the server Desktop starts for itself require a token Desktop cannot send: every screen says "start it without --token", closing the window asks instead of closing | fixed |
| d6-connection-2 | polish | `FOLO_BIN` naming a file that is no program shows Windows' raw "This version of %1 is not compatible ..." text | fixed |
| d6-connection-6 | polish | a run killed in its first seconds shows "Duration 0 s" (server reconcile ends it at its last `alive_at`, `crates/folo-server/src/runs/registry.rs`) | deferred: server internals, an approximation by design |
| d6-connection-3 | annoying (candidate) | a server that dies right after printing "listening" is restarted on every failed refresh: `connection.rs` calls `recovered()` when a server starts, resetting the back-off | deferred: reproduced only with a fake script, not with a real `folo` |

### Repro and fix

- **d6-connection-4.** Nothing on the port; `folo-desktop --server http://127.0.0.1:18110`; start `folo serve` on
  18110. Status line turns green, Datasets and Models reload, Runs and Overview stay red. Cause: `refresh_lists`
  reset a failed datasets or models list to Idle but only refreshed the run list when Ready. Fix: a failed run list
  goes back to Idle too (`src/app/data.rs`). Test: `connection_tests.rs`
  `every_list_that_failed_is_read_again_once_the_server_answers`.
- **d6-connection-5.** Proxy in `sse-stall` mode on `/events` only; start a run; open its page. Cause: the stream
  agent had no per-read timeout (`src/api/http/transport.rs`), so the reader thread blocked in `read` forever and the
  pump never saw the stream end. Fix: `timeout_per_read` of 45 s on the stream agent (`STREAM_SILENCE`,
  `src/api/http/mod.rs`; the server sends a keep-alive every 15 s, docs/API.md), and the pump body moved into
  `drain_streams`, which ends a run whose dropped stream the run list already reports as ended (so the final history
  loads) and removes finished subscriptions at once (an end is handled once). Tests: `streams.rs`
  `a_stream_that_goes_silent_ends`; `connection_tests.rs`
  `a_stream_that_ends_silently_after_its_run_finished_still_ends_the_run`, `a_delivered_end_is_handled_once`.
  Review round 1 widened it: an end the stream never delivered also reads a loaded log again and marks the latest
  samples for a new read (an epoch picked on the page stays); a stream that dropped mid-run and could not be opened
  again stays in `stream_lost` until it opens, or until the run list says the run ended, which then handles that end
  (`settle_lost_streams`, called from `install_runs`). Tests: `a_run_that_ends_while_its_lost_stream_cannot_reopen_still_ends`,
  `a_silent_end_reads_the_log_and_latest_samples_again`. SIBLING: Studio's `web/src/lib/api/sse.ts` relies on
  `EventSource.onerror` alone and has no silence deadline (deferred to the Studio side: it needs a fetch-based SSE
  reader with an idle deadline, and lane s3-live claims the same `web/src/lib/api/live.ts` stream code).
- **Review round 1, reconnect.** Pages that failed with `server_unreachable` while the server was away (a dataset
  and its images, issues, a model's evaluation, a run's history, samples or log, a benchmark's list, catalog,
  agents, entries or loops) kept their error after it came back: their loaders fetch only Idle entries. `reconnected`
  (`src/app/connection.rs`) now drops those failures so the page asks again. Test:
  `a_page_that_failed_while_the_server_was_away_loads_once_it_is_back`.
- **Review round 2, late failures and final history.** A request sent while the server was away that failed only
  after it answered again (a timeout still running at the reconnect) installed its failure after the sweep, for
  good; `fetch` (`src/app/data.rs`) now counts reconnects and clears such late `server_unreachable` failures too.
  A finished benchmark entry's result (`bench.results`, no Retry on its page) is in the sweep. An ended run whose
  final history read failed over a partial one kept the partial charts (its stream gone, nothing asked again); it
  is now read again at each run-list refresh until it is in (`history_retry`, `settle_lost_streams`), the end
  announced once. Tests: `a_request_that_fails_after_the_server_is_back_is_asked_again`,
  `a_benchmark_result_that_failed_while_the_server_was_away_is_asked_again`,
  `a_final_history_that_failed_is_read_again_at_the_next_refresh`.
- **Review round 3, cut bodies.** A response whose headers arrived but whose body was cut or stalled (the server
  killed, a proxy dropping the link) read as `bad_response` ("Unexpected response"), which no reconnect sweep clears:
  a finished benchmark's result kept "The result didn't load" with no Retry. `body_failed`
  (`src/api/http/transport.rs`) now makes a body that breaks in transport `server_unreachable` (JSON, images,
  weights), and keeps a body that arrived whole but is no JSON of the right shape a bad response. Test:
  `http::tests::a_body_cut_off_midway_is_unreachable_and_a_bad_body_is_not`.
- **d6-connection-1.** `FOLO_TOKEN` set, `FOLO_SERVER` at a dead port, auto mode. Fix: `serve_command` removes
  `FOLO_TOKEN` from the started server's environment (`src/api/http/server.rs`). Test:
  `the_started_server_never_inherits_a_token`.
- **d6-connection-2.** `FOLO_BIN` = a text file renamed `.exe`. Fix: `start_failed` maps os error 193 / 216 / ENOEXEC
  to "FOLO_BIN names X, which isn't a program this PC can run". Test: `a_file_that_is_no_program_is_named_so`.

## Checked and fine

`FOLO_BIN` missing or a directory (falls back next to the executable and finds the lane server via
`data_dir_in_use`); `FOLO_BIN` a console program that is not FOLO ("stopped before it was listening"); a script that
never prints ("did not start in time" after 15 s, child killed); server killed while idle (banner, lists kept,
reconnects in under 5 s); server killed mid-training and restarted (Interrupted banner right, live pump recovers);
auto mode: the spawned server killed (a new one started, same data dir); window close and hard kill leave no server
behind (job object).

## Not reached

Two Desktops on one server and on one data dir in auto mode; 10 s delay, wrong types and truncated JSON walked
through every screen (only the run list and the run page were walked); URL forms in Settings › Connection (trailing
slash, path, https, unresolvable host: lane d4 owns Settings); exit with requests in flight; handle growth over many
respawns (one job-object handle per spawn, `server.rs`, not measured).
