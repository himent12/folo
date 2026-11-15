# Stress round, lane x3-server: the server under both apps

Lane x3-server (dogfood 4) drove `folo serve` (the shared prebuilt binary of the round, CPU device, isolated data dir
and ports 18140-18149) the way FOLO Studio and FOLO Desktop drive it, and harder, with Node scripts (raw `http` and
`fetch`, so bodies, headers and sockets could be cut and forged), then looked at how each app shows the failure
(Studio in headless Chrome, Desktop with `--shot` against the damaged server).

## What was pushed

- Every route the front ends call (from `web/src/lib/api/` and `apps/desktop/src/api/http/`): malformed JSON, wrong
  types, missing fields, wrong content type on 11 POST routes; 100 MB bodies; 1000-character, unicode, `..%2F..`,
  `%00` and `con` ids on 11 id routes.
- Dataset registration: missing paths, files, UNC, a drive root and system folders, the folder above a dataset,
  names with spaces, unicode and 1000 characters, 20 parallel creates and deletes of one name.
- Uploads (25 cases): traversal, absolute, drive, backslash, ADS, symlink, hard-link, device entries; garbage and
  truncated archives; parts out of order, huge part numbers, retries, complete early and twice, a part cut off
  half-way (socket destroyed), a delete during a slow 30 MB part; models of 0 bytes, garbage, truncated; hostile
  model names; 20 parallel uploads with one fingerprint.
- Weights and images: Range (`0-0`, `10-`, past the end, invalid, several), `If-None-Match`, `If-Match`; thumb sizes
  0 to 100000; image list limits and offsets; predict on the 22 hostile files of the seed kit.
- `--token`: 61 routes x 4 bad headers; the token in responses and in the log.
- Kills by PID at 7 moments after a run was created, then restart; 20 parallel resumes and deletes of a running run.
- SSE: 50 subscribers on one run, 25 vanishing after 3 s, one never reading.
- 14 damaged copies of a data dir (truncated and empty `run.json`, missing and half weights, garbage and truncated
  `datasets.json`, a dataset folder gone, `runs` as a file, junk files, a run folder with the wrong or a duplicate
  id, garbage side files): server start, its log, a Studio and a Desktop screenshot each.

## Findings

| id | severity | what | status |
|---|---|---|---|
| x3-server-1 | blocker | Registering a folder that is not a dataset (a drive, `C:/Windows/WinSxS`) stats every file below it before it finds out (386 s), and every dataset delete waits for it meanwhile | fixed |
| x3-server-7 | blocker | A truncated `datasets.json` loads no dataset and `GET /api/datasets` answers an empty list: both apps say "No datasets yet" while every registration is refused | fixed |
| x3-server-6 | annoying | Registering the folder above a dataset says "has 40 images but no labels — label them first" | fixed |
| x3-server-9 | annoying | A copy of a run folder next to it (a backup) lists the run twice under one id | fixed |
| x3-server-4 | annoying | A dataset named `nul` cannot be made ("does not exist", upload 500) | duplicate of s1-datasets-5, fixed there |
| x3-server-5 | polish | A dataset archive with a file and a folder of one name answers `500 io_error` | fixed |
| x3-server-2 | polish | A token sent as `?token=` (never accepted) is written to the server log | fixed |
| x3-server-8 | polish | A damaged weights file is warned about on every `GET /api/models` (both apps poll it) | fixed |
| x3-server-3 | polish | 20 parallel synthetic sets named `race`: names and folders do not match (`race` lives in `race-6`) | deferred |

### x3-server-1: a non-dataset folder is walked before it is refused

Repro: `POST /api/datasets {"path":"C:/Windows/WinSxS"}`, 2 s later `DELETE /api/datasets/nope`. The register
answered `400 unknown_format` after 386 s, the unrelated delete 404 after 383 s (with `\\127.0.0.1\c$` neither
finished in 5 minutes). Cause: `Index::build` (`crates/folo-server/src/index.rs`) fingerprinted the whole folder (a
recursive walk following links, a stat per file) before `folo_data::open_with` detected that there is no dataset, and
`register` holds `RealStore::files` shared for all of it while `delete_dataset` takes it exclusively. Fix: a cheap
`folo_data::detect_format` first. No regression test: the difference is time on a big tree; the existing
registration-error tests cover that the answer is unchanged.

### x3-server-7: a damaged registry is shown as an empty one

Repro: copy a data dir, cut `datasets.json` in half, start: `GET /api/datasets` answered `200 {"datasets":[]}`;
Studio's Datasets screen said "No datasets yet" and offered Register folder, which then failed; the only clue was one
ERROR line in stderr. Cause: `RealStore::open_with_cloud` (`real/mod.rs`) loads an empty list on a registry error and
`list_datasets` (`real/store.rs`) served it as real. Fix: while nothing is loaded, the list reads the registry again
and answers its `500 internal` "… is malformed" with the hint "fix the JSON by hand or move the file away". Test:
`tests/real_datasets.rs` `a_damaged_registry_is_an_error_not_an_empty_list`.

### x3-server-6: the folder above a dataset is sent to labeling

Repro: `photos-yolo` copied to `wrapped/my-set/`; `POST /api/datasets {"path":".../wrapped"}` said "'.../wrapped' has
40 images but no labels", hint "label them first". Same for an upload whose archive wraps the dataset in a folder.
Cause: `data_error` (`index.rs`) counted images below without looking for a dataset below. Fix: `dataset_below` looks
one and two levels down (64 folders a level) and names it: "'…/wrapped' is not a dataset folder, but
'…/wrapped/my-set' inside it holds a dataset", hint "register '…/wrapped/my-set'". Test:
`a_folder_above_a_dataset_names_the_dataset_inside`.

### x3-server-5: file and folder of one name in an archive

Repro: a tar with `images/` (folder) then `images` (file), or `x` then `x/y.txt`: `500 io_error "creating …: Access is
denied"`. Fix: `file_folder_clash` in `uploads.rs` `unpack` refuses it as `422 invalid_upload` "images is both a file
and a folder in the archive", and the session is dropped like for the other hostile archives. Test:
`tests/uploads.rs` `a_file_and_a_folder_of_one_name_are_refused`.

### x3-server-2: `?token=` in the log

Repro: `folo serve --token …`; `GET /api/runs?token=…` answers 401 and stderr had `request{method=GET
uri=/api/runs?token=… }`. Cause: the trace layer's `DefaultMakeSpan` records the whole URI. Fix: `request_span`
(`lib.rs`) records method and path only, like `log_request` already did.

### x3-server-9: a copied run folder is listed twice

Repro: `runs/run_17a6` copied to `runs/run_17a7` (a backup next to it); on start `GET /api/runs` listed run_17a6 twice
and both apps showed two identical rows; nothing in the log. Cause: `Runs::open` (`runs/registry.rs`) loaded every
folder's `run.json` without looking at the id. Fix: one run per id, from the folder named after it (else the first by
name); the copies are skipped with a WARN naming both folders and are not touched. Test: `runs::tests`
`a_copied_run_folder_is_loaded_once`.

### x3-server-8: one warning per listing

Fix: `Models::warn_skipped` (`models.rs`) remembers each skipped file's version (path, size, modification time) and
warns once per version, for run models, the models folder and published weights. Test: `models::tests`
`a_damaged_weights_file_is_warned_about_once`.

### Deferred

- x3-server-3: the synthetic dataset's folder is claimed (`claim_dataset_dir`) and its name made unique
  (`Naming::Unique`) independently, so under parallel creates `race` can live in `datasets/race-6`. Nothing is mixed
  up (each dataset's id, name and folder belong together); tying the two is a design choice.

## Checked, not a bug

- Malformed bodies, wrong types and content types: 400/415/422 envelopes, no 5xx; 100 MB bodies: 413, fast.
- Bad ids on every id route: 404 envelopes. Synthetic names with traversal: slugged inside `datasets/`.
- `--token`: every `/api` route but health refuses a missing or wrong token; no token in any response.
- Kill -9 shortly after a run starts: on restart the run is `stopped` + `interrupted` with a resume hint; resume works;
  20 parallel resumes give one 200 and nineteen 409.
- Uploads: every hostile archive above is 422 with the session and its folder gone; a cut-off part leaves the session
  at its last whole part; a delete during a part answers the part 404 and removes the folder.
- Weights Range and ETag handling as documented (no `If-Range`; no client sends it).
- SSE: 50 subscribers with vanishing and stalled clients: the others got every event; handles back to normal.
- Damaged data dirs: a truncated or empty `run.json` is skipped with a WARN naming the file; missing weights drop the
  models and keep the run; a dataset folder gone stays listed with "the folder of dataset 'x' is gone".

- A second `folo serve` on one data dir exits with `data_dir_in_use` naming the running server's URL; a taken port is
  `bind_failed` with a hint; after a kill -9 the stale `serve.lock` is taken over at once.

## Not done

Kills during an upload, a label save and synthetic generation; memory after several runs (575 MB working set after
one tiny run, not followed up); `runs` as a file starts with no runs and no warning (not dug into).
