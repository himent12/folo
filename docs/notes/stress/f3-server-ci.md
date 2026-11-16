# Stress round, follow-up lane f3-server-ci: the server items and the round's red CI

A follow-up lane, no hunting: five items the round's lanes and reviewers confirmed but left open because they sat in
another lane's surface (`crates/folo-server`, its tests, the CI workflows). Each was checked on main (069935d6), fixed
at the root with a regression test that fails without the fix, or documented.

| id | item | severity | status |
|---|---|---|---|
| f3-server-ci-1 | A cloud run whose instance dies while its final weights are copied ends `cloud_sync_failed` ("answered 502 Bad Gateway ... the instance is gone"), not `cloud_instance_lost` | annoying (wrong error, red CI on three PRs) | fixed |
| f3-server-ci-2a | `tests/cloud/harness.rs` `finished` gave a whole run a fixed bound: eight tests "timed out waiting for run_... to end" on a loaded Windows runner | annoying (CI) | fixed |
| f3-server-ci-2b | `tests/bench/external.rs` `open_entries_expire_without_parts` raced a 300 ms open timeout against the wall clock | annoying (CI) | fixed |
| f3-server-ci-3 | `npm run check` / `npm run qa` killed by SIGTRAP / SIGILL (V8 "Fatal error in , line 0") on the self-hosted Linux runners | annoying (CI) | mitigated: retried once on a crash signal |
| f3-server-ci-4 (d1-datasets-3) | `broken_labels` documented as label lines in `folo-api`, counted per broken label file (YOLO) or entry (COCO) by both producers; apps say "broken label files" | annoying | fixed (docs, wording "N broken labels") |
| f3-server-ci-5 (x3-server) | After a restart loaded a labeling job's backup copy because the original did not load, the next restart loads the original again, hiding the work saved to the copy | blocker (work hidden) | fixed for labeling jobs; benchmarks and runs deferred |

## f3-server-ci-1: a gone instance is a lost one, whichever request notices

Mechanism: `cloud/executor/train.rs` sends the worker a stop and, when the first failed poll comes later than half
of `lost_grace` after it (a loaded runner), takes the "did not stop in time; collecting" branch. `collect` then copies
`weights/last.folo` from the instance; the proxy answers `502 ... the instance is gone`, and `mirror.rs` ended the run
with `cloud_sync_failed`. docs/CLOUD.md says an instance the provider reports gone fails the run with
`cloud_instance_lost` (resumable from the checkpoints copied so far), so the product was wrong, not the test.

Fix: `cloud/executor/mirror.rs` asks the provider when a weights copy fails; `Gone` or `Stopped` is
`cloud_instance_lost` with the copy's error and "resume the run" in the message. docs/CLOUD.md (Failure) says so.

Test: `tests/cloud/failures.rs` `an_instance_lost_while_its_weights_are_collected_loses_the_run`. A new harness gate
(`Gates::weights`) holds every `/weights` request at the proxy; the test stops the run, waits until the collection's
copy is held, kills the instance, then lets the copy through: no timing involved. It failed without the fix
(`cloud_sync_failed`). It takes 40-60 s, as the client's transfer retries a 502 for about 15 s per request.

Both front ends (review round 1, SIBLING): a run that failed while `collecting` read as "training failed, weights
collected" unless its code was `cloud_sync_failed`, so the new `cloud_instance_lost` there showed the collection as
done. Studio `web/src/lib/cloud.ts` `phaseStates` and Desktop `apps/desktop/src/cloud.rs` `phase_states` now treat
both codes as a failed collection. Tests: `web/src/lib/cloud.test.ts` and Desktop
`app/run_page_tests.rs` `a_cloud_run_that_failed_uploading_did_not_train` (also a worker failure, `oom`, still marks
training failed).

## f3-server-ci-2: tests that wait for the event, not the clock

- `tests/cloud/harness.rs` `finished`: a run that still moves (status, step, epoch, queue place, cloud phase,
  transfer progress, checkpoint copy, log length) is waited for however long the runner takes; only one that stands
  still for `TIMEOUT` fails, and the message says what it stood still as.
- `tests/bench/external.rs` `open_entries_expire_without_parts`: the server's sweeper takes its `now` from the caller
  (`bench/mod.rs` `expire`), and `RealStore::expire_bench_entries_at(now)` (test support, doc hidden) runs one look
  at a given instant. The test sets a one-hour timeout, which the sweeper's own looks never reach, and looks at
  `opened` (nothing expires) and at `opened + timeout` (the entry without parts expires, the one fed after `opened`
  does not).

## f3-server-ci-3: Node crashes on the self-hosted Linux runners

The crashes (exit 133 "Trace/breakpoint trap" with V8's `# Fatal error in , line 0 / unreachable code`, once exit
132 in a QA shard) came on unrelated heads in one evening, on the WSL runners of the host whose CPU already crashes
rustc (docs/CI.md, Unstable self-hosted Windows host). `tools/ci-retry.sh` now also counts as a crash a process
killed by SIGILL, SIGTRAP, SIGBUS or SIGSEGV (exit 132 / 133 / 135 / 139, npm's `signal SIG...` line, V8's banner),
and `web.yml` runs `npm run check`, `npm test` and `npm run qa` through it with two attempts on self-hosted runners
only. A type error, a failing test or a QA finding exits 1 and is never retried. Checked by hand: a `kill -TRAP` and
a `kill -SEGV` are retried once, `exit 1` is not. Not determined: whether the CPU or that Node build is at fault
(nothing to run on the runners from here); a crash that repeats on the retry would point at Node.

## f3-server-ci-4 (d1-datasets-3): what `broken_labels` counts

Both producers already count the same thing: images left out because of a broken label. The real server takes
folo-data's count (one `BrokenLabel` load issue per skipped image, `folo-data/src/dataset/mod.rs`), the mock adds one
per broken label file. docs/API.md (Datasets) says the same. Only `folo-api`'s doc comment ("label lines") and the
mock builder's (`src/stats.rs` `add_broken`) said lines. The "32 broken label files for 8 files" of d1-datasets-3 was
32 skipped val images (their files name a class beyond `names`) beside the 8 kept ones; `/issues` lists every broken
line, so it shows more errors than files.

Review round 1 found that for COCO the count is not one of images: folo-data's `BrokenLabel` issues also cover a
malformed category, an image or annotation entry without its ids and an annotation file that is not COCO JSON
(`folo-data/src/dataset/coco.rs`), each dropped without leaving an image out. Counting images apart from those would
need a new issue kind through folo-data's report and scan cache, so the contract now says what is counted: labels
that cannot be read, not images (docs/API.md, `folo-api` `DatasetStats::broken_labels`, folo-data
`LoadReport::broken_labels`). Desktop (`screens/datasets/health.rs`, `strip.rs`) and Studio
(`routes/dataset/health.ts`) say "N broken labels" (before: "N broken label files", wrong for COCO; the first
version of this PR said "images with broken labels", wrong for COCO's dropped entries).

Tests: `tests/issues.rs` `real_broken_labels_count_the_images_left_out_not_the_lines`, a YOLO dataset with two broken
files holding four broken lines: `broken_labels` 2, two images fewer, four errors in `/issues`; and
`real_coco_broken_labels_count_the_entries_dropped`, a COCO dataset with a malformed category and an annotation
without `image_id`: `broken_labels` 2, no image left out.

## f3-server-ci-5 (x3-server): the copy written last is loaded

`util.rs` `folders_by_id` put the folder named after the id first among readable records. A job whose original did
not load once (its recovery could not be written back) was loaded from its copy and saved there; at the next restart
the original loaded again and the copy's work was hidden. `job.json` now carries `saves`, the job's write count
(`label/job.rs`; 0 for files from before), and `folders_by_id` takes the readable record written most often first,
then the old order. A copy of an old backup put back beside the live folder has fewer saves and does not win (a
file-time order would have let it).

Review round 1: a backup older than the original (fewer saves) that was loaded once and saved to still lost at the
next restart, as its count went on from its own. `Job::load` now takes the group's most saves
(`util::most_saves`, given to the loader by `load_first`) and counts on from there, so the first save to a loaded
backup outranks every copy; a backup loaded but never saved to leaves the original first, as it holds the newer work.

Test: `label/job.rs` `the_copy_written_last_is_loaded_after_the_original_failed_once`: the original saved to 10, the
backup at 2; loaded once without a save the original stays first; loaded and saved once the backup wins.

Deferred: benchmarks, and runs (sibling, `runs/registry.rs` prefers the original once it reads again). `bench.json`
is written once; a benchmark's work lives in its `entries/` and `loops/` files, so a count in `bench.json` would not
move, and a run's `run.json` has no count either. Fixing them needs a per-record write counter written with every
piece of work, or a marker written when a copy is loaded in place of its original that the selection reads first:
a design decision for those stores, beyond this lane's labeling-job fix.

## Gates

All passed locally: root `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test -p folo-server -p folo-api`; apps/desktop fmt, clippy and `cargo test --workspace`; web `npm run check`,
`npm test`, `npm run build`; `bash tools/file-size.sh`. While the apps/desktop gates built: 20 of 20 runs of
`open_entries_expire_without_parts`, 20 of 20 of `an_instance_lost_while_its_weights_are_collected_loses_the_run`
(four at a time), 3 of 3 of the whole cloud suite. `f3-server-ci-4`'s test also passes on main: it pins the contract
the producers already kept. The `web.yml` change is only exercised by CI.
