# Stress round, follow-up lane f3-server-ci: the server items and the round's red CI

A follow-up lane, no hunting: five items the round's lanes and reviewers confirmed but left open because they sat in
another lane's surface (`crates/folo-server`, its tests, the CI workflows). Each was checked on main (069935d6), fixed
at the root with a regression test that fails without the fix, documented, or (item 5) deferred.

| id | item | severity | status |
|---|---|---|---|
| f3-server-ci-1 | A cloud run whose instance dies while its final weights are copied ends `cloud_sync_failed` ("answered 502 Bad Gateway ... the instance is gone"), not `cloud_instance_lost` | annoying (wrong error, red CI on three PRs) | fixed |
| f3-server-ci-2a | `tests/cloud/harness.rs` `finished` gave a whole run a fixed bound: eight tests "timed out waiting for run_... to end" on a loaded Windows runner | annoying (CI) | fixed |
| f3-server-ci-2b | `tests/bench/external.rs` `open_entries_expire_without_parts` raced a 300 ms open timeout against the wall clock | annoying (CI) | fixed |
| f3-server-ci-3 | `npm run check` / `npm run qa` killed by SIGTRAP / SIGILL (V8 "Fatal error in , line 0") on the self-hosted Linux runners | annoying (CI) | mitigated: retried once on a crash signal |
| f3-server-ci-4 (d1-datasets-3) | `broken_labels` documented as label lines in `folo-api`, counted per broken label file (YOLO) or entry (COCO) by both producers; apps say "broken label files" | annoying | fixed (docs, wording "N broken labels") |
| f3-server-ci-5 (x3-server) | After a restart loaded a labeling job's backup copy because the original did not load, the next restart loads the original again, hiding the work saved to the copy (benchmarks and runs too) | blocker (work hidden) | deferred: a design decision (lead, after review round 3); kept on `stress/f3-copy-selection` |

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
without `image_id`: `broken_labels` 2, no image left out; then two annotations with a bad `bbox` on one image
(review round 2: the COCO loader keeps one reason per image, so they count once): `broken_labels` 3, one image fewer.
The contract says so: for COCO one per image left out for annotations that cannot be read, however many it has.

## f3-server-ci-5 (x3-server): which copy of a folder is loaded after a restart (deferred)

Taken out of this PR by the lead's decision after review round 3; the work is kept on branch
`stress/f3-copy-selection` (no PR). For copy selection this branch behaves exactly as main: `util::folders_by_id` +
`load_first` for labeling jobs and benchmarks (the folder named after the id first among records the loader takes),
`util::one_per_id` for runs.

Scenario: a job, benchmark or run folder has a hand-made copy beside it (`lb_1` and `lb_1 - Copy`). At one start the
original does not load (its record unreadable, or its interrupted-run recovery cannot be written back), so the copy
is loaded and the user's work is saved there. At the next start the original loads again and wins by name: the work
saved to the copy is hidden (still on disk, logged as a skipped copy). It needs both the hand-made copy and an
original that fails at one restart and reads at the next.

What was tried, and what each review round found:
- Round 1 (two blocking): a write count `saves` in `job.json`, the copy written most often first. An older backup
  loaded once and saved to still lost to the original's higher count; the test only covered equal copies.
- Round 2 (two blocking): the loaded copy counted on from the group's highest count. An original whose record could
  not be read at all had no id, so its count was never seen and won once it read again; benchmarks and runs had no
  count at all, so they kept the bug.
- Round 3 (five blocking): a persisted rank (`.folo-chosen`) written into the chosen folder at start-up, used by all
  three stores. An unreadable rank counted as 0; the rank was written before the load, so a start where every copy
  failed promoted a stale original; a folder without its record was not seen, so a lone copy got no rank; a folder
  that could not take the rank was skipped, hiding healthy read-only records; and the cloud reaper looked for
  `cloud.lock` in `runs/<id>` while a run loaded from a copy locks its own folder (that last one is also on main
  whenever a run is loaded from a copy). The answer (ranks from the clock, marking after the load, `follow_dir` for
  cloud runs) was not reviewed, its main test was not shown to fail without it and the cloud part has no test.

Questions for a maintainer: where is the selection stored (in each data folder, in one store-level file, or not at
all, leaving copies to the user); what may a start-up write into a user's data folder; what "newest" means when clocks
move or a backup is restored; and whether the cloud reaper's lock path for a run loaded from a copy is fixed on its
own first.

## Gates

All passed locally: root `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test -p folo-server -p folo-api`; apps/desktop fmt, clippy and `cargo test --workspace`; web `npm run check`,
`npm test`, `npm run build`; `bash tools/file-size.sh`. While the apps/desktop gates built: 20 of 20 runs of
`open_entries_expire_without_parts`, 20 of 20 of `an_instance_lost_while_its_weights_are_collected_loses_the_run`
(four at a time), 3 of 3 of the whole cloud suite. `f3-server-ci-4`'s test also passes on main: it pins the contract
the producers already kept. The `web.yml` change is only exercised by CI.
