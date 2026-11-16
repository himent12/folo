# Dogfood report 4: the stress round of FOLO Studio and FOLO Desktop

Sixteen agents each took one area of Studio (`web/`), Desktop (`apps/desktop`) or the server behind them, used it
as a hard user would, wrote down what broke and fixed it with regression tests, one pull request per area. Their
reports are in [stress/](stress/README.md); this note sums them up. Counts are recounted from the lanes' own
records (`FINDINGS.md`, `REVIEWS.md`, under `D:/folo-build/stress/lanes/` on the maintainers' machine, not in git).

## Verdict
**Both apps survive what broke them at the start of the round: a server that starts late, dies or stalls, corrupt
state and answers of the wrong shape, ten thousand images, hundreds of runs.** The round logged 103 findings, 9 of
them blockers; 71 are fixed on `main`, 29 are deferred or open and 3 were duplicates of another lane's. All nine
blockers are fixed. What is left (below, [Still open](#still-open)) is wording, one-off edge cases and a few
server-side decisions.

The fixes a user would notice most:
1. **Desktop no longer runs out of memory in the image inspector** (d1-datasets-1): full images were held at their
   original size without a byte budget, about 2.2 GB after ten 8000×6000 photos; they are now capped at 768 MiB and
   4096 px a side.
2. **Desktop recovers when the server is late, gone or silent** (d6-connection-4, -5, -1): Runs and Overview no
   longer say "Could not reach folo serve" forever when Desktop opened first, a run's event stream that goes silent
   ends after 45 s instead of freezing the page, and a `FOLO_TOKEN` in the environment no longer locks Desktop out
   of the server it starts itself.
3. **Studio does not get stuck on bad state** (s6-shell-1, -2, -3): corrupt saved preferences, a screen that throws
   and list answers of the wrong shape each show an error with a way out instead of a frozen or blank page.
4. **Images that failed while the server was away load again** once it answers, in both apps (s1-datasets-8,
   d1-datasets-4), and Desktop retries a failed image a few times on its own.
5. **Registering the wrong folder is fast and says so** (x3-server-1, -6, -7): a folder that is not a dataset is
   refused before the tree is walked (it took minutes and blocked every other dataset), the folder above a dataset
   points to it, and a damaged `datasets.json` shows an error instead of "No datasets yet".
6. **Charts and numbers agree between the apps** (d2-runs-1, x1-parity-1, -2, -8): Desktop's mAP axis no longer says
   "× 10⁻⁵" beside unscaled ticks, ages round as in Studio, and validation samples start at the same confidence.

## How the round worked
By the lead's account: one setup agent built shared binaries and seed data (`D:/folo-build/stress/ENV.md`), then
16 lanes ran, each one Claude Opus 5.5 agent (high effort) with its own git worktree, branch `stress/<lane>` and pull
request. Every lane ran its own `folo serve` on its own ports and data directory, with Desktop off screen
(`FOLO_UI_OFFSCREEN=1`) and Studio in headless Chrome. Each pull request got three GPT-6.1 Sol reviewers
(correctness; tests and claims; the user and sibling code); Opus answered every finding in `REVIEWS.md`, then up to
two re-reviews by two fresh reviewers followed. Neither model decided merges: the orchestrator read `REVIEWS.md` and
the diffs, ordered extra reviews or rebases, and merged on green CI. The round ran from 2026-11-14 18:00 to
2026-11-15 about 21:30 Pacific time, with a 15-hour stop in the middle (a model rate limit, then two restarts of the
orchestrating app). A lane that found the same bug in the other app fixed its own and marked the finding `SIBLING`.

## The lanes
Severity as the lane gave it; "fixed" counts a finding fixed in part (d1-datasets-5, x4-soak-3) as fixed, "deferred"
includes findings left open; duplicates of another lane's finding are in neither. Merge order as in the PR column.

| lane | surface | PR | blocker | annoying | polish | fixed | deferred | dup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| [s6-shell](stress/s6-shell.md) | Studio shell, settings, router, bad answers | #268 | 2 | 2 | 3 | 4 | 3 | |
| [s1-datasets](stress/s1-datasets.md) | Studio datasets | #267 | | 4 | 4 | 5 | 3 | |
| [d5-window-input](stress/d5-window-input.md) | Desktop window, keyboard, text fields, arguments | #280 | | 3 | 4 | 7 | | |
| [d6-connection](stress/d6-connection.md) | Desktop finding and starting the server | #273 | 3 | 1 | 2 | 4 | 2 | |
| [d2-runs](stress/d2-runs.md) | Desktop runs, New run, charts | #282 | | 4 | 4 | 7 | 1 | |
| [s3-live](stress/s3-live.md) | Studio Run page live, Run compare | #269 | | 3 | 3 | 3 | 3 | |
| [d3-models](stress/d3-models.md) | Desktop models, Playground | #272 | | 3 | 4 | 3 | 4 | |
| [s5-label](stress/s5-label.md) | Studio labeling, benchmarks | #271 | | 3 | 2 | 3 | 2 | |
| [d1-datasets](stress/d1-datasets.md) | Desktop datasets, image cache | #277 | 1 | 3 | 2 | 5 | 1 | |
| [d4-bench-settings](stress/d4-bench-settings.md) | Desktop benchmarks, settings | #276 | | 3 | 1 | 4 | | |
| [x2-newcomer](stress/x2-newcomer.md) | both apps, first steps | #274 | | 2 | 5 | 3 | 3 | 1 |
| [x3-server](stress/x3-server.md) | the server under both apps | #278 | 2 | 3 | 4 | 7 | 1 | 1 |
| [s2-runs](stress/s2-runs.md) | Studio runs, New run | #275 | | 5 | 4 | 6 | 2 | 1 |
| [s4-models](stress/s4-models.md) | Studio models, Playground | #266 | | 2 | 2 | 3 | 1 | |
| [x4-soak](stress/x4-soak.md) | both apps, long sessions and scale | #281 | | 3 | | 3 | | |
| [x1-parity](stress/x1-parity.md) | Studio against Desktop | #270 | 1 | 4 | 2 | 4 | 3 | |
| **total** | | | **9** | **48** | **46** | **71** | **29** | **3** |

Notes on the count: s1-datasets-7 (a removed dataset linked from the run page, deferred to the Runs side) is in the
lane's `FINDINGS.md` but not its report's table; x1-parity-6 was checked and is not a bug, so it is not counted;
d3-models-2 counts as deferred (its caption was fixed with d3-models-7, the "–" tiles were not); s5-label-2 is
severity as in the report's table. #283 (`cli: keep the cloud command test independent of the calendar`) is the
one pull request of the round outside the lanes ([CI and machine lessons](#ci-and-machine-lessons)).

## What was fixed
The important ones; every finding is in its lane's report.

**Studio**
- s6-shell-1, -2, -3: corrupt prefs are put back to defaults (`web/src/lib/prefs.ts` `sanitize()`), a screen that
  throws shows `ScreenCrash` (`web/src/App.tsx`), list answers of the wrong shape are a `bad_response` error with
  Retry (`web/src/lib/api/client.ts`).
- s1-datasets-1, -2, -8: a dataset removed in another tab says so; thumbnails scrolled past cancel their requests;
  failed thumbnails load again when the server is back.
- s3-live-4, -5: the Run page of a run deleted elsewhere leaves "Reconnecting"; Run compare follows a live run to its
  end.
- s2-runs-1, -5, -6, -9: run names keep non-ASCII letters; a row menu near the bottom opens upwards; New run says when
  a train-only dataset is scored on its training images; a late Start answer no longer deletes a newer draft.
- s4-models-1: a model from a run stopped early says how many epochs it trained.
- x4-soak-1: past 200 runs, the Runs page no longer refetches every run's full metrics on every visit (one cached
  entry per run).
- s5-label-1: two tabs labeling one image no longer silently overwrite each other's boxes.

**Desktop**
- d1-datasets-1, -4: the image cache has a byte budget; images that failed during an outage are forgotten on
  reconnect and retried.
- d6-connection-1, -4, -5: see the Verdict; review widened d6-connection-5 to every page that failed while the server
  was away.
- d5-window-input-1 to -7: an IME message no longer deletes selected text; Shift+Home/End, Ctrl+Shift+arrows and
  Ctrl+Delete work in text fields; `--width inf` no longer crashes `--shot`; one bad value in `desktop.json` no longer
  throws away the whole file.
- d2-runs-1, -4, -7: the mAP axis label; New run on a server with no dataset shows a toast with Add dataset; a run
  stopped mid-epoch shows that epoch's steps.
- d3-models-1: the Playground on the local engine no longer runs every superseded request before the last.
- d4-bench-settings-1, -3: Esc, Enter and Ctrl+Enter work in the Benchmarks forms; the Plan tab no longer calls a
  hand-made plan "the default".
- x4-soak-2: after a fast scroll through a big grid, tiles in view load first (a bounded load queue instead of about
  700 threads).
- x1-parity-1, -2, -7, -8: the same model's age, an unreadable dataset and the samples' starting confidence read as
  in Studio.

**Server**
- x3-server-1, -7: see the Verdict; x3-server-2: a token sent in the query string is no longer written to the log;
  x3-server-9: a copied run folder no longer lists the run twice under one id.
- s1-datasets-5 (with x3-server-4): a dataset named `NUL` or another Windows device name can be made (`slug` in
  `crates/folo-server/src/util.rs`).
- s4-models-4: a model whose dataset folder is gone no longer answers "Pass the dataset root".

## What review caught
The Sol reviewers wrote 114 reports over the 16 pull requests and raised **128 blocking findings** (the `BLOCKING:`
tallies of the `REVIEWS.md` files): x3-server 25, s3-live 13, x1-parity 13, d4-bench-settings 11, s5-label 11,
s4-models 9, d2-runs 8, d6-connection 7, s6-shell 7, x4-soak 6, d5-window-input 5, s2-runs 4, d1-datasets 3,
d3-models 3, s1-datasets 2, x2-newcomer 1. Most were answered with a fix; each lane's `REVIEWS.md` has the
disposition under every finding. Rejections and deferrals cited evidence, for example x3-server's two deferred
blocking findings: a copied run reappearing after a restart and the unbounded top-level format check were both shown
to behave the same on the base commit `1436ad68`.

Review fixes that brought their own regression, caught by the next round:
- **x3-server, rounds 2 to 4.** The round-1 fix for an unbounded walk of a non-dataset folder moved the budget into a
  format probe; round 2 found nested probes escaped it; the round-2 fix read YAML files under the folder, and round 3
  found that read bypassed `--confine` through a symbolic link and hung on a FIFO; the round-3 fix opened files
  without waiting, and round 4 found that applied to Apple and Linux only. In parallel, the deduplication of copied
  labeling and benchmark folders modified a backup before choosing it (round 2), then hid an intact backup behind an
  invalid canonical record (round 3), then behind one that parsed but could not load (round 4).
- **d6-connection.** Round 1 made reconnect clear failures already on screen; round 2 found a request that failed
  only after the server was back stayed failed; round 3 found a body cut off mid-transfer read as a bad response,
  which no sweep clears.
- **d2-runs, round 2.** The fix that disables GPU 0 on a server without a GPU leaked that answer to the next server
  connected (`apps/desktop/src/app/new_run.rs`); round 3 confirmed the cache is cleared on a server switch.
- **s3-live, round 2.** Run compare's new polling dropped a run deleted elsewhere from the comparison but kept it in
  the URL (`web/src/routes/RunCompare.tsx`).

Code no reviewer had seen: by the lead's account, d1-datasets, x3-server, x4-soak and x1-parity each ended a round
with commits pushed after their last review; the lead read those diffs or ordered an extra review before merging.
x4-soak's fourth round is one such: it found that image loads finishing out of order had no attempt number
(`apps/desktop/src/app/data.rs`), which became the per-slot attempt number below.

## Where lanes collided
By the lead's account, checked against the lane reports:
- **Desktop image retries.** d1-datasets, x4-soak and x1-parity each changed how Desktop retries failed dataset
  images. d1-datasets' model was kept: a failure count and token per image, retries at 2, 8 and 32 s, reconnect and
  Retry forget failures, a byte budget. x4-soak kept its bounded load queue on top and added a per-slot attempt
  number; x1-parity dropped its own load tickets for that attempt number. x4-soak and x1-parity were rebased on
  d1-datasets and reviewed again.
- **New run with no dataset.** d2-runs (d2-runs-4) and x2-newcomer (x2-newcomer-1) both fixed New run doing nothing
  on a server with no dataset; d2-runs' toast with an Add dataset action, which leaves the user where they are, was
  kept.
- **The chart axis.** x1-parity carried a copy of d2-runs' axis fix (x1-parity-1); the one on `main` is d2-runs'.
- Smaller overlaps were left to one lane by a `CLAIM` in its `FINDINGS.md`: s2-runs-4 to x4-soak, x2-newcomer-4 to
  x3-server, x3-server-4 and s5-label-5 to s1-datasets' `slug` fix, s6-shell-4 to s1-datasets.

## CI and machine lessons
By the lead's account:
- **GitHub Actions answered every pull-request run with `startup_failure` for about two hours**, then recovered
  without a known cause. Pull requests were closed and reopened to get runs.
- **All runners live on the one PC the lanes used**, so the queue reached 32 workflow runs.
- **A test with fixed dates expired mid-round**: `crates/folo-cli/tests/cloud.rs` against the ledger's seven-day
  retention. #283 fixed it and `crates/folo-server/tests/cloud/failures.rs` with it.
- **Linux-only code is never compiled by the local gates on the Windows PC**: x1-parity's
  `apps/desktop/src/api/time.rs` failed clippy on Linux from its first push until its last.
- **Load-sensitive tests cost re-runs**: `crates/folo-server/tests/cloud`, `tests/bench/external.rs`
  `open_entries_expire_without_parts`, and a Node crash in `npm run check` on the Linux runners.
- **Exporting `FOLO_UI_OFFSCREEN=1` for `cargo test` in `apps/desktop` fails** folo-ui's
  `a_cancelled_press_does_not_click`: set it per launch, not for the shell.
- **A stopped agent leaves its build-slot lock behind**: someone has to release it by hand.

## Still open
From the lane reports, the lead's list and `main` after #270; three follow-up lanes (f1-studio-siblings,
f2-desktop-siblings, f3-server-ci) were started on 2026-11-15 to work through it. Nothing here is fixed yet.

**Studio**
- No silence deadline on event streams: `web/src/lib/api/sse.ts` relies on `EventSource` errors and reconnect alone
  (SIBLING of d6-connection-5).
- A restored GPU 0 draft on a server without a GPU lets the first Start through (`web/src/routes/runnew/jump.ts`;
  SIBLING of d2-runs-2, deferred in d2-runs' review).
- s2-runs-7: an unknown `?from=` run shows the server's agent hint and outlives the link.
- s3-live-3: the Run page reads the dataset list once (`web/src/routes/Run.tsx`), so a run whose dataset was deleted
  offers a Retry that cannot succeed.

**Desktop**
- No train-only note in New run (SIBLING of s2-runs-6).
- Missing datasets still counted in the Datasets header (`apps/desktop/src/screens/datasets/list.rs`, SIBLING of
  s1-datasets-6).
- A stopped run's model says it trained its planned epochs (`apps/desktop/src/screens/models/details.rs`, SIBLING of
  s4-models-1).
- A damaged PNG is called unsupported (`apps/desktop/src/app/playground.rs:106`, SIBLING of s4-models-2).
- `split_label` (`apps/desktop/src/ui/format.rs:100`) and other "1 images" strings.
- No failed state with Retry on a grid tile.
- A scoped IPv6 literal in the server address.

**Server and tests**
- A cloud run whose instance dies during a checkpoint copy ends with the copy's error instead of
  `cloud_instance_lost` (seen as the flaky `a_stop_that_finds_the_worker_gone_loses_the_run_instead_of_collecting`).
- The load-sensitive suites under [CI and machine lessons](#ci-and-machine-lessons).
- d1-datasets-3: `broken_labels` means label lines in docs/API.md and images in folo-data.
- x3-server, review round 4: top-level dataset detection is still unbounded.
- A deleted run whose folder had a copy reappears after a restart (x3-server, review round 1).
- s2-runs-8: the mock cloud's `ended_in` says `syncing` for a run stopped while provisioning or booting.

## Not tested
- **Training on the GPU**: it was busy with another long run; lanes trained tiny runs on the CPU.
- **Real cloud providers**: the mock only, no keys.
- **macOS.**
- **Anything needing the user's screen**: Desktop ran off screen, Studio in headless Chrome.
