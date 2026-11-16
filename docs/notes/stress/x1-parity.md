# Stress round (dogfood 4), lane x1-parity: Studio and Desktop side by side

## What was tested, and how

One `folo serve` (CPU device, `FOLO_GPU_BACKEND=metal`, port 18120) on a copy of the seed data dir, plus this lane's
own runs: `photos-yolo` registered (ds_d978), a 3-epoch fine-tune from `folo-n-coco` (run_5050), a 2-epoch fine-tune
from `m_560a` (run_c009). Studio in headless Chrome (playwright-core) and FOLO Desktop off screen (`--shot` and
`FOLO_UI_OFFSCREEN=1`, driven with window messages) on the same server at the same moment, both compared against the
API JSON and the files on disk: Overview, Datasets, Runs table, run page, Models list, model page (KPIs, per-class
table, confusion matrix), relative times, device labels, queue wording.

## Findings

| id | severity | surface | finding | status |
|---|---|---|---|---|
| x1-parity-1 | blocker | Desktop run page / Overview mAP chart, Precision / Recall chart | "× 10⁻²" caption over unscaled ticks on a fitted 0-1 axis | fixed (SIBLING of d2-runs-1: the same change as that lane's) |
| x1-parity-2 | annoying | every relative time | same moment "6d ago" (Studio) vs "5 d ago" (Desktop); Desktop's old dates in UTC; Studio "60m ago" | fixed |
| x1-parity-3 | polish | Runs table, queued run | "1st in queue" vs "Next to start" | deferred: wording, both true |
| x1-parity-4 | polish | run header, Overview hero | CPU device "(CPU)" vs "· cpu" | deferred: wording |
| x1-parity-5 | annoying | Desktop model page per-class table | class names cut to one letter on real data | deferred (see below) |
| x1-parity-6 | — | run page vs model page per-class | last epoch vs best checkpoint values | not a bug: both labelled |
| x1-parity-7 | annoying | Datasets list | unreadable dataset: Studio "Can't read", Desktop card silent | fixed |
| x1-parity-8 | annoying | run page › Validation samples | Min confidence starts 0.25 (Studio) vs 0.30 (Desktop): Desktop draws fewer boxes | fixed |

Agreeing (checked number by number against `/api/...` JSON): model page KPIs of m_5050 (mAP50-95 0.003, mAP50
0.019, P 0.006, R 0.219, conf 0.003) and its per-class values; the Runs table's status, epochs and "started from" of
every seed and lane run; dataset image / box / class counts of shapes and photos-yolo.

### Repro of the open and deferred ones

- **x1-parity-5** (Desktop): `folo-desktop --server http://127.0.0.1:<port> --open model:m_5050 --shot x.png` on any
  real server (not `--mock`, not `--scenario model`): the Class column is ~200 px wide but every name reads as one
  letter plus an ellipsis ("p…", "c…", "d…", "c…"; "All classes" as "…"). Reproduced 3 times, survives a re-render.
  Suspected mechanism: `TextLayout::layout`'s cache shortcut (`apps/desktop/ui/src/elements/text.rs:111-116`)
  answers an intrinsic-size query (no wrap width) with the size cached from a narrow, truncated measure; keyed
  measures persist across frames (`ui/src/window/layout.rs:164`), so a narrow first layout of the table beside the
  plot (`src/screens/models/pr.rs:208-218`) is a fixed point. Not proved: a folo-ui test of an ellipsised flex item
  laid out at 15 px and then at 200 px (`ui/src/elements/tests/text.rs` shape) shows the name in full, so the
  trigger is something of the real page (the eval arriving after the first frame, the table's own sizing). Left
  open with this description; a fix belongs in folo-ui's layout core or `src/screens/models/class_table.rs`.
- **x1-parity-1**: fixed after review round 2 (below), with d2-runs-1's own change, line for line.
- **x1-parity-3 / -4**: wording only ("1st in queue" vs "Next to start"; "(CPU)" vs "· cpu"), both true.
- Also seen, not filed: Desktop's dataset page for the broken `ds_7816` shows the server's message under a plug icon
  with a "Connection settings" button (it is not a connection problem); both apps' Datasets header counts the 24
  missing images of `ds_7816` in "304 images" (the stored list summary).

## Not done

Playground detections side by side, benchmarks, cloud mock states, act-in-one-watch-the-other latency (rename,
stop, delete), long unicode names and their truncation, and the CLI `--json` comparison: the build-slot waits of
turns 3 and 4 took that time.

## Fixes

- **x1-parity-2**: `apps/desktop/src/api/time.rs` `ago` rounds minutes, hours and days as Studio does, says
  "yesterday" for 24-48 h, counts days up to 14 and then shows the reader's local date (it showed the UTC date from
  7 days). The zone offset comes from Windows' zone rules, and on Linux and macOS from the C library's
  `localtime_r` (review round 1: it was UTC there). `web/src/lib/format.ts` `ago`: a unit that rounds up reads as
  the next one ("1h ago", not "60m ago"). Tests: `time.rs` `ages_round_like_studio`,
  `an_old_date_is_the_local_calendar_date` (fixed dates at +9 h, 0, −8 h and across New Year through
  `calendar_date`); `format.test.ts` "never says 60m, 24h or 14d ago".
- **x1-parity-7**: `apps/desktop/src/screens/overview/datasets.rs` `dataset_card` shows Studio's "Can't read" /
  "Folder missing" badge when the dataset's own read fails (`dataset_problem`, the mapping of
  `web/src/routes/datasets/problem.ts`), with the server's reason in the card's tip. After review round 1: a visit
  to Overview, Datasets or a dataset page reads every dataset read before again in place (`ds_recheck`,
  `app/data.rs`; a connection error leaves the card as it was), so a folder moved away or put back after the first
  read shows; and the page of an unreadable dataset offers All datasets, Remove from FOLO and Retry with the
  server's message (`screens/datasets/detail.rs`), not Connection settings. Tests:
  `an_unreadable_dataset_is_named_on_its_card`, `dataset_tests.rs` `a_dataset_card_follows_its_folder_away_and_back`
  (drawn Overview and dataset page over a mock whose dataset read breaks and mends; the warning badge counted in
  the painted scene). After review round 2: every read of a dataset takes a number (`ds_reads`), and only the
  latest one's answer is applied, so two Retries or visits answering in reverse order leave the newer answer
  (`an_older_dataset_read_answering_last_never_overwrites_a_newer_one`); a read again whose dataset differs from
  the one held (back from a failure, or other contents) drops its image list, the page of it on its way, its label
  issues, its decoded images (`ImageCache::forget_dataset`) and the grid's filter, so they are read again
  (`a_dataset_read_again_with_other_contents_lists_its_images_again`,
  `forgetting_a_dataset_keeps_the_others_and_drops_its_late_loads`). Labels edited on disk with every count
  unchanged are not noticed: the dataset record is the same. After review round 3: every issue check is numbered
  (`issue_reads`) and every image load is on lane x4's per-slot attempt (`ImageCache::attempt`, which
  `forget_dataset` ends for the dataset's images, also dropping their failures and their bytes from lane d1's
  byte budget), so one asked for before the files changed never fills what was asked for since, whichever
  answers first (`a_load_from_before_a_forget_never_fills_the_slot_asked_for_again`,
  `forgetting_a_dataset_ends_its_bytes_attempts_and_failures`,
  `an_issue_check_of_files_since_changed_never_comes_back`); the card test reads the rendered focus tip (its glyphs
  grow by the words added to the server's reason). With lane d6's reconnect sweep, a dataset that failed while
  the server was away is read once after the reconnect (`a_dataset_that_failed_while_away_is_read_once_after_the_reconnect`).
- **x1-parity-1** (review round 2): fixed on main by d2-runs (PR #282, `charts::axis::label_pow`); this lane's
  identical copy dropped out in the rebase.
- **x1-parity-8**: `apps/desktop/src/app/mod.rs` starts `sample_conf` at the slider's floor `SAMPLE_CONF.0` (0.25,
  every prediction the server keeps, as Studio's `Samples.tsx` starts) instead of 0.3. Test:
  `run_page_tests.rs` `validation_samples_start_at_every_prediction_the_server_keeps`.

## Gates

See the pull request body for the gates of the pushed head.
