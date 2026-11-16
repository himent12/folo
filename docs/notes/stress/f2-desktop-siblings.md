# Stress round 4, follow-up lane f2-desktop-siblings: Desktop siblings of Studio fixes

A follow-up lane with no hunting. Its plan was the seven Desktop findings the round's lanes (or their reviewers)
confirmed and left open because they sat in another lane's surface. Each was checked on main 069935d6 first: all
seven were still open there. Each fix ports Studio's rule or wording where Studio has one, and comes with a
behaviour test under `apps/desktop/src/app/` or a unit test next to the helper. No app was run; the tests drive
`FoloApp` against the stub server the way the other behaviour tests do.

## Findings

| id | origin | severity | what | status |
|---|---|---|---|---|
| f2-desktop-siblings-1 | s2-runs-6 SIBLING | annoying | New run says nothing about a dataset with training images and no validation images, though every epoch is then scored on the training images | fixed |
| f2-desktop-siblings-2 | s1-datasets-6 SIBLING | annoying | A dataset whose folder is gone still adds its last known image count to the Datasets header; its card shows the count unqualified | fixed |
| f2-desktop-siblings-3 | s4-models-1 SIBLING | annoying | The model of a run stopped after 1 of 50 epochs says "Trained for 50 epochs"; its `-last` weights' epoch is unnumbered | fixed |
| f2-desktop-siblings-4 | s4-models-2 SIBLING | annoying | A damaged PNG (or JPEG, BMP, WebP) in Playground is called an unsupported format ("Use a JPEG, PNG, BMP or WebP file.") | fixed |
| f2-desktop-siblings-5 | d1-datasets (deferred) | polish | "1 images" in split labels (Models confusion and details), New run's epoch hint and Benchmarks (page, entry, New benchmark) | fixed |
| f2-desktop-siblings-6 | d1-datasets, x4-soak (deferred) | annoying | A dataset tile whose image failed looks like one still loading | fixed (dataset grid, Playground samples, run samples, Benchmarks); Studio's benchmark images SIBLING, deferred |
| f2-desktop-siblings-7 | d4-bench-settings-4 (deferred half) | polish | "Can't reach folo serve, retrying" stays up for an address that is not one; a scoped IPv6 literal is refused | fixed (banner); scoped IPv6 deferred |

### f2-desktop-siblings-1: no word on a train-only dataset in New run
Repro: a dataset with train images and `splits.val == 0`; New run, pick it. Nothing says that the epochs' mAP is
measured on the training images (docs/API.md, "Split fallback"). Studio says so (`web/src/routes/runnew/valSplit.ts`).
Fix: `FoloApp::dataset_val_note` (`apps/desktop/src/app/new_run.rs:399`) gives Studio's sentence for such a dataset
(not for one that can't train or whose read failed: the picker already says why); `src/overlays/new_run.rs` shows
it under the dataset column as a neutral note, as Studio does: an info icon in `text_tertiary`, the sentence in
`text_secondary` (it informs; Start stays on). Test: `start_from.rs`
`a_train_only_dataset_says_its_scores_are_on_the_training_images` (also: a dataset with a val split says nothing).

### f2-desktop-siblings-2: a gone folder counted in the Datasets total
Repro: register a dataset, delete its folder; Datasets. The header's "N images" still includes it; its card says
"1,234 images" as if read. Studio leaves it out of the total and says "(last known)". Fix: building on x1-parity's
`dataset_problem`, `FoloApp::dataset_unreadable` (`src/screens/overview/datasets.rs:203`) names such a dataset;
`datasets_subtitle` (`src/screens/datasets/list.rs:16`) leaves it out of the image total, and `card_facts`
(`overview/datasets.rs:260`) adds "(last known)" to its counts. Test: `dataset_tests.rs`
`a_dataset_whose_folder_is_gone_counts_only_as_last_known`.

### f2-desktop-siblings-3: a stopped run's model claims its planned epochs
Repro: a run of 50 epochs stopped after epoch 1; Models, its `-last` model, Details. "Trained for 50 epochs at …"
and "last epoch" with no number. Fix: `finished_epochs` (`src/screens/models/details.rs:21`) ports Studio's
`finishedEpochs` (`web/src/routes/model/weightsEpoch.ts`): a completed or stopped run's `epoch` is the count, any
other run's recorded history (only when `epoch` is not past it). `trained_for` says "1 of 50 epochs at 320 px", or
"50 epochs planned at 320 px" while the count is unknown; `weights_epoch` numbers the last epoch ("last finished
epoch, 1 of 50"). Tests: Studio's cases ported in `details.rs` (`numbers_the_last_finished_epoch_of_a_run_that_ended_early`,
`a_completed_or_stopped_runs_epoch_counts_over_a_history_that_lags_it`,
`any_other_run_counts_its_history_when_its_epoch_is_not_past_it`, `trained_for_*`, and three more).

### f2-desktop-siblings-4: a damaged image called unsupported
Repro: Playground, open a PNG truncated after its header. Toast: unsupported format, "Use a JPEG, PNG, BMP or WebP
file." Fix: `engine_reads` (`src/app/playground.rs:30`) recognises the four formats by their magic bytes, as Studio's
`web/src/routes/playground/intake.ts`; `undecodable` (`:38`) says a recognised file is damaged and keeps the
unsupported-format words for the rest. Test: `playground.rs`
`a_damaged_png_is_called_damaged_not_an_unsupported_format`.

### f2-desktop-siblings-5: "1 images"
Repro: a split of one image: "val · 1 images" in Models (confusion, details); the same in New run's epoch hint and
Benchmarks' page, entry and New benchmark counts. Fix: `split_label` (`src/ui/format.rs:100`) and the other sites
use the existing `counted` helper. Test: `format.rs` `a_split_label_counts_one_image_in_the_singular`.

### f2-desktop-siblings-6: a failed tile looks like loading
Repro: a dataset whose thumbnails fail (the server answers 500 for them); the grid's tiles stay blank grey like
ones still loading, and only the inspector offered Retry. Fix: `thumb_unavailable` (`src/screens/mod.rs:126`, over
the cache's `image_failed`) and `unavailable_over` / `unavailable_mark` (`:208`, `:214`) mark the tile with an
image glyph and Studio's "Image unavailable" (`web/src/lib/detect/DatasetImage.tsx`); Playground's samples get the
glyph (`src/screens/playground/side.rs:563`), and opening a failed sample asks for its thumbnail again
(`src/app/playground.rs`). Run samples (`src/screens/runs/samples.rs:137`) say "Image unavailable" on the square
cell. In the grid, Playground and run samples, Retry stays the inspector's (or the opened image's), one Enter or
click on the tile away, so the tile stays a single focus stop. Benchmarks use the same pixel cache under their own
keys: `bench_image_failed` / `retry_bench_image` (`src/bench/images.rs:132`) mark a failed Plan preview
(`src/screens/bench/plan.rs:46`) and entry image (`src/screens/bench/entry_images.rs:89`), and the marked frame is
a Retry control (click, Enter, Space). Tests: `connection_tests.rs`
`a_thumbnail_past_its_retries_loads_again_once_the_inspector_loads_its_full_si…` (the cache state), and
`app/unavailable_tests.rs`, which paints each screen (dataset grid, Playground samples, run samples, Benchmarks
Plan) with the mock's pixels failing, loading and loaded, and checks the mark's glyph and words are painted only
when failed; the Playground one also opens the sample and checks the thumbnail loads again, the Benchmarks one
activates Retry.

### f2-desktop-siblings-7: the retry pill for a non-address; scoped IPv6
Repro: Settings, server address `abc def`. The sidebar says "Not a server address" but the top pill reads "Can't
reach folo serve, retrying…", though nothing is sent and no retry can help. Fix: `Store::bad_address`
(`src/store/mod.rs:202`); `lost_banner_shown` (`src/app/render.rs:27`) leaves such an address out. Test:
`connection_tests.rs` `a_local_address_that_is_not_one_is_kept_and_starts_nothing` (no pill) and
`a_server_that_never_answered_was_not_lost` (the pill still shows for a server that is down).
Deferred, scoped IPv6: `http://[fe80::1%25eth0]:7878` is still refused by `host_ok`
(`src/api/http/transport.rs:96`), since `Ipv6Addr` has no zone. Accepting it would not help: ureq's resolver hands
`[fe80::1%25eth0]:7878` to `to_socket_addrs`, which fails, and a numeric `%251` would parse as scope 251, not 1. A
working scoped address needs a resolver that decodes the zone (and maps an interface name to its index); that is a
transport change, not a validation one, so the refusal stays, and its "Not a server address" message is honest.

## Review round 1
rev_correct: no findings. rev_tests 1 (the old test only called `thumb_unavailable`, not a renderer) and rev_user
1-3 (Benchmarks, run samples, a Playground sample's thumbnail after opening it) are fixed in 301afae3, with the
painted-scene tests of `app/unavailable_tests.rs`; with `unavailable_mark` made to paint nothing, all four fail.
rev_tests 2: this report and the PR body now say the New run note is neutral. rev_user 4 (Studio's benchmark
gallery, MatchView and Preview swallow image failures) is a Studio surface: deferred, SIBLING, below.

## Gates
From `apps/desktop`, in a build slot, at 301afae3: `cargo fmt --all -- --check` OK, `cargo clippy --workspace
--all-targets -- -D warnings` OK, `cargo test --workspace` OK; `bash tools/file-size.sh` OK. No `cfg(unix)` code was
touched.

## Not done
- Scoped IPv6 server addresses (f2-desktop-siblings-7, above).
- SIBLING in Studio (review round 1, rev_user 4): Studio's benchmark images swallow pixel failures. The entry's
  gallery and large viewer (`web/src/routes/bench/entry/ImagesSection.tsx:72-76,117-120`) and `MatchView.tsx:99`
  have no error state; `web/src/routes/bench/Preview.tsx` records URL errors (`:88-104`) but shows them only while
  its metadata is missing (`:164`). A failed image stays blank with its boxes and no Retry. Desktop's Benchmarks now
  mark it and offer Retry; Studio's fix belongs to a web lane.
- None of the fixes was checked in a running Desktop window; they are covered by the behaviour and unit tests only.
