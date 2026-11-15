# Stress round, lane d1-datasets: FOLO Desktop › Datasets

Surface: the Datasets list, a dataset's images grid, statistics and health, the image inspector, the decoded-image
cache (`apps/desktop/src/store/images.rs`). Driven off screen (`FOLO_UI_OFFSCREEN=1`, window messages only) against a
real `folo serve` (CPU, own data dir, port 18060) built from the shared prebuilt binaries.

## How

- Datasets built for the lane: `tenk` (10,000 synthetic 96 px images), `photos-yolo` (seed photos, YOLO labels),
  `badlabels` (out-of-range class ids, NaN boxes, missing label files, an empty split), `long-unicode` (300-character
  unicode name, paths with spaces), `oddyolo` (odd formats and corrupt files from the seed kit), `one` (1 image),
  `yaml-no-names`, `huge` (8000x6000 JPEGs, a 12000x9000, a 20000x1000 and a 1000x20000). A bad `data.yaml` and a
  0-image folder were refused by the server, as they should be.
- Memory sampled as private bytes of the Desktop process while scrolling and stepping through the inspector.

## Findings

| id | severity | what | status |
|---|---|---|---|
| d1-datasets-1 | blocker | Inspector: every full-size image viewed stays decoded (W x H x 4 bytes), 600 of them may be kept: 2.2 GB after 10 big photos, never freed | fixed |
| d1-datasets-2 | polish | "1 images" on the dataset card, its tooltip, the list subtitle and the health hint | fixed |
| d1-datasets-3 | annoying | Health "32 broken label files" for 8 affected files: the server fills `broken_labels` from two producers with different units | deferred |
| d1-datasets-4 | annoying | Thumbnails (and inspector images) that failed while the server was away stay blank for good after it is back | fixed |
| d1-datasets-5 | annoying | Remove with "Also delete the generated images" refused because an unrelated dataset found no images; the toast dropped the way forward | Desktop fixed, server deferred |
| d1-datasets-6 | polish | Grid tiles captioned with the image id ("6" on a real server), not the file name Studio shows | fixed |

### d1-datasets-1: the decoded-image cache had no byte budget

Repro: dataset `huge`, open the first image in the inspector, press Right through 10 images. Private bytes: 385 MB on
the grid, +184 MB per 8000x6000 image, 2,230 MB after 10 images, still 2,230 MB after closing it. `ImageCache::BUDGET`
counted entries of any kind and the inspector's `ImgKind::Full` images were decoded at full resolution.
Fix: a byte budget (768 MiB) beside the count, the image just finished never evicted (so one image larger than the
budget still shows), and full-size images decoded at most 4096 px a side, so the budget always holds the one shown with
the grid behind it (no evict-refetch loop) and the texture fits every GPU. The inspector draws in the image's own
coordinates (`meta.width/height`), so boxes still line up. Tests: `store/images.rs`
`big_images_are_held_within_a_byte_budget_keeping_the_newest`, `full_images_are_scaled_down_to_the_cap_keeping_their_aspect`.

### d1-datasets-4: failed image slots were never retried

Repro: open `tenk`, stop the server, scroll a few notches, start it again; the tiles asked for meanwhile show their
boxes on an empty tile forever. `ImgSlot::Failed` was stored for any error and only a server switch cleared it.
Fix: `ImageCache::forget_failed` from `FoloApp::reconnected`, and an image that fails with `server_unreachable` has
`/system` asked at once, so an outage only the images saw is noticed and recovered too. Tests: `store/images.rs`
`failures_are_forgotten_on_request_and_ready_images_kept`, `app/connection_tests.rs`
`images_that_failed_while_the_server_was_away_load_once_it_answers`.
Review round 1: a failure with the server up (a body cut off after its headers, a file being rewritten) saw no outage,
so nothing retried it. Any failed image is now asked for again on its own 2, 8 and 32 s after its 1st, 2nd and 3rd
failure in a row (`ImageCache::retry_delay`, `FoloApp::retry_image`); past that it waits for a reconnect or the error
panel's Retry (`retry_all` forgets failed images too), so a corrupt file is not fetched for ever. Test:
`app/connection_tests.rs` `an_image_that_failed_with_the_server_up_is_asked_for_again_a_few_times`.
Review round 2: (a) a retry timer only knew the backend, so one set before a reconnect could ask again for the next
failure early, and overlapping timers could pass the fourth-failure stop; each failure now has a token and a timer only
answers its own (`ImageCache::forget(key, token)`). (b) An image timing out on a server whose `/system` answers marked it
down, and the next answer "reconnected" and reset its retries, so it was fetched for ever; an image's
`server_unreachable` now only has `/system` asked at once, and only `/system` marks the server down. (c) The inspector
says "Image unavailable" (or that it shows the thumbnail) with a Retry button (Tab, Enter) that asks for the image again
with its retries afresh (`FoloApp::retry_failed_image`). Tests: `store/images.rs`
`a_retry_timer_set_for_an_older_failure_leaves_a_newer_one_alone`, `app/connection_tests.rs`
`a_retry_timer_left_from_before_a_reconnect_leaves_the_next_failure_alone`, the server-up test above (timeouts with
`/system` answering; ends with the Retry). Deferred (SIBLING of Studio's `DatasetImage`): a failed grid tile still
looks like a loading one (boxes on an empty tile); Enter on it opens the inspector, which now says so and retries.
Telling the two apart on the tile needs a third state through `image_canvas` and every caller of `FoloApp::image`
(grid, Playground samples, Bench).

### d1-datasets-5: a refused file deletion left no way forward

Repro: any registered dataset whose images are gone on disk (`broken`), then Generate synthetic `dblgen`, Remove from
FOLO with "Also delete the 3,000 generated images" on. The server answers `409 dataset_in_use` "the files of broken
could not be checked (... no train or val images found), so they may be in that folder", and the toast showed only
that; the server's hint ("unregister this dataset without files=true") was dropped and speaks the API anyway.
Fix (Desktop): `removal_refused` (`app/actions.rs`) adds "To unregister it and keep its files, remove it again with
“Also delete” off." to the refusals made before anything changed (`dataset_in_use`, `invalid_query`); every other
error keeps its own hint (`run_active` refuses unregistering too, a folder that failed to delete was already
unregistered, `unauthorized` names the token), unless the hint names API flags or routes. Test: `app/actions.rs`
`a_refused_file_deletion_says_how_to_unregister_alone`.
Deferred (server): `crates/folo-server/src/real/deletion.rs:61` fails closed when `reads()` (`index.rs:299`,
`folo_data::open_with`) errors, and a dataset that finds no images errors, so one emptied dataset anywhere blocks file
deletion of every generated dataset. Whether "no images found" may count as "reads nothing" is a server decision.

### d1-datasets-2 and d1-datasets-6 (polish)

"1 images" came from `"{} images"` in `screens/overview/datasets.rs` (card and tooltip), `screens/datasets/list.rs`
(subtitle) and `screens/datasets/health.rs` ("only 1 images"); they now use the singular for one. Grid tiles printed
`meta.id` (`"6"` on a real server, `img_00007` in the demo, which hid it) in the caption and tooltip; they now show the
file name (`file_name`, either slash), as Studio's grid does (`web/src/routes/dataset/ImageGrid.tsx`). Test:
`screens/datasets/images.rs` `a_tile_is_named_by_its_file_not_its_id`.
Review round 1: the grid toolbar ("1 images", "1 val images") counts in the singular too, the health hint has a test
(`a_small_dataset_is_counted_in_the_singular_at_one`), and a dataset image picked in Playground is named by its file
when the dataset's image list is loaded (`app/playground.rs`, its id otherwise). SIBLING, not changed: "N images" with
no singular in `ui/format.rs` `split_label`, `screens/models/confusion.rs`, `overlays/new_run.rs`, `screens/bench/*`
and Studio's `web/src/routes/dataset/health.ts` (other lanes' surfaces).

### d1-datasets-3 (deferred)

`crates/folo-server/src/index.rs:177` takes folo-data's `broken_labels` (images skipped for a broken label,
`folo-data/src/dataset/mod.rs:259`), `crates/folo-server/src/stats.rs:56` adds label lines, and docs/API.md says lines.
The real server also keeps images whose label lines it drops for an out-of-range class (oddyolo val lists all 8),
although each issue says "the loader skips this image". Which unit is right is an API decision; the Desktop wording
follows once it is made.

## Checked, not bugs

- 10,000-image grid scrolled fast top to bottom and back: private bytes stayed near 100 MB, handles transient.
- EXIF-rotated photos: tiles and inspector upright; boxes at the right pixels on `photos-yolo`.
- Window at its minimum (960x620 logical) and 2560 px wide: list and grid lay out (the subtitle drops the split counts
  at the minimum). Keyboard: Tab reaches the tiles, arrows move, Enter opens the right image.
- A dataset unregistered through the API while its inspector is open: the inspector closes at the next refresh, no
  crash, no stale image.
- Generate synthetic: names `../x`, `con`, `a/b`, `NUL.txt`, 300 characters, spaces are sanitised by the server
  (`x`, `con`, `a-b`, 48-character cap); an image count of 23 digits is a field error with Generate off; a
  double-click on Generate makes one dataset; Esc while generating leaves a toast and the dataset arrives.
- Inspector: Left at the first image wraps to the last, Right at the last to the first; zoom and pan clamp at the
  image edge, `-` stops at Fit.
- Chips missing on boxes under two chip heights in thumbnails: by design (`src/viz.rs`).

## Not done

- Add dataset through the real file dialog on a sandbox desktop (only the path routes the app takes without the dialog
  were tried).
- Two windows on the same dataset; the dataset folder deleted on disk while its grid is open (only the unregister case).
- Studio's side of d1-datasets-5 (its delete dialog) was not checked.
