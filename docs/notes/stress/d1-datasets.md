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
Fix: `ImageCache::forget_failed` from `FoloApp::reconnected`, and an image that fails with `server_unreachable` marks the
server down itself, so an outage only the images saw is noticed and recovered too. Tests: `store/images.rs`
`failures_are_forgotten_on_request_and_ready_images_kept`, `app/connection_tests.rs`
`images_that_failed_while_the_server_was_away_load_once_it_answers`.

### d1-datasets-5: a refused file deletion left no way forward

Repro: any registered dataset whose images are gone on disk (`broken`), then Generate synthetic `dblgen`, Remove from
FOLO with "Also delete the 3,000 generated images" on. The server answers `409 dataset_in_use` "the files of broken
could not be checked (... no train or val images found), so they may be in that folder", and the toast showed only
that; the server's hint ("unregister this dataset without files=true") was dropped and speaks the API anyway.
Fix (Desktop): `removal_refused` (`app/actions.rs`) adds "To unregister it and keep its files, remove it again with
“Also delete” off."; other hints are shown unless they name API flags or routes. Test: `app/actions.rs`
`a_refused_file_deletion_says_how_to_unregister_alone`.
Deferred (server): `crates/folo-server/src/real/deletion.rs:61` fails closed when `reads()` (`index.rs:299`,
`folo_data::open_with`) errors, and a dataset that finds no images errors, so one emptied dataset anywhere blocks file
deletion of every generated dataset. Whether "no images found" may count as "reads nothing" is a server decision.

### d1-datasets-3 (deferred)

`crates/folo-server/src/index.rs:177` takes folo-data's `broken_labels` (images skipped for a broken label,
`folo-data/src/dataset/mod.rs:259`), `crates/folo-server/src/stats.rs:56` adds label lines, and docs/API.md says lines.
Which unit is right is an API decision; the Desktop wording follows once it is made.
