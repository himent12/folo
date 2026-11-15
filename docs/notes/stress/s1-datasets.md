# Stress round, lane s1-datasets: Studio › Datasets

Base `1436ad68`. Studio served by the shared prebuilt `folo serve` (CPU device, port 18000, own `--data-dir` and
`FOLO_HOME`), driven by headless Chrome (playwright-core, `channel: 'chrome'`). The lane's scripts, logs and
screenshots stay in `D:/folo-build/stress/lanes/s1-datasets/` (`scripts/*.mjs`, `logs/`, `shots/`).

## What was tested and how

- **Every way in.** Add dataset by local path through the dialog (`dlg.mjs`, 20+ cases) and the API (`reg.mjs`):
  the seed kit's YOLO and COCO sets, a path with spaces and unicode, trailing slash, backslashes, upper case,
  `..` segments, `\\?\D:\…`, `\\localhost\D$\…`, a relative path, a quoted "Copy as path", leading spaces, a file
  path, a data.yaml path, a path that does not exist, garbage / empty / nc-mismatched data.yaml, an unlabeled folder,
  0 images, 1 image. Generate sample set (`synth.mjs`, `gen.mjs`): unicode-only, `..`, `x/../../evil`, `C:\evil`,
  300 characters, duplicates, device names, bad counts, and a 12,000-image 64 px set.
- **Hostile content** (`mksets.mjs`): class ids out of range, negative, NaN, zero-area and >1 boxes, extra columns,
  missing and empty label files, an empty val split, and the seed kit's odd images (1x1, 8000x6000, 30 MB PNG,
  CMYK, 16-bit, WebP, EXIF orientation, animated GIF, zero-byte, truncated, text-as-jpg, unicode file name).
- **Numbers.** Splits, image and box counts, per-class counts, empty images and the boxes-per-image histogram
  checked against an independent count of the label files (`count.mjs`) for shapes, spaces-unicode, empty-val and
  odd-yolo; every bad label line of badlabels reported with its kind, line and fix.
- **The page.** Grid fast scrolling on 12,000 images (`grid.mjs`: wheel bursts, End, back and forth, then the
  inspector by keyboard), inspector deep links (`?image=-1|99999|abc`), garbage hashes and filters
  (`#/datasets/ds_nope`, `..%2Fruns`, `?class=99`, `?split=bogus`), the broken seed dataset ds_7816, two tabs on one
  dataset with a removal in one (`twotab.mjs`), double-click and Enter on Register (`dbl.mjs`), 800x600 and 390 px
  wide layouts, image endpoint traversal (`..%2F..%2Fdata.yaml`, `-1`, `1e3`).
- Datasets have no rename in Studio or the API, so rename was not tested.

## Findings

| id | sev | what | status |
|---|---|---|---|
| s1-datasets-1 | annoying | A dataset page keeps showing a dataset removed in another tab: its re-read gets 404, the page stays, every image says "Image unavailable · Retry" forever | fixed: `web/src/routes/Dataset.tsx` (a 404 on the quiet re-read shows "Dataset not found"); test `datasetPage.test.tsx` "says the dataset is not found when another tab removed it" |
| s1-datasets-2 | annoying | Fast scrolling the grid queues 5,800-6,200 thumbnail requests that are never cancelled; at the inspector 849 were still pending and its image stayed blank 2-5 s | fixed: `web/src/lib/detect/DetectionView.tsx` (`cancelOnRemove` clears the `src` of an `<img>` as it leaves the page; Chrome keeps loading a detached image with a `src`, proved by `imgcancel.mjs`: 400/400 finished vs 400/400 cancelled); test `DatasetImage.test.tsx` "stops loading an image that leaves the page" |
| s1-datasets-5 | annoying | A sample set named `NUL` (any case) fails with "'…/datasets/nul' does not exist": the folder slug was the Windows null device, so the generator wrote the whole set into nothing | fixed: `crates/folo-server/src/util.rs` `slug` appends `-set` to `con prn aux nul com0-9 lpt0-9` (also used by uploads and labeling exports); unit test `slugs_never_name_a_windows_device`; docs/API.md synthetic line |
| s1-datasets-8 | annoying | After the server comes back from an outage, every grid tile that failed meanwhile keeps "Image unavailable · Retry" (the page's data recovers by itself, the images need a click each) | fixed: `web/src/lib/detect/DatasetImage.tsx` retries an image in error when `hub.back` bumps, as a Resource does; test `DatasetImage.test.tsx` "tries a failed image again by itself when the server answers again after an outage"; verified in headless Chrome (`killmid.mjs`) |
| s1-datasets-6 | polish | The Datasets header counts the images of a dataset that can no longer be read (seed ds_7816: "12,369 images" includes 24 that are gone) | fixed: `web/src/routes/Datasets.tsx` facts skip rows with `problem`; test in `datasetsList.test.tsx`. The card still shows the registration's count next to "Can't read" (says what it was) |
| s1-datasets-3 | polish | One folder registers twice: once by its folder and once by its `data.yaml` path, or by a `\\localhost\D$\…` share path (16.5 s). `\\?\`, slashes, case, `..` are recognised | deferred: `crates/folo-server/src/real/datasets.rs:126-133` compares canonical paths of what was given; a yaml path canonicalises to the file and a yaml file may point anywhere, so "same dataset" for it is a design call |
| s1-datasets-4 | polish | An animated GIF in a dataset is neither listed nor reported as an issue; its label file is ignored too | deferred: GIF is not a supported image format; the gap is that unsupported image files are not reported by health |

Repro steps and evidence for each are in `D:/folo-build/stress/lanes/s1-datasets/FINDINGS.md`.

## Checked and fine

- Every bad path, file path, broken data.yaml, 0-image, unlabeled and relative-path case in the Add dialog gives a
  clear message with a way forward; a bad name gets a one-click fix; double-click and Enter on Register send one POST.
- Stats equal the independent label count for every set checked; every bad label line is reported.
- Thumbnails and full images of every odd image answer 200; corrupt ones are reported as issues; traversal ids 404.
- Garbage URLs show "Dataset not found"; out-of-range `?image=` is dropped; junk filters fall back to All (the junk
  stays in the URL).
- The broken seed dataset shows "Could not load this dataset" with the reason, Remove from FOLO and Retry.
- No horizontal overflow at 800x600 or 390 px on the index, the dataset page, the inspector or the Add dialog.

## Not done

- Desktop's side of findings 6 and 8 (review round 1, `SIBLING`, left to the Desktop lanes): its Datasets header
  still adds an unreadable dataset's images (`apps/desktop/src/screens/datasets/list.rs:21`), and a thumbnail that
  failed during an outage stays `ImgSlot::Failed` after recovery (`apps/desktop/src/store/images.rs:73`,
  `apps/desktop/src/app/connection.rs:156`).

## Review round 1

- The recovery retry of finding 8 now also covers an image still loading when the server came back and failing
  after (`DatasetImage.tsx` `failed`: an attempt started before the last recovery tries again).
- A dataset page whose folder stops being a dataset while open (broken `data.yaml`, a 400 on the quiet re-read) shows
  "Could not load this dataset", as on opening it, not the old grid and Train (`Dataset.tsx`, sibling of finding 1).
- An unreadable dataset's card and phone row say "24 images (last known)" (`DatasetCard.tsx` `cardFacts`).
- The inspector's neighbour prefetches and the benchmark gallery's images stop loading when they leave (siblings of
  finding 2: `Inspector.tsx`, `bench/entry/MatchView.tsx`).

## Also seen

- Fix 2 re-measured in headless Chrome against a `--web-dir` build of this branch (`grid.mjs`, same scroll): 0
  thumbnail requests still pending at the inspector (849 before), its image shows at once.
- The Statistics tab of the 12,000-image set matches the API and the label count (lane `FINDINGS.md`).
- Removing a dataset that runs used (registration only): the Runs list says "shapes (removed)", the run page
  header still links "shapes" to a page that says "Dataset not found" (polish, Runs surface, not fixed here).
  Registering the folder again gives the same id back and the run's link works again, as docs/API.md promises.
