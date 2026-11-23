# folo-data: datasets, augmentation, loading

`crates/folo-data` turns images and labels on disk into training batches. It does not depend on folo-core: it
produces plain buffers that the trainer wraps in tensors.

## Formats

`folo_data::open(path)` detects the format; `open_with(path, &OpenOptions { format, verify_images })` forces one.

| `path` | Detected as |
|---|---|
| `*.yaml` / `*.yml` file | YOLO `data.yaml` |
| `*.json` file | COCO instances JSON (one split, guessed from the name) |
| directory with `data.yaml` / `dataset.yaml` (or any YAML with a `train` key; `.yaml` / `.yml` in any case) | YOLO (a YAML without split keys, e.g. only `names`, reads the `images/` + `labels/` layout next to it) |
| directory with `annotations/instances_*.json`, COCO JSON at the root, or `train|val|valid|test/*.json` (Roboflow) | COCO |
| directory with `images/` + `labels/` (or `train/images`, `valid/images`, ...) and no YAML | YOLO, names from `classes.txt` or `class0..` |

### YOLO
- `data.yaml` keys: `path` (dataset root, relative to the YAML; default the YAML's folder), `train`, `val` (or
  `valid`), `test`: each a folder (searched recursively), a `.txt` image list, an image, or a list of those.
  Roboflow-style `../train/images` paths are resolved against the root too. `names` is a list or an `{id: name}` map;
  `nc` must agree with it when both are given.
- Image list lines: absolute, `./x` (relative to the list file), or relative to the root.
- Text files (`data.yaml`, image lists, `classes.txt`, label files) may start with a UTF-8 byte order mark, which
  is ignored, and may use CRLF line ends.
- Label path: the last `images` folder becomes `labels`, the extension `.txt` (`ds/images/train/a.jpg` →
  `ds/labels/train/a.txt`). With no `images` folder the label sits next to its image. An `images` folder at or above
  the dataset root counts only if the root's mirror exists (`x/labels/birds` for a root `x/images/birds`, `x/labels`
  for `x/images`); without it the labels sit next to the images, and another dataset's folder in `x/labels` changes
  nothing. With the mirror, a `.txt` beside an image is not its label. The `images` folder is looked for in the
  image's path as the dataset spells it: a dataset opened as a bare `data.yaml` from inside `x/images/birds` with
  relative split paths has none in its image paths and reads the labels next to the images, mirror or not (open
  it by a path that names the `images` folder to use the mirror). A missing label file means a background image.
- Label lines: `class cx cy w h` or a segment polygon `class x1 y1 x2 y2 ...` (normalised; the polygon's bounding
  box is used). Coordinates may exceed `[0, 1]` by up to 0.01 (clipped, with a warning); exact duplicates are
  removed; zero-size boxes are dropped.
- A label file is **broken** (and its image skipped) if a value is not a finite number, a line has the wrong number of
  values, a class id is negative, fractional or `>= nc`, a size is negative, or coordinates are clearly not
  normalised.

### COCO
- A file's split is the first of `train`, `val`, `test` found anywhere in its name, then in its folder's name
  (`instances_val2017.json`, `valsubset.json`, `valid/_annotations.coco.json`); with neither it is `train`. A short
  list of ordinary words that contain `val` or `test` (`valves`, `festival`, `latest`, `contest`, `protest`, ...) does
  not count when it is a whole run of letters; an unlisted word (`rival` is listed, `strain` is not) still matches.
- `images[].file_name` is resolved against the first folder that has the files: the JSON's folder, `images/<tag>`,
  `<tag>` (where `instances_train2017.json` has tag `train2017`), `images/<split>`, `<split>`, `images/`, the root.
- `bbox` is pixel `[x, y, w, h]`; boxes are clipped to the image and normalised. Crowd annotations (`iscrowd` of `1`, `1.0` or
  `true`) are skipped.
- Category ids (from all JSON files) are sorted and remapped to `0..n`; names come from `categories`. An integral
  float is its integer (`"category_id": 1.0` names `"id": 1`).
- An image with an annotation that has a bad `bbox` or an unknown category is skipped as a broken label.
- Image ids are numbers or strings; an integral float is its integer (`"image_id": 1.0` names `"id": 1`). Two
  `images` entries with one id: the first is kept with every annotation of the id, the second is left out and
  reported (`other`).
- The annotation array is read as borrowed raw JSON, and only `image_id`, `category_id`, `bbox` and `iscrowd` are
  parsed out of each entry (segmentation polygons are skipped unbuilt): COCO 2017's `instances_train2017.json`
  (448 MB, 860k annotations) opens with a peak of 1.2 GB instead of 7 GB, with identical results. The official COCO
  folder (`annotations/`, `train2017/`, `val2017/`) opens directly; see examples/coco.
- `read_coco_ground_truth(json)` reads one instances file strictly, with what the loader drops (crowd regions, the
  `area` field and boxes in f64, image and category ids); `open_coco_ground_truth(path, split)` does that for a
  split's file and resolves every image's path exactly as the loader does, so dataset entries match COCO images by
  path. `coco_annotation_files(path)` lists a COCO dataset's annotation files per split. Entries and the top level
  are read as structs only when they are JSON objects (a positional array would otherwise fill one). `folo val --coco` uses them for pycocotools-exact evaluation (docs/METRICS.md).

`open_splits(path, &opts, &[Split::Val])` opens some splits only: those come out exactly as `open_with` builds
them (same classes, entries and order) and the others empty, without reading or probing their images and labels.
COCO files of the other splits are still read for their categories (every file counts for the class ids); a YOLO
dataset without class names (counted from all labels then) is opened whole. `folo val` uses it, so COCO val2017
opens without checking train2017's 118 287 images (the probe of a file not opened before costs about 6 ms on our test
machine (an i7-14700KF), 0.35 ms each at 28 threads: 40-50 s for train2017).

### Scan cache
Opening a large dataset costs a header probe per image (COCO train2017: 30-80 s on our test machine when the files
were not opened since boot) and, for COCO, a parse of the JSON (5-8 s for train2017's 448 MB). With
`folo_data::set_scan_cache_dir(Some(dir))` (`folo` sets `<FOLO_HOME>/cache/scan`, docs/CLI.md Caches; off by default)
an open keeps what it learnt in `dir` and the next one reuses it (`dataset/cache/`):

- **Per folder** (`dir-<hash>.bin`, keyed by the folder's canonical path, links resolved; a folder that cannot be
  canonicalised or whose path is not UTF-8 is not cached): every image's probe result (its size, or
  the codec's message when it cannot be decoded) and every YOLO label's parse (boxes and warnings, or why it is
  broken, with the class count it was checked against), each with the file's length and modification time. One
  directory read gives those for every file of the folder (free on Windows, where a `stat` opens the file), and a
  file is reused only while both match. IO errors are never cached, and files modified in the last 2 s are not
  cached (a same-tick rewrite could keep both).
- **Per COCO annotation set** (`coco-<hash>.bin`, keyed by the JSON paths as given and canonical, and the wanted
  splits): everything derived from the JSON files alone (classes, issues, each image's boxes), reused only while
  every JSON file keeps its length and modification time. The image folder is still chosen and the images still
  checked against the disk (through the per-folder entries) at every open.

An open comes out exactly as without the cache (the same entries in the same order and the same issues and
messages) as long as every change to a file changes its length or modification time: a file rewritten under an
unchanged stamp is answered from the cache (tested, as are corrupt, truncated and stale cache files). The folder
listing is a snapshot taken when the open starts, so a file changed or deleted while the open runs may be seen as
it was then (without the cache an open is racy the same way; the dataset is the one at the open). A file listed but
gone when its header is probed is reported as missing, as without the cache. Every cache file is checked as a
whole (its length, at most 1 GiB and matching its header, then magic, format version, its key and an FNV-1a
checksum of its contents) and ignored when anything is off; writes are atomic (a temporary file
`.<name>.<pid>-<n>.tmp` + rename), and cache files (`dir-<16 hex>.bin`, `coco-<16 hex>.bin` and those temporary
files, never any other file of the folder) not written for 60 days are deleted when the next one is written, and so
are the least recently written ones while the folder's cache files take more than 2 GiB together (`MAX_TOTAL`; a
single file over 1 GiB is never written). Only folders holding at least `SCAN_CACHE_MIN_FILES` (256) files of a dataset
are listed and cached, and COCO sets with at least that many images. With a read guard installed nothing is listed
or cached. COCO train + val on our test machine: 51 s cold, 0.3 s warm (docs/notes/hill/r2-trainer.md).

`LoadedDataset::sources` lists every file and folder a dataset was read from (description files, split folders and
image lists, folders of every candidate image and label, split paths that do not exist yet), so callers such as
the server can detect changes.

### Robustness
Nothing in a dataset panics the loader. While opening, every image header is read in parallel (`verify_images`,
default on): missing and undecodable images are skipped and recorded in `LoadedDataset::report` with an `IssueKind`
(`corrupt_image`, `missing_image`, `broken_label`, `label_warning`, `other`). JPEG EXIF orientation is applied on
decode (and when probing sizes), so boxes refer to the upright image. If an image still fails to decode during
training, the loader records it (`DataLoader::failures`) and uses the next readable sample in its place
(`ImageMeta::index` names the image actually used).

### Read guard
`folo_data::guard::install(allow)` sets a process-wide check on every file and folder the dataset reader opens
(manifests, image lists, split folders, COCO JSON, images, labels, `check` included). A refused path fails like an
unreadable file, with a message containing `folo_data::guard::DENIED`; a refused folder is reported once and not
descended. `guard::installed()` tells callers to re-read instead of trusting caches (the server's `RealStore`
rebuilds its dataset indexes then). Nothing installs one by default; `folo mcp
--confine` does, to keep dataset reads inside `--data-dir` (docs/MCP.md).

### Statistics
`LoadedDataset::stats()` / `Dataset::stats()` return the `stats` object of `docs/API.md`: `boxes`, `empty_images`,
`broken_labels`, `class_counts`, `box_size_hist` (size = `sqrt(w * h)` of the normalised box, edges
`[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]`) and `boxes_per_image_hist` (edges `[0, 1, 2, 4, 8, 16, 32]`). Bins are
`[lo, hi)`; values past the last edge count in the last bin.

## Validation and conversion
- `check::check(path, &CheckOptions { format, decode })` reads every label line and every image (fully decoded
  unless `decode: false`) and returns a `CheckReport` with one `CheckIssue { kind, file, line, message, fix }` per
  problem: `broken_label`, `class_overflow`, `out_of_range_box`, `corrupt_image`, `missing_image` (errors: the loader
  skips the image), `clipped_box`, `zero_size_box`, `duplicate_label`, `duplicate_image` (exact byte duplicates,
  128-bit content hash), `other` (warnings) and `empty_image` (info). A YOLO label file no image reads is a
  `missing_image` too when its folder is a label folder inside the dataset root that the loader pairs with a separate
  folder still holding at least one image the dataset reads (`labels/train` with `images/train`), and no file of its
  stem (any extension or case) is in those image folders; links to one folder count as one, `classes.txt` is not a
  label. Other folders are not visited, so a label folder whose last image is gone, or a label whose image is a
  dangling link, is not reported. It never
  modifies files; `fix` says what a cleanup would do. The YOLO line rules are the loader's
  (`dataset::labels::parse_line`), so the checker and the loader cannot disagree. `folo dataset check` prints it
  (docs/CLI.md).
- `convert::convert(&loaded, out, Format::Yolo | Format::Coco)` writes an opened dataset in the other format
  (images copied unchanged, stems made unique per split, COCO category ids 1-based).
- `detect_format(path, format)` finds the format without reading labels; a path with nothing recognisable fails with
  a `DataError::Format` whose message starts with `UNDETECTED`. A folder a writer has not finished (below) is refused
  as unfinished first, whatever it holds.

### Unfinished datasets
Every writer (`synthetic::write_yolo`, `convert::convert`, the server's sample sets) starts with
`unfinished::claim(dir)`: it creates `dir` when missing, then a `.folo-unfinished` file in it with `create_new`, then,
holding that file, checks that nothing else is in the folder; the writer removes the file as its last step
(`Claim::finish`), so a write stopped part way (killed, Ctrl+C, a crash) leaves it. The claim fails with a
`DataError::Io` saying the folder "exists and is not an empty folder" and leaves the folder as it was: of kind
`AlreadyExists` when the file was there (another writer's folder, running or stopped), `DirectoryNotEmpty` when
other files are (it removes the file it created). Only one writer can hold a folder, and FOLO never clears or deletes
a marked one. Every reader (`open`, `open_splits`, `detect_format`, `check::check`) refuses a dataset whose folder
holds the file, found from where the dataset's files are (the folder, its `data.yaml` and the root that file's `path:`
names, a COCO JSON's own folder and the one above it), with `DataError::Unfinished` (CLI `dataset_unfinished`, exit
1; HTTP 400). A YAML elsewhere whose split paths point into a marked folder without a `path:` is not refused. To
recover, delete the folder and run the command again. When only removing the file failed, the error says so: the
dataset is complete and that one file is to be deleted.

## Packing
`pack::pack(&loaded, &mut out)` writes an opened dataset, of any format and layout, as one tar stream into any
`io::Write`, without temporary files: how FOLO sends a dataset to another FOLO server (`/api/uploads`, a cloud
worker; docs/CLOUD.md, Dataset). The stream is the canonical YOLO layout, as FOLO reads the dataset rather than as it
lies on disk:
- `data.yaml` first: the split folders that have images (`train: images/train`, ...) and `names` (every class, in
  order, as JSON-quoted strings);
- then for every split (train, val, test), every image in dataset order: `images/<split>/<index>.<ext>`, the source
  file's bytes and extension, followed by `labels/<split>/<index>.txt`, written from the parsed boxes (`class cx cy w
  h`, normalised, each `f32` in its shortest exact decimal; an empty file for an image without boxes). `<index>` is
  the image's 0-based place in its split, zero-padded to 7 digits (more for larger splits), so names sort in
  dataset order;
- ustar headers with mode `0644`, owner 0 and modification time 0, so the bytes depend only on the dataset; the
  end-of-archive marker last.

Images the loader skipped (corrupt, missing, broken labels) are not in the stream. `pack::packed_len(&loaded)` is
the stream's exact length from the image files' sizes (nothing is read), for progress. A dataset without training
or validation images is a `DataError::Config` before anything is written. The invariant (`tests/pack.rs`, from YOLO
and COCO sources): unpacked and opened again, the stream gives the same class names, split sizes, image order,
image bytes and sizes, and boxes (exactly). One exception: a COCO dataset with two identical boxes on one image
keeps one, as the YOLO reader drops duplicate label lines.

## Geometry
- Pixel `i` covers `[i, i + 1)`. Resize is bilinear with aligned pixel centres, so box edges scale exactly by
  `new / old`; shrinking by 2x or more first box-filters by an integer factor (no aliasing).
- `letterbox(img, w, h, opts)` keeps the aspect ratio, pads with grey 114 and centres (or top-left). `scale_up: false`
  never enlarges; `stride: Some(32)` pads only to the next multiple of the stride. The returned `Letterbox` holds
  `ratio = [new_w / w, new_h / h]` and `pad = [left, top]`: original `x` maps to `x * ratio[0] + pad[0]`.
  `unmap_xyxy` / `unmap_box` (and `ImageMeta::unmap_xyxy` on batches) map predictions back.
- `warp_affine` inverts the matrix and steps through each output row in 16.16 fixed point with 11-bit bilinear
  weights; boxes go through the same matrix by their four corners. When the source row does not change along an
  output row (no rotation or shear, as in the default augmentation), each output row blends its two source rows once
  into `u32` sums and interpolates along them, with the same products and the one rounding of the general path, so
  the same bytes. On x86_64 CPUs with AVX2, bilinear resize, planar/RGB0 letterbox, the warp's row path and general
  affine interior, and horizontal row flips run bit-exact 256-bit SIMD kernels (`src/simd/`, scalar fallback
  elsewhere).

## Augmentation
`AugmentConfig` deserialises from `TrainConfig.augment` (missing fields take the defaults):

| field | default | meaning |
|---|---|---|
| `mosaic` | 1.0 | probability of a 4-image mosaic (random centre in `[s/2, 3s/2)` of a `2s` canvas) |
| `mixup` | 0.0 | probability of blending in a second augmented sample (weight ≈ Beta(32, 32)) |
| `hsv` | `[0.015, 0.7, 0.4]` | hue shift (fraction of a turn), saturation and value gains: `1 ± U(-1, 1) * g` |
| `fliplr` / `flipud` | 0.5 / 0.0 | flip probabilities |
| `scale` | 0.5 | random scale `1 ± scale` |
| `translate` | 0.1 | random translation, fraction of the output size |
| `degrees` / `shear` | 0.0 / 0.0 | rotation and shear ranges in degrees |
| `close_mosaic` | 10 | mosaic and mixup are off for the last N epochs, at most half of the run (`min(N, floor(epochs / 2))`; needs `LoaderConfig::epochs`) |
| `min_box_px`, `min_area_ratio`, `max_aspect` | 2, 0.1, 100 | boxes narrower than 2 px, keeping < 10 % of their area, or thinner than 1:100 after a geometric op are dropped |

`TrainTransform` runs, per sample: mosaic (4 images resized to long side `s`) or letterbox to `s x s` → random affine
(centre, rotate + scale, shear, translate) to `s x s` → mixup → HSV (integer hexcone HSV with LUTs) → flips. The
mosaic and its affine run as one step (`mosaic_affine`, bit for bit `random_affine` of `mosaic4`): the warp reads the
four tiles where they lie on the `2s x 2s` canvas instead of painting the canvas first (with rotation or shear the
canvas is painted and warped as before). HSV uses explicit 256-bit AVX2 kernels on CPUs with AVX2 (packed `max`
lookups and exact division-free `/ 255` and `/ 65280` integer arithmetic), matching the scalar per-pixel kernel on all
2^24 colours. Inside `DataLoader`, workers write straight into their disjoint planar batch slots, fusing warp (or
letterbox), HSV, flips and planar `u8`/`f32` conversion in a single L1-resident row pass without allocating an
intermediate warp output HWC buffer or a separate collator copy pass (when mixup triggers, it blends two
materialised `imgsz x imgsz` interleaved images before running the fused HSV, flips and planar pass; every path stays
bit for bit identical, pinned by `tests/golden.rs`). With `augment: None` (validation) it only letterboxes straight into the planar slot and records
the geometry.

## DataLoader and determinism
- `LoaderConfig { batch_size, imgsz, workers, shuffle, drop_last, seed, augment, epochs, prefetch_batches,
  u8_images, cache }`. `workers` defaults to the logical CPU count, at most 16.
- `cache: CacheMode::Ram` (`folo train --cache ram`) keeps every image decoded and resized to long side `imgsz`
  (what `resize_long_side` makes of it) in memory, filled the first time a worker loads it, so later epochs skip
  the decode and the resize (for 1080p JPEGs about 89% of a mosaic sample). The transform gets the decoded size
  with the sample (`Fetched::orig_size`): its own resize of a cached image is a row copy, and the letterbox
  geometry (`ratio`, `orig_size`) is computed from the original size, so **batches are identical with and without
  the cache** (pixels, targets and meta; tested for mosaic + mixup, plain letterbox and validation). Before the
  loader starts, `ram_cache_bytes` (every image's resized size from the dataset's probed sizes, 3 bytes a pixel,
  plus its boxes and its entry) is reserved against `RAM_CACHE_SHARE` (80%) of the memory available to the
  process, and the loader is refused with the numbers when it does not fit. Available is the system's available
  memory and, on Linux, at most what the tightest memory limit among the process's cgroups and their ancestors
  has left (a container, a systemd scope, cgroup v2 or v1: the limit less the usage, not counting the inactive file
  cache; the process's cgroups from `/proc/self/cgroup`, located through the cgroup mounts in `/proc/self/mountinfo`,
  so co-mounted v1 controllers and mounted subtrees resolve). The reservation is process-wide and held until the
  loader is dropped, so two loaders created before either has filled are not both admitted against the same
  memory (a filled cache counts twice, in its reservation and in the memory it took: a further one is refused
  rather than risked). An image that cannot be read is not cached (the loader substitutes as without a cache).
- Batches: `images` is `n * 3 * s * s` `f32` in `0..1` (`NCHW`, RGB), or, with `u8_images`, `images_u8` holds the same
  pixels as `u8` (`value / 255` is the float; a quarter of the bytes, for a GPU that converts them itself, as the
  trainer and validation do on a GPU) and `images` is empty. `shape = [n, 3, s, s]`, `targets` rows are
  `[batch_idx, class, cx, cy, w, h]` normalised to the network input, `meta` has the dataset index, original size,
  letterbox ratio and pad of each image.
- The epoch order is a shuffle seeded by `(seed, epoch)`; sample position `p` draws all its randomness from
  `Rng::derive(&[seed, epoch, p])`. Workers claim positions from a shared counter, a collator thread restores the
  order, so **output is bit-identical for any number of workers** (including `workers: 0`, which runs inline).
- In-flight work is bounded (a window of positions plus `prefetch_batches` finished batches). Dropping the epoch
  iterator stops and joins every thread. `EpochIter::throughput()` reports images/s.
- Inside workers, image kernels run single-threaded (`folo_data::par::serial`); outside, they split rows with rayon.
- The workers and the collator reuse image-sized buffers (decoded, resized and warped images; `src/buffers.rs`): a
  fresh buffer of a megabyte faults its pages in on first write, which with 8-16 loader threads at once cost 0.5-1 ms
  per buffer on our test machine against 0.05-0.1 ms reused. Buffers are zeroed where `vec![0; n]` was and otherwise
  overwritten whole, so the bytes do not depend on the reuse. The pool keeps at most 512 MB of free buffers and is
  emptied when the loader's threads stop.
- `tests/golden.rs` pins the digests of the first batches for each path of the transform (mosaic, rotation and
  shear, mixup and up-down flips, letterbox + affine, validation), from memory and from JPEG and PNG files: a change
  that alters a batch's bytes fails there.

## Synthetic shapes
`synthetic::write_yolo(&cfg, dir, ImageFormat::Jpeg(90) | Png)` writes `images/{train,val}`, `labels/{train,val}`
and `data.yaml` (`write_yolo_with_progress` also reports each finished image); `SyntheticDataset::new(cfg, split)` renders the same images on demand. Classes are any of
`circle, square, triangle, star, ring, cross, hexagon, heart`, drawn with anti-aliased signed distance fields, soft
shadows and light shading on gradient, marbled, striped or grainy backgrounds, with small objects, partial occlusion,
objects cut by the frame and some empty images. Output is a pure function of the config and seed, with the same
bytes in every build (release or debug) and on every machine: the generator uses only operations whose result IEEE 754
specifies (`+ - * /` and `sqrt`, correctly rounded; `floor`, `ceil`, `round` and a remainder, exact; its sine,
cosine and powers of two are built from them in `synthetic/math.rs`), never an approximation from the platform's libm, whose last bits differ between systems and,
where the compiler folds a call whose argument it can see, between builds. `synthetic/tests.rs` pins the pixels and
boxes of 48 images, and `tests/synthetic.rs` the files of a small written dataset.
`SyntheticConfig::classes` are shape names; `names` (optional, same length) are the class names written to
`data.yaml`. `shapes_for(&names)` picks shapes for arbitrary names: a shape name keeps its shape, others take the
next unclaimed shape (at most 8 classes).
`cargo run --release -p folo-data --example synth_sheet -- sheet.png 0 --boxes` renders a contact sheet.

## Benchmark
`cargo bench -p folo-data --bench pipeline [-- --images N]` writes a synthetic 640 px JPEG dataset and reports stage
timings and loader images/s with 1, `min(8, cores)` and `cores` workers (override with `--workers A,B,...`).
`--data PATH` measures a real dataset (`SCAN_CACHE=DIR` opens
it through the scan cache), `--same-epoch` measures every round on epoch 1 after a warm-up pass over the same batches
(so two builds compared one after the other read equally warm files). `cargo run --release -p folo-data --example
loader_profile -- PATH [N]` splits a mosaic sample's time and allocations into its stages (file read, decode, resize,
mosaic + affine, HSV, flips, collate) on a real dataset; docs/notes/hill/r6-loader.md has the numbers.
