# folo-data: datasets, augmentation, loading

`crates/folo-data` turns images and labels on disk into training batches. It does not depend on folo-core: it
produces plain buffers that the trainer wraps in tensors.

## Formats

`folo_data::open(path)` detects the format; `open_with(path, &OpenOptions { format, verify_images })` forces one.

| `path` | Detected as |
|---|---|
| `*.yaml` / `*.yml` file | YOLO `data.yaml` |
| `*.json` file | COCO instances JSON (one split, guessed from the name) |
| directory with `data.yaml` / `dataset.yaml` (or any YAML with a `train` key) | YOLO (a YAML without split keys, e.g. only `names`, reads the `images/` + `labels/` layout next to it) |
| directory with `annotations/instances_*.json`, COCO JSON at the root, or `train|val|valid|test/*.json` (Roboflow) | COCO |
| directory with `images/` + `labels/` (or `train/images`, `valid/images`, ...) and no YAML | YOLO, names from `classes.txt` or `class0..` |

### YOLO
- `data.yaml` keys: `path` (dataset root, relative to the YAML; default the YAML's folder), `train`, `val` (or
  `valid`), `test`: each a folder (searched recursively), a `.txt` image list, an image, or a list of those.
  Roboflow-style `../train/images` paths are resolved against the root too. `names` is a list or an `{id: name}` map;
  `nc` must agree with it when both are given.
- Image list lines: absolute, `./x` (relative to the list file), or relative to the root.
- Label path: the last `images` folder becomes `labels`, the extension `.txt` (`ds/images/train/a.jpg` →
  `ds/labels/train/a.txt`). A missing label file means a background image.
- Label lines: `class cx cy w h` or a segment polygon `class x1 y1 x2 y2 ...` (normalised; the polygon's bounding
  box is used). Coordinates may exceed `[0, 1]` by up to 0.01 (clipped, with a warning); exact duplicates are
  removed; zero-size boxes are dropped.
- A label file is **broken** (and its image skipped) if a value is not a finite number, a line has the wrong number of
  values, a class id is negative, fractional or `>= nc`, a size is negative, or coordinates are clearly not
  normalised.

### COCO
- `images[].file_name` is resolved against the first folder that has the files: the JSON's folder, `images/<tag>`,
  `<tag>` (where `instances_train2017.json` has tag `train2017`), `images/<split>`, `<split>`, `images/`, the root.
- `bbox` is pixel `[x, y, w, h]`; boxes are clipped to the image and normalised. `iscrowd: 1` annotations are skipped.
- Category ids (from all JSON files) are sorted and remapped to `0..n`; names come from `categories`.
- An image with an annotation that has a bad `bbox` or an unknown category is skipped as a broken label.

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
  128-bit content hash), `other` (warnings) and `empty_image` (info). It never modifies files; `fix` says what a
  cleanup would do. The YOLO line rules are the loader's (`dataset::labels::parse_line`), so the checker and the
  loader cannot disagree. `folo dataset check` prints it (docs/CLI.md).
- `convert::convert(&loaded, out, Format::Yolo | Format::Coco)` writes an opened dataset in the other format
  (images copied unchanged, stems made unique per split, COCO category ids 1-based).
- `detect_format(path, format)` finds the format without reading labels; a path with nothing recognisable fails with
  a `DataError::Format` whose message starts with `UNDETECTED`.

## Geometry
- Pixel `i` covers `[i, i + 1)`. Resize is bilinear with aligned pixel centres, so box edges scale exactly by
  `new / old`; shrinking by 2x or more first box-filters by an integer factor (no aliasing).
- `letterbox(img, w, h, opts)` keeps the aspect ratio, pads with grey 114 and centres (or top-left). `scale_up: false`
  never enlarges; `stride: Some(32)` pads only to the next multiple of the stride. The returned `Letterbox` holds
  `ratio = [new_w / w, new_h / h]` and `pad = [left, top]`: original `x` maps to `x * ratio[0] + pad[0]`.
  `unmap_xyxy` / `unmap_box` (and `ImageMeta::unmap_xyxy` on batches) map predictions back.
- `warp_affine` inverts the matrix and steps through each output row in 16.16 fixed point with 11-bit bilinear
  weights; boxes go through the same matrix by their four corners.

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
| `close_mosaic` | 10 | mosaic and mixup are off for the last N epochs (needs `LoaderConfig::epochs`) |
| `min_box_px`, `min_area_ratio`, `max_aspect` | 2, 0.1, 100 | boxes narrower than 2 px, keeping < 10 % of their area, or thinner than 1:100 after a geometric op are dropped |

`TrainTransform` runs, per sample: mosaic (4 images resized to long side `s`) or letterbox to `s x s` → random affine
(centre, rotate + scale, shear, translate) to `s x s` → mixup → HSV (integer hexcone HSV with LUTs) → flips. With
`augment: None` (validation) it only letterboxes and records the geometry.

## DataLoader and determinism
- `LoaderConfig { batch_size, imgsz, workers, shuffle, drop_last, seed, augment, epochs, prefetch_batches,
  u8_images }`.
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

## Synthetic shapes
`synthetic::write_yolo(&cfg, dir, ImageFormat::Jpeg(90) | Png)` writes `images/{train,val}`, `labels/{train,val}`
and `data.yaml` (`write_yolo_with_progress` also reports each finished image); `SyntheticDataset::new(cfg, split)` renders the same images on demand. Classes are any of
`circle, square, triangle, star, ring, cross, hexagon, heart`, drawn with anti-aliased signed distance fields, soft
shadows and light shading on gradient, marbled, striped or grainy backgrounds, with small objects, partial occlusion,
objects cut by the frame and some empty images. Output is a pure function of the config and seed.
`SyntheticConfig::classes` are shape names; `names` (optional, same length) are the class names written to
`data.yaml`. `shapes_for(&names)` picks shapes for arbitrary names: a shape name keeps its shape, others take the
next unclaimed shape (at most 8 classes).
`cargo run --release -p folo-data --example synth_sheet -- sheet.png 0 --boxes` renders a contact sheet.

## Benchmark
`cargo bench -p folo-data --bench pipeline [-- --images N]` writes a synthetic 640 px JPEG dataset and reports stage
timings and loader images/s with 1, N/2 and N workers.
