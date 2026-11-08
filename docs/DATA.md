# folo-data: datasets, augmentation, loading

`crates/folo-data` turns images and labels on disk into training batches. It does not depend on folo-core: it
produces plain buffers that the trainer wraps in tensors.

## Formats

`folo_data::open(path)` detects the format; `open_with(path, &OpenOptions { format, verify_images })` forces one.

| `path` | Detected as |
|---|---|
| `*.yaml` / `*.yml` file | YOLO `data.yaml` |
| `*.json` file | COCO instances JSON (one split, guessed from the name) |
| directory with `data.yaml` / `dataset.yaml` (or any YAML with a `train` key) | YOLO |
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

### Robustness
Nothing in a dataset panics the loader. While opening, every image header is read in parallel (`verify_images`,
default on): missing and undecodable images are skipped and recorded in `LoadedDataset::report` with an `IssueKind`
(`corrupt_image`, `missing_image`, `broken_label`, `label_warning`, `other`). JPEG EXIF orientation is applied on
decode (and when probing sizes), so boxes refer to the upright image. If an image still fails to decode during
training, the loader records it (`DataLoader::failures`) and uses the next readable sample in its place.

### Statistics
`LoadedDataset::stats()` / `Dataset::stats()` return the `stats` object of `docs/API.md`: `boxes`, `empty_images`,
`broken_labels`, `class_counts`, `box_size_hist` (size = `sqrt(w * h)` of the normalised box, edges
`[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]`) and `boxes_per_image_hist` (edges `[0, 1, 2, 4, 8, 16, 32]`). Bins are
`[lo, hi)`; values past the last edge count in the last bin.

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
- `LoaderConfig { batch_size, imgsz, workers, shuffle, drop_last, seed, augment, epochs, prefetch_batches }`.
- Batches: `images` is `n * 3 * s * s` `f32` in `0..1` (`NCHW`, RGB), `shape = [n, 3, s, s]`, `targets` rows are
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
and `data.yaml`; `SyntheticDataset::new(cfg, split)` renders the same images on demand. Classes are any of
`circle, square, triangle, star, ring, cross, hexagon, heart`, drawn with anti-aliased signed distance fields, soft
shadows and light shading on gradient, marbled, striped or grainy backgrounds, with small objects, partial occlusion,
objects cut by the frame and some empty images. Output is a pure function of the config and seed.
`cargo run --release -p folo-data --example synth_sheet -- sheet.png 0 --boxes` renders a contact sheet.

## Benchmark
`cargo bench -p folo-data --bench pipeline [-- --images N]` writes a synthetic 640 px JPEG dataset and reports stage
timings and loader images/s with 1, N/2 and N workers.
