# folo-data

folo-data reads datasets and turns them into training batches: YOLO and COCO readers, image decoding, letterbox and
resize, box-aware augmentation, a deterministic multi-threaded data loader and procedural synthetic datasets. It
depends on no other workspace crate (not even folo-core): it hands out plain buffers (`Vec<f32>` images in `NCHW`
and `[batch_idx, class, cx, cy, w, h]` target rows) that the trainer wraps in tensors, and it carries its own seeded
RNG (`src/rng.rs`). folo-vision, folo-bench, folo-server, folo-mcp, folo-cli, folo-wasm and FOLO Desktop
(`apps/desktop`) depend on it. [DATA.md](../../docs/DATA.md) describes the formats, the augmentation parameters and
the determinism rules.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | Crate docs with an example, and the re-exports |
| `src/dataset/` | `mod.rs` (the `Dataset` trait, `Split`, `Format`, `open` / `open_with`, format detection, `LoadReport`), `yolo.rs` and `labels.rs` (YOLO), `coco.rs` (COCO instances JSON), `cache/` (the on-disk scan cache: `probe.rs`, `coco.rs`, `listing.rs`) |
| `src/image.rs` | `Image`, decoding with EXIF orientation, encoding, `Sample` |
| `src/bbox.rs` | `BBox`, normalised centre format |
| `src/letterbox.rs` | Letterbox resize with exact box mapping both ways |
| `src/resize.rs`, `src/warp.rs` | Bilinear resize and affine warp of 8-bit RGB images |
| `src/augment/` | `mod.rs` (`AugmentConfig`), `geometry.rs` (affine, flips), `hsv.rs`, `mosaic.rs` (mosaic and mixup), `pipeline.rs` (`TrainTransform`, the training transform) |
| `src/loader.rs` | `DataLoader`: worker threads, batches identical for a seed and epoch whatever the worker count |
| `src/ram_cache.rs` | `--cache ram`: decoded, resized training images kept in memory |
| `src/synthetic/` | `mod.rs` (the shapes generator, `write_yolo` / `write_yolo_into`, `SyntheticDataset`), `raster.rs` (rasteriser, backgrounds) |
| `src/check.rs` | `folo dataset check`: one `CheckIssue` per problem |
| `src/convert.rs` | `folo dataset convert`: writes an opened dataset as YOLO or COCO |
| `src/pack.rs` | A dataset as one tar stream in the canonical YOLO layout, for another FOLO server |
| `src/staging.rs` | `Staging`: a dataset folder written beside its place (`.<name>.partial`) and renamed there when complete; `write_yolo` and `convert` use it |
| `src/stats.rs` | `DatasetStats`, the `stats` object of docs/API.md |
| `src/guard.rs` | An optional process-wide read guard (`folo mcp --confine` installs one) |
| `src/par.rs` | Row parallelism that the loader's worker threads switch off |
| `src/paths.rs` | Path helpers shared by `folo dataset check --json` and the server |
| `src/rng.rs` | xoshiro256++ seeded through SplitMix64 |
| `src/error.rs` | `DataError` and `Result` |
| `tests/` | `formats.rs` (YOLO / COCO parsing, detection, EXIF), `augment.rs` (boxes stay on their pixels), `loader.rs` (determinism, shutdown), `pack.rs`, `synthetic.rs`, `tools.rs` (check and convert), `guard.rs` (its own binary: a guard stays installed for the process), `common/` (helpers) |
| `benches/pipeline.rs` | Stage timings and `DataLoader` images/s |
| `examples/` | `shapes.rs` (the MODEL.md §10.6 shapes dataset), `synth_sheet.rs` (a contact sheet of synthetic images), `aug_stats.rs` (statistics of the training transform's output), `open_time.rs` (times opening a dataset) |

## Start here

1. `open` and `LoadedDataset` in `src/dataset/mod.rs`: how a folder becomes a dataset, and what the load report
   records.
2. The `Dataset` trait in the same file: labels in memory, images decoded on `get`.
3. `DataLoader` in `src/loader.rs`: how samples are ordered, seeded and batched.
4. `TrainTransform` in `src/augment/pipeline.rs`: the training transform, mosaic to flips.
5. `Letterbox` and `letterbox` in `src/letterbox.rs`: the letterbox geometry that validation and folo-vision's
   predictor use.

## Common changes

| To do this | Touch these files |
|---|---|
| Add an augmentation | The function in `src/augment/` (a new file gets `mod` and `pub use` in `augment/mod.rs`); its parameter in `AugmentConfig` (`augment/mod.rs`); the step in `TrainTransform` (`augment/pipeline.rs`); a box / pixel check in `tests/augment.rs`; DATA.md (Augmentation) and MODEL.md §8.6. To expose it over the API: `Augment` in `crates/folo-api/src/runs.rs`, `TrainConfig::from_api` in `crates/folo-vision/src/train/config.rs`, and API.md |
| Change how YOLO or COCO is read | `src/dataset/yolo.rs`, `labels.rs` or `coco.rs`; `tests/formats.rs`; DATA.md (Formats). When a probe or a label parse returns something else, bump `FORMAT` in `src/dataset/cache/mod.rs` |
| Add a dataset check | `src/check.rs`; `tests/tools.rs`; DATA.md (Validation and conversion) |
| Add a synthetic shape | `SHAPES` and `shape_prims` in `src/synthetic/mod.rs` (primitives in `raster.rs`); `tests/synthetic.rs`; DATA.md (Synthetic shapes) |

## Tests and benchmarks

```sh
cargo test -p folo-data
cargo bench -p folo-data --bench pipeline [-- --data PATH --workers 1,8 --cache ram ...]
cargo run --release -p folo-data --example synth_sheet -- out.png [seed] [--boxes] [--augment] [--all]
```

Nothing here needs a GPU, and no test is `#[ignore]`d. Fixtures are written by the tests themselves, into
`folo_test_support::TestDir` folders.

## Further reading

- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [DATA.md](../../docs/DATA.md): formats, scan cache, augmentation, the loader and its determinism.
- [MODEL.md](../../docs/MODEL.md) §8.6 (data and augmentation in the recipe) and §9.2 (letterbox).
- Neighbours: [folo-vision](../folo-vision/README.md) (the trainer that drives the loader),
  [folo-bench](../folo-bench/README.md) (decodes images through `folo_data`).
