# folo-wasm

folo-wasm runs FOLO detectors in the browser: wasm-bindgen bindings over folo-vision's predictor, on WebGPU
(folo-gpu) or on the wasm CPU backend (folo-core, SIMD128). It loads a `.folo` file from bytes, and its results have
the shape of `POST /api/predict` in [API.md](../../docs/API.md). It depends on folo-api, folo-core, folo-data,
folo-gpu, folo-nn and folo-vision; no crate depends on it (`publish = false`). It builds as a `cdylib` for the
browser and as an `rlib`, so the workspace's native tests can use it. FOLO Studio loads the built module from
`folo-wasm/folo_wasm.js` under its base URL (`web/src/lib/inference/browser.ts`). [WASM.md](../../docs/WASM.md)
documents the JS API, how it works and its performance.

## Where things are

| Path | What it holds |
|---|---|
| `src/lib.rs` | The JS API: `FoloDetector` (`load`, `detect`, `detectTiled`, `detectTta`, `info`, `backend`), `FoloTracker` (`new`, `update`, `reset`, `config`, `info`), `hasWebGpu`, `version`, pooled RGBA input copies, the panic hook, and the hand-written TypeScript declarations (`TS_TYPES`) |
| `src/js_util.rs` | Safe catching `Reflect` traversal, strict `f64`-range option validation (for detector and tracker options), box array parsing, and detached-buffer-safe typed-array copies |
| `src/detector/mod.rs` | `Detector`: the logic behind `FoloDetector` for both `folo-*` (v1) and `folo2-*` (v2-Iso, including `o2o`, `iou_head`, and `folo.quant` CPU int8 checkpoints), free of wasm-bindgen types; batched candidate-compacted async GPU readback with typed OOM/limit chunk-splitting, async `iou_head` score fusion, reusable RGBA/RGB scratch buffers, and direct CPU path |
| `src/detector/options.rs` | `Backend`, `BackendChoice`, `Precision`, `LoadOptions`, `DetectOptions` (`perClassConf`, `confInclusive`), `TileDetectOptions`, `TtaDetectOptions`, `DetectorInfo`, `DetectResult`, and strict option/key validators |
| `src/detector/tiled.rs` | SAHI overlapping tiled inference (`Detector::detect_tiled`, `MAX_BROWSER_TILES`): `O(1)` tile-count cap, direct RGBA tile cropping on CPU, batched tile execution on WebGPU, optional full-image pass, and deterministic IoS/IoU merging |
| `src/detector/tta.rs` | Multi-scale & horizontal-flip Test-Time Augmentation (`Detector::detect_tta`): SIMD128 RGBA horizontal mirror on CPU, batched views on WebGPU, box un-flipping, and cross-view fusion |
| `src/tracker/mod.rs` | `Tracker`: pure-Rust ByteTrack multi-object tracker (`TrackerOptions`, `TrackerConfig`, `TrackerInfo`, `TrackInputBox`, `TrackedBox`) over `folo_vision::ByteTracker`, free of wasm-bindgen types |
| `src/simd/mod.rs`, `src/simd/wasm_simd.rs` | WASM SIMD128 (and portable fallback) RGBA→RGB stripping, box-prefilter downsampling, 2×4-unrolled direct RGBA→planar-`f32` letterbox preprocessing with reusable `RgbaScratch`, RGBA tile cropping, and 4-pixel-per-op RGBA horizontal flipping |
| `tests/detector/` | Native detector parity tests (`main.rs`, `thresholds.rs`, `int8.rs`): `folo-*` (v1) and `folo2-*` (v2-Iso, `o2m` / `o2m + iou_head` / `o2o`) single-pass, tiled, and TTA CPU & GPU paths against `folo_vision::Predictor`, per-class confidence thresholds (`perClassConf`) & inclusive threshold parity (`confInclusive`), quantized int8 `.folo` loading & CPU parity across `folo-*` and `folo2-*` (`o2m`, `iou_head`, `o2o`) with `perClassConf` / `confInclusive` across single-pass, tiled, and TTA, invalid parameter rejection, JSON shape, and options parsing |
| `tests/tracker.rs` | Native tracker integration tests: ByteTrack parity with `folo_vision::ByteTracker`, multi-frame ID persistence, stage-2 low-confidence occlusion recovery, coasting/retirement, `min_hits`, `class_aware`, `reset()`, end-to-end `folo2` → `Tracker`, and invalid config rejection |
| `demo/index.html` | A standalone page: pick a model and an image, pick the backend, detect, benchmark |
| `demo/test.mjs` | End-to-end check in headless Chrome (Playwright): each backend, latency, boxes against `folo predict --json` |
| `demo/api-check.ts`, `demo/tsconfig.json` | Type-checks the documented JS API against the generated declarations |

## Building

```sh
sh scripts/build-wasm.sh              # release build into web/public/folo-wasm/
sh scripts/build-wasm.sh --dev        # unoptimised, faster to build
sh scripts/build-wasm.sh --out DIR    # somewhere else
```

`scripts/build-wasm.ps1` is the PowerShell version. The script writes `folo_wasm.js` (an ES module),
`folo_wasm_bg.wasm` and their TypeScript declarations. It needs the `wasm32-unknown-unknown` target (added with
rustup when missing) and `wasm-bindgen-cli` at exactly the `wasm-bindgen` version in `Cargo.lock`; it builds with
`-C target-feature=+simd128` and runs `wasm-opt` on a release build when it is on `PATH`.

## Start here

1. The module docs and `FoloDetector` in `src/lib.rs`: the whole JS API in one place.
2. `Detector::load` in `src/detector/mod.rs`: parsing the `.folo` file and picking WebGPU or the CPU (`auto` checks
   WebGPU with one small forward pass and falls back to the CPU).
3. `Detector::detect` (`src/detector/mod.rs`), `Detector::detect_tiled` (`src/detector/tiled.rs`), and
   `Detector::detect_tta` (`src/detector/tta.rs`): folo-vision's `Predictor` pipeline, with batched device-side
   candidate extraction and an awaited `folo_gpu::download_async` on WebGPU (~28 KiB per image, with OOM
   chunk-splitting and a dense fallback), direct SIMD128 RGBA→planar-`f32` letterboxing + candidate-first decode
   on the CPU, and reusable RGBA/RGB scratch buffers across repeated frames.

## Common changes

| To do this | Touch these files |
|---|---|
| Add or change an option or a result field | `LoadOptions`, `DetectOptions`, `TileDetectOptions`, `TtaDetectOptions` or `DetectorInfo` in `src/detector/options.rs`; a result field in `DetectResult` (`src/detector/options.rs`) or in `folo_api::PredictResponse`, which it flattens; the matching interface in `TS_TYPES` in `src/lib.rs`; `tests/detector/`; `demo/api-check.ts`; WASM.md (JS API) |
| Change the JS bindings | `src/lib.rs` (keep `TS_TYPES` in step: `load`, `detect`, `detectTiled`, and `detectTta` are declared there, not generated); `demo/api-check.ts`; WASM.md |

## Tests

```sh
cargo test -p folo-wasm                                  # native: CPU path, GPU path, JSON shape, options
sh scripts/build-wasm.sh && web/node_modules/.bin/tsc -p crates/folo-wasm/demo/tsconfig.json
node crates/folo-wasm/demo/test.mjs --model shapes-n.folo --image img.jpg [--expect native.json]
```

The GPU test in `tests/detector/` skips without a hardware adapter (`FOLO_GPU_TEST_SOFTWARE=1` allows a software
one). `demo/test.mjs` needs `playwright-core`, from `web/node_modules` (`npm ci` in `web/`) or anywhere Node resolves
it, and a Chrome. No test is `#[ignore]`d.

## Further reading

- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [WASM.md](../../docs/WASM.md): building, the JS API, how it works, performance, the demo and its limits.
- Neighbours: [folo-vision](../folo-vision/README.md) (the predictor it wraps), [folo-gpu](../folo-gpu/README.md)
  (WebGPU through wgpu), [folo-core](../folo-core/README.md) (the CPU kernels).
