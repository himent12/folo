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
| `src/lib.rs` | The JS API: `FoloDetector` (`load`, `detect`, `info`, `backend`), `hasWebGpu`, `version`, the panic hook, and the hand-written TypeScript declarations (`TS_TYPES`) |
| `src/detector.rs` | `Detector`: the logic behind the bindings, free of wasm-bindgen types; `BackendChoice`, `LoadOptions`, `DetectOptions`, `DetectorInfo`; candidate-compacted async GPU readback and direct CPU path |
| `src/simd.rs` | WASM SIMD128 (and portable fallback) RGBA→RGB stripping, box-prefilter downsampling, and 2×4-unrolled direct RGBA→planar-`f32` letterbox preprocessing |
| `src/kernels/` | WASM SIMD128 (and portable fallback) inference/post-processing kernels: `pool.rs` (2D max/avg pooling), `upsample.rs` (2D nearest/bilinear upsample), `post.rs` (vectorized softmax and batched box-IoU) |
| `tests/detector.rs` | The detector run natively: the CPU path against folo-vision's `Predictor`, the GPU path against the CPU path, the JSON shape, the options |
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
2. `Detector::load` in `src/detector.rs`: parsing the `.folo` file and picking WebGPU or the CPU (`auto` checks
   WebGPU with one small forward pass and falls back to the CPU).
3. `Detector::detect`: folo-vision's `Predictor` pipeline, with device-side candidate extraction and an awaited
   `folo_gpu::download_async` on WebGPU (downloading only the above-threshold candidates, ~28 KiB per frame,
   with a dense fallback) and direct SIMD128 RGBA→planar-`f32` letterboxing + candidate-first decode on the CPU.

## Common changes

| To do this | Touch these files |
|---|---|
| Add or change an option or a result field | `LoadOptions`, `DetectOptions` or `DetectorInfo` in `src/detector.rs`; a result field in `DetectResult` (`src/detector.rs`) or in `folo_api::PredictResponse`, which it flattens; the matching interface in `TS_TYPES` in `src/lib.rs`; `tests/detector.rs`; `demo/api-check.ts`; WASM.md (JS API) |
| Change the JS bindings | `src/lib.rs` (keep `TS_TYPES` in step: `load` and `detect` are declared there, not generated); `demo/api-check.ts`; WASM.md |

## Tests

```sh
cargo test -p folo-wasm                                  # native: CPU path, GPU path, JSON shape, options
sh scripts/build-wasm.sh && web/node_modules/.bin/tsc -p crates/folo-wasm/demo/tsconfig.json
node crates/folo-wasm/demo/test.mjs --model shapes-n.folo --image img.jpg [--expect native.json]
```

The GPU test in `tests/detector.rs` skips without a hardware adapter (`FOLO_GPU_TEST_SOFTWARE=1` allows a software
one). `demo/test.mjs` needs `playwright-core`, from `web/node_modules` (`npm ci` in `web/`) or anywhere Node resolves
it, and a Chrome. No test is `#[ignore]`d.

## Further reading

- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md): how the crates fit together, and where to make a change.
- [WASM.md](../../docs/WASM.md): building, the JS API, how it works, performance, the demo and its limits.
- Neighbours: [folo-vision](../folo-vision/README.md) (the predictor it wraps), [folo-gpu](../folo-gpu/README.md)
  (WebGPU through wgpu), [folo-core](../folo-core/README.md) (the CPU kernels).
