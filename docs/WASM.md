# FOLO in the browser (folo-wasm)

`crates/folo-wasm` runs FOLO detectors entirely in the page: it loads a `.folo` file from bytes and detects objects
on **WebGPU** (folo-gpu's kernels through wgpu's browser backend) or, without WebGPU, on the **CPU** (folo-core's
kernels compiled to wasm with SIMD128). Results have the shape of `POST /api/predict` (docs/API.md), so a page can
switch between the server and the browser without changing how it reads them.

## Building

```sh
sh scripts/build-wasm.sh            # or: powershell -File scripts\build-wasm.ps1
```

writes `web/public/folo-wasm/` (git-ignored): `folo_wasm.js` (an ES module), `folo_wasm_bg.wasm` (about 5.8 MB,
0.75 MB gzipped) and TypeScript declarations (`folo_wasm.d.ts`). Requirements: the `wasm32-unknown-unknown` target
(the script adds it with rustup) and `wasm-bindgen-cli` at exactly the version of `wasm-bindgen` in `Cargo.lock`
(`cargo install wasm-bindgen-cli --version <v> --locked`; the script looks on `PATH` and in `$FOLO_TOOLS/bin`).
`--dev` builds unoptimised, `--out DIR` writes elsewhere, `--locked` (`-Locked`) refuses to update `Cargo.lock`; `wasm-opt` is applied when it is on `PATH`. CI's `wasm` job
runs the same script on every PR, and the release workflow runs it (with a pinned `wasm-opt`) before `vite build`,
so `web/dist/folo-wasm/` and the Studio embedded in release binaries include it (`folo doctor` reports
`web_folo_wasm`; docs/RELEASE.md). Vite serves `web/public/` as is, so Studio imports it as
`/folo-wasm/folo_wasm.js`.

## JS API

```js
import init, { FoloDetector, hasWebGpu, version } from "/folo-wasm/folo_wasm.js";

await init();                                          // fetches and instantiates folo_wasm_bg.wasm (once)
// Any .folo bytes; from folo serve: GET /api/models/{id}/weights (docs/API.md), ?format=f16 for the small export.
const bytes = new Uint8Array(await (await fetch("/api/models/m_31aa/weights?format=f16")).arrayBuffer());
const det = await FoloDetector.load(bytes, { backend: "auto", modelId: "m_31aa" });
det.backend;                                           // "webgpu" | "cpu"
det.info();                                            // see below

const res = await det.detect(imageData, undefined, undefined, { conf: 0.25, iou: 0.7, maxDet: 300 });
// or: await det.detect(rgbaBytes, width, height, options)
const tiled = await det.detectTiled(imageData, undefined, undefined, { tileSize: 640, overlap: 0.2, fullImage: true });
const tta = await det.detectTta(imageData, undefined, undefined, { hflip: true, scales: [0.83, 1.0, 1.17] });
det.free();                                            // releases the model (wasm memory, GPU buffers)
```

### `FoloDetector.load(bytes, options?) → Promise<FoloDetector>`
`bytes` is a `Uint8Array` or `ArrayBuffer` holding a `.folo` file: weights, a checkpoint (its EMA weights are used)
or an export (`folo export`, F32 or F16), exactly as `Predictor::open` reads them. The model is fused (Conv+BN) on
the CPU, then moved to the backend. Options (all optional; unknown fields are ignored):

| option | default | meaning |
|---|---|---|
| `backend` | `"auto"` | `"auto"`: WebGPU when it initialises and passes a test pass (a small gray image letterboxed to `imgsz`, which also compiles the kernels a real call needs), otherwise the CPU, with the reason in `info().fallback`. `"webgpu"`: WebGPU or a rejected promise. `"cpu"`: always the CPU |
| `modelId` | `""` | echoed as `model_id` in every result |
| `imgsz` | the file's `folo.imgsz` | letterbox size `S` (a multiple of 32): one image is padded `rect` to multiples of 32 inside `S × S`, like `folo predict` |
| `precision` | `"f32"` | `"f16"`: half precision (f16 weights and activations, f32 accumulation) on WebGPU when the adapter has `shader-f16`; on the CPU or without `shader-f16` the model runs in f32 and `info().precisionFallback` says why. `"f32"`: always f32 |

Rejects with an `Error` whose message says what is wrong (not a `.folo` file, unsupported config, no WebGPU, a
detached buffer, ...). `detect` likewise never throws: every problem rejects its promise.

### `detector.detect(image, width?, height?, options?) → Promise<FoloDetectResult>`
`image` is an `ImageData` (or any `{ data, width, height }`, in which case `width` / `height` may be `undefined`),
or raw RGBA bytes (`Uint8Array` / `Uint8ClampedArray`, `width × height × 4`, rows top to bottom) with `width` and
`height`. Alpha is ignored. The pixels are copied before the call returns, so the buffer may be reused at once.
Options are `/api/predict`'s, with its defaults: `conf` (0.25, `0..=1`), `iou` (0.7, `0..=1`), `maxDet` (300, `>= 1`).
The result is `/api/predict`'s response plus `backend`:

```jsonc
{ "model_id": "m_31aa", "image": { "width": 1280, "height": 720 },
  "boxes": [{ "class": 1, "name": "square", "score": 0.93, "x1": 10, "y1": 20, "x2": 200, "y2": 180 }],
  "timing": { "preprocess_ms": 2.8, "inference_ms": 10.8, "postprocess_ms": 0.1, "total_ms": 13.7 },
  "device": "gpu:0",            // "gpu:0" on WebGPU, "cpu" on the CPU, as the server reports them
  "backend": "webgpu" }         // "webgpu" | "cpu"
```

Boxes are in image pixels, highest score first, with the same letterbox, decode and NMS as the native predictor.
Timings use `performance.now()` (coarsened by browsers to 0.1 ms, or 5 µs on cross-origin isolated pages).
`preprocess_ms` is RGBA → RGB (SIMD128 `rgba_to_rgb`), letterbox and upload on WebGPU, or direct zero-intermediate-RGB
SIMD128 RGBA → planar-`f32` letterboxing on the CPU; on WebGPU `inference_ms` is the forward pass, device-side candidate
extraction (`Tensor::detect_candidates`, or dense decode fallback) and the awaited compact readback (the forward pass is
queued work, the readback is where the GPU time shows), and `postprocess_ms` is NMS and rescaling. On the CPU the split
matches the native predictor: on the candidate-first path (`predict_candidates`), forward, candidate selection, and
sparse box decode count in `inference_ms` while `finish_candidates` (NMS / top-k) and rescaling count in
`postprocess_ms`; on the dense fallback path (`out.detections`), decode counts in `postprocess_ms`.
Several `detect` calls may be in flight on one detector; each awaits its own readback. Repeated frames of the same
or smaller size reuse `FoloDetector`'s internal RGBA input copy buffer, RGB staging buffer (WebGPU), and letterbox
work buffers (CPU) with zero per-frame scratch reallocations. All numeric arguments (`width`, `height`, and every
numeric option) reject `NaN`, `±Infinity`, zero/negative dimensions, and out-of-range values with a rejected `Error`
instead of silently coercing or panicking.

### `detector.detectTiled(image, width?, height?, options?) → Promise<FoloDetectResult>`
Runs **SAHI (Slicing Aided Hyper Inference)** overlapping tiled detection for small objects in high-resolution
images, matching `folo_vision::Predictor::predict_tiled`: slices the image into uniform overlapping tiles, runs them
(in bounded batches of `tileBatch` on WebGPU with automatic OOM half-splitting, or direct zero-RGB-intermediate RGBA
crops on the CPU), optionally runs one full-image pass so large boundary-spanning objects stay intact, and merges all
detections deterministically in original-image pixel coordinates. When the image fits in a single tile, it degrades
directly to a single pass with zero slicing overhead.

Options extend `FoloDetectOptions` (`conf`, `iou`, `maxDet`):

| option | default | meaning |
|---|---|---|
| `tileSize` | `640` | convenience square tile size in pixels (`tileWidth` / `tileHeight` override it when set) |
| `tileWidth`, `tileHeight` | `tileSize` (or `640`) | target tile dimensions in pixels (`>= 1`, clamped to image size) |
| `overlap` | `0.2` | convenience symmetric overlap ratio in `[0, 1)` (`overlapX` / `overlapY` override it) |
| `overlapX`, `overlapY` | `overlap` (or `0.2`) | horizontal / vertical overlap ratios in `[0, 1)` |
| `fullImage` | `true` | also run one full-image pass alongside the tiles (also accepts `fullImagePass`) |
| `tileBatch` | `8` | maximum cropped tiles predicted per WebGPU batch (`>= 1`) |
| `mergeMetric` | `"ios"` | `"ios"` (Intersection-over-Smaller, suppresses truncated tile-seam fragments) or `"iou"` |
| `mergeThreshold` | `0.5` | cross-tile merge overlap threshold in `[0, 1]` (also accepts `mergeIou`) |
| `mergeFusion` | `"vote"` | cross-tile box refinement: `"vote"`, `"none"`, `"dfl"`, or `"sebf"` |
| `classAgnostic` | `false` | merge overlapping detections across different classes |

### `detector.detectTta(image, width?, height?, options?) → Promise<FoloDetectResult>`
Runs **Test-Time Augmentation (TTA)** matching `folo_vision::Predictor::predict_tta`: evaluates each resolved
stride-32 inference scale (and optionally a horizontally flipped mirror — batched together per scale on WebGPU, or
mirrored via SIMD128 4-pixel-per-instruction RGBA reversal on the CPU), maps mirrored boxes back to original pixel
coordinates, and merges all views deterministically. When TTA resolves to a single unflipped view, it runs a single
pass at that size without cross-view merging.

Options extend `FoloDetectOptions` (`conf`, `iou`, `maxDet`):

| option | default | meaning |
|---|---|---|
| `hflip` | `true` | also evaluate a horizontally flipped copy of the image at each scale |
| `scales` | `[0.83, 1.0, 1.17]` | positive finite multipliers applied to the detector's `imgsz` (each rounded multiple of 32 must lie in `32..=8192`; duplicates are deduplicated in first-seen order) |
| `mergeIou` | `iou` (`0.7`) | cross-view IoU threshold in `[0, 1]` |
| `mergeFusion` | `"vote"` | cross-view box refinement: `"vote"`, `"none"`, `"dfl"`, or `"sebf"` |
| `classAgnostic` | `false` | merge overlapping detections across different classes |

### `detector.info() → FoloDetectorInfo`
`{ backend, device, adapter, fallback, precision, precisionFallback, simd128, threads, arch, imgsz, classes, modelId }`:
`adapter` is the GPU adapter's name, or `null` when the browser does not disclose it (Chrome does not); `fallback` is
why `auto` chose the CPU (`null` otherwise); `precision` is what the model runs in (`"f32"` / `"f16"`) and
`precisionFallback` why a requested `"f16"` runs in f32 (`null` otherwise); `simd128` is whether the module was built with SIMD128 (always, from the script);
`threads` is 1 (see Limitations).

### Functions
- `hasWebGpu()`: whether `navigator.gpu` exists. A `true` does not guarantee a usable device; `auto` checks that.
- `version()`: the crate version.

`folo_wasm.d.ts` types all of the above (`FoloLoadOptions`, `FoloDetectOptions`, `FoloTileDetectOptions`,
`FoloTtaDetectOptions`, `FoloTileMergeMetric`, `FoloBoxFusion`, `FoloDetectResult`, `FoloBox`,
`FoloDetectorInfo`, `FoloImage`). wasm-bindgen's class declaration uses `Symbol.dispose`, so a strict type check of
it needs the `esnext.disposable` lib (or `skipLibCheck`, as Studio has). `crates/folo-wasm/demo/api-check.ts`
exercises the documented calls; CI runs `tsc -p crates/folo-wasm/demo/tsconfig.json` on it.

### Swapping it in for `/api/predict`
The result is a superset of the server's response, so code that renders `/api/predict` renders it unchanged. Load
the model's bytes once (from the page's own storage, an upload, or a URL the app serves), keep the `FoloDetector`
and reuse it per image: loading parses and fuses the model and, on WebGPU, compiles pipelines.

## How it works
- **Same engine.** folo-core, folo-nn, folo-vision and folo-gpu compile for `wasm32-unknown-unknown` with no forked
  code paths in the math. `folo-wasm` drives `folo_vision::Predictor`'s steps (`prepare` → `forward` → decode →
  `finish`), SAHI tiling (`folo_vision::tile`), and TTA (`folo_vision::tta`); `crates/folo-wasm/src/detector/` has
  no wasm-bindgen types and is tested natively (`cargo test -p folo-wasm`: identical boxes to `Predictor` on the
  CPU for single-pass, tiled, and TTA, and the GPU paths against the CPU).
- **WebGPU.** The browser creates adapters and devices asynchronously and never lets the page's thread block, so
  folo-gpu has async entry points used on every target: `GpuContext::new_async` / `shared_async`,
  `folo_gpu::init_async()` and `folo_gpu::download_async(ctx, tensor)` (a readback that resolves when WebGPU has
  mapped the staging buffer). The blocking ones (`GpuContext::new`, `sync`'s wait, `Tensor::to_vec` of a GPU
  tensor) return a "use the async variant" error in the browser instead of hanging. To avoid downloading the full
  `~2.8 MiB` decoded head on every frame, `Detector::detect` runs device-side candidate extraction
  (`Tensor::detect_candidates`, reusing a cached `[3, A]` device anchors tensor per canvas shape) and awaits a compact
  async readback (`download_candidate_records`: `~28 KiB` for the count plus up to `CANDIDATE_READBACK` = 1,024 7-word
  records, issuing a second readback only on overflow and falling back to dense `decoded_on_device()` when candidate
  extraction is unavailable). Shader compile errors are
  reported through the device's error sink at the next readback (natively they still come from an error scope at
  pipeline creation). The kernels are the native WGSL unchanged; Chrome's compiler (Tint) is stricter than Naga in
  places (it rejects `-3.4028235e38` as out of `f32` range, so `softmax` spells `-f32::MAX` by its bits).
- **CPU.** rayon runs on the calling thread on wasm (no threads without the `atomics` target feature), so the
  parallel paths need no separate code. `.cargo/config.toml` (and the build script, and CI) enable `+simd128` for
  wasm32, which every current browser supports:
  - `folo-wasm/src/simd/` provides explicit Wasm SIMD128 RGBA→RGB stripping (`rgba_to_rgb`, used before WebGPU
    uploads), 4-pixel-per-op RGBA horizontal mirroring (`flip_rgba_h`), and direct RGBA→planar-`f32` letterboxing
    (`letterbox_rgba_planar_f32_with_scratch`, zero intermediate RGB allocation, reusable `RgbaScratch` buffers,
    bit-exact with `folo_data::Letterbox::apply_planar_f32`), paired with candidate-first / sparse CPU decode.
  - `folo-core` adds explicit Wasm SIMD128 kernels for 4-lane non-FMA elementwise transcendentals and binary ops
    (`cpu/elementwise/wasm_simd.rs`), stride-1 depthwise convs with register-fused bias/SiLU/ReLU (`cpu/conv/depthwise.rs`),
    and 4-tile Winograd F(2×2, 3×3) (`cpu/conv/winograd_wasm.rs`) and F(4×4, 3×3) (`cpu/conv/winograd4/wasm.rs`)
    input/output transforms — all bit-exact with their scalar paths.
  - LLVM also vectorises the portable GEMM micro-kernel to SIMD128 (which originally took folo-n from 1758 ms scalar to
    491 ms at 640). A hand-written SIMD128 6×16 GEMM kernel (two 6×8 halves to fit 16 vector registers, multiply + add
    since base SIMD128 has no FMA) measured the same as the vectorised portable one (about 20 GFLOP/s on one thread,
    `cargo bench -p folo-core --bench gemm` built for `wasm32-wasip1` and run under Node's WASI), so it was not kept.
- **Timing.** `std::time::Instant` panics on `wasm32-unknown-unknown`; folo-vision and folo-wasm use `web-time`
  there.
- **Memory.** `ModelFile::from_bytes` parses in place (no mmap in the browser); the wasm heap holds the file once
  while loading and the fused F32 model after.

## Performance
folo-n at 640 (a 640 × 640 image, `rect` letterbox, conf 0.25, a small shapes model), median of 20 calls after 3
warm-up calls, wall time around `await detect(...)`, headless Chrome 156 on Windows 11 (RTX 4080 SUPER through
Chrome's D3D12 Dawn, i7-14700KF), `crates/folo-wasm/demo/test.mjs`:

| backend | median | min | split of one call | load |
|---|---|---|---|---|
| WebGPU | **14.1 ms** | 10.5 ms | preprocess 2.5, inference (forward, GPU decode, readback) 11.1, NMS 0.2 | 3.0–3.7 s |
| CPU (wasm SIMD128, 1 thread) | **496 ms** | 475 ms | preprocess 6, inference 477, decode + NMS 8 | 60–80 ms |
| CPU, scalar build (no SIMD128) | 1758 ms | 1682 ms | | |

- The GPU was shared with other jobs (33–90% busy) during these runs. Natively under the same load, folo-vision's
  end-to-end predict bench measured 12.7–26 ms median at batch 1 (8.69 ms on an idle GPU, docs/BENCHMARKS.md), so
  the browser is close to native; the remaining gap is per-dispatch cost through the WebGPU JS API and the
  `mapAsync` round trip.
- **Load time on WebGPU is pipeline compilation**: the test pass (at the letterbox size, so with the same kernel
  specialisations as a real call) compiles them through Chrome's shader compiler. The first call at 640 after loading
  took 26 ms. The CPU loads in tens of milliseconds.
- The CPU backend is single-threaded (see Limitations): about 13 GFLOP/s effective on folo-n's 6.5 GFLOPs.

## Demo and browser test
`crates/folo-wasm/demo/index.html` is a standalone page (vanilla JS): pick a `.folo` model and an image, choose the
backend, detect, draw the boxes, and benchmark 20 runs. Serve the repository root (any static server; `.wasm` should
be `application/wasm`) after building, and open `/crates/folo-wasm/demo/`.

`crates/folo-wasm/demo/test.mjs` is the automated check: it serves the repository, opens the demo in headless
Chrome through Playwright (`playwright-core` from `web/node_modules`, so run `npm ci` in `web/` first), runs the model
on each backend, prints the latency and compares the boxes with the native predictor's:

```sh
sh scripts/build-wasm.sh
folo predict --model shapes-n.folo img.jpg --imgsz 640 --device cpu --json > native.json
node crates/folo-wasm/demo/test.mjs --model shapes-n.folo --image img.jpg --imgsz 640 --expect native.json
```

Options: `--backends webgpu,cpu,auto`, `--runs 20`, `--conf 0.25`, `--imgsz <n>`, `--chrome <path>`,
`--chrome-args="<flags>"` (e.g. `--disable-gpu` to exercise `auto`'s CPU fallback), `--headed`, `--screenshots <dir>`.
Chrome runs with `--enable-unsafe-webgpu`, which headless Chrome needs for WebGPU. The script exits non-zero when a
backend fails, runs on another backend than requested, disagrees with the native boxes (same class, IoU ≥ 0.98,
score within 0.01), or does not reject bad inputs (detached buffers, wrong or overflowing sizes, bad options) with an
`Error`.

## Limitations and follow-ups
- **One CPU thread.** Threads need `wasm-bindgen-rayon`, a nightly toolchain with `-Z build-std` and
  `+atomics,+bulk-memory`, and a cross-origin isolated page (COOP / COEP headers). FOLO stays on stable Rust, so the
  CPU backend is single-threaded; WebGPU is the fast path.
- **f16 on WebGPU is opt-in** (`precision: "f16"`) and needs the adapter's `shader-f16` (Chrome exposes it on most
  desktop GPUs; WebGPU on D3D12 through Dawn may not). It is not benchmarked in the browser yet; natively it is a
  modest gain at large batches (docs/BENCHMARKS.md) and none at batch 1, which is what the page runs.
- No video / webcam streaming helper yet (call `detect` per frame); no batching (`predictBatch`) in the JS API.
- Training is not exposed (folo-vision's trainer compiles for wasm but needs threads and files).
