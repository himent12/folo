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
`--dev` builds unoptimised, `--out DIR` writes elsewhere; `wasm-opt` is applied when it is on `PATH`. CI's `wasm` job
runs the same script on every PR. Vite serves `web/public/` as is, so Studio imports it as
`/folo-wasm/folo_wasm.js`.

## JS API

```js
import init, { FoloDetector, hasWebGpu, version } from "/folo-wasm/folo_wasm.js";

await init();                                          // fetches and instantiates folo_wasm_bg.wasm (once)
const bytes = new Uint8Array(await (await fetch("/models/shapes-n.folo")).arrayBuffer());
const det = await FoloDetector.load(bytes, { backend: "auto", modelId: "m_31aa" });
det.backend;                                           // "webgpu" | "cpu"
det.info();                                            // see below

const res = await det.detect(imageData, undefined, undefined, { conf: 0.25, iou: 0.7, maxDet: 300 });
// or: await det.detect(rgbaBytes, width, height, options)
det.free();                                            // releases the model (wasm memory, GPU buffers)
```

### `FoloDetector.load(bytes, options?) → Promise<FoloDetector>`
`bytes` is a `Uint8Array` or `ArrayBuffer` holding a `.folo` file: weights, a checkpoint (its EMA weights are used)
or an export (`folo export`, F32 or F16), exactly as `Predictor::open` reads them. The model is fused (Conv+BN) on
the CPU, then moved to the backend. Options (all optional; unknown fields are ignored):

| option | default | meaning |
|---|---|---|
| `backend` | `"auto"` | `"auto"`: WebGPU when it initialises and passes one small test pass (64×64, which also compiles the core kernels), otherwise the CPU, with the reason in `info().fallback`. `"webgpu"`: WebGPU or a rejected promise. `"cpu"`: always the CPU |
| `modelId` | `""` | echoed as `model_id` in every result |
| `imgsz` | the file's `folo.imgsz` | letterbox size `S` (a multiple of 32): one image is padded `rect` to multiples of 32 inside `S × S`, like `folo predict` |

Rejects with an `Error` whose message says what is wrong (not a `.folo` file, unsupported config, no WebGPU, ...).

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
`preprocess_ms` is RGBA → RGB, letterbox and upload; on WebGPU `inference_ms` is the forward pass, the on-GPU decode
and the awaited readback (the forward pass is queued work, the readback is where the GPU time shows), and
`postprocess_ms` is NMS and rescaling. On the CPU the split is the server's: decode counts as postprocessing.
Several `detect` calls may be in flight on one detector; each awaits its own readback.

### `detector.info() → FoloDetectorInfo`
`{ backend, device, adapter, fallback, simd128, threads, arch, imgsz, classes, modelId }`: `adapter` is the GPU
adapter's name, or `null` when the browser does not disclose it (Chrome does not); `fallback` is why `auto` chose
the CPU (`null` otherwise); `simd128` is whether the module was built with SIMD128 (always, from the script);
`threads` is 1 (see Limitations).

### Functions
- `hasWebGpu()`: whether `navigator.gpu` exists. A `true` does not guarantee a usable device; `auto` checks that.
- `version()`: the crate version.

`folo_wasm.d.ts` types all of the above (`FoloLoadOptions`, `FoloDetectOptions`, `FoloDetectResult`, `FoloBox`,
`FoloDetectorInfo`, `FoloImage`).

### Swapping it in for `/api/predict`
The result is a superset of the server's response, so code that renders `/api/predict` renders it unchanged. Load
the model's bytes once (from the page's own storage, an upload, or a URL the app serves), keep the `FoloDetector`
and reuse it per image: loading parses and fuses the model and, on WebGPU, compiles pipelines.

## How it works
- **Same engine.** folo-core, folo-nn, folo-vision and folo-gpu compile for `wasm32-unknown-unknown` with no forked
  code paths in the math. `folo-wasm` drives `folo_vision::Predictor`'s steps (`prepare` → `forward` → decode →
  `finish`); `crates/folo-wasm/src/detector.rs` has no wasm-bindgen types and is tested natively
  (`cargo test -p folo-wasm`: identical boxes to `Predictor` on the CPU, the GPU path against the CPU).
- **WebGPU.** The browser creates adapters and devices asynchronously and never lets the page's thread block, so
  folo-gpu has async entry points used on every target: `GpuContext::new_async` / `shared_async`,
  `folo_gpu::init_async()` and `folo_gpu::download_async(ctx, tensor)` (a readback that resolves when WebGPU has
  mapped the staging buffer). The blocking ones (`GpuContext::new`, `sync`'s wait, `Tensor::to_vec` of a GPU
  tensor) return a "use the async variant" error in the browser instead of hanging. Shader compile errors are
  reported through the device's error sink at the next readback (natively they still come from an error scope at
  pipeline creation). The kernels are the native WGSL unchanged; Chrome's compiler (Tint) is stricter than Naga in
  places (it rejects `-3.4028235e38` as out of `f32` range, so `softmax` spells `-f32::MAX` by its bits).
- **CPU.** rayon runs on the calling thread on wasm (no threads without the `atomics` target feature), so the
  parallel paths need no separate code. `.cargo/config.toml` (and the build script, and CI) enable `+simd128` for
  wasm32, which every current browser supports: LLVM vectorises the GEMM micro-kernel and the elementwise loops to
  it, which makes folo-n 3.6x faster than a scalar build (491 vs 1758 ms at 640). A hand-written SIMD128 6×16 GEMM
  kernel (two 6×8 halves to fit 16 vector registers, multiply + add since base SIMD128 has no FMA) measured the same
  as the vectorised portable one (about 20 GFLOP/s on one thread, `cargo bench -p folo-core --bench gemm` built for
  `wasm32-wasip1` and run under Node's WASI), so it was not kept.
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
- **Load time on WebGPU is pipeline compilation**: the test pass compiles about 50 pipelines through Chrome's shader
  compiler. The first call at 640 after loading took 26 ms, so the 64×64 test pass covers the kernels a real input
  needs. The CPU loads in tens of milliseconds.
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

Options: `--backends webgpu,cpu`, `--runs 20`, `--conf 0.25`, `--chrome <path>`, `--headed`, `--screenshots <dir>`.
Chrome runs with `--enable-unsafe-webgpu`, which headless Chrome needs for WebGPU. The script exits non-zero when a
backend fails, runs on another backend than requested, or disagrees with the native boxes (same class, IoU ≥ 0.98,
score within 0.01).

## Limitations and follow-ups
- **One CPU thread.** Threads need `wasm-bindgen-rayon`, a nightly toolchain with `-Z build-std` and
  `+atomics,+bulk-memory`, and a cross-origin isolated page (COOP / COEP headers). FOLO stays on stable Rust, so the
  CPU backend is single-threaded; WebGPU is the fast path.
- **f32 on WebGPU.** The model computes in f32, as natively; `shader-f16` is requested when the adapter has it, but
  no inference path uses f16 compute yet.
- No video / webcam streaming helper yet (call `detect` per frame); no batching (`predictBatch`) in the JS API.
- Training is not exposed (folo-vision's trainer compiles for wasm but needs threads and files).
