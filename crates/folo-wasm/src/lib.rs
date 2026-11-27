//! FOLO detectors and ByteTrack multi-object tracking in the browser:
//! [wasm-bindgen](https://wasm-bindgen.github.io/wasm-bindgen/) bindings over `folo-vision`'s predictor (`folo-*`
//! and `folo2-*`) and `ByteTracker`, running on WebGPU (`folo-gpu`) or the wasm CPU backend (`folo-core`, SIMD128).
//!
//! The JS API (docs/WASM.md):
//!
//! ```js
//! import init, { FoloDetector, FoloTracker, hasWebGpu } from "./folo-wasm/folo_wasm.js";
//! await init();
//! const det = await FoloDetector.load(bytes, { backend: "auto" }); // "auto" | "webgpu" | "cpu"
//! const res = await det.detect(imageData, undefined, undefined, { conf: 0.25, iou: 0.7, maxDet: 300 });
//! const tiled = await det.detectTiled(imageData, undefined, undefined, { tileSize: 640, overlap: 0.2 });
//! const tta = await det.detectTta(imageData, undefined, undefined, { hflip: true, scales: [0.83, 1.0, 1.17] });
//! const tracker = new FoloTracker({ highThresh: 0.5, lowThresh: 0.1, trackBuffer: 30 });
//! const tracks = tracker.update(res.boxes);
//! // res: docs/API.md's /api/predict response, plus `backend`
//! ```
//!
//! The logic lives in [`detector`] and [`tracker`], which have no `wasm-bindgen` types and are tested natively.

pub mod detector;
mod js_util;
mod simd;
pub mod tracker;

use std::rc::Rc;

use js_sys::{Promise, Uint8Array};
use wasm_bindgen::prelude::*;

pub use detector::{
    Backend, BackendChoice, DetectOptions, DetectResult, Detector, DetectorInfo, LoadOptions,
    MAX_BROWSER_TILES, TileDetectOptions, TtaDetectOptions,
};
use js_util::{
    byte_view, bytes_of, js_err, object_dim, parse_dim, parse_options, parse_track_boxes,
    parse_track_options, to_js,
};
pub use tracker::{
    TrackInputBox, TrackStateLabel, TrackedBox, Tracker, TrackerConfig, TrackerInfo, TrackerOptions,
};

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
/** Options of `FoloDetector.load`. */
export interface FoloLoadOptions {
  /** "auto" (WebGPU when it works, else the CPU; the default), "webgpu" or "cpu". */
  backend?: "auto" | "webgpu" | "cpu";
  /** Echoed as `model_id` in results (docs/API.md). */
  modelId?: string;
  /** Letterbox size (a multiple of 32); defaults to the file's training size. */
  imgsz?: number;
  /** "f32" (the default) or "f16": half precision on WebGPU with shader-f16, f32 elsewhere (see info()). */
  precision?: "f32" | "f16";
}

/** Options of `FoloDetector.detect`, with /api/predict's defaults. */
export interface FoloDetectOptions {
  conf?: number;
  iou?: number;
  maxDet?: number;
}

/** Overlap metric used when merging cross-tile detections. */
export type FoloTileMergeMetric = "ios" | "iou";

/** Box fusion rule used when merging multi-view (tiled or TTA) detections. */
export type FoloBoxFusion = "none" | "vote" | "dfl" | "sebf";

/** Options of `FoloDetector.detectTiled` (SAHI overlapping tiled inference). */
export interface FoloTileDetectOptions extends FoloDetectOptions {
  /** Square tile size in pixels (overridden by `tileWidth` / `tileHeight`; default 640). */
  tileSize?: number;
  /** Target tile width in pixels (defaults to `tileSize` or 640). */
  tileWidth?: number;
  /** Target tile height in pixels (defaults to `tileSize` or 640). */
  tileHeight?: number;
  /** Symmetric overlap ratio in [0, 1) (overridden by `overlapX` / `overlapY`; default 0.2). */
  overlap?: number;
  /** Horizontal overlap ratio in [0, 1). */
  overlapX?: number;
  /** Vertical overlap ratio in [0, 1). */
  overlapY?: number;
  /** Also run a full-image pass alongside the tiles (default true). */
  fullImage?: boolean;
  /** Alias for `fullImage`. */
  fullImagePass?: boolean;
  /** Maximum tiles predicted per WebGPU batch (default 8). */
  tileBatch?: number;
  /** Cross-tile merge overlap metric ("ios" default, or "iou"). */
  mergeMetric?: FoloTileMergeMetric;
  /** Cross-tile merge threshold in [0, 1] (default 0.5). */
  mergeThreshold?: number;
  /** Alias for `mergeThreshold`. */
  mergeIou?: number;
  /** Cross-tile box fusion rule ("vote" default, "none", "dfl", or "sebf"). */
  mergeFusion?: FoloBoxFusion;
  /** Merge overlapping detections across different classes (default false). */
  classAgnostic?: boolean;
}

/** Options of `FoloDetector.detectTta` (Test-Time Augmentation). */
export interface FoloTtaDetectOptions extends FoloDetectOptions {
  /** Also evaluate a horizontally flipped copy at each scale (default true). */
  hflip?: boolean;
  /** Multi-scale multipliers applied to `imgsz` (default [0.83, 1.0, 1.17]). */
  scales?: number[];
  /** Cross-view merge IoU threshold in [0, 1] (defaults to `iou`). */
  mergeIou?: number;
  /** Cross-view box fusion rule ("vote" default, "none", "dfl", or "sebf"). */
  mergeFusion?: FoloBoxFusion;
  /** Merge overlapping detections across different classes (default false). */
  classAgnostic?: boolean;
}

/** docs/API.md's ScoredBox. */
export interface FoloBox {
  class: number;
  name: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** docs/API.md's /api/predict response, plus the backend that ran it. */
export interface FoloDetectResult {
  model_id: string;
  image: { width: number; height: number };
  boxes: FoloBox[];
  timing: { preprocess_ms: number; inference_ms: number; postprocess_ms: number; total_ms: number };
  device: string;
  backend: "webgpu" | "cpu";
}

/** `FoloDetector.info()`. */
export interface FoloDetectorInfo {
  backend: "webgpu" | "cpu";
  device: string;
  adapter: string | null;
  fallback: string | null;
  /** The precision the model runs in. */
  precision: "f32" | "f16";
  /** Why a requested "f16" runs in f32 (no WebGPU, or no shader-f16), else null. */
  precisionFallback: string | null;
  simd128: boolean;
  threads: number;
  /** Model family: "folo" (v1) or "folo2" (v2-Iso). */
  family: "folo" | "folo2";
  /** Model architecture identifier, e.g. "folo-n" or "folo2-n". */
  arch: string;
  /** Active evaluation head mode: "o2m" (NMS) or "o2o" (NMS-free top-k). */
  headMode: "o2m" | "o2o";
  /** Whether the loaded model includes a one-to-one ("o2o") head branch. */
  hasO2o: boolean;
  /** Whether the loaded FOLO v2 model includes an `iou_head` localization-quality branch. */
  hasIouHead: boolean;
  imgsz: number;
  classes: string[];
  modelId: string;
}

/** What `detect` accepts as an image: ImageData (or any { data, width, height }) or raw RGBA bytes. */
export type FoloImage = ImageData | { data: Uint8Array | Uint8ClampedArray; width: number; height: number } | Uint8Array | Uint8ClampedArray;

/** Options of `new FoloTracker(options?)` (ByteTrack multi-object tracking). */
export interface FoloTrackOptions {
  /** Minimum score for stage-1 high-confidence association (default 0.5, in [0, 1]). */
  highThresh?: number;
  /** Minimum score for stage-2 low-confidence occlusion recovery (default min(0.1, highThresh), in [0, highThresh]). */
  lowThresh?: number;
  /** Minimum score to spawn a new track (default max(0.6, highThresh), in [highThresh, 1]). */
  newTrackThresh?: number;
  /** Minimum IoU for stage-1 association (default 0.2, in [0, 1]). */
  matchIou?: number;
  /** Minimum IoU for stage-2 occlusion recovery (default 0.5, in [0, 1]). */
  secondMatchIou?: number;
  /** Minimum IoU for tentative tracks (default 0.3, in [0, 1]). */
  tentativeMatchIou?: number;
  /** Consecutive missed frames a lost track coasts before retiring (default 30). */
  trackBuffer?: number;
  /** Consecutive matched detections required before confirming a new track (default 1, >= 1). */
  minHits?: number;
  /** Require identical class ID when associating detections with tracks (default true). */
  classAware?: boolean;
  /** Optional class names indexed by `class` (used to populate `FoloTrackedBox.name`). */
  classes?: string[];
}

/** Resolved validated configuration returned by `FoloTracker.config()`. */
export interface FoloTrackConfig {
  highThresh: number;
  lowThresh: number;
  newTrackThresh: number;
  matchIou: number;
  secondMatchIou: number;
  tentativeMatchIou: number;
  trackBuffer: number;
  minHits: number;
  classAware: boolean;
}

/** Input box accepted by `FoloTracker.update(boxes)` (`FoloBox` is directly assignable). */
export interface FoloTrackInputBox {
  class: number;
  name?: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Lifecycle state reported on emitted tracked boxes. */
export type FoloTrackState = "tracked";

/** One active tracked object returned by `FoloTracker.update(boxes)` (extends `FoloBox`). */
export interface FoloTrackedBox extends FoloBox {
  /** Persistent monotonically increasing track ID (starts at 1). */
  track_id: number;
  /** Total frames elapsed since this track was created (1 on creation frame). */
  age: number;
  /** Total matched detection updates accumulated by this track. */
  hits: number;
  /** Lifecycle state of emitted active tracks ("tracked"). */
  state: FoloTrackState;
}

/** `FoloTracker.info()`. */
export interface FoloTrackerInfo {
  /** Number of frames processed since creation or the last `reset()`. */
  frameId: number;
  /** Number of currently active confirmed tracks. */
  activeTracks: number;
  /** Number of currently coasting lost tracks in the recovery buffer. */
  lostTracks: number;
  /** Effective validated tracker configuration. */
  config: FoloTrackConfig;
}

// `load`, `detect`, `detectTiled`, and `detectTta` are declared here (merged into the class) because
// wasm-bindgen cannot emit optional parameters with custom types.
export interface FoloDetector {
  /** Detect objects; `width` / `height` may be omitted for ImageData. Never throws: errors reject. */
  detect(image: FoloImage, width?: number, height?: number, options?: FoloDetectOptions): Promise<FoloDetectResult>;
  /** Detect objects with SAHI overlapping tiled inference. Never throws: errors reject. */
  detectTiled(image: FoloImage, width?: number, height?: number, options?: FoloTileDetectOptions): Promise<FoloDetectResult>;
  /** Detect objects with multi-scale / horizontal-flip Test-Time Augmentation. Never throws: errors reject. */
  detectTta(image: FoloImage, width?: number, height?: number, options?: FoloTtaDetectOptions): Promise<FoloDetectResult>;
}
export declare namespace FoloDetector {
  /** Load a `.folo` file (`folo-*` or `folo2-*`) onto WebGPU or the CPU. */
  function load(bytes: Uint8Array | ArrayBuffer, options?: FoloLoadOptions): Promise<FoloDetector>;
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(msg: &str);
}

/// Module start: route Rust panics to `console.error` (instead of a bare "unreachable").
#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| console_error(&format!("folo-wasm panicked: {info}"))));
}

/// Whether the page exposes WebGPU (`navigator.gpu`). `auto` loading still checks that a device works.
#[wasm_bindgen(js_name = hasWebGpu)]
pub fn has_webgpu() -> bool {
    let nav = js_sys::Reflect::get(&js_sys::global(), &"navigator".into()).unwrap_or(JsValue::UNDEFINED);
    !nav.is_undefined()
        && js_sys::Reflect::get(&nav, &"gpu".into()).is_ok_and(|g| !g.is_undefined() && !g.is_null())
}

/// The crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// RAII wrapper returning the pooled RGBA input buffer to [`Detector`] when an async call completes.
struct PooledBytes {
    inner: Rc<Detector>,
    buf: Vec<u8>,
}

impl Drop for PooledBytes {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        self.inner.return_rgba_buffer(buf);
    }
}

/// A FOLO detector (`folo-*` or `folo2-*`) loaded from `.folo` bytes. Create with `FoloDetector.load`; call
/// `free()` to release it.
#[wasm_bindgen]
pub struct FoloDetector {
    inner: Rc<Detector>,
}

impl FoloDetector {
    /// Validate JS arguments and copy RGBA pixels into a pooled WASM buffer synchronously before any `.await`.
    fn prepare_rgba_call<O: serde::de::DeserializeOwned + Default>(
        &self,
        image: JsValue,
        width: Option<JsValue>,
        height: Option<JsValue>,
        options_: Option<JsValue>,
        op: &str,
    ) -> Result<(PooledBytes, u32, u32, O), JsValue> {
        let w_arg = parse_dim(width.as_ref(), op, "width")?;
        let h_arg = parse_dim(height.as_ref(), op, "height")?;
        let (data, w, h) =
            if image.is_instance_of::<Uint8Array>() || image.is_instance_of::<js_sys::Uint8ClampedArray>() {
                (image, w_arg, h_arg)
            } else {
                let data = js_sys::Reflect::get(&image, &"data".into())
                    .map_err(|_| js_err(format!("{op}: expected ImageData or RGBA bytes")))?;
                let w = match w_arg {
                    Some(v) => Some(v),
                    None => object_dim(&image, op, "width")?,
                };
                let h = match h_arg {
                    Some(v) => Some(v),
                    None => object_dim(&image, op, "height")?,
                };
                (data, w, h)
            };
        let (w, h) = w.zip(h).ok_or_else(|| js_err(format!("{op}: width and height are required")))?;
        let expected = (w as usize)
            .checked_mul(h as usize)
            .and_then(|n| n.checked_mul(4))
            .filter(|&b| b <= isize::MAX as usize)
            .ok_or_else(|| js_err(format!("{op}: image is too large ({w}x{h})")))?;
        let (view, byte_len) = byte_view(&data, op)?;
        if byte_len != expected {
            return Err(js_err(format!(
                "{op}: expected {expected} RGBA bytes for a {w}x{h} image, got {byte_len}"
            )));
        }
        let opts: O = parse_options(options_.as_ref())?;
        let mut buf = self.inner.take_rgba_buffer(expected);
        if let Some(view) = view {
            if view.length() as usize != expected {
                self.inner.return_rgba_buffer(buf);
                return Err(js_err(format!(
                    "{op}: expected {expected} RGBA bytes for a {w}x{h} image, got {}",
                    view.length()
                )));
            }
            view.copy_to(&mut buf);
        }
        Ok((PooledBytes { inner: self.inner.clone(), buf }, w, h, opts))
    }
}

#[wasm_bindgen]
impl FoloDetector {
    /// Load a `.folo` file (`Uint8Array` or `ArrayBuffer`, `folo-*` or `folo2-*`) onto WebGPU or the CPU
    /// (`options.backend`). (TypeScript: declared in the custom section.)
    #[wasm_bindgen(skip_typescript)]
    pub async fn load(bytes: JsValue, options_: Option<JsValue>) -> Result<FoloDetector, JsValue> {
        let bytes = bytes_of(&bytes, "load")?;
        let opts: LoadOptions = parse_options(options_.as_ref())?;
        let det = Detector::load(bytes, opts).await.map_err(js_err)?;
        Ok(FoloDetector { inner: Rc::new(det) })
    }

    /// `"webgpu"` or `"cpu"`.
    #[wasm_bindgen(getter)]
    pub fn backend(&self) -> String {
        self.inner.backend().as_str().to_string()
    }

    /// Backend, adapter, fallback reason, model and class details.
    #[wasm_bindgen(unchecked_return_type = "FoloDetectorInfo")]
    pub fn info(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.info())
    }

    /// Detect objects. `image` is an `ImageData` (or `{ data, width, height }`; `width` / `height` may then be
    /// omitted) or raw RGBA bytes with `width` and `height`. Resolves to the /api/predict response shape.
    /// (TypeScript: declared in the custom section.)
    #[wasm_bindgen(skip_typescript)]
    pub fn detect(
        &self,
        image: JsValue,
        width: Option<JsValue>,
        height: Option<JsValue>,
        options_: Option<JsValue>,
    ) -> Promise {
        let prepared = self.prepare_rgba_call::<DetectOptions>(image, width, height, options_, "detect");
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let (rgba, w, h, opts) = prepared?;
            let res = inner.detect(&rgba.buf, w, h, opts).await.map_err(js_err)?;
            to_js(&res)
        })
    }

    /// Detect objects using SAHI overlapping tiled inference. Resolves to the /api/predict response shape.
    /// (TypeScript: declared in the custom section.)
    #[wasm_bindgen(js_name = detectTiled, skip_typescript)]
    pub fn detect_tiled(
        &self,
        image: JsValue,
        width: Option<JsValue>,
        height: Option<JsValue>,
        options_: Option<JsValue>,
    ) -> Promise {
        let prepared =
            self.prepare_rgba_call::<TileDetectOptions>(image, width, height, options_, "detectTiled");
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let (rgba, w, h, opts) = prepared?;
            let res = inner.detect_tiled(&rgba.buf, w, h, opts).await.map_err(js_err)?;
            to_js(&res)
        })
    }

    /// Detect objects using multi-scale and/or horizontal-flip Test-Time Augmentation. Resolves to the
    /// /api/predict response shape. (TypeScript: declared in the custom section.)
    #[wasm_bindgen(js_name = detectTta, skip_typescript)]
    pub fn detect_tta(
        &self,
        image: JsValue,
        width: Option<JsValue>,
        height: Option<JsValue>,
        options_: Option<JsValue>,
    ) -> Promise {
        let prepared =
            self.prepare_rgba_call::<TtaDetectOptions>(image, width, height, options_, "detectTta");
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let (rgba, w, h, opts) = prepared?;
            let res = inner.detect_tta(&rgba.buf, w, h, opts).await.map_err(js_err)?;
            to_js(&res)
        })
    }
}

/// Stateful ByteTrack multi-object tracker for frame-by-frame browser detection streams.
/// Create with `new FoloTracker(options?)`; call `free()` to release it.
#[wasm_bindgen]
pub struct FoloTracker {
    inner: Tracker,
}

#[wasm_bindgen]
impl FoloTracker {
    /// Create a new ByteTrack multi-object tracker with optional configuration (`FoloTrackOptions`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(unchecked_optional_param_type = "FoloTrackOptions")] options: Option<JsValue>,
    ) -> Result<FoloTracker, JsValue> {
        let opts = parse_track_options(options.as_ref())?;
        let inner = Tracker::new(opts).map_err(js_err)?;
        Ok(FoloTracker { inner })
    }

    /// Advance the tracker by one frame with an array of detection boxes (`FoloBox[]` or `FoloTrackInputBox[]`)
    /// and return all active confirmed tracked boxes sorted by ascending `track_id`.
    #[wasm_bindgen(unchecked_return_type = "FoloTrackedBox[]")]
    pub fn update(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "FoloTrackInputBox[]")] boxes: JsValue,
    ) -> Result<JsValue, JsValue> {
        let parsed = parse_track_boxes(&boxes)?;
        let tracks = self.inner.update(&parsed);
        to_js(&tracks)
    }

    /// Clear all active, tentative, and lost tracks and reset frame and track ID counters.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Return the resolved validated tracker configuration.
    #[wasm_bindgen(unchecked_return_type = "FoloTrackConfig")]
    pub fn config(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.config())
    }

    /// Return current frame/track counters alongside the resolved configuration.
    #[wasm_bindgen(unchecked_return_type = "FoloTrackerInfo")]
    pub fn info(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.info())
    }
}
