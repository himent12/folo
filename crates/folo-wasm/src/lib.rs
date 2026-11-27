//! FOLO detectors in the browser: [wasm-bindgen](https://wasm-bindgen.github.io/wasm-bindgen/) bindings over
//! folo-vision's predictor, running on WebGPU (folo-gpu) or the wasm CPU backend (folo-core, SIMD128).
//!
//! The JS API (docs/WASM.md):
//!
//! ```js
//! import init, { FoloDetector, hasWebGpu } from "./folo-wasm/folo_wasm.js";
//! await init();
//! const det = await FoloDetector.load(bytes, { backend: "auto" }); // "auto" | "webgpu" | "cpu"
//! const res = await det.detect(imageData, undefined, undefined, { conf: 0.25, iou: 0.7, maxDet: 300 });
//! const tiled = await det.detectTiled(imageData, undefined, undefined, { tileSize: 640, overlap: 0.2 });
//! const tta = await det.detectTta(imageData, undefined, undefined, { hflip: true, scales: [0.83, 1.0, 1.17] });
//! // res: docs/API.md's /api/predict response, plus `backend`
//! ```
//!
//! The logic lives in [`detector`], which has no wasm-bindgen types and is tested natively.

pub mod detector;
mod simd;

use std::rc::Rc;

use js_sys::{Promise, Uint8Array};
use wasm_bindgen::prelude::*;

pub use detector::{
    Backend, BackendChoice, DetectOptions, DetectResult, Detector, DetectorInfo, LoadOptions,
    TileDetectOptions, TtaDetectOptions,
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
  arch: string;
  imgsz: number;
  classes: string[];
  modelId: string;
}

/** What `detect` accepts as an image: ImageData (or any { data, width, height }) or raw RGBA bytes. */
export type FoloImage = ImageData | { data: Uint8Array | Uint8ClampedArray; width: number; height: number } | Uint8Array | Uint8ClampedArray;

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
  /** Load a `.folo` file onto WebGPU or the CPU. */
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

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&msg.to_string()).into()
}

/// Recursively reject `NaN` / `±Infinity` inside JS options before `JSON.stringify` would silently coerce them
/// into `null` (which serde would otherwise turn into `None` for optional numeric fields).
fn reject_non_finite_numbers(v: &JsValue, path: &str, depth: usize) -> Result<(), JsValue> {
    if depth > 8 {
        return Err(js_err("invalid options: object nesting too deep"));
    }
    if v.is_undefined() || v.is_null() {
        return Ok(());
    }
    if let Some(f) = v.as_f64() {
        if !f.is_finite() {
            return Err(js_err(format!("invalid options: {path} must be finite, got {f}")));
        }
        return Ok(());
    }
    if js_sys::Array::is_array(v) {
        let arr: &js_sys::Array = v.unchecked_ref();
        for i in 0..arr.length() {
            let item = js_sys::Reflect::get_u32(v, i)
                .map_err(|_| js_err(format!("invalid options: cannot read {path}[{i}]")))?;
            reject_non_finite_numbers(&item, &format!("{path}[{i}]"), depth + 1)?;
        }
        return Ok(());
    }
    if v.is_object() {
        let keys = js_sys::Object::keys(v.unchecked_ref::<js_sys::Object>());
        for i in 0..keys.length() {
            let key = keys.get(i);
            let name = key.as_string().unwrap_or_default();
            let val = js_sys::Reflect::get(v, &key)
                .map_err(|_| js_err(format!("invalid options: cannot read property `{name}`")))?;
            let child = if path.is_empty() { name } else { format!("{path}.{name}") };
            reject_non_finite_numbers(&val, &child, depth + 1)?;
        }
    }
    Ok(())
}

/// Parse a JS options object through JSON (undefined / null give the defaults).
fn options<T: serde::de::DeserializeOwned + Default>(v: Option<&JsValue>) -> Result<T, JsValue> {
    let Some(v) = v.filter(|v| !v.is_undefined() && !v.is_null()) else { return Ok(T::default()) };
    if !v.is_object() || js_sys::Array::is_array(v) {
        return Err(js_err("options must be an object"));
    }
    reject_non_finite_numbers(v, "", 0)?;
    let s: String = js_sys::JSON::stringify(v)
        .map_err(|_| js_err("options are not JSON-serialisable"))?
        .as_string()
        .ok_or_else(
            // `JSON.stringify` gives `undefined` for a function, a symbol or a `toJSON` returning undefined.
            || js_err("options must be an object"),
        )?;
    serde_json::from_str(&s).map_err(|e| js_err(format!("invalid options: {e}")))
}

fn to_js(v: &impl serde::Serialize) -> Result<JsValue, JsValue> {
    let s = serde_json::to_string(v).map_err(js_err)?;
    js_sys::JSON::parse(&s)
}

fn num(v: &JsValue, key: &str) -> Option<f64> {
    js_sys::Reflect::get(v, &key.into()).ok()?.as_f64()
}

/// Parse an optional JS pixel dimension, strictly rejecting `NaN`, `Infinity`, negatives, zero, fractions, or
/// non-numeric values rather than letting WebIDL `ToUint32` silently coerce them.
fn parse_dim(v: Option<&JsValue>, op: &str, name: &str) -> Result<Option<u32>, JsValue> {
    let Some(v) = v.filter(|v| !v.is_undefined() && !v.is_null()) else { return Ok(None) };
    let Some(f) = v.as_f64() else {
        return Err(js_err(format!("{op}: {name} must be a positive integer")));
    };
    if !f.is_finite() || f < 1.0 || f > u32::MAX as f64 || f.fract() != 0.0 {
        return Err(js_err(format!("{op}: {name} must be a positive integer, got {f}")));
    }
    Ok(Some(f as u32))
}

/// Read an image property (`width` or `height`) with the same strict validation as [`parse_dim`].
fn object_dim(v: &JsValue, op: &str, key: &str) -> Result<Option<u32>, JsValue> {
    let prop = js_sys::Reflect::get(v, &key.into())
        .map_err(|_| js_err(format!("{op}: expected ImageData or RGBA bytes")))?;
    parse_dim(Some(&prop), op, key)
}

/// Inspect a JS typed array or `ArrayBuffer`, validating that it is not detached and returning a `Uint8Array`
/// view along with its byte length.
fn byte_view(v: &JsValue, what: &str) -> Result<(Option<Uint8Array>, usize), JsValue> {
    let bad = || js_err(format!("{what}: expected a Uint8Array, Uint8ClampedArray or ArrayBuffer"));
    let (buf, off, len) = if v.is_instance_of::<js_sys::ArrayBuffer>() {
        (v.clone(), 0.0, num(v, "byteLength").ok_or_else(bad)?)
    } else {
        let buf = js_sys::Reflect::get(v, &"buffer".into()).map_err(|_| bad())?;
        if !buf.is_instance_of::<js_sys::ArrayBuffer>() {
            return Err(bad());
        }
        (buf, num(v, "byteOffset").ok_or_else(bad)?, num(v, "byteLength").ok_or_else(bad)?)
    };
    if js_sys::Reflect::get(&buf, &"detached".into()).is_ok_and(|d| d.is_truthy()) {
        return Err(js_err(format!("{what}: the buffer is detached (it was transferred)")));
    }
    if !off.is_finite() || off < 0.0 || !len.is_finite() || len < 0.0 || len > usize::MAX as f64 {
        return Err(bad());
    }
    let byte_len = len as usize;
    if byte_len == 0 {
        // Also what a detached buffer reports where `detached` is not implemented.
        return Ok((None, 0));
    }
    let ctor =
        js_sys::Reflect::get(&js_sys::global(), &"Uint8Array".into())?.unchecked_into::<js_sys::Function>();
    let args = js_sys::Array::of3(&buf, &off.into(), &len.into());
    let view = js_sys::Reflect::construct(&ctor, &args)?.unchecked_into::<Uint8Array>();
    Ok((Some(view), byte_len))
}

/// Copy the bytes of a typed array or an `ArrayBuffer`. Everything that can throw in JS goes through catching
/// calls (`Reflect`), so a detached buffer or a wrong type is an `Err`, never an exception through wasm (which
/// would leave a returned promise pending).
fn bytes_of(v: &JsValue, what: &str) -> Result<Vec<u8>, JsValue> {
    let (view, _) = byte_view(v, what)?;
    Ok(view.map_or_else(Vec::new, |v| v.to_vec()))
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

/// A FOLO detector loaded from `.folo` bytes. Create with `FoloDetector.load`; call `free()` to release it.
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
            .ok_or_else(|| js_err(format!("{op}: image is too large ({w}x{h})")))?;
        let (view, byte_len) = byte_view(&data, op)?;
        if byte_len != expected {
            return Err(js_err(format!(
                "{op}: expected {expected} RGBA bytes for a {w}x{h} image, got {byte_len}"
            )));
        }
        let opts: O = options(options_.as_ref())?;
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
    /// Load a `.folo` file (`Uint8Array` or `ArrayBuffer`) onto WebGPU or the CPU (`options.backend`).
    /// (TypeScript: declared in the custom section.)
    #[wasm_bindgen(skip_typescript)]
    pub async fn load(bytes: JsValue, options_: Option<JsValue>) -> Result<FoloDetector, JsValue> {
        let bytes = bytes_of(&bytes, "load")?;
        let opts: LoadOptions = options(options_.as_ref())?;
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
