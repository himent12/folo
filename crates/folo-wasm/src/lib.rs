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
    MAX_BROWSER_TILES, TileDetectOptions, TtaDetectOptions,
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

/// Call a global constructor's static method (`global[ctor][method](arg)`) through `Reflect::apply` so hostile
/// proxies / revoked proxies reject cleanly instead of throwing synchronously out of wasm.
fn call_static1(ctor_name: &str, method_name: &str, arg: &JsValue) -> Result<JsValue, JsValue> {
    let ctor = js_sys::Reflect::get(&js_sys::global(), &ctor_name.into())
        .map_err(|_| js_err(format!("missing global `{ctor_name}`")))?;
    let method = js_sys::Reflect::get(&ctor, &method_name.into())
        .map_err(|_| js_err(format!("missing `{ctor_name}.{method_name}`")))?;
    let func = method
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_err(format!("`{ctor_name}.{method_name}` is not a function")))?;
    let args = js_sys::Array::of1(arg);
    js_sys::Reflect::apply(&func, &ctor, &args).map_err(|_| js_err("options are not readable"))
}

/// Catching `Array.isArray(v)` (safe even on revoked proxies).
fn is_js_array(v: &JsValue) -> Result<bool, JsValue> {
    Ok(call_static1("Array", "isArray", v)?.is_truthy())
}

/// Catching `Object.keys(v)` (safe on proxies whose `ownKeys` / `getOwnPropertyDescriptor` trap throws).
fn own_enumerable_keys(v: &JsValue) -> Result<js_sys::Array, JsValue> {
    call_static1("Object", "keys", v)?
        .dyn_into::<js_sys::Array>()
        .map_err(|_| js_err("options must be an object"))
}

/// Read a single JS value into a [`serde_json::Value`] once via catching `Reflect` calls, honouring `toJSON()`,
/// unwrapping boxed primitives, and strictly rejecting `NaN` / `±Infinity` (unlike `JSON.stringify`, which turns
/// them into `null`).
fn js_to_json_value(v: &JsValue, path: &str, depth: usize) -> Result<Option<serde_json::Value>, JsValue> {
    if depth > 8 {
        return Err(js_err("invalid options: object nesting too deep"));
    }
    // Honour `toJSON()` once, just like `JSON.stringify`, then validate the returned value in the same pass.
    let owned;
    let cur = if v.is_object() {
        let to_json = js_sys::Reflect::get(v, &"toJSON".into())
            .map_err(|_| js_err("invalid options: cannot read property `toJSON`"))?;
        if let Ok(func) = to_json.dyn_into::<js_sys::Function>() {
            owned = js_sys::Reflect::apply(&func, v, &js_sys::Array::new())
                .map_err(|_| js_err("invalid options: `toJSON()` threw"))?;
            &owned
        } else {
            v
        }
    } else {
        v
    };

    if cur.is_undefined() || cur.is_function() || cur.is_symbol() {
        return Ok(None);
    }
    if cur.is_null() {
        return Ok(Some(serde_json::Value::Null));
    }
    if let Some(b) = cur.as_bool() {
        return Ok(Some(serde_json::Value::Bool(b)));
    }
    if let Some(s) = cur.as_string() {
        return Ok(Some(serde_json::Value::String(s)));
    }
    if let Some(f) = cur.as_f64() {
        return json_number_from_f64(f, path).map(Some);
    }
    if cur.is_bigint() {
        return Err(js_err(format!("invalid options: {path} cannot be a BigInt")));
    }
    if !cur.is_object() {
        return Err(js_err("invalid options: unsupported JS value"));
    }
    // Unwrap boxed JS primitives (`new Number(NaN)`, `new String(...)`, `new Boolean(...)`) safely.
    if cur.is_instance_of::<js_sys::Number>()
        || cur.is_instance_of::<js_sys::JsString>()
        || cur.is_instance_of::<js_sys::Boolean>()
    {
        let value_of = js_sys::Reflect::get(cur, &"valueOf".into())
            .map_err(|_| js_err(format!("invalid options: cannot unwrap {path}")))?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| js_err(format!("invalid options: cannot unwrap {path}")))?;
        let prim = js_sys::Reflect::apply(&value_of, cur, &js_sys::Array::new())
            .map_err(|_| js_err(format!("invalid options: cannot unwrap {path}")))?;
        if let Some(f) = prim.as_f64() {
            return json_number_from_f64(f, path).map(Some);
        }
        if let Some(b) = prim.as_bool() {
            return Ok(Some(serde_json::Value::Bool(b)));
        }
        if let Some(s) = prim.as_string() {
            return Ok(Some(serde_json::Value::String(s)));
        }
    }
    if is_js_array(cur)? {
        let len_val = js_sys::Reflect::get(cur, &"length".into())
            .map_err(|_| js_err(format!("invalid options: cannot read {path}.length")))?;
        let Some(len_f) = len_val.as_f64() else {
            return Err(js_err(format!("invalid options: {path}.length is not a number")));
        };
        if !len_f.is_finite() || len_f < 0.0 || len_f > 10_000.0 || len_f.fract() != 0.0 {
            return Err(js_err(format!("invalid options: invalid array length for {path}")));
        }
        let len = len_f as u32;
        let mut items = Vec::with_capacity(len as usize);
        for i in 0..len {
            let item = js_sys::Reflect::get_u32(cur, i)
                .map_err(|_| js_err(format!("invalid options: cannot read {path}[{i}]")))?;
            let val = js_to_json_value(&item, &format!("{path}[{i}]"), depth + 1)?
                .unwrap_or(serde_json::Value::Null);
            items.push(val);
        }
        return Ok(Some(serde_json::Value::Array(items)));
    }
    let keys = own_enumerable_keys(cur)?;
    let mut map = serde_json::Map::with_capacity(keys.length() as usize);
    for i in 0..keys.length() {
        let key = keys.get(i);
        let Some(name) = key.as_string() else {
            continue;
        };
        let prop = js_sys::Reflect::get(cur, &key)
            .map_err(|_| js_err(format!("invalid options: cannot read property `{name}`")))?;
        let child = if path.is_empty() { name.clone() } else { format!("{path}.{name}") };
        if let Some(val) = js_to_json_value(&prop, &child, depth + 1)? {
            map.insert(name, val);
        }
    }
    Ok(Some(serde_json::Value::Object(map)))
}

fn json_number_from_f64(f: f64, path: &str) -> Result<serde_json::Value, JsValue> {
    if !f.is_finite() {
        return Err(js_err(format!("invalid options: {path} must be finite, got {f}")));
    }
    // Preserve exact integer representation when possible so serde integer fields accept JS integers cleanly.
    if f.fract() == 0.0 && f >= 0.0 && f <= u64::MAX as f64 && (f as u64) as f64 == f {
        return Ok(serde_json::Value::Number(serde_json::Number::from(f as u64)));
    }
    let num = serde_json::Number::from_f64(f)
        .ok_or_else(|| js_err(format!("invalid options: {path} must be finite, got {f}")))?;
    Ok(serde_json::Value::Number(num))
}

/// Validate `f64` option ranges BEFORE `serde` narrows them to `f32` (preventing values like `1.00000001` or
/// `-1e-12` from silently rounding into `[0, 1]` when cast to `f32`).
fn validate_option_f64_ranges(map: &serde_json::Map<String, serde_json::Value>) -> Result<(), String> {
    let check_closed = |key: &str| -> Result<(), String> {
        if let Some(v) = map.get(key)
            && !v.is_null()
            && let Some(f) = v.as_f64()
            && (!f.is_finite() || !(0.0..=1.0).contains(&f))
        {
            return Err(format!("invalid options: {key} must be finite and in [0, 1], got {f}"));
        }
        Ok(())
    };
    let check_half_open = |key: &str| -> Result<(), String> {
        if let Some(v) = map.get(key)
            && !v.is_null()
            && let Some(f) = v.as_f64()
            && (!f.is_finite() || !(0.0..1.0).contains(&f) || (f as f32) >= 1.0)
        {
            return Err(format!("invalid options: {key} must be finite and in [0, 1), got {f}"));
        }
        Ok(())
    };
    for key in ["conf", "iou", "mergeThreshold", "mergeIou"] {
        check_closed(key)?;
    }
    for key in ["overlap", "overlapX", "overlapY"] {
        check_half_open(key)?;
    }
    if let Some(serde_json::Value::Array(scales)) = map.get("scales") {
        for (i, item) in scales.iter().enumerate() {
            if let Some(f) = item.as_f64()
                && (!f.is_finite() || f <= 0.0 || (f as f32) <= 0.0)
            {
                return Err(format!("invalid options: scales[{i}] must be finite and positive, got {f}"));
            }
        }
    }
    Ok(())
}

/// Parse a JS options object cleanly via single-pass catching `Reflect` traversal (undefined / null give defaults).
fn options<T: serde::de::DeserializeOwned + Default>(v: Option<&JsValue>) -> Result<T, JsValue> {
    let Some(v) = v.filter(|v| !v.is_undefined() && !v.is_null()) else { return Ok(T::default()) };
    if !v.is_object() || is_js_array(v)? {
        return Err(js_err("options must be an object"));
    }
    let Some(serde_json::Value::Object(map)) = js_to_json_value(v, "", 0)? else {
        return Err(js_err("options must be an object"));
    };
    validate_option_f64_ranges(&map).map_err(js_err)?;
    serde_json::from_value(serde_json::Value::Object(map))
        .map_err(|e| js_err(format!("invalid options: {e}")))
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
    if !off.is_finite() || off < 0.0 || !len.is_finite() || len < 0.0 || len > isize::MAX as f64 {
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
            .filter(|&b| b <= isize::MAX as usize)
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

#[cfg(test)]
mod tests {
    use super::validate_option_f64_ranges;

    #[test]
    fn f64_range_validator_catches_precision_narrowing_overflows() {
        let parse_map = |s: &str| match serde_json::from_str::<serde_json::Value>(s).unwrap() {
            serde_json::Value::Object(m) => m,
            _ => unreachable!(),
        };
        assert!(validate_option_f64_ranges(&parse_map(r#"{"conf":1.0,"iou":0.0,"overlap":0.9999}"#)).is_ok());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"conf":1.00000001}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"iou":-1e-12}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"mergeThreshold":1.00000001}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"mergeIou":1.00000001}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"overlap":0.99999999}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"scales":[1e-50]}"#)).is_err());
    }
}
