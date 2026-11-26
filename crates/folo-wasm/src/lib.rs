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
//! // res: docs/API.md's /api/predict response, plus `backend`
//! ```
//!
//! The logic lives in [`detector`], which has no wasm-bindgen types and is tested natively.

pub mod detector;
pub mod kernels;
mod simd;

use std::rc::Rc;

use js_sys::{Promise, Uint8Array};
use wasm_bindgen::prelude::*;

pub use detector::{
    Backend, BackendChoice, DetectOptions, DetectResult, Detector, DetectorInfo, LoadOptions,
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

// `load` and `detect` are declared here (merged into the class) because wasm-bindgen cannot emit optional
// parameters with custom types.
export interface FoloDetector {
  /** Detect objects; `width` / `height` may be omitted for ImageData. Never throws: errors reject. */
  detect(image: FoloImage, width?: number, height?: number, options?: FoloDetectOptions): Promise<FoloDetectResult>;
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

/// Parse a JS options object through JSON (undefined / null give the defaults).
fn options<T: serde::de::DeserializeOwned + Default>(v: Option<&JsValue>) -> Result<T, JsValue> {
    let Some(v) = v.filter(|v| !v.is_undefined() && !v.is_null()) else { return Ok(T::default()) };
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

/// Copy the bytes of a typed array or an `ArrayBuffer`. Everything that can throw in JS goes through catching
/// calls (`Reflect`), so a detached buffer or a wrong type is an `Err`, never an exception through wasm (which
/// would leave a returned promise pending).
fn bytes_of(v: &JsValue, what: &str) -> Result<Vec<u8>, JsValue> {
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
    if len == 0.0 {
        // Also what a detached buffer reports where `detached` is not implemented.
        return Ok(Vec::new());
    }
    let ctor =
        js_sys::Reflect::get(&js_sys::global(), &"Uint8Array".into())?.unchecked_into::<js_sys::Function>();
    let args = js_sys::Array::of3(&buf, &off.into(), &len.into());
    let view = js_sys::Reflect::construct(&ctor, &args)?.unchecked_into::<Uint8Array>();
    Ok(view.to_vec())
}

fn dim(v: &JsValue, key: &str) -> Option<u32> {
    num(v, key).map(|f| f as u32)
}

/// A FOLO detector loaded from `.folo` bytes. Create with `FoloDetector.load`; call `free()` to release it.
#[wasm_bindgen]
pub struct FoloDetector {
    inner: Rc<Detector>,
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
        width: Option<u32>,
        height: Option<u32>,
        options_: Option<JsValue>,
    ) -> Promise {
        // Everything borrowed from JS is copied before the first await.
        let prepared = (|| -> Result<_, JsValue> {
            let (data, w, h) = if image.is_instance_of::<Uint8Array>()
                || image.is_instance_of::<js_sys::Uint8ClampedArray>()
            {
                (image.clone(), width, height)
            } else {
                let data = js_sys::Reflect::get(&image, &"data".into())
                    .map_err(|_| js_err("detect: expected ImageData or RGBA bytes"))?;
                (data, width.or_else(|| dim(&image, "width")), height.or_else(|| dim(&image, "height")))
            };
            let (w, h) = w.zip(h).ok_or_else(|| js_err("detect: width and height are required"))?;
            let rgba = bytes_of(&data, "detect")?;
            let opts: DetectOptions = options(options_.as_ref())?;
            Ok((rgba, w, h, opts))
        })();
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let (rgba, w, h, opts) = prepared?;
            let res = inner.detect(&rgba, w, h, opts).await.map_err(js_err)?;
            to_js(&res)
        })
    }
}
