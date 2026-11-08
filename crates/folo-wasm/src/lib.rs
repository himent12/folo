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
  simd128: boolean;
  threads: number;
  arch: string;
  imgsz: number;
  classes: string[];
  modelId: string;
}

/** What `detect` accepts as an image: ImageData (or any { data, width, height }) or raw RGBA bytes. */
export type FoloImage = ImageData | { data: Uint8Array | Uint8ClampedArray; width: number; height: number } | Uint8Array | Uint8ClampedArray;
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
fn options<T: serde::de::DeserializeOwned + Default>(v: &JsValue) -> Result<T, JsValue> {
    if v.is_undefined() || v.is_null() {
        return Ok(T::default());
    }
    let s: String =
        js_sys::JSON::stringify(v).map_err(|_| js_err("options are not JSON-serialisable"))?.into();
    serde_json::from_str(&s).map_err(|e| js_err(format!("invalid options: {e}")))
}

fn to_js(v: &impl serde::Serialize) -> Result<JsValue, JsValue> {
    let s = serde_json::to_string(v).map_err(js_err)?;
    js_sys::JSON::parse(&s)
}

/// A `Uint8Array` view of a typed array or `ArrayBuffer` (no copy).
fn bytes_view(v: &JsValue) -> Option<Uint8Array> {
    if let Some(b) = v.dyn_ref::<js_sys::ArrayBuffer>() {
        return Some(Uint8Array::new(b));
    }
    let buf = js_sys::Reflect::get(v, &"buffer".into()).ok()?.dyn_into::<js_sys::ArrayBuffer>().ok()?;
    let off = js_sys::Reflect::get(v, &"byteOffset".into()).ok()?.as_f64()? as u32;
    let len = js_sys::Reflect::get(v, &"byteLength".into()).ok()?.as_f64()? as u32;
    Some(Uint8Array::new_with_byte_offset_and_length(&buf, off, len))
}

fn dim(v: &JsValue, key: &str) -> Option<u32> {
    js_sys::Reflect::get(v, &key.into()).ok()?.as_f64().map(|f| f as u32)
}

/// A FOLO detector loaded from `.folo` bytes. Create with `FoloDetector.load`; call `free()` to release it.
#[wasm_bindgen]
pub struct FoloDetector {
    inner: Rc<Detector>,
}

#[wasm_bindgen]
impl FoloDetector {
    /// Load a `.folo` file (`Uint8Array` or `ArrayBuffer`) onto WebGPU or the CPU (`options.backend`).
    pub async fn load(
        #[wasm_bindgen(unchecked_param_type = "Uint8Array | ArrayBuffer")] bytes: JsValue,
        #[wasm_bindgen(unchecked_param_type = "FoloLoadOptions | undefined")] options_: JsValue,
    ) -> Result<FoloDetector, JsValue> {
        let bytes = bytes_view(&bytes).ok_or_else(|| js_err("load: expected a Uint8Array or ArrayBuffer"))?;
        let opts: LoadOptions = options(&options_)?;
        let det = Detector::load(bytes.to_vec(), opts).await.map_err(js_err)?;
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
    #[wasm_bindgen(unchecked_return_type = "Promise<FoloDetectResult>")]
    pub fn detect(
        &self,
        #[wasm_bindgen(unchecked_param_type = "FoloImage")] image: JsValue,
        width: Option<u32>,
        height: Option<u32>,
        #[wasm_bindgen(unchecked_param_type = "FoloDetectOptions | undefined")] options_: JsValue,
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
            let rgba = bytes_view(&data).ok_or_else(|| js_err("detect: image data is not a byte array"))?;
            let opts: DetectOptions = options(&options_)?;
            Ok((rgba.to_vec(), w, h, opts))
        })();
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let (rgba, w, h, opts) = prepared?;
            let res = inner.detect(&rgba, w, h, opts).await.map_err(js_err)?;
            to_js(&res)
        })
    }
}
