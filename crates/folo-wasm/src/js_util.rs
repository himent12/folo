//! Safe JS reflection, strict numeric/options validation, and typed-array copy helpers for `folo-wasm`.
//!
//! Every property read and JS static call goes through catching [`js_sys::Reflect`] helpers so hostile or revoked
//! proxies, detached `ArrayBuffer`s, `NaN`/`±Infinity`, and `BigInt` values produce clean JS `Error`s rather than
//! panicking or throwing raw exceptions across the wasm boundary.

use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;

use crate::tracker::{TrackInputBox, TrackerOptions};

pub(crate) fn js_err(msg: impl std::fmt::Display) -> JsValue {
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
pub(crate) fn is_js_array(v: &JsValue) -> Result<bool, JsValue> {
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

fn check_f64_unit_closed(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<f64>, String> {
    let Some(v) = map.get(key).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let Some(f) = v.as_f64() else {
        return Ok(None);
    };
    if !f.is_finite() || !(0.0..=1.0).contains(&f) {
        return Err(format!("invalid options: {key} must be finite and in [0, 1], got {f}"));
    }
    Ok(Some(f))
}

/// Validate `f64` detector option ranges BEFORE `serde` narrows them to `f32` (preventing values like
/// `1.00000001` or `-1e-12` from silently rounding into `[0, 1]` when cast to `f32`).
pub(crate) fn validate_option_f64_ranges(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), String> {
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
        check_f64_unit_closed(map, key)?;
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

/// Validate `f64` tracker option ranges and ordering BEFORE `serde` narrows them to `f32`.
pub(crate) fn validate_track_option_f64_ranges(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), String> {
    let high = check_f64_unit_closed(map, "highThresh")?;
    let low = check_f64_unit_closed(map, "lowThresh")?;
    let new_tr = check_f64_unit_closed(map, "newTrackThresh")?;
    for key in ["matchIou", "secondMatchIou", "tentativeMatchIou"] {
        check_f64_unit_closed(map, key)?;
    }
    let eff_high = high.unwrap_or(0.5);
    if let Some(l) = low
        && l > eff_high
    {
        return Err(format!("invalid options: lowThresh ({l}) must not exceed highThresh ({eff_high})"));
    }
    if let Some(n) = new_tr
        && n < eff_high
    {
        return Err(format!("invalid options: newTrackThresh ({n}) must be >= highThresh ({eff_high})"));
    }
    Ok(())
}

/// Parse a JS options object cleanly via single-pass catching `Reflect` traversal (undefined / null give defaults).
pub(crate) fn parse_options<T: serde::de::DeserializeOwned + Default>(
    v: Option<&JsValue>,
) -> Result<T, JsValue> {
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

/// Parse and validate JS `FoloTrackOptions` (undefined / null give defaults).
pub(crate) fn parse_track_options(v: Option<&JsValue>) -> Result<TrackerOptions, JsValue> {
    let Some(v) = v.filter(|v| !v.is_undefined() && !v.is_null()) else {
        return Ok(TrackerOptions::default());
    };
    if !v.is_object() || is_js_array(v)? {
        return Err(js_err("options must be an object"));
    }
    let Some(serde_json::Value::Object(map)) = js_to_json_value(v, "", 0)? else {
        return Err(js_err("options must be an object"));
    };
    validate_track_option_f64_ranges(&map).map_err(js_err)?;
    serde_json::from_value(serde_json::Value::Object(map))
        .map_err(|e| js_err(format!("invalid options: {e}")))
}

/// Validate parsed JSON box entries (each must be a JSON object, and every float field must remain finite when
/// narrowed to `f32`) before deserializing into [`TrackInputBox`]. Prevents positional arrays or `f64` overflows
/// like `1e39` (which narrow to `f32::INFINITY`) from silently passing validation and advancing tracker state.
pub(crate) fn parse_track_boxes_json(items: Vec<serde_json::Value>) -> Result<Vec<TrackInputBox>, String> {
    for (i, item) in items.iter().enumerate() {
        let Some(map) = item.as_object() else {
            return Err(format!("update: boxes[{i}] must be an object"));
        };
        for key in ["score", "x1", "y1", "x2", "y2"] {
            if let Some(v) = map.get(key)
                && let Some(f) = v.as_f64()
                && (!f.is_finite() || !(f as f32).is_finite())
            {
                return Err(format!("update: boxes[{i}].{key} overflows f32 range, got {f}"));
            }
        }
    }
    serde_json::from_value(serde_json::Value::Array(items))
        .map_err(|e| format!("update: invalid detection boxes: {e}"))
}

/// Parse a JS array of detection boxes (`FoloTrackInputBox[]`) for `FoloTracker.update`.
pub(crate) fn parse_track_boxes(v: &JsValue) -> Result<Vec<TrackInputBox>, JsValue> {
    if !v.is_object() || !is_js_array(v)? {
        return Err(js_err("update: expected an array of detection boxes"));
    }
    let Some(serde_json::Value::Array(items)) = js_to_json_value(v, "boxes", 0)? else {
        return Err(js_err("update: expected an array of detection boxes"));
    };
    parse_track_boxes_json(items).map_err(js_err)
}

pub(crate) fn to_js(v: &impl serde::Serialize) -> Result<JsValue, JsValue> {
    let s = serde_json::to_string(v).map_err(js_err)?;
    js_sys::JSON::parse(&s)
}

fn num(v: &JsValue, key: &str) -> Option<f64> {
    js_sys::Reflect::get(v, &key.into()).ok()?.as_f64()
}

/// Parse an optional JS pixel dimension, strictly rejecting `NaN`, `Infinity`, negatives, zero, fractions, or
/// non-numeric values rather than letting WebIDL `ToUint32` silently coerce them.
pub(crate) fn parse_dim(v: Option<&JsValue>, op: &str, name: &str) -> Result<Option<u32>, JsValue> {
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
pub(crate) fn object_dim(v: &JsValue, op: &str, key: &str) -> Result<Option<u32>, JsValue> {
    let prop = js_sys::Reflect::get(v, &key.into())
        .map_err(|_| js_err(format!("{op}: expected ImageData or RGBA bytes")))?;
    parse_dim(Some(&prop), op, key)
}

/// Inspect a JS typed array or `ArrayBuffer`, validating that it is not detached and returning a `Uint8Array`
/// view along with its byte length.
pub(crate) fn byte_view(v: &JsValue, what: &str) -> Result<(Option<Uint8Array>, usize), JsValue> {
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
pub(crate) fn bytes_of(v: &JsValue, what: &str) -> Result<Vec<u8>, JsValue> {
    let (view, _) = byte_view(v, what)?;
    Ok(view.map_or_else(Vec::new, |v| v.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::{parse_track_boxes_json, validate_option_f64_ranges, validate_track_option_f64_ranges};
    use crate::tracker::Tracker;

    fn parse_map(s: &str) -> serde_json::Map<String, serde_json::Value> {
        match serde_json::from_str::<serde_json::Value>(s).unwrap() {
            serde_json::Value::Object(m) => m,
            _ => unreachable!(),
        }
    }

    #[test]
    fn f64_range_validator_catches_precision_narrowing_overflows() {
        assert!(validate_option_f64_ranges(&parse_map(r#"{"conf":1.0,"iou":0.0,"overlap":0.9999}"#)).is_ok());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"conf":1.00000001}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"iou":-1e-12}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"mergeThreshold":1.00000001}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"mergeIou":1.00000001}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"overlap":0.99999999}"#)).is_err());
        assert!(validate_option_f64_ranges(&parse_map(r#"{"scales":[1e-50]}"#)).is_err());
    }

    #[test]
    fn track_f64_range_validator_catches_narrowing_and_ordering() {
        assert!(
            validate_track_option_f64_ranges(&parse_map(
                r#"{"highThresh":0.5,"lowThresh":0.1,"newTrackThresh":0.6,"matchIou":0.2}"#
            ))
            .is_ok()
        );
        assert!(validate_track_option_f64_ranges(&parse_map(r#"{"highThresh":1.00000001}"#)).is_err());
        assert!(validate_track_option_f64_ranges(&parse_map(r#"{"lowThresh":-1e-12}"#)).is_err());
        assert!(
            validate_track_option_f64_ranges(&parse_map(r#"{"highThresh":0.5,"lowThresh":0.50000001}"#))
                .is_err()
        );
        assert!(
            validate_track_option_f64_ranges(&parse_map(r#"{"highThresh":0.5,"newTrackThresh":0.49999999}"#))
                .is_err()
        );
    }

    #[test]
    fn parse_track_boxes_rejects_positional_arrays_and_f32_overflow_without_advancing_tracker() {
        let mut trk = Tracker::default();
        let valid: serde_json::Value = serde_json::from_str(
            r#"[{"class":0,"name":"person","score":0.9,"x1":10,"y1":10,"x2":50,"y2":60}]"#,
        )
        .unwrap();
        let parsed = parse_track_boxes_json(valid.as_array().unwrap().clone()).unwrap();
        let out = trk.update(&parsed);
        assert_eq!((out.len(), trk.frame_id(), trk.active_tracks()), (1, 1, 1));

        // Positional array must be rejected (not coerced via serde's struct-from-seq).
        let seq_val: serde_json::Value = serde_json::from_str(r#"[[0,"person",0.9,10,10,50,60]]"#).unwrap();
        let err = parse_track_boxes_json(seq_val.as_array().unwrap().clone()).unwrap_err();
        assert!(err.contains("must be an object"), "{err}");

        // Finite f64 overflowing f32 (1e39 -> f32::INFINITY) must be rejected before Tracker::update runs.
        for bad_json in [
            r#"[{"class":0,"score":1e39,"x1":10,"y1":10,"x2":50,"y2":60}]"#,
            r#"[{"class":0,"score":0.9,"x1":10,"y1":10,"x2":1e39,"y2":60}]"#,
            r#"[{"class":0,"score":0.9,"x1":-1e39,"y1":10,"x2":50,"y2":60}]"#,
        ] {
            let v: serde_json::Value = serde_json::from_str(bad_json).unwrap();
            let err = parse_track_boxes_json(v.as_array().unwrap().clone()).unwrap_err();
            assert!(err.contains("overflows f32"), "{err}");
        }

        // Tracker frame_id and active tracks must remain completely untouched after rejected inputs.
        assert_eq!((trk.frame_id(), trk.active_tracks(), trk.lost_tracks()), (1, 1, 0));
    }
}
