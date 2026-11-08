//! JSON presentation helpers shared by the CLI's `--json` records and the MCP results: floats without f32 → f64
//! noise and paths with `/` separators.

use std::path::Path;

use serde::Serialize;
use serde_json::Value;

/// `v` as a JSON value whose floats read as they were stored ([`clean_floats`]).
pub fn to_value<T: Serialize + ?Sized>(v: &T) -> serde_json::Result<Value> {
    let mut v = serde_json::to_value(v)?;
    clean_floats(&mut v);
    Ok(v)
}

/// Rewrites every number that is an `f32` widened to `f64` as the shortest decimal of that `f32`, so
/// `0.009999999776482582` (the `f32` 0.01 through `serde_json::to_value` or `json!`) reads `0.01` again and a box
/// coordinate `207.92527770996094` reads `207.92528`. Numbers that are not exactly an `f32` (genuine `f64`s such
/// as durations) are left alone; one that is changes by less than half an `f32` step.
pub fn clean_floats(v: &mut Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            if let Some(x) = n.as_f64()
                && let Some(clean) = shortest_f32(x)
                && let Some(num) = serde_json::Number::from_f64(clean)
            {
                *n = num;
            }
        }
        Value::Array(a) => a.iter_mut().for_each(clean_floats),
        Value::Object(o) => o.values_mut().for_each(clean_floats),
        _ => {}
    }
}

fn shortest_f32(x: f64) -> Option<f64> {
    let f = x as f32;
    if !f.is_finite() || f as f64 != x {
        return None;
    }
    let clean: f64 = f.to_string().parse().ok()?;
    (clean != x).then_some(clean)
}

/// A path as FOLO reports it in JSON: `/` separators (Windows accepts them) and no `\\?\` verbatim prefix.
pub fn path(p: impl AsRef<Path>) -> String {
    let s = p.as_ref().to_string_lossy();
    let s = s
        .strip_prefix(r"\\?\UNC\")
        .map(|rest| format!(r"\\{rest}"))
        .unwrap_or_else(|| s.strip_prefix(r"\\?\").unwrap_or(&s).to_string());
    if cfg!(windows) { s.replace('\\', "/") } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_lose_their_f32_noise() {
        let v = to_value(
            &json!({ "lr0": 0.01f32, "hsv": [0.015f32, 0.7f32], "x": 207.925_28f32, "half": 0.5f32 }),
        )
        .unwrap();
        assert_eq!(v.to_string(), r#"{"lr0":0.01,"hsv":[0.015,0.7],"x":207.92528,"half":0.5}"#);
        // Genuine f64s and integers are untouched.
        let v = to_value(&json!({ "s": 31.2f64, "third": 1.0f64 / 3.0, "n": 7, "big": 1e300 })).unwrap();
        assert_eq!(v["s"], 31.2);
        assert_eq!(v["third"], 1.0f64 / 3.0);
        assert_eq!(v["n"], 7);
        assert_eq!(v["big"], 1e300);
    }

    #[test]
    fn paths_use_forward_slashes() {
        assert_eq!(path("runs/a/weights/best.folo"), "runs/a/weights/best.folo");
        if cfg!(windows) {
            assert_eq!(path(r"runs\shapes-n\weights/best.folo"), "runs/shapes-n/weights/best.folo");
            assert_eq!(path(r"\\?\D:\data\x"), "D:/data/x");
            assert_eq!(path(r"\\?\UNC\server\share\x"), "//server/share/x");
        }
    }
}
