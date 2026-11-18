//! The tools' input and output schemas: generated from the Rust types with schemars, then shrunk for
//! `tools/list` (docs/MCP.md "Schemas").

use std::sync::Arc;

use rmcp::handler::server::common::{schema_for_input, schema_for_output};
use rmcp::model::JsonObject;
use schemars::JsonSchema;
use serde_json::{Value, json};

use crate::types::*;

pub(super) fn input<T: JsonSchema + 'static>() -> Arc<JsonObject> {
    let mut s = (*schema_for_input::<T>().expect("tool inputs are objects")).clone();
    compact(&mut s, false);
    close_objects(&mut s);
    s.entry("additionalProperties").or_insert(Value::Bool(false)); // `{}` has no `properties`
    Arc::new(s)
}

/// [`input`] of a labeling tool, shrunk further: a `oneOf` of string constants becomes an `enum` and the `$defs`
/// lose their descriptions (the field's own description says what the values mean), as output schemas do.
pub(super) fn terse_input<T: JsonSchema + 'static>() -> Arc<JsonObject> {
    let mut s = (*input::<T>()).clone();
    if let Some(Value::Object(defs)) = s.get_mut("$defs") {
        for d in defs.values_mut().filter_map(Value::as_object_mut) {
            d.remove("description");
            const_enum(d);
        }
    }
    Arc::new(s)
}

/// Writes a `oneOf` of string constants (a Rust enum with documented variants) as an `enum`.
fn const_enum(o: &mut JsonObject) {
    let consts: Option<Vec<Value>> = o
        .get("oneOf")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().map(|b| b.get("const").filter(|c| c.is_string()).cloned()).collect());
    if let Some(values) = consts {
        o.remove("oneOf");
        o.insert("type".into(), json!("string"));
        o.insert("enum".into(), Value::Array(values));
    }
}

/// Marks every object schema with `properties` (the root and the `$defs`) `additionalProperties: false`: tools
/// reject arguments they do not know ([`unknown_argument`]) instead of silently ignoring a typo.
///
/// [`unknown_argument`]: super::arguments::unknown_argument
fn close_objects(s: &mut JsonObject) {
    if s.contains_key("properties") && !s.contains_key("additionalProperties") {
        s.insert("additionalProperties".into(), Value::Bool(false));
    }
    for v in s.values_mut() {
        match v {
            Value::Object(o) => close_objects(o),
            Value::Array(a) => a.iter_mut().filter_map(Value::as_object_mut).for_each(close_objects),
            _ => {}
        }
    }
}

/// `train`'s input: TrainConfig plus `resume_run_id`, where `dataset_id` is required unless `resume_run_id` is
/// given (a resume may leave every other field out). `name` says the server's limit (`maxLength`).
pub(super) fn train_input() -> Arc<JsonObject> {
    let mut s = (*input::<TrainArgs>()).clone();
    if let Some(Value::Object(name)) = s.get_mut("properties").and_then(|p| p.get_mut("name")) {
        name.insert("maxLength".into(), json!(folo_server::store::MAX_RUN_NAME_CHARS));
    }
    if let Some(Value::Array(required)) = s.get_mut("required") {
        required.retain(|r| r != "dataset_id");
        if required.is_empty() {
            s.remove("required");
        }
    }
    let resume =
        json!({ "required": ["resume_run_id"], "properties": { "resume_run_id": { "type": "string" } } });
    s.insert("anyOf".into(), json!([{ "required": ["dataset_id"] }, resume]));
    Arc::new(s)
}

pub(super) fn output<T: JsonSchema + 'static>() -> Arc<JsonObject> {
    let mut s = (*schema_for_output::<T>()).clone();
    compact(&mut s, true);
    Arc::new(s)
}

/// [`output`] of a shared `folo-api` type (`Run`, `Dataset`, ...): its fields lose their descriptions at the root
/// too, as they do in `$defs`; docs/API.md documents them.
pub(super) fn api_output<T: JsonSchema + 'static>() -> Arc<JsonObject> {
    let mut s = (*output::<T>()).clone();
    if let Some(Value::Object(props)) = s.get_mut("properties") {
        for p in props.values_mut().filter_map(Value::as_object_mut) {
            p.remove("description");
        }
    }
    Arc::new(s)
}

/// Shrinks a generated schema for `tools/list` (docs/MCP.md "Schemas"), keeping what validates. Everywhere: no
/// `$schema`, no numeric `format` (`uint32`, `float`: `integer` / `number` say enough), no `minimum: 0` from
/// unsigned types, and defaults without f32 noise. Output schemas also drop defaults and the descriptions inside
/// shared `$defs` types (the root's fields keep theirs; docs/API.md documents the shared types), and write a
/// `oneOf` of string constants as an `enum`.
fn compact(s: &mut JsonObject, output: bool) {
    /// `v` is a schema, or (`names`) a map from names to schemas (`properties`, `$defs`).
    fn walk(v: &mut Value, output: bool, in_defs: bool, names: bool) {
        match v {
            Value::Object(o) if names => o.values_mut().for_each(|c| walk(c, output, in_defs, false)),
            Value::Object(o) => {
                if o.get("format").and_then(Value::as_str).is_some()
                    && o.get("type") != Some(&json!("string"))
                {
                    o.remove("format");
                }
                // Also of an `Option` of one (`"type": ["integer", "null"]`).
                let integer = match o.get("type") {
                    Some(Value::String(t)) => t == "integer",
                    Some(Value::Array(ts)) => ts.contains(&json!("integer")),
                    _ => false,
                };
                if integer && o.get("minimum") == Some(&json!(0)) {
                    o.remove("minimum");
                }
                if output {
                    o.remove("default");
                    if in_defs {
                        o.remove("description");
                    }
                    const_enum(o);
                }
                for (k, child) in o.iter_mut() {
                    let defs = k == "$defs" || k == "definitions";
                    let names = defs || k == "properties" || k == "patternProperties";
                    // `default`, `enum` and `const` hold values, not schemas.
                    if !matches!(k.as_str(), "default" | "enum" | "const" | "examples") {
                        walk(child, output, in_defs || defs, names);
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|c| walk(c, output, in_defs, false)),
            _ => {}
        }
    }
    s.remove("$schema");
    let mut v = Value::Object(std::mem::take(s));
    walk(&mut v, output, false, false);
    folo_api::json::clean_floats(&mut v);
    if let Value::Object(o) = v {
        *s = o;
    }
}
