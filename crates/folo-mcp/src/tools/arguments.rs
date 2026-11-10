//! Checking a tool call's arguments: unknown names (with a suggestion), parsing, names and `0..=1` values.

use rmcp::model::JsonObject;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::results::{MAX_NAME_CHARS, short};
use crate::errors::{Error, invalid_arguments};

/// The object schema `s` stands for: itself, its `$ref` target in `root`'s `$defs`, or the object branch of an
/// `anyOf` / `oneOf` (how schemars writes `Option<T>`).
fn object_schema<'a>(s: &'a JsonObject, root: &'a JsonObject) -> Option<&'a JsonObject> {
    if s.contains_key("properties") {
        return Some(s);
    }
    if let Some(r) = s.get("$ref").and_then(Value::as_str) {
        let name = r.rsplit('/').next()?;
        let defs = root.get("$defs").or_else(|| root.get("definitions"))?.as_object()?;
        return object_schema(defs.get(name)?.as_object()?, root);
    }
    ["anyOf", "oneOf", "allOf"].iter().filter_map(|k| s.get(*k)?.as_array()).flatten().find_map(|b| {
        let b = b.as_object()?;
        object_schema(b, root)
    })
}

/// Common wrong names for tool arguments, and what they are called here.
const ALIASES: &[(&str, &str)] = &[
    ("lr", "lr0"),
    ("learning_rate", "lr0"),
    ("model", "arch"),
    ("model_name", "arch"),
    ("dataset", "dataset_id"),
    ("data", "dataset_id"),
    ("batch_size", "batch"),
    ("img_size", "imgsz"),
    ("image_size", "imgsz"),
    ("imgsize", "imgsz"),
    ("run", "run_id"),
    ("resume", "resume_run_id"),
    ("model_path", "model_id"),
    ("path", "image_path"),
    ("image", "image_path"),
    ("max_detections", "max_det"),
    ("confidence", "conf"),
    ("page_size", "limit"),
];

/// The known name closest to `key`: an alias, a name `key` is a prefix of, or one within two edits.
fn did_you_mean<'a>(key: &str, known: &[&'a str]) -> Option<&'a str> {
    if let Some((_, to)) = ALIASES.iter().find(|(from, _)| *from == key) {
        return known.iter().copied().find(|k| k == to);
    }
    fn edits(a: &str, b: &str) -> usize {
        let b: Vec<char> = b.chars().collect();
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.chars().enumerate() {
            let mut prev = row[0];
            row[0] = i + 1;
            for (j, cb) in b.iter().enumerate() {
                let cur = row[j + 1];
                row[j + 1] = (prev + usize::from(ca != *cb)).min(row[j] + 1).min(cur + 1);
                prev = cur;
            }
        }
        row[b.len()]
    }
    let prefix =
        known.iter().copied().filter(|k| k.starts_with(key) && key.len() >= 2).min_by_key(|k| k.len());
    prefix.or_else(|| {
        known
            .iter()
            .copied()
            .map(|k| (edits(key, k), k))
            .filter(|&(d, k)| d <= 2 && d < k.len().min(key.len()))
            .min_by_key(|&(d, _)| d)
            .map(|(_, k)| k)
    })
}

/// The first argument (dotted path for nested objects) that `schema` does not declare, as an `invalid_arguments`
/// error naming the accepted ones.
pub(super) fn unknown_argument(tool: &str, schema: &JsonObject, args: &JsonObject) -> Option<Error> {
    fn walk(
        prefix: &str,
        s: &JsonObject,
        root: &JsonObject,
        v: &JsonObject,
    ) -> Option<(String, Vec<String>)> {
        let empty = JsonObject::new();
        let props = s.get("properties").and_then(Value::as_object).unwrap_or(&empty);
        for (k, val) in v {
            let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            match props.get(k) {
                None => return Some((path, props.keys().cloned().collect())),
                Some(Value::Object(ps)) => {
                    if let (Value::Object(inner), Some(os)) = (val, object_schema(ps, root))
                        && let Some(found) = walk(&path, os, root, inner)
                    {
                        return Some(found);
                    }
                }
                Some(_) => {}
            }
        }
        None
    }
    let (path, known) = walk("", schema, schema, args)?;
    let names: Vec<&str> = known.iter().map(String::as_str).collect();
    let last = path.rsplit('.').next().unwrap_or(&path);
    let mut msg = format!("unknown argument `{}`", short(&path, 64));
    if let Some(s) = did_you_mean(last, &names) {
        msg.push_str(&format!("; did you mean `{s}`?"));
    }
    Some(
        invalid_arguments(msg)
            .hint(format!("{tool} takes {}; see its inputSchema (tools/list)", names.join(", "))),
    )
}

pub(super) fn check_name(what: &str, s: &str) -> Result<(), Error> {
    if s.chars().count() > MAX_NAME_CHARS {
        return Err(invalid_arguments(format!("{what} must be at most {MAX_NAME_CHARS} characters")));
    }
    Ok(())
}

pub(super) fn parse_args<T: DeserializeOwned>(args: JsonObject) -> Result<T, Error> {
    serde_json::from_value(Value::Object(args))
        .map_err(|e| invalid_arguments(format!("invalid arguments: {e}")))
}

pub(super) fn check_unit(name: &str, v: f32) -> Result<(), Error> {
    if (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(invalid_arguments(format!("{name} must be in 0..=1, got {v}")))
    }
}
