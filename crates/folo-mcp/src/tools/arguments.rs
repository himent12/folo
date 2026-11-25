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

/// Maximum input length for fuzzy suggestion matching so oversized inputs fail in O(1).
pub(crate) const MAX_SUGGEST_CHARS: usize = 128;

/// Optimal string alignment (Damerau-Levenshtein) distance: insertion, deletion, substitution, or
/// adjacent transposition each count as 1 edit.
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev2 = vec![0usize; n + 1];
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut cur = vec![0usize; n + 1];
    for i in 1..=m {
        cur[0] = i;
        for j in 1..=n {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(prev2[j - 2] + 1);
            }
            cur[j] = best;
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

/// The closest candidate to `needle` (excluding exact matches): a case-insensitive exact match,
/// an unambiguous prefix match (at least 2 chars), or the closest match within 1-2 edits.
pub(crate) fn closest_match<'a>(needle: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let needle_chars: Vec<char> = needle.to_lowercase().chars().collect();
    if needle_chars.is_empty() || needle_chars.len() > MAX_SUGGEST_CHARS || candidates.contains(&needle) {
        return None;
    }
    let needle_lc: String = needle_chars.iter().collect();

    // 1. Case-insensitive exact match (e.g. `api` -> `API`).
    if let Some(exact) =
        candidates.iter().copied().filter(|k| *k != needle && k.to_lowercase() == needle_lc).min()
    {
        return Some(exact);
    }

    // 2. Unambiguous case-insensitive prefix match (unique, or shortest is a prefix of every other
    //    prefix match such as `pred` -> `predict` vs `predict_batch`, while rejecting siblings like `ds_`).
    if needle_chars.len() >= 2 {
        let mut prefix_matches: Vec<(&'a str, String, usize)> = candidates
            .iter()
            .copied()
            .filter(|k| *k != needle)
            .filter_map(|k| {
                let lc = k.to_lowercase();
                lc.starts_with(&needle_lc).then(|| {
                    let len = lc.chars().count();
                    (k, lc, len)
                })
            })
            .collect();
        if !prefix_matches.is_empty() {
            prefix_matches.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(b.0)));
            let (best, best_lc, _) = &prefix_matches[0];
            if prefix_matches[1..].iter().all(|(_, lc, _)| lc.starts_with(best_lc.as_str())) {
                return Some(*best);
            }
        }
    }

    // 3. Edit distance <= 1 for short names (<= 3 chars), <= 2 for longer names (strictly less than
    //    the shorter string length).
    candidates
        .iter()
        .copied()
        .filter(|k| *k != needle)
        .filter_map(|k| {
            let kc: Vec<char> = k.to_lowercase().chars().collect();
            let min_len = kc.len().min(needle_chars.len());
            let max_edits = if min_len <= 3 { 1 } else { 2 };
            if kc.len().abs_diff(needle_chars.len()) > max_edits {
                return None;
            }
            let d = edit_distance(&needle_chars, &kc);
            (d <= max_edits && d < min_len).then_some((d, kc.len(), k))
        })
        .min()
        .map(|(_, _, k)| k)
}

/// Finds the closest `(id, name)` entry for `needle`, matching IDs first and falling back to an
/// unambiguous display name (so passing a dataset/model/run name instead of its ID suggests the ID).
pub(crate) fn closest_named<'a>(needle: &str, items: &'a [(String, String)]) -> Option<(&'a str, &'a str)> {
    let needle_len = needle.chars().count();
    if needle_len == 0 || needle_len > MAX_SUGGEST_CHARS || items.is_empty() {
        return None;
    }
    let ids: Vec<&str> = items.iter().map(|(id, _)| id.as_str()).collect();
    if let Some(id) = closest_match(needle, &ids)
        && let Some((id, name)) = items.iter().find(|(cand, _)| cand == id)
    {
        return Some((id.as_str(), name.as_str()));
    }
    let needle_lc = needle.to_lowercase();
    let exact_names: Vec<&(String, String)> =
        items.iter().filter(|(_, name)| !name.is_empty() && name.to_lowercase() == needle_lc).collect();
    if exact_names.len() == 1 {
        return Some((exact_names[0].0.as_str(), exact_names[0].1.as_str()));
    }
    let mut name_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (_, name) in items {
        if !name.is_empty() {
            *name_counts.entry(name.to_lowercase()).or_default() += 1;
        }
    }
    let unique_names: Vec<&str> = items
        .iter()
        .filter(|(_, name)| !name.is_empty() && name_counts.get(&name.to_lowercase()) == Some(&1))
        .map(|(_, name)| name.as_str())
        .collect();
    if let Some(matched) = closest_match(needle, &unique_names)
        && let Some((id, name)) = items.iter().find(|(_, name)| name == matched)
    {
        return Some((id.as_str(), name.as_str()));
    }
    None
}

/// The known argument name closest to `key`: an alias, a name `key` is a prefix of, or one within two edits.
fn did_you_mean<'a>(key: &str, known: &[&'a str]) -> Option<&'a str> {
    if let Some((_, to)) = ALIASES.iter().find(|(from, _)| from.eq_ignore_ascii_case(key)) {
        return known.iter().copied().find(|k| k == to);
    }
    closest_match(key, known)
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
    check_name_at(what, s, MAX_NAME_CHARS)
}

/// [`check_name`] with the limit the server sets for this kind of name (a run's: `MAX_RUN_NAME_CHARS`).
pub(super) fn check_name_at(what: &str, s: &str, max: usize) -> Result<(), Error> {
    if s.chars().count() > max {
        return Err(invalid_arguments(format!("{what} must be at most {max} characters")));
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
