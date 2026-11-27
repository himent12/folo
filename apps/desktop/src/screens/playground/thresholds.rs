//! Playground per-class confidence threshold state, effective filtering, cross-model class
//! remapping, and JSON import/export compatible with `folo_vision::ThresholdCalibration`
//! (`folo val --save-thresholds` / `folo predict --thresholds`) and Studio (`folo.thresholds.v1`).

use std::collections::{BTreeMap, BTreeSet};

use folo_vision::ThresholdCalibration;
use serde_json::{Value, json};

use crate::api::ScoredBox;

/// Maximum JSON bytes accepted when importing a thresholds file from disk.
pub const MAX_THRESHOLDS_JSON_BYTES: u64 = 4 * 1024 * 1024;

/// Playground per-class confidence override state owned by `FoloApp`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlayThresholdsState {
    /// Active per-class overrides keyed by Model A's class index (`0..classes.len()`).
    pub per_class: BTreeMap<u32, f32>,
    /// Whether the collapsible "Per-class thresholds" panel is expanded in the side rail.
    pub expanded: bool,
    /// Whether the inline JSON paste/file import form is currently open.
    pub import_open: bool,
    /// Validation error from the most recent JSON import attempt.
    pub import_error: Option<String>,
    /// Concise confirmation note after importing thresholds or handing off calibrated thresholds.
    pub import_note: Option<String>,
}

impl PlayThresholdsState {
    /// Number of active per-class overrides.
    pub fn override_count(&self) -> usize {
        self.per_class.len()
    }

    /// Resets overrides and transient UI state (used on backend switch).
    pub fn reset_for_backend(&mut self) {
        self.per_class.clear();
        self.expanded = false;
        self.import_open = false;
        self.import_error = None;
        self.import_note = None;
    }

    /// Clears transient import status/error messages without disturbing overrides.
    pub fn clear_feedback(&mut self) {
        self.import_error = None;
        self.import_note = None;
    }
}

/// Clamps `v` into `[0.0, 1.0]` and rounds to 4 decimal places (preserving 0.001 calibration grid steps).
pub fn round_conf(v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    ((v.clamp(0.0, 1.0) as f64) * 10_000.0).round() as f32 / 10_000.0
}

/// Formats a confidence threshold for badges and readouts: 2 decimals for standard 0.01 steps
/// (`0.25`, `0.40`), preserving 3–4 decimals for finer calibration points (`0.375`, `0.005`).
pub fn fmt_threshold(v: f32) -> String {
    let r = round_conf(v);
    let two = ((r as f64) * 100.0).round() / 100.0;
    if ((r as f64) - two).abs() <= 1e-6 {
        return format!("{two:.2}");
    }
    let three = ((r as f64) * 1000.0).round() / 1000.0;
    if ((r as f64) - three).abs() <= 1e-6 {
        return format!("{three:.3}");
    }
    format!("{r:.4}")
}

/// Effective confidence threshold for `class_id`: its override if set, else `global_conf`.
#[inline]
pub fn effective_conf(global_conf: f32, per_class: &BTreeMap<u32, f32>, class_id: u32) -> f32 {
    per_class.get(&class_id).copied().unwrap_or(global_conf)
}

/// True when a cached prediction fetched with `(cached_floor, cached_per_class)` can safely serve
/// `(current_conf, current_per_class)` without risking `max_det` truncation mismatches.
///
/// - When neither side has per-class overrides, any `current_conf >= cached_floor` is valid (so
///   dragging the global slider filters locally without extra inference calls).
/// - Whenever either side has per-class overrides, both the override map and global threshold must
///   match so per-class filtering always runs before the engine's `max_det` cap.
pub fn thresholds_match(
    cached_per_class: &BTreeMap<u32, f32>,
    cached_floor: f32,
    current_conf: f32,
    current_per_class: &BTreeMap<u32, f32>,
) -> bool {
    if cached_per_class.is_empty() && current_per_class.is_empty() {
        current_conf >= cached_floor - 1e-6
    } else {
        (current_conf - cached_floor).abs() <= 1e-6
            && cached_per_class.len() == current_per_class.len()
            && cached_per_class
                .iter()
                .zip(current_per_class.iter())
                .all(|((ka, va), (kb, vb))| ka == kb && (va - vb).abs() <= 1e-6)
    }
}

/// Filters `boxes` by effective per-class confidence thresholds (`score >= effective_conf`) and
/// optional class index filter.
pub fn filter_detections_with_per_class<'a>(
    boxes: &'a [ScoredBox],
    global_conf: f32,
    per_class: &BTreeMap<u32, f32>,
    class: Option<u32>,
) -> Vec<&'a ScoredBox> {
    boxes
        .iter()
        .filter(|b| {
            b.score >= effective_conf(global_conf, per_class, b.class) && class.is_none_or(|c| b.class == c)
        })
        .collect()
}

/// Remaps per-class confidence overrides from one model's class list onto another model's class
/// list by class name. Identical class lists preserve overrides verbatim; incompatible or missing
/// classes are dropped so stale overrides never silently leak across models.
pub fn map_per_class_to_classes(
    per_class: &BTreeMap<u32, f32>,
    from_classes: &[String],
    to_classes: &[String],
) -> BTreeMap<u32, f32> {
    if per_class.is_empty() {
        return BTreeMap::new();
    }
    if !from_classes.is_empty() && from_classes == to_classes {
        return per_class
            .iter()
            .filter(|&(&id, &v)| {
                (id as usize) < to_classes.len() && v.is_finite() && (0.0..=1.0).contains(&v)
            })
            .map(|(&id, &v)| (id, round_conf(v)))
            .collect();
    }
    if from_classes.is_empty() || to_classes.is_empty() {
        return BTreeMap::new();
    }
    let mut to_by_name: BTreeMap<&str, u32> = BTreeMap::new();
    for (idx, name) in to_classes.iter().enumerate() {
        to_by_name.entry(name.as_str()).or_insert(idx as u32);
    }
    let mut out = BTreeMap::new();
    for (&from_id, &v) in per_class {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            continue;
        }
        if let Some(name) = from_classes.get(from_id as usize)
            && let Some(&target_id) = to_by_name.get(name.as_str())
        {
            out.insert(target_id, round_conf(v));
        }
    }
    out
}

/// Exports the active global and per-class thresholds as pretty-printed JSON compatible with:
/// - `folo_vision::ThresholdCalibration` / `folo predict --thresholds` (`global`, `macro_f1`, `per_class`)
/// - Studio Playground (`folo.thresholds.v1`, `default_conf`, `global.best_conf`, `per_class[{class, name, best_conf}]`)
///
/// When no per-class overrides are active, populates `per_class` with every class in `classes` at
/// `global_conf` so `folo predict --thresholds` (`ThresholdCalibration::is_empty() == false`)
/// honors `global_conf` instead of silently falling back to the CLI's default `0.25`.
pub fn export_thresholds_json(
    global_conf: f32,
    per_class: &BTreeMap<u32, f32>,
    classes: &[String],
) -> anyhow::Result<String> {
    let g = round_conf(global_conf);
    let make_entry = |cid: u32, val: f32| {
        let mut obj = serde_json::Map::new();
        obj.insert("class".into(), json!(cid));
        if let Some(name) = classes.get(cid as usize) {
            obj.insert("name".into(), json!(name));
        }
        obj.insert("best_conf".into(), json!(round_conf(val)));
        obj.insert("best_f1".into(), json!(0.0));
        obj.insert("precision".into(), json!(0.0));
        obj.insert("recall".into(), json!(0.0));
        obj.insert("truths".into(), json!(0));
        obj.insert("preds".into(), json!(0));
        Value::Object(obj)
    };
    let per: Vec<Value> = if !per_class.is_empty() {
        per_class
            .iter()
            .filter(|(_, v)| v.is_finite() && (0.0..=1.0).contains(*v))
            .map(|(&cid, &val)| make_entry(cid, val))
            .collect()
    } else if !classes.is_empty() {
        (0..classes.len() as u32).map(|cid| make_entry(cid, g)).collect()
    } else {
        anyhow::bail!(
            "Select a model before exporting thresholds JSON (or pass --conf {} on CLI).",
            fmt_threshold(g)
        );
    };
    let payload = json!({
        "schema": "folo.thresholds.v1",
        "default_conf": g,
        "global": {
            "best_conf": g,
            "best_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        },
        "macro_f1": 0.0,
        "per_class": per,
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

/// Exports a full calibrated `ThresholdCalibration` (including F1/precision/recall/counts and
/// human-readable class names), matching `folo val --save-thresholds`.
pub fn export_calibration_json(cal: &ThresholdCalibration, classes: &[String]) -> anyhow::Result<String> {
    let per: Vec<Value> = cal
        .per_class
        .iter()
        .map(|c| {
            let mut obj = serde_json::Map::new();
            obj.insert("class".into(), json!(c.class));
            if let Some(name) = classes.get(c.class as usize) {
                obj.insert("name".into(), json!(name));
            }
            obj.insert("best_conf".into(), json!(c.best_conf));
            obj.insert("best_f1".into(), json!(c.best_f1));
            obj.insert("precision".into(), json!(c.precision));
            obj.insert("recall".into(), json!(c.recall));
            obj.insert("truths".into(), json!(c.truths));
            obj.insert("preds".into(), json!(c.preds));
            Value::Object(obj)
        })
        .collect();
    let payload = json!({
        "schema": "folo.thresholds.v1",
        "default_conf": cal.global.best_conf,
        "global": {
            "best_conf": cal.global.best_conf,
            "best_f1": cal.global.best_f1,
            "precision": cal.global.precision,
            "recall": cal.global.recall,
        },
        "macro_f1": cal.macro_f1,
        "per_class": per,
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

/// Parsed result from [`parse_thresholds_json`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedThresholds {
    /// Global confidence threshold in `[0, 1]` when supplied by the JSON, else `None`.
    pub global_conf: Option<f32>,
    /// Per-class confidence overrides keyed by resolved class ID.
    pub per_class: BTreeMap<u32, f32>,
    /// Non-fatal notes (for example, skipped unknown or out-of-range classes).
    pub warnings: Vec<String>,
}

fn is_canonical_uint(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s == "0" {
        return true;
    }
    let bytes = s.as_bytes();
    bytes[0].is_ascii_digit() && bytes[0] != b'0' && bytes.iter().all(u8::is_ascii_digit)
}

fn looks_numeric(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return false;
    }
    let rest = t.strip_prefix(['+', '-']).unwrap_or(t);
    let mut parts = rest.split('.');
    let Some(int_part) = parts.next() else { return false };
    if int_part.is_empty() || !int_part.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    match (parts.next(), parts.next()) {
        (None, _) => true,
        (Some(frac), None) => !frac.is_empty() && frac.bytes().all(|b| b.is_ascii_digit()),
        _ => false,
    }
}

fn parse_unit_f32(v: &Value) -> Option<f32> {
    let n = v.as_f64()?;
    if !n.is_finite() || !(0.0..=1.0).contains(&n) {
        return None;
    }
    Some(round_conf(n as f32))
}

enum ClassLookup {
    Id(u32),
    Unknown(String),
    Ambiguous(String),
}

fn resolve_class_key(raw: &str, classes: &[String]) -> ClassLookup {
    if is_canonical_uint(raw)
        && let Ok(id) = raw.parse::<u32>()
    {
        return ClassLookup::Id(id);
    }
    if let Some(idx) = classes.iter().position(|c| c == raw) {
        return ClassLookup::Id(idx as u32);
    }
    let needle = raw.trim().to_ascii_lowercase();
    if !needle.is_empty() {
        let matches: Vec<u32> = classes
            .iter()
            .enumerate()
            .filter(|(_, name)| name.trim().to_ascii_lowercase() == needle)
            .map(|(i, _)| i as u32)
            .collect();
        if matches.len() == 1 {
            return ClassLookup::Id(matches[0]);
        }
        if matches.len() > 1 {
            return ClassLookup::Ambiguous(raw.to_string());
        }
    }
    ClassLookup::Unknown(raw.to_string())
}

/// Parses threshold JSON in any supported FOLO schema:
/// 1. `ThresholdCalibration` (`{ global: { best_conf }, per_class: [{ class, name?, best_conf }] }`,
///    or wrapped under `{ thresholds: { ... } }`, or with `default_conf` / object `per_class` map),
/// 2. `folo val --json` output with top-level `best_f1_conf` and metrics-only `per_class`, OR
/// 3. Simple flat object map keyed by class ID (`{ "0": 0.4 }`) or class name (`{ "circle": 0.4 }`).
pub fn parse_thresholds_json(raw: &str, classes: &[String]) -> Result<ParsedThresholds, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Paste or load a JSON threshold file.".into());
    }
    let parsed: Value = serde_json::from_str(trimmed).map_err(|e| format!("Invalid JSON: {e}"))?;
    let mut root = parsed.as_object().ok_or_else(|| {
        "Expected a JSON object (ThresholdCalibration or { \"class\": conf } map).".to_string()
    })?;

    // Unwrap `{ "thresholds": { ... } }` (e.g. `folo val --json` or `POST /api/val` envelope).
    if let Some(inner) = root.get("thresholds").and_then(Value::as_object)
        && (inner.contains_key("per_class")
            || inner.contains_key("global")
            || inner.contains_key("default_conf"))
    {
        root = inner;
    }

    let mut warnings = Vec::new();

    // Format 1b: `folo val --json` with top-level `best_f1_conf` and metrics-only `per_class`.
    let is_val_metrics_only = root.get("best_f1_conf").is_some_and(Value::is_number)
        && !root.contains_key("global")
        && !root.contains_key("default_conf")
        && root.get("per_class").and_then(Value::as_array).is_some_and(|arr| {
            arr.iter().all(|e| {
                e.as_object().is_some_and(|o| !o.contains_key("best_conf") && !o.contains_key("conf"))
            })
        });
    if is_val_metrics_only {
        let g = parse_unit_f32(root.get("best_f1_conf").unwrap())
            .ok_or_else(|| "best_f1_conf must be a finite number in [0, 1].".to_string())?;
        warnings.push("No per-class best_conf entries found; applied global best_f1_conf only.".into());
        return Ok(ParsedThresholds { global_conf: Some(g), per_class: BTreeMap::new(), warnings });
    }

    // Format 1: Structured ThresholdCalibration / `folo.thresholds.v1` / `{ default_conf, per_class }`.
    let is_structured = root.contains_key("per_class")
        || root.contains_key("default_conf")
        || root.get("global").is_some_and(|v| v.is_object() || v.is_number());
    if is_structured {
        // Guard empty `ThresholdCalibration` files emitted by `folo val --save-thresholds` on a
        // split with no ground-truth boxes (`per_class: []`, zeroed `global`), matching Models'
        // handoff guard so Playground never silently resets confidence to `0.00`.
        if !root.contains_key("default_conf")
            && root.get("schema").and_then(Value::as_str) != Some("folo.thresholds.v1")
            && root.get("global").is_some_and(Value::is_object)
            && root.get("per_class").and_then(Value::as_array).is_some_and(Vec::is_empty)
            && serde_json::from_value::<ThresholdCalibration>(Value::Object(root.clone()))
                .is_ok_and(|cal| cal.is_empty())
        {
            return Err(
                "This calibration file has no calibrated classes (no ground-truth boxes in the evaluated split). Existing thresholds were kept."
                    .into(),
            );
        }
        let mut global_conf: Option<f32> = None;
        if let Some(gv) = root.get("global").filter(|v| !v.is_null()) {
            if gv.is_number() {
                global_conf = Some(
                    parse_unit_f32(gv)
                        .ok_or_else(|| "global threshold must be a finite number in [0, 1].".to_string())?,
                );
            } else if let Some(gobj) = gv.as_object() {
                if let Some(v) = gobj.get("best_conf").or_else(|| gobj.get("conf")).filter(|v| !v.is_null()) {
                    global_conf =
                        Some(parse_unit_f32(v).ok_or_else(|| {
                            "global.best_conf must be a finite number in [0, 1].".to_string()
                        })?);
                }
            } else {
                return Err("global must be an object with best_conf or a number in [0, 1].".into());
            }
        }
        if global_conf.is_none()
            && let Some(dv) = root.get("default_conf").filter(|v| !v.is_null())
        {
            global_conf = Some(
                parse_unit_f32(dv)
                    .ok_or_else(|| "default_conf must be a finite number in [0, 1].".to_string())?,
            );
        }

        let mut per_class = BTreeMap::new();
        if let Some(raw_pc) = root.get("per_class").filter(|v| !v.is_null()) {
            if let Some(arr) = raw_pc.as_array() {
                let mut seen = BTreeSet::new();
                for (i, entry) in arr.iter().enumerate() {
                    let rec =
                        entry.as_object().ok_or_else(|| format!("per_class[{i}] must be an object."))?;
                    let class_id = match rec.get("class") {
                        Some(v) if v.is_u64() => {
                            let id = v.as_u64().unwrap();
                            u32::try_from(id).map_err(|_| {
                                format!("per_class[{i}].class must be a non-negative integer.")
                            })?
                        }
                        Some(v) if v.is_string() => match resolve_class_key(v.as_str().unwrap(), classes) {
                            ClassLookup::Id(id) => id,
                            ClassLookup::Unknown(u) => {
                                warnings.push(format!("Skipped unknown class \"{u}\" in per_class[{i}]."));
                                continue;
                            }
                            ClassLookup::Ambiguous(a) => {
                                return Err(format!("Class name \"{a}\" is ambiguous."));
                            }
                        },
                        Some(v) if v.is_number() => {
                            return Err(format!("per_class[{i}].class must be a non-negative integer."));
                        }
                        _ => return Err(format!("per_class[{i}].class is required.")),
                    };
                    // When class name metadata is also provided and the current model has a known
                    // class list, remap by name if the index does not match (prevents cross-dataset leakage).
                    let resolved_id = if !classes.is_empty() {
                        if let Some(name) = rec.get("name").and_then(Value::as_str) {
                            if classes.get(class_id as usize).is_some_and(|c| c == name) {
                                class_id
                            } else if let Some(idx) = classes.iter().position(|c| c == name) {
                                idx as u32
                            } else {
                                warnings.push(format!(
                                    "Skipped class \"{name}\" (id {class_id}) not in this model."
                                ));
                                continue;
                            }
                        } else if (class_id as usize) >= classes.len() {
                            warnings.push(format!(
                                "Skipped out-of-range class id {class_id} (model has {} classes).",
                                classes.len()
                            ));
                            continue;
                        } else {
                            class_id
                        }
                    } else {
                        class_id
                    };

                    let raw_val = rec.get("best_conf").or_else(|| rec.get("conf")).ok_or_else(|| {
                        format!("per_class[{i}].best_conf must be a finite number in [0, 1].")
                    })?;
                    let conf = parse_unit_f32(raw_val).ok_or_else(|| {
                        format!("per_class[{i}].best_conf must be a finite number in [0, 1].")
                    })?;
                    if !seen.insert(resolved_id) {
                        return Err(format!("Duplicate per_class entry for class {resolved_id}."));
                    }
                    per_class.insert(resolved_id, conf);
                }
            } else if let Some(map) = raw_pc.as_object() {
                parse_class_map(map, classes, &mut per_class, &mut warnings, false)?;
            } else {
                return Err(
                    "per_class must be an array of { class, best_conf } objects or a { class: conf } map."
                        .into(),
                );
            }
        }
        // Uniform full-class global-only export from `folo.thresholds.v1`: restore the global
        // cutoff without cluttering Playground with redundant per-class overrides equal to global.
        if let Some(g) = global_conf
            && root.get("schema").and_then(Value::as_str) == Some("folo.thresholds.v1")
            && !classes.is_empty()
            && per_class.len() == classes.len()
            && per_class.values().all(|&v| (v - g).abs() <= 1e-6)
        {
            per_class.clear();
        }
        return Ok(ParsedThresholds { global_conf, per_class, warnings });
    }

    // Format 2: Flat `{ "0": 0.4, "circle": 0.65 }` map.
    let mut per_class = BTreeMap::new();
    parse_class_map(root, classes, &mut per_class, &mut warnings, true)?;
    Ok(ParsedThresholds { global_conf: None, per_class, warnings })
}

fn parse_class_map(
    map: &serde_json::Map<String, Value>,
    classes: &[String],
    out: &mut BTreeMap<u32, f32>,
    warnings: &mut Vec<String>,
    require_match_if_unknown: bool,
) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    let mut unknown = Vec::new();
    for (k, v) in map {
        let conf = parse_unit_f32(v)
            .ok_or_else(|| format!("Threshold for \"{k}\" must be a finite number in [0, 1]."))?;
        if !classes.iter().any(|c| c == k) && looks_numeric(k) && !is_canonical_uint(k) {
            return Err(format!("Invalid class ID \"{k}\": must be a non-negative integer or class name."));
        }
        let id = match resolve_class_key(k, classes) {
            ClassLookup::Id(id) => {
                if !classes.is_empty() && (id as usize) >= classes.len() {
                    unknown.push(k.clone());
                    continue;
                }
                id
            }
            ClassLookup::Unknown(u) => {
                unknown.push(u);
                continue;
            }
            ClassLookup::Ambiguous(a) => {
                return Err(format!("Class name \"{a}\" matches multiple classes; use numeric class IDs."));
            }
        };
        if !seen.insert(id) {
            return Err(format!("Duplicate threshold entry resolving to class {id} (\"{k}\")."));
        }
        out.insert(id, conf);
    }
    if require_match_if_unknown && !unknown.is_empty() && out.is_empty() {
        let sample = unknown.iter().take(4).cloned().collect::<Vec<_>>().join(", ");
        return Err(format!("No matching classes found in this model (unknown: {sample})."));
    }
    if !unknown.is_empty() {
        let sample = unknown.iter().take(4).cloned().collect::<Vec<_>>().join(", ");
        let suffix = if unknown.len() > 4 { "…" } else { "" };
        let noun = if unknown.len() == 1 { "class" } else { "classes" };
        warnings.push(format!("Skipped unknown {noun}: {sample}{suffix}."));
    }
    Ok(())
}

/// Converts a `ThresholdCalibration` into `(global_conf, per_class_map)` filtered to `classes`.
/// Returns `(None, empty)` for an empty calibration (`cal.is_empty()`), matching `folo-vision`'s
/// `Predictor::apply_calibrated_thresholds` no-op semantics so empty splits never drop conf to `0.0`.
pub fn calibration_to_overrides(
    cal: &ThresholdCalibration,
    classes: &[String],
) -> (Option<f32>, BTreeMap<u32, f32>) {
    if cal.is_empty() {
        return (None, BTreeMap::new());
    }
    let global = if cal.global.best_conf.is_finite() && (0.0..=1.0).contains(&cal.global.best_conf) {
        Some(round_conf(cal.global.best_conf))
    } else {
        None
    };
    let mut map = BTreeMap::new();
    for c in &cal.per_class {
        if c.best_conf.is_finite()
            && (0.0..=1.0).contains(&c.best_conf)
            && (classes.is_empty() || (c.class as usize) < classes.len())
        {
            map.insert(c.class, round_conf(c.best_conf));
        }
    }
    (global, map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sbox(class: u32, name: &str, score: f32) -> ScoredBox {
        ScoredBox { class, name: name.into(), score, x1: 0., y1: 0., x2: 10., y2: 10. }
    }

    #[test]
    fn effective_conf_filtering_and_min_floor_respect_overrides() {
        let mut overrides = BTreeMap::new();
        overrides.insert(0, 0.70);
        overrides.insert(1, 0.02);
        assert_eq!(effective_conf(0.25, &overrides, 0), 0.70);
        assert_eq!(effective_conf(0.25, &overrides, 1), 0.02);
        assert_eq!(effective_conf(0.25, &overrides, 2), 0.25);
        assert!(thresholds_match(&BTreeMap::new(), 0.05, 0.25, &BTreeMap::new()));
        assert!(!thresholds_match(&BTreeMap::new(), 0.05, 0.25, &overrides));
        assert!(thresholds_match(&overrides, 0.25, 0.25, &overrides));
        assert!(!thresholds_match(&overrides, 0.25, 0.30, &overrides));

        let boxes = vec![
            sbox(0, "circle", 0.50),
            sbox(0, "circle", 0.75),
            sbox(1, "square", 0.03),
            sbox(2, "triangle", 0.30),
        ];
        let kept = filter_detections_with_per_class(&boxes, 0.25, &overrides, None);
        assert_eq!(kept.len(), 3);
        assert_eq!(
            kept.iter().map(|b| (b.class, b.score)).collect::<Vec<_>>(),
            vec![(0, 0.75), (1, 0.03), (2, 0.30)]
        );
    }

    #[test]
    fn model_switch_remaps_by_class_name_and_drops_incompatible_classes() {
        let from = vec!["circle".into(), "square".into(), "triangle".into()];
        let to_reordered = vec!["triangle".into(), "circle".into(), "star".into()];
        let mut overrides = BTreeMap::new();
        overrides.insert(0, 0.40);
        overrides.insert(1, 0.60);
        overrides.insert(2, 0.15);

        let mapped = map_per_class_to_classes(&overrides, &from, &to_reordered);
        assert_eq!(mapped.get(&0), Some(&0.15), "triangle moved 2 -> 0");
        assert_eq!(mapped.get(&1), Some(&0.40), "circle moved 0 -> 1");
        assert_eq!(mapped.get(&2), None, "star has no override; square dropped");
        assert_eq!(mapped.len(), 2);

        let disjoint = vec!["person".into(), "car".into()];
        assert!(map_per_class_to_classes(&overrides, &from, &disjoint).is_empty());
    }

    #[test]
    fn exported_json_deserializes_directly_into_folo_vision_threshold_calibration() {
        let classes = vec!["circle".into(), "square".into()];
        let mut overrides = BTreeMap::new();
        overrides.insert(0, 0.375);
        overrides.insert(1, 0.62);

        let json = export_thresholds_json(0.30, &overrides, &classes).unwrap();
        // Compatible directly with `folo predict --thresholds` (`folo_vision::ThresholdCalibration`).
        let cal: ThresholdCalibration = serde_json::from_str(&json).expect("valid ThresholdCalibration");
        cal.validate().expect("passes folo_vision validation");
        assert!((cal.global.best_conf - 0.30).abs() < 1e-5);
        assert_eq!(cal.per_class.len(), 2);
        assert_eq!(cal.per_class[0].class, 0);
        assert!((cal.per_class[0].best_conf - 0.375).abs() < 1e-5);

        // Round-trips cleanly through Playground's parser as well.
        let parsed = parse_thresholds_json(&json, &classes).expect("round-trip parse");
        assert_eq!(parsed.global_conf, Some(0.30));
        assert_eq!(parsed.per_class, overrides);
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn parser_accepts_calibration_default_conf_val_json_and_flat_maps_and_rejects_bad_inputs() {
        let classes = vec!["circle".into(), "square".into(), "triangle".into()];

        // 1. Wrapped `{ thresholds: ThresholdCalibration }` with cross-model name remapping
        let wrapped = r#"{"thresholds":{"global":{"best_conf":0.42},"per_class":[{"class":9,"name":"square","best_conf":0.55},{"class":2,"name":"ghost","best_conf":0.3}]}}"#;
        let r1 = parse_thresholds_json(wrapped, &classes).unwrap();
        assert_eq!(r1.global_conf, Some(0.42));
        assert_eq!(r1.per_class.get(&1), Some(&0.55));
        assert_eq!(r1.per_class.len(), 1);
        assert_eq!(r1.warnings.len(), 1);

        // 2. `{ default_conf, per_class: { "circle": 0.45, "2": 0.125 } }`
        let def_map = r#"{"default_conf": 0.33, "per_class": {"circle": 0.45, "2": 0.125}}"#;
        let r2 = parse_thresholds_json(def_map, &classes).unwrap();
        assert_eq!(r2.global_conf, Some(0.33));
        assert_eq!(r2.per_class.get(&0), Some(&0.45));
        assert_eq!(r2.per_class.get(&2), Some(&0.125));

        // 3. Flat map by name (case-insensitive) and ID
        let flat = r#"{"CIRCLE": 0.61, "1": 0.22}"#;
        let r3 = parse_thresholds_json(flat, &classes).unwrap();
        assert_eq!(r3.global_conf, None);
        assert_eq!(r3.per_class.get(&0), Some(&0.61));
        assert_eq!(r3.per_class.get(&1), Some(&0.22));

        // 4. `folo val --json` metrics-only record
        let val_json = r#"{"best_f1_conf": 0.38, "per_class": [{"class": 0, "map50": 0.9}]}"#;
        let r4 = parse_thresholds_json(val_json, &classes).unwrap();
        assert_eq!(r4.global_conf, Some(0.38));
        assert!(r4.per_class.is_empty());
        assert_eq!(r4.warnings.len(), 1);

        // 5. Rejects malformed/invalid inputs and empty calibrations (while keeping explicit default_conf: 0 valid)
        assert!(parse_thresholds_json("", &classes).is_err());
        assert!(parse_thresholds_json("{bad", &classes).is_err());
        assert!(
            parse_thresholds_json(
                r#"{"global":{"best_conf":0,"best_f1":0,"precision":0,"recall":0},"macro_f1":0,"per_class":[]}"#,
                &classes
            )
            .is_err()
        );
        let explicit_zero = parse_thresholds_json(r#"{"default_conf": 0}"#, &classes).unwrap();
        assert_eq!(explicit_zero.global_conf, Some(0.0));
        assert!(explicit_zero.per_class.is_empty());
        assert!(parse_thresholds_json(r#"{"0": 1.5}"#, &classes).is_err());
        assert!(parse_thresholds_json(r#"{"-1": 0.5}"#, &classes).is_err());
        assert!(parse_thresholds_json(r#"{"unknown_cls": 0.5}"#, &classes).is_err());
        assert!(
            parse_thresholds_json(
                r#"{"per_class":[{"class":0,"best_conf":0.2},{"class":0,"best_conf":0.4}]}"#,
                &classes
            )
            .is_err()
        );
    }

    #[test]
    fn fmt_threshold_preserves_fine_calibration_precision() {
        assert_eq!(fmt_threshold(0.25), "0.25");
        assert_eq!(fmt_threshold(0.4), "0.40");
        assert_eq!(fmt_threshold(0.375), "0.375");
        assert_eq!(fmt_threshold(0.0025), "0.0025");
    }
}
