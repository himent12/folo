//! External predictions import: parsing JSON/JSONL prediction files, validating every record
//! against the benchmark before upload, chunking large files into bounded pass-by-pass parts,
//! and submitting single-shot or multi-part external entries.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use folo_ui::{AppContext, Context, Focusable, Window};
use serde::Deserialize;

use super::state::ExternalEntryForm;
use crate::api::*;
use crate::app::FoloApp;
use crate::input::TextInput;
use crate::ui::Tone;

/// Payloads over 8 MiB of estimated JSON go as an open entry filled in parts (matches Studio).
pub const PART_BYTES: usize = 8 * 1024 * 1024;

/// Maximum boxes allowed in a single prediction record (matches `folo-server`).
pub const MAX_BOXES_PER_RECORD: usize = 1000;

/// Maximum length of an external entry's leaderboard name.
pub const BENCH_NAME_MAX: usize = 80;

/// Summary of a validated external predictions file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalSummary {
    pub images: usize,
    pub conditions: Vec<String>,
    pub missing_conditions: Vec<String>,
    pub boxes: usize,
    pub has_scores: bool,
    pub record_count: usize,
    pub parts_count: usize,
}

/// Validated predictions ready to send to `/api/bench`.
#[derive(Clone, Debug)]
pub struct CheckedExternal {
    pub predictions: Arc<Vec<BenchPrediction>>,
    pub summary: ExternalSummary,
}

#[derive(Deserialize)]
struct RawDetection {
    class: i64,
    #[serde(default)]
    score: Option<f64>,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

#[derive(Deserialize)]
struct RawPrediction {
    image: String,
    #[serde(default)]
    condition: Option<String>,
    #[serde(default)]
    severity: Option<i64>,
    boxes: Vec<RawDetection>,
}

#[derive(Deserialize)]
struct RawWrapper {
    predictions: Vec<RawPrediction>,
}

/// Raw parsed records before benchmark-specific validation.
pub struct ParsedExternal {
    records: Vec<RawPrediction>,
    pub has_scores: bool,
}

/// Suggests a leaderboard name from a predictions file name (`yolo26n-predictions.json` -> `yolo26n`).
pub fn suggest_external_name(filename: &str) -> String {
    let stem = filename
        .trim()
        .strip_suffix(".jsonl")
        .or_else(|| filename.trim().strip_suffix(".JSONL"))
        .or_else(|| filename.trim().strip_suffix(".json"))
        .or_else(|| filename.trim().strip_suffix(".JSON"))
        .unwrap_or(filename.trim());
    let lower = stem.to_ascii_lowercase();
    let trimmed = if lower.ends_with("-predictions")
        || lower.ends_with("_predictions")
        || lower.ends_with(" predictions")
    {
        &stem[..stem.len() - 12]
    } else if lower.ends_with("-prediction")
        || lower.ends_with("_prediction")
        || lower.ends_with(" prediction")
    {
        &stem[..stem.len() - 11]
    } else {
        stem
    };
    let out = trimmed.trim();
    if out.is_empty() { "external".to_string() } else { out.chars().take(BENCH_NAME_MAX).collect() }
}

/// Parses a JSON (`{ "predictions": [...] }` or `[...]`) or JSONL predictions file.
pub fn parse_predictions_text(text: &str) -> Result<ParsedExternal, String> {
    let trimmed = text.trim_start_matches('\u{feff}').trim();
    if trimmed.is_empty() {
        return Err("The file is empty.".into());
    }
    let records: Vec<RawPrediction> = if trimmed.starts_with('[') {
        serde_json::from_str(trimmed).map_err(|e| format!("Invalid JSON predictions array: {e}"))?
    } else if trimmed.starts_with('{') {
        if let Ok(wrap) = serde_json::from_str::<RawWrapper>(trimmed) {
            wrap.predictions
        } else {
            // Try JSONL: one prediction object per non-empty line.
            let mut out = Vec::new();
            for (line_no, line) in trimmed.lines().enumerate() {
                let l = line.trim();
                if l.is_empty() {
                    continue;
                }
                let rec: RawPrediction = serde_json::from_str(l).map_err(|e| {
                    if line_no == 0 {
                        format!("Expected {{\"predictions\": [...]}}, a JSON array, or JSONL records ({e}).")
                    } else {
                        format!("Line {}: invalid prediction JSON ({e}).", line_no + 1)
                    }
                })?;
                out.push(rec);
            }
            out
        }
    } else {
        return Err(
            "Expected a JSON object with a \"predictions\" array, a JSON array, or JSONL records.".into()
        );
    };
    let has_scores = records.iter().any(|r| r.boxes.iter().any(|b| b.score.is_some()));
    Ok(ParsedExternal { records, has_scores })
}

#[inline]
fn canonical_bits(v: f32) -> u32 {
    if v == 0.0 { 0.0f32.to_bits() } else { v.to_bits() }
}

/// FNV-1a content digest of a record's boxes (matches `crates/folo-server/src/bench/external.rs`).
fn boxes_digest(boxes: &[BenchDetection], scored: bool) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01B3);
        }
    };
    eat(boxes.len() as u64);
    for d in boxes {
        eat(d.class as u64);
        eat(if scored { canonical_bits(d.score) as u64 } else { 0 });
        for v in [d.x1, d.y1, d.x2, d.y2] {
            eat(canonical_bits(v) as u64);
        }
    }
    h
}

/// Conservative JSON byte estimate for one record (matches Studio's `recordBytes`).
pub fn record_bytes(p: &BenchPrediction) -> usize {
    48 + p.image.len() + p.condition.as_ref().map_or(0, |c| c.len() + 16) + p.boxes.len() * 112
}

/// Estimated JSON size for a slice of prediction records.
pub fn estimate_prediction_bytes(preds: &[BenchPrediction]) -> usize {
    preds.iter().map(record_bytes).sum::<usize>() + 32
}

/// Splits `predictions` into pass-by-pass chunks capped at `max_bytes` (unchanged images first,
/// then planned `(condition, severity)` passes in plan order).
pub fn parts_of(
    predictions: &[BenchPrediction],
    plan: &BenchPlan,
    max_bytes: usize,
) -> Vec<Vec<BenchPrediction>> {
    let mut order: HashMap<(String, Option<u8>), usize> = HashMap::new();
    order.insert((String::new(), None), 0);
    for c in &plan.conditions {
        for &s in &c.severities {
            let next = order.len();
            order.entry((c.condition.clone(), Some(s))).or_insert(next);
        }
    }
    let mut passes: HashMap<(String, Option<u8>), Vec<BenchPrediction>> = HashMap::new();
    for p in predictions {
        let key = match (&p.condition, p.severity) {
            (Some(c), s) => (c.clone(), s),
            (None, _) => (String::new(), None),
        };
        passes.entry(key).or_default().push(p.clone());
    }
    let mut keys: Vec<(String, Option<u8>)> = passes.keys().cloned().collect();
    let fallback = order.len();
    keys.sort_by(|a, b| {
        order
            .get(a)
            .copied()
            .unwrap_or(fallback)
            .cmp(&order.get(b).copied().unwrap_or(fallback))
            .then_with(|| a.cmp(b))
    });
    let mut out = Vec::new();
    for k in keys {
        let mut part = Vec::new();
        let mut size = 32usize;
        for p in passes.remove(&k).unwrap_or_default() {
            let b = record_bytes(&p);
            if !part.is_empty() && size + b > max_bytes {
                out.push(part);
                part = Vec::new();
                size = 32;
            }
            part.push(p);
            size += b;
        }
        if !part.is_empty() {
            out.push(part);
        }
    }
    out
}

/// Checks that every box score in `predictions` is finite and in `0.0..=1.0`.
pub fn validate_box_scores(predictions: &[BenchPrediction]) -> Result<(), String> {
    for (i, p) in predictions.iter().enumerate() {
        for (bi, b) in p.boxes.iter().enumerate() {
            if !b.score.is_finite() || !(0.0..=1.0).contains(&b.score) {
                return Err(format!(
                    "Record {}, box {}: score {} must be between 0 and 1.",
                    i + 1,
                    bi + 1,
                    b.score
                ));
            }
        }
    }
    Ok(())
}

/// Validates parsed prediction records against `bench`, `image_ids` (when known), `catalog_ids`,
/// and `scored`. Mirrors `crates/folo-server/src/bench/external.rs`.
pub fn check_predictions(
    parsed: ParsedExternal,
    bench: &Bench,
    image_ids: Option<&[String]>,
    catalog_ids: &[String],
    scored: bool,
) -> Result<CheckedExternal, String> {
    if parsed.records.is_empty() {
        return Err("The file has no prediction records.".into());
    }
    let known: Option<HashSet<&str>> = image_ids.map(|ids| ids.iter().map(String::as_str).collect());
    let planned: HashMap<&str, &[u8]> =
        bench.plan.conditions.iter().map(|c| (c.condition.as_str(), c.severities.as_slice())).collect();
    let classes = bench.classes.len();
    type RecordKey = (String, Option<String>, Option<u8>);
    let mut seen: HashMap<RecordKey, (usize, u64)> = HashMap::new();
    let mut clean: HashSet<String> = HashSet::new();
    let mut under: HashMap<String, HashMap<u8, HashSet<String>>> = HashMap::new();
    let mut out: Vec<BenchPrediction> = Vec::with_capacity(parsed.records.len());
    let mut total_boxes = 0usize;

    for (i, rec) in parsed.records.into_iter().enumerate() {
        let idx = i + 1;
        if rec.image.trim().is_empty() {
            return Err(format!("Record {idx}: \"image\" must be a non-empty image id."));
        }
        if let Some(ref k) = known
            && !k.contains(rec.image.as_str())
        {
            return Err(format!("Record {idx}: this benchmark has no image \"{}\".", rec.image));
        }
        let (cond, sev) = match (rec.condition, rec.severity) {
            (None, None) => (None, None),
            (None, Some(_)) => return Err(format!("Record {idx}: \"severity\" without a \"condition\".")),
            (Some(c), None) => return Err(format!("Record {idx}: condition \"{c}\" needs a \"severity\".")),
            (Some(c), Some(s)) => {
                if !catalog_ids.is_empty() && !catalog_ids.iter().any(|x| x == &c) {
                    return Err(format!("Record {idx}: unknown condition \"{c}\"."));
                }
                let Some(sevs) = planned.get(c.as_str()) else {
                    return Err(format!("Record {idx}: \"{c}\" is not in this benchmark's plan."));
                };
                if !(1..=BENCH_SEVERITIES as i64).contains(&s) {
                    return Err(format!(
                        "Record {idx}: \"severity\" must be a whole number from 1 to {BENCH_SEVERITIES}."
                    ));
                }
                let s_u8 = s as u8;
                if !sevs.contains(&s_u8) {
                    return Err(format!("Record {idx}: \"{c}\" runs at severities {sevs:?}, not {s_u8}."));
                }
                (Some(c), Some(s_u8))
            }
        };
        if rec.boxes.len() > MAX_BOXES_PER_RECORD {
            return Err(format!(
                "Record {idx}: {} boxes (at most {MAX_BOXES_PER_RECORD} per record).",
                rec.boxes.len()
            ));
        }
        let mut boxes = Vec::with_capacity(rec.boxes.len());
        for (bi, b) in rec.boxes.into_iter().enumerate() {
            let bidx = bi + 1;
            if b.class < 0 || (b.class as usize) >= classes {
                return Err(format!(
                    "Record {idx}, box {bidx}: class {} is outside 0..={}.",
                    b.class,
                    classes.saturating_sub(1)
                ));
            }
            let (x1, y1, x2, y2) = (b.x1 as f32, b.y1 as f32, b.x2 as f32, b.y2 as f32);
            if ![x1, y1, x2, y2].iter().all(|v| v.is_finite()) {
                return Err(format!("Record {idx}, box {bidx}: coordinates must be finite numbers."));
            }
            if x2 < x1 || y2 < y1 {
                return Err(format!(
                    "Record {idx}, box {bidx}: x2 ({x2}) or y2 ({y2}) is less than x1 ({x1}) or y1 ({y1}). Boxes use corners (x1, y1, x2, y2), not x, y, width, height."
                ));
            }
            let score_f64 = b.score.unwrap_or(1.0);
            let score = score_f64 as f32;
            if scored && (!score.is_finite() || !(0.0..=1.0).contains(&score)) {
                return Err(format!("Record {idx}, box {bidx}: score must be between 0 and 1."));
            }
            boxes.push(BenchDetection { class: b.class as u32, score, x1, y1, x2, y2 });
        }
        let key = (rec.image.clone(), cond.clone(), sev);
        let hash = boxes_digest(&boxes, scored);
        if let Some(&(first, prev_hash)) = seen.get(&key) {
            if prev_hash != hash {
                return Err(format!(
                    "Record {idx} repeats record {first} (image \"{}\") with different boxes.",
                    rec.image
                ));
            }
            continue;
        }
        seen.insert(key, (idx, hash));
        total_boxes += boxes.len();
        match (&cond, sev) {
            (None, None) => {
                clean.insert(rec.image.clone());
            }
            (Some(c), Some(s)) => {
                under.entry(c.clone()).or_default().entry(s).or_default().insert(rec.image.clone());
            }
            _ => {}
        }
        out.push(BenchPrediction { image: rec.image, condition: cond, severity: sev, boxes });
    }

    let expected_images: Vec<&str> = match image_ids {
        Some(ids) => ids.iter().map(String::as_str).collect(),
        None => clean.iter().map(String::as_str).collect(),
    };
    if expected_images.is_empty() {
        return Err(
            "No unchanged-image records found: every image must be answered without a condition.".into()
        );
    }
    let missing_clean: Vec<&str> =
        expected_images.iter().copied().filter(|id| !clean.contains(*id)).collect();
    if !missing_clean.is_empty() {
        let more = if missing_clean.len() > 1 {
            format!(" (and {} more)", missing_clean.len() - 1)
        } else {
            String::new()
        };
        return Err(format!(
            "Image \"{}\" has no unchanged record{more}: every image must be answered (with empty \"boxes\" when nothing was found).",
            missing_clean[0]
        ));
    }
    for (cond, by_sev) in &under {
        let sevs = planned.get(cond.as_str()).copied().unwrap_or(&[]);
        for &s in sevs {
            let got = by_sev.get(&s);
            if let Some(gap) =
                expected_images.iter().copied().find(|id| !got.is_some_and(|m| m.contains(*id)))
            {
                return Err(format!(
                    "Condition \"{cond}\" is incomplete: image \"{gap}\" at severity {s} is missing. Answer every image at every planned severity or leave the condition out."
                ));
            }
        }
    }
    let conditions: Vec<String> =
        bench.plan.conditions.iter().map(|c| c.condition.clone()).filter(|c| under.contains_key(c)).collect();
    let missing_conditions: Vec<String> = bench
        .plan
        .conditions
        .iter()
        .map(|c| c.condition.clone())
        .filter(|c| !under.contains_key(c))
        .collect();
    let parts_count = if estimate_prediction_bytes(&out) > PART_BYTES {
        parts_of(&out, &bench.plan, PART_BYTES).len()
    } else {
        0
    };
    let summary = ExternalSummary {
        images: expected_images.len(),
        conditions,
        missing_conditions,
        boxes: total_boxes,
        has_scores: parsed.has_scores,
        record_count: out.len(),
        parts_count,
    };
    Ok(CheckedExternal { predictions: Arc::new(out), summary })
}

/// Uploads validated external predictions in one request or pass-by-pass parts. On any failure
/// or cancellation (`cancel`) after creating an entry, stops and deletes it so nothing is left behind.
#[allow(clippy::too_many_arguments)]
pub fn upload_external_predictions(
    api: &dyn Backend,
    bench_id: &str,
    name: String,
    scored: bool,
    plan: &BenchPlan,
    predictions: &[BenchPrediction],
    max_bytes: usize,
    cancel: &AtomicBool,
) -> ApiResult<BenchEntry> {
    let cancelled_err = || ApiError::new("cancelled", "Import cancelled.");
    if cancel.load(Ordering::Relaxed) {
        return Err(cancelled_err());
    }
    if estimate_prediction_bytes(predictions) <= max_bytes {
        let req = StartBenchEntry {
            contestant: BenchContestantSpec::External {
                scored,
                open: false,
                predictions: predictions.to_vec(),
            },
            name: Some(name),
            max_images: None,
            conditions: None,
        };
        let entry = api.start_bench_entry(bench_id, &req)?;
        if cancel.load(Ordering::Relaxed) {
            let _ = api.delete_bench_entry(bench_id, &entry.id);
            return Err(cancelled_err());
        }
        return Ok(entry);
    }
    let parts = parts_of(predictions, plan, max_bytes);
    let open_req = StartBenchEntry {
        contestant: BenchContestantSpec::External { scored, open: true, predictions: Vec::new() },
        name: Some(name),
        max_images: None,
        conditions: None,
    };
    let opened = api.start_bench_entry(bench_id, &open_req)?;
    let eid = opened.id.clone();
    let retryable = |e: &ApiError| e.code == "server_unreachable" || e.status.is_some_and(|s| s >= 500);
    let result: ApiResult<BenchEntry> = (|| {
        for part in parts {
            if cancel.load(Ordering::Relaxed) {
                return Err(cancelled_err());
            }
            let body = BenchPredictions { predictions: part };
            let r = api.send_bench_predictions(bench_id, &eid, &body);
            if let Err(ref e) = r
                && !cancel.load(Ordering::Relaxed)
                && retryable(e)
            {
                api.send_bench_predictions(bench_id, &eid, &body)?;
            } else {
                r?;
            }
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled_err());
        }
        let done = match api.finish_bench_entry(bench_id, &eid) {
            Ok(done) => done,
            Err(e) if !cancel.load(Ordering::Relaxed) && retryable(&e) => {
                match api.finish_bench_entry(bench_id, &eid) {
                    Ok(done) => done,
                    Err(e2) if e2.code == "bench_entry_not_open" => {
                        let cur = api.bench_entry(bench_id, &eid)?;
                        if cur.status == BenchEntryStatus::Done { cur } else { return Err(e2) }
                    }
                    Err(e2) => return Err(e2),
                }
            }
            Err(e) => return Err(e),
        };
        if cancel.load(Ordering::Relaxed) {
            let _ = api.delete_bench_entry(bench_id, &eid);
            return Err(cancelled_err());
        }
        Ok(done)
    })();
    if result.is_err() {
        let _ = api.stop_bench_entry(bench_id, &eid);
        let _ = api.delete_bench_entry(bench_id, &eid);
    }
    result
}

impl FoloApp {
    /// Opens the inline External Predictions import form on `bench_id`'s leaderboard.
    pub fn open_external_entry(&mut self, bench_id: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.bench.form_seq += 1;
        let seq = self.bench.form_seq;
        let file_path = cx.new(|cx| TextInput::new("Path to predictions .json or .jsonl", cx));
        let name = cx.new(|cx| TextInput::new("e.g. yolo26n", cx));
        let fh = file_path.read(cx).focus_handle(cx);
        self.bench.external = Some(ExternalEntryForm {
            bench: bench_id.to_string(),
            seq,
            file_path,
            checked_path: None,
            name,
            suggested_name: None,
            scored: true,
            scored_edited: false,
            reading: false,
            file_name: None,
            file_bytes: 0,
            check: None,
            submitting: false,
            cancel: None,
            error: None,
            show_how: false,
        });
        window.focus(&fh);
        cx.notify();
    }

    /// Opens the native file picker for a predictions JSON/JSONL file.
    pub fn pick_external_file(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(seq) = self.bench.external.as_ref().map(|f| f.seq) else { return };
        let handle = window.window_handle();
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Choose predictions JSON file".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Some(p) = crate::app::picked_path(&this, rx.await, cx) {
                handle
                    .update(cx, |_, _, cx| {
                        this.update(cx, |this, cx| {
                            if this.bench.external.as_ref().is_some_and(|f| f.seq == seq) {
                                let text = p.to_string_lossy().to_string();
                                if let Some(f) = this.bench.external.as_ref() {
                                    f.file_path.update(cx, |i, cx| i.set_text(&text, cx));
                                }
                                this.load_external_file_path(p, cx);
                            }
                        })
                    })
                    .ok();
            }
        })
        .detach();
    }

    /// Reads, parses, and validates the predictions file at `path` off the UI thread.
    pub fn load_external_file_path(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let Some(f) = self.bench.external.as_mut() else { return };
        if f.submitting {
            return;
        }
        let bid = f.bench.clone();
        let Some(bench) = self.bench.list.ready().and_then(|l| l.iter().find(|b| b.id == bid)).cloned()
        else {
            return;
        };
        self.bench.form_seq += 1;
        let seq = self.bench.form_seq;
        f.seq = seq;
        f.reading = true;
        f.checked_path = Some(path.to_string_lossy().trim().to_string());
        f.check = None;
        f.error = None;
        let fname = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        f.file_name = Some(fname.clone());
        let scored = f.scored;
        let scored_edited = f.scored_edited;
        let catalog_ids: Vec<String> = self
            .bench
            .catalog
            .ready()
            .map(|c| c.conditions.iter().map(|x| x.id.clone()).collect())
            .unwrap_or_else(|| bench.plan.conditions.iter().map(|c| c.condition.clone()).collect());
        cx.notify();
        self.fetch(
            cx,
            move |api| {
                let bytes = std::fs::read(&path).map_err(|e| {
                    ApiError::new("read_failed", format!("Could not read {}: {e}", path.display()))
                })?;
                let len = bytes.len() as u64;
                let text = String::from_utf8(bytes)
                    .map_err(|_| ApiError::new("read_failed", "The predictions file is not valid UTF-8."))?;
                // Fetch benchmark image IDs for full membership/completeness checking (best-effort).
                let mut ids = Vec::new();
                let mut ok_ids = true;
                let mut off = 0u32;
                while (ids.len() as u64) < bench.images {
                    match api.bench_images(&bench.id, off, 1000, None) {
                        Ok(page) => {
                            if page.images.is_empty() {
                                break;
                            }
                            let got = page.images.len() as u32;
                            ids.extend(page.images.into_iter().map(|im| im.id));
                            off = off.saturating_add(got);
                        }
                        Err(_) => {
                            ok_ids = false;
                            break;
                        }
                    }
                }
                let parsed = match parse_predictions_text(&text) {
                    Ok(p) => p,
                    Err(msg) => return Ok((len, Err(msg))),
                };
                // Respect an explicit user toggle; before the user touches Confidences, auto-detect from file.
                let effective_scored = if scored_edited { scored } else { parsed.has_scores };
                let checked = check_predictions(
                    parsed,
                    &bench,
                    ok_ids.then_some(ids.as_slice()),
                    &catalog_ids,
                    effective_scored,
                );
                Ok((len, checked))
            },
            move |this, r, cx| {
                let Some(f) = this.bench.external.as_mut().filter(|f| f.seq == seq) else { return };
                f.reading = false;
                match r {
                    Ok((len, checked)) => {
                        f.file_bytes = len;
                        if let Ok(ref ok) = checked {
                            if !f.scored_edited {
                                f.scored = ok.summary.has_scores;
                            }
                            let cur_name = f.name.read(cx).text().trim().to_string();
                            let can_replace =
                                cur_name.is_empty() || f.suggested_name.as_deref() == Some(cur_name.as_str());
                            if can_replace && !fname.is_empty() {
                                let suggested = suggest_external_name(&fname);
                                f.name.update(cx, |i, cx| i.set_text(&suggested, cx));
                                f.suggested_name = Some(suggested);
                            }
                        }
                        f.check = Some(checked);
                    }
                    Err(e) => {
                        f.check = Some(Err(e.message));
                    }
                }
            },
        );
    }

    /// Reads the currently typed file path in the External Predictions form and validates it.
    pub fn check_typed_external_file(&mut self, cx: &mut Context<Self>) {
        let Some(path) = self.bench.external.as_ref().map(|f| f.file_path.read(cx).text().trim().to_string())
        else {
            return;
        };
        if path.is_empty() {
            if let Some(f) = self.bench.external.as_mut() {
                f.checked_path = None;
                f.check = Some(Err("Choose or paste a path to a predictions JSON/JSONL file.".into()));
            }
            cx.notify();
            return;
        }
        self.load_external_file_path(PathBuf::from(path), cx);
    }

    /// Updates whether the external entry uses confidences; when enabling confidences after an
    /// already-checked file, re-reads and re-validates the original file so duplicate records whose
    /// scores differed or out-of-range scores cannot slip past.
    pub fn set_external_scored(&mut self, scored: bool, cx: &mut Context<Self>) {
        let should_recheck = {
            let Some(f) = self.bench.external.as_mut() else { return };
            if f.submitting {
                return;
            }
            let was_scored = f.scored;
            f.scored = scored;
            f.scored_edited = true;
            f.error = None;
            let cur_path = f.file_path.read(cx).text().trim().to_string();
            let same_checked =
                !cur_path.is_empty() && f.checked_path.as_deref().map(str::trim) == Some(cur_path.as_str());
            if scored && !was_scored && same_checked {
                true
            } else {
                if scored
                    && let Some(Ok(checked)) = f.check.as_ref()
                    && let Err(msg) = validate_box_scores(&checked.predictions)
                {
                    f.error = Some(ApiError::new("invalid_body", msg));
                }
                false
            }
        };
        if should_recheck {
            self.check_typed_external_file(cx);
        } else {
            cx.notify();
        }
    }

    /// Submits the validated External Predictions entry to `/api/bench`.
    pub fn submit_external_entry(&mut self, cx: &mut Context<Self>) {
        let Some(f) = self.bench.external.as_mut() else { return };
        if f.submitting || f.reading {
            return;
        }
        let cur_path = f.file_path.read(cx).text().trim().to_string();
        if f.checked_path.as_deref().map(str::trim) != Some(cur_path.as_str()) {
            f.error = Some(ApiError::new(
                "invalid_body",
                "The file path changed — click Check file before scoring.",
            ));
            cx.notify();
            return;
        }
        let name = f.name.read(cx).text().trim().to_string();
        if name.is_empty() {
            f.error = Some(ApiError::new("invalid_body", "Enter a name for the leaderboard."));
            cx.notify();
            return;
        }
        if name.chars().count() > BENCH_NAME_MAX {
            f.error = Some(ApiError::new(
                "invalid_body",
                format!("Name must be at most {BENCH_NAME_MAX} characters."),
            ));
            cx.notify();
            return;
        }
        let Some(Ok(checked)) = f.check.as_ref() else {
            f.error = Some(ApiError::new("invalid_body", "Choose and validate a predictions file first."));
            cx.notify();
            return;
        };
        if f.scored
            && let Err(msg) = validate_box_scores(&checked.predictions)
        {
            f.error = Some(ApiError::new("invalid_body", msg));
            cx.notify();
            return;
        }
        let bid = f.bench.clone();
        let Some(bench) = self.bench.list.ready().and_then(|l| l.iter().find(|b| b.id == bid)).cloned()
        else {
            return;
        };
        let seq = f.seq;
        let scored = f.scored;
        let preds = checked.predictions.clone();
        let plan = bench.plan.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_bg = cancel.clone();
        f.submitting = true;
        f.cancel = Some(cancel);
        f.error = None;
        let since = self.bench.ev_seq;
        let bid_req = bid.clone();
        let name_req = name.clone();
        self.fetch(
            cx,
            move |api| {
                upload_external_predictions(
                    api, &bid_req, name_req, scored, &plan, &preds, PART_BYTES, &cancel_bg,
                )
            },
            move |this, r, cx| {
                if let Some(f) = this.bench.external.as_mut().filter(|f| f.seq == seq) {
                    f.submitting = false;
                    f.cancel = None;
                    match r {
                        Ok(entry) if entry.status == BenchEntryStatus::Done => {
                            let label = entry.name.clone();
                            let detail = entry.score.as_ref().map(|s| format!("Score {:.1}", s.total).into());
                            this.bench.external = None;
                            this.apply_entry(&bid, entry, since);
                            this.load_bench_entries(&bid, cx);
                            this.refresh_benches(cx);
                            this.toast(Tone::Success, format!("{label} scored"), detail, cx);
                        }
                        Ok(entry) => {
                            let msg = entry.error.clone().unwrap_or_else(|| {
                                format!(
                                    "Entry ended with status {} instead of scoring.",
                                    entry.status.label()
                                )
                            });
                            let mut err = ApiError::new("bench_entry_failed", msg);
                            if let Some(hint) = entry.hint.clone() {
                                err = err.with_hint(hint);
                            }
                            f.error = Some(err);
                            this.apply_entry(&bid, entry, since);
                            this.load_bench_entries(&bid, cx);
                            this.refresh_benches(cx);
                        }
                        Err(e) => {
                            f.error = Some(e);
                        }
                    }
                }
            },
        );
        cx.notify();
    }
}
