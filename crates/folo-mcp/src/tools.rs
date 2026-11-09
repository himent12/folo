//! The tools of docs/MCP.md: their schemas (generated from the Rust types with schemars) and their dispatch onto
//! the [`Store`](folo_server::Store).

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use folo_api::IssuePage;
use folo_api::{
    Arch, Dataset, ExportResult, ExportSpec, IssueSeverity, LogLevel, LogLine, Model, OnnxSwitches,
    PredictResponse, QueueRequest, RegisterDataset, Run, RunMetrics, SyntheticDataset, ValReport, ValRequest,
};
use folo_server::store::{ImageQuery, IssueQuery, PredictInput, PredictSource, QueueMove};
use rmcp::handler::server::common::{schema_for_input, schema_for_output};
use rmcp::model::{
    ContentBlock, JsonObject, ProgressNotificationParam, ProgressToken, Resource, Tool, ToolAnnotations,
};
use rmcp::service::{Peer, RoleServer};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::errors::{Error, invalid_arguments, read_only};
use crate::paths::{MAX_BATCH, too_many_inputs};
use crate::server::Inner;
use crate::types::*;

/// No tool result is larger than this many bytes of JSON (docs/MCP.md "Paging and size limits").
pub const MAX_RESULT_BYTES: usize = 32 * 1024;
/// List pages stop adding items once they reach this many bytes, leaving room for the envelope.
const PAGE_BUDGET: usize = 30 * 1024;
/// Largest image accepted by `predict` (as for `POST /api/predict`).
const MAX_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
/// At most this many `predict_batch` failures are listed inline.
const MAX_FAILURES: usize = 50;
/// `get_dataset` lists at most this many issues.
const MAX_ISSUES: usize = 50;

/// One tool of the table in docs/MCP.md.
pub struct ToolDef {
    /// Tool name.
    pub name: &'static str,
    /// What it does (shown to the agent).
    pub description: &'static str,
    /// Changes state: hidden and rejected with `--read-only`.
    pub mutating: bool,
    input: fn() -> Arc<JsonObject>,
    output: fn() -> Arc<JsonObject>,
}

fn input<T: JsonSchema + 'static>() -> Arc<JsonObject> {
    let mut s = (*schema_for_input::<T>().expect("tool inputs are objects")).clone();
    compact(&mut s, false);
    close_objects(&mut s);
    s.entry("additionalProperties").or_insert(Value::Bool(false)); // `{}` has no `properties`
    Arc::new(s)
}

/// Marks every object schema with `properties` (the root and the `$defs`) `additionalProperties: false`: tools
/// reject arguments they do not know ([`unknown_argument`]) instead of silently ignoring a typo.
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
fn unknown_argument(tool: &str, schema: &JsonObject, args: &JsonObject) -> Option<Error> {
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

/// `train`'s input: TrainConfig plus `resume_run_id`, where `dataset_id` is required unless `resume_run_id` is
/// given (a resume may leave every other field out).
fn train_input() -> Arc<JsonObject> {
    let mut s = (*input::<TrainArgs>()).clone();
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

fn output<T: JsonSchema + 'static>() -> Arc<JsonObject> {
    let mut s = (*schema_for_output::<T>()).clone();
    compact(&mut s, true);
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
                if o.get("minimum") == Some(&json!(0)) && o.get("type") == Some(&json!("integer")) {
                    o.remove("minimum");
                }
                if output {
                    o.remove("default");
                    if in_defs {
                        o.remove("description");
                    }
                    let consts: Option<Vec<Value>> = o.get("oneOf").and_then(Value::as_array).and_then(|a| {
                        a.iter().map(|b| b.get("const").filter(|c| c.is_string()).cloned()).collect()
                    });
                    if let Some(values) = consts {
                        o.remove("oneOf");
                        o.insert("type".into(), json!("string"));
                        o.insert("enum".into(), Value::Array(values));
                    }
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

/// Every tool, in the order of docs/MCP.md.
pub const TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "folo_info",
        description: "FOLO version, compute devices (CPU, GPUs), data dir, read-only flag and the model scales \
                      (params / GFLOPs). Call this first to pick a device, scale and image size.",
        mutating: false,
        input: input::<NoArgs>,
        output: output::<FoloInfo>,
    },
    ToolDef {
        name: "list_datasets",
        description: "Lists registered datasets (summaries: id, name, format, classes, splits). Paged.",
        mutating: false,
        input: input::<PageArgs>,
        output: output::<Page<DatasetSummary>>,
    },
    ToolDef {
        name: "get_dataset",
        description: "One dataset with statistics (class counts, box sizes, boxes per image), issue counts by \
                      severity and kind, and its errors then warnings (first 50; infos are only counted).",
        mutating: false,
        input: input::<DatasetIdArgs>,
        output: output::<DatasetDetail>,
    },
    ToolDef {
        name: "list_dataset_issues",
        description: "Lists a dataset's issues (the `folo dataset check` findings: kind, severity, file, line, \
                      message, suggested fix) in dataset order, optionally one severity or kind. Paged.",
        mutating: false,
        input: input::<ListIssuesArgs>,
        output: output::<IssuesPage>,
    },
    ToolDef {
        name: "list_dataset_images",
        description: "Lists a dataset's images (optionally one split or class, optionally with boxes). Paged. \
                      Read folo://datasets/{dataset_id}/images/{image_id} (a 512 px JPEG; ?size=original: the file) to look at an image.",
        mutating: false,
        input: input::<ListImagesArgs>,
        output: output::<Page<ImageSummary>>,
    },
    ToolDef {
        name: "register_dataset",
        description: "Registers a YOLO or COCO dataset folder on disk (never moves or changes files) and returns \
                      it with statistics.",
        mutating: true,
        input: input::<RegisterArgs>,
        output: output::<Dataset>,
    },
    ToolDef {
        name: "create_synthetic_dataset",
        description: "Generates a synthetic shapes dataset (circles, squares, triangles, stars on textured \
                      backgrounds) for quick experiments. Reports progress.",
        mutating: true,
        input: input::<SyntheticArgs>,
        output: output::<Dataset>,
    },
    ToolDef {
        name: "list_models",
        description: "Lists trained models (summaries: id, name, arch, imgsz, classes, metrics). Paged.",
        mutating: false,
        input: input::<PageArgs>,
        output: output::<Page<ModelSummary>>,
    },
    ToolDef {
        name: "train",
        description: "Starts a training run (TrainConfig of docs/API.md; only dataset_id is required) and returns \
                      at once. Follow it with run_status or by subscribing to folo://runs/{run_id}.",
        mutating: true,
        input: train_input,
        output: output::<TrainResult>,
    },
    ToolDef {
        name: "run_status",
        description: "A run's status, progress, best epoch, latest step and epoch metrics, and its last log lines.",
        mutating: false,
        input: input::<RunStatusArgs>,
        output: output::<RunStatusResult>,
    },
    ToolDef {
        name: "list_runs",
        description: "Lists training runs, newest first, optionally filtered by status. Paged.",
        mutating: false,
        input: input::<ListRunsArgs>,
        output: output::<Page<Run>>,
    },
    ToolDef {
        name: "stop_run",
        description: "Stops a running run; its last checkpoint is kept.",
        mutating: true,
        input: input::<RunIdArgs>,
        output: output::<Run>,
    },
    ToolDef {
        name: "reorder_run",
        description: "Moves a run waiting in the training queue: action top / up / down / bottom, or a 1-based \
                      position; action start starts it now when nothing is training, else next (never preempting \
                      the active run). Returns its status and new queue_position.",
        mutating: true,
        input: input::<ReorderRunArgs>,
        output: output::<ReorderResult>,
    },
    ToolDef {
        name: "val",
        description: "Evaluates a model on a dataset split: mAP@0.5, mAP@0.5:0.95, precision, recall, per class. \
                      Reports progress.",
        mutating: false,
        input: input::<ValArgs>,
        output: output::<ValReport>,
    },
    ToolDef {
        name: "predict",
        description: "Detects objects in one image (a file path, base64 bytes, or a dataset image). Boxes are \
                      pixel xyxy, highest score first.",
        mutating: false,
        input: input::<PredictArgs>,
        output: output::<PredictResponse>,
    },
    ToolDef {
        name: "predict_batch",
        description: "Detects objects in up to 256 images (paths or a glob). Returns summaries; the per-image \
                      boxes are in the linked folo://predictions/{batch_id} resource. Reports progress.",
        mutating: false,
        input: input::<PredictBatchArgs>,
        output: output::<PredictBatchResult>,
    },
    ToolDef {
        name: "export",
        description: "Exports a model (folo-f16 or onnx) to a new file under the data dir.",
        mutating: true,
        input: input::<ExportArgs>,
        output: output::<ExportResult>,
    },
    ToolDef {
        name: "compare_runs",
        description: "Compares 2 to 8 runs: best metrics, final losses, epochs, duration, and the TrainConfig \
                      fields that differ.",
        mutating: false,
        input: input::<CompareArgs>,
        output: output::<CompareResult>,
    },
];

impl ToolDef {
    /// The MCP tool definition.
    pub fn tool(&self) -> Tool {
        let annotations = ToolAnnotations::new()
            .read_only(!self.mutating)
            .destructive(false)
            .idempotent(!self.mutating || self.name == "stop_run")
            .open_world(false);
        Tool::new(self.name, self.description, (self.input)())
            .with_raw_output_schema((self.output)())
            .with_annotations(annotations)
    }
}

/// Looks a tool up by name.
pub fn find(name: &str) -> Option<&'static ToolDef> {
    TOOLS.iter().find(|t| t.name == name)
}

/// The tools a server lists (`read_only` hides the mutating ones).
pub fn listed(read_only: bool) -> impl Iterator<Item = &'static ToolDef> {
    TOOLS.iter().filter(move |t| !(read_only && t.mutating))
}

/// The `tools/list` entries as JSON (for `folo describe`).
pub fn describe() -> Value {
    Value::Array(
        TOOLS
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "mutating": t.mutating,
                    "inputSchema": Value::Object((*(t.input)()).clone()),
                    "outputSchema": Value::Object((*(t.output)()).clone()),
                })
            })
            .collect(),
    )
}

/// What a successful tool call returns.
pub struct Output {
    /// `structuredContent`.
    pub value: Value,
    /// The one-line text item.
    pub summary: String,
    /// Extra content (resource links).
    pub links: Vec<ContentBlock>,
}

/// Longest one-line summary, in characters.
const MAX_SUMMARY_CHARS: usize = 300;
/// Longest name (dataset, run) accepted from tool arguments, in characters.
const MAX_NAME_CHARS: usize = 128;
/// List summaries carry at most this many class names (`num_classes` counts all).
const SUMMARY_CLASSES: usize = 100;

/// `s` cut to at most `max` characters (with an ellipsis when cut).
pub(crate) fn short(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        None => s.to_string(),
        Some((i, _)) => format!("{}…", &s[..i]),
    }
}

/// A successful result. Every tool result goes through here, so none exceeds [`MAX_RESULT_BYTES`]: callers shrink
/// what has a bounded form, and anything still too large is the `result_too_large` tool error.
fn out<T: Serialize>(value: &T, summary: impl Into<String>) -> Result<Output, Error> {
    let value =
        folo_api::json::to_value(value).map_err(|e| Error::internal(format!("serializing a result: {e}")))?;
    let n = json_len(&value);
    if n > MAX_RESULT_BYTES {
        return Err(Error::new(
            http::StatusCode::PAYLOAD_TOO_LARGE,
            "result_too_large",
            format!("the result would be {n} bytes of JSON, over the {MAX_RESULT_BYTES}-byte limit"),
        )
        .hint(
            "read the matching folo:// resource instead (e.g. folo://datasets/{id}), or narrow the request",
        ));
    }
    Ok(Output { value, summary: short(&summary.into(), MAX_SUMMARY_CHARS), links: Vec::new() })
}

fn check_name(what: &str, s: &str) -> Result<(), Error> {
    if s.chars().count() > MAX_NAME_CHARS {
        return Err(invalid_arguments(format!("{what} must be at most {MAX_NAME_CHARS} characters")));
    }
    Ok(())
}

/// The first [`SUMMARY_CLASSES`] class names, each cut to 64 characters.
fn summary_classes(classes: &[String]) -> Vec<String> {
    classes.iter().take(SUMMARY_CLASSES).map(|c| short(c, 64)).collect()
}

/// `Store::list_issues` for `q`, also against a `folo serve` from before severity filters and counts (proxy
/// mode): it ignores `severity` and leaves `counts` out, so the page is rebuilt from all its issues.
fn list_issues_compat(
    fetch: impl Fn(&IssueQuery) -> Result<IssuePage, Error>,
    q: &IssueQuery,
) -> Result<IssuePage, Error> {
    let counted = |p: &IssuePage| p.counts.error + p.counts.warning + p.counts.info > 0;
    let page = fetch(q)?;
    if counted(&page) {
        return Ok(page);
    }
    // No counts: a dataset without issues, or an older server. A filtered answer cannot tell them apart (an old
    // server honours `kind`), so ask for everything once.
    let filtered = q.kind.is_some() || q.severity.is_some();
    let probe =
        if filtered { fetch(&IssueQuery { limit: 1, ..IssueQuery::default() })? } else { page.clone() };
    if probe.total == 0 || counted(&probe) {
        return Ok(page);
    }
    let mut all = Vec::new();
    loop {
        let q = IssueQuery { offset: all.len() as u64, limit: 1000, ..IssueQuery::default() };
        let p = fetch(&q)?;
        let n = p.issues.len();
        all.extend(p.issues);
        if n == 0 || all.len() as u64 >= p.total {
            break;
        }
    }
    Ok(q.page(all.iter()))
}

/// An issue with its file, message and fix cut to 300 characters, so a page always makes progress.
fn short_issue(mut i: folo_api::DatasetIssue) -> folo_api::DatasetIssue {
    i.file = short(&i.file, 300);
    i.message = short(&i.message, 300);
    i.fix = i.fix.map(|f| short(&f, 300));
    i
}

/// A run as a list item: names and the error cut so every item is small.
fn run_summary(mut r: Run) -> Run {
    r.name = short(&r.name, MAX_NAME_CHARS);
    r.config.name = short(&r.config.name, MAX_NAME_CHARS);
    r.error = r.error.map(|e| short(&e, 500));
    r
}

/// Store errors keep their codes, except the HTTP layer's names for bad input (`invalid_body`, `invalid_query`,
/// `invalid_path`, `bad_request`), which a tool reports as `invalid_arguments` (one vocabulary with the CLI:
/// `folo_api::codes`).
fn mcp_code(mut e: Error) -> Error {
    if matches!(e.body.code.as_str(), "invalid_body" | "invalid_query" | "invalid_path" | "bad_request") {
        e.body.code = "invalid_arguments".into();
        e.body.hint.get_or_insert_with(|| "check the tool's inputSchema (tools/list)".into());
    }
    e
}

fn parse_args<T: DeserializeOwned>(args: JsonObject) -> Result<T, Error> {
    serde_json::from_value(Value::Object(args))
        .map_err(|e| invalid_arguments(format!("invalid arguments: {e}")))
}

fn json_len<T: Serialize>(v: &T) -> usize {
    serde_json::to_vec(v).map_or(usize::MAX, |b| b.len())
}

fn check_unit(name: &str, v: f32) -> Result<(), Error> {
    if (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(invalid_arguments(format!("{name} must be in 0..=1, got {v}")))
    }
}

/// `(offset, limit)` of a page request.
fn page_window(cursor: Option<&str>, limit: Option<u32>) -> Result<(u64, u64), Error> {
    let offset = match cursor {
        None => 0,
        Some(c) => c.strip_prefix("o").and_then(|n| n.parse::<u64>().ok()).ok_or_else(|| {
            invalid_arguments(format!(
                "invalid cursor `{}`; pass a next_cursor from a previous page",
                short(c, 40)
            ))
        })?,
    };
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    if !(1..=MAX_LIMIT).contains(&limit) {
        return Err(invalid_arguments(format!("limit must be in 1..={MAX_LIMIT}, got {limit}")));
    }
    Ok((offset, limit as u64))
}

fn cursor_at(offset: u64) -> String {
    format!("o{offset}")
}

/// A page of `items` (already starting at `offset`), shortened to fit [`PAGE_BUDGET`].
fn fit_page<T: Serialize>(
    items: impl IntoIterator<Item = T>,
    offset: u64,
    limit: u64,
    total: u64,
) -> Page<T> {
    let mut page = Vec::new();
    let mut bytes = 0;
    for item in items.into_iter().take(limit as usize) {
        let n = json_len(&item) + 1;
        if !page.is_empty() && bytes + n > PAGE_BUDGET {
            break;
        }
        bytes += n;
        page.push(item);
    }
    let end = offset + page.len() as u64;
    Page { next_cursor: (end < total).then(|| cursor_at(end)), total, items: page }
}

/// A page of an in-memory list.
fn page_of<T: Serialize>(all: Vec<T>, cursor: Option<&str>, limit: Option<u32>) -> Result<Page<T>, Error> {
    let (offset, limit) = page_window(cursor, limit)?;
    let total = all.len() as u64;
    Ok(fit_page(all.into_iter().skip(offset as usize), offset, limit, total))
}

/// Sends `notifications/progress` for one request, if it asked for progress: monotonically increasing values, at
/// most ~4 per second (the final value is always sent).
pub struct Progress {
    target: Option<(Peer<RoleServer>, ProgressToken)>,
    last_sent: Option<Instant>,
    last_value: f64,
}

impl Progress {
    /// Progress for a request with `token` (none without one).
    pub fn new(peer: Peer<RoleServer>, token: Option<ProgressToken>) -> Progress {
        Progress { target: token.map(|t| (peer, t)), last_sent: None, last_value: f64::NEG_INFINITY }
    }

    /// Reports `done` of `total`. `last` forces the send (the final update before the result).
    pub async fn report(&mut self, done: f64, total: Option<f64>, last: bool) {
        let Some((peer, token)) = &self.target else { return };
        if done <= self.last_value {
            return;
        }
        if !last && self.last_sent.is_some_and(|t| t.elapsed() < Duration::from_millis(250)) {
            return;
        }
        let mut p = ProgressNotificationParam::new(token.clone(), done);
        if let Some(t) = total {
            p = p.with_total(t);
        }
        if peer.notify_progress(p).await.is_ok() {
            self.last_sent = Some(Instant::now());
            self.last_value = done;
        }
    }
}

impl Inner {
    /// Dispatches a tool call. `Err` is a tool error (an `isError` result), never a protocol error.
    pub async fn call(
        &self,
        def: &ToolDef,
        args: JsonObject,
        progress: &mut Progress,
    ) -> Result<Output, Error> {
        if self.opts.read_only && def.mutating {
            return Err(read_only(def.name));
        }
        if let Some(e) = unknown_argument(def.name, &(def.input)(), &args) {
            return Err(e);
        }
        let res = match def.name {
            "folo_info" => self.folo_info(parse_args(args)?).await,
            "list_datasets" => self.list_datasets(parse_args(args)?).await,
            "get_dataset" => self.get_dataset(parse_args(args)?).await,
            "list_dataset_issues" => self.list_dataset_issues(parse_args(args)?).await,
            "list_dataset_images" => self.list_dataset_images(parse_args(args)?).await,
            "register_dataset" => self.register_dataset(parse_args(args)?).await,
            "create_synthetic_dataset" => self.create_synthetic(parse_args(args)?, progress).await,
            "list_models" => self.list_models(parse_args(args)?).await,
            "train" => self.train(args).await,
            "run_status" => self.run_status(parse_args(args)?).await,
            "list_runs" => self.list_runs(parse_args(args)?).await,
            "stop_run" => self.stop_run(parse_args(args)?).await,
            "reorder_run" => self.reorder_run(parse_args(args)?).await,
            "val" => self.val(parse_args(args)?, progress).await,
            "predict" => self.predict(parse_args(args)?).await,
            "predict_batch" => self.predict_batch(parse_args(args)?, progress).await,
            "export" => self.export(parse_args(args)?).await,
            "compare_runs" => self.compare_runs(parse_args(args)?).await,
            other => Err(Error::internal(format!("tool `{other}` has no handler"))),
        };
        res.map_err(|e| mcp_code(crate::confine::map_denied(e)))
    }

    /// With `--confine`, fails unless dataset `id` reads only files inside the data dir.
    async fn ensure_dataset(&self, id: &str) -> Result<(), Error> {
        if !self.confinement.active() {
            return Ok(());
        }
        let (c, id) = (self.confinement.clone(), id.to_string());
        self.blocking(move |s| c.dataset(s, &id)).await
    }

    /// Resolves an input path on the blocking pool (canonicalisation touches the disk).
    async fn resolve(&self, input: String) -> Result<std::path::PathBuf, Error> {
        let paths = self.paths.clone();
        tokio::task::spawn_blocking(move || paths.resolve(&input)).await.map_err(join)?
    }

    async fn folo_info(&self, _: NoArgs) -> Result<Output, Error> {
        let info = self.info().await?;
        let gpus = info.devices.gpus.len();
        let amp = match &info.capabilities {
            Some(c) if c.amp.available => ", amp available",
            Some(_) => ", no amp",
            None => "",
        };
        let summary = format!(
            "FOLO {} ({} mode{}), {} GPU{}, {} CPU threads{amp}",
            info.version,
            info.mode,
            if info.read_only { ", read-only" } else { "" },
            gpus,
            if gpus == 1 { "" } else { "s" },
            info.devices.cpu.threads
        );
        out(&info, summary)
    }

    async fn list_datasets(&self, a: PageArgs) -> Result<Output, Error> {
        let c = self.confinement.clone();
        // With --confine, datasets that read outside the data dir are not listed.
        let all: Vec<Dataset> = self
            .blocking(move |s| {
                Ok(s.list_datasets()?.into_iter().filter(|d| c.dataset(s, &d.id).is_ok()).collect())
            })
            .await?;
        let items: Vec<DatasetSummary> = all
            .into_iter()
            .map(|d: Dataset| DatasetSummary {
                id: d.id,
                name: short(&d.name, MAX_NAME_CHARS),
                format: d.format,
                num_classes: d.classes.len(),
                classes: summary_classes(&d.classes),
                splits: d.splits,
            })
            .collect();
        let page = page_of(items, a.cursor.as_deref(), a.limit)?;
        let summary = format!("{} of {} datasets", page.items.len(), page.total);
        out(&page, summary)
    }

    async fn get_dataset(&self, a: DatasetIdArgs) -> Result<Output, Error> {
        self.ensure_dataset(&a.dataset_id).await?;
        let id = a.dataset_id.clone();
        let (dataset, issues, counts) = self
            .blocking(move |s| {
                let dataset = s.get_dataset(&id)?;
                // Errors first, then warnings; infos (background images) are only counted.
                let mut issues = Vec::new();
                let mut counts = None;
                for severity in [IssueSeverity::Error, IssueSeverity::Warning] {
                    let limit = (MAX_ISSUES - issues.len()) as u64;
                    let q = IssueQuery { severity: Some(severity), limit, ..IssueQuery::default() };
                    let page = list_issues_compat(|q| s.list_issues(&id, q), &q)?;
                    issues.extend(
                        page.issues.into_iter().filter(|i| i.severity == severity).take(limit as usize),
                    );
                    counts.get_or_insert(page.counts);
                }
                Ok((dataset, issues, counts.unwrap_or_default()))
            })
            .await?;
        let mut detail = DatasetDetail {
            dataset,
            issues: issues.into_iter().map(short_issue).collect(),
            issues_total: counts.error + counts.warning + counts.info,
            issue_counts: counts,
        };
        while json_len(&detail) > MAX_RESULT_BYTES && detail.issues.pop().is_some() {}
        let s = &detail.dataset;
        let c = &detail.issue_counts;
        let summary = format!(
            "{} ({}): {} classes, {} train / {} val / {} test images; {} errors, {} warnings, {} infos",
            s.name,
            s.id,
            s.classes.len(),
            s.splits.train,
            s.splits.val,
            s.splits.test,
            c.error,
            c.warning,
            c.info
        );
        out(&detail, summary)
    }

    async fn list_dataset_issues(&self, a: ListIssuesArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        if let Some(k) = &a.kind
            && !folo_api::ISSUE_KINDS.contains(&k.as_str())
        {
            return Err(invalid_arguments(format!(
                "unknown issue kind `{}`; kind is one of {}",
                short(k, 40),
                folo_api::ISSUE_KINDS.join(", ")
            )));
        }
        self.ensure_dataset(&a.dataset_id).await?;
        let q = IssueQuery { kind: a.kind.clone(), severity: a.severity, offset, limit };
        let id = a.dataset_id.clone();
        let page = self.blocking(move |s| list_issues_compat(|q| s.list_issues(&id, q), &q)).await?;
        let counts = page.counts;
        let page = fit_page(page.issues.into_iter().map(short_issue), offset, limit, page.total);
        let filter: Vec<&str> = a.severity.map(|s| s.as_str()).into_iter().chain(a.kind.as_deref()).collect();
        let summary = format!(
            "{} of {} {}issues of {}; {} errors, {} warnings, {} infos in all",
            page.items.len(),
            page.total,
            if filter.is_empty() { String::new() } else { format!("{} ", filter.join(" ")) },
            a.dataset_id,
            counts.error,
            counts.warning,
            counts.info
        );
        let result =
            IssuesPage { items: page.items, next_cursor: page.next_cursor, total: page.total, counts };
        out(&result, summary)
    }

    async fn list_dataset_images(&self, a: ListImagesArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        self.ensure_dataset(&a.dataset_id).await?;
        let q = ImageQuery { split: a.split, offset, limit, class: a.class };
        let id = a.dataset_id.clone();
        let page = self.blocking(move |s| s.list_images(&id, &q)).await?;
        let with_boxes = a.include_boxes;
        let items = page.images.into_iter().map(|i| ImageSummary {
            image_id: i.id,
            file: i.file,
            split: i.split,
            width: i.width,
            height: i.height,
            boxes: with_boxes.then_some(i.boxes),
        });
        let page = fit_page(items, offset, limit, page.total);
        let summary = format!("{} of {} images of {}", page.items.len(), page.total, a.dataset_id);
        out(&page, summary)
    }

    async fn register_dataset(&self, a: RegisterArgs) -> Result<Output, Error> {
        if let Some(n) = &a.name {
            check_name("name", n)?;
        }
        let path = self.resolve(a.path).await?;
        let req = RegisterDataset { name: a.name, path: folo_api::json::path(&path), format: a.format };
        // The store returns the existing dataset for a folder registered before (same canonical path).
        let (d, existed) = self
            .blocking(move |s| {
                let before: HashSet<String> = s.list_datasets()?.into_iter().map(|d| d.id).collect();
                let d = s.register_dataset(req)?;
                let existed = before.contains(&d.id);
                Ok((d, existed))
            })
            .await?;
        // The root is inside the data dir; with --confine its contents must be too (a data.yaml or links can point
        // elsewhere). If not, unregister it again (files are never touched); one registered earlier stays.
        if let Err(e) = self.ensure_dataset(&d.id).await {
            if !existed {
                let (id, c) = (d.id.clone(), self.confinement.clone());
                let _ = self.blocking(move |s| s.delete_dataset(&id)).await;
                c.forget(&d.id);
            }
            return Err(e);
        }
        let summary = if existed {
            format!("{} is already registered as {} ({}); returned it", d.path, d.id, d.name)
        } else {
            format!("registered {} as {} ({} classes)", d.name, d.id, d.classes.len())
        };
        out(&d, summary)
    }

    async fn create_synthetic(&self, a: SyntheticArgs, progress: &mut Progress) -> Result<Output, Error> {
        if !(1..=100_000).contains(&a.images) {
            return Err(invalid_arguments(format!("images must be in 1..=100000, got {}", a.images)));
        }
        check_name("name", &a.name)?;
        if a.classes.len() > 80 || a.classes.iter().any(|c| c.chars().count() > 64) {
            return Err(invalid_arguments("classes must hold at most 80 names of at most 64 characters"));
        }
        let total = a.images as f64;
        progress.report(0.0, Some(total), false).await;
        let req = SyntheticDataset {
            name: a.name,
            images: a.images,
            classes: a.classes,
            imgsz: a.imgsz,
            seed: a.seed,
        };
        let d = self.blocking(move |s| s.create_synthetic(req)).await?;
        progress.report(total, Some(total), true).await;
        let summary = format!(
            "created {} ({}): {} train / {} val images, classes {}",
            d.name,
            d.id,
            d.splits.train,
            d.splits.val,
            d.classes.join(", ")
        );
        out(&d, summary)
    }

    async fn list_models(&self, a: PageArgs) -> Result<Output, Error> {
        let all = self.blocking(|s| s.list_models()).await?;
        let items: Vec<ModelSummary> = all
            .into_iter()
            .map(|m: Model| ModelSummary {
                id: m.id,
                name: short(&m.name, MAX_NAME_CHARS),
                arch: m.arch,
                imgsz: m.imgsz,
                num_classes: m.classes.len(),
                classes: summary_classes(&m.classes),
                metrics: m.metrics,
                head: m.head,
                trained_with: m.trained_with,
                precision: m.precision,
                fused: m.fused,
            })
            .collect();
        let page = page_of(items, a.cursor.as_deref(), a.limit)?;
        let summary = format!("{} of {} models", page.items.len(), page.total);
        out(&page, summary)
    }

    async fn train(&self, args: JsonObject) -> Result<Output, Error> {
        // Resuming takes the arguments as given: the other fields must match the run's config or be omitted, so
        // they are not filled with TrainConfig's defaults (and `dataset_id` may be left out).
        if let Some(id) = args.get("resume_run_id").filter(|v| !v.is_null()) {
            let id =
                id.as_str().ok_or_else(|| invalid_arguments("resume_run_id must be a string"))?.to_string();
            return self.resume(id, args).await;
        }
        // A TrainConfig value the trainer cannot use (`arch: "folo-q"`, a negative `epochs`) is `invalid_config`,
        // as in the CLI and `POST /api/runs`' range checks; a missing `dataset_id` stays `invalid_arguments`.
        let a: TrainArgs = parse_args(args).map_err(|mut e| {
            if !e.body.message.contains("missing field") {
                e.body.code = "invalid_config".into();
                if let Some(rest) = e.body.message.strip_prefix("invalid arguments: ") {
                    e.body.message = format!("invalid train config: {rest}");
                }
            }
            e
        })?;
        check_name("name", &a.config.name)?;
        self.ensure_dataset(&a.config.dataset_id).await?;
        let run = self.blocking(move |s| s.create_run(a.config)).await?;
        // Embedded runs end with this process, so they are stopped on disconnect; proxied runs belong to the
        // `folo serve` (and Studio) and outlive the session.
        if self.proxy.is_none() {
            self.created_runs.lock().unwrap().push(run.id.clone());
        }
        let summary = format!(
            "started {} ({}): {} on {}, {} epochs; follow it with run_status or folo://runs/{}",
            run.id,
            run.name,
            run.config.arch.as_str(),
            run.config.dataset_id,
            run.epochs,
            run.id
        );
        out(&TrainResult { run_id: run.id, status: run.status }, summary)
    }

    async fn resume(&self, id: String, args: JsonObject) -> Result<Output, Error> {
        let rid = id.clone();
        let dataset_id = self.blocking(move |s| Ok(s.get_run(&rid)?.config.dataset_id)).await?;
        self.ensure_dataset(&dataset_id).await?;
        let rid = id.clone();
        let run = self.blocking(move |s| s.resume_run(&rid, &args)).await?;
        if self.proxy.is_none() {
            let mut created = self.created_runs.lock().unwrap();
            if !created.contains(&run.id) {
                created.push(run.id.clone());
            }
        }
        let summary = format!(
            "resuming {} ({}) after epoch {} of {}; follow it with run_status or folo://runs/{}",
            run.id, run.name, run.epoch, run.epochs, run.id
        );
        out(&TrainResult { run_id: run.id, status: run.status }, summary)
    }

    async fn run_status(&self, a: RunStatusArgs) -> Result<Output, Error> {
        if a.log_lines > 200 {
            return Err(invalid_arguments(format!("log_lines must be in 0..=200, got {}", a.log_lines)));
        }
        let id = a.run_id.clone();
        let n = a.log_lines as usize;
        let (run, metrics, mut log) = self
            .blocking(move |s| {
                let run = s.get_run(&id)?;
                let metrics = s.run_metrics(&id, u64::MAX)?;
                // The whole log (as far as /logs serves it), for the warnings; `log` keeps the last n lines.
                let log = s.run_logs(&id, WARNING_SCAN_LINES)?;
                Ok((run, metrics, log))
            })
            .await?;
        let warnings = log_warnings(&log);
        log.drain(..log.len().saturating_sub(n));
        let mut r = RunStatusResult {
            run,
            last_step: metrics.steps.last().copied(),
            last_epoch: metrics.epochs.last().cloned(),
            warnings,
            log,
        };
        while json_len(&r) > MAX_RESULT_BYTES && !r.log.is_empty() {
            r.log.remove(0);
        }
        if json_len(&r) > MAX_RESULT_BYTES
            && let Some(e) = r.last_epoch.as_mut()
        {
            e.val.per_class.clear();
        }
        let mut line = run_line(&r.run);
        if let Some(w) = r.warnings.first() {
            line.push_str(&format!("; warning: {w}"));
            if r.warnings.len() > 1 {
                line.push_str(&format!(" (+{} more in `warnings`)", r.warnings.len() - 1));
            }
        }
        out(&r, line)
    }

    async fn list_runs(&self, a: ListRunsArgs) -> Result<Output, Error> {
        let mut all = self.blocking(|s| s.list_runs()).await?;
        if let Some(st) = a.status {
            all.retain(|r| r.status == st);
        }
        let all: Vec<Run> = all.into_iter().map(run_summary).collect();
        let page = page_of(all, a.cursor.as_deref(), a.limit)?;
        let summary = format!("{} of {} runs", page.items.len(), page.total);
        out(&page, summary)
    }

    async fn stop_run(&self, a: RunIdArgs) -> Result<Output, Error> {
        let run = self.blocking(move |s| s.stop_run(&a.run_id)).await?;
        out(&run, format!("{}: {}", run.id, run.status_str()))
    }

    async fn reorder_run(&self, a: ReorderRunArgs) -> Result<Output, Error> {
        let mv = QueueMove::parse(QueueRequest { action: a.action, position: a.position })?;
        let run = self.blocking(move |s| s.move_run(&a.run_id, mv)).await?;
        let place = run
            .queue_position
            .map_or_else(|| run.status_str().to_string(), |p| format!("queue position {p}"));
        let summary = format!("{}: {place}", run.id);
        out(
            &ReorderResult { run_id: run.id, status: run.status, queue_position: run.queue_position },
            summary,
        )
    }

    async fn val(&self, a: ValArgs, progress: &mut Progress) -> Result<Output, Error> {
        check_unit("conf", a.conf)?;
        check_unit("iou", a.iou)?;
        self.ensure_dataset(&a.dataset_id).await?;
        progress.report(0.0, Some(1.0), false).await;
        let req = ValRequest {
            model_id: a.model_id,
            dataset_id: a.dataset_id,
            split: a.split,
            imgsz: a.imgsz,
            conf: a.conf,
            iou: a.iou,
        };
        let r = self.blocking(move |s| s.val(req)).await?;
        progress.report(1.0, Some(1.0), true).await;
        let summary = format!(
            "{} on {}/{} ({} images): mAP50 {:.3}, mAP50-95 {:.3}, P {:.3}, R {:.3}",
            r.model_id,
            r.dataset_id,
            r.split.as_str(),
            r.images,
            r.val.map50,
            r.val.map50_95,
            r.val.precision,
            r.val.recall
        );
        out(&r, summary)
    }

    async fn predict(&self, a: PredictArgs) -> Result<Output, Error> {
        check_unit("conf", a.conf)?;
        check_unit("iou", a.iou)?;
        if !(1..=300).contains(&a.max_det) {
            return Err(invalid_arguments(format!("max_det must be in 1..=300, got {}", a.max_det)));
        }
        // The file an `image_path` names, for a decode error (the server only sees the uploaded bytes).
        let mut file = None;
        let source = match (a.image_path, a.image_base64, a.dataset_id, a.image_id) {
            (Some(p), None, None, None) => {
                file = Some(folo_api::json::path(&p));
                let paths = self.paths.clone();
                let read = move || read_image(&paths.resolve(&p)?);
                PredictSource::Upload(tokio::task::spawn_blocking(read).await.map_err(join)??)
            }
            (None, Some(b64), None, None) => {
                if b64.len() as u64 > MAX_IMAGE_BYTES * 4 / 3 + 4 {
                    return Err(invalid_arguments("image_base64 is larger than 64 MiB"));
                }
                let decode = move || {
                    base64::engine::general_purpose::STANDARD
                        .decode(b64.trim())
                        .map_err(|e| invalid_arguments(format!("image_base64 is not valid base64: {e}")))
                };
                PredictSource::Upload(tokio::task::spawn_blocking(decode).await.map_err(join)??)
            }
            (None, None, Some(dataset_id), Some(image_id)) => {
                self.ensure_dataset(&dataset_id).await?;
                PredictSource::Dataset { dataset_id, image_id }
            }
            _ => {
                return Err(invalid_arguments(
                    "give exactly one image source: image_path, image_base64, or dataset_id + image_id",
                ));
            }
        };
        let input =
            PredictInput { model_id: a.model_id, source, conf: a.conf, iou: a.iou, max_det: a.max_det };
        let mut r = self.blocking(move |s| s.predict(input)).await.map_err(|mut e| {
            if let Some(f) = &file
                && e.body.code == "image_decode_failed"
            {
                e.body.message = format!("{f}: {}", e.body.message);
            }
            e
        })?;
        r.boxes.truncate(a.max_det as usize);
        let mut dropped = 0;
        while json_len(&r) > MAX_RESULT_BYTES && r.boxes.pop().is_some() {
            dropped += 1;
        }
        let mut summary = format!(
            "{} boxes on a {}x{} image in {:.1} ms ({})",
            r.boxes.len(),
            r.image.width,
            r.image.height,
            r.timing.total_ms,
            r.device
        );
        if dropped > 0 {
            summary.push_str(&format!("; {dropped} lowest-scoring boxes left out to stay under 32 KB"));
        }
        out(&r, summary)
    }

    async fn predict_batch(&self, a: PredictBatchArgs, progress: &mut Progress) -> Result<Output, Error> {
        check_unit("conf", a.conf)?;
        check_unit("iou", a.iou)?;
        let model_id = a.model_id.clone();
        let model = self.blocking(move |s| s.get_model(&model_id)).await?;
        let paths: Vec<(String, std::path::PathBuf)> = match (a.paths, a.glob) {
            (Some(paths), None) => {
                if paths.len() > MAX_BATCH {
                    return Err(too_many_inputs(format!("{} paths given, at most {MAX_BATCH}", paths.len())));
                }
                if paths.is_empty() {
                    return Err(invalid_arguments("paths must not be empty"));
                }
                let policy = self.paths.clone();
                let resolve = move || {
                    paths
                        .into_iter()
                        .map(|p| policy.resolve(&p).map(|r| (p, r)))
                        .collect::<Result<Vec<_>, _>>()
                };
                tokio::task::spawn_blocking(resolve).await.map_err(join)??
            }
            (None, Some(g)) => {
                let paths = self.paths.clone();
                let found =
                    tokio::task::spawn_blocking(move || paths.expand_glob(&g)).await.map_err(join)??;
                if found.is_empty() {
                    return Err(invalid_arguments("the glob matches no files"));
                }
                found.into_iter().map(|p| (p.to_string_lossy().into_owned(), p)).collect()
            }
            _ => return Err(invalid_arguments("give exactly one of paths or glob")),
        };
        let started = Instant::now();
        let total = paths.len() as f64;
        progress.report(0.0, Some(total), false).await;
        let mut images = Vec::with_capacity(paths.len());
        let mut failed = Vec::new();
        let mut per_class: BTreeMap<String, u64> = BTreeMap::new();
        let (mut boxes_total, mut model_ms) = (0u64, 0f64);
        for (i, (shown, path)) in paths.into_iter().enumerate() {
            let model_id = a.model_id.clone();
            let (conf, iou) = (a.conf, a.iou);
            let res = self
                .blocking(move |s| {
                    let bytes = read_image(&path)?;
                    s.predict(PredictInput {
                        model_id,
                        source: PredictSource::Upload(bytes),
                        conf,
                        iou,
                        max_det: folo_api::default_max_det(),
                    })
                })
                .await;
            match res {
                Ok(r) => {
                    boxes_total += r.boxes.len() as u64;
                    model_ms += r.timing.total_ms;
                    for b in &r.boxes {
                        *per_class.entry(b.name.clone()).or_default() += 1;
                    }
                    images.push(
                        json!({ "path": shown, "image": r.image, "boxes": r.boxes, "timing": r.timing }),
                    );
                }
                Err(e) if e.body.code == "not_implemented" => return Err(e),
                Err(e) => {
                    images.push(json!({ "path": shown, "error": e.body }));
                    failed.push(BatchFailure { path: shown, code: e.body.code, message: e.body.message });
                }
            }
            progress.report((i + 1) as f64, Some(total), i + 1 == total as usize).await;
        }
        let processed = (images.len() - failed.len()) as u64;
        let batch_id = self.next_batch_id();
        let details_uri = format!("folo://predictions/{batch_id}");
        let details = json!({ "batch_id": batch_id, "model_id": model.id, "images": images });
        self.batches.lock().unwrap().insert(batch_id.clone(), Arc::new(details));
        let failed_total = failed.len() as u64;
        failed.truncate(MAX_FAILURES);
        let mut result = PredictBatchResult {
            batch_id: batch_id.clone(),
            processed,
            failed,
            failed_total,
            boxes_total,
            per_class_counts: per_class,
            timing: BatchTiming {
                total_ms: (started.elapsed().as_secs_f64() * 1e5).round() / 100.0,
                mean_ms: if processed > 0 {
                    (model_ms / processed as f64 * 100.0).round() / 100.0
                } else {
                    0.0
                },
            },
            details_uri: details_uri.clone(),
        };
        while json_len(&result) > MAX_RESULT_BYTES && result.failed.pop().is_some() {}
        let summary = format!(
            "{processed} images, {boxes_total} boxes, {failed_total} failed; per-image boxes at {details_uri}"
        );
        let mut o = out(&result, summary)?;
        o.links.push(ContentBlock::resource_link(
            Resource::new(details_uri, format!("predictions {batch_id}"))
                .with_description(format!("per-image boxes of predict_batch with {}", model.name))
                .with_mime_type("application/json"),
        ));
        Ok(o)
    }

    async fn export(&self, a: ExportArgs) -> Result<Output, Error> {
        let spec = ExportSpec {
            format: a.format,
            onnx: OnnxSwitches { decode: a.decode, dynamic: a.dynamic, half: a.half },
        };
        let r = self.blocking(move |s| s.export_model(&a.model_id, spec)).await?;
        out(&r, format!("wrote {} ({} bytes)", r.path, r.bytes))
    }

    async fn compare_runs(&self, a: CompareArgs) -> Result<Output, Error> {
        if !(2..=8).contains(&a.run_ids.len()) {
            return Err(invalid_arguments(format!("run_ids must hold 2 to 8 ids, got {}", a.run_ids.len())));
        }
        let ids = a.run_ids.clone();
        let runs: Vec<(Run, RunMetrics)> = self
            .blocking(move |s| {
                ids.iter().map(|id| Ok((s.get_run(id)?, s.run_metrics(id, u64::MAX)?))).collect()
            })
            .await?;
        let entries: Vec<CompareEntry> = runs
            .iter()
            .map(|(run, m)| {
                let best_metrics = run
                    .best
                    .and_then(|b| m.epochs.iter().find(|e| e.epoch == b.epoch))
                    .map(|e| metrics_of(&e.val));
                CompareEntry {
                    run_id: run.id.clone(),
                    name: run.name.clone(),
                    status: run.status,
                    arch: run.config.arch,
                    epochs_done: m.epochs.len() as u32,
                    epochs: run.epochs,
                    best: run.best,
                    best_metrics,
                    final_train_loss: m.epochs.last().map(|e| e.train_loss),
                    duration_s: (m.epochs.iter().map(|e| e.duration_s).sum::<f64>() * 10.0).round() / 10.0,
                }
            })
            .collect();
        let configs: Vec<Value> =
            runs.iter().map(|(r, _)| serde_json::to_value(&r.config).unwrap_or_default()).collect();
        let result = CompareResult { runs: entries, config_diff: config_diff(&configs) };
        let best = result
            .runs
            .iter()
            .filter_map(|e| e.best.map(|b| (e, b)))
            .max_by(|x, y| x.1.map50_95.total_cmp(&y.1.map50_95));
        let summary = match best {
            Some((e, b)) => format!(
                "{} runs; best {} ({}) with mAP50-95 {:.3} at epoch {}; {} config fields differ",
                result.runs.len(),
                e.run_id,
                e.name,
                b.map50_95,
                b.epoch,
                result.config_diff.len()
            ),
            None => format!("{} runs, none validated yet", result.runs.len()),
        };
        out(&result, summary)
    }

    /// The `folo_info` record.
    pub async fn info(&self) -> Result<FoloInfo, Error> {
        let (cpu, gpus, data_dir, capabilities) = match &self.proxy {
            Some(p) => {
                let p = p.clone();
                let sys = tokio::task::spawn_blocking(move || p.system()).await.map_err(join)??;
                let (data_dir, capabilities, sys) = (sys.data_dir, sys.capabilities, sys.system);
                let cpu = CpuDevice {
                    name: sys.cpu.name,
                    threads: sys.cpu.threads,
                    memory_bytes: sys.memory.total_bytes,
                };
                // The server's data dir: runs, models and exports live there, not under this process's default.
                (cpu, sys.gpus, data_dir, capabilities)
            }
            None => {
                let store = self.store.clone();
                tokio::task::spawn_blocking(move || {
                    let host = folo_server::system::host_info();
                    let cpu = CpuDevice {
                        name: host.cpu_name,
                        threads: host.cpu_threads,
                        memory_bytes: host.memory_total_bytes,
                    };
                    // Capabilities first, as `/api/system` does: they create the GPU context, and `gpus()` (cached
                    // on first use) puts the context's adapter at ordinal 0 only once it exists.
                    let caps = store.capabilities();
                    (cpu, store.gpus(), caps)
                })
                .await
                .map(|(cpu, gpus, caps)| (cpu, gpus, Some(folo_api::json::path(&self.paths.data_dir)), caps))
                .map_err(join)?
            }
        };
        Ok(FoloInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            mode: if self.proxy.is_some() {
                "proxy"
            } else if self.mock {
                "mock"
            } else {
                "embedded"
            }
            .to_string(),
            server: self.proxy.as_ref().map(|p| p.base().to_string()),
            data_dir,
            read_only: self.opts.read_only,
            confine: self.paths.confine,
            devices: Devices { cpu, gpus },
            capabilities,
            scales: SCALES
                .iter()
                .map(|&(arch, params, gflops_640, gflops_320)| ScaleInfo {
                    arch,
                    params,
                    gflops_640,
                    gflops_320,
                })
                .collect(),
        })
    }
}

/// Model scales at 80 classes (docs/MODEL.md, "Parameter and FLOP totals").
const SCALES: [(Arch, u64, f64, f64); 5] = [
    (Arch::N, 2_624_064, 6.54, 1.62),
    (Arch::S, 9_458_736, 21.59, 5.37),
    (Arch::M, 20_114_672, 68.10, 17.00),
    (Arch::L, 25_372_144, 87.15, 21.74),
    (Arch::X, 56_966_160, 195.27, 48.75),
];

trait StatusStr {
    fn status_str(&self) -> String;
}

impl StatusStr for Run {
    fn status_str(&self) -> String {
        serde_json::to_value(self.status)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default()
    }
}

/// One line describing a run.
/// Log lines `run_status` reads for its `warnings` (the server's `/logs` maximum).
const WARNING_SCAN_LINES: usize = 5000;

/// Distinct `warn` lines `run_status` reports.
const MAX_WARNINGS: usize = 5;

/// The distinct `warn`-level messages of a run's log, oldest first, at most [`MAX_WARNINGS`] (each capped at 400
/// characters).
fn log_warnings(log: &[LogLine]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for l in log.iter().filter(|l| l.level == LogLevel::Warn) {
        let m = folo_api::codes::cap(&l.message, 400);
        if !out.contains(&m) {
            out.push(m);
            if out.len() == MAX_WARNINGS {
                break;
            }
        }
    }
    out
}

fn run_line(r: &Run) -> String {
    let mut s =
        format!("{} ({}): {}, epoch {}/{}, step {}", r.id, r.name, r.status_str(), r.epoch, r.epochs, r.step);
    if let Some(b) = r.best {
        s.push_str(&format!(", best mAP50-95 {:.3} at epoch {}", b.map50_95, b.epoch));
    }
    if let Some(eta) = r.eta_s {
        s.push_str(&format!(", ETA {eta:.0} s"));
    }
    if let Some(m) = &r.model_id {
        s.push_str(&format!(", model {m}"));
    }
    if let Some(e) = &r.error {
        s.push_str(&format!(", error: {e}"));
    }
    s
}

pub(crate) fn join(e: tokio::task::JoinError) -> Error {
    Error::internal(format!("task failed: {e}"))
}

/// Reads an image file for inference (at most 64 MiB).
fn read_image(path: &Path) -> Result<Vec<u8>, Error> {
    let meta = std::fs::metadata(path).map_err(|_| {
        Error::new(
            http::StatusCode::NOT_FOUND,
            "path_not_found",
            format!("{} does not exist", folo_api::json::path(path)),
        )
    })?;
    if !meta.is_file() {
        return Err(invalid_arguments(format!("{} is not a file", folo_api::json::path(path))));
    }
    if meta.len() > MAX_IMAGE_BYTES {
        return Err(Error::new(
            http::StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            format!("{} is larger than 64 MiB", folo_api::json::path(path)),
        ));
    }
    std::fs::read(path).map_err(|e| Error::internal(format!("reading {}: {e}", folo_api::json::path(path))))
}

/// Dotted-path fields whose values differ between `configs`, with one value per config.
fn config_diff(configs: &[Value]) -> BTreeMap<String, Vec<Value>> {
    fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, Value>) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    flatten(&key, v, out);
                }
            }
            other => {
                out.insert(prefix.to_string(), other.clone());
            }
        }
    }
    let flat: Vec<BTreeMap<String, Value>> = configs
        .iter()
        .map(|c| {
            let mut m = BTreeMap::new();
            flatten("", c, &mut m);
            m
        })
        .collect();
    let keys: HashSet<&String> = flat.iter().flat_map(|m| m.keys()).collect();
    let mut diff = BTreeMap::new();
    for k in keys {
        let values: Vec<Value> = flat.iter().map(|m| m.get(k).cloned().unwrap_or(Value::Null)).collect();
        if values.windows(2).any(|w| w[0] != w[1]) {
            diff.insert(k.clone(), values);
        }
    }
    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_status_lists_distinct_warnings() {
        let line = |level, m: &str| LogLine { t: String::new(), level, message: m.into() };
        let mut log = vec![
            line(LogLevel::Info, "folo-n on gpu:0 …"),
            line(LogLevel::Warn, "this run takes only 36 optimizer steps"),
            line(LogLevel::Warn, "this run takes only 36 optimizer steps"),
            line(LogLevel::Error, "run failed"),
        ];
        log.extend((0..10).map(|i| line(LogLevel::Warn, &format!("epoch 1 batch {i}: non-finite loss"))));
        let w = log_warnings(&log);
        assert_eq!(w.len(), MAX_WARNINGS);
        assert_eq!(w[0], "this run takes only 36 optimizer steps");
        assert_eq!(w[1], "epoch 1 batch 0: non-finite loss");
        assert!(log_warnings(&log[..1]).is_empty());
    }

    #[test]
    fn every_schema_is_an_object() {
        for t in TOOLS {
            let tool = t.tool();
            assert_eq!(tool.input_schema.get("type"), Some(&json!("object")), "{}", t.name);
            let out = tool.output_schema.as_ref().unwrap();
            assert_eq!(out.get("type"), Some(&json!("object")), "{}", t.name);
        }
        let names: HashSet<_> = TOOLS.iter().map(|t| t.name).collect();
        assert_eq!(names.len(), TOOLS.len());
    }

    #[test]
    fn unknown_arguments_are_rejected_with_a_suggestion() {
        let args = |v: Value| v.as_object().unwrap().clone();
        let train = find("train").unwrap();
        let schema = (train.input)();
        assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
        let ok = args(json!({ "dataset_id": "ds_1", "epochs": 1, "lr0": 0.1, "augment": { "mosaic": 0.5 } }));
        assert!(unknown_argument("train", &schema, &ok).is_none());
        assert!(unknown_argument("train", &schema, &args(json!({ "resume_run_id": "run_1" }))).is_none());
        let e = unknown_argument("train", &schema, &args(json!({ "dataset_id": "d", "lr": 0.1 }))).unwrap();
        assert_eq!(e.body.code, "invalid_arguments");
        assert_eq!(e.body.message, "unknown argument `lr`; did you mean `lr0`?");
        assert!(e.body.hint.as_deref().unwrap().contains("dataset_id"));
        let e = unknown_argument("train", &schema, &args(json!({ "dataset_id": "d", "model": "folo-s" })))
            .unwrap();
        assert!(e.body.message.contains("did you mean `arch`?"), "{}", e.body.message);
        let nested = args(json!({ "dataset_id": "d", "augment": { "mosiac": 0.5 } }));
        let e = unknown_argument("train", &schema, &nested).unwrap();
        assert_eq!(e.body.message, "unknown argument `augment.mosiac`; did you mean `mosaic`?");
        for t in TOOLS {
            let s = (t.input)();
            assert_eq!(s.get("additionalProperties"), Some(&json!(false)), "{}", t.name);
            assert!(unknown_argument(t.name, &s, &args(json!({ "zzz_bogus": 1 }))).is_some(), "{}", t.name);
        }
    }

    #[test]
    fn issues_from_an_older_server_are_filtered_and_counted_here() {
        use folo_api::{DatasetIssue, IssueCounts};
        let issue = |kind: &str, severity| DatasetIssue {
            kind: kind.into(),
            severity,
            file: "f".into(),
            line: None,
            message: "m".into(),
            fix: None,
        };
        // 60 background images, then one error: an old server ignores `severity` and sends no counts.
        let mut all: Vec<DatasetIssue> = (0..60).map(|_| issue("empty_image", IssueSeverity::Info)).collect();
        all.push(issue("class_overflow", IssueSeverity::Error));
        let old = |q: &IssueQuery| -> Result<IssuePage, Error> {
            let page = all.iter().skip(q.offset as usize).take(q.limit as usize).cloned().collect();
            Ok(IssuePage { total: all.len() as u64, issues: page, counts: IssueCounts::default() })
        };
        let q = IssueQuery { severity: Some(IssueSeverity::Error), limit: 50, ..IssueQuery::default() };
        let page = list_issues_compat(old, &q).unwrap();
        assert_eq!((page.total, page.issues.len()), (1, 1));
        assert_eq!(page.issues[0].kind, "class_overflow");
        assert_eq!((page.counts.error, page.counts.info), (1, 60));
        // A new server's answer is used as is, also an empty one.
        let new = |q: &IssueQuery| Ok(q.page(all.iter()));
        assert_eq!(list_issues_compat(new, &q).unwrap(), q.page(all.iter()));
        let none =
            |_: &IssueQuery| Ok(IssuePage { total: 0, issues: vec![], counts: IssueCounts::default() });
        assert_eq!(list_issues_compat(none, &q).unwrap().total, 0);
        // An old server that honours `kind`: a kind without matches still gets the dataset-wide counts.
        let old_kind = |q: &IssueQuery| -> Result<IssuePage, Error> {
            let m: Vec<_> =
                all.iter().filter(|i| q.kind.as_ref().is_none_or(|k| *k == i.kind)).cloned().collect();
            let page = m.iter().skip(q.offset as usize).take(q.limit as usize).cloned().collect();
            Ok(IssuePage { total: m.len() as u64, issues: page, counts: IssueCounts::default() })
        };
        let q = IssueQuery { kind: Some("clipped_box".into()), limit: 50, ..IssueQuery::default() };
        let page = list_issues_compat(old_kind, &q).unwrap();
        assert_eq!((page.total, page.counts.error, page.counts.info), (0, 1, 60));
    }

    #[test]
    fn pages_fit_the_budget() {
        let items: Vec<String> = (0..500).map(|i| format!("{i:0>200}")).collect();
        let p = page_of(items, None, Some(200)).unwrap();
        assert!(p.items.len() < 200);
        assert!(json_len(&p) <= MAX_RESULT_BYTES);
        let next = p.next_cursor.clone().unwrap();
        assert_eq!(next, cursor_at(p.items.len() as u64));
        let items: Vec<u32> = (0..10).collect();
        let p = page_of(items.clone(), Some("o8"), Some(5)).unwrap();
        assert_eq!(p.items, vec![8, 9]);
        assert!(p.next_cursor.is_none());
        assert!(page_of(items.clone(), Some("bogus"), None).is_err());
        assert!(page_of(items, None, Some(201)).is_err());
    }

    #[test]
    fn oversized_results_are_tool_errors() {
        let big = vec!["z".repeat(1000); 40];
        let e = out(&big, "s").err().unwrap();
        assert_eq!(e.body.code, "result_too_large");
        let ok = out(&json!({ "a": 1 }), "w".repeat(1000)).ok().unwrap();
        assert_eq!(ok.summary.chars().count(), MAX_SUMMARY_CHARS + 1);
        assert_eq!(short("héllo", 2), "hé…");
        assert_eq!(short("hi", 2), "hi");
    }

    #[test]
    fn diff_lists_differing_fields() {
        let a = json!({ "epochs": 10, "augment": { "mosaic": 1.0, "fliplr": 0.5 }, "name": "a" });
        let b = json!({ "epochs": 20, "augment": { "mosaic": 1.0, "fliplr": 0.0 }, "name": "b" });
        let d = config_diff(&[a, b]);
        assert_eq!(d.keys().collect::<Vec<_>>(), ["augment.fliplr", "epochs", "name"]);
        assert_eq!(d["epochs"], vec![json!(10), json!(20)]);
    }
}
