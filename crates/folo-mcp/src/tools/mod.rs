//! The tools of docs/MCP.md: their schemas (generated from the Rust types with schemars) and their dispatch onto
//! the [`Store`](folo_server::Store).
//!
//! This file holds the table, [`TOOLS`] (name, description, `mutating`, input and output schema), in the order of
//! docs/MCP.md, and what `tools/list` and `folo describe` show of it. The rest of the folder:
//!
//! - `dispatch.rs`: a call from request to handler (the `--read-only` refusal, argument checks, progress).
//! - `schemas.rs`: schemas generated with schemars and shrunk; `arguments.rs`: unknown and malformed arguments.
//! - `results.rs`: size limits, summaries and pages of a result.
//! - Handlers: `info.rs` (`folo_info`), `datasets.rs` (the dataset tools), `models.rs` (`list_models`, `val`,
//!   `predict`, `predict_batch`, `export`), `runs.rs` (`train`, `run_status`, `list_runs`, `stop_run`,
//!   `reorder_run`, `compare_runs`), `cloud.rs` (the `cloud_*` tools), `label.rs` (the labeling tools, on
//!   [`crate::label::Labels`]).
//! - `tests.rs`: unit tests, including the tool count the docs state; `label_tests.rs`: the labeling tools in
//!   process and through a `folo serve`, with a scripted agent.

mod arguments;
mod cloud;
mod datasets;
mod dispatch;
mod info;
mod label;
mod models;
mod results;
mod runs;
mod schemas;

use std::sync::Arc;

use folo_api::{
    CloudConnectionList, CloudInstance, CloudInstanceList, CloudOfferList, Dataset, ExportResult,
    PredictResponse, Run, ValReport,
};
use rmcp::model::{JsonObject, Tool, ToolAnnotations};
use serde_json::{Value, json};

use crate::types::*;
use schemas::{api_output, input, output, terse_input, terse_output, train_input};

pub use dispatch::Progress;
pub(crate) use dispatch::join;
pub(crate) use results::short;

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
        output: api_output::<Dataset>,
    },
    ToolDef {
        name: "create_synthetic_dataset",
        description: "Generates a synthetic shapes dataset (circles, squares, triangles, stars on textured \
                      backgrounds) for quick experiments. Reports progress.",
        mutating: true,
        input: input::<SyntheticArgs>,
        output: api_output::<Dataset>,
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
        output: api_output::<Run>,
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
        output: api_output::<ValReport>,
    },
    ToolDef {
        name: "predict",
        description: "Detects objects in one image (a file path, base64 bytes, or a dataset image). Boxes are \
                      pixel xyxy, highest score first.",
        mutating: false,
        input: input::<PredictArgs>,
        output: api_output::<PredictResponse>,
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
        output: api_output::<ExportResult>,
    },
    ToolDef {
        name: "compare_runs",
        description: "Compares 2 to 8 runs: best metrics, final losses, epochs, duration, and the TrainConfig \
                      fields that differ.",
        mutating: false,
        input: input::<CompareArgs>,
        output: output::<CompareResult>,
    },
    ToolDef {
        name: "cloud_connections",
        description: "Lists the cloud connections (account, balance, last check). Connecting takes credentials, so \
                      it is done in Studio, Desktop or `folo cloud connect`.",
        mutating: false,
        input: input::<NoArgs>,
        output: api_output::<CloudConnectionList>,
    },
    ToolDef {
        name: "cloud_offers",
        description: "What a cloud connection can run on now, cheapest first; train on one with train's \
                      compute.offer_id.",
        mutating: false,
        input: input::<CloudOffersArgs>,
        output: api_output::<CloudOfferList>,
    },
    ToolDef {
        name: "cloud_instances",
        description: "Lists the cloud instances FOLO launched, alive first, with their run, status and cost.",
        mutating: false,
        input: input::<NoArgs>,
        output: api_output::<CloudInstanceList>,
    },
    ToolDef {
        name: "cloud_terminate",
        description: "Terminates a cloud instance now; a run still using it fails (cloud_instance_lost).",
        mutating: true,
        input: input::<InstanceIdArgs>,
        output: api_output::<CloudInstance>,
    },
    ToolDef {
        name: "list_label_jobs",
        description: "Lists labeling jobs, newest first. Paged.",
        mutating: false,
        input: terse_input::<PageArgs>,
        output: terse_output::<Page<LabelJobSummary>>,
    },
    ToolDef {
        name: "create_label_job",
        description: "Creates a labeling job (name, classes, instructions) from folders or image files on the \
                      server's machine; with job_id, adds paths to that job.",
        mutating: true,
        input: terse_input::<CreateLabelJobArgs>,
        output: terse_output::<LabelImagesAddedResult>,
    },
    ToolDef {
        name: "start_labeling",
        description: "Starts an agent run on the job's unlabeled images (or image_ids); returns at once, follow \
                      it with label_status.",
        mutating: true,
        input: terse_input::<StartLabelingArgs>,
        output: terse_output::<LabelRunStarted>,
    },
    ToolDef {
        name: "stop_labeling",
        description: "Stops a job's agent run; labeled images stay.",
        mutating: true,
        input: terse_input::<JobIdArgs>,
        output: terse_output::<LabelJobState>,
    },
    ToolDef {
        name: "label_status",
        description: "A job's counts, run progress, tokens and last log lines.",
        mutating: false,
        input: terse_input::<LabelStatusArgs>,
        output: terse_output::<LabelStatusResult>,
    },
    ToolDef {
        name: "list_label_images",
        description: "Lists a job's images, optionally by status or review state, with boxes. Paged.",
        mutating: false,
        input: terse_input::<ListLabelImagesArgs>,
        output: terse_output::<Page<LabelImageSummary>>,
    },
    ToolDef {
        name: "view_label_image",
        description: "Shows an image (default: the next to review) with its boxes numbered and pixel ticks on \
                      the edges, plus the box list [class,x1,y1,x2,y2] in pixels of the view and its revision.",
        mutating: false,
        input: terse_input::<ViewLabelImageArgs>,
        output: terse_output::<LabelImageViewResult>,
    },
    ToolDef {
        name: "set_label_boxes",
        description: "Corrects an image's boxes, in pixels of the view: all of them (boxes), or relabel / remove \
                      / add by number with expect: the view's revision. Saved as a reviewed hand edit.",
        mutating: true,
        input: terse_input::<SetLabelBoxesArgs>,
        output: terse_output::<LabelImageView>,
    },
    ToolDef {
        name: "review_label_images",
        description: "Marks labeled images whose boxes are right as reviewed (expect: id -> revision seen).",
        mutating: true,
        input: terse_input::<ReviewLabelImagesArgs>,
        output: terse_output::<ReviewLabelImagesResult>,
    },
    ToolDef {
        name: "export_label_dataset",
        description: "Writes the labeled images as a YOLO dataset, registers it and returns its dataset_id.",
        mutating: true,
        input: terse_input::<ExportLabelDatasetArgs>,
        output: terse_output::<LabelDatasetResult>,
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

#[cfg(test)]
mod label_tests;
#[cfg(test)]
mod tests;
