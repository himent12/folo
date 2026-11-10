//! The tools of docs/MCP.md: their schemas (generated from the Rust types with schemars) and their dispatch onto
//! the [`Store`](folo_server::Store).

mod arguments;
mod cloud;
mod datasets;
mod dispatch;
mod info;
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
use schemas::{api_output, input, output, train_input};

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
mod tests;
