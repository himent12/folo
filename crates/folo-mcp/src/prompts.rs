//! Prompts (docs/MCP.md "Prompts"). Retrieval only builds messages: it never starts jobs or touches files.

use rmcp::ErrorData as McpError;
use rmcp::model::{GetPromptRequestParams, GetPromptResult, Prompt, PromptArgument, PromptMessage, Role};
use serde_json::json;

use crate::server::Inner;

/// The prompts a server lists (`train_on_folder` starts training, so read-only servers hide it).
pub fn list(read_only: bool) -> Vec<Prompt> {
    let mut out = Vec::new();
    if !read_only {
        out.push(Prompt::new(
            "train_on_folder",
            Some("Register a dataset folder, choose a model scale and image size from its stats and the devices, and train."),
            Some(vec![
                PromptArgument::new("folder_path")
                    .with_description("The dataset folder (YOLO or COCO), absolute or relative to --data-dir.")
                    .with_required(true),
                PromptArgument::new("goal")
                    .with_description("Free text, e.g. \"fast on CPU\" or \"best accuracy\".")
                    .with_required(false),
            ]),
        ));
    }
    out.push(Prompt::new(
        "diagnose_run",
        Some("Summarise a run's metrics and logs and propose a concrete TrainConfig change."),
        Some(vec![
            PromptArgument::new("run_id").with_description("The run to diagnose.").with_required(true),
        ]),
    ));
    out
}

fn arg(req: &GetPromptRequestParams, name: &str) -> Option<String> {
    req.arguments.as_ref()?.get(name)?.as_str().map(str::to_string).filter(|s| !s.trim().is_empty())
}

fn missing(prompt: &str, name: &str) -> McpError {
    McpError::invalid_params(format!("prompt `{prompt}` needs the `{name}` argument"), None)
}

/// `prompts/get`.
pub async fn get(inner: &Inner, req: GetPromptRequestParams) -> Result<GetPromptResult, McpError> {
    match req.name.as_ref() {
        "train_on_folder" if !inner.opts.read_only => {
            let folder = arg(&req, "folder_path").ok_or_else(|| missing("train_on_folder", "folder_path"))?;
            let goal = arg(&req, "goal");
            let folder_json = serde_json::to_string(&folder).expect("strings serialize");
            let mut text = format!(
                "Train a FOLO object detector on the dataset folder {folder_json}.\n\n\
                 1. Call `register_dataset` with `{{ \"path\": {folder_json} }}`.\n\
                 2. Call `get_dataset` with the returned id. Read `stats` (class counts, box sizes, boxes per image) \
                 and `issues`; if there are broken labels or a class has very few boxes, tell the user before \
                 training.\n\
                 3. Call `folo_info` to see the devices and the model scales (params, GFLOPs).\n\
                 4. Choose `arch` (folo-n ... folo-x) and `imgsz` (a multiple of 32): small boxes (box_size_hist \
                 mass below 0.05) need a larger imgsz; a CPU-only machine or a small dataset favours folo-n or \
                 folo-s; a GPU with plenty of memory allows larger scales and batches.\n\
                 5. Call `train` with `dataset_id`, `arch`, `imgsz`, `epochs` and `batch`, then follow the run with \
                 `run_status` (or subscribe to `folo://runs/{{run_id}}`).\n\
                 6. Explain your choices to the user in two or three sentences."
            );
            if let Some(g) = goal {
                text.push_str(&format!(
                    "\n\nThe user's goal: {g}. Let it guide the scale, image size and epochs."
                ));
            }
            Ok(GetPromptResult::new(vec![PromptMessage::new_text(Role::User, text)])
                .with_description("Register a folder and train on it"))
        }
        "diagnose_run" => {
            let run_id = arg(&req, "run_id").ok_or_else(|| missing("diagnose_run", "run_id"))?;
            let id = run_id.clone();
            let (run, metrics, logs) = inner
                .blocking(move |s| Ok((s.get_run(&id)?, s.run_metrics(&id, u64::MAX)?, s.run_logs(&id, 20)?)))
                .await
                .map_err(|e| {
                    McpError::invalid_params(e.body.message.clone(), Some(json!({ "error": e.body })))
                })?;
            // Keep the embedded summary short: every epoch's headline numbers, not the per-class rows.
            let epochs: Vec<_> = metrics
                .epochs
                .iter()
                .map(|e| {
                    json!({
                        "epoch": e.epoch,
                        "train_loss": e.train_loss,
                        "map50": e.val.map50,
                        "map50_95": e.val.map50_95,
                        "precision": e.val.precision,
                        "recall": e.val.recall,
                    })
                })
                .collect();
            let last_per_class = metrics.epochs.last().map(|e| &e.val.per_class);
            let summary = json!({
                "run": { "id": run.id, "name": run.name, "status": run.status, "epoch": run.epoch,
                         "epochs": run.epochs, "best": run.best, "error": run.error },
                "config": run.config,
                "epochs": epochs,
                "last_epoch_per_class": last_per_class,
                "last_step": metrics.steps.last(),
            });
            let log: Vec<String> =
                logs.iter().map(|l| format!("{} {:?} {}", l.t, l.level, l.message)).collect();
            let text = format!(
                "Diagnose FOLO training run {run_id}.\n\nMetric summary (JSON):\n{}\n\nLast log lines:\n{}\n\n\
                 Identify overfitting (validation mAP falling or flat while training loss keeps dropping), \
                 underfitting (both losses high, mAP still rising at the end), class imbalance (per-class mAP far \
                 apart; compare with the dataset's class counts via `get_dataset`) or learning-rate problems \
                 (loss spikes, NaN, or no progress in the first epochs). Then propose a concrete TrainConfig diff \
                 (for example `{{ \"epochs\": 150, \"lr0\": 0.005, \"augment\": {{ \"mosaic\": 0.5 }} }}`) and \
                 explain why. Do not start a run unless the user asks.",
                serde_json::to_string_pretty(&summary).expect("JSON serializes"),
                if log.is_empty() { "(none)".to_string() } else { log.join("\n") }
            );
            Ok(GetPromptResult::new(vec![PromptMessage::new_text(Role::User, text)])
                .with_description(format!("Diagnose run {run_id}")))
        }
        other => Err(McpError::invalid_params(format!("unknown prompt: {other}"), None)),
    }
}
