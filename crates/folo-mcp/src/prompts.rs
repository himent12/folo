//! Prompts (docs/MCP.md "Prompts"): `train_on_folder`, `label_folder` and `diagnose_run`. Retrieval only builds
//! messages: it never starts jobs or touches files.

use folo_server::store::{LogsQuery, MetricsQuery};
use rmcp::ErrorData as McpError;
use rmcp::model::{GetPromptRequestParams, GetPromptResult, Prompt, PromptArgument, PromptMessage, Role};
use serde_json::json;

use crate::server::Inner;
use crate::tools::{self, NotFoundTarget};

/// The prompts a server lists (`train_on_folder` and `label_folder` change state, so read-only servers hide them).
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
    if !read_only {
        out.push(Prompt::new(
            "label_folder",
            Some("Label a folder of unlabeled images with a coding agent, review the boxes, export a dataset and train."),
            Some(vec![
                PromptArgument::new("folder_path")
                    .with_description("The folder of images on the server's machine, absolute or relative to --data-dir.")
                    .with_required(true),
                PromptArgument::new("classes")
                    .with_description("The classes to find, comma-separated, e.g. \"car, person\".")
                    .with_required(true),
                PromptArgument::new("instructions")
                    .with_description("What counts as each class and what to skip, in the user's words.")
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
                 4. Choose `arch` (folo-n ... folo-x, or folo2-n ... folo2-x) and `imgsz` (a multiple of 32): small boxes (box_size_hist \
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
        "label_folder" if !inner.opts.read_only => {
            let folder = arg(&req, "folder_path").ok_or_else(|| missing("label_folder", "folder_path"))?;
            let classes = arg(&req, "classes").ok_or_else(|| missing("label_folder", "classes"))?;
            let classes: Vec<&str> = classes.split(',').map(str::trim).filter(|c| !c.is_empty()).collect();
            let instructions = arg(&req, "instructions").unwrap_or_default();
            let create = json!({
                "name": std::path::Path::new(&folder).file_name().and_then(|n| n.to_str()).unwrap_or("images"),
                "classes": classes,
                "instructions": instructions,
                "paths": [folder],
            });
            let text = format!(
                "Label the images of a folder with a coding agent, check its boxes yourself, and make a training \
                 dataset. Keep tool results small: use ids, counts and pages, not full listings.\n\n\
                 1. `create_label_job` with {create}. Sharpen `instructions` first if the user's words leave \
                 a class unclear (look-alikes, what to skip, partly hidden objects).\n\
                 2. Try the instructions on a sample: `list_label_images` (limit 20) and `start_labeling` with \
                 those `image_ids` and agent `claude` (leave model and effort out: its defaults are the cheapest \
                 that label well). If it answers `not_supported` (not installed), try `codex` or `antigravity`; if \
                 none is installed, tell the user what the error's `hint` says and stop. Poll `label_status` every \
                 20-30 s until \
                 `status` is `idle`; if the run failed, report the error its `log` shows and stop.\n\
                 3. Review the sample one image at a time: `view_label_image` without `image_id` shows the next \
                 image to review, its boxes numbered, pixel ticks on the edges, the box list [class, x1, y1, x2, \
                 y2] in pixels of the view (`size`) and the boxes' `revision`. Settle each image before the next \
                 view, or the same image comes back: right: `review_label_images` with its id and `expect` \
                 {{id: revision}}; wrong: `set_label_boxes` with `expect` set to the revision and `relabel` / \
                 `remove` / `add` by number (or all `boxes`), in the view's pixels with `size` set to the view's \
                 `size` (this also marks it reviewed). If the same mistake repeats, improve the instructions in a `note` and ask again with \
                 `start_labeling` and those `image_ids` rather than fixing every image by hand.\n\
                 4. When the sample looks right, `start_labeling` with no `image_ids` (the rest), poll \
                 `label_status`, and review the same way until `to_review` is 0. More images later: \
                 `create_label_job` with the `job_id` and their `paths`.\n\
                 5. `export_label_dataset`, then `train` with its `dataset_id` (see the `train_on_folder` prompt \
                 for picking a scale and image size), and tell the user the tokens used (`label_status` tokens), \
                 how many boxes you corrected, and the run id."
            );
            Ok(GetPromptResult::new(vec![PromptMessage::new_text(Role::User, text)])
                .with_description("Label a folder, review, export and train"))
        }
        "diagnose_run" => {
            let run_id = arg(&req, "run_id").ok_or_else(|| missing("diagnose_run", "run_id"))?;
            let id = run_id.clone();
            let (run, metrics, logs) = match inner
                .blocking(move |s| {
                    let metrics = s.run_metrics(&id, &MetricsQuery::every(u64::MAX))?;
                    Ok((s.get_run(&id)?, metrics, s.run_logs(&id, &LogsQuery::last(20))?.lines))
                })
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    let e = inner.enrich_not_found(e, NotFoundTarget::Run(run_id)).await;
                    let msg = match &e.body.hint {
                        Some(h) if !h.is_empty() => format!("{} ({h})", e.body.message),
                        _ => e.body.message.clone(),
                    };
                    return Err(McpError::invalid_params(msg, Some(json!({ "error": e.body }))));
                }
            };
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
        other => {
            let available: &[&str] = if inner.opts.read_only {
                &["diagnose_run"]
            } else {
                &["train_on_folder", "label_folder", "diagnose_run"]
            };
            let mut msg = format!("unknown prompt: {}", tools::short(other, 64));
            if let Some(s) = tools::closest_match(other, available) {
                msg.push_str(&format!("; did you mean `{s}`?"));
            }
            Err(McpError::invalid_params(msg, None))
        }
    }
}
