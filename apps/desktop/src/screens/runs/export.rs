//! Run Detail metrics export (per-epoch CSV, per-class AP CSV, full JSON) and shell-safe
//! checkpoint CLI quick-copy (`folo val` / `folo predict`, matching `docs/CLI.md`).

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{
    BestEpoch, Dataset, Epoch, Loss, Model, ModelSource, ResolvedTrain, Run, RunStatus, TrainConfig,
};
use crate::app::{FoloApp, act, export_default_dir, picked_save};
use crate::controls::Btn;
use crate::screens::playground::write_export_bytes;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Active tab of the Run Detail Checkpoint CLI quick-copy block.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CheckpointCliMode {
    #[default]
    Val,
    Predict,
}

/// Shell-safe `folo val` and `folo predict` commands for a run's saved checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointCliBundle {
    pub val: String,
    pub predict: String,
    /// Display label when the checkpoint path is unknown and `<model>` is used in the command.
    pub model_placeholder: Option<String>,
    /// Display label when the dataset path is unknown and `<dataset>` is used in `folo val`.
    pub data_placeholder: Option<String>,
    /// Shell-quoting note when `model.path` contains shell-unsafe characters.
    pub model_quote_note: Option<String>,
    /// Shell-quoting note when `dataset.path` contains shell-unsafe characters (Val mode only).
    pub data_quote_note: Option<String>,
}

impl CheckpointCliBundle {
    /// Returns the shell-quoting note applicable to `mode` (`Predict` never surfaces `<dataset>` notes).
    pub fn quote_note(&self, mode: CheckpointCliMode) -> Option<String> {
        match mode {
            CheckpointCliMode::Predict => self.model_quote_note.clone(),
            CheckpointCliMode::Val => match (&self.model_quote_note, &self.data_quote_note) {
                (Some(m), Some(d)) => Some(format!("{m} {d}")),
                (Some(m), None) => Some(m.clone()),
                (None, Some(d)) => Some(d.clone()),
                (None, None) => None,
            },
        }
    }
}

/// One row of the latest epoch's per-class AP export, sorted best AP50-95 first.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RunPerClassExportRow {
    pub class: u32,
    pub name: String,
    pub map50_95: f32,
    pub map50: f32,
    pub precision: f32,
    pub recall: f32,
    pub delta_prev: Option<f32>,
}

fn class_name(classes: &[String], id: u32) -> String {
    classes.get(id as usize).cloned().unwrap_or_else(|| format!("class {id}"))
}

/// Computes the latest epoch's per-class AP table sorted by AP50-95 descending (ties by class id),
/// including `delta_prev` against the previous epoch when present.
pub fn build_latest_per_class(epochs: &[Epoch], classes: &[String]) -> Vec<RunPerClassExportRow> {
    let Some(last) = epochs.last() else { return Vec::new() };
    if last.val.per_class.is_empty() {
        return Vec::new();
    }
    let prev = epochs.len().checked_sub(2).map(|i| &epochs[i].val.per_class);
    let mut rows: Vec<RunPerClassExportRow> = last
        .val
        .per_class
        .iter()
        .map(|c| {
            let delta_prev =
                prev.and_then(|p| p.iter().find(|x| x.class == c.class)).map(|p| c.map50_95 - p.map50_95);
            RunPerClassExportRow {
                class: c.class,
                name: class_name(classes, c.class),
                map50_95: c.map50_95,
                map50: c.map50,
                precision: c.precision,
                recall: c.recall,
                delta_prev,
            }
        })
        .collect();
    rows.sort_by(|a, b| b.map50_95.total_cmp(&a.map50_95).then(a.class.cmp(&b.class)));
    rows
}

fn fmt_csv_f32(v: f32) -> String {
    if v.is_finite() { format!("{v}") } else { String::new() }
}

fn fmt_csv_f64(v: f64) -> String {
    if v.is_finite() { format!("{v}") } else { String::new() }
}

/// Formats real loaded per-epoch training & validation metrics as RFC 4180 CSV with LF endings.
pub fn build_run_metrics_csv(epochs: &[Epoch]) -> String {
    let mut out = String::from(
        "epoch,duration_s,train_box_loss,train_cls_loss,train_dfl_loss,train_total_loss,val_map50,val_map50_95,val_precision,val_recall,val_best_f1_conf\n",
    );
    for e in epochs {
        let best_f1 = e.val.best_f1_conf.map_or_else(String::new, fmt_csv_f32);
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            e.epoch,
            fmt_csv_f64(e.duration_s),
            fmt_csv_f32(e.train_loss.box_),
            fmt_csv_f32(e.train_loss.cls),
            fmt_csv_f32(e.train_loss.dfl),
            fmt_csv_f32(e.train_loss.total),
            fmt_csv_f32(e.val.map50),
            fmt_csv_f32(e.val.map50_95),
            fmt_csv_f32(e.val.precision),
            fmt_csv_f32(e.val.recall),
            best_f1,
        ));
    }
    out
}

/// Formats the latest epoch's per-class AP table as RFC 4180 CSV with LF endings.
pub fn build_run_per_class_csv(epochs: &[Epoch], classes: &[String]) -> String {
    let mut out = String::from("class_id,class_name,ap50_95,ap50,precision,recall,delta_prev\n");
    for r in build_latest_per_class(epochs, classes) {
        out.push_str(&r.class.to_string());
        out.push(',');
        push_csv_safe_text(&mut out, &r.name);
        out.push_str(&format!(
            ",{},{},{},{},{}\n",
            fmt_csv_f32(r.map50_95),
            fmt_csv_f32(r.map50),
            fmt_csv_f32(r.precision),
            fmt_csv_f32(r.recall),
            r.delta_prev.map_or_else(String::new, fmt_csv_f32),
        ));
    }
    out
}

/// Builds the structured JSON export payload combining run metadata, request config, resolved
/// settings, per-epoch metrics, and per-class AP.
pub fn build_run_metrics_json(
    r: &Run,
    epochs: &[Epoch],
    classes: &[String],
    dataset_name: Option<&str>,
) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct RunMeta<'a> {
        id: &'a str,
        name: &'a str,
        status: RunStatus,
        device: &'a str,
        dataset_id: &'a str,
        dataset_name: Option<&'a str>,
        model_id: Option<&'a str>,
        created: &'a str,
        started: Option<&'a str>,
        finished: Option<&'a str>,
        epoch: u32,
        epochs: u32,
        best: Option<&'a BestEpoch>,
    }
    #[derive(serde::Serialize)]
    struct ValExport {
        map50: f32,
        map50_95: f32,
        precision: f32,
        recall: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        best_f1_conf: Option<f32>,
    }
    #[derive(serde::Serialize)]
    struct ClassExport {
        class: u32,
        name: String,
        map50: f32,
        map50_95: f32,
        precision: f32,
        recall: f32,
    }
    #[derive(serde::Serialize)]
    struct EpochExport<'a> {
        epoch: u32,
        duration_s: f64,
        train_loss: &'a Loss,
        val: ValExport,
        per_class: Vec<ClassExport>,
    }
    #[derive(serde::Serialize)]
    struct Payload<'a> {
        run: RunMeta<'a>,
        config: &'a TrainConfig,
        resolved: Option<&'a ResolvedTrain>,
        epochs: Vec<EpochExport<'a>>,
        latest_per_class: Vec<RunPerClassExportRow>,
    }

    let resolved_ds = dataset_name.or_else(|| r.dataset.as_ref().map(|d| d.name.as_str()));
    let payload = Payload {
        run: RunMeta {
            id: &r.id,
            name: &r.name,
            status: r.status,
            device: &r.device,
            dataset_id: &r.config.dataset_id,
            dataset_name: resolved_ds,
            model_id: r.model_id.as_deref(),
            created: &r.created,
            started: r.started.as_deref(),
            finished: r.finished.as_deref(),
            epoch: r.epoch,
            epochs: r.epochs,
            best: r.best.as_ref(),
        },
        config: &r.config,
        resolved: r.resolved.as_ref(),
        epochs: epochs
            .iter()
            .map(|e| EpochExport {
                epoch: e.epoch,
                duration_s: e.duration_s,
                train_loss: &e.train_loss,
                val: ValExport {
                    map50: e.val.map50,
                    map50_95: e.val.map50_95,
                    precision: e.val.precision,
                    recall: e.val.recall,
                    best_f1_conf: e.val.best_f1_conf,
                },
                per_class: e
                    .val
                    .per_class
                    .iter()
                    .map(|c| ClassExport {
                        class: c.class,
                        name: class_name(classes, c.class),
                        map50: c.map50,
                        map50_95: c.map50_95,
                        precision: c.precision,
                        recall: c.recall,
                    })
                    .collect(),
            })
            .collect(),
        latest_per_class: build_latest_per_class(epochs, classes),
    };
    Ok(serde_json::to_string_pretty(&payload)?)
}

/// Suggested filename for a run export (`<slug>-metrics.csv`, `<slug>-per-class.csv`, `<slug>-metrics.json`).
pub fn run_export_filename(r: &Run, suffix: &str, ext: &str) -> String {
    let raw = if r.name.trim().is_empty() { r.id.as_str() } else { r.name.as_str() };
    let mut slug = String::with_capacity(raw.len());
    let mut prev_dash = false;
    for ch in raw.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            slug.push(lower);
            prev_dash = false;
        } else if !prev_dash && !slug.is_empty() {
            slug.push('-');
            prev_dash = true;
        }
    }
    let mut slug: String = slug.chars().take(64).collect();
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug = "run".to_string();
    }
    format!("{slug}-{suffix}.{ext}")
}

/// Builds shell-safe `folo val` and `folo predict` commands for a run's saved checkpoint
/// (`docs/CLI.md`). Returns `None` when the run has no saved checkpoint model yet.
pub fn build_checkpoint_cli(
    r: &Run,
    model: Option<&Model>,
    dataset: Option<&Dataset>,
    eval_split: &str,
) -> Option<CheckpointCliBundle> {
    let _mid = r.model_id.as_deref()?;
    let mut model_quote_note: Option<String> = None;
    let mut data_quote_note: Option<String> = None;
    let mut model_placeholder: Option<String> = None;

    let model_arg = if let Some(path) = model.and_then(|m| m.path.as_deref()).filter(|p| !p.trim().is_empty())
    {
        match folo_api::json::quote_path(path) {
            Some(q) => q,
            None => {
                model_quote_note =
                    Some(format!("Replace <model> with this path, quoted for your shell: {path}"));
                "<model>".to_string()
            }
        }
    } else if let Some(m) = model
        && m.source == Some(ModelSource::Published)
        && let Some(q) = folo_api::json::quote_path(&m.id)
    {
        q
    } else {
        let fallback = model
            .map(|m| m.name.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| if r.name.trim().is_empty() { r.id.as_str() } else { r.name.as_str() });
        model_placeholder = Some(format!("{fallback}.folo"));
        // Keep clipboard commands free of unescaped display names; show the label only in hints.
        "<model>".to_string()
    };

    let mut data_placeholder: Option<String> = None;
    let data_arg = if let Some(path) = dataset.map(|d| d.path.as_str()).filter(|p| !p.trim().is_empty()) {
        match folo_api::json::quote_path(path) {
            Some(q) => q,
            None => {
                data_quote_note =
                    Some(format!("Replace <dataset> with this path, quoted for your shell: {path}"));
                "<dataset>".to_string()
            }
        }
    } else {
        let ds_label = r
            .dataset
            .as_ref()
            .map(|d| d.name.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(r.config.dataset_id.as_str());
        data_placeholder = Some(ds_label.to_string());
        "<dataset>".to_string()
    };

    let mut val_parts = vec![
        "folo val".to_string(),
        "--model".to_string(),
        model_arg.clone(),
        "--data".to_string(),
        data_arg,
    ];
    if eval_split == "train" {
        val_parts.push("--split train".to_string());
    }
    let predict_parts =
        ["folo predict".to_string(), "--model".to_string(), model_arg, "<images>".to_string()];

    Some(CheckpointCliBundle {
        val: val_parts.join(" "),
        predict: predict_parts.join(" "),
        model_placeholder,
        data_placeholder,
        model_quote_note,
        data_quote_note,
    })
}

impl FoloApp {
    /// Class names for `r`: prefers a non-empty run snapshot, then the saved checkpoint's class
    /// order, then the registered dataset summary (skipping empty legacy snapshots).
    pub(crate) fn run_classes(&self, r: &Run) -> Vec<String> {
        r.dataset
            .as_ref()
            .filter(|d| !d.classes.is_empty())
            .map(|d| d.classes.clone())
            .or_else(|| {
                r.model_id
                    .as_deref()
                    .and_then(|mid| self.store.model(mid))
                    .filter(|m| !m.classes.is_empty())
                    .map(|m| m.classes.clone())
            })
            .or_else(|| {
                self.store
                    .dataset_summary(&r.config.dataset_id)
                    .filter(|d| !d.classes.is_empty())
                    .map(|d| d.classes.clone())
            })
            .unwrap_or_default()
    }

    /// Checkpoint CLI commands (`folo val` / `folo predict`) for `r`, if it has a saved model.
    pub(crate) fn checkpoint_cli_for(&self, r: &Run) -> Option<CheckpointCliBundle> {
        let model = r.model_id.as_deref().and_then(|mid| self.store.model(mid));
        let dataset = self.store.dataset_summary(&r.config.dataset_id);
        let split = self.eval_split(r);
        build_checkpoint_cli(r, model, dataset, split)
    }

    /// Prompts for a save path and exports per-epoch metrics as CSV.
    pub fn export_run_metrics_csv(&mut self, run_id: &str, cx: &mut Context<Self>) {
        let Some(r) = self.store.run(run_id).cloned() else { return };
        let epochs: Vec<Epoch> = self
            .store
            .run_data
            .get(run_id)
            .and_then(|l| l.ready())
            .map(|d| d.metrics.epochs.clone())
            .unwrap_or_default();
        if epochs.is_empty() {
            self.toast(
                Tone::Neutral,
                "Nothing to export yet",
                Some("Wait for at least one evaluated epoch first.".into()),
                cx,
            );
            return;
        }
        let csv = build_run_metrics_csv(&epochs);
        let suggested = run_export_filename(&r, "metrics", "csv");
        let summary = counted(epochs.len() as u64, "epoch", "epochs");
        self.save_run_export_file(suggested, csv, "Exported run metrics CSV", summary, cx);
    }

    /// Prompts for a save path and exports the latest epoch's per-class AP table as CSV.
    pub fn export_run_per_class_csv(&mut self, run_id: &str, cx: &mut Context<Self>) {
        let Some(r) = self.store.run(run_id).cloned() else { return };
        let epochs: Vec<Epoch> = self
            .store
            .run_data
            .get(run_id)
            .and_then(|l| l.ready())
            .map(|d| d.metrics.epochs.clone())
            .unwrap_or_default();
        let classes = self.run_classes(&r);
        let rows = build_latest_per_class(&epochs, &classes);
        if rows.is_empty() {
            self.toast(
                Tone::Neutral,
                "No per-class metrics to export",
                Some("Wait for an evaluated epoch with per-class AP metrics.".into()),
                cx,
            );
            return;
        }
        let csv = build_run_per_class_csv(&epochs, &classes);
        let suggested = run_export_filename(&r, "per-class", "csv");
        let summary = counted(rows.len() as u64, "class", "classes");
        self.save_run_export_file(suggested, csv, "Exported per-class AP CSV", summary, cx);
    }

    /// Prompts for a save path and exports full run config, epochs, and per-class AP as JSON.
    pub fn export_run_metrics_json(&mut self, run_id: &str, cx: &mut Context<Self>) {
        let Some(r) = self.store.run(run_id).cloned() else { return };
        let epochs: Vec<Epoch> = self
            .store
            .run_data
            .get(run_id)
            .and_then(|l| l.ready())
            .map(|d| d.metrics.epochs.clone())
            .unwrap_or_default();
        if epochs.is_empty() {
            self.toast(
                Tone::Neutral,
                "Nothing to export yet",
                Some("Wait for at least one evaluated epoch first.".into()),
                cx,
            );
            return;
        }
        let classes = self.run_classes(&r);
        let ds_name = self.store.run_dataset_name(&r);
        let json = match build_run_metrics_json(&r, &epochs, &classes, Some(&ds_name)) {
            Ok(s) => s,
            Err(e) => {
                self.toast(Tone::Danger, "JSON export failed", Some(e.to_string().into()), cx);
                return;
            }
        };
        let suggested = run_export_filename(&r, "metrics", "json");
        let summary = counted(epochs.len() as u64, "epoch", "epochs");
        self.save_run_export_file(suggested, json, "Exported run metrics JSON", summary, cx);
    }

    fn save_run_export_file(
        &mut self,
        suggested: String,
        content: String,
        ok_title: &'static str,
        summary: String,
        cx: &mut Context<Self>,
    ) {
        let dir = export_default_dir();
        let rx = cx.prompt_for_new_path(&dir, Some(&suggested));
        cx.spawn(async move |this, cx| {
            let Some(path) = picked_save(&this, rx.await, cx) else { return };
            let write_path = path.clone();
            let res = cx
                .background_executor()
                .spawn(async move { write_export_bytes(&write_path, content.as_bytes()) })
                .await;
            this.update(cx, |this, cx| match res {
                Ok(()) => {
                    let reveal = path.clone();
                    let shown = display_path(&path.to_string_lossy());
                    this.toast_with(
                        Tone::Success,
                        ok_title,
                        Some(format!("{shown} · {summary}").into()),
                        Some(("Show in folder".into(), act(move |_, _, cx| cx.reveal_path(&reveal)))),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(Tone::Danger, "Export failed", Some(e.to_string().into()), cx);
                }
            })
            .ok();
        })
        .detach();
    }

    /// Copies the run's checkpoint `folo val` or `folo predict` CLI command to the clipboard.
    pub fn copy_run_checkpoint_cli(&mut self, run_id: &str, mode: CheckpointCliMode, cx: &mut Context<Self>) {
        let Some(r) = self.store.run(run_id).cloned() else { return };
        let Some(bundle) = self.checkpoint_cli_for(&r) else {
            self.toast(
                Tone::Neutral,
                "No checkpoint saved yet",
                Some("A checkpoint command is ready once the run saves weights.".into()),
                cx,
            );
            return;
        };
        let quote_note = bundle.quote_note(mode);
        let (cmd, title) = match mode {
            CheckpointCliMode::Val => (bundle.val, "Copied val CLI command"),
            CheckpointCliMode::Predict => (bundle.predict, "Copied predict CLI command"),
        };
        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(cmd));
        let body = quote_note.or_else(|| {
            Some(match mode {
                CheckpointCliMode::Val => "Run in a terminal to evaluate this checkpoint.".into(),
                CheckpointCliMode::Predict => "Replace <images> with an image file or folder path.".into(),
            })
        });
        self.toast(Tone::Success, title, body.map(Into::into), cx);
    }

    /// Compact export footer inside the Per class panel when evaluated epochs are loaded.
    pub(super) fn run_metrics_export_footer(
        &self,
        r: &Run,
        epoch_count: usize,
        has_per_class: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let rid_csv = r.id.clone();
        let rid_pc = r.id.clone();
        let rid_json = r.id.clone();
        let csv_btn = self
            .button(
                "run-export-csv",
                "Metrics CSV",
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(move |this, _, cx| this.export_run_metrics_csv(&rid_csv, cx)),
            )
            .tooltip(crate::controls::tip(
                format!(
                    "Save {} of training & validation metrics as CSV",
                    counted(epoch_count as u64, "epoch", "epochs")
                ),
                None,
            ));
        let pc_btn = has_per_class.then(|| {
            self.button(
                "run-export-per-class-csv",
                "Classes CSV",
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(move |this, _, cx| this.export_run_per_class_csv(&rid_pc, cx)),
            )
            .tooltip(crate::controls::tip("Save latest epoch per-class AP table as CSV", None))
        });
        let json_btn = self
            .button(
                "run-export-json",
                "JSON",
                Btn::Secondary,
                Some("icons/file-text.svg"),
                window,
                cx,
                act(move |this, _, cx| this.export_run_metrics_json(&rid_json, cx)),
            )
            .tooltip(crate::controls::tip(
                "Save full run config, epoch metrics, and per-class AP as JSON",
                None,
            ));
        div()
            .w_full()
            .flex_none()
            .border_t_1()
            .border_color(t.line_subtle)
            .px(px(12.))
            .py(px(8.))
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(6.))
            .child(csv_btn)
            .children(pc_btn)
            .child(json_btn)
            .into_any_element()
    }

    /// Checkpoint CLI quick-copy panel shown below Logs when the run has a saved checkpoint model
    /// (mirrors Studio `#790`'s `CheckpointCli` val / predict command block).
    pub(super) fn run_checkpoint_cli_panel(
        &self,
        r: &Run,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let bundle = self.checkpoint_cli_for(r)?;
        let t = theme(cx);
        let mode = self.run_chart_opts.ckpt_cli_mode;
        let mode_idx = usize::from(mode == CheckpointCliMode::Predict);
        let seg = self.segmented(
            "run-ckpt-cli-mode",
            &["Validate", "Predict"],
            mode_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.run_chart_opts.ckpt_cli_mode =
                    if i == 1 { CheckpointCliMode::Predict } else { CheckpointCliMode::Val };
                cx.notify();
            }),
        );
        let rid = r.id.clone();
        let copy_label = match mode {
            CheckpointCliMode::Val => "Copy val CLI",
            CheckpointCliMode::Predict => "Copy predict CLI",
        };
        let copy_btn = self
            .button(
                "run-ckpt-cli-copy",
                copy_label,
                Btn::Secondary,
                Some("icons/copy.svg"),
                window,
                cx,
                act(move |this, _, cx| this.copy_run_checkpoint_cli(&rid, mode, cx)),
            )
            .tooltip(crate::controls::tip("Copy command to clipboard", None));
        let cmd = match mode {
            CheckpointCliMode::Val => bundle.val.clone(),
            CheckpointCliMode::Predict => bundle.predict.clone(),
        };
        let mut placeholders = Vec::new();
        if let Some(m) = &bundle.model_placeholder {
            placeholders.push(format!("<model> (path to {m})"));
        }
        if mode == CheckpointCliMode::Val
            && let Some(d) = &bundle.data_placeholder
        {
            placeholders.push(format!("<dataset> (path to {d})"));
        }
        if mode == CheckpointCliMode::Predict {
            placeholders.push("<images>".to_string());
        }
        let hint = (!placeholders.is_empty()).then(|| {
            let noun = if placeholders.len() == 1 { "a local path" } else { "local paths" };
            format!("Replace {} with {noun}.", placeholders.join(" and "))
        });
        let quote_note = bundle.quote_note(mode);

        Some(
            panel(&t)
                .flex_none()
                .flex()
                .flex_col()
                .child(panel_header(&t, "Checkpoint CLI").child(seg))
                .child(
                    div()
                        .p(px(12.))
                        .flex()
                        .flex_col()
                        .gap(px(8.))
                        .child(
                            div()
                                .px(px(10.))
                                .py(px(8.))
                                .rounded(px(radius::MD))
                                .bg(t.bg_inset)
                                .border_1()
                                .border_color(t.line_subtle)
                                .child(
                                    text(ty::SM, t.text_primary)
                                        .font_family(crate::theme::MONO)
                                        .whitespace_normal()
                                        .child(cmd),
                                ),
                        )
                        .when_some(hint, |d, h| {
                            d.child(text(ty::XS, t.text_tertiary).whitespace_normal().child(h))
                        })
                        .when_some(quote_note, |d, n| {
                            d.child(text(ty::XS, t.warning_text).whitespace_normal().child(n))
                        })
                        .child(div().flex().justify_end().child(copy_btn)),
                )
                .into_any_element(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, mock::MockBackend};

    #[test]
    fn run_metrics_csv_per_class_csv_and_json_export_real_loaded_epochs() {
        let api = MockBackend::new(4200, true);
        let run = api.runs().unwrap().into_iter().find(|r| r.id == "run_52e8").unwrap();
        let metrics = api.metrics(&run.id).unwrap();
        let ds = api.datasets().unwrap().into_iter().find(|d| d.id == run.config.dataset_id).unwrap();

        let csv = build_run_metrics_csv(&metrics.epochs);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "epoch,duration_s,train_box_loss,train_cls_loss,train_dfl_loss,train_total_loss,val_map50,val_map50_95,val_precision,val_recall,val_best_f1_conf"
        );
        assert_eq!(lines.len(), metrics.epochs.len() + 1);

        let pc_csv = build_run_per_class_csv(&metrics.epochs, &ds.classes);
        let pc_lines: Vec<&str> = pc_csv.lines().collect();
        assert_eq!(pc_lines[0], "class_id,class_name,ap50_95,ap50,precision,recall,delta_prev");
        assert!(pc_lines.len() >= 2);

        let json_str = build_run_metrics_json(&run, &metrics.epochs, &ds.classes, Some(&ds.name)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["run"]["id"], run.id);
        assert_eq!(v["run"]["dataset_name"], ds.name);
        assert_eq!(v["epochs"].as_array().unwrap().len(), metrics.epochs.len());
        assert!(!v["latest_per_class"].as_array().unwrap().is_empty());
        assert_eq!(v["config"]["epochs"], run.config.epochs);

        // Formula-like class labels are neutralized in CSV while preserved verbatim in JSON.
        let formula_classes = vec![
            "=HYPERLINK(\"http://example.com\")".into(),
            "+1+2".into(),
            " -3".into(),
            "@SUM(A1:A2)".into(),
        ];
        let safe_csv = build_run_per_class_csv(&metrics.epochs, &formula_classes);
        assert!(safe_csv.contains("\"'=HYPERLINK(\"\"http://example.com\"\")\""), "{safe_csv}");
        assert!(safe_csv.contains(",'+1+2,"), "{safe_csv}");
        assert!(safe_csv.contains(",' -3,"), "{safe_csv}");
        let raw_json: serde_json::Value = serde_json::from_str(
            &build_run_metrics_json(&run, &metrics.epochs, &formula_classes, Some(&ds.name)).unwrap(),
        )
        .unwrap();
        let names: Vec<&str> = raw_json["latest_per_class"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|row| row["name"].as_str())
            .collect();
        assert!(names.contains(&"=HYPERLINK(\"http://example.com\")"), "{names:?}");
    }

    #[test]
    fn checkpoint_cli_quotes_paths_and_handles_train_split_and_unsafe_chars() {
        let api = MockBackend::new(4200, true);
        let mut run = api.runs().unwrap().into_iter().find(|r| r.model_id.is_some()).unwrap();
        let mut model =
            api.models().unwrap().into_iter().find(|m| Some(&m.id) == run.model_id.as_ref()).unwrap();
        let mut ds = api.datasets().unwrap().into_iter().find(|d| d.id == run.config.dataset_id).unwrap();
        model.path = Some("D:/my models/best.folo".into());
        ds.path = "D:/my data/shapes".into();

        let bundle = build_checkpoint_cli(&run, Some(&model), Some(&ds), "val").unwrap();
        assert_eq!(bundle.val, "folo val --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\"");
        assert_eq!(bundle.predict, "folo predict --model \"D:/my models/best.folo\" <images>");
        assert!(bundle.quote_note(CheckpointCliMode::Val).is_none());
        assert!(bundle.quote_note(CheckpointCliMode::Predict).is_none());

        let train_split = build_checkpoint_cli(&run, Some(&model), Some(&ds), "train").unwrap();
        assert!(train_split.val.ends_with(" --split train"), "{}", train_split.val);

        // Dataset-only shell-unsafe `$` path warns in Val mode but never leaks into Predict mode.
        ds.path = "D:/bad$ds/shapes".into();
        let ds_unsafe = build_checkpoint_cli(&run, Some(&model), Some(&ds), "val").unwrap();
        assert!(ds_unsafe.val.contains("--data <dataset>"));
        assert!(
            ds_unsafe.quote_note(CheckpointCliMode::Val).as_deref().is_some_and(|n| n.contains("bad$ds"))
        );
        assert!(ds_unsafe.quote_note(CheckpointCliMode::Predict).is_none());

        // Shell-unsafe `$` model path falls back to `<model>` and warns in both modes.
        model.path = Some("D:/bad$dir/best.folo".into());
        let unsafe_b = build_checkpoint_cli(&run, Some(&model), Some(&ds), "val").unwrap();
        assert!(unsafe_b.val.contains("--model <model>"));
        assert!(unsafe_b.predict.contains("--model <model>"));
        assert!(
            unsafe_b.quote_note(CheckpointCliMode::Val).as_deref().is_some_and(|n| n.contains("bad$dir"))
        );
        assert!(
            unsafe_b.quote_note(CheckpointCliMode::Predict).as_deref().is_some_and(|n| n.contains("bad$dir"))
        );

        // Missing paths with shell metacharacters in display names never interpolate into commands.
        model.path = None;
        model.source = None;
        model.name = "run $(whoami); `id`".into();
        ds.path = String::new();
        if let Some(snap) = run.dataset.as_mut() {
            snap.name = "ds $(rm -rf /)".into();
        }
        let missing = build_checkpoint_cli(&run, Some(&model), Some(&ds), "val").unwrap();
        assert_eq!(missing.val, "folo val --model <model> --data <dataset>");
        assert_eq!(missing.predict, "folo predict --model <model> <images>");
        assert!(missing.model_placeholder.as_deref().is_some_and(|s| s.contains("whoami")));
        assert!(missing.data_placeholder.as_deref().is_some_and(|s| s.contains("rm -rf")));

        // Run without a saved checkpoint returns None.
        run.model_id = None;
        assert!(build_checkpoint_cli(&run, Some(&model), Some(&ds), "val").is_none());
    }

    #[test]
    fn run_detail_exports_csv_json_and_copies_checkpoint_cli_end_to_end() {
        use crate::app::Route;
        use crate::app::tests::{drive, open};

        let (mut app, window) = open();
        let dir = folo_test_support::TestDir::new("desktop-run-export-ui");
        let rid = "run_52e8";
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(rid.into()), w, cx));
        for _ in 0..3 {
            app.frame(window.into());
            app.run_until_idle();
        }
        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            for expected in [
                "run-export-csv",
                "run-export-per-class-csv",
                "run-export-json",
                "run-ckpt-cli-mode",
                "run-ckpt-cli-copy",
            ] {
                assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
            }
        });
        // Switching Checkpoint CLI mode toggles between Validate and Predict.
        drive(&mut app, window, |this, _, cx| {
            this.run_chart_opts.ckpt_cli_mode = CheckpointCliMode::Predict;
            cx.notify();
        });
        app.frame(window.into());

        let csv_path = dir.path().join("metrics.csv");
        app.platform().answer_prompts(Some(vec![csv_path.clone()]));
        drive(&mut app, window, |this, _, cx| this.export_run_metrics_csv(rid, cx));
        app.run_until_idle();
        let csv_text = std::fs::read_to_string(&csv_path).expect("metrics CSV written");
        assert!(csv_text.starts_with("epoch,duration_s,train_box_loss,"));

        let pc_path = dir.path().join("per-class.csv");
        app.platform().answer_prompts(Some(vec![pc_path.clone()]));
        drive(&mut app, window, |this, _, cx| this.export_run_per_class_csv(rid, cx));
        app.run_until_idle();
        let pc_text = std::fs::read_to_string(&pc_path).expect("per-class CSV written");
        assert!(pc_text.starts_with("class_id,class_name,ap50_95,"));

        let json_path = dir.path().join("metrics.json");
        app.platform().answer_prompts(Some(vec![json_path.clone()]));
        drive(&mut app, window, |this, _, cx| this.export_run_metrics_json(rid, cx));
        app.run_until_idle();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).expect("valid JSON");
        assert_eq!(v["run"]["id"], rid);
        assert!(!v["epochs"].as_array().unwrap().is_empty());

        // Checkpoint CLI quick-copy writes exact command to clipboard and raises a success toast.
        drive(&mut app, window, |this, _, cx| {
            this.copy_run_checkpoint_cli(rid, CheckpointCliMode::Val, cx);
            assert!(this.toasts.iter().any(|t| t.title == "Copied val CLI command"));
            this.copy_run_checkpoint_cli(rid, CheckpointCliMode::Predict, cx);
            assert!(this.toasts.iter().any(|t| t.title == "Copied predict CLI command"));
        });
    }
}
