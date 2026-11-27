//! Interactive Export & Inference CLI Command Builder on the Model Detail screen.
//!
//! Generates exact, shell-safe `folo export`, `folo quantize`, and `folo predict` commands
//! matching `docs/CLI.md`, with one-click clipboard copy + toast feedback and direct server
//! export when the selected format is served by `POST /api/models/{id}/export`.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window,
    div, px,
};

use crate::api::{Head, Model, ModelExportOptions, ModelSource};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Active tab of the Model Detail CLI builder.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ModelCliMode {
    #[default]
    Export,
    Predict,
}

/// Export format / workflow in the Model Detail CLI builder.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CliExportFormat {
    #[default]
    Onnx,
    FoloF16,
    FoloF32,
    Weights,
    Int8Quantize,
}

impl CliExportFormat {
    pub const ALL: [Self; 5] = [Self::Onnx, Self::FoloF16, Self::FoloF32, Self::Weights, Self::Int8Quantize];

    pub fn label(self) -> &'static str {
        match self {
            Self::Onnx => "ONNX",
            Self::FoloF16 => "FOLO f16",
            Self::FoloF32 => "FOLO f32",
            Self::Weights => "Weights",
            Self::Int8Quantize => "INT8 quantize",
        }
    }
}

/// Calibration method for `folo quantize`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum QuantMethod {
    #[default]
    Percentile,
    MinMax,
    Mse,
}

impl QuantMethod {
    pub const ALL: [Self; 3] = [Self::Percentile, Self::MinMax, Self::Mse];

    pub fn label(self) -> &'static str {
        match self {
            Self::Percentile => "percentile",
            Self::MinMax => "minmax",
            Self::Mse => "mse",
        }
    }
}

/// Precision option for `folo predict`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PredictPrecision {
    #[default]
    F32,
    F16,
    Int8,
}

impl PredictPrecision {
    pub const ALL: [Self; 3] = [Self::F32, Self::F16, Self::Int8];

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Int8 => "int8",
        }
    }
}

/// Slider ranges `(min, max, step)` for `folo predict` thresholds.
pub const MODEL_CLI_CONF: (f32, f32, f32) = (0.05, 0.95, 0.05);
pub const MODEL_CLI_IOU: (f32, f32, f32) = (0.30, 0.95, 0.05);

/// Interactive configuration state for the Model Detail Export & CLI builder.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelCliState {
    pub mode: ModelCliMode,
    pub export_format: CliExportFormat,
    /// ONNX symbolic batch axis (`--dynamic-batch`, or `--dynamic` when combined with `onnx_dynamic_hw`).
    pub onnx_dynamic_batch: bool,
    /// ONNX symbolic height/width axes (`--dynamic-hw`, or `--dynamic` when combined with `onnx_dynamic_batch`).
    pub onnx_dynamic_hw: bool,
    /// ONNX in-graph box decode (`--decode`).
    pub onnx_decode: bool,
    /// ONNX f16 weights & activations (`--half`).
    pub onnx_half: bool,
    /// For `o2o` models, export the `o2m` branch instead (`--head o2m`).
    pub onnx_head_o2m: bool,
    /// INT8 calibration method (`folo quantize --method`).
    pub quant_method: QuantMethod,
    /// SAHI tiled inference (`--tile 640`, mutually exclusive with `predict_tta`).
    pub predict_sahi: bool,
    /// Test-Time Augmentation (`--tta`, mutually exclusive with `predict_sahi`).
    pub predict_tta: bool,
    /// ByteTrack multi-object tracking (`--track`).
    pub predict_track: bool,
    /// Inference precision (`f32`, `f16`, `int8`).
    pub predict_precision: PredictPrecision,
    /// Confidence threshold (`--conf`, default 0.25).
    pub predict_conf: f32,
    /// NMS IoU threshold (`--iou`, default 0.70, unused for `o2o`).
    pub predict_iou: f32,
}

impl Default for ModelCliState {
    fn default() -> Self {
        Self {
            mode: ModelCliMode::Export,
            export_format: CliExportFormat::Onnx,
            onnx_dynamic_batch: false,
            onnx_dynamic_hw: false,
            onnx_decode: false,
            onnx_half: false,
            onnx_head_o2m: false,
            quant_method: QuantMethod::Percentile,
            predict_sahi: false,
            predict_tta: false,
            predict_track: false,
            predict_precision: PredictPrecision::F32,
            predict_conf: 0.25,
            predict_iou: 0.70,
        }
    }
}

/// Generated command and metadata ready for display, copy, and optional server export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltCliCommand {
    /// Exact copyable CLI command line.
    pub command: String,
    /// Shell-quoting note if a model or dataset path contains shell-unsafe characters.
    pub quote_note: Option<String>,
    /// Context note (e.g. opset 17 info, fused checkpoint warning, or o2o NMS note).
    pub hint: Option<String>,
    /// Server-side export options when the current export configuration is supported by
    /// `POST /api/models/{id}/export`.
    pub server_export: Option<ModelExportOptions>,
}

/// Resolves the `--model` argument safely for pasting into PowerShell, cmd, or POSIX shells.
/// Returns `(arg_token, unquotable_raw_path)`.
fn resolve_model_arg(m: &Model) -> (String, Option<String>) {
    if let Some(path) = m.path.as_deref().filter(|p| !p.trim().is_empty()) {
        return match folo_api::json::quote_path(path) {
            Some(quoted) => (quoted, None),
            None => ("<model>".to_string(), Some(path.to_string())),
        };
    }
    if m.source == Some(ModelSource::Published)
        && let Some(quoted) = folo_api::json::quote_path(&m.id)
    {
        return (quoted, None);
    }
    ("<model.folo>".to_string(), None)
}

/// Formats a threshold rounded to 2 decimal places (`0.25`, `0.70`).
fn fmt_thresh(v: f32) -> String {
    format!("{:.2}", v.clamp(0.0, 1.0))
}

/// Builds the exact, shell-safe CLI command matching `docs/CLI.md` for `m` under `state`.
pub fn build_model_cli(m: &Model, run_dataset_path: Option<&str>, state: &ModelCliState) -> BuiltCliCommand {
    let (model_arg, unquoted_model) = resolve_model_arg(m);
    let mut unquoted_notes: Vec<String> = Vec::new();
    if let Some(raw) = unquoted_model {
        unquoted_notes.push(format!("Replace <model> with this path, quoted for your shell: {raw}"));
    }

    match state.mode {
        ModelCliMode::Export => {
            let mut parts: Vec<String> = Vec::new();
            let hint: Option<String>;
            let mut server_export: Option<ModelExportOptions> = None;

            match state.export_format {
                CliExportFormat::Onnx => {
                    parts.push("folo export".into());
                    parts.push("--model".into());
                    parts.push(model_arg);
                    parts.push("--format onnx".into());
                    match (state.onnx_dynamic_batch, state.onnx_dynamic_hw) {
                        (true, true) => parts.push("--dynamic".into()),
                        (true, false) => parts.push("--dynamic-batch".into()),
                        (false, true) => parts.push("--dynamic-hw".into()),
                        (false, false) => {}
                    }
                    if state.onnx_decode {
                        parts.push("--decode".into());
                    }
                    if state.onnx_half {
                        parts.push("--half".into());
                    }
                    if m.head == Head::O2o && state.onnx_head_o2m {
                        parts.push("--head o2m".into());
                    }
                    // The server's `/api/models/{id}/export` exports the model's own head; a
                    // `--head o2m` override on an `o2o` model runs via CLI.
                    if !(m.head == Head::O2o && state.onnx_head_o2m) {
                        server_export = Some(ModelExportOptions {
                            format: "onnx",
                            decode: state.onnx_decode,
                            dynamic_batch: state.onnx_dynamic_batch,
                            dynamic_hw: state.onnx_dynamic_hw,
                            half: state.onnx_half,
                        });
                    }
                    hint = Some(if m.head == Head::O2o && !state.onnx_head_o2m && state.onnx_decode {
                        "ONNX opset 17 · end-to-end [B, K, 6] output (no NMS needed).".into()
                    } else if state.onnx_decode {
                        "ONNX opset 17 · decoded [B, A, 4 + nc] output in input pixels.".into()
                    } else {
                        "ONNX opset 17 · fused Conv+BN graph for onnxruntime, TensorRT, or OpenVINO.".into()
                    });
                }
                CliExportFormat::FoloF16 => {
                    parts.push("folo export".into());
                    parts.push("--model".into());
                    parts.push(model_arg);
                    parts.push("--format folo-f16".into());
                    server_export = Some(ModelExportOptions::plain("folo-f16"));
                    hint = Some("Fused half-precision .folo weights for fast GPU inference.".into());
                }
                CliExportFormat::FoloF32 => {
                    parts.push("folo export".into());
                    parts.push("--model".into());
                    parts.push(model_arg);
                    parts.push("--format folo".into());
                    hint = Some("Fused f32 .folo weights (run in terminal).".into());
                }
                CliExportFormat::Weights => {
                    parts.push("folo export".into());
                    parts.push("--model".into());
                    parts.push(model_arg);
                    parts.push("--format weights".into());
                    hint = Some(if m.fused == Some(true) {
                        "Unfused weights export requires a training checkpoint (this file is already fused)."
                            .into()
                    } else {
                        "Unfused EMA f32 weights without optimizer state, ready for fine-tuning or sharing."
                            .into()
                    });
                }
                CliExportFormat::Int8Quantize => {
                    let data_arg = match run_dataset_path.filter(|p| !p.trim().is_empty()) {
                        Some(p) => match folo_api::json::quote_path(p) {
                            Some(q) => q,
                            None => {
                                unquoted_notes.push(format!(
                                    "Replace <dataset> with this path, quoted for your shell: {p}"
                                ));
                                "<dataset>".to_string()
                            }
                        },
                        None => "<dataset>".to_string(),
                    };
                    parts.push("folo quantize".into());
                    parts.push("--model".into());
                    parts.push(model_arg);
                    parts.push("--data".into());
                    parts.push(data_arg);
                    if state.quant_method != QuantMethod::Percentile {
                        parts.push(format!("--method {}", state.quant_method.label()));
                    }
                    hint = Some(
                        "Calibrates CPU int8 inference (512 images by default); use with `folo predict --precision int8`."
                            .into(),
                    );
                }
            }

            BuiltCliCommand {
                command: parts.join(" "),
                quote_note: (!unquoted_notes.is_empty()).then(|| unquoted_notes.join(" ")),
                hint,
                server_export,
            }
        }
        ModelCliMode::Predict => {
            let mut parts: Vec<String> = vec!["folo predict".into(), "--model".into(), model_arg];
            parts.push("<images>".into());
            let conf = state.predict_conf.clamp(MODEL_CLI_CONF.0, MODEL_CLI_CONF.1);
            if (conf - 0.25).abs() > 0.004 {
                parts.push(format!("--conf {}", fmt_thresh(conf)));
            }
            let iou = state.predict_iou.clamp(MODEL_CLI_IOU.0, MODEL_CLI_IOU.1);
            if m.head != Head::O2o && (iou - 0.70).abs() > 0.004 {
                parts.push(format!("--iou {}", fmt_thresh(iou)));
            }
            match state.predict_precision {
                PredictPrecision::F32 => {}
                PredictPrecision::F16 => parts.push("--precision f16".into()),
                PredictPrecision::Int8 => parts.push("--precision int8".into()),
            }
            if state.predict_sahi {
                parts.push("--tile 640".into());
            } else if state.predict_tta {
                parts.push("--tta".into());
            }
            if state.predict_track {
                parts.push("--track".into());
            }
            let hint = if state.predict_precision == PredictPrecision::Int8 {
                Some("Requires a quantized .folo file calibrated with `folo quantize` (CPU only).".into())
            } else if m.head == Head::O2o {
                Some("End-to-end (o2o) head predicts without NMS; --iou is unused.".into())
            } else if state.predict_sahi {
                Some(
                    "SAHI slices large images into overlapping 640 px tiles (mutually exclusive with --tta)."
                        .into(),
                )
            } else if state.predict_tta {
                Some(
                    "TTA runs multi-scale + horizontal-flip ensembling (mutually exclusive with --tile)."
                        .into(),
                )
            } else if state.predict_track {
                Some("ByteTrack assigns persistent track IDs across ordered images or folders.".into())
            } else {
                Some("Replace <images> with an image file or folder path.".into())
            };
            BuiltCliCommand {
                command: parts.join(" "),
                quote_note: (!unquoted_notes.is_empty()).then(|| unquoted_notes.join(" ")),
                hint,
                server_export: None,
            }
        }
    }
}

impl FoloApp {
    /// Interactive Export & Inference CLI Command Builder panel on the Model Detail screen.
    pub(super) fn export_cli_panel(
        &mut self,
        m: &Model,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let run_ds_path = m
            .run_id
            .as_deref()
            .and_then(|rid| self.store.run(rid))
            .and_then(|r| self.store.dataset_summary(&r.config.dataset_id))
            .map(|d| d.path.clone());
        let built = build_model_cli(m, run_ds_path.as_deref(), &self.model_cli);
        let mode_idx = usize::from(self.model_cli.mode == ModelCliMode::Predict);
        let mode_seg = self.segmented(
            "model-cli-mode",
            &["Export", "Predict"],
            mode_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.model_cli.mode = if i == 0 { ModelCliMode::Export } else { ModelCliMode::Predict };
                cx.notify();
            }),
        );

        let controls = match self.model_cli.mode {
            ModelCliMode::Export => self.export_cli_export_controls(&t, m, window, cx),
            ModelCliMode::Predict => self.export_cli_predict_controls(&t, m, window, cx),
        };

        let copy_cmd = built.command.clone();
        let copy_note = built.quote_note.clone();
        let copy_title = match self.model_cli.mode {
            ModelCliMode::Export if self.model_cli.export_format == CliExportFormat::Int8Quantize => {
                "Copied quantize command"
            }
            ModelCliMode::Export => "Copied export command",
            ModelCliMode::Predict => "Copied predict command",
        };
        let copy_btn = self.button(
            "model-cli-copy",
            "Copy command",
            if built.server_export.is_some() { Btn::Secondary } else { Btn::Primary },
            Some("icons/copy.svg"),
            window,
            cx,
            act(move |this, _, cx| {
                cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(copy_cmd.clone()));
                this.toast(Tone::Success, copy_title, copy_note.clone().map(Into::into), cx);
            }),
        );

        let server_btn = built.server_export.map(|opts| {
            let mid = m.id.clone();
            let busy = self.exporting.contains(&(mid.clone(), opts.format));
            if busy {
                self.button_disabled(
                    "model-cli-server-export",
                    "Exporting…",
                    Btn::Primary,
                    Some("icons/download.svg"),
                    "An export in this format is already running",
                    window,
                    cx,
                )
                .into_any_element()
            } else {
                let label = if opts.format == "onnx" { "Export ONNX now" } else { "Export f16 now" };
                self.button(
                    "model-cli-server-export",
                    label,
                    Btn::Primary,
                    Some("icons/download.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.export_model_with_options(mid.clone(), opts, cx)),
                )
                .into_any_element()
            }
        });

        let header_badge = match self.model_cli.mode {
            ModelCliMode::Export if self.model_cli.export_format == CliExportFormat::Onnx => {
                Some(badge(&t, "opset 17", Tone::Neutral))
            }
            ModelCliMode::Export if self.model_cli.export_format == CliExportFormat::Int8Quantize => {
                Some(badge(&t, "CPU int8", Tone::Neutral))
            }
            ModelCliMode::Predict if self.model_cli.predict_sahi => Some(badge(&t, "SAHI", Tone::Accent)),
            ModelCliMode::Predict if self.model_cli.predict_tta => Some(badge(&t, "TTA", Tone::Accent)),
            _ => None,
        };

        panel(&t)
            .flex()
            .flex_col()
            .child(
                panel_header(&t, "Export & CLI")
                    .child(div().flex().items_center().gap(px(8.)).children(header_badge).child(mode_seg)),
            )
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(12.))
                    .child(controls)
                    .child(
                        div()
                            .px(px(12.))
                            .py(px(10.))
                            .rounded(px(radius::MD))
                            .bg(t.bg_inset)
                            .border_1()
                            .border_color(t.line_subtle)
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_family(crate::theme::MONO)
                                    .whitespace_normal()
                                    .child(built.command),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap(px(12.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(200.))
                                    .flex()
                                    .flex_col()
                                    .gap(px(2.))
                                    .when_some(built.hint, |d, h| {
                                        d.child(text(ty::SM, t.text_tertiary).whitespace_normal().child(h))
                                    })
                                    .when_some(built.quote_note, |d, note| {
                                        d.child(text(ty::SM, t.warning_text).whitespace_normal().child(note))
                                    }),
                            )
                            .child(
                                div()
                                    .flex_none()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(copy_btn)
                                    .children(server_btn),
                            ),
                    ),
            )
            .into_any_element()
    }

    fn export_cli_export_controls(
        &self,
        t: &crate::theme::Theme,
        m: &Model,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let fmt_labels: Vec<&str> = CliExportFormat::ALL.iter().map(|f| f.label()).collect();
        let fmt_idx =
            CliExportFormat::ALL.iter().position(|&f| f == self.model_cli.export_format).unwrap_or(0);
        let fmt_seg = self.segmented(
            "model-cli-fmt",
            &fmt_labels,
            fmt_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.model_cli.export_format = CliExportFormat::ALL[i.min(CliExportFormat::ALL.len() - 1)];
                cx.notify();
            }),
        );

        let extra: Option<AnyElement> = match self.model_cli.export_format {
            CliExportFormat::Onnx => {
                let dyn_batch = self
                    .chip(
                        "model-cli-dyn-batch",
                        "Dynamic batch",
                        self.model_cli.onnx_dynamic_batch,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_dynamic_batch = !this.model_cli.onnx_dynamic_batch;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Symbolic batch axis (--dynamic-batch; combines with Dynamic H×W into --dynamic)",
                        None,
                    ));
                let dyn_hw = self
                    .chip(
                        "model-cli-dyn-hw",
                        "Dynamic H×W",
                        self.model_cli.onnx_dynamic_hw,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_dynamic_hw = !this.model_cli.onnx_dynamic_hw;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Symbolic height and width axes (--dynamic-hw; multiples of 32)",
                        None,
                    ));
                let decode = self
                    .chip(
                        "model-cli-decode",
                        "Decoded outputs",
                        self.model_cli.onnx_decode,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_decode = !this.model_cli.onnx_decode;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Decode boxes and class scores inside the ONNX graph (--decode)",
                        None,
                    ));
                let half = self
                    .chip(
                        "model-cli-half",
                        "FP16 (--half)",
                        self.model_cli.onnx_half,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_half = !this.model_cli.onnx_half;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Half-precision ONNX weights, inputs, and activations (--half)",
                        None,
                    ));
                let head_o2m = (m.head == Head::O2o).then(|| {
                    self.chip(
                        "model-cli-head-o2m",
                        "Export o2m head",
                        self.model_cli.onnx_head_o2m,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_head_o2m = !this.model_cli.onnx_head_o2m;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Export this o2o model's one-to-many branch instead (--head o2m)",
                        None,
                    ))
                });
                Some(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(px(6.))
                        .child(dyn_batch)
                        .child(dyn_hw)
                        .child(decode)
                        .child(half)
                        .children(head_o2m)
                        .into_any_element(),
                )
            }
            CliExportFormat::Int8Quantize => {
                let labels: Vec<&str> = QuantMethod::ALL.iter().map(|q| q.label()).collect();
                let idx =
                    QuantMethod::ALL.iter().position(|&q| q == self.model_cli.quant_method).unwrap_or(0);
                let seg = self.segmented(
                    "model-cli-quant-method",
                    &labels,
                    idx,
                    window,
                    cx,
                    Rc::new(|this, i, _, cx| {
                        this.model_cli.quant_method = QuantMethod::ALL[i.min(QuantMethod::ALL.len() - 1)];
                        cx.notify();
                    }),
                );
                Some(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(text(ty::SM, t.text_tertiary).child("Calibration method"))
                        .child(seg)
                        .into_any_element(),
                )
            }
            _ => None,
        };

        div()
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(10.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_tertiary).font_weight(FontWeight::MEDIUM).child("Format"))
                    .child(fmt_seg),
            )
            .children(extra)
            .into_any_element()
    }

    fn export_cli_predict_controls(
        &self,
        t: &crate::theme::Theme,
        m: &Model,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let sahi = self
            .chip(
                "model-cli-sahi",
                "SAHI (--tile 640)",
                self.model_cli.predict_sahi,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.model_cli.predict_sahi = !this.model_cli.predict_sahi;
                    if this.model_cli.predict_sahi {
                        this.model_cli.predict_tta = false;
                    }
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "SAHI tiled inference for small objects in high-res images (mutually exclusive with TTA)",
                None,
            ));
        let tta = self
            .chip(
                "model-cli-tta",
                "TTA (--tta)",
                self.model_cli.predict_tta,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.model_cli.predict_tta = !this.model_cli.predict_tta;
                    if this.model_cli.predict_tta {
                        this.model_cli.predict_sahi = false;
                    }
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "Multi-scale + horizontal flip Test-Time Augmentation (mutually exclusive with SAHI)",
                None,
            ));
        let track = self
            .chip(
                "model-cli-track",
                "ByteTrack (--track)",
                self.model_cli.predict_track,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.model_cli.predict_track = !this.model_cli.predict_track;
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "Track objects across ordered images or folders with persistent IDs",
                None,
            ));

        let prec_labels: Vec<&str> = PredictPrecision::ALL.iter().map(|p| p.label()).collect();
        let prec_idx =
            PredictPrecision::ALL.iter().position(|&p| p == self.model_cli.predict_precision).unwrap_or(0);
        let prec_seg = self.segmented(
            "model-cli-prec",
            &prec_labels,
            prec_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.model_cli.predict_precision =
                    PredictPrecision::ALL[i.min(PredictPrecision::ALL.len() - 1)];
                cx.notify();
            }),
        );

        let conf = self.model_cli.predict_conf;
        let conf_slider = self.slider(
            "model-cli-conf",
            conf,
            MODEL_CLI_CONF.0,
            MODEL_CLI_CONF.1,
            MODEL_CLI_CONF.2,
            window,
            cx,
        );
        let iou_control = (m.head != Head::O2o).then(|| {
            let iou = self.model_cli.predict_iou;
            let iou_slider = self.slider(
                "model-cli-iou",
                iou,
                MODEL_CLI_IOU.0,
                MODEL_CLI_IOU.1,
                MODEL_CLI_IOU.2,
                window,
                cx,
            );
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(text(ty::SM, t.text_tertiary).child("IoU"))
                .child(div().w(px(100.)).child(iou_slider))
                .child(text(ty::SM, t.text_primary).tnum().w(px(30.)).child(fmt_thresh(iou)))
        });

        div()
            .flex()
            .flex_col()
            .gap(px(10.))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(px(10.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(6.))
                            .child(sahi)
                            .child(tta)
                            .child(track),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(text(ty::SM, t.text_tertiary).child("Precision"))
                            .child(prec_seg),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(20.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(text(ty::SM, t.text_tertiary).child("Confidence"))
                            .child(div().w(px(100.)).child(conf_slider))
                            .child(text(ty::SM, t.text_primary).tnum().w(px(30.)).child(fmt_thresh(conf))),
                    )
                    .children(iou_control),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, mock::MockBackend};

    fn sample_model() -> Model {
        let api = MockBackend::new(4200, true);
        let mut m = api.models().unwrap().remove(0);
        m.path = Some("D:/folo/runs/shapes-n/weights/best.folo".into());
        m.head = Head::O2m;
        m.fused = Some(false);
        m
    }

    #[test]
    fn onnx_export_combines_dynamic_axes_decode_half_and_server_spec() {
        let m = sample_model();
        let mut st = ModelCliState::default();

        // Plain ONNX.
        let b = build_model_cli(&m, None, &st);
        assert_eq!(b.command, "folo export --model D:/folo/runs/shapes-n/weights/best.folo --format onnx");
        assert_eq!(b.server_export, Some(ModelExportOptions::plain("onnx")));

        // Split dynamic batch only.
        st.onnx_dynamic_batch = true;
        let b = build_model_cli(&m, None, &st);
        assert!(b.command.contains("--dynamic-batch") && !b.command.contains("--dynamic "));
        assert_eq!(
            b.server_export,
            Some(ModelExportOptions {
                format: "onnx",
                decode: false,
                dynamic_batch: true,
                dynamic_hw: false,
                half: false,
            })
        );

        // Both dynamic batch + H×W collapse cleanly to `--dynamic`.
        st.onnx_dynamic_hw = true;
        st.onnx_decode = true;
        st.onnx_half = true;
        let b = build_model_cli(&m, None, &st);
        assert_eq!(
            b.command,
            "folo export --model D:/folo/runs/shapes-n/weights/best.folo --format onnx --dynamic --decode --half"
        );
        assert_eq!(
            b.server_export,
            Some(ModelExportOptions {
                format: "onnx",
                decode: true,
                dynamic_batch: true,
                dynamic_hw: true,
                half: true,
            })
        );

        // O2O model with `--head o2m` override drops server export and emits `--head o2m`.
        let mut o2o = m.clone();
        o2o.head = Head::O2o;
        st.onnx_head_o2m = true;
        let b = build_model_cli(&o2o, None, &st);
        assert!(b.command.ends_with("--head o2m"), "{}", b.command);
        assert!(b.server_export.is_none());
    }

    #[test]
    fn quantize_and_shell_unsafe_paths_quote_safely() {
        let mut m = sample_model();
        m.path = Some("D:/my models/best.folo".into());
        let mut st = ModelCliState {
            mode: ModelCliMode::Export,
            export_format: CliExportFormat::Int8Quantize,
            quant_method: QuantMethod::Mse,
            ..Default::default()
        };
        let b = build_model_cli(&m, Some("D:/my data/shapes"), &st);
        assert_eq!(
            b.command,
            "folo quantize --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\" --method mse"
        );
        assert!(b.quote_note.is_none());
        assert!(b.server_export.is_none());

        // Shell-unsafe `$` path must never be interpolated into the command string.
        m.path = Some("D:/bad$path/best.folo".into());
        st.quant_method = QuantMethod::Percentile;
        let b = build_model_cli(&m, Some("D:/ok/shapes"), &st);
        assert_eq!(b.command, "folo quantize --model <model> --data D:/ok/shapes");
        assert!(b.quote_note.as_deref().is_some_and(|n| n.contains("D:/bad$path/best.folo")));

        // Published weights without local path resolve by published id.
        let mut pub_m = sample_model();
        pub_m.path = None;
        pub_m.source = Some(ModelSource::Published);
        pub_m.id = "folo-n-coco-v5".into();
        let b = build_model_cli(&pub_m, None, &ModelCliState::default());
        assert_eq!(b.command, "folo export --model folo-n-coco-v5 --format onnx");
    }

    #[test]
    fn predict_flags_reflect_sahi_tta_track_precision_thresholds_and_o2o() {
        let m = sample_model();
        let mut st = ModelCliState {
            mode: ModelCliMode::Predict,
            predict_sahi: true,
            predict_track: true,
            predict_precision: PredictPrecision::F16,
            predict_conf: 0.40,
            predict_iou: 0.55,
            ..Default::default()
        };
        let b = build_model_cli(&m, None, &st);
        assert_eq!(
            b.command,
            "folo predict --model D:/folo/runs/shapes-n/weights/best.folo <images> --conf 0.40 --iou 0.55 --precision f16 --tile 640 --track"
        );

        // O2O head omits `--iou` even when non-default.
        let mut o2o = m.clone();
        o2o.head = Head::O2o;
        let b_o2o = build_model_cli(&o2o, None, &st);
        assert!(!b_o2o.command.contains("--iou"), "{}", b_o2o.command);

        // TTA replaces SAHI when SAHI is off.
        st.predict_sahi = false;
        st.predict_tta = true;
        st.predict_track = false;
        st.predict_precision = PredictPrecision::Int8;
        st.predict_conf = 0.25;
        st.predict_iou = 0.70;
        let b_tta = build_model_cli(&m, None, &st);
        assert_eq!(
            b_tta.command,
            "folo predict --model D:/folo/runs/shapes-n/weights/best.folo <images> --precision int8 --tta"
        );
    }
}
