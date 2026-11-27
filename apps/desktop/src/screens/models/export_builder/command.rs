//! Pure CLI command builder and server-export capability checks for the Model Detail
//! Export & Inference Command Builder (`docs/CLI.md`).

use crate::api::{Head, Model, ModelExportOptions, ModelSource};

/// Server feature flag advertising split dynamic ONNX axes (`dynamic_batch`, `dynamic_hw`).
pub const FEATURE_EXPORT_DYNAMIC_AXES: &str = "export_dynamic_axes";

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
    /// NMS / TTA merge IoU threshold (`--iou`, default 0.70).
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

/// Whether the current server can execute `opts` directly and deliver the artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerExportSupport {
    /// Supported on the connected server.
    Supported(ModelExportOptions),
    /// Server lacks required feature flags or cannot deliver the artifact over `/weights`;
    /// the user should copy and run the CLI command instead.
    Unsupported(&'static str),
}

/// Checks whether the connected server's advertised `features` and locality (`is_remote`)
/// safely support executing `opts` via `POST /api/models/{id}/export` without silently
/// ignoring ONNX switches or stranding a non-downloadable remote artifact.
pub fn check_server_export_support(
    opts: ModelExportOptions,
    features: &[String],
    is_remote: bool,
) -> ServerExportSupport {
    if opts.format != "onnx" {
        return ServerExportSupport::Supported(opts);
    }
    let has = |f: &str| features.iter().any(|x| x == f);
    let split_axes = opts.dynamic_batch != opts.dynamic_hw;
    let any_uniform_switch = opts.decode || opts.half || (opts.dynamic_batch && opts.dynamic_hw);
    if split_axes && !has(FEATURE_EXPORT_DYNAMIC_AXES) {
        return ServerExportSupport::Unsupported(
            "This server doesn't advertise single-axis dynamic ONNX export; copy the CLI command instead.",
        );
    }
    if (any_uniform_switch || (split_axes && (opts.decode || opts.half)))
        && !has(folo_api::FEATURE_EXPORT_ONNX_SWITCHES)
    {
        return ServerExportSupport::Unsupported(
            "This server doesn't advertise ONNX export switches (--dynamic/--decode/--half); copy the CLI command instead.",
        );
    }
    if is_remote && split_axes {
        return ServerExportSupport::Unsupported(
            "Remote servers can't download single-axis dynamic ONNX via /weights; enable both Dynamic batch and Dynamic H×W (--dynamic), or run the CLI command on the server.",
        );
    }
    ServerExportSupport::Supported(opts)
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

/// True when `--iou` affects `folo predict` output: always for `o2m`, and also for `o2o` when
/// `--tta` is enabled (TTA merges detections across augmented views using `--iou`).
pub fn predict_uses_iou(m: &Model, state: &ModelCliState) -> bool {
    m.head != Head::O2o || state.predict_tta
}

/// Derives the default `folo quantize` output path (`<stem>-int8.folo`) when targeting
/// `--precision int8` so users never accidentally pass an uncalibrated checkpoint.
fn derive_int8_model_path(path: &str) -> String {
    if path.ends_with("-int8.folo") {
        path.to_string()
    } else if let Some(stem) = path.strip_suffix(".folo") {
        format!("{stem}-int8.folo")
    } else {
        format!("{path}-int8.folo")
    }
}

/// Resolves the `--model` argument safely for pasting into PowerShell, cmd, or POSIX shells.
/// Returns `(arg_token, unquotable_raw_path)`.
fn resolve_model_arg(m: &Model, int8_target: bool) -> (String, Option<String>) {
    if let Some(path) = m.path.as_deref().filter(|p| !p.trim().is_empty()) {
        let candidate = if int8_target { derive_int8_model_path(path) } else { path.to_string() };
        return match folo_api::json::quote_path(&candidate) {
            Some(quoted) => (quoted, None),
            None => {
                let placeholder = if int8_target { "<model-int8>" } else { "<model>" };
                (placeholder.to_string(), Some(candidate))
            }
        };
    }
    if int8_target {
        return ("<model-int8.folo>".to_string(), None);
    }
    if m.source == Some(ModelSource::Published)
        && let Some(quoted) = folo_api::json::quote_path(&m.id)
    {
        return (quoted, None);
    }
    ("<model.folo>".to_string(), None)
}

/// Formats a threshold rounded to 2 decimal places (`0.25`, `0.70`).
pub(super) fn fmt_thresh(v: f32) -> String {
    format!("{:.2}", v.clamp(0.0, 1.0))
}

/// Builds the exact, shell-safe CLI command matching `docs/CLI.md` for `m` under `state`.
pub fn build_model_cli(m: &Model, run_dataset_path: Option<&str>, state: &ModelCliState) -> BuiltCliCommand {
    let int8_predict =
        state.mode == ModelCliMode::Predict && state.predict_precision == PredictPrecision::Int8;
    let (model_arg, unquoted_model) = resolve_model_arg(m, int8_predict);
    let mut unquoted_notes: Vec<String> = Vec::new();
    if let Some(raw) = unquoted_model {
        let placeholder = if int8_predict { "<model-int8>" } else { "<model>" };
        unquoted_notes.push(format!("Replace {placeholder} with this path, quoted for your shell: {raw}"));
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
                        "Calibrates CPU int8 inference (512 images by default, writing <name>-int8.folo); use with `folo predict --precision int8`."
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
            if predict_uses_iou(m, state) && (iou - 0.70).abs() > 0.004 {
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
                Some(
                    "Targets the calibrated `-int8.folo` file written by `folo quantize` (CPU only); run `folo quantize` first."
                        .into(),
                )
            } else if m.head == Head::O2o && state.predict_tta {
                Some(
                    "End-to-end (o2o) with TTA uses --iou to merge detections across augmented views.".into(),
                )
            } else if m.head == Head::O2o {
                Some("End-to-end (o2o) head predicts without NMS; --iou is unused unless TTA is on.".into())
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
    fn server_export_support_gates_older_servers_and_remote_split_axes() {
        let plain = ModelExportOptions::plain("onnx");
        let switched = ModelExportOptions {
            format: "onnx",
            decode: true,
            dynamic_batch: false,
            dynamic_hw: false,
            half: false,
        };
        let split = ModelExportOptions {
            format: "onnx",
            decode: false,
            dynamic_batch: true,
            dynamic_hw: false,
            half: false,
        };
        let full_dyn = ModelExportOptions {
            format: "onnx",
            decode: true,
            dynamic_batch: true,
            dynamic_hw: true,
            half: true,
        };

        let no_features: Vec<String> = Vec::new();
        let switches_only = vec![folo_api::FEATURE_EXPORT_ONNX_SWITCHES.to_string()];
        let all_features =
            vec![folo_api::FEATURE_EXPORT_ONNX_SWITCHES.to_string(), FEATURE_EXPORT_DYNAMIC_AXES.to_string()];

        assert!(matches!(
            check_server_export_support(plain, &no_features, false),
            ServerExportSupport::Supported(_)
        ));
        assert!(matches!(
            check_server_export_support(switched, &no_features, false),
            ServerExportSupport::Unsupported(_)
        ));
        assert!(matches!(
            check_server_export_support(switched, &switches_only, false),
            ServerExportSupport::Supported(_)
        ));
        assert!(matches!(
            check_server_export_support(split, &switches_only, false),
            ServerExportSupport::Unsupported(_)
        ));
        assert!(matches!(
            check_server_export_support(split, &all_features, false),
            ServerExportSupport::Supported(_)
        ));
        // Remote server rejects split axes (cannot download via /weights), accepts uniform --dynamic.
        assert!(matches!(
            check_server_export_support(split, &all_features, true),
            ServerExportSupport::Unsupported(_)
        ));
        assert!(matches!(
            check_server_export_support(full_dyn, &all_features, true),
            ServerExportSupport::Supported(_)
        ));
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
    fn predict_flags_reflect_sahi_tta_track_int8_target_and_o2o_tta_merge_iou() {
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

        // O2O head without TTA omits `--iou`.
        let mut o2o = m.clone();
        o2o.head = Head::O2o;
        let b_o2o = build_model_cli(&o2o, None, &st);
        assert!(!b_o2o.command.contains("--iou"), "{}", b_o2o.command);

        // O2O head WITH TTA keeps `--iou` (cross-view merge IoU).
        st.predict_sahi = false;
        st.predict_tta = true;
        st.predict_track = false;
        let b_o2o_tta = build_model_cli(&o2o, None, &st);
        assert!(b_o2o_tta.command.contains("--iou 0.55"), "{}", b_o2o_tta.command);

        // INT8 predict targets `<stem>-int8.folo`, never the uncalibrated checkpoint.
        st.predict_precision = PredictPrecision::Int8;
        st.predict_conf = 0.25;
        st.predict_iou = 0.70;
        let b_int8 = build_model_cli(&m, None, &st);
        assert_eq!(
            b_int8.command,
            "folo predict --model D:/folo/runs/shapes-n/weights/best-int8.folo <images> --precision int8 --tta"
        );
        // Without a known file path, INT8 predict uses `<model-int8.folo>` placeholder.
        let mut no_path = m.clone();
        no_path.path = None;
        no_path.source = Some(ModelSource::Published);
        let b_int8_ph = build_model_cli(&no_path, None, &st);
        assert!(b_int8_ph.command.contains("--model <model-int8.folo>"), "{}", b_int8_ph.command);
    }
}
