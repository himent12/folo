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
    if is_remote && !has(folo_api::FEATURE_WEIGHTS_ONNX) {
        return ServerExportSupport::Unsupported(
            "This remote server doesn't advertise ONNX weight downloads; copy the CLI command instead.",
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

/// Strips a trailing ASCII suffix case-insensitively on raw bytes (safe for arbitrary UTF-8).
fn strip_ascii_suffix_ignore_case<'a>(s: &'a str, suffix: &[u8]) -> Option<&'a str> {
    let bytes = s.as_bytes();
    if bytes.len() >= suffix.len() && bytes[bytes.len() - suffix.len()..].eq_ignore_ascii_case(suffix) {
        Some(&s[..s.len() - suffix.len()])
    } else {
        None
    }
}

/// Strips a recognized INT8 stem suffix (`-int8`, `-int8-<digits>`, or `-int8-<method>`, with an
/// optional `-recal` tag) using pure ASCII byte checks so multi-byte UTF-8 paths never panic.
fn strip_int8_stem_suffix(stem: &str) -> Option<&str> {
    let s = strip_ascii_suffix_ignore_case(stem, b"-recal").unwrap_or(stem);
    if let Some(base) = strip_ascii_suffix_ignore_case(s, b"-int8-percentile")
        .or_else(|| strip_ascii_suffix_ignore_case(s, b"-int8-minmax"))
        .or_else(|| strip_ascii_suffix_ignore_case(s, b"-int8-mse"))
        .or_else(|| strip_ascii_suffix_ignore_case(s, b"-int8"))
    {
        return Some(base);
    }
    // Recognize auto-numbered `folo quantize` outputs like `best-int8-2`.
    let bytes = s.as_bytes();
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1].is_ascii_digit() {
        end -= 1;
    }
    if end < bytes.len() {
        return strip_ascii_suffix_ignore_case(&s[..end], b"-int8-");
    }
    None
}

/// True when `path` already names an INT8-calibrated `.folo` checkpoint (`-int8.folo`,
/// `-int8-<n>.folo`, or `-int8-<method>[-recal].folo`), checked on raw ASCII bytes.
pub fn is_int8_model_path(path: &str) -> bool {
    strip_ascii_suffix_ignore_case(path, b".folo").and_then(strip_int8_stem_suffix).is_some()
}

/// True when `m` already points to a calibrated INT8 `.folo` file on disk.
pub fn model_is_int8_file(m: &Model) -> bool {
    m.path.as_deref().filter(|p| !p.trim().is_empty()).is_some_and(is_int8_model_path)
}

/// Derives an explicit per-method INT8 output path (`<stem>-int8-<method>.folo`) shared by
/// `folo quantize --output` and `folo predict --precision int8`, preventing `folo quantize`'s
/// auto-numbered `-2.folo` fallback from silently diverging from `folo predict`. Guarantees the
/// derived output differs from `path` (case-insensitively) so `folo quantize` never fails with
/// `--output must differ from --model`.
fn derive_quant_output_path(path: &str, method: QuantMethod) -> String {
    let stem = strip_ascii_suffix_ignore_case(path, b".folo").unwrap_or(path);
    let base = strip_int8_stem_suffix(stem).filter(|b| !b.is_empty()).unwrap_or(if stem.is_empty() {
        "model"
    } else {
        stem
    });
    let derived = format!("{base}-int8-{}.folo", method.label());
    if derived.eq_ignore_ascii_case(path) {
        format!("{base}-int8-{}-recal.folo", method.label())
    } else {
        derived
    }
}

/// Derives the INT8 checkpoint path for `folo predict --precision int8`: preserves an already
/// quantized `-int8*.folo` file as-is, otherwise matches `derive_quant_output_path`.
fn derive_int8_model_path(path: &str, method: QuantMethod) -> String {
    if is_int8_model_path(path) { path.to_string() } else { derive_quant_output_path(path, method) }
}

/// Resolves the `--model` argument safely for pasting into PowerShell, cmd, or POSIX shells.
/// Returns `(arg_token, unquotable_raw_path)`.
fn resolve_model_arg(m: &Model, int8_target: bool, method: QuantMethod) -> (String, Option<String>) {
    if let Some(path) = m.path.as_deref().filter(|p| !p.trim().is_empty()) {
        let candidate = if int8_target { derive_int8_model_path(path, method) } else { path.to_string() };
        return match folo_api::json::quote_path(&candidate) {
            Some(quoted) => (quoted, None),
            None => {
                let placeholder = if int8_target { "<model-int8>" } else { "<model>" };
                (placeholder.to_string(), Some(candidate))
            }
        };
    }
    if int8_target {
        if m.source == Some(ModelSource::Published) {
            let candidate = format!("{}-int8-{}.folo", m.id, method.label());
            if let Some(quoted) = folo_api::json::quote_path(&candidate) {
                return (quoted, None);
            }
        }
        return (format!("<model-int8-{}.folo>", method.label()), None);
    }
    if m.source == Some(ModelSource::Published)
        && let Some(quoted) = folo_api::json::quote_path(&m.id)
    {
        return (quoted, None);
    }
    ("<model.folo>".to_string(), None)
}

/// Resolves the explicit `--output` argument for `folo quantize` so re-running with a different
/// calibration method writes a distinct file instead of auto-numbering `-2.folo`.
fn resolve_quant_output_arg(m: &Model, method: QuantMethod) -> (Option<String>, Option<String>) {
    if let Some(path) = m.path.as_deref().filter(|p| !p.trim().is_empty()) {
        let candidate = derive_quant_output_path(path, method);
        return match folo_api::json::quote_path(&candidate) {
            Some(quoted) => (Some(quoted), None),
            None => (Some("<output-int8>".to_string()), Some(candidate)),
        };
    }
    if m.source == Some(ModelSource::Published) {
        let candidate = format!("{}-int8-{}.folo", m.id, method.label());
        if let Some(quoted) = folo_api::json::quote_path(&candidate) {
            return (Some(quoted), None);
        }
    }
    (Some(format!("<model-int8-{}.folo>", method.label())), None)
}

/// Formats a threshold rounded to 2 decimal places (`0.25`, `0.70`).
pub(super) fn fmt_thresh(v: f32) -> String {
    format!("{:.2}", v.clamp(0.0, 1.0))
}

/// Builds the exact, shell-safe CLI command matching `docs/CLI.md` for `m` under `state`.
pub fn build_model_cli(m: &Model, run_dataset_path: Option<&str>, state: &ModelCliState) -> BuiltCliCommand {
    let int8_predict =
        state.mode == ModelCliMode::Predict && state.predict_precision == PredictPrecision::Int8;
    let (model_arg, unquoted_model) = resolve_model_arg(m, int8_predict, state.quant_method);
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
                    let (out_arg, unquoted_out) = resolve_quant_output_arg(m, state.quant_method);
                    if let Some(raw) = unquoted_out {
                        unquoted_notes.push(format!(
                            "Replace <output-int8> with this path, quoted for your shell: {raw}"
                        ));
                    }
                    if let Some(out) = out_arg {
                        parts.push("--output".into());
                        parts.push(out);
                    }
                    hint = Some(
                        "Calibrates CPU int8 inference (512 images by default) and writes an explicit per-method file (never overwrites); use with `folo predict --precision int8`."
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
                Some(if model_is_int8_file(m) {
                    "Runs CPU int8 inference with this calibrated checkpoint.".into()
                } else {
                    "Matches the `--output` file written by `folo quantize` for the selected calibration method (CPU only).".into()
                })
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
        let switches_and_axes =
            vec![folo_api::FEATURE_EXPORT_ONNX_SWITCHES.to_string(), FEATURE_EXPORT_DYNAMIC_AXES.to_string()];
        let all_features = vec![
            folo_api::FEATURE_EXPORT_ONNX_SWITCHES.to_string(),
            FEATURE_EXPORT_DYNAMIC_AXES.to_string(),
            folo_api::FEATURE_WEIGHTS_ONNX.to_string(),
        ];

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
            check_server_export_support(split, &switches_and_axes, false),
            ServerExportSupport::Supported(_)
        ));
        // Remote server requires `weights_onnx` for ONNX downloads and rejects split axes.
        assert!(matches!(
            check_server_export_support(switched, &switches_only, true),
            ServerExportSupport::Unsupported(_)
        ));
        assert!(matches!(
            check_server_export_support(split, &all_features, true),
            ServerExportSupport::Unsupported(_)
        ));
        assert!(matches!(
            check_server_export_support(full_dyn, &all_features, true),
            ServerExportSupport::Supported(_)
        ));
        assert_eq!(
            derive_int8_model_path("D:/models/best.FOLO", QuantMethod::Percentile),
            "D:/models/best-int8-percentile.folo"
        );
        assert_eq!(
            derive_int8_model_path("D:/models/best-INT8.FOLO", QuantMethod::MinMax),
            "D:/models/best-INT8.FOLO"
        );
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
            "folo quantize --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\" --method mse --output \"D:/my models/best-int8-mse.folo\""
        );
        assert!(b.quote_note.is_none());
        assert!(b.server_export.is_none());

        // Switching calibration method changes both quantize `--output` and predict `--model` in lockstep.
        st.quant_method = QuantMethod::MinMax;
        let b_q = build_model_cli(&m, Some("D:/my data/shapes"), &st);
        assert!(
            b_q.command.ends_with("--method minmax --output \"D:/my models/best-int8-minmax.folo\""),
            "{}",
            b_q.command
        );
        st.mode = ModelCliMode::Predict;
        st.predict_precision = PredictPrecision::Int8;
        let b_p = build_model_cli(&m, Some("D:/my data/shapes"), &st);
        assert_eq!(
            b_p.command,
            "folo predict --model \"D:/my models/best-int8-minmax.folo\" <images> --precision int8"
        );

        // Shell-unsafe `$` path must never be interpolated into the command string.
        m.path = Some("D:/bad$path/best.folo".into());
        st.mode = ModelCliMode::Export;
        st.quant_method = QuantMethod::Percentile;
        let b = build_model_cli(&m, Some("D:/ok/shapes"), &st);
        assert_eq!(b.command, "folo quantize --model <model> --data D:/ok/shapes --output <output-int8>");
        assert!(b.quote_note.as_deref().is_some_and(|n| n.contains("D:/bad$path/best.folo")));
        assert!(b.quote_note.as_deref().is_some_and(|n| n.contains("D:/bad$path/best-int8-percentile.folo")));

        // Published weights without local path resolve by published id and share explicit INT8 file name.
        let mut pub_m = sample_model();
        pub_m.path = None;
        pub_m.source = Some(ModelSource::Published);
        pub_m.id = "folo-n-coco-v5".into();
        let b = build_model_cli(&pub_m, None, &ModelCliState::default());
        assert_eq!(b.command, "folo export --model folo-n-coco-v5 --format onnx");
        let b_pub_q = build_model_cli(&pub_m, Some("D:/ok/shapes"), &st);
        assert_eq!(
            b_pub_q.command,
            "folo quantize --model folo-n-coco-v5 --data D:/ok/shapes --output folo-n-coco-v5-int8-percentile.folo"
        );
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

        // INT8 predict targets `<stem>-int8-<method>.folo`, matching `folo quantize --output`.
        st.predict_precision = PredictPrecision::Int8;
        st.quant_method = QuantMethod::Percentile;
        st.predict_conf = 0.25;
        st.predict_iou = 0.70;
        let b_int8 = build_model_cli(&m, None, &st);
        assert_eq!(
            b_int8.command,
            "folo predict --model D:/folo/runs/shapes-n/weights/best-int8-percentile.folo <images> --precision int8 --tta"
        );
        // Published model without a local path matches its explicit `folo quantize --output` filename.
        let mut pub_no_path = m.clone();
        pub_no_path.path = None;
        pub_no_path.source = Some(ModelSource::Published);
        let b_int8_pub = build_model_cli(&pub_no_path, None, &st);
        assert!(b_int8_pub.command.contains("--model m_44c0-int8-percentile.folo"), "{}", b_int8_pub.command);
        // Unknown non-published model without a file path keeps Quantize `--output` and Predict
        // `--model` in lockstep with the selected calibration method.
        let mut no_path = m.clone();
        no_path.path = None;
        no_path.source = None;
        st.quant_method = QuantMethod::MinMax;
        let b_int8_ph = build_model_cli(&no_path, None, &st);
        assert!(b_int8_ph.command.contains("--model <model-int8-minmax.folo>"), "{}", b_int8_ph.command);
        let q_no_path = ModelCliState {
            mode: ModelCliMode::Export,
            export_format: CliExportFormat::Int8Quantize,
            quant_method: QuantMethod::MinMax,
            ..Default::default()
        };
        let b_q_ph = build_model_cli(&no_path, None, &q_no_path);
        assert!(b_q_ph.command.ends_with("--output <model-int8-minmax.folo>"), "{}", b_q_ph.command);

        // Auto-numbered `folo quantize` outputs (`-int8-2.folo`) are preserved as-is for Predict
        // and stripped cleanly when re-quantizing; same-method re-quantize never emits `--output == --model`.
        assert!(is_int8_model_path("D:/models/best-int8-2.folo"));
        assert!(is_int8_model_path("D:/models/best-int8-percentile-recal.folo"));
        assert_eq!(
            derive_int8_model_path("D:/models/best-int8-2.folo", QuantMethod::Percentile),
            "D:/models/best-int8-2.folo"
        );
        assert_eq!(
            derive_quant_output_path("D:/models/best-int8-2.folo", QuantMethod::MinMax),
            "D:/models/best-int8-minmax.folo"
        );
        assert_eq!(
            derive_quant_output_path("D:/models/best-int8-percentile.folo", QuantMethod::Percentile),
            "D:/models/best-int8-percentile-recal.folo"
        );
        assert_eq!(
            derive_quant_output_path("D:/models/best-INT8-PERCENTILE.FOLO", QuantMethod::Percentile),
            "D:/models/best-int8-percentile-recal.folo"
        );

        // Multi-byte UTF-8 paths whose byte length straddles suffix offsets must never panic.
        assert_eq!(
            derive_int8_model_path("cafébest.folo", QuantMethod::Percentile),
            "cafébest-int8-percentile.folo"
        );
        assert_eq!(derive_int8_model_path("模型.folo", QuantMethod::MinMax), "模型-int8-minmax.folo");
        assert_eq!(derive_int8_model_path("日本語", QuantMethod::Mse), "日本語-int8-mse.folo");
        assert_eq!(derive_int8_model_path("D:/模型-INT8.FOLO", QuantMethod::MinMax), "D:/模型-INT8.FOLO");
        assert_eq!(
            derive_quant_output_path("D:/模型-INT8.FOLO", QuantMethod::MinMax),
            "D:/模型-int8-minmax.folo"
        );
        assert_eq!(
            derive_int8_model_path("D:/模型-int8-2.folo", QuantMethod::Percentile),
            "D:/模型-int8-2.folo"
        );
        assert_eq!(derive_int8_model_path("best.FOLO", QuantMethod::Percentile), "best-int8-percentile.folo");
    }
}
