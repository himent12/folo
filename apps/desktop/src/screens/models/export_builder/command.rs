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

/// Selective-`f32` layer-skip policy for `folo quantize` (`docs/CLI.md`, `#779`/`#788`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum QuantF32Policy {
    /// Sensitivity-ranked promotion within `--max-f32-macs` and `--sqnr-threshold` (CLI default).
    #[default]
    Auto,
    /// Keep the P5 neck tail (`neck.out5.*`) in `f32`.
    P5Tail,
    /// Quantize every eligible conv to int8 with no selective `f32` layers.
    None,
}

impl QuantF32Policy {
    pub const ALL: [Self; 3] = [Self::Auto, Self::P5Tail, Self::None];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::P5Tail => "P5 tail",
            Self::None => "None",
        }
    }

    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::P5Tail => "p5-tail",
            Self::None => "none",
        }
    }
}

/// MAC-fraction budget preset for `--f32-policy auto` (`--max-f32-macs`, CLI default `0.05`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum QuantF32MacBudget {
    #[default]
    Pct5,
    Pct10,
    Pct15,
}

impl QuantF32MacBudget {
    pub const ALL: [Self; 3] = [Self::Pct5, Self::Pct10, Self::Pct15];

    pub fn label(self) -> &'static str {
        match self {
            Self::Pct5 => "5%",
            Self::Pct10 => "10%",
            Self::Pct15 => "15%",
        }
    }

    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Pct5 => "0.05",
            Self::Pct10 => "0.10",
            Self::Pct15 => "0.15",
        }
    }

    fn file_tag(self) -> Option<&'static str> {
        match self {
            Self::Pct5 => None,
            Self::Pct10 => Some("f32-10"),
            Self::Pct15 => Some("f32-15"),
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

/// Which checkpoint `folo predict --precision int8` targets when the opened model is already an
/// INT8 checkpoint (`-int8*.folo`): either the existing file as-is, or the recalibrated output
/// derived from the selected `folo quantize` settings.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PredictInt8Target {
    #[default]
    Current,
    QuantizedOutput,
}

impl PredictInt8Target {
    pub const ALL: [Self; 2] = [Self::Current, Self::QuantizedOutput];

    pub fn label(self) -> &'static str {
        match self {
            Self::Current => "Current checkpoint",
            Self::QuantizedOutput => "Quantized output",
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
    /// Selective-`f32` layer-skip policy (`folo quantize --f32-policy`).
    pub quant_f32_policy: QuantF32Policy,
    /// Selective-`f32` MAC budget preset when `quant_f32_policy == Auto` (`--max-f32-macs`).
    pub quant_f32_macs: QuantF32MacBudget,
    /// SAHI tiled inference (`--tile 640`, mutually exclusive with `predict_tta`).
    pub predict_sahi: bool,
    /// Test-Time Augmentation (`--tta`, mutually exclusive with `predict_sahi`).
    pub predict_tta: bool,
    /// ByteTrack multi-object tracking (`--track`).
    pub predict_track: bool,
    /// Inference precision (`f32`, `f16`, `int8`).
    pub predict_precision: PredictPrecision,
    /// Explicit override for which checkpoint Predict `int8` targets (`None` picks `Current` when
    /// the path matches `-int8*.folo`, else `QuantizedOutput`).
    pub predict_int8_target: Option<PredictInt8Target>,
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
            quant_f32_policy: QuantF32Policy::Auto,
            quant_f32_macs: QuantF32MacBudget::Pct5,
            predict_sahi: false,
            predict_tta: false,
            predict_track: false,
            predict_precision: PredictPrecision::F32,
            predict_int8_target: None,
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

/// Optional filename tag for non-default selective-`f32` settings so distinct quantized variants
/// don't collide on `--output` and stay in lockstep with `folo predict --precision int8`.
fn quant_policy_file_tag(policy: QuantF32Policy, macs: QuantF32MacBudget) -> Option<&'static str> {
    match policy {
        QuantF32Policy::Auto => macs.file_tag(),
        QuantF32Policy::P5Tail => Some("p5-tail"),
        QuantF32Policy::None => Some("full"),
    }
}

/// Strips a recognized INT8 stem suffix (`-int8`, `-int8-<digits>`, or `-int8-<method>` with
/// optional selective-f32 policy tag and `-recal` tag) using pure ASCII byte checks so multi-byte
/// UTF-8 paths never panic.
fn strip_int8_stem_suffix(stem: &str) -> Option<&str> {
    let s = strip_ascii_suffix_ignore_case(stem, b"-recal").unwrap_or(stem);
    let s = strip_ascii_suffix_ignore_case(s, b"-f32-10")
        .or_else(|| strip_ascii_suffix_ignore_case(s, b"-f32-15"))
        .or_else(|| strip_ascii_suffix_ignore_case(s, b"-p5-tail"))
        .or_else(|| strip_ascii_suffix_ignore_case(s, b"-full"))
        .unwrap_or(s);
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
/// `-int8-<n>.folo`, or `-int8-<method>[-<policy>][-recal].folo`), checked on raw ASCII bytes.
pub fn is_int8_model_path(path: &str) -> bool {
    strip_ascii_suffix_ignore_case(path, b".folo").and_then(strip_int8_stem_suffix).is_some()
}

/// True when `m` already points to a calibrated INT8 `.folo` file on disk.
pub fn model_is_int8_file(m: &Model) -> bool {
    m.path.as_deref().filter(|p| !p.trim().is_empty()).is_some_and(is_int8_model_path)
}

fn format_int8_suffix(method: QuantMethod, policy: QuantF32Policy, macs: QuantF32MacBudget) -> String {
    match quant_policy_file_tag(policy, macs) {
        Some(tag) => format!("int8-{}-{tag}", method.label()),
        None => format!("int8-{}", method.label()),
    }
}

/// Derives an explicit per-method/policy INT8 output path (`<stem>-int8-<method>[-<policy>].folo`)
/// shared by `folo quantize --output` and `folo predict --precision int8`, preventing `folo quantize`'s
/// auto-numbered `-2.folo` fallback from silently diverging from `folo predict`. Guarantees the
/// derived output differs from `path` (case-insensitively) so `folo quantize` never fails with
/// `--output must differ from --model`.
pub(super) fn derive_quant_output_path(
    path: &str,
    method: QuantMethod,
    policy: QuantF32Policy,
    macs: QuantF32MacBudget,
) -> String {
    let stem = strip_ascii_suffix_ignore_case(path, b".folo").unwrap_or(path);
    let base = strip_int8_stem_suffix(stem).filter(|b| !b.is_empty()).unwrap_or(if stem.is_empty() {
        "model"
    } else {
        stem
    });
    let suffix = format_int8_suffix(method, policy, macs);
    let derived = format!("{base}-{suffix}.folo");
    if derived.eq_ignore_ascii_case(path) { format!("{base}-{suffix}-recal.folo") } else { derived }
}

/// Resolves which INT8 checkpoint `folo predict --precision int8` targets: respects an explicit
/// user selection, otherwise defaults to `Current` for `-int8*.folo` filenames and `QuantizedOutput`
/// for all other checkpoints.
pub fn effective_predict_int8_target(m: &Model, state: &ModelCliState) -> PredictInt8Target {
    state.predict_int8_target.unwrap_or_else(|| {
        if model_is_int8_file(m) { PredictInt8Target::Current } else { PredictInt8Target::QuantizedOutput }
    })
}

/// Resolves the `--model` argument safely for pasting into PowerShell, cmd, or POSIX shells.
/// Returns `(arg_token, unquotable_raw_path)`.
fn resolve_model_arg(
    m: &Model,
    int8_target: bool,
    use_quantized_output: bool,
    method: QuantMethod,
    policy: QuantF32Policy,
    macs: QuantF32MacBudget,
) -> (String, Option<String>) {
    let suffix = format_int8_suffix(method, policy, macs);
    if let Some(path) = m.path.as_deref().filter(|p| !p.trim().is_empty()) {
        let candidate = if int8_target && use_quantized_output {
            derive_quant_output_path(path, method, policy, macs)
        } else {
            path.to_string()
        };
        return match folo_api::json::quote_path(&candidate) {
            Some(quoted) => (quoted, None),
            None => {
                let placeholder =
                    if int8_target && use_quantized_output { "<model-int8>" } else { "<model>" };
                (placeholder.to_string(), Some(candidate))
            }
        };
    }
    if int8_target && use_quantized_output {
        if m.source == Some(ModelSource::Published) {
            let candidate = format!("{}-{suffix}.folo", m.id);
            if let Some(quoted) = folo_api::json::quote_path(&candidate) {
                return (quoted, None);
            }
        }
        return (format!("<model-{suffix}.folo>"), None);
    }
    if m.source == Some(ModelSource::Published)
        && let Some(quoted) = folo_api::json::quote_path(&m.id)
    {
        return (quoted, None);
    }
    ("<model.folo>".to_string(), None)
}

/// Resolves the explicit `--output` argument for `folo quantize` so re-running with a different
/// calibration method or selective-`f32` policy writes a distinct file instead of `-2.folo`.
fn resolve_quant_output_arg(
    m: &Model,
    method: QuantMethod,
    policy: QuantF32Policy,
    macs: QuantF32MacBudget,
) -> (Option<String>, Option<String>) {
    let suffix = format_int8_suffix(method, policy, macs);
    if let Some(path) = m.path.as_deref().filter(|p| !p.trim().is_empty()) {
        let candidate = derive_quant_output_path(path, method, policy, macs);
        return match folo_api::json::quote_path(&candidate) {
            Some(quoted) => (Some(quoted), None),
            None => (Some("<output-int8>".to_string()), Some(candidate)),
        };
    }
    if m.source == Some(ModelSource::Published) {
        let candidate = format!("{}-{suffix}.folo", m.id);
        if let Some(quoted) = folo_api::json::quote_path(&candidate) {
            return (Some(quoted), None);
        }
    }
    (Some(format!("<model-{suffix}.folo>")), None)
}

/// Formats a threshold rounded to 2 decimal places (`0.25`, `0.70`).
pub(super) fn fmt_thresh(v: f32) -> String {
    format!("{:.2}", v.clamp(0.0, 1.0))
}

/// Builds the exact, shell-safe CLI command matching `docs/CLI.md` for `m` under `state`.
pub fn build_model_cli(m: &Model, run_dataset_path: Option<&str>, state: &ModelCliState) -> BuiltCliCommand {
    let int8_predict =
        state.mode == ModelCliMode::Predict && state.predict_precision == PredictPrecision::Int8;
    let effective_target = effective_predict_int8_target(m, state);
    let use_quantized_output = effective_target == PredictInt8Target::QuantizedOutput;
    let (model_arg, unquoted_model) = resolve_model_arg(
        m,
        int8_predict,
        use_quantized_output,
        state.quant_method,
        state.quant_f32_policy,
        state.quant_f32_macs,
    );
    let mut unquoted_notes: Vec<String> = Vec::new();
    if let Some(raw) = unquoted_model {
        let placeholder = if int8_predict && use_quantized_output { "<model-int8>" } else { "<model>" };
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
                    match state.quant_f32_policy {
                        QuantF32Policy::Auto => {
                            if state.quant_f32_macs != QuantF32MacBudget::Pct5 {
                                parts.push("--f32-policy auto".into());
                                parts.push(format!("--max-f32-macs {}", state.quant_f32_macs.cli_value()));
                            }
                        }
                        QuantF32Policy::P5Tail | QuantF32Policy::None => {
                            parts.push(format!("--f32-policy {}", state.quant_f32_policy.cli_value()));
                        }
                    }
                    let (out_arg, unquoted_out) = resolve_quant_output_arg(
                        m,
                        state.quant_method,
                        state.quant_f32_policy,
                        state.quant_f32_macs,
                    );
                    if let Some(raw) = unquoted_out {
                        unquoted_notes.push(format!(
                            "Replace <output-int8> with this path, quoted for your shell: {raw}"
                        ));
                    }
                    if let Some(out) = out_arg {
                        parts.push("--output".into());
                        parts.push(out);
                    }
                    hint = Some(match state.quant_f32_policy {
                        QuantF32Policy::Auto => format!(
                            "Selective-f32 auto policy (≤{} MACs in f32, SQNR < 35 dB, keeps neck.out5.cv2 & bottleneck pairs); calibrates 512 images on CPU.",
                            state.quant_f32_macs.label()
                        ),
                        QuantF32Policy::P5Tail => {
                            "Keeps the P5 neck tail (neck.out5.*) in f32 (`--f32-policy p5-tail`); calibrates 512 images on CPU."
                                .into()
                        }
                        QuantF32Policy::None => {
                            "Quantizes every eligible conv to int8 (`--f32-policy none`); fastest CPU int8 path with no selective f32 fallback."
                                .into()
                        }
                    });
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
                Some(if effective_target == PredictInt8Target::Current {
                    "Runs CPU int8 inference with this checkpoint as-is (requires an int8-calibrated .folo file).".into()
                } else {
                    "Matches the `--output` file written by `folo quantize` for the selected calibration method and selective-f32 policy (CPU only).".into()
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
