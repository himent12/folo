//! `folo export`: `.folo` (fused f16/f32, unfused weights) and ONNX (opset 17) export with independent
//! [`DynamicAxes`] (`--dynamic-batch` / `--dynamic-hw`) and FOLO v2-Iso (`folo2-*`) checkpoint support.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::Instant;

use folo_core::{DType, no_grad};
use folo_nn::Module;
use folo_nn::io::safetensors::{Container, FileDType, decode};
use folo_nn::io::{FORMAT_VERSION, FloatDType, ModelFile, Namespace};
use folo_vision::FoloModel;
use folo_vision::export::onnx::{self, DynamicAxes, OnnxExport, OnnxOptions, OnnxValue, proto::Dim};
use folo_vision::model::v2::{FoloV2Config, FoloV2Model};
use serde_json::json;

use crate::cli::{ExportArgs, ExportFormatArg, bounds};
use crate::engine::engine_error;
use crate::output::{CliError, ExitCode, Out};
use crate::output_file::OutputFile;

use super::{nn_error, weights_error};

/// Checkpoint-only metadata keys stripped when propagating metadata into an ONNX graph (matching `folo-nn`).
const CHECKPOINT_KEYS: [&str; 8] =
    ["folo.train", "folo.optim", "folo.sched", "folo.ema", "folo.rng", "folo.epoch", "folo.best", "folo.amp"];

/// Loaded FOLO v2-Iso checkpoint/weights ready for ONNX export.
struct LoadedV2 {
    model: FoloV2Model,
    arch: String,
    classes: Vec<String>,
    imgsz: u32,
    source: &'static str,
    metadata: BTreeMap<String, String>,
}

/// Checks whether `path` is a FOLO v2-Iso (`folo2-*`) `.folo` file by inspecting only its safetensors header.
fn is_v2_checkpoint(path: &Path) -> bool {
    let Ok(header) = Container::read_header(path) else {
        return false;
    };
    header.metadata.as_ref().and_then(|m| m.get("folo.arch")).is_some_and(|arch| arch.starts_with("folo2-"))
}

/// Opens and strictly loads a FOLO v2-Iso (`folo2-*`) `.folo` weights or checkpoint file, fuses it, and
/// returns it ready for [`onnx::export_v2_with_axes`].
fn load_v2_for_onnx(path: &Path) -> Result<LoadedV2, CliError> {
    let bad = |msg: String| weights_error("export", path, folo_nn::Error::Metadata(msg));
    let bytes = std::fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            CliError::new(
                ExitCode::NotFound,
                "path_not_found",
                format!("{} does not exist", folo_api::json::path(path)),
            )
        } else {
            CliError::new(ExitCode::Runtime, "io_error", format!("{}: {e}", folo_api::json::path(path)))
        }
    })?;
    let container = Container::parse(&bytes).map_err(|e| weights_error("export", path, e))?;
    let meta = container.metadata.as_ref().ok_or_else(|| bad("missing `__metadata__` header".into()))?;
    let format = meta.get("folo.format").ok_or_else(|| bad("missing required key `folo.format`".into()))?;
    if format.split('.').next() != Some(FORMAT_VERSION) {
        return Err(bad(format!("folo.format `{format}` is not supported (expected {FORMAT_VERSION})")));
    }
    let arch = meta.get("folo.arch").ok_or_else(|| bad("missing required key `folo.arch`".into()))?.clone();
    let cfg_str = meta.get("folo.config").ok_or_else(|| bad("missing required key `folo.config`".into()))?;
    let cfg: FoloV2Config = serde_json::from_str(cfg_str).map_err(|e| bad(format!("folo.config: {e}")))?;
    cfg.validate().map_err(|e| bad(format!("folo.config: {e}")))?;
    let expected_arch = format!("folo2-{}", cfg.scale);
    if arch != expected_arch {
        return Err(bad(format!("folo.arch `{arch}` does not match folo.config scale `{expected_arch}`")));
    }
    let classes: Vec<String> = match meta.get("folo.classes") {
        Some(s) => serde_json::from_str(s).map_err(|e| bad(format!("folo.classes: {e}")))?,
        None => (0..cfg.num_classes).map(|i| format!("class{i}")).collect(),
    };
    if classes.len() != cfg.num_classes {
        return Err(bad(format!(
            "folo.config.num_classes is {} but folo.classes has {} names",
            cfg.num_classes,
            classes.len()
        )));
    }
    let imgsz: u32 = match meta.get("folo.imgsz") {
        Some(s) => s.parse().map_err(|_| bad(format!("folo.imgsz `{s}` is not a positive integer")))?,
        None => 640,
    };
    if imgsz == 0 {
        return Err(bad("folo.imgsz must be positive".into()));
    }
    let fused = match meta.get("folo.fused").map(String::as_str) {
        None | Some("false") => false,
        Some("true") => true,
        Some(v) => return Err(bad(format!("folo.fused must be \"true\" or \"false\", got `{v}`"))),
    };
    if let Some(kind) = meta.get("folo.kind").map(String::as_str)
        && !matches!(kind, "weights" | "checkpoint")
    {
        return Err(bad(format!("folo.kind must be \"weights\" or \"checkpoint\", got `{kind}`")));
    }
    if let Some(dt) = meta.get("folo.dtype").map(String::as_str)
        && !matches!(dt, "F32" | "F16")
    {
        return Err(bad(format!("folo.dtype must be \"F32\" or \"F16\", got `{dt}`")));
    }

    let has_ema =
        container.tensors.iter().any(|t| t.name.starts_with("ema.") && !t.name.starts_with("ema.__"));
    let source = if has_ema { "ema" } else { "model" };
    let mut by_name: HashMap<String, usize> = HashMap::new();
    for (idx, t) in container.tensors.iter().enumerate() {
        let ns = Namespace::of(&t.name);
        if has_ema {
            if ns == Namespace::Ema && !t.name.starts_with("ema.__") {
                by_name.insert(t.name["ema.".len()..].to_string(), idx);
            }
        } else if ns == Namespace::Model {
            by_name.insert(t.name.clone(), idx);
        }
    }

    let mut model = FoloV2Model::new(cfg).map_err(|e| nn_error("export", e))?;
    if fused {
        model.fuse().map_err(|e| nn_error("export", e))?;
    }

    let targets = model.named_tensors();
    let mut model_names = BTreeSet::new();
    let mut problems = Vec::new();
    let mut plan = Vec::with_capacity(targets.len());
    for t in &targets {
        model_names.insert(t.name.as_str());
        let Some(&idx) = by_name.get(&t.name) else {
            problems.push(format!("missing `{}`", t.name));
            continue;
        };
        let info = &container.tensors[idx];
        if info.shape != t.tensor.dims() {
            problems.push(format!(
                "`{}` has shape {:?} in file, {:?} in model",
                t.name,
                info.shape,
                t.tensor.dims()
            ));
            continue;
        }
        let converts = info.dtype.to_core() == Some(t.tensor.dtype())
            || (info.dtype == FileDType::F16 && t.tensor.dtype() == DType::F32);
        if !converts {
            problems.push(format!(
                "`{}` has dtype {}, expected {:?}",
                t.name,
                info.dtype.as_str(),
                t.tensor.dtype()
            ));
            continue;
        }
        plan.push((&t.tensor, idx));
    }
    for name in by_name.keys() {
        if !model_names.contains(name.as_str()) {
            problems.push(format!("unexpected `{name}` (not in the model)"));
        }
    }
    if !problems.is_empty() {
        return Err(weights_error(
            "export",
            path,
            folo_nn::Error::Load(format!("strict load failed: {}", problems.join("; "))),
        ));
    }
    {
        let _g = no_grad();
        for (dst, idx) in plan {
            let info = &container.tensors[idx];
            let src = decode(info, container.data(&bytes, info))
                .map_err(|e| weights_error("export", path, e))?
                .to_dtype(dst.dtype())
                .map_err(|e| nn_error("export", e.into()))?;
            dst.copy_from(&src).map_err(|e| nn_error("export", e.into()))?;
        }
    }
    if !fused {
        model.fuse().map_err(|e| nn_error("export", e))?;
    }

    // Carry non-checkpoint metadata normalized as exported weights (matching v1 `Metadata::for_weights()`).
    let mut out_meta = BTreeMap::new();
    for (k, v) in meta {
        if !CHECKPOINT_KEYS.contains(&k.as_str()) {
            out_meta.insert(k.clone(), v.clone());
        }
    }
    out_meta.insert("folo.kind".into(), "weights".into());
    out_meta.insert("folo.fused".into(), "true".into());
    out_meta.insert("folo.version".into(), folo_cli::VERSION.into());
    let names_obj: serde_json::Map<String, serde_json::Value> =
        classes.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone().into())).collect();
    out_meta.insert("names".into(), serde_json::Value::Object(names_obj).to_string());
    out_meta.insert("folo.classes".into(), serde_json::to_string(&classes).expect("classes serialize"));

    Ok(LoadedV2 { model, arch, classes, imgsz, source, metadata: out_meta })
}

#[allow(clippy::too_many_arguments)]
fn emit_onnx_result(
    args: &ExportArgs,
    output: &Path,
    x: &OnnxExport,
    dtype_name: &str,
    source: &str,
    arch: &str,
    classes: &[String],
    imgsz: u32,
    opts: &OnnxOptions,
    axes: DynamicAxes,
    seconds: f64,
    out: Out,
) {
    let value = |v: &OnnxValue| {
        let shape: Vec<serde_json::Value> = v
            .shape
            .iter()
            .map(|d| match d {
                Dim::Value(n) => json!(n),
                Dim::Param(p) => json!(p),
            })
            .collect();
        json!({ "name": v.name, "dtype": v.dtype, "shape": shape })
    };
    out.record(&json!({
        "path": folo_api::json::path(output),
        "bytes": x.bytes.len(),
        "format": "onnx",
        "dtype": dtype_name,
        "fused": true,
        "source": source,
        "arch": arch,
        "classes": classes,
        "opset": onnx::OPSET,
        "imgsz": imgsz,
        "batch": if axes.batch { json!("dynamic") } else { json!(opts.batch) },
        "dynamic": axes.is_any(),
        "dynamic_batch": axes.batch,
        "dynamic_hw": axes.hw,
        "decode": args.decode,
        "head": x.metadata.get("folo.onnx.head"),
        "inputs": x.inputs.iter().map(value).collect::<Vec<_>>(),
        "outputs": x.outputs.iter().map(value).collect::<Vec<_>>(),
        "seconds": seconds,
    }));
    let shape = |v: &OnnxValue| v.shape.iter().map(ToString::to_string).collect::<Vec<_>>().join("x");
    out.human(format!(
        "Exported the {source} weights of {} to {} (ONNX opset {}, {}, {:.1} MB): {} {} -> {}",
        folo_api::json::path(&args.model),
        folo_api::json::path(output),
        onnx::OPSET,
        dtype_name,
        x.bytes.len() as f64 / 1e6,
        onnx::INPUT,
        shape(&x.inputs[0]),
        x.outputs.iter().map(|o| format!("{} {}", o.name, shape(o))).collect::<Vec<_>>().join(", "),
    ));
}

pub fn export(mut args: ExportArgs, out: Out) -> Result<(), CliError> {
    let onnx = args.format == ExportFormatArg::Onnx;
    let axes = DynamicAxes::new(args.dynamic || args.dynamic_batch, args.dynamic || args.dynamic_hw);
    if !onnx {
        let given = [
            ("--imgsz", args.imgsz.is_some()),
            ("--batch", args.batch.is_some()),
            ("--dynamic", args.dynamic),
            ("--dynamic-batch", args.dynamic_batch),
            ("--dynamic-hw", args.dynamic_hw),
            ("--half", args.half),
            ("--decode", args.decode),
            ("--head", args.head.is_some()),
        ];
        if let Some((flag, _)) = given.iter().find(|(_, on)| *on) {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                format!("{flag} only applies to --format onnx"),
            )
            .hint("`.folo` exports are size-independent; --format folo-f16 is the f16 one"));
        }
    }
    if axes.batch && args.batch.is_some() {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            "--batch and --dynamic / --dynamic-batch exclude each other",
        )
        .hint("symbolic batch makes the batch dimension dynamic; leave out --batch"));
    }
    if args.batch.is_some_and(|b| !bounds::POSITIVE_INT.contains_u64(b as u64)) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--batch must be at least 1"));
    }
    if !bounds::POSITIVE_INT.contains_u64(args.max_det as u64) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--max-det must be at least 1"));
    }
    if let Some(imgsz) = args.imgsz
        && onnx
        && !bounds::INFER_IMGSZ.contains_u64(u64::from(imgsz))
    {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--imgsz {imgsz} must be a multiple of 32 in 32..={}", bounds::INFER_IMGSZ_MAX),
        ));
    }
    args.model = crate::weights::resolve(&args.model)?;
    // A decoded graph is named apart from the raw one: `<name>-decoded.onnx` (`-decoded-f16` with --half).
    let (dtype, suffix, ext) = match args.format {
        ExportFormatArg::FoloF16 => (FloatDType::F16, "-f16", "folo"),
        ExportFormatArg::Folo => (FloatDType::F32, "-fused", "folo"),
        ExportFormatArg::Weights => (FloatDType::F32, "-weights", "folo"),
        ExportFormatArg::Onnx if args.half && args.decode => (FloatDType::F16, "-decoded-f16", "onnx"),
        ExportFormatArg::Onnx if args.half => (FloatDType::F16, "-f16", "onnx"),
        ExportFormatArg::Onnx if args.decode => (FloatDType::F32, "-decoded", "onnx"),
        ExportFormatArg::Onnx => (FloatDType::F32, "", "onnx"),
    };
    // The record's dtype is lowercase, as `val` and `predict` write it.
    let dtype_name = dtype.as_str().to_ascii_lowercase();

    if is_v2_checkpoint(&args.model) {
        if !onnx {
            return Err(CliError::new(
                ExitCode::NotImplemented,
                "not_supported",
                "FOLO v2-Iso (folo2-*) checkpoints currently support --format onnx export only",
            )
            .hint("pass --format onnx"));
        }
        let mut loaded = load_v2_for_onnx(&args.model)?;
        if let Some(h) = args.head {
            loaded.model.set_head_mode(h.into()).map_err(|e| {
                engine_error("export", folo_vision::Error::Config(format!("ONNX export: {e}")))
            })?;
        }
        let imgsz = args.imgsz.unwrap_or(loaded.imgsz);
        if !bounds::INFER_IMGSZ.contains_u64(u64::from(imgsz)) {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                format!("--imgsz {imgsz} must be a multiple of 32 in 32..={}", bounds::INFER_IMGSZ_MAX),
            ));
        }
        let mut output = OutputFile::new("export", &args.model, args.output.clone(), suffix, ext)?;
        let t = Instant::now();
        let opts = OnnxOptions {
            height: imgsz,
            width: imgsz,
            batch: args.batch.unwrap_or(1),
            dynamic: axes.batch && axes.hw,
            half: args.half,
            decode: args.decode,
            max_det: args.max_det,
            head: args.head.map(Into::into),
            metadata: loaded.metadata,
        };
        let x =
            onnx::export_v2_with_axes(&loaded.model, &opts, axes).map_err(|e| engine_error("export", e))?;
        output.write_bytes(&x.bytes)?;
        let output = output.publish()?;
        emit_onnx_result(
            &args,
            &output,
            &x,
            &dtype_name,
            loaded.source,
            &loaded.arch,
            &loaded.classes,
            imgsz,
            &opts,
            axes,
            t.elapsed().as_secs_f64(),
            out,
        );
        return Ok(());
    }

    let file = ModelFile::open(&args.model).map_err(|e| weights_error("export", &args.model, e))?;
    if args.format == ExportFormatArg::Weights && file.metadata().fused {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!(
                "{} is fused (Conv+BN folded): it cannot become unfused weights",
                folo_api::json::path(&args.model)
            ),
        )
        .hint(
            "export --format weights from the training checkpoint (best.folo) or an unfused weights file",
        ));
    }
    let imgsz = args.imgsz.unwrap_or(file.metadata().imgsz);
    if onnx && !bounds::INFER_IMGSZ.contains_u64(u64::from(imgsz)) {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--imgsz {imgsz} must be a multiple of 32 in 32..={}", bounds::INFER_IMGSZ_MAX),
        ));
    }
    let mut output = OutputFile::new("export", &args.model, args.output.clone(), suffix, ext)?;
    let t = Instant::now();
    let source = if file.has_ema() { "ema" } else { "model" };
    let meta = file.metadata();
    if onnx {
        let opts = OnnxOptions {
            height: imgsz,
            width: imgsz,
            batch: args.batch.unwrap_or(1),
            dynamic: axes.batch && axes.hw,
            half: args.half,
            decode: args.decode,
            max_det: args.max_det,
            head: args.head.map(Into::into),
            // The source's other keys (folo.metrics, folo.created, unknown folo.* keys) travel along.
            metadata: meta.for_weights().to_map(),
        };
        let x = onnx::export_file_with_axes(&file, &opts, axes).map_err(|e| engine_error("export", e))?;
        output.write_bytes(&x.bytes)?;
        let output = output.publish()?;
        emit_onnx_result(
            &args,
            &output,
            &x,
            &dtype_name,
            source,
            &meta.arch,
            &meta.classes,
            imgsz,
            &opts,
            axes,
            t.elapsed().as_secs_f64(),
            out,
        );
        return Ok(());
    }
    let fused = args.format != ExportFormatArg::Weights;
    if fused {
        output.write_with(|w| file.export_to(w, dtype).map_err(|e| nn_error("export", e)))?;
    } else {
        // The inference tensor set (EMA when present), unfused, written as a plain weights file
        // preserving the source file's non-checkpoint metadata (folo.metrics, folo.created, unknown folo.* keys).
        let (model, _) = FoloModel::from_file(&file, folo_nn::io::Purpose::Inference)
            .map_err(|e| nn_error("export", e))?;
        output.write_with(|w| {
            folo_nn::io::write_weights(w, file.metadata(), &model).map_err(|e| nn_error("export", e))
        })?;
    }
    let output = output.publish()?;
    let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    out.record(&json!({
        "path": folo_api::json::path(&output),
        "bytes": bytes,
        "format": match args.format {
            ExportFormatArg::FoloF16 => "folo-f16",
            ExportFormatArg::Weights => "weights",
            _ => "folo",
        },
        "dtype": dtype_name,
        "fused": fused,
        "source": source,
        "arch": meta.arch,
        "classes": meta.classes,
        "seconds": t.elapsed().as_secs_f64(),
    }));
    out.human(format!(
        "Exported the {source} weights of {} ({}, {}) to {} ({:.1} MB)",
        folo_api::json::path(&args.model),
        if fused { "fused" } else { "unfused" },
        dtype_name,
        folo_api::json::path(&output),
        bytes as f64 / 1e6
    ));
    Ok(())
}
