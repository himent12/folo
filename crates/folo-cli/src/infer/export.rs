//! `folo export`: `.folo` (fused f16/f32, unfused weights) and ONNX (opset 17) export with independent
//! [`DynamicAxes`] (`--dynamic-batch` / `--dynamic-hw`) for both v1 (`folo-*`) and v2-Iso (`folo2-*`).

use std::path::Path;
use std::time::Instant;

use folo_nn::io::{FloatDType, ModelFile, Purpose};
use folo_vision::Detector;
use folo_vision::export::onnx::{self, DynamicAxes, OnnxExport, OnnxOptions, OnnxValue, proto::Dim};
use folo_vision::model::v2::FoloV2Model;
use serde_json::json;

use crate::cli::{ExportArgs, ExportFormatArg, bounds};
use crate::engine::engine_error;
use crate::output::{CliError, ExitCode, Out};
use crate::output_file::OutputFile;

use super::{nn_error, weights_error};

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
        let mut opts = OnnxOptions {
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
        let x = if meta.config.is_v2() {
            let names: std::collections::BTreeMap<String, &str> =
                meta.classes.iter().enumerate().map(|(i, c)| (i.to_string(), c.as_str())).collect();
            if let Ok(json_names) = serde_json::to_string(&names) {
                opts.metadata.insert("names".into(), json_names);
            }
            let (mut model, _) =
                FoloV2Model::from_file(&file, Purpose::Inference).map_err(|e| nn_error("export", e))?;
            if let Some(h) = args.head {
                model.set_head_mode(h.into()).map_err(|e| {
                    engine_error("export", folo_vision::Error::Config(format!("ONNX export: {e}")))
                })?;
            }
            model.fuse().map_err(|e| nn_error("export", e))?;
            onnx::export_v2_with_axes(&model, &opts, axes).map_err(|e| engine_error("export", e))?
        } else {
            onnx::export_file_with_axes(&file, &opts, axes).map_err(|e| engine_error("export", e))?
        };
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
        if meta.config.is_v2() {
            let (mut model, _) =
                FoloV2Model::from_file(&file, Purpose::Inference).map_err(|e| nn_error("export", e))?;
            model.fuse().map_err(|e| nn_error("export", e))?;
            let mut out_meta = meta.clone();
            out_meta.fused = true;
            output.write_with(|w| {
                folo_nn::io::write_export(w, &out_meta, model.state(), dtype)
                    .map_err(|e| nn_error("export", e))
            })?;
        } else {
            output.write_with(|w| file.export_to(w, dtype).map_err(|e| nn_error("export", e)))?;
        }
    } else {
        // The inference tensor set (EMA when present), unfused, written as a plain weights file
        // preserving the source file's non-checkpoint metadata (folo.metrics, folo.created, unknown folo.* keys).
        let (model, _) = Detector::from_file(&file, Purpose::Inference).map_err(|e| nn_error("export", e))?;
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
