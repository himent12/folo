//! `folo val`, `folo predict` and `folo export`.
//!
//! - `mod.rs`: shared model loading / arg checks, `val`, and `export`.
//! - `predict.rs`: `predict`, recursive image walking, `--save` path checks and atomic image saving.

mod predict;

pub use predict::predict;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use folo_data::{Dataset, Split};
use folo_nn::io::{FloatDType, ModelFile};
use folo_vision::export::onnx::{self, OnnxOptions, OnnxValue, proto::Dim};
use folo_vision::predict::f16_unavailable;
use folo_vision::train::{can_place, resolve_device};
use folo_vision::val::{ValConfig, check_config, val_metrics, validate};
use folo_vision::{FoloModel, NmsConfig, Precision};
use serde_json::json;

use crate::cli::{ExportArgs, ExportFormatArg, PrecisionArg, SplitArg, ValArgs, bounds};
use crate::engine::{engine_error, open_dataset_splits};
use crate::output::{CliError, ExitCode, Out};
use crate::output_file::OutputFile;

fn nn_error(command: &str, e: folo_nn::Error) -> CliError {
    engine_error(command, e.into())
}

/// A weights file that cannot be read: `invalid_weights` saying what the file is not, instead of a parser
/// detail such as a header length.
pub(crate) fn weights_error(command: &str, path: &Path, e: folo_nn::Error) -> CliError {
    let mut err = nn_error(command, e);
    if err.code == "invalid_weights" {
        err.message = folo_api::codes::cap(
            &format!("{} is not a .folo weights file ({})", folo_api::json::path(path), err.message),
            folo_api::codes::MAX_MESSAGE_CHARS,
        );
        err = err.hint(
            "pass a .folo file written by `folo train` (runs/<name>/weights/best.folo) or `folo export`",
        );
    }
    err
}

/// Open weights for inference (EMA when present, fused) on `device`, in f16 with `half` where the device computes
/// in it (an f16 export is uploaded as it is); otherwise in f32, with a note on stderr when `half` asked for f16.
/// `int8` runs the backbone and neck in int8 on the CPU with the file's calibration ([`open_int8`]).
/// `tensor_cores` asks for the tensor-core f16 kernels (a note on stderr when they are unavailable or unused).
/// The kernels an earlier `command` with this model and `shape` (the options that change them) used start compiling
/// on every core while the weights load ([`folo_vision::precompile`]).
pub(super) fn open_model(
    command: &str,
    path: &Path,
    device: &str,
    precision: (bool, Option<PrecisionArg>),
    tensor_cores: bool,
    shape: &str,
) -> Result<FoloModel, CliError> {
    let path = &crate::weights::resolve(path)?;
    let half = match precision {
        (_, Some(PrecisionArg::Int8)) => return open_int8(command, path, device, tensor_cores),
        (half, p) => half || p == Some(PrecisionArg::F16),
    };
    if tensor_cores {
        folo_gpu::context::request_tensor_cores();
    }
    crate::engine::init_device(device);
    let device = resolve_device(device).map_err(|e| engine_error(command, e))?;
    can_place(device).map_err(|e| engine_error(command, e))?;
    let fallback = if half { f16_unavailable(device) } else { None };
    if let Some(why) = &fallback {
        eprintln!("note: --half: {why}");
    }
    let precision = if half && fallback.is_none() { Precision::F16 } else { Precision::F32 };
    if tensor_cores {
        if precision != Precision::F16 {
            eprintln!(
                "note: --tensor-cores only affects f16 inference (--half on a GPU); running without them"
            );
        } else if let Some(ctx) = folo_gpu::context_for(device)
            && !ctx.tensor_cores()
        {
            eprintln!("note: --tensor-cores: {}", ctx.tensor_cores_status());
        }
    }
    let file = ModelFile::open(path).map_err(|e| weights_error(command, path, e))?;
    folo_vision::precompile::precompile_inference(&file, device, precision.dtype(), command, shape);
    FoloModel::from_file_for_inference(&file, device, precision.dtype()).map_err(|e| nn_error(command, e))
}

/// [`open_model`] with `--precision int8`: on the CPU (`auto` picks it; a GPU is a usage error), in f32 with the
/// int8 plan of the calibration `folo quantize` stored in the file.
fn open_int8(command: &str, path: &Path, device: &str, tensor_cores: bool) -> Result<FoloModel, CliError> {
    if !matches!(device.trim().to_ascii_lowercase().as_str(), "auto" | "" | "cpu") {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--precision int8 runs on the CPU, not --device {device}"),
        )
        .hint("pass --device cpu (or auto)"));
    }
    if !folo_vision::model::quant::uses_vnni() {
        eprintln!(
            "note: --precision int8: the int8 kernels take their portable path, several times slower than f32 \
             (they need AVX-VNNI, which this CPU lacks, AVX-512 VNNI not being used; or FOLO_Q8_SCALAR is set)"
        );
    }
    if tensor_cores {
        eprintln!("note: --tensor-cores only affects f16 inference (--half on a GPU); running without them");
    }
    let file = ModelFile::open(path).map_err(|e| weights_error(command, path, e))?;
    // A missing, unreadable or mismatched calibration: the weights are fine, the int8 calibration is not.
    let calibration_error = |why: String| {
        CliError::new(ExitCode::Runtime, "invalid_weights", format!("{}: {why}", folo_api::json::path(path)))
            .hint("run `folo quantize --model <weights> --data <dataset>` and use the file it writes")
    };
    let cal = match folo_vision::model::quant::Calibration::from_metadata(file.metadata()) {
        Ok(Some(cal)) => cal,
        Ok(None) => return Err(calibration_error("no int8 calibration".into())),
        Err(e) => return Err(calibration_error(format!("unreadable int8 calibration ({e})"))),
    };
    let mut model = FoloModel::from_file_for_inference(
        &file,
        resolve_device("cpu").map_err(|e| engine_error(command, e))?,
        Precision::F32.dtype(),
    )
    .map_err(|e| nn_error(command, e))?;
    model.set_int8(Some(cal)).map_err(|e| calibration_error(e.to_string()))?;
    Ok(model)
}

/// The precision a model runs in: `f32`, `f16` or `int8`.
pub(super) fn precision_of(model: &FoloModel) -> &'static str {
    if model.int8().is_some() {
        "int8"
    } else if model.dtype() == Precision::F16.dtype() {
        Precision::F16.as_str()
    } else {
        Precision::F32.as_str()
    }
}

/// `--fusion-iou` is an IoU: in `0..=1` (a NaN would put every candidate of a class in every cluster).
pub(super) fn check_fusion_iou(v: f32) -> Result<(), CliError> {
    if bounds::UNIT_INTERVAL.contains_f64(f64::from(v)) {
        return Ok(());
    }
    Err(CliError::new(ExitCode::Usage, "usage", format!("--fusion-iou must be in 0..=1, got {v}")))
}

fn invalid_config(command: &str, message: impl Into<String>) -> CliError {
    CliError::new(ExitCode::Usage, "invalid_config", message)
        .hint(format!("see `folo describe {command} --json` for the accepted ranges"))
}

/// Validates `--conf`, `--iou`, `--max-det`, `--imgsz`, and (for `val`) `--batch` before any model or dataset is
/// opened, matching the engine's accepted ranges so typos fail immediately with exit code 2 (`invalid_config`).
pub(super) fn check_infer_args(
    command: &str,
    conf: f32,
    iou: f32,
    max_det: u32,
    imgsz: Option<u32>,
    imgsz_bounds: bounds::NumericBounds,
    batch: Option<u32>,
) -> Result<(), CliError> {
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(conf)) {
        return Err(invalid_config(command, format!("--conf must be in 0..=1, got {conf}")));
    }
    if !bounds::UNIT_INTERVAL.contains_f64(f64::from(iou)) {
        return Err(invalid_config(command, format!("--iou must be in 0..=1, got {iou}")));
    }
    if !bounds::POSITIVE_INT.contains_u64(u64::from(max_det)) {
        return Err(invalid_config(command, "--max-det must be at least 1"));
    }
    if let Some(s) = imgsz
        && !imgsz_bounds.contains_u64(u64::from(s))
    {
        let msg = match imgsz_bounds.max {
            Some(bounds::BoundValue::Int(max)) => {
                format!("--imgsz must be a multiple of 32 in 32..={max}, got {s}")
            }
            _ => format!("--imgsz must be a positive multiple of 32, got {s}"),
        };
        return Err(invalid_config(command, msg));
    }
    if batch.is_some_and(|b| !bounds::POSITIVE_INT.contains_u64(u64::from(b))) {
        return Err(invalid_config(command, "--batch must be at least 1"));
    }
    Ok(())
}

pub fn val(args: ValArgs, out: Out) -> Result<(), CliError> {
    check_fusion_iou(args.fusion_iou)?;
    check_infer_args(
        "val",
        args.conf,
        args.iou,
        args.max_det,
        args.imgsz,
        bounds::VAL_IMGSZ,
        Some(args.batch),
    )?;
    let shape = format!("imgsz {:?} batch {} head {:?}", args.imgsz, args.batch, args.head);
    let model =
        open_model("val", &args.model, &args.device, (args.half, args.precision), args.tensor_cores, &shape)?;
    let split = match args.split {
        SplitArg::Train => Split::Train,
        SplitArg::Val => Split::Val,
        SplitArg::Test => Split::Test,
    };
    // Only the evaluated split's images are checked (COCO val: 5 000 images, not the 118 287 of train2017 too).
    let (ds, _) = open_dataset_splits(&args.data, folo_data::Format::Auto, &[split])?;
    let part = ds.split(split);
    if part.is_empty() {
        let mut err = CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("the dataset has no {} images", split.as_str()),
        );
        if !ds.report.issues.is_empty() {
            err = err.hint(folo_api::json::cli_command(
                "run `folo dataset check {data}` to inspect missing/corrupt images or malformed labels",
                &[("data", Path::new(&args.data))],
            ));
        } else {
            err = err.hint("pass --split train, val or test for a split that has images");
        }
        return Err(err);
    }
    let imgsz = args.imgsz.unwrap_or(model.imgsz());
    let cfg = ValConfig {
        nms: NmsConfig {
            conf: args.conf,
            iou: args.iou,
            max_det: args.max_det as usize,
            fusion: args.fusion.into(),
            fusion_iou: args.fusion_iou,
            score: args.score.into(),
            ..NmsConfig::val()
        },
        samples: 0,
        workers: args.workers.unwrap_or_else(folo_vision::train::default_workers),
        keep_detections: args.coco || args.save_json.is_some(),
        head: args.head.map(Into::into),
        ..ValConfig::new(imgsz, args.batch as usize)
    };
    // Validate model/dataset config and COCO category mappings *before* creating `--save-json` folders or running
    // expensive model inference.
    check_config(&model, part.class_names().len(), &cfg).map_err(|e| engine_error("val", e))?;
    let coco_ctx = if args.coco || args.save_json.is_some() {
        let gt = crate::coco::ground_truth(&args.data, split)?;
        let cats = crate::coco::validate_categories(
            &args.data,
            &gt,
            &ds.classes,
            &ds.report.issues,
            model.classes(),
        )?;
        Some((gt, cats))
    } else {
        if ds.classes != model.classes() {
            out.human(format!(
                "warning: the dataset's classes {:?} differ from the model's {:?}; class ids are compared as is",
                ds.classes,
                model.classes()
            ));
        }
        None
    };
    let device = model.device().to_string();
    let precision = precision_of(&model);
    let head = cfg.head.unwrap_or(model.head_mode());
    if let Some(path) = &args.save_json {
        crate::coco::prepare_results(path)?;
    }
    let r = validate(&model, Arc::new(part.clone()), &cfg).map_err(|e| engine_error("val", e))?;
    let m = val_metrics(&r.report);
    let names = model.classes();
    let name = |c: u32| names.get(c as usize).cloned().unwrap_or_else(|| c.to_string());
    let mut coco = serde_json::Value::Null;
    let mut coco_text = None;
    if let Some((gt, cats)) = &coco_ctx {
        if let Some(path) = &args.save_json {
            let n = crate::coco::write_results(path, gt, cats, part, &r.detections)?;
            out.human(format!("wrote {n} detections to {}", folo_api::json::path(path)));
        }
        if args.coco {
            let c = crate::coco::evaluate(gt, cats, part, &r.detections);
            coco = json!({
                "ap": c.stats.ap, "ap50": c.stats.ap50, "ap75": c.stats.ap75,
                "ap_small": c.stats.ap_small, "ap_medium": c.stats.ap_medium, "ap_large": c.stats.ap_large,
                "ar1": c.stats.ar1, "ar10": c.stats.ar10, "ar100": c.stats.ar100,
                "ar_small": c.stats.ar_small, "ar_medium": c.stats.ar_medium, "ar_large": c.stats.ar_large,
                "images": c.stats.images, "truths": c.stats.truths, "crowd": c.stats.crowd,
                "detections": c.stats.detections, "missing_images": c.missing_images,
                "annotations": folo_api::json::path(&gt.path),
                "per_class": c.stats.per_class.iter().map(|p| json!({
                    "class": p.class, "name": name(p.class), "truths": p.truths, "ap": p.ap, "ap50": p.ap50,
                })).collect::<Vec<_>>(),
            });
            coco_text = Some(c.stats.summary());
        }
    }
    out.record(&json!({
        "model": folo_api::json::path(&args.model),
        "data": folo_api::json::path(&args.data),
        "split": split.as_str(),
        "imgsz": imgsz,
        "images": r.report.num_images,
        "boxes": r.report.num_truths,
        "map50": m.map50,
        "map50_95": m.map50_95,
        "precision": m.precision,
        "recall": m.recall,
        "best_f1_conf": m.best_f1_conf,
        "per_class": m.per_class.iter().map(|c| json!({
            "class": c.class, "name": name(c.class), "map50": c.map50, "map50_95": c.map50_95,
            "precision": c.precision, "recall": c.recall,
        })).collect::<Vec<_>>(),
        "device": device,
        "dtype": precision,
        "head": head,
        "fusion": cfg.nms.fusion.as_str(),
        "score": cfg.nms.score.as_str(),
        "seconds": r.seconds,
        "coco": coco,
    }));
    out.human(format!(
        "{} on {} ({} images, {} boxes) at {imgsz}px on {device} ({precision}, head {head}) in {:.1}s",
        folo_api::json::path(&args.model),
        split.as_str(),
        r.report.num_images,
        r.report.num_truths,
        r.seconds
    ));
    let at = m.best_f1_conf.map_or(String::new(), |c| format!("  (P and R at conf {c:.3}, the best F1)"));
    out.human(format!("  {:<16} {:>7} {:>9} {:>7} {:>7}{at}", "class", "mAP50", "mAP50-95", "P", "R"));
    out.human(format!(
        "  {:<16} {:>7.3} {:>9.3} {:>7.3} {:>7.3}",
        "all", m.map50, m.map50_95, m.precision, m.recall
    ));
    for c in &m.per_class {
        out.human(format!(
            "  {:<16} {:>7.3} {:>9.3} {:>7.3} {:>7.3}",
            name(c.class),
            c.map50,
            c.map50_95,
            c.precision,
            c.recall
        ));
    }
    if let Some(text) = coco_text {
        out.human("COCOeval (pycocotools-exact, original-image boxes):");
        out.human(text.trim_end());
    }
    Ok(())
}

pub fn export(mut args: ExportArgs, out: Out) -> Result<(), CliError> {
    let onnx = args.format == ExportFormatArg::Onnx;
    if !onnx {
        let given = [
            ("--imgsz", args.imgsz.is_some()),
            ("--batch", args.batch.is_some()),
            ("--dynamic", args.dynamic),
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
    if args.dynamic && args.batch.is_some() {
        return Err(CliError::new(ExitCode::Usage, "usage", "--batch and --dynamic exclude each other")
            .hint("--dynamic makes the batch symbolic; leave out --batch"));
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
        let opts = OnnxOptions {
            height: imgsz,
            width: imgsz,
            batch: args.batch.unwrap_or(1),
            dynamic: args.dynamic,
            half: args.half,
            decode: args.decode,
            max_det: args.max_det,
            head: args.head.map(Into::into),
            // The source's other keys (folo.metrics, folo.created, unknown folo.* keys) travel along.
            metadata: meta.for_weights().to_map(),
        };
        let x = onnx::export_file(&file, &opts).map_err(|e| engine_error("export", e))?;
        output.write_bytes(&x.bytes)?;
        let output = output.publish()?;
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
            "path": folo_api::json::path(&output),
            "bytes": x.bytes.len(),
            "format": "onnx",
            "dtype": dtype_name,
            "fused": true,
            "source": source,
            "arch": meta.arch,
            "classes": meta.classes,
            "opset": onnx::OPSET,
            "imgsz": imgsz,
            "batch": if args.dynamic { json!("dynamic") } else { json!(opts.batch) },
            "dynamic": args.dynamic,
            "decode": args.decode,
            "head": x.metadata.get("folo.onnx.head"),
            "inputs": x.inputs.iter().map(value).collect::<Vec<_>>(),
            "outputs": x.outputs.iter().map(value).collect::<Vec<_>>(),
            "seconds": t.elapsed().as_secs_f64(),
        }));
        let shape = |v: &OnnxValue| v.shape.iter().map(ToString::to_string).collect::<Vec<_>>().join("x");
        out.human(format!(
            "Exported the {source} weights of {} to {} (ONNX opset {}, {}, {:.1} MB): {} {} -> {}",
            folo_api::json::path(&args.model),
            folo_api::json::path(&output),
            onnx::OPSET,
            dtype_name,
            x.bytes.len() as f64 / 1e6,
            onnx::INPUT,
            shape(&x.inputs[0]),
            x.outputs.iter().map(|o| format!("{} {}", o.name, shape(o))).collect::<Vec<_>>().join(", "),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_infer_args_validates_thresholds_and_sizes() {
        // Boundary valid values pass (`val` has no 8192 upper bound, matching `val::check_config`).
        check_infer_args("predict", 0.0, 0.0, 1, Some(32), bounds::INFER_IMGSZ, None).unwrap();
        check_infer_args("predict", 1.0, 1.0, 300, Some(8192), bounds::INFER_IMGSZ, None).unwrap();
        check_infer_args("val", 1.0, 1.0, 300, Some(8224), bounds::VAL_IMGSZ, Some(1)).unwrap();

        for (conf, iou, max_det, imgsz, imgsz_bounds, batch, flag) in [
            (-0.01, 0.5, 10, None, bounds::INFER_IMGSZ, None, "--conf"),
            (1.01, 0.5, 10, None, bounds::INFER_IMGSZ, None, "--conf"),
            (f32::NAN, 0.5, 10, None, bounds::INFER_IMGSZ, None, "--conf"),
            (f32::INFINITY, 0.5, 10, None, bounds::INFER_IMGSZ, None, "--conf"),
            (0.25, -0.1, 10, None, bounds::INFER_IMGSZ, None, "--iou"),
            (0.25, 1.5, 10, None, bounds::INFER_IMGSZ, None, "--iou"),
            (0.25, f32::NAN, 10, None, bounds::INFER_IMGSZ, None, "--iou"),
            (0.25, 0.7, 0, None, bounds::INFER_IMGSZ, None, "--max-det"),
            (0.25, 0.7, 10, Some(0), bounds::INFER_IMGSZ, None, "--imgsz"),
            (0.25, 0.7, 10, Some(33), bounds::INFER_IMGSZ, None, "--imgsz"),
            (0.25, 0.7, 10, Some(8224), bounds::INFER_IMGSZ, None, "--imgsz"),
            (0.25, 0.7, 10, Some(33), bounds::VAL_IMGSZ, Some(1), "--imgsz"),
            (0.25, 0.7, 10, Some(64), bounds::VAL_IMGSZ, Some(0), "--batch"),
        ] {
            let e = check_infer_args("predict", conf, iou, max_det, imgsz, imgsz_bounds, batch).unwrap_err();
            assert_eq!((e.exit, e.code.as_str()), (ExitCode::Usage, "invalid_config"), "{flag}: {e:?}");
            assert!(e.message.contains(flag), "expected {flag} in {}", e.message);
            assert!(e.hint.as_deref().is_some_and(|h| h.contains("folo describe predict --json")));
        }
    }
}
