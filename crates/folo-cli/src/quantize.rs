//! `folo quantize`: calibrate int8 inference (docs/MODEL.md §9.6) on sample images of a dataset and write the
//! weights (fused, `f32`) with the calibration in their metadata (`folo.quant`, docs/FORMAT.md), for
//! `val` / `predict --precision int8`.

use std::time::Instant;

use folo_data::{Dataset, Split};
use folo_nn::Module;
use folo_nn::io::{FloatDType, ModelFile, export_tensors};
use folo_vision::model::quant::{self, Calibration, Method};
use folo_vision::train::resolve_device;
use folo_vision::{FoloModel, Precision};
use serde_json::json;

use crate::cli::{QuantMethodArg, QuantizeArgs, SplitArg};
use crate::engine::{engine_error, open_dataset_splits};
use crate::output::{CliError, ExitCode, Out};
use crate::output_file::OutputFile;

fn nn_error(e: folo_nn::Error) -> CliError {
    engine_error("quantize", e.into())
}

fn usage(message: String) -> CliError {
    CliError::new(ExitCode::Usage, "usage", message)
}

pub fn quantize(args: QuantizeArgs, out: Out) -> Result<(), CliError> {
    let method = match args.method {
        QuantMethodArg::Percentile if !(args.tail > 0.0 && args.tail < 0.5) => {
            return Err(usage(format!("--tail must be in (0, 0.5), got {}", args.tail)));
        }
        QuantMethodArg::Percentile => Method::Percentile { p: args.tail },
        QuantMethodArg::Minmax => Method::MinMax,
        QuantMethodArg::Mse => Method::Mse,
    };
    if args.images == 0 || args.batch == 0 {
        return Err(usage("--images and --batch must be at least 1".into()));
    }
    let path = crate::weights::resolve(&args.model)?;
    let file = ModelFile::open(&path).map_err(|e| crate::infer::weights_error("quantize", &path, e))?;
    let model = open_cpu(&file)?;
    let split = match args.split {
        SplitArg::Train => Split::Train,
        SplitArg::Val => Split::Val,
        SplitArg::Test => Split::Test,
    };
    let (ds, _) = open_dataset_splits(&args.data, folo_data::Format::Auto, &[split])?;
    let part = ds.split(split);
    if part.is_empty() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("the dataset has no {} images", split.as_str()),
        ));
    }
    let n = args.images.min(part.len());
    let imgsz = args.imgsz.unwrap_or(model.imgsz());
    if imgsz == 0 || !imgsz.is_multiple_of(32) || imgsz > 8192 {
        let from = if args.imgsz.is_some() { "--imgsz" } else { "the weights' imgsz (pass --imgsz)" };
        return Err(usage(format!("{from} {imgsz} must be a multiple of 32 in 32..=8192")));
    }
    let workers = args.workers.unwrap_or_else(folo_vision::train::default_workers);
    // Claimed first, so a taken --output fails before the calibration, not after it.
    let output = OutputFile::new("quantize", &path, args.output.clone(), "-int8", "folo")?;
    out.human(format!("Calibrating int8 on {n} {} images at {imgsz}px (two passes)", split.as_str()));
    let t = Instant::now();
    let sizes = (imgsz, args.batch as usize, workers);
    let done = quant::calibrate_sample(&model, part, args.images, args.seed, sizes, method)
        .map_err(nn_error)
        .and_then(|(cal, n)| write(&file, &model, &cal, imgsz, output.tmp()).map(|()| (cal, n)));
    let (cal, n) = done?;
    let output = output.publish()?;
    let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    let method_name = match args.method {
        QuantMethodArg::Percentile => "percentile",
        QuantMethodArg::Minmax => "minmax",
        QuantMethodArg::Mse => "mse",
    };
    out.record(&json!({
        "path": folo_api::json::path(&output),
        "bytes": bytes,
        "images": n,
        "imgsz": imgsz,
        "method": method_name,
        "tail": (args.method == QuantMethodArg::Percentile).then_some(args.tail),
        "boundaries": cal.ranges.len(),
        "seconds": t.elapsed().as_secs_f64(),
    }));
    out.human(format!(
        "Wrote {} ({:.1} MB): {} int8 ranges ({method_name}) from {n} images in {:.1}s; use it with --precision int8",
        folo_api::json::path(&output),
        bytes as f64 / 1e6,
        cal.ranges.len(),
        t.elapsed().as_secs_f64()
    ));
    Ok(())
}

/// Write `model`'s fused tensors with `cal` to `output`, then open it again and check that the weights it loads are
/// the ones `cal` was measured on.
fn write(
    file: &ModelFile,
    model: &FoloModel,
    cal: &Calibration,
    imgsz: u32,
    output: &std::path::Path,
) -> Result<(), CliError> {
    let mut meta = file.metadata().clone();
    meta.fused = true;
    // The file's default inference size is the one its ranges were measured at.
    meta.imgsz = imgsz;
    cal.store(&mut meta).map_err(nn_error)?;
    let tensors = model.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    export_tensors(output, &meta, tensors, FloatDType::F32).map_err(nn_error)?;
    let back = ModelFile::open(output).map_err(nn_error)?;
    let mut m = open_cpu(&back)?;
    let stored = Calibration::from_metadata(back.metadata()).map_err(nn_error)?;
    m.set_int8(stored).map_err(nn_error)?;
    if m.int8().is_none() {
        return Err(CliError::new(ExitCode::Runtime, "internal", "the written file lost its calibration"));
    }
    Ok(())
}

/// `file`'s fused `f32` model on the CPU.
fn open_cpu(file: &ModelFile) -> Result<FoloModel, CliError> {
    let cpu = resolve_device("cpu").map_err(|e| engine_error("quantize", e))?;
    FoloModel::from_file_for_inference(file, cpu, Precision::F32.dtype()).map_err(nn_error)
}
