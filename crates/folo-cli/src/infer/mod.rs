//! `folo val`, `folo predict` and `folo export`.
//!
//! - `mod.rs`: shared model loading / arg checks and `val`.
//! - `diag.rs`: `folo val --diagnostics` / `--worst` per-image error diagnosis.
//! - `export.rs`: `folo export` (`.folo` and ONNX with independent dynamic axes and `folo2-*` support).
//! - `predict.rs`: `predict`, ByteTrack `--track`, recursive image walking, `--save` path checks and atomic saving.
//! - `thresholds.rs`: calibrated per-class F1 threshold persistence (`--save-thresholds`) and loading (`--thresholds`).

mod diag;
mod export;
mod predict;
mod thresholds;

pub use export::export;
pub use predict::predict;

use std::path::Path;
use std::sync::Arc;

use folo_data::{Dataset, Split};
use folo_nn::io::ModelFile;
use folo_vision::predict::f16_unavailable;
use folo_vision::train::{can_place, resolve_device};
use folo_vision::val::{ValConfig, check_config_detector, val_metrics, validate_detector};
use folo_vision::{Detector, NmsConfig, Precision};
use serde_json::json;

use crate::cli::{PrecisionArg, SplitArg, ValArgs, bounds};
use crate::engine::{engine_error, open_dataset_splits};
use crate::output::{CliError, ExitCode, Out};

pub(super) fn nn_error(command: &str, e: folo_nn::Error) -> CliError {
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
) -> Result<Detector, CliError> {
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
    Detector::from_file_for_inference(&file, device, precision.dtype()).map_err(|e| nn_error(command, e))
}

/// [`open_model`] with `--precision int8`: on the CPU (`auto` picks it; a GPU is a usage error), in f32 with the
/// int8 plan of the calibration `folo quantize` stored in the file.
fn open_int8(command: &str, path: &Path, device: &str, tensor_cores: bool) -> Result<Detector, CliError> {
    if !matches!(device.trim().to_ascii_lowercase().as_str(), "auto" | "" | "cpu") {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--precision int8 runs on the CPU, not --device {device}"),
        )
        .hint("pass --device cpu (or auto)"));
    }
    if let Some(note) =
        int8_kernel_note(&folo_server::system::cpu_simd(), crate::doctor::SimdOverrides::from_env())
    {
        eprintln!("{note}");
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
    let mut model = Detector::from_file_for_inference(
        &file,
        resolve_device("cpu").map_err(|e| engine_error(command, e))?,
        Precision::F32.dtype(),
    )
    .map_err(|e| nn_error(command, e))?;
    model.set_int8(Some(cal)).map_err(|e| calibration_error(e.to_string()))?;
    Ok(model)
}

/// Returns a stderr note when `--precision int8` is not running on a fast VNNI kernel, naming the actual
/// selected kernel (`AVX2`, `portable`, etc.), the exact reason for dispatch, and any active `FOLO_Q8_*` overrides.
fn int8_kernel_note(
    simd: &folo_server::system::CpuSimd,
    overrides: crate::doctor::SimdOverrides,
) -> Option<String> {
    if simd.int8_fast {
        return None;
    }
    let active = overrides.active_names();
    let has_x86_simd = simd.avx2 || simd.avx_vnni || simd.avx512_vnni;
    let has_vnni = simd.avx_vnni || simd.avx512_vnni;
    if simd.int8_kernel == "portable" {
        if overrides.scalar && has_x86_simd {
            let extra = if active.len() > 1 {
                format!("; overrides set: {}", active.join(", "))
            } else {
                String::new()
            };
            Some(format!(
                "note: --precision int8: using portable kernel (fast SIMD disabled by FOLO_Q8_SCALAR{extra}); \
                 expect slower than f32"
            ))
        } else {
            Some(
                "note: --precision int8: using portable kernel (this CPU lacks AVX-512 VNNI, AVX-VNNI, and AVX2); \
                 expect several times slower than f32"
                    .to_string(),
            )
        }
    } else if overrides.no_vnni && has_vnni {
        let extra =
            if active.len() > 1 { format!("; overrides set: {}", active.join(", ")) } else { String::new() };
        Some(format!(
            "note: --precision int8: using {} kernel (VNNI disabled by FOLO_Q8_NO_VNNI{extra}); slower than VNNI",
            simd.int8_kernel
        ))
    } else {
        let extra = if !active.is_empty() && has_x86_simd {
            format!("; overrides set: {}", active.join(", "))
        } else {
            String::new()
        };
        Some(format!(
            "note: --precision int8: using {} kernel (this CPU lacks AVX-512 VNNI / AVX-VNNI{extra}); \
             slower than VNNI",
            simd.int8_kernel
        ))
    }
}

/// The precision a model runs in: `f32`, `f16` or `int8`.
pub(super) fn precision_of(model: &Detector) -> &'static str {
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
    let diag_settings = diag::check_diag_args(&args)?;
    thresholds::check_save_thresholds_path(
        args.save_thresholds.as_deref(),
        &args.model,
        args.save_json.as_deref(),
    )?;
    let resolved_model = crate::weights::resolve(&args.model)?;
    thresholds::check_save_thresholds_path(
        args.save_thresholds.as_deref(),
        &resolved_model,
        args.save_json.as_deref(),
    )?;
    let shape = format!("imgsz {:?} batch {} head {:?}", args.imgsz, args.batch, args.head);
    let model = open_model(
        "val",
        &resolved_model,
        &args.device,
        (args.half, args.precision),
        args.tensor_cores,
        &shape,
    )?;
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
        keep_detections: args.coco || args.save_json.is_some() || diag_settings.is_some(),
        head: args.head.map(Into::into),
        ..ValConfig::new(imgsz, args.batch as usize)
    };
    // Validate model/dataset config and COCO category mappings *before* creating `--save-json` folders or running
    // expensive model inference.
    check_config_detector(&model, part.class_names().len(), &cfg).map_err(|e| engine_error("val", e))?;
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
    if let Some(path) = &args.save_thresholds {
        thresholds::prepare_save_thresholds(path, &resolved_model, args.save_json.as_deref())?;
    }
    let r = validate_detector(&model, Arc::new(part.clone()), &cfg).map_err(|e| engine_error("val", e))?;
    let m = val_metrics(&r.report);
    let cal = r.thresholds();
    let names = model.classes();
    if let Some(path) = &args.save_thresholds {
        thresholds::save_thresholds(path, cal, names)?;
    }
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
    let diag_out = diag_settings.map(|s| diag::compute(part, &ds.root, &r.detections, s));
    let mut record = json!({
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
        "thresholds": thresholds::thresholds_json(cal, names),
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
    });
    if let Some(path) = &args.save_thresholds {
        record["thresholds_saved"] = json!(crate::output::record_path(path));
    }
    if let Some(d) = &diag_out {
        record["diagnostics"] = d.json.clone();
    }
    out.record(&record);
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
    for line in thresholds::human_val_lines(cal, names, args.save_thresholds.as_deref()) {
        out.human(line);
    }
    if let Some(text) = coco_text {
        out.human("COCOeval (pycocotools-exact, original-image boxes):");
        out.human(text.trim_end());
    }
    if let Some(d) = diag_out {
        for line in d.lines {
            out.human(line);
        }
    }
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

    #[test]
    fn int8_kernel_note_names_active_kernel_and_overrides() {
        use crate::doctor::SimdOverrides;
        use folo_server::system::CpuSimd;

        // Fast VNNI tiers emit no slowness note.
        let fast = CpuSimd {
            avx2: true,
            avx_vnni: true,
            avx512_vnni: false,
            neon: false,
            int8_fast: true,
            int8_kernel: "AVX-VNNI".into(),
        };
        assert_eq!(int8_kernel_note(&fast, SimdOverrides::default()), None);

        // `FOLO_Q8_NO_VNNI` on a VNNI CPU selecting AVX2 names AVX2 and the override (never claims portable/missing ISA).
        let avx2_override = CpuSimd { int8_fast: false, int8_kernel: "AVX2".into(), ..fast.clone() };
        let note =
            int8_kernel_note(&avx2_override, SimdOverrides { no_vnni: true, ..Default::default() }).unwrap();
        assert!(
            note.contains("AVX2") && note.contains("FOLO_Q8_NO_VNNI") && !note.contains("portable"),
            "{note}"
        );

        // `FOLO_Q8_SCALAR` on a VNNI CPU names portable and the override (never claims the CPU lacks AVX-VNNI).
        let scalar_override = CpuSimd { int8_fast: false, int8_kernel: "portable".into(), ..fast };
        let note =
            int8_kernel_note(&scalar_override, SimdOverrides { scalar: true, ..Default::default() }).unwrap();
        assert!(
            note.contains("portable") && note.contains("FOLO_Q8_SCALAR") && !note.contains("lacks"),
            "{note}"
        );

        // `FOLO_Q8_SCALAR` + `FOLO_Q8_NO_VNNI` on an AVX2-only CPU credits `FOLO_Q8_SCALAR` as the cause and lists
        // both set overrides separately without blaming `FOLO_Q8_NO_VNNI` for disabling AVX2.
        let avx2_only_portable = CpuSimd {
            avx2: true,
            avx_vnni: false,
            avx512_vnni: false,
            neon: false,
            int8_fast: false,
            int8_kernel: "portable".into(),
        };
        let note = int8_kernel_note(
            &avx2_only_portable,
            SimdOverrides { scalar: true, no_vnni: true, no_avx512: false },
        )
        .unwrap();
        assert!(
            note.contains("disabled by FOLO_Q8_SCALAR;")
                && note.contains("overrides set: FOLO_Q8_SCALAR, FOLO_Q8_NO_VNNI"),
            "{note}"
        );

        // Plain AVX2-only host without overrides (and with unrelated `FOLO_Q8_NO_AVX512`) reports missing VNNI.
        let avx2_only = CpuSimd { int8_kernel: "AVX2".into(), ..avx2_only_portable };
        let note = int8_kernel_note(&avx2_only, SimdOverrides::default()).unwrap();
        assert!(
            note.contains("using AVX2 kernel") && note.contains("lacks AVX-512 VNNI / AVX-VNNI"),
            "{note}"
        );
        let note =
            int8_kernel_note(&avx2_only, SimdOverrides { no_avx512: true, ..Default::default() }).unwrap();
        assert!(note.contains("lacks AVX-512 VNNI / AVX-VNNI; overrides set: FOLO_Q8_NO_AVX512"), "{note}");

        // Non-x86 host (ARM64 NEON) with `FOLO_Q8_NO_VNNI` does not blame the override for portable dispatch.
        let neon = CpuSimd {
            avx2: false,
            avx_vnni: false,
            avx512_vnni: false,
            neon: true,
            int8_fast: false,
            int8_kernel: "portable".into(),
        };
        let note = int8_kernel_note(&neon, SimdOverrides { no_vnni: true, ..Default::default() }).unwrap();
        assert!(note.contains("portable") && !note.contains("FOLO_Q8_NO_VNNI"), "{note}");
    }
}
