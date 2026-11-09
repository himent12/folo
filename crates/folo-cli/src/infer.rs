//! `folo val`, `folo predict` and `folo export`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use folo_data::{Dataset, ImageFormat, Split};
use folo_nn::io::{FloatDType, ModelFile};
use folo_vision::export::onnx::{self, OnnxOptions, OnnxValue, proto::Dim};
use folo_vision::predict::f16_unavailable;
use folo_vision::train::{can_place, resolve_device};
use folo_vision::val::{ValConfig, val_metrics, validate};
use folo_vision::{FoloModel, NmsConfig, Precision, PredictConfig, Predictor};
use serde_json::json;

use crate::cli::{ExportArgs, ExportFormatArg, PredictArgs, SplitArg, ValArgs};
use crate::draw::annotate;
use crate::engine::{engine_error, open_dataset};
use crate::output::{CliError, ExitCode, Out};

fn nn_error(e: folo_nn::Error) -> CliError {
    engine_error(e.into())
}

/// A weights file that cannot be read: `invalid_weights` saying what the file is not, instead of a parser
/// detail such as a header length.
fn weights_error(path: &Path, e: folo_nn::Error) -> CliError {
    let mut err = nn_error(e);
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
/// `tensor_cores` asks for the tensor-core f16 kernels (a note on stderr when they are unavailable or unused).
fn open_model(path: &Path, device: &str, half: bool, tensor_cores: bool) -> Result<FoloModel, CliError> {
    let path = &crate::weights::resolve(path)?;
    if tensor_cores {
        folo_gpu::context::request_tensor_cores();
    }
    crate::engine::init_device(device);
    let device = resolve_device(device).map_err(engine_error)?;
    can_place(device).map_err(engine_error)?;
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
    let file = ModelFile::open(path).map_err(|e| weights_error(path, e))?;
    FoloModel::from_file_for_inference(&file, device, precision.dtype()).map_err(nn_error)
}

/// The precision a model runs in.
fn precision_of(model: &FoloModel) -> Precision {
    if model.dtype() == Precision::F16.dtype() { Precision::F16 } else { Precision::F32 }
}

pub fn val(args: ValArgs, out: Out) -> Result<(), CliError> {
    let model = open_model(&args.model, &args.device, args.half, args.tensor_cores)?;
    let (ds, _) = open_dataset(&args.data)?;
    let split = match args.split {
        SplitArg::Train => Split::Train,
        SplitArg::Val => Split::Val,
        SplitArg::Test => Split::Test,
    };
    let part = ds.split(split);
    if part.is_empty() {
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!("the dataset has no {} images", split.as_str()),
        ));
    }
    if ds.classes != model.classes() {
        out.human(format!(
            "warning: the dataset's classes {:?} differ from the model's {:?}; class ids are compared as is",
            ds.classes,
            model.classes()
        ));
    }
    let imgsz = args.imgsz.unwrap_or(model.imgsz());
    // Read the COCO ground truth first, so a non-COCO dataset fails before the (long) evaluation.
    let coco_gt = if args.coco || args.save_json.is_some() {
        Some(crate::coco::ground_truth(&args.data, split)?)
    } else {
        None
    };
    let cfg = ValConfig {
        nms: NmsConfig { conf: args.conf, iou: args.iou, ..NmsConfig::val() },
        samples: 0,
        workers: args.workers.unwrap_or_else(folo_vision::train::default_workers),
        keep_detections: coco_gt.is_some(),
        head: args.head.map(Into::into),
        ..ValConfig::new(imgsz, args.batch.max(1) as usize)
    };
    let device = model.device().to_string();
    let precision = precision_of(&model);
    let head = cfg.head.unwrap_or(model.head_mode());
    let r = validate(&model, Arc::new(part.clone()), &cfg).map_err(engine_error)?;
    let m = val_metrics(&r.report);
    let names = model.classes();
    let name = |c: u32| names.get(c as usize).cloned().unwrap_or_else(|| c.to_string());
    let mut coco = serde_json::Value::Null;
    let mut coco_text = None;
    if let Some(gt) = &coco_gt {
        if let Some(path) = &args.save_json {
            let n = crate::coco::write_results(path, gt, part, &r.detections)?;
            out.human(format!("wrote {n} detections to {}", folo_api::json::path(path)));
        }
        if args.coco {
            let c = crate::coco::evaluate(gt, part, &r.detections)?;
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

/// Every image under `sources` (folders recursively, in name order), with its path relative to the folder it was
/// found in (just the file name for a file source).
fn collect_images(sources: &[PathBuf]) -> Result<Vec<(PathBuf, PathBuf)>, CliError> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        let mut entries: Vec<PathBuf> =
            std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, out)?;
            } else if folo_data::is_image_path(&p) {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    for s in sources {
        if s.is_dir() {
            let mut found = Vec::new();
            walk(s, &mut found).map_err(|e| {
                CliError::new(
                    ExitCode::Runtime,
                    "io_error",
                    format!("cannot read {}: {e}", folo_api::json::path(s)),
                )
            })?;
            out.extend(found.into_iter().map(|p| {
                let rel = p.strip_prefix(s).map(Path::to_path_buf).unwrap_or_else(|_| p.clone());
                (p, rel)
            }));
        } else if s.is_file() {
            out.push((s.clone(), s.file_name().map_or_else(|| PathBuf::from("image"), PathBuf::from)));
        } else {
            return Err(CliError::new(
                ExitCode::NotFound,
                "path_not_found",
                format!("{} does not exist", folo_api::json::path(s)),
            ));
        }
    }
    if out.is_empty() {
        return Err(CliError::new(
            ExitCode::NotFound,
            "path_not_found",
            "no images found in the given sources",
        )
        .hint(format!("supported extensions: {}", folo_data::IMAGE_EXTENSIONS.join(", "))));
    }
    Ok(out)
}

/// Where `--save` writes each image: `<dir>/<relative path>` with a `.png` (PNG sources) or `.jpg` extension,
/// suffixed `_2`, `_3`, … when two sources map to the same name. Refuses any destination that is one of the
/// inputs, so annotations never overwrite a source image.
fn save_paths(dir: &Path, images: &[(PathBuf, PathBuf)]) -> Result<Vec<(PathBuf, ImageFormat)>, CliError> {
    let canon = |p: &Path| std::fs::canonicalize(p).ok();
    let inputs: std::collections::HashSet<PathBuf> = images.iter().filter_map(|(p, _)| canon(p)).collect();
    let mut used = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(images.len());
    for (path, rel) in images {
        let png = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"));
        let (fmt, ext) = if png { (ImageFormat::Png, "png") } else { (ImageFormat::Jpeg(92), "jpg") };
        let base = dir.join(rel).with_extension("");
        // Not `with_extension(ext)` again: for `a.b.jpg` it would replace the `b`.
        let mut dst = PathBuf::from(format!("{}.{ext}", base.display()));
        let mut i = 2;
        // Keyed case-insensitively: Windows and macOS file systems usually are, so `Cat.jpg` and `cat.jpg` must not
        // share an output (on a case-sensitive one this only adds a harmless suffix).
        while !used.insert(dst.to_string_lossy().to_lowercase()) {
            dst = PathBuf::from(format!("{}_{i}.{ext}", base.display()));
            i += 1;
        }
        if canon(&dst).is_some_and(|c| inputs.contains(&c)) {
            return Err(CliError::new(
                ExitCode::Usage,
                "usage",
                format!("--save would overwrite the input image {}", folo_api::json::path(path)),
            )
            .hint("pass a --save folder outside the source folders"));
        }
        out.push((dst, fmt));
    }
    Ok(out)
}

/// Write `bytes` to `dst` through a fresh temporary file renamed over it, so an existing `dst` (even a hard link
/// to a source image) is replaced as a directory entry, never overwritten in place.
fn write_new(dst: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let (tmp, mut f) = folo_nn::io::safetensors::create_temp_beside(dst).map_err(|e| e.to_string())?;
    let r = f.write_all(bytes).and_then(|_| f.sync_all()).and_then(|_| {
        drop(f);
        std::fs::rename(&tmp, dst)
    });
    if let Err(e) = r {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    Ok(())
}

pub fn predict(args: PredictArgs, out: Out) -> Result<(), CliError> {
    let model = open_model(&args.model, &args.device, args.half, args.tensor_cores)?;
    let images = collect_images(&args.sources)?;
    let cfg = PredictConfig {
        imgsz: args.imgsz.unwrap_or(model.imgsz()),
        conf: args.conf,
        iou: args.iou,
        max_det: args.max_det.max(1) as usize,
        rect: true,
        head: args.head.map(Into::into),
    };
    let predictor = Predictor::new(model, cfg).map_err(engine_error)?;
    let destinations = match &args.save {
        Some(dir) => Some(save_paths(dir, &images)?),
        None => None,
    };
    let classes = predictor.classes().to_vec();
    let device = predictor.device().to_string();
    let precision = predictor.precision();
    let (mut boxes, mut failed) = (0usize, 0usize);
    let mut sum = [0f64; 4];
    let t_all = Instant::now();
    for (i, (path, _)) in images.iter().enumerate() {
        let t = Instant::now();
        let img = match folo_data::read_image(path) {
            Ok(i) => i,
            Err(e) => {
                failed += 1;
                out.record(&json!({ "event": "error", "file": folo_api::json::path(path), "error": { "code": "image_decode_failed", "message": e.to_string() } }));
                out.human(format!("{}: {e}", folo_api::json::path(path)));
                continue;
            }
        };
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut p = predictor.predict(&img).map_err(engine_error)?;
        p.timing.preprocess_ms += decode_ms;
        p.timing.total_ms += decode_ms;
        let ti = p.timing;
        for (s, v) in sum.iter_mut().zip([ti.preprocess_ms, ti.inference_ms, ti.postprocess_ms, ti.total_ms])
        {
            *s += v;
        }
        boxes += p.detections.len();
        let mut saved = None;
        if let Some(dests) = &destinations {
            let (dst, fmt) = &dests[i];
            let mut canvas = img;
            annotate(&mut canvas, &p.detections);
            let io = |e: String| {
                CliError::new(
                    ExitCode::Runtime,
                    "io_error",
                    format!("cannot write {}: {e}", folo_api::json::path(dst)),
                )
            };
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent).map_err(|e| io(e.to_string()))?;
            }
            write_new(dst, &folo_data::encode(&canvas, *fmt).map_err(|e| io(e.to_string()))?)
                .map_err(|e| io(e.to_string()))?;
            saved = Some(dst.clone());
        }
        out.record(&json!({
            "event": "prediction",
            "file": folo_api::json::path(path),
            "image": { "width": p.width, "height": p.height },
            "boxes": p.scored_boxes(&classes),
            "timing": ti.to_api(),
            "device": device,
            "dtype": precision,
            "saved": saved.as_ref().map(folo_api::json::path),
        }));
        let summary: Vec<String> =
            p.scored_boxes(&classes).iter().take(6).map(|b| format!("{} {:.2}", b.name, b.score)).collect();
        let more = if p.detections.len() > 6 {
            format!(", +{} more", p.detections.len() - 6)
        } else {
            String::new()
        };
        out.human(format!(
            "{}: {} boxes ({}{more}) {:.1} ms",
            folo_api::json::path(path),
            p.detections.len(),
            summary.join(", "),
            ti.total_ms
        ));
    }
    let n = (images.len() - failed).max(1) as f64;
    // Rounded like the per-image timings (`folo_api::json::round_ms`): `49.0546`, not `49.054555000000015`.
    let mean = |i: usize| folo_api::json::round_ms(sum[i] / n);
    out.record(&json!({
        "event": "done",
        "images": images.len() - failed,
        "failed": failed,
        "boxes": boxes,
        "mean_timing": { "preprocess_ms": mean(0), "inference_ms": mean(1), "postprocess_ms": mean(2), "total_ms": mean(3) },
        "seconds": t_all.elapsed().as_secs_f64(),
        "saved_to": args.save,
    }));
    out.human(format!(
        "{} images, {boxes} boxes; mean {:.1} ms (pre {:.1}, inference {:.1}, post {:.1}){}",
        images.len() - failed,
        sum[3] / n,
        sum[0] / n,
        sum[1] / n,
        sum[2] / n,
        args.save
            .as_ref()
            .map_or(String::new(), |d| format!("; annotated images in {}", folo_api::json::path(d)))
    ));
    if failed == images.len() {
        return Err(CliError::new(ExitCode::Runtime, "image_decode_failed", "no image could be decoded"));
    }
    Ok(())
}

pub fn export(mut args: ExportArgs, out: Out) -> Result<(), CliError> {
    args.model = crate::weights::resolve(&args.model)?;
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
    if args.batch == Some(0) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--batch must be at least 1"));
    }
    let (dtype, suffix, ext) = match args.format {
        ExportFormatArg::FoloF16 => (FloatDType::F16, "-f16", "folo"),
        ExportFormatArg::Folo => (FloatDType::F32, "-fused", "folo"),
        ExportFormatArg::Weights => (FloatDType::F32, "-weights", "folo"),
        ExportFormatArg::Onnx if args.half => (FloatDType::F16, "-f16", "onnx"),
        ExportFormatArg::Onnx => (FloatDType::F32, "", "onnx"),
    };
    let file = ModelFile::open(&args.model).map_err(|e| weights_error(&args.model, e))?;
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
    if onnx && (imgsz == 0 || !imgsz.is_multiple_of(32) || imgsz > 8192) {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--imgsz {imgsz} must be a multiple of 32 in 32..=8192"),
        ));
    }
    // Like MCP `export`, never overwrite: the output is claimed with `create_new` (so concurrent exports never
    // share a file; the export then replaces the empty claim atomically). The default name gets `-2`, `-3`, ...
    // when taken; an existing --output is an error.
    let claim = |p: &Path| std::fs::OpenOptions::new().write(true).create_new(true).open(p).map(drop);
    let io = |p: &Path, e: std::io::Error| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot create {}: {e}", folo_api::json::path(p)),
        )
    };
    let output = match args.output.clone() {
        Some(o) => {
            let same = |a: &Path, b: &Path| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
                (Ok(a), Ok(b)) => a == b,
                _ => a == b,
            };
            if same(&o, &args.model) {
                return Err(CliError::new(ExitCode::Usage, "usage", "--output must differ from --model"));
            }
            match claim(&o) {
                Ok(()) => o,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(CliError::new(
                        ExitCode::Runtime,
                        "output_exists",
                        format!("{} exists; export never overwrites", folo_api::json::path(&o)),
                    )
                    .hint("pass another --output, or leave it out for a free name next to --model"));
                }
                Err(e) => return Err(io(&o, e)),
            }
        }
        None => {
            let stem =
                args.model.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
            let mut n = 1;
            loop {
                let tag = if n == 1 { String::new() } else { format!("-{n}") };
                let p = args.model.with_file_name(format!("{stem}{suffix}{tag}.{ext}"));
                match claim(&p) {
                    Ok(()) => break p,
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
                    Err(e) => return Err(io(&p, e)),
                }
            }
        }
    };
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
            max_det: args.max_det.max(1),
            head: args.head.map(Into::into),
            // The source's other keys (folo.metrics, folo.created, unknown folo.* keys) travel along.
            metadata: meta.for_weights().to_map(),
        };
        let written = onnx::export_file(&file, &opts).and_then(|x| x.write_to(&output).map(|()| x));
        let x = match written {
            Ok(x) => x,
            Err(e) => {
                let _ = std::fs::remove_file(&output); // only the file claimed above
                return Err(engine_error(e));
            }
        };
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
            "dtype": dtype.as_str(),
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
            dtype.as_str(),
            x.bytes.len() as f64 / 1e6,
            onnx::INPUT,
            shape(&x.inputs[0]),
            x.outputs.iter().map(|o| format!("{} {}", o.name, shape(o))).collect::<Vec<_>>().join(", "),
        ));
        return Ok(());
    }
    let fused = args.format != ExportFormatArg::Weights;
    let written = if fused {
        file.export(&output, dtype)
    } else {
        // The inference tensor set (EMA when present), unfused, written as a plain weights file.
        FoloModel::from_file(&file, folo_nn::io::Purpose::Inference)
            .and_then(|(model, _)| model.save(&output))
    };
    if let Err(e) = written {
        let _ = std::fs::remove_file(&output); // only the file claimed above
        return Err(nn_error(e));
    }
    let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    out.record(&json!({
        "path": folo_api::json::path(&output),
        "bytes": bytes,
        "format": match args.format {
            ExportFormatArg::FoloF16 => "folo-f16",
            ExportFormatArg::Weights => "weights",
            _ => "folo",
        },
        "dtype": dtype.as_str(),
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
        dtype.as_str(),
        folo_api::json::path(&output),
        bytes as f64 / 1e6
    ));
    Ok(())
}
