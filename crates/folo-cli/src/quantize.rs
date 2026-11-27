//! `folo quantize`: calibrate int8 inference (docs/MODEL.md §9.6) on sample images of a dataset and write the
//! weights (fused, `f32`) with the calibration in their metadata (`folo.quant`, docs/FORMAT.md), for
//! `val` / `predict --precision int8`.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use folo_core::Tensor;
use folo_data::{DataLoader, Dataset, FileDataset, LoaderConfig, Split};
use folo_nn::Module;
use folo_nn::io::{FloatDType, ModelFile, write_export};
use folo_vision::model::quant::{self, Calibration, Method, SensitivityPolicy};
use folo_vision::train::resolve_device;
use folo_vision::{Detector, Precision};
use serde_json::json;

use crate::cli::{F32PolicyArg, QuantMethodArg, QuantizeArgs, SplitArg, bounds};
use crate::engine::{engine_error, open_dataset_splits};
use crate::output::{CliError, ExitCode, Out};
use crate::output_file::OutputFile;

fn nn_error(e: folo_nn::Error) -> CliError {
    engine_error("quantize", e.into())
}

fn usage(message: String) -> CliError {
    CliError::new(ExitCode::Usage, "usage", message)
}

/// Validated selective-`f32` calibration configuration derived from [`QuantizeArgs`].
struct SelectiveF32 {
    policy: F32PolicyArg,
    sensitivity: Option<SensitivityPolicy>,
    explicit_convs: Vec<String>,
}

fn check_selective_f32(args: &QuantizeArgs) -> Result<SelectiveF32, CliError> {
    let has_auto_tuning = args.max_f32_macs.is_some()
        || args.sqnr_threshold.is_some()
        || args.no_p5_head_out
        || args.no_pair_bottlenecks;
    if args.f32_policy != F32PolicyArg::Auto && has_auto_tuning {
        return Err(usage(
            "--max-f32-macs, --sqnr-threshold, --no-p5-head-out, and --no-pair-bottlenecks only apply to \
             `--f32-policy auto`"
                .into(),
        )
        .hint("remove those flags, or use `--f32-policy auto`"));
    }
    let sensitivity = if args.f32_policy == F32PolicyArg::Auto {
        let mut policy = SensitivityPolicy::default();
        if let Some(frac) = args.max_f32_macs {
            if !bounds::UNIT_INTERVAL.contains_f64(frac) {
                return Err(usage(format!("--max-f32-macs must be finite in 0..=1, got {frac}")));
            }
            policy.max_f32_mac_fraction = frac;
        }
        if let Some(db) = args.sqnr_threshold {
            if !db.is_finite() {
                return Err(usage(format!("--sqnr-threshold must be a finite number in dB, got {db}")));
            }
            policy.sqnr_threshold_db = db;
        }
        if args.no_p5_head_out {
            policy.include_p5_head_out = false;
        }
        if args.no_pair_bottlenecks {
            policy.pair_residual_bottlenecks = false;
        }
        policy.validate().map_err(|e| usage(e.to_string()))?;
        Some(policy)
    } else {
        None
    };

    let mut explicit_convs = Vec::new();
    for raw in &args.f32_conv {
        let name = raw.trim();
        if name.is_empty() {
            return Err(usage("--f32-conv entries must not be empty".into()));
        }
        if !explicit_convs.iter().any(|c| c == name) {
            explicit_convs.push(name.to_string());
        }
    }

    Ok(SelectiveF32 { policy: args.f32_policy, sensitivity, explicit_convs })
}

pub fn quantize(args: QuantizeArgs, out: Out) -> Result<(), CliError> {
    let method = match args.method {
        QuantMethodArg::Percentile if !bounds::QUANTIZE_TAIL.contains_f64(args.tail) => {
            return Err(usage(format!("--tail must be in (0, 0.5), got {}", args.tail)));
        }
        QuantMethodArg::Percentile => Method::Percentile { p: args.tail },
        QuantMethodArg::Minmax => Method::MinMax,
        QuantMethodArg::Mse => Method::Mse,
    };
    let selective = check_selective_f32(&args)?;
    if !bounds::POSITIVE_INT.contains_u64(args.images as u64)
        || !bounds::POSITIVE_INT.contains_u64(u64::from(args.batch))
    {
        return Err(usage("--images and --batch must be at least 1".into()));
    }
    if let Some(imgsz) = args.imgsz
        && !bounds::INFER_IMGSZ.contains_u64(u64::from(imgsz))
    {
        return Err(usage(format!(
            "--imgsz {imgsz} must be a multiple of 32 in 32..={}",
            bounds::INFER_IMGSZ_MAX
        )));
    }
    let path = crate::weights::resolve(&args.model)?;
    let file = ModelFile::open(&path).map_err(|e| crate::infer::weights_error("quantize", &path, e))?;
    let model = open_cpu(&file)?;
    if !selective.explicit_convs.is_empty() {
        validate_explicit_convs(&model, &selective.explicit_convs)?;
    }
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
    if !bounds::INFER_IMGSZ.contains_u64(u64::from(imgsz)) {
        let from = if args.imgsz.is_some() { "--imgsz" } else { "the weights' imgsz (pass --imgsz)" };
        return Err(usage(format!(
            "{from} {imgsz} must be a multiple of 32 in 32..={}",
            bounds::INFER_IMGSZ_MAX
        )));
    }
    let workers = args.workers.unwrap_or_else(folo_vision::train::default_workers);
    // Claimed first, so a taken --output fails before the calibration, not after it.
    let mut output = OutputFile::new("quantize", &path, args.output.clone(), "-int8", "folo")?;
    out.human(format!("Calibrating int8 on {n} {} images at {imgsz}px (two passes)", split.as_str()));
    let t = Instant::now();
    let (mut cal, n) =
        calibrate_raw(&model, part, args.images, args.seed, (imgsz, args.batch as usize, workers), method)?;
    let analysis = quant::analyze_sensitivity_detector(&model, &cal).map_err(nn_error)?;
    let by_name: HashMap<&str, &quant::ConvSensitivity> =
        analysis.iter().map(|e| (e.name.as_str(), e)).collect();
    for name in &selective.explicit_convs {
        if !by_name.contains_key(name.as_str()) {
            let sample: Vec<&str> = analysis.iter().take(6).map(|e| e.name.as_str()).collect();
            return Err(usage(format!(
                "--f32-conv `{name}` is not an int8-eligible backbone/neck conv of {}",
                model.arch()
            ))
            .hint(format!("examples of eligible conv names: {}", sample.join(", "))));
        }
    }
    cal.f32_convs = selective.explicit_convs.clone();
    cal = match selective.policy {
        F32PolicyArg::Auto => cal
            .with_sensitivity_f32_detector(&model, selective.sensitivity.expect("auto sets sensitivity"))
            .map_err(nn_error)?,
        F32PolicyArg::P5Tail => match &model {
            Detector::V1(m) => cal.with_p5_tail_f32(m),
            Detector::V2(m) => cal.with_p5_tail_f32_v2(m),
        },
        F32PolicyArg::None => cal,
    };
    write(&file, &model, &cal, imgsz, &mut output)?;
    let output = output.publish()?;
    let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    let method_name = match args.method {
        QuantMethodArg::Percentile => "percentile",
        QuantMethodArg::Minmax => "minmax",
        QuantMethodArg::Mse => "mse",
    };
    let policy_name = match selective.policy {
        F32PolicyArg::Auto => "auto",
        F32PolicyArg::P5Tail => "p5-tail",
        F32PolicyArg::None => "none",
    };
    let mut f32_mac_fraction = 0.0f64;
    let f32_layers: Vec<serde_json::Value> = cal
        .f32_convs
        .iter()
        .map(|name| {
            let explicit = selective.explicit_convs.iter().any(|c| c == name);
            match by_name.get(name.as_str()) {
                Some(s) => {
                    f32_mac_fraction += s.mac_fraction;
                    json!({
                        "name": s.name,
                        "mac_fraction": s.mac_fraction,
                        "folded_weight_sqnr_db": s.folded_weight_sqnr_db,
                        "unfolded_weight_sqnr_db": s.unfolded_weight_sqnr_db,
                        "explicit": explicit,
                    })
                }
                None => json!({ "name": name, "explicit": explicit }),
            }
        })
        .collect();
    let sensitivity_policy = selective.sensitivity.map(|p| {
        json!({
            "max_f32_mac_fraction": p.max_f32_mac_fraction,
            "sqnr_threshold_db": p.sqnr_threshold_db,
            "include_p5_head_out": p.include_p5_head_out,
            "pair_residual_bottlenecks": p.pair_residual_bottlenecks,
        })
    });
    out.record(&json!({
        "path": folo_api::json::path(&output),
        "bytes": bytes,
        "images": n,
        "imgsz": imgsz,
        "method": method_name,
        "tail": (args.method == QuantMethodArg::Percentile).then_some(args.tail),
        "boundaries": cal.ranges.len(),
        "f32_policy": policy_name,
        "sensitivity_policy": sensitivity_policy,
        "f32_convs": cal.f32_convs,
        "f32_mac_fraction": f32_mac_fraction,
        "f32_layers": f32_layers,
        "seconds": t.elapsed().as_secs_f64(),
    }));
    let f32_note = if cal.f32_convs.is_empty() {
        format!("0 f32 convs ({policy_name})")
    } else {
        format!(
            "{} f32 convs ({:.1}% MACs, {policy_name}: {})",
            cal.f32_convs.len(),
            f32_mac_fraction * 100.0,
            cal.f32_convs.join(", ")
        )
    };
    out.human(format!(
        "Wrote {} ({:.1} MB): {} int8 ranges ({method_name}), {f32_note} from {n} images in {:.1}s; use it with \
         --precision int8",
        folo_api::json::path(&output),
        bytes as f64 / 1e6,
        cal.ranges.len(),
        t.elapsed().as_secs_f64()
    ));
    Ok(())
}

/// Runs two-pass activation calibration on `images` randomly sampled images from `part` without yet applying any
/// selective-`f32` policy (mirroring `folo_vision::model::quant::calibrate_sample_detector`).
fn calibrate_raw(
    model: &Detector,
    part: &FileDataset,
    images: usize,
    seed: u64,
    (imgsz, batch, workers): (u32, usize, usize),
    method: Method,
) -> Result<(Calibration, usize), CliError> {
    let mut entries = part.entries().to_vec();
    folo_core::rng::Rng::seed(seed).shuffle(&mut entries);
    entries.truncate(images);
    let n = entries.len();
    let sample = FileDataset::new(part.split(), part.class_names().to_vec(), entries);
    let data_err = |e: folo_data::DataError| folo_nn::Error::Invalid(format!("calibration images: {e}"));
    let loader = DataLoader::new(
        std::sync::Arc::new(sample),
        LoaderConfig { workers, ..LoaderConfig::val(batch, imgsz) },
    )
    .map_err(|e| nn_error(data_err(e)))?;
    let batches = || {
        loader.epoch(0).map(|b| {
            let mut b = b.map_err(data_err)?;
            Ok(Tensor::from_vec(std::mem::take(&mut b.images), b.shape)?)
        })
    };
    let cal = quant::calibrate_detector(model, batches, method).map_err(nn_error)?;
    Ok((cal, n))
}

/// Write `model`'s fused tensors with `cal` to `output`, then open it again and check that the weights it loads are
/// the ones `cal` was measured on.
fn write(
    file: &ModelFile,
    model: &Detector,
    cal: &Calibration,
    imgsz: u32,
    output: &mut OutputFile,
) -> Result<(), CliError> {
    let mut meta = file.metadata().clone();
    meta.fused = true;
    // The file's default inference size is the one its ranges were measured at.
    meta.imgsz = imgsz;
    cal.store(&mut meta).map_err(nn_error)?;
    let tensors = model.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    output.write_with(|w| write_export(w, &meta, tensors, FloatDType::F32).map_err(nn_error))?;
    let back = ModelFile::open(output.tmp()).map_err(nn_error)?;
    let mut m = open_cpu(&back)?;
    let stored = Calibration::from_metadata(back.metadata()).map_err(nn_error)?;
    m.set_int8(stored).map_err(nn_error)?;
    if m.int8().is_none() {
        return Err(CliError::new(ExitCode::Runtime, "internal", "the written file lost its calibration"));
    }
    Ok(())
}

/// Zero-compute symbolic executor that walks a detector's quantised backbone/neck graph to discover every
/// boundary name so `quant::analyze_sensitivity_detector` can run upfront without calibration data.
struct BoundaryCollector {
    ranges: BTreeMap<String, [f32; 2]>,
    pending_attn_dims: Option<[usize; 4]>,
}

impl BoundaryCollector {
    fn record(&mut self, name: &str) {
        self.ranges.entry(name.to_string()).or_insert([0.0, 1.0]);
    }
}

impl quant::walk::Exec for BoundaryCollector {
    type A = [usize; 4];

    fn stem(&mut self, name: &str, conv: &folo_nn::ConvBnAct, x: &Tensor) -> folo_nn::Result<Self::A> {
        self.record(name);
        let [n, _, h, w] = x.dims4().map(|(n, c, h, w)| [n, c, h, w])?;
        let cfg = conv.conv.config();
        let (oh, ow) =
            folo_core::cpu::q8::out_size(h, w, cfg.kernel, cfg.stride, cfg.padding).unwrap_or((h / 2, w / 2));
        Ok([n, cfg.out_channels, oh, ow])
    }

    fn stem_v2(
        &mut self,
        name: &str,
        stem: &folo_vision::model::v2::StemV2,
        x: &Tensor,
    ) -> folo_nn::Result<Self::A> {
        self.record(name);
        let [n, _, h, w] = x.dims4().map(|(n, c, h, w)| [n, c, h, w])?;
        let cfg = stem.conv.conv.config();
        let (in_h, in_w) = if stem.spd { (h / 2, w / 2) } else { (h, w) };
        let (oh, ow) = folo_core::cpu::q8::out_size(in_h, in_w, cfg.kernel, cfg.stride, cfg.padding)
            .unwrap_or((h / 2, w / 2));
        Ok([n, cfg.out_channels, oh, ow])
    }

    fn conv(
        &mut self,
        name: &str,
        conv: &folo_nn::ConvBnAct,
        parts: &[&Self::A],
        res: Option<&Self::A>,
    ) -> folo_nn::Result<Self::A> {
        self.record(name);
        let [n, _, h, w] = *parts[0];
        let cfg = conv.conv.config();
        let (oh, ow) =
            folo_core::cpu::q8::out_size(h, w, cfg.kernel, cfg.stride, cfg.padding).unwrap_or((h, w));
        let (oh, ow) = match res {
            Some(r) => (oh.max(r[2]), ow.max(r[3])),
            None => (oh, ow),
        };
        Ok([n, cfg.out_channels, oh, ow])
    }

    fn rep_conv(
        &mut self,
        name: &str,
        rep: &folo_vision::model::v2::WsQARepConv3x3,
        x: &Self::A,
        _out_f32: bool,
    ) -> folo_nn::Result<Self::A> {
        self.record(name);
        let [n, _, h, w] = *x;
        let (oh, ow) = rep
            .fused
            .as_ref()
            .and_then(|c| {
                let cfg = c.config();
                folo_core::cpu::q8::out_size(h, w, cfg.kernel, cfg.stride, cfg.padding)
            })
            .unwrap_or((h, w));
        Ok([n, rep.out_channels, oh, ow])
    }

    fn conv_f32(&mut self, _name: &str, _conv: &folo_nn::ConvBnAct, _x: &Self::A) -> folo_nn::Result<Tensor> {
        Ok(Tensor::zeros([0]))
    }

    fn attend_and_pe(
        &mut self,
        a: &folo_vision::model::blocks::Attention,
        _qkv: &Tensor,
        b: usize,
        h: usize,
        w: usize,
    ) -> folo_nn::Result<Tensor> {
        self.pending_attn_dims = Some([b, a.proj.conv.config().in_channels, h, w]);
        Ok(Tensor::zeros([0]))
    }

    fn hyper_attend_and_scene(
        &mut self,
        h_ctx: &folo_vision::model::v2::HyperContext,
        _av: &Tensor,
        b: usize,
        h: usize,
        w: usize,
    ) -> folo_nn::Result<(Tensor, Tensor)> {
        self.pending_attn_dims = Some([b, h_ctx.c_g, h, w]);
        Ok((Tensor::zeros([0]), Tensor::zeros([0])))
    }

    fn fullpad_gate(
        &mut self,
        _gate: &folo_vision::model::v2::FullPadGate,
        _feat: &Tensor,
        _g_scene: &Tensor,
    ) -> folo_nn::Result<Tensor> {
        Ok(Tensor::zeros([0]))
    }

    fn conv_out_f32(
        &mut self,
        name: &str,
        _conv: &folo_nn::ConvBnAct,
        _parts: &[&Self::A],
    ) -> folo_nn::Result<Tensor> {
        self.record(name);
        Ok(Tensor::zeros([0]))
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> folo_nn::Result<Self::A> {
        self.record(name);
        let dims = match self.pending_attn_dims.take() {
            Some(d) => d,
            None => {
                let (n, c, h, w) = x.dims4()?;
                [n, c, h, w]
            }
        };
        Ok(dims)
    }

    fn dequantize(&mut self, _x: &Self::A) -> folo_nn::Result<Tensor> {
        Ok(Tensor::zeros([0]))
    }

    fn narrow(&mut self, x: &Self::A, _start: usize, len: usize) -> folo_nn::Result<Self::A> {
        Ok([x[0], len, x[2], x[3]])
    }

    fn max_pool(&mut self, x: &Self::A, _k: usize) -> folo_nn::Result<Self::A> {
        Ok(*x)
    }

    fn upsample2(&mut self, x: &Self::A) -> folo_nn::Result<Self::A> {
        Ok([x[0], x[1], 2 * x[2], 2 * x[3]])
    }

    fn space_to_depth2_parts(
        &mut self,
        x: &Self::A,
        _consumer_is_f32: bool,
    ) -> folo_nn::Result<[Self::A; 4]> {
        let part = [x[0], x[1], x[2] / 2, x[3] / 2];
        Ok([part; 4])
    }

    fn dims(&self, x: &Self::A) -> [usize; 4] {
        *x
    }
}

/// Symbolic zero-compute walk over `model`'s quantised backbone/neck graph to list every int8-eligible conv
/// name before opening calibration images or running two-pass calibration.
fn eligible_conv_names(model: &Detector) -> Result<Vec<String>, CliError> {
    let mut collector = BoundaryCollector { ranges: BTreeMap::new(), pending_attn_dims: None };
    let _g = folo_core::no_grad();
    match model {
        Detector::V1(m) => {
            let x = Tensor::zeros([1, m.config().in_channels, 64, 64]);
            let _ = quant::walk::features(m, &mut collector, &x).map_err(nn_error)?;
        }
        Detector::V2(m) => {
            let x = Tensor::zeros([1, m.config().in_channels, 64, 64]);
            let _ = quant::walk::features_v2(m, &mut collector, &x).map_err(nn_error)?;
        }
    }
    let dummy = Calibration {
        version: quant::VERSION_V2,
        method: Method::MinMax,
        images: 0,
        ranges: collector.ranges,
        channel_ranges: BTreeMap::new(),
        u8_stem: false,
        f32_convs: Vec::new(),
        weights: None,
    };
    let analysis = quant::analyze_sensitivity_detector(model, &dummy).map_err(nn_error)?;
    Ok(analysis.into_iter().map(|e| e.name).collect())
}

/// Validates every `--f32-conv` name against `model`'s int8-eligible convs upfront so typos fail before
/// calibration runs.
fn validate_explicit_convs(model: &Detector, explicit_convs: &[String]) -> Result<(), CliError> {
    let eligible = eligible_conv_names(model)?;
    for name in explicit_convs {
        if !eligible.iter().any(|c| c == name) {
            let sample: Vec<&str> = eligible.iter().take(6).map(String::as_str).collect();
            return Err(usage(format!(
                "--f32-conv `{name}` is not an int8-eligible backbone/neck conv of {}",
                model.arch()
            ))
            .hint(format!("examples of eligible conv names: {}", sample.join(", "))));
        }
    }
    Ok(())
}

/// `file`'s fused `f32` model on the CPU.
fn open_cpu(file: &ModelFile) -> Result<Detector, CliError> {
    let cpu = resolve_device("cpu").map_err(|e| engine_error("quantize", e))?;
    Detector::from_file_for_inference(file, cpu, Precision::F32.dtype()).map_err(nn_error)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn quantize_rejects_invalid_imgsz_before_opening_model_or_dataset() {
        for bad in [0, 31, 33, 8224] {
            let args = QuantizeArgs {
                model: PathBuf::from("nonexistent.folo"),
                data: "nonexistent-dataset".into(),
                split: SplitArg::Train,
                images: 8,
                seed: 0,
                method: QuantMethodArg::Percentile,
                tail: 1e-4,
                f32_policy: F32PolicyArg::Auto,
                max_f32_macs: None,
                sqnr_threshold: None,
                no_p5_head_out: false,
                no_pair_bottlenecks: false,
                f32_conv: Vec::new(),
                imgsz: Some(bad),
                batch: 4,
                workers: None,
                output: None,
            };
            let err = quantize(args, Out { json: false }).unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"), "imgsz {bad}: {err:?}");
            assert!(err.message.contains("--imgsz"), "{}", err.message);
        }
    }

    #[test]
    fn quantize_rejects_invalid_selective_f32_flags_upfront() {
        let base = || QuantizeArgs {
            model: PathBuf::from("nonexistent.folo"),
            data: "nonexistent-dataset".into(),
            split: SplitArg::Train,
            images: 8,
            seed: 0,
            method: QuantMethodArg::Percentile,
            tail: 1e-4,
            f32_policy: F32PolicyArg::Auto,
            max_f32_macs: None,
            sqnr_threshold: None,
            no_p5_head_out: false,
            no_pair_bottlenecks: false,
            f32_conv: Vec::new(),
            imgsz: None,
            batch: 4,
            workers: None,
            output: None,
        };
        // Out-of-range --max-f32-macs
        for bad in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            let err =
                quantize(QuantizeArgs { max_f32_macs: Some(bad), ..base() }, Out { json: true }).unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"), "{bad}: {err:?}");
            assert!(err.message.contains("--max-f32-macs"), "{}", err.message);
        }
        // Non-finite --sqnr-threshold
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = quantize(QuantizeArgs { sqnr_threshold: Some(bad), ..base() }, Out { json: true })
                .unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"), "{bad}: {err:?}");
            assert!(err.message.contains("--sqnr-threshold"), "{}", err.message);
        }
        // Auto-only flags with --f32-policy none / p5-tail
        for policy in [F32PolicyArg::None, F32PolicyArg::P5Tail] {
            let err = quantize(
                QuantizeArgs { f32_policy: policy, max_f32_macs: Some(0.1), ..base() },
                Out { json: true },
            )
            .unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"));
            assert!(err.message.contains("--f32-policy auto"), "{}", err.message);
        }
        // Empty --f32-conv entry
        let err =
            quantize(QuantizeArgs { f32_conv: vec!["  ".into()], ..base() }, Out { json: true }).unwrap_err();
        assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"));
        assert!(err.message.contains("--f32-conv"), "{}", err.message);
    }

    #[test]
    fn validate_explicit_convs_supports_v1_and_v2_detectors() {
        let mut v1 = folo_vision::FoloModel::with_rng(
            folo_vision::ModelConfig::new("n", 2).unwrap(),
            &mut folo_core::rng::Rng::seed(1),
        )
        .unwrap();
        v1.fuse().unwrap();
        let mut v2 = folo_vision::FoloV2Model::with_rng(
            folo_vision::FoloV2Config::new("n", 2).unwrap(),
            &mut folo_core::rng::Rng::seed(2),
        )
        .unwrap();
        v2.fuse().unwrap();

        for detector in [Detector::V1(Box::new(v1)), Detector::V2(Box::new(v2))] {
            validate_explicit_convs(&detector, &["neck.out5.cv2".to_string()])
                .expect("neck.out5.cv2 must be accepted as eligible");
            let err = validate_explicit_convs(&detector, &["bogus.conv".to_string()]).unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"), "{err:?}");
            assert!(err.message.contains("bogus.conv"), "{}", err.message);
            assert!(err.hint.as_deref().unwrap_or("").contains("examples of eligible conv names"), "{err:?}");
        }
    }
}
