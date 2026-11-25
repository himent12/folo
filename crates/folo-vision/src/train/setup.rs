//! Building a [`Trainer`]: a new run ([`Trainer::new`], optionally from weights), a resumed one
//! ([`Trainer::resume`]) and what both share: the model, EMA, optimizer, schedule, loader and the run's warnings.

use std::collections::BTreeMap;
use std::path::Path;

use folo_api::LogLevel;
use folo_core::Device;
use folo_core::rng::Rng;
use folo_data::{DataLoader, LoaderConfig};
use folo_nn::io::{BestState, ModelConfig, ModelFile};
use folo_nn::optim::{AdamW, GradAccumulator, MuSgd, OptimKind, Optimizer, Sgd, StandardGroups};
use folo_nn::{DecayKind, EmaState, GradScaler, LrSchedule, ModelEma, Module};

use super::events::Emitter;
use super::teacher::Teacher;
use super::{
    AMP_KEY, CLOSE_MOSAIC_KEY, NONFINITE_STREAK_KEY, ResumeOverrides, SHORT_RUN_STEPS, TRAIN_IMAGES_KEY,
    TrainConfig, TrainData, TrainOptions, Trainer, WIOU_IOU_MEAN_KEY, amp_support, can_place, place_model,
    resolve_device,
};
use crate::error::{Error, Result};
use crate::loss::LossConfig;
use crate::model::FoloModel;

/// The closing epochs a resumed run keeps: [`CLOSE_MOSAIC_KEY`] of its `folo.train`, or for a checkpoint from
/// before the cap, the old rule `min(close_mosaic, epochs)`.
pub fn resumed_close_mosaic(train: &serde_json::Value, cfg: &TrainConfig) -> u32 {
    match train.get(CLOSE_MOSAIC_KEY).and_then(|v| v.as_u64()) {
        Some(n) => u32::try_from(n).unwrap_or(u32::MAX).min(cfg.epochs),
        None => cfg.augment.close_mosaic.min(cfg.epochs),
    }
}

/// Warnings for a run too short to learn from, emitted as `warn` log lines after the run's first log line and
/// kept in `config.json` (`warnings`):
/// - a run from scratch (`from_scratch`) with fewer than [`SHORT_RUN_STEPS`] optimizer steps;
/// - a warmup longer than half of the run (docs/MODEL.md §8.4 caps only its 100-step floor, at 30% of the run,
///   so this takes an explicit `warmup_epochs` close to `epochs`).
pub fn schedule_warnings(
    cfg: &TrainConfig,
    sched: &LrSchedule,
    train_images: usize,
    from_scratch: bool,
) -> Vec<String> {
    let c = sched.contract();
    let (total, warmup, spe) = (sched.total_iters(), c.warmup_iters, c.steps_per_epoch);
    let mut out = Vec::new();
    if from_scratch && total < SHORT_RUN_STEPS {
        let acc = cfg.accumulate();
        let batch = if acc > 1 {
            format!("batch {} × accumulate {acc}", cfg.batch)
        } else {
            format!("batch {}", cfg.batch)
        };
        out.push(format!(
            "this run takes only {total} optimizer steps ({} epochs × {spe}: {train_images} training images, {batch}); \
             a detector trained from scratch usually needs a few hundred steps to learn anything, so expect an mAP \
             near 0. Train for at least {} epochs, add images, or start from pretrained weights",
            cfg.epochs,
            SHORT_RUN_STEPS.div_ceil(spe.max(1)),
        ));
    }
    if warmup.saturating_mul(2) > total {
        out.push(format!(
            "warmup takes {warmup} of the run's {total} optimizer steps, so the learning rate reaches lr0 only in the \
             second half of the run; lower warmup_epochs (now {}) or train for more epochs",
            cfg.warmup_epochs
        ));
    }
    out
}

impl Trainer {
    /// A new run of `cfg` on `data`.
    pub fn new(cfg: TrainConfig, data: TrainData, opts: TrainOptions) -> Result<Self> {
        let weights = opts.weights.clone();
        // Stored with the run (`folo.train`), so a resume resolves `auto` as this run started (§8.2, rule 5).
        let cfg = TrainConfig { transfer: weights.is_some(), ..cfg };
        let mut t = Self::build(cfg, data, opts)?;
        t.warnings = schedule_warnings(&t.cfg, &t.sched, t.data.train.len(), weights.is_none());
        if let Some(w) = weights {
            let file = ModelFile::open(&w)?;
            let report = t.model.transfer_from(&file)?;
            if t.cfg.gauge_norm {
                let gn = super::gauge::apply_floor_gauge_norm(&t.model)?;
                t.pending_logs.push((
                    LogLevel::Info,
                    format!(
                        "floor gauge norm: rescaled {}/{} channels across {}/{} Conv+BN layers",
                        gn.channels_rescaled, gn.total_channels, gn.layers_rescaled, gn.layers_visited
                    ),
                ));
            }
            if let Some(ema) = &mut t.ema {
                ema.reset(&t.model)?;
            }
            // What did not transfer keeps its fresh initialisation; usually the class outputs of another class
            // list (`skipped`: in the file with another shape), which is expected rather than a failure.
            let total = t.model.named_tensors().len();
            let kept = total.saturating_sub(report.loaded.len());
            let mut line = format!(
                "transferred {} of {total} tensors from {}",
                report.loaded.len(),
                folo_api::json::path(&w)
            );
            if kept > 0 {
                line.push_str(&format!("; the other {kept} keep their fresh initialisation"));
                if !report.skipped.is_empty() {
                    line.push_str(&format!(
                        " ({} of the file's tensors were not used: another shape, such as the class outputs of \
                         another class list, or not part of this model)",
                        report.skipped.len()
                    ));
                }
            }
            t.pending_logs.push((LogLevel::Info, line));
        }
        Ok(t)
    }

    /// Continue the run a checkpoint (`last.folo` or `best.folo`) belongs to, with its stored config
    /// (`folo.train`). `data` must be the same dataset (same class list, same number of training images).
    pub fn resume(
        checkpoint: impl AsRef<Path>,
        data: TrainData,
        opts: TrainOptions,
        overrides: ResumeOverrides,
    ) -> Result<Self> {
        let path = checkpoint.as_ref();
        let file = ModelFile::open(path)?;
        let c = file.checkpoint().ok_or_else(|| {
            Error::Config(format!(
                "{} is a weights file, not a training checkpoint",
                folo_api::json::path(path)
            ))
        })?;
        let mut cfg: TrainConfig = serde_json::from_value(c.train.clone()).map_err(|e| {
            Error::Config(format!("{}: folo.train is not a TrainConfig: {e}", folo_api::json::path(path)))
        })?;
        if let Some(w) = overrides.workers {
            cfg.workers = w;
        }
        if let Some(d) = overrides.device {
            cfg.device = d;
        }
        if let Some(f) = overrides.data_format {
            cfg.data_format = f;
        }
        if let Some(d) = overrides.distill {
            cfg.distill = Some(d);
        }
        cfg.augment.close_mosaic_fixed = Some(resumed_close_mosaic(&c.train, &cfg));
        if let Some(n) = c.train.get(TRAIN_IMAGES_KEY).and_then(|v| v.as_u64())
            && n != data.train.len() as u64
        {
            return Err(Error::Dataset(format!(
                "the checkpoint was trained on {n} training images, the dataset has {}; resume on the dataset the \
                 run started with",
                data.train.len()
            )));
        }
        if file.metadata().classes != data.classes {
            return Err(Error::Dataset(format!(
                "the checkpoint was trained on classes {:?}, the dataset has {:?}",
                file.metadata().classes,
                data.classes
            )));
        }
        let mut t = Self::build(cfg, data, opts)?;
        let ema = t.ema.as_mut().map(|e| e as &mut dyn EmaState);
        let r = file.resume(&t.model, t.opt.as_mut(), ema, Some(t.data.classes.clone()))?;
        let (want, got) = (t.sched.contract(), r.schedule.contract());
        if (want.epochs, want.steps_per_epoch, want.kind) != (got.epochs, got.steps_per_epoch, got.kind) {
            return Err(Error::Dataset(format!(
                "the checkpoint's schedule ({} epochs of {} steps) does not match this dataset ({} of {}); resume \
                 on the dataset the run started with",
                got.epochs, got.steps_per_epoch, want.epochs, want.steps_per_epoch
            )));
        }
        t.sched = r.schedule;
        // Whether the run started from weights is not recorded, so a resume repeats only the warmup warning.
        t.warnings = schedule_warnings(&t.cfg, &t.sched, t.data.train.len(), false);
        if t.scaler.is_some() {
            let v = file.metadata().extra.get(AMP_KEY).ok_or_else(|| {
                Error::Config(format!(
                    "{}: a mixed-precision run's checkpoint without {AMP_KEY}",
                    path.display()
                ))
            })?;
            t.scaler = Some(GradScaler::from_json(v)?);
        }
        t.epoch = u32::try_from(r.epoch).map_err(|_| Error::Config("folo.epoch out of range".into()))?;
        t.best = r.best;
        t.extra_meta = file.metadata().extra.clone();
        // Written afresh by `save_checkpoint` when the run has a scaler or uses WIoU v3.
        t.extra_meta.remove(AMP_KEY);
        t.extra_meta.remove(WIOU_IOU_MEAN_KEY);
        t.wiou_iou_mean = match file.metadata().extra.get(WIOU_IOU_MEAN_KEY) {
            Some(v) => {
                let parsed: f32 = v
                    .parse()
                    .map_err(|_| Error::Config(format!("{WIOU_IOU_MEAN_KEY} `{v}` is not a number")))?;
                if !(parsed.is_finite() && parsed > 0.0) {
                    return Err(Error::Config(format!("{WIOU_IOU_MEAN_KEY} `{v}` must be finite and > 0")));
                }
                parsed
            }
            None => 1.0,
        };
        t.last_saved = Some((path.to_path_buf(), t.epoch));
        t.consecutive_bad = match file.metadata().extra.get(NONFINITE_STREAK_KEY) {
            Some(v) => v
                .parse()
                .map_err(|_| Error::Config(format!("{NONFINITE_STREAK_KEY} `{v}` is not a count")))?,
            None => 0,
        };
        t.best_metrics = file.metadata().metrics.map(|m| folo_api::ValMetrics {
            map50: m.map50 as f32,
            map50_95: m.map50_95 as f32,
            precision: m.precision as f32,
            recall: m.recall as f32,
            per_class: Vec::new(),
            best_f1_conf: m.best_f1_conf.map(|c| c as f32),
        });
        t.pending_logs.push((
            LogLevel::Info,
            format!(
                "resumed from {} after epoch {} (optimizer step {})",
                folo_api::json::path(path),
                t.epoch,
                t.sched.step()
            ),
        ));
        Ok(t)
    }

    fn build(cfg: TrainConfig, data: TrainData, opts: TrainOptions) -> Result<Self> {
        cfg.validate()?;
        let n_train = data.train.len();
        let nc = data.classes.len();
        if nc == 0 {
            return Err(Error::Dataset("the dataset has no classes".into()));
        }
        if n_train == 0 {
            return Err(Error::Dataset("the dataset has no training images".into()));
        }
        if data.train.class_names().len() > nc {
            return Err(Error::Dataset(format!(
                "the training split names {} classes but the run has {nc}",
                data.train.class_names().len()
            )));
        }
        let mut pending_logs = Vec::new();
        let (close, close_eff) = (cfg.augment.close_mosaic, cfg.augment.close_mosaic_epochs(cfg.epochs));
        if cfg.augment.mosaic > 0.0 && cfg.augment.close_mosaic_fixed.is_none() && close > close_eff {
            pending_logs.push((
                LogLevel::Info,
                format!(
                    "close_mosaic {close} is more than half of the {} epochs: mosaic closes for the last {close_eff} \
                     instead",
                    cfg.epochs
                ),
            ));
        }
        let mut device = resolve_device(&cfg.device)?;
        if !device.is_cpu() && cfg.device.trim().eq_ignore_ascii_case("auto") {
            // `auto` must never pick a device the model cannot be placed on.
            if let Err(e) = can_place(device) {
                pending_logs.push((LogLevel::Warn, format!("{e}; training on the CPU")));
                device = Device::Cpu;
            }
        }

        if cfg.amp {
            amp_support(device)?;
        }
        let resolved = cfg.resolve_optimizer(n_train, nc);
        // The kernels an earlier run of this kind used compile on every core while the model and loader are built.
        crate::precompile::precompile(
            device,
            &format!(
                "train {} {} head {} {} nc {nc}",
                cfg.arch,
                if cfg.amp { "amp" } else { "f32" },
                cfg.head.as_str(),
                resolved.kind.as_str()
            ),
            &format!("imgsz {} batch {}", cfg.imgsz, cfg.batch),
        );
        let mut mconf = ModelConfig::new(cfg.scale()?, nc)?;
        mconf.head = cfg.head.as_str().into();
        let mut init = Rng::seed(cfg.seed);
        let mut model = FoloModel::with_rng(mconf.clone(), &mut init)?;
        // Safeguard #38: MAL requires focal bias prior b_0 = ln(0.01 / 0.99) = -4.59512 (not -11.54, which would
        // collapse epoch-1 negative gradients via p_0^gamma ~ 3e-8). Always apply to freshly initialized heads so
        // cross-class transfers (whose class bias is skipped by `transfer_from`) also start at -4.59512; same-class
        // transfers and resumes overwrite it when loading weights.
        if cfg.loss.mal_gamma > 0.0 {
            const MAL_BIAS_PRIOR: f32 = -4.595_12;
            for (name, p) in model.named_parameters() {
                if (name.starts_with("head.cls.") || name.starts_with("head.o2o.cls."))
                    && name.ends_with(".out.bias")
                {
                    folo_nn::init::constant_(&p, MAL_BIAS_PRIOR)?;
                }
            }
        }
        model.set_classes(data.classes.clone())?;
        model.set_imgsz(cfg.imgsz);
        let model = place_model(model, device)?;
        model.train();
        let ema = if cfg.ema {
            let mut shadow = FoloModel::new(mconf)?;
            shadow.set_classes(data.classes.clone())?;
            shadow.set_imgsz(cfg.imgsz);
            // §8.5: a fine-tune averages over a share of its optimizer steps (a resume restores the stored values).
            let total = cfg.epochs as u64 * n_train.div_ceil(cfg.batch).div_ceil(cfg.accumulate()) as u64;
            let (decay, tau) = cfg.ema_params(total);
            Some(ModelEma::with_params(place_model(shadow, device)?, &model, decay, tau)?)
        } else {
            None
        };
        // The frozen teacher of §8.9, if the run has one (a new run and a resume alike load it here).
        let teacher = match &cfg.distill {
            Some(d) => {
                let (t, logs) = Teacher::load(d, &model, device, cfg.amp, opts.run_dir.as_deref())?;
                pending_logs.extend(logs);
                Some(t)
            }
            None => None,
        };

        if resolved.kind == OptimKind::AdamW && cfg.cautious_wd {
            return Err(Error::Config("cautious_wd is supported for SGD and MuSGD only, not AdamW".into()));
        }
        if resolved.kind != OptimKind::MuSgd && cfg.cls_lr_close_scale != 1.0 {
            return Err(Error::Config(
                "cls_lr_close_scale applies only to MuSGD (which has a separate .cls parameter group)".into(),
            ));
        }

        let accumulate = cfg.accumulate();
        let wd = cfg.effective_wd(accumulate);
        let std_groups = StandardGroups::from_module(&model);
        let apply_wd_mode = |mut gs: Vec<folo_nn::optim::ParamGroup>| {
            if cfg.decoupled_wd || cfg.cautious_wd {
                for g in &mut gs {
                    g.decoupled_wd = cfg.decoupled_wd;
                    g.cautious_wd = cfg.cautious_wd;
                }
            }
            gs
        };
        let opt: Box<dyn Optimizer> = match resolved.kind {
            OptimKind::Sgd => Box::new(Sgd::new(apply_wd_mode(std_groups.into_groups(
                resolved.lr0,
                resolved.momentum,
                wd,
            )))?),
            OptimKind::AdamW => {
                Box::new(AdamW::new(std_groups.into_groups(resolved.lr0, resolved.momentum, wd))?)
            }
            // MODEL.md §8.2: the head's class branches (o2m and o2o) train at 3x the LR.
            OptimKind::MuSgd => Box::new(MuSgd::with_ortho_and_geo(
                apply_wd_mode(std_groups.into_boosted_groups(
                    resolved.lr0,
                    resolved.momentum,
                    wd,
                    |n| n.starts_with("head.cls.") || n.starts_with("head.o2o.cls."),
                    cfg.musgd.cls_lr,
                )),
                resolved.musgd,
                cfg.musgd.ortho,
                cfg.musgd.geo,
            )?),
        };
        let nb = n_train.div_ceil(cfg.batch);
        let spe = nb.div_ceil(accumulate);
        let sched = LrSchedule::with_warmup_iters(
            if cfg.cos_lr { DecayKind::Cosine } else { DecayKind::Linear },
            cfg.epochs as u64,
            spe as u64,
            cfg.lrf,
            cfg.warmup_iters(spe as u64),
            resolved.warmup_bias_lr,
            cfg.warmup_momentum,
        )?;
        let loader = DataLoader::new(
            data.train.clone(),
            LoaderConfig {
                batch_size: cfg.batch,
                imgsz: cfg.imgsz,
                workers: cfg.workers,
                shuffle: true,
                drop_last: false,
                seed: cfg.seed,
                augment: Some(cfg.augment),
                epochs: cfg.epochs,
                prefetch_batches: 2,
                u8_images: folo_gpu::context_for(device).is_some(),
                cache: opts.cache,
            },
        )?;
        let acc = GradAccumulator::new(&model.parameters());
        let loss_cfg = LossConfig { reg_max: model.head.reg_max, ..cfg.loss };

        let file = match &opts.run_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir.join("weights"))
                    .map_err(|e| Error::io(dir.join("weights"), e))?;
                let p = dir.join("metrics.jsonl");
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                    .map_err(|e| Error::io(&p, e))?;
                Some(std::io::BufWriter::new(f))
            }
            None => None,
        };
        let emitter = Emitter::new(opts.events, file, opts.step_interval, (cfg.epochs, nb as u64));
        let scaler = cfg.amp.then(GradScaler::new);
        Ok(Trainer {
            cfg,
            data,
            device,
            model,
            ema,
            teacher,
            opt,
            sched,
            acc,
            loader,
            loss_cfg,
            resolved,
            nb,
            accumulate,
            epoch: 0,
            best: BestState::default(),
            best_metrics: None,
            emitter,
            opts_dir: opts.run_dir,
            stop: opts.stop,
            pending_logs,
            warnings: Vec::new(),
            at_boundary: true,
            consecutive_bad: 0,
            wiou_iou_mean: 1.0,
            extra_meta: BTreeMap::new(),
            last_saved: None,
            scaler,
            prefetched: None,
            pending_write: None,
            after_epoch: opts.after_epoch,
        })
    }
}
