use super::*;

/// `auto_rule`: new configs use rule 4 (MuSGD for every run, momentum 0.937, bias warmup from 0.1), settings stored before
/// the key existed (a checkpoint's `folo.train`) resume with rule 1, the reference fit; rule 3 keeps AdamW for
/// short runs; other versions are invalid.
#[test]
fn auto_rule_versions() {
    let cfg = TrainConfig { epochs: 30, batch: 16, ..TrainConfig::default() };
    assert_eq!(cfg.auto_rule, AUTO_RULE);
    let m = cfg.resolve_optimizer(2000, 80);
    assert_eq!((m.kind, m.lr0, m.momentum, m.warmup_bias_lr), (OptimKind::MuSgd, 0.01, 0.937, 0.1));
    let r = cfg.resolved_train(2000, 80);
    assert_eq!((r.optimizer, r.betas, r.warmup_bias_lr), (folo_api::ResolvedOptimizer::Musgd, None, 0.1));
    let r3 = TrainConfig { auto_rule: 3, ..cfg.clone() };
    assert_eq!(r3.resolve_optimizer(2000, 80).lr0, 0.001);
    assert_eq!(r3.resolve_optimizer(2000, 4).lr0, 0.00125, "4 classes are above the floor");
    let mut old = serde_json::to_value(&cfg).unwrap();
    old.as_object_mut().unwrap().remove("auto_rule");
    let old: TrainConfig = serde_json::from_value(old).unwrap();
    assert_eq!(old.auto_rule, 1);
    assert_eq!(old.resolve_optimizer(2000, 80).lr0, 0.000119);
    // Long runs: MuSGD (rules 3 and 4) vs SGD (rules 1 and 2), lr0 0.01 and momentum 0.9 either way.
    let long = TrainConfig { epochs: 300, auto_rule: 3, ..cfg.clone() };
    let m = long.resolve_optimizer(118_287, 80);
    assert_eq!((m.kind, m.lr0, m.momentum, m.warmup_bias_lr), (OptimKind::MuSgd, 0.01, 0.9, 0.0));
    for rule in [1, 2] {
        assert_eq!(
            TrainConfig { auto_rule: rule, ..long.clone() }.resolve_optimizer(118_287, 80).kind,
            OptimKind::Sgd
        );
    }
    assert!(TrainConfig { auto_rule: 0, ..cfg.clone() }.validate().is_err());
    assert!(TrainConfig { auto_rule: AUTO_RULE + 1, ..cfg }.validate().is_err());
}

/// `Run.resolved` says which MuSGD step weights a run trains with (not the configured `train.musgd`), only for
/// MuSGD, and a `config.json` written before the field still loads, without it.
#[test]
fn resolved_names_the_musgd_step_weights() {
    let musgd = |muon, sgd| Some(folo_api::ResolvedMusgd { muon, sgd, cls_lr: 3.0 });
    let cfg = TrainConfig { epochs: 30, batch: 16, ..TrainConfig::default() };
    assert_eq!(cfg.resolved_train(2000, 80).musgd, musgd(0.5, 0.5), "auto, from scratch");
    let tuned = TrainConfig { transfer: true, ..cfg.clone() };
    assert_eq!(tuned.resolved_train(2000, 80).musgd, musgd(0.2, 1.0), "auto, a fine-tune");
    let named = TrainConfig { optimizer: OptimizerChoice::Musgd, ..cfg.clone() };
    assert_eq!(named.resolved_train(2000, 80).musgd, musgd(0.2, 1.0), "asked for by name");
    let sgd = TrainConfig { optimizer: OptimizerChoice::Sgd, ..cfg.clone() };
    assert_eq!(sgd.resolved_train(2000, 80).musgd, None);
    let json = serde_json::to_value(sgd.resolved_train(2000, 80)).unwrap();
    assert!(json.get("musgd").is_none(), "{json}");
    // Written before the field: a `resolved` without it, and a file from before `resolved` itself.
    let mut old = serde_json::to_value(cfg.resolved_train(2000, 80)).unwrap();
    old.as_object_mut().unwrap().remove("musgd");
    let file = serde_json::json!({ "resolved": old });
    assert_eq!(ResolvedOptim::from_run_config(&file).unwrap().musgd, None);
    let file = serde_json::json!({
        "optimizer": { "type": "musgd", "lr0": 0.01, "momentum": 0.937, "warmup_bias_lr": 0.1 },
        "accumulate": 4, "steps_per_epoch": 13, "train": serde_json::to_value(&cfg).unwrap(),
    });
    assert_eq!(ResolvedOptim::from_run_config(&file).unwrap().musgd, None);
}

/// `auto` rule 5 picks MuSGD's step weights: 0.5 / 0.5 from scratch, `musgd.*` (0.2 / 1.0) for a run that starts
/// from weights and for any explicit `musgd` or older rule; `transfer` is missing (false) in older configs.
#[test]
fn auto_rule_5_musgd_weights() {
    use folo_nn::optim::{AUTO_MUSGD_SCRATCH_WEIGHTS, MuSgdWeights};
    let cfg = TrainConfig { epochs: 30, batch: 16, ..TrainConfig::default() };
    assert_eq!(cfg.auto_rule, 5);
    assert_eq!(cfg.resolve_optimizer(2000, 80).musgd, AUTO_MUSGD_SCRATCH_WEIGHTS);
    let tuned = TrainConfig { transfer: true, ..cfg.clone() };
    assert_eq!(tuned.resolve_optimizer(2000, 80).musgd, MuSgdWeights::default());
    let rule4 = TrainConfig { auto_rule: 4, ..cfg.clone() };
    assert_eq!(rule4.resolve_optimizer(2000, 80).musgd, MuSgdWeights::default());
    assert_eq!(rule4.resolve_optimizer(2000, 80).kind, cfg.resolve_optimizer(2000, 80).kind);
    let mut explicit = TrainConfig { optimizer: OptimizerChoice::Musgd, ..cfg.clone() };
    explicit.musgd.muon = 0.3;
    assert_eq!(explicit.resolve_optimizer(2000, 80).musgd, MuSgdWeights { muon: 0.3, sgd: 1.0 });
    let mut old = serde_json::to_value(&tuned).unwrap();
    old.as_object_mut().unwrap().remove("transfer");
    assert!(!serde_json::from_value::<TrainConfig>(old).unwrap().transfer);
}

/// `warmup_floor`: 0 by default, so `warmup_epochs` alone sets the warmup (a 205-image fine-tune, 4 steps per
/// epoch: 12 warmup steps); configs from before the key keep §8.4's 100-iteration floor.
#[test]
fn warmup_floor_versions() {
    let cfg = TrainConfig::default();
    assert_eq!(cfg.warmup_iters(4), 12);
    assert_eq!(TrainConfig { warmup_floor: 100, ..cfg.clone() }.warmup_iters(4), 100);
    let mut old = serde_json::to_value(cfg).unwrap();
    old.as_object_mut().unwrap().remove("warmup_floor");
    let old: TrainConfig = serde_json::from_value(old).unwrap();
    assert_eq!(old.warmup_floor, 100);
}

/// `o2o_keeps_tal`: on for a new run; a config from before the key trained the one-to-one branch with the §7.3
/// defaults and keeps doing so when it resumes.
#[test]
fn o2o_keeps_tal_versions() {
    let cfg = TrainConfig::default();
    assert!(cfg.o2o_keeps_tal);
    let mut old = serde_json::to_value(cfg).unwrap();
    old.as_object_mut().unwrap().remove("o2o_keeps_tal");
    assert!(!serde_json::from_value::<TrainConfig>(old).unwrap().o2o_keeps_tal);
    // A new run through the API, whose config has no such key.
    let api: folo_api::TrainConfig = serde_json::from_str(r#"{"dataset_id":"d"}"#).unwrap();
    assert!(TrainConfig::from_api(&api).unwrap().o2o_keeps_tal);
}

/// `musgd.ortho`: `polar4` for new runs (`default`, `from_api`); configs from before the key (missing
/// `musgd.ortho`, or missing `musgd` altogether) read as legacy `ns5` so older runs resume with legacy `ns5`.
#[test]
fn muon_ortho_versions() {
    use folo_nn::optim::MuonOrtho;
    let cfg = TrainConfig::default();
    assert_eq!(cfg.musgd.ortho, MuonOrtho::Polar4);
    let api: folo_api::TrainConfig = serde_json::from_str(r#"{"dataset_id":"d"}"#).unwrap();
    assert_eq!(TrainConfig::from_api(&api).unwrap().musgd.ortho, MuonOrtho::Polar4);
    // Legacy config with `musgd` block but no `ortho` field.
    let mut old = serde_json::to_value(&cfg).unwrap();
    old["musgd"].as_object_mut().unwrap().remove("ortho");
    assert_eq!(serde_json::from_value::<TrainConfig>(old).unwrap().musgd.ortho, MuonOrtho::Ns5);
    // Older config with no `musgd` key at all.
    let mut older = serde_json::to_value(&cfg).unwrap();
    older.as_object_mut().unwrap().remove("musgd");
    assert_eq!(serde_json::from_value::<TrainConfig>(older).unwrap().musgd.ortho, MuonOrtho::Ns5);
    // Explicit `polar5` and `ns5` round-trip.
    for ortho in [MuonOrtho::Polar4, MuonOrtho::Polar5, MuonOrtho::Ns5] {
        let mut c = cfg.clone();
        c.musgd.ortho = ortho;
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(serde_json::from_value::<TrainConfig>(v).unwrap().musgd.ortho, ortho);
    }
}

/// `ema_run_fraction`: 0.25 by default, so a fine-tune of 400 optimizer steps averages over 100 (decay 0.99,
/// tau 20) and one of 40,000 steps or more, or any run from scratch, keeps the default EMA exactly; configs from
/// before the key keep the default for every run; a negative or non-finite fraction is refused.
#[test]
fn ema_run_fraction_versions() {
    use folo_nn::{EMA_DECAY, EMA_TAU};
    let tuned = TrainConfig { transfer: true, ..TrainConfig::default() };
    assert_eq!(tuned.ema_run_fraction, 0.25);
    assert_eq!(tuned.ema_params(400), (0.99, 20.0));
    let default = (EMA_DECAY.to_bits(), EMA_TAU.to_bits());
    let bits = |(d, t): (f64, f64)| (d.to_bits(), t.to_bits());
    assert_eq!(bits(tuned.ema_params(40_000)), default, "a COCO-length fine-tune");
    assert_eq!(bits(tuned.ema_params(1_000_000)), default);
    assert_eq!(tuned.ema_params(39_996), (1.0 - 1.0 / 9999.0, 9999.0 / 5.0), "just under the cap");
    let scratch = TrainConfig::default();
    assert_eq!(bits(scratch.ema_params(400)), default, "a run from scratch");
    let mut old = serde_json::to_value(&tuned).unwrap();
    old.as_object_mut().unwrap().remove("ema_run_fraction");
    let old: TrainConfig = serde_json::from_value(old).unwrap();
    assert_eq!((old.ema_run_fraction, bits(old.ema_params(400))), (0.0, default));
    // `Run.resolved` carries the pair; a run without an EMA has none.
    let r = tuned.resolved_train(205, 3);
    assert_eq!((r.ema_decay, r.ema_tau), (Some(0.99), Some(20.0)), "BCCD: 4 steps an epoch, 400 in all");
    let off = TrainConfig { ema: false, ..tuned.clone() }.resolved_train(205, 3);
    assert_eq!((off.ema_decay, off.ema_tau), (None, None));
    // A config.json from before the fields: the pair its `train` gives (a fine-tune before the key: the default).
    let mut file = serde_json::json!({ "resolved": r, "train": serde_json::to_value(&tuned).unwrap() });
    file["resolved"].as_object_mut().unwrap().retain(|k, _| !k.starts_with("ema_"));
    file["train"].as_object_mut().unwrap().remove("ema_run_fraction");
    let back = ResolvedOptim::from_run_config(&file).unwrap();
    assert_eq!((back.ema_decay, back.ema_tau), (Some(EMA_DECAY), Some(EMA_TAU)));
    for f in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(TrainConfig { ema_run_fraction: f, ..tuned.clone() }.validate().is_err(), "{f} passed");
    }
}

/// `loss.tal.eps` divides the soft targets (`m · u_max / (m_max + eps)`): at 0 an anchor whose prediction
/// misses its box gets a NaN target score, so validation takes only a finite, strictly positive value.
#[test]
fn tal_eps_must_be_finite_and_positive() {
    use crate::boxes::Xyxy;
    let zero = crate::assign::TalConfig { eps: 0.0, ..Default::default() };
    let gt = Xyxy::new(0.0, 0.0, 32.0, 32.0);
    let miss = Xyxy::new(100.0, 100.0, 110.0, 110.0);
    let a = crate::assign::assign(&[0.5], 1, &[miss], &[[16.0, 16.0]], &[gt], &[0], &zero);
    assert!(a.target_scores[0].is_nan(), "eps 0 divides 0 by 0");
    let cfg = TrainConfig::default();
    assert!(cfg.validate().is_ok());
    for eps in [0.0, -1e-9, f32::NAN, f32::INFINITY] {
        let mut c = cfg.clone();
        c.loss.tal.eps = eps;
        assert!(c.validate().is_err(), "loss.tal.eps = {eps} passed");
    }
}

/// `Run.resolved`: `auto` resolved for a short run, and the same numbers back from a `config.json` with the
/// `resolved` object and from one written before it (rebuilt from its other keys).
#[test]
fn resolved_train_and_run_config() {
    // Rule 3 and the 100-step warmup floor, the settings of runs made before rule 4.
    let cfg = TrainConfig { epochs: 30, batch: 8, auto_rule: 3, warmup_floor: 100, ..TrainConfig::default() };
    let r = cfg.resolved_train(1000, 3);
    // 125 batches · 30 epochs over max(batch, nbs) = 64 is 480 iterations: a short run, so AdamW at
    // 0.002 · 5 / (4 + 3), its β1 0.9 and β2 0.999, no warmup bias LR.
    assert_eq!(r.optimizer, folo_api::ResolvedOptimizer::Adamw);
    assert_eq!((r.lr0, r.momentum, r.betas, r.warmup_bias_lr), (0.001429, 0.9, Some([0.9, 0.999]), 0.0));
    // accumulate round(64 / 8) = 8; 125 batches → 16 optimizer steps; weight decay 0.0005 · 8 · 8 / 64.
    assert_eq!((r.batch, r.accumulate, r.nbs, r.optimizer_steps_per_epoch), (8, 8, 64, 16));
    assert!((r.weight_decay - 0.0005).abs() < 1e-12, "{}", r.weight_decay);
    // 3 warmup epochs of 16 steps = 48 < the 100-step floor (30% of 480 = 144 allows it).
    assert_eq!(r.warmup_iters, 100);
    assert_eq!((r.warmup_epochs, r.warmup_momentum, r.lrf, r.cos_lr), (3.0, 0.8, 0.01, false));
    // An explicit optimizer keeps the config's LR and momentum; SGD has no betas.
    let sgd =
        TrainConfig { optimizer: OptimizerChoice::Sgd, lr0: 0.02, ..cfg.clone() }.resolved_train(1000, 3);
    assert_eq!(
        (sgd.optimizer, sgd.lr0, sgd.momentum, sgd.betas),
        (folo_api::ResolvedOptimizer::Sgd, 0.02, 0.937, None)
    );

    let new = serde_json::json!({ "resolved": r, "optimizer": { "type": "sgd" } });
    assert_eq!(ResolvedOptim::from_run_config(&new), Some(r.clone()), "the resolved object wins");
    let old = serde_json::json!({
        "train": cfg,
        "optimizer": { "type": "adamw", "lr0": 0.001429, "momentum": 0.9, "warmup_bias_lr": 0.0,
                       "weight_decay": 0.0005 },
        "batches_per_epoch": 125, "accumulate": 8, "steps_per_epoch": 16, "warmup_iters": 100,
    });
    assert_eq!(ResolvedOptim::from_run_config(&old), Some(r));
    for broken in [serde_json::json!({}), serde_json::json!({ "optimizer": { "type": "auto" }, "train": {} })]
    {
        assert_eq!(ResolvedOptim::from_run_config(&broken), None, "{broken}");
    }
}

#[test]
fn validates_rpiou_otdfl_and_geomuon_settings() {
    let mut ok = TrainConfig::default();
    ok.loss.box_loss = crate::loss::BoxLossKind::Rpiou;
    ok.loss.rpiou_beta = 0.25;
    ok.loss.rpiou_kappa = 1.5;
    ok.loss.rpiou_delta = 0.4;
    ok.loss.dfl_ot_weight = 0.25;
    ok.musgd.geo = folo_nn::optim::MuonGeoMode::Geo;
    ok.validate().unwrap();

    for bad in [
        {
            let mut c = ok.clone();
            c.loss.rpiou_beta = -0.01;
            c
        },
        {
            let mut c = ok.clone();
            c.loss.rpiou_beta = 1.01;
            c
        },
        {
            let mut c = ok.clone();
            c.loss.rpiou_kappa = 0.0;
            c
        },
        {
            let mut c = ok.clone();
            c.loss.rpiou_kappa = f32::INFINITY;
            c
        },
        {
            let mut c = ok.clone();
            c.loss.rpiou_delta = f32::NAN;
            c
        },
        {
            let mut c = ok.clone();
            c.loss.dfl_ot_weight = -0.1;
            c
        },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn backbone_lr_scale_and_freeze_validation_and_serde() {
    let def = TrainConfig::default();
    assert_eq!(def.backbone_lr_scale, 1.0);
    assert_eq!(def.freeze_backbone_epochs, 0);
    let json = serde_json::to_string(&def).unwrap();
    assert!(json.contains("backbone_lr_scale") && json.contains("freeze_backbone_epochs"));
    // Legacy configs missing both keys deserialize to the no-op defaults.
    let mut old = serde_json::to_value(&def).unwrap();
    old.as_object_mut().unwrap().remove("backbone_lr_scale");
    old.as_object_mut().unwrap().remove("freeze_backbone_epochs");
    let loaded: TrainConfig = serde_json::from_value(old).unwrap();
    assert_eq!(loaded.backbone_lr_scale, 1.0);
    assert_eq!(loaded.freeze_backbone_epochs, 0);

    let ok = TrainConfig { backbone_lr_scale: 0.1, freeze_backbone_epochs: 5, ..def.clone() };
    ok.validate().unwrap();
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(TrainConfig { backbone_lr_scale: bad, ..def.clone() }.validate().is_err());
    }
}
