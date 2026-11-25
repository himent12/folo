//! MuSGD checkpoint contracts and bit-exact resume: step weights, Polar-FOLO schedules, and Conv-GeoMuon.

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_nn::io::*;
use folo_nn::optim::*;
use folo_nn::*;
use serde_json::Value;

use crate::checkpoint::{Run, new_run, new_run_with, save, snapshot, train_steps, with_parts};
use crate::{classes, meta, parse_independently, perturb, rewrite, tmp};

#[test]
fn musgd_weights_resume_through_the_checkpoint() {
    let w = MuSgdWeights { muon: 0.5, sgd: 0.5 };
    let mut a = new_run_with(OptimKind::MuSgd, 10, w);
    train_steps(&mut a, 3);
    let path = tmp("resume-musgd-weights.folo");
    save(&a, &path);
    let (mut map, tensors) =
        with_parts(&a, |p| checkpoint_contents(&meta(FileKind::Checkpoint, 3), p).unwrap());
    train_steps(&mut a, 3);
    let want = snapshot(&a);
    let optim_json = |path: &std::path::Path| -> Value {
        let p = parse_independently(&std::fs::read(path).unwrap());
        serde_json::from_str(p.header["__metadata__"]["folo.optim"].as_str().unwrap()).unwrap()
    };
    assert_eq!(optim_json(&path)["musgd"], serde_json::json!({ "muon": 0.5, "sgd": 0.5 }));

    // A fresh run with the default weights takes the file's and continues bit-exactly.
    let mut b = new_run(OptimKind::MuSgd, 77);
    let f = ModelFile::open(&path).unwrap();
    let res = f.resume(&b.model, b.opt.as_mut(), Some(&mut b.ema), Some(classes(3))).unwrap();
    assert_eq!(b.opt.contract().musgd, Some(w));
    b.sched = res.schedule;
    b.rng = res.rng;
    drop(f);
    train_steps(&mut b, 3);
    let got = snapshot(&b);
    assert_eq!(got.len(), want.len());
    for ((n1, b1), (n2, b2)) in got.iter().zip(&want) {
        assert_eq!(n1, n2);
        assert!(b1 == b2, "`{n1}` differs after resuming with MuSGD weights {w:?}");
    }

    // The default weights are not written: such a file reads as before the key.
    let d = new_run(OptimKind::MuSgd, 11);
    let dpath = tmp("resume-musgd-default.folo");
    save(&d, &dpath);
    assert!(optim_json(&dpath).get("musgd").is_none());

    // A contract without the key (written before it) keeps the weights the optimizer was built with: the defaults
    // for `MuSgd::new`, the trainer's `TrainConfig.musgd` (passed to `with_weights`) for a run that recorded custom
    // weights only in `folo.train`.
    let mut optim = optim_json(&path);
    optim.as_object_mut().unwrap().remove("musgd");
    map.insert("folo.optim".into(), optim.to_string());
    let legacy = rewrite(&map, &tensors).unwrap();
    let mut c = new_run_with(OptimKind::MuSgd, 78, w);
    legacy.resume(&c.model, c.opt.as_mut(), Some(&mut c.ema), Some(classes(3))).unwrap();
    assert_eq!(c.opt.contract().musgd, Some(w), "built with the trainer's weights: kept");
    let mut c = new_run(OptimKind::MuSgd, 79);
    legacy.resume(&c.model, c.opt.as_mut(), Some(&mut c.ema), Some(classes(3))).unwrap();
    assert_eq!(c.opt.contract().musgd, None, "built with the defaults: the defaults");

    // `musgd` belongs to MuSGD contracts only, and its weights are checked.
    optim["type"] = "sgd".into();
    optim["musgd"] = serde_json::json!({ "muon": 0.5, "sgd": 0.5 });
    map.insert("folo.optim".into(), optim.to_string());
    assert!(rewrite(&map, &tensors).err().unwrap().to_string().contains("musgd"));
    optim["type"] = "musgd".into();
    optim["musgd"] = serde_json::json!({ "muon": -1.0, "sgd": 0.5 });
    map.insert("folo.optim".into(), optim.to_string());
    assert!(rewrite(&map, &tensors).err().unwrap().to_string().contains("weights"));
}

#[test]
fn musgd_polar_ortho_resumes_through_the_checkpoint() {
    let new_ortho = |seed: u64, ortho: MuonOrtho| -> Run {
        let mut r = new_run_with(OptimKind::MuSgd, seed, MuSgdWeights::default());
        let sg = StandardGroups::from_module(&r.model);
        let last = sg.decay.last().unwrap().0.rsplit_once('.').unwrap().0.to_string() + ".";
        let groups = sg.into_boosted_groups(0.01, 0.9, 5e-4, |n| n.starts_with(&last), 3.0);
        r.opt = Box::new(MuSgd::with_ortho(groups, MuSgdWeights::default(), ortho).unwrap());
        r
    };
    let optim_json = |path: &std::path::Path| -> Value {
        let p = parse_independently(&std::fs::read(path).unwrap());
        serde_json::from_str(p.header["__metadata__"]["folo.optim"].as_str().unwrap()).unwrap()
    };

    // Default MuSGD (`Polar4`) writes `"muon_ortho": "polar4"` and resumes bit-identically even into an `Ns5` run.
    let mut a = new_run(OptimKind::MuSgd, 21);
    train_steps(&mut a, 3);
    let path = tmp("resume-musgd-polar4.folo");
    save(&a, &path);
    let (mut map, tensors) =
        with_parts(&a, |p| checkpoint_contents(&meta(FileKind::Checkpoint, 3), p).unwrap());
    train_steps(&mut a, 3);
    let want = snapshot(&a);
    assert_eq!(optim_json(&path)["muon_ortho"], "polar4");

    let mut b = new_ortho(88, MuonOrtho::Ns5);
    let f = ModelFile::open(&path).unwrap();
    let res = f.resume(&b.model, b.opt.as_mut(), Some(&mut b.ema), Some(classes(3))).unwrap();
    assert_eq!(b.opt.contract().muon_ortho, Some(MuonOrtho::Polar4));
    b.sched = res.schedule;
    b.rng = res.rng;
    drop(f);
    train_steps(&mut b, 3);
    assert_eq!(snapshot(&b), want, "Polar4 checkpoint resume must be bit-identical");

    // Legacy `Ns5` omits `muon_ortho` in `folo.optim` (byte-identical to pre-Polar `folo.optim`) and resumes
    // bit-identically on CPU as `Ns5` even when loaded into a default-`Polar4` `MuSgd::new`. Also verify a
    // synthetic pre-Polar checkpoint whose `folo.optim` has no `muon_ortho` and whose `folo.train` has a
    // pre-Polar `musgd` block without `ortho` (or no `musgd` key at all) loads and resumes cleanly as `Ns5`.
    let mut leg = new_ortho(22, MuonOrtho::Ns5);
    train_steps(&mut leg, 3);
    let dpath = tmp("resume-musgd-ns5.folo");
    save(&leg, &dpath);
    let (leg_map, leg_tensors) =
        with_parts(&leg, |p| checkpoint_contents(&meta(FileKind::Checkpoint, 3), p).unwrap());
    assert!(optim_json(&dpath).get("muon_ortho").is_none());
    let pre_polar_files: Vec<(&str, ModelFile)> = [
        (
            "pre-polar train with musgd without ortho",
            serde_json::json!({ "epochs": 2, "batch": 2, "musgd": { "muon": 0.2, "sgd": 1.0, "cls_lr": 3.0 } }),
        ),
        ("pre-polar train without musgd", serde_json::json!({ "epochs": 2, "batch": 2 })),
    ]
    .into_iter()
    .map(|(label, tj)| {
        let mut m = leg_map.clone();
        m.insert("folo.train".into(), tj.to_string());
        (label, rewrite(&m, &leg_tensors).unwrap())
    })
    .collect();
    train_steps(&mut leg, 3);
    let want_ns5 = snapshot(&leg);

    let mut cases: Vec<(&str, ModelFile)> = vec![("saved ns5", ModelFile::open(&dpath).unwrap())];
    cases.extend(pre_polar_files);
    for (label, f) in cases {
        let mut b_leg = new_run(OptimKind::MuSgd, 89);
        assert_eq!(f.checkpoint().unwrap().optim.muon_ortho, None, "{label}");
        let res = f.resume(&b_leg.model, b_leg.opt.as_mut(), Some(&mut b_leg.ema), Some(classes(3))).unwrap();
        assert_eq!(b_leg.opt.contract().muon_ortho, None, "{label}: restores Ns5");
        b_leg.sched = res.schedule;
        b_leg.rng = res.rng;
        drop(f);
        train_steps(&mut b_leg, 3);
        assert_eq!(snapshot(&b_leg), want_ns5, "{label}: CPU Ns5 resume must be bit-identical");
    }

    // `Polar5` also writes `"muon_ortho": "polar5"` and resumes bit-identically.
    let mut p5 = new_ortho(23, MuonOrtho::Polar5);
    train_steps(&mut p5, 3);
    let p5path = tmp("resume-musgd-polar5.folo");
    save(&p5, &p5path);
    assert_eq!(optim_json(&p5path)["muon_ortho"], "polar5");
    train_steps(&mut p5, 3);
    let want_p5 = snapshot(&p5);

    let mut b5 = new_run(OptimKind::MuSgd, 90);
    let f = ModelFile::open(&p5path).unwrap();
    let res = f.resume(&b5.model, b5.opt.as_mut(), Some(&mut b5.ema), Some(classes(3))).unwrap();
    assert_eq!(b5.opt.contract().muon_ortho, Some(MuonOrtho::Polar5));
    b5.sched = res.schedule;
    b5.rng = res.rng;
    drop(f);
    train_steps(&mut b5, 3);
    assert_eq!(snapshot(&b5), want_p5, "Polar5 checkpoint resume must be bit-identical");

    // `muon_ortho` is refused on non-MuSGD contracts.
    let mut optim = optim_json(&path);
    optim["type"] = "sgd".into();
    map.insert("folo.optim".into(), optim.to_string());
    assert!(rewrite(&map, &tensors).err().unwrap().to_string().contains("muon_ortho"));

    // `Conv-GeoMuon` (`muon_geo: "geo"` + `muon_depthwise`) round-trips bit-identically on a model with both a
    // tall `1x1` conv (`[8, 4, 1, 1]`, where `muon_scale_geo = sqrt(2) != 1`) and a depthwise conv (`groups = 4`),
    // restores the depthwise mask even when resuming into plain `ParamGroup::new` groups, diverges from `Legacy`
    // and from dense-Geo (no depthwise mask), and rejects invalid `muon_geo`/`muon_depthwise` contracts.
    let net_geo = |rng: &mut Rng| -> Sequential {
        let p3 = Sequential::new()
            .push_named("dw0", Conv2d::new(Conv2dConfig { groups: 4, ..Conv2dConfig::new(4, 4, 3) }, rng))
            .push_named("tall", ConvBnAct::new(4, 8, 1, 1, 1, Activation::Silu, rng))
            .push_named("out", Conv2d::new(Conv2dConfig::new(8, 3, 1), rng));
        Sequential::new()
            .push_named(
                "backbone",
                Sequential::new().push_named("stem", ConvBnAct::new(3, 4, 3, 2, 1, Activation::Silu, rng)),
            )
            .push_named("head", Sequential::new().push_named("cls", Sequential::new().push_named("p3", p3)))
    };
    let build_geo_run = |seed: u64, geo: MuonGeoMode, plain_groups: bool| -> Run {
        let mut init = Rng::seed(seed);
        let model = net_geo(&mut init);
        perturb(&model, &mut init);
        let ema = ModelEma::new(net_geo(&mut Rng::seed(seed + 100)), &model).unwrap();
        let sg = StandardGroups::from_module(&model);
        let last = sg.decay.last().unwrap().0.rsplit_once('.').unwrap().0.to_string() + ".";
        let groups = if plain_groups {
            // Plain groups constructed with `ParamGroup::new` (no `.with_depthwise(...)`): resume must restore
            // the depthwise branch from `muon_depthwise` in the checkpoint contract.
            sg.into_boosted_groups(0.01, 0.9, 5e-4, |n| n.starts_with(&last), 3.0)
                .into_iter()
                .map(|g| {
                    ParamGroup::new(g.name, g.params)
                        .lr(g.lr)
                        .momentum(g.momentum)
                        .weight_decay(g.weight_decay)
                })
                .collect()
        } else {
            sg.into_boosted_groups(0.01, 0.9, 5e-4, |n| n.starts_with(&last), 3.0)
        };
        let opt = Box::new(
            MuSgd::with_ortho_and_geo(groups, MuSgdWeights::default(), MuonOrtho::Polar4, geo).unwrap(),
        );
        let sched = LrSchedule::new(DecayKind::Cosine, 2, 5, 0.01, 0.0, 0.1, 0.8).unwrap();
        let mut c = sched.contract().clone();
        c.warmup_iters = 4;
        let sched = LrSchedule::from_contract(c).unwrap();
        let acc = GradAccumulator::new(&model.parameters());
        let rng = RngState { seed, streams: [("data".to_string(), Rng::seed(seed ^ 0xda7a))].into() };
        Run { model, opt, sched, ema, acc, rng }
    };

    // Mismatched `ParamGroup::depthwise` length must be rejected.
    {
        let mut rng = Rng::seed(25);
        let p = Tensor::randn([4, 1, 3, 3], 0.0, 0.1, &mut rng).requires_grad_();
        let bad_g = ParamGroup::new("decay", vec![("dw.weight".into(), p)]).with_depthwise(vec![true, false]);
        assert!(MuSgd::new(vec![bad_g]).is_err(), "mismatched depthwise mask length must fail");
    }

    let mut geo_run = build_geo_run(24, MuonGeoMode::Geo, false);
    train_steps(&mut geo_run, 3);
    let gpath = tmp("resume-musgd-geo.folo");
    save(&geo_run, &gpath);
    let (gmap, gtensors) =
        with_parts(&geo_run, |p| checkpoint_contents(&meta(FileKind::Checkpoint, 3), p).unwrap());
    let gj = optim_json(&gpath);
    assert_eq!(gj["muon_geo"], "geo");
    assert_eq!(gj["muon_depthwise"], serde_json::json!(["head.cls.p3.dw0.weight"]));
    assert!(optim_json(&path).get("muon_geo").is_none(), "legacy mode omits muon_geo");
    assert!(optim_json(&path).get("muon_depthwise").is_none(), "legacy mode omits muon_depthwise");
    train_steps(&mut geo_run, 3);
    let want_geo = snapshot(&geo_run);

    // Resume into a fresh `Legacy` MuSGD whose groups were built via plain `ParamGroup::new` (no depthwise mask):
    // both `muon_geo` and `muon_depthwise` must be restored from the checkpoint contract.
    let mut bg = build_geo_run(91, MuonGeoMode::Legacy, true);
    let fg = ModelFile::open(&gpath).unwrap();
    let res = fg.resume(&bg.model, bg.opt.as_mut(), Some(&mut bg.ema), Some(classes(3))).unwrap();
    assert_eq!(bg.opt.contract().muon_geo, Some(MuonGeoMode::Geo));
    assert_eq!(bg.opt.contract().muon_depthwise, vec!["head.cls.p3.dw0.weight".to_string()]);
    bg.sched = res.schedule;
    bg.rng = res.rng;
    drop(fg);
    train_steps(&mut bg, 3);
    assert_eq!(snapshot(&bg), want_geo, "Conv-GeoMuon checkpoint resume must be bit-identical");

    // A non-depthwise weight in `muon_depthwise` (e.g. a standard conv or rank-2 linear weight) must be rejected
    // both during checkpoint validation and direct `apply_contract`.
    {
        let mut bad_c = geo_run.opt.contract();
        bad_c.muon_depthwise = vec!["head.cls.p3.tall.conv.weight".into()];
        assert!(
            geo_run.opt.apply_contract(&bad_c).is_err(),
            "apply_contract must reject non-depthwise shape"
        );
        let mut m = gmap.clone();
        let mut oj: Value = serde_json::from_str(&m["folo.optim"]).unwrap();
        oj["muon_depthwise"] = serde_json::json!(["head.cls.p3.tall.conv.weight"]);
        m.insert("folo.optim".into(), oj.to_string());
        assert!(
            rewrite(&m, &gtensors).err().unwrap().to_string().contains("depthwise conv weight"),
            "checkpoint validation must reject non-depthwise shape"
        );
    }
    {
        let mut rng = Rng::seed(26);
        let lin = Tensor::randn([8, 4], 0.0, 0.1, &mut rng).requires_grad_();
        let g = ParamGroup::new("decay", vec![("fc.weight".into(), lin.clone())]);
        let mut opt =
            MuSgd::with_ortho_and_geo(vec![g], MuSgdWeights::default(), MuonOrtho::Polar4, MuonGeoMode::Geo)
                .unwrap();
        let mut bad_c = opt.contract();
        bad_c.muon_depthwise = vec!["fc.weight".into()];
        assert!(opt.apply_contract(&bad_c).is_err(), "apply_contract must reject rank-2 linear weight");
        let bad_dw =
            ParamGroup::new("decay", vec![("fc.weight".into(), lin.clone())]).with_depthwise(vec![true]);
        assert!(
            MuSgd::with_ortho_and_geo(
                vec![bad_dw.clone()],
                MuSgdWeights::default(),
                MuonOrtho::Polar4,
                MuonGeoMode::Geo
            )
            .is_err(),
            "Geo constructor must reject depthwise-marked rank-2 weight"
        );
        let mut leg_opt = MuSgd::with_ortho_and_geo(
            vec![bad_dw],
            MuSgdWeights::default(),
            MuonOrtho::Polar4,
            MuonGeoMode::Legacy,
        )
        .unwrap();
        assert!(
            leg_opt.set_geo_mode(MuonGeoMode::Geo).is_err(),
            "set_geo_mode must reject rank-2 depthwise weight"
        );
        assert_eq!(leg_opt.geo_mode(), MuonGeoMode::Legacy);
    }

    // Verify `Geo` with depthwise actually diverges from both `Legacy` and dense-`Geo` (without depthwise mask).
    let mut legacy_run = build_geo_run(24, MuonGeoMode::Legacy, false);
    train_steps(&mut legacy_run, 6);
    assert_ne!(snapshot(&legacy_run), want_geo, "Geo must diverge from Legacy on tall+depthwise model");
    let mut dense_geo_run = build_geo_run(24, MuonGeoMode::Geo, true);
    train_steps(&mut dense_geo_run, 6);
    assert_ne!(snapshot(&dense_geo_run), want_geo, "depthwise Polar LMO must diverge from dense Geo");

    // Resuming a legacy MuSGD checkpoint must preserve model-derived depthwise masks so switching to `Geo`
    // afterward via `set_geo_mode` still routes depthwise convs through the depthwise Polar LMO branch.
    {
        let mut leg_src = build_geo_run(24, MuonGeoMode::Legacy, false);
        train_steps(&mut leg_src, 3);
        let lpath = tmp("resume-musgd-legacy-dw.folo");
        save(&leg_src, &lpath);

        let mut init = Rng::seed(92);
        let model = net_geo(&mut init);
        let mut ema = ModelEma::new(net_geo(&mut Rng::seed(192)), &model).unwrap();
        let sg = StandardGroups::from_module(&model);
        let last = sg.decay.last().unwrap().0.rsplit_once('.').unwrap().0.to_string() + ".";
        let groups = sg.into_boosted_groups(0.01, 0.9, 5e-4, |n| n.starts_with(&last), 3.0);
        let mut opt = MuSgd::with_ortho_and_geo(
            groups,
            MuSgdWeights::default(),
            MuonOrtho::Polar4,
            MuonGeoMode::Legacy,
        )
        .unwrap();
        let fl = ModelFile::open(&lpath).unwrap();
        fl.resume(&model, &mut opt, Some(&mut ema), Some(classes(3))).unwrap();
        opt.set_geo_mode(MuonGeoMode::Geo).unwrap();
        assert_eq!(
            opt.contract().muon_depthwise,
            vec!["head.cls.p3.dw0.weight".to_string()],
            "legacy checkpoint resume must preserve model-derived depthwise masks"
        );
    }

    // Contract validation errors for `muon_geo` and `muon_depthwise`.
    let mut gopt = optim_json(&gpath);
    gopt["type"] = "sgd".into();
    gopt.as_object_mut().unwrap().remove("muon_ortho");
    gopt.as_object_mut().unwrap().remove("muon_depthwise");
    let mut m1 = gmap.clone();
    m1.insert("folo.optim".into(), gopt.to_string());
    assert!(rewrite(&m1, &gtensors).err().unwrap().to_string().contains("muon_geo"));

    let mut gopt2 = optim_json(&gpath);
    gopt2.as_object_mut().unwrap().remove("muon_geo");
    let mut m2 = gmap.clone();
    m2.insert("folo.optim".into(), gopt2.to_string());
    assert!(rewrite(&m2, &gtensors).err().unwrap().to_string().contains("muon_depthwise"));

    let mut gopt3 = optim_json(&gpath);
    gopt3["muon_depthwise"] = serde_json::json!(["head.cls.p3.dw0.weight", "head.cls.p3.dw0.weight"]);
    let mut m3 = gmap.clone();
    m3.insert("folo.optim".into(), gopt3.to_string());
    assert!(rewrite(&m3, &gtensors).err().unwrap().to_string().contains("duplicate"));

    let mut gopt4 = optim_json(&gpath);
    gopt4["muon_depthwise"] = serde_json::json!(["head.cls.p3.dw0.bias"]);
    let mut m4 = gmap;
    m4.insert("folo.optim".into(), gopt4.to_string());
    assert!(rewrite(&m4, &gtensors).err().unwrap().to_string().contains("muon_depthwise"));
}
