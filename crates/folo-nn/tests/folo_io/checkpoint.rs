//! Checkpoint save, layout, contracts, bit-exact resume, and mismatch validation.

use std::collections::BTreeMap;

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_nn::io::*;
use folo_nn::optim::*;
use folo_nn::*;
use serde_json::Value;

use crate::{bits, classes, meta, net, parse_independently, rewrite, state_bits, tmp};

pub(crate) struct Run {
    pub(crate) model: Sequential,
    pub(crate) opt: Box<dyn Optimizer>,
    pub(crate) sched: LrSchedule,
    pub(crate) ema: ModelEma<Sequential>,
    pub(crate) acc: GradAccumulator,
    pub(crate) rng: RngState,
}

pub(crate) fn new_run(kind: OptimKind, seed: u64) -> Run {
    new_run_with(kind, seed, MuSgdWeights::default())
}

/// [`new_run`] with MuSGD step weights (ignored by the other optimizers).
pub(crate) fn new_run_with(kind: OptimKind, seed: u64, weights: MuSgdWeights) -> Run {
    let mut rng = Rng::seed(seed);
    let model = net(3, &mut rng);
    let ema = ModelEma::new(net(3, &mut Rng::seed(seed + 100)), &model).unwrap();
    let sg = StandardGroups::from_module(&model);
    let opt: Box<dyn Optimizer> = match kind {
        OptimKind::Sgd => Box::new(Sgd::new(sg.into_groups(0.01, 0.9, 5e-4)).unwrap()),
        OptimKind::AdamW => Box::new(AdamW::new(sg.into_groups(0.01, 0.9, 5e-4)).unwrap()),
        // The last layer stands in for the head's class branch (`.cls` groups at 3x the LR).
        OptimKind::MuSgd => {
            let last = sg.decay.last().unwrap().0.rsplit_once('.').unwrap().0.to_string() + ".";
            let groups = sg.into_boosted_groups(0.01, 0.9, 5e-4, |n| n.starts_with(&last), 3.0);
            Box::new(MuSgd::with_weights(groups, weights).unwrap())
        }
    };
    let sched = LrSchedule::new(DecayKind::Cosine, 2, 5, 0.01, 0.0, 0.1, 0.8).unwrap();
    // Warmup over a few iterations so resume crosses the warmup interpolation.
    let mut c = sched.contract().clone();
    c.warmup_iters = 4;
    let sched = LrSchedule::from_contract(c).unwrap();
    let acc = GradAccumulator::new(&model.parameters());
    let rng = RngState { seed, streams: [("data".to_string(), Rng::seed(seed ^ 0xda7a))].into() };
    Run { model, opt, sched, ema, acc, rng }
}

pub(crate) fn train_steps(r: &mut Run, n: usize) {
    for _ in 0..n {
        let data = r.rng.streams.get_mut("data").unwrap();
        for _ in 0..2 {
            let x = Tensor::randn([2, 3, 8, 8], 0.0, 1.0, data);
            let loss = r.model.forward(&x).unwrap().square().unwrap().mean_all().unwrap();
            r.acc.add(&loss.backward().unwrap()).unwrap();
        }
        r.sched.apply(r.opt.as_mut());
        r.acc.clip_grad_norm(10.0).unwrap();
        r.opt.step(&r.acc).unwrap();
        r.ema.update(&r.model).unwrap();
        r.sched.advance();
        r.acc.zero();
    }
}

pub(crate) fn with_parts<T>(r: &Run, f: impl FnOnce(&CheckpointParts<'_>) -> T) -> T {
    let train = serde_json::json!({ "epochs": 2, "batch": 2 });
    f(&CheckpointParts {
        model: &r.model,
        ema: Some(&r.ema),
        optimizer: r.opt.as_ref(),
        schedule: &r.sched,
        rng: &r.rng,
        epoch: 1,
        best: BestState { epoch: 0, fitness: Some(0.25) },
        train_config: &train,
    })
}

pub(crate) fn save(r: &Run, path: &std::path::Path) {
    with_parts(r, |p| save_checkpoint(path, &meta(FileKind::Checkpoint, 3), p).unwrap());
}

pub(crate) fn snapshot(r: &Run) -> Vec<(String, Vec<u32>)> {
    let mut s = state_bits(&r.model);
    s.extend(state_bits(r.ema.module()).into_iter().map(|(n, b)| (format!("ema.{n}"), b)));
    s.extend(r.opt.state_tensors().unwrap().into_iter().map(|(n, t)| (n, bits(&t))));
    s
}

#[test]
fn checkpoint_resume_is_bit_identical() {
    for kind in [OptimKind::Sgd, OptimKind::AdamW, OptimKind::MuSgd] {
        let mut a = new_run(kind, 10);
        train_steps(&mut a, 3);
        let path = tmp(&format!("resume-{}.folo", kind.as_str()));
        save(&a, &path);
        train_steps(&mut a, 3);
        let want = snapshot(&a);

        let mut b = new_run(kind, 77); // different init: everything must come from the file
        let f = ModelFile::open(&path).unwrap();
        let ck = f.checkpoint().unwrap();
        assert_eq!((ck.epoch, ck.best.epoch, ck.best.fitness), (1, 0, Some(0.25)));
        assert_eq!((ck.optim.step, ck.sched.step, ck.ema_updates), (3, 3, Some(3)));
        let res = f.resume(&b.model, b.opt.as_mut(), Some(&mut b.ema), Some(classes(3))).unwrap();
        assert!(res.report.missing.is_empty());
        assert_eq!(res.train_config["batch"], 2);
        b.sched = res.schedule;
        b.rng = res.rng;
        drop(f);
        train_steps(&mut b, 3);
        assert_eq!(b.opt.step_count(), 6);
        assert_eq!(b.ema.updates(), 6);
        let got = snapshot(&b);
        assert_eq!(got.len(), want.len());
        for ((n1, b1), (n2, b2)) in got.iter().zip(&want) {
            assert_eq!(n1, n2);
            assert!(b1 == b2, "{kind:?}: `{n1}` differs after resume");
        }
    }
}

#[test]
fn a_scaled_ema_resumes_with_the_very_contract_it_saved() {
    // 44 optimizer steps at a quarter: window 11, decay 0.9090909090909091, a decimal that serde_json's fast float
    // path reads one ulp off.
    let (decay, tau) = folo_nn::ema_for_run(44, 0.25);
    assert_eq!((decay, tau), (1.0 - 1.0 / 11.0, 2.2));
    let mut a = new_run(OptimKind::Sgd, 10);
    a.ema = ModelEma::with_params(net(3, &mut Rng::seed(110)), &a.model, decay, tau).unwrap();
    train_steps(&mut a, 3);
    let path = tmp("resume-scaled-ema.folo");
    save(&a, &path);
    let live = a.ema.contract();

    let mut b = new_run(OptimKind::Sgd, 77); // the default EMA: the contract must come from the file
    let f = ModelFile::open(&path).unwrap();
    f.resume(&b.model, b.opt.as_mut(), Some(&mut b.ema), Some(classes(3))).unwrap();
    let got = b.ema.contract();
    assert_eq!((got.decay.to_bits(), got.tau.to_bits()), (live.decay.to_bits(), live.tau.to_bits()));

    // Every window the rule gives a run of up to 40,000 steps reads back to the bit.
    for steps in 1..=40_000 {
        let (decay, tau) = folo_nn::ema_for_run(steps, 0.25);
        let c = folo_nn::EmaContract { decay, tau, power: None };
        let back: folo_nn::EmaContract = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(
            (back.decay.to_bits(), back.tau.to_bits()),
            (decay.to_bits(), tau.to_bits()),
            "{steps} steps"
        );
    }
    // PC-PEMA contract round-trips to the bit and legacy JSON without `power` still parses cleanly.
    let p_c = folo_nn::EmaContract {
        decay: folo_nn::EMA_DECAY,
        tau: folo_nn::EMA_TAU,
        power: Some(folo_nn::EmaPowerConfig::new(12.0)),
    };
    let p_back: folo_nn::EmaContract = serde_json::from_str(&serde_json::to_string(&p_c).unwrap()).unwrap();
    assert_eq!(p_back, p_c);
    // A contract that overflows to infinity is refused when it is read, not when the EMA is restored.
    assert!(serde_json::from_str::<folo_nn::EmaContract>(r#"{"decay":0.99,"tau":1e400}"#).is_err());
}

#[test]
fn checkpoint_layout_and_contracts() {
    let mut r = new_run(OptimKind::Sgd, 1);
    train_steps(&mut r, 2);
    let path = tmp("layout.folo");
    save(&r, &path);
    let p = parse_independently(&std::fs::read(&path).unwrap());
    let entries: Vec<(&String, &Value)> = p.header.iter().filter(|(k, _)| *k != "__metadata__").collect();
    // I64 counters (8 bytes) come first, by name; then every F32 tensor by name.
    let names: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(&names[..3], ["ema.__updates", "optim.__step", "sched.__step"]);
    assert!(
        entries[..3].iter().all(|(_, e)| e["dtype"] == "I64" && e["shape"].as_array().unwrap().is_empty())
    );
    let rest = &names[3..];
    let mut sorted = rest.to_vec();
    sorted.sort();
    assert_eq!(rest, sorted);
    let md = p.header["__metadata__"].as_object().unwrap();
    assert_eq!(md["folo.kind"], "checkpoint");
    assert_eq!(md["folo.epoch"], "1");
    let optim: Value = serde_json::from_str(md["folo.optim"].as_str().unwrap()).unwrap();
    assert_eq!((optim["type"].as_str(), optim["step"].as_u64()), (Some("sgd"), Some(2)));
    assert_eq!(
        optim["groups"].as_array().unwrap().iter().map(|g| g["name"].as_str().unwrap()).collect::<Vec<_>>(),
        ["decay", "bn", "bias"]
    );
    assert_eq!(optim["groups"][0]["nesterov"], true);
    let sched: Value = serde_json::from_str(md["folo.sched"].as_str().unwrap()).unwrap();
    assert_eq!(
        (sched["type"].as_str(), sched["unit"].as_str(), sched["step"].as_u64()),
        (Some("cosine"), Some("iteration"), Some(2))
    );
    let ema: Value = serde_json::from_str(md["folo.ema"].as_str().unwrap()).unwrap();
    assert_eq!((ema["decay"].as_f64(), ema["tau"].as_f64()), (Some(0.9999), Some(2000.0)));
    let rng: Value = serde_json::from_str(md["folo.rng"].as_str().unwrap()).unwrap();
    assert_eq!(rng["algo"], "xoshiro256pp");
    assert_eq!(rng["streams"]["data"].as_str().unwrap().len(), 64);
    let best: Value = serde_json::from_str(md["folo.best"].as_str().unwrap()).unwrap();
    assert_eq!(best, serde_json::json!({ "epoch": 0, "fitness": 0.25 }));
    // Complete coverage: one ema. and one momentum buffer per tensor.
    let model_n = r.model.named_tensors().len();
    assert_eq!(names.iter().filter(|n| n.starts_with("ema.") && !n.starts_with("ema.__")).count(), model_n);
    assert_eq!(
        names.iter().filter(|n| n.ends_with(".momentum_buffer")).count(),
        r.model.named_parameters().len()
    );

    // Inference reads the EMA weights; training reads the live ones.
    let f = ModelFile::open(&path).unwrap();
    let m = net(3, &mut Rng::seed(50));
    assert_eq!(f.load_for_inference(&m).unwrap().source, Source::Ema);
    assert_eq!(state_bits(&m), state_bits(r.ema.module()));
    assert_eq!(f.load(&m, &LoadOptions::strict(Purpose::Training)).unwrap().source, Source::Model);
    assert_eq!(state_bits(&m), state_bits(&r.model));
}

#[test]
fn invalid_checkpoints_and_resume_mismatches() {
    let mut r = new_run(OptimKind::Sgd, 2);
    train_steps(&mut r, 1);
    let train = serde_json::json!({});
    let parts = CheckpointParts {
        model: &r.model,
        ema: Some(&r.ema),
        optimizer: r.opt.as_ref(),
        schedule: &r.sched,
        rng: &r.rng,
        epoch: 1,
        best: BestState::default(),
        train_config: &train,
    };
    let (map, tensors) = checkpoint_contents(&meta(FileKind::Checkpoint, 3), &parts).unwrap();
    assert!(rewrite(&map, &tensors).is_ok());

    let err = |map: &BTreeMap<String, String>, t: &[(String, Tensor)]| {
        rewrite(map, t).err().expect("must be rejected").to_string()
    };
    // A partial ema. set.
    let partial: Vec<_> =
        tensors.iter().filter(|(n, _)| n != "ema.backbone.stem.bn.running_var").cloned().collect();
    assert!(err(&map, &partial).contains("cover the model"));
    // Step counters that disagree.
    let mut bad = tensors.clone();
    let i = bad.iter().position(|(n, _)| n == "sched.__step").unwrap();
    bad[i].1 = Tensor::from_vec(vec![5i64], folo_core::Shape::scalar()).unwrap();
    assert!(err(&map, &bad).contains("disagree"));
    // Missing state on a step > 0 checkpoint.
    let nostate: Vec<_> =
        tensors.iter().filter(|(n, _)| !n.ends_with("out.weight.momentum_buffer")).cloned().collect();
    let f = rewrite(&map, &nostate).unwrap();
    let mut b = new_run(OptimKind::Sgd, 3);
    let e = f.resume(&b.model, b.opt.as_mut(), Some(&mut b.ema), None).unwrap_err().to_string();
    assert!(e.contains("head.cls.p3.out.weight.momentum_buffer") && e.contains("missing"), "{e}");
    // A weights file may not carry optimizer state.
    let mut wmap = map.clone();
    wmap.insert("folo.kind".into(), "weights".into());
    assert!(err(&wmap, &tensors).contains("only the model namespace"));
    // Unsupported format version.
    let mut v2 = map.clone();
    v2.insert("folo.format".into(), "2".into());
    assert!(err(&v2, &tensors).contains("not supported"));
    // Invalid PC-PEMA contract in `folo.ema` is rejected on open/validate.
    for bad_ema in [
        r#"{"decay":0.9999,"tau":2000.0,"power":{"gamma_0":-1.0,"rho":0.25,"lf_min":0.01,"t_0":0.0}}"#,
        r#"{"decay":0.9999,"tau":2000.0,"power":{"gamma_0":6.94,"rho":0.25,"lf_min":0.0,"t_0":0.0}}"#,
        r#"{"decay":0.9999,"tau":2000.0,"power":{"gamma_0":6.94,"rho":0.25,"lf_min":0.01,"t_0":-5.0}}"#,
    ] {
        let mut emap = map.clone();
        emap.insert("folo.ema".into(), bad_ema.into());
        assert!(err(&emap, &tensors).contains("folo.ema"));
    }

    let f = rewrite(&map, &tensors).unwrap();
    // Wrong optimizer type.
    let mut c = new_run(OptimKind::AdamW, 4);
    let e = f.resume(&c.model, c.opt.as_mut(), Some(&mut c.ema), None).unwrap_err().to_string();
    assert!(e.contains("sgd") && e.contains("adamw"), "{e}");
    // EMA in the file but not in the run.
    let mut d = new_run(OptimKind::Sgd, 5);
    assert!(f.resume(&d.model, d.opt.as_mut(), None, None).is_err());
    // A different parameter partition.
    let mut e2 = new_run(OptimKind::Sgd, 6);
    let mut groups: Vec<ParamGroup> = e2.opt.groups().to_vec();
    let moved = groups[2].params.pop().unwrap();
    groups[0].params.push(moved);
    e2.opt = Box::new(Sgd::new(groups).unwrap());
    let e = f.resume(&e2.model, e2.opt.as_mut(), Some(&mut e2.ema), None).unwrap_err().to_string();
    assert!(e.contains("parameter set"), "{e}");
}
