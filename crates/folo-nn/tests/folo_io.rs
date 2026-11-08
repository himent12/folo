//! `.folo` files: compatibility both ways with safetensors, loading rules, export, and checkpoint resume.

use std::collections::BTreeMap;
use std::path::PathBuf;

use folo_core::rng::Rng;
use folo_core::{Tensor, f16, no_grad};
use folo_nn::io::*;
use folo_nn::optim::*;
use folo_nn::*;
use serde_json::Value;

// ----- helpers -----

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("folo-nn-tests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A tiny model with detector-style names: `backbone.stem.*`, `head.cls.p3.pw0.*`, `head.cls.p3.out.*`.
fn net(nc: usize, rng: &mut Rng) -> Sequential {
    let p3 = Sequential::new()
        .push_named("pw0", ConvBnAct::new(4, 4, 1, 1, 1, Activation::Silu, rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, nc, 1), rng));
    Sequential::new()
        .push_named(
            "backbone",
            Sequential::new().push_named("stem", ConvBnAct::new(3, 4, 3, 2, 1, Activation::Silu, rng)),
        )
        .push_named("head", Sequential::new().push_named("cls", Sequential::new().push_named("p3", p3)))
}

/// The same model with fused Conv blocks (`conv.weight`, `conv.bias`).
fn net_fused(nc: usize, rng: &mut Rng) -> Sequential {
    let p3 = Sequential::new()
        .push_named("pw0", ConvBnAct::new_fused(4, 4, 1, 1, 1, Activation::Silu, rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, nc, 1), rng));
    Sequential::new()
        .push_named(
            "backbone",
            Sequential::new().push_named("stem", ConvBnAct::new_fused(3, 4, 3, 2, 1, Activation::Silu, rng)),
        )
        .push_named("head", Sequential::new().push_named("cls", Sequential::new().push_named("p3", p3)))
}

fn classes(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("class{i}")).collect()
}

fn meta(kind: FileKind, nc: usize) -> Metadata {
    Metadata::new(kind, ModelConfig::new("n", nc).unwrap(), classes(nc), 64).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
}

fn state_bits(m: &dyn Module) -> Vec<(String, Vec<u32>)> {
    m.named_tensors().into_iter().map(|t| (t.name, bits(&t.tensor))).collect()
}

/// Give BatchNorms non-trivial statistics and affine parameters.
fn perturb(m: &dyn Module, rng: &mut Rng) {
    m.train();
    for _ in 0..2 {
        m_forward(m, &Tensor::randn([2, 3, 8, 8], 0.3, 1.5, rng));
    }
    let _g = no_grad();
    for (n, t) in m.named_parameters() {
        if n.ends_with("bn.weight") {
            t.copy_from(&Tensor::rand_uniform(t.shape().clone(), 0.5, 1.5, rng)).unwrap();
        } else if n.ends_with("bias") {
            t.copy_from(&Tensor::rand_uniform(t.shape().clone(), -0.5, 0.5, rng)).unwrap();
        }
    }
}

fn m_forward(m: &dyn Module, x: &Tensor) -> Tensor {
    (m as &dyn std::any::Any)
        .downcast_ref::<Sequential>()
        .expect("test models are Sequential")
        .forward(x)
        .unwrap()
}

// ----- an independent safetensors reader (no folo-nn code) -----

struct Parsed {
    header_len: usize,
    raw_header: Vec<u8>,
    header: serde_json::Map<String, Value>,
    data: Vec<u8>,
}

fn parse_independently(bytes: &[u8]) -> Parsed {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let raw_header = bytes[8..8 + n].to_vec();
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&raw_header).unwrap();
    Parsed { header_len: n, raw_header, header, data: bytes[8 + n..].to_vec() }
}

fn le_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[test]
fn writer_output_parses_as_plain_safetensors() {
    let mut rng = Rng::seed(0);
    let model = net(3, &mut rng);
    perturb(&model, &mut rng);
    let path = tmp("weights.folo");
    save_weights(&path, &meta(FileKind::Weights, 3), &model).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let p = parse_independently(&bytes);

    assert_eq!((8 + p.header_len) % 8, 0, "data section is 8-aligned");
    assert_eq!(p.raw_header[0], b'{');
    let trimmed = p.raw_header.iter().rposition(|&b| b != b' ').unwrap();
    assert_eq!(p.raw_header[trimmed], b'}');

    let md = p.header["__metadata__"].as_object().unwrap();
    assert_eq!(md["folo.format"], "1");
    assert_eq!(md["folo.kind"], "weights");
    assert_eq!(md["folo.arch"], "folo-n");
    assert_eq!(md["folo.imgsz"], "64");
    let cfg: Value = serde_json::from_str(md["folo.config"].as_str().unwrap()).unwrap();
    assert_eq!((cfg["num_classes"].as_u64(), cfg["scale"].as_str()), (Some(3), Some("n")));
    let cls: Vec<String> = serde_json::from_str(md["folo.classes"].as_str().unwrap()).unwrap();
    assert_eq!(cls, classes(3));
    assert!(md.values().all(Value::is_string));

    // Layout: by descending dtype size, then name; contiguous from 0; data equals the model's tensors.
    let entries: Vec<(&String, &Value)> = p.header.iter().filter(|(k, _)| *k != "__metadata__").collect();
    let mut names: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    let header_order = names.clone();
    names.sort();
    assert_eq!(header_order, names, "all F32, so name order");
    let mut cursor = 0;
    let by_name: BTreeMap<String, Tensor> =
        model.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    assert_eq!(entries.len(), by_name.len());
    for (name, e) in entries {
        assert_eq!(e["dtype"], "F32");
        let off: Vec<usize> =
            e["data_offsets"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(off[0], cursor);
        cursor = off[1];
        let shape: Vec<usize> =
            e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let t = &by_name[name];
        assert_eq!(shape, t.dims());
        assert_eq!(le_f32(&p.data[off[0]..off[1]]), t.to_vec::<f32>().unwrap(), "{name}");
    }
    assert_eq!(cursor, p.data.len());

    // And our reader loads it back exactly.
    let back = net(3, &mut Rng::seed(99));
    let f = ModelFile::open(&path).unwrap();
    let r = f.load(&back, &LoadOptions::strict(Purpose::Training).with_classes(classes(3))).unwrap();
    assert_eq!(r.loaded.len(), by_name.len());
    assert!(r.skipped.is_empty() && r.missing.is_empty());
    assert_eq!(state_bits(&back), state_bits(&model));
}

/// A safetensors blob built by hand: keys out of order, an unaligned F16 layout, a reserved tensor, and
/// either FOLO metadata or none.
fn hand_built(with_meta: bool) -> Vec<u8> {
    let w = [f16::from_f32(0.5), f16::from_f32(-1.25)];
    let b = [f16::from_f32(2.0)];
    let mut data = vec![7u8, 8, 9];
    data.extend(b.iter().flat_map(|v| v.to_bits().to_le_bytes()));
    data.extend(w.iter().flat_map(|v| v.to_bits().to_le_bytes()));
    let mut md = meta(FileKind::Weights, 1).to_map();
    md.insert("folo.dtype".into(), "F16".into());
    md.insert("folo.future_key".into(), "kept".into());
    md.insert("other.tool".into(), "ignored".into());
    let md_json = serde_json::to_string(&md).unwrap();
    let header = format!(
        r#"{{"c.weight":{{"dtype":"F16","shape":[1,2,1,1],"data_offsets":[5,9]}},{}"__note":{{"dtype":"U8","shape":[3],"data_offsets":[0,3]}},"c.bias":{{"dtype":"F16","shape":[1],"data_offsets":[3,5]}}}}"#,
        if with_meta { format!(r#""__metadata__":{md_json},"#) } else { String::new() }
    );
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[test]
fn reads_a_hand_built_safetensors_blob() {
    let tiny =
        || Sequential::new().push_named("c", Conv2d::new(Conv2dConfig::new(2, 1, 1), &mut Rng::seed(1)));
    let f = ModelFile::from_bytes(hand_built(true)).unwrap();
    assert_eq!(f.metadata().dtype, FloatDType::F16);
    assert_eq!(f.metadata().extra.get("folo.future_key").map(String::as_str), Some("kept"));
    assert_eq!(f.warnings().len(), 1, "{:?}", f.warnings());
    let m = tiny();
    let r = f.load_for_inference(&m).unwrap();
    assert_eq!(r.source, Source::Model);
    assert_eq!(r.loaded, ["c.weight", "c.bias"]);
    let c = &m.named_parameters();
    assert_eq!(c[0].1.to_vec::<f32>().unwrap(), [0.5, -1.25]);
    assert_eq!(c[1].1.to_vec::<f32>().unwrap(), [2.0]);

    // Without folo metadata: a valid container, but loading needs the import path.
    let plain = hand_built(false);
    assert!(safetensors::Container::parse(&plain).is_ok());
    let e = ModelFile::from_bytes(plain.clone()).err().unwrap().to_string();
    assert!(e.contains("import"), "{e}");
    let mut md = meta(FileKind::Weights, 1);
    md.dtype = FloatDType::F16;
    let f = ModelFile::import(plain, md).unwrap();
    let m = tiny();
    f.load_for_inference(&m).unwrap();
    assert_eq!(m.named_parameters()[0].1.to_vec::<f32>().unwrap(), [0.5, -1.25]);
}

#[test]
fn strict_and_partial_loading() {
    let mut rng = Rng::seed(2);
    let src = net(3, &mut rng);
    perturb(&src, &mut rng);
    let path = tmp("strict.folo");
    save_weights(&path, &meta(FileKind::Weights, 3), &src).unwrap();
    let f = ModelFile::open(&path).unwrap();

    // Strict into a model with a different class count: every offending name is listed.
    let other = net(5, &mut Rng::seed(3));
    let e = f.load(&other, &LoadOptions::strict(Purpose::Training)).unwrap_err().to_string();
    assert!(e.contains("head.cls.p3.out.weight") && e.contains("head.cls.p3.out.bias"), "{e}");
    // Strict with a class list that differs only in order: refused, suggesting partial loading.
    let mut swapped = classes(3);
    swapped.swap(0, 1);
    let same = net(3, &mut Rng::seed(4));
    let e = f
        .load(&same, &LoadOptions::strict(Purpose::Training).with_classes(swapped.clone()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("partial"), "{e}");
    // Nothing was written by the failed strict loads.
    assert_ne!(state_bits(&same), state_bits(&src));

    // Partial, same shapes but different class order: the class outputs are skipped anyway.
    let before =
        bits(&same.named_parameters().iter().find(|(n, _)| n == "head.cls.p3.out.weight").unwrap().1);
    let r = f.load(&same, &LoadOptions::partial(Purpose::Training).with_classes(swapped)).unwrap();
    let skipped: Vec<_> = r.skipped.iter().map(|s| (s.name.as_str(), &s.reason)).collect();
    assert_eq!(
        skipped,
        [
            ("head.cls.p3.out.bias", &SkipReason::ClassMismatch),
            ("head.cls.p3.out.weight", &SkipReason::ClassMismatch)
        ]
    );
    assert_eq!(r.missing, ["head.cls.p3.out.weight", "head.cls.p3.out.bias"]);
    let after = bits(&same.named_parameters().iter().find(|(n, _)| n == "head.cls.p3.out.weight").unwrap().1);
    assert_eq!(before, after);
    assert_eq!(r.loaded.len(), src.named_tensors().len() - 2);

    // Partial into a different class count: shape mismatches are skipped and reported.
    let r = f.load(&other, &LoadOptions::partial(Purpose::Training)).unwrap();
    assert!(r.skipped.iter().all(|s| matches!(s.reason, SkipReason::ShapeMismatch { .. })));
    assert_eq!(r.skipped.len(), 2);

    // Unexpected and missing names in a model with an extra layer.
    let bigger = Sequential::new()
        .push_named(
            "backbone",
            Sequential::new().push_named("stem", ConvBnAct::new(3, 4, 3, 2, 1, Activation::Silu, &mut rng)),
        )
        .push_named("extra", Conv2d::new(Conv2dConfig::new(4, 4, 1), &mut rng));
    let e = f.load(&bigger, &LoadOptions::strict(Purpose::Training)).unwrap_err().to_string();
    assert!(
        e.contains("missing `extra.weight`") && e.contains("unexpected `head.cls.p3.pw0.conv.weight`"),
        "{e}"
    );
    let r = f.load(&bigger, &LoadOptions::partial(Purpose::Training)).unwrap();
    assert_eq!(r.missing, ["extra.weight", "extra.bias"]);
    assert!(r.skipped.iter().any(|s| s.name == "head.cls.p3.out.bias" && s.reason == SkipReason::NotInModel));
}

#[test]
fn export_fuses_and_matches_unfused_eval() {
    let mut rng = Rng::seed(5);
    let model = net(3, &mut rng);
    perturb(&model, &mut rng);
    model.eval();
    let x = Tensor::randn([2, 3, 8, 8], 0.0, 1.0, &mut rng);
    let want = m_forward(&model, &x).to_vec::<f32>().unwrap();

    let ck = tmp("export-src.folo");
    save_weights(&ck, &meta(FileKind::Weights, 3), &model).unwrap();
    let src = ModelFile::open(&ck).unwrap();
    for (dtype, tol) in [(FloatDType::F32, 1e-5f32), (FloatDType::F16, 2e-2)] {
        let out = tmp(&format!("export-{}.folo", dtype.as_str()));
        src.export(&out, dtype).unwrap();
        let f = ModelFile::open(&out).unwrap();
        assert!(f.metadata().fused && f.metadata().dtype == dtype && f.metadata().kind == FileKind::Weights);
        assert!(f.names(Namespace::Model).iter().all(|n| !n.contains(".bn.")));
        assert!(f.info("backbone.stem.conv.bias").is_some());
        let fused = net_fused(3, &mut Rng::seed(6));
        fused.eval();
        let e = f.load(&fused, &LoadOptions::strict(Purpose::Training)).unwrap_err().to_string();
        assert!(e.contains("inference"), "{e}");
        f.load_for_inference(&fused).unwrap();
        let got = m_forward(&fused, &x).to_vec::<f32>().unwrap();
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst <= tol, "{dtype:?}: {worst}");
    }
    // The in-memory path gives the same fused tensors.
    let fused_t =
        fuse_bn_tensors(model.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect(), BN_EPS)
            .unwrap();
    let f = ModelFile::open(tmp("export-F32.folo")).unwrap();
    for (n, t) in fused_t {
        assert_eq!(bits(&t), bits(&f.tensor(&n).unwrap()), "{n}");
    }
}

// ----- checkpoints -----

struct Run {
    model: Sequential,
    opt: Box<dyn Optimizer>,
    sched: LrSchedule,
    ema: ModelEma<Sequential>,
    acc: GradAccumulator,
    rng: RngState,
}

fn new_run(kind: OptimKind, seed: u64) -> Run {
    let mut rng = Rng::seed(seed);
    let model = net(3, &mut rng);
    let ema = ModelEma::new(net(3, &mut Rng::seed(seed + 100)), &model).unwrap();
    let groups = StandardGroups::from_module(&model).into_groups(0.01, 0.9, 5e-4);
    let opt: Box<dyn Optimizer> = match kind {
        OptimKind::Sgd => Box::new(Sgd::new(groups).unwrap()),
        OptimKind::AdamW => Box::new(AdamW::new(groups).unwrap()),
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

fn train_steps(r: &mut Run, n: usize) {
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

fn save(r: &Run, path: &std::path::Path) {
    let train = serde_json::json!({ "epochs": 2, "batch": 2 });
    let parts = CheckpointParts {
        model: &r.model,
        ema: Some(&r.ema),
        optimizer: r.opt.as_ref(),
        schedule: &r.sched,
        rng: &r.rng,
        epoch: 1,
        best: BestState { epoch: 0, fitness: Some(0.25) },
        train_config: &train,
    };
    save_checkpoint(path, &meta(FileKind::Checkpoint, 3), &parts).unwrap();
}

fn snapshot(r: &Run) -> Vec<(String, Vec<u32>)> {
    let mut s = state_bits(&r.model);
    s.extend(state_bits(r.ema.module()).into_iter().map(|(n, b)| (format!("ema.{n}"), b)));
    s.extend(r.opt.state_tensors().unwrap().into_iter().map(|(n, t)| (n, bits(&t))));
    s
}

#[test]
fn checkpoint_resume_is_bit_identical() {
    for kind in [OptimKind::Sgd, OptimKind::AdamW] {
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

fn rewrite(map: &BTreeMap<String, String>, tensors: &[(String, Tensor)]) -> folo_nn::Result<ModelFile> {
    let mut buf = Vec::new();
    safetensors::write(&mut buf, Some(map), tensors).unwrap();
    ModelFile::from_bytes(buf)
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

#[test]
fn root_block_export_fuses_and_loads_into_new_fused() {
    let mut rng = Rng::seed(8);
    let blk = ConvBnAct::new(3, 4, 3, 1, 1, Activation::Silu, &mut rng);
    for _ in 0..2 {
        blk.forward(&Tensor::randn([2, 3, 6, 6], 0.5, 2.0, &mut rng)).unwrap();
    }
    blk.eval();
    let x = Tensor::randn([1, 3, 6, 6], 0.0, 1.0, &mut rng);
    let want = blk.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let path = tmp("root-block.folo");
    export_module(&path, &meta(FileKind::Weights, 3), &blk, FloatDType::F32).unwrap();
    let f = ModelFile::open(&path).unwrap();
    assert_eq!(f.names(Namespace::Model), ["conv.bias", "conv.weight"]);
    let fused = ConvBnAct::new_fused(3, 4, 3, 1, 1, Activation::Silu, &mut Rng::seed(9));
    f.load_for_inference(&fused).unwrap();
    let got = fused.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst <= 1e-5, "{worst}");
    // A stray root BatchNorm tensor without its weight is an error, not a silent pass-through.
    let stray = vec![
        ("conv.weight".to_string(), Tensor::ones([1, 1, 1, 1])),
        ("bn.running_mean".to_string(), Tensor::zeros([1])),
    ];
    assert!(fuse_bn_tensors(stray, BN_EPS).is_err());
}
