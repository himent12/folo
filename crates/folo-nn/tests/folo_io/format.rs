//! Safetensors container compatibility, strict/partial loading rules, and Conv+BN fused export.

use std::collections::BTreeMap;

use folo_core::rng::Rng;
use folo_core::{Tensor, f16};
use folo_nn::io::*;
use folo_nn::*;
use folo_test_support::TestDir;
use serde_json::Value;

use crate::{
    bits, classes, le_f32, m_forward, meta, net, net_fused, parse_independently, perturb, state_bits, tmp,
};

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

    let dir = TestDir::new("nn-tests-export");
    let ck = dir.join("export-src.folo");
    save_weights(&ck, &meta(FileKind::Weights, 3), &model).unwrap();
    let src = ModelFile::open(&ck).unwrap();
    for (dtype, tol) in [(FloatDType::F32, 1e-5f32), (FloatDType::F16, 2e-2)] {
        let out = dir.join(format!("export-{}.folo", dtype.as_str()));
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
    let f = ModelFile::open(dir.join("export-F32.folo")).unwrap();
    for (n, t) in fused_t {
        assert_eq!(bits(&t), bits(&f.tensor(&n).unwrap()), "{n}");
    }
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
