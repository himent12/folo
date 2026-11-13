//! docs/MODEL.md §10.1 / §10.2: block and whole-model tests for the FOLO detector family.

use std::path::PathBuf;

use folo_core::rng::Rng;
use folo_core::testing::grad_check;
use folo_core::{DType, Tensor, no_grad};
use folo_nn::io::{FloatDType, LoadOptions, ModelFile, Purpose, SkipReason};
use folo_nn::{Forward, Layer, Module, init};
use folo_test_support::TestFile;
use folo_vision::model::blocks::{Attention, Bottleneck, C2Psa, C3k, C3k2, Init, PsaBlock, Sppf};
use folo_vision::model::{FoloModel, LAYER_NAMES, ModelConfig};
use folo_vision::nms::NmsConfig;

const SCALES: [&str; 5] = ["n", "s", "m", "l", "x"];

/// MODEL.md §4, the `params` column of each scale's table (nc = 80), by layer.
const LAYER_PARAMS: [[usize; 24]; 5] = [
    [
        464, 4672, 6640, 36992, 26080, 147712, 87040, 295424, 346112, 164608, 249728, 0, 0, 111296, 0, 0,
        32096, 36992, 0, 86720, 147712, 0, 378880, 464896,
    ],
    [
        928, 18560, 26080, 147712, 103360, 590336, 346112, 1180672, 1380352, 656896, 990976, 0, 0, 443776, 0,
        0, 127680, 147712, 0, 345472, 590336, 0, 1511424, 850352,
    ],
    [
        1856, 73984, 111872, 590336, 444928, 2360320, 1380352, 2360320, 1380352, 656896, 990976, 0, 0,
        1642496, 0, 0, 542720, 590336, 0, 1511424, 2360320, 0, 1642496, 1472688,
    ],
    [
        1856, 73984, 173824, 590336, 691712, 2360320, 2234368, 2360320, 2234368, 656896, 1455616, 0, 0,
        2496512, 0, 0, 756736, 590336, 0, 2365440, 2360320, 0, 2496512, 1472688,
    ],
    [
        2784, 166272, 389760, 1327872, 1553664, 5309952, 5022720, 5309952, 5022720, 1476864, 3264768, 0, 0,
        5612544, 0, 0, 1700352, 1327872, 0, 5317632, 5309952, 0, 5612544, 3237936,
    ],
];

/// MODEL.md §4 totals for nc = 80, 4, 1.
const TOTALS: [[usize; 3]; 5] = [
    [2_624_064, 2_590_604, 2_590_019],
    [9_458_736, 9_429_324, 9_428_163],
    [20_114_672, 20_056_076, 20_053_763],
    [25_372_144, 25_313_548, 25_311_235],
    [56_966_160, 56_878_380, 56_874_915],
];

/// MODEL.md §4: named tensors (parameters, buffers).
const NAMED: [(usize, usize); 5] = [(255, 162), (255, 162), (330, 212), (513, 334), (513, 334)];

/// MODEL.md §4, output channels of layers 0–22.
const LAYER_CHANNELS: [[usize; 23]; 5] = [
    [
        16, 32, 64, 64, 128, 128, 128, 256, 256, 256, 256, 256, 384, 128, 128, 256, 64, 64, 192, 128, 128,
        384, 256,
    ],
    [
        32, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 768, 256, 256, 512, 128, 128, 384, 256,
        256, 768, 512,
    ],
    [
        64, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 1024, 512, 512, 1024, 256, 256, 768, 512,
        512, 1024, 512,
    ],
    [
        64, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 1024, 512, 512, 1024, 256, 256, 768, 512,
        512, 1024, 512,
    ],
    [
        96, 192, 384, 384, 768, 768, 768, 768, 768, 768, 768, 768, 1536, 768, 768, 1536, 384, 384, 1152, 768,
        768, 1536, 768,
    ],
];

/// The spatial stride of layers 0–22 (`H_out = H / stride`).
const LAYER_STRIDE: [usize; 23] =
    [2, 4, 4, 8, 8, 16, 16, 32, 32, 32, 32, 16, 16, 16, 8, 8, 8, 16, 16, 16, 32, 32, 32];

fn model(scale: &str, nc: usize) -> FoloModel {
    FoloModel::new(ModelConfig::new(scale, nc).unwrap()).unwrap()
}

fn layer_of(name: &str) -> usize {
    LAYER_NAMES
        .iter()
        .position(|p| !p.is_empty() && name.strip_prefix(p).is_some_and(|r| r.starts_with('.')))
        .unwrap_or_else(|| panic!("`{name}` belongs to no layer"))
}

fn per_layer_params(m: &FoloModel) -> [usize; 24] {
    let mut out = [0; 24];
    for (n, t) in m.named_parameters() {
        out[layer_of(&n)] += t.numel();
    }
    out
}

fn scratch(tag: &str) -> TestFile {
    TestFile::new(&format!("vision-model-{tag}"), "m.folo")
}

fn input(b: usize, h: usize, w: usize, seed: u64) -> Tensor {
    Tensor::rand_uniform([b, 3, h, w], 0.0, 1.0, &mut Rng::seed(seed))
}

/// Push every BatchNorm away from its identity initialisation, so fusion and loading are tested on
/// non-trivial statistics.
fn randomize_bn(m: &FoloModel, seed: u64) {
    let mut rng = Rng::seed(seed);
    for t in m.named_tensors() {
        let (lo, hi) = if t.name.ends_with("bn.weight") || t.name.ends_with("running_var") {
            (0.8, 1.2)
        } else if t.name.ends_with("bn.bias") || t.name.ends_with("running_mean") {
            (-0.1, 0.1)
        } else {
            continue;
        };
        init::copy_(&t.tensor, &Tensor::rand_uniform(t.tensor.shape().clone(), lo, hi, &mut rng)).unwrap();
    }
}

/// `|a − b| ≤ rel · max(|a|, |b|, 1)`: relative, with a floor of one pixel (boxes) or one unit (scores), so a
/// coordinate that happens to land near 0 is not held to a tighter absolute bound than its neighbours.
fn assert_close(a: &[f32], b: &[f32], rel: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let mut worst = (0.0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && y.is_finite(), "{what}: non-finite value at {i} ({x} vs {y})");
        let err = (x - y).abs() / x.abs().max(y.abs()).max(1.0);
        if err > worst.0 {
            worst = (err, i);
        }
    }
    assert!(
        worst.0 <= rel,
        "{what}: relative error {} at {} ({} vs {})",
        worst.0,
        worst.1,
        a[worst.1],
        b[worst.1]
    );
}

fn bits(m: &FoloModel) -> Vec<(String, Vec<usize>, Vec<u32>)> {
    m.named_tensors()
        .into_iter()
        .map(|t| {
            let v = t.tensor.to_vec::<f32>().unwrap();
            (t.name, t.tensor.dims().to_vec(), v.iter().map(|v| v.to_bits()).collect())
        })
        .collect()
}

// ---- §10.1 blocks ----

type Build = Box<dyn Fn(&mut Init) -> Box<dyn Layer>>;

#[test]
fn block_parameter_counts_and_shapes() {
    let mut rng = Rng::seed(0);
    let mut i = Init::new(&mut rng);

    let conv = i.conv(3, 16, 3, 2);
    assert_eq!(conv.num_parameters(), 464);
    conv.eval();
    assert_eq!(conv.forward(&Tensor::zeros([1, 3, 640, 640])).unwrap().dims(), [1, 16, 320, 320]);

    let cases: Vec<(Build, usize, usize, usize)> = vec![
        (Box::new(|i| Box::new(Bottleneck::new(16, 16, 0.5, i))), 16, 16, 2_352),
        (Box::new(|i| Box::new(Bottleneck::new(32, 32, 1.0, i))), 32, 32, 18_560),
        (Box::new(|i| Box::new(C3k::new(64, 64, 2, 0.5, i))), 64, 64, 45_568),
        (Box::new(|i| Box::new(C3k2::new(32, 64, 1, false, 0.25, i))), 32, 64, 6_640),
        (Box::new(|i| Box::new(C3k2::new(64, 64, 2, false, 0.5, i))), 64, 64, 31_168),
        (Box::new(|i| Box::new(C3k2::new(128, 128, 1, true, 0.5, i))), 128, 128, 87_040),
        (Box::new(|i| Box::new(Sppf::new(256, 256, 5, i))), 256, 256, 164_608),
        (Box::new(|i| Box::new(Attention::new(128, 2, i).unwrap())), 128, 128, 51_328),
        (Box::new(|i| Box::new(C2Psa::new(256, 1, i).unwrap())), 256, 256, 249_728),
        (Box::new(|i| Box::new(C2Psa::new(256, 2, i).unwrap())), 256, 256, 367_360),
    ];
    for (k, (build, c1, c2, params)) in cases.into_iter().enumerate() {
        let b = build(&mut i);
        assert_eq!(b.num_parameters(), params, "case {k}");
        b.eval();
        assert_eq!(b.forward(&Tensor::zeros([1, c1, 12, 12])).unwrap().dims(), [1, c2, 12, 12], "case {k}");
    }
    assert!(Attention::new(10, 3, &mut i).is_err());
}

#[test]
fn block_names() {
    let mut rng = Rng::seed(0);
    let mut i = Init::new(&mut rng);
    let names = |m: &dyn Module| m.named_parameters().into_iter().map(|(n, _)| n).collect::<Vec<_>>();
    let c3k2 = names(&C3k2::new(16, 16, 1, true, 0.5, &mut i));
    assert_eq!(c3k2.first().map(String::as_str), Some("cv1.conv.weight"));
    assert!(c3k2.contains(&"block0.block1.cv2.bn.bias".to_string()));
    assert!(c3k2.contains(&"block0.cv3.conv.weight".to_string()));
    let psa = names(&C2Psa::new(16, 1, &mut i).unwrap());
    let want = [
        "cv1.conv.weight",
        "cv1.bn.weight",
        "cv1.bn.bias",
        "cv2.conv.weight",
        "cv2.bn.weight",
        "cv2.bn.bias",
        "block0.attn.qkv.conv.weight",
    ];
    assert_eq!(&psa[..7], want);
    assert_eq!(psa.last().map(String::as_str), Some("block0.ffn2.bn.bias"));
}

#[test]
fn block_gradients() {
    // MODEL.md §10.1: finite differences at a tiny size, train mode (BatchNorm on batch statistics).
    let mut rng = Rng::seed(3);
    let mut i = Init::new(&mut rng);
    let blocks: Vec<(&str, Box<dyn Layer>, usize)> = vec![
        ("bottleneck", Box::new(Bottleneck::new(8, 8, 0.5, &mut i)), 8),
        ("c3k", Box::new(C3k::new(8, 8, 2, 0.5, &mut i)), 8),
        ("c3k2", Box::new(C3k2::new(8, 8, 1, false, 0.5, &mut i)), 8),
        ("c3k2/c3k", Box::new(C3k2::new(8, 8, 1, true, 0.5, &mut i)), 8),
        ("sppf", Box::new(Sppf::new(8, 8, 5, &mut i)), 8),
        ("attention", Box::new(Attention::new(8, 1, &mut i).unwrap()), 8),
        ("psa", Box::new(PsaBlock::new(8, &mut i).unwrap()), 8),
        ("c2psa", Box::new(C2Psa::new(16, 1, &mut i).unwrap()), 16),
    ];
    for (name, b, c) in blocks {
        let x = Tensor::randn([2, c, 6, 6], 0.0, 1.0, &mut Rng::seed(7));
        grad_check(|xs| b.forward(&xs[0]).map_err(|e| folo_core::Error::Autograd(e.to_string())), &[x])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

// ---- §10.2 whole model ----

#[test]
fn parameter_counts_all_scales() {
    for (k, scale) in SCALES.iter().enumerate() {
        for (j, nc) in [80, 4, 1].into_iter().enumerate() {
            let m = model(scale, nc);
            assert_eq!(m.num_parameters(), TOTALS[k][j], "folo-{scale} nc={nc}");
            let layers = per_layer_params(&m);
            for l in 0..23 {
                assert_eq!(
                    layers[l], LAYER_PARAMS[k][l],
                    "folo-{scale} nc={nc} layer {l} ({})",
                    LAYER_NAMES[l]
                );
            }
            if nc == 80 {
                assert_eq!(layers[23], LAYER_PARAMS[k][23], "folo-{scale} head");
            }
            let (p, b) = (m.named_parameters().len(), m.named_buffers().len());
            assert_eq!((p, b), NAMED[k], "folo-{scale} named tensors");
            let names = m.named_tensors();
            assert_eq!(names.first().unwrap().name, "backbone.stem.conv.weight");
            assert_eq!(names.last().unwrap().name, "head.cls.p5.out.bias");
        }
    }
}

#[test]
fn golden_names_folo_n() {
    let m = model("n", 80);
    let got: String =
        m.named_tensors().iter().map(|t| format!("{} {:?}\n", t.name, t.tensor.dims())).collect();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/folo-n-names.txt");
    if std::env::var_os("FOLO_BLESS").is_some() {
        std::fs::write(&path, &got).unwrap();
    }
    let want = std::fs::read_to_string(&path).expect("golden file (regenerate with FOLO_BLESS=1)");
    assert!(got == want.replace("\r\n", "\n"), "folo-n tensor names/shapes changed; review and re-bless");
    // MODEL.md §5 examples.
    let shape =
        |n: &str| m.named_tensors().into_iter().find(|t| t.name == n).map(|t| t.tensor.dims().to_vec());
    assert_eq!(shape("backbone.stem.conv.weight"), Some(vec![16, 3, 3, 3]));
    assert_eq!(shape("backbone.stage2.block0.cv1.conv.weight"), Some(vec![8, 16, 3, 3]));
    assert_eq!(shape("backbone.stage4.block0.block1.cv2.bn.running_var"), Some(vec![32]));
    assert_eq!(shape("backbone.psa.block0.attn.pe.conv.weight"), Some(vec![128, 1, 3, 3]));
    assert_eq!(shape("head.cls.p5.out.bias"), Some(vec![80]));
}

fn check_layer_shapes(k: usize, m: &FoloModel, b: usize, h: usize, w: usize) {
    let mut seen = 0;
    let out = m
        .forward_traced(&Tensor::zeros([b, 3, h, w]), &mut |l, y| {
            let s = LAYER_STRIDE[l];
            assert_eq!(y.dims(), [b, LAYER_CHANNELS[k][l], h / s, w / s], "folo-{} layer {l}", SCALES[k]);
            seen += 1;
        })
        .unwrap();
    assert_eq!(seen, 23);
    let a = (h / 8) * (w / 8) + (h / 16) * (w / 16) + (h / 32) * (w / 32);
    assert_eq!(out.box_raw.dims(), [b, 64, a]);
    assert_eq!(out.cls_raw.dims(), [b, m.num_classes(), a]);
    assert_eq!(out.num_anchors(), a);
}

#[test]
fn forward_shapes_folo_n() {
    let m = model("n", 80);
    m.eval();
    let _g = no_grad();
    check_layer_shapes(0, &m, 1, 640, 640);
    let out = m.forward(&Tensor::zeros([1, 3, 320, 320])).unwrap();
    assert_eq!(out.num_anchors(), 2100);
    // Rectangular 384×640 (H × W): 48·80 + 24·40 + 12·20 anchors.
    let out = m.forward(&Tensor::zeros([1, 3, 384, 640])).unwrap();
    assert_eq!(out.num_anchors(), 5040);
    let levels: Vec<_> = out.anchors.levels.iter().map(|l| (l.height, l.width)).collect();
    assert_eq!(levels, [(48, 80), (24, 40), (12, 20)]);
    for bad in [[1, 3, 100, 64], [1, 1, 64, 64], [0, 3, 64, 64]] {
        assert!(m.forward(&Tensor::zeros(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn forward_shapes_all_scales_small() {
    // Channels of every layer for every scale, at a small input (spatial sizes follow the strides).
    for (k, scale) in SCALES.iter().enumerate() {
        let m = model(scale, 80);
        m.eval();
        let _g = no_grad();
        check_layer_shapes(k, &m, 1, 64, 96);
    }
}

#[test]
fn head_priors() {
    // MODEL.md §10.2: with the head output weights zeroed, cls_raw is the prior at every anchor and every box
    // is 15 strides wide and tall.
    for nc in [80, 4] {
        let m = model("n", nc);
        for l in &m.head.levels {
            init::constant_(&l.box_branch.out.weight, 0.0).unwrap();
            init::constant_(&l.cls_branch.out.weight, 0.0).unwrap();
        }
        m.eval();
        let out = m.forward(&input(1, 64, 64, 1)).unwrap();
        let cls = out.cls_raw.to_vec::<f32>().unwrap();
        let a = out.num_anchors();
        for c in 0..nc {
            for i in 0..a {
                let s = out.anchors.strides[i] as u32;
                assert_eq!(cls[c * a + i], init::cls_bias_prior(nc, s), "class {c} anchor {i}");
            }
        }
        let (boxes, _) = &out.decode_images().unwrap()[0];
        for (i, b) in boxes.iter().enumerate() {
            let s = out.anchors.strides[i];
            assert!(
                (b.width() - 15.0 * s).abs() < 1e-3 && (b.height() - 15.0 * s).abs() < 1e-3,
                "anchor {i}"
            );
        }
    }
    assert!((init::cls_bias_prior(80, 8) - -11.5366).abs() < 1e-4);
}

#[test]
fn forward_backward_smoke_folo_n_320() {
    let m = model("n", 80);
    let out = m.forward(&input(2, 320, 320, 2)).unwrap();
    assert_eq!(out.num_anchors(), 2100);
    let box_term = out.box_raw.square().unwrap().mean_all().unwrap();
    let loss = box_term.add(&out.cls_raw.sigmoid().unwrap().mean_all().unwrap()).unwrap();
    let grads = loss.backward().unwrap();
    for (name, p) in m.named_parameters() {
        let g = grads.get(&p).unwrap_or_else(|| panic!("no gradient for `{name}`"));
        assert_eq!(g.dims(), p.dims(), "{name}");
        assert!(g.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()), "{name}");
    }
    // Training updated the running statistics.
    let rm = m.backbone.stem.bn.as_ref().unwrap().running_mean.to_vec::<f32>().unwrap();
    assert!(rm.iter().any(|&v| v != 0.0));
}

#[test]
fn fused_matches_unfused() {
    // MODEL.md §9.1: fused inference matches unfused eval-mode inference to 1e-4 relative on the decoded output.
    let mut m = model("n", 80);
    randomize_bn(&m, 11);
    m.eval();
    let x = input(2, 128, 160, 4);
    let _g = no_grad();
    let want = m.forward(&x).unwrap().decode().unwrap();
    m.fuse().unwrap();
    assert!(m.is_fused());
    assert!(m.named_tensors().iter().all(|t| !t.name.contains(".bn.")));
    assert_eq!(m.named_buffers().len(), 0);
    let got = m.forward(&x).unwrap().decode().unwrap();
    assert_eq!(got.dims(), [2, 20 * 16 + 10 * 8 + 5 * 4, 84]);
    assert_close(&got.to_vec::<f32>().unwrap(), &want.to_vec::<f32>().unwrap(), 1e-4, "fused decode");
}

#[test]
fn save_load_round_trip() {
    let mut m = FoloModel::with_rng(ModelConfig::new("n", 3).unwrap(), &mut Rng::seed(5)).unwrap();
    m.set_classes(vec!["cat".into(), "dog".into(), "bird".into()]).unwrap();
    m.set_imgsz(320);
    randomize_bn(&m, 6);
    let path = scratch("roundtrip");
    m.save(&path).unwrap();

    let file = ModelFile::open(&path).unwrap();
    let meta = file.metadata();
    assert_eq!((meta.arch.as_str(), meta.imgsz, meta.fused), ("folo-n", 320, false));
    assert_eq!(&meta.config, m.config());
    assert_eq!(meta.classes, m.classes());
    let (back, report) = FoloModel::from_file(&file, Purpose::Training).unwrap();
    assert!(report.skipped.is_empty() && report.missing.is_empty());
    assert_eq!(report.loaded.len(), 255 + 162);
    assert!(back.is_training());
    assert!(bits(&back) == bits(&m), "round trip is not bit-identical");
    assert_eq!((back.classes(), back.imgsz()), (m.classes(), 320));
    drop(file);

    // Same outputs, bit for bit.
    m.eval();
    let opened = FoloModel::open(&path).unwrap();
    assert!(!opened.is_training());
    let x = input(1, 64, 64, 8);
    let _g = no_grad();
    let (a, b) = (m.forward(&x).unwrap(), opened.forward(&x).unwrap());
    assert_eq!(a.box_raw.to_vec::<f32>().unwrap(), b.box_raw.to_vec::<f32>().unwrap());
    assert_eq!(a.cls_raw.to_vec::<f32>().unwrap(), b.cls_raw.to_vec::<f32>().unwrap());

    // Fused files: a fused model round-trips into a fused model; training from it is refused.
    m.fuse().unwrap();
    let fused_path = scratch("fused");
    m.save(&fused_path).unwrap();
    let fused = FoloModel::open(&fused_path).unwrap();
    assert!(fused.is_fused());
    assert!(bits(&fused) == bits(&m), "fused round trip is not bit-identical");
    let (a, b) = (m.forward(&x).unwrap(), fused.forward(&x).unwrap());
    assert_eq!(a.box_raw.to_vec::<f32>().unwrap(), b.box_raw.to_vec::<f32>().unwrap());
    let err = FoloModel::from_file(&ModelFile::open(&fused_path).unwrap(), Purpose::Training).err().unwrap();
    assert!(err.to_string().contains("fused"), "{err}");
    // A fused model refuses training loads of an unfused file before writing anything.
    let before = bits(&m);
    let unfused = ModelFile::open(&path).unwrap();
    for res in [m.transfer_from(&unfused), m.load(&unfused, LoadOptions::strict(Purpose::Training))] {
        assert!(res.err().unwrap().to_string().contains("fused"));
    }
    assert!(bits(&m) == before, "a refused load changed the model");
    drop(unfused);

    // F16 export of the unfused model loads for inference (fused, computing in F32), close to the F32 model.
    let half = scratch("f16");
    opened.export(&half, FloatDType::F16).unwrap();
    let f16 = FoloModel::open(&half).unwrap();
    assert!(f16.is_fused());
    assert_eq!(f16.backbone.stem.conv.weight.dtype(), DType::F32);
    let want = opened.forward(&x).unwrap().decode().unwrap().to_vec::<f32>().unwrap();
    let got = f16.forward(&x).unwrap().decode().unwrap().to_vec::<f32>().unwrap();
    assert_close(&got, &want, 2e-2, "f16 export");
    for p in [path, fused_path, half] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn transfer_to_new_class_count() {
    let src = model("n", 80);
    let path = scratch("transfer");
    src.save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();

    let dst = FoloModel::with_rng(ModelConfig::new("n", 4).unwrap(), &mut Rng::seed(9)).unwrap();
    assert!(dst.load(&file, LoadOptions::strict(Purpose::Training)).is_err());
    let report = dst.transfer_from(&file).unwrap();
    let class_skips: Vec<&str> = report
        .skipped
        .iter()
        .filter(|s| s.reason == SkipReason::ClassMismatch)
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(class_skips.len(), 6, "{class_skips:?}");
    assert!(class_skips.iter().all(|n| n.starts_with("head.cls.") && n.contains(".out.")));
    // c_cls changes 80 → 64, so pw0 / dw1 / pw1 change shape too and are skipped by the shape check.
    assert!(
        report.skipped.iter().any(|s| s.name == "head.cls.p3.pw1.conv.weight"
            && matches!(s.reason, SkipReason::ShapeMismatch { .. }))
    );
    assert!(report.loaded.contains(&"head.cls.p3.dw0.conv.weight".to_string()));
    assert!(report.loaded.contains(&"head.box.p5.out.bias".to_string()));
    assert_eq!(report.loaded.len() + report.missing.len(), dst.named_tensors().len());
    // The new model keeps its own class priors; transferable weights come from the file.
    let bias = dst.head.levels[0].cls_branch.out.bias.as_ref().unwrap().to_vec::<f32>().unwrap();
    assert!(bias.iter().all(|&b| b == init::cls_bias_prior(4, 8)));
    let a = src.backbone.stage4.cv1.conv.weight.to_vec::<f32>().unwrap();
    assert_eq!(a, dst.backbone.stage4.cv1.conv.weight.to_vec::<f32>().unwrap());

    // Same count, different names: the output layers are still skipped (never loaded for the wrong classes).
    let mut renamed = model("n", 80);
    renamed.set_classes((0..80).map(|i| format!("c{i}")).collect()).unwrap();
    let r = renamed.transfer_from(&file).unwrap();
    assert_eq!(r.skipped.len(), 6);
    assert!(r.skipped.iter().all(|s| s.reason == SkipReason::ClassMismatch));
    drop(file);
    let _ = std::fs::remove_file(path);
}

#[test]
fn predict_raw_decodes_and_suppresses() {
    let m = model("n", 2);
    let x = input(2, 64, 64, 3);
    assert!(m.predict_raw(&x, &NmsConfig::default()).is_err(), "training mode must be refused");
    m.eval();
    let cfg = NmsConfig { conf: 0.0, max_det: 50, ..NmsConfig::default() };
    let p = m.predict_raw(&x, &cfg).unwrap();
    assert_eq!(p.images.len(), 2);
    let decoded = p.output.decode().unwrap().to_vec::<f32>().unwrap();
    let a = p.output.num_anchors();
    for (b, img) in p.images.iter().enumerate() {
        assert_eq!((img.boxes.len(), img.scores.len()), (a, 2 * a));
        let row = &decoded[(b * a) * 6..(b * a + 1) * 6];
        assert_eq!([img.boxes[0].x1, img.boxes[0].y1, img.boxes[0].x2, img.boxes[0].y2], row[..4]);
        assert_eq!(img.scores[..2], row[4..]);
        assert!(!img.detections.is_empty() && img.detections.len() <= 50);
        assert!(img.detections.windows(2).all(|w| w[0].score >= w[1].score));
    }
    // Default conf: an untrained model's prior scores (about 5 objects per image) are far below 0.25.
    let p = m.predict_raw(&x, &NmsConfig::default()).unwrap();
    assert!(p.images.iter().all(|i| i.detections.is_empty()));
}

#[test]
fn config_serde_and_seeds() {
    let m = model("s", 7);
    let json = serde_json::to_string(m.config()).unwrap();
    assert!(json.contains("\"scale\":\"s\"") && json.contains("\"num_classes\":7"));
    assert_eq!(&serde_json::from_str::<ModelConfig>(&json).unwrap(), m.config());
    assert_eq!(m.arch(), "folo-s");
    // One seed, one model; another seed, another model.
    let a = FoloModel::with_rng(ModelConfig::new("n", 4).unwrap(), &mut Rng::seed(1)).unwrap();
    let b = FoloModel::with_rng(ModelConfig::new("n", 4).unwrap(), &mut Rng::seed(1)).unwrap();
    let c = FoloModel::with_rng(ModelConfig::new("n", 4).unwrap(), &mut Rng::seed(2)).unwrap();
    assert!(bits(&a) == bits(&b));
    assert!(bits(&a) != bits(&c));
    let mut d = model("n", 4);
    assert!(d.set_classes(vec!["a".into()]).is_err());
}

/// `HeadOutput::detections` on the CPU (anchors prefiltered by their largest logit, only candidates decoded,
/// in blocks of 1024 anchors on the rayon pool) is bit for bit the full decode + `non_max_suppression`, on a
/// small grid (one block) and on a 640×384 one (5040 anchors, five blocks); with box voting and DFL fusion too
/// (the candidates' side precisions read in the blocks, the full decode's from every anchor).
#[test]
fn cpu_detections_match_full_decode() {
    use folo_vision::fusion::BoxFusion;
    use folo_vision::nms::{NmsConfig, non_max_suppression, non_max_suppression_with};
    let rng = &mut folo_core::rng::Rng::seed(11);
    let m = model("n", 7);
    let cases = [
        (2usize, 7usize, -2.5f32, NmsConfig::default()),
        (1, 7, 0.0, NmsConfig { conf: 0.6, iou: 0.4, max_det: 9, ..NmsConfig::default() }),
        (3, 7, -4.0, NmsConfig { conf: 0.05, classes: Some(vec![0, 3, 6]), ..NmsConfig::default() }),
    ];
    for ((b, nc, mean, cfg), [h, w]) in
        cases.iter().cloned().map(|c| (c, [96, 64])).chain(cases.iter().cloned().map(|c| (c, [640, 384])))
    {
        let anchors = m.anchors(h, w);
        let a = anchors.len();
        let out = folo_vision::model::HeadOutput {
            box_raw: Tensor::randn([b, 64, a], 0.0, 2.0, rng),
            cls_raw: Tensor::randn([b, nc, a], mean, 2.0, rng),
            anchors: anchors.clone(),
            reg_max: 16,
            num_classes: nc,
            mode: folo_vision::model::HeadMode::O2m,
            o2o: None,
        };
        let want: Vec<_> = out
            .decode_images()
            .unwrap()
            .iter()
            .map(|(bx, sc)| non_max_suppression(bx, sc, nc, &cfg))
            .collect();
        let got = out.detections(&cfg).unwrap();
        assert!(want.iter().any(|d| !d.is_empty()), "the case should keep some detections");
        assert_eq!(got, want, "{cfg:?}");
        // The same outputs as channel slices of one `[B, 64 + nc, A]` tensor (one storage, strided views for
        // B > 1): same detections.
        let joint = Tensor::cat(&[&out.box_raw, &out.cls_raw], 1).unwrap();
        let shared = folo_vision::model::HeadOutput {
            box_raw: joint.narrow(1, 0, 64).unwrap(),
            cls_raw: joint.narrow(1, 64, nc).unwrap(),
            ..out.clone()
        };
        assert!(shared.box_raw.same_storage(&shared.cls_raw));
        assert_eq!(shared.detections(&cfg).unwrap(), want, "{cfg:?}, one storage");
        let prec = out.side_precisions().unwrap();
        for fusion in [BoxFusion::Vote, BoxFusion::Dfl] {
            let cfg = NmsConfig { fusion, fusion_iou: 0.5, ..cfg.clone() };
            let fused: Vec<_> = out
                .decode_images()
                .unwrap()
                .iter()
                .zip(&prec)
                .map(|((bx, sc), p)| non_max_suppression_with(bx, sc, Some(p), nc, &cfg))
                .collect();
            assert_eq!(out.detections(&cfg).unwrap(), fused, "{cfg:?}");
            assert_eq!(
                fused.iter().map(Vec::len).collect::<Vec<_>>(),
                want.iter().map(Vec::len).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn letterbox_helpers_reject_mismatched_geometry() {
    use folo_data::{Image, Letterbox, LetterboxOptions};
    use folo_vision::predict::letterbox_on_host;
    let opts = LetterboxOptions::default();
    let imgs = [Image::new(40, 30, [1, 2, 3]), Image::new(20, 50, [4, 5, 6])];
    let lb = |i: &Image, s: u32| Letterbox::compute(i.width, i.height, s, s, &opts);
    let ok = [lb(&imgs[0], 64), lb(&imgs[1], 64)];
    assert_eq!(letterbox_on_host(&imgs, &ok).unwrap().dims(), [2, 3, 64, 64]);
    assert!(letterbox_on_host(&imgs, &ok[..1]).is_err(), "fewer geometries than images");
    assert!(letterbox_on_host(&[], &[]).is_err(), "empty batch");
    assert!(letterbox_on_host(&imgs, &[ok[1], ok[0]]).is_err(), "geometry of another image size");
    assert!(letterbox_on_host(&imgs, &[ok[0], lb(&imgs[1], 32)]).is_err(), "different canvases");
}

/// On the CPU a batch runs as chunks of images side by side; every image's predictions are exactly what
/// predicting it alone on the square canvas gives.
#[test]
fn cpu_predict_batch_matches_single_images() {
    use folo_vision::predict::{PredictConfig, Predictor};
    let model = FoloModel::new(ModelConfig::new("n", 4).unwrap()).unwrap();
    let cfg = PredictConfig { imgsz: 96, conf: 0.001, rect: false, ..PredictConfig::default() };
    let p = Predictor::new(model, cfg).unwrap();
    let mut rng = Rng::seed(21);
    let imgs: Vec<folo_data::Image> = (0..5)
        .map(|i| {
            let (w, h) = (80 + 8 * i, 96 - 4 * i);
            let data = (0..w * h * 3).map(|_| (rng.uniform(0.0, 256.0) as u32).min(255) as u8).collect();
            folo_data::Image::from_raw(w, h, data).unwrap()
        })
        .collect();
    let batch = p.predict_batch(&imgs).unwrap();
    assert_eq!(batch.len(), imgs.len());
    let mut any = false;
    for (img, b) in imgs.iter().zip(&batch) {
        let one = p.predict_batch(std::slice::from_ref(img)).unwrap().pop().unwrap();
        assert_eq!(b.detections, one.detections);
        assert_eq!((b.width, b.height), (one.width, one.height));
        any |= !one.detections.is_empty();
    }
    assert!(any, "no detections to compare");
}
