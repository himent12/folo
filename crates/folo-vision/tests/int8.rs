//! int8 inference of the backbone and neck (docs/MODEL.md §9.6): the int8 plan against the fake-quant reference
//! (the same grids in `f32` arithmetic), bit-identical across thread counts, and wired into the forward.

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_vision::model::quant::{self, Calibration, FakeQuant, Int8Plan, Method};
use folo_vision::model::{FoloModel, ModelConfig};

/// A fused folo-n whose BatchNorm statistics come from one training-mode forward (momentum 0.999), so every layer
/// sees activations of about unit scale, as a trained model does (a fresh model's shrink towards zero).
fn model(seed: u64) -> FoloModel {
    let mut c = ModelConfig::new("n", 4).unwrap();
    c.bn_momentum = 0.999;
    let mut m = FoloModel::with_rng(c, &mut Rng::seed(seed)).unwrap();
    {
        let _g = no_grad();
        m.forward(&input(2, 64, 64, seed + 100)).unwrap();
    }
    m.fuse().unwrap();
    m
}

fn input(b: usize, h: usize, w: usize, seed: u64) -> Tensor {
    Tensor::rand_uniform([b, 3, h, w], 0.0, 1.0, &mut Rng::seed(seed))
}

fn calibrated(m: &FoloModel, x: &Tensor) -> Calibration {
    quant::calibrate(m, || std::iter::once(Ok(x.clone())), Method::Percentile { p: 1e-4 }).unwrap()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()
}

#[test]
fn int8_features_match_the_fake_quant_reference() {
    let m = model(1);
    let x = input(2, 64, 96, 2);
    let cal = calibrated(&m, &x);
    let _g = no_grad();
    let want = FakeQuant::new(&cal).features(&m, &x).unwrap();
    let plan = Int8Plan::new(cal.clone());
    let got = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
    // The int8 sums are exact where the reference rounds in f32, so a code may differ by one step at a rounding
    // boundary, and that difference may grow through later layers; most codes agree exactly.
    for (i, name) in ["neck.out3.cv2", "neck.out4.cv2", "neck.out5.cv2"].iter().enumerate() {
        let s = cal.param(name).unwrap().scale;
        let (a, b) = (want[i].to_vec::<f32>().unwrap(), got[i].to_vec::<f32>().unwrap());
        assert_eq!(a.len(), b.len());
        let steps: Vec<f32> = a.iter().zip(&b).map(|(p, q)| ((p - q) / s).abs().round()).collect();
        let exact = steps.iter().filter(|&&d| d == 0.0).count() as f64 / steps.len() as f64;
        let worst = steps.iter().copied().fold(0f32, f32::max);
        assert!(exact > 0.99 && worst <= 2.0, "{name}: {:.4} exact, worst {worst} steps", exact);
    }
}

#[test]
fn int8_forward_is_the_same_at_every_thread_count() {
    let mut m = model(3);
    let x = input(1, 96, 64, 4);
    let cal = calibrated(&m, &x);
    m.set_int8(Some(cal)).unwrap();
    let _g = no_grad();
    let run = |t| pool(t).install(|| m.forward(&x).unwrap());
    let (a, b) = (run(1), run(7));
    for (p, q) in [(&a.box_raw, &b.box_raw), (&a.cls_raw, &b.cls_raw)] {
        let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
        assert!(p.iter().zip(&q).all(|(u, v)| u.to_bits() == v.to_bits()));
    }
    // And it is the int8 plan, not the f32 forward.
    m.set_int8(None).unwrap();
    let f = m.forward(&x).unwrap().box_raw.to_vec::<f32>().unwrap();
    let q = a.box_raw.to_vec::<f32>().unwrap();
    let diff = f.iter().zip(&q).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max);
    assert!(diff > 1e-2, "int8 and f32 forwards differ by at most {diff}");
}

#[test]
fn calibration_round_trips_and_names_its_gaps() {
    let m = model(5);
    let x = input(1, 64, 64, 6);
    let mut cal = calibrated(&m, &x);
    let json = serde_json::to_string(&cal).unwrap();
    assert_eq!(serde_json::from_str::<Calibration>(&json).unwrap(), cal);
    // A missing or unusable range is refused when the plan is set, naming the boundary.
    let mut m = m;
    let mut reversed = cal.clone();
    reversed.ranges.insert("backbone.down2".into(), [1.0, -1.0]);
    let e = m.set_int8(Some(reversed)).unwrap_err().to_string();
    assert!(e.contains("backbone.down2"), "{e}");
    // Finite ends whose width is not (an infinite step) are refused too.
    let mut wide = cal.clone();
    wide.ranges.insert("backbone.stage2.cv1".into(), [-f32::MAX, f32::MAX]);
    let e = m.set_int8(Some(wide)).unwrap_err().to_string();
    assert!(e.contains("backbone.stage2.cv1"), "{e}");
    let mut flat = cal.clone();
    flat.ranges.insert("backbone.stage2.cv1".into(), [0.0, 0.0]);
    assert!(flat.param("backbone.stage2.cv1").unwrap().inv().is_finite());
    cal.ranges.remove("neck.out4.cv1");
    let e = m.set_int8(Some(cal)).unwrap_err().to_string();
    assert!(e.contains("neck.out4.cv1"), "{e}");
    assert!(m.int8().is_none());
    // Only a fused f32 model takes a plan.
    let unfused = FoloModel::with_rng(ModelConfig::new("n", 4).unwrap(), &mut Rng::seed(5)).unwrap();
    let mut unfused = unfused;
    assert!(unfused.set_int8(Some(calibrated(&m, &x))).is_err());
}

#[test]
fn calibration_rides_in_the_metadata_and_only_fits_its_weights() {
    let m = model(7);
    let x = input(1, 64, 64, 8);
    let cal = calibrated(&m, &x);
    assert_eq!(cal.weights.as_deref(), Some(quant::fingerprint(&m).unwrap().as_str()));
    let (kind, classes) = (folo_nn::io::FileKind::Weights, (0..4).map(|c| c.to_string()).collect());
    let mut meta =
        folo_nn::io::Metadata::new(kind, folo_nn::io::ModelConfig::new("n", 4).unwrap(), classes, 640)
            .unwrap();
    assert_eq!(Calibration::from_metadata(&meta).unwrap(), None);
    cal.store(&mut meta).unwrap();
    let back = folo_nn::io::Metadata::from_map(&meta.to_map()).unwrap();
    assert_eq!(Calibration::from_metadata(&back).unwrap(), Some(cal.clone()));
    // Other weights (another seed) refuse it; without a fingerprint it is not checked.
    let mut other = model(9);
    let e = other.set_int8(Some(cal.clone())).unwrap_err().to_string();
    assert!(e.contains("other weights"), "{e}");
    other.set_int8(Some(Calibration { weights: None, ..cal })).unwrap();
}

#[test]
fn calibration_needs_two_passes_over_the_same_images() {
    let m = model(11);
    let e = quant::calibrate(&m, std::iter::empty::<folo_nn::Result<Tensor>>, Method::MinMax)
        .unwrap_err()
        .to_string();
    assert!(e.contains("at least one"), "{e}");
    // The public Calibrator too: nothing observed, one pass only, or passes over different images.
    let x = input(1, 64, 64, 12);
    assert!(quant::calib::Calibrator::new().finish(Method::MinMax).is_err());
    let mut c = quant::calib::Calibrator::new();
    c.observe(&m, &x).unwrap();
    assert!(c.finish(Method::MinMax).is_err());
    c.next_pass();
    c.observe(&m, &x).unwrap();
    c.observe(&m, &x).unwrap();
    assert!(c.finish(Method::MinMax).is_err());
    // An observation that fails part way spoils the calibration, though both passes saw the same image: at
    // 40 × 40 the backbone records its boundaries, then the neck's concatenation (5 × 5 against 6 × 6) fails.
    let mut c = quant::calib::Calibrator::new();
    c.observe(&m, &x).unwrap();
    assert!(c.observe(&m, &input(1, 40, 40, 13)).is_err());
    c.next_pass();
    c.observe(&m, &x).unwrap();
    let e = c.finish(Method::MinMax).unwrap_err().to_string();
    assert!(e.contains("observation failed"), "{e}");
}

#[test]
fn a_changed_activation_repacks_its_conv() {
    let mut m = model(17);
    let x = input(1, 64, 64, 18);
    let cal = calibrated(&m, &x);
    m.set_int8(Some(Calibration { weights: None, ..cal.clone() })).unwrap();
    let _g = no_grad();
    m.forward(&x).unwrap(); // packs every conv with its SiLU
    m.backbone.down2.act = folo_nn::Activation::Identity;
    let a = m.forward(&x).unwrap().box_raw.to_vec::<f32>().unwrap();
    // The same weights in another model (nothing packed yet), changed before its first forward.
    let mut fresh = model(17);
    fresh.backbone.down2.act = folo_nn::Activation::Identity;
    fresh.set_int8(Some(Calibration { weights: None, ..cal })).unwrap();
    let b = fresh.forward(&x).unwrap().box_raw.to_vec::<f32>().unwrap();
    assert!(
        a.iter().zip(&b).all(|(u, v)| u.to_bits() == v.to_bits()),
        "stale epilogue after an activation change"
    );
}

#[test]
fn reloading_weights_repacks_and_an_explicit_precision_turns_int8_off() {
    use folo_nn::Module;
    let (mut m, other) = (model(13), model(15));
    let x = input(1, 64, 64, 16);
    let cal = Calibration { weights: None, ..calibrated(&m, &x) };
    m.set_int8(Some(cal.clone())).unwrap();
    let _g = no_grad();
    m.forward(&x).unwrap(); // packs the int8 convs of these weights
    // Load `other`'s weights in place: the forward must use them, not the convs packed before.
    let mut bytes = Vec::new();
    let tensors = other.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    folo_nn::io::write_export(&mut bytes, &other.metadata().unwrap(), tensors, folo_nn::io::FloatDType::F32)
        .unwrap();
    let file = folo_nn::io::ModelFile::from_bytes(bytes).unwrap();
    m.load(&file, folo_nn::io::LoadOptions::strict(folo_nn::io::Purpose::Inference)).unwrap();
    let mut fresh = other;
    fresh.set_int8(Some(cal)).unwrap();
    let (a, b) = (m.forward(&x).unwrap().box_raw, fresh.forward(&x).unwrap().box_raw);
    let (a, b) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
    assert!(a.iter().zip(&b).all(|(u, v)| u.to_bits() == v.to_bits()), "stale int8 convs after load");
    // Predictor::with_precision runs the whole model at that precision.
    let cfg = folo_vision::PredictConfig::default();
    let p = folo_vision::Predictor::with_precision(m, cfg, folo_vision::Precision::F32).unwrap();
    assert!(p.model().int8().is_none());
}

#[test]
fn convs_the_kernel_does_not_take_run_in_f32() {
    // A `qkv` whose stride (65) is past the int8 kernel's limit but whose output still fits the graph (2 × 2 at
    // 64 × 64): both executors keep it `f32`, and the forward runs.
    let mut m = model(19);
    let attn = &mut m.backbone.psa.blocks.items_mut()[0].attn;
    let c = *attn.qkv.conv.config();
    let cfg = folo_nn::Conv2dConfig::new(c.in_channels, c.out_channels, 1).stride(65).padding(32);
    let conv = folo_nn::Conv2d::new(cfg, &mut Rng::seed(20));
    attn.qkv = folo_nn::ConvBnAct { conv, bn: None, act: folo_nn::Activation::Identity };
    let x = input(1, 64, 64, 21);
    let cal = Calibration { weights: None, ..calibrated(&m, &x) };
    m.set_int8(Some(cal.clone())).unwrap();
    let _g = no_grad();
    let want = FakeQuant::new(&cal).features(&m, &x).unwrap();
    let got = folo_core::cpu::in_pool(|| quant::exec::features(&m, &Int8Plan::new(cal), &x)).unwrap();
    assert_eq!(want[2].dims(), got[2].dims());
    m.forward(&x).unwrap();
}
