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

/// `m`'s int8 forward on `x` with conv `name` left to the executors' own eligibility check, against the same with
/// `name` listed in `f32_convs`: the fake-quant features and the int8 forward must be the same bits, so both
/// executors kept that conv `f32` (and nothing panicked).
fn assert_runs_in_f32(mut m: FoloModel, name: &str, x: &Tensor) {
    let cal = Calibration { weights: None, ..calibrated(&m, x) };
    let listed = Calibration { f32_convs: vec![name.to_string()], ..cal.clone() };
    let _g = no_grad();
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let (a, b) =
        (FakeQuant::new(&cal).features(&m, x).unwrap(), FakeQuant::new(&listed).features(&m, x).unwrap());
    for (p, q) in a.iter().zip(&b) {
        assert_eq!(bits(p), bits(q), "{name}: the fake-quant reference quantised its weights");
    }
    m.set_int8(Some(listed)).unwrap();
    let want = m.forward(x).unwrap().box_raw;
    m.set_int8(Some(cal)).unwrap();
    let got = m.forward(x).unwrap().box_raw;
    assert_eq!(bits(&want), bits(&got), "{name}: the int8 forward differs from its f32 conv");
}

/// A fused conv `c → o` with `k`, `stride` and `padding` (no activation).
fn fused(c: usize, o: usize, k: usize, stride: usize, padding: usize, seed: u64) -> folo_nn::ConvBnAct {
    let cfg = folo_nn::Conv2dConfig::new(c, o, k).stride(stride).padding(padding);
    let conv = folo_nn::Conv2d::new(cfg, &mut Rng::seed(seed));
    folo_nn::ConvBnAct { conv, bn: None, act: folo_nn::Activation::Identity }
}

#[test]
fn convs_the_kernel_does_not_take_run_in_f32() {
    let x = input(1, 64, 64, 21);
    // A `qkv` whose stride (65) is past the kernel's limit, though its output still fits the graph: on the 2 × 2
    // input, padding 64 gives a 2 × 2 output whose second row and column read input pixel (1, 1).
    let mut m = model(19);
    let attn = &mut m.backbone.psa.blocks.items_mut()[0].attn;
    let c = *attn.qkv.conv.config();
    attn.qkv = fused(c.in_channels, c.out_channels, 1, 65, 64, 20);
    // The fake-quant `qkv` output itself (no requantisation after it to hide a difference): the same bits as with
    // the conv listed `f32`.
    let name = "backbone.psa.block0.attn.qkv";
    let cal = Calibration { weights: None, ..calibrated(&m, &x) };
    let listed = Calibration { f32_convs: vec![name.to_string()], ..cal.clone() };
    let q = cal.param("backbone.psa.cv1").unwrap();
    let t = quant::fake::fake_quantize(
        &Tensor::rand_uniform([1, c.in_channels, 2, 2], -1.0, 3.0, &mut Rng::seed(24)),
        q,
    )
    .unwrap();
    let fq = quant::fake::Fq { t, q };
    let qkv = &m.backbone.psa.blocks.items()[0].attn.qkv;
    let run = |cal: &Calibration| {
        let _g = no_grad();
        let y = quant::walk::Exec::conv_f32(&mut FakeQuant::new(cal), name, qkv, &fq).unwrap();
        y.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    };
    assert_eq!(run(&cal), run(&listed), "the fake-quant reference quantised the unsupported conv's weights");
    assert_runs_in_f32(m, name, &x);
    // A `proj` with a 2 × 2 kernel: its 1 × 1 output is broadcast over the 2 × 2 residual in f32.
    let mut m = model(19);
    let attn = &mut m.backbone.psa.blocks.items_mut()[0].attn;
    let c = attn.proj.conv.config().in_channels;
    attn.proj = fused(c, c, 2, 1, 0, 22);
    assert_runs_in_f32(m, "backbone.psa.block0.attn.proj", &x);
    // A weight of another shape (4 × 4) than the conv's config (3 × 3): the f32 conv follows the weight.
    let mut m = model(19);
    let d = m.backbone.down2.conv.weight.dims().to_vec();
    m.backbone.down2.conv.weight = Tensor::rand_uniform([d[0], d[1], 4, 4], -0.05, 0.05, &mut Rng::seed(23));
    assert_runs_in_f32(m, "backbone.down2", &x);
}

/// Legacy `version: 1` calibrations must keep coupled single-part concat quantization even when two concat
/// branches have >1.5× scale ratio, while `version: 2` decouples them (matching `FakeQuant` on both).
#[test]
fn v1_calibration_keeps_coupled_concat_weights() {
    let x = input(2, 64, 96, 2);
    let m = model(1);
    let mut cal_v2 = calibrated(&m, &x);
    cal_v2.weights = None;
    cal_v2.u8_stem = false;
    // Build a v1-style per-tensor map where `neck.td4.cv1`'s concat branches (`psa.cv2` vs `stage4.cv2`)
    // differ in scale by ~4× (> 1.5× decoupling threshold) without clipping any layer's activations.
    cal_v2.channel_ranges.clear();
    if let Some(r) = cal_v2.ranges.get_mut("backbone.stage4.cv2") {
        *r = [r[0] * 4.0, r[1] * 4.0];
    }
    let cal_v1 = Calibration { version: quant::VERSION_V1, ..cal_v2.clone() };
    let (fq_v1, fq_v2) = (
        FakeQuant::new(&cal_v1).features(&m, &x).unwrap(),
        FakeQuant::new(&cal_v2).features(&m, &x).unwrap(),
    );
    let mut m1 = model(1);
    let mut m2 = model(1);
    m1.set_int8(Some(cal_v1.clone())).unwrap();
    m2.set_int8(Some(cal_v2.clone())).unwrap();
    let _g = no_grad();
    let (y1, y2) = (m1.forward(&x).unwrap(), m2.forward(&x).unwrap());
    // Executor matches FakeQuant for BOTH v1 (coupled) and v2 (decoupled), while v1 and v2 stay distinct.
    let fq_p3_v1 = fq_v1[0].to_vec::<f32>().unwrap();
    let fq_p3_v2 = fq_v2[0].to_vec::<f32>().unwrap();
    assert_ne!(fq_p3_v1, fq_p3_v2, "v1 must not decouple unequal concat branch scales like v2");
    let to_bits = |t: &Tensor| t.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
    assert_ne!(
        to_bits(&y1.box_raw),
        to_bits(&y2.box_raw),
        "int8 executor v1 must stay coupled and distinct from v2 decoupled concat"
    );
    for (ver, cal, fq) in [("v1", &cal_v1, &fq_v1), ("v2", &cal_v2, &fq_v2)] {
        let plan = Int8Plan::new(cal.clone());
        let exec_f = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
        for (i, name) in ["neck.out3.cv2", "neck.out4.cv2", "neck.out5.cv2"].iter().enumerate() {
            let s = cal.param(name).unwrap().scale;
            let (a, b) = (fq[i].to_vec::<f32>().unwrap(), exec_f[i].to_vec::<f32>().unwrap());
            assert_eq!(a.len(), b.len());
            let steps: Vec<f32> = a.iter().zip(&b).map(|(p, q)| ((p - q) / s).abs().round()).collect();
            let exact = steps.iter().filter(|&&d| d == 0.0).count() as f64 / steps.len() as f64;
            let worst = steps.iter().copied().fold(0f32, f32::max);
            assert!(exact > 0.99 && worst <= 2.0, "{ver} {name}: {:.4} exact, worst {worst} steps", exact);
        }
    }
}

#[test]
fn f32_convs_subgraphs_skip_redundant_quantization_and_match_fake_quant() {
    let m = model(29);
    let x = input(2, 64, 96, 30);
    let base_cal = Calibration { weights: None, ..calibrated(&m, &x) };
    // Include P5 tail (`neck.out5.*`, consecutive f32 convs feeding the f32 head) plus the entire PSA
    // attention/FFN block (`backbone.psa.*`, exercising deferred `proj.in`, `narrow`, `upsample2`, and the
    // multi-consumer `x10` skip into both int8 `neck.td4.cv1` and f32 `neck.out5.cv1`).
    let mut mixed_cal = base_cal.clone().with_p5_tail_f32(&m);
    for name in [
        "backbone.psa.cv1",
        "backbone.psa.block0.attn.qkv",
        "backbone.psa.block0.attn.proj",
        "backbone.psa.block0.ffn1",
        "backbone.psa.block0.ffn2",
        "backbone.psa.cv2",
    ] {
        mixed_cal.f32_convs.push(name.to_string());
    }

    let _g = no_grad();
    let want = FakeQuant::new(&mixed_cal).features(&m, &x).unwrap();
    let plan = Int8Plan::new(mixed_cal.clone());
    let got = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
    for (i, name) in ["neck.out3.cv2", "neck.out4.cv2", "neck.out5.cv2"].iter().enumerate() {
        let s = mixed_cal.param(name).unwrap().scale;
        let (a, b) = (want[i].to_vec::<f32>().unwrap(), got[i].to_vec::<f32>().unwrap());
        let steps: Vec<f32> = a.iter().zip(&b).map(|(p, q)| ((p - q) / s).abs().round()).collect();
        let exact = steps.iter().filter(|&&d| d == 0.0).count() as f64 / steps.len() as f64;
        let worst = steps.iter().copied().fold(0f32, f32::max);
        assert!(exact > 0.99 && worst <= 2.0, "mixed {name}: {:.4} exact, worst {worst} steps", exact);
    }

    // Thread-count bit-determinism on the mixed f32/int8 graph (exercises lazy QAct::F32 -> U8 caching on
    // multi-consumer skips like `x10` into both int8 `neck.td4.cv1` and f32 `neck.out5.cv1`, with `no_grad`
    // created inside each worker closure since `no_grad` is thread-local).
    let mut m_mixed = model(29);
    m_mixed.set_int8(Some(mixed_cal)).unwrap();
    let y_t1 = pool(1).install(|| {
        let _g = no_grad();
        m_mixed.forward(&x).unwrap()
    });
    let y_t7 = pool(7).install(|| {
        let _g = no_grad();
        m_mixed.forward(&x).unwrap()
    });
    for (p, q) in [(&y_t1.box_raw, &y_t7.box_raw), (&y_t1.cls_raw, &y_t7.cls_raw)] {
        let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
        assert!(
            p.iter().zip(&q).all(|(u, v)| u.to_bits() == v.to_bits()),
            "mixed int8 not thread-deterministic"
        );
    }

    // When every boundary conv is in `f32_convs` (with `u8_stem: false`), lazy QAct::F32 skips every
    // intermediate u8 quantise/dequantise boundary, matching the full `f32` model bit-for-bit.
    let mut all_f32_cal = Calibration { u8_stem: false, ..base_cal };
    all_f32_cal.f32_convs = all_f32_cal.ranges.keys().cloned().collect();
    all_f32_cal.f32_convs.push("backbone.psa.block0.attn.qkv".to_string());
    let f32_out = m.forward(&x).unwrap();
    let mut m_all_f32 = model(29);
    m_all_f32.set_int8(Some(all_f32_cal)).unwrap();
    let y_all_f32 = m_all_f32.forward(&x).unwrap();
    for (p, q) in [(&f32_out.box_raw, &y_all_f32.box_raw), (&f32_out.cls_raw, &y_all_f32.cls_raw)] {
        let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
        assert!(
            p.iter().zip(&q).all(|(u, v)| u.to_bits() == v.to_bits()),
            "all-f32_convs int8 plan must skip all u8 ping-pong and match f32 bit-for-bit"
        );
    }
}

#[test]
fn bias_in_place_mutation_repacks_int8_conv() {
    let mut m = model(31);
    let x = input(1, 64, 64, 32);
    let cal = Calibration { weights: None, ..calibrated(&m, &x) };
    m.set_int8(Some(cal.clone())).unwrap();
    let _g = no_grad();
    m.forward(&x).unwrap(); // populates BiasCache + Packed QConv on every conv

    // Mutate `backbone.down2` bias in place (simulating weight/bias reload without replacing the Tensor).
    let bias = m.backbone.down2.conv.bias.as_ref().expect("fused conv has bias");
    let shifted: Vec<f32> = bias.to_vec::<f32>().unwrap().into_iter().map(|v| v + 0.5).collect();
    let shifted_t = Tensor::from_vec(shifted, bias.dims().to_vec()).unwrap();
    bias.copy_from(&shifted_t).unwrap();
    let got = m.forward(&x).unwrap().box_raw.to_vec::<f32>().unwrap();

    // Compare against a fresh model with the same shifted bias (nothing cached yet).
    let mut fresh = model(31);
    fresh.backbone.down2.conv.bias.as_ref().unwrap().copy_from(&shifted_t).unwrap();
    fresh.set_int8(Some(cal)).unwrap();
    let want = fresh.forward(&x).unwrap().box_raw.to_vec::<f32>().unwrap();
    assert!(
        got.iter().zip(&want).all(|(u, v)| u.to_bits() == v.to_bits()),
        "stale BiasCache / QConv after in-place bias mutation"
    );
}

#[test]
fn fused_cpu_attention_matches_tensor_reference_and_is_thread_deterministic() {
    use folo_nn::Forward;
    use folo_vision::model::blocks::{Attention, Fuse, Init};

    let mut rng = Rng::seed(37);
    let mut attn = Attention::new(128, 2, &mut Init::new(&mut rng)).unwrap();
    attn.fuse_().unwrap();

    // Test standard 20×20 plus border edge cases (1×1, 1×3, 3×1, 2×2).
    for (b, h, w) in [(1, 20, 20), (2, 4, 6), (1, 1, 1), (1, 1, 3), (1, 3, 1), (1, 2, 2)] {
        let x = Tensor::rand_uniform([b, 128, h, w], -1.0, 1.0, &mut Rng::seed(100 + (h * 31 + w) as u64));
        // With grad enabled, `Attention::forward` takes the generic tensor reference path.
        let ref_y = attn.forward(&x).unwrap().to_vec::<f32>().unwrap();
        // Under `no_grad` (created inside each worker closure because `no_grad` is thread-local),
        // `Attention::forward` takes the fused CPU path.
        let fast_t1 = pool(1).install(|| {
            let _g = no_grad();
            attn.forward(&x).unwrap().to_vec::<f32>().unwrap()
        });
        let fast_t7 = pool(7).install(|| {
            let _g = no_grad();
            attn.forward(&x).unwrap().to_vec::<f32>().unwrap()
        });
        assert!(
            fast_t1.iter().zip(&fast_t7).all(|(u, v)| u.to_bits() == v.to_bits()),
            "fused attention ({b}x128x{h}x{w}) differs across thread counts"
        );
        let max_err = ref_y.iter().zip(&fast_t1).map(|(u, v)| (u - v).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "fused attention ({b}x128x{h}x{w}) max err {max_err} >= 1e-4");
    }

    // Regression: cancelling large PE taps (+/- 2^24) must not swallow a small bias (0.5).
    // With `v = 1.0` everywhere, attention output `o = 1.0`; with PE weights `+2^24` and `-2^24` on the
    // left/right neighbors and bias `0.5`, center pixel (1, 1) must yield `1.0 + (0.0 + 0.5) = 1.5`
    // (starting the accumulation with `bias` would round `0.5 + 2^24` to `2^24` and yield `1.0`).
    let mut rng = Rng::seed(41);
    let mut cancel_attn = Attention::new(4, 1, &mut Init::new(&mut rng)).unwrap();
    cancel_attn.fuse_().unwrap();
    let mut pe_w = vec![0.0f32; 4 * 9];
    for ch in 0..4 {
        pe_w[ch * 9 + 3] = 16_777_216.0; // +2^24 on left neighbor
        pe_w[ch * 9 + 5] = -16_777_216.0; // -2^24 on right neighbor
    }
    cancel_attn.pe.conv.weight = Tensor::from_vec(pe_w, [4, 1, 3, 3]).unwrap();
    cancel_attn.pe.conv.bias = Some(Tensor::from_vec(vec![0.5f32; 4], [4]).unwrap());
    let c_qkv = cancel_attn.qkv.conv.config().out_channels;
    cancel_attn.qkv.conv.weight = Tensor::zeros([c_qkv, 4, 1, 1]);
    let mut qkv_b = vec![0.0f32; c_qkv];
    for v_b in &mut qkv_b[2 * cancel_attn.key_dim..] {
        *v_b = 1.0;
    }
    cancel_attn.qkv.conv.bias = Some(Tensor::from_vec(qkv_b, [c_qkv]).unwrap());
    let mut eye = vec![0.0f32; 16];
    for i in 0..4 {
        eye[i * 4 + i] = 1.0;
    }
    cancel_attn.proj.conv.weight = Tensor::from_vec(eye, [4, 4, 1, 1]).unwrap();
    cancel_attn.proj.conv.bias = Some(Tensor::zeros([4]));

    let x_c = Tensor::zeros([1, 4, 3, 3]);
    let ref_c = cancel_attn.forward(&x_c).unwrap().to_vec::<f32>().unwrap();
    let fast_c = {
        let _g = no_grad();
        cancel_attn.forward(&x_c).unwrap().to_vec::<f32>().unwrap()
    };
    assert_eq!(ref_c, fast_c, "PE bias handling diverged from reference");
    for ch in 0..4 {
        assert_eq!(fast_c[ch * 9 + 4], 1.5, "center pixel lost bias on cancelling taps (ch {ch})");
    }
}

#[test]
fn scratch_pool_reuses_buffers_across_layers_and_calls() {
    use rayon::prelude::*;

    let m = model(43);
    let x = input(1, 64, 96, 44);
    let cal = calibrated(&m, &x);
    let plan = Int8Plan::new(cal);
    let _g = no_grad();

    // First forward populates the pool and already reuses buffers across sequential layers.
    let y1 = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
    let s1 = plan.scratch_stats();
    assert!(s1.allocs > 0 && s1.reuses > s1.allocs, "expected intra-forward reuse: {s1:?}");

    // Second + third forwards settle to zero fresh allocations while reusing every C4 activation buffer.
    let y2 = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
    let s2 = plan.scratch_stats();
    let y3 = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
    let s3 = plan.scratch_stats();
    assert_eq!(s3.allocs, s2.allocs, "steady-state forward allocated fresh buffers: {s2:?} -> {s3:?}");
    assert!(s3.reuses > s2.reuses, "steady-state forward did not record buffer reuses: {s2:?} -> {s3:?}");

    let to_bits = |t: &Tensor| t.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for i in 0..3 {
        assert_eq!(to_bits(&y1[i]), to_bits(&y2[i]), "pooled forward 2 diverged on level {i}");
        assert_eq!(to_bits(&y1[i]), to_bits(&y3[i]), "pooled forward 3 diverged on level {i}");
    }

    // Concurrent forwards sharing one `Int8Plan` (as in `Predictor::predict_batch`) must each check out their
    // own `ScratchPool` and match bit-for-bit.
    let par_outs: Vec<[Tensor; 3]> = pool(4).install(|| {
        (0..4)
            .into_par_iter()
            .map(|_| {
                let _g = no_grad();
                quant::exec::features(&m, &plan, &x).unwrap()
            })
            .collect()
    });
    for (idx, out) in par_outs.iter().enumerate() {
        for i in 0..3 {
            assert_eq!(to_bits(&y1[i]), to_bits(&out[i]), "concurrent forward {idx} diverged on level {i}");
        }
    }
}

#[test]
fn consumer_aware_f32_out_upsample_sppf_and_narrow_shortcuts_match_fake_quant() {
    let m = model(47);
    let x = input(2, 64, 96, 48);
    let base_cal = Calibration { weights: None, ..calibrated(&m, &x) };
    // Exercise every consumer-aware boundary shortcut simultaneously:
    // - `neck.out5.*`: `neck.down4` (int8) writes `QOut::F32` straight into `neck.out5.cv1` (f32).
    // - `backbone.stage2.block0.cv2` alone: `stage2.block0.cv1` (int8) writes `QOut::F32` straight into `cv2`.
    // - `backbone.psa.cv2` (f32) into int8 `neck.td4.cv1`: `upsample2_for(&x10, false)` quantises at 2×2 first.
    // - `backbone.sppf.cv1` (f32) into int8 `sppf.cv2`: `ensure_u8` quantises once before 3× `q8::max_pool`.
    // - `backbone.stage3.cv1` (f32) into int8 `stage3.cv2`: `ensure_u8` quantises once before `narrow` + `cv2`.
    let mut mixed = base_cal.with_p5_tail_f32(&m);
    for name in ["backbone.stage2.block0.cv2", "backbone.psa.cv2", "backbone.sppf.cv1", "backbone.stage3.cv1"]
    {
        mixed.f32_convs.push(name.to_string());
    }

    let _g = no_grad();
    let want = FakeQuant::new(&mixed).features(&m, &x).unwrap();
    let plan = Int8Plan::new(mixed.clone());
    let got = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
    for (i, name) in ["neck.out3.cv2", "neck.out4.cv2", "neck.out5.cv2"].iter().enumerate() {
        let s = mixed.param(name).unwrap().scale;
        let (a, b) = (want[i].to_vec::<f32>().unwrap(), got[i].to_vec::<f32>().unwrap());
        let steps: Vec<f32> = a.iter().zip(&b).map(|(p, q)| ((p - q) / s).abs().round()).collect();
        let exact = steps.iter().filter(|&&d| d == 0.0).count() as f64 / steps.len() as f64;
        let worst = steps.iter().copied().fold(0f32, f32::max);
        assert!(exact > 0.99 && worst <= 2.0, "shortcut {name}: {:.4} exact, worst {worst} steps", exact);
    }

    let mut m_det = model(47);
    m_det.set_int8(Some(mixed)).unwrap();
    let y1 = pool(1).install(|| {
        let _g = no_grad();
        m_det.forward(&x).unwrap()
    });
    let y5 = pool(5).install(|| {
        let _g = no_grad();
        m_det.forward(&x).unwrap()
    });
    for (p, q) in [(&y1.box_raw, &y5.box_raw), (&y1.cls_raw, &y5.cls_raw)] {
        let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
        assert!(
            p.iter().zip(&q).all(|(u, v)| u.to_bits() == v.to_bits()),
            "shortcut graph not deterministic"
        );
    }
}

#[test]
fn sensitivity_analysis_and_selective_f32_policy_wire_end_to_end() {
    let mut m = model(53);
    let x = input(2, 64, 64, 54);
    let cal = calibrated(&m, &x);

    let ranked = quant::analyze_sensitivity(&m, &cal).unwrap();
    assert!(ranked.len() >= 50, "expected all int8-eligible convs, got {}", ranked.len());
    let total_frac: f64 = ranked.iter().map(|e| e.mac_fraction).sum();
    assert!((total_frac - 1.0).abs() < 1e-6, "MAC fractions should sum to 1.0, got {total_frac}");
    for pair in ranked.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "sensitivity list not sorted descending: {} ({}) < {} ({})",
            pair[0].name,
            pair[0].score,
            pair[1].name,
            pair[1].score
        );
    }
    // Residual bottleneck partners must be linked symmetrically.
    let s2_cv1 = ranked.iter().find(|e| e.name == "backbone.stage2.block0.cv1").unwrap();
    let s2_cv2 = ranked.iter().find(|e| e.name == "backbone.stage2.block0.cv2").unwrap();
    assert_eq!(s2_cv1.residual_partner.as_deref(), Some("backbone.stage2.block0.cv2"));
    assert_eq!(s2_cv2.residual_partner.as_deref(), Some("backbone.stage2.block0.cv1"));

    // Zero budget selects nothing.
    let empty = quant::select_f32_convs(
        &m,
        &cal,
        quant::SensitivityPolicy { max_f32_mac_fraction: 0.0, ..Default::default() },
    )
    .unwrap();
    assert!(empty.is_empty());

    // Budget cap and atomic residual bottleneck pairing hold.
    let policy = quant::SensitivityPolicy {
        max_f32_mac_fraction: 0.05,
        sqnr_threshold_db: 45.0,
        include_p5_head_out: true,
        pair_residual_bottlenecks: true,
    };
    let picked = quant::select_f32_convs(&m, &cal, policy).unwrap();
    assert!(picked.contains(&"neck.out5.cv2".to_string()), "expected neck.out5.cv2 in {picked:?}");
    let used_macs: f64 =
        picked.iter().map(|n| ranked.iter().find(|e| &e.name == n).unwrap().mac_fraction).sum();
    assert!(used_macs <= policy.max_f32_mac_fraction + 1e-9, "exceeded MAC budget: {used_macs}");
    for name in &picked {
        if let Some(partner) =
            ranked.iter().find(|e| &e.name == name).and_then(|e| e.residual_partner.as_ref())
        {
            assert!(
                picked.contains(partner),
                "residual bottleneck `{name}` promoted without partner `{partner}`: {picked:?}"
            );
        }
    }

    // Wired end-to-end via `Calibration::with_sensitivity_f32` and `FoloModel::set_int8`.
    let sens_cal = cal.with_sensitivity_f32(&m, policy).unwrap();
    let _g = no_grad();
    let want = FakeQuant::new(&sens_cal).features(&m, &x).unwrap();
    m.set_int8(Some(sens_cal.clone())).unwrap();
    let plan = Int8Plan::new(sens_cal.clone());
    let got = folo_core::cpu::in_pool(|| quant::exec::features(&m, &plan, &x)).unwrap();
    for (i, name) in ["neck.out3.cv2", "neck.out4.cv2", "neck.out5.cv2"].iter().enumerate() {
        let s = sens_cal.param(name).unwrap().scale;
        let (a, b) = (want[i].to_vec::<f32>().unwrap(), got[i].to_vec::<f32>().unwrap());
        let steps: Vec<f32> = a.iter().zip(&b).map(|(p, q)| ((p - q) / s).abs().round()).collect();
        let exact = steps.iter().filter(|&&d| d == 0.0).count() as f64 / steps.len() as f64;
        let worst = steps.iter().copied().fold(0f32, f32::max);
        assert!(exact > 0.99 && worst <= 2.0, "sens {name}: {:.4} exact, worst {worst} steps", exact);
    }

    // End-to-end `calibrate_sample` on a tiny synthetic dataset under `TestDir` wires sensitivity selection.
    let dir = folo_test_support::TestDir::new("folo-int8-sens");
    let syn_cfg = folo_data::synthetic::SyntheticConfig {
        images: 3,
        imgsz: 64,
        seed: 55,
        ..folo_data::synthetic::SyntheticConfig::default()
    };
    folo_data::synthetic::write_yolo(&syn_cfg, dir.path(), folo_data::ImageFormat::Png).unwrap();
    let ds = folo_data::open(dir.path()).unwrap();
    let m_fresh = model(53);
    let (sample_cal, n_imgs) = quant::calibrate_sample(
        &m_fresh,
        ds.split(folo_data::Split::Train),
        2,
        0,
        (64, 2, 1),
        Method::Percentile { p: 1e-4 },
    )
    .unwrap();
    assert_eq!(n_imgs, 2);
    assert_eq!(sample_cal.version, quant::VERSION_V2);
    assert!(
        sample_cal.f32_convs.contains(&"neck.out5.cv2".to_string()),
        "calibrate_sample should wire sensitivity f32 selection: {:?}",
        sample_cal.f32_convs
    );
    let mut m_check = model(53);
    m_check.set_int8(Some(sample_cal)).unwrap();
}

#[test]
fn sensitivity_edge_cases_and_scratch_pool_error_recovery() {
    // 1. Unfused training-mode model: `analyze_sensitivity` must NOT mutate `pe.bn` running stats.
    let mut unfused = FoloModel::with_rng(ModelConfig::new("n", 4).unwrap(), &mut Rng::seed(61)).unwrap();
    // Seed non-trivial running stats on `pe.bn` so any zero-tensor forward would visibly shift them.
    {
        let bn = unfused.backbone.psa.blocks.items_mut()[0].attn.pe.bn.as_mut().expect("unfused pe has bn");
        let n = bn.running_mean.dims()[0];
        bn.running_mean.copy_from(&Tensor::from_vec(vec![1.25f32; n], [n]).unwrap()).unwrap();
        bn.running_var.copy_from(&Tensor::from_vec(vec![2.5f32; n], [n]).unwrap()).unwrap();
    }
    let fused_ref = model(61);
    let x = input(1, 64, 64, 62);
    let cal = calibrated(&fused_ref, &x);
    let before_mean = unfused.backbone.psa.blocks.items()[0]
        .attn
        .pe
        .bn
        .as_ref()
        .unwrap()
        .running_mean
        .to_vec::<f32>()
        .unwrap();
    let before_var = unfused.backbone.psa.blocks.items()[0]
        .attn
        .pe
        .bn
        .as_ref()
        .unwrap()
        .running_var
        .to_vec::<f32>()
        .unwrap();
    let _ = quant::analyze_sensitivity(&unfused, &cal).unwrap();
    let after_mean = unfused.backbone.psa.blocks.items()[0]
        .attn
        .pe
        .bn
        .as_ref()
        .unwrap()
        .running_mean
        .to_vec::<f32>()
        .unwrap();
    let after_var = unfused.backbone.psa.blocks.items()[0]
        .attn
        .pe
        .bn
        .as_ref()
        .unwrap()
        .running_var
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(before_mean, after_mean, "analyze_sensitivity mutated pe.bn running_mean");
    assert_eq!(before_var, after_var, "analyze_sensitivity mutated pe.bn running_var");

    // 2. Policy validation rejects NaN / out-of-range budgets and NaN SQNR threshold.
    for bad in [
        quant::SensitivityPolicy { max_f32_mac_fraction: f64::NAN, ..Default::default() },
        quant::SensitivityPolicy { max_f32_mac_fraction: -0.05, ..Default::default() },
        quant::SensitivityPolicy { max_f32_mac_fraction: 1.5, ..Default::default() },
        quant::SensitivityPolicy { max_f32_mac_fraction: f64::INFINITY, ..Default::default() },
        quant::SensitivityPolicy { sqnr_threshold_db: f64::NAN, ..Default::default() },
    ] {
        assert!(
            quant::select_f32_convs(&fused_ref, &cal, bad).is_err(),
            "invalid policy should error: {bad:?}"
        );
    }

    // 3. Chaining `with_p5_tail_f32` + `with_sensitivity_f32` enforces the MAC cap on the combined set.
    let p5_only = cal.clone().with_p5_tail_f32(&fused_ref);
    let chained = p5_only
        .clone()
        .with_sensitivity_f32(
            &fused_ref,
            quant::SensitivityPolicy {
                max_f32_mac_fraction: 0.05,
                sqnr_threshold_db: 100.0,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        chained.f32_convs, p5_only.f32_convs,
        "with_sensitivity_f32 must not append new convs when existing f32_convs already exceed the 5% cap"
    );

    // 4. Broadcast `f32` residual shape propagation (`2x2` `attn.proj` producing 1x1 broadcast over 2x2 at 64x64).
    let mut bcast_m = model(63);
    let c_attn = bcast_m.backbone.psa.blocks.items()[0].attn.proj.conv.config().in_channels;
    bcast_m.backbone.psa.blocks.items_mut()[0].attn.proj = fused(c_attn, c_attn, 2, 1, 0, 64);
    let bcast_entries = quant::analyze_sensitivity_at(&bcast_m, &cal, (64, 64)).unwrap();
    let ffn1 = bcast_entries.iter().find(|e| e.name == "backbone.psa.block0.ffn1").unwrap();
    let ffn2 = bcast_entries.iter().find(|e| e.name == "backbone.psa.block0.ffn2").unwrap();
    assert_eq!(ffn1.out_hw, (2, 2), "ffn1 lost broadcast residual shape after 2x2 attn.proj");
    assert_eq!(ffn2.out_hw, (2, 2), "ffn2 lost broadcast residual shape after 2x2 attn.proj");

    // 5. All-zero weights yield 100 dB SQNR (never promoted), and `version: 1` calibrations analyse cleanly.
    let mut zero_m = model(61);
    let z_dims = zero_m.backbone.stage2.cv1.conv.weight.dims().to_vec();
    zero_m.backbone.stage2.cv1.conv.weight = Tensor::zeros(z_dims);
    let zero_entries = quant::analyze_sensitivity(&zero_m, &cal).unwrap();
    let z_entry = zero_entries.iter().find(|e| e.name == "backbone.stage2.cv1").unwrap();
    assert_eq!(z_entry.folded_weight_sqnr_db, 100.0);
    assert_eq!(z_entry.score, 0.0);
    let cal_v1 =
        Calibration { version: quant::VERSION_V1, channel_ranges: Default::default(), ..cal.clone() };
    let v1_entries = quant::analyze_sensitivity(&fused_ref, &cal_v1).unwrap();
    assert!(v1_entries.len() >= 50);

    // 6. `ScratchPool` error recovery: a mid-graph failure (34×34 fails at `neck.td4` concat while live
    //    `SlotHandle`s exist) drops every handle and returns the pool cleanly so the next 64×64 forward
    //    allocates zero new buffers (instead of leaking the pool and re-allocating ~12 buffers).
    let plan = Int8Plan::new(cal);
    let _g = no_grad();
    let _ = folo_core::cpu::in_pool(|| quant::exec::features(&fused_ref, &plan, &x)).unwrap();
    let bad_x = input(1, 34, 34, 65);
    assert!(folo_core::cpu::in_pool(|| quant::exec::features(&fused_ref, &plan, &bad_x)).is_err());
    let after_err = plan.scratch_stats();
    let _ = folo_core::cpu::in_pool(|| quant::exec::features(&fused_ref, &plan, &x)).unwrap();
    let after_recovery = plan.scratch_stats();
    assert_eq!(
        after_recovery.allocs, after_err.allocs,
        "ScratchPool was not returned after an erroring forward: {after_err:?} -> {after_recovery:?}"
    );
    assert!(
        after_recovery.reuses > after_err.reuses,
        "expected buffer reuses on recovery forward: {after_err:?} -> {after_recovery:?}"
    );
}
