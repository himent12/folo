//! Recorded GPU forwards ([`folo_gpu::ForwardPlans`] behind a fused eval-mode `FoloModel`): a replayed forward gives
//! the same bits as running it, every change that could make a plan stale retires it, and plans are off on a
//! model on its own and on in a `Predictor`. The tests share the GPU context's switches (tensor cores, plans,
//! memory release), so they run one at a time.
//!
//! Skips (with a note) without a hardware adapter; set `FOLO_GPU_TEST_SOFTWARE=1` to run on WARP / llvmpipe.

use std::sync::Mutex;

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_nn::Module;
use folo_vision::fusion::{BoxFusion, ScoreRule};
use folo_vision::model::{FoloModel, HeadMode, ImageCandidates, ModelConfig};
use folo_vision::nms::NmsConfig;
use folo_vision::predict::{PredictConfig, Predictor, f16_unavailable};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn gpu() -> Option<(GpuContext, Device)> {
    let ctx = match GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    Some((ctx, folo_gpu::init().unwrap()))
}

/// A fused model on `dev` with plans on, as inside a `Predictor`.
fn model(dev: Device, nc: usize, seed: u64) -> FoloModel {
    let m = bare(dev, nc, seed);
    m.set_forward_plans(true);
    m
}

/// A fused model on `dev` as built: plans off.
fn bare(dev: Device, nc: usize, seed: u64) -> FoloModel {
    let mut m = FoloModel::with_rng(ModelConfig::new("n", nc).unwrap(), &mut Rng::seed(seed)).unwrap();
    m.fuse().unwrap();
    m.to_device(dev).unwrap();
    m
}

fn input(dev: Device, dims: [usize; 4], dtype: DType, seed: u64) -> Tensor {
    Tensor::rand_uniform(dims, 0.0, 1.0, &mut Rng::seed(seed))
        .to_device(dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

/// The bits of both raw outputs (f16 widened to f32, which is exact).
fn bits(m: &FoloModel, x: &Tensor) -> Vec<u32> {
    let o = m.forward(x).unwrap();
    let mut v = Vec::new();
    for t in [&o.box_raw, &o.cls_raw] {
        v.extend(t.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap().iter().map(|f| f.to_bits()));
    }
    v
}

/// The forward's bits with plans off for this model.
fn unplanned(m: &FoloModel, x: &Tensor) -> Vec<u32> {
    m.set_forward_plans(false);
    let b = bits(m, x);
    m.set_forward_plans(true);
    b
}

/// Calls the forward until it is answered from a plan (the third call: the first is noted, the second recorded),
/// checking every call's bits against `want`.
#[track_caller]
fn until_replayed(m: &FoloModel, x: &Tensor, want: &[u32]) {
    let before = m.plan_replays();
    for call in 0..3 {
        assert_eq!(bits(m, x), want, "call {call}");
    }
    assert!(m.plan_replays() > before, "no replay after three calls: {:?}", m.plan_refusal());
}

/// The next forward's bits, asserting it was not answered from a plan.
#[track_caller]
fn not_replayed(m: &FoloModel, x: &Tensor) -> Vec<u32> {
    let before = m.plan_replays();
    let b = bits(m, x);
    assert_eq!(m.plan_replays(), before, "a stale plan answered");
    b
}

#[test]
fn replays_are_bit_identical_across_shapes_dtypes_and_tensor_cores() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let tc = ctx.tensor_cores();
    let mut configs = vec![(DType::F32, false)];
    if f16_unavailable(dev).is_none() {
        configs.push((DType::F16, false));
        if ctx.can_enable_tensor_cores() {
            configs.push((DType::F16, true));
        }
    }
    for (dtype, cores) in configs {
        ctx.set_tensor_cores(cores);
        let mut m = model(dev, 4, 0);
        m.to_dtype(dtype).unwrap();
        for dims in [[1, 3, 64, 96], [1, 3, 96, 64], [16, 3, 64, 64]] {
            let x = input(dev, dims, dtype, 3);
            let want = unplanned(&m, &x);
            until_replayed(&m, &x, &want);
        }
    }
    ctx.set_tensor_cores(tc);
}

#[test]
fn two_shapes_two_models_and_two_head_modes_alternate() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let (a, b) = (model(dev, 4, 0), model(dev, 4, 1));
    let xs = [input(dev, [1, 3, 64, 64], DType::F32, 1), input(dev, [1, 3, 32, 96], DType::F32, 2)];
    let want: Vec<Vec<Vec<u32>>> =
        [&a, &b].iter().map(|m| xs.iter().map(|x| unplanned(m, x)).collect()).collect();
    for round in 0..4 {
        for (mi, m) in [&a, &b].iter().enumerate() {
            for (xi, x) in xs.iter().enumerate() {
                assert_eq!(bits(m, x), want[mi][xi], "round {round}, model {mi}, input {xi}");
            }
        }
    }
    assert!(a.plan_replays() >= 4 && b.plan_replays() >= 4, "{} {}", a.plan_replays(), b.plan_replays());

    // A model with the one-to-one branch keeps a plan per head mode.
    let mut cfg = ModelConfig::new("n", 4).unwrap();
    cfg.head = "o2o".into();
    let mut m = FoloModel::new(cfg).unwrap();
    m.fuse().unwrap();
    m.to_device(dev).unwrap();
    let x = &xs[0];
    let o2o = unplanned(&m, x);
    m.set_head_mode(HeadMode::O2m).unwrap();
    let o2m = unplanned(&m, x);
    assert_ne!(o2o, o2m);
    for _ in 0..3 {
        m.set_head_mode(HeadMode::O2o).unwrap();
        assert_eq!(bits(&m, x), o2o);
        m.set_head_mode(HeadMode::O2m).unwrap();
        assert_eq!(bits(&m, x), o2m);
    }
    assert!(m.plan_replays() >= 2, "{:?}", m.plan_refusal());
}

#[test]
fn a_thousand_replays_keep_their_bits_and_results_stay_put() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let m = model(dev, 4, 0);
    let x = input(dev, [1, 3, 32, 32], DType::F32, 5);
    let y = input(dev, [1, 3, 32, 32], DType::F32, 6);
    let (want_x, want_y) = (unplanned(&m, &x), unplanned(&m, &y));
    until_replayed(&m, &x, &want_x);
    // A result held across later replays (of other inputs) keeps its values: replays hand out fresh results.
    let held = m.forward(&x).unwrap();
    let before = m.plan_replays();
    for i in 0..1000 {
        let o = m.forward(&y).unwrap();
        if i % 250 == 0 {
            drop(o);
            assert_eq!(bits(&m, &y), want_y, "replay {i}");
        }
    }
    assert!(m.plan_replays() >= before + 1000);
    let mut got = held.box_raw.to_vec::<f32>().unwrap();
    got.extend(held.cls_raw.to_vec::<f32>().unwrap());
    assert_eq!(got.iter().map(|f| f.to_bits()).collect::<Vec<_>>(), want_x);
}

#[test]
fn weight_changes_retire_the_plan() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    // `m` plans; `r`, the same weights, never does and gives the bits each change should lead to.
    let (mut m, mut r) = (model(dev, 4, 0), bare(dev, 4, 0));
    let x = input(dev, [1, 3, 64, 64], DType::F32, 7);
    let want = bits(&r, &x);
    until_replayed(&m, &x, &want);

    // An in-place write (as an optimizer step makes).
    for m in [&mut m, &mut r] {
        m.backbone.stem.conv.weight.mul_scalar_assign(0.5).unwrap();
    }
    let halved = bits(&r, &x);
    assert_ne!(halved, want);
    assert_eq!(not_replayed(&m, &x), halved);
    until_replayed(&m, &x, &halved);

    // A tensor replaced by another (the old one still alive elsewhere).
    let old = m.backbone.stem.conv.weight.clone();
    for m in [&mut m, &mut r] {
        m.backbone.stem.conv.weight = m.backbone.stem.conv.weight.mul_scalar(2.0).unwrap();
    }
    let doubled = bits(&r, &x);
    assert_ne!(doubled, halved);
    assert_eq!(not_replayed(&m, &x), doubled);
    until_replayed(&m, &x, &doubled);
    drop(old);

    // A round trip through the CPU (in place, same ids, new storage versions): the values do not change.
    m.to_device(Device::Cpu).unwrap();
    m.to_device(dev).unwrap();
    assert_eq!(not_replayed(&m, &x), doubled);
    until_replayed(&m, &x, &doubled);

    // `to_dtype` there and back: the values are rounded, so the plan must follow them.
    for m in [&mut m, &mut r] {
        m.to_dtype(DType::F16).unwrap();
        m.to_dtype(DType::F32).unwrap();
    }
    let rounded = bits(&r, &x);
    assert_ne!(rounded, doubled);
    assert_eq!(not_replayed(&m, &x), rounded);
    until_replayed(&m, &x, &rounded);
}

#[test]
fn context_switches_retire_plans_and_release_frees_them() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let mut m = model(dev, 4, 0);
    m.to_dtype(DType::F16).ok();
    let dtype = m.dtype();
    let x = input(dev, [1, 3, 64, 64], dtype, 8);
    let tc = ctx.tensor_cores();
    let flip = dtype == DType::F16 && ctx.can_enable_tensor_cores();
    let want = unplanned(&m, &x);
    let other = flip.then(|| {
        ctx.set_tensor_cores(!tc);
        let other = unplanned(&m, &x);
        ctx.set_tensor_cores(tc);
        other
    });
    until_replayed(&m, &x, &want);
    let (plans, bytes) = ctx.plan_stats();
    assert!(plans >= 1 && bytes > 0, "{plans} plans, {bytes} bytes");

    // The tensor-core switch picks other kernels: the plan is not used for the other setting (which gets its own),
    // and is used again once the switch is back.
    if let Some(other) = other {
        ctx.set_tensor_cores(!tc);
        let before = m.plan_replays();
        assert_eq!(bits(&m, &x), other);
        assert_eq!(m.plan_replays(), before, "the plan of the other setting answered");
        until_replayed(&m, &x, &other);
        ctx.set_tensor_cores(tc);
        let before = m.plan_replays();
        assert_eq!(bits(&m, &x), want);
        assert_eq!(m.plan_replays(), before + 1);
    }

    // Trimming the pool leaves a plan intact (it owns its buffers).
    ctx.trim_pool();
    let before = m.plan_replays();
    assert_eq!(bits(&m, &x), want);
    assert!(m.plan_replays() > before);

    // Releasing memory drops every plan.
    ctx.release_memory().unwrap();
    assert_eq!(ctx.plan_stats(), (0, 0));
    until_replayed(&m, &x, &want);

    // Plans off for the context: the forward runs as it is.
    ctx.set_forward_plans(false);
    let before = m.plan_replays();
    for _ in 0..3 {
        assert_eq!(bits(&m, &x), want);
    }
    assert_eq!(m.plan_replays(), before);
    ctx.set_forward_plans(true);

    // A dropped model takes its plans with it.
    until_replayed(&m, &x, &want);
    let held = ctx.plan_stats().0;
    drop(m);
    assert_eq!(ctx.plan_stats().0, held - 1);
}

#[test]
fn threads_forwarding_one_model_get_their_own_results() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let m = model(dev, 4, 0);
    let xs: Vec<Tensor> = (0..4).map(|i| input(dev, [1, 3, 32, 64], DType::F32, 20 + i)).collect();
    let want: Vec<Vec<u32>> = {
        let _g = no_grad();
        xs.iter().map(|x| unplanned(&m, x)).collect()
    };
    std::thread::scope(|s| {
        for (x, want) in xs.iter().zip(&want) {
            let m = &m;
            s.spawn(move || {
                let _g = no_grad();
                for call in 0..100 {
                    assert_eq!(&bits(m, x), want, "call {call}");
                }
            });
        }
    });
    assert!(m.plan_replays() > 0, "{:?}", m.plan_refusal());
}

#[test]
fn no_plan_with_gradients_on_or_in_training_mode() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let m = model(dev, 4, 0);
    let x = input(dev, [1, 3, 32, 32], DType::F32, 9);
    for _ in 0..4 {
        m.forward(&x).unwrap();
    }
    assert_eq!(m.plan_replays(), 0);
    let _g = no_grad();
    m.train();
    for _ in 0..4 {
        m.forward(&x).unwrap();
    }
    assert_eq!(m.plan_replays(), 0);
}

#[test]
fn no_plan_while_tracing() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    // (Autocast, the other context-wide refusal, is folo-gpu's `tests/plan.rs`: a fused model does not run under it.)
    let m = model(dev, 4, 0);
    let x = input(dev, [1, 3, 32, 32], DType::F32, 10);
    let want = bits(&m, &x);
    // A trace wants every operation run (a replay would leave it empty).
    ctx.trace_begin();
    for _ in 0..4 {
        assert_eq!(bits(&m, &x), want);
    }
    let ops = ctx.trace_end();
    assert_eq!(m.plan_replays(), 0);
    assert!(!ops.is_empty());
}

#[test]
fn a_model_on_its_own_does_not_plan_and_a_predictor_does() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let x = input(dev, [1, 3, 32, 32], DType::F32, 11);
    let m = bare(dev, 4, 0);
    for _ in 0..4 {
        m.forward(&x).unwrap();
    }
    assert_eq!(m.plan_replays(), 0);
    assert_eq!(m.plan_refusal(), None);
    let pred = Predictor::new(m, PredictConfig::default()).unwrap();
    for _ in 0..4 {
        pred.model().forward(&x).unwrap();
    }
    assert!(pred.model().plan_replays() > 0, "{:?}", pred.model().plan_refusal());
}

/// Every field of every candidate (and precision) as bits.
fn cand_bits(c: &[ImageCandidates]) -> Vec<Vec<u32>> {
    c.iter()
        .map(|ic| {
            let mut v = Vec::new();
            for c in &ic.cands {
                let b = c.bbox;
                v.extend([c.anchor, c.class, c.score.to_bits(), b.x1.to_bits(), b.y1.to_bits()]);
                v.extend([b.x2.to_bits(), b.y2.to_bits()]);
            }
            v.push(u32::MAX);
            for p in ic.precisions.iter().flatten() {
                v.extend(p.map(f32::to_bits));
            }
            v.push(ic.precisions.is_some() as u32);
            for p in ic.bin_precisions.iter().flatten() {
                v.extend(p.map(f32::to_bits));
            }
            v.push(ic.bin_precisions.is_some() as u32);
            for e in ic.entropies.iter().flatten() {
                v.push(e.to_bits());
            }
            v.push(ic.entropies.is_some() as u32);
            for s in ic.strides.iter().flatten() {
                v.push(s.to_bits());
            }
            v.push(ic.strides.is_some() as u32);
            v
        })
        .collect()
}

/// Candidate-first prediction until it is answered from a plan, every call's candidates checked against `want`.
#[track_caller]
fn cands_until_replayed(m: &FoloModel, x: &Tensor, nms: &NmsConfig, want: &[Vec<u32>]) {
    let before = m.plan_replays();
    for call in 0..3 {
        assert_eq!(cand_bits(&m.predict_candidates(x, nms).unwrap()), want, "call {call}");
    }
    assert!(m.plan_replays() > before, "no replay after three calls: {:?}", m.plan_refusal());
}

/// A threshold and no candidate cap.
fn low_conf(conf: f32, fusion: BoxFusion) -> NmsConfig {
    NmsConfig { conf, fusion, max_candidates: usize::MAX, ..NmsConfig::default() }
}

/// `m` with zero class biases, so its scores spread around 0.5 (a fresh model's prior keeps them near 0.01).
fn scored(m: FoloModel) -> FoloModel {
    let _g = no_grad();
    for t in m.named_tensors() {
        if t.name.contains("cls") && t.name.ends_with(".out.bias") {
            t.tensor.copy_from(&t.tensor.zeros_like().unwrap()).unwrap();
        }
    }
    m
}

#[test]
fn candidate_first_replays_are_bit_identical_with_and_without_dfl_precisions() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let tc = ctx.tensor_cores();
    let mut configs = vec![(DType::F32, false)];
    if f16_unavailable(dev).is_none() {
        configs.push((DType::F16, false));
        if ctx.can_enable_tensor_cores() {
            configs.push((DType::F16, true));
        }
    }
    for (dtype, cores) in configs {
        ctx.set_tensor_cores(cores);
        let (mut m, mut r) = (scored(model(dev, 4, 0)), scored(bare(dev, 4, 0)));
        m.to_dtype(dtype).unwrap();
        r.to_dtype(dtype).unwrap();
        for dims in [[1, 3, 64, 96], [2, 3, 96, 64]] {
            let x = input(dev, dims, dtype, 12);
            for (fusion, score) in [
                (BoxFusion::None, ScoreRule::None),
                (BoxFusion::Dfl, ScoreRule::None),
                (BoxFusion::None, ScoreRule::DflRepulse),
                (BoxFusion::Dfl, ScoreRule::DflRepulse),
                (BoxFusion::Sebf, ScoreRule::DflCgr),
            ] {
                let nms = NmsConfig { score, ..low_conf(0.3, fusion) };
                let want = cand_bits(&r.predict_candidates(&x, &nms).unwrap());
                assert!(
                    want.iter().all(|v| v.len() > 5),
                    "{dtype:?} {dims:?} {fusion:?} {score:?}: no candidates to compare"
                );
                cands_until_replayed(&m, &x, &nms, &want);
            }
        }
        assert_eq!(r.plan_replays(), 0);
    }
    ctx.set_tensor_cores(tc);
}

#[test]
fn a_threshold_change_gets_its_own_candidate_plan() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let (m, r) = (scored(model(dev, 4, 0)), scored(bare(dev, 4, 0)));
    let x = input(dev, [1, 3, 64, 64], DType::F32, 13);
    let (a, b) = (low_conf(0.45, BoxFusion::None), low_conf(0.5, BoxFusion::None));
    let (want_a, want_b) = (
        cand_bits(&r.predict_candidates(&x, &a).unwrap()),
        cand_bits(&r.predict_candidates(&x, &b).unwrap()),
    );
    assert_ne!(
        want_a, want_b,
        "the two thresholds keep the same candidates: the test cannot tell them apart"
    );
    cands_until_replayed(&m, &x, &a, &want_a);
    // The threshold is in a parameter block: another value is not answered by the first plan.
    let before = m.plan_replays();
    assert_eq!(cand_bits(&m.predict_candidates(&x, &b).unwrap()), want_b);
    assert_eq!(m.plan_replays(), before, "the plan of another threshold answered");
    cands_until_replayed(&m, &x, &b, &want_b);
    // Both plans are kept.
    let before = m.plan_replays();
    assert_eq!(cand_bits(&m.predict_candidates(&x, &a).unwrap()), want_a);
    assert_eq!(m.plan_replays(), before + 1);
}

#[test]
fn an_image_over_its_slots_is_answered_by_the_dense_head_after_a_replay() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some((_ctx, dev)) = gpu() else { return };
    let _g = no_grad();
    let (m, r) = (model(dev, 4, 0), bare(dev, 4, 0));
    // 320×320: 1600 P3 anchors, every one a candidate at `conf = 0`, over the 1024 slots.
    let x = input(dev, [1, 3, 320, 320], DType::F32, 14);
    for fusion in [BoxFusion::None, BoxFusion::Dfl] {
        let nms = low_conf(0.0, fusion);
        let want = cand_bits(&r.predict_candidates(&x, &nms).unwrap());
        let got = r.predict_candidates(&x, &nms).unwrap();
        assert!(got[0].cands.iter().filter(|c| c.anchor < 1600).count() > 1024, "the image fits its slots");
        cands_until_replayed(&m, &x, &nms, &want);
    }
}

/// DFL fusion's cached bins follow the context serving a device: a context registered again at the same ordinal
/// (`folo_gpu::register`) gets its own, so a model there predicts as on the first one (a private ordinal, so the
/// shared context the other tests use is untouched).
#[test]
fn dfl_bins_follow_a_context_registered_again_at_the_same_device() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some(_) = gpu() else { return };
    let _g = no_grad();
    let dev = Device::Gpu(7);
    let nms = low_conf(0.3, BoxFusion::Dfl);
    let mut first = None;
    for round in 0..2 {
        folo_gpu::register(GpuContext::new().unwrap(), dev).unwrap();
        let m = scored(model(dev, 4, 0));
        let x = input(dev, [1, 3, 64, 96], DType::F32, 12);
        let got = cand_bits(&m.predict_candidates(&x, &nms).unwrap());
        assert!(got[0].len() > 5, "round {round}: no candidates to compare");
        match &first {
            None => first = Some(got),
            Some(want) => assert_eq!(&got, want, "the second context's candidates"),
        }
    }
}
