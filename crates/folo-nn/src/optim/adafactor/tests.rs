//! Unit tests for [`Adafactor`]: factored row/column second moments for 2-D and 4-D tensors, unfactored 1-D
//! fallback, RMS update clipping, relative step-size schedule (`warmup_init` on/off), optional `beta1`
//! momentum, hand-computed step verification, checkpoint resume parity, and validation.

use std::collections::HashMap;

use folo_core::Tensor;
use folo_core::rng::Rng;

use super::*;
use crate::{Conv2d, Conv2dConfig, Forward, Linear, Module};

#[test]
fn adafactor_factored_shapes_and_hand_computed_step() {
    // 2x3 matrix with known values & gradient: verify exact factored row/col moments, RMS clipping, and update.
    let p = Tensor::from_slice(&[1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5], [2, 3]).unwrap().requires_grad_();
    let cfg = AdafactorConfig {
        lr: Some(0.1),
        eps: [1e-30, 1e-3],
        clip_threshold: 1.0,
        decay_rate: -0.8,
        beta1: None,
        weight_decay: 0.0,
        scale_parameter: true,
        relative_step: false,
        warmup_init: false,
    };
    let mut opt = Adafactor::from_params(vec![("w".into(), p.clone())], cfg).unwrap();

    // Factored buffers have O(R + C) = 2 + 3 = 5 floats instead of 2*3 = 6.
    let (vr, vc) = opt.factored_moments(0, 0).unwrap();
    assert_eq!(vr.dims(), &[2]);
    assert_eq!(vc.dims(), &[3]);
    assert!(opt.unfactored_moment(0, 0).is_none());

    // Step 1 on loss = 0.5 * sum((w * [1, 2, 3, 4, 5, 6])^2) -> arbitrary non-uniform gradient.
    let scale = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let loss = p.mul(&scale).unwrap().sum_all().unwrap(); // g = [1, 2, 3, 4, 5, 6]
    let grads = loss.backward().unwrap();
    let p0 = p.to_vec::<f32>().unwrap();
    opt.step(&grads).unwrap();
    let p1 = p.to_vec::<f32>().unwrap();

    // Hand calculation for step 1 (beta2_1 = 0):
    let g = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let eps1 = 1e-30f64;
    let rm = [(1.0 + 4.0 + 9.0 + 3.0 * eps1) / 3.0, (16.0 + 25.0 + 36.0 + 3.0 * eps1) / 3.0];
    let cm =
        [(1.0 + 16.0 + 2.0 * eps1) / 2.0, (4.0 + 25.0 + 2.0 * eps1) / 2.0, (9.0 + 36.0 + 2.0 * eps1) / 2.0];
    let vr_mean = 0.5 * (rm[0] + rm[1]);
    let mut u = [0.0f64; 6];
    for r in 0..2 {
        for c in 0..3 {
            let v_hat = (rm[r] / vr_mean) * cm[c];
            u[r * 3 + c] = g[r * 3 + c] / v_hat.sqrt();
        }
    }
    let rms_u = (u.iter().map(|v| v * v).sum::<f64>() / 6.0).sqrt();
    let clip = (rms_u / 1.0).max(1.0);
    let rms_p = (p0.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / 6.0).sqrt().max(1e-3);
    let alpha = 0.1 * rms_p;
    for i in 0..6 {
        let want = (p0[i] as f64 - alpha * (u[i] / clip)) as f32;
        assert!((p1[i] - want).abs() < 1e-6, "elem {i}: got {} vs want {want}", p1[i]);
    }
}

#[test]
fn adafactor_4d_conv_convergence_relative_step_and_rms_clipping() {
    let mut rng = Rng::seed(704);
    let conv = Conv2d::new(Conv2dConfig::new(3, 6, 3).bias(true), &mut rng);
    let lin = Linear::new(6, 2, true, &mut rng);

    let mut params = conv.named_parameters();
    for (n, t) in lin.named_parameters() {
        params.push((format!("lin.{n}"), t));
    }

    // Verify 4D conv weight [6, 3, 3, 3] factors to [6] + [27] = 33 floats (vs 162 full).
    let cfg = AdafactorConfig {
        lr: Some(0.05),
        beta1: Some(0.9),
        weight_decay: 0.01,
        ..AdafactorConfig::default()
    };
    let mut opt = Adafactor::from_params(params, cfg).unwrap();
    let (vr, vc) = opt.factored_moments(0, 0).unwrap();
    assert_eq!((vr.dims(), vc.dims()), (&[6][..], &[27][..]));
    assert!(opt.unfactored_moment(0, 1).is_some()); // conv.bias [6]

    let x = Tensor::randn([2, 3, 4, 4], 0.0, 1.0, &mut rng);
    let target = Tensor::randn([2, 2], 0.0, 0.5, &mut rng);
    let loss_fn = || {
        let h = conv.forward(&x).unwrap().mean([2, 3], false).unwrap();
        lin.forward(&h).unwrap().sub(&target).unwrap().square().unwrap().mean_all().unwrap()
    };
    let l0 = loss_fn().item::<f32>().unwrap();
    for _ in 0..30 {
        let grads = loss_fn().backward().unwrap();
        opt.step(&grads).unwrap();
    }
    let l_end = loss_fn().item::<f32>().unwrap();
    assert!(l_end < 0.25 * l0, "Adafactor failed to converge: l0={l0} -> l_end={l_end}");

    // Relative-step mode (hyperparameter-free) also decreases loss cleanly.
    let w_rel = Tensor::from_slice(&[2.0f32, -3.0, 1.5, -0.5], [2, 2]).unwrap().requires_grad_();
    let mut opt_rel = Adafactor::from_params(
        vec![("w".into(), w_rel.clone())],
        AdafactorConfig { warmup_init: false, ..AdafactorConfig::relative_step() },
    )
    .unwrap();
    let r0 = w_rel.square().unwrap().mean_all().unwrap().item::<f32>().unwrap();
    for _ in 0..20 {
        let g = w_rel.square().unwrap().mean_all().unwrap().backward().unwrap();
        opt_rel.step(&g).unwrap();
    }
    let r1 = w_rel.square().unwrap().mean_all().unwrap().item::<f32>().unwrap();
    assert!(r1 < r0, "relative-step Adafactor should reduce quadratic loss: {r0} -> {r1}");

    // RMS clipping caps step size even under a massive spike gradient.
    let w_clip = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], [4]).unwrap().requires_grad_();
    let mut opt_clip = Adafactor::from_params(
        vec![("v".into(), w_clip.clone())],
        AdafactorConfig {
            lr: Some(0.1),
            clip_threshold: 0.25,
            scale_parameter: false,
            ..AdafactorConfig::default()
        },
    )
    .unwrap();
    let g_spike = w_clip.mul_scalar(1e6).unwrap().sum_all().unwrap().backward().unwrap();
    opt_clip.step(&g_spike).unwrap();
    // Max possible step per element at RMS(u) <= 0.25 with uniform g is 0.1 * 0.25 = 0.025.
    for v in w_clip.to_vec::<f32>().unwrap() {
        assert!((v - 0.975).abs() < 1e-5, "RMS clip did not bound spike step: got {v}");
    }
}

#[test]
fn adafactor_resume_parity_and_validation() {
    let mut rng = Rng::seed(705);
    let w1 = Tensor::randn([3, 4], 0.0, 1.0, &mut rng).requires_grad_();
    let b1 = Tensor::randn([3], 0.0, 1.0, &mut rng).requires_grad_();
    let w2 = Tensor::from_vec(w1.to_vec::<f32>().unwrap(), w1.shape().clone()).unwrap().requires_grad_();
    let b2 = Tensor::from_vec(b1.to_vec::<f32>().unwrap(), b1.shape().clone()).unwrap().requires_grad_();

    let cfg = AdafactorConfig { lr: Some(0.02), beta1: Some(0.9), ..AdafactorConfig::default() };
    let mut opt1 =
        Adafactor::from_params(vec![("w".into(), w1.clone()), ("b".into(), b1.clone())], cfg).unwrap();
    let mut opt2 =
        Adafactor::from_params(vec![("w".into(), w2.clone()), ("b".into(), b2.clone())], cfg).unwrap();

    for _ in 0..3 {
        let loss =
            w1.square().unwrap().sum_all().unwrap().add(&b1.square().unwrap().sum_all().unwrap()).unwrap();
        opt1.step(&loss.backward().unwrap()).unwrap();
    }

    let c_json = serde_json::to_string(&opt1.contract()).unwrap();
    let contract: AdafactorContract = serde_json::from_str(&c_json).unwrap();
    let state: HashMap<String, Tensor> = opt1.state_tensors().unwrap().into_iter().collect();
    {
        let _ng = folo_core::no_grad();
        w2.copy_from(&w1).unwrap();
        b2.copy_from(&b1).unwrap();
    }
    opt2.apply_contract(&contract).unwrap();
    opt2.load_state(contract.step, &state).unwrap();

    let l1 = w1.square().unwrap().sum_all().unwrap().add(&b1.square().unwrap().sum_all().unwrap()).unwrap();
    let l2 = w2.square().unwrap().sum_all().unwrap().add(&b2.square().unwrap().sum_all().unwrap()).unwrap();
    opt1.step(&l1.backward().unwrap()).unwrap();
    opt2.step(&l2.backward().unwrap()).unwrap();
    assert_eq!(w1.to_vec::<f32>().unwrap(), w2.to_vec::<f32>().unwrap());
    assert_eq!(b1.to_vec::<f32>().unwrap(), b2.to_vec::<f32>().unwrap());

    // Invalid configs rejected cleanly.
    assert!(
        AdafactorConfig { lr: None, relative_step: false, ..AdafactorConfig::default() }.validate().is_err()
    );
    assert!(
        AdafactorConfig { lr: Some(1e-3), relative_step: true, ..AdafactorConfig::default() }
            .validate()
            .is_err()
    );
    assert!(
        AdafactorConfig { warmup_init: true, relative_step: false, ..AdafactorConfig::default() }
            .validate()
            .is_err()
    );
    assert!(AdafactorConfig { decay_rate: 0.5, ..AdafactorConfig::default() }.validate().is_err());
    assert!(AdafactorConfig { clip_threshold: 0.0, ..AdafactorConfig::default() }.validate().is_err());
}

struct MapGrads(HashMap<folo_core::TensorId, Tensor>);

impl GradSource for MapGrads {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.0.get(&param.id())
    }
}

#[test]
fn adafactor_r1_regressions_spike_aliasing_reordered_resume_and_beta1_varying_lr() {
    // 1. Finite gradient spike 2e20 does not poison factored or unfactored state to Inf/NaN.
    let wf = Tensor::ones([2, 3]).requires_grad_();
    let bu = Tensor::ones([3]).requires_grad_();
    let mut opt = Adafactor::from_params(
        vec![("w".into(), wf.clone()), ("b".into(), bu.clone())],
        AdafactorConfig { lr: Some(0.01), scale_parameter: false, ..AdafactorConfig::default() },
    )
    .unwrap();
    let mut g_spike = HashMap::new();
    g_spike.insert(wf.id(), Tensor::from_slice(&[2e20f32; 6], [2, 3]).unwrap());
    g_spike.insert(bu.id(), Tensor::from_slice(&[2e20f32; 3], [3]).unwrap());
    opt.step(&MapGrads(g_spike)).unwrap();
    let mut g_norm = HashMap::new();
    g_norm.insert(wf.id(), Tensor::from_slice(&[1.0f32; 6], [2, 3]).unwrap());
    g_norm.insert(bu.id(), Tensor::from_slice(&[1.0f32; 3], [3]).unwrap());
    opt.step(&MapGrads(g_norm)).unwrap();
    assert!(wf.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert!(bu.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));

    // Non-finite gradient fails atomically before any write.
    let step_before = opt.step_count();
    let mut g_nan = HashMap::new();
    g_nan.insert(wf.id(), Tensor::from_slice(&[f32::NAN; 6], [2, 3]).unwrap());
    assert!(opt.step(&MapGrads(g_nan)).is_err());
    assert_eq!(opt.step_count(), step_before);

    // 2. Parameter-state aliasing rejected in preflight.
    let p_a = Tensor::ones([3]).requires_grad_();
    let mut opt_a = Adafactor::from_params(vec![("b".into(), p_a)], AdafactorConfig::default()).unwrap();
    let st_buf = opt_a.state_tensors().unwrap()[0].1.clone();
    opt_a.groups_mut()[0].params[0].1 = st_buf;
    assert!(opt_a.step(&MapGrads(HashMap::new())).is_err());

    // 3. Reordered parameter resume maps param_steps by name.
    let a1 = Tensor::ones([2]).requires_grad_();
    let b1 = Tensor::ones([2]).requires_grad_();
    let mut src = Adafactor::from_params(
        vec![("a".into(), a1.clone()), ("b".into(), b1.clone())],
        AdafactorConfig::default(),
    )
    .unwrap();
    src.step(&a1.square().unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
    let contract = src.contract();
    let state: HashMap<String, Tensor> = src.state_tensors().unwrap().into_iter().collect();
    let a2 = Tensor::ones([2]).requires_grad_();
    let b2 = Tensor::ones([2]).requires_grad_();
    let mut dst = Adafactor::from_params(
        vec![("b".into(), b2.clone()), ("a".into(), a2.clone())],
        AdafactorConfig::default(),
    )
    .unwrap();
    dst.apply_contract(&contract).unwrap();
    dst.load_state(contract.step, &state).unwrap();
    assert_eq!(dst.param_step(0, 0), Some(0)); // "b"
    assert_eq!(dst.param_step(0, 1), Some(1)); // "a"

    // 4. Optional beta1 scales by alpha_t BEFORE EMA (Transformers parity under varying LR).
    let p = Tensor::from_slice(&[10.0f32], [1]).unwrap().requires_grad_();
    let mut opt_b1 = Adafactor::from_params(
        vec![("p".into(), p.clone())],
        AdafactorConfig {
            lr: Some(0.1),
            beta1: Some(0.5),
            scale_parameter: false,
            weight_decay: 0.0,
            ..AdafactorConfig::default()
        },
    )
    .unwrap();
    let mut g1 = HashMap::new();
    g1.insert(p.id(), Tensor::from_slice(&[2.0f32], [1]).unwrap());
    let g1 = MapGrads(g1);
    // Step 1 (lr=0.1): u_clip = 1.0, scaled = 0.1, m1 = 0.5*0 + 0.5*0.1 = 0.05, p1 = 9.95
    opt_b1.step(&g1).unwrap();
    assert!((p.to_vec::<f32>().unwrap()[0] - 9.95).abs() < 1e-5);
    // Step 2 (lr=0.4): u_clip = 1.0, scaled = 0.4, m2 = 0.5*0.05 + 0.5*0.4 = 0.225, p2 = 9.725
    opt_b1.groups_mut()[0].lr = 0.4;
    opt_b1.step(&g1).unwrap();
    assert!((p.to_vec::<f32>().unwrap()[0] - 9.725).abs() < 1e-5);
}
