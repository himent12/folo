//! Unit tests for standalone [`Muon`]: Newton–Schulz orthogonality, aspect-ratio scaling modes, Nesterov vs
//! plain momentum, decoupled weight decay, 4-D conv weight flattening, AdamW and SGD 1-D fallback paths,
//! missing-gradient skip, checkpoint contract/state resume parity, and validation.

use std::collections::HashMap;

use folo_core::Tensor;
use folo_core::rng::Rng;

use super::*;
use crate::{Conv2d, Conv2dConfig, Forward, Linear, Module};

#[test]
fn newton_schulz_quintic_produces_near_orthogonal_rows_and_handles_tall_wide_4d() {
    let mut rng = Rng::seed(701);

    // Wide 4x8 matrix: O * O^T ≈ I_4 (classic 5-step quintic NS maps singular values into ~[0.68, 1.15],
    // so diagonal entries are near 1 and off-diagonals stay bounded; exact parity with `ortho::orthogonalize`).
    let wide = Tensor::randn([4, 8], 0.0, 1.0, &mut rng);
    let ow = newton_schulz_quintic(&wide, 5).unwrap();
    assert_eq!(ow.dims(), &[4, 8]);
    let ow_ref = crate::optim::ortho::orthogonalize(&wide).unwrap();
    for (a, b) in ow.to_vec::<f32>().unwrap().iter().zip(ow_ref.to_vec::<f32>().unwrap()) {
        assert!((a - b).abs() < 1e-5, "wide NS5 parity mismatch: {a} vs {b}");
    }
    let gram_w = ow.matmul(&ow.t().unwrap()).unwrap().to_vec::<f32>().unwrap();
    for r in 0..4 {
        for c in 0..4 {
            let v = gram_w[r * 4 + c];
            if r == c {
                assert!((0.5..=1.35).contains(&v), "wide gram[{r},{c}] diag out of range: {v}");
            } else {
                assert!(v.abs() < 0.25, "wide gram[{r},{c}] off-diag too large: {v}");
            }
        }
    }

    // Tall 8x4 matrix: exact parity with `ortho::orthogonalize` (undoing `muon_scale = sqrt(8/4)`)
    // and bounded Gram matrix `O^T * O` (NS5 maps singular values into ~[0.68, 1.15], so `O^T O` has
    // eigenvalues in ~[0.46, 1.32] and off-diagonals bounded by ~(1.32 - 0.46)/2 = 0.43).
    let tall = Tensor::randn([8, 4], 0.0, 1.0, &mut rng);
    let ot = newton_schulz_quintic(&tall, 5).unwrap();
    assert_eq!(ot.dims(), &[8, 4]);
    let ot_ref =
        crate::optim::ortho::orthogonalize(&tall).unwrap().div_scalar((8.0f32 / 4.0).sqrt()).unwrap();
    for (a, b) in ot.to_vec::<f32>().unwrap().iter().zip(ot_ref.to_vec::<f32>().unwrap()) {
        assert!((a - b).abs() < 1e-5, "tall NS5 parity mismatch: {a} vs {b}");
    }
    let gram_t = ot.t().unwrap().matmul(&ot).unwrap().to_vec::<f32>().unwrap();
    for r in 0..4 {
        for c in 0..4 {
            let v = gram_t[r * 4 + c];
            if r == c {
                assert!((0.45..=1.40).contains(&v), "tall gram[{r},{c}] diag out of range: {v}");
            } else {
                assert!(v.abs() < 0.45, "tall gram[{r},{c}] off-diag too large: {v}");
            }
        }
    }

    // 4-D conv weight [6, 3, 3, 3] flattens to [6, 27] and preserves shape.
    let conv_w = Tensor::randn([6, 3, 3, 3], 0.0, 1.0, &mut rng);
    let oc = newton_schulz_quintic(&conv_w, 5).unwrap();
    assert_eq!(oc.dims(), &[6, 3, 3, 3]);

    // Zero matrix stays exact zero without NaN.
    let z = Tensor::zeros([3, 5]);
    assert!(newton_schulz_quintic(&z, 5).unwrap().to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn muon_converges_mixed_matrix_conv_and_bias_and_matches_hand_step() {
    let mut rng = Rng::seed(702);
    let conv = Conv2d::new(Conv2dConfig::new(2, 4, 3).bias(true), &mut rng);
    let lin = Linear::new(4, 3, true, &mut rng);

    let mut params = conv.named_parameters();
    for (n, t) in lin.named_parameters() {
        params.push((format!("lin.{n}"), t));
    }
    let cfg = MuonConfig {
        lr: 0.05,
        momentum: 0.9,
        nesterov: true,
        weight_decay: 0.01,
        ns_steps: 5,
        ortho: None,
        aspect_scale: MuonAspectScale::SqrtRatio,
        fallback: MuonFallback::AdamW,
        aux_lr: Some(0.01),
        aux_betas: [0.9, 0.999],
        aux_eps: 1e-8,
        aux_weight_decay: Some(0.0),
    };
    let mut opt = Muon::from_params(params, cfg).unwrap();
    assert_eq!(opt.is_matrix_param(0, 0), Some(true)); // conv.weight [4, 2, 3, 3]
    assert_eq!(opt.is_matrix_param(0, 1), Some(false)); // conv.bias [4]
    assert_eq!(opt.is_matrix_param(0, 2), Some(true)); // lin.weight [3, 4]
    assert_eq!(opt.is_matrix_param(0, 3), Some(false)); // lin.bias [3]

    let x = Tensor::randn([2, 2, 4, 4], 0.0, 1.0, &mut rng);
    let target = Tensor::randn([2, 3], 0.0, 0.5, &mut rng);
    let loss_fn = || {
        let h = conv.forward(&x).unwrap().mean([2, 3], false).unwrap();
        let y = lin.forward(&h).unwrap();
        y.sub(&target).unwrap().square().unwrap().mean_all().unwrap()
    };

    let l0 = loss_fn().item::<f32>().unwrap();
    for _ in 0..25 {
        let loss = loss_fn();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }
    let l_end = loss_fn().item::<f32>().unwrap();
    assert!(l_end < 0.25 * l0, "Muon should strongly reduce loss: l0={l0} -> l_end={l_end}");
    assert_eq!(opt.step_count(), 25);
    assert_eq!(opt.param_step(0, 0), Some(25));

    // Verify Polar4 ortho schedule with AspectScale::None does NOT double-scale tall matrices.
    let tall_data = [1.0f32, 0.0, 0.0, 1.0, 0.5, -0.5, -0.5, 0.5];
    let tall_a = Tensor::from_slice(&tall_data, [4, 2]).unwrap().requires_grad_();
    let tall_b = Tensor::from_slice(&tall_data, [4, 2]).unwrap().requires_grad_();
    let mut opt_none = Muon::from_params(
        vec![("w".into(), tall_a.clone())],
        MuonConfig {
            lr: 0.1,
            momentum: 0.0,
            nesterov: false,
            weight_decay: 0.0,
            ortho: Some(MuonOrtho::Polar4),
            aspect_scale: MuonAspectScale::None,
            ..MuonConfig::default()
        },
    )
    .unwrap();
    let mut opt_ratio = Muon::from_params(
        vec![("w".into(), tall_b.clone())],
        MuonConfig {
            lr: 0.1,
            momentum: 0.0,
            nesterov: false,
            weight_decay: 0.0,
            ortho: Some(MuonOrtho::Polar4),
            aspect_scale: MuonAspectScale::SqrtRatio,
            ..MuonConfig::default()
        },
    )
    .unwrap();
    let g_a = tall_a.square().unwrap().sum_all().unwrap().backward().unwrap();
    let g_b = tall_b.square().unwrap().sum_all().unwrap().backward().unwrap();
    let before = tall_a.to_vec::<f32>().unwrap();
    opt_none.step(&g_a).unwrap();
    opt_ratio.step(&g_b).unwrap();
    let d_none: Vec<f32> = before.iter().zip(tall_a.to_vec::<f32>().unwrap()).map(|(a, b)| a - b).collect();
    let d_ratio: Vec<f32> = before.iter().zip(tall_b.to_vec::<f32>().unwrap()).map(|(a, b)| a - b).collect();
    let expected_factor = (4.0f32 / 2.0).sqrt();
    for (dn, dr) in d_none.iter().zip(&d_ratio) {
        assert!(
            (dr - dn * expected_factor).abs() < 1e-5,
            "aspect scaling mismatch: {dr} vs {}",
            dn * expected_factor
        );
    }
}

#[test]
fn muon_resume_parity_sgd_fallback_and_validation() {
    let mut rng = Rng::seed(703);
    let w1 = Tensor::randn([3, 4], 0.0, 1.0, &mut rng).requires_grad_();
    let b1 = Tensor::randn([3], 0.0, 1.0, &mut rng).requires_grad_();
    let w2 = Tensor::from_vec(w1.to_vec::<f32>().unwrap(), w1.shape().clone()).unwrap().requires_grad_();
    let b2 = Tensor::from_vec(b1.to_vec::<f32>().unwrap(), b1.shape().clone()).unwrap().requires_grad_();

    let cfg = MuonConfig { fallback: MuonFallback::Sgd, ..MuonConfig::default() };
    let mut opt1 = Muon::from_params(vec![("w".into(), w1.clone()), ("b".into(), b1.clone())], cfg).unwrap();
    let mut opt2 = Muon::from_params(vec![("w".into(), w2.clone()), ("b".into(), b2.clone())], cfg).unwrap();

    // Run 3 steps on opt1 (step 2 skips `b1` to test partial-gradient skipping).
    for step in 0..3 {
        let loss = if step == 1 {
            w1.square().unwrap().sum_all().unwrap()
        } else {
            w1.square().unwrap().sum_all().unwrap().add(&b1.square().unwrap().sum_all().unwrap()).unwrap()
        };
        opt1.step(&loss.backward().unwrap()).unwrap();
    }
    assert_eq!(opt1.param_step(0, 0), Some(3));
    assert_eq!(opt1.param_step(0, 1), Some(2));

    // Serialize contract + state tensors and resume into opt2.
    let contract_json = serde_json::to_string(&opt1.contract()).unwrap();
    let contract: MuonContract = serde_json::from_str(&contract_json).unwrap();
    let state_map: HashMap<String, Tensor> = opt1.state_tensors().unwrap().into_iter().collect();
    {
        let _ng = folo_core::no_grad();
        w2.copy_from(&w1).unwrap();
        b2.copy_from(&b1).unwrap();
    }
    opt2.apply_contract(&contract).unwrap();
    opt2.load_state(contract.step, &state_map).unwrap();

    // Subsequent step must match bit-for-bit.
    let l1 = w1.square().unwrap().sum_all().unwrap().add(&b1.square().unwrap().sum_all().unwrap()).unwrap();
    let l2 = w2.square().unwrap().sum_all().unwrap().add(&b2.square().unwrap().sum_all().unwrap()).unwrap();
    opt1.step(&l1.backward().unwrap()).unwrap();
    opt2.step(&l2.backward().unwrap()).unwrap();
    assert_eq!(w1.to_vec::<f32>().unwrap(), w2.to_vec::<f32>().unwrap());
    assert_eq!(b1.to_vec::<f32>().unwrap(), b2.to_vec::<f32>().unwrap());

    // Invalid configs rejected cleanly.
    assert!(MuonConfig { lr: -0.1, ..MuonConfig::default() }.validate().is_err());
    assert!(MuonConfig { momentum: 1.0, ..MuonConfig::default() }.validate().is_err());
    assert!(MuonConfig { ns_steps: 0, ortho: None, ..MuonConfig::default() }.validate().is_err());
}
