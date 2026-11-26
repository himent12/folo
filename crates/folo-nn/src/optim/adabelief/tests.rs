//! Deterministic unit tests for [`AdaBelief`].

use std::collections::HashMap;

use folo_core::Tensor;
use folo_core::rng::Rng;

use super::*;
use crate::optim::GradSource;

struct MapGrads(Vec<(Tensor, Tensor)>);

impl GradSource for MapGrads {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.0.iter().find(|(k, _)| k.id() == param.id()).map(|(_, g)| g)
    }
}

fn check_adabelief_reference(decoupled_wd: bool, amsgrad: bool) {
    let w_init = [1.2f32, -0.7, 0.5, 2.1];
    let g1 = [0.4f32, -0.3, 0.8, -0.2];
    let g2 = [-0.5f32, -0.25, 0.1, 0.6];
    let lr = 0.02;
    let b1 = 0.9;
    let b2 = 0.999;
    let eps = 1e-8;
    let wd = 0.05;

    let p = Tensor::from_slice(&w_init, [4]).unwrap().requires_grad_();
    let cfg = AdaBeliefConfig { lr, betas: [b1, b2], eps, weight_decay: wd, decoupled_wd, amsgrad };
    let mut opt = AdaBelief::from_params(vec![("w".into(), p.clone())], cfg).unwrap();

    let mut w_ref: [f64; 4] = w_init.map(|v| v as f64);
    let mut m_ref = [0.0f64; 4];
    let mut s_ref = [0.0f64; 4];
    let mut max_s_ref = [0.0f64; 4];

    for (t, g_arr) in [g1, g2].into_iter().enumerate() {
        let step = (t + 1) as i32;
        opt.step(&MapGrads(vec![(p.clone(), Tensor::from_slice(&g_arr, [4]).unwrap())])).unwrap();

        let bc1 = 1.0 - b1.powi(step);
        let bc2 = 1.0 - b2.powi(step);
        for i in 0..4 {
            let g_eff = if decoupled_wd { g_arr[i] as f64 } else { g_arr[i] as f64 + wd * w_ref[i] };
            m_ref[i] = b1 * m_ref[i] + (1.0 - b1) * g_eff;
            let diff = g_eff - m_ref[i];
            s_ref[i] = b2 * s_ref[i] + (1.0 - b2) * diff * diff + eps;
            let s_hat = s_ref[i] / bc2;
            let denom_var = if amsgrad {
                max_s_ref[i] = max_s_ref[i].max(s_hat);
                max_s_ref[i]
            } else {
                s_hat
            };
            let p_base = if decoupled_wd { w_ref[i] * (1.0 - lr * wd) } else { w_ref[i] };
            w_ref[i] = p_base - (lr / bc1) * m_ref[i] / (denom_var.sqrt() + eps);

            // Match f32 storage rounding between steps.
            w_ref[i] = (w_ref[i] as f32) as f64;
            m_ref[i] = (m_ref[i] as f32) as f64;
            s_ref[i] = (s_ref[i] as f32) as f64;
            max_s_ref[i] = (max_s_ref[i] as f32) as f64;
        }
        let actual = p.to_vec::<f32>().unwrap();
        for i in 0..4 {
            assert!(
                (actual[i] as f64 - w_ref[i]).abs() < 1e-6,
                "decoupled={decoupled_wd} amsgrad={amsgrad} step {step} idx {i}: got {}, want {}",
                actual[i],
                w_ref[i]
            );
        }
    }
    assert_eq!(opt.step_count(), 2);
    assert_eq!(opt.param_step(0, 0), Some(2));
}

#[test]
fn adabelief_matches_hand_computed_reference_all_modes() {
    check_adabelief_reference(true, false);
    check_adabelief_reference(false, false);
    check_adabelief_reference(true, true);
    check_adabelief_reference(false, true);
}

#[test]
fn adabelief_takes_larger_steps_for_consistent_vs_noisy_gradients() {
    // Parameter A sees a steady gradient +1.0; Parameter B sees alternating +2.0 / 0.0 (same mean +1.0
    // but large prediction surprise (g - m)^2). AdaBelief should assign much smaller variance s_t and
    // therefore make larger progress on A than on B.
    let pa = Tensor::from_slice(&[0.0f32], [1]).unwrap().requires_grad_();
    let pb = Tensor::from_slice(&[0.0f32], [1]).unwrap().requires_grad_();
    let cfg =
        AdaBeliefConfig { lr: 0.01, betas: [0.9, 0.9], eps: 1e-8, weight_decay: 0.0, ..Default::default() };
    let mut opt =
        AdaBelief::from_params(vec![("a".into(), pa.clone()), ("b".into(), pb.clone())], cfg).unwrap();

    for step in 0..40 {
        let ga = Tensor::from_slice(&[1.0f32], [1]).unwrap();
        let gb_val = if step % 2 == 0 { 2.0f32 } else { 0.0f32 };
        let gb = Tensor::from_slice(&[gb_val], [1]).unwrap();
        opt.step(&MapGrads(vec![(pa.clone(), ga), (pb.clone(), gb)])).unwrap();
    }

    let sa = opt.moments(0, 0).unwrap().1.to_vec::<f32>().unwrap()[0];
    let sb = opt.moments(0, 1).unwrap().1.to_vec::<f32>().unwrap()[0];
    let da = pa.to_vec::<f32>().unwrap()[0].abs();
    let db = pb.to_vec::<f32>().unwrap()[0].abs();
    assert!(sa < sb * 0.1, "expected consistent grad variance {sa} << oscillating {sb}");
    assert!(da > db * 2.0, "expected consistent param displacement {da} >> noisy {db}");
}

#[test]
fn adabelief_skips_non_finite_grads_and_converges() {
    let mut rng = Rng::seed(7);
    let target = Tensor::from_slice(&[-1.5f32, 2.0, 0.5, -0.75], [4]).unwrap();
    let w = Tensor::randn([4], 0.0, 1.0, &mut rng).requires_grad_();
    let init_loss: f32 = w.sub(&target).unwrap().square().unwrap().mean_all().unwrap().item().unwrap();

    let mut opt = AdaBelief::from_params(
        vec![("w".into(), w.clone())],
        AdaBeliefConfig { lr: 0.1, weight_decay: 0.0, ..Default::default() },
    )
    .unwrap();

    // Inject a NaN gradient first and verify state & weights remain untouched.
    let before = w.to_vec::<f32>().unwrap();
    opt.step(&MapGrads(vec![(w.clone(), Tensor::from_slice(&[f32::NAN, 1.0, 0.0, 0.0], [4]).unwrap())]))
        .unwrap();
    assert_eq!(w.to_vec::<f32>().unwrap(), before);
    assert_eq!(opt.param_step(0, 0), Some(0));

    for _ in 0..80 {
        let loss = w.sub(&target).unwrap().square().unwrap().mean_all().unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }
    let final_loss: f32 = w.sub(&target).unwrap().square().unwrap().mean_all().unwrap().item().unwrap();
    assert!(
        final_loss < init_loss * 1e-3 && final_loss < 1e-3,
        "AdaBelief failed to converge: init={init_loss} final={final_loss}"
    );
}

#[test]
fn adabelief_checkpoint_contract_and_amsgrad_bit_exact_resume() {
    let make_pair = || {
        let w1 = Tensor::from_slice(&[0.8f32, -1.2, 0.3], [3]).unwrap().requires_grad_();
        let w2 = Tensor::from_slice(&[1.5f32, -0.4], [2]).unwrap().requires_grad_();
        let g1 = ParamGroup::new("decay", vec![("w1".into(), w1.clone())])
            .lr(0.01)
            .momentum(0.9)
            .adam(0.99, 1e-8)
            .weight_decay(0.02)
            .decoupled_wd(true);
        let g2 = ParamGroup::new("bias", vec![("w2".into(), w2.clone())])
            .lr(0.005)
            .momentum(0.85)
            .adam(0.999, 1e-8)
            .weight_decay(0.0)
            .decoupled_wd(false);
        let opt = AdaBelief::new(vec![g1, g2], true).unwrap();
        (w1, w2, opt)
    };

    let (w1_a, w2_a, mut opt_a) = make_pair();
    let gs = [
        (vec![0.4f32, -0.6, 0.1], Some(vec![0.2f32, -0.3])),
        (vec![-0.1f32, 0.8, -0.5], None),
        (vec![0.2f32, -0.2, 0.7], Some(vec![-0.5f32, 0.4])),
        (vec![-0.3f32, 0.1, -0.2], Some(vec![0.1f32, -0.2])),
    ];

    for (idx, (g1v, g2o)) in gs[..2].iter().enumerate() {
        if idx == 1 {
            opt_a.groups_mut()[0].lr = 0.008;
        }
        let mut entries = vec![(w1_a.clone(), Tensor::from_slice(g1v, [3]).unwrap())];
        if let Some(g2v) = g2o {
            entries.push((w2_a.clone(), Tensor::from_slice(g2v, [2]).unwrap()));
        }
        opt_a.step(&MapGrads(entries)).unwrap();
    }

    let contract_json = serde_json::to_string(&opt_a.contract()).unwrap();
    let contract: AdaBeliefContract = serde_json::from_str(&contract_json).unwrap();
    let state_map: HashMap<String, Tensor> = opt_a.state_tensors().unwrap().into_iter().collect();
    assert!(state_map.contains_key("w1.max_exp_avg_var"));

    let (w1_b, w2_b, mut opt_b) = make_pair();
    {
        let _ng = folo_core::no_grad();
        w1_b.copy_from(&w1_a).unwrap();
        w2_b.copy_from(&w2_a).unwrap();
    }
    assert!(opt_b.load_state(contract.step, &state_map).is_err());

    opt_b.apply_contract(&contract).unwrap();
    let mut bad_state = state_map.clone();
    bad_state.insert("w1.max_exp_avg_var".into(), Tensor::from_slice(&[-1.0f32, 0.0, 0.0], [3]).unwrap());
    assert!(opt_b.load_state(contract.step, &bad_state).is_err());

    let mut bad_inf = state_map.clone();
    bad_inf.insert("w1.exp_avg_var".into(), Tensor::from_slice(&[f32::INFINITY, 0.0, 0.0], [3]).unwrap());
    assert!(opt_b.load_state(contract.step, &bad_inf).is_err());

    opt_b.load_state(contract.step, &state_map).unwrap();
    for (g1v, g2o) in &gs[2..] {
        let mut ea = vec![(w1_a.clone(), Tensor::from_slice(g1v, [3]).unwrap())];
        let mut eb = vec![(w1_b.clone(), Tensor::from_slice(g1v, [3]).unwrap())];
        if let Some(g2v) = g2o {
            ea.push((w2_a.clone(), Tensor::from_slice(g2v, [2]).unwrap()));
            eb.push((w2_b.clone(), Tensor::from_slice(g2v, [2]).unwrap()));
        }
        opt_a.step(&MapGrads(ea)).unwrap();
        opt_b.step(&MapGrads(eb)).unwrap();
    }

    assert_eq!(
        w1_a.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        w1_b.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        w2_a.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        w2_b.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(opt_a.contract(), opt_b.contract());
}

#[test]
fn adabelief_rejects_invalid_configs() {
    let w = Tensor::zeros([2]).requires_grad_();
    for bad in [
        AdaBeliefConfig { lr: -1e-3, ..Default::default() },
        AdaBeliefConfig { betas: [1.0, 0.999], ..Default::default() },
        AdaBeliefConfig { betas: [0.9, 1.0], ..Default::default() },
        AdaBeliefConfig { eps: 0.0, ..Default::default() },
        AdaBeliefConfig { weight_decay: -0.1, ..Default::default() },
    ] {
        assert!(AdaBelief::from_params(vec![("w".into(), w.clone())], bad).is_err());
    }
}
