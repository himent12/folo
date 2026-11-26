//! Unit tests for [`Lookahead`]: exact k-step fast exploration & alpha slow interpolation math,
//! `sync_slow_to_fast` / `sync_fast_to_slow` / `with_slow_weights` (including error-safe restoration),
//! convergence with wrapped `Sgd` and `AdamW`, state serialization/resume round-trip, and validation.

use std::collections::HashMap;

use folo_core::Tensor;

use super::*;
use crate::optim::{AdamW, Optimizer, ParamGroup, Sgd};

/// Exact analytical verification of k-step fast exploration followed by alpha slow-weight interpolation.
#[test]
fn lookahead_exact_k_step_interpolation_and_sync_invariants() {
    // Plain SGD (lr=0.1, momentum=0, wd=0) with constant gradient g = [1.0, -2.0], Lookahead(k=3, alpha=0.5).
    // Initial theta_0 = phi_0 = [10.0, -4.0].
    // Fast steps 1..=3:
    //   step 1: theta = [9.9, -3.8], phi stays [10.0, -4.0]
    //   step 2: theta = [9.8, -3.6], phi stays [10.0, -4.0]
    //   step 3: fast theta = [9.7, -3.4] -> slow sync:
    //           phi_1 = phi_0 + 0.5 * (theta - phi_0) = [9.85, -3.7], theta <- phi_1!
    let p = Tensor::from_slice(&[10.0f32, -4.0], [2]).unwrap().requires_grad_();
    let sgd =
        Sgd::new(vec![ParamGroup::new("p", vec![("p".into(), p.clone())]).lr(0.1).momentum(0.0)]).unwrap();
    let mut opt = Lookahead::new(sgd, LookaheadConfig { k: 3, alpha: 0.5 }).unwrap();
    let g_const = Tensor::from_slice(&[1.0f32, -2.0], [2]).unwrap();

    // Step 1: fast moves, slow unchanged.
    let grads = p.mul(&g_const).unwrap().sum_all().unwrap().backward().unwrap();
    opt.step(&grads).unwrap();
    assert_eq!((opt.fast_step_count(), opt.slow_step_count()), (1, 0));
    let pv1 = p.to_vec::<f32>().unwrap();
    assert!((pv1[0] - 9.9).abs() < 1e-6 && (pv1[1] + 3.8).abs() < 1e-6);
    assert_eq!(opt.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap(), vec![10.0, -4.0]);

    // Step 2: `with_slow_weights` temporarily exposes slow weights and restores fast weights even on Err.
    let grads = p.mul(&g_const).unwrap().sum_all().unwrap().backward().unwrap();
    opt.step(&grads).unwrap();
    assert_eq!((opt.fast_step_count(), opt.slow_step_count()), (2, 0));
    let fast_before = p.to_vec::<f32>().unwrap();
    let seen_inside = opt.with_slow_weights(|| Ok(p.to_vec::<f32>().unwrap())).unwrap();
    assert_eq!(seen_inside, vec![10.0, -4.0]);
    assert_eq!(p.to_vec::<f32>().unwrap(), fast_before);

    // Error inside `with_slow_weights` must STILL restore fast weights!
    let err_res: Result<()> = opt.with_slow_weights(|| {
        assert_eq!(p.to_vec::<f32>().unwrap(), vec![10.0, -4.0]);
        Err(Error::Invalid("synthetic eval failure".into()))
    });
    assert!(err_res.is_err());
    assert_eq!(p.to_vec::<f32>().unwrap(), fast_before);

    // Step 3: triggers slow interpolation and syncs fast <- slow.
    let grads = p.mul(&g_const).unwrap().sum_all().unwrap().backward().unwrap();
    opt.step(&grads).unwrap();
    assert_eq!((opt.fast_step_count(), opt.slow_step_count()), (3, 1));
    let synced = p.to_vec::<f32>().unwrap();
    let slow = opt.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(synced, slow);
    assert!((synced[0] - 9.85).abs() < 1e-6 && (synced[1] + 3.7).abs() < 1e-6);

    // Step 4 followed by explicit `sync_slow_to_fast` and `sync_fast_to_slow`.
    let grads = p.mul(&g_const).unwrap().sum_all().unwrap().backward().unwrap();
    opt.step(&grads).unwrap();
    assert_ne!(p.to_vec::<f32>().unwrap(), slow);
    opt.sync_slow_to_fast().unwrap();
    assert_eq!(p.to_vec::<f32>().unwrap(), slow);

    let grads = p.mul(&g_const).unwrap().sum_all().unwrap().backward().unwrap();
    opt.step(&grads).unwrap();
    let fast5 = p.to_vec::<f32>().unwrap();
    opt.sync_fast_to_slow().unwrap();
    assert_eq!(opt.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap(), fast5);
}

/// Convergence of `Lookahead<AdamW>` on a quadratic objective, bit-exact `state_tensors` / `load_state`
/// checkpoint resume mid-cycle, and config validation.
#[test]
fn lookahead_convergence_resume_and_validation() {
    let target = Tensor::from_slice(&[3.0f32, -2.0, 1.5, -0.5], [4]).unwrap();
    let x = Tensor::from_slice(&[-2.0f32, 4.0, -3.0, 2.5], [4]).unwrap().requires_grad_();
    let adamw = AdamW::new(vec![ParamGroup::new("x", vec![("x".into(), x.clone())]).lr(0.15)]).unwrap();
    let mut opt = Lookahead::new(adamw, LookaheadConfig { k: 5, alpha: 0.5 }).unwrap();

    for _ in 0..300 {
        let loss = x.sub(&target).unwrap().square().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }
    assert_eq!(opt.step_count(), 300);
    assert_eq!(opt.slow_step_count(), 60);
    let xv = x.to_vec::<f32>().unwrap();
    let tv = target.to_vec::<f32>().unwrap();
    for (a, b) in xv.iter().zip(&tv) {
        assert!((a - b).abs() < 1e-2, "Lookahead<AdamW> did not converge: got {xv:?} vs {tv:?}");
    }

    // Mid-cycle checkpoint save & bit-exact resume (step 3 of k=5).
    let p1 = Tensor::from_slice(&[1.0f32, -1.5, 2.0], [3]).unwrap().requires_grad_();
    let p2 = Tensor::from_slice(&[1.0f32, -1.5, 2.0], [3]).unwrap().requires_grad_();
    let cfg = LookaheadConfig { k: 5, alpha: 0.6 };
    let mut o1 = Lookahead::new(
        AdamW::new(vec![ParamGroup::new("p", vec![("p".into(), p1.clone())]).lr(0.05)]).unwrap(),
        cfg,
    )
    .unwrap();
    let mut o2 = Lookahead::new(
        AdamW::new(vec![ParamGroup::new("p", vec![("p".into(), p2.clone())]).lr(0.05)]).unwrap(),
        cfg,
    )
    .unwrap();

    for s in 0..3 {
        let g = Tensor::from_slice(&[0.4f32, -0.3, 0.2 * (s + 1) as f32], [3]).unwrap();
        o1.step(&p1.mul(&g).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
    }
    let state: HashMap<String, Tensor> = o1.state_tensors().unwrap().into_iter().collect();
    {
        let _ng = folo_core::no_grad();
        p2.copy_from(&p1.detach()).unwrap();
    }
    o2.load_state(o1.step_count(), &state).unwrap();
    assert_eq!(o2.fast_step_count(), 3);
    assert_eq!(o2.slow_step_count(), 0);

    // Run 4 more steps (crossing step 5 slow sync) -> bit-exact parity!
    for s in 3..7 {
        let g = Tensor::from_slice(&[-0.2f32, 0.5, -0.1 * s as f32], [3]).unwrap();
        o1.step(&p1.mul(&g).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
        o2.step(&p2.mul(&g).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
    }
    assert_eq!(p1.to_vec::<f32>().unwrap(), p2.to_vec::<f32>().unwrap());
    assert_eq!(
        o1.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap(),
        o2.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap()
    );

    // Config validation & serde round-trip.
    let json = serde_json::to_string(&cfg).unwrap();
    assert_eq!(serde_json::from_str::<LookaheadConfig>(&json).unwrap(), cfg);
    let q = Tensor::zeros([2]).requires_grad_();
    let make_sgd = || Sgd::new(vec![ParamGroup::new("q", vec![("q".into(), q.clone())])]).unwrap();
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 0, alpha: 0.5 }).is_err());
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: 0.0 }).is_err());
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: -0.2 }).is_err());
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: 1.1 }).is_err());
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: f64::NAN }).is_err());
}
