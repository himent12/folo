//! Unit tests for [`Lookahead`]: exact k-step fast exploration & alpha slow interpolation math,
//! `sync_slow_to_fast` / `sync_fast_to_slow` / `with_slow_weights` (including error-safe restoration),
//! convergence with wrapped `Sgd` and `AdamW`, state serialization/resume round-trip, and validation.

use std::collections::HashMap;

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_test_support::TestDir;

use super::*;
use crate::io::{
    BestState, CheckpointParts, FileKind, Metadata, ModelConfig, ModelFile, RngState, save_checkpoint,
};
use crate::optim::{AdamW, Optimizer, ParamGroup, Sgd, StandardGroups};
use crate::{Conv2d, Conv2dConfig, DecayKind, Forward, LrSchedule};

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
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: 1e-50 }).is_err());
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: -0.2 }).is_err());
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: 1.1 }).is_err());
    assert!(Lookahead::new(make_sgd(), LookaheadConfig { k: 5, alpha: f64::NAN }).is_err());

    // Tied/aliased parameter storage and nested Lookahead must both be rejected.
    let tied_sgd =
        Sgd::new(vec![ParamGroup::new("t", vec![("a".into(), q.clone()), ("b".into(), q.clone())])]).unwrap();
    assert!(Lookahead::with_defaults(tied_sgd).is_err());
    let la_once = Lookahead::with_defaults(make_sgd()).unwrap();
    assert!(Lookahead::with_defaults(la_once).is_err());

    // `alpha == 1.0` copies fast weights directly without `(fast - slow)` cancellation:
    // slow = 1.0, step 1 moves fast to 0.0, step 2 moves fast to 1e-8 and syncs with alpha = 1.0.
    let p_end = Tensor::from_slice(&[1.0f32], [1]).unwrap().requires_grad_();
    let sgd_end =
        Sgd::new(vec![ParamGroup::new("e", vec![("e".into(), p_end.clone())]).lr(1.0).momentum(0.0)])
            .unwrap();
    let mut la_end = Lookahead::new(sgd_end, LookaheadConfig { k: 2, alpha: 1.0 }).unwrap();
    let g1 = Tensor::from_slice(&[1.0f32], [1]).unwrap();
    let g2 = Tensor::from_slice(&[-1e-8f32], [1]).unwrap();
    la_end.step(&p_end.mul(&g1).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
    la_end.step(&p_end.mul(&g2).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
    assert_eq!(p_end.to_vec::<f32>().unwrap(), vec![1e-8f32]);
    assert_eq!(la_end.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap(), vec![1e-8f32]);

    // Cross-parameter slow-weight aliasing via `groups_mut()` must be rejected.
    let a = Tensor::zeros([2]).requires_grad_();
    let b = Tensor::zeros([2]).requires_grad_();
    let sgd_ab =
        Sgd::new(vec![ParamGroup::new("ab", vec![("a".into(), a.clone()), ("b".into(), b.clone())])])
            .unwrap();
    let mut la_ab = Lookahead::with_defaults(sgd_ab).unwrap();
    let slow0 = la_ab.slow_weight(0, 0).unwrap().clone();
    la_ab.groups_mut()[0].params[1].1 = slow0;
    assert!(la_ab.sync_slow_to_fast().is_err());
}

/// End-to-end on-disk `.folo` checkpoint save & `ModelFile::resume` round-trip mid-cycle, including
/// preserving Lookahead phase when wrapping a pre-stepped inner optimizer, and rejecting mismatched
/// plain-vs-Lookahead resume.
#[test]
fn lookahead_folo_checkpoint_resume_round_trip() {
    let build =
        |seed: u64| Conv2d::new(Conv2dConfig::new(2, 2, 1).padding(0).bias(true), &mut Rng::seed(seed));
    let m1 = build(11);
    let mut inner1 = AdamW::new(StandardGroups::from_module(&m1).into_groups(0.05, 0.9, 0.0)).unwrap();
    let mut sched1 = LrSchedule::new(DecayKind::Linear, 1, 20, 0.05, 0.0, 0.0, 0.9).unwrap();

    // Advance inner optimizer 2 steps BEFORE wrapping in Lookahead so `fast_steps != step_count()`.
    let mut rng = Rng::seed(77);
    for _ in 0..2 {
        let x = Tensor::randn([1, 2, 3, 3], 0.0, 1.0, &mut rng);
        let grads = m1.forward(&x).unwrap().square().unwrap().mean_all().unwrap().backward().unwrap();
        inner1.step(&grads).unwrap();
        sched1.advance();
    }
    let cfg = LookaheadConfig { k: 4, alpha: 0.6 };
    let mut opt1 = Lookahead::new(inner1, cfg).unwrap();
    for _ in 0..3 {
        let x = Tensor::randn([1, 2, 3, 3], 0.0, 1.0, &mut rng);
        let grads = m1.forward(&x).unwrap().square().unwrap().mean_all().unwrap().backward().unwrap();
        opt1.step(&grads).unwrap();
        sched1.advance();
    }
    // At checkpoint: inner step = 5, Lookahead fast_steps = 3, slow_steps = 0 (1 step away from sync!).
    assert_eq!((opt1.step_count(), opt1.fast_step_count(), opt1.slow_step_count()), (5, 3, 0));

    let dir = TestDir::new("folo-nn-lookahead-ckpt");
    let path = dir.path().join("la.folo");
    let meta = Metadata::new(FileKind::Checkpoint, ModelConfig::new("n", 1).unwrap(), vec!["obj".into()], 32)
        .unwrap();
    save_checkpoint(
        &path,
        &meta,
        &CheckpointParts {
            model: &m1,
            ema: None,
            optimizer: &opt1,
            schedule: &sched1,
            rng: &RngState::default(),
            epoch: 1,
            best: BestState::default(),
            train_config: &serde_json::json!({}),
        },
    )
    .unwrap();

    let file = ModelFile::open(&path).unwrap();
    // Plain AdamW must reject resuming a Lookahead checkpoint.
    let m_plain = build(99);
    let mut plain_opt =
        AdamW::new(StandardGroups::from_module(&m_plain).into_groups(0.05, 0.9, 0.0)).unwrap();
    assert!(file.resume(&m_plain, &mut plain_opt, None, None).is_err());

    // Conversely, Lookahead<AdamW> must reject resuming a plain (non-Lookahead) checkpoint.
    let mut plain_contract = opt1.contract();
    plain_contract.lookahead = None;
    let m_tmp = build(99);
    let mut la_reject = Lookahead::new(
        AdamW::new(StandardGroups::from_module(&m_tmp).into_groups(0.05, 0.9, 0.0)).unwrap(),
        cfg,
    )
    .unwrap();
    assert!(la_reject.apply_contract(&plain_contract).is_err());

    // Lookahead<AdamW> resumes with exact phase (`fast_steps = 3, slow_steps = 0`) and bit-exact parity!
    let m2 = build(99);
    let mut opt2 = Lookahead::new(
        AdamW::new(StandardGroups::from_module(&m2).into_groups(0.01, 0.9, 0.0)).unwrap(),
        LookaheadConfig { k: 2, alpha: 0.2 }, // contract adopts saved (k=4, alpha=0.6)
    )
    .unwrap();
    file.resume(&m2, &mut opt2, None, None).unwrap();
    assert_eq!(opt2.config(), cfg);
    assert_eq!((opt2.step_count(), opt2.fast_step_count(), opt2.slow_step_count()), (5, 3, 0));

    // Step 1 more time (crosses fast_steps == 4 -> slow_steps == 1!) and 2 more times after: bit-exact!
    for _ in 0..3 {
        let x = Tensor::randn([1, 2, 3, 3], 0.0, 1.0, &mut rng);
        let g1 = m1.forward(&x).unwrap().square().unwrap().mean_all().unwrap().backward().unwrap();
        let g2 = m2.forward(&x).unwrap().square().unwrap().mean_all().unwrap().backward().unwrap();
        opt1.step(&g1).unwrap();
        opt2.step(&g2).unwrap();
    }
    assert_eq!((opt1.fast_step_count(), opt1.slow_step_count()), (6, 1));
    assert_eq!((opt2.fast_step_count(), opt2.slow_step_count()), (6, 1));
    assert_eq!(m1.weight.to_vec::<f32>().unwrap(), m2.weight.to_vec::<f32>().unwrap());
    assert_eq!(
        opt1.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap(),
        opt2.slow_weight(0, 0).unwrap().to_vec::<f32>().unwrap()
    );
}
