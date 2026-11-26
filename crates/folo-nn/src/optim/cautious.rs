//! Cautious update masking (Liang et al. 2024, "Cautious Optimizers: Improving Training with One Line of Code",
//! arXiv:2411.16085) for [`super::AdamW`] and [`super::Sgd`].
//!
//! Given a proposed update direction `u` (e.g. momentum / AdamW ratio) and the reference gradient `g` (the raw
//! gradient for `AdamW` and decoupled `Sgd`, or the L2-regularized gradient `d = g + wd * p` for coupled `Sgd`),
//! cautious masking zeroes coordinates whose update disagrees in sign with `g` and rescales active coordinates
//! by the inverse active fraction:
//! ```text
//! mask_i   = 1 if (u_i > 0 && g_i > 0) || (u_i < 0 && g_i < 0) else 0
//! active   = sum_i mask_i
//! alpha    = n / max(active, 1)
//! u_eff_i  = alpha * (u_i * mask_i)
//! ```
//! Because every active coordinate satisfies `u_i * g_i > 0`, the effective direction guarantees
//! `<u_eff, g> >= 0` (Lyapunov descent on the unregularized loss for `AdamW` / decoupled `Sgd`, and on the
//! L2-regularized objective for coupled `Sgd`) even when momentum overshoots, while preserving the exact
//! `0.0 * NaN = NaN` propagation contract of the tensor path.

use folo_core::{DType, Result, Tensor};

#[inline]
fn same_sign(u: f32, g: f32) -> bool {
    (u > 0.0 && g > 0.0) || (u < 0.0 && g < 0.0)
}

/// Generic (any-device / non-contiguous) cautious sign-agreement mask `mask_i in {0, 1}` and on-device rank-0
/// rescaling factor `n / max(active, 1)` with zero host readbacks (safe on browser WebGPU and native GPU).
pub(super) fn cautious_mask_and_scale_tensor(u: &Tensor, g: &Tensor) -> Result<(Tensor, Tensor)> {
    let dev = u.device();
    let n = u.numel();
    if n == 0 {
        return Ok((u.zeros_like()?, Tensor::full_with([], 1.0, DType::F32, dev)?));
    }
    let pos = u.gt_scalar(0.0)?.mul(&g.gt_scalar(0.0)?)?;
    let neg = u.lt_scalar(0.0)?.mul(&g.lt_scalar(0.0)?)?;
    let mask = pos.add(&neg)?;
    let one = Tensor::full_with([], 1.0, DType::F32, dev)?;
    let active = mask.sum_all()?.maximum(&one)?;
    let n_t = Tensor::full_with([], n as f64, DType::F32, dev)?;
    let scale = n_t.div(&active)?;
    Ok((mask, scale))
}

/// Zero-allocation CPU cautious AdamW step on contiguous `f32` slices, matching the generic tensor path
/// bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(super) fn adamw_cautious_slice(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    decay: Option<f32>,
    b1: f32,
    one_minus_b1: f32,
    b2: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2_sqrt: f32,
    eps: f32,
) {
    let n = p.len();
    assert_eq!(n, g.len());
    assert_eq!(n, m.len());
    assert_eq!(n, v.len());
    if n == 0 {
        return;
    }

    let mut active = 0usize;
    for ((&gi, mi), vi) in g.iter().zip(m.iter_mut()).zip(v.iter_mut()) {
        *mi = b1 * *mi + one_minus_b1 * gi;
        *vi = b2 * *vi + one_minus_b2 * (gi * gi);
        let denom = vi.sqrt() / bc2_sqrt + eps;
        let u = *mi / denom;
        if same_sign(u, gi) {
            active += 1;
        }
    }

    let scale = (n as f32) / (active.max(1) as f32);
    let neg_step = scale * (-step_size);
    for (((pi, &gi), &mi), &vi) in p.iter_mut().zip(g).zip(m.iter()).zip(v.iter()) {
        if let Some(d) = decay {
            *pi *= d;
        }
        let denom = vi.sqrt() / bc2_sqrt + eps;
        let u = mi / denom;
        let mask = if same_sign(u, gi) { 1.0f32 } else { 0.0f32 };
        *pi += (u * mask) * neg_step;
    }
}

/// Zero-allocation CPU cautious coupled-SGD step on contiguous `f32` slices, matching the generic tensor path
/// bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(super) fn sgd_coupled_cautious_slice(
    p: &mut [f32],
    g: &[f32],
    buf: Option<&mut [f32]>,
    lr: f32,
    mu: f32,
    wd: f32,
    nesterov: bool,
) {
    let n = p.len();
    assert_eq!(n, g.len());
    if n == 0 {
        return;
    }
    let has_wd = wd != 0.0;
    match buf {
        Some(b) if mu != 0.0 => {
            assert_eq!(n, b.len());
            let mut active = 0usize;
            for ((&pi, &gi), bi) in p.iter().zip(g).zip(b.iter_mut()) {
                let d = if has_wd { gi + wd * pi } else { gi };
                *bi = mu * *bi + d;
                let u = if nesterov { d + mu * *bi } else { *bi };
                if same_sign(u, d) {
                    active += 1;
                }
            }
            let scale = (n as f32) / (active.max(1) as f32);
            let neg_lr = scale * (-lr);
            for ((pi, &gi), &bi) in p.iter_mut().zip(g).zip(b.iter()) {
                let d = if has_wd { gi + wd * *pi } else { gi };
                let u = if nesterov { d + mu * bi } else { bi };
                let mask = if same_sign(u, d) { 1.0f32 } else { 0.0f32 };
                *pi += (u * mask) * neg_lr;
            }
        }
        _ => {
            let mut active = 0usize;
            for (&pi, &gi) in p.iter().zip(g) {
                let d = if has_wd { gi + wd * pi } else { gi };
                if same_sign(d, d) {
                    active += 1;
                }
            }
            let scale = (n as f32) / (active.max(1) as f32);
            let neg_lr = scale * (-lr);
            for (pi, &gi) in p.iter_mut().zip(g) {
                let d = if has_wd { gi + wd * *pi } else { gi };
                let mask = if same_sign(d, d) { 1.0f32 } else { 0.0f32 };
                *pi += (d * mask) * neg_lr;
            }
        }
    }
}

/// Zero-allocation CPU cautious decoupled-SGD step on contiguous `f32` slices, matching the generic tensor
/// path bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(super) fn sgd_decoupled_cautious_slice(
    p: &mut [f32],
    g: &[f32],
    buf: Option<&mut [f32]>,
    lr: f32,
    mu: f32,
    wd_step: f32,
    nesterov: bool,
    cautious_wd: bool,
) {
    let n = p.len();
    assert_eq!(n, g.len());
    if n == 0 {
        return;
    }
    let has_wd = wd_step != 0.0;
    match buf {
        Some(b) if mu != 0.0 => {
            assert_eq!(n, b.len());
            let mut active = 0usize;
            for (&gi, bi) in g.iter().zip(b.iter_mut()) {
                *bi = mu * *bi + gi;
                let u = if nesterov { gi + mu * *bi } else { *bi };
                if same_sign(u, gi) {
                    active += 1;
                }
            }
            let scale = (n as f32) / (active.max(1) as f32);
            let neg_lr = scale * (-lr);
            for ((pi, &gi), &bi) in p.iter_mut().zip(g).zip(b.iter()) {
                let u = if nesterov { gi + mu * bi } else { bi };
                let mask = if same_sign(u, gi) { 1.0f32 } else { 0.0f32 };
                let u_m = u * mask;
                let decay = if has_wd {
                    if cautious_wd {
                        if u_m * *pi > 0.0 { wd_step * *pi } else { 0.0 }
                    } else {
                        wd_step * *pi
                    }
                } else {
                    0.0
                };
                *pi += u_m * neg_lr;
                if has_wd {
                    *pi += -decay;
                }
            }
        }
        _ => {
            let mut active = 0usize;
            for &gi in g {
                if same_sign(gi, gi) {
                    active += 1;
                }
            }
            let scale = (n as f32) / (active.max(1) as f32);
            let neg_lr = scale * (-lr);
            for (pi, &gi) in p.iter_mut().zip(g) {
                let mask = if same_sign(gi, gi) { 1.0f32 } else { 0.0f32 };
                let u_m = gi * mask;
                let decay = if has_wd {
                    if cautious_wd {
                        if u_m * *pi > 0.0 { wd_step * *pi } else { 0.0 }
                    } else {
                        wd_step * *pi
                    }
                } else {
                    0.0
                };
                *pi += u_m * neg_lr;
                if has_wd {
                    *pi += -decay;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use folo_core::{Grads, Tensor, no_grad};

    use crate::optim::{AdamW, MuSgd, OptimContract, OptimKind, Optimizer, ParamGroup, Sgd};

    fn grads_for(p: &Tensor, v: &[f32]) -> Grads {
        let mut g = Grads::default();
        g.insert(p, Tensor::from_slice(v, p.shape()).unwrap());
        g
    }

    #[test]
    fn cautious_adamw_exact_masking_rescaling_and_step1_identity() {
        let p_caut = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], [4]).unwrap().requires_grad_();
        let p_std = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], [4]).unwrap().requires_grad_();
        let g1 = [1.0f32, -1.0, 2.0, -0.5];
        // Step 2 flips sign on indices 1 and 3 while indices 0 and 2 agree -> active = 2, scale = 4/2 = 2.0.
        let g2 = [0.5f32, 0.2, 1.0, 0.1];

        let mut opt_caut = AdamW::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_caut.clone())])
                .lr(0.1)
                .momentum(0.9)
                .weight_decay(0.05)
                .cautious(true),
        ])
        .unwrap();
        let mut opt_std = AdamW::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_std.clone())]).lr(0.1).momentum(0.9).weight_decay(0.05),
        ])
        .unwrap();

        // Step 1: all non-zero grads match m = (1 - b1) * g -> active = 4, scale = 1.0 -> bit-identical to AdamW!
        opt_caut.step(&grads_for(&p_caut, &g1)).unwrap();
        opt_std.step(&grads_for(&p_std, &g1)).unwrap();
        assert_eq!(
            p_caut.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            p_std.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let p_before = p_caut.to_vec::<f32>().unwrap();
        let (m0, v0) = opt_caut.moments(0, 0).unwrap();
        let m_prev = m0.to_vec::<f32>().unwrap();
        let v_prev = v0.to_vec::<f32>().unwrap();

        // Step 2: compute exact hand reference.
        opt_caut.step(&grads_for(&p_caut, &g2)).unwrap();
        let p_after = p_caut.to_vec::<f32>().unwrap();

        let (b1, b2, eps, lr, wd) = (0.9f64, 0.999f64, 1e-8f64, 0.1f64, 0.05f64);
        let decay = (1.0 - lr * wd) as f32;
        let step_size = (lr / (1.0 - b1.powi(2))) as f32;
        let bc2_sqrt = (1.0 - b2.powi(2)).sqrt() as f32;
        let mut expected = [0.0f32; 4];
        for i in 0..4 {
            let m = b1 as f32 * m_prev[i] + (1.0 - b1) as f32 * g2[i];
            let v = b2 as f32 * v_prev[i] + (1.0 - b2) as f32 * (g2[i] * g2[i]);
            let u = m / (v.sqrt() / bc2_sqrt + eps as f32);
            let active = (u > 0.0 && g2[i] > 0.0) || (u < 0.0 && g2[i] < 0.0);
            let decayed = p_before[i] * decay;
            if i == 0 || i == 2 {
                assert!(active, "index {i} should be active");
                expected[i] = decayed - (step_size * 2.0) * u;
            } else {
                assert!(!active, "index {i} should be masked out");
                expected[i] = decayed;
            }
        }
        for i in 0..4 {
            assert!(
                (p_after[i] - expected[i]).abs() < 1e-6,
                "idx {i}: got {} vs {}",
                p_after[i],
                expected[i]
            );
        }
    }

    #[test]
    fn cautious_sgd_lyapunov_descent_coupled_l2_and_edge_cases() {
        // 1. Lyapunov descent: momentum strongly disagrees on 3 of 4 coords so standard SGD moves uphill
        // (<p_old - p_new, g> < 0), while Cautious SGD guarantees descent (<p_old - p_new, g> > 0).
        let p_caut = Tensor::from_slice(&[0.0f32; 4], [4]).unwrap().requires_grad_();
        let p_std = Tensor::from_slice(&[0.0f32; 4], [4]).unwrap().requires_grad_();
        let mut opt_caut = Sgd::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_caut.clone())])
                .lr(0.1)
                .momentum(0.9)
                .nesterov(false)
                .cautious(true),
        ])
        .unwrap();
        let mut opt_std = Sgd::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_std.clone())]).lr(0.1).momentum(0.9).nesterov(false),
        ])
        .unwrap();

        let g1 = [10.0f32, 10.0, 10.0, 1.0];
        let g2 = [-0.5f32, -0.5, -0.5, 1.0];
        opt_caut.step(&grads_for(&p_caut, &g1)).unwrap();
        opt_std.step(&grads_for(&p_std, &g1)).unwrap();
        let before_caut = p_caut.to_vec::<f32>().unwrap();
        let before_std = p_std.to_vec::<f32>().unwrap();
        opt_caut.step(&grads_for(&p_caut, &g2)).unwrap();
        opt_std.step(&grads_for(&p_std, &g2)).unwrap();
        let after_caut = p_caut.to_vec::<f32>().unwrap();
        let after_std = p_std.to_vec::<f32>().unwrap();

        let dot_std: f64 = (0..4).map(|i| (before_std[i] - after_std[i]) as f64 * g2[i] as f64).sum();
        let dot_caut: f64 = (0..4).map(|i| (before_caut[i] - after_caut[i]) as f64 * g2[i] as f64).sum();
        assert!(dot_std < 0.0, "standard SGD should overshoot uphill here, got {dot_std}");
        assert!(dot_caut > 0.0, "cautious SGD must guarantee descent, got {dot_caut}");
        for i in 0..3 {
            assert_eq!(after_caut[i], before_caut[i], "masked coord {i} must not move");
        }
        let buf = opt_caut.momentum_buffer(0, 0).unwrap().to_vec::<f32>().unwrap();
        assert!((buf[0] - 8.5).abs() < 1e-6);

        // 2. Coupled L2 SGD compares u against regularized gradient d = g + wd * p (even when raw g disagrees).
        // p = [10.0, 10.0], wd = 1.0, g = [-2.0, -2.0] => d = [+8.0, +8.0] > 0 while raw g < 0.
        // Step 1 sets b = +8.0, u = +8.0, which agrees with d > 0 -> both active, parameter decreases.
        let p_l2 = Tensor::from_slice(&[10.0f32, 10.0], [2]).unwrap().requires_grad_();
        let mut opt_l2 = Sgd::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_l2.clone())])
                .lr(0.1)
                .momentum(0.5)
                .nesterov(false)
                .weight_decay(1.0)
                .cautious(true),
        ])
        .unwrap();
        opt_l2.step(&grads_for(&p_l2, &[-2.0, -2.0])).unwrap();
        let v_l2 = p_l2.to_vec::<f32>().unwrap();
        assert!((v_l2[0] - 9.2).abs() < 1e-6 && (v_l2[1] - 9.2).abs() < 1e-6, "coupled L2: {v_l2:?}");

        // 3. Momentum == 0 (both coupled and decoupled without cautious_wd) exercises the nomom arms.
        for decoupled in [false, true] {
            let p_nm = Tensor::from_slice(&[2.0f32, -3.0], [2]).unwrap().requires_grad_();
            let grp = ParamGroup::new("w", vec![("w".into(), p_nm.clone())])
                .lr(0.1)
                .momentum(0.0)
                .weight_decay(0.05)
                .decoupled_wd(decoupled)
                .cautious(true);
            let mut opt_nm = Sgd::new(vec![grp]).unwrap();
            // Zero gradient on coord 1 -> active = 1, scale = 2.0 on coord 0.
            opt_nm.step(&grads_for(&p_nm, &[1.0, 0.0])).unwrap();
            let v = p_nm.to_vec::<f32>().unwrap();
            if decoupled {
                // coord 0: 2.0 - (0.1 * 2.0) * 1.0 - (0.1 * 0.05) * 2.0 = 1.79; coord 1: -3.0 - 0.005 * (-3.0) = -2.985
                assert!(
                    (v[0] - 1.79).abs() < 1e-6 && (v[1] - (-2.985)).abs() < 1e-6,
                    "decoupled nomom: {v:?}"
                );
            } else {
                // coupled: d = [1.1, -0.15], both non-zero -> active = 2, scale = 1.0 -> [1.89, -2.985]
                assert!((v[0] - 1.89).abs() < 1e-6 && (v[1] - (-2.985)).abs() < 1e-6, "coupled nomom: {v:?}");
            }
        }

        // 4. active == 0 (all coords reversed): no NaN, parameter untouched.
        let p_zero = Tensor::from_slice(&[2.0f32, -3.0], [2]).unwrap().requires_grad_();
        let mut opt_z = Sgd::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_zero.clone())])
                .lr(0.1)
                .momentum(0.9)
                .nesterov(false)
                .cautious(true),
        ])
        .unwrap();
        opt_z.step(&grads_for(&p_zero, &[10.0, -10.0])).unwrap();
        let mid = p_zero.to_vec::<f32>().unwrap();
        opt_z.step(&grads_for(&p_zero, &[-0.1, 0.1])).unwrap();
        assert_eq!(p_zero.to_vec::<f32>().unwrap(), mid, "all-disagree step must leave params unchanged");

        // 5. Subnormal tiny gradients (1e-25 * 1e-25 underflows f32 product to 0, but sign check stays exact).
        let p_tiny = Tensor::from_slice(&[1.0f32, 1.0], [2]).unwrap().requires_grad_();
        let mut opt_tiny = AdamW::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_tiny.clone())])
                .lr(0.1)
                .momentum(0.9)
                .adam(0.999, 1e-30)
                .cautious(true),
        ])
        .unwrap();
        opt_tiny.step(&grads_for(&p_tiny, &[1e-25, -1e-25])).unwrap();
        let v_tiny = p_tiny.to_vec::<f32>().unwrap();
        assert!(v_tiny[0] < 1.0 && v_tiny[1] > 1.0, "tiny gradients must stay active: {v_tiny:?}");

        // 6. NaN propagation is preserved (not masked away).
        let p_nan = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap().requires_grad_();
        let mut opt_nan =
            AdamW::new(vec![ParamGroup::new("w", vec![("w".into(), p_nan.clone())]).lr(0.1).cautious(true)])
                .unwrap();
        opt_nan.step(&grads_for(&p_nan, &[f32::NAN, 1.0])).unwrap();
        let vn = p_nan.to_vec::<f32>().unwrap();
        assert!(vn[0].is_nan() && vn[1].is_finite(), "NaN must propagate: {vn:?}");

        // 7. Empty [0] tensor succeeds cleanly.
        let p_empty = Tensor::zeros([0]).requires_grad_();
        let mut opt_e = AdamW::new(vec![
            ParamGroup::new("w", vec![("w".into(), p_empty.clone())]).lr(0.1).cautious(true),
        ])
        .unwrap();
        opt_e.step(&grads_for(&p_empty, &[])).unwrap();
    }

    #[test]
    fn cautious_fast_path_matches_generic_path_bitwise_and_contract_round_trips() {
        let vals: Vec<f32> = (0..6).map(|i| i as f32 * 0.3 - 0.7).collect();
        let g1: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 0.4).collect();
        let g2: Vec<f32> = vec![-0.1, 0.2, 0.0, -0.05, 0.3, -0.2];

        let make_pair = |v: &[f32]| {
            let fast = Tensor::from_slice(v, [2, 3]).unwrap().requires_grad_();
            let base = Tensor::from_slice(v, [3, 2]).unwrap();
            let base_t = base.t().unwrap();
            {
                let _g = no_grad();
                base_t.copy_from(&fast.detach()).unwrap();
            }
            let slow = base_t.requires_grad_();
            assert!(!slow.is_contiguous());
            (fast, slow)
        };

        // Test Cautious AdamW, Cautious coupled SGD (Nesterov), and Cautious decoupled + cautious_wd SGD.
        for mode in 0..3 {
            let (fast, slow) = make_pair(&vals);
            let mk = |t: &Tensor| {
                let g = ParamGroup::new("decay", vec![("w".into(), t.clone())])
                    .lr(0.05)
                    .momentum(0.9)
                    .weight_decay(0.02)
                    .cautious(true);
                if mode == 2 { g.decoupled_wd(true).cautious_wd(true) } else { g }
            };
            let (mut a, mut b): (Box<dyn Optimizer>, Box<dyn Optimizer>) = if mode == 0 {
                (
                    Box::new(AdamW::new(vec![mk(&fast)]).unwrap()),
                    Box::new(AdamW::new(vec![mk(&slow)]).unwrap()),
                )
            } else {
                (Box::new(Sgd::new(vec![mk(&fast)]).unwrap()), Box::new(Sgd::new(vec![mk(&slow)]).unwrap()))
            };
            for g in [&g1, &g2, &g1] {
                a.step(&grads_for(&fast, g)).unwrap();
                b.step(&grads_for(&slow, g)).unwrap();
                let (x, y) = (fast.to_vec::<f32>().unwrap(), slow.to_vec::<f32>().unwrap());
                assert_eq!(
                    x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "mode {mode} fast vs generic mismatch"
                );
            }
        }

        // Masked +Inf with cautious + cautious_wd must match bitwise between contiguous slice & strided tensor.
        let inf_vals = [f32::INFINITY, -f32::INFINITY, 1.0, -2.0, 0.5, 3.0];
        let inf_grad = [0.0f32, 0.0, 0.5, -0.5, 0.0, 1.0];
        let (fast_inf, slow_inf) = make_pair(&inf_vals);
        let mk_inf = |t: &Tensor| {
            ParamGroup::new("decay", vec![("w".into(), t.clone())])
                .lr(0.05)
                .momentum(0.0)
                .weight_decay(0.1)
                .decoupled_wd(true)
                .cautious_wd(true)
                .cautious(true)
        };
        let mut opt_fi = Sgd::new(vec![mk_inf(&fast_inf)]).unwrap();
        let mut opt_si = Sgd::new(vec![mk_inf(&slow_inf)]).unwrap();
        opt_fi.step(&grads_for(&fast_inf, &inf_grad)).unwrap();
        opt_si.step(&grads_for(&slow_inf, &inf_grad)).unwrap();
        let (xf, xs) = (fast_inf.to_vec::<f32>().unwrap(), slow_inf.to_vec::<f32>().unwrap());
        assert_eq!(
            xf.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            xs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "masked +Inf cautious_wd mismatch: {xf:?} vs {xs:?}"
        );
        assert!(xf[0].is_infinite() && xf[0].is_sign_positive());

        // Contract serde: default cautious=false omits key; cautious=true round-trips for AdamW and SGD;
        // MuSgd rejects cautious=true both in constructor and in apply_contract.
        let p = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap().requires_grad_();
        let opt_default = AdamW::new(vec![ParamGroup::new("w", vec![("w".into(), p.clone())])]).unwrap();
        let json_default = serde_json::to_string(&opt_default.contract()).unwrap();
        assert!(!json_default.contains("cautious"), "default contract must omit cautious: {json_default}");

        for kind in [OptimKind::AdamW, OptimKind::Sgd] {
            let grp = ParamGroup::new("w", vec![("w".into(), p.clone())]).cautious(true);
            let opt: Box<dyn Optimizer> = match kind {
                OptimKind::AdamW => Box::new(AdamW::new(vec![grp]).unwrap()),
                OptimKind::Sgd => Box::new(Sgd::new(vec![grp]).unwrap()),
                OptimKind::MuSgd => unreachable!(),
            };
            let json = serde_json::to_string(&opt.contract()).unwrap();
            assert!(json.contains("\"cautious\":true"), "{kind:?} contract json: {json}");
            let parsed: OptimContract = serde_json::from_str(&json).unwrap();
            let grp2 = ParamGroup::new("w", vec![("w".into(), p.clone())]);
            let mut opt2: Box<dyn Optimizer> = match kind {
                OptimKind::AdamW => Box::new(AdamW::new(vec![grp2]).unwrap()),
                OptimKind::Sgd => Box::new(Sgd::new(vec![grp2]).unwrap()),
                OptimKind::MuSgd => unreachable!(),
            };
            opt2.apply_contract(&parsed).unwrap();
            assert!(opt2.groups()[0].cautious, "{kind:?} apply_contract must restore cautious=true");
        }

        assert!(
            MuSgd::new(vec![ParamGroup::new("decay", vec![("w".into(), p.clone())]).cautious(true)]).is_err()
        );
        let mut musgd = MuSgd::new(vec![ParamGroup::new("decay", vec![("w".into(), p.clone())])]).unwrap();
        let mut bad_contract = musgd.contract();
        bad_contract.groups[0].cautious = true;
        assert!(musgd.apply_contract(&bad_contract).is_err());
    }
}
