//! Optimizers, groups, accumulation, clipping, schedule application and EMA against hand-computed values.

use folo_core::rng::Rng;
use folo_core::{Grads, Tensor, no_grad};
use folo_nn::optim::*;
use folo_nn::*;

fn close(a: &[f32], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x as f64 - y).abs() <= tol * (1.0 + y.abs()), "element {i}: {x} vs {y}");
    }
}

/// Gradients for `p` from a backward pass of `sum(p * g)`, which is exactly `g`.
fn grads_for(p: &Tensor, g: &[f32]) -> Grads {
    let gt = Tensor::from_slice(g, p.shape().clone()).unwrap();
    p.mul(&gt).unwrap().sum_all().unwrap().backward().unwrap()
}

fn param(v: &[f32]) -> Tensor {
    Tensor::from_slice(v, [v.len()]).unwrap().requires_grad_()
}

const G1: [f32; 2] = [0.5, 0.25];
const G2: [f32; 2] = [-1.0, 2.0];

#[test]
fn sgd_nesterov_with_l2_weight_decay() {
    let p = param(&[1.0, -2.0]);
    let g = ParamGroup::new("decay", vec![("p".into(), p.clone())]).lr(0.1).momentum(0.9).weight_decay(0.01);
    let mut opt = Sgd::new(vec![g]).unwrap();
    opt.step(&grads_for(&p, &G1)).unwrap();
    close(&p.to_vec::<f32>().unwrap(), &[0.9031, -2.0437], 1e-6);
    close(&opt.momentum_buffer(0, 0).unwrap().to_vec::<f32>().unwrap(), &[0.51, 0.23], 1e-6);
    opt.step(&grads_for(&p, &G2)).unwrap();
    close(&p.to_vec::<f32>().unwrap(), &[1.05007411, -2.43844697], 1e-6);
    close(&opt.momentum_buffer(0, 0).unwrap().to_vec::<f32>().unwrap(), &[-0.531969, 2.186563], 1e-6);
    assert_eq!(opt.step_count(), 2);
}

#[test]
fn sgd_plain_momentum_and_missing_grads() {
    let (p, q) = (param(&[1.0, -2.0]), param(&[3.0]));
    let g = ParamGroup::new("g", vec![("p".into(), p.clone()), ("q".into(), q.clone())])
        .lr(0.1)
        .momentum(0.9)
        .nesterov(false);
    let mut opt = Sgd::new(vec![g]).unwrap();
    opt.step(&grads_for(&p, &G1)).unwrap();
    opt.step(&grads_for(&p, &G2)).unwrap();
    close(&p.to_vec::<f32>().unwrap(), &[1.005, -2.2475], 1e-6);
    // `q` never had a gradient: untouched, zero state (which is still written to checkpoints).
    assert_eq!(q.to_vec::<f32>().unwrap(), [3.0]);
    let st = opt.state_tensors().unwrap();
    assert_eq!(
        st.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        ["p.momentum_buffer", "q.momentum_buffer"]
    );
    assert_eq!(st[1].1.to_vec::<f32>().unwrap(), [0.0]);
}

#[test]
fn adamw_decoupled_decay_and_groups() {
    let (p, b) = (param(&[1.0, -2.0]), param(&[1.0, -2.0]));
    let groups = vec![
        ParamGroup::new("decay", vec![("w".into(), p.clone())]).lr(0.01).momentum(0.9).weight_decay(0.1),
        ParamGroup::new("bias", vec![("b".into(), b.clone())]).lr(0.01).momentum(0.9),
    ];
    let mut opt = AdamW::new(groups).unwrap();
    for g in [G1, G2] {
        let mut grads = grads_for(&p, &g);
        let gb = grads_for(&b, &g);
        grads.insert(&b, gb.get(&b).unwrap().clone());
        opt.step(&grads).unwrap();
    }
    close(&p.to_vec::<f32>().unwrap(), &[0.9916720354238566, -2.0142066571261106], 1e-6);
    close(&b.to_vec::<f32>().unwrap(), &[0.9936610354240566, -2.0182146571257107], 1e-6);
    let c = opt.contract();
    assert_eq!(c.kind, OptimKind::AdamW);
    assert_eq!(c.groups[0].betas, Some([0.9, 0.999]));
    assert_eq!(c.groups[0].momentum, None);
}

#[test]
fn generic_path_matches_fast_path_bitwise() {
    // A transposed (non-contiguous) parameter takes the op-by-op path; the result must be identical.
    let vals: Vec<f32> = (0..6).map(|i| i as f32 * 0.3 - 0.7).collect();
    let fast = Tensor::from_slice(&vals, [2, 3]).unwrap().requires_grad_();
    let base = Tensor::from_slice(&vals, [3, 2]).unwrap();
    let base_t = base.t().unwrap();
    {
        let _g = no_grad();
        base_t.copy_from(&fast.detach()).unwrap();
    }
    let slow = base_t.requires_grad_();
    assert!(!slow.is_contiguous());
    let gv: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 0.11).collect();
    for kind in [OptimKind::Sgd, OptimKind::AdamW] {
        let mk = |t: &Tensor| {
            ParamGroup::new("decay", vec![("w".into(), t.clone())]).lr(0.05).momentum(0.9).weight_decay(0.01)
        };
        let (mut a, mut b): (Box<dyn Optimizer>, Box<dyn Optimizer>) = match kind {
            OptimKind::Sgd => {
                (Box::new(Sgd::new(vec![mk(&fast)]).unwrap()), Box::new(Sgd::new(vec![mk(&slow)]).unwrap()))
            }
            OptimKind::AdamW => (
                Box::new(AdamW::new(vec![mk(&fast)]).unwrap()),
                Box::new(AdamW::new(vec![mk(&slow)]).unwrap()),
            ),
            OptimKind::MuSgd => unreachable!("not in the loop"),
        };
        for _ in 0..3 {
            a.step(&grads_for(&fast, &gv)).unwrap();
            b.step(&grads_for(&slow, &gv)).unwrap();
            let (x, y) = (fast.to_vec::<f32>().unwrap(), slow.to_vec::<f32>().unwrap());
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{kind:?}"
            );
        }
    }
}

#[test]
fn standard_groups_and_schedule_application() {
    let mut rng = Rng::seed(0);
    let model = Sequential::new()
        .push_named("stem", ConvBnAct::new(3, 4, 3, 1, 1, Activation::Silu, &mut rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, 2, 1), &mut rng));
    let sg = StandardGroups::from_module(&model);
    let names = |v: &[(String, Tensor)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(names(&sg.decay), ["stem.conv.weight", "out.weight"]);
    assert_eq!(names(&sg.bn), ["stem.bn.weight"]);
    assert_eq!(names(&sg.bias), ["stem.bn.bias", "out.bias"]);
    let groups = sg.into_groups(0.01, 0.937, 5e-4);
    assert_eq!(groups.iter().map(|g| g.weight_decay).collect::<Vec<_>>(), [5e-4, 0.0, 0.0]);
    let mut opt = Sgd::new(groups).unwrap();
    // 30 epochs of 13 steps: T = 390, W = 100 (docs/MODEL.md §10.4).
    let mut sched = LrSchedule::new(DecayKind::Linear, 30, 13, 0.01, 3.0, 0.1, 0.8).unwrap();
    sched.apply(&mut opt);
    let g = opt.groups();
    assert_eq!((g[0].lr, g[1].lr, g[2].lr), (0.0, 0.0, 0.1));
    assert_eq!(g[0].momentum, 0.8);
    for _ in 0..50 {
        sched.advance();
    }
    sched.apply(&mut opt);
    let target = 0.01 * ((1.0 - 50.0 / 389.0) * 0.99 + 0.01);
    assert!((opt.groups()[0].lr - target * 0.5).abs() < 1e-15);
    assert!((opt.groups()[2].lr - (0.1 + (target - 0.1) * 0.5)).abs() < 1e-15);
    assert!((opt.groups()[1].momentum - (0.8 + 0.137 * 0.5)).abs() < 1e-15);
    // Duplicate parameters are rejected.
    let p = param(&[1.0]);
    assert!(
        Sgd::new(vec![
            ParamGroup::new("a", vec![("p".into(), p.clone())]),
            ParamGroup::new("b", vec![("p".into(), p)])
        ])
        .is_err()
    );
}

#[test]
fn standard_groups_puts_v2_scale_and_gate_params_in_no_decay_bn_group() {
    struct V2Like {
        w3x3: Tensor,
        gamma_id: Tensor,
        alpha_raw: Tensor,
        gamma_id_extra: Tensor,
        bias: Tensor,
    }
    impl Module for V2Like {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("block.w3x3", &self.w3x3, TensorKind::Parameter);
            f("block.gamma_id", &self.gamma_id, TensorKind::Parameter);
            f("head.o2o_cls.0.suppress.alpha_raw", &self.alpha_raw, TensorKind::Parameter);
            f("block.gamma_id_extra", &self.gamma_id_extra, TensorKind::Parameter);
            f("head.bias", &self.bias, TensorKind::Parameter);
        }
    }
    let m = V2Like {
        w3x3: Tensor::ones([4, 4, 3, 3]).requires_grad_(),
        gamma_id: Tensor::ones([4]).requires_grad_(),
        alpha_raw: Tensor::full([3], -2.0).requires_grad_(),
        gamma_id_extra: Tensor::ones([4, 4]).requires_grad_(),
        bias: Tensor::zeros([3]).requires_grad_(),
    };
    let sg = StandardGroups::from_module(&m);
    let names = |v: &[(String, Tensor)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(names(&sg.bn), ["block.gamma_id", "head.o2o_cls.0.suppress.alpha_raw"]);
    assert_eq!(names(&sg.decay), ["block.w3x3", "block.gamma_id_extra"]);
    assert_eq!(names(&sg.bias), ["head.bias"]);
    // Top-level parameter names without dots match too.
    struct TopScale(Tensor, Tensor, Tensor);
    impl Module for TopScale {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("gamma_id", &self.0, TensorKind::Parameter);
            f("alpha_raw", &self.1, TensorKind::Parameter);
            // Rank-2 tensor with an exact scale name must keep weight decay.
            f("proj.gamma_id", &self.2, TensorKind::Parameter);
        }
    }
    let top = StandardGroups::from_module(&TopScale(
        Tensor::ones([4]).requires_grad_(),
        Tensor::ones([4]).requires_grad_(),
        Tensor::ones([4, 4]).requires_grad_(),
    ));
    assert_eq!(names(&top.bn), ["gamma_id", "alpha_raw"]);
    assert_eq!(names(&top.decay), ["proj.gamma_id"]);
    assert!(top.bias.is_empty());
}

#[test]
fn accumulation_and_clipping() {
    let (p, q) = (param(&[1.0, 2.0]), param(&[3.0]));
    let mut acc = GradAccumulator::new(&[p.clone(), q.clone()]);
    // One backward where p and q receive the same gradient tensor (shared storage): buffers must not alias.
    let x = Tensor::from_slice(&[1.0f32], [1]).unwrap();
    let y = p.sum_all().unwrap().add(&q.mul(&x).unwrap().sum_all().unwrap()).unwrap();
    acc.add(&y.backward().unwrap()).unwrap();
    acc.add(&grads_for(&p, &[3.0, 4.0])).unwrap();
    assert_eq!(acc.passes(), 2);
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [4.0, 5.0]);
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [1.0]);
    // ‖g‖ = sqrt(16 + 25 + 1) = sqrt(42); clipped to 1 by 1 / (n + 1e-6).
    let n = acc.clip_grad_norm(1.0).unwrap();
    assert!((n - 42f64.sqrt()).abs() < 1e-12);
    let c = (1.0 / (42f64.sqrt() + 1e-6)) as f32;
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [4.0 * c, 5.0 * c]);
    assert!((acc.global_norm().unwrap() - 1.0).abs() < 1e-5);
    assert_eq!(acc.clip_grad_norm(10.0).unwrap(), acc.global_norm().unwrap(), "below the limit: unchanged");
    acc.zero();
    assert!(!acc.has_grads() && acc.grad(&p).is_none());
    acc.add(&grads_for(&q, &[2.0])).unwrap();
    assert!(acc.grad(&p).is_none(), "a parameter without a gradient in this window has none");
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [2.0]);
}

#[test]
fn ema_values() {
    let live = Conv2d::from_tensors(
        Conv2dConfig::new(1, 1, 1),
        Tensor::full([1, 1, 1, 1], 2.0).requires_grad_(),
        Some(Tensor::zeros([1]).requires_grad_()),
    )
    .unwrap();
    let shadow = Conv2d::from_tensors(
        Conv2dConfig::new(1, 1, 1),
        Tensor::zeros([1, 1, 1, 1]),
        Some(Tensor::zeros([1])),
    )
    .unwrap();
    let mut ema = ModelEma::new(shadow, &live).unwrap();
    assert_eq!(ema.module().weight.to_vec::<f32>().unwrap(), [2.0]);
    assert!(!ema.module().weight.requires_grad());
    let d = |u: f64| 0.9999 * (1.0 - (-u / 2000.0).exp());
    assert!((d(1.0) - 4.99825e-4).abs() < 1e-9);
    let mut want = 2.0f64;
    for (u, v) in [(1.0, 4.0f32), (2.0, 6.0)] {
        {
            let _g = no_grad();
            live.weight.fill_assign(v).unwrap();
        }
        ema.update(&live).unwrap();
        want = d(u) * want + (1.0 - d(u)) * v as f64;
        close(&ema.module().weight.to_vec::<f32>().unwrap(), &[want], 1e-6);
    }
    assert_eq!(ema.updates(), 2);
    // BatchNorm running statistics are tracked too.
    let bn = BatchNorm2d::new(2);
    let mut e2 = ModelEma::new(BatchNorm2d::new(2), &bn).unwrap();
    bn.forward(&Tensor::full([2, 2, 2, 2], 3.0)).unwrap();
    e2.update(&bn).unwrap();
    let rm = 0.03 * 3.0;
    close(&e2.module().running_mean.to_vec::<f32>().unwrap(), &[(1.0 - d(1.0)) * rm; 2], 1e-6);
}

#[test]
fn accumulator_with_duplicate_handles() {
    let (p, q) = (param(&[1.0]), param(&[1.0]));
    let mut acc = GradAccumulator::new(&[p.clone(), p.clone(), q.clone()]);
    let mut grads = grads_for(&p, &[2.0]);
    grads.insert(&q, Tensor::from_slice(&[7.0f32], [1]).unwrap());
    acc.add(&grads).unwrap();
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [2.0]);
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [7.0]);
    // Only repeats: one slot, no out-of-range index.
    let mut acc = GradAccumulator::new(&[p.clone(), p.clone()]);
    acc.add(&grads_for(&p, &[3.0])).unwrap();
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [3.0]);
    assert!(acc.grad(&q).is_none());
}

/// Newton–Schulz / Polar-FOLO in `f64` on a row-major `rows x cols` matrix: the reference of `orthogonalize_with`.
fn ortho_f64(m: &[f64], rows: usize, cols: usize, ortho: MuonOrtho) -> Vec<f64> {
    let t = rows > cols;
    let (r, c) = if t { (cols, rows) } else { (rows, cols) };
    let at = |i: usize, j: usize| if t { m[j * cols + i] } else { m[i * cols + j] };
    let norm = m.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
    let mut x: Vec<f64> = (0..r * c).map(|k| at(k / c, k % c) / norm).collect();
    let mul = |p: &[f64], q: &[f64], n: usize, k: usize, m: usize| {
        let mut o = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                o[i * m + j] = (0..k).map(|l| p[i * k + l] * q[l * m + j]).sum();
            }
        }
        o
    };
    for (step, &(a, b, cc)) in ortho.schedule().iter().enumerate() {
        let (a, b, cc) = (a as f64, b as f64, cc as f64);
        let xt: Vec<f64> = (0..c * r).map(|k| x[(k % r) * c + k / r]).collect();
        let am = mul(&x, &xt, r, c, r);
        let aa = mul(&am, &am, r, r, r);
        if step == 0 && ortho.schatten4() {
            let a_frob = am.iter().map(|v| v * v).sum::<f64>().sqrt();
            let alpha = a_frob.sqrt() + 1e-7 / norm;
            let inv_a = if alpha == 0.0 { 0.0 } else { 1.0 / alpha };
            let inv_a2 = inv_a * inv_a;
            let inv_a4 = inv_a2 * inv_a2;
            let bm: Vec<f64> =
                am.iter().zip(&aa).map(|(p, q)| (b * inv_a2) * p + (cc * inv_a4) * q).collect();
            let bx = mul(&bm, &x, r, r, c);
            x = x.iter().zip(&bx).map(|(p, q)| (a * inv_a) * p + q * inv_a).collect();
        } else {
            let bm: Vec<f64> = am.iter().zip(&aa).map(|(p, q)| b * p + cc * q).collect();
            let bx = mul(&bm, &x, r, r, c);
            x = x.iter().zip(&bx).map(|(p, q)| a * p + q).collect();
        }
    }
    if t { (0..rows * cols).map(|k| x[(k % cols) * rows + k / cols]).collect() } else { x }
}

fn ns5_f64(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    ortho_f64(m, rows, cols, MuonOrtho::Ns5)
}

#[test]
fn polar_folo_matches_f64_and_improves_orthogonality() {
    let mut rng = Rng::seed(17);
    let shapes: [&[usize]; 4] = [&[3, 5], &[6, 2], &[4, 2, 1, 3], &[1, 6]];
    let ts: Vec<Tensor> = shapes.iter().map(|s| Tensor::randn(s.to_vec(), 0.0, 1.0, &mut rng)).collect();
    for ortho in [MuonOrtho::Polar4, MuonOrtho::Polar5] {
        let many = orthogonalize_many_with(&ts, ortho).unwrap();
        for (t, o) in ts.iter().zip(&many) {
            let d = t.dims().to_vec();
            let (rows, cols) = (d[0], t.numel() / d[0]);
            let v: Vec<f64> = t.to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect();
            let scale = muon_scale(&d) as f64;
            let want: Vec<f64> = ortho_f64(&v, rows, cols, ortho).iter().map(|x| x * scale).collect();
            close(&o.to_vec::<f32>().unwrap(), &want, 2e-4);
            // Alone, without bucket padding: Schatten-4 norm is invariant to zero-padded rows.
            let one = orthogonalize_with(t, ortho).unwrap().to_vec::<f32>().unwrap();
            close(&one, &o.to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect::<Vec<_>>(), 1e-5);
        }
    }
    // Check orthogonality error ||O O^T - I||_F / sqrt(r) on a well- and moderately-conditioned 16x64 matrix:
    // both Polar4 (12 GEMMs) and Polar5 (15 GEMMs) strictly beat NS5 (15 GEMMs).
    let g = Tensor::randn([16, 64], 0.0, 1.0, &mut rng);
    let ortho_err = |ortho: MuonOrtho| -> f64 {
        let o = orthogonalize_with(&g, ortho).unwrap();
        let oot = o.matmul(&o.t().unwrap().contiguous().unwrap()).unwrap().to_vec::<f32>().unwrap();
        let mut s = 0.0f64;
        for i in 0..16 {
            for j in 0..16 {
                let d = oot[i * 16 + j] as f64 - if i == j { 1.0 } else { 0.0 };
                s += d * d;
            }
        }
        (s / 16.0).sqrt()
    };
    let e_ns5 = ortho_err(MuonOrtho::Ns5);
    let e_p4 = ortho_err(MuonOrtho::Polar4);
    let e_p5 = ortho_err(MuonOrtho::Polar5);
    assert!(e_p4 < e_ns5 * 0.5, "Polar4 ({e_p4:.4}) should be >2x more orthogonal than NS5 ({e_ns5:.4})");
    assert!(e_p5 < e_ns5 * 0.5, "Polar5 ({e_p5:.4}) should be >2x more orthogonal than NS5 ({e_ns5:.4})");
}

/// Finite matrices whose unscaled squared Frobenius norm overflows `f32::MAX` (`1×1` `2e19`, `16×16` `1e19`,
/// `4×4` `1e38`): Polar-FOLO (`Polar4`, `Polar5`) uses power-of-two downscaling and returns the finite
/// orthogonalised direction matching the `f64` reference, legacy `Ns5` preserves its exact pre-Polar zero
/// direction, and all three propagate `NaN`.
#[test]
fn polar_folo_handles_frobenius_overflow_and_nan() {
    let huge_1x1 = Tensor::from_slice(&[2e19_f32], [1, 1]).unwrap();
    let huge_16x16 = Tensor::from_vec(vec![1e19_f32; 16 * 16], [16, 16]).unwrap();
    let huge_4x4 = Tensor::from_vec(vec![1e38_f32; 4 * 4], [4, 4]).unwrap();
    let mut nan_data = vec![0.25_f32; 4 * 4];
    nan_data[5] = f32::NAN;
    let with_nan = Tensor::from_vec(nan_data, [4, 4]).unwrap();
    let batch = [huge_1x1, huge_16x16, huge_4x4, with_nan];
    for ortho in [MuonOrtho::Ns5, MuonOrtho::Polar4, MuonOrtho::Polar5] {
        let many = orthogonalize_many_with(&batch, ortho).unwrap();
        for (idx, t) in batch.iter().enumerate() {
            let one = orthogonalize_with(t, ortho).unwrap().to_vec::<f32>().unwrap();
            let got = many[idx].to_vec::<f32>().unwrap();
            if idx == 3 {
                assert!(got.iter().all(|v| v.is_nan()), "{ortho:?} batch NaN must propagate");
                assert!(one.iter().all(|v| v.is_nan()), "{ortho:?} one NaN must propagate");
            } else if ortho == MuonOrtho::Ns5 {
                assert!(got.iter().all(|&v| v == 0.0), "legacy Ns5 [{idx}] must keep exact 0, got {got:?}");
                assert!(
                    one.iter().all(|&v| v == 0.0),
                    "legacy Ns5 one[{idx}] must keep exact 0, got {one:?}"
                );
            } else {
                let d = t.dims();
                let (rows, cols) = (d[0], t.numel() / d[0]);
                let v: Vec<f64> = t.to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect();
                let want = ortho_f64(&v, rows, cols, ortho);
                assert!(
                    got.iter().all(|v| v.is_finite()) && got.iter().any(|&v| v != 0.0),
                    "{ortho:?} [{idx}]"
                );
                close(&got, &want, 2e-4);
                close(&one, &want, 2e-4);
            }
        }
    }
}

#[test]
fn newton_schulz_matches_f64_and_batches_exactly() {
    let mut rng = Rng::seed(3);
    // Wide, tall (transposed internally), a conv weight, and one zero-padded in its bucket.
    let shapes: [&[usize]; 4] = [&[3, 5], &[6, 2], &[4, 2, 1, 3], &[1, 6]];
    let ts: Vec<Tensor> = shapes.iter().map(|s| Tensor::randn(s.to_vec(), 0.0, 1.0, &mut rng)).collect();
    let many = orthogonalize_many(&ts).unwrap();
    for (t, o) in ts.iter().zip(&many) {
        let d = t.dims().to_vec();
        let (rows, cols) = (d[0], t.numel() / d[0]);
        let v: Vec<f64> = t.to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect();
        assert_eq!(o.dims(), &d[..]);
        // Scaled by sqrt(max(1, d[-2] / d[-1])): sqrt(3) for the [6, 2] matrix, 1 for the others.
        let scale = muon_scale(&d) as f64;
        let want: Vec<f64> = ns5_f64(&v, rows, cols).iter().map(|x| x * scale).collect();
        close(&o.to_vec::<f32>().unwrap(), &want, 1e-4);
        // Alone, without bucket padding: the same numbers.
        let one = orthogonalize(t).unwrap().to_vec::<f32>().unwrap();
        close(&one, &o.to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect::<Vec<_>>(), 1e-5);
    }
    // The reference scales by sqrt(max(1, d[-2] / d[-1])): 1 for a square kernel, sqrt(3) for [.., 3, 1].
    assert_eq!(muon_scale(&[8, 4, 3, 3]), 1.0);
    assert_eq!(muon_scale(&[2, 3, 1]), 3f32.sqrt());
    assert_eq!(muon_scale(&[4, 6]), 1.0);
}

fn sgd_ref(p: &mut [f64], s: &mut [f64], g: &[f64], lr: f64, wd: f64) {
    let mu = 0.9;
    for ((p, s), g) in p.iter_mut().zip(s.iter_mut()).zip(g) {
        let d = g + wd * *p;
        *s = mu * *s + d;
        *p -= lr * (d + mu * *s);
    }
}

/// Two MuSGD steps against an `f64` reference (muon 0.2, sgd 1.0, Nesterov) across all three orthogonalisation
/// schedules (`Polar4` default, legacy `Ns5` matching Ultralytics 8.4, and `Polar5`): a Muon conv weight, a
/// boosted `.cls` weight at 3x the LR, and a BatchNorm weight that only takes SGD.
#[test]
fn musgd_matches_reference_steps() {
    for ortho in [MuonOrtho::Polar4, MuonOrtho::Ns5, MuonOrtho::Polar5] {
        musgd_reference_steps(MuSgdWeights::default(), ortho);
    }
}

/// The same with other step weights (the reference's `MuSGD(muon=0.5, sgd=0.5)`): the SGD factor scales only the
/// Muon groups' SGD step, and the groups' LRs are left as the schedule set them.
#[test]
fn musgd_weights_match_reference_steps() {
    musgd_reference_steps(MuSgdWeights { muon: 0.5, sgd: 0.5 }, MuonOrtho::Polar4);
    musgd_reference_steps(MuSgdWeights { muon: 0.5, sgd: 0.5 }, MuonOrtho::Ns5);
    let bad = MuSgdWeights { muon: f64::NAN, sgd: 1.0 };
    assert!(
        MuSgd::with_weights(vec![ParamGroup::new("decay", vec![("w".into(), param(&[1.0]))])], bad).is_err()
    );
}

fn musgd_reference_steps(weights: MuSgdWeights, ortho: MuonOrtho) {
    let w = Tensor::from_slice(&[0.3f32, -0.2, 0.5, 0.1, 0.4, -0.6], [2, 3, 1, 1]).unwrap().requires_grad_();
    let h = Tensor::from_slice(&[0.2f32, -0.1, 0.3, 0.7], [2, 2, 1, 1]).unwrap().requires_grad_();
    let bn = param(&[1.0, 0.9]);
    let (lr, mu, wd) = (0.05f64, 0.9f64, 0.01f64);
    let groups = vec![
        ParamGroup::new("decay", vec![("w".into(), w.clone())]).lr(lr).momentum(mu).weight_decay(wd),
        ParamGroup::new("decay.cls", vec![("h".into(), h.clone())])
            .lr(3.0 * lr)
            .momentum(mu)
            .weight_decay(wd),
        ParamGroup::new("bn", vec![("bn".into(), bn.clone())]).lr(lr).momentum(mu),
    ];
    let mut opt = if ortho == MuonOrtho::default() {
        MuSgd::with_weights(groups, weights).unwrap()
    } else {
        MuSgd::with_ortho(groups, weights, ortho).unwrap()
    };
    assert_eq!((opt.kind(), opt.weights(), opt.ortho()), (OptimKind::MuSgd, weights, ortho));
    let gw = [[0.1f32, -0.3, 0.2, 0.05, 0.4, -0.1], [-0.2, 0.1, 0.3, 0.2, -0.05, 0.15]];
    let gh = [[0.3f32, 0.1, -0.2, 0.4], [0.1, -0.1, 0.2, 0.05]];
    let gb = [G1, G2];

    let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let (mut rw, mut rh) = (f(&w.to_vec::<f32>().unwrap()), f(&h.to_vec::<f32>().unwrap()));
    let mut rb = vec![1.0, 0.9];
    let (mut mw, mut mh) = (vec![0.0; 6], vec![0.0; 4]);
    let (mut sw, mut sh, mut sb) = (vec![0.0; 6], vec![0.0; 4], vec![0.0; 2]);
    let muon = |p: &mut [f64], m: &mut [f64], s: &mut [f64], g: &[f64], lr: f64| {
        for (m, g) in m.iter_mut().zip(g) {
            *m = mu * *m + (1.0 - mu) * g;
        }
        let u: Vec<f64> = m.iter().zip(g).map(|(m, g)| mu * m + (1.0 - mu) * g).collect();
        let o = ortho_f64(&u, 2, u.len() / 2, ortho);
        for (p, o) in p.iter_mut().zip(&o) {
            *p -= weights.muon * lr * o;
        }
        sgd_ref(p, s, g, weights.sgd * lr, wd);
    };
    for k in 0..2 {
        let dot = |p: &Tensor, g: &[f32]| {
            p.mul(&Tensor::from_slice(g, p.shape().clone()).unwrap()).unwrap().sum_all().unwrap()
        };
        let loss = dot(&w, &gw[k]).add(&dot(&h, &gh[k])).unwrap().add(&dot(&bn, &gb[k])).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
        muon(&mut rw, &mut mw, &mut sw, &f(&gw[k]), lr);
        muon(&mut rh, &mut mh, &mut sh, &f(&gh[k]), 3.0 * lr);
        sgd_ref(&mut rb, &mut sb, &f(&gb[k]), lr, 0.0);
        close(&w.to_vec::<f32>().unwrap(), &rw, 1e-5);
        close(&h.to_vec::<f32>().unwrap(), &rh, 1e-5);
        close(&bn.to_vec::<f32>().unwrap(), &rb, 1e-6);
    }
    let names: Vec<String> = opt.state_tensors().unwrap().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        ["w.momentum_buffer", "h.momentum_buffer", "bn.momentum_buffer", "w.muon_buffer", "h.muon_buffer"]
    );
    close(&opt.muon_buffer(0, 0).unwrap().to_vec::<f32>().unwrap(), &mw, 1e-6);
    let c = opt.contract();
    assert_eq!(c.kind, OptimKind::MuSgd);
    assert_eq!((c.groups[1].lr, c.groups[1].momentum), (3.0 * lr, Some(mu)));
}

#[test]
fn boosted_groups_and_bias_warmup() {
    let mut rng = Rng::seed(0);
    let model = Sequential::new()
        .push_named("stem", ConvBnAct::new(3, 4, 3, 1, 1, Activation::Silu, &mut rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, 2, 1), &mut rng));
    let gs = StandardGroups::from_module(&model).into_boosted_groups(
        0.01,
        0.9,
        5e-4,
        |n| n.starts_with("out."),
        3.0,
    );
    let summary: Vec<(&str, usize, f64, f64)> =
        gs.iter().map(|g| (g.name.as_str(), g.params.len(), g.base_lr, g.weight_decay)).collect();
    assert_eq!(
        summary,
        [
            ("decay", 1, 0.01, 5e-4),
            ("decay.cls", 1, 0.03, 5e-4),
            ("bn", 1, 0.01, 0.0),
            ("bias", 1, 0.01, 0.0),
            ("bias.cls", 1, 0.03, 0.0)
        ]
    );
    assert!(
        is_bias_group("bias")
            && is_bias_group("bias.cls")
            && !is_bias_group("biased")
            && !is_bias_group("bn")
    );
    assert!(is_muon_group("decay") && is_muon_group("decay.cls") && !is_muon_group("bn"));
    // Warmup starts both bias groups at warmup_bias_lr, the others at 0.
    let sched = LrSchedule::new(DecayKind::Linear, 30, 13, 0.01, 3.0, 0.1, 0.8).unwrap();
    assert_eq!(sched.lr_at(0, &gs[4]), 0.1);
    assert_eq!(sched.lr_at(0, &gs[1]), 0.0);
}

struct EmaMixedNet {
    seq: Sequential,
    list: ModuleList<Conv2d>,
    big: Tensor,
    odd: Tensor,
    int_buf: Tensor,
}

impl EmaMixedNet {
    fn new(rng: &mut Rng) -> Self {
        let seq = Sequential::new()
            .push_named("stem", ConvBnAct::new(3, 17, 3, 1, 1, Activation::Silu, rng))
            .push_named("bn", BatchNorm2d::new(17));
        let list = ModuleList::new(vec![
            Conv2d::new(Conv2dConfig::new(17, 31, 3), rng),
            Conv2d::new(Conv2dConfig::new(31, 33, 1), rng),
        ]);
        let big = Tensor::randn([65_573], 0.0, 1.0, rng).requires_grad_();
        let odd = Tensor::randn([39], 0.0, 1.0, rng).requires_grad_();
        let int_buf = Tensor::zeros_with([4], folo_core::DType::I64, folo_core::Device::Cpu).unwrap();
        EmaMixedNet { seq, list, big, odd, int_buf }
    }
}

impl Module for EmaMixedNet {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("big", &self.big, TensorKind::Parameter);
        f("odd", &self.odd, TensorKind::Parameter);
        f("int_buf", &self.int_buf, TensorKind::Buffer);
    }

    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("seq", &self.seq);
        f("", &self.list);
    }
}

fn tracked_tensors(m: &dyn Module) -> Vec<(String, Tensor)> {
    m.named_tensors()
        .into_iter()
        .filter(|t| t.kind == TensorKind::Parameter || t.tensor.dtype() == folo_core::DType::F32)
        .map(|t| (t.name, t.tensor))
        .collect()
}

fn perturb_tracked(m: &dyn Module, rng: &mut Rng) {
    let _g = no_grad();
    for (_, t) in tracked_tensors(m) {
        let delta = Tensor::randn(t.shape().clone(), 0.0, 0.05, rng);
        t.add_assign(&delta).unwrap();
    }
}

fn tracked_bits(m: &dyn Module) -> Vec<(String, Vec<u32>)> {
    tracked_tensors(m)
        .into_iter()
        .map(|(n, t)| (n, t.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect()))
        .collect()
}

#[test]
fn ema_bit_exact_100_steps_and_resume_reset() {
    let mut rng = Rng::seed(42);
    let live = EmaMixedNet::new(&mut rng);
    let mut ema = ModelEma::with_params(EmaMixedNet::new(&mut rng), &live, 0.999, 50.0).unwrap();
    let mut ref_state: Vec<Vec<f32>> =
        tracked_tensors(&live).into_iter().map(|(_, t)| t.to_vec::<f32>().unwrap()).collect();

    let x = Tensor::randn([2, 3, 4, 4], 0.0, 1.0, &mut rng);
    for step in 1..=100u64 {
        perturb_tracked(&live, &mut rng);
        if step % 25 == 0 {
            let _ = live.seq.forward(&x).unwrap();
        }
        ema.update(&live).unwrap();
        let d = 0.999 * (1.0 - (-(step as f64) / 50.0).exp());
        let (d32, w) = (d as f32, (1.0 - d) as f32);
        for (e, (_, m)) in ref_state.iter_mut().zip(tracked_tensors(&live)) {
            for (ev, mv) in e.iter_mut().zip(m.to_vec::<f32>().unwrap()) {
                *ev = *ev * d32 + mv * w;
            }
        }
    }
    assert_eq!(ema.updates(), 100);
    for ((name, got), want) in tracked_bits(ema.module()).iter().zip(&ref_state) {
        let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, &want_bits, "tensor `{name}` diverged after 100 steps");
    }

    // Resume/reload consistency via EmaState::restore.
    let saved: std::collections::HashMap<String, Tensor> = ema
        .ema_tensors()
        .into_iter()
        .map(|(n, t)| (n, Tensor::from_vec(t.to_vec::<f32>().unwrap(), t.shape().clone()).unwrap()))
        .collect();
    let mut ema_b = ModelEma::with_params(EmaMixedNet::new(&mut rng), &live, 0.5, 5.0).unwrap();
    ema_b.restore(&saved, ema.updates(), ema.contract()).unwrap();
    assert_eq!(tracked_bits(ema.module()), tracked_bits(ema_b.module()));
    for _ in 0..15 {
        perturb_tracked(&live, &mut rng);
        ema.update(&live).unwrap();
        ema_b.update(&live).unwrap();
    }
    assert_eq!(tracked_bits(ema.module()), tracked_bits(ema_b.module()));

    // Reset matches a fresh EMA.
    ema.reset(&live).unwrap();
    assert_eq!(ema.updates(), 0);
    let mut fresh = ModelEma::with_params(EmaMixedNet::new(&mut rng), &live, 0.999, 50.0).unwrap();
    for _ in 0..5 {
        perturb_tracked(&live, &mut rng);
        ema.update(&live).unwrap();
        fresh.update(&live).unwrap();
    }
    assert_eq!(tracked_bits(ema.module()), tracked_bits(fresh.module()));
}

#[test]
fn ema_mismatches_non_contiguous_aliasing_and_thread_determinism() {
    struct Single(&'static str, Tensor);
    impl Module for Single {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f(self.0, &self.1, TensorKind::Parameter);
        }
    }
    let live = Single("weight", Tensor::full([4], 2.0).requires_grad_());
    let mut ema = ModelEma::new(Single("weight", Tensor::zeros([4])), &live).unwrap();
    ema.update(&live).unwrap();
    let before = tracked_bits(ema.module());
    assert!(ema.update(&Single("other", Tensor::full([4], 9.0))).is_err());
    assert!(ema.update(&Single("weight", Tensor::full([5], 9.0))).is_err());
    assert!(
        ema.update(&Single(
            "weight",
            Tensor::full_with([4], 9.0, folo_core::DType::F16, folo_core::Device::Cpu).unwrap()
        ))
        .is_err()
    );
    assert!(ema.update(&Sequential::new()).is_err());
    assert_eq!((ema.updates(), tracked_bits(ema.module())), (1, before));

    // Non-contiguous EMA & non-contiguous source parity.
    let base_live = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let base_ema = Tensor::zeros([3, 2]).t().unwrap();
    let mut ema_nc =
        ModelEma::with_params(Single("w", base_ema), &Single("w", base_live.clone()), 0.9, 10.0).unwrap();
    let mut ema_c =
        ModelEma::with_params(Single("w", Tensor::zeros([2, 3])), &Single("w", base_live.clone()), 0.9, 10.0)
            .unwrap();
    {
        let _g = no_grad();
        base_live.add_assign(&Tensor::full([2, 3], 1.5)).unwrap();
    }
    ema_nc.update(&Single("w", base_live.clone())).unwrap();
    ema_c.update(&Single("w", base_live)).unwrap();
    assert_eq!(tracked_bits(ema_nc.module()), tracked_bits(ema_c.module()));

    let src_t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap().t().unwrap();
    let src_c = Tensor::from_vec(src_t.to_vec::<f32>().unwrap(), [2, 3]).unwrap();
    let mut s_nc = ModelEma::with_params(
        Single("w", Tensor::zeros([2, 3])),
        &Single("w", Tensor::full([2, 3], 2.0)),
        0.9,
        10.0,
    )
    .unwrap();
    let mut s_c = ModelEma::with_params(
        Single("w", Tensor::zeros([2, 3])),
        &Single("w", Tensor::full([2, 3], 2.0)),
        0.9,
        10.0,
    )
    .unwrap();
    s_nc.update(&Single("w", src_t)).unwrap();
    s_c.update(&Single("w", src_c)).unwrap();
    assert_eq!(tracked_bits(s_nc.module()), tracked_bits(s_c.module()));

    // Thread count determinism (1 vs 4).
    let mut rng = Rng::seed(99);
    let m_live = EmaMixedNet::new(&mut rng);
    let mut e1 = ModelEma::with_params(EmaMixedNet::new(&mut rng), &m_live, 0.99, 10.0).unwrap();
    let mut e4 = ModelEma::with_params(EmaMixedNet::new(&mut rng), &m_live, 0.99, 10.0).unwrap();
    perturb_tracked(&m_live, &mut rng);
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| e1.update(&m_live).unwrap());
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(|| e4.update(&m_live).unwrap());
    assert_eq!(tracked_bits(e1.module()), tracked_bits(e4.module()));

    // Self-alias (detach + transpose), crossed alias, tied weights, and in-place buffer dtype change.
    struct Pair(Tensor, Tensor);
    impl Module for Pair {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("a", &self.0, TensorKind::Parameter);
            f("b", &self.1, TensorKind::Parameter);
        }
    }
    let w0 = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0], [2, 2]).unwrap();
    let mut ema_al = ModelEma::with_params(
        Pair(Tensor::zeros([2, 2]), Tensor::zeros([2, 2])),
        &Pair(w0.clone(), w0),
        0.5,
        1.0,
    )
    .unwrap();
    let d = ema_al.decay_at(1);
    let (d32, w) = (d as f32, (1.0 - d) as f32);
    let alias_src = Pair(ema_al.module().0.detach(), ema_al.module().1.t().unwrap());
    ema_al.update(&alias_src).unwrap();
    assert_eq!(
        ema_al.module().0.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        [2.0f32, 4.0, 6.0, 8.0].iter().map(|&v| (v * d32 + v * w).to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        ema_al.module().1.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        [2.0f32, 4.0, 6.0, 8.0]
            .iter()
            .zip([2.0f32, 6.0, 4.0, 8.0])
            .map(|(&e, m)| (e * d32 + m * w).to_bits())
            .collect::<Vec<_>>()
    );

    let n = 65_552;
    let mut ema_cross = ModelEma::with_params(
        Pair(Tensor::zeros([n]), Tensor::zeros([n])),
        &Pair(Tensor::full([n], 1.0), Tensor::full([n], 3.0)),
        0.8,
        2.0,
    )
    .unwrap();
    let crossed = Pair(ema_cross.module().1.clone(), ema_cross.module().0.clone());
    let d1 = ema_cross.decay_at(1);
    let (d1_32, w1) = (d1 as f32, (1.0 - d1) as f32);
    ema_cross.update(&crossed).unwrap();
    let a_want = 1.0f32 * d1_32 + 3.0f32 * w1;
    let b_want = 3.0f32 * d1_32 + a_want * w1;
    assert_eq!(ema_cross.module().0.to_vec::<f32>().unwrap()[0].to_bits(), a_want.to_bits());
    assert_eq!(ema_cross.module().1.to_vec::<f32>().unwrap()[0].to_bits(), b_want.to_bits());

    let tied_w = Tensor::full([n], 2.0);
    let mut ema_tied = ModelEma::with_params(
        Pair(tied_w.clone(), tied_w),
        &Pair(Tensor::full([n], 2.0), Tensor::full([n], 2.0)),
        0.8,
        2.0,
    )
    .unwrap();
    ema_tied.update(&Pair(Tensor::full([n], 5.0), Tensor::full([n], 9.0))).unwrap();
    let step2 = (2.0f32 * d1_32 + 5.0f32 * w1) * d1_32 + 9.0f32 * w1;
    assert_eq!(ema_tied.module().0.to_vec::<f32>().unwrap()[0].to_bits(), step2.to_bits());

    struct WithBuf(Tensor, Tensor);
    impl Module for WithBuf {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("w", &self.0, TensorKind::Parameter);
            f("buf", &self.1, TensorKind::Buffer);
        }
    }
    let live_b = WithBuf(Tensor::full([4], 1.0), Tensor::full([4], 2.0));
    let mut ema_b = ModelEma::new(WithBuf(Tensor::zeros([4]), Tensor::zeros([4])), &live_b).unwrap();
    assert_eq!(ema_b.ema_tensors().len(), 2);
    live_b.1.to_dtype_(folo_core::DType::F16).unwrap();
    ema_b.module().1.to_dtype_(folo_core::DType::F16).unwrap();
    assert_eq!(ema_b.ema_tensors().len(), 1);
    ema_b.update(&live_b).unwrap();
}

#[test]
fn unscale_clip_and_global_norm_large_cpu_parity() {
    let mut rng = Rng::seed(123);
    // Exceeds PAR_THRESHOLD (65,536) across multiple uneven tensors to exercise parallel AVX2 paths
    // (9,216 + 18,432 + 41,472 + 37 = 69,157 > 65,536).
    let shapes: [&[usize]; 4] = [&[32, 32, 3, 3], &[64, 32, 3, 3], &[144, 32, 3, 3], &[37]];
    let params: Vec<Tensor> =
        shapes.iter().map(|s| Tensor::randn(s.to_vec(), 0.0, 1.0, &mut rng).requires_grad_()).collect();
    let total_numel: usize = params.iter().map(Tensor::numel).sum();
    assert!(total_numel > 65_536, "fixture must exercise parallel path: {total_numel}");
    let mut grads = Grads::default();
    let mut ref_vecs = Vec::new();
    for p in &params {
        let g = Tensor::randn(p.shape().clone(), 0.0, 4.0, &mut rng);
        ref_vecs.push(g.to_vec::<f32>().unwrap());
        grads.insert(p, g);
    }
    let mut acc = GradAccumulator::new(&params);
    acc.add(&grads).unwrap();

    // Thread-count determinism (1 vs 4 workers) for both global_norm and unscale_clip.
    let mut acc1 = GradAccumulator::new(&params);
    let mut acc4 = GradAccumulator::new(&params);
    acc1.add(&grads).unwrap();
    acc4.add(&grads).unwrap();
    let n1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| acc1.global_norm().unwrap());
    let n4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| acc4.global_norm().unwrap());
    assert_eq!(n1.to_bits(), n4.to_bits(), "global_norm must be thread-count deterministic");

    // Unscale by 0.25 and clip to max_norm = 5.0.
    let inv_scale = 0.25f32;
    let max_norm = 5.0f64;
    let ok1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| acc1.unscale_clip(inv_scale, max_norm).unwrap());
    let ok4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| acc4.unscale_clip(inv_scale, max_norm).unwrap());
    assert!(ok1 && ok4);
    for p in &params {
        let b1: Vec<u32> =
            acc1.grad(p).unwrap().to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
        let b4: Vec<u32> =
            acc4.grad(p).unwrap().to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
        assert_eq!(b1, b4, "unscale_clip must be bitwise thread-count deterministic");
    }

    assert!(acc.unscale_clip(inv_scale, max_norm).unwrap());
    let mut ref_sq = 0.0f64;
    for v in &mut ref_vecs {
        for x in v.iter_mut() {
            *x *= inv_scale;
            ref_sq += (*x as f64) * (*x as f64);
        }
    }
    let ref_n = ref_sq.sqrt();
    assert!(ref_n > max_norm);
    let c = (max_norm / (ref_n + 1e-6)) as f32;
    for (p, want) in params.iter().zip(&mut ref_vecs) {
        for x in want.iter_mut() {
            *x *= c;
        }
        let got = acc.grad(p).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
    assert!((acc.global_norm().unwrap() - max_norm).abs() < 1e-3);

    // Broadcastable (scalar) gradient on a multi-element parameter must work across multiple add() passes.
    let v = param(&[1.0, 2.0, 3.0]);
    let mut bacc = GradAccumulator::new(std::slice::from_ref(&v));
    let mut sg = Grads::default();
    sg.insert(&v, Tensor::from_slice(&[2.5f32], [1]).unwrap());
    bacc.add(&sg).unwrap();
    bacc.add(&sg).unwrap();
    assert_eq!(bacc.grad(&v).unwrap().to_vec::<f32>().unwrap(), [5.0, 5.0, 5.0]);

    // Non-finite gradient causes unscale_clip to return false (both with and without clipping).
    for clip in [0.0, 10.0] {
        acc.zero();
        let mut bad_grads = Grads::default();
        for (i, p) in params.iter().enumerate() {
            let mut v = vec![1.0f32; p.numel()];
            if i == 2 {
                v[1234] = f32::INFINITY;
            }
            bad_grads.insert(p, Tensor::from_vec(v, p.shape().clone()).unwrap());
        }
        acc.add(&bad_grads).unwrap();
        assert!(!acc.unscale_clip(0.5, clip).unwrap(), "clip={clip}");
    }

    // `orthogonalize_with` must never mutate its input tensor in place.
    let m = Tensor::randn([8, 12], 0.0, 1.0, &mut rng);
    let before: Vec<u32> = m.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
    let _ = orthogonalize_with(&m, MuonOrtho::Polar4).unwrap();
    let after: Vec<u32> = m.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
    assert_eq!(before, after, "orthogonalize_with mutated its input");

    // Cross-parameter gradient aliasing in MuSgd (parameter A's gradient aliases parameter B's muon_buffer)
    // must not deadlock and must match non-aliased execution.
    let pa = Tensor::from_slice(&[0.5f32, -0.3, 0.2, 0.4], [2, 2]).unwrap().requires_grad_();
    let pb = Tensor::from_slice(&[-0.1f32, 0.6, -0.4, 0.2], [2, 2]).unwrap().requires_grad_();
    let mut mopt = MuSgd::new(vec![
        ParamGroup::new("decay", vec![("a".into(), pa.clone()), ("b".into(), pb.clone())])
            .lr(0.05)
            .momentum(0.9),
    ])
    .unwrap();
    let ba = mopt.muon_buffer(0, 0).unwrap().clone();
    let bb = mopt.muon_buffer(0, 1).unwrap().clone();
    {
        let _g = no_grad();
        ba.copy_from(&Tensor::from_slice(&[0.1f32, 0.2, -0.1, 0.3], [2, 2]).unwrap()).unwrap();
        bb.copy_from(&Tensor::from_slice(&[-0.2f32, 0.1, 0.4, -0.1], [2, 2]).unwrap()).unwrap();
    }
    let mut aliased = Grads::default();
    aliased.insert(&pa, bb);
    aliased.insert(&pb, ba);
    mopt.step(&aliased).unwrap();
}
