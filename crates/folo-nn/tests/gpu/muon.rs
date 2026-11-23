//! MuSGD's fused Muon update (`Tensor::muon_step_many_`) against the tensor ops (the Nesterov mix as tensor ops,
//! `orthogonalize_many_with`, then `p · 1 + O · (−lr)`, all on the CPU), for both default `NS5` and Polar-FOLO
//! (`Polar4`, `Polar5`) schedules, plus multi-group fused `MuSgd` (`decay` + `decay.cls`). `lr = 1`, so the
//! parameters carry `O` at full size; values agree to `2e-4 + 2e-3·|ref|`, NaNs exactly where the reference has them.

use super::*;

fn refs(v: &[Tensor]) -> Vec<&Tensor> {
    v.iter().collect()
}

#[track_caller]
fn close_or_nan(got: &Tensor, want: &Tensor, what: &str) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let (g, w) = (got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
    for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
        assert_eq!(a.is_nan(), b.is_nan(), "{what}[{i}]: got {a}, want {b}");
        assert!(b.is_nan() || (a - b).abs() <= 2e-4 + 2e-3 * b.abs(), "{what}[{i}]: got {a}, want {b}");
    }
}

/// The fused step on copies of `ps`, `ms`, `gs` on `dev`, with `μ = 0.9`, `lr = 1`; checks that the momenta and
/// gradients are only read and returns the parameters after it.
fn fused(
    dev: Device,
    ps: &[Tensor],
    ms: &[Tensor],
    gs: &[Tensor],
    nesterov: bool,
    ortho: MuonOrtho,
) -> Vec<Tensor> {
    let up = |ts: &[Tensor]| ts.iter().map(|t| t.to_device(dev).unwrap()).collect::<Vec<_>>();
    let (pg, mg, gg) = (up(ps), up(ms), up(gs));
    let scales: Vec<f32> = ms.iter().map(|m| muon_scale(m.dims())).collect();
    let settings = folo_core::backend::MuonSettings::from_schedule(
        0.9,
        nesterov,
        ortho.schedule(),
        ortho.schatten4(),
        NS_EPS,
        1.0,
    );
    let _g = no_grad();
    assert!(
        Tensor::muon_step_many_(&refs(&pg), &refs(&mg), &refs(&gg), &scales, &settings).unwrap(),
        "a GPU kernel"
    );
    for (a, b) in mg.iter().zip(ms).chain(gg.iter().zip(gs)) {
        let (a, b) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "momenta and gradients are only read"
        );
    }
    pg
}

/// The tensor ops' result on the CPU (`μ = 0.9`, `lr = 1`).
fn reference(ps: &[Tensor], ms: &[Tensor], gs: &[Tensor], nesterov: bool, ortho: MuonOrtho) -> Vec<Tensor> {
    let mu = 0.9f32;
    let us: Vec<Tensor> = ms
        .iter()
        .zip(gs)
        .map(|(m, g)| {
            if nesterov {
                g.mul_scalar(1.0 - mu).unwrap().add(&m.mul_scalar(mu).unwrap()).unwrap()
            } else {
                m.clone()
            }
        })
        .collect();
    let os = orthogonalize_many_with(&us, ortho).unwrap();
    ps.iter().zip(&os).map(|(p, o)| p.add(&o.mul_scalar(-1.0).unwrap()).unwrap()).collect()
}

/// Wide and tall matrices (tall ones are transposed), several tiles along both sides, single rows and
/// columns, a square one, scales other than 1, and more matrices than one multi-tensor dispatch takes — for
/// default `Ns5` and Polar-FOLO (`Polar4`, `Polar5`).
#[test]
fn fused_muon_matches_tensor_ops() {
    let dev = gpu!();
    let shapes: [&[usize]; 9] = [
        &[16, 3, 3, 3],
        &[130, 40],
        &[200, 7, 3, 3],
        &[70, 300],
        &[1, 5],
        &[5, 1, 1, 1],
        &[96, 96, 1, 1],
        &[3, 9],
        &[33, 2, 3, 1],
    ];
    let rng = &mut Rng::seed(11);
    let (mut ps, mut ms, mut gs) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..20 {
        let s = shapes[k % shapes.len()];
        ps.push(Tensor::rand_uniform(s, -1.0, 1.0, rng));
        ms.push(Tensor::rand_uniform(s, -1.0, 1.0, rng));
        gs.push(Tensor::rand_uniform(s, -0.5, 0.5, rng));
    }
    for ortho in [MuonOrtho::Ns5, MuonOrtho::Polar4, MuonOrtho::Polar5] {
        for nesterov in [false, true] {
            let got = fused(dev, &ps, &ms, &gs, nesterov, ortho);
            let want = reference(&ps, &ms, &gs, nesterov, ortho);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                close_or_nan(g, w, &format!("{ortho:?} nesterov {nesterov}: matrix {i} {:?}", ps[i].dims()));
            }
        }
    }
}

/// A zero momentum and gradient (the eps keeps `0 / ‖0‖` finite: `p` stays as it was), one near zero (its norm
/// about the eps), one of 184,320 elements (720 partial sums: the norm's loop runs three times per invocation),
/// a NaN in one gradient (that parameter turns NaN where the tensor ops' does, with Nesterov; nothing else does)
/// and an empty one, left alone — tested for both `Ns5` and `Polar4` (with Schatten-4).
#[test]
fn fused_muon_edge_cases() {
    let dev = gpu!();
    let rng = &mut Rng::seed(12);
    let shapes: [&[usize]; 7] =
        [&[8, 12], &[20, 30], &[160, 1152], &[24, 40], &[260, 290], &[16, 3, 3, 3], &[0, 5]];
    let (mut ps, mut ms, mut gs) = (Vec::new(), Vec::new(), Vec::new());
    for s in shapes {
        ps.push(Tensor::rand_uniform(s, -1.0, 1.0, rng));
        ms.push(Tensor::rand_uniform(s, -1.0, 1.0, rng));
        gs.push(Tensor::rand_uniform(s, -0.5, 0.5, rng));
    }
    (ms[0], gs[0]) = (Tensor::zeros(shapes[0]), Tensor::zeros(shapes[0]));
    (ms[1], gs[1]) = (ms[1].mul_scalar(1e-8).unwrap(), gs[1].mul_scalar(1e-8).unwrap());
    let mut g3 = gs[3].to_vec::<f32>().unwrap();
    g3[77] = f32::NAN;
    gs[3] = Tensor::from_vec(g3, shapes[3]).unwrap();
    for ortho in [MuonOrtho::Ns5, MuonOrtho::Polar4] {
        for nesterov in [false, true] {
            let got = fused(dev, &ps, &ms, &gs, nesterov, ortho);
            let want = reference(&ps, &ms, &gs, nesterov, ortho);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                close_or_nan(g, w, &format!("{ortho:?} nesterov {nesterov}: matrix {i} {:?}", ps[i].dims()));
            }
            assert_eq!(got[0].to_vec::<f32>().unwrap(), ps[0].to_vec::<f32>().unwrap(), "{ortho:?} zero");
            let (a, b) = (got[1].to_vec::<f32>().unwrap(), ps[1].to_vec::<f32>().unwrap());
            assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3), "{ortho:?} near zero: still a step");
            let nans = got[3].to_vec::<f32>().unwrap().iter().filter(|x| x.is_nan()).count();
            assert_eq!(nans, if nesterov { 24 * 40 } else { 0 }, "{ortho:?} nesterov {nesterov}: NaN grad");
            assert_eq!(got[6].dims(), [0, 5]);
        }
    }
}

/// Multi-group fused MuSGD (`decay` + `decay.cls` at 3x LR + `bn`) on GPU matches CPU across multiple steps for
/// `Ns5`, `Polar4`, and `Polar5`, and every kernel used is registered for precompile.
#[test]
fn musgd_multi_group_gpu_matches_cpu() {
    let dev = gpu!();
    for ortho in [MuonOrtho::Ns5, MuonOrtho::Polar4, MuonOrtho::Polar5] {
        let mut rng = Rng::seed(31);
        let mk = |rng: &mut Rng| {
            let w0 = Tensor::rand_uniform([32, 16, 3, 3], -0.5, 0.5, rng).requires_grad_();
            let w1 = Tensor::rand_uniform([70, 24, 1, 1], -0.5, 0.5, rng).requires_grad_();
            let cls = Tensor::rand_uniform([80, 64, 1, 1], -0.5, 0.5, rng).requires_grad_();
            let dw = Tensor::rand_uniform([64, 1, 3, 3], -0.5, 0.5, rng).requires_grad_();
            let bn = Tensor::rand_uniform([32], 0.5, 1.5, rng).requires_grad_();
            vec![w0, w1, cls, dw, bn]
        };
        let cpu_ps = mk(&mut rng);
        let gpu_ps: Vec<Tensor> =
            cpu_ps.iter().map(|t| t.detach().to_device(dev).unwrap().requires_grad_()).collect();
        let make_opt = |ps: &[Tensor]| {
            let gs = vec![
                ParamGroup::new("decay", vec![("w0".into(), ps[0].clone()), ("w1".into(), ps[1].clone())])
                    .lr(0.02)
                    .momentum(0.937)
                    .weight_decay(5e-4),
                ParamGroup::new(
                    "decay.cls",
                    vec![("cls".into(), ps[2].clone()), ("dw".into(), ps[3].clone())],
                )
                .lr(0.06)
                .momentum(0.937)
                .weight_decay(5e-4),
                ParamGroup::new("bn", vec![("bn".into(), ps[4].clone())]).lr(0.02).momentum(0.937),
            ];
            MuSgd::with_ortho(gs, MuSgdWeights { muon: 0.5, sgd: 0.5 }, ortho).unwrap()
        };
        let mut cpu_opt = make_opt(&cpu_ps);
        let mut gpu_opt = make_opt(&gpu_ps);
        for _ in 0..3 {
            let mut g_cpu = folo_core::Grads::default();
            let mut g_gpu = folo_core::Grads::default();
            for (pc, pg) in cpu_ps.iter().zip(&gpu_ps) {
                let g = Tensor::rand_uniform(pc.shape().clone(), -0.3, 0.3, &mut rng);
                g_gpu.insert(pg, g.to_device(dev).unwrap());
                g_cpu.insert(pc, g);
            }
            cpu_opt.step(&g_cpu).unwrap();
            gpu_opt.step(&g_gpu).unwrap();
        }
        for (i, (pg, pc)) in gpu_ps.iter().zip(&cpu_ps).enumerate() {
            close_or_nan(pg, pc, &format!("{ortho:?} multi-group param {i} {:?}", pc.dims()));
        }
    }
    let ctx = folo_gpu::GpuContext::shared().unwrap();
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
