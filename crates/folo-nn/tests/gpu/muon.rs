//! MuSGD's fused Muon update (`Tensor::muon_step_many_`) against the tensor ops (the Nesterov mix as tensor ops,
//! `orthogonalize_many_with`, then `p · 1 + O · (−lr)`, all on the CPU), for both legacy `Ns5` and Polar-FOLO
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
/// legacy `Ns5` and Polar-FOLO (`Polar4`, `Polar5`).
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

/// Finite Polar-FOLO inputs whose unscaled squared Frobenius norm would overflow `f32::MAX` (`1×1` `2e19`,
/// `16×16` `1e19`, `4×4` `1e38`, and a multi-workgroup `16×64` mixing `> 2^48` and `<= 2^48` workgroups) use
/// exact power-of-two downscaling on both CPU and GPU (zero non-finite intermediates under WGSL finite-math
/// semantics), matching closely across `Polar4` and `Polar5`. (`NaN` propagation and legacy `Ns5` overflow are
/// verified on CPU in `tests/optim.rs` because WGSL assumes finite-math.)
#[test]
fn fused_muon_frobenius_overflow_finite() {
    let dev = gpu!();
    let rng = &mut Rng::seed(41);
    let p0 = Tensor::rand_uniform([1, 1], -1.0, 1.0, rng);
    let p1 = Tensor::rand_uniform([16, 16], -1.0, 1.0, rng);
    let p2 = Tensor::rand_uniform([4, 4], -1.0, 1.0, rng);
    let p3 = Tensor::rand_uniform([16, 64], -1.0, 1.0, rng);
    let m0 = Tensor::from_slice(&[2e19_f32], [1, 1]).unwrap();
    let g0 = Tensor::from_slice(&[2e19_f32], [1, 1]).unwrap();
    let m1 = Tensor::from_vec(vec![1e19_f32; 16 * 16], [16, 16]).unwrap();
    let g1 = Tensor::from_vec(vec![1e19_f32; 16 * 16], [16, 16]).unwrap();
    let m2 = Tensor::from_vec(vec![1e38_f32; 4 * 4], [4, 4]).unwrap();
    let g2 = Tensor::from_vec(vec![1e38_f32; 4 * 4], [4, 4]).unwrap();
    let mut mixed = vec![1.0_f32; 16 * 64];
    mixed[0] = 3e14; // > 2^48 in workgroup 0
    mixed[500] = 1e14; // <= 2^48 in workgroup 1 (rescaled in muon_norm)
    let m3 = Tensor::from_vec(mixed.clone(), [16, 64]).unwrap();
    let g3 = Tensor::from_vec(mixed, [16, 64]).unwrap();
    let (ps, ms, gs) = ([p0, p1, p2, p3], [m0, m1, m2, m3], [g0, g1, g2, g3]);
    for ortho in [MuonOrtho::Polar4, MuonOrtho::Polar5] {
        for nesterov in [false, true] {
            let got = fused(dev, &ps, &ms, &gs, nesterov, ortho);
            let want = reference(&ps, &ms, &gs, nesterov, ortho);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                close_or_nan(g, w, &format!("{ortho:?} nesterov {nesterov}: overflow[{i}]"));
            }
            for (i, (before_t, after_t)) in ps.iter().zip(&got).enumerate() {
                let before = before_t.to_vec::<f32>().unwrap();
                let after = after_t.to_vec::<f32>().unwrap();
                assert!(
                    after.iter().all(|v| v.is_finite())
                        && before.iter().zip(&after).any(|(a, b)| (a - b).abs() > 1e-3),
                    "{ortho:?} nesterov {nesterov}: overflow[{i}] must take a finite non-zero orthogonal step"
                );
            }
        }
    }
}

/// When Polar-FOLO multi-group fusion sees a cross-group storage alias (group B's gradient sharing storage with
/// group A's parameter), `MuSgd` detects the alias BEFORE updating momentum buffers and runs per-group so GPU
/// matches CPU's exact sequential per-group semantics.
#[test]
fn polar_cross_group_grad_alias_matches_cpu() {
    let dev = gpu!();
    let mut rng = Rng::seed(71);
    let init0 = Tensor::rand_uniform([8, 12], -0.5, 0.5, &mut rng);
    let init1 = Tensor::rand_uniform([8, 12], -0.5, 0.5, &mut rng);
    let g0_cpu = Tensor::rand_uniform([8, 12], -0.3, 0.3, &mut rng);

    let p0_cpu = init0.detach().requires_grad_();
    let p1_cpu = init1.detach().requires_grad_();
    let p0_gpu = init0.to_device(dev).unwrap().requires_grad_();
    let p1_gpu = init1.to_device(dev).unwrap().requires_grad_();
    let g0_gpu = g0_cpu.to_device(dev).unwrap();

    let make_opt = |p0: &Tensor, p1: &Tensor| {
        let gs = vec![
            ParamGroup::new("decay", vec![("w0".into(), p0.clone())])
                .lr(0.05)
                .momentum(0.9)
                .weight_decay(0.0),
            ParamGroup::new("decay.cls", vec![("w1".into(), p1.clone())])
                .lr(0.15)
                .momentum(0.9)
                .weight_decay(0.0),
        ];
        MuSgd::with_ortho(gs, MuSgdWeights { muon: 0.5, sgd: 0.0 }, MuonOrtho::Polar4).unwrap()
    };
    let mut cpu_opt = make_opt(&p0_cpu, &p1_cpu);
    let mut gpu_opt = make_opt(&p0_gpu, &p1_gpu);

    for _ in 0..2 {
        let mut g_cpu = folo_core::Grads::default();
        g_cpu.insert(&p0_cpu, g0_cpu.clone());
        g_cpu.insert(&p1_cpu, p0_cpu.detach()); // shares storage with p0_cpu!
        let mut g_gpu = folo_core::Grads::default();
        g_gpu.insert(&p0_gpu, g0_gpu.clone());
        g_gpu.insert(&p1_gpu, p0_gpu.detach()); // shares storage with p0_gpu!
        assert!(p0_cpu.same_storage(g_cpu.get(&p1_cpu).unwrap()));
        assert!(p0_gpu.same_storage(g_gpu.get(&p1_gpu).unwrap()));
        cpu_opt.step(&g_cpu).unwrap();
        gpu_opt.step(&g_gpu).unwrap();
    }
    close_or_nan(&p0_gpu, &p0_cpu, "cross-group alias w0");
    close_or_nan(&p1_gpu, &p1_cpu, "cross-group alias w1");
}

/// When Polar-FOLO multi-group fusion would overflow `muon_scale * group_lr` in `f32` (e.g. `scale = sqrt(17/5)`
/// and `lr = 2e38`), `MuSgd` skips folding and runs per-group so a zero update direction stays `0.0` (no
/// `0.0 * inf = NaN`).
#[test]
fn polar_folded_scale_overflow_falls_back_cleanly() {
    let dev = gpu!();
    let mut rng = Rng::seed(61);
    let p0_init = Tensor::rand_uniform([17, 5], -1.0, 1.0, &mut rng);
    let p1_init = Tensor::rand_uniform([17, 5], -1.0, 1.0, &mut rng);
    let p0 = p0_init.to_device(dev).unwrap().requires_grad_();
    let p1 = p1_init.to_device(dev).unwrap().requires_grad_();
    let gs = vec![
        ParamGroup::new("decay", vec![("w0".into(), p0.clone())]).lr(1.0).momentum(0.9).weight_decay(0.0),
        ParamGroup::new("decay.cls", vec![("w1".into(), p1.clone())])
            .lr(2.0e38)
            .momentum(0.9)
            .weight_decay(0.0),
    ];
    let mut opt = MuSgd::with_ortho(gs, MuSgdWeights { muon: 1.0, sgd: 0.0 }, MuonOrtho::Polar4).unwrap();
    let mut grads = folo_core::Grads::default();
    grads.insert(&p0, Tensor::zeros_with([17, 5], folo_core::DType::F32, dev).unwrap());
    grads.insert(&p1, Tensor::zeros_with([17, 5], folo_core::DType::F32, dev).unwrap());
    opt.step(&grads).unwrap();
    for (name, got, init) in [("w0", &p0, &p0_init), ("w1", &p1, &p1_init)] {
        let a = got.to_vec::<f32>().unwrap();
        let b = init.to_vec::<f32>().unwrap();
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name} must stay finite and unchanged when folded scale * lr overflows"
        );
    }
}

/// Legacy `Ns5` with multiple Muon groups at differing LRs (`decay` + `decay.cls`) and non-unit `muon_scale`
/// stays on the per-group path and matches separate per-group GPU calls bit for bit, and a pre-Polar `folo.optim`
/// contract omitting `muon_ortho` transitions a default-`Polar4` GPU `MuSgd` to `Ns5` deterministically.
#[test]
fn ns5_multi_group_matches_per_group_steps() {
    let dev = gpu!();
    let mut rng = Rng::seed(53);
    let shapes: [&[usize]; 3] = [&[13, 7, 3, 1], &[17, 5], &[9, 4, 2, 1]];
    let init: Vec<Tensor> = shapes.iter().map(|s| Tensor::rand_uniform(*s, -1.0, 1.0, &mut rng)).collect();
    let ps_opt: Vec<Tensor> = init.iter().map(|t| t.to_device(dev).unwrap().requires_grad_()).collect();
    let ps_res: Vec<Tensor> = init.iter().map(|t| t.to_device(dev).unwrap().requires_grad_()).collect();
    let ps_ref: Vec<Tensor> = init.iter().map(|t| t.to_device(dev).unwrap().requires_grad_()).collect();
    let ms_ref: Vec<Tensor> =
        shapes.iter().map(|s| Tensor::zeros_with(*s, folo_core::DType::F32, dev).unwrap()).collect();

    let weights = MuSgdWeights { muon: 0.2, sgd: 1.0 };
    let (lr0, lr1, mu, wd) = (0.02_f64, 0.06_f64, 0.937_f64, 5e-4_f64);
    let make_groups = |ps: &[Tensor]| {
        vec![
            ParamGroup::new("decay", vec![("w0".into(), ps[0].clone()), ("w1".into(), ps[1].clone())])
                .lr(lr0)
                .momentum(mu)
                .weight_decay(wd),
            ParamGroup::new("decay.cls", vec![("w2".into(), ps[2].clone())])
                .lr(lr1)
                .momentum(mu)
                .weight_decay(wd),
        ]
    };
    let mut opt = MuSgd::with_ortho(make_groups(&ps_opt), weights, MuonOrtho::Ns5).unwrap();
    // Simulate resuming a pre-Polar `folo.optim` contract (no `muon_ortho`) into a default-`Polar4` GPU `MuSgd`.
    let legacy_contract = opt.contract();
    assert_eq!(legacy_contract.muon_ortho, None);
    let mut resumed = MuSgd::with_weights(make_groups(&ps_res), weights).unwrap();
    assert_eq!(resumed.ortho(), MuonOrtho::Polar4);
    resumed.apply_contract(&legacy_contract).unwrap();
    assert_eq!(resumed.ortho(), MuonOrtho::Ns5);

    let mut sgd_ref = Sgd::new(make_groups(&ps_ref)).unwrap();

    for _ in 0..2 {
        let gs: Vec<Tensor> = shapes
            .iter()
            .map(|s| Tensor::rand_uniform(*s, -0.3, 0.3, &mut rng).to_device(dev).unwrap())
            .collect();
        let (mut g_opt, mut g_res, mut g_ref) =
            (folo_core::Grads::default(), folo_core::Grads::default(), folo_core::Grads::default());
        for (((p_opt, p_res), p_ref), g) in ps_opt.iter().zip(&ps_res).zip(&ps_ref).zip(&gs) {
            g_opt.insert(p_opt, g.clone());
            g_res.insert(p_res, g.clone());
            g_ref.insert(p_ref, g.clone());
        }
        opt.step(&g_opt).unwrap();
        resumed.step(&g_res).unwrap();

        // Separate per-group GPU calls: un-folded `scales[i] = muon_scale`, `settings.lr = muon * group_lr`.
        {
            let _ng = no_grad();
            let mu32 = mu as f32;
            let step_axpby = folo_core::backend::OptimStep::Axpby { a: mu32, b: 1.0 - mu32 };
            for (idxs, lr) in [
                (&[0usize, 1][..], (weights.muon * lr0) as f32),
                (&[2usize][..], (weights.muon * lr1) as f32),
            ] {
                let p_sl: Vec<&Tensor> = idxs.iter().map(|&i| &ps_ref[i]).collect();
                let m_sl: Vec<&Tensor> = idxs.iter().map(|&i| &ms_ref[i]).collect();
                let g_sl: Vec<&Tensor> = idxs.iter().map(|&i| &gs[i]).collect();
                let none: Vec<Vec<&Tensor>> = vec![Vec::new(); idxs.len()];
                assert!(Tensor::optimizer_step_many_(&m_sl, &g_sl, &none, &step_axpby).unwrap());
                let scales: Vec<f32> = p_sl.iter().map(|p| muon_scale(p.dims())).collect();
                assert!(scales.iter().all(|&s| s > 1.0));
                let settings = folo_core::backend::MuonSettings::from_schedule(
                    mu32,
                    true,
                    MuonOrtho::Ns5.schedule(),
                    false,
                    NS_EPS,
                    lr,
                );
                assert!(Tensor::muon_step_many_(&p_sl, &m_sl, &g_sl, &scales, &settings).unwrap());
            }
        }
        sgd_ref.step(&g_ref).unwrap();
    }
    for (i, (((p_opt, p_res), p_ref), shape)) in
        ps_opt.iter().zip(&ps_res).zip(&ps_ref).zip(&shapes).enumerate()
    {
        let got = p_opt.to_vec::<f32>().unwrap();
        let res_v = p_res.to_vec::<f32>().unwrap();
        let want = p_ref.to_vec::<f32>().unwrap();
        assert!(
            got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "Ns5 param {i} {shape:?} must match separate per-group GPU steps bit for bit",
        );
        assert!(
            res_v.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "pre-Polar contract GPU resume param {i} {shape:?} must match Ns5 bit for bit",
        );
    }
}
