//! MuSGD's fused Muon update (`Tensor::muon_step_many_`) against the tensor ops (the Nesterov mix as tensor ops,
//! `orthogonalize_many`, then `p · 1 + O · (−lr)`, all on the CPU). `lr = 1`, so the parameters carry `O` at
//! full size; values agree to `2e-4 + 2e-3·|ref|`, NaNs exactly where the reference has them.

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
fn fused(dev: Device, ps: &[Tensor], ms: &[Tensor], gs: &[Tensor], nesterov: bool) -> Vec<Tensor> {
    let up = |ts: &[Tensor]| ts.iter().map(|t| t.to_device(dev).unwrap()).collect::<Vec<_>>();
    let (pg, mg, gg) = (up(ps), up(ms), up(gs));
    let scales: Vec<f32> = ms.iter().map(|m| muon_scale(m.dims())).collect();
    let settings = folo_core::backend::MuonSettings {
        momentum: 0.9,
        nesterov,
        coeffs: NS_COEFFS,
        steps: NS_STEPS,
        eps: NS_EPS,
        lr: 1.0,
    };
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
fn reference(ps: &[Tensor], ms: &[Tensor], gs: &[Tensor], nesterov: bool) -> Vec<Tensor> {
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
    let os = orthogonalize_many(&us).unwrap();
    ps.iter().zip(&os).map(|(p, o)| p.add(&o.mul_scalar(-1.0).unwrap()).unwrap()).collect()
}

/// Wide and tall matrices (tall ones are transposed), several 64×64 tiles along both sides, single rows and
/// columns, a square one, scales other than 1, and more matrices than one multi-tensor dispatch takes.
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
    for nesterov in [false, true] {
        let got = fused(dev, &ps, &ms, &gs, nesterov);
        let want = reference(&ps, &ms, &gs, nesterov);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            close_or_nan(g, w, &format!("nesterov {nesterov}: matrix {i} {:?}", ps[i].dims()));
        }
    }
}

/// A zero momentum and gradient (the eps keeps `0 / ‖0‖` finite: `p` stays as it was), one near zero (its norm
/// about the eps), one of 184,320 elements (720 partial sums: the norm's loop runs three times per invocation),
/// a NaN in one gradient (that parameter turns NaN where the tensor ops' does, with Nesterov; nothing else does)
/// and an empty one, left alone.
#[test]
fn fused_muon_edge_cases() {
    let dev = gpu!();
    let rng = &mut Rng::seed(12);
    let shapes: [&[usize]; 6] = [&[8, 12], &[20, 30], &[160, 1152], &[24, 40], &[16, 3, 3, 3], &[0, 5]];
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
    for nesterov in [false, true] {
        let got = fused(dev, &ps, &ms, &gs, nesterov);
        let want = reference(&ps, &ms, &gs, nesterov);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            close_or_nan(g, w, &format!("nesterov {nesterov}: matrix {i} {:?}", ps[i].dims()));
        }
        assert_eq!(got[0].to_vec::<f32>().unwrap(), ps[0].to_vec::<f32>().unwrap(), "zero: p unchanged");
        let (a, b) = (got[1].to_vec::<f32>().unwrap(), ps[1].to_vec::<f32>().unwrap());
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3), "near zero: still a step");
        let nans = got[3].to_vec::<f32>().unwrap().iter().filter(|x| x.is_nan()).count();
        assert_eq!(nans, if nesterov { 24 * 40 } else { 0 }, "nesterov {nesterov}: NaN gradient");
        assert_eq!(got[5].dims(), [0, 5]);
    }
}
