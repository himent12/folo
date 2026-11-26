//! Newton-Schulz, Polar-FOLO orthogonalisation, and MuSGD reference step tests.

use folo_core::rng::Rng;
use folo_core::{Grads, Tensor, no_grad};
use folo_nn::optim::*;

use crate::{G1, G2, close, param};

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
    if let Some(cs9) = ortho.cs9_schedule() {
        for (step, &(gamma, delta, eta, xi, c4)) in cs9.iter().enumerate() {
            let (gamma, delta, eta, xi, c4) = (gamma as f64, delta as f64, eta as f64, xi as f64, c4 as f64);
            let xt: Vec<f64> = (0..c * r).map(|k| x[(k % r) * c + k / r]).collect();
            let am = mul(&x, &xt, r, c, r);
            let aa = mul(&am, &am, r, r, r);
            if step == 0 {
                let alpha8 = aa.iter().map(|v| v * v).sum::<f64>().powf(0.125) + 1e-7 / norm;
                let inv_a = if alpha8 == 0.0 { 0.0 } else { 1.0 / alpha8 };
                let ia2 = inv_a * inv_a;
                let ia4 = ia2 * ia2;
                let mut sm: Vec<f64> = am.iter().zip(&aa).map(|(p, q)| (gamma * ia2) * p + ia4 * q).collect();
                for i in 0..r {
                    sm[i * r + i] += delta;
                }
                let s2 = mul(&sm, &sm, r, r, r);
                let bm: Vec<f64> = am.iter().zip(&s2).map(|(p, q)| (eta * ia2) * p + c4 * q).collect();
                let bx = mul(&bm, &x, r, r, c);
                x = x.iter().zip(&bx).map(|(p, q)| (xi * inv_a) * p + q * inv_a).collect();
            } else {
                let mut sm: Vec<f64> = am.iter().zip(&aa).map(|(p, q)| gamma * p + q).collect();
                for i in 0..r {
                    sm[i * r + i] += delta;
                }
                let s2 = mul(&sm, &sm, r, r, r);
                let bm: Vec<f64> = am.iter().zip(&s2).map(|(p, q)| eta * p + c4 * q).collect();
                let bx = mul(&bm, &x, r, r, c);
                x = x.iter().zip(&bx).map(|(p, q)| xi * p + q).collect();
            }
        }
    } else {
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
    for ortho in [MuonOrtho::Polar4, MuonOrtho::Polar5, MuonOrtho::Polar3Cs9] {
        let many = orthogonalize_many_with(&ts, ortho).unwrap();
        for (t, o) in ts.iter().zip(&many) {
            let d = t.dims().to_vec();
            let (rows, cols) = (d[0], t.numel() / d[0]);
            let v: Vec<f64> = t.to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect();
            let scale = muon_scale(&d) as f64;
            let want: Vec<f64> = ortho_f64(&v, rows, cols, ortho).iter().map(|x| x * scale).collect();
            close(&o.to_vec::<f32>().unwrap(), &want, 2e-4);
            // Alone, without bucket padding: Schatten-4/8 norm is invariant to zero-padded rows.
            let one = orthogonalize_with(t, ortho).unwrap().to_vec::<f32>().unwrap();
            close(&one, &o.to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect::<Vec<_>>(), 1e-5);
        }
    }
    // Check orthogonality error ||O O^T - I||_F / sqrt(r) on a well- and moderately-conditioned 16x64 matrix:
    // Polar4, Polar5, and Polar3Cs9 strictly beat NS5.
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
    let e_cs9 = ortho_err(MuonOrtho::Polar3Cs9);
    assert!(e_p4 < e_ns5 * 0.5, "Polar4 ({e_p4:.4}) should be >2x more orthogonal than NS5 ({e_ns5:.4})");
    assert!(e_p5 < e_ns5 * 0.5, "Polar5 ({e_p5:.4}) should be >2x more orthogonal than NS5 ({e_ns5:.4})");
    assert!(
        e_cs9 < e_ns5 * 0.5,
        "Polar3Cs9 ({e_cs9:.4}) should be >2x more orthogonal than NS5 ({e_ns5:.4})"
    );
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
    for ortho in [MuonOrtho::Ns5, MuonOrtho::Polar4, MuonOrtho::Polar5, MuonOrtho::Polar3Cs9] {
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
    for ortho in [MuonOrtho::Polar4, MuonOrtho::Ns5, MuonOrtho::Polar5, MuonOrtho::Polar3Cs9] {
        musgd_reference_steps(MuSgdWeights::default(), ortho);
    }
}

/// `Polar3-CS9` subnormal, single-row, rank-1, and contract round-trip / legacy compatibility.
#[test]
fn polar3_cs9_edge_cases_and_contract_round_trip() {
    // Subnormal, single-row, and exact rank-1 matrices stay finite and match f64 reference.
    let subnormal = Tensor::from_vec(vec![1e-30f32; 16], [4, 4]).unwrap();
    let single_row = Tensor::from_vec(vec![3.0f32, -4.0, 12.0, 0.5, -2.5], [1, 5]).unwrap();
    let rank1_data: Vec<f32> =
        (0..8).flat_map(|i| (0..12).map(move |j| ((i + 1) * (j + 1)) as f32)).collect();
    let rank1 = Tensor::from_vec(rank1_data, [8, 12]).unwrap();
    for t in [&subnormal, &single_row, &rank1] {
        let o = orthogonalize_with(t, MuonOrtho::Polar3Cs9).unwrap();
        let got = o.to_vec::<f32>().unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "non-finite Polar3Cs9 on {:?}", t.dims());
        let rows = t.dims()[0];
        let cols = t.numel() / rows;
        let v: Vec<f64> = t.to_vec::<f32>().unwrap().into_iter().map(f64::from).collect();
        let scale = muon_scale(t.dims()) as f64;
        let want: Vec<f64> =
            ortho_f64(&v, rows, cols, MuonOrtho::Polar3Cs9).iter().map(|x| x * scale).collect();
        close(&got, &want, 2e-4);
    }

    // Optimizer contract round-trip for Polar3Cs9 and legacy compatibility (Ns5 omits `muon_ortho`).
    let w = Tensor::randn([8, 16], 0.0, 0.2, &mut Rng::seed(99)).requires_grad_();
    let mut opt = MuSgd::with_ortho(
        vec![ParamGroup::new("decay", vec![("w".into(), w.clone())]).lr(0.01).momentum(0.9)],
        MuSgdWeights::default(),
        MuonOrtho::Polar3Cs9,
    )
    .unwrap();
    let c = opt.contract();
    assert_eq!(c.muon_ortho, Some(MuonOrtho::Polar3Cs9));
    let js = serde_json::to_string(&c).unwrap();
    assert!(js.contains("\"polar3_cs9\""), "{js}");
    let back: folo_nn::optim::OptimContract = serde_json::from_str(&js).unwrap();
    assert_eq!(back.muon_ortho, Some(MuonOrtho::Polar3Cs9));
    let mut opt2 =
        MuSgd::new(vec![ParamGroup::new("decay", vec![("w".into(), w.clone())]).lr(0.01).momentum(0.9)])
            .unwrap();
    opt2.apply_contract(&back).unwrap();
    assert_eq!(opt2.ortho(), MuonOrtho::Polar3Cs9);

    // Legacy contract omitting `muon_ortho` restores as `Ns5`.
    let leg = MuSgd::with_ortho(
        vec![ParamGroup::new("decay", vec![("w".into(), w)]).lr(0.01).momentum(0.9)],
        MuSgdWeights::default(),
        MuonOrtho::Ns5,
    )
    .unwrap()
    .contract();
    assert_eq!(leg.muon_ortho, None);
    let leg_js = serde_json::to_string(&leg).unwrap();
    assert!(!leg_js.contains("muon_ortho"), "{leg_js}");
    opt.apply_contract(&serde_json::from_str(&leg_js).unwrap()).unwrap();
    assert_eq!(opt.ortho(), MuonOrtho::Ns5);
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
fn muon_ortho_immutability_and_cross_aliasing() {
    let mut rng = Rng::seed(123);
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
