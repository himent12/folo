use super::*;

fn ref_layer_norm(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    outer: usize,
    d: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; outer * d];
    let mut mean = vec![0.0f32; outer];
    let mut rstd = vec![0.0f32; outer];
    for o in 0..outer {
        let row = &x[o * d..(o + 1) * d];
        let m = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / d as f64;
        let r = 1.0 / (var + eps as f64).sqrt();
        mean[o] = m as f32;
        rstd[o] = r as f32;
        for j in 0..d {
            let mut val = (row[j] as f64 - m) * r;
            if let Some(w) = w {
                val *= w[j] as f64;
            }
            if let Some(b) = b {
                val += b[j] as f64;
            }
            y[o * d + j] = val as f32;
        }
    }
    (y, mean, rstd)
}

fn ref_rms_norm(x: &[f32], w: Option<&[f32]>, outer: usize, d: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; outer * d];
    let mut rstd = vec![0.0f32; outer];
    for o in 0..outer {
        let row = &x[o * d..(o + 1) * d];
        let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / d as f64;
        let r = 1.0 / (ms + eps as f64).sqrt();
        rstd[o] = r as f32;
        for j in 0..d {
            let mut val = (row[j] as f64) * r;
            if let Some(w) = w {
                val *= w[j] as f64;
            }
            y[o * d + j] = val as f32;
        }
    }
    (y, rstd)
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: len mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let diff = (x - y).abs();
        let lim = tol * y.abs().max(1.0);
        assert!(diff <= lim, "{ctx}[{i}]: got {x} vs {y} (diff {diff} > {lim})");
    }
}

#[test]
fn layer_norm_forward_and_backward_match_f64_reference() {
    let eps = 1e-5;
    for &(outer, d) in &[(1, 1), (1, 17), (1, 10_003), (5, 31), (128, 300), (0, 16)] {
        let x: Vec<f32> = (0..outer * d).map(|i| ((i as f32) * 0.23).sin() * 2.5 + 0.7).collect();
        let g: Vec<f32> = (0..outer * d).map(|i| ((i as f32) * 0.17).cos() * 1.5).collect();
        let w: Vec<f32> = (0..d).map(|j| 0.5 + ((j as f32) * 0.11).sin() * 0.4).collect();
        let b: Vec<f32> = (0..d).map(|j| ((j as f32) * 0.07).cos() * 0.3).collect();

        for (use_w, use_b) in [(true, true), (true, false), (false, true), (false, false)] {
            let wo = use_w.then_some(w.as_slice());
            let bo = use_b.then_some(b.as_slice());
            let (y, mean, rstd) = layer_norm(&x, wo, bo, outer, d, eps);
            let (ry, rm, rr) = ref_layer_norm(&x, wo, bo, outer, d, eps);
            assert_close(&y, &ry, 1e-4, &format!("ln fwd ({outer},{d}) w={use_w} b={use_b}"));
            assert_close(&mean, &rm, 1e-4, "ln mean");
            assert_close(&rstd, &rr, 1e-4, "ln rstd");

            let (dx, dw, db) = layer_norm_backward(&x, &g, wo, &mean, &rstd, outer, d, use_w, use_b);
            assert_eq!(dw.is_some(), use_w);
            assert_eq!(db.is_some(), use_b);
            // Reference backward in f64.
            let mut rdx = vec![0.0f32; outer * d];
            let mut rdw = vec![0.0f64; d];
            let mut rdb = vec![0.0f64; d];
            for o in 0..outer {
                let m = rm[o] as f64;
                let r = rr[o] as f64;
                let mut c1 = 0.0f64;
                let mut c2 = 0.0f64;
                for j in 0..d {
                    let xh = (x[o * d + j] as f64 - m) * r;
                    let gv = g[o * d + j] as f64;
                    let gw = if use_w { gv * w[j] as f64 } else { gv };
                    c1 += gw;
                    c2 += gw * xh;
                    rdw[j] += gv * xh;
                    rdb[j] += gv;
                }
                for j in 0..d {
                    let xh = (x[o * d + j] as f64 - m) * r;
                    let gv = g[o * d + j] as f64;
                    let gw = if use_w { gv * w[j] as f64 } else { gv };
                    rdx[o * d + j] = (r * (gw - (c1 + xh * c2) / d as f64)) as f32;
                }
            }
            assert_close(&dx, &rdx, 2e-4, &format!("ln dx ({outer},{d}) w={use_w} b={use_b}"));
            if let Some(dw) = dw {
                let rdw32: Vec<f32> = rdw.iter().map(|&v| v as f32).collect();
                assert_close(&dw, &rdw32, 2e-4, &format!("ln dw ({outer},{d})"));
            }
            if let Some(db) = db {
                let rdb32: Vec<f32> = rdb.iter().map(|&v| v as f32).collect();
                assert_close(&db, &rdb32, 2e-4, &format!("ln db ({outer},{d})"));
            }
        }
    }
}

#[test]
fn rms_norm_forward_and_backward_match_f64_reference() {
    let eps = 1e-5;
    for &(outer, d) in &[(1, 1), (1, 19), (1, 10_005), (7, 33), (128, 300), (0, 16)] {
        let x: Vec<f32> = (0..outer * d).map(|i| ((i as f32) * 0.29).sin() * 2.0 - 0.3).collect();
        let g: Vec<f32> = (0..outer * d).map(|i| ((i as f32) * 0.13).cos() * 1.2).collect();
        let w: Vec<f32> = (0..d).map(|j| 0.7 + ((j as f32) * 0.19).sin() * 0.3).collect();

        for use_w in [true, false] {
            let wo = use_w.then_some(w.as_slice());
            let (y, rstd) = rms_norm(&x, wo, outer, d, eps);
            let (ry, rr) = ref_rms_norm(&x, wo, outer, d, eps);
            assert_close(&y, &ry, 1e-4, &format!("rms fwd ({outer},{d}) w={use_w}"));
            assert_close(&rstd, &rr, 1e-4, "rms rstd");

            let (dx, dw) = rms_norm_backward(&x, &g, wo, &rstd, outer, d, use_w);
            assert_eq!(dw.is_some(), use_w);
            let mut rdx = vec![0.0f32; outer * d];
            let mut rdw = vec![0.0f64; d];
            for o in 0..outer {
                let r = rr[o] as f64;
                let mut c = 0.0f64;
                for j in 0..d {
                    let xh = x[o * d + j] as f64 * r;
                    let gv = g[o * d + j] as f64;
                    let gw = if use_w { gv * w[j] as f64 } else { gv };
                    c += gw * xh;
                    rdw[j] += gv * xh;
                }
                for j in 0..d {
                    let xh = x[o * d + j] as f64 * r;
                    let gv = g[o * d + j] as f64;
                    let gw = if use_w { gv * w[j] as f64 } else { gv };
                    rdx[o * d + j] = (r * (gw - xh * (c / d as f64))) as f32;
                }
            }
            assert_close(&dx, &rdx, 2e-4, &format!("rms dx ({outer},{d}) w={use_w}"));
            if let Some(dw) = dw {
                let rdw32: Vec<f32> = rdw.iter().map(|&v| v as f32).collect();
                assert_close(&dw, &rdw32, 2e-4, &format!("rms dw ({outer},{d})"));
            }
        }
    }
}

#[test]
fn norm_backward_all_param_combos_tails_and_grad_check() {
    use crate::Tensor;
    use crate::rng::Rng;

    // Exercise all (w, dw, db) specializations across 0, <8, 8, 9..15, 16, 17, 33 lengths.
    for &d in &[1usize, 7, 8, 9, 15, 16, 17, 32, 33] {
        let outer = 3usize;
        let x: Vec<f32> = (0..outer * d).map(|i| ((i as f32) * 0.31).sin() * 1.7 + 0.2).collect();
        let g: Vec<f32> = (0..outer * d).map(|i| ((i as f32) * 0.19).cos() * 1.1).collect();
        let w: Vec<f32> = (0..d).map(|j| 0.8 + ((j as f32) * 0.13).sin() * 0.3).collect();
        let b: Vec<f32> = (0..d).map(|j| ((j as f32) * 0.09).cos() * 0.2).collect();
        let (_, mean, rstd_ln) = layer_norm(&x, Some(&w), Some(&b), outer, d, 1e-5);
        let (_, rstd_rms) = rms_norm(&x, Some(&w), outer, d, 1e-5);

        for has_w in [false, true] {
            let wo = has_w.then_some(w.as_slice());
            for need_w in [false, true] {
                for need_b in [false, true] {
                    let (dx, dw, db) =
                        layer_norm_backward(&x, &g, wo, &mean, &rstd_ln, outer, d, need_w, need_b);
                    assert_eq!(dx.len(), outer * d);
                    assert_eq!(dw.is_some(), need_w);
                    assert_eq!(db.is_some(), need_b);
                    // Direct parity check against `ln_bwd_write_scalar` on row 0.
                    let (c1, c2) =
                        ln_bwd_coeffs_row(&x[..d], &g[..d], wo, mean[0], rstd_ln[0], 1.0 / d as f32, false);
                    let mut ref_dx = vec![0.0f32; d];
                    let mut ref_dw = need_w.then(|| vec![0.0f32; d]);
                    let mut ref_db = need_b.then(|| vec![0.0f32; d]);
                    ln_bwd_write_scalar(
                        &x[..d],
                        &g[..d],
                        wo,
                        &mut ref_dx,
                        ref_dw.as_deref_mut(),
                        ref_db.as_deref_mut(),
                        mean[0],
                        rstd_ln[0],
                        c1,
                        c2,
                    );
                    assert_close(
                        &dx[..d],
                        &ref_dx,
                        1e-5,
                        &format!("ln_bwd_write dx d={d} w={has_w} dw={need_w} db={need_b}"),
                    );
                }
                let (dx, dw) = rms_norm_backward(&x, &g, wo, &rstd_rms, outer, d, need_w);
                assert_eq!(dx.len(), outer * d);
                assert_eq!(dw.is_some(), need_w);
                let c_d = super::rms::rms_bwd_c_row_for_test(&x[..d], &g[..d], wo, rstd_rms[0]) / d as f32;
                let mut ref_dx = vec![0.0f32; d];
                let mut ref_dw = need_w.then(|| vec![0.0f32; d]);
                super::rms::rms_bwd_write_scalar_for_test(
                    &x[..d],
                    &g[..d],
                    wo,
                    &mut ref_dx,
                    ref_dw.as_deref_mut(),
                    rstd_rms[0],
                    c_d,
                );
                assert_close(
                    &dx[..d],
                    &ref_dx,
                    1e-5,
                    &format!("rms_bwd_write dx d={d} w={has_w} dw={need_w}"),
                );
            }
        }
    }

    // Finite-difference grad_check on both aligned (32) and ragged (23) widths.
    let mut rng = Rng::seed(404);
    for &d in &[23usize, 32] {
        let x = Tensor::randn([3, d], 0.0, 1.0, &mut rng).requires_grad_();
        let w = Tensor::randn([d], 1.0, 0.2, &mut rng).requires_grad_();
        let b = Tensor::randn([d], 0.0, 0.2, &mut rng).requires_grad_();
        crate::testing::grad_check(
            |v| v[0].layer_norm(Some(&v[1]), Some(&v[2]), 1e-5),
            &[x.clone(), w.clone(), b.clone()],
        )
        .unwrap();
        crate::testing::grad_check(|v| v[0].layer_norm(None, None, 1e-5), std::slice::from_ref(&x)).unwrap();
        crate::testing::grad_check(|v| v[0].rms_norm(Some(&v[1]), 1e-5), &[x.clone(), w]).unwrap();
        crate::testing::grad_check(|v| v[0].rms_norm(None, 1e-5), &[x]).unwrap();
    }
}
