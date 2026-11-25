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
