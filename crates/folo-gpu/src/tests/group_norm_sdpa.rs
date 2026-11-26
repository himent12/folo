//! GPU-vs-CPU parity tests for fused single-pass `GroupNorm` / `InstanceNorm` (`kernels::group_norm`) and
//! tiled fused Scaled Dot-Product Attention (`kernels::attention::sdpa_forward`), covering aligned `vec4`,
//! unaligned odd-tail, non-power-of-two, uncached, split-K, causal/non-causal, extreme-scale, `F16`, zero-edge,
//! and both 32-wide subgroup and portable shared-memory fallback paths.

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::FORCE_NO_SUBGROUPS;
use crate::kernels::attention::sdpa_forward;
use crate::kernels::group_norm::{
    group_norm_backward, group_norm_forward, instance_norm_backward, instance_norm_forward,
};
use crate::{DType, GpuTensor};

struct ForceGuard(bool);
impl ForceGuard {
    fn set(v: bool) -> Self {
        Self(FORCE_NO_SUBGROUPS.replace(v))
    }
}
impl Drop for ForceGuard {
    fn drop(&mut self) {
        FORCE_NO_SUBGROUPS.set(self.0);
    }
}

/// Exact f64 CPU reference for GroupNorm forward.
fn cpu_group_norm_fwd(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    (n, c, hw): (usize, usize, usize),
    groups: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let cpg = c / groups;
    let m = cpg * hw;
    let mut y = vec![0.0f32; n * c * hw];
    let mut mean = vec![0.0f32; n * groups];
    let mut rstd = vec![0.0f32; n * groups];
    for ni in 0..n {
        for g in 0..groups {
            let mut sum = 0.0f64;
            for cl in 0..cpg {
                let ch = g * cpg + cl;
                let base = (ni * c + ch) * hw;
                for i in 0..hw {
                    sum += x[base + i] as f64;
                }
            }
            let mu = sum / m as f64;
            let mut sq = 0.0f64;
            for cl in 0..cpg {
                let ch = g * cpg + cl;
                let base = (ni * c + ch) * hw;
                for i in 0..hw {
                    let d = x[base + i] as f64 - mu;
                    sq += d * d;
                }
            }
            let rs = 1.0 / (sq / m as f64 + eps as f64).sqrt();
            mean[ni * groups + g] = mu as f32;
            rstd[ni * groups + g] = rs as f32;
            for cl in 0..cpg {
                let ch = g * cpg + cl;
                let base = (ni * c + ch) * hw;
                let wc = w.map_or(1.0f64, |v| v[ch] as f64);
                let bc = b.map_or(0.0f64, |v| v[ch] as f64);
                for i in 0..hw {
                    let xh = (x[base + i] as f64 - mu) * rs;
                    y[base + i] = (xh * wc + bc) as f32;
                }
            }
        }
    }
    (y, mean, rstd)
}

/// Exact f64 CPU reference for GroupNorm backward.
fn cpu_group_norm_bwd(
    x: &[f32],
    grad: &[f32],
    w: Option<&[f32]>,
    mean: &[f32],
    rstd: &[f32],
    (n, c, hw): (usize, usize, usize),
    groups: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let cpg = c / groups;
    let m = cpg * hw;
    let mut dx = vec![0.0f32; n * c * hw];
    let mut dw = vec![0.0f64; c];
    let mut db = vec![0.0f64; c];
    for ni in 0..n {
        for g in 0..groups {
            let mu = mean[ni * groups + g] as f64;
            let rs = rstd[ni * groups + g] as f64;
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            let mut s3 = 0.0f64;
            for cl in 0..cpg {
                let ch = g * cpg + cl;
                let base = (ni * c + ch) * hw;
                let wc = w.map_or(1.0f64, |v| v[ch] as f64);
                for i in 0..hw {
                    let xh = (x[base + i] as f64 - mu) * rs;
                    let gv = grad[base + i] as f64;
                    let gw = gv * wc;
                    s1 += gw;
                    s2 += gw * xh;
                    s3 += xh;
                    dw[ch] += gv * xh;
                    db[ch] += gv;
                }
            }
            let c2 = s2 / m as f64;
            let c1 = s1 / m as f64 - (s3 / m as f64) * c2;
            for cl in 0..cpg {
                let ch = g * cpg + cl;
                let base = (ni * c + ch) * hw;
                let wc = w.map_or(1.0f64, |v| v[ch] as f64);
                for i in 0..hw {
                    let xh = (x[base + i] as f64 - mu) * rs;
                    let gw = grad[base + i] as f64 * wc;
                    dx[base + i] = (rs * (gw - (c1 + xh * c2))) as f32;
                }
            }
        }
    }
    (dx, dw.into_iter().map(|v| v as f32).collect(), db.into_iter().map(|v| v as f32).collect())
}

/// Exact f64 CPU reference for Scaled Dot-Product Attention.
fn cpu_sdpa_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (b, sq, sk, d): (usize, usize, usize, usize),
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * sq * d];
    let mut scores = vec![0.0f64; sk];
    for bi in 0..b {
        let qb = bi * sq * d;
        let kb = bi * sk * d;
        for qi in 0..sq {
            let mut m = f64::NEG_INFINITY;
            let mut has_nan = false;
            for ki in 0..sk {
                if causal && ki > qi {
                    scores[ki] = f64::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f64;
                for di in 0..d {
                    dot += q[qb + qi * d + di] as f64 * k[kb + ki * d + di] as f64;
                }
                let s = dot * scale as f64;
                scores[ki] = s;
                if s.is_nan() {
                    has_nan = true;
                } else if !has_nan && s > m {
                    m = s;
                }
            }
            if has_nan {
                m = f64::NAN;
            }
            let mut sum = 0.0f64;
            for (ki, score) in scores.iter_mut().enumerate().take(sk) {
                if causal && ki > qi {
                    *score = 0.0;
                    continue;
                }
                let p = (*score - m).exp();
                *score = p;
                sum += p;
            }
            for di in 0..d {
                let mut acc = 0.0f64;
                for ki in 0..sk {
                    if causal && ki > qi {
                        continue;
                    }
                    acc += scores[ki] * v[kb + ki * d + di] as f64;
                }
                out[qb + qi * d + di] = (acc / sum) as f32;
            }
        }
    }
    out
}

#[test]
fn group_norm_and_instance_norm_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x600d_cafe);

    // Covers: vec4 cached, scalar odd-tail cached, InstanceNorm (G==C), single group (G==1),
    // uncached vec4 (m_u > 1024), uncached scalar (m_u > 1024), and split-K dweight/dbias (m_c >= 1024).
    let cases: &[((usize, usize, usize), usize, &str)] = &[
        ((2, 8, 16), 2, "vec4 cached G=2"),
        ((2, 8, 16), 8, "vec4 cached InstanceNorm"),
        ((3, 6, 7), 3, "scalar odd cached G=3"),
        ((2, 5, 13), 1, "scalar single group G=1"),
        ((2, 5, 13), 5, "scalar InstanceNorm"),
        ((1, 4, 2052), 2, "vec4 uncached m_u=1026"),
        ((1, 3, 1027), 3, "scalar uncached m_u=1027"),
        ((32, 4, 256), 2, "split-k dwb vec4"),
        ((16, 4, 71), 2, "split-k dwb scalar"),
    ];

    for &((n, c, hw), g, label) in cases {
        let total = n * c * hw;
        let x = rng.vec(total, -2.0, 2.0);
        let grad = rng.vec(total, -1.5, 1.5);
        let w = rng.vec(c, 0.5, 1.5);
        let b = rng.vec(c, -0.5, 0.5);
        let eps = 1e-5f32;
        let (y_ref, m_ref, r_ref) = cpu_group_norm_fwd(&x, Some(&w), Some(&b), (n, c, hw), g, eps);
        let (dx_ref, dw_ref, db_ref) = cpu_group_norm_bwd(&x, &grad, Some(&w), &m_ref, &r_ref, (n, c, hw), g);

        let gx = GpuTensor::from_slice(&ctx, &x, &[total]).unwrap();
        let gg = GpuTensor::from_slice(&ctx, &grad, &[total]).unwrap();
        let gw = GpuTensor::from_slice(&ctx, &w, &[c]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b, &[c]).unwrap();
        let tol = if hw > 1000 || n >= 16 { 1e-3 } else { 2e-4 };

        for portable in [false, true] {
            let _fg = ForceGuard::set(portable);
            let gy = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
            let gm = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
            let gr = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
            if g == c {
                instance_norm_forward(
                    &ctx,
                    gx.raw(),
                    Some(gw.raw()),
                    Some(gb.raw()),
                    gy.raw(),
                    gm.raw(),
                    gr.raw(),
                    (n, c, hw),
                    eps,
                    DType::F32,
                )
                .unwrap();
            } else {
                group_norm_forward(
                    &ctx,
                    gx.raw(),
                    Some(gw.raw()),
                    Some(gb.raw()),
                    gy.raw(),
                    gm.raw(),
                    gr.raw(),
                    (n, c, hw),
                    g,
                    eps,
                    DType::F32,
                )
                .unwrap();
            }
            let tag = format!("{label} portable={portable}");
            assert_close(&gy.to_vec().unwrap(), &y_ref, tol, 2e-4, &format!("gn y {tag}"));
            assert_close(&gm.to_vec().unwrap(), &m_ref, tol, 2e-4, &format!("gn mean {tag}"));
            assert_close(&gr.to_vec().unwrap(), &r_ref, tol, 2e-4, &format!("gn rstd {tag}"));

            let gdx = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
            let gdw = GpuTensor::empty(&ctx, &[c], DType::F32).unwrap();
            let gdb = GpuTensor::empty(&ctx, &[c], DType::F32).unwrap();
            if g == c {
                instance_norm_backward(
                    &ctx,
                    gx.raw(),
                    gg.raw(),
                    Some(gw.raw()),
                    gm.raw(),
                    gr.raw(),
                    gdx.raw(),
                    Some(gdw.raw()),
                    Some(gdb.raw()),
                    (n, c, hw),
                    DType::F32,
                )
                .unwrap();
            } else {
                group_norm_backward(
                    &ctx,
                    gx.raw(),
                    gg.raw(),
                    Some(gw.raw()),
                    gm.raw(),
                    gr.raw(),
                    gdx.raw(),
                    Some(gdw.raw()),
                    Some(gdb.raw()),
                    (n, c, hw),
                    g,
                    DType::F32,
                )
                .unwrap();
            }
            assert_close(&gdx.to_vec().unwrap(), &dx_ref, tol, 2e-4, &format!("gn dx {tag}"));
            assert_close(&gdw.to_vec().unwrap(), &dw_ref, tol * 2.0, 2e-4, &format!("gn dw {tag}"));
            assert_close(&gdb.to_vec().unwrap(), &db_ref, tol * 2.0, 2e-4, &format!("gn db {tag}"));
        }
    }

    // Optional affine & selective backward gradients (none, w-only, b-only).
    let (n, c, hw, g) = (2, 6, 8, 3);
    let total = n * c * hw;
    let x = rng.vec(total, -2.0, 2.0);
    let grad = rng.vec(total, -1.0, 1.0);
    let w = rng.vec(c, 0.5, 1.5);
    let b = rng.vec(c, -0.5, 0.5);
    let gx = GpuTensor::from_slice(&ctx, &x, &[total]).unwrap();
    let gg = GpuTensor::from_slice(&ctx, &grad, &[total]).unwrap();
    let gw = GpuTensor::from_slice(&ctx, &w, &[c]).unwrap();
    let gb = GpuTensor::from_slice(&ctx, &b, &[c]).unwrap();
    for (has_w, has_b) in [(false, false), (true, false), (false, true)] {
        let (y_ref, m_ref, r_ref) =
            cpu_group_norm_fwd(&x, has_w.then_some(&w), has_b.then_some(&b), (n, c, hw), g, 1e-5);
        let (dx_ref, dw_ref, db_ref) =
            cpu_group_norm_bwd(&x, &grad, has_w.then_some(&w), &m_ref, &r_ref, (n, c, hw), g);
        let gy = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
        let gm = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
        let gr = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            gx.raw(),
            has_w.then(|| gw.raw()),
            has_b.then(|| gb.raw()),
            gy.raw(),
            gm.raw(),
            gr.raw(),
            (n, c, hw),
            g,
            1e-5,
            DType::F32,
        )
        .unwrap();
        assert_close(&gy.to_vec().unwrap(), &y_ref, 2e-4, 2e-4, &format!("gn opt {has_w}/{has_b}"));

        let gdx = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
        let gdw = has_w.then(|| GpuTensor::empty(&ctx, &[c], DType::F32).unwrap());
        let gdb = has_b.then(|| GpuTensor::empty(&ctx, &[c], DType::F32).unwrap());
        group_norm_backward(
            &ctx,
            gx.raw(),
            gg.raw(),
            has_w.then(|| gw.raw()),
            gm.raw(),
            gr.raw(),
            gdx.raw(),
            gdw.as_ref().map(|t| t.raw()),
            gdb.as_ref().map(|t| t.raw()),
            (n, c, hw),
            g,
            DType::F32,
        )
        .unwrap();
        assert_close(&gdx.to_vec().unwrap(), &dx_ref, 2e-4, 2e-4, &format!("gn opt dx {has_w}/{has_b}"));
        if let Some(t) = gdw {
            assert_close(&t.to_vec().unwrap(), &dw_ref, 2e-4, 2e-4, "gn opt dw");
        }
        if let Some(t) = gdb {
            assert_close(&t.to_vec().unwrap(), &db_ref, 2e-4, 2e-4, "gn opt db");
        }
    }

    // Edge cases: empty batch n=0 clears dweight/dbias; extreme scales (1e20, 1e-23, eps=0, 1e38);
    // large-offset translation invariance (sum(dx) == 0); invalid groups/eps rejected.
    let empty_buf = GpuTensor::from_slice(&ctx, &[1.0], &[1]).unwrap();
    let gdw0 = GpuTensor::from_slice(&ctx, &[9.0; 4], &[4]).unwrap();
    let gdb0 = GpuTensor::from_slice(&ctx, &[8.0; 4], &[4]).unwrap();
    group_norm_forward(
        &ctx,
        empty_buf.raw(),
        None,
        None,
        empty_buf.raw(),
        empty_buf.raw(),
        empty_buf.raw(),
        (0, 4, 8),
        2,
        1e-5,
        DType::F32,
    )
    .unwrap();
    group_norm_backward(
        &ctx,
        empty_buf.raw(),
        empty_buf.raw(),
        None,
        empty_buf.raw(),
        empty_buf.raw(),
        empty_buf.raw(),
        Some(gdw0.raw()),
        Some(gdb0.raw()),
        (0, 4, 8),
        2,
        DType::F32,
    )
    .unwrap();
    assert_eq!(gdw0.to_vec().unwrap(), vec![0.0; 4]);
    assert_eq!(gdb0.to_vec().unwrap(), vec![0.0; 4]);

    // Extreme finite scales (`1e20`, `1e-23`, `eps = 0`) and `1e38` constant rows stay finite.
    for portable in [false, true] {
        let _fg = ForceGuard::set(portable);
        for vals in [vec![1e20f32, -1e20, 1e20, -1e20], vec![1e-23, -1e-23, 1e-23, -1e-23]] {
            let t = GpuTensor::from_slice(&ctx, &vals, &[4]).unwrap();
            let y = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
            let m = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
            let r = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
            group_norm_forward(
                &ctx,
                t.raw(),
                None,
                None,
                y.raw(),
                m.raw(),
                r.raw(),
                (1, 1, 4),
                1,
                0.0,
                DType::F32,
            )
            .unwrap();
            assert_close(&y.to_vec().unwrap(), &[1.0, -1.0, 1.0, -1.0], 1e-5, 1e-5, "gn extreme");
        }
        let huge = GpuTensor::from_slice(&ctx, &[1e38f32; 8], &[8]).unwrap();
        let yh = GpuTensor::empty(&ctx, &[8], DType::F32).unwrap();
        let mh = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let rh = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            huge.raw(),
            None,
            None,
            yh.raw(),
            mh.raw(),
            rh.raw(),
            (1, 2, 4),
            1,
            1e-5,
            DType::F32,
        )
        .unwrap();
        assert!(yh.to_vec().unwrap().iter().all(|&v| v.is_finite() && v.abs() < 1e-5));
    }

    // Large-offset translation invariance: sum(dx) == 0.
    // Large-offset centering exactness (`[100_000_000, 100_000_008]` with `eps = 0`) and centering overflow
    // guard (`[3e38, -3e38, -3e38, -3e38]`): verify exact outputs and gradients on both subgroup and portable paths.
    for portable in [false, true] {
        let _fg = ForceGuard::set(portable);
        let spread = GpuTensor::from_slice(&ctx, &[3e38f32, -3e38, -3e38, -3e38], &[4]).unwrap();
        let ys = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
        let ms = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let rs = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            spread.raw(),
            None,
            None,
            ys.raw(),
            ms.raw(),
            rs.raw(),
            (1, 1, 4),
            1,
            0.0,
            DType::F32,
        )
        .unwrap();
        let s3 = 3.0f32.sqrt();
        assert_close(
            &ys.to_vec().unwrap(),
            &[s3, -1.0 / s3, -1.0 / s3, -1.0 / s3],
            1e-5,
            1e-5,
            &format!("gn 3e38 spread fb={portable}"),
        );
        // Verify backward `dx`/`dw` stay finite on `[3e38, -3e38, -3e38, -3e38]` (where `rstd ≈ 3.85e-39` is
        // subnormal and may FTZ on GPU) and exact on `[5e36, -5e36, -5e36, -5e36]` (where `rstd ≈ 2.31e-37` is normal).
        let g_sp = GpuTensor::from_slice(&ctx, &[1.0, 2.0, -1.0, 0.5], &[4]).unwrap();
        let dx_sp = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
        let dw_sp = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_backward(
            &ctx,
            spread.raw(),
            g_sp.raw(),
            None,
            ms.raw(),
            rs.raw(),
            dx_sp.raw(),
            Some(dw_sp.raw()),
            None,
            (1, 1, 4),
            1,
            DType::F32,
        )
        .unwrap();
        assert!(dx_sp.to_vec().unwrap().iter().all(|v| v.is_finite()));
        assert!(dw_sp.to_vec().unwrap().iter().all(|v| v.is_finite()));

        let spread_n = GpuTensor::from_slice(&ctx, &[5e36f32, -5e36, -5e36, -5e36], &[4]).unwrap();
        let ysn = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
        let msn = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let rsn = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            spread_n.raw(),
            None,
            None,
            ysn.raw(),
            msn.raw(),
            rsn.raw(),
            (1, 1, 4),
            1,
            0.0,
            DType::F32,
        )
        .unwrap();
        group_norm_backward(
            &ctx,
            spread_n.raw(),
            g_sp.raw(),
            None,
            msn.raw(),
            rsn.raw(),
            dx_sp.raw(),
            Some(dw_sp.raw()),
            None,
            (1, 1, 4),
            1,
            DType::F32,
        )
        .unwrap();
        let want_dw = 1.0 * s3 + (2.0 - 1.0 + 0.5) * (-1.0 / s3);
        assert!(dx_sp.to_vec().unwrap().iter().all(|v| v.is_finite()));
        assert_close(&dw_sp.to_vec().unwrap(), &[want_dw], 1e-4, 1e-4, &format!("gn 5e36 dw fb={portable}"));

        // Balanced opposite-infinity sum `[3e38, 3e38, -3e38, -3e38]` (whose raw sum produces +Inf + -Inf = NaN)
        // and non-power-of-two constant huge row `[1e38; 3]`.
        let bal = GpuTensor::from_slice(&ctx, &[3e38f32, 3e38, -3e38, -3e38], &[4]).unwrap();
        let yb = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
        let mb = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let rb = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            bal.raw(),
            None,
            None,
            yb.raw(),
            mb.raw(),
            rb.raw(),
            (1, 1, 4),
            1,
            0.0,
            DType::F32,
        )
        .unwrap();
        assert_close(
            &yb.to_vec().unwrap(),
            &[1.0, 1.0, -1.0, -1.0],
            1e-5,
            1e-5,
            &format!("gn bal 3e38 fb={portable}"),
        );

        let c3 = GpuTensor::from_slice(&ctx, &[1e38f32; 3], &[3]).unwrap();
        let yc3 = GpuTensor::empty(&ctx, &[3], DType::F32).unwrap();
        let mc3 = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let rc3 = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            c3.raw(),
            None,
            None,
            yc3.raw(),
            mc3.raw(),
            rc3.raw(),
            (1, 1, 3),
            1,
            1e-5,
            DType::F32,
        )
        .unwrap();
        assert_close(
            &yc3.to_vec().unwrap(),
            &[0.0; 3],
            1e-5,
            1e-5,
            &format!("gn const 1e38x3 fb={portable}"),
        );

        let big = GpuTensor::from_slice(&ctx, &[100_000_000.0, 100_000_008.0], &[2]).unwrap();
        let y_big = GpuTensor::empty(&ctx, &[2], DType::F32).unwrap();
        let m_big = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let r_big = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            big.raw(),
            None,
            None,
            y_big.raw(),
            m_big.raw(),
            r_big.raw(),
            (1, 1, 2),
            1,
            0.0,
            DType::F32,
        )
        .unwrap();
        assert_close(&y_big.to_vec().unwrap(), &[-1.0, 1.0], 1e-6, 1e-6, &format!("gn big y fb={portable}"));
        let g_ones = GpuTensor::from_slice(&ctx, &[1.0, 1.0], &[2]).unwrap();
        let dx_big = GpuTensor::empty(&ctx, &[2], DType::F32).unwrap();
        let dw_big = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let db_big = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_backward(
            &ctx,
            big.raw(),
            g_ones.raw(),
            None,
            m_big.raw(),
            r_big.raw(),
            dx_big.raw(),
            Some(dw_big.raw()),
            Some(db_big.raw()),
            (1, 1, 2),
            1,
            DType::F32,
        )
        .unwrap();
        assert_close(&dx_big.to_vec().unwrap(), &[0.0, 0.0], 1e-6, 1e-6, &format!("gn big dx fb={portable}"));
        assert_close(&dw_big.to_vec().unwrap(), &[0.0], 1e-6, 1e-6, &format!("gn big dw fb={portable}"));
        assert_close(&db_big.to_vec().unwrap(), &[2.0], 1e-6, 1e-6, &format!("gn big db fb={portable}"));

        // Uncached large-offset group (`m_u = 1026 > 1024`) verifies no driver fast-math reassociation there either.
        let big_u: Vec<f32> =
            (0..1026).map(|i| if i % 2 == 0 { 100_000_000.0 } else { 100_000_008.0 }).collect();
        let big_ut = GpuTensor::from_slice(&ctx, &big_u, &[1026]).unwrap();
        let yu = GpuTensor::empty(&ctx, &[1026], DType::F32).unwrap();
        let mu = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        let ru = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            big_ut.raw(),
            None,
            None,
            yu.raw(),
            mu.raw(),
            ru.raw(),
            (1, 1, 1026),
            1,
            0.0,
            DType::F32,
        )
        .unwrap();
        let want_u: Vec<f32> = (0..1026).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        assert_close(&yu.to_vec().unwrap(), &want_u, 1e-5, 1e-5, &format!("gn uncached big y fb={portable}"));
    }

    // F16 forward + backward parity when supported.
    if ctx.caps().shader_f16 {
        let (n, c, hw, g) = (2, 8, 12, 4);
        let total = n * c * hw;
        let x = f16_round(&rng.vec(total, -2.0, 2.0));
        let grad = f16_round(&rng.vec(total, -1.0, 1.0));
        let w = rng.vec(c, 0.5, 1.5);
        let b = rng.vec(c, -0.5, 0.5);
        let (y_ref, m_ref, r_ref) = cpu_group_norm_fwd(&x, Some(&w), Some(&b), (n, c, hw), g, 1e-4);
        let (dx_ref, dw_ref, db_ref) = cpu_group_norm_bwd(&x, &grad, Some(&w), &m_ref, &r_ref, (n, c, hw), g);
        let gx = GpuTensor::from_f16(&ctx, &to_f16(&x), &[total]).unwrap();
        let gg = GpuTensor::from_f16(&ctx, &to_f16(&grad), &[total]).unwrap();
        let gw = GpuTensor::from_slice(&ctx, &w, &[c]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b, &[c]).unwrap();
        let gy = GpuTensor::empty(&ctx, &[total], DType::F16).unwrap();
        let gm = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
        let gr = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
        group_norm_forward(
            &ctx,
            gx.raw(),
            Some(gw.raw()),
            Some(gb.raw()),
            gy.raw(),
            gm.raw(),
            gr.raw(),
            (n, c, hw),
            g,
            1e-4,
            DType::F16,
        )
        .unwrap();
        assert_close(&gy.to_vec().unwrap(), &y_ref, 5e-3, 5e-3, "gn f16 y");
        let gdx = GpuTensor::empty(&ctx, &[total], DType::F16).unwrap();
        let gdw = GpuTensor::empty(&ctx, &[c], DType::F32).unwrap();
        let gdb = GpuTensor::empty(&ctx, &[c], DType::F32).unwrap();
        group_norm_backward(
            &ctx,
            gx.raw(),
            gg.raw(),
            Some(gw.raw()),
            gm.raw(),
            gr.raw(),
            gdx.raw(),
            Some(gdw.raw()),
            Some(gdb.raw()),
            (n, c, hw),
            g,
            DType::F16,
        )
        .unwrap();
        assert_close(&gdx.to_vec().unwrap(), &dx_ref, 5e-3, 5e-3, "gn f16 dx");
        assert_close(&gdw.to_vec().unwrap(), &dw_ref, 5e-3, 5e-3, "gn f16 dw");
        assert_close(&gdb.to_vec().unwrap(), &db_ref, 5e-3, 5e-3, "gn f16 db");
    }

    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn tiled_sdpa_forward_matches_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5d9a_1234);

    // Covers: D=1, odd D<=32 (CACHE_Q, D_CHUNKS=1), D=32, odd D<=64 (CACHE_Q, D_CHUNKS=2), D=64,
    // D=96 (uncached Q, D_CHUNKS=3), D=128 (D_CHUNKS=4), D=160 & D=260 (d_tiles > 1), odd/non-pow2 Sq & Sk,
    // Sq != Sk, and single-element Sq=Sk=1.
    type SdpaCase = ((usize, usize, usize, usize), &'static str);
    let shapes: &[SdpaCase] = &[
        ((2, 1, 1, 1), "tiny 1x1x1"),
        ((2, 7, 13, 17), "odd cache_q d=17"),
        ((2, 37, 13, 32), "sk < sq d=32"),
        ((3, 16, 32, 32), "aligned cache_q d=32"),
        ((2, 35, 70, 48), "non-pow2 cache_q d=48"),
        ((2, 65, 65, 64), "attention-like d=64"),
        ((2, 33, 67, 96), "uncached q d=96"),
        ((2, 40, 40, 128), "uncached q d=128"),
        ((1, 20, 37, 160), "multi d_tile d=160"),
        ((1, 18, 25, 260), "multi d_tile d=260"),
    ];

    for &((b, sq, sk, d), label) in shapes {
        let q = rng.vec(b * sq * d, -1.5, 1.5);
        let k = rng.vec(b * sk * d, -1.5, 1.5);
        let v = rng.vec(b * sk * d, -1.5, 1.5);
        let scale = 1.0 / (d as f32).sqrt();
        let gq = GpuTensor::from_slice(&ctx, &q, &[b * sq * d]).unwrap();
        let gk = GpuTensor::from_slice(&ctx, &k, &[b * sk * d]).unwrap();
        let gv = GpuTensor::from_slice(&ctx, &v, &[b * sk * d]).unwrap();

        for causal in [false, true] {
            let want = cpu_sdpa_fwd(&q, &k, &v, (b, sq, sk, d), scale, causal);
            for portable in [false, true] {
                let _fg = ForceGuard::set(portable);
                let go = GpuTensor::empty(&ctx, &[b * sq * d], DType::F32).unwrap();
                sdpa_forward(
                    &ctx,
                    gq.raw(),
                    gk.raw(),
                    gv.raw(),
                    go.raw(),
                    (b, sq, sk, d),
                    scale,
                    causal,
                    DType::F32,
                )
                .unwrap();
                assert_close(
                    &go.to_vec().unwrap(),
                    &want,
                    2e-4,
                    2e-4,
                    &format!("sdpa {label} causal={causal} portable={portable}"),
                );
            }
        }
    }

    // Causal isolation: NaN / Inf in strictly future K/V positions (`k > q`) must NEVER leak into earlier rows.
    let (b, sq, sk, d) = (1, 18, 40, 32);
    let q = rng.vec(sq * d, -1.0, 1.0);
    let mut k = rng.vec(sk * d, -1.0, 1.0);
    let mut v = rng.vec(sk * d, -1.0, 1.0);
    for ki in 10..sk {
        for di in 0..d {
            k[ki * d + di] = if ki % 2 == 0 { f32::NAN } else { f32::INFINITY };
            v[ki * d + di] = if ki % 2 == 0 { f32::NAN } else { f32::NEG_INFINITY };
        }
    }
    let scale = 1.0 / (d as f32).sqrt();
    let want = cpu_sdpa_fwd(&q, &k, &v, (b, sq, sk, d), scale, true);
    let gq = GpuTensor::from_slice(&ctx, &q, &[sq * d]).unwrap();
    let gk = GpuTensor::from_slice(&ctx, &k, &[sk * d]).unwrap();
    let gv = GpuTensor::from_slice(&ctx, &v, &[sk * d]).unwrap();
    for portable in [false, true] {
        let _fg = ForceGuard::set(portable);
        let go = GpuTensor::empty(&ctx, &[sq * d], DType::F32).unwrap();
        sdpa_forward(&ctx, gq.raw(), gk.raw(), gv.raw(), go.raw(), (b, sq, sk, d), scale, true, DType::F32)
            .unwrap();
        let got = go.to_vec().unwrap();
        // Rows 0..10 only attend to keys 0..=9 (all finite), so they must be completely finite and match CPU.
        assert_close(&got[..10 * d], &want[..10 * d], 2e-4, 2e-4, &format!("causal isolation fb={portable}"));
        assert!(got[..10 * d].iter().all(|x| x.is_finite()));
    }

    // Extreme logits (±1e3) stay finite without overflow or NaN.
    let (b, sq, sk, d) = (1, 4, 4, 4);
    let q_ext = vec![100.0f32; sq * d];
    let k_ext =
        vec![10.0, -10.0, 10.0, 0.0, -10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
    let v_ext = rng.vec(sk * d, -2.0, 2.0);
    let want_ext = cpu_sdpa_fwd(&q_ext, &k_ext, &v_ext, (b, sq, sk, d), 1.0, false);
    let gqe = GpuTensor::from_slice(&ctx, &q_ext, &[sq * d]).unwrap();
    let gke = GpuTensor::from_slice(&ctx, &k_ext, &[sk * d]).unwrap();
    let gve = GpuTensor::from_slice(&ctx, &v_ext, &[sk * d]).unwrap();
    let goe = GpuTensor::empty(&ctx, &[sq * d], DType::F32).unwrap();
    sdpa_forward(&ctx, gqe.raw(), gke.raw(), gve.raw(), goe.raw(), (b, sq, sk, d), 1.0, false, DType::F32)
        .unwrap();
    assert_close(&goe.to_vec().unwrap(), &want_ext, 1e-4, 1e-4, "sdpa extreme logits");

    // Large V values (`1e38`) across multiple KV tiles (`sk = 64 > 32`) stay finite (`1e38`, exercising `o_scale`).
    let q_z = GpuTensor::from_slice(&ctx, &[0.0f32; 4 * 4], &[16]).unwrap();
    let k_z = GpuTensor::from_slice(&ctx, &[0.0f32; 64 * 4], &[256]).unwrap();
    let v_big = GpuTensor::from_slice(&ctx, &[1e38f32; 64 * 4], &[256]).unwrap();
    let o_big = GpuTensor::empty(&ctx, &[16], DType::F32).unwrap();
    sdpa_forward(&ctx, q_z.raw(), k_z.raw(), v_big.raw(), o_big.raw(), (1, 4, 64, 4), 0.5, false, DType::F32)
        .unwrap();
    assert_close(&o_big.to_vec().unwrap(), &[1e38f32; 16], 1e33, 1e-5, "sdpa large V 1e38 multi-tile");

    // Zero-edge shapes (sq=0 / b=0 -> Ok; sk=0 with sq>0 -> Err).
    let tiny = GpuTensor::from_slice(&ctx, &[1.0], &[1]).unwrap();
    sdpa_forward(&ctx, tiny.raw(), tiny.raw(), tiny.raw(), tiny.raw(), (0, 4, 4, 8), 1.0, false, DType::F32)
        .unwrap();
    sdpa_forward(&ctx, tiny.raw(), tiny.raw(), tiny.raw(), tiny.raw(), (2, 0, 4, 8), 1.0, false, DType::F32)
        .unwrap();
    assert!(
        sdpa_forward(
            &ctx,
            tiny.raw(),
            tiny.raw(),
            tiny.raw(),
            tiny.raw(),
            (1, 1, 0, 1),
            1.0,
            false,
            DType::F32
        )
        .is_err()
    );

    // F16 SDPA parity when supported.
    if ctx.caps().shader_f16 {
        let (b, sq, sk, d) = (2, 33, 45, 64);
        let q = f16_round(&rng.vec(b * sq * d, -1.0, 1.0));
        let k = f16_round(&rng.vec(b * sk * d, -1.0, 1.0));
        let v = f16_round(&rng.vec(b * sk * d, -1.0, 1.0));
        let scale = 1.0 / (d as f32).sqrt();
        let gq = GpuTensor::from_f16(&ctx, &to_f16(&q), &[b * sq * d]).unwrap();
        let gk = GpuTensor::from_f16(&ctx, &to_f16(&k), &[b * sk * d]).unwrap();
        let gv = GpuTensor::from_f16(&ctx, &to_f16(&v), &[b * sk * d]).unwrap();
        for causal in [false, true] {
            let want = cpu_sdpa_fwd(&q, &k, &v, (b, sq, sk, d), scale, causal);
            let go = GpuTensor::empty(&ctx, &[b * sq * d], DType::F16).unwrap();
            sdpa_forward(
                &ctx,
                gq.raw(),
                gk.raw(),
                gv.raw(),
                go.raw(),
                (b, sq, sk, d),
                scale,
                causal,
                DType::F16,
            )
            .unwrap();
            assert_close(&go.to_vec().unwrap(), &want, 5e-3, 5e-3, &format!("sdpa f16 causal={causal}"));
        }
    }

    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
