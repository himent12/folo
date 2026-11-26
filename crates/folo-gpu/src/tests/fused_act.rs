//! GPU-vs-CPU parity and finite-difference gradient tests for:
//! - Fused activation backward / epilogue kernels (`kernels::fused_act`: `silu_backward`, `gelu_backward`,
//!   `swiglu_forward`/`backward`, `bias_add_silu`).
//! - Fused `residual_add_layer_norm` and `residual_add_rms_norm` forward & backward (`kernels::residual_norm`).
//! - Extra GroupNorm / InstanceNorm backward edge cases (tiny spatial dims `hw = 1, 2, 3`, non-power-of-two
//!   channels, epsilon stability, and finite-difference gradient verification).

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::FORCE_NO_SUBGROUPS;
use crate::kernels::fused_act::{
    bias_add_silu, gelu_backward, silu_backward, swiglu_backward, swiglu_forward,
};
use crate::kernels::group_norm::{
    group_norm_backward, group_norm_forward, instance_norm_backward, instance_norm_forward,
};
use crate::kernels::residual_norm::{
    residual_add_layer_norm_backward, residual_add_layer_norm_forward, residual_add_rms_norm_backward,
    residual_add_rms_norm_forward,
};
use crate::{DType, GpuTensor, UnaryOp};

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

fn cpu_silu(x: f64) -> f64 {
    if x < -88.0 { 0.0 } else { x / (1.0 + (-x).exp()) }
}

fn cpu_silu_deriv(x: f64) -> f64 {
    if x > 88.0 {
        1.0
    } else if x < -88.0 {
        0.0
    } else {
        let s = 1.0 / (1.0 + (-x).exp());
        s + x * s * (1.0 - s)
    }
}

fn cpu_gelu_deriv(x: f64) -> f64 {
    if x.abs() >= 10.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let k = (2.0f64 / std::f64::consts::PI).sqrt();
    let c = 0.044715f64;
    let u = k * (x + c * x * x * x);
    let t = u.tanh();
    let sech2 = 1.0 - t * t;
    let du = k * (1.0 + 3.0 * c * x * x);
    0.5 * (1.0 + t + x * sech2 * du)
}

#[test]
fn fused_activations_and_swiglu_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0xac71_b000);

    // Covers vec4 aligned (1024), odd scalar tail (1027), tiny (1), and extreme magnitudes (+-1e4, +-100).
    for &n in &[1usize, 7, 1024, 1027] {
        let mut x = rng.vec(n, -6.0, 6.0);
        if n >= 7 {
            x[0] = -1e4;
            x[1] = -100.0;
            x[2] = 0.0;
            x[3] = 100.0;
            x[4] = 1e4;
        }
        let up = rng.vec(n, -3.0, 3.0);
        let g = rng.vec(n, -2.0, 2.0);
        let gx = GpuTensor::from_slice(&ctx, &x, &[n]).unwrap();
        let gu = GpuTensor::from_slice(&ctx, &up, &[n]).unwrap();
        let gg = GpuTensor::from_slice(&ctx, &g, &[n]).unwrap();

        // 1. silu_backward
        let dx_silu = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        silu_backward(&ctx, gx.raw(), gg.raw(), dx_silu.raw(), n, DType::F32).unwrap();
        let want_silu: Vec<f32> =
            x.iter().zip(&g).map(|(&xv, &gv)| (gv as f64 * cpu_silu_deriv(xv as f64)) as f32).collect();
        let got_silu = dx_silu.to_vec().unwrap();
        assert!(got_silu.iter().all(|v| v.is_finite()), "silu_bwd non-finite n={n}");
        assert_close(&got_silu, &want_silu, 2e-5, 2e-5, &format!("silu_bwd n={n}"));

        // 2. gelu_backward
        let dx_gelu = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        gelu_backward(&ctx, gx.raw(), gg.raw(), dx_gelu.raw(), n, DType::F32).unwrap();
        let want_gelu: Vec<f32> =
            x.iter().zip(&g).map(|(&xv, &gv)| (gv as f64 * cpu_gelu_deriv(xv as f64)) as f32).collect();
        let got_gelu = dx_gelu.to_vec().unwrap();
        assert!(got_gelu.iter().all(|v| v.is_finite()), "gelu_bwd non-finite n={n}");
        assert_close(&got_gelu, &want_gelu, 3e-5, 3e-5, &format!("gelu_bwd n={n}"));

        // 3. swiglu_forward + swiglu_backward
        let gy = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        swiglu_forward(&ctx, gx.raw(), gu.raw(), gy.raw(), n, DType::F32).unwrap();
        let want_y: Vec<f32> =
            x.iter().zip(&up).map(|(&xv, &uv)| (cpu_silu(xv as f64) * uv as f64) as f32).collect();
        assert_close(&gy.to_vec().unwrap(), &want_y, 2e-5, 2e-5, &format!("swiglu_fwd n={n}"));

        let gdg = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        let gdu = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        swiglu_backward(&ctx, gx.raw(), gu.raw(), gg.raw(), gdg.raw(), gdu.raw(), n, DType::F32).unwrap();
        let want_dg: Vec<f32> = x
            .iter()
            .zip(&up)
            .zip(&g)
            .map(|((&xv, &uv), &gv)| (gv as f64 * uv as f64 * cpu_silu_deriv(xv as f64)) as f32)
            .collect();
        let want_du: Vec<f32> =
            x.iter().zip(&g).map(|(&xv, &gv)| (gv as f64 * cpu_silu(xv as f64)) as f32).collect();
        assert_close(&gdg.to_vec().unwrap(), &want_dg, 3e-5, 3e-5, &format!("swiglu_bwd dgate n={n}"));
        assert_close(&gdu.to_vec().unwrap(), &want_du, 3e-5, 3e-5, &format!("swiglu_bwd dup n={n}"));
    }

    // Finite-difference check for GELU backward against UnaryOp::Gelu forward.
    let x_fd = rng.vec(16, -4.0, 4.0);
    let g_fd = rng.vec(16, -1.5, 1.5);
    let gx = GpuTensor::from_slice(&ctx, &x_fd, &[16]).unwrap();
    let gg = GpuTensor::from_slice(&ctx, &g_fd, &[16]).unwrap();
    let gdx = GpuTensor::empty(&ctx, &[16], DType::F32).unwrap();
    gelu_backward(&ctx, gx.raw(), gg.raw(), gdx.raw(), 16, DType::F32).unwrap();
    let got_fd = gdx.to_vec().unwrap();
    let h = 1e-3f32;
    for i in 0..16 {
        let fd = g_fd[i] * (UnaryOp::Gelu.apply(x_fd[i] + h) - UnaryOp::Gelu.apply(x_fd[i] - h)) / (2.0 * h);
        assert!((got_fd[i] - fd).abs() < 2e-3, "gelu fd[{i}]: got {}, fd {fd}", got_fd[i]);
    }

    // 4. bias_add_silu: MODE 0 (NCHW inner % 4 == 0), MODE 1 (Linear inner == 1, C % 4 == 0), MODE 2 (odd scalar),
    //    and bias = None (vec4 + scalar).
    for &(n, c, inner, label) in &[
        (2 * 6 * 16, 6, 16, "mode0 nchw vec4"),
        (7 * 12, 12, 1, "mode1 linear vec4"),
        (2 * 5 * 9, 5, 9, "mode2 odd scalar"),
        (3 * 7, 7, 1, "mode2 odd linear"),
    ] {
        let x = rng.vec(n, -4.0, 4.0);
        let b = rng.vec(c, -1.0, 1.0);
        let gx = GpuTensor::from_slice(&ctx, &x, &[n]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b, &[c]).unwrap();
        let gy = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        bias_add_silu(&ctx, gx.raw(), Some(gb.raw()), gy.raw(), n, c, inner, DType::F32).unwrap();
        let want: Vec<f32> = (0..n).map(|i| cpu_silu((x[i] + b[(i / inner) % c]) as f64) as f32).collect();
        assert_close(&gy.to_vec().unwrap(), &want, 2e-5, 2e-5, &format!("bias_add_silu {label}"));

        let gy_nb = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        bias_add_silu(&ctx, gx.raw(), None, gy_nb.raw(), n, c, inner, DType::F32).unwrap();
        let want_nb: Vec<f32> = x.iter().map(|&v| cpu_silu(v as f64) as f32).collect();
        assert_close(&gy_nb.to_vec().unwrap(), &want_nb, 2e-5, 2e-5, &format!("bias_add_silu none {label}"));
    }

    // 5. F16 parity for all fused activation kernels when supported.
    if ctx.caps().shader_f16 {
        let n = 64;
        let x = f16_round(&rng.vec(n, -3.0, 3.0));
        let up = f16_round(&rng.vec(n, -2.0, 2.0));
        let g = f16_round(&rng.vec(n, -1.5, 1.5));
        let b = f16_round(&rng.vec(8, -0.5, 0.5));
        let gx = GpuTensor::from_f16(&ctx, &to_f16(&x), &[n]).unwrap();
        let gu = GpuTensor::from_f16(&ctx, &to_f16(&up), &[n]).unwrap();
        let gg = GpuTensor::from_f16(&ctx, &to_f16(&g), &[n]).unwrap();
        let gb = GpuTensor::from_f16(&ctx, &to_f16(&b), &[8]).unwrap();
        let out = GpuTensor::empty(&ctx, &[n], DType::F16).unwrap();
        let out2 = GpuTensor::empty(&ctx, &[n], DType::F16).unwrap();

        silu_backward(&ctx, gx.raw(), gg.raw(), out.raw(), n, DType::F16).unwrap();
        let want: Vec<f32> =
            x.iter().zip(&g).map(|(&xv, &gv)| (gv as f64 * cpu_silu_deriv(xv as f64)) as f32).collect();
        assert_close(&out.to_vec().unwrap(), &want, 5e-3, 5e-3, "silu_bwd f16");

        gelu_backward(&ctx, gx.raw(), gg.raw(), out.raw(), n, DType::F16).unwrap();
        let want: Vec<f32> =
            x.iter().zip(&g).map(|(&xv, &gv)| (gv as f64 * cpu_gelu_deriv(xv as f64)) as f32).collect();
        assert_close(&out.to_vec().unwrap(), &want, 5e-3, 5e-3, "gelu_bwd f16");

        swiglu_forward(&ctx, gx.raw(), gu.raw(), out.raw(), n, DType::F16).unwrap();
        let want: Vec<f32> =
            x.iter().zip(&up).map(|(&xv, &uv)| (cpu_silu(xv as f64) * uv as f64) as f32).collect();
        assert_close(&out.to_vec().unwrap(), &want, 5e-3, 5e-3, "swiglu_fwd f16");

        swiglu_backward(&ctx, gx.raw(), gu.raw(), gg.raw(), out.raw(), out2.raw(), n, DType::F16).unwrap();
        let want_dg: Vec<f32> = x
            .iter()
            .zip(&up)
            .zip(&g)
            .map(|((&xv, &uv), &gv)| (gv as f64 * uv as f64 * cpu_silu_deriv(xv as f64)) as f32)
            .collect();
        let want_du: Vec<f32> =
            x.iter().zip(&g).map(|(&xv, &gv)| (gv as f64 * cpu_silu(xv as f64)) as f32).collect();
        assert_close(&out.to_vec().unwrap(), &want_dg, 8e-3, 8e-3, "swiglu_bwd dgate f16");
        assert_close(&out2.to_vec().unwrap(), &want_du, 5e-3, 5e-3, "swiglu_bwd dup f16");

        bias_add_silu(&ctx, gx.raw(), Some(gb.raw()), out.raw(), n, 8, 1, DType::F16).unwrap();
        let want_b: Vec<f32> = (0..n).map(|i| cpu_silu((x[i] + b[i % 8]) as f64) as f32).collect();
        assert_close(&out.to_vec().unwrap(), &want_b, 5e-3, 5e-3, "bias_add_silu f16");
    }

    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn cpu_res_norm_fwd(
    x: &[f32],
    res: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    outer: usize,
    d: usize,
    eps: f32,
    rms: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut h = vec![0.0f32; outer * d];
    let mut y = vec![0.0f32; outer * d];
    let mut mean = vec![0.0f32; outer];
    let mut rstd = vec![0.0f32; outer];
    for o in 0..outer {
        let base = o * d;
        for i in 0..d {
            h[base + i] = x[base + i] + res[base + i];
        }
        let mu =
            if rms { 0.0f64 } else { h[base..base + d].iter().map(|&v| v as f64).sum::<f64>() / d as f64 };
        let ms = h[base..base + d].iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / d as f64;
        let rs = 1.0 / (ms + eps as f64).sqrt();
        mean[o] = mu as f32;
        rstd[o] = rs as f32;
        for i in 0..d {
            let xh = (h[base + i] as f64 - mu) * rs;
            let wc = w.map_or(1.0f64, |v| v[i] as f64);
            let bc = b.map_or(0.0f64, |v| v[i] as f64);
            y[base + i] = (xh * wc + bc) as f32;
        }
    }
    (h, y, mean, rstd)
}

#[allow(clippy::too_many_arguments)]
fn cpu_res_norm_bwd(
    h: &[f32],
    gy: &[f32],
    gres: Option<&[f32]>,
    w: Option<&[f32]>,
    mean: &[f32],
    rstd: &[f32],
    outer: usize,
    d: usize,
    rms: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dh = vec![0.0f32; outer * d];
    let mut dw = vec![0.0f64; d];
    let mut db = vec![0.0f64; d];
    for o in 0..outer {
        let base = o * d;
        let mu = if rms { 0.0 } else { mean[o] as f64 };
        let rs = rstd[o] as f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        for i in 0..d {
            let xh = (h[base + i] as f64 - mu) * rs;
            let gv = gy[base + i] as f64;
            let gw = gv * w.map_or(1.0f64, |v| v[i] as f64);
            s1 += gw;
            s2 += gw * xh;
            dw[i] += gv * xh;
            db[i] += gv;
        }
        let c1 = if rms { 0.0 } else { s1 / d as f64 };
        let c2 = s2 / d as f64;
        for i in 0..d {
            let xh = (h[base + i] as f64 - mu) * rs;
            let gw = gy[base + i] as f64 * w.map_or(1.0f64, |v| v[i] as f64);
            let dx = rs * (gw - c1 - xh * c2);
            let gr = gres.map_or(0.0f64, |v| v[base + i] as f64);
            dh[base + i] = (dx + gr) as f32;
        }
    }
    (dh, dw.into_iter().map(|v| v as f32).collect(), db.into_iter().map(|v| v as f32).collect())
}

#[test]
fn residual_add_norm_forward_and_backward_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x4e50_9043);

    // Covers: d=1, scalar odd cached (d=17), vec4 cached (d=64), uncached scalar (d=1025),
    // uncached vec4 (d=4100), and split-K dweight/dbias (outer=300 >= 256).
    let cases: &[(usize, usize, &str)] = &[
        (3, 1, "d=1"),
        (5, 17, "scalar cached"),
        (4, 64, "vec4 cached"),
        (2, 1025, "scalar uncached"),
        (2, 4100, "vec4 uncached"),
        (300, 32, "split-k vec4"),
    ];

    for &(outer, d, label) in cases {
        let total = outer * d;
        let x = rng.vec(total, -1.5, 1.5);
        let res = rng.vec(total, -1.5, 1.5);
        let gy = rng.vec(total, -1.0, 1.0);
        let gres = rng.vec(total, -1.0, 1.0);
        let w = rng.vec(d, 0.5, 1.5);
        let b = rng.vec(d, -0.5, 0.5);
        let eps = 1e-5f32;
        let gx = GpuTensor::from_slice(&ctx, &x, &[total]).unwrap();
        let gr = GpuTensor::from_slice(&ctx, &res, &[total]).unwrap();
        let ggy = GpuTensor::from_slice(&ctx, &gy, &[total]).unwrap();
        let ggr = GpuTensor::from_slice(&ctx, &gres, &[total]).unwrap();
        let gw = GpuTensor::from_slice(&ctx, &w, &[d]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b, &[d]).unwrap();
        let tol = if d > 1000 || outer >= 256 { 2e-3 } else { 2e-4 };

        for rms in [false, true] {
            let (h_ref, y_ref, m_ref, r_ref) =
                cpu_res_norm_fwd(&x, &res, Some(&w), (!rms).then_some(&b), outer, d, eps, rms);
            let (dh_ref, dw_ref, db_ref) =
                cpu_res_norm_bwd(&h_ref, &gy, Some(&gres), Some(&w), &m_ref, &r_ref, outer, d, rms);

            for portable in [false, true] {
                let _fg = ForceGuard::set(portable);
                let gh = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
                let gyo = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
                let gm = GpuTensor::empty(&ctx, &[outer], DType::F32).unwrap();
                let grs = GpuTensor::empty(&ctx, &[outer], DType::F32).unwrap();
                let gdh = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
                let gdw = GpuTensor::empty(&ctx, &[d], DType::F32).unwrap();
                let gdb = GpuTensor::empty(&ctx, &[d], DType::F32).unwrap();

                if rms {
                    residual_add_rms_norm_forward(
                        &ctx,
                        gx.raw(),
                        gr.raw(),
                        Some(gw.raw()),
                        gh.raw(),
                        gyo.raw(),
                        grs.raw(),
                        outer,
                        d,
                        eps,
                        DType::F32,
                    )
                    .unwrap();
                    residual_add_rms_norm_backward(
                        &ctx,
                        gh.raw(),
                        ggy.raw(),
                        Some(ggr.raw()),
                        Some(gw.raw()),
                        grs.raw(),
                        gdh.raw(),
                        Some(gdw.raw()),
                        outer,
                        d,
                        DType::F32,
                    )
                    .unwrap();
                } else {
                    residual_add_layer_norm_forward(
                        &ctx,
                        gx.raw(),
                        gr.raw(),
                        Some(gw.raw()),
                        Some(gb.raw()),
                        gh.raw(),
                        gyo.raw(),
                        gm.raw(),
                        grs.raw(),
                        outer,
                        d,
                        eps,
                        DType::F32,
                    )
                    .unwrap();
                    residual_add_layer_norm_backward(
                        &ctx,
                        gh.raw(),
                        ggy.raw(),
                        Some(ggr.raw()),
                        Some(gw.raw()),
                        gm.raw(),
                        grs.raw(),
                        gdh.raw(),
                        Some(gdw.raw()),
                        Some(gdb.raw()),
                        outer,
                        d,
                        DType::F32,
                    )
                    .unwrap();
                }
                let tag = format!("{label} rms={rms} portable={portable}");
                assert_close(&gh.to_vec().unwrap(), &h_ref, 1e-5, 1e-5, &format!("res_norm h {tag}"));
                assert_close(&gyo.to_vec().unwrap(), &y_ref, tol, 2e-4, &format!("res_norm y {tag}"));
                assert_close(&gdh.to_vec().unwrap(), &dh_ref, tol, 2e-4, &format!("res_norm dh {tag}"));
                assert_close(&gdw.to_vec().unwrap(), &dw_ref, tol * 2.0, 2e-4, &format!("res_norm dw {tag}"));
                if !rms {
                    assert_close(
                        &gdb.to_vec().unwrap(),
                        &db_ref,
                        tol * 2.0,
                        2e-4,
                        &format!("res_norm db {tag}"),
                    );
                }
            }
        }
    }

    // F16 residual_add_norm forward + backward parity when supported.
    if ctx.caps().shader_f16 {
        let (outer, d) = (6, 32);
        let total = outer * d;
        let x = f16_round(&rng.vec(total, -1.0, 1.0));
        let res = f16_round(&rng.vec(total, -1.0, 1.0));
        let h_f16: Vec<f32> =
            x.iter().zip(&res).map(|(&a, &b)| half::f16::from_f32(a + b).to_f32()).collect();
        let gy = f16_round(&rng.vec(total, -1.0, 1.0));
        let w = rng.vec(d, 0.5, 1.5);
        let b = rng.vec(d, -0.5, 0.5);
        let zero = vec![0.0f32; total];
        let (_, y_ref, m_ref, r_ref) =
            cpu_res_norm_fwd(&h_f16, &zero, Some(&w), Some(&b), outer, d, 1e-4, false);
        let (dh_ref, dw_ref, db_ref) =
            cpu_res_norm_bwd(&h_f16, &gy, None, Some(&w), &m_ref, &r_ref, outer, d, false);

        let gx = GpuTensor::from_f16(&ctx, &to_f16(&x), &[total]).unwrap();
        let gr = GpuTensor::from_f16(&ctx, &to_f16(&res), &[total]).unwrap();
        let ggy = GpuTensor::from_f16(&ctx, &to_f16(&gy), &[total]).unwrap();
        let gw = GpuTensor::from_slice(&ctx, &w, &[d]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b, &[d]).unwrap();
        let gh = GpuTensor::empty(&ctx, &[total], DType::F16).unwrap();
        let gyo = GpuTensor::empty(&ctx, &[total], DType::F16).unwrap();
        let gm = GpuTensor::empty(&ctx, &[outer], DType::F32).unwrap();
        let grs = GpuTensor::empty(&ctx, &[outer], DType::F32).unwrap();
        let gdh = GpuTensor::empty(&ctx, &[total], DType::F16).unwrap();
        let gdw = GpuTensor::empty(&ctx, &[d], DType::F32).unwrap();
        let gdb = GpuTensor::empty(&ctx, &[d], DType::F32).unwrap();

        residual_add_layer_norm_forward(
            &ctx,
            gx.raw(),
            gr.raw(),
            Some(gw.raw()),
            Some(gb.raw()),
            gh.raw(),
            gyo.raw(),
            gm.raw(),
            grs.raw(),
            outer,
            d,
            1e-4,
            DType::F16,
        )
        .unwrap();
        residual_add_layer_norm_backward(
            &ctx,
            gh.raw(),
            ggy.raw(),
            None,
            Some(gw.raw()),
            gm.raw(),
            grs.raw(),
            gdh.raw(),
            Some(gdw.raw()),
            Some(gdb.raw()),
            outer,
            d,
            DType::F16,
        )
        .unwrap();
        assert_close(&gh.to_vec().unwrap(), &h_f16, 1e-3, 1e-3, "res_norm f16 h");
        assert_close(&gyo.to_vec().unwrap(), &y_ref, 5e-3, 5e-3, "res_norm f16 y");
        assert_close(&gdh.to_vec().unwrap(), &dh_ref, 5e-3, 5e-3, "res_norm f16 dh");
        assert_close(&gdw.to_vec().unwrap(), &dw_ref, 8e-3, 8e-3, "res_norm f16 dw");
        assert_close(&gdb.to_vec().unwrap(), &db_ref, 5e-3, 5e-3, "res_norm f16 db");
    }

    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn group_and_instance_norm_backward_tiny_spatial_and_finite_diff() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x9a00_b4d1);

    // Tiny spatial dims (hw = 1, 2, 3), non-power-of-two channels (c = 3, 5, 7, 9), varying epsilons.
    for &((n, c, hw), g, eps) in &[
        ((2, 3, 1), 3, 1e-5f32),  // InstanceNorm with single-pixel spatial dim (zero variance)
        ((2, 6, 1), 2, 1e-12f32), // GroupNorm with hw = 1, tiny eps
        ((2, 5, 2), 1, 1e-3f32),  // non-pow2 C=5, hw=2
        ((2, 9, 3), 3, 1e-5f32),  // non-pow2 C=9, G=3, hw=3
    ] {
        let total = n * c * hw;
        let x = rng.vec(total, -1.5, 1.5);
        let grad = rng.vec(total, -1.0, 1.0);
        let w = rng.vec(c, 0.5, 1.5);
        let b = rng.vec(c, -0.5, 0.5);

        let gx = GpuTensor::from_slice(&ctx, &x, &[total]).unwrap();
        let gg = GpuTensor::from_slice(&ctx, &grad, &[total]).unwrap();
        let gw = GpuTensor::from_slice(&ctx, &w, &[c]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b, &[c]).unwrap();
        let gy = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
        let gm = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
        let gr = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
        let gdx = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
        let gdw = GpuTensor::empty(&ctx, &[c], DType::F32).unwrap();
        let gdb = GpuTensor::empty(&ctx, &[c], DType::F32).unwrap();

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

        // Finite-difference check on dx, dweight, dbias using GPU forward as oracle.
        let run_loss = |xv: &[f32], wv: &[f32], bv: &[f32]| -> f64 {
            let tx = GpuTensor::from_slice(&ctx, xv, &[total]).unwrap();
            let tw = GpuTensor::from_slice(&ctx, wv, &[c]).unwrap();
            let tb = GpuTensor::from_slice(&ctx, bv, &[c]).unwrap();
            let ty = GpuTensor::empty(&ctx, &[total], DType::F32).unwrap();
            let tm = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
            let tr = GpuTensor::empty(&ctx, &[n * g], DType::F32).unwrap();
            group_norm_forward(
                &ctx,
                tx.raw(),
                Some(tw.raw()),
                Some(tb.raw()),
                ty.raw(),
                tm.raw(),
                tr.raw(),
                (n, c, hw),
                g,
                eps,
                DType::F32,
            )
            .unwrap();
            ty.to_vec().unwrap().iter().zip(&grad).map(|(&a, &grv)| a as f64 * grv as f64).sum()
        };
        let dx_got = gdx.to_vec().unwrap();
        let dw_got = gdw.to_vec().unwrap();
        let db_got = gdb.to_vec().unwrap();
        let h = 1e-3f32;
        for i in (0..total).step_by(3) {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += h;
            xm[i] -= h;
            let fd = ((run_loss(&xp, &w, &b) - run_loss(&xm, &w, &b)) / (2.0 * h as f64)) as f32;
            assert!(
                (dx_got[i] - fd).abs() < 3e-3,
                "gn fd dx[{i}] ({n},{c},{hw}) G={g}: got {}, fd {fd}",
                dx_got[i]
            );
        }
        for ci in 0..c {
            let mut wp = w.clone();
            let mut wm = w.clone();
            wp[ci] += h;
            wm[ci] -= h;
            let fd_w = ((run_loss(&x, &wp, &b) - run_loss(&x, &wm, &b)) / (2.0 * h as f64)) as f32;
            assert!((dw_got[ci] - fd_w).abs() < 3e-3, "gn fd dw[{ci}]: got {}, fd {fd_w}", dw_got[ci]);
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[ci] += h;
            bm[ci] -= h;
            let fd_b = ((run_loss(&x, &w, &bp) - run_loss(&x, &w, &bm)) / (2.0 * h as f64)) as f32;
            assert!((db_got[ci] - fd_b).abs() < 3e-3, "gn fd db[{ci}]: got {}, fd {fd_b}", db_got[ci]);
        }
    }
}
