//! GPU-vs-CPU parity tests for Sprint 8 fused epilogue and F16/extreme-scale norm kernels:
//! - Fused GEMM / Conv2d + Bias + Residual + Activation epilogues (`gemm_epilogue`, `conv2d_epilogue`,
//!   `bias_residual_act`).
//! - `F16` + extreme-scale `RMSNorm` / `LayerNorm` forward & backward (`layer_norm::*_dtype`).

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::FORCE_NO_SUBGROUPS;
use crate::kernels::conv::{Conv2dDesc, conv2d_epilogue};
use crate::kernels::fused_act::{FusedAct, bias_residual_act};
use crate::kernels::gemm::{
    BatchStrides, GemmBias, GemmDesc, ResidualDesc, ResidualMode, TileConfig, gemm_epilogue,
};
use crate::kernels::layer_norm::{
    layer_norm_backward_dtype, layer_norm_forward_dtype, rms_norm_backward_dtype, rms_norm_forward_dtype,
};
use crate::kernels::pooling::Window2d;
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

fn upload_dt(ctx: &crate::GpuContext, v: &[f32], dtype: DType) -> GpuTensor {
    match dtype {
        DType::F32 => GpuTensor::from_slice(ctx, v, &[v.len()]).unwrap(),
        DType::F16 => GpuTensor::from_f16(ctx, &to_f16(v), &[v.len()]).unwrap(),
    }
}

#[test]
fn gemm_and_conv2d_fused_epilogues_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x58e0_0001);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // 1. gemm_epilogue: aligned vec4 & non-multiple-of-4 odd shapes, Row/Col bias, PreAct/PostAct residual,
    //    SiLU / GELU / ReLU, transposed, batched, and K = 0 degenerate.
    for &dtype in dtypes {
        for &(m, n, k, batch, ta, tb, bias_kind, res_mode, act) in &[
            (
                64usize,
                64usize,
                32usize,
                1usize,
                false,
                false,
                GemmBias::Col,
                ResidualMode::PreAct,
                UnaryOp::Silu,
            ),
            (13, 19, 11, 2, false, true, GemmBias::Row, ResidualMode::PostAct, UnaryOp::Gelu),
            (17, 12, 15, 1, true, false, GemmBias::Col, ResidualMode::PreAct, UnaryOp::Relu),
            (9, 7, 0, 2, false, false, GemmBias::Col, ResidualMode::PostAct, UnaryOp::Silu),
        ] {
            let mut a = rng.vec(batch * m * k, -0.8, 0.8);
            let mut b = rng.vec(batch * k * n, -0.8, 0.8);
            let mut res = rng.vec(batch * m * n, -0.5, 0.5);
            let bias_len = match bias_kind {
                GemmBias::None => 0,
                GemmBias::Row => m,
                GemmBias::Col => n,
            };
            let mut bias = rng.vec(bias_len, -0.4, 0.4);
            if dtype == DType::F16 {
                a = f16_round(&a);
                b = f16_round(&b);
                res = f16_round(&res);
                bias = f16_round(&bias);
            }
            let alpha = 0.75f32;
            let beta = 0.5f32;
            let mut d = GemmDesc::new(m, n, k, ta, tb, batch);
            d.alpha = alpha;
            d.bias = bias_kind;
            d.bias_strides = BatchStrides::default();
            d.act = act;
            d.dtype = dtype;
            let rdesc = ResidualDesc::packed(m, n, beta, res_mode);

            let mut want = vec![0.0f32; batch * m * n];
            for bi in 0..batch {
                for r in 0..m {
                    for c in 0..n {
                        let mut acc = 0.0f64;
                        for kk in 0..k {
                            let av = if ta { a[bi * m * k + kk * m + r] } else { a[bi * m * k + r * k + kk] };
                            let bv = if tb { b[bi * k * n + c * k + kk] } else { b[bi * k * n + kk * n + c] };
                            acc += av as f64 * bv as f64;
                        }
                        let mut v = acc as f32 * alpha;
                        match bias_kind {
                            GemmBias::Row => v += bias[r],
                            GemmBias::Col => v += bias[c],
                            GemmBias::None => {}
                        }
                        let rv = res[bi * m * n + r * n + c];
                        let out_v = match res_mode {
                            ResidualMode::PreAct => act.apply(v + beta * rv),
                            ResidualMode::PostAct => act.apply(v) + beta * rv,
                            ResidualMode::None => act.apply(v),
                        };
                        want[bi * m * n + r * n + c] = out_v;
                    }
                }
            }

            let ga = if k == 0 { None } else { Some(upload_dt(&ctx, &a, dtype)) };
            let gb = if k == 0 { None } else { Some(upload_dt(&ctx, &b, dtype)) };
            let gr = upload_dt(&ctx, &res, dtype);
            let gbias = upload_dt(&ctx, &bias, dtype);
            let gc = GpuTensor::empty(&ctx, &[batch * m * n], dtype).unwrap();
            let a_raw = ga.as_ref().map_or(ctx.dummy(), |t| t.raw());
            let b_raw = gb.as_ref().map_or(ctx.dummy(), |t| t.raw());
            gemm_epilogue(
                &ctx,
                &d,
                a_raw,
                b_raw,
                Some(gbias.raw()),
                Some((gr.raw(), &rdesc)),
                gc.raw(),
                Some(TileConfig::SMALL),
            )
            .unwrap();
            let (atol, rtol) = if dtype == DType::F16 { (8e-3, 5e-3) } else { (5e-5, 2e-5) };
            assert_close(
                &gc.to_vec().unwrap(),
                &want,
                atol,
                rtol,
                &format!("gemm_epilogue {m}x{n}x{k} {res_mode:?} {act:?} {dtype:?}"),
            );
        }
    }

    // Empty M = 0 no-op, strided residual (mismatched VEC_C / VEC_R), and output-aliasing rejection.
    {
        let d = GemmDesc::new(0, 8, 8, false, false, 1);
        let rdesc = ResidualDesc::packed(0, 8, 1.0, ResidualMode::PreAct);
        gemm_epilogue(
            &ctx,
            &d,
            ctx.dummy(),
            ctx.dummy(),
            None,
            Some((ctx.dummy(), &rdesc)),
            ctx.dummy(),
            None,
        )
        .unwrap();

        // Strided residual: M=6, N=8 (VEC_C=true), ldr=10 (VEC_R=false), offset=1.
        let (m, n, k, ldr, roff) = (6usize, 8usize, 8usize, 10usize, 1usize);
        let a = rng.vec(m * k, -0.5, 0.5);
        let b = rng.vec(k * n, -0.5, 0.5);
        let res = rng.vec(roff + m * ldr, -0.5, 0.5);
        let mut ds = GemmDesc::new(m, n, k, false, false, 1);
        ds.act = UnaryOp::Silu;
        let rs = ResidualDesc {
            beta: 0.5,
            mode: ResidualMode::PostAct,
            ldr,
            strides: BatchStrides { offset: roff, outer: 0, inner: 0 },
        };
        let mut want = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let acc: f64 = (0..k).map(|kk| a[r * k + kk] as f64 * b[kk * n + c] as f64).sum();
                want[r * n + c] = UnaryOp::Silu.apply(acc as f32) + 0.5 * res[roff + r * ldr + c];
            }
        }
        let ga = GpuTensor::from_slice(&ctx, &a, &[a.len()]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b, &[b.len()]).unwrap();
        let gr = GpuTensor::from_slice(&ctx, &res, &[res.len()]).unwrap();
        let gc = GpuTensor::empty(&ctx, &[m * n], DType::F32).unwrap();
        gemm_epilogue(&ctx, &ds, ga.raw(), gb.raw(), None, Some((gr.raw(), &rs)), gc.raw(), None).unwrap();
        assert_close(&gc.to_vec().unwrap(), &want, 5e-5, 2e-5, "gemm_epilogue strided res");

        // Output-residual aliasing must be rejected cleanly before dispatch.
        assert!(
            gemm_epilogue(
                &ctx,
                &ds,
                ga.raw(),
                gb.raw(),
                None,
                Some((gc.raw(), &ResidualDesc::packed(m, n, 1.0, ResidualMode::PreAct))),
                gc.raw(),
                None
            )
            .is_err()
        );
        assert!(
            bias_residual_act(
                &ctx,
                gc.raw(),
                None,
                Some((gr.raw(), 1.0, ResidualMode::PreAct)),
                gc.raw(),
                m * n,
                1,
                1,
                FusedAct::Silu,
                DType::F32
            )
            .is_err()
        );
        assert!(
            bias_residual_act(
                &ctx,
                ga.raw(),
                None,
                Some((gc.raw(), 1.0, ResidualMode::PreAct)),
                gc.raw(),
                m * n,
                1,
                1,
                FusedAct::Silu,
                DType::F32
            )
            .is_err()
        );
    }

    // Single-dispatch trace check for gemm_epilogue and conv2d_epilogue (pointwise & implicit-GEMM).
    {
        let fresh = crate::GpuContext::new().unwrap();
        let d = GemmDesc {
            act: UnaryOp::Gelu,
            bias: GemmBias::Col,
            ..GemmDesc::new(16, 16, 16, false, false, 1)
        };
        let rdesc = ResidualDesc::packed(16, 16, 1.0, ResidualMode::PreAct);
        let z = GpuTensor::zeros(&fresh, &[256], DType::F32).unwrap();
        let b = GpuTensor::zeros(&fresh, &[16], DType::F32).unwrap();
        let out = GpuTensor::empty(&fresh, &[256], DType::F32).unwrap();
        fresh.trace_begin();
        gemm_epilogue(&fresh, &d, z.raw(), z.raw(), Some(b.raw()), Some((z.raw(), &rdesc)), out.raw(), None)
            .unwrap();
        assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["gemm_epilogue"]);
    }

    // 2. conv2d_epilogue: pointwise 1x1, 3x3 implicit-GEMM (odd 5x7 & aligned 8x8), grouped, and depthwise.
    for &(n, cin, cout, h, w, kh, kw, groups, pad, res_mode, act) in &[
        (
            2usize,
            8usize,
            6usize,
            8usize,
            8usize,
            1usize,
            1usize,
            1usize,
            0usize,
            ResidualMode::PreAct,
            UnaryOp::Silu,
        ),
        (2, 6, 9, 5, 7, 3, 3, 1, 1, ResidualMode::PostAct, UnaryOp::Gelu),
        (1, 8, 8, 6, 6, 3, 3, 2, 1, ResidualMode::PreAct, UnaryOp::Relu),
        (2, 6, 6, 5, 7, 3, 3, 6, 1, ResidualMode::PostAct, UnaryOp::Silu),
    ] {
        let d = Conv2dDesc {
            n,
            c_in: cin,
            h,
            w,
            c_out: cout,
            groups,
            win: Window2d { kernel: (kh, kw), stride: (1, 1), padding: (pad, pad), dilation: (1, 1) },
            act,
            dtype: DType::F32,
        };
        let (oh, ow) = d.output().unwrap();
        let x = rng.vec(n * cin * h * w, -0.7, 0.7);
        let wt = rng.vec(cout * (cin / groups) * kh * kw, -0.5, 0.5);
        let bias = rng.vec(cout, -0.3, 0.3);
        let res = rng.vec(n * cout * oh * ow, -0.5, 0.5);
        let beta = 0.6f32;

        // CPU reference conv2d + bias + residual + act.
        let mut want = vec![0.0f32; n * cout * oh * ow];
        let cin_g = cin / groups;
        let cout_g = cout / groups;
        for bi in 0..n {
            for (co, &b_co) in bias.iter().enumerate() {
                let g = co / cout_g;
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut s = 0.0f64;
                        for ci in 0..cin_g {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = oy as isize - pad as isize + ky as isize;
                                    let ix = ox as isize - pad as isize + kx as isize;
                                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                        let xi =
                                            ((bi * cin + g * cin_g + ci) * h + iy as usize) * w + ix as usize;
                                        let wi = ((co * cin_g + ci) * kh + ky) * kw + kx;
                                        s += x[xi] as f64 * wt[wi] as f64;
                                    }
                                }
                            }
                        }
                        let idx = ((bi * cout + co) * oh + oy) * ow + ox;
                        let v = s as f32 + b_co;
                        want[idx] = match res_mode {
                            ResidualMode::PreAct => act.apply(v + beta * res[idx]),
                            ResidualMode::PostAct => act.apply(v) + beta * res[idx],
                            ResidualMode::None => act.apply(v),
                        };
                    }
                }
            }
        }
        let gx = GpuTensor::from_slice(&ctx, &x, &[x.len()]).unwrap();
        let gw = GpuTensor::from_slice(&ctx, &wt, &[wt.len()]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &bias, &[cout]).unwrap();
        let gr = GpuTensor::from_slice(&ctx, &res, &[res.len()]).unwrap();
        let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F32).unwrap();
        conv2d_epilogue(
            &ctx,
            &d,
            gx.raw(),
            gw.raw(),
            Some(gb.raw()),
            Some((gr.raw(), beta, res_mode)),
            gy.raw(),
        )
        .unwrap();
        assert_close(
            &gy.to_vec().unwrap(),
            &want,
            1e-4,
            1e-4,
            &format!("conv2d_epilogue {cin}->{cout} {kh}x{kw} g={groups} {res_mode:?} {act:?}"),
        );
        // Output-residual aliasing rejected cleanly.
        assert!(
            conv2d_epilogue(
                &ctx,
                &d,
                gx.raw(),
                gw.raw(),
                Some(gb.raw()),
                Some((gy.raw(), beta, res_mode)),
                gy.raw(),
            )
            .is_err()
        );
    }

    // Depthwise F16 single-rounding check: large raw conv sum (~70000, exceeds F16 max 65504) cancelled by
    // residual (`-70000`) stays finite (~1.0), proving F32 intermediate storage before bias/residual/act.
    if ctx.caps().shader_f16 {
        let d = Conv2dDesc {
            n: 1,
            c_in: 4,
            h: 4,
            w: 4,
            c_out: 4,
            groups: 4,
            win: Window2d::square(3, 1, 1),
            act: UnaryOp::Identity,
            dtype: DType::F16,
        };
        // Center pixels (1..3, 1..3) see 9 taps of 100 * 80 = 72000 (> F16 max 65504); bias=-36000,
        // res=-35968 (exact in F16, ULP=32 at 2^15) -> +32.0 (would be +Inf if intermediate rounded to F16).
        let x = vec![100.0f32; 4 * 4 * 4];
        let wt = vec![80.0f32; 4 * 9];
        let bias = vec![-36000.0f32; 4];
        let res = vec![-35968.0f32; 4 * 4 * 4];
        let gx = upload_dt(&ctx, &x, DType::F16);
        let gw = upload_dt(&ctx, &wt, DType::F16);
        let gb = upload_dt(&ctx, &bias, DType::F16);
        let gr = upload_dt(&ctx, &res, DType::F16);
        let gy = GpuTensor::empty(&ctx, &[4 * 4 * 4], DType::F16).unwrap();
        conv2d_epilogue(
            &ctx,
            &d,
            gx.raw(),
            gw.raw(),
            Some(gb.raw()),
            Some((gr.raw(), 1.0, ResidualMode::PreAct)),
            gy.raw(),
        )
        .unwrap();
        let got = gy.to_vec().unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "depthwise f16 epilogue overflowed intermediate: {got:?}");
        // Interior pixel (1, 1) of channel 0 has all 9 taps: 9*8000 - 36000 - 35968 = 32.0.
        assert!((got[5] - 32.0).abs() < 1e-3, "depthwise f16 interior got {}", got[5]);
    }

    // 3. Standalone bias_residual_act (NCHW vec4, linear vec4, odd scalar; Silu/Gelu/Relu/Identity; F32 & F16).
    for &dtype in dtypes {
        for &(total, c, inner, act, res_mode) in &[
            (2 * 6 * 16, 6, 16, FusedAct::Gelu, ResidualMode::PreAct),
            (5 * 12, 12, 1, FusedAct::Relu, ResidualMode::PostAct),
            (2 * 5 * 7, 5, 7, FusedAct::Silu, ResidualMode::PreAct),
            (13, 13, 1, FusedAct::Identity, ResidualMode::PostAct),
        ] {
            let mut x = rng.vec(total, -3.0, 3.0);
            let mut b = rng.vec(c, -0.5, 0.5);
            let mut res = rng.vec(total, -1.0, 1.0);
            if dtype == DType::F16 {
                x = f16_round(&x);
                b = f16_round(&b);
                res = f16_round(&res);
            }
            let beta = 0.7f32;
            let apply_act = |v: f32| match act {
                FusedAct::Identity => v,
                FusedAct::Silu => UnaryOp::Silu.apply(v),
                FusedAct::Gelu => UnaryOp::Gelu.apply(v),
                FusedAct::Relu => v.max(0.0),
            };
            let want: Vec<f32> = (0..total)
                .map(|i| {
                    let v = x[i] + b[(i / inner) % c];
                    match res_mode {
                        ResidualMode::PreAct => apply_act(v + beta * res[i]),
                        ResidualMode::PostAct => apply_act(v) + beta * res[i],
                        ResidualMode::None => apply_act(v),
                    }
                })
                .collect();
            let gx = upload_dt(&ctx, &x, dtype);
            let gb = upload_dt(&ctx, &b, dtype);
            let gr = upload_dt(&ctx, &res, dtype);
            let gy = GpuTensor::empty(&ctx, &[total], dtype).unwrap();
            bias_residual_act(
                &ctx,
                gx.raw(),
                Some(gb.raw()),
                Some((gr.raw(), beta, res_mode)),
                gy.raw(),
                total,
                c,
                inner,
                act,
                dtype,
            )
            .unwrap();
            let (atol, rtol) = if dtype == DType::F16 { (8e-3, 5e-3) } else { (5e-5, 2e-5) };
            assert_close(
                &gy.to_vec().unwrap(),
                &want,
                atol,
                rtol,
                &format!("bias_residual_act n={total} {act:?} {res_mode:?} {dtype:?}"),
            );
        }
    }
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn rms_and_layer_norm_f16_and_extreme_backward_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x58e0_0002);

    // 1. Extreme-scale RMSNorm backward stability (`1e20`, `1e-23`, and `1e35` where `rstd = 1e-35 < 1e-30`
    //    exercises the `ldexp` guard in `norm_xh` while remaining normal in IEEE-754 `f32`).
    for portable in [false, true] {
        let _fg = ForceGuard::set(portable);
        for vals in
            [[1e20f32, -1e20, 1e20, -1e20], [1e-23, -1e-23, 1e-23, -1e-23], [1e35, -1e35, 1e35, -1e35]]
        {
            let g_vals = [1.0f32, 2.0, -1.0, 0.5];
            let gx = GpuTensor::from_slice(&ctx, &vals, &[4]).unwrap();
            let gg = GpuTensor::from_slice(&ctx, &g_vals, &[4]).unwrap();
            let gy = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
            let grs = GpuTensor::empty(&ctx, &[1], DType::F32).unwrap();
            let gdx = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
            let gdw = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
            rms_norm_forward_dtype(&ctx, gx.raw(), None, gy.raw(), grs.raw(), 1, 4, 0.0, DType::F32).unwrap();
            rms_norm_backward_dtype(
                &ctx,
                gx.raw(),
                gg.raw(),
                None,
                grs.raw(),
                gdx.raw(),
                Some(gdw.raw()),
                1,
                4,
                DType::F32,
            )
            .unwrap();
            let got_dx = gdx.to_vec().unwrap();
            let got_dw = gdw.to_vec().unwrap();
            assert!(got_dx.iter().all(|v| v.is_finite()), "rms ext dx non-finite: {got_dx:?}");
            assert!(got_dw.iter().all(|v| v.is_finite()), "rms ext dw non-finite: {got_dw:?}");
            // xh = [1, -1, 1, -1] => dw = g * xh = [1, -2, -1, -0.5].
            assert_close(&got_dw, &[1.0, -2.0, -1.0, -0.5], 1e-5, 1e-5, &format!("rms ext dw p={portable}"));
        }
    }

    // 2. F16 RMSNorm and LayerNorm forward + backward parity (vec4 cached, odd scalar, split-K outer=300).
    if ctx.caps().shader_f16 {
        for &(outer, d, label) in
            &[(6usize, 32usize, "f16 vec4"), (5, 17, "f16 scalar"), (300, 16, "f16 split-k")]
        {
            let total = outer * d;
            let x = f16_round(&rng.vec(total, -1.5, 1.5));
            let grad = f16_round(&rng.vec(total, -1.0, 1.0));
            let w = rng.vec(d, 0.5, 1.5);
            let b = rng.vec(d, -0.5, 0.5);
            let eps = 1e-4f32;

            // CPU f64 references for RMSNorm and LayerNorm.
            let mut y_rms = vec![0.0f32; total];
            let mut dx_rms = vec![0.0f32; total];
            let mut dw_rms = vec![0.0f64; d];
            let mut y_ln = vec![0.0f32; total];
            let mut dx_ln = vec![0.0f32; total];
            let mut dw_ln = vec![0.0f64; d];
            let mut db_ln = vec![0.0f64; d];
            for o in 0..outer {
                let row = &x[o * d..(o + 1) * d];
                let gr = &grad[o * d..(o + 1) * d];
                let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / d as f64;
                let rstd_r = 1.0 / (ms + eps as f64).sqrt();
                let mut c_r = 0.0f64;
                for i in 0..d {
                    let xh = row[i] as f64 * rstd_r;
                    y_rms[o * d + i] = (xh * w[i] as f64) as f32;
                    let gw = gr[i] as f64 * w[i] as f64;
                    c_r += gw * xh;
                    dw_rms[i] += gr[i] as f64 * xh;
                }
                c_r /= d as f64;
                for i in 0..d {
                    let xh = row[i] as f64 * rstd_r;
                    let gw = gr[i] as f64 * w[i] as f64;
                    dx_rms[o * d + i] = (rstd_r * (gw - xh * c_r)) as f32;
                }

                let mu = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
                let var = row.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / d as f64;
                let rstd_l = 1.0 / (var + eps as f64).sqrt();
                let (mut s1, mut s2) = (0.0f64, 0.0f64);
                for i in 0..d {
                    let xh = (row[i] as f64 - mu) * rstd_l;
                    y_ln[o * d + i] = (xh * w[i] as f64 + b[i] as f64) as f32;
                    let gw = gr[i] as f64 * w[i] as f64;
                    s1 += gw;
                    s2 += gw * xh;
                    dw_ln[i] += gr[i] as f64 * xh;
                    db_ln[i] += gr[i] as f64;
                }
                for i in 0..d {
                    let xh = (row[i] as f64 - mu) * rstd_l;
                    let gw = gr[i] as f64 * w[i] as f64;
                    dx_ln[o * d + i] = (rstd_l * (gw - s1 / d as f64 - xh * (s2 / d as f64))) as f32;
                }
            }
            let dw_rms: Vec<f32> = dw_rms.into_iter().map(|v| v as f32).collect();
            let dw_ln: Vec<f32> = dw_ln.into_iter().map(|v| v as f32).collect();
            let db_ln: Vec<f32> = db_ln.into_iter().map(|v| v as f32).collect();

            let gx = upload_dt(&ctx, &x, DType::F16);
            let gg = upload_dt(&ctx, &grad, DType::F16);
            let gw = GpuTensor::from_slice(&ctx, &w, &[d]).unwrap();
            let gb = GpuTensor::from_slice(&ctx, &b, &[d]).unwrap();
            let gy = GpuTensor::empty(&ctx, &[total], DType::F16).unwrap();
            let gm = GpuTensor::empty(&ctx, &[outer], DType::F32).unwrap();
            let grs = GpuTensor::empty(&ctx, &[outer], DType::F32).unwrap();
            let gdx = GpuTensor::empty(&ctx, &[total], DType::F16).unwrap();
            let gdw = GpuTensor::empty(&ctx, &[d], DType::F32).unwrap();
            let gdb = GpuTensor::empty(&ctx, &[d], DType::F32).unwrap();

            for portable in [false, true] {
                let _fg = ForceGuard::set(portable);
                rms_norm_forward_dtype(
                    &ctx,
                    gx.raw(),
                    Some(gw.raw()),
                    gy.raw(),
                    grs.raw(),
                    outer,
                    d,
                    eps,
                    DType::F16,
                )
                .unwrap();
                rms_norm_backward_dtype(
                    &ctx,
                    gx.raw(),
                    gg.raw(),
                    Some(gw.raw()),
                    grs.raw(),
                    gdx.raw(),
                    Some(gdw.raw()),
                    outer,
                    d,
                    DType::F16,
                )
                .unwrap();
                let dw_tol = if outer >= 256 { 3e-2 } else { 8e-3 };
                assert_close(
                    &gy.to_vec().unwrap(),
                    &y_rms,
                    6e-3,
                    5e-3,
                    &format!("rms {label} y p={portable}"),
                );
                assert_close(
                    &gdx.to_vec().unwrap(),
                    &dx_rms,
                    8e-3,
                    6e-3,
                    &format!("rms {label} dx p={portable}"),
                );
                assert_close(
                    &gdw.to_vec().unwrap(),
                    &dw_rms,
                    dw_tol,
                    5e-3,
                    &format!("rms {label} dw p={portable}"),
                );

                layer_norm_forward_dtype(
                    &ctx,
                    gx.raw(),
                    Some(gw.raw()),
                    Some(gb.raw()),
                    gy.raw(),
                    gm.raw(),
                    grs.raw(),
                    outer,
                    d,
                    eps,
                    DType::F16,
                )
                .unwrap();
                layer_norm_backward_dtype(
                    &ctx,
                    gx.raw(),
                    gg.raw(),
                    Some(gw.raw()),
                    gm.raw(),
                    grs.raw(),
                    gdx.raw(),
                    Some(gdw.raw()),
                    Some(gdb.raw()),
                    outer,
                    d,
                    DType::F16,
                )
                .unwrap();
                assert_close(&gy.to_vec().unwrap(), &y_ln, 6e-3, 5e-3, &format!("ln {label} y p={portable}"));
                assert_close(
                    &gdx.to_vec().unwrap(),
                    &dx_ln,
                    8e-3,
                    6e-3,
                    &format!("ln {label} dx p={portable}"),
                );
                assert_close(
                    &gdw.to_vec().unwrap(),
                    &dw_ln,
                    dw_tol,
                    5e-3,
                    &format!("ln {label} dw p={portable}"),
                );
                assert_close(
                    &gdb.to_vec().unwrap(),
                    &db_ln,
                    dw_tol,
                    5e-3,
                    &format!("ln {label} db p={portable}"),
                );
            }
        }
    }
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
