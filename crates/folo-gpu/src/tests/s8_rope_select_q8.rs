//! GPU-vs-CPU parity, edge-case, and finite-difference gradient tests for:
//! - 2D Rotary Positional Embedding (`rope::rope2d_*` table & on-the-fly trig, forward & backward).
//! - Vectorized `where_select` with non-multiple-of-4 tails & broadcasts (`select::where_select`).
//! - Packed int8 weight-dequantized matmul & GEMV (`q8_matmul::q8_matmul`).

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::FORCE_NO_SUBGROUPS;
use crate::kernels::q8_matmul::{Q8MatmulDesc, pack_i8_weights, q8_matmul};
use crate::kernels::rope::{
    Rope2dDesc, RopeLayout, rope2d_backward, rope2d_forward, rope2d_trig_backward, rope2d_trig_forward,
};
use crate::kernels::select::{SelectBroadcast, SelectOperand, where_select};
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

#[allow(clippy::too_many_arguments)]
fn cpu_rope2d(
    x: &[f32],
    desc: &Rope2dDesc,
    cos_y: &[f32],
    sin_y: &[f32],
    cos_x: &[f32],
    sin_x: &[f32],
    backward: bool,
) -> Vec<f32> {
    let mut y = x.to_vec();
    let d = desc.head_dim;
    let hy_half = desc.dim_y / 2;
    let hx_half = desc.dim_x / 2;
    let sign = if backward { -1.0f64 } else { 1.0f64 };
    for o in 0..desc.outer {
        for gy in 0..desc.height {
            for gx in 0..desc.width {
                let base = ((o * desc.height + gy) * desc.width + gx) * d;
                for p in 0..hy_half {
                    let (i0, i1) = match desc.layout {
                        RopeLayout::Interleaved => (2 * p, 2 * p + 1),
                        RopeLayout::HalfSplit => (p, p + hy_half),
                    };
                    let c = cos_y[gy * hy_half + p] as f64;
                    let s = sign * sin_y[gy * hy_half + p] as f64;
                    let (a, b) = (x[base + i0] as f64, x[base + i1] as f64);
                    y[base + i0] = (a * c - b * s) as f32;
                    y[base + i1] = (a * s + b * c) as f32;
                }
                for p in 0..hx_half {
                    let (i0, i1) = match desc.layout {
                        RopeLayout::Interleaved => (desc.dim_y + 2 * p, desc.dim_y + 2 * p + 1),
                        RopeLayout::HalfSplit => (desc.dim_y + p, desc.dim_y + p + hx_half),
                    };
                    let c = cos_x[gx * hx_half + p] as f64;
                    let s = sign * sin_x[gx * hx_half + p] as f64;
                    let (a, b) = (x[base + i0] as f64, x[base + i1] as f64);
                    y[base + i0] = (a * c - b * s) as f32;
                    y[base + i1] = (a * s + b * c) as f32;
                }
            }
        }
    }
    y
}

#[test]
fn rope2d_forward_and_backward_match_cpu_and_finite_diff() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x58e0_0003);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // Covers: vec4 interleaved aligned (D=16), HalfSplit (D=16), non-multiple-of-4 odd D (D=10, D=7 with
    // partial rotation + pass-through tail), 1x1 grid, and dim_x = 0 degenerate.
    let cases = [
        (3usize, 5usize, 7usize, 16usize, 8usize, 8usize, RopeLayout::Interleaved),
        (2, 4, 6, 16, 8, 8, RopeLayout::HalfSplit),
        (2, 3, 5, 10, 4, 4, RopeLayout::Interleaved),
        (2, 3, 4, 7, 2, 2, RopeLayout::HalfSplit),
        (1, 1, 1, 8, 4, 4, RopeLayout::Interleaved),
        (2, 4, 3, 8, 4, 0, RopeLayout::HalfSplit),
    ];

    for &dtype in dtypes {
        for &(outer, h, w, d, dy, dx, layout) in &cases {
            let desc =
                Rope2dDesc { outer, height: h, width: w, head_dim: d, dim_y: dy, dim_x: dx, layout, dtype };
            let total = outer * h * w * d;
            let mut x = rng.vec(total, -2.0, 2.0);
            let mut grad = rng.vec(total, -1.5, 1.5);
            if dtype == DType::F16 {
                x = f16_round(&x);
                grad = f16_round(&grad);
            }
            let (base_y, base_x) = (10000.0f32, 5000.0f32);
            let mut cos_y = Vec::with_capacity(h * (dy / 2));
            let mut sin_y = Vec::with_capacity(h * (dy / 2));
            for gy in 0..h {
                for p in 0..(dy / 2) {
                    let ang = gy as f64 * (base_y as f64).powf(-2.0 * p as f64 / dy as f64);
                    cos_y.push(ang.cos() as f32);
                    sin_y.push(ang.sin() as f32);
                }
            }
            let mut cos_x = Vec::with_capacity(w * (dx / 2));
            let mut sin_x = Vec::with_capacity(w * (dx / 2));
            for gx in 0..w {
                for p in 0..(dx / 2) {
                    let ang = gx as f64 * (base_x as f64).powf(-2.0 * p as f64 / dx as f64);
                    cos_x.push(ang.cos() as f32);
                    sin_x.push(ang.sin() as f32);
                }
            }

            let want_fwd = cpu_rope2d(&x, &desc, &cos_y, &sin_y, &cos_x, &sin_x, false);
            let want_bwd = cpu_rope2d(&grad, &desc, &cos_y, &sin_y, &cos_x, &sin_x, true);

            let gx = upload_dt(&ctx, &x, dtype);
            let gg = upload_dt(&ctx, &grad, dtype);
            let gy_t = GpuTensor::empty(&ctx, &[total], dtype).unwrap();
            let gdx_t = GpuTensor::empty(&ctx, &[total], dtype).unwrap();
            let gcy = if cos_y.is_empty() {
                None
            } else {
                Some(GpuTensor::from_slice(&ctx, &cos_y, &[cos_y.len()]).unwrap())
            };
            let gsy = if sin_y.is_empty() {
                None
            } else {
                Some(GpuTensor::from_slice(&ctx, &sin_y, &[sin_y.len()]).unwrap())
            };
            let gcx = if cos_x.is_empty() {
                None
            } else {
                Some(GpuTensor::from_slice(&ctx, &cos_x, &[cos_x.len()]).unwrap())
            };
            let gsx = if sin_x.is_empty() {
                None
            } else {
                Some(GpuTensor::from_slice(&ctx, &sin_x, &[sin_x.len()]).unwrap())
            };

            rope2d_forward(
                &ctx,
                &desc,
                gx.raw(),
                gcy.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gsy.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gcx.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gsx.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gy_t.raw(),
            )
            .unwrap();
            rope2d_backward(
                &ctx,
                &desc,
                gg.raw(),
                gcy.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gsy.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gcx.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gsx.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                gdx_t.raw(),
            )
            .unwrap();

            let (atol, rtol) = if dtype == DType::F16 { (6e-3, 5e-3) } else { (1e-5, 1e-5) };
            assert_close(&gy_t.to_vec().unwrap(), &want_fwd, atol, rtol, &format!("rope table fwd {desc:?}"));
            assert_close(
                &gdx_t.to_vec().unwrap(),
                &want_bwd,
                atol,
                rtol,
                &format!("rope table bwd {desc:?}"),
            );

            // On-the-fly trig forward & backward + round-trip invertibility `bwd(fwd(x)) == x`.
            let gy_trig = GpuTensor::empty(&ctx, &[total], dtype).unwrap();
            let gdx_trig = GpuTensor::empty(&ctx, &[total], dtype).unwrap();
            let roundtrip = GpuTensor::empty(&ctx, &[total], dtype).unwrap();
            rope2d_trig_forward(&ctx, &desc, gx.raw(), gy_trig.raw(), base_y, base_x).unwrap();
            rope2d_trig_backward(&ctx, &desc, gg.raw(), gdx_trig.raw(), base_y, base_x).unwrap();
            rope2d_trig_backward(&ctx, &desc, gy_trig.raw(), roundtrip.raw(), base_y, base_x).unwrap();
            assert_close(
                &gy_trig.to_vec().unwrap(),
                &want_fwd,
                atol * 2.0,
                rtol,
                &format!("rope trig fwd {desc:?}"),
            );
            assert_close(
                &gdx_trig.to_vec().unwrap(),
                &want_bwd,
                atol * 2.0,
                rtol,
                &format!("rope trig bwd {desc:?}"),
            );
            assert_close(
                &roundtrip.to_vec().unwrap(),
                &x,
                if dtype == DType::F16 { 1e-2 } else { 2e-5 },
                rtol,
                &format!("rope roundtrip {desc:?}"),
            );
        }
    }

    // Empty tensor & validation checks.
    let empty_desc = Rope2dDesc::split_even(0, 4, 4, 8, RopeLayout::HalfSplit, DType::F32);
    rope2d_trig_forward(&ctx, &empty_desc, ctx.dummy(), ctx.dummy(), 10000.0, 10000.0).unwrap();
    let bad_desc = Rope2dDesc {
        outer: 1,
        height: 2,
        width: 2,
        head_dim: 6,
        dim_y: 3,
        dim_x: 2,
        layout: RopeLayout::HalfSplit,
        dtype: DType::F32,
    };
    assert!(rope2d_trig_forward(&ctx, &bad_desc, ctx.dummy(), ctx.dummy(), 10000.0, 10000.0).is_err());

    // Finite-difference gradient check on RoPE2d (both layouts).
    for layout in [RopeLayout::Interleaved, RopeLayout::HalfSplit] {
        let desc = Rope2dDesc {
            outer: 1,
            height: 2,
            width: 3,
            head_dim: 6,
            dim_y: 2,
            dim_x: 2,
            layout,
            dtype: DType::F32,
        };
        let n = 2 * 3 * 6;
        let x = rng.vec(n, -1.5, 1.5);
        let grad = rng.vec(n, -1.0, 1.0);
        let gg = GpuTensor::from_slice(&ctx, &grad, &[n]).unwrap();
        let gdx = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        rope2d_trig_backward(&ctx, &desc, gg.raw(), gdx.raw(), 100.0, 100.0).unwrap();
        let got_dx = gdx.to_vec().unwrap();
        let eps = 1e-3f32;
        for i in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let tp = GpuTensor::from_slice(&ctx, &xp, &[n]).unwrap();
            let tm = GpuTensor::from_slice(&ctx, &xm, &[n]).unwrap();
            let yp = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
            let ym = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
            rope2d_trig_forward(&ctx, &desc, tp.raw(), yp.raw(), 100.0, 100.0).unwrap();
            rope2d_trig_forward(&ctx, &desc, tm.raw(), ym.raw(), 100.0, 100.0).unwrap();
            let lp: f64 = yp.to_vec().unwrap().iter().zip(&grad).map(|(&a, &b)| a as f64 * b as f64).sum();
            let lm: f64 = ym.to_vec().unwrap().iter().zip(&grad).map(|(&a, &b)| a as f64 * b as f64).sum();
            let fd = ((lp - lm) / (2.0 * eps as f64)) as f32;
            assert!((got_dx[i] - fd).abs() < 2e-3, "rope fd {layout:?}[{i}]: got {}, fd {fd}", got_dx[i]);
        }
    }
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn where_select_vec4_tails_and_broadcasts_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x58e0_0004);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // 1. Dense + non-multiple-of-4 tails (`n = 1, 3, 4, 5, 7, 13, 1024, 1027`), F32 & U8 cond, F32 & F16 values.
    for &dtype in dtypes {
        for &n in &[1usize, 3, 4, 5, 7, 13, 1024, 1027] {
            let mut a = rng.vec(n, -5.0, 5.0);
            let mut b = rng.vec(n, -5.0, 5.0);
            if dtype == DType::F16 {
                a = f16_round(&a);
                b = f16_round(&b);
            }
            let mut cond_f32: Vec<f32> =
                (0..n).map(|i| if i % 3 == 0 { 0.0 } else { (i as f32) - 1.5 }).collect();
            if n >= 5 {
                cond_f32[1] = -0.0;
                cond_f32[2] = f32::NAN;
            }
            let cond_u8: Vec<u8> =
                (0..n).map(|i| if i % 5 == 0 { 0u8 } else { ((i % 7) + 1) as u8 }).collect();
            let mut cond_u8_padded = cond_u8.clone();
            cond_u8_padded.resize(cond_u8.len().next_multiple_of(4), 0);

            let ga = upload_dt(&ctx, &a, dtype);
            let gb = upload_dt(&ctx, &b, dtype);
            let gcf = GpuTensor::from_slice(&ctx, &cond_f32, &[n]).unwrap();
            let gcu = ctx.upload(&cond_u8_padded).unwrap();
            let out = GpuTensor::empty(&ctx, &[n], dtype).unwrap();

            where_select(
                &ctx,
                SelectOperand::dense(gcf.raw()),
                false,
                SelectOperand::dense(ga.raw()),
                SelectOperand::dense(gb.raw()),
                out.raw(),
                n,
                dtype,
            )
            .unwrap();
            let want_f: Vec<f32> = (0..n)
                .map(|i| if (cond_f32[i].to_bits() & 0x7fff_ffff) != 0 { a[i] } else { b[i] })
                .collect();
            assert_close(
                &out.to_vec().unwrap(),
                &want_f,
                0.0,
                0.0,
                &format!("where_select f32 n={n} {dtype:?}"),
            );

            where_select(
                &ctx,
                SelectOperand::dense(gcu.raw()),
                true,
                SelectOperand::dense(ga.raw()),
                SelectOperand::dense(gb.raw()),
                out.raw(),
                n,
                dtype,
            )
            .unwrap();
            let want_u: Vec<f32> = (0..n).map(|i| if cond_u8[i] != 0 { a[i] } else { b[i] }).collect();
            assert_close(
                &out.to_vec().unwrap(),
                &want_u,
                0.0,
                0.0,
                &format!("where_select u8 n={n} {dtype:?}"),
            );
        }
    }

    // 2. Broadcast patterns: Scalar, RepeatInner (aligned & odd), TileInner (aligned & odd), and empty n = 0.
    for &(n, inner) in &[(48usize, 8usize), (35, 7)] {
        let outer = n / inner;
        let cond_rep: Vec<u8> = (0..outer).map(|i| u8::from(i % 2 == 1)).collect();
        let mut cond_rep_pad = cond_rep.clone();
        cond_rep_pad.resize(cond_rep.len().next_multiple_of(4), 0);
        let a_tile = rng.vec(inner, -3.0, 3.0);
        let b_scalar = [-42.25f32];
        let gcu = ctx.upload(&cond_rep_pad).unwrap();
        let ga = GpuTensor::from_slice(&ctx, &a_tile, &[inner]).unwrap();
        let gb = GpuTensor::from_slice(&ctx, &b_scalar, &[1]).unwrap();
        let out = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        where_select(
            &ctx,
            SelectOperand { buf: gcu.raw(), broadcast: SelectBroadcast::RepeatInner { inner } },
            true,
            SelectOperand { buf: ga.raw(), broadcast: SelectBroadcast::TileInner { inner } },
            SelectOperand::scalar(gb.raw()),
            out.raw(),
            n,
            DType::F32,
        )
        .unwrap();
        let want: Vec<f32> =
            (0..n).map(|i| if cond_rep[i / inner] != 0 { a_tile[i % inner] } else { b_scalar[0] }).collect();
        assert_close(
            &out.to_vec().unwrap(),
            &want,
            0.0,
            0.0,
            &format!("where_select bcast n={n} inner={inner}"),
        );
    }

    where_select(
        &ctx,
        SelectOperand::dense(ctx.dummy()),
        true,
        SelectOperand::scalar(ctx.dummy()),
        SelectOperand::dense(ctx.dummy()),
        ctx.dummy(),
        0,
        DType::F32,
    )
    .unwrap();
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn q8_matmul_and_gemv_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x58e0_0005);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    let cases = [
        (1usize, 1usize, 13usize, 19usize, 0usize, UnaryOp::Identity, true),
        (2, 1, 64, 96, 32, UnaryOp::Silu, true),
        (1, 1, 7, 13, 4, UnaryOp::Gelu, false),
        (1, 64, 64, 64, 0, UnaryOp::Relu, true),
        (2, 17, 29, 37, 16, UnaryOp::Silu, true),
        (1, 67, 71, 41, 0, UnaryOp::Gelu, true),
        (2, 5, 9, 0, 0, UnaryOp::Silu, true),
    ];

    for &dtype in dtypes {
        for &(batch, m, n, k, group_size, act, has_bias) in &cases {
            let d = Q8MatmulDesc { batch, m, n, k, group_size, act, dtype };
            let mut a = rng.vec(batch * m * k, -1.0, 1.0);
            let mut bias = rng.vec(n, -0.5, 0.5);
            if dtype == DType::F16 {
                a = f16_round(&a);
                bias = f16_round(&bias);
            }
            let weights_i8: Vec<i8> = (0..n * k).map(|_| (rng.next_u64() % 255) as i32 as i8).collect();
            let num_groups = d.num_groups();
            let scales = rng.vec(n * num_groups, 0.002, 0.02);
            let packed = pack_i8_weights(&weights_i8, n, k);

            let mut want = vec![0.0f32; batch * m * n];
            for b in 0..batch {
                for r in 0..m {
                    for c in 0..n {
                        let mut acc = 0.0f64;
                        for kk in 0..k {
                            let g = kk.checked_div(group_size).unwrap_or(0);
                            let w_dq = weights_i8[c * k + kk] as f64 * scales[c * num_groups + g] as f64;
                            acc += a[(b * m + r) * k + kk] as f64 * w_dq;
                        }
                        let v = acc as f32 + if has_bias { bias[c] } else { 0.0 };
                        want[(b * m + r) * n + c] = act.apply(v);
                    }
                }
            }

            let ga = if a.is_empty() { None } else { Some(upload_dt(&ctx, &a, dtype)) };
            let gw = if packed.is_empty() {
                None
            } else {
                Some(ctx.upload(bytemuck::cast_slice(&packed)).unwrap())
            };
            let gs = GpuTensor::from_slice(&ctx, &scales, &[scales.len()]).unwrap();
            let gb = upload_dt(&ctx, &bias, dtype);
            let gy = GpuTensor::empty(&ctx, &[batch * m * n], dtype).unwrap();

            let modes: &[bool] = if m == 1 { &[false, true] } else { &[false] };
            for &portable in modes {
                let _fg = ForceGuard::set(portable);
                q8_matmul(
                    &ctx,
                    &d,
                    ga.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                    gw.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                    gs.raw(),
                    has_bias.then(|| gb.raw()),
                    gy.raw(),
                )
                .unwrap();
                let (atol, rtol) = if dtype == DType::F16 {
                    (1.5e-2, 8e-3)
                } else {
                    (5e-5 * (k.max(1) as f32).sqrt(), 5e-5)
                };
                assert_close(
                    &gy.to_vec().unwrap(),
                    &want,
                    atol,
                    rtol,
                    &format!(
                        "q8_matmul b={batch} {m}x{n}x{k} gs={group_size} {act:?} {dtype:?} p={portable}"
                    ),
                );
            }
        }
    }

    let empty_d = Q8MatmulDesc::per_channel(1, 0, 8, 16, DType::F32);
    let dummy_scales = GpuTensor::zeros(&ctx, &[8], DType::F32).unwrap();
    let dummy_w = GpuTensor::zeros(&ctx, &[8 * 4], DType::F32).unwrap();
    q8_matmul(&ctx, &empty_d, ctx.dummy(), dummy_w.raw(), dummy_scales.raw(), None, ctx.dummy()).unwrap();
    let bad_gs = Q8MatmulDesc { group_size: 6, ..Q8MatmulDesc::per_channel(1, 2, 4, 12, DType::F32) };
    assert!(q8_matmul(&ctx, &bad_gs, ctx.dummy(), ctx.dummy(), ctx.dummy(), None, ctx.dummy()).is_err());
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
