//! GPU-vs-CPU parity, edge-case, and finite-difference gradient tests for Sprint 9 WGSL kernels:
//! - Q4 block-dequantizing matmul & GEMV (`q4_matmul::q4_matmul`, matching `folo_core::cpu::q4::Q4Block`).
//! - Parallel prefix-sum (`cumsum::cumsum` / `cumsum_ex`) along arbitrary axis (`F32` and `F16`).
//! - Batched arbitrary-axis `topk` (`batched_topk::batched_topk`) with deterministic tie-breaking matching CPU.
//! - Shared-memory tiled `im2col` & atomic-free tiled gather `col2im` (`im2col::{im2col, col2im}`).
//! - Fused `SwiGLU` & `GeGLU` forward + backward (`fused_act::{swiglu_*, geglu_*}`) + finite-difference `grad_check`.

use folo_core::cpu::q4::{Q4Matrix, matmul_q4_f32};

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::FORCE_NO_SUBGROUPS;
use crate::kernels::batched_topk::{BatchedTopkDesc, batched_topk};
use crate::kernels::cumsum::{cumsum, cumsum_ex};
use crate::kernels::fused_act::{geglu_backward, geglu_forward, swiglu_backward, swiglu_forward};
use crate::kernels::im2col::{Im2ColDesc, col2im, im2col};
use crate::kernels::pooling::Window2d;
use crate::kernels::q4_matmul::{Q4MatmulDesc, pack_q4_matrix, q4_matmul};
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

fn read_i64(ctx: &crate::GpuContext, buf: &wgpu::Buffer, count: usize) -> Vec<i64> {
    ctx.read_with(buf, 0, (count * 8) as u64, |bytes| {
        bytes.as_chunks::<8>().0.iter().map(|&c| i64::from_le_bytes(c)).collect()
    })
    .unwrap()
}

#[test]
fn q4_matmul_and_gemv_match_folo_core_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5900_0001);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // Covers M == 1 (GEMV, subgroup + portable), tile-aligned 64x64x32, odd tails on M/N/K (including K < 32,
    // K = 31, K = 33, K = 97), batched, fused bias + activations, and K == 0.
    let cases = [
        (1usize, 1usize, 13usize, 19usize, UnaryOp::Identity, true),
        (2, 1, 64, 96, UnaryOp::Silu, true),
        (1, 1, 7, 33, UnaryOp::Gelu, false),
        (1, 64, 64, 64, UnaryOp::Relu, true),
        (2, 17, 29, 31, UnaryOp::Identity, true),
        (1, 67, 71, 97, UnaryOp::Silu, true),
        (2, 5, 9, 0, UnaryOp::Gelu, true),
    ];

    for &dtype in dtypes {
        for &(batch, m, n, k, act, has_bias) in &cases {
            let d = Q4MatmulDesc { batch, m, n, k, act, dtype };
            let mut a = rng.vec(batch * m * k, -1.2, 1.2);
            let mut bias = rng.vec(n, -0.4, 0.4);
            if dtype == DType::F16 {
                a = f16_round(&a);
                bias = f16_round(&bias);
            }
            let raw_w = rng.vec(n * k, -1.5, 1.5);
            let q4 = Q4Matrix::quantize(&raw_w, n, k);
            let packed = pack_q4_matrix(&q4);

            // Reference directly from `folo_core::cpu::q4::matmul_q4_f32`.
            let mut want = vec![0.0f32; batch * m * n];
            for b in 0..batch {
                let a_slice = &a[b * m * k..(b + 1) * m * k];
                let out_slice = &mut want[b * m * n..(b + 1) * m * n];
                matmul_q4_f32(a_slice, m, k, &q4, has_bias.then_some(bias.as_slice()), out_slice);
                for v in out_slice.iter_mut() {
                    *v = act.apply(*v);
                }
            }

            let ga = if a.is_empty() { None } else { Some(upload_dt(&ctx, &a, dtype)) };
            let gw = if packed.is_empty() {
                None
            } else {
                Some(ctx.upload(bytemuck::cast_slice(&packed)).unwrap())
            };
            let gb = upload_dt(&ctx, &bias, dtype);
            let gy = GpuTensor::empty(&ctx, &[batch * m * n], dtype).unwrap();

            let modes: &[bool] = if m == 1 { &[false, true] } else { &[false] };
            for &portable in modes {
                let _fg = ForceGuard::set(portable);
                q4_matmul(
                    &ctx,
                    &d,
                    ga.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                    gw.as_ref().map_or(ctx.dummy(), |t| t.raw()),
                    has_bias.then(|| gb.raw()),
                    gy.raw(),
                )
                .unwrap();
                let (atol, rtol) =
                    if dtype == DType::F16 { (2e-2, 1e-2) } else { (5e-5 * (k.max(1) as f32).sqrt(), 5e-5) };
                assert_close(
                    &gy.to_vec().unwrap(),
                    &want,
                    atol,
                    rtol,
                    &format!("q4_matmul b={batch} {m}x{n}x{k} {act:?} {dtype:?} p={portable}"),
                );
            }
        }
    }

    // Empty M == 0 no-op and aliasing rejection.
    let empty_d = Q4MatmulDesc::new(1, 0, 8, 32, DType::F32);
    let dummy_w = GpuTensor::zeros(&ctx, &[8 * 5], DType::F32).unwrap();
    q4_matmul(&ctx, &empty_d, ctx.dummy(), dummy_w.raw(), None, ctx.dummy()).unwrap();
    let alias_d = Q4MatmulDesc::new(1, 2, 2, 2, DType::F32);
    let buf = GpuTensor::zeros(&ctx, &[16], DType::F32).unwrap();
    assert!(q4_matmul(&ctx, &alias_d, buf.raw(), dummy_w.raw(), None, buf.raw()).is_err());
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

fn cpu_cumsum(x: &[f32], outer: usize, d: usize, inner: usize, exclusive: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; outer * d * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f64;
            for k in 0..d {
                let idx = (o * d + k) * inner + i;
                if exclusive {
                    out[idx] = acc as f32;
                    acc += x[idx] as f64;
                } else {
                    acc += x[idx] as f64;
                    out[idx] = acc as f32;
                }
            }
        }
    }
    out
}

#[test]
fn cumsum_arbitrary_axis_inclusive_and_exclusive_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5900_0002);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // Covers: tiny d=1, single-block Blelloch (d=17, d=512), multi-block Blelloch (d=513, d=1600),
    // strided short/wide columns (inner=7, inner=64), and long narrow strided hierarchical scan (outer=2, d=700, inner=5).
    let cases = [
        (3usize, 1usize, 1usize, "row d=1"),
        (4, 17, 1, "row odd single-block"),
        (2, 512, 1, "row exact 512"),
        (3, 513, 1, "row 513 multi-block"),
        (2, 1600, 1, "row 1600 multi-block"),
        (3, 19, 7, "cols odd"),
        (2, 48, 64, "cols wide"),
        (2, 700, 5, "cols long hierarchical"),
    ];

    for &dtype in dtypes {
        for &(outer, d, inner, label) in &cases {
            let total = outer * d * inner;
            let mut x = rng.vec(total, -0.5, 0.5);
            if dtype == DType::F16 {
                x = f16_round(&x);
            }
            let gx = upload_dt(&ctx, &x, dtype);
            let gy = GpuTensor::empty(&ctx, &[total], dtype).unwrap();
            let (atol, rtol) =
                if dtype == DType::F16 { (2e-2, 1e-2) } else { (2e-5 * (d as f32).sqrt(), 2e-5) };

            for exclusive in [false, true] {
                if !exclusive {
                    cumsum(&ctx, gx.raw(), gy.raw(), (outer, d, inner), dtype).unwrap();
                } else {
                    cumsum_ex(&ctx, gx.raw(), gy.raw(), (outer, d, inner), true, dtype).unwrap();
                }
                let want = cpu_cumsum(&x, outer, d, inner, exclusive);
                assert_close(
                    &gy.to_vec().unwrap(),
                    &want,
                    atol,
                    rtol,
                    &format!("cumsum {label} exc={exclusive} {dtype:?}"),
                );
            }
        }
    }

    // Empty no-op and aliasing rejection.
    cumsum(&ctx, ctx.dummy(), ctx.dummy(), (0, 8, 1), DType::F32).unwrap();
    let buf = GpuTensor::zeros(&ctx, &[16], DType::F32).unwrap();
    assert!(cumsum(&ctx, buf.raw(), buf.raw(), (2, 8, 1), DType::F32).is_err());
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

fn cpu_batched_topk(
    x: &[f32],
    outer: usize,
    d: usize,
    inner: usize,
    k: usize,
    largest: bool,
) -> (Vec<f32>, Vec<i64>) {
    let mut vals = vec![0.0f32; outer * k * inner];
    let mut idxs = vec![0i64; outer * k * inner];
    let key = |v: f32| if v == 0.0 { 0.0 } else { v };
    for o in 0..outer {
        for i in 0..inner {
            let mut items: Vec<(f32, usize)> = (0..d).map(|r| (x[(o * d + r) * inner + i], r)).collect();
            items.sort_by(|a, b| {
                let ord = if largest { key(b.0).total_cmp(&key(a.0)) } else { key(a.0).total_cmp(&key(b.0)) };
                ord.then(a.1.cmp(&b.1))
            });
            for (j, &(val, idx)) in items.iter().enumerate().take(k) {
                let dst = (o * k + j) * inner + i;
                vals[dst] = val;
                idxs[dst] = idx as i64;
            }
        }
    }
    (vals, idxs)
}

#[test]
fn batched_topk_all_paths_and_deterministic_ties_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5900_0003);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // Covers:
    // - single-pass bitonic (d=1, d=19, d=512)
    // - 2-stage parallel block merge (d=1100, k=16 -> tiles=3, n_cand=48 <= 512)
    // - multi-pass subgroup & portable selection fallback (d=600, k=70 > 64)
    // - strided axis SMALL_K (inner=7, k=5) and large k (inner=5, k=40 > 32)
    let cases = [
        (3usize, 1usize, 1usize, 1usize, "d=1"),
        (5, 19, 1, 7, "bitonic odd"),
        (2, 512, 1, 16, "bitonic 512"),
        (3, 1100, 1, 16, "2-stage block merge"),
        (2, 600, 1, 70, "select fallback"),
        (3, 41, 7, 5, "strided small_k"),
        (2, 55, 5, 40, "strided large_k"),
    ];

    for &dtype in dtypes {
        for &(outer, d, inner, k, label) in &cases {
            let total = outer * d * inner;
            // Quantized discrete values create frequent exact ties to stress deterministic ascending-index breaking.
            let mut x: Vec<f32> = (0..total).map(|_| ((rng.next_u64() % 17) as f32 - 8.0) * 0.25).collect();
            if d >= 6 && dtype == DType::F32 {
                x[0] = -0.0;
                x[1] = 0.0;
                x[2] = f32::INFINITY;
                x[3] = f32::NEG_INFINITY;
                x[4] = f32::NAN;
                x[5] = -f32::NAN;
            }
            if dtype == DType::F16 {
                x = f16_round(&x);
            }
            let gx = upload_dt(&ctx, &x, dtype);
            let out_n = outer * k * inner;
            let gv = GpuTensor::empty(&ctx, &[out_n], dtype).unwrap();
            let gi = ctx.alloc((out_n * 8) as u64).unwrap();

            let modes: &[bool] = if inner == 1 && d > 512 && k > 64 { &[false, true] } else { &[false] };
            for largest in [true, false] {
                let (want_v, want_i) = cpu_batched_topk(&x, outer, d, inner, k, largest);
                for &portable in modes {
                    let _fg = ForceGuard::set(portable);
                    let desc = BatchedTopkDesc { outer, dim_len: d, inner, k, largest, dtype };
                    batched_topk(&ctx, &desc, gx.raw(), gv.raw(), gi.raw()).unwrap();
                    let got_v = gv.to_vec().unwrap();
                    let got_i = read_i64(&ctx, gi.raw(), out_n);
                    assert_eq!(
                        got_i, want_i,
                        "batched_topk indices {label} largest={largest} {dtype:?} p={portable}"
                    );
                    assert_close(
                        &got_v,
                        &want_v,
                        0.0,
                        0.0,
                        &format!("batched_topk values {label} largest={largest} {dtype:?} p={portable}"),
                    );
                }
            }
        }
    }

    // F16 distinct NaN payload ordering: raw bit order `[0x7e01, 0x7e02]` must select index 1 for largest=true.
    if ctx.caps().shader_f16 {
        let nan_pair = [half::f16::from_bits(0x7e01), half::f16::from_bits(0x7e02)];
        let gx_nan = GpuTensor::from_f16(&ctx, &nan_pair, &[2]).unwrap();
        let gv_nan = GpuTensor::empty(&ctx, &[1], DType::F16).unwrap();
        let gi_nan = ctx.alloc(8).unwrap();
        let desc = BatchedTopkDesc { outer: 1, dim_len: 2, inner: 1, k: 1, largest: true, dtype: DType::F16 };
        batched_topk(&ctx, &desc, gx_nan.raw(), gv_nan.raw(), gi_nan.raw()).unwrap();
        assert_eq!(read_i64(&ctx, gi_nan.raw(), 1), vec![1]);
    }

    // Validation & empty k == 0 no-op.
    let empty_d = BatchedTopkDesc { outer: 2, dim_len: 8, inner: 1, k: 0, largest: true, dtype: DType::F32 };
    let dummy_x = GpuTensor::zeros(&ctx, &[16], DType::F32).unwrap();
    batched_topk(&ctx, &empty_d, dummy_x.raw(), ctx.dummy(), ctx.dummy()).unwrap();
    let bad_k = BatchedTopkDesc { k: 9, ..empty_d };
    assert!(batched_topk(&ctx, &bad_k, dummy_x.raw(), ctx.dummy(), ctx.dummy()).is_err());
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

fn cpu_im2col(x: &[f32], d: &Im2ColDesc) -> Vec<f32> {
    let (oh, ow) = d.output_hw().unwrap();
    let (kh, kw) = d.win.kernel;
    let (sh, sw) = d.win.stride;
    let (ph, pw) = d.win.padding;
    let (dh, dw) = d.win.dilation;
    let hw_out = oh * ow;
    let col_rows = d.c * kh * kw;
    let mut out = vec![0.0f32; d.n * col_rows * hw_out];
    for n in 0..d.n {
        for c in 0..d.c {
            for ky in 0..kh {
                for kx in 0..kw {
                    let row = (c * kh + ky) * kw + kx;
                    for oy in 0..oh {
                        let iy = (oy * sh + ky * dh) as isize - ph as isize;
                        for ox in 0..ow {
                            let ix = (ox * sw + kx * dw) as isize - pw as isize;
                            let v = if iy >= 0 && iy < d.h as isize && ix >= 0 && ix < d.w as isize {
                                x[((n * d.c + c) * d.h + iy as usize) * d.w + ix as usize]
                            } else {
                                0.0
                            };
                            out[(n * col_rows + row) * hw_out + oy * ow + ox] = v;
                        }
                    }
                }
            }
        }
    }
    out
}

fn cpu_col2im(col: &[f32], d: &Im2ColDesc) -> Vec<f32> {
    let (oh, ow) = d.output_hw().unwrap();
    let (kh, kw) = d.win.kernel;
    let (sh, sw) = d.win.stride;
    let (ph, pw) = d.win.padding;
    let (dh, dw) = d.win.dilation;
    let hw_out = oh * ow;
    let col_rows = d.c * kh * kw;
    let mut x = vec![0.0f64; d.n * d.c * d.h * d.w];
    for n in 0..d.n {
        for c in 0..d.c {
            for ky in 0..kh {
                for kx in 0..kw {
                    let row = (c * kh + ky) * kw + kx;
                    for oy in 0..oh {
                        let iy = (oy * sh + ky * dh) as isize - ph as isize;
                        if iy < 0 || iy >= d.h as isize {
                            continue;
                        }
                        for ox in 0..ow {
                            let ix = (ox * sw + kx * dw) as isize - pw as isize;
                            if ix >= 0 && ix < d.w as isize {
                                let v = col[(n * col_rows + row) * hw_out + oy * ow + ox] as f64;
                                x[((n * d.c + c) * d.h + iy as usize) * d.w + ix as usize] += v;
                            }
                        }
                    }
                }
            }
        }
    }
    x.into_iter().map(|v| v as f32).collect()
}

#[test]
fn tiled_im2col_and_atomic_free_col2im_match_cpu_and_adjoint() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5900_0004);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // Covers stride 1 & 2, symmetric & asymmetric padding/kernel, dilation 2, multi-tile (>16x16),
    // and generic fallback (large dilated kernel exceeding 2048-float shared tile).
    let cases = [
        (2usize, 3usize, 7usize, 9usize, Window2d::square(3, 1, 1), "3x3 s1 p1"),
        (2, 4, 19, 21, Window2d::square(3, 2, 1), "3x3 s2 p1 multi-tile"),
        (
            1,
            3,
            18,
            17,
            Window2d { kernel: (3, 5), stride: (2, 1), padding: (1, 2), dilation: (2, 2) },
            "asym dilated",
        ),
        (1, 2, 5, 5, Window2d::square(1, 1, 0), "1x1 s1 p0"),
        (
            1,
            2,
            25,
            25,
            Window2d { kernel: (9, 9), stride: (2, 2), padding: (4, 4), dilation: (3, 3) },
            "generic fallback",
        ),
    ];

    for &dtype in dtypes {
        for &(n, c, h, w, win, label) in &cases {
            let desc = Im2ColDesc { n, c, h, w, win, dtype };
            let (oh, ow) = desc.output_hw().unwrap();
            let x_len = n * c * h * w;
            let col_len = n * desc.col_rows().unwrap() * oh * ow;

            let mut x = rng.vec(x_len, -2.0, 2.0);
            let mut g_col = rng.vec(col_len, -1.5, 1.5);
            if dtype == DType::F16 {
                x = f16_round(&x);
                g_col = f16_round(&g_col);
            }

            let gx = upload_dt(&ctx, &x, dtype);
            let gcol_out = GpuTensor::empty(&ctx, &[col_len], dtype).unwrap();
            im2col(&ctx, &desc, gx.raw(), gcol_out.raw()).unwrap();
            let got_col = gcol_out.to_vec().unwrap();
            let want_col = cpu_im2col(&x, &desc);
            assert_close(&got_col, &want_col, 0.0, 0.0, &format!("im2col {label} {dtype:?}"));

            let ggcol = upload_dt(&ctx, &g_col, dtype);
            let gx_out = GpuTensor::empty(&ctx, &[x_len], dtype).unwrap();
            col2im(&ctx, &desc, ggcol.raw(), gx_out.raw()).unwrap();
            let got_x = gx_out.to_vec().unwrap();
            let want_x = cpu_col2im(&g_col, &desc);
            let (atol, rtol) = if dtype == DType::F16 { (2e-2, 1e-2) } else { (1e-5, 1e-5) };
            assert_close(&got_x, &want_x, atol, rtol, &format!("col2im {label} {dtype:?}"));

            // Adjoint identity: `<im2col(x), g_col> == <x, col2im(g_col)>`.
            if dtype == DType::F32 {
                let lhs: f64 = got_col.iter().zip(&g_col).map(|(&a, &b)| a as f64 * b as f64).sum();
                let rhs: f64 = x.iter().zip(&got_x).map(|(&a, &b)| a as f64 * b as f64).sum();
                assert!(
                    (lhs - rhs).abs() <= 1e-3 * (1.0 + lhs.abs()),
                    "im2col/col2im adjoint mismatch {label}: lhs={lhs}, rhs={rhs}"
                );
            }
        }
    }

    // Empty batch no-op and aliasing rejection.
    let empty = Im2ColDesc { n: 0, c: 2, h: 4, w: 4, win: Window2d::square(3, 1, 1), dtype: DType::F32 };
    im2col(&ctx, &empty, ctx.dummy(), ctx.dummy()).unwrap();
    col2im(&ctx, &empty, ctx.dummy(), ctx.dummy()).unwrap();
    let d1 = Im2ColDesc { n: 1, ..empty };
    let buf = GpuTensor::zeros(&ctx, &[1024], DType::F32).unwrap();
    assert!(im2col(&ctx, &d1, buf.raw(), buf.raw()).is_err());
    assert!(col2im(&ctx, &d1, buf.raw(), buf.raw()).is_err());
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

fn cpu_gelu(x: f64) -> f64 {
    let k = (2.0f64 / std::f64::consts::PI).sqrt();
    let c = 0.044715f64;
    0.5 * x * (1.0 + (k * (x + c * x * x * x)).tanh())
}

fn cpu_gelu_deriv(x: f64) -> f64 {
    if x.abs() >= 11.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let k = (2.0f64 / std::f64::consts::PI).sqrt();
    let c = 0.044715f64;
    let u = k * (x + c * x * x * x);
    let e = (-2.0 * u.abs()).exp();
    let inv = 1.0 / (1.0 + e);
    let one_plus_t = if u >= 0.0 { 2.0 * inv } else { 2.0 * e * inv };
    let sech2 = 4.0 * e * inv * inv;
    let du = k * (1.0 + 3.0 * c * x * x);
    0.5 * (one_plus_t + x * sech2 * du)
}

#[test]
fn fused_geglu_and_swiglu_match_cpu_and_finite_diff() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5900_0005);
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    for &dtype in dtypes {
        for &n in &[1usize, 7, 64, 1027] {
            let mut gate = rng.vec(n, -4.0, 4.0);
            if n >= 7 && dtype == DType::F32 {
                gate[0] = -1e4;
                gate[1] = -12.0;
                gate[2] = 0.0;
                gate[3] = 12.0;
                gate[4] = 1e4;
            }
            let mut up = rng.vec(n, -2.5, 2.5);
            let mut grad = rng.vec(n, -1.5, 1.5);
            if dtype == DType::F16 {
                gate = f16_round(&gate);
                up = f16_round(&up);
                grad = f16_round(&grad);
            }

            let gg = upload_dt(&ctx, &gate, dtype);
            let gu = upload_dt(&ctx, &up, dtype);
            let ggrad = upload_dt(&ctx, &grad, dtype);
            let gy = GpuTensor::empty(&ctx, &[n], dtype).unwrap();
            let gdg = GpuTensor::empty(&ctx, &[n], dtype).unwrap();
            let gdu = GpuTensor::empty(&ctx, &[n], dtype).unwrap();

            geglu_forward(&ctx, gg.raw(), gu.raw(), gy.raw(), n, dtype).unwrap();
            geglu_backward(&ctx, gg.raw(), gu.raw(), ggrad.raw(), gdg.raw(), gdu.raw(), n, dtype).unwrap();

            let want_y: Vec<f32> =
                gate.iter().zip(&up).map(|(&a, &u)| (cpu_gelu(a as f64) * u as f64) as f32).collect();
            let want_dg: Vec<f32> = gate
                .iter()
                .zip(&up)
                .zip(&grad)
                .map(|((&a, &u), &gr)| (gr as f64 * u as f64 * cpu_gelu_deriv(a as f64)) as f32)
                .collect();
            let want_du: Vec<f32> =
                gate.iter().zip(&grad).map(|(&a, &gr)| (gr as f64 * cpu_gelu(a as f64)) as f32).collect();

            let (atol, rtol) = if dtype == DType::F16 { (1e-2, 8e-3) } else { (3e-5, 3e-5) };
            assert_close(&gy.to_vec().unwrap(), &want_y, atol, rtol, &format!("geglu_fwd n={n} {dtype:?}"));
            assert_close(
                &gdg.to_vec().unwrap(),
                &want_dg,
                atol,
                rtol,
                &format!("geglu_bwd dgate n={n} {dtype:?}"),
            );
            assert_close(
                &gdu.to_vec().unwrap(),
                &want_du,
                atol,
                rtol,
                &format!("geglu_bwd dup n={n} {dtype:?}"),
            );
        }
    }

    // Overflow guard + tail precision for GeGLU backward:
    // 1. `grad = up = 1e20, gate = -6` and `gate = -10` stay finite and accurate.
    // 2. `gate = -5.5, up = 1024, grad = 1024` preserves the representable tail gradient (`~ -0.0490`) in F32 & F16.
    let gate_o = [-6.0f32, -10.0, -6.0, -5.5];
    let up_o = [1e20f32, 1e20, 1e30, 1024.0];
    let g_o = [1e20f32, 1e20, 1e10, 1024.0];
    let t_gate = GpuTensor::from_slice(&ctx, &gate_o, &[4]).unwrap();
    let t_up = GpuTensor::from_slice(&ctx, &up_o, &[4]).unwrap();
    let t_g = GpuTensor::from_slice(&ctx, &g_o, &[4]).unwrap();
    let t_dg = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
    let t_du = GpuTensor::empty(&ctx, &[4], DType::F32).unwrap();
    geglu_backward(&ctx, t_gate.raw(), t_up.raw(), t_g.raw(), t_dg.raw(), t_du.raw(), 4, DType::F32).unwrap();
    let got_dg = t_dg.to_vec().unwrap();
    assert!(got_dg.iter().all(|v| v.is_finite()), "geglu_bwd overflowed: {got_dg:?}");
    let want_g10 = (1e20f64 * 1e20 * cpu_gelu_deriv(-10.0)) as f32;
    assert_close(&got_dg[1..2], &[want_g10], 1e-1, 1e-3, "geglu_bwd -10.0 tail f32");
    let want_tail = (1024.0f64 * 1024.0 * cpu_gelu_deriv(-5.5)) as f32;
    assert_close(&got_dg[3..4], &[want_tail], 1e-4, 1e-3, "geglu_bwd -5.5 tail f32");
    if ctx.caps().shader_f16 {
        let f16_gate = GpuTensor::from_f16(&ctx, &to_f16(&[-5.5; 4]), &[4]).unwrap();
        let f16_up = GpuTensor::from_f16(&ctx, &to_f16(&[1024.0; 4]), &[4]).unwrap();
        let f16_g = GpuTensor::from_f16(&ctx, &to_f16(&[1024.0; 4]), &[4]).unwrap();
        let f16_dg = GpuTensor::empty(&ctx, &[4], DType::F16).unwrap();
        let f16_du = GpuTensor::empty(&ctx, &[4], DType::F16).unwrap();
        geglu_backward(
            &ctx,
            f16_gate.raw(),
            f16_up.raw(),
            f16_g.raw(),
            f16_dg.raw(),
            f16_du.raw(),
            4,
            DType::F16,
        )
        .unwrap();
        assert_close(&f16_dg.to_vec().unwrap(), &[want_tail; 4], 5e-4, 5e-3, "geglu_bwd -5.5 tail f16");
    }

    // Finite-difference gradient check for BOTH SwiGLU and GeGLU (both dgate and dup).
    let n = 12;
    let gate = rng.vec(n, -3.0, 3.0);
    let up = rng.vec(n, -2.0, 2.0);
    let grad = rng.vec(n, -1.5, 1.5);
    let eps = 1e-3f32;
    let gg = GpuTensor::from_slice(&ctx, &gate, &[n]).unwrap();
    let gu = GpuTensor::from_slice(&ctx, &up, &[n]).unwrap();
    let ggrad = GpuTensor::from_slice(&ctx, &grad, &[n]).unwrap();

    for is_geglu in [false, true] {
        let gdg = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        let gdu = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
        if is_geglu {
            geglu_backward(&ctx, gg.raw(), gu.raw(), ggrad.raw(), gdg.raw(), gdu.raw(), n, DType::F32)
                .unwrap();
        } else {
            swiglu_backward(&ctx, gg.raw(), gu.raw(), ggrad.raw(), gdg.raw(), gdu.raw(), n, DType::F32)
                .unwrap();
        }
        let got_dg = gdg.to_vec().unwrap();
        let got_du = gdu.to_vec().unwrap();
        let eval_loss = |g_vec: &[f32], u_vec: &[f32]| -> f64 {
            let tg = GpuTensor::from_slice(&ctx, g_vec, &[n]).unwrap();
            let tu = GpuTensor::from_slice(&ctx, u_vec, &[n]).unwrap();
            let ty = GpuTensor::empty(&ctx, &[n], DType::F32).unwrap();
            if is_geglu {
                geglu_forward(&ctx, tg.raw(), tu.raw(), ty.raw(), n, DType::F32).unwrap();
            } else {
                swiglu_forward(&ctx, tg.raw(), tu.raw(), ty.raw(), n, DType::F32).unwrap();
            }
            ty.to_vec().unwrap().iter().zip(&grad).map(|(&y, &gr)| y as f64 * gr as f64).sum()
        };
        for i in 0..n {
            let (mut gp, mut gm) = (gate.clone(), gate.clone());
            gp[i] += eps;
            gm[i] -= eps;
            let fd_g = ((eval_loss(&gp, &up) - eval_loss(&gm, &up)) / (2.0 * eps as f64)) as f32;
            assert!(
                (got_dg[i] - fd_g).abs() < 2e-3,
                "glu(geglu={is_geglu}) fd dgate[{i}]: got {}, fd {fd_g}",
                got_dg[i]
            );
            let (mut up_p, mut up_m) = (up.clone(), up.clone());
            up_p[i] += eps;
            up_m[i] -= eps;
            let fd_u = ((eval_loss(&gate, &up_p) - eval_loss(&gate, &up_m)) / (2.0 * eps as f64)) as f32;
            assert!(
                (got_du[i] - fd_u).abs() < 2e-3,
                "glu(geglu={is_geglu}) fd dup[{i}]: got {}, fd {fd_u}",
                got_du[i]
            );
        }
    }

    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
