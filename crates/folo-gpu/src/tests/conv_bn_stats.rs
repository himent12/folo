//! The BatchNorm statistics of the tensor-core forward conv's epilogue (`conv::conv2d_bn_stats`): the Welford
//! partials, merged on the CPU, against the f64 statistics of the conv output the same call wrote.

use super::Rng;
use super::coop::{coop_ctx, upload};
use crate::kernels::conv::{self, BnPartials, Conv2dDesc, ConvInput, ConvPlan};
use crate::kernels::pooling::Window2d;
use crate::{DType, GpuContext, GpuTensor, UnaryOp};

/// Per channel `(count, mean, biased variance)`, merging the partials in chunk order.
fn merged(ctx: &GpuContext, part: &BnPartials, c: usize) -> Vec<(f64, f64, f64)> {
    let bytes = ctx.read(part.buf.raw(), 0, (c * part.chunks * 3 * 4) as u64).unwrap();
    let v: Vec<f64> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b) as f64).collect();
    (0..c)
        .map(|ch| {
            let (mut n, mut m, mut q) = (0.0, 0.0, 0.0);
            for k in 0..part.chunks {
                let o = (ch * part.chunks + k) * 3;
                let (nb, mb, qb) = (v[o], v[o + 1], v[o + 2]);
                if nb == 0.0 {
                    continue;
                }
                let tot = n + nb;
                let d = mb - m;
                m += d * nb / tot;
                q += qb + d * d * n * nb / tot;
                n = tot;
            }
            (n, m, q / n.max(1.0))
        })
        .collect()
}

/// Per channel `(count, mean, biased variance)` of `y` (`[n, c, hw]`) in f64.
fn reference(y: &[f32], n: usize, c: usize, hw: usize) -> Vec<(f64, f64, f64)> {
    (0..c)
        .map(|ch| {
            let vals = || (0..n).flat_map(move |i| y[(i * c + ch) * hw..][..hw].iter().map(|&v| v as f64));
            let cnt = (n * hw) as f64;
            let mean = vals().sum::<f64>() / cnt;
            (cnt, mean, vals().map(|v| (v - mean).powi(2)).sum::<f64>() / cnt)
        })
        .collect()
}

/// Runs `d` with statistics under `plan` and checks them: the counts exactly, the mean and variance to `tol`
/// relative (f16 outputs are rounded after the statistics were taken from the f32 sums).
fn check(ctx: &GpuContext, d: &Conv2dDesc, plan: ConvPlan, seed: u64, what: &str) {
    let (oh, ow) = d.output().unwrap();
    let mut rng = Rng::new(seed);
    // Inputs centred away from zero, so the channel means are not near zero.
    let x = upload(ctx, &rng.vec(d.n * d.c_in * d.h * d.w, -0.5, 1.5), d.dtype);
    let (kh, kw) = d.win.kernel;
    let w = upload(ctx, &rng.vec(d.c_out * d.c_in * kh * kw, -0.3, 0.4), d.dtype);
    let y = GpuTensor::empty(ctx, &[d.n * d.c_out * oh * ow], d.dtype).unwrap();
    let xi = (x.raw(), ConvInput { offset: 0, n_stride: d.c_in * d.h * d.w });
    let part = conv::conv2d_bn_stats(ctx, d, xi, w.raw(), y.raw(), plan).unwrap();
    let part = part.unwrap_or_else(|| panic!("{what}: no statistics"));
    let got = merged(ctx, &part, d.c_out);
    let want = reference(&y.to_vec().unwrap(), d.n, d.c_out, oh * ow);
    let tol = if d.dtype == DType::F16 { 2e-3 } else { 1e-5 };
    for (ch, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.0, w.0, "{what}: channel {ch} count");
        let sd = w.2.sqrt();
        assert!((g.1 - w.1).abs() <= tol * (sd + w.1.abs()), "{what}: channel {ch} mean {} vs {}", g.1, w.1);
        assert!((g.2 - w.2).abs() <= 2.0 * tol * w.2, "{what}: channel {ch} variance {} vs {}", g.2, w.2);
    }
}

/// `(n, c_in, h, w, c_out, k, s)`, padding `k / 2`.
fn desc(
    (n, c_in, h, w, c_out, k, s): (usize, usize, usize, usize, usize, usize, usize),
    dtype: DType,
) -> Conv2dDesc {
    Conv2dDesc {
        n,
        c_in,
        h,
        w,
        c_out,
        groups: 1,
        win: Window2d::square(k, s, k / 2),
        act: UnaryOp::Identity,
        dtype,
    }
}

/// Shapes with pixels and channels that do not fill the tiles (column masks, rows past `M`), several column tiles
/// per image, pointwise (the GEMM) and windowed (the implicit GEMM), strided.
const CASES: [(usize, usize, usize, usize, usize, usize, usize); 6] = [
    (2, 8, 9, 7, 12, 1, 1),
    (2, 32, 40, 40, 64, 1, 1),
    (3, 5, 11, 13, 70, 3, 1),
    (2, 16, 20, 20, 32, 3, 2),
    (1, 3, 33, 31, 16, 3, 2),
    (2, 64, 12, 12, 160, 3, 1),
];

#[test]
fn tensor_core_conv_statistics_match_cpu() {
    let Some((ctx, _g)) = coop_ctx() else { return };
    for (i, &case) in CASES.iter().enumerate() {
        let d = desc(case, DType::F16);
        check(
            &ctx,
            &d,
            ConvPlan { tiles: None, splits: Some(1) },
            30 + i as u64,
            &format!("tensor cores case {i}"),
        );
    }
}

/// No statistics (the BatchNorm gathers its own) from the regular kernels, nor from a tensor-core conv the plan
/// splits along `K`.
#[test]
fn regular_and_split_convs_have_no_statistics() {
    const SHAPE: (usize, usize, usize, usize, usize, usize, usize) = (1, 256, 8, 8, 64, 3, 1);
    let Some((ctx, _g)) = coop_ctx() else { return };
    let run = |dtype: DType, plan: ConvPlan| {
        let d = desc(SHAPE, dtype);
        let x = GpuTensor::empty(&ctx, &[256 * 64], dtype).unwrap();
        let w = GpuTensor::empty(&ctx, &[64 * 256 * 9], dtype).unwrap();
        let y = GpuTensor::empty(&ctx, &[64 * 64], dtype).unwrap();
        let xi = (x.raw(), ConvInput { offset: 0, n_stride: 256 * 64 });
        conv::conv2d_bn_stats(&ctx, &d, xi, w.raw(), y.raw(), plan).unwrap().is_some()
    };
    assert!(run(DType::F16, ConvPlan { tiles: None, splits: Some(1) }), "tensor cores, unsplit");
    assert!(!run(DType::F16, ConvPlan { tiles: None, splits: Some(4) }), "tensor cores, split");
    assert!(!run(DType::F32, ConvPlan::default()), "f32");
    let (oh, ow) = desc(SHAPE, DType::F16).output().unwrap();
    let g = desc(SHAPE, DType::F16).gemm_view(oh, ow, false);
    let tiles = crate::kernels::gemm::pick_config_conv(&ctx, &g);
    assert!(!run(DType::F16, ConvPlan { tiles: Some(tiles), splits: Some(1) }), "f16, regular kernels");
}
