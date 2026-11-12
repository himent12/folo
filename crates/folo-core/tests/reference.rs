//! Forward results against naive reference implementations, at sizes that exercise the parallel paths.

use folo_core::rng::Rng;
use folo_core::{Conv2dParams, Pool2dParams, Tensor};

fn randn(shape: &[usize], seed: u64) -> Tensor {
    Tensor::randn(shape, 0.0, 1.0, &mut Rng::seed(seed))
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}: element {i}: {g} vs {w}");
    }
}

fn naive_conv(x: &Tensor, w: &Tensor, b: Option<&Tensor>, p: &Conv2dParams) -> Vec<f32> {
    let (n, c, h, wd) = x.dims4().unwrap();
    let (o, cg, kh, kw) = w.dims4().unwrap();
    let [oh, ow] = p.output_hw(h, wd, kh, kw).unwrap();
    let (xv, wv) = (x.to_vec::<f32>().unwrap(), w.to_vec::<f32>().unwrap());
    let bv = b.map(|b| b.to_vec::<f32>().unwrap());
    let og = o / p.groups;
    let mut out = vec![0f32; n * o * oh * ow];
    for img in 0..n {
        for oc in 0..o {
            let grp = oc / og;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s = bv.as_ref().map_or(0.0, |b| b[oc]) as f64;
                    for ci in 0..cg {
                        let ic = grp * cg + ci;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy =
                                    (oy * p.stride[0] + ky * p.dilation[0]) as isize - p.padding[0] as isize;
                                let ix =
                                    (ox * p.stride[1] + kx * p.dilation[1]) as isize - p.padding[1] as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                    continue;
                                }
                                let xi = ((img * c + ic) * h + iy as usize) * wd + ix as usize;
                                let wi = ((oc * cg + ci) * kh + ky) * kw + kx;
                                s += xv[xi] as f64 * wv[wi] as f64;
                            }
                        }
                    }
                    out[((img * o + oc) * oh + oy) * ow + ox] = s as f32;
                }
            }
        }
    }
    out
}

/// `(n, c_in, c_out, [h, w], [kh, kw], params, bias)`.
type ConvCase = (usize, usize, usize, [usize; 2], [usize; 2], Conv2dParams, bool);

#[test]
fn conv2d_matches_naive() {
    let p = |s: [usize; 2], pad: [usize; 2], d: [usize; 2], g: usize| Conv2dParams {
        stride: s,
        padding: pad,
        dilation: d,
        groups: g,
    };
    let cases: Vec<ConvCase> = vec![
        (2, 3, 4, [9, 7], [3, 3], p([1, 1], [1, 1], [1, 1], 1), true),
        (3, 4, 6, [8, 8], [3, 3], p([2, 2], [1, 1], [1, 1], 2), false),
        (1, 3, 5, [10, 9], [3, 2], p([1, 2], [2, 0], [2, 1], 1), true),
        (2, 6, 6, [7, 7], [3, 3], p([1, 1], [1, 1], [1, 1], 6), true),
        (2, 4, 8, [9, 9], [5, 5], p([2, 2], [2, 2], [1, 1], 4), false),
        (2, 5, 7, [6, 6], [1, 1], p([1, 1], [0, 0], [1, 1], 1), true),
        (1, 4, 4, [6, 6], [1, 1], p([2, 2], [1, 1], [1, 1], 1), false),
        (1, 2, 3, [4, 4], [3, 3], p([1, 1], [3, 3], [1, 1], 1), true), // padding larger than kernel reach
        // Large enough for the parallel GEMM, implicit im2col and k-split paths.
        (2, 32, 48, [24, 24], [3, 3], p([1, 1], [1, 1], [1, 1], 1), true),
        (4, 16, 16, [20, 20], [3, 3], p([2, 2], [1, 1], [1, 1], 1), false),
        (1, 64, 32, [16, 16], [1, 1], p([1, 1], [0, 0], [1, 1], 1), false),
    ];
    for (i, (n, c, o, hw, k, prm, bias)) in cases.into_iter().enumerate() {
        let x = randn(&[n, c, hw[0], hw[1]], 100 + i as u64);
        let w = randn(&[o, c / prm.groups, k[0], k[1]], 200 + i as u64);
        let b = bias.then(|| randn(&[o], 300 + i as u64));
        let y = x.conv2d_with(&w, b.as_ref(), &prm).unwrap();
        assert_close(
            &y.to_vec::<f32>().unwrap(),
            &naive_conv(&x, &w, b.as_ref(), &prm),
            1e-4,
            &format!("case {i}"),
        );
    }
}

#[test]
fn conv2d_backward_matches_naive_transpose_at_scale() {
    // Gradient of sum(conv(x, w) * r) w.r.t. x and w, checked against naive formulas, on sizes that take the
    // parallel and k-split GEMM paths.
    let (n, c, o, h, k) = (3, 24, 40, 18, 3);
    let p = Conv2dParams { stride: [2, 1], padding: [1, 1], dilation: [1, 2], groups: 2 };
    let x = randn(&[n, c, h, h], 400).requires_grad_();
    let w = randn(&[o, c / 2, k, k], 401).requires_grad_();
    let y = x.conv2d_with(&w, None, &p).unwrap();
    let r = randn(y.dims(), 402);
    let g = y.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
    let (gx, gw) = (g.get(&x).unwrap().to_vec::<f32>().unwrap(), g.get(&w).unwrap().to_vec::<f32>().unwrap());
    let (xv, wv, rv) = (x.to_vec::<f32>().unwrap(), w.to_vec::<f32>().unwrap(), r.to_vec::<f32>().unwrap());
    let (_, _, oh, ow) = y.dims4().unwrap();
    let (cg, og) = (c / 2, o / 2);
    let mut ex = vec![0f64; xv.len()];
    let mut ew = vec![0f64; wv.len()];
    for img in 0..n {
        for oc in 0..o {
            let grp = oc / og;
            for oy in 0..oh {
                for ox in 0..ow {
                    let gr = rv[((img * o + oc) * oh + oy) * ow + ox] as f64;
                    for ci in 0..cg {
                        let ic = grp * cg + ci;
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = (oy * 2 + ky) as isize - 1;
                                let ix = (ox + kx * 2) as isize - 1;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= h as isize {
                                    continue;
                                }
                                let xi = ((img * c + ic) * h + iy as usize) * h + ix as usize;
                                let wi = ((oc * cg + ci) * k + ky) * k + kx;
                                ex[xi] += gr * wv[wi] as f64;
                                ew[wi] += gr * xv[xi] as f64;
                            }
                        }
                    }
                }
            }
        }
    }
    let ex: Vec<f32> = ex.iter().map(|&v| v as f32).collect();
    let ew: Vec<f32> = ew.iter().map(|&v| v as f32).collect();
    assert_close(&gx, &ex, 1e-4, "grad input");
    assert_close(&gw, &ew, 1e-4, "grad weight");
}

#[test]
fn pooling_matches_naive() {
    let x = randn(&[2, 3, 9, 8], 500);
    let xv = x.to_vec::<f32>().unwrap();
    for p in [
        Pool2dParams { kernel: [2, 2], stride: [2, 2], padding: [0, 0] },
        Pool2dParams { kernel: [3, 3], stride: [2, 2], padding: [1, 1] },
        Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] },
        Pool2dParams { kernel: [3, 2], stride: [1, 2], padding: [1, 0] },
    ] {
        let [oh, ow] = p.output_hw(9, 8).unwrap();
        let (mut mx, mut av) = (vec![], vec![]);
        for plane in 0..6 {
            for oy in 0..oh {
                for ox in 0..ow {
                    let (mut m, mut s) = (f32::NEG_INFINITY, 0f32);
                    for ky in 0..p.kernel[0] {
                        for kx in 0..p.kernel[1] {
                            let iy = (oy * p.stride[0] + ky) as isize - p.padding[0] as isize;
                            let ix = (ox * p.stride[1] + kx) as isize - p.padding[1] as isize;
                            if iy >= 0 && ix >= 0 && iy < 9 && ix < 8 {
                                let v = xv[plane * 72 + iy as usize * 8 + ix as usize];
                                m = m.max(v);
                                s += v;
                            }
                        }
                    }
                    mx.push(m);
                    av.push(s / (p.kernel[0] * p.kernel[1]) as f32);
                }
            }
        }
        assert_close(&x.max_pool2d_with(&p).unwrap().to_vec::<f32>().unwrap(), &mx, 0.0, "max_pool2d");
        assert_close(&x.avg_pool2d_with(&p).unwrap().to_vec::<f32>().unwrap(), &av, 1e-6, "avg_pool2d");
    }
    // Adaptive pooling with uneven windows, and global pooling.
    let a = x.adaptive_avg_pool2d([4, 3]).unwrap().to_vec::<f32>().unwrap();
    let mut want = vec![];
    for plane in 0..6 {
        for oy in 0..4usize {
            let (y0, y1) = (oy * 9 / 4, ((oy + 1) * 9).div_ceil(4));
            for ox in 0..3usize {
                let (x0, x1) = (ox * 8 / 3, ((ox + 1) * 8).div_ceil(3));
                let mut s = 0f32;
                for iy in y0..y1 {
                    for ix in x0..x1 {
                        s += xv[plane * 72 + iy * 8 + ix];
                    }
                }
                want.push(s / ((y1 - y0) * (x1 - x0)) as f32);
            }
        }
    }
    assert_close(&a, &want, 1e-5, "adaptive_avg_pool2d");
    let gap = x.adaptive_avg_pool2d([1, 1]).unwrap();
    assert_close(
        &gap.flatten_all().unwrap().to_vec::<f32>().unwrap(),
        &x.mean([2, 3], false).unwrap().flatten_all().unwrap().to_vec::<f32>().unwrap(),
        1e-5,
        "global pool",
    );
    // Upsample.
    let u = x.upsample_nearest2d_hw([2, 3]).unwrap();
    assert_eq!(u.dims(), &[2, 3, 18, 24]);
    let uv = u.to_vec::<f32>().unwrap();
    for (i, &v) in uv.iter().enumerate() {
        let (plane, rem) = (i / (18 * 24), i % (18 * 24));
        assert_eq!(v, xv[plane * 72 + (rem / 24 / 2) * 8 + (rem % 24) / 3]);
    }
}

#[test]
fn pooling_large_parallel() {
    // SPPF-like: max_pool2d(5, 1, 2) on a big tensor vs. the composition of smaller windows.
    let x = randn(&[4, 32, 40, 40], 510);
    let a = x.max_pool2d(5, 1, 2).unwrap();
    let b = x.max_pool2d(3, 1, 1).unwrap().max_pool2d(3, 1, 1).unwrap();
    assert_eq!(a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
}

#[test]
fn matmul_batched_broadcasting() {
    let naive = |a: &[f32], b: &[f32], m: usize, k: usize, n: usize| {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).map(|p| a[i * k + p] as f64 * b[p * n + j] as f64).sum::<f64>() as f32;
            }
        }
        c
    };
    let a = randn(&[2, 3, 5, 7], 600);
    let b = randn(&[3, 7, 4], 601);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3, 5, 4]);
    let (av, bv, cv) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap(), c.to_vec::<f32>().unwrap());
    for i in 0..2 {
        for j in 0..3 {
            let want = naive(&av[(i * 3 + j) * 35..][..35], &bv[j * 28..][..28], 5, 7, 4);
            assert_close(&cv[(i * 3 + j) * 20..][..20], &want, 1e-5, "batched matmul");
        }
    }
    // Shared weight (folded into one GEMM), non-contiguous lhs.
    let x = randn(&[4, 6, 33], 602).transpose(0, 1).unwrap(); // [6,4,33]
    let w = randn(&[33, 9], 603);
    let y = x.matmul(&w).unwrap();
    let xc = x.contiguous().unwrap().to_vec::<f32>().unwrap();
    let want = naive(&xc, &w.to_vec::<f32>().unwrap(), 24, 33, 9);
    assert_close(&y.to_vec::<f32>().unwrap(), &want, 1e-5, "folded matmul");
    // Large, parallel and k-split paths.
    let a = randn(&[70, 3000], 604);
    let b = randn(&[3000, 50], 605);
    let want = naive(&a.to_vec::<f32>().unwrap(), &b.to_vec::<f32>().unwrap(), 70, 3000, 50);
    assert_close(&a.matmul(&b).unwrap().to_vec::<f32>().unwrap(), &want, 1e-4, "k-split matmul");
    let a = randn(&[300, 200], 606);
    let b = randn(&[200, 500], 607);
    let want = naive(&a.to_vec::<f32>().unwrap(), &b.to_vec::<f32>().unwrap(), 300, 200, 500);
    assert_close(&a.matmul(&b).unwrap().to_vec::<f32>().unwrap(), &want, 1e-4, "parallel matmul");
    assert!(randn(&[2, 3], 0).matmul(&randn(&[4, 2], 1)).is_err());
    assert!(randn(&[3], 0).matmul(&randn(&[3, 2], 1)).is_err());
}

#[test]
fn constructors() {
    assert_eq!(Tensor::arange(0.0, 1.0, 0.25).unwrap().to_vec::<f32>().unwrap(), vec![0.0, 0.25, 0.5, 0.75]);
    assert_eq!(Tensor::arange(3.0, 0.0, -1.0).unwrap().to_vec::<f32>().unwrap(), vec![3.0, 2.0, 1.0]);
    assert!(Tensor::arange(0.0, 1.0, 0.0).is_err());
    assert_eq!(Tensor::linspace(0.0, 1.0, 5).to_vec::<f32>().unwrap(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_eq!(Tensor::eye(3).to_vec2::<f32>().unwrap()[1], vec![0.0, 1.0, 0.0]);
    assert_eq!(Tensor::scalar(2.5).item::<f32>().unwrap(), 2.5);
    assert_eq!(Tensor::full([2, 2], 3.0).to_vec::<f32>().unwrap(), vec![3.0; 4]);
    assert!(Tensor::from_vec(vec![1.0f32, 2.0], [3]).is_err());
    let a = Tensor::rand_uniform([1000], -2.0, 3.0, &mut Rng::seed(1)).to_vec::<f32>().unwrap();
    assert!(a.iter().all(|&v| (-2.0..3.0).contains(&v)));
    let b = Tensor::rand_uniform([1000], -2.0, 3.0, &mut Rng::seed(1)).to_vec::<f32>().unwrap();
    assert_eq!(a, b, "seeded constructors are reproducible");
    let z = Tensor::zeros_with([2], folo_core::DType::I64, folo_core::Device::Cpu).unwrap();
    assert_eq!(z.to_vec::<i64>().unwrap(), vec![0, 0]);
    assert!(z.to_vec::<f32>().is_err());
    assert_eq!(Tensor::ones([2]).ones_like().unwrap().to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
}

/// Naive gradients of `sum(conv2d(x, w) * r)` with respect to `x` and `w`.
fn naive_conv_grads(x: &Tensor, w: &Tensor, r: &Tensor, p: &Conv2dParams) -> (Vec<f32>, Vec<f32>) {
    let (n, c, h, wd) = x.dims4().unwrap();
    let (o, cg, kh, kw) = w.dims4().unwrap();
    let (_, _, oh, ow) = r.dims4().unwrap();
    let (xv, wv, rv) = (x.to_vec::<f32>().unwrap(), w.to_vec::<f32>().unwrap(), r.to_vec::<f32>().unwrap());
    let og = o / p.groups;
    let mut gx = vec![0f64; xv.len()];
    let mut gw = vec![0f64; wv.len()];
    for img in 0..n {
        for oc in 0..o {
            let grp = oc / og;
            for oy in 0..oh {
                for ox in 0..ow {
                    let gr = rv[((img * o + oc) * oh + oy) * ow + ox] as f64;
                    for ci in 0..cg {
                        let ic = grp * cg + ci;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy =
                                    (oy * p.stride[0] + ky * p.dilation[0]) as isize - p.padding[0] as isize;
                                let ix =
                                    (ox * p.stride[1] + kx * p.dilation[1]) as isize - p.padding[1] as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                    continue;
                                }
                                let xi = ((img * c + ic) * h + iy as usize) * wd + ix as usize;
                                let wi = ((oc * cg + ci) * kh + ky) * kw + kx;
                                gx[xi] += gr * wv[wi] as f64;
                                gw[wi] += gr * xv[xi] as f64;
                            }
                        }
                    }
                }
            }
        }
    }
    (gx.iter().map(|&v| v as f32).collect(), gw.iter().map(|&v| v as f32).collect())
}

#[test]
fn conv2d_backward_matches_naive_over_geometries() {
    let p = |s: [usize; 2], pad: [usize; 2], d: [usize; 2], g: usize| Conv2dParams {
        stride: s,
        padding: pad,
        dilation: d,
        groups: g,
    };
    // Strides 1-3 with kernels smaller than, equal to and larger than the stride, odd sizes, asymmetric
    // parameters, dilation, groups and padding beyond the kernel's reach (the col2im fallback).
    let cases: Vec<ConvCase> = vec![
        (2, 3, 4, [9, 7], [3, 3], p([1, 1], [1, 1], [1, 1], 1), false),
        (2, 4, 6, [9, 8], [3, 3], p([2, 2], [1, 1], [1, 1], 2), false),
        (1, 3, 5, [11, 10], [5, 5], p([3, 3], [2, 2], [1, 1], 1), false),
        (1, 3, 4, [10, 9], [3, 3], p([2, 2], [0, 0], [1, 1], 1), false),
        (1, 2, 3, [9, 9], [4, 4], p([2, 2], [1, 1], [1, 1], 1), false),
        (1, 2, 3, [10, 11], [2, 2], p([3, 3], [0, 0], [1, 1], 1), false),
        (1, 3, 5, [10, 9], [3, 2], p([1, 2], [2, 0], [2, 1], 1), false),
        (1, 2, 3, [9, 9], [3, 3], p([1, 1], [2, 2], [2, 2], 1), false),
        (1, 2, 3, [9, 9], [3, 3], p([2, 2], [2, 2], [2, 2], 1), false),
        (1, 4, 4, [6, 6], [1, 1], p([2, 2], [1, 1], [1, 1], 1), false),
        (1, 4, 4, [7, 7], [1, 1], p([2, 2], [0, 0], [1, 1], 1), false),
        (1, 2, 3, [4, 4], [3, 3], p([1, 1], [3, 3], [1, 1], 1), false),
        (2, 6, 6, [7, 7], [3, 3], p([2, 2], [1, 1], [1, 1], 6), false),
        // Large enough for the parallel GEMM paths and a batch smaller than the pool.
        (3, 32, 40, [20, 20], [3, 3], p([1, 1], [1, 1], [1, 1], 1), false),
        (3, 16, 24, [21, 21], [3, 3], p([2, 2], [1, 1], [1, 1], 1), false),
        (1, 48, 64, [16, 16], [1, 1], p([1, 1], [0, 0], [1, 1], 1), false),
        // The weight gradient over shifted rows: pointwise with many input channels, groups, K parts that
        // cross images, rows that are not a whole number of vectors.
        (3, 64, 24, [9, 11], [1, 1], p([1, 1], [0, 0], [1, 1], 1), false),
        (2, 8, 36, [9, 8], [3, 3], p([1, 1], [1, 1], [1, 1], 2), false),
        (5, 24, 20, [30, 27], [3, 3], p([1, 1], [1, 1], [1, 1], 1), false),
        // Strided input gradients: an input smaller than the stride, rectangular strides and kernels, padding
        // wider than the kernel's half, input pixels no tap reaches (kernel 1 at stride 3), groups, and a
        // folo-n-like 64 -> 64 3x3 stride 2 at 16x16.
        (2, 3, 4, [1, 2], [3, 3], p([2, 2], [1, 1], [1, 1], 1), false),
        (1, 4, 6, [11, 9], [3, 5], p([2, 3], [1, 2], [1, 1], 2), false),
        (1, 3, 5, [9, 10], [3, 3], p([2, 2], [2, 2], [1, 1], 1), false),
        (1, 2, 3, [8, 8], [1, 1], p([3, 3], [1, 1], [1, 1], 1), false),
        (2, 64, 64, [16, 16], [3, 3], p([2, 2], [1, 1], [1, 1], 1), false),
    ];
    for (i, (n, c, o, hw, k, prm, _)) in cases.into_iter().enumerate() {
        let x = randn(&[n, c, hw[0], hw[1]], 500 + i as u64).requires_grad_();
        let w = randn(&[o, c / prm.groups, k[0], k[1]], 600 + i as u64).requires_grad_();
        let y = x.conv2d_with(&w, None, &prm).unwrap();
        let r = randn(y.dims(), 700 + i as u64);
        let g = y.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let (ex, ew) = naive_conv_grads(&x, &w, &r, &prm);
        assert_close(
            &g.get(&x).unwrap().to_vec::<f32>().unwrap(),
            &ex,
            1e-4,
            &format!("case {i} grad input"),
        );
        assert_close(
            &g.get(&w).unwrap().to_vec::<f32>().unwrap(),
            &ew,
            1e-4,
            &format!("case {i} grad weight"),
        );
    }
}

#[test]
fn fused_conv_act_matches_composite() {
    use folo_core::{UnaryOp, no_grad};
    let _g = no_grad();
    let p = |s: usize, pad: usize, g: usize| Conv2dParams {
        stride: [s; 2],
        padding: [pad; 2],
        dilation: [1; 2],
        groups: g,
    };
    let cases = [
        (2, 8, 16, 12, 3, p(1, 1, 1)),
        (1, 16, 16, 20, 3, p(2, 1, 16)), // depthwise
        (2, 12, 24, 9, 1, p(1, 0, 1)),   // pointwise
        (1, 32, 48, 40, 3, p(1, 1, 2)),  // parallel GEMM, groups
        (3, 64, 64, 10, 3, p(1, 1, 1)),
    ];
    let acts = [None, Some(UnaryOp::Silu), Some(UnaryOp::Relu), Some(UnaryOp::Sigmoid), Some(UnaryOp::Tanh)];
    for (i, &(n, c, o, hw, k, prm)) in cases.iter().enumerate() {
        let x = randn(&[n, c, hw, hw], 800 + i as u64);
        let w = randn(&[o, c / prm.groups, k, k], 900 + i as u64).mul_scalar(0.2).unwrap();
        let b = randn(&[o], 1000 + i as u64);
        for act in acts {
            for bias in [None, Some(&b)] {
                let want = x.conv2d_with(&w, bias, &prm).unwrap();
                let want = match act {
                    Some(op) => want.unary(op).unwrap(),
                    None => want,
                };
                let got = x.conv2d_act(&w, bias, &prm, act).unwrap();
                assert_eq!(got.dims(), want.dims());
                assert_close(
                    &got.to_vec::<f32>().unwrap(),
                    &want.to_vec::<f32>().unwrap(),
                    1e-5,
                    &format!("case {i} {act:?} bias {}", bias.is_some()),
                );
            }
        }
    }
}

#[test]
fn conv2d_act_cat_matches_cat_then_conv() {
    use folo_core::{UnaryOp, no_grad};
    let p = |s: usize, pad: usize, g: usize| Conv2dParams {
        stride: [s; 2],
        padding: [pad; 2],
        dilation: [1; 2],
        groups: g,
    };
    // (batch, part channels, out channels, size, kernel, params)
    type CatCase = (usize, &'static [usize], usize, usize, usize, Conv2dParams);
    let cases: [CatCase; 5] = [
        (1, &[8, 8, 4], 16, 12, 1, p(1, 0, 1)),
        (2, &[16, 8], 24, 9, 3, p(1, 1, 1)),
        (3, &[4, 4, 4, 4], 8, 10, 3, p(2, 1, 2)),
        (1, &[32, 32], 32, 20, 1, p(1, 0, 1)),
        (2, &[6, 10], 16, 7, 3, p(1, 1, 16)), // depthwise over the concatenation
    ];
    for (i, &(n, cs, o, hw, k, prm)) in cases.iter().enumerate() {
        let _g = no_grad();
        let c: usize = cs.iter().sum();
        // The parts are channel slices of one wider tensor (strided views for batch > 1) and contiguous tensors.
        let wide = randn(&[n, c + 3, hw, hw], 1100 + i as u64);
        let mut parts = Vec::new();
        let mut at = 0;
        for (j, &ci) in cs.iter().enumerate() {
            parts.push(if j % 2 == 0 {
                wide.narrow(1, 3 + at, ci).unwrap()
            } else {
                randn(&[n, ci, hw, hw], 1200 + (i * 10 + j) as u64)
            });
            at += ci;
        }
        let w = randn(&[o, c / prm.groups, k, k], 1300 + i as u64).mul_scalar(0.2).unwrap();
        let b = randn(&[o], 1400 + i as u64);
        let refs: Vec<&Tensor> = parts.iter().collect();
        let want =
            Tensor::cat(&refs, 1).unwrap().conv2d_act(&w, Some(&b), &prm, Some(UnaryOp::Silu)).unwrap();
        let got = Tensor::conv2d_act_cat(&refs, &w, Some(&b), &prm, Some(UnaryOp::Silu)).unwrap();
        assert_eq!(got.dims(), want.dims());
        assert_eq!(got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap(), "case {i}");
    }
    // Recording: the composite path (cat, then conv), with gradients into every part.
    let a = randn(&[1, 3, 5, 5], 1500).requires_grad_();
    let bp = randn(&[1, 2, 5, 5], 1501).requires_grad_();
    let w = randn(&[4, 5, 3, 3], 1502);
    let y = Tensor::conv2d_act_cat(&[&a, &bp], &w, None, &p(1, 1, 1), None).unwrap();
    let grads = y.square().unwrap().mean_all().unwrap().backward().unwrap();
    assert_eq!(grads.get(&a).unwrap().dims(), a.dims());
    assert_eq!(grads.get(&bp).unwrap().dims(), bp.dims());
}

#[test]
fn conv2d_large_k_matches_naive() {
    // K = 512 · 9 = 4608 with 1024 output channels needs several packed-A super-blocks.
    let prm = Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };
    let x = randn(&[1, 512, 4, 5], 1600);
    let w = randn(&[1024, 512, 3, 3], 1601).mul_scalar(0.05).unwrap();
    let y = x.conv2d_with(&w, None, &prm).unwrap();
    assert_close(&y.to_vec::<f32>().unwrap(), &naive_conv(&x, &w, None, &prm), 1e-4, "large K");
}

#[test]
fn conv2d_huge_stride_allocates_only_used_phases() {
    // Stride 65536 on a 4x4 input: one output pixel; the shifted-rows copy must hold only the phase the 1x1
    // kernel reads, not 65536² of them.
    let prm = Conv2dParams { stride: [65536, 65536], padding: [0, 0], dilation: [1, 1], groups: 1 };
    let x = randn(&[1, 2, 4, 4], 1700);
    let w = randn(&[3, 2, 1, 1], 1701);
    let y = x.conv2d_with(&w, None, &prm).unwrap();
    assert_eq!(y.dims(), [1, 3, 1, 1]);
    assert_close(&y.to_vec::<f32>().unwrap(), &naive_conv(&x, &w, None, &prm), 1e-5, "huge stride");
    let prm = Conv2dParams { stride: [3, 5], padding: [2, 1], dilation: [2, 3], groups: 1 };
    let x = randn(&[2, 3, 11, 13], 1702);
    let w = randn(&[4, 3, 3, 2], 1703);
    let y = x.conv2d_with(&w, None, &prm).unwrap();
    assert_close(&y.to_vec::<f32>().unwrap(), &naive_conv(&x, &w, None, &prm), 1e-4, "stride 3x5 dilated");
}

#[test]
fn conv2d_weight_grad_ignores_recycled_buffer_contents() {
    // The stride-1 weight gradient reads the input's shifted rows as A, and the dropped columns of the last
    // output row run past each plane into the gap before the next (multiplied by the zero columns of the
    // packed gradient). Recycled NaN buffers of the shifted-rows copy's length must not reach the result.
    let p = |pad: [usize; 2], d: [usize; 2], g: usize| Conv2dParams {
        stride: [1, 1],
        padding: pad,
        dilation: d,
        groups: g,
    };
    // (n, c, o, [h, w], [kh, kw], params): kw 3 and 5, a dilation of 2, groups, ragged sizes, one image and several.
    type Case = (usize, usize, usize, [usize; 2], [usize; 2], Conv2dParams);
    let cases: Vec<Case> = vec![
        (1, 32, 8, [11, 11], [3, 3], p([1, 1], [1, 1], 1)),
        (2, 24, 12, [13, 10], [5, 5], p([2, 2], [1, 1], 1)),
        (3, 20, 10, [12, 10], [3, 3], p([2, 2], [2, 2], 1)),
        (2, 48, 16, [11, 13], [3, 3], p([1, 1], [1, 1], 2)),
        (1, 20, 8, [9, 17], [3, 5], p([1, 2], [1, 1], 1)),
        (5, 24, 20, [30, 27], [3, 3], p([1, 1], [1, 1], 1)),
    ];
    for (i, (n, c, o, hw, k, prm)) in cases.into_iter().enumerate() {
        // The shifted-rows copy of one image's group: `cg` planes `pstride` apart (an odd number of 16-float
        // lines), then `wq + 16` floats of slack.
        let (hq, wq) = (hw[0] + 2 * prm.padding[0], hw[1] + 2 * prm.padding[1]);
        let pstride = ((hq * wq).div_ceil(16) | 1) * 16;
        let len = c / prm.groups * pstride + wq + 16;
        assert!(len >= 4096, "case {i}: the copy must be long enough to be recycled");
        drop((0..2 * n * prm.groups + 4).map(|_| Tensor::full([len], f32::NAN)).collect::<Vec<_>>());
        let x = randn(&[n, c, hw[0], hw[1]], 1800 + i as u64).requires_grad_();
        let w = randn(&[o, c / prm.groups, k[0], k[1]], 1900 + i as u64).requires_grad_();
        let y = x.conv2d_with(&w, None, &prm).unwrap();
        let r = randn(y.dims(), 2000 + i as u64);
        let g = y.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
        let (_, ew) = naive_conv_grads(&x, &w, &r, &prm);
        assert_close(
            &g.get(&w).unwrap().to_vec::<f32>().unwrap(),
            &ew,
            1e-4,
            &format!("case {i} grad weight after recycled NaN buffers"),
        );
    }
}

#[test]
fn conv2d_output_size_overflow_is_an_error() {
    // Padding whose double overflows, and an output whose element count overflows: errors, not panics (or, in
    // release builds, a wrapped size).
    let x = Tensor::ones([1, 1, 1, 1]);
    let w = Tensor::ones([1, 1, 1, 1]);
    for pad in [usize::MAX / 2 + 1, 1 << 33] {
        assert!(x.conv2d(&w, None, 1, pad, 1, 1).is_err(), "padding {pad}");
        let _g = folo_core::no_grad();
        assert!(x.conv2d_act(&w, None, &Conv2dParams { padding: [pad; 2], ..p1() }, None).is_err());
        assert!(
            Tensor::conv2d_act_cat(
                &[&x, &x],
                &Tensor::ones([1, 2, 1, 1]),
                None,
                &Conv2dParams { padding: [pad; 2], ..p1() },
                None
            )
            .is_err()
        );
    }
}

fn p1() -> Conv2dParams {
    Conv2dParams { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 }
}

#[test]
fn upsample_output_size_overflow_is_an_error() {
    let x = Tensor::ones([1, 1, 2, 2]);
    assert!(x.upsample_nearest2d(usize::MAX / 2 + 1).is_err());
    assert!(x.upsample_nearest2d(1 << 33).is_err()); // each side fits, the element count does not
}

#[test]
fn pooling_size_overflow_is_an_error() {
    let x = Tensor::ones([1, 1, 2, 2]);
    // 2 · padding overflows; h + 2 · padding overflows; the window's element count overflows.
    assert!(x.max_pool2d(usize::MAX, 1, usize::MAX / 2 + 1).is_err());
    assert!(x.max_pool2d(usize::MAX, 1, usize::MAX / 2).is_err());
    assert!(x.avg_pool2d(usize::MAX, 1, usize::MAX / 2).is_err());
    let r = x.avg_pool2d(1 << 33, 1, 1 << 32);
    if let Ok(y) = r {
        assert_eq!(y.dims(), &[1, 1, 3, 3]);
        assert!(y.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }
}
