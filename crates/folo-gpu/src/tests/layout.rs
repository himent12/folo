//! Permute / transpose, concat, upsample, max_pool2d and casts.

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::pooling::Window2d;
use crate::{DType, GpuTensor};

fn cpu_permute(x: &[f32], shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let r = shape.len();
    let mut strides = vec![1; r];
    for i in (0..r.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let out: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let n: usize = out.iter().product();
    (0..n)
        .map(|i| {
            let (mut rem, mut src) = (i, 0);
            for d in (0..r).rev() {
                src += (rem % out[d]) * strides[perm[d]];
                rem /= out[d];
            }
            x[src]
        })
        .collect()
}

#[test]
fn permute_and_transpose() {
    let ctx = gpu!();
    let mut rng = Rng::new(30);
    let cases: &[(&[usize], &[usize])] = &[
        (&[2, 3, 4, 5], &[0, 2, 3, 1]), // NCHW -> NHWC (batched transpose)
        (&[2, 3, 4, 5], &[0, 3, 1, 2]), // NHWC -> NCHW
        (&[37, 45], &[1, 0]),           // plain 2-D transpose with ragged tiles
        (&[3, 65, 33], &[0, 2, 1]),
        (&[2, 3, 4, 5], &[3, 1, 0, 2]), // generic
        (&[2, 3, 4], &[0, 1, 2]),       // identity
        (&[1, 6, 1, 7], &[3, 2, 1, 0]),
        (&[2, 2, 2, 2, 2, 2], &[5, 3, 1, 4, 2, 0]),
    ];
    for &(shape, perm) in cases {
        let x = rng.vec(shape.iter().product(), -1.0, 1.0);
        let gx = GpuTensor::from_slice(&ctx, &x, shape).unwrap();
        let got = gx.permute(perm).unwrap();
        let want_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        assert_eq!(got.shape(), want_shape.as_slice());
        assert_eq!(got.to_vec().unwrap(), cpu_permute(&x, shape, perm), "permute {shape:?} {perm:?}");
    }
    let gx = GpuTensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(gx.transpose(0, 1).unwrap().to_vec().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(gx.permute(&[0, 0]).is_err());
}

#[test]
fn concat_along_dims() {
    let ctx = gpu!();
    let mut rng = Rng::new(31);
    for (shapes, dim) in [
        (vec![vec![2, 3, 5, 5], vec![2, 1, 5, 5], vec![2, 4, 5, 5]], 1), // odd planes: scalar copy
        (vec![vec![2, 8, 4, 4], vec![2, 4, 4, 4]], 1),                   // vec4 copy
        (vec![vec![1, 8, 4, 4], vec![1, 4, 4, 4]], 1),                   // single block: buffer copies
        (vec![vec![3, 2], vec![1, 2]], 0),
        (vec![vec![2, 3], vec![2, 1], vec![2, 2]], 1),
    ] {
        let data: Vec<Vec<f32>> = shapes.iter().map(|s| rng.vec(s.iter().product(), -1.0, 1.0)).collect();
        let parts: Vec<GpuTensor> =
            data.iter().zip(&shapes).map(|(d, s)| GpuTensor::from_slice(&ctx, d, s).unwrap()).collect();
        let refs: Vec<&GpuTensor> = parts.iter().collect();
        let got = GpuTensor::concat(&refs, dim).unwrap();
        let outer: usize = shapes[0][..dim].iter().product();
        let mut want = Vec::new();
        for o in 0..outer {
            for (d, s) in data.iter().zip(&shapes) {
                let row: usize = s[dim..].iter().product();
                want.extend_from_slice(&d[o * row..(o + 1) * row]);
            }
        }
        assert_eq!(got.to_vec().unwrap(), want, "concat {shapes:?} dim {dim}");
    }
    let a = GpuTensor::zeros(&ctx, &[2, 3], DType::F32).unwrap();
    let b = GpuTensor::zeros(&ctx, &[3, 3], DType::F32).unwrap();
    assert!(GpuTensor::concat(&[&a, &b], 1).is_err());
}

#[test]
fn upsample_nearest() {
    let ctx = gpu!();
    let mut rng = Rng::new(32);
    for (h, w, oh, ow) in [(5, 4, 10, 8), (3, 3, 8, 5), (6, 6, 3, 3)] {
        let x = rng.vec(2 * 3 * h * w, -1.0, 1.0);
        let got = GpuTensor::from_slice(&ctx, &x, &[2, 3, h, w]).unwrap().upsample_nearest2d(oh, ow).unwrap();
        let mut want = Vec::new();
        for p in 0..6 {
            for y in 0..oh {
                for xx in 0..ow {
                    want.push(x[(p * h + y * h / oh) * w + xx * w / ow]);
                }
            }
        }
        assert_eq!(got.to_vec().unwrap(), want, "upsample {h}x{w} -> {oh}x{ow}");
    }
}

fn cpu_max_pool(x: &[f32], planes: usize, h: usize, w: usize, win: &Window2d) -> Vec<f32> {
    let (oh, ow) = win.output(h, w).unwrap();
    let mut out = Vec::new();
    for p in 0..planes {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = f32::NEG_INFINITY;
                for ky in 0..win.kernel.0 {
                    for kx in 0..win.kernel.1 {
                        let iy = (oy * win.stride.0 + ky * win.dilation.0) as isize - win.padding.0 as isize;
                        let ix = (ox * win.stride.1 + kx * win.dilation.1) as isize - win.padding.1 as isize;
                        if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                            m = m.max(x[(p * h + iy as usize) * w + ix as usize]);
                        }
                    }
                }
                out.push(m);
            }
        }
    }
    out
}

#[test]
fn max_pool() {
    let ctx = gpu!();
    let mut rng = Rng::new(33);
    for win in [
        Window2d::square(2, 2, 0),
        Window2d::square(3, 2, 1),
        Window2d::square(5, 1, 2), // SPPF
        Window2d { kernel: (3, 2), stride: (1, 2), padding: (1, 0), dilation: (2, 1) },
    ] {
        let (h, w) = (9, 11);
        let x = rng.vec(2 * 3 * h * w, -1.0, 1.0);
        let got = GpuTensor::from_slice(&ctx, &x, &[2, 3, h, w]).unwrap().max_pool2d(win).unwrap();
        let (oh, ow) = win.output(h, w).unwrap();
        assert_eq!(got.shape(), &[2, 3, oh, ow]);
        assert_eq!(got.to_vec().unwrap(), cpu_max_pool(&x, 6, h, w, &win), "max_pool {win:?}");
    }
    let gx = GpuTensor::zeros(&ctx, &[1, 1, 4, 4], DType::F32).unwrap();
    assert!(gx.max_pool2d(Window2d::square(2, 2, 2)).is_err());
}

#[test]
fn casts_round_trip() {
    let ctx = gpu!();
    let mut rng = Rng::new(34);
    for n in [1, 2, 7, 1000] {
        let x = rng.vec(n, -100.0, 100.0);
        let gx = GpuTensor::from_slice(&ctx, &x, &[n]).unwrap();
        let h = gx.to_dtype(DType::F16).unwrap();
        assert_eq!(h.dtype(), DType::F16);
        // pack2x16float may round either way at the last bit: allow one f16 ulp.
        assert_close(&h.to_vec().unwrap(), &f16_round(&x), 0.0, 1e-3, &format!("f32 -> f16 n={n}"));
        let back = h.to_dtype(DType::F32).unwrap().to_vec().unwrap();
        assert_eq!(back, h.to_vec().unwrap(), "f16 -> f32 n={n}");
        let up = GpuTensor::from_f16(&ctx, &to_f16(&x), &[n]).unwrap();
        assert_eq!(up.to_f16_vec().unwrap(), to_f16(&x));
    }
}

#[test]
fn f16_layout_ops() {
    let ctx = gpu!();
    if !ctx.caps().shader_f16 {
        eprintln!("note: skipping f16 layout test, adapter lacks shader-f16");
        return;
    }
    let mut rng = Rng::new(35);
    let x = f16_round(&rng.vec(2 * 4 * 6 * 6, -1.0, 1.0));
    let gx = GpuTensor::from_f16(&ctx, &to_f16(&x), &[2, 4, 6, 6]).unwrap();
    let perm = [0, 2, 3, 1];
    assert_eq!(gx.permute(&perm).unwrap().to_vec().unwrap(), cpu_permute(&x, &[2, 4, 6, 6], &perm));
    let win = Window2d::square(3, 2, 1);
    assert_eq!(gx.max_pool2d(win).unwrap().to_vec().unwrap(), cpu_max_pool(&x, 8, 6, 6, &win));
    let up = gx.upsample_nearest2d(12, 12).unwrap().to_vec().unwrap();
    // Plane 1, pixel (1, 1) comes from plane 1, pixel (0, 0); pixel (3, 5) from (1, 2).
    assert_eq!(up[144 + 13], x[36]);
    assert_eq!(up[144 + 3 * 12 + 5], x[36 + 6 + 2]);
    let cat = GpuTensor::concat(&[&gx, &gx], 1).unwrap();
    assert_eq!(cat.shape(), &[2, 8, 6, 6]);
    let cat = cat.to_vec().unwrap();
    assert_eq!(&cat[..144], &x[..144]);
    assert_eq!(&cat[144..288], &x[..144]);
}
