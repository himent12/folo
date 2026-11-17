//! BatchNorm statistics from the tensor-core conv's epilogue (`Backend::conv2d_bn_stats`, used by
//! `batch_norm_act`'s `stats`) against the CPU: the mean and variance of the conv output, the normalised output and
//! the running statistics.

use std::sync::{MutexGuard, OnceLock};

use folo_core::backend::{Conv2dParams, UnaryOp};
use folo_core::{BnStats, DType, Device, Tensor, no_grad};

use super::fixtures::{assert_close_t, randn};

/// A backend device on the tensor-core test context (registered once as `Gpu(3)`), with the tensor-core tests'
/// lock held so they stay on; `None` where the adapter has no tensor cores.
fn tensor_core_device() -> Option<(Device, MutexGuard<'static, ()>)> {
    static DEV: OnceLock<Device> = OnceLock::new();
    let (ctx, guard) = crate::tests::coop::coop_ctx()?;
    let dev = *DEV.get_or_init(|| {
        crate::register(ctx, Device::Gpu(3)).unwrap();
        Device::Gpu(3)
    });
    Some((dev, guard))
}

/// Per-channel mean and biased variance of `z` (`[N, C, H, W]`, any device and float dtype), in f64 on the CPU.
fn reference_stats(z: &Tensor) -> (Vec<f64>, Vec<f64>) {
    let (n, c, h, w) = z.dims4().unwrap();
    let v = z.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    let m = (n * h * w) as f64;
    let mut mean = vec![0.0; c];
    let mut var = vec![0.0; c];
    for ch in 0..c {
        let vals = (0..n).flat_map(|i| v[(i * c + ch) * h * w..][..h * w].iter().map(|&x| x as f64));
        mean[ch] = vals.clone().sum::<f64>() / m;
        var[ch] = vals.map(|x| (x - mean[ch]).powi(2)).sum::<f64>() / m;
    }
    (mean, var)
}

/// `silu((z − mean) · invstd · w + b)` from the reference statistics, f32 `[N, C, H, W]` on the CPU.
fn reference_y(z: &Tensor, mean: &[f64], var: &[f64], w: &[f32], b: &[f32], eps: f64) -> Tensor {
    let (n, c, h, wi) = z.dims4().unwrap();
    let v = z.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    let y: Vec<f32> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let ch = i / (h * wi) % c;
            let t = (x as f64 - mean[ch]) / (var[ch] + eps).sqrt() * w[ch] as f64 + b[ch] as f64;
            (t / (1.0 + (-t).exp())) as f32
        })
        .collect();
    Tensor::from_vec(y, [n, c, h, wi]).unwrap()
}

/// The statistics a conv gathers in its GEMM epilogue match the CPU's statistics of its output, so the BatchNorm
/// after it normalises (and updates its running statistics) as with its own statistics pass: pointwise and
/// implicit-GEMM convs, strided, with output channels and pixels that do not fill the GEMM tiles, in `F16` (on
/// tensor cores). `F32` convs have none, and the BatchNorm then gathers its own.
#[test]
fn conv_bn_stats_match_cpu() {
    let Some((dev, _lock)) = tensor_core_device() else { return };
    // (n, c_in, c_out, k, stride, h, w)
    let cases = [
        (2, 8, 12, 1, 1, 9, 7),
        (2, 32, 64, 1, 1, 40, 40),
        (3, 5, 70, 3, 1, 11, 13),
        (2, 16, 32, 3, 2, 20, 20),
        (1, 3, 16, 3, 2, 33, 31),
    ];
    let eps = 1e-3;
    for (i, &(n, ci, co, k, s, h, w)) in cases.iter().enumerate() {
        for dtype in [DType::F32, DType::F16] {
            let what = format!("{n}x{ci}x{h}x{w} -> {co} k{k} s{s} {dtype:?}");
            let seed = 100 + 10 * i as u64;
            // A per-channel offset keeps the means away from zero, so a wrong count or mean shows.
            let x = randn(&[n, ci, h, w], seed).add_scalar(0.5).unwrap();
            let wt = (&randn(&[co, ci, k, k], seed + 1) * 0.3).unwrap();
            let gamma = randn(&[co], seed + 2);
            let beta = randn(&[co], seed + 3);
            let p = Conv2dParams { stride: [s, s], padding: [k / 2, k / 2], ..Default::default() };
            let (xg, wg) = (x.to_dtype(dtype).unwrap().to_device(dev).unwrap(), wt.to_dtype(dtype).unwrap());
            let wg = wg.to_device(dev).unwrap();
            let (gg, bg) = (gamma.to_device(dev).unwrap(), beta.to_device(dev).unwrap());
            let _g = no_grad();
            let (z, stats) = xg.conv2d_bn_stats(&wg, &p).unwrap();
            assert_eq!(stats.is_some(), dtype == DType::F16, "{what}: statistics from the conv");
            // The conv output itself is the plain conv's.
            let plain = xg.conv2d_with(&wg, None, &p).unwrap();
            assert_close_t(&z, &plain, 1e-5, 1e-5, &format!("{what}: conv output"));

            let (mean, var) = reference_stats(&z);
            let running = |v: f32| Tensor::full([co], v).to_device(dev).unwrap();
            let (rm, rv) = (running(0.5), running(2.0));
            let out = z
                .batch_norm_act_train_stats(
                    &gg,
                    &bg,
                    eps,
                    Some(UnaryOp::Silu),
                    Some((&rm, &rv, 0.1)),
                    stats.as_ref(),
                )
                .unwrap()
                .expect("fused BatchNorm on the GPU");
            let tol = if dtype == DType::F16 { 2e-3 } else { 1e-4 };
            let f = |v: &[f64]| Tensor::from_vec(v.iter().map(|&x| x as f32).collect(), [co]).unwrap();
            assert_close_t(&out.mean, &f(&mean), tol, tol, &format!("{what}: mean"));
            let invstd: Vec<f64> = var.iter().map(|v| 1.0 / (v + eps).sqrt()).collect();
            assert_close_t(&out.invstd, &f(&invstd), tol, tol, &format!("{what}: invstd"));
            let m = (n * z.dims()[2] * z.dims()[3]) as f64;
            let rm_ref: Vec<f64> = mean.iter().map(|&x| 0.9 * 0.5 + 0.1 * x).collect();
            let rv_ref: Vec<f64> = var.iter().map(|&x| 0.9 * 2.0 + 0.1 * x * m / (m - 1.0)).collect();
            assert_close_t(&rm, &f(&rm_ref), tol, tol, &format!("{what}: running mean"));
            assert_close_t(&rv, &f(&rv_ref), tol, tol, &format!("{what}: running var"));
            let y_ref = reference_y(&z, &mean, &var, &gamma.to_vec().unwrap(), &beta.to_vec().unwrap(), eps);
            let ytol = if dtype == DType::F16 { 1e-2 } else { 1e-4 };
            let y = out.y.to_dtype(DType::F32).unwrap();
            assert_close_t(&y, &y_ref, ytol, ytol, &format!("{what}: y"));
        }
    }
}

/// Statistics are used only for the tensor they describe as it is: after an in-place write to the conv output,
/// or on another tensor, the BatchNorm gathers its own.
#[test]
fn stale_bn_stats_are_ignored() {
    let Some((dev, _lock)) = tensor_core_device() else { return };
    let x = randn(&[2, 8, 10, 10], 7).to_dtype(DType::F16).unwrap().to_device(dev).unwrap();
    let w = randn(&[16, 8, 1, 1], 8).to_dtype(DType::F16).unwrap().to_device(dev).unwrap();
    let (g, b) = (Tensor::ones([16]).to_device(dev).unwrap(), Tensor::zeros([16]).to_device(dev).unwrap());
    let _g = no_grad();
    let (z, stats) = x.conv2d_bn_stats(&w, &Conv2dParams::default()).unwrap();
    let stats: BnStats = stats.expect("statistics from the conv");
    let bn =
        |t: &Tensor| t.batch_norm_act_train_stats(&g, &b, 1e-3, None, None, Some(&stats)).unwrap().unwrap();
    // Another tensor of the same shape.
    let other = (&z * 2.0).unwrap().add_scalar(3.0).unwrap();
    let f = |v: Vec<f64>| Tensor::from_vec(v.into_iter().map(|x| x as f32).collect(), [16]).unwrap();
    assert_close_t(&bn(&other).mean, &f(reference_stats(&other).0), 1e-3, 1e-3, "another tensor");
    // The conv output after an in-place write.
    z.copy_from(&other).unwrap();
    assert_close_t(&bn(&z).mean, &f(reference_stats(&z).0), 1e-3, 1e-3, "modified in place");
}
