//! The ops mixed-precision training adds, through the folo-core backend: the fused BatchNorm on f16 data with f32
//! parameters and statistics, the f16 max/avg pooling and upsampling gradients, and folo-core's `autocast` conv and
//! matmul (f16 compute, gradients back into f32 leaves, the weight gradient written as f32), each against the same
//! computation in f32 on the f16-rounded inputs. These run with the shared context's kernels (tensor cores only if
//! `FOLO_GPU_TENSOR_CORES` is set); `tests/coop.rs` covers the tensor-core gradient kernels themselves.
//!
//! Tolerances: f16 storage rounds each stored value by at most 2^-11 relative; BatchNorm's statistics and every sum
//! are f32, so outputs and gradients that are stored once in f16 are within `2e-3` relative (plus a small absolute
//! term for values near zero) of the f32 run.

use folo_core::backend::UnaryOp;
use folo_core::{DType, Tensor, autocast, no_grad};

use super::backend::{assert_close_t, dev, randn};

/// The f16 rounding of `t`, as f32.
fn q(t: &Tensor) -> Tensor {
    t.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap()
}

macro_rules! dev16 {
    () => {{
        let dev = dev!();
        if !folo_core::backend::computes_f16(dev) {
            eprintln!("note: skipping f16 test, the adapter has no shader-f16");
            return;
        }
        dev
    }};
}

#[test]
fn batch_norm_on_f16_data_matches_f32() {
    let dev = dev16!();
    let _g = no_grad();
    for (shape, act) in
        [([4, 6, 9, 11], Some(UnaryOp::Silu)), ([2, 5, 8, 8], None), ([3, 4, 4, 4], Some(UnaryOp::Relu))]
    {
        let c = shape[1];
        let x =
            q(&randn(&shape, 1).mul_scalar(3.0).unwrap().add_scalar(0.5).unwrap()).to_device(dev).unwrap();
        let g = q(&randn(&shape, 2)).to_device(dev).unwrap();
        let w = randn(&[c], 3).add_scalar(1.0).unwrap().to_device(dev).unwrap();
        let b = randn(&[c], 4).to_device(dev).unwrap();
        let run = |x: &Tensor, g: &Tensor| {
            let (rm, rv) =
                (Tensor::zeros([c]).to_device(dev).unwrap(), Tensor::ones([c]).to_device(dev).unwrap());
            let out =
                x.batch_norm_act_train(&w, &b, 1e-3, act, Some((&rm, &rv, 0.03))).unwrap().expect("kernels");
            let (dx, dw, db) = x
                .batch_norm_act_train_backward(g, &w, &b, &out.mean, &out.invstd, act)
                .unwrap()
                .expect("kernels");
            (out, rm, rv, dx, dw, db)
        };
        let (x16, g16) = (x.to_dtype(DType::F16).unwrap(), g.to_dtype(DType::F16).unwrap());
        let (o16, rm16, rv16, dx16, dw16, db16) = run(&x16, &g16);
        let (o32, rm32, rv32, dx32, dw32, db32) = run(&x, &g);
        assert_eq!((o16.y.dtype(), dx16.dtype()), (DType::F16, DType::F16));
        assert_eq!((o16.mean.dtype(), dw16.dtype(), rm16.dtype()), (DType::F32, DType::F32, DType::F32));
        let what = format!("{shape:?} {act:?}");
        // Statistics and parameter gradients are f32 sums of the same values: summation order only.
        assert_close_t(&o16.mean, &o32.mean, 1e-5, 1e-5, &format!("{what} mean"));
        assert_close_t(&o16.invstd, &o32.invstd, 1e-5, 1e-5, &format!("{what} invstd"));
        assert_close_t(&rm16, &rm32, 1e-6, 1e-5, &format!("{what} running mean"));
        assert_close_t(&rv16, &rv32, 1e-6, 1e-5, &format!("{what} running var"));
        assert_close_t(&dw16, &dw32, 1e-4, 1e-4, &format!("{what} dweight"));
        assert_close_t(&db16, &db32, 1e-4, 1e-4, &format!("{what} dbias"));
        // Outputs and input gradients: one f16 rounding.
        assert_close_t(&o16.y.to_dtype(DType::F32).unwrap(), &o32.y, 2e-3, 2e-3, &format!("{what} y"));
        assert_close_t(&dx16.to_dtype(DType::F32).unwrap(), &dx32, 2e-3, 2e-3, &format!("{what} dx"));
    }
}

#[test]
fn pooling_and_upsampling_gradients_in_f16() {
    let dev = dev16!();
    let x = q(&randn(&[2, 3, 9, 10], 5)).to_device(dev).unwrap();
    let g_pool = q(&randn(&[2, 3, 9, 10], 6)).to_device(dev).unwrap();
    let g_avg = q(&randn(&[2, 3, 5, 5], 60)).to_device(dev).unwrap();
    let g_up = q(&randn(&[2, 3, 18, 30], 7)).to_device(dev).unwrap();
    let g_up2 = q(&randn(&[2, 3, 18, 20], 61)).to_device(dev).unwrap();
    // SPPF's 5×5 stride-1 max pooling, native f16 avg_pool2d, and the neck's 2×/2×3 upsampling, f16 vs f32.
    let grads = |dtype: DType| {
        let xl = x.to_dtype(dtype).unwrap().detach().requires_grad_();
        let pooled = xl.max_pool2d(5, 1, 2).unwrap();
        let avg = xl.avg_pool2d(3, 2, 1).unwrap();
        let up = xl.upsample_nearest2d_hw([2, 3]).unwrap();
        let up2 = xl.upsample_nearest2d(2).unwrap();
        assert_eq!((avg.dtype(), up2.dtype()), (dtype, dtype));
        let loss = pooled
            .mul(&g_pool.to_dtype(dtype).unwrap())
            .unwrap()
            .sum_all()
            .unwrap()
            .add(&avg.mul(&g_avg.to_dtype(dtype).unwrap()).unwrap().sum_all().unwrap())
            .unwrap()
            .add(&up.mul(&g_up.to_dtype(dtype).unwrap()).unwrap().sum_all().unwrap())
            .unwrap()
            .add(&up2.mul(&g_up2.to_dtype(dtype).unwrap()).unwrap().sum_all().unwrap())
            .unwrap();
        let gr = loss.backward().unwrap();
        let gx = gr.get(&xl).unwrap().clone();
        assert_eq!(gx.dtype(), dtype);
        gx.to_dtype(DType::F32).unwrap()
    };
    assert_close_t(
        &grads(DType::F16),
        &grads(DType::F32),
        2e-3,
        2e-3,
        "max_pool2d + avg_pool2d + upsample grad",
    );
}

/// folo-core's autocast conv on the GPU: f16 compute against the f32 conv of the same f16-rounded operands; the
/// gradients land in the f32 leaves (the weight gradient written as f32), the bias gradient summed in f32.
#[test]
fn autocast_conv_and_matmul_train_f32_leaves() {
    let dev = dev16!();
    let x = q(&randn(&[2, 8, 12, 12], 8)).to_device(dev).unwrap();
    let w = q(&randn(&[16, 8, 3, 3], 9).mul_scalar(0.2).unwrap()).to_device(dev).unwrap();
    let b = q(&randn(&[16], 10)).to_device(dev).unwrap();
    let m = q(&randn(&[16, 16], 11).mul_scalar(0.1).unwrap()).to_device(dev).unwrap();
    let run = |amp: bool| {
        let (xl, wl, bl, ml) = (
            x.detach().requires_grad_(),
            w.detach().requires_grad_(),
            b.detach().requires_grad_(),
            m.detach().requires_grad_(),
        );
        let y = {
            let _a = amp.then(autocast);
            let y = xl.conv2d(&wl, Some(&bl), 2, 1, 1, 1).unwrap();
            assert_eq!(y.dtype(), if amp { DType::F16 } else { DType::F32 });
            // [2, 16, 6, 6] → [2, 36, 16] @ [16, 16]
            let t = y.reshape([2, 16, 36]).unwrap().transpose(1, 2).unwrap();
            t.matmul(&ml).unwrap()
        };
        let loss = y.to_dtype(DType::F32).unwrap().square().unwrap().sum_all().unwrap();
        let gr = loss.backward().unwrap();
        let out: Vec<Tensor> = [&xl, &wl, &bl, &ml].iter().map(|t| gr.get(t).unwrap().clone()).collect();
        for g in &out {
            assert_eq!(g.dtype(), DType::F32, "the leaves are f32, so are their gradients");
        }
        (y.to_dtype(DType::F32).unwrap(), out)
    };
    let (y16, g16) = run(true);
    let (y32, g32) = run(false);
    assert_close_t(&y16, &y32, 1e-2, 1e-2, "autocast forward");
    for (name, (a, b)) in ["x", "weight", "bias", "matmul rhs"].iter().zip(g16.iter().zip(&g32)) {
        // Intermediate f16 roundings (the conv output, its gradient) compound to about 1e-2.
        let scale = b.abs().unwrap().max_all().unwrap().item::<f32>().unwrap();
        assert_close_t(a, b, 1e-2 * scale, 2e-2, &format!("autocast gradient of {name}"));
    }
    // Outside the guard nothing is cast.
    let y = x.conv2d(&w, Some(&b), 1, 1, 1, 1).unwrap();
    assert_eq!(y.dtype(), DType::F32);
    // The CPU never computes in f16, so autocast leaves CPU tensors alone.
    let _a = autocast();
    let xc = randn(&[1, 2, 5, 5], 12);
    assert_eq!(xc.conv2d(&randn(&[3, 2, 3, 3], 13), None, 1, 1, 1, 1).unwrap().dtype(), DType::F32);
}

/// B4 verification: vectorized `vec4` 4-lane Welford + pairwise Chan subgroup merge `bn_stats` / `bn_bwd_reduce`
/// match between the 32-wide subgroup fast path and the portable shared-memory fallback across both `hw % 4 == 0`
/// (VEC4) and odd `hw`, all activations, large shifted means (`+500.0`), and partial-subgroup extreme constants.
#[test]
fn batch_norm_subgroups_and_portable_fallback_agree() {
    struct ForceGuard(bool);
    impl ForceGuard {
        fn set(v: bool) -> Self {
            let prev = crate::kernels::norm::FORCE_NO_SUBGROUPS.replace(v);
            Self(prev)
        }
    }
    impl Drop for ForceGuard {
        fn drop(&mut self) {
            crate::kernels::norm::FORCE_NO_SUBGROUPS.set(self.0);
        }
    }
    let dev = dev16!();
    let _g = no_grad();
    for &(shape, act, shift) in &[
        ([4, 16, 20, 20], Some(UnaryOp::Silu), 500.0f32),
        ([3, 8, 9, 11], Some(UnaryOp::Silu), 120.0f32),
        ([2, 12, 8, 8], Some(UnaryOp::Relu), -300.0f32),
        ([2, 8, 16, 16], None, 0.0f32),
    ] {
        let c = shape[1];
        let x = q(&randn(&shape, 41).add_scalar(shift).unwrap()).to_device(dev).unwrap();
        let g = q(&randn(&shape, 42)).to_device(dev).unwrap();
        let w = randn(&[c], 43).add_scalar(1.0).unwrap().to_device(dev).unwrap();
        let b = randn(&[c], 44).to_device(dev).unwrap();
        let run = |force_portable: bool, dtype: DType| {
            let _guard = ForceGuard::set(force_portable);
            let xd = x.to_dtype(dtype).unwrap();
            let gd = g.to_dtype(dtype).unwrap();
            let out = xd.batch_norm_act_train(&w, &b, 1e-3, act, None).unwrap().expect("kernels");
            let (dx, dw, db) = xd
                .batch_norm_act_train_backward(&gd, &w, &b, &out.mean, &out.invstd, act)
                .unwrap()
                .expect("kernels");
            (
                out.y.to_dtype(DType::F32).unwrap(),
                out.mean,
                out.invstd,
                dx.to_dtype(DType::F32).unwrap(),
                dw,
                db,
            )
        };
        for dtype in [DType::F32, DType::F16] {
            let (y_sg, m_sg, iv_sg, dx_sg, dw_sg, db_sg) = run(false, dtype);
            let (y_fb, m_fb, iv_fb, dx_fb, dw_fb, db_fb) = run(true, dtype);
            let what = format!("bn sg vs portable {shape:?} {act:?} {dtype:?} shift={shift}");
            assert_close_t(&m_sg, &m_fb, 1e-4, 1e-5, &format!("{what} mean"));
            assert_close_t(&iv_sg, &iv_fb, 1e-4, 1e-4, &format!("{what} invstd"));
            assert_close_t(&y_sg, &y_fb, 2e-3, 2e-3, &format!("{what} y"));
            assert_close_t(&dx_sg, &dx_fb, 2e-3, 2e-3, &format!("{what} dx"));
            assert_close_t(&dw_sg, &dw_fb, 2e-3, 1e-3, &format!("{what} dw"));
            assert_close_t(&db_sg, &db_fb, 2e-3, 2e-3, &format!("{what} db"));
        }
    }

    // Extreme finite inputs against f64 reference: no sum2 overflow (`1e19`), no subgroup `cnt * mean` overflow
    // (`1e37`), and no first-element outlier cancellation (`1e5` at index 0 with ±1 background).
    let w1 = Tensor::from_slice(&[1.0f32], [1]).unwrap().to_device(dev).unwrap();
    let b0 = Tensor::from_slice(&[0.0f32], [1]).unwrap().to_device(dev).unwrap();
    let mut spike = vec![0.0f32; 2048];
    spike[0] = 1e19;
    let huge_const = vec![1e37f32; 1024];
    let mut outlier: Vec<f32> = (0..1024).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    outlier[0] = 1e5;
    for (name, vals) in [("spike_1e19", spike), ("const_1e37", huge_const), ("outlier_1e5", outlier)] {
        let n = vals.len();
        let mean_ref = vals.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var_ref = vals.iter().map(|&v| (v as f64 - mean_ref).powi(2)).sum::<f64>() / n as f64;
        let invstd_ref = (var_ref + 1e-5).sqrt().recip() as f32;
        let x = Tensor::from_vec(vals, [1, 1, 1, n]).unwrap().to_device(dev).unwrap();
        for force_portable in [false, true] {
            let _guard = ForceGuard::set(force_portable);
            let out = x.batch_norm_act_train(&w1, &b0, 1e-5, None, None).unwrap().expect("kernels");
            let m = out.mean.to_vec::<f32>().unwrap()[0];
            let iv = out.invstd.to_vec::<f32>().unwrap()[0];
            assert!(
                m.is_finite() && iv.is_finite(),
                "{name} portable={force_portable}: non-finite mean={m} invstd={iv}"
            );
            assert!(
                (m - mean_ref as f32).abs() <= 1e-4 * (mean_ref.abs() as f32).max(1.0),
                "{name} portable={force_portable}: mean {m} vs {mean_ref}"
            );
            assert!(
                iv > 0.0 && (iv - invstd_ref).abs() <= 1e-3 * invstd_ref,
                "{name} portable={force_portable}: invstd {iv} vs {invstd_ref}"
            );
        }
    }

    // Partial-subgroup constant inputs (`3` identical elements): pairwise Chan subgroup merge must keep `d == 0`
    // bit-exact so variance is strictly zero (`invstd = 1/sqrt(eps)`, `y == 0`), including `3 x 65504.0` in F16
    // and `3 x 1e37` in F32 (where weighted `subgroupAdd` rounding caused fake nonzero variance / M2 overflow).
    let expected_invstd = (1e-5f64).sqrt().recip() as f32;
    for (name, dt, val) in
        [("partial_f16_65504", DType::F16, 65504.0f32), ("partial_f32_1e37", DType::F32, 1e37f32)]
    {
        let x = Tensor::from_vec(vec![val; 3], [1, 1, 1, 3])
            .unwrap()
            .to_device(dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap();
        for force_portable in [false, true] {
            let _guard = ForceGuard::set(force_portable);
            let out = x.batch_norm_act_train(&w1, &b0, 1e-5, None, None).unwrap().expect("kernels");
            let m = out.mean.to_vec::<f32>().unwrap()[0];
            let iv = out.invstd.to_vec::<f32>().unwrap()[0];
            let yv = out.y.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
            assert_eq!(m, val, "{name} portable={force_portable}: mean drift {m} vs {val}");
            assert!(
                (iv - expected_invstd).abs() <= 1e-3 * expected_invstd,
                "{name} portable={force_portable}: invstd {iv} vs {expected_invstd}"
            );
            assert!(
                yv.iter().all(|&v| v.abs() < 1e-4),
                "{name} portable={force_portable}: constant BN output non-zero: {yv:?}"
            );
        }
    }

    // Multi-chunk (`chunks = 40 > 32`) with `RUNNING = true`: exercises both intra-subgroup and inter-subgroup
    // merges in `bn_finalize` and `bn_bwd_finalize` plus running-mean / running-var updates.
    {
        let shape = [4usize, 2, 64, 320];
        let c = shape[1];
        let x = randn(&shape, 51).add_scalar(25.0).unwrap().to_device(dev).unwrap();
        let g = randn(&shape, 52).to_device(dev).unwrap();
        let w = randn(&[c], 53).add_scalar(1.0).unwrap().to_device(dev).unwrap();
        let b = randn(&[c], 54).to_device(dev).unwrap();
        let run_mc = |force_portable: bool| {
            let _guard = ForceGuard::set(force_portable);
            let rm = Tensor::from_slice(&[1.0f32, -2.0], [c]).unwrap().to_device(dev).unwrap();
            let rv = Tensor::from_slice(&[0.5f32, 1.5], [c]).unwrap().to_device(dev).unwrap();
            let out = x
                .batch_norm_act_train(&w, &b, 1e-3, Some(UnaryOp::Silu), Some((&rm, &rv, 0.03)))
                .unwrap()
                .expect("kernels");
            let (dx, dw, db) = x
                .batch_norm_act_train_backward(&g, &w, &b, &out.mean, &out.invstd, Some(UnaryOp::Silu))
                .unwrap()
                .expect("kernels");
            (out.y, out.mean, out.invstd, rm, rv, dx, dw, db)
        };
        let (y_sg, m_sg, iv_sg, rm_sg, rv_sg, dx_sg, dw_sg, db_sg) = run_mc(false);
        let (y_fb, m_fb, iv_fb, rm_fb, rv_fb, dx_fb, dw_fb, db_fb) = run_mc(true);
        assert_close_t(&m_sg, &m_fb, 1e-4, 1e-5, "bn multi-chunk mean");
        assert_close_t(&iv_sg, &iv_fb, 1e-4, 1e-4, "bn multi-chunk invstd");
        assert_close_t(&rm_sg, &rm_fb, 1e-4, 1e-5, "bn multi-chunk running_mean");
        assert_close_t(&rv_sg, &rv_fb, 1e-4, 1e-4, "bn multi-chunk running_var");
        assert_close_t(&y_sg, &y_fb, 1e-4, 1e-4, "bn multi-chunk y");
        assert_close_t(&dx_sg, &dx_fb, 1e-4, 1e-4, "bn multi-chunk dx");
        assert_close_t(&dw_sg, &dw_fb, 1e-3, 1e-4, "bn multi-chunk dw");
        assert_close_t(&db_sg, &db_fb, 1e-3, 1e-4, "bn multi-chunk db");
    }
}
