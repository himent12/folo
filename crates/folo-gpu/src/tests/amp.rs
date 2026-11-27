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

/// Verifies single-dispatch register-cached `bn_fused_fwd` / `bn_fused_bwd` against the CPU f64 reference
/// across `REGS = 4` / `REGS = 8`, exact boundaries (`1024`, `2048`, `2049`), odd `hw`, prime channels,
/// `RUNNING = true/false`, all activations, F32 & F16, and both subgroup and portable paths, plus dispatch trace.
#[test]
fn batch_norm_fused_single_dispatch_parity_and_trace() {
    use folo_core::backend::{Backend, RunningUpdate, View};
    use folo_core::{Layout, Shape};

    struct ForceGuard(bool);
    impl ForceGuard {
        fn set(v: bool) -> Self {
            Self(crate::kernels::norm::FORCE_NO_SUBGROUPS.replace(v))
        }
    }
    impl Drop for ForceGuard {
        fn drop(&mut self) {
            crate::kernels::norm::FORCE_NO_SUBGROUPS.set(self.0);
        }
    }

    let dev = dev!();
    let ctx = crate::context_for(dev).unwrap();
    let has_f16 = ctx.caps().shader_f16;
    let _g = no_grad();

    // Shapes covering REGS=4 (m_u<=1024), exact REGS=4 boundary (m_u=1024), REGS=8 (m_u=1600),
    // exact REGS=8 boundary (m_u=2048), multi-chunk boundary (m_u=2049 odd), odd hw, and 1x1x1x1.
    let cases: &[([usize; 4], Option<UnaryOp>, bool)] = &[
        ([4, 32, 8, 8], Some(UnaryOp::Silu), true),
        ([16, 32, 16, 16], Some(UnaryOp::Relu), true), // m_u = 1024 (REGS=4 exact)
        ([16, 32, 20, 20], Some(UnaryOp::Silu), false), // m_u = 1600 (REGS=8)
        ([32, 32, 16, 16], None, true),                // m_u = 2048 (REGS=8 exact)
        ([4, 37, 7, 9], Some(UnaryOp::Silu), true),    // odd hw + prime C (REGS=4 scalar)
        ([4, 32, 21, 21], Some(UnaryOp::Silu), true),  // odd hw, m_u=1764 (REGS=8 scalar)
        ([3, 1, 1, 2049], Some(UnaryOp::Silu), false), // m_u = 6147 > 2048 (multi-chunk odd)
        ([1, 3, 1, 1], Some(UnaryOp::Silu), true),     // m = 1 edge case
    ];
    let eps = 1e-3f64;
    let mom = 0.1f64;

    for &(shape, act, with_running) in cases {
        let [n, c, h, w_dim] = shape;
        let hw = h * w_dim;
        let m = n * hw;
        let x_cpu = randn(&shape, 100 + c as u64 + hw as u64);
        let g_cpu = randn(&shape, 200 + c as u64 + hw as u64);
        let w_cpu = randn(&[c], 300 + c as u64).add_scalar(1.0).unwrap();
        let b_cpu = randn(&[c], 400 + c as u64);

        // Exact f64 CPU reference for forward & backward.
        let xv = x_cpu.to_vec::<f32>().unwrap();
        let gv = g_cpu.to_vec::<f32>().unwrap();
        let wv = w_cpu.to_vec::<f32>().unwrap();
        let bv = b_cpu.to_vec::<f32>().unwrap();
        let mut ref_mean = vec![0.0f32; c];
        let mut ref_inv = vec![0.0f32; c];
        let mut ref_rm = vec![0.25f32; c];
        let mut ref_rv = vec![0.75f32; c];
        let mut ref_y = vec![0.0f32; n * c * hw];
        let mut ref_dx = vec![0.0f32; n * c * hw];
        let mut ref_dw = vec![0.0f32; c];
        let mut ref_db = vec![0.0f32; c];
        for ch in 0..c {
            let mut sum = 0.0f64;
            for bi in 0..n {
                for j in 0..hw {
                    sum += xv[(bi * c + ch) * hw + j] as f64;
                }
            }
            let mean = sum / m as f64;
            let mut sq = 0.0f64;
            for bi in 0..n {
                for j in 0..hw {
                    let d = xv[(bi * c + ch) * hw + j] as f64 - mean;
                    sq += d * d;
                }
            }
            let var_b = sq / m as f64;
            let inv = 1.0 / (var_b + eps).sqrt();
            ref_mean[ch] = mean as f32;
            ref_inv[ch] = inv as f32;
            if with_running {
                let unbias = if m > 1 { m as f64 / (m as f64 - 1.0) } else { 1.0 };
                ref_rm[ch] = (ref_rm[ch] as f64 * (1.0 - mom) + mom * mean) as f32;
                ref_rv[ch] = (ref_rv[ch] as f64 * (1.0 - mom) + mom * unbias * var_b) as f32;
            }
            let (wc, bc) = (wv[ch] as f64, bv[ch] as f64);
            let mut db_acc = 0.0f64;
            let mut dw_acc = 0.0f64;
            for bi in 0..n {
                for j in 0..hw {
                    let idx = (bi * c + ch) * hw + j;
                    let xh = (xv[idx] as f64 - mean) * inv;
                    let z = xh * wc + bc;
                    let (y, dz) = match act {
                        None => (z, gv[idx] as f64),
                        Some(UnaryOp::Relu) => (z.max(0.0), if z > 0.0 { gv[idx] as f64 } else { 0.0 }),
                        Some(UnaryOp::Silu) => {
                            let s = 1.0 / (1.0 + (-z).exp());
                            (z * s, gv[idx] as f64 * s * (1.0 + z * (1.0 - s)))
                        }
                        _ => unreachable!(),
                    };
                    ref_y[idx] = y as f32;
                    db_acc += dz;
                    dw_acc += dz * xh;
                }
            }
            ref_db[ch] = db_acc as f32;
            ref_dw[ch] = dw_acc as f32;
            for bi in 0..n {
                for j in 0..hw {
                    let idx = (bi * c + ch) * hw + j;
                    let xh = (xv[idx] as f64 - mean) * inv;
                    let z = xh * wc + bc;
                    let dz = match act {
                        None => gv[idx] as f64,
                        Some(UnaryOp::Relu) => {
                            if z > 0.0 {
                                gv[idx] as f64
                            } else {
                                0.0
                            }
                        }
                        Some(UnaryOp::Silu) => {
                            let s = 1.0 / (1.0 + (-z).exp());
                            gv[idx] as f64 * s * (1.0 + z * (1.0 - s))
                        }
                        _ => unreachable!(),
                    };
                    ref_dx[idx] = (wc * inv * (dz - db_acc / m as f64 - xh * dw_acc / m as f64)) as f32;
                }
            }
        }

        let x_g = x_cpu.to_device(dev).unwrap();
        let g_g = g_cpu.to_device(dev).unwrap();
        let w_g = w_cpu.to_device(dev).unwrap();
        let b_g = b_cpu.to_device(dev).unwrap();
        let y_ref_t = Tensor::from_vec(ref_y, shape).unwrap();
        let dx_ref_t = Tensor::from_vec(ref_dx, shape).unwrap();
        let mean_ref_t = Tensor::from_vec(ref_mean, [c]).unwrap();
        let inv_ref_t = Tensor::from_vec(ref_inv, [c]).unwrap();
        let rm_ref_t = Tensor::from_vec(ref_rm, [c]).unwrap();
        let rv_ref_t = Tensor::from_vec(ref_rv, [c]).unwrap();
        let dw_ref_t = Tensor::from_vec(ref_dw, [c]).unwrap();
        let db_ref_t = Tensor::from_vec(ref_db, [c]).unwrap();

        for portable in [false, true] {
            let _guard = ForceGuard::set(portable);
            let rm = Tensor::full([c], 0.25).to_device(dev).unwrap();
            let rv = Tensor::full([c], 0.75).to_device(dev).unwrap();
            let run_opt = with_running.then_some((&rm, &rv, mom));
            let out = x_g.batch_norm_act_train(&w_g, &b_g, eps, act, run_opt).unwrap().expect("kernels");
            let (dx, dw, db) = x_g
                .batch_norm_act_train_backward(&g_g, &w_g, &b_g, &out.mean, &out.invstd, act)
                .unwrap()
                .expect("kernels");
            let tag = format!("bn_fused {shape:?} {act:?} run={with_running} fb={portable}");
            assert_close_t(&out.mean, &mean_ref_t, 1e-4, 1e-4, &format!("{tag} mean"));
            assert_close_t(&out.invstd, &inv_ref_t, 2e-4, 1e-4, &format!("{tag} invstd"));
            assert_close_t(&out.y, &y_ref_t, 5e-4, 2e-4, &format!("{tag} y"));
            assert_close_t(&dx, &dx_ref_t, 5e-4, 2e-4, &format!("{tag} dx"));
            assert_close_t(&dw, &dw_ref_t, 2e-3, 5e-4, &format!("{tag} dw"));
            assert_close_t(&db, &db_ref_t, 2e-3, 5e-4, &format!("{tag} db"));
            if with_running {
                assert_close_t(&rm, &rm_ref_t, 1e-4, 1e-4, &format!("{tag} rm"));
                assert_close_t(&rv, &rv_ref_t, 2e-4, 1e-4, &format!("{tag} rv"));
            }
        }
    }

    // F16 single-dispatch fused parity on REGS=8 shape ([16, 32, 20, 20]) and odd hw ([4, 32, 9, 11]).
    if has_f16 {
        for shape in [[16usize, 32, 20, 20], [4, 32, 9, 11]] {
            let c = shape[1];
            let x = q(&randn(&shape, 501)).to_device(dev).unwrap();
            let g = q(&randn(&shape, 502)).to_device(dev).unwrap();
            let w = randn(&[c], 503).add_scalar(1.0).unwrap().to_device(dev).unwrap();
            let b = randn(&[c], 504).to_device(dev).unwrap();
            let o32 = x.batch_norm_act_train(&w, &b, eps, Some(UnaryOp::Silu), None).unwrap().unwrap();
            let (dx32, dw32, db32) = x
                .batch_norm_act_train_backward(&g, &w, &b, &o32.mean, &o32.invstd, Some(UnaryOp::Silu))
                .unwrap()
                .unwrap();
            let (x16, g16) = (x.to_dtype(DType::F16).unwrap(), g.to_dtype(DType::F16).unwrap());
            let o16 = x16.batch_norm_act_train(&w, &b, eps, Some(UnaryOp::Silu), None).unwrap().unwrap();
            let (dx16, dw16, db16) = x16
                .batch_norm_act_train_backward(&g16, &w, &b, &o16.mean, &o16.invstd, Some(UnaryOp::Silu))
                .unwrap()
                .unwrap();
            assert_close_t(&o16.mean, &o32.mean, 1e-4, 1e-4, &format!("f16 fused {shape:?} mean"));
            assert_close_t(&o16.invstd, &o32.invstd, 1e-4, 1e-4, &format!("f16 fused {shape:?} invstd"));
            assert_close_t(
                &o16.y.to_dtype(DType::F32).unwrap(),
                &o32.y,
                3e-3,
                3e-3,
                &format!("f16 fused {shape:?} y"),
            );
            assert_close_t(
                &dx16.to_dtype(DType::F32).unwrap(),
                &dx32,
                3e-3,
                3e-3,
                &format!("f16 fused {shape:?} dx"),
            );
            assert_close_t(&dw16, &dw32, 2e-3, 1e-3, &format!("f16 fused {shape:?} dw"));
            assert_close_t(&db16, &db32, 2e-3, 1e-3, &format!("f16 fused {shape:?} db"));
        }
    }

    // High-offset F32 inputs (`1e7 ± 1`, REGS=8 boundary at 8192 elems/ch) must keep exact variance and invstd
    // without Welford running-mean rounding drift, including on asymmetric fractional-mean blocks.
    {
        let c = 32usize;
        let hw = 8192usize;
        let mut data = vec![0.0f32; c * hw];
        let pattern = [-1.0f32, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
        for ch in 0..c {
            let base = 10_000_000.0f32;
            for (blk, &delta) in pattern.iter().enumerate() {
                let d = if ch == 1 && blk == 0 {
                    // Asymmetric channel with fractional mean (7 of 8 blocks at +1, 1 at -1 -> var = 7/16).
                    -1.0
                } else if ch == 1 {
                    1.0
                } else {
                    delta
                };
                data[ch * hw + blk * 1024..ch * hw + (blk + 1) * 1024].fill(base + d);
            }
        }
        let x = Tensor::from_vec(data, [1, c, 1, hw]).unwrap().to_device(dev).unwrap();
        let w = Tensor::ones([c]).to_device(dev).unwrap();
        let b = Tensor::zeros([c]).to_device(dev).unwrap();
        let out = x.batch_norm_act_train(&w, &b, 1e-3, None, None).unwrap().unwrap();
        let mean_v = out.mean.to_vec::<f32>().unwrap();
        let inv_v = out.invstd.to_vec::<f32>().unwrap();
        let y_v = out.y.to_vec::<f32>().unwrap();
        let want_inv0 = 1.0f32 / (1.0f32 + 1e-3).sqrt();
        assert_eq!(mean_v[0], 10_000_000.0, "high-offset exact mean");
        assert!(
            (inv_v[0] - want_inv0).abs() < 1e-6,
            "high-offset invstd: got {}, want {want_inv0}",
            inv_v[0]
        );
        assert!((y_v[0] + want_inv0).abs() < 1e-5, "high-offset y[0]: got {}, want {}", y_v[0], -want_inv0);
        // Channel 1: var = 7/16 = 0.4375.
        let want_inv1 = 1.0f32 / (0.4375f32 + 1e-3).sqrt();
        assert!(
            (inv_v[1] - want_inv1).abs() < 1e-5,
            "high-offset asymmetric invstd: got {}, want {want_inv1}",
            inv_v[1]
        );
    }

    // Dispatch trace on a fresh context: register-cached fused single-dispatch (REGS=4/8 up to 2048 units),
    // streaming fused single-dispatch (REGS=0 up to 6400 units with C >= 32, and up to 25600 units with C >= 64),
    // vs 3-pass multi-chunk above or with fewer channels. `bn_fused_bwd` requires 9 storage bindings; adapters
    // capped at 8 fall back to 3-dispatch bwd.
    let fresh = crate::GpuContext::new().unwrap();
    let backend = crate::GpuBackend::new(fresh.clone(), dev);
    let fused_bwd_trace = if fresh.caps().max_storage_buffers >= 9 {
        vec!["bn_fused_bwd"]
    } else {
        vec!["bn_bwd_reduce", "bn_bwd_finalize", "bn_bwd_dx"]
    };
    for (dims, expect_fwd, expect_bwd) in [
        ([4usize, 32, 8, 8], vec!["bn_fused_fwd"], fused_bwd_trace.clone()),
        ([16, 32, 20, 20], vec!["bn_fused_fwd"], fused_bwd_trace.clone()),
        ([16, 64, 40, 40], vec!["bn_fused_fwd"], fused_bwd_trace.clone()),
        ([16, 32, 40, 40], vec!["bn_fused_fwd"], fused_bwd_trace.clone()),
        ([16, 64, 80, 80], vec!["bn_fused_fwd"], fused_bwd_trace),
        (
            [16, 16, 40, 40],
            vec!["bn_stats", "bn_finalize", "bn_apply"],
            vec!["bn_bwd_reduce", "bn_bwd_finalize", "bn_bwd_dx"],
        ),
        (
            [16, 32, 80, 80],
            vec!["bn_stats", "bn_finalize", "bn_apply"],
            vec!["bn_bwd_reduce", "bn_bwd_finalize", "bn_bwd_dx"],
        ),
    ] {
        let s = Shape::new(&dims);
        let c = dims[1];
        let sc = Shape::new(&[c]);
        let x = backend.full(dev, DType::F32, s.numel(), 1.0).unwrap();
        let g = backend.full(dev, DType::F32, s.numel(), 0.5).unwrap();
        let w = backend.full(dev, DType::F32, c, 1.0).unwrap();
        let b = backend.full(dev, DType::F32, c, 0.0).unwrap();
        let mut rm = backend.full(dev, DType::F32, c, 0.0).unwrap();
        let mut rv = backend.full(dev, DType::F32, c, 1.0).unwrap();
        let lx = Layout::contiguous(s.clone());
        let lc = Layout::contiguous(sc.clone());
        fresh.trace_begin();
        let (_, mean, inv) = backend
            .batch_norm_act(
                View::new(&x, &lx),
                View::new(&w, &lc),
                View::new(&b, &lc),
                eps as f32,
                Some(UnaryOp::Silu),
                Some(RunningUpdate { mean: &mut rm, var: &mut rv, momentum: 0.03 }),
            )
            .unwrap();
        assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), expect_fwd);
        fresh.trace_begin();
        let _ = backend
            .batch_norm_act_backward(
                View::new(&x, &lx),
                View::new(&g, &lx),
                View::new(&w, &lc),
                View::new(&b, &lc),
                View::new(&mean, &lc),
                View::new(&inv, &lc),
                Some(UnaryOp::Silu),
            )
            .unwrap();
        assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), expect_bwd);
    }
    assert!(fresh.kernels_without_registry().is_empty());
}

/// Streaming single-dispatch BatchNorm (`REGS = 0`, above 2,048 units up to 6,400 units with `C >= 64`) matches
/// the CPU reference in both `F32` and `F16`, on both 4-aligned and odd spatial sizes, across subgroup and
/// portable fallback paths, and preserves exact variance on high-offset inputs.
#[test]
fn batch_norm_streaming_fused_matches_cpu_f32_and_f16() {
    struct ResetSubgroups(bool);
    impl Drop for ResetSubgroups {
        fn drop(&mut self) {
            crate::kernels::norm::FORCE_NO_SUBGROUPS.set(self.0);
        }
    }

    let dev = dev!();
    let has_f16 = folo_core::backend::computes_f16(dev);
    let _g = no_grad();
    let eps = 1e-3f64;

    // 1. VEC4 [16, 64, 40, 40] (m_u = 6400, REGS=0) in F32 and F16, plus odd-hw [8, 64, 17, 17] (m_u = 2312, REGS=0).
    let cases = [
        ([16usize, 64, 40, 40], Some(UnaryOp::Silu), false),
        ([8, 64, 17, 17], Some(UnaryOp::Relu), false),
        ([16, 64, 40, 40], Some(UnaryOp::Silu), true),
    ];
    for (shape, act, portable) in cases {
        let c = shape[1];
        let x_cpu = randn(&shape, 71).mul_scalar(2.5).unwrap().add_scalar(-0.3).unwrap();
        let g_cpu = randn(&shape, 72);
        let w_cpu = randn(&[c], 73).add_scalar(1.0).unwrap();
        let b_cpu = randn(&[c], 74);

        let m_f = (shape[0] * shape[2] * shape[3]) as f64;
        let x_v = x_cpu.to_vec::<f32>().unwrap();
        let g_v = g_cpu.to_vec::<f32>().unwrap();
        let w_v = w_cpu.to_vec::<f32>().unwrap();
        let b_v = b_cpu.to_vec::<f32>().unwrap();
        let hw = shape[2] * shape[3];
        let mut ref_mean = vec![0.0f32; c];
        let mut ref_inv = vec![0.0f32; c];
        let mut ref_y = vec![0.0f32; x_v.len()];
        let mut ref_dw = vec![0.0f32; c];
        let mut ref_db = vec![0.0f32; c];
        let mut ref_dx = vec![0.0f32; x_v.len()];
        for ch in 0..c {
            let mut sum = 0.0f64;
            for n in 0..shape[0] {
                let off = (n * c + ch) * hw;
                for i in 0..hw {
                    sum += x_v[off + i] as f64;
                }
            }
            let mean = sum / m_f;
            let mut sq = 0.0f64;
            for n in 0..shape[0] {
                let off = (n * c + ch) * hw;
                for i in 0..hw {
                    let d = x_v[off + i] as f64 - mean;
                    sq += d * d;
                }
            }
            let inv = 1.0 / (sq / m_f + eps).sqrt();
            ref_mean[ch] = mean as f32;
            ref_inv[ch] = inv as f32;
            let (wc, bc) = (w_v[ch] as f64, b_v[ch] as f64);
            let (mut db, mut dw) = (0.0f64, 0.0f64);
            for n in 0..shape[0] {
                let off = (n * c + ch) * hw;
                for i in 0..hw {
                    let xh = (x_v[off + i] as f64 - mean) * inv;
                    let z = xh * wc + bc;
                    let (y, da) = match act {
                        Some(UnaryOp::Silu) => {
                            let s = 1.0 / (1.0 + (-z).exp());
                            (z * s, s * (1.0 + z * (1.0 - s)))
                        }
                        Some(UnaryOp::Relu) => (z.max(0.0), if z > 0.0 { 1.0 } else { 0.0 }),
                        _ => (z, 1.0),
                    };
                    ref_y[off + i] = y as f32;
                    let dz = g_v[off + i] as f64 * da;
                    db += dz;
                    dw += dz * xh;
                }
            }
            ref_db[ch] = db as f32;
            ref_dw[ch] = dw as f32;
            let (db_n, dw_n, sc) = (db / m_f, dw / m_f, wc * inv);
            for n in 0..shape[0] {
                let off = (n * c + ch) * hw;
                for i in 0..hw {
                    let xh = (x_v[off + i] as f64 - mean) * inv;
                    let z = xh * wc + bc;
                    let da = match act {
                        Some(UnaryOp::Silu) => {
                            let s = 1.0 / (1.0 + (-z).exp());
                            s * (1.0 + z * (1.0 - s))
                        }
                        Some(UnaryOp::Relu) => {
                            if z > 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => 1.0,
                    };
                    let dz = g_v[off + i] as f64 * da;
                    ref_dx[off + i] = (sc * (dz - db_n - xh * dw_n)) as f32;
                }
            }
        }

        let _reset = ResetSubgroups(crate::kernels::norm::FORCE_NO_SUBGROUPS.replace(portable));
        let x_g = x_cpu.to_device(dev).unwrap();
        let g_g = g_cpu.to_device(dev).unwrap();
        let w_g = w_cpu.to_device(dev).unwrap();
        let b_g = b_cpu.to_device(dev).unwrap();
        let mom = 0.1f64;
        let check_running = shape == [16, 64, 40, 40];
        let (rm_g, rv_g) = if check_running {
            (
                Some(Tensor::full([c], 0.25f32).to_device(dev).unwrap()),
                Some(Tensor::full([c], 1.5f32).to_device(dev).unwrap()),
            )
        } else {
            (None, None)
        };
        let running = match (&rm_g, &rv_g) {
            (Some(rm), Some(rv)) => Some((rm, rv, mom)),
            _ => None,
        };
        let out = x_g.batch_norm_act_train(&w_g, &b_g, eps, act, running).unwrap().unwrap();
        let (dx, dw, db) = x_g
            .batch_norm_act_train_backward(&g_g, &w_g, &b_g, &out.mean, &out.invstd, act)
            .unwrap()
            .unwrap();
        drop(_reset);

        let tag = format!("streaming {shape:?} {act:?} portable={portable}");
        assert_close_t(
            &out.mean,
            &Tensor::from_vec(ref_mean.clone(), [c]).unwrap(),
            1e-4,
            1e-4,
            &format!("{tag} mean"),
        );
        assert_close_t(
            &out.invstd,
            &Tensor::from_vec(ref_inv.clone(), [c]).unwrap(),
            1e-4,
            1e-4,
            &format!("{tag} inv"),
        );
        if let (Some(rm), Some(rv)) = (&rm_g, &rv_g) {
            let ref_rm: Vec<f32> =
                ref_mean.iter().map(|&m| (0.25 * (1.0 - mom) + mom * m as f64) as f32).collect();
            let ref_rv: Vec<f32> = ref_inv
                .iter()
                .map(|&inv| {
                    let var_b = (1.0 / (inv as f64 * inv as f64) - eps).max(0.0);
                    let var_u = var_b * m_f / (m_f - 1.0);
                    (1.5 * (1.0 - mom) + mom * var_u) as f32
                })
                .collect();
            assert_close_t(rm, &Tensor::from_vec(ref_rm, [c]).unwrap(), 1e-4, 1e-4, &format!("{tag} rm"));
            assert_close_t(rv, &Tensor::from_vec(ref_rv, [c]).unwrap(), 1e-4, 1e-4, &format!("{tag} rv"));
        }
        assert_close_t(&out.y, &Tensor::from_vec(ref_y, shape).unwrap(), 1e-4, 1e-4, &format!("{tag} y"));
        assert_close_t(&dw, &Tensor::from_vec(ref_dw, [c]).unwrap(), 2e-3, 2e-3, &format!("{tag} dw"));
        assert_close_t(&db, &Tensor::from_vec(ref_db, [c]).unwrap(), 2e-3, 2e-3, &format!("{tag} db"));
        assert_close_t(&dx, &Tensor::from_vec(ref_dx, shape).unwrap(), 2e-4, 2e-4, &format!("{tag} dx"));

        if has_f16 && !portable && shape == [16, 64, 40, 40] {
            let x16 = x_g.to_dtype(DType::F16).unwrap();
            let g16 = g_g.to_dtype(DType::F16).unwrap();
            let o16 = x16.batch_norm_act_train(&w_g, &b_g, eps, act, None).unwrap().unwrap();
            let (dx16, dw16, db16) = x16
                .batch_norm_act_train_backward(&g16, &w_g, &b_g, &o16.mean, &o16.invstd, act)
                .unwrap()
                .unwrap();
            assert_close_t(&o16.mean, &out.mean, 2e-3, 2e-3, "streaming f16 mean");
            assert_close_t(&o16.invstd, &out.invstd, 2e-3, 2e-3, "streaming f16 invstd");
            assert_close_t(&o16.y.to_dtype(DType::F32).unwrap(), &out.y, 4e-3, 4e-3, "streaming f16 y");
            assert_close_t(&dw16, &dw, 5e-2, 5e-3, "streaming f16 dw");
            assert_close_t(&db16, &db, 5e-2, 5e-3, "streaming f16 db");
            assert_close_t(&dx16.to_dtype(DType::F32).unwrap(), &dx, 4e-3, 4e-3, "streaming f16 dx");
        }
    }

    // 2. High-offset stability on streaming REGS=0 (`[1, 64, 1, 16384]`, m_u = 4096): first 8,192 elements are
    // 10_000_000.0 and last 8,192 are 10_000_002.0 so every vec4 lane sees 8 copies of each value. Without anchor
    // centering, 2/9 rounds away at 1e7 and doubles variance to 2; with anchor centering, mean = 10_000_001.0 and
    // var = 1.0 in both subgroup and portable modes.
    let c = 64usize;
    let hw = 16_384usize;
    let mut data = vec![0.0f32; c * hw];
    for ch in 0..c {
        let base = 10_000_000.0f32;
        for i in 0..hw {
            data[ch * hw + i] = if i < hw / 2 { base } else { base + 2.0 };
        }
    }
    let x = Tensor::from_vec(data, [1, c, 1, hw]).unwrap().to_device(dev).unwrap();
    let w = Tensor::ones([c]).to_device(dev).unwrap();
    let b = Tensor::zeros([c]).to_device(dev).unwrap();
    let want_inv = 1.0f32 / (1.0f32 + eps as f32).sqrt();
    for portable in [false, true] {
        let _reset = ResetSubgroups(crate::kernels::norm::FORCE_NO_SUBGROUPS.replace(portable));
        let rm = Tensor::zeros([c]).to_device(dev).unwrap();
        let rv = Tensor::zeros([c]).to_device(dev).unwrap();
        let out = x.batch_norm_act_train(&w, &b, eps, None, Some((&rm, &rv, 1.0))).unwrap().unwrap();
        let mean_v = out.mean.to_vec::<f32>().unwrap();
        let inv_v = out.invstd.to_vec::<f32>().unwrap();
        let y_v = out.y.to_vec::<f32>().unwrap();
        let rm_v = rm.to_vec::<f32>().unwrap();
        let rv_v = rv.to_vec::<f32>().unwrap();
        assert!(
            (mean_v[0] - 10_000_001.0).abs() < 1e-3,
            "streaming high-offset mean (portable={portable}): {}",
            mean_v[0]
        );
        assert!(
            (inv_v[0] - want_inv).abs() < 1e-4,
            "streaming high-offset invstd (portable={portable}): {} vs {want_inv}",
            inv_v[0]
        );
        assert!(
            (y_v[0] + want_inv).abs() < 1e-4 && (y_v[hw - 1] - want_inv).abs() < 1e-4,
            "streaming high-offset y (portable={portable}): first={} last={} want=±{want_inv}",
            y_v[0],
            y_v[hw - 1]
        );
        assert!(
            (rm_v[0] - 10_000_001.0).abs() < 1e-3 && (rv_v[0] - 1.0).abs() < 1e-3,
            "streaming high-offset running stats (portable={portable}): rm={} rv={}",
            rm_v[0],
            rv_v[0]
        );
    }
}
