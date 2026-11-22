//! The ops mixed-precision training adds, through the folo-core backend: the fused BatchNorm on f16 data with f32
//! parameters and statistics, the f16 max-pooling and upsampling gradients, and folo-core's `autocast` conv and
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
    let g_up = q(&randn(&[2, 3, 18, 30], 7)).to_device(dev).unwrap();
    // SPPF's 5×5 stride-1 max pooling and the neck's 2× upsampling, f16 against f32.
    let grads = |dtype: DType| {
        let xl = x.to_dtype(dtype).unwrap().detach().requires_grad_();
        let pooled = xl.max_pool2d(5, 1, 2).unwrap();
        let up = xl.upsample_nearest2d_hw([2, 3]).unwrap();
        let loss = pooled
            .mul(&g_pool.to_dtype(dtype).unwrap())
            .unwrap()
            .sum_all()
            .unwrap()
            .add(&up.mul(&g_up.to_dtype(dtype).unwrap()).unwrap().sum_all().unwrap())
            .unwrap();
        let gr = loss.backward().unwrap();
        let gx = gr.get(&xl).unwrap().clone();
        assert_eq!(gx.dtype(), dtype);
        gx.to_dtype(DType::F32).unwrap()
    };
    // Each input gradient sums at most 25 + 6 f16 terms in f32 and rounds once.
    assert_close_t(&grads(DType::F16), &grads(DType::F32), 2e-3, 2e-3, "max_pool2d + upsample gradient");
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

/// B4 verification: vectorized `vec4` + block-shifted 0-div `bn_stats` / `bn_bwd_reduce` match between the
/// 32-wide subgroup fast path and the portable shared-memory fallback across both `hw % 4 == 0` (VEC4) and
/// odd `hw`, all activations, and large shifted means (`+500.0`, verifying zero catastrophic cancellation).
#[test]
fn batch_norm_subgroups_and_portable_fallback_agree() {
    use std::sync::atomic::Ordering;
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
            crate::kernels::norm::FORCE_NO_SUBGROUPS.store(force_portable, Ordering::Relaxed);
            let xd = x.to_dtype(dtype).unwrap();
            let gd = g.to_dtype(dtype).unwrap();
            let out = xd.batch_norm_act_train(&w, &b, 1e-3, act, None).unwrap().expect("kernels");
            let (dx, dw, db) = xd
                .batch_norm_act_train_backward(&gd, &w, &b, &out.mean, &out.invstd, act)
                .unwrap()
                .expect("kernels");
            crate::kernels::norm::FORCE_NO_SUBGROUPS.store(false, Ordering::Relaxed);
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
}
