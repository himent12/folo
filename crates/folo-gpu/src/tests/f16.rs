//! Half-precision compute through the folo-core backend: every forward op folo-core lets `F16` through, on f16
//! GPU tensors, against the f32 CPU reference on the same (f16-rounded) inputs.
//!
//! Tolerances: storage is f16 (relative rounding 2^-11 ≈ 4.9e-4 per store), the arithmetic f32. Elementwise ops,
//! softmax, pooling and copies round once: `1e-3` relative. Reductions, GEMM and convolutions accumulate in f32
//! and round once at the end: `2e-3` relative plus an absolute term for cancellation in long sums.

use folo_core::backend::UnaryOp as U;
use folo_core::{DType, Device, Error, Result, Tensor, no_grad};

use super::backend::{assert_close_t, dev, randn};

/// The input rounded through f16 (so the CPU reference sees exactly the GPU's inputs), as f32.
fn q(t: &Tensor) -> Tensor {
    t.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap()
}

/// Runs `f` on f32 CPU copies of `inputs` (rounded through f16) and on f16 GPU copies; the GPU result must be
/// f16, stay on the GPU, and match within `atol + rtol·|cpu|`.
#[track_caller]
fn check16<F>(dev: Device, inputs: &[Tensor], atol: f32, rtol: f32, what: &str, f: F)
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let _g = no_grad();
    let cpu: Vec<Tensor> =
        inputs.iter().map(|t| if t.dtype() == DType::F32 { q(t) } else { t.clone() }).collect();
    let gpu: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            let t = t.to_device(dev).unwrap();
            if t.dtype() == DType::F32 { t.to_dtype(DType::F16).unwrap() } else { t }
        })
        .collect();
    let yc = f(&cpu).unwrap_or_else(|e| panic!("{what}: cpu: {e}"));
    let yg = f(&gpu).unwrap_or_else(|e| panic!("{what}: gpu: {e}"));
    assert!(yg.device().is_gpu(), "{what}: result left the GPU");
    let want = if yc.dtype() == DType::F32 { DType::F16 } else { yc.dtype() };
    assert_eq!(yg.dtype(), want, "{what}: dtype");
    let yg = if want == DType::F16 { yg.to_dtype(DType::F32).unwrap() } else { yg };
    assert_close_t(&yg, &yc, atol, rtol, what);
}

#[test]
fn backend_reports_f16_compute() {
    let dev = dev!();
    let ctx = crate::context_for(dev).expect("registered");
    assert_eq!(folo_core::backend::computes_f16(dev), ctx.caps().shader_f16);
    assert!(!folo_core::backend::computes_f16(Device::Cpu));
}

#[test]
fn cpu_refuses_f16_compute() {
    let x = Tensor::ones([4]).to_dtype(DType::F16).unwrap();
    assert!(matches!(x.exp(), Err(Error::DTypeMismatch { .. })));
    assert!(matches!(x.add(&x), Err(Error::DTypeMismatch { .. })));
    // Storage ops keep working.
    assert_eq!(x.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(), vec![1.0; 4]);
}

/// Skips (with a note) on adapters without `shader-f16`.
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
fn unary_ops() {
    let dev = dev16!();
    // 4·k elements (vec4 path), an odd count (scalar path) and a transposed view (strided path).
    for shape in [vec![8, 64], vec![3, 37], vec![5, 7, 9]] {
        let x = randn(&shape, 1);
        for op in
            [U::Silu, U::Sigmoid, U::Exp, U::Relu, U::Tanh, U::MulScalar(0.37), U::AddScalar(-1.5), U::Neg]
        {
            check16(dev, std::slice::from_ref(&x), 1e-4, 1e-3, &format!("{op:?} {shape:?}"), |t| {
                t[0].unary(op)
            });
        }
        check16(dev, std::slice::from_ref(&x), 1e-4, 1e-3, "transposed silu", |t| {
            t[0].transpose(0, 1)?.silu()
        });
    }
}

#[test]
fn binary_ops_broadcast_and_compare() {
    let dev = dev16!();
    let a = randn(&[2, 6, 5, 7], 2);
    let b = randn(&[1, 6, 1, 1], 3);
    let c = randn(&[2, 6, 5, 7], 4);
    let ins = [a, b, c];
    check16(dev, &ins, 1e-4, 1e-3, "add same shape", |t| t[0].add(&t[2]));
    check16(dev, &ins, 1e-4, 1e-3, "mul broadcast", |t| t[0].mul(&t[1]));
    check16(dev, &ins, 1e-4, 1e-3, "sub transposed", |t| t[0].transpose(2, 3)?.sub(&t[2].transpose(2, 3)?));
    check16(dev, &ins, 0.0, 0.0, "gt mask", |t| t[0].gt(&t[2]));
    check16(dev, &ins, 1e-4, 1e-3, "where f32 mask", |t| {
        let m = t[0].gt(&t[2])?.to_dtype(DType::F32)?;
        m.where_cond(&t[0], &t[2])
    });
    check16(dev, &ins, 1e-4, 1e-3, "scalar compare", |t| t[0].gt_scalar(0.25)?.mul(&t[0]));
    // Vec4 broadcast binary + where (f16 storage, u8 & f32 conditions, channel & scalar broadcasts).
    let x4 = randn(&[2, 5, 8, 8], 40);
    let y4 = randn(&[2, 5, 8, 8], 41);
    let b4 = randn(&[1, 5, 1, 1], 42);
    let s4 = randn(&[1], 43);
    let ins4 = [x4, y4, b4, s4];
    check16(dev, &ins4, 1e-4, 1e-3, "f16 vec4 nchw bcast add/mul", |t| t[0].add(&t[2])?.mul(&t[3]));
    check16(dev, &ins4, 1e-4, 1e-3, "f16 vec4 where u8 dense", |t| {
        t[0].gt(&t[1])?.to_dtype(DType::U8)?.where_cond(&t[0], &t[1])
    });
    check16(dev, &ins4, 1e-4, 1e-3, "f16 vec4 where u8 bcast cond+rhs", |t| {
        t[2].gt_scalar(0.0)?.to_dtype(DType::U8)?.where_cond(&t[0], &t[2])
    });
    check16(dev, &ins4, 1e-4, 1e-3, "f16 vec4 where f32 bcast lhs", |t| {
        t[0].gt_scalar(0.0)?.to_dtype(DType::F32)?.where_cond(&t[2], &t[1])
    });
}

#[test]
fn mixed_dtypes_are_refused() {
    let dev = dev16!();
    let a = Tensor::ones([4]).to_device(dev).unwrap();
    let h = a.to_dtype(DType::F16).unwrap();
    assert!(matches!(a.add(&h), Err(Error::DTypeMismatch { .. })));
    assert!(matches!(h.matmul(&a.reshape([4, 1]).unwrap()), Err(Error::DTypeMismatch { .. })));
}

#[test]
fn reductions_accumulate_in_f32() {
    let dev = dev16!();
    let x = randn(&[3, 50, 41], 5);
    check16(dev, std::slice::from_ref(&x), 1e-3, 2e-3, "sum last", |t| t[0].sum(2, false));
    check16(dev, std::slice::from_ref(&x), 1e-3, 2e-3, "sum middle", |t| t[0].sum(1, true));
    check16(dev, std::slice::from_ref(&x), 0.0, 1e-3, "max middle", |t| t[0].max(1, false));
    check16(dev, std::slice::from_ref(&x), 0.0, 1e-3, "min first", |t| t[0].min(0, false));
    check16(dev, std::slice::from_ref(&x), 1e-3, 2e-3, "mean", |t| t[0].mean([0, 2], false));
    check16(dev, std::slice::from_ref(&x), 0.0, 0.0, "argmax", |t| t[0].argmax(2, false));
    // A long reduction over few outputs takes the split path (f32 partials). 1 + 2^-11 + ... summed in f16
    // would stall at 2048; in f32 the sum of 20000 ones is exact.
    let ones = Tensor::ones([2, 20000]);
    check16(dev, std::slice::from_ref(&ones), 0.0, 1e-3, "long sum", |t| t[0].sum(1, false));
    // A sum over separated axes (two passes) whose partials overflow f16 but cancel: f32 partials give 0.
    let big = Tensor::from_vec(
        vec![40000f32, 40000., 40000., 40000., -40000., -40000., -40000., -40000.],
        [2, 2, 2],
    )
    .unwrap();
    check16(dev, std::slice::from_ref(&big), 0.0, 0.0, "separated-axis sum", |t| t[0].sum([0, 2], false));
    // The mean of 100k ones: an f16 sum would overflow (65504) before the division.
    let many = Tensor::ones([100_000]);
    check16(dev, std::slice::from_ref(&many), 0.0, 1e-3, "mean of many", |t| t[0].mean_all());
    check16(dev, std::slice::from_ref(&many), 0.0, 1e-3, "var of many", |t| {
        t[0].add_scalar(0.0)?.var(0, false, false)
    });
}

#[test]
fn softmax_rows_and_columns() {
    let dev = dev16!();
    let x = randn(&[4, 3, 33], 6).mul_scalar(4.0).unwrap();
    check16(dev, std::slice::from_ref(&x), 1e-4, 1e-3, "softmax last", |t| t[0].softmax(-1));
    check16(dev, std::slice::from_ref(&x), 1e-4, 1e-3, "softmax middle", |t| t[0].softmax(1));
    check16(dev, std::slice::from_ref(&x), 2e-3, 1e-3, "log_softmax", |t| t[0].log_softmax(-1));
    check16(dev, std::slice::from_ref(&x), 2e-3, 1e-3, "log_softmax middle", |t| t[0].log_softmax(1));
    // Exercise uncached rows (n > 1024) and general columns (d > 32) in f16 storage too.
    let long_row = randn(&[2, 1057], 16).mul_scalar(4.0).unwrap();
    check16(dev, std::slice::from_ref(&long_row), 1e-4, 1e-3, "softmax long row", |t| t[0].softmax(-1));
    check16(dev, std::slice::from_ref(&long_row), 2e-3, 1e-3, "log_softmax long row", |t| {
        t[0].log_softmax(-1)
    });
    let tall_col = randn(&[2, 48, 17], 17).mul_scalar(4.0).unwrap();
    check16(dev, std::slice::from_ref(&tall_col), 1e-4, 1e-3, "softmax tall col", |t| t[0].softmax(1));
    check16(dev, std::slice::from_ref(&tall_col), 2e-3, 1e-3, "log_softmax tall col", |t| {
        t[0].log_softmax(1)
    });
}

#[test]
fn attention_shaped_matmul() {
    let dev = dev16!();
    // C2PSA's shapes: q, k, v slices of one qkv tensor, scores = qᵀ·k, out = v·attnᵀ.
    let (b, heads, kd, hd, n) = (2, 2, 16, 32, 100);
    let qkv = randn(&[b, heads, 2 * kd + hd, n], 7);
    check16(dev, std::slice::from_ref(&qkv), 2e-3, 2e-3, "attention", |t| {
        let q = t[0].narrow(2, 0, kd)?;
        let k = t[0].narrow(2, kd, kd)?;
        let v = t[0].narrow(2, 2 * kd, hd)?;
        let attn = q.transpose(2, 3)?.matmul(&k)?.mul_scalar((kd as f32).powf(-0.5))?.softmax(-1)?;
        v.matmul(&attn.transpose(2, 3)?)
    });
    let a = randn(&[3, 17, 29], 8);
    let w = randn(&[29, 11], 9);
    check16(dev, &[a, w], 2e-3, 2e-3, "shared rhs", |t| t[0].matmul(&t[1]));
}

#[test]
fn convolutions_with_fused_epilogues() {
    let dev = dev16!();
    let x = randn(&[2, 16, 19, 23], 10);
    let cases: [(usize, usize, usize, usize, usize); 5] = [
        // (c_out, k, stride, groups, padding)
        (24, 3, 1, 1, 1),
        (32, 3, 2, 1, 1),
        (8, 1, 1, 1, 0),
        (16, 3, 1, 16, 1),
        (32, 3, 1, 16, 1),
    ];
    for (o, k, s, g, p) in cases {
        let w = randn(&[o, 16 / g, k, k], 11).mul_scalar(0.2).unwrap();
        let bias = randn(&[o], 12);
        let ins = [x.clone(), w, bias];
        let params = folo_core::Conv2dParams { stride: [s; 2], padding: [p; 2], dilation: [1; 2], groups: g };
        check16(dev, &ins, 3e-3, 2e-3, &format!("conv2d_act silu {o} k{k} s{s} g{g}"), |t| {
            t[0].conv2d_act(&t[1], Some(&t[2]), &params, Some(U::Silu))
        });
        check16(dev, &ins, 3e-3, 2e-3, &format!("conv2d {o} k{k} s{s} g{g}"), |t| {
            t[0].conv2d_with(&t[1], Some(&t[2]), &params)
        });
    }
    // The concatenated input of a C3k2 cv2 (the GPU writes the concatenation, then convolves).
    let a = randn(&[2, 8, 10, 10], 13);
    let b = randn(&[2, 5, 10, 10], 14);
    let w = randn(&[6, 13, 1, 1], 15);
    let params = folo_core::Conv2dParams::default();
    check16(dev, &[a, b, w], 3e-3, 2e-3, "conv2d_act_cat", |t| {
        Tensor::conv2d_act_cat(&[&t[0], &t[1]], &t[2], None, &params, Some(U::Silu))
    });
}

#[test]
fn pooling_upsample_and_concat() {
    let dev = dev16!();
    let x = randn(&[2, 5, 13, 11], 16);
    check16(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool2d k5 s1 p2", |t| t[0].max_pool2d(5, 1, 2));
    check16(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool2d k2 s2", |t| t[0].max_pool2d(2, 2, 0));
    check16(dev, std::slice::from_ref(&x), 1e-4, 1e-3, "avg_pool2d", |t| t[0].avg_pool2d(3, 2, 1));
    check16(dev, std::slice::from_ref(&x), 1e-4, 1e-3, "adaptive_avg_pool2d", |t| {
        t[0].adaptive_avg_pool2d([3, 2])
    });
    check16(dev, std::slice::from_ref(&x), 0.0, 0.0, "upsample 2x", |t| t[0].upsample_nearest2d(2));
    // Odd sizes along every concatenated dim: f16 elements share words with their neighbours.
    let y = randn(&[2, 3, 13, 11], 17);
    check16(dev, &[x.clone(), y], 0.0, 0.0, "cat channels", |t| Tensor::cat(&[&t[0], &t[1]], 1));
    let z = randn(&[2, 5, 13, 3], 18);
    check16(dev, &[x, z], 0.0, 0.0, "cat last", |t| Tensor::cat(&[&t[0], &t[1]], 3));
}

/// Word-aligned sub-word copies (even f16 runs, u8 runs of four) move whole words; odd ones keep the per-element
/// path. Both must give the same bytes as the CPU.
#[test]
fn word_aligned_concat_and_slices() {
    let dev = dev!();
    for dtype in [DType::F16, DType::U8] {
        let cpu = |shape: &[usize], seed| {
            randn(shape, seed).mul_scalar(20.0).unwrap().abs().unwrap().to_dtype(dtype).unwrap()
        };
        // Channels of 8 × 6 = 48 elements: every block and offset is word-aligned.
        let (a, b) = (cpu(&[2, 4, 8, 6], 30), cpu(&[2, 3, 8, 6], 31));
        let (ga, gb) = (a.to_device(dev).unwrap(), b.to_device(dev).unwrap());
        let want = Tensor::cat(&[&a, &b], 1).unwrap();
        let got = Tensor::cat(&[&ga, &gb], 1).unwrap();
        assert_eq!(
            got.to_cpu_storage().unwrap().as_bytes(),
            want.to_cpu_storage().unwrap().as_bytes(),
            "{dtype} cat"
        );
        // A channel slice and a strided (every other row) view, made dense.
        for (what, f) in [
            (
                "narrow",
                (|t: &Tensor| t.narrow(1, 2, 2).unwrap().contiguous().unwrap()) as fn(&Tensor) -> Tensor,
            ),
            ("rows", |t: &Tensor| {
                t.reshape([2, 4, 4, 12]).unwrap().narrow(3, 4, 8).unwrap().contiguous().unwrap()
            }),
            ("odd", |t: &Tensor| t.narrow(3, 1, 5).unwrap().contiguous().unwrap()),
        ] {
            let (w, g) = (f(&a), f(&ga));
            assert_eq!(
                g.to_cpu_storage().unwrap().as_bytes(),
                w.to_cpu_storage().unwrap().as_bytes(),
                "{dtype} {what}"
            );
        }
    }
}

#[test]
fn model_conversion_in_place() {
    let dev = dev16!();
    let x = randn(&[3, 5], 19).to_device(dev).unwrap();
    let view = x.narrow(1, 1, 3).unwrap();
    let id = x.id();
    x.to_dtype_(DType::F16).unwrap();
    assert_eq!((x.id(), x.dtype(), view.dtype()), (id, DType::F16, DType::F16));
    let want = q(&randn(&[3, 5], 19)).narrow(1, 1, 3).unwrap();
    assert_close_t(&view.to_dtype(DType::F32).unwrap(), &want, 0.0, 0.0, "view after to_dtype_");
    // Only leaves convert.
    let y = randn(&[2], 20).requires_grad_().mul_scalar(2.0).unwrap();
    assert!(y.to_dtype_(DType::F16).is_err());
}

/// The f16 gradients mixed-precision training uses run (conv and average pooling here, tested for accuracy in
/// `amp.rs` and `coop.rs`); the ones it does not use still refuse f16 (adaptive average pooling's).
#[test]
fn gradients_in_f16() {
    let dev = dev16!();
    let x = randn(&[1, 2, 6, 6], 21).to_device(dev).unwrap().to_dtype(DType::F16).unwrap().requires_grad_();
    let w = randn(&[3, 2, 3, 3], 22).to_device(dev).unwrap().to_dtype(DType::F16).unwrap().requires_grad_();
    let y = x.conv2d(&w, None, 1, 1, 1, 1).unwrap().sum_all().unwrap();
    let g = y.backward().unwrap();
    assert_eq!((g.get(&x).unwrap().dtype(), g.get(&w).unwrap().dtype()), (DType::F16, DType::F16));
    let y = x.avg_pool2d(2, 2, 0).unwrap().sum_all().unwrap();
    let g = y.backward().unwrap();
    assert_eq!(g.get(&x).unwrap().dtype(), DType::F16);
    let y = x.adaptive_avg_pool2d([3, 3]).unwrap().sum_all().unwrap();
    let Err(e) = y.backward() else { panic!("f16 adaptive average-pooling backward is refused") };
    assert!(matches!(e, Error::UnsupportedDType { dtype: DType::F16, .. }), "{e}");
}

#[test]
fn adapter_without_f16_falls_back_to_f32_errors() {
    let _ = dev!();
    let opts = crate::GpuOptions { enable_f16: false, ..crate::GpuOptions::from_env() };
    let ctx = match crate::GpuContext::with_options(&opts) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping: {e}");
            return;
        }
    };
    assert!(!ctx.caps().shader_f16);
    // A device of its own (ordinal 7 is used by no other test).
    let d = Device::Gpu(7);
    crate::register(ctx, d).unwrap();
    assert!(!folo_core::backend::computes_f16(d));
    let x = Tensor::ones([8]).to_device(d).unwrap().to_dtype(DType::F16).unwrap();
    assert!(matches!(x.silu(), Err(Error::DTypeMismatch { .. })));
    // Storage, casts and copies need no shader-f16.
    assert_eq!(x.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(), vec![1.0; 8]);
}
