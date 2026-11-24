//! Max, average and adaptive pooling and nearest upsampling.

use folo_core::Tensor;

use super::fixtures::{assert_close_t, check, dev, randn};

#[test]
fn pooling_and_upsampling() {
    let dev = dev!();
    let x = randn(&[2, 3, 11, 13], 70);
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool k3 s2 p1", |v| v[0].max_pool2d(3, 2, 1));
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool k5 s1 p2 (SPPF)", |v| v[0].max_pool2d(5, 1, 2));
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool k2 s2", |v| v[0].max_pool2d(2, 2, 0));
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "avg_pool k2 s2", |v| v[0].avg_pool2d(2, 2, 0));
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "avg_pool k3 s1 p1", |v| v[0].avg_pool2d(3, 1, 1));
    for hw in [[1usize, 1], [3, 5], [11, 13], [20, 7]] {
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, &format!("adaptive {hw:?}"), move |v| {
            v[0].adaptive_avg_pool2d(hw)
        });
    }
    // Window arithmetic that would wrap u32 is rejected rather than computed wrongly (65536 · 65537 > 2^32).
    let tall = Tensor::ones([1, 1, 65536, 1]).to_device(dev).unwrap();
    let e = tall.adaptive_avg_pool2d([65536, 1]).unwrap_err().to_string();
    assert!(e.contains("overflows"), "{e}");
    assert!(tall.adaptive_avg_pool2d([1, 1]).is_ok());
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "upsample 2x", |v| v[0].upsample_nearest2d(2));
    check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "upsample 2x3", |v| v[0].upsample_nearest2d_hw([2, 3]));
    // Huge gradients: each window's share is scaled before it is summed, so nothing overflows.
    let small = randn(&[1, 1, 3, 3], 71);
    let (cs, gs) = (small.detach().requires_grad_(), small.to_device(dev).unwrap().requires_grad_());
    let big = Tensor::full([1, 1, 3, 3], 1e38);
    let gc = cs.avg_pool2d(3, 1, 1).unwrap().backward_with(&big).unwrap();
    let gg = gs.avg_pool2d(3, 1, 1).unwrap().backward_with(&big.to_device(dev).unwrap()).unwrap();
    assert_close_t(gg.get(&gs).unwrap(), gc.get(&cs).unwrap(), 0.0, 1e-6, "avg_pool backward of 1e38");
    // Ties: constant planes pick the first element of each window.
    let flat = Tensor::ones([1, 1, 4, 4]);
    check(dev, &[flat], 0.0, 0.0, "max_pool ties", |v| v[0].max_pool2d(2, 1, 0));
    // A transposed (non-dense) input.
    check(dev, std::slice::from_ref(&x), 0.0, 0.0, "max_pool of transposed", |v| {
        v[0].transpose(2, 3)?.max_pool2d(3, 2, 1)
    });
}

#[test]
fn max_pool2d_values_inference_parity_and_edges() {
    use folo_core::backend::{Backend, Pool2dParams};
    use folo_core::{DType, Layout, no_grad};

    let dev = dev!();
    let ctx = crate::context_for(dev).unwrap();
    let x = randn(&[2, 3, 11, 13], 72);
    let _g = no_grad();
    let dtypes: Vec<DType> =
        if ctx.caps().shader_f16 { vec![DType::F32, DType::F16] } else { vec![DType::F32] };

    let shapes: &[(&str, Pool2dParams)] = &[
        ("sppf k5 s1 p2", Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] }),
        ("k3 s2 p1", Pool2dParams { kernel: [3, 3], stride: [2, 2], padding: [1, 1] }),
        ("k2 s2 p0", Pool2dParams { kernel: [2, 2], stride: [2, 2], padding: [0, 0] }),
        ("asym k3x2 s1x2 p1x0", Pool2dParams { kernel: [3, 2], stride: [1, 2], padding: [1, 0] }),
    ];

    let run_pair =
        |src: &Tensor, dt: DType, f: &dyn Fn(&Tensor) -> folo_core::Result<Tensor>, label: &str| {
            let rounded = src.to_dtype(dt).unwrap();
            let cpu_in =
                if dt == DType::F16 { rounded.to_dtype(DType::F32).unwrap() } else { rounded.clone() };
            let want = f(&cpu_in).unwrap().to_dtype(dt).unwrap();
            let got = f(&rounded.to_device(dev).unwrap()).unwrap();
            assert_eq!(got.dtype(), dt, "{label}: dtype");
            assert_close_t(&got, &want, 0.0, 0.0, label);
        };

    for &dt in &dtypes {
        for (label, p) in shapes {
            run_pair(&x, dt, &|t| t.max_pool2d_with(p), &format!("max_pool2d_values {dt:?} {label}"));
        }
        // Transposed (non-contiguous) input.
        run_pair(
            &x,
            dt,
            &|t| t.transpose(2, 3)?.max_pool2d(5, 1, 2),
            &format!("max_pool2d_values {dt:?} transposed"),
        );

        // Heavily padded 1x1 boundary plane (padding must count as -inf and never beat negative values).
        let one = Tensor::from_slice(&[-42.0f32], [1, 1, 1, 1]).unwrap();
        run_pair(&one, dt, &|t| t.max_pool2d(5, 1, 2), &format!("max_pool2d_values {dt:?} 1x1 k5 p2"));

        // Empty batch.
        let empty = Tensor::zeros([0, 3, 8, 8]).to_dtype(dt).unwrap();
        let got = empty.to_device(dev).unwrap().max_pool2d(5, 1, 2).unwrap();
        assert_eq!(got.dims(), [0, 3, 8, 8]);
        assert_eq!(got.dtype(), dt);

        // NaN, ±inf, and signed-zero semantics (NaN must win over finite values and +inf; all -inf stays -inf).
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let ninf = f32::NEG_INFINITY;
        // 4 planes of 2x2 pooled with k=2, s=2, p=0 -> 1 output per plane.
        let special = Tensor::from_slice(
            &[
                nan, 1.0, 2.0, inf, // NaN first beats +inf
                1.0, inf, nan, 2.0, // NaN after +inf still wins
                ninf, ninf, ninf, ninf, // all -inf stays -inf
                -0.0, -inf, -0.0, ninf, // -0.0 beats -inf
            ],
            [1, 4, 2, 2],
        )
        .unwrap();
        run_pair(&special, dt, &|t| t.max_pool2d(2, 2, 0), &format!("max_pool2d_values {dt:?} special"));
        // Also check NaN propagation across overlapping 3x3 p=1 windows.
        let mut grid = vec![-1.0f32; 9];
        grid[4] = nan;
        let g_t = Tensor::from_slice(&grid, [1, 1, 3, 3]).unwrap();
        run_pair(&g_t, dt, &|t| t.max_pool2d(3, 1, 1), &format!("max_pool2d_values {dt:?} center nan 3x3"));
    }

    // Verify `GpuBackend::max_pool2d_values` uses the value-only `max_pool2d` kernel and skips allocating the
    // 8-byte-per-element I64 index buffer on an SPPF-shaped tensor.
    let fresh = crate::GpuContext::new().unwrap();
    let backend = crate::GpuBackend::new(fresh.clone(), dev);
    let shape = folo_core::Shape::new(&[1, 512, 20, 20]); // 204,800 elements -> 800 KiB f32 vs +1,600 KiB i64
    let layout = Layout::contiguous(shape.clone());
    let src = backend.full(dev, DType::F32, shape.numel(), 1.0).unwrap();
    let sppf = Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] };
    let before = fresh.pool_stats().allocated_bytes;
    fresh.trace_begin();
    let out = backend.max_pool2d_values(folo_core::backend::View::new(&src, &layout), &sppf).unwrap();
    let trace = fresh.trace_end();
    let after_values = fresh.pool_stats().allocated_bytes;
    assert_eq!(trace.iter().map(|op| op.name).collect::<Vec<_>>(), vec!["max_pool2d"]);
    let values_delta = after_values.saturating_sub(before);
    // Only the f32 output buffer (~800 KiB, rounded to pool size class) should be allocated — far less than
    // adding a 1,600 KiB I64 index buffer.
    let i64_bytes = (shape.numel() * 8) as u64;
    assert!(
        values_delta < i64_bytes,
        "max_pool2d_values allocated {values_delta} bytes (expected < i64 index size {i64_bytes})"
    );
    drop(out);
    // Contrast with `max_pool2d`, which must allocate the I64 index buffer too.
    let (v, idx) = backend.max_pool2d(folo_core::backend::View::new(&src, &layout), &sppf).unwrap();
    let after_both = fresh.pool_stats().allocated_bytes;
    assert!(
        after_both - after_values >= i64_bytes,
        "max_pool2d should allocate at least {i64_bytes} bytes for I64 indices"
    );
    drop((v, idx));
}
