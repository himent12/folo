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
        // Transposed (non-contiguous) and narrowed (non-zero offset) views.
        run_pair(
            &x,
            dt,
            &|t| t.transpose(2, 3)?.max_pool2d(5, 1, 2),
            &format!("max_pool2d_values {dt:?} transposed"),
        );
        run_pair(
            &x,
            dt,
            &|t| t.narrow(1, 1, 2)?.max_pool2d(5, 1, 2),
            &format!("max_pool2d_values {dt:?} narrowed"),
        );

        // Heavily padded 1x1 boundary plane (padding must count as -inf and never beat negative values).
        let one = Tensor::from_slice(&[-42.0f32], [1, 1, 1, 1]).unwrap();
        run_pair(&one, dt, &|t| t.max_pool2d(5, 1, 2), &format!("max_pool2d_values {dt:?} 1x1 k5 p2"));

        // Empty batch and empty channel.
        for dims in [[0, 3, 8, 8], [2, 0, 8, 8]] {
            let empty = Tensor::zeros(dims).to_dtype(dt).unwrap();
            let got = empty.to_device(dev).unwrap().max_pool2d(5, 1, 2).unwrap();
            assert_eq!(got.dims(), dims);
            assert_eq!(got.dtype(), dt);
        }

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
        // Bit-exact negative zero when `-0.0` is the strict maximum over `-inf`.
        let neg_zero = Tensor::from_slice(&[-0.0, -inf, -0.0, ninf], [1, 1, 2, 2]).unwrap();
        let zr = neg_zero.to_dtype(dt).unwrap();
        let zc = if dt == DType::F16 { zr.to_dtype(DType::F32).unwrap() } else { zr.clone() };
        let zw = zc.max_pool2d(2, 2, 0).unwrap().to_dtype(dt).unwrap();
        let zg = zr.to_device(dev).unwrap().max_pool2d(2, 2, 0).unwrap();
        super::fixtures::assert_same(&zg, &zw, &format!("max_pool2d_values {dt:?} neg zero bits"));
        // Signed-zero ties evaluate to zero (GPU FMNMX may canonicalize -0.0 vs +0.0 ties to +0.0).
        let zero_ties =
            Tensor::from_slice(&[-0.0, 0.0, ninf, ninf, 0.0, -0.0, ninf, ninf], [1, 2, 2, 2]).unwrap();
        run_pair(&zero_ties, dt, &|t| t.max_pool2d(2, 2, 0), &format!("max_pool2d_values {dt:?} zero ties"));
        // Also check NaN propagation across overlapping 3x3 p=1 windows.
        let mut grid = vec![-1.0f32; 9];
        grid[4] = nan;
        let g_t = Tensor::from_slice(&grid, [1, 1, 3, 3]).unwrap();
        run_pair(&g_t, dt, &|t| t.max_pool2d(3, 1, 1), &format!("max_pool2d_values {dt:?} center nan 3x3"));
    }

    // Fallback when window exceeds `Window2d::check_limits()` (> 65,536): uses `pool_idx` and matches CPU.
    let small = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [1, 1, 2, 2]).unwrap();
    let huge_stride = Pool2dParams { kernel: [1, 1], stride: [65_537, 1], padding: [0, 0] };
    let want_fb = small.max_pool2d_with(&huge_stride).unwrap();
    let got_fb = small.to_device(dev).unwrap().max_pool2d_with(&huge_stride).unwrap();
    assert_close_t(&got_fb, &want_fb, 0.0, 0.0, "max_pool2d_values fallback huge stride");

    // Empty-batch plane with huge spatial dim (> i32::MAX) succeeds without error.
    let dummy = ctx.alloc(16).unwrap();
    let w1 = crate::kernels::pooling::Window2d::square(1, 1, 0);
    assert_eq!(
        crate::kernels::pooling::max_pool2d(
            &ctx,
            dummy.raw(),
            dummy.raw(),
            0,
            i32::MAX as usize + 1,
            1,
            &w1,
            crate::DType::F32
        )
        .unwrap(),
        (i32::MAX as usize + 1, 1)
    );

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

#[test]
fn upsample_2x_vec4_and_native_f16_avg_pool2d() {
    use folo_core::backend::{Backend, Pool2dParams, View};
    use folo_core::{DType, Layout, Shape};

    let dev = dev!();
    let ctx = crate::context_for(dev).expect("registered");
    let has_f16 = ctx.caps().shader_f16;

    // 1. 2x2 vec4 upsample across even widths (fast path) and odd widths (generic fallback), F32 + F16.
    for shape in [[1usize, 1, 1, 2], [2, 8, 20, 20], [1, 4, 7, 14], [2, 3, 5, 7]] {
        let x = randn(&shape, 80 + shape[3] as u64);
        assert_close_t(
            &x.to_device(dev).unwrap().upsample_nearest2d(2).unwrap(),
            &x.upsample_nearest2d(2).unwrap(),
            0.0,
            0.0,
            &format!("upsample 2x f32 fwd exact {shape:?}"),
        );
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, &format!("upsample 2x f32 {shape:?}"), |v| {
            v[0].upsample_nearest2d(2)
        });
        if has_f16 {
            let xf = x.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
            let want = xf.upsample_nearest2d(2).unwrap().to_dtype(DType::F16).unwrap();
            let got = x.to_device(dev).unwrap().to_dtype(DType::F16).unwrap().upsample_nearest2d(2).unwrap();
            assert_eq!(got.dtype(), DType::F16);
            assert_close_t(&got, &want, 0.0, 0.0, &format!("upsample 2x f16 {shape:?}"));
        }
    }

    // 2. Native f16 avg_pool2d forward + backward accuracy vs CPU f32 on f16-rounded inputs.
    if has_f16 {
        let x = randn(&[2, 4, 11, 13], 83).to_dtype(DType::F16).unwrap();
        let x_cpu = x.to_dtype(DType::F32).unwrap().detach().requires_grad_();
        let y_cpu = x_cpu.avg_pool2d(3, 2, 1).unwrap();
        let gy = randn(y_cpu.dims(), 84).to_dtype(DType::F16).unwrap();
        let gx_cpu =
            y_cpu.backward_with(&gy.to_dtype(DType::F32).unwrap()).unwrap().get(&x_cpu).unwrap().clone();

        let x_gpu = x.to_device(dev).unwrap().detach().requires_grad_();
        let y_gpu = x_gpu.avg_pool2d(3, 2, 1).unwrap();
        assert_eq!(y_gpu.dtype(), DType::F16);
        assert_close_t(&y_gpu.to_dtype(DType::F32).unwrap(), &y_cpu, 2e-3, 2e-3, "native f16 avg_pool2d fwd");

        let gx_gpu = y_gpu.backward_with(&gy.to_device(dev).unwrap()).unwrap().get(&x_gpu).unwrap().clone();
        assert_eq!(gx_gpu.dtype(), DType::F16);
        assert_close_t(
            &gx_gpu.to_dtype(DType::F32).unwrap(),
            &gx_cpu,
            2e-3,
            2e-3,
            "native f16 avg_pool2d bwd",
        );
    }

    // 3. Isolated trace check on a fresh context: even 2x2 -> `upsample_2x_vec4`, odd -> `upsample`,
    //    and native f16 `avg_pool2d` fwd + bwd each dispatch a single `pool` kernel with no f32 cast detour.
    let fresh = crate::GpuContext::new().unwrap();
    let backend = crate::GpuBackend::new(fresh.clone(), dev);

    let s_even = Shape::new(&[1, 4, 10, 12]);
    let b_even = backend.full(dev, DType::F32, s_even.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let _ = backend.upsample_nearest2d(View::new(&b_even, &Layout::contiguous(s_even)), [2, 2]).unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["upsample_2x_vec4"]);

    let s_odd = Shape::new(&[1, 4, 10, 11]);
    let b_odd = backend.full(dev, DType::F32, s_odd.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let _ = backend.upsample_nearest2d(View::new(&b_odd, &Layout::contiguous(s_odd)), [2, 2]).unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["upsample"]);

    if fresh.caps().shader_f16 {
        let s_in = Shape::new(&[1, 4, 11, 13]);
        let p = Pool2dParams { kernel: [3, 3], stride: [2, 2], padding: [1, 1] };
        let b_in = backend.full(dev, DType::F16, s_in.numel(), 1.0).unwrap();
        fresh.trace_begin();
        let b_out = backend.avg_pool2d(View::new(&b_in, &Layout::contiguous(s_in.clone())), &p).unwrap();
        assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["pool"]);

        let s_out = Shape::new(&[1, 4, 6, 7]);
        fresh.trace_begin();
        let _ =
            backend.avg_pool2d_backward(View::new(&b_out, &Layout::contiguous(s_out)), &s_in, &p).unwrap();
        assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["pool"]);
    }
}

#[test]
fn tiled_max_pool_vec4_2x2_and_gap_fast_paths() {
    use folo_core::backend::{Backend, Pool2dParams, View};
    use folo_core::{DType, Layout, Shape};

    let dev = dev!();
    let ctx = crate::context_for(dev).expect("registered");
    let has_f16 = ctx.caps().shader_f16;

    // 1. Multi-tile (crossing 16x16 boundaries), partial-tile, and 1x1 stride-1 max_pool2d fwd + bwd
    //    (tiled path for k=2..=7) including exact ties, NaN first-winner tie-breaking, and asymmetric windows.
    for &(shape, p) in &[
        ([2usize, 3, 20, 20], Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] }),
        ([1, 4, 35, 37], Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] }),
        ([2, 2, 17, 19], Pool2dParams { kernel: [3, 3], stride: [1, 1], padding: [1, 1] }),
        ([1, 2, 18, 21], Pool2dParams { kernel: [7, 3], stride: [1, 1], padding: [3, 1] }),
        ([1, 2, 1, 1], Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] }),
        ([1, 2, 17, 18], Pool2dParams { kernel: [2, 2], stride: [1, 1], padding: [1, 1] }), // 2*p == k
    ] {
        let x = randn(&shape, 90 + shape[2] as u64);
        check(
            dev,
            std::slice::from_ref(&x),
            0.0,
            0.0,
            &format!("tiled max_pool {shape:?} {p:?}"),
            move |v| v[0].max_pool2d_with(&p),
        );
    }

    // Multi-tile F16 tiled stride-1 max_pool2d fwd + bwd parity (crossing 16x16 tile edges).
    if has_f16 {
        let sppf = Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] };
        let xf = randn(&[2, 3, 20, 20], 95).to_dtype(DType::F16).unwrap();
        let x_cpu = xf.to_dtype(DType::F32).unwrap().detach().requires_grad_();
        let y_cpu = x_cpu.max_pool2d_with(&sppf).unwrap();
        let gy = randn(y_cpu.dims(), 96).to_dtype(DType::F16).unwrap();
        let gx_cpu =
            y_cpu.backward_with(&gy.to_dtype(DType::F32).unwrap()).unwrap().get(&x_cpu).unwrap().clone();
        let x_gpu = xf.to_device(dev).unwrap().detach().requires_grad_();
        let y_gpu = x_gpu.max_pool2d_with(&sppf).unwrap();
        assert_eq!(y_gpu.dtype(), DType::F16);
        assert_close_t(&y_gpu.to_dtype(DType::F32).unwrap(), &y_cpu, 0.0, 0.0, "tiled f16 max_pool fwd");
        let gx_gpu = y_gpu.backward_with(&gy.to_device(dev).unwrap()).unwrap().get(&x_gpu).unwrap().clone();
        assert_eq!(gx_gpu.dtype(), DType::F16);
        assert_close_t(&gx_gpu.to_dtype(DType::F32).unwrap(), &gx_cpu, 2e-3, 2e-3, "tiled f16 max_pool bwd");
    }

    // Bitwise signed-zero preservation on the tiled stride-1 value-only path:
    // -0.0 surrounded by -inf stays -0.0, and mixed (-0.0, +0.0) / (+0.0, -0.0) ties retain the first operand.
    {
        let _g = folo_core::no_grad();
        let ninf = f32::NEG_INFINITY;
        let z_ninf =
            Tensor::from_slice(&[ninf, ninf, ninf, ninf, -0.0, ninf, ninf, ninf, ninf], [1, 1, 3, 3])
                .unwrap();
        let got = z_ninf.to_device(dev).unwrap().max_pool2d(3, 1, 1).unwrap();
        let want = z_ninf.max_pool2d(3, 1, 1).unwrap();
        super::fixtures::assert_same(&got, &want, "tiled stride-1 max_pool_values -0.0 vs -inf bits");

        let z_ties =
            Tensor::from_slice(&[-0.0f32, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0], [1, 2, 2, 2]).unwrap();
        let got_t = z_ties.to_device(dev).unwrap().max_pool2d(2, 1, 0).unwrap();
        let want_t = z_ties.max_pool2d(2, 1, 0).unwrap();
        super::fixtures::assert_same(&got_t, &want_t, "tiled stride-1 max_pool_values signed-zero ties bits");
    }

    // Exact tie-breaking and first-NaN preservation across tile boundary (row 15..16, col 15..16).
    let mut tie_data = vec![1.0f32; 20 * 20];
    tie_data[15 * 20 + 17] = f32::NAN; // upper row, right col
    tie_data[16 * 20 + 15] = f32::NAN; // lower row, left col (must NOT beat row 15 NaN)
    tie_data[3 * 20 + 4] = 5.0;
    tie_data[3 * 20 + 6] = 5.0; // tie in same row (leftmost wins)
    tie_data[4 * 20 + 3] = 5.0; // tie in lower row (upper row wins)
    let tie_t = Tensor::from_slice(&tie_data, [1, 1, 20, 20]).unwrap();
    let sppf = Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] };
    // Compare both values AND argmax-routed gradients against CPU bit-for-bit.
    let c_in = tie_t.detach().requires_grad_();
    let g_in = tie_t.to_device(dev).unwrap().detach().requires_grad_();
    let c_out = c_in.max_pool2d_with(&sppf).unwrap();
    let g_out = g_in.max_pool2d_with(&sppf).unwrap();
    assert_close_t(&g_out, &c_out, 0.0, 0.0, "tiled max_pool ties+nan fwd");
    let grad_out = randn(&[1, 1, 20, 20], 99);
    let c_gx = c_out.backward_with(&grad_out).unwrap().get(&c_in).unwrap().clone();
    let g_gx = g_out.backward_with(&grad_out.to_device(dev).unwrap()).unwrap().get(&g_in).unwrap().clone();
    assert_close_t(&g_gx, &c_gx, 1e-6, 1e-6, "tiled max_pool ties+nan bwd");

    // 2. Vectorized 2x2 stride-2 avg_pool2d (fwd + bwd) and upsample_nearest2d_backward across even/odd shapes.
    for shape in [[2usize, 4, 20, 20], [1, 3, 18, 24], [2, 3, 7, 10], [1, 2, 5, 7]] {
        let x = randn(&shape, 110 + shape[3] as u64);
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, &format!("avg_pool 2x2 s2 {shape:?}"), |v| {
            v[0].avg_pool2d(2, 2, 0)
        });
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, &format!("upsample 2x2 {shape:?}"), |v| {
            v[0].upsample_nearest2d(2)
        });
        if has_f16 {
            let xf = x.to_dtype(DType::F16).unwrap();
            let x_cpu = xf.to_dtype(DType::F32).unwrap().detach().requires_grad_();
            let y_cpu = x_cpu.avg_pool2d(2, 2, 0).unwrap();
            let gy = randn(y_cpu.dims(), 120 + shape[3] as u64).to_dtype(DType::F16).unwrap();
            let gx_cpu =
                y_cpu.backward_with(&gy.to_dtype(DType::F32).unwrap()).unwrap().get(&x_cpu).unwrap().clone();
            let x_gpu = xf.to_device(dev).unwrap().detach().requires_grad_();
            let y_gpu = x_gpu.avg_pool2d(2, 2, 0).unwrap();
            assert_eq!(y_gpu.dtype(), DType::F16);
            assert_close_t(&y_gpu.to_dtype(DType::F32).unwrap(), &y_cpu, 2e-3, 2e-3, "f16 avg 2x2 fwd");
            let gx_gpu =
                y_gpu.backward_with(&gy.to_device(dev).unwrap()).unwrap().get(&x_gpu).unwrap().clone();
            assert_eq!(gx_gpu.dtype(), DType::F16);
            assert_close_t(&gx_gpu.to_dtype(DType::F32).unwrap(), &gx_cpu, 2e-3, 2e-3, "f16 avg 2x2 bwd");

            let ux_cpu = xf.to_dtype(DType::F32).unwrap().detach().requires_grad_();
            let uy_cpu = ux_cpu.upsample_nearest2d(2).unwrap();
            let ugy = randn(uy_cpu.dims(), 130 + shape[3] as u64).to_dtype(DType::F16).unwrap();
            let ugx_cpu = uy_cpu
                .backward_with(&ugy.to_dtype(DType::F32).unwrap())
                .unwrap()
                .get(&ux_cpu)
                .unwrap()
                .clone();
            let ux_gpu = xf.to_device(dev).unwrap().detach().requires_grad_();
            let uy_gpu = ux_gpu.upsample_nearest2d(2).unwrap();
            let ugx_gpu =
                uy_gpu.backward_with(&ugy.to_device(dev).unwrap()).unwrap().get(&ux_gpu).unwrap().clone();
            assert_eq!(ugx_gpu.dtype(), DType::F16);
            assert_close_t(
                &ugx_gpu.to_dtype(DType::F32).unwrap(),
                &ugx_cpu,
                5e-3,
                5e-3,
                "f16 upsample 2x2 bwd",
            );
        }
    }

    // 3. Adaptive GAP 1x1 fwd + bwd: vec4-aligned, unaligned odd area, and split-K large plane (> 8192).
    for shape in [[2usize, 8, 20, 20], [2, 3, 11, 13], [1, 4, 100, 100], [1, 2, 1, 1]] {
        let x = randn(&shape, 140 + shape[2] as u64);
        check(dev, std::slice::from_ref(&x), 1e-5, 1e-5, &format!("adaptive GAP 1x1 {shape:?}"), |v| {
            v[0].adaptive_avg_pool2d([1, 1])
        });
    }

    // 4. Trace dispatch verification on a fresh context.
    let fresh = crate::GpuContext::new().unwrap();
    let backend = crate::GpuBackend::new(fresh.clone(), dev);
    let p2 = Pool2dParams { kernel: [2, 2], stride: [2, 2], padding: [0, 0] };

    // Even 2x2 avg_pool2d fwd -> pool_2x2_s2_vec4, bwd -> avg_2x2_s2_bwd_vec4.
    let s_even = Shape::new(&[1, 4, 16, 16]);
    let b_even = backend.full(dev, DType::F32, s_even.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let b_half = backend.avg_pool2d(View::new(&b_even, &Layout::contiguous(s_even.clone())), &p2).unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["pool_2x2_s2_vec4"]);

    let s_half = Shape::new(&[1, 4, 8, 8]);
    fresh.trace_begin();
    let _ = backend
        .avg_pool2d_backward(View::new(&b_half, &Layout::contiguous(s_half.clone())), &s_even, &p2)
        .unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["avg_2x2_s2_bwd_vec4"]);

    // Even 2x2 upsample backward -> pool_2x2_s2_vec4.
    fresh.trace_begin();
    let _ = backend
        .upsample_nearest2d_backward(View::new(&b_even, &Layout::contiguous(s_even.clone())), [2, 2])
        .unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["pool_2x2_s2_vec4"]);

    // Odd-width 2x2 avg_pool2d fwd/bwd and upsample backward fall back to generic `pool`.
    let s_odd = Shape::new(&[1, 4, 10, 10]);
    let b_odd = backend.full(dev, DType::F32, s_odd.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let b_odd_out = backend.avg_pool2d(View::new(&b_odd, &Layout::contiguous(s_odd.clone())), &p2).unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["pool"]);
    let s_odd_out = Shape::new(&[1, 4, 5, 5]);
    fresh.trace_begin();
    let _ = backend
        .avg_pool2d_backward(View::new(&b_odd_out, &Layout::contiguous(s_odd_out)), &s_odd, &p2)
        .unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["pool"]);

    // GAP 1x1 fwd -> reduce_rows, bwd -> adaptive_gap_bwd.
    fresh.trace_begin();
    let b_gap =
        backend.adaptive_avg_pool2d(View::new(&b_even, &Layout::contiguous(s_even.clone())), [1, 1]).unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["reduce_rows"]);
    let s_gap = Shape::new(&[1, 4, 1, 1]);
    fresh.trace_begin();
    let _ =
        backend.adaptive_avg_pool2d_backward(View::new(&b_gap, &Layout::contiguous(s_gap)), &s_even).unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["adaptive_gap_bwd"]);

    // Whole-plane `PLANE_VEC4` stride-1 max_pool2d (values-only, indexed fwd, and bwd) on [1, 4, 20, 20]:
    // 4 planes -> 4 workgroups (1 workgroup/plane, down from 16 in 16x16 scalar tiling).
    let s_sppf = Shape::new(&[1, 4, 20, 20]);
    let b_sppf = backend.full(dev, DType::F32, s_sppf.numel(), 1.0).unwrap();
    let sppf_p = Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] };
    fresh.trace_begin();
    let _ =
        backend.max_pool2d_values(View::new(&b_sppf, &Layout::contiguous(s_sppf.clone())), &sppf_p).unwrap();
    let tr = fresh.trace_end();
    assert_eq!(tr.len(), 1);
    assert_eq!((tr[0].name, tr[0].groups), ("max_pool2d", [4, 1, 1]));

    fresh.trace_begin();
    let (b_mv, b_mi) =
        backend.max_pool2d(View::new(&b_sppf, &Layout::contiguous(s_sppf.clone())), &sppf_p).unwrap();
    let tr = fresh.trace_end();
    assert_eq!(tr.len(), 1);
    assert_eq!((tr[0].name, tr[0].groups), ("pool_idx", [4, 1, 1]));

    fresh.trace_begin();
    let _ = backend
        .max_pool2d_backward(
            View::new(&b_mv, &Layout::contiguous(s_sppf.clone())),
            View::new(&b_mi, &Layout::contiguous(s_sppf.clone())),
            &s_sppf,
            &sppf_p,
        )
        .unwrap();
    let tr = fresh.trace_end();
    assert_eq!(tr.len(), 1);
    assert_eq!((tr[0].name, tr[0].groups), ("pool_bw_idx", [4, 1, 1]));

    // Larger same-padded plane [1, 2, 80, 80] uses `TILED_VEC4` (32x16 tiles = ceil(20/8) * ceil(80/16) = 15 tiles/plane -> 30 workgroups).
    let s_80 = Shape::new(&[1, 2, 80, 80]);
    let b_80 = backend.full(dev, DType::F32, s_80.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let _ = backend.max_pool2d_values(View::new(&b_80, &Layout::contiguous(s_80.clone())), &sppf_p).unwrap();
    let tr = fresh.trace_end();
    assert_eq!(tr.len(), 1);
    assert_eq!((tr[0].name, tr[0].groups), ("max_pool2d", [30, 1, 1]));

    // Odd-width plane [1, 4, 21, 21] falls back to 16x16 scalar `TILED` (4 planes * 2x2 = 16 workgroups).
    let s_odd_s1 = Shape::new(&[1, 4, 21, 21]);
    let b_odd_s1 = backend.full(dev, DType::F32, s_odd_s1.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let _ = backend
        .max_pool2d_values(View::new(&b_odd_s1, &Layout::contiguous(s_odd_s1.clone())), &sppf_p)
        .unwrap();
    let tr = fresh.trace_end();
    assert_eq!(tr.len(), 1);
    assert_eq!((tr[0].name, tr[0].groups), ("max_pool2d", [16, 1, 1]));

    // 2x2 stride-2 pad-0 values-only max_pool2d S2_VEC4 fast path: even width uses 1/4 threads of output pixels.
    let s_m2 = Shape::new(&[1, 4, 16, 16]);
    let b_m2 = backend.full(dev, DType::F32, s_m2.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let _ = backend.max_pool2d_values(View::new(&b_m2, &Layout::contiguous(s_m2.clone())), &p2).unwrap();
    let tr_m2 = fresh.trace_end();
    assert_eq!(tr_m2.len(), 1);
    // 4 * 8 * (8 / 2) = 128 vec2 units -> 1 workgroup (vs 4 * 8 * 8 = 256 -> still 1 wg; check larger [4, 4, 16, 16]).
    let s_m2b = Shape::new(&[4, 4, 16, 16]);
    let b_m2b = backend.full(dev, DType::F32, s_m2b.numel(), 1.0).unwrap();
    fresh.trace_begin();
    let _ = backend.max_pool2d_values(View::new(&b_m2b, &Layout::contiguous(s_m2b)), &p2).unwrap();
    let tr_m2b = fresh.trace_end();
    // 16 planes * 8 * 4 = 512 vec2 units -> 2 workgroups (scalar would be 1024 -> 4 workgroups).
    assert_eq!((tr_m2b[0].name, tr_m2b[0].groups), ("max_pool2d", [2, 1, 1]));

    // NaN in every one of the 4 positions of a 2x2 window + signed-zero bit preservation on S2_VEC4 (F32 & F16).
    {
        let _g = folo_core::no_grad();
        let dtypes: Vec<DType> = if has_f16 { vec![DType::F32, DType::F16] } else { vec![DType::F32] };
        let mut vals = vec![1.0f32; 4 * 4];
        // Four 2x2 windows across a 4x4 plane (ow=2 even -> hits S2_VEC4): place NaN in each corner.
        vals[0] = f32::NAN; // top-left of (0,0)
        vals[3] = f32::NAN; // top-right of (0,1)
        vals[8] = f32::NAN; // bottom-left of (1,0)
        vals[15] = f32::NAN; // bottom-right of (1,1)
        let t_nan = Tensor::from_vec(vals, [1, 1, 4, 4]).unwrap();
        let t_neg0 = Tensor::from_vec(
            vec![
                -0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                -0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                -0.0,
                -0.0,
                2.0,
                -5.0,
                f32::NEG_INFINITY,
                -0.0,
                -1.0,
                3.0,
            ],
            [1, 1, 4, 4],
        )
        .unwrap();
        for &dt in &dtypes {
            let rn = t_nan.to_dtype(dt).unwrap();
            let got = rn.to_device(dev).unwrap().max_pool2d(2, 2, 0).unwrap();
            let got_v = got.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
            assert!(got_v.iter().all(|v| v.is_nan()), "S2_VEC4 {dt:?} missed NaN: {got_v:?}");
            let rz = t_neg0.to_dtype(dt).unwrap();
            let cz = if dt == DType::F16 { rz.to_dtype(DType::F32).unwrap() } else { rz.clone() };
            let want = cz.max_pool2d(2, 2, 0).unwrap().to_dtype(dt).unwrap();
            let got_z = rz.to_device(dev).unwrap().max_pool2d(2, 2, 0).unwrap();
            super::fixtures::assert_same(&got_z, &want, &format!("S2_VEC4 {dt:?} -0.0 bits"));
        }
    }

    // Every new kernel is registered in `DEFS` for precompile.
    assert!(
        fresh.kernels_without_registry().is_empty(),
        "unregistered kernels: {:?}",
        fresh.kernels_without_registry()
    );
}

/// Parity and edge-case coverage for the stride-1 `PLANE_VEC4` (`PLANE_CAP = 256` & `512`) and `TILED_VEC4`
/// max-pool paths across all supported odd kernel sizes (`3×3`, `5×5`, `7×7`, and rectangular `3×5` / `5×7`),
/// both `F32` and `F16`, verifying values-only inference, indexed forward, exact `i64` indices, backward
/// gradients, row-major tie-breaking, NaN propagation, and `-0.0` sign preservation.
#[test]
fn stride1_vec4_max_pool_all_tiers_and_edges() {
    use folo_core::backend::Pool2dParams;
    use folo_core::{DType, no_grad};

    let dev = dev!();
    let ctx = crate::context_for(dev).unwrap();
    let has_f16 = ctx.caps().shader_f16;
    let dtypes: Vec<DType> = if has_f16 { vec![DType::F32, DType::F16] } else { vec![DType::F32] };

    // Shapes exercising every tier:
    // - [1, 2, 8, 8] & [2, 3, 20, 20]: PLANE_VEC4 small (PLANE_CAP=256)
    // - [1, 2, 40, 40]: PLANE_VEC4 large (PLANE_CAP=512 when workgroup storage permits)
    // - [1, 2, 52, 44]: TILED_VEC4 with partial right/bottom tiles (w4=11 -> 2 tiles of 8, h=52 -> 4 tiles of 16)
    let shapes = [[1usize, 2, 8, 8], [2, 3, 20, 20], [1, 2, 40, 40], [1, 2, 52, 44]];
    let kernels = [(3usize, 3usize), (5, 5), (7, 7), (3, 5), (5, 7)];

    for &dt in &dtypes {
        for shape in shapes {
            let n: usize = shape.iter().product();
            // Quantized values create frequent horizontal and vertical ties to stress row-major tie-breaking.
            let mut data: Vec<f32> = (0..n).map(|i| (((i * 37 + 13) % 19) as f32 - 9.0) * 0.25).collect();
            let hw = shape[2] * shape[3];
            // Inject NaNs across edge, interior, and vec4-lane boundaries on plane 0; keep plane 1 finite.
            if hw >= 64 {
                data[0] = f32::NAN;
                data[3] = f32::NAN;
                data[4] = f32::NAN;
                data[shape[3] + 5] = f32::NAN;
                data[hw - 1] = f32::NAN;
            }
            let x_raw = Tensor::from_slice(&data, shape).unwrap().to_dtype(dt).unwrap();
            let x_cpu_f32 =
                if dt == DType::F16 { x_raw.to_dtype(DType::F32).unwrap() } else { x_raw.clone() };
            let g_data: Vec<f32> = (0..n).map(|i| (((i * 29 + 5) % 17) as f32 - 8.0) * 0.125).collect();
            let g_raw = Tensor::from_slice(&g_data, shape).unwrap().to_dtype(dt).unwrap();
            let g_cpu_f32 =
                if dt == DType::F16 { g_raw.to_dtype(DType::F32).unwrap() } else { g_raw.clone() };

            for &(kh, kw) in &kernels {
                let p = Pool2dParams { kernel: [kh, kw], stride: [1, 1], padding: [kh / 2, kw / 2] };

                // 1. Values-only inference path (`no_grad` -> `max_pool2d_values` -> `pooling::max_pool2d`).
                {
                    let _g = no_grad();
                    let want = x_cpu_f32.max_pool2d_with(&p).unwrap().to_dtype(dt).unwrap();
                    let got = x_raw.to_device(dev).unwrap().max_pool2d_with(&p).unwrap();
                    assert_close_t(&got, &want, 0.0, 0.0, &format!("max_val {dt:?} {shape:?} {kh}x{kw}"));
                }

                // 2. Indexed forward + autograd backward parity (`pool_idx` + `pool_bw_idx`).
                let xc = x_cpu_f32.detach().requires_grad_();
                let yc = xc.max_pool2d_with(&p).unwrap();
                let gc = yc.backward_with(&g_cpu_f32).unwrap();
                let want_y = yc.to_dtype(dt).unwrap();
                let want_gx = gc.get(&xc).unwrap().to_dtype(dt).unwrap();

                let xg = x_raw.to_device(dev).unwrap().detach().requires_grad_();
                let yg = xg.max_pool2d_with(&p).unwrap();
                let gg = yg.backward_with(&g_raw.to_device(dev).unwrap()).unwrap();
                assert_close_t(&yg, &want_y, 0.0, 0.0, &format!("max_idx val {dt:?} {shape:?} {kh}x{kw}"));
                let tol = if dt == DType::F16 { 1e-2 } else { 1e-5 };
                assert_close_t(
                    gg.get(&xg).unwrap(),
                    &want_gx,
                    tol,
                    tol,
                    &format!("max_bwd {dt:?} {shape:?} {kh}x{kw}"),
                );
            }
        }

        // Exact signed-zero (-0.0 over -inf) preservation on PLANE_VEC4 (8x8) and TILED_VEC4 (48x48).
        for side in [8usize, 48] {
            let mut z = vec![f32::NEG_INFINITY; side * side];
            z[0] = -0.0;
            z[side * side / 2 + 3] = -0.0;
            let tz = Tensor::from_slice(&z, [1, 1, side, side]).unwrap().to_dtype(dt).unwrap();
            let _g = no_grad();
            let cz = if dt == DType::F16 { tz.to_dtype(DType::F32).unwrap() } else { tz.clone() };
            let want = cz.max_pool2d(5, 1, 2).unwrap().to_dtype(dt).unwrap();
            let got = tz.to_device(dev).unwrap().max_pool2d(5, 1, 2).unwrap();
            super::fixtures::assert_same(&got, &want, &format!("max_val -0.0 {dt:?} {side}x{side}"));
        }
    }

    // Direct kernel-level exact `i64` argmax index verification (first max in row-major order, first NaN)
    // and synthetic out-of-window index rejection in `max_pool2d_backward` on PLANE_VEC4 & TILED_VEC4.
    {
        use crate::DType as Gdt;
        use crate::kernels::pool_grad::{self, Pool2d};
        let win = Pool2d { kernel: (5, 5), stride: (1, 1), padding: (2, 2) };
        for (h, w) in [(20usize, 20usize), (40, 40), (48, 44)] {
            let hw = h * w;
            let mut vals: Vec<f32> = (0..hw).map(|i| ((i * 11 + 3) % 7) as f32).collect();
            vals[2 * w + 5] = f32::NAN;
            vals[2 * w + 6] = f32::NAN; // earlier NaN at (2, 5) must win
            let gx_buf = ctx.upload(bytemuck::cast_slice(&vals)).unwrap();
            let gy_buf = ctx.alloc(Gdt::F32.bytes(hw)).unwrap();
            let gi_buf = ctx.alloc(Gdt::F32.bytes(hw * 2)).unwrap();
            pool_grad::max_pool2d_indices(
                &ctx,
                gx_buf.raw(),
                gy_buf.raw(),
                gi_buf.raw(),
                (1, h, w),
                &win,
                Gdt::F32,
            )
            .unwrap();
            let raw_i = ctx.read(gi_buf.raw(), 0, (hw * 8) as u64).unwrap();
            let got_i: Vec<i64> = bytemuck::cast_slice::<u8, i64>(&raw_i).to_vec();
            let mut want_i = vec![0i64; hw];
            for oy in 0..h {
                for ox in 0..w {
                    let mut best = f32::NEG_INFINITY;
                    let mut bi = -1i64;
                    for ky in 0..5usize {
                        let iy = oy as isize + ky as isize - 2;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..5usize {
                            let ix = ox as isize + kx as isize - 2;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let at = (iy as usize * w + ix as usize) as i64;
                            let v = vals[at as usize];
                            if bi < 0 || (!best.is_nan() && (v.is_nan() || v > best)) {
                                best = v;
                                bi = at;
                            }
                        }
                    }
                    want_i[oy * w + ox] = bi;
                }
            }
            assert_eq!(got_i, want_i, "exact i64 argmax indices {h}x{w}");

            // Synthetic backward index pointing outside each output's 5x5 window must NOT accumulate.
            let synth_i: Vec<i64> = vec![0i64; hw]; // pixel 0 only covered by oy<=2, ox<=2 -> 9 outputs
            let ones = vec![1.0f32; hw];
            let g_buf = ctx.upload(bytemuck::cast_slice(&ones)).unwrap();
            let si_buf = ctx.upload(bytemuck::cast_slice(&synth_i)).unwrap();
            let out_buf = ctx.alloc(Gdt::F32.bytes(hw)).unwrap();
            pool_grad::max_pool2d_backward(
                &ctx,
                g_buf.raw(),
                si_buf.raw(),
                out_buf.raw(),
                (1, h, w),
                &win,
                Gdt::F32,
            )
            .unwrap();
            let raw_gx = ctx.read(out_buf.raw(), 0, (hw * 4) as u64).unwrap();
            let got_gx: &[f32] = bytemuck::cast_slice(&raw_gx);
            assert_eq!(got_gx[0], 9.0, "out-of-window synthetic index leaked on {h}x{w}");
            assert!(got_gx[1..].iter().all(|&v| v == 0.0), "unexpected non-zero grad on {h}x{w}");
        }
    }
    assert!(ctx.kernels_without_registry().is_empty());
}
