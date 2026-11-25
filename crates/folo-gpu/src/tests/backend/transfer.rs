//! Moving data: uploads and downloads (blocking and async), strided copies, casts, and device mixing.

use folo_core::{DType, Device, Tensor, no_grad};

use super::fixtures::{assert_close_t, assert_same, dev, randn};

#[test]
fn memory_roundtrip_and_fill() {
    let dev = dev!();
    let f = Tensor::from_slice(&[1.5f32, -2.0, f32::NAN, 0.0, -0.0], [5]).unwrap();
    let h = f.to_dtype(DType::F16).unwrap();
    let b = Tensor::from_slice(&[1u8, 2, 3, 250, 7], [5]).unwrap();
    let u = Tensor::from_slice(&[1u32, u32::MAX, 0, 9, 4], [5]).unwrap();
    let i = Tensor::from_slice(&[-1i64, i64::MAX, 0, 3, i64::MIN], [5]).unwrap();
    for t in [&f, &h, &b, &u, &i] {
        let g = t.to_device(dev).unwrap();
        assert_eq!(g.device(), dev);
        assert_same(&g, t, &format!("roundtrip {}", t.dtype()));
        // Odd sub-ranges: offsets that are not word-aligned for u8 / f16.
        let n = t.narrow(0, 1, 3).unwrap();
        assert_same(&g.narrow(0, 1, 3).unwrap(), &n, &format!("narrow {}", t.dtype()));
    }
    for (dtype, v) in
        [(DType::F32, -2.5), (DType::F16, 3.25), (DType::U8, 7.0), (DType::U32, 70000.0), (DType::I64, -5.0)]
    {
        for len in [0usize, 1, 3, 7] {
            let g = Tensor::full_with([len], v, dtype, dev).unwrap();
            let c = Tensor::full_with([len], v, dtype, Device::Cpu).unwrap();
            assert_same(&g, &c, &format!("full {dtype} x{len}"));
        }
    }
    let z = Tensor::zeros_with([2, 3], DType::F32, dev).unwrap();
    assert_eq!(z.to_vec::<f32>().unwrap(), vec![0.0; 6]);
    folo_core::backend::synchronize(dev).unwrap();
}

type ViewFn = Box<dyn Fn(&Tensor) -> Tensor>;

#[test]
fn strided_copies_all_dtypes() {
    let dev = dev!();
    let base = Tensor::arange(0.0, 60.0, 1.0).unwrap().reshape([3, 4, 5]).unwrap();
    for dtype in [DType::F32, DType::F16, DType::U8, DType::U32, DType::I64] {
        let c = base.to_dtype(dtype).unwrap();
        let g = c.to_device(dev).unwrap();
        let views: Vec<(&str, ViewFn)> = vec![
            ("transpose", Box::new(|t: &Tensor| t.transpose(0, 2).unwrap())),
            ("permute", Box::new(|t: &Tensor| t.permute([1, 2, 0]).unwrap())),
            ("slice step", Box::new(|t: &Tensor| t.slice(2, 1, 5, 2).unwrap())),
            ("narrow", Box::new(|t: &Tensor| t.narrow(1, 1, 3).unwrap())),
            (
                "broadcast",
                Box::new(|t: &Tensor| t.narrow(0, 1, 1).unwrap().broadcast_to([4, 3, 4, 5]).unwrap()),
            ),
        ];
        for (name, v) in &views {
            assert_same(
                &v(&g).contiguous().unwrap(),
                &v(&c).contiguous().unwrap(),
                &format!("{name} {dtype}"),
            );
        }
        // copy_into through cat (sub-word elements of neighbouring pieces share words) and pad.
        for dim in 0..3 {
            let n = c.dims()[dim];
            let pieces = |t: &Tensor| {
                vec![
                    t.narrow(dim, n - 1, 1).unwrap(),
                    t.narrow(dim, 1, n - 2).unwrap(),
                    t.narrow(dim, 0, 1).unwrap(),
                ]
            };
            let cc = Tensor::cat(&pieces(&c), dim).unwrap();
            let gg = Tensor::cat(&pieces(&g), dim).unwrap();
            assert_same(&gg, &cc, &format!("cat dim {dim} {dtype}"));
        }
        if dtype == DType::F32 {
            assert_same(
                &g.pad(&[(1, 2), (0, 3)], -1.0).unwrap(),
                &c.pad(&[(1, 2), (0, 3)], -1.0).unwrap(),
                "pad",
            );
        }
        assert_same(
            &g.t()
                .unwrap()
                .index_select(1, &Tensor::from_slice(&[4i64, 0, 4], [3]).unwrap().to_device(dev).unwrap())
                .unwrap(),
            &c.t().unwrap().index_select(1, &Tensor::from_slice(&[4i64, 0, 4], [3]).unwrap()).unwrap(),
            &format!("index_select {dtype}"),
        );
    }
    // copy_from / fill_assign into strided views.
    let cpu = Tensor::zeros([4, 6]);
    let gpu = cpu.to_device(dev).unwrap();
    let src = randn(&[6, 2], 3);
    let _g = no_grad();
    for (t, s) in [(&cpu, src.clone()), (&gpu, src.to_device(dev).unwrap())] {
        t.narrow(0, 1, 2).unwrap().t().unwrap().copy_from(&s).unwrap();
        t.slice(1, 0, 6, 3).unwrap().fill_assign(9.0).unwrap();
        t.narrow(0, 3, 1).unwrap().add_scaled_assign(&s.t().unwrap().narrow(0, 0, 1).unwrap(), -0.5).unwrap();
        t.mul_scalar_assign(2.0).unwrap();
    }
    assert_close_t(&gpu, &cpu, 0.0, 1e-6, "in-place updates");
}

#[test]
fn casts_all_pairs() {
    let dev = dev!();
    let vals = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        1.7,
        -1.7,
        255.4,
        256.0,
        300.0,
        -300.0,
        65504.0,
        70000.0,
        3e9,
        5e9,
        -3e9,
        1e19,
        -1e19,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.25,
    ];
    let f = Tensor::from_slice(&vals, [vals.len()]).unwrap();
    let ints = Tensor::from_slice(
        &[0i64, 1, -1, 255, 256, -300, 4294967295, 4294967296, i64::MAX, i64::MIN, 12345],
        [11],
    )
    .unwrap();
    let all = [DType::F32, DType::F16, DType::U8, DType::U32, DType::I64];
    for src in [&f, &ints] {
        for &from in &all {
            let c = src.to_dtype(from).unwrap();
            let g = c.to_device(dev).unwrap();
            for &to in &all {
                let (cc, gc) = (c.to_dtype(to).unwrap(), g.to_dtype(to).unwrap());
                assert_same(
                    &gc,
                    &cc,
                    &format!(
                        "cast {from} -> {to} of {:?}",
                        c.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap()
                    ),
                );
            }
            // A strided source.
            let (cs, gs) = (c.slice(0, 1, c.dims()[0], 3).unwrap(), g.slice(0, 1, c.dims()[0], 3).unwrap());
            assert_same(
                &gs.to_dtype(DType::U8).unwrap(),
                &cs.to_dtype(DType::U8).unwrap(),
                &format!("strided cast {from}"),
            );
        }
    }
}

#[test]
fn casts_round_like_the_cpu() {
    let dev = dev!();
    // Integers next to f32 / f16 rounding midpoints, including the f64 double-rounding cases
    // (2^62 + 2^38 + 1024 rounds up only because f64 keeps the sticky bit).
    let mut ints: Vec<i64> = vec![
        (1 << 62) + (1 << 38) + 1024,
        (1 << 24) + 1,
        (1 << 24) + 3,
        (1 << 25) + 2,
        2049,
        2051,
        4097,
        65519,
        65520,
        65535,
    ];
    ints.extend([(1i64 << 53) + 1, (1 << 54) + 3, i64::MAX, i64::MAX - 1, 4294967295, 16777217 * 3]);
    let neg: Vec<i64> = ints.iter().map(|v| v.wrapping_neg()).collect();
    ints.extend(neg);
    ints.push(i64::MIN);
    let n = ints.len();
    let i = Tensor::from_slice(&ints, [n]).unwrap();
    let u = Tensor::from_slice(&ints.iter().map(|&v| v as u32).collect::<Vec<_>>(), [n]).unwrap();
    // f32 values on f16 midpoints, subnormal boundaries and overflow.
    let mut fl = vec![1.0f32 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0, 2049.0, 2051.0, 65504.0, 65519.0, 65520.0];
    fl.extend([6.0e-8, 2.9802322e-8, 2.9802326e-8, 5.9604645e-8 * 1.5, 6.1035156e-5, 6.1e-5, 1e-10, -3.0e-5]);
    fl.extend((0..64).map(|k| f32::from_bits(0x3880_0000 + k * 0x1001)));
    let f = Tensor::from_slice(&fl, [fl.len()]).unwrap();
    for (src, to) in
        [(&i, DType::F32), (&i, DType::F16), (&u, DType::F32), (&u, DType::F16), (&f, DType::F16)]
    {
        let g = src.to_device(dev).unwrap().to_dtype(to).unwrap();
        assert_same(&g, &src.to_dtype(to).unwrap(), &format!("{} -> {to}", src.dtype()));
    }
}

#[test]
fn casts_i64_to_float_round_once() {
    let dev = dev!();
    // 2^k + 2^(k - 24) + 1 sits just above an f32 midpoint; rounding to f64's 53 bits first drops the low 1 and
    // leaves an exact tie that rounds down, so only a single rounding gets these right.
    let mut ints: Vec<i64> = (54..63).map(|k| (1i64 << k) + (1 << (k - 24)) + 1).collect();
    ints.extend([
        (1 << 62) + (1 << 38) - 1,
        (1 << 62) + (1 << 38),
        (1 << 62) + (3 << 38),
        i64::MAX,
        i64::MAX - 1,
    ]);
    ints.extend([0, 1, 7, 2049, 65519, 65520, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 3, 1 << 53]);
    ints.extend([(1 << 53) + 1, i64::MIN + (1 << 39) + 1, i64::MIN + (1 << 39) - 1]);
    let neg: Vec<i64> = ints.iter().map(|v| v.wrapping_neg()).collect();
    ints.extend(neg);
    ints.push(i64::MIN);
    let i = Tensor::from_slice(&ints, [ints.len()]).unwrap();
    let g = i.to_device(dev).unwrap();
    for to in [DType::F32, DType::F16] {
        assert_same(&g.to_dtype(to).unwrap(), &i.to_dtype(to).unwrap(), &format!("i64 -> {to} of {ints:?}"));
    }
}

#[test]
fn errors_name_the_backend() {
    let dev = dev!();
    let g = Tensor::ones([2]).to_device(dev).unwrap();
    let c = Tensor::ones([2]);
    let e = g.add(&c).unwrap_err().to_string();
    assert!(e.contains("device"), "{e}");
    // An op on a GPU tensor stays on the GPU; mixing in a CPU index tensor is rejected.
    assert!(g.index_select(0, &Tensor::from_slice(&[0i64], [1]).unwrap()).is_err());
}

#[test]
fn download_async_matches_to_vec() {
    let dev = dev!();
    let ctx = crate::GpuContext::shared().unwrap();
    let x = randn(&[3, 5, 7], 41);
    let g = x.to_device(dev).unwrap();
    // A strided view (made dense on the GPU first), an offset view and the whole tensor.
    let views = [g.transpose(0, 2).unwrap(), g.narrow(1, 1, 3).unwrap(), g.clone()];
    let cpu_views = [x.transpose(0, 2).unwrap(), x.narrow(1, 1, 3).unwrap(), x.clone()];
    for (v, c) in views.iter().zip(&cpu_views) {
        let got = pollster::block_on(crate::download_async(&ctx, v)).unwrap();
        assert!(got.device().is_cpu() && got.dims() == c.dims());
        assert_eq!(got.to_vec::<f32>().unwrap(), c.to_vec::<f32>().unwrap());
    }
    // Other dtypes and the empty tensor.
    let i = Tensor::from_vec(vec![-3i64, 0, 7, i64::MAX], [4]).unwrap();
    let got = pollster::block_on(crate::download_async(&ctx, &i.to_device(dev).unwrap())).unwrap();
    assert_eq!(got.to_vec::<i64>().unwrap(), vec![-3, 0, 7, i64::MAX]);
    let e = Tensor::zeros([0, 4]).to_device(dev).unwrap();
    assert_eq!(pollster::block_on(crate::download_async(&ctx, &e)).unwrap().dims(), [0, 4]);
}

#[test]
fn download_async_copies_cpu_tensors() {
    let ctx = match crate::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => return eprintln!("note: skipping GPU backend test: {e}"),
    };
    let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], [3]).unwrap();
    let got = pollster::block_on(crate::download_async(&ctx, &x)).unwrap();
    let _g = no_grad();
    x.fill_assign(9.0).unwrap();
    assert_eq!(got.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0], "an independent copy");
}

#[test]
fn strided_fast_vec4_tiled_transpose_and_f16_cast() {
    let dev = dev!();
    // 1. Vec4 fill across all dtypes (4-word-aligned lengths: 16, 32, 64 elements).
    for (dtype, v) in [
        (DType::F32, -3.5),
        (DType::F16, 2.75),
        (DType::U8, 42.0),
        (DType::U32, f64::from(0x1234_5678u32)),
        (DType::I64, -9_000_000_003_f64),
    ] {
        for len in [16usize, 32, 64] {
            let g = Tensor::full_with([len], v, dtype, dev).unwrap();
            let c = Tensor::full_with([len], v, dtype, Device::Cpu).unwrap();
            assert_same(&g, &c, &format!("vec4 full {dtype} x{len}"));
        }
    }

    // 2. Vec4 strided gather, scatter (`cat` / channel slice), and `index_select` (i64 and u32) across all dtypes.
    let base = Tensor::arange(0.0, (2 * 6 * 16) as f32, 1.0).unwrap().reshape([2, 6, 16]).unwrap();
    let idx_i64 = Tensor::from_slice(&[5i64, 0, 2, 5, 1], [5]).unwrap();
    let gidx_i64 = idx_i64.to_device(dev).unwrap();
    let idx_u32 = Tensor::from_slice(&[5u32, 0, 2, 5, 1], [5]).unwrap();
    let gidx_u32 = idx_u32.to_device(dev).unwrap();
    for dtype in [DType::F32, DType::F16, DType::U8, DType::U32, DType::I64] {
        let c = base.to_dtype(dtype).unwrap();
        let g = c.to_device(dev).unwrap();
        // Strided channel slice (16 elements wide -> 16B aligned for every dtype including u8).
        assert_same(
            &g.narrow(1, 2, 3).unwrap().contiguous().unwrap(),
            &c.narrow(1, 2, 3).unwrap().contiguous().unwrap(),
            &format!("vec4 gather {dtype}"),
        );
        // Channel cat -> vec4 scatter.
        let cc = Tensor::cat(&[&c.narrow(1, 3, 3).unwrap(), &c.narrow(1, 0, 3).unwrap()], 1).unwrap();
        let gg = Tensor::cat(&[&g.narrow(1, 3, 3).unwrap(), &g.narrow(1, 0, 3).unwrap()], 1).unwrap();
        assert_same(&gg, &cc, &format!("vec4 scatter cat {dtype}"));
        // Vec4 index_select (inner = 16 -> 16B aligned for every dtype) with both i64 and u32 indices.
        assert_same(
            &g.index_select(1, &gidx_i64).unwrap(),
            &c.index_select(1, &idx_i64).unwrap(),
            &format!("vec4 index_select i64 {dtype}"),
        );
        assert_same(
            &g.index_select(1, &gidx_u32).unwrap(),
            &c.index_select(1, &idx_u32).unwrap(),
            &format!("vec4 index_select u32 {dtype}"),
        );
    }

    // Direct kernel check: vec4 `index_select` zero-fills OOB i64 (negative & >= d) and u32 (>= d) indices.
    {
        use crate::kernels::strided::{self, Elem, Indices};
        let ctx = crate::context_for(dev).expect("registered");
        let src: Vec<u32> = (1..=32u32).collect();
        let x_buf = ctx.upload(bytemuck::cast_slice(&src)).unwrap();
        let i64_idx = ctx.upload(bytemuck::cast_slice(&[-1i64, 1, 4, 99])).unwrap();
        let out_i64 = ctx.alloc(4 * 8 * 4).unwrap();
        strided::index_select(
            &ctx,
            Elem::U32,
            x_buf.raw(),
            (1, 4, 8),
            Indices { buf: i64_idx.raw(), len: 4, i64: true },
            out_i64.raw(),
        )
        .unwrap();
        let got_i64: Vec<u32> =
            bytemuck::cast_slice(&ctx.read(out_i64.raw(), 0, 4 * 8 * 4).unwrap()).to_vec();
        let mut want_i64 = vec![0u32; 32];
        want_i64[8..16].copy_from_slice(&src[8..16]);
        assert_eq!(got_i64, want_i64, "vec4 index_select i64 OOB zero-fill");

        let u32_idx = ctx.upload(bytemuck::cast_slice(&[4u32, 2, 99, 0])).unwrap();
        let out_u32 = ctx.alloc(4 * 8 * 4).unwrap();
        strided::index_select(
            &ctx,
            Elem::U32,
            x_buf.raw(),
            (1, 4, 8),
            Indices { buf: u32_idx.raw(), len: 4, i64: false },
            out_u32.raw(),
        )
        .unwrap();
        let got_u32: Vec<u32> =
            bytemuck::cast_slice(&ctx.read(out_u32.raw(), 0, 4 * 8 * 4).unwrap()).to_vec();
        let mut want_u32 = vec![0u32; 32];
        want_u32[8..16].copy_from_slice(&src[16..24]);
        want_u32[24..32].copy_from_slice(&src[0..8]);
        assert_eq!(got_u32, want_u32, "vec4 index_select u32 OOB zero-fill");
    }

    // 3. Shared-memory 32x32 tiled transpose: 2-D, batched 3-D, non-multiples of 32, and sliced rows.
    for shape in [vec![37usize, 65], vec![64, 32], vec![3, 35, 67], vec![2, 33, 33]] {
        let n: usize = shape.iter().product();
        let raw = Tensor::arange(0.0, n as f32, 1.0).unwrap().reshape(shape.clone()).unwrap();
        let last = shape.len() - 1;
        for dtype in [DType::F32, DType::F16, DType::U32, DType::I64, DType::U8] {
            let c = raw.to_dtype(dtype).unwrap();
            let g = c.to_device(dev).unwrap();
            assert_same(
                &g.transpose(last - 1, last).unwrap().contiguous().unwrap(),
                &c.transpose(last - 1, last).unwrap().contiguous().unwrap(),
                &format!("tiled transpose {shape:?} {dtype}"),
            );
            // Sliced-row transpose (non-zero offset and row_stride > C).
            assert_same(
                &g.narrow(last, 1, shape[last] - 3)
                    .unwrap()
                    .transpose(last - 1, last)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
                &c.narrow(last, 1, shape[last] - 3)
                    .unwrap()
                    .transpose(last - 1, last)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
                &format!("tiled sliced transpose {shape:?} {dtype}"),
            );
        }
    }
    // Batched 3-D raw-bit F16 tiled transpose slicing BOTH matrix axes (independent b_stride gap, non-zero
    // offset, subnormals, -0.0, ±inf, and quiet NaNs).
    {
        let specials: [u16; 10] =
            [0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x7bff, 0x7c00, 0xfc00, 0x7e05, 0xfe09];
        let bits: Vec<half::f16> =
            (0..(3 * 38 * 39)).map(|i| half::f16::from_bits(specials[i % specials.len()])).collect();
        let c = Tensor::from_slice(&bits, [3, 38, 39]).unwrap();
        let g = c.to_device(dev).unwrap();
        let cs = c
            .narrow(1, 2, 34)
            .unwrap()
            .narrow(2, 3, 35)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        let gs = g
            .narrow(1, 2, 34)
            .unwrap()
            .narrow(2, 3, 35)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        assert_same(&gs, &cs, "tiled 3D F16 double-sliced raw-bit transpose");
    }

    // 4. Branchless integer vec4 f32 <-> f16 cast: 4-aligned buffers exercising fast path (normals, ±0.0) and
    // slow path (subnormals, midpoints, overflow to ±inf, ±inf, quiet/custom NaN payloads) plus offset 4.
    let mut fl: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0 + 1.0 / 2048.0,
        1.0 + 3.0 / 2048.0,
        2049.0,
        2051.0,
        65504.0,
        -65504.0,
        65519.0,
        65520.0,
        -65520.0,
        70000.0,
        -70000.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_2000),
        f32::from_bits(0xffc0_4000),
        6.0e-8,
        2.9802322e-8,
        2.9802326e-8,
        5.9604645e-8 * 1.5,
        6.1035156e-5,
        6.1e-5,
        1e-10,
        -3.0e-5,
    ];
    fl.extend((0..64).map(|k| f32::from_bits(0x3880_0000 + k * 0x1001)));
    while !fl.len().is_multiple_of(4) {
        fl.push(0.5);
    }
    let f = Tensor::from_slice(&fl, [fl.len()]).unwrap();
    let gf = f.to_device(dev).unwrap();
    let h_cpu = f.to_dtype(DType::F16).unwrap();
    let h_gpu = gf.to_dtype(DType::F16).unwrap();
    assert_same(&h_gpu, &h_cpu, "vec4 f32 -> f16");
    assert_same(
        &h_gpu.to_dtype(DType::F32).unwrap(),
        &h_cpu.to_dtype(DType::F32).unwrap(),
        "vec4 f16 -> f32",
    );
    // Non-zero 4-aligned offset slice.
    assert_same(
        &gf.narrow(0, 4, fl.len() - 4).unwrap().to_dtype(DType::F16).unwrap(),
        &f.narrow(0, 4, fl.len() - 4).unwrap().to_dtype(DType::F16).unwrap(),
        "vec4 f32 -> f16 offset 4",
    );
    assert_same(
        &h_gpu.narrow(0, 4, fl.len() - 4).unwrap().to_dtype(DType::F32).unwrap(),
        &h_cpu.narrow(0, 4, fl.len() - 4).unwrap().to_dtype(DType::F32).unwrap(),
        "vec4 f16 -> f32 offset 4",
    );

    // 5. Raw-bit F16 -> F32 widening (subnormals, ±0.0, ±inf, signaling and quiet NaNs) on BOTH 4-aligned
    // (`cast_f16_vec4`) and odd-length / odd-offset (`cast_any`) inputs.
    let raw_h: Vec<half::f16> = [
        0x0000u16, 0x8000, 0x0001, 0x8001, 0x0002, 0x03fe, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00,
        0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e05, 0xfe09,
    ]
    .into_iter()
    .map(half::f16::from_bits)
    .collect();
    let hc = Tensor::from_slice(&raw_h, [raw_h.len()]).unwrap();
    let hg = hc.to_device(dev).unwrap();
    assert_same(&hg.to_dtype(DType::F32).unwrap(), &hc.to_dtype(DType::F32).unwrap(), "raw vec4 f16 -> f32");
    assert_same(
        &hg.narrow(0, 4, raw_h.len() - 4).unwrap().to_dtype(DType::F32).unwrap(),
        &hc.narrow(0, 4, raw_h.len() - 4).unwrap().to_dtype(DType::F32).unwrap(),
        "raw vec4 f16 -> f32 offset 4",
    );
    assert_same(
        &hg.narrow(0, 1, raw_h.len() - 3).unwrap().to_dtype(DType::F32).unwrap(),
        &hc.narrow(0, 1, raw_h.len() - 3).unwrap().to_dtype(DType::F32).unwrap(),
        "raw odd f16 -> f32 offset 1",
    );
}
