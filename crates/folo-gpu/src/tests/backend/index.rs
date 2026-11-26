//! `index_select`, `index_add`, `gather`, `scatter_add` and `topk`.

use folo_core::{DType, Tensor};

use super::fixtures::{assert_same, check, dev, randn};

#[test]
fn index_ops() {
    let dev = dev!();
    let x = randn(&[4, 5, 3], 80);
    let idx = Tensor::from_slice(&[4i64, 0, 2, 2, 1], [5]).unwrap();
    let idx32 = Tensor::from_slice(&[3u32, 3, 0], [3]).unwrap();
    check(dev, &[x.clone(), idx.clone()], 0.0, 1e-6, "index_select dim 1", |v| v[0].index_select(1, &v[1]));
    check(dev, &[x.clone(), idx32.clone()], 0.0, 1e-6, "index_select u32 dim 0", |v| {
        v[0].index_select(0, &v[1])
    });
    let src = randn(&[4, 5, 3], 81);
    check(dev, &[x.clone(), idx.clone(), src], 1e-6, 1e-5, "index_add duplicates", |v| {
        v[0].index_add(1, &v[1], &v[2])
    });
    let gi = Tensor::from_slice(&[0i64, 2, 1, 1, 0, 2, 2, 2, 1, 0, 0, 1], [4, 1, 3]).unwrap();
    check(dev, &[x.clone(), gi.clone()], 0.0, 1e-6, "gather dim 1", |v| v[0].gather(1, &v[1]));
    let gs = randn(&[4, 1, 3], 82);
    check(dev, &[x.clone(), gi, gs], 1e-6, 1e-5, "scatter_add", |v| v[0].scatter_add(1, &v[1], &v[2]));
    let gi2 = Tensor::from_slice(&[4i64, 4, 0, 1, 1, 1], [1, 2, 3])
        .unwrap()
        .broadcast_to([4, 2, 3])
        .unwrap()
        .contiguous()
        .unwrap();
    let gs2 = randn(&[4, 2, 3], 83);
    check(dev, &[x.clone(), gi2, gs2], 1e-6, 1e-5, "scatter_add collisions", |v| {
        v[0].scatter_add(1, &v[1], &v[2])
    });
    // Integer data through index_select and gather.
    let ints = Tensor::arange_i64(0, 20).reshape([4, 5]).unwrap();
    let gpu_ints = ints.to_device(dev).unwrap();
    let ii = Tensor::from_slice(&[1i64, 1, 3], [3]).unwrap();
    assert_same(
        &gpu_ints.index_select(1, &ii.to_device(dev).unwrap()).unwrap(),
        &ints.index_select(1, &ii).unwrap(),
        "index_select i64",
    );
    let bytes = ints.to_dtype(DType::U8).unwrap();
    let gb = bytes.to_device(dev).unwrap();
    let g3 = Tensor::from_slice(&[4i64, 0, 3, 1, 2, 2, 0, 0], [4, 2]).unwrap();
    assert_same(
        &gb.gather(1, &g3.to_device(dev).unwrap()).unwrap(),
        &bytes.gather(1, &g3).unwrap(),
        "gather u8",
    );
    // topk with ties, both directions.
    let t = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 5.0, 2.0, 0.0, -1.0, 4.0, 4.0, -0.0, 0.0, 9.0], [3, 4])
        .unwrap();
    for (k, largest) in [(2usize, true), (3, false), (4, true)] {
        let (cv, ci) = t.topk(k, largest).unwrap();
        let (gv, gi) = t.to_device(dev).unwrap().topk(k, largest).unwrap();
        assert_same(&gv, &cv, &format!("topk values k={k} {largest}"));
        assert_same(&gi, &ci, &format!("topk indices k={k} {largest}"));
    }
    let r = randn(&[6, 300], 84);
    check(dev, &[r], 0.0, 0.0, "topk grad", |v| Ok(v[0].topk(10, true)?.0));
}

#[test]
fn gather_index_add_vec4_and_direct_backwards_match_cpu() {
    use folo_core::backend::{Backend, View};
    use folo_core::{Layout, Shape};

    let dev = dev!();
    // Vectorized gather & index_add (inner % 4 == 0) + unaligned odd tails, both I64 and U32 indices,
    // including duplicate indices in index_select/gather/index_add backwards.
    for &(outer, d, inner, label) in &[
        (3usize, 7usize, 4usize, "inner=4 vec4"),
        (2, 9, 16, "inner=16 vec4"),
        (2, 5, 3, "inner=3 scalar"),
        (4, 6, 1, "inner=1 scalar"),
    ] {
        let x = randn(&[outer, d, inner], 900 + inner as u64);
        let idx_i64 = Tensor::from_slice(&[0i64, (d - 1) as i64, 2, 2, 0], [5]).unwrap();
        let idx_u32 = Tensor::from_slice(&[1u32, 1, (d - 1) as u32, 0], [4]).unwrap();
        check(dev, &[x.clone(), idx_i64.clone()], 0.0, 1e-5, &format!("isel i64 {label}"), |v| {
            v[0].index_select(1, &v[1])
        });
        check(dev, &[x.clone(), idx_u32.clone()], 0.0, 1e-5, &format!("isel u32 {label}"), |v| {
            v[0].index_select(1, &v[1])
        });
        let src = randn(&[outer, 5, inner], 910 + inner as u64);
        check(dev, &[x.clone(), idx_i64.clone(), src], 1e-5, 1e-5, &format!("iadd i64 {label}"), |v| {
            v[0].index_add(1, &v[1], &v[2])
        });
        let src_u = randn(&[outer, 4, inner], 920 + inner as u64);
        check(dev, &[x.clone(), idx_u32, src_u], 1e-5, 1e-5, &format!("iadd u32 {label}"), |v| {
            v[0].index_add(1, &v[1], &v[2])
        });

        // Gather + gather_backward with collisions across both I64 and U32 indices.
        let di = 3usize;
        let g_vals: Vec<i64> = (0..outer * di * inner).map(|k| ((k * 3 + 1) % d) as i64).collect();
        let gi64 = Tensor::from_vec(g_vals.clone(), [outer, di, inner]).unwrap();
        let gu32 =
            Tensor::from_vec(g_vals.into_iter().map(|v| v as u32).collect(), [outer, di, inner]).unwrap();
        check(dev, &[x.clone(), gi64], 0.0, 1e-5, &format!("gather i64 {label}"), |v| v[0].gather(1, &v[1]));
        check(dev, &[x, gu32], 0.0, 1e-5, &format!("gather u32 {label}"), |v| v[0].gather(1, &v[1]));
    }

    // Dispatch trace check: inner=16 dispatches `gather_vec4` and `index_add_vec4`, and `index_select_backward` /
    // `gather_backward` run in a single kernel dispatch (zero extra `fill`/`move` copy dispatch).
    let fresh = crate::GpuContext::new().unwrap();
    let backend = crate::GpuBackend::new(fresh.clone(), dev);
    let xs = Shape::new(&[2, 8, 16]);
    let gs = Shape::new(&[2, 4, 16]);
    let is_1d = Shape::new(&[4]);
    let x_buf = backend.full(dev, DType::F32, xs.numel(), 1.0).unwrap();
    let g_buf = backend.full(dev, DType::F32, gs.numel(), 0.5).unwrap();
    let i_1d = backend.full(dev, DType::U32, is_1d.numel(), 2.0).unwrap();
    let i_3d = backend.full(dev, DType::U32, gs.numel(), 3.0).unwrap();

    fresh.trace_begin();
    let _ = backend
        .gather(
            View::new(&x_buf, &Layout::contiguous(xs.clone())),
            1,
            View::new(&i_3d, &Layout::contiguous(gs.clone())),
        )
        .unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["gather_vec4"]);

    fresh.trace_begin();
    let _ = backend
        .index_select_backward(
            View::new(&g_buf, &Layout::contiguous(gs.clone())),
            1,
            View::new(&i_1d, &Layout::contiguous(is_1d)),
            &xs,
        )
        .unwrap();
    assert_eq!(fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(), vec!["clear", "index_add_vec4"]);

    fresh.trace_begin();
    let _ = backend
        .gather_backward(
            View::new(&g_buf, &Layout::contiguous(gs.clone())),
            1,
            View::new(&i_3d, &Layout::contiguous(gs)),
            &xs,
        )
        .unwrap();
    assert_eq!(
        fresh.trace_end().iter().map(|o| o.name).collect::<Vec<_>>(),
        vec!["clear", "scatter_add_vec4"]
    );

    // Direct kernel test for out-of-range / negative I64 and out-of-range U32 indices on `gather_vec4` /
    // `index_add_vec4` (`inner = 4`): bad lanes must produce 0 in `gather` and be skipped in `index_add`.
    let src_data: Vec<f32> = (1..=32).map(|v| v as f32).collect();
    let x_raw = fresh.upload(bytemuck::cast_slice(&src_data)).unwrap();
    let bad_i64: Vec<i64> = vec![0, -1, i64::MIN, 1i64 << 33, 4, 3, 2, -5]; // [outer=1, di=2, inner=4]
    let bad_i64_buf = fresh.upload(bytemuck::cast_slice(&bad_i64)).unwrap();
    let g_out = fresh.alloc(8 * 4).unwrap();
    crate::kernels::strided::gather(
        &fresh,
        crate::kernels::strided::Elem::F32,
        x_raw.raw(),
        (1, 4, 4),
        crate::kernels::strided::Indices { buf: bad_i64_buf.raw(), i64: true, len: 8 },
        2,
        g_out.raw(),
    )
    .unwrap();
    let got_g: Vec<f32> = bytemuck::pod_collect_to_vec(&fresh.read(g_out.raw(), 0, 8 * 4).unwrap());
    // row 0: [x[0,0]=1, BAD=0, BAD=0, BAD=0]; row 1: [BAD=0, x[3,1]=14, x[2,2]=11, BAD=0].
    assert_eq!(got_g, vec![1.0, 0.0, 0.0, 0.0, 0.0, 14.0, 11.0, 0.0]);

    let bad_u32: Vec<u32> = vec![1, 4, u32::MAX, 2]; // [outer=1, di=1, inner=4]
    let bad_u32_buf = fresh.upload(bytemuck::cast_slice(&bad_u32)).unwrap();
    let g_out_u = fresh.alloc(4 * 4).unwrap();
    crate::kernels::strided::gather(
        &fresh,
        crate::kernels::strided::Elem::F32,
        x_raw.raw(),
        (1, 4, 4),
        crate::kernels::strided::Indices { buf: bad_u32_buf.raw(), i64: false, len: 4 },
        1,
        g_out_u.raw(),
    )
    .unwrap();
    let got_gu: Vec<f32> = bytemuck::pod_collect_to_vec(&fresh.read(g_out_u.raw(), 0, 4 * 4).unwrap());
    assert_eq!(got_gu, vec![5.0, 0.0, 0.0, 12.0]);

    let ia_out = fresh.upload(bytemuck::cast_slice(&[0.0f32; 16])).unwrap();
    let ia_src = fresh.upload(bytemuck::cast_slice(&[1.0f32; 16])).unwrap();
    let ia_idx_i64 = fresh.upload(bytemuck::cast_slice(&[0i64, -1, 4, 1i64 << 33])).unwrap();
    crate::kernels::strided::index_add(
        &fresh,
        ia_out.raw(),
        (1, 4, 4),
        crate::kernels::strided::Indices { buf: ia_idx_i64.raw(), i64: true, len: 4 },
        ia_src.raw(),
    )
    .unwrap();
    let got_ia: Vec<f32> = bytemuck::pod_collect_to_vec(&fresh.read(ia_out.raw(), 0, 16 * 4).unwrap());
    assert_eq!(got_ia, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Direct `scatter_add_vec4` test with collisions and out-of-range I64 & U32 indices (`inner = 4`).
    let sa_out = fresh.upload(bytemuck::cast_slice(&[0.0f32; 16])).unwrap();
    let sa_src =
        fresh.upload(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])).unwrap();
    let sa_idx_i64: Vec<i64> = vec![0, -1, 2, 1i64 << 33, 0, 3, 2, 4];
    let sa_idx_buf = fresh.upload(bytemuck::cast_slice(&sa_idx_i64)).unwrap();
    crate::kernels::strided::scatter_add(
        &fresh,
        sa_out.raw(),
        (1, 4, 4),
        crate::kernels::strided::Indices { buf: sa_idx_buf.raw(), i64: true, len: 8 },
        2,
        sa_src.raw(),
    )
    .unwrap();
    let got_sa: Vec<f32> = bytemuck::pod_collect_to_vec(&fresh.read(sa_out.raw(), 0, 16 * 4).unwrap());
    assert_eq!(
        got_sa,
        vec![11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33.0, 0.0, 0.0, 20.0, 0.0, 0.0]
    );
    let sa_out_u = fresh.upload(bytemuck::cast_slice(&[0.0f32; 16])).unwrap();
    let sa_idx_u32: Vec<u32> = vec![1, 4, u32::MAX, 2];
    let sa_idx_u_buf = fresh.upload(bytemuck::cast_slice(&sa_idx_u32)).unwrap();
    crate::kernels::strided::scatter_add(
        &fresh,
        sa_out_u.raw(),
        (1, 4, 4),
        crate::kernels::strided::Indices { buf: sa_idx_u_buf.raw(), i64: false, len: 4 },
        1,
        sa_src.raw(),
    )
    .unwrap();
    let got_sau: Vec<f32> = bytemuck::pod_collect_to_vec(&fresh.read(sa_out_u.raw(), 0, 16 * 4).unwrap());
    assert_eq!(got_sau, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);

    // Backend validation on direct backward hooks: invalid grad dtype (U32), wrong idx rank (no panic),
    // mismatched grad shape, and invalid idx dtype even when output shape is empty.
    let u32_s = backend.full(dev, DType::U32, 128, 1.0).unwrap();
    let f32_s = backend.full(dev, DType::F32, 128, 1.0).unwrap();
    let l_2416 = Layout::contiguous(Shape::new(&[2, 4, 16]));
    let l_1d = Layout::contiguous(Shape::new(&[4]));
    let l_bad_grad = Layout::contiguous(Shape::new(&[2, 3, 16]));
    assert!(
        backend.index_select_backward(View::new(&u32_s, &l_2416), 1, View::new(&u32_s, &l_1d), &xs).is_err()
    );
    assert!(
        backend
            .index_select_backward(View::new(&f32_s, &l_bad_grad), 1, View::new(&u32_s, &l_1d), &xs)
            .is_err()
    );
    assert!(
        backend
            .index_select_backward(
                View::new(&f32_s, &Layout::contiguous(Shape::new(&[0, 4, 16]))),
                1,
                View::new(&f32_s, &l_1d),
                &Shape::new(&[0, 8, 16]),
            )
            .is_err()
    );
    assert!(backend.gather_backward(View::new(&f32_s, &l_2416), 1, View::new(&u32_s, &l_1d), &xs).is_err());
    assert!(backend.gather_backward(View::new(&u32_s, &l_2416), 1, View::new(&u32_s, &l_2416), &xs).is_err());
    assert!(
        backend.gather_backward(View::new(&f32_s, &l_bad_grad), 1, View::new(&u32_s, &l_2416), &xs).is_err()
    );
    assert!(
        backend
            .gather_backward(
                View::new(&f32_s, &Layout::contiguous(Shape::new(&[0, 4, 16]))),
                1,
                View::new(&f32_s, &Layout::contiguous(Shape::new(&[0, 4, 16]))),
                &Shape::new(&[0, 8, 16]),
            )
            .is_err()
    );
    assert!(fresh.kernels_without_registry().is_empty());
}
