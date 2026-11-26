//! Reductions along every dim, softmax and log-softmax.

use super::{Rng, assert_close, gpu};
use crate::GpuTensor;
use crate::kernels::reduce::ReduceOp;

fn cpu_reduce(op: ReduceOp, x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let outer: usize = shape[..dim].iter().product();
    let r = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let vals = (0..r).map(|k| x[(o * r + k) * inner + i] as f64);
            out.push(match op {
                ReduceOp::Sum => vals.sum::<f64>() as f32,
                ReduceOp::Mean => (vals.sum::<f64>() / r as f64) as f32,
                ReduceOp::Max => vals.fold(f64::NEG_INFINITY, f64::max) as f32,
                ReduceOp::Min => vals.fold(f64::INFINITY, f64::min) as f32,
            });
        }
    }
    out
}

#[test]
fn reductions_along_each_dim() {
    let ctx = gpu!();
    let mut rng = Rng::new(5);
    for shape in [vec![3, 70, 5], vec![2, 1000], vec![513], vec![4, 3, 2, 67], vec![1, 300, 1]] {
        let x = rng.vec(shape.iter().product(), -2.0, 2.0);
        let gx = GpuTensor::from_slice(&ctx, &x, &shape).unwrap();
        for dim in 0..shape.len() {
            for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
                let want = cpu_reduce(op, &x, &shape, dim);
                let got = gx.reduce(op, dim, false).unwrap();
                let mut want_shape = shape.clone();
                want_shape.remove(dim);
                assert_eq!(got.shape(), want_shape.as_slice());
                // Sums of up to 1000 values in [-2, 2] in a different order: 1e-4 absolute.
                assert_close(
                    &got.to_vec().unwrap(),
                    &want,
                    1e-4,
                    1e-5,
                    &format!("{op:?} {shape:?} dim {dim}"),
                );
            }
        }
    }
    let gx = GpuTensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(gx.sum(1, true).unwrap().shape(), &[2, 1]);
    assert_eq!(gx.max(0, false).unwrap().to_vec().unwrap(), vec![3.0, 4.0]);
    assert!(gx.sum(2, false).is_err());
}

#[test]
fn max_handles_infinities() {
    let ctx = gpu!();
    let x = [f32::NEG_INFINITY, f32::NEG_INFINITY, -1.0, f32::INFINITY];
    let gx = GpuTensor::from_slice(&ctx, &x, &[2, 2]).unwrap();
    assert_eq!(gx.max(1, false).unwrap().to_vec().unwrap(), vec![f32::NEG_INFINITY, f32::INFINITY]);
    assert_eq!(
        gx.reduce(ReduceOp::Min, 0, false).unwrap().to_vec().unwrap(),
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY]
    );
}

fn cpu_softmax(x: &[f32], n: usize, log: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(n) {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        for &v in row {
            out.push(if log { (v as f64 - m - s.ln()) as f32 } else { ((v as f64 - m).exp() / s) as f32 });
        }
    }
    out
}

#[test]
fn softmax_and_log_softmax() {
    let ctx = gpu!();
    let mut rng = Rng::new(6);
    for (rows, n) in [(1, 1), (7, 80), (3, 1000), (130, 16), (2, 257)] {
        let mut x = rng.vec(rows * n, -10.0, 10.0);
        // Large logits must not overflow.
        x[0] = 500.0;
        let gx = GpuTensor::from_slice(&ctx, &x, &[rows, n]).unwrap();
        let got = gx.softmax().unwrap().to_vec().unwrap();
        assert_close(&got, &cpu_softmax(&x, n, false), 1e-6, 1e-5, &format!("softmax {rows}x{n}"));
        let got = gx.log_softmax().unwrap().to_vec().unwrap();
        assert_close(&got, &cpu_softmax(&x, n, true), 1e-4, 1e-5, &format!("log_softmax {rows}x{n}"));
    }
}

#[test]
fn optim_norm_partials_and_unscale_subgroups_match_portable() {
    use crate::kernels::optim::{
        NormPartial, norm_partials_len, norm_partials_many, unscale_check, unscale_check_many,
    };

    struct ForceGuard(bool);
    impl ForceGuard {
        fn set(v: bool) -> Self {
            Self(crate::kernels::FORCE_NO_SUBGROUPS.replace(v))
        }
    }
    impl Drop for ForceGuard {
        fn drop(&mut self) {
            crate::kernels::FORCE_NO_SUBGROUPS.set(self.0);
        }
    }

    let ctx = gpu!();
    let mut rng = Rng::new(42);
    let t0 = rng.vec(300, -3.0, 3.0);
    let mut t1 = rng.vec(600, -5.0, 5.0);
    t1[513] = f32::NAN;
    let t2 = rng.vec(17, -2.0, 2.0);
    let b0 = ctx.upload(bytemuck::cast_slice(&t0)).unwrap();
    let b1 = ctx.upload(bytemuck::cast_slice(&t1)).unwrap();
    let b2 = ctx.upload(bytemuck::cast_slice(&t2)).unwrap();
    let h_buf = ctx.upload(bytemuck::cast_slice(&[0.5f32])).unwrap();
    let xs = [(b0.raw(), t0.len()), (b1.raw(), t1.len()), (b2.raw(), t2.len())];
    let plen = norm_partials_len(&[t0.len(), t1.len(), t2.len()]);

    let run_norm = |op: NormPartial, portable: bool| -> Vec<f32> {
        let _g = ForceGuard::set(portable);
        let part = ctx.alloc((plen * 4) as u64).unwrap();
        norm_partials_many(
            &ctx,
            op,
            &xs,
            (op == NormPartial::ScaledSquares).then_some(h_buf.raw()),
            part.raw(),
        )
        .unwrap();
        bytemuck::pod_collect_to_vec(&ctx.read(part.raw(), 0, (plen * 4) as u64).unwrap())
    };

    for op in [NormPartial::AbsMax, NormPartial::ScaledSquares] {
        let sg = run_norm(op, false);
        let fb = run_norm(op, true);
        assert_eq!(sg.len(), fb.len());
        for (i, (&a, &b)) in sg.iter().zip(&fb).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            let err = (a - b).abs();
            let lim = 1e-4 + 1e-5 * b.abs();
            assert!(err <= lim, "{op:?} partial[{i}]: sg={a} fb={b} err={err}");
        }
        // Workgroup containing t1[513] = NaN (index 2 + 2 = 4) must produce NaN on both paths.
        assert!(sg[4].is_nan() && fb[4].is_nan(), "{op:?} must propagate NaN in partial[4]");
    }

    // unscale_check and unscale_check_many: finite case (flag 0), full overflow (flag 1), and a single NaN in a
    // non-leader lane of the final partial tail subgroup (verifying both the flag and the scaled buffer contents).
    let mut tail_nan = vec![2.0f32; 300];
    tail_nan[299] = f32::NAN;
    let mut small_tail_nan = vec![3.0f32; 19];
    small_tail_nan[18] = f32::NAN;

    for (x0_in, x1_in, factor, expect_single, expect_many) in [
        (vec![1.0f32; 300], vec![2.0f32; 19], 0.5f32, 0u32, 0u32),
        (vec![1e38f32; 300], vec![2.0f32; 19], 10.0f32, 1u32, 1u32),
        (tail_nan, vec![2.0f32; 19], 0.5f32, 1u32, 1u32),
        (vec![1.0f32; 300], small_tail_nan, 0.5f32, 0u32, 1u32),
    ] {
        let run_single = |portable: bool| -> (u32, Vec<f32>) {
            let _g = ForceGuard::set(portable);
            let x = ctx.upload(bytemuck::cast_slice(&x0_in)).unwrap();
            let flag = ctx.upload(bytemuck::cast_slice(&[0u32])).unwrap();
            unscale_check(&ctx, x.raw(), x0_in.len(), factor, flag.raw()).unwrap();
            let got_flag: Vec<u32> = bytemuck::pod_collect_to_vec(&ctx.read(flag.raw(), 0, 4).unwrap());
            let got_x: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(x.raw(), 0, (x0_in.len() * 4) as u64).unwrap());
            (got_flag[0], got_x)
        };
        let (fs, xs_sg) = run_single(false);
        let (fp, xs_fb) = run_single(true);
        assert_eq!(fs, expect_single, "unscale_check sg flag factor={factor}");
        assert_eq!(fp, expect_single, "unscale_check fb flag factor={factor}");
        for (i, (&a, &b)) in xs_sg.iter().zip(&xs_fb).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "unscale_check[{i}]: {a} vs {b}"
            );
        }

        let run_many = |portable: bool| -> (u32, Vec<f32>, Vec<f32>) {
            let _g = ForceGuard::set(portable);
            let x0 = ctx.upload(bytemuck::cast_slice(&x0_in)).unwrap();
            let x1 = ctx.upload(bytemuck::cast_slice(&x1_in)).unwrap();
            let flag = ctx.upload(bytemuck::cast_slice(&[0u32])).unwrap();
            unscale_check_many(&ctx, &[(x0.raw(), x0_in.len()), (x1.raw(), x1_in.len())], factor, flag.raw())
                .unwrap();
            let got_flag: Vec<u32> = bytemuck::pod_collect_to_vec(&ctx.read(flag.raw(), 0, 4).unwrap());
            let g0: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(x0.raw(), 0, (x0_in.len() * 4) as u64).unwrap());
            let g1: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(x1.raw(), 0, (x1_in.len() * 4) as u64).unwrap());
            (got_flag[0], g0, g1)
        };
        let (ms, m0_sg, m1_sg) = run_many(false);
        let (mp, m0_fb, m1_fb) = run_many(true);
        assert_eq!(ms, expect_many, "unscale_check_many sg flag factor={factor}");
        assert_eq!(mp, expect_many, "unscale_check_many fb flag factor={factor}");
        for (i, (&a, &b)) in m0_sg.iter().zip(&m0_fb).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "unscale_many x0[{i}]: {a} vs {b}"
            );
        }
        for (i, (&a, &b)) in m1_sg.iter().zip(&m1_fb).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "unscale_many x1[{i}]: {a} vs {b}"
            );
        }
    }
}

#[test]
fn reduce_split_ceil_div_and_mean_match_cpu() {
    use crate::DType;
    use crate::kernels::reduce::reduce_split;

    let ctx = gpu!();
    let mut rng = Rng::new(71);
    // Prime / non-divisible `r >= 8192` across both `inner == 1` (rows) and `inner > 1` (cols), all 4 ops.
    for &(outer, r, inner) in &[(2usize, 8193usize, 1usize), (1, 10007, 1), (2, 8193, 5), (2, 10007, 70)] {
        let shape = [outer, r, inner];
        let x = rng.vec(outer * r * inner, -2.0, 2.0);
        let bx = ctx.upload(bytemuck::cast_slice(&x)).unwrap();
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            let want = cpu_reduce(op, &x, &shape, 1);
            let by = ctx.alloc((outer * inner * 4) as u64).unwrap();
            reduce_split(&ctx, op, bx.raw(), by.raw(), outer, r, inner, DType::F32).unwrap();
            let got: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(by.raw(), 0, (outer * inner * 4) as u64).unwrap());
            let atol = if op == ReduceOp::Sum { 2e-3 } else { 1e-4 };
            assert_close(&got, &want, atol, 1e-4, &format!("reduce_split {op:?} {shape:?}"));
        }
        // NaN placed in the final element of the ragged last split chunk must propagate for all 4 ops.
        let mut x_nan = x.clone();
        let last_idx = ((outer - 1) * r + (r - 1)) * inner + (inner - 1);
        x_nan[last_idx] = f32::NAN;
        let bx_nan = ctx.upload(bytemuck::cast_slice(&x_nan)).unwrap();
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            let by = ctx.alloc((outer * inner * 4) as u64).unwrap();
            reduce_split(&ctx, op, bx_nan.raw(), by.raw(), outer, r, inner, DType::F32).unwrap();
            let got: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(by.raw(), 0, (outer * inner * 4) as u64).unwrap());
            assert!(got[outer * inner - 1].is_nan(), "reduce_split {op:?} {shape:?} missed tail NaN");
            if outer * inner > 1 {
                assert!(got[0].is_finite(), "reduce_split {op:?} {shape:?} leaked NaN to output 0");
            }
        }
    }
    // F16 storage round-trip on prime-r rows & cols (Sum, Mean, Max, Min) when shader-f16 is supported.
    if ctx.caps().shader_f16 {
        for &(outer, r, inner) in &[(2usize, 8193usize, 1usize), (2, 8193, 5)] {
            let shape = [outer, r, inner];
            let x_raw = rng.vec(outer * r * inner, -1.0, 1.0);
            let x_h: Vec<half::f16> = x_raw.iter().map(|&v| half::f16::from_f32(v)).collect();
            let x_f: Vec<f32> = x_h.iter().map(|v| v.to_f32()).collect();
            let x_bytes: Vec<u8> = x_h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let bx = ctx.upload(&x_bytes).unwrap();
            for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
                let want: Vec<f32> = cpu_reduce(op, &x_f, &shape, 1)
                    .into_iter()
                    .map(|v| half::f16::from_f32(v).to_f32())
                    .collect();
                let by = ctx.alloc(DType::F16.bytes(outer * inner)).unwrap();
                reduce_split(&ctx, op, bx.raw(), by.raw(), outer, r, inner, DType::F16).unwrap();
                let raw = ctx.read(by.raw(), 0, DType::F16.bytes(outer * inner)).unwrap();
                let got: Vec<f32> =
                    raw.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect();
                assert_close(&got, &want, 2e-2, 5e-3, &format!("reduce_split f16 {op:?} {shape:?}"));
            }
        }
    }
}

#[test]
fn reduce_rows_vec4_matches_cpu_and_propagates_nan() {
    use crate::DType;
    use crate::kernels::reduce::reduce_split;

    let ctx = gpu!();
    let mut rng = Rng::new(97);
    // Exercises VEC4 single-pass (r = 4, 256, 1024), VEC4 split-K (r = 8192, 8196, 16388), and empty r = 0.
    for &(outer, r) in &[(3usize, 4usize), (2, 256), (4, 1024), (2, 8192), (2, 8196), (1, 16388)] {
        let shape = [outer, r, 1];
        let x = rng.vec(outer * r, -2.0, 2.0);
        let bx = ctx.upload(bytemuck::cast_slice(&x)).unwrap();
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            let want = cpu_reduce(op, &x, &shape, 1);
            let by = ctx.alloc((outer * 4) as u64).unwrap();
            reduce_split(&ctx, op, bx.raw(), by.raw(), outer, r, 1, DType::F32).unwrap();
            let got: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(by.raw(), 0, (outer * 4) as u64).unwrap());
            let atol = if op == ReduceOp::Sum { 2e-3 } else { 1e-4 };
            assert_close(&got, &want, atol, 1e-4, &format!("reduce_rows vec4 {op:?} {shape:?}"));
        }
        // NaN in every lane (0, 1, 2, 3) of the last vec4 must propagate on every op.
        for lane in 0..4 {
            let mut x_nan = x.clone();
            x_nan[(outer - 1) * r + (r - 4 + lane)] = f32::NAN;
            let bx_nan = ctx.upload(bytemuck::cast_slice(&x_nan)).unwrap();
            for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
                let by = ctx.alloc((outer * 4) as u64).unwrap();
                reduce_split(&ctx, op, bx_nan.raw(), by.raw(), outer, r, 1, DType::F32).unwrap();
                let got: Vec<f32> =
                    bytemuck::pod_collect_to_vec(&ctx.read(by.raw(), 0, (outer * 4) as u64).unwrap());
                assert!(got[outer - 1].is_nan(), "vec4 lane {lane} {op:?} {shape:?} missed NaN");
                if outer > 1 {
                    assert!(got[0].is_finite(), "vec4 lane {lane} {op:?} {shape:?} leaked NaN to row 0");
                }
            }
        }
    }
    // F16 VEC4 storage round-trip (single-pass and split-K) when shader-f16 is supported.
    if ctx.caps().shader_f16 {
        for &(outer, r) in &[(3usize, 1024usize), (2, 8196)] {
            let shape = [outer, r, 1];
            let x_raw = rng.vec(outer * r, -1.0, 1.0);
            let x_h: Vec<half::f16> = x_raw.iter().map(|&v| half::f16::from_f32(v)).collect();
            let x_f: Vec<f32> = x_h.iter().map(|v| v.to_f32()).collect();
            let x_bytes: Vec<u8> = x_h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let bx = ctx.upload(&x_bytes).unwrap();
            for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
                let want: Vec<f32> = cpu_reduce(op, &x_f, &shape, 1)
                    .into_iter()
                    .map(|v| half::f16::from_f32(v).to_f32())
                    .collect();
                let by = ctx.alloc(DType::F16.bytes(outer)).unwrap();
                reduce_split(&ctx, op, bx.raw(), by.raw(), outer, r, 1, DType::F16).unwrap();
                let raw = ctx.read(by.raw(), 0, (outer * 2) as u64).unwrap();
                let got: Vec<f32> =
                    raw.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect();
                assert_close(&got, &want, 2e-2, 5e-3, &format!("reduce_rows vec4 f16 {op:?} {shape:?}"));
            }
        }
    }
    assert!(ctx.kernels_without_registry().is_empty());
}

#[test]
fn detect_cached_anchors_and_subgroup_loss_match_portable() {
    use crate::detect::{
        AnchorTable, BoxLossKind, ClsLossKind, ImageGt, LossSettings, TalSettings, detection_loss,
    };
    use folo_core::{Device, Tensor};

    struct ForceGuard(bool);
    impl ForceGuard {
        fn set(v: bool) -> Self {
            Self(crate::kernels::FORCE_NO_SUBGROUPS.replace(v))
        }
    }
    impl Drop for ForceGuard {
        fn drop(&mut self) {
            crate::kernels::FORCE_NO_SUBGROUPS.set(self.0);
        }
    }

    let _shared = gpu!();
    // Private context on ordinal 8 to verify anchor cache lifecycle and release_memory cleanly.
    let ctx = match crate::GpuContext::new() {
        Ok(c) => c,
        Err(e) => return eprintln!("note: skipping, no second context: {e}"),
    };
    let dev = Device::Gpu(8);
    crate::register(ctx.clone(), dev).unwrap();

    // 1. Anchor cache hit, distinct-table miss, LRU bound (cap 8), and release_memory clear.
    let pts: Vec<[f32; 2]> = (0..1200).map(|i| [(i % 40) as f32 + 0.5, (i / 40) as f32 + 0.5]).collect();
    let strides = vec![8.0f32; pts.len()];
    let b1 = ctx.cached_anchors(&pts, &strides).unwrap();
    let b2 = ctx.cached_anchors(&pts, &strides).unwrap();
    assert_eq!(b1, b2, "identical anchors must hit the device cache");
    assert_eq!(ctx.anchor_cache_len(), 1);
    for k in 0..10 {
        let p: Vec<[f32; 2]> = vec![[k as f32 + 0.5, 0.5]; 16];
        let s = vec![8.0f32; 16];
        let _ = ctx.cached_anchors(&p, &s).unwrap();
    }
    assert!(ctx.anchor_cache_len() <= 8, "anchor cache must stay bounded by MAX_ANCHOR_CACHE");
    ctx.release_memory().unwrap();
    assert_eq!(ctx.anchor_cache_len(), 0, "release_memory must clear cached anchors");

    // 2. End-to-end detection_loss subgroup vs portable parity (plain BCE, MAL + OT-DFL, and non-finite logit).
    let mut rng = Rng::new(1234);
    let (b, a, nc, reg_max) = (2usize, pts.len(), 3usize, 16usize);
    let box_data = rng.vec(b * 4 * reg_max * a, -1.5, 1.5);
    let cls_data = rng.vec(b * nc * a, -2.0, 2.0);
    let box_t = Tensor::from_slice(&box_data, [b, 4 * reg_max, a]).unwrap().to_device(dev).unwrap();
    let cls_t = Tensor::from_slice(&cls_data, [b, nc, a]).unwrap().to_device(dev).unwrap();
    let anchors = AnchorTable { points: &pts, strides: &strides };
    let gt0_boxes = [[16.0f32, 16.0, 96.0, 88.0], [120.0, 40.0, 220.0, 160.0]];
    let gt0_labels = [0u32, 2u32];
    let gt1_boxes = [[32.0f32, 48.0, 160.0, 180.0]];
    let gt1_labels = [1u32];
    let gts = [
        ImageGt { boxes: &gt0_boxes, labels: &gt0_labels },
        ImageGt { boxes: &gt1_boxes, labels: &gt1_labels },
    ];
    let tal = TalSettings {
        topk: 10,
        topk2: None,
        alpha: 0.5,
        beta: 6.0,
        eps: 1e-9,
        min_box_size: 16.0,
        eps_free: false,
        nwd_weight: 0.0,
    };

    for (mal_gamma, dfl_ot_weight) in [(0.0f32, 0.0f32), (1.5, 0.25)] {
        let settings = LossSettings {
            reg_max,
            gains: [7.5, 0.5, 1.5],
            tal,
            mal_gamma,
            cls_loss: ClsLossKind::Bce,
            vfl_alpha: 0.75,
            vfl_gamma: 2.0,
            qfl_beta: 2.0,
            nwd_weight: 0.0,
            box_loss: BoxLossKind::Ciou,
            rpiou_beta: 0.5,
            rpiou_kappa: 2.0,
            rpiou_delta: 0.5,
            dfl_ot_weight,
            dfl_crps_weight: 0.0,
        };
        let run = |portable: bool| -> Vec<f32> {
            let _g = ForceGuard::set(portable);
            let loss = detection_loss(&box_t, &cls_t, &anchors, &gts, &settings).unwrap();
            loss.stats.to_vec::<f32>().unwrap()
        };
        let sg = run(false);
        let fb = run(true);
        assert_close(&sg, &fb, 1e-4, 1e-5, &format!("detect sg vs fb mal={mal_gamma} ot={dfl_ot_weight}"));
        assert!(sg[0].is_finite() && sg[4] > 0.0, "expected finite loss with positives, got {sg:?}");
    }
    assert_eq!(ctx.anchor_cache_len(), 1, "repeated detection_loss steps must reuse the cached anchor table");

    // Non-finite logit in a non-leader lane of the second chunk must trip MISC[0] and return NaN on both paths.
    let mut bad_cls = cls_data;
    bad_cls[1199] = f32::INFINITY;
    let bad_cls_t = Tensor::from_slice(&bad_cls, [b, nc, a]).unwrap().to_device(dev).unwrap();
    let settings = LossSettings {
        reg_max,
        gains: [7.5, 0.5, 1.5],
        tal,
        mal_gamma: 0.0,
        cls_loss: ClsLossKind::Bce,
        vfl_alpha: 0.75,
        vfl_gamma: 2.0,
        qfl_beta: 2.0,
        nwd_weight: 0.0,
        box_loss: BoxLossKind::Ciou,
        rpiou_beta: 0.5,
        rpiou_kappa: 2.0,
        rpiou_delta: 0.5,
        dfl_ot_weight: 0.0,
        dfl_crps_weight: 0.0,
    };
    for portable in [false, true] {
        let _g = ForceGuard::set(portable);
        let stats = detection_loss(&box_t, &bad_cls_t, &anchors, &gts, &settings)
            .unwrap()
            .stats
            .to_vec::<f32>()
            .unwrap();
        assert!(
            stats[0].is_nan() && stats[6] == 1.0,
            "portable={portable} missed non-finite logit: {stats:?}"
        );
    }
    assert!(ctx.kernels_without_registry().is_empty());
}

#[test]
fn reduce_cols_vec4_matches_cpu_and_propagates_nan() {
    use crate::DType;
    use crate::kernels::reduce::reduce_split;

    let ctx = gpu!();
    let mut rng = Rng::new(131);
    // Exercises VEC4 single-pass (`inner = 4, 260, 512`) and VEC4 split-K (`r >= 8192`).
    for &(outer, r, inner) in
        &[(3usize, 17usize, 4usize), (2, 70, 260), (2, 128, 512), (2, 8193, 4), (2, 8193, 260)]
    {
        let shape = [outer, r, inner];
        let x = rng.vec(outer * r * inner, -2.0, 2.0);
        let bx = ctx.upload(bytemuck::cast_slice(&x)).unwrap();
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            let want = cpu_reduce(op, &x, &shape, 1);
            let by = ctx.alloc((outer * inner * 4) as u64).unwrap();
            reduce_split(&ctx, op, bx.raw(), by.raw(), outer, r, inner, DType::F32).unwrap();
            let got: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(by.raw(), 0, (outer * inner * 4) as u64).unwrap());
            let atol = if op == ReduceOp::Sum { 2e-3 } else { 1e-4 };
            assert_close(&got, &want, atol, 1e-4, &format!("reduce_cols vec4 {op:?} {shape:?}"));
        }
        // NaN in every lane (0..4) of the final vec4 column must propagate ONLY to that column lane.
        for lane in 0..4 {
            let mut x_nan = x.clone();
            let col = inner - 4 + lane;
            x_nan[((outer - 1) * r + (r - 1)) * inner + col] = f32::NAN;
            let bx_nan = ctx.upload(bytemuck::cast_slice(&x_nan)).unwrap();
            for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
                let by = ctx.alloc((outer * inner * 4) as u64).unwrap();
                reduce_split(&ctx, op, bx_nan.raw(), by.raw(), outer, r, inner, DType::F32).unwrap();
                let got: Vec<f32> =
                    bytemuck::pod_collect_to_vec(&ctx.read(by.raw(), 0, (outer * inner * 4) as u64).unwrap());
                let out_base = (outer - 1) * inner + (inner - 4);
                for l in 0..4 {
                    if l == lane {
                        assert!(
                            got[out_base + l].is_nan(),
                            "cols vec4 lane {lane} {op:?} {shape:?} missed NaN"
                        );
                    } else {
                        assert!(
                            got[out_base + l].is_finite(),
                            "cols vec4 lane {lane} {op:?} {shape:?} leaked NaN to lane {l}"
                        );
                    }
                }
            }
        }
    }
    if ctx.caps().shader_f16 {
        for &(outer, r, inner) in &[(2usize, 70usize, 260usize), (2, 8193, 16)] {
            let shape = [outer, r, inner];
            let x_raw = rng.vec(outer * r * inner, -1.0, 1.0);
            let x_h: Vec<half::f16> = x_raw.iter().map(|&v| half::f16::from_f32(v)).collect();
            let x_f: Vec<f32> = x_h.iter().map(|v| v.to_f32()).collect();
            let x_bytes: Vec<u8> = x_h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let bx = ctx.upload(&x_bytes).unwrap();
            for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
                let want: Vec<f32> = cpu_reduce(op, &x_f, &shape, 1)
                    .into_iter()
                    .map(|v| half::f16::from_f32(v).to_f32())
                    .collect();
                let by = ctx.alloc(DType::F16.bytes(outer * inner)).unwrap();
                reduce_split(&ctx, op, bx.raw(), by.raw(), outer, r, inner, DType::F16).unwrap();
                let raw = ctx.read(by.raw(), 0, DType::F16.bytes(outer * inner)).unwrap();
                let got: Vec<f32> =
                    raw.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect();
                assert_close(&got, &want, 2e-2, 5e-3, &format!("reduce_cols vec4 f16 {op:?} {shape:?}"));
            }
        }
    }
    // Empty `r == 0` column reduction with a tightly sized 4-byte raw storage buffer must stay scalar so WebGPU
    // binding validation (16 B minimum for `array<vec4<f32>>`) never rejects it.
    let tiny_x = ctx.device().unwrap().create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce_cols_empty_r"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let by_empty = ctx.alloc(16).unwrap();
    reduce_split(&ctx, ReduceOp::Sum, &tiny_x, by_empty.raw(), 1, 0, 4, DType::F32).unwrap();
    let got_empty: Vec<f32> = bytemuck::pod_collect_to_vec(&ctx.read(by_empty.raw(), 0, 16).unwrap());
    assert_eq!(got_empty, [0.0; 4]);
    assert!(ctx.kernels_without_registry().is_empty());
}

fn cpu_optim_step(
    u: &crate::kernels::optim::OptimUpdate,
    p: &mut [f32],
    g: &[f32],
    mut s0: Option<&mut [f32]>,
    mut s1: Option<&mut [f32]>,
) {
    use crate::kernels::optim::OptimUpdate::*;
    for i in 0..p.len() {
        let gi = g[i];
        let mut pi = p[i];
        match *u {
            Sgd { lr, momentum, weight_decay, nesterov } => {
                let mut d = if weight_decay != 0.0 { gi + weight_decay * pi } else { gi };
                if let Some(buf) = s0.as_deref_mut() {
                    let b = momentum * buf[i] + d;
                    buf[i] = b;
                    d = if nesterov { d + momentum * b } else { b };
                }
                pi += (-lr) * d;
            }
            SgdDecoupled { lr, momentum, wd_step, nesterov, cautious } => {
                let u_val = if let Some(buf) = s0.as_deref_mut() {
                    let b = momentum * buf[i] + gi;
                    buf[i] = b;
                    if nesterov { gi + momentum * b } else { b }
                } else {
                    gi
                };
                let decay =
                    if wd_step != 0.0 && (!cautious || u_val * pi > 0.0) { wd_step * pi } else { 0.0 };
                pi = pi - lr * u_val - decay;
            }
            AdamW { decay, beta1, one_minus_beta1, beta2, one_minus_beta2, step_size, bc2_sqrt, eps } => {
                if let Some(d) = decay {
                    pi *= d;
                }
                let b0 = s0.as_deref_mut().unwrap();
                let b1 = s1.as_deref_mut().unwrap();
                let m = beta1 * b0[i] + one_minus_beta1 * gi;
                let v = beta2 * b1[i] + one_minus_beta2 * (gi * gi);
                b0[i] = m;
                b1[i] = v;
                let denom = v.sqrt() / bc2_sqrt + eps;
                pi += (-step_size) * (m / denom);
            }
            Axpby { a, b } => {
                pi = pi * a + gi * b;
            }
        }
        p[i] = pi;
    }
}

#[test]
fn optim_vec4_and_mixed_many_match_cpu() {
    use crate::kernels::optim::{
        ManyItem, NormPartial, OptimUpdate, norm_partials_len, norm_partials_many, optimizer_step,
        optimizer_step_many, scale_many, unscale_check, unscale_check_many,
    };

    struct ForceGuard(bool);
    impl ForceGuard {
        fn set(v: bool) -> Self {
            Self(crate::kernels::FORCE_NO_SUBGROUPS.replace(v))
        }
    }
    impl Drop for ForceGuard {
        fn drop(&mut self) {
            crate::kernels::FORCE_NO_SUBGROUPS.set(self.0);
        }
    }

    let ctx = gpu!();
    let mut rng = Rng::new(211);
    let updates: &[(OptimUpdate, usize, &str)] = &[
        (OptimUpdate::Sgd { lr: 0.01, momentum: 0.0, weight_decay: 0.05, nesterov: false }, 0, "sgd0"),
        (OptimUpdate::Sgd { lr: 0.01, momentum: 0.9, weight_decay: 0.05, nesterov: true }, 1, "sgd1_nes"),
        (
            OptimUpdate::SgdDecoupled {
                lr: 0.02,
                momentum: 0.0,
                wd_step: 0.001,
                nesterov: false,
                cautious: false,
            },
            0,
            "sgdd0",
        ),
        (
            OptimUpdate::SgdDecoupled {
                lr: 0.02,
                momentum: 0.85,
                wd_step: 0.001,
                nesterov: true,
                cautious: false,
            },
            1,
            "sgdd1_nes",
        ),
        (
            OptimUpdate::SgdDecoupled {
                lr: 0.02,
                momentum: 0.0,
                wd_step: 0.003,
                nesterov: false,
                cautious: true,
            },
            0,
            "sgdc0",
        ),
        (
            OptimUpdate::SgdDecoupled {
                lr: 0.02,
                momentum: 0.85,
                wd_step: 0.003,
                nesterov: true,
                cautious: true,
            },
            1,
            "sgdc1_nes",
        ),
        (
            OptimUpdate::AdamW {
                decay: Some(0.998),
                beta1: 0.9,
                one_minus_beta1: 0.1,
                beta2: 0.999,
                one_minus_beta2: 0.001,
                step_size: 1e-3,
                bc2_sqrt: 0.95,
                eps: 1e-8,
            },
            2,
            "adamw",
        ),
        (OptimUpdate::Axpby { a: 0.999, b: 0.001 }, 0, "axpby"),
    ];

    // 1. Single-tensor optimizer_step (vec4 n=1028 and scalar n=1027) and mixed vec4+scalar optimizer_step_many.
    for &(u, ns, label) in updates {
        for n in [1028usize, 1027] {
            let mut p_ref = rng.vec(n, -2.0, 2.0);
            let mut g_in = rng.vec(n, -2.0, 2.0);
            // Force opposing signs and nonzero `p` with zero `u` (`u * p == 0.0`) inside the same vec4 for cautious SGD.
            p_ref[0] = 1.5;
            g_in[0] = 0.5;
            p_ref[1] = 1.5;
            g_in[1] = -0.5;
            p_ref[2] = 1.5;
            g_in[2] = 0.0;
            p_ref[3] = -1.5;
            g_in[3] = -0.5;
            let mut s0_ref = (ns >= 1).then(|| {
                let mut v = rng.vec(n, -1.0, 1.0);
                v[2] = 0.0;
                v
            });
            let mut s1_ref = (ns >= 2).then(|| rng.vec(n, 0.01, 2.0));
            let bp = ctx.upload(bytemuck::cast_slice(&p_ref)).unwrap();
            let bg = ctx.upload(bytemuck::cast_slice(&g_in)).unwrap();
            let bs0 = s0_ref.as_ref().map(|v| ctx.upload(bytemuck::cast_slice(v)).unwrap());
            let bs1 = s1_ref.as_ref().map(|v| ctx.upload(bytemuck::cast_slice(v)).unwrap());
            cpu_optim_step(&u, &mut p_ref, &g_in, s0_ref.as_deref_mut(), s1_ref.as_deref_mut());
            optimizer_step(
                &ctx,
                &u,
                bp.raw(),
                bg.raw(),
                bs0.as_ref().map(|b| b.raw()),
                bs1.as_ref().map(|b| b.raw()),
                n,
            )
            .unwrap();
            let got_p: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(bp.raw(), 0, (n * 4) as u64).unwrap());
            assert_close(&got_p, &p_ref, 1e-6, 1e-6, &format!("optim_step {label} n={n} p"));
            if let (Some(b), Some(r)) = (&bs0, &s0_ref) {
                let got: Vec<f32> =
                    bytemuck::pod_collect_to_vec(&ctx.read(b.raw(), 0, (n * 4) as u64).unwrap());
                assert_close(&got, r, 1e-6, 1e-6, &format!("optim_step {label} n={n} s0"));
            }
            if let (Some(b), Some(r)) = (&bs1, &s1_ref) {
                let got: Vec<f32> =
                    bytemuck::pod_collect_to_vec(&ctx.read(b.raw(), 0, (n * 4) as u64).unwrap());
                assert_close(&got, r, 1e-6, 1e-6, &format!("optim_step {label} n={n} s1"));
            }
        }

        // Exercise both an all-4-aligned chunk (VEC4) and a mixed chunk (scalar fallback), checking p, s0, and s1.
        for lens in [&[4usize, 1028, 256, 0][..], &[4usize, 1028, 7, 256, 3, 0][..]] {
            let mut ps: Vec<Vec<f32>> = lens.iter().map(|&n| rng.vec(n, -2.0, 2.0)).collect();
            let gs: Vec<Vec<f32>> = lens.iter().map(|&n| rng.vec(n, -2.0, 2.0)).collect();
            let mut s0s: Vec<Option<Vec<f32>>> =
                lens.iter().map(|&n| (ns >= 1).then(|| rng.vec(n, -1.0, 1.0))).collect();
            let mut s1s: Vec<Option<Vec<f32>>> =
                lens.iter().map(|&n| (ns >= 2).then(|| rng.vec(n, 0.01, 2.0))).collect();
            let bps: Vec<_> = ps.iter().map(|v| ctx.upload(bytemuck::cast_slice(v)).unwrap()).collect();
            let bgs: Vec<_> = gs.iter().map(|v| ctx.upload(bytemuck::cast_slice(v)).unwrap()).collect();
            let bs0s: Vec<_> = s0s
                .iter()
                .map(|o| o.as_ref().map(|v| ctx.upload(bytemuck::cast_slice(v)).unwrap()))
                .collect();
            let bs1s: Vec<_> = s1s
                .iter()
                .map(|o| o.as_ref().map(|v| ctx.upload(bytemuck::cast_slice(v)).unwrap()))
                .collect();
            let items: Vec<_> = (0..lens.len())
                .map(|i| ManyItem {
                    p: bps[i].raw(),
                    g: bgs[i].raw(),
                    s0: bs0s[i].as_ref().map(|b| b.raw()),
                    s1: bs1s[i].as_ref().map(|b| b.raw()),
                    n: lens[i],
                })
                .collect();
            optimizer_step_many(&ctx, &u, &items).unwrap();
            for i in 0..lens.len() {
                cpu_optim_step(&u, &mut ps[i], &gs[i], s0s[i].as_deref_mut(), s1s[i].as_deref_mut());
                if lens[i] == 0 {
                    continue;
                }
                let got: Vec<f32> =
                    bytemuck::pod_collect_to_vec(&ctx.read(bps[i].raw(), 0, (lens[i] * 4) as u64).unwrap());
                assert_close(&got, &ps[i], 1e-6, 1e-6, &format!("optim_many {label}[{i}] n={} p", lens[i]));
                if let (Some(b), Some(r)) = (&bs0s[i], &s0s[i]) {
                    let got: Vec<f32> =
                        bytemuck::pod_collect_to_vec(&ctx.read(b.raw(), 0, (lens[i] * 4) as u64).unwrap());
                    assert_close(&got, r, 1e-6, 1e-6, &format!("optim_many {label}[{i}] n={} s0", lens[i]));
                }
                if let (Some(b), Some(r)) = (&bs1s[i], &s1s[i]) {
                    let got: Vec<f32> =
                        bytemuck::pod_collect_to_vec(&ctx.read(b.raw(), 0, (lens[i] * 4) as u64).unwrap());
                    assert_close(&got, r, 1e-6, 1e-6, &format!("optim_many {label}[{i}] n={} s1", lens[i]));
                }
            }
        }
    }

    // 2. unscale_check & unscale_check_many vec4: NaN/Inf in each of the 4 lanes of the last vec4, sg vs portable.
    for lane in 0..4 {
        for bad_val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x_vec4 = vec![2.0f32; 1028];
            x_vec4[1024 + lane] = bad_val;
            let x_vec4_ok = vec![3.0f32; 20];
            for portable in [false, true] {
                let _g = ForceGuard::set(portable);
                let bx = ctx.upload(bytemuck::cast_slice(&x_vec4)).unwrap();
                let flag = ctx.upload(bytemuck::cast_slice(&[0u32])).unwrap();
                unscale_check(&ctx, bx.raw(), x_vec4.len(), 0.5, flag.raw()).unwrap();
                let f: Vec<u32> = bytemuck::pod_collect_to_vec(&ctx.read(flag.raw(), 0, 4).unwrap());
                assert_eq!(f[0], 1, "unscale_check vec4 lane {lane} val={bad_val} portable={portable}");

                let b0 = ctx.upload(bytemuck::cast_slice(&x_vec4)).unwrap();
                let b1 = ctx.upload(bytemuck::cast_slice(&x_vec4_ok)).unwrap();
                let flag_m = ctx.upload(bytemuck::cast_slice(&[0u32])).unwrap();
                unscale_check_many(
                    &ctx,
                    &[(b0.raw(), x_vec4.len()), (b1.raw(), x_vec4_ok.len())],
                    0.5,
                    flag_m.raw(),
                )
                .unwrap();
                let fm: Vec<u32> = bytemuck::pod_collect_to_vec(&ctx.read(flag_m.raw(), 0, 4).unwrap());
                assert_eq!(fm[0], 1, "unscale_check_many vec4 lane {lane} val={bad_val} portable={portable}");
            }
        }
    }

    // 3. scale_many across all-4-aligned (VEC4) and mixed (scalar) chunks (exact two roundings).
    let f1_val = 0.3f32;
    let f2_val = 1.7f32;
    let bf1 = ctx.upload(bytemuck::cast_slice(&[f1_val])).unwrap();
    let bf2 = ctx.upload(bytemuck::cast_slice(&[f2_val])).unwrap();
    for s_lens in [&[4usize, 1028, 256][..], &[4usize, 1028, 7, 256, 3][..]] {
        let s_in: Vec<Vec<f32>> = s_lens.iter().map(|&n| rng.vec(n, -5.0, 5.0)).collect();
        let s_bufs: Vec<_> = s_in.iter().map(|v| ctx.upload(bytemuck::cast_slice(v)).unwrap()).collect();
        let s_xs: Vec<_> = s_bufs.iter().zip(s_lens).map(|(b, &n)| (b.raw(), n)).collect();
        scale_many(&ctx, &s_xs, bf1.raw(), bf2.raw()).unwrap();
        for (i, (&n, orig)) in s_lens.iter().zip(&s_in).enumerate() {
            let got: Vec<f32> =
                bytemuck::pod_collect_to_vec(&ctx.read(s_bufs[i].raw(), 0, (n * 4) as u64).unwrap());
            let want: Vec<f32> = orig.iter().map(|&x| (x * f1_val) * f2_val).collect();
            // Bit-for-bit check ensures the GPU did not reassociate `(x * f1) * f2` into `x * (f1 * f2)`.
            assert_eq!(got, want, "scale_many[{i}] n={n} bit-exact two roundings");
        }
    }

    // Extreme finite values near `f32::MAX`: `h = 1 / sqrt(f32::MAX)` makes `h * h` a subnormal that flushes to
    // 0 on FTZ GPU backends, so `norm_partials_many(ScaledSquares)` must keep `(x * h) * h` separate.
    let max_in = [f32::MAX; 4];
    let h_max = 1.0f32 / f32::MAX.sqrt();
    let bmax = ctx.upload(bytemuck::cast_slice(&max_in)).unwrap();
    let bh_max = ctx.upload(bytemuck::cast_slice(&[h_max])).unwrap();
    let bpart_max = ctx.alloc(4).unwrap();
    norm_partials_many(
        &ctx,
        NormPartial::ScaledSquares,
        &[(bmax.raw(), 4)],
        Some(bh_max.raw()),
        bpart_max.raw(),
    )
    .unwrap();
    let got_max: Vec<f32> = bytemuck::pod_collect_to_vec(&ctx.read(bpart_max.raw(), 0, 4).unwrap());
    let t_max = (f32::MAX * h_max) * h_max;
    assert_close(&got_max, &[4.0 * t_max * t_max], 1e-4, 1e-4, "norm_partials vec4 f32::MAX");

    // 4. norm_partials_many vec4 (all-4-aligned chunk) with NaN in every lane 0..4 (covering both subgroup 0 in
    // `t0`/`t2` and subgroup 1 `lid=63` in `t1`), subgroups vs portable.
    let h_val = 0.5f32;
    let bh = ctx.upload(bytemuck::cast_slice(&[h_val])).unwrap();
    let np_lens = [4usize, 256, 260, 1024];
    let plen = norm_partials_len(&np_lens);
    for lane in 0..4 {
        let mut t0 = rng.vec(np_lens[0], -3.0, 3.0);
        let mut t1 = rng.vec(np_lens[1], -3.0, 3.0);
        let mut t2 = rng.vec(np_lens[2], -3.0, 3.0);
        let t3 = rng.vec(np_lens[3], -3.0, 3.0);
        t0[lane] = f32::NAN;
        t1[252 + lane] = f32::NAN;
        t2[256 + lane] = f32::NAN;
        let b0 = ctx.upload(bytemuck::cast_slice(&t0)).unwrap();
        let b1 = ctx.upload(bytemuck::cast_slice(&t1)).unwrap();
        let b2 = ctx.upload(bytemuck::cast_slice(&t2)).unwrap();
        let b3 = ctx.upload(bytemuck::cast_slice(&t3)).unwrap();
        let xs = [(b0.raw(), t0.len()), (b1.raw(), t1.len()), (b2.raw(), t2.len()), (b3.raw(), t3.len())];
        for op in [NormPartial::AbsMax, NormPartial::ScaledSquares] {
            let run = |portable: bool| -> Vec<f32> {
                let _g = ForceGuard::set(portable);
                let part = ctx.alloc((plen * 4) as u64).unwrap();
                norm_partials_many(
                    &ctx,
                    op,
                    &xs,
                    (op == NormPartial::ScaledSquares).then_some(bh.raw()),
                    part.raw(),
                )
                .unwrap();
                bytemuck::pod_collect_to_vec(&ctx.read(part.raw(), 0, (plen * 4) as u64).unwrap())
            };
            let sg = run(false);
            let fb = run(true);
            assert_eq!(sg.len(), plen);
            // Partials layout: t0 -> [0], t1 -> [1] (sgid=1 NaN), t2 -> [2, 3], t3 -> [4..8].
            assert!(sg[0].is_nan() && fb[0].is_nan(), "norm_partials vec4 {op:?} lane {lane} missed t0 NaN");
            assert!(
                sg[1].is_nan() && fb[1].is_nan(),
                "norm_partials vec4 {op:?} lane {lane} missed t1 sg1 NaN"
            );
            assert!(sg[2].is_finite() && fb[2].is_finite(), "norm_partials vec4 {op:?} leaked to t2[0]");
            assert!(
                sg[3].is_nan() && fb[3].is_nan(),
                "norm_partials vec4 {op:?} lane {lane} missed t2 tail NaN"
            );
            assert!(sg[4].is_finite() && fb[4].is_finite(), "norm_partials vec4 {op:?} leaked to t3");
            for (i, (&a, &b)) in sg.iter().zip(&fb).enumerate() {
                if a.is_nan() && b.is_nan() {
                    continue;
                }
                assert_close(&[a], &[b], 1e-4, 1e-5, &format!("norm_partials vec4 {op:?} lane {lane} [{i}]"));
            }
        }
    }
    assert!(ctx.kernels_without_registry().is_empty());
}
