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
