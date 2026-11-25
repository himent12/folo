//! Reductions, softmax and the ordering of `topk`.

use folo_core::{Device, Tensor};

use super::fixtures::{assert_same, check, dev, randn};

#[test]
fn reductions_and_softmax() {
    let dev = dev!();
    let x = randn(&[3, 5, 7], 12);
    for dims in [vec![0usize], vec![1], vec![2], vec![0, 2], vec![0, 1, 2]] {
        let d2 = dims.clone();
        check(dev, std::slice::from_ref(&x), 1e-5, 1e-5, &format!("sum {dims:?}"), move |v| {
            v[0].sum(d2.clone(), false)
        });
        let d2 = dims.clone();
        check(dev, std::slice::from_ref(&x), 1e-5, 1e-5, &format!("mean {dims:?}"), move |v| {
            v[0].mean(d2.clone(), true)
        });
        let d2 = dims.clone();
        check(dev, std::slice::from_ref(&x), 0.0, 0.0, &format!("max {dims:?}"), move |v| {
            v[0].max(d2.clone(), false)
        });
        let d2 = dims.clone();
        check(dev, std::slice::from_ref(&x), 0.0, 0.0, &format!("min {dims:?}"), move |v| {
            v[0].min(d2.clone(), false)
        });
    }
    for d in 0..3 {
        check(dev, std::slice::from_ref(&x), 0.0, 0.0, "argmax", |v| v[0].argmax(d, false));
        check(dev, std::slice::from_ref(&x), 0.0, 0.0, "argmin", |v| v[0].argmin(d, true));
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, "softmax", |v| v[0].softmax(d));
        check(dev, std::slice::from_ref(&x), 1e-5, 1e-5, "log_softmax", |v| v[0].log_softmax(d));
    }
    // Non-contiguous input, variance, logsumexp, and ties for max / argmax.
    check(dev, std::slice::from_ref(&x), 1e-5, 1e-4, "var of transposed", |v| {
        v[0].transpose(0, 2)?.var([1, 2], true, false)
    });
    check(dev, std::slice::from_ref(&x), 1e-5, 1e-5, "logsumexp", |v| v[0].logsumexp(1, false));
    let ties = Tensor::from_slice(&[1.0f32, 3.0, 3.0, 2.0, 0.0, 0.0], [2, 3]).unwrap();
    check(dev, std::slice::from_ref(&ties), 0.0, 0.0, "max ties", |v| v[0].max(1, false));
    check(dev, &[ties], 0.0, 0.0, "argmax ties", |v| v[0].argmax(1, false));
    // NaN propagates through max / min on the row, column and split paths, as on the CPU.
    let mut v = randn(&[3, 4, 9000], 14).to_vec::<f32>().unwrap();
    for i in [5usize, 9000 * 4 + 17, 9000 * 9 + 8999] {
        v[i] = f32::NAN;
    }
    let nan = Tensor::from_vec(v, [3, 4, 9000]).unwrap();
    for dims in [vec![2usize], vec![0], vec![1], vec![0, 2]] {
        let (d1, d2) = (dims.clone(), dims.clone());
        check(dev, std::slice::from_ref(&nan), 0.0, 0.0, &format!("max with NaN {dims:?}"), move |v| {
            v[0].detach().max(d1.clone(), false)
        });
        check(dev, std::slice::from_ref(&nan), 0.0, 0.0, &format!("min with NaN {dims:?}"), move |v| {
            v[0].detach().min(d2.clone(), false)
        });
    }
    // A long reduction over few outputs takes the two-pass split (BatchNorm statistics at 640x640).
    let long = randn(&[2, 3, 160, 160], 13);
    check(dev, &[long], 2e-3, 1e-4, "bn statistics", |v| v[0].sum([0, 2, 3], false));
}

#[test]
fn softmax_with_infinite_entries() {
    // A non-finite maximum is replaced by 0, as on the CPU: a row with +inf gives 0 at its finite entries
    // (NaN at the infinity), a row of -inf NaN; over the last dim (cached and uncached rows kernel) and another
    // dim (small-D and general columns kernel).
    let dev = dev!();
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let cases: [[f32; 4]; 9] = [
        [1.0, inf, 2.0, -3.0],
        [inf, inf, 1.0, 0.5],
        [inf, nan, 1.0, 0.0],
        [nan, inf, 1.0, 0.0],
        [-inf, -inf, -inf, -inf],
        [-inf, nan, -inf, -inf],
        [nan, nan, nan, nan],
        [nan, 1.0, 0.5, -2.0],
        [-inf, 3.0, -1.0, -inf],
    ];
    let v: Vec<f32> = cases.iter().flatten().copied().collect();
    let x = Tensor::from_vec(v, [cases.len(), 4]).unwrap();
    for log in [false, true] {
        let f = move |v: &[Tensor]| if log { v[0].log_softmax(1) } else { v[0].softmax(1) };
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-6, &format!("rows cached log={log}"), f);
        let f = move |v: &[Tensor]| if log { v[0].t()?.log_softmax(0) } else { v[0].t()?.softmax(0) };
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-6, &format!("cols small log={log}"), f);
    }
    // Exercise uncached rows (n > 1024) and general columns (d > 32) on the same non-finite edge cases.
    for (n, label, dim) in [(1057usize, "rows uncached", 1usize), (48usize, "cols general", 0usize)] {
        let mut padded = Vec::with_capacity(cases.len() * n);
        for row in &cases {
            let fill = if row.iter().all(|v| *v == -inf || v.is_nan()) { -inf } else { -10.0 };
            padded.extend_from_slice(row);
            padded.resize(padded.len() + n - row.len(), fill);
        }
        let t = Tensor::from_vec(padded, [cases.len(), n]).unwrap();
        let t = if dim == 0 { t.t().unwrap().contiguous().unwrap() } else { t };
        for log in [false, true] {
            let f = move |v: &[Tensor]| if log { v[0].log_softmax(dim) } else { v[0].softmax(dim) };
            check(dev, std::slice::from_ref(&t), 1e-6, 1e-5, &format!("{label} log={log}"), f);
        }
    }
    // Extreme logits (±1e4, and ±3e38 where differences overflow) must not overflow or produce NaN on any of the
    // 4 kernel paths. Column fixtures use 2 columns (`inner = 2 > 1`) so they dispatch `softmax_cols.wgsl`.
    // Note: GPU drivers may hoist `m + ls` in `x - m - ls`, introducing up to 0.5 ULP(1e4) ≈ 5e-4 on log_softmax.
    let extreme = [1e4f32, -1e4, 1e4 - 1.0, 0.0];
    let huge = [3e38f32, -3e38, 3e38, 0.0];
    for (n, label, dim) in [
        (4usize, "extreme rows cached", 1usize),
        (1057, "extreme rows uncached", 1),
        (4, "extreme cols small", 0),
        (48, "extreme cols general", 0),
    ] {
        let mut data = Vec::with_capacity(2 * n);
        for _ in 0..2 {
            data.extend_from_slice(&extreme);
            data.resize(data.len() + n - extreme.len(), -1e4);
        }
        let t = Tensor::from_vec(data, [2, n]).unwrap();
        let t = if dim == 0 { t.t().unwrap().contiguous().unwrap() } else { t };
        check(dev, std::slice::from_ref(&t), 1e-6, 1e-5, &format!("{label} softmax"), |v| v[0].softmax(dim));
        check(dev, std::slice::from_ref(&t), 1e-3, 1e-3, &format!("{label} log_softmax"), |v| {
            v[0].log_softmax(dim)
        });
        // ±3e38 forward softmax parity (difference `-3e38 - 3e38 = -inf` maps to 0.0, tied maxes get 0.5 each).
        let mut hdata = Vec::with_capacity(2 * n);
        for _ in 0..2 {
            hdata.extend_from_slice(&huge);
            hdata.resize(hdata.len() + n - huge.len(), -3e38);
        }
        let ht = Tensor::from_vec(hdata, [2, n]).unwrap();
        let ht = if dim == 0 { ht.t().unwrap().contiguous().unwrap() } else { ht };
        let cpu = ht.softmax(dim).unwrap();
        let gpu = ht.to_device(dev).unwrap().softmax(dim).unwrap();
        assert_same(&gpu, &cpu, &format!("{label} 3e38 softmax"));
    }
}

#[test]
fn online_softmax_paths_and_backward_grads() {
    let dev = dev!();
    // Cover register-cached rows (n <= 1024), 2-pass rows (n > 1024), small-D cols (d <= 32, incl. DFL d = 16),
    // and general 2-pass cols (d > 32); `check` verifies both forward and autograd backward parity vs CPU.
    let shapes_and_dims: &[(&[usize], usize, &str)] = &[
        (&[3, 16], 1, "rows n=16"),
        (&[2, 256], 1, "rows n=256"),
        (&[2, 400], 1, "rows n=400"),
        (&[2, 1024], 1, "rows n=1024"),
        (&[2, 1025], 1, "rows n=1025"),
        (&[2, 16, 45], 1, "cols d=16"),
        (&[2, 32, 17], 1, "cols d=32"),
        (&[2, 33, 13], 1, "cols d=33"),
        (&[2, 64, 11], 1, "cols d=64"),
    ];
    for &(shape, dim, label) in shapes_and_dims {
        let x = randn(shape, 77);
        for log in [false, true] {
            let f = move |v: &[Tensor]| if log { v[0].log_softmax(dim) } else { v[0].softmax(dim) };
            check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, &format!("{label} log={log}"), f);
        }
    }
}

#[test]
fn topk_orders_nan_and_infinities_by_total_cmp() {
    // `f32::total_cmp`: -NaN < -inf < .. < -0 = +0 < .. < inf < NaN, NaN payloads kept. The two zeros are equal:
    // they tie by index, as on the CPU (the last row has -0 before +0, the first two the other order).
    let dev = dev!();
    let nan2 = f32::from_bits(0x7fc0_0123);
    let v = [
        0.0f32,
        f32::NAN,
        -1.0,
        f32::NEG_INFINITY,
        1.0,
        f32::INFINITY,
        -f32::NAN,
        2.0,
        nan2,
        -0.0,
        0.0,
        -nan2,
        0.0,
        -0.0,
        -0.0,
        0.0,
    ];
    let t = Tensor::from_slice(&v, [4, 4]).unwrap();
    for k in 1..=4 {
        for largest in [true, false] {
            let (cv, ci) = t.topk(k, largest).unwrap();
            let (gv, gi) = t.to_device(dev).unwrap().topk(k, largest).unwrap();
            assert_same(&gv, &cv, &format!("topk values k={k} {largest}"));
            assert_same(&gi, &ci, &format!("topk indices k={k} {largest}"));
        }
    }
    // The zeros of the last row come out in index order, whatever their signs.
    let (_, zi) = t.to_device(dev).unwrap().topk(4, true).unwrap();
    assert_eq!(zi.to_device(Device::Cpu).unwrap().to_vec2::<i64>().unwrap()[3], vec![0, 1, 2, 3]);
}

#[test]
fn topk_bitonic_and_parallel_select_match_cpu() {
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

    let dev = dev!();
    // Bitonic path (`d <= 512`): non-power-of-two, exact power-of-two, `d = 1`, `k = d`, ties, and special floats.
    for &(rows, d, k) in &[(3usize, 1usize, 1usize), (4, 17, 7), (2, 256, 256), (2, 300, 15), (2, 512, 32)] {
        let mut vals = randn(&[rows, d], 101 + d as u64).to_vec::<f32>().unwrap();
        if d >= 6 {
            // Inject ties, -0/+0, ±inf, and ±NaN with distinct payloads.
            vals[0] = -0.0;
            vals[1] = 0.0;
            vals[2] = f32::from_bits(0x7fc0_0456);
            vals[3] = f32::INFINITY;
            vals[4] = f32::NEG_INFINITY;
            vals[5] = f32::from_bits(0xffc0_0789);
        }
        let t = Tensor::from_vec(vals, [rows, d]).unwrap();
        for largest in [true, false] {
            let (cv, ci) = t.topk(k, largest).unwrap();
            let (gv, gi) = t.to_device(dev).unwrap().topk(k, largest).unwrap();
            assert_same(&gv, &cv, &format!("bitonic vals {rows}x{d} k={k} largest={largest}"));
            assert_same(&gi, &ci, &format!("bitonic idx {rows}x{d} k={k} largest={largest}"));
        }
    }
    // Parallel selection path (`d > 512`): test both subgroup and portable fallback, including ties, -0/+0, ±inf, ±NaN.
    for &(rows, d, k) in &[(2usize, 513usize, 12usize), (3, 2051, 20)] {
        let mut vals = randn(&[rows, d], 201 + d as u64).to_vec::<f32>().unwrap();
        // Put ties and edge values across multiple workgroup strides (including the final tail element).
        vals[0] = 5.0;
        vals[256] = 5.0;
        vals[d - 1] = 5.0;
        vals[d + 1] = -0.0;
        vals[d + 257] = 0.0;
        vals[d + 300] = f32::from_bits(0x7fc0_0abc);
        vals[d + 512] = f32::NEG_INFINITY;
        vals[2 * d - 1] = f32::from_bits(0xffc0_0def);
        let t = Tensor::from_vec(vals, [rows, d]).unwrap();
        for largest in [true, false] {
            let (cv, ci) = t.topk(k, largest).unwrap();
            for portable in [false, true] {
                let _g = ForceGuard::set(portable);
                let (gv, gi) = t.to_device(dev).unwrap().topk(k, largest).unwrap();
                assert_same(
                    &gv,
                    &cv,
                    &format!("select vals {rows}x{d} k={k} largest={largest} fb={portable}"),
                );
                assert_same(
                    &gi,
                    &ci,
                    &format!("select idx {rows}x{d} k={k} largest={largest} fb={portable}"),
                );
            }
        }
    }
}

#[test]
fn subgroup_vs_portable_row_reductions_and_softmax() {
    use super::fixtures::assert_close_t;
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
    let dev = dev!();
    // Row reductions (Sum/Mean/Max/Min) and softmax/log_softmax (cached n <= 1024 and uncached n > 1024).
    for &(rows, n) in &[(3usize, 17usize), (2, 256), (2, 400), (2, 1057)] {
        let x = randn(&[rows, n], 91).to_device(dev).unwrap();
        let run = |portable: bool| {
            let _g = ForceGuard::set(portable);
            (
                x.sum(1, false).unwrap(),
                x.mean(1, false).unwrap(),
                x.max(1, false).unwrap(),
                x.min(1, false).unwrap(),
                x.softmax(1).unwrap(),
                x.log_softmax(1).unwrap(),
            )
        };
        let (sum_sg, mean_sg, max_sg, min_sg, sm_sg, lsm_sg) = run(false);
        let (sum_fb, mean_fb, max_fb, min_fb, sm_fb, lsm_fb) = run(true);
        let tag = format!("sg vs portable {rows}x{n}");
        assert_close_t(&sum_sg, &sum_fb, 1e-4, 1e-5, &format!("{tag} sum"));
        assert_close_t(&mean_sg, &mean_fb, 1e-5, 1e-5, &format!("{tag} mean"));
        assert_same(&max_sg, &max_fb, &format!("{tag} max"));
        assert_same(&min_sg, &min_fb, &format!("{tag} min"));
        assert_close_t(&sm_sg, &sm_fb, 1e-6, 1e-5, &format!("{tag} softmax"));
        assert_close_t(&lsm_sg, &lsm_fb, 1e-5, 1e-5, &format!("{tag} log_softmax"));
    }
    // NaN and ±inf propagation in max/min and softmax must match CPU on BOTH subgroup and portable paths.
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let mut vals = vec![-2.0f32; 5 * 300];
    vals[17] = nan;
    vals[300 + 299] = inf;
    vals[2 * 300..3 * 300].fill(-inf);
    vals[3 * 300 + 42] = inf;
    vals[3 * 300 + 250] = nan;
    vals[4 * 300 + 1] = 10.0;
    let special = Tensor::from_vec(vals, [5, 300]).unwrap();
    for portable in [false, true] {
        let _g = ForceGuard::set(portable);
        let xg = special.to_device(dev).unwrap();
        assert_close_t(
            &xg.max(1, false).unwrap(),
            &special.max(1, false).unwrap(),
            0.0,
            0.0,
            &format!("max nan fb={portable}"),
        );
        assert_close_t(
            &xg.min(1, false).unwrap(),
            &special.min(1, false).unwrap(),
            0.0,
            0.0,
            &format!("min nan fb={portable}"),
        );
        assert_close_t(
            &xg.sum(1, false).unwrap(),
            &special.sum(1, false).unwrap(),
            0.0,
            0.0,
            &format!("sum nan fb={portable}"),
        );
        check(dev, std::slice::from_ref(&special), 1e-6, 1e-5, &format!("sm special fb={portable}"), |v| {
            v[0].softmax(1)
        });
        check(dev, std::slice::from_ref(&special), 1e-6, 1e-5, &format!("lsm special fb={portable}"), |v| {
            v[0].log_softmax(1)
        });
    }
}
