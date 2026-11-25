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
            let fill = if row.iter().all(|v| *v == -inf) { -inf } else { -10.0 };
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
    // Extreme logits (±1e4) must not overflow or underflow on any of the 4 kernel paths.
    // Note: GPU drivers may hoist `m + ls` in `x - m - ls`, introducing up to 0.5 ULP(1e4) ≈ 5e-4 on log_softmax.
    let extreme = [1e4f32, -1e4, 1e4 - 1.0, 0.0];
    for (n, label, dim) in [
        (4usize, "extreme rows cached", 1usize),
        (1057, "extreme rows uncached", 1),
        (4, "extreme cols small", 0),
        (48, "extreme cols general", 0),
    ] {
        let mut row = extreme.to_vec();
        row.resize(n, -1e4);
        let t = Tensor::from_vec(row, [1, n]).unwrap();
        let t = if dim == 0 { t.t().unwrap().contiguous().unwrap() } else { t };
        check(dev, std::slice::from_ref(&t), 1e-6, 1e-5, &format!("{label} softmax"), |v| v[0].softmax(dim));
        check(dev, std::slice::from_ref(&t), 1e-3, 1e-3, &format!("{label} log_softmax"), |v| {
            v[0].log_softmax(dim)
        });
    }
}

#[test]
fn online_softmax_paths_and_backward_grads() {
    let dev = dev!();
    // Cover register-cached rows (n <= 1024), 2-pass rows (n > 1024), small-D cols (d <= 32, incl. DFL d = 16),
    // and general 2-pass cols (d > 32), checking forward parity and autograd backward gradients vs CPU.
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
        let w = randn(shape, 78);
        for log in [false, true] {
            let f = move |v: &[Tensor]| if log { v[0].log_softmax(dim) } else { v[0].softmax(dim) };
            check(dev, std::slice::from_ref(&x), 1e-6, 1e-5, &format!("{label} fwd log={log}"), f);

            let xc = x.detach().requires_grad_();
            let yc = if log { xc.log_softmax(dim).unwrap() } else { xc.softmax(dim).unwrap() };
            let lc = yc.mul(&w).unwrap().sum_all().unwrap();
            let gc = lc.backward().unwrap().get(&xc).unwrap().clone();

            let xg = x.to_device(dev).unwrap().detach().requires_grad_();
            let wg = w.to_device(dev).unwrap();
            let yg = if log { xg.log_softmax(dim).unwrap() } else { xg.softmax(dim).unwrap() };
            let lg = yg.mul(&wg).unwrap().sum_all().unwrap();
            let gg = lg.backward().unwrap().get(&xg).unwrap().to_device(Device::Cpu).unwrap();

            let gv = gg.to_vec::<f32>().unwrap();
            let cv = gc.to_vec::<f32>().unwrap();
            for (i, (&a, &b)) in gv.iter().zip(&cv).enumerate() {
                let err = (a - b).abs();
                let lim = 1e-4 + 1e-4 * b.abs();
                assert!(err <= lim, "{label} bwd log={log} [{i}]: gpu={a} cpu={b} err={err}");
            }
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
