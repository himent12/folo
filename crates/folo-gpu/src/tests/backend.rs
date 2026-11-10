//! The folo-core backend against folo-core's CPU backend: every op the trait dispatches, forward and (through
//! autograd) backward, on seeded random data with strided, broadcast and odd-sized operands.
//!
//! Tolerances: elementwise ops match to `1e-5` relative (GPU `exp` / `tanh` / `pow` are a few ulp from the
//! CPU's); reductions, GEMM and convolutions to `1e-4` relative + a small absolute term scaled to the
//! reduction length (different summation order, FMA); copies, casts and index ops exactly.

use folo_core::backend::UnaryOp as U;
use folo_core::rng::Rng;
use folo_core::{DType, Device, Result, Tensor, no_grad};

/// The registered GPU device, or `None` (with a note) when there is no suitable adapter.
pub(crate) fn device() -> Option<Device> {
    crate::tests::ctx()?;
    match crate::init() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("note: skipping GPU backend test: {e}");
            None
        }
    }
}

/// `let dev = dev!();`: the GPU device or an early return.
macro_rules! dev {
    () => {
        match $crate::tests::backend::device() {
            Some(d) => d,
            None => return,
        }
    };
}
pub(crate) use dev;

pub(crate) fn randn(shape: &[usize], seed: u64) -> Tensor {
    Tensor::randn(shape, 0.0, 1.0, &mut Rng::seed(seed))
}

fn uniform(shape: &[usize], lo: f32, hi: f32, seed: u64) -> Tensor {
    Tensor::rand_uniform(shape, lo, hi, &mut Rng::seed(seed))
}

/// Asserts two f32 tensors (on any devices) match elementwise within `atol + rtol·|cpu|`.
#[track_caller]
pub(crate) fn assert_close_t(gpu: &Tensor, cpu: &Tensor, atol: f32, rtol: f32, what: &str) {
    assert_eq!(gpu.shape(), cpu.shape(), "{what}: shape");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}: dtype");
    let g = gpu.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    let c = cpu.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    crate::tests::assert_close(&g, &c, atol, rtol, what);
}

#[track_caller]
fn assert_same(gpu: &Tensor, cpu: &Tensor, what: &str) {
    assert_eq!(gpu.shape(), cpu.shape(), "{what}: shape");
    assert_eq!(gpu.dtype(), cpu.dtype(), "{what}: dtype");
    assert!(gpu.device().is_gpu(), "{what}: result left the GPU");
    let (g, c) = (gpu.to_cpu_storage().unwrap(), cpu.to_cpu_storage().unwrap());
    // Compare bit patterns so NaN payloads and signed zeros count.
    assert_eq!(g.as_bytes(), c.as_bytes(), "{what}: {g:?} vs {c:?}");
}

/// Runs `f` on CPU and GPU copies of `inputs`; compares outputs and, when the output requires grad, the
/// gradient of every f32 input for a random output gradient.
#[track_caller]
pub(crate) fn check<F>(dev: Device, inputs: &[Tensor], atol: f32, rtol: f32, what: &str, f: F)
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let lift = |t: &Tensor, d: Device| -> Tensor {
        let t = t.to_device(d).unwrap().detach();
        if t.dtype() == DType::F32 { t.requires_grad_() } else { t }
    };
    let cpu: Vec<Tensor> = inputs.iter().map(|t| lift(t, Device::Cpu)).collect();
    let gpu: Vec<Tensor> = inputs.iter().map(|t| lift(t, dev)).collect();
    let yc = f(&cpu).unwrap_or_else(|e| panic!("{what}: cpu: {e}"));
    let yg = f(&gpu).unwrap_or_else(|e| panic!("{what}: gpu: {e}"));
    assert!(yg.device().is_gpu(), "{what}: result left the GPU");
    if yc.dtype() == DType::F32 {
        assert_close_t(&yg, &yc, atol, rtol, what);
    } else {
        assert_same(&yg, &yc, what);
    }
    if !yc.requires_grad() {
        return;
    }
    let seed = randn(yc.dims(), 0xfeed);
    let gc = yc.backward_with(&seed).unwrap();
    let gg = yg
        .backward_with(&seed.to_device(dev).unwrap())
        .unwrap_or_else(|e| panic!("{what}: gpu backward: {e}"));
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        if c.dtype() != DType::F32 {
            continue;
        }
        match (gc.get(c), gg.get(g)) {
            (Some(a), Some(b)) => {
                assert!(b.device().is_gpu(), "{what}: grad {i} left the GPU");
                assert_close_t(b, a, atol, rtol, &format!("{what}: grad of input {i}"));
            }
            (None, None) => {}
            (a, b) => panic!("{what}: grad {i} present on cpu {} / gpu {}", a.is_some(), b.is_some()),
        }
    }
}

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
fn unary_ops_and_gradients() {
    let dev = dev!();
    let ops = [
        U::Neg,
        U::Abs,
        U::Exp,
        U::Log,
        U::Sqrt,
        U::Rsqrt,
        U::Recip,
        U::Square,
        U::Sin,
        U::Cos,
        U::Tanh,
        U::Sigmoid,
        U::Silu,
        U::Relu,
        U::LeakyRelu(0.1),
        U::Hardswish,
        U::Softplus,
        U::Clamp(-0.5, 0.7),
        U::PowScalar(2.0),
        U::PowScalar(0.5),
        U::PowScalar(3.0),
        U::PowScalar(-1.5),
        U::AddScalar(0.5),
        U::MulScalar(-2.0),
        U::RSubScalar(1.0),
        U::RDivScalar(2.0),
        U::DivScalar(4.0),
        U::MaxScalar(0.1),
        U::MinScalar(-0.1),
        U::Floor,
        U::Ceil,
        U::Round,
        U::Sign,
    ];
    // [7, 9] (odd: scalar path) and [4, 16] (vec4 path); a transposed view takes the strided path.
    for (shape, seed) in [([7usize, 9], 1u64), ([4, 16], 2)] {
        let x = uniform(&shape, -4.0, 4.0, seed);
        let pos = uniform(&shape, 0.1, 4.0, seed + 10);
        for op in ops {
            let positive = matches!(op, U::Log | U::Sqrt | U::Rsqrt | U::PowScalar(0.5) | U::PowScalar(-1.5));
            let input = if positive { &pos } else { &x };
            let what = format!("{op:?} {shape:?}");
            check(dev, std::slice::from_ref(input), 1e-5, 1e-5, &what, |v| v[0].unary(op));
            check(dev, std::slice::from_ref(input), 1e-5, 1e-5, &format!("{what} transposed"), |v| {
                v[0].t()?.unary(op)
            });
        }
    }
    // Special values: rounding ties, signed zeros, NaN propagation of relu / clamp (Rust max semantics).
    let s =
        Tensor::from_slice(&[0.5f32, -0.5, 1.5, -2.5, 2.4999998, 0.0, -0.0, f32::NAN, 100.0, -100.0], [10])
            .unwrap();
    for op in [U::Round, U::Sign, U::Relu, U::Abs, U::Floor, U::Ceil, U::Sigmoid, U::Tanh, U::Softplus] {
        let g = s.to_device(dev).unwrap().unary(op).unwrap();
        assert_close_t(&g, &s.unary(op).unwrap(), 1e-6, 1e-6, &format!("{op:?} special values"));
    }
}

#[test]
fn binary_ops_broadcasting() {
    let dev = dev!();
    let a = uniform(&[2, 3, 5], -3.0, 3.0, 4);
    let b = uniform(&[3, 1], -3.0, 3.0, 5);
    let pos = uniform(&[2, 3, 5], 0.2, 3.0, 6);
    let c = uniform(&[5], -2.0, 2.0, 7);
    check(dev, &[a.clone(), b.clone()], 1e-5, 1e-5, "add bcast", |v| v[0].add(&v[1]));
    check(dev, &[a.clone(), b.clone()], 1e-5, 1e-5, "sub bcast", |v| v[0].sub(&v[1]));
    check(dev, &[a.clone(), c.clone()], 1e-5, 1e-5, "mul bcast", |v| v[0].mul(&v[1]));
    check(dev, &[a.clone(), pos.clone()], 1e-5, 1e-5, "div", |v| v[0].div(&v[1]));
    check(dev, &[pos.clone(), b.clone()], 1e-5, 1e-5, "pow", |v| v[0].pow(&v[1]));
    check(dev, &[a.clone(), c.clone()], 1e-5, 1e-5, "maximum", |v| v[0].maximum(&v[1]));
    check(dev, &[a.clone(), c.clone()], 1e-5, 1e-5, "minimum", |v| v[0].minimum(&v[1]));
    // Transposed lhs, scalar rhs, and the NCHW bias pattern.
    check(dev, &[a.clone(), b.clone()], 1e-5, 1e-5, "transposed lhs", |v| v[0].transpose(0, 2)?.mul(&v[1]));
    check(dev, std::slice::from_ref(&a), 1e-5, 1e-5, "scalar rhs", |v| v[0].add(&v[0].sum_all()?));
    let x = randn(&[2, 4, 6, 6], 8);
    let bias = randn(&[1, 4, 1, 1], 9);
    check(dev, &[x, bias], 1e-5, 1e-5, "nchw bias", |v| v[0].add(&v[1])?.mul(&v[1]));
    // Comparisons with ties and NaN.
    let p = Tensor::from_slice(&[1.0f32, 2.0, f32::NAN, -1.0, 0.0, 3.0], [2, 3]).unwrap();
    let q = Tensor::from_slice(&[1.0f32, 3.0, 2.0], [3]).unwrap();
    for (name, f) in [
        ("eq", Tensor::eq as fn(&Tensor, &Tensor) -> Result<Tensor>),
        ("ne", Tensor::ne),
        ("gt", Tensor::gt),
        ("ge", Tensor::ge),
        ("lt", Tensor::lt),
        ("le", Tensor::le),
        ("maximum", Tensor::maximum),
        ("minimum", Tensor::minimum),
    ] {
        check(dev, &[p.clone(), q.clone()], 0.0, 0.0, name, |v| f(&v[0].detach(), &v[1].detach()));
    }
    // Negative bases with integer exponents (C powf).
    let nb = Tensor::from_slice(&[-2.0f32, -0.5, 0.0, -3.0], [4]).unwrap();
    let ex = Tensor::from_slice(&[3.0f32, 2.0, -1.0, 0.5], [4]).unwrap();
    let g = nb.to_device(dev).unwrap().pow(&ex.to_device(dev).unwrap()).unwrap();
    assert_close_t(&g, &nb.pow(&ex).unwrap(), 1e-6, 1e-6, "pow special");
}

#[test]
fn where_and_masks() {
    let dev = dev!();
    let a = randn(&[3, 4], 10);
    let b = randn(&[4], 11);
    check(dev, &[a.clone(), b.clone()], 0.0, 0.0, "where f32 mask", |v| {
        v[0].gt_scalar(0.0)?.where_cond(&v[0], &v[1])
    });
    check(dev, &[a.clone(), b.clone()], 0.0, 0.0, "where u8 mask", |v| {
        v[1].lt_scalar(0.2)?.to_dtype(DType::U8)?.where_cond(&v[0], &v[1].mul_scalar(3.0)?)
    });
    check(dev, &[a], 0.0, 0.0, "masked_fill", |v| v[0].masked_fill(&v[0].t()?.t()?.gt_scalar(0.5)?, -7.0));
}

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
    // (NaN at the infinity), a row of -inf NaN; over the last dim (rows kernel) and another dim (columns kernel).
    let dev = dev!();
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let v = [1.0f32, inf, 2.0, -inf, -inf, -inf, -inf, 3.0, -1.0, nan, 1.0, 0.5];
    let x = Tensor::from_slice(&v, [4, 3]).unwrap();
    for log in [false, true] {
        let f = move |v: &[Tensor]| if log { v[0].log_softmax(1) } else { v[0].softmax(1) };
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-6, &format!("rows log={log}"), f);
        let f = move |v: &[Tensor]| if log { v[0].t()?.log_softmax(0) } else { v[0].t()?.softmax(0) };
        check(dev, std::slice::from_ref(&x), 1e-6, 1e-6, &format!("columns log={log}"), f);
    }
}

#[test]
fn matmul_layouts() {
    let dev = dev!();
    let a = randn(&[5, 37, 19], 20);
    let b = randn(&[19, 23], 21);
    let bb = randn(&[5, 19, 23], 22);
    check(dev, &[a.clone(), b.clone()], 1e-4, 1e-4, "shared rhs (folded)", |v| v[0].matmul(&v[1]));
    check(dev, &[a.clone(), bb.clone()], 1e-4, 1e-4, "batched", |v| v[0].matmul(&v[1]));
    check(dev, &[a.clone(), bb.clone()], 1e-4, 1e-4, "transposed views", |v| {
        v[0].transpose(1, 2)?
            .contiguous()?
            .transpose(1, 2)?
            .matmul(&v[1].transpose(1, 2)?.contiguous()?.transpose(1, 2)?)
    });
    check(dev, &[bb.clone(), a.clone()], 1e-4, 1e-4, "both transposed", |v| {
        v[0].transpose(1, 2)?.matmul(&v[1].transpose(1, 2)?)
    });
    let m4 = randn(&[2, 1, 6, 8], 23);
    let n4 = randn(&[3, 8, 4], 24);
    check(dev, &[m4, n4], 1e-4, 1e-4, "4-d broadcast batch", |v| v[0].matmul(&v[1]));
    let big = randn(&[64, 128], 25);
    let w = randn(&[128, 96], 26);
    check(dev, &[big.clone(), w.clone()], 1e-4, 1e-4, "aligned 2-d", |v| v[0].matmul(&v[1]));
    check(dev, &[big, w], 1e-4, 1e-4, "offset slices", |v| {
        v[0].narrow(1, 3, 101)?.matmul(&v[1].narrow(0, 1, 101)?.narrow(1, 2, 90)?)
    });
    let v1 = randn(&[1, 7], 27);
    let v2 = randn(&[7, 1], 28);
    check(dev, &[v1.clone(), v2.clone()], 1e-5, 1e-5, "dot", |v| v[0].matmul(&v[1]));
    check(dev, &[v2, v1], 1e-5, 1e-5, "outer", |v| v[0].matmul(&v[1]));
    let e1 = Tensor::zeros([3, 0]);
    let e2 = Tensor::zeros([0, 4]);
    check(dev, &[e1, e2], 0.0, 0.0, "k = 0", |v| v[0].matmul(&v[1]));
}

#[test]
fn matmul_batches_beyond_one_dispatch() {
    // More batch matrices than one dispatch's 65535 workgroup layers: one level (per-anchor dot products),
    // two levels (a transposed batch that does not coalesce) with the outer and with the inner count too large.
    let dev = dev!();
    let a = randn(&[70000, 1, 3], 90);
    let b = randn(&[70000, 3, 1], 91);
    check(dev, &[a, b], 1e-5, 1e-5, "batch 70000", |v| v[0].matmul(&v[1]));
    let a = randn(&[300, 300, 2, 3], 92);
    let b = randn(&[300, 300, 3, 2], 93);
    check(dev, &[a, b], 1e-5, 1e-5, "batch 300 x 300", |v| {
        v[0].transpose(0, 1)?.matmul(&v[1].transpose(0, 1)?)
    });
    let a = randn(&[70000, 2, 1, 2], 94);
    let b = randn(&[2, 2, 3], 95);
    // The broadcast operand's gradient sums 70000 products: reduction tolerance.
    check(dev, &[a, b], 1e-3, 1e-4, "batch 2 x 70000", |v| {
        v[0].transpose(0, 1)?.matmul(&v[1].reshape([2, 1, 2, 3])?.expand([2, 70000, 2, 3])?)
    });
}

struct ConvCase {
    x: [usize; 4],
    w: [usize; 4],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
}

#[test]
fn conv2d_forward_and_backward() {
    let dev = dev!();
    let cases = [
        ConvCase {
            x: [2, 5, 9, 11],
            w: [7, 5, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 6, 13, 10],
            w: [8, 6, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 8, 12, 12],
            w: [16, 8, 1, 1],
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 3, 17, 17],
            w: [5, 3, 1, 1],
            stride: 2,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 4, 15, 14],
            w: [6, 4, 5, 5],
            stride: 1,
            padding: 4,
            dilation: 2,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 8, 9, 9],
            w: [12, 4, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 2,
            bias: true,
        },
        ConvCase {
            x: [2, 6, 11, 11],
            w: [6, 1, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 6,
            bias: true,
        },
        ConvCase {
            x: [1, 4, 12, 13],
            w: [8, 1, 5, 5],
            stride: 2,
            padding: 2,
            dilation: 1,
            groups: 4,
            bias: false,
        },
        ConvCase {
            x: [1, 3, 32, 32],
            w: [16, 3, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        // Stride phases (the input gradient's decomposition): groups, no padding on odd sizes, a kernel smaller
        // than the stride (phases without taps), a kernel of 4 at stride 2, stride 2 with dilation 2 (not phased).
        ConvCase {
            x: [2, 8, 11, 9],
            w: [12, 4, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 2,
            bias: false,
        },
        ConvCase {
            x: [2, 5, 13, 12],
            w: [7, 5, 3, 3],
            stride: 2,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 6, 14, 13],
            w: [9, 6, 2, 2],
            stride: 3,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [2, 4, 12, 15],
            w: [8, 4, 4, 4],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 4, 15, 15],
            w: [6, 4, 3, 3],
            stride: 2,
            padding: 2,
            dilation: 2,
            groups: 1,
            bias: false,
        },
        // Weight-gradient tiles picked by shape (`conv::pick_config_weight_grad`): 32×128 for few output
        // channels, 128×128 and 64×128 for more, each with at least 128 columns (C_in·KH·KW).
        ConvCase {
            x: [2, 16, 20, 19],
            w: [24, 16, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true,
        },
        ConvCase {
            x: [2, 16, 12, 12],
            w: [128, 16, 3, 3],
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
        ConvCase {
            x: [1, 20, 10, 11],
            w: [70, 20, 3, 3],
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: false,
        },
    ];
    for (i, c) in cases.iter().enumerate() {
        let x = randn(&c.x, 30 + i as u64);
        let w = randn(&c.w, 40 + i as u64).mul_scalar(0.3).unwrap();
        let b = randn(&[c.w[0]], 50 + i as u64);
        let k = (c.w[1] * c.w[2] * c.w[3]) as f32;
        let atol = 1e-4 * k.sqrt();
        let what = format!("conv2d case {i}");
        let (s, p, d, g, has_bias) = (c.stride, c.padding, c.dilation, c.groups, c.bias);
        check(dev, &[x, w, b], atol, 1e-4, &what, move |v| {
            v[0].conv2d(&v[1], if has_bias { Some(&v[2]) } else { None }, s, p, d, g)
        });
    }
}

#[test]
fn conv2d_weight_gradient_split_k() {
    // A small weight with a large image: the weight gradient splits K over many workgroups.
    let dev = dev!();
    let x = randn(&[2, 3, 96, 96], 60);
    let w = randn(&[4, 3, 3, 3], 61).mul_scalar(0.3).unwrap();
    check(dev, &[x.clone(), w.clone()], 2e-3, 1e-4, "split-k weight gradient", |v| {
        v[0].conv2d(&v[1], None, 1, 1, 1, 1)
    });
    let dw = randn(&[4, 1, 3, 3], 62);
    let xd = randn(&[2, 4, 80, 80], 63);
    check(dev, &[xd, dw], 2e-3, 1e-4, "depthwise weight gradient chunks", |v| {
        v[0].conv2d(&v[1], None, 1, 1, 1, 4)
    });
}

#[test]
fn conv2d_input_gradient_stride_beyond_kernel() {
    // Strides far beyond the kernel leave most stride phases without taps: the input gradient's scratch and host
    // work must scale with the phases that have taps, not with stride².
    let dev = dev!();
    let ones = |shape: &[usize]| Tensor::ones(shape);
    check(dev, &[ones(&[1, 2, 4, 4]), ones(&[3, 2, 1, 1])], 1e-5, 0.0, "stride 65536", |v| {
        v[0].conv2d(&v[1], None, 65536, 0, 1, 1)
    });
    let x = ones(&[1, 2, 4, 4]).to_device(dev).unwrap().requires_grad_();
    let w = ones(&[3, 2, 1, 1]).to_device(dev).unwrap();
    let y = x.conv2d(&w, None, 65536, 0, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 3, 1, 1]);
    let gx = y.sum_all().unwrap().backward().unwrap().get(&x).unwrap().to_vec::<f32>().unwrap();
    let want: Vec<f32> = (0..32).map(|i| if i % 16 == 0 { 3.0 } else { 0.0 }).collect();
    assert_eq!(gx, want, "stride 65536 input gradient");
    for (i, (xs, ws, s, p)) in [
        ([2, 3, 17, 19], [4, 3, 1, 1], 4, 0),
        ([1, 4, 23, 21], [5, 4, 3, 3], 5, 1),
        ([1, 2, 9, 9], [3, 2, 2, 2], 7, 3),
    ]
    .into_iter()
    .enumerate()
    {
        let x = randn(&xs, 70 + i as u64);
        let w = randn(&ws, 80 + i as u64).mul_scalar(0.3).unwrap();
        let atol = 1e-4 * ((ws[1] * ws[2] * ws[3]) as f32).sqrt();
        check(dev, &[x, w], atol, 1e-4, &format!("kernel {} stride {s}", ws[2]), move |v| {
            v[0].conv2d(&v[1], None, s, p, 1, 1)
        });
    }
}

#[test]
fn conv2d_batches_beyond_one_dispatch() {
    // The conv kernels put (image, group) on the grid's z axis, at most 65535 workgroups: larger N · groups run
    // in pieces of the batch. A 3×3 conv, a strided one (stride-phase input gradient), a grouped 1×1 (80000
    // layers) and a depthwise one, forward and both gradients; the bias gradient sums 70000 images.
    let dev = dev!();
    for (i, (xs, ws, s, p, g)) in [
        ([70000, 1, 3, 3], [2, 1, 3, 3], 1, 1, 1),
        ([70000, 2, 4, 4], [2, 2, 3, 3], 2, 1, 1),
        ([40000, 4, 2, 2], [4, 2, 1, 1], 1, 0, 2),
        ([70000, 2, 3, 3], [2, 1, 3, 3], 1, 1, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let x = randn(&xs, 110 + i as u64);
        let w = randn(&ws, 120 + i as u64).mul_scalar(0.3).unwrap();
        let b = randn(&[ws[0]], 130 + i as u64);
        check(dev, &[x, w, b], 2e-3, 1e-4, &format!("conv {xs:?} {ws:?} s{s} g{g}"), move |v| {
            v[0].conv2d(&v[1], Some(&v[2]), s, p, 1, g)
        });
    }
    // f16 with an odd image size (9 halves) and an odd last piece: the piece copies stay 4-byte aligned.
    if folo_core::backend::backend(dev).is_ok_and(|b| b.computes_f16()) {
        let x = randn(&[70001, 1, 3, 3], 140).to_dtype(DType::F16).unwrap();
        let w = randn(&[2, 1, 3, 3], 141).mul_scalar(0.3).unwrap().to_dtype(DType::F16).unwrap();
        let y = x.to_device(dev).unwrap().conv2d(&w.to_device(dev).unwrap(), None, 1, 1, 1, 1).unwrap();
        let (xf, wf) = (x.to_dtype(DType::F32).unwrap(), w.to_dtype(DType::F32).unwrap());
        let want = xf.conv2d(&wf, None, 1, 1, 1, 1).unwrap();
        assert_close_t(&y.to_dtype(DType::F32).unwrap(), &want, 2e-3, 2e-3, "f16 conv batch 70001");
    }
}

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
fn topk_orders_nan_and_infinities_by_total_cmp() {
    // `f32::total_cmp`: -NaN < -inf < .. < -0 < +0 < .. < inf < NaN, NaN payloads kept.
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
    ];
    let t = Tensor::from_slice(&v, [3, 4]).unwrap();
    for k in 1..=4 {
        for largest in [true, false] {
            let (cv, ci) = t.topk(k, largest).unwrap();
            let (gv, gi) = t.to_device(dev).unwrap().topk(k, largest).unwrap();
            assert_same(&gv, &cv, &format!("topk values k={k} {largest}"));
            assert_same(&gi, &ci, &format!("topk indices k={k} {largest}"));
        }
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
