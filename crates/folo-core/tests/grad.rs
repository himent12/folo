//! Finite-difference gradient checks for every differentiable op.

use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check, grad_check_with};
use folo_core::{Conv2dParams, Pool2dParams, Result, Tensor};

fn randn(shape: &[usize], seed: u64) -> Tensor {
    Tensor::randn(shape, 0.0, 1.0, &mut Rng::seed(seed))
}

/// Values in `[lo, hi)`, away from kinks and singularities.
fn uniform(shape: &[usize], lo: f32, hi: f32, seed: u64) -> Tensor {
    Tensor::rand_uniform(shape, lo, hi, &mut Rng::seed(seed))
}

/// Distinct values `0.037 + 0.1 * k` in shuffled order: at least 0.037 away from 0, +-0.5, +-3 and each other,
/// so piecewise ops (relu, clamp, max, pooling, ...) are never probed across a kink or a tie.
fn grid(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut v: Vec<f32> = (0..n).map(|i| 0.037 + 0.1 * (i as f32 - (n / 2) as f32)).collect();
    Rng::seed(seed).shuffle(&mut v);
    Tensor::from_vec(v, shape).unwrap()
}

fn check1(f: impl Fn(&Tensor) -> Result<Tensor>, x: Tensor) {
    grad_check(|v| f(&v[0]), &[x]).unwrap();
}

#[test]
fn unary_ops() {
    let x = grid(&[3, 5], 1);
    let pos = uniform(&[3, 5], 0.5, 2.0, 2);
    check1(|x| x.neg(), x.clone());
    check1(|x| x.abs(), x.clone());
    check1(|x| x.exp(), x.clone());
    check1(|x| x.log(), pos.clone());
    check1(|x| x.sqrt(), pos.clone());
    check1(|x| x.rsqrt(), pos.clone());
    check1(|x| x.recip(), pos.clone());
    check1(|x| x.square(), x.clone());
    check1(|x| x.sin(), x.clone());
    check1(|x| x.cos(), x.clone());
    check1(|x| x.tanh(), x.clone());
    check1(|x| x.atan(), x.mul_scalar(2.0).unwrap());
    let inside = uniform(&[3, 5], -0.9, 0.9, 6);
    check1(|x| x.asin(), inside.clone());
    check1(|x| x.acos(), inside.clone());
    check1(|x| x.erf(), x.clone());
    // Zero-gradient rounding ops, away from their jumps.
    for f in [Tensor::floor, Tensor::ceil, Tensor::round, Tensor::sign] {
        check1(f, x.clone());
    }
    check1(|x| x.sigmoid(), x.clone());
    check1(|x| x.silu(), x.clone());
    check1(|x| x.relu(), x.clone());
    check1(|x| x.leaky_relu(0.1), x.clone());
    check1(|x| x.softplus(), x.mul_scalar(3.0).unwrap());
    check1(|x| x.clamp(-0.5, 0.5), x.clone());
    check1(|x| x.powf(3.0), x.clone());
    check1(|x| x.powf(0.7), pos.clone());
    check1(|x| x.add_scalar(2.0), x.clone());
    check1(|x| x.sub_scalar(2.0), x.clone());
    check1(|x| x.mul_scalar(-3.0), x.clone());
    check1(|x| x.div_scalar(4.0), x.clone());
    check1(|x| x.rsub_scalar(1.0), x.clone());
    check1(|x| x.rdiv_scalar(2.0), pos.clone());
    check1(|x| x.maximum_scalar(0.05), x.clone());
    check1(|x| x.minimum_scalar(0.05), x.clone());
    // Hardswish away from its kinks at -3 and 3.
    check1(|x| x.hardswish(), uniform(&[3, 5], -2.5, 2.5, 3));
    check1(|x| x.mul_scalar(3.0)?.hardswish(), grid(&[3, 5], 4));
    // Non-contiguous input.
    check1(|x| x.t()?.silu(), randn(&[4, 3], 5));
}

#[test]
fn binary_ops_with_broadcasting() {
    let a = randn(&[2, 3, 4], 10);
    let b = randn(&[3, 1], 11);
    let pos = uniform(&[3, 1], 0.5, 2.0, 12);
    let grad2 = |f: fn(&Tensor, &Tensor) -> Result<Tensor>, x: Tensor, y: Tensor| {
        grad_check(|v| f(&v[0], &v[1]), &[x, y]).unwrap();
    };
    grad2(|x, y| x.add(y), a.clone(), b.clone());
    grad2(|x, y| x.sub(y), a.clone(), b.clone());
    grad2(|x, y| x.mul(y), a.clone(), b.clone());
    grad2(|x, y| x.div(y), a.clone(), pos.clone());
    // atan2 in all four quadrants (randn), with broadcasting both ways.
    grad2(|x, y| x.atan2(y), a.clone(), b.clone());
    grad2(|x, y| y.atan2(x), a.clone(), b.clone());
    let (ga, gb) = (grid(&[2, 3, 4], 16), grid(&[3, 1], 17).add_scalar(0.05).unwrap());
    grad2(|x, y| x.maximum(y), ga.clone(), gb.clone());
    grad2(|x, y| x.minimum(y), ga, gb);
    grad2(|x, y| x.pow(y), uniform(&[2, 3, 4], 0.5, 2.0, 13), b.clone());
    // Broadcast on the left operand and a transposed operand.
    grad2(|x, y| x.mul(&y.t()?), randn(&[1, 4], 14), randn(&[4, 3], 15));
    // Operators.
    grad2(|x, y| ((x + y)? * 2.0)? - x, a.clone(), b.clone());
    // Same tensor used twice accumulates.
    check1(|x| x.mul(x)?.add(x), a.clone());
}

#[test]
fn views_and_copies() {
    let x = randn(&[2, 3, 4], 20);
    check1(|x| x.reshape([4, -1]), x.clone());
    check1(|x| x.transpose(0, 2)?.reshape([-1]), x.clone());
    check1(|x| x.permute([2, 0, 1]), x.clone());
    check1(|x| x.narrow(1, 1, 2), x.clone());
    check1(|x| x.slice(2, 0, 4, 3), x.clone());
    check1(|x| x.unsqueeze(1)?.squeeze(1), x.clone());
    check1(|x| x.flatten(1, 2), x.clone());
    check1(|x| x.narrow(0, 0, 1)?.broadcast_to([3, 3, 4]), x.clone());
    check1(|x| x.t()?.contiguous(), x.clone());
    check1(|x| x.pad(&[(1, 0), (2, 1)], 0.5), x.clone());
    check1(|x| x.repeat(&[2, 1, 2, 1]), x.clone());
    check1(|x| Ok(x.split(2, 2)?.remove(1)), x.clone());
    check1(|x| Ok(x.chunk(2, 1)?.remove(0)), x.clone());
    let y = randn(&[2, 1, 4], 21);
    grad_check(|v| Tensor::cat(&[&v[0], &v[1], &v[0]], 1), &[x.clone(), y]).unwrap();
    grad_check(|v| Tensor::stack(&[&v[0], &v[1]], 3), &[x.clone(), randn(&[2, 3, 4], 22)]).unwrap();
}

#[test]
fn dtype_roundtrip_grad() {
    // Finite differences through f16 rounding are meaningless; check the pass-through gradient directly.
    use folo_core::DType;
    let x = randn(&[3, 4], 23).requires_grad_();
    let y = x.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
    let g = y.mul_scalar(2.0).unwrap().sum_all().unwrap().backward().unwrap();
    assert_eq!(g.get(&x).unwrap().to_vec::<f32>().unwrap(), vec![2.0; 12]);
}

#[test]
fn indexing_ops() {
    let x = randn(&[4, 5], 30);
    let idx = Tensor::from_slice(&[3i64, 0, 3, 1], [4]).unwrap();
    grad_check(|v| v[0].index_select(0, &v[1]), &[x.clone(), idx.clone()]).unwrap();
    grad_check(|v| v[0].index_select(1, &v[1]), &[x.clone(), idx.clone()]).unwrap();
    let gidx = Tensor::from_slice(&[0i64, 4, 4, 1, 2, 2, 3, 0], [4, 2]).unwrap();
    grad_check(|v| v[0].gather(1, &v[1]), &[x.clone(), gidx.clone()]).unwrap();
    let src = randn(&[4, 2], 31);
    grad_check(|v| v[0].scatter_add(1, &v[2], &v[1]), &[x.clone(), src, gidx]).unwrap();
    let rows = randn(&[4, 5], 32);
    grad_check(|v| v[0].index_add(0, &v[2], &v[1]), &[x.clone(), rows, idx]).unwrap();
    let mask = uniform(&[4, 1], -1.0, 1.0, 33).gt_scalar(0.0).unwrap();
    grad_check(|v| mask.where_cond(&v[0], &v[1]), &[x.clone(), randn(&[5], 34)]).unwrap();
    check1(|x| x.masked_fill(&mask, 2.0), x.clone());
    check1(|x| Ok(x.topk(2, true)?.0), grid(&[4, 5], 35));
    check1(|x| Ok(x.topk(3, false)?.0), grid(&[4, 5], 36));
}

#[test]
fn reductions_and_softmax() {
    let x = randn(&[3, 4, 5], 40);
    check1(|x| x.sum(1, false), x.clone());
    check1(|x| x.sum([0, 2], true), x.clone());
    check1(|x| x.sum_all(), x.clone());
    check1(|x| x.mean([1, 2], false), x.clone());
    check1(|x| x.var(2, true, false), x.clone());
    check1(|x| x.var([0, 1], false, true), x.clone());
    let gx = grid(&[3, 4, 5], 41);
    check1(|x| x.max(2, false), gx.clone());
    check1(|x| x.min([0, 1], true), gx.clone());
    check1(|x| x.max_all(), gx.clone());
    check1(|x| x.softmax(-1), x.clone());
    check1(|x| x.softmax(1), x.clone());
    check1(|x| x.log_softmax(-1), x.clone());
    check1(|x| x.log_softmax(0), x.clone());
    check1(|x| x.logsumexp(2, false), x.clone());
    check1(|x| x.logsumexp([0, 1], true), x.clone());
    check1(|x| x.transpose(0, 2)?.softmax(-1), x.clone());
}

#[test]
fn matmul_grads() {
    let g = |a: Tensor, b: Tensor| grad_check(|v| v[0].matmul(&v[1]), &[a, b]).unwrap();
    g(randn(&[3, 4], 50), randn(&[4, 5], 51));
    g(randn(&[2, 3, 4], 52), randn(&[4, 5], 53));
    g(randn(&[2, 3, 4], 54), randn(&[2, 4, 5], 55));
    g(randn(&[2, 1, 3, 4], 56), randn(&[3, 4, 2], 57));
    g(randn(&[3, 4], 58), randn(&[2, 4, 5], 59));
    // Transposed operands.
    grad_check(|v| v[0].t()?.matmul(&v[1].t()?), &[randn(&[4, 3], 60), randn(&[5, 4], 61)]).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn conv_case(
    n: usize,
    c: usize,
    o: usize,
    hw: [usize; 2],
    k: [usize; 2],
    p: Conv2dParams,
    bias: bool,
    seed: u64,
) {
    let x = randn(&[n, c, hw[0], hw[1]], seed);
    let w = randn(&[o, c / p.groups, k[0], k[1]], seed + 1).mul_scalar(0.5).unwrap();
    let opts = GradCheck { max_checks: 24, ..GradCheck::default() };
    if bias {
        let b = randn(&[o], seed + 2);
        grad_check_with(|v| v[0].conv2d_with(&v[1], Some(&v[2]), &p), &[x, w, b], &opts).unwrap();
    } else {
        grad_check_with(|v| v[0].conv2d_with(&v[1], None, &p), &[x, w], &opts).unwrap();
    }
}

#[test]
fn conv2d_grads() {
    let p = |s: usize, pad: usize, d: usize, g: usize| Conv2dParams {
        stride: [s; 2],
        padding: [pad; 2],
        dilation: [d; 2],
        groups: g,
    };
    conv_case(2, 3, 4, [6, 5], [3, 3], p(1, 1, 1, 1), true, 70);
    conv_case(1, 4, 6, [7, 7], [3, 3], p(2, 1, 1, 2), false, 71);
    conv_case(1, 2, 3, [8, 7], [3, 2], p(1, 2, 2, 1), true, 72);
    conv_case(2, 4, 4, [5, 6], [3, 3], p(1, 1, 1, 4), true, 73); // depthwise
    conv_case(1, 3, 6, [6, 6], [3, 3], p(2, 1, 1, 3), false, 74); // depthwise, multiplier 2, stride 2
    conv_case(2, 3, 5, [4, 5], [1, 1], p(1, 0, 1, 1), true, 75); // pointwise
    conv_case(1, 2, 2, [5, 5], [1, 1], p(2, 1, 1, 1), false, 76); // 1x1 with stride and padding
    let asym = Conv2dParams { stride: [1, 2], padding: [0, 1], dilation: [2, 1], groups: 1 };
    conv_case(1, 2, 3, [7, 6], [2, 3], asym, true, 77);
    // conv2d_act records the composite conv2d + activation, so it trains like the separate ops.
    let opts = GradCheck { max_checks: 24, ..GradCheck::default() };
    let (x, w, b) =
        (randn(&[1, 3, 5, 5], 78), randn(&[4, 3, 3, 3], 79).mul_scalar(0.5).unwrap(), randn(&[4], 80));
    let act = Some(folo_core::UnaryOp::Silu);
    grad_check_with(|v| v[0].conv2d_act(&v[1], Some(&v[2]), &p(1, 1, 1, 1), act), &[x, w, b], &opts).unwrap();
    // Non-contiguous input.
    let x = randn(&[1, 2, 5, 6], 78);
    let w = randn(&[3, 2, 3, 3], 79);
    grad_check(|v| v[0].transpose(2, 3)?.conv2d(&v[1], None, 1, 1, 1, 1), &[x, w]).unwrap();
}

#[test]
fn pooling_and_upsample_grads() {
    let x = grid(&[2, 3, 7, 6], 80).mul_scalar(0.2).unwrap();
    check1(|x| x.max_pool2d(2, 2, 0), x.clone());
    check1(|x| x.max_pool2d(3, 2, 1), x.clone());
    check1(|x| x.max_pool2d(5, 1, 2), x.clone()); // SPPF-style
    check1(|x| x.avg_pool2d(2, 2, 0), x.clone());
    check1(|x| x.avg_pool2d(3, 1, 1), x.clone());
    check1(|x| x.adaptive_avg_pool2d([1, 1]), x.clone());
    check1(|x| x.adaptive_avg_pool2d([3, 4]), x.clone());
    check1(|x| x.upsample_nearest2d(2), x.clone());
    check1(|x| x.upsample_nearest2d_hw([1, 3]), x.clone());
    let asym = Pool2dParams { kernel: [2, 3], stride: [1, 2], padding: [1, 1] };
    check1(|x| x.max_pool2d_with(&asym), x.clone());
    check1(|x| x.avg_pool2d_with(&asym), x.clone());
}

#[test]
fn composite_network_grad() {
    // A miniature conv block + head, as a smoke test of everything composed.
    let x = randn(&[2, 3, 8, 8], 90);
    let w1 = randn(&[4, 3, 3, 3], 91).mul_scalar(0.3).unwrap();
    let w2 = randn(&[2, 4, 1, 1], 92).mul_scalar(0.3).unwrap();
    let opts = GradCheck { max_checks: 16, ..GradCheck::default() };
    grad_check_with(
        |v| {
            let h = v[0].conv2d(&v[1], None, 1, 1, 1, 1)?.silu()?.avg_pool2d(2, 2, 0)?;
            let u = h.upsample_nearest2d(2)?;
            let h = Tensor::cat(&[&u, &v[0].narrow(1, 0, 1)?.repeat(&[1, 4, 1, 1])?], 1)?;
            let h = h.narrow(1, 0, 4)?.conv2d(&v[2], None, 1, 0, 1, 1)?;
            h.flatten(2, 3)?.log_softmax(-1)
        },
        &[x, w1, w2],
        &opts,
    )
    .unwrap();
}
