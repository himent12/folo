//! Unary, broadcasting binary and fused bias + activation against CPU references.

use super::{Rng, assert_close, gpu};
use crate::kernels::elementwise::broadcast_shape;
use crate::{BinaryOp, GpuTensor, UnaryOp};

const UNARY: [UnaryOp; 22] = [
    UnaryOp::Identity,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sqrt,
    UnaryOp::Rsqrt,
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Sigmoid,
    UnaryOp::Silu,
    UnaryOp::Relu,
    UnaryOp::Tanh,
    UnaryOp::Gelu,
    UnaryOp::LeakyRelu(0.1),
    UnaryOp::Hardswish,
    UnaryOp::Hardsigmoid,
    UnaryOp::Square,
    UnaryOp::Recip,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Relu6,
];

/// Inputs in the op's domain: positive for log / sqrt / rsqrt / recip, wide for the activations.
fn input(op: UnaryOp, rng: &mut Rng, n: usize) -> Vec<f32> {
    match op {
        UnaryOp::Log | UnaryOp::Sqrt | UnaryOp::Rsqrt | UnaryOp::Recip => rng.vec(n, 0.05, 8.0),
        UnaryOp::Exp => rng.vec(n, -10.0, 10.0),
        _ => rng.vec(n, -8.0, 8.0),
    }
}

#[test]
fn unary_ops_match_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(1);
    // 1027 elements: the scalar path; 1024: the vec4 path.
    for n in [1027, 1024, 1] {
        for op in UNARY {
            let x = input(op, &mut rng, n);
            let want: Vec<f32> = x.iter().map(|&v| op.apply(v)).collect();
            let got = GpuTensor::from_slice(&ctx, &x, &[n]).unwrap().unary(op).unwrap().to_vec().unwrap();
            // Transcendental builtins are accurate to a few ulp on hardware; sin / cos only promise an absolute
            // error (WGSL: 2^-11) and software adapters (WARP) are off by ~2e-6 there.
            let atol = if matches!(op, UnaryOp::Sin | UnaryOp::Cos) { 1e-5 } else { 2e-6 };
            assert_close(&got, &want, atol, 2e-6, &format!("{op:?} n={n}"));
        }
    }
}

#[test]
fn unary_extremes_are_finite() {
    let ctx = gpu!();
    let x = [-1e4f32, -100.0, -30.0, 0.0, 30.0, 100.0, 1e4];
    for op in [UnaryOp::Sigmoid, UnaryOp::Silu, UnaryOp::Tanh, UnaryOp::Gelu] {
        let got = GpuTensor::from_slice(&ctx, &x, &[7]).unwrap().unary(op).unwrap().to_vec().unwrap();
        let want: Vec<f32> = x.iter().map(|&v| op.apply(v)).collect();
        assert!(got.iter().all(|v| v.is_finite()), "{op:?}: {got:?}");
        assert_close(&got, &want, 1e-5, 1e-6, &format!("{op:?} extremes"));
    }
}

/// Reference broadcasting binary op.
fn cpu_binary(op: BinaryOp, a: &[f32], sa: &[usize], b: &[f32], sb: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let out = broadcast_shape(sa, sb).unwrap();
    let r = out.len();
    let pad = |s: &[usize]| [vec![1; r - s.len()], s.to_vec()].concat();
    let (pa, pb) = (pad(sa), pad(sb));
    let n: usize = out.iter().product();
    let mut res = Vec::with_capacity(n);
    for i in 0..n {
        let (mut rem, mut ia, mut ib, mut st_a, mut st_b) = (i, 0, 0, 1, 1);
        for d in (0..r).rev() {
            let c = rem % out[d];
            rem /= out[d];
            if pa[d] != 1 {
                ia += c * st_a;
            }
            if pb[d] != 1 {
                ib += c * st_b;
            }
            st_a *= pa[d];
            st_b *= pb[d];
        }
        res.push(op.apply(a[ia], b[ib]));
    }
    (res, out)
}

#[test]
fn binary_broadcasting_matches_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(2);
    let cases: &[(&[usize], &[usize])] = &[
        (&[1000], &[1000]),
        (&[1024], &[1024]),
        (&[2, 3, 5, 7], &[1, 3, 1, 1]),
        (&[4, 1, 6], &[5, 1]),
        (&[7], &[1]),
        (&[1], &[3, 4]),
        (&[2, 3, 4, 5, 6], &[3, 1, 5, 1]),
        (&[1, 1, 1], &[1]),
        (&[3, 1, 2, 1, 2, 1, 2, 1], &[1, 2, 1, 2, 1, 2, 1, 2]),
    ];
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Max, BinaryOp::Min] {
        for &(sa, sb) in cases {
            let a = rng.vec(sa.iter().product(), -4.0, 4.0);
            let mut b = rng.vec(sb.iter().product(), -4.0, 4.0);
            if op == BinaryOp::Div {
                b.iter_mut().for_each(|v| *v = if v.abs() < 0.1 { 0.5 } else { *v });
            }
            let (want, shape) = cpu_binary(op, &a, sa, &b, sb);
            let ga = GpuTensor::from_slice(&ctx, &a, sa).unwrap();
            let gb = GpuTensor::from_slice(&ctx, &b, sb).unwrap();
            let got = ga.binary(op, &gb).unwrap();
            assert_eq!(got.shape(), shape.as_slice());
            assert_close(&got.to_vec().unwrap(), &want, 1e-6, 1e-6, &format!("{op:?} {sa:?} {sb:?}"));
        }
    }
    let ga = GpuTensor::from_slice(&ctx, &[1.0; 6], &[2, 3]).unwrap();
    let gb = GpuTensor::from_slice(&ctx, &[1.0; 4], &[4]).unwrap();
    assert!(ga.add(&gb).is_err());
}

#[test]
fn pow_follows_powf() {
    let ctx = gpu!();
    let a = [2.0f32, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 3.0, 0.5, -0.5, 4.0, -8.0];
    let b = [3.0f32, 3.0, 2.0, 0.5, 2.0, 0.0, -1.0, 0.0, -2.0, -3.0, 0.5, 1.0 / 3.0];
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x.powf(*y)).collect();
    let ga = GpuTensor::from_slice(&ctx, &a, &[12]).unwrap();
    let gb = GpuTensor::from_slice(&ctx, &b, &[12]).unwrap();
    let got = ga.binary(BinaryOp::Pow, &gb).unwrap().to_vec().unwrap();
    assert_close(&got, &want, 1e-5, 1e-5, "pow");
    // C99 Annex F special cases must match bit for bit (signed zeros and infinities included).
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let special = [
        (-0.0f32, -3.0f32),
        (-0.0, 3.0),
        (-0.0, 2.0),
        (0.0, -2.5),
        (1.0, inf),
        (1.0, nan),
        (nan, 0.0),
        (nan, -0.0),
        (nan, 2.0),
        (2.0, nan),
        (-1.0, inf),
        (-1.0, -inf),
        (0.5, inf),
        (0.5, -inf),
        (2.0, inf),
        (-2.0, -inf),
        (-inf, 3.0),
        (-inf, -3.0),
        (-inf, 2.0),
        (-inf, -0.5),
        (inf, -1.0),
        (inf, 0.5),
        (-2.0, 0.5),
        (-8.0, 3.0),
        (-0.5, -3.0),
    ];
    let (xs, ys): (Vec<f32>, Vec<f32>) = special.iter().copied().unzip();
    let ga = GpuTensor::from_slice(&ctx, &xs, &[xs.len()]).unwrap();
    let gb = GpuTensor::from_slice(&ctx, &ys, &[ys.len()]).unwrap();
    let got = ga.binary(BinaryOp::Pow, &gb).unwrap().to_vec().unwrap();
    for ((x, y), g) in special.iter().zip(&got) {
        let w = x.powf(*y);
        let same = if w.is_nan() {
            g.is_nan()
        } else if w == 0.0 || w.is_infinite() {
            g.to_bits() == w.to_bits()
        } else {
            (g - w).abs() <= 1e-5 * w.abs()
        };
        assert!(same, "pow({x}, {y}): got {g}, want {w}");
    }
    let mut rng = Rng::new(3);
    let x = rng.vec(4 * 33, 0.1, 3.0);
    let e = rng.vec(33, -2.0, 2.0);
    let (want, _) = cpu_binary(BinaryOp::Pow, &x, &[4, 33], &e, &[33]);
    let got = GpuTensor::from_slice(&ctx, &x, &[4, 33])
        .unwrap()
        .binary(BinaryOp::Pow, &GpuTensor::from_slice(&ctx, &e, &[33]).unwrap())
        .unwrap();
    assert_close(&got.to_vec().unwrap(), &want, 1e-5, 1e-5, "pow broadcast");
}

#[test]
fn bias_activation_fused() {
    let ctx = gpu!();
    let mut rng = Rng::new(4);
    // NCHW with odd plane size (scalar path) and even (vec4 path), plus [rows, features].
    for (shape, dim) in [(vec![2, 5, 3, 3], 1), (vec![2, 6, 4, 4], 1), (vec![7, 12], 1), (vec![3, 9], 1)] {
        let n: usize = shape.iter().product();
        let x = rng.vec(n, -3.0, 3.0);
        let c = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let bias = rng.vec(c, -1.0, 1.0);
        for act in [UnaryOp::Silu, UnaryOp::Relu, UnaryOp::Identity, UnaryOp::LeakyRelu(0.2)] {
            let want: Vec<f32> = (0..n).map(|i| act.apply(x[i] + bias[(i / inner) % c])).collect();
            let gx = GpuTensor::from_slice(&ctx, &x, &shape).unwrap();
            let gb = GpuTensor::from_slice(&ctx, &bias, &[c]).unwrap();
            let got = gx.bias_act(Some(&gb), dim, act).unwrap().to_vec().unwrap();
            assert_close(&got, &want, 1e-6, 1e-6, &format!("bias_act {shape:?} {act:?}"));
            let got = gx.bias_act(None, dim, act).unwrap().to_vec().unwrap();
            let want: Vec<f32> = x.iter().map(|&v| act.apply(v)).collect();
            assert_close(&got, &want, 1e-6, 1e-6, &format!("act only {shape:?} {act:?}"));
        }
    }
}
