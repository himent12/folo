//! Unary, binary (broadcasting) and `where` ops with their gradients.

use folo_core::backend::UnaryOp as U;
use folo_core::{DType, Result, Tensor};

use super::fixtures::{assert_close_t, check, dev, randn, uniform};

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
fn broadcast_vec4_binary_and_where() {
    let dev = dev!();
    // NCHW bias/scale (innermost HW = 64 multiple of 4; rhs broadcast with odd offset via narrow).
    let x = randn(&[2, 5, 8, 8], 20);
    let bias_full = randn(&[1, 7, 1, 1], 21);
    check(dev, &[x.clone(), bias_full.clone()], 1e-5, 1e-5, "nchw vec4 bcast odd offset", |v| {
        let b = v[1].narrow(1, 1, 5)?;
        v[0].add(&b)?.mul(&b)
    });
    // Reversed operand order (lhs broadcast, rhs contiguous vec4) and trailing-4 collapse ([5, 4] + [5, 1]).
    check(dev, &[bias_full, x.clone()], 1e-5, 1e-5, "lhs bcast vec4", |v| v[0].narrow(1, 1, 5)?.sub(&v[1]));
    let m4 = randn(&[5, 4], 22);
    let col = randn(&[5, 1], 23);
    check(dev, &[m4, col], 1e-5, 1e-5, "trailing 4 collapse", |v| v[0].add(&v[1]));
    // 1D scalar broadcast (RANK = 0, VEC4 = true, BCAST = 0b10 and 0b01) + non-4-aligned vector offset fallback.
    let v128 = randn(&[130], 24);
    let s1 = randn(&[1], 25);
    check(dev, &[v128.clone(), s1.clone()], 1e-5, 1e-5, "1d scalar bcast vec4", |v| {
        v[0].narrow(0, 0, 128)?.add(&v[1])
    });
    check(dev, &[v128, s1], 1e-5, 1e-5, "1d unaligned offset fallback", |v| {
        v[0].narrow(0, 2, 128)?.add(&v[1])
    });
    // Outer broadcast with contiguous vec4 inner dim ([4, 1, 16] + [1, 3, 16]).
    let p = randn(&[4, 1, 16], 26);
    let q = randn(&[1, 3, 16], 27);
    check(dev, &[p, q], 1e-5, 1e-5, "outer bcast vec4", |v| v[0].mul(&v[1]));

    // `where` with u8 and f32 conditions: contiguous vec4, broadcast cond, broadcast values, -0.0 and NaN conds.
    let y = randn(&[2, 5, 8, 8], 28);
    let c_ch = randn(&[1, 5, 1, 1], 29);
    let val_ch = randn(&[1, 5, 1, 1], 30);
    check(dev, &[x.clone(), y.clone(), c_ch.clone(), val_ch.clone()], 0.0, 0.0, "where u8 vec4 dense", |v| {
        v[0].gt(&v[1])?.to_dtype(DType::U8)?.where_cond(&v[0], &v[1])
    });
    check(
        dev,
        &[x.clone(), y.clone(), c_ch.clone(), val_ch.clone()],
        1e-5,
        1e-5,
        "where u8 vec4 bcast cond+rhs",
        |v| v[2].gt_scalar(0.0)?.to_dtype(DType::U8)?.where_cond(&v[0], &v[3]),
    );
    check(
        dev,
        &[x.clone(), y.clone(), c_ch.clone(), val_ch.clone()],
        1e-5,
        1e-5,
        "where f32 vec4 bcast lhs",
        |v| v[0].gt_scalar(0.0)?.where_cond(&v[3], &v[1]),
    );
    check(dev, &[x, y, c_ch, val_ch], 0.0, 0.0, "where f32 vec4 bcast cond", |v| {
        v[2].gt_scalar(0.0)?.where_cond(&v[0], &v[1])
    });
    // Special f32 condition bits (-0.0 is false, NaN is true) in both dense and broadcast vec4 paths.
    let cond_spec =
        Tensor::from_slice(&[0.0f32, -0.0, f32::NAN, 1.0, -0.0, 2.0, 0.0, f32::NAN], [2, 4]).unwrap();
    let t_val = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]).unwrap();
    let f_val = Tensor::from_slice(&[-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], [2, 4]).unwrap();
    assert_close_t(
        &cond_spec
            .to_device(dev)
            .unwrap()
            .where_cond(&t_val.to_device(dev).unwrap(), &f_val.to_device(dev).unwrap())
            .unwrap(),
        &cond_spec.where_cond(&t_val, &f_val).unwrap(),
        0.0,
        0.0,
        "where f32 special bits vec4",
    );
    let cond_bcast = Tensor::from_slice(&[-0.0f32, f32::NAN], [2, 1]).unwrap();
    assert_close_t(
        &cond_bcast
            .to_device(dev)
            .unwrap()
            .where_cond(&t_val.to_device(dev).unwrap(), &f_val.to_device(dev).unwrap())
            .unwrap(),
        &cond_bcast.where_cond(&t_val, &f_val).unwrap(),
        0.0,
        0.0,
        "where f32 special bits bcast vec4",
    );
}
