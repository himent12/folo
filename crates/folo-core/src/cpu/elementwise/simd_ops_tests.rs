//! Exhaustive bitwise correctness and edge-case tests for AVX2 elementwise activations, math/arithmetic ops,
//! binary ops, bias epilogues, and `where_cond` kernels.

use super::*;
use crate::Shape;

fn assert_bits_or_nan(got: f32, want: f32, label: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{label}: expected NaN, got {got} ({:#010x})", got.to_bits());
    } else {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}: got {got} ({:#010x}), want {want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

fn edge_f32_values() -> Vec<f32> {
    let mut v = vec![
        f32::NEG_INFINITY,
        f32::MIN,
        -1e20,
        -100.0,
        -3.5,
        -3.0,
        -2.999_999_5,
        -1.0,
        -0.5,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.25,
        0.5,
        1.0,
        2.999_999_5,
        3.0,
        3.5,
        100.0,
        1e20,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xffc0_4321),
    ];
    // Pad with varied values so total length hits 16-wide, 8-wide, and unaligned tail (mod 16 == 13).
    while v.len() < 45 {
        let i = v.len() as f32;
        v.push((i * 0.67).sin() * 5.0);
    }
    v
}

fn scalar_unary_ref(op: UnaryOp, v: f32) -> f32 {
    use UnaryOp::*;
    match op {
        Neg => -v,
        Abs => v.abs(),
        Sqrt => v.sqrt(),
        Rsqrt => 1.0 / v.sqrt(),
        Recip => 1.0 / v,
        Square => v * v,
        Relu => max_num(v, 0.0),
        LeakyRelu(a) => {
            if v > 0.0 {
                v
            } else {
                a * v
            }
        }
        Hardswish => v * (v + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0),
        Clamp(lo, hi) => min_num(max_num(v, lo), hi),
        PowScalar(2.0) => v * v,
        PowScalar(0.5) => v.sqrt(),
        AddScalar(c) => v + c,
        MulScalar(c) => v * c,
        RSubScalar(c) => c - v,
        RDivScalar(c) => c / v,
        DivScalar(c) => v / c,
        MaxScalar(c) => max_num(v, c),
        MinScalar(c) => min_num(v, c),
        _ => unreachable!(),
    }
}

fn scalar_unary_bwd_ref(op: UnaryOp, x: f32, g: f32) -> f32 {
    use UnaryOp::*;
    match op {
        Neg | RSubScalar(_) => -g,
        AddScalar(_) => g,
        MulScalar(c) => g * c,
        DivScalar(c) => g / c,
        RDivScalar(c) => -(g * (c / x)) / x,
        Sqrt => g * 0.5 / x.sqrt(),
        Rsqrt => -0.5 * g / x.sqrt() / x,
        Recip => -(g / x) / x,
        Square => 2.0 * x * g,
        Relu => {
            if x > 0.0 {
                g
            } else {
                0.0
            }
        }
        LeakyRelu(a) => {
            if x > 0.0 {
                g
            } else {
                a * g
            }
        }
        Hardswish => {
            if x < -3.0 {
                0.0
            } else if x > 3.0 {
                g
            } else {
                g * (2.0 * x + 3.0) * (1.0 / 6.0)
            }
        }
        Clamp(lo, hi) => {
            if x >= lo && x <= hi {
                g
            } else {
                0.0
            }
        }
        MaxScalar(c) => {
            if x >= c {
                g
            } else {
                0.0
            }
        }
        MinScalar(c) => {
            if x <= c {
                g
            } else {
                0.0
            }
        }
        _ => unreachable!(),
    }
}

fn scalar_binary_ref(op: BinaryOp, a: f32, b: f32) -> f32 {
    use BinaryOp::*;
    match op {
        Add => a + b,
        Sub => a - b,
        Mul => a * b,
        Div => a / b,
        Maximum => max_num(a, b),
        Minimum => min_num(a, b),
        Eq => mask(a == b),
        Ne => mask(a != b),
        Gt => mask(a > b),
        Ge => mask(a >= b),
        Lt => mask(a < b),
        Le => mask(a <= b),
        _ => unreachable!(),
    }
}

#[test]
fn avx2_unary_ops_bit_exact_across_edges_tails_and_layouts() {
    use UnaryOp::*;
    let ops = [
        Neg,
        Abs,
        Sqrt,
        Rsqrt,
        Recip,
        Square,
        Relu,
        LeakyRelu(0.01),
        LeakyRelu(-0.2),
        Hardswish,
        Clamp(-1.5, 2.5),
        Clamp(-0.0, 0.0),
        Clamp(0.0, 0.0),
        Clamp(f32::NAN, 1.0),
        Clamp(-1.0, f32::NAN),
        PowScalar(2.0),
        PowScalar(0.5),
        AddScalar(1.25),
        MulScalar(-0.75),
        RSubScalar(3.0),
        RDivScalar(2.5),
        DivScalar(0.5),
        MaxScalar(0.0),
        MaxScalar(-0.0),
        MaxScalar(f32::NAN),
        MinScalar(0.0),
        MinScalar(-0.0),
        MinScalar(f32::NAN),
    ];

    let xs = edge_f32_values();
    for &op in &ops {
        // Test every length 0..=33 to cover empty, scalar, tail-only, 8-wide, 16-wide, and combined paths.
        for len in 0..=33 {
            let l = Layout::contiguous(Shape::new(&[len]));
            let got = unary(op, &xs[..len], &l);
            assert_eq!(got.len(), len);
            for (i, &y) in got.iter().enumerate() {
                assert_bits_or_nan(y, scalar_unary_ref(op, xs[i]), &format!("{op:?} len={len} i={i}"));
            }
        }
        // Unit-stride non-contiguous rows, broadcast (stride 0), and transposed (non-unit stride).
        let (rows, cols, stride) = (2usize, 19usize, 22usize);
        let l_unit = Layout::new(Shape::new(&[rows, cols]), vec![stride, 1].into(), 1);
        let l_bc = Layout::new(Shape::new(&[rows, cols]), vec![1, 0].into(), 3);
        let l_trans = Layout::new(Shape::new(&[cols, rows]), vec![1, stride].into(), 1);
        let got_u = unary(op, &xs, &l_unit);
        let got_b = unary(op, &xs, &l_bc);
        let got_t = unary(op, &xs, &l_trans);
        for r in 0..rows {
            for c in 0..cols {
                let xv = xs[1 + r * stride + c];
                let want = scalar_unary_ref(op, xv);
                assert_bits_or_nan(got_u[r * cols + c], want, &format!("{op:?} unit ({r},{c})"));
                assert_bits_or_nan(got_t[c * rows + r], want, &format!("{op:?} trans ({r},{c})"));
                let want_bc = scalar_unary_ref(op, xs[3 + r]);
                assert_bits_or_nan(got_b[r * cols + c], want_bc, &format!("{op:?} bc ({r},{c})"));
            }
        }
    }

    // Parallel contiguous (> PAR_MIN) with odd tail length.
    let n_par = PAR_MIN + 29;
    let x_par: Vec<f32> = (0..n_par).map(|i| xs[i % xs.len()]).collect();
    let l_par = Layout::contiguous(Shape::new(&[n_par]));
    for op in [Relu, LeakyRelu(0.1), Hardswish, Clamp(-1.0, 1.0), Sqrt, Rsqrt, Recip] {
        let got = unary(op, &x_par, &l_par);
        for i in 0..n_par {
            assert_bits_or_nan(got[i], scalar_unary_ref(op, x_par[i]), &format!("par {op:?} i={i}"));
        }
    }
}

#[test]
fn avx2_unary_backward_bit_exact_across_edges_and_broadcasts() {
    use UnaryOp::*;
    let ops = [
        Neg,
        AddScalar(1.0),
        MulScalar(-2.5),
        RSubScalar(4.0),
        RDivScalar(3.0),
        DivScalar(2.0),
        Sqrt,
        Rsqrt,
        Recip,
        Square,
        Relu,
        LeakyRelu(0.05),
        Hardswish,
        Clamp(-1.5, 2.5),
        Clamp(f32::NAN, 1.0),
        MaxScalar(-0.0),
        MinScalar(0.0),
    ];

    let xs = edge_f32_values();
    let gs: Vec<f32> = (0..xs.len()).map(|i| xs[(i * 7 + 3) % xs.len()]).collect();

    let (rows, cols, stride) = (2usize, 19usize, 22usize);
    let l_u = Layout::new(Shape::new(&[rows, cols]), vec![stride, 1].into(), 1);
    let l_bc = Layout::new(Shape::new(&[rows, cols]), vec![1, 0].into(), 2);
    let l_t = Layout::new(Shape::new(&[cols, rows]), vec![1, stride].into(), 1);
    let l_empty = Layout::contiguous(Shape::new(&[0, cols]));

    for &op in &ops {
        // Contiguous with lengths 0..=25.
        for len in 0..=25 {
            let l = Layout::contiguous(Shape::new(&[len]));
            let got = unary_backward(op, &xs[..len], &l, &gs[..len], &l);
            for i in 0..len {
                assert_bits_or_nan(
                    got[i],
                    scalar_unary_bwd_ref(op, xs[i], gs[i]),
                    &format!("bwd {op:?} len={len} i={i}"),
                );
            }
        }
        // (1, 1), (1, 0), (0, 1), (0, 0), general stride, and empty.
        let g11 = unary_backward(op, &xs, &l_u, &gs, &l_u);
        let g10 = unary_backward(op, &xs, &l_u, &gs, &l_bc);
        let g01 = unary_backward(op, &xs, &l_bc, &gs, &l_u);
        let g00 = unary_backward(op, &xs, &l_bc, &gs, &l_bc);
        let gtt = unary_backward(op, &xs, &l_t, &gs, &l_t);
        assert!(unary_backward(op, &[], &l_empty, &[], &l_empty).is_empty());
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let (xu, gu) = (xs[1 + r * stride + c], gs[1 + r * stride + c]);
                let (xb, gb) = (xs[2 + r], gs[2 + r]);
                assert_bits_or_nan(g11[idx], scalar_unary_bwd_ref(op, xu, gu), &format!("bwd (1,1) {op:?}"));
                assert_bits_or_nan(g10[idx], scalar_unary_bwd_ref(op, xu, gb), &format!("bwd (1,0) {op:?}"));
                assert_bits_or_nan(g01[idx], scalar_unary_bwd_ref(op, xb, gu), &format!("bwd (0,1) {op:?}"));
                assert_bits_or_nan(g00[idx], scalar_unary_bwd_ref(op, xb, gb), &format!("bwd (0,0) {op:?}"));
                assert_bits_or_nan(
                    gtt[c * rows + r],
                    scalar_unary_bwd_ref(op, xu, gu),
                    &format!("bwd trans {op:?}"),
                );
            }
        }
    }
}

#[test]
fn avx2_bias_act_inplace_matches_scalar_across_segments() {
    let xs = edge_f32_values();
    let ops = [None, Some(UnaryOp::Relu), Some(UnaryOp::LeakyRelu(0.1)), Some(UnaryOp::Hardswish)];
    for bias in [-1.5f32, 0.0, 0.25] {
        for op in ops {
            let mut got = xs.clone();
            bias_act_inplace(op, bias, &mut got);
            for (&x, &y) in xs.iter().zip(&got) {
                let want = match op {
                    None if bias == 0.0 => x,
                    None => x + bias,
                    Some(u) => scalar_unary_ref(u, x + bias),
                };
                assert_bits_or_nan(y, want, &format!("bias_act {op:?} bias={bias} x={x}"));
            }
            // Segment independence (lengths 1..19).
            let mut pieces = xs.clone();
            let mut at = 0;
            for len in (1..19).cycle() {
                if at >= pieces.len() {
                    break;
                }
                let end = (at + len).min(pieces.len());
                bias_act_inplace(op, bias, &mut pieces[at..end]);
                at = end;
            }
            for (i, (&a, &b)) in got.iter().zip(&pieces).enumerate() {
                assert_bits_or_nan(a, b, &format!("bias_act seg {op:?} i={i}"));
            }
        }
    }
}

#[test]
fn avx2_binary_ops_bit_exact_across_all_pairs_and_layouts() {
    use BinaryOp::*;
    let ops = [Add, Sub, Mul, Div, Maximum, Minimum, Eq, Ne, Gt, Ge, Lt, Le];

    // Cartesian product of key IEEE-754 edge values so every (NaN, +-0, +-inf, subnormal) pair is checked!
    let atoms = [
        f32::NEG_INFINITY,
        -2.5,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        2.5,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1111),
    ];
    let mut a_all = Vec::with_capacity(atoms.len() * atoms.len() + 11);
    let mut b_all = Vec::with_capacity(atoms.len() * atoms.len() + 11);
    for &x in &atoms {
        for &y in &atoms {
            a_all.push(x);
            b_all.push(y);
        }
    }
    // Add 11 elements so length = 111 (6 * 16 + 8 + 7, exercising 16-wide, 8-wide, and 7-lane tail).
    for k in 0..11 {
        a_all.push((k as f32) - 5.0);
        b_all.push(2.0 - (k as f32) * 0.5);
    }
    assert_eq!(a_all.len() % 16, 15);
    // Explicit literal-bit assertions for the tie-to-right-operand signed-zero rule across SIMD and tail lanes.
    let pz = 0.0f32.to_bits();
    let nz = (-0.0f32).to_bits();
    let sz_a = [0.0f32, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0];
    let sz_b = [-0.0f32, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0];
    let l_sz = Layout::contiguous(Shape::new(&[sz_a.len()]));
    let max_sz = binary(Maximum, &sz_a, &l_sz, &sz_b, &l_sz);
    let min_sz = binary(Minimum, &sz_a, &l_sz, &sz_b, &l_sz);
    let relu_sz = unary(UnaryOp::Relu, &sz_a, &l_sz);
    let clamp_sz = unary(UnaryOp::Clamp(-0.0, 0.0), &sz_a, &l_sz);
    for i in 0..sz_a.len() {
        let want_tie = if i % 2 == 0 { nz } else { pz };
        assert_eq!(max_sz[i].to_bits(), want_tie, "Maximum signed-zero tie at {i}");
        assert_eq!(min_sz[i].to_bits(), want_tie, "Minimum signed-zero tie at {i}");
        assert_eq!(relu_sz[i].to_bits(), pz, "Relu signed-zero at {i}");
        assert_eq!(clamp_sz[i].to_bits(), pz, "Clamp(-0.0, +0.0) signed-zero at {i}");
    }
    let l_all = Layout::contiguous(Shape::new(&[a_all.len()]));

    let (rows, cols, stride) = (3usize, 19usize, 23usize);
    let l_u = Layout::new(Shape::new(&[rows, cols]), vec![stride, 1].into(), 2);
    let l_bc = Layout::new(Shape::new(&[rows, cols]), vec![1, 0].into(), 4);
    let l_t = Layout::new(Shape::new(&[cols, rows]), vec![1, stride].into(), 2);
    let l_empty = Layout::contiguous(Shape::new(&[0, cols]));

    for &op in &ops {
        let got = binary(op, &a_all, &l_all, &b_all, &l_all);
        for i in 0..a_all.len() {
            assert_bits_or_nan(
                got[i],
                scalar_binary_ref(op, a_all[i], b_all[i]),
                &format!("bin {op:?} pair ({}, {})", a_all[i], b_all[i]),
            );
        }
        // Broadcast and strided layouts: (1, 1), (1, 0), (0, 1), (0, 0), transposed, empty.
        let r11 = binary(op, &a_all, &l_u, &b_all, &l_u);
        let r10 = binary(op, &a_all, &l_u, &b_all, &l_bc);
        let r01 = binary(op, &a_all, &l_bc, &b_all, &l_u);
        let r00 = binary(op, &a_all, &l_bc, &b_all, &l_bc);
        let rtt = binary(op, &a_all, &l_t, &b_all, &l_t);
        assert!(binary(op, &[], &l_empty, &[], &l_empty).is_empty());
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let (au, bu) = (a_all[2 + r * stride + c], b_all[2 + r * stride + c]);
                let (ab, bb) = (a_all[4 + r], b_all[4 + r]);
                assert_bits_or_nan(r11[idx], scalar_binary_ref(op, au, bu), &format!("bin (1,1) {op:?}"));
                assert_bits_or_nan(r10[idx], scalar_binary_ref(op, au, bb), &format!("bin (1,0) {op:?}"));
                assert_bits_or_nan(r01[idx], scalar_binary_ref(op, ab, bu), &format!("bin (0,1) {op:?}"));
                assert_bits_or_nan(r00[idx], scalar_binary_ref(op, ab, bb), &format!("bin (0,0) {op:?}"));
                assert_bits_or_nan(
                    rtt[c * rows + r],
                    scalar_binary_ref(op, au, bu),
                    &format!("bin trans {op:?}"),
                );
            }
        }
    }

    // Parallel contiguous (>= PAR_MIN) binary ops.
    let n_par = PAR_MIN + 31;
    let ap: Vec<f32> = (0..n_par).map(|i| a_all[i % a_all.len()]).collect();
    let bp: Vec<f32> = (0..n_par).map(|i| b_all[i % b_all.len()]).collect();
    let lp = Layout::contiguous(Shape::new(&[n_par]));
    for op in [Add, Mul, Div, Maximum, Minimum, Gt] {
        let got = binary(op, &ap, &lp, &bp, &lp);
        for i in 0..n_par {
            assert_bits_or_nan(got[i], scalar_binary_ref(op, ap[i], bp[i]), &format!("par bin {op:?} i={i}"));
        }
    }
}

#[test]
fn avx2_where3_f32_and_u8_bit_exact_across_all_stride_combos() {
    let (rows, cols, stride) = (4usize, 27usize, 31usize);
    let total = rows * stride + 12;
    let mut cf: Vec<f32> = (0..total).map(|i| if i % 3 == 0 { 0.0 } else { (i as f32) * 0.25 }).collect();
    cf[2] = -0.0;
    cf[3] = f32::NAN;
    cf[4] = f32::NEG_INFINITY;
    cf[5] = -f32::from_bits(1);
    let cu: Vec<u8> = (0..total)
        .map(|i| match i % 5 {
            0 => 0,
            1 => 1,
            2 => 127,
            3 => 128,
            _ => 255,
        })
        .collect();
    let mut a: Vec<f32> = (0..total).map(|i| i as f32 + 0.125).collect();
    a[3] = f32::from_bits(0x7fc0_2222);
    a[2] = -0.0;
    let mut b: Vec<f32> = (0..total).map(|i| -(i as f32) - 0.5).collect();
    b[2] = -0.0;
    b[4] = f32::from_bits(0x7fc0_3333);

    let make = |s_inner: usize, off: usize| {
        let s_outer = if s_inner == 0 { 1 } else { stride };
        Layout::new(Shape::new(&[rows, cols]), vec![s_outer, s_inner].into(), off)
    };
    for sc in [0, 1] {
        for sa in [0, 1] {
            for sb in [0, 1] {
                let (lc, la, lb) = (make(sc, 1), make(sa, 2), make(sb, 3));
                let got_f = where3_f32(&cf, &lc, &a, &la, &b, &lb);
                let ref_f = where3(&cf, &lc, &a, &la, &b, &lb, |v| v != 0.0);
                assert_eq!(
                    got_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "where3_f32 ({sc},{sa},{sb})"
                );
                let got_u = where3_u8(&cu, &lc, &a, &la, &b, &lb);
                let ref_u = where3(&cu, &lc, &a, &la, &b, &lb, |v| v != 0);
                assert_eq!(
                    got_u.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ref_u.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "where3_u8 ({sc},{sa},{sb})"
                );
            }
        }
    }

    // Contiguous lengths 0..=33 + parallel (> PAR_MIN) + transposed + empty.
    for len in 0..=33 {
        let l = Layout::contiguous(Shape::new(&[len]));
        let gf = where3_f32(&cf[..len], &l, &a[..len], &l, &b[..len], &l);
        let rf = where3(&cf[..len], &l, &a[..len], &l, &b[..len], &l, |v| v != 0.0);
        assert_eq!(
            gf.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            rf.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "where3_f32 contig len={len}"
        );
        let gu = where3_u8(&cu[..len], &l, &a[..len], &l, &b[..len], &l);
        let ru = where3(&cu[..len], &l, &a[..len], &l, &b[..len], &l, |v| v != 0);
        assert_eq!(
            gu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ru.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "where3_u8 contig len={len}"
        );
    }

    // Transposed / general non-unit inner stride and empty layout for both F32 and U8 masks.
    let l_trans = Layout::new(Shape::new(&[cols, rows]), vec![1, stride].into(), 2);
    let l_step = Layout::new(Shape::new(&[rows, 11]), vec![stride, 2].into(), 1);
    for l_gen in [&l_trans, &l_step] {
        let got_f = where3_f32(&cf, l_gen, &a, l_gen, &b, l_gen);
        let ref_f = where3(&cf, l_gen, &a, l_gen, &b, l_gen, |v| v != 0.0);
        assert_eq!(
            got_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "where3_f32 general stride {:?}",
            l_gen.strides()
        );
        let got_u = where3_u8(&cu, l_gen, &a, l_gen, &b, l_gen);
        let ref_u = where3(&cu, l_gen, &a, l_gen, &b, l_gen, |v| v != 0);
        assert_eq!(
            got_u.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_u.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "where3_u8 general stride {:?}",
            l_gen.strides()
        );
    }
    let l_empty = Layout::new(Shape::new(&[0, cols]), vec![cols, 1].into(), 5);
    assert!(where3_f32(&[], &l_empty, &[], &l_empty, &[], &l_empty).is_empty());
    assert!(where3_u8(&[], &l_empty, &[], &l_empty, &[], &l_empty).is_empty());

    let n_par = PAR_MIN + 27;
    let cf_p: Vec<f32> = (0..n_par).map(|i| cf[i % cf.len()]).collect();
    let cu_p: Vec<u8> = (0..n_par).map(|i| cu[i % cu.len()]).collect();
    let a_p: Vec<f32> = (0..n_par).map(|i| a[i % a.len()]).collect();
    let b_p: Vec<f32> = (0..n_par).map(|i| b[i % b.len()]).collect();
    let lp = Layout::contiguous(Shape::new(&[n_par]));
    let gf = where3_f32(&cf_p, &lp, &a_p, &lp, &b_p, &lp);
    let rf = where3(&cf_p, &lp, &a_p, &lp, &b_p, &lp, |v| v != 0.0);
    assert_eq!(
        gf.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        rf.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let gu = where3_u8(&cu_p, &lp, &a_p, &lp, &b_p, &lp);
    let ru = where3(&cu_p, &lp, &a_p, &lp, &b_p, &lp, |v| v != 0);
    assert_eq!(
        gu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ru.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
