use super::simd::silu_epilogue_scalar;
use super::*;
use crate::Shape;

fn assert_same_val(got: f32, want: f32, label: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{label}: expected NaN, got {got}");
    } else {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}: {got} ({:#x}) vs {want} ({:#x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

#[test]
fn fast_exp_accuracy() {
    let mut worst = 0f64;
    let mut x = -87.3f32;
    while x < 88.72 {
        let e = fast_exp(x) as f64;
        let t = (x as f64).exp();
        worst = worst.max(((e - t) / t).abs());
        x += 0.0137;
    }
    assert!(worst < 4e-7, "worst relative error {worst}");
    // Near the top of the range and in the subnormal range (absolute error there is limited by spacing).
    for x in [88.0f32, 88.5, 88.7, 88.72] {
        let (e, t) = (fast_exp(x) as f64, (x as f64).exp());
        assert!(((e - t) / t).abs() < 4e-7, "{x}: {e} vs {t}");
    }
    for x in [-88.0f32, -95.0, -100.0, -103.0] {
        let (e, t) = (fast_exp(x) as f64, (x as f64).exp());
        assert!((e - t).abs() <= 2.0 * f32::from_bits(1) as f64 + 1e-6 * t, "{x}: {e} vs {t}");
    }
    assert_eq!(fast_exp(88.8), f32::INFINITY);
    assert_eq!(fast_exp(f32::INFINITY), f32::INFINITY);
    assert_eq!(fast_exp(-110.0), 0.0);
    assert_eq!(fast_exp(f32::NEG_INFINITY), 0.0);
    assert!(fast_exp(f32::NAN).is_nan());
    assert_eq!(fast_exp(0.0), 1.0);
}

#[test]
fn silu_and_sigmoid_bias_do_not_depend_on_segment_boundaries() {
    let xs: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin() * 9.0).collect();
    for op in [UnaryOp::Silu, UnaryOp::Sigmoid, UnaryOp::Tanh] {
        let mut whole = xs.clone();
        bias_act_inplace(Some(op), 0.25, &mut whole);
        let mut pieces = xs.clone();
        let mut at = 0;
        for len in (1..20).cycle() {
            if at >= pieces.len() {
                break;
            }
            let end = (at + len).min(pieces.len());
            bias_act_inplace(Some(op), 0.25, &mut pieces[at..end]);
            at = end;
        }
        assert!(whole.iter().zip(&pieces).all(|(a, b)| a.to_bits() == b.to_bits()), "{op:?}");
        // Matches scalar twin bit-for-bit across all segments.
        for (&x, &y) in xs.iter().zip(&whole) {
            let sc = match op {
                UnaryOp::Silu => simd::silu_epilogue_scalar(x + 0.25),
                UnaryOp::Sigmoid => simd::sigmoid_scalar(x + 0.25),
                UnaryOp::Tanh => simd_log_tanh::tanh_scalar(x + 0.25),
                _ => unreachable!(),
            };
            assert_eq!(y.to_bits(), sc.to_bits(), "{op:?} at {x}");
        }
    }
}

#[test]
fn silu_avx2_error_bound() {
    let silu_ref = |x: f64| x / (1.0 + (-x).exp());
    let check = |xs: &[f32]| {
        let mut v = xs.to_vec();
        silu_bias_inplace(&mut v, 0.0);
        let mut worst = 0f64;
        for (&x, &y) in xs.iter().zip(&v) {
            let want = silu_ref(x as f64);
            let err = (y as f64 - want).abs();
            // Once exp(-|x|) is subnormal its spacing, times |x|, limits any f32 x * sigmoid(x).
            let slack = (x.abs() as f64 + 1.0) * 2.0 * f32::from_bits(1) as f64;
            assert!(err <= SILU_MAX_REL_ERR * want.abs() + slack, "silu({x}) = {y}, want {want}");
            if want.abs() > 1e-30 {
                worst = worst.max(err / want.abs());
            }
        }
        worst
    };
    // A dense sweep (every lane position sees every kind of value), the tails, and the subnormal range.
    let xs: Vec<f32> = (0..400_003).map(|i| -100.0 + i as f32 * 5e-4).collect();
    let worst = check(&xs);
    let tails: Vec<f32> =
        [-104.0f32, -103.5, -95.0, -88.0, -87.3, -20.0, -1e-30, 1e-30, 1e-7, 20.0, 88.7, 1e30, f32::MAX]
            .into_iter()
            .chain((0..64).map(|i| -0.5 + i as f32 / 64.0))
            .collect();
    check(&tails);
    println!("worst relative error of the vectorised SiLU: {worst:e}");
    // Special values match the scalar path.
    let mut sp = vec![f32::NAN, f32::INFINITY, -0.0, 0.0, -110.0, f32::MIN, 3.0, -3.0, 1.0];
    silu_bias_inplace(&mut sp, 0.0);
    assert!(sp[0].is_nan());
    assert_eq!(sp[1], f32::INFINITY);
    assert_eq!(sp[2], 0.0);
    assert_eq!(sp[3], 0.0);
    assert!(sp[4] <= 0.0 && sp[4] > -1e-40);
    // The bias is added before the activation.
    let mut b = vec![0.5f32; 19];
    silu_bias_inplace(&mut b, -1.5);
    for y in b {
        assert!(((y as f64) - silu_ref(-1.0)).abs() < 1e-6);
    }
    // Scalar twin matches AVX2 bit for bit on normal, subnormal, tail and special values.
    let mut scalar_check = tails;
    scalar_check.extend([0.0f32, -0.0, f32::INFINITY, -110.0, f32::MIN, 3.0, -3.0, 1.0, -1.0, 0.25]);
    let mut avx_out = scalar_check.clone();
    silu_bias_inplace(&mut avx_out, 0.0);
    for (&x, &y) in scalar_check.iter().zip(&avx_out) {
        let sc = silu_epilogue_scalar(x + 0.0);
        assert_eq!(sc.to_bits(), y.to_bits(), "silu_bias scalar vs AVX2 mismatch at {x}: {sc} vs {y}");
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        for ch in scalar_check.chunks(8) {
            let mut v = [0f32; 8];
            v[..ch.len()].copy_from_slice(ch);
            // SAFETY: AVX2 and FMA were detected.
            let got: [f32; 8] = unsafe {
                use std::arch::x86_64::*;
                let mut o = [0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), silu8(_mm256_loadu_ps(v.as_ptr())));
                o
            };
            for (&x, &y) in v.iter().zip(&got) {
                assert_eq!(
                    silu_epilogue_scalar(x).to_bits(),
                    y.to_bits(),
                    "silu8 direct scalar vs AVX2 mismatch at {x}"
                );
            }
        }
    }
    assert!(silu_epilogue_scalar(f32::NAN).is_nan());
}

#[test]
fn exp_sigmoid_silu_simd_accuracy_and_edges() {
    // 100_000 + 27 = 100_027 (mod 16 == 11), exercising 16-wide, 8-wide, and 3-lane padded tail paths.
    let mut xs: Vec<f32> = (0..100_000).map(|i| -105.0 + i as f32 * 0.002_05).collect();
    xs.extend([
        f32::NEG_INFINITY,
        -110.0,
        -104.0,
        -103.5,
        -95.0,
        -90.0,
        -88.0,
        -87.3,
        -20.0,
        -1.0,
        -1e-30,
        -0.0,
        0.0,
        1e-30,
        1e-7,
        1.0,
        20.0,
        88.0,
        88.72,
        88.8,
        90.0,
        110.0,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1234),
    ]);
    assert_eq!(xs.len() % 16, 11);
    let mut gs: Vec<f32> = (0..xs.len()).map(|i| ((i as f32 * 0.73).sin()) * 2.5).collect();
    // Place special upstream gradients (0, -0, ±inf, distinct NaN payload) on finite and edge inputs.
    let g_edges = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc0_5678)];
    for (k, &gv) in g_edges.iter().enumerate() {
        gs[k] = gv;
        gs[xs.len() - 1 - k] = gv;
    }
    let n = xs.len();

    let (mut y_exp, mut y_sig, mut y_silu) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let (mut b_exp, mut b_sig, mut b_silu, mut b_sp) =
        (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    simd::exp_slice(&xs, &mut y_exp);
    simd::sigmoid_slice(&xs, &mut y_sig);
    simd::silu_slice(&xs, &mut y_silu);
    simd::exp_bwd_slice(&xs, &gs, &mut b_exp);
    simd::sigmoid_bwd_slice(&xs, &gs, &mut b_sig);
    simd::silu_bwd_slice(&xs, &gs, &mut b_silu);
    simd::softplus_bwd_slice(&xs, &gs, &mut b_sp);

    let sub = f32::from_bits(1) as f64;
    for i in 0..n {
        let (x, g) = (xs[i], gs[i]);
        // Single-input exp/sigmoid preserve NaN payload; all kernels match scalar twin bitwise on non-NaNs.
        assert_eq!(y_exp[i].to_bits(), simd::exp_scalar(x).to_bits(), "exp bits at {x}");
        assert_eq!(y_sig[i].to_bits(), simd::sigmoid_scalar(x).to_bits(), "sigmoid bits at {x}");
        assert_same_val(y_silu[i], simd::silu_scalar(x), "silu");
        assert_same_val(y_silu[i], silu(x), "public silu");
        assert_same_val(b_exp[i], g * simd::exp_scalar(x), "exp_bwd");
        assert_same_val(b_sig[i], g * simd::sigmoid_deriv_scalar(x), "sig_bwd");
        assert_same_val(b_silu[i], simd::silu_bwd_scalar(x, g), "silu_bwd");
        assert_same_val(b_silu[i], silu_backward(x, g), "public silu_backward");
        assert_same_val(b_sp[i], g * simd::sigmoid_scalar(x), "softplus_bwd");

        if x.is_nan() {
            assert!(y_exp[i].is_nan() && y_sig[i].is_nan() && y_silu[i].is_nan());
            assert!(b_exp[i].is_nan() && b_sig[i].is_nan() && b_silu[i].is_nan() && b_sp[i].is_nan());
            continue;
        }
        let x64 = x as f64;
        let g64 = g as f64;
        let want_exp = x64.exp();
        if want_exp.is_infinite() || want_exp > f32::MAX as f64 {
            assert_eq!(y_exp[i], f32::INFINITY, "exp overflow at {x}");
        } else {
            let err = (y_exp[i] as f64 - want_exp).abs();
            assert!(err <= 4e-7 * want_exp + 2.0 * sub, "exp({x}) = {}, want {want_exp}", y_exp[i]);
        }
        let want_sig = 1.0 / (1.0 + (-x64).exp());
        let sig_err = (y_sig[i] as f64 - want_sig).abs();
        assert!(sig_err <= 5e-7 * want_sig + 2.0 * sub, "sigmoid({x}) = {}, want {want_sig}", y_sig[i]);

        if !g.is_finite() {
            continue;
        }
        let want_exp_bwd = g64 * want_exp;
        if want_exp_bwd.is_finite() && want_exp_bwd.abs() <= f32::MAX as f64 && want_exp <= f32::MAX as f64 {
            let eb_err = (b_exp[i] as f64 - want_exp_bwd).abs();
            assert!(
                eb_err <= 5e-7 * want_exp_bwd.abs() + (2.0 * g64.abs() + 1.0) * sub,
                "exp_bwd({x}, {g}) = {}, want {want_exp_bwd}",
                b_exp[i]
            );
        }
        let want_sp_bwd = g64 * want_sig;
        let sp_err = (b_sp[i] as f64 - want_sp_bwd).abs();
        assert!(
            sp_err <= 6e-7 * want_sp_bwd.abs() + (2.0 * g64.abs() + 1.0) * sub,
            "softplus_bwd({x}, {g}) = {}, want {want_sp_bwd}",
            b_sp[i]
        );
        let dsig = (-x64.abs()).exp() / (1.0 + (-x64.abs()).exp()).powi(2);
        let want_sig_bwd = g64 * dsig;
        let sb_err = (b_sig[i] as f64 - want_sig_bwd).abs();
        assert!(
            sb_err <= 8e-7 * want_sig_bwd.abs() + (2.0 * g64.abs() + 1.0) * sub,
            "sig_bwd({x}) = {}, want {want_sig_bwd}",
            b_sig[i]
        );
        if x.is_finite() {
            let want_silu_bwd = g64 * (want_sig + x64 * dsig);
            let scale = g64.abs() * (want_sig + x64.abs() * dsig);
            let sib_err = (b_silu[i] as f64 - want_silu_bwd).abs();
            let slack = ((x64.abs() + 1.0) * g64.abs() * 2.0 + 1.0) * sub;
            assert!(sib_err <= 8e-7 * scale + slack, "silu_bwd({x}) = {}, want {want_silu_bwd}", b_silu[i]);
        }
    }

    // Exact edge-value checks.
    assert_eq!(simd::exp_scalar(0.0), 1.0);
    assert_eq!(simd::exp_scalar(-0.0), 1.0);
    assert_eq!(simd::exp_scalar(90.0), f32::INFINITY);
    assert_eq!(simd::exp_scalar(f32::INFINITY), f32::INFINITY);
    assert_eq!(simd::exp_scalar(f32::NEG_INFINITY), 0.0);
    assert!(simd::exp_scalar(-90.0) > 0.0 && simd::exp_scalar(-90.0) < 1e-38);
    assert_eq!(simd::sigmoid_scalar(0.0), 0.5);
    assert_eq!(simd::sigmoid_scalar(-0.0), 0.5);
    assert_eq!(simd::sigmoid_scalar(90.0), 1.0);
    assert_eq!(simd::sigmoid_scalar(f32::INFINITY), 1.0);
    assert_eq!(simd::sigmoid_scalar(f32::NEG_INFINITY), 0.0);
    assert_eq!(2.0 * simd::sigmoid_deriv_scalar(0.0), 0.5);
    assert_eq!(2.0 * simd::sigmoid_deriv_scalar(f32::INFINITY), 0.0);
    assert_eq!(2.0 * simd::sigmoid_deriv_scalar(f32::NEG_INFINITY), 0.0);
    assert_eq!(simd::silu_bwd_scalar(0.0, 2.0), 1.0);
    assert_eq!(simd::silu_bwd_scalar(90.0, 2.0), 2.0);
    assert!(simd::silu_bwd_scalar(f32::INFINITY, 2.0).is_nan());
    assert!(simd::silu_bwd_scalar(f32::NEG_INFINITY, 2.0).is_nan());

    // Broadcast-g backward matches dense-g backward across all four vectorized backwards.
    let g_const = vec![1.25f32; n];
    let (mut bc, mut dense) = (vec![0f32; n], vec![0f32; n]);
    type BcFn = fn(&[f32], f32, &mut [f32]);
    type DenseFn = fn(&[f32], &[f32], &mut [f32]);
    for (name, bc_fn, dense_fn) in [
        ("exp_bwd", simd::exp_bwd_broadcast_g as BcFn, simd::exp_bwd_slice as DenseFn),
        ("sig_bwd", simd::sigmoid_bwd_broadcast_g as BcFn, simd::sigmoid_bwd_slice as DenseFn),
        ("silu_bwd", simd::silu_bwd_broadcast_g as BcFn, simd::silu_bwd_slice as DenseFn),
        ("softplus_bwd", simd::softplus_bwd_broadcast_g as BcFn, simd::softplus_bwd_slice as DenseFn),
    ] {
        bc_fn(&xs, 1.25, &mut bc);
        dense_fn(&xs, &g_const, &mut dense);
        for i in 0..n {
            assert_same_val(bc[i], dense[i], name);
        }
    }
}

#[test]
fn vectorized_ops_match_across_strided_and_broadcast_layouts() {
    // Row length 29 = 16 + 8 + 5 hits all three AVX2 widths, with signed zeros and subnormals in the tail.
    let (rows, cols) = (3usize, 29usize);
    let mut x_data: Vec<f32> = (0..rows * (cols + 3)).map(|i| -4.0 + (i as f32) * 0.09).collect();
    let g_data: Vec<f32> = (0..rows * (cols + 3)).map(|i| ((i as f32) * 0.41).cos() * 1.7).collect();
    for r in 0..rows {
        let base = r * (cols + 3);
        x_data[base + 24] = -0.0;
        x_data[base + 25] = 0.0;
        x_data[base + 26] = -90.0;
        x_data[base + 27] = -103.0;
        x_data[base + 28] = 88.5;
    }
    // Non-contiguous unit-stride rows (stride = [cols + 3, 1]).
    let l_unit = Layout::new(Shape::new(&[rows, cols]), vec![cols + 3, 1].into(), 0);
    // Transposed (non-unit inner stride: [1, cols + 3], shape [cols, rows]).
    let l_trans = Layout::new(Shape::new(&[cols, rows]), vec![1, cols + 3].into(), 0);
    let x_contig = copy_strided(&x_data, &l_unit);
    let g_contig = copy_strided(&g_data, &l_unit);
    let l_c = Layout::contiguous(Shape::new(&[rows, cols]));

    for op in [UnaryOp::Exp, UnaryOp::Log, UnaryOp::Tanh, UnaryOp::Sigmoid, UnaryOp::Silu, UnaryOp::Softplus]
    {
        let ref_y = unary(op, &x_contig, &l_c);
        // Unit-stride non-contiguous rows.
        let unit_y = unary(op, &x_data, &l_unit);
        assert_eq!(
            unit_y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "unary {op:?} unit-stride rows"
        );
        // Non-unit inner stride.
        let trans_y = unary(op, &x_data, &l_trans);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(
                    trans_y[c * rows + r].to_bits(),
                    ref_y[r * cols + c].to_bits(),
                    "unary {op:?} non-unit stride ({r}, {c})"
                );
            }
        }
        // Empty tensor.
        let l_empty = Layout::contiguous(Shape::new(&[0, cols]));
        assert!(unary(op, &[], &l_empty).is_empty());
    }

    // Broadcast layouts for backward: (1, 1) strided, (1, 0) broadcast g, (0, 1) broadcast x, and non-unit.
    let g_row_bc = Layout::new(Shape::new(&[rows, cols]), vec![1, 0].into(), 0);
    let g_row_contig = copy_strided(&g_data, &g_row_bc);
    let x_row_bc = Layout::new(Shape::new(&[rows, cols]), vec![1, 0].into(), 0);
    let x_row_contig = copy_strided(&x_data, &x_row_bc);
    let l_empty = Layout::contiguous(Shape::new(&[0, cols]));

    for op in [UnaryOp::Exp, UnaryOp::Log, UnaryOp::Tanh, UnaryOp::Sigmoid, UnaryOp::Silu, UnaryOp::Softplus]
    {
        // (1, 1) strided rows vs contiguous.
        let ref_b = unary_backward(op, &x_contig, &l_c, &g_contig, &l_c);
        let unit_b = unary_backward(op, &x_data, &l_unit, &g_data, &l_unit);
        for i in 0..ref_b.len() {
            assert_same_val(unit_b[i], ref_b[i], "bwd (1,1)");
        }
        // (1, 0) broadcast g vs materialized g.
        let ref_bc_g = unary_backward(op, &x_contig, &l_c, &g_row_contig, &l_c);
        let got_bc_g = unary_backward(op, &x_data, &l_unit, &g_data, &g_row_bc);
        for i in 0..ref_bc_g.len() {
            assert_same_val(got_bc_g[i], ref_bc_g[i], "bwd (1,0)");
        }
        // (0, 1) broadcast x vs materialized x.
        let ref_bc_x = unary_backward(op, &x_row_contig, &l_c, &g_contig, &l_c);
        let got_bc_x = unary_backward(op, &x_data, &x_row_bc, &g_data, &l_unit);
        for i in 0..ref_bc_x.len() {
            assert_same_val(got_bc_x[i], ref_bc_x[i], "bwd (0,1)");
        }
        // Non-unit inner stride on both operands.
        let trans_b = unary_backward(op, &x_data, &l_trans, &g_data, &l_trans);
        for r in 0..rows {
            for c in 0..cols {
                assert_same_val(trans_b[c * rows + r], ref_b[r * cols + c], "bwd non-unit");
            }
        }
        // Empty backward.
        assert!(unary_backward(op, &[], &l_empty, &[], &l_empty).is_empty());
    }

    // Large (>= PAR_MIN) single-row (1, 0) broadcast backward (hits parallel chunk split in fill_rows).
    let n_big = PAR_MIN + 29;
    let x_big: Vec<f32> = (0..n_big).map(|i| -3.0 + (i % 97) as f32 * 0.06).collect();
    let g_scalar = [0.75f32];
    let g_dense = vec![0.75f32; n_big];
    let l_big = Layout::contiguous(Shape::new(&[n_big]));
    let l_bc = Layout::new(Shape::new(&[n_big]), vec![0].into(), 0);
    for op in [UnaryOp::Exp, UnaryOp::Log, UnaryOp::Tanh, UnaryOp::Sigmoid, UnaryOp::Silu, UnaryOp::Softplus]
    {
        let got = unary_backward(op, &x_big, &l_big, &g_scalar, &l_bc);
        let want = unary_backward(op, &x_big, &l_big, &g_dense, &l_big);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "parallel (1,0) bwd {op:?}"
        );
    }
}

#[test]
fn erf_accuracy() {
    // Reference: the Maclaurin series in f64 (converges everywhere; 200 terms suffice for |x| <= 4.5).
    let reference = |x: f64| {
        let (mut term, mut sum) = (x, x);
        for n in 1..200 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    };
    let mut worst = 0f64;
    let mut x = -4.5f32;
    while x <= 4.5 {
        let (got, want) = (erf(x) as f64, reference(x as f64));
        let a = (want.abs() as f32).max(f32::MIN_POSITIVE);
        let ulp = (f32::from_bits(a.to_bits() + 1) - a) as f64;
        worst = worst.max((got - want).abs() / ulp);
        x += 0.00731;
    }
    assert!(worst <= 2.0, "worst error {worst} ulp");
    for x in [1e-30f32, 1e-8, -3e-5] {
        let want = x as f64 * std::f64::consts::FRAC_2_SQRT_PI;
        assert!(((erf(x) as f64) - want).abs() <= 1e-7 * want.abs(), "{x}");
    }
    assert_eq!(erf(0.0), 0.0);
    assert!(erf(-0.0).is_sign_negative());
    assert_eq!(erf(10.0), 1.0);
    assert_eq!(erf(f32::NEG_INFINITY), -1.0);
    assert!(erf(f32::NAN).is_nan());
}

#[test]
fn log_tanh_softplus_simd_accuracy_and_edges() {
    // 1. Log: log-spaced sweep over the full positive normal & subnormal range plus special values.
    let mut ln_xs: Vec<f32> = (0..80_000)
        .map(|i| {
            let e = -126.0 + (i as f64) * (253.0 / 80_000.0);
            (2.0f64.powf(e)) as f32
        })
        .collect();
    // Dense sweep around 1.0 and sqrt(2)/2..sqrt(2).
    ln_xs.extend((0..20_000).map(|i| 0.5 + i as f32 * 7.5e-5));
    ln_xs.extend([
        f32::from_bits(1),
        f32::from_bits(0x0040_0000),
        f32::from_bits(0x007F_FFFF),
        f32::MIN_POSITIVE,
        std::f32::consts::FRAC_1_SQRT_2,
        1.0,
        std::f32::consts::SQRT_2,
        2.0,
        f32::MAX,
        0.0,
        -0.0,
        -1.0,
        -1e-30,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1234),
    ]);
    // Ensure padded tail (mod 16 != 0).
    assert_ne!(ln_xs.len() % 16, 0);
    let ln_gs: Vec<f32> = (0..ln_xs.len()).map(|i| ((i as f32 * 0.53).cos()) * 3.0).collect();
    let (mut y_ln, mut b_ln, mut b_ln_bc) =
        (vec![0f32; ln_xs.len()], vec![0f32; ln_xs.len()], vec![0f32; ln_xs.len()]);
    simd_log_tanh::ln_slice(&ln_xs, &mut y_ln);
    simd_log_tanh::log_bwd_slice(&ln_xs, &ln_gs, &mut b_ln);
    simd_log_tanh::log_bwd_broadcast_g(&ln_xs, 1.5, &mut b_ln_bc);

    for i in 0..ln_xs.len() {
        let (x, g) = (ln_xs[i], ln_gs[i]);
        assert_eq!(y_ln[i].to_bits(), simd_log_tanh::ln_scalar(x).to_bits(), "ln bits at {x}");
        assert_same_val(b_ln[i], g / x, "log_bwd");
        assert_same_val(b_ln_bc[i], 1.5 / x, "log_bwd_bc");
        if x.is_nan() {
            assert_eq!(y_ln[i].to_bits(), x.to_bits(), "ln NaN payload");
        } else if x < 0.0 {
            assert!(y_ln[i].is_nan(), "ln({x}) should be NaN");
        } else if x == 0.0 {
            assert_eq!(y_ln[i], f32::NEG_INFINITY);
        } else if x.is_infinite() {
            assert_eq!(y_ln[i], f32::INFINITY);
        } else {
            let want = (x as f64).ln();
            let err = (y_ln[i] as f64 - want).abs();
            assert!(err <= 4e-7 * want.abs() + 2e-7, "ln({x}) = {}, want {want} (err {err})", y_ln[i]);
        }
    }
    assert_eq!(simd_log_tanh::ln_scalar(1.0), 0.0);

    // 2. Tanh, tanh_bwd, and softplus across full range + cutoff boundary + edges.
    let mut xs: Vec<f32> = (0..100_000).map(|i| -105.0 + i as f32 * 0.002_1).collect();
    // Dense sweep around 0 and the +-0.625 polynomial/exp cutoff.
    xs.extend((0..1000).map(|i| -1.0 + i as f32 * 0.002));
    xs.extend([
        f32::NEG_INFINITY,
        -110.0,
        -104.0,
        -90.0,
        -52.0,
        -40.0,
        -20.0,
        -10.0,
        -0.625,
        -1e-7,
        -1e-30,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        1e-30,
        1e-7,
        0.625,
        10.0,
        20.0,
        40.0,
        52.0,
        90.0,
        110.0,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_4321),
    ]);
    assert_ne!(xs.len() % 16, 0);
    let mut gs: Vec<f32> = (0..xs.len()).map(|i| ((i as f32 * 0.71).sin()) * 2.5).collect();
    for (k, &gv) in
        [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc0_5678)].iter().enumerate()
    {
        gs[k] = gv;
        gs[xs.len() - 1 - k] = gv;
    }

    let n = xs.len();
    let (mut y_th, mut b_th, mut b_th_bc, mut y_sp) =
        (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    simd_log_tanh::tanh_slice(&xs, &mut y_th);
    simd_log_tanh::tanh_bwd_slice(&xs, &gs, &mut b_th);
    simd_log_tanh::tanh_bwd_broadcast_g(&xs, 1.25, &mut b_th_bc);
    simd_log_tanh::softplus_slice(&xs, &mut y_sp);

    let sub = f32::from_bits(1) as f64;
    for i in 0..n {
        let (x, g) = (xs[i], gs[i]);
        assert_eq!(y_th[i].to_bits(), simd_log_tanh::tanh_scalar(x).to_bits(), "tanh bits at {x}");
        assert_eq!(y_sp[i].to_bits(), simd_log_tanh::softplus_scalar(x).to_bits(), "softplus bits at {x}");
        assert_same_val(b_th[i], g * simd_log_tanh::tanh_deriv_scalar(x), "tanh_bwd");
        assert_same_val(b_th_bc[i], 1.25 * simd_log_tanh::tanh_deriv_scalar(x), "tanh_bwd_bc");
        if x.is_nan() {
            assert_eq!(y_th[i].to_bits(), x.to_bits(), "tanh NaN payload");
            assert_eq!(y_sp[i].to_bits(), x.to_bits(), "softplus NaN payload");
            assert!(b_th[i].is_nan());
            continue;
        }
        let x64 = x as f64;
        let want_th = x64.tanh();
        let th_err = (y_th[i] as f64 - want_th).abs();
        assert!(th_err <= 5e-7 * want_th.abs() + 2.0 * sub, "tanh({x}) = {}, want {want_th}", y_th[i]);

        let want_sp = x64.max(0.0) + (-x64.abs()).exp().ln_1p();
        if want_sp.is_infinite() {
            assert_eq!(y_sp[i], f32::INFINITY);
        } else {
            let sp_err = (y_sp[i] as f64 - want_sp).abs();
            assert!(sp_err <= 6e-7 * want_sp + 2.0 * sub, "softplus({x}) = {}, want {want_sp}", y_sp[i]);
        }

        if g.is_finite() {
            let e2 = (-2.0 * x64.abs()).exp();
            let dth = 4.0 * e2 / ((1.0 + e2) * (1.0 + e2));
            let want_tb = g as f64 * dth;
            let tb_err = (b_th[i] as f64 - want_tb).abs();
            let slack = (4.0 * (g as f64).abs() + 1.0) * sub;
            assert!(
                tb_err <= 8e-7 * want_tb.abs() + slack,
                "tanh_bwd({x}, {g}) = {}, want {want_tb}",
                b_th[i]
            );
        }
    }

    // Exact signed zero and saturation checks.
    assert_eq!(simd_log_tanh::tanh_scalar(0.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(simd_log_tanh::tanh_scalar(-0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(simd_log_tanh::tanh_scalar(f32::INFINITY), 1.0);
    assert_eq!(simd_log_tanh::tanh_scalar(f32::NEG_INFINITY), -1.0);
    assert_eq!(simd_log_tanh::tanh_deriv_scalar(0.0), 1.0);
    assert!(simd_log_tanh::tanh_deriv_scalar(1e-6) <= 1.0);
    assert!(simd_log_tanh::tanh_deriv_scalar(1e-6) * f32::MAX <= f32::MAX);
    assert_eq!(simd_log_tanh::tanh_deriv_scalar(f32::INFINITY), 0.0);
    assert_eq!(simd_log_tanh::tanh_deriv_scalar(f32::NEG_INFINITY), 0.0);
    assert_eq!(simd_log_tanh::softplus_scalar(f32::INFINITY), f32::INFINITY);
    assert_eq!(simd_log_tanh::softplus_scalar(f32::NEG_INFINITY), 0.0);
}

#[test]
fn strided_update_and_copy_into_fast_paths() {
    let ref_update = |dst: &[f32], dl: &Layout, src: &[f32], sl: &Layout, f: fn(f32, f32) -> f32| {
        let mut out = dst.to_vec();
        let p = RowPlan::<2>::new(dl.dims(), [dl.strides(), sl.strides()], [dl.offset(), sl.offset()]);
        p.for_rows(0..p.rows, |_, [od, os]| {
            for i in 0..p.len {
                let di = od + i * p.inner[0];
                let si = os + i * p.inner[1];
                out[di] = f(out[di], src[si]);
            }
        });
        out
    };

    // Contiguous serial & parallel (>= PAR_MIN), (1, 1) strided rows, (1, 0) broadcast, (_, 0) strided dst +
    // broadcast src, and general (ds, ss) strided layouts.
    let cases = [
        // (shape, dst_strides, dst_off, src_strides, src_off, dst_buf_len, src_buf_len)
        (vec![5, 19], vec![19, 1], 3usize, vec![19, 1], 7usize, 120usize, 120usize),
        (vec![2, 18_000], vec![18_000, 1], 0, vec![18_000, 1], 0, 36_000, 36_000),
        (vec![6, 25], vec![29, 1], 2, vec![31, 1], 5, 200, 210),
        (vec![6, 25], vec![29, 1], 1, vec![1, 0], 4, 200, 20),
        (vec![1, 40_000], vec![1, 1], 0, vec![0, 0], 3, 40_000, 10),
        (vec![7, 13], vec![1, 9], 2, vec![1, 0], 1, 150, 20),
        (vec![7, 13], vec![2, 16], 1, vec![3, 25], 2, 260, 400),
        // Empty layout with non-zero offset against an empty buffer.
        (vec![0, 8], vec![8, 1], 5, vec![8, 1], 9, 0, 0),
    ];

    for (dims, ds, doff, ss, soff, dlen, slen) in cases {
        let dl = Layout::new(Shape::new(&dims), ds.into(), doff);
        let sl = Layout::new(Shape::new(&dims), ss.into(), soff);
        let init_dst: Vec<f32> = (0..dlen).map(|i| (i as f32 * 0.13) - 2.0).collect();
        let src: Vec<f32> = (0..slen).map(|i| (i as f32 * 0.29) + 1.5).collect();

        let mut got_copy = init_dst.clone();
        copy_into(&mut got_copy, &dl, &src, &sl);
        let want_copy = ref_update(&init_dst, &dl, &src, &sl, |_, s| s);
        assert_eq!(got_copy, want_copy, "copy_into mismatch for shape {dims:?}");

        let mut got_add = init_dst.clone();
        fill_rows_inplace(&mut got_add, &dl, &src, &sl, |d, s| d + s);
        let want_add = ref_update(&init_dst, &dl, &src, &sl, |d, s| d + s);
        assert_eq!(got_add, want_add, "strided_update mismatch for shape {dims:?}");
    }
}

#[test]
fn where3_contiguous_broadcast_and_strided_paths() {
    let ref_where = |c: &[f32], lc: &Layout, a: &[f32], la: &Layout, b: &[f32], lb: &Layout| -> Vec<f32> {
        let mut out = vec![0.0f32; la.numel()];
        let p = RowPlan::<3>::new(
            la.dims(),
            [lc.strides(), la.strides(), lb.strides()],
            [lc.offset(), la.offset(), lb.offset()],
        );
        let mut idx = 0;
        p.for_rows(0..p.rows, |_, [oc, oa, ob]| {
            for i in 0..p.len {
                out[idx] = if c[oc + i * p.inner[0]] != 0.0 {
                    a[oa + i * p.inner[1]]
                } else {
                    b[ob + i * p.inner[2]]
                };
                idx += 1;
            }
        });
        out
    };

    // Test all 8 combinations of (sc, sa, sb) in {0, 1}^3 plus general non-unit strides.
    let (rows, cols) = (5usize, 29usize);
    let stride_row = cols + 3;
    let mut c_buf: Vec<f32> =
        (0..rows * stride_row + 10).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
    c_buf[1] = -0.0;
    c_buf[2] = f32::NAN;
    let mut a_buf: Vec<f32> = (0..rows * stride_row + 10).map(|i| (i as f32) + 0.25).collect();
    a_buf[2] = f32::from_bits(0x7fc0_1111);
    let mut b_buf: Vec<f32> = (0..rows * stride_row + 10).map(|i| -(i as f32) - 0.5).collect();
    b_buf[1] = -0.0;

    let make = |s_inner: usize, off: usize| {
        let s_outer = if s_inner == 0 { 1 } else { stride_row };
        Layout::new(Shape::new(&[rows, cols]), vec![s_outer, s_inner].into(), off)
    };
    for sc in [0, 1] {
        for sa in [0, 1] {
            for sb in [0, 1] {
                let (lc, la, lb) = (make(sc, 1), make(sa, 2), make(sb, 3));
                let got = where3(&c_buf, &lc, &a_buf, &la, &b_buf, &lb, |v| v != 0.0);
                let want = ref_where(&c_buf, &lc, &a_buf, &la, &b_buf, &lb);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "where3 mismatch for strides ({sc}, {sa}, {sb})"
                );
            }
        }
    }

    // General non-unit stride + empty with offset + parallel contiguous (>= PAR_MIN).
    let l_trans = Layout::new(Shape::new(&[cols, rows]), vec![1, stride_row].into(), 0);
    let got_t = where3(&c_buf, &l_trans, &a_buf, &l_trans, &b_buf, &l_trans, |v| v != 0.0);
    let want_t = ref_where(&c_buf, &l_trans, &a_buf, &l_trans, &b_buf, &l_trans);
    assert_eq!(
        got_t.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want_t.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let l_empty = Layout::new(Shape::new(&[0, 8]), vec![8, 1].into(), 4);
    assert!(where3::<u8>(&[], &l_empty, &[], &l_empty, &[], &l_empty, |v| v != 0).is_empty());

    let n_par = PAR_MIN + 777;
    let c_par: Vec<u8> = (0..n_par).map(|i| (i % 5 == 0) as u8).collect();
    let a_par: Vec<f32> = (0..n_par).map(|i| i as f32).collect();
    let b_par: Vec<f32> = (0..n_par).map(|i| -(i as f32)).collect();
    let l_par = Layout::contiguous(Shape::new(&[n_par]));
    let got_par = where3(&c_par, &l_par, &a_par, &l_par, &b_par, &l_par, |v| v != 0);
    for i in 0..n_par {
        let want = if c_par[i] != 0 { a_par[i] } else { b_par[i] };
        assert_eq!(got_par[i].to_bits(), want.to_bits());
    }
}

#[test]
fn transpose_2d_and_copy_strided_f32_bitwise_exact() {
    // 2D direct transpose across small, exact multiples of 8/32, ragged, padded strides, and parallel sizes.
    for &(rows, cols, src_pad, dst_pad) in &[
        (1, 1, 0, 0),
        (7, 9, 0, 0),
        (8, 8, 0, 0),
        (13, 29, 3, 5),
        (32, 32, 0, 0),
        (37, 41, 7, 2),
        (73, 19, 1, 4),
        (257, 193, 5, 3),
    ] {
        let src_rs = cols + src_pad;
        let dst_rs = rows + dst_pad;
        let mut src: Vec<f32> = (0..rows * src_rs).map(|i| (i as f32) * 0.5 - 10.0).collect();
        if src.len() > 3 {
            src[0] = -0.0;
            src[1] = f32::from_bits(0x7fc0_1234);
            src[2] = f32::NEG_INFINITY;
        }
        let mut dst = vec![-999.0f32; cols * dst_rs];
        transpose_f32_2d(&src, src_rs, &mut dst, dst_rs, rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                assert_eq!(
                    dst[j * dst_rs + i].to_bits(),
                    src[i * src_rs + j].to_bits(),
                    "transpose ({rows}x{cols}) mismatch at ({j}, {i})"
                );
            }
        }
    }

    // Batched transpose and NCHW <-> NHWC permutes via copy_strided_f32 (serial and parallel).
    let layouts: Vec<(Vec<usize>, Vec<usize>, usize, usize)> = vec![
        // [B, N, M] transpose of [B, M, N]
        (vec![5, 37, 29], vec![37 * 29, 1, 37], 0, 5 * 37 * 29),
        // Large single-batch transpose (>= PAR_MIN)
        (vec![193, 211], vec![1, 193], 7, 193 * 211 + 10),
        // Large 2-batch transpose (hits batches < 4 parallel path)
        (vec![2, 181, 191], vec![181 * 191, 1, 181], 0, 2 * 181 * 191),
        // Large 6-batch transpose (hits batches >= 4 parallel path)
        (vec![6, 73, 83], vec![73 * 83, 1, 73], 0, 6 * 73 * 83),
        // NCHW -> NHWC ([2, 17, 13, 19] from [2, 19, 17, 13])
        (vec![2, 17, 13, 19], vec![19 * 17 * 13, 13, 1, 17 * 13], 0, 2 * 19 * 17 * 13),
        // NHWC -> NCHW ([2, 19, 17, 13] from [2, 17, 13, 19])
        (vec![2, 19, 17, 13], vec![17 * 13 * 19, 1, 13 * 19, 19], 0, 2 * 17 * 13 * 19),
        // Overlapping read strides (src_rs < cols)
        (vec![15, 11], vec![1, 3], 2, 15 + 10 * 3 + 4),
        // Broadcast inner stride s == 0
        (vec![4, 17], vec![1, 0], 1, 10),
        // Empty layout with non-zero offset
        (vec![0, 12], vec![12, 1], 5, 0),
    ];

    for (dims, strides, off, buf_len) in layouts {
        let l = Layout::new(Shape::new(&dims), strides.into(), off);
        let mut src: Vec<f32> = (0..buf_len).map(|i| (i as f32) * 0.25 - 3.0).collect();
        if buf_len > 2 {
            src[0] = -0.0;
            src[1] = f32::from_bits(0x7fc0_beef);
        }
        let got = copy_strided_f32(&src, &l);
        let want = copy_strided(&src, &l);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "copy_strided_f32 mismatch for {dims:?}"
        );
    }
}

#[test]
fn binary_ops_simd_matches_scalar_with_edges() {
    let edges: [f32; 14] = [
        -0.0,
        0.0,
        -1.5,
        1.5,
        -1e-30,
        1e-30,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1234),
        42.25,
    ];
    let ops = [
        (BinaryOp::Add, (|x: f32, y: f32| x + y) as fn(f32, f32) -> f32),
        (BinaryOp::Sub, |x, y| x - y),
        (BinaryOp::Mul, |x, y| x * y),
        (BinaryOp::Div, |x, y| x / y),
        (BinaryOp::Maximum, f32::max),
        (BinaryOp::Minimum, f32::min),
    ];

    // Contiguous pairwise Cartesian product of all edge values plus ragged lengths (0, 1, 3, 4, 7, 8, 11, 19).
    let mut a_all = Vec::new();
    let mut b_all = Vec::new();
    for &x in &edges {
        for &y in &edges {
            a_all.push(x);
            b_all.push(y);
        }
    }
    for len in [0usize, 1, 3, 4, 7, 8, 11, 19, a_all.len()] {
        let la = Layout::contiguous(Shape::new(&[len]));
        let a = &a_all[..len];
        let b = &b_all[..len];
        for &(op, scalar) in &ops {
            let got = binary(op, a, &la, b, &la);
            assert_eq!(got.len(), len);
            for (i, &g) in got.iter().enumerate() {
                assert_same_val(g, scalar(a[i], b[i]), &format!("{op:?} contig len={len} i={i}"));
            }
        }
    }

    // Strided (1, 1), broadcast-b (1, 0), broadcast-a (0, 1), and general (2, 3) strides.
    let (rows, cols) = (4usize, 19usize);
    let stride_row = cols + 3;
    let mut a_buf: Vec<f32> = (0..rows * stride_row + 10).map(|i| edges[i % edges.len()]).collect();
    let mut b_buf: Vec<f32> = (0..rows * stride_row + 10).map(|i| edges[(i * 5 + 1) % edges.len()]).collect();
    a_buf[0] = -0.0;
    b_buf[0] = 0.0;
    let make = |s_inner: usize, off: usize| {
        let s_outer = if s_inner == 0 { 1 } else { stride_row };
        Layout::new(Shape::new(&[rows, cols]), vec![s_outer, s_inner].into(), off)
    };
    for (sa, sb) in [(1usize, 1usize), (1, 0), (0, 1)] {
        let (la, lb) = (make(sa, 1), make(sb, 2));
        for &(op, scalar) in &ops {
            let got = binary(op, &a_buf, &la, &b_buf, &lb);
            let want = map2(&a_buf, &la, &b_buf, &lb, scalar);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_same_val(g, w, &format!("{op:?} strides=({sa},{sb}) i={i}"));
            }
        }
    }
}
