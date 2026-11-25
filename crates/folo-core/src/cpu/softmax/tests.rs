use super::*;

fn ref_softmax(x: &[f32], (outer, d, inner): (usize, usize, usize), log: bool) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for o in 0..outer {
        for j in 0..inner {
            let mut m = f64::NEG_INFINITY;
            for r in 0..d {
                let v = x[(o * d + r) * inner + j] as f64;
                if !v.is_nan() {
                    m = m.max(v);
                }
            }
            let m = if m.is_finite() { m } else { 0.0 };
            let s: f64 = (0..d).map(|r| ((x[(o * d + r) * inner + j] as f64) - m).exp()).sum();
            for r in 0..d {
                let idx = (o * d + r) * inner + j;
                let v = x[idx] as f64;
                out[idx] = if log { ((v - m) - s.ln()) as f32 } else { ((v - m).exp() / s) as f32 };
            }
        }
    }
    out
}

#[test]
fn fast_exp8_is_bitwise_fast_exp() {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        let mut xs: Vec<f32> = (0..10_000).map(|i| -105.0 + i as f32 * 0.02).collect();
        xs.extend([
            f32::NEG_INFINITY,
            -104.0,
            -88.0,
            -0.0,
            0.0,
            88.72,
            89.0,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
        ]);
        for ch in xs.chunks(8) {
            let mut buf = [0f32; 8];
            buf[..ch.len()].copy_from_slice(ch);
            // SAFETY: AVX2 detected; `buf` holds 8 floats.
            let got = unsafe {
                use std::arch::x86_64::*;
                let mut o = [0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), fast_exp8(_mm256_loadu_ps(buf.as_ptr())));
                o
            };
            for (&x, &y) in buf.iter().zip(&got) {
                assert_eq!(y.to_bits(), fast_exp(x).to_bits(), "fast_exp8 mismatch at {x}");
            }
        }
    }
}

#[test]
fn softmax_short_online_and_column_tiled_match_f64_reference() {
    let shapes = [
        (3, 16, 1),   // DFL reg_max = 16
        (4, 37, 1),   // ragged inner == 1
        (2, 5003, 1), // > ROW_TILE (online tiled path, ragged tail)
        (1, 13, 600), // outer == 1, inner > COL_TILE (parallel column-tiled)
        (3, 19, 275), // outer > 1, multi-strip inner > 1 with ragged tail
    ];
    for (outer, d, inner) in shapes {
        let n = outer * d * inner;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.17).sin()) * 6.0 + 100.0).collect();
        let g: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.31).cos()) * 1.5).collect();
        for log in [false, true] {
            let got = softmax(&x, (outer, d, inner), log);
            let want = ref_softmax(&x, (outer, d, inner), log);
            for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                let scale = 1.0 + b.abs();
                assert!((a - b).abs() <= 2e-5 * scale, "{outer}x{d}x{inner} log={log} [{i}]: {a} vs {b}");
            }
            let bwd = softmax_backward(&got, &g, (outer, d, inner), log);
            // Compare against f64 backward reference.
            for o in 0..outer {
                for j in 0..inner {
                    let red: f64 = (0..d)
                        .map(|r| {
                            let idx = (o * d + r) * inner + j;
                            if log { g[idx] as f64 } else { g[idx] as f64 * got[idx] as f64 }
                        })
                        .sum();
                    for r in 0..d {
                        let idx = (o * d + r) * inner + j;
                        let want_g = if log {
                            g[idx] as f64 - (got[idx] as f64).exp() * red
                        } else {
                            got[idx] as f64 * (g[idx] as f64 - red)
                        };
                        assert!(
                            (bwd[idx] as f64 - want_g).abs() <= 3e-5 * (1.0 + want_g.abs()),
                            "bwd {outer}x{d}x{inner} log={log} [{idx}]: {} vs {want_g}",
                            bwd[idx]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn softmax_online_handles_nan_and_infinities() {
    // Long row (> ROW_TILE) with NaN hidden in an all-(-inf) tile, and all-(-inf), and +inf.
    let mut with_nan = vec![f32::NEG_INFINITY; ROW_TILE + 500];
    with_nan[0] = 1.0;
    with_nan[ROW_TILE + 100] = f32::NAN;
    for log in [false, true] {
        let y = softmax(&with_nan, (1, with_nan.len(), 1), log);
        assert!(y.iter().all(|v| v.is_nan()), "log={log}: NaN must propagate across all elements");
    }
    let all_neg_inf = vec![f32::NEG_INFINITY; ROW_TILE + 17];
    for log in [false, true] {
        let y = softmax(&all_neg_inf, (1, all_neg_inf.len(), 1), log);
        assert!(y.iter().all(|v| v.is_nan()), "log={log}: all -inf yields NaN");
    }
    let mut with_pos_inf = vec![1.0f32; ROW_TILE + 17];
    with_pos_inf[ROW_TILE + 5] = f32::INFINITY;
    let y = softmax(&with_pos_inf, (1, with_pos_inf.len(), 1), false);
    assert!(y[ROW_TILE + 5].is_nan());
    assert_eq!(y[0], 0.0);
}

#[test]
fn softmax_backward_preserves_reduce_block_precision_on_long_rows() {
    // 8193 = REDUCE_BLOCK + 1: `2^24` and `-2^24` cancel inside block 0 instead of swallowing the `1.0` in block 1.
    let n = crate::cpu::reduce::REDUCE_BLOCK + 1;
    let mut y_log = vec![-100.0f32; n];
    y_log[n - 1] = 0.0;
    let mut g = vec![0.0f32; n];
    g[0] = (1u32 << 24) as f32;
    g[1] = -((1u32 << 24) as f32);
    g[n - 1] = 1.0;
    let gx_log = softmax_backward(&y_log, &g, (1, n, 1), true);
    assert_eq!(gx_log[n - 1], 0.0, "blocked sum_g should be 1.0, giving g[last] - 1*1 = 0");

    let mut y_prob = vec![1.0f32; n];
    y_prob[0] = 1.0;
    y_prob[1] = 1.0;
    let gx_prob = softmax_backward(&y_prob, &g, (1, n, 1), false);
    assert_eq!(gx_prob[n - 1], 0.0, "blocked dot should be 1.0, giving 1*(1 - 1) = 0");
}
