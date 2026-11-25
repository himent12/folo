use super::*;

#[test]
fn minmax_avx2_matches_scalar_on_nans_signed_zeros_and_tails() {
    let mut xs: Vec<f32> = (0..1037).map(|i| ((i as f32) * 0.37).sin() * 100.0).collect();
    // Exercise clean data (no NaN), signed zeros, infinities, and NaN in early/mid/tail lanes.
    for op in [ReduceOp::Max, ReduceOp::Min] {
        assert_eq!(reduce_block(op, &xs).to_bits(), minmax_slice_scalar(op, &xs).to_bits());
        // Signed zeros only: exact tie behavior.
        let zeros: Vec<f32> = (0..37).map(|i| if i % 2 == 0 { -0.0 } else { 0.0 }).collect();
        assert_eq!(reduce_block(op, &zeros).to_bits(), minmax_slice_scalar(op, &zeros).to_bits());
        let zeros_pos: Vec<f32> = (0..37).map(|i| if i % 2 == 0 { 0.0 } else { -0.0 }).collect();
        assert_eq!(reduce_block(op, &zeros_pos).to_bits(), minmax_slice_scalar(op, &zeros_pos).to_bits());
    }
    // NaN in first lane, middle lane, and tail lane.
    for pos in [0, 7, 8, 15, 512, xs.len() - 1] {
        let mut v = xs.clone();
        v[pos] = f32::from_bits(0x7fc0_0000 + pos as u32);
        for op in [ReduceOp::Max, ReduceOp::Min] {
            let got = reduce_block(op, &v);
            let want = minmax_slice_scalar(op, &v);
            assert!(got.is_nan() && want.is_nan());
            assert_eq!(got.to_bits(), want.to_bits(), "{op:?} NaN at {pos}");
        }
    }
    // Column reduction (inner > 1) with NaNs and signed zeros across > 8 columns.
    let (outer, d, inner) = (2, 7, 29);
    let mut grid: Vec<f32> = (0..outer * d * inner).map(|i| ((i as f32) * 0.23).cos() * 5.0).collect();
    grid[inner + 3] = f32::from_bits(0x7fc0_1234);
    grid[3 * inner + 17] = -0.0;
    grid[4 * inner + 17] = 0.0;
    for op in [ReduceOp::Max, ReduceOp::Min] {
        let got = reduce_f32(op, &grid, (outer, d, inner));
        for o in 0..outer {
            for c in 0..inner {
                let mut want = identity(op);
                for r in 0..d {
                    want = combine(op, want, grid[(o * d + r) * inner + c]);
                }
                assert_eq!(got[o * inner + c].to_bits(), want.to_bits(), "col {op:?} at ({o}, {c})");
            }
        }
    }
    // Keep xs used cleanly.
    xs[0] = f32::INFINITY;
    assert_eq!(reduce_block(ReduceOp::Max, &xs), f32::INFINITY);
}

#[test]
fn topk_bounded_k_matches_full_sort_with_nans_ties_and_signed_zeros() {
    let ref_topk = |x: &[f32], rows: usize, d: usize, k: usize, largest: bool| {
        let key = |v: f32| if v == 0.0 { 0.0 } else { v };
        let cmp = |a: &(f32, usize), b: &(f32, usize)| {
            let o = if largest { key(b.0).total_cmp(&key(a.0)) } else { key(a.0).total_cmp(&key(b.0)) };
            o.then(a.1.cmp(&b.1))
        };
        let mut vals = Vec::with_capacity(rows * k);
        let mut idxs = Vec::with_capacity(rows * k);
        for r in 0..rows {
            let mut items: Vec<(f32, usize)> = x[r * d..(r + 1) * d].iter().copied().zip(0..).collect();
            items.sort_unstable_by(cmp);
            for &(v, i) in items.iter().take(k) {
                vals.push(v);
                idxs.push(i as i64);
            }
        }
        (vals, idxs)
    };

    // Construct rows containing ties, -0.0/+0.0, ±inf, +NaN, -NaN, and distinct NaN payloads.
    let (rows, d) = (6, 257);
    let mut x: Vec<f32> = (0..rows * d).map(|i| (((i * 37) % 19) as f32) - 9.0).collect();
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0x7fc0_0002),
        f32::from_bits(0xffc0_0001), // -NaN
        f32::from_bits(0xffc0_0002), // -NaN
    ];
    for r in 0..rows {
        for (j, &s) in specials.iter().enumerate() {
            x[r * d + (j * 29 + r * 3) % d] = s;
        }
    }
    // Cover k=0, k=1 (zero-alloc scan), k=2/3/10/32 (heap), k=100/257 (select_nth/full).
    for k in [0, 1, 2, 3, 10, 32, 100, d] {
        for largest in [true, false] {
            let (gv, gi) = topk(&x, rows, d, k, largest);
            let (wv, wi) = ref_topk(&x, rows, d, k, largest);
            assert_eq!(gi, wi, "topk idx mismatch (k={k}, largest={largest})");
            assert_eq!(
                gv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                wv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "topk val bits mismatch (k={k}, largest={largest})"
            );
        }
    }
    // Empty rows and parallel rows (> PAR_MIN).
    assert_eq!(topk(&[], 0, 10, 3, true), (vec![], vec![]));
    let (pr, pd, pk) = (64, 600, 12);
    let px: Vec<f32> = (0..pr * pd).map(|i| ((i as f32) * 0.17).sin()).collect();
    let (gv, gi) = topk(&px, pr, pd, pk, true);
    let (wv, wi) = ref_topk(&px, pr, pd, pk, true);
    assert_eq!(gi, wi);
    assert_eq!(
        gv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        wv.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn sum_and_col_sum_avx2_match_scalar_bitwise() {
    for len in [0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 255, 257, 1039] {
        let xs: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.41).sin() * 17.5).collect();
        assert_eq!(sum_slice(&xs).to_bits(), sum_slice_scalar(&xs).to_bits(), "sum len={len}");
        let ys: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.19).cos() * 3.25).collect();
        let mut got = xs.clone();
        let mut want = xs;
        col_sum_slice(&mut got, &ys);
        for (a, &b) in want.iter_mut().zip(&ys) {
            *a += b;
        }
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "col_sum len={len}"
        );
    }
    // Special IEEE values: signed zero, infinity, NaN.
    let special = vec![
        -0.0f32,
        0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        2.0,
        -0.0,
        0.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
    ];
    assert_eq!(sum_slice(&special).to_bits(), sum_slice_scalar(&special).to_bits());
    let mut with_nan = special.clone();
    with_nan[9] = f32::NAN;
    assert!(sum_slice(&with_nan).is_nan());
}

#[test]
fn reduce_var_matches_f64_reference_across_shapes_and_edges() {
    for &(outer, d, inner) in &[(3, 19, 1), (2, 10003, 1), (2, 17, 300), (1, 1, 5), (0, 7, 3)] {
        let x: Vec<f32> = (0..outer * d * inner).map(|i| ((i as f32) * 0.13).sin() * 4.0 + 10.0).collect();
        for unbiased in [false, true] {
            let got = reduce_var_f32(&x, (outer, d, inner), unbiased);
            assert_eq!(got.len(), outer * inner);
            for o in 0..outer {
                for c in 0..inner {
                    let g = got[o * inner + c];
                    let denom = if unbiased { d.saturating_sub(1) } else { d };
                    if denom == 0 {
                        assert!(g.is_nan(), "expected NaN for denom=0 at ({o},{c})");
                        continue;
                    }
                    let mean = (0..d).map(|r| x[(o * d + r) * inner + c] as f64).sum::<f64>() / d as f64;
                    let var = (0..d)
                        .map(|r| {
                            let df = x[(o * d + r) * inner + c] as f64 - mean;
                            df * df
                        })
                        .sum::<f64>()
                        / denom as f64;
                    assert!(
                        (g as f64 - var).abs() <= 1e-4 * var.abs().max(1.0),
                        "var mismatch ({outer},{d},{inner}) unbiased={unbiased}: got {g} vs {var}"
                    );
                }
            }
        }
    }
}
