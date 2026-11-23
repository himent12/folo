//! The CPU inference conv's Winograd dispatch and its transformed weights: taken by the inference conv only,
//! transformed once per weight tensor, never stale, and never another storage's. Dispatch and transforms are
//! observed through the calls counted on the test's thread (`conv::winograd_calls`).

use super::CpuBackend;
use super::conv::{transform4_attempts, winograd_calls, winograd4_calls};
use crate::backend::Backend;
use crate::rng::Rng;
use crate::{Conv2dParams, CpuStorage, Storage, Tensor, UnaryOp, no_grad};

const P: Conv2dParams = Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };

/// `f`'s result and the `[weight transforms, Winograd forwards]` it ran on this thread.
fn counting<R>(f: impl FnOnce() -> R) -> (R, [usize; 2]) {
    let before = winograd_calls();
    let r = f();
    let after = winograd_calls();
    (r, [after[0] - before[0], after[1] - before[1]])
}

/// `f`'s result and `[F(4x4) transforms, F(4x4) forwards]` on this thread.
fn counting4<R>(f: impl FnOnce() -> R) -> (R, [usize; 2]) {
    let before = winograd4_calls();
    let r = f();
    let after = winograd4_calls();
    (r, [after[0] - before[0], after[1] - before[1]])
}

/// `act(conv2d(x, w) + b)` through the fused inference entry point, as bits, and the calls it made.
fn conv_act(x: &Tensor, w: &Tensor, b: &Tensor) -> (Vec<u32>, [usize; 2]) {
    counting(|| {
        let y = x.conv2d_act(w, Some(b), &P, Some(UnaryOp::Silu)).unwrap();
        y.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
    })
}

/// A fresh tensor (its own storage, nothing derived yet) with `t`'s values.
fn fresh(t: &Tensor) -> Tensor {
    Tensor::from_vec(t.to_vec::<f32>().unwrap(), t.dims().to_vec()).unwrap()
}

/// 3x3 stride-1 convs take Winograd (`F(2x2)` on 32->32 32x32 and `F(4x4)` on 64->64 40x40) through the fused
/// inference entry point, and the direct conv through the autograd one: close, not bit for bit.
#[test]
fn inference_conv_takes_winograd() {
    let mut rng = Rng::seed(1);
    for (c, hw, expect_f4) in [(32usize, 32usize, false), (64, 40, true)] {
        let x = Tensor::randn([1, c, hw, hw], 0.0, 1.0, &mut rng);
        let w = Tensor::randn([c, c, 3, 3], 0.0, 0.1, &mut rng);
        let b = Tensor::randn([c], 0.0, 0.1, &mut rng);
        let (direct, calls) = counting(|| x.conv2d_with(&w, Some(&b), &P).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(calls, [0, 0], "the autograd conv ran Winograd ({c}@{hw})");
        let _g = no_grad();
        let ((wino, calls4), calls) =
            counting(|| counting4(|| x.conv2d_act(&w, Some(&b), &P, None).unwrap().to_vec::<f32>().unwrap()));
        assert_eq!(calls, [1, 1], "the inference conv did not run Winograd ({c}@{hw})");
        assert_eq!(
            calls4,
            if expect_f4 { [1, 1] } else { [0, 0] },
            "unexpected F(4x4) dispatch for ({c}@{hw})"
        );
        assert!(direct.iter().all(|v| v.is_finite()) && wino.iter().all(|v| v.is_finite()));
        let max_err = direct.iter().zip(&wino).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_err < 1e-4, "Winograd and direct differ by {max_err} ({c}@{hw})");
        assert!(
            direct.iter().zip(&wino).any(|(a, b)| a.to_bits() != b.to_bits()),
            "bit for bit the direct conv ({c}@{hw})"
        );
    }
}

/// The transformed weights are made once and reused while the weights are unchanged, made again after every
/// in-place write (`copy_from`, `update_f32`, `sub_assign`), and kept per layout for views of one storage.
#[test]
fn transformed_weights_follow_the_weights() {
    let mut rng = Rng::seed(2);
    let x = Tensor::randn([2, 32, 34, 33], 0.0, 1.0, &mut rng);
    let w = Tensor::randn([32, 32, 3, 3], 0.0, 0.1, &mut rng);
    let b = Tensor::randn([32], 0.0, 0.1, &mut rng);
    let _g = no_grad();
    let (first, calls) = conv_act(&x, &w, &b);
    assert_eq!(calls, [1, 1], "first call");
    assert_eq!(conv_act(&x, &w, &b), (first.clone(), [0, 1]), "a second call did not reuse the weights");
    assert_eq!(conv_act(&x, &fresh(&w), &b), (first.clone(), [1, 1]), "cached and fresh weights differ");

    let w2 = Tensor::randn([32, 32, 3, 3], 0.0, 0.1, &mut rng);
    w.copy_from(&w2).unwrap();
    let (after_copy, calls) = conv_act(&x, &w, &b);
    assert_eq!(calls, [1, 1], "copy_from did not clear the transformed weights");
    assert_ne!(after_copy, first);
    assert_eq!(after_copy, conv_act(&x, &w2, &b).0, "stale weights after copy_from");
    assert_eq!(conv_act(&x, &w, &b), (after_copy, [0, 1]), "not reused after copy_from");

    w.update_f32(|v| v.iter_mut().for_each(|e| *e *= -2.0)).unwrap();
    let (after_update, calls) = conv_act(&x, &w, &b);
    assert_eq!(calls, [1, 1], "update_f32 did not clear the transformed weights");
    assert_eq!(after_update, conv_act(&x, &fresh(&w), &b).0, "stale weights after update_f32");

    w.sub_assign(&w2).unwrap();
    let (after_sub, calls) = conv_act(&x, &w, &b);
    assert_eq!(calls, [1, 1], "sub_assign did not clear the transformed weights");
    assert_eq!(after_sub, conv_act(&x, &fresh(&w), &b).0, "stale weights after sub_assign");

    // Two halves of one storage: each its own transformed weights, made once.
    let both = Tensor::randn([64, 32, 3, 3], 0.0, 0.1, &mut rng);
    let (lo, hi) = (both.narrow(0, 0, 32).unwrap(), both.narrow(0, 32, 32).unwrap());
    let (want_lo, want_hi) = (conv_act(&x, &fresh(&lo), &b).0, conv_act(&x, &fresh(&hi), &b).0);
    for transforms in [1, 0] {
        assert_eq!(conv_act(&x, &lo, &b), (want_lo.clone(), [transforms, 1]), "first half");
        assert_eq!(conv_act(&x, &hi, &b), (want_hi.clone(), [transforms, 1]), "second half");
    }
}

/// A tensor's view whose `storage` field was pointed at another storage neither uses nor fills the first
/// storage's transformed weights.
#[test]
fn rebound_view_does_not_use_the_cache() {
    let x = Tensor::from_vec(vec![1f32; 64 * 4 * 4], vec![1, 64, 4, 4]).unwrap();
    let w = Tensor::from_vec(vec![1f32; 64 * 64 * 9], vec![64, 64, 3, 3]).unwrap();
    let zeros = Storage::Cpu(CpuStorage::F32(vec![0.0; 64 * 64 * 9]));
    let _g = no_grad();
    let run = |rebind: bool| {
        counting(|| {
            x.with_view(|xv| {
                w.with_view(|wv| {
                    // A shorter-lived copy of the view, which may borrow `zeros`.
                    let mut wv = wv;
                    if rebind {
                        wv.storage = &zeros;
                    }
                    match CpuBackend.conv2d_act(xv, wv, None, &P, None).unwrap() {
                        Storage::Cpu(CpuStorage::F32(v)) => v,
                        s => panic!("not f32: {s:?}"),
                    }
                })
            })
        })
    };
    let (ones, calls) = run(false);
    assert_eq!(calls, [1, 1]);
    // Corners see 2x2 of the 3x3 taps, edges 2x3, the middle all 9: times 64 channels.
    assert_eq!(&ones[..4], &[256.0, 384.0, 384.0, 256.0]);
    assert_eq!(ones[5], 576.0);
    let (zero, calls) = run(true);
    assert_eq!(calls, [1, 1], "the rebound view did not transform its own weights");
    assert!(zero.iter().all(|&v| v == 0.0), "the rebound view used the first storage's weights");
    assert_eq!(run(false), (ones, [0, 1]), "the rebound view replaced the first storage's weights");
}

/// Public `conv2d_act` edge cases around Winograd F(4x4) dispatch:
/// - `groups > 1`, `dilation > 1`, and `stride != 1` never take Winograd and stay bit-exact with direct conv.
/// - Non-unit and asymmetric paddings (`[0,0]`, `[2,2]`, `[5,5]`, `[0,2]`) with odd sizes take F(4x4) and
///   match direct conv within `1e-4`.
#[test]
fn winograd4_public_conv_edge_cases() {
    let mut rng = Rng::seed(3);
    let _g = no_grad();
    // Non-Winograd geometries must bypass Winograd completely and match `conv2d_with` bit for bit.
    let bypass = [
        (64, 64, 32, 32, Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 2 }),
        (64, 64, 32, 32, Conv2dParams { stride: [1, 1], padding: [2, 2], dilation: [2, 2], groups: 1 }),
        (64, 64, 32, 32, Conv2dParams { stride: [2, 2], padding: [1, 1], dilation: [1, 1], groups: 1 }),
    ];
    for (c, o, h, w_dim, p) in bypass {
        let x = Tensor::randn([1, c, h, w_dim], 0.0, 1.0, &mut rng);
        let w = Tensor::randn([o, c / p.groups, 3, 3], 0.0, 0.1, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let direct = x.conv2d_with(&w, Some(&b), &p).unwrap().to_vec::<f32>().unwrap();
        let ((got, calls4), calls) =
            counting(|| counting4(|| x.conv2d_act(&w, Some(&b), &p, None).unwrap().to_vec::<f32>().unwrap()));
        assert_eq!(calls, [0, 0], "unexpected Winograd call for {p:?}");
        assert_eq!(calls4, [0, 0], "unexpected F(4x4) call for {p:?}");
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            direct.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "bypass mismatch for {p:?}"
        );
    }
    // Non-unit and asymmetric paddings with odd spatial dimensions take F(4x4) cleanly.
    let x = Tensor::randn([1, 64, 33, 35], 0.0, 1.0, &mut rng);
    let b = Tensor::randn([64], 0.0, 0.1, &mut rng);
    for padding in [[0, 0], [2, 2], [5, 5], [0, 2]] {
        let p = Conv2dParams { stride: [1, 1], padding, dilation: [1, 1], groups: 1 };
        let w = Tensor::randn([64, 64, 3, 3], 0.0, 0.1, &mut rng);
        let direct = x.conv2d_with(&w, Some(&b), &p).unwrap().to_vec::<f32>().unwrap();
        let (got, calls4) =
            counting4(|| x.conv2d_act(&w, Some(&b), &p, None).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(calls4, [1, 1], "expected F(4x4) for padding {padding:?}");
        assert!(direct.iter().all(|v| v.is_finite()) && got.iter().all(|v| v.is_finite()));
        let max_err = direct.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_err < 2e-4, "F(4x4) padding {padding:?} max_err {max_err}");
    }
}

/// Huge finite weights (e.g., `1e38` at tap `(0, 0)`, which `G4[0,0]^2 = 16` overflows past `f32::MAX`)
/// fall back cleanly to the direct conv instead of producing `0 * inf = NaN` in Winograd F(4x4).
#[test]
fn winograd4_large_finite_weight_falls_back_without_nan() {
    let _g = no_grad();
    let mut w_data = vec![0.0f32; 64 * 64 * 9];
    w_data[0] = 1e38_f32;
    let w = Tensor::from_vec(w_data, [64, 64, 3, 3]).unwrap();

    // Zero input: must return all exact zeros (zero NaNs), matching `conv2d_with`, and cache the `None` rejection.
    let x_zero = Tensor::zeros([1, 64, 40, 40]);
    let direct_zero = x_zero.conv2d_with(&w, None, &P).unwrap().to_vec::<f32>().unwrap();
    for pass in 0..2 {
        let attempts_before = transform4_attempts();
        let (got, calls4) =
            counting4(|| x_zero.conv2d_act(&w, None, &P, None).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(
            transform4_attempts() - attempts_before,
            usize::from(pass == 0),
            "pass {pass}: expected cached None rejection"
        );
        assert_eq!(calls4, [0, 0], "pass {pass}: overflowed weights must not run F(4x4)");
        assert!(got.iter().all(|&v| v.to_bits() == 0), "pass {pass}: expected all exact zeros");
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            direct_zero.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }
    let at = Tensor::from_vec(vec![0u32, 41, 1599], [3]).unwrap();
    let attempts_before = transform4_attempts();
    let (got_tiles, calls4) =
        counting4(|| x_zero.conv2d_act_at_tiles(&w, None, &P, None, &at).unwrap().to_vec::<f32>().unwrap());
    assert_eq!(
        transform4_attempts() - attempts_before,
        0,
        "conv2d_act_at_tiles must reuse cached None rejection"
    );
    assert_eq!(calls4, [0, 0]);
    let (got_at, calls4_at) =
        counting4(|| x_zero.conv2d_act_at(&w, None, &P, None, &at).unwrap().to_vec::<f32>().unwrap());
    assert_eq!(calls4_at, [0, 0]);
    for &s in &[0usize, 41, 1599] {
        for oc in 0..64 {
            assert_eq!(got_tiles[oc * 1600 + s].to_bits(), 0);
            assert_eq!(got_at[oc * 1600 + s].to_bits(), 0);
        }
    }

    // Small non-zero input (`1e-10 * 1e38 = 1e28`, finite): must match direct conv bit for bit.
    let x_small = Tensor::from_vec(vec![1e-10_f32; 64 * 40 * 40], [1, 64, 40, 40]).unwrap();
    let direct_small = x_small.conv2d_with(&w, None, &P).unwrap().to_vec::<f32>().unwrap();
    let (got_small, calls4) =
        counting4(|| x_small.conv2d_act(&w, None, &P, None).unwrap().to_vec::<f32>().unwrap());
    assert_eq!(calls4, [0, 0]);
    assert!(got_small.iter().all(|v| v.is_finite()));
    assert_eq!(
        got_small.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        direct_small.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );
}

/// Transform-domain GEMM (`U * V`) or input-transform (`Bᵀ d B`) overflow with finite `U`:
/// per-tile pre-epilogue finiteness detection falls back to the exact direct 3x3 sum for the overflowing
/// `(oc, tile)` so neither `conv2d_act` nor `conv2d_act_at_tiles` emits `NaN`/`inf` even under `ReLU`/`SiLU`/`Sigmoid`.
#[test]
fn winograd4_transform_domain_gemm_overflow_falls_back_per_tile() {
    let _g = no_grad();
    // Exact repro: `weight[0,0,0,0] = 2e37` (`U[0,0] = 3.2e38`, finite), `input[0,0,5,5] = 2.0` (`U*V` overflows).
    let mut w_data = vec![0.0f32; 64 * 64 * 9];
    w_data[0] = 2e37_f32;
    let w = Tensor::from_vec(w_data, [64, 64, 3, 3]).unwrap();
    let mut x_data = vec![0.0f32; 64 * 40 * 40];
    x_data[5 * 40 + 5] = 2.0_f32;
    let x = Tensor::from_vec(x_data, [1, 64, 40, 40]).unwrap();

    let sites_idx = [4 * 40 + 4, 6 * 40 + 6, 0, 39 * 40 + 39];
    let at = Tensor::from_vec(sites_idx.map(|s| s as u32).to_vec(), [sites_idx.len()]).unwrap();
    let direct = x.conv2d_with(&w, None, &P).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(direct[4 * 40 + 4], 0.0);
    assert_eq!(direct[6 * 40 + 6], 4e37_f32);

    for (i, act) in
        [None, Some(UnaryOp::Relu), Some(UnaryOp::Silu), Some(UnaryOp::Sigmoid)].into_iter().enumerate()
    {
        let (dense, calls4) = counting4(|| x.conv2d_act(&w, None, &P, act).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(calls4, [usize::from(i == 0), 1], "act={act:?}: expected F(4x4) dispatch");
        assert!(dense.iter().all(|v| v.is_finite()), "act={act:?}: non-finite output in dense");
        if act.is_none() || act == Some(UnaryOp::Relu) {
            assert_eq!(dense[4 * 40 + 4], 0.0, "act={act:?}: (4,4)");
            assert_eq!(dense[6 * 40 + 6], 4e37_f32, "act={act:?}: (6,6)");
        }
        let (sparse_tiles, calls4_tiles) =
            counting4(|| x.conv2d_act_at_tiles(&w, None, &P, act, &at).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(calls4_tiles, [0, 1], "act={act:?}: expected F(4x4) tiled sparse dispatch");
        for &s in &sites_idx {
            for oc in 0..64 {
                let idx = oc * 1600 + s;
                assert_eq!(
                    sparse_tiles[idx].to_bits(),
                    dense[idx].to_bits(),
                    "act={act:?}: sparse_tiles != dense at oc={oc}, s={s}"
                );
            }
        }
        // `conv2d_act_at` must stay off F(4x4) and also produce exact finite results.
        let (sparse_rf, calls4_rf) =
            counting4(|| x.conv2d_act_at(&w, None, &P, act, &at).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(calls4_rf, [0, 0], "act={act:?}: conv2d_act_at must not run F(4x4)");
        if act.is_none() || act == Some(UnaryOp::Relu) {
            assert_eq!(sparse_rf[4 * 40 + 4], 0.0, "act={act:?}: conv2d_act_at (4,4)");
            assert_eq!(sparse_rf[6 * 40 + 6], 4e37_f32, "act={act:?}: conv2d_act_at (6,6)");
        }
    }

    // Input-transform `Bᵀ d B` overflow (`input = 3e38`, `weight = 0`): must return exact zeros (zero NaNs).
    let w_zero = Tensor::zeros([64, 64, 3, 3]);
    let mut x_huge = vec![0.0f32; 64 * 40 * 40];
    x_huge[5 * 40 + 5] = 3e38_f32;
    let x_huge = Tensor::from_vec(x_huge, [1, 64, 40, 40]).unwrap();
    let (got, calls4) =
        counting4(|| x_huge.conv2d_act(&w_zero, None, &P, None).unwrap().to_vec::<f32>().unwrap());
    assert_eq!(calls4, [1, 1]);
    assert!(got.iter().all(|&v| v.to_bits() == 0), "V overflow with zero weight must return exact zeros");
}

/// Per-tile direct fallback accumulates in `f64`, so canceling terms whose intermediate product exceeds
/// `f32::MAX` (e.g. `-2e38 * 1 + 2e38 * 2 = 2e38`, in either channel order) return the exact finite `2e38`
/// rather than overflowing to `+inf` before canceling.
#[test]
fn winograd4_direct_fallback_cancellation_stays_finite() {
    let _g = no_grad();
    let site = 4 * 40 + 4;
    let at = Tensor::from_vec(vec![site as u32], [1]).unwrap();
    let bias = Tensor::from_vec(vec![0.5f32; 64], [64]).unwrap();

    // Test both `(-2e38 * 1) + (2e38 * 2)` and `(2e38 * 2) + (-2e38 * 1)` channel orders.
    for (a_val, a_wt, b_val, b_wt) in
        [(-2e38_f32, 1.0f32, 2e38_f32, 2.0f32), (2e38_f32, 2.0f32, -2e38_f32, 1.0f32)]
    {
        let mut w_data = vec![0.0f32; 64 * 64 * 9];
        w_data[0] = a_wt; // (oc=0, ic=0, ky=0, kx=0) reads input (3, 3) for output (4, 4)
        w_data[9] = b_wt; // (oc=0, ic=1, ky=0, kx=0) reads input (3, 3) for output (4, 4)
        let w = Tensor::from_vec(w_data, [64, 64, 3, 3]).unwrap();
        let mut x_data = vec![0.0f32; 64 * 40 * 40];
        x_data[3 * 40 + 3] = a_val;
        x_data[1600 + 3 * 40 + 3] = b_val;
        let x = Tensor::from_vec(x_data, [1, 64, 40, 40]).unwrap();

        for (i, (b_opt, act)) in [(None, None), (Some(&bias), Some(UnaryOp::Silu))].into_iter().enumerate() {
            let (dense, calls4) =
                counting4(|| x.conv2d_act(&w, b_opt, &P, act).unwrap().to_vec::<f32>().unwrap());
            assert_eq!(calls4, [usize::from(i == 0), 1], "expected F(4x4) with per-tile direct fallback");
            assert!(dense.iter().all(|v| v.is_finite()), "non-finite output on cancellation case");
            assert_eq!(dense[site], 2e38_f32, "expected exact finite 2e38 at (4,4)");
            let (sparse, calls4_at) = counting4(|| {
                x.conv2d_act_at_tiles(&w, b_opt, &P, act, &at).unwrap().to_vec::<f32>().unwrap()
            });
            assert_eq!(calls4_at, [0, 1]);
            for oc in 0..64 {
                assert_eq!(sparse[oc * 1600 + site].to_bits(), dense[oc * 1600 + site].to_bits());
            }
        }
    }
}
