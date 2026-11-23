//! The CPU inference conv's Winograd dispatch and its transformed weights: taken by the inference conv only,
//! transformed once per weight tensor, never stale, and never another storage's. Dispatch and transforms are
//! observed through the calls counted on the test's thread (`conv::winograd_calls`).

use super::CpuBackend;
use super::conv::winograd_calls;
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
    for (c, hw) in [(32usize, 32usize), (64, 40)] {
        let x = Tensor::randn([1, c, hw, hw], 0.0, 1.0, &mut rng);
        let w = Tensor::randn([c, c, 3, 3], 0.0, 0.1, &mut rng);
        let b = Tensor::randn([c], 0.0, 0.1, &mut rng);
        let (direct, calls) = counting(|| x.conv2d_with(&w, Some(&b), &P).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(calls, [0, 0], "the autograd conv ran Winograd ({c}@{hw})");
        let _g = no_grad();
        let (wino, calls) =
            counting(|| x.conv2d_act(&w, Some(&b), &P, None).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(calls, [1, 1], "the inference conv did not run Winograd ({c}@{hw})");
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
