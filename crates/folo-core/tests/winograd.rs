//! The CPU inference conv's Winograd weights: transformed once per weight tensor, and never stale.

use folo_core::rng::Rng;
use folo_core::{Conv2dParams, Tensor, UnaryOp, no_grad};

const P: Conv2dParams = Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };

/// `act(conv2d(x, w) + b)` through the fused inference entry point, as bits.
fn conv_act(x: &Tensor, w: &Tensor, b: &Tensor) -> Vec<u32> {
    let y = x.conv2d_act(w, Some(b), &P, Some(UnaryOp::Silu)).unwrap();
    y.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// A fresh tensor (its own storage, nothing derived yet) with `t`'s values.
fn fresh(t: &Tensor) -> Tensor {
    Tensor::from_vec(t.to_vec::<f32>().unwrap(), t.dims().to_vec()).unwrap()
}

/// A 3x3 stride-1 conv with 32 channels at 32x32 runs as Winograd F(2x2, 3x3) when nothing is recorded: close to
/// the autograd conv (the direct GEMM) but not bit for bit.
#[test]
fn inference_conv_takes_winograd() {
    let mut rng = Rng::seed(1);
    let x = Tensor::randn([1, 32, 32, 32], 0.0, 1.0, &mut rng);
    let w = Tensor::randn([32, 32, 3, 3], 0.0, 0.1, &mut rng);
    let b = Tensor::randn([32], 0.0, 0.1, &mut rng);
    let direct = x.conv2d_with(&w, Some(&b), &P).unwrap().silu().unwrap().to_vec::<f32>().unwrap();
    let _g = no_grad();
    let wino: Vec<f32> = conv_act(&x, &w, &b).into_iter().map(f32::from_bits).collect();
    let max_err = direct.iter().zip(&wino).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max_err < 1e-5, "Winograd and direct differ by {max_err}");
    assert!(
        direct.iter().zip(&wino).any(|(a, b)| a.to_bits() != b.to_bits()),
        "the inference conv was not Winograd"
    );
}

/// The transformed weights are reused while the weights are unchanged, follow every in-place write
/// (`copy_from`, `update_f32`, `sub_assign`), and are kept per layout for views of one storage.
#[test]
fn transformed_weights_follow_the_weights() {
    let mut rng = Rng::seed(2);
    let x = Tensor::randn([2, 32, 34, 33], 0.0, 1.0, &mut rng);
    let w = Tensor::randn([32, 32, 3, 3], 0.0, 0.1, &mut rng);
    let b = Tensor::randn([32], 0.0, 0.1, &mut rng);
    let _g = no_grad();
    let first = conv_act(&x, &w, &b);
    assert_eq!(conv_act(&x, &w, &b), first, "a second call (cached weights) differs");
    assert_eq!(conv_act(&x, &fresh(&w), &b), first, "cached and fresh weights differ");

    let w2 = Tensor::randn([32, 32, 3, 3], 0.0, 0.1, &mut rng);
    w.copy_from(&w2).unwrap();
    let after_copy = conv_act(&x, &w, &b);
    assert_ne!(after_copy, first);
    assert_eq!(after_copy, conv_act(&x, &w2, &b), "stale weights after copy_from");

    w.update_f32(|v| v.iter_mut().for_each(|e| *e *= -2.0)).unwrap();
    assert_eq!(conv_act(&x, &w, &b), conv_act(&x, &fresh(&w), &b), "stale weights after update_f32");

    w.sub_assign(&w2).unwrap();
    assert_eq!(conv_act(&x, &w, &b), conv_act(&x, &fresh(&w), &b), "stale weights after sub_assign");

    // Two halves of one storage: each its own transformed weights.
    let both = Tensor::randn([64, 32, 3, 3], 0.0, 0.1, &mut rng);
    let (lo, hi) = (both.narrow(0, 0, 32).unwrap(), both.narrow(0, 32, 32).unwrap());
    for _ in 0..2 {
        assert_eq!(conv_act(&x, &lo, &b), conv_act(&x, &fresh(&lo), &b), "first half");
        assert_eq!(conv_act(&x, &hi, &b), conv_act(&x, &fresh(&hi), &b), "second half");
    }
}
