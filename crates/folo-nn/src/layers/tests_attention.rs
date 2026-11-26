//! Unit tests for differentiable vision building blocks (`SqueezeExcitation`, `Eca`, `Cbam`,
//! `ChannelAttention`, `SpatialAttention`, `WeightStandardizedConv2d`, `standardize_weight`):
//! - Central finite-difference `grad_check_with` on inputs AND every trainable parameter,
//! - Exact mathematical invariants (`eca_kernel_size`, zero-mean / unit-variance / shift-invariance of
//!   `standardize_weight`),
//! - Clean parameter naming and `.folo` state dict save/load round-trip (via `TestDir`),
//! - Zero-batch (`N == 0`), odd spatial dimensions (`[2, C, 5, 7]`, `[1, C, 1, 1]`), `fan_in == 1`,
//!   constant filters, and invalid input/config rejection.

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_test_support::TestDir;

use super::{
    Activation, Cbam, ChannelAttention, Conv2d, Conv2dConfig, Eca, Sequential, SpatialAttention,
    SqueezeExcitation, WS_CONV_DEFAULT_EPS, WeightStandardizedConv2d, eca_kernel_size, standardize_weight,
};
use crate::io::{FileKind, LoadOptions, Metadata, ModelConfig, ModelFile, Purpose, save_weights};
use crate::module::{Forward, Module};

/// Central finite-difference `grad_check_with` on input `x` and EVERY trainable parameter of
/// `SqueezeExcitation` (`fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`).
#[test]
fn squeeze_excitation_grad_check_and_edge_cases() {
    let mut rng = Rng::seed(101);
    let x = Tensor::randn([2, 4, 3, 5], 0.0, 1.0, &mut rng); // odd spatial dims 3x5
    let se = SqueezeExcitation::with_hidden(4, 2, Activation::Silu, &mut rng).unwrap();
    assert_eq!((se.channels(), se.hidden(), se.activation()), (4, 2, Activation::Silu));

    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 11 };
    let w1 = se.fc1.weight.detach();
    let b1 = se.fc1.bias.as_ref().unwrap().detach();
    let w2 = se.fc2.weight.detach();
    let b2 = se.fc2.bias.as_ref().unwrap().detach();
    grad_check_with(
        |inps| {
            let mut m = SqueezeExcitation::with_hidden(4, 2, Activation::Silu, &mut Rng::seed(0)).unwrap();
            m.fc1 = Conv2d::from_tensors(
                Conv2dConfig::new(4, 2, 1).padding(0).bias(true),
                inps[1].clone(),
                Some(inps[2].clone()),
            )
            .unwrap();
            m.fc2 = Conv2d::from_tensors(
                Conv2dConfig::new(2, 4, 1).padding(0).bias(true),
                inps[3].clone(),
                Some(inps[4].clone()),
            )
            .unwrap();
            m.forward(&inps[0]).unwrap().square()?.sum_all()
        },
        &[x.clone(), w1, b1, w2, b2],
        &gc,
    )
    .unwrap();

    // ReLU default reduction path + 1x1 spatial + zero-batch.
    let se_def = SqueezeExcitation::new(4, 2, &mut rng).unwrap();
    assert_eq!(
        se_def.forward(&Tensor::randn([1, 4, 1, 1], 0.0, 1.0, &mut rng)).unwrap().dims(),
        &[1, 4, 1, 1]
    );
    let zero_batch = Tensor::zeros([0, 4, 5, 7]).requires_grad_();
    let y0 = se_def.forward(&zero_batch).unwrap();
    assert_eq!(y0.dims(), &[0, 4, 5, 7]);
    assert_eq!(y0.sum_all().unwrap().backward().unwrap().get(&zero_batch).unwrap().dims(), &[0, 4, 5, 7]);

    // Validation.
    assert!(SqueezeExcitation::new(0, 4, &mut rng).is_err());
    assert!(SqueezeExcitation::new(4, 0, &mut rng).is_err());
    assert!(se_def.forward(&Tensor::zeros([1, 3, 4, 4])).is_err());
    assert!(se_def.forward(&Tensor::zeros([1, 4, 0, 4])).is_err());
    assert!(se_def.forward(&Tensor::zeros([4, 4])).is_err());
}

/// Adaptive `eca_kernel_size`, central finite-difference `grad_check_with` on `Eca` input and kernel,
/// odd spatial sizes, zero-batch, and validation.
#[test]
fn eca_kernel_formula_grad_check_and_edge_cases() {
    assert_eq!(eca_kernel_size(1, 2.0, 1.0).unwrap(), 1);
    assert_eq!(eca_kernel_size(64, 2.0, 1.0).unwrap(), 3);
    assert_eq!(eca_kernel_size(256, 2.0, 1.0).unwrap(), 5);
    assert!(eca_kernel_size(0, 2.0, 1.0).is_err());
    assert!(eca_kernel_size(64, 0.0, 1.0).is_err());
    assert!(eca_kernel_size(64, 1e-30, 1.0).is_err()); // huge kernel rejected instead of saturating usize::MAX
    assert!(eca_kernel_size(64, 2.0, f32::NAN).is_err());

    let mut rng = Rng::seed(202);
    let eca = Eca::with_kernel_size(5, 3, &mut rng).unwrap();
    assert_eq!((eca.channels(), eca.kernel_size()), (5, 3));
    let x = Tensor::randn([2, 5, 3, 5], 0.0, 1.0, &mut rng);
    let w = eca.weight.detach();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 22 };
    grad_check_with(
        |inps| {
            let mut m = Eca::with_kernel_size(5, 3, &mut Rng::seed(0)).unwrap();
            m.weight = inps[1].clone();
            m.forward(&inps[0]).unwrap().square()?.sum_all()
        },
        &[x, w],
        &gc,
    )
    .unwrap();

    // Adaptive constructor + zero-batch + odd spatial + invalid kernel size.
    let eca_auto = Eca::new(64, &mut rng).unwrap();
    assert_eq!(eca_auto.kernel_size(), 3);
    assert_eq!(eca_auto.forward(&Tensor::zeros([0, 64, 3, 5])).unwrap().dims(), &[0, 64, 3, 5]);
    assert!(Eca::with_kernel_size(4, 0, &mut rng).is_err());
    assert!(Eca::with_kernel_size(4, 4, &mut rng).is_err()); // even kernel rejected
    assert!(eca.forward(&Tensor::zeros([1, 4, 3, 3])).is_err());
}

/// Central finite-difference `grad_check_with` on `Cbam` input and EVERY trainable parameter
/// (`channel.fc1.weight`, `channel.fc2.weight`, `spatial.conv.weight`), odd spatial dims, and zero-batch.
#[test]
fn cbam_grad_check_and_edge_cases() {
    let mut rng = Rng::seed(303);
    let cbam = Cbam::new(4, 2, 3, &mut rng).unwrap();
    assert_eq!((cbam.channels(), cbam.spatial_kernel()), (4, 3));

    // Distinct well-separated random entries avoid exact max-tie kinks in finite differences.
    let x = Tensor::randn([2, 4, 3, 5], 0.2, 1.2, &mut rng);
    let fc1_w = cbam.channel.fc1.weight.detach();
    let fc2_w = cbam.channel.fc2.weight.detach();
    let sp_w = cbam.spatial.conv.weight.detach();
    let gc = GradCheck { eps: 1e-3, atol: 3e-3, rtol: 3e-2, max_checks: 16, seed: 33 };
    grad_check_with(
        |inps| {
            let mut m = Cbam::new(4, 2, 3, &mut Rng::seed(0)).unwrap();
            m.channel.fc1 = Conv2d::from_tensors(
                Conv2dConfig::new(4, 2, 1).padding(0).bias(false),
                inps[1].clone(),
                None,
            )
            .unwrap();
            m.channel.fc2 = Conv2d::from_tensors(
                Conv2dConfig::new(2, 4, 1).padding(0).bias(false),
                inps[2].clone(),
                None,
            )
            .unwrap();
            m.spatial.conv = Conv2d::from_tensors(
                Conv2dConfig::new(2, 1, 3).padding(1).bias(false),
                inps[3].clone(),
                None,
            )
            .unwrap();
            m.forward(&inps[0]).unwrap().square()?.sum_all()
        },
        &[x, fc1_w, fc2_w, sp_w],
        &gc,
    )
    .unwrap();

    // Odd spatial 1x1, 5x7, and zero-batch [0, 4, 5, 7].
    assert_eq!(cbam.forward(&Tensor::randn([1, 4, 1, 1], 0.0, 1.0, &mut rng)).unwrap().dims(), &[1, 4, 1, 1]);
    assert_eq!(cbam.forward(&Tensor::randn([2, 4, 5, 7], 0.0, 1.0, &mut rng)).unwrap().dims(), &[2, 4, 5, 7]);
    assert_eq!(cbam.forward(&Tensor::zeros([0, 4, 5, 7])).unwrap().dims(), &[0, 4, 5, 7]);

    // Validation.
    assert!(ChannelAttention::new(0, 2, &mut rng).is_err());
    assert!(ChannelAttention::new(4, 0, &mut rng).is_err());
    assert!(SpatialAttention::new(0, &mut rng).is_err());
    assert!(SpatialAttention::new(4, &mut rng).is_err()); // even kernel rejected
    assert!(SpatialAttention::new(usize::MAX, &mut rng).is_err()); // overflow rejected
    assert!(Cbam::new(4, 2, 6, &mut rng).is_err());
}

/// Exact zero-mean / unit-variance / shift-invariance properties of `standardize_weight`, central
/// finite-difference `grad_check_with` on `standardize_weight` and `WeightStandardizedConv2d` (input,
/// weight, bias), constant-filter & `fan_in == 1` safety, and zero-batch.
#[test]
fn weight_standardized_conv2d_properties_grad_check_and_edge_cases() {
    let mut rng = Rng::seed(404);
    let w = Tensor::randn([4, 3, 3, 3], 2.5, 1.7, &mut rng);
    let w_hat = standardize_weight(&w, WS_CONV_DEFAULT_EPS).unwrap();
    assert_eq!(w_hat.dims(), w.dims());

    // Per-output-channel zero mean and ~unit variance.
    let vals = w_hat.to_vec::<f32>().unwrap();
    let fan_in = 3 * 3 * 3;
    for o in 0..4 {
        let row = &vals[o * fan_in..(o + 1) * fan_in];
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / fan_in as f64;
        let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / fan_in as f64;
        assert!(mean.abs() < 1e-5, "channel {o} mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "channel {o} var {var}");
    }

    // Shift invariance: adding a constant offset leaves W_hat unchanged up to f32 input rounding.
    let shifted = w.add_scalar(100.0).unwrap();
    let w_hat_shifted = standardize_weight(&shifted, WS_CONV_DEFAULT_EPS).unwrap();
    for (a, b) in w_hat.to_vec::<f32>().unwrap().iter().zip(w_hat_shifted.to_vec::<f32>().unwrap()) {
        assert!((a - b).abs() < 1e-4, "shift invariance: {a} vs {b}");
    }

    // Constant filter (including huge finite 1e38 values that would overflow an unscaled sum) and
    // fan_in == 1 stay finite in both forward and backward.
    let huge_const = Tensor::full([1, 4, 1, 1], 1e38f32);
    let huge_hat = standardize_weight(&huge_const, WS_CONV_DEFAULT_EPS).unwrap();
    assert!(huge_hat.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));

    let const_w = Tensor::full([2, 3, 2, 2], 5.0).requires_grad_();
    let cw_hat = standardize_weight(&const_w, WS_CONV_DEFAULT_EPS).unwrap();
    assert!(cw_hat.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
    let cw_g = cw_hat
        .square()
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&const_w)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!(cw_g.iter().all(|v| v.is_finite()));

    let fan1_w = Tensor::from_slice(&[3.0f32, -2.0], [2, 1, 1, 1]).unwrap().requires_grad_();
    let f1_hat = standardize_weight(&fan1_w, WS_CONV_DEFAULT_EPS).unwrap();
    assert!(f1_hat.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));

    // Full central FD grad_check_with on WeightStandardizedConv2d wrt input x, weight, and bias.
    // Use `mean_all()` so the scalar loss stays O(1) (avoiding f32 cancellation in finite differences).
    let cfg = Conv2dConfig::new(3, 4, 3).padding(1).bias(true);
    let ws_conv = WeightStandardizedConv2d::new(cfg, &mut rng).unwrap();
    assert_eq!(ws_conv.eps(), WS_CONV_DEFAULT_EPS);
    let x = Tensor::randn([2, 3, 5, 5], 0.0, 1.0, &mut rng);
    let weight0 = ws_conv.weight.detach();
    let bias0 = ws_conv.bias.as_ref().unwrap().detach();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 44 };
    grad_check_with(
        |inps| {
            let m = WeightStandardizedConv2d::from_tensors(
                cfg,
                inps[1].clone(),
                Some(inps[2].clone()),
                WS_CONV_DEFAULT_EPS,
            )
            .unwrap();
            m.forward(&inps[0]).unwrap().square()?.mean_all()
        },
        &[x, weight0, bias0],
        &gc,
    )
    .unwrap();

    // Zero-batch and validation checks.
    assert_eq!(ws_conv.forward(&Tensor::zeros([0, 3, 5, 7])).unwrap().dims(), &[0, 4, 5, 7]);
    assert!(standardize_weight(&Tensor::ones([4]), WS_CONV_DEFAULT_EPS).is_err());
    assert!(standardize_weight(&w, 0.0).is_err());
    assert!(standardize_weight(&w, f32::NAN).is_err());
    assert!(WeightStandardizedConv2d::with_eps(cfg, -1e-5, &mut rng).is_err());
    assert!(WeightStandardizedConv2d::new(Conv2dConfig::new(1, 1, usize::MAX), &mut rng).is_err());
}

/// Clean parameter naming and on-disk `.folo` weights save/load round-trip across all four new blocks.
#[test]
fn attention_and_ws_conv_folo_state_dict_round_trip() {
    let build = |seed: u64| {
        let mut rng = Rng::seed(seed);
        Sequential::new()
            .push_named(
                "ws_conv",
                WeightStandardizedConv2d::new(Conv2dConfig::new(4, 4, 3).padding(1).bias(true), &mut rng)
                    .unwrap(),
            )
            .push_named("se", SqueezeExcitation::new(4, 2, &mut rng).unwrap())
            .push_named("eca", Eca::with_kernel_size(4, 3, &mut rng).unwrap())
            .push_named("cbam", Cbam::new(4, 2, 3, &mut rng).unwrap())
    };

    let src = build(10);
    let names: Vec<String> = src.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        vec![
            "ws_conv.weight",
            "ws_conv.bias",
            "se.fc1.weight",
            "se.fc1.bias",
            "se.fc2.weight",
            "se.fc2.bias",
            "eca.weight",
            "cbam.channel.fc1.weight",
            "cbam.channel.fc2.weight",
            "cbam.spatial.conv.weight",
        ]
    );

    let dir = TestDir::new("folo-nn-attn-roundtrip");
    let path = dir.path().join("attn_blocks.folo");
    let meta =
        Metadata::new(FileKind::Weights, ModelConfig::new("n", 1).unwrap(), vec!["obj".into()], 64).unwrap();
    save_weights(&path, &meta, &src).unwrap();

    let dst = build(99);
    let file = ModelFile::open(&path).unwrap();
    let report = file.load(&dst, &LoadOptions::strict(Purpose::Training)).unwrap();
    assert_eq!(report.loaded.len(), names.len());
    assert!(report.skipped.is_empty());

    // Forward output must be bit-for-bit identical after `.folo` reload.
    let x = Tensor::randn([2, 4, 5, 7], 0.0, 1.0, &mut Rng::seed(55));
    let y_src = src.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let y_dst = dst.forward(&x).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(y_src, y_dst);
}
