//! Deterministic unit tests for [`LoraLinear`], [`LoraConv2d`], [`GeM2d`] / [`gem_pool2d`], and
//! `.folo` state-dict round-tripping (including [`Eca`]).

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_test_support::TestDir;

use super::{
    Conv2d, Conv2dConfig, Eca, GeM2d, GeM2dConfig, Linear, LoraConfig, LoraConv2d, LoraLinear, Sequential,
    gem_pool2d,
};
use crate::io::{FileKind, LoadOptions, Metadata, ModelConfig, ModelFile, Purpose, save_weights};
use crate::module::{Forward, Module};
use crate::optim::StandardGroups;

#[test]
fn lora_linear_zero_init_frozen_base_merge_unmerge_and_dropout() {
    let mut rng = Rng::seed(11);
    let base = Linear::new(6, 4, true, &mut rng);
    let x = Tensor::randn([3, 5, 6], 0.0, 1.0, &mut rng); // 3-D input [B, T, in]
    let y_base = base.forward(&x).unwrap().to_vec::<f32>().unwrap();

    let cfg = LoraConfig { rank: 2, alpha: 6.0, dropout: 0.25, seed: 77 };
    let mut lora = LoraLinear::new(base, cfg, &mut rng).unwrap();
    assert!(!lora.is_merged());
    assert!((lora.scaling() - 3.0).abs() < 1e-6);

    // Base tensors are frozen buffers; only lora_a and lora_b are trainable parameters.
    assert!(!lora.base().weight.requires_grad());
    assert!(!lora.base().bias.as_ref().unwrap().requires_grad());
    assert!(lora.lora_a.requires_grad());
    assert!(lora.lora_b.requires_grad());
    let pnames: Vec<String> = lora.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(pnames, vec!["lora_a", "lora_b"]);
    let groups = StandardGroups::from_module(&lora);
    assert_eq!(groups.decay.len(), 2);
    assert!(groups.bn.is_empty() && groups.bias.is_empty());

    // Zero-initialized lora_b means initial output matches base exactly.
    lora.eval();
    let y_init = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in y_init.iter().zip(&y_base) {
        assert!((a - b).abs() < 1e-6);
    }

    // Populate non-zero lora_b so adapter contributes meaningfully.
    {
        let _ng = folo_core::no_grad();
        let b_vals = Tensor::randn([4, 2], 0.0, 0.5, &mut rng);
        lora.lora_b.copy_from(&b_vals).unwrap();
    }

    // Seeded dropout is active in train mode, deterministic across reseed, and inactive in eval mode.
    lora.train();
    lora.reseed_dropout(999);
    let y_tr1 = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    lora.reseed_dropout(999);
    let y_tr2 = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(y_tr1, y_tr2);

    lora.eval();
    let y_unmerged = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    assert_ne!(y_tr1, y_unmerged);

    // Merge -> exact forward equivalence + idempotency -> unmerge -> exact restoration.
    let base_before = lora.base().weight.to_vec::<f32>().unwrap();
    lora.merge_weights().unwrap();
    assert!(lora.is_merged());
    lora.merge_weights().unwrap(); // idempotent
    let y_merged = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in y_merged.iter().zip(&y_unmerged) {
        assert!((a - b).abs() < 1e-5, "merged={a} unmerged={b}");
    }

    lora.unmerge_weights().unwrap();
    assert!(!lora.is_merged());
    lora.unmerge_weights().unwrap(); // idempotent
    let base_after = lora.base().weight.to_vec::<f32>().unwrap();
    for (a, b) in base_before.iter().zip(&base_after) {
        assert!((a - b).abs() < 1e-5, "before={a} after={b}");
    }
    let y_restored = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in y_restored.iter().zip(&y_unmerged) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn lora_linear_grad_check_and_validation() {
    let mut rng = Rng::seed(23);
    let base_w = Tensor::randn([3, 4], 0.0, 0.5, &mut rng);
    let base_b = Tensor::randn([3], 0.0, 0.2, &mut rng);
    let lora_a = Tensor::randn([2, 4], 0.0, 0.5, &mut rng);
    let lora_b = Tensor::randn([3, 2], 0.0, 0.5, &mut rng);
    let x = Tensor::randn([2, 4], 0.0, 1.0, &mut rng);

    let cfg = LoraConfig { rank: 2, alpha: 4.0, dropout: 0.0, seed: 0 };
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 5 };
    grad_check_with(
        |inps| {
            let base = Linear::from_tensors(base_w.detach(), Some(base_b.detach())).unwrap();
            let mut m = LoraLinear::new(base, cfg, &mut Rng::seed(0)).unwrap();
            m.lora_a = inps[1].clone();
            m.lora_b = inps[2].clone();
            m.eval();
            m.forward(&inps[0]).unwrap().square()?.sum_all()
        },
        &[x.clone(), lora_a, lora_b],
        &gc,
    )
    .unwrap();

    // Base receives no gradient when calling backward directly on LoraLinear.
    let base = Linear::new(4, 3, true, &mut rng);
    let m = LoraLinear::new(base, cfg, &mut rng).unwrap();
    {
        let _ng = folo_core::no_grad();
        m.lora_b.copy_from(&Tensor::ones([3, 2])).unwrap();
    }
    let grads = m.forward(&x).unwrap().square().unwrap().sum_all().unwrap().backward().unwrap();
    assert!(grads.get(&m.base().weight).is_none());
    assert!(grads.get(&m.lora_a).is_some());
    assert!(grads.get(&m.lora_b).is_some());

    // Invalid configs.
    for bad in [
        LoraConfig { rank: 0, ..Default::default() },
        LoraConfig { alpha: -1.0, ..Default::default() },
        LoraConfig { alpha: f32::NAN, ..Default::default() },
        LoraConfig { dropout: -0.1, ..Default::default() },
        LoraConfig { dropout: 1.0, ..Default::default() },
    ] {
        let base = Linear::new(4, 3, false, &mut rng);
        assert!(LoraLinear::new(base, bad, &mut rng).is_err());
    }
}

#[test]
fn lora_conv2d_merge_unmerge_grad_check_and_validation() {
    let mut rng = Rng::seed(31);
    // Strided, padded, dilated 3x3 conv.
    let conv_cfg = Conv2dConfig::new(3, 5, 3).stride(2).padding(2).dilation(2).bias(true);
    let base = Conv2d::new(conv_cfg, &mut rng);
    let x = Tensor::randn([2, 3, 7, 7], 0.0, 1.0, &mut rng);
    let y_base = base.forward(&x).unwrap().to_vec::<f32>().unwrap();

    let cfg = LoraConfig { rank: 2, alpha: 3.0, dropout: 0.0, seed: 4 };
    let mut lora = LoraConv2d::new(base, cfg, &mut rng).unwrap();
    assert!(!lora.base().weight.requires_grad());
    assert_eq!(
        lora.named_parameters().into_iter().map(|(n, _)| n).collect::<Vec<_>>(),
        vec!["lora_a", "lora_b"]
    );
    lora.eval();

    // Zero-init lora_b matches base exactly.
    let y_init = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in y_init.iter().zip(&y_base) {
        assert!((a - b).abs() < 1e-6);
    }

    // Set non-zero lora_b and check merge/unmerge exact equivalence.
    {
        let _ng = folo_core::no_grad();
        lora.lora_b.copy_from(&Tensor::randn([5, 2, 1, 1], 0.0, 0.4, &mut rng)).unwrap();
    }
    let y_unmerged = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let base_before = lora.base().weight.to_vec::<f32>().unwrap();

    lora.merge_weights().unwrap();
    assert!(lora.is_merged());
    let y_merged = lora.forward(&x).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in y_merged.iter().zip(&y_unmerged) {
        assert!((a - b).abs() < 1e-5, "merged={a} unmerged={b}");
    }

    lora.unmerge_weights().unwrap();
    assert!(!lora.is_merged());
    let base_after = lora.base().weight.to_vec::<f32>().unwrap();
    for (a, b) in base_before.iter().zip(&base_after) {
        assert!((a - b).abs() < 1e-5);
    }

    // Central finite-difference grad check on x, lora_a, lora_b.
    let base_w = lora.base().weight.detach();
    let base_b = lora.base().bias.as_ref().unwrap().detach();
    let la = lora.lora_a.detach();
    let lb = lora.lora_b.detach();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 9 };
    grad_check_with(
        |inps| {
            let b = Conv2d::from_tensors(conv_cfg, base_w.detach(), Some(base_b.detach())).unwrap();
            let mut m = LoraConv2d::new(b, cfg, &mut Rng::seed(0)).unwrap();
            m.lora_a = inps[1].clone();
            m.lora_b = inps[2].clone();
            m.eval();
            m.forward(&inps[0]).unwrap().square()?.sum_all()
        },
        &[x, la, lb],
        &gc,
    )
    .unwrap();

    // Depthwise / grouped conv (groups > 1) must be rejected cleanly.
    let dw = Conv2d::new(Conv2dConfig::new(4, 4, 3).groups(4).padding(1), &mut rng);
    assert!(LoraConv2d::new(dw, cfg, &mut rng).is_err());
}

#[test]
fn gem_pool2d_limits_stability_and_grad_checks() {
    let mut rng = Rng::seed(51);
    let x = Tensor::rand_uniform([2, 3, 4, 5], 0.2, 2.5, &mut rng);

    // p = 1.0 must match exact spatial average pooling.
    let gem_avg = GeM2d::new(GeM2dConfig { p_init: 1.0, p_min: 1.0, p_max: 10.0, eps: 1e-6 }).unwrap();
    let y_avg = gem_avg.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let x_vec = x.to_vec::<f32>().unwrap();
    for ni in 0..2 {
        for ci in 0..3 {
            let base = (ni * 3 + ci) * 20;
            let mean: f32 = x_vec[base..base + 20].iter().sum::<f32>() / 20.0;
            assert!((y_avg[ni * 3 + ci] - mean).abs() < 1e-5);
        }
    }

    // Large p (50.0) on large activations (1e4) must stay finite and approach spatial max pooling.
    let x_large = x.mul_scalar(1e4).unwrap();
    let p_hi = Tensor::from_slice(&[50.0f32], [1]).unwrap();
    let y_hi = gem_pool2d(&x_large, &p_hi, 1.0, 100.0, 1e-6).unwrap().to_vec::<f32>().unwrap();
    let xl_vec = x_large.to_vec::<f32>().unwrap();
    for ni in 0..2 {
        for ci in 0..3 {
            let base = (ni * 3 + ci) * 20;
            let mx = xl_vec[base..base + 20].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(y_hi[ni * 3 + ci].is_finite());
            // (1/20)^(1/50) ~= 0.9418, so GeM(p=50) is within ~6% of exact max.
            assert!(y_hi[ni * 3 + ci] <= mx * 1.0001 && y_hi[ni * 3 + ci] >= mx * 0.93);
        }
    }

    // -Inf entries clamp safely to eps with zero gradient.
    let x_neg =
        Tensor::from_slice(&[f32::NEG_INFINITY, 2.0, -5.0, 3.0], [1, 1, 2, 2]).unwrap().requires_grad_();
    let p_in = Tensor::from_slice(&[2.0f32], [1]).unwrap().requires_grad_();
    let y_neg = gem_pool2d(&x_neg, &p_in, 1.0, 10.0, 1e-4).unwrap();
    assert!(y_neg.to_vec::<f32>().unwrap()[0].is_finite());
    let g_neg = y_neg.sum_all().unwrap().backward().unwrap();
    let gx_neg = g_neg.get(&x_neg).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(gx_neg[0], 0.0);
    assert_eq!(gx_neg[2], 0.0);
    assert!(gx_neg[1] > 0.0 && gx_neg[3] > 0.0);

    // Clamped p (outside [p_min, p_max]) produces zero gradient into p.
    let p_clamped = Tensor::from_slice(&[0.5f32, 12.0, 2.5], [3]).unwrap().requires_grad_();
    let y_clamped = gem_pool2d(&x, &p_clamped, 1.0, 10.0, 1e-6).unwrap();
    let gp =
        y_clamped.sum_all().unwrap().backward().unwrap().get(&p_clamped).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(gp[0], 0.0);
    assert_eq!(gp[1], 0.0);
    assert!(gp[2].abs() > 1e-4);

    // Central finite-difference grad check for shared p ([1]) AND per-channel p ([C]).
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 7 };
    let p_shared = Tensor::from_slice(&[3.0f32], [1]).unwrap();
    grad_check_with(
        |inps| gem_pool2d(&inps[0], &inps[1], 1.0, 10.0, 1e-6).unwrap().square()?.sum_all(),
        &[x.clone(), p_shared],
        &gc,
    )
    .unwrap();

    let p_chan = Tensor::from_slice(&[1.8f32, 3.2, 5.5], [3]).unwrap();
    grad_check_with(
        |inps| gem_pool2d(&inps[0], &inps[1], 1.0, 10.0, 1e-6).unwrap().square()?.sum_all(),
        &[x.clone(), p_chan],
        &gc,
    )
    .unwrap();

    // Zero-batch and validation.
    let zero_x = Tensor::zeros([0, 3, 4, 4]).requires_grad_();
    let gem_pc = GeM2d::per_channel(3, GeM2dConfig::default()).unwrap();
    let y0 = gem_pc.forward(&zero_x).unwrap();
    assert_eq!(y0.dims(), &[0, 3, 1, 1]);
    let g0 = y0.sum_all().unwrap().backward().unwrap();
    assert_eq!(g0.get(&zero_x).unwrap().dims(), &[0, 3, 4, 4]);
    assert_eq!(g0.get(&gem_pc.p).unwrap().to_vec::<f32>().unwrap(), vec![0.0; 3]);

    assert!(GeM2d::new(GeM2dConfig { p_min: 0.0, ..Default::default() }).is_err());
    assert!(GeM2d::new(GeM2dConfig { p_min: 5.0, p_max: 2.0, ..Default::default() }).is_err());
    assert!(GeM2d::new(GeM2dConfig { eps: 0.0, ..Default::default() }).is_err());
    assert!(GeM2d::per_channel(0, GeM2dConfig::default()).is_err());
}

#[test]
fn lora_gem_and_eca_folo_state_dict_round_trip() {
    let build_conv_pipe = |seed: u64| {
        let mut rng = Rng::seed(seed);
        let conv = Conv2d::new(Conv2dConfig::new(4, 4, 3).padding(1).bias(true), &mut rng);
        let lora_c =
            LoraConv2d::new(conv, LoraConfig { rank: 2, alpha: 4.0, dropout: 0.0, seed: 1 }, &mut rng)
                .unwrap();
        {
            let _ng = folo_core::no_grad();
            lora_c.lora_b.copy_from(&Tensor::randn([4, 2, 1, 1], 0.0, 0.3, &mut rng)).unwrap();
        }
        let eca = Eca::with_kernel_size(4, 3, &mut rng).unwrap();
        let gem = GeM2d::per_channel(4, GeM2dConfig { p_init: 2.5, ..Default::default() }).unwrap();
        {
            let _ng = folo_core::no_grad();
            gem.p.copy_from(&Tensor::from_slice(&[1.5f32, 2.5, 3.5, 4.5], [4]).unwrap()).unwrap();
        }
        Sequential::new().push_named("lora_conv", lora_c).push_named("eca", eca).push_named("gem", gem)
    };

    let build_lin = |seed: u64| {
        let mut rng = Rng::seed(seed);
        let lin = Linear::new(4, 3, true, &mut rng);
        let lora_l =
            LoraLinear::new(lin, LoraConfig { rank: 2, alpha: 6.0, dropout: 0.0, seed: 2 }, &mut rng)
                .unwrap();
        {
            let _ng = folo_core::no_grad();
            lora_l.lora_b.copy_from(&Tensor::randn([3, 2], 0.0, 0.3, &mut rng)).unwrap();
        }
        Sequential::new().push_named("lora_lin", lora_l)
    };

    let dir = TestDir::new("folo-nn-lora-gem-eca-roundtrip");
    let meta =
        Metadata::new(FileKind::Weights, ModelConfig::new("n", 1).unwrap(), vec!["obj".into()], 64).unwrap();

    // 1. LoraConv2d + Eca + GeM2d round-trip.
    let src_conv = build_conv_pipe(10);
    src_conv.eval();
    let p_conv = dir.path().join("conv_eca_gem.folo");
    save_weights(&p_conv, &meta, &src_conv).unwrap();

    let dst_conv = build_conv_pipe(99);
    dst_conv.eval();
    let report =
        ModelFile::open(&p_conv).unwrap().load(&dst_conv, &LoadOptions::strict(Purpose::Training)).unwrap();
    assert_eq!(report.loaded.len(), src_conv.named_tensors().len());
    assert!(report.skipped.is_empty() && report.missing.is_empty());

    let x_img = Tensor::rand_uniform([2, 4, 5, 5], 0.1, 2.0, &mut Rng::seed(7));
    assert_eq!(
        src_conv.forward(&x_img).unwrap().to_vec::<f32>().unwrap(),
        dst_conv.forward(&x_img).unwrap().to_vec::<f32>().unwrap()
    );

    // 2. LoraLinear round-trip.
    let src_lin = build_lin(20);
    src_lin.eval();
    let p_lin = dir.path().join("lora_lin.folo");
    save_weights(&p_lin, &meta, &src_lin).unwrap();

    let dst_lin = build_lin(88);
    dst_lin.eval();
    let report_l =
        ModelFile::open(&p_lin).unwrap().load(&dst_lin, &LoadOptions::strict(Purpose::Training)).unwrap();
    assert_eq!(report_l.loaded.len(), src_lin.named_tensors().len());
    assert!(report_l.skipped.is_empty() && report_l.missing.is_empty());

    let x_vec = Tensor::randn([3, 4], 0.0, 1.0, &mut Rng::seed(8));
    assert_eq!(
        src_lin.forward(&x_vec).unwrap().to_vec::<f32>().unwrap(),
        dst_lin.forward(&x_vec).unwrap().to_vec::<f32>().unwrap()
    );
}
