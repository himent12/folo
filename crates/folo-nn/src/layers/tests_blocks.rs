//! Unit tests for modern vision blocks: [`ConvNeXtBlock`] (+ [`LayerNorm`]), [`InvertedResidual`]
//! (MBConv), and [`DropBlock2d`].
//!
//! Covers central finite-difference `grad_check_with` on inputs AND trainable parameters, train/eval parity,
//! `StandardGroups` no-decay / depthwise routing, BatchNorm fusion equivalence, zero-batch / odd-spatial
//! edge cases, validation, and on-disk `.folo` save/load round-trips via `TestDir`.

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_test_support::TestDir;

use super::{
    Activation, Conv2d, ConvNeXtBlock, ConvNeXtConfig, ConvNeXtNorm, ConvNeXtNormLayer, DropBlock2d,
    InvertedResidual, InvertedResidualConfig, LayerNorm, LayerScale, Linear, Sequential, dropblock_gamma,
};
use crate::io::{FileKind, LoadOptions, Metadata, ModelConfig, ModelFile, Purpose, save_weights};
use crate::module::{Forward, Module};
use crate::optim::StandardGroups;

/// Central finite-difference `grad_check_with` on [`ConvNeXtBlock`] (`LayerNorm` and `RmsNorm` variants)
/// w.r.t. input `x` and all trainable parameters (`dwconv`, `norm`, `pwconv1`, `pwconv2`, `layer_scale`),
/// plus `StandardGroups` routing and zero-batch / validation checks.
#[test]
fn convnext_block_grad_check_standard_groups_and_edge_cases() {
    let mut rng = Rng::seed(501);
    let cfg = ConvNeXtConfig::new(4)
        .kernel_size(3)
        .expand_ratio(2)
        .norm(ConvNeXtNorm::LayerNorm)
        .layer_scale_init(Some(0.5))
        .drop_path(0.0, 0);
    let blk = ConvNeXtBlock::new(cfg, &mut rng).unwrap();
    assert_eq!((blk.dim(), blk.hidden_dim()), (4, 8));

    // StandardGroups routes `norm.weight` and `layer_scale.gamma` into `bn` (no decay), biases into `bias`,
    // and marks `dwconv.weight` as depthwise inside `decay`.
    let sg = StandardGroups::from_module(&blk);
    let bn_names: Vec<&str> = sg.bn.iter().map(|(n, _)| n.as_str()).collect();
    let bias_names: Vec<&str> = sg.bias.iter().map(|(n, _)| n.as_str()).collect();
    let decay_names: Vec<&str> = sg.decay.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(bn_names, vec!["norm.weight", "layer_scale.gamma"]);
    assert_eq!(bias_names, vec!["dwconv.bias", "norm.bias", "pwconv1.bias", "pwconv2.bias"]);
    assert_eq!(decay_names, vec!["dwconv.weight", "pwconv1.weight", "pwconv2.weight"]);
    assert!(sg.depthwise.contains("dwconv.weight"));
    assert_eq!(sg.depthwise.len(), 1);

    // Full central finite-difference gradient check on input x and all trainable parameters.
    let x = Tensor::randn([2, 4, 3, 5], 0.0, 0.8, &mut rng);
    let dw_cfg = *blk.dwconv.config();
    let ConvNeXtNormLayer::Layer(ln) = &blk.norm else { unreachable!() };
    let inps = vec![
        x.clone(),
        blk.dwconv.weight.detach(),
        blk.dwconv.bias.as_ref().unwrap().detach(),
        ln.weight.as_ref().unwrap().detach(),
        ln.bias.as_ref().unwrap().detach(),
        blk.pwconv1.weight.detach(),
        blk.pwconv1.bias.as_ref().unwrap().detach(),
        blk.pwconv2.weight.detach(),
        blk.pwconv2.bias.as_ref().unwrap().detach(),
        blk.layer_scale.as_ref().unwrap().gamma.detach(),
    ];
    let gc = GradCheck { eps: 2e-3, atol: 3e-3, rtol: 3e-2, max_checks: 10, seed: 51 };
    grad_check_with(
        |t| {
            let mut m = ConvNeXtBlock::new(cfg, &mut Rng::seed(0)).unwrap();
            m.dwconv = Conv2d::from_tensors(dw_cfg, t[1].clone(), Some(t[2].clone())).unwrap();
            m.norm = ConvNeXtNormLayer::Layer(
                LayerNorm::from_tensors(Some(t[3].clone()), Some(t[4].clone()), 4, cfg.norm_eps).unwrap(),
            );
            m.pwconv1 = Linear::from_tensors(t[5].clone(), Some(t[6].clone())).unwrap();
            m.pwconv2 = Linear::from_tensors(t[7].clone(), Some(t[8].clone())).unwrap();
            m.layer_scale = Some(LayerScale::channels_last(4, 0.5).unwrap());
            m.layer_scale.as_mut().unwrap().gamma = t[9].clone();
            m.forward(&t[0]).unwrap().square()?.mean_all()
        },
        &inps,
        &gc,
    )
    .unwrap();

    // RmsNorm + DropPath train/eval behavior + zero-batch.
    let cfg_rms = ConvNeXtConfig::new(4)
        .kernel_size(5)
        .expand_ratio(2)
        .norm(ConvNeXtNorm::RmsNorm)
        .layer_scale_init(None)
        .drop_path(0.5, 99);
    let blk_rms = ConvNeXtBlock::new(cfg_rms, &mut rng).unwrap();
    let sg_rms = StandardGroups::from_module(&blk_rms);
    assert_eq!(sg_rms.bn.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["norm.weight"]);
    grad_check_with(
        |t| {
            blk_rms.drop_path.as_ref().unwrap().reseed(99);
            blk_rms.forward(&t[0]).unwrap().square()?.mean_all()
        },
        std::slice::from_ref(&x),
        &gc,
    )
    .unwrap();
    blk_rms.eval();
    let y_eval_1 = blk_rms.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let y_eval_2 = blk_rms.forward(&x).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(y_eval_1, y_eval_2);

    // Zero-batch keeps autograd wired.
    let z0 = Tensor::zeros([0, 4, 3, 5]).requires_grad_();
    let y0 = blk.forward(&z0).unwrap();
    assert_eq!(y0.dims(), &[0, 4, 3, 5]);
    assert_eq!(y0.sum_all().unwrap().backward().unwrap().get(&z0).unwrap().dims(), &[0, 4, 3, 5]);

    // Validation.
    assert!(ConvNeXtBlock::new(ConvNeXtConfig::new(0), &mut rng).is_err());
    assert!(ConvNeXtBlock::new(ConvNeXtConfig::new(4).kernel_size(4), &mut rng).is_err());
    assert!(ConvNeXtBlock::new(ConvNeXtConfig::new(4).expand_ratio(0), &mut rng).is_err());
    assert!(ConvNeXtBlock::new(ConvNeXtConfig::new(4).drop_path(1.0, 0), &mut rng).is_err());
    assert!(blk.forward(&Tensor::zeros([1, 3, 4, 4])).is_err());
    assert!(blk.forward(&Tensor::zeros([1, 4, 0, 4])).is_err());
}

/// Central finite-difference `grad_check_with` on [`InvertedResidual`] (MBConv with expansion + SE +
/// residual AND downsample stride=2 without residual), BatchNorm fusion equivalence, and edge cases.
#[test]
fn inverted_residual_mbconv_grad_check_fusion_and_edge_cases() {
    let mut rng = Rng::seed(601);

    // 1. Residual MBConv with expansion=2, SE reduction=2, SiLU.
    let cfg_res = InvertedResidualConfig::new(4, 4)
        .stride(1)
        .expand_ratio(2)
        .kernel_size(3)
        .se_reduction(Some(2))
        .activation(Activation::Silu);
    let mb = InvertedResidual::new(cfg_res, &mut rng).unwrap();
    assert!(mb.use_residual());
    assert!(mb.expand.is_some());
    assert!(mb.se.is_some());
    assert_eq!((mb.in_channels(), mb.out_channels(), mb.hidden_channels()), (4, 4, 8));

    // StandardGroups marks dwconv.conv.weight as depthwise and routes all BN weights to `bn`.
    let sg = StandardGroups::from_module(&mb);
    assert!(sg.depthwise.contains("dwconv.conv.weight"));
    assert_eq!(sg.bn.len(), 3); // expand.bn.weight, dwconv.bn.weight, project.bn.weight

    // Run one training step to populate non-trivial BN running stats, then grad_check in eval mode w.r.t.
    // input AND key weights (expand, dwconv, SE, project) plus input grad_check in train mode.
    let x = Tensor::randn([2, 4, 5, 5], 0.0, 0.9, &mut rng);
    let _ = mb.forward(&x).unwrap();
    let gc = GradCheck { eps: 2e-3, atol: 3e-3, rtol: 3e-2, max_checks: 10, seed: 61 };
    grad_check_with(|t| mb.forward(&t[0]).unwrap().square()?.mean_all(), std::slice::from_ref(&x), &gc)
        .unwrap();

    // Eval mode: grad_check wrt input + fusion equivalence on the exact same BN running stats.
    let mut mb = mb;
    mb.eval();
    let y_unfused = mb.forward(&x).unwrap().to_vec::<f32>().unwrap();
    grad_check_with(|t| mb.forward(&t[0]).unwrap().square()?.mean_all(), std::slice::from_ref(&x), &gc)
        .unwrap();
    mb.fuse_().unwrap();
    let y_fused = mb.forward(&x).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in y_unfused.iter().zip(&y_fused) {
        assert!((a - b).abs() < 1e-5, "MBConv fuse mismatch: {a} vs {b}");
    }
    // Verify grad_check wrt trainable weights (SE + fused projection conv) as well.
    let se_w1 = mb.se.as_ref().unwrap().fc1.weight.detach();
    let se_b1 = mb.se.as_ref().unwrap().fc1.bias.as_ref().unwrap().detach();
    let se_fc1_cfg = *mb.se.as_ref().unwrap().fc1.config();
    let proj_w = mb.project.conv.weight.detach();
    let proj_b = mb.project.conv.bias.as_ref().unwrap().detach();
    let proj_cfg = *mb.project.conv.config();
    grad_check_with(
        |t| {
            let mut m = InvertedResidual::new(cfg_res, &mut Rng::seed(601)).unwrap();
            m.eval();
            m.fuse_().unwrap();
            m.se.as_mut().unwrap().fc1 =
                Conv2d::from_tensors(se_fc1_cfg, t[1].clone(), Some(se_b1.clone())).unwrap();
            m.project.conv = Conv2d::from_tensors(proj_cfg, t[2].clone(), Some(t[3].clone())).unwrap();
            m.forward(&t[0]).unwrap().square()?.mean_all()
        },
        &[x.clone(), se_w1, proj_w, proj_b],
        &gc,
    )
    .unwrap();

    // 2. Non-residual stride=2, expand_ratio=1 (no expand stage), kernel_size=5.
    let cfg_down =
        InvertedResidualConfig::new(4, 6).stride(2).expand_ratio(1).kernel_size(5).se_reduction(None);
    let mb_down = InvertedResidual::new(cfg_down, &mut rng).unwrap();
    assert!(!mb_down.use_residual());
    assert!(mb_down.expand.is_none());
    let y_down = mb_down.forward(&x).unwrap();
    assert_eq!(y_down.dims(), &[2, 6, 3, 3]);
    grad_check_with(|t| mb_down.forward(&t[0]).unwrap().square()?.mean_all(), std::slice::from_ref(&x), &gc)
        .unwrap();

    // Zero-batch keeps autograd wired with exact downsampled spatial shape.
    let z0 = Tensor::zeros([0, 4, 5, 5]).requires_grad_();
    let y0 = mb_down.forward(&z0).unwrap();
    assert_eq!(y0.dims(), &[0, 6, 3, 3]);
    assert_eq!(y0.sum_all().unwrap().backward().unwrap().get(&z0).unwrap().dims(), &[0, 4, 5, 5]);

    // Validation.
    assert!(InvertedResidual::new(InvertedResidualConfig::new(0, 4), &mut rng).is_err());
    assert!(InvertedResidual::new(InvertedResidualConfig::new(4, 4).stride(0), &mut rng).is_err());
    assert!(InvertedResidual::new(InvertedResidualConfig::new(4, 4).expand_ratio(0), &mut rng).is_err());
    assert!(InvertedResidual::new(InvertedResidualConfig::new(4, 4).kernel_size(4), &mut rng).is_err());
    assert!(
        InvertedResidual::new(InvertedResidualConfig::new(4, 4).se_reduction(Some(0)), &mut rng).is_err()
    );
    assert!(mb.forward(&Tensor::zeros([1, 3, 5, 5])).is_err());
}

/// [`DropBlock2d`]: exact paper `gamma` formula, deterministic seeded sampling, spatial block structure,
/// exact mask-count normalization, train/eval parity, central FD `grad_check_with`, oversized `block_size`,
/// and validation.
#[test]
fn dropblock2d_gamma_structure_normalization_grad_check_and_edge_cases() {
    // Paper Eq. 1 exact check: drop_prob=0.1, block_size=3, H=W=7 -> valid=5x5 -> gamma = (0.1/9)*(49/25).
    let g = dropblock_gamma(0.1, 3, 7, 7).unwrap();
    let want = ((0.1f32 as f64) / 9.0) * (49.0 / 25.0);
    assert!((g - want).abs() < 1e-12, "gamma {g} vs {want}");
    // block_size == 1 reduces to exact elementwise dropout rate.
    assert!((dropblock_gamma(0.25, 1, 7, 9).unwrap() - 0.25).abs() < 1e-12);
    // block_size > min(H, W) clamps cleanly without division by zero.
    let g_big = dropblock_gamma(0.2, 7, 5, 5).unwrap();
    assert!((0.0..=1.0).contains(&g_big));

    let db = DropBlock2d::try_new(0.2, 3, 777).unwrap();
    assert_eq!((db.drop_prob(), db.block_size()), (0.2, 3));

    // Exact normalization invariant on all-ones input: mean(DropBlock(1)) == 1.0 (whenever kept > 0).
    let ones = Tensor::ones([4, 8, 9, 9]);
    let y = db.forward(&ones).unwrap();
    let yv = y.to_vec::<f32>().unwrap();
    let zeros = yv.iter().filter(|&&v| v == 0.0).count();
    let kept = yv.len() - zeros;
    assert!(zeros > 0 && kept > 0, "expected partial dropping: zeros={zeros}, kept={kept}");
    let mean: f64 = yv.iter().map(|&v| v as f64).sum::<f64>() / yv.len() as f64;
    assert!((mean - 1.0).abs() < 1e-5, "exact mask normalization failed: mean={mean}");

    // Determinism via reseed & RNG state snapshot/restore.
    let st = db.rng_state();
    let y_a = db.forward(&ones).unwrap().to_vec::<f32>().unwrap();
    db.set_rng_state(st).unwrap();
    let y_b = db.forward(&ones).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(y_a, y_b);

    // Central finite-difference grad_check_with (reseeding inside the closure fixes the mask).
    let mut rng = Rng::seed(707);
    let x = Tensor::randn([2, 3, 6, 7], 0.0, 1.0, &mut rng);
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 71 };
    grad_check_with(
        |t| {
            db.reseed(777);
            db.forward(&t[0]).unwrap().square()?.sum_all()
        },
        std::slice::from_ref(&x),
        &gc,
    )
    .unwrap();

    // Eval mode is exact bit-for-bit identity (and does not advance RNG).
    db.eval();
    let st_eval = db.rng_state();
    assert_eq!(db.forward(&x).unwrap().to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
    assert_eq!(db.rng_state(), st_eval);

    // Oversized block_size on small spatial map [2, 2, 3, 3] with block_size = 5 works cleanly.
    let db_big = DropBlock2d::try_new(0.25, 5, 11).unwrap();
    let small_out = db_big.forward(&Tensor::ones([2, 2, 3, 3])).unwrap();
    assert_eq!(small_out.dims(), &[2, 2, 3, 3]);
    assert!(small_out.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));

    // Zero-batch and validation.
    assert_eq!(db_big.forward(&Tensor::zeros([0, 2, 4, 4])).unwrap().dims(), &[0, 2, 4, 4]);
    assert!(DropBlock2d::try_new(-0.1, 3, 0).is_err());
    assert!(DropBlock2d::try_new(1.0, 3, 0).is_err());
    assert!(DropBlock2d::try_new(0.1, 0, 0).is_err());
    assert!(DropBlock2d::try_new(0.1, 4, 0).is_err()); // even block_size rejected
    assert!(db_big.forward(&Tensor::zeros([2, 3, 4])).is_err());
}

/// On-disk `.folo` weight save/load round-trip across [`ConvNeXtBlock`], [`InvertedResidual`] (MBConv + SE),
/// and [`DropBlock2d`] in a [`Sequential`] pipeline, verifying exact forward parity after reload.
#[test]
fn vision_blocks_folo_state_dict_round_trip() {
    let build = |seed: u64| {
        let mut rng = Rng::seed(seed);
        Sequential::new()
            .push_named(
                "convnext",
                ConvNeXtBlock::new(
                    ConvNeXtConfig::new(4).kernel_size(5).expand_ratio(2).layer_scale_init(Some(0.25)),
                    &mut rng,
                )
                .unwrap(),
            )
            .push_named("dropblock", DropBlock2d::try_new(0.15, 3, 42).unwrap())
            .push_named(
                "mbconv",
                InvertedResidual::new(
                    InvertedResidualConfig::new(4, 4)
                        .stride(1)
                        .expand_ratio(2)
                        .kernel_size(3)
                        .se_reduction(Some(2)),
                    &mut rng,
                )
                .unwrap(),
            )
    };

    let src = build(11);
    let x = Tensor::randn([2, 4, 7, 7], 0.0, 1.0, &mut Rng::seed(22));
    // Run one train forward to make BN running stats non-default, then switch both to eval.
    let _ = src.forward(&x).unwrap();
    src.eval();

    let dir = TestDir::new("folo-nn-blocks-roundtrip");
    let path = dir.path().join("vision_blocks.folo");
    let meta =
        Metadata::new(FileKind::Weights, ModelConfig::new("n", 1).unwrap(), vec!["obj".into()], 64).unwrap();
    save_weights(&path, &meta, &src).unwrap();

    let dst = build(99);
    dst.eval();
    let file = ModelFile::open(&path).unwrap();
    let report = file.load(&dst, &LoadOptions::strict(Purpose::Training)).unwrap();
    assert!(report.skipped.is_empty());
    assert!(report.missing.is_empty());
    assert_eq!(report.loaded.len(), src.named_tensors().len());

    let y_src = src.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let y_dst = dst.forward(&x).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(y_src, y_dst);
}
