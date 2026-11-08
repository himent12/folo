//! Layers and the Module trait: names, modes, gradient checks, hand-computed BatchNorm values, fusion.

use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_core::{Tensor, no_grad};
use folo_nn::*;

fn close(a: &[f32], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x as f64 - y).abs() <= tol * (1.0 + y.abs()), "element {i}: {x} vs {y}");
    }
}

#[test]
fn names_order_counts_and_modes() {
    let mut rng = Rng::seed(0);
    let stem = ConvBnAct::new(3, 16, 3, 2, 1, Activation::Silu, &mut rng);
    // docs/MODEL.md §10.1: Conv 3→16, k3 s2 has 464 parameters and maps 640×640 to 320×320.
    assert_eq!(stem.num_parameters(), 464);
    stem.eval();
    let y = no_grad_fwd(&stem, &Tensor::zeros([1, 3, 640, 640]));
    assert_eq!(y.dims(), &[1, 16, 320, 320]);

    let model = Sequential::new()
        .push_named("stem", stem)
        .push_named(
            "blocks",
            ModuleList::new(vec![ConvBnAct::new(16, 16, 3, 1, 4, Activation::Silu, &mut rng)]),
        )
        .push_named("out", Conv2d::new(Conv2dConfig::new(16, 4, 1), &mut rng));
    let names: Vec<_> = model.named_tensors().into_iter().map(|t| (t.name, t.kind)).collect();
    let want = [
        "stem.conv.weight",
        "stem.bn.weight",
        "stem.bn.bias",
        "stem.bn.running_mean",
        "stem.bn.running_var",
        "blocks.0.conv.weight",
        "blocks.0.bn.weight",
        "blocks.0.bn.bias",
        "blocks.0.bn.running_mean",
        "blocks.0.bn.running_var",
        "out.weight",
        "out.bias",
    ];
    assert_eq!(names.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), want);
    assert_eq!(model.named_buffers().len(), 4);
    assert!(names.iter().filter(|(n, _)| n.contains("running")).all(|(_, k)| *k == TensorKind::Buffer));
    assert_eq!(model.named_parameters()[3].1.dims(), &[16, 4, 3, 3]); // grouped conv weight

    // Mode propagation reaches every BatchNorm; the visitor sees every submodule with its path.
    assert!(!model.is_training(), "`stem` was put in eval mode above");
    model.train();
    assert!(model.is_training());
    model.eval();
    let mut bns = Vec::new();
    model.for_each_module(&mut |path, m| {
        if let Some(bn) = (m as &dyn std::any::Any).downcast_ref::<BatchNorm2d>() {
            bns.push((path.to_string(), bn.mode().unwrap().is_training()));
        }
    });
    assert_eq!(bns, [("stem.bn".to_string(), false), ("blocks.0.bn".to_string(), false)]);
    model.train();
    assert!(model.is_training());
    // Determinism: the same seed gives the same weights.
    let mut r2 = Rng::seed(0);
    let again = ConvBnAct::new(3, 16, 3, 2, 1, Activation::Silu, &mut r2);
    let first = model.named_parameters()[0].1.to_vec::<f32>().unwrap();
    assert_eq!(again.conv.weight.to_vec::<f32>().unwrap(), first);
    let b = 1.0 / 27f32.sqrt();
    assert!(first.iter().all(|v| v.abs() <= b));
}

fn no_grad_fwd(m: &dyn Layer, x: &Tensor) -> Tensor {
    let _g = no_grad();
    m.forward(x).unwrap()
}

#[test]
fn simple_layers() {
    let x = Tensor::from_vec((0..32).map(|v| v as f32).collect(), [1, 2, 4, 4]).unwrap();
    let up = Upsample::nearest(2).forward(&x).unwrap();
    assert_eq!(up.dims(), &[1, 2, 8, 8]);
    assert_eq!(up.to_vec::<f32>().unwrap()[..4], [0., 0., 1., 1.]);
    let mp = MaxPool2d::new(5, 1, 2).forward(&x).unwrap();
    assert_eq!(mp.dims(), &[1, 2, 4, 4]);
    assert_eq!(mp.to_vec::<f32>().unwrap()[0], 10.0); // padding never wins
    let c = Concat::channels().forward(&[&x, &mp]).unwrap();
    assert_eq!(c.dims(), &[1, 4, 4, 4]);
    assert_eq!(Identity.forward(&x).unwrap().to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
    let a = Tensor::from_slice(&[-1.0f32, 2.0], [2]).unwrap();
    assert_eq!(Activation::LeakyRelu(0.1).forward(&a).unwrap().to_vec::<f32>().unwrap(), [-0.1, 2.0]);
    assert_eq!(Activation::Relu.forward(&a).unwrap().to_vec::<f32>().unwrap(), [0.0, 2.0]);
    let s = Activation::Silu.forward(&a).unwrap().to_vec::<f32>().unwrap();
    close(&s, &[-1.0 / (1.0 + 1f64.exp()), 2.0 / (1.0 + (-2f64).exp())], 1e-6);
    let mut rng = Rng::seed(3);
    let lin = Linear::new(4, 3, true, &mut rng);
    let y = lin.forward(&Tensor::ones([2, 5, 4])).unwrap();
    assert_eq!(y.dims(), &[2, 5, 3]);
    let w = lin.weight.to_vec2::<f32>().unwrap();
    let b = lin.bias.as_ref().unwrap().to_vec::<f32>().unwrap();
    let want: Vec<f64> = (0..3).map(|o| w[o].iter().map(|&v| v as f64).sum::<f64>() + b[o] as f64).collect();
    close(&y.to_vec::<f32>().unwrap()[..3], &want, 1e-6);
    assert!(Conv2d::try_new(Conv2dConfig::new(3, 4, 3).groups(2), &mut rng).is_err());
}

#[test]
fn conv2d_grad_check_groups_stride_dilation() {
    let mut rng = Rng::seed(1);
    let opts = GradCheck { max_checks: 24, ..Default::default() };
    for cfg in [
        Conv2dConfig::new(4, 6, 3).stride(2).groups(2),
        Conv2dConfig::new(4, 4, 3).groups(4).padding(2).dilation(2),
        Conv2dConfig::new(3, 2, 1).bias(false),
    ] {
        let conv = Conv2d::new(cfg, &mut rng);
        let x = Tensor::randn([2, cfg.in_channels, 6, 6], 0.0, 1.0, &mut rng);
        let mut inputs = vec![x, conv.weight.detach()];
        if let Some(b) = &conv.bias {
            inputs.push(b.detach());
        }
        grad_check_with(
            |v| {
                let m = Conv2d::from_tensors(cfg, v[1].clone(), v.get(2).cloned()).map_err(core)?;
                m.forward(&v[0]).map_err(core)
            },
            &inputs,
            &opts,
        )
        .unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
    }
}

fn core(e: Error) -> folo_core::Error {
    match e {
        Error::Core(c) => c,
        other => folo_core::Error::InvalidArgument { op: "test", msg: other.to_string() },
    }
}

#[test]
fn batchnorm_and_conv_block_grad_check_in_train_mode() {
    let mut rng = Rng::seed(2);
    let x = Tensor::randn([2, 3, 4, 4], 0.0, 1.0, &mut rng);
    let g = Tensor::rand_uniform([3], 0.5, 1.5, &mut rng);
    let b = Tensor::rand_uniform([3], -0.5, 0.5, &mut rng);
    let opts = GradCheck { max_checks: 32, ..Default::default() };
    grad_check_with(
        |v| {
            let bn = BatchNorm2d::from_tensors(
                v[1].clone(),
                v[2].clone(),
                Tensor::zeros([3]),
                Tensor::ones([3]),
                BN_EPS,
                BN_MOMENTUM,
            )
            .map_err(core)?;
            bn.forward(&v[0]).map_err(core)
        },
        &[x.clone(), g, b],
        &opts,
    )
    .unwrap();
    // The whole Conv block (grouped, strided conv + BN in train mode + SiLU), MODEL.md §10.1's tiny size.
    let x = Tensor::randn([2, 8, 6, 6], 0.0, 1.0, &mut rng);
    let blk = ConvBnAct::new(8, 8, 3, 2, 2, Activation::Silu, &mut rng);
    grad_check_with(
        |v| {
            let conv = Conv2d::from_tensors(*blk.conv.config(), v[1].clone(), None).map_err(core)?;
            let bn = BatchNorm2d::from_tensors(
                v[2].clone(),
                v[3].clone(),
                Tensor::zeros([8]),
                Tensor::ones([8]),
                BN_EPS,
                BN_MOMENTUM,
            )
            .map_err(core)?;
            ConvBnAct { conv, bn: Some(bn), act: Activation::Silu }.forward(&v[0]).map_err(core)
        },
        &[x, blk.conv.weight.detach(), Tensor::rand_uniform([8], 0.5, 1.5, &mut rng), Tensor::zeros([8])],
        &opts,
    )
    .unwrap();
}

#[test]
fn batchnorm_hand_computed_train_and_eval() {
    let bn = BatchNorm2d::new(1);
    {
        let _g = no_grad();
        bn.weight.fill_assign(2.0).unwrap();
        bn.bias.fill_assign(0.5).unwrap();
    }
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 1, 1, 2]).unwrap();
    // mean 2.5, biased var 1.25 over N·H·W = 4.
    let y = bn.forward(&x).unwrap().to_vec::<f32>().unwrap();
    close(&y, &[-2.1822089039291006, -0.3940696346430336, 1.3940696346430337, 3.1822089039291006], 1e-6);
    // running = 0.97 · running + 0.03 · stat, with the unbiased variance 1.25 · 4/3.
    close(&bn.running_mean.to_vec::<f32>().unwrap(), &[0.075], 1e-7);
    close(&bn.running_var.to_vec::<f32>().unwrap(), &[1.02], 1e-7);
    bn.eval();
    let rm = bn.running_mean.to_vec::<f32>().unwrap();
    let y = bn.forward(&x).unwrap().to_vec::<f32>().unwrap();
    close(&y, &[2.3308756862901383, 4.310200752549747, 6.289525818809356, 8.268850885068964], 1e-6);
    assert_eq!(bn.running_mean.to_vec::<f32>().unwrap(), rm, "eval leaves the statistics alone");
    // Training rejects N·H·W < 2.
    bn.train();
    assert!(bn.forward(&Tensor::ones([1, 1, 1, 1])).is_err());
    assert!(bn.forward(&Tensor::ones([1, 2, 2, 2])).is_err()); // wrong channel count
}

#[test]
fn running_stats_guard_restores_after_non_finite_batch() {
    let mut rng = Rng::seed(4);
    let m = Sequential::new().push_named("a", ConvBnAct::new(2, 3, 3, 1, 1, Activation::Silu, &mut rng));
    m.forward(&Tensor::randn([2, 2, 4, 4], 0.0, 1.0, &mut rng)).unwrap();
    let before: Vec<Vec<f32>> = m.named_buffers().iter().map(|(_, t)| t.to_vec::<f32>().unwrap()).collect();
    let snap = RunningStats::capture(&m).unwrap();
    let mut bad = Tensor::randn([2, 2, 4, 4], 0.0, 1.0, &mut rng).to_vec::<f32>().unwrap();
    bad[5] = f32::NAN;
    m.forward(&Tensor::from_vec(bad, [2, 2, 4, 4]).unwrap()).unwrap();
    assert!(!RunningStats::all_finite(&m).unwrap());
    snap.restore().unwrap();
    assert!(RunningStats::all_finite(&m).unwrap());
    let after: Vec<Vec<f32>> = m.named_buffers().iter().map(|(_, t)| t.to_vec::<f32>().unwrap()).collect();
    assert_eq!(before, after);
}

#[test]
fn fused_conv_matches_unfused_eval() {
    let mut rng = Rng::seed(5);
    for (k, s, g) in [(3, 1, 1), (3, 2, 4), (1, 1, 1)] {
        let mut blk = ConvBnAct::new(8, 8, k, s, g, Activation::Silu, &mut rng);
        // Non-trivial statistics and affine parameters.
        for _ in 0..3 {
            blk.forward(&Tensor::randn([2, 8, 6, 6], 0.5, 2.0, &mut rng)).unwrap();
        }
        {
            let bn = blk.bn.as_ref().unwrap();
            let _g = no_grad();
            bn.weight.copy_from(&Tensor::rand_uniform([8], 0.5, 1.5, &mut rng)).unwrap();
            bn.bias.copy_from(&Tensor::rand_uniform([8], -0.5, 0.5, &mut rng)).unwrap();
        }
        blk.eval();
        let x = Tensor::randn([2, 8, 6, 6], 0.0, 1.0, &mut rng);
        // Reference: the separate ops (eval BatchNorm on running statistics), not the block's folded fast path.
        let want = {
            let _g = no_grad();
            let y = blk.bn.as_ref().unwrap().forward(&blk.conv.forward(&x).unwrap()).unwrap();
            blk.act.forward(&y).unwrap().to_vec::<f32>().unwrap()
        };
        // The unfused block in eval mode under no_grad folds on the fly; it must agree too.
        let auto = no_grad_fwd(&blk, &x).to_vec::<f32>().unwrap();
        let worst = auto.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst <= 1e-5, "k{k} s{s} g{g}: on-the-fly fold max diff {worst}");
        blk.fuse_().unwrap();
        assert!(blk.is_fused());
        let names: Vec<_> = blk.named_tensors().into_iter().map(|t| t.name).collect();
        assert_eq!(names, ["conv.weight", "conv.bias"]);
        let got = no_grad_fwd(&blk, &x).to_vec::<f32>().unwrap();
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst <= 1e-5, "k{k} s{s} g{g}: max diff {worst}");
    }
}
