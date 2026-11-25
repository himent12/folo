//! folo-nn hot paths: `cargo bench -p folo-nn --bench nn`.
//!
//! BatchNorm train forward + backward (fused CPU kernel vs the composite-op path), optimizer steps and the EMA
//! update over a folo-n-sized parameter set (2.6M scalars in 255 tensors), and `.folo` save / load.

use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Grads, Tensor};
use folo_nn::io::{FileKind, LoadOptions, Metadata, ModelConfig, ModelFile, Purpose, save_weights};
use folo_nn::optim::{AdamW, GradAccumulator, MuSgd, Optimizer, ParamGroup, Sgd};
use folo_nn::{BatchNorm2d, Conv2d, Conv2dConfig, Forward, ModelEma, ModuleList};

fn bench(budget: Duration, mut f: impl FnMut()) -> Duration {
    f();
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 3 || start.elapsed() < budget {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        runs += 1;
    }
    best
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The composite-op BatchNorm the fused kernel replaces (what non-CPU devices run).
fn bn_composite(x: &Tensor, bn: &BatchNorm2d) -> Tensor {
    let c = bn.channels();
    let mean = x.mean([0, 2, 3], true).unwrap();
    let diff = x.sub(&mean).unwrap();
    let var = diff.square().unwrap().mean([0, 2, 3], true).unwrap();
    let inv = var.add_scalar(bn.eps as f32).unwrap().rsqrt().unwrap();
    let scale = inv.mul(&bn.weight.reshape([1, c, 1, 1]).unwrap()).unwrap();
    diff.mul(&scale).unwrap().add(&bn.bias.reshape([1, c, 1, 1]).unwrap()).unwrap()
}

fn main() {
    let budget = Duration::from_millis(1500);
    let mut rng = Rng::seed(0);

    println!("BatchNorm2d train, forward + backward");
    for (n, c, hw) in [(16, 32, 160), (16, 64, 80), (16, 256, 20)] {
        let x = Tensor::randn([n, c, hw, hw], 0.0, 1.0, &mut rng).requires_grad_();
        let bn = BatchNorm2d::new(c);
        let fused = bench(budget, || {
            let _ = bn.forward(&x).unwrap().sum_all().unwrap().backward().unwrap();
        });
        let comp = bench(budget, || {
            let _ = bn_composite(&x, &bn).sum_all().unwrap().backward().unwrap();
        });
        let gb = (n * c * hw * hw * 4) as f64 / 1e9;
        println!(
            "  [{n}, {c}, {hw}, {hw}] ({gb:.2} GB): fused {:.2} ms, composite {:.2} ms ({:.1}x)",
            ms(fused),
            ms(comp),
            comp.as_secs_f64() / fused.as_secs_f64()
        );
    }

    // 255 tensors, 2.6M scalars: a 3x3 conv weight stack shaped like folo-n's parameter count.
    let convs: Vec<Conv2d> =
        (0..255).map(|_| Conv2d::new(Conv2dConfig::new(32, 36, 3).bias(false), &mut rng)).collect();
    let model = ModuleList::new(convs);
    let params: Vec<(String, Tensor)> = folo_nn::Module::named_parameters(&model);
    let numel: usize = params.iter().map(|(_, t)| t.numel()).sum();
    let mut grads = Grads::default();
    for (_, p) in &params {
        grads.insert(p, Tensor::randn(p.shape().clone(), 0.0, 1e-3, &mut rng));
    }
    println!("optimizer step over {} tensors, {:.2}M scalars", params.len(), numel as f64 / 1e6);
    let group = || vec![ParamGroup::new("decay", params.clone()).lr(1e-3).momentum(0.9).weight_decay(5e-4)];
    let mut sgd = Sgd::new(group()).unwrap();
    println!("  SGD (Nesterov, L2): {:.2} ms", ms(bench(budget, || sgd.step(&grads).unwrap())));
    let mut adamw = AdamW::new(group()).unwrap();
    println!("  AdamW: {:.2} ms", ms(bench(budget, || adamw.step(&grads).unwrap())));
    let mut musgd = MuSgd::new(group()).unwrap();
    println!("  MuSGD (Polar4): {:.2} ms", ms(bench(budget, || musgd.step(&grads).unwrap())));
    let p_only: Vec<Tensor> = params.iter().map(|(_, p)| p.clone()).collect();
    let mut acc = GradAccumulator::new(&p_only);
    acc.add(&grads).unwrap();
    println!(
        "  GradAccumulator::unscale_clip (max_norm=10): {:.2} ms",
        ms(bench(budget, || {
            let _ = acc.unscale_clip(1.0, 10.0).unwrap();
        }))
    );
    let shadow = ModuleList::new(
        (0..255).map(|_| Conv2d::new(Conv2dConfig::new(32, 36, 3).bias(false), &mut rng)).collect::<Vec<_>>(),
    );
    let mut ema = ModelEma::new(shadow, &model).unwrap();
    println!("  EMA update: {:.2} ms", ms(bench(budget, || ema.update(&model).unwrap())));

    let dir = std::env::temp_dir().join("folo-nn-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("w.folo");
    let meta =
        Metadata::new(FileKind::Weights, ModelConfig::new("n", 1).unwrap(), vec!["a".into()], 640).unwrap();
    println!("  save weights: {:.2} ms", ms(bench(budget, || save_weights(&path, &meta, &model).unwrap())));
    println!(
        "  open + strict load: {:.2} ms",
        ms(bench(budget, || {
            let f = ModelFile::open(&path).unwrap();
            f.load(&model, &LoadOptions::strict(Purpose::Training)).unwrap();
        }))
    );
}
