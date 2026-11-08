//! Autograd through a small YOLO-style conv net on the GPU: conv, BatchNorm built from composite ops
//! (batch statistics), SiLU, max pooling, nearest upsampling and concat.
//!
//! - `net_gradients_match_cpu`: the same seeded parameters on CPU and GPU; loss and every parameter gradient
//!   must agree within `atol 1e-4 + rtol 1e-3` (f32 everywhere; the GPU sums in a different order and fuses
//!   multiply-adds, and BatchNorm divides by a batch standard deviation, which amplifies those differences).
//! - `net_gradients_match_finite_differences`: folo-core's `grad_check` (central differences, f64 loss
//!   accumulation) with the forward pass on the GPU, on the net without max pooling (its kinks make finite
//!   differences unreliable; the CPU comparison above covers it).

use folo_core::testing::{GradCheck, grad_check_with};
use folo_core::{Device, Result, Tensor};

use super::backend::{assert_close_t, dev, randn};

/// `(x - mean) / sqrt(var + eps) * gamma + beta` over the batch and spatial dims.
fn batch_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let mean = x.mean([0, 2, 3], true)?;
    let d = x.sub(&mean)?;
    let var = d.square()?.mean([0, 2, 3], true)?;
    d.mul(&var.add_scalar(1e-5)?.rsqrt()?)?.mul(gamma)?.add(beta)
}

/// Inputs: `[x, w1, g1, b1, w2, g2, b2, w3, b3]`.
fn net(v: &[Tensor], pool: bool) -> Result<Tensor> {
    let y1 = batch_norm(&v[0].conv2d(&v[1], None, 1, 1, 1, 1)?, &v[2], &v[3])?.silu()?; // [2, 8, 16, 16]
    let y2 = batch_norm(&y1.conv2d(&v[4], None, 2, 1, 1, 1)?, &v[5], &v[6])?.silu()?; // [2, 8, 8, 8]
    let p = if pool { y2.max_pool2d(3, 1, 1)? } else { y2 };
    let u = p.upsample_nearest2d(2)?; // [2, 8, 16, 16]
    let c = Tensor::cat(&[&y1, &u], 1)?; // [2, 16, 16, 16]
    c.conv2d(&v[7], Some(&v[8]), 1, 0, 1, 1) // [2, 4, 16, 16]
}

fn params() -> Vec<Tensor> {
    let s = |shape: &[usize], seed: u64, scale: f32| randn(shape, seed).mul_scalar(scale).unwrap();
    vec![
        s(&[2, 3, 16, 16], 1, 1.0),
        s(&[8, 3, 3, 3], 2, 0.3),
        s(&[1, 8, 1, 1], 3, 0.2).add_scalar(1.0).unwrap(),
        s(&[1, 8, 1, 1], 4, 0.1),
        s(&[8, 8, 3, 3], 5, 0.15),
        s(&[1, 8, 1, 1], 6, 0.2).add_scalar(1.0).unwrap(),
        s(&[1, 8, 1, 1], 7, 0.1),
        s(&[4, 16, 1, 1], 8, 0.25),
        s(&[4], 9, 0.1),
    ]
}

#[test]
fn net_gradients_match_cpu() {
    let dev = dev!();
    let run = |d: Device| {
        let leaves: Vec<Tensor> =
            params().iter().map(|t| t.to_device(d).unwrap().detach().requires_grad_()).collect();
        let out = net(&leaves, true).unwrap();
        let loss = out.square().unwrap().mean_all().unwrap().add(&out.mean_all().unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        let gs: Vec<Tensor> =
            leaves.iter().map(|l| grads.get(l).expect("every input gets a gradient").clone()).collect();
        (out, loss, gs)
    };
    let (oc, lc, gc) = run(Device::Cpu);
    let (og, lg, gg) = run(dev);
    assert!(og.device().is_gpu() && lg.device().is_gpu());
    assert_close_t(&og, &oc, 1e-4, 1e-3, "net output");
    assert_close_t(&lg, &lc, 1e-5, 1e-4, "net loss");
    let names = ["x", "w1", "g1", "b1", "w2", "g2", "b2", "w3", "b3"];
    for ((g, c), name) in gg.iter().zip(&gc).zip(names) {
        assert!(g.device().is_gpu(), "grad of {name} left the GPU");
        assert_close_t(g, c, 1e-4, 1e-3, &format!("grad of {name}"));
    }
}

#[test]
fn net_gradients_match_finite_differences() {
    let dev = dev!();
    let opts = GradCheck { eps: 1e-2, atol: 3e-3, rtol: 2e-2, max_checks: 6, seed: 7 };
    let f = |v: &[Tensor]| {
        let g: Vec<Tensor> = v.iter().map(|t| t.to_device(dev)).collect::<Result<_>>()?;
        net(&g, false)?.to_device(Device::Cpu)
    };
    grad_check_with(f, &params(), &opts).unwrap();
}

/// A few optimizer steps of folo-nn's SGD (Nesterov, weight decay) and AdamW on GPU parameters, through the
/// device-generic tensor-op paths, match the same steps on the CPU (`1e-5` relative).
#[test]
fn optimizer_steps_match_cpu() {
    use folo_nn::optim::{AdamW, Optimizer, ParamGroup, Sgd};
    let dev = dev!();
    let run = |d: Device, adam: bool| -> Vec<Tensor> {
        let p = params();
        let (x, w, b) =
            (p[0].to_device(d).unwrap(), p[1].to_device(d).unwrap().detach(), p[8].to_device(d).unwrap());
        let w = w.requires_grad_();
        let head = randn(&[4, 8, 3, 3], 11).to_device(d).unwrap().detach().requires_grad_();
        let group = ParamGroup::new("decay", vec![("w".into(), w.clone()), ("head".into(), head.clone())])
            .lr(0.05)
            .momentum(0.9)
            .weight_decay(5e-4);
        let mut opt: Box<dyn Optimizer> = if adam {
            Box::new(AdamW::new(vec![group.adam(0.999, 1e-8)]).unwrap())
        } else {
            Box::new(Sgd::new(vec![group]).unwrap())
        };
        for _ in 0..3 {
            let y = x.conv2d(&w, None, 1, 1, 1, 1).unwrap().silu().unwrap();
            let z = y.conv2d(&head, Some(&b), 2, 1, 1, 1).unwrap();
            let grads = z.square().unwrap().mean_all().unwrap().backward().unwrap();
            opt.step(&grads).unwrap();
        }
        vec![w, head]
    };
    for adam in [false, true] {
        let (c, g) = (run(Device::Cpu, adam), run(dev, adam));
        for (i, (g, c)) in g.iter().zip(&c).enumerate() {
            assert!(g.device().is_gpu());
            assert_close_t(g, c, 1e-5, 1e-5, &format!("{} param {i}", if adam { "adamw" } else { "sgd" }));
        }
    }
}
