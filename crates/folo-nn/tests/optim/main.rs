//! Optimizers, groups, accumulation, clipping, schedule application, MuSGD/orthogonalisation, and EMA.

mod ema;
mod grads;
mod muon;
mod sched_math;
mod step;

use folo_core::{Grads, Tensor};

pub(crate) const G1: [f32; 2] = [0.5, 0.25];
pub(crate) const G2: [f32; 2] = [-1.0, 2.0];

pub(crate) fn close(a: &[f32], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x as f64 - y).abs() <= tol * (1.0 + y.abs()), "element {i}: {x} vs {y}");
    }
}

/// Gradients for `p` from a backward pass of `sum(p * g)`, which is exactly `g`.
pub(crate) fn grads_for(p: &Tensor, g: &[f32]) -> Grads {
    let gt = Tensor::from_slice(g, p.shape().clone()).unwrap();
    p.mul(&gt).unwrap().sum_all().unwrap().backward().unwrap()
}

pub(crate) fn param(v: &[f32]) -> Tensor {
    Tensor::from_slice(v, [v.len()]).unwrap().requires_grad_()
}
