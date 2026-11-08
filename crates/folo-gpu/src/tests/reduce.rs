//! Reductions along every dim, softmax and log-softmax.

use super::{Rng, assert_close, gpu};
use crate::GpuTensor;
use crate::kernels::reduce::ReduceOp;

fn cpu_reduce(op: ReduceOp, x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let outer: usize = shape[..dim].iter().product();
    let r = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let vals = (0..r).map(|k| x[(o * r + k) * inner + i] as f64);
            out.push(match op {
                ReduceOp::Sum => vals.sum::<f64>() as f32,
                ReduceOp::Mean => (vals.sum::<f64>() / r as f64) as f32,
                ReduceOp::Max => vals.fold(f64::NEG_INFINITY, f64::max) as f32,
                ReduceOp::Min => vals.fold(f64::INFINITY, f64::min) as f32,
            });
        }
    }
    out
}

#[test]
fn reductions_along_each_dim() {
    let ctx = gpu!();
    let mut rng = Rng::new(5);
    for shape in [vec![3, 70, 5], vec![2, 1000], vec![513], vec![4, 3, 2, 67], vec![1, 300, 1]] {
        let x = rng.vec(shape.iter().product(), -2.0, 2.0);
        let gx = GpuTensor::from_slice(&ctx, &x, &shape).unwrap();
        for dim in 0..shape.len() {
            for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
                let want = cpu_reduce(op, &x, &shape, dim);
                let got = gx.reduce(op, dim, false).unwrap();
                let mut want_shape = shape.clone();
                want_shape.remove(dim);
                assert_eq!(got.shape(), want_shape.as_slice());
                // Sums of up to 1000 values in [-2, 2] in a different order: 1e-4 absolute.
                assert_close(
                    &got.to_vec().unwrap(),
                    &want,
                    1e-4,
                    1e-5,
                    &format!("{op:?} {shape:?} dim {dim}"),
                );
            }
        }
    }
    let gx = GpuTensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(gx.sum(1, true).unwrap().shape(), &[2, 1]);
    assert_eq!(gx.max(0, false).unwrap().to_vec().unwrap(), vec![3.0, 4.0]);
    assert!(gx.sum(2, false).is_err());
}

#[test]
fn max_handles_infinities() {
    let ctx = gpu!();
    let x = [f32::NEG_INFINITY, f32::NEG_INFINITY, -1.0, f32::INFINITY];
    let gx = GpuTensor::from_slice(&ctx, &x, &[2, 2]).unwrap();
    assert_eq!(gx.max(1, false).unwrap().to_vec().unwrap(), vec![f32::NEG_INFINITY, f32::INFINITY]);
    assert_eq!(
        gx.reduce(ReduceOp::Min, 0, false).unwrap().to_vec().unwrap(),
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY]
    );
}

fn cpu_softmax(x: &[f32], n: usize, log: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(n) {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        for &v in row {
            out.push(if log { (v as f64 - m - s.ln()) as f32 } else { ((v as f64 - m).exp() / s) as f32 });
        }
    }
    out
}

#[test]
fn softmax_and_log_softmax() {
    let ctx = gpu!();
    let mut rng = Rng::new(6);
    for (rows, n) in [(1, 1), (7, 80), (3, 1000), (130, 16), (2, 257)] {
        let mut x = rng.vec(rows * n, -10.0, 10.0);
        // Large logits must not overflow.
        x[0] = 500.0;
        let gx = GpuTensor::from_slice(&ctx, &x, &[rows, n]).unwrap();
        let got = gx.softmax().unwrap().to_vec().unwrap();
        assert_close(&got, &cpu_softmax(&x, n, false), 1e-6, 1e-5, &format!("softmax {rows}x{n}"));
        let got = gx.log_softmax().unwrap().to_vec().unwrap();
        assert_close(&got, &cpu_softmax(&x, n, true), 1e-4, 1e-5, &format!("log_softmax {rows}x{n}"));
    }
}
