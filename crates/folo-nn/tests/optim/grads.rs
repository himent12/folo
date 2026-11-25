//! Gradient accumulation, global norm, clipping, and AMP unscale+clip tests.

use folo_core::rng::Rng;
use folo_core::{Grads, Tensor};
use folo_nn::optim::*;

use crate::{grads_for, param};

#[test]
fn accumulation_and_clipping() {
    let (p, q) = (param(&[1.0, 2.0]), param(&[3.0]));
    let mut acc = GradAccumulator::new(&[p.clone(), q.clone()]);
    // One backward where p and q receive the same gradient tensor (shared storage): buffers must not alias.
    let x = Tensor::from_slice(&[1.0f32], [1]).unwrap();
    let y = p.sum_all().unwrap().add(&q.mul(&x).unwrap().sum_all().unwrap()).unwrap();
    acc.add(&y.backward().unwrap()).unwrap();
    acc.add(&grads_for(&p, &[3.0, 4.0])).unwrap();
    assert_eq!(acc.passes(), 2);
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [4.0, 5.0]);
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [1.0]);
    // ‖g‖ = sqrt(16 + 25 + 1) = sqrt(42); clipped to 1 by 1 / (n + 1e-6).
    let n = acc.clip_grad_norm(1.0).unwrap();
    assert!((n - 42f64.sqrt()).abs() < 1e-12);
    let c = (1.0 / (42f64.sqrt() + 1e-6)) as f32;
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [4.0 * c, 5.0 * c]);
    assert!((acc.global_norm().unwrap() - 1.0).abs() < 1e-5);
    assert_eq!(acc.clip_grad_norm(10.0).unwrap(), acc.global_norm().unwrap(), "below the limit: unchanged");
    acc.zero();
    assert!(!acc.has_grads() && acc.grad(&p).is_none());
    acc.add(&grads_for(&q, &[2.0])).unwrap();
    assert!(acc.grad(&p).is_none(), "a parameter without a gradient in this window has none");
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [2.0]);
}

#[test]
fn accumulator_with_duplicate_handles() {
    let (p, q) = (param(&[1.0]), param(&[1.0]));
    let mut acc = GradAccumulator::new(&[p.clone(), p.clone(), q.clone()]);
    let mut grads = grads_for(&p, &[2.0]);
    grads.insert(&q, Tensor::from_slice(&[7.0f32], [1]).unwrap());
    acc.add(&grads).unwrap();
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [2.0]);
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [7.0]);
    // Only repeats: one slot, no out-of-range index.
    let mut acc = GradAccumulator::new(&[p.clone(), p.clone()]);
    acc.add(&grads_for(&p, &[3.0])).unwrap();
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [3.0]);
    assert!(acc.grad(&q).is_none());
}

#[test]
fn unscale_clip_and_global_norm_large_cpu_parity() {
    let mut rng = Rng::seed(123);
    // Exceeds PAR_THRESHOLD (65,536) across multiple uneven tensors to exercise parallel AVX2 paths
    // (9,216 + 18,432 + 41,472 + 37 = 69,157 > 65,536).
    let shapes: [&[usize]; 4] = [&[32, 32, 3, 3], &[64, 32, 3, 3], &[144, 32, 3, 3], &[37]];
    let params: Vec<Tensor> =
        shapes.iter().map(|s| Tensor::randn(s.to_vec(), 0.0, 1.0, &mut rng).requires_grad_()).collect();
    let total_numel: usize = params.iter().map(Tensor::numel).sum();
    assert!(total_numel > 65_536, "fixture must exercise parallel path: {total_numel}");
    let mut grads = Grads::default();
    let mut ref_vecs = Vec::new();
    for p in &params {
        let g = Tensor::randn(p.shape().clone(), 0.0, 4.0, &mut rng);
        ref_vecs.push(g.to_vec::<f32>().unwrap());
        grads.insert(p, g);
    }
    let mut acc = GradAccumulator::new(&params);
    acc.add(&grads).unwrap();

    // Thread-count determinism (1 vs 4 workers) for both global_norm and unscale_clip.
    let mut acc1 = GradAccumulator::new(&params);
    let mut acc4 = GradAccumulator::new(&params);
    acc1.add(&grads).unwrap();
    acc4.add(&grads).unwrap();
    let n1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| acc1.global_norm().unwrap());
    let n4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| acc4.global_norm().unwrap());
    assert_eq!(n1.to_bits(), n4.to_bits(), "global_norm must be thread-count deterministic");

    // Unscale by 0.25 and clip to max_norm = 5.0.
    let inv_scale = 0.25f32;
    let max_norm = 5.0f64;
    let ok1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| acc1.unscale_clip(inv_scale, max_norm).unwrap());
    let ok4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| acc4.unscale_clip(inv_scale, max_norm).unwrap());
    assert!(ok1 && ok4);
    for p in &params {
        let b1: Vec<u32> =
            acc1.grad(p).unwrap().to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
        let b4: Vec<u32> =
            acc4.grad(p).unwrap().to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
        assert_eq!(b1, b4, "unscale_clip must be bitwise thread-count deterministic");
    }

    assert!(acc.unscale_clip(inv_scale, max_norm).unwrap());
    let mut ref_sq = 0.0f64;
    for v in &mut ref_vecs {
        for x in v.iter_mut() {
            *x *= inv_scale;
            ref_sq += (*x as f64) * (*x as f64);
        }
    }
    let ref_n = ref_sq.sqrt();
    assert!(ref_n > max_norm);
    let c = (max_norm / (ref_n + 1e-6)) as f32;
    for (p, want) in params.iter().zip(&mut ref_vecs) {
        for x in want.iter_mut() {
            *x *= c;
        }
        let got = acc.grad(p).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
    assert!((acc.global_norm().unwrap() - max_norm).abs() < 1e-3);

    // Broadcastable (scalar) gradient on a multi-element parameter must work across multiple add() passes.
    let v = param(&[1.0, 2.0, 3.0]);
    let mut bacc = GradAccumulator::new(std::slice::from_ref(&v));
    let mut sg = Grads::default();
    sg.insert(&v, Tensor::from_slice(&[2.5f32], [1]).unwrap());
    bacc.add(&sg).unwrap();
    bacc.add(&sg).unwrap();
    assert_eq!(bacc.grad(&v).unwrap().to_vec::<f32>().unwrap(), [5.0, 5.0, 5.0]);

    // Non-finite gradient causes unscale_clip to return false (both with and without clipping).
    for clip in [0.0, 10.0] {
        acc.zero();
        let mut bad_grads = Grads::default();
        for (i, p) in params.iter().enumerate() {
            let mut v = vec![1.0f32; p.numel()];
            if i == 2 {
                v[1234] = f32::INFINITY;
            }
            bad_grads.insert(p, Tensor::from_vec(v, p.shape().clone()).unwrap());
        }
        acc.add(&bad_grads).unwrap();
        assert!(!acc.unscale_clip(0.5, clip).unwrap(), "clip={clip}");
    }
}
