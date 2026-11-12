//! The fused Muon step's groups and declines (`kernels::muon`): split into groups by a small workspace limit and
//! run with the fewest bindings, it gives the one-group result bit for bit (every matrix is its own problem);
//! too few bindings or a matrix past the limit decline the call and write nothing.

use super::{Rng, gpu};
use crate::kernels::muon::{MuonMatrix, MuonSettings, muon_step_within};
use crate::{GpuContext, GpuTensor};

/// `rows × cols`: (130, 40) is transposed; (70, 300) is the largest, 21,000 elements (84,000 bytes).
const SHAPES: [(usize, usize); 6] = [(16, 27), (130, 40), (1, 5), (70, 300), (96, 96), (5, 1)];
const LARGEST: u64 = 70 * 300 * 4;

/// The step on fresh copies of the same parameters, momenta and gradients: whether it ran, and the parameters
/// before and after.
fn run(ctx: &GpuContext, max_bytes: u64, bindings: usize) -> (bool, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let rng = &mut Rng::new(29);
    let mut ts = Vec::new();
    for &(r, c) in &SHAPES {
        let [p, m, g] =
            [1.0, 1.0, 0.5].map(|a| GpuTensor::from_slice(ctx, &rng.vec(r * c, -a, a), &[r, c]).unwrap());
        ts.push((p, m, g, r, c));
    }
    let before: Vec<Vec<f32>> = ts.iter().map(|t| t.0.to_vec().unwrap()).collect();
    let items: Vec<MuonMatrix<'_>> = ts
        .iter()
        .map(|(p, m, g, r, c)| MuonMatrix {
            p: p.raw(),
            m: m.raw(),
            g: g.raw(),
            rows: *r,
            cols: *c,
            scale: (*r as f32 / *c as f32).max(1.0).sqrt(),
        })
        .collect();
    let s = MuonSettings {
        momentum: 0.9,
        nesterov: true,
        coeffs: (3.4445, -4.775, 2.0315),
        steps: 5,
        eps: 1e-7,
        lr: 0.5,
    };
    let ran = muon_step_within(ctx, &items, &s, max_bytes, bindings).unwrap();
    let after = ts.iter().map(|t| t.0.to_vec().unwrap()).collect();
    (ran, before, after)
}

#[test]
fn muon_groups_match_one_group() {
    let ctx = gpu!();
    let (ran, before, whole) = run(&ctx, 1 << 30, ctx.caps().max_storage_buffers as usize);
    assert!(ran);
    for (i, (b, w)) in before.iter().zip(&whole).enumerate() {
        assert_ne!(b, w, "matrix {i} updated");
    }
    // Three groups ([16×27, 130×40, 1×5], [70×300], [96×96, 5×1]), one matrix per gather, three per update.
    let (ran, _, grouped) = run(&ctx, LARGEST, 5);
    assert!(ran);
    for (i, (g, w)) in grouped.iter().zip(&whole).enumerate() {
        assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "matrix {i} {:?}", SHAPES[i]);
    }
}

#[test]
fn muon_declines_without_writing() {
    let ctx = gpu!();
    for (max_bytes, bindings, what) in
        [(1 << 30, 4, "four storage bindings"), (LARGEST - 4, 8, "a matrix past the binding size")]
    {
        let (ran, before, after) = run(&ctx, max_bytes, bindings);
        assert!(!ran, "{what}: declined");
        assert_eq!(before, after, "{what}: nothing written");
    }
}
