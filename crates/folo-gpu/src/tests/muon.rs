//! The fused Muon step's groups and declines (`kernels::muon`): split into groups by a small workspace limit and
//! run with the fewest bindings, it gives the one-group result bit for bit (every matrix is its own problem);
//! too few bindings or a matrix past the limit decline the call and write nothing.

use super::{Rng, gpu};
use crate::kernels::muon::{MuonMatrix, MuonSettings, muon_step_within};
use crate::{GpuContext, GpuTensor};

/// `rows × cols`: (130, 40) is transposed; (260, 290) triggers `UP64X128_D16` (`r > 256`) in the one-group run
/// while the split groups below run `SQ64_D16`, verifying bit-exact agreement across tile sizes.
/// Padded workspace of `(260, 290)`: `260 * align4(290) * 4 = 303,680` bytes.
const SHAPES: [(usize, usize); 7] = [(16, 27), (130, 40), (1, 5), (260, 290), (70, 300), (96, 96), (5, 1)];
const LARGEST: u64 = 260 * 292 * 4;

/// The step on fresh copies of the same parameters, momenta and gradients: whether it ran, and the parameters
/// before and after.
fn run(ctx: &GpuContext, max_bytes: u64, bindings: usize) -> (bool, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let s = MuonSettings::uniform(0.9, true, (3.4445, -4.775, 2.0315), 5, 1e-7, 0.5);
    run_with(ctx, max_bytes, bindings, &s)
}

fn run_with(
    ctx: &GpuContext,
    max_bytes: u64,
    bindings: usize,
    s: &MuonSettings,
) -> (bool, Vec<Vec<f32>>, Vec<Vec<f32>>) {
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
            depthwise: false,
        })
        .collect();
    let ran = muon_step_within(ctx, &items, s, max_bytes, bindings).unwrap();
    let after = ts.iter().map(|t| t.0.to_vec().unwrap()).collect();
    (ran, before, after)
}

#[test]
fn muon_polar_groups_match_one_group() {
    let ctx = gpu!();
    let mut step_coeffs = [(0.0, 0.0, 0.0); crate::kernels::muon::MAX_STEPS];
    step_coeffs[0] = (4.216_272_4, -12.289_586, 9.009_986);
    step_coeffs[1] = (4.152_54, -11.770_327, 8.558_353);
    step_coeffs[2] = (3.887_979, -9.771_957, 6.839_154_7);
    step_coeffs[3] = (3.287_126, -5.261_409, 3.046_717_4);
    let s = MuonSettings {
        momentum: 0.9,
        nesterov: true,
        step_coeffs,
        cs9_coeffs: [(0.0, 0.0, 0.0, 0.0, 0.0); crate::kernels::muon::MAX_STEPS],
        cs9: false,
        steps: 4,
        schatten4: true,
        eps: 1e-7,
        lr: 0.5,
    };
    let (ran, before, whole) = run_with(&ctx, 1 << 30, ctx.caps().max_storage_buffers as usize, &s);
    assert!(ran);
    for (i, (b, w)) in before.iter().zip(&whole).enumerate() {
        assert_ne!(b, w, "matrix {i} updated");
    }
    let (ran, _, grouped) = run_with(&ctx, LARGEST, 5, &s);
    assert!(ran);
    for (i, (g, w)) in grouped.iter().zip(&whole).enumerate() {
        assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "matrix {i} {:?}", SHAPES[i]);
    }
}

#[test]
fn muon_cs9_groups_match_one_group() {
    let ctx = gpu!();
    let mut cs9_coeffs = [(0.0, 0.0, 0.0, 0.0, 0.0); crate::kernels::muon::MAX_STEPS];
    cs9_coeffs[0] = (-1.149_836, 0.234_306_05, -0.042_710_006, 0.069_763_21, 135.661_48);
    cs9_coeffs[1] = (-1.158_462, 0.242_525_31, -0.298_174_05, 0.491_120_82, 113.502_03);
    cs9_coeffs[2] = (-1.261_140_8, 0.343_998, -1.070_805_2, 1.935_225_4, 29.709_389);
    let s = MuonSettings {
        momentum: 0.9,
        nesterov: true,
        step_coeffs: [(0.0, 0.0, 0.0); crate::kernels::muon::MAX_STEPS],
        cs9_coeffs,
        cs9: true,
        steps: 3,
        schatten4: true,
        eps: 1e-7,
        lr: 0.5,
    };
    let (ran, before, whole) = run_with(&ctx, 1 << 30, ctx.caps().max_storage_buffers as usize, &s);
    assert!(ran);
    for (i, (b, w)) in before.iter().zip(&whole).enumerate() {
        assert_ne!(b, w, "cs9 matrix {i} updated");
    }
    let (ran, _, grouped) = run_with(&ctx, LARGEST, 5, &s);
    assert!(ran);
    for (i, (g, w)) in grouped.iter().zip(&whole).enumerate() {
        assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "cs9 matrix {i} {:?}", SHAPES[i]);
    }
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn muon_groups_match_one_group() {
    let ctx = gpu!();
    let (ran, before, whole) = run(&ctx, 1 << 30, ctx.caps().max_storage_buffers as usize);
    assert!(ran);
    for (i, (b, w)) in before.iter().zip(&whole).enumerate() {
        assert_ne!(b, w, "matrix {i} updated");
    }
    // Three groups ([16×27, 130×40, 1×5], [260×290], [70×300, 96×96, 5×1]): group 1 uses UP64X128_D16 while
    // groups 0 and 2 use SQ64_D16, one matrix per gather, three per update.
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
