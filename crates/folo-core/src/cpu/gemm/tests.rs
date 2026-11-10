use super::kernel::{Kernel, kernel, kernel_generic};
use super::*;

fn naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f64;
            for p in 0..k {
                s += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = s as f32;
        }
    }
    c
}

fn data(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

#[test]
fn matches_naive_on_odd_sizes() {
    for &(m, n, k) in &[(1, 1, 1), (7, 17, 5), (13, 33, 300), (64, 64, 64), (100, 37, 520), (5, 300, 3000)] {
        let a = data(m * k, 1);
        let b = data(k * n, 2);
        let want = naive(m, n, k, &a, &b);
        let mut c = vec![0f32; m * n];
        sgemm(m, n, k, &a, k, 1, &b, n, 1, &mut c, n);
        for (x, y) in c.iter().zip(&want) {
            assert!((x - y).abs() <= 1e-3 * (1.0 + y.abs()), "{m}x{n}x{k}: {x} vs {y}");
        }
        // Transposed storage of both operands.
        let at: Vec<f32> = (0..k * m).map(|idx| a[(idx % m) * k + idx / m]).collect();
        let bt: Vec<f32> = (0..n * k).map(|idx| b[(idx % k) * n + idx / k]).collect();
        let mut c2 = vec![0f32; m * n];
        sgemm(m, n, k, &at, 1, m, &bt, 1, k, &mut c2, n);
        for (x, y) in c2.iter().zip(&want) {
            assert!((x - y).abs() <= 1e-3 * (1.0 + y.abs()));
        }
    }
}

#[test]
#[should_panic(expected = "C buffer too small")]
fn rejects_overflowing_ldc() {
    let (a, b) = ([1f32], [1f32]);
    let mut c = [0f32];
    gemm(3, 1, 1, &Strided::lhs(&a, 0, 0, 0), &Strided::rhs(&b, 0, 0, 0), &mut c, 1 << 63, false, false);
}

#[test]
fn generic_kernel_matches() {
    let kc = 9;
    let a = data(kc * MR, 3);
    let b = data(kc * NR, 4);
    let mut c1 = vec![1f32; MR * NR];
    let mut c2 = c1.clone();
    // SAFETY: buffers sized as the kernels require.
    unsafe {
        kernel_generic(kc, a.as_ptr(), b.as_ptr(), c1.as_mut_ptr(), NR, MR, NR, true, None);
        kernel()(kc, a.as_ptr(), b.as_ptr(), c2.as_mut_ptr(), NR, MR, NR, true, None);
    }
    for (x, y) in c1.iter().zip(&c2) {
        assert!((x - y).abs() < 1e-4);
    }
}

/// The epilogue a micro-kernel applies to its registers is bit for bit the row pass over the stored block
/// ([`BiasAct::apply`]), on the portable and the dispatched kernel, for full and ragged blocks, overwriting and
/// accumulating, with zero, negative and positive biases.
#[test]
fn kernel_epilogue_is_bitwise_the_row_pass() {
    let kc = 37;
    let a = data(kc * MR, 5);
    let b = data(kc * NR, 6);
    let bias = [0.0f32, -0.75, 1.5, 0.0, -2.0, 0.25];
    let kernels: [Kernel; 2] = [kernel_generic, kernel()];
    for kern in kernels {
        for (mr, nr) in [(MR, NR), (4, NR), (MR, 11), (1, 3)] {
            for acc in [false, true] {
                for act in [None, Some(UnaryOp::Silu)] {
                    for with_bias in [false, true] {
                        let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                        let prior = data(MR * NR, 7);
                        let (mut fused, mut want) = (prior.clone(), prior.clone());
                        let re = epi.kernel_rows(0, mr);
                        // SAFETY: the panels and blocks are sized as the kernels require.
                        unsafe {
                            kern(kc, a.as_ptr(), b.as_ptr(), fused.as_mut_ptr(), NR, mr, nr, acc, Some(&re));
                            kern(kc, a.as_ptr(), b.as_ptr(), want.as_mut_ptr(), NR, mr, nr, acc, None);
                        }
                        for i in 0..mr {
                            epi.apply(i, &mut want[i * NR..i * NR + nr]);
                        }
                        let same = fused.iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits());
                        assert!(same, "mr {mr} nr {nr} acc {acc} act {act:?} bias {with_bias}");
                    }
                }
            }
        }
    }
}
