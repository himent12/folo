//! GEMM unit tests: results against a naive GEMM (`f64` sums) on odd sizes and transposed operands, stride
//! overflow checks, the portable micro-kernel against the dispatched one, the register epilogue bit for bit
//! against the row pass, and the row-pointer A paths bit for bit against the packed path.

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
    gemm(
        3,
        1,
        1,
        &Strided::lhs(&a, 0, 0, 0),
        &Strided::rhs(&b, 0, 0, 0),
        &mut c,
        1usize << (usize::BITS - 1),
        false,
        false,
    );
}

/// A's extent is checked without overflow before its rows are read in place: a row stride that wraps
/// `offset + (m - 1) * so + k` round to a small number panics instead of reading out of bounds.
#[test]
#[should_panic(expected = "A buffer too small")]
fn rejects_overflowing_a_row_stride() {
    let mut c = [0f32; 2];
    sgemm(2, 1, 2, &[1.0], usize::MAX, 1, &[1.0, 1.0], 1, 1, &mut c, 1);
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
        for (mr, nr) in [(MR, NR), (5, NR), (4, NR), (3, 11), (2, NR), (1, 3)] {
            for acc in [false, true] {
                for act in [
                    None,
                    Some(UnaryOp::Silu),
                    Some(UnaryOp::Relu),
                    Some(UnaryOp::LeakyRelu(0.1)),
                    Some(UnaryOp::Hardswish),
                ] {
                    for with_bias in [false, true] {
                        let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                        let mut prior = data(MR * NR, 7);
                        // Include edge values for Relu/LeakyRelu/Hardswish: -0.0, 0.0, ±3.0, ±10.0, NaN.
                        prior[0] = -0.0;
                        prior[1] = 0.0;
                        prior[2] = -3.0;
                        prior[3] = 3.0;
                        prior[4] = -10.0;
                        prior[5] = 10.0;
                        prior[6] = f32::NAN;
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
                        let same = fused
                            .iter()
                            .zip(&want)
                            .all(|(x, y)| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits());
                        assert!(same, "mr {mr} nr {nr} acc {acc} act {act:?} bias {with_bias}");
                    }
                }
            }
        }
    }
}

/// A row-major A is read in place through row pointers ([`Operand::rows_in_place`]) instead of packed: the same
/// FMAs in the same order, so the same bits as the packed path (here forced by storing A column-major), with and
/// without the in-kernel epilogue, for ragged row and column blocks, K over one `KC` block, the K split, and
/// any thread budget; and the same for [`gemm_rows`], whose B is read through row pointers too.
#[test]
fn a_in_place_is_bitwise_the_packed_path() {
    let bias = data(64, 9);
    // The last two split K, so K parts start A at a nonzero k0.
    let split = [(16, 16, 4096), (8, 64, 4096)];
    assert!(split.iter().all(|&(m, n, k)| super::blocked::k_split_parts(m, n, k) > 1));
    for &(m, n, k) in [(13, 37, 5), (6, 16, 256), (13, 37, 300), (64, 50, 600)].iter().chain(&split) {
        let a = data(m * k, 1);
        let at: Vec<f32> = (0..k * m).map(|idx| a[(idx % m) * k + idx / m]).collect();
        let b = data(k * n, 2);
        let (rowmaj, colmaj) = (Strided::lhs(&a, 0, k, 1), Strided::lhs(&at, 0, 1, m));
        assert!(rowmaj.rows_in_place().is_some() && colmaj.rows_in_place().is_none());
        for act in [
            None,
            Some(UnaryOp::Silu),
            Some(UnaryOp::Relu),
            Some(UnaryOp::LeakyRelu(0.1)),
            Some(UnaryOp::Hardswish),
        ] {
            let epi = BiasAct { bias: Some(&bias[..m]), act };
            for threads in [1, 8] {
                let (mut c1, mut c2) = (vec![0f32; m * n], vec![0f32; m * n]);
                let rhs = Strided::rhs(&b, 0, n, 1);
                gemm_ex(m, n, k, &rowmaj, &rhs, &mut c1, n, false, threads, Some(epi));
                gemm_ex(m, n, k, &colmaj, &rhs, &mut c2, n, false, threads, Some(epi));
                let same = c1.iter().zip(&c2).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same, "gemm_ex {m}x{n}x{k} act {act:?} threads {threads}");
                let rows: Vec<*const f32> = (0..k).map(|p| b[p * n..].as_ptr()).collect();
                let bp = RowPtrs { rows: &rows, n };
                let map = ColumnMap { wp: n, ow: n };
                let (mut r1, mut r2) = (vec![0f32; m * n], vec![0f32; m * n]);
                gemm_rows(m, k, &rowmaj, &bp, &mut r1, map, threads, Some(epi));
                gemm_rows(m, k, &colmaj, &bp, &mut r2, map, threads, Some(epi));
                let same = r1.iter().zip(&r2).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same, "gemm_rows {m}x{n}x{k} act {act:?} threads {threads}");
            }
        }
    }
}

/// The in-place-A micro-kernels (portable and dispatched; A rows through pointers, B packed or through row
/// pointers) give the packed-panel kernel's bits, epilogue and ragged rows included.
#[test]
fn a_rows_kernels_are_bitwise_the_packed_kernel() {
    use super::arows::{arows_kernel, arows_kernel_generic};
    use super::rows::{RowsAKernel, rows_a_kernel, rows_a_kernel_generic};
    let bias = [0.5f32, -1.0, 0.0, 2.0, -0.25, 1.0];
    let pairs: [(Kernel, RowsAKernel, super::arows::ARowsKernel); 2] = [
        (kernel_generic, rows_a_kernel_generic, arows_kernel_generic),
        (kernel(), rows_a_kernel(), arows_kernel()),
    ];
    for (kern, rakern, akern) in pairs {
        for kc in [0, 1, 2, 3, 4, 5, 41] {
            let arows = data(MR * kc, 11);
            let b = data(kc * NR, 12);
            let brows: Vec<*const f32> = (0..kc).map(|p| b[p * NR..].as_ptr()).collect();
            for mr in [MR, 5, 4, 3, 2, 1] {
                // The packed panel of rows 0..mr (zeros past them); the row pointers repeat row 0 past them.
                let mut pa = vec![0f32; kc * MR];
                for p in 0..kc {
                    for i in 0..mr {
                        pa[p * MR + i] = arows[i * kc + p];
                    }
                }
                let ap: [*const f32; MR] =
                    std::array::from_fn(|i| arows[if i < mr { i } else { 0 } * kc..].as_ptr());
                for act in
                    [RegAct::None, RegAct::Silu, RegAct::Relu, RegAct::LeakyRelu(0.1), RegAct::Hardswish]
                {
                    let re = RegEpi { bias, act };
                    let prior = data(MR * NR, 13);
                    let (mut want, mut got_r, mut got_a) = (prior.clone(), prior.clone(), prior.clone());
                    // SAFETY: the panels, row pointers and blocks are sized as the kernels require.
                    unsafe {
                        kern(kc, pa.as_ptr(), b.as_ptr(), want.as_mut_ptr(), NR, mr, NR, true, Some(&re));
                        rakern(kc, &ap, brows.as_ptr(), 0, got_r.as_mut_ptr(), NR, mr, true, Some(&re));
                        akern(kc, &ap, b.as_ptr(), got_a.as_mut_ptr(), NR, mr, NR, true, Some(&re));
                    }
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&got_r), bits(&want), "rows-A kernel, kc {kc}, mr {mr}, act {act:?}");
                    assert_eq!(bits(&got_a), bits(&want), "A-rows kernel, kc {kc}, mr {mr}, act {act:?}");
                }
            }
        }
    }
}
