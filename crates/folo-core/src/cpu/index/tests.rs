use super::*;

fn serial_index_add(out: &mut [f32], (outer, d, inner): (usize, usize, usize), idx: &[usize], src: &[f32]) {
    let n = idx.len();
    for o in 0..outer {
        for (i, &j) in idx.iter().enumerate() {
            for k in 0..inner {
                out[(o * d + j) * inner + k] += src[(o * n + i) * inner + k];
            }
        }
    }
}

fn serial_scatter_add(
    out: &mut [f32],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    src: &[f32],
    di: usize,
) {
    for o in 0..outer {
        for i in 0..di {
            for k in 0..inner {
                let e = (o * di + i) * inner + k;
                out[(o * d + idx[e]) * inner + k] += src[e];
            }
        }
    }
}

fn vals(n: usize, seed: u64) -> Vec<f32> {
    let mut r = crate::rng::Rng::seed(seed);
    (0..n).map(|_| r.uniform(-1.0, 1.0) * 1e3f32.powf(r.uniform(-1.0, 1.0))).collect()
}

#[test]
fn to_usize_avx2_matches_scalar_and_reports_first_oob() {
    for len in [0usize, 1, 3, 4, 7, 8, 9, 15, 16, 17, 33, PAR_MIN + 19] {
        let size = 100usize;
        let idx_i64: Vec<i64> = (0..len).map(|i| (i % size) as i64).collect();
        let idx_u32: Vec<u32> = (0..len).map(|i| (i % size) as u32).collect();
        assert_eq!(to_usize_i64("t", &idx_i64, size).unwrap(), to_usize_scalar("t", &idx_i64, size).unwrap());
        assert_eq!(to_usize_u32("t", &idx_u32, size).unwrap(), to_usize_scalar("t", &idx_u32, size).unwrap());
        if len > 0 {
            // Inject two OOB elements and verify the FIRST one is reported.
            for bad_pos in [0usize, len / 2, len - 1] {
                let mut bad_i = idx_i64.clone();
                bad_i[bad_pos] = if bad_pos % 2 == 0 { -1 } else { size as i64 };
                bad_i[len - 1] = -99;
                let got = to_usize_i64("t", &bad_i, size).unwrap_err();
                let want = to_usize_scalar("t", &bad_i, size).unwrap_err();
                assert_eq!(got.to_string(), want.to_string(), "i64 len={len} bad_pos={bad_pos}");

                let mut bad_u = idx_u32.clone();
                bad_u[bad_pos] = size as u32;
                bad_u[len - 1] = u32::MAX;
                let got = to_usize_u32("t", &bad_u, size).unwrap_err();
                let want = to_usize_scalar("t", &bad_u, size).unwrap_err();
                assert_eq!(got.to_string(), want.to_string(), "u32 len={len} bad_pos={bad_pos}");
            }
        }
    }
    // Empty valid with size == 0; non-empty fails at index 0.
    assert!(to_usize_i64("t", &[], 0).unwrap().is_empty());
    assert!(to_usize_u32("t", &[], 0).unwrap().is_empty());
    assert!(to_usize_i64("t", &[0], 0).is_err());
    assert!(to_usize_u32("t", &[0], 0).is_err());
}

#[test]
fn gather_and_index_select_f32_match_generic_bitwise() {
    let mut r = crate::rng::Rng::seed(19);
    for &(outer, d, inner, n) in &[
        (1usize, 13, 1, 1),
        (1, 13, 1, 17),
        (3, 19, 1, 33),
        (2, 64, 1, PAR_MIN + 7),
        (40, 17, 1, 2000),
        (3, 19, 7, 11),
        (2, 32, 16, 15),
    ] {
        let x = vals(outer * d * inner, 11);
        let idx: Vec<usize> = (0..n).map(|_| (r.next_u64() % d as u64) as usize).collect();
        let got = index_select_f32(&x, (outer, d, inner), &idx);
        let want = index_select(&x, (outer, d, inner), &idx);
        assert!(
            got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "index_select {outer} {d} {inner} {n}"
        );
    }
    for &(outer, d, inner, di) in &[
        (5usize, 11, 1, 1),
        (PAR_MIN + 5, 13, 1, 1),
        (3, 17, 1, 19),
        (2, 64, 1, PAR_MIN / 2 + 13),
        (64, 32, 1, 1024),
        (3, 11, 7, 5),
        (2, 17, 33, 9),
        (1, 256, 513, 128),
        (2, 16, 4097, 8),
    ] {
        let x = vals(outer * d * inner, 21);
        let idx: Vec<usize> = (0..outer * di * inner).map(|_| (r.next_u64() % d as u64) as usize).collect();
        let got = gather_f32(&x, (outer, d, inner), &idx, di);
        let want = gather(&x, (outer, d, inner), &idx, di);
        assert!(
            got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "gather {outer} {d} {inner} {di}"
        );
    }
}

#[test]
fn parallel_accumulation_and_avx2_scatter_add_are_bitwise_serial() {
    let mut r = crate::rng::Rng::seed(7);
    // (outer, d, inner, n): long rows, many duplicates, few rows of a big table, inner == 1.
    for &(outer, d, inner, n) in
        &[(1, 50, 4096, 300), (3, 1000, 64, 2000), (1, 100_000, 8, 16), (2, 64, 1, 70_000)]
    {
        let idx: Vec<usize> = (0..n).map(|_| (r.next_u64() % d as u64) as usize).collect();
        let src = vals(outer * n * inner, 1);
        let base = vals(outer * d * inner, 2);
        let mut want = base.clone();
        serial_index_add(&mut want, (outer, d, inner), &idx, &src);
        let got = index_add(&base, (outer, d, inner), &idx, &src);
        assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()), "{outer} {d} {inner} {n}");
    }
    // scatter_add strategies: ragged inner widths (7, 8, 9, 17, 65), uniform-index chunks, mixed chunks,
    // many outer blocks, few wide blocks (column-split AVX2), and few narrow blocks.
    for &(outer, d, inner, di) in &[
        (1, 5, 7, 13),
        (2, 9, 8, 17),
        (2, 9, 9, 17),
        (3, 11, 17, 23),
        (64, 10, 64, 20),
        (2, 30, 4096, 40),
        (1, 13, 65, 600),
        (1, 500, 1, 100_000),
        (3, 7, 3, 9000),
    ] {
        // Mix uniform 8-column runs (exercises contiguous fast path) and random indices (gather path).
        let mut idx = vec![0usize; outer * di * inner];
        for o in 0..outer {
            for i in 0..di {
                let row_off = (o * di + i) * inner;
                if i % 3 == 0 {
                    let uniform_j = (r.next_u64() % d as u64) as usize;
                    idx[row_off..row_off + inner].fill(uniform_j);
                } else {
                    for slot in &mut idx[row_off..row_off + inner] {
                        *slot = (r.next_u64() % d as u64) as usize;
                    }
                }
            }
        }
        let src = vals(idx.len(), 3);
        let base = vals(outer * d * inner, 4);
        let mut want = base.clone();
        serial_scatter_add(&mut want, (outer, d, inner), &idx, &src, di);
        let got = scatter_add(&base, (outer, d, inner), &idx, &src, di);
        assert!(
            want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
            "scatter_add {outer} {d} {inner} {di}"
        );
    }
    // Signed zeros survive every strategy: -0.0 + -0.0 + ... stays -0.0.
    for &(outer, d, inner, di) in &[(1, 1, 1, 40_000), (1, 2, 128, 300), (40, 1, 1, 1000), (2, 4, 17, 9)] {
        let idx = vec![0usize; outer * di * inner];
        let src = vec![-0.0f32; idx.len()];
        let got = scatter_add(&vec![-0.0; outer * d * inner], (outer, d, inner), &idx, &src, di);
        assert!(got.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()), "{outer} {d} {inner} {di}");
    }
    let got = index_add(&vec![-0.0; 4 * 64], (1, 4, 64), &vec![1; 1000], &vec![-0.0; 1000 * 64]);
    assert!(got.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
}

#[test]
fn empty_offset_f32_views_do_not_panic_in_index_select_or_gather() {
    use crate::backend::{Backend, View};
    use crate::{CpuStorage, Layout, Shape, Storage};

    let backend = crate::cpu::CpuBackend;
    let x_empty = Storage::Cpu(CpuStorage::F32(vec![]));
    let l_empty = Layout::new(Shape::new(&[0, 4]), vec![4, 1].into(), 10);
    let idx_s = Storage::Cpu(CpuStorage::I64(vec![0, 2]));
    let l_idx = Layout::contiguous(Shape::new(&[2]));
    let out_sel = backend.index_select(View::new(&x_empty, &l_empty), 1, View::new(&idx_s, &l_idx)).unwrap();
    assert_eq!(out_sel.len(), 0);

    let idx_empty = Storage::Cpu(CpuStorage::I64(vec![]));
    let l_gidx = Layout::new(Shape::new(&[0, 2]), vec![2, 1].into(), 7);
    let out_g = backend.gather(View::new(&x_empty, &l_empty), 1, View::new(&idx_empty, &l_gidx)).unwrap();
    assert_eq!(out_g.len(), 0);
}
