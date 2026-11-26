use super::*;
use crate::rng::Rng;

fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..len).map(|_| rng.normal() * 1.7 - 0.15).collect()
}

#[test]
fn quantize_and_dequantize_match_scalar_bit_exact_and_handle_edges() {
    for &len in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 256] {
        let src = rand_vec(len, 100 + len as u64);
        let blocks = quantize_q4(&src);
        assert_eq!(blocks.len(), len.div_ceil(BLOCK_SIZE));

        // Compare against explicit scalar block reference bit-for-bit.
        let mut scalar_blocks = Vec::with_capacity(blocks.len());
        let (chunks, tail) = src.as_chunks::<BLOCK_SIZE>();
        for c in chunks {
            scalar_blocks.push(quantize_block_q4_scalar(c));
        }
        if !tail.is_empty() {
            let mut pad = [0.0f32; BLOCK_SIZE];
            pad[..tail.len()].copy_from_slice(tail);
            scalar_blocks.push(quantize_block_q4_scalar(&pad));
        }
        assert_eq!(blocks, scalar_blocks, "quantize_q4 mismatch at len={len}");

        let mut got = vec![0.0f32; len];
        dequantize_q4(&blocks, &mut got);
        let mut want = vec![0.0f32; len];
        let (wchunks, wtail) = want.as_chunks_mut::<BLOCK_SIZE>();
        for (blk, wc) in scalar_blocks.iter().zip(&mut *wchunks) {
            dequantize_block_q4_scalar(blk, wc);
        }
        if !wtail.is_empty() {
            let mut tmp = [0.0f32; BLOCK_SIZE];
            dequantize_block_q4_scalar(&scalar_blocks[wchunks.len()], &mut tmp);
            wtail.copy_from_slice(&tmp[..wtail.len()]);
        }
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "dequantize_q4 bit mismatch at len={len}"
        );

        // Reconstruction error bounded by `0.5 * scale` per element.
        for (i, (&orig, &dq)) in src.iter().zip(&got).enumerate() {
            let scale = blocks[i / BLOCK_SIZE].scale;
            assert!(
                (orig - dq).abs() <= 0.501 * scale + 1e-6,
                "len={len} idx={i}: orig={orig} dq={dq} scale={scale}"
            );
        }
    }

    // Edge cases: all zeros, constant, -0.0, subnormals, NaN, +/-Inf.
    let mut special = [0.0f32; BLOCK_SIZE];
    special[0] = -0.0;
    special[1] = f32::NAN;
    special[2] = f32::INFINITY;
    special[3] = f32::NEG_INFINITY;
    special[4] = 7.0;
    special[19] = -3.5;
    special[31] = 1.0e-42;
    let b = quantize_q4(&special);
    let b_ref = quantize_block_q4_scalar(&special);
    assert_eq!(b[0], b_ref);
    assert_eq!(b[0].scale, 1.0);
    let mut dq = [0.0f32; BLOCK_SIZE];
    dequantize_q4(&b, &mut dq);
    assert_eq!(dq[1], 0.0);
    assert_eq!(dq[2], 0.0);
    assert_eq!(dq[3], 0.0);
    assert_eq!(dq[4], 7.0);
    assert!((dq[19] - -4.0).abs() < 1e-6 || (dq[19] - -3.0).abs() < 1e-6);

    // Pure non-finite / subnormal-underflow block produces Q4Block::ZERO.
    let all_bad = [f32::NAN; BLOCK_SIZE];
    assert_eq!(quantize_q4(&all_bad)[0], Q4Block::ZERO);
    let tiny = [f32::MIN_POSITIVE * 1e-3; BLOCK_SIZE];
    assert_eq!(quantize_q4(&tiny)[0], Q4Block::ZERO);
}

#[test]
fn q8_blocks_and_dot_q4_q8_match_scalar_bit_exact() {
    for &len in &[0usize, 1, 16, 31, 32, 33, 64, 127, 256] {
        let w = rand_vec(len, 200 + len as u64);
        let x = rand_vec(len, 300 + len as u64);
        let wq = quantize_q4(&w);
        let xq = quantize_q8_blocks(&x);

        let mut xq_ref = Vec::with_capacity(xq.len());
        let (chunks, tail) = x.as_chunks::<BLOCK_SIZE>();
        for c in chunks {
            xq_ref.push(quantize_block_q8_scalar(c));
        }
        if !tail.is_empty() {
            let mut pad = [0.0f32; BLOCK_SIZE];
            pad[..tail.len()].copy_from_slice(tail);
            xq_ref.push(quantize_block_q8_scalar(&pad));
        }
        assert_eq!(xq, xq_ref, "quantize_q8_blocks mismatch at len={len}");

        let got = dot_q4_q8(&wq, &xq);
        let want = dot_q4_q8_scalar(&wq, &xq);
        assert_eq!(got.to_bits(), want.to_bits(), "dot_q4_q8 bit mismatch at len={len}");
    }

    // Exercise extreme nibbles (q = -8 -> nibble 0, q = +7 -> nibble 15) and q8 = +/-127.
    let extreme_w = [Q4Block { scale: 0.25, qs: [0xF0; PACKED_BLOCK_BYTES] }];
    let mut extreme_x_vals = [0.0f32; BLOCK_SIZE];
    for (i, v) in extreme_x_vals.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 127.0 } else { -127.0 };
    }
    let extreme_x = quantize_q8_blocks(&extreme_x_vals);
    assert_eq!(
        dot_q4_q8(&extreme_w, &extreme_x).to_bits(),
        dot_q4_q8_scalar(&extreme_w, &extreme_x).to_bits()
    );
}

#[test]
fn dot_q4_f32_gemv_and_matmul_match_scalar_and_reference() {
    for &len in &[0usize, 1, 5, 15, 16, 17, 31, 32, 33, 64, 95, 128, 257] {
        let w = rand_vec(len, 400 + len as u64);
        let x = rand_vec(len, 500 + len as u64);
        let wq = quantize_q4(&w);
        let got = dot_q4_f32(&wq, &x);
        let want = dot_q4_f32_scalar(&wq, &x);
        let tol = 1e-5 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "dot_q4_f32 len={len}: got={got} want={want} tol={tol}");
    }

    // Test GEMV (both f32 and q8 activations) and Matmul across serial & parallel sizes.
    for &(m, n, k) in &[(1usize, 1usize, 1usize), (3, 7, 35), (4, 13, 64), (8, 67, 131), (16, 256, 160)] {
        let a = rand_vec(m * k, (m * 1000 + n * 10 + k) as u64);
        let b = rand_vec(n * k, (m * 2000 + n * 17 + k) as u64);
        let bias = rand_vec(n, (n * 31 + k) as u64);
        let bq = Q4Matrix::quantize(&b, n, k);
        let bdq = bq.dequantize();
        assert_eq!(bdq.len(), n * k);

        // Single-row GEMV check against dequantized f32 reference.
        let mut y_gemv = vec![0.0f32; n];
        gemv_q4_f32(&bq, &a[..k], &mut y_gemv);
        let mut y_gemv_q8 = vec![0.0f32; n];
        gemv_q4_q8(&bq, &a[..k], &mut y_gemv_q8);
        for j in 0..n {
            let ref_dot: f32 = a[..k].iter().zip(&bdq[j * k..(j + 1) * k]).map(|(&u, &v)| u * v).sum();
            let tol = 1e-4 * ref_dot.abs().max(1.0);
            assert!(
                (y_gemv[j] - ref_dot).abs() <= tol,
                "gemv_q4_f32 ({m},{n},{k}) row={j}: got={} want={ref_dot}",
                y_gemv[j]
            );
            // Q8 activation quantization error stays small relative to vector norm.
            let norm_a: f32 = a[..k].iter().map(|v| v * v).sum::<f32>().sqrt();
            let norm_b: f32 = bdq[j * k..(j + 1) * k].iter().map(|v| v * v).sum::<f32>().sqrt();
            let q8_tol = 0.05 * norm_a * norm_b + 1e-3;
            assert!(
                (y_gemv_q8[j] - ref_dot).abs() <= q8_tol,
                "gemv_q4_q8 ({m},{n},{k}) row={j}: got={} want={ref_dot} tol={q8_tol}",
                y_gemv_q8[j]
            );
        }

        // Full matmul with and without bias.
        for use_bias in [false, true] {
            let mut out = vec![0.0f32; m * n];
            matmul_q4_f32(&a, m, k, &bq, use_bias.then_some(bias.as_slice()), &mut out);
            for i in 0..m {
                for j in 0..n {
                    let mut want: f32 = a[i * k..(i + 1) * k]
                        .iter()
                        .zip(&bdq[j * k..(j + 1) * k])
                        .map(|(&u, &v)| u * v)
                        .sum();
                    if use_bias {
                        want += bias[j];
                    }
                    let got = out[i * n + j];
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tol,
                        "matmul_q4_f32 ({m},{n},{k}) bias={use_bias} [{i},{j}]: got={got} want={want}"
                    );
                }
            }
        }
    }

    // Zero-dimension edge cases.
    let empty = Q4Matrix::quantize(&[], 0, 0);
    assert!(empty.dequantize().is_empty());
    let mut out = Vec::new();
    gemv_q4_f32(&empty, &[], &mut out);
    matmul_q4_f32(&[], 0, 0, &empty, None, &mut out);

    let zero_k = Q4Matrix::quantize(&[], 3, 0);
    let mut out_zk = vec![1.0; 6];
    let bias_zk = [1.5, -2.0, 3.25];
    matmul_q4_f32(&[], 2, 0, &zero_k, Some(&bias_zk), &mut out_zk);
    assert_eq!(out_zk, vec![1.5, -2.0, 3.25, 1.5, -2.0, 3.25]);
}
