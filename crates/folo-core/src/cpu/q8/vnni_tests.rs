//! Bit-exactness and edge-case tests for x86-64 VNNI (512-bit AVX-512 VNNI, 256-bit AVX-VNNI, AVX-512 VL,
//! and AVX2) microkernels, 2× K-unrolling boundaries, `VECS` tail specialization, and odd spatial widths.

use super::conv::{QConv, QOut, conv, conv_per_channel};
use super::kernel::{MR, NR, Tile, kernel_scalar};
use super::silu::silu_q8;
use super::stem::{QStem, stem_u8};
use super::{C4, Grid, SLACK, alloc, quantize, quantize_per_channel, recycle};

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}

/// Independent scalar reference directly evaluating `cv`'s packed `QPart`s (`i64` dot sums + exact FMA epilogue).
#[allow(clippy::too_many_arguments)]
fn conv_ref_packed(
    xs: &[C4<'_>],
    n: usize,
    h: usize,
    w: usize,
    cv: &QConv,
    res: Option<(C4<'_>, &[Grid])>,
    out_grids: Option<&[Grid]>,
) -> (Vec<u32>, Vec<f32>) {
    let (o, c, k, stride, pad) = (cv.o, cv.c, cv.k, cv.stride, cv.pad);
    let (oh, ow) = cv.out_hw(h, w);
    let hw = h * w;
    let ohw = oh * ow;
    let og = o / 4;
    let groups = c / 4;
    let taps = k * k;
    let mut out_u8 = vec![0u32; n * og * ohw];
    let mut out_f32 = vec![0.0f32; n * o * ohw];

    // Flatten C4 words `[n][groups][hw]` for simple indexing.
    let mut x_words = vec![0u32; n * groups * hw];
    for img in 0..n {
        let mut g0 = 0;
        for x in xs {
            let base = x.offset + img * x.img_stride;
            let xg = x.c / 4;
            for lg in 0..xg {
                x_words[(img * groups + g0 + lg) * hw..(img * groups + g0 + lg + 1) * hw]
                    .copy_from_slice(&x.data[base + lg * hw..base + (lg + 1) * hw]);
            }
            g0 += xg;
        }
    }
    let zero_word = |g: usize| {
        u32::from_le_bytes([cv.zeros[4 * g], cv.zeros[4 * g + 1], cv.zeros[4 * g + 2], cv.zeros[4 * g + 3]])
    };

    for img in 0..n {
        for g in 0..og {
            for oy in 0..oh {
                for ox in 0..ow {
                    let px = oy * ow + ox;
                    let mut word = 0u32;
                    for r in 0..MR {
                        let oc = 4 * g + r;
                        let mut y = if cv.parts.len() == 1 {
                            let part = &cv.parts[0];
                            let mut acc = 0i64;
                            for ic_g in 0..groups {
                                let zw = zero_word(ic_g);
                                for ky in 0..k {
                                    let iy = (oy * stride + ky) as isize - pad as isize;
                                    for kx in 0..k {
                                        let ix = (ox * stride + kx) as isize - pad as isize;
                                        let xw = if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                            zw
                                        } else {
                                            x_words
                                                [(img * groups + ic_g) * hw + iy as usize * w + ix as usize]
                                        };
                                        let p = ic_g * taps + ky * k + kx;
                                        let xb = xw.to_le_bytes();
                                        let wb = part.a[(g * part.ks + p) * MR + r].to_le_bytes();
                                        for t in 0..4 {
                                            acc += xb[t] as i64 * wb[t] as i8 as i64;
                                        }
                                    }
                                }
                            }
                            super::silu::fmadd(
                                (acc as i32).wrapping_sub(part.corr[oc]) as f32,
                                part.scale[oc],
                                cv.bias[oc],
                            )
                        } else {
                            let mut sum = 0.0f32;
                            for part in &cv.parts {
                                let g_start = part.row_offset / taps;
                                let g_cnt = part.ks / taps;
                                let mut acc = 0i64;
                                for lg in 0..g_cnt {
                                    let ic_g = g_start + lg;
                                    let zw = zero_word(ic_g);
                                    for ky in 0..k {
                                        let iy = (oy * stride + ky) as isize - pad as isize;
                                        for kx in 0..k {
                                            let ix = (ox * stride + kx) as isize - pad as isize;
                                            let xw =
                                                if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                                    zw
                                                } else {
                                                    x_words[(img * groups + ic_g) * hw
                                                        + iy as usize * w
                                                        + ix as usize]
                                                };
                                            let p = lg * taps + ky * k + kx;
                                            let xb = xw.to_le_bytes();
                                            let wb = part.a[(g * part.ks + p) * MR + r].to_le_bytes();
                                            for t in 0..4 {
                                                acc += xb[t] as i64 * wb[t] as i8 as i64;
                                            }
                                        }
                                    }
                                }
                                sum = super::silu::fmadd(
                                    (acc as i32).wrapping_sub(part.corr[oc]) as f32,
                                    part.scale[oc],
                                    sum,
                                );
                            }
                            sum + cv.bias[oc]
                        };
                        if cv.silu {
                            y = silu_q8(y);
                        }
                        if let Some((rv, rg)) = &res {
                            let rw = rv.data[rv.offset + img * rv.img_stride + g * ohw + px];
                            let rq = ((rw >> (8 * r)) & 0xFF) as i32;
                            y = super::silu::fmadd(rg[oc].scale, (rq - rg[oc].zero as i32) as f32, y);
                        }
                        if let Some(ogs) = out_grids {
                            word |= (ogs[oc].quantize(y) as u32) << (8 * r);
                        } else {
                            out_f32[(img * o + oc) * ohw + px] = y;
                        }
                    }
                    if out_grids.is_some() {
                        out_u8[(img * og + g) * ohw + px] = word;
                    }
                }
            }
        }
    }
    (out_u8, out_f32)
}

#[test]
fn microkernels_bit_exact_across_ks_and_extreme_values() {
    let mut seed = 0xD15EA5E_u64;
    let ks_list = [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65];
    for &ks in &ks_list {
        for mode in 0..4 {
            let mut a = vec![0i32; ks * MR];
            let mut b_buf = vec![0u32; ks * (NR + 8)];
            for w in &mut a {
                let bytes: [u8; 4] = match mode {
                    0 => [(-128i8) as u8; 4],
                    1 => [127u8; 4],
                    2 => [(-128i8) as u8, 127u8, 0u8, (-1i8) as u8],
                    _ => {
                        let r = lcg(&mut seed);
                        [(r >> 8) as u8, (r >> 16) as u8, (r >> 24) as u8, (r >> 32) as u8]
                    }
                };
                *w = i32::from_le_bytes(bytes);
            }
            for x in &mut b_buf {
                let bytes: [u8; 4] = match mode {
                    0 | 1 => [255u8; 4],
                    2 => [255u8, 0u8, 128u8, 1u8],
                    _ => {
                        let r = lcg(&mut seed);
                        [(r >> 8) as u8, (r >> 16) as u8, (r >> 24) as u8, (r >> 32) as u8]
                    }
                };
                *x = u32::from_le_bytes(bytes);
            }
            let rows: Vec<*const u32> = (0..ks).map(|p| b_buf[p * (NR + 8)..].as_ptr()).collect();
            let col = 3;
            let mut expected: Tile = [[0; NR]; MR];
            // SAFETY: `a` has `ks * MR` words; each `rows[p]` has `NR + 8 >= col + NR` words.
            unsafe { kernel_scalar(ks, a.as_ptr(), rows.as_ptr(), col, &mut expected) };

            let mut active: Tile = [[0; NR]; MR];
            // SAFETY: as above.
            unsafe { (super::kernel::kernel())(ks, a.as_ptr(), rows.as_ptr(), col, &mut active) };
            assert_eq!(active, expected, "active kernel mismatch at ks={ks} mode={mode}");

            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    let mut got: Tile = [[0; NR]; MR];
                    // SAFETY: avx2 detected; pointer bounds valid.
                    unsafe { super::kernel::kernel_avx2(ks, a.as_ptr(), rows.as_ptr(), col, &mut got) };
                    assert_eq!(got, expected, "kernel_avx2 mismatch at ks={ks} mode={mode}");
                }
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni") {
                    let mut got: Tile = [[0; NR]; MR];
                    // SAFETY: avx2+avxvnni detected; pointer bounds valid.
                    unsafe { super::kernel::kernel_vnni(ks, a.as_ptr(), rows.as_ptr(), col, &mut got) };
                    assert_eq!(got, expected, "kernel_vnni mismatch at ks={ks} mode={mode}");
                }
                if std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("avx512vnni")
                    && std::is_x86_feature_detected!("avx512vl")
                {
                    let mut got: Tile = [[0; NR]; MR];
                    // SAFETY: avx512vnni+avx512vl detected; pointer bounds valid.
                    unsafe { super::kernel::kernel_avx512vnni(ks, a.as_ptr(), rows.as_ptr(), col, &mut got) };
                    assert_eq!(got, expected, "kernel_avx512vnni mismatch at ks={ks} mode={mode}");
                }
                if std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma")
                    && std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl")
                    && std::is_x86_feature_detected!("avx512vnni")
                {
                    let mut got: Tile = [[0; NR]; MR];
                    // SAFETY: full AVX-512 VNNI detected; pointer bounds valid.
                    unsafe {
                        super::vnni512::kernel_avx512vnni_zmm(ks, a.as_ptr(), rows.as_ptr(), col, &mut got)
                    };
                    assert_eq!(got, expected, "kernel_avx512vnni_zmm mismatch at ks={ks} mode={mode}");
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn silu_avx512_matches_scalar_bit_for_bit() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }
    let mut xs = vec![
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        85.0,
        -85.0,
        86.0,
        -86.0,
        1e6,
        -1e6,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::NAN,
        -f32::NAN,
    ];
    for i in -1000..=1000 {
        xs.push(i as f32 * 0.1);
    }
    while !xs.len().is_multiple_of(32) {
        xs.push(0.25);
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn eval32(src: &[f32]) -> [f32; 32] {
        use std::arch::x86_64::*;
        let mut out = [0.0f32; 32];
        // SAFETY: `src.len() >= 32` and `out.len() == 32`.
        unsafe {
            let v0 = _mm512_loadu_ps(src.as_ptr());
            let v1 = _mm512_loadu_ps(src.as_ptr().add(16));
            let [r0, r1] = super::vnni512::silu_q8_avx512_x2([v0, v1]);
            _mm512_storeu_ps(out.as_mut_ptr(), r0);
            _mm512_storeu_ps(out.as_mut_ptr().add(16), r1);
        }
        out
    }
    for chunk in xs.as_chunks::<32>().0 {
        // SAFETY: avx512f detected above.
        let got = unsafe { eval32(chunk) };
        for (i, &x) in chunk.iter().enumerate() {
            let exp = silu_q8(x);
            if x.is_nan() {
                assert!(got[i].is_nan(), "silu_q8_avx512_x2 lost NaN at lane {i}");
                assert!(exp.is_nan());
            } else {
                assert_eq!(got[i].to_bits(), exp.to_bits(), "silu_q8_avx512_x2 mismatch at x={x}");
            }
        }
    }
}

#[test]
fn conv_1x1_and_3x3_and_5x5_odd_tail_shapes_bit_exact() {
    let mut seed = 0x516_C0DE_u64;
    let mut next_f32 = |lo: f32, hi: f32| {
        let u = ((lcg(&mut seed) >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };

    // 1x1 single-part across every mod-8 / mod-16 / mod-24 / mod-48 tail boundary and odd/even `ks`.
    let shapes_1x1: &[(usize, usize, usize, usize, bool, bool, bool)] = &[
        (1, 1, 4, 4, true, false, true),
        (1, 7, 12, 8, false, true, true),
        (1, 8, 8, 8, true, true, true),
        (1, 9, 12, 4, true, false, false),
        (1, 15, 20, 8, false, true, true),
        (1, 16, 16, 12, true, true, true),
        (1, 17, 4, 8, true, false, true),
        (1, 23, 12, 8, false, false, false),
        (1, 24, 12, 8, true, true, true),
        (1, 25, 8, 8, true, false, true),
        (1, 31, 12, 8, true, true, false),
        (1, 32, 12, 8, true, true, true),
        (1, 47, 12, 8, false, true, true),
        (1, 48, 12, 8, true, false, true),
        (1, 49, 20, 8, true, true, true),
        (5, 5, 12, 8, true, true, true),
        (20, 20, 12, 16, true, true, true),
    ];
    for &(h, w, c, o, silu, has_res, out_u8) in shapes_1x1 {
        let n = 1;
        let hw = h * w;
        let x_grid = Grid::from_range(-2.0, 3.0);
        let res_grid = Grid::from_range(-1.5, 1.5);
        let out_grid = Grid::from_range(-3.0, 4.0);
        let xf: Vec<f32> = (0..n * c * hw).map(|_| next_f32(-2.0, 3.0)).collect();
        let mut xb = alloc(n, c, hw);
        quantize(&xf, n, c, hw, x_grid, &mut xb);
        let xv = C4 { data: &xb, offset: 0, img_stride: (c / 4) * hw, c, zero: x_grid.zero };

        let rf: Vec<f32> = (0..n * o * hw).map(|_| next_f32(-1.5, 1.5)).collect();
        let mut rb = alloc(n, o, hw);
        quantize(&rf, n, o, hw, res_grid, &mut rb);
        let rv = C4 { data: &rb, offset: 0, img_stride: (o / 4) * hw, c: o, zero: res_grid.zero };

        let wf: Vec<f32> = (0..o * c).map(|_| next_f32(-1.0, 1.0)).collect();
        let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.5, 0.5)).collect();
        let x_grids = vec![x_grid; c];
        let res_grids = vec![res_grid; o];
        let out_grids = vec![out_grid; o];
        let cv = QConv::new(&wf, &bias, (o, c, 1), 1, 0, silu, &x_grids);

        let (ref_u8, ref_f32) = conv_ref_packed(
            &[xv],
            n,
            h,
            w,
            &cv,
            has_res.then_some((rv, res_grids.as_slice())),
            out_u8.then_some(out_grids.as_slice()),
        );
        if out_u8 {
            let mut got = alloc(n, o, hw);
            let out_len = n * (o / 4) * hw;
            conv(
                &[xv],
                n,
                h,
                w,
                &cv,
                has_res.then_some((rv, res_grid)),
                QOut::U8 { data: &mut got, grid: out_grid },
            );
            assert_eq!(
                &got[..out_len],
                &ref_u8[..],
                "1x1 u8 mismatch at {h}x{w} c={c} o={o} silu={silu} res={has_res}"
            );
            recycle(got);
        } else {
            let mut got = vec![0.0f32; n * o * hw];
            conv(&[xv], n, h, w, &cv, has_res.then_some((rv, res_grid)), QOut::F32 { data: &mut got });
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "1x1 f32 mismatch at {h}x{w} c={c} o={o} silu={silu} res={has_res}"
            );
        }
        recycle(xb);
        recycle(rb);
    }

    // 1x1 multi-part (both coupled and decoupled) + per-channel output grids across tail shapes.
    for &(h, w) in &[(1, 7), (1, 16), (1, 25), (1, 48), (1, 53), (4, 5)] {
        for decouple in [false, true] {
            let (c0, c1, o) = (8, 12, 8);
            let c = c0 + c1;
            let hw = h * w;
            let g0 = Grid::from_range(-1.0, 2.0);
            let g1 = Grid::from_range(-4.0, 1.0);
            let pg0 = vec![g0; c0];
            let pg1 = vec![g1; c1];
            let z0 = vec![g0.zero; c0];
            let z1 = vec![g1.zero; c1];
            let out_grids: Vec<Grid> = (0..o).map(|i| Grid::from_range(-2.0 - 0.1 * i as f32, 3.0)).collect();
            let x0f: Vec<f32> = (0..c0 * hw).map(|_| next_f32(-1.0, 2.0)).collect();
            let x1f: Vec<f32> = (0..c1 * hw).map(|_| next_f32(-4.0, 1.0)).collect();
            let mut x0b = alloc(1, c0, hw);
            let mut x1b = alloc(1, c1, hw);
            quantize(&x0f, 1, c0, hw, g0, &mut x0b);
            quantize(&x1f, 1, c1, hw, g1, &mut x1b);
            let x0v = C4 { data: &x0b, offset: 0, img_stride: (c0 / 4) * hw, c: c0, zero: g0.zero };
            let x1v = C4 { data: &x1b, offset: 0, img_stride: (c1 / 4) * hw, c: c1, zero: g1.zero };
            let wf: Vec<f32> = (0..o * c).map(|_| next_f32(-0.8, 0.8)).collect();
            let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.5, 0.5)).collect();
            let cv = QConv::new_parts(&wf, &bias, (o, c, 1), 1, 0, true, &[&pg0, &pg1], decouple);
            let (ref_u8, _) = conv_ref_packed(&[x0v, x1v], 1, h, w, &cv, None, Some(&out_grids));
            let mut got = alloc(1, o, hw);
            conv_per_channel(
                &[x0v, x1v],
                &[&z0, &z1],
                1,
                h,
                w,
                &cv,
                None,
                QOut::U8PerChannel { data: &mut got, grids: &out_grids },
            );
            assert_eq!(
                &got[..(o / 4) * hw],
                &ref_u8[..],
                "1x1 multi-part (decouple={decouple}) mismatch at {h}x{w}"
            );
            recycle(x0b);
            recycle(x1b);
            recycle(got);
        }
    }

    // 3x3 (s=1 & s=2, pad=1) and 5x5 (s=1, pad=2) across odd/tail widths hitting both fused_3x3 and Rows.
    let spatial_cases: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
        (5, 7, 8, 8, 3, 2, 1),
        (5, 16, 12, 8, 3, 2, 1),
        (5, 32, 8, 8, 3, 2, 1),
        (5, 48, 8, 8, 3, 2, 1),
        (5, 50, 8, 8, 3, 2, 1),
        (5, 98, 8, 8, 3, 2, 1),
        (9, 17, 12, 8, 3, 1, 1),
        (9, 49, 8, 8, 3, 1, 1),
        (7, 25, 8, 8, 5, 1, 2),
        (7, 51, 8, 8, 5, 1, 2),
    ];
    for &(h, w, c, o, k, stride, pad) in spatial_cases {
        let hw = h * w;
        let oh = (h + 2 * pad - k) / stride + 1;
        let ow = (w + 2 * pad - k) / stride + 1;
        let ohw = oh * ow;
        let x_grid = Grid::from_range(-1.5, 2.5);
        let res_grid = Grid::from_range(-1.0, 1.0);
        let out_grid = Grid::from_range(-2.5, 3.5);
        let xf: Vec<f32> = (0..c * hw).map(|_| next_f32(-1.5, 2.5)).collect();
        let mut xb = alloc(1, c, hw);
        quantize(&xf, 1, c, hw, x_grid, &mut xb);
        let xv = C4 { data: &xb, offset: 0, img_stride: (c / 4) * hw, c, zero: x_grid.zero };
        let rf: Vec<f32> = (0..o * ohw).map(|_| next_f32(-1.0, 1.0)).collect();
        let mut rb = alloc(1, o, ohw);
        quantize(&rf, 1, o, ohw, res_grid, &mut rb);
        let rv = C4 { data: &rb, offset: 0, img_stride: (o / 4) * ohw, c: o, zero: res_grid.zero };
        let wf: Vec<f32> = (0..o * c * k * k).map(|_| next_f32(-0.7, 0.7)).collect();
        let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.3, 0.3)).collect();
        let x_grids = vec![x_grid; c];
        let res_grids = vec![res_grid; o];
        let out_grids = vec![out_grid; o];
        let cv = QConv::new(&wf, &bias, (o, c, k), stride, pad, true, &x_grids);
        let (ref_u8, _) = conv_ref_packed(&[xv], 1, h, w, &cv, Some((rv, &res_grids)), Some(&out_grids));
        let mut got = alloc(1, o, ohw);
        conv(&[xv], 1, h, w, &cv, Some((rv, res_grid)), QOut::U8 { data: &mut got, grid: out_grid });
        assert_eq!(&got[..(o / 4) * ohw], &ref_u8[..], "{k}x{k} s={stride} p={pad} mismatch at {h}x{w}");
        recycle(xb);
        recycle(rb);
        recycle(got);
    }
}

#[test]
fn stem_u8_odd_and_48_boundary_widths_bit_exact() {
    let mut seed = 0x57E4_0048_u64;
    let mut next_f32 = |lo: f32, hi: f32| {
        let u = ((lcg(&mut seed) >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };
    let o = 16;
    let wf: Vec<f32> = (0..o * 3 * 9).map(|_| next_f32(-0.8, 0.8)).collect();
    let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.5, 0.5)).collect();
    let stem = QStem::new(&wf, &bias, (o, 3), 2, 1, true);
    let out_grid = Grid::from_range(-2.0, 4.0);

    // Output widths covering every mod-8, mod-16, mod-24, and mod-48 boundary (including ow = 24 and ow = 48).
    for &ow in &[1usize, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 47, 48, 49, 64, 80, 97] {
        let (h, w) = (6usize, 2 * ow);
        let oh = (h + 2 - 3) / 2 + 1;
        let ohw = oh * ow;
        let x_rgb0: Vec<u32> = (0..h * w)
            .map(|_| {
                let r = lcg(&mut seed);
                u32::from_le_bytes([(r >> 8) as u8, (r >> 16) as u8, (r >> 24) as u8, 0])
            })
            .collect();
        let mut got = alloc(1, o, ohw);
        stem_u8(&x_rgb0, 1, h, w, &stem, QOut::U8 { data: &mut got, grid: out_grid });

        let mut expected = vec![0u32; (o / 4) * ohw];
        for g in 0..o / 4 {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut word = 0u32;
                    for r in 0..MR {
                        let oc = 4 * g + r;
                        let mut acc = 0i32;
                        for ky in 0..3 {
                            let iy = (2 * oy + ky) as isize - 1;
                            for kx in 0..3 {
                                let ix = (2 * ox + kx) as isize - 1;
                                let px = if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                    [0u8; 4]
                                } else {
                                    x_rgb0[iy as usize * w + ix as usize].to_le_bytes()
                                };
                                let tap = ky * 3 + kx;
                                let wb = stem.a[(g * 9 + tap) * MR + r].to_le_bytes();
                                for t in 0..4 {
                                    acc += px[t] as i32 * wb[t] as i8 as i32;
                                }
                            }
                        }
                        let y = silu_q8(super::silu::fmadd(acc as f32, stem.scale[oc], stem.bias[oc]));
                        word |= (out_grid.quantize(y) as u32) << (8 * r);
                    }
                    expected[g * ohw + oy * ow + ox] = word;
                }
            }
        }
        assert_eq!(&got[..(o / 4) * ohw], &expected[..], "stem_u8 mismatch at ow={ow} (w={w})");
        recycle(got);
    }
}

#[test]
fn per_channel_large_3x3_s1_hits_rotating_strip_bit_exact() {
    // `c * oh * (ow + 2) > 384 KiB` triggers `fused_3x3` stride-1 rotating strip path.
    let (h, w, c, o) = (32, 32, 400, 8);
    let hw = h * w;
    let mut seed = 0x33_51_u64;
    let mut next_f32 = |lo: f32, hi: f32| {
        let u = ((lcg(&mut seed) >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };
    let x_grids: Vec<Grid> = (0..c).map(|i| Grid::from_range(-1.0 - (i % 4) as f32 * 0.2, 2.0)).collect();
    let x_zeros: Vec<u8> = x_grids.iter().map(|g| g.zero).collect();
    let out_grids: Vec<Grid> = (0..o).map(|i| Grid::from_range(-3.0, 3.0 + i as f32 * 0.1)).collect();
    let xf: Vec<f32> = (0..c * hw).map(|_| next_f32(-1.0, 2.0)).collect();
    let mut xb = vec![0u32; (c / 4) * hw + SLACK];
    quantize_per_channel(&xf, 1, c, hw, &x_grids, &mut xb);
    let xv = C4 { data: &xb, offset: 0, img_stride: (c / 4) * hw, c, zero: 0 };
    let wf: Vec<f32> = (0..o * c * 9).map(|_| next_f32(-0.5, 0.5)).collect();
    let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.2, 0.2)).collect();
    let cv = QConv::new(&wf, &bias, (o, c, 3), 1, 1, true, &x_grids);
    let mut got = alloc(1, o, hw);
    conv_per_channel(
        &[xv],
        &[&x_zeros],
        1,
        h,
        w,
        &cv,
        None,
        QOut::U8PerChannel { data: &mut got, grids: &out_grids },
    );

    let p0 = &cv.parts[0];
    let test_pixels = [(0, 0), (0, 15), (0, 16), (0, 23), (0, 24), (0, 31), (15, 17), (31, 0), (31, 31)];
    for &(oy, ox) in &test_pixels {
        for g in 0..o / 4 {
            let mut acc_i32 = [0i32; MR];
            for ic_g in 0..c / 4 {
                for ky in 0..3 {
                    let iy = (oy + ky) as isize - 1;
                    for kx in 0..3 {
                        let ix = (ox + kx) as isize - 1;
                        let xw = if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                            u32::from_le_bytes([
                                x_grids[4 * ic_g].zero,
                                x_grids[4 * ic_g + 1].zero,
                                x_grids[4 * ic_g + 2].zero,
                                x_grids[4 * ic_g + 3].zero,
                            ])
                        } else {
                            xb[ic_g * hw + iy as usize * w + ix as usize]
                        };
                        let xbytes = xw.to_le_bytes();
                        let p = (ic_g * 3 + ky) * 3 + kx;
                        for r in 0..MR {
                            let wbytes = p0.a[(g * p0.ks + p) * MR + r].to_le_bytes();
                            for t in 0..4 {
                                acc_i32[r] += xbytes[t] as i32 * wbytes[t] as i8 as i32;
                            }
                        }
                    }
                }
            }
            let mut word = 0u32;
            for (r, &acc) in acc_i32.iter().enumerate() {
                let oc = 4 * g + r;
                let y = silu_q8(super::silu::fmadd((acc - p0.corr[oc]) as f32, p0.scale[oc], cv.bias[oc]));
                word |= (out_grids[oc].quantize(y) as u32) << (8 * r);
            }
            assert_eq!(got[g * hw + oy * w + ox], word, "large 3x3 s1 mismatch at ({oy}, {ox}) g={g}");
        }
    }
    recycle(got);
}

#[test]
fn zero_channel_concat_and_decoupled_padded_wide_multipart_bit_exact() {
    let mut seed = 0x00C4_0097_u64;
    let mut next_f32 = |lo: f32, hi: f32| {
        let u = ((lcg(&mut seed) >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };

    // 1) Zero-channel C4 view backed by exactly `SLACK` words concatenated with a 4-channel view on
    // `hw > SLACK` (1x97 and 1x104, hitting col > SLACK and 1..=8 narrow tails) for coupled & decoupled 1x1.
    for &(h, w) in &[(1usize, 97usize), (1, 104)] {
        let hw = h * w;
        let (c0, c1, o) = (0usize, 4usize, 8usize);
        let g0 = Grid::from_range(-1.0, 1.0);
        let g1 = Grid::from_range(-2.0, 3.0);
        let out_grid = Grid::from_range(-3.0, 4.0);
        let empty_buf = vec![0u32; SLACK];
        let x0v = C4 { data: &empty_buf, offset: 0, img_stride: 0, c: c0, zero: g0.zero };
        let x1f: Vec<f32> = (0..c1 * hw).map(|_| next_f32(-2.0, 3.0)).collect();
        let mut x1b = alloc(1, c1, hw);
        quantize(&x1f, 1, c1, hw, g1, &mut x1b);
        let x1v = C4 { data: &x1b, offset: 0, img_stride: (c1 / 4) * hw, c: c1, zero: g1.zero };
        let wf: Vec<f32> = (0..o * c1).map(|_| next_f32(-0.8, 0.8)).collect();
        let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.4, 0.4)).collect();
        let pg0: [Grid; 0] = [];
        let pg1 = vec![g1; c1];
        let out_grids = vec![out_grid; o];
        for decouple in [false, true] {
            let cv = QConv::new_parts(&wf, &bias, (o, c1, 1), 1, 0, true, &[&pg0, &pg1], decouple);
            let (ref_u8, _) = conv_ref_packed(&[x0v, x1v], 1, h, w, &cv, None, Some(&out_grids));
            let mut got = alloc(1, o, hw);
            conv(&[x0v, x1v], 1, h, w, &cv, None, QOut::U8 { data: &mut got, grid: out_grid });
            assert_eq!(
                &got[..(o / 4) * hw],
                &ref_u8[..],
                "zero-channel concat 1x1 (decouple={decouple}) mismatch at {h}x{w}"
            );
            recycle(got);
        }
        recycle(x1b);
    }

    // 2) Decoupled padded-width multi-part 3x3 (ow = 52 >= 48, `t.wq != t.ow`) with residuals (U8 & F32).
    let (h, w, c0, c1, o) = (4usize, 52usize, 8usize, 8usize, 8usize);
    let c = c0 + c1;
    let hw = h * w;
    let g0 = Grid::from_range(-1.5, 2.0);
    let g1 = Grid::from_range(-5.0, 1.5);
    let res_grid = Grid::from_range(-1.0, 1.0);
    let out_grid = Grid::from_range(-3.0, 4.0);
    let pg0 = vec![g0; c0];
    let pg1 = vec![g1; c1];
    let res_grids = vec![res_grid; o];
    let out_grids = vec![out_grid; o];
    let x0f: Vec<f32> = (0..c0 * hw).map(|_| next_f32(-1.5, 2.0)).collect();
    let x1f: Vec<f32> = (0..c1 * hw).map(|_| next_f32(-5.0, 1.5)).collect();
    let rf: Vec<f32> = (0..o * hw).map(|_| next_f32(-1.0, 1.0)).collect();
    let mut x0b = alloc(1, c0, hw);
    let mut x1b = alloc(1, c1, hw);
    let mut rb = alloc(1, o, hw);
    quantize(&x0f, 1, c0, hw, g0, &mut x0b);
    quantize(&x1f, 1, c1, hw, g1, &mut x1b);
    quantize(&rf, 1, o, hw, res_grid, &mut rb);
    let x0v = C4 { data: &x0b, offset: 0, img_stride: (c0 / 4) * hw, c: c0, zero: g0.zero };
    let x1v = C4 { data: &x1b, offset: 0, img_stride: (c1 / 4) * hw, c: c1, zero: g1.zero };
    let rv = C4 { data: &rb, offset: 0, img_stride: (o / 4) * hw, c: o, zero: res_grid.zero };
    let wf: Vec<f32> = (0..o * c * 9).map(|_| next_f32(-0.6, 0.6)).collect();
    let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.3, 0.3)).collect();
    let cv = QConv::new_parts(&wf, &bias, (o, c, 3), 1, 1, true, &[&pg0, &pg1], true);

    let (ref_u8, _) = conv_ref_packed(&[x0v, x1v], 1, h, w, &cv, Some((rv, &res_grids)), Some(&out_grids));
    let mut got_u8 = alloc(1, o, hw);
    conv(&[x0v, x1v], 1, h, w, &cv, Some((rv, res_grid)), QOut::U8 { data: &mut got_u8, grid: out_grid });
    assert_eq!(&got_u8[..(o / 4) * hw], &ref_u8[..], "decoupled padded 3x3 u8 mismatch");
    recycle(got_u8);

    let (_, ref_f32) = conv_ref_packed(&[x0v, x1v], 1, h, w, &cv, Some((rv, &res_grids)), None);
    let mut got_f32 = vec![0.0f32; o * hw];
    conv(&[x0v, x1v], 1, h, w, &cv, Some((rv, res_grid)), QOut::F32 { data: &mut got_f32 });
    assert_eq!(
        got_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ref_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "decoupled padded 3x3 f32 mismatch"
    );

    recycle(x0b);
    recycle(x1b);
    recycle(rb);
}

#[test]
fn kernel_name_and_uses_vnni_consistent() {
    let name = super::kernel_name();
    let vnni = super::uses_vnni();
    assert_eq!(
        vnni,
        matches!(name, "AVX-512 VNNI (512-bit)" | "AVX-VNNI" | "AVX-512 VNNI"),
        "kernel_name={name} disagrees with uses_vnni={vnni}"
    );
}

#[test]
fn packed_1x1_direct_and_multipart_bit_exact_across_tails_and_outputs() {
    let mut seed = 0x9AC4_1011_u64;
    let mut next_f32 = |lo: f32, hi: f32| {
        let u = ((lcg(&mut seed) >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };
    // Exercises `fused_pack::should_pack_1x1`: `ks >= 12`, `og >= 4`, both `hw >= 1024` (with odd tails
    // hitting VECS ∈ {1, 2, 3} and exact-8/16 direct tails) and `hw < 1024` with `ks >= 48`, single-part
    // and multi-input `Pointwise1x1Parts`, U8 + F32 direct stores, SiLU + identity, with & without residual.
    struct PackedCase {
        h: usize,
        w: usize,
        parts: &'static [usize],
        o: usize,
        silu: bool,
        has_res: bool,
        out_u8: bool,
    }
    let cases = [
        PackedCase { h: 32, w: 33, parts: &[48], o: 16, silu: true, has_res: false, out_u8: true },
        PackedCase { h: 33, w: 33, parts: &[48], o: 16, silu: true, has_res: true, out_u8: true },
        PackedCase { h: 32, w: 34, parts: &[24, 24], o: 20, silu: false, has_res: true, out_u8: false },
        PackedCase { h: 31, w: 35, parts: &[28, 20, 4], o: 16, silu: true, has_res: false, out_u8: false },
        PackedCase { h: 32, w: 35, parts: &[48], o: 16, silu: false, has_res: false, out_u8: true },
        PackedCase { h: 33, w: 35, parts: &[48], o: 16, silu: true, has_res: true, out_u8: true },
        PackedCase { h: 20, w: 20, parts: &[192], o: 16, silu: true, has_res: true, out_u8: true },
        PackedCase { h: 17, w: 19, parts: &[96, 96], o: 16, silu: true, has_res: false, out_u8: false },
    ];
    for &PackedCase { h, w, parts, o, silu, has_res, out_u8 } in &cases {
        let hw = h * w;
        let c: usize = parts.iter().sum();
        let x_grid = Grid::from_range(-2.0, 3.0);
        let res_grid = Grid::from_range(-1.5, 1.5);
        let out_grid = Grid::from_range(-3.0, 4.0);
        let mut x_bufs = Vec::with_capacity(parts.len());
        for &pc in parts {
            let xf: Vec<f32> = (0..pc * hw).map(|_| next_f32(-2.0, 3.0)).collect();
            let mut xb = alloc(1, pc, hw);
            quantize(&xf, 1, pc, hw, x_grid, &mut xb);
            x_bufs.push(xb);
        }
        let xvs: Vec<C4<'_>> = x_bufs
            .iter()
            .zip(parts)
            .map(|(xb, &pc)| C4 { data: xb, offset: 0, img_stride: (pc / 4) * hw, c: pc, zero: x_grid.zero })
            .collect();
        let rf: Vec<f32> = (0..o * hw).map(|_| next_f32(-1.5, 1.5)).collect();
        let mut rb = alloc(1, o, hw);
        quantize(&rf, 1, o, hw, res_grid, &mut rb);
        let rv = C4 { data: &rb, offset: 0, img_stride: (o / 4) * hw, c: o, zero: res_grid.zero };
        let wf: Vec<f32> = (0..o * c).map(|_| next_f32(-0.6, 0.6)).collect();
        let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.4, 0.4)).collect();
        let x_grids = vec![x_grid; c];
        let res_grids = vec![res_grid; o];
        let out_grids = vec![out_grid; o];
        let cv = QConv::new(&wf, &bias, (o, c, 1), 1, 0, silu, &x_grids);
        let (ref_u8, ref_f32) = conv_ref_packed(
            &xvs,
            1,
            h,
            w,
            &cv,
            has_res.then_some((rv, res_grids.as_slice())),
            out_u8.then_some(out_grids.as_slice()),
        );
        if out_u8 {
            let mut got = alloc(1, o, hw);
            conv(
                &xvs,
                1,
                h,
                w,
                &cv,
                has_res.then_some((rv, res_grid)),
                QOut::U8 { data: &mut got, grid: out_grid },
            );
            assert_eq!(
                &got[..(o / 4) * hw],
                &ref_u8[..],
                "packed 1x1 u8 mismatch at {h}x{w} parts={parts:?} o={o} silu={silu} res={has_res}"
            );
            recycle(got);
        } else {
            let mut got = vec![0.0f32; o * hw];
            conv(&xvs, 1, h, w, &cv, has_res.then_some((rv, res_grid)), QOut::F32 { data: &mut got });
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "packed 1x1 f32 mismatch at {h}x{w} parts={parts:?} o={o} silu={silu} res={has_res}"
            );
        }
        for xb in x_bufs {
            recycle(xb);
        }
        recycle(rb);
    }
}

#[test]
fn fused_3x3_multirow_and_shared_small_s1_bit_exact() {
    let mut seed = 0x3333_B47C_u64;
    let mut next_f32 = |lo: f32, hi: f32| {
        let u = ((lcg(&mut seed) >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };
    // Exercises:
    // - shared-strip small-spatial `s=1` (`oh <= 24`: 1x7, 7x20, 20x20, 24x25, n=2)
    // - multi-row batched `s=1` (`oh >= 25, ow <= 96`: 33x40, 40x33, RB=4 & RB=2)
    // - multi-row batched `s=2` (`ow <= 96`: 10x18, 22x40, 40x40, RB=2 & RB=4 with slot rotation)
    struct Conv3x3Case {
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        o: usize,
        stride: usize,
        silu: bool,
        has_res: bool,
        out_u8: bool,
    }
    let cases = [
        Conv3x3Case { n: 1, h: 1, w: 7, c: 8, o: 8, stride: 1, silu: true, has_res: false, out_u8: true },
        Conv3x3Case { n: 2, h: 7, w: 20, c: 16, o: 12, stride: 1, silu: true, has_res: true, out_u8: true },
        Conv3x3Case { n: 1, h: 20, w: 20, c: 64, o: 16, stride: 1, silu: true, has_res: false, out_u8: true },
        Conv3x3Case {
            n: 1,
            h: 24,
            w: 25,
            c: 16,
            o: 12,
            stride: 1,
            silu: false,
            has_res: true,
            out_u8: false,
        },
        Conv3x3Case { n: 1, h: 33, w: 40, c: 16, o: 12, stride: 1, silu: true, has_res: true, out_u8: true },
        Conv3x3Case {
            n: 1,
            h: 40,
            w: 33,
            c: 32,
            o: 16,
            stride: 1,
            silu: true,
            has_res: false,
            out_u8: false,
        },
        Conv3x3Case { n: 2, h: 10, w: 18, c: 12, o: 8, stride: 2, silu: true, has_res: true, out_u8: true },
        Conv3x3Case {
            n: 1,
            h: 22,
            w: 40,
            c: 16,
            o: 16,
            stride: 2,
            silu: false,
            has_res: false,
            out_u8: false,
        },
        Conv3x3Case { n: 1, h: 40, w: 40, c: 32, o: 16, stride: 2, silu: true, has_res: true, out_u8: true },
    ];
    for &Conv3x3Case { n, h, w, c, o, stride, silu, has_res, out_u8 } in &cases {
        let hw = h * w;
        let x_grid = Grid::from_range(-1.8, 2.4);
        let res_grid = Grid::from_range(-1.2, 1.2);
        let out_grid = Grid::from_range(-2.5, 3.5);
        let xf: Vec<f32> = (0..n * c * hw).map(|_| next_f32(-1.8, 2.4)).collect();
        let mut xb = alloc(n, c, hw);
        quantize(&xf, n, c, hw, x_grid, &mut xb);
        let xv = C4 { data: &xb, offset: 0, img_stride: (c / 4) * hw, c, zero: x_grid.zero };
        let wf: Vec<f32> = (0..o * c * 9).map(|_| next_f32(-0.5, 0.5)).collect();
        let bias: Vec<f32> = (0..o).map(|_| next_f32(-0.3, 0.3)).collect();
        let x_grids = vec![x_grid; c];
        let res_grids = vec![res_grid; o];
        let out_grids = vec![out_grid; o];
        let cv = QConv::new(&wf, &bias, (o, c, 3), stride, 1, silu, &x_grids);
        let (oh, ow) = cv.out_hw(h, w);
        let ohw = oh * ow;
        let rf: Vec<f32> = (0..n * o * ohw).map(|_| next_f32(-1.2, 1.2)).collect();
        let mut rb = alloc(n, o, ohw);
        quantize(&rf, n, o, ohw, res_grid, &mut rb);
        let rv = C4 { data: &rb, offset: 0, img_stride: (o / 4) * ohw, c: o, zero: res_grid.zero };
        let (ref_u8, ref_f32) = conv_ref_packed(
            &[xv],
            n,
            h,
            w,
            &cv,
            has_res.then_some((rv, res_grids.as_slice())),
            out_u8.then_some(out_grids.as_slice()),
        );
        if out_u8 {
            let mut got = alloc(n, o, ohw);
            conv(
                &[xv],
                n,
                h,
                w,
                &cv,
                has_res.then_some((rv, res_grid)),
                QOut::U8 { data: &mut got, grid: out_grid },
            );
            assert_eq!(
                &got[..n * (o / 4) * ohw],
                &ref_u8[..],
                "fused_3x3 u8 mismatch at n={n} {h}x{w} c={c} o={o} s={stride} silu={silu} res={has_res}"
            );
            recycle(got);
        } else {
            let mut got = vec![0.0f32; n * o * ohw];
            conv(&[xv], n, h, w, &cv, has_res.then_some((rv, res_grid)), QOut::F32 { data: &mut got });
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "fused_3x3 f32 mismatch at n={n} {h}x{w} c={c} o={o} s={stride} silu={silu} res={has_res}"
            );
        }
        recycle(xb);
        recycle(rb);
    }
}

#[test]
fn pack_f32_to_rgb0_u8_and_deinterleave_unrolled_bit_exact() {
    let mut seed = 0x57E4_2222_u64;
    let mut next_f32 = |lo: f32, hi: f32| {
        let u = ((lcg(&mut seed) >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };
    // Both serial (< PAR_MIN) and parallel (>= PAR_MIN) lengths across 16-wide, 8-wide, and scalar tails.
    for &(n, h, w) in &[(1usize, 1usize, 33usize), (2, 17, 19), (1, 160, 211)] {
        let hw = h * w;
        let src: Vec<f32> = (0..n * 3 * hw).map(|_| next_f32(-0.1, 1.1)).collect();
        let mut got = vec![0u32; n * hw];
        super::pack_f32_to_rgb0_u8(&src, n, h, w, &mut got);
        for img in 0..n {
            let base = img * 3 * hw;
            for p in 0..hw {
                let cvt = |v: f32| {
                    let s = v * 255.0;
                    let s = if s > 0.0 { s } else { 0.0 };
                    let s = if s < 255.0 { s } else { 255.0 };
                    s.round_ties_even() as i32 as u8
                };
                let exp = u32::from_le_bytes([
                    cvt(src[base + p]),
                    cvt(src[base + hw + p]),
                    cvt(src[base + 2 * hw + p]),
                    0,
                ]);
                assert_eq!(got[img * hw + p], exp, "rgb0 mismatch at n={n} {h}x{w} img={img} p={p}");
            }
        }
    }
    // 2× unrolled deinterleave_even_odd across 32-word, 16-word, and odd tails.
    for &w in &[0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
        let src: Vec<u32> = (0..w).map(|_| lcg(&mut seed) as u32).collect();
        let pairs = w.div_ceil(2) + 3;
        let pad = 0x1122_3344u32;
        let mut ev = vec![0u32; pairs];
        let mut od = vec![0u32; pairs];
        super::deinterleave_even_odd(&src, &mut ev, &mut od, pad);
        for j in 0..pairs {
            let e_exp = src.get(2 * j).copied().unwrap_or(pad);
            let o_exp = src.get(2 * j + 1).copied().unwrap_or(pad);
            assert_eq!((ev[j], od[j]), (e_exp, o_exp), "deinterleave mismatch at w={w} j={j}");
        }
    }
}
