use super::*;

/// The direct window scan the separable kernel must reproduce, argmax included.
fn direct(x: &[f32], (np, h, w): Planes, p: &Pool2dParams, [oh, ow]: [usize; 2]) -> (Vec<f32>, Vec<i64>) {
    let (mut v, mut ix) = (Vec::new(), Vec::new());
    for pl in 0..np {
        for oy in 0..oh {
            for ox in 0..ow {
                let (mut best, mut bi) = (f32::NEG_INFINITY, -1i64);
                for ky in 0..p.kernel[0] {
                    for kx in 0..p.kernel[1] {
                        let yy = (oy * p.stride[0] + ky) as isize - p.padding[0] as isize;
                        let xx = (ox * p.stride[1] + kx) as isize - p.padding[1] as isize;
                        if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                            continue;
                        }
                        let i = yy as usize * w + xx as usize;
                        if beats(x[pl * h * w + i], best, bi) {
                            best = x[pl * h * w + i];
                            bi = i as i64;
                        }
                    }
                }
                v.push(best);
                ix.push(bi.max(0));
            }
        }
    }
    (v, ix)
}

#[test]
fn separable_max_pool_matches_direct_scan_with_ties_and_nans() {
    // Few distinct values, so most windows have tied maxima; a sprinkling of NaNs.
    let (np, h, w) = (3, 11, 9);
    let x: Vec<f32> =
        (0..np * h * w).map(|i| if i % 37 == 5 { f32::NAN } else { ((i * 7919) % 5) as f32 }).collect();
    for (k, s, pad) in [([5, 5], [1, 1], [2, 2]), ([3, 2], [2, 1], [1, 1]), ([2, 3], [2, 3], [0, 1])] {
        let p = Pool2dParams { kernel: k, stride: s, padding: pad };
        let ohw = p.output_hw(h, w).unwrap();
        let (v, ix) = max_pool2d(&x, (np, h, w), &p, ohw);
        let (wv, wix) = direct(&x, (np, h, w), &p, ohw);
        assert_eq!(ix, wix, "{p:?}");
        for (a, b) in v.iter().zip(&wv) {
            assert!(a == b || a.is_nan() && b.is_nan(), "{p:?}: {a} vs {b}");
        }
    }
}

#[test]
fn max_pool_values_are_the_indexed_values_bit_for_bit() {
    // Ties between -0.0 and +0.0, NaNs with distinct payloads, -inf (whole windows of it), and planes
    // narrower or shorter than the window.
    let special = [0.0f32, -0.0, f32::NEG_INFINITY, f32::from_bits(0x7fc0_0001), f32::from_bits(0xffc0_0002)];
    for (np, h, w) in [(3, 11, 9), (2, 2, 3), (5, 1, 1), (17, 20, 20)] {
        let x: Vec<f32> = (0..np * h * w)
            .map(|i| match (i * 2654435761) % 11 {
                k @ 0..=4 => special[k],
                k => (k as f32 - 7.0) * 0.5,
            })
            .collect();
        for (k, s, pad) in [
            ([5, 5], [1, 1], [2, 2]),
            ([3, 2], [2, 1], [1, 1]),
            ([2, 3], [2, 3], [0, 1]),
            ([1, 1], [1, 1], [0, 0]),
            ([3, 3], [3, 2], [1, 0]),
        ] {
            let p = Pool2dParams { kernel: k, stride: s, padding: pad };
            let Some(ohw) = p.output_hw(h, w) else { continue };
            let (v, _) = max_pool2d(&x, (np, h, w), &p, ohw);
            let fast = max_pool2d_values(&x, (np, h, w), &p, ohw);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&fast), bits(&v), "{np}x{h}x{w} {p:?}");
        }
    }
}

/// Windows far larger than the plane (up to `usize::MAX` wide, padding half of it) and empty planes: both
/// paths give the direct scan's values without materialising the padding, and write every output.
#[test]
fn max_pool_with_huge_windows_or_empty_planes() {
    let big = |k: usize| Pool2dParams { kernel: [k, 1], stride: [1, 1], padding: [k / 2, 0] };
    let cases = [
        ((1, 1, 1), big(isize::MAX as usize)),
        ((1, 1, 1), big(usize::MAX)),
        ((2, 3, 4), Pool2dParams { kernel: [41, 5], stride: [7, 1], padding: [20, 2] }),
        ((2, 0, 3), Pool2dParams { kernel: [2, 1], stride: [1, 1], padding: [1, 0] }),
        ((2, 3, 0), Pool2dParams { kernel: [1, 4], stride: [1, 2], padding: [0, 2] }),
    ];
    for ((np, h, w), p) in cases {
        let x: Vec<f32> = (0..np * h * w).map(|i| (i * 7 % 5) as f32 - 1.5).collect();
        let ohw = p.output_hw(h, w).unwrap();
        let (v, ix) = max_pool2d(&x, (np, h, w), &p, ohw);
        let fast = max_pool2d_values(&x, (np, h, w), &p, ohw);
        // A 1x1 plane under a window too wide to scan directly is its own maximum.
        let (wv, wix) =
            if p.kernel[0] > 1 << 20 { (x.clone(), vec![0; np]) } else { direct(&x, (np, h, w), &p, ohw) };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&v), bits(&wv), "{np}x{h}x{w} {p:?}");
        assert_eq!(bits(&fast), bits(&wv), "{np}x{h}x{w} {p:?}");
        assert_eq!(ix, wix, "{np}x{h}x{w} {p:?}");
    }
}

#[test]
fn separable_avg_pool_and_backward_match_direct_including_edges() {
    for (np, h, w) in [(3, 19, 23), (2, 16, 16), (4, 1, 1), (2, 0, 3)] {
        let x: Vec<f32> = (0..np * h * w).map(|i| ((i as f32 * 0.37).sin()) * 3.0).collect();
        for (k, s, pad) in [
            ([2, 2], [2, 2], [0, 0]),
            ([3, 3], [1, 1], [1, 1]),
            ([5, 5], [1, 1], [2, 2]),
            ([3, 2], [2, 3], [1, 1]),
            ([41, 5], [7, 1], [20, 2]),
        ] {
            let p = Pool2dParams { kernel: k, stride: s, padding: pad };
            let Some([oh, ow]) = p.output_hw(h, w) else { continue };
            let got = avg_pool2d(&x, (np, h, w), &p, [oh, ow]);
            let inv = 1.0 / (k[0] * k[1]) as f64;
            for pl in 0..np {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut sum = 0.0f64;
                        for ky in 0..k[0] {
                            for kx in 0..k[1] {
                                let yy = (oy * s[0] + ky) as isize - pad[0] as isize;
                                let xx = (ox * s[1] + kx) as isize - pad[1] as isize;
                                if yy >= 0 && xx >= 0 && yy < h as isize && xx < w as isize {
                                    sum += x[(pl * h + yy as usize) * w + xx as usize] as f64;
                                }
                            }
                        }
                        let want = (sum * inv) as f32;
                        let g = got[(pl * oh + oy) * ow + ox];
                        assert!(
                            (g - want).abs() <= 1e-5 * (1.0 + want.abs()),
                            "{np}x{h}x{w} {p:?}: {g} vs {want}"
                        );
                    }
                }
            }
            let gy: Vec<f32> = (0..np * oh * ow).map(|i| ((i as f32 * 0.19).cos()) * 2.0).collect();
            let gx = avg_pool2d_backward(&gy, (np, h, w), &p, [oh, ow]);
            let mut ref_gx = vec![0.0f64; np * h * w];
            for pl in 0..np {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let v = gy[(pl * oh + oy) * ow + ox] as f64 * inv;
                        for ky in 0..k[0] {
                            for kx in 0..k[1] {
                                let yy = (oy * s[0] + ky) as isize - pad[0] as isize;
                                let xx = (ox * s[1] + kx) as isize - pad[1] as isize;
                                if yy >= 0 && xx >= 0 && yy < h as isize && xx < w as isize {
                                    ref_gx[(pl * h + yy as usize) * w + xx as usize] += v;
                                }
                            }
                        }
                    }
                }
            }
            for (i, (&a, &b)) in gx.iter().zip(&ref_gx).enumerate() {
                assert!(
                    (a as f64 - b).abs() <= 1e-5 * (1.0 + b.abs()),
                    "bwd {np}x{h}x{w} {p:?} [{i}]: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn upsample2x2_forward_and_backward_match_reference_bit_for_bit() {
    // Exercises AVX2 8-wide chunks + ragged tails (widths 1, 7, 8, 9, 19, 37) and parallel batching.
    for (np, h, w) in [(2, 5, 1), (3, 7, 7), (4, 8, 8), (3, 9, 19), (64, 21, 37)] {
        let x: Vec<f32> = (0..np * h * w).map(|i| ((i as f32 * 0.29).sin()) * 4.0).collect();
        let up = upsample_nearest2d(&x, (np, h, w), [2, 2]);
        let (oh, ow) = (2 * h, 2 * w);
        for pl in 0..np {
            for oy in 0..oh {
                for ox in 0..ow {
                    let got = up[(pl * oh + oy) * ow + ox];
                    let want = x[(pl * h + oy / 2) * w + ox / 2];
                    assert_eq!(got.to_bits(), want.to_bits(), "up {np}x{h}x{w} ({pl},{oy},{ox})");
                }
            }
        }
        let gy: Vec<f32> = (0..np * oh * ow).map(|i| ((i as f32 * 0.43).cos()) * 2.5).collect();
        let gx = upsample_nearest2d_backward(&gy, (np, h, w), [2, 2]);
        for pl in 0..np {
            for y in 0..h {
                for x0 in 0..w {
                    let r0 = &gy[(pl * oh + 2 * y) * ow..];
                    let r1 = &gy[(pl * oh + 2 * y + 1) * ow..];
                    let want = (r0[2 * x0] + r0[2 * x0 + 1]) + (r1[2 * x0] + r1[2 * x0 + 1]);
                    let got = gx[(pl * h + y) * w + x0];
                    assert_eq!(got.to_bits(), want.to_bits(), "up_bwd {np}x{h}x{w} ({pl},{y},{x0})");
                }
            }
        }
    }
}

#[test]
fn avx2_max_pool_values_and_indexed_match_direct_across_widths_and_edge_cases() {
    let special = [
        0.0f32,
        -0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0007),
    ];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    // Unaligned widths across 8-wide AVX2 boundary + scalar tails, plus a parallel batch (`>= PAR_MIN`).
    for (np, h, w) in [(2, 5, 1), (2, 7, 7), (2, 8, 8), (2, 9, 9), (2, 11, 17), (3, 20, 20), (2, 13, 37)] {
        let x: Vec<f32> = (0..np * h * w)
            .map(|i| match (i * 2654435761) % 13 {
                k @ 0..=5 => special[k],
                k => (k as f32 - 8.0) * 0.25,
            })
            .collect();
        for (k, s, pad) in [
            ([5, 5], [1, 1], [2, 2]),
            ([3, 3], [1, 1], [1, 1]),
            ([2, 2], [2, 2], [0, 0]),
            ([2, 2], [2, 2], [1, 1]),
            ([7, 7], [1, 1], [3, 3]),
            ([2, 3], [2, 1], [1, 1]),
            // All-padding corner windows (`pad >= kernel`).
            ([2, 2], [1, 1], [2, 2]),
        ] {
            let p = Pool2dParams { kernel: k, stride: s, padding: pad };
            let Some(ohw) = p.output_hw(h, w) else { continue };
            let (v, ix) = max_pool2d(&x, (np, h, w), &p, ohw);
            let fast = max_pool2d_values(&x, (np, h, w), &p, ohw);
            let (wv, wix) = direct(&x, (np, h, w), &p, ohw);
            assert_eq!(ix, wix, "idx {np}x{h}x{w} {p:?}");
            assert_eq!(bits(&v), bits(&wv), "val {np}x{h}x{w} {p:?}");
            assert_eq!(bits(&fast), bits(&wv), "fast {np}x{h}x{w} {p:?}");
        }
    }
    // Extreme stride where `ow == 1` and `pw > wp` (`w = 1, k = 1, s = 10, pad = 2` -> `ow = 1, wp = 1, pw = 2`).
    let p_ext = Pool2dParams { kernel: [1, 1], stride: [10, 10], padding: [2, 2] };
    let ohw = p_ext.output_hw(1, 1).unwrap();
    let (v, ix) = max_pool2d(&[42.0], (1, 1, 1), &p_ext, ohw);
    let fast = max_pool2d_values(&[42.0], (1, 1, 1), &p_ext, ohw);
    let (wv, wix) = direct(&[42.0], (1, 1, 1), &p_ext, ohw);
    assert_eq!(ix, wix);
    assert_eq!(bits(&v), bits(&wv));
    assert_eq!(bits(&fast), bits(&wv));

    // All `-inf` plane: first valid real `-inf` wins (index 0 for unpadded), not padding.
    let neg_inf = vec![f32::NEG_INFINITY; 2 * 10 * 19];
    for p in [
        Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] },
        Pool2dParams { kernel: [2, 2], stride: [2, 2], padding: [0, 0] },
    ] {
        let ohw = p.output_hw(10, 19).unwrap();
        let (v, ix) = max_pool2d(&neg_inf, (2, 10, 19), &p, ohw);
        let (wv, wix) = direct(&neg_inf, (2, 10, 19), &p, ohw);
        assert_eq!(ix, wix, "all -inf {p:?}");
        assert_eq!(bits(&v), bits(&wv), "all -inf {p:?}");
    }

    // Parallel path (`>= PAR_MIN`): `[128, 17, 19]` = 41,344 outputs.
    let (np, h, w) = (128, 17, 19);
    let x: Vec<f32> = (0..np * h * w).map(|i| ((i as f32 * 0.13).sin()) * 3.0).collect();
    for p in [
        Pool2dParams { kernel: [5, 5], stride: [1, 1], padding: [2, 2] },
        Pool2dParams { kernel: [2, 2], stride: [2, 2], padding: [0, 0] },
    ] {
        let ohw = p.output_hw(h, w).unwrap();
        let (v, ix) = max_pool2d(&x, (np, h, w), &p, ohw);
        let fast = max_pool2d_values(&x, (np, h, w), &p, ohw);
        let (wv, wix) = direct(&x, (np, h, w), &p, ohw);
        assert_eq!(ix, wix, "par idx {p:?}");
        assert_eq!(bits(&v), bits(&wv), "par val {p:?}");
        assert_eq!(bits(&fast), bits(&wv), "par fast {p:?}");
    }
}

#[test]
fn adaptive_avg_pool_and_backward_match_direct_across_sizes() {
    for ((np, h, w), out_hw) in [
        ((3, 19, 23), [1, 1]),
        ((3, 19, 23), [3, 5]),
        ((2, 9, 41), [2, 2]), // exercises wide horizontal bin (`xs.len() >= 16`)
        ((2, 3, 5), [7, 11]), // exercises upsampling (`oh > h`, `ow > w`)
        ((2, 7, 17), [7, 17]),
        ((128, 17, 19), [1, 1]),
        ((64, 21, 29), [5, 7]),
        ((2, 0, 5), [1, 1]),
    ] {
        let x: Vec<f32> = (0..np * h * w).map(|i| ((i as f32 * 0.31).sin()) * 2.5).collect();
        let got = adaptive_avg_pool2d(&x, (np, h, w), out_hw);
        let [oh, ow] = out_hw;
        if h * w > 0 {
            for pl in 0..np {
                for oy in 0..oh {
                    let ys = adaptive_range(oy, h, oh);
                    for ox in 0..ow {
                        let xs = adaptive_range(ox, w, ow);
                        let mut s = 0.0f64;
                        for yy in ys.clone() {
                            for xx in xs.clone() {
                                s += x[(pl * h + yy) * w + xx] as f64;
                            }
                        }
                        let want = (s / (ys.len() * xs.len()) as f64) as f32;
                        let g = got[(pl * oh + oy) * ow + ox];
                        assert!(
                            (g - want).abs() <= 1e-5 * (1.0 + want.abs()),
                            "adap fwd {np}x{h}x{w} -> {out_hw:?}: {g} vs {want}"
                        );
                    }
                }
            }
        }
        let gy: Vec<f32> = (0..np * oh * ow).map(|i| ((i as f32 * 0.23).cos()) * 1.5).collect();
        let gx = adaptive_avg_pool2d_backward(&gy, (np, h, w), out_hw);
        let mut ref_gx = vec![0.0f64; np * h * w];
        if h * w > 0 {
            for pl in 0..np {
                for oy in 0..oh {
                    let ys = adaptive_range(oy, h, oh);
                    for ox in 0..ow {
                        let xs = adaptive_range(ox, w, ow);
                        let v = gy[(pl * oh + oy) * ow + ox] as f64 / (ys.len() * xs.len()) as f64;
                        for yy in ys.clone() {
                            for xx in xs.clone() {
                                ref_gx[(pl * h + yy) * w + xx] += v;
                            }
                        }
                    }
                }
            }
        }
        for (i, (&a, &b)) in gx.iter().zip(&ref_gx).enumerate() {
            assert!(
                (a as f64 - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "adap bwd {np}x{h}x{w} -> {out_hw:?} [{i}]: {a} vs {b}"
            );
        }
    }

    // Finite extreme cancellation in forward (`2x4 -> 1x2`): must return `0.0`, not `NaN`.
    let cancel_in = vec![f32::MAX, -f32::MAX, 0.0, 0.0, f32::MAX, -f32::MAX, 0.0, 0.0];
    let cancel_out = adaptive_avg_pool2d(&cancel_in, (1, 2, 4), [1, 2]);
    assert_eq!(cancel_out, vec![0.0, 0.0]);
    let cancel_gap = adaptive_avg_pool2d(&cancel_in, (1, 2, 4), [1, 1]);
    assert_eq!(cancel_gap, vec![0.0]);

    // Overlapping-bin cancellation in backward (`3x3 -> 2x2`): center cell sees `1e8 + 0 - 1e8 + 1 = 1.0`.
    let bwd_cancel = adaptive_avg_pool2d_backward(&[4e8, 0.0, -4e8, 4.0], (1, 3, 3), [2, 2]);
    assert_eq!(bwd_cancel[4], 1.0);
}

#[test]
fn avg_pool_backward_handles_huge_stride1_kernel_and_all_padding() {
    // Huge single-axis kernel (`kw = usize::MAX`) must finish in O(1), not loop `usize::MAX` times.
    let p_huge = Pool2dParams { kernel: [1, usize::MAX], stride: [1, 1], padding: [0, usize::MAX / 2] };
    let ohw = p_huge.output_hw(1, 1).unwrap();
    let fwd = avg_pool2d(&[3.0], (1, 1, 1), &p_huge, ohw);
    let bwd = avg_pool2d_backward(&[5.0], (1, 1, 1), &p_huge, ohw);
    assert_eq!(fwd.len(), 1);
    assert_eq!(bwd, vec![5.0 / usize::MAX as f32]);

    // Extreme padding with `ow == 1` and `pw > wp`: all-padding window returns 0 without panicking.
    let p_ext = Pool2dParams { kernel: [1, 1], stride: [10, 10], padding: [2, 2] };
    let ohw_ext = p_ext.output_hw(1, 1).unwrap();
    assert_eq!(avg_pool2d(&[7.0], (1, 1, 1), &p_ext, ohw_ext), vec![0.0]);
    assert_eq!(avg_pool2d_backward(&[7.0], (1, 1, 1), &p_ext, ohw_ext), vec![0.0]);

    // Stride-1 with `pw > ow` (`w = 1, k = 3, s = 1, pad = 2` -> `ow = 3`? Try `w = 2, k = 5, s = 1, pad = 2` and `pw = 4, kw = 9`).
    let p_pad = Pool2dParams { kernel: [1, 9], stride: [1, 1], padding: [0, 4] };
    let ohw_pad = p_pad.output_hw(1, 2).unwrap();
    let gy = vec![1.0, 2.0];
    let gx = avg_pool2d_backward(&gy, (1, 1, 2), &p_pad, ohw_pad);
    assert_eq!(gx, vec![3.0 / 9.0, 3.0 / 9.0]);
}
