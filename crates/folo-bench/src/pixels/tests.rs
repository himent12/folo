use super::*;

#[test]
fn math_matches_libm_closely() {
    for &x in &[1e-9, 0.003, 0.25, 0.5, 0.999, 1.0, 1.5, 7.0, 1234.5] {
        assert!((ln(x) - x.ln()).abs() < 1e-14 * x.ln().abs().max(1.0), "ln {x}");
    }
    for &x in &[-650.0, -20.0, -1.0, -0.1, 0.0, 0.3, 2.0, 40.0] {
        assert!((exp(x) - x.exp()).abs() <= 1e-14 * x.exp(), "exp {x}");
    }
    assert_eq!(pow(0.0, 2.4), 0.0);
    assert!((pow(0.5, 2.4) - 0.5f64.powf(2.4)).abs() < 1e-15);
}

#[test]
fn srgb_round_trips_every_code() {
    let t = srgb();
    for c in 0..=255u8 {
        assert_eq!(t.enc(t.lin(c)), c, "{c}");
    }
    assert_eq!((t.enc(-1.0), t.enc(2.0), t.enc(f32::NAN)), (0, 255, 0));
}

#[test]
fn gauss_has_unit_variance() {
    let mut rng = Rng::seed(3);
    let n = 200_000;
    let v: Vec<f64> = (0..n).map(|_| gauss(&mut rng) as f64).collect();
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.01, "{mean} {var}");
}

#[test]
fn taps_sum_to_one() {
    for (a, b) in [(640, 427), (7, 3), (5, 5), (3, 7), (1, 1), (10, 1)] {
        for t in [area_taps(a, b), bilinear_taps(a, b)] {
            for i in 0..b {
                let s: f32 = t.get(i).1.iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "{a}->{b} at {i}: {s}");
            }
        }
    }
    let k = gaussian_kernel(2.0);
    assert_eq!(k.len(), 13);
    assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

fn ref_separable(
    raw: &[u8],
    w: usize,
    h: usize,
    kx: &[f32],
    ky: &[f32],
    haze: Option<(f32, [f32; 3])>,
) -> Vec<u8> {
    let stride = w * 3;
    let (rx, ry) = (kx.len() / 2, ky.len() / 2);
    let plane: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
    let mut tmp = vec![0.0f32; plane.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (k, &wk) in kx.iter().enumerate() {
                    let sx = (x + k).saturating_sub(rx).min(w - 1);
                    acc += wk * plane[(y * w + sx) * 3 + c];
                }
                tmp[y * stride + x * 3 + c] = acc;
            }
        }
    }
    let mut out = vec![0u8; raw.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (k, &wk) in ky.iter().enumerate() {
                    let sy = (y + k).saturating_sub(ry).min(h - 1);
                    acc += wk * tmp[sy * stride + x * 3 + c];
                }
                if let Some((t, veil)) = haze {
                    acc = acc * t + veil[c];
                }
                out[y * stride + x * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
            }
        }
    }
    out
}

fn ref_taps(raw: &[u8], w: usize, h: usize, taps: &[(isize, isize, f32)]) -> Vec<u8> {
    let mut out = vec![0u8; raw.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for &(dx, dy, wk) in taps {
                    let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    acc += wk * raw[(sy * w + sx) * 3 + c] as f32;
                }
                out[(y * w + x) * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
            }
        }
    }
    out
}

fn ref_resample(raw: &[u8], w: usize, h: usize, w2: usize, h2: usize, xt: &Taps, yt: &Taps) -> Vec<u8> {
    let mut tmp = vec![0.0f32; h * w2 * 3];
    for y in 0..h {
        for x in 0..w2 {
            let (s, ws) = xt.get(x);
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (j, &wj) in ws.iter().enumerate() {
                    acc += wj * raw[(y * w + s + j) * 3 + c] as f32;
                }
                tmp[(y * w2 + x) * 3 + c] = acc;
            }
        }
    }
    let mut out = vec![0u8; h2 * w2 * 3];
    for y in 0..h2 {
        let (s, ws) = yt.get(y);
        for x in 0..w2 {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (j, &wj) in ws.iter().enumerate() {
                    acc += wj * tmp[((s + j) * w2 + x) * 3 + c];
                }
                out[(y * w2 + x) * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
            }
        }
    }
    out
}

#[test]
fn register_tiled_convolutions_and_resample_match_reference_bit_exact() {
    let mut rng = Rng::seed(0xB17_E4AC);
    let k_small = gaussian_kernel(1.7); // klen = 13 (ring-buffered path)
    let k_wide = gaussian_kernel(5.0); // klen = 31 (two-pass fallback path)
    let taps = vec![(-3, -2, 0.08), (-1, -1, 0.14), (0, 0, 0.36), (2, 1, 0.25), (4, 2, 0.17)];
    let veil = [220.0 * 0.35, 225.0 * 0.35, 232.0 * 0.35];
    for (w, h) in
        [(1, 1), (2, 3), (7, 5), (8, 4), (9, 7), (15, 6), (16, 8), (17, 9), (25, 11), (65, 13), (300, 240)]
    {
        let raw: Vec<u8> = (0..w * h * 3).map(|_| rng.below(256) as u8).collect();
        let mut sum = [0u64; 3];
        for px in raw.as_chunks::<3>().0 {
            sum[0] += px[0] as u64;
            sum[1] += px[1] as u64;
            sum[2] += px[2] as u64;
        }
        let n = (w * h) as u64;
        assert_eq!(mean_rgb(&raw), sum.map(|s| ((s + n / 2) / n) as u8), "mean_rgb {w}x{h}");
        for k in [&k_small, &k_wide] {
            assert_eq!(
                convolve_separable_u8(&raw, w, h, k, k),
                ref_separable(&raw, w, h, k, k, None),
                "separable klen={} {w}x{h}",
                k.len()
            );
            assert_eq!(
                convolve_separable_haze_u8(&raw, w, h, k, k, 0.65, veil),
                ref_separable(&raw, w, h, k, k, Some((0.65, veil))),
                "haze klen={} {w}x{h}",
                k.len()
            );
        }
        assert_eq!(convolve_taps_u8(&raw, w, h, &taps), ref_taps(&raw, w, h, &taps), "taps {w}x{h}");
        let (w2, h2) = ((w / 2).max(1), (h / 2).max(1));
        let (xt, yt) = (area_taps(w, w2), area_taps(h, h2));
        assert_eq!(
            resample_u8_to_u8(&raw, w, h, w2, h2, &xt, &yt),
            ref_resample(&raw, w, h, w2, h2, &xt, &yt),
            "resample u8->u8 {w}x{h}->{w2}x{h2}"
        );
        let small = resample_u8_to_plane(&raw, w, h, w2, h2, &xt, &yt);
        let (bx, by) = (bilinear_taps(w2, w), bilinear_taps(h2, h));
        let up = resample_plane_to_u8(&small, w, h, &bx, &by);
        // Cross-check the two-stage plane resample against reference scalar formula on `small.data`.
        let mut ref_up = vec![0u8; raw.len()];
        let mut tmp = vec![0.0f32; h2 * w * 3];
        for y in 0..h2 {
            for x in 0..w {
                let (s, ws) = bx.get(x);
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (j, &wj) in ws.iter().enumerate() {
                        acc += wj * small.data[(y * w2 + s + j) * 3 + c];
                    }
                    tmp[(y * w + x) * 3 + c] = acc;
                }
            }
        }
        for y in 0..h {
            let (s, ws) = by.get(y);
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (j, &wj) in ws.iter().enumerate() {
                        acc += wj * tmp[((s + j) * w + x) * 3 + c];
                    }
                    ref_up[(y * w + x) * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
                }
            }
        }
        assert_eq!(up, ref_up, "resample plane->u8 {w}x{h}");
    }
}
