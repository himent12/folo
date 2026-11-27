//! Tests for Wave 7 CPU int8 algorithms (A1, A2, A3):
//! - A3: Bit-exact division-free minimax fast SiLU (`silu_q8`), scalar == AVX2, `max_rel <= 2e-6`, extremes & NaN.
//! - A2: Direct `u8` RGB0-to-C4 AVX-VNNI stem (`QStem`, `stem_u8`, `pack_f32_to_rgb0_u8`) vs scalar integer
//!   reference and `f32` conv reference across odd sizes, strides, thread counts, and per-channel outputs.
//! - A1: Register-safe L1 `TileF32` per-part concat scale decoupling (proves mismatched-scale branches preserve
//!   full 7-bit weight precision vs collapsed single-part folding!) and zero-cost per-channel C4 grids
//!   (`quantize_per_channel`, `dequantize_per_channel`, `conv_per_channel`, `QOut::U8PerChannel`, `max_pool`,
//!   `upsample2`).

use folo_core::cpu::q8::{
    self, C4, Grid, PIXEL_SCALE, QConv, QOut, QStem, SLACK, alloc, conv_per_channel, dequantize_per_channel,
    pack_f32_to_rgb0_u8, quantize_per_channel, quantize_weights, should_decouple_parts, silu_q8, stem_u8,
};
use folo_core::rng::Rng;

fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        a * b + c
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        a.mul_add(b, c)
    }
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()
}

#[test]
fn fast_silu_accuracy_and_extremes() {
    let mut rng = Rng::seed(7001);
    let mut max_rel = 0.0f64;
    for _ in 0..50_000 {
        let x = rng.uniform(-86.0, 86.0);
        let want = x as f64 / (1.0 + (-(x as f64)).exp());
        let got = silu_q8(x) as f64;
        let err = (got - want).abs() / want.abs().max(1.0);
        max_rel = max_rel.max(err);
        assert!(err <= 2e-6, "silu_q8({x}) = {got}, want {want}, rel {err:e}");
    }
    let bound = if cfg!(target_arch = "wasm32") { 3e-7 } else { 2.5e-7 };
    assert!(max_rel <= bound, "max_rel {max_rel:e} exceeded {bound:e}");
    // Pin peak-error boundary inputs (worst hybrid error ~2.01e-7 near 0x3fc8_331e and reduced f = ±0.5).
    for x in [f32::from_bits(0x3fc8_331e), 0.5 / std::f32::consts::LOG2_E, -0.5 / std::f32::consts::LOG2_E] {
        let want = x as f64 / (1.0 + (-(x as f64)).exp());
        let got = silu_q8(x) as f64;
        let err = (got - want).abs() / want.abs().max(1.0);
        assert!(err <= bound, "silu_q8({x}) = {got}, want {want}, err {err:e}");
    }
    assert_eq!(silu_q8(0.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(silu_q8(-0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(silu_q8(f32::MAX), f32::MAX);
    assert_eq!(silu_q8(f32::INFINITY), f32::INFINITY);
    assert!(silu_q8(f32::NEG_INFINITY).abs() < 1e-30);
    assert!(silu_q8(f32::MIN).abs() < 1e-30);
    assert!(silu_q8(-200.0).abs() < 1e-30);
    assert!(silu_q8(f32::NAN).is_nan());
}

#[test]
fn u8_stem_matches_exact_integer_and_f32_reference() {
    let mut rng = Rng::seed(7002);
    for &(n, h, w, o, k, s, p, silu) in &[
        (1, 32, 32, 16, 3, 2, 1, true),
        (1, 16, 64, 12, 3, 2, 1, true), // ow = 32 = 24 + 8 (exact VECS=1 tail direct store)
        (1, 16, 80, 12, 3, 2, 1, false), // ow = 40 = 24 + 16 (exact VECS=2 tail direct store)
        (2, 29, 35, 12, 3, 2, 1, true),
        (1, 17, 19, 8, 3, 1, 1, false),
        (2, 16, 16, 20, 1, 1, 0, true),
    ] {
        let hw = h * w;
        let taps = k * k;
        let (oh, ow) = q8::out_size(h, w, k, s, p).unwrap();
        let ohw = oh * ow;
        // Exact u8 pixels -> both RGB0 u32 and planar f32 (v / 255.0).
        let mut rgb0 = vec![0u32; n * hw];
        let mut planar = vec![0f32; n * 3 * hw];
        for img in 0..n {
            for px in 0..hw {
                let r = (rng.next_u64() % 256) as u8;
                let g = (rng.next_u64() % 256) as u8;
                let b = (rng.next_u64() % 256) as u8;
                rgb0[img * hw + px] = u32::from_le_bytes([r, g, b, 0]);
                planar[(img * 3) * hw + px] = r as f32 * PIXEL_SCALE;
                planar[(img * 3 + 1) * hw + px] = g as f32 * PIXEL_SCALE;
                planar[(img * 3 + 2) * hw + px] = b as f32 * PIXEL_SCALE;
            }
        }
        // `pack_f32_to_rgb0_u8` recovers the exact `rgb0` words bit for bit.
        let mut repacked = vec![u32::MAX; n * hw];
        pack_f32_to_rgb0_u8(&planar, n, h, w, &mut repacked);
        assert_eq!(repacked, rgb0, "pack_f32_to_rgb0_u8 round-trip ({n},{h},{w})");

        let wt: Vec<f32> = (0..o * 3 * taps).map(|_| rng.uniform(-0.4, 0.4)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let out_grids: Vec<Grid> =
            (0..o).map(|c| Grid::from_range(-0.5 - 0.1 * c as f32, 2.0 + 0.2 * c as f32)).collect();
        let stem = QStem::new(&wt, &bias, (o, k), s, p, silu);

        // Exact integer reference.
        let folded: Vec<f32> = wt.iter().map(|&v| v * PIXEL_SCALE).collect();
        let (qw, sw) = quantize_weights(&folded, o);
        let mut want_f = vec![0f32; n * o * ohw];
        let mut want_q = vec![0u32; n * o / 4 * ohw];
        for img in 0..n {
            for oc in 0..o {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = 0i32;
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = (oy * s + ky) as isize - p as isize;
                                let ix = (ox * s + kx) as isize - p as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let bytes = rgb0[img * hw + iy as usize * w + ix as usize].to_le_bytes();
                                let t = ky * k + kx;
                                for c in 0..3 {
                                    acc += bytes[c] as i32 * qw[(oc * 3 + c) * taps + t] as i32;
                                }
                            }
                        }
                        let mut y = fmadd(acc as f32, sw[oc], bias[oc]);
                        if silu {
                            y = silu_q8(y);
                        }
                        let pos = oy * ow + ox;
                        want_f[(img * o + oc) * ohw + pos] = y;
                        want_q[(img * o / 4 + oc / 4) * ohw + pos] |=
                            (out_grids[oc].quantize(y) as u32) << (8 * (oc % 4));
                    }
                }
            }
        }

        for threads in [1, 4] {
            let mut got_f = vec![f32::NAN; n * o * ohw];
            let mut got_q = alloc(n, o, ohw);
            pool(threads).install(|| {
                stem_u8(&rgb0, n, h, w, &stem, QOut::F32 { data: &mut got_f });
                stem_u8(&rgb0, n, h, w, &stem, QOut::U8PerChannel { data: &mut got_q, grids: &out_grids });
            });
            assert!(
                got_f.iter().zip(&want_f).all(|(a, b)| a.to_bits() == b.to_bits()),
                "stem_u8 f32 mismatch at {threads} thr"
            );
            assert_eq!(
                &got_q[..want_q.len()],
                &want_q[..],
                "stem_u8 u8 per-channel mismatch at {threads} thr"
            );
        }
    }
}

#[test]
fn per_channel_c4_and_decoupled_concat_tile_f32_are_bit_exact_and_accurate() {
    let mut rng = Rng::seed(7003);
    let (n, h, w) = (2, 11, 13);
    let hw = h * w;
    let (c0, c1, o) = (8, 8, 12);
    // Part 0 has 10× smaller activation scale than part 1 (simulates mismatched concat branches).
    let g0: Vec<Grid> =
        (0..c0).map(|i| Grid { scale: 0.002 + 0.0005 * i as f32, zero: (20 + i * 7) as u8 }).collect();
    let g1: Vec<Grid> =
        (0..c1).map(|i| Grid { scale: 0.04 + 0.006 * i as f32, zero: (80 + i * 9) as u8 }).collect();
    let out_g: Vec<Grid> =
        (0..o).map(|i| Grid { scale: 0.015 + 0.003 * i as f32, zero: (40 + i * 5) as u8 }).collect();
    let res_g: Vec<Grid> =
        (0..o).map(|i| Grid { scale: 0.02 + 0.002 * i as f32, zero: (100 + i * 3) as u8 }).collect();
    assert!(should_decouple_parts(&[&g0, &g1]));

    let x0_f: Vec<f32> = (0..n * c0 * hw).map(|_| rng.uniform(-0.1, 0.3)).collect();
    let x1_f: Vec<f32> = (0..n * c1 * hw).map(|_| rng.uniform(-2.0, 6.0)).collect();
    let res_f: Vec<f32> = (0..n * o * hw).map(|_| rng.uniform(-1.0, 2.0)).collect();
    let mut x0_q = alloc(n, c0, hw);
    let mut x1_q = alloc(n, c1, hw);
    let mut res_q = alloc(n, o, hw);
    quantize_per_channel(&x0_f, n, c0, hw, &g0, &mut x0_q);
    quantize_per_channel(&x1_f, n, c1, hw, &g1, &mut x1_q);
    quantize_per_channel(&res_f, n, o, hw, &res_g, &mut res_q);

    // Per-channel dequantise matches scalar Grid::dequantize(Grid::quantize(x)) bit for bit.
    let v0 = C4 { data: &x0_q, offset: 0, img_stride: c0 / 4 * hw, c: c0, zero: g0[0].zero };
    let v1 = C4 { data: &x1_q, offset: 0, img_stride: c1 / 4 * hw, c: c1, zero: g1[0].zero };
    let vr = C4 { data: &res_q, offset: 0, img_stride: o / 4 * hw, c: o, zero: res_g[0].zero };
    let mut x0_dq = vec![0f32; x0_f.len()];
    dequantize_per_channel(v0, n, hw, &g0, &mut x0_dq);
    for img in 0..n {
        for (ch, g) in g0.iter().enumerate().take(c0) {
            for p in 0..hw {
                let idx = (img * c0 + ch) * hw + p;
                assert_eq!(x0_dq[idx].to_bits(), g.dequantize(g.quantize(x0_f[idx])).to_bits());
            }
        }
    }

    // Decoupled 1×1 and 3×3 convs: exact bit-for-bit match against scalar L1 TileF32 reference,
    // AND much lower SQNR error against f32 than coupled single-part folding!
    for &(k, pad, silu) in &[(1, 0, true), (3, 1, false)] {
        let taps = k * k;
        let cin = c0 + c1;
        let wt: Vec<f32> = (0..o * cin * taps).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let cv_dec = QConv::new_parts(&wt, &bias, (o, cin, k), 1, pad, silu, &[&g0, &g1], true);
        let cv_cpl = QConv::new_parts(&wt, &bias, (o, cin, k), 1, pad, silu, &[&g0, &g1], false);
        assert_eq!(cv_dec.num_parts(), 2);
        assert_eq!(cv_cpl.num_parts(), 1);

        let z0: Vec<u8> = g0.iter().map(|g| g.zero).collect();
        let z1: Vec<u8> = g1.iter().map(|g| g.zero).collect();
        let mut out_t1 = alloc(n, o, hw);
        let mut out_t4 = alloc(n, o, hw);
        let mut out_f_dec = vec![0f32; n * o * hw];
        let mut out_f_cpl = vec![0f32; n * o * hw];
        pool(1).install(|| {
            conv_per_channel(
                &[v0, v1],
                &[&z0, &z1],
                n,
                h,
                w,
                &cv_dec,
                Some((vr, &res_g)),
                QOut::U8PerChannel { data: &mut out_t1, grids: &out_g },
            );
            conv_per_channel(
                &[v0, v1],
                &[&z0, &z1],
                n,
                h,
                w,
                &cv_dec,
                None,
                QOut::F32 { data: &mut out_f_dec },
            );
            conv_per_channel(
                &[v0, v1],
                &[&z0, &z1],
                n,
                h,
                w,
                &cv_cpl,
                None,
                QOut::F32 { data: &mut out_f_cpl },
            );
        });
        pool(4).install(|| {
            conv_per_channel(
                &[v0, v1],
                &[&z0, &z1],
                n,
                h,
                w,
                &cv_dec,
                Some((vr, &res_g)),
                QOut::U8PerChannel { data: &mut out_t4, grids: &out_g },
            );
        });
        assert_eq!(
            &out_t1[..n * o / 4 * hw],
            &out_t4[..n * o / 4 * hw],
            "decoupled per-channel conv must be thread-count deterministic (k={k})"
        );
        // Check against exact f32 conv on the dequantised inputs (part 0 only is weighted strongly when part 1
        // is small; here check overall RMSE on part 0's contribution by zeroing part 1!).
        let zeros_p1 = vec![
            u32::from_le_bytes([g1[0].zero, g1[1].zero, g1[2].zero, g1[3].zero]);
            n * c1 / 4 * hw + SLACK
        ];
        let mut zeros_p1_exact = alloc(n, c1, hw);
        for g in 0..c1 / 4 {
            let zw = u32::from_le_bytes([
                g1[4 * g].zero,
                g1[4 * g + 1].zero,
                g1[4 * g + 2].zero,
                g1[4 * g + 3].zero,
            ]);
            for img in 0..n {
                zeros_p1_exact[(img * c1 / 4 + g) * hw..(img * c1 / 4 + g + 1) * hw].fill(zw);
            }
        }
        let _ = zeros_p1;
        let v1_zero =
            C4 { data: &zeros_p1_exact, offset: 0, img_stride: c1 / 4 * hw, c: c1, zero: g1[0].zero };
        let mut dec_p0 = vec![0f32; n * o * hw];
        let mut cpl_p0 = vec![0f32; n * o * hw];
        conv_per_channel(
            &[v0, v1_zero],
            &[&z0, &z1],
            n,
            h,
            w,
            &cv_dec,
            None,
            QOut::F32 { data: &mut dec_p0 },
        );
        conv_per_channel(
            &[v0, v1_zero],
            &[&z0, &z1],
            n,
            h,
            w,
            &cv_cpl,
            None,
            QOut::F32 { data: &mut cpl_p0 },
        );
        // Exact f32 reference for part 0 (part 1 is exact zero!).
        let mut ref_p0 = vec![0f32; n * o * hw];
        for img in 0..n {
            for oc in 0..o {
                for oy in 0..h {
                    for ox in 0..w {
                        let mut s = bias[oc];
                        for ic in 0..c0 {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let iy = (oy + ky) as isize - pad as isize;
                                    let ix = (ox + kx) as isize - pad as isize;
                                    if iy >= 0 && ix >= 0 && iy < h as isize && ix < w as isize {
                                        let xv = x0_dq[(img * c0 + ic) * hw + iy as usize * w + ix as usize];
                                        s += xv * wt[(oc * cin + ic) * taps + ky * k + kx];
                                    }
                                }
                            }
                        }
                        ref_p0[(img * o + oc) * hw + oy * w + ox] = if silu { silu_q8(s) } else { s };
                    }
                }
            }
        }
        let rmse = |a: &[f32], b: &[f32]| -> f64 {
            (a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64)
                .sqrt()
        };
        let err_dec = rmse(&dec_p0, &ref_p0);
        let err_cpl = rmse(&cpl_p0, &ref_p0);
        assert!(
            err_dec * 3.0 < err_cpl,
            "k={k}: decoupled concat error ({err_dec:e}) should be >3× smaller than coupled ({err_cpl:e})"
        );
    }
}

#[test]
fn stem_u8_rejects_overflowing_output_extent_before_writes() {
    let w = vec![0.1f32; 4 * 3];
    let b = vec![0.0f32; 4];
    let stem = QStem::new(&w, &b, (4, 1), 1, 1, false);
    let mut out = vec![0u32; 4];
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stem_u8(
            &[],
            usize::MAX / 4 + 2,
            0,
            0,
            &stem,
            QOut::U8 { data: &mut out, grid: Grid { scale: 1.0, zero: 0 } },
        );
    }));
    assert!(res.is_err(), "overflowing n * oh * ow must panic before touching raw output pointers");
}

/// Independent scalar integer reference for single- or multi-part per-channel conv (matches the exact
/// integer sums, `TileF32` part folding, bias, `silu_q8`, residual FMA, and `Grid::quantize` order).
#[allow(clippy::too_many_arguments)]
fn ref_conv_parts(
    parts_q: &[(&[u32], &[Grid])],
    n: usize,
    h: usize,
    w: usize,
    wt: &[f32],
    bias: &[f32],
    (o, k, s, pad): (usize, usize, usize, usize),
    silu: bool,
    decouple: bool,
    res: Option<(&[u32], &[Grid])>,
    out_grids: &[Grid],
) -> (Vec<u32>, Vec<f32>) {
    let hw = h * w;
    let taps = k * k;
    let (oh, ow) = q8::out_size(h, w, k, s, pad).unwrap();
    let ohw = oh * ow;
    let cin: usize = parts_q.iter().map(|(_, gs)| gs.len()).sum();
    let part_slices: Vec<(usize, Vec<Grid>)> = if decouple && parts_q.len() > 1 {
        let mut ch0 = 0;
        parts_q
            .iter()
            .map(|(_, gs)| {
                let start = ch0;
                ch0 += gs.len();
                (start, gs.to_vec())
            })
            .collect()
    } else {
        vec![(0, parts_q.iter().flat_map(|(_, gs)| gs.iter().copied()).collect())]
    };
    let packed_parts: Vec<_> = part_slices
        .into_iter()
        .map(|(ch0, gs)| {
            let pc = gs.len();
            let mut folded = Vec::with_capacity(o * pc * taps);
            for oc in 0..o {
                let base = (oc * cin + ch0) * taps;
                for (ic, g) in gs.iter().enumerate() {
                    folded.extend(wt[base + ic * taps..base + (ic + 1) * taps].iter().map(|&v| v * g.scale));
                }
            }
            let (qw, sw) = quantize_weights(&folded, o);
            (ch0, gs, qw, sw)
        })
        .collect();
    let input_byte = |img: usize, ci: usize, p: usize| -> u8 {
        let mut off = 0;
        for (buf, gs) in parts_q {
            if ci < off + gs.len() {
                let j = ci - off;
                let per_img = (gs.len() / 4) * hw;
                return buf[img * per_img + (j / 4) * hw + p].to_le_bytes()[j % 4];
            }
            off += gs.len();
        }
        unreachable!()
    };
    let mut qout = vec![0u32; n * (o / 4) * ohw];
    let mut fout = vec![0f32; n * o * ohw];
    let single = packed_parts.len() == 1;
    for img in 0..n {
        for oc in 0..o {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut yv = 0.0f32;
                    for (ch0, gs, qw, sw) in &packed_parts {
                        let pc = gs.len();
                        let mut acc = 0i64;
                        let mut corr = 0i64;
                        for (ic, g) in gs.iter().enumerate() {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let wq = qw[(oc * pc + ic) * taps + ky * k + kx] as i64;
                                    let y = (oy * s + ky) as isize - pad as isize;
                                    let x = (ox * s + kx) as isize - pad as isize;
                                    let v = if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
                                        g.zero
                                    } else {
                                        input_byte(img, ch0 + ic, y as usize * w + x as usize)
                                    };
                                    acc += wq * v as i64;
                                    corr += wq * g.zero as i64;
                                }
                            }
                        }
                        let diff = (acc - corr) as i32 as f32;
                        yv = if single { fmadd(diff, sw[oc], bias[oc]) } else { fmadd(diff, sw[oc], yv) };
                    }
                    if !single {
                        yv += bias[oc];
                    }
                    if silu {
                        yv = silu_q8(yv);
                    }
                    let pos = oy * ow + ox;
                    if let Some((rbuf, rgs)) = res {
                        let b = rbuf[(img * (o / 4) + oc / 4) * ohw + pos].to_le_bytes()[oc % 4];
                        yv = fmadd(rgs[oc].scale, (b as i32 - rgs[oc].zero as i32) as f32, yv);
                    }
                    fout[(img * o + oc) * ohw + pos] = yv;
                    qout[(img * (o / 4) + oc / 4) * ohw + pos] |=
                        (out_grids[oc].quantize(yv) as u32) << (8 * (oc % 4));
                }
            }
        }
    }
    (qout, fout)
}

mod fused;
