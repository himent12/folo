//! GPU-vs-CPU parity and finite-difference gradient tests for tiled SDPA forward (with additive masks + LSE)
//! and FlashAttention-2 tiled SDPA backward (`dQ`, `dK`, `dV`).

use super::{Rng, assert_close, f16_round, gpu, to_f16};
use crate::kernels::FORCE_NO_SUBGROUPS;
use crate::kernels::attention::{SdpaMask, sdpa_backward, sdpa_forward_masked};
use crate::{DType, GpuTensor};

struct ForceGuard(bool);
impl ForceGuard {
    fn set(v: bool) -> Self {
        Self(FORCE_NO_SUBGROUPS.replace(v))
    }
}
impl Drop for ForceGuard {
    fn drop(&mut self) {
        FORCE_NO_SUBGROUPS.set(self.0);
    }
}

/// Exact f64 CPU reference for SDPA forward with optional additive mask, returning `(out, lse)`.
fn cpu_sdpa_fwd_masked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<(&[f32], bool)>,
    (b, sq, sk, d): (usize, usize, usize, usize),
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; b * sq * d];
    let mut lse = vec![0.0f32; b * sq];
    let mut scores = vec![0.0f64; sk];
    let mut valid = vec![false; sk];
    for bi in 0..b {
        let qb = bi * sq * d;
        let kb = bi * sk * d;
        for qi in 0..sq {
            let mut m = f64::NEG_INFINITY;
            let mut has_nan = false;
            for ki in 0..sk {
                if causal && ki > qi {
                    valid[ki] = false;
                    scores[ki] = f64::NEG_INFINITY;
                    continue;
                }
                let mv = mask.map_or(0.0f64, |(mb, batched)| {
                    let mb_off = if batched { bi * sq * sk } else { 0 };
                    mb[mb_off + qi * sk + ki] as f64
                });
                if mv == f64::NEG_INFINITY {
                    valid[ki] = false;
                    scores[ki] = f64::NEG_INFINITY;
                    continue;
                }
                valid[ki] = true;
                let mut dot = 0.0f64;
                for di in 0..d {
                    dot += q[qb + qi * d + di] as f64 * k[kb + ki * d + di] as f64;
                }
                let s = dot * scale as f64 + mv;
                scores[ki] = s;
                if s.is_nan() {
                    has_nan = true;
                } else if !has_nan && s > m {
                    m = s;
                }
            }
            if has_nan {
                m = f64::NAN;
            }
            if !has_nan && m == f64::NEG_INFINITY {
                lse[bi * sq + qi] = f32::NEG_INFINITY;
                continue;
            }
            let mut sum = 0.0f64;
            for ki in 0..sk {
                if !valid[ki] {
                    scores[ki] = 0.0;
                    continue;
                }
                let p = (scores[ki] - m).exp();
                scores[ki] = p;
                sum += p;
            }
            lse[bi * sq + qi] = (m + sum.ln()) as f32;
            for di in 0..d {
                let mut acc = 0.0f64;
                for ki in 0..sk {
                    if !valid[ki] {
                        continue;
                    }
                    acc += scores[ki] * v[kb + ki * d + di] as f64;
                }
                out[qb + qi * d + di] = (acc / sum) as f32;
            }
        }
    }
    (out, lse)
}

/// Exact f64 CPU reference for SDPA backward, returning `(dq, dk, dv)`.
#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_bwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    do_grad: &[f32],
    mask: Option<(&[f32], bool)>,
    (b, sq, sk, d): (usize, usize, usize, usize),
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dq = vec![0.0f64; b * sq * d];
    let mut dk = vec![0.0f64; b * sk * d];
    let mut dv = vec![0.0f64; b * sk * d];
    let mut p_row = vec![0.0f64; sk];
    let mut valid = vec![false; sk];
    for bi in 0..b {
        let qb = bi * sq * d;
        let kb = bi * sk * d;
        for qi in 0..sq {
            let mut m = f64::NEG_INFINITY;
            for ki in 0..sk {
                if causal && ki > qi {
                    valid[ki] = false;
                    continue;
                }
                let mv = mask.map_or(0.0f64, |(mb, batched)| {
                    let mb_off = if batched { bi * sq * sk } else { 0 };
                    mb[mb_off + qi * sk + ki] as f64
                });
                if mv == f64::NEG_INFINITY {
                    valid[ki] = false;
                    continue;
                }
                valid[ki] = true;
                let mut dot = 0.0f64;
                for di in 0..d {
                    dot += q[qb + qi * d + di] as f64 * k[kb + ki * d + di] as f64;
                }
                let s = dot * scale as f64 + mv;
                p_row[ki] = s;
                if s > m {
                    m = s;
                }
            }
            if m == f64::NEG_INFINITY {
                continue;
            }
            let mut sum = 0.0f64;
            for ki in 0..sk {
                if !valid[ki] {
                    p_row[ki] = 0.0;
                    continue;
                }
                let e = (p_row[ki] - m).exp();
                p_row[ki] = e;
                sum += e;
            }
            for ki in 0..sk {
                if valid[ki] {
                    p_row[ki] /= sum;
                }
            }
            // Output O_q and delta = dot(dO_q, O_q).
            let mut delta = 0.0f64;
            for di in 0..d {
                let mut o_val = 0.0f64;
                for ki in 0..sk {
                    if valid[ki] {
                        o_val += p_row[ki] * v[kb + ki * d + di] as f64;
                    }
                }
                delta += do_grad[qb + qi * d + di] as f64 * o_val;
            }
            // Accumulate dV, dK, dQ.
            for ki in 0..sk {
                if !valid[ki] {
                    continue;
                }
                let p = p_row[ki];
                let mut dp = 0.0f64;
                for di in 0..d {
                    let dov = do_grad[qb + qi * d + di] as f64;
                    dv[kb + ki * d + di] += p * dov;
                    dp += dov * v[kb + ki * d + di] as f64;
                }
                let ds = p * (dp - delta) * scale as f64;
                for di in 0..d {
                    dq[qb + qi * d + di] += ds * k[kb + ki * d + di] as f64;
                    dk[kb + ki * d + di] += ds * q[qb + qi * d + di] as f64;
                }
            }
        }
    }
    (
        dq.into_iter().map(|x| x as f32).collect(),
        dk.into_iter().map(|x| x as f32).collect(),
        dv.into_iter().map(|x| x as f32).collect(),
    )
}

#[test]
fn tiled_sdpa_backward_matches_cpu() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5d9a_b4c0);

    // Covers: tiny 1x1x1, odd non-multiples (Sq, Sk, D), sk < sq, sq < sk (causal tail keys with zero grad),
    // D <= 64 (CACHE_Q), D = 96/128 (uncached Q), and D = 160 (d_tiles > 1).
    type Case = ((usize, usize, usize, usize), &'static str);
    let cases: &[Case] = &[
        ((2, 1, 1, 1), "tiny 1x1x1"),
        ((2, 7, 13, 17), "odd cache_q d=17"),
        ((2, 37, 13, 32), "sk < sq d=32"),
        ((2, 13, 37, 32), "sq < sk d=32"),
        ((2, 35, 49, 48), "non-pow2 cache_q d=48"),
        ((2, 33, 33, 64), "cache_q d=64"),
        ((2, 21, 35, 96), "uncached q d=96"),
        ((1, 20, 35, 160), "multi d_tile d=160"),
    ];

    for &((b, sq, sk, d), label) in cases {
        let q = rng.vec(b * sq * d, -1.2, 1.2);
        let k = rng.vec(b * sk * d, -1.2, 1.2);
        let v = rng.vec(b * sk * d, -1.2, 1.2);
        let do_g = rng.vec(b * sq * d, -1.0, 1.0);
        let scale = 1.0 / (d as f32).sqrt();
        let gq = GpuTensor::from_slice(&ctx, &q, &[b * sq * d]).unwrap();
        let gk = GpuTensor::from_slice(&ctx, &k, &[b * sk * d]).unwrap();
        let gv = GpuTensor::from_slice(&ctx, &v, &[b * sk * d]).unwrap();
        let gdo = GpuTensor::from_slice(&ctx, &do_g, &[b * sq * d]).unwrap();

        for causal in [false, true] {
            let (o_ref, lse_ref) = cpu_sdpa_fwd_masked(&q, &k, &v, None, (b, sq, sk, d), scale, causal);
            let (dq_ref, dk_ref, dv_ref) =
                cpu_sdpa_bwd(&q, &k, &v, &do_g, None, (b, sq, sk, d), scale, causal);
            for portable in [false, true] {
                let _fg = ForceGuard::set(portable);
                let go = GpuTensor::empty(&ctx, &[b * sq * d], DType::F32).unwrap();
                let glse = GpuTensor::empty(&ctx, &[b * sq], DType::F32).unwrap();
                sdpa_forward_masked(
                    &ctx,
                    gq.raw(),
                    gk.raw(),
                    gv.raw(),
                    None,
                    go.raw(),
                    Some(glse.raw()),
                    (b, sq, sk, d),
                    scale,
                    causal,
                    DType::F32,
                )
                .unwrap();
                let tag = format!("{label} causal={causal} portable={portable}");
                assert_close(&go.to_vec().unwrap(), &o_ref, 2e-4, 2e-4, &format!("sdpa o {tag}"));
                assert_close(&glse.to_vec().unwrap(), &lse_ref, 2e-4, 2e-4, &format!("sdpa lse {tag}"));

                let gdq = GpuTensor::empty(&ctx, &[b * sq * d], DType::F32).unwrap();
                let gdk = GpuTensor::empty(&ctx, &[b * sk * d], DType::F32).unwrap();
                let gdv = GpuTensor::empty(&ctx, &[b * sk * d], DType::F32).unwrap();
                sdpa_backward(
                    &ctx,
                    gq.raw(),
                    gk.raw(),
                    gv.raw(),
                    go.raw(),
                    glse.raw(),
                    gdo.raw(),
                    None,
                    gdq.raw(),
                    gdk.raw(),
                    gdv.raw(),
                    (b, sq, sk, d),
                    scale,
                    causal,
                    DType::F32,
                )
                .unwrap();
                assert_close(&gdq.to_vec().unwrap(), &dq_ref, 5e-4, 5e-4, &format!("sdpa dq {tag}"));
                assert_close(&gdk.to_vec().unwrap(), &dk_ref, 5e-4, 5e-4, &format!("sdpa dk {tag}"));
                assert_close(&gdv.to_vec().unwrap(), &dv_ref, 5e-4, 5e-4, &format!("sdpa dv {tag}"));
            }
        }
    }
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn tiled_sdpa_additive_mask_and_stability() {
    let ctx = gpu!();
    let mut rng = Rng::new(0x5d9a_4a5c);

    // 1. Both batched `[b, sq, sk]` and broadcast `[1, sq, sk]` additive masks, with `-inf` masked entries,
    //    an all-masked query row (must yield O = 0, LSE = -inf, dQ = 0, zero contribution to dK/dV),
    //    and combined causal + additive mask.
    for &(batched, causal, d) in &[(true, false, 32usize), (false, true, 48usize), (true, true, 72usize)] {
        let (b, sq, sk) = (2, 19, 25);
        let q = rng.vec(b * sq * d, -1.0, 1.0);
        let k = rng.vec(b * sk * d, -1.0, 1.0);
        let v = rng.vec(b * sk * d, -1.0, 1.0);
        let do_g = rng.vec(b * sq * d, -1.0, 1.0);
        let mask_len = if batched { b * sq * sk } else { sq * sk };
        let mut mask = rng.vec(mask_len, -0.5, 0.5);
        // Mask out scattered positions and mask out the entire row qi = 3 in slice 0.
        for ki in 0..sk {
            mask[3 * sk + ki] = f32::NEG_INFINITY;
            if ki % 5 == 4 {
                mask[7 * sk + ki] = f32::NEG_INFINITY;
            }
        }
        let scale = 1.0 / (d as f32).sqrt();
        let (o_ref, lse_ref) =
            cpu_sdpa_fwd_masked(&q, &k, &v, Some((&mask, batched)), (b, sq, sk, d), scale, causal);
        let (dq_ref, dk_ref, dv_ref) =
            cpu_sdpa_bwd(&q, &k, &v, &do_g, Some((&mask, batched)), (b, sq, sk, d), scale, causal);

        let gq = GpuTensor::from_slice(&ctx, &q, &[b * sq * d]).unwrap();
        let gk = GpuTensor::from_slice(&ctx, &k, &[b * sk * d]).unwrap();
        let gv = GpuTensor::from_slice(&ctx, &v, &[b * sk * d]).unwrap();
        let gdo = GpuTensor::from_slice(&ctx, &do_g, &[b * sq * d]).unwrap();
        let gm = GpuTensor::from_slice(&ctx, &mask, &[mask_len]).unwrap();
        let smask = Some(SdpaMask { buf: gm.raw(), batched });

        for portable in [false, true] {
            let _fg = ForceGuard::set(portable);
            let go = GpuTensor::empty(&ctx, &[b * sq * d], DType::F32).unwrap();
            let glse = GpuTensor::empty(&ctx, &[b * sq], DType::F32).unwrap();
            sdpa_forward_masked(
                &ctx,
                gq.raw(),
                gk.raw(),
                gv.raw(),
                smask,
                go.raw(),
                Some(glse.raw()),
                (b, sq, sk, d),
                scale,
                causal,
                DType::F32,
            )
            .unwrap();
            let tag = format!("mask batched={batched} causal={causal} d={d} portable={portable}");
            let got_o = go.to_vec().unwrap();
            let got_lse = glse.to_vec().unwrap();
            assert_close(&got_o, &o_ref, 3e-4, 3e-4, &format!("o {tag}"));
            assert_close(&got_lse, &lse_ref, 3e-4, 3e-4, &format!("lse {tag}"));
            // All-masked row 3 must have exact 0 output and -inf LSE.
            assert!(got_o[3 * d..4 * d].iter().all(|&x| x == 0.0), "all-masked O non-zero: {tag}");
            assert_eq!(got_lse[3], f32::NEG_INFINITY, "all-masked LSE != -inf: {tag}");

            let gdq = GpuTensor::empty(&ctx, &[b * sq * d], DType::F32).unwrap();
            let gdk = GpuTensor::empty(&ctx, &[b * sk * d], DType::F32).unwrap();
            let gdv = GpuTensor::empty(&ctx, &[b * sk * d], DType::F32).unwrap();
            sdpa_backward(
                &ctx,
                gq.raw(),
                gk.raw(),
                gv.raw(),
                go.raw(),
                glse.raw(),
                gdo.raw(),
                smask,
                gdq.raw(),
                gdk.raw(),
                gdv.raw(),
                (b, sq, sk, d),
                scale,
                causal,
                DType::F32,
            )
            .unwrap();
            let got_dq = gdq.to_vec().unwrap();
            assert_close(&got_dq, &dq_ref, 5e-4, 5e-4, &format!("dq {tag}"));
            assert_close(&gdk.to_vec().unwrap(), &dk_ref, 5e-4, 5e-4, &format!("dk {tag}"));
            assert_close(&gdv.to_vec().unwrap(), &dv_ref, 5e-4, 5e-4, &format!("dv {tag}"));
            assert!(got_dq[3 * d..4 * d].iter().all(|&x| x == 0.0), "all-masked dQ non-zero: {tag}");
        }
    }

    // 2. Additive `-inf` mask isolation: NaN/Inf in `K`/`V` at masked columns must NOT leak into O or gradients
    //    of unmasked positions, and NaN/Inf in `dO` at earlier queries `q < k` under causal mask must NOT leak
    //    into `dK[k]`/`dV[k]`.
    let (b, sq, sk, d) = (1, 10, 12, 16);
    let q = rng.vec(sq * d, -1.0, 1.0);
    let mut k = rng.vec(sk * d, -1.0, 1.0);
    let mut v = rng.vec(sk * d, -1.0, 1.0);
    let do_g = rng.vec(sq * d, -1.0, 1.0);
    // Poison columns 8..12 with NaN/Inf and mask them out with `-inf` in additive mask for every query.
    for ki in 8..sk {
        for di in 0..d {
            k[ki * d + di] = if ki % 2 == 0 { f32::NAN } else { f32::INFINITY };
            v[ki * d + di] = if ki % 2 == 0 { f32::NAN } else { f32::NEG_INFINITY };
        }
    }
    let mut mask = vec![0.0f32; sq * sk];
    for qi in 0..sq {
        for ki in 8..sk {
            mask[qi * sk + ki] = f32::NEG_INFINITY;
        }
    }
    let gq = GpuTensor::from_slice(&ctx, &q, &[sq * d]).unwrap();
    let gk = GpuTensor::from_slice(&ctx, &k, &[sk * d]).unwrap();
    let gv = GpuTensor::from_slice(&ctx, &v, &[sk * d]).unwrap();
    let gdo = GpuTensor::from_slice(&ctx, &do_g, &[sq * d]).unwrap();
    let gm = GpuTensor::from_slice(&ctx, &mask, &[sq * sk]).unwrap();
    let go = GpuTensor::empty(&ctx, &[sq * d], DType::F32).unwrap();
    let glse = GpuTensor::empty(&ctx, &[sq], DType::F32).unwrap();
    let gdq = GpuTensor::empty(&ctx, &[sq * d], DType::F32).unwrap();
    let gdk = GpuTensor::empty(&ctx, &[sk * d], DType::F32).unwrap();
    let gdv = GpuTensor::empty(&ctx, &[sk * d], DType::F32).unwrap();
    let sm = Some(SdpaMask { buf: gm.raw(), batched: false });
    sdpa_forward_masked(
        &ctx,
        gq.raw(),
        gk.raw(),
        gv.raw(),
        sm,
        go.raw(),
        Some(glse.raw()),
        (b, sq, sk, d),
        0.25,
        false,
        DType::F32,
    )
    .unwrap();
    sdpa_backward(
        &ctx,
        gq.raw(),
        gk.raw(),
        gv.raw(),
        go.raw(),
        glse.raw(),
        gdo.raw(),
        sm,
        gdq.raw(),
        gdk.raw(),
        gdv.raw(),
        (b, sq, sk, d),
        0.25,
        false,
        DType::F32,
    )
    .unwrap();
    assert!(go.to_vec().unwrap().iter().all(|x| x.is_finite()), "masked NaN leaked into O");
    assert!(gdq.to_vec().unwrap().iter().all(|x| x.is_finite()), "masked NaN leaked into dQ");
    assert!(
        gdk.to_vec().unwrap()[..8 * d].iter().all(|x| x.is_finite()),
        "masked NaN leaked into unmasked dK"
    );
    assert!(
        gdv.to_vec().unwrap()[..8 * d].iter().all(|x| x.is_finite()),
        "masked NaN leaked into unmasked dV"
    );
    assert!(gdk.to_vec().unwrap()[8 * d..].iter().all(|&x| x == 0.0), "masked dK should be zero");
    assert!(gdv.to_vec().unwrap()[8 * d..].iter().all(|&x| x == 0.0), "masked dV should be zero");

    // 3. Finite-difference gradient check on GPU SDPA backward (with both causal + additive mask).
    let (b, sq, sk, d) = (1, 5, 6, 8);
    let q = rng.vec(sq * d, -0.8, 0.8);
    let k = rng.vec(sk * d, -0.8, 0.8);
    let v = rng.vec(sk * d, -0.8, 0.8);
    let do_g = rng.vec(sq * d, -0.8, 0.8);
    let mask = rng.vec(sq * sk, -0.3, 0.3);
    let scale = 1.0 / (d as f32).sqrt();
    let gq = GpuTensor::from_slice(&ctx, &q, &[sq * d]).unwrap();
    let gk = GpuTensor::from_slice(&ctx, &k, &[sk * d]).unwrap();
    let gv = GpuTensor::from_slice(&ctx, &v, &[sk * d]).unwrap();
    let gdo = GpuTensor::from_slice(&ctx, &do_g, &[sq * d]).unwrap();
    let gm = GpuTensor::from_slice(&ctx, &mask, &[sq * sk]).unwrap();
    let go = GpuTensor::empty(&ctx, &[sq * d], DType::F32).unwrap();
    let glse = GpuTensor::empty(&ctx, &[sq], DType::F32).unwrap();
    let gdq = GpuTensor::empty(&ctx, &[sq * d], DType::F32).unwrap();
    let gdk = GpuTensor::empty(&ctx, &[sk * d], DType::F32).unwrap();
    let gdv = GpuTensor::empty(&ctx, &[sk * d], DType::F32).unwrap();
    let sm = Some(SdpaMask { buf: gm.raw(), batched: false });
    sdpa_forward_masked(
        &ctx,
        gq.raw(),
        gk.raw(),
        gv.raw(),
        sm,
        go.raw(),
        Some(glse.raw()),
        (b, sq, sk, d),
        scale,
        true,
        DType::F32,
    )
    .unwrap();
    sdpa_backward(
        &ctx,
        gq.raw(),
        gk.raw(),
        gv.raw(),
        go.raw(),
        glse.raw(),
        gdo.raw(),
        sm,
        gdq.raw(),
        gdk.raw(),
        gdv.raw(),
        (b, sq, sk, d),
        scale,
        true,
        DType::F32,
    )
    .unwrap();
    let loss_fn = |qv: &[f32], kv: &[f32], vv: &[f32]| -> f64 {
        let (o, _) = cpu_sdpa_fwd_masked(qv, kv, vv, Some((&mask, false)), (b, sq, sk, d), scale, true);
        o.iter().zip(&do_g).map(|(&a, &g)| a as f64 * g as f64).sum()
    };
    let h = 1e-3f32;
    let dq_got = gdq.to_vec().unwrap();
    let dk_got = gdk.to_vec().unwrap();
    let dv_got = gdv.to_vec().unwrap();
    for idx in (0..q.len()).step_by(5) {
        let mut qp = q.clone();
        let mut qm = q.clone();
        qp[idx] += h;
        qm[idx] -= h;
        let num = ((loss_fn(&qp, &k, &v) - loss_fn(&qm, &k, &v)) / (2.0 * h as f64)) as f32;
        assert!((dq_got[idx] - num).abs() < 2e-3, "sdpa fd dq[{idx}]: got {}, fd {num}", dq_got[idx]);
    }
    for idx in (0..k.len()).step_by(7) {
        let mut kp = k.clone();
        let mut km = k.clone();
        kp[idx] += h;
        km[idx] -= h;
        let num = ((loss_fn(&q, &kp, &v) - loss_fn(&q, &km, &v)) / (2.0 * h as f64)) as f32;
        assert!((dk_got[idx] - num).abs() < 2e-3, "sdpa fd dk[{idx}]: got {}, fd {num}", dk_got[idx]);
        let mut vp = v.clone();
        let mut vm = v.clone();
        vp[idx] += h;
        vm[idx] -= h;
        let num_v = ((loss_fn(&q, &k, &vp) - loss_fn(&q, &k, &vm)) / (2.0 * h as f64)) as f32;
        assert!((dv_got[idx] - num_v).abs() < 2e-3, "sdpa fd dv[{idx}]: got {}, fd {num_v}", dv_got[idx]);
    }

    // 4. Empty `sq == 0` zeroes `dK` and `dV`; `sk == 0` returns Err; `F16` backward parity when supported.
    let dummy = GpuTensor::from_slice(&ctx, &[1.0], &[1]).unwrap();
    let gdk0 = GpuTensor::from_slice(&ctx, &[9.0; 8], &[8]).unwrap();
    let gdv0 = GpuTensor::from_slice(&ctx, &[8.0; 8], &[8]).unwrap();
    sdpa_backward(
        &ctx,
        dummy.raw(),
        dummy.raw(),
        dummy.raw(),
        dummy.raw(),
        dummy.raw(),
        dummy.raw(),
        None,
        dummy.raw(),
        gdk0.raw(),
        gdv0.raw(),
        (1, 0, 2, 4),
        0.5,
        false,
        DType::F32,
    )
    .unwrap();
    assert_eq!(gdk0.to_vec().unwrap(), vec![0.0; 8]);
    assert_eq!(gdv0.to_vec().unwrap(), vec![0.0; 8]);
    assert!(
        sdpa_backward(
            &ctx,
            dummy.raw(),
            dummy.raw(),
            dummy.raw(),
            dummy.raw(),
            dummy.raw(),
            dummy.raw(),
            None,
            dummy.raw(),
            dummy.raw(),
            dummy.raw(),
            (1, 1, 0, 1),
            1.0,
            false,
            DType::F32,
        )
        .is_err()
    );

    if ctx.caps().shader_f16 {
        let (b, sq, sk, d) = (2, 18, 22, 32);
        let q = f16_round(&rng.vec(b * sq * d, -0.8, 0.8));
        let k = f16_round(&rng.vec(b * sk * d, -0.8, 0.8));
        let v = f16_round(&rng.vec(b * sk * d, -0.8, 0.8));
        let do_g = f16_round(&rng.vec(b * sq * d, -0.8, 0.8));
        let scale = 1.0 / (d as f32).sqrt();
        let (o_ref, _) = cpu_sdpa_fwd_masked(&q, &k, &v, None, (b, sq, sk, d), scale, true);
        let (dq_ref, dk_ref, dv_ref) = cpu_sdpa_bwd(&q, &k, &v, &do_g, None, (b, sq, sk, d), scale, true);

        let gq = GpuTensor::from_f16(&ctx, &to_f16(&q), &[b * sq * d]).unwrap();
        let gk = GpuTensor::from_f16(&ctx, &to_f16(&k), &[b * sk * d]).unwrap();
        let gv = GpuTensor::from_f16(&ctx, &to_f16(&v), &[b * sk * d]).unwrap();
        let gdo = GpuTensor::from_f16(&ctx, &to_f16(&do_g), &[b * sq * d]).unwrap();
        let go = GpuTensor::empty(&ctx, &[b * sq * d], DType::F16).unwrap();
        let glse = GpuTensor::empty(&ctx, &[b * sq], DType::F32).unwrap();
        let gdq = GpuTensor::empty(&ctx, &[b * sq * d], DType::F16).unwrap();
        let gdk = GpuTensor::empty(&ctx, &[b * sk * d], DType::F16).unwrap();
        let gdv = GpuTensor::empty(&ctx, &[b * sk * d], DType::F16).unwrap();
        sdpa_forward_masked(
            &ctx,
            gq.raw(),
            gk.raw(),
            gv.raw(),
            None,
            go.raw(),
            Some(glse.raw()),
            (b, sq, sk, d),
            scale,
            true,
            DType::F16,
        )
        .unwrap();
        sdpa_backward(
            &ctx,
            gq.raw(),
            gk.raw(),
            gv.raw(),
            go.raw(),
            glse.raw(),
            gdo.raw(),
            None,
            gdq.raw(),
            gdk.raw(),
            gdv.raw(),
            (b, sq, sk, d),
            scale,
            true,
            DType::F16,
        )
        .unwrap();
        assert_close(&go.to_vec().unwrap(), &o_ref, 5e-3, 5e-3, "sdpa f16 o");
        assert_close(&gdq.to_vec().unwrap(), &dq_ref, 1e-2, 1e-2, "sdpa f16 dq");
        assert_close(&gdk.to_vec().unwrap(), &dk_ref, 1e-2, 1e-2, "sdpa f16 dk");
        assert_close(&gdv.to_vec().unwrap(), &dv_ref, 1e-2, 1e-2, "sdpa f16 dv");
    }

    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
