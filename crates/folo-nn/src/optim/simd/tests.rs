use super::norm::{
    check_finite_scalar, check_sum_sq_f64_scalar, scale2_scalar, unscale_check_scalar,
    unscale_check_sum_sq_scalar,
};
use super::vec_ops::{axpby_scalar, muon_momentum_nesterov_scalar};
use super::*;

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_kernels_match_scalar_bitwise() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    for n in [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 257] {
        let p0: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 4.5).collect();
        let g0: Vec<f32> = (0..n).map(|i| (i as f32) * -0.07 + 1.25).collect();
        let b0: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 0.5).collect();
        let v0: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 + 0.01).collect();

        for (wd, nesterov) in [(0.0, false), (0.0, true), (5e-4, false), (5e-4, true)] {
            let (mut ps, mut bs) = (p0.clone(), b0.clone());
            let (mut pv, mut bv) = (p0.clone(), b0.clone());
            sgd_coupled_mom_scalar(&mut ps, &g0, &mut bs, 0.01, 0.9, wd, wd != 0.0, nesterov);
            sgd_coupled_slice(&mut pv, &g0, Some(&mut bv), 0.01, 0.9, wd, nesterov);
            assert_eq!(
                ps.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                pv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sgd coupled p n={n} wd={wd} nesterov={nesterov}"
            );
            assert_eq!(
                bs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sgd coupled b n={n} wd={wd} nesterov={nesterov}"
            );

            let (mut ps, mut pv) = (p0.clone(), p0.clone());
            sgd_coupled_nomom_scalar(&mut ps, &g0, 0.01, wd, wd != 0.0);
            sgd_coupled_slice(&mut pv, &g0, None, 0.01, 0.0, wd, nesterov);
            assert_eq!(
                ps.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                pv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sgd coupled nomom n={n} wd={wd}"
            );
        }

        for (wd_step, cautious, nesterov) in [
            (0.0, false, false),
            (0.0, false, true),
            (1e-4, false, false),
            (1e-4, false, true),
            (1e-4, true, false),
            (1e-4, true, true),
        ] {
            let (mut ps, mut bs) = (p0.clone(), b0.clone());
            let (mut pv, mut bv) = (p0.clone(), b0.clone());
            sgd_decoupled_mom_scalar(
                &mut ps,
                &g0,
                &mut bs,
                0.01,
                0.9,
                wd_step,
                wd_step != 0.0,
                cautious,
                nesterov,
            );
            sgd_decoupled_slice(&mut pv, &g0, Some(&mut bv), 0.01, 0.9, wd_step, nesterov, cautious);
            assert_eq!(
                ps.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                pv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sgd decoupled p n={n} wd={wd_step} c={cautious} nest={nesterov}"
            );
            assert_eq!(
                bs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sgd decoupled b n={n} wd={wd_step} c={cautious} nest={nesterov}"
            );

            let (mut ps, mut pv) = (p0.clone(), p0.clone());
            sgd_decoupled_nomom_scalar(&mut ps, &g0, 0.01, wd_step, wd_step != 0.0, cautious);
            sgd_decoupled_slice(&mut pv, &g0, None, 0.01, 0.0, wd_step, nesterov, cautious);
            assert_eq!(
                ps.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                pv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "sgd decoupled nomom n={n} wd={wd_step} c={cautious}"
            );
        }

        for decay in [None, Some(0.999)] {
            let (mut ps, mut ms, mut vs) = (p0.clone(), b0.clone(), v0.clone());
            let (mut pv, mut mv, mut vv) = (p0.clone(), b0.clone(), v0.clone());
            adamw_scalar(&mut ps, &g0, &mut ms, &mut vs, decay, 0.9, 0.1, 0.999, 0.001, 0.005, 0.5, 1e-8);
            adamw_slice(&mut pv, &g0, &mut mv, &mut vv, decay, 0.9, 0.1, 0.999, 0.001, 0.005, 0.5, 1e-8);
            assert_eq!(
                ps.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                pv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "adamw p n={n} decay={decay:?}"
            );
            assert_eq!(
                ms.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                mv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "adamw m n={n}"
            );
            assert_eq!(
                vs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                vv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "adamw v n={n}"
            );
        }

        for (a, b) in [(1.0, 1.0), (1.0, -0.02), (0.9, 1.0), (0.9, 0.1)] {
            let (mut ds, mut dv) = (p0.clone(), p0.clone());
            axpby_scalar(&mut ds, &g0, a, b);
            axpby_slice(&mut dv, &g0, a, b);
            assert_eq!(
                ds.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                dv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "axpby n={n} a={a} b={b}"
            );
        }
        let (mut bs, mut us) = (b0.clone(), vec![0.0; n]);
        let (mut bv, mut uv) = (b0.clone(), vec![0.0; n]);
        muon_momentum_nesterov_scalar(&mut bs, &g0, &mut us, 0.937, 1.0 - 0.937);
        muon_momentum_nesterov_slice(&mut bv, &g0, &mut uv, 0.937);
        assert_eq!(
            bs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "muon b n={n}"
        );
        assert_eq!(
            us.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            uv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "muon u n={n}"
        );

        let (mut as_, mut av) = (p0.clone(), p0.clone());
        let fs = unscale_check_scalar(&mut as_, 0.25);
        let fv = unscale_check_slice(&mut av, 0.25);
        assert_eq!(fs, fv);
        assert_eq!(
            as_.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            av.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "unscale n={n}"
        );

        let (mut as_, mut av) = (p0.clone(), p0.clone());
        let (fs, sqs) = unscale_check_sum_sq_scalar(&mut as_, 0.25);
        let (fv, sqv) = unscale_check_sum_sq_slice(&mut av, 0.25);
        assert_eq!(fs, fv);
        assert_eq!(
            as_.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            av.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "unscale_sum_sq buf n={n}"
        );
        assert!((sqs - sqv).abs() <= 1e-12 * (1.0 + sqs.abs()), "unscale_sum_sq n={n}: {sqs} vs {sqv}");
        let sq_only = sum_sq_f64_slice(&av);
        assert!((sqs - sq_only).abs() <= 1e-12 * (1.0 + sqs.abs()), "sum_sq_f64 n={n}: {sqs} vs {sq_only}");

        let (mut s2s, mut s2v) = (p0.clone(), p0.clone());
        scale2_scalar(&mut s2s, 1e-19, 2.5e-20);
        scale2_slice(&mut s2v, 1e-19, 2.5e-20);
        assert_eq!(
            s2s.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            s2v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "scale2 n={n}"
        );

        assert_eq!(check_finite_scalar(&p0), check_finite_slice(&p0), "check_finite n={n}");
        let (cfs, csqs) = check_sum_sq_f64_scalar(&p0);
        let (cfv, csqv) = check_sum_sq_f64_slice(&p0);
        assert_eq!(cfs, cfv, "check_sum_sq finite n={n}");
        assert!((csqs - csqv).abs() <= 1e-12 * (1.0 + csqs.abs()), "check_sum_sq n={n}: {csqs} vs {csqv}");
    }

    for pos in [0usize, 7, 8, 15, 31, 32, 34] {
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut v = vec![1.0f32; 35];
            v[pos] = bad;
            assert!(!check_finite_slice(&v), "check_finite pos={pos} bad={bad}");
            assert!(!check_sum_sq_f64_slice(&v).0, "check_sum_sq pos={pos} bad={bad}");
            let mut v2 = v.clone();
            assert!(!unscale_check_slice(&mut v, 0.5), "unscale pos={pos} bad={bad}");
            let (ok, _) = unscale_check_sum_sq_slice(&mut v2, 0.5);
            assert!(!ok, "unscale_sum_sq pos={pos} bad={bad}");
        }
    }
    let mut ov = vec![1.0f32; 35];
    ov[17] = f32::MAX;
    assert!(!unscale_check_slice(&mut ov, 2.0));
    let mut ov2 = vec![1.0f32; 35];
    ov2[17] = f32::MAX;
    assert!(!unscale_check_sum_sq_slice(&mut ov2, 2.0).0);
}

#[test]
fn scaled_sum_sq_extreme_and_nonfinite() {
    // Huge f32 values: 4 copies of f32::MAX -> norm = 2 * f32::MAX without overflow.
    let big = ScaledSumSq::from_sq_f64(sum_sq_f64_slice(&[f32::MAX; 4]));
    let want_big = 2.0 * (f32::MAX as f64);
    assert!((big.norm() - want_big).abs() <= 1e-12 * want_big);

    // Subnormal f32 values: 4 copies of min positive subnormal -> norm = 2 * subnormal without underflow.
    let tiny_val = f32::from_bits(1);
    let tiny = ScaledSumSq::from_sq_f64(sum_sq_f64_slice(&[tiny_val; 4]));
    let want_tiny = 2.0 * (tiny_val as f64);
    assert!((tiny.norm() - want_tiny).abs() <= 1e-12 * want_tiny);

    // Huge scales near f64::MAX via merge: no overflow.
    let huge = ScaledSumSq { scale: f64::MAX / 4.0, ssq: 1.0 };
    let a = huge.merge(huge).merge(huge).merge(huge);
    let want_f64 = f64::MAX / 2.0;
    assert!((a.norm() - want_f64).abs() <= 1e-12 * want_f64);

    // Tiny scales near f64::MIN_POSITIVE via merge: no underflow.
    let small = ScaledSumSq { scale: f64::MIN_POSITIVE, ssq: 1.0 };
    let b = small.merge(small).merge(small).merge(small);
    assert!((b.norm() - 2.0 * f64::MIN_POSITIVE).abs() <= 1e-12 * f64::MIN_POSITIVE);

    // Merge combines different scales accurately and preserves zero identity.
    assert_eq!(ScaledSumSq::ZERO.merge(big).norm(), big.norm());
    assert_eq!(big.merge(ScaledSumSq::ZERO).norm(), big.norm());
    let c1 = ScaledSumSq::from_sq_f64(9.0);
    let c2 = ScaledSumSq::from_sq_f64(16.0);
    assert!((c1.merge(c2).norm() - 5.0).abs() <= 1e-12);
    assert!((c2.merge(c1).norm() - 5.0).abs() <= 1e-12);

    // Non-finite propagation: NaN dominates +Inf, +Inf dominates finite.
    let inf = ScaledSumSq::from_sq_f64(f64::INFINITY);
    let nan = ScaledSumSq::from_sq_f64(f64::NAN);
    assert_eq!(inf.merge(big).norm(), f64::INFINITY);
    assert_eq!(big.merge(inf).norm(), f64::INFINITY);
    assert!(nan.merge(inf).norm().is_nan());
    assert!(inf.merge(nan).norm().is_nan());
}
