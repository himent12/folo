use super::vec_ops::{
    axpby_scalar, muon_momentum_nesterov_scalar, unscale_check_scalar, unscale_check_sum_sq_scalar,
};
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
    }

    for pos in [0usize, 7, 8, 15, 31, 32, 34] {
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut v = vec![1.0f32; 35];
            v[pos] = bad;
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
