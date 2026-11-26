use super::*;

#[test]
fn binary_and_multiclass_derivatives_match_finite_differences() {
    let logits = [-3.2f32, -1.1, 0.0, 0.7, 2.4, 5.1];
    let targets = [0.0f32, 0.25, 0.5, 1.0, 0.8, 1.0];
    let eps = 1e-6f64;
    for &beta in &[0.1f64, 0.5, 1.0, 2.3, 5.0] {
        let (_, g, h) = binary_nll_grad_hess(&logits, &targets, beta);
        assert!(h >= 0.0, "binary Hessian must be non-negative, got {h}");
        let (fp, gp, _) = binary_nll_grad_hess(&logits, &targets, beta + eps);
        let (fm, gm, _) = binary_nll_grad_hess(&logits, &targets, beta - eps);
        let fd_g = (fp - fm) / (2.0 * eps);
        let fd_h = (gp - gm) / (2.0 * eps);
        assert!((g - fd_g).abs() < 1e-8, "binary g at beta={beta}: {g} vs fd {fd_g}");
        assert!((h - fd_h).abs() < 1e-7, "binary h at beta={beta}: {h} vs fd {fd_h}");
    }

    // Cancellation-free precision on confident positive logit (z = 40, y = 1, beta = 1).
    let (nll40, g40, h40) = binary_nll_grad_hess(&[40.0], &[1.0], 1.0);
    let e40 = (-40.0f64).exp();
    assert!((nll40 - e40.ln_1p()).abs() < 1e-30, "nll40 = {nll40:e} vs {:e}", e40.ln_1p());
    let want_g40 = -40.0 * e40 / (1.0 + e40);
    assert!((g40 - want_g40).abs() < 1e-28, "g40 = {g40:e} vs {want_g40:e}");
    assert!(h40 > 0.0 && h40.is_finite());

    let mc_logits = [
        2.1f32, -0.5, 0.3, -1.2, // sample 0
        -0.8, 1.7, 0.4, 0.9, // sample 1
        0.2, 0.2, 3.0, -2.1, // sample 2
        -1.5, -0.4, 0.1, 1.2, // sample 3
    ];
    let mc_labels = [0usize, 1, 2, 3];
    for &beta in &[0.15f64, 0.8, 1.0, 2.0, 4.0] {
        let (_, g, h) = multiclass_nll_grad_hess(&mc_logits, 4, &mc_labels, beta);
        assert!(h >= 0.0, "multiclass Hessian must be non-negative, got {h}");
        let (fp, gp, _) = multiclass_nll_grad_hess(&mc_logits, 4, &mc_labels, beta + eps);
        let (fm, gm, _) = multiclass_nll_grad_hess(&mc_logits, 4, &mc_labels, beta - eps);
        let fd_g = (fp - fm) / (2.0 * eps);
        let fd_h = (gp - gm) / (2.0 * eps);
        assert!((g - fd_g).abs() < 1e-8, "mc g at beta={beta}: {g} vs fd {fd_g}");
        assert!((h - fd_h).abs() < 1e-7, "mc h at beta={beta}: {h} vs fd {fd_h}");
    }
}

#[test]
fn multiclass_shift_invariance_and_large_constant_row_regression() {
    let cfg = CalibrationConfig::default();

    // 1. Two-row regression: constant huge row [1e20; 5] + separable row [0, 1, 0, 0, 0] must reach min_temp.
    let two_rows = [1e20f32, 1e20, 1e20, 1e20, 1e20, 0.0, 1.0, 0.0, 0.0, 0.0];
    let two_labels = [0usize, 1];
    let fit = TemperatureScaler::fit_multiclass(&two_rows, 5, &two_labels, &cfg).unwrap();
    assert!(fit.converged, "two-row multiclass fit should converge: {fit:?}");
    assert!(
        (fit.scaler.temperature() - cfg.min_temp).abs() < 1e-12,
        "expected min_temp={}, got {}",
        cfg.min_temp,
        fit.scaler.temperature()
    );
    assert!(fit.final_nll < fit.initial_nll);

    // 2. Exact shift invariance on [0, -30] vs [1e6, 1e6 - 30] (exact in f32).
    let base = [0.0f32, -30.0, 1.0, -2.0];
    let shifted: Vec<f32> = base.iter().map(|&z| z + 1e6).collect();
    let labels = [0usize, 0];
    let (n1, g1, h1) = multiclass_nll_grad_hess(&base, 2, &labels, 1.0);
    let (n2, g2, h2) = multiclass_nll_grad_hess(&shifted, 2, &labels, 1.0);
    assert!((n1 - n2).abs() < 1e-15, "shift NLL: {n1} vs {n2}");
    assert!((g1 - g2).abs() < 1e-15, "shift grad: {g1} vs {g2}");
    assert!((h1 - h2).abs() < 1e-15, "shift hess: {h1} vs {h2}");

    let fit_base = TemperatureScaler::fit_multiclass(&base, 2, &labels, &cfg).unwrap();
    let fit_shifted = TemperatureScaler::fit_multiclass(&shifted, 2, &labels, &cfg).unwrap();
    assert!((fit_base.scaler.temperature() - fit_shifted.scaler.temperature()).abs() < 1e-12);
}

#[test]
fn binary_temperature_scaling_recovers_overconfident_and_underconfident_scales() {
    let mut rng = folo_core::rng::Rng::seed(2026);
    let n = 4000usize;
    let mut true_logits = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let z = rng.uniform(-3.0, 3.0);
        let p = sigmoid_f64(z as f64) as f32;
        let u = rng.uniform(0.0, 1.0);
        true_logits.push(z);
        labels.push(u < p);
    }

    let cfg = CalibrationConfig::default();

    // 1. Overconfident logits: z = 2.5 * z_true -> optimal T ≈ 2.5.
    let over_logits: Vec<f32> = true_logits.iter().map(|&z| 2.5 * z).collect();
    let fit_over = TemperatureScaler::fit_binary(&over_logits, &labels, &cfg).unwrap();
    assert!(fit_over.converged, "overconfident fit should converge: {fit_over:?}");
    assert!(
        (fit_over.scaler.temperature() - 2.5).abs() < 0.25,
        "expected T ≈ 2.5, got {}",
        fit_over.scaler.temperature()
    );
    assert!(fit_over.final_nll < fit_over.initial_nll - 0.05);

    let raw_probs: Vec<f32> = over_logits.iter().map(|&z| sigmoid_f64(z as f64) as f32).collect();
    let cal_probs = fit_over.scaler.calibrate_logits_binary(&over_logits);
    let m_raw = calibration_metrics(&raw_probs, &labels, 15).unwrap();
    let m_cal = calibration_metrics(&cal_probs, &labels, 15).unwrap();
    assert!(m_cal.ece < m_raw.ece * 0.4, "ECE should drop sharply: raw={} vs cal={}", m_raw.ece, m_cal.ece);
    assert!(m_cal.brier < m_raw.brier, "Brier should improve: raw={} vs cal={}", m_raw.brier, m_cal.brier);

    // 2. Underconfident logits: z = 0.4 * z_true -> optimal T ≈ 0.4.
    let under_logits: Vec<f32> = true_logits.iter().map(|&z| 0.4 * z).collect();
    let fit_under = TemperatureScaler::fit_binary(&under_logits, &labels, &cfg).unwrap();
    assert!(fit_under.converged);
    assert!(
        (fit_under.scaler.temperature() - 0.4).abs() < 0.08,
        "expected T ≈ 0.4, got {}",
        fit_under.scaler.temperature()
    );
    assert!(fit_under.final_nll < fit_under.initial_nll - 0.05);

    // 3. Exact soft-target identity: y_i = sigmoid(z_i / 2.0) -> exact T == 2.0.
    let soft_targets: Vec<f32> = true_logits.iter().map(|&z| sigmoid_f64(z as f64 / 2.0) as f32).collect();
    let fit_exact = TemperatureScaler::fit_binary_targets(&true_logits, &soft_targets, &cfg).unwrap();
    assert!(fit_exact.converged);
    assert!(
        (fit_exact.scaler.temperature() - 2.0).abs() < 1e-6,
        "expected exact T = 2.0, got {}",
        fit_exact.scaler.temperature()
    );
}

#[test]
fn multiclass_temperature_scaling_recovers_scale_and_improves_brier() {
    let mut rng = folo_core::rng::Rng::seed(4242);
    let n = 3000usize;
    let nc = 4usize;
    let true_temp = 3.0f64;
    let mut logits = Vec::with_capacity(n * nc);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let row =
            [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)];
        let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: [f64; 4] = std::array::from_fn(|c| (row[c] - max_v).exp() as f64);
        let sum: f64 = exps.iter().sum();
        let u = rng.uniform(0.0, 1.0) as f64;
        let mut cdf = 0.0;
        let mut y = nc - 1;
        for (c, &e) in exps.iter().enumerate() {
            cdf += e / sum;
            if u < cdf {
                y = c;
                break;
            }
        }
        labels.push(y);
        for v in row {
            logits.push(v * true_temp as f32);
        }
    }

    let cfg = CalibrationConfig::default();
    let fit = TemperatureScaler::fit_multiclass(&logits, nc, &labels, &cfg).unwrap();
    assert!(fit.converged);
    assert!(
        (fit.scaler.temperature() - true_temp).abs() < 0.25,
        "expected multiclass T ≈ {true_temp}, got {}",
        fit.scaler.temperature()
    );
    assert!(fit.final_nll < fit.initial_nll - 0.1);

    let raw_probs = TemperatureScaler::default().calibrate_logits_multiclass(&logits, nc).unwrap();
    let cal_probs = fit.scaler.calibrate_logits_multiclass(&logits, nc).unwrap();
    for row in cal_probs.chunks_exact(nc) {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax row sum = {s}");
    }
    let raw_brier = multiclass_brier_score(&raw_probs, nc, &labels).unwrap();
    let cal_brier = multiclass_brier_score(&cal_probs, nc, &labels).unwrap();
    assert!(cal_brier < raw_brier, "multiclass Brier should decrease: {raw_brier} -> {cal_brier}");
}

#[test]
fn boundary_extreme_and_serde_invariants() {
    let cfg = CalibrationConfig::default();

    // 1. Perfectly separable data: mild logits hit min_temp bound; large logits reach near-zero NLL.
    let mild_sep = [-0.5f32, -0.2, 0.2, 0.5];
    let sep_labels = [false, false, true, true];
    let fit_mild_sep = TemperatureScaler::fit_binary(&mild_sep, &sep_labels, &cfg).unwrap();
    assert!(fit_mild_sep.converged);
    assert!((fit_mild_sep.scaler.temperature() - cfg.min_temp).abs() < 1e-12);

    let sep_logits = [-5.0f32, -2.0, 2.0, 5.0];
    let fit_sep = TemperatureScaler::fit_binary(&sep_logits, &sep_labels, &cfg).unwrap();
    assert!(fit_sep.converged);
    assert!(fit_sep.final_nll < 1e-8 && fit_sep.scaler.temperature() < 0.2);

    // 2. Completely inverted labels hit max_temp cleanly.
    let inv_labels = [true, true, false, false];
    let fit_inv = TemperatureScaler::fit_binary(&sep_logits, &inv_labels, &cfg).unwrap();
    assert!(fit_inv.converged);
    assert!((fit_inv.scaler.temperature() - cfg.max_temp).abs() < 1e-12);

    // 2b. Non-self-inverse IEEE-754 bounds (1.0 / (1.0 / T) != T) preserve exact bound & converged=true.
    let bound_cfg = CalibrationConfig { min_temp: 1.46, max_temp: 99.0, ..cfg };
    let fit_lo_bound = TemperatureScaler::fit_binary(&[1.0], &[true], &bound_cfg).unwrap();
    assert!(fit_lo_bound.converged, "min_temp=1.46 active bound must converge: {fit_lo_bound:?}");
    assert_eq!(fit_lo_bound.scaler.temperature(), 1.46);
    let fit_hi_bound = TemperatureScaler::fit_binary(&[1.0], &[false], &bound_cfg).unwrap();
    assert!(fit_hi_bound.converged, "max_temp=99.0 active bound must converge: {fit_hi_bound:?}");
    assert_eq!(fit_hi_bound.scaler.temperature(), 99.0);

    // 3. All-zero logits have flat objective -> stays at T = 1.0.
    let zero_logits = [0.0f32; 6];
    let zero_labels = [true, false, true, false, true, false];
    let fit_zero = TemperatureScaler::fit_binary(&zero_logits, &zero_labels, &cfg).unwrap();
    assert!(fit_zero.converged);
    assert_eq!(fit_zero.scaler.temperature(), 1.0);

    // 4. Huge logits (±1000) stay finite without overflow.
    let huge_logits = [-1000.0f32, -500.0, 500.0, 1000.0, 1.0, -1.0];
    let huge_labels = [false, false, true, true, false, true];
    let fit_huge = TemperatureScaler::fit_binary(&huge_logits, &huge_labels, &cfg).unwrap();
    assert!(fit_huge.final_nll.is_finite());
    assert!(fit_huge.scaler.temperature().is_finite() && fit_huge.scaler.temperature() > 0.0);

    // 5. calibrate_prob matches calibrate_binary on logit(p).
    let scaler = TemperatureScaler::new(2.25).unwrap();
    for &z in &[-4.0f32, -1.5, 0.0, 1.2, 3.8] {
        let p_raw = sigmoid_f64(z as f64) as f32;
        let via_logit = scaler.calibrate_binary(z);
        let via_prob = scaler.calibrate_prob(p_raw);
        assert!((via_logit - via_prob).abs() < 1e-6, "z={z}: {via_logit} vs {via_prob}");
    }
    assert_eq!(scaler.calibrate_prob(0.0), 0.0);
    assert_eq!(scaler.calibrate_prob(1.0), 1.0);
    assert!(scaler.calibrate_prob(f32::NAN).is_nan());

    // 6. Serde round-trip AND invalid payload rejection.
    let json = serde_json::to_string(&scaler).unwrap();
    assert_eq!(serde_json::from_str::<TemperatureScaler>(&json).unwrap(), scaler);
    assert!(serde_json::from_str::<TemperatureScaler>(r#"{"temperature": 0.0}"#).is_err());
    assert!(serde_json::from_str::<TemperatureScaler>(r#"{"temperature": -1.5}"#).is_err());
}

#[test]
fn hand_computed_ece_mce_brier_bin_edges_and_validation() {
    // 4 samples across 2 bins ([0, 0.5), [0.5, 1.0]):
    let probs = [0.0f32, 0.2, 0.8, 1.0];
    let labels = [false, false, true, false];
    let m = calibration_metrics(&probs, &labels, 2).unwrap();
    assert_eq!(m.count, 4);
    assert_eq!(m.bins.len(), 2);
    assert_eq!(m.bins[0].count, 2);
    assert_eq!(m.bins[1].count, 2);
    assert!((m.ece - 0.25).abs() < 1e-6, "ece = {}", m.ece);
    assert!((m.mce - 0.40).abs() < 1e-6, "mce = {}", m.mce);
    assert!((m.brier - 0.27).abs() < 1e-6, "brier = {}", m.brier);
    assert!(m.nll.is_finite() && m.nll > 0.0);
    assert_eq!(brier_score(&probs, &labels).unwrap(), m.brier);
    assert_eq!(expected_calibration_error(&probs, &labels, 2).unwrap(), m.ece);
    assert_eq!(max_calibration_error(&probs, &labels, 2).unwrap(), m.mce);

    // Non-binary-exact bin edge agreement: with 10 bins, 0.7_f32 equals bin 7's lower edge (bin 6's
    // exclusive upper edge), so it must land in bin 7.
    let m10 = calibration_metrics(&[0.7f32, 0.3f32, 0.1f32], &[true, false, true], 10).unwrap();
    assert_eq!(m10.bins[6].count, 0, "0.7_f32 must not fall below published edge {}", m10.bins[7].lower);
    assert_eq!(
        m10.bins[7].count, 1,
        "0.7_f32 must land in bin 7 [{}, {})",
        m10.bins[7].lower, m10.bins[7].upper
    );
    assert_eq!(m10.bins[3].count, 1, "0.3_f32 must land in bin 3");
    assert_eq!(m10.bins[1].count, 1, "0.1_f32 must land in bin 1");

    // Empty input returns clean zeros.
    let empty = calibration_metrics(&[], &[], 10).unwrap();
    assert_eq!((empty.ece, empty.mce, empty.brier, empty.nll, empty.count), (0.0, 0.0, 0.0, 0.0, 0));
    assert_eq!(empty.bins.len(), 10);
    assert_eq!(brier_score(&[], &[]).unwrap(), 0.0);
    assert_eq!(multiclass_brier_score(&[], 3, &[]).unwrap(), 0.0);

    // Validation rejections.
    assert!(TemperatureScaler::new(0.0).is_err());
    assert!(TemperatureScaler::new(-1.0).is_err());
    assert!(TemperatureScaler::new(f64::NAN).is_err());
    assert!(calibration_metrics(&probs, &labels, 0).is_err());
    assert!(calibration_metrics(&[1.1], &[true], 5).is_err());
    assert!(calibration_metrics(&[-0.1], &[false], 5).is_err());
    assert!(calibration_metrics(&[f32::NAN], &[true], 5).is_err());
    assert!(calibration_metrics(&[0.5], &[true, false], 5).is_err());
    let cfg = CalibrationConfig::default();
    assert!(TemperatureScaler::fit_binary(&[], &[], &cfg).is_err());
    assert!(TemperatureScaler::fit_binary(&[f32::NAN], &[true], &cfg).is_err());
    assert!(TemperatureScaler::fit_binary_targets(&[1.0], &[1.2], &cfg).is_err());
    assert!(TemperatureScaler::fit_multiclass(&[0.0, 1.0], 1, &[0], &cfg).is_err());
    assert!(TemperatureScaler::fit_multiclass(&[0.0, 1.0], 2, &[2], &cfg).is_err());
    let bad_cfg = CalibrationConfig { min_temp: 2.0, max_temp: 1.0, ..cfg };
    assert!(TemperatureScaler::fit_binary(&[1.0], &[true], &bad_cfg).is_err());
}
