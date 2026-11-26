//! Unit tests for Online Hard Example Mining (`ohem_select`, `OhemLoss`):
//! top-k / fraction / neg-pos-ratio budgets, deterministic tie-breaking (including signed zero),
//! positive-preservation masking, central finite-difference `grad_check_with` (standalone and
//! composed over differentiable losses), zero-gradient guarantee on unselected entries, empty &
//! non-finite propagation, and config/input validation.

use folo_core::testing::{GradCheck, grad_check_with};
use folo_core::{DType, Tensor};

use super::terms::{LossReduction, OhemBudget, OhemConfig, OhemLoss, VflConfig, ohem_select, varifocal_loss};

#[test]
fn ohem_topk_fraction_neg_pos_ratio_and_tie_breaking() {
    // Ties at value 2.0 (indices 1, 3, 5) and signed zeros -0.0 / +0.0 (indices 0, 4).
    let losses = Tensor::from_slice(&[-0.0f32, 2.0, 5.0, 2.0, 0.0, 2.0, 0.5, 4.0], [2, 4]).unwrap();

    // 1. TopK(3): picks 5.0 (idx 2), 4.0 (idx 7), and lowest-index 2.0 (idx 1).
    let cfg = OhemConfig { budget: OhemBudget::TopK(3), min_kept: 0, reduction: LossReduction::Sum };
    let sel = ohem_select(&losses, None, &cfg).unwrap();
    assert_eq!(sel.selected_indices, vec![1, 2, 7]);
    assert_eq!(sel.num_positives, 0);
    assert_eq!(sel.num_mined, 3);
    assert!((sel.loss.item::<f32>().unwrap() - 11.0).abs() < 1e-6);

    // TopK(4): adds next lowest-index 2.0 (idx 3).
    let cfg4 = OhemConfig { budget: OhemBudget::TopK(4), min_kept: 0, reduction: LossReduction::Mean };
    let sel4 = ohem_select(&losses, None, &cfg4).unwrap();
    assert_eq!(sel4.selected_indices, vec![1, 2, 3, 7]);
    assert!((sel4.loss.item::<f32>().unwrap() - 13.0 / 4.0).abs() < 1e-6);

    // 2. Signed-zero tie-break: all zeros -> picks lowest flat indices first.
    let zeros = Tensor::from_slice(&[0.0f32, -0.0, 0.0, -0.0], [4]).unwrap();
    let sel_z = ohem_select(
        &zeros,
        None,
        &OhemConfig { budget: OhemBudget::TopK(2), min_kept: 0, reduction: LossReduction::Sum },
    )
    .unwrap();
    assert_eq!(sel_z.selected_indices, vec![0, 1]);

    // 3. Fraction(0.25) over 8 elements -> ceil(2.0) = 2 hardest (idx 2, 7).
    let cfg_f = OhemConfig { budget: OhemBudget::Fraction(0.25), min_kept: 0, reduction: LossReduction::Sum };
    let sel_f = ohem_select(&losses, None, &cfg_f).unwrap();
    assert_eq!(sel_f.selected_indices, vec![2, 7]);
    assert_eq!(sel_f.num_mined, 2);

    // Fraction(0.0) with min_kept = 0 keeps 0; with min_kept = 2 keeps 2.
    let sel_zero = ohem_select(
        &losses,
        None,
        &OhemConfig { budget: OhemBudget::Fraction(0.0), min_kept: 0, reduction: LossReduction::Mean },
    )
    .unwrap();
    assert!(sel_zero.selected_indices.is_empty());
    assert_eq!(sel_zero.loss.item::<f32>().unwrap(), 0.0);

    let sel_min = ohem_select(
        &losses,
        None,
        &OhemConfig { budget: OhemBudget::Fraction(0.0), min_kept: 2, reduction: LossReduction::Sum },
    )
    .unwrap();
    assert_eq!(sel_min.selected_indices, vec![2, 7]);

    // 4. Positive-preservation mask: indices 0 (loss -0.0) and 6 (loss 0.5) are positive -> always kept.
    // NegPosRatio { ratio: 1.5, min_neg: 1 } with 2 positives -> ceil(3.0) = 3 hardest negatives:
    // negative pool indices {1:2.0, 2:5.0, 3:2.0, 4:0.0, 5:2.0, 7:4.0} -> selects {2, 7, 1}.
    let pos_mask = [true, false, false, false, false, false, true, false];
    let cfg_np = OhemConfig {
        budget: OhemBudget::NegPosRatio { ratio: 1.5, min_neg: 1 },
        min_kept: 0,
        reduction: LossReduction::None,
    };
    let ohem = OhemLoss::new(cfg_np).unwrap();
    let sel_np = ohem.select(&losses, Some(&pos_mask)).unwrap();
    assert_eq!(sel_np.num_positives, 2);
    assert_eq!(sel_np.num_mined, 3);
    assert_eq!(sel_np.selected_indices, vec![0, 1, 2, 6, 7]);
    let masked = sel_np.loss.to_vec::<f32>().unwrap();
    assert_eq!(masked, vec![-0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.5, 4.0]);
}

#[test]
fn ohem_grad_check_and_exact_gradient_routing() {
    // Values separated by > 2*eps across the mining cut so the selected set is locally constant.
    let raw = Tensor::from_slice(&[0.2f32, 3.4, 1.1, 4.8, 0.5, 2.3], [2, 3]).unwrap();
    let pos_mask = [true, false, false, false, false, false];
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 6, seed: 7 };

    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let cfg = OhemConfig { budget: OhemBudget::TopK(2), min_kept: 0, reduction };
        // Without positive mask (mines indices 3 and 1).
        grad_check_with(
            |ins| {
                let out = OhemLoss::new(cfg)?.forward(&ins[0], None)?;
                if reduction == LossReduction::None { out.sum_all() } else { Ok(out) }
            },
            std::slice::from_ref(&raw),
            &gc,
        )
        .unwrap();

        // With positive mask (preserves index 0 + mines indices 3 and 1).
        grad_check_with(
            |ins| {
                let out = OhemLoss::new(cfg)?.forward(&ins[0], Some(&pos_mask))?;
                if reduction == LossReduction::None { out.sum_all() } else { Ok(out) }
            },
            std::slice::from_ref(&raw),
            &gc,
        )
        .unwrap();
    }

    // Verify unselected elements receive strictly 0.0 gradient even with non-uniform upstream weights.
    let x = raw.detach().requires_grad_();
    let cfg_none = OhemConfig { budget: OhemBudget::TopK(2), min_kept: 0, reduction: LossReduction::None };
    let sel = ohem_select(&x, Some(&pos_mask), &cfg_none).unwrap();
    assert_eq!(sel.selected_indices, vec![0, 1, 3]);
    let weights = Tensor::from_slice(&[2.0f32, 3.0, 5.0, 7.0, 11.0, 13.0], [2, 3]).unwrap();
    let weighted = sel.loss.mul(&weights).unwrap().sum_all().unwrap();
    let grads = weighted.backward().unwrap();
    let gx = grads.get(&x).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(gx, vec![2.0, 3.0, 0.0, 7.0, 0.0, 0.0]);

    // Composed end-to-end with varifocal_loss passes grad_check_with.
    let logits = Tensor::from_slice(&[-1.5f32, 0.8, -0.2, 2.1, -2.5, 0.3], [2, 3]).unwrap();
    let targets = Tensor::from_slice(&[0.8f32, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 3]).unwrap();
    let vfl_none = VflConfig { alpha: 0.75, gamma: 2.0, reduction: LossReduction::None };
    let ohem_cfg =
        OhemConfig { budget: OhemBudget::Fraction(0.4), min_kept: 0, reduction: LossReduction::Mean };
    grad_check_with(
        |ins| {
            let per_elem = varifocal_loss(&ins[0], &targets, &vfl_none)?;
            ohem_select(&per_elem, Some(&pos_mask), &ohem_cfg).map(|s| s.loss)
        },
        std::slice::from_ref(&logits),
        &gc,
    )
    .unwrap();
}

#[test]
fn ohem_empty_non_finite_serde_and_validation() {
    // Empty tensor returns 0.0 and zero gradient cleanly.
    let empty = Tensor::zeros_with([0, 3], DType::F32, folo_core::Device::Cpu).unwrap().requires_grad_();
    let cfg = OhemConfig::default();
    let sel = ohem_select(&empty, Some(&[]), &cfg).unwrap();
    assert!(sel.selected_indices.is_empty());
    assert_eq!(sel.loss.item::<f32>().unwrap(), 0.0);
    let grads = sel.loss.backward().unwrap();
    assert_eq!(grads.get(&empty).unwrap().numel(), 0);

    // Non-finite loss (NaN) sorts first and propagates NaN rather than being hidden.
    let with_nan = Tensor::from_slice(&[1.0f32, f32::NAN, 2.0, 3.0], [4]).unwrap();
    let sel_nan = ohem_select(
        &with_nan,
        None,
        &OhemConfig { budget: OhemBudget::TopK(1), min_kept: 0, reduction: LossReduction::Mean },
    )
    .unwrap();
    assert_eq!(sel_nan.selected_indices, vec![1]);
    assert!(sel_nan.loss.item::<f32>().unwrap().is_nan());

    // Serde round-trip.
    let ohem = OhemLoss::new(OhemConfig {
        budget: OhemBudget::NegPosRatio { ratio: 3.0, min_neg: 4 },
        min_kept: 2,
        reduction: LossReduction::Sum,
    })
    .unwrap();
    let json = serde_json::to_string(&ohem).unwrap();
    assert_eq!(serde_json::from_str::<OhemLoss>(&json).unwrap(), ohem);

    // Invalid configs and mismatched mask rejected.
    for bad in [
        OhemConfig { budget: OhemBudget::Fraction(-0.1), ..Default::default() },
        OhemConfig { budget: OhemBudget::Fraction(1.1), ..Default::default() },
        OhemConfig { budget: OhemBudget::Fraction(f32::NAN), ..Default::default() },
        OhemConfig { budget: OhemBudget::NegPosRatio { ratio: -1.0, min_neg: 0 }, ..Default::default() },
        OhemConfig {
            budget: OhemBudget::NegPosRatio { ratio: f32::INFINITY, min_neg: 0 },
            ..Default::default()
        },
    ] {
        assert!(OhemLoss::new(bad).is_err());
    }
    let valid = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3]).unwrap();
    assert!(ohem_select(&valid, Some(&[true, false]), &cfg).is_err());
    assert!(ohem_select(&empty, Some(&[true]), &cfg).is_err());
}
