//! Unit tests for differentiable COCO Object Keypoint Similarity (`oks_similarity`, `oks_loss`):
//! `f64` central finite-difference verification, tensor `grad_check_with` for both `pred_kpts` and
//! `gt_kpts` across all reductions, visibility `0/1/2` masking, scale invariance, cancellation-free
//! small-offset accuracy, unlabeled-instance handling, and error validation.

use folo_core::testing::{GradCheck, grad_check_with};
use folo_core::{DType, Tensor};

use super::terms::{LossReduction, OksLossConfig, oks_instance_f64, oks_loss, oks_similarity};
use crate::metrics::COCO_KEYPOINT_SIGMAS;

#[test]
fn oks_f64_central_fd_and_properties() {
    let pred = [[10.4, 20.8], [32.1, 41.5], [999.0, -500.0], [50.0, 50.0]];
    let gt = [[10.0, 20.0], [30.0, 40.0], [0.0, 0.0], [50.0, 50.0]];
    // v=2 visible, v=1 occluded, v=0 unlabeled (must have 0 gradient & no effect), v=2 exact match.
    let vis = [2.0, 1.0, 0.0, 2.0];
    let sigmas = [0.026, 0.079, 0.107, 0.062];
    let area = 2500.0;
    let eps = 1e-9;

    for return_sim in [false, true] {
        let (val, n_vis, grad) = oks_instance_f64(&pred, &gt, &vis, area, &sigmas, eps, return_sim);
        assert_eq!(n_vis, 3);
        assert!(val.is_finite() && (0.0..=1.0).contains(&val));
        // Unlabeled keypoint 2 must have strictly zero gradient.
        assert_eq!(grad[2], [0.0, 0.0]);
        // Exact match keypoint 3 must have strictly zero gradient.
        assert_eq!(grad[3], [0.0, 0.0]);

        let h = 1e-6;
        for k in 0..pred.len() {
            for axis in 0..2 {
                let mut pp = pred;
                let mut pm = pred;
                pp[k][axis] += h;
                pm[k][axis] -= h;
                let vp = oks_instance_f64(&pp, &gt, &vis, area, &sigmas, eps, return_sim).0;
                let vm = oks_instance_f64(&pm, &gt, &vis, area, &sigmas, eps, return_sim).0;
                let fd = (vp - vm) / (2.0 * h);
                let tol = 1e-9 + 1e-6 * fd.abs().max(grad[k][axis].abs());
                assert!(
                    (grad[k][axis] - fd).abs() <= tol,
                    "sim={return_sim} k={k} axis={axis}: ana={} fd={fd}",
                    grad[k][axis]
                );
            }
        }
    }

    // Similarity + loss == 1.0 exactly to machine precision.
    let (loss, _, _) = oks_instance_f64(&pred, &gt, &vis, area, &sigmas, eps, false);
    let (sim, _, _) = oks_instance_f64(&pred, &gt, &vis, area, &sigmas, eps, true);
    assert!(((loss + sim) - 1.0).abs() < 1e-15);

    // Scale invariance: scaling coordinates by s and area by s^2 keeps OKS identical (for eps = 0).
    let scale = 7.5;
    let pred_s = pred.map(|[x, y]| [x * scale, y * scale]);
    let gt_s = gt.map(|[x, y]| [x * scale, y * scale]);
    let (l1, _, _) = oks_instance_f64(&pred, &gt, &vis, area, &sigmas, 0.0, false);
    let (l2, _, _) = oks_instance_f64(&pred_s, &gt_s, &vis, area * scale * scale, &sigmas, 0.0, false);
    assert!((l1 - l2).abs() < 1e-14);
}

#[test]
fn oks_loss_tensor_grad_check_pred_and_gt() {
    // 2 instances, 4 keypoints (instance 0 has 3 labeled incl. occluded; instance 1 has 2 labeled).
    let pred = Tensor::from_slice(
        &[
            11.0f32, 19.2, 28.5, 42.0, 100.0, 200.0, 52.0, 49.0, // inst 0
            5.5, 6.2, 14.0, 18.0, 22.0, 25.0, 31.0, 29.0, // inst 1
        ],
        [2, 4, 2],
    )
    .unwrap();
    let gt = Tensor::from_slice(
        &[
            10.0f32, 20.0, 30.0, 40.0, 0.0, 0.0, 50.0, 50.0, // inst 0
            5.0, 6.0, 15.0, 15.0, 20.0, 20.0, 30.0, 30.0, // inst 1
        ],
        [2, 4, 2],
    )
    .unwrap();
    let vis = Tensor::from_slice(&[2.0f32, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 0.0], [2, 4]).unwrap();
    let areas = Tensor::from_slice(&[900.0f32, 400.0], [2]).unwrap();
    let sigmas = &COCO_KEYPOINT_SIGMAS[..4];

    let gc = GradCheck { eps: 5e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 29 };

    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let cfg = OksLossConfig { eps: 1e-9, reduction };
        grad_check_with(
            |ins| {
                let out = oks_loss(&ins[0], &ins[1], &vis, &areas, sigmas, &cfg)?;
                if reduction == LossReduction::None { out.sum_all() } else { Ok(out) }
            },
            &[pred.clone(), gt.clone()],
            &gc,
        )
        .unwrap();
    }

    // Differentiable oks_similarity passes grad_check_with for both pred and gt.
    grad_check_with(
        |ins| oks_similarity(&ins[0], &ins[1], &vis, &areas, sigmas, 1e-9)?.sum_all(),
        &[pred, gt],
        &gc,
    )
    .unwrap();
}

#[test]
fn oks_loss_unlabeled_instances_exact_match_and_validation() {
    let sigmas = [0.05f32, 0.1];
    // Instance 0: exact match (loss 0, sim 1). Instance 1: all unlabeled (loss 0, excluded from Mean).
    // Instance 2: known offset dx=10, dy=0 on kpt 0 (v=2), kpt 1 unlabeled, area=10000, sigma=0.05
    // -> e = 100 / (8 * 10000 * 0.0025) = 100 / 200 = 0.5 -> loss = 1 - exp(-0.5).
    let pred =
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 99.0, 88.0, 77.0, 66.0, 10.0, 0.0, 5.0, 5.0], [3, 2, 2])
            .unwrap();
    let gt = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3, 2, 2])
        .unwrap();
    let vis = Tensor::from_slice(&[2.0f32, 1.0, 0.0, 0.0, 2.0, 0.0], [3, 2]).unwrap();
    let areas = Tensor::from_slice(&[100.0f32, 100.0, 10000.0], [3]).unwrap();

    let per = oks_loss(
        &pred,
        &gt,
        &vis,
        &areas,
        &sigmas,
        &OksLossConfig { eps: 0.0, reduction: LossReduction::None },
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    let want2 = 1.0 - (-0.5f64).exp();
    assert_eq!(per[0], 0.0);
    assert_eq!(per[1], 0.0);
    assert!((per[2] as f64 - want2).abs() < 1e-6);

    // Mean divides by 2 active instances (inst 0 and inst 2), not 3!
    let mean = oks_loss(
        &pred,
        &gt,
        &vis,
        &areas,
        &sigmas,
        &OksLossConfig { eps: 0.0, reduction: LossReduction::Mean },
    )
    .unwrap()
    .item::<f32>()
    .unwrap();
    assert!((mean as f64 - 0.5 * want2).abs() < 1e-6);

    // Empty batch [0, 2, 2] stays finite and differentiable.
    let empty_p = Tensor::zeros_with([0, 2, 2], DType::F32, folo_core::Device::Cpu).unwrap().requires_grad_();
    let empty_g = Tensor::zeros_with([0, 2, 2], DType::F32, folo_core::Device::Cpu).unwrap();
    let empty_v = Tensor::zeros_with([0, 2], DType::F32, folo_core::Device::Cpu).unwrap();
    let empty_a = Tensor::zeros_with([0], DType::F32, folo_core::Device::Cpu).unwrap();
    let l_empty =
        oks_loss(&empty_p, &empty_g, &empty_v, &empty_a, &sigmas, &OksLossConfig::default()).unwrap();
    assert_eq!(l_empty.item::<f32>().unwrap(), 0.0);
    assert_eq!(l_empty.backward().unwrap().get(&empty_p).unwrap().numel(), 0);

    // Masked (v == 0) keypoints MUST receive strictly 0.0 gradient even when area == 0 (eps == 0),
    // area is NaN, or another labeled keypoint is NaN.
    let vis_mask = Tensor::from_slice(&[2.0f32, 0.0], [1, 2]).unwrap();
    let gt1 = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0], [1, 2, 2]).unwrap();
    let p_zero_area = Tensor::from_slice(&[3.0f32, 4.0, 12.0, 15.0], [1, 2, 2]).unwrap().requires_grad_();
    let a_zero = Tensor::from_slice(&[0.0f32], [1]).unwrap();
    let l_zero = oks_loss(
        &p_zero_area,
        &gt1,
        &vis_mask,
        &a_zero,
        &sigmas,
        &OksLossConfig { eps: 0.0, reduction: LossReduction::Mean },
    )
    .unwrap();
    assert_eq!(l_zero.item::<f32>().unwrap(), 0.0);
    let g_zero = l_zero.backward().unwrap().get(&p_zero_area).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(g_zero, vec![0.0, 0.0, 0.0, 0.0]);

    let p_nan = Tensor::from_slice(&[f32::NAN, 4.0, 12.0, 15.0], [1, 2, 2]).unwrap().requires_grad_();
    let a_ok = Tensor::from_slice(&[100.0f32], [1]).unwrap();
    let l_nan = oks_loss(&p_nan, &gt1, &vis_mask, &a_ok, &sigmas, &OksLossConfig::default()).unwrap();
    assert!(l_nan.item::<f32>().unwrap().is_nan());
    let g_nan = l_nan.backward().unwrap().get(&p_nan).unwrap().to_vec::<f32>().unwrap();
    assert!(g_nan[0].is_nan() && g_nan[1].is_nan());
    assert_eq!((g_nan[2], g_nan[3]), (0.0, 0.0), "unlabeled keypoint must stay strictly 0.0");

    // Empty k == 0 with n > 0 still validates negative areas.
    let p_k0 = Tensor::zeros_with([1, 0, 2], DType::F32, folo_core::Device::Cpu).unwrap();
    let v_k0 = Tensor::zeros_with([1, 0], DType::F32, folo_core::Device::Cpu).unwrap();
    let a_neg1 = Tensor::from_slice(&[-5.0f32], [1]).unwrap();
    assert!(oks_loss(&p_k0, &p_k0, &v_k0, &a_neg1, &[], &OksLossConfig::default()).is_err());

    // Validation errors.
    let bad_vis = Tensor::from_slice(&[3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], [3, 2]).unwrap();
    assert!(oks_loss(&pred, &gt, &bad_vis, &areas, &sigmas, &OksLossConfig::default()).is_err());
    let bad_area = Tensor::from_slice(&[-1.0f32, 1.0, 1.0], [3]).unwrap();
    assert!(oks_loss(&pred, &gt, &vis, &bad_area, &sigmas, &OksLossConfig::default()).is_err());
    assert!(oks_loss(&pred, &gt, &vis, &areas, &[0.0, 0.1], &OksLossConfig::default()).is_err());
    assert!(oks_loss(&pred, &gt, &vis, &areas, &[0.05], &OksLossConfig::default()).is_err());
    assert!(
        oks_loss(&pred, &gt, &vis, &areas, &sigmas, &OksLossConfig { eps: -1e-6, ..Default::default() })
            .is_err()
    );
}
