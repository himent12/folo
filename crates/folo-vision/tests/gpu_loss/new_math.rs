//! CPU-vs-GPU parity and numerical edge-case regressions for `R-PIoU` and `OT-DFL`.

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_vision::boxes::Xyxy;
use folo_vision::decode::{Anchors, make_anchors};
use folo_vision::loss::{
    BoxLossKind, ImageTargets, LossConfig, assign_batch, assign_batch_device, detection_loss,
    detection_loss_host,
};

use crate::{assert_same_assignment, case, close_vec, gpu};

/// CPU-vs-GPU parity for `R-PIoU`, `OT-DFL`, `R-PIoU + OT-DFL`, `R-PIoU + NWD`, and `reg_max = 2`.
#[test]
fn new_math_formulas_match_host() {
    let dev = gpu!();
    let rpiou = LossConfig { box_loss: BoxLossKind::Rpiou, ..LossConfig::default() };
    let otdfl = LossConfig { dfl_ot_weight: 0.25, ..LossConfig::default() };
    let box_combo = LossConfig { dfl_ot_weight: 0.25, ..rpiou };
    let rpiou_nwd = LossConfig { nwd_weight: 0.5, ..rpiou };

    for (name, seed, cfg) in [
        ("rpiou", 61, rpiou),
        ("otdfl", 62, otdfl),
        ("rpiou+otdfl", 63, box_combo),
        ("rpiou+nwd", 64, rpiou_nwd),
    ] {
        let (anchors, box_raw, cls_raw, targets) = case(seed, 128, 5);
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want_asg = assign_batch(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got_asg = assign_batch_device(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert_same_assignment(&got_asg, &want_asg);
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite(), "{name}: non-finite");
        assert_eq!(got.num_fg, want.num_fg, "{name}: num_fg");
        let rel = |g: f32, w: f32| (g - w).abs() <= 2e-4 * w.abs().max(1e-3);
        let (gi, wi) = (got.items, want.items);
        assert!(
            rel(gi.box_, wi.box_) && rel(gi.cls, wi.cls) && rel(gi.dfl, wi.dfl),
            "{name}: {gi:?} vs {wi:?}"
        );
        assert!(
            rel(got.total_value, want.total_value),
            "{name}: {} vs {}",
            got.total_value,
            want.total_value
        );
        let (gg, gw) = (got.total.backward().unwrap(), want.total.backward().unwrap());
        for (tensor_name, d, h) in [("box_raw", &bd, &bh), ("cls_raw", &cd, &ch)] {
            let g = gg.get(d).unwrap().to_vec::<f32>().unwrap();
            let w = gw.get(h).unwrap().to_vec::<f32>().unwrap();
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            close_vec(&g, &w, 2e-4 * scale.max(1e-4), 2e-3, &format!("{name}/{tensor_name}"));
        }
    }

    // `reg_max = 2` (OT-DFL vanishes identically; R-PIoU + DFL still match CPU).
    {
        let anchors = make_anchors(64, 64, &[8, 16], 0.5);
        let mut rng = Rng::seed(65);
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            dfl_ot_weight: 0.5,
            ..LossConfig::default()
        };
        let box_raw = Tensor::randn([1, 8, anchors.len()], 0.0, 1.0, &mut rng);
        let cls_raw = Tensor::rand_uniform([1, 3, anchors.len()], -4.0, 1.0, &mut rng);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 38.0, 44.0)], labels: vec![1] }];
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0);
        assert!((got.total_value - want.total_value).abs() <= 2e-4 * want.total_value.abs().max(1e-3));
        let (gg, gw) = (got.total.backward().unwrap(), want.total.backward().unwrap());
        let g = gg.get(&bd).unwrap().to_vec::<f32>().unwrap();
        let w = gw.get(&bh).unwrap().to_vec::<f32>().unwrap();
        let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&g, &w, 2e-4 * scale.max(1e-4), 2e-3, "reg_max=2/box_raw");
    }
}

/// Edge-case CPU-vs-GPU regressions covering extreme logits, tiny excluded masses, extreme `kappa`/`delta0`,
/// and extreme `box_gain`/`dfl_ot_weight` across single- and multi-image batches.
#[test]
fn new_math_extreme_edge_cases_match_host() {
    let dev = gpu!();

    // 1. Extreme OT-DFL logits on active foreground anchor (`R = 4`, `j = 0`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let mut box_data = vec![0.0f32; 16 * a];
        for side in 0..4 {
            for k in 0..3 {
                for j in 0..a {
                    box_data[(side * 4 + k) * a + j] = -1e8;
                }
            }
        }
        let box_raw = Tensor::from_slice(&box_data, [1, 16, a]).unwrap();
        let cls_raw = Tensor::from_slice(&vec![0.0f32; a], [1, 1, a]).unwrap();
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(26.0, 26.0, 30.0, 30.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.0,
            dfl_ot_weight: 1.0,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "extreme OT-DFL must be finite with fg > 0");
        assert!(
            (got.total_value - want.total_value).abs() <= 1e-4 * want.total_value.abs().max(1.0),
            "extreme OT-DFL loss: {} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_gpu.iter().all(|v| v.is_finite()), "extreme OT-DFL GPU grad must be finite");
        assert!(g_gpu.iter().any(|v| v.abs() > 0.0), "extreme OT-DFL GPU grad must be non-zero");
        close_vec(&g_gpu, &g_cpu, 1e-4, 1e-4, "extreme_otdfl/box_raw");
        for j in 0..a {
            for side in 0..4 {
                let s: f32 = (0..4).map(|k| g_gpu[(side * 4 + k) * a + j]).sum();
                assert!(s.abs() < 1e-4, "anchor {j} side {side}: non-zero logit grad sum {s}");
            }
        }
    }

    // 2. Saturated `R-PIoU` (`rpiou_kappa = 3e38`, `box_gain = 7.5`).
    {
        let anchors = make_anchors(64, 64, &[8, 16], 0.5);
        let mut rng = Rng::seed(71);
        let cfg = LossConfig {
            box_loss: BoxLossKind::Rpiou,
            rpiou_kappa: 3e38,
            box_gain: 7.5,
            ..LossConfig::default()
        };
        let box_raw = Tensor::randn([1, 64, anchors.len()], 0.0, 1.0, &mut rng);
        let cls_raw = Tensor::rand_uniform([1, 2, anchors.len()], -3.0, 1.0, &mut rng);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 46.0, 50.0)], labels: vec![0] }];
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "saturated R-PIoU must be finite");
        assert!(
            (got.total_value - want.total_value).abs() <= 2e-4 * want.total_value.abs().max(1e-3),
            "saturated R-PIoU loss: {} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_gpu.iter().all(|v| v.is_finite()), "saturated R-PIoU GPU grad must be finite");
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&g_gpu, &g_cpu, 2e-4 * scale.max(1e-4), 2e-3, "saturated_rpiou/box_raw");
    }

    // 3. Exact-match `R-PIoU` (`reg_max = 2`, equal logits, `rpiou_kappa = 3e38`, `box_gain = 7.5`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let box_raw = Tensor::zeros([1, 8, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_kappa: 3e38,
            box_gain: 7.5,
            dfl_gain: 0.0,
            cls_gain: 0.0,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "exact-match R-PIoU must be finite");
        assert!(
            (got.total_value - want.total_value).abs() <= 1e-5,
            "{} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()), "exact-match R-PIoU CPU grad must be finite");
        assert!(g_gpu.iter().all(|v| v.is_finite()), "exact-match R-PIoU GPU grad must be finite");
        close_vec(&g_gpu, &g_cpu, 1e-5, 1e-4, "exact_match_rpiou/box_raw");
    }

    // 4. Extreme `OT-DFL` weight (`reg_max = 3`, uniform logits, `dfl_ot_weight = 3e38`, `dfl_gain = 1.5`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let box_raw = Tensor::zeros([1, 12, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 3,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.5,
            dfl_ot_weight: 3e38,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "huge-weight OT-DFL must be finite");
        assert!(
            (got.total_value - want.total_value).abs() <= 2e-4 * want.total_value.abs().max(1.0),
            "huge-weight OT-DFL loss: {} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()), "huge-weight OT-DFL CPU grad must be finite");
        assert!(g_gpu.iter().all(|v| v.is_finite()), "huge-weight OT-DFL GPU grad must be finite");
        assert!(g_gpu.iter().any(|v| v.abs() > 0.0), "huge-weight OT-DFL GPU grad must be non-zero");
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&g_gpu, &g_cpu, 2e-4 * scale.max(1.0), 2e-3, "huge_weight_otdfl/box_raw");
    }

    // 5. Multi-positive extreme `OT-DFL` (`reg_max = 3`, uniform logits, 3 exact-match positives, `3e38`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let box_raw = Tensor::zeros([1, 12, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        let targets = vec![ImageTargets {
            boxes: vec![
                Xyxy::new(4.0, 4.0, 20.0, 20.0),
                Xyxy::new(36.0, 4.0, 52.0, 20.0),
                Xyxy::new(4.0, 36.0, 20.0, 52.0),
            ],
            labels: vec![0, 0, 0],
        }];
        let cfg = LossConfig {
            reg_max: 3,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.5,
            dfl_ot_weight: 3e38,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let ref_loss = (1.5f64 * 3e38f64 * 1.5f64.ln()) as f32;
        assert!(got.is_finite() && want.is_finite(), "3-pos extreme OT-DFL must be finite");
        assert_eq!((got.num_fg, want.num_fg), (3, 3));
        assert!(
            (want.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "CPU 3-pos OT-DFL {} vs f64 ref {ref_loss}",
            want.total_value
        );
        assert!(
            (got.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "GPU 3-pos OT-DFL {} vs f64 ref {ref_loss}",
            got.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()) && g_gpu.iter().all(|v| v.is_finite()));
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e35, "expected large finite OT-DFL gradient, got {scale}");
        close_vec(&g_gpu, &g_cpu, 2e-4 * scale, 2e-3, "three_pos_otdfl/box_raw");
    }

    // 6. Extreme `R-PIoU` floor (`rpiou_delta = 1e20`, `rpiou_kappa = 3e38`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let box_raw = Tensor::zeros([1, 12, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 3,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 3e38,
            rpiou_delta: 1e20,
            box_gain: 1.0,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let ke2_ref = 3e38f64 * 0.25f64 * (4.0 * 0.25f64 / (1.0f64 + 1e40f64));
        let edge_ref = ke2_ref / (1.0 + ke2_ref);
        let focal_ref = (1.0 - 0.25f64) * (1.0 + 0.5 * 0.25f64);
        let t_w = want.target_score_sum as f64;
        let ref_loss = (t_w * (focal_ref + edge_ref)) as f32;
        assert_eq!((got.num_fg, want.num_fg), (1, 1));
        assert!((t_w - 0.25).abs() < 1e-5, "expected target_score_sum 0.25, got {t_w}");
        assert!(
            (want.total_value - ref_loss).abs() <= 1e-5,
            "CPU scaled R-PIoU {} vs f64 ref {ref_loss}",
            want.total_value
        );
        assert!(
            (got.total_value - ref_loss).abs() <= 1e-5,
            "GPU scaled R-PIoU {} vs f64 ref {ref_loss}",
            got.total_value
        );
        let cfg_no_edge = LossConfig { rpiou_kappa: 1e-30, ..cfg };
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        let bh0 = Tensor::zeros([1, 12, a]).requires_grad_();
        let ch0 = Tensor::zeros([1, 1, a]).requires_grad_();
        let g_no_edge = detection_loss_host(&bh0, &ch0, &anchors, &targets, &cfg_no_edge)
            .unwrap()
            .total
            .backward()
            .unwrap()
            .get(&bh0)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let fg = 3 * 8 + 3;
        let edge_d = (t_w * 0.5f64 * 3e38 * 0.5 / ((1.0 + 1e40) * (1.0 + ke2_ref).powi(2)) / 3.0) as f32;
        let got_edge_d = g_gpu[2 * a + fg] - g_no_edge[2 * a + fg];
        assert!(
            (got_edge_d - edge_d).abs() <= 1e-5,
            "GPU scaled R-PIoU edge grad delta {got_edge_d} vs analytical {edge_d}"
        );
        close_vec(&g_gpu, &g_cpu, 1e-5, 1e-4, "scaled_rpiou_delta1e20/box_raw");
    }

    // 7. Extreme `R-PIoU` ratio (`gt = [0, 0, 2.5e-20, 1]`, `pred = [1.25e-38, 0, 2.5e-20, 1]`, `kappa = 1e38`).
    {
        let b1 = Tensor::from_slice(&[1.25e-38f32, 0.0, 2.5e-20, 1.0], [1, 4]).unwrap().requires_grad_();
        let b2 = Tensor::from_slice(&[0.0f32, 0.0, 2.5e-20, 1.0], [1, 4]).unwrap();
        let l = folo_vision::loss::rpiou_loss(&b1, &b2, 0.5, 1e38, 1e-30).unwrap();
        let lv = l.item::<f32>().unwrap();
        let g = l.backward().unwrap().get(&b1).unwrap().to_vec::<f32>().unwrap();
        let ke2_ref = 1e38f64 * 0.25 * ((1.25e-38f64).powi(2) / ((2.5e-20f64).powi(2) + (1e-30f64).powi(2)));
        let edge_ref = (ke2_ref / (1.0 + ke2_ref)) as f32;
        let loss_ref = 1.0 + edge_ref;
        let grad_ref = (0.5f64 * 1e38 * 1.25e-38
            / (((2.5e-20f64).powi(2) + (1e-30f64).powi(2)) * (1.0 + ke2_ref).powi(2)))
            as f32;
        assert!((lv - loss_ref).abs() <= 1e-5, "CPU rpiou_loss {lv} vs f64 ref {loss_ref}");
        assert!((g[0] - grad_ref).abs() <= 1e-4 * grad_ref, "CPU rpiou grad {} vs f64 ref {grad_ref}", g[0]);

        let l_left = (1.25e-38f32).ln();
        let d_left = Tensor::from_slice(&[0.0f32, l_left], [1, 2])
            .unwrap()
            .softmax(1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[1];
        let ax = 2.0 * d_left;
        let anchors = Anchors { points: vec![[ax, 0.5]], strides: vec![1.0], ..Anchors::default() };
        let box_raw =
            Tensor::from_slice(&[0.0, l_left, 0.0, 0.0, 0.0, -1000.0, 0.0, 0.0], [1, 8, 1]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, 1]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(0.0, 0.0, ax, 1.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 1e38,
            rpiou_delta: 2.5e-20,
            box_gain: 1.0,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            tal: folo_vision::assign::TalConfig {
                nwd_weight: 1.0,
                ..folo_vision::assign::TalConfig::default()
            },
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let t_w = want.target_score_sum as f64;
        assert!(t_w > 0.9 && got.num_fg == 1 && want.num_fg == 1);
        let ke2_g = 1e38f64 * 0.25 * ((d_left as f64).powi(2) / ((ax as f64).powi(2) + (2.5e-20f64).powi(2)));
        let loss_g_ref = 1.0 + ke2_g / (1.0 + ke2_g);
        let total_ref = (t_w * loss_g_ref) as f32;
        assert!(
            (want.total_value - total_ref).abs() <= 1e-4 * total_ref,
            "CPU exponent-scaled R-PIoU {} vs ref {total_ref}",
            want.total_value
        );
        assert!(
            (got.total_value - total_ref).abs() <= 1e-3 * total_ref,
            "GPU exponent-scaled R-PIoU {} vs ref {total_ref}",
            got.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        let grad_g_ref = 0.5f64 * 1e38 * (d_left as f64)
            / (((ax as f64).powi(2) + (2.5e-20f64).powi(2)) * (1.0 + ke2_g).powi(2));
        let logit_g_ref = -(t_w * grad_g_ref * d_left as f64) as f32;
        assert!(
            (g_gpu[1] - logit_g_ref).abs() <= 2e-2 * logit_g_ref.abs(),
            "GPU exponent-scaled R-PIoU logit grad {} vs ref {logit_g_ref}",
            g_gpu[1]
        );
        close_vec(&g_gpu, &g_cpu, 2e-2 * logit_g_ref.abs(), 2e-2, "exponent_scaled_rpiou/box_raw");
    }

    // 8. Extreme `OT-DFL` logits (`reg_max = 6`, logits `[-1e38; 5, 0]` per side, `dfl_ot_weight = 0.25`):
    //    widened CPU `f64` and GPU exponent-split reduction must match the finite `f64` reference (`~2e38 * t`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let mut box_data = vec![0.0f32; 24 * a];
        for side in 0..4 {
            for k in 0..5 {
                for j in 0..a {
                    box_data[(side * 6 + k) * a + j] = -1e38;
                }
            }
        }
        let box_raw = Tensor::from_slice(&box_data, [1, 24, a]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, a]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 6,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.0,
            dfl_ot_weight: 0.25,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let t_w = want.target_score_sum as f64;
        assert!(t_w > 0.0 && got.num_fg == 1 && want.num_fg == 1);
        let ref_loss = (2e38f64 * t_w) as f32;
        assert!(got.is_finite() && want.is_finite(), "extreme -1e38 OT-DFL must be finite");
        assert!(
            (want.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "CPU -1e38 OT-DFL {} vs f64 ref {ref_loss}",
            want.total_value
        );
        assert!(
            (got.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "GPU -1e38 OT-DFL {} vs f64 ref {ref_loss}",
            got.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()) && g_gpu.iter().all(|v| v.is_finite()));
        close_vec(&g_gpu, &g_cpu, 1e-4, 1e-3, "extreme_neg1e38_otdfl/box_raw");
    }

    // 9. Equal tail and full sums in `OT-DFL` (`reg_max = 4`, logits `[-1000, 0, 0, -1000]`, `1e8`).
    {
        let anchors = Anchors { points: vec![[3.5, 3.5]], strides: vec![8.0], ..Anchors::default() };
        let a = anchors.len();
        let mut box_data = vec![0.0f32; 16 * a];
        for side in 0..4 {
            box_data[side * 4] = -1000.0;
            box_data[side * 4 + 3] = -1000.0;
        }
        let box_raw = Tensor::from_slice(&box_data, [1, 16, a]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, a]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(16.0, 16.0, 40.0, 40.0)], labels: vec![0] }];
        let cfg0 = LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.0,
            dfl_ot_weight: 0.0,
            ..LossConfig::default()
        };
        let cfg_ot = LossConfig { dfl_ot_weight: 1e8, ..cfg0 };
        let base_cpu = detection_loss_host(&box_raw, &cls_raw, &anchors, &targets, &cfg0).unwrap();
        let ot_cpu = detection_loss_host(&box_raw, &cls_raw, &anchors, &targets, &cfg_ot).unwrap();
        let (bd, cd) = (box_raw.to_device(dev).unwrap(), cls_raw.to_device(dev).unwrap());
        let ot_gpu = detection_loss(&bd, &cd, &anchors, &targets, &cfg_ot).unwrap();
        assert_eq!((ot_cpu.num_fg, ot_gpu.num_fg), (1, 1));
        assert!(
            (ot_cpu.total_value - base_cpu.total_value).abs() <= 1e-6,
            "CPU equal-sum OT-DFL added spurious loss: {} vs base {}",
            ot_cpu.total_value,
            base_cpu.total_value
        );
        assert!(
            (ot_gpu.total_value - base_cpu.total_value).abs() <= 1e-5,
            "GPU equal-sum OT-DFL {} vs base {}",
            ot_gpu.total_value,
            base_cpu.total_value
        );
    }

    // 10. Extreme `box_gain * kappa` (`d_left ≈ 1e-20`, `kappa = 1e38`, `box_gain = 1e25`).
    {
        let l_left = (1e-20f32).ln();
        let d_left = Tensor::from_slice(&[0.0f32, l_left], [1, 2])
            .unwrap()
            .softmax(1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[1];
        let ax = 2.0 * d_left;
        let anchors = Anchors { points: vec![[ax, 0.5]], strides: vec![1.0], ..Anchors::default() };
        let box_raw =
            Tensor::from_slice(&[0.0, l_left, 0.0, 0.0, 0.0, -1000.0, 0.0, 0.0], [1, 8, 1]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, 1]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(0.0, 0.0, ax, 1.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 1e38,
            rpiou_delta: 0.1,
            box_gain: 1e25,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            tal: folo_vision::assign::TalConfig {
                nwd_weight: 1.0,
                ..folo_vision::assign::TalConfig::default()
            },
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let t_w = want.target_score_sum as f64;
        assert!(got.is_finite() && want.is_finite() && t_w > 0.9);
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()), "CPU logit grad must be finite: {g_cpu:?}");
        assert!(g_gpu.iter().all(|v| v.is_finite()), "GPU logit grad must be finite: {g_gpu:?}");
        let d2 = (0.1f32 as f64).powi(2);
        let ke2 = 1e38f64 * 0.25 * (d_left as f64).powi(2) / ((ax as f64).powi(2) + d2);
        let coord_g = 0.5f64 * (t_w * 1e25f64) * 1e38f64 * (d_left as f64)
            / (((ax as f64).powi(2) + d2) * (1.0 + ke2).powi(2));
        let logit_g_ref = -(coord_g * d_left as f64) as f32;
        assert!(
            (g_cpu[1] - logit_g_ref).abs() <= 1e-3 * logit_g_ref.abs(),
            "CPU logit grad {} vs f64 ref {logit_g_ref}",
            g_cpu[1]
        );
        assert!(
            (g_gpu[1] - logit_g_ref).abs() <= 2e-2 * logit_g_ref.abs(),
            "GPU logit grad {} vs f64 ref {logit_g_ref}",
            g_gpu[1]
        );
        close_vec(&g_gpu, &g_cpu, 2e-2 * logit_g_ref.abs(), 2e-2, "extreme_box_gain_rpiou/box_raw");
    }

    // 11. Small excluded OT-DFL mass (`[-20, 0, 0, -1000]`, `dfl_ot_weight = 1e8`, `dfl_gain = 1.0`).
    {
        let anchors = Anchors { points: vec![[3.5, 3.5]], strides: vec![8.0], ..Anchors::default() };
        let mut bd_v = vec![0.0f32; 16];
        for s in 0..4 {
            bd_v[s * 4] = -20.0;
            bd_v[s * 4 + 3] = -1000.0;
        }
        let bx = Tensor::from_slice(&bd_v, [1, 16, 1]).unwrap();
        let cx = Tensor::zeros([1, 1, 1]);
        let tgts = vec![ImageTargets { boxes: vec![Xyxy::new(16.0, 16.0, 40.0, 40.0)], labels: vec![0] }];
        let cfg0 =
            LossConfig { reg_max: 4, box_gain: 0.0, cls_gain: 0.0, dfl_gain: 1.0, ..LossConfig::default() };
        let cfg1 = LossConfig { dfl_ot_weight: 1e8, ..cfg0 };
        let b0 = detection_loss_host(&bx, &cx, &anchors, &tgts, &cfg0).unwrap();
        let (bd, cd) =
            (bx.to_device(dev).unwrap().requires_grad_(), cx.to_device(dev).unwrap().requires_grad_());
        let (bh, ch) = (bx.requires_grad_(), cx.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &tgts, &cfg1).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &tgts, &cfg1).unwrap();
        let ot_ref = (1e8f64 * ((-20.0f64 - 2.0f64.ln()).exp().ln_1p())) as f32;
        assert!(
            ((want.total_value - b0.total_value) - ot_ref).abs() <= 1e-4,
            "CPU small-excl OT {} vs {ot_ref}",
            want.total_value - b0.total_value
        );
        assert!(
            ((got.total_value - b0.total_value) - ot_ref).abs() <= 1e-4,
            "GPU small-excl OT {} vs {ot_ref}",
            got.total_value - b0.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        close_vec(&g_gpu, &g_cpu, 1e-4, 1e-3, "small_excl_otdfl/box_raw");
    }

    // 12. Subnormal-range excluded OT-DFL tail (`[-89, 0, 0, -1000]`, `dfl_ot_weight = 1e38`, `dfl_gain = 1.0`):
    //     `delta = -89 - ln(2) = -89.693` is below normal `f32` `exp`, so exponent-split GPU reduction and backward
    //     must preserve the `~0.11137` OT loss and matching gradients instead of dropping to `0`.
    {
        let anchors = Anchors { points: vec![[3.5, 3.5]], strides: vec![8.0], ..Anchors::default() };
        let mut bd_v = vec![0.0f32; 16];
        for s in 0..4 {
            bd_v[s * 4] = -89.0;
            bd_v[s * 4 + 3] = -1000.0;
        }
        let bx = Tensor::from_slice(&bd_v, [1, 16, 1]).unwrap();
        let cx = Tensor::zeros([1, 1, 1]);
        let tgts = vec![ImageTargets { boxes: vec![Xyxy::new(16.0, 16.0, 40.0, 40.0)], labels: vec![0] }];
        let cfg0 =
            LossConfig { reg_max: 4, box_gain: 0.0, cls_gain: 0.0, dfl_gain: 1.0, ..LossConfig::default() };
        let cfg1 = LossConfig { dfl_ot_weight: 1e38, ..cfg0 };
        let b0 = detection_loss_host(&bx, &cx, &anchors, &tgts, &cfg0).unwrap();
        let (bd, cd) =
            (bx.to_device(dev).unwrap().requires_grad_(), cx.to_device(dev).unwrap().requires_grad_());
        let (bh, ch) = (bx.requires_grad_(), cx.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &tgts, &cfg1).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &tgts, &cfg1).unwrap();
        let ot_ref = (1e38f64 * ((-89.0f64 - 2.0f64.ln()).exp().ln_1p())) as f32; // ~0.111368
        assert!(
            ((want.total_value - b0.total_value) - ot_ref).abs() <= 1e-4,
            "CPU -89 tail OT {} vs {ot_ref}",
            want.total_value - b0.total_value
        );
        assert!(
            ((got.total_value - b0.total_value) - ot_ref).abs() <= 1e-4,
            "GPU -89 tail OT {} vs {ot_ref}",
            got.total_value - b0.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        // Bin 0 on every side gets `+0.25 * ot_ref ≈ +0.02784` gradient from OT-DFL.
        assert!(
            (g_gpu[0] - 0.25 * ot_ref).abs() <= 1e-4,
            "GPU -89 tail bin-0 grad {} vs {}",
            g_gpu[0],
            0.25 * ot_ref
        );
        close_vec(&g_gpu, &g_cpu, 1e-4, 1e-3, "tiny_tail_neg89_otdfl/box_raw");
    }

    // 13. Extreme `box_gain = 2e38` on R-PIoU focal-IoU across `B = 1`, `B = 2` (2 positives, `tss = 2`), and
    //     `B = 2` (1 positive + 1 empty image, `tss = 1`, where even `(B / tss) * box_gain = 4e38 > f32::MAX`):
    //     forward losses and per-bin logit gradients stay finite and match across CPU and GPU.
    {
        let anchors = Anchors { points: vec![[0.5, 0.5]], strides: vec![1.0], ..Anchors::default() };
        let l1 = (1.0f32 / 7.0).ln();
        let row = [0.0, l1, 0.0, l1, 0.0, l1, 0.0, l1];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.0,
            rpiou_kappa: 1e-30,
            box_gain: 2e38,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            tal: folo_vision::assign::TalConfig {
                nwd_weight: 1.0,
                ..folo_vision::assign::TalConfig::default()
            },
            ..LossConfig::default()
        };
        let gt = ImageTargets { boxes: vec![Xyxy::new(0.25, 0.25, 0.75, 0.75)], labels: vec![0] };
        let cases: [(&str, usize, Vec<ImageTargets>, f32); 3] = [
            ("b1_focal_iou", 1, vec![gt.clone()], 1.5e38),
            ("b2_two_pos_focal_iou", 2, vec![gt.clone(), gt.clone()], 3.0e38),
            ("b2_one_pos_focal_iou", 2, vec![gt, ImageTargets::default()], 3.0e38),
        ];
        for (tag, b, tgts, expected_loss) in cases {
            let mut bdata = Vec::with_capacity(b * 8);
            for _ in 0..b {
                bdata.extend_from_slice(&row);
            }
            let bx = Tensor::from_slice(&bdata, [b, 8, 1]).unwrap();
            let cx = Tensor::zeros([b, 1, 1]);
            let (bd, cd) =
                (bx.to_device(dev).unwrap().requires_grad_(), cx.to_device(dev).unwrap().requires_grad_());
            let (bh, ch) = (bx.requires_grad_(), cx.requires_grad_());
            let want = detection_loss_host(&bh, &ch, &anchors, &tgts, &cfg).unwrap();
            let got = detection_loss(&bd, &cd, &anchors, &tgts, &cfg).unwrap();
            assert!(got.is_finite() && want.is_finite(), "{tag}: non-finite loss");
            assert!(
                (want.total_value - expected_loss).abs() <= 2e-2 * expected_loss,
                "{tag}: CPU loss {} vs ~{expected_loss}",
                want.total_value
            );
            assert!(
                (got.total_value - want.total_value).abs() <= 1e-3 * want.total_value,
                "{tag}: GPU loss {} vs CPU {}",
                got.total_value,
                want.total_value
            );
            let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
            let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
            assert!(
                g_cpu.iter().all(|v| v.is_finite()) && g_gpu.iter().all(|v| v.is_finite()),
                "{tag}: non-finite grad"
            );
            let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(scale > 1e36, "{tag}: expected large finite focal-IoU logit grad, got {scale}");
            close_vec(&g_gpu, &g_cpu, 1e-3 * scale, 2e-3, &format!("{tag}/box_raw"));
        }
    }

    // 14. Matching 3x3 grid-unit boxes under large `box_gain = 2e38` (`beta = 0.5`): naive `1 - 9 / (9 + 1e-7)`
    //     rounds to `0` in `f32`, whereas the non-overlapping deficit retains `~3.33e30` matching CPU `f64`.
    {
        let anchors = Anchors { points: vec![[3.5, 3.5]], strides: vec![1.0], ..Anchors::default() };
        let mut bd_v = vec![0.0f32; 16];
        for s in 0..4 {
            bd_v[s * 4] = -1000.0;
            bd_v[s * 4 + 3] = -1000.0;
        }
        let bx = Tensor::from_slice(&bd_v, [1, 16, 1]).unwrap();
        let cx = Tensor::zeros([1, 1, 1]);
        let tgts = vec![ImageTargets { boxes: vec![Xyxy::new(2.0, 2.0, 5.0, 5.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 4,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 2.0,
            box_gain: 2e38,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            ..LossConfig::default()
        };
        let (bd, cd) = (bx.to_device(dev).unwrap(), cx.to_device(dev).unwrap());
        let want = detection_loss_host(&bx, &cx, &anchors, &tgts, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &tgts, &cfg).unwrap();
        let t_w = want.target_score_sum as f64;
        let uni = 9.0f64 + 1e-7f64;
        let ref_loss = (t_w * (1e-7f64 / uni) * (1.0 + 0.5 * (9.0 / uni)) * 2e38f64) as f32;
        assert!(ref_loss > 1e30 && want.is_finite() && got.is_finite());
        assert!(
            (want.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "CPU matching-3x3 R-PIoU {} vs ref {ref_loss}",
            want.total_value
        );
        assert!(
            (got.total_value - ref_loss).abs() <= 1e-3 * ref_loss,
            "GPU matching-3x3 R-PIoU {} vs ref {ref_loss}",
            got.total_value
        );
    }

    // 15. Subnormal-range softmax Jacobian in R-PIoU (`[-89, 0, 0, -1000]`, `box_gain = 1e38`, predicted 3x3 inside
    //     target 3.5x3.5): carrying `q` and `jac` in exponent-split form preserves bin 0's appreciable gradient even
    //     on FTZ GPUs instead of flushing to zero before `box_gain`.
    {
        let anchors = Anchors { points: vec![[3.5, 3.5]], strides: vec![1.0], ..Anchors::default() };
        let mut bd_v = vec![0.0f32; 16];
        for s in 0..4 {
            bd_v[s * 4] = -89.0;
            bd_v[s * 4 + 3] = -1000.0;
        }
        let bx = Tensor::from_slice(&bd_v, [1, 16, 1]).unwrap();
        let cx = Tensor::zeros([1, 1, 1]);
        let tgts = vec![ImageTargets { boxes: vec![Xyxy::new(1.75, 1.75, 5.25, 5.25)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 4,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 2.0,
            box_gain: 1e38,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            ..LossConfig::default()
        };
        let (bd, cd) =
            (bx.to_device(dev).unwrap().requires_grad_(), cx.to_device(dev).unwrap().requires_grad_());
        let (bh, ch) = (bx.requires_grad_(), cx.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &tgts, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &tgts, &cfg).unwrap();
        assert!(want.is_finite() && got.is_finite());
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu[0].abs() > 0.01, "expected appreciable CPU bin-0 grad, got {}", g_cpu[0]);
        assert!(g_gpu[0].abs() > 0.01, "expected appreciable GPU bin-0 grad, got {}", g_gpu[0]);
        assert!(
            (g_gpu[0] - g_cpu[0]).abs() <= 1e-3 * g_cpu[0].abs(),
            "GPU bin-0 grad {} vs CPU {}",
            g_gpu[0],
            g_cpu[0]
        );
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&g_gpu, &g_cpu, 1e-3 * scale, 2e-3, "subnormal_jac_rpiou/box_raw");
    }
}
