//! CPU-vs-GPU parity tests for `S-AEIoU`, `CRPS-DFL`, and GPU-fused `VFL` / `QFL`.

use folo_vision::loss::{
    BoxLossKind, ClsLossKind, LossConfig, assign_batch, assign_batch_device, detection_loss,
    detection_loss_host, device_loss_supported,
};

use crate::{assert_same_assignment, case, close_vec, gpu};

#[test]
fn saeiou_crps_vfl_qfl_match_host() {
    let dev = gpu!();
    let configs = [
        ("saeiou", 71u64, LossConfig { box_loss: BoxLossKind::Saeiou, ..LossConfig::default() }),
        (
            "saeiou+nwd",
            72,
            LossConfig { box_loss: BoxLossKind::Saeiou, nwd_weight: 0.5, ..LossConfig::default() },
        ),
        ("crps_dfl", 73, LossConfig { dfl_crps_weight: 0.5, ..LossConfig::default() }),
        ("vfl", 77, LossConfig { cls_loss: ClsLossKind::Vfl, ..LossConfig::default() }),
        ("qfl", 78, LossConfig { cls_loss: ClsLossKind::Qfl, ..LossConfig::default() }),
        (
            "combo",
            80,
            LossConfig {
                box_loss: BoxLossKind::Saeiou,
                cls_loss: ClsLossKind::Vfl,
                dfl_crps_weight: 0.25,
                ..LossConfig::default()
            },
        ),
    ];
    for (name, seed, cfg) in configs {
        let (anchors, box_raw, cls_raw, targets) = case(seed, 128, 5);
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        assert!(device_loss_supported(&bd, &cfg), "{name}: must run on GPU");
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want_asg = assign_batch(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got_asg = assign_batch_device(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert_same_assignment(&got_asg, &want_asg);
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite(), "{name}: non-finite GPU loss");
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
    let ctx = folo_gpu::GpuContext::shared().unwrap();
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

#[test]
fn saeiou_and_crps_gpu_extreme_params_and_validation() {
    let dev = gpu!();
    // 1. Huge `rpiou_delta = f32::MAX` stays finite and matches CPU on GPU.
    // 2. Thin large box (`[0, 3.2, 1.6e7, 4.8]` -> stride-8 width 2e6, height 0.2) matches CPU.
    // 3. `dfl_crps_weight = 3e38` with `dfl_gain = 0.0` does not overflow per-slot terms on GPU.
    let anchors = folo_vision::decode::make_anchors(32, 32, &[8], 0.5);
    let a = anchors.len();
    let mut rng = folo_core::rng::Rng::seed(999);
    let bx = folo_core::Tensor::rand_uniform([1, 64, a], -1.0, 2.0, &mut rng);
    let cx = folo_core::Tensor::rand_uniform([1, 2, a], -2.0, 1.0, &mut rng);
    let norm_targets = vec![folo_vision::loss::ImageTargets {
        boxes: vec![folo_vision::boxes::Xyxy::new(2.0, 2.0, 26.0, 26.0)],
        labels: vec![0],
    }];
    let thin_targets = vec![folo_vision::loss::ImageTargets {
        boxes: vec![folo_vision::boxes::Xyxy::new(0.0, 3.2, 1.6e7, 4.8)],
        labels: vec![0],
    }];

    for (tag, cfg, tgts) in [
        (
            "delta_max",
            LossConfig { box_loss: BoxLossKind::Saeiou, rpiou_delta: f32::MAX, ..LossConfig::default() },
            &norm_targets,
        ),
        ("thin_large", LossConfig { box_loss: BoxLossKind::Saeiou, ..LossConfig::default() }, &thin_targets),
        (
            "crps_3e38_zero_gain",
            LossConfig {
                box_loss: BoxLossKind::Saeiou,
                dfl_crps_weight: 3e38,
                dfl_gain: 0.0,
                ..LossConfig::default()
            },
            &norm_targets,
        ),
    ] {
        let (bd, cd) = (
            bx.detach().to_device(dev).unwrap().requires_grad_(),
            cx.detach().to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (bx.detach().requires_grad_(), cx.detach().requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, tgts, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, tgts, &cfg).unwrap();
        assert!(got.is_finite() && want.is_finite(), "{tag}: non-finite {got:?} vs {want:?}");
        assert!(want.num_fg > 0, "{tag}: expected foreground anchors");
        let gi = got.items;
        let wi = want.items;
        let rel = |g: f32, w: f32| (g - w).abs() <= 2e-4 * w.abs().max(1e-3);
        assert!(
            rel(gi.box_, wi.box_) && rel(gi.cls, wi.cls) && rel(gi.dfl, wi.dfl),
            "{tag}: items {gi:?} vs {wi:?}"
        );
        let (gg, gw) = (got.total.backward().unwrap(), want.total.backward().unwrap());
        let gb = gg.get(&bd).unwrap().to_vec::<f32>().unwrap();
        let wb = gw.get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(gb.iter().all(|v| v.is_finite()), "{tag}: non-finite GPU grad");
        let scale = wb.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&gb, &wb, 2e-4 * scale.max(1e-4), 2e-3, &format!("{tag}/box_raw"));
    }

    // `cls_loss != Bce && mal_gamma != 0` rejected identically in `folo_gpu::detect`.
    let anchor_t = folo_gpu::detect::AnchorTable { points: &anchors.points, strides: &anchors.strides };
    let settings = folo_gpu::detect::LossSettings {
        reg_max: 16,
        gains: [7.5, 0.5, 1.5],
        tal: folo_gpu::detect::TalSettings {
            topk: 10,
            topk2: None,
            alpha: 0.5,
            beta: 6.0,
            eps: 1e-9,
            min_box_size: 16.0,
            eps_free: false,
            nwd_weight: 0.0,
        },
        mal_gamma: 1.0,
        cls_loss: folo_gpu::detect::ClsLossKind::Vfl,
        vfl_alpha: 0.75,
        vfl_gamma: 2.0,
        qfl_beta: 2.0,
        nwd_weight: 0.0,
        box_loss: folo_gpu::detect::BoxLossKind::Ciou,
        rpiou_beta: 0.5,
        rpiou_kappa: 2.0,
        rpiou_delta: 0.5,
        dfl_ot_weight: 0.0,
        dfl_crps_weight: 0.0,
    };
    let bd = bx.to_device(dev).unwrap();
    let cd = cx.to_device(dev).unwrap();
    let gt = [folo_gpu::detect::ImageGt { boxes: &[[2.0, 2.0, 26.0, 26.0]], labels: &[0] }];
    assert!(folo_gpu::detect::detection_loss_with_assignment(&bd, &cd, &anchor_t, &gt, &settings).is_err());
}
