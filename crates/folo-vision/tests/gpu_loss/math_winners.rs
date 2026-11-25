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
