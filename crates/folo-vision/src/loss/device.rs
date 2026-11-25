//! The loss on a folo-gpu device: [`folo_gpu::detect`]'s kernels (assigner, fused BCE, fused CIoU + DFL), and
//! their assignment downloaded in [`assign_batch`]'s form for tests and tools.
//!
//! [`assign_batch`]: super::assign_batch

use folo_core::{Result, Tensor};

use super::{
    BatchAssignment, BoxLossKind, ClsLossKind, ImageTargets, LossConfig, LossItems, LossOutput, check_inputs,
    nan_loss,
};
use crate::boxes::Xyxy;
use crate::decode::Anchors;

/// The ground truth as folo-gpu's per-image tables.
fn gpu_targets(targets: &[ImageTargets]) -> Vec<(Vec<[f32; 4]>, &[u32])> {
    targets
        .iter()
        .map(|t| (t.boxes.iter().map(|b| [b.x1, b.y1, b.x2, b.y2]).collect(), &t.labels[..]))
        .collect()
}

fn gpu_tal(cfg: &LossConfig) -> folo_gpu::detect::TalSettings {
    let t = &cfg.tal;
    folo_gpu::detect::TalSettings {
        topk: t.topk,
        topk2: t.topk2,
        alpha: t.alpha,
        beta: t.beta,
        eps: t.eps,
        min_box_size: t.min_box_size,
        eps_free: t.eps_free,
        nwd_weight: t.nwd_weight,
    }
}

pub(super) fn detection_loss_device(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<LossOutput> {
    Ok(detection_loss_device_with_assignment(box_raw, cls_raw, anchors, targets, cfg)?.0)
}

pub(super) fn detection_loss_device_with_assignment(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<(LossOutput, Option<folo_gpu::detect::Assignment>)> {
    check_inputs(box_raw, cls_raw, anchors, cfg)?;
    let tables = gpu_targets(targets);
    let gts: Vec<_> = tables.iter().map(|(b, l)| folo_gpu::detect::ImageGt { boxes: b, labels: l }).collect();
    let at = folo_gpu::detect::AnchorTable { points: &anchors.points, strides: &anchors.strides };
    let cls_loss = match cfg.cls_loss {
        ClsLossKind::Bce => folo_gpu::detect::ClsLossKind::Bce,
        ClsLossKind::Vfl => folo_gpu::detect::ClsLossKind::Vfl,
        ClsLossKind::Qfl => folo_gpu::detect::ClsLossKind::Qfl,
    };
    let box_loss = match cfg.box_loss {
        BoxLossKind::Ciou => folo_gpu::detect::BoxLossKind::Ciou,
        BoxLossKind::Rpiou => folo_gpu::detect::BoxLossKind::Rpiou,
        BoxLossKind::Saeiou => folo_gpu::detect::BoxLossKind::Saeiou,
        other => unreachable!("device_loss_supported excludes {other:?}"),
    };
    let s = folo_gpu::detect::LossSettings {
        reg_max: cfg.reg_max,
        gains: [cfg.box_gain, cfg.cls_gain, cfg.dfl_gain],
        tal: gpu_tal(cfg),
        mal_gamma: cfg.mal_gamma,
        cls_loss,
        vfl_alpha: cfg.vfl_alpha,
        vfl_gamma: cfg.vfl_gamma,
        qfl_beta: cfg.qfl_beta,
        nwd_weight: cfg.nwd_weight,
        box_loss,
        rpiou_beta: cfg.rpiou_beta,
        rpiou_kappa: cfg.rpiou_kappa,
        rpiou_delta: cfg.rpiou_delta,
        dfl_ot_weight: cfg.dfl_ot_weight,
        dfl_crps_weight: cfg.dfl_crps_weight,
    };
    let (out, asg) = folo_gpu::detect::detection_loss_with_assignment(box_raw, cls_raw, &at, &gts, &s)?;
    // The loss's one readback: the reported items, the positives and the target score sum.
    let st = out.stats.to_vec::<f32>()?;
    if st[6] != 0.0 {
        return Ok((nan_loss(cls_raw.device())?, None));
    }
    Ok((
        LossOutput {
            total: out.total,
            total_value: st[0],
            items: LossItems { box_: st[1], cls: st[2], dfl: st[3] },
            num_fg: st[4] as usize,
            target_score_sum: st[5],
            mean_iou_loss: None,
        },
        Some(asg),
    ))
}

/// [`assign_batch`] on a folo-gpu device (the kernels [`detection_loss`] uses there), downloaded into the same
/// [`BatchAssignment`] form (image, then anchor order). For tests and tools: training never downloads it.
///
/// [`assign_batch`]: super::assign_batch
/// [`detection_loss`]: super::detection_loss
pub fn assign_batch_device(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<BatchAssignment> {
    let (b, nc, n_a) = check_inputs(box_raw, cls_raw, anchors, cfg)?;
    let tables = gpu_targets(targets);
    let gts: Vec<_> = tables.iter().map(|(b, l)| folo_gpu::detect::ImageGt { boxes: b, labels: l }).collect();
    let at = folo_gpu::detect::AnchorTable { points: &anchors.points, strides: &anchors.strides };
    let asg = folo_gpu::detect::assign(box_raw, cls_raw, &at, &gts, cfg.reg_max, &gpu_tal(cfg))?;
    let (g, k) = (asg.desc.g, asg.desc.k);
    let picks = asg.picks.to_vec::<u32>()?;
    let slots = asg.slots.to_vec::<u32>()?;
    let mut fg: Vec<(u32, u32, u32, f32, Xyxy)> = Vec::new();
    for s in 0..b * g * k {
        if slots[2 * s + 1] == 1 {
            let (i, gi) = (s / (g * k), (s / k) % g);
            let a = picks[3 * s];
            fg.push((i as u32, a, targets[i].labels[gi], f32::from_bits(slots[2 * s]), targets[i].boxes[gi]));
        }
    }
    fg.sort_by_key(|e| (e.0, e.1));
    // The loss kernels flag non-finite head values themselves; this test helper checks on the host like
    // `assign_batch`.
    let finite = [box_raw, cls_raw]
        .iter()
        .map(|t| t.to_vec::<f32>().map(|v| v.iter().all(|x| x.is_finite())))
        .collect::<Result<Vec<bool>>>()?
        .into_iter()
        .all(|f| f);
    let mut out = BatchAssignment {
        batch: b,
        num_anchors: n_a,
        num_classes: nc,
        inputs_finite: finite,
        ..Default::default()
    };
    if finite {
        for (i, a, l, t, bx) in fg {
            out.image.push(i);
            out.anchor.push(a);
            out.label.push(l);
            out.score.push(t);
            out.target.push(bx);
        }
    }
    Ok(out)
}
