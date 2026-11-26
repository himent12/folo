//! The differentiable terms of §7.4 for a fixed assignment ([`loss_with_assignment`]):
//! - [`cls`]: BCE, MAL, Varifocal Loss (VFL), Quality Focal Loss (QFL),
//! - [`vfl_qfl`]: Standalone cancellation-free Varifocal Loss (VFL) and Quality Focal Loss (QFL) with reductions,
//! - [`asl_poly`]: Asymmetric Loss (ASL), Poly-1 BCE, and Poly-1 Focal Loss,
//! - [`box_loss`]: CIoU, Inner-CIoU, R-PIoU, S-AEIoU,
//! - [`nwd_wiou`]: Normalized Wasserstein Distance (NWD) and Wise-IoU v1 / v3,
//! - [`geom_loss`]: MPDIoU, Shape-IoU, and Focaler-IoU remapping,
//! - [`siou_eiou`]: EIoU (Efficient-IoU) and SIoU (Scylla-IoU),
//! - [`dfl`]: DFL cross-entropy, 1D Optimal-Transport tail barrier (`OT-DFL`), and `CRPS-DFL`.

mod asl_poly;
mod box_loss;
mod cls;
mod dfl;
mod geom_loss;
mod nwd_wiou;
mod siou_eiou;
mod vfl_qfl;

#[cfg(test)]
pub(crate) use asl_poly::{asl_elem_f64, poly1_bce_elem_f64, poly1_focal_elem_f64};
pub use asl_poly::{asymmetric_loss, poly1_bce_loss, poly1_focal_loss, poly1_focal_loss_with_alpha};
pub use vfl_qfl::{LossReduction, QflConfig, VflConfig, quality_focal_loss, varifocal_loss};
#[cfg(test)]
pub(crate) use vfl_qfl::{qfl_elem_f64, vfl_elem_f64};

#[cfg(test)]
pub(crate) use box_loss::saeiou_box_f64;
pub use box_loss::{ciou, inner_ciou_loss, rpiou_loss, saeiou_loss};
pub(crate) use dfl::crps_dfl;
pub use geom_loss::{focaler_iou, focaler_iou_loss, mpdiou_loss, mpdiou_loss_with_scale, shape_iou_loss};
#[cfg(test)]
pub(crate) use geom_loss::{mpdiou_box_f64, shape_iou_box_f64};
#[cfg(test)]
pub(crate) use nwd_wiou::{
    nwd_box_eps_f64, nwd_box_f64, wiou_v1_box_f64, wiou_v3_box_f64, wiou_v3_focus_gain,
};
pub use nwd_wiou::{nwd_loss, nwd_loss_with_c, wiou_v1_loss, wiou_v3_loss};
#[cfg(test)]
pub(crate) use siou_eiou::{eiou_box_f64, siou_box_f64};
pub use siou_eiou::{eiou_loss, siou_loss};

use folo_core::{DType, Result, Tensor};

use super::{BatchAssignment, BoxLossKind, LossConfig, LossItems, LossOutput, check_inputs, invalid};
use crate::assign::{bbox2dist_clamped, dfl_bins, nwd_lambda};
use crate::decode::Anchors;

use box_loss::{rpiou_from_logits, saeiou_from_logits};
use cls::cls_loss_sum;
use dfl::ot_dfl_tail;
use geom_loss::{mpdiou_from_logits, shape_iou_from_logits};
use nwd_wiou::{nwd_blend_loss_eps, nwd_from_logits, wiou_v1_from_logits, wiou_v3_from_logits};
use siou_eiou::{eiou_from_logits, siou_from_logits};

/// Sum of every element as a rank-0 tensor, in an order that does not depend on thread scheduling: long inputs are
/// summed in rows of 4096 (each row serially), then the row sums the same way.
pub(super) fn sum_fixed_order(x: &Tensor) -> Result<Tensor> {
    const ROW: usize = 4096;
    let n = x.numel();
    let flat = x.flatten_all()?;
    if n <= ROW {
        return flat.sum_all();
    }
    let rows = n.div_ceil(ROW);
    let padded = if rows * ROW > n { flat.pad(&[(0, rows * ROW - n)], 0.0)? } else { flat };
    sum_fixed_order(&padded.reshape([rows, ROW])?.sum(1, false)?)
}

/// `x.flatten_all().index_select(0, idx)` for a CPU tensor, with a backward that scatters into a fresh zeroed
/// buffer (lazily zeroed pages) instead of copying a dense zero tensor.
fn gather_flat_cpu(x: &Tensor, idx: Vec<usize>) -> Result<Tensor> {
    let picked = x.with_f32_slice(|v| idx.iter().map(|&i| v[i]).collect::<Vec<f32>>())?;
    let out = Tensor::from_vec(picked, [idx.len()])?;
    let shape = x.shape().clone();
    Tensor::custom_op("gather_flat", &[x], out, move |g| {
        let mut grad = vec![0f32; shape.numel()];
        g.with_f32_slice(|g| {
            for (&i, &g) in idx.iter().zip(g) {
                grad[i] += g;
            }
        })?;
        Ok(vec![Some(Tensor::from_vec(grad, &shape)?)])
    })
}

/// The differentiable terms of §7.4 for a fixed assignment (from [`assign_batch`] on the same outputs).
///
/// The assignment is a constant, so this is the function whose gradient training uses; [`detection_loss`] is
/// `assign_batch` followed by this. Errors if the assignment does not match the outputs' shapes.
///
/// [`assign_batch`]: super::assign_batch
/// [`detection_loss`]: super::detection_loss
pub fn loss_with_assignment(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    cfg: &LossConfig,
) -> Result<LossOutput> {
    loss_impl_wiou(box_raw, cls_raw, anchors, asg, cfg, box_raw.device().is_cpu(), 1.0)
}

/// [`loss_with_assignment`] with an explicit WIoU v3 running-EMA denominator `wiou_iou_mean`.
pub(super) fn loss_with_assignment_wiou(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    cfg: &LossConfig,
    wiou_iou_mean: f32,
) -> Result<LossOutput> {
    loss_impl_wiou(box_raw, cls_raw, anchors, asg, cfg, box_raw.device().is_cpu(), wiou_iou_mean)
}

/// [`loss_with_assignment`] with the kernel choice explicit: `fused` uses the CPU custom ops, otherwise the
/// device-generic composite of tensor ops. Both give the same values and gradients (the tests compare them).
#[cfg(test)]
pub(super) fn loss_impl(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    cfg: &LossConfig,
    fused: bool,
) -> Result<LossOutput> {
    loss_impl_wiou(box_raw, cls_raw, anchors, asg, cfg, fused, 1.0)
}

fn loss_impl_wiou(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    cfg: &LossConfig,
    fused: bool,
    wiou_iou_mean: f32,
) -> Result<LossOutput> {
    let (b, nc, n_a) = check_inputs(box_raw, cls_raw, anchors, cfg)?;
    if (asg.batch, asg.num_classes, asg.num_anchors) != (b, nc, n_a) {
        return Err(invalid(format!(
            "assignment is for B {} nc {} A {}, outputs are B {b} nc {nc} A {n_a}",
            asg.batch, asg.num_classes, asg.num_anchors
        )));
    }
    let n = asg.num_fg();
    if [asg.image.len(), asg.label.len(), asg.score.len(), asg.target.len()] != [n; 4] {
        return Err(invalid("assignment vectors differ in length"));
    }
    let dev = cls_raw.device();
    let r = cfg.reg_max;
    let tss_raw = asg.target_score_sum();
    let tss = tss_raw.max(1.0) as f32;

    let mut idx_cls = Vec::with_capacity(n);
    let mut idx_box = Vec::with_capacity(n * 4 * r);
    let mut anchor4 = Vec::with_capacity(n * 4);
    let mut tbox = Vec::with_capacity(n * 4);
    let mut bins = Vec::with_capacity(n * 8);
    let mut bin_w = Vec::with_capacity(n * 8);
    let use_ot = cfg.dfl_ot_weight > 0.0 && r > 2;
    let use_crps = cfg.dfl_crps_weight > 0.0;
    let need_side_meta = use_ot || use_crps;
    let mut left_bins = if need_side_meta { Vec::with_capacity(n * 4) } else { Vec::new() };
    let mut left_w = if use_crps { Vec::with_capacity(n * 4) } else { Vec::new() };
    let mut side_wt = if need_side_meta { Vec::with_capacity(n * 4) } else { Vec::new() };
    for k in 0..n {
        let (i, a, c) = (asg.image[k] as usize, asg.anchor[k] as usize, asg.label[k] as usize);
        if i >= b || a >= n_a || c >= nc {
            return Err(invalid(format!(
                "assignment entry {k} (image {i}, anchor {a}, class {c}) is out of range"
            )));
        }
        idx_cls.push((i * nc + c) * n_a + a);
        let base = i * 4 * r * n_a + a;
        idx_box.extend((0..4 * r).map(|ch| base + ch * n_a));
        let p = anchors.points[a];
        anchor4.extend([p[0], p[1], p[0], p[1]]);
        let inv = 1.0 / anchors.strides[a];
        let g = asg.target[k].scale(inv, inv);
        tbox.extend([g.x1, g.y1, g.x2, g.y2]);
        let wt = asg.score[k] * 0.25;
        for d in bbox2dist_clamped(p, &g, r) {
            let (j, wl, wr) = dfl_bins(d);
            bins.extend([j as i64, j as i64 + 1]);
            bin_w.extend([wl * wt, wr * wt]);
            if need_side_meta {
                left_bins.push(j);
                side_wt.push(wt);
            }
            if use_crps {
                left_w.push(wl);
            }
        }
    }
    // An anchor has at most one gt (§7.3 step 4); the classification kernels rely on distinct foreground positions.
    let mut keys: Vec<usize> = (0..n).map(|k| asg.image[k] as usize * n_a + asg.anchor[k] as usize).collect();
    keys.sort_unstable();
    if keys.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid("assignment lists an anchor of an image more than once"));
    }
    let on_dev = |t: Tensor| t.to_device(dev);
    let index = |v: Vec<usize>| -> Result<Tensor> {
        let len = v.len();
        on_dev(Tensor::from_vec(v.into_iter().map(|i| i as i64).collect::<Vec<_>>(), [len])?)
    };
    let t = on_dev(Tensor::from_slice(&asg.score, [n])?)?;

    let cls_sum = cls_loss_sum(cls_raw, idx_cls, &asg.score, cfg, fused)?;

    let (wb, wd, mean_iou_loss) = if n == 0 {
        (Tensor::zeros_with([], DType::F32, dev)?, Tensor::zeros_with([], DType::F32, dev)?, None)
    } else {
        let logits = if fused {
            gather_flat_cpu(box_raw, idx_box)?
        } else {
            box_raw.flatten_all()?.index_select(0, &index(idx_box)?)?
        };
        let logits = logits.reshape([n, 4, r])?;

        // Box: DFL expectation → grid-unit xyxy → CIoU / Inner-CIoU / WIoU v3 / R-PIoU (+ optional NWD blend).
        let proj = on_dev(Tensor::arange(0.0, r as f32, 1.0)?)?;
        let probs = logits.softmax(2)?;
        let dist = probs.mul(&proj)?.sum(2, false)?;
        let sign = on_dev(Tensor::from_slice(&[-1.0f32, -1.0, 1.0, 1.0], [4])?)?;
        let pd = on_dev(Tensor::from_vec(anchor4, [n, 4])?)?.add(&dist.mul(&sign)?)?;
        let tb = on_dev(Tensor::from_slice(&tbox, [n, 4])?)?;
        let blend_nwd = |l_base: Tensor| -> Result<Tensor> {
            let box_per = if cfg.nwd_weight > 0.0 {
                let lams: Vec<f32> = (0..n).map(|k| nwd_lambda(&asg.target[k], cfg.nwd_weight)).collect();
                let cs: Vec<f32> = (0..n).map(|k| 12.8 / anchors.strides[asg.anchor[k] as usize]).collect();
                let lam_t = on_dev(Tensor::from_vec(lams, [n])?)?;
                let c_t = on_dev(Tensor::from_vec(cs, [n])?)?;
                let l_nwd = nwd_blend_loss_eps(&pd, &tb, &c_t)?;
                lam_t.rsub_scalar(1.0)?.mul(&l_base)?.add(&lam_t.mul(&l_nwd)?)?
            } else {
                l_base
            };
            sum_fixed_order(&box_per.mul(&t)?)?.div_scalar(tss)?.mul_scalar(cfg.box_gain)
        };
        let mut mean_iou_loss = None;
        let wb = match cfg.box_loss {
            BoxLossKind::Ciou => blend_nwd(ciou(&pd, &tb)?.rsub_scalar(1.0)?)?,
            BoxLossKind::InnerCiou => blend_nwd(inner_ciou_loss(&pd, &tb, cfg.inner_ratio)?)?,
            BoxLossKind::Nwd
            | BoxLossKind::WiouV1
            | BoxLossKind::WiouV3
            | BoxLossKind::Rpiou
            | BoxLossKind::Saeiou
            | BoxLossKind::Mpdiou
            | BoxLossKind::ShapeIou
            | BoxLossKind::Eiou
            | BoxLossKind::Siou => {
                let mut eval_direct = |lams_opt: Option<&[f32]>| -> Result<Tensor> {
                    match cfg.box_loss {
                        BoxLossKind::Nwd => {
                            let cs: Vec<f32> =
                                (0..n).map(|k| cfg.nwd_c / anchors.strides[asg.anchor[k] as usize]).collect();
                            nwd_from_logits(&logits, &pd, &tbox, &cs, &asg.score, lams_opt, tss, cfg.box_gain)
                        }
                        BoxLossKind::WiouV1 => {
                            wiou_v1_from_logits(&logits, &pd, &tbox, &asg.score, lams_opt, tss, cfg.box_gain)
                        }
                        BoxLossKind::WiouV3 => {
                            let (loss, batch_mean) = wiou_v3_from_logits(
                                &logits,
                                &pd,
                                &tbox,
                                &asg.score,
                                lams_opt,
                                tss,
                                cfg.box_gain,
                                cfg.wiou_alpha,
                                cfg.wiou_delta,
                                wiou_iou_mean,
                            )?;
                            mean_iou_loss = batch_mean;
                            Ok(loss)
                        }
                        BoxLossKind::Rpiou => rpiou_from_logits(
                            &logits,
                            &pd,
                            &tbox,
                            &asg.score,
                            lams_opt,
                            tss,
                            cfg.box_gain,
                            cfg.rpiou_beta,
                            cfg.rpiou_kappa,
                            cfg.rpiou_delta,
                        ),
                        BoxLossKind::Saeiou => saeiou_from_logits(
                            &logits,
                            &pd,
                            &tbox,
                            &asg.score,
                            lams_opt,
                            tss,
                            cfg.box_gain,
                            cfg.rpiou_beta,
                            cfg.rpiou_kappa,
                            cfg.rpiou_delta,
                        ),
                        BoxLossKind::Mpdiou => {
                            mpdiou_from_logits(&logits, &pd, &tbox, &asg.score, lams_opt, tss, cfg.box_gain)
                        }
                        BoxLossKind::ShapeIou => shape_iou_from_logits(
                            &logits,
                            &pd,
                            &tbox,
                            &asg.score,
                            lams_opt,
                            tss,
                            cfg.box_gain,
                            cfg.shape_scale,
                        ),
                        BoxLossKind::Eiou => {
                            eiou_from_logits(&logits, &pd, &tbox, &asg.score, lams_opt, tss, cfg.box_gain)
                        }
                        BoxLossKind::Siou => siou_from_logits(
                            &logits,
                            &pd,
                            &tbox,
                            &asg.score,
                            lams_opt,
                            tss,
                            cfg.box_gain,
                            cfg.siou_theta,
                        ),
                        _ => unreachable!(),
                    }
                };
                if cfg.nwd_weight > 0.0 && cfg.box_loss != BoxLossKind::Nwd {
                    let lams: Vec<f32> = (0..n).map(|k| nwd_lambda(&asg.target[k], cfg.nwd_weight)).collect();
                    let wb_alg = eval_direct(Some(&lams))?;
                    let cs: Vec<f32> =
                        (0..n).map(|k| 12.8 / anchors.strides[asg.anchor[k] as usize]).collect();
                    let lam_t = on_dev(Tensor::from_vec(lams, [n])?)?;
                    let c_t = on_dev(Tensor::from_vec(cs, [n])?)?;
                    let l_nwd = nwd_blend_loss_eps(&pd, &tb, &c_t)?;
                    let wb_nwd = sum_fixed_order(&lam_t.mul(&l_nwd)?.mul(&t)?)?
                        .div_scalar(tss)?
                        .mul_scalar(cfg.box_gain)?;
                    wb_alg.add(&wb_nwd)?
                } else {
                    eval_direct(None)?
                }
            }
        };

        // DFL: cross-entropy against the two neighbouring bins, plus optional OT tail barrier & CRPS-DFL.
        let bins = on_dev(Tensor::from_vec(bins, [n, 4, 2])?)?;
        let bin_w = on_dev(Tensor::from_vec(bin_w, [n, 4, 2])?)?;
        let picked = logits.log_softmax(2)?.gather(2, &bins)?;
        let l_dfl_ce = sum_fixed_order(&picked.mul(&bin_w)?)?.neg()?.div_scalar(tss)?;
        let mut wd = l_dfl_ce.mul_scalar(cfg.dfl_gain)?;
        if use_ot {
            wd =
                wd.add(&ot_dfl_tail(&logits, &left_bins, &side_wt, cfg.dfl_ot_weight, tss, cfg.dfl_gain)?)?;
        }
        if use_crps {
            wd = wd.add(&crps_dfl(
                &logits,
                &left_bins,
                &left_w,
                &side_wt,
                cfg.dfl_crps_weight,
                tss,
                cfg.dfl_gain,
            )?)?;
        }
        (wb, wd, mean_iou_loss)
    };
    let wc = cls_sum.div_scalar(tss)?.mul_scalar(cfg.cls_gain)?;

    let items = LossItems { box_: wb.item::<f32>()?, cls: wc.item::<f32>()?, dfl: wd.item::<f32>()? };
    let total = wb.add(&wc)?.add(&wd)?.mul_scalar(b as f32)?;
    let total_value = total.item::<f32>()?;
    Ok(LossOutput { total, total_value, items, num_fg: n, target_score_sum: tss_raw as f32, mean_iou_loss })
}
