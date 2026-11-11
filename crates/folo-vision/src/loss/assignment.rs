//! The detached part of the loss on the host (§7.1, §7.3): the head's predictions decoded per image and the
//! task-aligned assigner run on every image, giving the batch's foreground anchors ([`assign_batch`]).

use folo_core::cpu::fast_exp;
use folo_core::{Result, Tensor};
use rayon::prelude::*;

use super::{BatchAssignment, ImageTargets, LossConfig, check_inputs, host_f32, invalid};
use crate::assign::assign;
use crate::boxes::Xyxy;
use crate::decode::{Anchors, dist2bbox, sigmoid};

/// DFL decode of one image's channel-major box logits `[4 · reg_max, A]` (§6.4) to boxes in input pixels. Loops
/// run over contiguous anchor rows so they vectorise; the same max-subtracted softmax as
/// [`crate::decode::softmax_expectation`].
fn decode_channel_major(reg: &[f32], anchors: &Anchors, reg_max: usize) -> Vec<Xyxy> {
    let n_a = anchors.len();
    let mut ltrb = vec![[0f32; 4]; n_a];
    let (mut m, mut z, mut e) = (vec![0f32; n_a], vec![0f32; n_a], vec![0f32; n_a]);
    for side in 0..4 {
        let bins = &reg[side * reg_max * n_a..(side + 1) * reg_max * n_a];
        m.fill(f32::NEG_INFINITY);
        for row in bins.chunks_exact(n_a) {
            m.iter_mut().zip(row).for_each(|(m, &x)| *m = m.max(x));
        }
        z.fill(0.0);
        e.fill(0.0);
        for (i, row) in bins.chunks_exact(n_a).enumerate() {
            let fi = i as f32;
            for (((z, e), &m), &x) in z.iter_mut().zip(e.iter_mut()).zip(&m).zip(row) {
                let p = fast_exp(x - m);
                *z += p;
                *e += p * fi;
            }
        }
        for ((d, &z), &e) in ltrb.iter_mut().zip(&z).zip(&e) {
            d[side] = e / z;
        }
    }
    ltrb.iter()
        .enumerate()
        .map(|(a, &d)| {
            let s = anchors.strides[a];
            dist2bbox(d, anchors.points[a]).scale(s, s)
        })
        .collect()
}

/// One image's foreground entries.
#[derive(Default)]
struct ImageFg {
    anchor: Vec<u32>,
    label: Vec<u32>,
    score: Vec<f32>,
    target: Vec<Xyxy>,
}

/// The detached part of the loss (§7.1, §7.3): decode the predictions on the host and run the task-aligned
/// assigner on every image.
///
/// Only the score columns of the classes present in an image are computed (the assigner reads no others), so the
/// host work is `O(B · A · (4 · reg_max + classes present))`, independent of `nc`. Errors on mismatched shapes,
/// `targets.len() != B`, or a gt label `>= nc`.
pub fn assign_batch(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<BatchAssignment> {
    let (b, nc, n_a) = check_inputs(box_raw, cls_raw, anchors, cfg)?;
    if targets.len() != b {
        return Err(invalid(format!("{} target images for a batch of {b}", targets.len())));
    }
    for (i, t) in targets.iter().enumerate() {
        if t.labels.len() != t.boxes.len() {
            return Err(invalid(format!("image {i}: {} boxes but {} labels", t.boxes.len(), t.labels.len())));
        }
        if let Some(&l) = t.labels.iter().find(|&&l| l as usize >= nc) {
            return Err(invalid(format!("image {i}: class {l} >= nc {nc}")));
        }
    }
    let reg_max = cfg.reg_max;
    let points_px = anchors.points_px();
    let (box_len, cls_len) = (4 * reg_max * n_a, nc * n_a);

    let per_image: Vec<(bool, ImageFg)> = host_f32(box_raw, |boxes| {
        host_f32(cls_raw, |cls| {
            (0..b)
                .into_par_iter()
                .map(|i| {
                    let reg = &boxes[i * box_len..(i + 1) * box_len];
                    let logits = &cls[i * cls_len..(i + 1) * cls_len];
                    let finite = reg.iter().chain(logits).all(|v| v.is_finite());
                    let gt = &targets[i];
                    if !finite || gt.is_empty() {
                        return (finite, ImageFg::default());
                    }
                    let pd_px = decode_channel_major(reg, anchors, reg_max);
                    // The assigner sees only the classes present, renumbered 0..k.
                    let mut classes = gt.labels.clone();
                    classes.sort_unstable();
                    classes.dedup();
                    let k = classes.len();
                    let local: Vec<u32> = gt
                        .labels
                        .iter()
                        .map(|l| classes.binary_search(l).expect("class present") as u32)
                        .collect();
                    let mut scores = vec![0f32; n_a * k];
                    for (j, &c) in classes.iter().enumerate() {
                        let row = &logits[c as usize * n_a..(c as usize + 1) * n_a];
                        for (s, &x) in scores.iter_mut().skip(j).step_by(k).zip(row) {
                            *s = sigmoid(x);
                        }
                    }
                    let asg = assign(&scores, k, &pd_px, &points_px, &gt.boxes, &local, &cfg.tal);
                    let mut fg = ImageFg::default();
                    for a in (0..n_a).filter(|&a| asg.fg_mask[a]) {
                        fg.anchor.push(a as u32);
                        fg.label.push(gt.labels[asg.target_gt[a]]);
                        fg.score.push(asg.target_scores[a]);
                        fg.target.push(asg.target_bboxes[a]);
                    }
                    (true, fg)
                })
                .collect()
        })
    })??;

    let mut out = BatchAssignment {
        batch: b,
        num_anchors: n_a,
        num_classes: nc,
        inputs_finite: per_image.iter().all(|(f, _)| *f),
        ..Default::default()
    };
    if out.inputs_finite {
        for (i, (_, fg)) in per_image.into_iter().enumerate() {
            out.image.extend(std::iter::repeat_n(i as u32, fg.anchor.len()));
            out.anchor.extend(fg.anchor);
            out.label.extend(fg.label);
            out.score.extend(fg.score);
            out.target.extend(fg.target);
        }
    }
    Ok(out)
}
