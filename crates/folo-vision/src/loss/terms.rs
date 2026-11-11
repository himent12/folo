//! The differentiable terms of §7.4 for a fixed assignment ([`loss_with_assignment`]): BCE, CIoU ([`ciou`]) and
//! DFL, with the fused CPU custom ops (BCE sum, foreground gather) and fixed-order sums they are built from.

use folo_core::cpu::fast_exp;
use folo_core::{DType, Result, Tensor, no_grad};
use rayon::prelude::*;

use super::{BatchAssignment, LossConfig, LossItems, LossOutput, check_inputs, invalid};
use crate::assign::{bbox2dist_clamped, dfl_bins};
use crate::decode::Anchors;

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

/// `atan(x)` elementwise. folo-core has no `atan` kernel, so the forward runs on the host (the CIoU inputs are one
/// value per foreground anchor) and the backward `g / (1 + x²)` uses tensor ops on `x`'s device.
fn atan(x: &Tensor) -> Result<Tensor> {
    let v: Vec<f32> = x.to_vec::<f32>()?.into_iter().map(f32::atan).collect();
    let out = Tensor::from_vec(v, x.shape())?.to_device(x.device())?;
    let xs = x.detach();
    Tensor::custom_op("atan", &[x], out, move |g| Ok(vec![Some(g.div(&xs.square()?.add_scalar(1.0)?)?)]))
}

/// Rows per parallel task of [`bce_sum_cpu`].
const PAR_ROWS: usize = 8;

/// `softplus(x) = max(x, 0) + ln(1 + e^-|x|)`, as folo-core computes it.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + fast_exp(-x.abs()).ln_1p()
}

/// `Σ softplus(x)` with 8 independent accumulators (vectorises).
fn softplus_sum(r: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (chunks, rest) = r.as_chunks::<8>();
    for ch in chunks {
        for (a, &x) in acc.iter_mut().zip(ch) {
            *a += softplus(x);
        }
    }
    acc.iter().sum::<f32>() + rest.iter().map(|&x| softplus(x)).sum::<f32>()
}

/// Stable sigmoid, as folo-core computes it.
#[inline(always)]
fn sigmoid_fast(x: f32) -> f32 {
    let e = fast_exp(-x.abs());
    let r = 1.0 / (1.0 + e);
    if x >= 0.0 { r } else { e * r }
}

/// `Σ_i softplus(x_i) − Σ_k x[idx_k] · t_k` over a CPU tensor, i.e. `Σ BCE(x, T)` for a target `T` that is `t_k` at
/// the flat positions `idx_k` and 0 elsewhere, with the backward `g · (sigmoid(x) − T)` in one pass.
///
/// This fuses what the composite form does in several full-size passes (softplus, sums, and an `index_select`
/// whose backward materialises a dense zero gradient). `x` is viewed as rows of its last dimension; rows are
/// summed in parallel, each serially, and combined in row order in `f64`, so the result is reproducible.
fn bce_sum_cpu(x: &Tensor, idx: Vec<usize>, t: Vec<f32>) -> Result<Tensor> {
    let row = x.dims().last().copied().unwrap_or(1).max(1);
    // Foreground positions sorted, so each row finds its own with a binary search. They are distinct (checked by
    // the caller). The dense sum skips them and each gets its own `BCE(x, t)`: summing `softplus(x)` and `x · t`
    // separately would cancel catastrophically for confident positives (large `x`, `t ≈ 1`).
    let mut order: Vec<usize> = (0..idx.len()).collect();
    order.sort_unstable_by_key(|&k| idx[k]);
    let pos: Vec<usize> = order.iter().map(|&k| idx[k]).collect();
    let value = x.with_f32_slice(|v| {
        let rows: Vec<f32> = v
            .par_chunks(row * PAR_ROWS)
            .enumerate()
            .flat_map_iter(|(ci, c)| {
                let pos = &pos;
                c.chunks(row).enumerate().map(move |(ri, r)| {
                    let start = (ci * PAR_ROWS + ri) * row;
                    let lo = pos.partition_point(|&p| p < start);
                    let hi = pos.partition_point(|&p| p < start + r.len());
                    let mut s = 0f32;
                    let mut from = 0;
                    for &p in &pos[lo..hi] {
                        s += softplus_sum(&r[from..p - start]);
                        from = p - start + 1;
                    }
                    s + softplus_sum(&r[from..])
                })
            })
            .collect();
        let dense: f64 = rows.iter().map(|&r| r as f64).sum();
        // softplus(x) >= max(x, 0) and t in [0, 1], so every foreground term is >= 0.
        let fg: f64 =
            order.iter().map(|&k| softplus(v[idx[k]]) as f64 - v[idx[k]] as f64 * t[k] as f64).sum();
        (dense + fg) as f32
    })?;
    let xs = x.detach();
    Tensor::custom_op("bce_with_logits_sum", &[x], Tensor::scalar(value), move |g| {
        let g = g.item::<f32>()?;
        let mut grad = xs.with_f32_slice(|v| {
            let mut out = vec![0f32; v.len()];
            out.par_chunks_mut(row * PAR_ROWS).zip(v.par_chunks(row * PAR_ROWS)).for_each(|(o, x)| {
                for (o, &x) in o.iter_mut().zip(x) {
                    *o = g * sigmoid_fast(x);
                }
            });
            out
        })?;
        for (&i, &t) in idx.iter().zip(&t) {
            grad[i] -= g * t;
        }
        Ok(vec![Some(Tensor::from_vec(grad, xs.shape())?)])
    })
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

/// Differentiable CIoU of `docs/MODEL.md` §7.2 between rows of `b1` and `b2` (`[N, 4]`, `xyxy`), as `[N]`.
///
/// The same formula and `eps` placements as [`crate::boxes::ciou_eps`] (which the assigner uses); α is computed
/// under `no_grad`, so no gradient flows through it. Gradients flow into both inputs.
pub fn ciou(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    const EPS: f32 = 1e-7;
    let col = |b: &Tensor, k: usize| b.narrow(1, k, 1)?.squeeze(1);
    let (ax1, ay1, ax2, ay2) = (col(b1, 0)?, col(b1, 1)?, col(b1, 2)?, col(b1, 3)?);
    let (bx1, by1, bx2, by2) = (col(b2, 0)?, col(b2, 1)?, col(b2, 2)?, col(b2, 3)?);
    let (w1, h1) = (ax2.sub(&ax1)?, ay2.sub(&ay1)?.add_scalar(EPS)?);
    let (w2, h2) = (bx2.sub(&bx1)?, by2.sub(&by1)?.add_scalar(EPS)?);
    let iw = ax2.minimum(&bx2)?.sub(&ax1.maximum(&bx1)?)?.relu()?;
    let ih = ay2.minimum(&by2)?.sub(&ay1.maximum(&by1)?)?.relu()?;
    let inter = iw.mul(&ih)?;
    let union = w1.mul(&h1)?.add(&w2.mul(&h2)?)?.sub(&inter)?.add_scalar(EPS)?;
    let iou = inter.div(&union)?;
    let cw = ax2.maximum(&bx2)?.sub(&ax1.minimum(&bx1)?)?;
    let ch = ay2.maximum(&by2)?.sub(&ay1.minimum(&by1)?)?;
    let c2 = cw.square()?.add(&ch.square()?)?.add_scalar(EPS)?;
    let dx = bx1.add(&bx2)?.sub(&ax1)?.sub(&ax2)?;
    let dy = by1.add(&by2)?.sub(&ay1)?.sub(&ay2)?;
    let rho2 = dx.square()?.add(&dy.square()?)?.mul_scalar(0.25)?;
    let d = atan(&w2.div(&h2)?)?.sub(&atan(&w1.div(&h1)?)?)?;
    let v = d.square()?.mul_scalar(4.0 / (std::f32::consts::PI * std::f32::consts::PI))?;
    let alpha = {
        let _g = no_grad();
        v.div(&v.sub(&iou)?.add_scalar(1.0 + EPS)?)?
    };
    iou.sub(&rho2.div(&c2)?.add(&v.mul(&alpha)?)?)
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
    loss_impl(box_raw, cls_raw, anchors, asg, cfg, box_raw.device().is_cpu())
}

/// [`loss_with_assignment`] with the kernel choice explicit: `fused` uses the CPU custom ops ([`bce_sum_cpu`],
/// [`gather_flat_cpu`]), otherwise the device-generic composite of tensor ops. Both give the same values and
/// gradients (the tests compare them).
pub(super) fn loss_impl(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    cfg: &LossConfig,
    fused: bool,
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

    // Per foreground entry: the flat positions of its class logit and of its 4·reg_max box logits, its anchor
    // point and grid-unit target box, and its DFL bins with weights t[a] / 4 (the mean over sides), so the DFL term
    // is one weighted sum.
    let mut idx_cls = Vec::with_capacity(n);
    let mut idx_box = Vec::with_capacity(n * 4 * r);
    let mut anchor4 = Vec::with_capacity(n * 4);
    let mut tbox = Vec::with_capacity(n * 4);
    let mut bins = Vec::with_capacity(n * 8);
    let mut bin_w = Vec::with_capacity(n * 8);
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
        }
    }
    // An anchor has at most one gt (§7.3 step 4); the BCE kernels rely on distinct foreground positions.
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

    // Classification: Σ BCE(x, T) = Σ_bg softplus(x) + Σ_fg (softplus(x) − x · t), since T is t at the assigned
    // class of each foreground anchor and 0 elsewhere. Foreground terms are formed per element before any sum, so a
    // confident positive (large x, t ≈ 1) does not cancel against a large dense sum.
    let cls_sum = if fused {
        bce_sum_cpu(cls_raw, idx_cls, asg.score.clone())?
    } else if n == 0 {
        sum_fixed_order(&cls_raw.softplus()?.sum(2, false)?)?
    } else {
        // A constant 0/1 mask, built on the device (ones with −1 added at the foreground), removes the foreground
        // positions from the dense sum exactly, value and gradient. Per-(image, class) row sums keep a fixed order.
        let idx = index(idx_cls)?;
        let x_fg = cls_raw.flatten_all()?.index_select(0, &idx)?;
        let bg_mask = {
            let _g = no_grad();
            let ones = Tensor::full_with([b * nc * n_a], 1.0, DType::F32, dev)?;
            ones.index_add(0, &idx, &Tensor::full_with([n], -1.0, DType::F32, dev)?)?.reshape([b, nc, n_a])?
        };
        let dense = sum_fixed_order(&cls_raw.softplus()?.mul(&bg_mask)?.sum(2, false)?)?;
        dense.add(&sum_fixed_order(&x_fg.softplus()?.sub(&x_fg.mul(&t)?)?)?)?
    };

    let (l_box, l_dfl) = if n == 0 {
        (Tensor::zeros_with([], DType::F32, dev)?, Tensor::zeros_with([], DType::F32, dev)?)
    } else {
        let logits = if fused {
            gather_flat_cpu(box_raw, idx_box)?
        } else {
            box_raw.flatten_all()?.index_select(0, &index(idx_box)?)?
        };
        let logits = logits.reshape([n, 4, r])?;

        // Box: DFL expectation → grid-unit xyxy → CIoU against the target.
        let proj = on_dev(Tensor::arange(0.0, r as f32, 1.0)?)?;
        let dist = logits.softmax(2)?.mul(&proj)?.sum(2, false)?;
        let sign = on_dev(Tensor::from_slice(&[-1.0f32, -1.0, 1.0, 1.0], [4])?)?;
        let pd = on_dev(Tensor::from_vec(anchor4, [n, 4])?)?.add(&dist.mul(&sign)?)?;
        let tb = on_dev(Tensor::from_vec(tbox, [n, 4])?)?;
        let l_box = sum_fixed_order(&ciou(&pd, &tb)?.rsub_scalar(1.0)?.mul(&t)?)?.div_scalar(tss)?;

        // DFL: cross-entropy against the two neighbouring bins of each side's target distance.
        let bins = on_dev(Tensor::from_vec(bins, [n, 4, 2])?)?;
        let bin_w = on_dev(Tensor::from_vec(bin_w, [n, 4, 2])?)?;
        let picked = logits.log_softmax(2)?.gather(2, &bins)?;
        let l_dfl = sum_fixed_order(&picked.mul(&bin_w)?)?.neg()?.div_scalar(tss)?;
        (l_box, l_dfl)
    };
    let l_cls = cls_sum.div_scalar(tss)?;

    let (wb, wc, wd) =
        (l_box.mul_scalar(cfg.box_gain)?, l_cls.mul_scalar(cfg.cls_gain)?, l_dfl.mul_scalar(cfg.dfl_gain)?);
    let items = LossItems { box_: wb.item::<f32>()?, cls: wc.item::<f32>()?, dfl: wd.item::<f32>()? };
    let total = wb.add(&wc)?.add(&wd)?.mul_scalar(b as f32)?;
    let total_value = total.item::<f32>()?;
    Ok(LossOutput { total, total_value, items, num_fg: n, target_score_sum: tss_raw as f32 })
}
