//! The differentiable terms of §7.4 for a fixed assignment ([`loss_with_assignment`]): BCE, CIoU ([`ciou`]) and
//! DFL, with the fused CPU custom ops (BCE sum, foreground gather) and fixed-order sums they are built from.

use folo_core::cpu::fast_exp;
use folo_core::{DType, Result, Tensor, no_grad};
use rayon::prelude::*;

use super::{BatchAssignment, BoxLossKind, LossConfig, LossItems, LossOutput, check_inputs, invalid};
use crate::assign::{bbox2dist_clamped, dfl_bins, nwd_lambda};
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
fn mal_neg_loss(x: f32, gamma: f32) -> f32 {
    let p = sigmoid_fast(x);
    let sp = softplus(x);
    if p <= 0.0 { 0.0 } else { p.powf(gamma) * sp }
}

fn mal_neg_grad(x: f32, gamma: f32) -> f32 {
    let p = sigmoid_fast(x);
    if p <= 0.0 {
        return 0.0;
    }
    let q = sigmoid_fast(-x);
    let sp = softplus(x);
    p.powf(gamma) * (gamma * q * sp + p)
}

fn bce_sum_cpu(x: &Tensor, idx: Vec<usize>, t: Vec<f32>, mal_gamma: f32) -> Result<Tensor> {
    if mal_gamma > 0.0 {
        return mal_sum_cpu(x, idx, t, mal_gamma);
    }
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

/// Matchability-Aware Loss (MAL) sum on CPU: positives (`t > 0`) use unweighted `softplus(x) - x·t`,
/// negatives (`t == 0`) use `p^γ · softplus(x)` with exact gradient `p^γ · (γ · (1-p) · softplus(x) + p)`.
fn mal_sum_cpu(x: &Tensor, idx: Vec<usize>, t: Vec<f32>, gamma: f32) -> Result<Tensor> {
    let row = x.dims().last().copied().unwrap_or(1).max(1);
    let mut order: Vec<usize> = (0..idx.len()).filter(|&k| t[k] > 0.0).collect();
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
                        for &xv in &r[from..p - start] {
                            s += mal_neg_loss(xv, gamma);
                        }
                        from = p - start + 1;
                    }
                    for &xv in &r[from..] {
                        s += mal_neg_loss(xv, gamma);
                    }
                    s
                })
            })
            .collect();
        let dense: f64 = rows.iter().map(|&r| r as f64).sum();
        let fg: f64 =
            order.iter().map(|&k| softplus(v[idx[k]]) as f64 - v[idx[k]] as f64 * t[k] as f64).sum();
        (dense + fg) as f32
    })?;
    let xs = x.detach();
    Tensor::custom_op("mal_with_logits_sum", &[x], Tensor::scalar(value), move |g| {
        let g = g.item::<f32>()?;
        let mut grad = xs.with_f32_slice(|v| {
            let mut out = vec![0f32; v.len()];
            out.par_chunks_mut(row * PAR_ROWS).zip(v.par_chunks(row * PAR_ROWS)).for_each(|(o, x)| {
                for (o, &xv) in o.iter_mut().zip(x) {
                    *o = g * mal_neg_grad(xv, gamma);
                }
            });
            out
        })?;
        xs.with_f32_slice(|v| {
            for (&i, &tk) in idx.iter().zip(&t) {
                if tk > 0.0 {
                    grad[i] = g * (sigmoid_fast(v[i]) - tk);
                }
            }
        })?;
        Ok(vec![Some(Tensor::from_vec(grad, xs.shape())?)])
    })
}

/// Differentiable Normalized Wasserstein Distance loss `1 − NWD(b1, b2; c)` between rows of `b1` and `b2`
/// (`[N, 4]`, `xyxy`), with per-row scale constant `c` (`[N]`, in the boxes' coordinate units), returning `[N]`.
pub fn nwd_loss(b1: &Tensor, b2: &Tensor, c: &Tensor) -> Result<Tensor> {
    let col = |b: &Tensor, k: usize| b.narrow(1, k, 1)?.squeeze(1);
    let (ax1, ay1, ax2, ay2) = (col(b1, 0)?, col(b1, 1)?, col(b1, 2)?, col(b1, 3)?);
    let (bx1, by1, bx2, by2) = (col(b2, 0)?, col(b2, 1)?, col(b2, 2)?, col(b2, 3)?);
    let dcx = ax1.add(&ax2)?.sub(&bx1)?.sub(&bx2)?.mul_scalar(0.5)?;
    let dcy = ay1.add(&ay2)?.sub(&by1)?.sub(&by2)?.mul_scalar(0.5)?;
    let dw = ax2.sub(&ax1)?.sub(&bx2.sub(&bx1)?)?;
    let dh = ay2.sub(&ay1)?.sub(&by2.sub(&by1)?)?;
    let w2_sq =
        dcx.square()?.add(&dcy.square()?)?.add(&dw.square()?.add(&dh.square()?)?.mul_scalar(0.25)?)?;
    let dist = w2_sq.add_scalar(1e-6)?.sqrt()?;
    dist.div(c)?.neg()?.exp()?.rsub_scalar(1.0)
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

/// Differentiable Rational-Focal Edge-Decoupled IoU loss (`R-PIoU`) between rows of `b1` (predicted) and `b2`
/// (ground truth), both `[N, 4]` `xyxy` in grid units, returning `[N]`:
///
/// `L = (1 - u)(1 + beta * u) + kappa * E^2 / (1 + kappa * E^2)`
///
/// where `u = IoU(b1, b2)` and
/// `E^2 = 0.25 * (((ax1 - bx1)^2 + (ax2 - bx2)^2) / (w_g^2 + delta0^2) + ((ay1 - by1)^2 + (ay2 - by2)^2) / (h_g^2 + delta0^2))`.
/// Purely algebraic (no `atan`, `exp`, `sqrt` or host round-trip), with zero enclosing-box expansion and a global
/// gradient bound.
pub fn rpiou_loss(b1: &Tensor, b2: &Tensor, beta: f32, kappa: f32, delta0: f32) -> Result<Tensor> {
    const EPS: f32 = 1e-7;
    const KE2_MAX: f32 = 1e16;
    let col = |b: &Tensor, k: usize| b.narrow(1, k, 1)?.squeeze(1);
    let (ax1, ay1, ax2, ay2) = (col(b1, 0)?, col(b1, 1)?, col(b1, 2)?, col(b1, 3)?);
    let (bx1, by1, bx2, by2) = (col(b2, 0)?, col(b2, 1)?, col(b2, 2)?, col(b2, 3)?);
    let (w1, h1) = (ax2.sub(&ax1)?.relu()?, ay2.sub(&ay1)?.relu()?);
    let (w2, h2) = (bx2.sub(&bx1)?.relu()?, by2.sub(&by1)?.relu()?);
    let iw = ax2.minimum(&bx2)?.sub(&ax1.maximum(&bx1)?)?.relu()?;
    let ih = ay2.minimum(&by2)?.sub(&ay1.maximum(&by1)?)?.relu()?;
    let inter = iw.mul(&ih)?;
    let union = w1.mul(&h1)?.add(&w2.mul(&h2)?)?.sub(&inter)?.add_scalar(EPS)?;
    let iou = inter.div(&union)?;
    let focal_iou = iou.rsub_scalar(1.0)?.mul(&iou.mul_scalar(beta)?.add_scalar(1.0)?)?;
    // GT denominators are constants w.r.t. the prediction so predicted sizes never enter the denominator.
    let (denom_x, denom_y) = {
        let _g = no_grad();
        let d2 = delta0 * delta0;
        (w2.square()?.add_scalar(d2)?, h2.square()?.add_scalar(d2)?)
    };
    let ex = ax1.sub(&bx1)?.square()?.add(&ax2.sub(&bx2)?.square()?)?.div(&denom_x)?;
    let ey = ay1.sub(&by1)?.square()?.add(&ay2.sub(&by2)?.square()?)?.div(&denom_y)?;
    let ke2_raw = ex.add(&ey)?.mul_scalar(0.25 * kappa)?;
    let cap = Tensor::full_with(ke2_raw.shape(), KE2_MAX as f64, DType::F32, ke2_raw.device())?;
    let ke2 = ke2_raw.minimum(&cap)?;
    let edge_pen = ke2.div(&ke2.add_scalar(1.0)?)?;
    focal_iou.add(&edge_pen)
}

/// Numerically stable `ln(exp(a) + exp(b))` matching `detect_common.wgsl::lse2`.
#[inline(always)]
fn lse2(a: f32, b: f32) -> f32 {
    a.max(b) + fast_exp(-(a - b).abs()).ln_1p()
}

/// Exact `O(R)` log-space 1D optimal-transport cumulative tail-barrier DFL term:
/// `ot_weight * Σ_{n, side} side_wt * (Σ_{k < j} max(-log S_k, 0) + Σ_{k > j} max(-log F_k, 0))`
/// for `logits: [N, 4, R]`, with exact overflow/underflow-free `O(R)` backward.
fn ot_dfl_tail(logits: &Tensor, left_bins: &[usize], side_wt: &[f32], ot_weight: f32) -> Result<Tensor> {
    let (n, sides, r) = logits.dims3()?;
    debug_assert_eq!(sides, 4);
    let dev = logits.device();
    if r <= 2 || ot_weight <= 0.0 || n == 0 {
        return Tensor::zeros_with([], DType::F32, dev);
    }
    let data = logits.to_vec::<f32>()?;
    let mut grad_unit = vec![0.0f32; n * 4 * r];
    let mut total = 0.0f64;
    let r_m2 = (r - 2) as f32;
    let mut p = vec![0.0f32; r];
    let mut lp = vec![0.0f32; r];
    let mut m_suf = vec![0.0f32; r];
    let mut m_pre = vec![0.0f32; r];
    for idx in 0..n * 4 {
        let wt = side_wt[idx];
        if wt == 0.0 {
            continue;
        }
        let j = left_bins[idx];
        let row = &data[idx * r..(idx + 1) * r];
        let mut m = f32::NEG_INFINITY;
        for &x in row {
            m = m.max(x);
        }
        let mut z = 0.0f32;
        for ((pk, lpk), &x) in p.iter_mut().zip(&mut lp).zip(row) {
            let xm = x - m;
            *lpk = xm;
            let e = fast_exp(xm);
            *pk = e;
            z += e;
        }
        let lz = z.ln();
        for (pk, lpk) in p.iter_mut().zip(&mut lp) {
            *pk /= z;
            *lpk -= lz;
        }
        let mut l_tail = 0.0f32;
        if j > 0 {
            let mut acc_suf = lp[r - 1];
            for step in 0..r - 1 {
                let k = (r - 2) - step;
                if step > 0 {
                    acc_suf = lse2(acc_suf, lp[k + 1]);
                }
                if k < j {
                    l_tail += (-acc_suf).max(0.0);
                    m_suf[k] = -acc_suf;
                }
            }
            for i in 1..j {
                m_suf[i] = lse2(m_suf[i - 1], m_suf[i]);
            }
        }
        if j + 2 < r {
            let mut acc_pre = lp[0];
            for k in 1..r - 1 {
                acc_pre = lse2(acc_pre, lp[k]);
                if k > j {
                    l_tail += (-acc_pre).max(0.0);
                    m_pre[k] = -acc_pre;
                }
            }
            let mut i = r - 2;
            while i > j + 1 {
                i -= 1;
                m_pre[i] = lse2(m_pre[i + 1], m_pre[i]);
            }
        }
        let scale = ot_weight * wt;
        total += (scale * l_tail) as f64;
        let out_row = &mut grad_unit[idx * r..(idx + 1) * r];
        for (k, out_k) in out_row.iter_mut().enumerate() {
            let lpk = lp[k];
            let bm = if k > 0 && j > 0 { fast_exp(lpk + m_suf[(k - 1).min(j - 1)]) } else { 0.0 };
            let am = if k + 1 < r && j + 2 < r { fast_exp(lpk + m_pre[k.max(j + 1)]) } else { 0.0 };
            *out_k = scale * (r_m2 * p[k] - bm - am);
        }
    }
    let out = Tensor::scalar(total as f32).to_device(dev)?;
    let shape = [n, 4, r];
    Tensor::custom_op("ot_dfl_tail", &[logits], out, move |g| {
        let gv = g.item::<f32>()?;
        let scaled: Vec<f32> = grad_unit.iter().map(|&u| u * gv).collect();
        Ok(vec![Some(Tensor::from_vec(scaled, shape)?.to_device(dev)?)])
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
    let mut left_bins = Vec::with_capacity(n * 4);
    let mut side_wt = Vec::with_capacity(n * 4);
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
            left_bins.push(j);
            side_wt.push(wt);
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
    let mal_gamma = cfg.mal_gamma;
    // Evaluates `sigmoid(x)^gamma * softplus(x)` as `exp(-gamma * softplus(-x)) * softplus(x)` so fractional
    // `0 < gamma < 1` never hits `0^(gamma - 1) = inf` in `powf` backward when `sigmoid(x)` underflows to `0`.
    let mal_neg_composite = |x: &Tensor| -> Result<Tensor> {
        if mal_gamma > 0.0 {
            let log_p = x.neg()?.softplus()?;
            Ok(log_p.mul_scalar(-mal_gamma)?.exp()?.mul(&x.softplus()?)?)
        } else {
            Ok(x.softplus()?)
        }
    };
    let cls_sum = if fused {
        bce_sum_cpu(cls_raw, idx_cls, asg.score.clone(), mal_gamma)?
    } else if n == 0 {
        let neg_term = mal_neg_composite(cls_raw)?;
        sum_fixed_order(&neg_term.sum(2, false)?)?
    } else {
        // Only t > 0 positions are positives under MAL (t == 0 stays negative, matching GPU).
        let (fg_idx, fg_t): (Vec<usize>, Vec<f32>) = if mal_gamma > 0.0 {
            idx_cls.into_iter().zip(asg.score.iter().copied()).filter(|&(_, s)| s > 0.0).unzip()
        } else {
            (idx_cls, asg.score.clone())
        };
        let n_pos = fg_idx.len();
        if n_pos == 0 {
            let neg_term = mal_neg_composite(cls_raw)?;
            sum_fixed_order(&neg_term.sum(2, false)?)?
        } else {
            let idx = index(fg_idx)?;
            let t_pos = on_dev(Tensor::from_slice(&fg_t, [n_pos])?)?;
            let x_fg = cls_raw.flatten_all()?.index_select(0, &idx)?;
            let bg_mask = {
                let _g = no_grad();
                let ones = Tensor::full_with([b * nc * n_a], 1.0, DType::F32, dev)?;
                ones.index_add(0, &idx, &Tensor::full_with([n_pos], -1.0, DType::F32, dev)?)?
                    .reshape([b, nc, n_a])?
            };
            let neg_base = mal_neg_composite(cls_raw)?;
            let dense = sum_fixed_order(&neg_base.mul(&bg_mask)?.sum(2, false)?)?;
            dense.add(&sum_fixed_order(&x_fg.softplus()?.sub(&x_fg.mul(&t_pos)?)?)?)?
        }
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

        // Box: DFL expectation → grid-unit xyxy → CIoU or R-PIoU against the target.
        let proj = on_dev(Tensor::arange(0.0, r as f32, 1.0)?)?;
        let dist = logits.softmax(2)?.mul(&proj)?.sum(2, false)?;
        let sign = on_dev(Tensor::from_slice(&[-1.0f32, -1.0, 1.0, 1.0], [4])?)?;
        let pd = on_dev(Tensor::from_vec(anchor4, [n, 4])?)?.add(&dist.mul(&sign)?)?;
        let tb = on_dev(Tensor::from_vec(tbox, [n, 4])?)?;
        let l_base = match cfg.box_loss {
            BoxLossKind::Ciou => ciou(&pd, &tb)?.rsub_scalar(1.0)?,
            BoxLossKind::Rpiou => rpiou_loss(&pd, &tb, cfg.rpiou_beta, cfg.rpiou_kappa, cfg.rpiou_delta)?,
        };
        let box_per = if cfg.nwd_weight > 0.0 {
            let lams: Vec<f32> = (0..n).map(|k| nwd_lambda(&asg.target[k], cfg.nwd_weight)).collect();
            let cs: Vec<f32> = (0..n).map(|k| 12.8 / anchors.strides[asg.anchor[k] as usize]).collect();
            let lam_t = on_dev(Tensor::from_vec(lams, [n])?)?;
            let c_t = on_dev(Tensor::from_vec(cs, [n])?)?;
            let l_nwd = nwd_loss(&pd, &tb, &c_t)?;
            lam_t.rsub_scalar(1.0)?.mul(&l_base)?.add(&lam_t.mul(&l_nwd)?)?
        } else {
            l_base
        };
        let l_box = sum_fixed_order(&box_per.mul(&t)?)?.div_scalar(tss)?;

        // DFL: cross-entropy against the two neighbouring bins, plus optional OT cumulative tail barrier.
        let bins = on_dev(Tensor::from_vec(bins, [n, 4, 2])?)?;
        let bin_w = on_dev(Tensor::from_vec(bin_w, [n, 4, 2])?)?;
        let picked = logits.log_softmax(2)?.gather(2, &bins)?;
        let l_dfl_ce = sum_fixed_order(&picked.mul(&bin_w)?)?.neg()?;
        let l_dfl = if cfg.dfl_ot_weight > 0.0 && r > 2 {
            l_dfl_ce.add(&ot_dfl_tail(&logits, &left_bins, &side_wt, cfg.dfl_ot_weight)?)?.div_scalar(tss)?
        } else {
            l_dfl_ce.div_scalar(tss)?
        };
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
