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

/// Single-box `R-PIoU` forward `(loss, dL/d(ax1, ay1, ax2, ay2))` in `f64`, matching `det_box.wgsl` tie-breaking.
fn rpiou_box_f64(a: [f64; 4], b: [f64; 4], beta: f64, kappa: f64, delta0: f64) -> (f64, [f64; 4]) {
    const EPS: f64 = 1e-7;
    const KE2_MAX: f64 = 1e16;
    let w_min = |x: f64, y: f64| {
        if x < y {
            1.0
        } else if x == y {
            0.5
        } else {
            0.0
        }
    };
    let w_max = |x: f64, y: f64| {
        if x > y {
            1.0
        } else if x == y {
            0.5
        } else {
            0.0
        }
    };
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let w1 = (ax2 - ax1).max(0.0);
    let h1 = (ay2 - ay1).max(0.0);
    let w2 = (bx2 - bx1).max(0.0);
    let h2 = (by2 - by1).max(0.0);
    let ix = ax2.min(bx2) - ax1.max(bx1);
    let iy = ay2.min(by2) - ay1.max(by1);
    let iw = ix.max(0.0);
    let ih = iy.max(0.0);
    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let inter = iw * ih;
    let uni = area1 + area2 - inter + EPS;
    let iou = inter / uni;
    let deficit = (area1 - inter).max(0.0) + (area2 - inter).max(0.0) + EPS;
    let one_minus_iou = deficit / uni;
    let focal = one_minus_iou * (1.0 + beta * iou);
    let g_u = -(1.0 - beta + 2.0 * beta * iou);
    let g_scale = g_u / uni;
    let h1_act = if ax2 > ax1 { h1 } else { 0.0 };
    let w1_act = if ay2 > ay1 { w1 } else { 0.0 };
    let c_x1 = if ix > 0.0 { w_max(ax1, bx1) } else { 0.0 };
    let c_x2 = if ix > 0.0 { w_min(ax2, bx2) } else { 0.0 };
    let c_y1 = if iy > 0.0 { w_max(ay1, by1) } else { 0.0 };
    let c_y2 = if iy > 0.0 { w_min(ay2, by2) } else { 0.0 };
    let mut g = [
        -g_scale * ((inter * (2.0 * c_x1 * ih - h1_act) + deficit * (c_x1 * ih)) / uni),
        -g_scale * ((inter * (2.0 * c_y1 * iw - w1_act) + deficit * (c_y1 * iw)) / uni),
        g_scale * ((inter * (2.0 * c_x2 * ih - h1_act) + deficit * (c_x2 * ih)) / uni),
        g_scale * ((inter * (2.0 * c_y2 * iw - w1_act) + deficit * (c_y2 * iw)) / uni),
    ];
    let d2 = delta0 * delta0;
    let denom_x = w2 * w2 + d2;
    let denom_y = h2 * h2 + d2;
    let (dx1, dx2, dy1, dy2) = (ax1 - bx1, ax2 - bx2, ay1 - by1, ay2 - by2);
    let ke2_raw = kappa * 0.25 * ((dx1 * dx1 + dx2 * dx2) / denom_x + (dy1 * dy1 + dy2 * dy2) / denom_y);
    let edge = if ke2_raw.is_finite() {
        let ke2 = ke2_raw.min(KE2_MAX);
        ke2 / (1.0 + ke2)
    } else if ke2_raw.is_infinite() {
        1.0
    } else {
        ke2_raw
    };
    if ke2_raw.is_finite() && ke2_raw < KE2_MAX {
        let inv1 = 1.0 / (1.0 + ke2_raw);
        let c = 0.5 * kappa * inv1 * inv1;
        if dx1 != 0.0 {
            g[0] += c * (dx1 / denom_x);
        }
        if dy1 != 0.0 {
            g[1] += c * (dy1 / denom_y);
        }
        if dx2 != 0.0 {
            g[2] += c * (dx2 / denom_x);
        }
        if dy2 != 0.0 {
            g[3] += c * (dy2 / denom_y);
        }
    }
    (focal + edge, g)
}

/// Differentiable Rational-Focal Edge-Decoupled IoU loss (`R-PIoU`) between rows of `b1` (predicted) and `b2`
/// (ground truth), both `[N, 4]` `xyxy` in grid units, returning `[N]`:
///
/// `L = (1 - u)(1 + beta * u) + kappa * E^2 / (1 + kappa * E^2)`
///
/// where `u = IoU(b1, b2)` and
/// `E^2 = 0.25 * (((ax1 - bx1)^2 + (ax2 - bx2)^2) / (w_g^2 + delta0^2) + ((ay1 - by1)^2 + (ay2 - by2)^2) / (h_g^2 + delta0^2))`.
/// Purely algebraic (no `atan` or `exp`), with zero enclosing-box expansion and a global gradient bound; `b2` is
/// treated as constant ground truth (evaluated in `f64` on CPU; GPU training uses the fused `det_box.wgsl` kernel).
pub fn rpiou_loss(b1: &Tensor, b2: &Tensor, beta: f32, kappa: f32, delta0: f32) -> Result<Tensor> {
    let (n, c1) = b1.dims2()?;
    let (n2, c2) = b2.dims2()?;
    if c1 != 4 || c2 != 4 || n != n2 {
        return Err(invalid(format!(
            "rpiou_loss expects matching [N, 4] box tensors, got {:?} and {:?}",
            b1.dims(),
            b2.dims()
        )));
    }
    if b1.device() != b2.device() {
        return Err(folo_core::Error::DeviceMismatch {
            op: "rpiou_loss",
            lhs: b1.device(),
            rhs: b2.device(),
        });
    }
    let dev = b1.device();
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let (b64, k64, d64) = (beta as f64, kappa as f64, delta0 as f64);
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv) = rpiou_box_f64(ai, bi, b64, k64, d64);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("rpiou_loss", &[b1], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let mut grad = vec![0.0f32; 4 * n];
        for i in 0..n {
            let gi = gv[i] as f64;
            if gi == 0.0 {
                continue;
            }
            for s in 0..4 {
                grad[4 * i + s] = (gi * g_box[i][s]) as f32;
            }
        }
        Ok(vec![Some(Tensor::from_vec(grad, [n, 4])?.to_device(dev)?)])
    })
}

/// Weighted `R-PIoU` scalar `(box_gain / tss) * Σ_i t_i * (1 - lam_i) * L_rpiou_i` directly from DFL
/// `logits: [N, 4, R]`, reusing the tensor-decoded `pd` so focal-IoU and edge terms evaluate the exact same
/// predicted box, and computing the DFL softmax Jacobian centered on the dominant bin `k*` in `f64` alongside
/// batch scaling, `box_gain`, `tss`, `t`, and `(1 - lam)` before rounding the final logit gradient to `f32`.
#[allow(clippy::too_many_arguments)]
fn rpiou_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
    beta: f32,
    kappa: f32,
    delta0: f32,
) -> Result<Tensor> {
    let (n, sides, r) = logits.dims3()?;
    debug_assert_eq!(sides, 4);
    let dev = logits.device();
    if n == 0 {
        return Tensor::zeros_with([], DType::F32, dev);
    }
    let raw_logits = logits.to_vec::<f32>()?;
    let pds = pd_t.to_vec::<f32>()?;
    let (b64, k64, d64) = (beta as f64, kappa as f64, delta0 as f64);
    let mut sum_w = 0.0f64;
    let mut base_d = vec![0.0f64; n * 4];
    let mut weights = vec![0.0f64; n];
    for i in 0..n {
        let ai = [pds[4 * i] as f64, pds[4 * i + 1] as f64, pds[4 * i + 2] as f64, pds[4 * i + 3] as f64];
        let bi = [tbox[4 * i] as f64, tbox[4 * i + 1] as f64, tbox[4 * i + 2] as f64, tbox[4 * i + 3] as f64];
        let (lv, g_xyxy) = rpiou_box_f64(ai, bi, b64, k64, d64);
        let wi = (t_scores[i] as f64) * lams.map_or(1.0, |v| 1.0 - v[i] as f64);
        weights[i] = wi;
        sum_w += wi * lv;
        base_d[4 * i] = -g_xyxy[0];
        base_d[4 * i + 1] = -g_xyxy[1];
        base_d[4 * i + 2] = g_xyxy[2];
        base_d[4 * i + 3] = g_xyxy[3];
    }
    let out_val = ((sum_w / tss as f64) * box_gain as f64) as f32;
    let out_t = Tensor::scalar(out_val).to_device(dev)?;
    Tensor::custom_op("rpiou_from_logits", &[logits], out_t, move |g| {
        let gv = g.item::<f32>()? as f64;
        let scale = gv * (box_gain as f64 / tss as f64);
        let mut grad = vec![0.0f32; n * 4 * r];
        if scale != 0.0 {
            let mut p_row = vec![0.0f64; r];
            for i in 0..n {
                let gi = scale * weights[i];
                if gi == 0.0 {
                    continue;
                }
                for side in 0..4 {
                    let gd = gi * base_d[4 * i + side];
                    if gd == 0.0 {
                        continue;
                    }
                    let row = (4 * i + side) * r..(4 * i + side + 1) * r;
                    let lrow = &raw_logits[row.clone()];
                    let grow = &mut grad[row];
                    let (k_star, m_side) =
                        lrow.iter().enumerate().fold((0usize, lrow[0]), |(bi, bm), (k, &x)| {
                            if k > 0 && x > bm { (k, x) } else { (bi, bm) }
                        });
                    let mut z_side = 0.0f64;
                    for (k, &x) in lrow.iter().enumerate() {
                        let e = exp_neg_f64(x, m_side);
                        p_row[k] = e;
                        z_side += e;
                    }
                    for pk in p_row.iter_mut() {
                        *pk /= z_side;
                    }
                    let mut delta = 0.0f64;
                    for m in 1..r {
                        let pr = if k_star + m < r { p_row[k_star + m] } else { 0.0 };
                        let pl = if m <= k_star { p_row[k_star - m] } else { 0.0 };
                        delta += (m as f64) * (pr - pl);
                    }
                    for (k, (&pk, gk)) in p_row.iter().zip(grow.iter_mut()).enumerate() {
                        let k_minus_d =
                            if k == k_star { -delta } else { (k as isize - k_star as isize) as f64 - delta };
                        *gk = (gd * pk * k_minus_d) as f32;
                    }
                }
            }
        }
        Ok(vec![Some(Tensor::from_vec(grad, [n, 4, r])?.to_device(dev)?)])
    })
}

/// Overflow/underflow-safe `exp(a - b)` in `f64` for `a <= b` down to `-700.0`.
#[inline(always)]
fn exp_neg_f64(a: f32, b: f32) -> f64 {
    let d = a as f64 - b as f64;
    if d > -700.0 { d.exp() } else { 0.0 }
}

/// Exact `O(R)` shifted `(max, sum_exp)` 1D optimal-transport cumulative tail-barrier DFL term:
/// `(dfl_gain * ot_weight / tss) * Σ_{n, side} side_wt * Σ_{k != j} softplus(log Z_excl(k) - log Z_ret(k))`
/// for `logits: [N, 4, R]`, accumulating retained and excluded masses separately in `f64` so tiny excluded masses
/// (e.g. `[-89, 0, 0, -1000]`), extreme logits (`[-1e38; 5, 0]`, `[-1e8, -1e8, -1e8, 0]`), and multi-positive sums
/// never suffer cancellation or premature overflow.
fn ot_dfl_tail(
    logits: &Tensor,
    left_bins: &[usize],
    side_wt: &[f32],
    ot_weight: f32,
    tss: f32,
    dfl_gain: f32,
) -> Result<Tensor> {
    let (n, sides, r) = logits.dims3()?;
    debug_assert_eq!(sides, 4);
    let dev = logits.device();
    if r <= 2 || ot_weight <= 0.0 || n == 0 {
        return Tensor::zeros_with([], DType::F32, dev);
    }
    let softplus_sig = |m_e: f32, a_e: f64, m_r: f32, a_r: f64| -> (f64, f64) {
        let delta = (m_e as f64 - m_r as f64) + (a_e.ln() - a_r.ln());
        if delta >= 0.0 {
            let e = (-delta).exp();
            (delta + e.ln_1p(), 1.0 / (1.0 + e))
        } else if delta > -700.0 {
            let e = delta.exp();
            (e.ln_1p(), e / (1.0 + e))
        } else {
            (0.0, 0.0)
        }
    };
    let data = logits.to_vec::<f32>()?;
    let mut grad_unit = vec![0.0f64; n * 4 * r];
    let mut total = 0.0f64;
    let mut pref_m = vec![0.0f32; r];
    let mut pref_a = vec![0.0f64; r];
    let mut m_ret = vec![0.0f32; r];
    let mut ret_mant = vec![0.0f64; r];
    let mut m_exc = vec![0.0f32; r];
    let mut exc_mant = vec![0.0f64; r];
    for idx in 0..n * 4 {
        let wt = side_wt[idx] as f64;
        if wt == 0.0 {
            continue;
        }
        let j = left_bins[idx];
        let row = &data[idx * r..(idx + 1) * r];
        let mut pm = row[0];
        let mut pa = 1.0f64;
        pref_m[0] = pm;
        pref_a[0] = pa;
        for k in 1..r - 1 {
            let x = row[k];
            if x > pm {
                pa = pa * exp_neg_f64(pm, x) + 1.0;
                pm = x;
            } else {
                pa += exp_neg_f64(x, pm);
            }
            pref_m[k] = pm;
            pref_a[k] = pa;
        }
        let mut l_tail = 0.0f64;
        let mut sm = row[r - 1];
        let mut sa = 1.0f64;
        for step in 0..r - 1 {
            let k = (r - 2) - step;
            if step > 0 {
                let x = row[k + 1];
                if x > sm {
                    sa = sa * exp_neg_f64(sm, x) + 1.0;
                    sm = x;
                } else {
                    sa += exp_neg_f64(x, sm);
                }
            }
            if k < j {
                let (sp, sig) = softplus_sig(pref_m[k], pref_a[k], sm, sa);
                l_tail += sp;
                m_exc[k] = pref_m[k];
                exc_mant[k] = sig / pref_a[k];
                m_ret[k] = sm;
                ret_mant[k] = sig / sa;
            } else if k > j {
                let (sp, sig) = softplus_sig(sm, sa, pref_m[k], pref_a[k]);
                l_tail += sp;
                m_ret[k] = pref_m[k];
                ret_mant[k] = sig / pref_a[k];
                m_exc[k] = sm;
                exc_mant[k] = sig / sa;
            }
        }
        if j > 0 {
            for t in 1..j {
                ret_mant[t] += ret_mant[t - 1] * exp_neg_f64(m_ret[t], m_ret[t - 1]);
            }
            let mut t = j - 1;
            while t > 0 {
                t -= 1;
                exc_mant[t] += exc_mant[t + 1] * exp_neg_f64(m_exc[t], m_exc[t + 1]);
            }
        }
        if j + 2 < r {
            let mut t = r - 2;
            while t > j + 1 {
                t -= 1;
                ret_mant[t] += ret_mant[t + 1] * exp_neg_f64(m_ret[t], m_ret[t + 1]);
            }
            for t in j + 2..r - 1 {
                exc_mant[t] += exc_mant[t - 1] * exp_neg_f64(m_exc[t], m_exc[t - 1]);
            }
        }
        total += wt * l_tail;
        let out_row = &mut grad_unit[idx * r..(idx + 1) * r];
        for (k, out_k) in out_row.iter_mut().enumerate() {
            let xk = row[k];
            let mut d_ot = 0.0f64;
            if j > 0 {
                if k < j {
                    d_ot += exp_neg_f64(xk, m_exc[k]) * exc_mant[k];
                }
                if k > 0 {
                    let ti = (k - 1).min(j - 1);
                    d_ot -= exp_neg_f64(xk, m_ret[ti]) * ret_mant[ti];
                }
            }
            if j + 2 < r {
                if k + 1 < r {
                    let ti = k.max(j + 1);
                    d_ot -= exp_neg_f64(xk, m_ret[ti]) * ret_mant[ti];
                }
                if k > j + 1 {
                    let ti = (k - 1).min(r - 2);
                    d_ot += exp_neg_f64(xk, m_exc[ti]) * exc_mant[ti];
                }
            }
            *out_k = wt * d_ot;
        }
    }
    let out_val = ((total / tss as f64) * ot_weight as f64 * dfl_gain as f64) as f32;
    let out = Tensor::scalar(out_val).to_device(dev)?;
    let shape = [n, 4, r];
    Tensor::custom_op("ot_dfl_tail", &[logits], out, move |g| {
        let gv = g.item::<f32>()? as f64;
        let scale = gv * (dfl_gain as f64 / tss as f64) * (ot_weight as f64);
        let scaled: Vec<f32> = grad_unit.iter().map(|&u| (u * scale) as f32).collect();
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
        let raw = v.sub(&iou)?.add_scalar(1.0 + EPS)?;
        let eps = raw.zeros_like()?.add_scalar(EPS)?;
        v.div(&raw.maximum(&eps)?)?
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
    let use_ot = cfg.dfl_ot_weight > 0.0 && r > 2;
    let mut left_bins = if use_ot { Vec::with_capacity(n * 4) } else { Vec::new() };
    let mut side_wt = if use_ot { Vec::with_capacity(n * 4) } else { Vec::new() };
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
            if use_ot {
                left_bins.push(j);
                side_wt.push(wt);
            }
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

    let (wb, wd) = if n == 0 {
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
        let probs = logits.softmax(2)?;
        let dist = probs.mul(&proj)?.sum(2, false)?;
        let sign = on_dev(Tensor::from_slice(&[-1.0f32, -1.0, 1.0, 1.0], [4])?)?;
        let pd = on_dev(Tensor::from_vec(anchor4, [n, 4])?)?.add(&dist.mul(&sign)?)?;
        let tb = on_dev(Tensor::from_slice(&tbox, [n, 4])?)?;
        let wb = match cfg.box_loss {
            BoxLossKind::Ciou => {
                let l_base = ciou(&pd, &tb)?.rsub_scalar(1.0)?;
                let box_per = if cfg.nwd_weight > 0.0 {
                    let lams: Vec<f32> = (0..n).map(|k| nwd_lambda(&asg.target[k], cfg.nwd_weight)).collect();
                    let cs: Vec<f32> =
                        (0..n).map(|k| 12.8 / anchors.strides[asg.anchor[k] as usize]).collect();
                    let lam_t = on_dev(Tensor::from_vec(lams, [n])?)?;
                    let c_t = on_dev(Tensor::from_vec(cs, [n])?)?;
                    let l_nwd = nwd_loss(&pd, &tb, &c_t)?;
                    lam_t.rsub_scalar(1.0)?.mul(&l_base)?.add(&lam_t.mul(&l_nwd)?)?
                } else {
                    l_base
                };
                sum_fixed_order(&box_per.mul(&t)?)?.div_scalar(tss)?.mul_scalar(cfg.box_gain)?
            }
            BoxLossKind::Rpiou => {
                if cfg.nwd_weight > 0.0 {
                    let lams: Vec<f32> = (0..n).map(|k| nwd_lambda(&asg.target[k], cfg.nwd_weight)).collect();
                    let wb_rpiou = rpiou_from_logits(
                        &logits,
                        &pd,
                        &tbox,
                        &asg.score,
                        Some(&lams),
                        tss,
                        cfg.box_gain,
                        cfg.rpiou_beta,
                        cfg.rpiou_kappa,
                        cfg.rpiou_delta,
                    )?;
                    let cs: Vec<f32> =
                        (0..n).map(|k| 12.8 / anchors.strides[asg.anchor[k] as usize]).collect();
                    let lam_t = on_dev(Tensor::from_vec(lams, [n])?)?;
                    let c_t = on_dev(Tensor::from_vec(cs, [n])?)?;
                    let l_nwd = nwd_loss(&pd, &tb, &c_t)?;
                    let wb_nwd = sum_fixed_order(&lam_t.mul(&l_nwd)?.mul(&t)?)?
                        .div_scalar(tss)?
                        .mul_scalar(cfg.box_gain)?;
                    wb_rpiou.add(&wb_nwd)?
                } else {
                    rpiou_from_logits(
                        &logits,
                        &pd,
                        &tbox,
                        &asg.score,
                        None,
                        tss,
                        cfg.box_gain,
                        cfg.rpiou_beta,
                        cfg.rpiou_kappa,
                        cfg.rpiou_delta,
                    )?
                }
            }
        };

        // DFL: cross-entropy against the two neighbouring bins, plus optional OT cumulative tail barrier.
        let bins = on_dev(Tensor::from_vec(bins, [n, 4, 2])?)?;
        let bin_w = on_dev(Tensor::from_vec(bin_w, [n, 4, 2])?)?;
        let picked = logits.log_softmax(2)?.gather(2, &bins)?;
        let l_dfl_ce = sum_fixed_order(&picked.mul(&bin_w)?)?.neg()?.div_scalar(tss)?;
        let wd = if cfg.dfl_ot_weight > 0.0 && r > 2 {
            l_dfl_ce.mul_scalar(cfg.dfl_gain)?.add(&ot_dfl_tail(
                &logits,
                &left_bins,
                &side_wt,
                cfg.dfl_ot_weight,
                tss,
                cfg.dfl_gain,
            )?)?
        } else {
            l_dfl_ce.mul_scalar(cfg.dfl_gain)?
        };
        (wb, wd)
    };
    let wc = cls_sum.div_scalar(tss)?.mul_scalar(cfg.cls_gain)?;

    let items = LossItems { box_: wb.item::<f32>()?, cls: wc.item::<f32>()?, dfl: wd.item::<f32>()? };
    let total = wb.add(&wc)?.add(&wd)?.mul_scalar(b as f32)?;
    let total_value = total.item::<f32>()?;
    Ok(LossOutput { total, total_value, items, num_fg: n, target_score_sum: tss_raw as f32 })
}
