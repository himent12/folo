//! Classification loss terms: BCE, Matchability-Aware Loss (MAL), Varifocal Loss (VFL, Zhang et al. 2021),
//! and Quality Focal Loss (QFL, Li et al. 2020), with both fused CPU custom ops and device-generic tensor
//! composites.

use folo_core::cpu::fast_exp;
use folo_core::{DType, Result, Tensor, no_grad};
use rayon::prelude::*;

use super::asl_poly::{
    asl_elem_f64, asymmetric_loss, poly1_bce_elem_f64, poly1_focal_elem_f64, sparse_cls_sum,
};
use super::sum_fixed_order;
use crate::loss::{ClsLossKind, LossConfig};

/// Rows per parallel task of the CPU fused classification sums.
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

/// Negative-anchor focal BCE term `scale * p^gamma * softplus(x)`.
#[inline(always)]
fn focal_neg_loss(x: f32, scale: f32, gamma: f32) -> f32 {
    if gamma == 0.0 {
        return scale * softplus(x);
    }
    let p = sigmoid_fast(x);
    let sp = softplus(x);
    if p <= 0.0 { 0.0 } else { scale * p.powf(gamma) * sp }
}

/// Derivative `d/dx [scale * p^gamma * softplus(x)] = scale * p^gamma * (gamma * (1 - p) * softplus(x) + p)`.
#[inline(always)]
fn focal_neg_grad(x: f32, scale: f32, gamma: f32) -> f32 {
    if gamma == 0.0 {
        return scale * sigmoid_fast(x);
    }
    let p = sigmoid_fast(x);
    if p <= 0.0 {
        return 0.0;
    }
    let q = sigmoid_fast(-x);
    let sp = softplus(x);
    scale * p.powf(gamma) * (gamma * q * sp + p)
}

/// Positive-anchor QFL `(loss, dL/dx)` in `f64` for `|t - p|^beta * (softplus(x) - x * t)` with `beta == 0`
/// or `beta >= 1`. Takes `p_f32` computed by `Tensor::sigmoid()` so the cusp at `p == t` (notably `beta == 1`)
/// matches the device-generic composite path `x.sigmoid() - t` bit-for-bit on every input.
#[inline]
fn qfl_pos_f64(x: f32, p_f32: f32, t: f32, beta: f32) -> (f64, f64) {
    let x64 = x as f64;
    let t64 = t as f64;
    let b64 = beta as f64;
    let sp = x64.max(0.0) + (-x64.abs()).exp().ln_1p();
    let bce = sp - x64 * t64;
    let p = p_f32 as f64;
    let q = 1.0 - p;
    if b64 == 0.0 {
        return (bce, p - t64);
    }
    let diff = p - t64;
    let d = diff.abs();
    if d == 0.0 && bce.is_finite() {
        return (0.0, 0.0);
    }
    let pow_b = if b64 == 2.0 {
        d * d
    } else if b64 == 1.0 {
        d
    } else {
        d.powf(b64)
    };
    let d_pow = if b64 == 2.0 {
        2.0 * diff
    } else if b64 == 1.0 {
        if d == 0.0 { 0.0 } else { diff.signum() }
    } else {
        b64 * d.powf(b64 - 1.0) * diff.signum()
    };
    let loss = pow_b * bce;
    let grad = pow_b * diff + d_pow * p * q * bce;
    (loss, grad)
}

/// `Σ_i softplus(x_i) − Σ_k x[idx_k] · t_k` over a CPU tensor, i.e. `Σ BCE(x, T)` for a target `T` that is `t_k` at
/// the flat positions `idx_k` and 0 elsewhere, with the backward `g · (sigmoid(x) − T)` in one pass.
///
/// This fuses what the composite form does in several full-size passes (softplus, sums, and an `index_select`
/// whose backward materialises a dense zero gradient). `x` is viewed as rows of its last dimension; rows are
/// summed in parallel, each serially, and combined in row order in `f64`, so the result is reproducible.
fn bce_sum_cpu(x: &Tensor, idx: Vec<usize>, t: Vec<f32>) -> Result<Tensor> {
    let row = x.dims().last().copied().unwrap_or(1).max(1);
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

/// Fused CPU sum for focal-negative classification losses:
/// - MAL (`neg_scale = 1`, `neg_gamma = mal_gamma`, unweighted positive `BCE(x, t)`),
/// - VFL (`neg_scale = vfl_alpha`, `neg_gamma = vfl_gamma`, quality-weighted positive `t * BCE(x, t)`),
/// - QFL (`neg_scale = 1`, `neg_gamma = qfl_beta`, positive `|t - p|^beta * BCE(x, t)`).
#[derive(Clone, Copy)]
enum FocalPosRule {
    UnweightedBce,
    VflTargetWeighted,
    Qfl { beta: f32 },
}

fn focal_cls_sum_cpu(
    x: &Tensor,
    idx: Vec<usize>,
    t: Vec<f32>,
    neg_scale: f32,
    neg_gamma: f32,
    pos_rule: FocalPosRule,
    op_name: &'static str,
) -> Result<Tensor> {
    let row = x.dims().last().copied().unwrap_or(1).max(1);
    // Only `t > 0` positions are positives (anchors with `t == 0` stay negative, matching GPU/composite).
    let mut order: Vec<usize> = (0..idx.len()).filter(|&k| t[k] > 0.0).collect();
    order.sort_unstable_by_key(|&k| idx[k]);
    let pos: Vec<usize> = order.iter().map(|&k| idx[k]).collect();
    let pos_t: Vec<f32> = order.iter().map(|&k| t[k]).collect();
    let qfl_p: Vec<f32> = if matches!(pos_rule, FocalPosRule::Qfl { .. }) && !pos.is_empty() {
        let fg_x: Vec<f32> = x.with_f32_slice(|v| pos.iter().map(|&i| v[i]).collect())?;
        let _g = no_grad();
        Tensor::from_vec(fg_x, [pos.len()])?.sigmoid()?.to_vec::<f32>()?
    } else {
        Vec::new()
    };
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
                            s += focal_neg_loss(xv, neg_scale, neg_gamma);
                        }
                        from = p - start + 1;
                    }
                    for &xv in &r[from..] {
                        s += focal_neg_loss(xv, neg_scale, neg_gamma);
                    }
                    s
                })
            })
            .collect();
        let dense: f64 = rows.iter().map(|&r| r as f64).sum();
        let fg: f64 = pos
            .iter()
            .zip(&pos_t)
            .enumerate()
            .map(|(m, (&i, &tk))| {
                let xv = v[i];
                match pos_rule {
                    FocalPosRule::UnweightedBce => softplus(xv) as f64 - xv as f64 * tk as f64,
                    FocalPosRule::VflTargetWeighted => {
                        tk as f64 * (softplus(xv) as f64 - xv as f64 * tk as f64)
                    }
                    FocalPosRule::Qfl { beta } => qfl_pos_f64(xv, qfl_p[m], tk, beta).0,
                }
            })
            .sum();
        (dense + fg) as f32
    })?;
    let xs = x.detach();
    Tensor::custom_op(op_name, &[x], Tensor::scalar(value), move |g| {
        let g = g.item::<f32>()?;
        let mut grad = xs.with_f32_slice(|v| {
            let mut out = vec![0f32; v.len()];
            out.par_chunks_mut(row * PAR_ROWS).zip(v.par_chunks(row * PAR_ROWS)).for_each(|(o, x)| {
                for (o, &xv) in o.iter_mut().zip(x) {
                    *o = g * focal_neg_grad(xv, neg_scale, neg_gamma);
                }
            });
            out
        })?;
        xs.with_f32_slice(|v| {
            for (m, (&i, &tk)) in pos.iter().zip(&pos_t).enumerate() {
                let xv = v[i];
                grad[i] = match pos_rule {
                    FocalPosRule::UnweightedBce => g * (sigmoid_fast(xv) - tk),
                    FocalPosRule::VflTargetWeighted => g * tk * (sigmoid_fast(xv) - tk),
                    FocalPosRule::Qfl { beta } => (g as f64 * qfl_pos_f64(xv, qfl_p[m], tk, beta).1) as f32,
                };
            }
        })?;
        Ok(vec![Some(Tensor::from_vec(grad, xs.shape())?)])
    })
}

/// Compute the unnormalised classification loss sum over `[B, nc, A]` logits using either the fused CPU
/// custom ops or the device-generic composite of `folo-core` tensor ops.
pub(super) fn cls_loss_sum(
    cls_raw: &Tensor,
    idx_cls: Vec<usize>,
    scores: &[f32],
    cfg: &LossConfig,
    fused: bool,
) -> Result<Tensor> {
    let cls_loss = cfg.cls_loss;
    let mal_gamma = cfg.mal_gamma;
    let vfl_alpha = cfg.vfl_alpha;
    let vfl_gamma = cfg.vfl_gamma;
    let qfl_beta = cfg.qfl_beta;

    if fused {
        return match cls_loss {
            ClsLossKind::Bce if mal_gamma <= 0.0 => bce_sum_cpu(cls_raw, idx_cls, scores.to_vec()),
            ClsLossKind::Bce => focal_cls_sum_cpu(
                cls_raw,
                idx_cls,
                scores.to_vec(),
                1.0,
                mal_gamma,
                FocalPosRule::UnweightedBce,
                "mal_with_logits_sum",
            ),
            ClsLossKind::Vfl => focal_cls_sum_cpu(
                cls_raw,
                idx_cls,
                scores.to_vec(),
                vfl_alpha,
                vfl_gamma,
                FocalPosRule::VflTargetWeighted,
                "vfl_with_logits_sum",
            ),
            ClsLossKind::Qfl => focal_cls_sum_cpu(
                cls_raw,
                idx_cls,
                scores.to_vec(),
                1.0,
                qfl_beta,
                FocalPosRule::Qfl { beta: qfl_beta },
                "qfl_with_logits_sum",
            ),
            ClsLossKind::Asl => {
                let (gp, gn, m) = (cfg.asl_gamma_pos as f64, cfg.asl_gamma_neg as f64, cfg.asl_clip as f64);
                sparse_cls_sum(cls_raw, idx_cls, scores.to_vec(), "asl_with_logits_sum", move |x, t| {
                    asl_elem_f64(x, t, gp, gn, m)
                })
            }
            ClsLossKind::Poly1Bce => {
                let eps = cfg.poly1_epsilon as f64;
                sparse_cls_sum(cls_raw, idx_cls, scores.to_vec(), "poly1_bce_with_logits_sum", move |x, t| {
                    poly1_bce_elem_f64(x, t, eps)
                })
            }
            ClsLossKind::Poly1Focal => {
                let (gamma, eps) = (cfg.poly1_focal_gamma as f64, cfg.poly1_epsilon as f64);
                sparse_cls_sum(
                    cls_raw,
                    idx_cls,
                    scores.to_vec(),
                    "poly1_focal_with_logits_sum",
                    move |x, t| poly1_focal_elem_f64(x, t, gamma, eps),
                )
            }
        };
    }

    let (b, nc, n_a) = cls_raw.dims3()?;
    let dev = cls_raw.device();

    // ASL composite path builds dense target tensor and delegates to `asymmetric_loss` so the margin shift
    // clamp `p_m = max(p - m, 0)` avoids `0^(gamma - 1)` backward hazards for fractional `0 < gamma_neg < 1`.
    if cls_loss == ClsLossKind::Asl {
        let dense_t = {
            let _g = no_grad();
            let zeros = Tensor::zeros_with([b * nc * n_a], DType::F32, dev)?;
            if idx_cls.is_empty() {
                zeros.reshape([b, nc, n_a])?
            } else {
                let idx = Tensor::from_vec(
                    idx_cls.into_iter().map(|i| i as i64).collect::<Vec<_>>(),
                    [scores.len()],
                )?
                .to_device(dev)?;
                let vals = Tensor::from_slice(scores, [scores.len()])?.to_device(dev)?;
                zeros.index_add(0, &idx, &vals)?.reshape([b, nc, n_a])?
            }
        };
        let loss = asymmetric_loss(cls_raw, &dense_t, cfg.asl_gamma_pos, cfg.asl_gamma_neg, cfg.asl_clip)?;
        return sum_fixed_order(&loss.sum(2, false)?);
    }

    let (neg_scale, neg_gamma, filter_pos) = match cls_loss {
        ClsLossKind::Bce => (1.0f32, mal_gamma, mal_gamma > 0.0),
        ClsLossKind::Vfl => (vfl_alpha, vfl_gamma, true),
        ClsLossKind::Qfl => (1.0f32, qfl_beta, true),
        ClsLossKind::Poly1Bce => (1.0f32, 0.0, true),
        ClsLossKind::Poly1Focal => (1.0f32, cfg.poly1_focal_gamma, true),
        ClsLossKind::Asl => unreachable!(),
    };
    let poly_eps = cfg.poly1_epsilon;
    // Evaluates `scale * sigmoid(x)^gamma * (softplus(x) + eps * sigmoid(x))` using
    // `exp(-gamma * softplus(-x))` so fractional `0 < gamma < 1` never hits `0^(gamma - 1) = inf` in `powf`.
    let neg_composite = |x: &Tensor| -> Result<Tensor> {
        let inner = match cls_loss {
            ClsLossKind::Poly1Bce | ClsLossKind::Poly1Focal if poly_eps != 0.0 => {
                x.softplus()?.add(&x.sigmoid()?.mul_scalar(poly_eps)?)?
            }
            _ => x.softplus()?,
        };
        let base = if neg_gamma > 0.0 {
            let log_p = x.neg()?.softplus()?;
            log_p.mul_scalar(-neg_gamma)?.exp()?.mul(&inner)?
        } else {
            inner
        };
        if neg_scale == 1.0 { Ok(base) } else { base.mul_scalar(neg_scale) }
    };

    let (fg_idx, fg_t): (Vec<usize>, Vec<f32>) = if filter_pos {
        idx_cls.into_iter().zip(scores.iter().copied()).filter(|&(_, s)| s > 0.0).unzip()
    } else {
        (idx_cls, scores.to_vec())
    };
    let n_pos = fg_idx.len();
    if n_pos == 0 {
        let neg_term = neg_composite(cls_raw)?;
        return sum_fixed_order(&neg_term.sum(2, false)?);
    }

    let idx = Tensor::from_vec(fg_idx.into_iter().map(|i| i as i64).collect::<Vec<_>>(), [n_pos])?
        .to_device(dev)?;
    let t_pos = Tensor::from_slice(&fg_t, [n_pos])?.to_device(dev)?;
    let x_fg = cls_raw.flatten_all()?.index_select(0, &idx)?;
    let bg_mask = {
        let _g = no_grad();
        let ones = Tensor::full_with([b * nc * n_a], 1.0, DType::F32, dev)?;
        ones.index_add(0, &idx, &Tensor::full_with([n_pos], -1.0, DType::F32, dev)?)?.reshape([b, nc, n_a])?
    };
    let neg_base = neg_composite(cls_raw)?;
    let dense = sum_fixed_order(&neg_base.mul(&bg_mask)?.sum(2, false)?)?;
    let pos_terms = match cls_loss {
        ClsLossKind::Bce => x_fg.softplus()?.sub(&x_fg.mul(&t_pos)?)?,
        ClsLossKind::Vfl => x_fg.softplus()?.sub(&x_fg.mul(&t_pos)?)?.mul(&t_pos)?,
        ClsLossKind::Qfl => {
            let bce_fg = x_fg.softplus()?.sub(&x_fg.mul(&t_pos)?)?;
            if qfl_beta == 0.0 {
                bce_fg
            } else {
                let diff = x_fg.sigmoid()?.sub(&t_pos)?;
                let weight = if qfl_beta == 2.0 {
                    diff.square()?
                } else if qfl_beta == 1.0 {
                    diff.abs()?
                } else {
                    diff.abs()?.powf(qfl_beta)?
                };
                weight.mul(&bce_fg)?
            }
        }
        ClsLossKind::Poly1Bce | ClsLossKind::Poly1Focal => {
            let neg_x = x_fg.neg()?;
            let pos_inner = if poly_eps != 0.0 {
                neg_x.softplus()?.add(&neg_x.sigmoid()?.mul_scalar(poly_eps)?)?
            } else {
                neg_x.softplus()?
            };
            let pos_part = if neg_gamma > 0.0 {
                x_fg.softplus()?.mul_scalar(-neg_gamma)?.exp()?.mul(&pos_inner)?
            } else {
                pos_inner
            };
            let neg_part = neg_composite(&x_fg)?;
            t_pos.mul(&pos_part)?.add(&t_pos.rsub_scalar(1.0)?.mul(&neg_part)?)?
        }
        ClsLossKind::Asl => unreachable!(),
    };
    dense.add(&sum_fixed_order(&pos_terms)?)
}
