//! Differentiable Object Keypoint Similarity (OKS) similarity & loss for pose estimation.
//!
//! For each instance `i` with `K` predicted keypoints `(x_hat_k, y_hat_k)`, ground-truth keypoints
//! `(x_k, y_k)`, COCO visibility flags `v_k in {0, 1, 2}` (`0` = unlabeled, `1` = occluded, `2` =
//! visible), object scale area `A_i = s_i^2`, and per-keypoint falloff sigmas `sigma_k > 0`:
//!
//! ```text
//! e_{i,k}    = ((x_hat_k - x_k)^2 + (y_hat_k - y_k)^2) / (8 * (A_i + eps) * sigma_k^2)
//! oks_{i,k}  = exp(-e_{i,k})
//! L_{i,k}    = 1 - exp(-e_{i,k}) = -expm1(-e_{i,k})
//! L_i        = (sum_{k: v_k > 0} L_{i,k}) / |{k: v_k > 0}|   (0 when no keypoints are labeled)
//! ```
//!
//! All terms and analytical gradients are evaluated in `f64` using cancellation-free `-expm1(-e)`
//! and flow cleanly into both `pred_kpts` and `gt_kpts` (when `requires_grad()` is set).

use folo_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::vfl_qfl::LossReduction;
use crate::loss::invalid;

/// Default additive area epsilon for [`OksLossConfig`].
pub const DEFAULT_OKS_EPS: f32 = 1e-9;

/// Configuration for [`oks_loss`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OksLossConfig {
    /// Positive additive epsilon added to object area `A_i` in the denominator (`8 * (A_i + eps) * sigma_k^2`).
    pub eps: f32,
    /// Output reduction across instances (default [`LossReduction::Mean`]).
    ///
    /// - [`LossReduction::None`]: per-instance loss `[N]` (`0.0` for instances with zero labeled keypoints).
    /// - [`LossReduction::Sum`]: scalar sum over all instances.
    /// - [`LossReduction::Mean`]: scalar mean over active instances (those with at least one labeled keypoint;
    ///   `0.0` when no instance has any labeled keypoint).
    pub reduction: LossReduction,
}

impl Default for OksLossConfig {
    fn default() -> Self {
        Self { eps: DEFAULT_OKS_EPS, reduction: LossReduction::Mean }
    }
}

impl OksLossConfig {
    /// Validate `eps` (finite and `>= 0.0`).
    pub fn validate(&self) -> Result<()> {
        if !(self.eps.is_finite() && self.eps >= 0.0) {
            return Err(invalid(format!("oks_loss: eps must be finite and >= 0, got {}", self.eps)));
        }
        Ok(())
    }
}

/// Evaluate one instance's OKS loss `1 - OKS` (or similarity `OKS` when `return_similarity` is true)
/// and exact `f64` partial derivatives w.r.t. predicted `(x, y)` coordinates.
///
/// Returns `(value, num_labeled, d_val_d_pred)` where `d_val_d_gt = -d_val_d_pred`.
pub(crate) fn oks_instance_f64(
    pred: &[[f64; 2]],
    gt: &[[f64; 2]],
    vis: &[f64],
    area: f64,
    sigmas: &[f64],
    eps: f64,
    return_similarity: bool,
) -> (f64, usize, Vec<[f64; 2]>) {
    let k_len = pred.len();
    let mut grad = vec![[0.0f64; 2]; k_len];
    let n_vis = vis.iter().filter(|&&v| v > 0.0).count();
    if n_vis == 0 {
        return (0.0, 0, grad);
    }
    let eff_area = area + eps;
    if !(eff_area.is_finite() && eff_area > 0.0) {
        return (f64::NAN, n_vis, vec![[f64::NAN; 2]; k_len]);
    }
    let inv_n = 1.0 / (n_vis as f64);
    let mut sum = 0.0f64;
    let sign = if return_similarity { -1.0 } else { 1.0 };

    for k in 0..k_len {
        if vis[k] <= 0.0 {
            continue;
        }
        let (px, py) = (pred[k][0], pred[k][1]);
        let (gx, gy) = (gt[k][0], gt[k][1]);
        if !(px.is_finite() && py.is_finite() && gx.is_finite() && gy.is_finite()) {
            return (f64::NAN, n_vis, vec![[f64::NAN; 2]; k_len]);
        }
        let s = sigmas[k];
        let denom = 8.0 * eff_area * s * s;
        if !(denom.is_finite() && denom > 0.0) {
            return (f64::NAN, n_vis, vec![[f64::NAN; 2]; k_len]);
        }
        let dx = px - gx;
        let dy = py - gy;
        let d2 = dx * dx + dy * dy;
        let e = d2 / denom;
        let exp_neg_e = (-e).exp();
        let term = if return_similarity { exp_neg_e } else { -(-e).exp_m1() };
        sum += term;
        if exp_neg_e > 0.0 {
            let coeff = sign * inv_n * exp_neg_e * (2.0 / denom);
            grad[k] = [coeff * dx, coeff * dy];
        }
    }

    (sum * inv_n, n_vis, grad)
}

fn validate_oks_inputs(
    pred_kpts: &Tensor,
    gt_kpts: &Tensor,
    visibility: &Tensor,
    areas: &Tensor,
    sigmas: &[f32],
    eps: f32,
    op: &'static str,
) -> Result<(usize, usize)> {
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(invalid(format!("{op}: eps must be finite and >= 0, got {eps}")));
    }
    for (name, t) in
        [("pred_kpts", pred_kpts), ("gt_kpts", gt_kpts), ("visibility", visibility), ("areas", areas)]
    {
        if t.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch { op, expected: DType::F32, got: t.dtype() });
        }
        if t.device() != pred_kpts.device() {
            return Err(Error::DeviceMismatch { op, lhs: pred_kpts.device(), rhs: t.device() });
        }
        let _ = name;
    }
    let (n, k, xy) = pred_kpts
        .dims3()
        .map_err(|_| invalid(format!("{op}: pred_kpts must be [N, K, 2], got {}", pred_kpts.shape())))?;
    if xy != 2 {
        return Err(invalid(format!("{op}: pred_kpts last dim must be 2, got {xy}")));
    }
    if gt_kpts.shape() != pred_kpts.shape() {
        return Err(invalid(format!(
            "{op}: gt_kpts shape {} != pred_kpts shape {}",
            gt_kpts.shape(),
            pred_kpts.shape()
        )));
    }
    let (vn, vk) = visibility
        .dims2()
        .map_err(|_| invalid(format!("{op}: visibility must be [N, K], got {}", visibility.shape())))?;
    if vn != n || vk != k {
        return Err(invalid(format!("{op}: visibility shape {} != [{n}, {k}]", visibility.shape())));
    }
    let &[an] = areas.dims() else {
        return Err(invalid(format!("{op}: areas must be [N], got {}", areas.shape())));
    };
    if an != n {
        return Err(invalid(format!("{op}: areas len {an} != N ({n})")));
    }
    if sigmas.len() != k {
        return Err(invalid(format!("{op}: sigmas len {} != K ({k})", sigmas.len())));
    }
    for (idx, &s) in sigmas.iter().enumerate() {
        if !(s.is_finite() && s > 0.0) {
            return Err(invalid(format!("{op}: sigmas[{idx}] must be finite and > 0, got {s}")));
        }
    }
    Ok((n, k))
}

fn oks_op(
    pred_kpts: &Tensor,
    gt_kpts: &Tensor,
    visibility: &Tensor,
    areas: &Tensor,
    sigmas: &[f32],
    cfg: &OksLossConfig,
    return_similarity: bool,
) -> Result<Tensor> {
    let op = if return_similarity { "oks_similarity" } else { "oks_loss" };
    let eps = cfg.eps;
    let reduction = cfg.reduction;
    let (n, k) = validate_oks_inputs(pred_kpts, gt_kpts, visibility, areas, sigmas, eps, op)?;
    let dev = pred_kpts.device();
    let need_pred = pred_kpts.requires_grad();
    let need_gt = gt_kpts.requires_grad();

    if n == 0 || k == 0 {
        let out_t = match reduction {
            LossReduction::None => Tensor::zeros_with([n], DType::F32, dev)?,
            LossReduction::Sum | LossReduction::Mean => Tensor::scalar(0.0).to_device(dev)?,
        };
        return Tensor::custom_op(op, &[pred_kpts, gt_kpts], out_t, move |_g| {
            let gp = if need_pred { Some(Tensor::zeros_with([n, k, 2], DType::F32, dev)?) } else { None };
            let gg = if need_gt { Some(Tensor::zeros_with([n, k, 2], DType::F32, dev)?) } else { None };
            Ok(vec![gp, gg])
        });
    }

    let pv = pred_kpts.to_vec::<f32>()?;
    let gv = gt_kpts.to_vec::<f32>()?;
    let vv = visibility.to_vec::<f32>()?;
    let av = areas.to_vec::<f32>()?;

    for (idx, &v) in vv.iter().enumerate() {
        if !(v.is_finite() && (v == 0.0 || v == 1.0 || v == 2.0)) {
            return Err(invalid(format!("{op}: visibility[{idx}] must be 0, 1, or 2, got {v}")));
        }
    }
    for (idx, &a) in av.iter().enumerate() {
        if a.is_finite() && a < 0.0 {
            return Err(invalid(format!("{op}: areas[{idx}] must be >= 0, got {a}")));
        }
    }

    let sigmas64: Vec<f64> = sigmas.iter().map(|&s| s as f64).collect();
    let eps64 = eps as f64;

    let mut per_inst = vec![0.0f64; n];
    let mut active_instances = 0usize;
    let mut grads_p = if need_pred || need_gt { vec![[0.0f64; 2]; n * k] } else { Vec::new() };

    let mut p_buf = vec![[0.0f64; 2]; k];
    let mut g_buf = vec![[0.0f64; 2]; k];
    let mut v_buf = vec![0.0f64; k];

    for i in 0..n {
        let base_k = i * k;
        for j in 0..k {
            let idx2 = (base_k + j) * 2;
            p_buf[j] = [pv[idx2] as f64, pv[idx2 + 1] as f64];
            g_buf[j] = [gv[idx2] as f64, gv[idx2 + 1] as f64];
            v_buf[j] = vv[base_k + j] as f64;
        }
        let (val, n_vis, g_inst) =
            oks_instance_f64(&p_buf, &g_buf, &v_buf, av[i] as f64, &sigmas64, eps64, return_similarity);
        per_inst[i] = val;
        if n_vis > 0 {
            active_instances += 1;
        }
        if need_pred || need_gt {
            grads_p[base_k..base_k + k].copy_from_slice(&g_inst);
        }
    }

    match reduction {
        LossReduction::None => {
            let out: Vec<f32> = per_inst.iter().map(|&v| v as f32).collect();
            let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
            Tensor::custom_op(op, &[pred_kpts, gt_kpts], out_t, move |g| {
                let g_out = g.to_vec::<f32>()?;
                let gp = if need_pred {
                    let mut out_g = vec![0.0f32; n * k * 2];
                    for i in 0..n {
                        let gi = g_out[i] as f64;
                        for j in 0..k {
                            let d = grads_p[i * k + j];
                            if gi == 0.0 && d[0].is_finite() && d[1].is_finite() {
                                continue;
                            }
                            let idx2 = (i * k + j) * 2;
                            out_g[idx2] = (gi * d[0]) as f32;
                            out_g[idx2 + 1] = (gi * d[1]) as f32;
                        }
                    }
                    Some(Tensor::from_vec(out_g, [n, k, 2])?.to_device(dev)?)
                } else {
                    None
                };
                let gg = if need_gt {
                    let mut out_g = vec![0.0f32; n * k * 2];
                    for i in 0..n {
                        let gi = g_out[i] as f64;
                        for j in 0..k {
                            let d = grads_p[i * k + j];
                            if gi == 0.0 && d[0].is_finite() && d[1].is_finite() {
                                continue;
                            }
                            let idx2 = (i * k + j) * 2;
                            out_g[idx2] = (-gi * d[0]) as f32;
                            out_g[idx2 + 1] = (-gi * d[1]) as f32;
                        }
                    }
                    Some(Tensor::from_vec(out_g, [n, k, 2])?.to_device(dev)?)
                } else {
                    None
                };
                Ok(vec![gp, gg])
            })
        }
        LossReduction::Sum | LossReduction::Mean => {
            let scale = match reduction {
                LossReduction::Mean => {
                    if active_instances == 0 {
                        0.0
                    } else {
                        1.0 / (active_instances as f64)
                    }
                }
                LossReduction::Sum => 1.0,
                LossReduction::None => unreachable!(),
            };
            let val = if scale == 0.0 { 0.0f32 } else { (per_inst.iter().sum::<f64>() * scale) as f32 };
            let out_t = Tensor::scalar(val).to_device(dev)?;
            Tensor::custom_op(op, &[pred_kpts, gt_kpts], out_t, move |g| {
                let gs = (g.item::<f32>()? as f64) * scale;
                let gp = if need_pred {
                    let mut out_g = vec![0.0f32; n * k * 2];
                    if scale != 0.0 {
                        for idx in 0..(n * k) {
                            let d = grads_p[idx];
                            if gs == 0.0 && d[0].is_finite() && d[1].is_finite() {
                                continue;
                            }
                            out_g[2 * idx] = (gs * d[0]) as f32;
                            out_g[2 * idx + 1] = (gs * d[1]) as f32;
                        }
                    }
                    Some(Tensor::from_vec(out_g, [n, k, 2])?.to_device(dev)?)
                } else {
                    None
                };
                let gg = if need_gt {
                    let mut out_g = vec![0.0f32; n * k * 2];
                    if scale != 0.0 {
                        for idx in 0..(n * k) {
                            let d = grads_p[idx];
                            if gs == 0.0 && d[0].is_finite() && d[1].is_finite() {
                                continue;
                            }
                            out_g[2 * idx] = (-gs * d[0]) as f32;
                            out_g[2 * idx + 1] = (-gs * d[1]) as f32;
                        }
                    }
                    Some(Tensor::from_vec(out_g, [n, k, 2])?.to_device(dev)?)
                } else {
                    None
                };
                Ok(vec![gp, gg])
            })
        }
    }
}

/// Differentiable per-instance COCO Object Keypoint Similarity `OKS` in `[0, 1]` for `[N, K, 2]`
/// predicted and ground-truth keypoints, `[N, K]` visibility flags (`0` / `1` / `2`), `[N]` object
/// areas, and `K` per-keypoint `sigmas`. Returns shape `[N]`.
pub fn oks_similarity(
    pred_kpts: &Tensor,
    gt_kpts: &Tensor,
    visibility: &Tensor,
    areas: &Tensor,
    sigmas: &[f32],
    eps: f32,
) -> Result<Tensor> {
    let cfg = OksLossConfig { eps, reduction: LossReduction::None };
    oks_op(pred_kpts, gt_kpts, visibility, areas, sigmas, &cfg, true)
}

/// Differentiable COCO Object Keypoint Similarity loss `1 - OKS` in `[0, 1]` for `[N, K, 2]`
/// predicted and ground-truth keypoints, `[N, K]` visibility flags (`0` / `1` / `2`), `[N]` object
/// areas, and `K` per-keypoint `sigmas`.
///
/// Unlabeled keypoints (`visibility == 0`) are masked out of both the numerator and denominator.
/// Instances with zero labeled keypoints contribute `0.0` loss and `0.0` gradient, and are excluded
/// from the [`LossReduction::Mean`] denominator.
pub fn oks_loss(
    pred_kpts: &Tensor,
    gt_kpts: &Tensor,
    visibility: &Tensor,
    areas: &Tensor,
    sigmas: &[f32],
    cfg: &OksLossConfig,
) -> Result<Tensor> {
    cfg.validate()?;
    oks_op(pred_kpts, gt_kpts, visibility, areas, sigmas, cfg, false)
}
