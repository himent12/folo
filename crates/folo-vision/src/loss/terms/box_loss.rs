//! Differentiable box regression losses: CIoU, Inner-CIoU (Zhang et al. 2023), Wise-IoU v3
//! (Tong et al. 2023), Rational-Focal Edge-Decoupled IoU (R-PIoU), and Normalized Wasserstein Distance (NWD).

use folo_core::{DType, Result, Tensor, no_grad};

use super::dfl::exp_neg_f64;
use crate::loss::invalid;

/// `atan(x)` elementwise. folo-core has no `atan` kernel, so the forward runs on the host (the CIoU inputs are one
/// value per foreground anchor) and the backward `g / (1 + x²)` uses tensor ops on `x`'s device.
fn atan(x: &Tensor) -> Result<Tensor> {
    let v: Vec<f32> = x.to_vec::<f32>()?.into_iter().map(f32::atan).collect();
    let out = Tensor::from_vec(v, x.shape())?.to_device(x.device())?;
    let xs = x.detach();
    Tensor::custom_op("atan", &[x], out, move |g| Ok(vec![Some(g.div(&xs.square()?.add_scalar(1.0)?)?)]))
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

/// Validate that `b1` and `b2` are matching `[N, 4]` `F32` tensors on the same device.
pub(super) fn check_box_pair(b1: &Tensor, b2: &Tensor, op: &'static str) -> Result<usize> {
    if b1.dtype() != DType::F32 {
        return Err(folo_core::Error::DTypeMismatch { op, expected: DType::F32, got: b1.dtype() });
    }
    if b2.dtype() != DType::F32 {
        return Err(folo_core::Error::DTypeMismatch { op, expected: DType::F32, got: b2.dtype() });
    }
    if b1.device() != b2.device() {
        return Err(folo_core::Error::DeviceMismatch { op, lhs: b1.device(), rhs: b2.device() });
    }
    let (n1, c1) =
        b1.dims2().map_err(|_| invalid(format!("{op}: b1 must be rank 2 [N, 4], got {}", b1.shape())))?;
    let (n2, c2) =
        b2.dims2().map_err(|_| invalid(format!("{op}: b2 must be rank 2 [N, 4], got {}", b2.shape())))?;
    if c1 != 4 || c2 != 4 || n1 != n2 {
        return Err(invalid(format!(
            "{op}: b1 {} and b2 {} must be matching [N, 4] tensors",
            b1.shape(),
            b2.shape()
        )));
    }
    Ok(n1)
}

/// `[N]` tensor equal to `0.0` (with zero gradient) on every finite row of `(b1, b2)`, and `NaN`
/// (with `NaN` gradients on both `b1` and `b2`) on any row containing `NaN` or `±Inf`.
fn nonfinite_box_poison(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let mask = {
        let _g = no_grad();
        b1.mul_scalar(0.0)?.add(&b2.mul_scalar(0.0)?)?.sum(1, false)?.unsqueeze(1)?
    };
    b1.mul(&mask)?.add(&b2.mul(&mask)?)?.sum(1, false)
}

/// Differentiable Inner-CIoU loss (Zhang et al. 2023) between rows of `b1` and `b2` (`[N, 4]`, `xyxy`),
/// returning `[N]`:
///
/// ```text
/// L_Inner-CIoU = (1 - IoU_inner(ratio)) + rho² / c² + alpha * v
/// ```
///
/// where `IoU_inner` is the plain IoU of auxiliary boxes scaled about their centres by `ratio` (typically
/// `0.7` in `[0.5, 1.5]`), while the distance and aspect-ratio penalties stay on the full boxes. At
/// `ratio == 1.0`, reduces identically to `1 - ciou(b1, b2)` on valid non-inverted boxes.
pub fn inner_ciou_loss(b1: &Tensor, b2: &Tensor, ratio: f32) -> Result<Tensor> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(invalid(format!("inner_ratio must be finite and > 0, got {ratio}")));
    }
    let n = check_box_pair(b1, b2, "inner_ciou_loss")?;
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, b1.device());
    }
    const EPS: f32 = 1e-7;
    let col = |b: &Tensor, k: usize| b.narrow(1, k, 1)?.squeeze(1);
    let (ax1, ay1, ax2, ay2) = (col(b1, 0)?, col(b1, 1)?, col(b1, 2)?, col(b1, 3)?);
    let (bx1, by1, bx2, by2) = (col(b2, 0)?, col(b2, 1)?, col(b2, 2)?, col(b2, 3)?);
    let poison = nonfinite_box_poison(b1, b2)?;

    // Full-box CIoU terms with non-negative side clamps so inverted boxes never hit 0-division or negative union.
    let (w1, h1) = (ax2.sub(&ax1)?.relu()?, ay2.sub(&ay1)?.relu()?.add_scalar(EPS)?);
    let (w2, h2) = (bx2.sub(&bx1)?.relu()?, by2.sub(&by1)?.relu()?.add_scalar(EPS)?);
    let iw = ax2.minimum(&bx2)?.sub(&ax1.maximum(&bx1)?)?.relu()?;
    let ih = ay2.minimum(&by2)?.sub(&ay1.maximum(&by1)?)?.relu()?;
    let inter = iw.mul(&ih)?;
    let union = w1.mul(&h1)?.add(&w2.mul(&h2)?)?.sub(&inter)?.relu()?.add_scalar(EPS)?;
    let full_iou = inter.div(&union)?;
    let cw = ax2.maximum(&bx2)?.sub(&ax1.minimum(&bx1)?)?.relu()?;
    let ch = ay2.maximum(&by2)?.sub(&ay1.minimum(&by1)?)?.relu()?;
    let c2 = cw.square()?.add(&ch.square()?)?.add_scalar(EPS)?;
    let dx = bx1.add(&bx2)?.sub(&ax1)?.sub(&ax2)?;
    let dy = by1.add(&by2)?.sub(&ay1)?.sub(&ay2)?;
    let rho2 = dx.square()?.add(&dy.square()?)?.mul_scalar(0.25)?;
    let d = atan(&w2.div(&h2)?)?.sub(&atan(&w1.div(&h1)?)?)?;
    let v = d.square()?.mul_scalar(4.0 / (std::f32::consts::PI * std::f32::consts::PI))?;
    let alpha = {
        let _g = no_grad();
        let raw = v.sub(&full_iou)?.add_scalar(1.0 + EPS)?;
        let eps = raw.zeros_like()?.add_scalar(EPS)?;
        v.div(&raw.maximum(&eps)?)?
    };

    // At ratio == 1.0, auxiliary boxes equal the full boxes directly (avoids midpoint rounding on thin boxes).
    let inner_iou = if ratio == 1.0 {
        full_iou
    } else {
        let ca = 0.5 * (1.0 + ratio);
        let cb = 0.5 * (1.0 - ratio);
        let ix1_a = ax1.mul_scalar(ca)?.add(&ax2.mul_scalar(cb)?)?;
        let iy1_a = ay1.mul_scalar(ca)?.add(&ay2.mul_scalar(cb)?)?;
        let ix2_a = ax2.mul_scalar(ca)?.add(&ax1.mul_scalar(cb)?)?;
        let iy2_a = ay2.mul_scalar(ca)?.add(&ay1.mul_scalar(cb)?)?;
        let ix1_b = bx1.mul_scalar(ca)?.add(&bx2.mul_scalar(cb)?)?;
        let iy1_b = by1.mul_scalar(ca)?.add(&by2.mul_scalar(cb)?)?;
        let ix2_b = bx2.mul_scalar(ca)?.add(&bx1.mul_scalar(cb)?)?;
        let iy2_b = by2.mul_scalar(ca)?.add(&by1.mul_scalar(cb)?)?;
        let in_iw = ix2_a.minimum(&ix2_b)?.sub(&ix1_a.maximum(&ix1_b)?)?.relu()?;
        let in_ih = iy2_a.minimum(&iy2_b)?.sub(&iy1_a.maximum(&iy1_b)?)?.relu()?;
        let in_inter = in_iw.mul(&in_ih)?;
        let in_w1 = ix2_a.sub(&ix1_a)?.relu()?;
        let in_h1 = iy2_a.sub(&iy1_a)?.relu()?.add_scalar(EPS)?;
        let in_w2 = ix2_b.sub(&ix1_b)?.relu()?;
        let in_h2 = iy2_b.sub(&iy1_b)?.relu()?.add_scalar(EPS)?;
        let in_union =
            in_w1.mul(&in_h1)?.add(&in_w2.mul(&in_h2)?)?.sub(&in_inter)?.relu()?.add_scalar(EPS)?;
        in_inter.div(&in_union)?
    };

    inner_iou.rsub_scalar(1.0)?.add(&rho2.div(&c2)?)?.add(&v.mul(&alpha)?)?.add(&poison)
}

/// Differentiable Wise-IoU v3 loss (Tong et al. 2023) between rows of `b1` (predicted) and `b2` (target)
/// (`[N, 4]`, `xyxy`), returning `[N]`:
///
/// ```text
/// L_IoU = 1 - IoU(b1, b2)
/// R_WIoU = exp(((cx1 - cx2)² + (cy1 - cy2)²) / ((cw² + ch²)* + eps))
/// beta = L_IoU* / max(iou_mean, 1e-4)
/// r = beta / (delta * alpha^(beta - delta))       (detached non-monotonic focusing gain)
/// L_WIoU_v3 = r * R_WIoU * L_IoU
/// ```
///
/// `iou_mean` is the running exponential moving average `L_IoU_bar` across training steps (Tong et al. §III-B,
/// initialized to `1.0`). Both the enclosing diagonal denominator `(cw² + ch²)*` and the dynamic focusing gain
/// `r` are detached under `no_grad()` so gradients never suffer enclosing-box hindrance or outlier amplification.
pub fn wiou_v3_loss(b1: &Tensor, b2: &Tensor, alpha: f32, delta: f32, iou_mean: f32) -> Result<Tensor> {
    Ok(wiou_v3_loss_and_mean(b1, b2, alpha, delta, iou_mean)?.0)
}

/// [`wiou_v3_loss`] also returning the current batch's unweighted mean `L_IoU` (`None` when `N == 0`), so
/// the trainer can update its running EMA `L_IoU_bar`.
pub(super) fn wiou_v3_loss_and_mean(
    b1: &Tensor,
    b2: &Tensor,
    alpha: f32,
    delta: f32,
    iou_mean: f32,
) -> Result<(Tensor, Option<f32>)> {
    if !alpha.is_finite() || alpha <= 1.0 {
        return Err(invalid(format!("wiou_alpha must be finite and > 1, got {alpha}")));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(invalid(format!("wiou_delta must be finite and > 0, got {delta}")));
    }
    if !iou_mean.is_finite() || iou_mean <= 0.0 {
        return Err(invalid(format!("wiou iou_mean must be finite and > 0, got {iou_mean}")));
    }
    let n = check_box_pair(b1, b2, "wiou_v3_loss")?;
    if n == 0 {
        return Ok((Tensor::zeros_with([0], DType::F32, b1.device())?, None));
    }
    const EPS: f32 = 1e-7;
    let col = |b: &Tensor, k: usize| b.narrow(1, k, 1)?.squeeze(1);
    let (ax1, ay1, ax2, ay2) = (col(b1, 0)?, col(b1, 1)?, col(b1, 2)?, col(b1, 3)?);
    let (bx1, by1, bx2, by2) = (col(b2, 0)?, col(b2, 1)?, col(b2, 2)?, col(b2, 3)?);
    let poison = nonfinite_box_poison(b1, b2)?;
    let (w1, h1) = (ax2.sub(&ax1)?.relu()?, ay2.sub(&ay1)?.relu()?);
    let (w2, h2) = (bx2.sub(&bx1)?.relu()?, by2.sub(&by1)?.relu()?);
    let iw = ax2.minimum(&bx2)?.sub(&ax1.maximum(&bx1)?)?.relu()?;
    let ih = ay2.minimum(&by2)?.sub(&ay1.maximum(&by1)?)?.relu()?;
    let inter = iw.mul(&ih)?;
    let union = w1.mul(&h1)?.add(&w2.mul(&h2)?)?.sub(&inter)?.relu()?.add_scalar(EPS)?;
    let l_iou = inter.div(&union)?.clamp(0.0, 1.0)?.rsub_scalar(1.0)?.add(&poison)?;

    // Distance attention R_WIoU = exp(rho² / c2*) with detached enclosing diagonal c2*.
    let dx = bx1.add(&bx2)?.sub(&ax1)?.sub(&ax2)?;
    let dy = by1.add(&by2)?.sub(&ay1)?.sub(&ay2)?;
    let rho2 = dx.square()?.add(&dy.square()?)?.mul_scalar(0.25)?;
    let c2_detached = {
        let _g = no_grad();
        let cw = ax2.maximum(&bx2)?.sub(&ax1.minimum(&bx1)?)?.relu()?;
        let ch = ay2.maximum(&by2)?.sub(&ay1.minimum(&by1)?)?.relu()?;
        cw.square()?.add(&ch.square()?)?.add_scalar(EPS)?
    };
    let r_dist = rho2.div(&c2_detached)?.clamp(0.0, 10.0)?.exp()?;

    // Non-monotonic dynamic focusing gain r = beta / (delta * alpha^(beta - delta)), completely detached.
    let (r_focus, batch_mean) = {
        let _g = no_grad();
        let batch_mean = l_iou.mean_all()?.item::<f32>()?;
        let beta = l_iou.div_scalar(iou_mean.max(1e-4))?;
        let ln_a = alpha.ln();
        let exponent = beta.sub_scalar(delta)?.mul_scalar(ln_a)?.clamp(-80.0, 80.0)?;
        let denom = exponent.exp()?.mul_scalar(delta)?;
        (beta.div(&denom)?, batch_mean)
    };
    let loss = r_focus.mul(&r_dist)?.mul(&l_iou)?.add(&poison)?;
    Ok((loss, Some(batch_mean)))
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

type BoxLossF64Fn = fn([f64; 4], [f64; 4], f64, f64, f64) -> (f64, [f64; 4]);

/// Weighted box-loss scalar `(box_gain / tss) * Σ_i t_i * (1 - lam_i) * L_box_i` directly from DFL
/// `logits: [N, 4, R]`, reusing the tensor-decoded `pd` so IoU and geometric terms evaluate the exact same
/// predicted box, and computing the DFL softmax Jacobian centered on the dominant bin `k*` in `f64` alongside
/// batch scaling, `box_gain`, `tss`, `t`, and `(1 - lam)` before rounding the final logit gradient to `f32`.
#[allow(clippy::too_many_arguments)]
pub(super) fn dfl_box_from_logits(
    op_name: &'static str,
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
    mut box_fn: impl FnMut([f64; 4], [f64; 4]) -> (f64, [f64; 4]),
) -> Result<Tensor> {
    let (n, sides, r) = logits.dims3()?;
    debug_assert_eq!(sides, 4);
    let dev = logits.device();
    if n == 0 {
        return Tensor::zeros_with([], DType::F32, dev);
    }
    let raw_logits = logits.to_vec::<f32>()?;
    let pds = pd_t.to_vec::<f32>()?;
    let mut sum_w = 0.0f64;
    let mut base_d = vec![0.0f64; n * 4];
    let mut weights = vec![0.0f64; n];
    for i in 0..n {
        let ai = [pds[4 * i] as f64, pds[4 * i + 1] as f64, pds[4 * i + 2] as f64, pds[4 * i + 3] as f64];
        let bi = [tbox[4 * i] as f64, tbox[4 * i + 1] as f64, tbox[4 * i + 2] as f64, tbox[4 * i + 3] as f64];
        let (lv, g_xyxy) = box_fn(ai, bi);
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
    Tensor::custom_op(op_name, &[logits], out_t, move |g| {
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

/// Weighted `R-PIoU` or `S-AEIoU` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
fn algebraic_box_from_logits(
    op_name: &'static str,
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
    box_fn: BoxLossF64Fn,
) -> Result<Tensor> {
    let (b64, k64, d64) = (beta as f64, kappa as f64, delta0 as f64);
    dfl_box_from_logits(op_name, logits, pd_t, tbox, t_scores, lams, tss, box_gain, move |a, b| {
        box_fn(a, b, b64, k64, d64)
    })
}

/// Weighted `R-PIoU` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn rpiou_from_logits(
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
    algebraic_box_from_logits(
        "rpiou_from_logits",
        logits,
        pd_t,
        tbox,
        t_scores,
        lams,
        tss,
        box_gain,
        beta,
        kappa,
        delta0,
        rpiou_box_f64,
    )
}

/// Single-box `S-AEIoU` forward `(loss, dL/d(ax1, ay1, ax2, ay2))` in `f64`, matching `det_box.wgsl`.
pub(crate) fn saeiou_box_f64(
    a: [f64; 4],
    b: [f64; 4],
    beta: f64,
    kappa: f64,
    delta0: f64,
) -> (f64, [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4]);
    }
    const EPS_S: f64 = 1e-7;
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
    let mx = w1.max(w2);
    let my = h1.max(h2);
    let inv_sx = if mx > 1e6 { 1.0 / mx } else { 1.0 };
    let inv_sy = if my > 1e6 { 1.0 / my } else { 1.0 };
    let (w1s, w2s) = (w1 * inv_sx, w2 * inv_sx);
    let (h1s, h2s) = (h1 * inv_sy, h2 * inv_sy);
    let ix = ax2.min(bx2) - ax1.max(bx1);
    let iy = ay2.min(by2) - ay1.max(by1);
    let (iws, ihs) = (ix.max(0.0) * inv_sx, iy.max(0.0) * inv_sy);
    let area1 = w1s * h1s;
    let area2 = w2s * h2s;
    let inter = iws * ihs;
    let eps_eff = (EPS_S * inv_sx * inv_sy).max(1e-30);
    let uni = area1 + area2 - inter + eps_eff;
    let iou = inter / uni;
    let deficit = (area1 - inter).max(0.0) + (area2 - inter).max(0.0) + eps_eff;
    let focal = (deficit / uni) * (1.0 + beta * iou);
    let g_u = -(1.0 - beta + 2.0 * beta * iou);
    let g_scale_x = (g_u * inv_sx) / uni;
    let g_scale_y = (g_u * inv_sy) / uni;
    let h1s_act = if ax2 > ax1 { h1s } else { 0.0 };
    let w1s_act = if ay2 > ay1 { w1s } else { 0.0 };
    let c_x1 = if ix > 0.0 { w_max(ax1, bx1) } else { 0.0 };
    let c_x2 = if ix > 0.0 { w_min(ax2, bx2) } else { 0.0 };
    let c_y1 = if iy > 0.0 { w_max(ay1, by1) } else { 0.0 };
    let c_y2 = if iy > 0.0 { w_min(ay2, by2) } else { 0.0 };
    let mut g = [
        -g_scale_x * ((inter * (2.0 * c_x1 * ihs - h1s_act) + deficit * (c_x1 * ihs)) / uni),
        -g_scale_y * ((inter * (2.0 * c_y1 * iws - w1s_act) + deficit * (c_y1 * iws)) / uni),
        g_scale_x * ((inter * (2.0 * c_x2 * ihs - h1s_act) + deficit * (c_x2 * ihs)) / uni),
        g_scale_y * ((inter * (2.0 * c_y2 * iws - w1s_act) + deficit * (c_y2 * iws)) / uni),
    ];
    let sx = w2 + delta0;
    let sy = h2 + delta0;
    let inv_sum = 1.0 / (sx + sy);
    let wx = 2.0 * sx * inv_sum;
    let wy = 2.0 * sy * inv_sum;
    let rk = kappa.sqrt();
    let qt = |u: f64| -> (f64, f64) {
        let au = u.abs();
        if au > 1e8 {
            (1.0 / au, if u >= 0.0 { 1.0 } else { -1.0 })
        } else {
            let q = 1.0 / (1.0 + u * u).sqrt();
            (q, u * q)
        }
    };
    let (qx1, tx1) = qt(rk * (ax1 - bx1) / sx);
    let (qx2, tx2) = qt(rk * (ax2 - bx2) / sx);
    let (qy1, ty1) = qt(rk * (ay1 - by1) / sy);
    let (qy2, ty2) = qt(rk * (ay2 - by2) / sy);
    let phi = |q: f64, t: f64| (t * t) / (1.0 + q);
    let edge = 0.25 * (wx * (phi(qx1, tx1) + phi(qx2, tx2)) + wy * (phi(qy1, ty1) + phi(qy2, ty2)));
    let c = 0.5 * rk * inv_sum;
    g[0] += c * tx1 * (qx1 * qx1);
    g[1] += c * ty1 * (qy1 * qy1);
    g[2] += c * tx2 * (qx2 * qx2);
    g[3] += c * ty2 * (qy2 * qy2);
    (focal + edge, g)
}

/// Differentiable Scale-normalized Algebraic Edge IoU loss (`S-AEIoU`) between rows of `b1` (predicted) and
/// `b2` (ground truth), both `[N, 4]` `xyxy` in grid units, returning `[N]`:
///
/// `L = (1 - u)(1 + beta * u) + 0.25 * (wx * (phi(u_x1) + phi(u_x2)) + wy * (phi(u_y1) + phi(u_y2)))`
///
/// where `u` is scale-normalized IoU, `sx = w_g + delta0`, `sy = h_g + delta0`, `wx = 2*sx/(sx+sy)`,
/// `wy = 2*sy/(sx+sy)`, `q(u) = (1 + u^2)^(-1/2)`, `t(u) = u * q(u)`, and `phi(u) = t(u)^2 / (1 + q(u))`.
/// Purely algebraic (no `atan`, `exp`, or `norm_f32`), with shared edge gradient prefactor
/// `C = sqrt(kappa) / (2 * (sx + sy))`.
pub fn saeiou_loss(b1: &Tensor, b2: &Tensor, beta: f32, kappa: f32, delta0: f32) -> Result<Tensor> {
    if !(beta.is_finite() && (0.0..=1.0).contains(&beta)) {
        return Err(invalid(format!("saeiou_loss: beta must be finite and in 0..=1, got {beta}")));
    }
    if !(kappa.is_finite() && kappa > 0.0) {
        return Err(invalid(format!("saeiou_loss: kappa must be finite and > 0, got {kappa}")));
    }
    if !(delta0.is_finite() && delta0 > 0.0) {
        return Err(invalid(format!("saeiou_loss: delta0 must be finite and > 0, got {delta0}")));
    }
    let n = check_box_pair(b1, b2, "saeiou_loss")?;
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let (b64, k64, d64) = (beta as f64, kappa as f64, delta0 as f64);
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv) = saeiou_box_f64(ai, bi, b64, k64, d64);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("saeiou_loss", &[b1], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let mut grad = vec![0.0f32; 4 * n];
        for i in 0..n {
            let gi = gv[i] as f64;
            if gi == 0.0 && g_box[i].iter().all(|v| v.is_finite()) {
                continue;
            }
            for s in 0..4 {
                grad[4 * i + s] = (gi * g_box[i][s]) as f32;
            }
        }
        Ok(vec![Some(Tensor::from_vec(grad, [n, 4])?.to_device(dev)?)])
    })
}

/// Weighted `S-AEIoU` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn saeiou_from_logits(
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
    algebraic_box_from_logits(
        "saeiou_from_logits",
        logits,
        pd_t,
        tbox,
        t_scores,
        lams,
        tss,
        box_gain,
        beta,
        kappa,
        delta0,
        saeiou_box_f64,
    )
}
