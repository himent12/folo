//! Geometric bounding-box regression losses with closed-form `f64` analytical gradients:
//!
//! - **MPDIoU** (Ma & Xu 2023): Minimum Point Distance IoU, in both scale-invariant enclosing-diagonal form
//!   ([`mpdiou_loss`], [`crate::loss::BoxLossKind::Mpdiou`]) and fixed image/grid diagonal form
//!   ([`mpdiou_loss_with_scale`]).
//! - **Shape-IoU** (Zhang & Zhang 2023): aspect-weighted center distance and shape-value penalty
//!   ([`shape_iou_loss`], [`crate::loss::BoxLossKind::ShapeIou`]).
//! - **Focaler-IoU** (Zhang & Zhang 2024): composable linear interval remapping for IoU-based losses
//!   ([`focaler_iou`], [`focaler_iou_loss`]).

use folo_core::{DType, Result, Tensor};

use super::box_loss::{check_box_pair, dfl_box_from_logits};
use crate::loss::invalid;

#[inline]
pub(super) fn w_min(x: f64, y: f64) -> f64 {
    if x < y {
        1.0
    } else if x == y {
        0.5
    } else {
        0.0
    }
}

#[inline]
pub(super) fn w_max(x: f64, y: f64) -> f64 {
    if x > y {
        1.0
    } else if x == y {
        0.5
    } else {
        0.0
    }
}

/// Scale-normalized `1 - IoU` and its analytical gradient `d(1 - IoU) / d(ax1, ay1, ax2, ay2)` in `f64`,
/// together with normalized side lengths so geometric penalties are exact and scale-invariant at every
/// positive scale (from `1e-30` to `1e30`) with explicit zero-denominator guards.
pub(super) struct IouTerm {
    pub(super) one_minus_iou: f64,
    pub(super) grad: [f64; 4],
    pub(super) w1s: f64,
    pub(super) h1s: f64,
    pub(super) w2s: f64,
    pub(super) h2s: f64,
    pub(super) w2: f64,
    pub(super) h2: f64,
    pub(super) inv_sx: f64,
    pub(super) inv_sy: f64,
}

pub(super) fn one_minus_iou_f64(a: [f64; 4], b: [f64; 4]) -> IouTerm {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let w1 = (ax2 - ax1).max(0.0);
    let h1 = (ay2 - ay1).max(0.0);
    let w2 = (bx2 - bx1).max(0.0);
    let h2 = (by2 - by1).max(0.0);
    let mx = w1.max(w2);
    let my = h1.max(h2);
    let inv_sx = if mx > 0.0 { 1.0 / mx } else { 1.0 };
    let inv_sy = if my > 0.0 { 1.0 / my } else { 1.0 };
    let (w1s, w2s) = (w1 * inv_sx, w2 * inv_sx);
    let (h1s, h2s) = (h1 * inv_sy, h2 * inv_sy);
    let ix = ax2.min(bx2) - ax1.max(bx1);
    let iy = ay2.min(by2) - ay1.max(by1);
    let (iws, ihs) = (ix.max(0.0) * inv_sx, iy.max(0.0) * inv_sy);
    let area1 = w1s * h1s;
    let area2 = w2s * h2s;
    let inter = iws * ihs;
    let uni = (area1 - inter).max(0.0) + area2;
    if uni <= 0.0 {
        return IouTerm { one_minus_iou: 1.0, grad: [0.0; 4], w1s, h1s, w2s, h2s, w2, h2, inv_sx, inv_sy };
    }
    let deficit = (area1 - inter).max(0.0) + (area2 - inter).max(0.0);
    let one_minus_iou = deficit / uni;
    let g_scale_x = -inv_sx / uni;
    let g_scale_y = -inv_sy / uni;
    let h1s_act = if ax2 > ax1 { h1s } else { 0.0 };
    let w1s_act = if ay2 > ay1 { w1s } else { 0.0 };
    let c_x1 = if ix > 0.0 { w_max(ax1, bx1) } else { 0.0 };
    let c_x2 = if ix > 0.0 { w_min(ax2, bx2) } else { 0.0 };
    let c_y1 = if iy > 0.0 { w_max(ay1, by1) } else { 0.0 };
    let c_y2 = if iy > 0.0 { w_min(ay2, by2) } else { 0.0 };
    let grad = [
        -g_scale_x * ((inter * (2.0 * c_x1 * ihs - h1s_act) + deficit * (c_x1 * ihs)) / uni),
        -g_scale_y * ((inter * (2.0 * c_y1 * iws - w1s_act) + deficit * (c_y1 * iws)) / uni),
        g_scale_x * ((inter * (2.0 * c_x2 * ihs - h1s_act) + deficit * (c_x2 * ihs)) / uni),
        g_scale_y * ((inter * (2.0 * c_y2 * iws - w1s_act) + deficit * (c_y2 * iws)) / uni),
    ];
    IouTerm { one_minus_iou, grad, w1s, h1s, w2s, h2s, w2, h2, inv_sx, inv_sy }
}

/// Single-box MPDIoU forward `(loss, dL/d(ax1, ay1, ax2, ay2))` in `f64`.
///
/// - When `scale_sq == None` (scale-invariant enclosing-diagonal form):
///   ```text
///   d1² = (ax1 - bx1)² + (ay1 - by1)²,   d2² = (ax2 - bx2)² + (ay2 - by2)²
///   c²  = cw² + ch²
///   L   = (1 - IoU) + (d1² + d2²) / (2 * c²)      (0 penalty when cw == ch == 0)
///   ```
///   Note that `(d1² + d2²) / 2 = rho² + (dw² + dh²) / 4`, so this reduces identically to DIoU loss for
///   equal-sized boxes while adding an exact scale/aspect mismatch penalty for concentric boxes and staying
///   strictly scale-invariant across all positive scales (from `1e-30` to `1e30`) and in `[0, 2]` on valid boxes.
/// - When `scale_sq == Some(s2)` with `s2 = W² + H² > 0` (Ma & Xu 2023 Eq. 6-8 fixed-scale form):
///   ```text
///   L = (1 - IoU) + (d1² + d2²) / (W² + H²)
///   ```
pub(crate) fn mpdiou_box_f64(a: [f64; 4], b: [f64; 4], scale_sq: Option<f64>) -> (f64, [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4]);
    }
    let iou = one_minus_iou_f64(a, b);
    let mut g = iou.grad;
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let (dx1, dy1, dx2, dy2) = (ax1 - bx1, ay1 - by1, ax2 - bx2, ay2 - by2);

    let mpd_pen = if let Some(s2) = scale_sq {
        let inv_s = 1.0 / s2.sqrt();
        let (dx1s, dy1s, dx2s, dy2s) = (dx1 * inv_s, dy1 * inv_s, dx2 * inv_s, dy2 * inv_s);
        let pen = dx1s * dx1s + dy1s * dy1s + dx2s * dx2s + dy2s * dy2s;
        let coeff = 2.0 * inv_s;
        g[0] += coeff * dx1s;
        g[1] += coeff * dy1s;
        g[2] += coeff * dx2s;
        g[3] += coeff * dy2s;
        pen
    } else {
        let cw_raw = ax2.max(bx2) - ax1.min(bx1);
        let ch_raw = ay2.max(by2) - ay1.min(by1);
        let cw = cw_raw.max(0.0);
        let ch = ch_raw.max(0.0);
        let m_c = cw.max(ch);
        if m_c <= 0.0 {
            0.0
        } else {
            let inv_c = 1.0 / m_c;
            let (dx1s, dy1s, dx2s, dy2s) = (dx1 * inv_c, dy1 * inv_c, dx2 * inv_c, dy2 * inv_c);
            let (cws, chs) = (cw * inv_c, ch * inv_c);
            let num = dx1s * dx1s + dy1s * dy1s + dx2s * dx2s + dy2s * dy2s;
            let c2 = cws * cws + chs * chs; // strictly in [1.0, 2.0]
            let pen = num / (2.0 * c2);
            let num_coeff = inv_c / c2;
            let dc_coeff = 2.0 * pen * inv_c / c2;
            let e_x1 = if cw_raw > 0.0 { w_min(ax1, bx1) } else { 0.0 };
            let e_x2 = if cw_raw > 0.0 { w_max(ax2, bx2) } else { 0.0 };
            let e_y1 = if ch_raw > 0.0 { w_min(ay1, by1) } else { 0.0 };
            let e_y2 = if ch_raw > 0.0 { w_max(ay2, by2) } else { 0.0 };
            g[0] += num_coeff * dx1s + dc_coeff * cws * e_x1;
            g[1] += num_coeff * dy1s + dc_coeff * chs * e_y1;
            g[2] += num_coeff * dx2s - dc_coeff * cws * e_x2;
            g[3] += num_coeff * dy2s - dc_coeff * chs * e_y2;
            pen
        }
    };

    (iou.one_minus_iou + mpd_pen, g)
}

/// Differentiable scale-invariant MPDIoU loss (Ma & Xu 2023) between rows of `b1` (predicted) and `b2`
/// (ground truth), both `[N, 4]` `xyxy`, returning `[N]`:
///
/// ```text
/// L_MPDIoU = (1 - IoU) + (d1² + d2²) / (2 * (cw² + ch²))
/// ```
///
/// where `d1² = (ax1 - bx1)² + (ay1 - by1)²` and `d2² = (ax2 - bx2)² + (ay2 - by2)²` are the squared
/// distances between corresponding top-left and bottom-right corners, and `cw² + ch²` is the squared
/// enclosing-box diagonal (with explicit zero-denominator guards so tiny and huge boxes stay exact and
/// scale-invariant). Evaluates in `f64` with closed-form analytical gradients w.r.t. `b1` (`b2` is treated
/// as constant ground truth).
pub fn mpdiou_loss(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    mpdiou_loss_inner(b1, b2, None, "mpdiou_loss")
}

/// Differentiable fixed-scale MPDIoU loss (Ma & Xu 2023 Eq. 6-8) between rows of `b1` (predicted) and `b2`
/// (ground truth), both `[N, 4]` `xyxy`, normalized by `width² + height²`:
///
/// ```text
/// L_MPDIoU(W, H) = (1 - IoU) + (d1² + d2²) / (W² + H²)
/// ```
///
/// Errors if `width` or `height` is non-finite or `<= 0`, or if `width² + height²` overflows `f64`.
pub fn mpdiou_loss_with_scale(b1: &Tensor, b2: &Tensor, width: f32, height: f32) -> Result<Tensor> {
    if !(width.is_finite() && width > 0.0 && height.is_finite() && height > 0.0) {
        return Err(invalid(format!(
            "mpdiou_loss_with_scale: width and height must be finite and > 0, got ({width}, {height})"
        )));
    }
    let s2 = (width as f64) * (width as f64) + (height as f64) * (height as f64);
    if !s2.is_finite() || s2 <= 0.0 {
        return Err(invalid(format!(
            "mpdiou_loss_with_scale: width² + height² must be finite and > 0, got {s2}"
        )));
    }
    mpdiou_loss_inner(b1, b2, Some(s2), "mpdiou_loss_with_scale")
}

fn mpdiou_loss_inner(
    b1: &Tensor,
    b2: &Tensor,
    scale_sq: Option<f64>,
    op_name: &'static str,
) -> Result<Tensor> {
    let n = check_box_pair(b1, b2, op_name)?;
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv) = mpdiou_box_f64(ai, bi, scale_sq);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op(op_name, &[b1], out_t, move |g| {
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

/// Weighted scale-invariant `MPDIoU` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn mpdiou_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
) -> Result<Tensor> {
    dfl_box_from_logits("mpdiou_from_logits", logits, pd_t, tbox, t_scores, lams, tss, box_gain, |a, b| {
        mpdiou_box_f64(a, b, None)
    })
}

/// Single-box Shape-IoU forward `(loss, dL/d(ax1, ay1, ax2, ay2))` in `f64` (Zhang & Zhang 2023).
///
/// ```text
/// r_w = w_gt / (w_gt + h_gt),   r_h = h_gt / (w_gt + h_gt)
/// ww  = 2 * r_w^scale / (r_w^scale + r_h^scale)
/// hh  = 2 * r_h^scale / (r_w^scale + r_h^scale)
/// distance^shape = (hh * (cx_a - cx_b)² + ww * (cy_a - cy_b)²) / (cw² + ch²)
/// omega_w = hh * |w_a - w_b| / max(w_a, w_b)
/// omega_h = ww * |h_a - h_b| / max(h_a, h_b)
/// Omega^shape = (1 - exp(-omega_w))^4 + (1 - exp(-omega_h))^4
/// L_Shape-IoU = (1 - IoU) + distance^shape + 0.5 * Omega^shape
/// ```
///
/// Zero denominators are guarded explicitly (yielding 0 penalty and 0 gradient) rather than via fixed
/// epsilons, so identical boxes at any positive scale have exact `0.0` loss and the entire formula is strictly
/// scale-invariant. Because the shape-value exponent is `theta = 4`, `(1 - exp(-omega))^4` is `C^3`-smooth
/// across `w_a == w_b` and `h_a == h_b` with zero derivative at equality.
pub(crate) fn shape_iou_box_f64(a: [f64; 4], b: [f64; 4], scale: f64) -> (f64, [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4]);
    }
    let iou = one_minus_iou_f64(a, b);
    let mut g = iou.grad;
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);

    // Ground-truth directional weights (constant w.r.t. predicted box `a`).
    let sum_wh = iou.w2 + iou.h2;
    let (ww, hh) = if sum_wh > 0.0 {
        let rw = iou.w2 / sum_wh;
        let rh = iou.h2 / sum_wh;
        let ws = rw.powf(scale);
        let hs = rh.powf(scale);
        let denom = ws + hs;
        (2.0 * ws / denom, 2.0 * hs / denom)
    } else {
        (1.0, 1.0)
    };

    // Shape-weighted center distance penalty (exact zero-denominator guard when cw == ch == 0).
    let dcx = 0.5 * ((ax1 - bx1) + (ax2 - bx2));
    let dcy = 0.5 * ((ay1 - by1) + (ay2 - by2));
    let cw_raw = ax2.max(bx2) - ax1.min(bx1);
    let ch_raw = ay2.max(by2) - ay1.min(by1);
    let cw = cw_raw.max(0.0);
    let ch = ch_raw.max(0.0);
    let m_c = cw.max(ch);
    let dist_shape = if m_c <= 0.0 {
        0.0
    } else {
        let inv_c = 1.0 / m_c;
        let (dcxs, dcys) = (dcx * inv_c, dcy * inv_c);
        let (cws, chs) = (cw * inv_c, ch * inv_c);
        let num = hh * dcxs * dcxs + ww * dcys * dcys;
        let c2 = cws * cws + chs * chs; // strictly in [1.0, 2.0]
        let dist = num / c2;
        let g_dcx = (hh * dcxs * inv_c) / c2;
        let g_dcy = (ww * dcys * inv_c) / c2;
        let dc_coeff = 2.0 * dist * inv_c / c2;
        let e_x1 = if cw_raw > 0.0 { w_min(ax1, bx1) } else { 0.0 };
        let e_x2 = if cw_raw > 0.0 { w_max(ax2, bx2) } else { 0.0 };
        let e_y1 = if ch_raw > 0.0 { w_min(ay1, by1) } else { 0.0 };
        let e_y2 = if ch_raw > 0.0 { w_max(ay2, by2) } else { 0.0 };
        g[0] += g_dcx + dc_coeff * cws * e_x1;
        g[1] += g_dcy + dc_coeff * chs * e_y1;
        g[2] += g_dcx - dc_coeff * cws * e_x2;
        g[3] += g_dcy - dc_coeff * chs * e_y2;
        dist
    };

    // Shape-value penalty 0.5 * ((1 - exp(-omega_w))^4 + (1 - exp(-omega_h))^4).
    let shape_axis = |s1: f64, s2: f64, weight: f64, inv_s: f64, active: bool| -> (f64, f64) {
        let m = s1.max(s2);
        if m <= 0.0 {
            return (0.0, 0.0);
        }
        let omega = weight * (s1 - s2).abs() / m;
        let e = (-omega).exp();
        let om_e = -(-omega).exp_m1();
        let om_e2 = om_e * om_e;
        let pen = 0.5 * om_e2 * om_e2;
        if !active || s1 == s2 {
            return (pen, 0.0);
        }
        let d_omega_ds1 = if s1 > s2 { weight * s2 / (m * m) } else { -weight / m };
        let df_ds1 = (2.0 * om_e2 * om_e * e) * d_omega_ds1 * inv_s;
        (pen, df_ds1)
    };
    let (pen_w, df_dw1) = shape_axis(iou.w1s, iou.w2s, hh, iou.inv_sx, ax2 > ax1);
    let (pen_h, df_dh1) = shape_axis(iou.h1s, iou.h2s, ww, iou.inv_sy, ay2 > ay1);
    g[0] -= df_dw1;
    g[2] += df_dw1;
    g[1] -= df_dh1;
    g[3] += df_dh1;

    (iou.one_minus_iou + dist_shape + pen_w + pen_h, g)
}

/// Differentiable Shape-IoU loss (Zhang & Zhang 2023) between rows of `b1` (predicted) and `b2` (ground truth),
/// both `[N, 4]` `xyxy`, with scale factor `scale` in `[0, 2]` (default `0.0` for standard datasets, `0.5..=1.5`
/// for small-object sensitivity), returning `[N]`:
///
/// ```text
/// L_Shape-IoU = (1 - IoU) + distance^shape + 0.5 * Omega^shape
/// ```
///
/// Evaluates in `f64` with closed-form analytical gradients w.r.t. `b1` (`b2` is constant ground truth).
pub fn shape_iou_loss(b1: &Tensor, b2: &Tensor, scale: f32) -> Result<Tensor> {
    if !(scale.is_finite() && (0.0..=2.0).contains(&scale)) {
        return Err(invalid(format!("shape_iou_loss: scale must be finite and in 0..=2, got {scale}")));
    }
    let n = check_box_pair(b1, b2, "shape_iou_loss")?;
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let scale64 = scale as f64;
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv) = shape_iou_box_f64(ai, bi, scale64);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("shape_iou_loss", &[b1], out_t, move |g| {
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

/// Weighted `Shape-IoU` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn shape_iou_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
    scale: f32,
) -> Result<Tensor> {
    let scale64 = scale as f64;
    dfl_box_from_logits(
        "shape_iou_from_logits",
        logits,
        pd_t,
        tbox,
        t_scores,
        lams,
        tss,
        box_gain,
        move |a, b| shape_iou_box_f64(a, b, scale64),
    )
}

/// Differentiable Focaler-IoU linear interval remapping (Zhang & Zhang 2024):
///
/// ```text
/// IoU^focaler = clamp((IoU - d) / (u - d), 0, 1),   0 <= d < u <= 1
/// ```
///
/// Amplifies the IoU gradient by `1 / (u - d)` for samples inside `(d, u)` while saturating samples below `d`
/// to `0` and above `u` to `1`. Preserves `NaN` propagation in forward and backward.
pub fn focaler_iou(iou: &Tensor, d: f32, u: f32) -> Result<Tensor> {
    if iou.dtype() != DType::F32 {
        return Err(folo_core::Error::DTypeMismatch {
            op: "focaler_iou",
            expected: DType::F32,
            got: iou.dtype(),
        });
    }
    if !(d.is_finite() && u.is_finite() && d >= 0.0 && u <= 1.0 && d < u) {
        return Err(invalid(format!("focaler_iou: require 0 <= d < u <= 1, got d={d}, u={u}")));
    }
    let dev = iou.device();
    let shape = iou.shape().clone();
    let vals = iou.to_vec::<f32>()?;
    let inv_span = 1.0f64 / (u as f64 - d as f64);
    let (d64, u64) = (d as f64, u as f64);
    let mut out = Vec::with_capacity(vals.len());
    let mut slope = Vec::with_capacity(vals.len());
    for &x in &vals {
        if !x.is_finite() {
            out.push(f32::NAN);
            slope.push(f64::NAN);
        } else {
            let x64 = x as f64;
            if x64 <= d64 {
                out.push(0.0);
                slope.push(0.0);
            } else if x64 >= u64 {
                out.push(1.0);
                slope.push(0.0);
            } else {
                out.push(((x64 - d64) * inv_span) as f32);
                slope.push(inv_span);
            }
        }
    }
    let out_t = Tensor::from_vec(out, &shape)?.to_device(dev)?;
    Tensor::custom_op("focaler_iou", &[iou], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let grad: Vec<f32> = gv
            .iter()
            .zip(&slope)
            .map(|(&gi, &s)| if gi == 0.0 && s.is_finite() { 0.0 } else { (gi as f64 * s) as f32 })
            .collect();
        Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
    })
}

/// Composable Focaler-IoU loss wrapper (Zhang & Zhang 2024 Eq. 11):
///
/// ```text
/// L_Focaler = L_base + (IoU - IoU^focaler(IoU; d, u))
/// ```
///
/// For any base IoU loss `L_base = (1 - IoU) + R` (such as CIoU, MPDIoU, or Shape-IoU), replaces the linear
/// `1 - IoU` term with `1 - IoU^focaler` while keeping the geometric penalty `R` intact. Grouping
/// `(IoU - IoU^focaler)` first prevents catastrophic cancellation of small `L_base` values when `IoU ≈ 1`
/// and reduces bit-identically to `L_base` at `(d = 0, u = 1)` for all `IoU` in `[0, 1]`.
pub fn focaler_iou_loss(base_loss: &Tensor, iou: &Tensor, d: f32, u: f32) -> Result<Tensor> {
    if base_loss.dtype() != DType::F32 {
        return Err(folo_core::Error::DTypeMismatch {
            op: "focaler_iou_loss",
            expected: DType::F32,
            got: base_loss.dtype(),
        });
    }
    if iou.dtype() != DType::F32 {
        return Err(folo_core::Error::DTypeMismatch {
            op: "focaler_iou_loss",
            expected: DType::F32,
            got: iou.dtype(),
        });
    }
    if base_loss.device() != iou.device() {
        return Err(folo_core::Error::DeviceMismatch {
            op: "focaler_iou_loss",
            lhs: base_loss.device(),
            rhs: iou.device(),
        });
    }
    if base_loss.shape() != iou.shape() {
        return Err(invalid(format!(
            "focaler_iou_loss: shape mismatch between base_loss {} and iou {}",
            base_loss.shape(),
            iou.shape()
        )));
    }
    let remapped = focaler_iou(iou, d, u)?;
    base_loss.add(&iou.sub(&remapped)?)
}
