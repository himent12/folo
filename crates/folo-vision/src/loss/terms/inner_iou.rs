//! Generalized Inner-IoU auxiliary-box loss wrapper (Zhang et al., 2023, arXiv:2311.02877) with
//! exact closed-form `f64` analytical gradients.
//!
//! Inner-IoU replaces the primary `1 - IoU` overlap term of any geometric bounding-box regression
//! loss `L_X = (1 - IoU) + R_X` with `1 - IoU_inner`, where `IoU_inner` is computed on auxiliary
//! predicted and target boxes scaled about their centers by `ratio > 0` (typically `0.5..=1.5`;
//! `< 1.0` accelerates regression on high-IoU boxes, `> 1.0` expands overlap gradients on low-IoU
//! boxes), while retaining the full-box geometric penalty `R_X`:
//!
//! ```text
//! L_Inner-X = (1 - IoU_inner(ratio)) + R_X
//!           = L_X + (IoU - IoU_inner(ratio))
//! ```
//!
//! This module provides:
//! - [`InnerIouBase`]: selects any of `IoU`, `GIoU`, `DIoU`, `CIoU`, `EIoU`, `SIoU`, or `MPDIoU`.
//! - [`inner_iou`]: differentiable `[N]` auxiliary-box `IoU_inner` in `[0, 1]`.
//! - [`inner_iou_loss`]: cancellation-free closed-form `f64` forward & analytical backward for all
//!   seven [`InnerIouBase`] variants (computing `(1 - IoU_inner) + R_X` directly without subtracting
//!   `1 - IoU`).
//! - [`wrap_inner_iou_loss`]: composable wrapper `base_loss + (IoU - IoU_inner)` for any custom
//!   differentiable `[N]` box loss tensor.
//!
//! All functions evaluate geometry in `f64` with explicit scale normalization and zero-denominator
//! guards so identical valid boxes at any positive scale yield exact `0.0` loss and gradient,
//! degenerate/zero-area/inverted boxes remain finite, and `NaN` / `±Inf` inputs propagate `NaN`
//! cleanly through both forward and backward.

use folo_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::box_loss::check_box_pair;
use super::geom_loss::{one_minus_iou_f64, w_max, w_min};
use super::siou_eiou::{eiou_box_f64, enclosing_axis, siou_box_f64};
use crate::loss::invalid;

/// Base geometric bounding-box regression loss composed with [`inner_iou_loss`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InnerIouBase {
    /// Plain Inner-IoU: `1 - IoU_inner`.
    Iou,
    /// Inner-GIoU: `(1 - IoU_inner) + (C - U) / C` (enclosing-area penalty on full boxes).
    Giou,
    /// Inner-DIoU: `(1 - IoU_inner) + (dcx² + dcy²) / (cw² + ch²)`.
    Diou,
    /// Inner-CIoU: `(1 - IoU_inner) + (dcx² + dcy²) / (cw² + ch²) + alpha* * v` (with `alpha*`
    /// detached from the backward pass, matching standard CIoU).
    #[default]
    Ciou,
    /// Inner-EIoU: `(1 - IoU_inner) + (dcx² + dcy²) / (cw² + ch²) + (w1 - w2)² / cw² + (h1 - h2)² / ch²`.
    Eiou,
    /// Inner-SIoU: `(1 - IoU_inner) + 0.5 * (Delta + Omega)` with shape exponent `theta in [2, 6]`.
    Siou {
        /// Shape-cost exponent `theta` in `[2, 6]` (default `4.0`).
        #[serde(default = "default_siou_theta")]
        theta: f32,
    },
    /// Inner-MPDIoU: `(1 - IoU_inner) + (d1² + d2²) / (2 * (cw² + ch²))`.
    Mpdiou,
}

fn default_siou_theta() -> f32 {
    4.0
}

impl InnerIouBase {
    /// Validate any variant-specific parameters (such as `SIoU`'s `theta`).
    pub fn validate(&self) -> Result<()> {
        if let InnerIouBase::Siou { theta } = *self
            && !(theta.is_finite() && (2.0..=6.0).contains(&theta))
        {
            return Err(invalid(format!(
                "inner_iou_loss: SIoU theta must be finite and in 2..=6, got {theta}"
            )));
        }
        Ok(())
    }
}

/// Validate the auxiliary-box scale `ratio` (finite and strictly positive).
fn check_ratio(ratio: f32, op: &'static str) -> Result<f64> {
    if !(ratio.is_finite() && ratio > 0.0) {
        return Err(invalid(format!("{op}: ratio must be finite and > 0, got {ratio}")));
    }
    Ok(ratio as f64)
}

/// Scale one `xyxy` box about its center by `ratio > 0` in `f64`.
///
/// For non-inverted axes (`x2 >= x1`), uses the exact affine combination
/// `ix1 = ca * x1 + cb * x2`, `ix2 = cb * x1 + ca * x2` with `ca = (1 + ratio)/2`,
/// `cb = (1 - ratio)/2` (reducing bit-identically to the original box at `ratio == 1.0`).
/// Inverted axes (`x2 < x1`) collapse to zero extent at their midpoint so scaling never flips
/// an inverted box into positive area.
#[inline]
pub(crate) fn scale_box_f64(b: [f64; 4], ratio: f64) -> [f64; 4] {
    if ratio == 1.0 {
        return [b[0], b[1], b[2].max(b[0]), b[3].max(b[1])];
    }
    let ca = 0.5 * (1.0 + ratio);
    let cb = 0.5 * (1.0 - ratio);
    let (ix1, ix2) = if b[2] >= b[0] {
        (ca * b[0] + cb * b[2], cb * b[0] + ca * b[2])
    } else {
        let cx = 0.5 * (b[0] + b[2]);
        (cx, cx)
    };
    let (iy1, iy2) = if b[3] >= b[1] {
        (ca * b[1] + cb * b[3], cb * b[1] + ca * b[3])
    } else {
        let cy = 0.5 * (b[1] + b[3]);
        (cy, cy)
    };
    [ix1, iy1, ix2, iy2]
}

/// Single-box `1 - IoU_inner(ratio)` forward and exact analytical gradient w.r.t. `a` in `f64`.
pub(crate) fn inner_one_minus_iou_f64(a: [f64; 4], b: [f64; 4], ratio: f64) -> (f64, [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4]);
    }
    if ratio == 1.0 {
        let t = one_minus_iou_f64(a, b);
        return (t.one_minus_iou, t.grad);
    }
    let a_in = scale_box_f64(a, ratio);
    let b_in = scale_box_f64(b, ratio);
    let t = one_minus_iou_f64(a_in, b_in);
    let ca = 0.5 * (1.0 + ratio);
    let cb = 0.5 * (1.0 - ratio);
    let (gx1, gx2) = if a[2] > a[0] {
        (ca * t.grad[0] + cb * t.grad[2], cb * t.grad[0] + ca * t.grad[2])
    } else {
        (0.0, 0.0)
    };
    let (gy1, gy2) = if a[3] > a[1] {
        (ca * t.grad[1] + cb * t.grad[3], cb * t.grad[1] + ca * t.grad[3])
    } else {
        (0.0, 0.0)
    };
    (t.one_minus_iou, [gx1, gy1, gx2, gy2])
}

/// Single-box `IoU_inner(ratio)` forward and exact analytical gradient w.r.t. `a` in `f64`.
pub(crate) fn inner_iou_box_f64(a: [f64; 4], b: [f64; 4], ratio: f64) -> (f64, [f64; 4]) {
    let (om, g) = inner_one_minus_iou_f64(a, b, ratio);
    if !om.is_finite() {
        return (f64::NAN, [f64::NAN; 4]);
    }
    ((1.0 - om).clamp(0.0, 1.0), [-g[0], -g[1], -g[2], -g[3]])
}

/// Pure GIoU geometric penalty `(C - U) / C` and its exact analytical gradient w.r.t. `a` in `f64`,
/// scale-normalized by enclosing extents `(cw, ch)` so extreme scales (`1e-30..1e30`) never overflow
/// or underflow and identical valid boxes yield exact `0.0`.
fn giou_penalty_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4]) {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let (cw, dcw_da1, dcw_da2) = enclosing_axis(ax1, ax2, bx1, bx2);
    let (ch, dch_da1, dch_da2) = enclosing_axis(ay1, ay2, by1, by2);
    if cw <= 0.0 || ch <= 0.0 {
        return (0.0, [0.0; 4]);
    }
    let inv_cw = 1.0 / cw;
    let inv_ch = 1.0 / ch;
    let w1s = (ax2 - ax1).max(0.0) * inv_cw;
    let h1s = (ay2 - ay1).max(0.0) * inv_ch;
    let w2s = (bx2 - bx1).max(0.0) * inv_cw;
    let h2s = (by2 - by1).max(0.0) * inv_ch;
    let ix = ax2.min(bx2) - ax1.max(bx1);
    let iy = ay2.min(by2) - ay1.max(by1);
    let iws = ix.max(0.0) * inv_cw;
    let ihs = iy.max(0.0) * inv_ch;
    let inter_s = iws * ihs;
    let u_s = ((w1s * h1s - inter_s).max(0.0) + w2s * h2s).clamp(0.0, 1.0);
    let pen = (1.0 - u_s).max(0.0);

    // d(1 - U/C)/da = (U/C) * (dC/da)/C - (dU/da)/C.
    let h1s_act = if ax2 > ax1 { h1s } else { 0.0 };
    let w1s_act = if ay2 > ay1 { w1s } else { 0.0 };
    let c_x1 = if ix > 0.0 { w_max(ax1, bx1) } else { 0.0 };
    let c_x2 = if ix > 0.0 { w_min(ax2, bx2) } else { 0.0 };
    let c_y1 = if iy > 0.0 { w_max(ay1, by1) } else { 0.0 };
    let c_y2 = if iy > 0.0 { w_min(ay2, by2) } else { 0.0 };

    // d(area1 - inter)/d(ax1) = -h1 + c_x1 * ih  (scaled by 1/(cw*ch) = inv_cw * (h_s)).
    let du_da1 = (c_x1 * ihs - h1s_act) * inv_cw;
    let du_da2 = (h1s_act - c_x2 * ihs) * inv_cw;
    let du_db1 = (c_y1 * iws - w1s_act) * inv_ch;
    let du_db2 = (w1s_act - c_y2 * iws) * inv_ch;

    let g = [
        u_s * dcw_da1 * inv_cw - du_da1,
        u_s * dch_da1 * inv_ch - du_db1,
        u_s * dcw_da2 * inv_cw - du_da2,
        u_s * dch_da2 * inv_ch - du_db2,
    ];
    (pen, g)
}

/// Pure DIoU center-distance penalty `(dcx² + dcy²) / (cw² + ch²)` and its exact gradient w.r.t. `a`.
fn diou_penalty_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4]) {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let (cw, dcw_da1, dcw_da2) = enclosing_axis(ax1, ax2, bx1, bx2);
    let (ch, dch_da1, dch_da2) = enclosing_axis(ay1, ay2, by1, by2);
    let m_c = cw.max(ch);
    if m_c <= 0.0 {
        return (0.0, [0.0; 4]);
    }
    let dcx = 0.5 * ((ax1 - bx1) + (ax2 - bx2));
    let dcy = 0.5 * ((ay1 - by1) + (ay2 - by2));
    let inv_c = 1.0 / m_c;
    let (dcxs, dcys) = (dcx * inv_c, dcy * inv_c);
    let (cws, chs) = (cw * inv_c, ch * inv_c);
    let c2 = cws * cws + chs * chs; // in [1, 2]
    let dist = (dcxs * dcxs + dcys * dcys) / c2;
    let g_dcx = (dcxs * inv_c) / c2;
    let g_dcy = (dcys * inv_c) / c2;
    let dc_coeff = 2.0 * dist * inv_c / c2;
    let g = [
        g_dcx - dc_coeff * cws * dcw_da1,
        g_dcy - dc_coeff * chs * dch_da1,
        g_dcx - dc_coeff * cws * dcw_da2,
        g_dcy - dc_coeff * chs * dch_da2,
    ];
    (dist, g)
}

/// Pure CIoU aspect-ratio term `alpha* * v` and its exact gradient w.r.t. `a` with detached `alpha*`.
///
/// Returns `(alpha * v, alpha * dv/da, alpha)` so tests can also verify exact frozen-`alpha*`
/// central finite differences.
pub(crate) fn ciou_aspect_penalty_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4], f64) {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let w1 = (ax2 - ax1).max(0.0);
    let h1 = (ay2 - ay1).max(0.0);
    let w2 = (bx2 - bx1).max(0.0);
    let h2 = (by2 - by1).max(0.0);
    let full = one_minus_iou_f64(a, b);
    let d = w2.atan2(h2) - w1.atan2(h1);
    let c_pi = 4.0 / (std::f64::consts::PI * std::f64::consts::PI);
    let v = c_pi * d * d;
    let denom = full.one_minus_iou + v;
    let alpha = if denom > 0.0 && v > 0.0 { v / denom } else { 0.0 };
    if alpha == 0.0 || d == 0.0 {
        return (alpha * v, [0.0; 4], alpha);
    }
    let m1 = w1.max(h1);
    if m1 <= 0.0 {
        return (alpha * v, [0.0; 4], alpha);
    }
    // d(atan2(w1, h1))/dw1 = h1 / (w1² + h1²), d(atan2(w1, h1))/dh1 = -w1 / (w1² + h1²), scale-normalized.
    let inv_m1 = 1.0 / m1;
    let (w1s, h1s) = (w1 * inv_m1, h1 * inv_m1);
    let r2 = w1s * w1s + h1s * h1s; // in [1, 2]
    let dv_dd = 2.0 * c_pi * d;
    let coeff = -alpha * dv_dd * inv_m1 / r2;
    let dw = if ax2 > ax1 { coeff * h1s } else { 0.0 };
    let dh = if ay2 > ay1 { -coeff * w1s } else { 0.0 };
    (alpha * v, [-dw, -dh, dw, dh], alpha)
}

/// Pure MPDIoU corner-distance penalty `(d1² + d2²) / (2 * (cw² + ch²))` and its exact gradient w.r.t. `a`.
fn mpdiou_penalty_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4]) {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let (cw, dcw_da1, dcw_da2) = enclosing_axis(ax1, ax2, bx1, bx2);
    let (ch, dch_da1, dch_da2) = enclosing_axis(ay1, ay2, by1, by2);
    let m_c = cw.max(ch);
    if m_c <= 0.0 {
        return (0.0, [0.0; 4]);
    }
    let inv_c = 1.0 / m_c;
    let (dx1s, dy1s, dx2s, dy2s) =
        ((ax1 - bx1) * inv_c, (ay1 - by1) * inv_c, (ax2 - bx2) * inv_c, (ay2 - by2) * inv_c);
    let (cws, chs) = (cw * inv_c, ch * inv_c);
    let num = dx1s * dx1s + dy1s * dy1s + dx2s * dx2s + dy2s * dy2s;
    let c2 = cws * cws + chs * chs;
    let pen = num / (2.0 * c2);
    let num_coeff = inv_c / c2;
    let dc_coeff = 2.0 * pen * inv_c / c2;
    let g = [
        num_coeff * dx1s - dc_coeff * cws * dcw_da1,
        num_coeff * dy1s - dc_coeff * chs * dch_da1,
        num_coeff * dx2s - dc_coeff * cws * dcw_da2,
        num_coeff * dy2s - dc_coeff * chs * dch_da2,
    ];
    (pen, g)
}

/// Single-box generalized Inner-IoU loss forward and exact analytical gradient w.r.t. `a` in `f64`
/// for any [`InnerIouBase`] variant.
pub(crate) fn inner_iou_loss_box_f64(
    a: [f64; 4],
    b: [f64; 4],
    base: InnerIouBase,
    ratio: f64,
) -> (f64, [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4]);
    }
    let (om_in, mut g) = inner_one_minus_iou_f64(a, b, ratio);
    let (pen, gp) = match base {
        InnerIouBase::Iou => (0.0, [0.0; 4]),
        InnerIouBase::Giou => giou_penalty_f64(a, b),
        InnerIouBase::Diou => diou_penalty_f64(a, b),
        InnerIouBase::Ciou => {
            let (pd, gd) = diou_penalty_f64(a, b);
            let (pa, ga, _) = ciou_aspect_penalty_f64(a, b);
            (pd + pa, [gd[0] + ga[0], gd[1] + ga[1], gd[2] + ga[2], gd[3] + ga[3]])
        }
        InnerIouBase::Eiou => {
            if ratio == 1.0 {
                return eiou_box_f64(a, b);
            }
            let full = one_minus_iou_f64(a, b);
            let (le, ge) = eiou_box_f64(a, b);
            (
                (le - full.one_minus_iou).max(0.0),
                [ge[0] - full.grad[0], ge[1] - full.grad[1], ge[2] - full.grad[2], ge[3] - full.grad[3]],
            )
        }
        InnerIouBase::Siou { theta } => {
            let theta64 = theta as f64;
            if ratio == 1.0 {
                return siou_box_f64(a, b, theta64);
            }
            let full = one_minus_iou_f64(a, b);
            let (ls, gs) = siou_box_f64(a, b, theta64);
            (
                (ls - full.one_minus_iou).max(0.0),
                [gs[0] - full.grad[0], gs[1] - full.grad[1], gs[2] - full.grad[2], gs[3] - full.grad[3]],
            )
        }
        InnerIouBase::Mpdiou => mpdiou_penalty_f64(a, b),
    };
    for s in 0..4 {
        g[s] += gp[s];
    }
    (om_in + pen, g)
}

/// Differentiable Inner-IoU overlap (Zhang et al., 2023) between rows of `b1` (predicted) and `b2`
/// (ground truth), both `[N, 4]` `xyxy`, using auxiliary boxes scaled about their centers by
/// `ratio > 0` (typically `0.5..=1.5`), returning `[N]` in `[0, 1]`.
///
/// Evaluates in `f64` with exact closed-form analytical gradients w.r.t. `b1` (`b2` is constant
/// ground truth). At `ratio == 1.0`, equals exact scale-normalized `IoU(b1, b2)`.
pub fn inner_iou(b1: &Tensor, b2: &Tensor, ratio: f32) -> Result<Tensor> {
    let ratio64 = check_ratio(ratio, "inner_iou")?;
    let n = check_box_pair(b1, b2, "inner_iou")?;
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
        let (v, gv) = inner_iou_box_f64(ai, bi, ratio64);
        out[i] = v as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("inner_iou", &[b1], out_t, move |g| {
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

/// Differentiable generalized Inner-IoU box regression loss (Zhang et al., 2023) composing
/// auxiliary-box `1 - IoU_inner(ratio)` with any [`InnerIouBase`] geometric penalty (`IoU`,
/// `GIoU`, `DIoU`, `CIoU`, `EIoU`, `SIoU`, `MPDIoU`), returning `[N]`:
///
/// ```text
/// L_Inner-X = (1 - IoU_inner(ratio)) + R_X
/// ```
///
/// Evaluates in `f64` with exact closed-form analytical gradients w.r.t. `b1` (`b2` is constant
/// ground truth). For [`InnerIouBase::Ciou`], `alpha*` is detached as in standard CIoU.
pub fn inner_iou_loss(b1: &Tensor, b2: &Tensor, base: InnerIouBase, ratio: f32) -> Result<Tensor> {
    let ratio64 = check_ratio(ratio, "inner_iou_loss")?;
    base.validate()?;
    let n = check_box_pair(b1, b2, "inner_iou_loss")?;
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
        let (lv, gv) = inner_iou_loss_box_f64(ai, bi, base, ratio64);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("inner_iou_loss", &[b1], out_t, move |g| {
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

/// Composable Inner-IoU loss wrapper (Zhang et al., 2023 Eq. 8–12):
///
/// ```text
/// L_wrapped = base_loss + (IoU(b1, b2) - IoU_inner(b1, b2, ratio))
/// ```
///
/// Replaces the `(1 - IoU)` term of any differentiable `[N]` base box loss tensor `base_loss` with
/// `(1 - IoU_inner(ratio))`, while preserving gradients through both `base_loss` and `b1`. At
/// `ratio == 1.0`, the delta term is identically `0.0` with zero gradient for all finite boxes.
pub fn wrap_inner_iou_loss(base_loss: &Tensor, b1: &Tensor, b2: &Tensor, ratio: f32) -> Result<Tensor> {
    let ratio64 = check_ratio(ratio, "wrap_inner_iou_loss")?;
    if base_loss.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch {
            op: "wrap_inner_iou_loss",
            expected: DType::F32,
            got: base_loss.dtype(),
        });
    }
    let n = check_box_pair(b1, b2, "wrap_inner_iou_loss")?;
    let dev = b1.device();
    if base_loss.device() != dev {
        return Err(Error::DeviceMismatch { op: "wrap_inner_iou_loss", lhs: base_loss.device(), rhs: dev });
    }
    let n_loss = match base_loss.dims() {
        &[len] => len,
        _ => {
            return Err(invalid(format!(
                "wrap_inner_iou_loss: base_loss must be rank 1 [N], got {}",
                base_loss.shape()
            )));
        }
    };
    if n_loss != n {
        return Err(invalid(format!("wrap_inner_iou_loss: base_loss len {n_loss} != box count {n}")));
    }
    if n == 0 {
        return base_loss.add(&Tensor::zeros_with([0], DType::F32, dev)?);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let mut delta = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        if !ai.iter().chain(&bi).all(|v| v.is_finite()) {
            delta[i] = f32::NAN;
            g_box[i] = [f64::NAN; 4];
            continue;
        }
        if ratio64 == 1.0 {
            continue;
        }
        // IoU - IoU_inner = (1 - IoU_inner) - (1 - IoU), avoiding cancellation near IoU == 1.
        let full = one_minus_iou_f64(ai, bi);
        let (om_in, g_in) = inner_one_minus_iou_f64(ai, bi, ratio64);
        delta[i] = (om_in - full.one_minus_iou) as f32;
        for s in 0..4 {
            g_box[i][s] = g_in[s] - full.grad[s];
        }
    }
    let delta_t = Tensor::from_vec(delta, [n])?.to_device(dev)?;
    let diff = Tensor::custom_op("inner_iou_delta", &[b1], delta_t, move |g| {
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
    })?;
    base_loss.add(&diff)
}
