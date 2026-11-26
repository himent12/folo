//! EIoU (Efficient-IoU, Zhang et al. 2022) and SIoU (Scylla-IoU, Gevorgyan 2022) bounding-box regression
//! losses with exact closed-form `f64` analytical gradients.
//!
//! Both losses share the scale-normalized `1 - IoU` base term from [`super::geom_loss`] and guard all
//! enclosing-box and side-length denominators explicitly so that:
//! - identical valid boxes at any positive scale yield exact `0.0` loss and `0.0` gradient,
//! - every geometric term is strictly scale-invariant across `1e-30..1e30`,
//! - degenerate (zero-area, collinear, inverted, extreme-scale) boxes produce finite losses and gradients,
//! - `NaN` and `±Inf` inputs propagate `NaN` through both forward and backward.

use folo_core::{DType, Result, Tensor};

use super::box_loss::{check_box_pair, dfl_box_from_logits};
use super::geom_loss::{one_minus_iou_f64, w_max, w_min};
use crate::loss::invalid;

/// Enclosing span `c = max(a1, a2, b1, b2) - min(a1, a2, b1, b2) >= 0` and its exact partial derivatives
/// `(dc/da1, dc/da2)`. Reduces identically to `max(a2, b2) - min(a1, b1)` on valid boxes (`a1 <= a2`,
/// `b1 <= b2`), while guaranteeing that inverted boxes remain enclosed (`|dcx| <= c`) so center-distance and
/// size ratios stay bounded in `[0, 1]` without overflow even on extreme inverted coordinates.
#[inline]
pub(super) fn enclosing_axis(a1: f64, a2: f64, b1: f64, b2: f64) -> (f64, f64, f64) {
    let (a_lo, a_hi) = if a1 <= a2 { (a1, a2) } else { (a2, a1) };
    let (b_lo, b_hi) = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
    let c = a_hi.max(b_hi) - a_lo.min(b_lo);
    if c <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let d_lo = w_min(a_lo, b_lo);
    let d_hi = w_max(a_hi, b_hi);
    let (da_lo_1, da_lo_2) = if a1 < a2 {
        (1.0, 0.0)
    } else if a1 > a2 {
        (0.0, 1.0)
    } else {
        (0.5, 0.5)
    };
    let (da_hi_1, da_hi_2) = if a1 > a2 {
        (1.0, 0.0)
    } else if a1 < a2 {
        (0.0, 1.0)
    } else {
        (0.5, 0.5)
    };
    let dc_da1 = d_hi * da_hi_1 - d_lo * da_lo_1;
    let dc_da2 = d_hi * da_hi_2 - d_lo * da_lo_2;
    (c, dc_da1, dc_da2)
}

/// Single-box EIoU forward `(loss, dL/d(ax1, ay1, ax2, ay2))` in `f64` (Zhang et al. 2022):
///
/// ```text
/// L_EIoU = (1 - IoU) + (dcx² + dcy²) / (cw² + ch²) + (w1 - w2)² / cw² + (h1 - h2)² / ch²
/// ```
///
/// where `dcx = cx1 - cx2`, `dcy = cy1 - cy2`, `w1 = max(ax2 - ax1, 0)`, `h1 = max(ay2 - ay1, 0)`,
/// `w2 = max(bx2 - bx1, 0)`, `h2 = max(by2 - by1, 0)`, and `(cw, ch)` are the enclosing-box extents over all
/// endpoints of `a` and `b`. Zero denominators are guarded explicitly (0 penalty and 0 gradient).
pub(crate) fn eiou_box_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4]);
    }
    let iou = one_minus_iou_f64(a, b);
    let mut g = iou.grad;
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);

    let w1 = (ax2 - ax1).max(0.0);
    let h1 = (ay2 - ay1).max(0.0);
    let w2 = (bx2 - bx1).max(0.0);
    let h2 = (by2 - by1).max(0.0);

    let (cw, dcw_da1, dcw_da2) = enclosing_axis(ax1, ax2, bx1, bx2);
    let (ch, dch_da1, dch_da2) = enclosing_axis(ay1, ay2, by1, by2);

    // Center-distance penalty (dcx² + dcy²) / (cw² + ch²).
    let dcx = 0.5 * ((ax1 - bx1) + (ax2 - bx2));
    let dcy = 0.5 * ((ay1 - by1) + (ay2 - by2));
    let m_c = cw.max(ch);
    let l_dis = if m_c <= 0.0 {
        0.0
    } else {
        let inv_c = 1.0 / m_c;
        let (dcxs, dcys) = (dcx * inv_c, dcy * inv_c);
        let (cws, chs) = (cw * inv_c, ch * inv_c);
        let c2 = cws * cws + chs * chs; // strictly in [1.0, 2.0]
        let dist = (dcxs * dcxs + dcys * dcys) / c2;
        let g_dcx = (dcxs * inv_c) / c2;
        let g_dcy = (dcys * inv_c) / c2;
        let dc_coeff = 2.0 * dist * inv_c / c2;
        g[0] += g_dcx - dc_coeff * cws * dcw_da1;
        g[1] += g_dcy - dc_coeff * chs * dch_da1;
        g[2] += g_dcx - dc_coeff * cws * dcw_da2;
        g[3] += g_dcy - dc_coeff * chs * dch_da2;
        dist
    };

    // Decoupled width penalty ((w1 - w2) / cw)².
    let l_w = if cw <= 0.0 {
        0.0
    } else {
        let rw = (w1 - w2) / cw;
        let pen = rw * rw;
        if rw != 0.0 {
            let inv_cw = 1.0 / cw;
            if ax2 > ax1 {
                let d_w1 = 2.0 * rw * inv_cw;
                g[0] -= d_w1;
                g[2] += d_w1;
            }
            let d_cw = 2.0 * pen * inv_cw;
            g[0] -= d_cw * dcw_da1;
            g[2] -= d_cw * dcw_da2;
        }
        pen
    };

    // Decoupled height penalty ((h1 - h2) / ch)².
    let l_h = if ch <= 0.0 {
        0.0
    } else {
        let rh = (h1 - h2) / ch;
        let pen = rh * rh;
        if rh != 0.0 {
            let inv_ch = 1.0 / ch;
            if ay2 > ay1 {
                let d_h1 = 2.0 * rh * inv_ch;
                g[1] -= d_h1;
                g[3] += d_h1;
            }
            let d_ch = 2.0 * pen * inv_ch;
            g[1] -= d_ch * dch_da1;
            g[3] -= d_ch * dch_da2;
        }
        pen
    };

    (iou.one_minus_iou + l_dis + l_w + l_h, g)
}

/// Differentiable Efficient-IoU (EIoU) loss (Zhang et al. 2022) between rows of `b1` (predicted) and `b2`
/// (ground truth), both `[N, 4]` `xyxy`, returning `[N]`:
///
/// ```text
/// L_EIoU = (1 - IoU) + (dcx² + dcy²) / (cw² + ch²) + (w1 - w2)² / cw² + (h1 - h2)² / ch²
/// ```
///
/// Replaces CIoU's coupled aspect-ratio penalty with explicit per-axis width and height penalties normalized by
/// the enclosing box extents `cw` and `ch`. Evaluates in `f64` with closed-form analytical gradients w.r.t.
/// `b1` (`b2` is treated as constant ground truth).
pub fn eiou_loss(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let n = check_box_pair(b1, b2, "eiou_loss")?;
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
        let (lv, gv) = eiou_box_f64(ai, bi);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("eiou_loss", &[b1], out_t, move |g| {
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

/// Weighted `EIoU` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn eiou_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
) -> Result<Tensor> {
    dfl_box_from_logits("eiou_from_logits", logits, pd_t, tbox, t_scores, lams, tss, box_gain, eiou_box_f64)
}

#[inline]
fn sgn0(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Single-box SIoU forward `(loss, dL/d(ax1, ay1, ax2, ay2))` in `f64` (Gevorgyan 2022, arXiv:2205.12740):
///
/// ```text
/// dx = cx2 - cx1,   dy = cy2 - cy1
/// Lambda = 1 - 2 * sin²(arcsin(min(|dx|, |dy|) / sqrt(dx² + dy²)) - pi/4)
///        = 2 * |dx| * |dy| / (dx² + dy²)              (exact trig identity; 0 when dx == dy == 0)
/// gamma  = 2 - Lambda                                  (in [1, 2])
/// rho_x  = (dx / cw)²,   rho_y = (dy / ch)²
/// Delta  = (1 - exp(-gamma * rho_x)) + (1 - exp(-gamma * rho_y))
/// omega_w = |w1 - w2| / max(w1, w2),   omega_h = |h1 - h2| / max(h1, h2)
/// Omega  = (1 - exp(-omega_w))^theta + (1 - exp(-omega_h))^theta
/// L_SIoU = (1 - IoU) + 0.5 * (Delta + Omega)
/// ```
///
/// The exact algebraic identity `Lambda = 2 |dx| |dy| / (dx² + dy²)` eliminates the `arcsin` branch switch
/// at 45° (where `Lambda` is `C^∞`-smooth with zero normal derivative) and stays numerically exact at every
/// positive scale. On exact horizontal or vertical collinear alignment (`dx == 0` or `dy == 0`), `Lambda`
/// has a symmetric V-shaped cusp whose symmetric subgradient (`sgn(0) = 0`) matches central finite differences.
pub(crate) fn siou_box_f64(a: [f64; 4], b: [f64; 4], theta: f64) -> (f64, [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4]);
    }
    let iou = one_minus_iou_f64(a, b);
    let mut g = iou.grad;
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);

    // Center displacements: dx = cx2 - cx1, dy = cy2 - cy1 (d/dax = -0.5, d/day = -0.5).
    let dx = 0.5 * ((bx1 - ax1) + (bx2 - ax2));
    let dy = 0.5 * ((by1 - ay1) + (by2 - ay2));
    let u = dx.abs();
    let v = dy.abs();
    let m_uv = u.max(v);

    // Angle cost Lambda = 2 * u * v / (u² + v²) and gamma = 2 - Lambda.
    let (gamma, dgamma_ddx, dgamma_ddy) = if m_uv <= 0.0 {
        (2.0, 0.0, 0.0)
    } else {
        let inv_s = 1.0 / m_uv;
        let uh = u * inv_s;
        let vh = v * inv_s;
        let r2 = uh * uh + vh * vh; // strictly in [1.0, 2.0]
        let lambda = (2.0 * uh * vh) / r2;
        let r4 = r2 * r2;
        // dLambda/du = 2 * v * (v² - u²) / (u² + v²)² = 2 * vh * (vh² - uh²) / (m_uv * r4)
        let dlam_du = (2.0 * vh * (vh * vh - uh * uh) * inv_s) / r4;
        let dlam_dv = (2.0 * uh * (uh * uh - vh * vh) * inv_s) / r4;
        (2.0 - lambda, -dlam_du * sgn0(dx), -dlam_dv * sgn0(dy))
    };

    // Enclosing box extents over all endpoints (handles valid and inverted boxes uniformly).
    let (cw, dcw_da1, dcw_da2) = enclosing_axis(ax1, ax2, bx1, bx2);
    let (ch, dch_da1, dch_da2) = enclosing_axis(ay1, ay2, by1, by2);

    // Distance cost 0.5 * Delta = 0.5 * ((1 - exp(-gamma * rho_x)) + (1 - exp(-gamma * rho_y))).
    let (qx, rho_x) = if cw > 0.0 {
        let q = dx / cw;
        (q, q * q)
    } else {
        (0.0, 0.0)
    };
    let (qy, rho_y) = if ch > 0.0 {
        let q = dy / ch;
        (q, q * q)
    } else {
        (0.0, 0.0)
    };
    let gx = gamma * rho_x;
    let gy = gamma * rho_y;
    let ex = (-gx).exp();
    let ey = (-gy).exp();
    let delta_half = 0.5 * (-(-gx).exp_m1() - (-gy).exp_m1());

    // Analytical gradient of 0.5 * Delta w.r.t. (ax1, ay1, ax2, ay2).
    let d_delta_dgamma = rho_x * ex + rho_y * ey;
    let mut d_delta_ddx = d_delta_dgamma * dgamma_ddx;
    let mut d_delta_ddy = d_delta_dgamma * dgamma_ddy;
    if cw > 0.0 && qx != 0.0 {
        let inv_cw = 1.0 / cw;
        let ge = gamma * ex;
        d_delta_ddx += ge * (2.0 * qx * inv_cw);
        let d_cw_half = 0.5 * ge * (2.0 * rho_x * inv_cw);
        g[0] -= d_cw_half * dcw_da1;
        g[2] -= d_cw_half * dcw_da2;
    }
    if ch > 0.0 && qy != 0.0 {
        let inv_ch = 1.0 / ch;
        let ge = gamma * ey;
        d_delta_ddy += ge * (2.0 * qy * inv_ch);
        let d_ch_half = 0.5 * ge * (2.0 * rho_y * inv_ch);
        g[1] -= d_ch_half * dch_da1;
        g[3] -= d_ch_half * dch_da2;
    }
    // d(0.5 * Delta)/dax = 0.5 * dDelta/ddx * (-0.5) = -0.25 * dDelta/ddx.
    let gx_center = -0.25 * d_delta_ddx;
    let gy_center = -0.25 * d_delta_ddy;
    g[0] += gx_center;
    g[2] += gx_center;
    g[1] += gy_center;
    g[3] += gy_center;

    // Shape cost 0.5 * Omega = 0.5 * ((1 - exp(-omega_w))^theta + (1 - exp(-omega_h))^theta).
    let shape_axis = |s1: f64, s2: f64, inv_s: f64, active: bool| -> (f64, f64) {
        let m = s1.max(s2);
        if m <= 0.0 {
            return (0.0, 0.0);
        }
        let omega = (s1 - s2).abs() / m;
        let e = (-omega).exp();
        let om_e = -(-omega).exp_m1();
        let pen = 0.5 * om_e.powf(theta);
        if !active || s1 == s2 {
            return (pen, 0.0);
        }
        let d_omega_ds1 = if s1 > s2 { s2 / (m * m) } else { -1.0 / m };
        let df_ds1 = 0.5 * theta * om_e.powf(theta - 1.0) * e * d_omega_ds1 * inv_s;
        (pen, df_ds1)
    };
    let (pen_w, df_dw1) = shape_axis(iou.w1s, iou.w2s, iou.inv_sx, ax2 > ax1);
    let (pen_h, df_dh1) = shape_axis(iou.h1s, iou.h2s, iou.inv_sy, ay2 > ay1);
    g[0] -= df_dw1;
    g[2] += df_dw1;
    g[1] -= df_dh1;
    g[3] += df_dh1;

    (iou.one_minus_iou + delta_half + pen_w + pen_h, g)
}

/// Differentiable Scylla-IoU (SIoU) loss (Gevorgyan 2022, arXiv:2205.12740) between rows of `b1` (predicted)
/// and `b2` (ground truth), both `[N, 4]` `xyxy`, with shape-cost exponent `theta` in `[2, 6]` (default `4.0`),
/// returning `[N]`:
///
/// ```text
/// L_SIoU = (1 - IoU) + 0.5 * (Delta + Omega)
/// ```
///
/// Incorporates the direction-aware angle cost `Lambda`, angle-modulated distance cost `Delta`, and shape cost
/// `Omega`. Evaluates in `f64` with closed-form analytical gradients w.r.t. `b1` (`b2` is constant ground truth).
pub fn siou_loss(b1: &Tensor, b2: &Tensor, theta: f32) -> Result<Tensor> {
    if !(theta.is_finite() && (2.0..=6.0).contains(&theta)) {
        return Err(invalid(format!("siou_loss: theta must be finite and in 2..=6, got {theta}")));
    }
    let n = check_box_pair(b1, b2, "siou_loss")?;
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let theta64 = theta as f64;
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv) = siou_box_f64(ai, bi, theta64);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("siou_loss", &[b1], out_t, move |g| {
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

/// Weighted `SIoU` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn siou_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
    theta: f32,
) -> Result<Tensor> {
    let theta64 = theta as f64;
    dfl_box_from_logits("siou_from_logits", logits, pd_t, tbox, t_scores, lams, tss, box_gain, move |a, b| {
        siou_box_f64(a, b, theta64)
    })
}
