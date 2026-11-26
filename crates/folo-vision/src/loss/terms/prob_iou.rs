//! Gaussian-box bounding-box regression losses for smooth small-object & non-overlapping gradients:
//!
//! - **ProbIoU** ([`prob_iou`], [`prob_iou_loss`], [`prob_iou_l2_loss`]; Llerena et al. 2021,
//!   arXiv:2106.06072): models each axis-aligned box as the exact uniform-moment 2D Gaussian
//!   `N((cx, cy), diag(w²/12, h²/12))`, computes the Bhattacharyya distance `B_D = B_1 + B_2`, Hellinger
//!   distance `H_D = sqrt(1 - exp(-B_D))`, and `ProbIoU = 1 - H_D` (`L_1 = H_D = 1 - ProbIoU`, `L_2 = B_D`).
//!   Strictly scale-invariant across arbitrary box scales.
//! - **GWD** ([`gwd_loss`]; Yang et al. ICML 2021, arXiv:2101.11952): models boxes as
//!   `N((cx, cy), diag(w²/4, h²/4))`, computes squared 2-Wasserstein distance `W_2²`, and maps it through
//!   `f(W_2²) / (tau + f(W_2²))` (`f = ln(1 + W_2²)` or `sqrt(W_2²)`).
//! - **KLD** ([`kld_loss`]; Yang et al. NeurIPS 2021, arXiv:2106.01883): computes the exact asymmetric
//!   Kullback–Leibler divergence `D_KL(N_pred || N_target)` between `N((cx, cy), diag(w²/4, h²/4))`
//!   Gaussians and maps it through `ln(1 + D_KL) / (tau + ln(1 + D_KL))`. Strictly scale-invariant with
//!   automatic small-object center-error amplification proportional to `1 / w_target²`.
//!
//! Every kernel evaluates in `f64` with cancellation-free algebra (`ln_1p`, `expm1`, scale normalization) and
//! exact closed-form analytical gradients flowing into both `b1` and `b2`.

use folo_core::{DType, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::box_loss::check_box_pair;
use crate::loss::invalid;

/// Positive extent floor for degenerate/inverted boxes in ProbIoU and KLD (zero distortion for normal boxes).
const EXTENT_EPS: f64 = 1e-7;

/// Nonlinear transform `f(W_2²)` used by [`gwd_loss`] (Yang et al. ICML 2021).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GwdTransform {
    /// `f(W_2²) = ln(1 + W_2²)`: `C^infinity` smooth everywhere including at `b1 == b2` (default).
    #[default]
    Log1p,
    /// `f(W_2²) = sqrt(W_2²)`: linear in Wasserstein distance near `b1 == b2`.
    Sqrt,
}

/// Cancellation-free `0.5 * ln((w1² + w2²) / (2 * w1 * w2))` and its exact partial derivatives
/// `(d/dw1, d/dw2)` in `f64` for strictly positive `w1, w2`.
#[inline]
fn bhattacharyya_shape_axis_f64(w1: f64, w2: f64) -> (f64, f64, f64) {
    let m = w1.max(w2);
    let u1 = w1 / m;
    let u2 = w2 / m;
    let du = u1 - u2;
    let denom = 2.0 * u1 * u2;
    let val = 0.5 * ((du * du) / denom).ln_1p();
    let s = u1 * u1 + u2 * u2;
    let sum_u = u1 + u2;
    // (w1² - w2²) / (2 * w1 * (w1² + w2²)) = (du * sum_u) / (2 * w1 * s)
    let dw1 = (du * sum_u) / (2.0 * w1 * s);
    let dw2 = -(du * sum_u) / (2.0 * w2 * s);
    (val, dw1, dw2)
}

/// Center term `3 * (dx² / (w1² + w2²))` and partial derivatives `(d/dcx1, d/dcx2, d/dw1, d/dw2)` in `f64`
/// for strictly positive `w1, w2`, scale-normalized against overflow/underflow.
#[inline]
fn bhattacharyya_center_axis_f64(cx1: f64, cx2: f64, w1: f64, w2: f64) -> (f64, f64, f64, f64, f64) {
    let dx = cx1 - cx2;
    let m = w1.max(w2);
    let u1 = w1 / m;
    let u2 = w2 / m;
    let s = u1 * u1 + u2 * u2; // strictly in [1.0, 2.0]
    let d_scaled = dx / m;
    let val = 3.0 * (d_scaled * d_scaled) / s;
    let d_cx1 = (6.0 * d_scaled) / (m * s);
    let d_cx2 = -d_cx1;
    let factor = -6.0 * (d_scaled * d_scaled) / (m * s * s);
    let d_w1 = factor * u1;
    let d_w2 = factor * u2;
    (val, d_cx1, d_cx2, d_w1, d_w2)
}

/// Convert `(cx, cy, w, h)` derivatives to `(x1, y1, x2, y2)` derivatives, zeroing the extent contribution
/// when the box side was clamped at `EXTENT_EPS`.
#[inline]
fn cxcywh_to_xyxy_grad(d_cx: f64, d_cy: f64, d_w: f64, d_h: f64, w_active: bool, h_active: bool) -> [f64; 4] {
    let gw = if w_active { d_w } else { 0.0 };
    let gh = if h_active { d_h } else { 0.0 };
    [0.5 * d_cx - gw, 0.5 * d_cy - gh, 0.5 * d_cx + gw, 0.5 * d_cy + gh]
}

/// Single-box Bhattacharyya distance `B_D` (ProbIoU `L_2` loss) and exact gradients `(B_D, dB_D/da, dB_D/db)`
/// in `f64` (Llerena et al. 2021 Eq. 4 & Eq. 7).
pub(crate) fn prob_iou_l2_box_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4], [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4], [f64::NAN; 4]);
    }
    let (cx1, cy1) = (0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3]));
    let (cx2, cy2) = (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]));
    let raw_w1 = a[2] - a[0];
    let raw_h1 = a[3] - a[1];
    let raw_w2 = b[2] - b[0];
    let raw_h2 = b[3] - b[1];
    let (w1, w1_act) = if raw_w1 > EXTENT_EPS { (raw_w1, true) } else { (EXTENT_EPS, false) };
    let (h1, h1_act) = if raw_h1 > EXTENT_EPS { (raw_h1, true) } else { (EXTENT_EPS, false) };
    let (w2, w2_act) = if raw_w2 > EXTENT_EPS { (raw_w2, true) } else { (EXTENT_EPS, false) };
    let (h2, h2_act) = if raw_h2 > EXTENT_EPS { (raw_h2, true) } else { (EXTENT_EPS, false) };

    let (bx, dcx1, dcx2, dw1_c, dw2_c) = bhattacharyya_center_axis_f64(cx1, cx2, w1, w2);
    let (by, dcy1, dcy2, dh1_c, dh2_c) = bhattacharyya_center_axis_f64(cy1, cy2, h1, h2);
    let (sx, dw1_s, dw2_s) = bhattacharyya_shape_axis_f64(w1, w2);
    let (sy, dh1_s, dh2_s) = bhattacharyya_shape_axis_f64(h1, h2);

    let bd = (bx + by) + (sx + sy);
    let ga = cxcywh_to_xyxy_grad(dcx1, dcy1, dw1_c + dw1_s, dh1_c + dh1_s, w1_act, h1_act);
    let gb = cxcywh_to_xyxy_grad(dcx2, dcy2, dw2_c + dw2_s, dh2_c + dh2_s, w2_act, h2_act);
    (bd, ga, gb)
}

/// Single-box ProbIoU `L_1` loss (`H_D = sqrt(1 - exp(-B_D)) = 1 - ProbIoU`) and exact gradients
/// `(H_D, dH_D/da, dH_D/db)` in `f64` (Llerena et al. 2021 Eq. 6).
pub(crate) fn prob_iou_l1_box_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4], [f64; 4]) {
    let (bd, gbd_a, gbd_b) = prob_iou_l2_box_f64(a, b);
    if !bd.is_finite() {
        return (f64::NAN, [f64::NAN; 4], [f64::NAN; 4]);
    }
    if bd <= 0.0 {
        return (0.0, [0.0; 4], [0.0; 4]);
    }
    let one_minus_bc = -(-bd).exp_m1();
    let hd = one_minus_bc.sqrt();
    if hd == 0.0 {
        return (0.0, [0.0; 4], [0.0; 4]);
    }
    let e = (-bd).exp();
    let scale = if e == 0.0 { 0.0 } else { e / (2.0 * hd) };
    let ga = [scale * gbd_a[0], scale * gbd_a[1], scale * gbd_a[2], scale * gbd_a[3]];
    let gb = [scale * gbd_b[0], scale * gbd_b[1], scale * gbd_b[2], scale * gbd_b[3]];
    (hd, ga, gb)
}

/// Single-box ProbIoU similarity `1 - H_D = exp(-B_D) / (1 + H_D)` and exact gradients
/// `(ProbIoU, dProbIoU/da, dProbIoU/db)` in `f64`.
///
/// Rationalizing `1 - sqrt(1 - exp(-B_D))` as `exp(-B_D) / (1 + H_D)` avoids catastrophic cancellation when
/// `B_D` is large (e.g. `B_D = 37.5` where `H_D` rounds to `1.0` in `f64` while `exp(-B_D) / 2 ≈ 2.59e-17`
/// remains cleanly representable in `f32`).
pub(crate) fn prob_iou_sim_box_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4], [f64; 4]) {
    let (bd, gbd_a, gbd_b) = prob_iou_l2_box_f64(a, b);
    if !bd.is_finite() {
        return (f64::NAN, [f64::NAN; 4], [f64::NAN; 4]);
    }
    if bd <= 0.0 {
        return (1.0, [0.0; 4], [0.0; 4]);
    }
    let one_minus_bc = -(-bd).exp_m1();
    let hd = one_minus_bc.sqrt();
    if hd == 0.0 {
        return (1.0, [0.0; 4], [0.0; 4]);
    }
    let e = (-bd).exp();
    let sim = e / (1.0 + hd);
    let scale = if e == 0.0 { 0.0 } else { -e / (2.0 * hd) };
    let ga = [scale * gbd_a[0], scale * gbd_a[1], scale * gbd_a[2], scale * gbd_a[3]];
    let gb = [scale * gbd_b[0], scale * gbd_b[1], scale * gbd_b[2], scale * gbd_b[3]];
    (sim, ga, gb)
}

/// Single-box GWD loss `f(W_2²) / (tau + f(W_2²))` and exact gradients `(loss, dL/da, dL/db)` in `f64`
/// (Yang et al. ICML 2021 Eq. 7). By antisymmetry, `dL/db = -dL/da`.
pub(crate) fn gwd_box_f64(
    a: [f64; 4],
    b: [f64; 4],
    tau: f64,
    transform: GwdTransform,
) -> (f64, [f64; 4], [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) || !(tau.is_finite() && tau > 0.0) {
        return (f64::NAN, [f64::NAN; 4], [f64::NAN; 4]);
    }
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    let m = d[0].abs().max(d[1].abs()).max(d[2].abs()).max(d[3].abs());
    if m == 0.0 {
        return (0.0, [0.0; 4], [0.0; 4]);
    }
    if !m.is_finite() {
        return (f64::NAN, [f64::NAN; 4], [f64::NAN; 4]);
    }
    let inv_m = 1.0 / m;
    let u = [d[0] * inv_m, d[1] * inv_m, d[2] * inv_m, d[3] * inv_m];
    let r2 = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3]); // in [0.5, 2.0]
    let (f_val, coeff) = match transform {
        GwdTransform::Log1p => {
            // W_2² = m² * r2; evaluate ln(1 + W_2²) safely even for huge m.
            let w2_sq = (m * m) * r2;
            let f = if w2_sq.is_finite() { w2_sq.ln_1p() } else { 2.0 * m.ln() + r2.ln() };
            let denom = tau + f;
            // dL/da_k = (tau / denom²) * (d_k / (1 + W_2²))
            let c = if w2_sq.is_finite() { (tau / (denom * denom)) / (1.0 + w2_sq) } else { 0.0 };
            (f, c)
        }
        GwdTransform::Sqrt => {
            let r = r2.sqrt();
            let f = m * r;
            let denom = tau + f;
            // dL/da_k = (tau / denom²) * (d_k / (2 * W_2)) = (tau / (2 * denom² * m * r)) * d_k
            let c = tau / (2.0 * denom * denom * f);
            (f, c)
        }
    };
    let loss = f_val / (tau + f_val);
    let ga = [coeff * d[0], coeff * d[1], coeff * d[2], coeff * d[3]];
    let gb = [-ga[0], -ga[1], -ga[2], -ga[3]];
    (loss, ga, gb)
}

/// Cancellation-free `0.5 * (r² - 1 - 2 * ln(r))` for `r > 0`:
/// - near `r == 1` (`|r - 1| < 1e-4`), evaluates the exact Taylor series in `r - 1`;
/// - for `r >= 0.5`, uses `ln_1p(r - 1)` to preserve precision around unity;
/// - for `r < 0.5`, evaluates `0.5 * (r² - 1 - 2 * ln(r))` directly so tiny ratios `r << 1` (down to
///   `f64::MIN_POSITIVE`) never round `r - 1` to `-1.0` and blow up `ln_1p`.
#[inline]
fn kld_scale_term_f64(r: f64) -> f64 {
    if r < 0.5 {
        return 0.5 * (r * r - 1.0 - 2.0 * r.ln());
    }
    let delta = r - 1.0;
    if delta.abs() < 1e-4 {
        let d = delta;
        let d2 = d * d;
        d2 * (1.0 + d * (-1.0 / 3.0 + d * (0.25 + d * (-0.2 + d / 6.0))))
    } else {
        0.5 * (delta * delta + 2.0 * (delta - delta.ln_1p()))
    }
}

/// Single-box asymmetric KLD loss `ln(1 + D_KL(N_a || N_b)) / (tau + ln(1 + D_KL(N_a || N_b)))` and exact
/// gradients `(loss, dL/da, dL/db)` in `f64` (Yang et al. NeurIPS 2021 Eq. 4 & Eq. 7).
pub(crate) fn kld_box_f64(a: [f64; 4], b: [f64; 4], tau: f64) -> (f64, [f64; 4], [f64; 4]) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) || !(tau.is_finite() && tau > 0.0) {
        return (f64::NAN, [f64::NAN; 4], [f64::NAN; 4]);
    }
    let (cx1, cy1) = (0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3]));
    let (cx2, cy2) = (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]));
    let raw_w1 = a[2] - a[0];
    let raw_h1 = a[3] - a[1];
    let raw_w2 = b[2] - b[0];
    let raw_h2 = b[3] - b[1];
    let (w1, w1_act) = if raw_w1 > EXTENT_EPS { (raw_w1, true) } else { (EXTENT_EPS, false) };
    let (h1, h1_act) = if raw_h1 > EXTENT_EPS { (raw_h1, true) } else { (EXTENT_EPS, false) };
    let (w2, w2_act) = if raw_w2 > EXTENT_EPS { (raw_w2, true) } else { (EXTENT_EPS, false) };
    let (h2, h2_act) = if raw_h2 > EXTENT_EPS { (raw_h2, true) } else { (EXTENT_EPS, false) };

    let dx_w = (cx1 - cx2) / w2;
    let dy_h = (cy1 - cy2) / h2;
    let rx = w1 / w2;
    let ry = h1 / h2;

    let d_kl = 2.0 * (dx_w * dx_w + dy_h * dy_h) + kld_scale_term_f64(rx) + kld_scale_term_f64(ry);
    if !d_kl.is_finite() {
        return (1.0, [0.0; 4], [0.0; 4]);
    }
    let u = d_kl.ln_1p();
    let denom = tau + u;
    let loss = u / denom;
    let d_loss_d_kl = (tau / (denom * denom)) / (1.0 + d_kl);

    let dcx1 = 4.0 * dx_w / w2;
    let dcy1 = 4.0 * dy_h / h2;
    let dw1 = (rx - 1.0 / rx) / w2;
    let dh1 = (ry - 1.0 / ry) / h2;

    let dcx2 = -dcx1;
    let dcy2 = -dcy1;
    let dw2 = (1.0 - rx * rx - 4.0 * dx_w * dx_w) / w2;
    let dh2 = (1.0 - ry * ry - 4.0 * dy_h * dy_h) / h2;

    let ga_raw = cxcywh_to_xyxy_grad(dcx1, dcy1, dw1, dh1, w1_act, h1_act);
    let gb_raw = cxcywh_to_xyxy_grad(dcx2, dcy2, dw2, dh2, w2_act, h2_act);
    let ga =
        [d_loss_d_kl * ga_raw[0], d_loss_d_kl * ga_raw[1], d_loss_d_kl * ga_raw[2], d_loss_d_kl * ga_raw[3]];
    let gb =
        [d_loss_d_kl * gb_raw[0], d_loss_d_kl * gb_raw[1], d_loss_d_kl * gb_raw[2], d_loss_d_kl * gb_raw[3]];
    (loss, ga, gb)
}

/// Shared custom-op wrapper for pairwise `[N, 4]` box ops with exact `f64` gradients flowing into both
/// `b1` and `b2`.
fn gaussian_box_op<F>(b1: &Tensor, b2: &Tensor, op: &'static str, eval: F) -> Result<Tensor>
where
    F: Fn([f64; 4], [f64; 4]) -> (f64, [f64; 4], [f64; 4]),
{
    let n = check_box_pair(b1, b2, op)?;
    let dev = b1.device();
    let need_a = b1.requires_grad();
    let need_b = b2.requires_grad();
    if n == 0 {
        let out_t = Tensor::zeros_with([0], DType::F32, dev)?;
        return Tensor::custom_op(op, &[b1, b2], out_t, move |_g| {
            let ga = if need_a { Some(Tensor::zeros_with([0, 4], DType::F32, dev)?) } else { None };
            let gb = if need_b { Some(Tensor::zeros_with([0, 4], DType::F32, dev)?) } else { None };
            Ok(vec![ga, gb])
        });
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let mut out = vec![0.0f32; n];
    let mut g_a = if need_a { vec![[0.0f64; 4]; n] } else { Vec::new() };
    let mut g_b = if need_b { vec![[0.0f64; 4]; n] } else { Vec::new() };
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, ga, gb) = eval(ai, bi);
        out[i] = lv as f32;
        if need_a {
            g_a[i] = ga;
        }
        if need_b {
            g_b[i] = gb;
        }
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op(op, &[b1, b2], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let grad_a = if need_a {
            let mut grad = vec![0.0f32; 4 * n];
            for i in 0..n {
                let gi = gv[i] as f64;
                if gi == 0.0 && g_a[i].iter().all(|v| v.is_finite()) {
                    continue;
                }
                for s in 0..4 {
                    grad[4 * i + s] = (gi * g_a[i][s]) as f32;
                }
            }
            Some(Tensor::from_vec(grad, [n, 4])?.to_device(dev)?)
        } else {
            None
        };
        let grad_b = if need_b {
            let mut grad = vec![0.0f32; 4 * n];
            for i in 0..n {
                let gi = gv[i] as f64;
                if gi == 0.0 && g_b[i].iter().all(|v| v.is_finite()) {
                    continue;
                }
                for s in 0..4 {
                    grad[4 * i + s] = (gi * g_b[i][s]) as f32;
                }
            }
            Some(Tensor::from_vec(grad, [n, 4])?.to_device(dev)?)
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    })
}

/// Differentiable Probabilistic IoU (`ProbIoU = 1 - H_D = exp(-B_D) / (1 + H_D)`, Llerena et al. 2021)
/// between rows of `b1` and `b2` (`[N, 4]`, `xyxy`), returning `[N]` in `[0, 1]`.
pub fn prob_iou(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    gaussian_box_op(b1, b2, "prob_iou", prob_iou_sim_box_f64)
}

/// Differentiable ProbIoU `L_1` regression loss (`1 - ProbIoU = H_D = sqrt(1 - exp(-B_D))`, Llerena et al.
/// 2021 Eq. 6) between rows of `b1` and `b2` (`[N, 4]`, `xyxy`), returning `[N]` in `[0, 1]`.
pub fn prob_iou_loss(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    gaussian_box_op(b1, b2, "prob_iou_loss", prob_iou_l1_box_f64)
}

/// Differentiable ProbIoU `L_2` (Bhattacharyya distance `B_D`, Llerena et al. 2021 Eq. 7) regression loss
/// between rows of `b1` and `b2` (`[N, 4]`, `xyxy`), returning `[N]` in `[0, +inf)`. Smooth (`C^infinity`)
/// even at `b1 == b2`.
pub fn prob_iou_l2_loss(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    gaussian_box_op(b1, b2, "prob_iou_l2_loss", prob_iou_l2_box_f64)
}

/// Differentiable Gaussian Wasserstein Distance (GWD, Yang et al. ICML 2021) regression loss
/// `f(W_2²) / (tau + f(W_2²))` between rows of `b1` and `b2` (`[N, 4]`, `xyxy`), returning `[N]` in `[0, 1)`.
///
/// Uses `tau > 0` (paper default `1.0`) and [`GwdTransform`] (`Log1p` default, or `Sqrt`).
pub fn gwd_loss(b1: &Tensor, b2: &Tensor, tau: f32, transform: GwdTransform) -> Result<Tensor> {
    if !(tau.is_finite() && tau > 0.0) {
        return Err(invalid(format!("gwd_loss: tau must be finite and > 0, got {tau}")));
    }
    let t64 = tau as f64;
    gaussian_box_op(b1, b2, "gwd_loss", |a, b| gwd_box_f64(a, b, t64, transform))
}

/// Differentiable Kullback–Leibler Divergence (KLD, Yang et al. NeurIPS 2021) regression loss
/// `ln(1 + D_KL(N_b1 || N_b2)) / (tau + ln(1 + D_KL(N_b1 || N_b2)))` between predicted boxes `b1` and target
/// boxes `b2` (`[N, 4]`, `xyxy`), returning `[N]` in `[0, 1)`.
///
/// Strictly scale-invariant with automatic small-object gradient emphasis (`tau > 0`, paper default `1.0`).
pub fn kld_loss(b1: &Tensor, b2: &Tensor, tau: f32) -> Result<Tensor> {
    if !(tau.is_finite() && tau > 0.0) {
        return Err(invalid(format!("kld_loss: tau must be finite and > 0, got {tau}")));
    }
    let t64 = tau as f64;
    gaussian_box_op(b1, b2, "kld_loss", |a, b| kld_box_f64(a, b, t64))
}
