//! Normalized Wasserstein Distance (NWD, Wang et al. 2021) and Wise-IoU v1 / v3 (Tong et al. 2023)
//! bounding-box regression losses with exact closed-form `f64` analytical gradients.
//!
//! - **NWD** ([`nwd_loss`], [`nwd_loss_with_c`], [`crate::loss::BoxLossKind::Nwd`]): models axis-aligned
//!   boxes as 2D Gaussians `N((cx, cy), diag(w²/4, h²/4))` using signed extents `w = x2 - x1`, `h = y2 - y1`
//!   (matching [`crate::assign::nwd_box`] so inverted predictions receive restoring gradients toward ordered
//!   target corners) and evaluates `1 - exp(-sqrt(W_2²) / c)` with scale-normalized `f64` distance (`0.0` loss
//!   and `0.0` gradient on identical boxes, smooth non-vanishing gradients on tiny and non-overlapping boxes,
//!   overflow-free across `1e-30..1e30`).
//! - **Wise-IoU v1** ([`wiou_v1_loss`], [`crate::loss::BoxLossKind::WiouV1`]): distance-attended IoU loss
//!   `R_WIoU * (1 - IoU)` with `R_WIoU = exp(rho² / (cw² + ch²)*)` and detached enclosing-box diagonal
//!   `c2*` to eliminate enclosing-box expansion hindrance.
//! - **Wise-IoU v3** ([`wiou_v3_loss`], [`crate::loss::BoxLossKind::WiouV3`]): multiplies WIoU v1 by the
//!   detached non-monotonic dynamic focusing gain `r = beta / (delta * alpha^(beta - delta))` where
//!   `beta = L_IoU* / max(iou_mean, 1e-4)` tracks the running exponential moving average `L_IoU_bar`.

use folo_core::{DType, Result, Tensor};

use super::box_loss::{check_box_pair, dfl_box_from_logits};
use super::geom_loss::one_minus_iou_f64;
use super::siou_eiou::enclosing_axis;
use crate::loss::invalid;

/// Single-box Normalized Wasserstein Distance forward `(loss, dL/d(ax1, ay1, ax2, ay2), dL/dc)` in `f64`
/// (Wang et al. 2021, arXiv:2110.13389):
///
/// ```text
/// W_2²(N_a, N_b) = (cx_a - cx_b)² + (cy_a - cy_b)² + ((w_a - w_b)² + (h_a - h_b)²) / 4
///                = ((ax1 - bx1)² + (ay1 - by1)² + (ax2 - bx2)² + (ay2 - by2)²) / 2
/// L_NWD = 1 - exp(-sqrt(W_2²) / c)
/// ```
///
/// Uses signed box extents `w = x2 - x1`, `h = y2 - y1` (matching [`crate::assign::nwd_box`] and the GPU
/// `det_box.wgsl` kernel), so valid boxes match the exact 2D Gaussian Wasserstein metric while inverted
/// predictions receive smooth restoring gradients toward ordered target corners. Evaluated with `hypot`-style
/// scale normalization so tiny (`1e-30`) and huge (`1e30`) boxes never underflow or overflow, using
/// `-expm1(-W_2 / c)` to avoid catastrophic cancellation near `a == b`. By antisymmetry, `dL/db = -dL/da`.
pub(crate) fn nwd_box_f64(a: [f64; 4], b: [f64; 4], c: f64) -> (f64, [f64; 4], f64) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) || !c.is_finite() || c <= 0.0 {
        return (f64::NAN, [f64::NAN; 4], f64::NAN);
    }
    let (dx1, dy1, dx2, dy2) = (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]);
    let m = dx1.abs().max(dy1.abs()).max(dx2.abs()).max(dy2.abs());
    if m == 0.0 {
        return (0.0, [0.0; 4], 0.0);
    }
    if !m.is_finite() {
        return (f64::NAN, [f64::NAN; 4], f64::NAN);
    }
    let inv_m = 1.0 / m;
    let (ux1, uy1, ux2, uy2) = (dx1 * inv_m, dy1 * inv_m, dx2 * inv_m, dy2 * inv_m);
    // r = sqrt(W_2²) / m lies strictly in [sqrt(0.5), sqrt(2.0)].
    let r = (0.5 * (ux1 * ux1 + uy1 * uy1 + ux2 * ux2 + uy2 * uy2)).sqrt();
    let w2 = m * r;
    let u = w2 / c;
    let loss = -(-u).exp_m1();
    let e = (-u).exp();
    // dL/da_k = (exp(-u) / c) * (d_k / (2 * W_2)) = (exp(-u) / (2 * c * r)) * u_k.
    // dL/dc   = -exp(-u) * (W_2 / c²) = -(exp(-u) * u) / c.
    let (coeff, grad_c) = if e == 0.0 { (0.0, 0.0) } else { (e / (2.0 * c * r), -(e * u) / c) };
    let grad = [coeff * ux1, coeff * uy1, coeff * ux2, coeff * uy2];
    (loss, grad, grad_c)
}

/// Smoothed NWD `1 - exp(-sqrt(W_2² + 1e-6) / c)` matching GPU `det_box.wgsl` / `detect_common.wgsl`
/// bit-for-bit on existing `LossConfig::nwd_weight > 0` blends (including near `W_2 -> 0`).
pub(crate) fn nwd_box_eps_f64(a: [f64; 4], b: [f64; 4], c: f64) -> (f64, [f64; 4]) {
    const EPS: f64 = 1e-6;
    if !a.iter().chain(&b).all(|v| v.is_finite()) || !c.is_finite() || c <= 0.0 {
        return (f64::NAN, [f64::NAN; 4]);
    }
    let (dx1, dy1, dx2, dy2) = (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]);
    let m = dx1.abs().max(dy1.abs()).max(dx2.abs()).max(dy2.abs()).max(1e-3);
    if !m.is_finite() {
        return (f64::NAN, [f64::NAN; 4]);
    }
    let inv_m = 1.0 / m;
    let (ux1, uy1, ux2, uy2) = (dx1 * inv_m, dy1 * inv_m, dx2 * inv_m, dy2 * inv_m);
    let r = (0.5 * (ux1 * ux1 + uy1 * uy1 + ux2 * ux2 + uy2 * uy2) + EPS * (inv_m * inv_m)).sqrt();
    let dist = m * r;
    let u = dist / c;
    let loss = -(-u).exp_m1();
    let e = (-u).exp();
    let coeff = if e == 0.0 { 0.0 } else { e / (2.0 * c * r) };
    (loss, [coeff * ux1, coeff * uy1, coeff * ux2, coeff * uy2])
}

/// GPU-parity `nwd_weight` blend loss (`1 - exp(-sqrt(W_2² + 1e-6) / c)`) used by [`super::loss_with_assignment`]
/// so CPU and fused `det_box.wgsl` remain parity-matched even when `pd` approaches `tb`.
pub(super) fn nwd_blend_loss_eps(b1: &Tensor, b2: &Tensor, c: &Tensor) -> Result<Tensor> {
    let n = check_box_pair(b1, b2, "nwd_blend_loss_eps")?;
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let cv = c.to_vec::<f32>()?;
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv) = nwd_box_eps_f64(ai, bi, cv[i] as f64);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("nwd_blend_loss_eps", &[b1], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let mut ga = vec![0.0f32; 4 * n];
        for i in 0..n {
            let gi = gv[i] as f64;
            if gi == 0.0 && g_box[i].iter().all(|v| v.is_finite()) {
                continue;
            }
            for s in 0..4 {
                ga[4 * i + s] = (gi * g_box[i][s]) as f32;
            }
        }
        Ok(vec![Some(Tensor::from_vec(ga, [n, 4])?.to_device(dev)?)])
    })
}

/// Differentiable Normalized Wasserstein Distance loss `1 - exp(-sqrt(W_2²) / c)` between rows of `b1`
/// and `b2` (`[N, 4]`, `xyxy`), with per-row positive scale constant `c` (`[N]`, in the boxes' coordinate
/// units), returning `[N]`.
///
/// Uses signed box extents `w = x2 - x1`, `h = y2 - y1` so inverted predictions receive restoring gradients
/// toward ordered target corners. Evaluates in `f64` with exact analytical gradients flowing into `b1`,
/// `b2` (`dL/db2 = -dL/db1`), and `c`. Errors on shape / dtype / device mismatches or if any entry of `c`
/// is non-finite or `<= 0`.
pub fn nwd_loss(b1: &Tensor, b2: &Tensor, c: &Tensor) -> Result<Tensor> {
    let n = check_box_pair(b1, b2, "nwd_loss")?;
    if c.dtype() != DType::F32 {
        return Err(folo_core::Error::DTypeMismatch { op: "nwd_loss", expected: DType::F32, got: c.dtype() });
    }
    if c.device() != b1.device() {
        return Err(folo_core::Error::DeviceMismatch { op: "nwd_loss", lhs: b1.device(), rhs: c.device() });
    }
    let nc = match c.dims() {
        &[nc] => nc,
        _ => return Err(invalid(format!("nwd_loss: c must be rank 1 [N], got {}", c.shape()))),
    };
    if nc != n {
        return Err(invalid(format!("nwd_loss: c length {nc} does not match box row count {n}")));
    }
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let cv = c.to_vec::<f32>()?;
    for (i, &ci) in cv.iter().enumerate() {
        if !(ci.is_finite() && ci > 0.0) {
            return Err(invalid(format!("nwd_loss: c[{i}] must be finite and > 0, got {ci}")));
        }
    }
    nwd_loss_inner(b1, b2, Some(c), &cv, "nwd_loss")
}

/// Differentiable Normalized Wasserstein Distance loss `1 - exp(-sqrt(W_2²) / c)` between rows of `b1`
/// and `b2` (`[N, 4]`, `xyxy`) with a shared positive scalar constant `c`, returning `[N]`.
///
/// Uses signed box extents `w = x2 - x1`, `h = y2 - y1` so inverted predictions receive restoring gradients
/// toward ordered target corners. Errors if `c` is non-finite or `<= 0`, or on box shape / dtype / device
/// mismatches.
pub fn nwd_loss_with_c(b1: &Tensor, b2: &Tensor, c: f32) -> Result<Tensor> {
    if !(c.is_finite() && c > 0.0) {
        return Err(invalid(format!("nwd_loss_with_c: c must be finite and > 0, got {c}")));
    }
    let n = check_box_pair(b1, b2, "nwd_loss_with_c")?;
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, b1.device());
    }
    let cv = vec![c; n];
    nwd_loss_inner(b1, b2, None, &cv, "nwd_loss_with_c")
}

fn nwd_loss_inner(
    b1: &Tensor,
    b2: &Tensor,
    c_tensor: Option<&Tensor>,
    cv: &[f32],
    op_name: &'static str,
) -> Result<Tensor> {
    let n = cv.len();
    let dev = b1.device();
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    let mut g_c = vec![0.0f64; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv, gc) = nwd_box_f64(ai, bi, cv[i] as f64);
        out[i] = lv as f32;
        g_box[i] = gv;
        g_c[i] = gc;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    let need_a = b1.requires_grad();
    let need_b = b2.requires_grad();
    let need_c = c_tensor.is_some_and(Tensor::requires_grad);
    let has_c = c_tensor.is_some();
    let mut inputs: Vec<&Tensor> = Vec::with_capacity(3);
    inputs.push(b1);
    inputs.push(b2);
    if let Some(ct) = c_tensor {
        inputs.push(ct);
    }
    Tensor::custom_op(op_name, &inputs, out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let grad_a = if need_a {
            let mut ga = vec![0.0f32; 4 * n];
            for i in 0..n {
                let gi = gv[i] as f64;
                if gi == 0.0 && g_box[i].iter().all(|v| v.is_finite()) {
                    continue;
                }
                for s in 0..4 {
                    ga[4 * i + s] = (gi * g_box[i][s]) as f32;
                }
            }
            Some(Tensor::from_vec(ga, [n, 4])?.to_device(dev)?)
        } else {
            None
        };
        let grad_b = if need_b {
            let mut gb = vec![0.0f32; 4 * n];
            for i in 0..n {
                let gi = gv[i] as f64;
                if gi == 0.0 && g_box[i].iter().all(|v| v.is_finite()) {
                    continue;
                }
                for s in 0..4 {
                    gb[4 * i + s] = -(gi * g_box[i][s]) as f32;
                }
            }
            Some(Tensor::from_vec(gb, [n, 4])?.to_device(dev)?)
        } else {
            None
        };
        let mut grads = vec![grad_a, grad_b];
        if has_c {
            let grad_c = if need_c {
                let mut gc_out = vec![0.0f32; n];
                for i in 0..n {
                    let gi = gv[i] as f64;
                    if gi == 0.0 && g_c[i].is_finite() {
                        continue;
                    }
                    gc_out[i] = (gi * g_c[i]) as f32;
                }
                Some(Tensor::from_vec(gc_out, [n])?.to_device(dev)?)
            } else {
                None
            };
            grads.push(grad_c);
        }
        Ok(grads)
    })
}

/// Weighted `NWD` scalar directly from DFL `logits: [N, 4, R]` with per-anchor scale constant `cs`.
#[allow(clippy::too_many_arguments)]
pub(super) fn nwd_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    cs: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
) -> Result<Tensor> {
    let mut idx = 0usize;
    dfl_box_from_logits("nwd_from_logits", logits, pd_t, tbox, t_scores, lams, tss, box_gain, |a, b| {
        let c = cs[idx] as f64;
        idx += 1;
        let (lv, gv, _) = nwd_box_f64(a, b, c);
        (lv, gv)
    })
}

/// Single-box Wise-IoU v1 forward `(loss, dL/d(ax1, ay1, ax2, ay2), L_IoU)` in `f64`
/// (Tong et al. 2023, arXiv:2301.10051 §III-A):
///
/// ```text
/// L_IoU     = 1 - IoU(a, b)
/// R_WIoU    = exp(((cx_a - cx_b)² + (cy_a - cy_b)²) / (cw² + ch²)*)   in [1, e]
/// L_WIoU_v1 = R_WIoU * L_IoU
/// ```
///
/// The enclosing-box diagonal `(cw² + ch²)*` is detached from the analytical gradient so updates never
/// suffer enclosing-box expansion hindrance. Strictly scale-invariant across `1e-30..1e30`, with exact
/// `0.0` loss and `0.0` gradient on identical valid boxes. By symmetry, `dL/db` is `wiou_v1_box_f64(b, a).1`.
pub(crate) fn wiou_v1_box_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4], f64) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4], f64::NAN);
    }
    let iou = one_minus_iou_f64(a, b);
    let l_iou = iou.one_minus_iou;
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);
    let (cw, _, _) = enclosing_axis(ax1, ax2, bx1, bx2);
    let (ch, _, _) = enclosing_axis(ay1, ay2, by1, by2);
    let dcx = 0.5 * ((ax1 - bx1) + (ax2 - bx2));
    let dcy = 0.5 * ((ay1 - by1) + (ay2 - by2));
    let m_c = cw.max(ch);
    let (r_dist, gx_dist, gy_dist) = if m_c <= 0.0 {
        (1.0, 0.0, 0.0)
    } else {
        let inv_c = 1.0 / m_c;
        let (dcxs, dcys) = (dcx * inv_c, dcy * inv_c);
        let (cws, chs) = (cw * inv_c, ch * inv_c);
        let c2s = cws * cws + chs * chs; // strictly in [1.0, 2.0]
        let ratio = ((dcxs * dcxs + dcys * dcys) / c2s).min(1.0);
        let r = ratio.exp(); // strictly in [1.0, e]
        // With detached c2* = cw² + ch²: d(rho² / c2*) / d(ax1) = d(rho² / c2*) / d(ax2) = dcx / c2*.
        let gx = (dcxs * inv_c) / c2s;
        let gy = (dcys * inv_c) / c2s;
        (r, gx, gy)
    };
    let loss = r_dist * l_iou;
    let dist_scale = l_iou * r_dist;
    let grad = [
        r_dist * iou.grad[0] + dist_scale * gx_dist,
        r_dist * iou.grad[1] + dist_scale * gy_dist,
        r_dist * iou.grad[2] + dist_scale * gx_dist,
        r_dist * iou.grad[3] + dist_scale * gy_dist,
    ];
    (loss, grad, l_iou)
}

/// Non-monotonic dynamic focusing gain `r = beta / (delta * alpha^(beta - delta))` evaluated in log-space
/// so large-`beta` outliers monotonically decay smoothly to `0.0` (never reversing via an exponent floor).
#[inline]
pub(crate) fn wiou_v3_focus_gain(l_iou: f64, alpha: f64, delta: f64, iou_mean: f64) -> f64 {
    let beta = l_iou / iou_mean.max(1e-4);
    if beta <= 0.0 {
        return 0.0;
    }
    let log_r = beta.ln() - delta.ln() - (beta - delta) * alpha.ln();
    // Cap only positive overflow (for extreme user parameters); leave negative tail unclamped so
    // outlier attenuation stays strictly monotonic all the way to 0.0.
    log_r.min(80.0).exp()
}

/// Single-box Wise-IoU v3 forward `(loss, dL/d(ax1, ay1, ax2, ay2), L_IoU)` in `f64`
/// (Tong et al. 2023, arXiv:2301.10051 §III-B):
///
/// ```text
/// beta      = L_IoU* / max(iou_mean, 1e-4)
/// r         = beta / (delta * alpha^(beta - delta))     (detached non-monotonic focusing gain)
/// L_WIoU_v3 = r * L_WIoU_v1
/// ```
///
/// Both the enclosing diagonal `(cw² + ch²)*` and the focusing gain `r` are detached from the gradient.
pub(crate) fn wiou_v3_box_f64(
    a: [f64; 4],
    b: [f64; 4],
    alpha: f64,
    delta: f64,
    iou_mean: f64,
) -> (f64, [f64; 4], f64) {
    let (l_v1, g_v1, l_iou) = wiou_v1_box_f64(a, b);
    if !l_v1.is_finite() {
        return (f64::NAN, [f64::NAN; 4], f64::NAN);
    }
    let r = wiou_v3_focus_gain(l_iou, alpha, delta, iou_mean);
    (r * l_v1, [r * g_v1[0], r * g_v1[1], r * g_v1[2], r * g_v1[3]], l_iou)
}

/// Differentiable Wise-IoU v1 loss (Tong et al. 2023 §III-A) between rows of `b1` (predicted) and `b2`
/// (target), both `[N, 4]` `xyxy`, returning `[N]`:
///
/// ```text
/// L_WIoU_v1 = exp(((cx1 - cx2)² + (cy1 - cy2)²) / (cw² + ch²)*) * (1 - IoU(b1, b2))
/// ```
///
/// Evaluates in `f64` with closed-form analytical gradients flowing into both `b1` and `b2` while detaching
/// the enclosing diagonal `(cw² + ch²)*`.
pub fn wiou_v1_loss(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let n = check_box_pair(b1, b2, "wiou_v1_loss")?;
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let need_a = b1.requires_grad();
    let need_b = b2.requires_grad();
    let mut out = vec![0.0f32; n];
    let mut g_a = if need_a { vec![[0.0f64; 4]; n] } else { Vec::new() };
    let mut g_b = if need_b { vec![[0.0f64; 4]; n] } else { Vec::new() };
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, ga, _) = wiou_v1_box_f64(ai, bi);
        out[i] = lv as f32;
        if need_a {
            g_a[i] = ga;
        }
        if need_b {
            g_b[i] = wiou_v1_box_f64(bi, ai).1;
        }
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("wiou_v1_loss", &[b1, b2], out_t, move |g| {
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

/// Differentiable Wise-IoU v3 loss (Tong et al. 2023 §III-B) between rows of `b1` (predicted) and `b2`
/// (target), both `[N, 4]` `xyxy`, returning `[N]`:
///
/// ```text
/// L_IoU     = 1 - IoU(b1, b2)
/// R_WIoU    = exp(((cx1 - cx2)² + (cy1 - cy2)²) / (cw² + ch²)*)
/// beta      = L_IoU* / max(iou_mean, 1e-4)
/// r         = beta / (delta * alpha^(beta - delta))     (detached non-monotonic focusing gain)
/// L_WIoU_v3 = r * R_WIoU * L_IoU
/// ```
///
/// `iou_mean` is the running exponential moving average `L_IoU_bar` across training steps (initialized to
/// `1.0`). Evaluates in `f64` with closed-form analytical gradients flowing into both `b1` and `b2` while
/// detaching `(cw² + ch²)*` and `r` so gradients never suffer enclosing-box hindrance or outlier amplification.
pub fn wiou_v3_loss(b1: &Tensor, b2: &Tensor, alpha: f32, delta: f32, iou_mean: f32) -> Result<Tensor> {
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
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let (a64, d64, m64) = (alpha as f64, delta as f64, iou_mean as f64);
    let need_a = b1.requires_grad();
    let need_b = b2.requires_grad();
    let mut out = vec![0.0f32; n];
    let mut g_a = if need_a { vec![[0.0f64; 4]; n] } else { Vec::new() };
    let mut g_b = if need_b { vec![[0.0f64; 4]; n] } else { Vec::new() };
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, ga, _) = wiou_v3_box_f64(ai, bi, a64, d64, m64);
        out[i] = lv as f32;
        if need_a {
            g_a[i] = ga;
        }
        if need_b {
            g_b[i] = wiou_v3_box_f64(bi, ai, a64, d64, m64).1;
        }
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("wiou_v3_loss", &[b1, b2], out_t, move |g| {
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

/// Weighted `WIoU v1` scalar directly from DFL `logits: [N, 4, R]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn wiou_v1_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
) -> Result<Tensor> {
    dfl_box_from_logits("wiou_v1_from_logits", logits, pd_t, tbox, t_scores, lams, tss, box_gain, |a, b| {
        let (lv, gv, _) = wiou_v1_box_f64(a, b);
        (lv, gv)
    })
}

/// Weighted `WIoU v3` scalar directly from DFL `logits: [N, 4, R]`, also returning the batch's unweighted
/// mean `L_IoU` (`None` when `N == 0`) so the trainer can update its running EMA `L_IoU_bar`.
#[allow(clippy::too_many_arguments)]
pub(super) fn wiou_v3_from_logits(
    logits: &Tensor,
    pd_t: &Tensor,
    tbox: &[f32],
    t_scores: &[f32],
    lams: Option<&[f32]>,
    tss: f32,
    box_gain: f32,
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
    let (a64, d64, m64) = (alpha as f64, delta as f64, iou_mean as f64);
    let mut sum_l_iou = 0.0f64;
    let mut count = 0usize;
    let loss = dfl_box_from_logits(
        "wiou_v3_from_logits",
        logits,
        pd_t,
        tbox,
        t_scores,
        lams,
        tss,
        box_gain,
        |a, b| {
            let (lv, gv, l_iou) = wiou_v3_box_f64(a, b, a64, d64, m64);
            sum_l_iou += l_iou;
            count += 1;
            (lv, gv)
        },
    )?;
    let mean_iou = (count > 0).then(|| (sum_l_iou / count as f64) as f32);
    Ok((loss, mean_iou))
}
