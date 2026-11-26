//! Powerful-IoU v1 & v2 (Liu et al., Neural Networks 2024) bounding-box regression losses with exact
//! closed-form `f64` analytical gradients.
//!
//! - **PIoU v1** ([`piou_loss`]): replaces enclosing-box penalties (which suffer from anchor-box expansion)
//!   with a target-size-adapted edge penalty `P` and smooth saturating factor `f(P) = 1 - exp(-P²)`:
//!   ```text
//!   P       = 0.25 * ((|ax1 - bx1| + |ax2 - bx2|) / w_t + (|ay1 - by1| + |ay2 - by2|) / h_t)
//!   L_PIoU  = (1 - IoU) + (1 - exp(-P²))
//!   ```
//! - **PIoU v2** ([`piou2_loss`]): applies FOLO's detached non-monotonic attention focusing factor
//!   `u(lambda * q) = 3 * (lambda * q) * exp(-(lambda * q)²)` over anchor quality `q = exp(-P)` (with
//!   hyperparameter `lambda > 0`, paper default `1.3`):
//!   ```text
//!   L_PIoU2 = u(lambda * q)* * L_PIoU
//!   ```
//!   Unlike reference implementations that differentiate through `u`, FOLO detaches `u(lambda * q)*`
//!   (matching [`super::wiou_v3_loss`]) so non-saturated boxes always receive a descent gradient toward the
//!   target (never repelling high-`P` boxes toward `P -> +inf` where `u -> 0`) while peaking gradient
//!   attention on medium-quality anchors around `q = 1 / (lambda * sqrt(2))`.
//!
//! Both losses share the scale-normalized `1 - IoU` base term from [`super::geom_loss`] and guard all
//! target side lengths and exponential tails explicitly so that:
//! - identical valid boxes at any positive scale yield exact `0.0` loss and `0.0` gradient,
//! - every term is strictly scale-invariant across normal `f32` scales (`1e-30..1e30`),
//! - degenerate (zero-area, collinear, nested, disjoint, inverted) boxes produce finite losses and gradients
//!   in `f64` (distant boxes with `P >= 30` saturate cleanly to `f(P) = 1`, `f'(P) = 0`; subnormal targets
//!   with `w_t < 1e-38` can still overflow the final `f32` gradient cast),
//! - `NaN` and `±Inf` inputs propagate `NaN` cleanly through both forward and backward.

use folo_core::{DType, Result, Tensor};

use super::box_loss::check_box_pair;
use super::geom_loss::one_minus_iou_f64;
use crate::loss::invalid;

/// Threshold above which `exp(-x²)` underflows to exact `0.0` in IEEE-754 `f64` (`exp(-900) < 1e-390`),
/// preventing `x * x` overflow and `Inf * 0 = NaN` products on extreme-scale or tiny-target boxes.
const EXP_SQ_CUTOFF: f64 = 30.0;

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

/// Non-monotonic PIoU v2 attention focusing factor `u(lambda * q) = 3 * x * exp(-x²)` in `f64`, where
/// `x = lambda * q` for anchor quality `q = exp(-P)` in `[0, 1]` and focusing hyperparameter `lambda > 0`.
///
/// Peaks at `x = 1 / sqrt(2)` with maximum value `3 / sqrt(2 * e) ≈ 1.2866`, decaying smoothly toward `0`
/// for both low-quality outliers (`q -> 0`) and extreme `x >> 1`. Returns `NaN` on invalid inputs.
#[inline]
pub fn piou_attention_f64(q: f64, lambda: f64) -> f64 {
    if !(q.is_finite() && (0.0..=1.0).contains(&q) && lambda.is_finite() && lambda > 0.0) {
        return f64::NAN;
    }
    if q == 0.0 {
        return 0.0;
    }
    let x = lambda * q;
    if x >= EXP_SQ_CUTOFF {
        return 0.0;
    }
    3.0 * x * (-x * x).exp()
}

/// Internal `P`-domain helper evaluating [`piou_attention_f64`] at `q = exp(-P)` for `P >= 0` (including `+Inf`).
#[inline]
pub(crate) fn piou_attention_from_p_f64(p: f64, lambda: f64) -> f64 {
    if p == f64::INFINITY && lambda.is_finite() && lambda > 0.0 {
        return 0.0;
    }
    if !(p.is_finite() && p >= 0.0 && lambda.is_finite() && lambda > 0.0) {
        return f64::NAN;
    }
    piou_attention_f64((-p).exp(), lambda)
}

/// Non-monotonic PIoU v2 attention focusing factor `u(lambda * q) = 3 * (lambda * q) * exp(-(lambda * q)²)`
/// for anchor quality `q = exp(-P)` in `[0, 1]` and focusing hyperparameter `lambda > 0` (paper default `1.3`).
#[inline]
pub fn piou_attention(q: f32, lambda: f32) -> f32 {
    piou_attention_f64(q as f64, lambda as f64) as f32
}

/// Single-box Powerful-IoU v1 forward `(loss, dL/d(ax1, ay1, ax2, ay2), P)` in `f64` (Liu et al. 2024):
///
/// ```text
/// P      = 0.25 * ((|ax1 - bx1| + |ax2 - bx2|) / w_t + (|ay1 - by1| + |ay2 - by2|) / h_t)
/// f(P)   = 1 - exp(-P²)
/// L_PIoU = (1 - IoU) + f(P)
/// ```
///
/// Target widths/heights `<= 0` are guarded explicitly (zero penalty and zero gradient along degenerate target
/// axes). On exact side alignment (`ax1 == bx1`, etc.), `|ax1 - bx1|` uses its symmetric subgradient
/// `sgn0(0) = 0`, matching central finite differences.
pub(crate) fn piou_box_f64(a: [f64; 4], b: [f64; 4]) -> (f64, [f64; 4], f64) {
    if !a.iter().chain(&b).all(|v| v.is_finite()) {
        return (f64::NAN, [f64::NAN; 4], f64::NAN);
    }
    let iou = one_minus_iou_f64(a, b);
    let mut g = iou.grad;
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[2], b[3]);

    let wt = (bx2 - bx1).max(0.0);
    let ht = (by2 - by1).max(0.0);
    let (dx1, dx2) = (ax1 - bx1, ax2 - bx2);
    let (dy1, dy2) = (ay1 - by1, ay2 - by2);

    // Divide each edge distance by target extent before adding to avoid intermediate overflow.
    let px = if wt > 0.0 { (dx1.abs() / wt) + (dx2.abs() / wt) } else { 0.0 };
    let py = if ht > 0.0 { (dy1.abs() / ht) + (dy2.abs() / ht) } else { 0.0 };
    let p = 0.25 * (px + py);

    let (fp, df_dp) = if p <= 0.0 {
        (0.0, 0.0)
    } else if p >= EXP_SQ_CUTOFF {
        (1.0, 0.0)
    } else {
        let p2 = p * p;
        // -expm1(-p²) avoids catastrophic cancellation near p == 0.
        (-(-p2).exp_m1(), 2.0 * p * (-p2).exp())
    };

    if df_dp > 0.0 {
        if wt > 0.0 {
            let cx = 0.25 * df_dp / wt;
            g[0] += cx * sgn0(dx1);
            g[2] += cx * sgn0(dx2);
        }
        if ht > 0.0 {
            let cy = 0.25 * df_dp / ht;
            g[1] += cy * sgn0(dy1);
            g[3] += cy * sgn0(dy2);
        }
    }

    (iou.one_minus_iou + fp, g, p)
}

/// Single-box Powerful-IoU v2 forward `(loss, dL/d(ax1, ay1, ax2, ay2), u)` in `f64` (Liu et al. 2024):
///
/// ```text
/// q       = exp(-P)
/// u       = 3 * (lambda * q) * exp(-(lambda * q)²)     (detached non-monotonic attention factor)
/// L_PIoU2 = u* * L_PIoU
/// ```
///
/// The attention factor `u` is detached from the backward pass so every non-identical box receives a strict
/// descent gradient toward the ground truth while focusing capacity on medium-quality anchors.
pub(crate) fn piou2_box_f64(a: [f64; 4], b: [f64; 4], lambda: f64) -> (f64, [f64; 4], f64) {
    let (l1, g1, p) = piou_box_f64(a, b);
    if !l1.is_finite() {
        return (f64::NAN, [f64::NAN; 4], f64::NAN);
    }
    let u = piou_attention_from_p_f64(p, lambda);
    if !u.is_finite() {
        return (f64::NAN, [f64::NAN; 4], f64::NAN);
    }
    (u * l1, [u * g1[0], u * g1[1], u * g1[2], u * g1[3]], u)
}

/// Differentiable Powerful-IoU v1 (`PIoU`) loss (Liu et al., Neural Networks 2024) between rows of `b1`
/// (predicted) and `b2` (ground truth), both `[N, 4]` `xyxy`, returning `[N]`:
///
/// ```text
/// P      = 0.25 * ((|ax1 - bx1| + |ax2 - bx2|) / w_t + (|ay1 - by1| + |ay2 - by2|) / h_t)
/// L_PIoU = (1 - IoU) + (1 - exp(-P²))
/// ```
///
/// Evaluates in `f64` with exact closed-form analytical gradients w.r.t. `b1` (`b2` is treated as constant
/// ground truth). Scale-invariant across `1e-30..1e30`, with exact `0.0` loss and gradient on identical valid
/// boxes and finite gradients on nested, disjoint, degenerate, zero-area, and inverted boxes.
pub fn piou_loss(b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let n = check_box_pair(b1, b2, "piou_loss")?;
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
        let (lv, gv, _) = piou_box_f64(ai, bi);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("piou_loss", &[b1], out_t, move |g| {
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

/// Differentiable Powerful-IoU v2 (`PIoU2`) loss (Liu et al., Neural Networks 2024) between rows of `b1`
/// (predicted) and `b2` (ground truth), both `[N, 4]` `xyxy`, with non-monotonic attention focusing parameter
/// `lambda > 0` (paper default `1.3`), returning `[N]`:
///
/// ```text
/// q       = exp(-P)
/// u       = 3 * (lambda * q) * exp(-(lambda * q)²)     (detached non-monotonic attention factor)
/// L_PIoU2 = u* * L_PIoU
/// ```
///
/// Evaluates in `f64` with exact closed-form analytical gradients w.r.t. `b1` (`b2` is constant ground truth
/// and `u*` is detached). Errors if `lambda` is non-finite or `<= 0`, or on shape / dtype / device mismatches.
pub fn piou2_loss(b1: &Tensor, b2: &Tensor, lambda: f32) -> Result<Tensor> {
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(invalid(format!("piou2_loss: lambda must be finite and > 0, got {lambda}")));
    }
    let n = check_box_pair(b1, b2, "piou2_loss")?;
    let dev = b1.device();
    if n == 0 {
        return Tensor::zeros_with([0], DType::F32, dev);
    }
    let a = b1.to_vec::<f32>()?;
    let b = b2.to_vec::<f32>()?;
    let lam64 = lambda as f64;
    let mut out = vec![0.0f32; n];
    let mut g_box = vec![[0.0f64; 4]; n];
    for i in 0..n {
        let ai = [a[4 * i] as f64, a[4 * i + 1] as f64, a[4 * i + 2] as f64, a[4 * i + 3] as f64];
        let bi = [b[4 * i] as f64, b[4 * i + 1] as f64, b[4 * i + 2] as f64, b[4 * i + 3] as f64];
        let (lv, gv, _) = piou2_box_f64(ai, bi, lam64);
        out[i] = lv as f32;
        g_box[i] = gv;
    }
    let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
    Tensor::custom_op("piou2_loss", &[b1], out_t, move |g| {
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
