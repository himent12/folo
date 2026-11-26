//! Asymmetric Loss (ASL, Ridnik et al. ICCV 2021) and PolyLoss (Poly-1 BCE & Poly-1 Focal, Leng et al.
//! ICLR 2022) for dense multi-label object detection classification, with exact closed-form `f64` analytical
//! gradients.
//!
//! Every kernel computes `p = sigmoid(x)` and `q = sigmoid(-x)` directly in `f64` without `1 - p`
//! catastrophic cancellation and factors out `p^gamma` / `q^gamma` / `p_m^gamma_neg` so fractional exponents
//! `0 < gamma < 1` never encounter `0^(gamma - 1) = inf` in the backward pass:
//!
//! - **ASL** (`gamma_pos >= 0`, `gamma_neg >= 0`, probability margin shift `clip = m in [0, 1)`):
//!   ```text
//!   L_+(x) = q^{gamma_pos} * softplus(-x)
//!   L_-(x) = 0                                      if p <= m
//!          = p_m^{gamma_neg} * (-ln(1 - p_m))       if p > m,  p_m = p - m, 1 - p_m = q + m
//!   L_ASL(x, t) = t * L_+(x) + (1 - t) * L_-(x)
//!   ```
//! - **Poly-1 BCE** (`epsilon >= -1`):
//!   ```text
//!   L_Poly1BCE(x, t) = t * (softplus(-x) + eps * q) + (1 - t) * (softplus(x) + eps * p)
//!   ```
//! - **Poly-1 Focal** (`gamma >= 0`, `epsilon >= -1`):
//!   ```text
//!   L_Poly1Focal(x, t) = t * q^gamma * (softplus(-x) + eps * q)
//!                      + (1 - t) * p^gamma * (softplus(x) + eps * p)
//!   ```
//!
//! Non-finite inputs (`NaN`, `±Inf`) propagate `NaN` through both forward and backward.

use folo_core::{DType, Error, Result, Tensor};
use rayon::prelude::*;

use crate::loss::invalid;

/// Rows per parallel task of the CPU fused classification sums.
const PAR_ROWS: usize = 8;

/// Numerically stable `sigmoid(x)` in `f64`.
#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `softplus(x) = ln(1 + exp(x))` in `f64`.
#[inline]
fn softplus_f64(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Exact single-element ASL `(loss, dL/dx)` in `f64` for logit `x`, target `t in [0, 1]`,
/// focusing exponents `gamma_pos >= 0`, `gamma_neg >= 0`, and margin shift `clip in [0, 1)`.
#[inline]
pub(crate) fn asl_elem_f64(x: f64, t: f64, gamma_pos: f64, gamma_neg: f64, clip: f64) -> (f64, f64) {
    if !x.is_finite() || !t.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let p = sigmoid_f64(x);
    let q = sigmoid_f64(-x);

    let (l_pos, g_pos) = if t == 0.0 {
        (0.0, 0.0)
    } else {
        let s_pos = softplus_f64(-x);
        if gamma_pos == 0.0 {
            (s_pos, -q)
        } else if q == 0.0 {
            (0.0, 0.0)
        } else {
            let qg = q.powf(gamma_pos);
            let p_s = if p == 0.0 { 0.0 } else { p * s_pos };
            (qg * s_pos, -qg * (gamma_pos * p_s + q))
        }
    };

    let (l_neg, g_neg) = if t == 1.0 { (0.0, 0.0) } else { asl_neg_f64(x, p, q, gamma_neg, clip) };

    let loss = (if t == 0.0 { 0.0 } else { t * l_pos }) + (if t == 1.0 { 0.0 } else { (1.0 - t) * l_neg });
    let grad = (if t == 0.0 { 0.0 } else { t * g_pos }) + (if t == 1.0 { 0.0 } else { (1.0 - t) * g_neg });
    (loss, grad)
}

/// Negative-only ASL `(L_-(x), dL_-/dx)` in `f64`.
#[inline]
fn asl_neg_f64(x: f64, p: f64, q: f64, gamma_neg: f64, clip: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    if clip == 0.0 {
        let s_neg = softplus_f64(x);
        if gamma_neg == 0.0 {
            return (s_neg, p);
        }
        if p == 0.0 {
            return (0.0, 0.0);
        }
        let pg = p.powf(gamma_neg);
        let q_s = if q == 0.0 { 0.0 } else { q * s_neg };
        return (pg * s_neg, pg * (gamma_neg * q_s + p));
    }

    // Precision-preserving shifted probability `p_m = p - clip` on both sides of `x = 0`.
    let pm = if x <= 0.0 { p - clip } else { (1.0 - clip) - q };
    if pm <= 0.0 {
        return (0.0, 0.0);
    }
    // `1 - p_m = (1 - p) + clip = q + clip >= clip > 0`, never cancels even at `x = +50`.
    let one_minus_pm = q + clip;
    let s_neg = if pm <= 0.5 { -(-pm).ln_1p() } else { -one_minus_pm.ln() };
    let s_over_pm = if pm < 1e-8 { 1.0 + pm * (0.5 + pm / 3.0) } else { s_neg / pm };
    let pm_g = if gamma_neg == 0.0 { 1.0 } else { pm.powf(gamma_neg) };
    let loss = pm_g * s_neg;
    let grad = pm_g * p * q * (gamma_neg * s_over_pm + 1.0 / one_minus_pm);
    (loss, grad)
}

/// Cancellation-free evaluation of `A(r, s, eps) = s + eps * r = (s - r) + (1 + eps) * r`,
/// where `r = 1 - p_t` (`q` for positives, `p` for negatives) and `s = -ln(1 - r) = softplus(∓x)`.
///
/// For small `r <= 1e-3`, evaluates `-ln(1 - r) - r = r^2 * (1/2 + r/3 + r^2/4 + ...)` directly so
/// `epsilon = -1.0` never cancels representable `O(r^2)` values at large logits (e.g. `x = 40`).
#[inline]
fn poly1_inner_f64(r: f64, s: f64, eps: f64) -> f64 {
    let rem = if r <= 1e-3 {
        r * r * (0.5 + r * (1.0 / 3.0 + r * (0.25 + r * (0.2 + r * (1.0 / 6.0 + r / 7.0)))))
    } else {
        s - r
    };
    rem + (1.0 + eps) * r
}

/// Exact single-element Poly-1 BCE `(loss, dL/dx)` in `f64`:
/// `t * (softplus(-x) + eps * q) + (1 - t) * (softplus(x) + eps * p)`.
#[inline]
pub(crate) fn poly1_bce_elem_f64(x: f64, t: f64, eps: f64) -> (f64, f64) {
    if !x.is_finite() || !t.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let p = sigmoid_f64(x);
    let q = sigmoid_f64(-x);
    let one_plus_eps = 1.0 + eps;
    let (l_pos, g_pos) = if t == 0.0 {
        (0.0, 0.0)
    } else {
        (poly1_inner_f64(q, softplus_f64(-x), eps), -q * (q + one_plus_eps * p))
    };
    let (l_neg, g_neg) = if t == 1.0 {
        (0.0, 0.0)
    } else {
        (poly1_inner_f64(p, softplus_f64(x), eps), p * (p + one_plus_eps * q))
    };
    let loss = (if t == 0.0 { 0.0 } else { t * l_pos }) + (if t == 1.0 { 0.0 } else { (1.0 - t) * l_neg });
    let grad = (if t == 0.0 { 0.0 } else { t * g_pos }) + (if t == 1.0 { 0.0 } else { (1.0 - t) * g_neg });
    (loss, grad)
}

/// Exact single-element Poly-1 Focal `(loss, dL/dx)` in `f64`:
/// `t * q^gamma * (softplus(-x) + eps * q) + (1 - t) * p^gamma * (softplus(x) + eps * p)`.
#[inline]
pub(crate) fn poly1_focal_elem_f64(x: f64, t: f64, gamma: f64, eps: f64) -> (f64, f64) {
    if !x.is_finite() || !t.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    if gamma == 0.0 {
        return poly1_bce_elem_f64(x, t, eps);
    }
    let p = sigmoid_f64(x);
    let q = sigmoid_f64(-x);
    let one_plus_eps = 1.0 + eps;
    let (l_pos, g_pos) = if t == 0.0 || q == 0.0 {
        (0.0, 0.0)
    } else {
        let a_pos = poly1_inner_f64(q, softplus_f64(-x), eps);
        let qg = q.powf(gamma);
        let p_a = if p == 0.0 { 0.0 } else { p * a_pos };
        (qg * a_pos, -qg * (gamma * p_a + q * (q + one_plus_eps * p)))
    };
    let (l_neg, g_neg) = if t == 1.0 || p == 0.0 {
        (0.0, 0.0)
    } else {
        let a_neg = poly1_inner_f64(p, softplus_f64(x), eps);
        let pg = p.powf(gamma);
        let q_a = if q == 0.0 { 0.0 } else { q * a_neg };
        (pg * a_neg, pg * (gamma * q_a + p * (p + one_plus_eps * q)))
    };
    let loss = (if t == 0.0 { 0.0 } else { t * l_pos }) + (if t == 1.0 { 0.0 } else { (1.0 - t) * l_neg });
    let grad = (if t == 0.0 { 0.0 } else { t * g_pos }) + (if t == 1.0 { 0.0 } else { (1.0 - t) * g_neg });
    (loss, grad)
}

/// Check matching `F32` dtype, device, and shape for an element-wise classification loss pair.
fn check_cls_pair(logits: &Tensor, targets: &Tensor, op: &'static str) -> Result<()> {
    for t in [logits, targets] {
        if t.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch { op, expected: DType::F32, got: t.dtype() });
        }
    }
    if logits.device() != targets.device() {
        return Err(Error::DeviceMismatch { op, lhs: logits.device(), rhs: targets.device() });
    }
    if logits.shape() != targets.shape() {
        return Err(Error::InvalidArgument {
            op,
            msg: format!("logits shape {} != targets shape {}", logits.shape(), targets.shape()),
        });
    }
    Ok(())
}

/// Shared host-evaluated element-wise custom op with exact `f64` analytical backward w.r.t. `logits`.
fn elemwise_cls_op<F>(logits: &Tensor, targets: &Tensor, op: &'static str, eval: F) -> Result<Tensor>
where
    F: Fn(f64, f64) -> (f64, f64) + Send + Sync + 'static,
{
    check_cls_pair(logits, targets, op)?;
    let dev = logits.device();
    let shape = logits.shape().clone();
    let n = shape.numel();
    if n == 0 {
        return Tensor::zeros_with(&shape, DType::F32, dev);
    }
    let xv = logits.to_vec::<f32>()?;
    let tv = targets.to_vec::<f32>()?;
    let mut out = vec![0.0f32; n];
    let mut dldx = vec![0.0f64; n];
    for i in 0..n {
        let (lv, gv) = eval(xv[i] as f64, tv[i] as f64);
        out[i] = lv as f32;
        dldx[i] = gv;
    }
    let out_t = Tensor::from_vec(out, &shape)?.to_device(dev)?;
    Tensor::custom_op(op, &[logits], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let gi = gv[i] as f64;
            if gi == 0.0 && dldx[i].is_finite() {
                continue;
            }
            grad[i] = (gi * dldx[i]) as f32;
        }
        Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
    })
}

/// Element-wise Asymmetric Loss (ASL, Ridnik et al. ICCV 2021) for multi-label classification logits and
/// targets in `[0, 1]`, with exact closed-form `f64` analytical backward gradients w.r.t. `logits`.
///
/// - `gamma_pos >= 0.0` (default `0.0`): positive focusing exponent.
/// - `gamma_neg >= 0.0` (default `4.0`): negative focusing exponent.
/// - `clip` in `[0.0, 1.0)` (default `0.05`): probability margin shift `p_m = max(p - clip, 0)` that
///   hard-thresholds easy negatives below `clip`.
pub fn asymmetric_loss(
    logits: &Tensor,
    targets: &Tensor,
    gamma_pos: f32,
    gamma_neg: f32,
    clip: f32,
) -> Result<Tensor> {
    if !(gamma_pos.is_finite() && gamma_pos >= 0.0) {
        return Err(invalid(format!("asymmetric_loss: gamma_pos must be finite and >= 0, got {gamma_pos}")));
    }
    if !(gamma_neg.is_finite() && gamma_neg >= 0.0) {
        return Err(invalid(format!("asymmetric_loss: gamma_neg must be finite and >= 0, got {gamma_neg}")));
    }
    if !(clip.is_finite() && (0.0..1.0).contains(&clip)) {
        return Err(invalid(format!("asymmetric_loss: clip must be finite and in [0, 1), got {clip}")));
    }
    let (gp, gn, m) = (gamma_pos as f64, gamma_neg as f64, clip as f64);
    elemwise_cls_op(logits, targets, "asymmetric_loss", move |x, t| asl_elem_f64(x, t, gp, gn, m))
}

/// Element-wise Poly-1 Binary Cross-Entropy loss (Leng et al. ICLR 2022) for logits and targets in `[0, 1]`,
/// with exact closed-form `f64` analytical backward gradients w.r.t. `logits`:
///
/// ```text
/// L_Poly1BCE(x, t) = BCE(x, t) + epsilon * (t * (1 - p) + (1 - t) * p)
/// ```
///
/// Requires finite `epsilon >= -1.0` (default `1.0`; `epsilon = 0.0` reduces identically to BCE).
pub fn poly1_bce_loss(logits: &Tensor, targets: &Tensor, epsilon: f32) -> Result<Tensor> {
    if !(epsilon.is_finite() && epsilon >= -1.0) {
        return Err(invalid(format!("poly1_bce_loss: epsilon must be finite and >= -1, got {epsilon}")));
    }
    let eps = epsilon as f64;
    elemwise_cls_op(logits, targets, "poly1_bce_loss", move |x, t| poly1_bce_elem_f64(x, t, eps))
}

/// Element-wise Poly-1 Focal Loss (Leng et al. ICLR 2022) for logits and targets in `[0, 1]`, with exact
/// closed-form `f64` analytical backward gradients w.r.t. `logits`:
///
/// ```text
/// L_Poly1Focal(x, t) = t * (1 - p)^gamma * (-ln(p) + epsilon * (1 - p))
///                    + (1 - t) * p^gamma * (-ln(1 - p) + epsilon * p)
/// ```
///
/// Requires finite `gamma >= 0.0` (default `2.0`) and finite `epsilon >= -1.0` (default `1.0`). Reduces
/// identically to [`poly1_bce_loss`] when `gamma = 0.0` and to standard Focal Loss when `epsilon = 0.0`.
pub fn poly1_focal_loss(logits: &Tensor, targets: &Tensor, gamma: f32, epsilon: f32) -> Result<Tensor> {
    if !(gamma.is_finite() && gamma >= 0.0) {
        return Err(invalid(format!("poly1_focal_loss: gamma must be finite and >= 0, got {gamma}")));
    }
    if !(epsilon.is_finite() && epsilon >= -1.0) {
        return Err(invalid(format!("poly1_focal_loss: epsilon must be finite and >= -1, got {epsilon}")));
    }
    let (g64, eps) = (gamma as f64, epsilon as f64);
    elemwise_cls_op(logits, targets, "poly1_focal_loss", move |x, t| poly1_focal_elem_f64(x, t, g64, eps))
}

/// Element-wise alpha-balanced Poly-1 Focal Loss (`alpha` in `(0, 1)` weights positive terms by `alpha` and
/// negative terms by `1 - alpha`).
pub fn poly1_focal_loss_with_alpha(
    logits: &Tensor,
    targets: &Tensor,
    gamma: f32,
    epsilon: f32,
    alpha: f32,
) -> Result<Tensor> {
    if !(gamma.is_finite() && gamma >= 0.0) {
        return Err(invalid(format!(
            "poly1_focal_loss_with_alpha: gamma must be finite and >= 0, got {gamma}"
        )));
    }
    if !(epsilon.is_finite() && epsilon >= -1.0) {
        return Err(invalid(format!(
            "poly1_focal_loss_with_alpha: epsilon must be finite and >= -1, got {epsilon}"
        )));
    }
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(invalid(format!(
            "poly1_focal_loss_with_alpha: alpha must be finite and in (0, 1), got {alpha}"
        )));
    }
    let (g64, eps, a64) = (gamma as f64, epsilon as f64, alpha as f64);
    elemwise_cls_op(logits, targets, "poly1_focal_loss_with_alpha", move |x, t| {
        if !x.is_finite() || !t.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let (lp, gp) = if t == 0.0 { (0.0, 0.0) } else { poly1_focal_elem_f64(x, 1.0, g64, eps) };
        let (ln, gn) = if t == 1.0 { (0.0, 0.0) } else { poly1_focal_elem_f64(x, 0.0, g64, eps) };
        let wp = if t == 0.0 { 0.0 } else { a64 * t };
        let wn = if t == 1.0 { 0.0 } else { (1.0 - a64) * (1.0 - t) };
        (wp * lp + wn * ln, wp * gp + wn * gn)
    })
}

/// Fused sparse-foreground / dense-background classification sum over `[B, nc, A]` logits (any device,
/// parallel deterministic row-order `f64` sum on host). Used by `detection_loss` for `Asl`, `Poly1Bce`,
/// and `Poly1Focal`.
pub(super) fn sparse_cls_sum<F>(
    x: &Tensor,
    idx: Vec<usize>,
    t: Vec<f32>,
    op_name: &'static str,
    eval: F,
) -> Result<Tensor>
where
    F: Fn(f64, f64) -> (f64, f64) + Send + Sync + 'static,
{
    let dev = x.device();
    let row = x.dims().last().copied().unwrap_or(1).max(1);
    // Preserve all non-zero and non-finite target scores so `NaN` / `-Inf` targets propagate through `eval`.
    let mut order: Vec<usize> = (0..idx.len()).filter(|&k| t[k] != 0.0).collect();
    order.sort_unstable_by_key(|&k| idx[k]);
    let pos: Vec<usize> = order.iter().map(|&k| idx[k]).collect();
    let pos_t: Vec<f32> = order.iter().map(|&k| t[k]).collect();

    let compute_val = |v: &[f32]| -> f32 {
        let rows: Vec<f64> = v
            .par_chunks(row * PAR_ROWS)
            .enumerate()
            .flat_map_iter(|(ci, c)| {
                let pos = &pos;
                let eval = &eval;
                c.chunks(row).enumerate().map(move |(ri, r)| {
                    let start = (ci * PAR_ROWS + ri) * row;
                    let lo = pos.partition_point(|&p| p < start);
                    let hi = pos.partition_point(|&p| p < start + r.len());
                    let mut s = 0.0f64;
                    let mut from = 0;
                    for &p in &pos[lo..hi] {
                        for &xv in &r[from..p - start] {
                            s += eval(xv as f64, 0.0).0;
                        }
                        from = p - start + 1;
                    }
                    for &xv in &r[from..] {
                        s += eval(xv as f64, 0.0).0;
                    }
                    s
                })
            })
            .collect();
        let dense: f64 = rows.iter().sum();
        let fg: f64 = pos.iter().zip(&pos_t).map(|(&i, &tk)| eval(v[i] as f64, tk as f64).0).sum();
        (dense + fg) as f32
    };

    let value = if dev.is_cpu() {
        x.with_f32_slice(compute_val)?
    } else {
        let host = x.to_vec::<f32>()?;
        compute_val(&host)
    };
    let xs = x.detach();
    let shape = xs.shape().clone();
    Tensor::custom_op(op_name, &[x], Tensor::scalar(value).to_device(dev)?, move |g| {
        let g = g.item::<f32>()? as f64;
        let compute_grad = |v: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; v.len()];
            out.par_chunks_mut(row * PAR_ROWS).zip(v.par_chunks(row * PAR_ROWS)).for_each(|(o, chunk)| {
                for (o, &xv) in o.iter_mut().zip(chunk) {
                    let dx = eval(xv as f64, 0.0).1;
                    *o = if g == 0.0 && dx.is_finite() { 0.0 } else { (g * dx) as f32 };
                }
            });
            for (&i, &tk) in pos.iter().zip(&pos_t) {
                let dx = eval(v[i] as f64, tk as f64).1;
                out[i] = if g == 0.0 && dx.is_finite() { 0.0 } else { (g * dx) as f32 };
            }
            out
        };
        let grad = if dev.is_cpu() {
            xs.with_f32_slice(compute_grad)?
        } else {
            let host = xs.to_vec::<f32>()?;
            compute_grad(&host)
        };
        Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
    })
}
