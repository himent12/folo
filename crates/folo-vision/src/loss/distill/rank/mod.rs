//! Class-wise rank distillation (docs/MODEL.md §8.9, `distill.rank_gain`): per class, the student's class logits
//! over the batch's candidates should be an increasing linear function of the teacher's. COCO AP reads each class's
//! scores mostly by their order (only the per-image detection caps compare classes), so the term asks for the
//! teacher's order and leaves each class's shift and scale to the supervised loss.
//!
//! With `x` the student's logits, `y` the teacher's, `p = sigmoid(x)` (detached), `p~ = sigmoid(y)`, the floor
//! `l = logit(tau)`, per element `i = (b, c, a)` the constants
//!
//! - `w_i = max(p_i, p~_i)^gamma` if `max(p_i, p~_i) > tau`, else 0 (the candidates of either model),
//! - `y'_i = max(y_i, l)`, and the student's `x'_i = max(x_i, l)` (its gradient passes where `x_i >= l`),
//!
//! and per class `c`, over every `(b, a)` of the batch: `W = Σ w`, `n = #(w > 0)`, the weighted means `mx`, `my`
//! (constants: detaching them leaves the gradients exact, since `Σ w (x' − mx) = 0`), the variances `vx`, `vy` and
//! covariance `cxy` about them, and `rho = cxy / (sqrt(vx + eps) · sqrt(vy + eps))`, `eps = 1e-4`. A class is used
//! iff `n >= rank_min` and `vy > eps`, and
//!
//! `L_rank = Σ_used W_c · (1 − rho_c) / Σ_used W_c` (0 when no class is used),
//!
//! with the gradient `dL/dx_i = −(w_i / Σ_used W) · [(y'_i − my) / (sx · sy) − rho_c · (x'_i − mx) / sx²]` where
//! `x_i >= l`, else 0. For a class with no student logit under the floor, the gradient sums to 0 (the term does not
//! move the class's mean logit) and its product with `x' − mx` is `−(W_c / Σ_used W) · rho_c · eps / (vx + eps)`:
//! the spread is left alone up to the share `eps / (vx + eps)`, which pushes it up for `rho_c > 0` (down for
//! `rho_c < 0`) and matters only for a class whose logits sit within about `sqrt(eps) = 0.01` of their mean. A floored logit gets no gradient, so a
//! class with some of them can see its other logits' gradient sum move off 0.
//!
//! Every sum is a row sum over `a` per `(b, c)` and then a sum over `b` per class, both in a fixed order, so the
//! term is bit-reproducible on the CPU and made only of tensor ops every backend has.

#[cfg(test)]
mod tests;

use folo_core::{Result, Tensor, no_grad};

use super::DistillConfig;

/// `eps` of the rank term's standard deviations and of its "the teacher varies" test.
pub const RANK_EPS: f32 = 1e-4;

/// `logit(tau) = ln(tau / (1 − tau))`, the rank term's logit floor.
pub fn floor_logit(tau: f32) -> f32 {
    (tau / (1.0 - tau)).ln()
}

/// The result of [`rank_loss`].
#[derive(Clone, Debug)]
pub struct RankLoss {
    /// `L_rank`, rank 0, differentiable with respect to the student's logits (without `rank_gain` and `B`).
    pub loss: Tensor,
    /// The number of classes used, rank 0, a constant.
    pub classes: Tensor,
}

/// `[B, nc, A]` to its per-class sums `[nc]`: row sums over `A`, then the sum over `B`.
fn per_class(t: &Tensor) -> Result<Tensor> {
    t.sum(2, false)?.sum(0, false)
}

/// The rank term for the student's class logits `x: [B, nc, A]` (`F32`, with gradient), the teacher's floored
/// logits `y = max(cls_raw_T, logit(rank_floor))` and probabilities `teacher_prob = sigmoid(cls_raw_T)` (constants
/// of the same shape), with `cfg`'s `rank_floor`, `rank_power` and `rank_min` (see the module docs).
pub fn rank_loss(x: &Tensor, y: &Tensor, teacher_prob: &Tensor, cfg: &DistillConfig) -> Result<RankLoss> {
    let (_, nc, _) = x.dims3()?;
    let l = floor_logit(cfg.rank_floor);
    let col = |t: Tensor| t.reshape([1, nc, 1]);
    // The constants: weights, the classes used and their share of the weight, the means, the teacher's deviations.
    let (w, mx, dy, sy, w_sum, coef, classes) = {
        let _g = no_grad();
        let m = x.detach().sigmoid()?.maximum(teacher_prob)?;
        let cand = m.gt_scalar(cfg.rank_floor)?;
        let w = if cfg.rank_power == 0.0 { cand.clone() } else { m.powf(cfg.rank_power)?.mul(&cand)? };
        // A class without candidates has W = 0 and every weighted sum 0: dividing by `tiny` keeps it at 0.
        let w_sum = per_class(&w)?.clamp(f32::MIN_POSITIVE, f32::MAX)?;
        let n = per_class(&cand)?;
        let mx = per_class(&w.mul(&x.detach().maximum_scalar(l)?)?)?.div(&w_sum)?;
        let my = per_class(&w.mul(y)?)?.div(&w_sum)?;
        let dy = y.sub(&col(my)?)?;
        let vy = per_class(&w.mul(&dy)?.mul(&dy)?)?.div(&w_sum)?;
        let used = n.ge_scalar(cfg.rank_min as f32)?.mul(&vy.gt_scalar(RANK_EPS)?)?;
        let used_w = w_sum.mul(&used)?;
        let coef = used_w.div(&used_w.sum_all()?.clamp(f32::MIN_POSITIVE, f32::MAX)?)?;
        (w, mx, dy, vy.add_scalar(RANK_EPS)?.sqrt()?, w_sum, coef, used.sum_all()?)
    };
    let dx = x.maximum_scalar(l)?.sub(&col(mx)?)?;
    let wdx = dx.mul(&w)?;
    let vx = per_class(&wdx.mul(&dx)?)?.div(&w_sum)?;
    let cxy = per_class(&wdx.mul(&dy)?)?.div(&w_sum)?;
    let rho = cxy.div(&vx.add_scalar(RANK_EPS)?.sqrt()?.mul(&sy)?)?;
    let loss = rho.rsub_scalar(1.0)?.mul(&coef)?.sum_all()?;
    Ok(RankLoss { loss, classes })
}
