//! Exact function-preserving Floor Conv+BN Gauge Normalization for transfer learning and continuations.
//!
//! Long coupled-WD continuations (such as `folo-n-coco-v5`) can shrink scale-invariant `Conv + BatchNorm` filter
//! norms `‖W_c‖₂` well below the canonical Kaiming scale `√2` while collapsing `running_var`, inflating effective
//! directional LR by `1 / ‖W_c‖₂²` during downstream fine-tuning.
//!
//! For every unfused, bias-free `ConvBnAct` output channel `c` with `‖W_c‖₂ > 1e-4`, floor gauge normalization
//! computes
//! ```text
//! α_c = clamp(√2 / ‖W_c‖₂, 1.0, 4.0)
//! W_c ← α_c · W_c
//! μ_c ← α_c · μ_c
//! v_c ← α_c² · (v_c + ε) − ε
//! ```
//! Because `α_c ≥ 1.0`, high-norm layers (such as the input stem) are left untouched, `v_c` stays strictly
//! positive without altering `γ` or `β`, and the forward output of every layer is preserved to float rounding
//! (`~1e-6` relative error).

use folo_core::{DType, Tensor, no_grad};
use folo_nn::{ConvBnAct, Module};

use crate::error::Result;

/// Canonical per-output-channel Kaiming L2 floor (`√2`).
pub const GAUGE_TARGET_NORM: f32 = std::f32::consts::SQRT_2;

/// Minimum channel L2 norm eligible for rescaling (dead channels stay untouched).
const MIN_LIVE_NORM: f32 = 1e-4;

/// Maximum per-channel upward scale factor.
const MAX_ALPHA: f32 = 4.0;

/// Summary of [`apply_floor_gauge_norm`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GaugeNormReport {
    /// Unfused bias-free `ConvBnAct` layers visited.
    pub layers_visited: usize,
    /// Layers that had at least one channel rescaled upward.
    pub layers_rescaled: usize,
    /// Output channels rescaled upward (`α_c > 1.0`).
    pub channels_rescaled: usize,
    /// Total output channels inspected.
    pub total_channels: usize,
}

/// Apply exact function-preserving Floor Conv+BN Gauge Normalization in place to every unfused, bias-free
/// [`ConvBnAct`] inside `model`.
///
/// Works on both CPU and GPU tensors (`F32`); `F16` weights are skipped cleanly.
pub fn apply_floor_gauge_norm(model: &dyn Module) -> Result<GaugeNormReport> {
    let _g = no_grad();
    let mut report = GaugeNormReport::default();
    let mut err: Option<crate::error::Error> = None;

    let mut visit = |m: &dyn Module| {
        if err.is_some() {
            return;
        }
        let Some(block) = (m as &dyn std::any::Any).downcast_ref::<ConvBnAct>() else {
            return;
        };
        if block.conv.bias.is_some() {
            return;
        }
        let Some(bn) = &block.bn else {
            return;
        };
        if block.conv.weight.dtype() != DType::F32
            || bn.running_mean.dtype() != DType::F32
            || bn.running_var.dtype() != DType::F32
        {
            return;
        }
        match rescale_conv_bn(&block.conv.weight, &bn.running_mean, &bn.running_var, bn.eps as f32) {
            Ok((rescaled, total)) => {
                report.layers_visited += 1;
                if rescaled > 0 {
                    report.layers_rescaled += 1;
                }
                report.channels_rescaled += rescaled;
                report.total_channels += total;
            }
            Err(e) => err = Some(e),
        }
    };

    visit(model);
    model.for_each_module(&mut |_, m| visit(m));
    if let Some(e) = err {
        return Err(e);
    }
    Ok(report)
}

fn rescale_conv_bn(
    weight: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f32,
) -> Result<(usize, usize)> {
    let dims = weight.dims();
    if dims.len() != 4 || dims[0] == 0 {
        return Ok((0, 0));
    }
    let c_out = dims[0];
    let fan = weight.numel() / c_out;
    let dev = weight.device();

    let mut w = weight.to_vec::<f32>()?;
    let mut mu = running_mean.to_vec::<f32>()?;
    let mut var = running_var.to_vec::<f32>()?;
    if mu.len() != c_out || var.len() != c_out {
        return Ok((0, c_out));
    }

    let mut rescaled = 0usize;
    for c in 0..c_out {
        let row = &mut w[c * fan..(c + 1) * fan];
        let sq: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let norm = sq.sqrt() as f32;
        if norm <= MIN_LIVE_NORM || norm >= GAUGE_TARGET_NORM {
            continue;
        }
        let alpha = (GAUGE_TARGET_NORM / norm).clamp(1.0, MAX_ALPHA);
        if alpha <= 1.0 {
            continue;
        }
        for v in row.iter_mut() {
            *v *= alpha;
        }
        mu[c] *= alpha;
        // Exact variance gauge transform: (v' + eps) = alpha^2 * (v + eps).
        // Because alpha >= 1.0 and v >= 0, v' >= v >= 0 without clamping.
        let a2 = f64::from(alpha) * f64::from(alpha);
        let v_new = a2 * (f64::from(var[c]) + f64::from(eps)) - f64::from(eps);
        var[c] = (v_new as f32).max(0.0);
        rescaled += 1;
    }

    if rescaled > 0 {
        let w_t = Tensor::from_vec(w, weight.shape())?.to_device(dev)?;
        let mu_t = Tensor::from_vec(mu, running_mean.shape())?.to_device(dev)?;
        let var_t = Tensor::from_vec(var, running_var.shape())?.to_device(dev)?;
        weight.copy_from(&w_t)?;
        running_mean.copy_from(&mu_t)?;
        running_var.copy_from(&var_t)?;
    }
    Ok((rescaled, c_out))
}
