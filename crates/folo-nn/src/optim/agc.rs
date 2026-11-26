//! Adaptive Gradient Clipping (AGC; Brock et al. 2021, NFNet).
//!
//! Clips each output unit's gradient L2 norm proportionally to the corresponding parameter unit's L2 norm:
//!
//! ```text
//! ||W_i||_F^* = max(||W_i||_F, eps)
//! G_i <- (clip_factor * ||W_i||_F^* / ||G_i||_F) * G_i    if ||G_i||_F > clip_factor * ||W_i||_F^*
//! ```
//!
//! For tensors of rank >= 2 (`[out_c, in_c, kH, kW]` convolutions or `[out_f, in_f]` linear weights), axis `0`
//! indexes the output units; for 1-D or 0-D tensors, the whole tensor is treated as one unit (or skipped when
//! `skip_1d` is set in [`super::GradAccumulator::clip_grad_agc`]).
//!
//! Unit norms are accumulated in `f64` (exact, overflow- and underflow-free across the entire finite `f32`
//! range). Units with non-finite weights or gradients are left untouched so upstream NaN/Inf detection is
//! preserved.

use folo_core::{DType, Tensor, no_grad};

use super::cpu_fast;
use super::simd::{scale_slice, scale2_slice, sum_sq_f64_slice};
use crate::{Error, Result};

/// Default AGC clipping factor `lambda` from Brock et al. 2021 (`0.01`).
pub const AGC_DEFAULT_CLIP_FACTOR: f64 = 0.01;
/// Default AGC weight-norm floor `eps` from Brock et al. 2021 (`1e-3`).
pub const AGC_DEFAULT_EPS: f64 = 1e-3;

/// Summary statistics returned by [`clip_grad_agc_tensor`] and [`super::GradAccumulator::clip_grad_agc`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AgcStats {
    /// Number of units whose gradient was scaled down.
    pub clipped_units: usize,
    /// Total number of units inspected (excluding skipped 1-D/0-D parameters or zero-element tensors).
    pub total_units: usize,
    /// Maximum pre-clipping norm ratio `||G_i||_F / max(||W_i||_F, eps)` across inspected units
    /// (`0.0` if no units were inspected; `NaN` if any inspected unit had non-finite `W` or `G`;
    /// `+Inf` if any finite unit had `max(||W_i||_F, eps) == 0` with `||G_i||_F > 0`).
    pub max_ratio: f64,
}

impl AgcStats {
    #[inline]
    pub(crate) fn merge(self, other: Self) -> Self {
        let max_ratio = if self.max_ratio.is_nan() || other.max_ratio.is_nan() {
            f64::NAN
        } else {
            self.max_ratio.max(other.max_ratio)
        };
        Self {
            clipped_units: self.clipped_units + other.clipped_units,
            total_units: self.total_units + other.total_units,
            max_ratio,
        }
    }
}

/// Validate `clip_factor` (`finite && > 0`) and `eps` (`finite && >= 0`).
pub(crate) fn validate_agc_params(clip_factor: f64, eps: f64) -> Result<()> {
    if !(clip_factor.is_finite() && clip_factor > 0.0) {
        return Err(Error::Invalid(format!(
            "clip_grad_agc: clip_factor must be finite and > 0, got {clip_factor}"
        )));
    }
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(Error::Invalid(format!("clip_grad_agc: eps must be finite and >= 0, got {eps}")));
    }
    Ok(())
}

/// Scale one contiguous unit slice by `scale` in `[0, 1)` without subnormal `f32` underflow.
#[inline]
fn scale_unit_f64(unit: &mut [f32], scale: f64) {
    if scale == 0.0 {
        unit.fill(0.0);
    } else if scale < f32::MIN_POSITIVE as f64 {
        let f1 = scale.sqrt() as f32;
        let f2 = if f1 >= f32::MIN_POSITIVE { (scale / f1 as f64) as f32 } else { 0.0 };
        if f1 >= f32::MIN_POSITIVE && f2 >= f32::MIN_POSITIVE && f2.is_finite() {
            scale2_slice(unit, f1, f2);
        } else {
            for v in unit {
                *v = (*v as f64 * scale) as f32;
            }
        }
    } else {
        scale_slice(unit, scale as f32);
    }
}

/// Core per-unit AGC pass over contiguous `F32` slices `w` and `g` partitioned into `num_units` equal chunks.
fn agc_slices(w: &[f32], g: &mut [f32], num_units: usize, clip_factor: f64, eps: f64) -> AgcStats {
    debug_assert_eq!(w.len(), g.len());
    if num_units == 0 || w.is_empty() {
        return AgcStats::default();
    }
    let unit_len = w.len() / num_units;
    debug_assert_eq!(unit_len * num_units, w.len());

    let mut clipped = 0usize;
    let mut max_ratio = 0.0f64;

    for (wu, gu) in w.chunks_exact(unit_len).zip(g.chunks_exact_mut(unit_len)) {
        let w_norm = sum_sq_f64_slice(wu).sqrt();
        let g_norm = sum_sq_f64_slice(gu).sqrt();
        if !w_norm.is_finite() || !g_norm.is_finite() {
            max_ratio = f64::NAN;
            continue;
        }
        let w_star = w_norm.max(eps);
        let ratio = if w_star > 0.0 {
            g_norm / w_star
        } else if g_norm > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        if !max_ratio.is_nan() && ratio > max_ratio {
            max_ratio = ratio;
        }
        let max_norm = clip_factor * w_star;
        if g_norm > max_norm {
            let scale = max_norm / g_norm;
            scale_unit_f64(gu, scale);
            clipped += 1;
        }
    }

    AgcStats { clipped_units: clipped, total_units: num_units, max_ratio }
}

/// Apply Adaptive Gradient Clipping (Brock et al. 2021) in place to `grad` given its parameter `param`.
///
/// - Rank >= 2 tensors are partitioned along axis `0` (`dims[0]` units).
/// - Rank <= 1 tensors are treated as a single unit.
/// - Works on CPU (zero-copy fast path for contiguous `F32`) and on any other device (bit-exact `f64`
///   unit norm evaluation).
/// - Leaves units with non-finite weights or gradients untouched.
///
/// # Errors
/// Returns [`Error::Invalid`] if `clip_factor` is not finite/`> 0`, `eps` is not finite/`>= 0`, `param` and
/// `grad` differ in shape/device, or either tensor is not `F32`.
pub fn clip_grad_agc_tensor(param: &Tensor, grad: &Tensor, clip_factor: f64, eps: f64) -> Result<AgcStats> {
    validate_agc_params(clip_factor, eps)?;
    if param.dtype() != DType::F32 {
        return Err(Error::Invalid(format!("clip_grad_agc: param dtype is {}, expected f32", param.dtype())));
    }
    if grad.dtype() != DType::F32 {
        return Err(Error::Invalid(format!("clip_grad_agc: grad dtype is {}, expected f32", grad.dtype())));
    }
    if param.shape() != grad.shape() {
        return Err(Error::Invalid(format!(
            "clip_grad_agc: shape mismatch param {:?} vs grad {:?}",
            param.dims(),
            grad.dims()
        )));
    }
    if param.device() != grad.device() {
        return Err(Error::Invalid(format!(
            "clip_grad_agc: device mismatch param {:?} vs grad {:?}",
            param.device(),
            grad.device()
        )));
    }
    if param.numel() == 0 {
        return Ok(AgcStats::default());
    }
    let dims = param.dims();
    let num_units = if dims.len() >= 2 { dims[0] } else { 1 };
    let _g = no_grad();

    if cpu_fast(&[grad]) && param.device().is_cpu() && !param.same_storage(grad) {
        let stats = param
            .with_f32_slice(|ws| grad.update_f32(|gs| agc_slices(ws, gs, num_units, clip_factor, eps)))??;
        return Ok(stats);
    }

    // Aliased CPU storage, non-contiguous tensor, or device tensor: evaluate exact f64 per-unit norms on host
    // and write back only when at least one unit was clipped.
    let w_vec = param.to_vec::<f32>()?;
    let mut g_vec = grad.to_vec::<f32>()?;
    let stats = agc_slices(&w_vec, &mut g_vec, num_units, clip_factor, eps);
    if stats.clipped_units > 0 {
        let updated = Tensor::from_vec(g_vec, grad.shape().clone())?.to_device(grad.device())?;
        grad.copy_from(&updated)?;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::{GradAccumulator, GradSource};
    use folo_core::{Grads, rng::Rng};

    #[test]
    fn agc_exact_per_unit_math_and_floor() {
        // 2 output units of length 3:
        // Unit 0: w = [3, 4, 0] (||w|| = 5), g = [6, 8, 0] (||g|| = 10), ratio = 2.0 > clip = 0.1
        //   -> target ||g|| = 0.1 * 5 = 0.5 -> scale = 0.05 -> g' = [0.3, 0.4, 0.0].
        // Unit 1: w = [0, 0, 0] (||w|| = 0 -> w* = eps = 1e-2), g = [0, 3e-4, 4e-4] (||g|| = 5e-4)
        //   -> max_norm = 0.1 * 1e-2 = 1e-3 >= 5e-4 -> UNCHANGED!
        let w = Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0], [2, 3]).unwrap();
        let g = Tensor::from_slice(&[6.0f32, 8.0, 0.0, 0.0, 3e-4, 4e-4], [2, 3]).unwrap();
        let stats = clip_grad_agc_tensor(&w, &g, 0.1, 1e-2).unwrap();
        assert_eq!(stats.clipped_units, 1);
        assert_eq!(stats.total_units, 2);
        assert!((stats.max_ratio - 2.0).abs() < 1e-12);
        let gv = g.to_vec::<f32>().unwrap();
        assert!((gv[0] - 0.3).abs() < 1e-6);
        assert!((gv[1] - 0.4).abs() < 1e-6);
        assert_eq!(gv[2], 0.0);
        assert_eq!(gv[4], 3e-4);
        assert_eq!(gv[5], 4e-4);
    }

    #[test]
    fn agc_conv4d_unit_partition_and_skip_1d() {
        let mut rng = Rng::seed(42);
        let w4 = Tensor::rand_uniform([4, 2, 3, 3], -1.0, 1.0, &mut rng).requires_grad_();
        let b1 = Tensor::zeros([4]).requires_grad_();
        let g_w4 = Tensor::from_vec(vec![10.0f32; 4 * 2 * 3 * 3], [4, 2, 3, 3]).unwrap();
        let g_b1 = Tensor::from_vec(vec![10.0f32; 4], [4]).unwrap();

        let mut grads = Grads::default();
        grads.insert(&w4, g_w4);
        grads.insert(&b1, g_b1);

        let mut acc = GradAccumulator::new(&[w4.clone(), b1.clone()]);
        acc.add(&grads).unwrap();
        let stats = acc.clip_grad_agc(0.01, 1e-3, true).unwrap();
        assert_eq!(stats.total_units, 4, "1D bias skipped when skip_1d=true");
        assert_eq!(stats.clipped_units, 4);
        // Bias gradient untouched:
        assert_eq!(acc.grad(&b1).unwrap().to_vec::<f32>().unwrap(), vec![10.0; 4]);
        // Each conv output channel unit has ||g_i|| <= 0.01 * max(||w_i||, 1e-3):
        let wv = w4.to_vec::<f32>().unwrap();
        let gv = acc.grad(&w4).unwrap().to_vec::<f32>().unwrap();
        for c in 0..4 {
            let wn = sum_sq_f64_slice(&wv[c * 18..(c + 1) * 18]).sqrt().max(1e-3);
            let gn = sum_sq_f64_slice(&gv[c * 18..(c + 1) * 18]).sqrt();
            assert!((gn - 0.01 * wn).abs() < 1e-5 * wn, "unit {c}: gn={gn} vs {}", 0.01 * wn);
        }

        // With skip_1d=false on a fresh copy, all 5 units (4 conv channels + 1 bias unit) are clipped.
        let mut acc2 = GradAccumulator::new(&[w4.clone(), b1.clone()]);
        acc2.add(&grads).unwrap();
        let stats2 = acc2.clip_grad_agc(0.01, 1e-3, false).unwrap();
        assert_eq!(stats2.total_units, 5);
        assert_eq!(stats2.clipped_units, 5);
        let bn = sum_sq_f64_slice(&acc2.grad(&b1).unwrap().to_vec::<f32>().unwrap()).sqrt();
        assert!((bn - 1e-5).abs() < 1e-8);
    }

    #[test]
    fn agc_extreme_scales_nonfinite_aliasing_and_validation() {
        // Extreme f32::MAX/4 gradient with tiny weight -> subnormal scale handled cleanly without NaN.
        let w_tiny = Tensor::from_slice(&[1e-20f32, 0.0], [1, 2]).unwrap();
        let g_huge = Tensor::from_slice(&[1e35f32, 0.0], [1, 2]).unwrap();
        let st = clip_grad_agc_tensor(&w_tiny, &g_huge, 0.01, 0.0).unwrap();
        assert_eq!(st.clipped_units, 1);
        let gv = g_huge.to_vec::<f32>().unwrap();
        assert!(gv[0].is_finite() && (gv[0] - 1e-22).abs() < 1e-25, "got {gv:?}");

        // Non-finite gradient leaves unit untouched and reports NaN max_ratio.
        let w = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], [2, 2]).unwrap();
        let g = Tensor::from_slice(&[f32::INFINITY, 1.0, 100.0, 0.0], [2, 2]).unwrap();
        let st_nf = clip_grad_agc_tensor(&w, &g, 0.1, 1e-3).unwrap();
        assert_eq!(st_nf.clipped_units, 1, "finite unit 1 still clipped");
        assert!(st_nf.max_ratio.is_nan());
        let gv_nf = g.to_vec::<f32>().unwrap();
        assert!(gv_nf[0].is_infinite(), "non-finite unit untouched");
        assert!((gv_nf[2] - 0.1).abs() < 1e-6);

        // Aliased storage (param == grad) does not deadlock and clips cleanly.
        let aliased = Tensor::from_slice(&[3.0f32, 4.0], [1, 2]).unwrap();
        let st_al = clip_grad_agc_tensor(&aliased, &aliased, 0.5, 1e-3).unwrap();
        assert_eq!(st_al.clipped_units, 1);
        let av = aliased.to_vec::<f32>().unwrap();
        assert!((av[0] - 1.5).abs() < 1e-6 && (av[1] - 2.0).abs() < 1e-6);

        // Zero weight with eps == 0: nonzero gradient clips to 0 with max_ratio == +Inf; zero gradient gives 0.0.
        let wz = Tensor::zeros([2, 2]);
        let gz = Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0], [2, 2]).unwrap();
        let st_z = clip_grad_agc_tensor(&wz, &gz, 0.01, 0.0).unwrap();
        assert_eq!(st_z.clipped_units, 1);
        assert!(st_z.max_ratio.is_infinite() && st_z.max_ratio.is_sign_positive());
        assert_eq!(gz.to_vec::<f32>().unwrap(), vec![0.0; 4]);
        let st_zz = clip_grad_agc_tensor(&wz, &gz, 0.01, 0.0).unwrap();
        assert_eq!(st_zz.clipped_units, 0);
        assert_eq!(st_zz.max_ratio, 0.0);

        // NaN weight leaves unit untouched and propagates NaN to max_ratio.
        let w_nan = Tensor::from_slice(&[f32::NAN, 1.0], [1, 2]).unwrap();
        let g_ok = Tensor::from_slice(&[10.0f32, 0.0], [1, 2]).unwrap();
        let st_wn = clip_grad_agc_tensor(&w_nan, &g_ok, 0.1, 1e-3).unwrap();
        assert_eq!(st_wn.clipped_units, 0);
        assert!(st_wn.max_ratio.is_nan());
        assert_eq!(g_ok.to_vec::<f32>().unwrap(), vec![10.0, 0.0]);

        // Non-contiguous (transposed) tensors match contiguous per-unit AGC.
        let w_nc = Tensor::from_slice(&[3.0f32, 0.0, 4.0, 0.0], [2, 2]).unwrap().t().unwrap();
        let g_nc = Tensor::from_slice(&[6.0f32, 0.0, 8.0, 0.0], [2, 2]).unwrap().t().unwrap();
        let st_nc = clip_grad_agc_tensor(&w_nc, &g_nc, 0.1, 1e-3).unwrap();
        assert_eq!(st_nc.clipped_units, 1);
        let gv_nc = g_nc.to_vec::<f32>().unwrap();
        assert!((gv_nc[0] - 0.3).abs() < 1e-6 && (gv_nc[1] - 0.4).abs() < 1e-6);

        // Zero-length tensor and invalid parameters.
        let empty = Tensor::zeros([0, 4]);
        assert_eq!(clip_grad_agc_tensor(&empty, &empty, 0.01, 1e-3).unwrap(), AgcStats::default());
        assert!(clip_grad_agc_tensor(&w, &w, 0.0, 1e-3).is_err());
        assert!(clip_grad_agc_tensor(&w, &w, -0.1, 1e-3).is_err());
        assert!(clip_grad_agc_tensor(&w, &w, f64::NAN, 1e-3).is_err());
        assert!(clip_grad_agc_tensor(&w, &w, 0.01, -1e-3).is_err());
        assert!(clip_grad_agc_tensor(&w, &w, 0.01, f64::INFINITY).is_err());
        assert!(clip_grad_agc_tensor(&w, &Tensor::zeros([2, 3]), 0.01, 1e-3).is_err());
    }
}
