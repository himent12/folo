//! Root Mean Square Layer Normalization ([`RmsNorm`]; Zhang & Sennrich, NeurIPS 2019).
//!
//! Normalizes the last dimension `d` of an input tensor `x: [..., d]` by its root-mean-square:
//!
//! ```text
//! RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
//! ```
//!
//! Delegates directly to [`folo_core::Tensor::rms_norm`] (fused CPU & WGSL GPU kernels with overflow-safe
//! composite fallback), with full autograd support for both the input and optional learnable scale `weight`.
//! [`crate::optim::StandardGroups::from_module`] automatically places [`RmsNorm::weight`] in the no-decay
//! `bn` group.

use folo_core::{DType, Tensor};

use crate::module::{Forward, Module, TensorKind};
use crate::{Error, Result};

/// Default numerical stability epsilon for [`RmsNorm`] (`1e-6`, LLaMA / Gemma / Qwen default).
pub const RMSNORM_DEFAULT_EPS: f32 = 1e-6;

/// Root Mean Square Layer Normalization over the last dimension (`dim >= 1`).
///
/// Parameter name: `weight` (`[dim]`, initialized to ones when `elementwise_affine` is enabled).
pub struct RmsNorm {
    /// Optional learnable per-feature gain `[dim]`.
    pub weight: Option<Tensor>,
    dim: usize,
    eps: f32,
}

impl RmsNorm {
    /// Create an [`RmsNorm`] layer with learnable unit-initialized `weight: [dim]` and stability floor `eps >= 0`.
    pub fn new(dim: usize, eps: f32) -> Result<Self> {
        validate_rmsnorm(dim, eps)?;
        let weight = Tensor::ones([dim]).requires_grad_();
        Ok(Self { weight: Some(weight), dim, eps })
    }

    /// Create an [`RmsNorm`] layer without a learnable affine `weight`.
    pub fn without_affine(dim: usize, eps: f32) -> Result<Self> {
        validate_rmsnorm(dim, eps)?;
        Ok(Self { weight: None, dim, eps })
    }

    /// Wrap an existing 1-D `weight: [dim]` tensor (`F32` or `F16`) in an [`RmsNorm`] layer.
    pub fn from_weight(weight: Tensor, eps: f32) -> Result<Self> {
        if weight.dims().len() != 1 {
            return Err(Error::Invalid(format!(
                "RmsNorm: weight must be 1-D [dim], got {:?}",
                weight.dims()
            )));
        }
        let dim = weight.dims()[0];
        validate_rmsnorm(dim, eps)?;
        if !matches!(weight.dtype(), DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!(
                "RmsNorm: weight dtype must be f32 or f16, got {}",
                weight.dtype()
            )));
        }
        Ok(Self { weight: Some(weight), dim, eps })
    }

    /// Normalized feature dimension (last axis size).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Configured epsilon floor.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

fn validate_rmsnorm(dim: usize, eps: f32) -> Result<()> {
    if dim == 0 {
        return Err(Error::Invalid("RmsNorm: dim must be >= 1, got 0".into()));
    }
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(Error::Invalid(format!("RmsNorm: eps must be finite and >= 0, got {eps}")));
    }
    Ok(())
}

impl Module for RmsNorm {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        if let Some(w) = &self.weight {
            f("weight", w, TensorKind::Parameter);
        }
    }
}

impl Forward for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last = *x.dims().last().ok_or_else(|| Error::Invalid("RmsNorm: scalar input".into()))?;
        if last != self.dim {
            return Err(Error::Invalid(format!(
                "RmsNorm: input {:?} has last dim {last}, expected {}",
                x.dims(),
                self.dim
            )));
        }
        if let Some(w) = &self.weight {
            if w.dims() != [self.dim] {
                return Err(Error::Invalid(format!(
                    "RmsNorm: weight shape {:?} does not match [{}]",
                    w.dims(),
                    self.dim
                )));
            }
            if w.dtype() == DType::F32
                && x.dtype() == DType::F16
                && folo_core::autocast_dtype() == Some(DType::F16)
            {
                let w16 = w.to_dtype(DType::F16)?;
                return Ok(x.rms_norm(Some(&w16), self.eps)?);
            }
        }
        Ok(x.rms_norm(self.weight.as_ref(), self.eps)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::StandardGroups;
    use folo_core::rng::Rng;
    use folo_core::testing::{GradCheck, grad_check_with};

    #[test]
    fn rmsnorm_values_grad_check_standard_groups_and_validation() {
        let mut rng = Rng::seed(314);
        let norm = RmsNorm::new(6, RMSNORM_DEFAULT_EPS).unwrap();
        let x = Tensor::randn([2, 3, 6], 0.0, 1.5, &mut rng);

        // Exact reference check in f64.
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 3, 6]);
        let xv = x.to_vec::<f32>().unwrap();
        let yv = y.to_vec::<f32>().unwrap();
        for row in 0..6 {
            let xr = &xv[row * 6..(row + 1) * 6];
            let yr = &yv[row * 6..(row + 1) * 6];
            let ms: f64 = xr.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 6.0;
            let inv_rms = 1.0 / (ms + RMSNORM_DEFAULT_EPS as f64).sqrt();
            for j in 0..6 {
                let want = (xr[j] as f64) * inv_rms;
                assert!((yr[j] as f64 - want).abs() < 1e-5, "row {row} j {j}: {} vs {want}", yr[j]);
            }
        }

        // Finite-difference gradient check w.r.t. both input x and learnable weight.
        let w = Tensor::rand_uniform([6], 0.5, 1.8, &mut rng);
        let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 7 };
        grad_check_with(
            |inps| {
                let layer = RmsNorm::from_weight(inps[1].clone(), RMSNORM_DEFAULT_EPS).unwrap();
                let out = layer.forward(&inps[0]).unwrap();
                out.square()?.sum_all()
            },
            &[x.clone(), w],
            &gc,
        )
        .unwrap();

        // Without affine weight.
        let no_aff = RmsNorm::without_affine(6, RMSNORM_DEFAULT_EPS).unwrap();
        assert!(no_aff.named_parameters().is_empty());
        grad_check_with(
            |inps| no_aff.forward(&inps[0]).unwrap().square()?.sum_all(),
            std::slice::from_ref(&x),
            &gc,
        )
        .unwrap();

        // StandardGroups automatically routes RmsNorm::weight into the no-decay `bn` group.
        let sg = StandardGroups::from_module(&norm);
        assert!(sg.decay.is_empty() && sg.bias.is_empty() && sg.bn.len() == 1);
        assert_eq!(sg.bn[0].0, "weight");

        // Validation rejections.
        assert!(RmsNorm::new(0, RMSNORM_DEFAULT_EPS).is_err());
        assert!(RmsNorm::new(4, -1e-6).is_err());
        assert!(RmsNorm::new(4, f32::NAN).is_err());
        assert!(RmsNorm::from_weight(Tensor::ones([2, 3]), RMSNORM_DEFAULT_EPS).is_err());
        assert!(norm.forward(&Tensor::zeros([2, 5])).is_err());
        assert!(norm.forward(&Tensor::scalar(1.0)).is_err());
    }
}
