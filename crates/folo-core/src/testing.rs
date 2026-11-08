//! Test helpers: finite-difference gradient checking.

use crate::rng::Rng;
use crate::{DType, Error, Result, Tensor, no_grad};

/// Tolerances for [`grad_check_with`].
#[derive(Debug, Clone)]
pub struct GradCheck {
    /// Central-difference step.
    pub eps: f64,
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance (against the larger of the two magnitudes).
    pub rtol: f64,
    /// Check at most this many elements per input (sampled with `seed`); 0 means all.
    pub max_checks: usize,
    /// Seed for the projection weights and the element sample.
    pub seed: u64,
}

impl Default for GradCheck {
    fn default() -> Self {
        GradCheck { eps: 1e-2, atol: 2e-3, rtol: 1e-2, max_checks: 48, seed: 0x5eed }
    }
}

/// [`grad_check_with`] with default tolerances.
pub fn grad_check<F>(f: F, inputs: &[Tensor]) -> Result<()>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    grad_check_with(f, inputs, &GradCheck::default())
}

/// Compare autograd gradients of `f` with central finite differences.
///
/// `f` maps the inputs to a tensor of any shape; it is reduced to a scalar with fixed random weights so every
/// output element is exercised. Every `F32` input is checked (non-`F32` inputs, such as index tensors, are
/// passed through). The loss is accumulated in `f64` to keep the finite differences accurate. Returns an
/// [`Error::Autograd`] naming the input, element, and both values on the first mismatch.
pub fn grad_check_with<F>(f: F, inputs: &[Tensor], opts: &GradCheck) -> Result<()>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let leaves: Vec<Tensor> = inputs
        .iter()
        .map(|x| {
            if x.dtype() == DType::F32 {
                Ok(Tensor::from_vec(x.to_vec::<f32>()?, x.shape())?.requires_grad_())
            } else {
                Ok(x.clone())
            }
        })
        .collect::<Result<_>>()?;
    let y = f(&leaves)?;
    let mut rng = Rng::seed(opts.seed);
    let weights = Tensor::rand_uniform(y.shape(), -1.0, 1.0, &mut rng);
    let wv: Vec<f64> = weights.to_vec::<f32>()?.iter().map(|&v| v as f64).collect();
    let grads = y.mul(&weights)?.sum_all()?.backward()?;

    let loss = |xs: &[Tensor]| -> Result<f64> {
        let _g = no_grad();
        let y = f(xs)?;
        Ok(y.to_vec::<f32>()?.iter().zip(&wv).map(|(&a, &w)| a as f64 * w).sum())
    };

    for (i, leaf) in leaves.iter().enumerate() {
        if leaf.dtype() != DType::F32 {
            continue;
        }
        let base = leaf.to_vec::<f32>()?;
        let analytic = match grads.get(leaf) {
            Some(g) => g.to_vec::<f32>()?,
            None => vec![0.0; base.len()],
        };
        let mut elems: Vec<usize> = (0..base.len()).collect();
        if opts.max_checks > 0 && elems.len() > opts.max_checks {
            rng.shuffle(&mut elems);
            elems.truncate(opts.max_checks);
            elems.sort_unstable();
        }
        for &e in &elems {
            let probe = |delta: f64| -> Result<f64> {
                let mut v = base.clone();
                v[e] = (v[e] as f64 + delta) as f32;
                let mut xs = leaves.clone();
                xs[i] = Tensor::from_vec(v, leaf.shape())?;
                loss(&xs)
            };
            let numeric = (probe(opts.eps)? - probe(-opts.eps)?) / (2.0 * opts.eps);
            let a = analytic[e] as f64;
            if (a - numeric).abs() > opts.atol + opts.rtol * a.abs().max(numeric.abs()) {
                return Err(Error::Autograd(format!(
                    "grad_check: input {i} (shape {}) element {e}: analytic {a:.6} vs numeric {numeric:.6}",
                    leaf.shape()
                )));
            }
        }
    }
    Ok(())
}
