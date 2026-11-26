//! Online Hard Example Mining (OHEM, Shrivastava et al. CVPR 2016) differentiable loss selector.
//!
//! Given a per-element loss tensor of any shape (`F32`) and an optional boolean positive-preservation
//! mask:
//! - When `positive_mask` is `None`, all elements form the mining pool and the top-`m` hardest
//!   examples (largest loss values) are selected according to [`OhemBudget`].
//! - When `positive_mask` is `Some(mask)`, all positive elements (`mask[i] == true`) are
//!   unconditionally preserved, and the top-`m` hardest negative elements (`mask[i] == false`) are
//!   mined from the negative pool.
//!
//! Ranking uses descending loss value with `-0.0` normalized to `+0.0` and deterministic tie-breaking
//! by ascending flat element index. Non-finite (`NaN` / `+Inf`) losses sort ahead of finite losses
//! so corrupted entries propagate `NaN` rather than being silently dropped whenever `m > 0`.
//!
//! Gradients route strictly through selected elements and are identically `0.0` for unselected
//! elements. For [`LossReduction::Mean`], the denominator is the count of selected elements `K`
//! (returning scalar `0.0` with zero gradient when `K == 0`).

use folo_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::vfl_qfl::LossReduction;
use crate::loss::invalid;

/// Selection budget controlling how many hard examples are mined from the candidate pool.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OhemBudget {
    /// Keep at most the `k` hardest candidates from the mining pool.
    TopK(usize),
    /// Keep `ceil(fraction * pool_len)` hardest candidates (`fraction` in `[0.0, 1.0]`; `0.0` selects none).
    Fraction(f32),
    /// Keep `max(ceil(ratio * num_positives), min_neg)` hardest negatives (`ratio >= 0.0`), clamped to
    /// the pool size. When `positive_mask` is `None`, `num_positives` is `0` so `min_neg` applies.
    NegPosRatio {
        /// Maximum negative-to-positive ratio (`>= 0.0`, e.g. `3.0` for SSD-style 3:1 mining).
        ratio: f32,
        /// Minimum number of hard negatives to mine even when `num_positives == 0`.
        min_neg: usize,
    },
}

impl Default for OhemBudget {
    fn default() -> Self {
        Self::Fraction(0.25)
    }
}

/// Configuration for [`ohem_select`] and [`OhemLoss`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OhemConfig {
    /// Hard-example mining budget (default [`OhemBudget::Fraction`]`(0.25)`).
    pub budget: OhemBudget,
    /// Minimum number of candidates to mine from the mining pool, clamped to `pool_len` (default `0`).
    pub min_kept: usize,
    /// Output reduction over selected elements (default [`LossReduction::Mean`]).
    ///
    /// - [`LossReduction::Mean`]: mean over the `K` selected elements (`0.0` when `K == 0`).
    /// - [`LossReduction::Sum`]: sum over the `K` selected elements (`0.0` when `K == 0`).
    /// - [`LossReduction::None`]: same shape as input with unselected elements zeroed.
    pub reduction: LossReduction,
}

impl Default for OhemConfig {
    fn default() -> Self {
        Self { budget: OhemBudget::default(), min_kept: 0, reduction: LossReduction::Mean }
    }
}

impl OhemConfig {
    /// Validate budget parameters (`Fraction` in `[0, 1]`, `NegPosRatio::ratio >= 0`, both finite).
    pub fn validate(&self) -> Result<()> {
        match self.budget {
            OhemBudget::TopK(_) => Ok(()),
            OhemBudget::Fraction(f) => {
                if !(f.is_finite() && (0.0..=1.0).contains(&f)) {
                    return Err(invalid(format!(
                        "ohem_select: fraction must be finite and in [0, 1], got {f}"
                    )));
                }
                Ok(())
            }
            OhemBudget::NegPosRatio { ratio, .. } => {
                if !(ratio.is_finite() && ratio >= 0.0) {
                    return Err(invalid(format!(
                        "ohem_select: neg_pos_ratio must be finite and >= 0, got {ratio}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// Compute the number of candidates to mine from a pool of `pool_len` elements given `num_positives`.
    fn target_mine_count(&self, pool_len: usize, num_positives: usize) -> usize {
        if pool_len == 0 {
            return 0;
        }
        let base = match self.budget {
            OhemBudget::TopK(k) => k,
            OhemBudget::Fraction(f) => {
                if f == 0.0 {
                    0
                } else {
                    let raw = (pool_len as f64 * f as f64).ceil();
                    if raw >= pool_len as f64 { pool_len } else { (raw as usize).max(1) }
                }
            }
            OhemBudget::NegPosRatio { ratio, min_neg } => {
                let raw = (num_positives as f64 * ratio as f64).ceil();
                let scaled = if raw >= pool_len as f64 { pool_len } else { raw as usize };
                scaled.max(min_neg)
            }
        };
        base.max(self.min_kept).min(pool_len)
    }
}

/// Output of [`ohem_select`] / [`OhemLoss::select`]: the differentiable loss tensor plus selection metadata.
#[derive(Clone, Debug)]
pub struct OhemSelection {
    /// Differentiable selected loss (scalar for `Mean`/`Sum`, masked input shape for `None`).
    pub loss: Tensor,
    /// Flat indices of all kept elements (preserved positives + mined hard examples), sorted ascending.
    pub selected_indices: Vec<usize>,
    /// Number of unconditionally preserved positive elements.
    pub num_positives: usize,
    /// Number of hard examples mined from the candidate pool.
    pub num_mined: usize,
}

/// Normalize `-0.0` to `+0.0` so signed zeroes tie-break deterministically by ascending index.
#[inline]
fn canonical_loss_key(v: f32) -> f32 {
    if v == 0.0 { 0.0 } else { v }
}

/// Differentiable Online Hard Example Mining (OHEM) loss selector over a per-element `F32` loss tensor.
///
/// - `per_element_loss`: differentiable `F32` tensor of arbitrary shape (`N = numel()`).
/// - `positive_mask`: optional flat boolean slice of length `N`; when `Some(mask)`, all `true`
///   entries are unconditionally preserved and mining runs on the `false` entries.
/// - `cfg`: mining budget, `min_kept`, and reduction mode.
///
/// Exact autograd gradients flow only into the selected elements (unselected elements receive `0.0`).
pub fn ohem_select(
    per_element_loss: &Tensor,
    positive_mask: Option<&[bool]>,
    cfg: &OhemConfig,
) -> Result<OhemSelection> {
    const OP: &str = "ohem_select";
    cfg.validate()?;
    if per_element_loss.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch { op: OP, expected: DType::F32, got: per_element_loss.dtype() });
    }
    let dev = per_element_loss.device();
    let shape = per_element_loss.shape().clone();
    let n = shape.numel();
    if let Some(mask) = positive_mask
        && mask.len() != n
    {
        return Err(invalid(format!("{OP}: positive_mask len {} != loss numel {n}", mask.len())));
    }

    if n == 0 {
        let out_t = match cfg.reduction {
            LossReduction::None => Tensor::zeros_with(&shape, DType::F32, dev)?,
            LossReduction::Sum | LossReduction::Mean => Tensor::scalar(0.0).to_device(dev)?,
        };
        let loss = Tensor::custom_op(OP, &[per_element_loss], out_t, move |_g| {
            Ok(vec![Some(Tensor::zeros_with(&shape, DType::F32, dev)?)])
        })?;
        return Ok(OhemSelection { loss, selected_indices: Vec::new(), num_positives: 0, num_mined: 0 });
    }

    let values = per_element_loss.to_vec::<f32>()?;
    let mut selected = Vec::new();
    let mut pool: Vec<(f32, usize)> = Vec::new();

    match positive_mask {
        Some(mask) => {
            for (idx, (&val, &is_pos)) in values.iter().zip(mask).enumerate() {
                if is_pos {
                    selected.push(idx);
                } else {
                    pool.push((canonical_loss_key(val), idx));
                }
            }
        }
        None => {
            pool.reserve(n);
            for (idx, &val) in values.iter().enumerate() {
                pool.push((canonical_loss_key(val), idx));
            }
        }
    }

    let num_positives = selected.len();
    let num_mined = cfg.target_mine_count(pool.len(), num_positives);

    if num_mined > 0 {
        let cmp = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
        if num_mined < pool.len() {
            pool.select_nth_unstable_by(num_mined - 1, cmp);
            pool.truncate(num_mined);
        }
        selected.reserve(num_mined);
        for &(_, idx) in &pool {
            selected.push(idx);
        }
    }
    selected.sort_unstable();

    let kept_count = selected.len();
    let selected_for_bwd = selected.clone();

    let loss = match cfg.reduction {
        LossReduction::None => {
            let mut out = vec![0.0f32; n];
            for &idx in &selected {
                out[idx] = values[idx];
            }
            let out_t = Tensor::from_vec(out, &shape)?.to_device(dev)?;
            Tensor::custom_op(OP, &[per_element_loss], out_t, move |g| {
                let gv = g.to_vec::<f32>()?;
                let mut grad = vec![0.0f32; n];
                for &idx in &selected_for_bwd {
                    grad[idx] = gv[idx];
                }
                Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
            })?
        }
        LossReduction::Sum | LossReduction::Mean => {
            let (out_val, scale) = if kept_count == 0 {
                (0.0f32, 0.0f64)
            } else {
                let sum: f64 = selected.iter().map(|&idx| values[idx] as f64).sum();
                let s = if cfg.reduction == LossReduction::Mean { 1.0 / (kept_count as f64) } else { 1.0 };
                ((sum * s) as f32, s)
            };
            let out_t = Tensor::scalar(out_val).to_device(dev)?;
            Tensor::custom_op(OP, &[per_element_loss], out_t, move |g| {
                let mut grad = vec![0.0f32; n];
                if !selected_for_bwd.is_empty() {
                    let gs = ((g.item::<f32>()? as f64) * scale) as f32;
                    for &idx in &selected_for_bwd {
                        grad[idx] = gs;
                    }
                }
                Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
            })?
        }
    };

    Ok(OhemSelection { loss, selected_indices: selected, num_positives, num_mined })
}

/// Reusable Online Hard Example Mining (OHEM) loss wrapper holding a validated [`OhemConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OhemLoss {
    cfg: OhemConfig,
}

impl OhemLoss {
    /// Construct a validated [`OhemLoss`] from `cfg`.
    pub fn new(cfg: OhemConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Underlying [`OhemConfig`].
    #[inline]
    pub fn config(&self) -> OhemConfig {
        self.cfg
    }

    /// Apply OHEM selection and return both the differentiable loss tensor and selection metadata.
    pub fn select(&self, per_element_loss: &Tensor, positive_mask: Option<&[bool]>) -> Result<OhemSelection> {
        ohem_select(per_element_loss, positive_mask, &self.cfg)
    }

    /// Apply OHEM selection and return the differentiable loss tensor.
    pub fn forward(&self, per_element_loss: &Tensor, positive_mask: Option<&[bool]>) -> Result<Tensor> {
        Ok(self.select(per_element_loss, positive_mask)?.loss)
    }
}
