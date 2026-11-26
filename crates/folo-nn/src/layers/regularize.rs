//! Stochastic depth ([`DropPath`]) and residual diagonal channel scaling ([`LayerScale`]).
//!
//! - [`DropPath`] (Huang et al. 2016): drops entire residual samples independently with probability `p ∈ [0, 1)`
//!   in training mode while rescaling surviving samples by `1 / (1 − p)` (inverted scaling so evaluation is an
//!   exact identity), driven by an internal seeded [`Rng`].
//! - [`LayerScale`] (Touvron et al. 2021): learnable per-channel diagonal multiplier `γ ⊙ x` initialized to a
//!   small constant (e.g. `1e-5` or `1e-4`), supporting both channels-first (`NCHW` / `NC`) and channels-last
//!   (`NLC` / `NC` / `NHWC`) layouts cleanly.

use std::sync::Mutex;

use folo_core::rng::Rng;
use folo_core::{DType, Tensor};

use crate::module::{Forward, Mode, Module, TensorKind};
use crate::{Error, Result};

/// Default [`LayerScale`] initialization value (`1e-5`, Touvron et al. 2021).
pub const LAYERSCALE_DEFAULT_INIT: f32 = 1e-5;

/// Per-sample Stochastic Depth (Huang et al. 2016).
///
/// In training mode with `drop_prob > 0`, independently keeps each sample `n ∈ 0..N` along batch axis 0 with
/// probability `1 − p`, scaling kept samples by `1 / (1 − p)` and zeroing dropped samples. In eval mode or
/// when `drop_prob == 0.0`, returns the input unchanged (zero-allocation identity fast path).
///
/// Owns a [`Mode`] flag so [`Module::train`] / [`Module::eval`] propagate automatically through containers,
/// and a seeded [`Rng`] behind a [`Mutex`] for deterministic, thread-safe sampling.
pub struct DropPath {
    drop_prob: f32,
    mode: Mode,
    rng: Mutex<Rng>,
}

impl DropPath {
    /// Construct a [`DropPath`] layer with drop probability `drop_prob ∈ [0, 1)` and RNG `seed`.
    ///
    /// # Errors
    /// Returns [`Error::Invalid`] if `drop_prob` is non-finite or outside `[0.0, 1.0)`.
    pub fn try_new(drop_prob: f32, seed: u64) -> Result<Self> {
        if !(drop_prob.is_finite() && (0.0..1.0).contains(&drop_prob)) {
            return Err(Error::Invalid(format!(
                "DropPath: drop_prob must be finite and in [0, 1), got {drop_prob}"
            )));
        }
        Ok(DropPath { drop_prob, mode: Mode::new(), rng: Mutex::new(Rng::seed(seed)) })
    }

    /// Construct a [`DropPath`] layer with `drop_prob ∈ [0, 1)` and RNG `seed`.
    ///
    /// # Panics
    /// If `drop_prob` is non-finite or outside `[0.0, 1.0)`.
    pub fn new(drop_prob: f32, seed: u64) -> Self {
        Self::try_new(drop_prob, seed).expect("DropPath: invalid drop_prob")
    }

    /// The configured drop probability `p ∈ [0, 1)`.
    pub fn drop_prob(&self) -> f32 {
        self.drop_prob
    }

    /// The keep probability `1 − p ∈ (0, 1]`.
    pub fn keep_prob(&self) -> f32 {
        1.0 - self.drop_prob
    }

    /// Reset the internal RNG stream to `seed` (useful for deterministic replay or gradient checks).
    pub fn reseed(&self, seed: u64) {
        *self.rng.lock().unwrap_or_else(|e| e.into_inner()) = Rng::seed(seed);
    }

    /// Snapshot the internal xoshiro256++ RNG state words.
    pub fn rng_state(&self) -> [u64; 4] {
        self.rng.lock().unwrap_or_else(|e| e.into_inner()).state()
    }

    /// Restore the internal xoshiro256++ RNG state words (errors on the all-zero state).
    pub fn set_rng_state(&self, state: [u64; 4]) -> Result<()> {
        let rng = Rng::from_state(state)
            .ok_or_else(|| Error::Invalid("DropPath: all-zero RNG state is invalid".into()))?;
        *self.rng.lock().unwrap_or_else(|e| e.into_inner()) = rng;
        Ok(())
    }
}

impl Module for DropPath {
    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}

impl Forward for DropPath {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.is_empty() {
            return Err(Error::Invalid("DropPath: input must have rank >= 1 (batch axis 0)".into()));
        }
        if !self.mode.is_training() || self.drop_prob == 0.0 || dims[0] == 0 {
            return Ok(x.clone());
        }
        let batch = dims[0];
        let keep_prob = 1.0 - self.drop_prob as f64;
        let inv_keep = (1.0 / keep_prob) as f32;
        let mask_vals: Vec<f32> = {
            let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
            (0..batch).map(|_| if rng.bernoulli(keep_prob) { inv_keep } else { 0.0 }).collect()
        };
        let mut mask_shape = vec![1usize; dims.len()];
        mask_shape[0] = batch;
        let mask = Tensor::from_vec(mask_vals, mask_shape)?.to_device(x.device())?;
        if x.dtype() == DType::F16 && inv_keep > 65_504.0 {
            // Avoid casting an inv_keep > F16::MAX to +Inf before multiplying (which would turn kept 0.0 into NaN).
            return Ok(x.to_dtype(DType::F32)?.mul(&mask)?.to_dtype(DType::F16)?);
        }
        let mask = if mask.dtype() != x.dtype() { mask.to_dtype(x.dtype())? } else { mask };
        Ok(x.mul(&mask)?)
    }
}

/// Channel axis layout for [`LayerScale`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// Channels-first (`dim = 1`, requiring `rank >= 2`: `[N, C]`, `[N, C, L]`, `[N, C, H, W]`, ...).
    /// Matches FOLO's `NCHW` vision tensors and 2-D `NC` tensors.
    ChannelsFirst,
    /// Channels-last (`dim = rank - 1`, requiring `rank >= 1`: `[C]`, `[N, C]`, `[N, L, C]`, `[N, H, W, C]`, ...).
    /// Matches transformer token sequences (`NLC`) and [`crate::Linear`] outputs (`NC`).
    ChannelsLast,
}

/// Learnable per-channel diagonal residual scaling `y = γ ⊙ x` (Touvron et al. 2021, CaiT).
///
/// Tensor name: `gamma` (`[channels]`, learnable parameter automatically placed in the no-decay `bn` group by
/// [`crate::optim::StandardGroups::from_module`]).
pub struct LayerScale {
    /// Per-channel scale vector `[channels]`.
    pub gamma: Tensor,
    layout: ChannelLayout,
}

impl LayerScale {
    /// Create a channels-first (`NCHW` / `NC`) [`LayerScale`] with `gamma` initialized to `init_value >= 0`.
    ///
    /// # Errors
    /// Returns [`Error::Invalid`] if `channels == 0` or `init_value` is non-finite or negative.
    pub fn new(channels: usize, init_value: f32) -> Result<Self> {
        Self::with_layout(channels, init_value, ChannelLayout::ChannelsFirst)
    }

    /// Create a channels-last (`NLC` / `NC` / `NHWC`) [`LayerScale`] with `gamma` initialized to `init_value >= 0`.
    ///
    /// # Errors
    /// Returns [`Error::Invalid`] if `channels == 0` or `init_value` is non-finite or negative.
    pub fn channels_last(channels: usize, init_value: f32) -> Result<Self> {
        Self::with_layout(channels, init_value, ChannelLayout::ChannelsLast)
    }

    /// Create a [`LayerScale`] with an explicit [`ChannelLayout`] and `gamma` initialized to `init_value >= 0`.
    ///
    /// # Errors
    /// Returns [`Error::Invalid`] if `channels == 0` or `init_value` is non-finite or negative.
    pub fn with_layout(channels: usize, init_value: f32, layout: ChannelLayout) -> Result<Self> {
        if channels == 0 {
            return Err(Error::Invalid("LayerScale: channels must be > 0".into()));
        }
        if !(init_value.is_finite() && init_value >= 0.0) {
            return Err(Error::Invalid(format!(
                "LayerScale: init_value must be finite and >= 0, got {init_value}"
            )));
        }
        let gamma = Tensor::full([channels], init_value).requires_grad_();
        Ok(LayerScale { gamma, layout })
    }

    /// Wrap an existing 1-D `[C]` (`C > 0`) `gamma` tensor with the given [`ChannelLayout`].
    pub fn from_tensor(gamma: Tensor, layout: ChannelLayout) -> Result<Self> {
        match gamma.dims() {
            &[c] if c > 0 => Ok(LayerScale { gamma, layout }),
            other => {
                Err(Error::Invalid(format!("LayerScale: gamma must be 1-D [C] with C > 0, got {other:?}")))
            }
        }
    }

    /// Number of channels `C`.
    pub fn channels(&self) -> usize {
        self.gamma.dims()[0]
    }

    /// The configured channel layout.
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }
}

impl Module for LayerScale {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("gamma", &self.gamma, TensorKind::Parameter);
    }
}

impl Forward for LayerScale {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let rank = dims.len();
        let c = self.channels();
        let scale_shape = match self.layout {
            ChannelLayout::ChannelsFirst => {
                if rank < 2 {
                    return Err(Error::Invalid(format!(
                        "LayerScale(ChannelsFirst): expected rank >= 2, got {dims:?}"
                    )));
                }
                if dims[1] != c {
                    return Err(Error::Invalid(format!(
                        "LayerScale(ChannelsFirst): input {dims:?} has {} channels at dim 1, expected {c}",
                        dims[1]
                    )));
                }
                let mut s = vec![1usize; rank];
                s[1] = c;
                s
            }
            ChannelLayout::ChannelsLast => {
                if rank < 1 {
                    return Err(Error::Invalid("LayerScale(ChannelsLast): scalar input".into()));
                }
                if dims[rank - 1] != c {
                    return Err(Error::Invalid(format!(
                        "LayerScale(ChannelsLast): input {dims:?} has {} channels at last dim, expected {c}",
                        dims[rank - 1]
                    )));
                }
                let mut s = vec![1usize; rank];
                s[rank - 1] = c;
                s
            }
        };
        let scale = self.gamma.reshape(scale_shape)?;
        if self.gamma.dtype() == DType::F32
            && x.dtype() == DType::F16
            && folo_core::autocast_dtype() == Some(DType::F16)
        {
            // Multiply in F32 under F16 autocast so broadcast-sum gradients into the F32 master `gamma`
            // accumulate in F32 without overflowing F16::MAX on large spatial planes.
            return Ok(x.to_dtype(DType::F32)?.mul(&scale)?.to_dtype(DType::F16)?);
        }
        Ok(x.mul(&scale)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::StandardGroups;
    use folo_core::testing::{GradCheck, grad_check_with};

    fn core_err(e: Error) -> folo_core::Error {
        match e {
            Error::Core(c) => c,
            other => folo_core::Error::InvalidArgument { op: "test", msg: other.to_string() },
        }
    }

    #[test]
    fn drop_path_modes_determinism_inverted_scaling_and_validation() {
        // Invalid drop_prob rejected.
        for bad in [-0.01f32, 1.0, 1.5, f32::NAN, f32::INFINITY] {
            assert!(DropPath::try_new(bad, 0).is_err(), "drop_prob={bad} should fail");
        }
        assert!(DropPath::new(0.2, 0).forward(&Tensor::scalar(1.0)).is_err());

        // p == 0.0 fast path and eval mode return exact identity without advancing RNG.
        let mut rng = Rng::seed(7);
        let x = Tensor::randn([8, 3, 2, 2], 0.0, 1.0, &mut rng);
        let dp0 = DropPath::new(0.0, 123);
        let s0 = dp0.rng_state();
        let y0 = dp0.forward(&x).unwrap();
        assert_eq!(y0.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        assert_eq!(dp0.rng_state(), s0, "p == 0 must not consume RNG");

        let dp = DropPath::new(0.5, 42);
        dp.eval();
        assert!(!dp.is_training());
        let s_eval = dp.rng_state();
        assert_eq!(dp.forward(&x).unwrap().to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        assert_eq!(dp.rng_state(), s_eval, "eval mode must not consume RNG");

        // Train mode: deterministic per-sample drop with inverted 1/(1-p) scaling; empty batch succeeds.
        dp.train();
        let y_a = dp.forward(&x).unwrap().to_vec::<f32>().unwrap();
        let dp_same = DropPath::new(0.5, 42);
        let y_b = dp_same.forward(&x).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(y_a, y_b, "identical seeds must produce identical DropPath masks");

        // Check every sample is either all-zero or scaled by 2.0 = 1 / (1 - 0.5).
        let x_vec = x.to_vec::<f32>().unwrap();
        let sample_len = 3 * 2 * 2;
        let (mut kept_samples, mut dropped_samples) = (0usize, 0usize);
        for n in 0..8 {
            let xs = &x_vec[n * sample_len..(n + 1) * sample_len];
            let ys = &y_a[n * sample_len..(n + 1) * sample_len];
            if ys.iter().all(|&v| v == 0.0) {
                dropped_samples += 1;
            } else {
                kept_samples += 1;
                for (&xv, &yv) in xs.iter().zip(ys) {
                    assert!((yv - 2.0 * xv).abs() < 1e-6);
                }
            }
        }
        assert!(kept_samples > 0 && dropped_samples > 0, "expected mix of kept and dropped samples");
        // Empty batch [0, 3, 2, 2] passes cleanly.
        assert_eq!(dp.forward(&Tensor::zeros([0, 3, 2, 2])).unwrap().dims(), &[0, 3, 2, 2]);
        // Zero RNG state rejected.
        assert!(dp.set_rng_state([0; 4]).is_err());
    }

    #[test]
    fn drop_path_grad_check_nchw_and_nlc() {
        let mut rng = Rng::seed(99);
        let dp = DropPath::new(0.4, 777);
        let opts = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 32, seed: 11 };
        for shape in [vec![6usize, 3, 2, 2], vec![6, 4, 5], vec![6, 4]] {
            let x = Tensor::randn(shape, 0.0, 1.0, &mut rng);
            grad_check_with(
                |inputs| {
                    dp.reseed(777);
                    dp.forward(&inputs[0]).map_err(core_err)
                },
                &[x],
                &opts,
            )
            .unwrap();
        }
    }

    #[test]
    fn layer_scale_shapes_validation_no_decay_and_grad_check() {
        // Validation rejects invalid channels / init_value / mismatched channel axes.
        assert!(LayerScale::new(0, 1e-5).is_err());
        assert!(LayerScale::new(4, -1e-5).is_err());
        assert!(LayerScale::new(4, f32::NAN).is_err());
        assert!(LayerScale::new(4, f32::INFINITY).is_err());
        assert!(LayerScale::from_tensor(Tensor::ones([2, 2]), ChannelLayout::ChannelsFirst).is_err());

        let ls_nchw = LayerScale::new(4, 1e-4).unwrap();
        assert_eq!(ls_nchw.channels(), 4);
        assert_eq!(ls_nchw.layout(), ChannelLayout::ChannelsFirst);
        assert_eq!(ls_nchw.named_parameters()[0].0, "gamma");
        assert!(ls_nchw.forward(&Tensor::ones([4])).is_err(), "ChannelsFirst needs rank >= 2");
        assert!(ls_nchw.forward(&Tensor::ones([2, 3, 4, 4])).is_err(), "wrong channel count");
        assert_eq!(ls_nchw.forward(&Tensor::zeros([0, 4, 2, 2])).unwrap().dims(), &[0, 4, 2, 2]);
        // ChannelsFirst also handles 2-D NC [N, C].
        let y_nc = ls_nchw.forward(&Tensor::ones([3, 4])).unwrap().to_vec::<f32>().unwrap();
        assert!(y_nc.iter().all(|&v| (v - 1e-4).abs() < 1e-8));

        // ChannelsLast handles NLC, NC, and 1-D [C], plus empty sequence [2, 0, C].
        let ls_nlc = LayerScale::channels_last(5, 1e-5).unwrap();
        assert_eq!(ls_nlc.layout(), ChannelLayout::ChannelsLast);
        assert!(ls_nlc.forward(&Tensor::scalar(1.0)).is_err());
        assert!(ls_nlc.forward(&Tensor::ones([2, 3, 4])).is_err());
        assert_eq!(ls_nlc.forward(&Tensor::zeros([2, 0, 5])).unwrap().dims(), &[2, 0, 5]);
        let y_nlc = ls_nlc.forward(&Tensor::ones([2, 3, 5])).unwrap().to_vec::<f32>().unwrap();
        assert!(y_nlc.iter().all(|&v| (v - 1e-5).abs() < 1e-9));

        // StandardGroups places LayerScale::gamma in the no-decay `bn` group.
        let sg = StandardGroups::from_module(&ls_nchw);
        assert_eq!(sg.bn.len(), 1);
        assert_eq!(sg.bn[0].0, "gamma");
        assert!(sg.decay.is_empty() && sg.bias.is_empty());

        // Finite-difference grad_check for both x and gamma on NCHW, NLC, and NC.
        let mut rng = Rng::seed(123);
        let opts = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 32, seed: 13 };
        for (shape, layout, c) in [
            (vec![2usize, 3, 4, 4], ChannelLayout::ChannelsFirst, 3usize),
            (vec![2, 4, 3], ChannelLayout::ChannelsLast, 3),
            (vec![3, 4], ChannelLayout::ChannelsFirst, 4),
            (vec![3, 4], ChannelLayout::ChannelsLast, 4),
        ] {
            let x = Tensor::randn(shape, 0.0, 1.0, &mut rng);
            let gamma = Tensor::rand_uniform([c], 0.1, 1.5, &mut rng);
            grad_check_with(
                |inputs| {
                    let ls = LayerScale::from_tensor(inputs[1].clone(), layout).map_err(core_err)?;
                    ls.forward(&inputs[0]).map_err(core_err)
                },
                &[x, gamma],
                &opts,
            )
            .unwrap();
        }
    }

    #[test]
    fn drop_path_and_layer_scale_f16_edge_cases() {
        // 1. DropPath with inv_keep > F16::MAX (65504): kept 0.0 must stay 0.0 (never NaN) and small kept values stay finite.
        let x_f16 =
            Tensor::from_slice(&[0.0f32, 1e-4, 0.0, 1e-4], [2, 2]).unwrap().to_dtype(DType::F16).unwrap();
        let dp = DropPath::try_new(0.99999, 1).unwrap();
        let mut saw_kept = false;
        for seed in 0..200_000u64 {
            dp.reseed(seed);
            let mut probe = Rng::seed(seed);
            let k0 = probe.bernoulli(1.0 - 0.99999f64);
            let k1 = probe.bernoulli(1.0 - 0.99999f64);
            if k0 || k1 {
                let y = dp.forward(&x_f16).unwrap();
                assert_eq!(y.dtype(), DType::F16);
                let vals = y.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
                assert!(
                    vals.iter().all(|v| v.is_finite()),
                    "unexpected non-finite F16 DropPath output: {vals:?}"
                );
                if k0 {
                    assert_eq!(vals[0], 0.0, "kept 0.0 must not become NaN");
                    assert!((vals[1] - 10.0).abs() < 0.2, "kept 1e-4 * 1e5 ~= 10, got {}", vals[1]);
                }
                if k1 {
                    assert_eq!(vals[2], 0.0, "kept 0.0 must not become NaN");
                    assert!((vals[3] - 10.0).abs() < 0.2, "kept 1e-4 * 1e5 ~= 10, got {}", vals[3]);
                }
                saw_kept = true;
                break;
            }
        }
        assert!(saw_kept, "expected to find a seed that keeps at least one sample");

        // 2. LayerScale with F32 master gamma under F16 autocast must keep dL/dgamma finite above F16::MAX (65504).
        let ls = LayerScale::new(1, 1.0).unwrap();
        let x = Tensor::ones([1, 1, 256, 256]).to_dtype(DType::F16).unwrap().requires_grad_();
        let (y_dtype, dgamma) = {
            let _ac = folo_core::autocast();
            let y = ls.forward(&x).unwrap();
            let dt = y.dtype();
            let loss = y.to_dtype(DType::F32).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let dg = grads.get(&ls.gamma).unwrap().item::<f32>().unwrap();
            (dt, dg)
        };
        assert_eq!(y_dtype, DType::F16);
        assert!(
            dgamma.is_finite() && (dgamma - 65_536.0).abs() < 1.0,
            "expected finite 65536.0, got {dgamma}"
        );
    }
}
