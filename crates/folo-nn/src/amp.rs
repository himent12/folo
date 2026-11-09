//! Dynamic loss scaling for mixed-precision training (PyTorch's `torch.cuda.amp.GradScaler`).
//!
//! An `F16` backward pass loses gradients below `2^-24` to underflow, so the loss is multiplied by a scale `S`
//! before `backward()` and the gradients are divided by `S` again before clipping and the optimizer step
//! ([`crate::optim::GradAccumulator::unscale_clip`]). `S` adapts: a step whose gradients overflowed (any
//! non-finite value) is skipped and `S` shrinks by [`GradScaler::backoff_factor`]; after
//! [`GradScaler::growth_interval`] finite steps in a row it grows by [`GradScaler::growth_factor`]. The defaults
//! are PyTorch's: `S = 2^16`, growth 2 every 2000 steps, backoff 0.5.
//!
//! The state is part of a training checkpoint (docs/FORMAT.md, `folo.amp`): [`GradScaler::contract`] /
//! [`GradScaler::from_contract`] round-trip it exactly (every scale reached from a power of two by the default
//! factors is a power of two).

use folo_core::{DType, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// PyTorch's initial loss scale, `2^16`.
pub const INIT_SCALE: f64 = 65536.0;
/// Growth factor after [`GROWTH_INTERVAL`] finite steps.
pub const GROWTH_FACTOR: f64 = 2.0;
/// Backoff factor after a step with non-finite gradients.
pub const BACKOFF_FACTOR: f64 = 0.5;
/// Finite steps in a row before the scale grows.
pub const GROWTH_INTERVAL: u64 = 2000;

/// Multiply every tensor of `grads` by `inv_scale` in place and return whether every result is finite: the loss
/// scaler's unscale and overflow check. On a device each tensor takes one fused kernel
/// ([`Tensor::unscale_check_`]) that also raises a shared device flag, and the flag is read back once; tensors the
/// kernel does not fit, and CPU tensors, use tensor ops. The tensors must not share storage (pass private copies,
/// such as [`crate::optim::GradAccumulator`]'s buffers: one backward pass may hand several parameters views of
/// one gradient).
pub fn unscale_(grads: &[&Tensor], inv_scale: f32) -> Result<bool> {
    let _g = no_grad();
    let mut flags: Vec<Tensor> = Vec::new();
    let mut sums: Vec<Tensor> = Vec::new();
    let mut finite = true;
    for g in grads {
        let dev = g.device();
        if !dev.is_cpu() {
            if !flags.iter().any(|f| f.device() == dev) {
                flags.push(Tensor::zeros_with([1], DType::U32, dev)?);
            }
            let flag = flags.iter().find(|f| f.device() == dev).expect("pushed");
            if g.unscale_check_(inv_scale, flag)? {
                continue;
            }
        }
        g.mul_scalar_assign(inv_scale)?;
        if dev.is_cpu() && g.dtype() == DType::F32 {
            finite &= g.contiguous()?.with_f32_slice(|s| s.iter().all(|v| v.is_finite()))?;
        } else {
            // Σ 0·x is 0 exactly when every element is finite.
            sums.push(g.mul_scalar(0.0)?.sum_all()?.to_dtype(DType::F32)?.to_device(folo_core::Device::Cpu)?);
        }
    }
    for f in &flags {
        finite &= f.to_vec::<u32>()?[0] == 0;
    }
    for s in &sums {
        finite &= s.item::<f32>()? == 0.0;
    }
    Ok(finite)
}

/// What [`GradScaler::update`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleUpdate {
    /// The step was taken; the scale is unchanged.
    Kept,
    /// The step was taken and completed a growth interval: the scale grew.
    Grew,
    /// The gradients were not finite: the step was skipped and the scale backed off.
    BackedOff,
}

/// The `folo.amp` checkpoint entry (docs/FORMAT.md): the scaler's settings and state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmpContract {
    /// Current loss scale.
    pub scale: f64,
    /// [`GradScaler::growth_factor`].
    pub growth_factor: f64,
    /// [`GradScaler::backoff_factor`].
    pub backoff_factor: f64,
    /// [`GradScaler::growth_interval`].
    pub growth_interval: u64,
    /// Finite steps since the last growth or backoff.
    pub growth_tracker: u64,
    /// Steps skipped for non-finite gradients over the whole run.
    pub skipped_steps: u64,
}

/// Dynamic loss scaling (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: u64,
    growth_tracker: u64,
    skipped_steps: u64,
}

impl Default for GradScaler {
    fn default() -> Self {
        GradScaler {
            scale: INIT_SCALE,
            growth_factor: GROWTH_FACTOR,
            backoff_factor: BACKOFF_FACTOR,
            growth_interval: GROWTH_INTERVAL,
            growth_tracker: 0,
            skipped_steps: 0,
        }
    }
}

impl GradScaler {
    /// PyTorch's defaults: scale `2^16`, growth 2 every 2000 finite steps, backoff 0.5.
    pub fn new() -> Self {
        Self::default()
    }

    /// Explicit settings. `init_scale` must be positive and finite, `growth_factor > 1`,
    /// `0 < backoff_factor < 1` and `growth_interval >= 1`.
    pub fn with_options(
        init_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: u64,
    ) -> Result<Self> {
        let s = GradScaler {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            ..Self::default()
        };
        s.validate()?;
        Ok(s)
    }

    fn validate(&self) -> Result<()> {
        let ok = self.scale.is_finite()
            && self.scale > 0.0
            && self.growth_factor.is_finite()
            && self.growth_factor > 1.0
            && self.backoff_factor > 0.0
            && self.backoff_factor < 1.0
            && self.growth_interval >= 1
            && self.growth_tracker < self.growth_interval;
        if !ok {
            return Err(Error::Invalid(format!("invalid loss scaler settings or state: {self:?}")));
        }
        Ok(())
    }

    /// The current loss scale `S`: backward runs on `loss · S`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// `1 / S` as `f32`, the factor that unscales the gradients (exact: `S` is a power of two in practice).
    pub fn inv_scale(&self) -> f32 {
        (1.0 / self.scale) as f32
    }

    /// Growth factor.
    pub fn growth_factor(&self) -> f64 {
        self.growth_factor
    }

    /// Backoff factor.
    pub fn backoff_factor(&self) -> f64 {
        self.backoff_factor
    }

    /// Finite steps in a row before the scale grows.
    pub fn growth_interval(&self) -> u64 {
        self.growth_interval
    }

    /// Finite steps since the last growth or backoff.
    pub fn growth_tracker(&self) -> u64 {
        self.growth_tracker
    }

    /// Steps skipped for non-finite gradients so far.
    pub fn skipped_steps(&self) -> u64 {
        self.skipped_steps
    }

    /// Record the outcome of an optimizer step (`found_inf`: some unscaled gradient was not finite, so the step
    /// was skipped), as `torch`'s `GradScaler.update`: a skip multiplies the scale by the backoff factor and
    /// restarts the growth count; otherwise the count advances and, at the growth interval, the scale grows (if
    /// the grown scale is still finite) and the count restarts.
    pub fn update(&mut self, found_inf: bool) -> ScaleUpdate {
        if found_inf {
            self.scale *= self.backoff_factor;
            self.growth_tracker = 0;
            self.skipped_steps += 1;
            return ScaleUpdate::BackedOff;
        }
        self.growth_tracker += 1;
        if self.growth_tracker < self.growth_interval {
            return ScaleUpdate::Kept;
        }
        self.growth_tracker = 0;
        let grown = self.scale * self.growth_factor;
        if grown.is_finite() {
            self.scale = grown;
            ScaleUpdate::Grew
        } else {
            ScaleUpdate::Kept
        }
    }

    /// The `folo.amp` checkpoint entry.
    pub fn contract(&self) -> AmpContract {
        AmpContract {
            scale: self.scale,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
            growth_tracker: self.growth_tracker,
            skipped_steps: self.skipped_steps,
        }
    }

    /// The scaler a `folo.amp` entry describes (validated as [`GradScaler::with_options`], plus
    /// `growth_tracker < growth_interval`).
    pub fn from_contract(c: &AmpContract) -> Result<Self> {
        let s = GradScaler {
            scale: c.scale,
            growth_factor: c.growth_factor,
            backoff_factor: c.backoff_factor,
            growth_interval: c.growth_interval,
            growth_tracker: c.growth_tracker,
            skipped_steps: c.skipped_steps,
        };
        s.validate()?;
        Ok(s)
    }

    /// The `folo.amp` metadata value: [`GradScaler::contract`] as compact JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.contract()).expect("the contract serialises")
    }

    /// Parses a `folo.amp` metadata value ([`GradScaler::to_json`]).
    pub fn from_json(s: &str) -> Result<Self> {
        let c: AmpContract = serde_json::from_str(s)
            .map_err(|e| Error::Metadata(format!("folo.amp is not a loss scaler state: {e}")))?;
        Self::from_contract(&c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_torch_defaults() {
        let s = GradScaler::new();
        assert_eq!(s.scale(), 65536.0);
        assert_eq!((s.growth_factor(), s.backoff_factor(), s.growth_interval()), (2.0, 0.5, 2000));
        assert_eq!(s.inv_scale(), 1.0 / 65536.0);
    }

    #[test]
    fn overflow_skips_and_backs_off() {
        let mut s = GradScaler::new();
        for _ in 0..5 {
            assert_eq!(s.update(false), ScaleUpdate::Kept);
        }
        assert_eq!(s.growth_tracker(), 5);
        assert_eq!(s.update(true), ScaleUpdate::BackedOff);
        assert_eq!((s.scale(), s.growth_tracker(), s.skipped_steps()), (32768.0, 0, 1));
        assert_eq!(s.update(true), ScaleUpdate::BackedOff);
        assert_eq!((s.scale(), s.skipped_steps()), (16384.0, 2));
    }

    #[test]
    fn grows_after_the_interval_of_finite_steps() {
        let mut s = GradScaler::with_options(8.0, 2.0, 0.5, 3).unwrap();
        assert_eq!([s.update(false), s.update(false)], [ScaleUpdate::Kept; 2]);
        assert_eq!(s.update(false), ScaleUpdate::Grew);
        assert_eq!((s.scale(), s.growth_tracker()), (16.0, 0));
        // An overflow restarts the count: two more finite steps do not grow it.
        assert_eq!(s.update(true), ScaleUpdate::BackedOff);
        assert_eq!([s.update(false), s.update(false)], [ScaleUpdate::Kept; 2]);
        assert_eq!(s.scale(), 8.0);
        assert_eq!(s.update(false), ScaleUpdate::Grew);
        assert_eq!(s.scale(), 16.0);
        // The default interval: 2000 finite steps double the scale exactly once.
        let mut d = GradScaler::new();
        let grew = (0..2000).filter(|_| d.update(false) == ScaleUpdate::Grew).count();
        assert_eq!((grew, d.scale(), d.growth_tracker()), (1, 131072.0, 0));
    }

    #[test]
    fn growth_stops_at_infinity() {
        let mut s = GradScaler::with_options(f64::MAX, 2.0, 0.5, 1).unwrap();
        assert_eq!(s.update(false), ScaleUpdate::Kept);
        assert_eq!(s.scale(), f64::MAX);
    }

    #[test]
    fn contract_round_trips_exactly() {
        let mut s = GradScaler::new();
        for i in 0..37 {
            s.update(i % 11 == 3);
        }
        let back = GradScaler::from_json(&s.to_json()).unwrap();
        assert_eq!(back, s);
        assert_eq!(back.scale().to_bits(), s.scale().to_bits());
        assert!(GradScaler::from_json("{\"scale\": 1.0}").is_err());
        let mut c = s.contract();
        c.growth_tracker = c.growth_interval;
        assert!(GradScaler::from_contract(&c).is_err());
        for bad in [(0.0, 2.0, 0.5, 1), (1.0, 1.0, 0.5, 1), (1.0, 2.0, 1.0, 1), (1.0, 2.0, 0.5, 0)] {
            assert!(GradScaler::with_options(bad.0, bad.1, bad.2, bad.3).is_err(), "{bad:?}");
        }
    }
}
