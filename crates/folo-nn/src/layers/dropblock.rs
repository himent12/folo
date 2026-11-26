//! Structured spatial block dropout ([`DropBlock2d`]; Ghiasi, Lin & Le, NeurIPS 2018 /
//! arXiv:1810.12890).
//!
//! Unlike element-wise dropout, DropBlock drops contiguous `block_size × block_size` spatial regions from
//! `NCHW` feature maps so convolutional layers cannot trivially recover dropped activations from adjacent
//! pixels.
//!
//! For an input feature map `[N, C, H, W]` and effective block size `bs = min(block_size, H, W)`:
//! ```text
//! valid_h = H - bs + 1
//! valid_w = W - bs + 1
//! gamma   = clamp((drop_prob / bs^2) * (H * W) / (valid_h * valid_w), 0, 1)
//! ```
//!
//! Seed centers are sampled independently with probability `gamma` over `valid_h × valid_w` per `(n, c)`
//! channel using a seeded internal [`Rng`], dilated in exact `O(H · W)` time into `bs × bs` zero blocks, and
//! remaining kept elements are rescaled by `numel / kept` (`0` if all elements are dropped) so activation
//! magnitudes match evaluation mode (`eval()` is a zero-allocation identity).

use std::sync::Mutex;

use folo_core::rng::Rng;
use folo_core::{DType, Tensor};

use crate::module::{Forward, Mode, Module};
use crate::{Error, Result};

/// Compute DropBlock2d's Bernoulli seed-center sampling rate `gamma in [0, 1]` (Ghiasi et al., NeurIPS 2018
/// Eq. 1) for `drop_prob in [0, 1)`, odd `block_size >= 1`, and feature-map spatial size `height × width`.
///
/// When `block_size > min(height, width)`, clamps the effective block size to `bs = min(block_size, height,
/// width)` so `valid_h = height - bs + 1 >= 1` and `valid_w = width - bs + 1 >= 1` never divide by zero.
pub fn dropblock_gamma(drop_prob: f32, block_size: usize, height: usize, width: usize) -> Result<f64> {
    validate_dropblock_params(drop_prob, block_size)?;
    if height == 0 || width == 0 {
        return Err(Error::Invalid(format!(
            "dropblock_gamma: spatial dims must be >= 1, got {height}x{width}"
        )));
    }
    let bs = block_size.min(height).min(width);
    let valid_h = height - bs + 1;
    let valid_w = width - bs + 1;
    let num = (drop_prob as f64) * (height as f64) * (width as f64);
    let den = (bs as f64) * (bs as f64) * (valid_h as f64) * (valid_w as f64);
    Ok((num / den).clamp(0.0, 1.0))
}

fn validate_dropblock_params(drop_prob: f32, block_size: usize) -> Result<()> {
    if !(drop_prob.is_finite() && (0.0..1.0).contains(&drop_prob)) {
        return Err(Error::Invalid(format!(
            "DropBlock2d: drop_prob must be finite and in [0, 1), got {drop_prob}"
        )));
    }
    if block_size == 0 || block_size.is_multiple_of(2) || block_size.checked_mul(block_size).is_none() {
        return Err(Error::Invalid(format!(
            "DropBlock2d: block_size must be an odd integer >= 1 without overflow, got {block_size}"
        )));
    }
    Ok(())
}

/// Structured 2-D spatial block dropout over `NCHW` tensors (Ghiasi et al., NeurIPS 2018).
///
/// Owns a [`Mode`] flag (automatically toggled by [`Module::train`] / [`Module::eval`]) and a seeded
/// [`Rng`] behind a [`Mutex`] for deterministic, thread-safe sampling.
pub struct DropBlock2d {
    drop_prob: f32,
    block_size: usize,
    mode: Mode,
    rng: Mutex<Rng>,
}

impl DropBlock2d {
    /// Construct a [`DropBlock2d`] layer with `drop_prob in [0, 1)`, odd `block_size >= 1`, and RNG `seed`.
    pub fn try_new(drop_prob: f32, block_size: usize, seed: u64) -> Result<Self> {
        validate_dropblock_params(drop_prob, block_size)?;
        Ok(DropBlock2d { drop_prob, block_size, mode: Mode::new(), rng: Mutex::new(Rng::seed(seed)) })
    }

    /// Construct a [`DropBlock2d`] layer, panicking on invalid `drop_prob` or `block_size`.
    pub fn new(drop_prob: f32, block_size: usize, seed: u64) -> Self {
        Self::try_new(drop_prob, block_size, seed).expect("DropBlock2d: invalid configuration")
    }

    /// Target drop probability `p in [0, 1)`.
    pub fn drop_prob(&self) -> f32 {
        self.drop_prob
    }

    /// Configured odd spatial block size `>= 1`.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Reset the internal RNG stream to `seed` (useful for deterministic replay or finite-difference checks).
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
            .ok_or_else(|| Error::Invalid("DropBlock2d: all-zero RNG state is invalid".into()))?;
        *self.rng.lock().unwrap_or_else(|e| e.into_inner()) = rng;
        Ok(())
    }
}

impl Module for DropBlock2d {
    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}

impl Forward for DropBlock2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !matches!(x.dtype(), DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!(
                "DropBlock2d: input dtype must be f32 or f16, got {}",
                x.dtype()
            )));
        }
        let (n, c, h, w) = x.dims4().map_err(|_| {
            Error::Invalid(format!("DropBlock2d: input must be 4-D NCHW, got {:?}", x.dims()))
        })?;
        if c == 0 || h == 0 || w == 0 {
            return Err(Error::Invalid(format!(
                "DropBlock2d: channels and spatial dims must be >= 1, got [{n}, {c}, {h}, {w}]"
            )));
        }
        if !self.mode.is_training() || self.drop_prob == 0.0 || n == 0 {
            return Ok(x.clone());
        }

        let bs = self.block_size.min(h).min(w);
        let valid_h = h - bs + 1;
        let valid_w = w - bs + 1;
        let gamma = dropblock_gamma(self.drop_prob, self.block_size, h, w)?;

        let plane = h * w;
        let numel = n * c * plane;
        let mut mask = vec![1.0f32; numel];
        let mut kept: usize = 0;

        {
            let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
            let mut row_drop = vec![false; valid_h * w];
            for nc in 0..(n * c) {
                // Pass 1: sample seed centers on [0..valid_h, 0..valid_w] and dilate horizontally by `bs`.
                for sy in 0..valid_h {
                    let row = &mut row_drop[sy * w..(sy + 1) * w];
                    let mut last_seed: Option<usize> = None;
                    for (col, cell) in row.iter_mut().enumerate() {
                        if col < valid_w && rng.bernoulli(gamma) {
                            last_seed = Some(col);
                        }
                        *cell = last_seed.is_some_and(|lx| col - lx < bs);
                    }
                }
                // Pass 2: dilate vertically by `bs` into the output `[h, w]` mask slice.
                let out_plane = &mut mask[nc * plane..(nc + 1) * plane];
                for col in 0..w {
                    let mut last_row: Option<usize> = None;
                    for y in 0..h {
                        if y < valid_h && row_drop[y * w + col] {
                            last_row = Some(y);
                        }
                        if last_row.is_some_and(|ly| y - ly < bs) {
                            out_plane[y * w + col] = 0.0;
                        } else {
                            kept += 1;
                        }
                    }
                }
            }
        }

        let scale = if kept == 0 { 0.0f32 } else { (numel as f64 / kept as f64) as f32 };
        let keep_mask = Tensor::from_vec(mask, [n, c, h, w])?.to_device(x.device())?.gt_scalar(0.0)?;
        let x_f32 = if x.dtype() == DType::F16 { x.to_dtype(DType::F32)? } else { x.clone() };
        let zeros = Tensor::zeros_with([n, c, h, w], DType::F32, x.device())?;
        // Select `0.0` directly via `where_cond` instead of multiplying by `0.0` so dropped `±Inf` values
        // become exact `0.0` rather than `NaN`. Widening `F16` to `F32` also supports CPU `F16` tensors
        // and prevents `F16::MAX` overflow when `scale > 65_504.0`.
        let out_f32 = if kept == 0 {
            keep_mask.where_cond(&x_f32, &zeros)?
        } else {
            let scaled = if scale == 1.0 { x_f32 } else { x_f32.mul_scalar(scale)? };
            keep_mask.where_cond(&scaled, &zeros)?
        };
        if x.dtype() == DType::F16 { Ok(out_f32.to_dtype(DType::F16)?) } else { Ok(out_f32) }
    }
}
