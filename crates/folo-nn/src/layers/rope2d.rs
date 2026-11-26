//! 2D Rotary Positional Embedding ([`RoPE2d`]; Su et al. 2021 / Heo et al. ECCV 2024 Vision RoPE).
//!
//! Splits each head vector of dimension `head_dim` (must be positive and divisible by 4) into two equal
//! halves:
//! - Channels `0 .. head_dim / 2` encode the vertical (height / row) coordinate `y in 0..H` via `head_dim / 4`
//!   orthogonal 2-D rotations using inverse frequencies `theta_k = base^(-4k / head_dim)`.
//! - Channels `head_dim / 2 .. head_dim` encode the horizontal (width / column) coordinate `x in 0..W` via
//!   the same `head_dim / 4` inverse frequencies.
//!
//! Each half uses the half-split pairing `(u, v) -> (u cos(phi) - v sin(phi), u sin(phi) + v cos(phi))`,
//! guaranteeing exact L2 norm preservation and exact 2-D relative-position inner-product invariance
//! (`<RoPE2d(q, y1, x1), RoPE2d(k, y2, x2)>` depends only on `(y1 - y2, x1 - x2)`).
//!
//! Stores `inv_freq: [head_dim / 4]` as a non-trainable [`Module`] buffer so `.folo` save/load and
//! `Module::to_device` round-trip cleanly.

use folo_core::{DType, Tensor};

use crate::module::{Module, TensorKind};
use crate::{Error, Result};

/// Default RoPE frequency base (`10_000.0`).
pub const ROPE2D_DEFAULT_BASE: f64 = 10_000.0;

/// 2D axial Rotary Positional Embedding module.
///
/// Tensor name (buffer): `inv_freq` of shape `[head_dim / 4]`.
pub struct RoPE2d {
    /// Inverse angular frequencies `theta_k = coord_scale * base^(-4k / head_dim)` (`[head_dim / 4]`, buffer).
    pub inv_freq: Tensor,
    head_dim: usize,
    base: f64,
    coord_scale: f64,
}

impl RoPE2d {
    /// Create a [`RoPE2d`] module with the default frequency base `10_000.0` and unit coordinate scale.
    ///
    /// # Errors
    /// Returns [`Error::Invalid`] if `head_dim == 0` or `head_dim % 4 != 0`.
    pub fn new(head_dim: usize) -> Result<Self> {
        Self::with_options(head_dim, ROPE2D_DEFAULT_BASE, 1.0)
    }

    /// Create a [`RoPE2d`] module with explicit frequency `base > 0` and coordinate scale `coord_scale > 0`.
    pub fn with_options(head_dim: usize, base: f64, coord_scale: f64) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return Err(Error::Invalid(format!(
                "RoPE2d: head_dim must be positive and divisible by 4, got {head_dim}"
            )));
        }
        if !(base.is_finite() && base > 0.0) {
            return Err(Error::Invalid(format!("RoPE2d: base must be finite and > 0, got {base}")));
        }
        if !(coord_scale.is_finite() && coord_scale > 0.0) {
            return Err(Error::Invalid(format!(
                "RoPE2d: coord_scale must be finite and > 0, got {coord_scale}"
            )));
        }
        let quarter = head_dim / 4;
        let half = (head_dim / 2) as f64;
        let mut freqs = Vec::with_capacity(quarter);
        for k in 0..quarter {
            let f = (coord_scale * base.powf(-(2.0 * k as f64) / half)) as f32;
            if !f.is_finite() {
                return Err(Error::Invalid(format!(
                    "RoPE2d: inv_freq[{k}] overflows f32 for base={base}, coord_scale={coord_scale}"
                )));
            }
            freqs.push(f);
        }
        let inv_freq = Tensor::from_vec(freqs, [quarter])?;
        Ok(Self { inv_freq, head_dim, base, coord_scale })
    }

    /// Head dimension `D` (`D % 4 == 0`).
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Construction-time frequency base. Note: loading an `inv_freq` buffer from a `.folo` file updates
    /// [`Self::inv_freq`] directly without changing this metadata field; construct with matching options if
    /// callers inspect `base()` after load.
    #[inline]
    pub fn base(&self) -> f64 {
        self.base
    }

    /// Construction-time coordinate scaling multiplier (see [`Self::base`] regarding `.folo` loads).
    #[inline]
    pub fn coord_scale(&self) -> f64 {
        self.coord_scale
    }

    /// Compute `(cos_y, sin_y, cos_x, sin_x)` tables of shape `[1, 1, height * width, head_dim / 4]`
    /// on `self.inv_freq`'s device and dtype `dtype` (tables are synthesized in `f64` on the host and
    /// uploaded to `self.inv_freq.device()`).
    fn tables(&self, height: usize, width: usize, dtype: DType) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let quarter = self.head_dim / 4;
        let len = height.checked_mul(width).ok_or_else(|| {
            Error::Invalid(format!("RoPE2d: height ({height}) * width ({width}) overflows"))
        })?;
        let freqs = self.inv_freq.to_dtype(DType::F32)?.to_vec::<f32>()?;
        if freqs.len() != quarter {
            return Err(Error::Invalid(format!(
                "RoPE2d: inv_freq length {} != head_dim / 4 ({quarter})",
                freqs.len()
            )));
        }
        if freqs.iter().any(|f| !f.is_finite()) {
            return Err(Error::Invalid("RoPE2d: inv_freq contains non-finite value".into()));
        }
        let mut cy = Vec::with_capacity(len * quarter);
        let mut sy = Vec::with_capacity(len * quarter);
        let mut cx = Vec::with_capacity(len * quarter);
        let mut sx = Vec::with_capacity(len * quarter);
        for y in 0..height {
            let yf = y as f64;
            for x in 0..width {
                let xf = x as f64;
                for &f in &freqs {
                    let phi_y = yf * f as f64;
                    let phi_x = xf * f as f64;
                    let (sv_y, cv_y) = phi_y.sin_cos();
                    let (sv_x, cv_x) = phi_x.sin_cos();
                    cy.push(cv_y as f32);
                    sy.push(sv_y as f32);
                    cx.push(cv_x as f32);
                    sx.push(sv_x as f32);
                }
            }
        }
        let dev = self.inv_freq.device();
        let make = |v: Vec<f32>| -> Result<Tensor> {
            Ok(Tensor::from_vec(v, [1, 1, len, quarter])?.to_dtype(dtype)?.to_device(dev)?)
        };
        Ok((make(cy)?, make(sy)?, make(cx)?, make(sx)?))
    }

    /// Apply 2D rotary positional embeddings to `x` of shape `[B, num_heads, H * W, head_dim]`,
    /// preserving shape, dtype, device, and full differentiability.
    pub fn apply(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        if !matches!(x.dtype(), DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!("RoPE2d: input dtype must be f32 or f16, got {}", x.dtype())));
        }
        if x.device() != self.inv_freq.device() {
            return Err(Error::Invalid(format!(
                "RoPE2d: input on {:?} does not match inv_freq on {:?}",
                x.device(),
                self.inv_freq.device()
            )));
        }
        let (b, heads, seq_len, d) = x
            .dims4()
            .map_err(|_| Error::Invalid(format!("RoPE2d: expected [B, H, L, D], got {:?}", x.dims())))?;
        if d != self.head_dim {
            return Err(Error::Invalid(format!("RoPE2d: last dim {d} != head_dim {}", self.head_dim)));
        }
        let expected_len = height.checked_mul(width).ok_or_else(|| {
            Error::Invalid(format!("RoPE2d: height ({height}) * width ({width}) overflows"))
        })?;
        if seq_len != expected_len {
            return Err(Error::Invalid(format!(
                "RoPE2d: sequence length {seq_len} != height * width ({height} * {width} = {expected_len})"
            )));
        }
        if b == 0 || heads == 0 || seq_len == 0 {
            return Ok(x.mul_scalar(1.0)?);
        }

        let quarter = d / 4;
        let half = d / 2;
        let (cos_y, sin_y, cos_x, sin_x) = self.tables(height, width, x.dtype())?;

        let y0 = x.narrow(3, 0, quarter)?;
        let y1 = x.narrow(3, quarter, quarter)?;
        let x0 = x.narrow(3, half, quarter)?;
        let x1 = x.narrow(3, half + quarter, quarter)?;

        let y0_rot = y0.mul(&cos_y)?.sub(&y1.mul(&sin_y)?)?;
        let y1_rot = y0.mul(&sin_y)?.add(&y1.mul(&cos_y)?)?;
        let x0_rot = x0.mul(&cos_x)?.sub(&x1.mul(&sin_x)?)?;
        let x1_rot = x0.mul(&sin_x)?.add(&x1.mul(&cos_x)?)?;

        Ok(Tensor::cat(&[&y0_rot, &y1_rot, &x0_rot, &x1_rot], 3)?)
    }

    /// Apply 2D rotary positional embeddings directly to an `NCHW` feature map `[B, num_heads * head_dim, H, W]`,
    /// returning the rotated `[B, C, H, W]` tensor.
    pub fn apply_nchw(&self, x: &Tensor, num_heads: usize) -> Result<Tensor> {
        if num_heads == 0 {
            return Err(Error::Invalid("RoPE2d::apply_nchw: num_heads must be >= 1, got 0".into()));
        }
        let (b, c, h, w) = x
            .dims4()
            .map_err(|_| Error::Invalid(format!("RoPE2d::apply_nchw: expected NCHW, got {:?}", x.dims())))?;
        let expected_c = num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::Invalid("RoPE2d::apply_nchw: num_heads * head_dim overflows".into()))?;
        if c != expected_c {
            return Err(Error::Invalid(format!(
                "RoPE2d::apply_nchw: channels {c} != num_heads * head_dim ({num_heads} * {})",
                self.head_dim
            )));
        }
        let seq_len = h.checked_mul(w).ok_or_else(|| {
            Error::Invalid(format!("RoPE2d::apply_nchw: height ({h}) * width ({w}) overflows"))
        })?;
        // [B, num_heads, head_dim, H*W] -> [B, num_heads, H*W, head_dim]
        let tokens = x.reshape([b, num_heads, self.head_dim, seq_len])?.transpose(2, 3)?.contiguous()?;
        let rotated = self.apply(&tokens, h, w)?;
        Ok(rotated.transpose(2, 3)?.contiguous()?.reshape([b, c, h, w])?)
    }
}

impl Module for RoPE2d {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("inv_freq", &self.inv_freq, TensorKind::Buffer);
    }
}
