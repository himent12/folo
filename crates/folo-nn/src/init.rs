//! Weight initialisation, seeded through [`folo_core::rng::Rng`].
//!
//! Every function draws the tensor's elements in row-major order from `rng`, so a model initialised in
//! canonical order from one seed is identical on every machine (docs/MODEL.md §2.9). Fans follow PyTorch:
//! for a weight `[out, in, k...]`, `fan_in = in · Πk` and `fan_out = out · Πk` (for grouped convs `in` is
//! already `in / groups`).

use folo_core::rng::Rng;
use folo_core::{Shape, Tensor, no_grad};

use crate::{Error, Result};

/// Which fan a Kaiming init preserves the variance of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanMode {
    /// Forward pass (PyTorch's default).
    FanIn,
    /// Backward pass.
    FanOut,
}

/// `(fan_in, fan_out)` of a weight shape, PyTorch's `_calculate_fan_in_and_fan_out`. Needs rank ≥ 2; a rank-1
/// shape (a bias) gives `(n, n)`.
pub fn fans(shape: &Shape) -> (usize, usize) {
    let d = shape.dims();
    match d.len() {
        0 => (1, 1),
        1 => (d[0], d[0]),
        _ => {
            let rf: usize = d[2..].iter().product();
            (d[1] * rf, d[0] * rf)
        }
    }
}

/// The recommended gain for an activation with a leaky-ReLU slope `a` (`sqrt(2 / (1 + a²))`; `a = 0` is ReLU).
pub fn leaky_relu_gain(a: f64) -> f64 {
    (2.0 / (1.0 + a * a)).sqrt()
}

fn fan(shape: &Shape, mode: FanMode) -> Result<usize> {
    let (fi, fo) = fans(shape);
    let f = match mode {
        FanMode::FanIn => fi,
        FanMode::FanOut => fo,
    };
    if f == 0 {
        return Err(Error::Invalid(format!("init: zero fan for shape {shape}")));
    }
    Ok(f)
}

/// `U(-bound, bound)` samples of `shape`.
pub fn uniform(shape: impl Into<Shape>, bound: f64, rng: &mut Rng) -> Tensor {
    let b = bound as f32;
    Tensor::rand_uniform(shape, -b, b, rng)
}

/// `N(0, std²)` samples of `shape`.
pub fn normal(shape: impl Into<Shape>, std: f64, rng: &mut Rng) -> Tensor {
    Tensor::randn(shape, 0.0, std as f32, rng)
}

/// Kaiming (He) uniform: `U(-b, b)`, `b = gain · sqrt(3 / fan)`, `gain = sqrt(2 / (1 + a²))`.
///
/// `a = √5` with [`FanMode::FanIn`] is PyTorch's default conv/linear init, `b = 1 / sqrt(fan_in)`, which is
/// what docs/MODEL.md §2.9 specifies (see [`default_conv_bound`]).
pub fn kaiming_uniform(shape: impl Into<Shape>, a: f64, mode: FanMode, rng: &mut Rng) -> Result<Tensor> {
    let shape = shape.into();
    let bound = leaky_relu_gain(a) * (3.0 / fan(&shape, mode)? as f64).sqrt();
    Ok(uniform(shape, bound, rng))
}

/// Kaiming (He) normal: `N(0, std²)`, `std = gain / sqrt(fan)`.
pub fn kaiming_normal(shape: impl Into<Shape>, a: f64, mode: FanMode, rng: &mut Rng) -> Result<Tensor> {
    let shape = shape.into();
    let std = leaky_relu_gain(a) / (fan(&shape, mode)? as f64).sqrt();
    Ok(normal(shape, std, rng))
}

/// Xavier (Glorot) uniform: `U(-b, b)`, `b = gain · sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_uniform(shape: impl Into<Shape>, gain: f64, rng: &mut Rng) -> Result<Tensor> {
    let shape = shape.into();
    let (fi, fo) = fans(&shape);
    if fi + fo == 0 {
        return Err(Error::Invalid(format!("init: zero fans for shape {shape}")));
    }
    Ok(uniform(shape, gain * (6.0 / (fi + fo) as f64).sqrt(), rng))
}

/// Xavier (Glorot) normal: `N(0, std²)`, `std = gain · sqrt(2 / (fan_in + fan_out))`.
pub fn xavier_normal(shape: impl Into<Shape>, gain: f64, rng: &mut Rng) -> Result<Tensor> {
    let shape = shape.into();
    let (fi, fo) = fans(&shape);
    if fi + fo == 0 {
        return Err(Error::Invalid(format!("init: zero fans for shape {shape}")));
    }
    Ok(normal(shape, gain * (2.0 / (fi + fo) as f64).sqrt(), rng))
}

/// The bound `1 / sqrt(fan_in)` of docs/MODEL.md §2.9 (PyTorch's default `kaiming_uniform(a = √5)`), used for
/// conv and linear weights and biases. `fan_in = (c_in / groups) · k · k`.
pub fn default_conv_bound(fan_in: usize) -> f64 {
    1.0 / (fan_in.max(1) as f64).sqrt()
}

/// Overwrite every element of `t` with `value` (no graph; works on parameters).
pub fn constant_(t: &Tensor, value: f32) -> Result<()> {
    let _g = no_grad();
    Ok(t.fill_assign(value)?)
}

/// Overwrite `t` with `src` (same shape after broadcasting; no graph). For re-initialising a parameter in
/// place with one of the samplers above: `copy_(&conv.weight, &init::xavier_uniform(shape, 1.0, rng)?)`.
pub fn copy_(t: &Tensor, src: &Tensor) -> Result<()> {
    let _g = no_grad();
    Ok(t.copy_from(src)?)
}

/// The box-regression output bias prior of docs/MODEL.md §6.5: every element of `head.box.L.out.bias` is 2.0.
pub const BOX_BIAS_PRIOR: f32 = 2.0;

/// The class-output bias prior of docs/MODEL.md §6.5: `ln(5 / nc / (640 / stride)²)`, about 5 objects per
/// 640×640 image spread over the level's cells. `nc = 80`, stride 8 gives −11.5366.
pub fn cls_bias_prior(num_classes: usize, stride: u32) -> f32 {
    let cells = 640.0 / stride as f64;
    (5.0 / num_classes as f64 / (cells * cells)).ln() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_and_priors() {
        assert_eq!(fans(&Shape::new(&[16, 3, 3, 3])), (27, 144));
        let mut rng = Rng::seed(1);
        let w = kaiming_uniform([64, 8, 3, 3], 5f64.sqrt(), FanMode::FanIn, &mut rng).unwrap();
        let b = default_conv_bound(72) as f32;
        let v = w.to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| x.abs() <= b * (1.0 + 1e-6)));
        assert!(v.iter().any(|x| x.abs() > 0.9 * b));
        // Same draws as the bound computed directly.
        let mut r2 = Rng::seed(1);
        assert_eq!(uniform([64, 8, 3, 3], 1.0 / 72f64.sqrt(), &mut r2).to_vec::<f32>().unwrap(), v);
        for (nc, s, want) in [(80, 8, -11.5366), (80, 16, -10.1503), (80, 32, -8.7641), (4, 8, -8.5409)] {
            assert!((cls_bias_prior(nc, s) - want).abs() < 1e-4, "{nc} {s}");
        }
        let n = kaiming_normal([2000, 50], 0.0, FanMode::FanIn, &mut rng).unwrap().to_vec::<f32>().unwrap();
        let var = n.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n.len() as f64;
        assert!((var - 2.0 / 50.0).abs() < 2e-3, "{var}");
        let x = xavier_uniform([30, 20], 1.0, &mut rng).unwrap().to_vec::<f32>().unwrap();
        assert!(x.iter().all(|v| v.abs() <= (6.0f32 / 50.0).sqrt()));
        let xn = xavier_normal([400, 600], 1.0, &mut rng).unwrap().to_vec::<f32>().unwrap();
        let var = xn.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / xn.len() as f64;
        assert!((var - 2.0 / 1000.0).abs() < 1e-4, "{var}");
    }
}
