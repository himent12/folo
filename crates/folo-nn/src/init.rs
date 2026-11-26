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

/// Seeded Orthogonal initialization (Saxe et al. 2014; Mezzadri 2007; PyTorch `nn.init.orthogonal_`).
///
/// Reshapes a tensor of rank `>= 2` as `[rows, cols]` where `rows = dims[0]` and `cols = dims[1..].product()`
/// (so 2D `[out, in]` and 4D `[out, in, kH, kW]` weights are both handled directly). Draws a standard normal
/// matrix of shape `[max(rows, cols), min(rows, cols)]` in row-major order from `rng`, computes its thin
/// Householder QR factorization `A = Q R` in `f64`, applies Mezzadri's positive-diagonal sign correction
/// (`Q[:, k] *= sign(R[k, k])` with `sign(0) = +1`) so `Q` is Haar-uniform on the orthogonal / Stiefel
/// manifold, scales by `gain`, and transposes back when `rows < cols`.
///
/// Satisfies:
/// - `W^T W = gain² · I_cols` when `rows >= cols` (tall or square),
/// - `W W^T = gain² · I_rows` when `rows < cols` (wide).
///
/// Errors if `shape` has rank `< 2`, has any zero-sized dimension, or `gain` is non-finite.
pub fn orthogonal(shape: impl Into<Shape>, gain: f64, rng: &mut Rng) -> Result<Tensor> {
    let shape = shape.into();
    let dims = shape.dims();
    if dims.len() < 2 {
        return Err(Error::Invalid(format!(
            "init::orthogonal: expected rank >= 2 tensor, got shape {shape}"
        )));
    }
    if dims.contains(&0) {
        return Err(Error::Invalid(format!("init::orthogonal: zero dimension in shape {shape}")));
    }
    if !(gain.is_finite() && (gain as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "init::orthogonal: gain must be finite in f32 range, got {gain}"
        )));
    }
    let rows = dims[0];
    let cols = dims[1..].iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| {
        Error::Invalid(format!("init::orthogonal: flattened size overflow for shape {shape}"))
    })?;
    let total = rows.checked_mul(cols).ok_or_else(|| {
        Error::Invalid(format!("init::orthogonal: element count overflow for shape {shape}"))
    })?;
    let transposed = rows < cols;
    let (m, n) = if transposed { (cols, rows) } else { (rows, cols) };

    // Draw standard normal entries in canonical row-major order from `rng`.
    let raw = Tensor::randn([m, n], 0.0, 1.0, rng).to_vec::<f32>()?;
    let mut a: Vec<f64> = raw.into_iter().map(f64::from).collect();
    let q = householder_qr_thin_q(&mut a, m, n, gain);

    let out: Vec<f32> = if transposed {
        let mut t = vec![0.0f32; total];
        for i in 0..m {
            for j in 0..n {
                t[j * m + i] = q[i * n + j] as f32;
            }
        }
        t
    } else {
        q.into_iter().map(|v| v as f32).collect()
    };
    if !out.iter().all(|v| v.is_finite()) {
        return Err(Error::Invalid(format!(
            "init::orthogonal: gain {gain} overflowed f32 output for shape {shape}"
        )));
    }
    Ok(Tensor::from_vec(out, shape)?)
}

/// Thin Householder QR orthogonalization of row-major `a` (`m x n`, `m >= n`) in `f64`, returning
/// `gain * Q * diag(sign(diag(R)))` (`m x n`) with `sign(0) = +1`.
fn householder_qr_thin_q(a: &mut [f64], m: usize, n: usize, gain: f64) -> Vec<f64> {
    debug_assert!(m >= n && a.len() == m * n);
    let mut betas = vec![0.0f64; n];
    let mut r_diag = vec![0.0f64; n];

    // Factor A = H_0 H_1 ... H_{n-1} R, storing v_k[1..] in the lower triangle A[k+1..m, k].
    for k in 0..n {
        let mut scale = 0.0f64;
        for i in k..m {
            scale = scale.max(a[i * n + k].abs());
        }
        if scale == 0.0 {
            betas[k] = 0.0;
            r_diag[k] = 0.0;
            continue;
        }
        let mut sum_sq = 0.0f64;
        for i in k..m {
            let s = a[i * n + k] / scale;
            sum_sq += s * s;
        }
        let norm = scale * sum_sq.sqrt();
        let x0 = a[k * n + k];
        // Choose alpha with opposite sign to x0 so v0 = x0 - alpha never cancels.
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        let v0 = x0 - alpha;
        r_diag[k] = alpha;

        let mut v_norm_sq = 1.0f64;
        for i in (k + 1)..m {
            let vi = a[i * n + k] / v0;
            a[i * n + k] = vi;
            v_norm_sq += vi * vi;
        }
        let beta = 2.0 / v_norm_sq;
        betas[k] = beta;

        // Apply H_k = I - beta * v * v^T from the left to trailing columns j = k+1..n.
        for j in (k + 1)..n {
            let mut dot = a[k * n + j];
            for i in (k + 1)..m {
                dot += a[i * n + k] * a[i * n + j];
            }
            let tau = beta * dot;
            a[k * n + j] -= tau;
            for i in (k + 1)..m {
                a[i * n + j] -= tau * a[i * n + k];
            }
        }
    }

    // Backward accumulation of thin Q = H_0 H_1 ... H_{n-1} [I_n; 0].
    let mut q = vec![0.0f64; m * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    for k in (0..n).rev() {
        let beta = betas[k];
        if beta == 0.0 {
            continue;
        }
        for j in k..n {
            let mut dot = q[k * n + j];
            for i in (k + 1)..m {
                dot += a[i * n + k] * q[i * n + j];
            }
            let tau = beta * dot;
            q[k * n + j] -= tau;
            for i in (k + 1)..m {
                q[i * n + j] -= tau * a[i * n + k];
            }
        }
    }

    // Mezzadri (2007) positive-diagonal sign correction and gain scaling.
    for k in 0..n {
        let s = if r_diag[k] < 0.0 { -gain } else { gain };
        for i in 0..m {
            q[i * n + k] *= s;
        }
    }
    q
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

    /// Check `W^T W == gain² I_cols` (if `rows >= cols`) or `W W^T == gain² I_rows` (if `rows < cols`).
    fn assert_orthogonal(w: &[f32], rows: usize, cols: usize, gain: f64, tol: f64) {
        let g2 = gain * gain;
        if rows >= cols {
            for i in 0..cols {
                for j in 0..cols {
                    let mut dot = 0.0f64;
                    for r in 0..rows {
                        dot += w[r * cols + i] as f64 * w[r * cols + j] as f64;
                    }
                    let want = if i == j { g2 } else { 0.0 };
                    assert!(
                        (dot - want).abs() <= tol,
                        "W^T W[{i},{j}] = {dot} vs {want} (shape {rows}x{cols}, gain={gain})"
                    );
                }
            }
        } else {
            for i in 0..rows {
                for j in 0..rows {
                    let mut dot = 0.0f64;
                    for c in 0..cols {
                        dot += w[i * cols + c] as f64 * w[j * cols + c] as f64;
                    }
                    let want = if i == j { g2 } else { 0.0 };
                    assert!(
                        (dot - want).abs() <= tol,
                        "W W^T[{i},{j}] = {dot} vs {want} (shape {rows}x{cols}, gain={gain})"
                    );
                }
            }
        }
    }

    #[test]
    fn orthogonal_2d_4d_orthogonality_gain_and_determinism() {
        let mut rng = Rng::seed(2026);
        let gain_relu = leaky_relu_gain(0.0); // sqrt(2)

        // Square, tall, wide 2D, and tall/wide 4D Conv2d shapes, plus 1x1 edge cases.
        let cases: &[(&[usize], f64)] = &[
            (&[16, 16], 1.0),
            (&[16, 16], gain_relu),
            (&[32, 12], 1.0),
            (&[12, 32], 1.5),
            (&[32, 3, 3, 3], gain_relu), // 32 x 27 (tall)
            (&[16, 8, 3, 3], 0.75),      // 16 x 72 (wide)
            (&[1, 1], 2.5),
            (&[1, 1, 1, 1], 1.0),
            (&[8, 8], 0.0), // zero gain -> exact zero tensor
        ];
        for &(dims, gain) in cases {
            let w = orthogonal(dims, gain, &mut rng).unwrap();
            assert_eq!(w.dims(), dims);
            let data = w.to_vec::<f32>().unwrap();
            assert!(data.iter().all(|v| v.is_finite()));
            let rows = dims[0];
            let cols: usize = dims[1..].iter().product();
            assert_orthogonal(&data, rows, cols, gain, 1e-5);
            if rows == cols {
                // Square matrices are orthogonal on both sides: W^T W == W W^T == gain² I.
                assert_orthogonal(&data, cols, rows, gain, 1e-5);
            }
        }

        // Exact seed determinism and seed sensitivity.
        let a1 = orthogonal([12, 4, 3, 3], gain_relu, &mut Rng::seed(77)).unwrap().to_vec::<f32>().unwrap();
        let a2 = orthogonal([12, 4, 3, 3], gain_relu, &mut Rng::seed(77)).unwrap().to_vec::<f32>().unwrap();
        let b1 = orthogonal([12, 4, 3, 3], gain_relu, &mut Rng::seed(78)).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(
            a1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            a2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "same seed must be bitwise deterministic"
        );
        assert_ne!(a1, b1, "different seeds must produce distinct orthogonal matrices");
    }

    #[test]
    fn householder_qr_positive_diagonal_and_rank_deficient_orthogonality() {
        // 1. Verify A = Q * R with R upper-triangular and diag(R) >= 0 (Mezzadri 2007).
        let (m, n) = (10usize, 6usize);
        let mut rng = Rng::seed(123);
        let orig: Vec<f64> = Tensor::randn([m, n], 0.0, 1.0, &mut rng)
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        let mut work = orig.clone();
        let q = householder_qr_thin_q(&mut work, m, n, 1.0);

        // Compute R = Q^T A (n x n).
        let mut r = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for row in 0..m {
                    s += q[row * n + i] * orig[row * n + j];
                }
                r[i * n + j] = s;
            }
        }
        for i in 0..n {
            assert!(
                r[i * n + i] >= 0.0,
                "Mezzadri sign correction requires R[{i},{i}] >= 0, got {}",
                r[i * n + i]
            );
            for j in 0..i {
                assert!(
                    r[i * n + j].abs() < 1e-12,
                    "R[{i},{j}] = {} must be zero below diagonal",
                    r[i * n + j]
                );
            }
        }
        // Reconstruct Q * R == orig.
        for row in 0..m {
            for col in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += q[row * n + k] * r[k * n + col];
                }
                assert!((s - orig[row * n + col]).abs() < 1e-12, "Q*R mismatch at ({row},{col})");
            }
        }

        // 2. Rank-deficient (zero matrix AND linearly dependent non-zero columns) still produces exact Q^T Q == I_n.
        let mut zeros = vec![0.0f64; m * n];
        let q_zero = householder_qr_thin_q(&mut zeros, m, n, 1.0);
        let mut rank1 = vec![0.0f64; m * n];
        for row in 0..m {
            let base = (row + 1) as f64;
            for col in 0..n {
                rank1[row * n + col] = base * (col + 1) as f64;
            }
        }
        let q_rank1 = householder_qr_thin_q(&mut rank1, m, n, 1.0);
        for (label, q_mat) in [("zero", &q_zero), ("rank1", &q_rank1)] {
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for row in 0..m {
                        dot += q_mat[row * n + i] * q_mat[row * n + j];
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-13, "{label}-matrix Q^T Q[{i},{j}] = {dot} vs {want}");
                }
            }
        }

        // 3. Invalid shapes, overflowing products, and non-finite / f32-overflowing gains are rejected.
        assert!(orthogonal([], 1.0, &mut rng).is_err());
        assert!(orthogonal([8], 1.0, &mut rng).is_err());
        assert!(orthogonal([4, 0], 1.0, &mut rng).is_err());
        assert!(orthogonal([0, 4, 3, 3], 1.0, &mut rng).is_err());
        assert!(orthogonal([2, usize::MAX / 2, 2], 1.0, &mut rng).is_err());
        assert!(orthogonal([4, 4], f64::NAN, &mut rng).is_err());
        assert!(orthogonal([4, 4], f64::INFINITY, &mut rng).is_err());
        assert!(orthogonal([4, 4], f64::NEG_INFINITY, &mut rng).is_err());
        assert!(orthogonal([1, 1], 1e40, &mut rng).is_err());
    }
}
