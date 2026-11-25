//! Newton–Schulz / Polar-FOLO matrix orthogonalisation and geometric aspect scaling for [`super::MuSgd`].

use folo_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use super::cpu_fast;
use super::simd::{axpby_slice, ortho_normalize_batch, sum_sq_f32_slice};
use crate::{Error, Result};

/// Geometric spectral conditioning mode of [`super::MuSgd`] (`Conv-GeoMuon`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MuonGeoMode {
    /// Legacy `muon_scale` (`dims[n-2]/dims[n-1]`) and dense orthogonalisation on all rank-2/4 decay weights.
    #[default]
    Legacy,
    /// True 2-D flattened aspect scale `sqrt(max(1, m/n))` + exact zero-GEMM per-channel Polar LMO for depthwise convs.
    Geo,
}

impl MuonGeoMode {
    /// True when this is legacy mode.
    pub fn is_legacy(&self) -> bool {
        *self == MuonGeoMode::Legacy
    }

    /// Canonical lowercase name (`"legacy"`, `"geo"`).
    pub fn as_str(self) -> &'static str {
        match self {
            MuonGeoMode::Legacy => "legacy",
            MuonGeoMode::Geo => "geo",
        }
    }
}

/// Newton–Schulz iterations of [`orthogonalize`].
pub const NS_STEPS: usize = 5;
/// The quintic Newton–Schulz coefficients `(a, b, c)`: `X ← a·X + (b·A + c·A²)·X`, `A = X·Xᵀ`.
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);
/// Fixed 5-step Newton–Schulz schedule (`NS_COEFFS` repeated 5 times).
pub const NS5_SCHEDULE: [(f32, f32, f32); NS_STEPS] = [NS_COEFFS; NS_STEPS];
/// Polar-FOLO 4-step minimax Remez odd-quintic schedule (`l₀ = 5e-3`, safety cushion `1.005`, paired with step-0
/// Schatten-4 spectral rescaling): 12 GEMMs (20% fewer than `NS5`) with ~3× lower polar Frobenius distance and
/// tighter spectral accuracy across `κ = 10..10⁴`.
pub const POLAR4_COEFFS: [(f32, f32, f32); 4] = [
    (4.216_272_4, -12.289_586, 9.009_986),
    (4.152_54, -11.770_327, 8.558_353),
    (3.887_979, -9.771_957, 6.839_154_7),
    (3.287_126, -5.261_409, 3.046_717_4),
];
/// Polar-FOLO 5-step minimax Remez odd-quintic schedule (`l₀ = 1e-3`, safety cushion `1.005`, paired with step-0
/// Schatten-4 spectral rescaling): 15 GEMMs with ultra-sharp orthogonalisation (`≤ 0.12` spectral error down to
/// `κ = 100`, 3.3× lower polar error at `κ = 1000`).
pub const POLAR5_COEFFS: [(f32, f32, f32); 5] = [
    (4.232_037, -12.420_36, 9.124_008),
    (4.219_302, -12.314_649, 9.031_83),
    (4.165_359_5, -11.873_566, 8.648_002),
    (3.940_082_3, -10.145_948, 7.158_359_5),
    (3.537_624_1, -6.125_873_6, 3.688_620_8),
];
/// Completed-Square degree-9 coefficients `(gamma, delta, eta, xi, c4)` for one `Polar3-CS9` step.
pub type Cs9StepCoeffs = (f32, f32, f32, f32, f32);

/// Polar3-CS9 3-step Completed-Square Paterson–Stockmeyer degree-9 odd polynomial schedule
/// (`l₀ = 4e-3`, safety cushion `1.005`, paired with step-0 Schatten-8 spectral rescaling):
/// `(gamma, delta, eta, xi, c4)` with `S = delta·I + gamma·A + A²`, `q4(A) = xi·I + eta·A + c4·S²`, `X' = q4(A)·X`.
/// Uses 6 wide `r × c` GEMMs (vs 8 in `Polar4`) for total degree `729 > 625`.
pub const POLAR3_CS9_COEFFS: [Cs9StepCoeffs; 3] = [
    (-1.149_836, 0.234_306_05, -0.042_710_006, 0.069_763_21, 135.661_48),
    (-1.158_462, 0.242_525_31, -0.298_174_05, 0.491_120_82, 113.502_03),
    (-1.261_140_8, 0.343_998, -1.070_805_2, 1.935_225_4, 29.709_389),
];
/// Added to the Frobenius norm before [`orthogonalize`] divides by it.
pub const NS_EPS: f32 = 1e-7;
/// A bucket of [`orthogonalize_many`] keeps its row counts within this factor (bounds the zero padding).
const BUCKET_SPAN: usize = 16;

/// Orthogonalisation polynomial schedule used by [`super::MuSgd`]'s Muon step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MuonOrtho {
    /// Legacy fixed 5-step quintic Newton–Schulz `(3.4445, -4.7750, 2.0315)` (15 GEMMs).
    Ns5,
    /// Default Polar-FOLO 4-step minimax Remez odd-quintic schedule with step-0 Schatten-4 spectral rescaling
    /// (12 GEMMs, 20% fewer GEMMs than `Ns5` with ~3× lower polar Frobenius error).
    #[default]
    Polar4,
    /// Polar-FOLO 5-step minimax Remez odd-quintic schedule with step-0 Schatten-4 spectral rescaling (15 GEMMs).
    Polar5,
    /// Polar3-CS9 3-step Completed-Square Paterson–Stockmeyer degree-9 odd polynomial with step-0 Schatten-8
    /// spectral rescaling (6 wide `r × c` GEMMs, total degree 729).
    #[serde(rename = "polar3_cs9")]
    Polar3Cs9,
}

impl MuonOrtho {
    /// True when this is the default [`MuonOrtho::Polar4`] schedule.
    pub fn is_default(&self) -> bool {
        *self == MuonOrtho::Polar4
    }

    /// Per-step odd-quintic coefficients `(a_k, b_k, c_k)` (empty for `Polar3Cs9`; use [`Self::cs9_schedule`]).
    pub fn schedule(self) -> &'static [(f32, f32, f32)] {
        match self {
            MuonOrtho::Ns5 => &NS5_SCHEDULE,
            MuonOrtho::Polar4 => &POLAR4_COEFFS,
            MuonOrtho::Polar5 => &POLAR5_COEFFS,
            MuonOrtho::Polar3Cs9 => &[],
        }
    }

    /// Per-step Completed-Square degree-9 coefficients `(gamma, delta, eta, xi, c4)` when `Polar3Cs9`.
    pub fn cs9_schedule(self) -> Option<&'static [Cs9StepCoeffs]> {
        match self {
            MuonOrtho::Polar3Cs9 => Some(&POLAR3_CS9_COEFFS),
            _ => None,
        }
    }

    /// True when this schedule uses the Completed-Square degree-9 polynomial (`Polar3Cs9`).
    pub fn is_cs9(self) -> bool {
        self == MuonOrtho::Polar3Cs9
    }

    /// Whether step 0 applies zero-extra-GEMM Schatten-4 (or Schatten-8 for `Polar3Cs9`) spectral rescaling.
    pub fn schatten4(self) -> bool {
        !matches!(self, MuonOrtho::Ns5)
    }

    /// Canonical lowercase name (`"ns5"`, `"polar4"`, `"polar5"`, `"polar3_cs9"`).
    pub fn as_str(self) -> &'static str {
        match self {
            MuonOrtho::Ns5 => "ns5",
            MuonOrtho::Polar4 => "polar4",
            MuonOrtho::Polar5 => "polar5",
            MuonOrtho::Polar3Cs9 => "polar3_cs9",
        }
    }
}

/// The scale [`super::MuSgd`] gives a parameter's orthogonalised update: `sqrt(max(1, d₋₂ / d₋₁))` of its last two
/// dimensions.
pub fn muon_scale(dims: &[usize]) -> f32 {
    let n = dims.len();
    if n < 2 {
        return 1.0;
    }
    (dims[n - 2] as f32 / dims[n - 1] as f32).max(1.0).sqrt()
}

/// True 2-D flattened aspect scale for `Conv-GeoMuon`: `sqrt(max(1, m / n))` where `m = dims[0]` and
/// `n = prod(dims[1..])`.
pub fn muon_scale_geo(dims: &[usize]) -> f32 {
    if dims.len() < 2 {
        return 1.0;
    }
    let rows = dims[0];
    let cols = dims.iter().product::<usize>() / rows.max(1);
    if cols == 0 {
        return 1.0;
    }
    (rows as f32 / cols as f32).max(1.0).sqrt()
}

/// Newton–Schulz orthogonalisation of one matrix `g` (`[rows, cols]`): `X = g / (‖g‖_F + 1e-7)`, transposed so
/// that `rows ≤ cols`, then [`NS_STEPS`] times `A = X·Xᵀ`, `X = a·X + (b·A + c·A·A)·X` with [`NS_COEFFS`], and
/// transposed back. In `f32` (the reference computes in `bfloat16`).
pub fn orthogonalize(g: &Tensor) -> Result<Tensor> {
    orthogonalize_with(g, MuonOrtho::Ns5)
}

/// [`orthogonalize`] with a chosen [`MuonOrtho`] schedule (e.g. [`MuonOrtho::Polar4`] or [`MuonOrtho::Polar5`]).
pub fn orthogonalize_with(g: &Tensor, ortho: MuonOrtho) -> Result<Tensor> {
    Ok(orthogonalize_many_with(std::slice::from_ref(g), ortho)?.remove(0))
}

/// [`super::MuSgd`]'s update directions for `us` (any shapes of rank ≥ 2) using the legacy [`MuonOrtho::Ns5`]
/// schedule.
pub fn orthogonalize_many(us: &[Tensor]) -> Result<Vec<Tensor>> {
    orthogonalize_many_with(us, MuonOrtho::Ns5)
}

/// [`super::MuSgd`]'s update directions for `us` (any shapes of rank ≥ 2): each flattened to `[d₀, rest]`,
/// orthogonalised with `ortho`, scaled by [`muon_scale`] and reshaped back. Matrices are oriented so that
/// rows ≤ cols and batched by column count, zero-padding the rows (a zero row changes neither the Frobenius norm,
/// nor the Schatten-4 norm `‖X·Xᵀ‖_F^(1/2)`, nor the other rows' result).
pub fn orthogonalize_many_with(us: &[Tensor], ortho: MuonOrtho) -> Result<Vec<Tensor>> {
    orthogonalize_many_geo(us, ortho, MuonGeoMode::Legacy)
}

/// [`orthogonalize_many_with`] extended with [`MuonGeoMode`] (true 2-D aspect scaling).
pub(super) fn orthogonalize_many_geo(
    us: &[Tensor],
    ortho: MuonOrtho,
    geo: MuonGeoMode,
) -> Result<Vec<Tensor>> {
    let _g = folo_core::no_grad();
    let mut mats = Vec::with_capacity(us.len());
    for (i, u) in us.iter().enumerate() {
        let d = u.dims();
        if d.len() < 2 {
            return Err(Error::Invalid(format!("musgd: orthogonalize needs rank >= 2, got {d:?}")));
        }
        let rows = d[0];
        let m = u.reshape([rows, u.numel() / rows.max(1)])?;
        let t = m.dims()[0] > m.dims()[1];
        let m = if t { m.transpose(0, 1)?.contiguous()? } else { m.contiguous()? };
        mats.push((i, m, t));
    }
    let mut by_cols: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut order: Vec<usize> = (0..mats.len()).collect();
    order.sort_by_key(|&k| (mats[k].1.dims()[1], std::cmp::Reverse(mats[k].1.dims()[0]), k));
    for k in order {
        let (r, c) = (mats[k].1.dims()[0], mats[k].1.dims()[1]);
        match by_cols.last_mut() {
            Some((cc, ks)) if *cc == c && mats[ks[0]].1.dims()[0] <= BUCKET_SPAN * r => ks.push(k),
            _ => by_cols.push((c, vec![k])),
        }
    }
    let mut out: Vec<Option<Tensor>> = vec![None; us.len()];
    let schedule = ortho.schedule();
    let cs9_schedule = ortho.cs9_schedule();
    let use_s4 = ortho.schatten4();
    for (_, ks) in by_cols {
        let rmax = mats[ks[0]].1.dims()[0];
        let padded = ks
            .iter()
            .map(|&k| {
                let m = &mats[k].1;
                let r = m.dims()[0];
                if r == rmax { Ok(m.clone()) } else { m.pad(&[(0, rmax - r), (0, 0)], 0.0) }
            })
            .collect::<folo_core::Result<Vec<_>>>()?;
        let mut x = if padded.len() == 1
            && padded[0].numel() > 0
            && padded[0].device().is_cpu()
            && padded[0].dtype() == DType::F32
        {
            // `Tensor::stack` of a single tensor returns an `unsqueeze(0)` view sharing storage with the
            // caller's input; allocate a private copy before mutating `x` in place.
            let c = padded[0].dims()[1];
            let data = padded[0].with_f32_slice(<[f32]>::to_vec)?;
            Tensor::from_vec(data, [1, rmax, c])?
        } else {
            Tensor::stack(&padded, 0)?
        };
        if x.numel() > 0 && cpu_fast(&[&x]) {
            let batch = x.dims()[0];
            let stride = x.numel() / batch;
            let (norms, eps_ss) =
                x.update_f32(|xs| ortho_normalize_batch(xs, batch, stride, use_s4, NS_EPS))?;
            if let Some(cs9) = cs9_schedule {
                for (step, &(gamma, delta, eta, xi, c4)) in cs9.iter().enumerate() {
                    let xt = x.transpose(1, 2)?.contiguous()?;
                    let am = x.matmul(&xt)?;
                    let aa = am.matmul(&am)?;
                    let am_stride = am.numel() / batch;
                    if step == 0 {
                        let mut inv_as = Vec::with_capacity(batch);
                        am.with_f32_slice(|am_s| {
                            aa.update_f32(|aa_s| {
                                for bi in 0..batch {
                                    let am_b = &am_s[bi * am_stride..(bi + 1) * am_stride];
                                    let aa_b = &mut aa_s[bi * am_stride..(bi + 1) * am_stride];
                                    // Schatten-8 norm: ||A_2||_F^(1/4) = (sum(A_2^2))^(1/8).
                                    let alpha8 = sum_sq_f32_slice(aa_b, 1.0).sqrt().sqrt().sqrt()
                                        + eps_ss[bi] / norms[bi];
                                    let inv_a = if alpha8 == 0.0 { 0.0 } else { 1.0 / alpha8 };
                                    let ia2 = inv_a * inv_a;
                                    let ia4 = ia2 * ia2;
                                    axpby_slice(aa_b, am_b, ia4, gamma * ia2);
                                    for r in 0..rmax {
                                        aa_b[r * rmax + r] += delta;
                                    }
                                    inv_as.push(inv_a);
                                }
                            })
                        })??;
                        let s2 = aa.matmul(&aa)?;
                        am.with_f32_slice(|am_s| {
                            s2.update_f32(|s2_s| {
                                for (bi, &inv_a) in inv_as.iter().enumerate() {
                                    let am_b = &am_s[bi * am_stride..(bi + 1) * am_stride];
                                    let s2_b = &mut s2_s[bi * am_stride..(bi + 1) * am_stride];
                                    axpby_slice(s2_b, am_b, c4, eta * inv_a * inv_a);
                                }
                            })
                        })??;
                        let bx = s2.matmul(&x)?;
                        bx.with_f32_slice(|bx_s| {
                            x.update_f32(|x_s| {
                                for (bi, &inv_a) in inv_as.iter().enumerate() {
                                    let x_b = &mut x_s[bi * stride..(bi + 1) * stride];
                                    let bx_b = &bx_s[bi * stride..(bi + 1) * stride];
                                    axpby_slice(x_b, bx_b, xi * inv_a, inv_a);
                                }
                            })
                        })??;
                    } else {
                        am.with_f32_slice(|am_s| {
                            aa.update_f32(|aa_s| {
                                axpby_slice(aa_s, am_s, 1.0, gamma);
                                for bi in 0..batch {
                                    let aa_b = &mut aa_s[bi * am_stride..(bi + 1) * am_stride];
                                    for r in 0..rmax {
                                        aa_b[r * rmax + r] += delta;
                                    }
                                }
                            })
                        })??;
                        let s2 = aa.matmul(&aa)?;
                        am.with_f32_slice(|am_s| s2.update_f32(|s2_s| axpby_slice(s2_s, am_s, c4, eta)))??;
                        let bx = s2.matmul(&x)?;
                        bx.with_f32_slice(|bx_s| x.update_f32(|x_s| axpby_slice(x_s, bx_s, xi, 1.0)))??;
                    }
                }
            } else {
                for (step, &(a, b, c)) in schedule.iter().enumerate() {
                    let xt = x.transpose(1, 2)?.contiguous()?;
                    let am = x.matmul(&xt)?;
                    let aa = am.matmul(&am)?;
                    if step == 0 && use_s4 {
                        let am_stride = am.numel() / batch;
                        let mut inv_as = Vec::with_capacity(batch);
                        am.with_f32_slice(|am_s| {
                            aa.update_f32(|aa_s| {
                                for bi in 0..batch {
                                    let am_b = &am_s[bi * am_stride..(bi + 1) * am_stride];
                                    let aa_b = &mut aa_s[bi * am_stride..(bi + 1) * am_stride];
                                    let a_frob = sum_sq_f32_slice(am_b, 1.0).sqrt();
                                    let alpha = a_frob.sqrt() + eps_ss[bi] / norms[bi];
                                    let inv_a = if alpha == 0.0 { 0.0 } else { 1.0 / alpha };
                                    let inv_a2 = inv_a * inv_a;
                                    let inv_a4 = inv_a2 * inv_a2;
                                    axpby_slice(aa_b, am_b, c * inv_a4, b * inv_a2);
                                    inv_as.push(inv_a);
                                }
                            })
                        })??;
                        let bx = aa.matmul(&x)?;
                        bx.with_f32_slice(|bx_s| {
                            x.update_f32(|x_s| {
                                for (bi, &inv_a) in inv_as.iter().enumerate() {
                                    let x_b = &mut x_s[bi * stride..(bi + 1) * stride];
                                    let bx_b = &bx_s[bi * stride..(bi + 1) * stride];
                                    axpby_slice(x_b, bx_b, a * inv_a, inv_a);
                                }
                            })
                        })??;
                    } else {
                        am.with_f32_slice(|am_s| aa.update_f32(|aa_s| axpby_slice(aa_s, am_s, c, b)))??;
                        let bx = aa.matmul(&x)?;
                        bx.with_f32_slice(|bx_s| x.update_f32(|x_s| axpby_slice(x_s, bx_s, a, 1.0)))??;
                    }
                }
            }
        } else if x.numel() > 0 {
            let (norm, eps_s) = if use_s4 {
                const BIG_THRESH: f32 = f32::from_bits(0x5780_0000);
                const DOWN_SCALE: f32 = f32::from_bits(0x1780_0000);
                let max_abs = x.abs()?.max([1, 2], true)?;
                let is_big = max_abs.gt_scalar(BIG_THRESH)?;
                let one = max_abs.ones_like()?;
                let scale = is_big.where_cond(&one.mul_scalar(DOWN_SCALE)?, &one)?;
                let xs = x.mul(&scale)?;
                let eps_s = scale.mul_scalar(NS_EPS)?;
                let norm = xs.square()?.sum([1, 2], true)?.sqrt()?.add(&eps_s)?;
                x = xs.div(&norm)?;
                (norm, Some(eps_s))
            } else {
                let norm = x.square()?.sum([1, 2], true)?.sqrt()?.add_scalar(NS_EPS)?;
                x = x.div(&norm)?;
                (norm, None)
            };
            if let Some(cs9) = cs9_schedule {
                let mut eye_v = vec![0.0f32; rmax * rmax];
                for r in 0..rmax {
                    eye_v[r * rmax + r] = 1.0;
                }
                let eye = Tensor::from_vec(eye_v, [1, rmax, rmax])?.to_device(x.device())?;
                for (step, &(gamma, delta, eta, xi, c4)) in cs9.iter().enumerate() {
                    let xt = x.transpose(1, 2)?.contiguous()?;
                    let am = x.matmul(&xt)?;
                    let a2 = am.matmul(&am)?;
                    if step == 0 {
                        let a2_frob = a2.square()?.sum([1, 2], true)?.sqrt()?;
                        let eps_s = eps_s.as_ref().expect("set when cs9");
                        let alpha = a2_frob.sqrt()?.sqrt()?.add(&eps_s.div(&norm)?)?;
                        let zeros = Tensor::zeros_with(alpha.shape(), DType::F32, alpha.device())?;
                        let inv_a = alpha.eq_scalar(0.0)?.where_cond(&zeros, &alpha.recip()?)?;
                        let ia2 = inv_a.square()?;
                        let ia4 = ia2.square()?;
                        let s0 = eye
                            .mul_scalar(delta)?
                            .add(&am.mul(&ia2.mul_scalar(gamma)?)?)?
                            .add(&a2.mul(&ia4)?)?;
                        let s2 = s0.matmul(&s0)?;
                        let bm = am.mul(&ia2.mul_scalar(eta)?)?.add(&s2.mul_scalar(c4)?)?;
                        x = x.mul(&inv_a.mul_scalar(xi)?)?.add(&bm.matmul(&x)?.mul(&inv_a)?)?;
                    } else {
                        let sm = eye.mul_scalar(delta)?.add(&am.mul_scalar(gamma)?)?.add(&a2)?;
                        let s2 = sm.matmul(&sm)?;
                        let bm = am.mul_scalar(eta)?.add(&s2.mul_scalar(c4)?)?;
                        x = x.mul_scalar(xi)?.add(&bm.matmul(&x)?)?;
                    }
                }
            } else {
                for (step, &(a, b, c)) in schedule.iter().enumerate() {
                    let xt = x.transpose(1, 2)?.contiguous()?;
                    let am = x.matmul(&xt)?;
                    if step == 0 && use_s4 {
                        let a_frob = am.square()?.sum([1, 2], true)?.sqrt()?;
                        let eps_s = eps_s.as_ref().expect("set when use_s4");
                        let alpha = a_frob.sqrt()?.add(&eps_s.div(&norm)?)?;
                        let zeros = Tensor::zeros_with(alpha.shape(), DType::F32, alpha.device())?;
                        let inv_a = alpha.eq_scalar(0.0)?.where_cond(&zeros, &alpha.recip()?)?;
                        let inv_a2 = inv_a.square()?;
                        let inv_a4 = inv_a2.square()?;
                        let bm = am
                            .mul(&inv_a2.mul_scalar(b)?)?
                            .add(&am.matmul(&am)?.mul(&inv_a4.mul_scalar(c)?)?)?;
                        x = x.mul(&inv_a.mul_scalar(a)?)?.add(&bm.matmul(&x)?.mul(&inv_a)?)?;
                    } else {
                        let bm = am.mul_scalar(b)?.add(&am.matmul(&am)?.mul_scalar(c)?)?;
                        x = x.mul_scalar(a)?.add(&bm.matmul(&x)?)?;
                    }
                }
            }
        }
        for (j, &k) in ks.iter().enumerate() {
            let (i, m, t) = &mats[k];
            let r = m.dims()[0];
            let c = m.dims()[1];
            let o = x.narrow(0, j, 1)?.narrow(1, 0, r)?.reshape([r, c])?;
            let o = if *t { o.transpose(0, 1)? } else { o };
            let u = &us[*i];
            let s = if geo == MuonGeoMode::Legacy { muon_scale(u.dims()) } else { muon_scale_geo(u.dims()) };
            let o = o.contiguous()?.reshape(u.shape().clone())?;
            out[*i] = Some(if s != 1.0 { o.mul_scalar(s)? } else { o });
        }
    }
    Ok(out.into_iter().map(|o| o.expect("every matrix is in a bucket")).collect())
}
