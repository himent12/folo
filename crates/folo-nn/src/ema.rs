//! Exponential moving average of model weights (docs/MODEL.md §8.5).

use std::collections::{HashMap, HashSet};

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor, no_grad};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::module::{Module, TensorKind};
use crate::optim::storage_key;
use crate::{Error, Result};

/// Minimum total elements across tracked CPU tensors before parallelizing across tensors with rayon.
const EMA_PAR_THRESHOLD: usize = 32_768;
/// Minimum elements in a single contiguous CPU tensor before parallelizing chunks within the tensor.
const EMA_CHUNK_PAR_THRESHOLD: usize = 65_536;
/// Chunk size (in `f32` elements) when parallelizing within a single large tensor.
const EMA_CHUNK_SIZE: usize = 16_384;

/// docs/MODEL.md §8.5: `decay = 0.9999`.
pub const EMA_DECAY: f64 = 0.9999;
/// docs/MODEL.md §8.5: `tau = 2000`.
pub const EMA_TAU: f64 = 2000.0;
/// docs/MODEL.md §8.5: the nominal window `1 / (1 − decay)` of the default EMA, in updates: the cap of
/// [`ema_for_run`].
pub const EMA_MAX_WINDOW: f64 = 10_000.0;
/// `PC-PEMA` base power-law exponent `γ₀ = 6.94` (relative width `σ_rel = [(γ₀ + 1)(γ₀ + 3)]^(-1/2) ≈ 0.113`).
pub const PEMA_GAMMA_0: f64 = 6.94;
/// `PC-PEMA` LR-phase coupling exponent `ρ = 0.25`.
pub const PEMA_RHO: f64 = 0.25;
/// `PC-PEMA` minimum clamped LR factor `lf_min = 0.01`.
pub const PEMA_LF_MIN: f64 = 0.01;

/// EMA decay schedule mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmaMode {
    /// Legacy exponential ramp EMA (`d = decay · (1 − exp(−u / tau))`).
    #[default]
    Legacy,
    /// Prior-Shifted LR-Phase-Coupled Power-Law EMA (`PC-PEMA`).
    Power,
}

/// Configuration for [`EmaMode::Power`] (`PC-PEMA`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmaPowerConfig {
    /// Base power-law exponent `γ₀` (default [`PEMA_GAMMA_0`]).
    #[serde(deserialize_with = "exact_f64")]
    pub gamma_0: f64,
    /// LR-phase coupling exponent `ρ` (default [`PEMA_RHO`]).
    #[serde(deserialize_with = "exact_f64")]
    pub rho: f64,
    /// Lower clamp on `lf_u` (default [`PEMA_LF_MIN`]).
    #[serde(deserialize_with = "exact_f64")]
    pub lf_min: f64,
    /// Prior shift `t₀` (`0.0` from scratch; `warmup_iters` or `round(0.03 * total).max(1)` for transfer).
    #[serde(deserialize_with = "exact_f64")]
    pub t_0: f64,
}

impl EmaPowerConfig {
    /// Default `PC-PEMA` parameters (`γ₀ = 6.94`, `ρ = 0.25`, `lf_min = 0.01`) with prior shift `t_0`.
    pub fn new(t_0: f64) -> Self {
        EmaPowerConfig { gamma_0: PEMA_GAMMA_0, rho: PEMA_RHO, lf_min: PEMA_LF_MIN, t_0 }
    }

    /// Validate all fields are finite and in range.
    pub fn validate(&self) -> Result<()> {
        let ok = self.gamma_0.is_finite()
            && self.gamma_0 >= 0.0
            && self.rho.is_finite()
            && self.rho >= 0.0
            && self.lf_min.is_finite()
            && self.lf_min > 0.0
            && self.lf_min <= 1.0
            && self.t_0.is_finite()
            && self.t_0 >= 0.0;
        if ok { Ok(()) } else { Err(Error::Invalid(format!("ModelEma: invalid PC-PEMA config {self:?}"))) }
    }

    /// Compute `PC-PEMA` decay `β_u` at update `u >= 1` with LR factor `lf`.
    #[inline]
    pub fn decay_at(&self, u: u64, lf: f64) -> f64 {
        let denom = u as f64 + self.t_0;
        if denom <= 1.0 {
            return 0.0;
        }
        let lf_c = lf.clamp(self.lf_min, 1.0);
        let gamma_eff = self.gamma_0 * lf_c.powf(-self.rho);
        ((gamma_eff + 1.0) * (-1.0 / denom).ln_1p()).exp()
    }
}

/// docs/MODEL.md §8.5: `(decay, tau)` of an EMA whose nominal window is `fraction` of a run of `total_updates`
/// optimizer steps, `w = max(fraction · total_updates, 1)`: `decay = 1 − 1 / w`, `tau = max(w / 5, 1)` (the default's
/// ratio, 2000 / 10000). When `w` reaches [`EMA_MAX_WINDOW`] (a run of `EMA_MAX_WINDOW / fraction` steps or more),
/// or `fraction` is not positive, it is exactly the default ([`EMA_DECAY`], [`EMA_TAU`]).
///
/// ```
/// use folo_nn::{EMA_DECAY, EMA_TAU, ema_for_run};
/// assert_eq!(ema_for_run(400, 0.25), (0.99, 20.0));
/// assert_eq!(ema_for_run(40_000, 0.25), (EMA_DECAY, EMA_TAU));
/// assert_eq!(ema_for_run(400, 0.0), (EMA_DECAY, EMA_TAU));
/// ```
pub fn ema_for_run(total_updates: u64, fraction: f64) -> (f64, f64) {
    let w = fraction * total_updates as f64;
    if !(fraction.is_finite() && fraction > 0.0) || w >= EMA_MAX_WINDOW {
        return (EMA_DECAY, EMA_TAU);
    }
    let w = w.max(1.0);
    (1.0 - 1.0 / w, (w / 5.0).max(1.0))
}

/// `folo.ema` (docs/FORMAT.md).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmaContract {
    /// Asymptotic decay.
    #[serde(deserialize_with = "exact_f64")]
    pub decay: f64,
    /// Ramp time constant, in updates.
    #[serde(deserialize_with = "exact_f64")]
    pub tau: f64,
    /// Optional `PC-PEMA` power-law configuration (omitted in legacy mode for byte-identical compat).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<EmaPowerConfig>,
}

/// A JSON number parsed by `f64::from_str` (correctly rounded), not by serde_json's fast path, which can land one ulp
/// off a 17-digit decimal (`0.9090909090909091`, the decay of an 11-step window): the contract a checkpoint
/// resumes with is then the very `f64` it was saved with. A number that overflows to infinity is refused when the
/// file is opened.
fn exact_f64<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<f64, D::Error> {
    let raw = Box::<serde_json::value::RawValue>::deserialize(d)?;
    match raw.get().parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(serde::de::Error::custom(format!("{} is not a finite number", raw.get()))),
    }
}

/// An EMA copy of a model: every parameter and floating buffer (BatchNorm running statistics) of `M`.
///
/// The EMA model is a second instance of the same architecture, kept in eval mode with gradients off. After
/// every optimizer step, [`ModelEma::update`] sets `updates += 1`,
/// `d = decay · (1 − exp(−updates / tau))` and `ema = d · ema + (1 − d) · model` for every tracked tensor.
/// Validation and export use [`ModelEma::module`].
///
/// ```
/// use folo_nn::{BatchNorm2d, ModelEma, Module};
/// let live = BatchNorm2d::new(4);
/// let mut ema = ModelEma::new(BatchNorm2d::new(4), &live).unwrap();
/// ema.update(&live).unwrap();
/// assert_eq!(ema.updates(), 1);
/// assert!((ema.decay_at(1) - 4.99825e-4).abs() < 1e-9);
/// assert!(!ema.module().is_training());
/// ```
pub struct ModelEma<M: Module> {
    module: M,
    /// Cached tensors of `self.module` in canonical order with their [`TensorKind`]. Filtering by
    /// [`is_tracked`] on demand avoids module walks and `String` allocations while honoring in-place
    /// dtype changes on buffers.
    cached: Vec<CachedTensor>,
    /// Precomputed storage keys of `self.cached` for O(N) zero-allocation cross-storage alias checks.
    cached_keys: HashSet<usize>,
    /// True when all entries in `cached` have pairwise distinct underlying storages.
    cached_disjoint: bool,
    /// Reusable dotted-name buffer for zero-allocation traversal of `source`.
    prefix: String,
    /// Reusable scratch list of `(cached_index, source_tensor)` pairs; always cleared by [`ClearGuard`]
    /// before returning so `ModelEma` never retains handles to `source`.
    pair_scratch: Vec<(usize, Tensor)>,
    decay: f64,
    tau: f64,
    power: Option<EmaPowerConfig>,
    updates: u64,
}

struct CachedTensor {
    name: String,
    tensor: Tensor,
    kind: TensorKind,
}

#[inline]
fn is_tracked(kind: TensorKind, tensor: &Tensor) -> bool {
    kind == TensorKind::Parameter || tensor.dtype() == DType::F32
}

impl<M: Module> ModelEma<M> {
    /// Track `source` in `module` (a fresh instance of the same architecture) with docs/MODEL.md's decay
    /// 0.9999 and tau 2000. Moves `module` to `source`'s device, copies `source`'s tensors into it, puts it
    /// in eval mode and turns its gradients off.
    pub fn new(module: M, source: &dyn Module) -> Result<Self> {
        Self::with_params(module, source, EMA_DECAY, EMA_TAU)
    }

    /// [`ModelEma::new`] with an explicit decay and tau.
    pub fn with_params(module: M, source: &dyn Module, decay: f64, tau: f64) -> Result<Self> {
        Self::build_inner(module, source, decay, tau, None)
    }

    /// [`ModelEma::new`] configured for `PC-PEMA` (Prior-Shifted LR-Phase-Coupled Power-Law EMA).
    pub fn with_power(module: M, source: &dyn Module, power: EmaPowerConfig) -> Result<Self> {
        power.validate()?;
        Self::build_inner(module, source, EMA_DECAY, EMA_TAU, Some(power))
    }

    fn build_inner(
        module: M,
        source: &dyn Module,
        decay: f64,
        tau: f64,
        power: Option<EmaPowerConfig>,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&decay) || !(tau.is_finite() && tau > 0.0) {
            return Err(Error::Invalid(format!("ModelEma: invalid decay {decay} / tau {tau}")));
        }
        let mut ema = ModelEma {
            module,
            cached: Vec::new(),
            cached_keys: HashSet::new(),
            cached_disjoint: true,
            prefix: String::new(),
            pair_scratch: Vec::new(),
            decay,
            tau,
            power,
            updates: 0,
        };
        if let Some(dev) = first_device(source) {
            ema.module.to_device(dev)?;
        }
        ema.module.eval();
        ema.refresh_cached();
        for c in &ema.cached {
            c.tensor.set_requires_grad(false);
        }
        ema.copy_from_source(source)?;
        Ok(ema)
    }

    /// Rebuild the cached tensor list from `self.module` and size the scratch buffers.
    fn refresh_cached(&mut self) {
        self.cached = self
            .module
            .named_tensors()
            .into_iter()
            .map(|t| CachedTensor { name: t.name, tensor: t.tensor, kind: t.kind })
            .collect();
        self.cached_keys.clear();
        self.cached_keys.reserve(self.cached.len());
        let mut disjoint = true;
        for c in &self.cached {
            disjoint &= self.cached_keys.insert(storage_key(&c.tensor));
        }
        self.cached_disjoint = disjoint;
        let max_name = self.cached.iter().map(|c| c.name.len()).max().unwrap_or(0);
        self.prefix.clear();
        self.prefix.reserve(max_name);
        self.pair_scratch.clear();
        self.pair_scratch.reserve(self.cached.len());
    }

    /// Restart the average from `source` (`updates = 0`), as at the start of training.
    pub fn reset(&mut self, source: &dyn Module) -> Result<()> {
        self.copy_from_source(source)?;
        self.updates = 0;
        Ok(())
    }

    /// Move the averaged model to `device`, in place (call it when the source model moves).
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        self.module.to_device(device)?;
        self.refresh_cached();
        Ok(())
    }

    /// The EMA model (eval mode).
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Updates so far (`ema.__updates`).
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// `(decay, tau)` and optional `PC-PEMA` settings.
    pub fn contract(&self) -> EmaContract {
        EmaContract { decay: self.decay, tau: self.tau, power: self.power }
    }

    /// The effective decay after `updates` updates: `decay · (1 − exp(−updates / tau))` (legacy) or `PC-PEMA`
    /// at `lf = 1.0`.
    pub fn decay_at(&self, updates: u64) -> f64 {
        self.decay_at_with_lf(updates, 1.0)
    }

    /// The effective decay after `updates` updates given the current schedule LR factor `lf`.
    pub fn decay_at_with_lf(&self, updates: u64, lf: f64) -> f64 {
        if let Some(p) = self.power {
            p.decay_at(updates, lf)
        } else {
            self.decay * (1.0 - (-(updates as f64) / self.tau).exp())
        }
    }

    /// Fold the current `source` weights into the average (using `lf = 1.0` for `PC-PEMA`).
    pub fn update(&mut self, source: &dyn Module) -> Result<()> {
        self.update_with_lf(source, 1.0)
    }

    /// Fold the current `source` weights into the average with schedule LR factor `lf` (used by `PC-PEMA`;
    /// identical to [`ModelEma::update`] in legacy mode).
    pub fn update_with_lf(&mut self, source: &dyn Module, lf: f64) -> Result<()> {
        let d = self.decay_at_with_lf(self.updates + 1, lf);
        let (d32, w) = (d as f32, (1.0 - d) as f32);
        let Self { cached, cached_keys, cached_disjoint, prefix, pair_scratch, updates, .. } = self;
        let guard = ClearGuard(pair_scratch);
        collect_matching_source(source, cached, prefix, guard.0)?;
        let pairs: &[(usize, Tensor)] = guard.0;

        // On a device, every tracked tensor in one multi-tensor update (16 per dispatch on the GPU).
        if pairs.iter().any(|&(idx, _)| !cached[idx].tensor.device().is_cpu()) {
            let _g = no_grad();
            let es: Vec<&Tensor> = pairs.iter().map(|&(idx, _)| &cached[idx].tensor).collect();
            let ms: Vec<&Tensor> = pairs.iter().map(|(_, m)| m).collect();
            let none = vec![Vec::new(); es.len()];
            if Tensor::optimizer_step_many_(&es, &ms, &none, &OptimStep::Axpby { a: d32, b: w })? {
                drop(guard);
                *updates += 1;
                return Ok(());
            }
        }

        let all_fast_cpu = pairs.iter().all(|(idx, m)| is_fast_cpu_pair(&cached[*idx].tensor, m));
        if all_fast_cpu {
            let total_numel: usize = pairs.iter().map(|&(idx, _)| cached[idx].tensor.numel()).sum();
            let can_par_tensors = *cached_disjoint
                && total_numel >= EMA_PAR_THRESHOLD
                && pairs.len() > 1
                && pairs.iter().all(|(_, m)| !cached_keys.contains(&storage_key(m)));
            if can_par_tensors {
                folo_core::cpu::in_pool(|| {
                    pairs.par_iter().try_for_each(|(idx, m)| {
                        let _g = no_grad();
                        ema_update_cpu_tensor(&cached[*idx].tensor, m, d32, w, false)
                    })
                })?;
            } else {
                let _g = no_grad();
                for (idx, m) in pairs {
                    ema_update_cpu_tensor(&cached[*idx].tensor, m, d32, w, true)?;
                }
            }
        } else {
            let _g = no_grad();
            for (idx, m) in pairs {
                let e = &cached[*idx].tensor;
                if is_fast_cpu_pair(e, m) {
                    ema_update_cpu_tensor(e, m, d32, w, true)?;
                } else if e.device().is_cpu()
                    && m.device().is_cpu()
                    && e.dtype() == DType::F32
                    && m.dtype() == DType::F32
                    && e.is_contiguous()
                {
                    // `e` and `m` share storage: snapshot `m` first to avoid RwLock deadlock and
                    // read-after-write corruption.
                    let m_copy = snapshot_tensor(m)?;
                    ema_update_cpu_tensor(e, &m_copy, d32, w, true)?;
                } else {
                    let m_copy;
                    let m_ref = if e.same_storage(m) {
                        m_copy = snapshot_tensor(m)?;
                        &m_copy
                    } else {
                        m
                    };
                    if !e.device().is_cpu()
                        && e.optimizer_step_(m_ref, &[], &OptimStep::Axpby { a: d32, b: w })?
                    {
                        continue;
                    }
                    e.mul_scalar_assign(d32)?;
                    e.add_scaled_assign(m_ref, w)?;
                }
            }
        }
        drop(guard);
        *updates += 1;
        Ok(())
    }

    /// Copy every tracked tensor from `source` into `self.module` under `no_grad`.
    fn copy_from_source(&mut self, source: &dyn Module) -> Result<()> {
        let Self { cached, prefix, pair_scratch, .. } = self;
        let guard = ClearGuard(pair_scratch);
        collect_matching_source(source, cached, prefix, guard.0)?;
        let _g = no_grad();
        for (idx, m) in guard.0.iter() {
            cached[*idx].tensor.copy_from(m)?;
        }
        Ok(())
    }
}

/// Drops references to `source`'s tensors on every exit path (normal return, `?`, or panic).
struct ClearGuard<'a>(&'a mut Vec<(usize, Tensor)>);

impl Drop for ClearGuard<'_> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

fn snapshot_tensor(t: &Tensor) -> Result<Tensor> {
    let copy = Tensor::zeros_with(t.shape(), t.dtype(), t.device())?;
    copy.copy_from(t)?;
    Ok(copy)
}

#[inline]
fn is_fast_cpu_pair(e: &Tensor, m: &Tensor) -> bool {
    e.device().is_cpu()
        && m.device().is_cpu()
        && e.dtype() == DType::F32
        && m.dtype() == DType::F32
        && e.is_contiguous()
        && !e.same_storage(m)
}

#[inline]
fn ema_update_cpu_tensor(e: &Tensor, m: &Tensor, d32: f32, w: f32, par_chunks: bool) -> Result<()> {
    m.with_f32_slice(|ms| e.update_f32(|es| ema_update_slice(es, ms, d32, w, par_chunks)))??;
    Ok(())
}

/// Update `dst[i] = dst[i] * d + src[i] * w` in place, using AVX2 (without FMA) and optional chunked
/// rayon parallelism inside `folo_core::cpu::in_pool` when the caller is not already parallelizing
/// across tensors.
#[inline]
fn ema_update_slice(dst: &mut [f32], src: &[f32], d: f32, w: f32, par_chunks: bool) {
    assert_eq!(dst.len(), src.len());
    if par_chunks && dst.len() >= EMA_CHUNK_PAR_THRESHOLD {
        folo_core::cpu::in_pool(|| {
            dst.par_chunks_mut(EMA_CHUNK_SIZE)
                .zip(src.par_chunks(EMA_CHUNK_SIZE))
                .for_each(|(dc, sc)| ema_chunk(dc, sc, d, w));
        });
    } else {
        ema_chunk(dst, src, d, w);
    }
}

#[inline]
fn ema_chunk(dst: &mut [f32], src: &[f32], d: f32, w: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support is checked above; `ema_slice_avx2` enforces `dst.len() == src.len()`.
        unsafe { ema_slice_avx2(dst, src, d, w) };
        return;
    }
    ema_slice_scalar(dst, src, d, w);
}

#[inline]
fn ema_slice_scalar(dst: &mut [f32], src: &[f32], d: f32, w: f32) {
    for (e, &m) in dst.iter_mut().zip(src) {
        *e = *e * d + m * w;
    }
}

/// AVX2 implementation of `dst[i] = dst[i] * d + src[i] * w`.
///
/// Intentionally omits `fma` so every lane performs separate IEEE-754 multiply and add instructions,
/// matching [`ema_slice_scalar`] bit-for-bit on all non-NaN IEEE-754 values (Rust does not guarantee
/// identical NaN payloads across scalar and SIMD code generation).
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ema_slice_avx2(dst: &mut [f32], src: &[f32], d: f32, w: f32) {
    use std::arch::x86_64::*;

    assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let vd = _mm256_set1_ps(d);
    let vw = _mm256_set1_ps(w);
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `dst.len() == src.len() == n`.
        unsafe {
            let e0 = _mm256_loadu_ps(dst_ptr.add(i));
            let m0 = _mm256_loadu_ps(src_ptr.add(i));
            let e1 = _mm256_loadu_ps(dst_ptr.add(i + 8));
            let m1 = _mm256_loadu_ps(src_ptr.add(i + 8));
            let e2 = _mm256_loadu_ps(dst_ptr.add(i + 16));
            let m2 = _mm256_loadu_ps(src_ptr.add(i + 16));
            let e3 = _mm256_loadu_ps(dst_ptr.add(i + 24));
            let m3 = _mm256_loadu_ps(src_ptr.add(i + 24));

            let r0 = _mm256_add_ps(_mm256_mul_ps(e0, vd), _mm256_mul_ps(m0, vw));
            let r1 = _mm256_add_ps(_mm256_mul_ps(e1, vd), _mm256_mul_ps(m1, vw));
            let r2 = _mm256_add_ps(_mm256_mul_ps(e2, vd), _mm256_mul_ps(m2, vw));
            let r3 = _mm256_add_ps(_mm256_mul_ps(e3, vd), _mm256_mul_ps(m3, vw));

            _mm256_storeu_ps(dst_ptr.add(i), r0);
            _mm256_storeu_ps(dst_ptr.add(i + 8), r1);
            _mm256_storeu_ps(dst_ptr.add(i + 16), r2);
            _mm256_storeu_ps(dst_ptr.add(i + 24), r3);
        }
        i += 32;
    }

    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `dst.len() == src.len() == n`.
        unsafe {
            let e0 = _mm256_loadu_ps(dst_ptr.add(i));
            let m0 = _mm256_loadu_ps(src_ptr.add(i));
            let r0 = _mm256_add_ps(_mm256_mul_ps(e0, vd), _mm256_mul_ps(m0, vw));
            _mm256_storeu_ps(dst_ptr.add(i), r0);
        }
        i += 8;
    }

    ema_slice_scalar(&mut dst[i..], &src[i..], d, w);
}

/// Device of the first tensor in `m` (canonical order), without allocating.
fn first_device(m: &dyn Module) -> Option<Device> {
    let mut found = None;
    m.visit_tensors(&mut |_, t, _| {
        if found.is_none() {
            found = Some(t.device());
        }
    });
    if found.is_some() {
        return found;
    }
    m.visit_children(&mut |_, child| {
        if found.is_none() {
            found = first_device(child);
        }
    });
    found
}

fn join_prefix(prefix: &mut String, name: &str) -> usize {
    let len = prefix.len();
    if !name.is_empty() {
        if !prefix.is_empty() {
            prefix.push('.');
        }
        prefix.push_str(name);
    }
    len
}

/// Walk `source`'s tracked tensors in canonical order without allocating `String`s, verifying exact
/// name, shape, dtype, and count against the currently tracked entries of `cached`, and storing
/// matched `(cached_index, source_tensor)` pairs into `out`.
fn collect_matching_source(
    source: &dyn Module,
    cached: &[CachedTensor],
    prefix: &mut String,
    out: &mut Vec<(usize, Tensor)>,
) -> Result<()> {
    prefix.clear();
    out.clear();
    let mut expected = cached.iter().enumerate().filter(|(_, c)| is_tracked(c.kind, &c.tensor));
    let ok = walk_tracked(source, prefix, &mut |name, t| {
        let Some((cached_idx, exp)) = expected.next() else {
            return false;
        };
        if name != exp.name.as_str() || t.shape() != exp.tensor.shape() || t.dtype() != exp.tensor.dtype() {
            return false;
        }
        out.push((cached_idx, t.clone()));
        true
    });
    if !ok || expected.next().is_some() {
        out.clear();
        return Err(Error::Invalid(
            "ModelEma: the source model's tensors do not match the EMA model's".into(),
        ));
    }
    Ok(())
}

fn walk_tracked(m: &dyn Module, prefix: &mut String, f: &mut dyn FnMut(&str, &Tensor) -> bool) -> bool {
    let mut ok = true;
    m.visit_tensors(&mut |name, t, kind| {
        if !ok {
            return;
        }
        if is_tracked(kind, t) {
            let len = join_prefix(prefix, name);
            ok = f(prefix.as_str(), t);
            prefix.truncate(len);
        }
    });
    if !ok {
        return false;
    }
    m.visit_children(&mut |name, child| {
        if !ok {
            return;
        }
        let len = join_prefix(prefix, name);
        ok = walk_tracked(child, prefix, f);
        prefix.truncate(len);
    });
    ok
}

/// What a checkpoint needs from an EMA: object-safe, so checkpoint code does not depend on the model type.
pub trait EmaState {
    /// Every tracked tensor with its model name (no `ema.` prefix), canonical order.
    fn ema_tensors(&self) -> Vec<(String, Tensor)>;
    /// `ema.__updates`.
    fn updates(&self) -> u64;
    /// `folo.ema`.
    fn contract(&self) -> EmaContract;
    /// Restore: `tensors` maps model names to saved values and must cover every tracked tensor exactly, with
    /// the same shapes.
    fn restore(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        updates: u64,
        contract: EmaContract,
    ) -> Result<()>;
}

impl<M: Module> EmaState for ModelEma<M> {
    fn ema_tensors(&self) -> Vec<(String, Tensor)> {
        self.cached
            .iter()
            .filter(|c| is_tracked(c.kind, &c.tensor))
            .map(|c| (c.name.clone(), c.tensor.clone()))
            .collect()
    }

    fn updates(&self) -> u64 {
        self.updates
    }

    fn contract(&self) -> EmaContract {
        EmaContract { decay: self.decay, tau: self.tau, power: self.power }
    }

    fn restore(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        updates: u64,
        contract: EmaContract,
    ) -> Result<()> {
        let mine: Vec<(&str, &Tensor)> = self
            .cached
            .iter()
            .filter(|c| is_tracked(c.kind, &c.tensor))
            .map(|c| (c.name.as_str(), &c.tensor))
            .collect();
        let mut problems = Vec::new();
        for &(n, t) in &mine {
            match tensors.get(n) {
                None => problems.push(format!("`ema.{n}` is missing")),
                Some(s) if s.shape() != t.shape() => {
                    problems.push(format!("`ema.{n}` is {:?}, model has {:?}", s.dims(), t.dims()))
                }
                Some(_) => {}
            }
        }
        let names: std::collections::HashSet<&str> = mine.iter().map(|&(n, _)| n).collect();
        for n in tensors.keys() {
            if !names.contains(n.as_str()) {
                problems.push(format!("`ema.{n}` is not a tensor of this model"));
            }
        }
        if !problems.is_empty() {
            problems.sort();
            return Err(Error::Load(format!("EMA state: {}", problems.join("; "))));
        }
        if !(0.0..=1.0).contains(&contract.decay) || !(contract.tau.is_finite() && contract.tau > 0.0) {
            return Err(Error::Load(format!(
                "folo.ema: invalid decay {} / tau {}",
                contract.decay, contract.tau
            )));
        }
        if let Some(p) = contract.power {
            p.validate().map_err(|e| Error::Load(format!("folo.ema: {e}")))?;
        }
        let _g = no_grad();
        for &(n, t) in &mine {
            t.copy_from(&tensors[n].to_dtype(DType::F32)?.to_device(t.device())?)?;
        }
        self.decay = contract.decay;
        self.tau = contract.tau;
        self.power = contract.power;
        self.updates = updates;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_avx2_matches_scalar_bitwise() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let (d, w) = (0.9993f32, 0.0007f32);
            for n in [0, 1, 7, 8, 15, 31, 32, 33, 63, 64, 65, 1023] {
                let mut a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 5.0).collect();
                let mut b = a.clone();
                let src: Vec<f32> = (0..n).map(|i| (i as f32) * -0.07 + 1.25).collect();
                ema_slice_scalar(&mut a, &src, d, w);
                // SAFETY: AVX2 support was checked above.
                unsafe { ema_slice_avx2(&mut b, &src, d, w) };
                let ab: Vec<u32> = a.into_iter().map(f32::to_bits).collect();
                let bb: Vec<u32> = b.into_iter().map(f32::to_bits).collect();
                assert_eq!(ab, bb, "AVX2 vs scalar mismatch at len={n}");
            }
            // Special IEEE-754 non-NaN values: signed zeros, subnormals, infinities.
            let special_e = [
                0.0f32,
                -0.0,
                f32::MIN_POSITIVE / 8.0,
                -f32::MIN_POSITIVE / 4.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1.0,
                -1.0,
                0.0,
            ];
            let special_m = [
                -0.0f32,
                0.0,
                f32::MIN_POSITIVE / 16.0,
                f32::MIN_POSITIVE / 2.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -0.0,
                0.0,
                -0.0,
            ];
            let mut a = special_e.to_vec();
            let mut b = special_e.to_vec();
            ema_slice_scalar(&mut a, &special_m, 0.5, 0.5);
            // SAFETY: AVX2 support was checked above.
            unsafe { ema_slice_avx2(&mut b, &special_m, 0.5, 0.5) };
            let ab: Vec<u32> = a.into_iter().map(f32::to_bits).collect();
            let bb: Vec<u32> = b.into_iter().map(f32::to_bits).collect();
            assert_eq!(ab, bb, "AVX2 vs scalar mismatch on special IEEE-754 values");
        }
    }

    #[test]
    fn pc_pema_formula_prior_shift_and_contract_round_trip() {
        let scratch = EmaPowerConfig::new(0.0);
        scratch.validate().unwrap();
        assert_eq!(scratch.decay_at(1, 1.0), 0.0);
        let d10 = scratch.decay_at(10, 1.0);
        let want10 = ((PEMA_GAMMA_0 + 1.0) * (-1.0f64 / 10.0).ln_1p()).exp();
        assert!((d10 - want10).abs() < 1e-14);
        // Lower lr factor increases gamma_eff -> smaller decay (faster adaptation during tail/warmup).
        let d10_low = scratch.decay_at(10, 0.04);
        assert!(d10_low < d10, "lower lf should shorten effective EMA window: {d10_low} vs {d10}");
        // Clamped below lf_min = 0.01 and above 1.0.
        assert_eq!(scratch.decay_at(50, 0.0001).to_bits(), scratch.decay_at(50, PEMA_LF_MIN).to_bits());
        assert_eq!(scratch.decay_at(50, 1.5).to_bits(), scratch.decay_at(50, 1.0).to_bits());

        // Transfer prior shift t_0 > 0 keeps beta_1 > 0 so pretrained shadow is not overwritten at u=1.
        let xfer = EmaPowerConfig::new(39.0);
        let d1_xfer = xfer.decay_at(1, 1.0);
        let want1_xfer = ((PEMA_GAMMA_0 + 1.0) * (-1.0f64 / 40.0).ln_1p()).exp();
        assert!((d1_xfer - want1_xfer).abs() < 1e-14 && d1_xfer > 0.8);

        // Contract serde: legacy omits `power` byte-identically; PC-PEMA round-trips exact f64.
        let leg = EmaContract { decay: EMA_DECAY, tau: EMA_TAU, power: None };
        let leg_s = serde_json::to_string(&leg).unwrap();
        assert_eq!(leg_s, r#"{"decay":0.9999,"tau":2000.0}"#);
        assert_eq!(serde_json::from_str::<EmaContract>(&leg_s).unwrap(), leg);

        let pwr = EmaContract { decay: EMA_DECAY, tau: EMA_TAU, power: Some(xfer) };
        let pwr_s = serde_json::to_string(&pwr).unwrap();
        assert!(pwr_s.contains("power"), "{pwr_s}");
        assert_eq!(serde_json::from_str::<EmaContract>(&pwr_s).unwrap(), pwr);

        // Unknown fields on both EmaContract and EmaPowerConfig are rejected (`deny_unknown_fields`).
        assert!(serde_json::from_str::<EmaContract>(r#"{"decay":0.9999,"tau":2000.0,"extra":1}"#).is_err());
        assert!(
            serde_json::from_str::<EmaPowerConfig>(
                r#"{"gamma_0":6.94,"rho":0.25,"lf_min":0.01,"t_0":0.0,"extra":1}"#
            )
            .is_err()
        );

        // Invalid PC-PEMA configs are rejected by `validate()`.
        for bad in [
            EmaPowerConfig { gamma_0: -0.1, ..scratch },
            EmaPowerConfig { gamma_0: f64::NAN, ..scratch },
            EmaPowerConfig { rho: -0.1, ..scratch },
            EmaPowerConfig { lf_min: 0.0, ..scratch },
            EmaPowerConfig { lf_min: 1.1, ..scratch },
            EmaPowerConfig { t_0: -1.0, ..scratch },
            EmaPowerConfig { t_0: f64::INFINITY, ..scratch },
        ] {
            assert!(bad.validate().is_err(), "{bad:?} must fail validate()");
        }
    }
}
