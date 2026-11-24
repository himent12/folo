//! Exponential moving average of model weights (docs/MODEL.md §8.5).

use std::collections::HashMap;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor, no_grad};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::module::{Module, TensorKind};
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
    /// Cached tracked tensors of `self.module` (parameters and `F32` buffers, canonical order).
    /// Avoids walking `self.module` and allocating dotted names on every [`ModelEma::update`].
    tracked: Vec<(String, Tensor)>,
    /// Reusable dotted-name buffer for zero-allocation traversal of `source`.
    prefix: String,
    /// Reusable scratch list of `source`'s matched tracked tensors; always cleared before returning
    /// so `ModelEma` never retains handles to `source`.
    src_scratch: Vec<Tensor>,
    decay: f64,
    tau: f64,
    updates: u64,
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
        if !(0.0..=1.0).contains(&decay) || !(tau.is_finite() && tau > 0.0) {
            return Err(Error::Invalid(format!("ModelEma: invalid decay {decay} / tau {tau}")));
        }
        let mut ema = ModelEma {
            module,
            tracked: Vec::new(),
            prefix: String::new(),
            src_scratch: Vec::new(),
            decay,
            tau,
            updates: 0,
        };
        if let Some(dev) = first_device(source) {
            ema.module.to_device(dev)?;
        }
        for t in ema.module.named_tensors() {
            t.tensor.set_requires_grad(false);
        }
        ema.module.eval();
        ema.refresh_tracked();
        ema.copy_from_source(source)?;
        Ok(ema)
    }

    /// Rebuild the cached tracked tensor list from `self.module` and size the scratch buffers.
    fn refresh_tracked(&mut self) {
        self.tracked = tracked(&self.module);
        let max_name = self.tracked.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        if self.prefix.capacity() < max_name {
            self.prefix.reserve(max_name - self.prefix.capacity());
        }
        if self.src_scratch.capacity() < self.tracked.len() {
            self.src_scratch.reserve(self.tracked.len() - self.src_scratch.capacity());
        }
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
        self.refresh_tracked();
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

    /// `(decay, tau)`.
    pub fn contract(&self) -> EmaContract {
        EmaContract { decay: self.decay, tau: self.tau }
    }

    /// The effective decay after `updates` updates: `decay · (1 − exp(−updates / tau))`.
    pub fn decay_at(&self, updates: u64) -> f64 {
        self.decay * (1.0 - (-(updates as f64) / self.tau).exp())
    }

    /// Fold the current `source` weights into the average.
    pub fn update(&mut self, source: &dyn Module) -> Result<()> {
        let d = self.decay_at(self.updates + 1);
        let (d32, w) = (d as f32, (1.0 - d) as f32);
        let Self { tracked, prefix, src_scratch, updates, .. } = self;
        collect_matching_source(source, tracked, prefix, src_scratch)?;
        let guard = ClearGuard(src_scratch);
        let src: &[Tensor] = guard.0;

        // On a device, every tracked tensor in one multi-tensor update (16 per dispatch on the GPU).
        if tracked.iter().any(|(_, e)| !e.device().is_cpu()) {
            let _g = no_grad();
            let es: Vec<&Tensor> = tracked.iter().map(|(_, t)| t).collect();
            let ms: Vec<&Tensor> = src.iter().collect();
            let none = vec![Vec::new(); es.len()];
            if Tensor::optimizer_step_many_(&es, &ms, &none, &OptimStep::Axpby { a: d32, b: w })? {
                drop(guard);
                *updates += 1;
                return Ok(());
            }
        }

        let all_fast_cpu = tracked.iter().zip(src.iter()).all(|((_, e), m)| is_fast_cpu_pair(e, m));
        if all_fast_cpu {
            let total_numel: usize = tracked.iter().map(|(_, e)| e.numel()).sum();
            if total_numel >= EMA_PAR_THRESHOLD && tracked.len() > 1 {
                tracked.par_iter().zip(src.par_iter()).try_for_each(|((_, e), m)| {
                    let _g = no_grad();
                    ema_update_cpu_tensor(e, m, d32, w)
                })?;
            } else {
                let _g = no_grad();
                for ((_, e), m) in tracked.iter().zip(src.iter()) {
                    ema_update_cpu_tensor(e, m, d32, w)?;
                }
            }
        } else {
            let _g = no_grad();
            for ((_, e), m) in tracked.iter().zip(src.iter()) {
                if is_fast_cpu_pair(e, m) {
                    ema_update_cpu_tensor(e, m, d32, w)?;
                } else if !e.device().is_cpu()
                    && e.optimizer_step_(m, &[], &OptimStep::Axpby { a: d32, b: w })?
                {
                    continue;
                } else {
                    e.mul_scalar_assign(d32)?;
                    e.add_scaled_assign(m, w)?;
                }
            }
        }
        drop(guard);
        *updates += 1;
        Ok(())
    }

    /// Copy every tracked tensor from `source` into `self.module` under `no_grad`.
    fn copy_from_source(&mut self, source: &dyn Module) -> Result<()> {
        let Self { tracked, prefix, src_scratch, .. } = self;
        collect_matching_source(source, tracked, prefix, src_scratch)?;
        let guard = ClearGuard(src_scratch);
        let _g = no_grad();
        for ((_, e), m) in tracked.iter().zip(guard.0.iter()) {
            e.copy_from(m)?;
        }
        Ok(())
    }
}

/// Drops references to `source`'s tensors on every exit path (normal return, `?`, or panic).
struct ClearGuard<'a>(&'a mut Vec<Tensor>);

impl Drop for ClearGuard<'_> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

#[inline]
fn is_fast_cpu_pair(e: &Tensor, m: &Tensor) -> bool {
    e.device().is_cpu()
        && m.device().is_cpu()
        && e.dtype() == DType::F32
        && m.dtype() == DType::F32
        && e.is_contiguous()
        && e.id() != m.id()
}

#[inline]
fn ema_update_cpu_tensor(e: &Tensor, m: &Tensor, d32: f32, w: f32) -> Result<()> {
    m.with_f32_slice(|ms| e.update_f32(|es| ema_update_slice(es, ms, d32, w)))??;
    Ok(())
}

/// Update `dst[i] = dst[i] * d + src[i] * w` in place, using AVX2 (without FMA, so bit-exact with scalar)
/// and chunked rayon parallelism for large tensors.
#[inline]
fn ema_update_slice(dst: &mut [f32], src: &[f32], d: f32, w: f32) {
    debug_assert_eq!(dst.len(), src.len());
    if dst.len() >= EMA_CHUNK_PAR_THRESHOLD {
        dst.par_chunks_mut(EMA_CHUNK_SIZE)
            .zip(src.par_chunks(EMA_CHUNK_SIZE))
            .for_each(|(dc, sc)| ema_chunk(dc, sc, d, w));
    } else {
        ema_chunk(dst, src, d, w);
    }
}

#[inline]
fn ema_chunk(dst: &mut [f32], src: &[f32], d: f32, w: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        assert_eq!(dst.len(), src.len());
        // SAFETY: AVX2 support is checked above and `dst.len() == src.len()`.
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
/// matching [`ema_slice_scalar`] bit-for-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ema_slice_avx2(dst: &mut [f32], src: &[f32], d: f32, w: f32) {
    use std::arch::x86_64::*;

    let n = dst.len();
    debug_assert_eq!(n, src.len());
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
/// name, shape, dtype, and count against `tracked`, and storing matched handles into `out`.
fn collect_matching_source(
    source: &dyn Module,
    tracked: &[(String, Tensor)],
    prefix: &mut String,
    out: &mut Vec<Tensor>,
) -> Result<()> {
    prefix.clear();
    out.clear();
    let ok = walk_tracked(source, prefix, &mut |name, t| {
        let idx = out.len();
        let Some((exp_name, exp_tensor)) = tracked.get(idx) else {
            return false;
        };
        if name != exp_name.as_str() || t.shape() != exp_tensor.shape() || t.dtype() != exp_tensor.dtype() {
            return false;
        }
        out.push(t.clone());
        true
    });
    if !ok || out.len() != tracked.len() {
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
        if kind == TensorKind::Parameter || t.dtype() == DType::F32 {
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

/// Parameters and `F32` buffers, canonical order.
fn tracked<M: Module + ?Sized>(m: &M) -> Vec<(String, Tensor)> {
    m.named_tensors()
        .into_iter()
        .filter(|t| t.kind == TensorKind::Parameter || t.tensor.dtype() == DType::F32)
        .map(|t| (t.name, t.tensor))
        .collect()
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
        self.tracked.clone()
    }

    fn updates(&self) -> u64 {
        self.updates
    }

    fn contract(&self) -> EmaContract {
        EmaContract { decay: self.decay, tau: self.tau }
    }

    fn restore(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        updates: u64,
        contract: EmaContract,
    ) -> Result<()> {
        let mine = &self.tracked;
        let mut problems = Vec::new();
        for (n, t) in mine {
            match tensors.get(n) {
                None => problems.push(format!("`ema.{n}` is missing")),
                Some(s) if s.shape() != t.shape() => {
                    problems.push(format!("`ema.{n}` is {:?}, model has {:?}", s.dims(), t.dims()))
                }
                Some(_) => {}
            }
        }
        let names: std::collections::HashSet<&str> = mine.iter().map(|(n, _)| n.as_str()).collect();
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
        let _g = no_grad();
        for (n, t) in mine {
            t.copy_from(&tensors[n].to_dtype(DType::F32)?.to_device(t.device())?)?;
        }
        self.decay = contract.decay;
        self.tau = contract.tau;
        self.updates = updates;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use folo_core::rng::Rng;

    use super::*;
    use crate::{Activation, BatchNorm2d, Conv2d, Conv2dConfig, ConvBnAct, Forward, ModuleList, Sequential};

    /// Test module covering transparent (`""`) children, non-`F32` ignored buffers, odd lengths, and a
    /// large tensor above [`EMA_CHUNK_PAR_THRESHOLD`].
    struct MixedNet {
        seq: Sequential,
        list: ModuleList<Conv2d>,
        big: Tensor,
        odd: Tensor,
        int_buf: Tensor,
    }

    impl MixedNet {
        fn new(rng: &mut Rng) -> Self {
            let seq = Sequential::new()
                .push_named("stem", ConvBnAct::new(3, 17, 3, 1, 1, Activation::Silu, rng))
                .push_named("bn", BatchNorm2d::new(17));
            let list = ModuleList::new(vec![
                Conv2d::new(Conv2dConfig::new(17, 31, 3), rng),
                Conv2d::new(Conv2dConfig::new(31, 33, 1), rng),
            ]);
            let big = Tensor::randn([EMA_CHUNK_PAR_THRESHOLD + 37], 0.0, 1.0, rng).requires_grad_();
            let odd = Tensor::randn([39], 0.0, 1.0, rng).requires_grad_();
            let int_buf = Tensor::zeros_with([4], DType::I64, Device::Cpu).unwrap();
            MixedNet { seq, list, big, odd, int_buf }
        }
    }

    impl Module for MixedNet {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("big", &self.big, TensorKind::Parameter);
            f("odd", &self.odd, TensorKind::Parameter);
            f("int_buf", &self.int_buf, TensorKind::Buffer);
        }

        fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
            f("seq", &self.seq);
            f("", &self.list);
        }
    }

    /// Reference scalar EMA update matching the pre-optimization `ema.rs` expression bit-for-bit.
    fn ref_step(state: &mut [Vec<f32>], source: &dyn Module, decay: f64, tau: f64, next_update: u64) {
        let d = decay * (1.0 - (-(next_update as f64) / tau).exp());
        let (d32, w) = (d as f32, (1.0 - d) as f32);
        let src = tracked(source);
        assert_eq!(state.len(), src.len());
        for (e, (_, m)) in state.iter_mut().zip(&src) {
            let mv = m.to_vec::<f32>().unwrap();
            for (ev, &mv) in e.iter_mut().zip(&mv) {
                *ev = *ev * d32 + mv * w;
            }
        }
    }

    fn perturb_model(m: &dyn Module, rng: &mut Rng) {
        let _g = no_grad();
        for (_, t) in tracked(m) {
            let delta = Tensor::randn(t.shape().clone(), 0.0, 0.05, rng);
            t.add_assign(&delta).unwrap();
        }
    }

    fn tensor_bits(m: &dyn Module) -> Vec<(String, Vec<u32>)> {
        tracked(m)
            .into_iter()
            .map(|(n, t)| (n, t.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect()))
            .collect()
    }

    #[test]
    fn ema_bit_exact_100_steps_vs_scalar_reference() {
        let mut rng = Rng::seed(42);
        let live = MixedNet::new(&mut rng);
        let mut ema_fast = ModelEma::with_params(MixedNet::new(&mut rng), &live, 0.999, 50.0).unwrap();
        let mut ref_state: Vec<Vec<f32>> =
            tracked(&live).into_iter().map(|(_, t)| t.to_vec::<f32>().unwrap()).collect();

        // Also update BatchNorm running stats via a forward pass during training.
        let x = Tensor::randn([2, 3, 4, 4], 0.0, 1.0, &mut rng);
        for step in 1..=100u64 {
            perturb_model(&live, &mut rng);
            if step % 25 == 0 {
                let _ = live.seq.forward(&x).unwrap();
            }
            ema_fast.update(&live).unwrap();
            ref_step(&mut ref_state, &live, 0.999, 50.0, step);
        }

        assert_eq!(ema_fast.updates(), 100);
        let fast_bits = tensor_bits(ema_fast.module());
        for ((name, got), want) in fast_bits.iter().zip(&ref_state) {
            let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, &want_bits, "tensor `{name}` diverged from scalar reference after 100 steps");
        }

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let (d, w) = (0.9993f32, 0.0007f32);
            for n in [0, 1, 7, 8, 15, 31, 32, 33, 63, 64, 65, 1023] {
                let mut a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 5.0).collect();
                let mut b = a.clone();
                let src: Vec<f32> = (0..n).map(|i| (i as f32) * -0.07 + 1.25).collect();
                ema_slice_scalar(&mut a, &src, d, w);
                // SAFETY: AVX2 support was checked above and `b.len() == src.len()`.
                unsafe { ema_slice_avx2(&mut b, &src, d, w) };
                let ab: Vec<u32> = a.into_iter().map(f32::to_bits).collect();
                let bb: Vec<u32> = b.into_iter().map(f32::to_bits).collect();
                assert_eq!(ab, bb, "AVX2 vs scalar mismatch at len={n}");
            }
        }
    }

    #[test]
    fn ema_resume_reload_and_reset_consistency() {
        let mut rng = Rng::seed(7);
        let live = MixedNet::new(&mut rng);
        let mut ema_a = ModelEma::with_params(MixedNet::new(&mut rng), &live, 0.99, 20.0).unwrap();
        for _ in 0..15 {
            perturb_model(&live, &mut rng);
            ema_a.update(&live).unwrap();
        }
        // Snapshot & restore into a fresh EMA with different initial contract and weights.
        let saved: HashMap<String, Tensor> = ema_a
            .ema_tensors()
            .into_iter()
            .map(|(n, t)| (n, Tensor::from_vec(t.to_vec::<f32>().unwrap(), t.shape().clone()).unwrap()))
            .collect();
        let saved_updates = ema_a.updates();
        let saved_contract = ema_a.contract();

        let mut ema_b = ModelEma::with_params(MixedNet::new(&mut rng), &live, 0.5, 5.0).unwrap();
        ema_b.restore(&saved, saved_updates, saved_contract).unwrap();
        assert_eq!(ema_b.updates(), saved_updates);
        assert_eq!(ema_b.contract(), saved_contract);
        assert_eq!(tensor_bits(ema_a.module()), tensor_bits(ema_b.module()));

        for _ in 0..15 {
            perturb_model(&live, &mut rng);
            ema_a.update(&live).unwrap();
            ema_b.update(&live).unwrap();
        }
        assert_eq!(tensor_bits(ema_a.module()), tensor_bits(ema_b.module()));

        // Reset matches a newly initialized EMA.
        ema_a.reset(&live).unwrap();
        assert_eq!(ema_a.updates(), 0);
        let mut ema_fresh = ModelEma::with_params(MixedNet::new(&mut rng), &live, 0.99, 20.0).unwrap();
        for _ in 0..5 {
            perturb_model(&live, &mut rng);
            ema_a.update(&live).unwrap();
            ema_fresh.update(&live).unwrap();
        }
        assert_eq!(tensor_bits(ema_a.module()), tensor_bits(ema_fresh.module()));
    }

    #[test]
    fn ema_mismatch_errors_without_mutating() {
        struct Renamed(Tensor);
        impl Module for Renamed {
            fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
                f("other_name", &self.0, TensorKind::Parameter);
            }
        }
        struct Named(Tensor);
        impl Module for Named {
            fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
                f("weight", &self.0, TensorKind::Parameter);
            }
        }

        let live = Named(Tensor::full([4], 2.0).requires_grad_());
        let mut ema = ModelEma::new(Named(Tensor::zeros([4])), &live).unwrap();
        ema.update(&live).unwrap();
        let before = tensor_bits(ema.module());

        // Same shape & count, different name.
        let wrong_name = Renamed(Tensor::full([4], 9.0).requires_grad_());
        assert!(ema.update(&wrong_name).is_err());

        // Same name, different shape.
        let wrong_shape = Named(Tensor::full([5], 9.0).requires_grad_());
        assert!(ema.update(&wrong_shape).is_err());

        // Same name & shape, different dtype.
        let wrong_dtype = Named(Tensor::full_with([4], 9.0, DType::F16, Device::Cpu).unwrap());
        assert!(ema.update(&wrong_dtype).is_err());

        // Different count (empty module).
        let empty = Sequential::new();
        assert!(ema.update(&empty).is_err());

        assert_eq!(ema.updates(), 1);
        assert_eq!(tensor_bits(ema.module()), before);
    }

    #[test]
    fn ema_non_contiguous_fallback_and_thread_count_determinism() {
        // Non-contiguous EMA tensor exercises the generic fallback path.
        struct NonContig(Tensor);
        impl Module for NonContig {
            fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
                f("w", &self.0, TensorKind::Parameter);
            }
        }
        let base_live = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let base_ema = Tensor::zeros([3, 2]).t().unwrap();
        assert!(!base_ema.is_contiguous());
        let mut ema_nc =
            ModelEma::with_params(NonContig(base_ema), &NonContig(base_live.clone()), 0.9, 10.0).unwrap();
        let mut ema_c =
            ModelEma::with_params(NonContig(Tensor::zeros([2, 3])), &NonContig(base_live.clone()), 0.9, 10.0)
                .unwrap();
        {
            let _g = no_grad();
            base_live.add_assign(&Tensor::full([2, 3], 1.5)).unwrap();
        }
        ema_nc.update(&NonContig(base_live.clone())).unwrap();
        ema_c.update(&NonContig(base_live)).unwrap();
        assert_eq!(tensor_bits(ema_nc.module()), tensor_bits(ema_c.module()));

        // Determinism across rayon thread counts (1 vs 4).
        let mut rng = Rng::seed(99);
        let live = MixedNet::new(&mut rng);
        let mut ema_1 = ModelEma::with_params(MixedNet::new(&mut rng), &live, 0.99, 10.0).unwrap();
        let mut ema_4 = ModelEma::with_params(MixedNet::new(&mut rng), &live, 0.99, 10.0).unwrap();
        perturb_model(&live, &mut rng);
        let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let p4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        p1.install(|| ema_1.update(&live).unwrap());
        p4.install(|| ema_4.update(&live).unwrap());
        assert_eq!(tensor_bits(ema_1.module()), tensor_bits(ema_4.module()));
    }
}
