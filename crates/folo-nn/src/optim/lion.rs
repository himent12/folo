//! Lion (EvoLved Sign Momentum; Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023).
//!
//! Lion tracks a single first-moment buffer `m` per parameter (half the optimizer state of [`super::AdamW`]) and
//! applies sign-normalized updates with decoupled weight decay:
//!
//! ```text
//! c_t = β1 · m_{t-1} + (1 − β1) · g_t
//! θ_t = θ_{t-1} · (1 − η · λ) − η · sign(c_t)
//! m_t = β2 · m_{t-1} + (1 − β2) · g_t
//! ```
//!
//! where `sign(x)` is `+1` for `x > 0`, `-1` for `x < 0`, `0` for `x == 0`, and `NaN` for `NaN` (preserving
//! non-finite poison on both CPU and GPU). Hyperparameters are validated in `f64` (`lr >= 0`, `beta1 ∈ [0, 1)`,
//! `beta2 ∈ [0, 1)`, `weight_decay >= 0`); a parameter with no gradient in a step is skipped completely (no
//! weight decay and no state update), matching [`super::AdamW`] and [`super::Sgd`].

use std::collections::HashMap;

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{
    GradSource, ParamGroup, ParamPlan, check_param_sets, cpu_fast, for_each_param_planned, take_state,
    validate_groups,
};
use crate::{Error, Result};

/// Default Lion learning rate (Chen et al. 2023 recommend ~3–10× smaller than AdamW).
pub const LION_DEFAULT_LR: f64 = 1e-4;
/// Default Lion update-interpolation momentum `β1` (Chen et al. 2023).
pub const LION_DEFAULT_BETA1: f64 = 0.9;
/// Default Lion EMA-tracking momentum `β2` (Chen et al. 2023).
pub const LION_DEFAULT_BETA2: f64 = 0.99;

/// Serializable contract for a [`Lion`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LionGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate `η >= 0`.
    pub lr: f64,
    /// `[β1, β2]`, each in `[0, 1)`.
    pub betas: [f64; 2],
    /// Decoupled weight decay `λ >= 0`.
    pub weight_decay: f64,
}

/// Serializable contract for [`Lion`] checkpointing and resume parity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LionContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<LionGroupContract>,
    /// Completed optimizer steps.
    pub step: u64,
}

/// Lion optimizer (Evolved Sign Momentum; Chen et al. 2023).
///
/// Uses [`ParamGroup::momentum`] / [`ParamGroup::base_momentum`] for `β1` and [`ParamGroup::beta2`] for `β2`
/// (set cleanly via [`ParamGroup::lion`] or [`Lion::from_params`]), and integrates with [`crate::LrSchedule`]
/// via [`crate::LrSchedule::apply_to_groups`].
pub struct Lion {
    groups: Vec<ParamGroup>,
    /// Single `exp_avg` momentum buffer `m` per parameter.
    state: Vec<Vec<Tensor>>,
    plan: ParamPlan,
    coefs: Vec<Coef>,
    step: u64,
}

struct Coef {
    decay: Option<f32>,
    lr: f32,
    b1: f32,
    one_minus_b1: f32,
    b2: f32,
    one_minus_b2: f32,
}

impl Lion {
    /// Create a [`Lion`] optimizer over `groups` (no duplicate group or parameter names; all `F32`;
    /// finite `lr >= 0`, `beta1 ∈ [0, 1)`, `beta2 ∈ [0, 1)`, `weight_decay >= 0`).
    pub fn new(groups: Vec<ParamGroup>) -> Result<Self> {
        validate_lion_groups(&groups)?;
        let state = zero_state(&groups)?;
        let plan = ParamPlan::from_groups(&groups);
        let coefs = Vec::with_capacity(groups.len());
        Ok(Lion { groups, state, plan, coefs, step: 0 })
    }

    /// Convenience constructor for a single `"params"` group with explicit Lion hyperparameters.
    pub fn from_params(
        params: Vec<(String, Tensor)>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        let group = ParamGroup::new("params", params).lr(lr).lion(beta1, beta2).weight_decay(weight_decay);
        Self::new(vec![group])
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups (e.g. for [`crate::LrSchedule::apply_to_groups`]).
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Number of completed [`Lion::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// The momentum buffer `m` (`exp_avg`) of a parameter.
    pub fn exp_avg(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.state.get(group)?.get(param)
    }

    /// Perform one Lion optimization step over every parameter that has a gradient in `grads`.
    ///
    /// Validates all group hyperparameters and incoming gradient shapes before mutating any parameter or
    /// momentum buffer.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        validate_lion_step_groups(&self.groups)?;
        for grp in &self.groups {
            for (name, p) in &grp.params {
                if let Some(g) = grads.grad(p)
                    && g.shape() != p.shape()
                {
                    return Err(Error::Invalid(format!(
                        "lion: gradient of `{name}` is {:?}, parameter is {:?}",
                        g.dims(),
                        p.dims()
                    )));
                }
            }
        }
        self.plan.sync(&self.groups);
        self.coefs.clear();
        self.coefs.extend(self.groups.iter().map(|g| {
            let (b1, b2) = (g.momentum, g.beta2);
            Coef {
                decay: (g.weight_decay != 0.0).then_some((1.0 - g.lr * g.weight_decay) as f32),
                lr: g.lr as f32,
                b1: b1 as f32,
                one_minus_b1: (1.0 - b1) as f32,
                b2: b2 as f32,
                one_minus_b2: (1.0 - b2) as f32,
            }
        }));
        let (groups, state, plan, coefs) = (&self.groups, &self.state, &self.plan, &self.coefs);
        for_each_param_planned(plan, groups, grads, |gi, pi| {
            let (_, p) = &groups[gi].params[pi];
            let Some(g) = grads.grad(p) else { return Ok(()) };
            update(p, g, &state[gi][pi], &coefs[gi])
        })?;
        self.step += 1;
        Ok(())
    }

    /// Export state tensors as `(<param name>.exp_avg, tensor)` in canonical group / parameter order.
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), m) in g.params.iter().zip(st) {
                out.push((format!("{name}.exp_avg"), m.clone()));
            }
        }
        Ok(out)
    }

    /// Restore step count and `exp_avg` state tensors from a checkpoint map.
    ///
    /// Follows `folo-nn`'s optimizer contract: missing state tensors stay zero when `step == 0`, while
    /// missing or shape/dtype-mismatched state tensors error when `step > 0`.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                if let Some(t) = take_state(state, name, p, "exp_avg", step > 0, &mut problems) {
                    loaded.push((gi, pi, t));
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("Lion state: {}", problems.join("; "))));
        }
        let fresh = zero_state(&self.groups)?;
        let _g = no_grad();
        for (gi, pi, t) in loaded {
            let dst = &fresh[gi][pi];
            dst.copy_from(&t.to_device(dst.device())?)?;
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }

    /// Move all parameters and `exp_avg` state buffers to `device` in place, preserving tensor identities.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((_, p), m) in g.params.iter().zip(st) {
                p.to_device_(device)?;
                m.to_device_(device)?;
            }
        }
        Ok(())
    }

    /// Build a serializable [`LionContract`] describing this optimizer's groups, hyperparameters, and step.
    pub fn contract(&self) -> LionContract {
        LionContract {
            groups: self
                .groups
                .iter()
                .map(|g| LionGroupContract {
                    name: g.name.clone(),
                    params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                    lr: g.base_lr,
                    betas: [g.base_momentum, g.beta2],
                    weight_decay: g.weight_decay,
                })
                .collect(),
            step: self.step,
        }
    }

    /// Validate that `c` matches this optimizer's group names and parameter sets, then adopt its
    /// hyperparameters (used on resume before [`Lion::load_state`]).
    pub fn apply_contract(&mut self, c: &LionContract) -> Result<()> {
        let compat: Vec<super::GroupContract> = c
            .groups
            .iter()
            .map(|gc| super::GroupContract {
                name: gc.name.clone(),
                params: gc.params.clone(),
                lr: gc.lr,
                momentum: None,
                nesterov: None,
                weight_decay: gc.weight_decay,
                betas: Some(gc.betas),
                eps: Some(1e-8),
                decoupled_wd: false,
                cautious_wd: false,
                cautious: false,
            })
            .collect();
        check_param_sets(&self.groups, &compat)?;
        // Validate all incoming group hyperparameters before mutating self.
        for gc in &c.groups {
            let [b1, b2] = gc.betas;
            if !valid_lion_hparams(gc.lr, b1, b2, gc.weight_decay) {
                return Err(Error::Load(format!(
                    "lion contract group `{}`: invalid hyperparameters",
                    gc.name
                )));
            }
        }
        for (g, gc) in self.groups.iter_mut().zip(&c.groups) {
            let [b1, b2] = gc.betas;
            g.base_lr = gc.lr;
            g.lr = gc.lr;
            g.base_momentum = b1;
            g.momentum = b1;
            g.beta2 = b2;
            g.weight_decay = gc.weight_decay;
        }
        Ok(())
    }
}

#[inline]
fn valid_lion_hparams(lr: f64, beta1: f64, beta2: f64, weight_decay: f64) -> bool {
    lr.is_finite()
        && lr >= 0.0
        && beta1.is_finite()
        && (0.0..1.0).contains(&beta1)
        && beta2.is_finite()
        && (0.0..1.0).contains(&beta2)
        && weight_decay.is_finite()
        && weight_decay >= 0.0
}

fn validate_lion_step_groups(groups: &[ParamGroup]) -> Result<()> {
    for g in groups {
        if g.cautious {
            return Err(Error::Invalid(format!(
                "lion group `{}`: cautious masking is not supported",
                g.name
            )));
        }
        if !valid_lion_hparams(g.lr, g.momentum, g.beta2, g.weight_decay) {
            return Err(Error::Invalid(format!(
                "lion group `{}`: invalid hyperparameters (lr={}, beta1={}, beta2={}, weight_decay={})",
                g.name, g.lr, g.momentum, g.beta2, g.weight_decay
            )));
        }
    }
    Ok(())
}

fn validate_lion_groups(groups: &[ParamGroup]) -> Result<()> {
    validate_groups(groups)?;
    validate_lion_step_groups(groups)
}

fn zero_state(groups: &[ParamGroup]) -> Result<Vec<Vec<Tensor>>> {
    groups
        .iter()
        .map(|g| {
            g.params.iter().map(|(_, p)| Ok(Tensor::zeros_with(p.shape(), DType::F32, p.device())?)).collect()
        })
        .collect()
}

/// Mathematical sign preserving `0.0` (including `-0.0`) as `0.0`, `±Inf` as `±1.0`, and `NaN` as `NaN`.
#[inline]
fn sign_nan_preserving(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f32::NAN
    }
}

fn lion_slice(p: &mut [f32], g: &[f32], m: &mut [f32], c: &Coef) {
    let neg_lr = -c.lr;
    if let Some(d) = c.decay {
        for ((pi, &gi), mi) in p.iter_mut().zip(g).zip(m.iter_mut()) {
            let m_prev = *mi;
            let c_t = c.b1 * m_prev + c.one_minus_b1 * gi;
            let s = sign_nan_preserving(c_t);
            *pi = *pi * d + s * neg_lr;
            *mi = c.b2 * m_prev + c.one_minus_b2 * gi;
        }
    } else {
        for ((pi, &gi), mi) in p.iter_mut().zip(g).zip(m.iter_mut()) {
            let m_prev = *mi;
            let c_t = c.b1 * m_prev + c.one_minus_b1 * gi;
            let s = sign_nan_preserving(c_t);
            *pi += s * neg_lr;
            *mi = c.b2 * m_prev + c.one_minus_b2 * gi;
        }
    }
}

fn update(p: &Tensor, g: &Tensor, m: &Tensor, c: &Coef) -> Result<()> {
    if cpu_fast(&[p, m])
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
        && !p.same_storage(m)
        && !m.same_storage(g)
    {
        return g.with_f32_slice(|g| -> Result<()> {
            m.update_f32(|m| p.update_f32(|p| lion_slice(p, g, m, c)))??;
            Ok(())
        })?;
    }
    let mut g_f32 = g.to_dtype(DType::F32)?;
    if p.same_storage(&g_f32) || m.same_storage(&g_f32) {
        // Snapshot aliased gradient before mutating `p` or `m`.
        g_f32 = g_f32.add_scalar(0.0)?;
    }
    let c_t = m.mul_scalar(c.b1)?.add(&g_f32.mul_scalar(c.one_minus_b1)?)?;
    // Bit-identical to `sign_nan_preserving` on all IEEE-754 f32 values (finite, ±0, ±Inf, NaN) with zero host readbacks.
    let pos = c_t.gt_scalar(0.0)?;
    let neg = c_t.lt_scalar(0.0)?;
    let s_raw = pos.sub(&neg)?;
    let nonzero = pos.add(&neg)?;
    let zero_or_nan = c_t.mul_scalar(0.0)?;
    let s = nonzero.where_cond(&s_raw, &zero_or_nan)?;
    if let Some(d) = c.decay {
        p.mul_scalar_assign(d)?;
    }
    p.add_scaled_assign(&s, -c.lr)?;
    m.mul_scalar_assign(c.b2)?;
    m.add_scaled_assign(&g_f32, c.one_minus_b2)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, ConvBnAct, Linear, Sequential};
    use folo_core::Grads;
    use folo_core::rng::Rng;

    fn grads_of(p: &Tensor, vals: &[f32]) -> Grads {
        let mut g = Grads::default();
        g.insert(p, Tensor::from_slice(vals, p.shape()).unwrap());
        g
    }

    #[test]
    fn lion_hand_computed_steps_and_generic_fast_path_parity() {
        // Hand-computed 2-step trace:
        // p0 = [1.0, -2.0], m0 = [0.0, 0.0], lr = 0.1, b1 = 0.9, b2 = 0.99, wd = 0.5
        // Step 1 with g1 = [0.5, -0.2]:
        //   c1 = 0.1 * [0.5, -0.2] = [0.05, -0.02] -> sign = [1, -1]
        //   decay = 1 - 0.1 * 0.5 = 0.95
        //   p1 = [1.0*0.95 - 0.1, -2.0*0.95 + 0.1] = [0.85, -1.80]
        //   m1 = 0.01 * [0.5, -0.2] = [0.005, -0.002]
        // Step 2 with g2 = [-0.01, 0.0]:
        //   c2 = [0.9*0.005 + 0.1*(-0.01), 0.9*(-0.002) + 0.1*0.0] = [0.0035, -0.0018] -> sign = [1, -1]
        //   p2 = [0.85*0.95 - 0.1, -1.80*0.95 + 0.1] = [0.7075, -1.61]
        //   m2 = [0.99*0.005 + 0.01*(-0.01), 0.99*(-0.002)] = [0.00485, -0.00198]
        let p = Tensor::from_slice(&[1.0f32, -2.0], [2]).unwrap().requires_grad_();
        let mut opt = Lion::from_params(vec![("w".into(), p.clone())], 0.1, 0.9, 0.99, 0.5).unwrap();
        opt.step(&grads_of(&p, &[0.5, -0.2])).unwrap();
        let p1 = p.to_vec::<f32>().unwrap();
        assert!((p1[0] - 0.85).abs() < 1e-6 && (p1[1] - (-1.80)).abs() < 1e-6, "p1={p1:?}");
        let m1 = opt.exp_avg(0, 0).unwrap().to_vec::<f32>().unwrap();
        assert!((m1[0] - 0.005).abs() < 1e-7 && (m1[1] - (-0.002)).abs() < 1e-7, "m1={m1:?}");

        opt.step(&grads_of(&p, &[-0.01, 0.0])).unwrap();
        let p2 = p.to_vec::<f32>().unwrap();
        assert!((p2[0] - 0.7075).abs() < 1e-6 && (p2[1] - (-1.61)).abs() < 1e-6, "p2={p2:?}");
        let m2 = opt.exp_avg(0, 0).unwrap().to_vec::<f32>().unwrap();
        assert!((m2[0] - 0.00485).abs() < 1e-7 && (m2[1] - (-0.00198)).abs() < 1e-7, "m2={m2:?}");
        assert_eq!(opt.step_count(), 2);

        // Bitwise parity between contiguous CPU fast path and non-contiguous transposed generic path
        // (including zero, negative zero, positive, negative, ±Inf, and NaN gradients).
        let init = [1.0f32, -2.0, 0.5, 3.0, -1.5, 2.5];
        let fast = Tensor::from_slice(&init, [2, 3]).unwrap().requires_grad_();
        let base = Tensor::zeros([3, 2]);
        let slow_t = base.t().unwrap();
        {
            let _g = no_grad();
            slow_t.copy_from(&fast.detach()).unwrap();
        }
        let slow = slow_t.requires_grad_();
        assert!(!slow.is_contiguous());
        let mut opt_f = Lion::from_params(vec![("w".into(), fast.clone())], 0.05, 0.9, 0.99, 0.1).unwrap();
        let mut opt_s = Lion::from_params(vec![("w".into(), slow.clone())], 0.05, 0.9, 0.99, 0.1).unwrap();
        for gv in [[0.3f32, -0.4, 0.0, -0.0, 0.2, -0.1], [-0.5, 0.2, 0.0, 0.1, f32::INFINITY, f32::NAN]] {
            opt_f.step(&grads_of(&fast, &gv)).unwrap();
            opt_s.step(&grads_of(&slow, &gv)).unwrap();
            for (a, b) in fast.to_vec::<f32>().unwrap().iter().zip(slow.to_vec::<f32>().unwrap()) {
                if a.is_nan() {
                    assert!(b.is_nan(), "NaN mismatch: {a} vs {b}");
                } else {
                    assert_eq!(a.to_bits(), b.to_bits(), "bit mismatch: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn lion_converges_quadratic_rosenbrock_and_module() {
        // 1. Quadratic bowl: min (x - 3)^2 + 4*(y + 2)^2 with cosine LR decay (damping sign-step oscillation).
        let p = Tensor::from_slice(&[-4.0f32, 5.0], [2]).unwrap().requires_grad_();
        let mut opt = Lion::from_params(vec![("p".into(), p.clone())], 0.08, 0.9, 0.95, 0.0).unwrap();
        let mut q_sched =
            crate::LrSchedule::new(crate::DecayKind::Cosine, 1, 300, 0.01, 0.0, 0.0, 0.9).unwrap();
        let target = Tensor::from_slice(&[3.0f32, -2.0], [2]).unwrap();
        let scale = Tensor::from_slice(&[1.0f32, 4.0], [2]).unwrap();
        let mut init_loss = 0.0f32;
        let mut final_loss = 0.0f32;
        for step in 0..300 {
            let diff = p.sub(&target).unwrap();
            let loss = diff.square().unwrap().mul(&scale).unwrap().sum_all().unwrap();
            let v = loss.item::<f32>().unwrap();
            if step == 0 {
                init_loss = v;
            }
            final_loss = v;
            let grads = loss.backward().unwrap();
            q_sched.apply_to_groups(opt.groups_mut());
            opt.step(&grads).unwrap();
            q_sched.advance();
        }
        assert!(final_loss < 0.02 && final_loss < init_loss * 1e-4, "quadratic: {init_loss} -> {final_loss}");

        // 2. 2D Rosenbrock banana: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2 starting at (-0.5, 0.5).
        let x = Tensor::from_slice(&[-0.5f32], [1]).unwrap().requires_grad_();
        let y = Tensor::from_slice(&[0.5f32], [1]).unwrap().requires_grad_();
        let mut opt_r =
            Lion::from_params(vec![("x".into(), x.clone()), ("y".into(), y.clone())], 5e-3, 0.9, 0.99, 0.0)
                .unwrap();
        let eval_rosen = |x: &Tensor, y: &Tensor| -> Tensor {
            let t1 = x.rsub_scalar(1.0).unwrap().square().unwrap();
            let t2 = y.sub(&x.square().unwrap()).unwrap().square().unwrap().mul_scalar(100.0).unwrap();
            t1.add(&t2).unwrap().sum_all().unwrap()
        };
        let r0 = eval_rosen(&x, &y).item::<f32>().unwrap();
        for _ in 0..400 {
            let loss = eval_rosen(&x, &y);
            let grads = loss.backward().unwrap();
            opt_r.step(&grads).unwrap();
        }
        let r1 = eval_rosen(&x, &y).item::<f32>().unwrap();
        assert!(r1 < r0 * 0.05, "Rosenbrock must descend cleanly: {r0} -> {r1}");

        // 3. Small folo-nn module (ConvBnAct + Linear) regression convergence with StandardGroups + LrSchedule.
        let mut rng = Rng::seed(42);
        let net = Sequential::new()
            .push_named("stem", ConvBnAct::new(2, 4, 3, 1, 1, Activation::Silu, &mut rng))
            .push_named("proj", Linear::new(4, 2, true, &mut rng));
        let sg = super::super::StandardGroups::from_module(&net);
        let groups: Vec<ParamGroup> = sg
            .into_groups(1e-2, LION_DEFAULT_BETA1, 1e-2)
            .into_iter()
            .map(|g| g.lion(LION_DEFAULT_BETA1, LION_DEFAULT_BETA2))
            .collect();
        let mut opt_m = Lion::new(groups).unwrap();
        let mut sched = crate::LrSchedule::new(crate::DecayKind::Cosine, 1, 80, 0.05, 0.0, 0.0, 0.9).unwrap();
        let inp = Tensor::randn([2, 2, 4, 4], 0.0, 1.0, &mut rng);
        let tgt = Tensor::randn([2, 2], 0.0, 1.0, &mut rng);
        let forward_loss = |m: &Sequential| -> Tensor {
            let h = m.get(0).unwrap().forward(&inp).unwrap().mean(3, false).unwrap().mean(2, false).unwrap();
            let pred = m.get(1).unwrap().forward(&h).unwrap();
            pred.sub(&tgt).unwrap().square().unwrap().mean_all().unwrap()
        };
        let l0 = forward_loss(&net).item::<f32>().unwrap();
        for _ in 0..80 {
            let loss = forward_loss(&net);
            let grads = loss.backward().unwrap();
            sched.apply_to_groups(opt_m.groups_mut());
            opt_m.step(&grads).unwrap();
            sched.advance();
        }
        let l1 = forward_loss(&net).item::<f32>().unwrap();
        assert!(l1 < l0 * 0.2, "module Lion training must converge: {l0} -> {l1}");
    }

    #[test]
    fn lion_edge_cases_validation_and_checkpoint_resume() {
        // 1. Zero grad leaves zero-momentum parameter untouched (sign(0) == 0, NOT ±1).
        let p = Tensor::from_slice(&[2.0f32, -3.0], [2]).unwrap().requires_grad_();
        let mut opt = Lion::from_params(vec![("p".into(), p.clone())], 0.1, 0.9, 0.99, 0.0).unwrap();
        opt.step(&grads_of(&p, &[0.0, -0.0])).unwrap();
        assert_eq!(p.to_vec::<f32>().unwrap(), [2.0, -3.0]);
        assert_eq!(opt.exp_avg(0, 0).unwrap().to_vec::<f32>().unwrap(), [0.0, 0.0]);

        // 2. Non-finite NaN gradient poisons both parameter and exp_avg (never masked to 0).
        let p_nan = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap().requires_grad_();
        let mut opt_nan = Lion::from_params(vec![("p".into(), p_nan.clone())], 0.01, 0.9, 0.99, 0.1).unwrap();
        opt_nan.step(&grads_of(&p_nan, &[f32::NAN, 0.5])).unwrap();
        let pv = p_nan.to_vec::<f32>().unwrap();
        let mv = opt_nan.exp_avg(0, 0).unwrap().to_vec::<f32>().unwrap();
        assert!(pv[0].is_nan() && pv[1].is_finite());
        assert!(mv[0].is_nan() && mv[1].is_finite());

        // 3. Missing gradient skips parameter completely (no weight decay, no state update).
        let pa = Tensor::from_slice(&[4.0f32], [1]).unwrap().requires_grad_();
        let pb = Tensor::from_slice(&[5.0f32], [1]).unwrap().requires_grad_();
        let mut opt_skip =
            Lion::from_params(vec![("a".into(), pa.clone()), ("b".into(), pb.clone())], 0.1, 0.9, 0.99, 0.5)
                .unwrap();
        opt_skip.step(&grads_of(&pa, &[1.0])).unwrap();
        assert_eq!(pb.to_vec::<f32>().unwrap(), [5.0]);
        assert_eq!(opt_skip.exp_avg(0, 1).unwrap().to_vec::<f32>().unwrap(), [0.0]);

        // 4. Hyperparameter validation rejects invalid lr, beta1, beta2, weight_decay, and cautious=true.
        let q = Tensor::zeros([2]).requires_grad_();
        assert!(Lion::from_params(vec![("q".into(), q.clone())], -1e-4, 0.9, 0.99, 0.0).is_err());
        assert!(Lion::from_params(vec![("q".into(), q.clone())], f64::NAN, 0.9, 0.99, 0.0).is_err());
        assert!(Lion::from_params(vec![("q".into(), q.clone())], 1e-4, 1.0, 0.99, 0.0).is_err());
        assert!(Lion::from_params(vec![("q".into(), q.clone())], 1e-4, -0.1, 0.99, 0.0).is_err());
        assert!(Lion::from_params(vec![("q".into(), q.clone())], 1e-4, 0.9, 1.0, 0.0).is_err());
        assert!(Lion::from_params(vec![("q".into(), q.clone())], 1e-4, 0.9, 0.99, -0.01).is_err());
        assert!(Lion::new(vec![ParamGroup::new("g", vec![("q".into(), q.clone())]).cautious(true)]).is_err());

        // 5. State serialization, contract JSON round-trip, step-0 empty load, and bit-exact resume.
        let p_a = Tensor::from_slice(&[1.5f32, -0.5, 2.0], [3]).unwrap().requires_grad_();
        let p_b = Tensor::from_slice(&[1.5f32, -0.5, 2.0], [3]).unwrap().requires_grad_();
        let mut opt_a = Lion::from_params(vec![("w".into(), p_a.clone())], 0.05, 0.9, 0.99, 0.2).unwrap();
        // Step-0 load with empty state map succeeds and resets to step 0.
        opt_a.load_state(0, &HashMap::new()).unwrap();
        assert_eq!(opt_a.step_count(), 0);
        // Step > 0 load with missing state errors.
        assert!(opt_a.load_state(3, &HashMap::new()).is_err());

        opt_a.step(&grads_of(&p_a, &[0.4, -0.8, 0.2])).unwrap();
        opt_a.step(&grads_of(&p_a, &[-0.1, 0.3, -0.5])).unwrap();
        let contract_json = serde_json::to_string(&opt_a.contract()).unwrap();
        let parsed: LionContract = serde_json::from_str(&contract_json).unwrap();
        assert_eq!(parsed, opt_a.contract());
        let saved_state: HashMap<String, Tensor> = opt_a.state_tensors().unwrap().into_iter().collect();

        // Resume into fresh optimizer + copy weights, then step both identically and verify bitwise match.
        {
            let _g = no_grad();
            p_b.copy_from(&p_a.detach()).unwrap();
        }
        let mut opt_b = Lion::from_params(vec![("w".into(), p_b.clone())], 0.99, 0.5, 0.5, 0.0).unwrap();
        opt_b.apply_contract(&parsed).unwrap();
        opt_b.load_state(parsed.step, &saved_state).unwrap();
        assert_eq!(opt_b.step_count(), 2);

        for g in [[0.2f32, 0.1, -0.7], [-0.6, -0.2, 0.4]] {
            opt_a.step(&grads_of(&p_a, &g)).unwrap();
            opt_b.step(&grads_of(&p_b, &g)).unwrap();
        }
        assert_eq!(p_a.to_vec::<f32>().unwrap(), p_b.to_vec::<f32>().unwrap());
        assert_eq!(
            opt_a.exp_avg(0, 0).unwrap().to_vec::<f32>().unwrap(),
            opt_b.exp_avg(0, 0).unwrap().to_vec::<f32>().unwrap()
        );
    }
}
