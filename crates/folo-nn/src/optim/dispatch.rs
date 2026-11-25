//! Optimizer step dispatch helpers: O(1) storage identity, zero-allocation CPU parameter scheduling, and
//! multi-group GPU coalescing for [`Tensor::optimizer_step_many_`].

use std::collections::HashSet;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor};

use super::{GradSource, ParamGroup};
use crate::Result;

/// Stable storage identity key while `t` is alive (two live tensors share storage iff their keys match).
///
/// Matches [`Tensor::same_storage`] in O(1) so multi-tensor alias checks stay O(N) instead of O(N²).
#[inline]
pub(crate) fn storage_key(t: &Tensor) -> usize {
    t.with_view(|v| v.storage as *const folo_core::Storage as usize)
}

/// Above this many parameter elements in one step, parameters are updated in parallel.
pub(crate) const PAR_THRESHOLD: usize = 1 << 16;

/// Reusable flat `(group_index, param_index)` schedule so steady-state CPU optimizer steps perform zero heap
/// allocations even when parallelizing across parameter groups.
pub(crate) struct ParamPlan {
    group_lens: Vec<usize>,
    items: Vec<(usize, usize)>,
}

impl ParamPlan {
    pub(crate) fn from_groups(groups: &[ParamGroup]) -> Self {
        let mut plan = ParamPlan { group_lens: Vec::new(), items: Vec::new() };
        plan.sync(groups);
        plan
    }

    pub(crate) fn sync(&mut self, groups: &[ParamGroup]) {
        if self.group_lens.len() == groups.len()
            && self.group_lens.iter().zip(groups).all(|(&len, g)| len == g.params.len())
        {
            return;
        }
        self.group_lens.clear();
        self.group_lens.extend(groups.iter().map(|g| g.params.len()));
        let total_params: usize = self.group_lens.iter().sum();
        self.items.clear();
        self.items.reserve(total_params);
        for (gi, g) in groups.iter().enumerate() {
            for pi in 0..g.params.len() {
                self.items.push((gi, pi));
            }
        }
    }
}

/// Run `f` for every `(group index, param index)` in `plan`, in parallel when the step is large. `f` runs under
/// `no_grad` on whichever thread executes it.
pub(crate) fn for_each_param_planned<F>(plan: &ParamPlan, groups: &[ParamGroup], f: F) -> Result<()>
where
    F: Fn(usize, usize) -> Result<()> + Sync,
{
    use rayon::prelude::*;
    let total: usize = groups.iter().flat_map(|g| &g.params).map(|(_, t)| t.numel()).sum();
    if total < PAR_THRESHOLD {
        let _g = folo_core::no_grad();
        for &(gi, pi) in &plan.items {
            f(gi, pi)?;
        }
        return Ok(());
    }
    folo_core::cpu::in_pool(|| {
        plan.items.par_iter().try_for_each(|&(gi, pi)| {
            let _g = folo_core::no_grad();
            f(gi, pi)
        })
    })
}

/// Borrowed-or-converted `F32` gradient tensor to avoid cloning `Tensor` handles in the common `F32` case.
pub(crate) enum GradRef<'a> {
    Borrowed(&'a Tensor),
    Owned(Tensor),
}

impl<'a> GradRef<'a> {
    #[inline]
    pub(crate) fn as_f32(g: &'a Tensor) -> Result<Self> {
        if g.dtype() == DType::F32 { Ok(Self::Borrowed(g)) } else { Ok(Self::Owned(g.to_dtype(DType::F32)?)) }
    }

    #[inline]
    pub(crate) fn as_tensor(&self) -> &Tensor {
        match self {
            Self::Borrowed(t) => t,
            Self::Owned(t) => t,
        }
    }
}

/// Bitwise equality on [`OptimStep`] so GPU parameter groups with identical hyperparameters (for example `bn`
/// and `bias` after warmup) can be coalesced into one [`Tensor::optimizer_step_many_`] call.
fn optim_step_bits_eq(a: &OptimStep, b: &OptimStep) -> bool {
    match (a, b) {
        (
            OptimStep::Sgd { lr: a_lr, momentum: a_m, weight_decay: a_wd, nesterov: a_n },
            OptimStep::Sgd { lr: b_lr, momentum: b_m, weight_decay: b_wd, nesterov: b_n },
        ) => {
            a_lr.to_bits() == b_lr.to_bits()
                && a_m.to_bits() == b_m.to_bits()
                && a_wd.to_bits() == b_wd.to_bits()
                && a_n == b_n
        }
        (
            OptimStep::SgdDecoupled { lr: a_lr, momentum: a_m, wd_step: a_wd, nesterov: a_n, cautious: a_c },
            OptimStep::SgdDecoupled { lr: b_lr, momentum: b_m, wd_step: b_wd, nesterov: b_n, cautious: b_c },
        ) => {
            a_lr.to_bits() == b_lr.to_bits()
                && a_m.to_bits() == b_m.to_bits()
                && a_wd.to_bits() == b_wd.to_bits()
                && a_n == b_n
                && a_c == b_c
        }
        (
            OptimStep::AdamW {
                decay: a_d,
                beta1: a_b1,
                one_minus_beta1: a_ob1,
                beta2: a_b2,
                one_minus_beta2: a_ob2,
                step_size: a_ss,
                bc2_sqrt: a_bc2,
                eps: a_eps,
            },
            OptimStep::AdamW {
                decay: b_d,
                beta1: b_b1,
                one_minus_beta1: b_ob1,
                beta2: b_b2,
                one_minus_beta2: b_ob2,
                step_size: b_ss,
                bc2_sqrt: b_bc2,
                eps: b_eps,
            },
        ) => {
            a_d.map(f32::to_bits) == b_d.map(f32::to_bits)
                && a_b1.to_bits() == b_b1.to_bits()
                && a_ob1.to_bits() == b_ob1.to_bits()
                && a_b2.to_bits() == b_b2.to_bits()
                && a_ob2.to_bits() == b_ob2.to_bits()
                && a_ss.to_bits() == b_ss.to_bits()
                && a_bc2.to_bits() == b_bc2.to_bits()
                && a_eps.to_bits() == b_eps.to_bits()
        }
        (OptimStep::Axpby { a: a1, b: b1 }, OptimStep::Axpby { a: a2, b: b2 }) => {
            a1.to_bits() == a2.to_bits() && b1.to_bits() == b2.to_bits()
        }
        _ => false,
    }
}

/// The multi-tensor device path of an optimizer step: per group with parameters off the CPU, every parameter that
/// has a gradient is updated by [`Tensor::optimizer_step_many_`] with `step(group)` and the state tensors
/// `state(group, param)` (the GPU updates 16 parameters per dispatch, against one each). Non-CPU groups sharing
/// bitwise-identical [`OptimStep`], device, and state arity (such as `bn` and `bias` after warmup) are coalesced
/// into a single batched call when their storage buffers do not alias across groups.
///
/// Returns `None` without allocating when every parameter is on the CPU, else per-group completion flags; the
/// caller updates any remaining groups one parameter at a time. `what` names the optimizer in errors.
pub(crate) fn step_many<'a>(
    what: &str,
    groups: &'a [ParamGroup],
    grads: &'a dyn GradSource,
    step: impl Fn(usize) -> OptimStep,
    state: impl Fn(usize, usize) -> Vec<&'a Tensor>,
) -> Result<Option<Vec<bool>>> {
    if groups.iter().all(|grp| grp.params.iter().all(|(_, p)| p.device().is_cpu())) {
        return Ok(None);
    }
    let _g = folo_core::no_grad();
    struct GroupBatch<'a> {
        gi: usize,
        op: OptimStep,
        uniform_dev: Option<Device>,
        uniform_states: Option<usize>,
        ps: Vec<&'a Tensor>,
        gs: Vec<GradRef<'a>>,
        ss: Vec<Vec<&'a Tensor>>,
    }

    let mut done = vec![false; groups.len()];
    let mut batches: Vec<GroupBatch<'a>> = Vec::new();
    for (gi, grp) in groups.iter().enumerate() {
        if grp.params.iter().all(|(_, p)| p.device().is_cpu()) {
            continue;
        }
        let (mut ps, mut gs, mut ss) = (Vec::new(), Vec::new(), Vec::new());
        let mut uniform_dev: Option<Device> = None;
        let mut mixed_dev = false;
        let mut uniform_states: Option<usize> = None;
        let mut mixed_states = false;
        for (pi, (name, p)) in grp.params.iter().enumerate() {
            let Some(g) = grads.grad(p) else { continue };
            if g.shape() != p.shape() {
                return Err(crate::Error::Invalid(format!(
                    "{what}: gradient of `{name}` is {:?}, parameter is {:?}",
                    g.dims(),
                    p.dims()
                )));
            }
            let dev = p.device();
            if dev.is_cpu() {
                mixed_dev = true;
            } else {
                match uniform_dev {
                    None => uniform_dev = Some(dev),
                    Some(d) if d != dev => mixed_dev = true,
                    Some(_) => {}
                }
            }
            let st = state(gi, pi);
            match uniform_states {
                None => uniform_states = Some(st.len()),
                Some(n) if n != st.len() => mixed_states = true,
                Some(_) => {}
            }
            ps.push(p);
            gs.push(GradRef::as_f32(g)?);
            ss.push(st);
        }
        if ps.is_empty() {
            done[gi] = true;
            continue;
        }
        batches.push(GroupBatch {
            gi,
            op: step(gi),
            uniform_dev: if mixed_dev { None } else { uniform_dev },
            uniform_states: if mixed_states { None } else { uniform_states },
            ps,
            gs,
            ss,
        });
    }

    // Check in O(N) whether all non-CPU batches have pairwise-distinct mutated storages (params + states)
    // and no incoming gradient aliases any mutated storage across batches. When true, group order is
    // unobservable and batches with identical (OptimStep, Device, state arity) can be coalesced safely;
    // otherwise preserve strict canonical group order.
    let total_items: usize = batches.iter().map(|b| b.ps.len()).sum();
    let mut all_written: HashSet<usize> = HashSet::with_capacity(total_items * 3);
    let mut batches_disjoint = batches.len() >= 2;
    if batches_disjoint {
        'check: for b in &batches {
            for (p, st) in b.ps.iter().zip(&b.ss) {
                if !all_written.insert(storage_key(p)) {
                    batches_disjoint = false;
                    break 'check;
                }
                for s in st {
                    if !all_written.insert(storage_key(s)) {
                        batches_disjoint = false;
                        break 'check;
                    }
                }
            }
        }
    }
    if batches_disjoint {
        'gcheck: for b in &batches {
            for g in &b.gs {
                if all_written.contains(&storage_key(g.as_tensor())) {
                    batches_disjoint = false;
                    break 'gcheck;
                }
            }
        }
    }

    if !batches_disjoint {
        for b in &batches {
            let gr: Vec<&Tensor> = b.gs.iter().map(GradRef::as_tensor).collect();
            done[b.gi] = Tensor::optimizer_step_many_(&b.ps, &gr, &b.ss, &b.op)?;
        }
        return Ok(Some(done));
    }

    // Group non-CPU batches with identical (OptimStep, Device, state arity) into coalesced buckets.
    let mut visited = vec![false; batches.len()];
    for i in 0..batches.len() {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let mut bucket = vec![i];
        if let (Some(dev_i), Some(st_i)) = (batches[i].uniform_dev, batches[i].uniform_states) {
            for j in (i + 1)..batches.len() {
                if !visited[j]
                    && batches[j].uniform_dev == Some(dev_i)
                    && batches[j].uniform_states == Some(st_i)
                    && optim_step_bits_eq(&batches[i].op, &batches[j].op)
                {
                    visited[j] = true;
                    bucket.push(j);
                }
            }
        }
        let mut merged_ok = false;
        if bucket.len() >= 2 {
            let bucket_items: usize = bucket.iter().map(|&bi| batches[bi].ps.len()).sum();
            let mut ps = Vec::with_capacity(bucket_items);
            let mut gr = Vec::with_capacity(bucket_items);
            let mut ss = Vec::with_capacity(bucket_items);
            for &bi in &bucket {
                ps.extend_from_slice(&batches[bi].ps);
                gr.extend(batches[bi].gs.iter().map(GradRef::as_tensor));
                ss.extend(batches[bi].ss.iter().cloned());
            }
            if Tensor::optimizer_step_many_(&ps, &gr, &ss, &batches[i].op)? {
                for &bi in &bucket {
                    done[batches[bi].gi] = true;
                }
                merged_ok = true;
            }
        }
        if !merged_ok {
            for &bi in &bucket {
                let b = &batches[bi];
                let gr: Vec<&Tensor> = b.gs.iter().map(GradRef::as_tensor).collect();
                done[b.gi] = Tensor::optimizer_step_many_(&b.ps, &gr, &b.ss, &b.op)?;
            }
        }
    }
    Ok(Some(done))
}

/// True if every tensor is a contiguous CPU `F32` tensor (the fused-kernel fast path applies).
pub(crate) fn cpu_fast(ts: &[&Tensor]) -> bool {
    ts.iter().all(|t| t.device().is_cpu() && t.dtype() == DType::F32 && t.is_contiguous())
}
