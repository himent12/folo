//! [`ForwardPlans`]: a GPU call of one input tensor over a fixed set of tensors (a model's inference forward),
//! recorded on its second call per key and replayed from then on (see `context/plan.rs`).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use folo_core::{Result, Storage, Tensor, autocast_dtype, is_grad_enabled};

use super::convert::Core;
use super::{GpuStorage, context_for, contexts, dense_buffer, tensor_from_buffer};
use crate::context::{GpuContext, Plan, PlanOutput};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Calls its argument with each tensor a planned call reads besides its input ([`ForwardPlans::run`]).
pub type Tensors<'a> = dyn Fn(&mut dyn FnMut(&Tensor)) + 'a;

/// The recorded plans of one owner (a model): [`ForwardPlans::run`] replays a call it has seen before without the
/// tensor-op layer, bit for bit the same as running it.
///
/// A plan is used only when every condition holds; otherwise the call runs as it is:
/// - forward plans are on for the input's context ([`GpuContext::set_forward_plans`], `FOLO_GPU_NO_PLAN`) and for
///   this owner ([`ForwardPlans::set_enabled`]);
/// - gradients are off ([`folo_core::no_grad`]) and autocast is off; the context is not profiling or tracing and
///   its device is not lost; the input is a dense GPU tensor;
/// - the plan was recorded for the same key, input shape and dtype, tensor-core switch and context plan epoch;
/// - every tensor `tensors` lists is the tensor it was at recording ([`Tensor::id`]), at the same storage version
///   ([`Tensor::storage_version`]): an in-place write, `to_device_` or `to_dtype_` of a weight, or a tensor replaced
///   by another, retires the plan.
///
/// The call must be a pure function of its input and the listed tensors: what it reads from anywhere else, an
/// upload or readback inside it, or an intermediate it keeps make it unplannable ([`ForwardPlans::refusal`] says
/// why). Results are fresh tensors on every call. Plans live in the context, at most [`crate::MAX_PLANS`] and
/// [`crate::MAX_PLAN_BYTES`] per context ([`crate::MAX_PLAN_BYTES_EACH`] each), and go when their owner is dropped or the context releases its memory.
pub struct ForwardPlans {
    id: u64,
    enabled: AtomicBool,
    replays: AtomicU64,
    refusal: Mutex<Option<String>>,
}

impl Default for ForwardPlans {
    fn default() -> Self {
        Self {
            id: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            enabled: AtomicBool::new(true),
            replays: AtomicU64::new(0),
            refusal: Mutex::new(None),
        }
    }
}

impl std::fmt::Debug for ForwardPlans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForwardPlans").field("id", &self.id).finish()
    }
}

impl Drop for ForwardPlans {
    fn drop(&mut self) {
        self.forget();
    }
}

/// The buffer behind a dense GPU tensor that is its whole storage (what a planned result is rebuilt as).
fn whole_buffer(t: &Tensor) -> Option<wgpu::Buffer> {
    dense_buffer(t).filter(|_| t.with_view(|v| v.storage.len()) == t.numel())
}

/// The buffer behind any GPU tensor.
fn storage_buffer(t: &Tensor) -> Option<wgpu::Buffer> {
    t.with_view(|v| match v.storage {
        Storage::Device(ds) => ds.downcast_ref::<GpuStorage>().map(|s| s.buf.raw().clone()),
        Storage::Cpu(_) => None,
    })
}

impl ForwardPlans {
    /// An owner with no plans.
    pub fn new() -> Self {
        Self::default()
    }

    /// Turns this owner's plans on (the default) or off; off drops them.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
        if !on {
            self.forget();
        }
    }

    /// Calls answered from a plan so far.
    pub fn replays(&self) -> u64 {
        self.replays.load(Ordering::Relaxed)
    }

    /// Drops this owner's plans from every context.
    fn forget(&self) {
        let ctxs: Vec<GpuContext> =
            contexts().lock().unwrap_or_else(|p| p.into_inner()).iter().map(|(_, c)| c.clone()).collect();
        for ctx in ctxs {
            ctx.plans().remove_owner(self.id);
        }
    }

    /// Why the last recording of this owner did not become a plan, if it did not.
    pub fn refusal(&self) -> Option<String> {
        self.refusal.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// `f(x)`, replayed from this owner's plan for `key` when there is a valid one (see the type docs), recorded into
    /// one when `key` was seen once before, or just run. `tensors` lists every tensor `f` reads besides `x` (it is
    /// called once per replay to check them, so it should not allocate); `key` distinguishes calls whose work differs
    /// otherwise (a head mode).
    pub fn run<E: From<folo_core::Error>>(
        &self,
        key: u64,
        x: &Tensor,
        tensors: &Tensors<'_>,
        f: impl FnOnce(&Tensor) -> std::result::Result<Vec<Tensor>, E>,
    ) -> std::result::Result<Vec<Tensor>, E> {
        if !self.enabled.load(Ordering::Relaxed) {
            return f(x);
        }
        let Some((ctx, input)) = eligible(x) else { return f(x) };
        let mut signature = vec![key, x.dtype() as u64, input.size(), ctx.tensor_cores() as u64];
        signature.extend(x.dims().iter().map(|&d| d as u64));
        let key = crate::context::binds::id_of(&signature);
        let found = ctx.plans().get(self.id, key);
        if let Some(plan) = found {
            if plan.signature == signature
                && plan.epoch == ctx.plan_epoch()
                && unchanged(&plan.state, tensors)
            {
                self.replays.fetch_add(1, Ordering::Relaxed);
                return Ok(replay(&ctx, &plan, x, &input)?);
            }
            ctx.plans().remove(self.id, key);
        }
        if !ctx.plans().second_sighting(self.id, key) {
            return f(x);
        }
        let Some(guard) = ctx.record_begin() else { return f(x) };
        let out = f(x);
        let rec = guard.finish();
        let out = out?;
        let plan = (|| {
            let outputs: Vec<wgpu::Buffer> = out
                .iter()
                .map(|t| whole_buffer(t).ok_or("a result is not a whole dense GPU tensor"))
                .collect::<std::result::Result<_, _>>()?;
            if (1..outputs.len()).any(|i| outputs[..i].contains(&outputs[i])) {
                return Err("two results share a buffer".to_string());
            }
            let mut state = Vec::new();
            let mut externals = Vec::new();
            tensors(&mut |t| {
                state.push((t.id(), t.storage_version()));
                externals.extend(storage_buffer(t));
            });
            let mut plan = ctx.build_plan(rec, &input, &outputs, &externals)?;
            plan.results =
                out.iter().map(|t| PlanOutput { dtype: t.dtype(), shape: t.shape().clone() }).collect();
            plan.state = state;
            plan.signature = signature;
            Ok::<Plan, String>(plan)
        })();
        let refusal = match plan {
            Ok(plan) => (!ctx.plans().insert(self.id, key, plan))
                .then(|| format!("the plan would hold more than {} bytes", crate::MAX_PLAN_BYTES_EACH)),
            Err(why) => Some(why),
        };
        if refusal.is_some() {
            ctx.plans().refuse(self.id, key);
        }
        *self.refusal.lock().unwrap_or_else(|p| p.into_inner()) = refusal;
        Ok(out)
    }
}

/// The context and input buffer of a call that may use a plan.
fn eligible(x: &Tensor) -> Option<(GpuContext, wgpu::Buffer)> {
    if x.device().is_cpu() || is_grad_enabled() || autocast_dtype().is_some() {
        return None;
    }
    let ctx = context_for(x.device())?;
    if !ctx.forward_plans() || ctx.profiling() || ctx.tracing() || ctx.is_lost() {
        return None;
    }
    let input = dense_buffer(x)?;
    Some((ctx, input))
}

/// Whether `tensors` are the tensors of `state` at the same storage versions.
fn unchanged(state: &[(folo_core::TensorId, u64)], tensors: &Tensors<'_>) -> bool {
    let mut i = 0;
    let mut same = true;
    tensors(&mut |t| {
        same &= state.get(i).is_some_and(|&(id, v)| id == t.id() && v == t.storage_version());
        i += 1;
    });
    same && i == state.len()
}

fn replay(ctx: &GpuContext, plan: &Plan, x: &Tensor, input: &wgpu::Buffer) -> Result<Vec<Tensor>> {
    let bufs =
        plan.output_sizes.iter().map(|&n| ctx.alloc(n)).collect::<std::result::Result<Vec<_>, _>>().core()?;
    let refs: Vec<&wgpu::Buffer> = bufs.iter().map(|b| b.raw()).collect();
    ctx.replay_plan(plan, input, &refs);
    plan.results
        .iter()
        .zip(bufs)
        .map(|(r, buf)| tensor_from_buffer(x.device(), r.dtype, r.shape.clone(), buf))
        .collect()
}
