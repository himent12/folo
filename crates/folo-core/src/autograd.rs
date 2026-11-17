//! Define-by-run reverse-mode autograd.
//!
//! When grad mode is on ([`is_grad_enabled`]) and an op has an input that requires grad, the op's result
//! carries a node: its inputs, the storage versions of those inputs, and a backward closure mapping the
//! output gradient to input gradients. [`Tensor::backward`] walks the graph in reverse topological order and
//! returns [`Grads`] for the leaf tensors that require grad.
//!
//! Backward closures are built from ordinary tensor ops run under [`no_grad`], so they work on any backend that
//! implements the forward and `*_backward` kernels.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use std::sync::Arc;

use smallvec::SmallVec;

use crate::storage::StorageCell;
use crate::tensor::TensorId;
use crate::{Error, Result, Tensor};

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// True unless a [`no_grad`] guard is alive on this thread.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// RAII guard returned by [`no_grad`]; restores the previous grad mode when dropped. Not `Send`: grad mode is
/// per thread.
#[must_use = "grad recording is disabled only while the guard is alive"]
pub struct NoGradGuard {
    prev: bool,
    _not_send: PhantomData<*const ()>,
}

/// Disable graph recording on this thread until the returned guard is dropped (inference, optimizer updates).
///
/// ```
/// use folo_core::{no_grad, Tensor};
/// let w = Tensor::ones([3]).requires_grad_();
/// let y = {
///     let _g = no_grad();
///     (&w * 2.0).unwrap()
/// };
/// assert!(!y.requires_grad());
/// ```
pub fn no_grad() -> NoGradGuard {
    let prev = GRAD_ENABLED.with(|g| g.replace(false));
    NoGradGuard { prev, _not_send: PhantomData }
}

/// Set this thread's grad mode to `enabled` until the returned guard is dropped (carries a caller's mode
/// onto another thread, see [`crate::cpu::in_pool`]).
pub(crate) fn grad_mode(enabled: bool) -> NoGradGuard {
    let prev = GRAD_ENABLED.with(|g| g.replace(enabled));
    NoGradGuard { prev, _not_send: PhantomData }
}

/// Run `f` with grad recording disabled.
pub fn with_no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _g = no_grad();
    f()
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.prev));
    }
}

/// Maps the gradient of an op's output to the gradients of its inputs (`None` = no gradient for that input).
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Result<SmallVec<[Option<Tensor>; 2]>> + Send + Sync>;

/// One recorded op.
pub(crate) struct Node {
    pub op: &'static str,
    pub inputs: SmallVec<[Tensor; 2]>,
    pub versions: SmallVec<[u64; 2]>,
    /// Other tensors the backward closure reads (indices, masks, the op's own output), with their versions.
    pub saved: SmallVec<[(Arc<StorageCell>, u64); 1]>,
    pub backward: BackwardFn,
}

impl Node {
    /// Drop the closure, then hand the inputs to `stack` (for the iterative drop in `TensorInner`).
    pub fn into_inputs(self, stack: &mut Vec<Tensor>) {
        let Node { inputs, backward, .. } = self;
        drop(backward);
        stack.extend(inputs);
    }
}

/// Gradients from [`Tensor::backward`], keyed by tensor identity.
#[derive(Default)]
pub struct Grads(HashMap<TensorId, Tensor>);

impl Grads {
    /// The gradient of `t`, if it was reached.
    pub fn get(&self, t: &Tensor) -> Option<&Tensor> {
        self.0.get(&t.id())
    }

    /// Remove and return the gradient of `t`.
    pub fn remove(&mut self, t: &Tensor) -> Option<Tensor> {
        self.0.remove(&t.id())
    }

    /// Set the gradient of `t` (e.g. after clipping).
    pub fn insert(&mut self, t: &Tensor, grad: Tensor) {
        self.0.insert(t.id(), grad);
    }

    /// Number of tensors with a gradient.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True if no gradient was produced.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// All `(id, gradient)` pairs, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (TensorId, &Tensor)> {
        self.0.iter().map(|(&k, v)| (k, v))
    }
}

impl std::fmt::Debug for Grads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.0.iter().map(|(k, v)| (k, v.shape()))).finish()
    }
}

/// Post-order of the graph reachable from `root` (inputs before the ops that use them).
fn topo_order(root: &Tensor) -> Vec<Tensor> {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<(Tensor, usize)> = vec![(root.clone(), 0)];
    seen.insert(root.id());
    while let Some((t, i)) = stack.pop() {
        let next = t.node().and_then(|n| n.inputs.get(i).cloned());
        match next {
            Some(child) => {
                stack.push((t, i + 1));
                if child.requires_grad() && seen.insert(child.id()) {
                    stack.push((child, 0));
                }
            }
            None => order.push(t),
        }
    }
    order
}

/// `acc + g`, in place when nothing else can observe `acc` (a fresh, contiguous gradient that no node saved and
/// no other gradient shares), which saves an allocation and a pass over memory per fan-in: through the backend's
/// in-place add (any float dtype; one pass on the GPU), else `add_assign` (`F32`). Same rounding either way.
fn accumulate(acc: Tensor, g: &Tensor) -> Result<Tensor> {
    let dense = acc.is_contiguous() && acc.cell().read().len() == acc.numel();
    if acc.is_exclusive() && dense && !acc.same_storage(g) {
        // Not into a gradient that itself records (a custom backward's): the out-of-place sum drops the flag.
        if !acc.device().is_cpu() && !acc.requires_grad() && acc.dtype() == g.dtype() && acc.add_into_(g)? {
            return Ok(acc);
        }
        if acc.dtype() == crate::DType::F32 {
            acc.add_assign(g)?;
            return Ok(acc);
        }
    }
    acc.add(g)
}

pub(crate) fn backward(root: &Tensor, seed: Tensor) -> Result<Grads> {
    if !root.requires_grad() {
        return Err(Error::Autograd("backward() on a tensor that does not require grad".into()));
    }
    if seed.shape() != root.shape() {
        return Err(Error::ShapeMismatch {
            op: "backward",
            lhs: root.shape().clone(),
            rhs: seed.shape().clone(),
        });
    }
    let _guard = no_grad();
    let order = topo_order(root);
    let mut pending: HashMap<TensorId, Tensor> = HashMap::new();
    pending.insert(root.id(), seed);
    let mut out = Grads::default();
    for t in order.iter().rev() {
        let Some(g) = pending.remove(&t.id()) else { continue };
        let Some(node) = t.node() else {
            out.0.insert(t.id(), g);
            continue;
        };
        let recorded = node.inputs.iter().map(|t| t.cell()).zip(node.versions.iter().copied());
        for (cell, v) in recorded.chain(node.saved.iter().map(|(c, v)| (c, *v))) {
            let cur = cell.version();
            if cur != v {
                return Err(Error::ModifiedInPlace { op: node.op, recorded: v, current: cur });
            }
        }
        let gins = (node.backward)(&g)?;
        if gins.len() != node.inputs.len() {
            return Err(Error::Autograd(format!(
                "backward of `{}` returned {} gradients for {} inputs",
                node.op,
                gins.len(),
                node.inputs.len()
            )));
        }
        for (inp, gi) in node.inputs.iter().zip(gins) {
            let Some(gi) = gi else { continue };
            if !inp.requires_grad() {
                continue;
            }
            if gi.shape() != inp.shape() || gi.dtype() != inp.dtype() || gi.device() != inp.device() {
                return Err(Error::Autograd(format!(
                    "backward of `{}` produced a gradient [{} {} {}] for an input [{} {} {}]",
                    node.op,
                    gi.shape(),
                    gi.dtype(),
                    gi.device(),
                    inp.shape(),
                    inp.dtype(),
                    inp.device()
                )));
            }
            match pending.remove(&inp.id()) {
                Some(acc) => {
                    pending.insert(inp.id(), accumulate(acc, &gi)?);
                }
                None => {
                    pending.insert(inp.id(), gi);
                }
            }
        }
    }
    Ok(out)
}
