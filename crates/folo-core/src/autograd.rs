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
use std::collections::HashMap;
use std::collections::hash_map::Entry;
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

/// Sentinel slot for an op input that does not require grad.
const NO_GRAD_SLOT: usize = usize::MAX;

/// Reachable autograd graph indexed by dense slots `0..nodes.len()`, built once before backward so the reverse
/// walk accumulates into a flat `Vec<Option<Tensor>>` with no `HashMap` lookups.
struct Graph {
    /// Reachable tensors that require grad (`nodes[0]` is the backward root).
    nodes: Vec<Tensor>,
    /// For each slot `u`, the dense slot of each input of `nodes[u].node()`, or [`NO_GRAD_SLOT`] if that input
    /// did not require grad (empty for leaves).
    child_slots: Vec<SmallVec<[usize; 2]>>,
    /// Post-order of dense slot indices (inputs before ops that use them; `0` is last).
    order: Vec<usize>,
}

/// Post-order of the graph reachable from `root` (inputs before the ops that use them).
fn topo_order(root: &Tensor) -> Graph {
    let mut nodes: Vec<Tensor> = vec![root.clone()];
    let mut child_slots: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new()];
    let mut order: Vec<usize> = Vec::new();
    let mut ids: HashMap<TensorId, usize> = HashMap::new();
    ids.insert(root.id(), 0);

    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    while let Some((u, i)) = stack.pop() {
        let child_info = nodes[u].node().and_then(|n| n.inputs.get(i)).map(|c| (c.id(), c.requires_grad()));
        match child_info {
            Some((cid, req)) => {
                stack.push((u, i + 1));
                if req {
                    let v = match ids.entry(cid) {
                        Entry::Occupied(e) => *e.get(),
                        Entry::Vacant(e) => {
                            let idx = nodes.len();
                            let child = nodes[u].node().expect("node present above").inputs[i].clone();
                            e.insert(idx);
                            nodes.push(child);
                            child_slots.push(SmallVec::new());
                            stack.push((idx, 0));
                            idx
                        }
                    };
                    child_slots[u].push(v);
                } else {
                    child_slots[u].push(NO_GRAD_SLOT);
                }
            }
            None => order.push(u),
        }
    }
    Graph { nodes, child_slots, order }
}

/// `acc + g`, in place when nothing else can observe `acc` (a fresh, contiguous gradient that no node saved and
/// no other gradient shares), which saves an allocation and a pass over memory per fan-in.
fn accumulate(acc: Tensor, g: &Tensor) -> Result<Tensor> {
    let dense = acc.is_contiguous() && acc.cell().read().len() == acc.numel();
    if acc.is_exclusive() && dense && acc.dtype() == crate::DType::F32 && !acc.same_storage(g) {
        acc.add_assign(g)?;
        return Ok(acc);
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
    let graph = topo_order(root);
    let mut pending: Vec<Option<Tensor>> = vec![None; graph.nodes.len()];
    pending[0] = Some(seed);
    let mut out = Grads::default();
    for &u in graph.order.iter().rev() {
        let Some(g) = pending[u].take() else { continue };
        let t = &graph.nodes[u];
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
        for ((inp, &inp_slot), gi) in node.inputs.iter().zip(graph.child_slots[u].iter()).zip(gins) {
            let Some(gi) = gi else { continue };
            if inp_slot == NO_GRAD_SLOT || !inp.requires_grad() {
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
            match pending[inp_slot].take() {
                Some(acc) => {
                    pending[inp_slot] = Some(accumulate(acc, &gi)?);
                }
                None => {
                    pending[inp_slot] = Some(gi);
                }
            }
        }
    }
    Ok(out)
}
