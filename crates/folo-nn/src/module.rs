//! The [`Module`] trait: named parameters and buffers, train/eval mode, and submodule traversal.
//!
//! A module exposes its own tensors ([`Module::visit_tensors`]) and its direct children
//! ([`Module::visit_children`]) in canonical order (docs/MODEL.md §5: construction order). Everything else
//! (dotted names, parameter lists, counting, mode propagation, visitors) is provided on top of those two.
//!
//! Computation is a separate trait, [`Forward`], because not every module maps one tensor to one tensor (a
//! detector head returns several). [`Layer`] is any `Module + Forward`, the element type of
//! [`crate::Sequential`].

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};

use folo_core::{Device, Tensor};

use crate::Result;

/// Whether a tensor is learned (a parameter, updated by optimizers) or a buffer (state such as BatchNorm
/// running statistics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorKind {
    /// Learnable, requires grad.
    Parameter,
    /// Non-learnable state, saved in weight files.
    Buffer,
}

/// A tensor with its dotted name, as returned by [`Module::named_tensors`].
#[derive(Debug, Clone)]
pub struct NamedTensor {
    /// Dotted path, e.g. `backbone.stem.conv.weight`.
    pub name: String,
    /// The module's tensor handle (shares storage: in-place writes update the module).
    pub tensor: Tensor,
    /// Parameter or buffer.
    pub kind: TensorKind,
}

/// A train/eval flag owned by modules whose forward depends on the mode (BatchNorm). Interior mutability lets
/// [`Module::set_training`] work through `&self`, so mode changes propagate with the same visitor as names.
#[derive(Debug)]
pub struct Mode(AtomicBool);

impl Mode {
    /// A new flag in training mode (PyTorch's default).
    pub fn new() -> Self {
        Mode(AtomicBool::new(true))
    }

    /// True in training mode.
    pub fn is_training(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Set training (`true`) or evaluation (`false`) mode.
    pub fn set(&self, training: bool) {
        self.0.store(training, Ordering::Relaxed);
    }
}

impl Default for Mode {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Mode {
    fn clone(&self) -> Self {
        Mode(AtomicBool::new(self.is_training()))
    }
}

/// A node in a model tree: its tensors, its children and its mode.
///
/// Implementors provide [`visit_tensors`](Module::visit_tensors) and/or
/// [`visit_children`](Module::visit_children), and [`mode`](Module::mode) if their forward depends on it.
/// Both visitors must report in canonical order and must be deterministic: names and order are the
/// identity of a tensor in `.folo` files and in optimizer state.
///
/// ```
/// use folo_core::{rng::Rng, Tensor};
/// use folo_nn::{BatchNorm2d, Conv2d, Conv2dConfig, Module, TensorKind};
///
/// struct Block { conv: Conv2d, bn: BatchNorm2d }
/// impl Module for Block {
///     fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
///         f("conv", &self.conv);
///         f("bn", &self.bn);
///     }
/// }
///
/// let mut rng = Rng::seed(0);
/// let b = Block { conv: Conv2d::new(Conv2dConfig::new(3, 8, 3), &mut rng), bn: BatchNorm2d::new(8) };
/// let names: Vec<_> = b.named_tensors().into_iter().map(|t| t.name).collect();
/// assert_eq!(names, ["conv.weight", "conv.bias", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]);
/// assert_eq!(b.num_parameters(), 8 * 3 * 9 + 8 + 8 + 8);
/// b.eval();
/// assert!(!b.is_training());
/// ```
pub trait Module: Any + Send + Sync {
    /// Call `f` for each tensor this module owns directly (not its children's), in canonical order, with its
    /// local name (`weight`, `running_mean`, ...).
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        let _ = f;
    }

    /// Call `f` for each direct child, in canonical order, with its local name. An empty name makes the child
    /// transparent: its tensors appear directly under this module's prefix.
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        let _ = f;
    }

    /// This module's own train/eval flag, if its forward depends on the mode.
    fn mode(&self) -> Option<&Mode> {
        None
    }

    /// Every tensor of this module and its descendants with its dotted name, in canonical order (own tensors
    /// first, then each child's in turn).
    fn named_tensors(&self) -> Vec<NamedTensor> {
        let mut out = Vec::new();
        let mut prefix = String::new();
        collect_tensors(self, &mut prefix, &mut out);
        out
    }

    /// The parameters with dotted names, in canonical order.
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        filter_kind(self.named_tensors(), TensorKind::Parameter)
    }

    /// The buffers with dotted names, in canonical order.
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        filter_kind(self.named_tensors(), TensorKind::Buffer)
    }

    /// The parameters, in canonical order.
    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters().into_iter().map(|(_, t)| t).collect()
    }

    /// Total number of learnable scalars (buffers are not counted, as in docs/MODEL.md §4).
    fn num_parameters(&self) -> usize {
        self.named_parameters().iter().map(|(_, t)| t.numel()).sum()
    }

    /// Call `f` for every descendant module (not `self`), pre-order, with its dotted path. Use
    /// `(m as &dyn std::any::Any).downcast_ref::<T>()` to act on one layer type.
    fn for_each_module(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        let mut prefix = String::new();
        walk_children(self, &mut prefix, f);
    }

    /// Put this module and every descendant in training (`true`) or evaluation (`false`) mode.
    fn set_training(&self, training: bool) {
        if let Some(m) = self.mode() {
            m.set(training);
        }
        self.for_each_module(&mut |_, m| {
            if let Some(mode) = m.mode() {
                mode.set(training);
            }
        });
    }

    /// Training mode (`set_training(true)`).
    fn train(&self) {
        self.set_training(true);
    }

    /// Evaluation mode (`set_training(false)`).
    fn eval(&self) {
        self.set_training(false);
    }

    /// Move every parameter and buffer of this module and its descendants to `device`, in place
    /// ([`Tensor::to_device_`]). Each tensor keeps its identity, name, dtype and `requires_grad` flag, and
    /// the train/eval mode is unchanged, so handles taken earlier stay valid: optimizer parameter groups,
    /// [`crate::optim::GradAccumulator`] slots and a [`crate::ModelEma`] source all follow. Move the
    /// optimizer's own state with [`crate::optim::Optimizer::to_device`] and the EMA copy with
    /// [`crate::ModelEma::to_device`].
    ///
    /// ```
    /// use folo_core::{rng::Rng, Device};
    /// use folo_nn::{Conv2d, Conv2dConfig, Module};
    ///
    /// let mut conv = Conv2d::new(Conv2dConfig::new(3, 8, 3), &mut Rng::seed(0));
    /// let id = conv.weight.id();
    /// conv.to_device(Device::Cpu).unwrap(); // a no-op; `Device::Gpu(0)` after `folo_gpu::init()`
    /// assert_eq!(conv.weight.id(), id);
    /// assert!(conv.weight.requires_grad());
    /// ```
    ///
    /// Modules holding tensors outside [`Module::visit_tensors`] (caches, constants) override this and
    /// move those too.
    fn to_device(&mut self, device: Device) -> Result<()> {
        for t in self.named_tensors() {
            t.tensor.to_device_(device)?;
        }
        Ok(())
    }

    /// The mode of this module, or of its first descendant that has one; `true` if none does.
    fn is_training(&self) -> bool {
        if let Some(m) = self.mode() {
            return m.is_training();
        }
        let mut found = None;
        self.for_each_module(&mut |_, m| {
            if found.is_none() {
                found = m.mode().map(Mode::is_training);
            }
        });
        found.unwrap_or(true)
    }
}

/// A module that maps one tensor to one tensor.
pub trait Forward {
    /// Run the module on `x`.
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// A [`Module`] that is also a [`Forward`]: anything that can sit in a [`crate::Sequential`].
pub trait Layer: Module + Forward {}

impl<T: Module + Forward> Layer for T {}

fn filter_kind(ts: Vec<NamedTensor>, kind: TensorKind) -> Vec<(String, Tensor)> {
    ts.into_iter().filter(|t| t.kind == kind).map(|t| (t.name, t.tensor)).collect()
}

fn join(prefix: &mut String, name: &str) -> usize {
    let len = prefix.len();
    if !name.is_empty() {
        if !prefix.is_empty() {
            prefix.push('.');
        }
        prefix.push_str(name);
    }
    len
}

fn collect_tensors<M: Module + ?Sized>(m: &M, prefix: &mut String, out: &mut Vec<NamedTensor>) {
    m.visit_tensors(&mut |name, t, kind| {
        let len = join(prefix, name);
        out.push(NamedTensor { name: prefix.clone(), tensor: t.clone(), kind });
        prefix.truncate(len);
    });
    m.visit_children(&mut |name, child| {
        let len = join(prefix, name);
        collect_tensors(child, prefix, out);
        prefix.truncate(len);
    });
}

fn walk_children<M: Module + ?Sized>(m: &M, prefix: &mut String, f: &mut dyn FnMut(&str, &dyn Module)) {
    m.visit_children(&mut |name, child| {
        let len = join(prefix, name);
        f(prefix, child);
        walk_children(child, prefix, f);
        prefix.truncate(len);
    });
}
