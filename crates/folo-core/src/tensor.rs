//! The [`Tensor`] handle: construction, inspection and data access.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use smallvec::SmallVec;

use crate::autograd::{self, BackwardFn, Grads, Node};
use crate::backend::{self, Backend, View};
use crate::rng::Rng;
use crate::storage::StorageCell;
use crate::{CpuStorage, DType, Device, Element, Error, Layout, Result, Shape, Storage};

/// A unique, never-reused tensor identity (used to key gradients).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

impl TensorId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        TensorId(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

pub(crate) struct TensorInner {
    id: TensorId,
    storage: Arc<StorageCell>,
    layout: Layout,
    dtype: DType,
    requires_grad: AtomicBool,
    node: Option<Node>,
}

impl Drop for TensorInner {
    fn drop(&mut self) {
        // Tear long graphs down iteratively instead of recursing once per op.
        let Some(node) = self.node.take() else { return };
        let mut stack = Vec::new();
        node.into_inputs(&mut stack);
        while let Some(t) = stack.pop() {
            if let Ok(mut inner) = Arc::try_unwrap(t.0)
                && let Some(n) = inner.node.take()
            {
                n.into_inputs(&mut stack);
            }
        }
    }
}

/// An n-dimensional array with autograd.
///
/// `Tensor` is a cheap handle (`Arc` inside): `clone()` shares the data and the identity. Views (reshape,
/// transpose, narrow, ...) share the storage but are new tensors. Ops return `Result`; the arithmetic
/// operators (`+ - * /`) panic on error instead, with the same message.
///
/// Data is immutable through ops; the only writes are the explicit in-place helpers (`sub_assign`,
/// `copy_from`, `update_f32`, ...) meant for optimizers under [`crate::no_grad`].
#[derive(Clone)]
pub struct Tensor(pub(crate) Arc<TensorInner>);

impl Tensor {
    // ----- construction -----

    pub(crate) fn from_cell(storage: Arc<StorageCell>, layout: Layout, dtype: DType) -> Tensor {
        Tensor(Arc::new(TensorInner {
            id: TensorId::next(),
            storage,
            layout,
            dtype,
            requires_grad: AtomicBool::new(false),
            node: None,
        }))
    }

    /// Wrap a storage as a contiguous tensor of `shape`. The storage must hold exactly `shape.numel()` elements.
    /// This is how device backends hand results back.
    pub fn from_storage(storage: Storage, shape: impl Into<Shape>) -> Result<Tensor> {
        let shape = shape.into();
        if storage.len() != shape.numel() {
            return Err(Error::invalid_shape(
                "from_storage",
                &shape,
                format!("storage holds {} elements", storage.len()),
            ));
        }
        Ok(Self::make(storage, shape))
    }

    /// Wrap a storage whose length is known to match.
    pub(crate) fn make(storage: Storage, shape: Shape) -> Tensor {
        debug_assert_eq!(storage.len(), shape.numel());
        let dtype = storage.dtype();
        Self::from_cell(Arc::new(StorageCell::new(storage)), Layout::contiguous(shape), dtype)
    }

    /// A new tensor (new identity, no graph) over the same storage with `layout`.
    pub(crate) fn with_layout(&self, layout: Layout) -> Tensor {
        Self::from_cell(self.0.storage.clone(), layout, self.0.dtype)
    }

    /// A tensor from host data.
    pub fn from_vec<T: Element>(data: Vec<T>, shape: impl Into<Shape>) -> Result<Tensor> {
        Self::from_storage(Storage::Cpu(T::into_storage(data)), shape)
    }

    /// A tensor from a host slice (copied).
    pub fn from_slice<T: Element>(data: &[T], shape: impl Into<Shape>) -> Result<Tensor> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// A tensor from raw native-endian bytes.
    pub fn from_bytes(bytes: &[u8], dtype: DType, shape: impl Into<Shape>) -> Result<Tensor> {
        Self::from_storage(Storage::Cpu(CpuStorage::from_bytes(bytes, dtype)?), shape)
    }

    /// A rank-0 `F32` tensor.
    pub fn scalar(v: f32) -> Tensor {
        Self::make(Storage::Cpu(CpuStorage::F32(vec![v])), Shape::scalar())
    }

    /// `F32` zeros on the CPU.
    pub fn zeros(shape: impl Into<Shape>) -> Tensor {
        Self::full(shape, 0.0)
    }

    /// `F32` ones on the CPU.
    pub fn ones(shape: impl Into<Shape>) -> Tensor {
        Self::full(shape, 1.0)
    }

    /// `F32` tensor filled with `value` on the CPU.
    pub fn full(shape: impl Into<Shape>, value: f32) -> Tensor {
        let shape = shape.into();
        Self::make(Storage::Cpu(CpuStorage::F32(crate::cpu::alloc_filled(shape.numel(), value))), shape)
    }

    /// A tensor of `dtype` on `device` filled with `value`.
    pub fn full_with(shape: impl Into<Shape>, value: f64, dtype: DType, device: Device) -> Result<Tensor> {
        let shape = shape.into();
        let s = backend::backend(device)?.full(device, dtype, shape.numel(), value)?;
        Ok(Self::make(s, shape))
    }

    /// Zeros of `dtype` on `device`.
    pub fn zeros_with(shape: impl Into<Shape>, dtype: DType, device: Device) -> Result<Tensor> {
        Self::full_with(shape, 0.0, dtype, device)
    }

    /// A tensor whose contents are unspecified (zeros, or stale data from the CPU buffer cache), for ops that
    /// write every element before anything reads it.
    pub(crate) fn uninit_with(shape: impl Into<Shape>, dtype: DType, device: Device) -> Result<Tensor> {
        let shape = shape.into();
        if device.is_cpu() && dtype == DType::F32 {
            let data = crate::cpu::alloc_uninit(shape.numel());
            return Ok(Self::make(Storage::Cpu(CpuStorage::F32(data)), shape));
        }
        Self::zeros_with(shape, dtype, device)
    }

    /// Zeros with this tensor's shape, dtype and device.
    pub fn zeros_like(&self) -> Result<Tensor> {
        Self::full_with(self.shape(), 0.0, self.dtype(), self.device())
    }

    /// Ones with this tensor's shape, dtype and device.
    pub fn ones_like(&self) -> Result<Tensor> {
        Self::full_with(self.shape(), 1.0, self.dtype(), self.device())
    }

    /// `value` everywhere, with this tensor's shape, dtype and device.
    pub fn full_like(&self, value: f64) -> Result<Tensor> {
        Self::full_with(self.shape(), value, self.dtype(), self.device())
    }

    /// `start, start + step, ..` up to (excluding) `end`, as `F32`.
    pub fn arange(start: f32, end: f32, step: f32) -> Result<Tensor> {
        if step == 0.0 || !step.is_finite() || !start.is_finite() || !end.is_finite() {
            return Err(Error::invalid_arg("arange", format!("bad range {start}..{end} step {step}")));
        }
        let n = ((end - start) / step).ceil().max(0.0) as usize;
        let v: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
        Self::from_vec(v, [n])
    }

    /// `start..end` as `I64` (for index tensors).
    pub fn arange_i64(start: i64, end: i64) -> Tensor {
        let v: Vec<i64> = (start..end.max(start)).collect();
        let n = v.len();
        Self::make(Storage::Cpu(CpuStorage::I64(v)), Shape::new(&[n]))
    }

    /// `n` evenly spaced values from `start` to `end` inclusive, as `F32`.
    pub fn linspace(start: f32, end: f32, n: usize) -> Tensor {
        let v: Vec<f32> = match n {
            0 => vec![],
            1 => vec![start],
            _ => {
                let step = (end as f64 - start as f64) / (n - 1) as f64;
                (0..n)
                    .map(|i| if i == n - 1 { end } else { (start as f64 + i as f64 * step) as f32 })
                    .collect()
            }
        };
        Self::make(Storage::Cpu(CpuStorage::F32(v)), Shape::new(&[n]))
    }

    /// The `n x n` identity matrix, `F32`.
    pub fn eye(n: usize) -> Tensor {
        let mut v = vec![0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        Self::make(Storage::Cpu(CpuStorage::F32(v)), Shape::new(&[n, n]))
    }

    /// Uniform samples in `[lo, hi)`, `F32` on the CPU.
    pub fn rand_uniform(shape: impl Into<Shape>, lo: f32, hi: f32, rng: &mut Rng) -> Tensor {
        let shape = shape.into();
        let v: Vec<f32> = (0..shape.numel()).map(|_| rng.uniform(lo, hi)).collect();
        Self::make(Storage::Cpu(CpuStorage::F32(v)), shape)
    }

    /// Normal samples with the given mean and standard deviation, `F32` on the CPU.
    pub fn randn(shape: impl Into<Shape>, mean: f32, std: f32, rng: &mut Rng) -> Tensor {
        let shape = shape.into();
        let mut v = vec![0f32; shape.numel()];
        rng.fill_normal(&mut v, mean, std);
        Self::make(Storage::Cpu(CpuStorage::F32(v)), shape)
    }

    // ----- inspection -----

    /// The identity used by [`Grads`].
    #[inline]
    pub fn id(&self) -> TensorId {
        self.0.id
    }

    /// The shape.
    #[inline]
    pub fn shape(&self) -> &Shape {
        self.0.layout.shape()
    }

    /// The dimensions.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.0.layout.dims()
    }

    /// Number of dimensions.
    #[inline]
    pub fn rank(&self) -> usize {
        self.0.layout.rank()
    }

    /// Number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.0.layout.numel()
    }

    /// Size of dimension `d` (negative counts from the end).
    pub fn dim(&self, d: impl crate::Dim) -> Result<usize> {
        self.shape().dim(d)
    }

    /// The element type.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    /// The device.
    #[inline]
    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    /// The strided layout over the storage.
    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    /// True if the elements are row-major without gaps.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.0.layout.is_contiguous()
    }

    /// True if `self` and `other` share a storage buffer.
    pub fn same_storage(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.0.storage, &other.0.storage)
    }

    /// The two dims of a rank-2 tensor.
    pub fn dims2(&self) -> Result<(usize, usize)> {
        match self.dims() {
            &[a, b] => Ok((a, b)),
            _ => Err(Error::invalid_shape("dims2", self.shape(), "expected rank 2")),
        }
    }

    /// The three dims of a rank-3 tensor.
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        match self.dims() {
            &[a, b, c] => Ok((a, b, c)),
            _ => Err(Error::invalid_shape("dims3", self.shape(), "expected rank 3")),
        }
    }

    /// The four dims of a rank-4 tensor.
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        match self.dims() {
            &[a, b, c, d] => Ok((a, b, c, d)),
            _ => Err(Error::invalid_shape("dims4", self.shape(), "expected rank 4")),
        }
    }

    // ----- autograd -----

    /// True if gradients flow to or through this tensor.
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.0.node.is_some() || self.0.requires_grad.load(Ordering::Relaxed)
    }

    /// Mark this leaf as requiring grad and return it (builder style, PyTorch's `requires_grad_()`).
    /// The flag lives in the shared handle, so every clone sees it.
    pub fn requires_grad_(self) -> Tensor {
        self.set_requires_grad(true);
        self
    }

    /// Set whether this leaf requires grad. On a non-leaf (the result of a recorded op) this only affects the
    /// leaf flag; use [`Tensor::detach`] to cut it from the graph.
    pub fn set_requires_grad(&self, on: bool) {
        self.0.requires_grad.store(on, Ordering::Relaxed);
    }

    /// True if this tensor was produced by a recorded op.
    pub fn is_leaf(&self) -> bool {
        self.0.node.is_none()
    }

    /// Name of the op that produced this tensor, if it was recorded.
    pub fn grad_fn_name(&self) -> Option<&'static str> {
        self.0.node.as_ref().map(|n| n.op)
    }

    /// A new tensor sharing the data, cut from the graph (`requires_grad() == false`).
    pub fn detach(&self) -> Tensor {
        self.with_layout(self.0.layout.clone())
    }

    /// Reverse-mode gradients of this scalar (any shape with one element) with respect to every leaf that
    /// requires grad and contributed to it.
    pub fn backward(&self) -> Result<Grads> {
        if self.numel() != 1 {
            return Err(Error::Autograd(format!(
                "backward() needs a single-element tensor, got shape {}; use backward_with(grad)",
                self.shape()
            )));
        }
        let seed = self.ones_like()?;
        autograd::backward(self, seed)
    }

    /// Reverse-mode gradients seeded with `grad` (same shape as `self`).
    pub fn backward_with(&self, grad: &Tensor) -> Result<Grads> {
        autograd::backward(self, grad.detach())
    }

    pub(crate) fn node(&self) -> Option<&Node> {
        self.0.node.as_ref()
    }

    pub(crate) fn storage_version(&self) -> u64 {
        self.0.storage.version()
    }

    /// Attach a backward node if grad mode is on and any input requires grad. `self` must be a fresh result.
    pub(crate) fn track<F>(self, op: &'static str, inputs: &[&Tensor], backward: F) -> Tensor
    where
        F: Fn(&Tensor) -> Result<SmallVec<[Option<Tensor>; 2]>> + Send + Sync + 'static,
    {
        self.track_saved(op, inputs, &[], backward)
    }

    /// Record a custom differentiable op (a fused kernel, a hand-written loss, ...) built outside folo-core.
    ///
    /// `output` is the op's result, computed however the caller likes: raw kernels, [`Tensor::update_f32`],
    /// or ordinary ops under [`crate::no_grad`]. Any graph it already carries is ignored. The returned tensor
    /// shares `output`'s data and, when grad mode is on and any input requires grad, records a node whose
    /// gradient is `backward(grad_output)`: one entry per input, in order, `None` for "no gradient", each with
    /// that input's shape. `backward` runs under `no_grad` and may use any tensor ops.
    ///
    /// Inputs and `output` are version-checked like built-in ops: modifying them in place before `backward()`
    /// is an error. Other tensors the closure captures are not tracked, so save copies or make them inputs.
    ///
    /// ```
    /// use folo_core::{no_grad, Result, Tensor};
    ///
    /// /// `x * y` with a hand-written backward.
    /// fn fused_mul(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    ///     let out = { let _g = no_grad(); x.mul(y)? };
    ///     let (xs, ys) = (x.detach(), y.detach());
    ///     Tensor::custom_op("fused_mul", &[x, y], out, move |g| {
    ///         Ok(vec![Some(g.mul(&ys)?), Some(g.mul(&xs)?)])
    ///     })
    /// }
    ///
    /// let x = Tensor::from_slice(&[2.0f32], [1])?.requires_grad_();
    /// let y = Tensor::from_slice(&[3.0f32], [1])?;
    /// let grads = fused_mul(&x, &y)?.sum_all()?.backward()?;
    /// assert_eq!(grads.get(&x).unwrap().to_vec::<f32>()?, vec![3.0]);
    /// # Ok::<(), folo_core::Error>(())
    /// ```
    pub fn custom_op<F>(name: &'static str, inputs: &[&Tensor], output: Tensor, backward: F) -> Result<Tensor>
    where
        F: Fn(&Tensor) -> Result<Vec<Option<Tensor>>> + Send + Sync + 'static,
    {
        let out = output.detach();
        let saved = out.clone();
        Ok(out.track_saved(name, inputs, &[&saved], move |g| Ok(SmallVec::from_vec(backward(g)?))))
    }

    /// [`Tensor::track`] for a backward closure that also reads `saved` tensors that are not differentiable
    /// inputs (indices, masks, the op's own output). Their storage versions are checked like the inputs'.
    pub(crate) fn track_saved<F>(
        self,
        op: &'static str,
        inputs: &[&Tensor],
        saved: &[&Tensor],
        backward: F,
    ) -> Tensor
    where
        F: Fn(&Tensor) -> Result<SmallVec<[Option<Tensor>; 2]>> + Send + Sync + 'static,
    {
        if !autograd::is_grad_enabled() || !inputs.iter().any(|t| t.requires_grad()) {
            return self;
        }
        let node = Node {
            op,
            inputs: inputs.iter().map(|&t| t.clone()).collect(),
            versions: inputs.iter().map(|t| t.storage_version()).collect(),
            saved: saved.iter().map(|t| (t.cell().clone(), t.storage_version())).collect(),
            backward: Box::new(backward) as BackwardFn,
        };
        let inner = &self.0;
        Tensor(Arc::new(TensorInner {
            id: TensorId::next(),
            storage: inner.storage.clone(),
            layout: inner.layout.clone(),
            dtype: inner.dtype,
            requires_grad: AtomicBool::new(false),
            node: Some(node),
        }))
    }

    // ----- storage access -----

    pub(crate) fn cell(&self) -> &Arc<StorageCell> {
        &self.0.storage
    }

    /// True if no other handle, view or graph node can see this tensor's storage.
    pub(crate) fn is_exclusive(&self) -> bool {
        Arc::strong_count(&self.0) == 1 && Arc::strong_count(&self.0.storage) == 1
    }

    pub(crate) fn backend(&self) -> Result<Arc<dyn Backend>> {
        backend::backend(self.device())
    }

    /// Run `f` with a backend [`View`] of this tensor (holds a read lock for the duration).
    pub fn with_view<R>(&self, f: impl FnOnce(View<'_>) -> R) -> R {
        let s = self.0.storage.read();
        f(View::new(&s, &self.0.layout))
    }

    /// The elements in row-major order as host data (downloads from a device).
    pub fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let b = self.backend()?;
        self.with_view(|v| b.download(v))
    }

    /// The elements in row-major order. `T` must match the dtype (convert with [`Tensor::to_dtype`]).
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        if self.dtype() != T::DTYPE {
            return Err(Error::DTypeMismatch { op: "to_vec", expected: T::DTYPE, got: self.dtype() });
        }
        if self.device().is_cpu() {
            let s = self.0.storage.read();
            let data = T::slice(s.as_cpu("to_vec")?).expect("dtype checked");
            return Ok(crate::cpu::elementwise::copy_strided(data, &self.0.layout));
        }
        Ok(T::from_storage(self.to_cpu_storage()?).expect("dtype checked"))
    }

    /// The rows of a rank-2 tensor.
    pub fn to_vec2<T: Element>(&self) -> Result<Vec<Vec<T>>> {
        let (_, c) = self.dims2()?;
        let v = self.to_vec::<T>()?;
        Ok(if c == 0 { vec![vec![]; self.dims()[0]] } else { v.chunks(c).map(|r| r.to_vec()).collect() })
    }

    /// The single element of a one-element tensor.
    pub fn item<T: Element>(&self) -> Result<T> {
        if self.numel() != 1 {
            return Err(Error::invalid_shape("item", self.shape(), "expected exactly one element"));
        }
        Ok(self.to_vec::<T>()?[0])
    }

    /// Run `f` on the elements as one contiguous `&[f32]` (CPU, `F32`). Borrows the storage when the tensor is
    /// contiguous, copies otherwise.
    pub fn with_f32_slice<R>(&self, f: impl FnOnce(&[f32]) -> R) -> Result<R> {
        let s = self.0.storage.read();
        let data = s.as_cpu("with_f32_slice")?.as_f32("with_f32_slice")?;
        match self.0.layout.contiguous_range() {
            Some(r) => Ok(f(&data[r])),
            None => Ok(f(&crate::cpu::elementwise::copy_strided(data, &self.0.layout))),
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor{} {} {}", self.shape(), self.dtype(), self.device())?;
        if self.requires_grad() {
            write!(f, " requires_grad")?;
        }
        if self.device().is_cpu()
            && self.dtype() == DType::F32
            && let Ok(v) = self.to_vec::<f32>()
        {
            let n = v.len().min(8);
            write!(f, " {:?}{}", &v[..n], if v.len() > n { " .." } else { "" })?;
        }
        Ok(())
    }
}
