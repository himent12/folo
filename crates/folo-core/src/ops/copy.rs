//! Ops that build new storage: concatenation, splitting, indexing, selection, padding, conversion.

use std::borrow::Borrow;

use super::grads;
use crate::backend;
use crate::shape::DimVec;
use crate::{DType, Device, Dim, Error, Layout, Result, Shape, Tensor};

impl Tensor {
    /// Concatenate along `dim`. All tensors need the same rank, dtype, device and sizes except along `dim`.
    pub fn cat<T: Borrow<Tensor>>(tensors: &[T], dim: impl Dim) -> Result<Tensor> {
        let first = tensors.first().ok_or_else(|| Error::invalid_arg("cat", "no tensors given"))?.borrow();
        let d = dim.to_index(first.rank(), "cat")?;
        let mut total = 0;
        for t in tensors {
            let t = t.borrow();
            first.check_same_device(t, "cat")?;
            if t.dtype() != first.dtype() {
                return Err(Error::DTypeMismatch { op: "cat", expected: first.dtype(), got: t.dtype() });
            }
            let ok = t.rank() == first.rank()
                && t.dims().iter().zip(first.dims()).enumerate().all(|(i, (a, b))| i == d || a == b);
            if !ok {
                return Err(Error::ShapeMismatch {
                    op: "cat",
                    lhs: first.shape().clone(),
                    rhs: t.shape().clone(),
                });
            }
            total += t.dims()[d];
        }
        if tensors.len() == 1 {
            return Ok(first.clone());
        }
        let shape = first.shape().with_dim(d, total);
        // The pieces tile the output exactly, so it needs no initialisation.
        let out = Tensor::uninit_with(&shape, first.dtype(), first.device())?;
        let b = first.backend()?;
        let full = Layout::contiguous(shape);
        let mut sizes = Vec::with_capacity(tensors.len());
        {
            let mut os = out.cell().write();
            let mut start = 0;
            for t in tensors {
                let t = t.borrow();
                let n = t.dims()[d];
                t.with_view(|v| b.copy_into(&mut os, &full.slice(d, start, n, 1), v))?;
                sizes.push(n);
                start += n;
            }
        }
        let inputs: Vec<&Tensor> = tensors.iter().map(|t| t.borrow()).collect();
        Ok(out.track("cat", &inputs, move |g| {
            let mut start = 0;
            let mut gs = smallvec::SmallVec::new();
            for &n in &sizes {
                gs.push(Some(g.narrow(d, start, n)?));
                start += n;
            }
            Ok(gs)
        }))
    }

    /// Stack along a new dimension `dim` (`0..=rank`). All tensors need the same shape.
    pub fn stack<T: Borrow<Tensor>>(tensors: &[T], dim: impl Dim) -> Result<Tensor> {
        let first = tensors.first().ok_or_else(|| Error::invalid_arg("stack", "no tensors given"))?.borrow();
        let d = dim.to_index(first.rank() + 1, "stack")?;
        let parts = tensors
            .iter()
            .map(|t| {
                let t = t.borrow();
                if t.shape() != first.shape() {
                    return Err(Error::ShapeMismatch {
                        op: "stack",
                        lhs: first.shape().clone(),
                        rhs: t.shape().clone(),
                    });
                }
                t.unsqueeze(d)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&parts, d)
    }

    /// Split into pieces of `size` along `dim` (the last piece may be smaller). Pieces are views.
    pub fn split(&self, size: usize, dim: impl Dim) -> Result<Vec<Tensor>> {
        let d = dim.to_index(self.rank(), "split")?;
        if size == 0 {
            return Err(Error::invalid_arg("split", "size must be positive"));
        }
        let n = self.dims()[d];
        (0..n.div_ceil(size)).map(|i| self.narrow(d, i * size, size.min(n - i * size))).collect()
    }

    /// Split into pieces with the given sizes along `dim` (they must sum to the dimension). Pieces are views.
    pub fn split_sizes(&self, sizes: &[usize], dim: impl Dim) -> Result<Vec<Tensor>> {
        let d = dim.to_index(self.rank(), "split_sizes")?;
        if sizes.iter().sum::<usize>() != self.dims()[d] {
            return Err(Error::invalid_arg(
                "split_sizes",
                format!("sizes {sizes:?} do not sum to dimension {d} of {}", self.shape()),
            ));
        }
        let mut start = 0;
        sizes
            .iter()
            .map(|&s| {
                let t = self.narrow(d, start, s);
                start += s;
                t
            })
            .collect()
    }

    /// Split into `chunks` nearly equal pieces along `dim` (PyTorch's rule: size `ceil(n / chunks)`).
    pub fn chunk(&self, chunks: usize, dim: impl Dim) -> Result<Vec<Tensor>> {
        let d = dim.to_index(self.rank(), "chunk")?;
        if chunks == 0 {
            return Err(Error::invalid_arg("chunk", "chunks must be positive"));
        }
        self.split(self.dims()[d].div_ceil(chunks).max(1), d)
    }

    /// Select entries `indices` (1-D, `I64`/`U32`) along `dim`. Works for every dtype.
    pub fn index_select(&self, dim: impl Dim, indices: &Tensor) -> Result<Tensor> {
        let d = dim.to_index(self.rank(), "index_select")?;
        indices.check_index_dtype("index_select")?;
        self.check_same_device(indices, "index_select")?;
        if indices.rank() != 1 {
            return Err(Error::invalid_shape("index_select", indices.shape(), "indices must be 1-D"));
        }
        let b = self.backend()?;
        let s = super::with_views([self, indices], [self.layout(), indices.layout()], |[x, i]| {
            b.index_select(x, d, i)
        })?;
        let out = Tensor::wrap(s, self.shape().with_dim(d, indices.numel()), "index_select")?;
        let (idx, in_shape) = (indices.clone(), self.shape().clone());
        Ok(out.track_saved("index_select", &[self], &[indices], move |g| {
            let b = g.backend()?;
            let s = super::with_views([g, &idx], [g.layout(), idx.layout()], |[gv, iv]| {
                b.index_select_backward(gv, d, iv, &in_shape)
            })?;
            Ok(grads![Some(Tensor::wrap(s, in_shape.clone(), "index_select_backward")?)])
        }))
    }

    /// A copy of `self` with the slices of `src` added at `indices` (1-D) along `dim`; duplicates accumulate.
    pub fn index_add(&self, dim: impl Dim, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        let d = dim.to_index(self.rank(), "index_add")?;
        indices.check_index_dtype("index_add")?;
        self.check_f32("index_add")?;
        src.check_f32("index_add")?;
        self.check_same_device(src, "index_add")?;
        self.check_same_device(indices, "index_add")?;
        if indices.rank() != 1 || src.shape() != &self.shape().with_dim(d, indices.numel()) {
            return Err(Error::ShapeMismatch {
                op: "index_add",
                lhs: self.shape().clone(),
                rhs: src.shape().clone(),
            });
        }
        let b = self.backend()?;
        let s = super::with_views(
            [self, indices, src],
            [self.layout(), indices.layout(), src.layout()],
            |[x, i, s]| b.index_add(x, d, i, s),
        )?;
        let out = Tensor::wrap(s, self.shape().clone(), "index_add")?;
        let idx = indices.clone();
        Ok(out.track_saved("index_add", &[self, src], &[indices], move |g| {
            Ok(grads![Some(g.clone()), Some(g.index_select(d, &idx)?)])
        }))
    }

    fn check_gather_shapes(&self, op: &'static str, d: usize, index: &Tensor) -> Result<()> {
        index.check_index_dtype(op)?;
        self.check_same_device(index, op)?;
        let ok = index.rank() == self.rank()
            && index.dims().iter().zip(self.dims()).enumerate().all(|(i, (a, b))| i == d || a == b);
        if !ok {
            return Err(Error::ShapeMismatch { op, lhs: self.shape().clone(), rhs: index.shape().clone() });
        }
        Ok(())
    }

    /// `out[.., i, ..] = self[.., index[.., i, ..], ..]` along `dim`. `index` has the same shape as `self`
    /// except along `dim`; the output has `index`'s shape.
    pub fn gather(&self, dim: impl Dim, index: &Tensor) -> Result<Tensor> {
        let d = dim.to_index(self.rank(), "gather")?;
        self.check_gather_shapes("gather", d, index)?;
        let b = self.backend()?;
        let s =
            super::with_views([self, index], [self.layout(), index.layout()], |[x, i]| b.gather(x, d, i))?;
        let out = Tensor::wrap(s, index.shape().clone(), "gather")?;
        let (idx, in_shape) = (index.clone(), self.shape().clone());
        Ok(out.track_saved("gather", &[self], &[index], move |g| {
            Ok(grads![Some(g.gather_backward(d, &idx, &in_shape)?)])
        }))
    }

    /// Zeros of `in_shape` with `out[.., index[.., i, ..], ..] += self[.., i, ..]` along `dim`, without autograd:
    /// the gradient of [`Tensor::gather`] (and of [`Tensor::topk`]).
    pub(crate) fn gather_backward(&self, d: usize, index: &Tensor, in_shape: &Shape) -> Result<Tensor> {
        let b = self.backend()?;
        let s = super::with_views([self, index], [self.layout(), index.layout()], |[gv, iv]| {
            b.gather_backward(gv, d, iv, in_shape)
        })?;
        Tensor::wrap(s, in_shape.clone(), "gather_backward")
    }

    /// A copy of `self` with `out[.., index[.., i, ..], ..] += src[.., i, ..]` along `dim`. `index` and `src`
    /// have the same shape, equal to `self`'s except along `dim`.
    pub fn scatter_add(&self, dim: impl Dim, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let d = dim.to_index(self.rank(), "scatter_add")?;
        self.check_gather_shapes("scatter_add", d, index)?;
        self.check_f32("scatter_add")?;
        src.check_f32("scatter_add")?;
        self.check_same_device(src, "scatter_add")?;
        if src.shape() != index.shape() {
            return Err(Error::ShapeMismatch {
                op: "scatter_add",
                lhs: index.shape().clone(),
                rhs: src.shape().clone(),
            });
        }
        let b = self.backend()?;
        let s = super::with_views(
            [self, index, src],
            [self.layout(), index.layout(), src.layout()],
            |[x, i, s]| b.scatter_add(x, d, i, s),
        )?;
        let out = Tensor::wrap(s, self.shape().clone(), "scatter_add")?;
        let idx = index.clone();
        Ok(out.track_saved("scatter_add", &[self, src], &[index], move |g| {
            Ok(grads![Some(g.clone()), Some(g.gather(d, &idx)?)])
        }))
    }

    /// `self != 0 ? on_true : on_false` elementwise with broadcasting over all three. `self` is an `F32` or
    /// `U8` mask; the values are `F32` (or both `F16` on a backend that computes in it).
    pub fn where_cond(&self, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        if !matches!(self.dtype(), DType::F32 | DType::U8) {
            return Err(Error::UnsupportedDType { op: "where_cond", dtype: self.dtype() });
        }
        on_true.check_float("where_cond")?;
        on_false.check_float_like(on_true, "where_cond")?;
        self.check_same_device(on_true, "where_cond")?;
        self.check_same_device(on_false, "where_cond")?;
        let shape = self
            .shape()
            .broadcast(on_true.shape(), "where_cond")?
            .broadcast(on_false.shape(), "where_cond")?;
        let lc = self.layout().broadcast_to(&shape, "where_cond")?;
        let la = on_true.layout().broadcast_to(&shape, "where_cond")?;
        let lb = on_false.layout().broadcast_to(&shape, "where_cond")?;
        let b = self.backend()?;
        let s =
            super::with_views([self, on_true, on_false], [&lc, &la, &lb], |[c, a, f]| b.where_cond(c, a, f))?;
        let out = Tensor::wrap(s, shape, "where_cond")?;
        let cond = self.clone();
        let (sa, sb) = (on_true.shape().clone(), on_false.shape().clone());
        Ok(out.track_saved("where_cond", &[on_true, on_false], &[self], move |g| {
            let zero = g.scalar_like(0.0)?;
            Ok(grads![
                Some(cond.where_cond(g, &zero)?.sum_to(&sa)?),
                Some(cond.where_cond(&zero, g)?.sum_to(&sb)?)
            ])
        }))
    }

    /// `value` where `mask != 0`, else `self` (mask broadcasts to `self`).
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor> {
        let out = mask.where_cond(&self.scalar_like(value)?, self)?;
        if out.shape() != self.shape() {
            return Err(Error::ShapeMismatch {
                op: "masked_fill",
                lhs: self.shape().clone(),
                rhs: mask.shape().clone(),
            });
        }
        Ok(out)
    }

    /// Pad the last `pads.len()` dims with `value`: `pads[i] = (before, after)` applies to dim
    /// `rank - pads.len() + i`. E.g. `x.pad(&[(1, 1), (2, 2)], 0.0)` pads H by 1 and W by 2 on an NCHW tensor.
    pub fn pad(&self, pads: &[(usize, usize)], value: f32) -> Result<Tensor> {
        if pads.len() > self.rank() {
            return Err(Error::invalid_arg("pad", format!("{} pads for rank {}", pads.len(), self.rank())));
        }
        let off = self.rank() - pads.len();
        let mut dims: DimVec = self.dims().iter().copied().collect();
        for (i, &(a, b)) in pads.iter().enumerate() {
            dims[off + i] += a + b;
        }
        let shape = Shape::from(dims);
        let out = Tensor::full_with(&shape, value as f64, self.dtype(), self.device())?;
        let mut inner = Layout::contiguous(shape);
        for (i, &(a, _)) in pads.iter().enumerate() {
            inner = inner.slice(off + i, a, self.dims()[off + i], 1);
        }
        let b = self.backend()?;
        {
            let mut os = out.cell().write();
            self.with_view(|v| b.copy_into(&mut os, &inner, v))?;
        }
        let pads = pads.to_vec();
        let in_dims: DimVec = self.dims().iter().copied().collect();
        Ok(out.track("pad", &[self], move |g| {
            let mut gx = g.clone();
            for (i, &(a, _)) in pads.iter().enumerate() {
                gx = gx.narrow(off + i, a, in_dims[off + i])?;
            }
            Ok(grads![Some(gx)])
        }))
    }

    /// Tile the tensor: dim `i` is repeated `reps[i]` times. `reps` may be longer than the rank (new leading
    /// dims), as in PyTorch.
    pub fn repeat(&self, reps: &[usize]) -> Result<Tensor> {
        if reps.len() < self.rank() {
            return Err(Error::invalid_arg(
                "repeat",
                format!("{} repeats for rank {}", reps.len(), self.rank()),
            ));
        }
        let lead = reps.len() - self.rank();
        let dims: Vec<usize> = std::iter::repeat_n(1, lead).chain(self.dims().iter().copied()).collect();
        let mut inter = Vec::with_capacity(2 * dims.len());
        let mut expanded = Vec::with_capacity(2 * dims.len());
        let mut fin = Vec::with_capacity(dims.len());
        for (&d, &r) in dims.iter().zip(reps) {
            inter.extend([1, d]);
            expanded.extend([r, d]);
            fin.push(r * d);
        }
        self.reshape(inter)?.broadcast_to(expanded)?.contiguous()?.reshape(fin)
    }

    /// Convert to `dtype` (floats to integers truncate toward zero and saturate). Gradients flow between float
    /// dtypes.
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        let b = self.backend()?;
        let s = self.with_view(|v| b.cast(v, dtype))?;
        let out = Tensor::wrap(s, self.shape().clone(), "to_dtype")?;
        if !(dtype.is_float() && self.dtype().is_float()) {
            return Ok(out);
        }
        let src = self.dtype();
        Ok(out.track("to_dtype", &[self], move |g| Ok(grads![Some(g.to_dtype(src)?)])))
    }

    /// Copy to `device` (a no-op if already there). Gradients flow back to the source device.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        if device == self.device() {
            return Ok(self.clone());
        }
        let host = self.to_cpu_storage()?;
        let s = backend::backend(device)?.upload(device, &host)?;
        let out = Tensor::wrap(s, self.shape().clone(), "to_device")?;
        let src = self.device();
        Ok(out.track("to_device", &[self], move |g| Ok(grads![Some(g.to_device(src)?)])))
    }

    /// Move this tensor's storage to `device` **in place** (a no-op if it is already there): PyTorch's
    /// `param.data = param.data.to(device)`. The tensor keeps its identity ([`Tensor::id`], so [`crate::Grads`]
    /// keys and optimizer state still match), its `requires_grad` flag and its layout; every clone and view
    /// sharing the storage moves with it. This is how folo-nn moves a whole model (`Module::to_device`).
    ///
    /// Not an autograd op. The storage version is bumped, so a graph recorded before the move that saved
    /// this tensor fails `backward()` with [`Error::ModifiedInPlace`]. Only leaves can move (a tensor
    /// produced by a recorded op would send its gradient to the wrong device). Do not call it while another
    /// thread uses the tensor.
    pub fn to_device_(&self, device: Device) -> Result<()> {
        if device == self.device() {
            return Ok(());
        }
        if !self.is_leaf() {
            return Err(Error::Autograd(format!(
                "to_device_: {} is the result of a recorded op; detach it or use to_device",
                self.grad_fn_name().unwrap_or("?")
            )));
        }
        let dst = backend::backend(device)?;
        let cell = self.cell();
        let mut guard = cell.write();
        let new = match &*guard {
            crate::Storage::Cpu(host) => dst.upload(device, host)?,
            s => {
                let whole = Layout::contiguous(Shape::new(&[s.len()]));
                let host = backend::backend(s.device())?.download(backend::View::new(s, &whole))?;
                if device.is_cpu() { crate::Storage::Cpu(host) } else { dst.upload(device, &host)? }
            }
        };
        cell.replace(&mut guard, new);
        Ok(())
    }

    /// Convert this tensor's storage to `dtype` **in place** (a no-op if it already has it), on its device:
    /// PyTorch's `param.data = param.data.half()`. Like [`Tensor::to_device_`], the identity, `requires_grad`
    /// flag and layout are kept and every clone and view sharing the storage follows (layouts count elements,
    /// so they stay valid). This is how folo-nn converts a whole model (`Module::to_dtype`). The whole storage
    /// is converted, also elements no view of it reaches.
    ///
    /// Not an autograd op: the storage version is bumped, and only leaves convert. Do not call it while another
    /// thread uses the tensor.
    pub fn to_dtype_(&self, dtype: DType) -> Result<()> {
        if dtype == self.dtype() {
            return Ok(());
        }
        if !self.is_leaf() {
            return Err(Error::Autograd(format!(
                "to_dtype_: {} is the result of a recorded op; detach it or use to_dtype",
                self.grad_fn_name().unwrap_or("?")
            )));
        }
        let cell = self.cell();
        let mut guard = cell.write();
        let whole = Layout::contiguous(Shape::new(&[guard.len()]));
        let new = backend::backend(guard.device())?.cast(backend::View::new(&guard, &whole), dtype)?;
        cell.replace(&mut guard, new);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Tensor};

    fn t(v: &[f32], s: &[usize]) -> Tensor {
        Tensor::from_slice(v, s).unwrap()
    }

    #[test]
    fn cat_stack_split() {
        let a = t(&[1., 2., 3., 4.], &[2, 2]);
        let b = t(&[5., 6.], &[1, 2]);
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.to_vec2::<f32>().unwrap(), vec![vec![1., 2.], vec![3., 4.], vec![5., 6.]]);
        let c1 = Tensor::cat(&[a.clone(), a.t().unwrap()], 1).unwrap();
        assert_eq!(c1.to_vec2::<f32>().unwrap(), vec![vec![1., 2., 1., 3.], vec![3., 4., 2., 4.]]);
        assert!(Tensor::cat(&[&a, &b], 1).is_err());
        let s = Tensor::stack(&[&a, &a], 1).unwrap();
        assert_eq!(s.dims(), &[2, 2, 2]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![1., 2., 1., 2., 3., 4., 3., 4.]);
        let parts = Tensor::arange(0.0, 7.0, 1.0).unwrap().split(3, 0).unwrap();
        assert_eq!(parts.iter().map(|p| p.numel()).collect::<Vec<_>>(), vec![3, 3, 1]);
        let ch = Tensor::arange(0.0, 6.0, 1.0).unwrap().chunk(4, 0).unwrap();
        assert_eq!(ch.len(), 3);
        let ss = Tensor::arange(0.0, 6.0, 1.0).unwrap().split_sizes(&[1, 5], 0).unwrap();
        assert_eq!(ss[1].to_vec::<f32>().unwrap(), vec![1., 2., 3., 4., 5.]);
        // Integer cat.
        let i = Tensor::cat(&[Tensor::arange_i64(0, 2), Tensor::arange_i64(5, 6)], 0).unwrap();
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![0, 1, 5]);
    }

    #[test]
    fn indexing() {
        let x = Tensor::arange(0.0, 12.0, 1.0).unwrap().reshape([3, 4]).unwrap();
        let idx = Tensor::from_slice(&[2i64, 0, 2], [3]).unwrap();
        let r = x.index_select(0, &idx).unwrap();
        assert_eq!(r.to_vec2::<f32>().unwrap()[0], vec![8., 9., 10., 11.]);
        let c = x.index_select(1, &Tensor::from_slice(&[3u32, 1], [2]).unwrap()).unwrap();
        assert_eq!(c.to_vec2::<f32>().unwrap(), vec![vec![3., 1.], vec![7., 5.], vec![11., 9.]]);
        assert!(x.index_select(0, &Tensor::from_slice(&[3i64], [1]).unwrap()).is_err());
        let gi = Tensor::from_slice(&[0i64, 3, 1, 1, 2, 0], [3, 2]).unwrap();
        let g = x.gather(1, &gi).unwrap();
        assert_eq!(g.to_vec2::<f32>().unwrap(), vec![vec![0., 3.], vec![5., 5.], vec![10., 8.]]);
        let z = Tensor::zeros([3, 4]);
        let s = z.scatter_add(1, &gi, &Tensor::ones([3, 2])).unwrap();
        assert_eq!(s.to_vec2::<f32>().unwrap()[1], vec![0., 2., 0., 0.]);
        let ia = Tensor::zeros([3, 2]).index_add(0, &idx, &Tensor::ones([3, 2])).unwrap();
        assert_eq!(ia.to_vec2::<f32>().unwrap(), vec![vec![1., 1.], vec![0., 0.], vec![2., 2.]]);
        // index_select on u8 data.
        let u = Tensor::from_slice(&[1u8, 2, 3], [3]).unwrap();
        assert_eq!(
            u.index_select(0, &Tensor::from_slice(&[2i64, 2], [2]).unwrap()).unwrap().to_vec::<u8>().unwrap(),
            vec![3, 3]
        );
    }

    #[test]
    fn where_pad_repeat_cast() {
        let x = t(&[1., -2., 3., -4.], &[2, 2]);
        let m = x.gt_scalar(0.0).unwrap();
        let w = m.where_cond(&x, &t(&[0.0], &[])).unwrap();
        assert_eq!(w.to_vec::<f32>().unwrap(), vec![1., 0., 3., 0.]);
        let mu8 = m.to_dtype(DType::U8).unwrap();
        assert_eq!(x.masked_fill(&mu8, 9.0).unwrap().to_vec::<f32>().unwrap(), vec![9., -2., 9., -4.]);
        let p = x.pad(&[(1, 0), (0, 2)], 7.0).unwrap();
        assert_eq!(p.dims(), &[3, 4]);
        assert_eq!(
            p.to_vec2::<f32>().unwrap(),
            vec![vec![7., 7., 7., 7.], vec![1., -2., 7., 7.], vec![3., -4., 7., 7.]]
        );
        let r = t(&[1., 2.], &[2]).repeat(&[2, 3]).unwrap();
        assert_eq!(r.to_vec2::<f32>().unwrap(), vec![vec![1., 2., 1., 2., 1., 2.]; 2]);
        let h = x.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
        assert_eq!(h.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
        let i = t(&[1.7, -1.7, 300.0], &[3]).to_dtype(DType::U8).unwrap();
        assert_eq!(i.to_vec::<u8>().unwrap(), vec![1, 0, 255]);
        // I64 → F32 rounds once: 2^62 + 2^38 + 1 is just above the midpoint of two f32 values (2^39 apart), but
        // rounding through f64 lands exactly on it and then ties to even, one f32 step too low.
        let big = [(1i64 << 62) + (1 << 38) + 1, -((1i64 << 62) + (1 << 38) + 1), i64::MAX, i64::MIN];
        let f = Tensor::from_slice(&big, [4]).unwrap().to_dtype(DType::F32).unwrap();
        assert_eq!(f.to_vec::<f32>().unwrap(), big.map(|v| v as f32));
        let bytes = Tensor::from_bytes(&[1u8, 0, 0, 0, 2, 0, 0, 0], DType::U32, [2]).unwrap();
        assert_eq!(bytes.to_vec::<u32>().unwrap(), vec![1, 2]);
    }
}
