//! Views: ops that only change the layout (no copy), plus `contiguous`.

use super::grads;
use crate::backend::View;
use crate::shape::DimVec;
use crate::{Dim, Error, Layout, Result, Shape, ShapeSpec, Tensor};

impl Tensor {
    /// A contiguous copy without autograd (internal building block).
    pub(crate) fn copy_contiguous(&self) -> Result<Tensor> {
        let b = self.backend()?;
        let s = self.with_view(|v| b.copy_strided(v))?;
        Tensor::wrap(s, self.shape().clone(), "contiguous")
    }

    /// `self` if already contiguous, else a contiguous copy.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let out = self.copy_contiguous()?;
        Ok(out.track("contiguous", &[self], |g| Ok(grads![Some(g.clone())])))
    }

    /// Same elements, new shape (one entry may be `-1`). A view when the layout allows it, else a copy.
    pub fn reshape(&self, spec: impl ShapeSpec) -> Result<Tensor> {
        let shape = spec.resolve(self.numel(), "reshape")?;
        if &shape == self.shape() {
            return Ok(self.clone());
        }
        let out = match self.layout().reshape(&shape) {
            Some(l) => self.with_layout(l),
            None => self.copy_contiguous()?.with_layout(Layout::contiguous(shape)),
        };
        let in_shape = self.shape().clone();
        Ok(out.track("reshape", &[self], move |g| Ok(grads![Some(g.reshape(&in_shape)?)])))
    }

    /// Swap two dimensions (a view).
    pub fn transpose(&self, d0: impl Dim, d1: impl Dim) -> Result<Tensor> {
        let (a, b) = (d0.to_index(self.rank(), "transpose")?, d1.to_index(self.rank(), "transpose")?);
        if a == b {
            return Ok(self.clone());
        }
        let out = self.with_layout(self.layout().transpose(a, b));
        Ok(out.track("transpose", &[self], move |g| Ok(grads![Some(g.transpose(a, b)?)])))
    }

    /// Swap the last two dimensions (matrix transpose; batched for rank > 2).
    pub fn t(&self) -> Result<Tensor> {
        if self.rank() < 2 {
            return Err(Error::invalid_shape("t", self.shape(), "needs rank >= 2"));
        }
        self.transpose(-2, -1)
    }

    /// Reorder dimensions: output dim `i` is input dim `dims[i]` (a view).
    pub fn permute<D: Dim>(&self, dims: impl AsRef<[D]>) -> Result<Tensor> {
        let dims = dims.as_ref();
        let r = self.rank();
        if dims.len() != r {
            return Err(Error::invalid_arg("permute", format!("{} dims given for rank {r}", dims.len())));
        }
        let perm: DimVec = dims.iter().map(|d| d.to_index(r, "permute")).collect::<Result<_>>()?;
        let mut seen: smallvec::SmallVec<[bool; 8]> = smallvec::smallvec![false; r];
        for &p in &perm {
            if std::mem::replace(&mut seen[p], true) {
                return Err(Error::invalid_arg("permute", format!("{perm:?} is not a permutation")));
            }
        }
        let out = self.with_layout(self.layout().permute(&perm));
        let mut inv = DimVec::from_elem(0, r);
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        Ok(out.track("permute", &[self], move |g| Ok(grads![Some(g.permute(&inv)?)])))
    }

    /// `len` elements of dimension `dim` starting at `start` (a view).
    pub fn narrow(&self, dim: impl Dim, start: usize, len: usize) -> Result<Tensor> {
        let d = dim.to_index(self.rank(), "narrow")?;
        let size = self.dims()[d];
        if start.checked_add(len).is_none_or(|e| e > size) {
            return Err(Error::invalid_arg(
                "narrow",
                format!("range {start}..{} out of bounds for dimension {d} of {}", start + len, self.shape()),
            ));
        }
        self.slice_impl("narrow", d, start, len, 1)
    }

    /// Elements `start..end` with `step` along `dim` (a view). `end` is clamped to the dimension size.
    pub fn slice(&self, dim: impl Dim, start: usize, end: usize, step: usize) -> Result<Tensor> {
        let d = dim.to_index(self.rank(), "slice")?;
        if step == 0 {
            return Err(Error::invalid_arg("slice", "step must be positive"));
        }
        let end = end.min(self.dims()[d]);
        let len = if start >= end { 0 } else { (end - start).div_ceil(step) };
        self.slice_impl("slice", d, start.min(self.dims()[d]), len, step)
    }

    fn slice_impl(
        &self,
        op: &'static str,
        d: usize,
        start: usize,
        len: usize,
        step: usize,
    ) -> Result<Tensor> {
        if start == 0 && step == 1 && len == self.dims()[d] {
            return Ok(self.clone());
        }
        let out = self.with_layout(self.layout().slice(d, start, len, step));
        let in_shape = self.shape().clone();
        Ok(out.track(op, &[self], move |g| {
            Ok(grads![Some(g.scatter_into_zeros(&in_shape, |l| l.slice(d, start, len, step))?)])
        }))
    }

    /// Zeros of `shape` with `self` written into the view `view(contiguous(shape))` (view backward).
    pub(crate) fn scatter_into_zeros(
        &self,
        shape: &Shape,
        view: impl Fn(&Layout) -> Layout,
    ) -> Result<Tensor> {
        let z = Tensor::zeros_with(shape, self.dtype(), self.device())?;
        let dst = view(&Layout::contiguous(shape.clone()));
        let b = self.backend()?;
        {
            let mut zs = z.cell().write();
            self.with_view(|v| b.copy_into(&mut zs, &dst, v))?;
        }
        Ok(z)
    }

    /// Remove dimension `dim` if it has size 1 (otherwise a no-op, as in PyTorch).
    pub fn squeeze(&self, dim: impl Dim) -> Result<Tensor> {
        let d = dim.to_index(self.rank(), "squeeze")?;
        if self.dims()[d] != 1 {
            return Ok(self.clone());
        }
        let out = self.with_layout(self.layout().squeeze(d));
        let in_shape = self.shape().clone();
        Ok(out.track("squeeze", &[self], move |g| Ok(grads![Some(g.reshape(&in_shape)?)])))
    }

    /// Insert a dimension of size 1 at `dim` (`0..=rank`; `-1` appends).
    pub fn unsqueeze(&self, dim: impl Dim) -> Result<Tensor> {
        let d = dim.to_index(self.rank() + 1, "unsqueeze")?;
        let out = self.with_layout(self.layout().unsqueeze(d));
        let in_shape = self.shape().clone();
        Ok(out.track("unsqueeze", &[self], move |g| Ok(grads![Some(g.reshape(&in_shape)?)])))
    }

    /// Merge dimensions `start..=end` into one.
    pub fn flatten(&self, start: impl Dim, end: impl Dim) -> Result<Tensor> {
        if self.rank() == 0 {
            return self.reshape([1usize]);
        }
        let (s, e) = (start.to_index(self.rank(), "flatten")?, end.to_index(self.rank(), "flatten")?);
        if s > e {
            return Err(Error::invalid_arg("flatten", format!("start dim {s} is after end dim {e}")));
        }
        let d = self.dims();
        let mut nd: DimVec = d[..s].iter().copied().collect();
        nd.push(d[s..=e].iter().product());
        nd.extend_from_slice(&d[e + 1..]);
        self.reshape(Shape::from(nd))
    }

    /// All elements as one dimension.
    pub fn flatten_all(&self) -> Result<Tensor> {
        self.reshape([self.numel()])
    }

    /// Broadcast to `shape` (size-1 dims repeat, new leading dims are added) without copying.
    pub fn broadcast_to(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        let shape = shape.into();
        if &shape == self.shape() {
            return Ok(self.clone());
        }
        let out = self.with_layout(self.layout().broadcast_to(&shape, "broadcast_to")?);
        let in_shape = self.shape().clone();
        Ok(out.track("broadcast_to", &[self], move |g| Ok(grads![Some(g.sum_to(&in_shape)?)])))
    }

    /// Alias of [`Tensor::broadcast_to`] (PyTorch's `expand`).
    pub fn expand(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        self.broadcast_to(shape)
    }

    /// Run `f` with views of `self` and `other` broadcast to their common shape.
    pub(crate) fn with_broadcast<R>(
        &self,
        other: &Tensor,
        op: &'static str,
        f: impl FnOnce(&Shape, View<'_>, View<'_>) -> Result<R>,
    ) -> Result<R> {
        let shape = self.shape().broadcast(other.shape(), op)?;
        let la = self.layout().broadcast_to(&shape, op)?;
        let lb = other.layout().broadcast_to(&shape, op)?;
        super::with_views([self, other], [&la, &lb], |[a, b]| f(&shape, a, b))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn t(v: &[f32], s: &[usize]) -> Tensor {
        Tensor::from_slice(v, s).unwrap()
    }

    #[test]
    fn reshape_transpose_permute() {
        let x = Tensor::arange(0.0, 24.0, 1.0).unwrap().reshape([2, 3, 4]).unwrap();
        assert_eq!(x.reshape([4, -1]).unwrap().dims(), &[4, 6]);
        let y = x.transpose(0, 2).unwrap();
        assert_eq!(y.dims(), &[4, 3, 2]);
        assert!(!y.is_contiguous());
        assert!(y.same_storage(&x));
        assert_eq!(y.to_vec::<f32>().unwrap()[..4], [0.0, 12.0, 4.0, 16.0]);
        // Reshape of a non-contiguous view copies.
        let z = y.reshape([-1]).unwrap();
        assert!(z.is_contiguous() && !z.same_storage(&x));
        let p = x.permute([2, 0, 1]).unwrap();
        assert_eq!(p.dims(), &[4, 2, 3]);
        assert_eq!(p.to_vec::<f32>().unwrap()[..3], [0.0, 4.0, 8.0]);
        assert!(x.permute([0, 0, 1]).is_err());
    }

    #[test]
    fn narrow_slice_squeeze() {
        let x = Tensor::arange(0.0, 12.0, 1.0).unwrap().reshape([3, 4]).unwrap();
        let n = x.narrow(1, 1, 2).unwrap();
        assert_eq!(n.to_vec2::<f32>().unwrap(), vec![vec![1.0, 2.0], vec![5.0, 6.0], vec![9.0, 10.0]]);
        assert!(x.narrow(1, 3, 2).is_err());
        let s = x.slice(1, 0, 4, 2).unwrap();
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let s = x.slice(0, 1, 100, 1).unwrap();
        assert_eq!(s.dims(), &[2, 4]);
        let u = x.unsqueeze(0).unwrap().unsqueeze(-1).unwrap();
        assert_eq!(u.dims(), &[1, 3, 4, 1]);
        assert_eq!(u.squeeze(0).unwrap().squeeze(-1).unwrap().dims(), &[3, 4]);
        assert_eq!(x.squeeze(0).unwrap().dims(), &[3, 4]);
        assert_eq!(x.flatten(0, 1).unwrap().dims(), &[12]);
        let b = t(&[1.0, 2.0, 3.0], &[3, 1]).broadcast_to([2, 3, 2]).unwrap();
        assert_eq!(
            b.to_vec::<f32>().unwrap(),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
        assert!(t(&[1.0, 2.0], &[2]).broadcast_to([3]).is_err());
        assert!(b.contiguous().unwrap().is_contiguous());
    }
}
