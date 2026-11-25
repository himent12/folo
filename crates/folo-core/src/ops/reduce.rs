//! Reductions, softmax family and top-k.

use super::grads;
use crate::backend::ReduceOp;
use crate::shape::DimVec;
use crate::{AllDims, DType, Dim, Dims, Error, Layout, Result, Shape, Tensor};

impl Tensor {
    /// Reduce `dims` (sorted, unique) without autograd. Output keeps the reduced dims as size 1.
    fn reduce_keepdim(&self, op: ReduceOp, dims: &[usize], name: &'static str) -> Result<Tensor> {
        self.check_float(name)?;
        let b = self.backend()?;
        let mut keep_shape: DimVec = self.dims().iter().copied().collect();
        for &d in dims {
            keep_shape[d] = 1;
        }
        if dims.is_empty() {
            return Ok(self.detach());
        }
        // Merge runs of adjacent reduced dims (and adjacent kept dims) so each run is one backend call.
        let x = if self.is_contiguous() { self.detach() } else { self.copy_contiguous()? };
        let src = self.dims();
        let mut merged: DimVec = DimVec::new();
        let mut reduced: DimVec = DimVec::new();
        let mut i = 0;
        while i < src.len() {
            let red = dims.contains(&i);
            let mut size = 1;
            while i < src.len() && dims.contains(&i) == red {
                size *= src[i];
                i += 1;
            }
            if red {
                reduced.push(merged.len());
            }
            merged.push(size);
        }
        // An F16 sum over separated axes takes several passes: keep the partial sums in F32 (a partial can
        // overflow F16 or cancel only later) and round once at the end. Max and min are exact either way.
        let widen = op == ReduceOp::Sum && x.dtype() == DType::F16 && reduced.len() > 1;
        let x = if widen { x.to_dtype(DType::F32)?.detach() } else { x };
        let merged = Shape::from(merged);
        let strides = merged.contiguous_strides();
        let mut cur = x.with_layout(Layout::new(merged, strides, x.layout().offset()));
        // Innermost first: the first pass then reduces contiguous rows, the widest and most parallel case.
        for &d in reduced.iter().rev() {
            let s = cur.with_view(|v| b.reduce(op, v, d))?;
            let shape = cur.shape().with_dim(d, 1);
            cur = Tensor::wrap(s, shape, name)?;
        }
        if widen {
            cur = cur.to_dtype(DType::F16)?;
        }
        Ok(cur.with_layout(Layout::contiguous(Shape::from(keep_shape))))
    }

    /// Drop the reduced (size-1) dims of a fresh, untracked reduction result unless `keepdim`.
    fn finish_reduce(t: Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor> {
        if keepdim {
            return Ok(t);
        }
        let shape: DimVec =
            t.dims().iter().enumerate().filter(|(i, _)| !dims.contains(i)).map(|(_, &d)| d).collect();
        t.reshape(Shape::from(shape))
    }

    /// Sum over `dims` (a dim, a list of dims, or [`AllDims`]).
    pub fn sum(&self, dims: impl Dims, keepdim: bool) -> Result<Tensor> {
        let dims = dims.to_indices(self.rank(), "sum")?;
        let kept = self.reduce_keepdim(ReduceOp::Sum, &dims, "sum")?;
        let keep_shape = kept.shape().clone();
        let out = Self::finish_reduce(kept, &dims, keepdim)?;
        let in_shape = self.shape().clone();
        Ok(out.track("sum", &[self], move |g| {
            Ok(grads![Some(g.reshape(&keep_shape)?.broadcast_to(&in_shape)?)])
        }))
    }

    /// Sum of all elements as a rank-0 tensor.
    pub fn sum_all(&self) -> Result<Tensor> {
        self.sum(AllDims, false)
    }

    /// Mean over `dims`. An `F16` input is averaged in `F32` (its sum could overflow `F16` before the division)
    /// and the result rounded back to `F16`.
    pub fn mean(&self, dims: impl Dims, keepdim: bool) -> Result<Tensor> {
        let d = dims.to_indices(self.rank(), "mean")?;
        if self.dtype() == DType::F16 {
            self.check_float("mean")?;
            return self.to_dtype(DType::F32)?.mean(d.to_vec(), keepdim)?.to_dtype(DType::F16);
        }
        let n: usize = d.iter().map(|&i| self.dims()[i]).product();
        self.sum(d.to_vec(), keepdim)?.mul_scalar(1.0 / n.max(1) as f32)
    }

    /// Mean of all elements as a rank-0 tensor.
    pub fn mean_all(&self) -> Result<Tensor> {
        self.mean(AllDims, false)
    }

    /// Variance over `dims`; `unbiased` divides by `n - 1` (Bessel's correction). `F16` inputs are computed in
    /// `F32`, like [`Tensor::mean`].
    pub fn var(&self, dims: impl Dims, unbiased: bool, keepdim: bool) -> Result<Tensor> {
        let d = dims.to_indices(self.rank(), "var")?;
        if self.dtype() == DType::F16 {
            self.check_float("var")?;
            return self.to_dtype(DType::F32)?.var(d.to_vec(), unbiased, keepdim)?.to_dtype(DType::F16);
        }
        let n: usize = d.iter().map(|&i| self.dims()[i]).product();
        let denom = if unbiased { n.saturating_sub(1) } else { n };
        let mean = self.mean(d.to_vec(), true)?;
        let diff = self.sub(&mean)?;
        diff.square()?.sum(d.to_vec(), keepdim)?.mul_scalar(1.0 / denom as f32)
    }

    fn extreme(&self, op: ReduceOp, dims: impl Dims, keepdim: bool, name: &'static str) -> Result<Tensor> {
        let dims = dims.to_indices(self.rank(), name)?;
        if self.numel() == 0 {
            return Err(Error::invalid_shape(name, self.shape(), "cannot reduce an empty tensor"));
        }
        let kept = self.reduce_keepdim(op, &dims, name)?;
        let (x, y) = (self.clone(), kept.clone());
        let out = Self::finish_reduce(kept, &dims, keepdim)?;
        Ok(out.track_saved(name, &[self], &[&y.clone()], move |g| {
            // The gradient goes to the extreme elements, split evenly between ties.
            let mask = x.eq(&y)?;
            let count = mask.sum(dims.to_vec(), true)?;
            Ok(grads![Some(mask.mul(&g.reshape(y.shape())?.div(&count)?)?)])
        }))
    }

    /// Maximum over `dims`.
    pub fn max(&self, dims: impl Dims, keepdim: bool) -> Result<Tensor> {
        self.extreme(ReduceOp::Max, dims, keepdim, "max")
    }

    /// Minimum over `dims`.
    pub fn min(&self, dims: impl Dims, keepdim: bool) -> Result<Tensor> {
        self.extreme(ReduceOp::Min, dims, keepdim, "min")
    }

    /// Maximum of all elements as a rank-0 tensor.
    pub fn max_all(&self) -> Result<Tensor> {
        self.max(AllDims, false)
    }

    /// Minimum of all elements as a rank-0 tensor.
    pub fn min_all(&self) -> Result<Tensor> {
        self.min(AllDims, false)
    }

    fn arg_extreme(&self, op: ReduceOp, dim: impl Dim, keepdim: bool, name: &'static str) -> Result<Tensor> {
        self.check_float(name)?;
        let d = dim.to_index(self.rank(), name)?;
        if self.dims()[d] == 0 {
            return Err(Error::invalid_shape(name, self.shape(), "cannot reduce an empty dimension"));
        }
        let b = self.backend()?;
        let s = self.with_view(|v| b.reduce(op, v, d))?;
        let out = Tensor::wrap(s, self.shape().with_dim(d, 1), name)?;
        Self::finish_reduce(out, &[d], keepdim)
    }

    /// Index of the maximum along `dim` (first on ties), as `I64`.
    pub fn argmax(&self, dim: impl Dim, keepdim: bool) -> Result<Tensor> {
        self.arg_extreme(ReduceOp::ArgMax, dim, keepdim, "argmax")
    }

    /// Index of the minimum along `dim` (first on ties), as `I64`.
    pub fn argmin(&self, dim: impl Dim, keepdim: bool) -> Result<Tensor> {
        self.arg_extreme(ReduceOp::ArgMin, dim, keepdim, "argmin")
    }

    /// Softmax along `dim`, numerically stable.
    pub fn softmax(&self, dim: impl Dim) -> Result<Tensor> {
        self.softmax_impl(dim, false)
    }

    /// Log-softmax along `dim`, numerically stable.
    pub fn log_softmax(&self, dim: impl Dim) -> Result<Tensor> {
        self.softmax_impl(dim, true)
    }

    fn softmax_impl(&self, dim: impl Dim, log: bool) -> Result<Tensor> {
        let name = if log { "log_softmax" } else { "softmax" };
        // Under autocast an `F16` input runs in `F32`, as `torch.autocast` does: the backward `y · (g − Σ g·y)`
        // cancels, which f16 storage of its terms would turn into noise.
        if self.dtype() == DType::F16 && crate::autocast::autocast_dtype().is_some() {
            return self.to_dtype(DType::F32)?.softmax_impl(dim, log);
        }
        self.check_float(name)?;
        let d = dim.to_index(self.rank(), name)?;
        let b = self.backend()?;
        let s = self.with_view(|v| b.softmax(v, d, log))?;
        let out = Tensor::wrap(s, self.shape().clone(), name)?;
        let y = out.detach();
        Ok(out.track_saved(name, &[self], &[&y.clone()], move |g| {
            y.check_same_device(g, name)?;
            g.check_float_like(&y, name)?;
            let b = y.backend()?;
            let fused = super::with_views([&y, g], [y.layout(), g.layout()], |[vy, vg]| {
                b.softmax_backward(vy, vg, d, log)
            });
            let gx = match fused {
                Ok(s) => Tensor::wrap(s, y.shape().clone(), name)?,
                Err(Error::NotSupported { .. }) => {
                    if log {
                        g.sub(&y.exp()?.mul(&g.sum(d, true)?)?)?
                    } else {
                        y.mul(&g.sub(&g.mul(&y)?.sum(d, true)?)?)?
                    }
                }
                Err(e) => return Err(e),
            };
            Ok(grads![Some(gx)])
        }))
    }

    /// `log(sum(exp(x)))` over `dims`, numerically stable.
    pub fn logsumexp(&self, dims: impl Dims, keepdim: bool) -> Result<Tensor> {
        let d = dims.to_indices(self.rank(), "logsumexp")?;
        let m = self.max(d.to_vec(), true)?.detach().clamp(f32::MIN, f32::MAX)?;
        let s = self.sub(&m)?.exp()?.sum(d.to_vec(), true)?.log()?.add(&m)?;
        Self::finish_reduce(s, &d, keepdim)
    }

    /// The `k` largest (`largest = true`) or smallest entries along the last dim, sorted, with their `I64`
    /// indices. Ties are broken by lower index. Gradients flow to the selected values.
    pub fn topk(&self, k: usize, largest: bool) -> Result<(Tensor, Tensor)> {
        self.check_f32("topk")?;
        if self.rank() == 0 {
            return Err(Error::invalid_shape("topk", self.shape(), "needs rank >= 1"));
        }
        let d = self.dims()[self.rank() - 1];
        if k > d {
            return Err(Error::invalid_arg("topk", format!("k = {k} exceeds the last dimension ({d})")));
        }
        let b = self.backend()?;
        let (vs, is) = self.with_view(|v| b.topk(v, k, largest))?;
        let shape = self.shape().with_dim(self.rank() - 1, k);
        let values = Tensor::wrap(vs, shape.clone(), "topk")?;
        let indices = Tensor::wrap(is, shape, "topk")?;
        let (idx, in_shape) = (indices.clone(), self.shape().clone());
        let last = in_shape.rank() - 1;
        let values = values.track_saved("topk", &[self], &[&indices], move |g| {
            Ok(grads![Some(g.gather_backward(last, &idx, &in_shape)?)])
        });
        Ok((values, indices))
    }
}

#[cfg(test)]
mod tests {
    use crate::{AllDims, Tensor};

    #[test]
    fn reductions() {
        let x = Tensor::arange(0.0, 24.0, 1.0).unwrap().reshape([2, 3, 4]).unwrap();
        assert_eq!(
            x.sum(1, false).unwrap().to_vec::<f32>().unwrap(),
            vec![12., 15., 18., 21., 48., 51., 54., 57.]
        );
        assert_eq!(x.sum([0, 2], true).unwrap().dims(), &[1, 3, 1]);
        assert_eq!(x.sum([0, 2], false).unwrap().to_vec::<f32>().unwrap(), vec![60., 92., 124.]);
        assert_eq!(x.sum_all().unwrap().item::<f32>().unwrap(), 276.0);
        assert_eq!(x.mean(-1, false).unwrap().to_vec::<f32>().unwrap()[0], 1.5);
        assert_eq!(x.max(1, false).unwrap().to_vec::<f32>().unwrap()[..4], [8., 9., 10., 11.]);
        assert_eq!(x.min(AllDims, false).unwrap().item::<f32>().unwrap(), 0.0);
        assert_eq!(x.argmax(2, false).unwrap().to_vec::<i64>().unwrap(), vec![3; 6]);
        assert_eq!(x.argmin(0, true).unwrap().dims(), &[1, 3, 4]);
        // Non-contiguous input.
        let xt = x.transpose(0, 2).unwrap(); // [4,3,2]
        assert_eq!(xt.sum(2, false).unwrap().to_vec::<f32>().unwrap()[..3], [12., 20., 28.]);
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
        assert!((v.var(0, true, false).unwrap().item::<f32>().unwrap() - 5.0 / 3.0).abs() < 1e-6);
        assert!((v.var(0, false, false).unwrap().item::<f32>().unwrap() - 1.25).abs() < 1e-6);
        assert!(x.sum([1, 1], false).is_err());
        assert!(x.sum(3, false).is_err());
    }

    #[test]
    fn softmax_family() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 1000.0, 1000.0, 1000.0], [2, 3]).unwrap();
        let s = x.softmax(-1).unwrap().to_vec2::<f32>().unwrap();
        let e: Vec<f32> = [1.0f32, 2.0, 3.0].iter().map(|v| v.exp()).collect();
        let tot: f32 = e.iter().sum();
        for j in 0..3 {
            assert!((s[0][j] - e[j] / tot).abs() < 1e-6);
            assert!((s[1][j] - 1.0 / 3.0).abs() < 1e-6);
        }
        let ls = x.log_softmax(1).unwrap().to_vec2::<f32>().unwrap();
        assert!((ls[1][0] + 3f32.ln()).abs() < 1e-5);
        let lse = x.logsumexp(1, false).unwrap().to_vec::<f32>().unwrap();
        assert!((lse[0] - tot.ln()).abs() < 1e-5 && (lse[1] - (1000.0 + 3f32.ln())).abs() < 1e-3);
        // Softmax along dim 0 (inner > 1 path) equals the transposed last-dim softmax.
        let a = x.softmax(0).unwrap().to_vec::<f32>().unwrap();
        let b = x.t().unwrap().softmax(1).unwrap().t().unwrap().to_vec::<f32>().unwrap();
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn topk_sorted_with_ties() {
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 5.0, 2.0, 0.0, -1.0, 4.0], [2, 4]).unwrap();
        let (v, i) = x.topk(2, true).unwrap();
        assert_eq!(v.to_vec2::<f32>().unwrap(), vec![vec![5.0, 5.0], vec![4.0, 2.0]]);
        assert_eq!(i.to_vec2::<i64>().unwrap(), vec![vec![1, 3], vec![3, 0]]);
        let (v, i) = x.topk(3, false).unwrap();
        assert_eq!(v.to_vec2::<f32>().unwrap()[1], vec![-1.0, 0.0, 2.0]);
        assert_eq!(i.to_vec2::<i64>().unwrap()[0], vec![0, 2, 1]);
        assert!(x.topk(5, true).is_err());
    }

    #[test]
    fn topk_signed_zeros_tie_by_index() {
        // -0.0 == 0.0, so the lower index wins, as in argmax / argmin.
        let x = Tensor::from_slice(&[-0.0f32, 0.0, 0.0, -0.0], [4]).unwrap();
        assert_eq!(x.topk(1, true).unwrap().1.to_vec::<i64>().unwrap(), vec![0]);
        assert_eq!(x.topk(1, false).unwrap().1.to_vec::<i64>().unwrap(), vec![0]);
        assert_eq!(x.topk(4, true).unwrap().1.to_vec::<i64>().unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(x.argmax(0, false).unwrap().item::<i64>().unwrap(), 0);
    }
}
