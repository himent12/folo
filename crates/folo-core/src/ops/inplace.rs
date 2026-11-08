//! In-place updates for optimizers and buffers.
//!
//! These write into the tensor's storage, so every view and clone sharing it sees the change. Rules that keep
//! autograd sound:
//! - They refuse to run on a tensor that requires grad while grad mode is on: wrap parameter updates in
//!   [`crate::no_grad`].
//! - Each write bumps the storage version. A graph recorded before the write that saved this tensor fails
//!   `backward()` with [`Error::ModifiedInPlace`] instead of silently using the new values.
//! - The destination must not be a broadcast view.

use crate::autograd::is_grad_enabled;
use crate::backend::{BinaryOp, View};
use crate::cpu::elementwise::fill_rows_inplace;
use crate::{Error, Result, Tensor};

impl Tensor {
    fn check_inplace(&self, op: &'static str) -> Result<()> {
        if is_grad_enabled() && self.requires_grad() {
            return Err(Error::Autograd(format!(
                "{op}: in-place update of a tensor that requires grad; run it under no_grad()"
            )));
        }
        if self.layout().has_broadcast() {
            return Err(Error::invalid_shape(op, self.shape(), "cannot write into a broadcast view"));
        }
        Ok(())
    }

    /// Overwrite the elements with `src` (broadcast to this shape; same dtype and device).
    pub fn copy_from(&self, src: &Tensor) -> Result<()> {
        self.check_inplace("copy_from")?;
        self.check_same_device(src, "copy_from")?;
        if src.dtype() != self.dtype() {
            return Err(Error::DTypeMismatch { op: "copy_from", expected: self.dtype(), got: src.dtype() });
        }
        let src = if self.same_storage(src) { src.copy_contiguous()? } else { src.clone() };
        let ls = src.layout().broadcast_to(self.shape(), "copy_from")?;
        let b = self.backend()?;
        let sr = src.cell().read();
        let mut dst = self.cell().write();
        b.copy_into(&mut dst, self.layout(), View::new(&sr, &ls))
    }

    /// Set every element to `value`.
    pub fn fill_assign(&self, value: f32) -> Result<()> {
        let v = Tensor::full_with(crate::Shape::scalar(), value as f64, self.dtype(), self.device())?;
        self.copy_from(&v)
    }

    fn binary_assign(&self, op: BinaryOp, other: &Tensor, name: &'static str) -> Result<()> {
        self.check_inplace(name)?;
        self.check_f32(name)?;
        other.check_f32(name)?;
        self.check_same_device(other, name)?;
        let other = if self.same_storage(other) { other.copy_contiguous()? } else { other.clone() };
        let lo = other.layout().broadcast_to(self.shape(), name)?;
        if self.device().is_cpu() {
            let sr = other.cell().read();
            let src = sr.as_cpu(name)?.as_f32(name)?;
            let mut dst = self.cell().write();
            let d = dst.as_cpu_mut(name)?.as_f32_mut(name)?;
            let f: fn(f32, f32) -> f32 = match op {
                BinaryOp::Add => |a, b| a + b,
                BinaryOp::Sub => |a, b| a - b,
                BinaryOp::Mul => |a, b| a * b,
                BinaryOp::Div => |a, b| a / b,
                _ => unreachable!(),
            };
            fill_rows_inplace(d, self.layout(), src, &lo, f);
            return Ok(());
        }
        let _g = crate::no_grad();
        let r = self.binary(op, &other)?;
        self.copy_from(&r)
    }

    /// `self += other` (broadcast).
    pub fn add_assign(&self, other: &Tensor) -> Result<()> {
        self.binary_assign(BinaryOp::Add, other, "add_assign")
    }

    /// `self -= other` (broadcast).
    pub fn sub_assign(&self, other: &Tensor) -> Result<()> {
        self.binary_assign(BinaryOp::Sub, other, "sub_assign")
    }

    /// `self *= other` (broadcast).
    pub fn mul_assign(&self, other: &Tensor) -> Result<()> {
        self.binary_assign(BinaryOp::Mul, other, "mul_assign")
    }

    /// `self /= other` (broadcast).
    pub fn div_assign(&self, other: &Tensor) -> Result<()> {
        self.binary_assign(BinaryOp::Div, other, "div_assign")
    }

    /// `self += alpha * other` (broadcast), the SGD step: `p.add_scaled_assign(&grad, -lr)`.
    pub fn add_scaled_assign(&self, other: &Tensor, alpha: f32) -> Result<()> {
        if self.device().is_cpu() && !self.same_storage(other) {
            self.check_inplace("add_scaled_assign")?;
            self.check_f32("add_scaled_assign")?;
            other.check_f32("add_scaled_assign")?;
            self.check_same_device(other, "add_scaled_assign")?;
            let lo = other.layout().broadcast_to(self.shape(), "add_scaled_assign")?;
            let sr = other.cell().read();
            let src = sr.as_cpu("add_scaled_assign")?.as_f32("add_scaled_assign")?;
            let mut dst = self.cell().write();
            let d = dst.as_cpu_mut("add_scaled_assign")?.as_f32_mut("add_scaled_assign")?;
            fill_rows_inplace(d, self.layout(), src, &lo, move |a, b| a + alpha * b);
            return Ok(());
        }
        let scaled = {
            let _g = crate::no_grad();
            other.mul_scalar(alpha)?
        };
        self.add_assign(&scaled)
    }

    /// `self *= c`.
    pub fn mul_scalar_assign(&self, c: f32) -> Result<()> {
        let s = Tensor::full_with(crate::Shape::scalar(), c as f64, self.dtype(), self.device())?;
        self.mul_assign(&s)
    }

    /// Run `f` on the mutable `f32` elements (CPU, `F32`, contiguous). For custom optimizer kernels:
    ///
    /// ```
    /// use folo_core::{no_grad, Tensor};
    /// let p = Tensor::ones([4]).requires_grad_();
    /// let g = Tensor::full([4], 0.5);
    /// let _guard = no_grad();
    /// g.with_f32_slice(|g| p.update_f32(|p| p.iter_mut().zip(g).for_each(|(p, g)| *p -= 0.1 * g)))
    ///     .unwrap()
    ///     .unwrap();
    /// assert_eq!(p.to_vec::<f32>().unwrap(), vec![0.95; 4]);
    /// ```
    ///
    /// Do not read a tensor that shares this tensor's storage inside `f` (the storage is write-locked).
    pub fn update_f32<R>(&self, f: impl FnOnce(&mut [f32]) -> R) -> Result<R> {
        self.check_inplace("update_f32")?;
        let range = self
            .layout()
            .contiguous_range()
            .ok_or_else(|| Error::invalid_shape("update_f32", self.shape(), "tensor must be contiguous"))?;
        let mut dst = self.cell().write();
        let d = dst.as_cpu_mut("update_f32")?.as_f32_mut("update_f32")?;
        Ok(f(&mut d[range]))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Error, Tensor, no_grad};

    #[test]
    fn inplace_updates() {
        let p = Tensor::ones([2, 3]).requires_grad_();
        let g = Tensor::full([3], 0.5);
        assert!(p.sub_assign(&g).is_err(), "must refuse outside no_grad");
        {
            let _ng = no_grad();
            p.sub_assign(&g).unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), vec![0.5; 6]);
            p.add_scaled_assign(&Tensor::ones([2, 3]), -0.25).unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), vec![0.25; 6]);
            p.mul_scalar_assign(4.0).unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), vec![1.0; 6]);
            // Write through a strided view.
            let col = p.narrow(1, 1, 1).unwrap();
            col.fill_assign(7.0).unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), vec![1., 7., 1., 1., 7., 1.]);
            // Self-aliasing source.
            p.add_assign(&p.t().unwrap().t().unwrap()).unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), vec![2., 14., 2., 2., 14., 2.]);
            p.copy_from(&Tensor::zeros([2, 3])).unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), vec![0.; 6]);
            assert!(Tensor::ones([3]).broadcast_to([2, 3]).unwrap().fill_assign(1.0).is_err());
        }
    }

    #[test]
    fn modified_inputs_are_detected() {
        let w = Tensor::ones([3]).requires_grad_();
        let y = w.mul(&w).unwrap().sum_all().unwrap();
        {
            let _ng = no_grad();
            w.sub_assign(&Tensor::ones([3])).unwrap();
        }
        match y.backward() {
            Err(Error::ModifiedInPlace { op, .. }) => assert_eq!(op, "mul"),
            other => panic!("expected ModifiedInPlace, got {other:?}"),
        }
        // A fresh forward pass after the update works.
        let y = w.mul(&w).unwrap().sum_all().unwrap();
        assert!(y.backward().is_ok());
    }
}
