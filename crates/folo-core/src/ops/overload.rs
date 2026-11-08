//! `+ - * /` and unary `-` on `Tensor`, `&Tensor` and `f32`. Every operator returns `Result<Tensor>`, so
//! chains read `((&a + &b)? * 2.0)?`.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Result, Tensor};

macro_rules! bin_ops {
    ($trait:ident, $method:ident, $tensor_fn:ident, $scalar_fn:ident, $rscalar:expr) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: &Tensor) -> Result<Tensor> {
                self.$tensor_fn(rhs)
            }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: Tensor) -> Result<Tensor> {
                self.$tensor_fn(&rhs)
            }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: &Tensor) -> Result<Tensor> {
                (&self).$tensor_fn(rhs)
            }
        }
        impl $trait<Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: Tensor) -> Result<Tensor> {
                (&self).$tensor_fn(&rhs)
            }
        }
        impl $trait<f32> for &Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: f32) -> Result<Tensor> {
                self.$scalar_fn(rhs)
            }
        }
        impl $trait<f32> for Tensor {
            type Output = Result<Tensor>;
            fn $method(self, rhs: f32) -> Result<Tensor> {
                (&self).$scalar_fn(rhs)
            }
        }
        impl $trait<&Tensor> for f32 {
            type Output = Result<Tensor>;
            fn $method(self, rhs: &Tensor) -> Result<Tensor> {
                let f: fn(f32, &Tensor) -> Result<Tensor> = $rscalar;
                f(self, rhs)
            }
        }
        impl $trait<Tensor> for f32 {
            type Output = Result<Tensor>;
            fn $method(self, rhs: Tensor) -> Result<Tensor> {
                let f: fn(f32, &Tensor) -> Result<Tensor> = $rscalar;
                f(self, &rhs)
            }
        }
    };
}

bin_ops!(Add, add, add, add_scalar, |c, t| t.add_scalar(c));
bin_ops!(Sub, sub, sub, sub_scalar, |c, t| t.rsub_scalar(c));
bin_ops!(Mul, mul, mul, mul_scalar, |c, t| t.mul_scalar(c));
bin_ops!(Div, div, div, div_scalar, |c, t| t.rdiv_scalar(c));

impl Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> {
        Tensor::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Result, Tensor};

    #[test]
    fn operators() -> Result<()> {
        let a = Tensor::from_slice(&[1.0f32, 2.0], [2])?;
        let b = Tensor::from_slice(&[3.0f32, 4.0], [2])?;
        assert_eq!((&a + &b)?.to_vec::<f32>()?, vec![4.0, 6.0]);
        assert_eq!((&b - &a)?.to_vec::<f32>()?, vec![2.0, 2.0]);
        assert_eq!((a.clone() * b.clone())?.to_vec::<f32>()?, vec![3.0, 8.0]);
        assert_eq!((&b / 2.0)?.to_vec::<f32>()?, vec![1.5, 2.0]);
        assert_eq!((1.0 - &a)?.to_vec::<f32>()?, vec![0.0, -1.0]);
        assert_eq!((8.0 / &b)?.to_vec::<f32>()?, vec![8.0 / 3.0, 2.0]);
        assert_eq!((2.0 * a.clone())?.to_vec::<f32>()?, vec![2.0, 4.0]);
        assert_eq!((-&a)?.to_vec::<f32>()?, vec![-1.0, -2.0]);
        assert_eq!(((&a + 1.0)? * &b)?.to_vec::<f32>()?, vec![6.0, 12.0]);
        assert!((&a + &Tensor::zeros([3])).is_err());
        Ok(())
    }
}
